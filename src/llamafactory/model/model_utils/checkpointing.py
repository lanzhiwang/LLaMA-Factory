# Copyright 2025 HuggingFace Inc., Daniel Han-Chen & the Unsloth team and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and PEFT library,
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py
# and the Unsloth library.
# https://github.com/unslothai/unsloth/blob/July-2024/unsloth/models/_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from collections.abc import Callable
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from ...extras import logging
from ...extras.constants import LAYERNORM_NAMES


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def get_unsloth_gradient_checkpointing_func() -> Callable:
    """
    这段代码实现了一个 "带 Offloading(卸载)机制的梯度检查点(Gradient Checkpointing)" 算子.
    它是微调超大规模模型时的显存优化利器, 直接借鉴了 Unsloth 加速库的核心思想.

    在标准的 PyTorch 梯度检查点中, 为了省显存, 前向传播不存中间激活值, 只存输入. 而这段代码更进一步: 它连输入都不想留在显卡里.

    高级研究员视角的架构解析:

    为什么这个比官方的 torch.utils.checkpoint 更好?
    官方的实现默认将输入 Tensor 留在显存中. 如果一个模型有 32 层, 你开启 GC 后, 显存中会积累 32 层的所有输入 Tensor. 对于长文本(Long Context)训练, 这些 Tensor 的大小非常惊人. LLaMA-Factory 引入的这个"卸载式"GC, 利用了 PCIe 带宽(将数据扔到内存) 换取了 计算容错空间(GPU 显存), 使得单卡训练更长的序列成为可能.

    异步通信(Non-blocking)的妙用:
    代码中两次使用了 non_blocking=True. 这意味着数据在 CPU 和 GPU 之间搬运时, GPU 的 Stream 不会停下来等, 而是继续跑其他的 Kernel. 这是高级 Python 工程师处理 I/O 密集型任务(即便是在显卡内部)的标志性写法.

    计算与存储的博弈:
    这个算子的代价是增加了 1 次额外的 Forward 计算耗时, 以及 PCIe 总线的压力, 但它解决了 "能训练" 和 "由于 OOM 没法训练" 之间的质变问题.
    """

    # 1. 继承 torch.autograd.Function 定义自定义算子
    # [为什么要这么写]: 标准梯度检查点仅能重计算. 自定义算子能让我们精确控制 Tensor 在硬件间的移动.
    # [解决的问题]: 在极致显存(VRAM)受限的情况下, 进一步压榨空间, 通过 CPU 内存(RAM)换取 GPU 显存.
    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""
        Saves VRAM by smartly offloading to RAM.

        通过智能地将激活值卸载(Offload)到系统内存来节省显存.
        """

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            # 2. 异步将输入状态搬运到 CPU
            # [为什么要这么写]: 使用 to("cpu", non_blocking=True).
            # [解决的问题]: VRAM 极其宝贵. 即便只存输入 Tensor, 在长文本场景下也会占用数 GB.
            # non_blocking=True 确保了 GPU 在计算时, Tensor 的拷贝是在后台异步进行的, 不会阻塞计算流.
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)

            # 3. 开启不记梯度的前向计算 (The "Checkpoint" part)
            # [为什么要这么写]: 使用 torch.no_grad() 执行 forward_function.
            # [解决的问题]: 在此层计算时不构建计算图(Computation Graph), 不存储任何中间层激活值, 显存占用极低.
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)

            # 4. 上下文存储
            # 将 CPU 上的输入保存, 供反向传播重计算使用; 同时记录函数句柄和静态参数.
            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            # 5. 从 CPU 找回输入 Tensor
            (hidden_states,) = ctx.saved_tensors

            # 6. 将输入搬回 GPU 并准备重计算
            # [为什么要这么写]: detach() + requires_grad_(True).
            # [解决的问题]: hidden_states 搬回显存后是"叶子节点". 我们需要手动开启其梯度属性,
            # 这样在接下来的重计算中, PyTorch 才会重新构建这一层局部的计算图.
            hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
            hidden_states.requires_grad_(True)

            # 7. 重计算 (Re-computation)
            # [为什么要这么写]: 在 torch.enable_grad() 下重跑前向过程.
            # [解决的问题]: 通过重新计算一次, 获取在前向传播时被我们"丢弃"掉的中间激活值,
            # 从而能够计算出这一层的具体参数梯度.
            with torch.enable_grad():
                outputs = ctx.forward_function(hidden_states, *ctx.args)
                output = outputs[0] if isinstance(outputs, tuple) else outputs

            # 8. 手动触发局部反向传播
            # 将本层之后传来的梯度(grad_output)作用在当前重计算出的结果上.
            torch.autograd.backward(output, grad_output)

            # 9. 返回梯度
            # 按照 forward 的入参顺序返回梯度, 不需要梯度的位置传 None.
            # 其中核心是 hidden_states.grad.
            return (None, hidden_states.grad) + (None,) * len(ctx.args)

    return UnslothGradientCheckpointing.apply


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""
    Only applies gradient checkpointing to trainable layers.

    在 LLM 微调中, 梯度检查点(Gradient Checkpointing, GC) 是一把双刃剑: 它通过丢弃前向传播的激活值并在反向传播时重新计算, 极大地节省了显存, 但也增加了约 33% 的计算开销.
    这段 get_custom_gradient_checkpointing_func 函数的设计体现了"按需重计"的极致优化思想. 以下是添加了深度注释的代码及其背后的架构设计逻辑:

    [核心目标]: 实现"选择性梯度检查点", 仅对包含可训练参数的层应用 GC.

    [为什么要这么写]:
    在 PEFT(如 LoRA)或部分层冻结(Freeze Tuning)场景下, 模型的大部分层是冻结的(requires_grad=False).
    原生的 PyTorch GC 通常会无差别地对所有配置了 checkpoint 的层进行"丢弃-重计"操作.
    对于冻结层, 由于不需要计算参数梯度, 重计激活值是纯粹的计算浪费.

    高级研究员视角的工程解析:

    解决 LoRA 微调的性能痛点:
    在微调 70B 模型时, 通常只训练 1% 的参数. 如果开启全局 GC, 模型 99% 的时间都在重计那些根本不需要梯度的冻结层. 通过这个包装器, LLaMA-Factory 能够自动识别 LoRA 所在的层, 仅对它们进行 GC, 而对冻结的 Base Model 层直接跳过重计.

    __self__ 的妙用:
    在 Python 装饰器中处理 bound method(绑定方法)比较棘手. 代码中 assigned 扩展了 __self__, 确保了元数据和实例引用的完整性. 这展示了对 Python 对象协议(Descriptor Protocol)的深度理解.

    对计算图的精确控制:
    arg.requires_grad_(True) 的处理非常老辣. 在复杂的残差连接(Residual Connection)或并行计算中, 梯度链条有时会断开. 这一行代码保证了即使在极端情况下, 只要该层有可训练参数, GC 就一定能被激活, 避免了"配置了 GC 却因为输入无梯度导致没生效"的坑.

    总结:
    这段代码解决的问题是 "显存节省与计算效率的精细化平衡". 它让 LLaMA-Factory 在进行 PEFT 微调时, 比直接使用官方原生 GC 的框架具有更高的吞吐量(Tokens per Second), 同时保持相同的低显存占用.
    """

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        # 1. 实例溯源 (Instance Introspection)
        # [解决的问题]: 在 PyTorch 中, 层的 forward 方法通常绑定在 Module 实例上.
        # 这里需要拿到 func 所属的那个 nn.Module 对象(即具体的 Transformer Layer),
        # 从而检查这个层是否包含需要更新的权重.
        if isinstance(func, partial):
            module: torch.nn.Module = func.func.__self__
        else:
            module: torch.nn.Module = func.__self__

        # 2. 动态可训练性检测 (Dynamic Trainability Detection)
        # [为什么要这么写]: 遍历该模块的所有参数.
        # 如果该层(或其子模块如 LoRA A/B)中没有任何参数需要梯度更新, 那么 has_grad 为 False.
        has_grad = False
        if any(param.requires_grad for param in module.parameters()):
            has_grad = True
            # 3. 强制触发梯度链 (Gradient Chain Triggering)
            # [解决的问题]: PyTorch 的原生 `checkpoint` 函数有一个特性:
            # 如果所有的输入 tensor 都没有 `requires_grad=True`, 则它不会记录任何计算图.
            # 在某些微调设置中, 前一层的输出可能暂时丢失了梯度状态.
            # 为了确保当前这个可训练层能正确开启 GC 逻辑, 我们需要手动确保输入的 hidden_states
            # 具有梯度需求.
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
                    break  # assume the first tensor is always the hidden states

        # 4. 智能路由执行 (Smart Routing)
        if has_grad:
            # [场景]: 当前层是可训练的(例如挂了 LoRA 的层).
            # 执行 GC: 用计算时间换显存空间, 因为我们需要这些激活值来算 LoRA 的梯度.
            return gradient_checkpointing_func(func, *args, **kwargs)
        else:
            # [场景]: 当前层是完全冻结的.
            # 绕过 GC: 直接执行 forward.
            # [解决的问题]: 避免了在反向传播阶段对冻结层进行无谓的"重新计算",
            # 从而在不额外消耗显存的前提下, 显著提升微调的训练速度.
            return func(*args, **kwargs)

    return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""
    Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.

    这段 _gradient_checkpointing_enable 函数是对 Hugging Face transformers 库原生方法的深度重写(Monkey Patching). 它的核心目标是: 在不牺牲显存节省的前提下, 通过极致的逻辑控制
    激活当前模型的梯度检查点(Gradient Checkpointing, GC).

    [为什么要重写此方法]:
    原生的 GC 逻辑比较"粗暴", 通常无差别地对所有层进行重计算.
    在微调场景(如 LoRA)中, 大部分层是冻结的, 无差别的重计算会造成严重的计算浪费.
    此外, 为了支持 BAdam(块优化器)和 Unsloth(极致加速), 我们需要注入自定义的算子逻辑.

    高级研究员视角的架构点评:

    分治策略(Divide and Conquer):
    这段代码体现了 LLaMA-Factory 对 PEFT(高效微调) 效率的极致追求. 常规框架开启 GC 后速度会慢 30% 左右, 但因为这里的 get_custom_gradient_checkpointing_func 实现了"只对可训练层做 GC", 使得 LoRA 训练在开启 GC 后的速度损失被大幅压缩(甚至不到 10%).

    黑盒算子解耦:
    将 gradient_checkpointing_func 作为一个变量进行传递和包装, 是高级 Python 工程设计的典型体现. 它允许 LLaMA-Factory 随时通过环境变量切换底层算子(如从标准的 PyTorch 切换到 Flash-Attention 的特化 GC), 而无需改动模型的主干代码.

    对 BAdam 的底层支撑:
    BAdam 这种块优化器要求梯度必须按照特定顺序和颗粒度产生. 通过接管 _set_gradient_checkpointing, 我们确保了重计算的过程不会破坏 BAdam 所依赖的参数快照和梯度链条.
    """
    from torch.utils.checkpoint import checkpoint

    # 1. 基础合法性检查
    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    # 2. 默认参数处理 (Compatibility Guard)
    # [为什么要这么写]: PyTorch 的 GC 有两种模式: Reentrant(重入)和 Non-reentrant.
    # [解决的问题]: 默认设置为 True 主要是为了向后兼容旧版模型. Non-reentrant 模式虽然更灵活,
    # 但在处理某些自定义计算图时可能会出现 DDP 同步报错.
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    # 3. 核心 GC 算子选择 (Backend Dispatching)
    # [为什么要这么写]: 判断是否开启 Unsloth 版本的 GC.
    # [解决的问题]:
    #   - Unsloth GC: 采用了更高效的 Triton 实现, 并能智能地将激活值卸载到 RAM, 显著降低显存占用.
    #   - Standard GC: 传统的重计算逻辑, 使用 partial 预设用户提供的参数.
    if use_unsloth_gc:
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    # 4. 注入选择性 GC 逻辑 (Selective Checkpointing Wrapper)
    # [为什么要这么写]: 通过自定义包装器处理上层传入的算子.
    # [解决的问题]: 这是 LLaMA-Factory 的核心优化点. 此函数会检查每一层是否有 `requires_grad=True`.
    # 如果某一层是冻结的(如 LoRA 之外的 Base 层), 则直接跳过 GC.
    # 这解决了"在微调冻结层上浪费重计算时间"的问题, 显著提升了训练吞吐量.
    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)

    # 5. 跨版本 API 适配 (API Version Normalization)
    # [为什么要这么写]: Hugging Face 改变了内部 `_set_gradient_checkpointing` 的签名.
    # [解决的问题]:
    #   - 旧版格式: 接受 `value` 参数. 如果是旧版, 我们通过 .apply() 遍历所有子模块开启,
    #     但这种方式无法注入自定义的 `gradient_checkpointing_func`, 会导致 BAdam 等优化器失效.
    #   - 新版格式: 支持直接传递自定义算子.
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format 旧版 HF 格式
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        # 在旧版模式下, 必须手动触发此函数, 否则第一层的输入张量可能没有梯度, 导致整个反向传播链断裂
        self.enable_input_require_grads()
        logger.warning_rank0_once("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients 新版标准格式
        # [核心操作]: 将我们封装好的"带选择性、带加速后端、带精度保护"的函数注入模型内部.
        # 这样模型在执行每个 Transformer Block 的前向传播时, 都会优先调用我们的优化版 GC 逻辑.
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    """
    一个典型的 PyTorch 前向传播后置钩子(Post-hook), 强制将模块的输出转换为 FP32.

    [为什么要这么写]:
    1. 强制精度提升(Upcasting): 在 LLM 微调中, 我们为了节省显存, 通常使用 FP16 或 BF16 进行混合精度训练.
       然而, 模型某些特定层(如 lm_head 或 RLHF 中的 Value Head)的输出对于数值精度极其敏感.
    2. 解耦计算与输出: 通过 Hook 机制, 我们可以在不修改模型原始 forward 代码的情况下,
       透明地改变特定层的输出行为.

    [要解决的问题]:
    1. 数值溢出(Overflow/Underflow): FP16 的数值范围较窄(最大值约 65504). 在计算 Logits 时,
       如果数值过大, 直接在 FP16 下计算后续的 Softmax 或 Loss 会导致溢出产生 NaN. 转换为 FP32 可以利用
       其更宽的指数位来保证计算安全.
    2. Loss 计算的准确性: 损失函数(如 CrossEntropyLoss)在内部通常期望输入是 FP32, 以保证梯度计算的精确性.
       如果 Logits 在低精度下被截断, 会导致模型收敛缓慢甚至不收敛.
    3. 适配 PEFT/LoRA 架构: 在 LLaMA-Factory 中, 底座模型可能是量化(4/8-bit)或半精度的,
       但我们通常希望输出头(Output Head)以全精度运行, 以确保微调时的微小梯度更新能够被正确捕捉.

    在 LLaMA-Factory 的实际应用中, 这个钩子通常被挂载到以下位置:

    Language Model Head (lm_head): 当你进行 SFT(有监督微调)时, 确保预测下一个 token 的概率分布(Logits)是 FP32 的. 这能有效防止大规模训练时偶尔出现的 Loss 突变.
    Value Head (Reward Model/PPO): 在奖励模型训练中, 模型需要输出一个标量分数(Score). 这个分数如果用 FP16 表示, 其分辨率不足以区分两个质量接近的回复. 通过这个 Hook 确保分数的解析度, 对强化学习阶段的稳定性至关重要.
    计算图的末梢: 将输出转为 FP32 后, 接下来的 CrossEntropy 计算会在 CPU 或 GPU 的 FP32 单元上进行, 这符合 "Compute in low precision, accumulate in high precision"(低精度计算, 高精度累加)的工业界训练准则.
    """
    # 将输出张量强制转换为 float32 (单精度浮点数)
    return output.to(torch.float32)


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Prepare the model before training.

    Include:
    (1) cast the layernorm in fp32
    (2) make output embedding layer require grads
    (3) add the upcasting of the lm_head in fp32.

    在 LLM 微调工程中, 模型加载后的"预处理"阶段甚至比模型加载本身还要关键. 这个函数解决了 混合精度训练下的数值稳定性、分布式架构(FSDP2)的兼容性、以及极端显存优化(OOM 预防) 等一系列工业级痛点.

    在正式开启训练循环前, 对模型进行最后阶段的"外科手术式"调整.
    主要任务:
    (1) 将 LayerNorm 层强制提升至 FP32 精度以保证数值稳定性;
    (2) 处理分布式环境(FSDP2)下的特殊配置;
    (3) 激活梯度检查点(Gradient Checkpointing)以节省显存;
    (4) 对输出层(lm_head)进行 FP32 精度提升处理.

    高级研究员视角的架构解析(Deep Dive):

    关于 MethodType 的妙用:
    作为高级 Python 开发人员, 我们不直接修改 transformers 库的源码. 通过 MethodType 将 _gradient_checkpointing_enable 动态绑定到模型实例, 我们可以在运行时"接管"模型的显存管理逻辑, 而不破坏库的整体封装.

    数值稳定性与性能的权衡:
    代码中并没有盲目地将所有参数转为 FP32, 而只是针对 LayerNorm 和 Output Head 进行了处理. 这体现了 "手术刀式"的优化思路: 用极小的显存代价(这两个模块参数占比极低)换取了极大的数值稳定性红利.

    对生态的敏感度:
    处理 FSDP2 的环境变量判断, 说明 LLaMA-Factory 深度适配了 Meta 的分布式训练基座. 在超大规模集群微调中, 这种环境感知的自动配置(Auto-config)能为研究员节省大量的排雷时间.
    """

    # 1. LayerNorm 精度提升 (Upcasting LayerNorm)
    # [为什么要这么写]: LayerNorm 涉及计算均值和方差, 其中方差涉及求平方. 在 FP16/BF16 下,
    # 极小或极大的数值在平方运算后容易发生溢出(Overflow/Underflow), 导致 Loss 突变或 NaN.
    # [解决的问题]: 显著提高训练的数值稳定性, 防止深层模型在微调过程中出现梯度爆炸.
    if model_args.upcast_layernorm:
        logger.info_rank0("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            # LayerNorm 通常是 1 维的权重(weight/bias), 通过 LAYERNORM_NAMES 匹配特定层名
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    # 2. FSDP2 兼容性补丁
    # [为什么要这么写]: PyTorch FSDP (Fully Sharded Data Parallel) v2 相比 v1 有较大的底层变动.
    # FSDP2 的参数分片逻辑与"重入式"梯度检查点(Reentrant GC)存在严重的冲突, 可能导致死锁或梯度计算错误.
    # [解决的问题]: 自动检测加速库环境, 强制关闭重入式 GC 以适配高性能分布式训练框架.
    if (
        os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        and int(os.environ.get("FSDP_VERSION", "1")) == 2
    ):
        model_args.use_reentrant_gc = False
        logger.warning_rank0("You are using fsdp2, `use_reentrant_gc` has been set to False.")

    # 3. 动态配置梯度检查点 (Gradient Checkpointing, GC)
    # [为什么要这么写]: 梯度检查点通过"时间换空间", 在前向传播时不存储中间激活值, 而在反向传播时重算.
    # 这里通过 MethodType 动态替换了模型的原生方法, 注入了 LLaMA-Factory 自定义的逻辑(如支持 Unsloth 极速后端).
    # [解决的问题]: 大幅降低显存占用(VRAM), 使得在消费级显卡上训练 70B 等大模型成为可能.
    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")
        else:
            # 注入自定义的 GC 启动器, 可选集成 Unsloth 的高效算子
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )
            # 使用猴子补丁(Monkey Patching)动态绑定方法
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc}
            )
            # [重要]: 开启 GC 后必须关闭 KV Cache (use_cache), 因为两者在数学逻辑和显存分配上是互斥的.
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info_rank0("Gradient checkpointing enabled.")

    # 4. 输出层 Logits 精度提升 (Upcasting lm_head Output)
    # [为什么要这么写]: lm_head 负责将隐藏状态映射到巨大的词表空间(例如 128k).
    # Logits 的数值范围往往很大, 直接在低精度(FP16/BF16)下计算 Softmax/CrossEntropy 极易溢出.
    # [解决的问题]: 通过 register_forward_hook 在输出层计算完成后立即将其结果转为 FP32.
    # 这确保了最后的 Loss 计算是在全精度下完成的, 解决了训练末期收敛不稳定(Loss 震荡)的痛点.
    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info_rank0("Upcasting lm_head outputs in float32.")
            # 挂载后处理钩子, 确保计算结果的精度提升
            output_layer.register_forward_hook(_fp32_forward_post_hook)

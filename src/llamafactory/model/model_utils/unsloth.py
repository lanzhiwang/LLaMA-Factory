# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _get_unsloth_kwargs(
    config: "PretrainedConfig",
    model_name_or_path: str,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> dict[str, Any]:
    """
    _get_unsloth_kwargs 函数的主要目的是为 Unsloth 库的 FastLanguageModel.from_pretrained 方法准备初始化参数.
    Unsloth 是目前 LLM 微调领域中极具代表性的性能加速库(通常能提升 2 倍速度并节省 70% 显存).

    这段代码之所以这样设计, 是为了将 LLaMA-Factory 的通用配置平滑地映射到 Unsloth 的特化接口上, 同时解决内存碎片、计算精度和长文本适配等工程问题.

    构造 Unsloth 专用的初始化参数字典.

    [为什么要写这个函数]:
    Unsloth 并不完全遵循标准的 Hugging Face 加载流程, 它在底层重写了大量的 Triton 内核.
    通过这个转换函数, 我们确保了用户在 LLaMA-Factory 界面配置的参数能够正确驱动 Unsloth 的高性能后端.

    高级研究员视角下的架构点评:

    工程解耦:
    该函数体现了适配器模式 (Adapter Pattern) 的思想. LLaMA-Factory 的主干逻辑不需要理解 Unsloth 内部复杂的 API, 只需要调用这个函数生成一份"配置合同"即可. 这使得项目在未来升级 Unsloth 版本时, 只需要在这里调整字段名.

    显存性能优先:
    use_gradient_checkpointing: "unsloth" 是最关键的设置. 它不仅仅是开关, 更是一种算子劫持 (Operator Hijacking). 通过这种写法, 模型在执行 backward 过程中的中间激活值管理变得极其高效.

    对 Qwen/Llama 等架构的敏感性:
    max_seq_length 的传递非常直接. 作为高级工程师, 我们要预见到 Unsloth 会根据这个长度自动调整 RoPE 频率基数(Base), 这解决了长文本外推时模型"变笨"的数学痛点.
    """

    return {
        "model_name": model_name_or_path,

        # 1. 序列长度预分配 (Context Window Management)
        # [为什么要这么写]: Unsloth 会根据 max_seq_length 预先分配内存缓冲区并处理 RoPE 缩放.
        # [解决的问题]: 如果用户没指定, 默认 4096 是一条安全线. 它解决了因序列过长导致的 OOM(显存溢出)
        # 以及因序列过短导致模型在长上下文下计算不正确的问题.
        "max_seq_length": model_args.model_max_length or 4096,

        # 2. 精度自动对齐 (Dtype Optimization)
        # [为什么要这么写]: 直接透传计算精度(如 bf16/fp16/None).
        # [解决的问题]: 确保数值稳定性. 在 A100/H100 上使用 bf16 可以显著提升性能,
        # 而在旧卡上自动切回 fp16, 防止硬件不支持导致的训练崩溃.
        "dtype": model_args.compute_dtype,

        # 3. 4-bit 量化加载 (Memory Footprint Reduction)
        # [解决的问题]: 这是 QLoRA 的核心. 当设为 True 时, Unsloth 使用其特有的 4-bit 算子加载,
        # 极大地压缩了显存占用, 使得在 24GB 显存的显卡上微调 70B 模型成为可能.
        "load_in_4bit": model_args.quantization_bit == 4,

        "token": model_args.hf_hub_token,

        # 4. 微调模式区分 (Algorithm Dispatching)
        # [解决的问题]: Unsloth 对全量微调(Full)和参数高效微调(LoRA)有不同的算子优化路径.
        # 这里显式告知后端, 以启用对应的计算内核.
        "full_finetuning": finetuning_args.finetuning_type == "full",

        # 5. 设备锁定 (Device Locality)
        # [为什么要这么写]: 通过 {"": get_current_device()} 强制将模型加载到当前激活的 GPU 索引上.
        # [解决的问题]: 防止在多 GPU 环境下, Unsloth 默认的 "auto" 逻辑将模型分片到多个设备,
        # 从而避免跨卡通信(NVLink/PCIe)带来的延迟损耗.
        "device_map": {"": get_current_device()},

        # 6. 位置编码外推同步 (RoPE Scaling Carry-over)
        # [解决的问题]: 确保模型加载时继承了 config 中设置的 RoPE 插值参数(如 Linear/YaRN 缩放).
        # 解决了在长文本微调时, 模型位置感官不一致导致的输出乱码问题.
        "rope_scaling": getattr(config, "rope_scaling", None),

        # 7. 分词器修复隔离 (Tokenizer Consistency)
        # [为什么要这么写]: 设为 False.
        # [解决的问题]: LLaMA-Factory 内部有一套非常成熟的 `patch_tokenizer` 逻辑.
        # 禁用 Unsloth 的自动修复功能, 可以防止两个框架对分词器的双重修改, 避免特殊 Token 冲突.
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,

        # 8. 梯度检查点黑魔法 (Optimized Gradient Checkpointing)
        # [核心黑科技]: 设为 "unsloth".
        # [解决的问题]: 标准的 PyTorch 梯度检查点(GC)虽然省显存但很慢.
        # 设为 "unsloth" 会启用其自定义的 GC 算子, 在节省显存的同时, 消除了标准 GC 约 30% 的重计算时间损耗.
        "use_gradient_checkpointing": "unsloth",
    }


def load_unsloth_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["PreTrainedModel"]:
    r"""
    Optionally load pretrained model with unsloth. Used in training.

    Unsloth 是当前 LLM 微调领域的"黑马", 通过手写的 Triton 内核和模板优化, 能将微调速度提升 2 倍以上, 并将显存占用降低 70%.
    这段代码的设计核心在于可选依赖的优雅降级(Graceful Fallback)以及第三方高性能库与原生 Transformers 生态的桥接.

    尝试使用 Unsloth 库加载预训练模型.
    [设计意图]: 仅在训练阶段调用, 旨在通过 Unsloth 的优化内核实现极致的训练加速和显存压缩.

    高级研究员视角的架构深度解析:

    为什么不直接在 load_model 里写?
    作为高级工程师, 我们要遵循 SRP(单一职责原则). 将 Unsloth 的加载逻辑独立出来, 可以让主加载流程 load_model 保持清晰. 主流程只需要检查这个函数是否返回了有效的 model, 如果返回 None, 则走常规路径.

    显存预算的考量:
    FastLanguageModel 的核心优势是在加载时就进行了 4-bit 量化算子的融合. 这段代码在微调超大模型(如 70B)时非常关键. 如果加载成功, 用户甚至可以在单张 RTX 3090/4090 上完成微调, 这是原生 transformers 库很难做到的.

    对 model_args 的原地修改(In-place mutation):
    代码中修改了 model_args.use_unsloth = False. 在复杂的 Python 系统中, 这是一种状态同步行为. 它确保了后续的 init_adapter 或 Trainer 逻辑不会再尝试寻找 Unsloth 相关的特殊配置, 从而保证了参数流在整个训练生命周期中的一致性.
    """

    # 1. 延迟导入与隔离 (Lazy Import & Dependency Isolation)
    # [为什么要这么写]: Unsloth 是一个可选依赖. 将其放在函数内部导入, 可以确保
    # 在未安装 unsloth 的环境下, 只要不激活 use_unsloth 开关, 整个项目依然能正常运行.
    # [解决的问题]: 避免了因缺少非必要库而导致的程序启动时强制报错, 提高了框架的环境适应性.
    from unsloth import FastLanguageModel  # type: ignore

    # 2. 参数协议转换 (Parameter Protocol Translation)
    # [为什么要这么写]: Unsloth 的 API 参数与 Hugging Face 标准的 from_pretrained 不完全一致.
    # [解决的问题]: 通过调用内部辅助函数 _get_unsloth_kwargs, 将 LLaMA-Factory 的通用配置
    # 转换为 Unsloth 能够理解的特定参数(如 max_seq_length, load_in_4bit 等).
    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.model_name_or_path, model_args, finetuning_args)

    try:
        # 3. 尝试加速加载
        # [执行逻辑]: 调用 Unsloth 封装的加载器, 它会自动执行针对 Llama/Mistral/Gemma 等架构的内核替换.
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)

    # 4. 自动容错与平滑回退 (Automated Fallback Mechanism)
    # [为什么要这么写]: Unsloth 虽然高效, 但目前仅支持部分主流架构(如 Llama-3, Qwen-2 等).
    # 当用户尝试用 Unsloth 加载一个它尚未适配的模型(如某些复杂的混合专家模型 MoE)时, 会抛出 NotImplementedError.
    except NotImplementedError:
        # [解决的问题]:
        #   a. 健壮性: 防止因后端库不支持特定模型而导致整个实验脚本崩溃.
        #   b. 状态回滚: 显式将 use_unsloth 设为 False. 这样 LLaMA-Factory 的后续逻辑
        #      (在 load_model 函数中)会自动切换回使用标准的 Hugging Face AutoModel 加载,
        #      实现了"能加速则加速, 不能加速则保底"的工业级逻辑闭环.
        logger.warning_rank0("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model = None
        model_args.use_unsloth = False

    return model


def get_unsloth_peft_model(
    model: "PreTrainedModel", model_args: "ModelArguments", peft_kwargs: dict[str, Any]
) -> "PreTrainedModel":
    r"""
    Get the peft model for the pretrained model with unsloth. Used in training.

    这段代码的核心任务是: 在应用 LoRA/PEFT 适配器时, 通过 Unsloth 的后端接管, 实现比标准 Hugging Face PEFT 快 2-3 倍且显存占用降低 70% 的极致微调.

    为预训练模型获取 Unsloth 优化的 PEFT (适配器) 模型. 仅在训练阶段调用.

    高级研究员视角的架构深度解析:

    关于 use_gradient_checkpointing: "unsloth" 的秘密:
    作为研究人员, 我们知道传统的梯度检查点(Gradient Checkpointing)虽然省显存, 但会增加约 33% 的计算开销(因为要重跑一遍前向过程). Unsloth 这里的优化逻辑在于它重新定义了反向传播的算子流, 使得只有极小部分的激活值(Activations)被重新计算. 这是微调 70B 模型能在单张显卡上跑起来的关键.

    动态算子注入(Operator Injection):
    FastLanguageModel.get_peft_model 内部实际上执行了"猴子补丁(Monkey Patching)". 它会扫描 target_modules, 并把这些 nn.Linear 层的 forward 方法替换为自己高度优化的实现. 这种"侵入式"的加速, 比单纯在外部包一层装饰器要有效得多.

    对 max_seq_length 的强依赖:
    高级 Python 工程中需要注意, Unsloth 会在这一步对模型的位置嵌入(Position Embeddings)进行"即时修剪或扩展(On-the-fly scaling)". 如果不在这里传入正确的 model_max_length, 可能会导致长文本微调时位置感官错乱.
    """

    # 1. 局部导入隔离 (Lazy Import & Dependency Isolation)
    # [为什么要这么写]: Unsloth 是 LLaMA-Factory 的可选加速组件. 将其放在函数内部导入,
    # 能够确保不使用 Unsloth 的用户无需安装此库, 避免了在非 GPU/非 Linux 环境下引发的导入错误.
    from unsloth import FastLanguageModel  # type: ignore

    # 2. 构造 Unsloth 专用的硬件加速参数 (Hardware-level Optimization Injection)
    # [解决的问题]:
    # - 显存预分配优化: Unsloth 需要根据 max_seq_length 预先优化 RoPE (旋转位置编码) 缓存,
    #   防止在长序列训练中由于动态分配导致的性能波动.
    # - 梯度检查点黑魔法: 将 use_gradient_checkpointing 设为 "unsloth" 而非布尔值,
    #   会启用 Unsloth 自研的基于 Triton 实现的梯度检查点, 这比 PyTorch 原生的 GC 快 30% 以上.
    unsloth_peft_kwargs = {
        "model": model,
        "max_seq_length": model_args.model_max_length,
        "use_gradient_checkpointing": "unsloth",
    }

    # 3. 融合通用配置与底层加速配置 (Parameter Merging & Factory Dispatch)
    # [为什么要这么写]: LLaMA-Factory 内部有一套标准的 PEFT 参数(如 lora_rank, lora_alpha, target_modules),
    # 存储在 peft_kwargs 中. 通过双星号解包 (**), 我们将这些通用微调参数与 Unsloth 强绑定的底层优化参数合并.
    # [解决的问题]: 实现了"策略层"与"算子层"的解耦. 用户通过配置文件定义的 LoRA 结构依然有效,
    # 但底层的线性算子(Linear layers)会被替换为 Unsloth 手写的、能自动处理 4-bit 量化的高性能 Triton 模板.
    return FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)


def load_unsloth_peft_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Load peft model with unsloth. Used in both training and inference.

    在 LLM 工程领域, Unsloth 以其对 Triton 内核的底层优化著称, 能显著提升 LoRA 微调和推理的速度. 这段代码的关键在于它如何平衡训练与推理的差异, 以及如何处理第三方加速库的兼容性边界.

    使用 Unsloth 引擎加载 PEFT (适配器) 模型. 支持训练模式和推理模式.

    [解决的问题]:
    Unsloth 并不只是简单的权重加载工具, 它会通过 Monkey Patching(猴子补丁)修改模型的计算图.
    本函数确保了这种修改在训练态和推理态下都能以最优性能运行.

    资深专家视角的架构设计分析:

    关于 FastLanguageModel.from_pretrained 的深度逻辑:
    与 Hugging Face 原生的 PeftModel.from_pretrained 不同, Unsloth 的这个接口内部实现了"Base Model + Adapter 一体化加载". 它会直接将 LoRA 权重的 A/B 矩阵逻辑合并到经过 Triton 优化的线性算子中. 这解决了标准 PEFT 在多层嵌套调用时产生的性能损耗.

    状态隔离策略:
    代码中区分了 is_trainable. 在高级开发实践中, 这是一个非常严谨的逻辑闭环:
    训练态: 关注的是 VRAM(显存)压缩, 所以参数中通常会包含 load_in_4bit=True 和特殊的梯度检查点.
    推理态: 关注的是 Latency(延迟)和 Throughput(吞吐), 所以必须通过 for_inference 关闭训练相关的 Hook.

    对 _get_unsloth_kwargs 的依赖:
    注意到参数中传递了 model_args.adapter_name_or_path[0]. 在多适配器场景下, Unsloth 目前主要针对主适配器进行优化. 这种设计体现了 LLaMA-Factory 在处理加速后端时"针对高频场景进行极致性能定制"的工程策略.
    """

    # 1. 局部延迟导入 (Lazy Import)
    # [为什么要这么写]: Unsloth 是一个可选的加速库.
    # [解决的问题]: 避免在未安装 unsloth 环境的用户运行非加速模式时, 产生 ImportError 导致程序崩溃.
    from unsloth import FastLanguageModel  # type: ignore

    # 2. 统一参数映射 (Parameter Mapping)
    # [为什么要这么写]: 调用内部工具函数获取 unsloth 专用的参数字典.
    # [解决的问题]: 将 LLaMA-Factory 抽象的 ModelArgs 转换为 Unsloth 内部 API 所需的特定格式,
    # 例如: 将 adapter 路径对齐到 FastLanguageModel 所需的 model_name 参数上.
    unsloth_kwargs = _get_unsloth_kwargs(config, model_args.adapter_name_or_path[0], model_args, finetuning_args)

    try:
        # 3. 动态配置梯度检查点 (Gradient Checkpointing)
        # [为什么要这么写]: 如果是推理模式 (is_trainable=False), 强制关闭梯度检查点.
        # [解决的问题]: 梯度检查点是通过"时间换空间"的技术. 在推理阶段没有反向传播, 开启它
        # 会导致冗余的前向计算, 增加不必要的响应延迟.
        if not is_trainable:
            unsloth_kwargs["use_gradient_checkpointing"] = False

        # 执行底层加载: Unsloth 会在此步替换原生的线性层算子为优化的 Triton 实现.
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        # 4. 架构支持边界校验 (Architecture Guardrail)
        # [解决的问题]: LLaMA-Factory 支持上百种模型, 但 Unsloth 仅针对 Llama, Mistral, Gemma
        # 等主流架构做了深度优化. 如果用户尝试加载不支持的模型(如某些 MoE 变体),
        # 抛出清晰的错误信息, 而不是在训练中途由于计算图错误而崩溃.
        raise ValueError("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))

    # 5. 推理模式特化优化 (Inference Specialization)
    # [为什么要这么写]: 调用 Unsloth 提供的 for_inference 接口.
    # [解决的问题]: 该操作会进一步精简计算图, 例如:
    # - 禁用所有的 Dropout 层.
    # - 优化 KV Cache 的动态分配逻辑.
    # - 确保权重处于 eval 模式, 从而达到极致的生成速度.
    if not is_trainable:
        FastLanguageModel.for_inference(model)

    return model

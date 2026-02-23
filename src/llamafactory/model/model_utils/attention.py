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

from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    """
    注意力机制(Attention Implementation) 的配置是决定训练效率(显存占用)和数值稳定性(模型收敛)的关键.
    configure_attn_implementation 函数的核心使命是: 屏蔽不同模型架构、不同硬件平台(GPU/NPU)以及不同加速库版本之间的差异, 为当前模型寻找最优的注意力算子.

    资深架构师的深度总结:

    分治策略(Dispatching): 这段代码很好地展示了如何处理"标准"与"特例". 对于 90% 的模型走通用流程, 对于像 gpt_oss、gemma2、kimi_vl 这种"不听话"或有特殊数学要求的模型走专有路径.
    防御性编程: 在微调大型模型时, 最怕运行了半小时数据预处理, 最后在加载模型权重时因为算子不支持而崩溃. 代码中大量的 logger.warning 和 return 机制就是在初始化阶段尽早"排雷".
    对国产硬件的支持: 通过判断 is_torch_npu_available, 使得 LLaMA-Factory 能够平滑支持昇腾 NPU 等异构算力, 这是该项目在国内工业界广受欢迎的重要原因.
    """

    # 1. 动态检测环境能力
    # [为什么要这么写]: FlashAttention 不是 Python 标准库, 且与驱动/CUDA版本强绑定.
    # 延迟导入可以防止在非 GPU 环境下因找不到库而直接崩溃.
    from transformers.utils import is_flash_attn_2_available

    # 2. 针对特定模型架构的"硬编码"优化 (The FA3 bleeding edge)
    # [场景]: gpt_oss 模型.
    # [解决的问题]: gpt_oss 这种新型架构通常需要极致的吞吐. FlashAttention-3 (FA3) 提供了针对
    # Hopper 架构 (H100) 的特殊优化. 由于 transformers 还没完全原生支持 FA3, 这里通过
    # 手动加载并注册 hub_kernels, 强制注入 FA3 算子.
    if getattr(config, "model_type", None) == "gpt_oss":
        from transformers.integrations.hub_kernels import load_and_register_kernel

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        load_and_register_kernel(flash_attn3_kernel)
        # 强制修改私有属性, 绕过标准的 transformers 检查流程
        setattr(config, "_attn_implementation", flash_attn3_kernel)
        setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3

        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return

    # 3. 针对模型数学特性的鲁棒性检查 (Correctness over Speed)
    # [场景]: Gemma-2 模型.
    # [解决的问题]: Gemma-2 引入了 Logit Soft-capping 机制来防止 Attention Logit 过大.
    # - 问题点: PyTorch 原生的 SDPA 算子目前不完全支持这种 Soft-capping 操作.
    # - 风险点: 如果用户强行用 SDPA 训练 Gemma-2, 模型会由于数学逻辑不匹配而导致 Loss 无法下降.
    # - 方案: 强制用户切换到 FlashAttention-2(它支持该特性)或者退回到 eager 模式.
    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    # 4. 用户意图解析与环境版本校验
    # [为什么要这么写]: 将用户的命令行配置(AUTO/SDPA/FA2)映射为 transformers 内部标识符.
    if model_args.flash_attn == AttentionFunction.AUTO:
        # 尊重 transformers 原生的自动选择逻辑
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        # 最通用的 Python 实现, 不依赖加速库, 但慢且吃显存
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        # [解决的问题]: SDPA 是 PyTorch 2.x 的核心特性.
        # 这里进行版本强校验, 防止在旧版本 PyTorch 环境下启动失败.
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        # [跨平台支持]: 除了 NVIDIA, 还考虑了国产算力 NPU (华为昇腾).
        # [解决的问题]: NPU 也有对应的 FlashAttention 算子实现.
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            logger.warning_rank0("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    # 5. 动态注入模型配置 (Monkey Patching the Config)
    # [为什么要这么写]: 不同的自定义模型(Custom Models)对属性名的约定不一致.
    # [解决的问题]:
    # - InternLM2 使用的是 `attn_implementation`.
    # - Kimi-VL 是多模态模型, 需要同时配置 Vision 和 Text 两个子组件.
    # - 标准 Huggingface 模型使用私有的 `_attn_implementation`.
    # 通过分支判断, 确保注入的参数在模型实例化阶段能被正确识别.
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")

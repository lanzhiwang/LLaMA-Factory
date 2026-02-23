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

import inspect
from typing import TYPE_CHECKING

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def apply_liger_kernel(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    """
    Liger Kernel 是目前 LLM 微调领域非常前沿的优化技术(由 LinkedIn 开源).
    这段代码的核心使命是: 通过高性能的 Triton 算子替换原生的 PyTorch 实现, 极大地降低显存占用并提升训练吞吐量.

    深度架构解析(高级工程师视角):

    词表显存危机 (The Vocabulary Memory Bottleneck):
    现在的模型(如 Qwen, Llama 3)词表(Vocab Size)通常在 10 万到 15 万之间. 在普通的 PyTorch 训练中, 最后一层 Linear 输出的 Logits 会占据极其恐怖的显存. Liger Kernel 的核心贡献就是分块计算(Chunking), 它不一次性产出完整的 Logits 矩阵.

    解耦与动态性:
    代码中使用了大量的局部 from ... import .... 这是一种高级 Python 技巧, 避免了在文件顶部导入所有依赖导致的冷启动时间过长或循环导入问题, 同时也确保了只有在真正需要该模型内核时才触发加载.

    计算图稳定性:
    Liger Kernel 不仅仅是快, 它在计算 RMSNorm 和 CrossEntropy 时通常比原生 FP16 实现具有更好的数值稳定性. 通过自定义的 Triton Kernel, 它能更精细地管理累加器的精度, 降低在大规模微调中出现 Loss 突变(Spike)的概率.
    """

    # 1. 状态准入检查
    # [为什么要这么写]: Liger Kernel 主要是为了优化梯度下降过程中的反向传播.
    # [解决的问题]: 在推理模式(is_trainable=False)下开启 Liger 不仅没有意义, 还可能引入不必要的依赖.
    # 同时也尊重用户在命令行中通过 enable_liger_kernel 手动控制开关的意图.
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    # 2. 动态路由分发 (Dynamic Dispatching)
    # [为什么要这么写]: Liger Kernel 采用了"猴子补丁(Monkey Patching)"技术,
    # 针对每种模型架构(如 Llama, Qwen, Gemma)的层名称和拓扑结构进行了特化优化.
    # [解决的问题]: 屏蔽不同厂商模型实现的差异. 由于不同模型的模块命名不同(例如有的叫 mlp, 有的叫 feed_forward),
    # 必须根据 model_type 导入特定的应用函数, 确保补丁能准确"缝合"到对应的模型实例上.
    model_type = getattr(config, "model_type", None)
    if model_type == "gemma":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_liger_kernel
    elif model_type == "gemma2":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_liger_kernel
    elif model_type == "gemma3":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3 as apply_liger_kernel
    elif model_type == "gemma3_text":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text as apply_liger_kernel
    elif model_type == "glm4":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4 as apply_liger_kernel
    elif model_type == "glm4v":
        from liger_kernel.transformers import apply_liger_kernel_to_glm4v as apply_liger_kernel
    elif model_type == "granite":
        from liger_kernel.transformers import apply_liger_kernel_to_granite as apply_liger_kernel
    elif model_type == "llama":
        from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_liger_kernel
    elif model_type == "llava":
        from liger_kernel.transformers import apply_liger_kernel_to_llava as apply_liger_kernel
    elif model_type == "mistral":
        from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_liger_kernel
    elif model_type == "mixtral":
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_liger_kernel
    elif model_type == "mllama":
        from liger_kernel.transformers import apply_liger_kernel_to_mllama as apply_liger_kernel
    elif model_type == "olmo2":
        from liger_kernel.transformers import apply_liger_kernel_to_olmo2 as apply_liger_kernel
    elif model_type == "paligemma":
        from liger_kernel.transformers import apply_liger_kernel_to_paligemma as apply_liger_kernel
    elif model_type == "phi3":
        from liger_kernel.transformers import apply_liger_kernel_to_phi3 as apply_liger_kernel
    elif model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_liger_kernel
    elif model_type == "qwen2_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl as apply_liger_kernel
    elif model_type == "qwen2_5_vl":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl as apply_liger_kernel
    elif model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_liger_kernel
    elif model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_liger_kernel
    elif model_type == "gpt_oss":
        # 3. 实验性特性容错
        # [为什么要这么写]: 针对非官方主分支支持的模型(如 gpt_oss), 使用 try-except 保护.
        # [解决的问题]: 防止因为用户环境安装的 liger-kernel 版本过旧而导致整个加载流程崩溃.
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss as apply_liger_kernel
        except ImportError:
            logger.warning_rank0("Please install liger-kernel from https://github.com/Comet0322/Liger-Kernel.")
            return
    else:
        # 4. 架构兼容性兜底
        # 如果模型不在 Liger 的支持列表中, 优雅退出而非报错, 体现了微调框架的鲁棒性.
        logger.warning_rank0("Current model does not support liger kernel.")
        return

    # 5. 核心: 算子融合与逻辑冲突处理 (Operator Fusion Logic)
    # [为什么要这么写]: 这是全函数最体现"高级研究员"思考的地方.
    # - 背景: Liger 的核心优化之一是 fused_linear_cross_entropy. 它将 Linear 层和 CrossEntropy 合并,
    #   利用 Triton 直接计算 Loss, 从而不产出中间巨大的 Logits 张量(Logits 显存占用 = batch * seq_len * vocab_size).
    # [解决的问题]: 解决"算法需求"与"显存优化"的矛盾.
    # - 如果是 SFT/PT 阶段, 我们只需要 Loss, 所以可以用融合算子.
    # - 如果是 DPO, PPO 或 RM 阶段(即 require_logits=True), 算法需要原始的 Logits 来计算概率比率.
    # - 逻辑: 如果 require_logits 为 True, 我们通过 inspect 检测函数签名, 强制关闭"融合线性交叉熵",
    #   但保留其他的优化(如 RMSNorm 或 SwiGLU 的内核替换), 以确保微调算法逻辑的正确性.
    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters:
        logger.info_rank0("Current training stage does not support chunked cross entropy.")
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
    else:
        kwargs = {}

    # 6. 执行注入
    # 正式执行猴子补丁, 将优化后的算子注入内存中的 transformers 类定义.
    apply_liger_kernel(**kwargs)
    logger.info_rank0("Liger kernel has been applied to the model.")

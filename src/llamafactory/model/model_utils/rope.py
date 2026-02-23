# Copyright 2025 LMSYS and the LlamaFactory team.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
# This code is inspired by the LMSYS's FastChat library.
# https://github.com/lm-sys/FastChat/blob/v0.2.30/fastchat/train/train.py
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

import math
from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import RopeScaling


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    """
    在 LLM 微调中, 上下文长度扩展(Context Window Extension) 是最常见的需求之一. 如果一个模型预训练时只有 4k 长度, 但你想微调到 32k, 直接训练会导致模型对超出 4k 的位置编码完全陌生. 这段代码的作用就是在模型加载前, 通过修改配置来改变旋转位置编码的基频或线性比例, 从而让模型"理解"更长的序列.

    配置 RoPE 缩放策略, 用于扩展模型的最大上下文长度.

    高级架构师视角的深度解析:

    RoPE 缩放的物理意义:
    RoPE 本质上是通过旋转角度来编码位置. 当你扩展 context 时, 如果不缩放, 角度会转到预训练从未见过的范围(Out-of-distribution). 缩放(尤其是 Linear Scaling)本质上是将原本 0-8k 的角度范围"挤压"回 0-4k 的角度空间内, 让模型复用已有的注意力感知能力.

    Llama-3 特化处理的必要性:
    Llama-3.1 引入了 high_freq_factor. 在高频分量上保持较少插值, 在低频分量上进行较多插值. 这解决了早期 Linear Scaling 导致的短序列(Short-context)能力退化问题. LLaMA-Factory 这里的硬编码参数(1.0 和 4.0)是目前兼容 Llama-3.1 8B 到 70B 模型的最稳健工程值.

    对生产环境的健壮性考虑:
    注意代码中对 math.ceil 的使用和 max_position_embeddings 的同步修改. 作为高级工程师, 我们必须预见到: 如果只改 RoPE 因子而忘了改 Config 的最大长度, 微调脚本可能会在处理长数据时因为 RuntimeError 或是默认的 Position ID 生成器越界而中断.
    """

    # 1. 快速准入检查
    # [为什么要这么写]: 如果用户没指定缩放参数, 说明不需要扩展上下文, 直接跳过以保持模型原始行为.
    if model_args.rope_scaling is None:
        return

    # [解决的问题]: 并不是所有架构都使用 RoPE(例如一些较旧的模型).
    # 检查 config 是否有此属性, 防止在不支持的模型上强行注入导致运行时崩溃.
    if not hasattr(config, "rope_scaling"):
        logger.warning_rank0("Current model does not support RoPE scaling.")
        return

    # 2. 溯源原始上下文长度 (Find the True Base Length)
    # [为什么要这么写]: 为了计算缩放比例(Factor), 必须准确知道模型"预训练"时的原始长度.
    # [解决的问题]: 处理"二次缩放"的情况.
    # 如果模型已经带了缩放配置, 原始长度会被存在 original_max_position_embeddings.
    # 如果是原生模型, 则直接读取 max_position_embeddings.
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "original_max_position_embeddings" in rope_scaling:
        old_max_length = rope_scaling["original_max_position_embeddings"]
    elif hasattr(config, "max_position_embeddings"):
        old_max_length = getattr(config, "max_position_embeddings", None)
    else:
        logger.warning_rank0("Cannot find the max position embeddings in the config.")
        return

    # 3. 计算缩放因子 (Scaling Factor Calculation)
    if model_args.model_max_length is not None:  # training 处于训练阶段
        # [解决的问题]: 如果用户设置的长度比原始长度还短, 缩放反而会降低模型精度.
        if model_args.model_max_length <= old_max_length:
            logger.warning_rank0("Input length is smaller than max length. Disabling rope scaling.")
            return

        # [学术背景警示]: Dynamic NTK 缩放虽然在推理时好用(零样本扩展),
        # 但在微调时表现不佳, 因为它在训练过程中会改变位置频率的分布, 导致收敛不稳定.
        if model_args.rope_scaling == RopeScaling.DYNAMIC:
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        # [计算逻辑]: 使用 math.ceil 向上取整.
        # [为什么要这么写]: 因子必须足以覆盖用户请求的 model_max_length.
        rope_factor = float(math.ceil(model_args.model_max_length / old_max_length))
    else:  # inference 纯推理阶段
        # 如果推理时未指定长度, 默认给 2 倍冗余空间.
        rope_factor = 2.0

    # 4. 构建配置字典 (Standardized Config Construction)
    # [为什么要这么写]: 统一不同枚举和字符串的输入格式, 适配 Transformers 库要求的字典结构.
    rope_kwargs = {
        "rope_type": getattr(model_args.rope_scaling, "value", model_args.rope_scaling),  # handle enum
        "factor": rope_factor,
    }

    # [关键步骤]: 同步更新 config.max_position_embeddings.
    # [解决的问题]: 如果不更新这个值, 即便 RoPE 逻辑对齐了, 模型在输入超过 old_max_length 后
    # 依然会因为越界或配置检查被强行截断.
    setattr(config, "max_position_embeddings", old_max_length * rope_factor)
    logger.info_rank0(f"Enlarge max model length from {old_max_length} to {old_max_length * rope_factor}.")

    # 5. 处理特定缩放算法的附加参数 (Specialized Algorithm Patching)
    # [解决的问题]:
    # - DYNAMIC/YARN 需要保存原始长度以计算频率偏差.
    # - LLAMA3 采用了特殊的超参数化 RoPE (Llama-3.1 官方引入).
    if model_args.rope_scaling in [RopeScaling.DYNAMIC, RopeScaling.YARN]:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
    elif model_args.rope_scaling == RopeScaling.LLAMA3:
        # [为什么要这么写]: 这是 Llama-3.1 官方配置的"经验黑魔法".
        # low_freq_factor 和 high_freq_factor 决定了哪些维度的位置编码需要插值,
        # 哪些需要保持原样, 以在长短文本间取得平衡.
        rope_kwargs["original_max_position_embeddings"] = old_max_length
        rope_kwargs["low_freq_factor"] = 1.0
        rope_kwargs["high_freq_factor"] = 4.0

    # 6. 将补丁注入配置
    # [为什么要这么写]: 使用 setattr 进行运行时注入.
    # [解决的问题]: 通过这种方式, 原本只支持 4k 的模型, 在实例化瞬间就会加载对应的
    # Linear Scaling 或 YaRN 算子, 实现无缝的长文本能力支持.
    setattr(config, "rope_scaling", rope_kwargs)
    logger.info_rank0(
        f"Using {rope_kwargs['rope_type']} scaling strategy and setting scaling factor to {rope_kwargs['factor']}."
    )

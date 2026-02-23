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


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


def configure_kv_cache(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    """
    在 LLM 的工程实践中, KV Cache(键值缓存) 的管理是区分"训练态"和"推理态"的关键.
    这段代码看似简单, 实际上解决了 计算图一致性 和 显存溢出(OOM)风险 两个核心工程痛点.

    配置模型的 KV Cache 行为, 根据训练或推理模式动态调整.

    高级研究员视角的架构深度解析:

    关于 use_cache 的工业标准:
    在 Hugging Face 的生态中, use_cache 是一个"隐形陷阱". 很多新手在微调时会发现显存莫名其妙溢出, 或者报出 RuntimeError: Gradient computation has been unexpectedly disabled. LLaMA-Factory 在此通过 is_trainable 进行硬硬约束(Hard Constraint), 体现了框架的健壮性设计.

    对复合架构(Composite Models)的防御性编程:
    现在的 LLM 发展迅速, 多模态模型(VLM)层出不穷. 它们的 config 结构并不统一. 代码中对 text_config 的检测展示了对 LLaVA, Idefics, Qwen-VL 等主流架构的深度兼容. 如果漏掉这一步, 用户在训练多模态模型时, 底层文本分支可能依然尝试分配缓存空间, 导致极其隐蔽的显存浪费.

    动态属性注入(setattr)的必要性:
    由于 LLaMA-Factory 支持成百上千种不同的模型, 我们不能假设每个模型配置文件里都有 use_cache 字段. 使用 setattr 而不是 config.use_cache = ... 可以防止在某些非标准配置文件中出现 AttributeError, 确保了框架的通用适配能力.
    """

    # 1. 推理态处理 (Inference / Evaluation Mode)
    # [解决的问题]: 推理性能优化.
    if not is_trainable:
        # 设置模型配置中的 use_cache 属性.
        # [为什么要这么写]: 在生成式任务中, KV Cache 会保存先前 Token 的计算结果,
        # 从而避免在生成下一个 Token 时进行重复计算. 这能将推理速度提升数倍.
        setattr(config, "use_cache", model_args.use_kv_cache)

        # 兼容性处理: 针对多模态复合模型(如 LLaVA, Qwen-VL 等).
        # [为什么要这么写]: 这类模型的配置通常嵌套在 text_config 字段下.
        # 如果只修改外层 config 而不修改内层 text_config, 推理引擎可能无法识别该开关,
        # 导致缓存机制失效, 推理变慢.
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", model_args.use_kv_cache)

        if model_args.use_kv_cache:
            logger.info_rank0("KV cache is enabled for faster generation.")
        else:
            # 某些特殊场景(如长文本压力测试或显存极度紧张时)可能需要关闭 KV Cache 以节省空间.
            logger.info_rank0("KV cache is disabled.")

    # 2. 训练态处理 (Training Mode)
    # [解决的问题]: 计算逻辑正确性与显存冲突.
    else:
        # 强制在训练期间关闭 use_cache.
        # [为什么要这么写]:
        # 1. 数学逻辑: 训练是并行的(Parallel Prefilling), 模型一次性处理整个序列,
        #    不需要像推理那样逐个 Token 递进, 因此 KV Cache 无效.
        # 2. 梯度冲突: KV Cache 机制与梯度检查点(Gradient Checkpointing)在底层是冲突的.
        #    如果开启 KV Cache 训练, PyTorch 的计算图会变得非常混乱, 导致无法正确回传梯度或抛出运行时异常.
        # 3. 节省显存: 训练已经占用了大量显存, 开启无用的 KV Cache 会白白浪费额外的显存空间.
        setattr(config, "use_cache", False)

        # 同样对复合模型进行强制同步.
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", False)

        logger.info_rank0("KV cache is disabled during training.")

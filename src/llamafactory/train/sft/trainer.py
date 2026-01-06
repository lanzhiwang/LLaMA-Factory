# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

"""
这个类不仅仅是对 Hugging Face Seq2SeqTrainer 的简单继承, 它实际上是一个针对 LLM 微调工业级痛点设计的"增强补丁包".
它处理了包括 FP8 训练适配、复杂优化器集成、多模态模型支持以及推理评测中的 Token 偏移等核心工程问题.
"""

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.

    继承自 Seq2SeqTrainer, 用于计算生成式指标(如 BLEU 和 ROUGE)并集成了多种工业级微调优化.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 1. 低精度训练优化 (FP8 Environment)
        # [为什么要这么写]: FP8 训练是 H100/L40S 等新一代 GPU 的核心竞争力, 能大幅提升吞吐量.
        # [解决的问题]: 在初始化前配置 FP8 环境(如 TransformerEngine 适配), 防止后续算子因环境未就绪而报错.
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)

        # 2. 框架前向兼容性处理 (Transformers Version Compatibility)
        # [为什么要这么写]: Transformers v4.46+ 将内部的 tokenizer 统一更名为 processing_class.
        # [解决的问题]: 保证代码能够跨版本运行, 避免在升级 transformers 库后出现 AttributeError.
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        # 3. 梯度累积下的 Loss 计算修正
        # [为什么要这么写]: 显式关闭模型自动接受 Loss 相关的 kwargs.
        # [解决的问题]: 这是一个针对 HF Transformers 的深度 Hack. 在开启梯度累积时,
        # 如果模型内部处理 Loss 逻辑, 可能会导致梯度计算步数不一致. 将其设为 False
        # 强制让 Trainer 外部接管 Loss 计算, 确保梯度的数学正确性.
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        # 4. 生成参数持久化
        # [为什么要这么写]: 将推理参数(如 max_new_tokens, temperature)注入 Trainer 内部.
        # [解决的问题]: 确保在 evaluate/predict 阶段, 模型生成的配置与用户定义的生成策略严格一致.
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        # 5. 多模态支持 (Multi-modal Processor)
        # [解决的问题]: 对于 LLaVA 等多模态模型, 除了保存权重, 还必须同步保存 Processor(包含图像缩放、特征提取逻辑).
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # 6. BAdam 显存优化器集成
        # [为什么要这么写]: 通过 MethodType 动态替换梯度裁剪方法.
        # [解决的问题]: BAdam 是一种分块优化器, 原生的梯度裁剪会破坏其计算逻辑.
        # 这种"热更新"类方法的写法能无损地将第三方高性能优化器注入到原生流程中.
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # 7. 动态损失函数支持 (DFT Loss)
        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # 8. 硬件状态最终校验
        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        # [解决的问题]: 原生 Trainer 只支持 AdamW 等基础优化器. 这里通过重写支持了
        # GaLore, BAdam, AdamW_8bit 等 LLM 微调必备的高级优化器.
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        # [解决的问题]: 支持更加精细的学习率调度(如带有热启动的自定义余弦退火).
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        # [解决的问题]: 在特定的场景(如课程学习或调试)下, 需要完全禁止数据洗牌以观察模型收敛.
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.

        核心重写逻辑: 在生成的 Token 中移除 Prompt 部分.
        """

        # 1. 标签预处理
        # [为什么要这么写]: 如果是 generate 推理模式, 要把 labels 弹出, 防止模型在前向传播中"偷看"答案.
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        # 2. 调用原生生成逻辑
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        # 3. 结果对齐 (Prompt Masking)
        # [为什么要这么写]: Decoder-only 模型(如 Llama)的 generate() 通常返回 [Prompt + Response].
        # [解决的问题]: 计算指标(如 ROUGE)时只需 Response. 这里将 Prompt 部分替换为 pad_token_id,
        # 方便后续计算逻辑仅关注模型新生成的有效内容.
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.

        将模型预测结果(Prompt/Predict/Label)持久化为 JSONL 文件.
        """

        # 1. 分布式保护: 仅允许主卡执行 IO, 防止多进程写文件冲突.
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        # 2. 标签/预测值清理 (Handle IGNORE_INDEX)
        # [为什么要这么写]: 将训练时用的 -100 (IGNORE_INDEX) 替换回 padding, 否则 Tokenizer 无法解码.
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        # 3. 填充对齐 (Pad Token Shift)
        # [解决的问题]: 生成结果可能带有前置 Padding, 这里通过非零索引查找, 将有效文本移至最前.
        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        # 4. 批量解码与保存
        # [为什么要这么写]: 解码为人类可读的字符串并存储为结构化 JSONL.
        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

"""
资深专家总结:

工程设计的健壮性: 代码中对 transformers 版本的动态检测和对 processing_class 的兼容性处理, 体现了作为大型开源项目对外部依赖变化的敏感度和适应性.

推理与训练的逻辑闭环: 在 prediction_step 中对 Prompt 进行掩码处理, 是一个非常容易被新手忽视的细节. 如果不处理, 计算 ROUGE 等指标时会将 Prompt 的准确度也算进去, 导致评估结果虚高.

针对显存的极致压榨: 集成了 FP8 和 BAdam, 表明该 Trainer 旨在处理千亿级参数模型在有限资源下的训练任务.

通过这些修改和注释, 该类已经变成了一个功能完备、能够直接用于生产环境的 LLM 实验调度器.
"""

# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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
from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


"""
这个类是 奖励模型(Reward Model, RM) 训练的核心. 它继承自 Hugging Face 的 Trainer,
但针对"成对偏好学习"(Pairwise Preference Learning)进行了重度定制.
其核心挑战在于: 如何从双流(Chosen/Rejected)输入中提取标量分数, 并实现符合 Bradley-Terry 模型的损失函数.
"""

class PairwiseTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.

    继承自 Trainer 用于计算成对损失(Pairwise Loss).
    这是奖励模型(RM)训练的核心类.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        # 1. 兼容性适配 (Version Compatibility)
        # [为什么要这么写]: Transformers 4.46+ 版本为了统一多模态支持, 将 tokenizer 改名为 processing_class.
        # [解决的问题]: 确保 LLaMA-Factory 能够跨版本运行, 既支持旧版 transformers, 也能无缝对接最新的多模态模型 API.
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)

        # 2. 行为重写 (Trainer Behavior Overwrite)
        # [为什么要这么写]: 强制设置 model_accepts_loss_kwargs 为 False.
        # [解决的问题]: 默认的 Trainer 会尝试检测模型 forward 是否能处理 loss.
        # 在 RM 训练中, 我们需要由 Trainer 手动处理 Chosen/Rejected 的双流逻辑, 而不是由模型自身计算.
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss

        # 3. 关键回调注入 (Critical Callback Injection)
        # FixValueHeadModelCallback:
        #   RM 模型特有一个 ValueHead(通常是 linear(hidden_size, 1)). 这个回调确保在模型保存时,
        #   这个非原生 Transformer 的层级能够被正确持久化, 否则 PPO 阶段将无法加载奖励分数.
        self.add_callback(FixValueHeadModelCallback)

        # SaveProcessorCallback:
        #   如果是多模态模型(如 LLaVA), 需要保存对应的 Processor(处理图像等非文本输入).
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # 4. 显存优化器适配 (BAdam Support)
        # [为什么要这么写]: BAdam 是一种针对超大规模模型微调的显存优化技术.
        # [解决的问题]: 在有限显存下微调大模型. 这里通过动态替换方法(MethodType)修改了 accelerator 的梯度裁剪逻辑,
        # 绕过了底层框架对非标准优化器的兼容性限制.
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        # [为什么要这么写]: 调用工厂函数 create_custom_optimizer.
        # [解决的问题]: 支持除了 AdamW 之外的多种优化器(如 Adafactor, GaLore, LION 等),
        # 允许研究员针对不同尺寸的模型和硬件选择最优的收敛策略.
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        # [解决的问题]: 实现自定义的学习率衰减逻辑(如带有热启动的余弦退火), 确保长周期训练的稳定性.
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        # [为什么要这么写]: 提供禁止洗牌(Shuffling)的选项.
        # [解决的问题]: 在某些特定的微调场景(如课程学习或顺序敏感的指令微调)下,
        # 用户可能需要完全按照数据集原始顺序进行训练.
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Compute pairwise loss. The first n examples are chosen and the last n examples are rejected.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple.
        See: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L3842

        计算成对损失(Pairwise Loss).
        输入格式: 前 n 个样本是 chosen(用户偏好的), 后 n 个样本是 rejected(用户拒绝的).
        """

        # 1. 提取所有样本的分数 (Forward Pass)
        # model 返回的是 (batch_size, seq_len, 1), 代表每个 token 位置的标量分数
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

        # 2. 拆分 Batch (Batch Splitting)
        # [为什么要这么写]: 在数据预处理阶段, 我们将 Chosen 和 Rejected 拼接在了一个 batch 中.
        # 这里将其沿 batch 维度(dim 0)平分.
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)

        # 3. 提取序列末尾分数 (Token-level Reward Gathering)
        # [为什么要这么写]: RM 通常使用序列中[最后一个有效 token](即 EOS 之前的位置)的值作为整个句子的奖励分数.
        # [解决的问题]: 通过 attention_mask 计算序列长度, 并使用 .gather 动态获取最后一位的得分.
        # 这比简单取最后一个 token 更准确, 因为 batch 中存在 padding.
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
        rejected_scores = rejected_rewards.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
        chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()

        # 4. 核心损失函数 (Bradley-Terry Model Loss)
        # [为什么要这么写]: loss = -log(sigmoid(score_chosen - score_rejected)).
        # [解决的问题]: 这是 RM 训练的标准数学模型.
        # 它鼓励模型拉大 Chosen 和 Rejected 之间的分值差距. 使用 logsigmoid 能在差值较大时提供更稳定的梯度.
        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
        if return_outputs:
            return loss, (loss, chosen_scores, rejected_scores)
        else:
            return loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.

        将预测结果(Chosen 和 Rejected 的分数)保存到本地 JSONL 文件.
        """

        # 1. 多卡保护: 仅在主进程(Rank 0)执行保存逻辑, 防止文件写冲突.
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        # 2. 结构化持久化
        # [解决的问题]: 方便研究员在训练结束后对奖励模型的表现进行统计分析(如绘制分数分布直方图).
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: list[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))

            writer.write("\n".join(res))

"""
资深开发工程师视角下的亮点总结:

极简的接口逻辑: 尽管 LLM 的前向传播很复杂, 但通过巧妙的 batch_size // 2 拆分, PairwiseTrainer 成功复用了 Hugging Face 的高性能分布式数据流, 而无需重写底层的 DDP 逻辑.

鲁棒的奖励提取: 使用 attention_mask.sum() - 1 来精确定位奖励 token. 这是 RM 训练中最容易出错的地方(很多开源实现会错误地直接取最后一维, 导致在有 padding 的情况下结果全错), LLaMA-Factory 在这里做得很严谨.

针对 PPO 的前瞻性设计: FixValueHeadModelCallback 的注入反映了作者深厚的工程经验, 解决了训练与推理/对齐阶段模型参数不一致的痛点.

如果你正在研究如何改进偏好学习(例如从 Bradley-Terry 模型转向 DPO 架构), 这段代码是你最好的基石.
"""

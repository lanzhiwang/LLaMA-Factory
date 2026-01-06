# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy
from .trainer import PairwiseTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_rm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # 1. 加载 Tokenizer 及其配套组件
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # 2. 模板对齐与词表修正 (Template Alignment)
    # [为什么要这么写]: RM 训练极度依赖特定的 prompt 格式(通常是: Instruction + Input + Response).
    # [解决的问题]: 确保推理和训练时使用的特殊 Token(如 <|im_start|>)一致.
    # 如果 Tokenizer 缺失这些 Token, 该函数会负责将其补齐并同步调整模型的 Embedding 层大小.
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 3. 加载偏好数据集 (Pairwise Dataset Loading)
    # [为什么要这么写]: 传入 stage="rm" 参数.
    # [解决的问题]: RM 训练需要的是"成对"数据(Chosen vs Rejected).
    # 这里的 get_dataset 会将原始数据处理成两个输入流, 以便后续计算 Bradley-Terry 损失.
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)

    # 4. 加载模型并注入价值头 (ValueHead Injection)
    # [为什么要这么写]: 显式设置 add_valuehead=True.
    # [解决的问题]: 标准的 LLM 输出的是词表概率分布(Vocab Size), 而奖励模型需要输出的是一个标量分数(Scalar Score).
    # load_model 会在 Transformer 的最后一层隐藏状态上挂载一个 1x1 的线性层(ValueHead),
    # 从而将模型从"词语预测器"转变为"评分器".
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    # 5. 专门的成对数据整理器 (Pairwise Data Collator)
    # [为什么要这么写]: 使用 PairwiseDataCollatorWithPadding 并设置 pad_to_multiple_of=8.
    # [解决的问题]:
    # - 成对处理: 确保在一个 Batch 中, Chosen 和 Rejected 序列被正确对齐和填充.
    # - 性能优化: pad_to_multiple_of=8 能够触发 NVIDIA GPU 的 Tensor Cores 加速,
    #   在半精度(fp16/bf16)下极大提升矩阵运算效率.
    data_collator = PairwiseDataCollatorWithPadding(
        template=template, model=model, pad_to_multiple_of=8, **tokenizer_module
    )

    # 6. 初始化 PairwiseTrainer
    # [为什么要这么写]: 继承自标准的 Trainer 但重写了 compute_loss.
    # [解决的问题]: RM 训练不使用交叉熵损失, 而是使用排序损失(Ranking Loss).
    # 它需要计算 log(sigma(score_chosen - score_rejected)), 旨在让 Chosen 的分数尽可能高于 Rejected.
    # 同时传入 ComputeAccuracy 以监控训练过程中模型对人类偏好判断的准确率.
    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeAccuracy(),
        **dataset_module,
        **tokenizer_module,
    )

    # 7. 训练流程
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

        # 8. 修复 ValueHead 权重保存 (ValueHead Weight Persistence)
        # [为什么要这么写]: 在训练结束后调用 fix_valuehead_checkpoint.
        # [解决的问题]: Hugging Face 的 save_pretrained 默认不识别自定义的 ValueHead.
        # 如果不手动处理, 保存的 Checkpoint 会丢失这个关键评分层, 导致模型在 PPO 阶段无法作为奖励函数使用.
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # 9. 可视化指标 (Distributed Logging)
        # [为什么要这么写]: 仅在 world_process_zero(主进程)且开启 plot_loss 时绘图.
        # [解决的问题]: 防止多卡分布式训练时, 多个进程同时读写同一个图片文件导致的文件损坏.
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    # 10. 评估逻辑
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. 预测逻辑 (Inference for RM)
    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # 12. 自动生成 Model Card
    # [为什么要这么写]: 训练结束后自动上传元数据到 Hugging Face Hub.
    # [解决的问题]: 记录训练所用的数据集、超参数以及使用的 LLaMA-Factory 版本, 方便开源社区复现和溯源.
    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

"""
高级研究员视角的架构解析:

为什么 RM 需要特殊的 PairwiseTrainer?
标准的 Trainer 是为单一预测任务设计的. 在 RM 中, 我们需要模型同时处理同一提示下的两个不同回复, 并对比它们的得分. PairwiseTrainer 的内部逻辑确保了 Chosen 和 Rejected 序列在同一个 Forward Pass 中被计算, 或者通过精巧的索引管理来计算它们之间的相对差值.

准确率(Accuracy)在 RM 中的特殊含义:
这里的 ComputeAccuracy 并不是分类任务中的准确率, 而是 "偏好匹配率". 即: 模型给 Chosen 样本打的分数高于 Rejected 样本的比例. 如果这个指标达到 0.7-0.8, 通常说明奖励模型已经很好地捕捉到了偏好特征.

对 PPO 阶段的工程预案:
fix_valuehead_checkpoint 的存在体现了 LLaMA-Factory 的工程深度. 在后续的 PPO 阶段, 我们会加载两个模型: 一个是待优化的策略模型(Actor), 另一个就是这里的奖励模型(Reward). 如果奖励模型的 ValueHead 没有被正确持久化, 整个 RLHF 流程就会中断.

这段代码展示了如何在一个通用的微调框架内, 通过高度抽象的 load_model 和自定义 Trainer 来支持像 RM 这样复杂的特定算法. 希望对你有所帮助!
"""

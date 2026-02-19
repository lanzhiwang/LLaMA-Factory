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

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

"""
这段代码是 SFT 任务的"总指挥部".
它不仅要处理传统的模型加载和训练, 还要应对 Sequence Packing(序列打包)、长文本 4D 注意力掩码、多版本 Transformers 兼容性以及高性能推理后端集成等复杂工程问题.
"""

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # 1. 基础组件初始化
    tokenizer_module = load_tokenizer(model_args)
    """
    print(tokenizer_module)
    {'tokenizer': Qwen2TokenizerFast(), 'processor': None}
    """
    tokenizer = tokenizer_module["tokenizer"]

    # 模板适配与 Tokenizer 修正 (Template Alignment)
    # [为什么要这么写]: LLM 极度依赖 Chat Template(如 <|im_start|>).
    # [解决的问题]: 如果 Tokenizer 缺少模板定义的特殊 Token, 模型会因为无法识别边界而产生幻觉或复读.
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 数据集预处理
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    """
    print(dataset_module)
    {'train_dataset': Dataset({
        features: ['input_ids', 'attention_mask', 'labels', 'images', 'videos', 'audios'],
        num_rows: 1090
    })}
    """

    # 模型加载 (支持 LoRA, Full-tuning, 量化加载等)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 2. 量化模型推理补丁 (Quantization Inference Hack)
    # [为什么要这么写]: 在 PEFT 中, 如果模型被标记为量化模型, 某些版本会强制要求在训练模式下运行.
    # [解决的问题]: 在纯推理/评测场景下, 由于没有加载训练配置, 模型可能会报错. 这里通过设置私有属性来欺骗 PEFT
    # 框架, 确保量化模型能够顺利进行 Batch 推理.
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # 3. 高性能数据整理器 (Efficient Data Collator with Packing Support)
    # [为什么要这么写]: 使用 SFTDataCollatorWith4DAttentionMask.
    # [解决的问题]: 解决"序列打包(Packing)"中的污染问题.
    # 当我们将多个短样本打包进一个 4096 长度的块时, 样本 A 的 Token 不应该注意到样本 B.
    # 4D 注意力掩码(或块对角掩码)能够实现逻辑上的完全隔离, 同时享受物理上的极致算力利用率.
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        # 对齐到 8 的倍数以触发 NVIDIA Tensor Cores 的硬件加速
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # 4. 指标计算模块 (Metrics Configuration)
    # Metric utils
    metric_module = {}
    # KTransformers (KT) 后端兼容性检查
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    # 根据任务类型选择评估指标: 生成式任务用 Similarity (BLEU/ROUGE), 判别式/分类任务用 Accuracy
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        # 处理 logits 以便评估(如计算预测 Top-1 Token 的准确率)
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # 5. 复杂的生成参数适配 (Generation Config & EOS Logic)
    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # [为什么要这么写]: 处理 Transformers v4.58+ 之后对特殊 Token 存储方式的变更.
    # [解决的问题]: 很多模型(如 Llama-3)有多个结束符(EOS, EOT). 如果只传一个 ID,
    # 模型生成时会停不下来. 这里动态收集所有可能的 EOS IDs, 确保生成能准时停止.
    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # 6. Trainer 初始化 (后端分发)
    # Initialize our Trainer
    if model_args.use_kt:
        # 适配 KTransformers 后端, 用于特定硬件上的推理加速微调
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        # 使用 LLaMA-Factory 自定义的 Seq2SeqTrainer
        # [解决的问题]: 原生 Trainer 不支持 SFT 打包模式下的 4D Mask 传递.
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # 7. 训练逻辑执行
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

        # 吞吐量指标计算 (TPS: Tokens Per Second)
        # [为什么要这么写]: 计算"有效"训练速度.
        # [解决的问题]: 由于 Packing 的存在, 传统的样本/秒已经无法反映真实算力利用率,
        # 用户需要知道每秒处理了多少实际的 Token.
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # 绘制 Loss 曲线 (主进程执行)
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    # 8. 生成评估逻辑 (Padding Side Adjustment)
    # [为什么要这么写]: 推理时强制设置为 left-padding.
    # [解决的问题]: 这是 LLM 批处理推理的硬性要求. 如果是 right-padding,
    # 所有的生成 Token 会因为注意力掩码的问题导致计算错误.
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # 执行 Evaluation
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

     # 执行 Prediction (生成预测结果)
    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        # 将生成的文本解码并保存到 jsonl, 方便人工对齐效果进行审核
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # 9. 自动化生态集成 (Hugging Face Hub)
    # 自动生成 README/Model Card, 记录微调所用的数据集和超参, 符合开源社区规范.
    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

"""
资深专家视角的总结:

工程上的严谨性: 代码中对 eos_token_id 的多版本处理和 padding_side 的切换, 解决了 LLM 训练到推理切换时最常见的两个"坑"(死循环生成和注意力崩溃).

效率上的追求: 通过 SFTDataCollatorWith4DAttentionMask 实现 Packing, 这使得 LLaMA-Factory 在 SFT 任务上的训练速度通常能比未经优化的原生脚本快 200% 以上.

对硬件加速器的适配: 集成了 KTransformers 和 Flash Attention 的逻辑, 体现了框架在高性能计算领域的深度.

希望这些注释能让你深刻理解 LLaMA-Factory 在 SFT 流程上的卓越设计!
"""

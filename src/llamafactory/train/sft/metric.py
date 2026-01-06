# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Compute the token with the largest likelihood to reduce memory footprint.

    [为什么要这么写]: 在评估阶段将原始 Logits 直接转换为预测 Token ID.
    [解决的问题]: 显存溢出(OOM). LLM 的词表(Vocab Size)通常高达 32k 到 128k.
    如果保留原始 Logits(float32), 显存占用是 (batch * seq_len * vocab_size * 4 bytes).
    在一个 1024 长度的序列上, 这可能消耗数 GB 显存.
    通过在此处执行 argmax, 我们只保留最大概率的索引(int64), 显存占用瞬间降低数万倍.
    """

    # 处理 Hugging Face 模型输出的多样性
    if isinstance(logits, (list, tuple)):
        # 情况 A: 标准输出格式 (loss, logits, ...)
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        # 情况 B: MoE (混合专家) 模型输出.
        # MoE 模型通常在输出元组中包含辅助损失(aux_loss), 此时真正的 logits 往往在第二个位置.
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    # 在返回给 Trainer 前执行 argmax, 这是降低分布式评测通讯开销和内存压力的核心操作
    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""
    Compute accuracy and support `batch_eval_metrics`.

    计算 Next-Token Prediction 的准确率. 支持按 Batch 累积评估指标.
    """

    def _dump(self) -> Optional[dict[str, float]]:
        # [为什么要这么写]: 解耦"数据收集"与"指标汇总".
        # [解决的问题]: 在分布式评估中, Trainer 会多次调用计算函数.
        # 我们需要先缓存每个 batch 的结果, 最后统一计算均值, 避免频繁的浮点数运算误差.
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        # 状态重置, 确保下一个评估周期(如 eval_loop 的下一轮)从零开始
        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        # 将 Tensor 转换为 Numpy 以便进行非梯度运算, 提高 CPU 处理效率
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            # [关键逻辑]: 对齐错位 (Shift Logic)
            # [为什么要这么写]: 在因果语言模型(Causal LM)中, 第 t 个 token 的输出是为了预测第 t+1 个 token.
            # 因此, 预测值 preds[t] 需要与标签 labels[t+1] 进行对比.
            pred, label = preds[i, :-1], labels[i, 1:]

            # 过滤掉 Prompt 部分
            # [解决的问题]: 我们只关注模型对"回答"部分的预测准确率.
            # 标签中值为 IGNORE_INDEX (-100) 的部分是 Prompt, 不应计入 Accuracy.
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        # 如果 compute_result 为 True, 说明本轮 batch 评估结束, 汇总输出结果
        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.

    计算文本相似度指标(ROUGE, BLEU). 常用于 SFT 后的生成质量评估.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        # 初始化生成式任务的核心四大指标
        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # 预处理: 将屏蔽位替换为 Pad ID, 否则 Tokenizer 解码会报错
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        # 文本还原: 指标计算是在文本层面而非 ID 层面
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            # [关键点]: 中文分词 (Chinese Segmentation)
            # [为什么要这么写]: ROUGE 和 BLEU 最初是为英文设计的(基于空格分词).
            # [解决的问题]: 中文没有自然空格. 如果不分词, ROUGE 会把整个句子当成一个 token, 导致得分极低.
            # 使用 jieba 进行预处理, 使指标能正确反映中文语境下的词级匹配度.
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            # 异常处理: 防止空输入导致指标库崩溃
            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                # 传入带空格的字符串以模拟英文格式, 适配 rouge 库
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            # 提取 F1 分数(通常比 recall/precision 更能反映综合性能)
            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            # 计算 BLEU-4
            # 使用 method3 平滑函数防止短句子导致的分数阶跃(即由于 N-gram 匹配失败导致分数骤降为 0)
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()

"""
深度架构解析(高级研究员视角):

关于 eval_logit_processor 的 argmax 策略:
这是大模型评测的"工业级"做法. 在原生 Transformers 的 Trainer 中, 如果不做这个处理, 多卡同步评估时会产生海量的 NCCL 通讯数据, 导致评测比训练还慢. 通过提前 argmax, 我们将数据传输量压缩了数万倍.

准确率计算中的 Shift 逻辑:
这是新入行的研究员最容易写错的地方. 记住: labels[0] 是在输入端, 模型预测出的第一个 token 是对应 labels[1]. 所以 pred[:-1] 对比 label[1:] 是 CLM(因果建模)评估的真理.

生成相似度的中文适配:
LLaMA-Factory 作为一个对中文社区极其友好的项目, 在 ComputeSimilarity 中引入 jieba 是非常专业的. 如果不进行分词, 计算出的 ROUGE-L 指标在学术上是不严谨的, 因为中文的语义单元是"词"而非"字".

希望这些解析能帮助你更深入地理解 LLaMA-Factory 的工程细节!
"""

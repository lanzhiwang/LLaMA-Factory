# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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
from itertools import chain
from typing import Any

from .processor_utils import DatasetProcessor

"""
预训练(或增量预训练)的任务目标是 Causal Language Modeling (CLM), 即因果语言建模.
这段代码的设计核心在于如何最高效地组织文本数据, 以确保 GPU 算力不被浪费在无意义的 PAD Token 上.
"""

@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    """
    预训练数据处理器.
    [核心目标]: 将纯文本数据转化为自回归语言模型所需的输入流.
    [解决的问题]: 预训练通常涉及海量文本, 如何处理不同长度的文档以及如何优化计算效率是关键.
    """

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        # 1. 动态 EOS 处理 (Dynamic EOS Adaptation)
        # [为什么要这么写]: Llama3 等模型在预训练时对结束符有特殊要求(使用 <|end_of_text|> 而非默认的 <|eot_id|>).
        # [解决的问题]: 确保不同架构的模型在处理文档边界时的一致性, 防止模型学习到错误的停顿信号.
        """

        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token

        # 将多个文档内容与 EOS 拼接, 形成独立的文本块
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

        # 2. 模式分支: 非打包模式 vs 打包模式 (Non-Packing vs Packing)
        if not self.data_args.packing:
            # --- 非打包模式 (Standard Mode) ---
            # [解决的问题]: 处理简单的、短篇幅的文本, 每条原始数据对应一个训练样本.
            if getattr(self.tokenizer, "add_bos_token", False):
                # 显式添加起始符, 帮助模型识别序列开始
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            # 直接进行分词、截断. 缺点是如果文本短于 cutoff_len, 会产生大量 Padding, 降低吞吐量.
            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            # --- 打包模式 (Sequence Packing / Grouped Texts) ---
            # [为什么要这么写]: 这是预训练性能优化的核心. 将所有 Token 连成一条长链, 然后按固定长度(cutoff_len)切割.
            # [解决的问题]: 极大提升训练吞吐量(Throughput).
            # 在预训练中, 文档往往很短. 如果不打包, GPU 显存中会充斥着 50%-80% 的 Padding Token.
            # Packing 确保模型看到的每一个 Token 都是"有效 Token", 让算力利用率接近 100%.

            # 第一步: 全量分词, 暂不截断
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)

            # 第二步: 打平拼接 (Flattening)
            # 使用 chain(*...) 将所有文档的 Token 序列首尾相连, 形成一个一维长序列.
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}

            # 第三步: 对齐切块 (Chunking)
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len

            # 丢弃末尾不足一个 block_size 的碎料, 保证所有 batch 形状整齐
            total_length = (total_length // block_size) * block_size

            # 按照 block_size 进行等长度切割
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }

            # 第四步: 起始符注入 (BOS Injection)
            # [解决的问题]: 在长链切割后, 后续的块(Chunk)开头可能不再是文档起始.
            # 强制在每个 block 的第一个位置放上 BOS_ID, 能让模型在处理每一个计算块时都有明确的自回归起点.
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id

        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        调试打印.
        [解决的问题]: 验证数据流是否被正确处理.
        对于预训练, 重点检查文本之间是否通过 EOS 连接, 以及 BOS 是否出现在块首.
        """
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

"""
高级研究员视角的架构解析:

为什么预训练一定要做 Packing?
在 SFT(指令微调)中, 由于存在 Prompt 和 Response 的结构, 我们通常不轻易打破序列边界(或使用复杂的多序列 Mask). 但在预训练中, 目标只是预测下一个词. 将几百个小文档打包进一个 4096 或 8192 长度的序列中, 可以减少计算时的核函数启动次数, 并最大化 NVIDIA 张量核心(Tensor Cores)的利用率.

Llama 3 的适配细节:
代码中专门判断了 template == "llama3". 这是因为 Llama 3 官方在预训练阶段使用了非常特定的特殊 Token 处理方式. 作为一个高级开发工程师, 在入口处处理这类特殊情况(Hardcoding minimally at the edge)能避免底层逻辑变得过于臃肿.

BOS Token 的处理技巧:
在 Packing 模式下, result["input_ids"][i][0] = self.tokenizer.bos_token_id 是一种"强制注入". 这在某些论文中被证明有助于稳定长文本训练的 Loss 波动, 因为它为每一段独立的计算图提供了一个统一的起始隐状态.

这份代码展示了 LLaMA-Factory 如何在保证模型适配性的同时, 追求工业级的训练效率. 希望对你有帮助!
"""

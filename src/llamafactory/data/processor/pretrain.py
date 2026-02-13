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


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    """
    PretrainDatasetProcessor 是我们框架中专门用于增量预训练(Incremental Pre-training)的数据处理模块.
    它的核心逻辑围绕着 Causal Language Modeling (CLM) 展开.
    与指令微调(SFT)不同, 预训练不区分"指令"和"回答", 而是将所有文本视为一个流.
    这段代码最精彩的地方在于它实现了 Packing(打包) 策略, 能极大地提升训练效率.

    预训练数据处理器: 将原始文本转化为自回归语言模型所需的输入格式.
    """

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        """
        print(examples)
        {
            "_prompt": [
                [{"content": "Hello world "}],
                [{"content": "LLM training is fun "}],
                [{"content": "Data processing "}],
            ]
        }
        """

        # 1. 确定 EOS Token
        # [为什么要这么写]: Llama 3 官方在预训练时使用了特定的 <|end_of_text|> 而非 SFT 用的 <|eot_id|>.
        # [解决的问题]: 保证不同模型在处理文档边界时符合其预训练时的分布.
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token
        """
        print(eos_token)
        </s>
        """

        # 将每一条原始文本末尾加上 EOS, 表示文档结束
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]
        """
        print(text_examples)
        ['Hello world </s>', 'LLM training is fun </s>', 'Data processing </s>']
        """

        # 分支 A: 不开启 Packing(常规截断模式)
        if not self.data_args.packing:
            # 如果分词器配置了自动添加起始符(如 Llama 的 <s>)
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]
            """
            print(text_examples)
            ['<s>Hello world </s>', '<s>LLM training is fun </s>', '<s>Data processing </s>']
            """

            # 直接进行分词、截断. 如果文本短于 cutoff_len, 会产生大量 Padding, 效率较低.
            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
            """
            print(result)
            {'input_ids': [[8, 5, 4], [6, 8, 2, 3, 4], [7, 10, 4]]}
            """

        # 分支 B: 开启 Packing(打包模式, 预训练推荐使用)
        else:
            # [为什么要这么写]: 预训练数据往往由大量短文本组成.
            # [解决的问题]: 解决"计算浪费"问题. 如果不打包, 一个 4096 长度的窗口若只放 500 token, 剩下的 3500 都是 Padding.
            # 开启 Packing 后, 我们将所有文本连成一条长链, 然后按 cutoff_len 强行切块.

            # 第一步: 先对所有文本进行完整分词(不截断)
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            """
            print(tokenized_examples)
            {'input_ids': [[5, 5, 4], [3, 8, 2, 3, 4], [4, 10, 4]]}
            """

            # 第二步: 将整个 Batch 里的 Token 拍平拼接成一条长链 (Concatenation)
            # 使用 itertools.chain 高效合并多个 list
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            """
            print(concatenated_examples)
            {'input_ids': [5, 5, 4, 3, 8, 2, 3, 4, 4, 10, 4]}

            print(tokenized_examples.keys())
            dict_keys(['input_ids'])

            print(tokenized_examples['input_ids'])
            [[5, 5, 4], [3, 8, 2, 3, 4], [4, 10, 4]]

            print(chain(*tokenized_examples['input_ids']))
            <itertools.chain object at 0x7f29b87bce80>
            """

            # 第三步: 计算总长度, 并向下取整到 block_size 的倍数
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            """
            print(concatenated_examples.keys())
            dict_keys(['input_ids'])
            print(list(concatenated_examples.keys()))
            ['input_ids']
            print(list(concatenated_examples.keys())[0])
            input_ids
            print(concatenated_examples[list(concatenated_examples.keys())[0]])
            [5, 5, 4, 3, 8, 2, 3, 4, 4, 10, 4]
            print(total_length)
            11
            """
            block_size = self.data_args.cutoff_len
            """
            print(block_size)
            5
            """

            # 丢弃掉最后不足一个 block 的剩余 token, 保证所有数据形状都是 [N, cutoff_len]
            total_length = (total_length // block_size) * block_size
            """
            print(total_length)
            10
            """

            # 第四步: 将长链切分为一个个长度为 block_size 的数据块
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            """
            print(result)
            {'input_ids': [[5, 5, 4, 3, 8], [2, 3, 4, 4, 10]]}
            """

            # 第五步: 强制注入起始符 (BOS Injection)
            # [为什么要这么写]: 由于长链被切断, 后续的块开头可能在原文档的中间.
            # [解决的问题]: 手动在每个块的第一个位置放入 BOS_ID, 能引导模型正确开始自回归计算.
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id
            """
            print(result)
            {'input_ids': [[1, 5, 4, 3, 8], [1, 3, 4, 4, 10]]}
            """

        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

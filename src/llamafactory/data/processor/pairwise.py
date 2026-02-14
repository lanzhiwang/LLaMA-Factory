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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):
    """
    PairwiseDatasetProcessor 类是偏好对齐(Preference Alignment) 阶段(如 DPO、RM、ORPO)的数据处理核心.
    它的核心使命是: 将"一问两答"的原始数据, 转化为模型可以同时对比的成对 Token 张量.

    成对数据处理器.
    解决的问题: 在 RLHF 或 DPO 中, 我们需要模型学习"为什么回答 A 比回答 B 好".
    该类负责将 (Prompt, Chosen_Response, Rejected_Response) 转化为训练所需的格式.
    """

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        单条数据编码逻辑.
        为什么要单独写这个函数:
        1. 保证 Prompt 部分在 Chosen 和 Rejected 两条序列中完全一致(这对对比学习至关重要).
        2. 统一处理多模态信息插入位置.

        print(prompt)
        [{'role': 'user', 'content': '法国的首都是哪里?'}]
        print(response)
        [{'role': 'assistant', 'content': '巴黎.'}, {'role': 'assistant', 'content': '伦敦.'}]
        print(system)
        None
        print(tools)
        None
        print(images)
        []
        print(videos)
        []
        print(audios)
        []
        """
        # 1. 构造两条完整的对话流: Prompt + Chosen 和 Prompt + Rejected
        # 解决的问题: 多模态内容(如图片占位符)需要根据 Prompt 长度和位置进行预处理.
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        """
        print(chosen_messages)
        [{'role': 'user', 'content': '法国的首都是哪里?'}, {'role': 'assistant', 'content': '巴黎.'}]
        """

        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        """
        print(rejected_messages)
        [{'role': 'user', 'content': '法国的首都是哪里?'}, {'role': 'assistant', 'content': '伦敦.'}]
        """

        # 2. 调用模板进行 Tokenize 编码
        # 为什么要分别取 prompt_ids 和 chosen_ids:
        # 在 DPO 或 RM 中, 我们需要精确知道 Prompt 的结束位置, 以便在计算 Loss 时屏蔽掉 Prompt 部分.
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        """
        print(prompt_ids)
        [101, 102]
        print(chosen_ids)
        [201, 202]
        """
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)
        """
        print(rejected_ids)
        [301, 302]
        """

        # 3. 结束符(EOS)处理
        # 解决的问题: 确保模型学会在回答结束时停止. 如果开启了 efficient_eos, 手动在每个回复末尾添加结束符.
        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]
        """
        print(chosen_ids)
        [201, 202, 99]
        print(rejected_ids)
        [301, 302, 99]
        """

        # 4. 多模态 Token 占位符处理
        # 解决的问题: 将文本中的图片/视频占位符映射为特定的 ID.
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        """
        print(prompt_ids)
        [101, 102]
        """

        # 5. 智能截断策略 (Critical!)
        # 为什么要这么写:
        # - 使用 max(len(chosen_ids), len(rejected_ids)) 作为 Response 的参考长度.
        # - 核心逻辑: 回复(Response)包含偏好信息, 比问题(Prompt)更重要.
        # - 解决的问题: 当总长超过 cutoff_len 时, infer_seqlen 确保 Prompt 部分在两个对子中被截断到相同的长度,
        #   避免因为 Prompt 长度不同导致模型计算偏好损失时出现偏差.
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        """
        print(source_len)
        2
        print(target_len)
        3
        """
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        # 6. 构造最终输入与标签 (Label Masking)
        # 为什么要用 IGNORE_INDEX (-100):
        # 这是 PyTorch CrossEntropyLoss 的默认忽略值.
        # 解决的问题: 模型训练时只对 Response(回答)产生的 Loss 进行优化, 不为 Prompt(已知的问题)负责.
        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        批量预处理入口.
        解决的问题: 将 HuggingFace Datasets 读取的原始字典批量映射为训练用的张量列.

        print(examples)
        {
            "_prompt": [[{"role": "user", "content": "法国的首都是哪里?"}]],
            "_response": [
                [
                    {"role": "assistant", "content": "巴黎."},  # Chosen
                    {"role": "assistant", "content": "伦敦."},  # Rejected
                ]
            ],
            "_system": [None],
            "_tools": [None],
            "_images": [None],
            "_videos": [None],
            "_audios": [None],
        }
        """
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        """
        print(model_inputs)
        defaultdict(
            <class 'list'>,
            {
                'chosen_input_ids': [[101, 102, 201, 202, 99]],
                'chosen_attention_mask': [[1, 1, 1, 1, 1]],
                'chosen_labels': [[-100, -100, 201, 202, 99]],
                'rejected_input_ids': [[101, 102, 301, 302, 99]],
                'rejected_attention_mask': [[1, 1, 1, 1, 1]],
                'rejected_labels': [[-100, -100, 301, 302, 99]],
                'images': [None],
                'videos': [None],
                'audios': [None]
            }
        )
        """

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")

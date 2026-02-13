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


class FeedbackDatasetProcessor(DatasetProcessor):
    """
    这个类是 KTO (Kahneman-Tversky Optimization) 算法的核心数据处理模块.
    KTO 是一种人类偏好对齐算法, 与 DPO(对比"好"与"坏"的一对数据)不同, KTO 只需要单条数据并标注"满意"或"不满意".

    KTO (反馈对齐) 任务的数据处理器.
    功能: 处理单条反馈数据, 并利用 Batch 内错位偏移技术构造 KL 散度参考样本.

    KTO 的核心逻辑与 DPO 不同:
    DPO (Direct Preference Optimization) 需要成对的数据(一个 Chosen, 一个 Rejected).
    KTO 只需要单条数据, 并给它打上"优" (Desirable) 或"劣" (Undesirable) 的标签.
    此外, KTO 算法在数学上要求估计一个参考分布的 KL 散度. 为了在不加载额外参考模型的情况下实现这一点, LLaMA-Factory 采用了Batch 内负采样(错位偏移)的技巧.
    """

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        kl_response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int], bool]:
        """
        [功能]: 编码单条 KTO 样本.
        [解决的问题]: KTO 需要区分"期望"与"非期望"样本, 同时需要一组不匹配的 prompt-response 来计算 KL 惩罚, 防止模型坍缩.

        print(prompt)
        [{'role': 'user', 'content': '你好'}]
        print(response)
        [{'role': 'assistant', 'content': '你好呀'}, {'role': 'assistant', 'content': ''}]
        print(kl_response)
        [{'role': 'assistant', 'content': ''}, {'role': 'assistant', 'content': '不知道'}]
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

        # 1. 判定 KTO 标签 (Desirable/Undesirable)
        # [为什么这么写]: KTO 数据格式中, response 列表通常包含期望和非期望两个位置.
        # 如果第一个 response 有内容, 则标记为 True(用户满意的); 否则取第二个作为不满意的样本.
        if response[0]["content"]:  # desired example
            kto_tag = True
            messages = prompt + [response[0]]
        else:  # undesired example
            kto_tag = False
            messages = prompt + [response[1]]
        """
        print(kto_tag)
        True
        print(messages)
        [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好呀'}]
        """

        # 2. 构造 KL 参考消息
        # [为什么要这么写]: KTO 算法需要计算模型在"不匹配"回答上的概率来估计背景分布.
        if kl_response[0]["content"]:
            kl_messages = prompt + [kl_response[0]]
        else:
            kl_messages = prompt + [kl_response[1]]
        """
        print(kl_messages)
        [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '不知道'}]
        """

        # 3. 多模态处理与 Token 编码
        # 调用插件处理多模态占位符, 并使用模板编码成 Token ID
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        kl_messages = self.template.mm_plugin.process_messages(kl_messages, images, videos, audios, self.processor)

        # 分离 Prompt 和 Response 的 ID
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        kl_prompt_ids, kl_response_ids = self.template.encode_oneturn(self.tokenizer, kl_messages, system, tools)
        """
        print(messages)
        [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好呀'}]
        print(kl_messages)
        [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '不知道'}]
        print(prompt_ids)
        [1, 2, 3]
        print(response_ids)
        [4, 5]
        print(kl_prompt_ids)
        [1, 2, 3]
        print(kl_response_ids)
        [4, 5]
        """

        # 4. 停止符处理 (Efficient EOS)
        if self.template.efficient_eos:
            response_ids += [self.tokenizer.eos_token_id]
            kl_response_ids += [self.tokenizer.eos_token_id]
        """
        print(response_ids)
        [4, 5, 99]
        print(kl_response_ids)
        [4, 5, 99]
        """

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        kl_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            kl_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        """
        print(prompt_ids)
        [1, 2, 3]
        print(kl_prompt_ids)
        [1, 2, 3]
        """

        # 5. 智能截断 (Smart Truncation)
        # [解决的问题]: 当 Prompt+Response 超过 cutoff_len 时, 调用 infer_seqlen 优先保证 Response 完整.
        source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        response_ids = response_ids[:target_len]

        # 对 KL 样本执行同样的截断逻辑
        kl_source_len, kl_target_len = infer_seqlen(
            len(kl_prompt_ids), len(kl_response_ids), self.data_args.cutoff_len
        )
        kl_prompt_ids = kl_prompt_ids[:kl_source_len]
        kl_response_ids = kl_response_ids[:kl_target_len]
        """
        print(source_len, target_len)
        3 3
        print(prompt_ids)
        [1, 2, 3]
        print(response_ids)
        [4, 5, 99]
        print(kl_source_len, kl_target_len)
        3 3
        print(kl_prompt_ids)
        [1, 2, 3]
        print(kl_response_ids)
        [4, 5, 99]
        """

        # 6. 构造训练输入与 Label
        # [为什么要这么写]: Labels 中 Prompt 部分设为 IGNORE_INDEX (-100), 让模型只对 Response 的预测产生梯度.
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * source_len + response_ids
        kl_input_ids = kl_prompt_ids + kl_response_ids
        kl_labels = [IGNORE_INDEX] * kl_source_len + kl_response_ids
        """
        print(input_ids)
        [1, 2, 3, 4, 5, 99]
        print(labels)
        [-100, -100, -100, 4, 5, 99]
        print(kl_input_ids)
        [1, 2, 3, 4, 5, 99]
        print(kl_labels)
        [-100, -100, -100, 4, 5, 99]
        """

        return input_ids, labels, kl_input_ids, kl_labels, kto_tag

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of completions.
        """
        [核心逻辑]: 批量处理数据集.
        [解决的问题]: 如何高效构造"不匹配"的 KL 样本.
        [实现黑科技]: kl_response = [examples["_response"][-1]] + examples["_response"][:-1]
        通过将 Batch 内的 Response 整体向后偏移一位, 使得当前行的 Prompt 对应上一行的 Response.
        这样无需额外数据, 就在 Batch 内部构造了完美的不匹配对(Negative pairs for KL).

        print(examples)
        examples = {
            "_prompt": [
                [{"role": "user", "content": "你好"}],
                [{"role": "user", "content": "天气"}],
            ],
            "_response": [
                [
                    {"role": "assistant", "content": "你好呀"},
                    {"role": "assistant", "content": ""},
                ],
                [
                    {"role": "assistant", "content": ""},
                    {"role": "assistant", "content": "不知道"},
                ],
            ],
            "_system": [None, None],
            "_tools": [None, None],
            "_images": [None, None],
            "_videos": [None, None],
            "_audios": [None, None],
        }
        """
        kl_response = [examples["_response"][-1]] + examples["_response"][:-1]
        """
        print(kl_response)
        [
            [
                {'role': 'assistant', 'content': ''},
                {'role': 'assistant', 'content': '不知道'}
            ],
            [
                {'role': 'assistant', 'content': '你好呀'},
                {'role': 'assistant', 'content': ''}
            ]
        ]
        """

        model_inputs = defaultdict(list)
        """
        print(model_inputs)
        defaultdict(<class 'list'>, {})
        """

        for i in range(len(examples["_prompt"])):
            """
            print(examples["_prompt"][0])
            [{'role': 'user', 'content': '你好'}]
            print(len(examples["_prompt"][0]))
            1

            print(examples["_response"][0])
            [{'role': 'assistant', 'content': '你好呀'}, {'role': 'assistant', 'content': ''}]
            print(len(examples["_response"][0]))
            2
            """
            # 过滤非法数据
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, kl_input_ids, kl_labels, kto_tag = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                kl_response=kl_response[i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            # 收集 KTO 训练所需的全部字段
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            # KL 散度专用字段
            model_inputs["kl_input_ids"].append(kl_input_ids)
            model_inputs["kl_attention_mask"].append([1] * len(kl_input_ids))
            model_inputs["kl_labels"].append(kl_labels)
            # 标记该样本是"满意"还是"不满意"
            model_inputs["kto_tags"].append(kto_tag)
            # 多模态字段收集
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        """
        print(model_inputs)
        defaultdict(
            <class 'list'>,
            {
                'input_ids': [
                    [1, 2, 3, 4, 5, 99],
                    [1, 2, 3, 4, 5, 99]
                ],
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1]
                ],
                'labels': [
                    [-100, -100, -100, 4, 5, 99],
                    [-100, -100, -100, 4, 5, 99]
                ],
                'kl_input_ids': [
                    [1, 2, 3, 4, 5, 99],
                    [1, 2, 3, 4, 5, 99]
                ],
                'kl_attention_mask': [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1]
                ],
                'kl_labels': [
                    [-100, -100, -100, 4, 5, 99],
                    [-100, -100, -100, 4, 5, 99]
                ],
                'kto_tags': [True, False],
                'images': [None, None],
                'videos': [None, None],
                'audios': [None, None]
            }
        )
        """

        desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
        undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
        """
        print(desirable_num)
        1
        print(undesirable_num)
        1
        """
        if desirable_num == 0 or undesirable_num == 0:
            logger.warning_rank0("Your dataset only has one preference type.")

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        [调试功能]: 将编码后的 ID 还原为文本, 检查标签掩码 (IGNORE_INDEX) 是否正确.
        """
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

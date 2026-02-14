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
from ..data_utils import Role
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class UnsupervisedDatasetProcessor(DatasetProcessor):
    """
    在 LLaMA-Factory 的设计哲学中, 这个处理器主要用于无监督预训练(Pre-training)、评估任务(Evaluation)或某些特定的 PPL(困惑度)计算场景.
    它的特点是不会像 SFT(有监督微调)那样对 Prompt(提示词)部分进行 Loss 遮蔽(Masking), 而是将输入视为一个连续的 Token 流.

    无监督数据处理器.
    解决的问题: 处理非指令对齐的数据流, 或者在推理/评估阶段将输入和输出转化为模型可识别的张量.
    特点: labels 与 input_ids 通常是对齐的, 用于计算整个序列的似然概率.
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
    ) -> tuple[list[int], list[int]]:
        """
        print(prompt)
        [{'role': 'user', 'content': '你好, 你是谁?'}]
        print(response)
        [{'role': 'assistant', 'content': '我是 LLaMA-Factory 助手.'}]
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

        # 1. 消息组合逻辑
        # [为什么要这么写]: 确保对话流的完整性.
        # 如果提供了 response(通常长度为1), 则拼接;
        # 否则手动补一个空的 Assistant 角色, 这在推理预测阶段非常重要, 用于引导模型开始生成.
        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

        # 2. 多模态消息预处理
        # [解决的问题]: 将图片/视频等占位符插入到文本消息的正确位置.
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)

        # 3. 模板化编码 (Tokenization)
        # [为什么要这么写]: 调用 template.encode_oneturn 将角色信息(User/Assistant)转化为模型特定的特殊 Token(如 <|im_start|>).
        # 返回的 input_ids 是完整的输入, labels 在此阶段通常与 input_ids 对应.
        input_ids, labels = self.template.encode_oneturn(self.tokenizer, messages, system, tools)

        # 4. 显式停止符处理
        # [解决的问题]: 在某些"高效 EOS"设置下, 模型需要显式地学习在结尾处产生停止符, 否则生成会无休止进行.
        if self.template.efficient_eos:
            labels += [self.tokenizer.eos_token_id]

        # 5. 多模态 Token 修正
        # 将文本占位符转化为模型视觉编码器(Vision Encoder)对应的特殊 ID 序列.
        input_ids, _ = self.template.mm_plugin.process_token_ids(
            input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        # 6. 智能截断逻辑
        # [解决的问题]: LLM 显存有限. 当 Prompt+Response 超过 cutoff_len 时, 调用智能截断函数.
        # 它会动态平衡保留多少 Prompt 和多少 Response, 而不是简单地从末尾切断(导致答案丢失).
        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        print(examples)
        {
            "_prompt": [
                [{"role": "user", "content": "你好, 你是谁?"}],
                [{"role": "user", "content": "1+1等于几?"}],
            ],
            "_response": [
                [{"role": "assistant", "content": "我是 LLaMA-Factory 助手."}],
                [{"role": "assistant", "content": "等于2."}],
            ],
            "_system": [None, None],
            "_tools": [None, None],
            "_images": [None, None],
            "_videos": [None, None],
            "_audios": [None, None],
        }
        """

        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        """
        print(model_inputs)
        defaultdict(
            <class 'list'>,
            {
                'input_ids': [
                    [20320, 22909, 44, 32, 20320, 26159, 35841, 63, 25105, 26159, 32, 76, 76, 97, 77, 65, 45, 70, 97, 99, 116, 111, 114, 121, 32],
                    [49, 43, 49, 31561, 20110, 20960, 63, 31561, 20110, 50, 46, 0]
                ],
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ],
                'labels': [
                    [20320, 22909, 44, 32, 20320, 26159, 35841, 63, 25105, 26159, 32, 76, 76, 97, 77, 65, 45, 70, 97, 99, 116, 111, 114, 121, 32],
                    [49, 43, 49, 31561, 20110, 20960, 63, 31561, 20110, 50, 46, 0]
                ],
                'images': [None, None],
                'videos': [None, None],
                'audios': [None, None]
            }
        )
        """

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))

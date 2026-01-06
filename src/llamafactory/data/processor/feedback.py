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

"""
这段代码定义了 FeedbackDatasetProcessor 类, 它专门用于处理 KTO (Kahneman-Tversky Optimization) 这种偏好对齐算法的数据.
KTO 与常见的 DPO 不同, DPO 需要"成对"数据(Chosen vs Rejected), 而 KTO 只需要单条数据并打上"满意"或"不满意"的标签.
这大大降低了数据标注的门槛. 这段代码的核心在于: 如何高效处理 KTO 的二元标签, 并构建用于 KL 散度约束的对比样本.
"""

class FeedbackDatasetProcessor(DatasetProcessor):
    """
    KTO (Feedback) 任务的数据处理器.
    [核心逻辑]: KTO 算法通过人类对回答的二元反馈(Desirable/Undesirable)来优化模型.
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
        对单条数据进行编码, 生成模型输入、标签以及 KTO 专属标签.
        """

        # 1. KTO 标签判定逻辑 (KTO Tagging)
        # [为什么要这么写]: KTO 数据集格式通常在 response 列表中区分满意与否.
        # [解决的问题]: 如果 response[0] 有内容, 视为期望回答(Desirable), kto_tag = True;
        # 否则视为不期望回答(Undesirable), 获取 response[1].
        # 这种逻辑兼容了"二选一"数据作为 KTO 输入的灵活性.
        # 期望的例子
        if response[0]["content"]:  # desired example
            kto_tag = True
            messages = prompt + [response[0]]

        # 不期望的例子
        else:  # undesired example
            kto_tag = False
            messages = prompt + [response[1]]

        # 2. 构建 KL 散度参考样本 (KL Reference Sample)
        # [为什么要这么写]: KTO 损失函数包含一个 KL 惩罚项, 需要一个参考分布.
        # [解决的问题]: 在实际实现中, 我们通过 kl_response(通常是打乱后的回复)来估计参考分布的期望.
        if kl_response[0]["content"]:
            kl_messages = prompt + [kl_response[0]]
        else:
            kl_messages = prompt + [kl_response[1]]

        # 3. 多模态与模板编码 (Multimodal & Template Encoding)
        # [解决的问题]: 调用 mm_plugin 处理图像/视频/音频, 并应用 Chat Template.
        # 确保 prompt 和 response 被正确拼接, 并添加 BOS/EOS 等特殊 Token.
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        kl_messages = self.template.mm_plugin.process_messages(kl_messages, images, videos, audios, self.processor)

        # 得到 input_ids 并分离出 prompt 和 response 的长度
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        kl_prompt_ids, kl_response_ids = self.template.encode_oneturn(self.tokenizer, kl_messages, system, tools)

        # 4. 显式 EOS 处理 (Efficient EOS Handling)
        # [解决的问题]: 如果模板配置了 efficient_eos, 手动在 response 末尾添加 EOS.
        # 这确保了模型能学会正确结束预测, 防止推理时的"无限复读".
        if self.template.efficient_eos:
            response_ids += [self.tokenizer.eos_token_id]
            kl_response_ids += [self.tokenizer.eos_token_id]

        # 再次处理多模态 Token(如插入 <image> 占位符)
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        kl_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            kl_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        # 5. 智能截断 (Smart Truncation)
        # [为什么要这么写]: 调用 infer_seqlen.
        # [解决的问题]: 当 Prompt + Response 超过 cutoff_len 时, 不能简单从末尾砍掉.
        # 该算法会优先保留 Response, 截断 Prompt, 保证训练标签的完整性.
        source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        response_ids = response_ids[:target_len]
        kl_source_len, kl_target_len = infer_seqlen(
            len(kl_prompt_ids), len(kl_response_ids), self.data_args.cutoff_len
        )
        kl_prompt_ids = kl_prompt_ids[:kl_source_len]
        kl_response_ids = kl_response_ids[:kl_target_len]

        # 6. 构造 Labels (Masking Prompt)
        # [为什么要这么写]: labels 中 prompt 部分使用 IGNORE_INDEX.
        # [解决的问题]: 确保计算 Loss 时, 模型只为生成的 Response 负责,
        # 而不预测已经给出的 Prompt. 这是 SFT 和对齐训练的标准做法.
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * source_len + response_ids
        kl_input_ids = kl_prompt_ids + kl_response_ids
        kl_labels = [IGNORE_INDEX] * kl_source_len + kl_response_ids
        return input_ids, labels, kl_input_ids, kl_labels, kto_tag

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        全量数据集预处理.
        """

        # 1. 构造"错位对"用于 KL 估计 (Mismatched Pairs for KL)
        # [核心技巧]: 将整个 batch 的 response 列表向后偏移一位.
        # [解决的问题]: KTO 论文指出, 为了估计 KL 散度的参考项, 我们需要对比"当前 Prompt 的回答"
        # 和"其他随机 Prompt 的回答". 这种偏移法能高效地在一个 batch 内构造对比样本, 而无需加载额外数据.
        # Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of completions.
        kl_response = [examples["_response"][-1]] + examples["_response"][:-1]
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # 基础数据校验
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # 执行逐行编码
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

            # 汇总结果
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["kl_input_ids"].append(kl_input_ids)
            model_inputs["kl_attention_mask"].append([1] * len(kl_input_ids))
            model_inputs["kl_labels"].append(kl_labels)
            model_inputs["kto_tags"].append(kto_tag)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        # 2. 数据平衡预警 (Dataset Balance Warning)
        # [为什么要这么写]: 统计 Desirable vs Undesirable 的数量.
        # [解决的问题]: KTO 对标签比例有一定的敏感性. 如果用户提供的数据全是"满意"或全是"不满意",
        # 损失函数将无法正常收敛, 通过日志提醒用户检查数据质量.
        desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
        undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
        if desirable_num == 0 or undesirable_num == 0:
            logger.warning_rank0("Your dataset only has one preference type.")

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        打印调试示例.
        [解决的问题]: 让研究员能直观看到 Tokenization 后的结果.
        通过 decode 还原 labels, 可以检查 IGNORE_INDEX 是否正确遮蔽了 Prompt.
        """
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

"""
资深研究员视角下的技术亮点:

对 KTO 算法精髓的理解:
KTO 最核心的贡献是利用前景理论(Prospect Theory)来处理非成对数据. 代码中 kl_response 的构建是点睛之笔, 它通过简单的列表切片偏移, 就解决了训练中需要评估模型在"不匹配响应"上概率的问题, 极大地节省了计算开销.

多模态一致性:
代码中大量使用了 mm_plugin. 这反映了 LLaMA-Factory 的前瞻性设计: 不仅仅是文本对齐, 图像、视频、音频的反馈对齐逻辑在这里被高度统一了.

工程鲁棒性:
infer_seqlen 的使用避免了 LLM 训练中最常见的"截断了答案"的错误. 同时, 对数据集标签单一性的警告, 体现了框架对算法收敛细节的关注.

希望这份注释能帮你深入理解 KTO 数据处理的底层逻辑!
"""

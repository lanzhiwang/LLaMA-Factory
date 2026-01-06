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
在 LLM 的对齐阶段(如 DPO 定向偏好优化或 RM 奖励模型训练), 数据通常以"成对"形式出现:
一个问题(Prompt)对应一个"好的回答"(Chosen)和一个"差的回答"(Rejected).
这段代码的精髓在于确保 Chosen 和 Rejected 分支在 Token 层面严格对齐, 同时处理多模态输入和智能截断.
"""

class PairwiseDatasetProcessor(DatasetProcessor):
    """
    偏好对齐任务的数据处理器.
    [核心目标]: 将"一问两答"的原始数据转化为模型可训练的张量格式, 并处理 Prompt 掩码.
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
        单条数据编码逻辑: 将文本和多模态信息转为 Token IDs.
        """

        # 1. 多模态信息注入 (Multimodal Plugin Integration)
        # [为什么要这么写]: 调用 mm_plugin 将图像/视频占位符插入消息列表中.
        # [解决的问题]: 确保在文本编码前, 多模态特征的位置已经预留, 避免 Token 偏移.
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )

        # 2. 模板化编码 (Template-based Encoding)
        # [为什么要这么写]: 使用 template.encode_oneturn 分别获取 Prompt 和 Response 的 IDs.
        # [解决的问题]: 由于 Chosen 和 Rejected 共享同一个 Prompt, 这种方式保证了 Prompt 部分的 Token 序列在两个分支中是完全一致的, 这对于偏好损失函数的计算(如 DPO Loss)至关重要.
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        # 3. 高效 EOS 处理 (Efficient EOS Handling)
        # [解决的问题]: 某些模型需要显式的结束符来识别回答边界. 如果开启了 efficient_eos, 手动补齐 EOS.
        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        # 4. 多模态 Token 修正
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        # 5. 智能截断策略 (Heuristic Truncation Strategy)
        # [为什么要这么写]: 计算 Prompt 和"最长回答"之间的截断平衡.
        # [解决的问题]: 在长文本场景下, 如果直接从末尾截断, 会导致最重要的 Answer 部分丢失.
        # 这里通过 infer_seqlen 优先保证 Response 的长度, 截断 Prompt.
        # 使用 max(chosen, rejected) 作为参考, 确保两个分支使用相同的截断锚点, 维持 Prompt 对齐.
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        # 6. 构造训练输入与标签掩码 (Input & Label Construction)
        # [为什么要这么写]: 将 Prompt 和 Response 拼接, 并在 Labels 中将 Prompt 部分设为 IGNORE_INDEX.
        # [解决的问题]: 在对齐训练中, 我们只需要对模型生成的"回答"计算 Loss, 而不对"问题"计分.
        # 这种 Masking 机制引导模型专注于学习人类偏好, 而不是复读问题.
        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        数据集批量预处理.
        """

        # 使用 defaultdict 方便收集分布式环境下不规则长度的列表数据
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # 基础数据质量校验: Prompt 长度应为奇数(问-答-问...), 且必须有两个回答用于对比
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # 执行具体编码
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            # 这里的字段命名(chosen_xxx, rejected_xxx)是专门为了适配 DPO/RM Trainer 的输入要求
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)

            # 保留原始媒体数据用于多模态编码器
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        可视化调试工具.
        [解决的问题]: 让研究员能清晰看到 Tokenizer 是否正确处理了特殊字符, 以及 Label Mask 是否生效.
        通过 decode(filter(IGNORE_INDEX)), 可以直观看到模型真正"学习"的部分.
        """

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

"""
资深专家视角:

关于 infer_seqlen 的权衡:
在 LLM 对齐训练中, 回答(Response)包含了决定偏好的核心信息. 如果 Prompt + Chosen 是 2000 tokens, 但 cutoff_len 只有 1024, 我们必须截断. 代码中将两者放在一起比较并计算 source_len, target_len, 是为了确保在截断发生时, Chosen 和 Rejected 两个序列保留的 Prompt 长度是完全一致的. 如果长度不一致, 模型在计算 DPO 损失时会产生逻辑偏差.

对多模态的支持:
mm_plugin 的使用表明该项目不仅支持文本, 还支持多模态对齐. 这是目前 LLM 研究的前沿(如 LLaVA 等模型的对齐).

工程健壮性:
通过 IGNORE_INDEX 隔离 Prompt 是标准但极易出错的细节. 该代码清晰地展示了如何通过拼接 [IGNORE_INDEX] * source_len 来实现精准的 Loss 计算控制.

希望这份详尽的注释能帮助你更好地理解 LLaMA-Factory 的内部机制!
"""

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

"""
这段代码定义了 UnsupervisedDatasetProcessor 类.
虽然类名包含 "Unsupervised"(无监督), 但在 LLaMA-Factory 的语境下,
它通常用于处理非指令微调格式的数据, 或者在某些评估场景(如计算困惑度 PPL)下,
将 Prompt 和 Response 分开处理, 但不对 Prompt 进行 Loss 掩码(与 SFT 掩码 Prompt 不同).
"""

class UnsupervisedDatasetProcessor(DatasetProcessor):
    """
    无监督/无掩码数据处理器.
    [设计动机]: 与 Supervised 处理器不同, 它主要用于预训练风格的任务或特定评估逻辑.
    [解决的问题]: 处理原始对话流, 而不像 SFT 那样将 Prompt 部分的 Label 设为 -100.
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
        对单条数据进行编码, 生成 input_ids 和 labels.
        """
        # 1. 消息流构建 (Message Stream Construction)
        # [为什么要这么写]: 确保消息序列以 Assistant 的角色结尾.
        # [解决的问题]: 在无监督预测中, 如果 response 为空, 需要补一个空内容占位符,
        # 从而让模板(Template)正确插入"回答开始"的引导词(如 Assistant: ), 确保模型推理逻辑闭环.
        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

        # 2. 多模态插件预处理 (Multimodal Plugin Processing)
        # [解决的问题]: 在文本 Tokenize 之前, 处理图片、视频等媒体占位符.
        # 确保多模态特征(Vision Embeddings)在序列中的位置与文本描述严格对齐.
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)

        # 3. 模板化编码 (Template-based Encoding)
        # [为什么要这么写]: 调用 encode_oneturn 将角色信息和内容转化为 Token ID.
        # [解决的问题]: 不同的模型(如 Llama-3, Qwen, Yi)有完全不同的 Chat Template(控制字符).
        # 该方法屏蔽了底层差异, 确保生成的序列符合模型训练时的分布.
        input_ids, labels = self.template.encode_oneturn(self.tokenizer, messages, system, tools)

        # 4. 显式 EOS 处理 (Efficient EOS Handling)
        # [解决的问题]: 在非掩码模式下, 如果模型开启了高效结束符策略,
        # 需要在标签(Labels)末尾手动补全 EOS. 这解决了模型"不会停顿"或"生成冗余"的问题.
        if self.template.efficient_eos:
            labels += [self.tokenizer.eos_token_id]

        # 处理多模态 Token ID 占位符(如将图像标记替换为特定的索引范围)
        input_ids, _ = self.template.mm_plugin.process_token_ids(
            input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        # 5. 智能截断与长度分配 (Smart Truncation)
        # [为什么要这么写]: 调用 infer_seqlen 动态推导 source 和 target 的长度.
        # [解决的问题]: 由于没有 IGNORE_INDEX 掩码, 整个序列都是训练目标.
        # 但我们依然需要根据 cutoff_len 限制长度, 防止显存 OOM.
        # infer_seqlen 保证了截断时尽可能保留序列末尾(通常是核心答案部分)的信息.
        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        全量数据集转换逻辑.
        """

        # 使用 defaultdict 方便收集各种字段, 特别是在分布式数据并行场景下保持 key 的一致性
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # 基础数据校验: 对话历史必须是奇数(User-Assistant-User...), 确保格式合法性
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # 执行逐行编码转换
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            # 构建标准的 Transformer 训练格式
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

            # 保留原始媒体引用, 供 DataCollator 后续处理多模态张量
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """
        可视化调试辅助.
        [解决的问题]: 对于非 SFT 数据, 研究人员经常搞不清楚哪些部分会被模型看到, 哪些是预测目标.
        通过 print 还原编码后的文本, 可以直观确认控制字符(如 <|im_start|>)是否插入正确.
        """
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))

"""
资深专家视角下的设计分析:

无监督模式的特殊性:
在 UnsupervisedDatasetProcessor 中, input_ids 和 labels 的长度分配与 Supervised 有微妙的不同. 由于没有 IGNORE_INDEX(-100), 整个序列的所有 Token 都会产生梯度. 这种模式常用于增量预训练(Continual Pre-training), 其中数据已经以对话格式存在, 但我们希望模型学习整个分布.

多模态前瞻性:
代码中大量调用了 mm_plugin(多模态插件). 这反映了 LLaMA-Factory 的工业级工程能力——它不仅仅处理文本, 还将图片、视频和音频的"序列化"过程高度抽象, 使得无监督微调逻辑可以无缝扩展到多模态大模型.

鲁棒性考量:
len(prompt) % 2 != 1 的校验看起来简单, 但实际上解决了由于数据爬取错误导致的"两个 User 连续说话"或"Assistant 没说话"等格式异常. 在处理 PB 级别原始数据时, 这种前置校验能防止训练过程中出现无法收敛的异常梯度.

希望这份注释能帮你深入理解 LLaMA-Factory 的核心架构!
"""

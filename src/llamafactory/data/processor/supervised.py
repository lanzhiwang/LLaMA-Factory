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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

"""
这段代码实现了 有监督微调(SFT) 中的两大核心工程技术: 多轮对话的智能编码与高效序列打包(Packing).
它直接决定了微调时模型的收敛质量和显存利用率.
"""

@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    """
    标准有监督微调(SFT)处理器.
    [核心目标]: 处理多轮对话、多模态输入, 并精确控制 Loss 计算范围.
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
        # 1. 多模态预处理
        # [解决的问题]: 确保图片/视频占位符在文本编码前被正确插入到消息序列中, 避免 Token 偏移.
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)

        # 获取初始的多模态 Token 及其 Labels 掩码(通常多模态部分不计 Loss)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )

        # 2. 多轮对话编码 (Multi-turn Encoding)
        # [为什么要这么写]: 将对话拆分为 (Source, Target) 的 Pair 列表.
        # [解决的问题]: 统一处理单轮和多轮对话, 确保每一轮的 Prompt 格式(如 User: / Assistant:)都符合模型要求.
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)

        # 3. 优先级策略
        # [为什么要这么写]: 如果开启 mask_history, 则反转序列.
        # [解决的问题]: 在长对话截断时, 优先保留最后一轮(最新的指令和回答), 因为最后一轮通常是训练权重最高的部分.
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            # 4. 智能截断 (Truncation)
            # [解决的问题]: 当多轮对话总长超标时, 通过比例推导(infer_seqlen)决定截断多少.
            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            # 5. Label Masking 逻辑 (核心要点)
            # [为什么要这么写]:
            # - train_on_prompt: 如果开启, 则计算 Prompt 的 Loss(用于特定的知识灌输).
            # - efficient_eos: 在多轮对话中间插入 EOS, 帮助模型学习如何结束一轮对话.
            # - IGNORE_INDEX: [关键]被设为此值的 Token 不参与梯度计算.
            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                # 只有后续轮次的开头需要 EOS 标签, 确保多轮连贯性
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            # 6. 历史遮蔽 (History Masking)
            # [解决的问题]: 如果 mask_history 为真, 除最后一轮(转置后的 turn_idx=0)外, 所有 Target 都不计 Loss.
            # 这可以让模型专注于"在给定上下文后如何回答当前问题", 而不是复读历史.
            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            # 7. 序列拼接
            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        # 8. 结尾补齐
        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        遍历数据集, 处理多模态和对话字段, 输出标准的训练字典
        """
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # 基础过滤: 对话必须成对(问+答)
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
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

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    """
    序列打包(Packing)处理器.
    [核心价值]: 通过将多个短样本打包进一个 cutoff_len 长度的序列, 消除 Padding, 提升训练速度 2-4 倍.
    """
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # 1. 缓冲区初步编码
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # ... 过滤逻辑 ...
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
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
            length = len(input_ids)
            # 收集长度, 为下文的"贪心背包算法"做准备
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        # 2. 贪心打包 (Greedy Packing)
        # [为什么要这么写]: 通过背包算法尽可能填满每一个 cutoff_len 的容器.
        # [解决的问题]: 减少 GPU 对空闲 Padding 的计算, 极大提高吞吐量.
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                # 拼接序列
                packed_input_ids += batch_input_ids[index]
                # 3. Position ID 重置 (Positional Encoding Reset)
                # [为什么要这么写]: 打包后的每个小样本都应该有自己独立的起始位置(0, 1, 2...).
                # [解决的问题]: 防止后面样本的位置编码受到前面样本长度的影响, 导致模型混淆不同样本.
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]

                # 4. 精致打包掩码 (Neat Packing)
                # [为什么要这么写]: 将不同的样本分配不同的 ID(1, 2, 3...).
                # [解决的问题]: 配合 Flash Attention 的"分区块"注意力,
                # 确保样本 1 的 Token 绝对不会注意到样本 2 的内容, 实现逻辑上的并行训练, 互不干扰.
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            # 5. Flash Attention 长度适配
            # [为什么要这么写]: 补齐到 cutoff_len + 1 并使用 pad_token.
            # [解决的问题]: 某些 Flash Attention 实现对固定步长有严格要求, 这样写可以最大化内核计算效率.
            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            # 最终验证与分发
            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs

"""
资深专家总结:

工程与算法的平衡: SupervisedDatasetProcessor 通过 IGNORE_INDEX 完美实现了"只学回答, 不学指令"的 SFT 核心逻辑; 同时对多轮对话的转置处理展示了对长文本截断场景的深度考量.

极致的性能优化: PackedSupervisedDatasetProcessor 的 Packing 逻辑是 LLaMA-Factory 高效训练的"秘密武器". 通过重置 position_ids 和构造多值 attention_mask(Neat Packing), 它在物理层面将多个独立样本合并, 但在逻辑层面通过 Attention Mask 实现了它们之间的完美隔离.

多模态前瞻性: 代码中无处不在的 mm_plugin 意味着该逻辑已经为多模态大模型(如图片、视频理解)做好了底层适配, 体现了工业级微调框架的可扩展性.

希望这些深度注释能帮你透彻理解 LLaMA-Factory 的核心架构!
"""

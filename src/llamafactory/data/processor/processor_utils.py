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

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


@dataclass
class DatasetProcessor(ABC):
    r"""
    A class for data processors.

    数据处理器的抽象基类.
    [为什么要这么写]: 采用典型的工厂/模板模式设计.
    [解决的问题]: LLM 任务种类繁多(指令微调 SFT、预训练 PT、偏好对齐 DPO 等), 且涉及多模态.
    通过定义抽象接口, 强制要求所有处理器实现预处理(preprocess_dataset)和调试输出(print_data_example),
    保证了整个框架在处理不同数据集时具有统一的调用链, 极大降低了系统耦合度.
    """

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""
        Build model inputs from the examples.

        将原始数据转化为模型可接受的 input_ids, labels 等输入.
        """
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""
        Print a data example to stdout.

        将处理后的数据打印到终端, 用于开发者排查 Tokenizer 转换和 Template 拼接是否正确.
        """
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""
    Find the index of largest number that fits into the knapsack with the given capacity.

    使用二分查找寻找能放入剩余背包空间的最大数值索引.
    [为什么要这么写]: 利用 bisect 库实现 O(log N) 的搜索复杂度.
    [解决的问题]: 在高性能 Packing 算法中, 需要快速从候选长度列表中找到"最接近但不过载"的样本长度.
    相比于遍历搜索, 二分查找在处理数万个样本时能显著降低 CPU 预处理开销, 避免 GPU 在训练开始前处于闲置等待状态.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""
    Implement efficient greedy algorithm with binary search for the knapsack problem.

    高效的贪心背包算法(Packing 算法).
    [为什么要这么写]: 实现序列打包(Sequence Packing), 将多个短样本拼接成一个长度为 `cutoff_len` 的长样本.
    [解决的问题]: LLM 训练极其昂贵. 如果 90% 的样本长度远小于 `cutoff_len`(例如 512 < 4096),
    直接训练会导致显存内大量的 Padding 填充, 造成算力浪费.
    该算法通过贪心策略和二分优化, 将短样本"塞"进最大长度限制内, 从而将训练吞吐量(Tokens per Second)提升数倍.
    """

    # 排序是为了配合二分查找, 提高搜索效率
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            # 寻找当前剩余空间能容纳的最长样本
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                # 没有任何样本能塞入当前这个"包"了
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            # 将选中的样本长度移出待选池
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""
    Compute the real sequence length after truncation by the cutoff_len.

    根据最大长度限制(cutoff_len)智能推导 Source 和 Target 的实际保留长度.
    [为什么要这么写]: 实现比例自适应的截断策略.
    [解决的问题]: 传统的"简单截断"往往直接从序列末尾砍掉, 导致最重要的 Answer(Target)部分全失.
    这里的逻辑解决了三大痛点:
    1. 保护 Target: 如果 Target 很短, 优先截断过长的 Source(Question).
    2. 保护 Source: 如果 Source 很短, 优先截断过长的 Target(Answer).
    3. 比例协调: 如果两者都非常长, 则根据它们的长度比例分配截断额度, 确保上下文和答案都能保留关键部分.
    这是保证大模型微调数据质量(Data Quality)的工业级细节处理.
    """

    # 情况1: Target 特别短, 全力保留 Target, 截断 Source
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len

    # 情况2: Source 特别短, 全力保留 Source, 截断 Target
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len

    # 情况3: 两者都长, 按比例动态分配, 避免其中一方被完全截掉
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len

"""
高级研究员视角下的架构深度解析:

Packing 算法的必要性:
在分布式训练中, 通讯开销(Communication Overhead)是巨大的. 如果你不使用 greedy_knapsack 进行序列打包, 你的 batch 中可能充斥着大量的 PAD token, 这不仅浪费了显存, 还降低了梯度更新的效率. LLaMA-Factory 引入这个算法是为了确保每一张卡处理的每一个 batch 几乎都是"满载"的.

infer_seqlen 的启发式策略:
在推理(Inference)中我们常说"Context is King", 但在微调训练中, "Label is King". 如果截断不当, 模型会学到不完整的句子甚至错误的逻辑. 这段代码中的 2 * < cutoff_len 阈值设定是一个经典的经验值, 旨在处理极端长文本输入时的健壮性.

高性能二分查找:
bisect 是 Python 原生库. 之所以在这里使用它, 是因为在处理像 Pile 这种百万级规模的数据集时, 任何 O(N2) 的数据处理都会成为 CPU 瓶颈, 导致昂贵的 GPU 集群在空转. 这就是高级 Python 开发工程师对**性能瓶颈分析(Bottleneck Analysis)**的直觉体现.

希望这些解析能让你对 LLaMA-Factory 的底层设计有更深刻的认识!
"""

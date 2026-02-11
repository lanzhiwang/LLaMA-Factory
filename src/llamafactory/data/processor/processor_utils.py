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

    bisect.bisect(list, item) 用于在已排序列表 list 中查找 item 的插入位置.
    如果列表中已存在 item, 则返回插入在所有已有 item 的右边.
    用途: 寻找插入点, 使得序列在插入后保持有序.
    返回值: 索引 i.

    >>> import bisect
    >>> data = [10, 20, 20, 20, 30, 40]  # 列表必须是有序的
    >>> x = 20
    >>> pos = bisect.bisect(data, x)
    >>> pos
    4
    >>> bisect.bisect(data, 5)
    0
    >>> bisect.bisect(data, 50)
    6
    >>>

    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""
    Implement efficient greedy algorithm with binary search for the knapsack problem.

    实现基于二分查找的高效贪心算法, 解决"装箱问题"(Bin Packing Problem).

    [功能描述]:
    该函数将一组样本长度(numbers)尽可能高效地塞进一个个容量固定为 capacity 的"背包"中.

    [在 LLaMA-Factory 中的作用]:
    在 SFT(有监督微调)或预训练中, 我们将多个训练样本(Prompt + Response)打包在一起.
    例如: capacity(cutoff_len)是 4096, 我们有三个长度为 1000, 2000, 800 的样本.
    如果不打包, 这三个样本会占用 3 个 4096 的窗口(大量 Padding);
    如果打包, 它们可以挤进同一个 4096 的窗口中, 节省了近 3 倍的算力.

    [算法逻辑]:
    1. 首先对样本长度进行升序排列, 以便进行二分查找.
    2. 开启一个新背包, 尝试从剩余样本中寻找能塞进去的"最大"样本.
    3. 重复步骤 2, 直到没有任何样本能塞进当前背包, 然后开启下一个背包.
    4. 贪心策略(优先塞入能放下的最大者)能有效减少背包的总数.

    Args:
        numbers (list[int]): 样本长度列表, 例如 [120, 500, 2000, ...]
        capacity (int): 序列最大截断长度 (cutoff_len), 例如 4096

    Returns:
        list[list[int]]: 打包后的结果, 每个子列表代表一个"背包"里的样本长度组合.
    """

    # 1. 排序: 这是二分查找的前提条件
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    # 只要待选池里还有样本长度, 就继续打包
    while numbers:
        current_knapsack = []  # 当前背包(即一个训练窗口)
        remaining_capacity = capacity  # 当前窗口剩余可容纳的 Token 数

        while True:
            # 2. 二分搜索优化: 在 O(log N) 时间内找到剩余空间能装下的最大样本
            # 这种做法比循环遍历快得多, 尤其是在数据集很大时
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                # 没有任何样本能塞入当前这个窗口了, 关闭当前窗口
                break  # no more numbers fit in this knapsack

            # 3. 记录并更新:
            # 减去被占用的长度
            remaining_capacity -= numbers[index]  # update the remaining capacity
            # 将该样本长度从待选池中弹出, 并加入当前背包
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        # 将装满(或无法再装)的背包存入结果
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


if __name__ == "__main__":
    # 示例 A: 普通短样本打包
    # 样本长度列表
    sample_lengths = [100, 250, 150, 300, 50, 200, 400, 80]
    # 最大窗口长度
    cutoff_len = 512

    packed_results = greedy_knapsack(sample_lengths, cutoff_len)

    print(f"原始样本数: {len(sample_lengths)}")
    print(f"打包后的窗口数: {len(packed_results)}")
    for i, group in enumerate(packed_results):
        print(f"窗口 {i+1}: 包含长度 {group}, 总计: {sum(group)}")

    # 示例 B: 为什么它能节省算力？
    # 对比"不打包"与"打包"的资源消耗:
    lengths = [2000, 2100, 1500, 500, 400, 3000]
    capacity = 4096

    # 场景 1: 不打包 (Vanilla Training)
    # 每个样本独立占用一个 4096 的窗口
    unpacked_tokens = len(lengths) * capacity
    # 结果是 24,576 Tokens 的计算量 (大部分是无用的 Padding)

    # 场景 2: 使用 greedy_knapsack 打包
    packed = greedy_knapsack(lengths.copy(), capacity)
    packed_windows = len(packed)
    total_compute_tokens = packed_windows * capacity
    # 结果:
    # 窗口 1: [3000, 500, 400] -> 3900
    # 窗口 2: [2100, 1500] -> 3600
    # 窗口 3: [2000] -> 2000
    # 计算量只有 3 * 4096 = 12,288 Tokens

    efficiency_gain = unpacked_tokens / total_compute_tokens
    print(f"效率提升: {efficiency_gain:.2f} 倍")
    # 输出: 效率提升: 2.00 倍


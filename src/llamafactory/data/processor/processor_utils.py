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

    在 LLaMA-Factory 的架构中, DatasetProcessor 是数据生命周期的核心抽象层.
    它处于"原始数据"与"模型张量"之间, 负责将各种格式的 JSON 数据(如 Alpaca 或 ShareGPT 格式)转化为模型能够直接训练的 input_ids 和 labels.

    数据处理器的抽象基类.

    [为什么要这么写]:
    采用了"模板方法"设计模式. LLM 微调涉及多种任务(SFT, DPO, Pretrain), 每种任务对 Token 拼接、Loss Masking 的逻辑都不同.
    通过定义这个抽象基类, 框架可以统一调用流程, 而具体的拼接逻辑交给子类实现.

    [解决的问题]:
    1. 统一接口: 无论是文本微调还是多模态训练, Trainer 只需调用 preprocess_dataset 即可.
    2. 依赖注入: 将 Tokenizer、Chat Template 和数据配置(cutoff_len 等)封装在一起, 减少函数参数传递的复杂性.
    """

    # 注入对话模板(如 Llama-3, Qwen), 负责添加角色标记(如 <|im_start|>)
    template: "Template"
    # 注入分词器, 负责将拼接好的字符串转为 ID 序列
    tokenizer: "PreTrainedTokenizer"
    # 可选的多模态处理器(如处理图像/音频的处理器)
    processor: Optional["ProcessorMixin"]
    # 注入数据参数配置(如截断长度 cutoff_len, 是否打包 packing 等)
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""
        Build model inputs from the examples.

        核心方法: 将原始数据示例批量转换为模型输入.

        [功能]:
        接收 HuggingFace `datasets` 库生成的 Batch(字典格式, 值为列表),
        执行拼接、分词、截断、Labels 生成等操作, 返回包含 input_ids, labels 等字段的字典.

        [为什么重要]:
        这里是实现 SFT 中"只对回答计算 Loss(Masking Prompt)"逻辑的地方.
        """
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""
        Print a data example to stdout.

        调试方法: 将处理后的一个样本打印到控制台.

        [解决的问题]:
        Tokenize 过程是一个"黑盒". 通过这个方法, 开发者可以直观地看到:
        1. Chat Template 拼接是否正确(是否有空格丢失、角色标记是否放对).
        2. Labels 是否正确遮蔽了 Prompt(IGNORE_INDEX 是否在正确位置).
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

    """
    1. 排序: 这是二分查找的前提条件
    >>> sample_lengths = [100, 250, 150, 300, 50, 200, 400, 80]
    >>> sample_lengths.sort()
    >>> sample_lengths
    [50, 80, 100, 150, 200, 250, 300, 400]
    >>>
    """
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

    在 LLM 微调过程中, 数据的截断策略直接影响模型的训练质量.
    如果简单地使用 tokens[:cutoff_len] 进行截断, 往往会把序列末尾的 Target(预测答案) 全部砍掉, 导致模型学不到任何有意义的内容.
    infer_seqlen 的核心目标是: 在总长度超过限制时, 以一种"智能且平衡"的方式分配 Prompt(源)和 Response(目标)的长度, 尽可能保护最重要的信息.

    根据最大长度限制 (cutoff_len) 智能推导 Source 和 Target 的实际保留长度.

    [为什么要这么写]:
    在 SFT(有监督微调)中, Source 是问题, Target 是答案. 如果总长超标, 我们不能暴力截断.
    本函数通过启发式策略, 动态平衡两者的权重, 避免其中一方被完全截断.

    [解决的问题]:
    1. 保护 Label: 如果答案(Target)被截断, 模型会学到破碎的语义.
    2. 保护 Context: 如果问题(Source)被截断过重, 模型会因为丢失关键上下文而无法理解指令.
    3. 比例协调: 在两者都极长时, 按比例缩小, 维持原始数据的分布感.
    """

    # 情况 1: Target(答案)相对较短(不足最大长度的一半)
    # [策略]: 优先保障 Target 的完整性.
    # 只要总长度够, 就让 Target 占满它需要的空间, 剩余空间全部留给 Source.
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len

    # 情况 2: Source(问题)相对较短(不足最大长度的一半)
    # [策略]: 优先保障 Source 的完整性.
    # 如果问题本身很短, 我们没必要截断问题, 剩下的空间全部给 Target 自由发挥.
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len

    # 情况 3: Source 和 Target 都非常长
    # [策略]: 按比例截断(Proportional Truncation).
    # [解决的问题]: 避免某一方"吃掉"另一方. 例如在长对话中, 如果两者都长,
    # 我们根据它们原始长度的比例, 公平地分配 cutoff_len 的额度.
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    # --- 最终修正环节 ---

    # 1. 确定最终 Target 长度: 不能超过它原始的长度, 也不能超过上面分配的 quota
    new_target_len = min(max_target_len, target_len)

    # 2. 计算剩余给 Source 的空间: 总长度减去已经分给 Target 的长度, 确保不小于 0
    max_source_len = max(cutoff_len - new_target_len, 0)

    # 3. 确定最终 Source 长度: 同理, 不能超过原始长度, 也不能超过剩余空间
    new_source_len = min(max_source_len, source_len)

    return new_source_len, new_target_len


if __name__ == "__main__":
    # 示例 A: 普通短样本打包
    # 样本长度列表
    sample_lengths = [100, 250, 150, 300, 50, 200, 400, 80]
    # 最大窗口长度
    cutoff_len = 512
    print(f"原始样本数: {len(sample_lengths)}")
    # 原始样本数: 8

    packed_results = greedy_knapsack(sample_lengths, cutoff_len)

    print(f"打包后的样本分布: {packed_results}")
    print(f"打包后的窗口数: {len(packed_results)}")
    for i, group in enumerate(packed_results):
        print(f"窗口 {i+1}: 包含长度 {group}, 总计: {sum(group)}")

    print("---" * 10)

    # 示例 B: 为什么它能节省算力?
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
    print(f"打包后的样本分布: {packed}")
    print(f"打包后的窗口数: {packed_windows}")
    for i, group in enumerate(packed):
        print(f"窗口 {i+1}: 包含长度 {group}, 总计: {sum(group)}")

    total_compute_tokens = packed_windows * capacity

    efficiency_gain = unpacked_tokens / total_compute_tokens
    print(f"效率提升: {efficiency_gain:.2f} 倍")
    # 输出: 效率提升: 2.00 倍

    print("---" * 20)

    # 假设我们的最大长度限制 (cutoff_len) 为 512
    def test_truncation(s_len, t_len, cutoff=512):
        new_s, new_t = infer_seqlen(s_len, t_len, cutoff)
        print(f"原始: Source={s_len:<4} Target={t_len:<4} | 总和={s_len+t_len}")
        print(f"分配: Source={new_s:<4} Target={new_t:<4} | 总和={new_s+new_t}")
        print("*" * 10)

    # 场景 1: 总长度小于 cutoff_len (不进行任何截断)
    test_truncation(100, 200)
    # 输出: 保持 100, 200

    # 场景 2: Source 极长, Target 较短 (保护 Target, 截断 Source)
    # 例如: 长文章总结任务
    test_truncation(1000, 100)
    # 输出: Target 100 完整保留, Source 被截断为 412 (512-100)

    # 场景 3: Source 较短, Target 极长 (保护 Source, 截断 Target)
    # 例如: 根据短提示写长小说
    test_truncation(50, 1000)
    # 输出: Source 50 完整保留, Target 被截断为 462 (512-50)

    # 场景 4: 两者都非常长 (按比例分配)
    # 例如: 两个长文档的对比或改写
    test_truncation(1000, 1000)
    # 输出: 两者长度相同且都超标, 各分配一半空间 (256, 256)

    # 场景 5: 复杂比例场景
    test_truncation(1500, 500) # 3:1 的比例
    # 分配结果会接近 Source=384, Target=128

    print("---" * 20)
    class SimpleSFTProcessor(DatasetProcessor):
        def preprocess_dataset(self, examples):
            # 简化版: 假设输入是 {"instruction": [...], "output": [...]}
            model_inputs = {"input_ids": [], "labels": []}

            for i in range(len(examples["instruction"])):
                prompt = examples["instruction"][i]
                answer = examples["output"][i]

                # 使用 template 拼接字符串 (模拟行为)
                full_text = f"User: {prompt}\nAssistant: {answer}"

                # 分词
                ids = self.tokenizer.encode(full_text, add_special_tokens=True)

                # 简单的 Label 生成: 假设我们要对全文本计算 Loss
                labels = ids.copy()

                model_inputs["input_ids"].append(ids)
                model_inputs["labels"].append(labels)

            return model_inputs

        def print_data_example(self, example):
            # 解码并展示
            decoded_text = self.tokenizer.decode(example["input_ids"])
            print(f"--- 训练样本预览 ---\n{decoded_text}\n------------------")

    # 1. 模拟环境
    from transformers import AutoTokenizer

    # 假设我们使用 Qwen 的分词器
    tokenizer = AutoTokenizer.from_pretrained("./models/Qwen/Qwen3-4B-Instruct-2507/")

    # 2. 模拟数据
    raw_data = {
        "instruction": ["你是谁?", "今天天气怎么样?"],
        "output": ["我是智谱AI开发的大模型.", "今天天气晴朗."]
    }

    # 3. 初始化处理器 (此处省略 template 和 data_args 的具体复杂实例化)
    processor = SimpleSFTProcessor(
        template=None, # 实际中会有 Template 对象
        tokenizer=tokenizer,
        processor=None,
        data_args=None
    )

    # 4. 执行预处理
    processed_batch = processor.preprocess_dataset(raw_data)

    # 5. 查看第一个样本结果
    processor.print_data_example({"input_ids": processed_batch["input_ids"][0]})


"""
$ python src/llamafactory/data/processor/processor_utils.py
原始样本数: 8
打包后的样本分布: [[400, 100], [300, 200], [250, 150, 80], [50]]
打包后的窗口数: 4
窗口 1: 包含长度 [400, 100], 总计: 500
窗口 2: 包含长度 [300, 200], 总计: 500
窗口 3: 包含长度 [250, 150, 80], 总计: 480
窗口 4: 包含长度 [50], 总计: 50
------------------------------
打包后的样本分布: [[3000, 500, 400], [2100, 1500], [2000]]
打包后的窗口数: 3
窗口 1: 包含长度 [3000, 500, 400], 总计: 3900
窗口 2: 包含长度 [2100, 1500], 总计: 3600
窗口 3: 包含长度 [2000], 总计: 2000
效率提升: 2.00 倍
------------------------------------------------------------
原始: Source=100  Target=200  | 总和=300
分配: Source=100  Target=200  | 总和=300
**********
原始: Source=1000 Target=100  | 总和=1100
分配: Source=412  Target=100  | 总和=512
**********
原始: Source=50   Target=1000 | 总和=1050
分配: Source=50   Target=462  | 总和=512
**********
原始: Source=1000 Target=1000 | 总和=2000
分配: Source=256  Target=256  | 总和=512
**********
原始: Source=1500 Target=500  | 总和=2000
分配: Source=384  Target=128  | 总和=512
**********
------------------------------------------------------------
--- 训练样本预览 ---
User: 你是谁?
Assistant: 我是智谱AI开发的大模型.
------------------
$
"""

# Copyright 2025 Musab Gultekin and the LlamaFactory team.
#
# This code is based on the Musab Gultekin's functionary library.
# https://github.com/MeetKai/functionary/blob/main/functionary/train/packing/monkey_patch_packing.py
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
#
# MIT License
#
# Copyright (c) 2023 Musab Gultekin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r"""Get the sequence lengths in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [2, 3, 1, 2, 3]
    ```

    这段代码 get_seqlens_in_batch 是实现 高效序列打包(Sequence Packing / Multipack) 训练的关键. 在微调超大规模模型时, 为了压榨 GPU 算力, 我们通常不希望在 Batch 中看到大量的 Padding Token(0). 我们会将多条短序列拼接(Pack)在一起形成一个固定长度的序列.
    然而, 传统的注意力算子(Attention Kernel)无法区分这些拼接在一起的序列. 为了配合 Flash Attention 2 的 varlen(变长)接口, 我们需要精确计算出每个 Batch 中每条独立序列的实际长度.

    获取当前 Batch 中每条打包序列的真实长度.

    [为什么要这么写]:
    在开启序列打包(Packing)后, LLaMA-Factory 构造的 attention_mask 不再是只有 0 和 1 的二值张量,
    而是采用自增索引来标识不同的序列. 例如 [1, 1, 2, 2, 2, 0] 表示:
    - 前两个 token 属于序列 1
    - 后面三个 token 属于序列 2
    - 最后一个 0 是纯粹的 Padding

    [解决的问题]:
    1. 高效算子适配: Flash Attention 2 提供了一个变长接口(flash_attn_varlen_func), 它要求传入
       一个一维张量, 记录整个 Batch 中所有序列的长度, 从而在计算注意力时避免跨序列的"信息污染".
    2. 硬件利用率: 通过 Packing, 我们可以让每个 Batch 几乎没有 Padding, 从而提升 2-4 倍的吞吐量.

    高级研究员视角的架构解析:

    数据的"多重含义":
    在 LLaMA-Factory 中, attention_mask 在打包模式下被赋予了 Segment ID 的职责. 这种设计巧妙地复用了数据结构, 避免了在 DataLoader 中传递额外的元数据张量, 降低了分布式训练时的通信开销.

    规避算子限制:
    Flash Attention 虽然强大, 但它有一个硬性约束: 它在计算时必须知道序列边界, 否则会产生"跨样本注意力", 导致模型学习到错误的因果逻辑. get_seqlens_in_batch 产出的 seqlens 张量, 配合 torch.cumsum 产生的 cu_seqlens, 是打破 O(N2) 显存瓶颈、实现超长序列微调的数学入场券.

    性能瓶颈考量:
    虽然代码中包含一个 for i in range(max_num) 循环, 但在实际 Packing 场景中, max_num(即一个固定窗口内能塞进的短序列数量)通常很小(一般 < 50), 因此相对于 GPU 上的模型计算, 这个 CPU/GPU 同步转换的开销几乎可以忽略不计.
    """

    # 获取 batch_size
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device

    # 1. 探测 Batch 中最大的打包数
    # [为什么要这么写]: 不同的行可能包含不同数量的短序列.
    # 比如第一行打了 2 个包, 第二行打了 3 个包. max_num 能告诉我们计算矩阵的宽度.
    max_num = torch.max(attention_mask).item()

    # 2. 预分配计数矩阵 (Shape: [batch_size, max_num])
    # [解决的问题]: 利用矩阵化操作代替纯 Python 循环逻辑, 利用 PyTorch 的并行能力加速长度统计.
    counts: torch.Tensor = torch.zeros((bsz, max_num), dtype=dtype, device=device)

    # 3. 统计每个 ID 出现的频率
    # [逻辑解析]: 循环遍历从 1 到 max_num. 对于每个索引 i, 统计 mask 中等于 i+1 的元素个数.
    # 结果: counts[row, i] 存储了第 row 行中第 i+1 个序列的长度.
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    # 4. 拍平并过滤零值
    # [为什么要这么写]:
    # counts 矩阵中可能包含 0(例如某一行只有 2 个序列, 而 max_num 是 3, 那么第三列就是 0).
    # Padding Token (0) 统计出来的结果也是 0.
    # [解决的问题]: Flash Attention 的 varlen 接口不接受长度为 0 的序列.
    # 我们需要通过 nonzero() 找到所有真实的长度, 并将它们提取出来形成一个紧凑的一维长度向量.
    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]

    # 返回结果: 例如 [2, 3, 1, 2, 3], 代表 Batch 中按顺序排列的所有短序列长度
    return seqlens


def get_unpad_data(attention_mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", int]:
    r"""Prepare the indices and seqlens for flash attn varlen function.

    Returns:
        indices: indices of non-masked tokens from the flattened sequence.
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        max_seqlen_in_batch: the largest seqlen in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 2, 5, 6, 8, 11]
    3
    ```

    get_unpad_data 是 LLaMA-Factory 实现 高效序列打包(Sequence Packing / Multipack) 和 Flash Attention 2 变长算子(varlen functional) 适配的核心函数. 它的存在是为了彻底解决 LLM 训练中由于 Padding 填充导致的算力浪费 以及 长文本显存爆炸 的问题.

    为 Flash Attention 的 varlen (变长) 函数准备索引和序列长度信息.

    [为什么要这么写]:
    1. 消除无效计算(Unpadding): 标准的 Attention 算子在处理 [Batch, SeqLen] 张量时, 会计算所有的 Padding Token.
       对于填充较多的 Batch, 这会浪费 $O(N^2)$ 的计算量. 此函数通过提取有效 Token 的索引,
       配合 Flash Attention 2 的 varlen 接口, 实现只对"有意义"的 Token 进行计算.
    2. 解决"序列污染"问题(Packing): 在 LLaMA-Factory 中, 我们常将多个短序列(如 [Sample 1, Sample 2])
       打包在同一行. 如果按常规注意力计算, Sample 1 会关注到 Sample 2. 通过此函数生成的 `cu_seqlens`,
       Flash Attention 2 能够在物理存储连续的情况下, 逻辑上隔离不同序列的注意力范围.

    [要解决的问题]:
    - 将 2D 矩阵格式的 Batch 转换为 Flash Attention 2 要求的 1D 展平格式所需的元数据.
    - 精确标识 Batch 中每一段(包括 Pack 内部)独立序列的边界.

    Returns:
        indices: 展平后的序列中非 Mask(有效数据)Token 的原始索引.
        cu_seqlens: 当前 Batch 中所有序列长度的累积和(从 0 开始).
        max_seqlen_in_batch: 当前 Batch 中最长的一段独立序列长度.

    高级架构师视角的深度技术解析:

    内存效率(Memory Efficiency):
    在微调 70B 等超大规模模型时, 显存带宽是瓶颈. 传统的 attention_mask 方案即使使用了 Flash Attention, 如果 Batch 里有很多 0, 依然需要消耗显存存储这些 0. 通过 get_unpad_data 生成的 indices, 模型在进入 Transformer 层前就会把 hidden_states 压扁. 这使得 LLaMA-Factory 能够处理动态 Batch 大小, 并在同样的显存下微调更长的上下文.

    打破 O(N2) 屏障:
    序列打包(Packing)是 LLM 训练的"工业级秘密". 如果不配合 cu_seqlens, 模型会学习到错误的信息(比如把 Sample 2 的答案当成对 Sample 1 问题的回答). 这段代码通过精准构造累积偏移量, 使得 Flash Attention Kernel 在执行循环时能根据边界及时重置累加器.

    计算图的简洁性:
    将复杂的 2D 注意力逻辑转化为 1D 变长逻辑, 极大地简化了梯度回传的过程. 这也是为什么 LLaMA-Factory 在开启 packing 后, 训练速度(Tokens per Second)通常会有 200% - 400% 的质变提升.
    """

    # 1. 获取 Batch 中每个独立序列的长度列表
    # [为什么要这么写]: 在 Packing 模式下, attention_mask 包含序列 ID(如 1, 1, 2, 2, 2...).
    # 这个函数会返回类似 [2, 3] 的长度分布.
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)

    # 2. 提取非零(有效)Token 的索引
    # [为什么要这么写]: flatten() 后使用 nonzero().
    # [解决的问题]: 这个 `indices` 张量是后续执行 `hidden_states[indices]` 的凭证.
    # 它负责把 [Batch, SeqLen, Hidden] 的冗余张量"挤压"成 [Total_Valid_Tokens, Hidden] 的紧凑张量.
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # 3. 计算 Batch 内最大的序列长度
    # [为什么要这么写]: Flash Attention 2 的 CUDA Kernel 需要这个值来决定 GPU 线程块(Tiling)的划分.
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    # 4. 构造累积长度张量 (Cumulative Sequence Lengths)
    # [为什么要这么写]: 使用 torch.cumsum 并前置填充一个 0(F.pad(..., (1, 0))).
    # [解决的问题]: 这是 Flash Attention varlen 接口的硬性标准.
    # 例如长度为 [2, 3], `cu_seqlens` 就是 [0, 2, 5].
    # 它告诉 CUDA 内核: 第 1 个序列在索引 0-2 之间, 第 2 个序列在索引 2-5 之间.
    # 这在底层确保了"注意力计算不跨界", 是实现高性能序列打包微调的数学入场券.
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def configure_packing(model_args: "ModelArguments", is_trainable: bool) -> None:
    """
    这段代码虽然简短, 但它是 LLaMA-Factory 实现"极致训练效率"的核心黑魔法之一. 它通过 Monkey Patching(猴子补丁) 技术, 深度修改了 Hugging Face transformers 库的底层行为.

    高级研究员视角的深度解析:

    为什么必须用 Monkey Patch?
    Hugging Face 的 transformers 库虽然支持序列打包, 但其默认的 _get_unpad_data 实现通常只考虑了"有效 Token vs Padding Token"的二元关系. 它并没有原生支持在一个序列块内部存在"多个独立样本边界"的逻辑. 通过这种运行时的替换, 我们无需修改 transformers 的源码就能实现对底层算子的精准控制.

    对训练性能的质变提升:
    传统的 SFT 训练如果不用 Packing, 一个 Batch 中可能 50% 以上都是 Padding Token(0), 这白白浪费了 GPU 算力. 开启此配置后, 由于样本被紧凑打包且逻辑隔离, 训练吞吐量(Tokens per second)通常能提升 2 倍以上, 且模型效果与不打包时完全一致.

    算子依赖性:
    这段代码依赖于 Flash Attention 2. 只有 Flash Attention 2 的变长序列接口(flash_attn_varlen_func)才接受 cu_seqlens 参数. 这也是为什么在函数名和导入中都强调了 flash_attention 的原因.
    """

    # 1. 准入条件检查
    # [为什么要这么写]: 序列打包(Packing)主要用于提升训练时的吞吐量.
    # 如果处于推理模式(is_trainable=False), 或者用户没有开启 block_diag_attn 开关, 则跳过.
    if not is_trainable or not model_args.block_diag_attn:
        return

    # 2. 延迟导入与作用域隔离
    # [为什么要这么写]: 仅在确定要开启 Packing 时才导入 transformers 的底层工具包.
    # [解决的问题]: 避免在普通模式下加载不必要的底层模块, 减少内存占用,
    # 并防止在不支持 Flash Attention 的环境下提前触发导入错误.
    import transformers.modeling_flash_attention_utils

    # 3. 核心黑魔法: 劫持底层逻辑 (Monkey Patching)
    # [为什么要这么写]: 直接将 transformers 库内部定义的 `_get_unpad_data` 函数
    # 替换为我们自定义的 `get_unpad_data`(通常在本项目的前文或 utils 中定义).
    #
    # [要解决的问题 - 序列污染(Cross-attention Pollution)]:
    # 在 LLM 训练中, 为了压榨 GPU 性能, 我们会把多个短样本(如样本 A, 样本 B)打包(Pack)
    # 进一个固定长度(如 4096)的 Block 中.
    # 默认的 Flash Attention 会认为这是一个连续的文档, 导致样本 B 的 Token 会注意到样本 A 的内容.
    #
    # [解决方案]:
    # 我们自定义的 `get_unpad_data` 会利用 LLaMA-Factory 构造的带有 Segment ID 的 attention_mask,
    # 计算出每个 Block 中所有独立序列的累积长度(cu_seqlens).
    # 通过替换这个函数, 我们向 Flash Attention 注入了"边界感知"能力.
    # 结果是: 在物理存储上是连续的一个 Pack, 但在计算注意力时, 它表现为"块对角矩阵(Block Diagonal)",
    # 样本 A 和 B 在逻辑上是完全隔离的, 从而实现了无损的高效打包训练.
    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

    # 4. 日志记录
    # 告知用户当前正在使用这种高性能且无损的注意力机制.
    logger.info_rank0("Using block diagonal attention for sequence packing without cross-attention.")

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

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    """
    Initialize new token embeddings with mean + Gaussian noise.

    This is the default initialization method used by LlamaFactory.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added at the end of the embedding matrix

    在 LLM 微调中, 新增 Token(如特殊的角色占位符 <|im_start|> 或领域特定词汇)的初始化方式直接决定了微调的收敛速度和模型最终的性能.

    使用"均值 + 高斯噪声"初始化新增 Token 的嵌入向量.

    这是 LLaMA-Factory 默认的高级初始化方法.

    [为什么要这么写]:
    1. 解决"分布漂移"问题:
       如果我们使用随机初始化(如均匀分布或标准高斯分布), 新 Token 的向量会远离预训练 Token 的分布中心.
       在微调初期, 这会导致极大的 Loss, 迫使模型剧烈调整权重, 从而破坏预训练学到的知识(灾难性遗忘).
    2. 解决"冷启动"效率:
       通过计算已有 Token 的均值(avg_weight), 让新 Token 从预训练词表的"语义中心"开始演化.
    3. 解决"特征区分度"问题:
       如果所有新 Token 都只设为均值, 它们在数学上是完全相同的, 梯度更新会同步移动, 导致无法区分.
       加入适量的噪声(noise_weight)可以打破这种对称性, 让模型能更快地学习不同新 Token 之间的语义差异.

    高级研究员视角的补充建议:

    为什么不用零初始化?
    如果新 Token 向量全为 0, 在某些层(如 LayerNorm)中可能导致数值异常. 此外, 在自回归生成时, 模型会因为无法从零向量中提取有效特征而导致首个生成的 Token 出现严重的乱码.

    工程边界条件:
    注意 embed_weight[:-num_new_tokens] 的切片操作. 在 LLaMA-Factory 中, 该函数通常在 resize_token_embeddings 之后被立即调用. 此时, 新 Token 的权重默认是随机填充的, 本函数通过原地修改(Inplace Operation)将其"校准"回预训练分布.

    计算代价:
    这种初始化方法的时间复杂度是 O(V×D) (V 是原词表大小, D 是维度), 在微调开始前仅执行一次. 相比于随机初始化带来的收敛加速, 这点 CPU/GPU 计算成本几乎可以忽略不计.
    """

    # 1. 获取嵌入维度(例如 4096)
    embedding_dim = embed_weight.size(1)

    # 2. 计算已有词表的重心 (Centroid)
    # [逻辑]: 取前 [0 : 总长度 - 新增数] 个 Token 的平均值.
    # [目的]: 确保新 Token 继承了原模型的能量分布, 处于同一个数值空间内.
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)

    # 3. 构建高斯噪声
    # [逻辑]: 先创建一个空容器, 形状对应所有新增 Token.
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])

    # 4. 注入精密缩放后的正态分布噪声
    # [为什么要用 1.0 / math.sqrt(embedding_dim) 作为标准差]:
    # 这是深度学习中典型的 Xavier/Kaiming 初始化思想的变体.
    # 随着维度 d 的增加, 点积的方差会线性增长. 为了保持信号在多层网络传导时的稳定性(不爆炸也不消失),
    # 噪声的幅度必须根据维度的平方根进行缩减. 这能确保新 Token 在计算 Attention 时不会产生异常高的分数值.
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))

    # 5. 组合并应用
    # [执行]: 将中心点移动到新 Token 位置, 并叠加上述噪声.
    # [解决的问题]: 让新 Token 既具备"合理的初始语义", 又具备"进化的潜力".
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _description_based_initialization(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    descriptions: dict[str, str],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    add_noise: bool = False,
) -> None:
    """
    Initialize new token embeddings based on textual descriptions.

    For each new token, this function:
    1. Tokenizes its description text
    2. Gets embeddings of the description tokens
    3. Averages them to initialize the new token's embedding
    4. Optionally adds Gaussian noise

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added
        descriptions: Dict mapping token string to its description text
                      e.g., {"<think>": "A token representing reasoning process"}
        tokenizer: The tokenizer instance
        model: The model instance (used to get input embeddings)
        add_noise: Whether to add Gaussian noise to the initialization

    Example:
        descriptions = {
            "<|START_OF_SVG|>": "Marks the beginning of an SVG document",
            "<|END_OF_SVG|>": "Marks the end of an SVG document"
        }

    在 LLM 微调中, 新增特殊 Token(如 <thought>、<|SVG_START|>)的初始化是一个极具挑战的工程问题. 传统的随机初始化(Random Init)会让新 Token 在微调初期处于"无序状态", 导致模型需要消耗大量步骤来理解这个 Token 的基本含义, 甚至可能破坏预训练的知识空间.

    根据文本描述初始化新增 Token 的嵌入向量.

    [核心哲学]: 利用模型已有的"语义知识"来构建新 Token 的"初速度".
    [解决的问题]: 解决新 Token 的"冷启动"问题. 通过将描述文字(如"代表思维过程的标记")的语义均值
    作为初始向量, 让新 Token 在训练开始时就处于一个合理的语义空间内, 从而加速收敛并提高训练稳定性.

    高级架构师视角的深度解析:

    语义对齐的优势:
    在微调 DeepSeek 或 Llama 等模型时, 如果你添加了一个 <thought> 标记, 描述为 "Reasoning and internal monologue", 那么新 Token 的初始向量会非常靠近 "Reasoning" 这个词. 这意味着在 Attention 计算 时, 它能立即与上下文中的逻辑词产生关联. 这比随机初始化至少节省了前 50-100 个 Step 的"盲目搜索"时间.

    torch.no_grad() 的必要性:
    这是纯粹的权重预处理(Warm-up)操作. 在模型正式开始训练(即 Trainer 的 train() 被调用)之前完成. 如果不使用 no_grad(), 这一步可能会意外产生冗余的计算图, 白白浪费 CPU/GPU 内存.

    工程细节——索引管理:
    embed_weight[-num_new_tokens + i] 这种写法体现了对 PyTorch 张量操作的熟练. 在 LLaMA-Factory 中, 通过 resize_token_embeddings 扩充词表后, 新 Token 总是被追加到矩阵的最末尾. 这种负索引操作非常安全且直观.
    """
    embedding_dim = embed_weight.size(1)

    # 遍历描述字典. descriptions 的顺序必须与添加到词表后的新 Token 索引顺序严格对应
    for i, desc in enumerate(descriptions.values()):
        # 1. 将描述文本转为 Token IDs
        # [为什么要这么写]: 我们直接用模型自己的分词器去理解这段描述.
        # [解决的问题]: 确保描述文本被分解为模型最熟悉的原子语义单元.
        # Tokenize description text
        tokens = tokenizer(desc, return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            token_ids = tokens["input_ids"][0]

            # 2. 设备对齐 (Device Management)
            # [为什么要这么写]: 确保索引操作在同一硬件(CPU/GPU/NPU)上进行.
            # Move to the same device as embed_weight
            device = embed_weight.device
            token_ids = token_ids.to(device)

            # 3. 循环引用防御 (Circular Reference Defense)
            # [核心逻辑]: 过滤掉那些同样是"新加入"的 Token IDs.
            # [解决的问题]: 如果描述文本中包含另一个刚刚加入、还没初始化的新 Token,
            # 若不剔除, 会导致新 Token 的初始向量建立在另一个"垃圾向量"之上, 造成语义污染.
            # Filter out new tokens (they don't have valid embeddings yet)
            valid_token_ids = token_ids[token_ids < (len(tokenizer) - num_new_tokens)]

            if len(valid_token_ids) == 0:
                # 4. 鲁棒性回退机制 (Fallback Logic)
                # [为什么要这么写]: 如果描述里全是无法识别的词或全是新词, 则回退到"全局均值初始化".
                # [解决的问题]: 防止程序崩溃, 并保证该 Token 至少处于词表的中心位置而非零向量.
                # Fallback: use mean of all existing embeddings
                logger.warning_rank0(
                    f"Description for token {i + 1}/{num_new_tokens} contains no valid tokens. "
                    "Using mean of existing embeddings."
                )
                base_embedding = embed_weight[:-num_new_tokens].mean(dim=0)
            else:
                # 5. 语义重心提取 (Semantic Centroid Extraction)
                # [核心逻辑]: 获取描述中所有有效单词的 Embedding, 并计算它们的均值.
                # [解决的问题]: 例如, "推理"这个新 Token 的初始向量 = ("逻辑"+"思考"+"步骤") / 3.
                # 这样模型第一眼看到这个新 Token, 就能感应到它与"逻辑、思考"等词的关联性.
                # Get embeddings of description tokens and average them
                token_embeds = model.get_input_embeddings()(valid_token_ids)
                base_embedding = token_embeds.mean(dim=0)

            # 6. 对称性破缺 (Symmetry Breaking)
            # [为什么要添加噪声]: 如果两个新 Token 的描述非常相似(甚至相同),
            # 它们的初始向量会完全一样. 在神经网络中, 完全对称的神经元在某些优化器下会同步更新,
            # 导致学习缓慢. 添加适量噪声可以打破这种对称性.
            # [标准差选择]: 使用 1/sqrt(d), 遵循 Xavier 初始化原则, 保持信号方差稳定.
            # Add noise if requested (ensure correct device and dtype)
            if add_noise:
                noise = torch.randn_like(base_embedding) * (1.0 / math.sqrt(embedding_dim))
                embed_weight[-num_new_tokens + i] = base_embedding + noise
            else:
                # 索引计算: -num_new_tokens 指向第一个新 Token 的位置
                embed_weight[-num_new_tokens + i] = base_embedding


def _initialize_embeddings(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    init_method: str,
    new_special_tokens_config: Optional[dict],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
) -> None:
    """
    Single source of truth for embedding initialization.

    This function selects the appropriate initialization method and applies it.

    Args:
        embed_weight: The embedding weight matrix to initialize
        num_new_tokens: Number of new tokens added
        init_method: Initialization method ('noise_init', 'desc_init', 'desc_init_w_noise')
        new_special_tokens_config: Config dict with token descriptions (required for desc_init methods)
        tokenizer: The tokenizer instance
        model: The model instance

    在 LLM 微调中, 当我们向词表(Vocabulary)添加新的特殊 Token(如 <thought>、<|im_start|> 或特定的领域标记)时, 这些新 Token 在 Embedding 矩阵中对应的权重默认是随机初始化的. 这种"随机性"是微调初期 Loss 不稳定、模型收敛慢、甚至破坏预训练语义空间的元凶.
    这个函数的作用是作为"初始化决策中心", 根据用户提供的元数据(描述信息), 智能地将新 Token 映射到现有的语义空间中.

    词嵌入初始化的单一事实来源(Single Source of Truth).
    该函数负责根据策略选择最合适的初始化方法, 并将其应用到新添加的 Token 上.

    [为什么要这么写]:
    在大型工程中, 初始化逻辑散落在各处会导致难以维护. 通过这个决策函数, 我们确保了无论
    是简单的噪声初始化, 还是高级的语义描述初始化, 都遵循统一的入口和错误处理逻辑.

    高级研究员视角的架构解析:

    关于 init_method 的设计哲学:
    在 LLaMA-Factory 中, 我们引入 desc_init 是因为在微调 Reasoning 模型(如 DeepSeek-R1 风格)时, <thought> 等 Token 极其重要. 如果初始化不好, 模型可能在很长一段训练时间内都在试图"对齐"这个 Token 的基础表示, 而不是学习推理逻辑.

    工程健壮性:
    注意 else 块的处理. 在分布式训练中, 配置文件的丢失或路径错误很常见. 这里的回退逻辑保证了即使在配置缺失的情况下, 系统依然能产出一个"数值安全"的初始化结果(即 noise_init), 避免训练因 NaN 或 Inf 异常中断.

    内存与计算效率:
    所有的初始化都是针对 embed_weight 的原地操作(In-place operation). 这是因为大模型的 Embedding 矩阵通常非常大(如 128k 词表 * 4096 维度 ≈ 500MB), 原地操作能最小化显存和内存的峰值占用.
    """

    # 1. 语义化初始化策略 (Semantic-based Initialization)
    # [解决的问题]: 解决新 Token 的"冷启动"语义缺失问题.
    # 如果我们知道新 Token 的含义(通过 descriptions), 我们可以将其初始化在现有词表对应语义的中心.
    if init_method == "desc_init" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization (desc_init) for new special tokens")
        # 直接利用 _description_based_initialization 获取描述文本的语义均值.
        # 适用于那些含义非常明确且唯一的 Token.
        _description_based_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, add_noise=False
        )

    # 2. 带噪声的语义化初始化策略 (Semantic + Symmetry Breaking)
    # [解决的问题]: 防止"梯度对称"和"模式坍缩".
    # 如果多个新 Token 的描述非常接近, 它们可能会被初始化在完全相同的位置.
    # 在微调初期, 这可能导致它们的梯度更新方向过于一致, 难以区分彼此的细微差别.
    elif init_method == "desc_init_w_noise" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization with noise (desc_init_w_noise) for new special tokens")
        # add_noise=True 会在语义中心点基础上加上微小的高斯扰动, 实现"对称性破缺",
        # 帮助模型在微调过程中更快地为相似但不相同的 Token 学习到区分性特征.
        _description_based_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, add_noise=True
        )

    # 3. 兜底策略: 均值噪声初始化 (Fallback: Noisy Mean Initialization)
    else:
        # [防御性编程]:
        # 如果用户指定了需要描述信息的初始化方法, 但没有提供具体的描述配置文件,
        # 我们不能让程序崩溃, 也不能任由其保持完全随机(完全随机的方差通常太大).
        if init_method != "noise_init":
            logger.warning_rank0(
                f"init_method='{init_method}' requires descriptions config, falling back to 'noise_init'"
            )

        # [解决的问题]: 将新 Token 约束在预训练模型的"语义重心"附近.
        # 即使不知道具体含义, 将其初始化在所有 Token 的平均值附近, 也比随机初始化(通常是 0 均值、1 方差)
        # 要好得多, 因为这样新 Token 的模长(Norm)和分布能与旧词表保持一致, 极大降低了微调初期的 Loss 突变.
        logger.info_rank0("Using noisy mean initialization (noise_init) for new special tokens")
        _noisy_mean_initialization(embed_weight, num_new_tokens)


def resize_embedding_layer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    new_special_tokens_config: Optional[dict] = None,
    init_special_tokens: str = "noise_init",
) -> None:
    r"""
    Resize token embeddings and initialize new tokens.

    Args:
        model: The model to resize
        tokenizer: The tokenizer (used to get target vocab size)
        new_special_tokens_config: Optional dict with token descriptions for semantic initialization
        init_special_tokens: Initialization method ('noise_init', 'desc_init', 'desc_init_w_noise')

    在 LLM 微调中, 扩充词表(比如添加 <thought>、<|user|> 等特殊 Token)是一个高频操作. 然而, 在大规模分布式环境(特别是 DeepSpeed ZeRO-3)下, 直接修改词嵌入层会导致权重分片冲突. 此外, 新 Token 的初始化质量直接关系到模型微调的收敛速度.
    调整模型的词嵌入层大小, 并对新添加的 Token 权重进行智能化初始化.

    [为什么要这么写]:
    在微调过程中添加特殊 Token 后, Tokenizer 的长度会超过模型原始 Embedding 层的长度.
    如果不调整层大小, 模型遇到新 Token 会报 Index Out of Range 错误.

    高级研究员视角的架构点评:

    ZeRO-3 鲁棒性:
    这是许多微调框架(甚至包括一些主流框架)经常忽略的"坑". 在 ZeRO-3 开启时, 如果你不使用 GatheredParameters 就尝试修改 model.resize_token_embeddings, 程序会因为各进程权重状态不一致而发生死锁(Deadlock)或 Segment Fault. LLaMA-Factory 在此处的设计体现了对大规模分布式训练的深刻理解.

    数值稳定性 (Numerical Stability):
    通过 _initialize_embeddings 引入的初始化策略是 LLaMA-Factory 的一大特色. 尤其是针对 Reasoning 模型(如 DeepSeek-R1), 如果给新定义的 <thought> 标签一个好的初始语义向量(基于描述), 模型能更快地将推理行为与该标签对齐.

    计算对齐 (Warp Alignment):
    pad_to_multiple_of=64 的设定体现了极致的性能调优. 在处理像 Qwen 这种词表高达 15 万的模型时, 这种对齐能减少 GPU 的指令非对齐开销, 将 SFT 速度提升 5%-10%.
    """

    # 1. 处理 DeepSpeed ZeRO-3 分布式上下文
    # [为什么要这么写]: 在 ZeRO-3 模式下, 模型参数被切分并分布在所有 GPU 上.
    # 单个 GPU 只持有参数的一个分片, 无法直接获取或修改完整的权重矩阵.
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        # 收集输入嵌入层
        params = [model.get_input_embeddings().weight]

        # 如果输出层(LM Head)与输入层不共享权重(not tied), 则也需要收集输出层权重
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        # [解决的问题]: GatheredParameters 会触发集合通信, 将分布在各卡的参数暂时汇总到主卡.
        # modifier_rank=0 确保只有主进程有权对汇总后的参数进行原地(inplace)修改.
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        # 非 ZeRO-3 模式下, 使用空上下文, 不产生额外开销
        context_maybe_zero3 = nullcontext()

    # 获取当前 Embedding 的词表大小
    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    # 2. 判断是否需要扩容
    if len(tokenizer) > current_embedding_size:
        # [解决的问题]: 量化模型(如 4-bit/8-bit)权重是经过压缩存储的,
        # 无法直接通过 PyTorch 的 resize 操作增加行数, 否则会破坏量化结构.
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        # 确保输出层是标准的 Linear 层, 防止在某些特殊架构模型上操作失败
        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        # 3. 执行扩容操作
        # [为什么要这么写]: pad_to_multiple_of=64
        # [解决的问题]: 计算效率优化. 现代 GPU 的 Tensor Cores 在矩阵维度是 8, 32 或 64 的倍数时
        # 具有最高的计算吞吐量. 对齐到 64 可以显著提升后续训练中 LM Head 的算子执行速度.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            logger.info_rank0(
                f"Resizing embeddings: {current_embedding_size} -> {new_embedding_size} (+{num_new_tokens} tokens)"
            )

            # 4. 智能化初始化新 Token 权重 (Critical Step)
            # [为什么要这么写]: 默认的 resize 操作通常使用随机初始化.
            # [解决的问题]: 随机初始化会导致新 Token 的 Embedding 处于异常数值区间, 微调初期 Loss 会剧烈波动.
            # 这里调用 _initialize_embeddings, 根据配置(如语义描述 desc_init 或均值噪声 noise_init)
            # 让新 Token 的起点尽可能贴近已有词表的分布, 加速收敛.
            # Initialize input embeddings
            _initialize_embeddings(
                model.get_input_embeddings().weight.data,
                num_new_tokens,
                init_special_tokens,
                new_special_tokens_config,
                tokenizer,
                model,
            )

            # 5. 处理非权重共享场景
            # 如果输入和输出层不共享权重(例如 Llama-3), 必须独立对输出层的对应行执行初始化
            # Initialize output embeddings if not tied
            if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
                _initialize_embeddings(
                    model.get_output_embeddings().weight.data,
                    num_new_tokens,
                    init_special_tokens,
                    new_special_tokens_config,
                    tokenizer,
                    model,
                )

        # 同步更新配置, 确保保存模型时 metadata 正确
        model.config.vocab_size = new_embedding_size
        logger.info_rank0(f"Resized token embeddings from {current_embedding_size} to {new_embedding_size}.")

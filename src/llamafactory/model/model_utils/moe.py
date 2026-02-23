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

from typing import TYPE_CHECKING, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.misc import check_version
from ...extras.packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


def _set_z3_leaf_modules(model: "PreTrainedModel", leaf_modules: list[Union["nn.Module", str]]) -> None:
    """
    在分布式微调超大规模模型时, ZeRO-3 是必不可少的. 但如果不加干预, ZeRO-3 默认的参数分片行为有时会带来严重的性能瓶颈甚至逻辑错误.

    配置 DeepSpeed ZeRO-3 的"叶子模块", 指定哪些模块不应被进一步拆分.

    [为什么要这么写]:
    1. 版本防御逻辑 (Version Guard):
       `check_version("deepspeed>=0.13.0")` 确保环境满足要求.
       `set_z3_leaf_modules` 是 DeepSpeed 在 0.13.0 版本后引入的关键特性,
       在旧版本中调用会导致程序崩溃.

    2. 性能与解耦 (Lazy Import):
       将 `from deepspeed.utils import ...` 放在函数内部进行局部导入.
       LLaMA-Factory 支持多种后端(如 FSDP、标准 DDP). 这样写可以避免在
       未开启 DeepSpeed 的环境下加载沉重的 deepspeed 库, 减少内存占用并加快启动速度.

    [要解决的问题]:
    1. 通信开销优化 (Communication Overhead):
       ZeRO-3 默认会将模型的所有参数切片分发到所有 GPU 上. 在计算时, 它会动态地聚合参数.
       如果模块被拆得太细(例如拆分了一个非常小的线性层), 会导致频繁的网络通讯, 通讯开销远超计算收益.
       通过将其设为"叶子节点(Leaf Module)", DeepSpeed 将该模块作为一个整体进行聚合, 显著提升通讯效率.

    2. 保护自定义算子/特殊架构 (Functional Correctness):
       对于某些特殊的架构(如 MoE 专家层、特定的视觉编码器或自定义的 `torch.autograd.Function`):
       - 它们内部可能对参数有特殊的访问逻辑, ZeRO-3 细粒度的自动拦截可能会破坏其内部状态.
       - 将这些模块设为"叶子", 可以让它们在执行 `forward` 前确保内部所有参数已全量就绪, 避免计算逻辑出错.

    3. 处理混合架构 (Handling Composite Models):
       在多模态模型中, 视觉塔(Vision Tower)通常不建议进行参数级的碎片化切片.
       将视觉组件设为叶子模块, 可以在保证显存节省的同时, 维持其特征提取的稳定性.

    高级研究员视角的深度解析:

    ZeRO-3 的"精细化管理":
    传统的 ZeRO-3 就像一个激进的管家, 想把家里所有家具(模型参数)都拆散成零件存在不同房间里. 等到要用某个椅子时, 再把零件拼起来. _set_z3_leaf_modules 就像是贴个标签: "这把椅子(Leaf Module)必须整体搬运, 不要拆散它的螺丝".

    对 MoE 模型的重要性:
    在微调带有 MoE(混合专家)架构的模型时, 专家模块如果被过度分片, 会导致分布式通讯(All-Gather)的次数呈指数级增加, 导致每秒处理的 Token 数(TPS)暴跌. 将 Expert 层设为 leaf_module 是目前工业界优化 MoE 训练性能的标准黑魔法.

    开发者的防御性姿态:
    代码中使用 # type: ignore 是因为 deepspeed 作为一个频繁更新的 C++/Python 混合库, 其类型提示(Type Hints)在静态检查工具(如 Mypy)下经常不准确. 作为高级工程师, 我们了解底层逻辑, 因此通过 ignore 确保 CI/CD 流程不会因为第三方的类型定义缺陷而中断.
    """

    check_version("deepspeed>=0.13.0")

    # 局部导入以避免环境依赖污染
    from deepspeed.utils import set_z3_leaf_modules  # type: ignore

    # 显式执行注入, 告知 ZeRO-3 引擎停止在该层级及其子层级进行递归分片
    set_z3_leaf_modules(model, leaf_modules)


def add_z3_leaf_module(model: "PreTrainedModel") -> None:
    r"""
    Set module as a leaf module to skip partitioning in deepspeed zero3.

    在分布式微调超大规模模型(尤其是 MoE 混合专家模型)时, 这段代码是保证训练吞吐量(Throughput)和防止由于过度通信导致性能坍缩的关键.

    将特定的子模块设置为 ZeRO-3 的"叶子模块(Leaf Module)", 以跳过其内部参数的分片处理.

    [核心痛点]:
    DeepSpeed ZeRO-3 默认会将模型的所有参数切片(sharding)并分布到所有 GPU 上.
    在计算时, 它会动态地聚合参数. 对于 MoE 模型(如 Mixtral, DeepSeek), 专家层(Experts)数量多但
    单个专家参数相对较小. 如果 ZeRO-3 对每一个专家内部的线性层都进行独立的分片和聚合, 会导致:
    1. 通信风暴: 产生海量的微小网络通信请求, 网络延迟将远超计算时间.
    2. 计算效率极低: GPU 会频繁处于等待参数聚合的状态, 导致 TFLOPS 暴跌.

    [解决方案]:
    通过将整个 MoE Block(包含所有专家)声明为"叶子节点", 告知 ZeRO-3:
    "不要去拆分这个模块内部的细节, 把这个模块当作一个不可分割的整体进行参数管理."
    这样, 每次聚合通信都会一次性拉取整个专家组, 显著提升了通信效率(带宽利用率), 用空间换取了巨大的时间收益.

    高级研究员视角的架构点评:

    分治思想 (Divide and Conquer):
    LLM 的参数管理是一门平衡艺术. Leaf Module 并不是越多越好. 如果你把整个模型设为 Leaf, 那就变成了 ZeRO-1, 显存会爆炸. 这段代码体现了"只在关键痛点(MoE 专家层)做精细控制"的原则, 在显存节省和计算速度之间取得了完美的帕累托最优.

    延迟导入 (Lazy Import):
    在每个 if 分支内部进行 from transformers... import .... 作为资深开发, 我们知道 LLM 的依赖库非常庞大. 这种写法可以确保: 如果用户微调的是 Llama (非 MoE), 代码就不会去加载 Mixtral 或 Qwen 的建模代码, 从而显著减少 Python 冷启动时间 并避免由于缺少某些可选依赖导致的加载崩溃.

    多模态一致性:
    代码中检查 text_model_type(例如 InternVL 适配)反映了 LLaMA-Factory 的工程深度——它意识到了模型可能被嵌套在复杂的视觉/音频包装器内部, 从而确保了优化策略能下沉到最底层的核心计算模块.
    """

    # 1. 状态准入检查
    # 如果没开启 Z3, 则不需要进行这种精细化的通信干预
    if not is_deepspeed_zero3_enabled():
        return

    # 获取模型类型及潜在的多模态文本分支类型
    model_type = getattr(model.config, "model_type", None)
    text_config = getattr(model.config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", None)

    # 2. 针对不同架构的"外科手术式"适配
    # [为什么要这么写]: 不同厂商的模型在 transformers 库中的类名完全不同.
    # 我们需要根据 model_type 精准定位到那个负责"专家分发"的 Block 类.

    if model_type == "dbrx":
        # DBRX 使用 FFN 类作为专家容器
        from transformers.models.dbrx.modeling_dbrx import DbrxFFN

        _set_z3_leaf_modules(model, [DbrxFFN])

    if model_type == "deepseek_v2":
        # [解决的问题]: DeepSeek V2 经常使用自定义代码(Remote Code).
        # 对于这种动态加载的类, 我们使用字符串名称进行匹配, 以避免在导入阶段产生循环依赖或找不到类的错误.
        # deepseek v2 uses custom code
        _set_z3_leaf_modules(model, ["DeepseekV2MoE"])

    if model_type == "deepseek_v3" or model_type == "kimi_vl":
        # DeepSeek V3 架构更为复杂, 将其 MoE 核心设为叶子节点是跑通 V3 微调的工业级标准做法
        # deepseek v3 and kimi vl use custom code
        _set_z3_leaf_modules(model, ["DeepseekV3MoE"])

    if model_type == "ernie4_5_moe":
        from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Ernie4_5_MoeSparseMoeBlock])

    if model_type == "granitemoe":
        from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

        _set_z3_leaf_modules(model, [GraniteMoeMoE])

    if model_type == "glm4_moe":
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE

        _set_z3_leaf_modules(model, [Glm4MoeMoE])

    if model_type == "glm4v_moe":
        from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextMoE

        _set_z3_leaf_modules(model, [Glm4vMoeTextMoE])

    if model_type == "gpt_oss":
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

        _set_z3_leaf_modules(model, [GptOssMLP])

    if model_type == "jamba":
        from transformers.models.jamba.modeling_jamba import JambaSparseMoeBlock

        _set_z3_leaf_modules(model, [JambaSparseMoeBlock])

    if model_type == "jetmoe":
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE

        _set_z3_leaf_modules(model, [JetMoeMoA, JetMoeMoE])

    if model_type == "llama4":
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        _set_z3_leaf_modules(model, [Llama4TextMoe])

    if model_type == "mixtral":
        # Mixtral 是最早的开源 MoE 代表, 其 SparseMoeBlock 必须作为整体管理
        # 否则在 8 卡 A100 环境下, 训练速度会慢 5-10 倍
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        _set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "olmoe":
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

        _set_z3_leaf_modules(model, [OlmoeSparseMoeBlock])

    if model_type == "phimoe":
        from transformers.models.phimoe.modeling_phimoe import PhimoeSparseMoeBlock

        _set_z3_leaf_modules(model, [PhimoeSparseMoeBlock])

    if model_type == "qwen2_moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])

    if model_type == "qwen3_moe" or text_model_type == "qwen3_moe":  # internvl 3.5
        # [适配逻辑]: 考虑到像 InternVL 这样的多模态模型,
        # 它的 model_type 是 internvl, 但核心文本模型(text_config)可能是 qwen3_moe.
        # 这种嵌套检查确保了多模态模型也能享受到 MoE 的通信优化.
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if model_type == "qwen3_vl_moe":
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen3VLMoeTextSparseMoeBlock])

    if model_type in ("qwen3_omni_moe", "qwen3_omni_moe_thinker"):
        # 针对带思考能力(Reasoning)的 Omni-MoE, 其 MoE 逻辑与推理链条深度耦合
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        _set_z3_leaf_modules(model, [Qwen3OmniMoeThinkerTextSparseMoeBlock])


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    """
    这段代码的核心使命是: 屏蔽不同 MoE 模型架构在属性命名上的差异, 通过动态注入配置, 激活"负载均衡"机制, 确保模型微调时专家层不会发生"坍缩".

    高级研究员视角的架构解析:

    关于 MoE 的负载均衡(Load Balancing):
    在微调 MoE 模型时, 如果不加控制, 模型会倾向于始终选择表现最好的几个专家, 导致其他专家得不到训练(即"专家坍缩"). moe_aux_loss_coef 引入了罚项. 这段代码通过注入该系数, 确保模型在微调时依然维持专家的多样性.

    多模态嵌套处理逻辑:
    你会注意到 text_config 的大量检查. 在微调视觉语言模型(如 InternVL)时, config 根对象描述的是整个 VLM, 但真正的 MoE 结构嵌套在里面的 LLM 部分. 这段代码展示了对 复合模型架构 的深度理解, 确保补丁(Patch)能打到正确的子模块上.

    工业级健壮性:
    使用 getattr(..., None) 而不是直接访问属性, 是为了防止处理那些版本过旧、或者由第三方魔改后缺失某些字段的 config 对象. 这体现了 Python 高级开发中对"动态性"和"容错性"的追求.
    """

    # 1. 准入校验 (Guard Clause)
    # [为什么要这么写]: 辅助损失(Auxiliary Loss)仅在训练阶段用于优化路由器的选择概率.
    # [解决的问题]: 如果是非训练状态(如推理)或者用户未设置辅助损失系数, 则无需修改配置, 避免不必要的计算开销.
    if not is_trainable or not model_args.moe_aux_loss_coef:
        return

    # 获取模型主类型及多模态模型中的文本分支类型
    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", None)  # for multimodal model 针对多模态模型 (如 VLM) 的设计

    # 2. 激活路由器 Logits 输出 (Enable Router Logits)
    # [为什么要这么写]: 在 Hugging Face 模型实现中, 默认不输出 router_logits 以节省显存.
    # [解决的问题]: 负载均衡损失(Load Balancing Loss)需要根据路由器的输出计算.
    # 如果不将其设为 True, 模型 forward 时将不会返回计算辅助损失所需的张量.
    # 这里列出了遵循 Transformers 官方标准命名的主流 MoE 架构.
    if model_type in [
        "dbrx",
        "ernie4_5_moe",
        "granitemoe",
        "jamba",
        "jetmoe",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "output_router_logits", True)

    # 针对多模态复合模型(如 InternVL, GLM-4V)进行嵌套属性配置
    if text_config and getattr(text_config, "model_type", None) in [
        "glm4v_moe_text",  # glmv4_5
        "qwen3_moe",  # internvl_3_5
    ]:
        setattr(text_config, "output_router_logits", True)

    # 3. 屏蔽属性命名差异 (Architecture Agnosticism)
    # [为什么要这么写]: LLM 社区缺乏严格的 MoE 属性命名规范. 不同模型对"辅助损失权重"的变量定义各不相同.
    # [解决的问题]: 统一用户接口. 用户只需输入一个 `--moe_aux_loss_coef` 参数,
    # 代码负责将其映射到具体的模型属性名上, 避免了针对不同模型写不同微调脚本的麻烦.

    # 分类 A: 遵循标准 Transformers 定义的模型
    if model_type in [
        "ernie4_5_moe",
        "granitemoe",
        "jamba",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    # 处理多模态中嵌套的文本微调参数
    elif text_config and getattr(text_config, "model_type", None) in ["qwen3_moe"]:
        setattr(text_config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    # 分类 B: DeepSeek 家族 (使用 aux_loss_alpha)
    # [原因]: DeepSeek 团队自有的命名习惯, 通常在 DeepSeek-V2/V3 等模型中出现.
    elif model_type == "deepseek":
        setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

    # 分类 C: JetMoE 家族 (使用 aux_loss_coef)
    # [原因]: 针对基于这种特定学术架构微调的模型.
    elif model_type == "jetmoe":
        setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)


class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    """
    这段代码是针对 Qwen3-Omni 模型中 "Thinker" 模块(推理增强模块)的 Sparse MoE Block 的一种特殊实现. 与高性能推理框架(如 vLLM 或 DeepSpeed-Inference)中复杂的专家并行(Expert Parallelism)不同, 这段代码的设计初衷是极致的微调兼容性与数值稳定性.

    高级研究员视角的架构深度解析:

    为什么不使用更高效的 torch.index_select 或 einsum?
    在 LLaMA-Factory 的微调场景中, 用户可能使用 LoRA 挂载在专家内部. 这种显式的 for expert_idx in range(...) 循环虽然在推理时比"Token Dispatching"慢, 但它能完美兼容 Hugging Face PEFT 库. 如果使用重排逻辑, LoRA 的权重更新和梯度累积逻辑会变得极其复杂且容易报错.

    关于 router_logits 的处理:
    该 Block 显式返回了 router_logits. 这是为了在微调时能够注入 MoE Auxiliary Loss(辅助损失). 如果不加这个损失, 模型在微调几百步后, 可能会出现所有的 Token 都挤向某一个专家的情况(即"专家闲置"问题), 这会严重损害大模型的推理能力.

    对 Qwen3-Omni "Thinker" 的适配:
    Thinker 模块通常涉及长文本推理. 这里的 norm_topk_prob 和 torch.float 的 Softmax 保证了在处理数千个 Token 的长序列时, 数值精度不会因为深层累加而出现明显的 ϵ 偏差.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating 线性层: 用于计算每个 Token 应该路由到哪个专家的 Logits
        # [解决的问题]: 由于 MoE 层的参数量巨大, gating 层必须不带 bias 以减少微调时的参数偏移
        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # 专家列表: 包含多个独立的 MLP 模块
        # [为什么要这么写]: 使用 nn.ModuleList 确保这些专家能被 PyTorch 正确追踪梯度.
        # 在微调过程中, 我们可以通过 LLaMA-Factory 的配置灵活冻结或训练这些专家.
        self.experts = nn.ModuleList(
            [
                modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 将 Batch 和 Sequence 维度拍平, 方便进行全量矩阵运算
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 1. 计算路由概率 (Routing/Gating)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # [为什么要这么写]: 显式指定 dtype=torch.float
        # [解决的问题]: 在混合精度微调(FP16/BF16)时, Softmax 在低精度下极易溢出.
        # 强制在 FP32 下计算 Softmax 能确保路由权重的数学准确性, 避免模型训练崩坏.
        # Calculate the routing weights for all experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 2. Top-K 筛选逻辑 (Sparsity Strategy)
        # 获取得分最高的 K 个专家的权重和索引
        # Retain the weight of the top_k and reset the rest of the expert rights to 0 (instead of retaining only top_k experts)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # [为什么要这么写]: 构造一个全量权重的"稀疏占位符"矩阵
        # [解决的问题]: 在微调场景下, 直接使用索引选择(index_select)虽然快, 但在某些分布式环境(如 DeepSpeed ZeRO-2)
        # 中可能会导致梯度对齐困难. 通过 scatter_ 构造完整矩阵并乘以 0 的方式,
        # 能保持计算图的连续性, 虽然牺牲了少量计算量, 但换取了极高的训练稳定性.
        # Initialize the all-zero weight matrix (same shape as all experts)
        full_routing_weights = torch.zeros_like(routing_weights)
        # Only the weight of top_k experts is retained, and the weight of the rest of the experts remains at 0
        full_routing_weights.scatter_(1, top_k_indices, top_k_weights)

        # 3. 概率归一化 (Prob Normalization)
        # [解决的问题]: Top-K 筛选后, 保留下来的权重之和不再为 1.
        # 重新归一化可以防止模型在多层堆叠后由于激活值数值漂移导致的梯度爆炸.
        # Normalized top_k weights (keep the original logic consistent)
        if self.norm_topk_prob:
            # Calculate the sum of the weights top_k each row (for normalization)
            top_k_sum = full_routing_weights.sum(dim=-1, keepdim=True)
            # Avoid dividing by zero
            top_k_sum = torch.clamp(top_k_sum, min=1e-9)
            full_routing_weights /= top_k_sum

        # 转换回模型训练的原始精度(如 BF16), 以进行后续的 MLP 运算
        # Convert back to the input data type
        full_routing_weights = full_routing_weights.to(hidden_states.dtype)

        # 预分配结果输出张量
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 4. 专家并行计算模拟 (Expert Aggregation Loop)
        # [为什么要这么写]: 采用显式循环遍历专家, 而不是复杂的 Token 重排 (Token Shuffling)
        # [解决的问题]:
        # A. 兼容性: 这解决了在非标准硬件(如国产 NPU 或老款 GPU)上自定义 MoE Kernel 不支持的问题.
        # B. 显存优化: 在微调时, 这种写法配合梯度检查点(Gradient Checkpointing)能更稳定地回收中间显存.
        # C. 实现逻辑: 对于每个专家, 我们计算全量 Token 在该专家下的输出, 并乘以其路由权重.
        #    对于未选中的 Token, 其权重为 0, 相当于直接过滤.
        # Go through all the experts (not just the selected ones)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # 提取当前专家对所有 Token 的权重: (batch*seq, 1)
            # Get the weight of the current expert (inactive expert has a weight of 0 here)
            expert_weights = full_routing_weights[:, expert_idx, None]  # shape: (batch*seq, 1)

            # 计算当前专家的贡献
            # 只有当 expert_weights > 0 时才有实际输出, 这种写法确保了梯度的正确回传
            # All samples participate in the calculations of the current expert, the weight may be equal to 0
            current_hidden_states = expert_layer(hidden_states) * expert_weights
            # Add-up to all expert outputs (experts with a weight of 0 do not affect the result)
            final_hidden_states += current_hidden_states

        # 还原维度: (batch*seq, dim) -> (batch, seq, dim)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # 返回结果和 logits(用于计算负载均衡损失 Load Balancing Loss, 防止专家坍缩)
        return final_hidden_states, router_logits

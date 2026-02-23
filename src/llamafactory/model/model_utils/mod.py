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

from typing import TYPE_CHECKING

from ...extras.constants import MOD_SUPPORTED_MODELS


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


def load_mod_pretrained_model(**init_kwargs) -> "PreTrainedModel":
    """
    这段代码虽然极简, 但它代表了对 Google DeepMind 提出的 MoD 架构(动态计算分配)的集成支持. 在 LLM 领域, 这种技术通过让模型学会"跳过"某些层来节省计算资源.

    使用 MoD(Mixture-of-Depths)库加载预训练模型.

    [为什么要这么写]:
    1. 延迟导入 (Lazy Import): 将 from MoD import ... 放在函数内部.
    2. 特定类分发: 使用 AutoMoDModelForCausalLM 而不是 Transformers 官方的 AutoModelForCausalLM.

    高级研究员视角的架构深度解析:

    MoD 的技术本质:
    MoD(深度混合)允许模型在推理和训练时, 针对每一个 Token 动态决定是否通过当前的 Transformer 层. 对于某些"简单"的 Token, 模型可以跳过复杂的计算, 从而在保持性能的同时大幅提升预训练和推理的速度.

    解耦设计思想:
    LLaMA-Factory 作为一个通用的微调平台, 其核心 load_model 函数非常庞大. 通过将 MoD 的加载逻辑拆分成 load_mod_pretrained_model 这样的独立工具函数, 实现了插件化架构. 未来如果出现类似的 Mixture-of-Width 或其他异构架构, 只需增加类似的独立加载函数即可, 不会污染主干加载流程.

    对 init_kwargs 的透传:
    这里的参数透传非常关键. 它确保了 LLaMA-Factory 所有的底层黑魔法(比如 BitsAndBytes 量化、Flash Attention 2 开启、设备自动切分 device_map)都能无缝地传递给 MoD 模型, 让这种前沿架构也能享受到成熟的微调基础设施.
    """

    # 1. 处理第三方可选依赖 (Dependency Isolation)
    # [为什么要这么写]: MoD 并不是所有微调任务的必备库.
    # [解决的问题]: 采用局部导入(Local Import), 确保那些不使用 MoD 功能的用户
    # 不需要安装额外的 MoD 依赖包. 这降低了项目的环境准入门槛, 避免了"未安装依赖即报错"的糟糕体验.
    from MoD import AutoMoDModelForCausalLM

    # 2. 专用架构适配 (Specialized Architecture Handling)
    # [为什么要这么写]: 调用 AutoMoDModelForCausalLM.from_pretrained.
    # [解决的问题]: Mixture-of-Depths 架构在 Transformer 的层中引入了"动态路由"和"Token 选择(Router)"机制.
    # 标准的 Transformers 库目前无法完全解析这些带有特殊计算跳过逻辑的配置.
    # 通过使用专用的 Auto 类, 可以确保:
    #   - 正确识别模型中的 Router 权重.
    #   - 正确初始化层间跳过(Skip connection)的数学逻辑.
    #   - 兼容原本为标准 Transformer 设计的 init_kwargs(如 device_map, torch_dtype).
    return AutoMoDModelForCausalLM.from_pretrained(**init_kwargs)


def convert_pretrained_model_to_mod(
    model: "PreTrainedModel", config: "PretrainedConfig", model_args: "ModelArguments"
) -> "PreTrainedModel":
    """
    在 LLM 前沿研究中, Google 提出的 MoD 是一种动态计算分配技术. 它允许模型在处理不同 Token 时, 动态决定是否跳过某些 Transformer 层, 从而在不损失性能的前提下大幅降低推理和训练的计算开销.

    将标准架构的预训练模型(如 Llama)转换为支持 Mixture-of-Depths (MoD) 架构的模型.

    [为什么要这么写]:
    由于绝大多数开源预训练模型(Base Model)在设计时采用的是静态的全量计算架构,
    如果要实现 MoD 效果, 必须在不破坏原有权重的平衡下, 动态注入路由(Router)逻辑.

    高级研究员视角的架构深度解析:

    关于转换时机(In-memory Conversion):
    这段代码体现了 LLaMA-Factory 的"即时修改"设计理念. 我们不要求用户手动去修改模型源码文件, 而是在模型加载到内存后、正式训练前, 动态地修改 Python 对象. 这种做法极大地降低了用户尝试前沿技术的成本(只需改一个配置参数, 无需改代码).

    MoD 的研究价值:
    对于高级研究员来说, 使用这个转换器可以将现有的强大 Base Model(如 Llama-3-8B)快速转化为一个带有 MoD 能力的测试床. 通过后续的微调, 你可以观察模型在哪些 Token 上会选择"偷懒"(跳过计算), 这对于理解 LLM 的内部冗余性非常有帮助.

    工程上的健壮性:
    注意到代码中使用了 getattr(config, "model_type", None). 在高级开发中, 永远不要假设 config.model_type 一定存在, 这种防御性写法能有效避免在处理非标准或受损的配置文件时直接崩溃, 并提供更友好的错误信息.
    """

    # 1. 局部依赖导入 (Lazy Loading & Optional Dependency)
    # [为什么要这么写]: 将 'from MoD import ...' 放在函数内部.
    # [解决的问题]: MoD 是一个可选的高级特性. 这样做可以确保那些不需要 MoD 功能的用户
    # 无需安装额外的 MoD 依赖库. 这是大型 Python 项目保持"插件化"和"轻量化"的标准做法.
    from MoD import apply_mod_to_hf

    # 2. 架构兼容性硬校验 (Architecture Guardrail)
    # [为什么要这么写]: 通过检测 config 中的 model_type 是否在支持列表中.
    # [解决的问题]: MoD 转换涉及到对 Transformer 层结构的深度"手术"(注入预测器和路由),
    # 不同模型的 Layer 命名规范(如 self_attn vs attention)和残差结构不同.
    # 提前报错可以防止在后续复杂的分布式训练中出现难以排查的维度不匹配或属性错误.
    if getattr(config, "model_type", None) not in MOD_SUPPORTED_MODELS:
        raise ValueError("Current model is not supported by mixture-of-depth.")

    # 3. 执行核心逻辑转换 (Core Logic Transformation)
    # [解决的问题]: 调用 apply_mod_to_hf.
    # 该步骤会遍历模型的每一层, 插入 Router 模块, 并将模型重新包装.
    # 它解决了将"静态计算流"转变为"基于权重的动态丢弃计算流"的数学转换问题.
    model = apply_mod_to_hf(model)

    # 4. 计算精度重校准 (Precision Re-alignment)
    # [为什么要这么写]: 转换完成后显式调用 .to(model_args.compute_dtype).
    # [解决的问题]: 转换过程中新注入的参数(如 Router 层的初始化权重)可能是默认的 fp32.
    # 在混合精度训练(如 bf16 或 fp16)中, 如果不同层的 dtype 不统一, 会导致:
    #   a. 触发复杂的自动转换逻辑, 降低训练速度.
    #   b. 在分布式环境下(如 DeepSpeed)引发梯度同步的精度冲突.
    # 这一步确保了转换后的"新"模型在算力平台上能以最高性能运行.
    model = model.to(model_args.compute_dtype)
    return model

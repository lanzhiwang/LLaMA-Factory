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

from ...extras import logging
from .visual import COMPOSITE_MODELS


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> list[str]:
    r"""
    Find all available modules to apply LoRA, GaLore or APOLLO.

    find_all_linear_modules 函数是实现 "自动化微调" 的核心黑魔法之一.
    在 LLM 领域, LoRA(低秩自适应)通常默认只应用于 q_proj 和 v_proj, 但研究表明, 将 LoRA 应用于模型中所有的线性层(All-Linear LoRA)往往能获得更好的微调效果.

    这段代码解决的核心痛点是: 如何自动跨越不同厂商(Meta, 阿里, 智谱等)千差万别的命名规范, 精准找到所有适合挂载 LoRA 适配器的层, 同时避开那些"禁区"层.

    自动查找模型中所有可应用 LoRA、GaLore 或 APOLLO 的线性模块.

    [解决的问题]:
    1. 避免硬编码: 不同模型(如 Llama, Qwen, ChatGLM)的线性层命名不一, 手动指定极易出错.
    2. 提高微调质量: 研究证明全线性层微调效果通常优于部分微调.

    高级研究员视角的架构总结:

    动态探测机制: 这段代码的核心思想是 "Detection over Configuration"(探测优于配置). 它不强制要求用户知道模型内部有什么, 而是通过反射(Reflection)机制自动分析模型拓扑.
    解耦多模态复杂性: 通过 COMPOSITE_MODELS 的引用, 将纯文本微调逻辑与视觉、音频等多模态微调逻辑解耦. 即使未来增加了 3D 点云模型, 只需要在 COMPOSITE_MODELS 注册其关键键名, 该函数即可自动适配.
    工业级健壮性: 代码中对 Embedding 层的排除和对各厂家输出层名称的兜底, 反映了在微调数万个不同开源模型后总结出的工程实践经验, 极大降低了因 RuntimeError 导致的训练中断风险.
    """

    model_type = getattr(model.config, "model_type", None)

    # 1. 建立"禁止微调"黑名单 (Forbidden List)
    # [为什么要这么写]: 输出层(lm_head)通常负责将特征映射回巨大的词表.
    # 如果给这一层挂载 LoRA, 会导致极大的显存开销, 且在某些量化场景(如 QLoRA)中不稳定.
    forbidden_modules = {"lm_head"}

    # 针对特定厂商命名的兼容性处理 (Vendor-Specific Quirk Handling)
    # [解决的问题]: 模型厂商的自由散漫. ChatGLM 把输出层叫 "output_layer", InternLM2 叫 "output".
    # 通过这种方式实现底层架构的"去差异化".
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")

    # 2. 多模态架构下的模块隔离 (Multimodal Isolation)
    # [为什么要这么写]: 针对 VLM(视觉语言模型, 如 LLaVA).
    # COMPOSITE_MODELS 包含了视觉编码器与语言模型之间的"连接器(Projector)".
    # Projector 通常需要特定的微调策略(通常是全量微调或完全冻结), 不适合直接挂载通用的 LoRA.
    if model_type in COMPOSITE_MODELS:
        forbidden_modules.add(COMPOSITE_MODELS[model_type].projector_key)

    # 3. 视觉塔(Vision Tower)的动态处理
    # [解决的问题]: 在微调视觉大模型时, 通常希望保护预训练好的视觉特征(冻结视觉塔).
    # 如果用户选择了 freeze_vision_tower, 则将视觉组件的所有子模块加入黑名单.
    if freeze_vision_tower and model_type in COMPOSITE_MODELS:
        forbidden_modules.update(COMPOSITE_MODELS[model_type].vision_model_keys)

    module_names = set()

    # 4. 深度遍历模型图 (Recursive Graph Traversal)
    # [为什么要这么写]: 使用 PyTorch 的 named_modules() 遍历所有子组件.
    for name, module in model.named_modules():
        # 检查当前模块是否包含黑名单中的字符串关键字(如命中 lm_head 部分路径)
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        # 5. 精准类型匹配 (Precise Type Matching)
        # [解决的问题]: 识别真正的线性算子.
        # 检查类名中是否包含 "Linear", 但排除 "Embedding"(Embedding 虽然在底层也是线性映射,
        # 但在 LoRA 逻辑中需要特殊处理, 不能作为普通的 Linear 层对待).
        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            # 仅提取模块路径的末尾名称(例如: model.layers.0.self_attn.q_proj -> q_proj)
            # [为什么要这么写]: PEFT 库(LoRA 的底层实现)通常通过后缀匹配来统一为所有层的相同模块应用适配器.
            module_names.add(name.split(".")[-1])

    # 打印日志: 这是工业级代码的习惯, 让用户在启动训练前通过终端确认被选中的层, 方便排查由于命名冲突导致的"漏掉层"问题.
    logger.info_rank0("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def find_expanded_modules(model: "PreTrainedModel", target_modules: list[str], num_layer_trainable: int) -> list[str]:
    r"""
    Find the modules in the expanded blocks to apply lora.

    这段代码 find_expanded_modules 是专门为 LLaMA Pro 以及类似的 "块扩展(Block Expansion)" 技术设计的.
    背景背景: 为什么要写这段代码?
    在 LLaMA Pro 论文中, 研究员提出了一种不破坏原始模型能力而扩展其知识的方法: 在原始模型的 N 层之间均匀地插入 M 层新初始化层. 微调时, 我们通常只希望对这些"新加入的层"应用 LoRA, 而冻结原始层, 从而实现极其高效的知识迁移.
    这段代码解决的核心问题是: 在模型结构被改变后, 如何通过数学规律自动定位这些特定的"扩展层"并为其挂载 LoRA 适配器.

    在扩展后的模型块中寻找特定的模块, 用于精准应用 LoRA 适配器.

    [解决的问题]:
    当使用 LLaMA Pro 等技术对模型进行深度扩展(例如从 32 层扩展到 40 层)后,
    如果直接对 "all" 模块应用 LoRA, 会造成算力浪费且可能干扰原始权重.
    该函数通过计算"步长(Stride)", 自动识别出哪些层是新增的"扩展层".

    高级研究员视角的深度解析:

    关于 stride - 1 的设计:
    在 LLaMA Pro 的实现中, 新增的层是插在原始块之后的. 代码中使用 stride - 1 作为起始点(例如步长为 5, 则从索引 4 开始), 是因为程序员习惯使用 0-based 索引, 这样能精准捕捉到每组扩展块中的"最顶层".

    避免全量扫描带来的副作用:
    传统的 LoRA 设置 target_modules=["q_proj"] 会在所有层应用. 通过这个函数, LLaMA-Factory 实际上生成了一个显式的白名单. 这解决了在进行"架构感知型微调"时, 参数更新范围失控的问题.

    计算效率:
    在处理 70B 甚至更大参数规模的模型时, model.named_modules() 产生的列表非常庞大. 该函数通过 any() 的短路逻辑和预生成的 trainable_layers 字符串进行筛选, 虽然是 O(N) 复杂度, 但在 Python 层面已经是执行效率与可读性的平衡点.
    """

    # 1. 获取总层数
    # [为什么要这么写]: num_hidden_layers 是 Transformer 架构的标准元数据.
    # 需要以此为基数来计算扩展层的分布.
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not num_layers:
        raise ValueError("Model was not supported.")

    # 2. 逻辑合法性检查
    # [解决的问题]: 确保用户请求的可训练层数与模型总层数具有整除关系.
    # LLaMA Pro 的扩展是均匀插入的, 如果不能整除, 说明扩展逻辑或参数输入有误.
    if num_layers % num_layer_trainable != 0:
        raise ValueError(
            f"`num_layers` {num_layers} should be divisible by `num_layer_trainable` {num_layer_trainable}."
        )

    # 3. 计算步长 (Stride) 与 目标层索引 (Layer IDs)
    # [为什么要这么写]: 假设原始模型 32 层, 扩展后 40 层, 新增了 8 层.
    # 那么步长 Stride = 40 / 8 = 5.
    # 按照 LLaMA Pro 的设计, 新增层通常位于每个 Block 的末尾.
    # 例如 Stride=5 时, 新增层索引为 4, 9, 14, 19, 24, 29, 34, 39.
    stride = num_layers // num_layer_trainable
    trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)

    # 4. 构建层级匹配字符串
    # [为什么要这么写]: 在 PyTorch 的 named_modules 中, 层通常表示为 ".layers.4.".
    # 预先生成 ".4.", ".9." 这种特征字符串, 可以快速在模型树中进行路径匹配.
    trainable_layers = [f".{idx:d}." for idx in trainable_layer_ids]
    module_names = []

    # 5. 精准筛选模块名 (Full Module Path Discovery)
    # [为什么要这么写]: 遍历模型中所有的子模块.
    # [解决的问题]: PEFT (LoRA) 库通常根据模块名后缀匹配(如 "q_proj").
    # 但我们现在需要的是"路径+名称"的精准匹配(例如 "model.layers.4.self_attn.q_proj").
    # 通过双重逻辑判定:
    #   a. 模块名必须在用户指定的 target_modules(如 q_proj, v_proj)中.
    #   b. 模块所属的层索引必须在我们计算出的 trainable_layers 扩展层列表中.
    for name, _ in model.named_modules():
        if any(target_module in name for target_module in target_modules) and any(
            trainable_layer in name for trainable_layer in trainable_layers
        ):
            module_names.append(name)

    # 打印日志: 告知用户具体的训练范围, 这是工业级代码透明化的体现.
    logger.info_rank0("Apply lora to layers: {}.".format(",".join(map(str, trainable_layer_ids))))

    # 返回的是完整的模块路径列表, 后续会传给 LoraConfig 的 target_modules 参数.
    return module_names


def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    """
    在 Hugging Face 的 transformers 生态系统中, 这几行代码涉及到一个非常高级且重要的特性: 动态代码分发与自定义模型序列化. 如果你微调的是 Qwen、DeepSeek 或 Mixtral 这种带有自定义建模代码(trust_remote_code=True)的模型, 这段代码就是确保你训练完的模型能被别人"一键加载"的关键.

    将当前使用的配置、模型和分词器类注册到 Hugging Face 的 Auto 映射中.

    [为什么要这么写]:
    Hugging Face 的 `AutoModel` 等工厂类通过读取配置文件中的 `auto_map` 字段来决定使用哪个具体的类.
    当我们在 LLaMA-Factory 中对模型进行微调, 或者处理那些带有自定义建模脚本(即远程代码)的模型时,
    模型类(Class)往往是动态加载的.

    [解决的问题]:
    1. 解决"模型加载孤岛"问题: 如果你微调了一个非原生 Transformers 库支持的模型(比如一个全新的架构),
       如果不进行注册, 你保存后的模型在执行 `AutoModel.from_pretrained()` 时会因为找不到对应的类而报错.
    2. 增强可移植性: 通过 `register_for_auto_class()`, 我们会把当前的 Python 类逻辑与保存的权重绑定.
       这样, 用户在加载你微调的模型时, 不需要额外安装复杂的插件, HF 会自动从你的模型目录中寻找正确的类.
    3. 处理猴子补丁(Monkey Patching): LLaMA-Factory 经常会在运行时动态修改模型逻辑.
       通过注册, 我们可以确保被修改后的"增强版"类名被记录在案, 从而保证推理阶段与训练阶段的行为一致性.

    高级研究员视角的深度解析:

    关于 auto_map 的核心痛点:
    在 LLM 工业界, 很多厂商(如 智谱 GLM, 阿里 Qwen)为了快速迭代, 并没有直接把模型代码合并进 transformers 主仓库, 而是放在模型仓库的 .py 文件里. 当 LLaMA-Factory 加载这些模型时, 这些类在内存中是"孤立"的. 如果不执行 register_for_auto_class, 你导出的 Checkpoint 里的 auto_map 字段虽然还在, 但它指向的类可能与你实际运行时的类产生脱节.

    安全性与便利性的平衡:
    这段代码实际上是为 trust_remote_code=True 做铺垫. 注册行为保证了模型权重的"自解释性". 这意味着你微调后的模型发给别人, 别人只需要通过标准 API 就能运行, 而不需要去关心你当时用了什么版本的代码.

    工程细节:
    注意代码中使用了 getattr(config, "auto_map", {}) 这种防御性编程. 这是因为很多原生模型(如 Llama-2 正式版)并没有 auto_map(因为它们的代码已经内置在库里了). 这种写法兼容了"原生模型"与"自定义动态模型", 体现了 LLaMA-Factory 极强的生态普适性.
    """

    # 1. 注册配置类(Config)
    # 检查 config.json 中是否定义了 auto_map. 如果定义了 "AutoConfig": "custom_config.MyConfig",
    # 则调用 register_for_auto_class 将 MyConfig 这个 Python 类注入到 HF 的全局查找表中.
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()

    # 2. 注册模型类(Model)
    # 对于 CausalLM(因果语言模型), 如果它包含自定义架构,
    # 注册后可以确保 `AutoModelForCausalLM.from_pretrained("your_finetuned_path")` 能直接工作.
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()

    # 3. 注册分词器类(Tokenizer)
    # Tokenizer 的注册信息通常存储在 `tokenizer_config.json` 衍生的 `init_kwargs` 中.
    # 解决部分模型使用自定义 FastTokenizer 或特殊处理逻辑导致加载失败的问题.
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

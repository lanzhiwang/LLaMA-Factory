# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import transformers
import transformers.models
from transformers.activations import ACT2FN

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)
transformers_logger = transformers.utils.logging.get_logger(__name__)


@dataclass
class CompositeModel:
    """
    在多模态大模型(VLM, 如 LLaVA、InternVL、Qwen-VL)兴起的背景下, 模型不再仅仅是一个单一的语言模型(LLM), 而是由视觉编码器(Vision Tower)、投影层(Projector)和语言模型(Language Model)组合而成的复合体.
    这段代码的核心使命是: 提供一套标准化的元数据抽象, 使得微调框架能够跨越不同厂商的架构差异, 精准地定位并操控多模态模型内部的子组件.

    复合模型元数据类.

    [为什么要这么写]:
    多模态模型(如 LLaVA)的结构通常是嵌套的. 不同厂商对组件的命名极度不统一(例如: 有的叫 `mm_projector`, 有的叫 `mlp_adapter`).
    通过这个 Dataclass, 我们将这些碎片化的命名逻辑抽象化, 实现一套代码适配百模.

    高级研究员视角的架构深度解析:

    解耦"是什么"与"在哪里":
    在多模态研究中, 我们最关心的逻辑是: "冻结 Vision Tower, 训练 Projector, 对 Language Model 做 LoRA".
    如果没有这个 CompositeModel 类, 代码里会充斥着大量的 if model_type == "llava": ... elif model_type == "qwen_vl": ....
    有了它, 我们只需定义一个映射表(Registry), 主逻辑就变成了通用的 model.get_projector().requires_grad_(True).

    动态路径寻址的鲁棒性:
    get_projector 中的 split(".") 循环是高级 Python 工程师处理复杂对象树的标准做法. 它不仅解决了嵌套问题, 还通过 getattr 保持了代码的动态性. 这对于适配那些通过 trust_remote_code=True 加载的非标准架构(即建模代码不在库里而在本地文件里)的模型至关重要.

    LoRA 适配的防御性编程:
    lora_conflict_keys 体现了工业级微调框架的严谨. 在 PEFT 寻找线性层(Linear layers)进行替换时, 如果复合模型内部存在共享参数或循环引用的层(多模态模型常有此类 Hack), PEFT 可能会崩溃. 这个列表提供了一个"手动纠偏"的接口, 大大降低了微调新出炉模型时的排雷成本.
    """

    # 模型架构的唯一标识符(对应 config.model_type)
    model_type: str

    # 投影层(Projector)的路径
    # [解决的问题]: Projector 负责将视觉特征映射到文本空间, 是多模态对齐微调(Alignment)的核心.
    # 它的命名路径在不同模型中差异巨大(如 "model.mm_projector" 或 "visual.projector").
    projector_key: str

    # 视觉模型组件的关键字列表
    # [解决的问题]: 在微调过程中, 为了节省显存和保持稳定性, 通常需要冻结视觉编码器(Vision Tower).
    # 这个列表告诉框架哪些层属于视觉部分, 从而实现"外科手术式"的梯度冻结.
    vision_model_keys: list[str]

    # 语言模型组件的关键字列表
    # [解决的问题]: 定义 LLM 核心部分的范围. 当用户只想微调文本部分或对文本应用 LoRA 时,
    # 框架需要通过这些 Key 来定位具体的层级.
    language_model_keys: list[str]

    # LoRA 冲突关键字列表
    # [为什么要这么写]: 在使用 PEFT (LoRA) 注入适配器时, 某些复合模型的层可能会出现命名空间冲突,
    # 或者某些特殊的层(如特定的位置编码层)不能挂载 LoRA.
    # [解决的问题]: 通过黑名单机制屏蔽这些冲突 Key, 确保 LoRA 补丁能够稳定注入, 不会引发运行时错误.
    lora_conflict_keys: list[str]

    def get_projector(self, module: "torch.nn.Module") -> "torch.nn.Module":
        """
        根据 projector_key 的路径递归获取投影层模块实例.

        [为什么要这么写]:
        在 PyTorch 中, 子模块往往是多层嵌套的(例如: model.language_model.model.projector).
        直接使用 getattr(module, "a.b.c") 会失败, 因为 getattr 不支持点号分隔符.

        [解决的问题]:
        实现了类似 Xpath 的动态寻址能力. 它将字符串路径(如 "model.mm_projector")拆分,
        逐级深入搜索, 确保无论投影层埋藏得有多深, 微调框架都能动态地抓取到该实例,
        以便后续进行层级冻结、参数监控或精度转换.
        """
        for key in self.projector_key.split("."):
            module = getattr(module, key)

        return module


COMPOSITE_MODELS: dict[str, "CompositeModel"] = {}


def _register_composite_model(
    model_type: str,
    projector_key: Optional[str] = None,
    vision_model_keys: Optional[list[str]] = None,
    language_model_keys: Optional[list[str]] = None,
    lora_conflict_keys: Optional[list[str]] = None,
):
    r"""Register a new composite model.

    Args:
        model_type: model type
        projector_key: multi_modal_projector
        vision_model_keys: vision_tower
        language_model_keys: language_model
        lora_conflict_keys: None

    """
    COMPOSITE_MODELS[model_type] = CompositeModel(
        model_type=model_type,
        projector_key=projector_key or "multi_modal_projector",
        vision_model_keys=vision_model_keys or ["vision_tower"],
        language_model_keys=language_model_keys or ["language_model", "lm_head"],
        lora_conflict_keys=lora_conflict_keys or [],
    )


class LlavaMultiModalProjectorForYiVL(torch.nn.Module):
    """
    在视觉语言模型(VLM)中, Projector 是连接"视觉世界"与"语言世界"的桥梁. 这段代码之所以特殊, 是因为 Yi-VL 模型在架构设计上并没有采用标准的 LLaVA 简单 MLP 结构, 而是采用了一套更深、带有归一化层的对齐网络.

    高级研究员视角的架构解析(Deep Dive):

    为什么 Yi-VL 使用 LayerNorm 而不是 RMSNorm?
    虽然 LLM 主干网络(如 Llama)常用 RMSNorm, 但视觉特征在初次映射时往往包含更复杂的统计分布. LayerNorm 包含均值平移(Mean Shift), 在模态对齐阶段比 RMSNorm 更有助于捕捉视觉特征的整体偏移量.

    关于精度回拨的必要性:
    作为资深工程师, 我们需要特别注意 torch.nn.LayerNorm 在某些 PyTorch 版本下会强制输出 float32 以保证求和开方的精确度. 如果这段代码不加最后的 to(target_dtype), 那么拼接后的 input_embeds 会变成混合精度, 导致在后续执行 FlashAttention 时由于输入不支持 FP32 而报出 RuntimeError.

    对 _pre_quantization_dtype 的支持:
    这是 LLaMA-Factory 的工程细节. 在处理 QLoRA 或 4-bit 量化模型时, 权重本身是 int4 压缩的. 为了计算, 我们需要知道它对应的浮点精度是什么. 这里的判断逻辑确保了即使在极致压缩的微调场景下, Projector 输出的特征依然能保持高质量的表示.
    """

    def __init__(self, config: "LlavaConfig") -> None:
        super().__init__()

        self.config = config
        if config is None:
            return

        # 1. 深度对齐架构设计 (Yi-VL 特有的 Multi-stage Projection)
        # [为什么要这么写]: Yi-VL 的设计者认为视觉特征到文本空间的映射需要更强的非线性表达能力.
        # [解决的问题]: 相比标准 LLaVA 的简单两层 MLP, 这里引入了额外的线性层和 LayerNorm.
        # linear_1: 负责维度的初步映射(Vision Dim -> Text Dim).
        self.linear_1 = torch.nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        # linear_2 (LayerNorm): 在非线性激活前进行归一化.
        # [解决的问题]: 防止由于视觉编码器(CLIP)输出分布与 LLM 输入分布差异过大导致的梯度消失或爆炸.
        self.linear_2 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        # linear_3 & linear_4: 进一步精炼映射后的特征, 使其在语义上更贴近 LLM 的 Embedding 空间.
        self.linear_3 = torch.nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        # 动态获取激活函数(通常为 GELU)
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, image_features: "torch.Tensor") -> "torch.Tensor":
        # 顺序执行映射流程
        hidden_states = self.linear_1(image_features)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)

        # 2. 数值稳定性与精度自动修正逻辑 (Numerical Stability & Auto-precision Correction)
        # [为什么要这么写]: 在分布式训练(如 DeepSpeed 或 FSDP)开启 Autocast 时,
        # 算子可能会在计算过程中自动将中间变量提升为 float32.
        # [解决的问题]:
        #   a. 显存压力: 如果返回 FP32 张量给 LLM, 后续的 Attention 计算会因为类型不匹配或内存占用过高而崩溃.
        #   b. 精度对齐: LLM 的核心部分通常是以 BF16 或 FP16 运行. 这里强制将 Projector 的输出
        #      转换回目标精度, 确保它能与文本 Embedding 无缝拼接.
        if hidden_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                # 优先跟随当前环境的自动混精设置
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                # 如果是量化模型, 回退到量化前预设的原始精度
                target_dtype = self.config._pre_quantization_dtype
            else:
                # 默认回退到权重的精度(通常是 BF16/FP16)
                target_dtype = self.linear_1.weight.dtype

            # transformers_logger.warning_once 提示开发者:
            # 存在隐式的精度转换, 这对排查训练中的数值不稳定性非常有帮助.
            transformers_logger.warning_once("The hidden states seems to be silently casted in float32.")
            hidden_states = hidden_states.to(target_dtype)

        return hidden_states


class LlavaMultiModalProjectorForYiVLForVLLM(LlavaMultiModalProjectorForYiVL):
    """
    这段代码出现在 LLaMA-Factory 中, 反映了在 LLM 工程化中一个非常重要的环节: 模型在"微调框架(Hugging Face 体系)"与"高性能推理引擎(vLLM 体系)"之间的架构对齐.

    专门为 vLLM 推理引擎适配的 Yi-VL 多模态投影器.

    [为什么要这么写]:
    1. 解耦配置依赖: vLLM 在初始化模型组件时, 往往倾向于传递基础的维度参数(int)而非整个
       Hugging Face 的 Config 对象. 这使得组件在非 HF 环境下更具移植性.
    2. 复用前向逻辑: 继承自 `LlavaMultiModalProjectorForYiVL` 是为了复用父类中已经处理好的
       `forward` 逻辑(特别是其中关于 fp32 精度回拨的黑魔法).

    高级研究员与资深工程师的架构点评:

    解决"推理引擎适配"的痛点:
    vLLM 这种高性能引擎在加载模型时, 有时会绕过传统的 from_pretrained 逻辑, 转而手动构建模型图(Graph). 如果 LLaMA-Factory 训练出来的模型包含非标准组件(如 Yi-VL 的 4 层 Projector), vLLM 默认的 LLaVA 适配器会失效. 这个类的存在就是为 vLLM 提供一个"即插即用"的模块定义, 确保微调后的 Yi-VL 模型能在 vLLM 上正常跑起来.

    构造函数的原子化(Parameter Atomization):
    在父类中, 我们使用 config: LlavaConfig. 但在 ForVLLM 变体中, 我们改用 vision_hidden_size: int. 这是一种更底层、更通用的接口设计. 因为 vLLM 在处理模型权重加载(Weight Loading)时, 往往已经解析出了这些 int 维度, 直接传参比构造一个临时的伪 Config 对象要高效且优雅得多.

    继承与多态的妙用:
    注意它依然继承自父类. 这意味着这个类不仅可以被 vLLM 使用, 它还保留了 LLaMA-Factory 对 Yi-VL 做的所有数值稳定性补丁. 例如, 如果父类的 forward 中有针对特定硬件的 autocast 处理, 这个类会自动继承这些能力, 无需重写.

    工程一致性:
    在深度学习工程中, "训练一个模型, 推理另一个模型"是最大的忌讳. 这种显式的 ForVLLM 补丁确保了训练时定义的数学公式与推理时定义的数学公式严格一一对应, 解决了模型在 vLLM 部署时可能出现的"输出与预期不符"的疑难杂症.
    """
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projector_hidden_act: str) -> None:
        # [为什么要传 config=None]:
        # 调用父类初始化, 但显式禁止父类根据 config 再次初始化成员变量.
        # 解决的问题: 防止父类因找不到 config.vision_config 等属性而报错, 同时允许我们在此子类中
        # 按照 vLLM 的初始化风格(显式参数传递)手动定义网络结构.
        super().__init__(config=None)

        # 1. 结构完全对齐 (Architecture Parity)
        # [解决的问题]: 确保在 vLLM 中构建的 Projection 层与微调时使用的 Yi-VL 结构 100% 一致.
        # 这种"Linear -> LN -> Act -> Linear -> LN"的 4 层深层对齐结构是 Yi-VL 区别于标准 LLaVA 的特征.

        # 第一层: 模态维度转换
        self.linear_1 = torch.nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        # 第二层: 初步特征归一化
        self.linear_2 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        # 第三层: 深度语义精炼
        self.linear_3 = torch.nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        # 第四层: 最终模态对齐归一化
        self.linear_4 = torch.nn.LayerNorm(text_hidden_size, bias=True)

        # 动态激活函数映射
        # [解决的问题]: 从字符串(如 "gelu")映射为可执行的函数对象, 保持与训练配置一致.
        self.act = ACT2FN[projector_hidden_act]


def autocast_projector_dtype(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Cast projector output to half precision for fine-tuning quantized VLMs.

    在微调多模态视觉语言模型(VLM, 如 LLaVA、Qwen-VL)时, 特别是涉及到 QLoRA(量化微调) 场景下, 这段代码解决了多模态特征对齐中的一个极其隐蔽且致命的精度冲突(Dtype Mismatch)问题.

    在微调量化后的视觉模型(VLM)时, 强制将投影器(Projector)的输出转换为目标计算精度.

    [为什么要这么写]:
    在多模态模型中, Projector(通常是一个 MLP 或线性层)负责将视觉特征映射到语言模型的 Embedding 空间.
    当使用 QLoRA(4-bit/8-bit)量化微调时:
    1. 视觉编码器和 Projector 通常保持在 FP32 或 BF16 以维持特征质量.
    2. 语言模型(LLM)骨架是被量化的.
    3. 在反向传播或混合精度训练中, PyTorch 的 Autocast 可能会让 Projector 的输出保持在 FP32.

    [解决的问题]:
    1. 精度不一致导致的崩溃: 如果 Projector 输出 FP32, 而 LLM 期望输入 BF16/FP16, 在执行 `torch.cat`
       (将图像 Token 和文本 Token 拼接)时, 或者在进入 LLM 第一层线性层时, 会触发 `RuntimeError: expected scalar type...`.
    2. 显存冗余: 保持不必要的 FP32 输出会增加中间激活值的显存占用.
    3. 确保梯度流顺畅: 通过 Hook 强制转换, 确保模态融合后的数据类型与微调时指定的 `compute_dtype` 严格一致.

    深度技术解析(高级研究员视角):

    关于 torch.cat 的痛点:
    在视觉语言模型的前向传播中, 我们会得到 image_embeddings (来自视觉分支) 和 text_embeddings (来自语言分支).
    combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
    如果 image_embeds 是 FP32(因为 Projector 内部可能有 LayerNorm 强制输出了 FP32), 而 text_embeds 是 BF16, 那么 torch.cat 会抛出异常. 这段代码通过 Hook 确保了 image_embeds 在进入拼接逻辑前就已经被"驯化"成了正确的 compute_dtype.

    量化环境下的特殊性:
    在使用 bitsandbytes (QLoRA) 时, 模型权重是以 int4 存储的. 为了计算, 它们会被动态反量化为 compute_dtype. 如果来自外部模态的 Tensor 精度过高, 会强制触发计算图中的一系列隐式向上转型(Upcasting), 不仅拖慢速度, 还可能导致显存溢出(OOM).

    工程设计的解耦性:
    这种设计体现了高级开发工程师的解耦思想: 我们不需要针对每一个 VLM 模型写一遍精度转换逻辑. 只需在 COMPOSITE_MODELS 注册表中定义好 get_projector 的路径, 这个通用的补丁函数就能通过 Python 的反射机制(Reflection)处理所有多模态模型的精度对齐问题.
    """

    # 1. 定义后置钩子 (Post-hook)
    # [为什么要用 Hook]: 这是一种非侵入式修改. 我们不需要修改模型源码(modeling_xxx.py),
    # 就能在计算图执行过程中, 拦截 Projector 的输出并进行"手术式"的类型转换.
    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        # 强制将输出转换为用户设定的计算精度(如 bf16 或 fp16)
        return output.to(model_args.compute_dtype)

    # 2. 触发条件检查
    # 只有当模型被量化时, 这种精度冲突才会变得突出.
    if getattr(model, "quantization_method", None):
        model_type = getattr(model.config, "model_type", None)

        # 3. 动态定位 Projector 模块
        # [为什么要这么写]: 不同模型的 Projector 命名和路径完全不同(如 mm_projector, mlp_adapter 等).
        # COMPOSITE_MODELS 存储了 LLaMA-Factory 维护的模型映射元数据.
        if model_type in COMPOSITE_MODELS:
            # 递归获取真正的 Projector 模块实例
            mm_projector = COMPOSITE_MODELS[model_type].get_projector(model)
        else:
            # 非多模态模型或未适配的模型不执行此逻辑
            return

        # 4. 注册 Hook
        # 告知系统: 在每次 Projector 计算完成后, 立即执行 _mm_projector_forward_post_hook.
        logger.info_rank0(f"Casting multimodal projector outputs in {model_args.compute_dtype}.")
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


def configure_visual_model(config: "PretrainedConfig") -> None:
    r"""
    Patch VLMs before loading them.

    在微调视觉大语言模型(如 LLaVA、Yi-VL 等)时, 由于模型架构由多个组件(视觉塔、投影层、语言模型)复合而成, Hugging Face 的原始配置往往无法直接满足分布式训练(DeepSpeed)或强化学习(RLHF)的需求.

    在模型正式加载前, 对多模态模型(VLM)的配置进行"外科手术式"的修正.

    高级研究员视角的深度解析:

    关于元数据透明化(Metadata Hoisting):
    在高级 Python 开发中, 我们经常遇到类似 config.text_config.hidden_size 的嵌套结构. 虽然这种设计符合逻辑, 但对于外部的分布式优化器(如 DeepSpeed)来说, 它们往往通过反射(Reflection)去抓取通用属性. 将 hidden_size 提升到根部是实现"库级兼容性"的标准技巧.

    猴子补丁(Monkey Patching)的艺术:
    注意到代码中直接修改了 transformers.models.llava.modeling_llava.LlavaMultiModalProjector. 这是高级开发人员处理"非标模型"最强力的手段.

    为什么不写一个全新的 Model 类? : 写新类意味着要维护上千行代码.
    补丁的好处: 只修改最核心的出错组件(投影层), 其他逻辑(如注意力计算、权重加载顺序)依然享受 Hugging Face 官方的更新和优化. 这是一种"最小侵入式"的优化策略.

    针对国产/社区模型的深度适配:
    Yi-VL 是零一万物开源的优秀视觉模型. LLaMA-Factory 这种细粒度的适配说明该框架不仅仅是简单封装, 而是深入到了模型数学实现层的工程化项目, 确保了社区模型在生产环境下的可用性.
    """

    # 1. 顶层隐藏层维度补齐 (Hidden Size Hoisting)
    # [为什么要这么写]:
    # 很多复合多模态模型(如 LLaVA)将语言模型的参数存储在子配置 `text_config` 中.
    # 而主配置 `config` 顶层往往缺失 `hidden_size` 属性.
    # [解决的问题]:
    # - DeepSpeed ZeRO-3 兼容性: ZeRO-3 在初始化并行分片时, 需要通过 `config.hidden_size`
    #   来预估张量大小. 如果缺失, 会导致分布式环境启动失败.
    # - ValueHead 模型适配: 在进行 PPO 或奖励模型微调时, TRL 库等工具需要直接从顶层
    #   config 获取维度来初始化线性层.
    # 这一步确保了"隐藏维度"这一关键元数据在配置对象的根路径下可见.
    if getattr(config, "text_config", None) and not getattr(config, "hidden_size", None):
        # required for ds zero3 and valuehead models
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))

    # 2. 针对 Yi-VL 架构的动态补丁 (Monkey Patching for Yi-VL)
    # [为什么要这么写]:
    # Yi-VL 系列模型虽然基于 LLaVA 架构, 但在多模态投影器(Projector)的实现上与
    # Transformers 官方库中的标准 `LlavaMultiModalProjector` 存在显著差异(层数或归一化逻辑不同).
    # [解决的问题]:
    # 如果直接使用 Transformers 官方的 LLaVA 代码加载 Yi-VL 权重, 会导致形状不匹配或推理精度异常.
    # 这里的逻辑通过"猴子补丁(Monkey Patching)"技术, 在内存中动态替换 Transformers
    # 内部的类定义.
    # 这样做可以复用原生的加载流程, 同时注入 LLaMA-Factory 特化的对齐逻辑(LlavaMultiModalProjectorForYiVL),
    # 彻底解决了第三方变体模型与官方库实现不一致的工程痛点.
    if getattr(config, "is_yi_vl_derived_model", None):
        logger.info_rank0("Detected Yi-VL model, applying projector patch.")
        # 强制替换 transformers 库中的类指向
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVL


def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> set[str]:
    r"""
    Freeze vision tower and language model for VLM full/freeze tuning.

    在多模态大模型(VLM, 如 LLaVA、Qwen-VL)的微调过程中, 模型不再是一个单一的 Transformer, 而是由视觉编码器(Vision Tower)、线性投影层(Projector)和大语言模型(LLM)组成的复合体.
    这段代码的核心使命是: 在全量微调(Full Tuning)或冻结微调(Freeze Tuning)中, 根据实验需求精准划定"禁止更新梯度"的禁区, 防止预训练知识的灾难性遗忘, 并优化显存占用.

    在 VLM 全量/冻结微调中, 确定需要禁止训练(冻结参数)的模块名称集合.

    [核心目标]:
    针对多模态模型, 将逻辑上的"组件(如视觉塔)"映射为代码中真实的"参数前缀(Parameter Prefix)".

    高级研究员视角的深度解析:

    解耦与抽象的艺术:
    代码中没有出现 if model_type == "llava": .... 作为高级开发工程师, 我们使用了 COMPOSITE_MODELS[model_type]. 这意味着当社区出现新的 VLM 模型(比如 DeepSeek-VL 或最新的 GPT-Omni 架构)时, 我们只需要在 COMPOSITE_MODELS 映射表中增加一条元数据定义, 而不需要修改这个核心的 get_forbidden_modules 函数. 这符合 "开闭原则(Open-Closed Principle)".

    防御性编程:
    使用了 getattr(config, "model_type", None). 在大型项目中, config 对象可能由不同的加载器生成, 并不保证一定包含所有属性. 这种写法避免了 AttributeError 导致的程序崩溃.

    分布式训练的细节:
    使用了 logger.info_rank0. 在多机多卡训练中, 如果每个进程都打印"Set vision model not trainable", 日志会被瞬间刷屏. rank0 确保了只有主进程输出信息, 保证了终端日志的清晰, 方便研究员监控微调配置是否生效.

    工程闭环:
    该函数返回的 forbidden_modules 集合后续会被传入到 _setup_full_tuning 或 _setup_freeze_tuning 中. 在遍历模型所有 named_parameters 时, 框架会检查参数名是否以这些 Key 开头, 从而执行 param.requires_grad = False. 这解决了在大规模参数量下, 手动查找和冻结特定模块极其低效且容易出错的问题.
    """

    # 1. 识别模型类型
    model_type = getattr(config, "model_type", None)
    forbidden_modules = set()

    # 2. 检查是否属于已定义的"复合模型(Multimodal/Composite Models)"
    # [为什么要这么写]: VLM 的各组件命名极度不统一(有的叫 mm_projector, 有的叫 mlp_adapter).
    # [解决的问题]: 通过 COMPOSITE_MODELS 注册表实现了"架构解耦".
    # 框架不需要硬编码不同模型的层名, 而是通过 model_type 动态获取该架构下的组件 Key.
    if model_type in COMPOSITE_MODELS:

        # 3. 冻结视觉编码器 (Vision Tower)
        # [解决的问题]: 视觉塔通常使用 CLIP 等预训练好的强力编码器. 在模态对齐阶段,
        # 如果更新视觉塔, 极易导致预训练视觉特征坍缩, 且视觉塔参数量巨大, 冻结它能节省大量显存.
        if finetuning_args.freeze_vision_tower:
            vision_model_keys = COMPOSITE_MODELS[model_type].vision_model_keys
            logger.info_rank0(f"Set vision model not trainable: {vision_model_keys}.")
            forbidden_modules.update(vision_model_keys)

        # 4. 冻结多模态投影器 (Multi-modal Projector)
        # [为什么要这么写]: Projector 是连接视觉和语言的"桥梁".
        # [解决的问题]: 在某些阶段(如仅微调 LLM 的领域知识时), 我们希望保持模态对齐能力不变.
        # 此时通过此开关, 可以确保投影层的映射关系不被破坏.
        if finetuning_args.freeze_multi_modal_projector:
            projector_key = COMPOSITE_MODELS[model_type].projector_key
            logger.info_rank0(f"Set multi model projector not trainable: {projector_key}.")
            forbidden_modules.add(projector_key)

        # 5. 冻结语言模型核心 (Language Model Core)
        # [为什么要这么写]: 针对"模态对齐预训练(Pre-alignment)"任务.
        # [解决的问题]: 在 VLM 训练的第一阶段, 通常目标是让 Projector 学习如何将图像特征
        # 映射到 LLM 能理解的语义空间. 此时我们希望 LLM 作为"固定的知识库"不被改变,
        # 仅微调中间的连接层.
        if finetuning_args.freeze_language_model:
            language_model_keys = COMPOSITE_MODELS[model_type].language_model_keys
            logger.info_rank0(f"Set language model not trainable: {language_model_keys}.")
            forbidden_modules.update(language_model_keys)

    # 返回所有被标记为"禁止访问"的模块名称前缀集合
    return forbidden_modules


def patch_target_modules(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments", target_modules: list[str]
) -> list[str]:
    r"""
    Freeze vision tower for VLM LoRA tuning.

    在微调多模态大模型(VLM, 如 LLaVA、Qwen-VL)时, LoRA 的应用会变得非常复杂. 这个函数是处理 "全模型视野 vs 局部模块更新" 冲突的关键组件. 它解决的核心问题是: 如何防止 LoRA 适配器错误地注入到那些应当被冻结或不兼容的子模块(如视觉塔)中.

    针对多模态模型(VLM)微调, 动态修整 LoRA 目标模块列表.

    [解决的问题]:
    当用户指定 target_modules 为 ["q_proj", "v_proj"] 时, PEFT 库默认会扫描模型中所有包含这些
    名称的线性层. 但在 VLM 中, 视觉编码器(Vision Tower)和语言模型(LLM)往往都含有名为 "q_proj" 的层.
    如果用户只想微调语言模型(或者视觉塔不支持 LoRA), 这种简单的名称匹配会导致非预期的参数更新或显存爆炸.

    高级研究员与资深开发者的架构深度解析:

    解决多模态下的"命名污染"问题:
    在 Transformer 架构中, 层名高度重复. 在 VLM 中, 如果视觉组件(Vision Tower)是由另一个库加载的, 它的 q_proj 权重可能被设为了不可训练(Frozen). 如果 LoRA 强制注入, 不仅会改变视觉特征的提取逻辑(这通常不是我们想要的), 还会导致梯度图变得极其庞大. 这段代码通过显式的黑名单过滤确保了语言模型和视觉模型的物理隔离.

    解耦"用户意图"与"底层架构":
    用户在 Web UI 或命令行通常只想简单地说: "我要微调所有的线性层(all)". 该函数充当了翻译层: 它将"全量微调"的抽象意图, 结合当前模型的具体拓扑结构(Config)以及微调参数(Arguments), 转化成一份安全的、可执行的层级白名单.

    对 PEFT 库行为的重塑:
    标准的 peft.LoraConfig 如果接收到短字符串(如 ["q_proj"]), 它会在内部做全量模糊匹配. 通过这个函数, 我们改为传递全路径字符串(Full Path strings). 这是一种高级技巧, 能够强制让 PEFT 只在特定的上下文路径下工作, 彻底消除了"误伤"其他组件的可能性.
    """

    # 1. 识别当前模型架构是否属于复合模型(如 VLM)
    model_type = getattr(model.config, "model_type", None)
    if model_type in COMPOSITE_MODELS:
        # 2. 获取基于微调参数定义的"禁区"模块
        # [为什么要这么写]: get_forbidden_modules 会根据用户是否设置了 `freeze_vision_tower`
        # 等参数, 返回一组模块路径前缀(例如 "visual" 或 "vision_tower").
        forbidden_modules = get_forbidden_modules(model.config, finetuning_args)

        # 3. 注入特定模型的架构冲突黑名单
        # [解决的问题]: 某些模型的特定层由于权重拆分、算子特化或量化冲突, 严禁挂载 LoRA.
        # 这些信息存储在 COMPOSITE_MODELS 的元数据中, 确保了框架的稳健性.
        forbidden_modules.update(COMPOSITE_MODELS[model_type].lora_conflict_keys)
        module_names = []

        # 4. 精准过滤逻辑 (Precision Filtering)
        # [核心逻辑]: 遍历模型中所有命名的子模块.
        for name, _ in model.named_modules():

            # 判定条件:
            # A. 模块名称匹配用户指定的 target_modules(如 "q_proj").
            # B. 模块名称[不包含]任何禁区前缀(如该模块不在视觉塔内).
            # [为什么要这么写]: 通过"白名单名称 + 黑名单路径"的双重过滤,
            # 实现了对 LoRA 注入位置的"外科手术式"精确控制.
            if any(target_module in name for target_module in target_modules) and not any(
                forbidden_module in name for forbidden_module in forbidden_modules
            ):
                # 记录完整的模块路径(例如 "language_model.model.layers.0.self_attn.q_proj")
                module_names.append(name)

        # 返回精确的完整模块列表, 直接传给 PEFT, 绕过其简单的后缀匹配逻辑
        return module_names
    else:
        # 5. 回退逻辑 (Fallback)
        # 对于非复合模型(普通 LLM), 直接返回用户输入的原始列表, 保持原有兼容性.
        return target_modules


_register_composite_model(
    model_type="dots_ocr",
    projector_key="vision_tower.merger",
    vision_model_keys=["vision_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["merger"],
)


_register_composite_model(
    model_type="gemma3",
)


_register_composite_model(
    model_type="gemma3n",
    vision_model_keys=["vision_tower", "audio_tower"],
    lora_conflict_keys=["timm_model", "subsample_conv_projection"],
)


# copied from qwen2vl
_register_composite_model(
    model_type="glm4v",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="glm4v_moe",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="internvl",
)

_register_composite_model(
    model_type="interns1",
)

_register_composite_model(
    model_type="Keye",
    projector_key="mlp_AR",
    vision_model_keys=["visual.vision_model.patch_embedding", "visual.vision_model.encoder"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embedding"],
)


_register_composite_model(
    model_type="kimi_vl",
)


_register_composite_model(
    model_type="llama4",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="llava",
)


_register_composite_model(
    model_type="llava_next",
)


_register_composite_model(
    model_type="llava_next_video",
)


_register_composite_model(
    model_type="minicpmv",
    projector_key="resampler",
    vision_model_keys=["vpm"],
    language_model_keys=["llm"],
)


_register_composite_model(
    model_type="minicpmo",
    projector_key="resampler",
    vision_model_keys=["vpm", "apm", "audio_avg_pooler", "audio_projection_layer", "tts"],
    language_model_keys=["llm"],
    lora_conflict_keys=["audio_projection_layer"],
)


_register_composite_model(
    model_type="mistral3",
    projector_key="model.multi_modal_projector",
)


_register_composite_model(
    model_type="mllama",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="paligemma",
)


_register_composite_model(
    model_type="qwen2_audio",
    vision_model_keys=["audio_tower"],
)


_register_composite_model(
    model_type="qwen2_5_omni_thinker",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "audio_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_5_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_vl_moe",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen3_omni_moe_thinker",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list", "audio_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="video_llava",
)

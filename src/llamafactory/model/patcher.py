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

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    """
    高级研究员视角的架构深度解析:

    为什么 Qwen3-Omni-MoE 特别需要这个补丁?
    Qwen3-Omni 引入了特殊的"Thinker"机制, 其 MoE 层不仅要处理文本的 Token 路由, 还要维持思考链的长文本依赖.
    在 DeepSpeed ZeRO-2 下, MoE 专家的参数分布在不同卡上.
    官方早期的 4.x 实现可能在计算路由概率(Gating Logits)时没有正确处理不同进程间的 requires_grad 状态, 导致在分布式环境下产生 None 梯度或精度坍缩.

    ZeRO-2 与 FSDP2 的复杂性:
    这两个框架都涉及到参数的动态重组. 传统的 MoE 块如果不经过特殊优化, 在训练时会频繁触发跨卡全量同步.
    我们补丁后的 Qwen3OmniMoeThinkerTextSparseMoeBlock 通常使用了更高效的算子融合(Operator Fusion)或者专家路由重排(Expert Dispatching Reordering), 极大缓解了分布式计算中的通讯瓶颈.

    工程上的健壮性:
    这种补丁机制展示了 LLaMA-Factory 的核心哲学: "宁可在初始化阶段复杂, 也要在训练阶段稳定."
    对于研究人员来说, 不用去修改 Hugging Face 的源代码, 只需配置好环境, LLaMA-Factory 就会在后台自动完成所有的底层"排雷"工作.
    """

    # 1. 严格的版本探测逻辑 (Version Window Slicing)
    # [为什么要这么写]: 此 Bug 仅存在于特定的版本区间(4.57.0 <= transformers < 4.58.0).
    # [解决的问题]: 避免"误伤". 库的维护者通常会在下个版本修复 Bug, 但在修复版发布前的"窗口期",
    # 我们必须手动介入. 这体现了 Python 开发中针对依赖库生命周期的精细化管理.
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):

        # 2. 延迟导入修复组件 (Lazy Import of the Fix)
        # [为什么要这么写]: 仅在满足版本触发条件时才导入修复类 Qwen3OmniMoeThinkerTextSparseMoeBlock.
        # [解决的问题]: 减少内存占用, 并防止在旧版或未来版本 transformers 环境下由于 API 差异导致导入失败.
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        # 3. 极高性能影响警告 (High-Stake Warning)
        # [为什么要这么写]: 通过 logger 告知用户当前正在使用 LLaMA-Factory 的注入代码而非官方代码.
        # [解决的问题]: 透明化. 在分布式训练中, DeepSpeed ZeRO-2 和 FSDP2 对 MoE(混合专家模型)参数的
        # 分片(Sharding)和梯度同步极其敏感. 官方原始实现可能在梯度检查点(Gradient Checkpointing)
        # 或参数收集时存在死锁或性能大幅下降的问题.
        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        # 4. 猴子补丁注入 (Monkey Patching)
        # [为什么要这么写]: 直接修改导入进来的 modeling 模块中的类引用.
        # [解决的问题]: 这是最核心的一步. 它绕过了对 transformers 源码的修改,
        # 在程序运行初期, 将内存中官方的 MoE Block 类替换为我们优化后的实现类.
        # 优化点通常在于:
        #   a. 改进了 MoE Router 的计算逻辑, 使其能正确适配 DeepSpeed 的参数分区策略.
        #   b. 解决了 FSDP2 在处理 Sparse MoE 时的权重聚合(All-Gather)错误.
        #   c. 确保了 Qwen3 "Thinker"(逻辑思考部分)在推理和微调时的状态一致性.
        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    """
    在 LLM 微调工程中, 分词器(Tokenizer)是连接原始文本与神经网络的桥梁.
    不同模型厂商(如 OpenAI, Meta, 阿里, 零一万物)提供的 tokenizer_config.json 质量参差不齐, 且 transformers 库的基类方法有时会被子类以不兼容的方式重写.
    这段代码的核心使命是: 通过"猴子补丁(Monkey Patching)"和"工程防御逻辑", 强制统一不同模型 Tokenizer 的行为, 确保大规模微调时的鲁棒性.

    # 1. 强制回归标准 Padding 逻辑 (Method Hijacking)
    # [为什么要这么写]: 某些自定义 Tokenizer 子类(例如早期版本的 Qwen 或某些特定架构模型)
    # 重写了私有的 `_pad` 方法, 但其内部逻辑可能与 LLaMA-Factory 的 DataCollator 或高效并行算子冲突.
    # [解决的问题]: 通过检测 `_pad` 函数的来源, 如果它不是来自标准的 `PreTrainedTokenizerBase`,
    # 就使用 `MethodType` 将基类的标准补齐方法动态绑定到当前实例上.
    # 这确保了在 Batch 推理和训练时, Padding 行为是可预测且符合 Transformers 框架标准的.

    高级研究员视角下的架构点评:

    工程健壮性(Robustness):
    在生产环境中, 你永远不能假设用户提供的模型配置文件是完美的. patch_tokenizer 扮演了"质检员"的角色, 在模型进入训练循环之前, 修正了所有潜在的、会导致训练中断或精度异常的配置缺陷.

    猴子补丁的优雅性:
    使用 MethodType 进行运行时方法替换是 Python 高级开发的常见技巧. 它避免了修改 transformers 库源码的侵入式操作, 同时解决了 C++ 扩展或 Rust 分词器层面的底层调用分发问题.

    对 Embedding 层的保护:
    大模型训练最常见的报错之一就是"Embedding Index Out of Range". 通过在该函数中实时监测 num_added_tokens 并自动改写 model_args.resize_vocab, 我们成功地为用户屏蔽了复杂的层权重对齐逻辑, 将"微调失败"的可能性降到了最低.

    print(tokenizer._pad.__func__)
    <function PreTrainedTokenizerBase._pad at 0x7f0494ba0180>
    print(str(tokenizer._pad.__func__))
    <function PreTrainedTokenizerBase._pad at 0x7f0494ba0180>
    """
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    """
    # 2. 动态扩展上下文长度限制 (Context Window Enlargement)
    # [为什么要这么写]: 许多原始模型的 `tokenizer_config.json` 中 `model_max_length` 被硬编码为较小的值(如 2048).
    # 但实际微调时, 用户可能希望通过位置编码插值(RoPE Scaling)训练更长的序列(如 8192).
    # [解决的问题]: 如果用户通过命令行指定的 `model_max_length` 大于 Tokenizer 自带的限制,
    # 强制覆盖该值. 防止 Tokenizer 在处理长文本时弹出冗余的警告信息, 或意外地将数据截断.

    print(model_args.model_max_length)
    2048
    print(tokenizer.model_max_length)
    1010000
    """
    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    """
    # 3. 动态词表扩充: 普通 Token (Vocab Expansion)
    # [为什么要这么写]: 在垂直领域微调(如医疗、法律)中, 用户经常需要添加领域专属词汇.
    # [解决的问题]: 调用 `tokenizer.add_tokens`.
    # 这里的关键工程细节在于 `resize_vocab` 的联动.
    # 如果添加了新词但忘记缩放模型的 Embedding 层, 模型在训练遇到新 Token ID 时会报越界错误(IndexError).
    # 代码在这里做了"自动纠错", 一旦检测到新词加入, 强制将 `resize_vocab` 设为 True, 保证训练流程的自动化和安全性.

    print(model_args.add_tokens)
    None
    """
    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    """
    # 4. 动态词表扩充: 特殊 Token (Special Token Injection)
    # [为什么要这么写]: 微调过程中可能需要添加新的控制符(如工具调用的标记 <tool_call>).
    # [解决的问题]: 与普通 Token 不同, `special_tokens=True` 会确保这些 Token 永远不会被拆分成子词(Subwords).
    # 同样地, 这里也实现了 `resize_vocab` 的自动触发机制.
    # 这种"双管齐下"的设计, 保证了无论用户添加什么类型的词汇, 底层的权重矩阵都能在后续的 `load_model` 步骤中得到正确调整.

    print(model_args.add_special_tokens)
    None
    """
    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    """
    在微调多模态大模型(VLM/Speech-LLM, 如 LLaVA、Qwen-VL、Whisper 等)时, Processor 负责将非文本数据(图像、视频、音频)转化为模型能理解的张量(Tensors).
    这段代码的核心目的是通过动态补丁(Monkey Patching), 将用户自定义的硬件约束和算法参数强制注入到 Hugging Face 原生的 Processor 实例中.

    对多模态处理器进行动态修正和参数注入.

    [为什么要这么写]:
    Hugging Face 的 AutoProcessor 默认从模型路径下的 preprocessor_config.json 加载参数.
    但在实际微调中, 默认参数往往不能满足: 1. 显存优化需求(如限制像素); 2. 特定的预处理逻辑变更.

    [要解决的问题]:
    1. 统一性: 确保 Processor 使用的是我们已经 patch 过的 Tokenizer, 防止词表不一致.
    2. 显存控制(OOM 预防): 通过动态调整分辨率和帧数, 防止大分辨率图像或超长视频撑爆显存.
    3. 灵活性: 无需修改模型原始配置文件, 即可在命令行动态调整多模态处理行为.

    资深架构师视角的深度解析:

    为什么使用 setattr 而不是在构造函数里传参?
    Hugging Face 的 AutoProcessor 基类非常灵活, 但不同的模型子类(如 LlavaProcessor vs Qwen2VLProcessor)其参数列表是不统一的. 使用 Python 的动态属性设置 setattr, 可以实现**"鸭子类型(Duck Typing)"式的解耦**. 如果当前的 Processor 支持这些参数, 它就会被注入并生效; 如果不支持, 通常也不会导致崩溃, 这极大地增强了 LLaMA-Factory 对海量不同多模态架构的泛化适配能力.

    防御性编程与资源预算:
    在大模型微调场景中, 数据(Data)是不可控的. 用户可能传入一张 8K 分辨率的图片或一段 1 小时的视频. 如果没有 image_max_pixels 和 video_maxlen 这种强制性的补丁约束, 底层的 DataCollator 在处理这批数据时会瞬间因为序列长度爆炸而导致 GPU OOM. 这个函数实际上是微调框架的一道"安全防火墙".

    多模态对齐(Multimodal Alignment):
    将 tokenizer 强行塞进 processor 是为了确保 input_ids 的生成与文本编码完全对齐. 在 VLM 中, 图像生成的虚拟 Token 需要被插入到文本的特定位置, 如果两者逻辑不一致, 会导致模型生成的 Token 索引发生偏移, 造成训练完全失效.
    """

    # 1. 绑定已修正的 Tokenizer
    # [原因]: LLaMA-Factory 往往会对 tokenizer 进行特殊处理(如 resize 词表、添加特殊 token).
    # 将其重新绑定给 processor, 确保 processor.encode() 等操作使用的是同一套词表逻辑.
    setattr(processor, "tokenizer", tokenizer)

    # 2. 图像分辨率控制 (Image Scaling)
    # [解决问题]: 图像像素直接决定了 Vision Encoder 生成的 patch 数量.
    # 设置 max_pixels 和 min_pixels 可以通过缩放输入图像来控制计算量, 从而平衡精度与显存消耗.
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)

    # 3. 图像切片策略 (Tiling/Slicing)
    # [原因]: 对于像 LLaVA-1.6 或 CogVLM 这种支持"动态分辨率"的模型,
    # 它们通过切片(pan and scan / crop to patches)来处理超长宽比图像.
    # 显式设置这些参数, 可以覆盖模型默认的切块逻辑, 适配特定的微调实验需求.
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)

    # 4. 视频计算开销控制 (Video Budgeting)
    # [解决问题]: 视频是极其消耗算力的(帧数 * 每帧像素).
    # 通过注入 fps(采样频率)和 maxlen(最大帧数), 研究员可以根据显存大小强行"抽帧"或压缩视频.
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)

    # 5. 音视频复合处理 (Audio-Visual Integration)
    # [原因]: 针对视频模型, 决定是否提取音频流.
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)

    # 6. 音频采样校准 (Audio Re-sampling)
    # [解决问题]: LLM 音频组件(如 Whisper)通常严格要求 16000Hz 采样率.
    # 这里注入参数确保 Processor 在处理音频时执行正确的重采样, 防止因采样率不匹配导致的训练不收敛.
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    """
    在 LLM 领域, 虽然 Hugging Face 的 transformers 库提供了一套标准, 但不同厂商的模型实现千差万别(有的不符合标准 API, 有的存在特定的 Bug). patch_config 函数的使命是在模型被实例化之前, 通过"外科手术"式的修正, 确保模型能以高性能、正确且稳定的方式运行在各种硬件环境(单卡、DeepSpeed、FSDP)中.

    高级研究员视角的设计亮点:

    容错式设计 (Defensive Programming):
    代码中对 InternVL 和 LLaVA 的 ValueError 处理非常关键. 在大规模微调项目中, 最大的成本往往不是 GPU 算力, 而是研究员的时间. 这种拦截机制能瞬间指出数据/模型源的问题.

    分布式与单机的无缝平滑切换:
    第 9 和第 10 点处理了分布式环境下的经典矛盾: device_map 与 ZeRO-3. 初级开发者经常会遇到开启 DeepSpeed 后模型加载卡死或报错, 这段代码通过检测框架状态, 动态调整加载参数, 实现了"一份代码, 多卡通用".

    计算效能最大化:
    通过 Liger Kernel(在 apply_liger_kernel 中, 代码未展示但此处调用)和 Flash Attention 的强制注入, LLaMA-Factory 能够比原生 Transformers 节省 30%-50% 的显存, 这对于在消费级显卡(如 RTX 4090)上微调 70B 模型至关重要.
    """

    # 1. 自动对齐计算精度 (Precision Alignment)
    # [为什么要这么写]: LLM 训练对精度极度敏感. bf16 性能最好且稳定, fp16 次之.
    # [解决的问题]: 如果用户没指定 compute_dtype, 根据硬件环境自动推断最优精度.
    # 如果是推理模式且设置了 infer_dtype, 则优先尊重推理精度; 训练模式下则调用 infer_optim_dtype 确保梯度不溢出.
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # 2. 模块化特性配置 (Modular Feature Configuration)
    # [为什么要这么写]: 将复杂的配置逻辑解耦到独立的子函数中.
    # [解决的问题]:
    # - configure_attn_implementation: 决定使用 FlashAttention-2、SDPA 还是传统的 Eager 模式.
    # - configure_rope: 处理长文本外推(RoPE Scaling), 解决训练和推理长度不匹配.
    # - configure_quantization: 处理 4/8-bit 量化加载(如 bitsandbytes, AWQ, GPTQ).
    # - configure_moe: 优化专家模型的通信和显存分配.
    # - configure_packing: 针对高效 SFT 的序列打包技术.
    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    # 3. 针对 Qwen 系列的特殊 Hack (Vendor-specific Compatibility)
    # [为什么要这么写]: 早期 Qwen 模型在 HF 的实现不完全标准, 需要显式设置这些布尔标记.
    # [解决的问题]: 确保 Qwen 能够正确识别 compute_dtype 并启动 Flash Attention.
    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    # 4. 多模态/音频模型初始化修正
    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)  # 默认关闭 TTS 初始化, 减少非必要资源占用

    # 5. 特定专家选择策略修正
    # [解决的问题]: Kimi-VL 在训练时, Top-K 路由若使用默认方法可能不收敛, 强制改为 greedy 以稳定梯度.
    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    # 6. 生态陷阱拦截 (Ecosystem Trap Interception)
    # [为什么要这么写]: InternVL 和 LLaVA 有多种非官方分发的版本, 其代码格式与原生 transformers 不兼容.
    # [解决的问题]: 提前抛出易懂的错误, 防止用户在加载数 GB 的模型崩溃后才发现下错了版本.
    if "InternVLChatModel" in getattr(config, "architectures", []):
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    # 7. 依赖库版本强校验
    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    # 8. Qwen3 新特性的运行时补丁
    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # 9. 分布式环境下的内存安全 (Distributed Memory Safety)
    # [为什么要这么写]: DeepSpeed ZeRO-3 会将参数分布在所有卡上.
    # [解决的问题]: `low_cpu_mem_usage` 会让每个进程都尝试去 load 模型骨架, 这在 ZeRO-3 下会导致内存死锁.
    # 因此, 如果是 ZeRO-3 环境, 必须强行关闭此选项.
    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # 10. 智能设备分配逻辑 (Automatic Device Allocation)
    # [为什么要这么写]: FSDP 和 ZeRO-3 会自动管理设备分配.
    # [解决的问题]: 防止用户手动指定的 device_map 与分布式框架发生冲突.
    # 在非 ZeRO-3/FSDP 环境下且开启 low_cpu_mem_usage 时, 才允许注入自定义的 device_map.
    # 并在设置 device_map="auto" 时, 自动配置 offload_folder 防止本地显存溢出.
    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    """
    在 LLM 微调工程中, 加载完原始模型后, 通常不能直接开始训练. 由于不同厂商的模型实现(Modeling Code)存在差异、Tokenizer 词表可能扩充、或者需要适配特定的训练加速技术(如 DeepSpeed ZeRO-3), 我们必须对模型实例进行一次"外科手术"式的修正.

    高级研究员与资深开发者的视角解析:

    关于 MethodType 的使用:
    这是 Python 高级开发中的 Monkey Patching(猴子补丁) 技术. 在大模型领域, 我们经常需要"修正"那些已经加载到内存中的第三方库对象, 而不去直接改动库的源码. 这保证了 LLaMA-Factory 的非侵入性.

    resize_vocab 的严谨性:
    这是微调中最容易踩坑的地方. 简单的 resize_token_embeddings 会导致新 Token 的权重是随机的, 可能破坏原本稳定的模型输出. 代码中调用的 resize_embedding_layer 内部通常包含更复杂的逻辑, 比如将新 Token 初始化为已有 Token 的均值, 能显著加快微调收敛.

    ZeRO-3 兼容性:
    add_z3_leaf_module 体现了工业级微调框架对大规模分布式训练的深度适配. ZeRO-3 会尝试切分所有参数, 但某些模块(如某些模型的 MoE Router)如果被切分, 会极大地增加通讯延迟. 将其设为叶子节点是平衡显存和速度的关键.

    多模态前瞻性:
    autocast_projector_dtype 说明 LLaMA-Factory 已经从纯文本微调演进到了 VLM(视觉语言模型) 领域, 处理了视觉编码器与语言模型之间那层 Projector 的数值稳定性问题.
    """

    # 1. 生成配置的自动化纠错 (Generation Config Auto-fix)
    # [为什么要这么写]: 在 Hugging Face 逻辑中, 如果设置了 temperature 等采样参数, 但 do_sample 为 False, 会抛出警告或导致非预期行为.
    # [解决的问题]: 确保用户意图一致性. 如果用户在命令行设置了采样参数(非 1.0), 系统自动开启采样模式, 避免推理时报错.
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    # 2. 强制回归标准生成函数 (Standardizing .generate method)
    # [为什么要这么写]: 某些第三方模型(特别是通过 trust_remote_code 加载的模型)会重写自定义的 generate 函数.
    # [解决的问题]: 这些自定义函数往往不兼容标准的推理流(如流式输出、特定的停止词逻辑).
    # 除了特殊的多模态模型外, 我们利用 Monkey Patching 强制将模型的方法替换为 Hugging Face 官方标准的 GenerationMixin 逻辑.
    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    # 3. 强化学习/奖励模型准备 (RLHF/Reward Model Prep)
    # [为什么要这么写]: 如果是 RM(奖励模型)或 PPO 任务, 模型需要输出一个标量分数.
    # [解决的问题]: 此函数负责在 Base Model 顶部挂载一个线性层(Value Head), 并确保其梯度状态正确.
    if add_valuehead:
        prepare_valuehead_model(model)

    # 4. 词表大小与嵌入层对齐 (Vocab Resizing & Alignment)
    # [为什么要这么写]: 如果用户通过指令增加了特殊 Token(如 <tool_call>), 或者 Tokenizer 的 vocab 大于 Embedding 层.
    # [解决的问题]: 防止 Index Out of Range 错误. 此步骤会扩充 Embedding 和 LM_Head 矩阵, 并根据配置对新位置进行智能初始化(如均值初始化), 而不是随机初始化.
    if model_args.resize_vocab:
        resize_embedding_layer(
            model,
            tokenizer,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    # 5. 训练特定配置 (Training-specific Patching)
    if is_trainable:
        # 针对特定模型(如 gemma3n)的硬编码 Hack, 解决梯度检查点(Gradient Checkpointing)在某些架构上的崩溃问题.
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        # 核心预处理: 包括开启梯度检查点减少显存占用、处理 PEFT 的 k-bit 量化准备、以及冻结非必要层.
        prepare_model_for_training(model, model_args)
        # 确保多模态模型的 Projector(视觉-语言对齐层)具有正确的精度(通常是 fp32 或 bf16), 防止精度坍缩.
        autocast_projector_dtype(model, model_args)
        # DeepSpeed ZeRO-3 优化: 将特定模块标记为"叶子节点", 防止被 ZeRO-3 错误地拆分导致通讯死锁.
        add_z3_leaf_module(model)

    # 6. 注意力机制可视化
    # [解决的问题]: 让用户明确知道当前模型到底跑的是 FlashAttention-2、SDPA 还是传统的 Eager 模式, 方便排查性能瓶颈.
    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    # 7. 模型标记 (Attribution & Telemetry)
    # [解决的问题]: 在导出的模型权重中注入 llama-factory 标签, 方便在 Hugging Face Hub 上进行生态追踪和版本溯源.
    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    """
    在强化学习(RLHF)阶段, 如 PPO 或奖励模型(RM)训练时, 我们需要在原有的语言模型(CausalLM)顶层挂载一个名为 v_head 的线性层, 用于输出标量分数. 通常使用的是 trl 库的 AutoModelForCausalLMWithValueHead.
    然而, 这个类只是一个外壳包装器. 当我们将它放入复杂的训练流水线(如 Hugging Face Trainer)时, 外壳模型往往"弄丢"了底层模型的一些关键能力和元数据. 这段代码通过 Monkey Patching(猴子补丁) 重新建立了外壳与内核之间的桥梁.

    对带有 ValueHead 的模型进行深度补丁, 修复包装器导致的接口缺失问题.
    [核心目的]:
    AutoModelForCausalLMWithValueHead 将原始 LLM 封装在 self.pretrained_model 属性中.
    但许多 Transformers 的标准接口(如权重共享、嵌入层访问)无法透过这个外壳.
    如果不做补丁, 训练器在尝试保存模型、调整词表或计算 RoPE 时会崩溃或导致逻辑错误.

    高级研究员视角的架构解析:

    委托模式(Delegation Pattern)的工程实现:
    由于 Python 不支持像一些语言那样的"透明转发", 当一个类(Wrapper)包装了另一个类(Core)时, Wrapper 就成了 Core 的隔绝层. 这段代码本质上是在手动打通 I/O 通道.

    处理 PeftModel 的复杂性:
    注意 get_rope_index_func 中对 PeftModel 的多级探测(.base_model.model). 这是因为使用 LoRA 时, 层级会变得非常深, 标准的路径访问会失效. 这体现了对 peft 库底层结构的深度理解.

    对存储机制的优化:
    _keys_to_ignore_on_save 的设置反映了对生产环境存储空间的极致考量. 在 RLHF 中, 我们通常只关心 v_head 的权重和 LoRA 的权重, 而不希望每次保存都备份一份几十 GB 的 Base Model 权重.
    """

    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        """
        为什么要这么写]: 手动委托权重绑定逻辑.
        [解决的问题]: 许多模型(如 Gemma, Qwen)为了节省显存, 输入嵌入层(Input Embeddings)和
        输出层(LM Head)是共享权重的. 如果外壳模型不显式调用内核的 tie_weights,
        在词表扩充或初始化时, 权重共享关系会失效, 导致训练不收敛.
        """
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        """
        [为什么要这么写]: 重定向输入嵌入层的获取.
        [解决的问题]: DataCollator 或 Trainer 有时需要访问 Embedding 层(例如为了获取词表大小).
        包装器默认没有这个方法, 会导致 AttributeError.
        """
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        """
        [解决的问题]: 确保在保存或检查模型时, 能够正确访问到原始模型的输出头(LM Head).
        """
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        """
        [为什么要这么写]: 适配 PEFT (LoRA) 的模型卡生成.
        [解决的问题]: 如果使用 LoRA 进行 RLHF, 模型卡(README.md)的更新逻辑存在于 PeftModel 中.
        如果不委派, 微调后的模型目录将缺失关键的元数据信息.
        """
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        """
        [为什么要这么写]: 动态提取 RoPE (旋转位置编码) 的索引计算函数.
        [解决的问题]: 针对复杂架构(如 Qwen2-VL, GLM 等), 模型可能定义了特殊的 RoPE 逻辑.
        由于外壳模型改变了层级结构(self.pretrained_model.model...),
        这段逻辑负责从不同的深度级别自动"捞出"正确的 get_rope_index 方法, 防止位置编码计算错误.
        """
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    # 1. 权重保存过滤 (State Dict Management)
    # [为什么要这么写]: 找出所有包含 "pretrained_model" 的参数名.
    # [解决的问题]: 在保存 ValueHead 模型时, 如果不配置 ignore_modules,
    # 系统可能会尝试重复保存底层的全量权重, 导致导出的 Checkpoint 异常巨大或出现命名冲突.
    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)

    # 2. 方法绑定 (Dynamic Binding)
    # [为什么要这么写]: 使用 MethodType 将上面定义的局部函数绑定为实例的方法.
    # [解决的问题]: 这是 Python 高级开发中的核心技巧, 它能确保在 Trainer 调用 model.tie_weights() 时,
    # 实际上运行的是我们定义的委托逻辑, 且 `self` 能正确指向当前的 model 实例.
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))

    # 3. 注入特定功能
    # 将 RoPE 索引函数注入, 确保在处理超长文本或多模态序列时, 位置信息计算无误.
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))

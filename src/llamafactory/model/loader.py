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

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""
    Get arguments to load config/tokenizer/model.
    Note: including inplace operation of model_args.

    在 LLM 微调工程中, 如何稳健地从不同环境、不同 Hub 加载模型是一个巨大的挑战.
    这段代码的核心目标是标准化 Hugging Face 系列加载函数(AutoConfig/AutoTokenizer/AutoModel)的入口参数,
    并解决环境依赖导致的加载崩溃问题.

    获取加载 config/tokenizer/model 所需的标准关键字参数.

    Note: 包含对 model_args 的原地修改(inplace operation).

    高级研究员视角的架构深度解析:

    关于 skip_check_imports 的必要性:
    在 LLM 科研中, 我们经常遇到"环境半就绪"的状态. 例如, 用户的 A100 环境装好了, 但 flash-attention 编译报错没装上.
    如果你想先用 sdpa (PyTorch 原生注意力) 进行推理测试, 传统的加载方式会因为模型 py 文件里的一行 import flash_attn 而无法运行.
    LLaMA-Factory 这种写法体现了"容错式加载"的理念, 将环境报错推迟到真正的 forward 阶段(如果届时真的需要该算子), 而不是卡在初始化阶段.

    原地修改 (Inplace Operation) 的工程考量:
    注释里特意提到了 inplace operation. 在 Python 大型项目中, 修改入参通常被视为副作用(Side-effect).
    但在 model_args 这种庞大的配置类中, 一旦识别到用户意图(比如指定了 ModelScope 路径), 立即更新 model_name_or_path 是为了保证后续组件(如推理引擎、显存优化器等)在引用该路径时的一致性.

    对 trust_remote_code 的默认抽象:
    将 trust_remote_code 封装进 kwargs, 使得 LLaMA-Factory 可以轻松适配各种非 Hugging Face 官方支持的异构模型(如某些实验性质的线性注意力模型), 确保了框架的极强通用性.
    """

    # 1. 解决"依赖地狱"问题 (Monkey Patching for Dependency Resilience)
    # [为什么要这么写]: 调用自定义的 skip_check_imports 函数.
    # [解决的问题]: 许多自定义模型(如 Qwen, DeepSeek, Mixtral)在远程代码(remote code)中硬编码了
    # `import flash_attn` 或其他算子库. 如果用户环境没装这些库, 原生 transformers 会在加载模型文件时直接报错.
    # 这一行通过动态修改 transformers 的导入检查逻辑, 强制跳过这些非必要的静态导入检查,
    # 确保在缺少可选算子的情况下依然能把模型骨架加载起来.
    skip_check_imports()

    # 2. 跨生态 Hub 适配与路径重定向 (Hub Redirection & Path Resolution)
    # [为什么要这么写]: 通过检测环境变量或配置, 尝试从 ModelScope (魔搭) 或 OpenMind 等国内镜像 Hub 寻找模型.
    # [解决的问题]:
    #   a. 解决网络连通性: 中国区用户访问 Hugging Face 往往不稳定, 此举允许框架自动切换到国内可访问的下载源.
    #   b. 路径一致性: 确保一旦下载成功, 原地修改 model_args.model_name_or_path 为本地缓存路径.
    #      这样后续所有的加载操作(配置、分词器、权重)都直接指向本地, 避免重复触发 Hub 检查逻辑.
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    """
    print(model_args.model_name_or_path)
    /root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507
    """

    # 3. 统一参数封装 (Encapsulation for Transformers API)
    # [为什么要这么写]: 返回一个符合 Hugging Face `from_pretrained` 接口规范的字典.
    # [解决的问题]:
    #   - trust_remote_code: 必不可少. 现代 LLM 往往包含自定义建模代码, 必须显式授权框架运行远程 py 文件.
    #   - cache_dir: 资源管理. 确保在多进程(分布式训练)环境下, 所有 Rank 共享同一个模型缓存目录, 避免冗余存储.
    #   - revision/token: 确保版本控制与闭源模型(如 Llama-3 官方权重)的访问权限.
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Load pretrained tokenizer and optionally loads processor.
    Note: including inplace operation of model_args.

    在 LLM 微调工程中, 分词器(Tokenizer)是连接原始文本与神经网络的桥梁.
    这段代码的设计体现了极致的鲁棒性和对多模态模型(Multi-modal)的前瞻性支持.
    它解决了 Hugging Face transformers 库在加载不同厂商模型时可能遇到的兼容性、性能与类型识别问题.

    加载预训练的分词器, 并可选地加载多模态处理器(Processor).
    Note: 包含对 model_args 的原地修改(例如下载路径重定向).

    高级研究员视角的架构点评:

    容错优先 (Fault-Tolerant Design):
    try-except 块的层层嵌套并非代码不优雅, 而是 LLM 社区现状的必然选择.
    开源模型的 tokenizer_config.json 质量良莠不齐, 这种"自动回退"设计是工业级微调框架的核心竞争力, 它保证了极高的模型覆盖率.

    强制右填充 (Explicit Padding Side):
    在 load_tokenizer 中硬编码 padding_side="right" 是经过深思熟虑的.
    尽管在生成推理阶段常用 left 填充, 但在指令微调 (SFT) 训练阶段, 使用 right 填充配合 IGNORE_INDEX 掩码能保证计算梯度的准确性, 并有效利用计算算子的流水线加速.

    类型隔离 (Type Isolation):
    对 processor.__class__.__name__ 的检查解决了 Hugging Face 内部 API 语义模糊的问题, 防止了后续在处理图像/视频数据时, 因拿到了一个错误的"伪处理器"而产生难以排查的 AttributeError.
    """

    # 1. 获取基础初始化参数 (trust_remote_code, token, cache_dir 等)
    init_kwargs = _get_init_kwargs(model_args)
    """
    print(init_kwargs)
    {'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'token': None}
    print(model_args.model_name_or_path)
    /root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507
    print(model_args.use_fast_tokenizer)
    True
    print(model_args.split_special_tokens)
    False
    """

    # 2. 健壮性加载 Tokenizer (The Robust Loading Strategy)
    # [为什么要这么写]: Fast Tokenizer (基于 Rust) 虽然快, 但并非所有模型都完美支持.
    # [解决的问题]: 某些模型的远程代码或配置与 Fast Tokenizer 存在冲突. 如果用户指定的 use_fast
    # 模式导致 ValueError(通常是由于不支持的特性或路径问题), 我们自动切换到另一种模式进行重试.
    # 这确保了代码不会因为一个布尔值的配置错误而中断, 极大提升了用户体验.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",  # 强制设置右填充, 解决因果语言模型训练时的显存对齐与效率问题
            **init_kwargs,
        )
    except ValueError:  # try another one 如果模式 A 失败, 尝试模式 B
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    # 3. 动态修复逻辑 (Monkey Patching)
    # [为什么要这么写]: LLama-3、Qwen 等模型在原始权重中可能存在特殊的 Token 定义缺陷.
    # [解决的问题]: patch_tokenizer 会负责修复 pad_token 缺失、增加缺失的特殊标记、
    # 以及针对特定架构(如 GLM)进行词表长度对齐. 这是确保微调不崩溃的"保险丝".
    patch_tokenizer(tokenizer, model_args)

    # 4. 多模态支持 (Multimodal & Processor Integration)
    # [为什么要这么写]: 对于 LLaVA、Qwen-VL 等多模态模型, 不仅需要 Tokenizer 还需要 Processor
    # 来处理图像/音频特征.
    # [解决的问题]: 如果模型包含多模态组件, 系统会自动加载. 即便加载失败(例如纯文本模型),
    # 也会捕获异常并返回 None, 确保了框架在文本和多模态任务间的统一性.
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    """
    print(processor)
    Qwen2TokenizerFast(name_or_path='/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507', vocab_size=151643, model_max_length=1010000, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
        151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151657: AddedToken("<tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151658: AddedToken("</tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151665: AddedToken("<tool_response>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151666: AddedToken("</tool_response>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151667: AddedToken("<think>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
        151668: AddedToken("</think>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
    }
    )
    print(processor is not None)
    True
    print(processor.__class__.__name__)
    Qwen2TokenizerFast
    """

    # 5. 修正 Hugging Face AutoProcessor 的回退行为
    # [为什么要这么写]: 这是一个深度工程 Hack.
    # [解决的问题]: 在 transformers 库中, 如果找不到处理器配置, AutoProcessor
    # 有时会错误地返回一个 Tokenizer 实例. 为了避免上层逻辑(如 DataCollator)
    # 误以为拿到了多模态处理器, 我们检查类名. 如果它不是真正的 Processor, 则将其丢弃.
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    # 如果存在多模态处理器, 则对其进行必要的配置校准和 Tokenizer 同步
    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    # 返回包含两个核心组件的模块字典, 供后续加载 Model 和 Dataset 使用
    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Load model config.

    在 LLM 微调流水线中, 配置加载(Configuration Loading) 是所有操作的第一步. 它不仅仅是读取一个 JSON 文件, 更是在为后续的权重加载、显存分配、算子优化(如 RoPE Scaling)定下基调.

    加载模型的预训练配置(config.json).

    [设计逻辑]:
    这是模型加载流水线的"哨兵"步骤. 通过先加载 Config 而非直接加载 Model,
    我们可以获取模型的关键元数据(如隐藏层维度、层数、词表大小), 从而在不占用
    巨大显存的情况下完成参数校验和微调策略(如 LoRA 目标模块匹配)的准备工作.

    高级研究员视角的工程细节补充:

    权限透传 (Token Pass-through):
    通过 init_kwargs 传入 token 解决了闭源或受限权重(如 Llama-3-70B 官方权重)的下载授权问题, 避免了手动设置环境变量的繁琐步骤.

    原地修改的风险控制:
    注意 _get_init_kwargs 内部会修改 model_args.model_name_or_path. 这意味着 load_config 执行完后, model_args 里的路径已经从"Hub 标识符"(如 meta-llama/Llama-3-8b)变成了"本地磁盘路径". 这种设计在分布式环境下尤为重要, 确保了主进程下载完成后, 子进程能直接定位到文件.

    为量化/分布式做预判:
    虽然代码里没体现, 但高级开发者通常会在 load_config 之后立即检查 config.torch_dtype. 如果配置文件指定了 bfloat16 而用户硬件仅支持 float16, 我们可以在正式加载数 GB 权重前就抛出预警.
    """

    # 1. 统一初始化参数准备 (Parameters Standardization)
    # [为什么要这么写]: 调用内部工具函数 _get_init_kwargs.
    # [解决的问题]: 确保在加载 Config、Tokenizer 和 Model 时, 底层参数(如 trust_remote_code,
    # token, revision)保持严格的一致性. 同时, 该函数内部可能包含对模型路径的重定向逻辑(例如
    # 从 ModelScope 下载), 这确保了后续 AutoConfig 能够找到正确的本地路径.
    init_kwargs = _get_init_kwargs(model_args)

    # 2. 动态模型配置探测 (Dynamic Architecture Probing)
    # [为什么要这么写]: 利用 Hugging Face 的 AutoConfig 类.
    # [解决的问题]:
    #   a. 架构无关性: 无论用户加载的是 Llama-3、Qwen、Mistral 还是 DeepSeek, 该接口都能自动匹配
    #      对应的模型类, 实现"一套代码微调百模".
    #   b. 远程代码支持: 通过 init_kwargs 中的 trust_remote_code 字段, 解决了许多自定义架构
    #      (例如带有特殊 Attention 实现的模型)无法在原生 transformers 库中加载的痛点,
    #      允许系统执行模型目录下的自定义建模脚本.
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Load pretrained model.

    load_model 函数是整个框架的"心脏". 它不仅负责把模型加载进显存, 还承担了算子优化、多模态分发、PEFT 适配以及 RLHF 价值头注入等繁重任务.
    这个函数之所以写得如此复杂, 是为了在兼容上百种模型的同时, 还要榨干 GPU/NPU 的每一分性能.

    加载预训练模型, 并进行一系列工程化增强.

    深度解析(研究员视角):

    为什么不直接用 AutoModel.from_pretrained?
    在 LLM 开发中, 标准的 API 太过"笨重". LLaMA-Factory 在加载权重前, 通过 patch_config 和 patch_model 插入了大量的中间件逻辑, 这让它能支持 8-bit、4-bit 量化训练, 以及应对某些开源模型配置文件(Config)不标准的问题.

    多模态设计的精妙之处:
    你会发现代码中通过 type(config) in ...model_mapping.keys() 进行分发. 这体现了配置驱动架构. 只要 Hugging Face 的生态增加了一种新架构, LLaMA-Factory 几乎可以零代码修改支持该模型.

    对 RLHF 的原生支持:
    AutoModelForCausalLMWithValueHead 的注入说明 LLaMA-Factory 从底层就把 RM(奖励模型)和 PPO 任务作为一等公民对待, 而不是像其他框架那样通过外部包裹(Wrapper)来实现, 这保证了梯度流的稳定性.
    """

    # 1. 初始化标准参数
    # [为什么要这么写]: 统一获取 trust_remote_code, cache_dir, token 等基础参数.
    # [解决的问题]: 确保后续加载 Config/Tokenizer/Model 时, 底层通信协议和权限保持高度一致.
    init_kwargs = _get_init_kwargs(model_args)
    """
    print(init_kwargs)
    {'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'token': None}
    """

    # 2. 预先加载配置
    # [为什么要这么写]: 在加载数 GB 的权重前, 先通过 Config 嗅探模型的架构、层数和维度.
    config = load_config(model_args)
    """
    print(config)
    Qwen3Config {
        "architectures": [
            "Qwen3ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "layer_types": [
            "full_attention",
            "full_attention",
            ...
        ],
        "max_position_embeddings": 262144,
        "max_window_layers": 36,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 5000000,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "transformers_version": "4.57.1",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    }
    print(config.layer_types)
    ['full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention', 'full_attention']
    """

    # 3. 动态配置修正 (Monkey Patching)
    # [为什么要这么写]: LLama-3/Qwen 等模型在长文本支持(RoPE Scaling)上可能需要动态补丁.
    # [解决的问题]: 修正原始 Config 里的错误, 或根据 model_args 注入自定义的上下文窗口大小、词表长度.
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    # 4. 高性能算子融合 (Liger Kernel)
    # [为什么要这么写]: 集成 LinkedIn 开源的 Liger Kernel.
    # [解决的问题]: 大模型训练中 CrossEntropy 和 LayerNorm 极其吃显存. 通过算子融合(Triton 实现),
    # 在不损失精度的情况下, 能减少约 20%-40% 的 VRAM 占用, 并提升训练吞吐量.
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False

    # 5. 特定加速后端分发 (KTransformers / Unsloth)
    # [为什么要这么写]: 针对不同的硬件和加速库进行特殊加载.
    # [解决的问题]:
    #   - KTransformers: 解决本地异构算力(如 CPU+GPU 混合)下的推理加速.
    #   - Unsloth: 极致的显存优化黑魔法. Unsloth 需要在模型初始化阶段就介入, 以便替换底层注意力机制.
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            # 如果已有 Adapter, 延迟加载(Lazy Load), 交给 PEFT 统一处理
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    # 6. 通用模型加载与多模态分发
    # [为什么要这么写]: 根据 Config 的具体模型类名, 决定调用哪个 AutoModel.
    # [解决的问题]: Hugging Face 并不存在一个真正的"万能 AutoModel".
    # 我们需要根据 Config 自动判断该模型是: 图像-文本(LLaVA/Qwen-VL)、视频-文本、纯文本、还是音频-文本.
    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        init_kwargs["torch_dtype"] = "auto"  # 自动检测 dtype(通常是 bf16/fp16)

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            # 多模态分发链条 (Multi-modal Dispatching)
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text 常见图文大模型
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text 视觉序列模型
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text T5/BART 架构
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen omni 语音模型适配
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM  # 默认 Decoder-only 架构

            if model_args.train_from_scratch:
                # 随机初始化权重(用于从零开始训练)
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                # 加载预训练权重
                model = load_class.from_pretrained(**init_kwargs)
                # 针对 Qwen Omni 等复杂架构的 Hack
                # 这些模型通常有一个 wrapper 包装了内部的推理核心, 需要拆解出来进行微调.
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        # 处理 MoD (Mixture of Depths) 的转换逻辑
        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    # 7. 模型后处理与 AutoClass 注册
    # [为什么要这么写]: 在模型载入显存后, 修复层名、Embedding 权重对齐.
    # [解决的问题]: register_autoclass 确保模型在保存后, 依然能被 Hugging Face 的原版 pipeline 识别.
    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    # 8. Adapter 初始化 (LoRA / QLoRA / DoRA)
    # [为什么要这么写]: 通过 init_adapter 注入微调层.
    # [解决的问题]: 将冻结的模型参数转化为可训练的低秩矩阵. LLaMA-Factory 支持动态合并
    # 多个 Adapter 以及管理多轮微调的检查点加载.
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # 9. RLHF 价值头处理 (ValueHead for PPO/RM)
    # [为什么要这么写]: RLHF 任务需要模型输出一个标量分数(Reward).
    # [解决的问题]: Base 模型通常没有输出层来预测分数. 这里动态挂载一个 ValueHead 层,
    # 并尝试加载保存的 vhead 参数. 由于 vhead 属于自定义层, 必须使用 strict=False
    # 来防止加载时因多出来的权重参数而抛出异常.
    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    # 10. 状态控制
    # [为什么要这么写]: 显式调用 requires_grad_.
    # [解决的问题]: 在推理(Evaluaton)阶段, 绝对禁止权重更新, 节约显存和算力.
    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    # 11. 未来特性集成 (V1 Kernels)
    # [为什么要这么写]: 引入下一代架构中的 NPU 融合算子.
    # [解决的问题]: 为国产算力芯片(如华为昇腾 NPU)提供底层的极致性能优化.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = apply_default_kernels(model=model, include_kernels=model_args.use_v1_kernels)

    # 12. 统计报告 (Parameter Statistics)
    # [为什么要这么写]: 准确计算当前模型中有多少参数是真正可训练的.
    # [解决的问题]: 帮助用户验证 LoRA 是否设置成功. 如果是全量微调, 可训练占比应为 100%;
    # 如果是 LoRA, 通常在 0.1%-1% 之间, 这给用户直观的反馈, 避免配置错误带来的算力浪费.
    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    # 13. 调试输出 (Param Status Printing)
    # [为什么要这么写]: 仅在 Local Rank 0 打印详细的参数状态.
    # [解决的问题]: 解决分布式训练中多进程打印冲突, 清晰展示每一层的 dtype(如 fp16 还是 4bit)
    # 和设备分布, 对于排查分布式训练死锁和精度溢出至关重要.
    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model

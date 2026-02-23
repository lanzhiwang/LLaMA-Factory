# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and Optimum library.
# https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/utils/quantization_config.py
# https://github.com/huggingface/optimum/blob/v1.20.0/optimum/gptq/data.py
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
import random
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, EetqConfig, GPTQConfig, HqqConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ...extras import logging
from ...extras.constants import FILEEXT2TYPE, QuantizationMethod
from ...extras.misc import check_version, get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _get_quantization_dataset(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> list[dict[str, Any]]:
    r"""
    Prepare the tokenized dataset to perform AutoGPTQ. Do not use tensor output for JSON serialization.

    在 LLM 微调工程中, 微调后的模型如果需要部署在显存受限的设备上, 通常需要进行 PTQ(训练后量化). 这段 _get_quantization_dataset 函数的作用是为 AutoGPTQ 算法准备"校准数据集". 校准数据的质量直接决定了量化后模型是否会"变傻"(即精度掉点).

    准备用于执行 AutoGPTQ 的分词后的数据集.

    [核心目的]:
    GPTQ 算法需要一小部分(通常是 128-512 条)真实的文本数据来观察模型各层的激活值分布(Activations).
    通过这些统计信息, 算法可以计算出最合适的缩放因子, 从而最小化量化带来的精度损失.

    高级研究员视角的架构解析:

    为什么需要这部分代码?
    很多微调工具在导出 4-bit 模型时直接使用随机噪声校准, 这会导致模型生成能力大幅下降. LLaMA-Factory 坚持让用户提供 export_quantization_dataset, 是为了通过真实数据的分布来指导量化过程.

    工程上的"黑魔法":
    pad_to_multiple_of 在这里没有被使用, 取而代之的是严格的 random.randint 裁剪. 这是因为 GPTQ 的校准过程对输入形状的对齐要求很高, 固定的 maxlen 有助于稳定每一批次的校准梯度.

    对 maxlen 的权衡:
    注意 TODO: fix large maxlen. 在处理 128k 甚至更长的模型时, 如果 export_quantization_maxlen 设得太大, 这里的采样逻辑会因为找不到足够长的单条文本而报错. 作为高级开发工程师, 我们需要提醒用户: 校准集的序列长度通常不需要达到模型上限, 2048 通常已足够捕捉权重的重要分布.
    """

    # 1. 动态数据源路径解析 (Handling Diverse Data Sources)
    # [为什么要这么写]: LLaMA-Factory 支持从 Hugging Face Hub 在线加载或从本地文件加载.
    # [解决的问题]: 屏蔽文件系统差异. 如果是本地文件, 通过后缀名映射到 datasets 库识别的类型(如 json, jsonl, csv).
    if os.path.isfile(model_args.export_quantization_dataset):
        data_path = FILEEXT2TYPE.get(model_args.export_quantization_dataset.split(".")[-1], None)
        data_files = model_args.export_quantization_dataset
    else:
        data_path = model_args.export_quantization_dataset
        data_files = None

    # 加载数据集
    dataset = load_dataset(
        path=data_path,
        data_files=data_files,
        split="train",
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
    )

    samples = []
    # 获取量化所需的样本长度(通常为 2048 或 4096)
    maxlen = model_args.export_quantization_maxlen

    # 2. 代表性样本采集 (Representative Sampling)
    # [为什么要这么写]: 遍历采集指定数量(nsamples)的样本.
    for _ in range(model_args.export_quantization_nsamples):
        n_try = 0
        # 3. 健壮性质量控制 (Quality Control Loop)
        # [为什么要这么写]: 进入死循环直至找到一个足够长的样本.
        # [解决的问题]: 校准数据如果过短(例如只有几个词), 无法让模型激活足够的神经元, 会导致量化统计失真.
        # 这里的 n_try 计数器防止在全是短文本的数据集中陷入死循环.
        while True:
            if n_try > 100:
                raise ValueError("Cannot find satisfying example, considering decrease `export_quantization_maxlen`.")

            # 随机采样, 确保校准数据覆盖了原始数据的随机分布
            sample_idx = random.randint(0, len(dataset) - 1)
            sample: dict[str, torch.Tensor] = tokenizer(dataset[sample_idx]["text"], return_tensors="pt")
            n_try += 1
            # 只有当样本分词后的长度大于我们要求的 maxlen 时, 才符合裁剪条件
            if sample["input_ids"].size(1) > maxlen:
                break  # TODO: fix large maxlen

        # 4. 随机片段裁剪 (Random Cropping)
        # [为什么要这么写]: 在长文本中随机选择一个起始点 word_idx.
        # [解决的问题]: 如果每次都从句首(BOS token 处)截取, 量化参数会过度拟合于句首的激活特征.
        # 随机裁剪能让校准过程覆盖文本的各种上下文位置, 使量化后的权重更加稳健(Robust).
        word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)
        input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]
        attention_mask = sample["attention_mask"][:, word_idx : word_idx + maxlen]

        # 5. 序列化兼容性处理 (Serialization Readiness)
        # [为什么要这么写]: 将 Tensor 转回 Python List.
        # [解决的问题]: 正如 docstring 所述, 为了后续可能的 JSON 序列化或其他跨进程通讯,
        # 避免返回带有 CUDA/Grad 信息的复杂 PyTorch 对象, 确保结果是"纯净数据".
        samples.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()})

    return samples


def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    init_kwargs: dict[str, Any],
) -> None:
    r"""
    Priority: PTQ-quantized (train/infer) > AutoGPTQ (export) > On-the-fly quantization (train/infer).

    在 LLM 领域, 量化(Quantization)不仅是为了减少显存占用, 更涉及到训练精度与计算效率的博弈. 该函数设计了一个复杂的优先级逻辑, 旨在支持从早期的 BitsAndBytes (QLoRA) 到最新的 MXFP4、FP8 以及高效的 GPTQ 导出方案.

    配置量化策略.
    优先级: PTQ 已量化模型(加载) > AutoGPTQ 导出(量化并保存) > 运行时即时量化(训练/推理).

    资深专家视角下的架构点评:

    版本防御性 (Defensive Versioning):
    代码中大量使用了 check_version. 这是因为 LLM 加速库(如 bitsandbytes, gptqmodel)更新极快, 旧版本可能包含严重的显存泄露或精度 Bug. 在量化配置阶段就卡住版本, 比训练到一半报错要专业得多.

    解耦计算与存储:
    特别是在 QLoRA 部分, 代码深刻体现了"存储精度(4-bit)"与"计算精度(compute_dtype)"的区分. 这种区分是 LLaMA-Factory 能在 24G 显存上微调大尺寸模型的底层保障.

    多框架兼容性 (Framework Agnostic):
    通过对 is_deepspeed_zero3_enabled 和 is_fsdp_enabled 的判断, 函数实现了"一份配置, 多套引擎适配". 它解决了一个常见的初学者痛点: 为什么同样的量化配置, 单卡能跑, DeepSpeed 就会卡死.
    """

    # ---------------------------------------------------------------------------------
    # 第一部分: 处理已经是 PTQ(Post-Training Quantization)量化过的模型
    # ---------------------------------------------------------------------------------
    if getattr(config, "quantization_config", None):  # ptq
        # [为什么要这么写]: 用户可能在命令行误传了 --quantization_bit.
        # [解决的问题]: 对于已量化模型(如 GPTQ/AWQ 权重文件), 比特数已固化在权重中, 此参数不再生效, 需通过日志告知用户.
        if model_args.quantization_bit is not None:
            logger.warning_rank0("`quantization_bit` will not affect on the PTQ-quantized models.")

        quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        # [解决的问题]: 分布式兼容性陷阱.
        # 传统的 GPTQ/AWQ 算子通常不支持 DeepSpeed ZeRO-3 或 FSDP 的参数分片(Sharding).
        # 因为这些算子需要在计算前获取完整的量化权重和缩放因子, 而 Z3 会把参数切碎分布在不同显卡上.
        if quant_method not in (QuantizationMethod.MXFP4, QuantizationMethod.FP8) and (
            is_deepspeed_zero3_enabled() or is_fsdp_enabled()
        ):
            # mxfp4 will dequant the model weights
            raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")

        # [为什么这么写]: 处理最新的 MXFP4 (Microscaling) 格式.
        # [解决的问题]: MXFP4 是一种高性能存储格式, 但在计算时通常需要"反量化"回高精度(dequantize=True).
        # 设置 ignore_mismatched_sizes 是为了防止由于反量化后数据类型长度变化导致的加载报错.
        if quant_method == QuantizationMethod.MXFP4:
            from transformers import Mxfp4Config

            quant_config = Mxfp4Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.FP8:
            from transformers import FineGrainedFP8Config

            quant_config = FineGrainedFP8Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        # [为什么这么写]: 针对 GPTQ 模型的 Exllama 补丁.
        # [解决的问题]: Exllama 算子在训练过程中极不稳定, 且不支持某些分布式场景.
        # 强制关闭 use_exllama, 转而使用更为稳健的 gptqmodel 内部算子.
        if quant_method == QuantizationMethod.GPTQ:
            check_version("gptqmodel>=2.0.0", mandatory=True)
            quantization_config.pop("disable_exllama", None)  # remove deprecated args
            quantization_config["use_exllama"] = False  # disable exllama

        if quant_method == QuantizationMethod.AWQ:
            check_version("autoawq", mandatory=True)

        if quant_method == QuantizationMethod.AQLM:
            check_version("aqlm>=1.1.0", mandatory=True)
            quantization_config["bits"] = 2

        # ... (AWQ/AQLM 版本检查略) ...
        quant_bits = quantization_config.get("bits", "?")
        logger.info_rank0(f"Loading {quant_bits}-bit {quant_method.upper()}-quantized model.")

    # ---------------------------------------------------------------------------------
    # 第二部分: 处理模型量化导出(通常用于模型微调后的压缩部署)
    # ---------------------------------------------------------------------------------
    elif model_args.export_quantization_bit is not None:  # gptqmodel
        if model_args.export_quantization_bit not in [8, 4, 3, 2]:
            raise ValueError("AutoGPTQ only accepts 2/3/4/8-bit quantization.")

        # [解决的问题]: GPTQ 算法需要校准数据集.
        # 这里通过 _get_quantization_dataset 准备少量真实数据, 让 GPTQ 观察激活分布, 从而最小化量化误差.
        check_version("optimum>=1.24.0", mandatory=True)
        check_version("gptqmodel>=2.0.0", mandatory=True)
        from accelerate.utils import get_max_memory

        if getattr(config, "model_type", None) == "chatglm":
            raise ValueError("ChatGLM model is not supported yet.")

        # [为什么要这么写]: 手动修复特定模型的 Block Pattern.
        # [解决的问题]: Optimum 库内部硬编码了查找 Transformer 层的模式.
        # Gemma3/PaliGemma 等新型号采用了非标准命名(如 language_model.model.layers).
        # 动态插入此路径可以确保 GPTQ 能正确识别到需要量化的权重层.
        try:
            from optimum.gptq import utils as gq_utils

            if "language_model.model.layers" not in gq_utils.BLOCK_PATTERNS:
                gq_utils.BLOCK_PATTERNS.insert(0, "language_model.model.layers")
        except ImportError:
            pass

        block_name_to_quantize = None
        if getattr(config, "model_type", None) in ["gemma3", "paligemma"]:
            block_name_to_quantize = "language_model.model.layers"

        # 配置 GPTQ 导出参数, 强制使用 float16, 因为大多数 GPTQ 算子在 FP16 下校准最准.
        init_kwargs["quantization_config"] = GPTQConfig(
            bits=model_args.export_quantization_bit,
            tokenizer=tokenizer,
            dataset=_get_quantization_dataset(tokenizer, model_args),
            block_name_to_quantize=block_name_to_quantize,
        )
        init_kwargs["device_map"] = "auto"
        init_kwargs["max_memory"] = get_max_memory()
        model_args.compute_dtype = torch.float16  # force fp16 for gptqmodel
        logger.info_rank0(f"Quantizing model to {model_args.export_quantization_bit} bit with GPTQModel.")

    # ---------------------------------------------------------------------------------
    # 第三部分: 即时量化(On-the-fly), 最典型的场景是 QLoRA 训练
    # ---------------------------------------------------------------------------------
    elif model_args.quantization_bit is not None:  # on-the-fly
        if model_args.quantization_method == QuantizationMethod.BNB:
            # [核心逻辑]: BitsAndBytes (QLoRA) 配置.
            if model_args.quantization_bit == 8:
                check_version("bitsandbytes>=0.37.0", mandatory=True)
                init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif model_args.quantization_bit == 4:
                # [解决的问题]: FSDP 与 QLoRA 的兼容性黑魔法.
                # 必须设置 bnb_4bit_quant_storage=compute_dtype, 这决定了权重在内存中解压后的精度.
                # 如果不设置, FSDP 在收集分片参数时会因为精度不匹配而导致通讯死锁.
                check_version("bitsandbytes>=0.39.0", mandatory=True)
                init_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quantization,
                    bnb_4bit_quant_type=model_args.quantization_type,
                    bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp+qlora
                )
            else:
                raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

            # [为什么要这么写]: 分布式环境下设备映射的特殊处理.
            # [解决的问题]: 在 ZeRO-3 或 FSDP 下, 模型权重的物理位置由框架调度.
            # 如果此时传入 device_map="auto", Huggingface 可能会把权重锁死在某张卡上, 破坏 Z3 的分片逻辑.
            # 因此, 在分布式训练时绝对不能传入固定 device_map；仅在单卡推理/普通 DDP 时指定.
            # Do not assign device map if:
            # 1. deepspeed zero3 or fsdp (train)
            # 2. auto quantization device map (inference)
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":
                if model_args.quantization_bit != 4:
                    raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

                check_version("bitsandbytes>=0.43.0", mandatory=True)
            else:
                # 强制指定当前设备, 避免 Auto 逻辑在复杂环境中选错卡.
                init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference

            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with bitsandbytes.")
        elif model_args.quantization_method == QuantizationMethod.HQQ:
            if model_args.quantization_bit not in [8, 6, 5, 4, 3, 2, 1]:
                raise ValueError("HQQ only accepts 1/2/3/4/5/6/8-bit quantization.")

            # [解决的问题]: HQQ 是一种不依赖校准集的快速量化方案.
            # 同样, 它也目前不支持 ZeRO-3/FSDP 等需要参数动态聚合的训练框架.
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("HQQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            check_version("hqq", mandatory=True)
            # 使用 axis=0 的 ATEN kernel 是为了在推理时获得更好的计算亲和性和吞吐量.
            init_kwargs["quantization_config"] = HqqConfig(
                nbits=model_args.quantization_bit, quant_zero=False, quant_scale=False, axis=0
            )  # use ATEN kernel (axis=0) for performance
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with HQQ.")
        elif model_args.quantization_method == QuantizationMethod.EETQ:
            if model_args.quantization_bit != 8:
                raise ValueError("EETQ only accepts 8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("EETQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            check_version("eetq", mandatory=True)
            init_kwargs["quantization_config"] = EetqConfig()
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with EETQ.")

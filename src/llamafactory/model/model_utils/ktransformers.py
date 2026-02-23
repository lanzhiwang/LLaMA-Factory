# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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

import importlib.util as _u
from typing import TYPE_CHECKING, Any

import torch

from ...extras import logging
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments, ModelArguments

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


KT_AVAILABLE = _u.find_spec("ktransformers") is not None
if KT_AVAILABLE:
    from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
    from ktransformers.models.modeling_llama import LlamaForCausalLM
    from ktransformers.models.modeling_mixtral import MixtralForCausalLM
    from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
    from ktransformers.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
    from ktransformers.optimize.optimize import optimize_and_load_gguf
    from ktransformers.server.config.config import Config
    from ktransformers.sft.lora import inject_lora_layer
    from ktransformers.util.custom_loader import GGUFLoader, SafeTensorLoader
    from ktransformers.util.globals import GLOBAL_CONFIG
    from ktransformers.util.utils import load_weights

logger = logging.get_logger(__name__)


def _get_kt_kwargs(
    config: "PretrainedConfig",
    model_name_or_path: str,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> dict[str, Any]:
    return {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "full_finetuning": finetuning_args.finetuning_type == "full",
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,
        "use_gradient_checkpointing": "ktransformers",
    }


def load_kt_pretrained_model(config: "PretrainedConfig", model_args: "ModelArguments") -> "PreTrainedModel":
    r"""
    Optionally load pretrained model with KTransformers. Used in training.

    这段代码是 LLaMA-Factory 深度集成 KTransformers (简称 KT) 的核心入口.
    KTransformers 是由 AI-Step 团队开发的一个高性能推理/训练加速框架, 其核心优势在于通过极致的异构计算(CPU+GPU 混合)和自定义算子, 在消费级硬件上运行/微调超大规模模型(如 DeepSeek-V3, Qwen-MoE 等).

    使用 KTransformers 可选地加载预训练模型.
    [核心目的]: 利用 KTransformers 的异构调度能力, 将模型的部分参数(如 MoE 的专家层)
    合理分配到 CPU 内存或 GPU 显存中, 以支持在受限显存环境下微调超大参数量模型.

    高级研究员视角的架构解析:

    关于 Meta Device 的必要性:
    在微调 DeepSeek-V3 这种拥有 671B 参数的模型时, 普通的加载方式(即便是分布式)也会因为初始化时的 CPU 内存峰值而失败. 使用 torch.device("meta") 配合 optimize_and_load_gguf 实现了"边构建边加载边优化"的流式过程, 极大地提升了启动成功率.

    GGUF 的集成意义:
    通常微调框架只支持 Hugging Face 的 .safetensors 或 .bin. 引入 GGUF 支持意味着 LLaMA-Factory 可以直接微调那些已经在推理界高度普及的轻量化/量化后的模型, 极大地降低了个人开发者尝试大模型微调的门槛.

    异构计算的复杂性:
    Config().cpu_infer 的注入说明系统不仅在操作内存, 还在操作 CPU 的线程池和计算后端(如 OpenBLAS 或 MKL). 这种跨语言(Python 与 C++)的状态同步是高级 LLM 框架的标志性特征.
    """

    # 1. 显式映射自定义建模类 (Custom Modeling Injection)
    # [为什么要这么写]: Hugging Face 官方的 modeling 实现通常只考虑了通用兼容性.
    # KTransformers 针对这些特定架构(尤其是 DeepSeek V2/V3 和 Qwen MoE)提供了
    # 深度优化的本地 Python 类, 内部注入了高性能计算内核(如优化的 MoE Router 和算子).
    custom_models = {
        "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
        "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
        "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
        "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
        "LlamaForCausalLM": LlamaForCausalLM,
        "MixtralForCausalLM": MixtralForCausalLM,
    }

    # 2. 配置全局单例 (Global Engine Configuration)
    # [解决的问题]: KTransformers 的底层(C++ 层)需要知道硬件分配策略.
    # 通过 Config() 单例注入 cpu_infer(CPU 卸载比例)和 chunk_size(计算分块大小),
    # 确保在模型实例创建前, 加速引擎已完成硬件感知.
    Config().cpu_infer = model_args.cpu_infer
    Config().chunk_size = model_args.chunk_size
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)

    # 3. 动态 Dtype 控制 (Numerical Stability vs Memory)
    # [为什么要这么写]: 长文本模式(Long Context)会消耗海量的 KV Cache 空间.
    # 强制设为 float16 是为了在长序列训练中防止由于 FP32 或 BF16 导致的显存溢出,
    # 同时维持 Llama 类模型在长序列下的数值稳定性.
    if model_args.mode == "long_context":
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        # 默认模式尊重模型原始 config, 以保证微调精度不丢失
        torch.set_default_dtype(config.torch_dtype)

    # 4. Meta Device 延迟初始化 (Efficient Instantiation)
    # [为什么要这么写]: 使用 torch.device("meta").
    # [解决的问题]: 这是加载超大规模模型(如 DeepSeek-V3, 几百 GB 权重)的唯一工业级方案.
    # 在 meta 设备上初始化模型, 只构建"计算图骨架"而不分配真实内存和显存.
    # 如果直接在 GPU 加载, 即便显存够, 漫长的内存拷贝也会导致程序初始化超时崩溃(OOM).
    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            # 注意力算子适配 (Kernel Dispatching)
            # Qwen2Moe 在 Eager 模式下由于专家路由的 Logits 范围问题极易出现溢出(Overflow),
            # 必须强制使用 FlashAttention-2 的稳定数学实现.
            if "Qwen2Moe" in config.architectures[0]:  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"  # Llama 配合 KT 时通常使用其 MonkeyPatch 后的 eager 实现
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"

            # 使用 KT 自定义的建模类实例化模型
            model = custom_models[config.architectures[0]](config)
        else:
            # 兜底方案: 使用官方 AutoModel 类
            attn_implementation = "flash_attention_2"
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation=attn_implementation
            )

    # 5. GGUF 权重加载与 YAML 规则优化 (Quantization & Rule-based Optimization)
    # [核心工程思想]: KTransformers 的精髓在于它支持加载 GGUF 格式(被 llama.cpp 广泛使用的格式),
    # 并通过一个 YAML 规则文件(kt_optimize_rule)精确控制每一层的优化策略.
    optimize_config_path = model_args.kt_optimize_rule
    gguf_path = model_args.model_name_or_path

    assert optimize_config_path is not None, "optimize_config_path must be provided (path to YAML rules file)."
    assert gguf_path is not None, "gguf_path must be provided (path to a folder or .gguf file)."

    # 设置模式为推理/微调准备阶段
    GLOBAL_CONFIG._config["mod"] = "infer"

    # [解决的问题]: 这是最关键的一步. 它会解析 YAML 规则,
    # 将模型中的某些算子替换为 KT 的 C++ 高性能实现, 并分块加载 GGUF 权重.
    # 这使得一个 175B 的模型可以"智能地"把一部分专家层扔在 32GB 显存里, 剩下的扔在 128GB 内存里运行.
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)

    return model


def get_kt_peft_model(model: "PreTrainedModel", peft_kwargs: dict[str, Any]) -> "PreTrainedModel":
    r"""Get the peft model for the pretrained model with KTransformers. Used in training."""
    from ktransformers.sft.peft_utils.mapping import get_peft_model

    return get_peft_model(model, peft_kwargs)


def load_kt_peft_model(model_args: "ModelArguments", model: "PreTrainedModel") -> "PreTrainedModel":
    r"""
    Load peft model with KTransformers. Used in both training and inference.

    load_kt_peft_model 是 LLaMA-Factory 集成 KTransformers (KT) 框架时的关键逻辑. KTransformers 允许在异构设备(如 CPU+GPU 混合)上运行超大规模模型(如 DeepSeek-V3).
    由于 KTransformers 采用了高度自定义的算子(如基于 llama.cpp 的量化算子)和模型拓扑结构, 原生的 peft 库无法直接将其适配器应用到 KT 模型上. 这段代码解决的核心问题是: 如何在自定义的模型架构中, 手动"手术式"地注入并激活 LoRA 权重.

    使用 KTransformers 框架加载 PEFT (LoRA) 适配器.
    [设计动机]: KTransformers 的模型实例是经过 Monkey Patch 后的异构对象,
    标准的 PEFT 流程(如 PeftModel.from_pretrained)无法识别其内部的自定义线性层.
    本函数实现了手动解析权重并注入到 KT 自定义层中的逻辑.

    高级研究员视角下的架构解析:

    解耦加载与计算:
    这段代码并没有调用 peft.PeftModel. 作为高级研究人员, 我们意识到 KT 的核心是 C++ 编译的算子, peft 库的 forward 劫持机制(Hook)会导致 KT 无法利用其极致的推理加速(如 CPU/GPU 混合调度). 通过手动 inject 和 copy_, 我们让 LoRA 权重直接"生长"在 KT 的高性能算子内部.

    原地内存复制 (copy_) 的必要性:
    在分布式训练或大规模模型加载中, 重新分配内存(Re-allocation)是极其危险的. param.data.copy_ 确保了即便参数被 DeepSpeed 或 KTransformers 接管, 只要内存指针还在, 权重就能正确更新.

    对 GGUF 的前瞻支持:
    支持 .gguf 适配器体现了 LLaMA-Factory 试图打通"学术训练(HuggingFace)"与"工业推理(llama.cpp)"边界的尝试. 这意味着你可以直接加载社区中已经经过量化处理的轻量化适配器, 而无需先将其转回 FP32.
    """

    # 获取适配器路径
    load_adapter_name_or_path = model_args.adapter_name_or_path[0]

    # --- 分支 1: 处理 GGUF 格式的适配器 ---
    # [为什么要这么写]: GGUF 是一种高效的二进制权重格式, 常见于本地推理生态(llama.cpp).
    # [解决的问题]: 支持直接加载量化后的 LoRA 权重, 这对于显存极度受限的"端侧微调/推理"至关重要.
    if load_adapter_name_or_path.endswith(".gguf"):
        # 1. 结构注入: 在 KT 模型中动态创建 LoRA 层结构
        inject_lora_layer(model, load_adapter_name_or_path)
        # 2. 专用加载器: 解析 GGUF 文件的 Tensor Map
        adapter_gguf_loader = GGUFLoader(load_adapter_name_or_path)
        # 3. 权重对齐: 将 GGUF 的张量通过 KT 内部逻辑映射到 PyTorch 模型参数中
        load_weights(model, adapter_gguf_loader, adapter_gguf=True)
        # 确保模型处于训练状态, 以激活 Dropout 等逻辑(如果是训练任务)
        model.train()

    # --- 分支 2: 处理标准的 Safetensors 格式适配器 ---
    # [为什么要这么写]: 这是 Hugging Face 默认的 PEFT 保存格式.
    # [解决的问题]: 由于 KT 模型的参数命名空间可能与原生 Transformers 模型不同,
    # 这里采取了"手动遍历 + 键名转换"的策略来完成权重的跨架构迁移.
    else:
        # 在模型中注入 LoRA 结构(初始化 A 和 B 矩阵的容器)
        inject_lora_layer(model, load_adapter_name_or_path)

        # 使用 SafeTensorLoader 高效、安全地加载权重文件
        adapter_loader = SafeTensorLoader(load_adapter_name_or_path)
        # 探测当前模型所在的设备(CPU 或 GPU), 确保加载的 Tensor 与模型在同一位置
        device = next(model.parameters()).device

        # 遍历适配器文件中的所有权重键名
        for key in adapter_loader.tensor_file_map.keys():
            try:
                # 显式地将 Tensor 加载到目标设备, 避免内存跨设备复制的性能损失
                tensor = adapter_loader.load_tensor(key, device=device)

                # --- 核心逻辑: 键名重映射 (Key Mapping) ---
                # [解决的问题]: PEFT 存储的键名通常包含前缀 "base_model.model.".
                # KTransformers 模型在内部可能将这部分层级打平了(比如直接以 layers.0 开头).
                # 下面的逻辑通过字符串替换, 将 PEFT 的标准命名转化为 KT 识别的命名.
                model_key = key.replace("base_model.model.", "")

                # 处理 PEFT 适配器名称: 默认适配器在 PEFT 中标识为 ".default."
                # 这里的多次替换是为了处理嵌套情况, 确保最终键名符合 model.get_parameter() 的要求
                model_key = model_key.replace(".weight", ".default.weight")
                model_key = model_key.replace(".default.default.weight", ".default.weight")

                # 获取模型中对应的 Parameter 对象
                param = model.get_parameter(model_key)

                # [关键操作]: 使用原地复制 (In-place copy)
                # [为什么要这么写]: copy_ 能够直接修改底层存储, 而不会改变 Parameter 的引用.
                # 这样可以确保优化器(Optimizer)之前绑定的内存地址依然有效.
                param.data.copy_(tensor.data)

                print(f"Loaded adapter weight: {key} -> {model_key}")
            except AttributeError:
                # 容错处理: 某些 Meta-data 键名(如 rank, alpha)在文件中存在但不是 Parameter
                print(f"Skipping {key}: not a model parameter")
            except KeyError:
                # 架构差异警告: 如果适配器包含模型中没有的层, 输出提示但不中断流程
                print(f"Key not found in model: {model_key} (original: {key})")

    return model

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

import re
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, OFTConfig, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras import logging
from ..extras.constants import EngineName
from .model_utils.ktransformers import get_kt_peft_model, load_kt_peft_model
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    """
    在 LLM 的工程实践中, 全量微调不仅意味着"更新所有权重", 更涉及到数值稳定性(精度控制)以及多模态架构下的模块隔离.

    高级研究员与资深工程师的架构点评:

    数值稳定性 (The FP32 Trap):
    作为高级工程师, 我必须指出: 很多人直接在全量微调时使用 model.half() 或 model.to(torch.bfloat16). 虽然省显存, 但梯度更新在大规模参数下很容易出现 "下溢(Underflow)". cast_trainable_params_to_fp32 的开关体现了 LLaMA-Factory 对生产环境稳定性的极致考量 - 用少量的显存代价换取训练的绝对鲁棒性.

    多模态一致性 (Multimodal Consistency):
    get_forbidden_modules 的存在解决了 LLM 向多模态演进时的代码一致性. 如果你微调一个视觉大模型(VLM), 通常视觉分支是"不可变的", 只有语言分支是"可塑的". 这个函数让全量微调逻辑在不同架构之间保持了良好的解耦性.

    内存效率优化 (DDP/DeepSpeed 友好):
    这种参数级的遍历和转换, 是在模型被包装进分布式策略(如 DeepSpeed ZeRO 系列)之前完成的. 这非常关键, 因为一旦模型被切分到不同显卡上, 再进行这种全局参数转换会变得极其复杂且低效.
    """

    # 1. 状态快速检查
    # [为什么要这么写]: 如果是推理模式或评估模式, 无需进行参数状态转换.
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")

    # 2. 模块黑名单过滤 (Architectural Guardrails)
    # [为什么要这么写]: 调用 get_forbidden_modules 获取那些即便在"全量微调"下也不该动的部分.
    # [解决的问题]: 在多模态模型(如 LLaVA, Qwen-VL)或混合架构中, 我们通常只想微调 LLM 核心,
    # 而希望冻结视觉编码器(Vision Encoder)或某些特定的固定嵌入层.
    # 这一步确保了全量微调的"边界感", 防止破坏预训练好的感知层特征.
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)

    # 3. 遍历并配置参数
    for name, param in model.named_parameters():
        # 检查当前参数名是否包含黑名单中的模块名
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            # 4. 数值稳定性转换 (Precision Management)
            # [为什么要这么写]: 如果开启了 cast_trainable_params_to_fp32.
            # [解决的问题]: 这是解决 LLM 训练收敛问题的"工业级黑魔法".
            # 现代模型常以 BF16/FP16 加载以节省显存, 但在全量微调时, 权重更新量级非常小.
            # 如果依然在半精度上更新梯度, 会导致严重的精度舍入误差(Rounding Error), 造成 Loss 无法收敛.
            # 将可训练参数强制转为 FP32, 可以利用单精度的高动态范围来保证梯度更新的精确度.
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
            # 注意: 全量微调下默认 requires_grad 已经是 True, 所以此处主要处理精度
        else:
            # 5. 强制冻结黑名单模块 (Selective Freezing)
            # [为什么要这么写]: 如果参数属于禁区模块, 显式关闭其梯度.
            # [解决的问题]: 节省计算资源. 即便是在进行所谓的"全量"微调,
            # 这种"手术刀式"的精度控制也能有效防止不该更新的层发生漂移.
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    """
    在 LLM 微调中, Freeze Tuning(冻结微调) 是一种处于"全量微调"和"LoRA 微调"之间的折中方案. 它的核心逻辑是: 冻结模型的大部分参数, 只更新特定的层(如最后几层)或特定的组件(如 Embedding 层). 这种方法在节省显存的同时, 能比 LoRA 更好地保留预训练模型的特征提取能力.

    高级研究员视角的架构点评:

    动态性与通用性的平衡:
    这段代码最出彩的地方在于不依赖于任何特定的模型架构文件. 它通过正则匹配和参数名扫描, 在运行时"自动理解"了模型的拓扑结构. 这对于像 LLaMA-Factory 这样需要支持数百个不同模型的项目来说, 是降低维护成本的关键.

    Llama Pro 的集成:
    代码中对 use_llama_pro 的支持紧跟学术前沿. Llama Pro 论文提出的"分层训练"被转化为一个简单的 stride 逻辑, 这种工程实现非常优雅, 将复杂的学术构想变成了简单的参数控制.

    数值稳定性的保障:
    cast_trainable_params_to_fp32 处理了 LLM 在低比特/混合精度环境下训练的痛点. 即便原始模型是以 BF16 加载的, 但如果只有 1% 的参数在训练, 将这 1% 转为 FP32 几乎不增加显存开销, 却能显著提升微调的收敛速度和最终精度.
    """

    # 1. 状态快速检查
    # [为什么要这么写]: 如果是推理模式, 无需处理梯度和精度转换, 直接返回.
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze")

    # 2. 兼容性配置获取 (Config Abstraction)
    # [为什么要这么写]: 处理复合模型(如多模态 VLM).
    # [解决的问题]: 在多模态模型中, `model.config` 往往是一个包装类, 真正的文本分支参数在 `text_config` 下.
    # 这样写确保了代码能统一处理纯文本模型和多模态模型.
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    # 3. 统一层数命名规范 (Layer Normalization)
    # [为什么要这么写]: 不同厂商的模型对"总层数"的定义字段不同.
    # [解决的问题]: Llama 用 `num_hidden_layers`, 一些旧模型用 `n_layer`.
    # 通过这种链式获取方式, 实现了对 Hugging Face 生态中绝大多数模型的自动化适配.
    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    # 4. 计算待训练层的索引 (Trainable Layer Indexing)
    # [为什么要这么写]: 支持三种不同的冻结策略:
    #   A. Llama Pro 模式: 按照步长(stride)选取特定的层(通常是新插入的层).
    #   B. 训练后 N 层: 如果参数 > 0, 则训练模型最靠近输出端的层(LLM 高层通常捕捉语义, 适合微调).
    #   C. 训练前 N 层: 如果参数 < 0, 则训练最靠近输入端的层.
    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    # 5. 动态嗅探模块名称 (Dynamic Module Probing)
    # [为什么要这么写]: 不要硬编码 "self_attn" 或 "mlp".
    # [解决的问题]: 不同模型内部组件命名极其混乱. 这里通过遍历第 0 层或第 1 层(针对 MoD 模型)的参数名,
    # 动态分析出当前模型包含哪些子模块. 这体现了"数据驱动"而非"硬编码"的工程思想.
    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1 MoD (Mixture of Depths) 适配
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    # 6. 构建可训练参数的匹配关键字 (Trainable Filter Construction)
    # [为什么要这么写]: 将层号索引与具体的模块名结合, 生成类似 ".31.self_attn" 的匹配字符串.
    # [解决的问题]: 支持用户指定"只训练最后 4 层的 MLP 模块"这种精细化操作.
    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    # 7. 处理额外的可训练模块(如 Embedding 或 LM_Head)
    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    # 8. 多模态对齐层适配 (Multimodal Projector Adaptation)
    # [为什么要这么写]: 如果是视觉语言模型(VLM), 允许用户单独训练 Projector 层.
    # [解决的问题]: 在多模态微调中, Projector 负责将图像特征对齐到文本空间, 是微调最频繁的部分.
    model_type = getattr(model.config, "model_type", None)
    if not finetuning_args.freeze_multi_modal_projector and model_type in COMPOSITE_MODELS:
        trainable_layers.append(COMPOSITE_MODELS[model_type].projector_key)

    # 9. 最终参数状态应用 (The Final Loop)
    # [为什么要这么写]: 遍历模型所有参数, 执行"外科手术式"的冻结和精度设置.
    # [解决的问题]:
    #   - 匹配检查: 利用 `any` 进行字符串搜索, 命中 `trainable_layers` 且不在 `forbidden_modules` 内.
    #   - 数值稳定性: 如果参数可训练且开启了 `cast_trainable_params_to_fp32`, 则将其转为 FP32.
    #     原因是在混合精度训练中, 梯度更新在单精度下更稳定, 防止出现梯度消失.
    #   - 强力冻结: 对于没命中的参数, 显式调用 `requires_grad_(False)`.
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    """
    _setup_lora_tuning 是整个框架中最复杂也最核心的函数之一. 它不仅要处理标准的 LoRA 逻辑, 还要兼容 DoRA、OFT、PiSSA 等变体, 同时需要适配 Unsloth (极致加速)、KTransformers (异构算力优化) 以及 DeepSpeed ZeRO-3 (大规模分布式) 等多种后端.

    资深架构师的深度总结:

    "先加载, 后训练"的解耦: 代码清晰地分为了"已有 Adapter 处理"和"新训练配置初始化"两个大块, 确保了在复杂的微调工作流(如多次增量微调)中的逻辑一致性.

    防御性编程: 在函数开头的多个 assert 体现了对硬件和算法底层兼容性的深刻理解. 它能防止用户在不兼容的配置下运行数小时后才发现模型不收敛或发生爆炸.

    对数学正确性的极致追求: 最后的 cast_trainable_params_to_fp32 是工业级微调框架的标志. 很多简单的开源脚本漏掉了这一步, 导致其 LoRA 训练在低精度硬件上效果远逊于全量微调.
    """

    # 1. 训练方法日志记录
    # [为什么要这么写]: 在分布式训练中, 通过 rank0 日志明确告知用户当前使用的技术方案.
    if is_trainable:
        if finetuning_args.finetuning_type == "oft":
            logger.info_rank0("Fine-tuning method: OFT")
        else:
            # DoRA 是 LoRA 的增强版(权重分解), 这里做显式区分.
            logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None

    # 2. 处理已有的适配器 (Resuming or Merging)
    # [解决的问题]: 支持多适配器链式加载. 例如: 加载一个基础 LoRA, 再在其基础上继续微调.
    if model_args.adapter_name_or_path is not None:
        is_mergeable = True

        # [工程限制检查]: 以下场景由于底层架构或算子限制, 不支持多适配器并行计算, 必须强行报错或限制为单个.
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            # 量化模型的权重已截断, 多适配器合并会导致精度剧烈坍缩.
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            # ZeRO-3 将参数分片到了不同 GPU, 维护多个 Adapter 的参数收集极其复杂且低效.
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_kt:
            # KTransformers 使用了自定义的量化内核, 暂未实现多 Adapter 路由.
            assert len(model_args.adapter_name_or_path) == 1, "KTransformers model only accepts a single adapter"
            is_mergeable = False

        if model_args.use_unsloth:
            # Unsloth 追求极致速度, 其手写的 Triton 内核通常针对单 Adapter 优化.
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        # 3. 确定哪些 Adapter 需要被"融合", 哪个需要被"续训"
        # [逻辑说明]: 如果是续训模式, 除了最后一个 Adapter 外, 其余的都合并进 Base Model 以节省显存.
        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        # 构建标准的加载参数(支持从 Hub 加载)
        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        if model_args.use_kt:
            if model_args.infer_backend != EngineName.KT:
                raise ValueError(
                    "We should use ktransformers as backend to infer the adapter fine-tuned by ktransformers."
                )

        # 4. 执行 Adapter 融合 (Merge and Unload)
        # [为什么要这么写]: 通过 merge_and_unload() 将 Adapter 的数学增量叠加到主权重上.
        # [解决的问题]: 解决了加载多个 Adapter 导致的推理延迟问题(降低模型层级深度).
        for adapter in adapter_to_merge:
            model: LoraModel = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        # 5. 加载用于续训的最后一个 Adapter
        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_kt:
                model = load_kt_peft_model(model_args, model)
            elif model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, finetuning_args, is_trainable=is_trainable)
            else:
                # 使用原生 PEFT 逻辑加载
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    # 6. 初始化全新的 LoRA 权重 (Fresh Training)
    # [场景]: 用户开启训练且没有要 Resume 的 Adapter.
    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        # 自动探测目标模块
        # [为什么要这么写]: 如果指定 "all", 算法会自动找出模型中所有的 Linear 层.
        # [解决的问题]: 用户不需要手动查阅代码去写 "q_proj, v_proj", 提高了框架的通用性.
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        # 针对 KTransformers 后端的 MLP 映射修正
        if model_args.use_kt:
            new_list = []
            for m in target_modules:
                if m in ("down_proj", "up_proj", "gate_proj"):
                    new_list.extend([f"mlp.{m}", f"shared_experts.{m}"])
                elif m not in ("generate_linear", "orig_module", "prefill_linear"):
                    new_list.append(m)

            target_modules[:] = new_list

        # Llama Pro 支持: 只针对新扩展的层进行微调
        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        # 核心逻辑修正(针对特定架构的 Patch)
        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        # 7. 词表扩充时的 Embedding 微调保护 (Vocab Resizing)
        # [解决的问题]: 如果增加了特殊 Token, Embedding 层和 LM_Head 必须变为可训练,
        # 否则模型永远学不会这些新 Token 的表示.
        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        # 8. 构造具体的 PEFT 配置参数
        if finetuning_args.finetuning_type == "lora":
            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "use_rslora": finetuning_args.use_rslora,  # 秩稳定 LoRA
                "use_dora": finetuning_args.use_dora,  # 权重分解
                "modules_to_save": finetuning_args.additional_target,  # 全量训练的额外模块
            }
        elif finetuning_args.finetuning_type == "oft":
            # OFT (Orthogonal Fine-tuning) 采用正交变换, 对微调稳定性有帮助.
            peft_kwargs = {
                "r": finetuning_args.oft_rank,
                "oft_block_size": finetuning_args.oft_block_size,
                "target_modules": target_modules,
                "module_dropout": finetuning_args.module_dropout,
                "modules_to_save": finetuning_args.additional_target,
            }

        # 9. 后端分发与实例化 (Backend Dispatching)
        if model_args.use_kt:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("KTransformers is currently not supported for OFT.")
            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            else:
                raise ValueError("KTransformers is currently only supported for LoRA.")

            model = get_kt_peft_model(model, peft_config)
            print(f"KT_model:{model}")
        elif model_args.use_unsloth:
            if finetuning_args.finetuning_type == "oft":
                raise ValueError("Unsloth is currently not supported for OFT.")

            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

            if finetuning_args.finetuning_type == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            elif finetuning_args.finetuning_type == "oft":
                peft_config = OFTConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs,
                )
            model = get_peft_model(model, peft_config)

    # 10. 混合精度训练的数值稳定性保障
    # [为什么要这么写]: 将所有 requires_grad=True 的参数(即 LoRA 权重)强制转为 FP32.
    # [解决的问题]: LoRA 的秩(Rank)通常很小, 在 BF16/FP16 下进行极小幅度的梯度更新会导致严重的
    # 舍入误差(Rounding Error), 使模型难以收敛. 强制 FP32 可以利用单精度的高动态范围保证收敛质量.
    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.

    在 LLM 微调工程中, init_adapter 是连接模型加载与正式训练的关键桥梁. 它负责根据用户的配置(全量、冻结、LoRA等)重新编排模型的计算图和参数精度.

    初始化适配器(Adapter).
    支持全参数(Full-parameter)、参数冻结(Freeze)以及 LoRA/OFT 训练.

    [核心设计原则]:
    在微调阶段, 必须严格控制哪些参数可导, 并确保可训练参数的数值精度(Dtype)足以支撑梯度更新.

    高级研究员视角的深度总结:

    数值稳定性 (Numerical Stability):
    这段代码最体现水平的地方在于对 cast_trainable_params_to_fp32 开关的逻辑判断. 在大规模参数训练中, BF16/FP16 存储权重 + FP32 计算梯度/更新权重 是业界公认的最佳实践. LLaMA-Factory 在这里自动处理了复杂的场景判定, 极大降低了用户把模型训练崩(Loss NaN)的概率.

    解耦与抽象 (Decoupling):
    init_adapter 作为一个"工厂方法", 屏蔽了 transformers、peft 以及 deepspeed 之间繁琐的交互细节. 无论用户加载的是 Llama-3 还是 Qwen, 是 4-bit 量化还是全量权重, 进入这个函数后都会被归一化处理.

    对分布式架构的尊重:
    特别处理 is_deepspeed_zero3_enabled 体现了高级开发者的严谨性. ZeRO-3 会将权重分片(Partitioning)到不同卡上, 如果你在 ZeRO 初始化后手动强行修改 param.data, 可能会破坏 DeepSpeed 的静态计算图, 导致通讯死锁.
    """

    # 1. 量化模型兼容性校验 (Quantization Compatibility Check)
    # [为什么要这么写]: 量化模型(如 4-bit/8-bit)的底层权重是离散且冻结的, 无法直接计算梯度.
    # [解决的问题]: 防止用户尝试在量化模型上进行全量或冻结微调. 量化模型只能通过在其上方挂载浮点类型的
    # 适配器(如 LoRA)来进行"增量更新". 此外, PiSSA 初始化涉及奇异值分解(SVD), 量化过程破坏了
    # 原始权重的数学分布, 因此也互斥.
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantized models can only be used for the LoRA or OFT tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # 2. 自动判定是否需要将可训练参数提升至 FP32 (Automatic Upcasting Logic)
    # [核心逻辑]:
    # LLM 权重通常以 BF16 或 FP16 存储, 但对于增量参数(如 LoRA 权重), 其梯度往往非常微小.
    # 如果继续使用半精度训练, 极易出现"下溢(Underflow)"现象, 导致权重无法更新或 Loss 不收敛.

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        # 推理模式下不需要调整精度
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        # [情况 A]: 如果用户显式要求纯 BF16 训练, 或者使用 BAdam 优化器.
        # BAdam 内部有专门的内存管理机制, 因此我们尊重原始设置, 保持半精度.
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and is_deepspeed_zero3_enabled():
        # [情况 B]: DeepSpeed ZeRO-3 开启且非量化场景.
        # ZeRO-3 引擎本身会自动管理参数的分片和精度收集, 它通常要求在特定的生命周期内保持精度,
        # 过早的人为干预会导致 DeepSpeed 的内部状态冲突.
        logger.info_rank0("DeepSpeed ZeRO3 detected, remaining trainable params in float32.")
    else:
        # [情况 C]: 默认工业级标准 - 强制提升可训练参数至 FP32.
        # 这是为了解决梯度消失(Vanishing Gradients)问题. 将 requires_grad=True 的参数提升到 FP32,
        # 能显著提高微调的数值稳定性.
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    # 3. 任务分发 (Method Dispatching)
    # [设计模式]: 策略模式. 根据不同的微调类型, 路由到具体的配置函数.

    if finetuning_args.finetuning_type == "full":
        # 全量微调: 通常涉及解冻所有(或绝大部分)参数, 并处理 FP32 转换.
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        # 冻结微调: 根据层索引(Layer Index)选择性解冻最后几层.
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type in ["lora", "oft"]:
        # 适配器微调: 通过 PEFT 库向模型注入低秩矩阵或其他辅助结构.
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    return model

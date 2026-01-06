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

import os
import shutil
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_mcore_adapter_available, is_ray_available
from ..hparams import get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import get_ray_trainer, get_swanlab_callback


# 1. 动态依赖加载策略
# [为什么要这么写]: LLM 依赖环境极其复杂. Ray 是可选的大规模分布式框架.
# [解决的问题]: 避免在没有安装 Ray 的环境下因 import 失败导致整个框架无法启动.
if is_ray_available():
    import ray
    from ray.train.huggingface.transformers import RayTrainReportCallback


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    """
    实际执行训练的内部函数, 被 run_exp 或 Ray Trainer 调用.
    """
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    # 2. 强类型参数解析 (Type-Safe Argument Parsing)
    # [为什么要这么写]: 将混合的 dict 拆分为模型、数据、训练、微调、生成五个维度.
    # [解决的问题]: 这是典型的"分治"思想, 确保后续逻辑(如 SFT 或 DPO)只关注自己需要的参数, 提高代码鲁棒性.
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # 3. 回调函数编排 (Callback Orchestration)
    # [为什么要这么写]: 显式控制 Callback 顺序, 并将 ReporterCallback 放在最后.
    # [解决的问题]: 确保所有的监控(如 SwanLab)、模型转换(Pissa)逻辑先于最终的状态报告.
    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    # ReporterCallback 放在末尾是为了在所有计算结束后进行最终的状态同步和清理
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    # 4. 多架构分发 (Multi-Backend Dispatching)
    # [为什么要这么写]: 判断是否使用 MCA (Megatron-Core Adapter).
    # [解决的问题]: 解决大规模(通常是千亿参数级别)训练时性能瓶颈问题.
    # MCA 提供了针对 NVIDIA Megatron 的加速支持, 这里实现了"传统 HF 模式"与"高性能分布式模式"的逻辑解耦.
    if finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        # 动态导入 MCA 运行模块, 节省非 MCA 模式下的内存开销
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    # 5. 任务阶段路由 (Task Stage Routing)
    # [为什么要这么写]: 通过简单的 elif 路由到不同的算法实现(PT, SFT, RM, PPO, DPO, KTO).
    # [解决的问题]: 实现了"全流程微调"框架的核心价值: 统一入口, 支持从预训练到偏好对齐的所有阶段.
    elif finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    # 6. 分布式资源优雅清理 (Resource Cleanup)
    # [为什么要这么写]: 显式调用 destroy_process_group.
    # [解决的问题]: 防止在同一脚本进行多轮训练或 Ray 调度时, 残留的进程组导致显存泄露或死锁.
    if is_ray_available() and ray.is_initialized():
        return  # if ray is intialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    """
    外部入口点, 处理高层调度(如 Ray 运行 vs 本地运行).
    """
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []

    # 7. 云原生/集群扩展性设计 (Ray Integration)
    # [为什么要这么写]: 如果配置了 use_ray, 则通过 get_ray_trainer 启动.
    # [解决的问题]: 实现了"本地调试"与"生产集群训练"的零成本切换.
    # Ray 负责处理多节点的 Pod 分配和故障恢复, 而核心训练逻辑依然复用 _training_function.
    if ray_args.use_ray:
        callbacks.append(RayTrainReportCallback())
        trainer = get_ray_trainer(
            training_function=_training_function,
            train_loop_config={"args": args, "callbacks": callbacks},
            ray_args=ray_args,
        )
        trainer.fit()
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    """
    模型导出与融合逻辑.
    """
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    # 8. 导出前的前置校验 (Validation)
    # [为什么要这么写]: 禁止对已量化的模型进行 Adapter 融合.
    # [解决的问题]: 量化权重(如 4-bit)是离散的, 无法与浮点数 Adapter 直接进行数学求和.
    # 这类校验能防止用户在耗时加载后才发现配置错误.
    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    # 9. 依赖顺序加载策略 (Order-Dependent Loading)
    # [为什么要这么写]: 先加载 Tokenizer, 再加载 Model.
    # [解决的问题]: LLM 的 Embedding 层大小与 Tokenizer 词表强绑定.
    # 某些 Template 可能会扩展特殊 Token(如 <|im_start|>),
    # load_model 内部会调用 resize_token_embeddings 以确保模型能够识别这些新词.
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    # 10. 高精度自动推理 (Precision Inference & Conversion)
    # [为什么要这么写]: 处理 infer_dtype="auto" 的逻辑.
    # [解决的问题]: 如果硬件支持 BF16(由 infer_optim_dtype 判断), 则优先使用 BF16 而非 FP16.
    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    # 11. 权重分片与安全序列化 (Sharding & Safety)
    # [为什么要这么写]: 使用 save_pretrained 并控制 max_shard_size.
    # [解决的问题]: 默认不启用 legacy format 以支持 safetensors, 防止模型加载时的代码注入风险,
    # 同时分片存储方便在内存有限的设备上分块加载.
    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size=f"{model_args.export_size}GB",
        safe_serialization=(not model_args.export_legacy_format),
    )
    # ... (push_to_hub 逻辑)
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size=f"{model_args.export_size}GB",
            safe_serialization=(not model_args.export_legacy_format),
        )

    # 12. 奖励模型特殊处理 (Reward Model ValueHead Patching)
    # [为什么要这么写]: 针对 RM 阶段, 需要手动拷贝 vhead 权重.
    # [解决的问题]: Huggingface Transformers 的原生 `save_pretrained` 有时不会保存自定义的 ValueHead.
    # 通过手动检测并拷贝 `value_head.bin/safetensors`, 确保导出的奖励模型可以直接用于 PPO 阶段.
    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        # ... (权重拷贝逻辑)
        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    # 13. 推理适配性收尾 (Tokenizer Finalization)
    # [为什么要这么写]: 强制将 padding_side 设为 "left".
    # [解决的问题]: 虽然训练时常使用 right padding 以提高并行效率,
    # 但对于 LLM 生成推理, left padding 是标准做法, 确保生成的下一个 Token 位置是正确的.
    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    # 14. 边缘侧部署支持 (Edge Deployment Support)
    # [为什么要这么写]: 生成 Ollama 所需的 Modelfile.
    # [解决的问题]: 打通微调到端侧部署的最后一步.
    # 用户导出的模型可以直接被 Ollama 加载, 极大地缩短了从训练到产品的链路.
    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")

"""
高级研究员视角的架构点评:

高度的可组合性: 通过 _training_function 将所有算法逻辑(SFT/DPO等)抽象为统一的函数签名, 这使得 LLaMA-Factory 可以作为库被其他项目调用, 或者直接在分布式集群运行.

严谨的设备管理: 显式处理分布式进程组的销毁(destroy_process_group), 这是资深工程师处理多卡环境时的必备素质, 能有效解决显存莫名被占用的难题.

全生态链考虑: export_model 函数中对 safetensors、sharding、Ollama、Hub 的支持, 说明该框架不仅仅关注"微调", 更关注微调后的模型分发与落地.

希望这些注释对你深入研究 LLaMA-Factory 的核心架构有所帮助!
"""

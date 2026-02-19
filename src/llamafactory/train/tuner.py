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


"""
在处理像 transformers 这样庞大的依赖库以及项目中复杂的模块互相引用时, 这是一种 Python 高级工程实践.
# 1. 静态类型检查开关 (Static Type Analysis Guard)
# [为什么要这么写]:
# TYPE_CHECKING 是 typing 模块提供的一个特殊常量. 在程序实际运行时, 它的值永远是 False;
# 只有在静态类型检查工具(如 Mypy、Pyright 或 IDE 的代码提示引擎)扫描代码时, 它的值才为 True.
"""
if TYPE_CHECKING:
    # 2. 延迟导入类型定义 (Deferred Type Import)
    # [解决的问题]:
    # 问题 A: 循环依赖 (Circular Dependency)
    # LLaMA-Factory 内部逻辑耦合度较高. 例如, Tuner 模块可能需要 Callback 的类型定义,
    # 而 Callback 模块内部又反过来引用了 Tuner 里的配置类.
    # 如果在文件顶部直接 import, 会导致 Python 解释器抛出"无法从未初始化模块导入"的错误.
    # 使用这种写法, 在运行时不会触发真正的 import, 从而完美规避循环引用的死锁.

    # 问题 B: 运行时性能与内存开销 (Runtime Performance)
    # transformers 是一个非常沉重的库. 虽然在 LLM 项目中它最终会被加载, 但在模块初始化阶段,
    # 我们不希望为了仅仅做一个类型声明(Type Hinting)就去触发昂贵的模块加载逻辑.
    # 这能加快模块的初始加载速度, 减少不必要的命名空间污染.

    # 问题 C: 保持代码整洁与 IDE 友好
    # 这样写可以让开发人员在编写代码时, 享受到 IDE 提供的类成员自动补全和类型检查,
    # 而在执行时却完全不产生任何额外开销.
    from transformers import TrainerCallback
    # 注意: 原文中是 TrainerCallbacks, 通常在 transformers 中基类为 TrainerCallback
"""
高级研究员与资深开发者的视角:

关于架构稳定性:
在 LLaMA-Factory 中, 我们支持多种微调算法(SFT, DPO, PPO 等). 这些算法通常需要自定义 TrainerCallback 来监控 Loss 或进行特定的模型保存逻辑. 由于这些 Callback 往往需要引用全局的 FinetuningArguments, 通过 TYPE_CHECKING 引入类型, 可以确保我们在编写复杂的回调逻辑时, 类型系统能捕捉到可能的参数类型错误, 而不会在运行时增加模块间的耦合.

工程化细节:
作为高级工程师, 我们要区分 "类型空间 (Type Space)" 和 "值空间 (Value Space)".
if TYPE_CHECKING: 块内的东西只存在于类型空间, 用于辅助开发.
运行时的业务逻辑(值空间)不需要这些 import.
如果在代码后续的非类型注释部分(例如在函数体内部)使用了 TrainerCallback, 我们需要确保它仅作为类型标注使用(如 def on_step(cb: "TrainerCallback"):), 或者在运行时动态导入.

这种写法是区分 "初级脚本编写者" 与 "高级系统架构师" 的标志性细节之一.
"""


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    """
    实际执行训练的内部函数, 被 run_exp 或 Ray Trainer 调用.
    """
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    """
    print(args)
    {
        'model_name_or_path': '/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507',
        'trust_remote_code': True,
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'lora',
        'lora_rank': 8,
        'lora_target': 'all',
        'dataset': 'identity,alpaca_en_demo',
        'template': 'qwen3_nothink',
        'cutoff_len': 2048,
        'max_samples': 1000,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,
        'output_dir': 'saves/qwen3-4b/lora/sft',
        'logging_steps': 10,
        'save_steps': 500,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'save_only_model': False,
        'report_to': 'none',
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 0.0001,
        'num_train_epochs': 3.0,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'resume_from_checkpoint': None
    }
    print(callbacks)
    []
    """

    # 2. 强类型参数解析 (Type-Safe Argument Parsing)
    # [为什么要这么写]: 将混合的 dict 拆分为模型、数据、训练、微调、生成五个维度.
    # [解决的问题]: 这是典型的"分治"思想, 确保后续逻辑(如 SFT 或 DPO)只关注自己需要的参数, 提高代码鲁棒性.
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    """
    print(model_args)
    ModelArguments(
        model_name_or_path="/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507",
        adapter_name_or_path=None,
        adapter_folder=None,
        cache_dir=None,
        use_fast_tokenizer=True,
        resize_vocab=False,
        split_special_tokens=False,
        add_tokens=None,
        add_special_tokens=None,
        new_special_tokens_config=None,
        init_special_tokens="noise_init",
        model_revision="main",
        low_cpu_mem_usage=True,
        rope_scaling=None,
        flash_attn="<AttentionFunction.AUTO: 'auto'>",
        shift_attn=False,
        mixture_of_depths=None,
        use_unsloth=False,
        use_unsloth_gc=False,
        enable_liger_kernel=False,
        moe_aux_loss_coef=None,
        disable_gradient_checkpointing=False,
        use_reentrant_gc=True,
        upcast_layernorm=False,
        upcast_lmhead_output=False,
        train_from_scratch=False,
        infer_backend="<EngineName.HF: 'huggingface'>",
        offload_folder="offload",
        use_kv_cache=True,
        use_v1_kernels=False,
        infer_dtype="auto",
        hf_hub_token=None,
        ms_hub_token=None,
        om_hub_token=None,
        print_param_status=False,
        trust_remote_code=True,
        quantization_method="<QuantizationMethod.BNB: 'bnb'>",
        quantization_bit=None,
        quantization_type="nf4",
        double_quantization=True,
        quantization_device_map=None,
        fp8=False,
        fp8_backend="auto",
        fp8_enable_fsdp_float8_all_gather=False,
        image_max_pixels=589824,
        image_min_pixels=1024,
        image_do_pan_and_scan=False,
        crop_to_patches=False,
        video_max_pixels=65536,
        video_min_pixels=256,
        video_fps=2.0,
        video_maxlen=128,
        use_audio_in_video=False,
        audio_sampling_rate=16000,
        export_dir=None,
        export_size=5,
        export_device="cpu",
        export_quantization_bit=None,
        export_quantization_dataset=None,
        export_quantization_nsamples=128,
        export_quantization_maxlen=1024,
        export_legacy_format=False,
        export_hub_model_id=None,
        use_kt=False,
        kt_optimize_rule=None,
        cpu_infer=32,
        chunk_size=8192,
        mode="normal",
        kt_maxlen=4096,
        kt_use_cuda_graph=True,
        kt_mode="normal",
        kt_force_think=False,
        vllm_maxlen=4096,
        vllm_gpu_util=0.7,
        vllm_enforce_eager=False,
        vllm_max_lora_rank=32,
        vllm_config=None,
        sglang_maxlen=4096,
        sglang_mem_fraction=0.7,
        sglang_tp_size=-1,
        sglang_config=None,
        sglang_lora_backend="triton",
        compute_dtype=torch.bfloat16,
        device_map={"": device(type="cuda", index=0)},
        model_max_length=2048,
        block_diag_attn=False,
    )

    print(data_args)
    DataArguments(
        template="qwen3_nothink",
        dataset=["identity", "alpaca_en_demo"],
        eval_dataset=None,
        dataset_dir="data",
        media_dir="data",
        cutoff_len=2048,
        train_on_prompt=False,
        mask_history=False,
        streaming=False,
        buffer_size=16384,
        mix_strategy="concat",
        interleave_probs=None,
        overwrite_cache=False,
        preprocessing_batch_size=1000,
        preprocessing_num_workers=16,
        max_samples=1000,
        eval_num_beams=None,
        ignore_pad_token_for_loss=True,
        val_size=0.0,
        eval_on_each_dataset=False,
        packing=False,
        neat_packing=False,
        tool_format=None,
        default_system=None,
        enable_thinking=True,
        tokenized_path=None,
        data_shared_file_system=False,
    )

    print(training_args)
    TrainingArguments(
        _n_gpu=1,
        accelerator_config={
            "split_batches": False,
            "dispatch_batches": None,
            "even_batches": True,
            "use_seedable_sampler": True,
            "non_blocking": False,
            "gradient_accumulation_kwargs": None,
            "use_configured_state": False,
        },
        adafactor=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        auto_find_batch_size=False,
        average_tokens_across_devices=True,
        batch_eval_metrics=False,
        bf16=True,
        bf16_full_eval=False,
        data_seed=None,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=None,
        ddp_backend=None,
        ddp_broadcast_buffers=None,
        ddp_bucket_cap_mb=None,
        ddp_find_unused_parameters=None,
        ddp_timeout=180000000,
        debug=[],
        deepspeed=None,
        disable_tqdm=False,
        do_eval=False,
        do_predict=False,
        do_train=True,
        eval_accumulation_steps=None,
        eval_delay=0,
        eval_do_concat_batches=True,
        eval_on_start=False,
        eval_steps=None,
        eval_strategy=IntervalStrategy.NO,
        eval_use_gather_object=False,
        fp16=False,
        fp16_backend=auto,
        fp16_full_eval=False,
        fp16_opt_level=O1,
        fsdp=[],
        fsdp_config={
            "min_num_params": 0,
            "xla": False,
            "xla_fsdp_v2": False,
            "xla_fsdp_grad_ckpt": False,
        },
        fsdp_min_num_params=0,
        fsdp_transformer_layer_cls_to_wrap=None,
        full_determinism=False,
        generation_config=None,
        generation_max_length=2048,
        generation_num_beams=None,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        greater_is_better=None,
        group_by_length=False,
        half_precision_backend=auto,
        hub_always_push=False,
        hub_model_id=None,
        hub_private_repo=None,
        hub_revision=None,
        hub_strategy=HubStrategy.EVERY_SAVE,
        hub_token="<HUB_TOKEN>",
        ignore_data_skip=False,
        include_for_metrics=[],
        include_inputs_for_metrics=False,
        include_num_input_tokens_seen=no,
        include_tokens_per_second=False,
        jit_mode_eval=False,
        label_names=["labels"],
        label_smoothing_factor=0.0,
        learning_rate=0.0001,
        length_column_name=length,
        liger_kernel_config=None,
        load_best_model_at_end=False,
        local_rank=0,
        log_level=passive,
        log_level_replica=warning,
        log_on_each_node=True,
        logging_dir="saves/qwen3-4b/lora/sft/runs/Feb19_20-53-48_k8s-a40-node02",
        logging_first_step=False,
        logging_nan_inf_filter=True,
        logging_steps=10,
        logging_strategy=IntervalStrategy.STEPS,
        lr_scheduler_kwargs={},
        lr_scheduler_type=SchedulerType.COSINE,
        max_grad_norm=1.0,
        max_steps=-1,
        metric_for_best_model=None,
        mp_parameters="",
        neftune_noise_alpha=None,
        no_cuda=False,
        num_train_epochs=3.0,
        optim=OptimizerNames.ADAMW_TORCH_FUSED,
        optim_args=None,
        optim_target_modules=None,
        output_dir="saves/qwen3-4b/lora/sft",
        overwrite_output_dir=True,
        parallelism_config=None,
        past_index=-1,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=1,
        placement_strategy=PACK,
        predict_with_generate=False,
        prediction_loss_only=False,
        project=huggingface,
        push_to_hub=False,
        push_to_hub_model_id=None,
        push_to_hub_organization=None,
        push_to_hub_token="<PUSH_TO_HUB_TOKEN>",
        ray_init_kwargs=None,
        ray_num_workers=1,
        ray_run_name=None,
        ray_scope=last,
        ray_storage_filesystem=None,
        ray_storage_path="./saves",
        remove_unused_columns=False,
        report_to=[],
        resources_per_worker={"GPU": 1},
        restore_callback_states_from_checkpoint=False,
        resume_from_checkpoint=None,
        run_name=None,
        save_on_each_node=False,
        save_only_model=False,
        save_safetensors=True,
        save_steps=500,
        save_strategy=SaveStrategy.STEPS,
        save_total_limit=None,
        seed=42,
        skip_memory_metrics=True,
        sortish_sampler=False,
        tf32=None,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        torch_empty_cache_steps=None,
        torchdynamo=None,
        tpu_metrics_debug=False,
        tpu_num_cores=None,
        trackio_space_id=trackio,
        use_cpu=False,
        use_legacy_prediction_loop=False,
        use_liger_kernel=False,
        use_mps_device=False,
        warmup_ratio=0.1,
        warmup_steps=0,
        weight_decay=0.0,
    )

    print(finetuning_args)
    FinetuningArguments(
        freeze_trainable_layers=2,
        freeze_trainable_modules=["all"],
        freeze_extra_modules=None,
        additional_target=None,
        module_dropout=0.0,
        oft_rank=0,
        oft_block_size=32,
        oft_target=["all"],
        create_new_adapter=False,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_rank=8,
        lora_target=["all"],
        loraplus_lr_ratio=None,
        loraplus_lr_embedding=1e-06,
        use_rslora=False,
        use_dora=False,
        pissa_init=False,
        pissa_iter=16,
        pissa_convert=False,
        pref_beta=0.1,
        pref_ftx=0.0,
        pref_bco_weight=0.0,
        pref_loss="sigmoid",
        dpo_label_smoothing=0.0,
        kto_chosen_weight=1.0,
        kto_rejected_weight=1.0,
        simpo_gamma=0.5,
        ppo_buffer_size=1,
        ppo_epochs=4,
        ppo_score_norm=False,
        ppo_target=6.0,
        ppo_whiten_rewards=False,
        ref_model=None,
        ref_model_adapters=None,
        ref_model_quantization_bit=None,
        reward_model=None,
        reward_model_adapters=None,
        reward_model_quantization_bit=None,
        reward_model_type="lora",
        ld_alpha=None,
        use_galore=False,
        galore_target=["all"],
        galore_rank=16,
        galore_update_interval=200,
        galore_scale=2.0,
        galore_proj_type="std",
        galore_layerwise=False,
        use_apollo=False,
        apollo_target=["all"],
        apollo_rank=16,
        apollo_update_interval=200,
        apollo_scale=32.0,
        apollo_proj="random",
        apollo_proj_type="std",
        apollo_scale_type="channel",
        apollo_layerwise=False,
        apollo_scale_front=False,
        use_badam=False,
        badam_mode="layer",
        badam_start_block=None,
        badam_switch_mode="ascending",
        badam_switch_interval=50,
        badam_update_ratio=0.05,
        badam_mask_mode="adjacent",
        badam_verbose=0,
        use_swanlab=False,
        swanlab_project="llamafactory",
        swanlab_workspace=None,
        swanlab_run_name=None,
        swanlab_mode="cloud",
        swanlab_api_key=None,
        swanlab_logdir=None,
        swanlab_lark_webhook_url=None,
        swanlab_lark_secret=None,
        pure_bf16=False,
        stage="sft",
        finetuning_type="lora",
        use_llama_pro=False,
        use_adam_mini=False,
        use_mca=False,
        use_muon=False,
        use_dft_loss=False,
        freeze_vision_tower=True,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
        compute_accuracy=False,
        disable_shuffling=False,
        early_stopping_steps=None,
        plot_loss=True,
        include_effective_tokens_per_second=False,
    )

    print(generating_args)
    GeneratingArguments(
        do_sample=True,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        num_beams=1,
        max_length=1024,
        max_new_tokens=1024,
        repetition_penalty=1.0,
        length_penalty=1.0,
        skip_special_tokens=True,
    )
    """

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
    """
    print(callbacks)
    [<llamafactory.train.callbacks.LogCallback object at 0x7ff40e2e6f90>, <llamafactory.train.callbacks.ReporterCallback object at 0x7ff40e889640>]
    """

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

    print(args)
    None
    print(callbacks)
    None
    """
    args = read_args(args)
    """
    print(args)
    {
        'model_name_or_path': '/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507',
        'trust_remote_code': True,
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'lora',
        'lora_rank': 8,
        'lora_target': 'all',
        'dataset': 'identity,alpaca_en_demo',
        'template': 'qwen3_nothink',
        'cutoff_len': 2048,
        'max_samples': 1000,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,
        'output_dir': 'saves/qwen3-4b/lora/sft',
        'logging_steps': 10,
        'save_steps': 500,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'save_only_model': False,
        'report_to': 'none',
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 0.0001,
        'num_train_epochs': 3.0,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'resume_from_checkpoint': None
    }
    """
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    """
    print(ray_args)
    RayArguments(
        ray_run_name=None,
        ray_storage_path='./saves',
        ray_storage_filesystem=None,
        ray_num_workers=1,
        resources_per_worker={'GPU': 1},
        placement_strategy='PACK',
        ray_init_kwargs=None
    )

    print(ray_args.use_ray)
    False
    """
    callbacks = callbacks or []
    """
    print(callbacks)
    []
    """

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

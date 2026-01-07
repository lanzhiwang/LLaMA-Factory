# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES, EngineName
from ..extras.misc import check_dependencies, check_version, get_current_device, is_env_enabled
from ..extras.packages import is_mcore_adapter_available, is_transformers_version_greater_than
from .data_args import DataArguments
from .evaluation_args import EvaluationArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .training_args import RayArguments, TrainingArguments


logger = logging.get_logger(__name__)

check_dependencies()


_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]
_EVAL_CLS = tuple[ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments]

if is_mcore_adapter_available() and is_env_enabled("USE_MCA"):
    from mcore_adapter import TrainingArguments as McaTrainingArguments

    _TRAIN_MCA_ARGS = [ModelArguments, DataArguments, McaTrainingArguments, FinetuningArguments, GeneratingArguments]
    _TRAIN_MCA_CLS = tuple[
        ModelArguments, DataArguments, McaTrainingArguments, FinetuningArguments, GeneratingArguments
    ]
else:
    _TRAIN_MCA_ARGS = []
    _TRAIN_MCA_CLS = tuple()


def read_args(args: dict[str, Any] | list[str] | None = None) -> dict[str, Any] | list[str]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    """
    print(sys.argv)
    ['/root/miniconda3/envs/llamafactory/bin/llamafactory-cli', 'examples/train_lora/qwen3_lora_sft.yaml', 'model_name_or_path=/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507']
    """
    if sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif sys.argv[1].endswith(".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    else:
        return sys.argv[1:]


def _parse_args(
    parser: "HfArgumentParser", args: dict[str, Any] | list[str] | None = None, allow_extra_keys: bool = False
) -> tuple[Any]:
    args = read_args(args)
    if isinstance(args, dict):
        """
        print(parser.parse_dict(args, allow_extra_keys=allow_extra_keys))
        (
            ModelArguments(model_name_or_path='/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507', adapter_name_or_path=None, adapter_folder=None, cache_dir=None, use_fast_tokenizer=True, resize_vocab=False, split_special_tokens=False, add_tokens=None, add_special_tokens=None, new_special_tokens_config=None, init_special_tokens='noise_init', model_revision='main', low_cpu_mem_usage=True, rope_scaling=None, flash_attn=<AttentionFunction.AUTO: 'auto'>, shift_attn=False, mixture_of_depths=None, use_unsloth=False, use_unsloth_gc=False, enable_liger_kernel=False, moe_aux_loss_coef=None, disable_gradient_checkpointing=False, use_reentrant_gc=True, upcast_layernorm=False, upcast_lmhead_output=False, train_from_scratch=False, infer_backend=<EngineName.HF: 'huggingface'>, offload_folder='offload', use_kv_cache=True, use_v1_kernels=False, infer_dtype='auto', hf_hub_token=None, ms_hub_token=None, om_hub_token=None, print_param_status=False, trust_remote_code=True, quantization_method=<QuantizationMethod.BNB: 'bnb'>, quantization_bit=None, quantization_type='nf4', double_quantization=True, quantization_device_map=None, fp8=False, fp8_backend='auto', fp8_enable_fsdp_float8_all_gather=False, image_max_pixels=589824, image_min_pixels=1024, image_do_pan_and_scan=False, crop_to_patches=False, video_max_pixels=65536, video_min_pixels=256, video_fps=2.0, video_maxlen=128, use_audio_in_video=False, audio_sampling_rate=16000, export_dir=None, export_size=5, export_device='cpu', export_quantization_bit=None, export_quantization_dataset=None, export_quantization_nsamples=128, export_quantization_maxlen=1024, export_legacy_format=False, export_hub_model_id=None, use_kt=False, kt_optimize_rule=None, cpu_infer=32, chunk_size=8192, mode='normal', kt_maxlen=4096, kt_use_cuda_graph=True, kt_mode='normal', kt_force_think=False, vllm_maxlen=4096, vllm_gpu_util=0.7, vllm_enforce_eager=False, vllm_max_lora_rank=32, vllm_config=None, sglang_maxlen=4096, sglang_mem_fraction=0.7, sglang_tp_size=-1, sglang_config=None, sglang_lora_backend='triton', compute_dtype=None, device_map=None, model_max_length=None, block_diag_attn=False),
            DataArguments(template='qwen3_nothink', dataset=['identity', 'alpaca_en_demo'], eval_dataset=None, dataset_dir='data', media_dir='data', cutoff_len=2048, train_on_prompt=False, mask_history=False, streaming=False, buffer_size=16384, mix_strategy='concat', interleave_probs=None, overwrite_cache=False, preprocessing_batch_size=1000, preprocessing_num_workers=16, max_samples=1000, eval_num_beams=None, ignore_pad_token_for_loss=True, val_size=0.0, eval_on_each_dataset=False, packing=None, neat_packing=False, tool_format=None, default_system=None, enable_thinking=True, tokenized_path=None, data_shared_file_system=False),
            TrainingArguments(output_dir='saves/qwen3-4b/lora/sft', overwrite_output_dir=True, do_train=True, do_eval=False, do_predict=False, eval_strategy=<IntervalStrategy.NO: 'no'>, prediction_loss_only=False, per_device_train_batch_size=1, per_device_eval_batch_size=8, per_gpu_train_batch_size=None, per_gpu_eval_batch_size=None, gradient_accumulation_steps=8, eval_accumulation_steps=None, eval_delay=0, torch_empty_cache_steps=None, learning_rate=0.0001, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=-1, lr_scheduler_type=<SchedulerType.COSINE: 'cosine'>, lr_scheduler_kwargs={}, warmup_ratio=0.1, warmup_steps=0, log_level='passive', log_level_replica='warning', log_on_each_node=True, logging_dir='saves/qwen3-4b/lora/sft/runs/Jan07_10-10-01_nj-p-zdtc-h20-059', logging_strategy=<IntervalStrategy.STEPS: 'steps'>, logging_first_step=False, logging_steps=10, logging_nan_inf_filter=True, save_strategy=<SaveStrategy.STEPS: 'steps'>, save_steps=500, save_total_limit=None, save_safetensors=True, save_on_each_node=False, save_only_model=False, restore_callback_states_from_checkpoint=False, no_cuda=False, use_cpu=False, use_mps_device=False, seed=42, data_seed=None, jit_mode_eval=False, bf16=True, fp16=False, fp16_opt_level='O1', half_precision_backend='auto', bf16_full_eval=False, fp16_full_eval=False, tf32=None, local_rank=0, ddp_backend=None, tpu_num_cores=None, tpu_metrics_debug=False, debug=[], dataloader_drop_last=False, eval_steps=None, dataloader_num_workers=4, dataloader_prefetch_factor=None, past_index=-1, run_name=None, disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None, ignore_data_skip=False, fsdp=[], fsdp_min_num_params=0, fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}, fsdp_transformer_layer_cls_to_wrap=None, accelerator_config=AcceleratorConfig(split_batches=False, dispatch_batches=None, even_batches=True, use_seedable_sampler=True, non_blocking=False, gradient_accumulation_kwargs=None, use_configured_state=False), parallelism_config=None, deepspeed=None, label_smoothing_factor=0.0, optim=<OptimizerNames.ADAMW_TORCH_FUSED: 'adamw_torch_fused'>, optim_args=None, adafactor=False, group_by_length=False, length_column_name='length', report_to=[], project='huggingface', trackio_space_id='trackio', ddp_find_unused_parameters=None, ddp_bucket_cap_mb=None, ddp_broadcast_buffers=None, dataloader_pin_memory=True, dataloader_persistent_workers=False, skip_memory_metrics=True, use_legacy_prediction_loop=False, push_to_hub=False, resume_from_checkpoint=None, hub_model_id=None, hub_strategy=<HubStrategy.EVERY_SAVE: 'every_save'>, hub_token=None, hub_private_repo=None, hub_always_push=False, hub_revision=None, gradient_checkpointing=False, gradient_checkpointing_kwargs=None, include_inputs_for_metrics=False, include_for_metrics=[], eval_do_concat_batches=True, fp16_backend='auto', push_to_hub_model_id=None, push_to_hub_organization=None, push_to_hub_token=None, mp_parameters='', auto_find_batch_size=False, full_determinism=False, torchdynamo=None, ray_scope='last', ddp_timeout=180000000, torch_compile=False, torch_compile_backend=None, torch_compile_mode=None, include_tokens_per_second=False, include_num_input_tokens_seen='no', neftune_noise_alpha=None, optim_target_modules=None, batch_eval_metrics=False, eval_on_start=False, use_liger_kernel=False, liger_kernel_config=None, eval_use_gather_object=False, average_tokens_across_devices=True, sortish_sampler=False, predict_with_generate=False, generation_max_length=None, generation_num_beams=None, generation_config=None, ray_run_name=None, ray_storage_path='./saves', ray_storage_filesystem=None, ray_num_workers=1, resources_per_worker={'GPU': 1}, placement_strategy='PACK', ray_init_kwargs=None),
            FinetuningArguments(freeze_trainable_layers=2, freeze_trainable_modules=['all'], freeze_extra_modules=None, additional_target=None, module_dropout=0.0, oft_rank=0, oft_block_size=32, oft_target=['all'], create_new_adapter=False, lora_alpha=16, lora_dropout=0.0, lora_rank=8, lora_target=['all'], loraplus_lr_ratio=None, loraplus_lr_embedding=1e-06, use_rslora=False, use_dora=False, pissa_init=False, pissa_iter=16, pissa_convert=False, pref_beta=0.1, pref_ftx=0.0, pref_bco_weight=0.0, pref_loss='sigmoid', dpo_label_smoothing=0.0, kto_chosen_weight=1.0, kto_rejected_weight=1.0, simpo_gamma=0.5, ppo_buffer_size=1, ppo_epochs=4, ppo_score_norm=False, ppo_target=6.0, ppo_whiten_rewards=False, ref_model=None, ref_model_adapters=None, ref_model_quantization_bit=None, reward_model=None, reward_model_adapters=None, reward_model_quantization_bit=None, reward_model_type='lora', ld_alpha=None, use_galore=False, galore_target=['all'], galore_rank=16, galore_update_interval=200, galore_scale=2.0, galore_proj_type='std', galore_layerwise=False, use_apollo=False, apollo_target=['all'], apollo_rank=16, apollo_update_interval=200, apollo_scale=32.0, apollo_proj='random', apollo_proj_type='std', apollo_scale_type='channel', apollo_layerwise=False, apollo_scale_front=False, use_badam=False, badam_mode='layer', badam_start_block=None, badam_switch_mode='ascending', badam_switch_interval=50, badam_update_ratio=0.05, badam_mask_mode='adjacent', badam_verbose=0, use_swanlab=False, swanlab_project='llamafactory', swanlab_workspace=None, swanlab_run_name=None, swanlab_mode='cloud', swanlab_api_key=None, swanlab_logdir=None, swanlab_lark_webhook_url=None, swanlab_lark_secret=None, pure_bf16=False, stage='sft', finetuning_type='lora', use_llama_pro=False, use_adam_mini=False, use_mca=False, use_muon=False, use_dft_loss=False, freeze_vision_tower=True, freeze_multi_modal_projector=True, freeze_language_model=False, compute_accuracy=False, disable_shuffling=False, early_stopping_steps=None, plot_loss=True, include_effective_tokens_per_second=False),
            GeneratingArguments(do_sample=True, temperature=0.95, top_p=0.7, top_k=50, num_beams=1, max_length=1024, max_new_tokens=1024, repetition_penalty=1.0, length_penalty=1.0, skip_special_tokens=True)
        )
        """
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args)


def _set_transformers_logging() -> None:
    if os.getenv("LLAMAFACTORY_VERBOSITY", "INFO") in ["DEBUG", "INFO"]:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _set_env_vars() -> None:
    if is_torch_npu_available():
        # avoid JIT compile on NPU devices, see https://zhuanlan.zhihu.com/p/660875458
        torch.npu.set_compile_mode(jit_compile=is_env_enabled("NPU_JIT_COMPILE"))
        # avoid use fork method on NPU devices, see https://github.com/hiyouga/LLaMA-Factory/issues/7447
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _verify_model_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantization is only compatible with the LoRA or OFT method.")

        if finetuning_args.pissa_init:
            raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA for a quantized model.")

        if model_args.resize_vocab:
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")

    if data_args.template == "yi" and model_args.use_fast_tokenizer:
        logger.warning_rank0("We should use slow tokenizer for the Yi models. Change `use_fast_tokenizer` to False.")
        model_args.use_fast_tokenizer = False

    # Validate advanced training features
    if model_args.fp8 and model_args.quantization_bit is not None:
        raise ValueError("FP8 training is not compatible with quantization. Please disable one of them.")

    if model_args.fp8_enable_fsdp_float8_all_gather and not model_args.fp8:
        logger.warning_rank0("fp8_enable_fsdp_float8_all_gather requires fp8=True. Setting fp8=True.")
        model_args.fp8 = True


def _check_extra_dependencies(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    training_args: Optional["TrainingArguments"] = None,
) -> None:
    if model_args.use_kt:
        check_version("ktransformers", mandatory=True)

    if model_args.use_unsloth:
        check_version("unsloth", mandatory=True)

    if model_args.enable_liger_kernel:
        check_version("liger-kernel", mandatory=True)

    if model_args.mixture_of_depths is not None:
        check_version("mixture-of-depth>=1.1.6", mandatory=True)

    if model_args.infer_backend == EngineName.VLLM:
        check_version("vllm>=0.4.3,<=0.11.0")
        check_version("vllm", mandatory=True)
    elif model_args.infer_backend == EngineName.SGLANG:
        check_version("sglang>=0.4.5")
        check_version("sglang", mandatory=True)

    if finetuning_args.use_galore:
        check_version("galore_torch", mandatory=True)

    if finetuning_args.use_apollo:
        check_version("apollo_torch", mandatory=True)

    if finetuning_args.use_badam:
        check_version("badam>=1.2.1", mandatory=True)

    if finetuning_args.use_adam_mini:
        check_version("adam-mini", mandatory=True)

    if finetuning_args.use_swanlab:
        check_version("swanlab", mandatory=True)

    if finetuning_args.plot_loss:
        check_version("matplotlib", mandatory=True)

    if training_args is not None:
        if training_args.deepspeed:
            # pin deepspeed version < 0.17 because of https://github.com/deepspeedai/DeepSpeed/issues/7347
            check_version("deepspeed", mandatory=True)
            check_version("deepspeed>=0.10.0,<=0.16.9")

        if training_args.predict_with_generate:
            check_version("jieba", mandatory=True)
            check_version("nltk", mandatory=True)
            check_version("rouge_chinese", mandatory=True)


def _parse_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    """
    print(args)
    {'model_name_or_path': '/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507', 'trust_remote_code': True, 'stage': 'sft', 'do_train': True, 'finetuning_type': 'lora', 'lora_rank': 8, 'lora_target': 'all', 'dataset': 'identity,alpaca_en_demo', 'template': 'qwen3_nothink', 'cutoff_len': 2048, 'max_samples': 1000, 'preprocessing_num_workers': 16, 'dataloader_num_workers': 4, 'output_dir': 'saves/qwen3-4b/lora/sft', 'logging_steps': 10, 'save_steps': 500, 'plot_loss': True, 'overwrite_output_dir': True, 'save_only_model': False, 'report_to': 'none', 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 8, 'learning_rate': 0.0001, 'num_train_epochs': 3.0, 'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.1, 'bf16': True, 'ddp_timeout': 180000000, 'resume_from_checkpoint': None}
    """
    parser = HfArgumentParser(_TRAIN_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    """
    print(parser)
    HfArgumentParser(prog='llamafactory-cli', usage=None, description=None, formatter_class=<class 'argparse.ArgumentDefaultsHelpFormatter'>, conflict_handler='error', add_help=True)
    print(allow_extra_keys)
    False
    """
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_train_mca_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_MCA_CLS:
    parser = HfArgumentParser(_TRAIN_MCA_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_args(
        parser, args, allow_extra_keys=allow_extra_keys
    )

    _configure_mca_training_args(training_args, data_args, finetuning_args)

    return model_args, data_args, training_args, finetuning_args, generating_args


def _configure_mca_training_args(training_args, data_args, finetuning_args) -> None:
    """Patch training args to avoid args checking errors and sync MCA settings."""
    training_args.predict_with_generate = False
    training_args.generation_max_length = data_args.cutoff_len
    training_args.generation_num_beams = 1
    training_args.use_mca = True
    finetuning_args.use_mca = True


def _parse_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def _parse_eval_args(args: dict[str, Any] | list[str] | None = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)


def get_ray_args(args: dict[str, Any] | list[str] | None = None) -> RayArguments:
    parser = HfArgumentParser(RayArguments)
    (ray_args,) = _parse_args(parser, args, allow_extra_keys=True)
    return ray_args


def get_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    if is_env_enabled("USE_MCA"):
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_mca_args(args)
    else:
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
        finetuning_args.use_mca = False

    # Setup logging
    """
    print(training_args.should_log)
    True
    """
    if training_args.should_log:
        _set_transformers_logging()

    # Check arguments
    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt` or `mask_history` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo":
        if not training_args.do_train:
            raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

        if model_args.shift_attn:
            raise ValueError("PPO training is incompatible with S^2-Attn.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_kt:
            raise ValueError("KTransformers does not support lora reward model.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
            raise ValueError("Unsloth does not support lora reward model.")

        if training_args.report_to and training_args.report_to[0] not in ["wandb", "tensorboard"]:
            raise ValueError("PPO only accepts wandb or tensorboard logger.")

    if not model_args.use_kt and training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training with `llamafactory-cli` or `torchrun`.")

    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    if (training_args.do_eval or training_args.do_predict or training_args.predict_with_generate) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please make sure eval_dataset be provided or val_size >1e-6")

    if training_args.predict_with_generate:
        if is_deepspeed_zero3_enabled():
            raise ValueError("`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.")

        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")

    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("Please use scripts/pissa_init.py to initialize PiSSA in DeepSpeed ZeRO-3.")

    if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")

        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")

    if training_args.parallel_mode == ParallelMode.DISTRIBUTED:
        if finetuning_args.use_galore and finetuning_args.galore_layerwise:
            raise ValueError("Distributed training does not support layer-wise GaLore.")

        if finetuning_args.use_apollo and finetuning_args.apollo_layerwise:
            raise ValueError("Distributed training does not support layer-wise APOLLO.")

        if finetuning_args.use_badam:
            if finetuning_args.badam_mode == "ratio":
                raise ValueError("Radio-based BAdam does not yet support distributed training, use layer-wise BAdam.")
            elif not is_deepspeed_zero3_enabled():
                raise ValueError("Layer-wise BAdam only supports DeepSpeed ZeRO-3 training.")

    if training_args.deepspeed is not None and (finetuning_args.use_galore or finetuning_args.use_apollo):
        raise ValueError("GaLore and APOLLO are incompatible with DeepSpeed yet.")

    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI and Web.")

    if model_args.use_unsloth and is_deepspeed_zero3_enabled():
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if model_args.use_kt and is_deepspeed_zero3_enabled():
        raise ValueError("KTransformers is incompatible with DeepSpeed ZeRO-3.")

    if data_args.neat_packing and is_transformers_version_greater_than("4.53.0"):
        raise ValueError("Neat packing is incompatible with transformers>=4.53.0.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    if (
        training_args.do_train
        and finetuning_args.finetuning_type == "lora"
        and model_args.quantization_bit is None
        and model_args.resize_vocab
        and finetuning_args.additional_target is None
    ):
        logger.warning_rank0(
            "Remember to add embedding layers to `additional_target` to make the added tokens trainable."
        )

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning_rank0("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning_rank0("We recommend enable mixed precision training.")

    if (
        training_args.do_train
        and (finetuning_args.use_galore or finetuning_args.use_apollo)
        and not finetuning_args.pure_bf16
    ):
        logger.warning_rank0(
            "Using GaLore or APOLLO with mixed precision training may significantly increases GPU memory usage."
        )

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning_rank0("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning_rank0("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    if finetuning_args.finetuning_type == "lora":
        # https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/trainer.py#L782
        training_args.label_names = training_args.label_names or ["labels"]

    if "swanlab" in training_args.report_to and finetuning_args.use_swanlab:
        training_args.report_to.remove("swanlab")

    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.info_rank0("Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.")
        training_args.ddp_find_unused_parameters = False

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in ["full", "freeze"]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning_rank0("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info_rank0(f"Resuming training from {training_args.resume_from_checkpoint}.")
            logger.info_rank0("Change `output_dir` or use `overwrite_output_dir` to avoid.")

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning_rank0(
            f"Add {training_args.resume_from_checkpoint} to `adapter_name_or_path` to resume training from checkpoint."
        )

    # Post-process model arguments
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    # Log on each process the small summary
    logger.info(
        f"Process rank: {training_args.process_index}, "
        f"world size: {training_args.world_size}, device: {training_args.device}, "
        f"distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED}, "
        f"compute dtype: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(args)

    # Setup logging
    _set_transformers_logging()

    # Check arguments
    if model_args.infer_backend == "vllm":
        if finetuning_args.stage != "sft":
            raise ValueError("vLLM engine only supports auto-regressive models.")

        if model_args.quantization_bit is not None:
            raise ValueError("vLLM engine does not support bnb quantization (GPTQ and AWQ are supported).")

        if model_args.rope_scaling is not None:
            raise ValueError("vLLM engine does not support RoPE scaling.")

        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("vLLM only accepts a single adapter. Merge them first.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    # Post-process model arguments
    if model_args.export_dir is not None and model_args.export_device == "cpu":
        model_args.device_map = {"": torch.device("cpu")}
        if data_args.cutoff_len != DataArguments().cutoff_len:  # override cutoff_len if it is not default
            model_args.model_max_length = data_args.cutoff_len
    else:
        model_args.device_map = "auto"

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: dict[str, Any] | list[str] | None = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = _parse_eval_args(args)

    # Setup logging
    _set_transformers_logging()

    # Check arguments
    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI and Web.")

    _set_env_vars()
    _verify_model_args(model_args, data_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args)

    model_args.device_map = "auto"

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args

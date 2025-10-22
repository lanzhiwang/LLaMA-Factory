# 调优算法

参考: https://llamafactory.readthedocs.io/zh-cn/latest/advanced/adapters.html

## Full Parameter Fine-tuning

```
finetuning_type: full
```

## Freeze

```
finetuning_type: freeze
```

## LoRA

```
finetuning_type: lora
```

### LoRA+

```bash
# finetuning_type: lora
# loraplus_lr_ratio

# examples/extras/loraplus/llama3_lora_sft.yaml

llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
finetuning_type=lora \
loraplus_lr_ratio=16.0

```

### rsLoRA

```
finetuning_type: lora
use_rslora: True
```

### DoRA

```
finetuning_type: lora
use_dora: True
```

### PiSSA

```bash
# finetuning_type: lora
# pissa_init: True

# examples/extras/pissa/llama3_lora_sft.yaml

python scripts/pissa_init.py \
--model_name_or_path /root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
--output_dir saves/llama3-8b/lora/sft

$ python scripts/pissa_init.py --help
NAME
    pissa_init.py - Initialize LoRA weights with Principal Singular values and Singular vectors Adaptation (PiSSA).

SYNOPSIS
    pissa_init.py MODEL_NAME_OR_PATH OUTPUT_DIR <flags>

DESCRIPTION
    Usage: python pissa_init.py --model_name_or_path path_to_model --output_dir output_dir

POSITIONAL ARGUMENTS
    MODEL_NAME_OR_PATH
        Type: str
    OUTPUT_DIR
        Type: str

FLAGS
    -p, --pissa_iter=PISSA_ITER
        Type: int
        Default: 16
    --lora_alpha=LORA_ALPHA
        Type: Optional[int]
        Default: None
    --lora_rank=LORA_RANK
        Type: int
        Default: 16
    --lora_dropout=LORA_DROPOUT
        Type: float
        Default: 0
    --lora_target=LORA_TARGET
        Type: tuple
        Default: ('q_proj', 'v_proj')
    -s, --save_safetensors=SAVE_SAFETENSORS
        Type: bool
        Default: True

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS

```

## Galore

```bash
# 不要将 LoRA 和 GaLore/BAdam 一起使用
# galore_layerwise 为 true 时请不要设置 gradient_accumulation 参数
# Distributed training does not support layer-wise GaLore

# finetuning_type: full | freeze
# use_galore: true

# examples/extras/galore/llama3_full_sft.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
finetuning_type=full \
use_galore=true

```

## BAdam

```bash
# 不要将 LoRA 和 GaLore/BAdam 一起使用。
# 使用 BAdam 时请设置 finetuning_type 为 full 且 pure_bf16 为 True
# badam_mode = layer 时仅支持使用 DeepSpeed ZeRO3 进行单卡或多卡训练
# badam_mode = ratio 时仅支持单卡训练

# finetuning_type: full
# use_badam: true

# examples/extras/badam/llama3_full_sft.yaml

deepspeed --num_gpus 8 src/train.py \
--model_name_or_path /root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
--trust_remote_code true \
--stage sft \
--do_train true \
--finetuning_type full \
--use_badam true \
--badam_mode layer \
--badam_switch_mode ascending \
--badam_switch_interval 50 \
--badam_verbose 2 \
--deepspeed examples/deepspeed/ds_z3_config.json \
--dataset identity,alpaca_en_demo \
--template llama3 \
--cutoff_len 2048 \
--max_samples 1000 \
--overwrite_cache true \
--preprocessing_num_workers 16 \
--dataloader_num_workers 4 \
--output_dir saves/llama3-8b/full/sft \
--logging_steps 10 \
--save_steps 500 \
--plot_loss true \
--overwrite_output_dir true \
--save_only_model false \
--report_to none \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 1.0e-5 \
--num_train_epochs 3.0 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1

```

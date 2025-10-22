# 加速

参考: https://llamafactory.readthedocs.io/zh-cn/latest/advanced/acceleration.html#

## FlashAttention

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft \
flash_attn=fa2

```

## Unsloth

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft \
use_unsloth=True

```

## Liger Kernel

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft \
enable_liger_kernel=True

```

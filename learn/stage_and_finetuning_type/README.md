# stage

```bash

# stage == "pt"
llamafactory-cli train examples/train_lora/qwen3_lora_pretrain.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

# stage == "sft"
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

# stage == "rm"
llamafactory-cli train examples/train_lora/qwen3_lora_reward.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

# stage == "ppo"

# stage == "dpo"
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

# stage == "kto
llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

```

# finetuning_type

```bash

# finetuning_type: lora
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

# finetuning_type: oft
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./

llamafactory-cli train examples/extras/oft/qwen2_5vl_oft_sft.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen2.5-VL-7B-Instruct

# finetuning_type: freeze
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct --local_dir ./

llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/meta-llama/Meta-Llama-3-8B-Instruct

# finetuning_type: full
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml \
model_name_or_path=/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507

```

## SFT 训练

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft \
learning_rate=1e-5 \
logging_steps=1

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Qwen3-32B \
output_dir=saves/Qwen3-32B/lora/sft \
learning_rate=1e-5 \
logging_steps=1 \
template=qwen3

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft \
learning_rate=1e-5 \
logging_steps=1

#######################################################################

llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=1 python ./src/train.py examples/train_lora/qwen3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=1,2 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Qwen3-4B-Instruct-2507

```

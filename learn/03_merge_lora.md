## LoRA 合并

```bash
# LoRA 合并
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
export_dir=output/llama3_lora_sft

# LoRA 合并并量化
llamafactory-cli export examples/merge_lora/llama3_q_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
export_dir=output/llama3_qlora_sft

```

## 量化

```bash
# 量化基础模型
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
export_dir=output/llama3_gptq

# 量化 LoRA 合并之后的模型
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml \
model_name_or_path=output/llama3_lora_sft \
export_dir=output/llama3_lora_gptq

```

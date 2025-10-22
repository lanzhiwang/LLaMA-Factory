## 评估

```bash
# 通用能力评估

llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
save_dir=saves/llama3-8b/lora/eval

# NLG 评估
# 获得模型的 BLEU 和 ROUGE 分数以评价模型生成质量
llamafactory-cli train examples/extras/nlg_eval/llama3_lora_predict.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
output_dir=saves/llama3-8b/lora/predict

python scripts/vllm_infer.py --model_name_or_path output/llama3_lora_sft --dataset alpaca_en_demo

```

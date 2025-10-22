## 推理

```bash
# 原始模型推理配置
llamafactory-cli chat examples/inference/llama3.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
infer_backend=huggingface

llamafactory-cli chat examples/inference/llama3.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
infer_backend=vllm

# 微调模型推理配置
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
infer_backend=huggingface

$ ll /root/.cache/vllm
total 16
drwxr-xr-x  3 root root 4096 Jun 24 23:34 ./
drwx------ 16 root root 4096 Aug 19 18:29 ../
-rw-r--r--  1 root root 1154 Jun 24 23:31 gpu_p2p_access_cache_for_0,1,2,3,4,5,6,7.json
drwxr-xr-x  7 root root 4096 Aug 23 13:22 torch_compile_cache/

$ rm -rf  /root/.cache/vllm/*

# 这个没有运行成功
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
infer_backend=vllm

# 微调合并量化之后的模型推理配置
llamafactory-cli chat examples/inference/llama3.yaml \
model_name_or_path=output/llama3_qlora_sft \
infer_backend=huggingface

llamafactory-cli chat examples/inference/llama3.yaml \
model_name_or_path=output/llama3_qlora_sft \
infer_backend=vllm

# 多模态模型
llamafactory-cli webchat examples/inference/llava1_5.yaml \
model_name_or_path=/root/LLaMA-Factory/models/llava-1.5-7b-hf \
infer_backend=huggingface

llamafactory-cli webchat examples/inference/llava1_5.yaml \
model_name_or_path=/root/LLaMA-Factory/models/llava-1.5-7b-hf \
infer_backend=vllm

# 批量推理
python scripts/vllm_infer.py --model_name_or_path /root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct --dataset alpaca_en_demo

python scripts/vllm_infer.py --model_name_or_path output/llama3_qlora_sft --dataset alpaca_en_demo

# api
API_PORT=7860 CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli api examples/inference/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
infer_backend=huggingface

API_PORT=7860 CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli api examples/inference/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
finetuning_type=lora \
infer_backend=huggingface

# 这个没有运行成功
API_PORT=7860 CUDA_VISIBLE_DEVICES=0,1,2,4 llamafactory-cli api examples/inference/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
infer_backend=vllm

# 这个没有运行成功
API_PORT=7860 CUDA_VISIBLE_DEVICES=0,1,2,4 llamafactory-cli api examples/inference/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
adapter_name_or_path=saves/llama3-8b/lora/sft \
finetuning_type=lora \
infer_backend=vllm

python api_call_example.py

```

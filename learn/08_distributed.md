## 分布训练

### NativeDDP

#### 单机多卡

```bash
# llamafactory-cli
# 您可以使用 llamafactory-cli 启动 NativeDDP 引擎.

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/full/sft \
learning_rate=1e-5 \
logging_steps=1

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/full/sft \
learning_rate=1e-5 \
logging_steps=1

# torchrun
# 您也可以使用 torchrun 指令启动 NativeDDP 引擎进行单机多卡训练.

torchrun --standalone --nnodes=1 --nproc-per-node=8  src/train.py \
--stage sft \
--model_name_or_path /root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
--do_train true \
--dataset identity,alpaca_en_demo \
--template llama3 \
--finetuning_type lora \
--output_dir saves/llama3-8b/full/sft \
--overwrite_cache true \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--save_steps 500 \
--learning_rate 1e-5 \
--num_train_epochs 3.0 \
--plot_loss true \
--bf16 true \
--report_to none

# accelerate
# 您还可以使用 accelerate 指令启动进行单机多卡训练
accelerate launch \
--config_file examples/accelerate/accelerate_singleNode_config.yaml \
src/train.py examples/train_full/llama3_full_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/full/sft \
learning_rate=1e-5 \
logging_steps=1

```

### DeepSpeed

#### 单机多卡

```bash
# llamafactory-cli
# 您可以使用 llamafactory-cli 启动 DeepSpeed 引擎进行单机多卡训练

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/full/sft \
learning_rate=1e-5 \
logging_steps=1

# deepspeed
# 您也可以使用 deepspeed 指令启动 DeepSpeed 引擎进行单机多卡训练

deepspeed --num_gpus 8 src/train.py \
--deepspeed examples/deepspeed/ds_z3_config.json \
--stage sft \
--model_name_or_path /root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
--do_train true \
--dataset identity,alpaca_en_demo \
--template llama3 \
--finetuning_type lora \
--output_dir saves/llama3-8b/full/sft \
--overwrite_cache true \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--save_steps 500 \
--learning_rate 1e-5 \
--num_train_epochs 3.0 \
--plot_loss true \
--bf16 true \
--report_to none

```

### FSDP

```bash
accelerate launch \
--config_file examples/accelerate/fsdp_config.yaml \
src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft

```



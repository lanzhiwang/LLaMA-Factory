# 训练方法

参考: https://llamafactory.readthedocs.io/zh-cn/latest/advanced/trainers.html

## Pre-training

```bash
# c4_demo 数据集
{
    "text": "5 Bedroom House To Rent in London for £30,116 per throughout."
}

llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
stage=pt \
finetuning_type=lora \
dataset=c4_demo \
output_dir=saves/llama3-8b/lora/pretrain

```

## Post-training

### Supervised Fine-Tuning

```bash
# identity 数据集
{
    "instruction": "hi",
    "input": "",
    "output": "Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?"
}

# alpaca_en_demo 数据集
{
    "instruction": "Describe a process of making crepes.",
    "input": "",
    "output": "Making crepes is an easy and delicious process! Enjoy!"
}

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
stage=sft \
finetuning_type=lora \
dataset=identity,alpaca_en_demo \
output_dir=saves/llama3-8b/lora/sft

```

### RLHF

#### Reward model

**训练一个奖励模型**

```bash
# dpo_en_demo 数据集
{
    "conversations": [
        {
            "from": "human",
            "value": "Hi! I'd like to create a new language game simulating the first."
        }
    ],
    "chosen": {
        "from": "gpt",
        "value": "That sounds like a fun and engaging idea! Good luck!"
    },
    "rejected": {
        "from": "gpt",
        "value": "Hello! I'd be happy to help you and honest assistant!"
    }
}

llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
stage=rm \
finetuning_type=lora \
dataset=dpo_en_demo \
output_dir=saves/llama3-8b/lora/reward

```

#### PPO

**在训练奖励完模型之后，我们可以开始进行模型的强化学习部分**

```bash
# identity 数据集
{
    "instruction": "hi",
    "input": "",
    "output": "Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?"
}

# alpaca_en_demo 数据集
{
    "instruction": "Describe a process of making crepes.",
    "input": "",
    "output": "Making crepes is an easy and delicious process! Enjoy!"
}

llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
reward_model=saves/llama3-8b/lora/reward \
stage=ppo \
finetuning_type=lora \
dataset=identity,alpaca_en_demo \
output_dir=saves/llama3-8b/lora/ppo

```

### DPO

```bash
# dpo_en_demo 数据集
{
    "conversations": [
        {
            "from": "human",
            "value": "Hi! I'd like to create a new language game simulating the first."
        }
    ],
    "chosen": {
        "from": "gpt",
        "value": "That sounds like a fun and engaging idea! Good luck!"
    },
    "rejected": {
        "from": "gpt",
        "value": "Hello! I'd be happy to help you and honest assistant!"
    }
}

llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
stage=dpo \
finetuning_type=lora \
dataset=dpo_en_demo \
output_dir=saves/llama3-8b/lora/dpo

```

### KTO

```bash
# kto_en_demo 数据集
{
    "messages": [
        {
            "content": "The Federal Trade Commission is going after spyware",
            "role": "user"
        },
        {
            "content": "Ooh ooh ah ah! *giggle* Let me think... Ooh ooh ah ah!",
            "role": "assistant"
        }
    ],
    "label": true
}

{
    "messages": [
        {
            "content": "Develop a Java program that simulates a basic",
            "role": "user"
        },
        {
            "content": "import java.sql",
            "role": "assistant"
        },
        {
            "content": "Explain the importance.",
            "role": "user"
        }
    ],
    "label": false
}

llamafactory-cli train examples/train_lora/llama3_lora_kto.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
stage=kto \
finetuning_type=lora \
dataset=kto_en_demo \
output_dir=saves/llama3-8b/lora/kto

```


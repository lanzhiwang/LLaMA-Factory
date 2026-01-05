# 数据处理

[dataset_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json/) 包含了所有经过预处理的 **本地数据集** 以及 **在线数据集**. 如果您希望使用自定义数据集, 请 **务必** 在 `dataset_info.json` 文件中添加对数据集及其内容的定义.

目前我们支持 `Alpaca` 格式和 `ShareGPT` 格式的数据集.

## Alpaca

针对不同任务, 数据集格式要求如下:

- 指令监督微调

- 预训练

- 偏好训练

- KTO

- 多模态

### 指令监督微调数据集

样例数据集: [指令监督微调样例数据集](./alpaca_zh_demo.json)

指令监督微调(Instruct Tuning)通过让模型学习详细的指令以及对应的回答来优化模型在特定指令下的表现.

`instruction` 列对应的内容为人类指令,  `input` 列对应的内容为人类输入,  `output` 列对应的内容为模型回答. 下面是一个例子

```json
{
    "instruction": "计算这些物品的总费用.",
    "input": "输入: 汽车 - $3000, 衣服 - $100, 书 - $20.",
    "output": "汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120."
}
```

在进行指令监督微调时, `instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为最终的人类输入, 即人类输入为 `instruction\ninput`. 而 `output` 列对应的内容为模型回答. 在上面的例子中, 人类的最终输入是:

```
计算这些物品的总费用.
输入: 汽车 - $3000, 衣服 - $100, 书 - $20.
```

模型的回答是:

```
汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120.
```

如果指定, `system` 列对应的内容将被作为系统提示词.

`history` 列是由多个字符串二元组构成的列表, 分别代表历史消息中每轮对话的指令和回答. 注意在指令监督微调时, 历史消息中的回答内容也会被用于模型学习.

指令监督微调数据集 **格式要求** 如下:

```json
[
    {
        "instruction": "人类指令(必填)",
        "input": "人类输入(选填)",
        "output": "模型回答(必填)",
        "system": "系统提示词(选填)",
        "history": [
            [
                "第一轮指令(选填)",
                "第一轮回答(选填)"
            ],
            [
                "第二轮指令(选填)",
                "第二轮回答(选填)"
            ]
        ]
    }
]
```

下面提供一个 alpaca 格式 **多轮** 对话的例子, 对于单轮对话只需省略 `history` 列即可.

```json
[
    {
        "instruction": "今天的天气怎么样?",
        "input": "",
        "output": "今天的天气不错, 是晴天. ",
        "history": [
            [
                "今天会下雨吗?",
                "今天不会下雨, 是个好天气. "
            ],
            [
                "今天适合出去玩吗?",
                "非常适合, 空气质量很好."
            ]
        ]
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
            "history": "history"
        }
    }
}
```

### 预训练数据集

样例数据集: [预训练样例数据集](./c4_demo.jsonl)

```json
{"text": "Don’t think you need all the bells and whistles? No problem. McKinley Heating Service Experts Heating & Air Conditioning offers basic air cleaners that work to improve the quality of the air in your home without breaking the bank. It is a low-cost solution that will ensure you and your family are living comfortably.\nIt’s a good idea to understand the efficiency rate of the filters, which measures what size of molecules can get through the filter. Basic air cleaners can filter some of the dust, dander and pollen that need to be removed. They are 85% efficient, and usually have a 6-inch cleaning surface.\nBasic air cleaners are not too expensive and do the job well. If you do want to hear more about upgrading from a basic air cleaner, let the NATE-certified experts at McKinley Heating Service Experts in Edmonton talk to you about their selection.\nEither way, now’s a perfect time to enhance and protect the indoor air quality in your home, for you and your loved ones.\nIf you want expert advice and quality service in Edmonton, give McKinley Heating Service Experts a call at 780-800-7092 to get your questions or concerns related to your HVAC system addressed."}

```

大语言模型通过学习未被标记的文本进行预训练, 从而学习语言的表征. 通常, 预训练数据集从互联网上获得, 因为互联网上提供了大量的不同领域的文本信息, 有助于提升模型的泛化能力. 预训练数据集文本描述格式如下:

```json
[
    {
        "text": "document"
    },
    {
        "text": "document"
    }
]
```

在预训练时, 只有 `text` 列中的 **内容** (即 document )会用于模型学习.

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "columns": {
            "prompt": "text"
        }
    }
}
```

### 偏好数据集

偏好数据集用于**奖励模型训练**、**DPO 训练**和 **ORPO 训练**. 对于系统指令和人类输入, 偏好数据集给出了一个更优的回答和一个更差的回答.

[一些研究](https://openai.com/index/instruction-following/) 表明通过让模型学习"什么更好"可以使得模型更加迎合人类的需求. 甚至可以使得参数相对较少的模型的表现优于参数更多的模型.

偏好数据集需要在 `chosen` 列中提供更优的回答, 并在 `rejected` 列中提供更差的回答, 在一轮问答中其格式如下:

```json
[
    {
        "instruction": "人类指令(必填)",
        "input": "人类输入(选填)",
        "chosen": "优质回答(必填)",
        "rejected": "劣质回答(必填)"
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "ranking": true,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
}
```

### KTO 数据集

KTO 数据集与偏好数据集类似, 但不同于给出一个更优的回答和一个更差的回答, KTO 数据集对每一轮问答只给出一个 `true/false` 的 `label`. 除了 `instruction` 以及 `input` 组成的人类最终输入和模型回答 `output`, KTO 数据集还需要额外添加一个 `kto_tag` 列(`true/false`)来表示人类的反馈.

在一轮问答中其格式如下:

```json
[
    {
        "instruction": "人类指令(必填)",
        "input": "人类输入(选填)",
        "output": "模型回答(必填)",
        "kto_tag": "人类反馈 [true/false](必填)"
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "kto_tag": "kto_tag"
        }
    }
}
```

### 多模态数据集

目前我们支持 `多模态图像数据集`、`视频数据集` 以及 `音频数据集` 的输入.

#### 图像数据集

多模态图像数据集需要额外添加一个 `images` 列, 包含输入图像的路径. 注意图片的数量必须与文本中所有 `<image>` 标记的数量严格一致.

```json
[
    {
        "instruction": "人类指令(必填)",
        "input": "人类输入(选填)",
        "output": "模型回答(必填)",
        "images": [
            "图像路径(必填)"
        ]
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "images": "images"
        }
    }
}
```

#### 视频数据集

多模态视频数据集需要额外添加一个 `videos` 列, 包含输入视频的路径. 注意视频的数量必须与文本中所有 `<video>` 标记的数量严格一致.

```json
[
    {
        "instruction": "人类指令(必填)",
        "input": "人类输入(选填)",
        "output": "模型回答(必填)",
        "videos": [
            "视频路径(必填)"
        ]
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "videos": "videos"
        }
    }
}
```

#### 音频数据集

多模态音频数据集需要额外添加一个 `audio` 列, 包含输入图像的路径. 注意音频的数量必须与文本中所有 `<audio>` 标记的数量严格一致.

```json
[
    {
        "instruction": "人类指令(必填)",
        "input": "人类输入(选填)",
        "output": "模型回答(必填)",
        "audios": [
            "音频路径(必填)"
        ]
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "audios": "audios"
        }
    }
}
```

## ShareGPT

针对不同任务, 数据集格式要求如下:

- 指令监督微调

- 偏好训练

- OpenAI格式

>
> 备注:
>
> - ShareGPT 格式中的 KTO 数据集[样例](./kto_en_demo.json)和多模态数据集[样例](./mllm_demo.json) 与 Alpaca 格式的类似.
> - 预训练数据集不支持 ShareGPT 格式.
>

### 指令监督微调数据集

样例数据集: [指令监督微调样例数据集](./glaive_toolcall_zh_demo.json)

相比 `alpaca` 格式的数据集, `sharegpt` 格式支持更多的角色种类, 例如 human、gpt、observation、function 等等. 它们构成一个对象列表呈现在 `conversations` 列中. 下面是 `sharegpt` 格式的一个例子:

```json
{
    "conversations": [
        {
            "from": "human",
            "value": "你好, 我出生于1990年5月15日. 你能告诉我我今天几岁了吗?"
        },
        {
            "from": "function_call",
            "value": "{'name': 'calculate_age', 'arguments': {'birthdate': '1990-05-15'}}"
        },
        {
            "from": "observation",
            "value": "{'age': 31}"
        },
        {
            "from": "gpt",
            "value": "根据我的计算, 你今天31岁了."
        }
    ],
    "tools": "[{'name': 'calculate_age', 'description': '根据出生日期计算年龄', 'parameters': {'type': 'object', 'properties': {'birthdate': {'type': 'string', 'description': '出生日期以YYYY-MM-DD格式表示'}}, 'required': ['birthdate']}}]"
}
```

注意其中 human 和 observation 必须出现在奇数位置, gpt 和 function 必须出现在偶数位置.

```json
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "人类指令"
            },
            {
                "from": "function_call",
                "value": "工具参数"
            },
            {
                "from": "observation",
                "value": "工具结果"
            },
            {
                "from": "gpt",
                "value": "模型回答"
            }
        ],
        "system": "系统提示词(选填)",
        "tools": "工具描述(选填)"
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools"
        }
    }
}
```

### 偏好数据集

样例数据集: [偏好数据样例数据集](./dpo_zh_demo.json)

Sharegpt 格式的偏好数据集同样需要在 `chosen` 列中提供更优的消息, 并在 `rejected` 列中提供更差的消息. 下面是一个例子:

```json
{
    "conversations": [
        {
            "from": "human",
            "value": "老虎会吃哪一个, 草还是树叶?"
        }
    ],
    "chosen": {
        "from": "gpt",
        "value": "我很高兴能帮助你解答问题！不过, 我必须指出, 这个问题本身可能没有意义. 老虎是食肉动物, 通常不吃草或树叶. 它们的饮食主要包括大型有蹄动物、小型哺乳动物和鸟类. \n\n如果你提出这个问题是出于好奇或有特定原因, 我很乐意帮助你找到更多关于老虎饮食习惯的信息, 或者建议资源让你更多了解这些壮丽的生物. 请告诉我如何进一步协助你！"
    },
    "rejected": {
        "from": "gpt",
        "value": "一只老虎是食肉动物, 这意味着它的饮食主要是肉类. 老虎主要猎捕和吃其他动物, 如鹿和野猪. 它们不会选择吃草或树叶, 因为这些不是它们自然饮食的一部分. "
    }
}
```

其格式为:

```json
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "人类指令"
            },
            {
                "from": "gpt",
                "value": "模型回答"
            },
            {
                "from": "human",
                "value": "人类指令"
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": "优质回答"
        },
        "rejected": {
            "from": "gpt",
            "value": "劣质回答"
        }
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "formatting": "sharegpt",
        "ranking": true,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
}
```

### OpenAI 格式

OpenAI 格式仅仅是 `sharegpt` 格式的一种特殊情况, 其中第一条消息可能是系统提示词.

```json
[
    {
        "messages": [
            {
                "role": "system",
                "content": "系统提示词(选填)"
            },
            {
                "role": "user",
                "content": "人类指令"
            },
            {
                "role": "assistant",
                "content": "模型回答"
            }
        ]
    }
]
```

对于上述格式的数据, `dataset_info.json` 中的 **数据集描述** 应为:

```json
{
    "数据集名称": {
        "file_name": "data.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
}
```

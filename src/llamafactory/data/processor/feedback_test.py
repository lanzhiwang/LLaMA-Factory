import torch
from typing import Optional, Any
from collections import defaultdict
from llamafactory.data.processor import FeedbackDatasetProcessor

# --- 模拟 LLaMA-Factory 内部组件 ---
IGNORE_INDEX = -100


class MockTemplate:
    def __init__(self):
        self.mm_plugin = type(
            "MockPlugin",
            (),
            {
                "process_messages": lambda *args: args[0],
                "process_token_ids": lambda *args: (args[0], None),
            },
        )()
        self.efficient_eos = True

    def encode_oneturn(self, tokenizer, messages, system, tools):
        # 模拟 Token 编码: Prompt 长度固定为 5, Response 固定为 3
        return [1, 2, 3, 4, 5], [6, 7, 8]


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99

    def decode(self, ids, **kwargs):
        return f"Tokens({ids})"


# 模拟辅助函数
def infer_seqlen(source_len, target_len, cutoff):
    return source_len, target_len


# --- 构造示例数据 ---

# KTO 数据格式: _response 包含两个字典.
# 如果 index 0 有内容, 代表 Desirable (满意); 如果 index 1 有内容, 代表 Undesirable (不满意).
examples = {
    "_prompt": [
        # 样本 1
        [{"role": "user", "content": "你好, 请自我介绍"}],
        [{"role": "user", "content": "写一首关于大海的诗"}],  # 样本 2
    ],
    "_response": [
        [
            {"role": "assistant", "content": "我是小助手"},
            {"role": "assistant", "content": ""},
        ],  # 满意
        [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "大海很大"},
        ],  # 不满意
    ],
    "_system": [None, None],
    "_tools": [None, None],
    "_images": [None, None],
    "_videos": [None, None],
    "_audios": [None, None],
}

# --- 初始化并调用 ---

# 假设的基础类属性注入
processor = FeedbackDatasetProcessor(
    template=MockTemplate(),
    tokenizer=MockTokenizer(),
    processor=None,
    data_args=type("Args", (), {"cutoff_len": 128})()
)



# 执行预处理
output = processor.preprocess_dataset(examples)

# --- 结果演示 ---

print("=== KTO 预处理结果展示 ===")
for i in range(len(examples["_prompt"])):
    print(f"\n样本 {i+1}:")
    print(f"  满意标记 (kto_tags): {output['kto_tags'][i]}")
    print(f"  训练输入 (input_ids): {output['input_ids'][i]}")
    print(f"  训练标签 (labels):    {output['labels'][i]}")
    print(f"  KL参考输入 (kl_ids):  {output['kl_input_ids'][i]}")

# 演示 print_data_example 功能
print("\n=== 可视化示例 ===")
processor.print_data_example(
    {"input_ids": output["input_ids"][0], "labels": output["labels"][0]}
)

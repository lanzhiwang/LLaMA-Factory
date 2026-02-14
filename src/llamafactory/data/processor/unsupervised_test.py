from enum import Enum
from llamafactory.data.processor.unsupervised import UnsupervisedDatasetProcessor

# --- 模拟 LLaMA-Factory 内部类 ---
# class Role(Enum):
#     USER = "user"
#     ASSISTANT = "assistant"


# def infer_seqlen(source_len, target_len, cutoff):
#     # 模拟简单的截断逻辑
#     return min(source_len, cutoff), min(target_len, cutoff)


class MockPlugin:
    def process_messages(self, m, *args):
        return m

    def process_token_ids(self, ids, *args):
        return ids, None


class MockTemplate:
    def __init__(self):
        self.mm_plugin = MockPlugin()
        self.efficient_eos = True

    def encode_oneturn(self, tokenizer, messages, system, tools):
        # 简单模拟: 将每个字符的 ASCII 码作为 ID
        content = "".join([m["content"] for m in messages])
        ids = [ord(c) for c in content]
        return ids, ids


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join([chr(i) for i in ids if i != 0])


# --- 构造实际示例数据 ---
# 模拟从 JSON 文件加载后的原始数据格式
example_batch = {
    "_prompt": [
        [{"role": "user", "content": "你好, 你是谁?"}],
        [{"role": "user", "content": "1+1等于几?"}],
    ],
    "_response": [
        [{"role": "assistant", "content": "我是 LLaMA-Factory 助手."}],
        [{"role": "assistant", "content": "等于2."}],
    ],
    "_system": [None, None],
    "_tools": [None, None],
    "_images": [None, None],
    "_videos": [None, None],
    "_audios": [None, None],
}

# --- 执行演示 ---
# 1. 模拟组件初始化
processor = UnsupervisedDatasetProcessor(
    template=MockTemplate(),
    tokenizer=MockTokenizer(),
    processor=None,
    data_args=type("Args", (), {"cutoff_len": 50})(),
)
# processor.template = MockTemplate()
# processor.tokenizer = MockTokenizer()
# processor.data_args = type('Args', (), {'cutoff_len': 50})()
# processor.processor = None

# 2. 运行预处理
processed_data = processor.preprocess_dataset(example_batch)
"""
print(processed_data)
{
    "_prompt": [
        [{"role": "user", "content": "你好, 你是谁?"}],
        [{"role": "user", "content": "1+1等于几?"}],
    ],
    "_response": [
        [{"role": "assistant", "content": "我是 LLaMA-Factory 助手."}],
        [{"role": "assistant", "content": "等于2."}],
    ],
    "_system": [None, None],
    "_tools": [None, None],
    "_images": [None, None],
    "_videos": [None, None],
    "_audios": [None, None],
}
"""

# 3. 打印结果
print("=== 预处理后的 Tensor 数据 (第一个样本) ===")
print(f"Input IDs: {processed_data['input_ids'][0]}")
print(f"Labels:    {processed_data['labels'][0]}")
print(f"Mask:      {processed_data['attention_mask'][0]}")

print("\n=== 可视化结果 ===")
processor.print_data_example(
    {"input_ids": processed_data["input_ids"][0], "labels": processed_data["labels"][0]}
)

"""
$ python src/llamafactory/data/processor/unsupervised_test.py
=== 预处理后的 Tensor 数据 (第一个样本) ===
Input IDs: [20320, 22909, 44, 32, 20320, 26159, 35841, 63, 25105, 26159, 32, 76, 76, 97, 77, 65, 45, 70, 97, 99, 116, 111, 114, 121, 32]
Labels:    [20320, 22909, 44, 32, 20320, 26159, 35841, 63, 25105, 26159, 32, 76, 76, 97, 77, 65, 45, 70, 97, 99, 116, 111, 114, 121, 32]
Mask:      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

=== 可视化结果 ===
input_ids:
[20320, 22909, 44, 32, 20320, 26159, 35841, 63, 25105, 26159, 32, 76, 76, 97, 77, 65, 45, 70, 97, 99, 116, 111, 114, 121, 32]
inputs:
你好, 你是谁?我是 LLaMA-Factory
label_ids:
[20320, 22909, 44, 32, 20320, 26159, 35841, 63, 25105, 26159, 32, 76, 76, 97, 77, 65, 45, 70, 97, 99, 116, 111, 114, 121, 32]
labels:
你好, 你是谁?我是 LLaMA-Factory
$
"""

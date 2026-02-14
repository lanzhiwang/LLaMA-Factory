import sys
import os
from typing import List, Tuple
from dataclasses import dataclass

# 确保能导入项目中的类 (根据你的目录结构调整)
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# 假设你已经在根目录执行, 这里直接导入或模拟
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

# --- 模拟环境准备 ---
# IGNORE_INDEX = -100


class MockPlugin:
    """修复点: 显式定义类方法, 正确处理 self 参数"""

    def process_messages(self, messages, images, videos, audios, processor):
        # 正确接收并返回第一个业务参数 messages
        return messages

    def process_token_ids(
        self, input_ids, labels, images, videos, audios, tokenizer, processor
    ):
        # 必须返回 (input_ids, labels) 元组
        return input_ids, labels


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text, **kwargs):
        # 简化处理: 将字符 ASCII 码作为 ID
        return [ord(c) % 100 for c in text]

    def decode(self, ids, **kwargs):
        return "".join([chr(i + 32) for i in ids if i > 0])


class MockTemplate:
    def __init__(self):
        self.efficient_eos = True
        self.mm_plugin = MockPlugin()  # 使用修复后的 Mock 类

    def encode_multiturn(
        self, tokenizer, messages, system, tools
    ) -> List[Tuple[List[int], List[int]]]:
        """模拟多轮对话编码"""
        res = []
        # 按照 (User, Assistant) 成对拆分
        for i in range(0, len(messages) - 1, 2):
            source_ids = tokenizer.encode(messages[i]["content"])
            target_ids = tokenizer.encode(messages[i + 1]["content"])
            res.append((source_ids, target_ids))
        return res


# 模拟辅助函数
# def infer_seqlen(source_len, target_len, cutoff_len):
#     # 模拟简单截断: 不截断, 原样返回
#     return source_len, target_len


# --- 构造示例数据 ---
example_data = {
    "_prompt": [
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "我是助手"},
            {"role": "user", "content": "循环怎么写"},
        ],
        [{"role": "user", "content": "介绍北京"}],
    ],
    "_response": [
        [{"role": "assistant", "content": "用for循环"}],
        [{"role": "assistant", "content": "北京是首都"}],
    ],
    "_system": ["助手", "导游"],
    "_tools": [None, None],
    "_images": [None, None],
    "_videos": [None, None],
    "_audios": [None, None],
}


# --- 运行测试 ---
# 注意: 确保此处 SupervisedDatasetProcessor 已被正确定义或导入
# 如果你在独立脚本运行, 需要把原有的 SupervisedDatasetProcessor 类定义贴在上面
def run_test():
    # 初始化数据参数
    @dataclass
    class DataArgs:
        cutoff_len: int = 128
        mask_history: bool = False
        train_on_prompt: bool = False

    processor = SupervisedDatasetProcessor(
        template=MockTemplate(),
        tokenizer=MockTokenizer(),
        processor=None,
        data_args=DataArgs(),
    )

    print("开始预处理...")
    try:
        processed_results = processor.preprocess_dataset(example_data)
        """
        print(processed_results)
        defaultdict(
            <class 'list'>,
            {
                'input_ids': [
                    [20, 9, 5, 59, 61, 63, 90, 15, 90, 40, 89, 92, 2, 11, 14, 90, 15, 2],
                    [71, 61, 71, 40, 71, 40, 59, 18, 17, 2]
                ],
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                ],
                'labels': [
                    [-100, -100, 5, 59, 61, 63, 2, -100, -100, -100, -100, 92, 2, 11, 14, 90, 15, 2],
                    [-100, -100, -100, -100, 71, 40, 59, 18, 17, 2]
                ],
                'images': [None, None],
                'videos': [None, None],
                'audios': [None, None]
            }
        )
        """

        print("\n=== SFT 预处理结果展示 ===")
        for i in range(len(processed_results["input_ids"])):
            print(f"\n样本 {i+1} 信息:")
            print(f"Input IDs (前10个): {processed_results['input_ids'][i][:10]}")
            # Labels 中对应 User 的部分应该是 -100 (IGNORE_INDEX)
            print(f"Labels (前10个):    {processed_results['labels'][i][:10]}")

            # 可视化校验
            processor.print_data_example(
                {
                    "input_ids": processed_results["input_ids"][i],
                    "labels": processed_results["labels"][i],
                }
            )
    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()

"""
$ python src/llamafactory/data/processor/supervised_test1.py
开始预处理...

=== SFT 预处理结果展示 ===

样本 1 信息:
Input IDs (前10个): [20, 9, 5, 59, 61, 63, 90, 15, 90, 40]
Labels (前10个):    [-100, -100, 5, 59, 61, 63, 2, -100, -100, -100]
input_ids:
[20, 9, 5, 59, 61, 63, 90, 15, 90, 40, 89, 92, 2, 11, 14, 90, 15, 2]
inputs:
4)%[]_z/zHy|"+.z/"
label_ids:
[-100, -100, 5, 59, 61, 63, 2, -100, -100, -100, -100, 92, 2, 11, 14, 90, 15, 2]
labels:
%[]_"|"+.z/"

样本 2 信息:
Input IDs (前10个): [71, 61, 71, 40, 71, 40, 59, 18, 17, 2]
Labels (前10个):    [-100, -100, -100, -100, 71, 40, 59, 18, 17, 2]
input_ids:
[71, 61, 71, 40, 71, 40, 59, 18, 17, 2]
inputs:
g]gHgH[21"
label_ids:
[-100, -100, -100, -100, 71, 40, 59, 18, 17, 2]
labels:
gH[21"
$
"""

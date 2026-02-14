# import sys
# import os
from dataclasses import dataclass
from typing import List, Tuple

# 1. 确保环境可以导入 llamafactory
# 假设你在项目根目录下运行
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor

# from llamafactory.extras.constants import IGNORE_INDEX

# --- 2. 编写规范的 Mock 类以修复 self 参数问题 ---


class MockPlugin:
    """模拟多模态插件, 显式定义方法以正确处理 self 参数"""

    def process_messages(self, messages, images, videos, audios, processor):
        # 简单原样返回消息列表
        return messages

    def process_token_ids(
        self, input_ids, labels, images, videos, audios, tokenizer, processor
    ):
        # 必须返回 (list, list) 格式
        return input_ids, labels


class MockTemplate:
    """模拟对话模板"""

    def __init__(self):
        self.mm_plugin = MockPlugin()
        self.efficient_eos = True

    def encode_multiturn(
        self, tokenizer, messages, system, tools
    ) -> List[Tuple[List[int], List[int]]]:
        """
        模拟多轮对话编码.
        将 messages 中的 User/Assistant 对转化为 ID 元组.
        """
        pairs = []
        # 注意: messages 应该是列表, 修复后此处不会报错
        for i in range(0, len(messages) - 1, 2):
            source_text = messages[i]["content"]
            target_text = messages[i + 1]["content"]
            # 将字符转为 ASCII 码作为模拟 ID
            pairs.append(([ord(c) for c in source_text], [ord(c) for c in target_text]))
        return pairs


class MockTokenizer:
    """模拟分词器"""

    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join([chr(i) for i in ids if i > 2])


# --- 3. 构造实际示例数据 ---

example_batch = {
    "_prompt": [
        [{"role": "user", "content": "Hi"}],  # 样本 1
        [{"role": "user", "content": "Hello"}],  # 样本 2
        [{"role": "user", "content": "Hey"}],  # 样本 3
    ],
    "_response": [
        [{"role": "assistant", "content": "A"}],  # 样本 1 回答
        [{"role": "assistant", "content": "B"}],  # 样本 2 回答
        [{"role": "assistant", "content": "C"}],  # 样本 3 回答
    ],
    "_system": [None] * 3,
    "_tools": [None] * 3,
    "_images": [None] * 3,
    "_videos": [None] * 3,
    "_audios": [None] * 3,
}


@dataclass
class Args:
    cutoff_len: int = 16
    mask_history: bool = False
    train_on_prompt: bool = False
    neat_packing: bool = True  # 开启精致打包模式


# --- 4. 运行演示逻辑 ---


def run_packed_test():
    # 初始化
    args = Args(cutoff_len=20)
    processor = PackedSupervisedDatasetProcessor(
        template=MockTemplate(),
        tokenizer=MockTokenizer(),
        processor=None,
        data_args=args,
    )

    # 注入依赖
    # processor.template = MockTemplate()
    # processor.tokenizer = MockTokenizer()
    # processor.processor = None
    # processor.data_args = args

    print(
        f"正在启动 PackedSupervisedDatasetProcessor 演示 (窗口大小: {args.cutoff_len})..."
    )

    try:
        # 执行预处理
        output = processor.preprocess_dataset(example_batch)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()
        return

    # 展示结果
    print("\n✅ 数据打包成功! ")
    print("=" * 60)

    # 观察第一个打包后的序列 (Pack 0)
    input_ids = output["input_ids"][0]
    labels = output["labels"][0]
    position_ids = output["position_ids"][0]
    attention_mask = output["attention_mask"][0]

    print(f"Input IDs:      {input_ids}")
    print(f"Labels:         {labels}")
    print(f"Position IDs:   {position_ids}")
    print(f"Attention Mask: {attention_mask}")

    print("\n[工程点解析]:")
    print("- 序列拼接: 多个样本被紧凑排列, 中间补齐了 pad_token_id (0)")
    print("- 位置重置: Position IDs 在每个子样本开始处归零 (例如: ..., 65, 0, 1...)")
    print("- 注意力隔离: 不同样本的 Mask ID 不同 (1, 2, 3...), 防止样本间互相干扰")


if __name__ == "__main__":
    run_packed_test()

"""
$ python src/llamafactory/data/processor/supervised_test2.py
正在启动 PackedSupervisedDatasetProcessor 演示 (窗口大小: 20)...

✅ 数据打包成功!
============================================================
Input IDs:      [72, 101, 108, 108, 111, 66, 2, 72, 101, 121, 67, 2, 72, 105, 65, 2, 0, 0, 0, 0, 0]
Labels:         [-100, -100, -100, -100, -100, 66, 2, -100, -100, -100, 67, 2, -100, -100, 65, 2, -100, -100, -100, -100, -100]
Position IDs:   [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 0, 0, 0, 0]
Attention Mask: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0]

[工程点解析]:
- 序列拼接: 多个样本被紧凑排列, 中间补齐了 pad_token_id (0)
- 位置重置: Position IDs 在每个子样本开始处归零 (例如: ..., 65, 0, 1...)
- 注意力隔离: 不同样本的 Mask ID 不同 (1, 2, 3...), 防止样本间互相干扰
$
"""

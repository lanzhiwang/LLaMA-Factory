# import sys
# import os
# from collections import defaultdict
# from typing import Optional, Any, Union

# 确保能找到 llamafactory 源码
# 假设你在项目根目录运行, 或者将路径指向 src
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from llamafactory.data.processor.pairwise import PairwiseDatasetProcessor
# from llamafactory.extras.constants import IGNORE_INDEX

# --- 1. 稳健的 Mock 类定义 ---


class MockPlugin:
    """模拟多模态插件, 确保参数接收正确"""

    def process_messages(self, messages, images, videos, audios, processor):
        # 正确返回消息列表, 忽略多模态处理
        return messages

    def process_token_ids(
        self, prompt_ids, response_ids, images, videos, audios, tokenizer, processor
    ):
        # 必须返回 (prompt_ids, response_ids) 元组
        return prompt_ids, response_ids


class MockTemplate:
    def __init__(self):
        self.mm_plugin = MockPlugin()
        self.efficient_eos = True

    def encode_oneturn(self, tokenizer, messages, system, tools):
        """
        模拟模板编码逻辑
        messages: 传入的完整对话列表 [{'role': 'user', ...}, {'role': 'assistant', ...}]
        """
        # 模拟 Prompt 编码后的 ID
        prompt_ids = [101, 102]

        # 模拟根据 Assistant 内容生成不同的 ID
        # 这样我们可以区分输出结果里的 Chosen 和 Rejected
        content = messages[-1]["content"]
        if "巴黎" in content or "Paris" in content:
            res_ids = [201, 202]  # Chosen 结果
        else:
            res_ids = [301, 302]  # Rejected 结果

        return prompt_ids, res_ids


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0

    def decode(self, ids, **kwargs):
        return f"DecodedText{ids}"


# --- 2. 构造测试数据 (符合 LLaMA-Factory 格式) ---

examples = {
    "_prompt": [[{"role": "user", "content": "法国的首都是哪里?"}]],
    "_response": [
        [
            {"role": "assistant", "content": "巴黎."},  # Chosen
            {"role": "assistant", "content": "伦敦."},  # Rejected
        ]
    ],
    "_system": [None],
    "_tools": [None],
    "_images": [None],
    "_videos": [None],
    "_audios": [None],
}

# --- 3. 初始化并注入依赖 ---


def run_test():
    # 实例化真实的处理器
    processor = PairwiseDatasetProcessor(
        template=MockTemplate(),
        tokenizer=MockTokenizer(),
        processor=None,
        data_args=type("Args", (), {"cutoff_len": 50, "mask_history": False})(),
    )

    # 手动注入 Mock 依赖
    # processor.template = MockTemplate()
    # processor.tokenizer = MockTokenizer()
    # processor.processor = None  # SFT 处理器通常不需要这个

    # 模拟数据参数
    # processor.data_args = type("Args", (), {"cutoff_len": 50, "mask_history": False})()

    print("开始预处理成对偏好数据集...")

    try:
        output = processor.preprocess_dataset(examples)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 4. 验证结果 ---
    print("\n✅ 数据预处理成功!")
    print("=" * 50)

    # 检查第一个样本
    print(f"Prompt 部分 (ID): {output['chosen_input_ids'][0][:2]}")
    print(f"Chosen 回答 (ID): {output['chosen_input_ids'][0][2:]}")
    print(f"Chosen 标签 (Label): {output['chosen_labels'][0]}")
    print("-" * 50)
    print(f"Rejected 回答 (ID): {output['rejected_input_ids'][0][2:]}")
    print(f"Rejected 标签 (Label): {output['rejected_labels'][0]}")

    # 可视化调试
    print("\n[ print_data_example 输出 ]")
    processor.print_data_example(
        {
            "chosen_input_ids": output["chosen_input_ids"][0],
            "chosen_labels": output["chosen_labels"][0],
            "rejected_input_ids": output["rejected_input_ids"][0],
            "rejected_labels": output["rejected_labels"][0],
        }
    )


if __name__ == "__main__":
    run_test()

"""
$ python src/llamafactory/data/processor/pairwise_test.py
开始预处理成对偏好数据集...

✅ 数据预处理成功!
==================================================
Prompt 部分 (ID): [101, 102]
Chosen 回答 (ID): [201, 202, 99]
Chosen 标签 (Label): [-100, -100, 201, 202, 99]
--------------------------------------------------
Rejected 回答 (ID): [301, 302, 99]
Rejected 标签 (Label): [-100, -100, 301, 302, 99]

[ print_data_example 输出 ]
chosen_input_ids:
[101, 102, 201, 202, 99]
chosen_inputs:
DecodedText[101, 102, 201, 202, 99]
chosen_label_ids:
[-100, -100, 201, 202, 99]
chosen_labels:
DecodedText[201, 202, 99]
rejected_input_ids:
[101, 102, 301, 302, 99]
rejected_inputs:
DecodedText[101, 102, 301, 302, 99]
rejected_label_ids:
[-100, -100, 301, 302, 99]
rejected_labels:
DecodedText[301, 302, 99]
$
"""

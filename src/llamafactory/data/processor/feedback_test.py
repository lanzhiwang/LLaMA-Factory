from llamafactory.data.processor.feedback import FeedbackDatasetProcessor

# --- 1. 稳健的 Mock 类定义 ---


class MockPlugin:
    """模拟多模态插件, 确保返回正确的元组格式"""

    def process_messages(self, messages, images, videos, audios, processor):
        return messages

    def process_token_ids(
        self, prompt_ids, response_ids, images, videos, audios, tokenizer, processor
    ):
        # 必须返回 (list, any) 的格式
        return prompt_ids, response_ids


class MockTemplate:
    def __init__(self):
        self.mm_plugin = MockPlugin()
        self.efficient_eos = True

    def encode_oneturn(self, tokenizer, messages, system, tools):
        # 模拟 Token 编码: 返回两个列表
        # 假设 Prompt 编码后是 [1, 2, 3], Response 是 [4, 5]
        return [1, 2, 3], [4, 5]


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0

    def decode(self, ids, **kwargs):
        return f"DecodedTokens{ids}"


# --- 2. 构造测试数据 ---

# KTO 格式数据: _response[0] 为满意内容, _response[1] 为不满意内容
examples = {
    "_prompt": [
        [{"role": "user", "content": "你好"}],
        [{"role": "user", "content": "天气"}],
    ],
    "_response": [
        [
            {"role": "assistant", "content": "你好呀"},
            {"role": "assistant", "content": ""},
        ],
        [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "不知道"},
        ],
    ],
    "_system": [None, None],
    "_tools": [None, None],
    "_images": [None, None],
    "_videos": [None, None],
    "_audios": [None, None],
}

# --- 3. 初始化处理器并手动注入依赖 ---


def run_test():
    # 实例化真实的处理器
    processor = FeedbackDatasetProcessor(
        template=MockTemplate(),
        tokenizer=MockTokenizer(),
        processor=None,
        data_args=type("Args", (), {"cutoff_len": 20})(),
    )

    # 注入 Mock 依赖
    # processor.template = MockTemplate()
    # processor.tokenizer = MockTokenizer()
    # processor.processor = None
    # 注入配置参数
    # processor.data_args = type('Args', (), {'cutoff_len': 20})()

    print("开始预处理数据集...")
    # 执行预处理
    try:
        output = processor.preprocess_dataset(examples)
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 4. 展示结果 ---
    print("\n" + "=" * 30)
    print("KTO 数据处理成功!")
    print("=" * 30)

    for i in range(len(output["kto_tags"])):
        print(f"\n[样本 {i}]")
        print(f"满意状态 (kto_tag): {output['kto_tags'][i]}")
        print(f"输入 IDs (input_ids): {output['input_ids'][i]}")
        print(f"标签 IDs (labels):    {output['labels'][i]}")
        print(f"KL参考 IDs (kl_ids):   {output['kl_input_ids'][i]}")

    print("\n[可视化输出验证]:")
    processor.print_data_example(
        {"input_ids": output["input_ids"][0], "labels": output["labels"][0]}
    )


if __name__ == "__main__":
    run_test()

"""
$ python src/llamafactory/data/processor/feedback_test.py
开始预处理数据集...

==============================
KTO 数据处理成功!
==============================

[样本 0]
满意状态 (kto_tag): True
输入 IDs (input_ids): [1, 2, 3, 4, 5, 99]
标签 IDs (labels):    [-100, -100, -100, 4, 5, 99]
KL参考 IDs (kl_ids):   [1, 2, 3, 4, 5, 99]

[样本 1]
满意状态 (kto_tag): False
输入 IDs (input_ids): [1, 2, 3, 4, 5, 99]
标签 IDs (labels):    [-100, -100, -100, 4, 5, 99]
KL参考 IDs (kl_ids):   [1, 2, 3, 4, 5, 99]

[可视化输出验证]:
input_ids:
[1, 2, 3, 4, 5, 99]
inputs:
DecodedTokens[1, 2, 3, 4, 5, 99]
label_ids:
[-100, -100, -100, 4, 5, 99]
labels:
DecodedTokens[4, 5, 99]
$
"""

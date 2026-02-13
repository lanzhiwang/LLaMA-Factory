from llamafactory.data.processor.pretrain import PretrainDatasetProcessor


# --- 模拟环境准备 ---
class MockTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.add_bos_token = True

    def __call__(self, texts, **kwargs):
        """
        print(texts)
        ['<s>Hello world </s>', '<s>LLM training is fun </s>', '<s>Data processing </s>']
        print(kwargs)
        {'add_special_tokens': False, 'truncation': True, 'max_length': 4}
        """
        # 简单模拟分词: 每个单词变成其长度的 ID 列表
        input_ids = [[len(word) for word in text.split()] for text in texts]
        return {"input_ids": input_ids}

    def decode(self, ids, **kwargs):
        return "-".join([str(i) for i in ids])


class MockArgs:
    def __init__(self, packing, cutoff_len):
        self.template = "default"
        self.packing = packing
        self.cutoff_len = cutoff_len


# --- 准备示例数据 ---
# 模拟 LLaMA-Factory 的原始数据格式
raw_examples = {
    "_prompt": [
        [{"content": "Hello world "}],  # 长度短于 cutoff
        [{"content": "LLM training is fun "}],  # 长度短于 cutoff
        [{"content": "Data processing "}],  # 长度短于 cutoff
    ]
}

# --- 演示功能 ---

# 场景 1: 不开启 Packing (截断模式)
print("=== 场景 1: 不开启 Packing (cutoff_len=4) ===")
args_no_pack = MockArgs(packing=False, cutoff_len=4)
processor_no_pack = PretrainDatasetProcessor(
    template=None, tokenizer=MockTokenizer(), processor=None, data_args=args_no_pack
)
res1 = processor_no_pack.preprocess_dataset(raw_examples)
for ids in res1["input_ids"]:
    print(f"ids1: {ids}")
    processor_no_pack.print_data_example({"input_ids": ids})

# 场景 2: 开启 Packing (打包模式)
print("\n=== 场景 2: 开启 Packing (cutoff_len=5) ===")
args_pack = MockArgs(packing=True, cutoff_len=5)
processor_pack = PretrainDatasetProcessor(
    template=None, tokenizer=MockTokenizer(), processor=None, data_args=args_pack
)
res2 = processor_pack.preprocess_dataset(raw_examples)
for ids in res2["input_ids"]:
    print(f"ids2: {ids}")
    processor_pack.print_data_example({"input_ids": ids})

"""
$ python src/llamafactory/data/processor/pretrain_test.py
=== 场景 1: 不开启 Packing (cutoff_len=4) ===
ids1: [8, 5, 4]
input_ids:
[8, 5, 4]
inputs:
8-5-4
ids1: [6, 8, 2, 3, 4]
input_ids:
[6, 8, 2, 3, 4]
inputs:
6-8-2-3-4
ids1: [7, 10, 4]
input_ids:
[7, 10, 4]
inputs:
7-10-4

=== 场景 2: 开启 Packing (cutoff_len=5) ===
ids2: [1, 5, 4, 3, 8]
input_ids:
[1, 5, 4, 3, 8]
inputs:
1-5-4-3-8
ids2: [1, 3, 4, 4, 10]
input_ids:
[1, 3, 4, 4, 10]
inputs:
1-3-4-4-10
$
"""

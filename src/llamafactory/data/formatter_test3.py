# from collections import namedtuple
from llamafactory.data.formatter import FunctionFormatter

# 模拟 FunctionCall 对象
# FunctionCall = namedtuple("FunctionCall", ["name", "arguments"])


# 模拟工具处理逻辑(例如: 将解析出的 JSON 转化为 Qwen 格式的输出)
# class MockToolUtils:
#     def function_formatter(self, functions):
#         # 假设我们将函数调用转化为这种格式: [CALL: name(args)]
#         calls = [f"[CALL: {f.name}({f.arguments})]" for f in functions]
#         return " ".join(calls)


# def get_tool_utils(format_name):
#     return MockToolUtils()


# 1. 实例化 FunctionFormatter
# 假设模型的 Assistant 模板是: Assistant: {{content}}<|end|>
formatter = FunctionFormatter(
    slots=["Assistant: ", "{{content}}", {"token": "<|end|>"}],
    tool_format="qwen",  # 对应上面的 MockToolUtils
)

# 2. 构造示例数据
# 模拟一个 Reasoning 模型的输出:
# 包含思考内容(被 <thought> 包裹)和 工具调用(JSON 格式)
raw_model_output = """
<thought>
用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 city='Beijing'.
</thought>
{
    "name": "get_weather",
    "arguments": {"city": "Beijing", "unit": "celsius"}
}
"""

# 3. 调用 apply
# 传入 content 和 识别思考过程的关键词
result_slots = formatter.apply(
    content=raw_model_output, thought_words=["<thought>", "</thought>"]
)

# 4. 打印演示结果
print("--- 填充后的槽位列表 (SLOTS) ---")
for i, slot in enumerate(result_slots):
    print(f"Slot {i}: {slot}")

"""
$ python src/llamafactory/data/formatter_test3.py
--- 填充后的槽位列表 (SLOTS) ---
Slot 0: Assistant:
Slot 1: <thought>
用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 city='Beijing'.
</thought><tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}
</tool_call>
Slot 2: {'token': '<|end|>'}
$
"""

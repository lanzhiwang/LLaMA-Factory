import json
from llamafactory.data.formatter import (
    EmptyFormatter,
    StringFormatter,
    ToolFormatter,
    FunctionFormatter,
)

"""
关于 SLOTS: 注意 apply 返回的是一个列表(Slots). 在 LLaMA-Factory 的后续流程中, 这些列表项会被逐个进行 Tokenize.
如果某项是 dict(如 {"token": "..."}), 它会被处理为特殊的不参与截断的 Token.

关于 FunctionFormatter 的正则逻辑: 它提供了极强的鲁棒性. 即使模型在 JSON 之外输出了一些杂乱的字符, 只要你设置了 tool_call_words(如 ["[TOOL_CALLS]", "[/TOOL_CALLS]"]), 它就能精准定位并提取 JSON 内容.

多轮对话支持: 在 Template 类(即调用这些 Formatter 的上层类)中, 我们会循环调用这些 apply 方法. 通过将对话的不同阶段分配给不同的 Formatter, 我们可以轻松适配任何复杂的模型协议.

希望这些示例能帮助你和你的用户更好地理解 LLaMA-Factory 的内部构造!
"""

# 示例 1: 使用 EmptyFormatter (处理静态标识)
# EmptyFormatter 用于处理那些不含变量的静态槽位, 例如模型的起始符或角色前缀.
# 定义: Llama-3 的起始标识
header_formatter = EmptyFormatter(
    slots=["<|begin_of_text|>", "<|start_header_id|>system<|end_header_id|>\n\n"]
)

# 使用: apply 不需要传入参数, 直接返回定义的槽位
print("--- EmptyFormatter 示例 ---")
print(dir(header_formatter))
print(header_formatter.slots)
print(header_formatter.tool_format)
slots = header_formatter.apply()
print(slots)
# 输出: ['<|begin_of_text|>', '<|start_header_id|>system<|end_header_id|>\n\n']
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# ./src/llamafactory/data/formatter.py EmptyFormatter __post_init__
# --- EmptyFormatter 示例 ---
# ['__abstractmethods__', '__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__post_init__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', 'apply', 'extract', 'slots', 'tool_format']
# ['<|begin_of_text|>', '<|start_header_id|>system<|end_header_id|>\n\n']
# None
# ['<|begin_of_text|>', '<|start_header_id|>system<|end_header_id|>\n\n']


# 示例 2: 使用 StringFormatter (处理用户/系统消息)
# 这是最常用的格式化器, 它寻找 {{content}} 占位符并替换为实际文本.
# 定义: User 消息模板
user_formatter = StringFormatter(
    slots=["User: {{content}}\nAssistant: ", {"key": "value"}, set([4, 5, 6, 7])]
)

# 使用: 传入 content 变量
print("\n--- StringFormatter 示例 ---")
slots = user_formatter.apply(content="今天天气怎么样？")
print(slots)
# 输出: ['User: 今天天气怎么样？\nAssistant: ', {'key': 'value'}, {4, 5, 6, 7}]
# ['User: 今天天气怎么样？\nAssistant: ', {'key': 'value'}, {4, 5, 6, 7}]


# 示例 3: 使用 ToolFormatter (注入工具定义)
# 当你需要让模型知道它有哪些工具可以使用时, 这个类会将 JSON 格式的工具列表转化为模型能读懂的描述.
# 定义: 工具定义模板
tool_list_formatter = ToolFormatter(tool_format="default")
"""
print(tool_list_formatter)
ToolFormatter(slots=[], tool_format='default')
"""

# 模拟从数据集读取的 tools JSON 字符串
tools_json = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称, 如: 北京"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "计算器",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "int", "description": "城市名称, 如: 北京"},
                        "b": {"type": "int", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]
)


print("\n--- ToolFormatter 示例 ---")
slots = tool_list_formatter.apply(content=tools_json)
print(slots)
# 输出: ['Available Tools: get_weather, calc']

# 示例 4: 使用 FunctionFormatter (高级: 处理思考过程与工具调用)
# 这是最复杂的类. 它能识别模型输出中的"思考(Thought)"部分, 并解析 JSON 格式的工具调用. 这在微调 Reasoning 模型(如 DeepSeek-R1) 或 Agent 模型时非常有用.
# 定义: Assistant 回复模板
# 假设模型格式是: Assistant: {{content}}
func_formatter = FunctionFormatter(
    slots=["Assistant: {{content}}"], tool_format="default"
)

# 场景: 模型先输出了思考过程, 然后输出了一个 JSON 格式的工具调用
raw_content = '<thought>用户想知道天气, 我需要调用 get_weather</thought>{"name": "get_weather", "arguments": {"city": "Beijing"}}'

print("\n--- FunctionFormatter 示例 ---")
# 我们告诉格式化器, 思考过程是被 <thought> 标签包围的
slots = func_formatter.apply(
    content=raw_content, thought_words=["<thought>", "</thought>"]
)

print(slots)
# 解析过程:
# 1. 提取 <thought>...</thought>
# 2. 将剩余部分解析为 JSON 并通过 tool_utils 重新格式化
# 输出: ['Assistant: <thought>用户想知道天气, 我需要调用 get_weather</thought> [CALL: get_weather({"city": "Beijing"})] ']

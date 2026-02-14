# from collections import namedtuple
import json
from llamafactory.data.formatter import ToolFormatter

# 定义模拟的 FunctionCall 对象
# FunctionCall = namedtuple("FunctionCall", ["name", "arguments"])

# 模拟一个针对 "Llama-3" 的工具处理器
# class Llama3ToolUtils:
#     def tool_formatter(self, tools):
#         # 假设 Llama-3 需要这种格式: Environment: [tool1, tool2]
#         names = [t['name'] for t in tools]
#         return f"You can use these tools: {', '.join(names)}."

#     def tool_extractor(self, content):
#         # 简单模拟提取逻辑
#         if "Call:" in content:
#             return [FunctionCall(name="get_weather", arguments='{"city": "Beijing"}')]
#         return content

# 模拟工厂函数
# def get_tool_utils(tool_format):
#     if tool_format == "llama3":
#         return Llama3ToolUtils()
#     return None


# 1. 初始化 ToolFormatter
# 指定使用 llama3 格式转换逻辑
tool_formatter = ToolFormatter(tool_format="llama3")

# 2. 构造示例数据(标准的 JSON Schema 工具定义)
tools_definition = json.dumps(
    [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]
)

# 3. 演示 apply: 将 JSON 转化为 Prompt 中的文本
print("--- [演示 apply] ---")
formatted_slots = tool_formatter.apply(content=tools_definition)
"""
print(formatted_slots)
['Cutting Knowledge Date: December 2023\nToday Date: 14 Feb 2026\n\nYou have access to the following functions. To call a function, please respond with JSON for a function call. Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.\n\n{\n    "type": "function",\n    "function": {\n        "name": "get_weather",\n        "description": "Get the current weather in a given location",\n        "parameters": {\n            "type": "object",\n            "properties": {\n                "location": {\n                    "type": "string"\n                }\n            }\n        }\n    }\n}\n\n']
"""
print(f"生成的 Prompt 片段: {formatted_slots[0]}")
# 输出: 生成的 Prompt 片段: You can use these tools: get_weather.

# 4. 演示 extract: 从模型生成的内容中解析工具调用
print("\n--- [演示 extract] ---")
model_response = "I will help you. Call: get_weather(city='Beijing')"
extracted_calls = tool_formatter.extract(model_response)
"""
print(extracted_calls)
I will help you. Call: get_weather(city='Beijing')
"""

if isinstance(extracted_calls, list):
    for call in extracted_calls:
        print(f"检测到工具请求: 函数名={call.name}, 参数={call.arguments}")
# 输出: 检测到工具请求: 函数名=get_weather, 参数={"city": "Beijing"}

"""
$ python src/llamafactory/data/formatter_test4.py
--- [演示 apply] ---
生成的 Prompt 片段: Cutting Knowledge Date: December 2023
Today Date: 14 Feb 2026

You have access to the following functions. To call a function, please respond with JSON for a function call. Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string"
                }
            }
        }
    }
}



--- [演示 extract] ---
$
"""

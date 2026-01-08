"""
在 LLM 的微调和应用中, 函数调用(Function Calling) 最大的痛点在于: 每个模型对工具的"描述格式"和"调用格式"都完全不同.
例如, Qwen 喜欢用 XML 包裹 JSON, Llama 3 喜欢纯 JSON, 而一些旧模型则依赖于 Action/Action Input 的文本格式.
为了解决这个"模型协议不一致"的问题, 我们设计了 ToolUtils 系列类. 以下是这些代码的使用示例以及深度架构解析.
"""

from llamafactory.data.tool_utils import get_tool_utils, FunctionCall

# 准备工作: 定义你的工具(JSON Schema)
# 在 LLM 生态中, 工具通常以 JSON Schema 格式定义. 我们以此为例:


# 假设我们要给模型一个"查询天气"的工具
sample_tools = [
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
    }
]

# 示例 1: 为系统提示词格式化工具 (tool_formatter)
# [解决的问题]: 在训练或推理开始前, 如何告诉模型"你可以使用哪些工具".
# [为什么要这么写]: 不同模型对工具定义的敏感度不同, 必须严格匹配它们预训练/微调时的 Prompt 格式.

# 1. 针对 Qwen2.5 格式
qwen_utils = get_tool_utils("qwen")
system_prompt = qwen_utils.tool_formatter(sample_tools)
print("--- Qwen 格式系统提示词 ---")
print(system_prompt)
# 输出会包含 <tools> 标签, 并将 JSON 结构按 Qwen 习惯排列
# --- Qwen 格式系统提示词 ---


# # Tools

# You may call one or more functions to assist with the user query.

# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type": "function", "function": {"name": "get_weather", "description": "获取指定城市的实时天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称, 如: 北京"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
# </tools>

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>


# 2. 针对 Llama3 格式
llama_utils = get_tool_utils("llama3")
llama_prompt = llama_utils.tool_formatter(sample_tools)
print("\n--- Llama3 格式系统提示词 ---")
print(llama_prompt)
# 输出会包含 Llama 官方要求的 "Cutting Knowledge Date" 等前缀
# --- Llama3 格式系统提示词 ---
# Cutting Knowledge Date: December 2023
# Today Date: 08 Jan 2026

# You have access to the following functions. To call a function, please respond with JSON for a function call. Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

# {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "获取指定城市的实时天气",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {
#                     "type": "string",
#                     "description": "城市名称, 如: 北京"
#                 },
#                 "unit": {
#                     "type": "string",
#                     "enum": [
#                         "celsius",
#                         "fahrenheit"
#                     ]
#                 }
#             },
#             "required": [
#                 "city"
#             ]
#         }
#     }
# }


# 示例 2: 构建模型生成的工具调用 (function_formatter)
# [解决的问题]: 在 SFT(有监督微调)阶段, 我们需要构造"正确答案"; 在 Agent 逻辑中, 我们需要将解析出的调用转回文本.
# [为什么要这么写]: 确保模型学习到的输出格式是其架构最容易解析的.


# 模拟一个解析出的函数调用对象
calls = [
    FunctionCall(name="get_weather", arguments='{"city": "上海", "unit": "celsius"}')
]

# 1. 格式化为 Default (ReAct) 风格
default_utils = get_tool_utils("default")
print("--- Default (Action/Input) 风格 ---")
print(default_utils.function_formatter(calls))
# 输出: Action: get_weather\nAction Input: {"city": "上海", "unit": "celsius"}
# --- Default (Action/Input) 风格 ---
# Action: get_weather
# Action Input: {"city": "上海", "unit": "celsius"}


# 2. 格式化为 MiniMax-M2 (XML) 风格
minimax_utils = get_tool_utils("minimax2")
print("\n--- MiniMax XML 风格 ---")
print(minimax_utils.function_formatter(calls))
# 输出: <invoke name="get_weather"><parameter name="city">上海</parameter>...</invoke>
# --- MiniMax XML 风格 ---
# None


# 示例 3: 从模型回复中提取工具调用 (tool_extractor)
# [解决的问题]: 当模型生成了一长串文字, 里面夹杂着工具调用时, 如何精准地把它们提取出来执行.
# [为什么要这么写]: 模型输出具有随机性, 提取器使用了复杂的正则表达式和 JSON 校验, 增加了工程上的鲁棒性.

# 模拟模型在推理时的原始输出
raw_response = """
我看了一下, 需要为您查询上海的天气.
<tool_call>
{"name": "get_weather", "arguments": {"city": "上海"}}
</tool_call>
"""

qwen_utils = get_tool_utils("qwen")
extracted_calls = qwen_utils.tool_extractor(raw_response)

if isinstance(extracted_calls, list):
    for call in extracted_calls:
        print(f"检测到工具调用: {call.name}")
        print(f"参数内容: {call.arguments}")
else:
    print(f"这只是普通对话: {extracted_calls}")


"""
高级研究员的架构深度解析:

关于 NamedTuple 的使用:
代码中 FunctionCall 使用了 NamedTuple 而不是简单的 dict. 这是因为在 LLM 这种高性能计算场景中, NamedTuple 更轻量、内存占用更小, 且具有不可变性, 能防止在复杂的数据增强管道(Data Pipeline)中数据被意外修改.

tool_extractor 的反向逻辑:
每个类都实现了 tool_extractor. 这是微调的关键, 因为在训练时, 我们要确保模型生成的格式不仅"好看", 还得"可被程序解析". 如果 tool_extractor 解析失败, 通常意味着微调数据有问题.

对多轮对话的适配:
在 LLaMA-Factory 的 Template 类中, 我们会循环调用这些 ToolUtils. 对于支持 并行工具调用(Parallel Tool Calls) 的模型(如 Qwen, Mistral), 我们的格式化器能够一次性处理多个 FunctionCall 对象, 从而显著提升 Agent 的执行效率.

希望这些示例能帮助你更好地掌握 LLaMA-Factory 的工具链逻辑! 如果你在微调特定的模型(如最新的 DeepSeek 或 GLM 变体)时遇到格式不匹配, 只需要按照这个抽象基类(ABC)实现一个新的子类并注册到 TOOLS 字典中即可.
"""

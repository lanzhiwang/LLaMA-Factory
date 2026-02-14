# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import override

from .data_utils import SLOTS
from .tool_utils import FunctionCall, get_tool_utils

"""
这段代码是 LLaMA-Factory 对话模板(Chat Template)系统的灵魂.
LLM 微调中最痛苦的逻辑之一就是:
不同模型(如 Llama-3, Qwen, DeepSeek, GLM)对"角色标记"、"工具调用(Function Calling)"和"思考过程(Thinking)"的字符串格式要求完全不同.

为了实现"一套代码微调所有模型", 我们设计了这套 Formatter 抽象层.
它解决的核心问题是: 将结构化的对话数据, 精准且无损地转换为不同模型特定的 Token 序列, 并支持复杂的工具调用逻辑.
"""

@dataclass
class Formatter(ABC):
    r"""
    对话格式化器的抽象基类.
    [设计动机]: LLM 的输入不是单一的字符串, 而是由多个"槽位(Slots)"组成的序列.
    [解决的问题]: 定义统一接口, 使得上层逻辑(如 Tokenizer 编码)无需关心不同模型复杂的拼接规则.
    slots 存储了模板片段(如 ["<|im_start|>user\n", "{{content}}", "<|im_end|>"]).
    """
    slots: SLOTS = field(default_factory=list)
    tool_format: str | None = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        r"""
        Forms a list of slots according to the inputs to encode.

        核心方法: 将实际内容填充进模板槽位.
        """
        ...

    def extract(self, content: str) -> str | list["FunctionCall"]:
        r"""Extract a list of tuples from the response message if using tools.

        Each tuple consists of function name and function arguments.

        从模型输出中提取工具调用.
        [解决的问题]: 不同模型输出工具调用的方式不同(有的用 JSON, 有的用特殊 XML 标签),
        该方法提供了一个统一的逆向解析入口.
        """
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    r"""
    静态格式化器. 用于处理不需要填充变量的静态标记.
    [解决的问题]: 处理如 <|begin_of_text|> 这种固定在序列开头的特殊 Token.
    """
    def __post_init__(self):
        # 实例化 EmptyFormatter 对象之后开始执行 __post_init__ 方法
        # 严格性检查: 确保 EmptyFormatter 不包含 {{name}} 占位符
        has_placeholder = False

        """
        print(filter(lambda s: isinstance(s, str), self.slots))
        <filter object at 0x7fb8bc9fe5c0>
        print(list(filter(lambda s: isinstance(s, str), self.slots)))
        ['user']
        """
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        # 直接返回原始槽位, 不进行任何替换
        return self.slots


@dataclass
class StringFormatter(Formatter):
    r"""
    标准的字符串变量替换格式化器.
    [解决的问题]: 将用户输入的 content 或 system message 插入到模板的指定位置.
    """
    def __post_init__(self):
        # 校验: StringFormatter 必须包含至少一个占位符, 否则应该用 EmptyFormatter
        has_placeholder = False

        """
        print(filter(lambda s: isinstance(s, str), self.slots))
        <filter object at 0x7f96c709ffd0>
        print(list(filter(lambda s: isinstance(s, str), self.slots)))
        ['user', '\n\n', '{{content}}']
        """
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        """
        print(kwargs)
        {'content': '请问如何学习 Python?'}
        """

        elements = []
        """
        print(elements)
        [{'token': '<|start_header_id|>'}]
        print(elements)
        [{'token': '<|start_header_id|>'}, 'user']
        print(elements)
        [{'token': '<|start_header_id|>'}, 'user', {'token': '<|end_header_id|>'}]
        print(elements)
        [{'token': '<|start_header_id|>'}, 'user', {'token': '<|end_header_id|>'}, '\n\n']
        print(elements)
        [{'token': '<|start_header_id|>'}, 'user', {'token': '<|end_header_id|>'}, '\n\n', '请问如何学习 Python?']
        print(elements)
        [{'token': '<|start_header_id|>'}, 'user', {'token': '<|end_header_id|>'}, '\n\n', '请问如何学习 Python?', {'token': '<|eot_id|>'}]
        """
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {value}")

                    # 将模板中的 {{name}} 替换为实际传入的值
                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                # 如果槽位是 dict 或 set, 代表它是特殊 Token(在 DataCollator 中会特殊处理)
                elements.append(slot)
            else:
                raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}.")

        return elements


@dataclass
class FunctionFormatter(StringFormatter):
    r"""
    针对模型输出(Assistant Response)中"工具调用"逻辑的专项格式化器.
    [为什么要这么写]: 这是目前 LLM 工程中最复杂的部分.
    [解决的问题]:
    1. 思考链支持: DeepSeek-R1 等模型会先输出 <thought>...</thought> 再输出工具调用.
    2. 格式转换: 将模型生成的原始 JSON 或文本提取出来, 并按照 target 模板要求的工具格式重排.
    """
    def __post_init__(self):
        super().__post_init__()
        # 动态加载工具集对应的处理器(如 OpenAI 格式、GLM 格式等)
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        """
        print(kwargs)
        {
            'content': '\n<thought>\n用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 city=\'Beijing\'.\n</thought>\n{\n    "name": "get_weather",\n    "arguments": {"city": "Beijing", "unit": "celsius"}\n}\n',
            'thought_words': ['<thought>', '</thought>']
        }
        """
        content: str = kwargs.pop("content")
        # 思考标记, 如 ["<thought>", "</thought>"]
        thought_words = kwargs.pop("thought_words", None)
        # 工具调用标记
        tool_call_words = kwargs.pop("tool_call_words", None)
        """
        print(content)

        <thought>
        用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 city='Beijing'.
        </thought>
        {
            "name": "get_weather",
            "arguments": {"city": "Beijing", "unit": "celsius"}
        }

        print(thought_words)
        ['<thought>', '</thought>']
        print(tool_call_words)
        None
        print(kwargs)
        {}
        """

        def _parse_functions(json_content: str) -> list["FunctionCall"]:
            """
            将模型生成的 JSON 字符串解析为结构化的 FunctionCall 对象

            print(json_content)
            {
                "name": "get_weather",
                "arguments": {"city": "Beijing", "unit": "celsius"}
            }
            """
            try:
                tool_calls = json.loads(json_content)
                # 兼容非并行调用格式
                if not isinstance(tool_calls, list):  # parallel function call
                    tool_calls = [tool_calls]
                """
                print(tool_calls)
                [{'name': 'get_weather', 'arguments': {'city': 'Beijing', 'unit': 'celsius'}}]
                """

                return [FunctionCall(tc["name"], json.dumps(tc["arguments"], ensure_ascii=False)) for tc in tool_calls]
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON format in function message: {str([content])}.")

        # 逻辑: 识别并分离模型输出中的 [思考部分] 和 [工具调用部分]
        tool_call_match = None
        if tool_call_words and len(tool_call_words) == 2:
            # 尝试正则匹配被工具调用标记包裹的内容
            tool_call_regex = re.compile(
                rf"{re.escape(tool_call_words[0])}(.*?){re.escape(tool_call_words[1])}", re.DOTALL
            )
            tool_call_match = re.search(tool_call_regex, content)

        if tool_call_match is None:
            # 如果没有显式的工具标记, 则检查是否有思考标记
            thought_match = None
            if thought_words and len(thought_words) == 2:
                regex = re.compile(rf"{re.escape(thought_words[0])}(.*?){re.escape(thought_words[1])}", re.DOTALL)
                thought_match = re.search(regex, content)

            """
            print(thought_match)
            <re.Match object; span=(1, 75), match="<thought>\n用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 >
            print(thought_match.group(0))
            <thought>
            用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 city='Beijing'.
            </thought>
            """
            if thought_match:
                # 移除思考部分, 剩下的视为 JSON
                json_part = content.replace(thought_match.group(0), "")
            else:
                json_part = content
            """
            print(json_part)
            {
                "name": "get_weather",
                "arguments": {"city": "Beijing", "unit": "celsius"}
            }
            """
            functions = _parse_functions(json_part)
            """
            print(functions)
            [FunctionCall(name='get_weather', arguments='{"city": "Beijing", "unit": "celsius"}')]
            """
            # 使用工具处理器将 FunctionCall 对象转化为目标模板格式的字符串
            function_str = self.tool_utils.function_formatter(functions)
            """
            print(function_str)
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}
            </tool_call>
            """
            if thought_match:
                # 重新拼回思考部分
                function_str = thought_match.group(0) + function_str
        else:
            # 如果匹配到了工具标记(如 [TOOL_CALLS]), 提取中间的 JSON 并重排格式
            thought_content = content.replace(tool_call_match.group(0), "")
            functions = _parse_functions(tool_call_match.group(1))
            function_str = self.tool_utils.function_formatter(functions)
            function_str = thought_content + function_str

        # 最后复用 StringFormatter 的逻辑, 将其填充进模板槽位(如添加 Assistant 前缀)
        """
        print(function_str)
        <thought>
        用户想知道北京的天气. 我应该调用 get_weather 工具, 参数是 city='Beijing'.
        </thought><tool_call>
        {"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}
        </tool_call>
        """
        return super().apply(content=function_str)


@dataclass
class ToolFormatter(Formatter):
    r"""
    针对"系统层工具定义"的格式化器.
    [解决的问题]: 将数据集中可用的工具列表(Tools Definition)转化为模型在 System Prompt 中可见的描述字符串.
    同时也负责从模型推理结果中通过 extract 方法反向提取工具调用.
    """
    def __post_init__(self):
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        """
        print(kwargs)
        {'content': '[{"name": "get_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}]'}
        """

        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            """
            print(tools)
            [{'name': 'get_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}}}]
            """
            # 调用 tool_utils 将工具 JSON 数组转化为特定的模板描述(如 Markdown 表格或特定 YAML)
            return [self.tool_utils.tool_formatter(tools) if len(tools) != 0 else ""]
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}.")  # flat string

    @override
    def extract(self, content: str) -> str | list["FunctionCall"]:
        # 反向操作: 当模型输出文字时, 调用此方法判断模型是否想调用工具
        return self.tool_utils.tool_extractor(content)

"""
资深专家总结(为什么要这样设计):

高度解耦: Formatter 只管"填空", tool_utils 只管"JSON转换", Template 只管"拼接顺序". 这解决了 LLM 领域层出不穷的新模型格式适配问题 - 适配新模型通常只需在配置文件中定义几个新的 slots.

鲁棒性: 代码中大量的 json.loads 校验和正则匹配是为了应对真实世界中 LLM 输出的不确定性(比如模型在 JSON 前后加了废话).

支持推理模型(Reasoning Models): 通过对 thought_words 的处理, LLaMA-Factory 能够完美支持类似 DeepSeek-R1 的微调, 确保"思考"部分不被错误地当作"工具参数"解析.

这套设计使得 LLaMA-Factory 能够以工业级的标准处理数千种不同的对话模板, 是本项目保持领先地位的关键技术之一.

"""

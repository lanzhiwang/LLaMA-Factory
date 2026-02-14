from llamafactory.data.template import (
    register_template,
    Llama2Template,
    ReasoningTemplate,
    TEMPLATES,
)
from llamafactory.data.formatter import (
    StringFormatter,
    FunctionFormatter,
    ToolFormatter,
    EmptyFormatter,
)
from llamafactory.data.mm_plugin import get_mm_plugin

register_template(
    name="qwen_debug",
    format_user=StringFormatter(
        slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(
        slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
    ),
    format_function=FunctionFormatter(
        slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
    ),
    format_observation=StringFormatter(
        slots=[
            "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        ]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)

register_template(
    name="gemma3_debug",
    format_user=StringFormatter(
        slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("gemma3", image_token="<image_soft_token>"),
    template_class=Llama2Template,
)

register_template(
    name="deepseekr1_debug",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=ReasoningTemplate,
)

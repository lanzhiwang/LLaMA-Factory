TEMPLATES = {
    "alpaca": Template(
        format_user=StringFormatter(
            slots=["### Instruction:\n{{content}}\n\n### Response:\n"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}, "\n\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}, "\n\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["### Instruction:\n{{content}}\n\n### Response:\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "aquila": Template(
        format_user=StringFormatter(
            slots=["Human: {{content}}###Assistant:"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}###"], tool_format=None),
        format_system=StringFormatter(
            slots=["System: {{content}}###"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}###"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["Human: {{content}}###Assistant:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        stop_words=["</s>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "atom": Template(
        format_user=StringFormatter(
            slots=[
                {"bos_token"},
                "Human: {{content}}\n",
                {"eos_token"},
                {"bos_token"},
                "Assistant:",
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}\n", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}\n", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                {"bos_token"},
                "Human: {{content}}\n",
                {"eos_token"},
                {"bos_token"},
                "Assistant:",
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "baichuan": Template(
        format_user=StringFormatter(
            slots=[
                {"token": "<reserved_102>"},
                "{{content}}",
                {"token": "<reserved_103>"},
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=[
                {"token": "<reserved_102>"},
                "{{content}}",
                {"token": "<reserved_103>"},
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "baichuan2": Template(
        format_user=StringFormatter(
            slots=["<reserved_106>{{content}}<reserved_107>"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=["<reserved_106>{{content}}<reserved_107>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "bailing": Template(
        format_user=StringFormatter(
            slots=["<role>HUMAN</role>{{content}}<role>ASSISTANT</role>"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<role>SYSTEM</role>{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=["<role>OBSERVATION</role>{{content}}<role>ASSISTANT</role>"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|endoftext|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "bailing_v2": Template(
        format_user=StringFormatter(
            slots=["<role>HUMAN</role>{{content}}<|role_end|><role>ASSISTANT</role>"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|role_end|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<role>SYSTEM</role>{{content}}<|role_end|>"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|role_end|>"], tool_format="ling"
        ),
        format_observation=StringFormatter(
            slots=[
                "<role>OBSERVATION</role>\n<tool_response>\n{{content}}\n</tool_response><|role_end|><role>ASSISTANT</role>"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="ling"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|endoftext|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "belle": Template(
        format_user=StringFormatter(
            slots=["Human: {{content}}\n\nBelle: "], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}, "\n\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}, "\n\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["Human: {{content}}\n\nBelle: "], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "bluelm": Template(
        format_user=StringFormatter(
            slots=[{"token": "[|Human|]:"}, "{{content}}", {"token": "[|AI|]:"}],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[{"token": "[|Human|]:"}, "{{content}}", {"token": "[|AI|]:"}],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "breeze": Template(
        format_user=StringFormatter(
            slots=["[INST] {{content}} [/INST] "], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=["[INST] {{content}} [/INST] "], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "chatglm2": Template(
        format_user=StringFormatter(
            slots=["[Round {{idx}}]\n\n问：{{content}}\n\n答："], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=["[Round {{idx}}]\n\n问：{{content}}\n\n答："], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(
            slots=[{"token": "[gMASK]"}, {"token": "sop"}], tool_format=None
        ),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "chatglm3": Template(
        format_user=StringFormatter(
            slots=[
                {"token": "<|user|>"},
                "\n",
                "{{content}}",
                {"token": "<|assistant|>"},
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(slots=["\n", "{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=[{"token": "<|system|>"}, "\n", "{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
        format_observation=StringFormatter(
            slots=[
                {"token": "<|observation|>"},
                "\n",
                "{{content}}",
                {"token": "<|assistant|>"},
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4"),
        format_prefix=EmptyFormatter(
            slots=[{"token": "[gMASK]"}, {"token": "sop"}], tool_format=None
        ),
        default_system="",
        stop_words=["<|user|>", "<|observation|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "chatml": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>", "<|im_start|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "chatml_de": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="Du bist ein freundlicher und hilfsbereiter KI-Assistent.",
        stop_words=["<|im_end|>", "<|im_start|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "codegeex2": Template(
        format_user=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(
            slots=[{"token": "[gMASK]"}, {"token": "sop"}], tool_format=None
        ),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "codegeex4": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>\n"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
        format_observation=StringFormatter(
            slots=["<|observation|>\n{{content}}<|assistant|>\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4"),
        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"], tool_format=None),
        default_system="你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。",
        stop_words=["<|user|>", "<|observation|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "cohere": Template(
        format_user=StringFormatter(
            slots=[
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "cpm": Template(
        format_user=StringFormatter(slots=["<用户>{{content}}<AI>"], tool_format=None),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<用户>{{content}}<AI>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "cpm3": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "cpm4": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "dbrx": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.\nYOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.\nYou assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).\n(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)\nThis is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "deepseek": Template(
        format_user=StringFormatter(
            slots=["User: {{content}}\n\nAssistant:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["User: {{content}}\n\nAssistant:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "deepseek3": Template(
        format_user=StringFormatter(
            slots=["<｜User｜>{{content}}<｜Assistant｜>"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<｜User｜>{{content}}<｜Assistant｜>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "deepseekr1": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<｜User｜>{{content}}<｜Assistant｜>"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<｜User｜>{{content}}<｜Assistant｜>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "deepseekcoder": Template(
        format_user=StringFormatter(
            slots=["### Instruction:\n{{content}}\n### Response:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["\n{{content}}\n<|EOT|>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["\n{{content}}\n<|EOT|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["### Instruction:\n{{content}}\n### Response:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "default": Template(
        format_user=StringFormatter(
            slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}, "\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["System: {{content}}", {"eos_token"}, "\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}, "\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "dots_ocr": Template(
        format_user=StringFormatter(
            slots=["<|user|>{{content}}<|endofuser|><|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|endofassistant|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|system|>{{content}}<|endofsystem|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|endofassistant|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>{{content}}<|endofuser|><|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|endofassistant|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2VLPlugin(
            image_token="<|imgpad|>",
            video_token="<|vidpad|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|img|>",
            vision_eos_token="<|endofimg|>",
        ),
    ),
    "empty": Template(
        format_user=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "ernie": Template(
        format_user=StringFormatter(
            slots=[
                "<|im_start|>user\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>tool\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="<global_setting>\nthink_mode=True\n</global_setting>",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "ernie_nothink": Template(
        format_user=StringFormatter(
            slots=["User: {{content}}\nAssistant: "], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end_of_sentence|>"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end_of_sentence|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["User: {{content}}\nAssistant: "], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"], tool_format=None),
        default_system="",
        stop_words=["<|end_of_sentence|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "ernie_vl": ReasoningTemplate(
        format_user=StringFormatter(slots=["User: {{content}}"], tool_format=None),
        format_assistant=StringFormatter(
            slots=["\nAssistant: {{content}}<|end_of_sentence|>"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["\nAssistant: {{content}}<|end_of_sentence|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["User: {{content}}"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|end_of_sentence|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=ErnieVLPlugin(
            image_token="<|IMAGE_PLACEHOLDER|>",
            video_token="<|VIDEO_PLACEHOLDER|>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "exaone": Template(
        format_user=StringFormatter(
            slots=["[|user|]{{content}}\n[|assistant|]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}, "\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["[|system|]{{content}}[|endofturn|]\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}, "\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["[|user|]{{content}}\n[|assistant|]"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "falcon": Template(
        format_user=StringFormatter(
            slots=["User: {{content}}\nFalcon:"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["User: {{content}}\nFalcon:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "falcon_h1": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>", "<|end_of_text|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "fewshot": Template(
        format_user=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_assistant=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}\n\n"], tool_format="default"
        ),
        format_observation=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "gemma": Llama2Template(
        format_user=StringFormatter(
            slots=[
                "<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<end_of_turn>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "gemma2": Llama2Template(
        format_user=StringFormatter(
            slots=[
                "<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<eos>", "<end_of_turn>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "gemma3": Llama2Template(
        format_user=StringFormatter(
            slots=[
                "<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<end_of_turn>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Gemma3Plugin(
            image_token="<image_soft_token>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "gemma3n": Llama2Template(
        format_user=StringFormatter(
            slots=[
                "<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<end_of_turn>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Gemma3nPlugin(
            image_token="<image_soft_token>",
            video_token=None,
            audio_token="<audio_soft_token>",
            expand_mm_tokens=True,
        ),
    ),
    "glm4": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["\n{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
        format_observation=StringFormatter(
            slots=["<|observation|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4"),
        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"], tool_format=None),
        default_system="",
        stop_words=["<|user|>", "<|observation|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "glm4_moe": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["\n{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}"], tool_format="glm4_moe"
        ),
        format_observation=StringFormatter(
            slots=["<|observation|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4_moe"),
        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"], tool_format=None),
        default_system="",
        stop_words=["<|user|>", "<|observation|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "glm4v": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["\n{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
        format_observation=StringFormatter(
            slots=["<|observation|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4"),
        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"], tool_format=None),
        default_system="",
        stop_words=["<|user|>", "<|observation|>", "</answer>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=GLM4VPlugin(
            image_token="<|image|>",
            video_token="<|video|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "glm4_5v": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["\n{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}"], tool_format="glm4_moe"
        ),
        format_observation=StringFormatter(
            slots=["<|observation|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4_moe"),
        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"], tool_format=None),
        default_system="",
        stop_words=["<|user|>", "<|observation|>", "</answer>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=GLM4VPlugin(
            image_token="<|image|>",
            video_token="<|video|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "glmz1": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["\n{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
        format_observation=StringFormatter(
            slots=["<|observation|>\n{{content}}<|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="glm4"),
        format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"], tool_format=None),
        default_system="",
        stop_words=["<|user|>", "<|observation|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "gpt_oss": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|start|>user<|message|>{{content}}<|end|><|start|>assistant"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|start|>system<|message|>{{content}}<|end|>"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|start|>user<|message|>{{content}}<|end|><|start|>assistant"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are ChatGPT, a large language model trained by OpenAI.",
        stop_words=[],
        thought_words=(
            "<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        ),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "granite3": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_of_role|>user<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end_of_text|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|start_of_role|>system<|end_of_role|>{{content}}<|end_of_text|>\n"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end_of_text|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_of_role|>user<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "granite3_vision": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}\n<|assistant|>\n"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>\n{{content}}\n<|assistant|>\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "granite4": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_of_role|>user<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end_of_text|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|start_of_role|>system<|end_of_role|>{{content}}<|end_of_text|>\n"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end_of_text|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_of_role|>tool<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are Granite, developed by IBM. You are a helpful AI assistant.",
        stop_words=["<|end_of_text|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "index": Template(
        format_user=StringFormatter(
            slots=["reserved_0{{content}}reserved_1"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(slots=["<unk>{{content}}"], tool_format=None),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=["reserved_0{{content}}reserved_1"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "hunyuan": Template(
        format_user=StringFormatter(slots=["{{content}}<|extra_0|>"], tool_format=None),
        format_assistant=StringFormatter(
            slots=["{{content}}<|eos|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["{{content}}<|extra_4|>"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|eos|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["{{content}}<|extra_0|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=["<|startoftext|>"], tool_format=None),
        default_system="",
        stop_words=["<|eos|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "intern": Template(
        format_user=StringFormatter(
            slots=["<|User|>:{{content}}\n<|Bot|>:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<eoa>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|System|>:{{content}}\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<eoa>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|User|>:{{content}}\n<|Bot|>:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
        stop_words=["<eoa>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "intern2": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "intern_vl": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=InternVLPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "intern_s1": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=InternVLPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "keye_vl": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2VLPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "kimi_vl": ReasoningTemplate(
        format_user=StringFormatter(
            slots=[
                "<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant",
        stop_words=["<|im_end|>"],
        thought_words=("◁think▷", "◁/think▷"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=KimiVLPlugin(
            image_token="<|media_pad|>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llama2": Llama2Template(
        format_user=StringFormatter(
            slots=[{"bos_token"}, "[INST] {{content}} [/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[{"bos_token"}, "[INST] {{content}} [/INST]"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "llama2_zh": Llama2Template(
        format_user=StringFormatter(
            slots=[{"bos_token"}, "[INST] {{content}} [/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[{"bos_token"}, "[INST] {{content}} [/INST]"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "llama3": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format="llama3"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "llama4": Template(
        format_user=StringFormatter(
            slots=[
                "<|header_start|>user<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|eot|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|header_start|>system<|header_end|>\n\n{{content}}<|eot|>"],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|eot|>"], tool_format="llama3"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|header_start|>ipython<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|eot|>", "<|eom|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Llama4Plugin(
            image_token="<|image|>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "mllama": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format="llama3"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=MllamaPlugin(
            image_token="<|image|>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "moonlight": Template(
        format_user=StringFormatter(
            slots=[
                "<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant provided by Moonshot-AI.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "llava": Template(
        format_user=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next": Template(
        format_user=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_llama3": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format="llama3"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_mistral": Llama2Template(
        format_user=StringFormatter(
            slots=["[INST] {{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=[" {{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_qwen": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_yi": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_video": Template(
        format_user=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextVideoPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_video_mistral": Llama2Template(
        format_user=StringFormatter(
            slots=["[INST] {{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=[" {{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextVideoPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "llava_next_video_yi": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaNextVideoPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "marco": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.\n## 重要！！！！！\n当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。\n<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。\n",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "mimo": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "mimo_v2": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are MiMo, a helpful AI assistant engineered by Xiaomi.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>", "</think>"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "mimo_vl": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are MiMo, an AI assistant developed by Xiaomi.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2VLPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "minicpm_v": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=MiniCPMVPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "minicpm_o": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant. You can accept audio and text input and output voice and text.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=MiniCPMVPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token="<audio>",
            expand_mm_tokens=True,
        ),
    ),
    "minimax1": Template(
        format_user=StringFormatter(
            slots=[
                "<beginning_of_sentence>user name=user\n{{content}}<end_of_sentence>\n<beginning_of_sentence>ai name=assistant\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_sentence>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<beginning_of_sentence>system ai_setting=assistant\n{{content}}<end_of_sentence>\n"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<end_of_sentence>\n"], tool_format="minimax1"
        ),
        format_observation=StringFormatter(
            slots=[
                "<beginning_of_sentence>tool name=tools\n{{content}}<end_of_sentence>\n<beginning_of_sentence>ai name=assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="minimax1"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<end_of_sentence>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "minimax2": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["]~b]user\n{{content}}[e~[\n]~b]ai\n"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}[e~[\n"], tool_format=None),
        format_system=StringFormatter(
            slots=["]~!b[]~b]system\n{{content}}[e~[\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}[e~[\n"], tool_format="minimax2"
        ),
        format_observation=StringFormatter(
            slots=["]~b]tool\n<response>{{content}}</response>[e~[\n]~b]ai\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="minimax2"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant. Your name is MiniMax-M2.1 and is built by MiniMax.",
        stop_words=["[e~["],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "ministral": Llama2Template(
        format_user=StringFormatter(
            slots=["[INST]{{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "mistral": Llama2Template(
        format_user=StringFormatter(
            slots=["[INST] {{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=[" {{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "mistral_small": Template(
        format_user=StringFormatter(
            slots=["[INST]{{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["[SYSTEM_PROMPT]{{content}}[/SYSTEM_PROMPT]"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=PixtralPlugin(
            image_token="[IMG]",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "ministral3": Llama2Template(
        format_user=StringFormatter(
            slots=["[INST]{{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=PixtralPlugin(
            image_token="[IMG]",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "olmo": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>\n"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>\n{{content}}<|assistant|>\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"eos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "openchat": Template(
        format_user=StringFormatter(
            slots=[
                "GPT4 Correct User: {{content}}",
                {"eos_token"},
                "GPT4 Correct Assistant:",
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "GPT4 Correct User: {{content}}",
                {"eos_token"},
                "GPT4 Correct Assistant:",
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "openchat-3.6": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<|eot_id|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "opencoder": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are OpenCoder, created by OpenCoder Team.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "orion": Template(
        format_user=StringFormatter(
            slots=["Human: {{content}}\n\nAssistant: ", {"eos_token"}], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["Human: {{content}}\n\nAssistant: ", {"eos_token"}], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "paligemma": Llama2Template(
        format_user=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=PaliGemmaPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "paligemma_chat": Llama2Template(
        format_user=StringFormatter(
            slots=[
                "<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<end_of_turn>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=["<end_of_turn>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=PaliGemmaPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "phi": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}<|end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "phi_small": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}<|end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[{"<|endoftext|>"}], tool_format=None),
        default_system="",
        stop_words=["<|end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "phi4": Template(
        format_user=StringFormatter(
            slots=[
                "<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system<|im_sep|>{{content}}<|im_end|>"],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "pixtral": Llama2Template(
        format_user=StringFormatter(
            slots=["[INST]{{content}}[/INST]"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"
        ),
        format_observation=StringFormatter(
            slots=['[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]'],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="mistral"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=PixtralPlugin(
            image_token="[IMG]",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "qwen": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "qwen3": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "qwen3_nothink": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "qwen2_audio": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2AudioPlugin(
            image_token=None,
            video_token=None,
            audio_token="<|AUDIO|>",
            expand_mm_tokens=True,
        ),
    ),
    "qwen2_omni": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2OmniPlugin(
            image_token="<|IMAGE|>",
            video_token="<|VIDEO|>",
            audio_token="<|AUDIO|>",
            expand_mm_tokens=True,
            vision_bos_token="<|vision_bos|>",
            vision_eos_token="<|vision_eos|>",
            audio_bos_token="<|audio_bos|>",
            audio_eos_token="<|audio_eos|>",
        ),
    ),
    "qwen3_omni": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2OmniPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token="<|audio_pad|>",
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
            audio_bos_token="<|audio_start|>",
            audio_eos_token="<|audio_end|>",
        ),
    ),
    "qwen3_omni_nothink": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2OmniPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token="<|audio_pad|>",
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
            audio_bos_token="<|audio_start|>",
            audio_eos_token="<|audio_end|>",
        ),
    ),
    "qwen2_vl": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen2VLPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "qwen3_vl": ReasoningTemplate(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen3VLPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "qwen3_vl_nothink": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=Qwen3VLPlugin(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            audio_token=None,
            expand_mm_tokens=True,
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
        ),
    ),
    "sailor": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>question\n{{content}}<|im_end|>\n<|im_start|>answer\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>question\n{{content}}<|im_end|>\n<|im_start|>answer\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are an AI assistant named Sailor created by Sea AI Lab. Your answer should be friendly, unbiased, faithful, informative and detailed.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "seed_coder": Template(
        format_user=StringFormatter(
            slots=[
                {"bos_token"},
                "user\n{{content}}",
                {"eos_token"},
                {"bos_token"},
                "assistant\n",
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[{"bos_token"}, "system\n{{content}}", {"eos_token"}],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                {"bos_token"},
                "user\n{{content}}",
                {"eos_token"},
                {"bos_token"},
                "assistant\n",
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are an AI programming assistant, utilizing the Seed-Coder model, developed by ByteDance Seed, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "seed_oss": ReasoningTemplate(
        format_user=StringFormatter(
            slots=[
                {"bos_token"},
                "user\n{{content}}",
                {"eos_token"},
                {"bos_token"},
                "assistant\n",
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[{"bos_token"}, "system\n{{content}}", {"eos_token"}],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=[{"bos_token"}, "\n{{content}}", {"eos_token"}],
            tool_format="seed_oss",
        ),
        format_observation=StringFormatter(
            slots=[
                {"bos_token"},
                "user\n{{content}}",
                {"eos_token"},
                {"bos_token"},
                "assistant\n",
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="seed_oss"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<seed:think>", "</seed:think>"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "skywork_o1": Template(
        format_user=StringFormatter(
            slots=[
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=[
                "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"
            ],
            tool_format=None,
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|eot_id|>"], tool_format="llama3"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}], tool_format=None),
        default_system="You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. After completing your thoughts, you then provide a detailed explanation of the solution process in your response.",
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "smollm": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "smollm2": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful AI assistant named SmolLM, trained by Hugging Face.",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "solar": Template(
        format_user=StringFormatter(
            slots=["### User:\n{{content}}\n\n### Assistant:\n"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_system=StringFormatter(
            slots=["### System:\n{{content}}\n\n"], tool_format=None
        ),
        format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
        format_observation=StringFormatter(
            slots=["### User:\n{{content}}\n\n### Assistant:\n"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "starchat": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}<|end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "telechat": Template(
        format_user=StringFormatter(
            slots=["<_user>{{content}}<_bot>"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<_system>{{content}}<_end>"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<_user>{{content}}<_bot>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "telechat2": Template(
        format_user=StringFormatter(
            slots=["<_user>{{content}}<_bot>"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["<_system>{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<_user>{{content}}<_bot>"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "vicuna": Template(
        format_user=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=True,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "video_llava": Template(
        format_user=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["USER: {{content}} ASSISTANT:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=VideoLlavaPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "xuanyuan": Template(
        format_user=StringFormatter(
            slots=["Human: {{content}} Assistant:"], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["Human: {{content}} Assistant:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "xverse": Template(
        format_user=StringFormatter(
            slots=["Human: {{content}}\n\nAssistant: "], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["Human: {{content}}\n\nAssistant: "], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "yayi": Template(
        format_user=StringFormatter(
            slots=[
                {"token": "<|Human|>"},
                ":\n{{content}}\n\n",
                {"token": "<|YaYi|>"},
                ":",
            ],
            tool_format=None,
        ),
        format_assistant=StringFormatter(slots=["{{content}}\n\n"], tool_format=None),
        format_system=StringFormatter(
            slots=[{"token": "<|System|>"}, ":\n{{content}}\n\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}\n\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=[
                {"token": "<|Human|>"},
                ":\n{{content}}\n\n",
                {"token": "<|YaYi|>"},
                ":",
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are a helpful, respectful and honest assistant named YaYi developed by Beijing Wenge Technology Co.,Ltd. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        stop_words=["<|End|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "yi": Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "yi_vl": Template(
        format_user=StringFormatter(
            slots=["### Human: {{content}}\n### Assistant:"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["### Human: {{content}}\n### Assistant:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n",
        stop_words=["###"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=True,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=LlavaPlugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
            expand_mm_tokens=True,
        ),
    ),
    "yuan": Template(
        format_user=StringFormatter(
            slots=["{{content}}", {"token": "<sep>"}], tool_format=None
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<eod>\n"], tool_format=None
        ),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}<eod>\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["{{content}}", {"token": "<sep>"}], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<eod>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "zephyr": Template(
        format_user=StringFormatter(
            slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|system|>\n{{content}}", {"eos_token"}], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}", {"eos_token"}], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>\n"],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="You are Zephyr, a helpful assistant.",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
    "ziya": Template(
        format_user=StringFormatter(
            slots=["<human>:{{content}}\n<bot>:"], tool_format=None
        ),
        format_assistant=StringFormatter(slots=["{{content}}\n"], tool_format=None),
        format_system=StringFormatter(slots=["{{content}}"], tool_format=None),
        format_function=FunctionFormatter(
            slots=["{{content}}\n"], tool_format="default"
        ),
        format_observation=StringFormatter(
            slots=["<human>:{{content}}\n<bot>:"], tool_format=None
        ),
        format_tools=ToolFormatter(slots=[], tool_format="default"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    ),
}

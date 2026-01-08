"""
get_dataset_list 函数是 LLaMA-Factory 数据解耦架构的关键.
它解决了 LLM 微调中一个非常棘手的痛点: 如何从不同的来源(本地、HF Hub、ModelScope)以统一的接口加载格式各异(Alpaca, ShareGPT)的数据集.
通过该函数, 用户只需提供一个简单的字符串列表, 系统就能自动映射到复杂的 DatasetAttr 对象.

准备工作: 了解 dataset_info.json
get_dataset_list 默认会读取 dataset_dir 下的 dataset_info.json(由 DATA_CONFIG 常量定义). 这个文件描述了数据集的映射关系.
"""

import os
from llamafactory.data.parser import get_dataset_list

# 示例 1: 最常见的本地数据加载场景
# 这是绝大多数算法工程师在本地服务器上微调时的用法. 你需要指定本地数据目录, 并传入 dataset_info.json 中定义的 key.
# 假设本地目录 'data/' 下有一个 dataset_info.json, 内容包含 'alpaca_zh' 和 'identity'
# dataset_dir 是本地路径
dataset_dir = "./data"
dataset_names = ["alpaca_zh", "identity"]

# 获取属性列表
dataset_list = get_dataset_list(dataset_names, dataset_dir)

for attr in dataset_list:
    print(f"成功加载数据集: {attr.dataset_name}")
    print(f"数据来源类型: {attr.load_from}")  # 通常为 'file'
    print(f"提示词格式: {attr.formatting}")  # 例如 'alpaca'
    print(f"映射列名: Prompt->{attr.prompt}, Response->{attr.response}")


# 示例 2: 使用 "ONLINE" 模式(Hub 自动选择)
# 当你不想维护本地配置文件, 且希望直接从 Hugging Face 或 ModelScope 加载官方数据集时, 可以使用 "ONLINE" 关键字.
# 场景: 你身处国内环境, 希望优先从 ModelScope 加载
os.environ["USE_MODELSCOPE_HUB"] = "1"

# 在 ONLINE 模式下, dataset_names 直接被视为 Hub 上的 Repo 路径
dataset_names = ["hiyouga/alpaca-gpt4-zh"]
dataset_dir = "ONLINE"

dataset_list = get_dataset_list(dataset_names, dataset_dir)

# 即使没有本地 json, 系统也会自动推断 load_from 为 'ms_hub' 或 'hf_hub'
attr = dataset_list[0]
print(f"Hub 路径: {attr.dataset_name}")
print(f"自动识别来源: {attr.load_from}")


# 示例 3: 远程配置同步场景(REMOTE 模式)
# 在集群训练或大规模生产环境中, 我们通常会把数据集的配置文件(dataset_info.json)放在一个中心化的 Git Repo(如 Hugging Face Dataset Repo)中, 而不是每台服务器存一份.
# 使用 REMOTE: 前缀指向一个 HF 数据集仓库
# 系统会自动下载该仓库里的 dataset_info.json 并解析
dataset_dir = "REMOTE:hiyouga/LLaMA-Factory"
dataset_names = ["alpaca_en_demo"]

dataset_list = get_dataset_list(dataset_names, dataset_dir)

print(f"从远程配置解析到的本地文件名: {dataset_list[0].dataset_name}")


# 示例 4: 纯编程模式(字典注入)
# 如果你正在开发一个自动化流水线, 数据集的信息是动态生成的(例如从数据库读取), 你可以直接传入一个字典作为 dataset_dir.
# 动态构建配置字典, 无需物理 JSON 文件
dynamic_config = {
    "custom_sft_data": {
        "file_name": "my_data.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "sys_prompt"},
        "tags": {"role_tag": "from", "content_tag": "value"},
    }
}

dataset_names = ["custom_sft_data"]
# 直接传入字典
dataset_list = get_dataset_list(dataset_names, dataset_dir=dynamic_config)

attr = dataset_list[0]
print(f"解析后的 ShareGPT 角色标签: {attr.role_tag}")  # 输出: from


"""
高级研究员视角: 为什么要这么设计?

作为一名高级开发工程师, 我设计/维护这部分代码时考虑了以下几点:
多源透明化: 通过 load_from 的 Literal 定义, 将 hf_hub, ms_hub, om_hub 抽象化. 上层加载器(Loader)只需要看 attr.load_from 就能决定是用 load_dataset 还是 MsDataset.load.

Schema 映射兼容性: DatasetAttr 里的 set_attr 和 join 方法解决了字段名不统一的问题. 无论你的 JSON 里叫 instruction 还是 question, 解析后都统一映射到 attr.prompt, 这极大地简化了后续 Template 类的工作.

网络健壮性: get_dataset_list 内部的 try-except 逻辑确保了即使 dataset_dir 配置有误, 只要不请求特定数据集, 系统就不会崩溃. 这对于支持 Web UI 展示默认信息非常重要.

希望这些示例能帮助你和你的用户更好地理解 LLaMA-Factory 的数据引擎！如果有关于 columns 或 tags 映射的具体问题, 欢迎随时提问.
"""

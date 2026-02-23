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

from typing import TYPE_CHECKING

import torch
from transformers.utils import cached_file

from ...extras import logging
from ...extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def load_valuehead_params(path_or_repo_id: str, model_args: "ModelArguments") -> dict[str, torch.Tensor]:
    r"""
    Load value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.

    在 RLHF(基于人类反馈的强化学习) 流程中, 我们需要训练 奖励模型 (Reward Model) 或 PPO 阶段的价值模型 (Value Model). 这些模型在标准语言模型(CausalLM)的基础上增加了一个额外的线性层, 称为 Value Head(通常是一个输出维度为 1 的线性层).
    由于 transformers 库的原生类并不总是包含这个自定义层, 我们需要一套稳健的逻辑来单独处理这些权重的加载.

    从 Hugging Face Hub 或本地磁盘加载价值头 (Value Head) 参数.

    返回值: 包含键 `v_head.summary.weight` 和 `v_head.summary.bias` 的张量字典.

    高级研究员视角的架构深度解析:

    为什么不使用 model.load_state_dict 直接加载?
    ValueHead 往往是在 AutoModelForCausalLM 之外定义的. 在 LLaMA-Factory 的 load_model 流程中, 基础模型和价值头模型通常是异步加载或动态注入的(例如通过 AutoModelForCausalLMWithValueHead). 将参数加载独立出来, 可以灵活地将一个训练好的 奖励模型权重 注入到任何具有相同隐藏层维度的 基础模型 中.

    内存管理考量:
    注意函数中显式使用了 device="cpu" 和 map_location="cpu". 作为高级开发工程师, 我们必须防止在加载权重的瞬间出现显存爆满. 通过在 CPU 上完成参数字典的构建, 待后续模型实例就绪后, 再利用 model.load_state_dict 配合 to(device) 进行统一搬运.

    多格式探测的必要性:
    在开源社区, 由于模型分发格式的不统一, 一个稳健的微调项目必须具备"自适应探测"能力. 这种"先试 A, 不行再试 B"的模式, 极大降低了用户在使用过程中的环境排错成本.
    """

    # 1. 统一构造加载参数 (Resource Resolution)
    # [为什么要这么写]: 通过封装 path、cache_dir 和 token, 适配 Hugging Face 的生态系统.
    # [解决的问题]: 支持从本地路径直接读取, 也支持自动从远程仓库下载并管理本地缓存, 同时处理需要 Token 验证的闭源模型.
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}
    err_text = ""

    # 2. 优先尝试现代安全格式 (Modern & Secure Serialization)
    # [为什么要这么写]: 首先尝试加载 .safetensors 格式的文件(V_HEAD_SAFE_WEIGHTS_NAME).
    # [解决的问题]:
    #   a. 安全性: 传统的 pickle (.bin) 格式存在反序列化攻击风险, safetensors 只存储张量数据, 绝对安全.
    #   b. 性能: safetensors 支持内存映射 (Mmap), 在多进程加载大模型时能显著降低内存峰值并提升速度.
    try:
        from safetensors import safe_open

        # cached_file 是 HF 提供的工具, 能自动判断本地是否存在, 不存在则下载并返回本地绝对路径
        vhead_file = cached_file(filename=V_HEAD_SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            # 遍历所有的 key, 将张量读入内存
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        # 记录错误并继续尝试, 因为可能是格式不匹配
        err_text = str(err)

    # 3. 兼容性降级: 尝试旧版 PyTorch 格式 (Legacy Support)
    # [为什么要这么写]: 如果找不到 safetensors 文件, 则尝试加载传统的 .bin 或 .pt 文件(V_HEAD_WEIGHTS_NAME).
    # [解决的问题]: 确保向后兼容性, 使用户能够加载早期版本 LLaMA-Factory 或其他框架训练产出的旧版权重.
    try:
        vhead_file = cached_file(filename=V_HEAD_WEIGHTS_NAME, **kwargs)
        # weights_only=True 是 PyTorch 2.x 推荐的安全实践, 防止 pickle 注入攻击
        return torch.load(vhead_file, map_location="cpu", weights_only=True)
    except Exception as err:
        err_text = str(err)

    # 4. 容错逻辑与日志反馈 (Fault Tolerance)
    # [为什么要这么写]: 如果两种格式都加载失败, 不直接中断程序, 而是打印信息并返回 None.
    # [解决的问题]: 在 RLHF 流程中, 用户可能仅加载了基础模型(Base Model)而没有加载奖励模型.
    # 如果这不是一次"断点续训 (Resuming)", 那么没有价值头权重是正常的, 不应抛出致命异常.
    logger.info_rank0(f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}.")
    logger.info_rank0("Ignore the above message if you are not resuming the training of a value head model.")
    return None


def prepare_valuehead_model(model: "PreTrainedModel") -> None:
    """
    在强化学习(RLHF)或奖励模型(RM)微调中, 我们通常使用 trl 库的 AutoModelForCausalLMWithValueHead. 这个包装器在标准语言模型顶部挂载一个 ValueHead(价值头).
    然而, trl 的内部逻辑高度依赖于模型必须拥有一个名为 lm_head 的属性. 由于开源模型社区的命名规范极不统一, 这段代码的作用就是通过"动态属性对齐"和"保存逻辑修剪", 解决不同架构模型在强化学习阶段的兼容性问题.

    为强化学习(PPO)或奖励模型(RM)阶段准备模型, 确保价值头(ValueHead)能够正确挂载.

    [核心问题]:
    trl 库在初始化 ValueHead 模型时, 会尝试访问 `model.lm_head` 以获取隐藏层到词表的映射.
    但许多模型(如 LLaVA, ChatGLM, InternLM2)并不把输出层命名为 `lm_head`.
    如果不进行手动映射, 程序会报 `AttributeError`.

    高级研究员与资深工程师的架构点评:

    架构无关性 (Architecture Agnosticism):
    高级 Python 工程设计中, 这种方法被称为"猴子补丁(Monkey Patching)"的优雅应用. 我们不修改模型原本的 forward 逻辑, 而是通过设置属性别名, 让下游框架(如 trl)认为它正在处理一个标准的 Llama 架构模型. 这使得 LLaMA-Factory 能够支持数百种模型而不需要重写训练 loop.

    显存与存储优化:
    _keys_to_ignore_on_save 的处理展示了对工业级微调的考量. 在微调 70B 模型时, 一个线性层的权重可能就有几百 MB. 如果存在多个别名引用, 忽略掉这些副本对于保持 Checkpoint 的纯净和节省磁盘 I/O 至关重要.

    多模态前瞻性:
    对 llava 的处理说明该函数已经考虑到了 VLM(视觉语言模型) 的强化学习需求. 在 VLM 的 RLHF 中, 我们通常只针对文本部分的输出计算价值奖励, 这里的路径映射确保了视觉特征不会干扰到价值头的梯度计算.
    """

    # 1. 处理 LLaVA (视觉语言模型)
    if getattr(model.config, "model_type", None) == "llava":
        # [为什么要这么写]: LLaVA 模型是一个复合架构, 真正的文本部分嵌套在 `language_model` 属性中.
        # [解决的问题]: 建立一个名为 `lm_head` 的别名, 指向嵌套在内部的输出层, 使 trl 能够透明访问.
        setattr(model, "lm_head", model.language_model.get_output_embeddings())

        # [为什么要这么写]: 设置保存时忽略的 Key.
        # [解决的问题]: 防止"权重冗余保存". 因为 `lm_head` 现在只是一个引用(Alias),
        # 如果不忽略, 在保存模型状态字典(State Dict)时, 同一份权重会被存两份(原名和 lm_head),
        # 导致导出的 Checkpoint 文件白白增大, 且可能导致加载时的权重名冲突.
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    # 2. 处理 ChatGLM 系列
    if getattr(model.config, "model_type", None) == "chatglm":
        # [解决的问题]: ChatGLM 官方建模代码中, 输出层被命名为 `transformer.output_layer`.
        # 这里将其对齐到标准名称 `lm_head`.
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    # 3. 处理 InternLM2 系列
    if getattr(model.config, "model_type", None) == "internlm2":
        # [解决的问题]: InternLM2 将输出层命名为 `output`.
        # 这种不统一的命名是微调框架最需要通过工程手段消除的差异(Architecture Agnosticism).
        setattr(model, "lm_head", model.output)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

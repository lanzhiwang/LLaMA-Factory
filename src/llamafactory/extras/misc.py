# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's PEFT library.
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/peft_model.py
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

import gc
import os
import socket
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch
import torch.distributed as dist
import transformers.dynamic_module_utils
from huggingface_hub.utils import WeakFileLock
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.dynamic_module_utils import get_relative_imports
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.utils.versions import require_version

from . import logging


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())
except Exception:
    _is_bf16_available = False


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


class AverageMeter:
    r"""Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_version(requirement: str, mandatory: bool = False) -> None:
    r"""Optionally check the package version."""
    if is_env_enabled("DISABLE_VERSION_CHECK") and not mandatory:
        logger.warning_rank0_once("Version checking has been disabled, may lead to unexpected behaviors.")
        return

    if "gptmodel" in requirement or "autoawq" in requirement:
        pip_command = f"pip install {requirement} --no-build-isolation"
    else:
        pip_command = f"pip install {requirement}"

    if mandatory:
        hint = f"To fix: run `{pip_command}`."
    else:
        hint = f"To fix: run `{pip_command}` or set `DISABLE_VERSION_CHECK=1` to skip this check."

    require_version(requirement, hint)


def check_dependencies() -> None:
    r"""Check the version of the required packages."""
    check_version("transformers>=4.51.0,<=4.57.1")
    check_version("datasets>=2.16.0,<=4.0.0")
    check_version("accelerate>=1.3.0,<=1.11.0")
    check_version("peft>=0.14.0,<=0.17.1")
    check_version("trl>=0.18.0,<=0.24.0")


def calculate_tps(dataset: list[dict[str, Any]], metrics: dict[str, float], stage: Literal["sft", "rm"]) -> float:
    r"""Calculate effective tokens per second."""
    effective_token_num = 0
    for data in dataset:
        if stage == "sft":
            effective_token_num += len(data["input_ids"])
        elif stage == "rm":
            effective_token_num += len(data["chosen_input_ids"]) + len(data["rejected_input_ids"])

    result = effective_token_num * metrics["epoch"] / metrics["train_runtime"]
    return result / dist.get_world_size() if dist.is_initialized() else result


def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    r"""Return the number of trainable parameters and number of all parameters in the model."""
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_current_device() -> "torch.device":
    r"""Get the current available device."""
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.getenv("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.getenv("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_count() -> int:
    r"""Get the number of available devices."""
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_mps_available():
        return torch.mps.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_logits_processor() -> "LogitsProcessorList":
    r"""Get logits processor that removes NaN and Inf logits."""
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def get_current_memory() -> tuple[int, int]:
    r"""Get the available and total memory for the current device (in Bytes)."""
    if is_torch_xpu_available():
        return torch.xpu.mem_get_info()
    elif is_torch_npu_available():
        return torch.npu.mem_get_info()
    elif is_torch_mps_available():
        return torch.mps.current_allocated_memory(), torch.mps.recommended_max_memory()
    elif is_torch_cuda_available():
        return torch.cuda.mem_get_info()
    else:
        return 0, -1


def get_peak_memory() -> tuple[int, int]:
    r"""Get the peak memory usage (allocated, reserved) for the current device (in Bytes)."""
    if is_torch_xpu_available():
        return torch.xpu.max_memory_allocated(), torch.xpu.max_memory_reserved()
    elif is_torch_npu_available():
        return torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()
    elif is_torch_mps_available():
        return torch.mps.current_allocated_memory(), -1
    elif is_torch_cuda_available():
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    else:
        return 0, -1


def has_tokenized_data(path: "os.PathLike") -> bool:
    r"""Check if the path has a tokenized dataset."""
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def infer_optim_dtype(model_dtype: Optional["torch.dtype"]) -> "torch.dtype":
    r"""Infer the optimal dtype according to the model_dtype and device compatibility."""
    if _is_bf16_available and (model_dtype == torch.bfloat16 or model_dtype is None):
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def is_accelerator_available() -> bool:
    r"""Check if the accelerator is available."""
    return (
        is_torch_xpu_available() or is_torch_npu_available() or is_torch_mps_available() or is_torch_cuda_available()
    )


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""Check if the environment variable is enabled."""
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]


def numpify(inputs: Union["NDArray", "torch.Tensor"]) -> "NDArray":
    r"""Cast a torch tensor or a numpy array to a numpy array."""
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu()
        if inputs.dtype == torch.bfloat16:  # numpy does not support bfloat16 until 1.21.4
            inputs = inputs.to(torch.float32)

        inputs = inputs.numpy()

    return inputs


def skip_check_imports() -> None:
    r"""
    Avoid flash attention import error in custom model files.

    [为什么要这么写]:
    这是一种"猴子补丁"(Monkey Patch), 通过将 transformers 内部用于检查导入的函数
    `check_imports` 替换为只获取相对路径的 `get_relative_imports`.

    [要解决的问题]:
    1. 绕过远程代码(Remote Code)的依赖死锁:
       当加载支持 `trust_remote_code=True` 的自定义模型(如 Qwen, DeepSeek, Mixtral 等)时,
       transformers 会自动调用 `check_imports` 来检查模型代码文件(.py)中所有的 `import` 语句是否在本地环境存在.

    2. 解决 Flash Attention 导致的加载失败:
       很多现代模型在代码中硬编码了 `import flash_attn`. 如果用户的环境中没有安装 Flash Attention
       (例如在 CPU 环境调试、CI/CD 容器中, 或者用户只想使用 SDPA), 原生 transformers 会直接抛出
       ImportError 并终止模型加载, 即使该模型逻辑内部可能对 flash_attn 做了可选处理(try-except).

    3. 提升离线/受限环境的鲁棒性:
       在 LLaMA-Factory 这种微调框架中, 我们希望尽可能让模型先"加载进来", 具体的算子兼容性逻辑由框架
       后续动态判断, 而不是被 Hugging Face 静态的导入检查卡死.

    深度架构解析(高级研究员视角):

    为什么 transformers 要设计 check_imports?
    Hugging Face 的初衷是安全与确定性. 当从 Hub 下载模型脚本时, 它希望在运行任何代码前, 先确认当前环境满足该脚本的所有依赖, 避免运行到一半崩溃.

    为什么 LLaMA-Factory 必须绕过它?
    在工业级微调场景中, 硬件环境千差万别. 例如:
    用户可能在多模态训练中只使用了文本组件, 但模型脚本里写了图像处理库的 import.
    用户可能使用了 Unsloth 或 DeepSpeed 等第三方加速方案, 这些方案会动态替换掉模型内部的算子实现, 此时模型原始脚本里的那些 import 检查反而成了障碍.

    工程风险点:
    这种补丁属于"先斩后奏". 如果模型代码中真的存在一段必须运行但又缺少依赖的逻辑, 报错会被推迟到真正的 forward 阶段(Runtime Error).
    但对于高级开发者来说, 这种灵活性是必须的, 因为我们有能力在更上层处理这些异常.
    这种设计体现了 LLaMA-Factory "兼容性优先" 的工程哲学, 确保了框架在面对各种自定义程度极高的远程模型代码时, 依然能保持极高的起跑成功率.
    """
    # 增加一个开关逻辑, 允许高级用户通过环境变量强制恢复标准检查
    if not is_env_enabled("FORCE_CHECK_IMPORTS"):
        # 将 check_imports 函数替换为 get_relative_imports.
        # get_relative_imports 仅解析文件依赖关系, 而不会像原生的 check_imports 那样去验证第三方库是否存在.
        # 这样即便模型文件中写了 `import some_missing_lib`, 模型也能成功加载到内存中.
        transformers.dynamic_module_utils.check_imports = get_relative_imports


def torch_gc() -> None:
    r"""Collect the device memory."""
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


def try_download_model_from_other_hub(model_args: "ModelArguments") -> str:
    if (not use_modelscope() and not use_openmind()) or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    if use_modelscope():
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import snapshot_download  # type: ignore
        from modelscope.hub.api import HubApi  # type: ignore

        if model_args.ms_hub_token:
            api = HubApi()
            api.login(model_args.ms_hub_token)

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        with WeakFileLock(os.path.abspath(os.path.expanduser("~/.cache/llamafactory/modelscope.lock"))):
            model_path = snapshot_download(
                model_args.model_name_or_path,
                revision=revision,
                cache_dir=model_args.cache_dir,
            )

        return model_path

    if use_openmind():
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind.utils.hub import snapshot_download  # type: ignore

        with WeakFileLock(os.path.abspath(os.path.expanduser("~/.cache/llamafactory/openmind.lock"))):
            model_path = snapshot_download(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                cache_dir=model_args.cache_dir,
            )

        return model_path


def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")


def use_openmind() -> bool:
    return is_env_enabled("USE_OPENMIND_HUB")


def use_ray() -> bool:
    return is_env_enabled("USE_RAY")


def use_kt() -> bool:
    return is_env_enabled("USE_KT")


def find_available_port() -> int:
    r"""Find an available port on the local machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def fix_proxy(ipv6_enabled: bool = False) -> None:
    r"""Fix proxy settings for gradio ui."""
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    if ipv6_enabled:
        os.environ.pop("http_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("all_proxy", None)
        os.environ.pop("ALL_PROXY", None)

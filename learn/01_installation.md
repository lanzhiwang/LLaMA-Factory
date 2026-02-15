# 环境准备

```bash

conda env list
conda create --name llamafactory python=3.12 -y

conda activate llamafactory

conda deactivate
conda env remove -n llamafactory -y

# 调试时选择 python 解释器
Python: Select Interpreter

pip -v install hatch==1.16.2 uv==0.9.21 -i https://pypi.tuna.tsinghua.edu.cn/simple

uv pip -v install --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -e ".[dev,metrics,deepspeed]"

$ python
Python 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:16:04) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'2.10.0+cu128'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
8
>>> torch.cuda.get_device_name(0)
'NVIDIA H20'
>>> torch.cuda.get_device_name(1)
'NVIDIA H20'
>>> torch.cuda.get_device_name(7)
'NVIDIA H20'
>>>

$ llamafactory-cli version
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.4                |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
$
$ llamafactory-cli -h
Unknown command: -h.
----------------------------------------------------------------------
| Usage:                                                             |
|   llamafactory-cli api -h: launch an OpenAI-style API server       |
|   llamafactory-cli chat -h: launch a chat interface in CLI         |
|   llamafactory-cli export -h: merge LoRA adapters and export model |
|   llamafactory-cli train -h: train models                          |
|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |
|   llamafactory-cli webui: launch LlamaBoard                        |
|   llamafactory-cli env: show environment info                      |
|   llamafactory-cli version: show version info                      |
| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |
----------------------------------------------------------------------
$

$ pwd
/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct
$ modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct --local_dir ./

$ modelscope download --model llava-hf/llava-1.5-7b-hf --local_dir ./
$ modelscope download --model Qwen/Qwen3-32B --local_dir ./
$ modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ./
$ modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir ./

find . -name __pycache__ -exec rm -rf {} \;
find . -name .DS_Store -exec rm -rf {} \;

```

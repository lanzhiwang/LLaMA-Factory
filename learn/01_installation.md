# 环境准备

```bash

conda env list
conda create --name llamafactory python=3.12 -y

conda activate llamafactory

conda deactivate
conda env remove -n llamafactory -y

# 调试时选择 python 解释器
Python: Select Interpreter

pip -v install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU
不安装 torch-npu,eetq

uv pip -v install --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes,hqq,gptq,aqlm,vllm,sglang,galore,apollo,badam,adam-mini,minicpm_v,modelscope,openmind,swanlab,dev]" unsloth

uv pip -v install --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple logbar tokenicer device_smi

添加 --no-build-isolation 参数
uv pip -v install --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes,hqq,gptq,aqlm,vllm,sglang,galore,apollo,badam,adam-mini,minicpm_v,modelscope,openmind,swanlab,dev]" --no-build-isolation

# NPU

uv pip -v install --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -e ".[torch-npu,metrics,deepspeed,bitsandbytes,hqq,galore,apollo,badam,adam-mini,minicpm_v,modelscope,openmind,swanlab,dev]"

$ python
Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
8
>>>

$ llamafactory-cli version
[2025-08-23 12:35:55,282] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 08-23 12:36:00 [__init__.py:239] Automatically detected platform cuda.
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.3                |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
$

$ llamafactory-cli -h
[2025-08-31 13:59:28,946] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 08-31 13:59:30 [__init__.py:239] Automatically detected platform cuda.
Unknown command: -h.
----------------------------------------------------------------------
| Usage:                                                             |
|   llamafactory-cli api -h: launch an OpenAI-style API server       |
|   llamafactory-cli chat -h: launch a chat interface in CLI         |
|   llamafactory-cli eval -h: evaluate models                        |
|   llamafactory-cli export -h: merge LoRA adapters and export model |
|   llamafactory-cli train -h: train models                          |
|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |
|   llamafactory-cli webui: launch LlamaBoard                        |
|   llamafactory-cli version: show version info                      |
----------------------------------------------------------------------
$

$ pwd
/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct

$ modelscope download --model llava-hf/llava-1.5-7b-hf --local_dir ./
$ modelscope download --model Qwen/Qwen3-32B --local_dir ./

find . -name __pycache__ -exec rm -rf {} \;

```

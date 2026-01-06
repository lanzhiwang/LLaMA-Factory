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

import os
import subprocess
import sys
from copy import deepcopy


# 1. 交互设计: 自说明文档 (CLI Self-Documentation)
# [为什么要这么写]: 定义一个美观且清晰的 USAGE 字符串.
# [解决的问题]: LLM 项目参数极其复杂, 提供直观的命令指南能显著降低用户的学习成本.
# 这里的 Hint 提到 `lmf` 快捷方式, 体现了对开发者体验(DX)的极致追求.
USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli env: show environment info                      |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)


def launch():
    # 2. 延迟加载 (Lazy Loading)
    # [为什么要这么写]: 在函数内部导入 logging 和 env 工具.
    # [解决的问题]: Python 的 import 是耗时的. 如果用户只是想看个版本号,
    # 提前导入大量的依赖会导致 CLI 响应非常缓慢. 这种设计确保了"按需加载".
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_kt, use_ray

    logger = logging.get_logger(__name__)
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to LLaMA Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
        + "-" * 58
    )

    # 3. 命令解析逻辑 (Sub-command Dispatching)
    # [为什么要这么写]: 通过 pop(1) 提取子命令(如 train/api).
    # [解决的问题]: 实现了类似 git 或 docker 的多级指令结构, 让一个入口文件可以管理整个微调生命周期.
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    # 4. 强制分布式加速策略 (Force Distributed Strategy)
    if is_env_enabled("USE_MCA"):  # force use torchrun
        # MCA 环境下强制走 torchrun
        os.environ["FORCE_TORCHRUN"] = "1"

    # 5. 自动分布式训练检测 (Automated Distributed Launching)
    # [为什么要这么写]: 这是全片最核心的逻辑. 判断是否需要启动 torchrun.
    # [解决的问题]: 手动写 `torchrun --nproc_per_node=8 ...` 极其繁琐且易错.
    # 这里通过检测 GPU 数量(get_device_count() > 1)并排除 Ray 等框架,
    # 实现"自动判断、自动封装、自动启动"分布式任务, 用户只需输入 `llamafactory-cli train`.
    if command == "train" and (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray() and not use_kt())
    ):
        # 获取分布式配置: 优先从环境变量读(适配 K8s/Slurm), 否则提供默认值
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        # 自动寻找可用端口: 解决多任务并行的端口冲突痛点
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # 6. 弹性训练支持 (Elastic Launch Support)
        # [为什么要这么写]: 支持 rdzv (Rendezvous) 相关参数.
        # [解决的问题]: 适配云原生环境下的容错和弹性扩缩容. 如果某个节点挂了, Elastic 机制能尝试重启任务.
        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        # 7. 环境沙箱化 (Environment Sandboxing)
        # [为什么要这么写]: 使用 deepcopy 复制环境变量.
        # [解决的问题]: 在子进程启动前, 根据配置注入特定的优化参数(如内存分配策略),
        # 同时确保不会污染当前主进程的环境变量, 保证系统的纯净.
        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # 这里的优化是 LLM 训练的经验总结:
            # expandable_segments 减少显存碎片; AVOID_RECORD_STREAMS 提高 NCCL 通信效率.
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # 8. 递归自启动 (Recursive Self-Invocation)
        # [为什么要这么写]: torchrun 后面接的是 __file__ (即当前脚本自身).
        # [解决的问题]: 这是一个非常精妙的设计. 脚本检测到多卡环境后, 不再往下执行,
        # 而是调用 `torchrun` 再次启动自己, 但第二次启动时, 由于已经在 torchrun 环境中,
        # 它将直接进入真正的训练分支(见文件末尾的 if __name__ == "__main__").
        if rdzv_id is not None:
            # 弹性启动模式
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            process = subprocess.run(
                (
                    "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                    "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                    "--max-restarts {max_restarts} {file_name} {args}"
                )
                .format(
                    rdzv_nnodes=rdzv_nnodes,
                    nproc_per_node=nproc_per_node,
                    rdzv_id=rdzv_id,
                    master_addr=master_addr,
                    master_port=master_port,
                    max_restarts=max_restarts,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )
        else:
            # 标准多卡模式
            # 注意: 不使用 shell=True 是为了防止 Shell 注入攻击, 且更利于信号传递(如 Ctrl+C 停止训练).
            # NOTE: DO NOT USE shell=True to avoid security risk
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=nnodes,
                    node_rank=node_rank,
                    nproc_per_node=nproc_per_node,
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )

        sys.exit(process.returncode)

    # 9. 业务路由 (Business Routing)
    # [为什么要这么写]: 简单的 elif 结构将各个功能模块(API/Chat/Export/Train/UI)清晰地分开.
    elif command == "api":
        from .api.app import run_api

        run_api()

    elif command == "chat":
        from .chat.chat_model import run_chat

        run_chat()

    elif command == "eval":
        # 版本管理策略: 通过 NotImplementedError 告知功能废弃, 比直接删除更友好.
        raise NotImplementedError("Evaluation will be deprecated in the future.")

    elif command == "export":
        from .train.tuner import export_model

        export_model()

    elif command == "train":
        # 如果代码走到这里, 说明是单卡训练或者已经被 torchrun 启动的子进程.
        from .train.tuner import run_exp

        run_exp()

    elif command == "webchat":
        from .webui.interface import run_web_demo

        run_web_demo()

    elif command == "webui":
        from .webui.interface import run_web_ui

        run_web_ui()

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    # 10. 递归调用的出口 (The Recusion Exit)
    # [为什么要这么写]: 当 `torchrun` 启动当前文件作为子进程时, 会触发这里.
    # [解决的问题]: 由于 `torchrun` 会把环境变量设置好, 此时直接调用 `run_exp`.
    # 使用绝对导入 `from llamafactory.train.tuner` 确保在各种 PYTHONPATH 场景下都能定位到训练核心.
    from llamafactory.train.tuner import run_exp  # use absolute import

    run_exp()

"""
高级研究员视角下的架构总结:

自动化运维(AIOps)倾向: 这段代码通过 find_available_port 和 get_device_count 自动探测硬件, 解决了 LLM 训练中最头疼的"环境适配"问题.

优雅的递归启动: 利用 subprocess.run(["torchrun", ..., __file__]) 将自己作为训练脚本再次启动. 这种"脚本即启动器"的设计, 让用户不需要在 python、torchrun、deepspeed 各种启动命令之间纠结, 实现了单一入口点(Single Point of Entry).

工业级健壮性: 代码中处理了 rdzv(弹性)、env(隔离)、OPTIM_TORCH(底层内存优化)等, 说明这不仅是一个学术 Demo, 而是考虑了在高性能计算集群(HPC)长期运行的工业级工具.

希望这段深度注释能帮助你掌握 LLaMA-Factory 的设计精髓!
"""

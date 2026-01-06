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

from llamafactory.train.tuner import run_exp


def main():
    # 1. 统一的任务编排入口 (Unified Orchestration)
    # [为什么要这么写]: run_exp 是 LLaMA-Factory 的核心调度器.
    # 它内部集成了配置解析、环境初始化、以及不同阶段(预训练、指令微调、奖励建模、PPO/DPO)的逻辑.
    # [解决的问题]: 通过一个统一的入口, 用户只需要修改外部配置文件(如 YAML 或命令行参数),
    # 而不需要修改代码逻辑. 这种"配置驱动"的设计极大地降低了算法工程师在不同微调策略间切换的成本.
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    # 2. 针对 TPU 架构的多进程兼容性 (XLA/TPU Multi-processing Support)
    # [为什么要这么写]: 这是专门为 Google TPU (Tensor Processing Units) 和 torch_xla 库预留的函数签名.
    # `index` 参数由 torch_xla 的多进程生成器(xla_spawn)自动传入, 表示当前计算核心的索引.
    # [解决的问题]: 解决 LLM 在异构计算设备(非 NVIDIA GPU)上的启动差异.
    # 在 TPU 环境下, PyTorch 需要通过一个特定的回调函数来生成多进程.
    # 这里的 `_mp_fn` 允许 LLaMA-Factory 无缝运行在 Google Cloud TPUs 上, 确保了框架的硬件通用性.
    run_exp()


if __name__ == "__main__":
    # 3. 标准的脚本入口保护与进程隔离
    # [为什么要这么写]: 这是 Python 开发的最佳实践.
    # [解决的问题]: 防止在多进程/多卡训练(Distributed Data Parallel, DDP)过程中,
    # 子进程在 import 该脚本时意外地递归触发 main() 函数, 导致显存溢出或逻辑混乱.
    # 同时, 它也确保了该脚本既可以作为独立的命令行工具运行, 也可以被其他模块安全地调用.
    main()

"""
深度架构解析(高级研究员视角):

为什么不直接在 main 里写逻辑?
作为高级工程师, 我们追求的是 "Clean Entry Point". 将所有复杂的逻辑封装在 run_exp 内部, 意味着 llamafactory.train.tuner 模块已经处理了:
参数解耦: 从命令行或环境变量中读取 TrainArgs.
算力自动检测: 自动判断是使用 DistributedDataParallel (DDP)、DeepSpeed 还是 FSDP.
全流程覆盖: 无论是加载 7B 还是 70B 模型, 无论是 Full-Tuning 还是 LoRA, 对于入口脚本来说都是透明的.

关于 _mp_fn 的必要性:
在 LLM 微调领域, 硬件多样性(NVIDIA GPU, AMD GPU, TPU, 昇腾 NPU 等)是一个巨大的挑战. _mp_fn 的存在标志着该框架在设计之初就考虑到了 OpenXLA 生态. 即使你现在只在 NVIDIA GPU 上训练, 这个接口也为未来迁移到更廉价、更高带宽的 TPU 集群提供了"零代码修改"的可能性.

工业级健壮性:
这种简短的入口文件非常利于 CI/CD(持续集成). 测试脚本可以非常容易地调用这个 main 函数, 或者通过 run_exp 进行单元测试, 而不用担心复杂的依赖注入问题.

这段代码体现了"大繁至简"的工程思想: 最核心的逻辑隐藏在最坚固的抽象之后.
"""

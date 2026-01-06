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


def main():
    # 1. 延迟导入与局部作用域隔离 (Lazy Import & Scope Isolation)
    # [为什么要这么写]: 在函数内部导入, 而不是在文件顶部导入.
    # [解决的问题]: 避免在未确定版本前就加载大量复杂的依赖库. LLM 项目依赖极多(如 torch, transformers, deepspeed),
    # 顶层导入会导致即便只是查看帮助信息也会产生漫长的等待(Cold Start), 且容易引发循环导入(Circular Dependency)问题.
    from .extras.misc import is_env_enabled

    # 2. 特性开关与版本灰度控制 (Feature Flagging & Versioning)
    # [为什么要这么写]: 通过环境变量 "USE_V1" 动态决定加载哪个版本的启动器.
    # [解决的问题]: 这是解决"破坏性重构(Breaking Changes)"的最佳实践.
    # 当项目从 V1 升级到 V2 时, 底层逻辑可能完全重写, 但用户习惯和旧脚本可能仍依赖 V1.
    # 这种写法允许同一套代码库同时兼容两套逻辑:
    # - 默认走最新的架构(.launcher)
    # - 通过设置环境变量(USE_V1=1)回退到稳定旧版(.v1.launcher)
    # 极大地降低了用户升级框架后业务崩溃的风险.
    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    else:
        from . import launcher

    # 3. 统一接口抽象 (Polymorphism / Interface Abstraction)
    # [为什么要这么写]: 无论哪个版本的 launcher, 都必须实现 .launch() 方法.
    # [解决的问题]: 对上层调用者屏蔽底层实现的复杂性, 实现"即插即用".
    launcher.launch()


if __name__ == "__main__":
    # 4. Windows 平台与打包环境兼容性 (Multiprocessing Freeze Support)
    # [为什么要这么写]: 在执行 main 之前显式调用 freeze_support().
    # [解决的问题]:
    # - LLM 训练频繁使用 multiprocessing(多进程)来加速数据加载或分布式训练.
    # - 在 Windows 平台下, 或者在使用 PyInstaller 等工具将 Python 脚本打包成 exe 之后,
    #   创建子进程时可能会出现"无限递归启动"的问题(即子进程又执行一遍 main 逻辑).
    # freeze_support() 能确保子进程在启动时能正确识别自己是子进程, 从而避免"分身炸弹"导致内存和 CPU 瞬间溢出.
    from multiprocessing import freeze_support

    freeze_support()

    # 5. 执行主逻辑
    main()

"""
深度技术解析(高级研究员视角):

关于 is_env_enabled 的工程考量:
在 LLM 开发中, 我们经常需要在不同的实验环境(例如 A100 集群 vs 个人笔记本)切换配置. 直接修改 config.py 很容易被 Git 误提交. 使用环境变量(Env Vars)结合 is_env_enabled 是工业界最推荐的 "无侵入式配置" 方案.

版本隔离策略:
from .v1 import launcher 说明 LLaMA-Factory 在内部维护了一个完整的代码快照. 这在高级重构中非常有用 - 你可以完全改变 V2 的数据流架构, 而不必担心破坏 V1 的稳定性. 这解决了 "快速迭代" 与 "生产环境稳定" 之间的天然矛盾.

多进程的安全性:
由于 LLM 涉及大量的 GPU 算力调度, 主进程与子进程之间的通信非常频繁. if __name__ == "__main__": 配合 freeze_support() 是高级 Python 工程师处理 Distributed Data Parallel (DDP) 或 多张显卡并行计算 时的标配. 如果不加这一行, 在 Windows 环境下运行分布式微调脚本时, 程序几乎百分之百会崩溃并报出 RuntimeError.

"""

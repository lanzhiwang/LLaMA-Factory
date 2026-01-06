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

from llamafactory.extras.misc import fix_proxy, is_env_enabled
from llamafactory.webui.interface import create_ui


def main():
    # 1. 环境感知配置 (Environment-Aware Configuration)
    # [为什么要这么写]: 通过自定义的 is_env_enabled 工具函数解析环境变量.
    # [解决的问题]: 在 Docker 或 HPC 集群中, 用户通常无法直接修改 Python 代码.
    # 通过环境变量(如 GRADIO_IPV6=1)来控制布尔开关, 比直接解析字符串更健壮, 支持 true/1/on 等多种写法.
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")

    # 2. 智能网卡绑定 (Network Interface Binding Strategy)
    # [为什么要这么写]: 根据是否启用 IPv6 动态选择监听地址.
    # [解决的问题]:
    # - "0.0.0.0" (IPv4) 是最通用的绑定方式, 允许局域网访问.
    # - "[::]" (IPv6) 则是为了兼容现代纯 IPv6 网络环境(如某些大学内网).
    # 这种写法解决了 Web UI 在不同云服务器环境下"启动成功但外部打不开"的常见网络配置故障.
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")

    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")

    # 3. 代理环境自动修复 (Proxy Conflict Resolution)
    # [为什么要这么写]: 调用内部工具函数 fix_proxy.
    # [解决的问题]: 在很多开发环境下, 系统设置了 http_proxy 代理.
    # Gradio 底层组件(如组件间的 HTTP 请求)常因错误地经过代理导致 127.0.0.1 回环地址连接超时.
    # 此函数会在启动前清理不必要的环境变量或设置 no_proxy, 确保本地服务通信正常, 避免"网页加载卡死"或"API 请求无响应".
    fix_proxy(ipv6_enabled=gradio_ipv6)

    # 4. 生产级 UI 组件初始化与分发 (Queueing & Launching)
    # [为什么要这么写]:
    # - create_ui(): 采用工厂模式构建复杂的 Gradio 界面.
    # - .queue(): [关键点]LLM 训练或推理是长耗时操作. 不启用 queue 会导致 HTTP 连接在任务结束前超时.
    #   启用 queue 后, Gradio 会利用 WebSocket 维护连接, 并支持多用户排队.
    # - launch(): 正式启动 Web Server.
    # - share=gradio_share: 解决没有公网 IP 的开发者远程访问的需求(通过 Gradio 内网穿透).
    # - inbrowser=True: 自动打开浏览器, 提升本地开发者的交互体验.
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


if __name__ == "__main__":
    # [安全保护]: 确保多进程启动(如 Gradio 的 reload 模式)时不会重复执行主逻辑.
    main()

"""
深度架构解析(高级研究员视角):

为什么 Web UI 如此重要?
在 LLM 研究中, 微调参数(Learning Rate, LoRA Rank, Epochs)极其琐碎. LLaMA-Factory 坚持做一个强大的 Web UI, 是为了将复杂的命令行(CLI)参数可视化. 这极大降低了研究人员因敲错一个参数名而导致训练失败的概率.

关于 fix_proxy 的工程实践:
在深度学习环境中, 这是一个非常容易被忽视的"坑". 很多研究员在公司内网使用服务器, 设置了 export http_proxy=.... 如果不做处理, Gradio 的内部前后端通信也会去走代理, 导致莫名其妙的连接失败. 这段代码体现了 LLaMA-Factory 对工业界真实开发场景的深度适配.

高性能队列系统:
create_ui().queue() 的调用是必须的. 在进行 LLM 微调时, 显卡会长时间处于满载状态, Python GIL 也可能受到影响. 通过 Gradio 的 queue 机制, 系统可以优雅地处理成百上千个并发连接, 并按照 FIFO(先来先服务)原则分配 GPU 资源, 防止服务器因请求过多而崩溃.

通过这份注释, 你应该能感受到, 这段看似简单的启动脚本, 实际上涵盖了环境变量管理、网络通信协议、代理冲突解决以及并发流量控制等多个维度的考量.
"""

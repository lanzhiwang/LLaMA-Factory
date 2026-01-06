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

import uvicorn

from llamafactory.api.app import create_app
from llamafactory.chat import ChatModel


def main():
    # 1. 初始化推理核心 (Resource Initialization & Cold Start)
    # [为什么要这么写]: 在启动 Web 服务前先实例化 ChatModel.
    # [解决的问题]: LLM 权重加载(Weights Loading)通常涉及数 GB 甚至上百 GB 的显存占用.
    # 将其放在 main 中作为单例(Singleton)初始化, 可以确保模型只加载一次.
    # 同时, 如果显存不足或模型路径错误, 程序会在服务启动前直接报错退出, 避免了服务启动后调用才发现异常.
    chat_model = ChatModel()

    # 2. 依赖注入 (Dependency Injection) 创建 FastAPI 实例
    # [为什么要这么写]: 通过工厂模式 create_app 并传入 chat_model 实例.
    # [解决的问题]: 实现了 Web 框架逻辑(FastAPI)与底层模型推理逻辑(ChatModel)的解耦.
    # 这样做便于单元测试(可以传入 Mock 模型)以及灵活扩展.
    # 所有的 API 路由(如 /v1/chat/completions)都将共享这个已经加载好的 chat_model 资源.
    app = create_app(chat_model)

    # 3. 动态配置管理 (Environment Variable Configuration)
    # [为什么要这么写]: 使用 os.getenv 获取配置, 并提供默认值.
    # [解决的问题]: 这是符合"12-Factor App"原则的微服务设计规范.
    # 在容器化环境(如 Docker, Kubernetes)中, 我们通常不希望修改代码来改变 IP 或端口,
    # 而是通过环境变量动态注入, 使得同一个镜像可以在不同的环境(开发、测试、生产)中无缝运行.
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))

    # 4. 交互性提示 (User Experience)
    print(f"Visit http://localhost:{api_port}/docs for API document.")

    # 5. 启动高性能 ASGI 服务器 (Production-Ready Server)
    # [为什么要这么写]: 调用 uvicorn.run 启动异步服务器.
    # [解决的问题]: FastAPI 是异步框架, 需要 ASGI 服务器支持.
    # uvicorn 提供了极高的并发处理能力, 能够有效处理 LLM 推理这种长耗时(Long-running)的 HTTP 请求,
    # 配合 Streaming 输出(流式响应)能显著提升用户体验.
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    # [为什么要这么写]: 标准的 Python 入口保护.
    # [解决的问题]: 防止该脚本在被其他模块 import 时意外执行模型加载和服务器启动逻辑.
    # 这在大型项目和多进程环境下(如多卡并行、分布式推理)是必须的安全规范.
    main()

"""
深度架构解析(高级研究员视角):

为什么不直接在 create_app 内部实例化 ChatModel?
在高级 Python 开发中, 我们要遵循"显式优于隐式". 将 chat_model 在外部初始化, 可以让开发者清晰地控制模型的生命周期. 例如, 如果未来需要支持多模型切换或热更新, 这种结构更容易扩展.

关于 API 的兼容性设计:
LLaMA-Factory 的这个 API 结构通常是为了适配 OpenAI API 格式. create_app 内部会定义符合 OpenAI 协议的 Request/Response Schema. 这解决了生态对接问题, 使得用户可以无缝使用各种支持 OpenAI 接口的开源工具(如 LangChain、ChatBox 等).

性能考量:
0.0.0.0 作为默认 Host 是为了确保在 Docker 容器内运行时, 外部宿主机能够成功访问映射端口(如果设置为 127.0.0.1, 容器外将无法连接).
"""


--------------------------------------------------------------------------------------------------------------------------

Q: 假设你是一位精通 python 的高级开发人员, 我使用如下命令启动了一个容器
```
docker run -ti --rm \
--entrypoint /usr/bin/env \
--security-opt seccomp=unconfined \
--gpus '"device=0,1,2,3,4,5,6,7"' \
--name pytorch-server \
pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel bash
```
我在容器中想安装 python 第三方依赖, 但是有如下报错
```
root@0dfb9f4ef521:/workspace# uv pip install --system av
Using Python 3.12.3 environment at: /usr
error: The interpreter at /usr is externally managed, and indicates the following:

  To install Python packages system-wide, try apt install
  python3-xyz, where xyz is the package you are trying to
  install.

  If you wish to install a non-Debian-packaged Python package,
  create a virtual environment using python3 -m venv path/to/venv.
  Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
  sure you have python3-full installed.

  If you wish to install a non-Debian packaged Python application,
  it may be easiest to use pipx install xyz, which will manage a
  virtual environment for you. Make sure you have pipx installed.

  See /usr/share/doc/python3.12/README.venv for more information.

hint: Virtual environments were not considered due to the `--system` flag
root@0dfb9f4ef521:/workspace# pip install av
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.

    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.

    If you wish to install a non-Debian packaged Python application,
    it may be easiest to use pipx install xyz, which will manage a
    virtual environment for you. Make sure you have pipx installed.

    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
root@0dfb9f4ef521:/workspace#
```
请问这是什么原因, 应该如何解决

这个报错的根本原因以及几种推荐的解决方法如下.

---

### 一、报错原因

该报错是由于 PEP 668(Marking Python environments as externally managed)机制引起的.

在较新的 Linux 发行版(如 Ubuntu 23.04+、Debian 12+, 即你容器内正在使用的底层系统)中, 系统自带的 Python 环境(通常在 `/usr` 目录下)被标记为"外部管理环境"(Externally Managed Environment).

系统通过在 Python 的库目录中放置一个名为 `EXTERNALLY-MANAGED` 的配置文件来触发此保护. 这样做的初衷是为了防止用户通过 `pip` 意外升级或修改系统级别(通过 `apt` 安装)的 Python 包, 从而避免破坏依赖系统 Python 的底层 OS 工具.

尽管在容器(Docker)中我们通常把容器本身视为隔离环境, 并不担心破坏宿主机系统, 但容器内的基础镜像依然保留了这一限制机制.

---

### 二、解决方案

根据你在容器中的具体开发习惯, 可以选择以下几种解决方法:

#### 方法 1: 绕过系统安全限制(推荐, 最省事)
在 Docker 容器这种本身已经隔离的环境中, 直接向系统 Python 安装第三方库通常是安全的. 你可以通过添加参数来显式允许这一操作.

* 使用 `uv` 时的命令:
    ```bash
    uv pip install --system --break-system-packages av
    ```
* 使用 `pip` 时的命令:
    ```bash
    pip install --break-system-packages av
    ```
* 或者设置环境变量(一劳永逸):
    如果你不想每次安装都带上长参数, 可以在 shell 中设置环境变量:
    ```bash
    export PIP_BREAK_SYSTEM_PACKAGES=1
    pip install av
    ```
    (如果是写在 `Dockerfile` 中, 可以加上 `ENV PIP_BREAK_SYSTEM_PACKAGES=1`)

---

#### 方法 2: 彻底删除系统保护文件(Docker 构建时推荐)
如果你不希望在使用 `pip` 或 `uv` 时每次都加上 `--break-system-packages`, 可以直接删除触发该限制的配置文件.

在容器中执行以下命令:
```bash
rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED
```
删除后, 系统 Python 就不再是"外部管理环境"了. 你可以像以前一样直接运行:
```bash
pip install av
# 或者
uv pip install --system av
```

---

#### 方法 3: 遵循最佳实践, 创建虚拟环境(标准、规范)
如果你希望严格遵循现代 Python 包管理规范, 可以使用虚拟环境. 这可以确保你的第三方依赖与系统 Python 彻底隔离. 由于你正在使用性能极佳的 `uv`, 推荐直接用 `uv` 来管理虚拟环境.

1. 创建虚拟环境:
    ```bash
    uv venv
    ```
    (这会在当前目录下创建一个 `.venv` 文件夹)

2. 激活虚拟环境:
    ```bash
    source .venv/bin/activate
    ```

3. 安装依赖(此时无需加 `--system` 或 `--break-system-packages`):
    ```bash
    uv pip install av
    ```

--------------------------------------------------------------------------------------------------------------------------

Q: 我现在想以 pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel 这个镜像为基础镜像构建我自己的镜像, 我应该如何在 Dockerfile 遵循最佳实践, 创建虚拟环境?
在基础镜像 pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel 中已经存在很多的第三方依赖, 比如 torch==2.12.1+cu126, 我在创建的虚拟环境中可以使用这些已经存在的依赖吗?

在以深度学习镜像(如 `pytorch/pytorch`)为基础镜像构建你自己的 Docker 镜像时, 关于是否使用虚拟环境, 以及如何合理继承镜像中已经预装的大量重型依赖(如 PyTorch、CUDA 相关的库), 有两种不同的最佳实践路径.

下面为您详细解答"能否使用已有依赖"的问题, 并提供两种符合最佳实践的 Dockerfile 编写方案.

---

### 一、能在创建的虚拟环境中使用已有的依赖吗?

可以, 但需要显式声明.

默认情况下, Python 虚拟环境(不管是通过 `python3 -m venv` 还是 `uv venv` 创建)是完全干净且隔离的, 它们无法访问外部系统环境中的任何第三方包.

若想在虚拟环境中直接调用基础镜像中已经安装好的 `torch` 等依赖, 必须在创建虚拟环境时加上 `--system-site-packages` 参数. 这样, 虚拟环境在运行时如果发现自身没有某个包, 就会自动去寻找并调用系统自带的包.

---

### 二、Dockerfile 最佳实践方案

针对深度学习容器的特殊性, 业内目前有以下两种主流的实践方案:

#### 方案 A: 标准的虚拟环境流(使用 `--system-site-packages`)

如果你希望坚持在 Docker 内部也使用虚拟环境, 可以通过注入环境变量的方式, 让后续所有的容器层和运行期都默认激活该虚拟环境.

Dockerfile 示例:

```dockerfile
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel

# 1. 采用多阶段构建的方式优雅地将最新的 uv 工具引入到我们的镜像中
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# 2. 创建一个"允许访问系统级 site-packages"的虚拟环境
# 这能保证我们在虚拟环境中可以直接 import torch 等预装库
RUN uv venv --system-site-packages /opt/venv

# 3. 最佳实践: 通过修改 PATH 环境变量来"自动激活"虚拟环境
# 这样不仅在构建时(后续的 RUN), 在容器运行时(CMD/ENTRYPOINT)也无需手动执行 source activate
ENV PATH="/opt/venv/bin:$PATH"

# 4. 复制并安装你自己的依赖
COPY requirements.txt .

# [⚠️重要避坑指南]
# 由于 uv 的底层设计机制: 当使用 `--system-site-packages` 创建虚拟环境后,
# Python 在运行期可以访问系统包, 但 `uv pip install` 默认并不会将系统包纳入其依赖解析图.
# 也就是说, 如果你的 requirements.txt 中有其他依赖声明了"需要 torch"(即便系统已预装),
# `uv pip install` 可能会因为"看不见系统包"而尝试重新下载一个巨大的 PyTorch 包.
#
# 解决方法:
# 如果 requirements.txt 中包含需要解析 PyTorch 依赖的包, 建议在此处使用标准 pip 进行安装,
# 标准 pip 会正确识别系统里已存在的 torch 库并显示 "Requirement already satisfied":
RUN pip install -r requirements.txt

# 如果你的依赖里完全不涉及 torch 相关的依赖包解析, 也可以直接用 uv:
# RUN uv pip install -r requirements.txt
```

---

#### 方案 B: 容器即是虚拟环境(深度学习领域的更主流实践)

在云原生和深度学习工程中, 有许多人认为: "Docker 容器本身就是一个天然的、完全隔离的虚拟环境".

既然我们已经通过 `docker run --rm` 或 K8s 保证了容器间的环境隔离, 那么在容器内部, 直接向系统级的 Python 环境中安装第三方库, 是最不易出错、路径最简单、最能充分发挥基础镜像原本环境优势的做法.

Dockerfile 示例:

```dockerfile
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel

# 1. 引入 uv 工具
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# 2. 核心: 通过环境变量绕过系统 Python 的安全保护限制 (PEP 668)
# 这可以让我们安全地直接向容器的系统环境里安装包, 而不必创建额外的虚拟环境
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt .

# 3. 直接使用 --system 将包安装到系统环境中
# 此时 uv 会无缝读取并合并系统中的 torch、cuda 等依赖, 绝不会发生"重新下载 torch"的情况
RUN uv pip install --system -r requirements.txt
```

### 三、总结建议

* 如果您追求绝对的包隔离与规范: 选择 方案 A, 但请注意在使用 `uv` 安装涉及 torch 依赖的第三方包时, 改用 `pip install` 来避免 `uv` 重新下载庞大的 torch 安装包.
* 如果您追求最简单、最高效、且不易在路径或依赖冲突上踩坑: 强烈建议选择 方案 B. 这是目前绝大多数深度学习开源镜像(如 Hugging Face、vLLM 等)在编写官方 Dockerfile 时普遍采用的成熟方案.

--------------------------------------------------------------------------------------------------------------------------

Q: 我使用如下 Dockerfile 在 github action 上构建了自定义镜像
```
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel

ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

COPY . /workspace

RUN set -eux && \
    echo ${PIP_BREAK_SYSTEM_PACKAGES} && \
    env && \
    ls -al && \
    uv pip install --system -e ".[dev,metrics,deepspeed]" && \
    llamafactory-cli version && \
    pip cache purge && \
    uv cache clean
```
详细的输出日志如下:
```
#9 [4/4] RUN set -eux &&     echo 1 &&     env &&     ls -al &&     uv pip install --system -e ".[dev,metrics,deepspeed]" &&     llamafactory-cli version &&     pip cache purge &&     uv cache clean
#9 0.247 1
#9 0.247 + echo 1
#9 0.247 + env
#9 0.250 LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#9 0.250 HOME=/root
#9 0.250 NVIDIA_DRIVER_CAPABILITIES=compute,utility
#9 0.250 PIP_BREAK_SYSTEM_PACKAGES=1
#9 0.250 PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#9 0.250 PYTORCH_VERSION=2.12.1
#9 0.250 CUDA_HOME=/usr/local/cuda
#9 0.250 *
#9 0.250 NVIDIA_VISIBLE_DEVICES=all
#9 0.250 + ls -al
#9 0.256 total 224
#9 0.256 drwxr-xr-x  1 root root  4096 Jun 24 08:13 .
#9 0.256 drwxr-xr-x  1 root root  4096 Jun 24 08:16 ..
#9 0.256 -rw-r--r--  1 root root   705 Jun 24 08:11 .env.local
#9 0.256 drwxr-xr-x  4 root root  4096 Jun 24 08:11 .github-bak
#9 0.256 -rw-r--r--  1 root root  3297 Jun 24 08:11 .gitignore-bak
#9 0.256 -rw-r--r--  1 root root   713 Jun 24 08:11 .pre-commit-config.yaml
#9 0.256 -rw-r--r--  1 root root  1378 Jun 24 08:11 CITATION.cff
#9 0.256 -rw-r--r--  1 root root 11324 Jun 24 08:11 LICENSE
#9 0.256 -rw-r--r--  1 root root    16 Jun 24 08:11 MANIFEST.in
#9 0.256 -rw-r--r--  1 root root   741 Jun 24 08:11 Makefile
#9 0.256 -rw-r--r--  1 root root 71290 Jun 24 08:11 README.md
#9 0.256 -rw-r--r--  1 root root 70055 Jun 24 08:11 README_zh.md
#9 0.256 drwxr-xr-x  4 root root  4096 Jun 24 08:11 assets
#9 0.256 drwxr-xr-x  3 root root  4096 Jun 24 08:11 data
#9 0.256 drwxr-xr-x 14 root root  4096 Jun 24 08:11 examples
#9 0.256 -rw-r--r--  1 root root  3588 Jun 24 08:11 pyproject.toml
#9 0.256 drwxr-xr-x  5 root root  4096 Jun 24 08:11 scripts
#9 0.256 drwxr-xr-x  3 root root  4096 Jun 24 08:11 src
#9 0.256 drwxr-xr-x  7 root root  4096 Jun 24 08:11 tests
#9 0.256 drwxr-xr-x  7 root root  4096 Jun 24 08:11 tests_v1
#9 0.256 + uv pip install --system -e .[dev,metrics,deepspeed]
#9 0.466 Using Python 3.12.3 environment at: /usr
#9 0.473 error: The interpreter at /usr is externally managed, and indicates the following:
#9 0.473
#9 0.473   To install Python packages system-wide, try apt install
#9 0.473   python3-xyz, where xyz is the package you are trying to
#9 0.473   install.
#9 0.473
#9 0.473   If you wish to install a non-Debian-packaged Python package,
#9 0.473   create a virtual environment using python3 -m venv path/to/venv.
#9 0.473   Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
#9 0.473   sure you have python3-full installed.
#9 0.473
#9 0.473   If you wish to install a non-Debian packaged Python application,
#9 0.473   it may be easiest to use pipx install xyz, which will manage a
#9 0.473   virtual environment for you. Make sure you have pipx installed.
#9 0.473
#9 0.473   See /usr/share/doc/python3.12/README.venv for more information.
#9 0.473
#9 0.473 hint: Virtual environments were not considered due to the `--system` flag
```
似乎环境变量没有起作用, 请问这是什么原因, 应该如何解决

原因在于: `PIP_BREAK_SYSTEM_PACKAGES` 是官方 `pip` 专用的环境变量, 而 `uv` 作为一个用 Rust 编写的独立包管理工具, 默认并不会读取这个变量.

尽管 `uv pip` 的命令格式设计得和 `pip` 非常相似, 但它的配置项和环境变量拥有自己的命名空间.

解决这个问题有两种最直接的方法:

---

### 解决方法 1: 使用 `uv` 专属环境变量(推荐)

`uv` 读取的环境变量是 `UV_BREAK_SYSTEM_PACKAGES=true`.

建议在 Dockerfile 中将两个环境变量都写上, 这样不仅能兼容 `uv`, 也能兼容后续可能调用到的标准 `pip`(例如部分第三方库构建脚本内部会调用 `pip`):

```dockerfile
# 兼容 pip
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# 兼容 uv (必须设置为 true 字符串)
ENV UV_BREAK_SYSTEM_PACKAGES=true
```

---

### 解决方法 2: 直接在命令行传入参数

除了设置环境变量, 你也可以直接在 `uv` 的命令行指令中显式带上 `--break-system-packages` 参数:

```bash
uv pip install --system --break-system-packages -e ".[dev,metrics,deepspeed]"
```

---

### 修改后的 Dockerfile 示例

你可以直接将 Dockerfile 调整为如下内容:

```dockerfile
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-devel

# 显式设置兼容 pip 和 uv 的解除限制变量
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=true

WORKDIR /workspace

COPY . /workspace

RUN set -eux && \
    echo ${PIP_BREAK_SYSTEM_PACKAGES} && \
    env && \
    ls -al && \
    # 直接执行, uv 会自动读取 UV_BREAK_SYSTEM_PACKAGES 变量
    uv pip install --system -e ".[dev,metrics,deepspeed]" && \
    llamafactory-cli version && \
    pip cache purge && \
    uv cache clean
```

--------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------

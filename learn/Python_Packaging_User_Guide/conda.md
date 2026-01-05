假设你是一位精通 python 的高级开发工程师, 现在我拿到一个新的 Python 项目, 在这个 Python 项目中有 pyproject.toml 文件, 文件内容如下:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llamafactory"
dynamic = ["version"]
description = "Unified Efficient Fine-Tuning of 100+ LLMs"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11.0"
authors = [
    { name = "hiyouga", email = "hiyouga@buaa.edu.cn" }
]
keywords = [
    "AI",
    "LLM",
    "GPT",
    "ChatGPT",
    "Llama",
    "Transformer",
    "DeepSeek",
    "Pytorch"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]
dependencies = [
    # core deps
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "torchaudio>=2.4.0",
    "transformers>=4.51.0,<=4.57.1,!=4.52.0,!=4.57.0",
    "datasets>=2.16.0,<=4.0.0",
    "accelerate>=1.3.0,<=1.11.0",
    "peft>=0.14.0,<=0.17.1",
    "trl>=0.18.0,<=0.24.0",
    "torchdata>=0.10.0,<=0.11.0",
    # gui
    "gradio>=4.38.0,<=5.50.0",
    "matplotlib>=3.7.0",
    "tyro<0.9.0",
    # ops
    "einops",
    "numpy",
    "pandas",
    "scipy",
    # model and tokenizer
    "sentencepiece",
    "tiktoken",
    "modelscope",
    "hf-transfer",
    "safetensors",
    # python
    "av",
    "fire",
    "omegaconf",
    "packaging",
    "protobuf",
    "pyyaml",
    "pydantic",
    # api
    "uvicorn",
    "fastapi",
    "sse-starlette"
]

[project.optional-dependencies]
dev = ["pre-commit", "ruff", "pytest", "build"]
metrics = ["nltk", "jieba", "rouge-chinese"]
deepspeed = ["deepspeed>=0.10.0,<=0.16.9"]

[project.scripts]
llamafactory-cli = "llamafactory.cli:main"
lmf = "llamafactory.cli:main"

[project.urls]
Homepage = "https://github.com/hiyouga/LLaMA-Factory"
Repository = "https://github.com/hiyouga/LLaMA-Factory"

[tool.hatch.build.targets.wheel]
packages = ["src/llamafactory"]

[tool.hatch.version]
path = "src/llamafactory/extras/env.py"
pattern = "VERSION = \"(?P<version>[^\"]+)\""

[tool.ruff]
target-version = "py311"
line-length = 119
indent-width = 4

[tool.ruff.lint]
ignore = [
    "C408",  # collection
    "C901",  # complex
    "E501",  # line too long
    "E731",  # lambda function
    "E741",  # ambiguous var name
    "UP007", # no upgrade union
    "UP045", # no upgrade optional
    "D100",  # no doc public module
    "D101",  # no doc public class
    "D102",  # no doc public method
    "D103",  # no doc public function
    "D104",  # no doc public package
    "D105",  # no doc magic method
    "D107",  # no doc __init__
]
extend-select = [
    "C",      # complexity
    "E",      # error
    "F",      # pyflakes
    "I",      # isort
    "W",      # warning
    "UP",     # pyupgrade
    "D",      # pydocstyle
    "PT009",  # pytest assert
    "RUF022", # sort __all__
]

[tool.ruff.lint.isort]
lines-after-imports = 2
known-first-party = ["llamafactory"]
known-third-party = [
    "accelerate",
    "datasets",
    "gradio",
    "numpy",
    "peft",
    "torch",
    "transformers",
    "trl",
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
skip-magic-trailing-comma = false
line-ending = "auto"

```

从这个 pyproject.toml 文件中可以看到这个项目使用 hatch 做虚拟环境和依赖管理, 我现在想安装这个项目的所有依赖, 请问我该如何做

你好! 作为一名高级开发工程师, 我很高兴为你解析这个项目的环境配置.

从 pyproject.toml 文件来看, 该项目(LLaMA-Factory)确实采用了现代 Python 打包标准(PEP 517/518), 并使用 Hatchling 作为构建后端. 虽然 hatchling 是构建后端, 但你可以通过多种方式安装依赖.

以下是针对不同场景的安装方案, 从"最原生"到"最通用"依次排列:

方案一: 使用 Hatch(项目原生推荐)

由于项目中定义了 tool.hatch 相关配置, 使用 hatch 命令行工具可以获得最完整的一致性体验.

安装 Hatch(如果你还没安装):

```Bash
pip install hatch
```

创建并进入环境:
运行以下命令, Hatch 会自动读取 pyproject.toml, 创建虚拟环境并安装 dependencies 中定义的所有核心依赖:

```Bash
hatch shell
```

安装可选依赖:
如果你需要开发环境、指标计算或 DeepSpeed 支持, Hatch 允许你方便地管理这些. 不过, 最直接的安装方式通常还是配合 pip.

方案二: 使用 Pip(最通用、简单)

现代版本的 pip(21.3+)完全支持 pyproject.toml. 这是目前大多数 AI 开发者最常用的方式.

1. 基础安装(仅核心依赖):

```Bash
pip install -e .
```

-e 表示可编辑模式(Editable), 这对开发和微调 LLM 项目非常重要, 因为你会经常改动源码.

2. 全量安装(包括所有可选依赖):
这个项目定义了 dev、metrics 和 deepspeed 三个可选包. 如果你想"全都要", 执行:

```Bash
pip install -e ".[dev,metrics,deepspeed]"
``

方案三: 使用 uv(速度最快, 强烈推荐)

如果你希望在几秒钟内完成数以百计的依赖解析和安装(特别是像 torch 这样的大包), uv 是目前业界的明星工具.

安装 uv:

```Bash
pip install uv
```

一键同步环境:

```Bash
uv pip install -e ".[dev,metrics,deepspeed]"
```

方案四: 针对 AI 项目的专业建议(Conda + Pip)

由于 LLaMA-Factory 涉及复杂的显卡驱动、CUDA 和 C++ 编译(DeepSpeed 需要), 高级工程师通常会先用 Conda 隔离 Python 版本和基础库:

创建 Conda 环境:

```Bash
conda create -n llama_factory python=3.11  # 对应项目要求的 >=3.11.0
conda activate llama_factory
```

安装核心依赖:

```Bash
pip install -e ".[dev,metrics,deepspeed]"
```

高级工程师的避坑指南(关键点):

Python 版本限制: 注意配置文件中的 requires-python = ">=3.11.0". 如果你当前的 Python 版本是 3.10 或更低, 安装会报错. 请确保环境版本达标.

PyTorch 版本: 该项目要求 torch>=2.4.0. 如果你的机器有特定的 CUDA 版本需求, 建议先手动安装对应 CUDA 版本的 torch, 然后再运行项目的安装命令. 例如:

```Bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

动态版本: 配置中 dynamic = ["version"] 说明版本号是从源码中动态读取的. 安装时如果报错找不到版本, 通常是因为没有使用 -e . 或者当前路径不对.

Ruff 格式检查: 项目中配置了非常详尽的 tool.ruff. 如果你打算贡献代码, 安装完 dev 依赖后, 建议运行 `ruff check .` 来检查代码质量.

总结: 最简单有效的命令是 pip install -e ".[dev,metrics,deepspeed]". 如果你追求极致速度, 请使用 uv.

-----------------------------------------------------------------------------------------------

我使用的是 uv 进行安装, 现在我想知道 uv 安装的依赖到底使用了哪个版本, 我应该如何查看

使用 uv 安装依赖后, 查看具体版本的核心方法有以下几种. 作为高级开发人员, 我会根据你查看的需求(是看当前环境、看项目依赖树、还是看锁定文件)将方法分为三类:

1. 查看当前环境已安装的包(类似 pip)

如果你想直观地列出当前虚拟环境中所有已安装的包及其版本, 可以使用 uv 提供的 pip 兼容接口:

列出所有包:

```Bash
uv pip list
```

这会输出一个列表, 显示所有安装在当前环境(通常是 .venv)中的包和对应版本.

查看特定包的详细信息:

```Bash
uv pip show transformers
```

这会显示 transformers 的版本、安装路径、依赖项等详细元数据.

以 requirements 格式输出:

```Bash
uv pip freeze
```

2. 查看项目的依赖树(推荐)

如果你是按照项目模式(即通过 uv sync 或 uv add)进行的安装, uv 提供了一个非常强大的命令来查看依赖关系及其层级:

查看依赖树:

```Bash
uv tree
```

为什么推荐这个?
它不仅显示你直接安装的包(如 torch), 还会显示这些包带进来的间接依赖. 你能清楚地看到某个特定的包是因为哪个核心库的需求而被安装的, 以及它最终确定的版本.

查看包含开发依赖的完整树:

```Bash
uv tree --all
```

3. 查看锁定文件(底层确定的版本)

uv 在安装过程中会生成或更新一个 uv.lock 文件. 这是项目的"版本真理来源".

直接打开 uv.lock:

这是一个 TOML 格式的文件. 你可以直接用编辑器打开它, 搜索某个包名, 你会看到如下内容:

```Toml
[[package]]
name = "transformers"
version = "4.51.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "huggingface-hub" },
    { name = "numpy" },
    # ... 其他依赖
]
```

注意: uv.lock 记录了多平台兼容的精确版本, 它是确保你、你的同事以及生产环境版本完全一致的关键.

专家提示: uv pip vs uv 项目命令

如果你是用 uv pip install -e . 安装的: 请优先使用 uv pip list. 此时 uv 更多是作为 pip 的超快替代品.

如果你是用 uv sync 或 uv run 这种项目管理模式: 请优先使用 uv tree. 这是 uv 作为现代项目管理器的核心优势, 能帮你快速定位依赖冲突.

总结建议:
如果你想快速看一眼: 执行 uv pip list.
如果你想研究依赖冲突或层级: 执行 uv tree.

-----------------------------------------------------------------------------------------------

有一个疑问, 我先创建 Conda 环境:

```
conda create -n llama_factory python=3.12
conda activate llama_factory
```

然后使用命令 uv pip -v install -e ".[dev,metrics,deepspeed]" 安装了项目的依赖

我在创建 Conda 环境是就已经有一些 Python 包在里面了, uv pip 又安装了一些, 我应该如何区分哪些是真正的依赖, 哪些是 Conda 环境自带的

这是一个非常专业且敏锐的观察. 在高级开发工作流中, "混合环境"(Conda 提供底层环境 + uv/pip 提供 PyPI 依赖)是主流, 但也确实会带来"哪些包是谁带来的"这种混乱感.

作为高级开发人员, 我建议通过以下三个维度来清晰地剥离和区分这些依赖:

1. 从安装源进行区分(最直观的方法)

Conda 的 list 命令其实已经帮你标记了来源.

执行命令:

```Bash
conda list
```

如何解读结果:
在输出的列表里, 观察最后一列 Channel:

pypi: 如果这一列显示 pypi, 说明这些包是通过 uv pip 或 pip 安装的. 这些就是你项目的真实依赖.

defaults 或 conda-forge 或 空: 如果显示的是这些, 说明它们是 Conda 在执行 conda create 时预装的底层工具或基础库.

2. 使用 uv tree 隔离"噪声"(最精准的方法)

这是我最推荐的方法. uv pip install -e . 安装后, 你的项目是以"可编辑模式"存在的. uv tree 会以你的项目为根节点构建依赖树.

执行命令:

```Bash
uv tree
```

为什么这能解决你的疑问?

树内包: 所有出现在 llamafactory 树状结构下的包, 都是你在 pyproject.toml 中定义的真正依赖及其间接依赖.

树外包: 如果一个包(比如 wheel、setuptools、ca-certificates)没有出现在 uv tree 的输出中, 但出现在了 conda list 中, 那么它就是 Conda 环境自带的"噪声".

3. Conda 3.12 环境通常自带哪些"噪声"?

为了让你心里有数, 当你运行 conda create -n llama_factory python=3.12 时, Conda 默认会塞进大约 15-20 个基础包.

常见的 Conda 自带包清单(通常与项目依赖无关):

Python 运行环境: python, libffi, ncurses, openssl, readline, sqlite, tk, xz, zlib

包管理工具: pip, setuptools, wheel, certifi (根证书)

系统底层: ca-certificates, tzdata

而你的项目真实依赖通常是:
torch, transformers, datasets, accelerate, fastapi 等等.

4. 终极对比法: 导出环境快照

如果你想做非常严格的区分, 可以执行以下操作:

查看 Conda 带来的基础包(排除 PyPI):

```Bash
conda list --no-pip
```

这会只显示 Conda 渠道安装的包.

查看你刚才通过 uv 安装的包:

```Bash
uv pip freeze
```

这会显示所有通过 PyPI 协议安装的包.

专家建议:

在生产环境或需要提交 requirements.txt 时, 千万不要直接用 conda list -e > requirements.txt, 因为这会把 Conda 的底层路径和系统库也带进去.

正确的做法是:
利用 uv 的导出功能, 它只会导出你在 pyproject.toml 中定义的依赖逻辑:

```Bash
uv pip compile pyproject.toml -o requirements.txt
```

这样生成的 requirements.txt 才是纯净的、与项目真正相关的依赖清单, 去除了 Conda 环境本身的干扰.

总结:

想看谁安装的? 用 conda list 看 pypi 标签.

想看项目关联? 用 uv tree 看树状结构.

想看基础噪声? 用 conda list --no-pip.

-----------------------------------------------------------------------------------------------

```bash
$ conda list
# packages in environment at /root/miniconda3/envs/llamafactory:
#
# Name                     Version          Build            Channel
_libgcc_mutex              0.1              main
_openmp_mutex              5.1              1_gnu
bzip2                      1.0.8            h5eee18b_6
ca-certificates            2025.12.2        h06a4308_0
expat                      2.7.3            h7354ed3_4
ld_impl_linux-64           2.44             h153f514_2
libexpat                   2.7.3            h7354ed3_4
libffi                     3.4.4            h6a678d5_1
libgcc                     15.2.0           h69a1729_7
libgcc-ng                  15.2.0           h166f726_7
libgomp                    15.2.0           h4751f2c_7
libnsl                     2.0.0            h5eee18b_0
libstdcxx                  15.2.0           h39759b7_7
libstdcxx-ng               15.2.0           hc03a8fd_7
libuuid                    1.41.5           h5eee18b_0
libxcb                     1.17.0           h9b100fa_0
libzlib                    1.3.1            hb25bd0a_0
ncurses                    6.5              h7934f7d_0
openssl                    3.0.18           hd6dcaed_0
pip                        25.3             pyhc872135_0
pthread-stubs              0.3              h0ce48e5_1
python                     3.12.12          hd17a9e1_1
readline                   8.3              hc2a1206_0
setuptools                 80.9.0           py312h06a4308_0
sqlite                     3.51.0           h2a70700_0
tk                         8.6.15           h54e0aa7_0
tzdata                     2025b            h04d1e81_0
wheel                      0.45.1           py312h06a4308_0
xorg-libx11                1.8.12           h9b100fa_1
xorg-libxau                1.0.12           h9b100fa_0
xorg-libxdmcp              1.1.5            h9b100fa_0
xorg-xorgproto             2024.1           h5eee18b_1
xz                         5.6.4            h5eee18b_1
zlib                       1.3.1            hb25bd0a_0
$
$ conda list --no-pip
# packages in environment at /root/miniconda3/envs/llamafactory:
#
# Name                     Version          Build            Channel
_libgcc_mutex              0.1              main
_openmp_mutex              5.1              1_gnu
bzip2                      1.0.8            h5eee18b_6
ca-certificates            2025.12.2        h06a4308_0
expat                      2.7.3            h7354ed3_4
ld_impl_linux-64           2.44             h153f514_2
libexpat                   2.7.3            h7354ed3_4
libffi                     3.4.4            h6a678d5_1
libgcc                     15.2.0           h69a1729_7
libgcc-ng                  15.2.0           h166f726_7
libgomp                    15.2.0           h4751f2c_7
libnsl                     2.0.0            h5eee18b_0
libstdcxx                  15.2.0           h39759b7_7
libstdcxx-ng               15.2.0           hc03a8fd_7
libuuid                    1.41.5           h5eee18b_0
libxcb                     1.17.0           h9b100fa_0
libzlib                    1.3.1            hb25bd0a_0
ncurses                    6.5              h7934f7d_0
openssl                    3.0.18           hd6dcaed_0
pip                        25.3             pyhc872135_0
pthread-stubs              0.3              h0ce48e5_1
python                     3.12.12          hd17a9e1_1
readline                   8.3              hc2a1206_0
setuptools                 80.9.0           py312h06a4308_0
sqlite                     3.51.0           h2a70700_0
tk                         8.6.15           h54e0aa7_0
tzdata                     2025b            h04d1e81_0
wheel                      0.45.1           py312h06a4308_0
xorg-libx11                1.8.12           h9b100fa_1
xorg-libxau                1.0.12           h9b100fa_0
xorg-libxdmcp              1.1.5            h9b100fa_0
xorg-xorgproto             2024.1           h5eee18b_1
xz                         5.6.4            h5eee18b_1
zlib                       1.3.1            hb25bd0a_0
$

####################################################################################

$ conda list
# packages in environment at /root/miniconda3/envs/llamafactory:
#
# Name                      Version          Build            Channel
_libgcc_mutex               0.1              main
_openmp_mutex               5.1              1_gnu
accelerate                  1.11.0           pypi_0           pypi
aiofiles                    24.1.0           pypi_0           pypi
aiohappyeyeballs            2.6.1            pypi_0           pypi
aiohttp                     3.13.3           pypi_0           pypi
aiosignal                   1.4.0            pypi_0           pypi
annotated-doc               0.0.4            pypi_0           pypi
annotated-types             0.7.0            pypi_0           pypi
antlr4-python3-runtime      4.9.3            pypi_0           pypi
anyio                       4.12.0           pypi_0           pypi
attrs                       25.4.0           pypi_0           pypi
av                          16.0.1           pypi_0           pypi
backports-zstd              1.3.0            pypi_0           pypi
brotli                      1.2.0            pypi_0           pypi
build                       1.3.0            pypi_0           pypi
bzip2                       1.0.8            h5eee18b_6
ca-certificates             2025.12.2        h06a4308_0
certifi                     2026.1.4         pypi_0           pypi
cffi                        2.0.0            pypi_0           pypi
cfgv                        3.5.0            pypi_0           pypi
charset-normalizer          3.4.4            pypi_0           pypi
click                       8.3.1            pypi_0           pypi
contourpy                   1.3.3            pypi_0           pypi
cryptography                46.0.3           pypi_0           pypi
cycler                      0.12.1           pypi_0           pypi
datasets                    4.0.0            pypi_0           pypi
deepspeed                   0.16.9           pypi_0           pypi
dill                        0.3.8            pypi_0           pypi
distlib                     0.4.0            pypi_0           pypi
docstring-parser            0.17.0           pypi_0           pypi
einops                      0.8.1            pypi_0           pypi
expat                       2.7.3            h7354ed3_4
fastapi                     0.128.0          pypi_0           pypi
ffmpy                       1.0.0            pypi_0           pypi
filelock                    3.20.2           pypi_0           pypi
fire                        0.7.1            pypi_0           pypi
fonttools                   4.61.1           pypi_0           pypi
frozenlist                  1.8.0            pypi_0           pypi
fsspec                      2025.3.0         pypi_0           pypi
gradio                      5.50.0           pypi_0           pypi
gradio-client               1.14.0           pypi_0           pypi
groovy                      0.1.2            pypi_0           pypi
h11                         0.16.0           pypi_0           pypi
hatch                       1.16.2           pypi_0           pypi
hatchling                   1.28.0           pypi_0           pypi
hf-transfer                 0.1.9            pypi_0           pypi
hf-xet                      1.2.0            pypi_0           pypi
hjson                       3.1.0            pypi_0           pypi
httpcore                    1.0.9            pypi_0           pypi
httpx                       0.28.1           pypi_0           pypi
huggingface-hub             0.36.0           pypi_0           pypi
hyperlink                   21.0.0           pypi_0           pypi
identify                    2.6.15           pypi_0           pypi
idna                        3.11             pypi_0           pypi
iniconfig                   2.3.0            pypi_0           pypi
jaraco-classes              3.4.0            pypi_0           pypi
jaraco-context              6.0.2            pypi_0           pypi
jaraco-functools            4.4.0            pypi_0           pypi
jeepney                     0.9.0            pypi_0           pypi
jieba                       0.42.1           pypi_0           pypi
jinja2                      3.1.6            pypi_0           pypi
joblib                      1.5.3            pypi_0           pypi
keyring                     25.7.0           pypi_0           pypi
kiwisolver                  1.4.9            pypi_0           pypi
ld_impl_linux-64            2.44             h153f514_2
libexpat                    2.7.3            h7354ed3_4
libffi                      3.4.4            h6a678d5_1
libgcc                      15.2.0           h69a1729_7
libgcc-ng                   15.2.0           h166f726_7
libgomp                     15.2.0           h4751f2c_7
libnsl                      2.0.0            h5eee18b_0
libstdcxx                   15.2.0           h39759b7_7
libstdcxx-ng                15.2.0           hc03a8fd_7
libuuid                     1.41.5           h5eee18b_0
libxcb                      1.17.0           h9b100fa_0
libzlib                     1.3.1            hb25bd0a_0
llamafactory                0.9.4            pypi_0           pypi
markdown-it-py              4.0.0            pypi_0           pypi
markupsafe                  3.0.3            pypi_0           pypi
matplotlib                  3.10.8           pypi_0           pypi
mdurl                       0.1.2            pypi_0           pypi
modelscope                  1.33.0           pypi_0           pypi
more-itertools              10.8.0           pypi_0           pypi
mpmath                      1.3.0            pypi_0           pypi
msgpack                     1.1.2            pypi_0           pypi
multidict                   6.7.0            pypi_0           pypi
multiprocess                0.70.16          pypi_0           pypi
ncurses                     6.5              h7934f7d_0
networkx                    3.6.1            pypi_0           pypi
ninja                       1.13.0           pypi_0           pypi
nltk                        3.9.2            pypi_0           pypi
nodeenv                     1.10.0           pypi_0           pypi
numpy                       2.4.0            pypi_0           pypi
nvidia-cublas-cu12          12.8.4.1         pypi_0           pypi
nvidia-cuda-cupti-cu12      12.8.90          pypi_0           pypi
nvidia-cuda-nvrtc-cu12      12.8.93          pypi_0           pypi
nvidia-cuda-runtime-cu12    12.8.90          pypi_0           pypi
nvidia-cudnn-cu12           9.10.2.21        pypi_0           pypi
nvidia-cufft-cu12           11.3.3.83        pypi_0           pypi
nvidia-cufile-cu12          1.13.1.3         pypi_0           pypi
nvidia-curand-cu12          10.3.9.90        pypi_0           pypi
nvidia-cusolver-cu12        11.7.3.90        pypi_0           pypi
nvidia-cusparse-cu12        12.5.8.93        pypi_0           pypi
nvidia-cusparselt-cu12      0.7.1            pypi_0           pypi
nvidia-nccl-cu12            2.27.5           pypi_0           pypi
nvidia-nvjitlink-cu12       12.8.93          pypi_0           pypi
nvidia-nvshmem-cu12         3.3.20           pypi_0           pypi
nvidia-nvtx-cu12            12.8.90          pypi_0           pypi
omegaconf                   2.3.0            pypi_0           pypi
openssl                     3.0.18           hd6dcaed_0
orjson                      3.11.5           pypi_0           pypi
packaging                   25.0             pypi_0           pypi
pandas                      2.3.3            pypi_0           pypi
pathspec                    0.12.1           pypi_0           pypi
peft                        0.17.1           pypi_0           pypi
pexpect                     4.9.0            pypi_0           pypi
pillow                      11.3.0           pypi_0           pypi
pip                         25.3             pyhc872135_0
platformdirs                4.5.1            pypi_0           pypi
pluggy                      1.6.0            pypi_0           pypi
pre-commit                  4.5.1            pypi_0           pypi
propcache                   0.4.1            pypi_0           pypi
protobuf                    6.33.2           pypi_0           pypi
psutil                      7.2.1            pypi_0           pypi
pthread-stubs               0.3              h0ce48e5_1
ptyprocess                  0.7.0            pypi_0           pypi
py-cpuinfo                  9.0.0            pypi_0           pypi
pyarrow                     22.0.0           pypi_0           pypi
pycparser                   2.23             pypi_0           pypi
pydantic                    2.12.3           pypi_0           pypi
pydantic-core               2.41.4           pypi_0           pypi
pydub                       0.25.1           pypi_0           pypi
pygments                    2.19.2           pypi_0           pypi
pyparsing                   3.3.1            pypi_0           pypi
pyproject-hooks             1.2.0            pypi_0           pypi
pytest                      9.0.2            pypi_0           pypi
python                      3.12.12          hd17a9e1_1
python-dateutil             2.9.0.post0      pypi_0           pypi
python-multipart            0.0.21           pypi_0           pypi
pytz                        2025.2           pypi_0           pypi
pyyaml                      6.0.3            pypi_0           pypi
readline                    8.3              hc2a1206_0
regex                       2025.11.3        pypi_0           pypi
requests                    2.32.5           pypi_0           pypi
rich                        14.2.0           pypi_0           pypi
rouge-chinese               1.0.3            pypi_0           pypi
ruff                        0.14.10          pypi_0           pypi
safehttpx                   0.1.7            pypi_0           pypi
safetensors                 0.7.0            pypi_0           pypi
scipy                       1.16.3           pypi_0           pypi
secretstorage               3.5.0            pypi_0           pypi
semantic-version            2.10.0           pypi_0           pypi
sentencepiece               0.2.1            pypi_0           pypi
setuptools                  80.9.0           py312h06a4308_0
shellingham                 1.5.4            pypi_0           pypi
shtab                       1.8.0            pypi_0           pypi
six                         1.17.0           pypi_0           pypi
sqlite                      3.51.0           h2a70700_0
sse-starlette               3.1.2            pypi_0           pypi
starlette                   0.50.0           pypi_0           pypi
sympy                       1.14.0           pypi_0           pypi
termcolor                   3.3.0            pypi_0           pypi
tiktoken                    0.12.0           pypi_0           pypi
tk                          8.6.15           h54e0aa7_0
tokenizers                  0.22.1           pypi_0           pypi
tomli-w                     1.2.0            pypi_0           pypi
tomlkit                     0.13.3           pypi_0           pypi
torch                       2.9.1            pypi_0           pypi
torchaudio                  2.9.1            pypi_0           pypi
torchdata                   0.11.0           pypi_0           pypi
torchvision                 0.24.1           pypi_0           pypi
tqdm                        4.67.1           pypi_0           pypi
transformers                4.57.1           pypi_0           pypi
triton                      3.5.1            pypi_0           pypi
trl                         0.24.0           pypi_0           pypi
trove-classifiers           2025.12.1.14     pypi_0           pypi
typer                       0.21.0           pypi_0           pypi
typing-extensions           4.15.0           pypi_0           pypi
typing-inspection           0.4.2            pypi_0           pypi
tyro                        0.8.14           pypi_0           pypi
tzdata                      2025.3           pypi_0           pypi
urllib3                     2.6.2            pypi_0           pypi
userpath                    1.9.2            pypi_0           pypi
uv                          0.9.21           pypi_0           pypi
uvicorn                     0.40.0           pypi_0           pypi
virtualenv                  20.35.4          pypi_0           pypi
websockets                  15.0.1           pypi_0           pypi
wheel                       0.45.1           py312h06a4308_0
xorg-libx11                 1.8.12           h9b100fa_1
xorg-libxau                 1.0.12           h9b100fa_0
xorg-libxdmcp               1.1.5            h9b100fa_0
xorg-xorgproto              2024.1           h5eee18b_1
xxhash                      3.6.0            pypi_0           pypi
xz                          5.6.4            h5eee18b_1
yarl                        1.22.0           pypi_0           pypi
zlib                        1.3.1            hb25bd0a_0
$

$ uv tree --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --all-groups
Using CPython 3.12.12 interpreter at: /root/miniconda3/envs/llamafactory/bin/python3
Resolved 146 packages in 2m 44s
llamafactory
├── accelerate v1.11.0
│   ├── huggingface-hub v0.36.0
│   │   ├── filelock v3.20.2
│   │   ├── fsspec v2025.3.0
│   │   │   └── aiohttp v3.13.3 (extra: http)
│   │   │       ├── aiohappyeyeballs v2.6.1
│   │   │       ├── aiosignal v1.4.0
│   │   │       │   ├── frozenlist v1.8.0
│   │   │       │   └── typing-extensions v4.15.0
│   │   │       ├── attrs v25.4.0
│   │   │       ├── frozenlist v1.8.0
│   │   │       ├── multidict v6.7.0
│   │   │       ├── propcache v0.4.1
│   │   │       └── yarl v1.22.0
│   │   │           ├── idna v3.11
│   │   │           ├── multidict v6.7.0
│   │   │           └── propcache v0.4.1
│   │   ├── hf-xet v1.2.0
│   │   ├── packaging v25.0
│   │   ├── pyyaml v6.0.3
│   │   ├── requests v2.32.5
│   │   │   ├── certifi v2026.1.4
│   │   │   ├── charset-normalizer v3.4.4
│   │   │   ├── idna v3.11
│   │   │   └── urllib3 v2.6.2
│   │   ├── tqdm v4.67.1
│   │   └── typing-extensions v4.15.0
│   ├── numpy v2.4.0
│   ├── packaging v25.0
│   ├── psutil v7.2.1
│   ├── pyyaml v6.0.3
│   ├── safetensors v0.7.0
│   └── torch v2.9.1
│       ├── filelock v3.20.2
│       ├── fsspec v2025.3.0 (*)
│       ├── jinja2 v3.1.6
│       │   └── markupsafe v3.0.3
│       ├── networkx v3.6.1
│       ├── nvidia-cublas-cu12 v12.8.4.1
│       ├── nvidia-cuda-cupti-cu12 v12.8.90
│       ├── nvidia-cuda-nvrtc-cu12 v12.8.93
│       ├── nvidia-cuda-runtime-cu12 v12.8.90
│       ├── nvidia-cudnn-cu12 v9.10.2.21
│       │   └── nvidia-cublas-cu12 v12.8.4.1
│       ├── nvidia-cufft-cu12 v11.3.3.83
│       │   └── nvidia-nvjitlink-cu12 v12.8.93
│       ├── nvidia-cufile-cu12 v1.13.1.3
│       ├── nvidia-curand-cu12 v10.3.9.90
│       ├── nvidia-cusolver-cu12 v11.7.3.90
│       │   ├── nvidia-cublas-cu12 v12.8.4.1
│       │   ├── nvidia-cusparse-cu12 v12.5.8.93
│       │   │   └── nvidia-nvjitlink-cu12 v12.8.93
│       │   └── nvidia-nvjitlink-cu12 v12.8.93
│       ├── nvidia-cusparse-cu12 v12.5.8.93 (*)
│       ├── nvidia-cusparselt-cu12 v0.7.1
│       ├── nvidia-nccl-cu12 v2.27.5
│       ├── nvidia-nvjitlink-cu12 v12.8.93
│       ├── nvidia-nvshmem-cu12 v3.3.20
│       ├── nvidia-nvtx-cu12 v12.8.90
│       ├── setuptools v80.9.0
│       ├── sympy v1.14.0
│       │   └── mpmath v1.3.0
│       ├── triton v3.5.1
│       └── typing-extensions v4.15.0
├── av v16.0.1
├── datasets v4.0.0
│   ├── dill v0.3.8
│   ├── filelock v3.20.2
│   ├── fsspec[http] v2025.3.0 (*)
│   ├── huggingface-hub v0.36.0 (*)
│   ├── multiprocess v0.70.16
│   │   └── dill v0.3.8
│   ├── numpy v2.4.0
│   ├── packaging v25.0
│   ├── pandas v2.3.3
│   │   ├── numpy v2.4.0
│   │   ├── python-dateutil v2.9.0.post0
│   │   │   └── six v1.17.0
│   │   ├── pytz v2025.2
│   │   └── tzdata v2025.3
│   ├── pyarrow v22.0.0
│   ├── pyyaml v6.0.3
│   ├── requests v2.32.5 (*)
│   ├── tqdm v4.67.1
│   └── xxhash v3.6.0
├── einops v0.8.1
├── fastapi v0.128.0
│   ├── annotated-doc v0.0.4
│   ├── pydantic v2.12.3
│   │   ├── annotated-types v0.7.0
│   │   ├── pydantic-core v2.41.4
│   │   │   └── typing-extensions v4.15.0
│   │   ├── typing-extensions v4.15.0
│   │   └── typing-inspection v0.4.2
│   │       └── typing-extensions v4.15.0
│   ├── starlette v0.50.0
│   │   ├── anyio v4.12.0
│   │   │   ├── idna v3.11
│   │   │   └── typing-extensions v4.15.0
│   │   └── typing-extensions v4.15.0
│   └── typing-extensions v4.15.0
├── fire v0.7.1
│   └── termcolor v3.3.0
├── gradio v5.50.0
│   ├── aiofiles v24.1.0
│   ├── anyio v4.12.0 (*)
│   ├── brotli v1.2.0
│   ├── fastapi v0.128.0 (*)
│   ├── ffmpy v1.0.0
│   ├── gradio-client v1.14.0
│   │   ├── fsspec v2025.3.0 (*)
│   │   ├── httpx v0.28.1
│   │   │   ├── anyio v4.12.0 (*)
│   │   │   ├── certifi v2026.1.4
│   │   │   ├── httpcore v1.0.9
│   │   │   │   ├── certifi v2026.1.4
│   │   │   │   └── h11 v0.16.0
│   │   │   └── idna v3.11
│   │   ├── huggingface-hub v0.36.0 (*)
│   │   ├── packaging v25.0
│   │   ├── typing-extensions v4.15.0
│   │   └── websockets v15.0.1
│   ├── groovy v0.1.2
│   ├── httpx v0.28.1 (*)
│   ├── huggingface-hub v0.36.0 (*)
│   ├── jinja2 v3.1.6 (*)
│   ├── markupsafe v3.0.3
│   ├── numpy v2.4.0
│   ├── orjson v3.11.5
│   ├── packaging v25.0
│   ├── pandas v2.3.3 (*)
│   ├── pillow v11.3.0
│   ├── pydantic v2.12.3 (*)
│   ├── pydub v0.25.1
│   ├── python-multipart v0.0.21
│   ├── pyyaml v6.0.3
│   ├── ruff v0.14.10
│   ├── safehttpx v0.1.7
│   │   └── httpx v0.28.1 (*)
│   ├── semantic-version v2.10.0
│   ├── starlette v0.50.0 (*)
│   ├── tomlkit v0.13.3
│   ├── typer v0.21.0
│   │   ├── click v8.3.1
│   │   ├── rich v14.2.0
│   │   │   ├── markdown-it-py v4.0.0
│   │   │   │   └── mdurl v0.1.2
│   │   │   └── pygments v2.19.2
│   │   ├── shellingham v1.5.4
│   │   └── typing-extensions v4.15.0
│   ├── typing-extensions v4.15.0
│   └── uvicorn v0.40.0
│       ├── click v8.3.1
│       └── h11 v0.16.0
├── hf-transfer v0.1.9
├── matplotlib v3.10.8
│   ├── contourpy v1.3.3
│   │   └── numpy v2.4.0
│   ├── cycler v0.12.1
│   ├── fonttools v4.61.1
│   ├── kiwisolver v1.4.9
│   ├── numpy v2.4.0
│   ├── packaging v25.0
│   ├── pillow v11.3.0
│   ├── pyparsing v3.3.1
│   └── python-dateutil v2.9.0.post0 (*)
├── modelscope v1.33.0
│   ├── filelock v3.20.2
│   ├── requests v2.32.5 (*)
│   ├── setuptools v80.9.0
│   ├── tqdm v4.67.1
│   └── urllib3 v2.6.2
├── numpy v2.4.0
├── omegaconf v2.3.0
│   ├── antlr4-python3-runtime v4.9.3
│   └── pyyaml v6.0.3
├── packaging v25.0
├── pandas v2.3.3 (*)
├── peft v0.17.1
│   ├── accelerate v1.11.0 (*)
│   ├── huggingface-hub v0.36.0 (*)
│   ├── numpy v2.4.0
│   ├── packaging v25.0
│   ├── psutil v7.2.1
│   ├── pyyaml v6.0.3
│   ├── safetensors v0.7.0
│   ├── torch v2.9.1 (*)
│   ├── tqdm v4.67.1
│   └── transformers v4.57.1
│       ├── filelock v3.20.2
│       ├── huggingface-hub v0.36.0 (*)
│       ├── numpy v2.4.0
│       ├── packaging v25.0
│       ├── pyyaml v6.0.3
│       ├── regex v2025.11.3
│       ├── requests v2.32.5 (*)
│       ├── safetensors v0.7.0
│       ├── tokenizers v0.22.1
│       │   └── huggingface-hub v0.36.0 (*)
│       └── tqdm v4.67.1
├── protobuf v6.33.2
├── pydantic v2.12.3 (*)
├── pyyaml v6.0.3
├── safetensors v0.7.0
├── scipy v1.16.3
│   └── numpy v2.4.0
├── sentencepiece v0.2.1
├── sse-starlette v3.1.2
│   ├── anyio v4.12.0 (*)
│   └── starlette v0.50.0 (*)
├── tiktoken v0.12.0
│   ├── regex v2025.11.3
│   └── requests v2.32.5 (*)
├── torch v2.9.1 (*)
├── torchaudio v2.9.1
│   └── torch v2.9.1 (*)
├── torchdata v0.11.0
│   ├── requests v2.32.5 (*)
│   ├── torch v2.9.1 (*)
│   └── urllib3 v2.6.2
├── torchvision v0.24.1
│   ├── numpy v2.4.0
│   ├── pillow v11.3.0
│   └── torch v2.9.1 (*)
├── transformers v4.57.1 (*)
├── trl v0.24.0
│   ├── accelerate v1.11.0 (*)
│   ├── datasets v4.0.0 (*)
│   └── transformers v4.57.1 (*)
├── tyro v0.8.14
│   ├── docstring-parser v0.17.0
│   ├── rich v14.2.0 (*)
│   ├── shtab v1.8.0
│   └── typing-extensions v4.15.0
├── uvicorn v0.40.0 (*)
├── deepspeed v0.16.9 (extra: deepspeed)
│   ├── einops v0.8.1
│   ├── hjson v3.1.0
│   ├── msgpack v1.1.2
│   ├── ninja v1.13.0
│   ├── numpy v2.4.0
│   ├── packaging v25.0
│   ├── psutil v7.2.1
│   ├── py-cpuinfo v9.0.0
│   ├── pydantic v2.12.3 (*)
│   ├── torch v2.9.1 (*)
│   └── tqdm v4.67.1
├── build v1.3.0 (extra: dev)
│   ├── packaging v25.0
│   └── pyproject-hooks v1.2.0
├── pre-commit v4.5.1 (extra: dev)
│   ├── cfgv v3.5.0
│   ├── identify v2.6.15
│   ├── nodeenv v1.10.0
│   ├── pyyaml v6.0.3
│   └── virtualenv v20.35.4
│       ├── distlib v0.4.0
│       ├── filelock v3.20.2
│       └── platformdirs v4.5.1
├── pytest v9.0.2 (extra: dev)
│   ├── iniconfig v2.3.0
│   ├── packaging v25.0
│   ├── pluggy v1.6.0
│   └── pygments v2.19.2
├── ruff v0.14.10 (extra: dev)
├── jieba v0.42.1 (extra: metrics)
├── nltk v3.9.2 (extra: metrics)
│   ├── click v8.3.1
│   ├── joblib v1.5.3
│   ├── regex v2025.11.3
│   └── tqdm v4.67.1
└── rouge-chinese v1.0.3 (extra: metrics)
    └── six v1.17.0
(*) Package tree already displayed
$

$ uv pip compile --index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple pyproject.toml --all-extras -o requirements.txt
Resolved 143 packages in 1.20s
# This file was autogenerated by uv via the following command:
#    uv pip compile pyproject.toml --all-extras -o requirements.txt
accelerate==1.11.0
    # via
    #   llamafactory (pyproject.toml)
    #   peft
    #   trl
aiofiles==24.1.0
    # via gradio
aiohappyeyeballs==2.6.1
    # via aiohttp
aiohttp==3.13.3
    # via fsspec
aiosignal==1.4.0
    # via aiohttp
annotated-doc==0.0.4
    # via fastapi
annotated-types==0.7.0
    # via pydantic
antlr4-python3-runtime==4.9.3
    # via omegaconf
anyio==4.12.0
    # via
    #   gradio
    #   httpx
    #   sse-starlette
    #   starlette
attrs==25.4.0
    # via aiohttp
av==16.0.1
    # via llamafactory (pyproject.toml)
brotli==1.2.0
    # via gradio
build==1.3.0
    # via llamafactory (pyproject.toml)
certifi==2026.1.4
    # via
    #   httpcore
    #   httpx
    #   requests
cfgv==3.5.0
    # via pre-commit
charset-normalizer==3.4.4
    # via requests
click==8.3.1
    # via
    #   nltk
    #   typer
    #   uvicorn
contourpy==1.3.3
    # via matplotlib
cycler==0.12.1
    # via matplotlib
datasets==4.0.0
    # via
    #   llamafactory (pyproject.toml)
    #   trl
deepspeed==0.16.9
    # via llamafactory (pyproject.toml)
dill==0.3.8
    # via
    #   datasets
    #   multiprocess
distlib==0.4.0
    # via virtualenv
docstring-parser==0.17.0
    # via tyro
einops==0.8.1
    # via
    #   llamafactory (pyproject.toml)
    #   deepspeed
fastapi==0.128.0
    # via
    #   llamafactory (pyproject.toml)
    #   gradio
ffmpy==1.0.0
    # via gradio
filelock==3.20.2
    # via
    #   datasets
    #   huggingface-hub
    #   modelscope
    #   torch
    #   transformers
    #   virtualenv
fire==0.7.1
    # via llamafactory (pyproject.toml)
fonttools==4.61.1
    # via matplotlib
frozenlist==1.8.0
    # via
    #   aiohttp
    #   aiosignal
fsspec==2025.3.0
    # via
    #   datasets
    #   gradio-client
    #   huggingface-hub
    #   torch
gradio==5.50.0
    # via llamafactory (pyproject.toml)
gradio-client==1.14.0
    # via gradio
groovy==0.1.2
    # via gradio
h11==0.16.0
    # via
    #   httpcore
    #   uvicorn
hf-transfer==0.1.9
    # via llamafactory (pyproject.toml)
hf-xet==1.2.0
    # via huggingface-hub
hjson==3.1.0
    # via deepspeed
httpcore==1.0.9
    # via httpx
httpx==0.28.1
    # via
    #   gradio
    #   gradio-client
    #   safehttpx
huggingface-hub==0.36.0
    # via
    #   accelerate
    #   datasets
    #   gradio
    #   gradio-client
    #   peft
    #   tokenizers
    #   transformers
identify==2.6.15
    # via pre-commit
idna==3.11
    # via
    #   anyio
    #   httpx
    #   requests
    #   yarl
iniconfig==2.3.0
    # via pytest
jieba==0.42.1
    # via llamafactory (pyproject.toml)
jinja2==3.1.6
    # via
    #   gradio
    #   torch
joblib==1.5.3
    # via nltk
kiwisolver==1.4.9
    # via matplotlib
markdown-it-py==4.0.0
    # via rich
markupsafe==3.0.3
    # via
    #   gradio
    #   jinja2
matplotlib==3.10.8
    # via llamafactory (pyproject.toml)
mdurl==0.1.2
    # via markdown-it-py
modelscope==1.33.0
    # via llamafactory (pyproject.toml)
mpmath==1.3.0
    # via sympy
msgpack==1.1.2
    # via deepspeed
multidict==6.7.0
    # via
    #   aiohttp
    #   yarl
multiprocess==0.70.16
    # via datasets
networkx==3.6.1
    # via torch
ninja==1.13.0
    # via deepspeed
nltk==3.9.2
    # via llamafactory (pyproject.toml)
nodeenv==1.10.0
    # via pre-commit
numpy==2.4.0
    # via
    #   llamafactory (pyproject.toml)
    #   accelerate
    #   contourpy
    #   datasets
    #   deepspeed
    #   gradio
    #   matplotlib
    #   pandas
    #   peft
    #   scipy
    #   torchvision
    #   transformers
nvidia-cublas-cu12==12.8.4.1
    # via
    #   nvidia-cudnn-cu12
    #   nvidia-cusolver-cu12
    #   torch
nvidia-cuda-cupti-cu12==12.8.90
    # via torch
nvidia-cuda-nvrtc-cu12==12.8.93
    # via torch
nvidia-cuda-runtime-cu12==12.8.90
    # via torch
nvidia-cudnn-cu12==9.10.2.21
    # via torch
nvidia-cufft-cu12==11.3.3.83
    # via torch
nvidia-cufile-cu12==1.13.1.3
    # via torch
nvidia-curand-cu12==10.3.9.90
    # via torch
nvidia-cusolver-cu12==11.7.3.90
    # via torch
nvidia-cusparse-cu12==12.5.8.93
    # via
    #   nvidia-cusolver-cu12
    #   torch
nvidia-cusparselt-cu12==0.7.1
    # via torch
nvidia-nccl-cu12==2.27.5
    # via torch
nvidia-nvjitlink-cu12==12.8.93
    # via
    #   nvidia-cufft-cu12
    #   nvidia-cusolver-cu12
    #   nvidia-cusparse-cu12
    #   torch
nvidia-nvshmem-cu12==3.3.20
    # via torch
nvidia-nvtx-cu12==12.8.90
    # via torch
omegaconf==2.3.0
    # via llamafactory (pyproject.toml)
orjson==3.11.5
    # via gradio
packaging==25.0
    # via
    #   llamafactory (pyproject.toml)
    #   accelerate
    #   build
    #   datasets
    #   deepspeed
    #   gradio
    #   gradio-client
    #   huggingface-hub
    #   matplotlib
    #   peft
    #   pytest
    #   transformers
pandas==2.3.3
    # via
    #   llamafactory (pyproject.toml)
    #   datasets
    #   gradio
peft==0.17.1
    # via llamafactory (pyproject.toml)
pillow==11.3.0
    # via
    #   gradio
    #   matplotlib
    #   torchvision
platformdirs==4.5.1
    # via virtualenv
pluggy==1.6.0
    # via pytest
pre-commit==4.5.1
    # via llamafactory (pyproject.toml)
propcache==0.4.1
    # via
    #   aiohttp
    #   yarl
protobuf==6.33.2
    # via llamafactory (pyproject.toml)
psutil==7.2.1
    # via
    #   accelerate
    #   deepspeed
    #   peft
py-cpuinfo==9.0.0
    # via deepspeed
pyarrow==22.0.0
    # via datasets
pydantic==2.12.3
    # via
    #   llamafactory (pyproject.toml)
    #   deepspeed
    #   fastapi
    #   gradio
pydantic-core==2.41.4
    # via pydantic
pydub==0.25.1
    # via gradio
pygments==2.19.2
    # via
    #   pytest
    #   rich
pyparsing==3.3.1
    # via matplotlib
pyproject-hooks==1.2.0
    # via build
pytest==9.0.2
    # via llamafactory (pyproject.toml)
python-dateutil==2.9.0.post0
    # via
    #   matplotlib
    #   pandas
python-multipart==0.0.21
    # via gradio
pytz==2025.2
    # via pandas
pyyaml==6.0.3
    # via
    #   llamafactory (pyproject.toml)
    #   accelerate
    #   datasets
    #   gradio
    #   huggingface-hub
    #   omegaconf
    #   peft
    #   pre-commit
    #   transformers
regex==2025.11.3
    # via
    #   nltk
    #   tiktoken
    #   transformers
requests==2.32.5
    # via
    #   datasets
    #   huggingface-hub
    #   modelscope
    #   tiktoken
    #   torchdata
    #   transformers
rich==14.2.0
    # via
    #   typer
    #   tyro
rouge-chinese==1.0.3
    # via llamafactory (pyproject.toml)
ruff==0.14.10
    # via
    #   llamafactory (pyproject.toml)
    #   gradio
safehttpx==0.1.7
    # via gradio
safetensors==0.7.0
    # via
    #   llamafactory (pyproject.toml)
    #   accelerate
    #   peft
    #   transformers
scipy==1.16.3
    # via llamafactory (pyproject.toml)
semantic-version==2.10.0
    # via gradio
sentencepiece==0.2.1
    # via llamafactory (pyproject.toml)
setuptools==80.9.0
    # via
    #   modelscope
    #   torch
shellingham==1.5.4
    # via typer
shtab==1.8.0
    # via tyro
six==1.17.0
    # via
    #   python-dateutil
    #   rouge-chinese
sse-starlette==3.1.2
    # via llamafactory (pyproject.toml)
starlette==0.50.0
    # via
    #   fastapi
    #   gradio
    #   sse-starlette
sympy==1.14.0
    # via torch
termcolor==3.3.0
    # via fire
tiktoken==0.12.0
    # via llamafactory (pyproject.toml)
tokenizers==0.22.1
    # via transformers
tomlkit==0.13.3
    # via gradio
torch==2.9.1
    # via
    #   llamafactory (pyproject.toml)
    #   accelerate
    #   deepspeed
    #   peft
    #   torchaudio
    #   torchdata
    #   torchvision
torchaudio==2.9.1
    # via llamafactory (pyproject.toml)
torchdata==0.11.0
    # via llamafactory (pyproject.toml)
torchvision==0.24.1
    # via llamafactory (pyproject.toml)
tqdm==4.67.1
    # via
    #   datasets
    #   deepspeed
    #   huggingface-hub
    #   modelscope
    #   nltk
    #   peft
    #   transformers
transformers==4.57.1
    # via
    #   llamafactory (pyproject.toml)
    #   peft
    #   trl
triton==3.5.1
    # via torch
trl==0.24.0
    # via llamafactory (pyproject.toml)
typer==0.21.0
    # via gradio
typing-extensions==4.15.0
    # via
    #   aiosignal
    #   anyio
    #   fastapi
    #   gradio
    #   gradio-client
    #   huggingface-hub
    #   pydantic
    #   pydantic-core
    #   starlette
    #   torch
    #   typer
    #   typing-inspection
    #   tyro
typing-inspection==0.4.2
    # via pydantic
tyro==0.8.14
    # via llamafactory (pyproject.toml)
tzdata==2025.3
    # via pandas
urllib3==2.6.2
    # via
    #   modelscope
    #   requests
    #   torchdata
uvicorn==0.40.0
    # via
    #   llamafactory (pyproject.toml)
    #   gradio
virtualenv==20.35.4
    # via pre-commit
websockets==15.0.1
    # via gradio-client
xxhash==3.6.0
    # via datasets
yarl==1.22.0
    # via aiohttp
$

```

-----------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------



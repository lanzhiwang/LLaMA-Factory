# Writing your `pyproject.toml`
编写你的 `pyproject.toml` 文件

* https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

`pyproject.toml` is a configuration file used by packaging tools, as well as other tools such as linters, type checkers, etc. There are three possible TOML tables in this file.
`pyproject.toml` 是打包工具以及其他工具(如代码检查器、类型检查器等)使用的配置文件. 此文件中有三个可能的 TOML 表.

- The `[build-system]` table is **strongly recommended**. It allows you to declare which [build backend](https://packaging.python.org/en/latest/glossary/#term-Build-Backend) you use and which other dependencies are needed to build your project.
  强烈建议使用 `[build-system]` 表. 它允许您声明使用的[构建后端](https://packaging.python.org/en/latest/glossary/#term-Build-Backend)以及构建项目所需的其他依赖项.

- The `[project]` table is the format that most build backends use to specify your project's basic metadata, such as the dependencies, your name, etc.
  `[project]` 表是大多数构建后端用来指定项目基本元数据的格式, 例如依赖项、您的名称等.

- The `[tool]` table has tool-specific subtables, e.g., `[tool.hatch]`, `[tool.black]`, `[tool.mypy]`. We only touch upon this table here because its contents are defined by each tool. Consult the particular tool's documentation to know what it can contain.
  `[tool]` 表包含特定于工具的子表, 例如 `[tool.hatch]`, `[tool.black]`, `[tool.mypy]`. 我们在此仅简要提及此表, 因为其内容由各个工具自行定义. 请查阅相应工具的文档以了解其具体内容.

> Note
>
> The `[build-system]` table should always be present, regardless of which build backend you use (`[build-system]` *defines* the build tool you use).
> 无论你使用哪个构建后端, `[build-system]` 表都应该始终存在(`[build-system]` 定义了你使用的构建工具).
>
> On the other hand, the `[project]` table is understood by *most* build backends, but some build backends use a different format.
> 另一方面, 大多数构建后端都能理解 `[project]` 表, 但有些构建后端使用不同的格式.
>
> A notable exception is [Poetry](https://python-poetry.org/), which before version 2.0 (released January 5, 2025) did not use the `[project]` table, it used the `[tool.poetry]` table instead. With version 2.0, it supports both. Also, the [setuptools](https://setuptools.pypa.io/) build backend supports both the `[project]` table, and the older format in `setup.cfg` or `setup.py`.
> 值得注意的是, [Poetry](https://python-poetry.org/) 是一个例外, 在 2.0 版本(2025 年 1 月 5 日发布)之前, 它不使用 `[project]` 表, 而是使用 `[tool.poetry]` 表. 2.0 版本则同时支持这两种方式. 此外, [setuptools](https://setuptools.pypa.io/) 构建后端同时支持 `[project]` 表和 `setup.cfg` 或 `setup.py` 中的旧格式.
>
> For new projects, use the `[project]` table, and keep `setup.py` only if some programmatic configuration is needed (such as building C extensions), but the `setup.cfg` and `setup.py` formats are still valid. See [Is setup.py deprecated?](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/#setup-py-deprecated).
> 对于新项目, 请使用 `[project]` 表, 仅当需要某些程序化配置(例如构建 C 扩展)时才保留 `setup.py` 文件, 但 `setup.cfg` 和 `setup.py` 格式仍然有效. 请参阅 [setup.py 是否已弃用? ](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/#setup-py-deprecated)

## Declaring the build backend
声明构建后端

The `[build-system]` table contains a `build-backend` key, which specifies the build backend to be used. It also contains a `requires` key, which is a list of dependencies needed to build the project – this is typically just the build backend package, but it may also contain additional dependencies. You can also constrain the versions, e.g., `requires = ["setuptools >= 61.0"]`.
`[build-system]` 表包含一个 `build-backend` 键, 用于指定要使用的构建后端. 它还包含一个 `requires` 键, 其中列出了构建项目所需的依赖项 – 通常仅包含构建后端包, 但也可能包含其他依赖项. 您还可以限制版本, 例如 `requires = ["setuptools >= 61.0"]`.

Usually, you'll just copy what your build backend's documentation suggests (after [choosing your build backend](https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-build-backend)). Here are the values for some common build backends:
通常情况下, 您只需复制构建后端文档中的建议(在[选择构建后端](https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-build-backend)之后). 以下是一些常用构建后端的相应值:

**Hatchling**
```
[build-system]
requires = ["hatchling >= 1.26"]
build-backend = "hatchling.build"
```

**setuptools**
```
[build-system]
requires = ["setuptools >= 77.0.3"]
build-backend = "setuptools.build_meta"
```

**Flit**
```
[build-system]
requires = ["flit_core >= 3.12.0, <4"]
build-backend = "flit_core.buildapi"
```

**PDM**
```
[build-system]
requires = ["pdm-backend >= 2.4.0"]
build-backend = "pdm.backend"
```

**uv-build**
```
[build-system]
requires = ["uv_build >= 0.9.18, <0.10.0"]
build-backend = "uv_build"
```

## Static vs. dynamic metadata
静态元数据与动态元数据

The rest of this guide is devoted to the `[project]` table.
本指南的其余部分将专门介绍 `[project]` 表.

Most of the time, you will directly write the value of a `[project]` field. For example: `requires-python = ">= 3.8"`, or `version = "1.0"`.
大多数情况下, 你会直接写出 `[project]` 的值. 例如: `requires-python = ">= 3.8"`, 或 `version = "1.0"`.

However, in some cases, it is useful to let your build backend compute the metadata for you. For example: many build backends can read the version from a `__version__` attribute in your code, a Git tag, or similar. In such cases, you should mark the field as dynamic using, e.g.,
然而, 在某些情况下, 让构建后端自动计算元数据会很有用. 例如: 许多构建后端可以从代码中的 `__version__` 属性、Git 标签或类似位置读取版本信息. 在这种情况下, 您应该使用例如 `__version__` 将该字段标记为动态字段.

```
[project]
dynamic = ["version"]
```

When a field is dynamic, it is the build backend's responsibility to fill it. Consult your build backend's documentation to learn how it does it.
当某个字段是动态的, 则由构建后端负责填充. 请查阅构建后端的文档, 了解其具体实现方式.

## Basic information
基本信息

### `name`

Put the name of your project on PyPI. This field is required and is the only field that cannot be marked as dynamic.
请在 PyPI 上填写您的项目名称. 此字段为必填项, 也是唯一不能标记为动态字段的字段.

```
[project]
name = "spam-eggs"
```

The project name must consist of ASCII letters, digits, underscores "`_`", hyphens "`-`" and periods "`.`". It must not start or end with an underscore, hyphen or period.
项目名称必须由 ASCII 字母、数字、下划线"`_`"、连字符"`-`"和句点"`.`"组成. 项目名称不得以下划线、连字符或句点开头或结尾.

Comparison of project names is case insensitive and treats arbitrarily long runs of underscores, hyphens, and/or periods as equal. For example, if you register a project named `cool-stuff`, users will be able to download it or declare a dependency on it using any of the following spellings: `Cool-Stuff`, `cool.stuff`, `COOL_STUFF`, `CoOl__-.-__sTuFF`.
项目名称的比较不区分大小写, 并且将任意长度的下划线、连字符和/或句点视为相同. 例如, 如果您注册了一个名为 `cool-stuff` 项目, 用户可以使用以下任何拼写方式下载该项目或声明对其的依赖关系: `Cool-Stuff`, `cool.stuff`, `COOL_STUFF`, `CoOl__-.-__sTuFF`.

### `version`

Put the version of your project.
请填写您的项目版本.

```
[project]
version = "2020.0.0"
```

Some more complicated version specifiers like `2020.0.0a1` (for an alpha release) are possible; see the [specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers) for full details.  
一些更复杂的版本号指定方式也是可能的, 例如 `2020.0.0a1` (用于 alpha 版本); 请参阅[规范](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers). 详情请见下文.

This field is required, although it is often marked as dynamic using
此字段为必填项, 尽管它通常被标记为动态字段.

```
[project]
dynamic = ["version"]
```

This allows use cases such as filling the version from a `__version__` attribute or a Git tag. Consult the [Single-sourcing the Project Version](https://packaging.python.org/en/latest/discussions/single-source-version/#single-source-version) discussion for more details.
这样就允许一些使用场景, 例如从 `__version__` 中填充版本信息. 属性或 Git 标签. 请参阅 ["项目版本的单一来源"](https://packaging.python.org/en/latest/discussions/single-source-version/#single-source-version) 部分. 更多详情请讨论.

## Dependencies and requirements
依赖关系和要求

### `dependencies`/`optional-dependencies`

If your project has dependencies, list them like this:
如果你的项目有依赖项, 请按如下方式列出:

```
[project]
dependencies = [
  "httpx",
  "gidgethub[httpx]>4.0.0",
  "django>2.1; os_name != 'nt'",
  "django>2.0; os_name == 'nt'",
]
```

See [Dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#dependency-specifiers) for the full syntax you can use to constrain versions.
有关可用于约束版本的完整语法, 请参阅 [依赖项说明符](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#dependency-specifiers).

You may want to make some of your dependencies optional, if they are only needed for a specific feature of your package. In that case, put them in `optional-dependencies`.
如果某些依赖项仅用于软件包的特定功能, 则可以将其设为可选依赖项. 在这种情况下, 请将它们放在 `optional-dependencies` 目录中.

```
[project.optional-dependencies]
gui = ["PyQt5"]
cli = [
  "rich",
  "click",
]
```

Each of the keys defines a "packaging extra". In the example above, one could use, e.g., `pip install your-project-name[gui]` to install your project with GUI support, adding the PyQt5 dependency.
每个键都定义了一个"打包附加组件". 例如, 在上面的示例中, 可以使用 `pip install your-project-name[gui]` 来安装带有 GUI 支持的项目, 并添加 PyQt5 依赖项.

### `requires-python`

This lets you declare the minimum version of Python that you support [1](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#requires-python-upper-bounds).
这样, 您可以声明您支持的最低 Python 版本. [1](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#requires-python-upper-bounds)

```
[project]
requires-python = ">= 3.8"
```

## Creating executable scripts
创建可执行脚本

To install a command as part of your package, declare it in the `[project.scripts]` table.
要将命令安装为软件包的一部分, 请在其中声明它.  `[project.scripts]` 表.

```
[project.scripts]
spam-cli = "spam:main_cli"
```

In this example, after installing your project, a `spam-cli` command will be available. Executing this command will do the equivalent of `import sys; from spam import main_cli; sys.exit(main_cli())`.
在这个例子中, 安装完你的项目后, 需要执行 `spam-cli` 命令. 将会可用. 执行此命令将相当于: `import sys; from spam import main_cli; sys.exit(main_cli())`.

On Windows, scripts packaged this way need a terminal, so if you launch them from within a graphical application, they will make a terminal pop up. To prevent this from happening, use the `[project.gui-scripts]` table instead of `[project.scripts]`.
在 Windows 系统中, 以这种方式打包的脚本需要终端, 因此如果您从图形应用程序内部启动它们, 将会弹出一个终端窗口. 要防止这种情况发生, 请使用 `[project.gui-scripts]` 文件. 表格而不是 `[project.scripts]`.

```
[project.gui-scripts]
spam-gui = "spam:main_gui"
```

In that case, launching your script from the command line will give back control immediately, leaving the script to run in the background.
在这种情况下, 从命令行启动脚本会立即将控制权交还给系统, 让脚本在后台运行.

The difference between `[project.scripts]` and `[project.gui-scripts]` is only relevant on Windows.
`[project.scripts]` 之间的区别 `[project.gui-scripts]` 仅在 Windows 系统上有效.

## About your project
关于您的项目

### `authors`/`maintainers`

Both of these fields contain lists of people identified by a name and/or an email address.
这两个字段都包含以姓名和/或电子邮件地址标识的人员列表.

```
[project]
authors = [
  {name = "Pradyun Gedam", email = "pradyun@example.com"},
  {name = "Tzu-Ping Chung", email = "tzu-ping@example.com"},
  {name = "Another person"},
  {email = "different.person@example.com"},
]
maintainers = [
  {name = "Brett Cannon", email = "brett@example.com"}
]
```

### `description`

This should be a one-line description of your project, to show as the "headline" of your project page on PyPI ([example](https://pypi.org/project/pip)), and other places such as lists of search results ([example](https://pypi.org/search?q=pip)).
这应该是对您的项目的一行描述, 显示为 PyPI 上项目页面的"标题"([示例](https://pypi.org/project/pip)), 以及搜索结果列表等其他地方([示例](https://pypi.org/search?q=pip)).

```
[project]
description = "Lovely Spam! Wonderful Spam!"
```

### `readme`

This is a longer description of your project, to display on your project page on PyPI. Typically, your project will have a `README.md` or `README.rst` file and you just put its file name here.
这是对您项目的更详细描述, 将显示在 PyPI 上的项目页面上. 通常, 您的项目会有一个 `README.md` 或其他类似文件 `README.rst` 文件, 你只需把它的文件名放在这里即可.

```
[project]
readme = "README.md"
```

The README's format is auto-detected from the extension:
README 文件的格式会根据文件扩展名自动检测:

- `README.md` → [GitHub-flavored Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax),

- `README.rst` → [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (without Sphinx extensions).

You can also specify the format explicitly, like this:
您也可以显式指定格式, 如下所示:

```
[project]
readme = {file = "README.txt", content-type = "text/markdown"}
# or
readme = {file = "README.txt", content-type = "text/x-rst"}
```

### `license` and `license-files`

As per [**PEP 639**](https://peps.python.org/pep-0639/), licenses should be declared with two fields:
根据 [PEP 639](https://peps.python.org/pep-0639/), 许可证应使用两个字段进行声明:

- `license` is an [SPDX license expression](https://packaging.python.org/en/latest/glossary/#term-License-Expression) consisting of one or more [license identifiers](https://packaging.python.org/en/latest/glossary/#term-License-Identifier).
  license 是 SPDX 许可证表达式 由一个或多个许可证标识符组成.

- `license-files` is a list of license file glob patterns.
  license-files 是许可证文件 glob 模式的列表.

A previous PEP had specified `license` to be a table with a `file` or a `text` key, this format is now deprecated. Most [build backends](https://packaging.python.org/en/latest/glossary/#term-Build-Backend) now support the new format as shown in the following table.  
之前的 PEP 曾规定 `license` 是一个包含 `file` 表或一个 `text` 键, 此格式现已弃用. 大多数[构建后端](https://packaging.python.org/en/latest/glossary/#term-Build-Backend)现在都支持如下表所示的新格式.


build backend versions that introduced [PEP 639](https://peps.python.org/pep-0639/) support

| hatchling | setuptools | flit-core | pdm-backend | poetry-core | uv-build |
| --------- |----------- |---------- |------------ | ----------- |--------- |
| 1.27.0    | 77.0.3     | 3.12      | 2.4.0       | 2.2.0       | 0.7.19   |

#### `license`

The new format for `license` is a valid [SPDX license expression](https://packaging.python.org/en/latest/glossary/#term-License-Expression) consisting of one or more [license identifiers](https://packaging.python.org/en/latest/glossary/#term-License-Identifier). The full license list is available at the [SPDX license list page](https://spdx.org/licenses/). The supported list version is 3.17 or any later compatible one.
新的 `license` 格式是有效的 SPDX 许可证表达式. 由一个或多个许可证标识符组成. 完整的许可证列表可在以下网址获取: [SPDX 许可证列表页面](https://spdx.org/licenses/). 支持的版本为 3.17 或任何更高版本的兼容版本.

```
[project]
license = "GPL-3.0-or-later"
# or
license = "MIT AND (Apache-2.0 OR BSD-2-Clause)"
```

> Note
>
> If you get a build error that `license` should be a dict/table, your build backend doesn't yet support the new format. See the [above section](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license-and-license-files) for more context. The now deprecated format is [described in PEP 621](https://peps.python.org/pep-0621/#license).
> 如果构建时出现错误, 提示 `license` 应该是一个字典/表, 您的构建后端尚不支持新格式. 请参阅 更多上下文请参见[上文](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license-and-license-files). 现已弃用的格式[在 PEP 621 中有详细描述](https://peps.python.org/pep-0621/#license).

As a general rule, it is a good idea to use a standard, well-known license, both to avoid confusion and because some organizations avoid software whose license is unapproved.
一般来说, 使用标准、知名的许可证是个好主意, 这样既可以避免混淆, 也可以避免使用未经批准的许可证的软件.

If your [Distribution Archive](https://packaging.python.org/en/latest/glossary/#term-Distribution-Archive) is licensed with a license that doesn't have an existing SPDX identifier, you can create a custom one in format `LicenseRef-[idstring]`. The custom identifiers must follow the SPDX specification, [clause 10.1](https://spdx.github.io/spdx-spec/v2.2.2/other-licensing-information-detected/) of the version 2.2 or any later compatible one.
如果您的分发存档使用的是不符合规定的许可证, 则无法获得许可. 如果已有 SPDX 标识符, 您可以按格式创建自定义标识符. `LicenseRef-[idstring]`. 自定义标识符必须遵循 SPDX 规范[第 10.1 条](https://spdx.github.io/spdx-spec/v2.2.2/other-licensing-information-detected/), 版本 2.2 或任何后续兼容版本.

```
[project]
license = "LicenseRef-My-Custom-License"
```

#### `license-files`

This is a list of license files and files containing other legal information you want to distribute with your package.
这是您希望与软件包一起分发的许可证文件和其他法律信息文件的列表.

```
[project]
license-files = ["LICEN[CS]E*", "vendored/licenses/*.txt", "AUTHORS.md"]
```

The glob patterns must follow the specification:
glob 模式必须遵循以下规范:

- Alphanumeric characters, underscores (`_`), hyphens (`-`) and dots (`.`) will be matched verbatim.
  字母数字字符、下划线( `_` )、连字符( `-` )和点( `.` )将逐字匹配.

- Special characters: `*`, `?`, `**` and character ranges: [] are supported.
  支持特殊字符:  `*` 、 `?` 、 `**` 和字符范围: [].

- Path delimiters must be the forward slash character (`/`).
  路径分隔符必须是正斜杠字符( `/` ).

- Patterns are relative to the directory containing `pyproject.toml`, and thus may not start with a slash character.
  模式是相对于包含 `pyproject.toml` 的目录而言的, 因此不能以斜杠字符开头.

- Parent directory indicators (`..`) must not be used.
  不得使用父目录指示符( `..` ).

- Each glob must match at least one file.
  每个 glob 模式必须至少匹配一个文件.

Literal paths are valid globs. Any characters or character sequences not covered by this specification are invalid.
字面路径是有效的通配符. 任何未在此规范中涵盖的字符或字符序列均无效.

### `keywords`

This will help PyPI's search box to suggest your project when people search for these keywords.
这将有助于 PyPI 的搜索框在人们搜索这些关键词时推荐您的项目.

```
[project]
keywords = ["egg", "bacon", "sausage", "tomatoes", "Lobster Thermidor"]
```

### `classifiers`

A list of PyPI classifiers that apply to your project. Check the [full list of possibilities](https://pypi.org/classifiers).
适用于您项目的 PyPI 分类器列表. 请查看 [所有可能性列表](https://pypi.org/classifiers).

```
classifiers = [
  # How mature is this project? Common values are
  #   3 - Alpha
  #   4 - Beta
  #   5 - Production/Stable
  "Development Status :: 4 - Beta",

  # Indicate who your project is intended for
  "Intended Audience :: Developers",
  "Topic :: Software Development :: Build Tools",

  # Specify the Python versions you support here.
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.6",
  "Programming Language :: Python :: 3.7",
  "Programming Language :: Python :: 3.8",
  "Programming Language :: Python :: 3.9",
]
```

Although the list of classifiers is often used to declare what Python versions a project supports, this information is only used for searching and browsing projects on PyPI, not for installing projects. To actually restrict what Python versions a project can be installed on, use the [requires-python](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#requires-python) argument.
虽然分类器列表通常用于声明项目支持的 Python 版本, 但此信息仅用于在 PyPI 上搜索和浏览项目, 而不用于安装项目. 要真正限制项目可以安装的 Python 版本, 请使用 [requires-python](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#requires-python) 参数.

To prevent a package from being uploaded to PyPI, use the special `Private :: Do Not Upload` classifier. PyPI will always reject packages with classifiers beginning with `Private ::`.
要阻止软件包上传到 PyPI, 请使用特殊的 `Private :: Do Not Upload` 分类器. PyPI 将始终拒绝分类器以 `Private ::` 开头的软件包.

### `urls`

A list of URLs associated with your project, displayed on the left sidebar of your PyPI project page.
与您的项目关联的 URL 列表, 显示在 PyPI 项目页面的左侧边栏中.


> Note
> See [Well-known labels](https://packaging.python.org/en/latest/specifications/well-known-project-urls/#well-known-labels) for a listing of labels that PyPI and other packaging tools are specifically aware of, and [PyPI's project metadata docs](https://docs.pypi.org/project_metadata/#project-urls) for PyPI-specific URL processing.
> 请参阅常用标签以获取 PyPI 和其他打包工具专门识别的标签列表, 以及 [PyPI 的项目元数据文档.](https://docs.pypi.org/project_metadata/#project-urls) 用于 PyPI 特有的 URL 处理.

```
[project.urls]
Homepage = "https://example.com"
Documentation = "https://readthedocs.org"
Repository = "https://github.com/me/spam.git"
Issues = "https://github.com/me/spam/issues"
Changelog = "https://github.com/me/spam/blob/master/CHANGELOG.md"
```

Note that if the label contains spaces, it needs to be quoted, e.g., `Website = "https://example.com"` but `"Official Website" = "https://example.com"`.
请注意, 如果标签包含空格, 则需要用引号括起来, 例如: `Website = "https://example.com"` 但 `"Official Website" = "https://example.com"`.

Users are advised to use [Well-known labels](https://packaging.python.org/en/latest/specifications/well-known-project-urls/#well-known-labels) for their project URLs where appropriate, since consumers of metadata (like package indices) can specialize their presentation.
建议用户在适当情况下使用[知名标签](https://packaging.python.org/en/latest/specifications/well-known-project-urls/#well-known-labels)作为其项目 URL, 因为元数据(如软件包索引)的使用者可以对其进行专门化展示.

For example in the following metadata, neither `MyHomepage` nor `"Download Link"` is a well-known label, so they will be rendered verbatim:
例如, 在以下元数据中, 既没有 `MyHomepage`, 也没有 "MyHomepage".  `"Download Link"` 是一个众所周知的标签, 因此将逐字逐句地呈现:

```
[project.urls]
MyHomepage = "https://example.com"
"Download Link" = "https://example.com/abc.tar.gz"
```

Whereas in this metadata `HomePage` and `DOWNLOAD` both have well-known equivalents (`homepage` and `download`), and can be presented with those semantics in mind (the project's home page and its external download location, respectively).
而在这个元数据中, `HomePage` 和 `DOWNLOAD` 都有众所周知的对应词( `homepage` 和 `download` ), 并且可以按照这些语义来表示(分别是项目的主页及其外部下载位置).

```
[project.urls]
HomePage = "https://example.com"
DOWNLOAD = "https://example.com/abc.tar.gz"
```

## Advanced plugins
高级插件

Some packages can be extended through plugins. Examples include [Pytest](https://pytest.org/) and [Pygments](https://pygments.org/). To create such a plugin, you need to declare it in a subtable of `[project.entry-points]` like this:
有些软件包可以通过插件进行扩展. 例如 [Pytest](https://pytest.org/) 以及 [Pygments](https://pygments.org/). 要创建这样的插件, 需要在 `[project.entry-points]` 的子表中声明它, 如下所示:

```
[project.entry-points."spam.magical"]
tomatoes = "spam:main_tomatoes"
```

See the [Plugin guide](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#plugin-entry-points) for more information.
更多信息请参阅[插件指南](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#plugin-entry-points).

## A full example
完整示例

```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "spam-eggs"
version = "2020.0.0"
dependencies = [
  "httpx",
  "gidgethub[httpx]>4.0.0",
  "django>2.1; os_name != 'nt'",
  "django>2.0; os_name == 'nt'",
]
requires-python = ">=3.8"
authors = [
  {name = "Pradyun Gedam", email = "pradyun@example.com"},
  {name = "Tzu-Ping Chung", email = "tzu-ping@example.com"},
  {name = "Another person"},
  {email = "different.person@example.com"},
]
maintainers = [
  {name = "Brett Cannon", email = "brett@example.com"}
]
description = "Lovely Spam! Wonderful Spam!"
readme = "README.rst"
license = "MIT"
license-files = ["LICEN[CS]E.*"]
keywords = ["egg", "bacon", "sausage", "tomatoes", "Lobster Thermidor"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Programming Language :: Python"
]

[project.optional-dependencies]
gui = ["PyQt5"]
cli = [
  "rich",
  "click",
]

[project.urls]
Homepage = "https://example.com"
Documentation = "https://readthedocs.org"
Repository = "https://github.com/me/spam.git"
"Bug Tracker" = "https://github.com/me/spam/issues"
Changelog = "https://github.com/me/spam/blob/master/CHANGELOG.md"

[project.scripts]
spam-cli = "spam:main_cli"

[project.gui-scripts]
spam-gui = "spam:main_gui"

[project.entry-points."spam.magical"]
tomatoes = "spam:main_tomatoes"
```

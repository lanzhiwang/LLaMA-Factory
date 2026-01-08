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
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Load a single dataset and aligns it to the standard format.

    这个函数是整个数据准备管线的"第一道关卡".
    在 LLM 微调中, 数据的来源(Hub、本地、脚本)和格式(JSON、CSV、文件夹)极其复杂. 该函数的核心任务是屏蔽底层存储差异, 实现异构数据源的标准化加载.

    加载单个数据集并将其对齐为标准格式.
    [设计动机]: LLM 微调往往需要从不同生态(HF、ModelScope、本地)拉取数据.
    [解决的问题]: 该函数作为一个"调度器", 根据 dataset_attr 定义的来源, 动态切换加载逻辑.

    深度解析(高级研究员视角):

    多生态兼容 (Ecosystem Agnostic):
    代码中对 ms_hub 和 om_hub 的处理展示了该项目不仅是面向 HuggingFace, 而是试图构建一个通用的微调基座.
    通过 to_hf_dataset() 将不同 SDK 加载的对象归一化, 是保证后续代码逻辑简洁(Decoupling)的关键.

    鲁棒的采样逻辑 (Robust Resampling):
    在多数据集混合训练中, 数据集权重的平衡(Balancing)至关重要.
    代码中通过 permutation 和 choice 的结合, 既保证了数据尽可能被遍历(无放回), 又支持了权重放大(有放回), 这是在工业级微调配置中非常实用的细节.

    内存安全 (Memory Safety):
    在 load_dataset 环节对 streaming 的处理以及针对本地文件的 to_iterable_dataset 转化, 体现了对处理海量数据(TB 级预训练数据)时的内存保护考量.

    希望这些注释和分析能帮助你更深入地理解 LLaMA-Factory 数据加载层的设计之美!
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")

    # 初始化加载参数, 准备传给不同的加载后端
    data_path, data_name, data_dir, data_files = None, None, None, None

    # 1. 路径解析分支: 根据 load_from 属性确定数据定位逻辑
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        # [场景]: 从各大主流模型中心加载(HuggingFace, ModelScope, OpenMind)
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        # [场景]: 使用自定义 Python 加载脚本(支持更复杂的预处理)
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        # [场景]: 从云端路径(如 S3/OSS 映射或特定的云 JSON 协议)加载
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        # 2. 本地文件系统处理逻辑
        # [解决的问题]: 支持用户传入单一文件或包含多个数据的文件夹.
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)

        # 如果路径是目录
        if os.path.isdir(local_path):  # is directory
            # 自动扫描目录下所有文件, 实现批量加载
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        # 如果是单文件
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        # 3. 自动识别文件格式 (File Extension Inference)
        # [为什么要这么写]: 通过后缀名自动推断 datasets 库所需的加载器类型(json/csv等).
        # [解决的问题]: 防止用户在一个数据集目录下混用不同格式的文件, 确保加载的一致性.
        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    # 4. 后端执行分支: 调用特定的 SDK 进行数据拉取
    if dataset_attr.load_from == "ms_hub":
        # [兼容性]: 支持 ModelScope 生态, 这在国内网络环境下是必备的
        check_version("modelscope>=1.14.0", mandatory=True)
        # 延迟导入, 减少非必要依赖占用
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            # 将 ModelScope 对象统一转化为 HF 格式, 确保后续 align_dataset 通用
            dataset = dataset.to_hf_dataset()

    # [兼容性]: 支持昇腾 OpenMind 生态
    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )

    # [为什么要这么写]: 针对云端大规模分布式训练, 绕过本地磁盘读取.
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        # 5. 标准 HuggingFace `load_dataset` 调用
        # [解决的问题]: 处理大多数开源数据集和本地 JSON/JSONL/CSV 文件.
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            # 文件加载模式下, 如果开启流式, 需要后续手动转化
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        # 6. 流式加载适配 (Streaming Adaptation)
        # [解决的问题]: 本地大文件如果不转为 IterableDataset, 会一次性加载到内存导致 OOM.
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    # 7. 数据量控制与重采样逻辑 (Resampling/Oversampling)
    # [为什么要这么写]: 实现数据集的"精准比例混合".
    # [解决的问题]: 如果用户要求从一个 100 条的数据集中采样 500 条(为了增加其在混合训练中的权重),
    # 这里的代码支持"有放回采样"(np.random.choice), 实现数据集的扩充.
    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        # 基础洗牌并取前 N 条
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            # 如果还需要更多(即 target_num > len(dataset)), 则进行有放回的随机扩充
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    # 8. 强制截断逻辑 (Debug/Max Samples Control)
    # [解决的问题]: 允许用户通过 `max_samples` 参数快速在小规模数据上验证流程.
    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    # 9. 列名对齐 (Schema Alignment)
    # [为什么要这么写]: 最后一步调用 align_dataset.
    # [解决的问题]: 不同数据集的列名可能叫 'instruction', 'question' 或 'prompt'.
    # 该函数将它们统一映射为 LLaMA-Factory 内部标准字段(如 'prompt', 'query', 'response'),
    # 这样后续的 Template 系统才能无视数据集差异进行分词.
    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""
    Return the merged datasets in the standard format.

    _get_merged_dataset 函数不仅是简单的数据加载器, 它实际上承担了数据合规性检查、多源异构数据整合、以及评估策略路由等多重职责.
    在处理成百上千 GB 的多源数据时, 这种设计能有效防止"垃圾进, 垃圾出"(Garbage In, Garbage Out).

    获取并合并多个数据集, 将其转化为标准格式.

    [设计动机]: LLM 微调经常需要混合多个数据集(例如: 通用能力数据 + 垂直领域数据).
    [解决的问题]: 统一多数据源加载入口, 并强制执行"阶段-数据格式"一致性检查.

    高级研究员视角下的架构深度解析:

    数据契约化 (Contract-based Data Loading):
    代码中对 dataset_attr.ranking 的检查是一种典型的"契约编程".
    在大型项目中, 数据集通常由不同团队准备, 文件名往往具有误导性. 通过在加载最早期强制进行格式校验, 可以避免极其昂贵的 GPU 算力浪费在错误的数据格式上.

    解耦评估逻辑:
    return_dict 的设计体现了对 Multi-task Evaluation (多任务评测) 的支持. 作为高级研究员, 我知道"合并评估"会掩盖模型在特定任务上的坍缩(Regression).
    这种设计允许评测脚本遍历字典, 为每个数据集生成独立的指标报告.

    确定性混合 (Deterministic Mixing):
    在 merge_dataset 步骤传入 seed 是 Python 高级开发工程师的直觉体现.
    在分布式训练(DDP)中, 如果各进程合并数据集的随机数种子不一致, 会导致不同显卡看到的样本流完全乱序, 甚至出现数据泄露, 最终影响梯度聚合的有效性.

    希望这份详细的解析能让你更透彻地理解 LLaMA-Factory 在数据处理层面的工程严谨性!
    """

    # 1. 容错处理
    if dataset_names is None:
        return None

    datasets = {}
    # 2. 遍历并解析数据集元数据 (Dataset Metadata Parsing)
    # [为什么要这么写]: 通过 zip 结合名称和 get_dataset_list 获取的属性(DatasetAttr).
    # [解决的问题]: 解耦了"数据集名称"与"数据集物理存储细节"(如路径、格式、列映射).
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        # 3. 核心一致性检查 (Integrity & Consistency Check)
        # [为什么要这么写]: 判断当前任务阶段(stage)与数据集类型(ranking)是否匹配.
        # [解决的问题]: 防止工程事故.
        #    - 如果是 RM(奖励建模)阶段, 数据集必须是带有排序(ranking)信息的对齐数据.
        #    - 如果是 SFT/PT 等阶段, 数据集不能是排序格式, 否则后续计算 CrossEntropy Loss 时会因维度不匹配崩溃.
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        # 4. 单个数据集加载 (Single Source Loading)
        # [为什么要这么写]: 调用底层函数加载单一数据源, 返回统一的 HuggingFace Dataset 对象.
        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    # 5. 返回策略分发 (Routing Strategy)
    # [解决的问题]: 平衡"训练效率"与"评测精细度".
    if return_dict:
        # [为什么要这么写]: 返回原始字典格式.
        # [应用场景]: 通常用于评估(Evaluation). 在微调多个领域数据集后, 研究员往往需要
        # 查看模型在各个子数据集上的独立得分(如代码集、数学集分别的表现), 而不是合并后的模糊均值.
        return datasets
    else:
        # [为什么要这么写]: 调用 merge_dataset 进行物理或逻辑上的合并.
        # [应用场景]: 用于训练(Training).
        # [解决的问题]: 根据 data_args 中的配置, 处理多数据集的混合权重、采样策略(Concat 还是 Interleave).
        # 传入 training_args.seed 确保在分布式多卡训练时, 各卡间的数据混合顺序是同步且可复现的.
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"] | None:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Get the train dataset and optionally gets the evaluation dataset.

    在大型 LLM 微调工程中, 数据处理往往是最大的瓶颈.
    这个函数不仅仅是"读取文件", 它实际上是一个工业级的数据编排流水线, 解决了分布式环境下的 I/O 冲突、分词效率、内存管理以及实验可复现性等核心痛点.

    获取训练数据集, 并根据配置可选地获取评估数据集.
    该函数是 LLaMA-Factory 数据流的核心枢纽, 负责加载、合并、分词、缓存及保存.

    深度架构解析(高级研究员视角):

    分布式屏障(Main Process First):
    这是在 torch.distributed 环境下编写 Python 脚本的最高准则. 由于 LLM 数据预处理是 CPU 密集型任务, 而 GPU 训练是显存密集型任务,
    在 SFT 阶段, 如果每个进程都去重复处理一遍数据, 会造成 CPU 负载瞬间爆表, 甚至导致主进程(Rank 0)因为系统资源被抢占而无法发出 NCCL 同步信号, 进而导致训练在启动阶段就卡死.

    数据流式(Streaming)与磁盘存储的权衡:
    代码中对 tokenized_path 和 streaming 的校验非常严谨. 流式读取是为了处理"内存装不下"的数据(如数亿条 Token), 而磁盘存储是为了"下次启动快".
    LLaMA-Factory 在这里做到了很好的平衡: 允许离线处理完后存盘, 下次训练时再根据需要选择是否流式读取.

    对多任务阶段(Stage)的适配:
    注意函数参数中的 stage. 在 _get_preprocessed_dataset 内部, 它会根据是 pt(预训练)、sft(指令微调)还是 rm(奖励建模)来选择不同的 Data Collator 和 Template.
    这体现了该函数高度的通用性和抽象能力.

    希望这些注释能够帮助你彻底理解这个核心函数的工程价值!
    """

    # 1. 断点续传与缓存机制 (Fast-loading Strategy)
    # [为什么要这么写]: 大规模数据集(如 TB 级的预训练数据)分词(Tokenization)极其耗时.
    # [解决的问题]: 如果用户之前已经运行过预处理并保存了 tokenized 后的数据,
    # 我们可以通过 `load_from_disk` 秒级加载, 避免重复花费数小时进行 CPU 密集型的分词运算.
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            # 警告用户: 一旦加载已分词的数据, 其他的原始数据处理参数(如提示词模板改变)将被忽略
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)

            # 适配流式模式: 即使是从磁盘加载, 也可以转为 IterableDataset 以节省内存占用
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        # 逻辑保护: 保存数据集到磁盘和流式读取在 datasets 库中通常是互斥的操作逻辑
        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # 2. 分布式环境下的数据加载保护 (Distributed I/O Barrier)
    # [为什么要这么写]: 使用 `main_process_first` 上下文管理器.
    # [解决的问题]: 在多卡(DDP/DeepSpeed)环境下, 通常会有 8 个甚至更多进程同时运行.
    # 如果不加保护, 所有进程会同时尝试下载、解压、读取同一个数据集, 这会导致:
    # 1) 网络带宽被占满; 2) 磁盘 I/O 阻塞; 3) 文件系统死锁.
    # 这里确保只有主进程执行加载, 其他进程等待主进程写好缓存后直接读取缓存.
    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        # 加载并合并多个数据集(LLaMA-Factory 支持将多个 json/jsonl 数据动态混合)
        # Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        # 加载评估集, 支持对每个评估集进行独立评估(eval_on_each_dataset)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    # 3. 数据集预处理与分词 (Tokenization Pipeline)
    # [为什么要这么写]: 再次使用 `main_process_first` 保护预处理过程.
    # [解决的问题]: 预处理涉及复杂的 Template 拼接和 Tokenizer 运算.
    # 同样只需主进程计算一次并存入 HF datasets 缓存, 其他子进程之后可直接 map 到内存镜像.
    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # 处理数据集划分(如果没有显式的验证集, 则根据 val_size 比例从训练集中切分)
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        # 对训练集进行预处理: 将原始对话转为 input_ids 和 labels
        if "train" in train_dict:
            train_dict["train"] = _get_preprocessed_dataset(
                train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )

        # 对所有评估集(可能有多个)进行预处理
        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        # 使用 DatasetDict 结构统一管理, 这是符合 Hugging Face Trainer 标准的最佳实践
        # Combine train and eval dictionaries
        dataset_dict = DatasetDict({**train_dict, **eval_dict})

        # 4. 数据持久化 (Artifact Persistence)
        # [为什么要这么写]: 如果用户指定了 tokenized_path 且当前是初次运行.
        # [解决的问题]: 将费了半天劲处理好的 Token 数据序列化到磁盘.
        # 这样下次训练同一个任务时, 就可以直接走步骤 1 的"快车道".
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            # 仅在主进程中执行保存操作, 避免并发写入冲突
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        # 最终将 DatasetDict 转换为 Trainer 能够直接消费的格式(包含 train/eval 句柄)
        return get_dataset_module(dataset_dict)

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
        """
        print(local_path)
        data/identity.json
        """

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
        """
        print(data_files)
        ['data/identity.json']

        print(os.path.splitext("data/identity.json"))
        ('data/identity', '.json')
        """

        # 3. 自动识别文件格式 (File Extension Inference)
        # [为什么要这么写]: 通过后缀名自动推断 datasets 库所需的加载器类型(json/csv等).
        # [解决的问题]: 防止用户在一个数据集目录下混用不同格式的文件, 确保加载的一致性.
        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        """
        print(data_path)
        json
        """
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
        """
        print(dataset)
        Dataset({
            features: ['instruction', 'input', 'output'],
            num_rows: 91
        })
        """
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
    print(dataset_names)
    ['identity', 'alpaca_en_demo']
    print(return_dict)
    False

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
    """
    print(datasets)
    {
        'identity': Dataset({
            features: ['_prompt', '_response', '_system', '_tools', '_images', '_videos', '_audios'],
            num_rows: 91
        }),
        'alpaca_en_demo': Dataset({
            features: ['_prompt', '_response', '_system', '_tools', '_images', '_videos', '_audios'],
            num_rows: 999
        })
    }
    """

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
    template = Template(
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
            tool_format=None,
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"], tool_format=None
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ],
            tool_format=None,
        ),
        format_tools=ToolFormatter(slots=[], tool_format="qwen"),
        format_prefix=EmptyFormatter(slots=[], tool_format=None),
        default_system="",
        stop_words=["<|im_end|>"],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=True,
        replace_jinja_template=False,
        enable_thinking=True,
        mm_plugin=BasePlugin(
            image_token=None, video_token=None, audio_token=None, expand_mm_tokens=True
        ),
    )
    model_args = ModelArguments(
        model_name_or_path="/root/huzhi/LLaMA-Factory/models/Qwen/Qwen3-4B-Instruct-2507",
        adapter_name_or_path=None,
        adapter_folder=None,
        cache_dir=None,
        use_fast_tokenizer=True,
        resize_vocab=False,
        split_special_tokens=False,
        add_tokens=None,
        add_special_tokens=None,
        new_special_tokens_config=None,
        init_special_tokens="noise_init",
        model_revision="main",
        low_cpu_mem_usage=True,
        rope_scaling=None,
        flash_attn="<AttentionFunction.AUTO: 'auto'>",
        shift_attn=False,
        mixture_of_depths=None,
        use_unsloth=False,
        use_unsloth_gc=False,
        enable_liger_kernel=False,
        moe_aux_loss_coef=None,
        disable_gradient_checkpointing=False,
        use_reentrant_gc=True,
        upcast_layernorm=False,
        upcast_lmhead_output=False,
        train_from_scratch=False,
        infer_backend="<EngineName.HF: 'huggingface'>",
        offload_folder="offload",
        use_kv_cache=True,
        use_v1_kernels=False,
        infer_dtype="auto",
        hf_hub_token=None,
        ms_hub_token=None,
        om_hub_token=None,
        print_param_status=False,
        trust_remote_code=True,
        quantization_method="<QuantizationMethod.BNB: 'bnb'>",
        quantization_bit=None,
        quantization_type="nf4",
        double_quantization=True,
        quantization_device_map=None,
        fp8=False,
        fp8_backend="auto",
        fp8_enable_fsdp_float8_all_gather=False,
        image_max_pixels=589824,
        image_min_pixels=1024,
        image_do_pan_and_scan=False,
        crop_to_patches=False,
        video_max_pixels=65536,
        video_min_pixels=256,
        video_fps=2.0,
        video_maxlen=128,
        use_audio_in_video=False,
        audio_sampling_rate=16000,
        export_dir=None,
        export_size=5,
        export_device="cpu",
        export_quantization_bit=None,
        export_quantization_dataset=None,
        export_quantization_nsamples=128,
        export_quantization_maxlen=1024,
        export_legacy_format=False,
        export_hub_model_id=None,
        use_kt=False,
        kt_optimize_rule=None,
        cpu_infer=32,
        chunk_size=8192,
        mode="normal",
        kt_maxlen=4096,
        kt_use_cuda_graph=True,
        kt_mode="normal",
        kt_force_think=False,
        vllm_maxlen=4096,
        vllm_gpu_util=0.7,
        vllm_enforce_eager=False,
        vllm_max_lora_rank=32,
        vllm_config=None,
        sglang_maxlen=4096,
        sglang_mem_fraction=0.7,
        sglang_tp_size=-1,
        sglang_config=None,
        sglang_lora_backend="triton",
        compute_dtype=torch.bfloat16,
        device_map={"": device(type="cuda", index=0)},
        model_max_length=2048,
        block_diag_attn=False,
    )
    data_args = DataArguments(
        template="qwen3_nothink",
        dataset=["identity", "alpaca_en_demo"],
        eval_dataset=None,
        dataset_dir="data",
        media_dir="data",
        cutoff_len=2048,
        train_on_prompt=False,
        mask_history=False,
        streaming=False,
        buffer_size=16384,
        mix_strategy="concat",
        interleave_probs=None,
        overwrite_cache=False,
        preprocessing_batch_size=1000,
        preprocessing_num_workers=16,
        max_samples=1000,
        eval_num_beams=None,
        ignore_pad_token_for_loss=True,
        val_size=0.0,
        eval_on_each_dataset=False,
        packing=False,
        neat_packing=False,
        tool_format=None,
        default_system=None,
        enable_thinking=True,
        tokenized_path=None,
        data_shared_file_system=False,
    )
    training_args = TrainingArguments(
        _n_gpu=1,
        accelerator_config={
            "split_batches": False,
            "dispatch_batches": None,
            "even_batches": True,
            "use_seedable_sampler": True,
            "non_blocking": False,
            "gradient_accumulation_kwargs": None,
            "use_configured_state": False,
        },
        adafactor=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        auto_find_batch_size=False,
        average_tokens_across_devices=True,
        batch_eval_metrics=False,
        bf16=True,
        bf16_full_eval=False,
        data_seed=None,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=None,
        ddp_backend=None,
        ddp_broadcast_buffers=None,
        ddp_bucket_cap_mb=None,
        ddp_find_unused_parameters=None,
        ddp_timeout=180000000,
        debug=[],
        deepspeed=None,
        disable_tqdm=False,
        do_eval=False,
        do_predict=False,
        do_train=True,
        eval_accumulation_steps=None,
        eval_delay=0,
        eval_do_concat_batches=True,
        eval_on_start=False,
        eval_steps=None,
        eval_strategy="IntervalStrategy.NO",  #
        eval_use_gather_object=False,
        fp16=False,
        fp16_backend="auto",  #
        fp16_full_eval=False,
        fp16_opt_level="O1",  #
        fsdp=[],
        fsdp_config={
            "min_num_params": 0,
            "xla": False,
            "xla_fsdp_v2": False,
            "xla_fsdp_grad_ckpt": False,
        },
        fsdp_min_num_params=0,
        fsdp_transformer_layer_cls_to_wrap=None,
        full_determinism=False,
        generation_config=None,
        generation_max_length=2048,
        generation_num_beams=None,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        greater_is_better=None,
        group_by_length=False,
        half_precision_backend="auto",  #
        hub_always_push=False,
        hub_model_id=None,
        hub_private_repo=None,
        hub_revision=None,
        hub_strategy="HubStrategy.EVERY_SAVE",  #
        hub_token="<HUB_TOKEN>",  #
        ignore_data_skip=False,
        include_for_metrics=[],
        include_inputs_for_metrics=False,
        include_num_input_tokens_seen="no",  #
        include_tokens_per_second=False,
        jit_mode_eval=False,
        label_names=["labels"],
        label_smoothing_factor=0.0,
        learning_rate=0.0001,
        length_column_name="length",  #
        liger_kernel_config=None,
        load_best_model_at_end=False,
        local_rank=0,
        log_level="passive",  #
        log_level_replica="warning",  #
        log_on_each_node=True,
        logging_dir="saves/qwen3-4b/lora/sft/runs/Feb14_21-45-52_k8s-a40-node02",  #
        logging_first_step=False,
        logging_nan_inf_filter=True,
        logging_steps=10,
        logging_strategy="IntervalStrategy.STEPS",  #
        lr_scheduler_kwargs={},
        lr_scheduler_type="SchedulerType.COSINE",  #
        max_grad_norm=1.0,
        max_steps=-1,
        metric_for_best_model=None,
        mp_parameters=None,  #
        neftune_noise_alpha=None,
        no_cuda=False,
        num_train_epochs=3.0,
        optim="OptimizerNames.ADAMW_TORCH_FUSED",  #
        optim_args=None,
        optim_target_modules=None,
        output_dir="saves/qwen3-4b/lora/sft",  #
        overwrite_output_dir=True,
        parallelism_config=None,
        past_index=-1,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=1,
        placement_strategy="PACK",  #
        predict_with_generate=False,
        prediction_loss_only=False,
        project="huggingface",  #
        push_to_hub=False,
        push_to_hub_model_id=None,
        push_to_hub_organization=None,
        push_to_hub_token="<PUSH_TO_HUB_TOKEN>",  #
        ray_init_kwargs=None,
        ray_num_workers=1,
        ray_run_name=None,
        ray_scope="last",  #
        ray_storage_filesystem=None,
        ray_storage_path="./saves",  #
        remove_unused_columns=False,
        report_to=[],
        resources_per_worker={"GPU": 1},
        restore_callback_states_from_checkpoint=False,
        resume_from_checkpoint=None,
        run_name=None,
        save_on_each_node=False,
        save_only_model=False,
        save_safetensors=True,
        save_steps=500,
        save_strategy="SaveStrategy.STEPS",  #
        save_total_limit=None,
        seed=42,
        skip_memory_metrics=True,
        sortish_sampler=False,
        tf32=None,
        torch_compile=False,
        torch_compile_backend=None,
        torch_compile_mode=None,
        torch_empty_cache_steps=None,
        torchdynamo=None,
        tpu_metrics_debug=False,
        tpu_num_cores=None,
        trackio_space_id="trackio",  #
        use_cpu=False,
        use_legacy_prediction_loop=False,
        use_liger_kernel=False,
        use_mps_device=False,
        warmup_ratio=0.1,
        warmup_steps=0,
        weight_decay=0.0,
    )
    stage = "sft"
    tokenizer = "Qwen2TokenizerFast"
    processor = None

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
        """
        print(dataset)
        Dataset({
            features: ['_prompt', '_response', '_system', '_tools', '_images', '_videos', '_audios'],
            num_rows: 1090
        })
        """
        # 加载评估集, 支持对每个评估集进行独立评估(eval_on_each_dataset)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )
        """
        print(eval_dataset)
        None
        """

    # 3. 数据集预处理与分词 (Tokenization Pipeline)
    # [为什么要这么写]: 再次使用 `main_process_first` 保护预处理过程.
    # [解决的问题]: 预处理涉及复杂的 Template 拼接和 Tokenizer 运算.
    # 同样只需主进程计算一次并存入 HF datasets 缓存, 其他子进程之后可直接 map 到内存镜像.
    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # 处理数据集划分(如果没有显式的验证集, 则根据 val_size 比例从训练集中切分)
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        """
        print(train_dict)
        {'train': Dataset({
            features: ['_prompt', '_response', '_system', '_tools', '_images', '_videos', '_audios'],
            num_rows: 1090
        })}
        print(eval_dict)
        {}
        """

        # 对训练集进行预处理: 将原始对话转为 input_ids 和 labels
        if "train" in train_dict:
            train_dict["train"] = _get_preprocessed_dataset(
                train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )
            """
            print(train_dict)
            {'train': Dataset({
                features: ['input_ids', 'attention_mask', 'labels', 'images', 'videos', 'audios'],
                num_rows: 1090
            })}
            """

        # 对所有评估集(可能有多个)进行预处理
        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        # 使用 DatasetDict 结构统一管理, 这是符合 Hugging Face Trainer 标准的最佳实践
        # Combine train and eval dictionaries
        dataset_dict = DatasetDict({**train_dict, **eval_dict})
        """
        print(dataset_dict)
        DatasetDict({
            train: Dataset({
                features: ['input_ids', 'attention_mask', 'labels', 'images', 'videos', 'audios'],
                num_rows: 1090
            })
        })
        """

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

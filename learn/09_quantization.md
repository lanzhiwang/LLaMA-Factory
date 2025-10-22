# 量化

参考: https://llamafactory.readthedocs.io/zh-cn/latest/advanced/quantization.html

```python
# ./src/llamafactory/extras/constants.py
@unique
class QuantizationMethod(str, Enum):
    r"""Borrowed from `transformers.utils.quantization_config.QuantizationMethod`."""

    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"

# ./src/llamafactory/hparams/model_args.py
@dataclass
class QuantizationArguments:
    r"""Arguments pertaining to the quantization method."""

    quantization_method: QuantizationMethod = field(
        default=QuantizationMethod.BNB,
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )

```

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_bnb_npu.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft

llamafactory-cli train examples/train_qlora/llama3_lora_sft_otfq.yaml \
model_name_or_path=/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct \
output_dir=saves/llama3-8b/lora/sft

```

"""
train_student.py
================

This script fine‑tunes a compact language model (the "student") on a
labelled dataset produced from a teacher model.  It supports LoRA or
DoRA adapters, mixed precision, gradient checkpointing and optional
QLoRA/k‑bit training.  At the end of training the merged model is
saved to the specified output directory where it can later be exported
to GGUF.

Example usage::

    python train_student.py \
        --dataset data/labelled.jsonl \
        --base_model qwen/Qwen2.5-1.5B-Instruct \
        --output_dir models/student \
        --use_dora \
        --qlora \
        --qlora_preset ampere-balanced \
        --num_epochs 1 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5

Recommended QLoRA presets (validated via ``tests/test_train_student_qlora.py``)
cover the following matrix of 4-bit configurations:

+-----------------+------------------------------------------+------------+----------------+--------------+-----------+---------+---------------+--------------------------------------------------------------------------+
| Preset          | Target GPU families                      | Quant type | Compute dtype  | Double quant | LoRA rank | LoRA α  | LoRA dropout | Notes                                                                    |
+=================+==========================================+============+================+==============+===========+=========+===============+==========================================================================+
| ampere-balanced | NVIDIA Ampere/Hopper (RTX30+/A100/A800/  | nf4        | bfloat16       | ✅            | 64        | 128     | 0.05          | Throughput baseline for bf16-capable cards.                              |
|                 | H100)                                    |            |                |              |           |         |               |                                                                          |
+-----------------+------------------------------------------+------------+----------------+--------------+-----------+---------+---------------+--------------------------------------------------------------------------+
| turing-safe     | NVIDIA Turing (RTX20/T4)                 | nf4        | float16        | ✅            | 32        | 64      | 0.10          | fp16 compute keeps compatibility with older GPUs.                        |
+-----------------+------------------------------------------+------------+----------------+--------------+-----------+---------+---------------+--------------------------------------------------------------------------+
| ada-memory      | NVIDIA Ada (RTX40 24 GB+)                | fp4        | bfloat16       | ❌            | 16        | 32      | 0.05          | Minimises memory by extending adapters to attention + MLP projections.   |
+-----------------+------------------------------------------+------------+----------------+--------------+-----------+---------+---------------+--------------------------------------------------------------------------+

Each preset can be selected via ``--qlora_preset`` (or through the UI)
to pre-populate safe defaults for adapter rank, alpha, dropout, target
modules and quantisation behaviour.

It is recommended to run this script on a machine with a GPU.  For
QLoRA, you must have the ``bitsandbytes`` library installed.

"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import (  # type: ignore
        LoraConfig,
        get_peft_model,
    )
    try:
        from peft import DORAConfig  # type: ignore
    except ImportError:  # pragma: no cover - depends on peft version
        DORAConfig = None  # type: ignore
except ImportError as e:
    raise ImportError(
        "Peft library is required for LoRA/DoRA. Install with `pip install peft`"
    ) from e

try:
    from peft.tuners.lora import (
        prepare_model_for_kbit_training,  # type: ignore
    )
except Exception:
    prepare_model_for_kbit_training = None  # type: ignore


@dataclass(frozen=True)
class QLoRAPreset:
    """Describe a vetted 4-bit configuration for QLoRA training."""

    label: str
    target_gpus: str
    quant_type: str
    compute_dtype: str
    double_quant: bool
    lora_rank: Optional[int] = None
    lora_alpha: Optional[float] = None
    lora_dropout: Optional[float] = None
    target_modules: Optional[str] = None
    notes: str = ""


QLORA_PRESETS: Dict[str, QLoRAPreset] = {
    "ampere-balanced": QLoRAPreset(
        label="Ampere balanced (bf16 NF4)",
        target_gpus="NVIDIA Ampere/Hopper (RTX30+/A100/A800/H100)",
        quant_type="nf4",
        compute_dtype="bfloat16",
        double_quant=True,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.05,
        notes="Optimised for throughput on bf16-capable cards.",
    ),
    "turing-safe": QLoRAPreset(
        label="Turing safe (fp16 NF4)",
        target_gpus="NVIDIA Turing (RTX20/T4) and older architectures",
        quant_type="nf4",
        compute_dtype="float16",
        double_quant=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.10,
        notes="fp16 compute keeps compatibility with GPUs lacking bf16 support.",
    ),
    "ada-memory": QLoRAPreset(
        label="Ada memory saver (bf16 FP4)",
        target_gpus="NVIDIA Ada (RTX40 24GB+)",
        quant_type="fp4",
        compute_dtype="bfloat16",
        double_quant=False,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj",
        notes="Reduces peak memory by targeting MLP projections in addition to attention blocks.",
    ),
}


def _resolve_compute_dtype(name: str) -> torch.dtype:
    """Convert a human readable dtype string into a torch dtype."""

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - defensive guard for new presets
        raise ValueError(f"Unsupported compute dtype '{name}' in QLoRA preset") from exc


def _build_4bit_config(preset: Optional[QLoRAPreset]) -> BitsAndBytesConfig:
    """Create a BitsAndBytes configuration using either preset or default values."""

    if preset is not None:
        compute_dtype = _resolve_compute_dtype(preset.compute_dtype)
        if compute_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            warnings.warn(
                (
                    f"Preset '{preset.label}' requests bf16 compute but support is missing; "
                    "falling back to float16."
                ),
                RuntimeWarning,
            )
            compute_dtype = torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=preset.double_quant,
            bnb_4bit_quant_type=preset.quant_type,
        )

    default_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=default_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _build_adapter_config(
    args: argparse.Namespace, preset: Optional[QLoRAPreset]
) -> Union["LoraConfig", "DORAConfig"]:
    """Create the appropriate LoRA/DoRA configuration using presets when available."""

    def pick(field: str):
        if preset is not None and getattr(preset, field) is not None:
            return getattr(preset, field)
        return getattr(args, field)

    target_spec = (
        preset.target_modules if preset is not None and preset.target_modules else args.target_modules
    )
    target_modules = [module.strip() for module in target_spec.split(",") if module.strip()]
    common_kwargs = dict(
        r=pick("lora_rank"),
        lora_alpha=pick("lora_alpha"),
        lora_dropout=pick("lora_dropout"),
        target_modules=target_modules,
    )

    if args.use_dora:
        if DORAConfig is None:
            raise RuntimeError("DoRA is not available in this PEFT version")
        return DORAConfig(**common_kwargs)

    return LoraConfig(bias="none", task_type="CAUSAL_LM", **common_kwargs)


def load_dataset_from_jsonl(path: Path) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset of strings."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "assistant" not in rec:
                raise ValueError(
                    f"Missing 'assistant' in record: {rec}. Make sure to label with teacher_label.py first."
                )
            records.append(rec)
    if not records:
        raise ValueError(f"No records found in dataset {path}. Ensure the file is not empty.")
    return Dataset.from_list(records)


def preprocess_function(examples: Dict[str, List[Any]], tokenizer, max_length: int) -> Dict[str, Any]:
    """Tokenize and concatenate the system/user/assistant for causal LM training."""
    input_texts = []
    assistants = examples["assistant"]
    system_hints = examples.get("system_hint", [""] * len(assistants))
    users = examples.get("user", [""] * len(assistants))
    if not (len(system_hints) == len(users) == len(assistants)):
        raise ValueError("Dataset fields must have matching lengths")
    for sys_hint, user, assistant in zip(system_hints, users, assistants):
        parts = []
        if sys_hint:
            parts.append(str(sys_hint).strip())
        parts.append(str(user).strip())
        parts.append(str(assistant).strip())
        input_texts.append("\n\n".join(parts))
    tokenized = tokenizer(
        input_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine tune a student model with LoRA/DoRA and QLoRA options."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the labelled JSONL dataset produced by teacher_label.py",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model identifier (e.g., qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the fine tuned model",
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        help="Use DoRA adapter instead of LoRA (requires peft>=0.10.0)",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Enable QLoRA (k-bit) training; requires bitsandbytes and a supported GPU",
    )
    parser.add_argument(
        "--qlora_preset",
        type=str,
        choices=sorted(QLORA_PRESETS.keys()),
        help="Name of a vetted QLoRA configuration preset (implies --qlora)",
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size (before gradient accumulation)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of steps to accumulate gradients (for effective batch size)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="LR scheduler type",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for scheduler",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank of the LoRA/DoRA adapters",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32,
        help="Alpha scaling factor for LoRA/DoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA/DoRA",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma separated list of target modules for LoRA/DoRA adapters",
    )
    args = parser.parse_args()
    if args.qlora_preset:
        args.qlora = True
    return args


def _prepare_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_data_pipeline(
    dataset_path: Path, tokenizer, max_length: int
) -> Tuple[Dataset, DataCollatorForLanguageModeling]:
    raw_dataset = load_dataset_from_jsonl(dataset_path)
    tokenized_dataset = raw_dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return tokenized_dataset, data_collator


def _initialise_model(args: argparse.Namespace, tokenizer) -> Any:
    preset = QLORA_PRESETS.get(args.qlora_preset) if args.qlora_preset else None
    model_kwargs: Dict[str, Any] = {}
    torch_dtype: Optional[torch.dtype] = torch.float16 if torch.cuda.is_available() else torch.float32
    if args.qlora:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError(
                "prepare_model_for_kbit_training is unavailable. "
                "Ensure peft>=0.7.0 and bitsandbytes are installed."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA-capable GPU.")
        quant_config = _build_4bit_config(preset)
        model_kwargs.update(device_map="auto", quantization_config=quant_config)
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        **model_kwargs,
    )

    model.config.use_cache = False
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    if args.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    adapter_config = _build_adapter_config(args, preset)
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    return model


def _select_mixed_precision_flags(args: argparse.Namespace) -> Tuple[bool, bool]:
    """Determine mutually exclusive mixed-precision flags for TrainingArguments."""

    if not torch.cuda.is_available() or args.qlora:
        return False, False

    if torch.cuda.is_bf16_supported():
        return False, True

    return True, False


def _create_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    fp16, bf16 = _select_mixed_precision_flags(args)

    return TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
    )


def _merge_and_save_model(model, tokenizer, output_dir: Path, use_qlora: bool) -> None:
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    else:
        merged = model

    if use_qlora:
        merged = merged.to(torch.float16)

    if hasattr(merged, "to"):
        merged = merged.to("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

def main() -> None:
    args = _parse_args()

    tokenizer = _prepare_tokenizer(args.base_model)
    tokenized_dataset, data_collator = _build_data_pipeline(
        args.dataset, tokenizer, args.max_length
    )
    model = _initialise_model(args, tokenizer)
    training_args = _create_training_arguments(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    _merge_and_save_model(model, tokenizer, args.output_dir, args.qlora)


if __name__ == "__main__":
    main()

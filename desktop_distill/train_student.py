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
        --num_epochs 1 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5

It is recommended to run this script on a machine with a GPU.  For
QLoRA, you must have the `bitsandbytes` library installed.

"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    from peft import LoraConfig, get_peft_model  # type: ignore
    # DoRA is optional and may not be available
    from peft import DORAConfig  # type: ignore
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
    for sys_hint, user, assistant in zip(
        examples.get("system_hint", [""] * len(examples["assistant"])),
        examples.get("user", [""] * len(examples["assistant"])),
        examples["assistant"],
    ):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine tune a student model with LoRA/DoRA and QLoRA options.")
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
        help="Enable QLoRA (k‑bit) training; requires bitsandbytes and a supported GPU",
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per‑device train batch size (before gradient accumulation)",
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
        default="cosine",  # can be linear, cosine, cosine_with_restarts, polynomial
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

    # Load dataset
    raw_dataset = load_dataset_from_jsonl(args.dataset)
    # Tokenizer must handle unknown tokens gracefully
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess the dataset
    tokenized_dataset = raw_dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

    # Create data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    model_kwargs: Dict[str, Any] = {}
    torch_dtype: Optional[torch.dtype]
    if args.qlora:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError(
                "prepare_model_for_kbit_training is unavailable. Please ensure you have peft >= 0.7.0 and bitsandbytes installed."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA-capable GPU.")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
        torch_dtype = None
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load base model
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

    # Build LoRA or DoRA adapter config
    target_mods = [m.strip() for m in args.target_modules.split(",")]
    if args.use_dora:
        # Use DoRA if available; fall back to LoRA if DORAConfig is missing
        if "DORAConfig" not in globals():
            raise RuntimeError("DoRA is not available in your version of peft. Use --use_dora only if supported.")
        adapter_config = DORAConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_mods,
        )
    else:
        adapter_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_mods,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()

    # Define training args; default to cosine scheduler but allow override
    fp16 = torch.cuda.is_available() and not args.qlora
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.qlora

    training_args = TrainingArguments(
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Merge adapters into the base model (for LoRA/DoRA) before saving
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    else:
        merged = model

    # For QLoRA, move back to CPU in float16 to ensure compatibility with export scripts
    if args.qlora:
        merged = merged.to(torch.float16)

    if hasattr(merged, "to"):
        merged = merged.to("cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()

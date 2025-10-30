# -*- coding: utf-8 -*-
"""
Train a student model with QLoRA+DoRA on labeled Chat dataset.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

from prompt_templates import extract_fields, llama_inst


@dataclass
class TrainConfig:
    student_model: str
    teacher_model: str
    data_train: str
    data_val: str
    data_train_labeled: str
    data_val_labeled: str
    output_dir: str
    max_seq_len: int
    packing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    lora_bias: str
    use_dora: bool
    load_in_4bit: bool
    bnb_quant_type: str
    bnb_double_quant: bool
    bnb_compute_dtype: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    gradient_checkpointing: bool
    optim: str
    bf16: bool
    fp16: bool
    seed: int


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(path: str) -> TrainConfig:
    cfg_dict = load_yaml(Path(path))
    d = cfg_dict
    return TrainConfig(
        student_model=d["student_model"],
        teacher_model=d.get("teacher_model", ""),
        data_train=d["data"]["train_file"],
        data_val=d["data"]["val_file"],
        data_train_labeled=d["data"]["labeled_train_file"],
        data_val_labeled=d["data"]["labeled_val_file"],
        output_dir=d["train"]["output_dir"],
        max_seq_len=int(d["data"]["max_seq_len"]),
        packing=bool(d["data"]["packing"]),
        lora_r=int(d["lora"]["r"]),
        lora_alpha=int(d["lora"]["alpha"]),
        lora_dropout=float(d["lora"]["dropout"]),
        lora_target_modules=list(d["lora"]["target_modules"]),
        lora_bias=str(d["lora"]["bias"]),
        use_dora=bool(d["lora"]["use_dora"]),
        load_in_4bit=bool(d["quantization"]["load_in_4bit"]),
        bnb_quant_type=str(d["quantization"]["bnb_4bit_quant_type"]),
        bnb_double_quant=bool(d["quantization"]["bnb_4bit_use_double_quant"]),
        bnb_compute_dtype=str(d["quantization"]["bnb_4bit_compute_dtype"]),
        num_train_epochs=int(d["train"]["num_train_epochs"]),
        per_device_train_batch_size=int(d["train"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(d["train"]["gradient_accumulation_steps"]),
        learning_rate=float(d["train"]["learning_rate"]),
        lr_scheduler_type=str(d["train"]["lr_scheduler_type"]),
        warmup_ratio=float(d["train"]["warmup_ratio"]),
        weight_decay=float(d["train"]["weight_decay"]),
        logging_steps=int(d["train"]["logging_steps"]),
        save_steps=int(d["train"]["save_steps"]),
        eval_steps=int(d["train"]["eval_steps"]),
        save_total_limit=int(d["train"]["save_total_limit"]),
        gradient_checkpointing=bool(d["train"]["gradient_checkpointing"]),
        optim=str(d["train"].get("optim", "adamw_torch")),
        bf16=bool(d["train"]["bf16"]),
        fp16=bool(d["train"]["fp16"]),
        seed=int(d["train"]["seed"]),
    )


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if "bfloat" in name:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported compute dtype: {name}")


def _format_single(example: Dict[str, Any]) -> str:
    fields = extract_fields(example)
    prompt = llama_inst(fields["system"], fields["user"], fields["assistant"])["prompt"]
    return prompt


def formatting_func(examples: Dict[str, Any] | List[Dict[str, Any]]) -> List[str]:
    if isinstance(examples, dict) and examples and isinstance(next(iter(examples.values())), list):
        batch_size = len(next(iter(examples.values())))
        outputs: List[str] = []
        for idx in range(batch_size):
            item = {key: value[idx] for key, value in examples.items()}
            outputs.append(_format_single(item))
        return outputs
    if isinstance(examples, list):
        return [_format_single(example) for example in examples]
    return [_format_single(examples)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = parse_config(args.config)
    set_seed(cfg.seed)

    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_double_quant,
            bnb_4bit_compute_dtype=dtype_from_name(cfg.bnb_compute_dtype),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(cfg.student_model, **model_kwargs)

    if hasattr(model.config, "use_cache") and cfg.gradient_checkpointing:
        model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )
    model = get_peft_model(model, peft_cfg)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        report_to="none",
        seed=cfg.seed,
    )

    train_data = load_dataset("json", data_files=cfg.data_train_labeled, split="train")
    val_data = load_dataset("json", data_files=cfg.data_val_labeled, split="train")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        formatting_func=formatting_func,
        max_seq_length=cfg.max_seq_len,
        packing=cfg.packing,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(json.dumps({"status": "ok", "output_dir": cfg.output_dir}))


if __name__ == "__main__":
    main()

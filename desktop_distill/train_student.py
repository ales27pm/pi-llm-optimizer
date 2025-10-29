#!/usr/bin/env python3
# Lightweight student trainer with PEFT (QLoRA/DoRA) and teacher-guided labels.
import argparse, json, math, os, sys
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

@dataclass
class Args:
    teacher: str
    student: str
    dataset: str
    out: str
    hf_token: str|None
    lr: float = 2e-4
    epochs: int = 1
    micro_bsz: int = 8
    grad_acc: int = 4
    cutoff_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

# For simplicity we assume the dataset JSONL already contains user/system + assistant fields for supervised distill.
# If assistant is empty, you can pre-label with Dolphin externally and then run this trainer.

SYSTEM_KEY = "system_hint"
USER_KEY = "user"
ASSIST_KEY = "assistant"

PROMPT_TMPL = "<|system|>\n{}\n<|user|>\n{}\n<|assistant|>\n"

def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--student', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hf-token', dest='hf_token')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--micro-bsz', type=int, default=8)
    ap.add_argument('--grad-acc', type=int, default=4)
    ap.add_argument('--cutoff-len', type=int, default=1024)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    a = ap.parse_args()
    return Args(a.teacher, a.student, a.dataset, a.out, a.hf_token, a.lr, a.epochs, a.micro_bsz, a.grad_acc, a.cutoff_len, a.lora_r, a.lora_alpha, a.lora_dropout)


def build_dataset(path: str):
    # Load JSONL as HF dataset
    ds = load_dataset('json', data_files={'train': path})['train']
    # Filter rows that have non-empty assistant text
    def has_label(x):
        return x.get(ASSIST_KEY) is not None and len(str(x.get(ASSIST_KEY))) > 0
    ds = ds.filter(has_label)
    return ds


def format_example(ex: dict) -> str:
    sysmsg = ex.get(SYSTEM_KEY, "Tu es un assistant québécois, honnête et curieux.")
    user = ex.get(USER_KEY, "")
    assistant = ex.get(ASSIST_KEY, "")
    return PROMPT_TMPL.format(sysmsg, user) + assistant


def tokenize(ds, tok, cutoff):
    def _tok(x):
        text = format_example(x)
        return tok(text, truncation=True, max_length=cutoff)
    return ds.map(_tok, batched=False, remove_columns=ds.column_names)


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    token = args.hf_token
    tok = AutoTokenizer.from_pretrained(args.student, use_fast=True, token=token)
    base = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float16, device_map='auto', token=token)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(base, lora)

    ds = build_dataset(args.dataset)
    tok_ds = tokenize(ds, tok, args.cutoff_len)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    steps_per_epoch = math.ceil(len(tok_ds) / (args.micro_bsz * args.grad_acc))

    training_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.micro_bsz,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=max(1, steps_per_epoch//10),
        save_strategy='epoch',
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tok_ds, data_collator=collator)
    trainer.train()

    # Merge LoRA and save as HF format
    model = model.merge_and_unload()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(json.dumps({"saved": str(out)}, indent=2))

if __name__ == '__main__':
    main()

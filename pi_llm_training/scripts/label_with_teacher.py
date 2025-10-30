# -*- coding: utf-8 -*-
"""
Generate assistant labels using a teacher LLM (e.g., Dolphin).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_templates import extract_fields, llama_inst


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batchify(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument(
        "--teacher-model",
        default="cognitivecomputations/Dolphin3.0-Llama3.1-8B",
    )
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if args.load_in_4bit
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

from typing import Dict, Iterable, List

def batchify(items: Iterable[Dict], batch_size: int):
    batch: List[Dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

    src_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = 0
    with out_path.open("w", encoding="utf-8") as w:
        for batch in batchify(iter_jsonl(src_path), args.batch_size):
            prompts = []
            metas = []
            for ex in batch:
                fld = extract_fields(ex)
                pack = llama_inst(fld["system"], fld["user"], assistant=None)
                prompts.append(pack["prompt"])
                metas.append(ex)
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for prompt, full_text, meta in zip(prompts, decoded, metas):
                assistant = full_text[len(prompt) :]
                if not full_text.startswith(prompt):
                    assistant = full_text
                meta_out = dict(meta)
                meta_out["assistant"] = assistant.strip()
                w.write(json.dumps(meta_out, ensure_ascii=False) + "\n")
                labeled += 1

    print(
        json.dumps(
            {"input": args.in_file, "output": args.out_file, "labeled": labeled},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

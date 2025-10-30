"""Label a dataset with a teacher LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
from prompt_templates import extract_fields, llama_inst
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield JSON objects from a newline-delimited JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def generate_batches(items: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    """Generate batches with up to ``batch_size`` elements."""
    batch: List[Dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", required=True, help="Path to the input JSONL file")
    parser.add_argument("--out-file", required=True, help="Path to write the labelled JSONL")
    parser.add_argument(
        "--teacher-model",
        default="cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        help="HuggingFace model identifier for the teacher LLM",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the teacher model with 4-bit quantization",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_quant_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    quant_config = build_quant_config(args.load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    src_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = 0
    with out_path.open("w", encoding="utf-8") as output_handle:
        for batch in generate_batches(iter_jsonl(src_path), args.batch_size):
            prompts: List[str] = []
            metadata: List[Dict] = []
            for example in batch:
                fields = extract_fields(example)
                packed = llama_inst(fields["system"], fields["user"], assistant=None)
                prompts.append(packed["prompt"])
                metadata.append(example)

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
            for prompt, full_text, meta in zip(prompts, decoded, metadata, strict=False):
                assistant = full_text[len(prompt) :]
                if not full_text.startswith(prompt):
                    assistant = full_text
                enriched = dict(meta)
                enriched["assistant"] = assistant.strip()
                output_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                labeled += 1

    print(
        json.dumps(
            {"input": args.in_file, "output": args.out_file, "labeled": labeled},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

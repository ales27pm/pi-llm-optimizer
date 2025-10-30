"""
teacher_label.py
===================

This script populates the `assistant` field of a JSONL dataset using a
pre‑trained language model (the "teacher").  It loads an input file
containing user prompts and optional metadata and produces an output file
containing the teacher's responses.  The resulting dataset can then be used
to train a smaller student model via knowledge distillation.

Each line in the input JSONL should at minimum define a `user` field with
the user's instruction.  Additional fields that influence the generation:

  - `system_hint`: text to prepend to the prompt.  Use this to supply a
    system message or domain context.  When present it will be separated
    from the user message by two newlines.
  - `requires_json`: boolean flag.  If true, the prompt will be
    augmented with an instruction to respond in JSON conforming to the
    provided schema.  The schema is passed verbatim via the
    `tool_schema` field and will be appended after the JSON instruction.
  - `tool_schema`: optional JSON schema (string) describing the
    expected shape of the JSON response.  Only relevant when
    `requires_json` is true.

Usage::

    python teacher_label.py --model openai/dolphin --input dataset.jsonl --output labelled.jsonl

You can set `--max_new_tokens` to control response length and disable
sampling for deterministic outputs with `--do_sample`.

Note: Generating responses from a large model can require a powerful GPU
or use of model parallelism.  Adjust the `device_map` argument if
necessary.
"""

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional dependency for CPU-only labelling
    torch = None  # type: ignore

from tqdm.auto import tqdm  # type: ignore

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )
except ImportError as exc:
    raise ImportError(
        "Please install transformers to use this script: pip install transformers"
    ) from exc


def build_prompt(record: Dict[str, object]) -> str:
    """Construct a textual prompt from a dataset record."""
    parts = []
    system_hint = record.get("system_hint")
    if system_hint:
        parts.append(str(system_hint).strip())
    user = record.get("user")
    if not user:
        raise ValueError("Missing 'user' field in record: {}".format(record))
    parts.append(str(user).strip())
    prompt = "\n\n".join(parts)
    # Append JSON instruction if needed
    if record.get("requires_json"):
        prompt += "\n\nVeuillez répondre en JSON."
        schema = record.get("tool_schema")
        if schema:
            prompt += "\nLe JSON doit respecter le schéma suivant:\n" + str(schema).strip()
    return prompt


def _batched(iterable: Iterable[Dict[str, object]], batch_size: int) -> Iterator[List[Dict[str, object]]]:
    """Yield successive batches from *iterable*."""

    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def generate_responses(
    model_name: str,
    records: Iterable[Dict[str, object]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    device_map: Optional[str] = None,
    batch_size: int = 1,
) -> Iterable[str]:
    """Yield generated responses from the teacher model for each record.

    This function initialises the HF model and tokenizer once and uses a
    text-generation pipeline for convenience.  Generation parameters can be
    adjusted via the function arguments.  The implementation handles batch
    generation to keep GPU utilisation high and avoids forcing an invalid
    ``device`` argument when Accelerate is already managing placement via
    ``device_map``.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_kwargs = {"device_map": device_map} if device_map else {}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # Prevent the pipeline from inserting BOS tokens multiple times
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    pipeline_kwargs = {
        "task": "text-generation",
        "model": model,
        "tokenizer": tokenizer,
    }
    if not device_map and torch is not None and torch.cuda.is_available():
        pipeline_kwargs["device"] = 0

    generator = pipeline(**pipeline_kwargs)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "eos_token_id": tokenizer.eos_token_id,
        "return_full_text": False,
    }

    for batch in _batched(records, batch_size):
        prompts = [build_prompt(rec) for rec in batch]
        results = generator(prompts, batch_size=batch_size, **generation_kwargs)
        for generated in results:
            sequences = generated if isinstance(generated, list) else [generated]
            if not sequences:
                raise RuntimeError("Teacher pipeline returned no completions for a prompt.")
            first = sequences[0]
            text = first.get("generated_text")
            if text is None:
                token_ids = first.get("generated_token_ids")
                if token_ids is None:
                    raise RuntimeError(
                        "Unexpected pipeline output without generated_text or generated_token_ids."
                    )
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
            yield text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a dataset with a teacher model")
    parser.add_argument("--model", type=str, required=True, help="HF model identifier of the teacher")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSONL file")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per example")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for model loading")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts to generate per batch")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip records that already contain an 'assistant' field",
    )
    args = parser.parse_args()

    records: List[Dict[str, object]] = []
    to_label: List[Dict[str, object]] = []
    with args.input.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if args.skip_existing and record.get("assistant"):
                record["_skip"] = True
            else:
                record["_skip"] = False
                to_label.append(record)
            records.append(record)

    responses_iter: Iterator[str]
    if to_label:
        responses_iter = iter(
            generate_responses(
                model_name=args.model,
                records=to_label,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                device_map=args.device_map,
                batch_size=args.batch_size,
            )
        )
    else:
        responses_iter = iter(())

    # Write augmented records to output
    with args.output.open("w", encoding="utf-8") as f_out:
        total_to_label = len(to_label)
        progress = tqdm(total=total_to_label, desc="Labelling") if total_to_label else None
        labelled = 0
        try:
            for rec in records:
                skip_flag = rec.pop("_skip")
                if not skip_flag:
                    try:
                        rec["assistant"] = next(responses_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Teacher model returned fewer responses than requested (stopped after "
                            f"{labelled} of {total_to_label})."
                        ) from exc
                    labelled += 1
                    if progress:
                        progress.update(1)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if labelled != total_to_label:
                raise RuntimeError(
                    f"Expected {total_to_label} labelled examples but wrote {labelled}."
                )
        finally:
            if progress:
                progress.close()


if __name__ == "__main__":
    main()

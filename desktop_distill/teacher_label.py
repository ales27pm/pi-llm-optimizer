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
from pathlib import Path
from typing import Dict, Iterable, Optional

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


def generate_responses(
    model_name: str,
    records: Iterable[Dict[str, object]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    device_map: Optional[str] = None,
) -> Iterable[str]:
    """Yield generated responses from the teacher model for each record.

    This function initialises the HF model and tokenizer once and uses a
    text‑generation pipeline for convenience.  Generation parameters can be
    adjusted via the function arguments.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    # Prevent the pipeline from inserting BOS tokens multiple times
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device_map else -1,
    )
    for rec in records:
        prompt = build_prompt(rec)
        # The pipeline returns a list of dicts with the key 'generated_text'.
        # We slice off the prompt to yield only the newly generated portion.
        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"]
        response = result[len(prompt) :].strip()
        yield response


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a dataset with a teacher model")
    parser.add_argument("--model", type=str, required=True, help="HF model identifier of the teacher")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSONL file")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per example")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for model loading")
    args = parser.parse_args()

    records = []
    with args.input.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    responses = generate_responses(
        model_name=args.model,
        records=records,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        device_map=args.device_map,
    )

    # Write augmented records to output
    with args.output.open("w", encoding="utf-8") as f_out:
        for rec, resp in tqdm(zip(records, responses), total=len(records), desc="Labelling"):
            rec["assistant"] = resp
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
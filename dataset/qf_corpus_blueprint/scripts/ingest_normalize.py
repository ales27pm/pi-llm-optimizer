#!/usr/bin/env python3
"""Normalize and enrich Québec French corpus records."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Set

try:  # pragma: no cover - import path differs between CLI and package usage
    from .jsonl_utils import JsonlWriter, iter_jsonl
    from .normalizer import NormalizationEngine, NeuralLemmatizer, normalize_text_qf
except ImportError:  # pragma: no cover - fallback for direct script execution
    from jsonl_utils import JsonlWriter, iter_jsonl
    from normalizer import NormalizationEngine, NeuralLemmatizer, normalize_text_qf

LOGGER = logging.getLogger("qf.normalize")

SCRIPT_DIR = Path(__file__).resolve().parent
VOCAB_DIR = SCRIPT_DIR.parent / "vocab"


def _resolve_regex_flags(flag_names: Sequence[str]) -> int:
    import re

    flag_value = 0
    for name in flag_names:
        try:
            flag_value |= getattr(re, name)
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported regex flag '{name}' in configuration") from exc
    return flag_value


def _load_dialect_rules() -> List[tuple]:
    import re

    rules_path = VOCAB_DIR / "dialect_rules.json"
    raw_rules = json.loads(rules_path.read_text(encoding="utf-8"))
    compiled: List[tuple] = []
    for entry in raw_rules:
        pattern = re.compile(entry["pattern"], _resolve_regex_flags(entry.get("flags", [])))
        compiled.append((pattern, entry["tags"]))
    return compiled


DIALECT_RULES = _load_dialect_rules()
JOUAL_TRIGGER_TAGS = {
    "Contraction_QF",
    "Anglicism_Lexical",
    "Lexeme_Tabarnak",
    "Asteur_Form",
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize & enrich Québec French JSONL data.")
    parser.add_argument("--in-file", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--out-file", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--default-register", type=str, default="Casual", help="Register fallback.")
    parser.add_argument("--default-region", type=str, default="Montréal", help="Region fallback.")
    parser.add_argument("--default-age-group", type=str, default="26_40", help="Age group fallback.")
    parser.add_argument("--default-education", type=str, default="University", help="Education fallback.")
    parser.add_argument("--default-gender", type=str, default="Unspecified", help="Gender fallback.")
    parser.add_argument(
        "--time-period", type=str, default="2001-2050", help="Time period code to apply when missing."
    )
    parser.add_argument(
        "--source-corpus-id",
        type=str,
        default="qf_blueprint",
        help="Source corpus identifier used when records omit the field.",
    )
    parser.add_argument(
        "--append", action="store_true", help="Append to the output file instead of overwriting."
    )
    parser.add_argument(
        "--lemmatizer-backend",
        type=str,
        default="auto",
        help="Neural lemmatizer backend: auto, spacy, stanza or none.",
    )
    parser.add_argument(
        "--lemmatizer-model",
        type=str,
        help="Optional model identifier for the neural lemmatizer backend.",
    )
    return parser.parse_args(list(argv))


def _initialise_lemmatizer(args: argparse.Namespace) -> NeuralLemmatizer:
    backend = (args.lemmatizer_backend or "auto").lower()
    if backend == "auto":
        for candidate in ("spacy", "stanza"):
            lemmatizer = NeuralLemmatizer(candidate, args.lemmatizer_model)
            if lemmatizer.is_available():
                LOGGER.info("Using %s backend for neural lemmatization", candidate)
                return lemmatizer
        LOGGER.info("Falling back to rule-only normalization (no neural lemmatizer available)")
        return NeuralLemmatizer("none")
    return NeuralLemmatizer(backend, args.lemmatizer_model)


def detect_dialect_tags(text: str) -> Set[str]:
    tags: Set[str] = set()
    for pattern, tag_list in DIALECT_RULES:
        if pattern.search(text):
            tags.update(tag_list)
    if "Register_Joual" not in tags and tags & JOUAL_TRIGGER_TAGS:
        tags.add("Register_Joual")
    return tags


def enrich_record(record: dict, defaults: argparse.Namespace, *, engine: NormalizationEngine, lemmatizer: NeuralLemmatizer) -> dict:
    record = dict(record)  # copy to avoid mutating caller state
    raw_text = record.get("text_orthographic_raw", "")
    if not raw_text:
        raise ValueError("Each record must include 'text_orthographic_raw'.")

    normalized_payload = normalize_text_qf(raw_text, engine=engine, lemmatizer=lemmatizer)
    record["text_normalized_mf"] = normalized_payload["output"]
    record["normalization_audit"] = normalized_payload["applied_rules"]

    tags: Set[str] = set(record.get("dialect_tag_list", []))
    tags.update(normalized_payload.get("rule_tags", []))
    tags.update(detect_dialect_tags(raw_text))
    record["dialect_tag_list"] = sorted(tags)

    if normalized_payload.get("lemmatizer", {}).get("lemmas"):
        record["lemmatization"] = normalized_payload["lemmatizer"]

    sociolinguistic = dict(record.get("sociolinguistic_parameters", {}))
    sociolinguistic.setdefault("region_qc", defaults.default_region)
    sociolinguistic.setdefault("age_group", defaults.default_age_group)
    sociolinguistic.setdefault("education_level", defaults.default_education)
    sociolinguistic.setdefault("gender", defaults.default_gender)
    record["sociolinguistic_parameters"] = sociolinguistic

    record.setdefault("register", defaults.default_register)
    record.setdefault("time_period_code", defaults.time_period)
    record.setdefault("source_corpus_id", defaults.source_corpus_id)

    return record


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.in_file.exists():
        LOGGER.error("Input file not found: %s", args.in_file)
        return 2

    if args.append:
        LOGGER.warning(
            "Appending to the output file may introduce duplicate records; new records will be "
            "deduplicated within this run."
        )

    stats = Counter()
    written_records = 0

    lemmatizer = _initialise_lemmatizer(args)
    engine = NormalizationEngine.from_yaml()

    try:
        writer = JsonlWriter(args.out_file, append=args.append, deduplicate=args.append)
    except OSError as exc:  # pragma: no cover - early filesystem error
        LOGGER.error("Unable to prepare output %s: %s", args.out_file, exc)
        return 4

    with writer:
        try:
            for line_number, record in iter_jsonl(args.in_file):
                try:
                    enriched_record = enrich_record(record, args, engine=engine, lemmatizer=lemmatizer)
                except ValueError as exc:
                    LOGGER.error(
                        "Skipping record at line %s missing critical data: %s", line_number, exc
                    )
                    continue
                if writer.write(enriched_record):
                    stats[enriched_record["register"]] += 1
                    written_records += 1
        except ValueError as exc:
            LOGGER.error("Failed to read input: %s", exc)
            return 3
        except OSError as exc:
            LOGGER.error("Unable to write output %s: %s", args.out_file, exc)
            return 4

    LOGGER.info("Wrote %s enriched records to %s", written_records, args.out_file)
    if stats:
        LOGGER.info("Register distribution: %s", dict(stats))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

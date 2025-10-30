#!/usr/bin/env python3
"""Normalize and enrich Québec French corpus records."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

LOGGER = logging.getLogger("qf.normalize")

NORMALIZATION_RULES = [
    (re.compile(r"\bà matin\b", re.IGNORECASE), "ce matin"),
    (re.compile(r"\ba matin\b", re.IGNORECASE), "ce matin"),
    (re.compile(r"\basteur\b", re.IGNORECASE), "maintenant"),
    (re.compile(r"\bsti\b", re.IGNORECASE), "cela dit"),
    (re.compile(r"\bchu\b", re.IGNORECASE), "je suis"),
    (re.compile(r"\bchuis\b", re.IGNORECASE), "je suis"),
    (re.compile(r"\bj't\b", re.IGNORECASE), "je te"),
    (re.compile(r"\b(j'|t')?\s*es\b", re.IGNORECASE), "tu es"),
]

ENGLISH_HEURISTIC = re.compile(r"\b(?:hey|ok|anyway|whatever|cool|weekend|lunch)\b", re.IGNORECASE)
CONTRACTION_PATTERN = re.compile(r"\b[tdjlcs]'\w+", re.IGNORECASE)
ELISION_PATTERN = re.compile(r"\b[dtjl]'[a-zàâçéèêëîïôùûüœ]+", re.IGNORECASE)
AFFRICATION_PATTERN = re.compile(r"\bts[ié]|dz[ié]", re.IGNORECASE)
QUESTION_PARTICLE_PATTERN = re.compile(r"\b\w+-tu\b", re.IGNORECASE)
SACRE_PATTERN = re.compile(r"\b(tabarnak|câlice|osti|sacrament)\b", re.IGNORECASE)
ASTEURN_PATTERN = re.compile(r"\basteur\b", re.IGNORECASE)
A_MATIN_PATTERN = re.compile(r"\b[àa]\s+matin\b", re.IGNORECASE)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize & enrich Québec French JSONL data.")
    parser.add_argument("--in-file", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--out-file", type=Path, required=True, help="Destination JSONL file.")
    parser.add_argument("--default-register", type=str, default="Casual", help="Register fallback.")
    parser.add_argument("--default-region", type=str, default="Montréal", help="Region fallback.")
    parser.add_argument("--default-age-group", type=str, default="26_40", help="Age group fallback.")
    parser.add_argument("--default-education", type=str, default="University", help="Education fallback.")
    parser.add_argument("--default-gender", type=str, default="Unspecified", help="Gender fallback.")
    parser.add_argument("--time-period", type=str, default="2001-2050", help="Time period code to apply when missing.")
    parser.add_argument(
        "--source-corpus-id",
        type=str,
        default="qf_blueprint",
        help="Source corpus identifier used when records omit the field.",
    )
    parser.add_argument(
        "--append", action="store_true", help="Append to the output file instead of overwriting."
    )
    return parser.parse_args(list(argv))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ")
    for pattern, replacement in NORMALIZATION_RULES:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def detect_dialect_tags(text: str) -> Set[str]:
    tags: Set[str] = set()
    if ENGLISH_HEURISTIC.search(text):
        tags.add("Anglicism_Lexical")
        tags.add("CodeSwitch_English")
    if CONTRACTION_PATTERN.search(text):
        tags.add("Contraction_QF")
    if ELISION_PATTERN.search(text):
        tags.add("Elision_Determiner")
    if AFFRICATION_PATTERN.search(text):
        tags.add("Affrication_Coronal")
    if QUESTION_PARTICLE_PATTERN.search(text):
        tags.add("Phonological_Tu_S")
    if SACRE_PATTERN.search(text):
        tags.add("Lexeme_Tabarnak")
        tags.add("Register_Joual")
    if ASTEURN_PATTERN.search(text):
        tags.add("Asteur_Form")
    if A_MATIN_PATTERN.search(text):
        tags.add("A_Matin_Form")
    if "Register_Joual" not in tags and ("Contraction_QF" in tags or "Anglicism_Lexical" in tags):
        tags.add("Register_Joual")
    return tags


def enrich_record(record: dict, defaults: argparse.Namespace) -> dict:
    record = dict(record)  # copy to avoid mutating caller state
    raw_text = record.get("text_orthographic_raw", "")
    if not raw_text:
        raise ValueError("Each record must include 'text_orthographic_raw'.")

    normalized = record.get("text_normalized_mf") or normalize_text(raw_text)
    record["text_normalized_mf"] = normalized

    tags: Set[str] = set(record.get("dialect_tag_list", []))
    tags.update(detect_dialect_tags(raw_text))
    record["dialect_tag_list"] = sorted(tags)

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


def load_records(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc
    return data


def write_records(path: Path, records: Iterable[dict], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.in_file.exists():
        LOGGER.error("Input file not found: %s", args.in_file)
        return 2

    try:
        records = load_records(args.in_file)
    except ValueError as exc:
        LOGGER.error("Failed to read input: %s", exc)
        return 3

    enriched = []
    stats = Counter()
    for record in records:
        try:
            enriched_record = enrich_record(record, args)
        except ValueError as exc:
            LOGGER.error("Skipping record missing critical data: %s", exc)
            continue
        enriched.append(enriched_record)
        stats[enriched_record["register"]] += 1

    try:
        write_records(args.out_file, enriched, args.append)
    except OSError as exc:
        LOGGER.error("Unable to write output %s: %s", args.out_file, exc)
        return 4

    LOGGER.info("Wrote %s enriched records to %s", len(enriched), args.out_file)
    if stats:
        LOGGER.info("Register distribution: %s", dict(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())

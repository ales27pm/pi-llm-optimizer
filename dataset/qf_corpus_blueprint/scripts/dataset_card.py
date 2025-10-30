"""Build dataset cards for Québec French corpus splits.

This module derives distribution statistics, provenance metadata and a
cryptographic fingerprint for enriched Québec French JSONL datasets. The
resulting payload complies with
``dataset/qf_corpus_blueprint/schema/dataset.card.schema.json`` and can be
written directly to disk as a dataset card.

The script is intentionally importable so higher level automation (tests,
Textual UI, notebooks) can reuse the helpers, while also exposing a CLI for
standalone use::

    python dataset_card.py \
        --data dataset/qf_corpus_blueprint/examples/enriched.jsonl \
        --output cards/train.json \
        --split-name train \
        --license "CC-BY-4.0" \
        --schema-version 1.0.0

The CLI validates basic invariants (non-empty dataset, at least two registers,
valid ISO date) and reports actionable errors to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

LOGGER = logging.getLogger("qf.dataset_card")

try:  # pragma: no cover - import path differs when executed as a script
    from .jsonl_utils import load_jsonl
except ImportError:  # pragma: no cover - fallback for ``python dataset_card.py``
    from jsonl_utils import load_jsonl  # type: ignore


SplitName = str


def _normalise_distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: counter[key] / total for key in sorted(counter)}


def _collect_registers(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        value = str(record.get("register") or "unspecified").strip()
        counter[value or "unspecified"] += 1
    return counter


def _collect_dialect_tags(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for tag in record.get("dialect_tag_list", []) or []:
            counter[str(tag)] += 1
    return counter


def _collect_source_ids(records: Iterable[Mapping[str, object]]) -> list[str]:
    source_ids = {str(record.get("source_corpus_id") or "unknown").strip() or "unknown" for record in records}
    return sorted(source_ids)


def compute_register_distribution(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Return the register probability distribution for the dataset."""

    counter = _collect_registers(records)
    distribution = _normalise_distribution(counter)
    if len(distribution) < 2:
        raise ValueError(
            "Dataset card requires at least two register categories to describe balance."
        )
    return distribution


def compute_dialect_distribution(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Return the dialect tag probability distribution when tags are present."""

    counter = _collect_dialect_tags(records)
    return _normalise_distribution(counter)


def compute_register_balance(distribution: Mapping[str, float]) -> dict[str, float]:
    """Return summary statistics describing register balance."""

    if not distribution:
        return {}
    shares = list(distribution.values())
    return {
        "max_share": max(shares),
        "min_share": min(shares),
        "std_dev": statistics.pstdev(shares) if len(shares) > 1 else 0.0,
    }


def compute_file_sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_card(
    records: Sequence[MutableMapping[str, object]],
    *,
    split_name: SplitName,
    license_text: str,
    schema_version: str,
    creation_date: str,
    dataset_sha: str,
    source_ids: Sequence[str] | None = None,
    notes: str | None = None,
    aggregated_from: Sequence[str] | None = None,
    processing_steps: Sequence[str] | None = None,
    tool_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Assemble a dataset card structure from the provided records."""

    if not records:
        raise ValueError("Cannot build a dataset card from an empty dataset.")

    register_distribution = compute_register_distribution(records)
    dialect_distribution = compute_dialect_distribution(records)
    register_balance = compute_register_balance(register_distribution)

    source_id_list = sorted(source_ids) if source_ids else _collect_source_ids(records)
    if not source_id_list:
        raise ValueError("At least one source identifier is required to document provenance.")

    provenance: dict[str, object] = {}
    aggregated = list(aggregated_from) if aggregated_from else list(source_id_list)
    if aggregated:
        provenance["aggregated_from"] = aggregated
    if processing_steps:
        provenance["processing_steps"] = list(processing_steps)
    if tool_versions:
        provenance["tool_versions"] = {key: tool_versions[key] for key in sorted(tool_versions)}

    card: dict[str, object] = {
        "split_name": split_name,
        "record_count": len(records),
        "register_distribution": register_distribution,
        "source_ids": source_id_list,
        "schema_version": schema_version,
        "license": license_text,
        "sha256": dataset_sha,
        "creation_date": creation_date,
    }
    if notes:
        card["notes"] = notes
    if dialect_distribution:
        card["dialect_distribution"] = dialect_distribution
    if register_balance:
        card["register_balance_check"] = register_balance
    if provenance:
        card["provenance"] = provenance
    return card


def _parse_tool_versions(entries: Sequence[str] | None) -> dict[str, str]:
    tool_versions: dict[str, str] = {}
    if not entries:
        return tool_versions
    for entry in entries:
        if "=" not in entry:
            raise ValueError("Tool versions must use the NAME=VERSION format.")
        name, version = entry.split("=", 1)
        name = name.strip()
        version = version.strip()
        if not name or not version:
            raise ValueError("Tool version entries cannot be empty.")
        tool_versions[name] = version
    return tool_versions


def _validate_iso_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc
    return value


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Québec French dataset card.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the enriched JSONL dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the dataset card JSON.")
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        choices=["train", "validation", "test", "dev", "production", "analysis"],
        help="Dataset split name.",
    )
    parser.add_argument("--license", dest="license_text", type=str, required=True, help="License identifier for the split.")
    parser.add_argument("--schema-version", type=str, default="1.0.0", help="Semantic version of the dataset card schema.")
    parser.add_argument(
        "--creation-date",
        type=_validate_iso_date,
        default=date.today().isoformat(),
        help="ISO-8601 creation date (defaults to today).",
    )
    parser.add_argument("--note", dest="notes", type=str, help="Optional free-form notes to embed in the card.")
    parser.add_argument("--source-id", dest="source_ids", action="append", help="Explicit source corpus identifier.")
    parser.add_argument(
        "--aggregated-from",
        action="append",
        help="Explicit provenance list; defaults to the detected source identifiers.",
    )
    parser.add_argument(
        "--processing-step",
        action="append",
        help="Processing step applied to produce this split (may be repeated).",
    )
    parser.add_argument(
        "--tool-version",
        action="append",
        help="Record tool versions in NAME=VERSION form (may be repeated).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.data.exists():
        LOGGER.error("Dataset not found: %s", args.data)
        return 2

    try:
        records = load_jsonl(args.data)
    except ValueError as exc:
        LOGGER.error("Failed to read dataset: %s", exc)
        return 3

    if not records:
        LOGGER.error("Dataset %s is empty; cannot build a dataset card.", args.data)
        return 4

    try:
        tool_versions = _parse_tool_versions(args.tool_version)
        card = build_dataset_card(
            records,
            split_name=args.split_name,
            license_text=args.license_text,
            schema_version=args.schema_version,
            creation_date=args.creation_date,
            dataset_sha=compute_file_sha256(args.data),
            source_ids=args.source_ids,
            notes=args.notes,
            aggregated_from=args.aggregated_from,
            processing_steps=args.processing_step,
            tool_versions=tool_versions,
        )
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 5

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote dataset card to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "build_dataset_card",
    "compute_file_sha256",
    "compute_register_distribution",
    "compute_dialect_distribution",
    "compute_register_balance",
    "main",
    "parse_args",
]

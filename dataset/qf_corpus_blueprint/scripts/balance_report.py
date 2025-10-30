#!/usr/bin/env python3
"""Generate balance & coverage stats for Québec French corpus JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

LOGGER = logging.getLogger("qf.balance")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute coverage statistics for a QF JSONL dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the enriched JSONL file.")
    parser.add_argument(
        "--top-n-tags", type=int, default=10, help="Number of dialect tags to show in the summary table."
    )
    return parser.parse_args(list(argv))


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc
    return records


def summarize(records: List[dict]) -> Dict[str, Counter]:
    register_counter = Counter()
    time_counter = Counter()
    region_counter = Counter()
    dialect_counter = Counter()
    missing_fields = defaultdict(int)

    for record in records:
        register = record.get("register")
        if register:
            register_counter[register] += 1
        else:
            missing_fields["register"] += 1

        time_period = record.get("time_period_code")
        if time_period:
            time_counter[time_period] += 1
        else:
            missing_fields["time_period_code"] += 1

        sociolinguistic = record.get("sociolinguistic_parameters") or {}
        region = sociolinguistic.get("region_qc")
        if region:
            region_counter[region] += 1
        else:
            missing_fields["sociolinguistic_parameters.region_qc"] += 1

        for tag in record.get("dialect_tag_list", []):
            dialect_counter[tag] += 1

    return {
        "register": register_counter,
        "time_period": time_counter,
        "region": region_counter,
        "dialect": dialect_counter,
        "missing": Counter(missing_fields),
    }


def print_section(title: str, counter: Counter, total: int) -> None:
    if not counter:
        print(f"\n{title}: none")
        return
    print(f"\n{title} (total={total}):")
    for key, count in counter.most_common():
        percentage = (count / total * 100) if total else 0
        print(f"  - {key}: {count} ({percentage:.1f}%)")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.data.exists():
        LOGGER.error("Data file not found: %s", args.data)
        return 2

    try:
        records = load_jsonl(args.data)
    except ValueError as exc:
        LOGGER.error("Failed to read dataset: %s", exc)
        return 3

    if not records:
        LOGGER.warning("No records found in %s", args.data)
        return 0

    summary = summarize(records)
    total_records = len(records)

    print(f"Dataset: {args.data}")
    print(f"Records: {total_records}")

    print_section("Register distribution", summary["register"], total_records)
    print_section("Time period distribution", summary["time_period"], total_records)
    print_section("Region distribution", summary["region"], total_records)

    top_n = args.top_n_tags
    dialect_counter = summary["dialect"]
    if dialect_counter:
        print(f"\nTop {top_n} dialect tags:")
        for tag, count in dialect_counter.most_common(top_n):
            percentage = count / total_records * 100
            print(f"  - {tag}: {count} occurrences ({percentage:.1f}% of records)")
    else:
        print("\nNo dialect tags present in the dataset.")

    missing_counter = summary["missing"]
    if missing_counter:
        print("\nMissing metadata fields:")
        for key, count in missing_counter.most_common():
            print(f"  - {key}: missing in {count} record(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

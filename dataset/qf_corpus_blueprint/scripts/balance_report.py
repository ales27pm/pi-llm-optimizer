#!/usr/bin/env python3
"""Generate balance & coverage stats for Québec French corpus JSONL files."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from jsonl_utils import load_jsonl

LOGGER = logging.getLogger("qf.balance")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute coverage statistics for a QF JSONL dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the enriched JSONL file.")
    parser.add_argument(
        "--top-n-tags", type=int, default=10, help="Number of dialect tags to show in the summary table."
    )
    return parser.parse_args(list(argv))


def summarize(records: List[dict]) -> Dict[str, Counter]:
    register_counter = Counter()
    time_counter = Counter()
    region_counter = Counter()
    dialect_counter = Counter()
    missing_fields_absent = defaultdict(int)
    missing_fields_null = defaultdict(int)

    for record in records:
        if (register := record.get("register")):
            register_counter[register] += 1
        else:
            if "register" in record:
                missing_fields_null["register"] += 1
            else:
                missing_fields_absent["register"] += 1

        if (time_period := record.get("time_period_code")):
            time_counter[time_period] += 1
        else:
            key = "time_period_code"
            if key in record:
                missing_fields_null[key] += 1
            else:
                missing_fields_absent[key] += 1

        sociolinguistic = record.get("sociolinguistic_parameters") or {}
        if (region := sociolinguistic.get("region_qc")):
            region_counter[region] += 1
        else:
            field_name = "sociolinguistic_parameters.region_qc"
            if "region_qc" in sociolinguistic:
                missing_fields_null[field_name] += 1
            else:
                missing_fields_absent[field_name] += 1

        for tag in record.get("dialect_tag_list", []):
            dialect_counter[tag] += 1

    return {
        "register": register_counter,
        "time_period": time_counter,
        "region": region_counter,
        "dialect": dialect_counter,
        "missing_absent": Counter(missing_fields_absent),
        "missing_null": Counter(missing_fields_null),
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

    if dialect_counter := summary["dialect"]:
        top_n = args.top_n_tags
        print(f"\nTop {top_n} dialect tags:")
        for tag, count in dialect_counter.most_common(top_n):
            percentage = count / total_records * 100
            print(f"  - {tag}: {count} occurrences ({percentage:.1f}% of records)")
    else:
        print("\nNo dialect tags present in the dataset.")

    if missing_absent := summary["missing_absent"]:
        print("\nMissing metadata fields (absent):")
        for key, count in missing_absent.most_common():
            print(f"  - {key}: absent in {count} record(s)")

    if missing_null := summary["missing_null"]:
        print("\nMissing metadata fields (null/empty):")
        for key, count in missing_null.most_common():
            print(f"  - {key}: null or empty in {count} record(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

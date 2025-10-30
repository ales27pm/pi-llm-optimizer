#!/usr/bin/env python3
"""Validate JSONL corpus records against the Québec French schema."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

try:
    from jsonschema import Draft7Validator  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "jsonschema is required for validation. Install it with 'pip install jsonschema'."
    ) from exc

from jsonl_utils import iter_jsonl


LOGGER = logging.getLogger("qf.validate")


def validate_records(schema_path: Path, data_path: Path, fail_fast: bool) -> int:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)

    error_count = 0
    for line_number, record in iter_jsonl(data_path):
        if errors := sorted(validator.iter_errors(record), key=lambda e: e.path):
            error_count += len(errors)
            for err in errors:
                location = " / ".join(map(str, err.absolute_path)) or "<root>"
                LOGGER.error(
                    "Record at line %s failed validation (%s): %s",
                    line_number,
                    location,
                    err.message,
                )
            if fail_fast:
                break
    return error_count


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Québec French corpus JSONL data against the schema."
    )
    parser.add_argument("--schema", type=Path, required=True, help="Path to the JSON Schema file.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the JSONL data file.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first validation error instead of reporting all issues.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.schema.exists():
        LOGGER.error("Schema file not found: %s", args.schema)
        return 2
    if not args.data.exists():
        LOGGER.error("Data file not found: %s", args.data)
        return 2

    try:
        error_count = validate_records(args.schema, args.data, args.fail_fast)
    except json.JSONDecodeError:
        LOGGER.error("Stopped due to JSON parsing error.")
        return 3
    except Exception as exc:  # pragma: no cover - safeguard unexpected
        LOGGER.exception("Unexpected validation failure: %s", exc)
        return 4

    if error_count:
        LOGGER.error("Validation completed with %s error(s).", error_count)
        return 1

    LOGGER.info("All records are valid 🎉")
    return 0


if __name__ == "__main__":
    sys.exit(main())

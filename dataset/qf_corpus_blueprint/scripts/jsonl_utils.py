"""Utilities for streaming JSONL data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Tuple


def iter_jsonl(path: Path) -> Iterator[Tuple[int, dict]]:
    """Yield (line_number, record) pairs from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - validation occurs upstream
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict]:
    """Load every record from a JSONL file into memory."""

    return [record for _, record in iter_jsonl(path)]


class JsonlWriter:
    """Context manager that writes JSON objects to a JSONL file."""

    def __init__(self, path: Path, *, append: bool = False, deduplicate: bool = False) -> None:
        self._path = path
        self._append = append
        self._deduplicate = deduplicate
        self._handle = None
        self._seen: set[str] | None = set() if deduplicate else None

    def __enter__(self) -> "JsonlWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        self._handle = self._path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def write(self, record: dict) -> bool:
        if self._handle is None:  # pragma: no cover - misuse guard
            raise RuntimeError("JsonlWriter must be used as a context manager")

        serialized = json.dumps(record, ensure_ascii=False)
        if self._seen is not None:
            if serialized in self._seen:
                return False
            self._seen.add(serialized)

        self._handle.write(serialized)
        self._handle.write("\n")
        return True


def write_jsonl(path: Path, records: Iterable[dict], *, append: bool = False) -> None:
    """Write an iterable of records to disk."""

    with JsonlWriter(path, append=append) as writer:
        for record in records:
            writer.write(record)

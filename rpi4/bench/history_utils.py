from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

DEFAULT_HISTORY_LIMIT = 200


def load_history(path: Path) -> list[dict[str, Any]]:
    """Return benchmark history records sorted by timestamp.

    The file is expected to contain a JSON array of objects. Any malformed
    entries are dropped and the function returns an empty list when the file does
    not exist or cannot be parsed. Errors are logged as warnings so callers can
    surface them to operators without interrupting the workflow.
    """

    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse benchmark history at %s: %s", path, exc)
        return []

    if not isinstance(payload, list):
        LOGGER.warning("Unexpected history payload at %s: expected list, got %s", path, type(payload).__name__)
        return []

    records: list[dict[str, Any]] = []
    for entry in payload:
        if isinstance(entry, dict):
            records.append(entry)
        else:
            LOGGER.warning("Discarding malformed history entry in %s: %r", path, entry)

    records.sort(key=lambda item: item.get("timestamp", ""))
    return records


def _trim_history(records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    items = list(records)
    if limit <= 0:
        return items
    if len(items) <= limit:
        return items
    return items[-limit:]


def append_history(path: Path, record: dict[str, Any], *, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, Any]]:
    """Append ``record`` to the JSON history file and return the updated list."""

    history = load_history(path)
    history.append(record)
    trimmed = _trim_history(history, limit)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trimmed, indent=2, sort_keys=True), encoding="utf-8")
    return trimmed

from __future__ import annotations

import json
from pathlib import Path

from rpi4.bench.history_utils import append_history, load_history


def test_append_history_creates_file(tmp_path: Path) -> None:
    history_path = tmp_path / "bench_history.json"
    record = {
        "timestamp": "2024-01-01T00:00:00Z",
        "model": "model.gguf",
    }

    updated = append_history(history_path, record, limit=5)

    assert history_path.exists()
    assert updated == [record]
    saved = json.loads(history_path.read_text(encoding="utf-8"))
    assert saved == [record]


def test_append_history_trims_records(tmp_path: Path) -> None:
    history_path = tmp_path / "bench_history.json"
    for index in range(10):
        append_history(history_path, {"timestamp": f"2024-01-01T00:00:{index:02d}Z"}, limit=3)

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(history) == 3
    assert history[0]["timestamp"].endswith("07Z")
    assert history[-1]["timestamp"].endswith("09Z")


def test_load_history_handles_invalid_payload(tmp_path: Path) -> None:
    history_path = tmp_path / "bench_history.json"
    history_path.write_text("{}", encoding="utf-8")

    records = load_history(history_path)

    assert records == []

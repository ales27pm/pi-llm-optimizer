from __future__ import annotations

import json
import logging
import builtins
from pathlib import Path

import pytest

from dataset.qf_corpus_blueprint.scripts.dataset_card import (
    build_dataset_card,
    compute_dialect_distribution,
    compute_file_sha256,
    compute_register_balance,
    compute_register_distribution,
    main as build_card_cli,
    ExitCode,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sample_records() -> list[dict]:
    return [
        {
            "record_id": "r1",
            "register": "Casual",
            "dialect_tag_list": ["A", "B"],
            "source_corpus_id": "demo_a",
        },
        {
            "record_id": "r2",
            "register": "Formal",
            "dialect_tag_list": ["B"],
            "source_corpus_id": "demo_b",
        },
        {
            "record_id": "r3",
            "register": "Casual",
            "dialect_tag_list": [],
            "source_corpus_id": "demo_a",
        },
    ]


def test_compute_register_distribution_requires_multiple_categories():
    with pytest.raises(ValueError):
        compute_register_distribution([{"register": "Only"}])


def test_build_dataset_card_produces_expected_payload() -> None:
    records = _sample_records()
    card = build_dataset_card(
        records,
        split_name="train",
        license_text="CC-BY-4.0",
        schema_version="1.2.3",
        creation_date="2024-05-01",
        dataset_sha="abc123",
        processing_steps=["normalize", "balance"],
        tool_versions={"normalizer": "0.3.0"},
    )

    assert card["record_count"] == 3
    assert pytest.approx(card["register_distribution"]["Casual"], rel=1e-6) == 2 / 3
    assert "dialect_distribution" in card
    balance = card["register_balance_check"]
    assert balance["max_share"] >= balance["min_share"]
    assert card["provenance"]["aggregated_from"] == ["demo_a", "demo_b"]
    assert card["provenance"]["tool_versions"]["normalizer"] == "0.3.0"


def test_distribution_helpers_include_tags() -> None:
    records = _sample_records()
    register = compute_register_distribution(records)
    dialect = compute_dialect_distribution(records)
    balance = compute_register_balance(register)

    assert set(register) == {"Casual", "Formal"}
    assert pytest.approx(dialect["B"], rel=1e-6) == 2 / 3
    assert balance["std_dev"] >= 0


def test_compute_file_sha256(tmp_path: Path) -> None:
    data = tmp_path / "data.txt"
    data.write_text("abc", encoding="utf-8")
    assert compute_file_sha256(data) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_cli_writes_card(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    card_path = tmp_path / "card.json"
    records = _sample_records() + [
        {
            "record_id": "r4",
            "register": "Formal",
            "dialect_tag_list": ["C"],
            "source_corpus_id": "demo_c",
        },
        {
            "record_id": "r5",
            "register": "Formal",
            "dialect_tag_list": ["D"],
            "source_corpus_id": "demo_d",
        },
    ]
    _write_jsonl(data_path, records)

    exit_code = build_card_cli(
        [
            "--data",
            str(data_path),
            "--output",
            str(card_path),
            "--split-name",
            "train",
            "--license",
            "CC-BY-4.0",
            "--schema-version",
            "1.0.0",
            "--creation-date",
            "2024-05-01",
            "--processing-step",
            "normalize",
            "--tool-version",
            "normalizer=0.3.0",
        ]
    )

    assert exit_code == ExitCode.SUCCESS
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    assert payload["split_name"] == "train"
    assert payload["provenance"]["processing_steps"] == ["normalize"]


def test_cli_validate_requires_schema(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")

    data_path = tmp_path / "dataset.jsonl"
    card_path = tmp_path / "card.json"
    records = _sample_records() + [
        {
            "record_id": "r4",
            "register": "Formal",
            "dialect_tag_list": ["C"],
            "source_corpus_id": "demo_c",
        },
        {
            "record_id": "r5",
            "register": "Formal",
            "dialect_tag_list": ["D"],
            "source_corpus_id": "demo_d",
        },
    ]
    _write_jsonl(data_path, records)

    exit_code = build_card_cli(
        [
            "--data",
            str(data_path),
            "--output",
            str(card_path),
            "--split-name",
            "analysis",
            "--license",
            "CC-BY-4.0",
            "--creation-date",
            "2024-05-01",
            "--validate",
        ]
    )

    assert exit_code == ExitCode.SUCCESS
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    assert payload["split_name"] == "analysis"


def test_cli_validate_reports_schema_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("jsonschema")

    data_path = tmp_path / "dataset.jsonl"
    card_path = tmp_path / "card.json"
    _write_jsonl(data_path, _sample_records())

    def _invalid_card(*_args, **_kwargs) -> dict:
        return {"split_name": "train"}  # Missing required fields like record_count

    monkeypatch.setattr(
        "dataset.qf_corpus_blueprint.scripts.dataset_card.build_dataset_card",
        _invalid_card,
    )

    caplog.set_level(logging.ERROR)
    exit_code = build_card_cli(
        [
            "--data",
            str(data_path),
            "--output",
            str(card_path),
            "--split-name",
            "analysis",
            "--license",
            "CC-BY-4.0",
            "--creation-date",
            "2024-05-01",
            "--validate",
        ]
    )

    assert exit_code == ExitCode.SCHEMA_VALIDATION_FAILED
    assert "validation error" in caplog.text.lower()


def test_cli_validate_missing_jsonschema_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_path = tmp_path / "dataset.jsonl"
    card_path = tmp_path / "card.json"
    _write_jsonl(data_path, _sample_records())

    real_import = builtins.__import__

    def _blocking_import(name: str, *args, **kwargs):
        if name == "jsonschema":  # pragma: no cover - executed in test
            raise ImportError("No module named 'jsonschema'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    caplog.set_level(logging.ERROR)
    exit_code = build_card_cli(
        [
            "--data",
            str(data_path),
            "--output",
            str(card_path),
            "--split-name",
            "analysis",
            "--license",
            "CC-BY-4.0",
            "--creation-date",
            "2024-05-01",
            "--validate",
        ]
    )

    assert exit_code == ExitCode.MISSING_VALIDATION_DEPENDENCY
    assert "jsonschema is required" in caplog.text

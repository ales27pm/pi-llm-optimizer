from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.qf_corpus_blueprint.scripts.dataset_card import (
    build_dataset_card,
    compute_dialect_distribution,
    compute_file_sha256,
    compute_register_balance,
    compute_register_distribution,
    main as build_card_cli,
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


def test_build_dataset_card_produces_expected_payload(tmp_path: Path) -> None:
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
    _write_jsonl(data_path, _sample_records())

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

    assert exit_code == 0
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    assert payload["split_name"] == "train"
    assert payload["provenance"]["processing_steps"] == ["normalize"]

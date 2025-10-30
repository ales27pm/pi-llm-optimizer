import importlib
from pathlib import Path

import pytest

from desktop_distill import export_gguf


def test_determine_output_filename_uses_directory_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "My Student Model"
    model_dir.mkdir()

    filename = export_gguf._determine_output_filename(model_dir, "q4_k_m")

    assert filename == "My_Student_Model-q4_k_m.gguf"


def test_iter_tokenizer_assets_includes_known_files(tmp_path: Path) -> None:
    expected_files = {
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer_config.json",
    }
    for name in expected_files:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "vocab.txt").write_text("test", encoding="utf-8")

    assets = list(export_gguf._iter_tokenizer_assets(tmp_path))
    asset_names = {asset.name for asset in assets}

    for name in expected_files:
        assert name in asset_names
    assert any(asset.is_dir() for asset in assets)


def test_resolve_model_path_prefers_local_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with export_gguf._resolve_model_path(str(model_dir), revision=None, token=None) as resolved:
        assert resolved == model_dir


def test_get_snapshot_download_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    with pytest.raises(ImportError):
        export_gguf._get_snapshot_download()

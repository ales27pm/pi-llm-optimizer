import importlib
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_GGUF_MODULE = PROJECT_ROOT / "desktop_distill" / "export_gguf.py"


def _load_export_gguf():
    spec = importlib.util.spec_from_file_location(
        "desktop_distill.export_gguf", EXPORT_GGUF_MODULE
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("Unable to locate desktop_distill/export_gguf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("desktop_distill.export_gguf", module)
    spec.loader.exec_module(module)
    return module


export_gguf = _load_export_gguf()


def test_determine_output_filename_uses_directory_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "My Student Model"
    model_dir.mkdir()

    filename = export_gguf._determine_output_filename(str(model_dir), model_dir, "q4_k_m")

    assert filename == "My_Student_Model-q4_k_m.gguf"


def test_determine_output_filename_remote_identifier(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    filename = export_gguf._determine_output_filename("org/my-model", model_dir, "q4_k_m")

    assert filename == "org_my-model-q4_k_m.gguf"


def test_determine_output_filename_remote_identifiers_unique(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    first = export_gguf._determine_output_filename("org/model-one", model_dir, "q4_k_m")
    second = export_gguf._determine_output_filename("another/model-one", model_dir, "q4_k_m")

    assert first != second


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


def test_iter_tokenizer_assets_missing_files(tmp_path: Path) -> None:
    assets = list(export_gguf._iter_tokenizer_assets(tmp_path))

    assert assets == []


def test_resolve_model_path_prefers_local_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with export_gguf._resolve_model_path(str(model_dir), revision=None, token=None) as resolved:
        assert resolved == model_dir


def test_resolve_model_path_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        tqdm_cls = kwargs.get("tqdm_class")
        if tqdm_cls is not None:
            bar = tqdm_cls(total=100, desc="weights")
            bar.update(100)
            bar.close()
        return str(local_dir)

    monkeypatch.setattr(export_gguf, "_get_snapshot_download", lambda: fake_snapshot_download)

    with export_gguf._resolve_model_path("org/my-model", revision="main", token="abc") as resolved:
        assert resolved.exists()
        assert resolved.name == "model"

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == "org/my-model"
    assert call["revision"] == "main"
    assert call["token"] == "abc"
    assert call["local_dir_use_symlinks"] is False
    tqdm_cls = call.get("tqdm_class")
    if tqdm_cls is None:
        pytest.skip("tqdm is unavailable; progress logging disabled")
    assert Path(call["local_dir"]).name == "model"


def test_resolve_model_path_remote_logs_progress(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    emitted: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        tqdm_cls = kwargs.get("tqdm_class")
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        if tqdm_cls is not None:
            bar = tqdm_cls(total=200, desc="weights")
            bar.update(100)
            bar.update(100)
            bar.close()
        return str(local_dir)

    monkeypatch.setattr(export_gguf, "_get_snapshot_download", lambda: fake_snapshot_download)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=export_gguf.logger.name):
        with export_gguf._resolve_model_path("org/my-model", revision=None, token=None) as resolved:
            assert resolved.exists()

    for record in caplog.records:
        if record.name == export_gguf.logger.name:
            try:
                emitted.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                continue

    if not emitted:
        pytest.skip("tqdm is unavailable; no progress logs emitted")

    events = {entry.get("event") for entry in emitted}
    assert {"download_start", "download_progress", "download_complete"}.issubset(events)


def test_get_snapshot_download_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    with pytest.raises(ImportError):
        export_gguf._get_snapshot_download()


def test_run_cmd_logs_and_returns(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    completed = subprocess.CompletedProcess(["echo"], 0, stdout="hello\n", stderr="warning\n")

    monkeypatch.setattr(export_gguf.subprocess, "run", lambda *args, **kwargs: completed)

    caplog.set_level(logging.DEBUG, logger=export_gguf.logger.name)

    result = export_gguf.run_cmd(["echo"])

    assert result is completed
    assert "Running command: echo" in caplog.text
    assert "echo stdout" in caplog.text
    assert "echo stderr" in caplog.text


def test_run_cmd_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(["false"], 1, stdout="", stderr="boom")

    monkeypatch.setattr(export_gguf.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError) as exc_info:
        export_gguf.run_cmd(["false"])

    assert "failed with exit code 1" in str(exc_info.value)

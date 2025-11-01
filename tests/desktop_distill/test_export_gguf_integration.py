"""Integration coverage for remote HuggingFace resolution failures."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_GGUF_MODULE = PROJECT_ROOT / "desktop_distill" / "export_gguf.py"
MODULE_NAME = "desktop_distill.export_gguf"


def _load_export_gguf():
    module = sys.modules.get(MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(MODULE_NAME, EXPORT_GGUF_MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError("Unable to load desktop_distill.export_gguf")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


export_gguf = _load_export_gguf()


@pytest.mark.integration
def test_remote_resolution_missing_repository(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def fake_snapshot_download(**_kwargs):
        raise RepositoryNotFoundError("404 Client Error: Not Found for url")

    monkeypatch.setattr(export_gguf, "_get_snapshot_download", lambda: fake_snapshot_download)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=export_gguf.logger.name):
        with pytest.raises(export_gguf.ModelResolutionError) as exc_info:
            with export_gguf._resolve_model_path("org/missing-model", revision=None, token=None):
                pass

    message = str(exc_info.value)
    assert "Repository was not found or access is denied" in message
    assert "Remediation steps" in message
    assert "access token" in message

    error_logs = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == export_gguf.logger.name and record.levelno >= logging.ERROR
    ]
    assert error_logs, "expected structured error logs"
    payload = error_logs[-1]
    assert payload["event"] == "model_resolution"
    assert payload["status"] == "error"
    assert payload["repo"] == "org/missing-model"


@pytest.mark.integration
def test_remote_resolution_bad_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(**_kwargs):
        raise RevisionNotFoundError("Revision not found")

    monkeypatch.setattr(export_gguf, "_get_snapshot_download", lambda: fake_snapshot_download)

    with pytest.raises(export_gguf.ModelResolutionError) as exc_info:
        with export_gguf._resolve_model_path("org/model", revision="draft", token=None):
            pass

    message = str(exc_info.value)
    assert "Revision 'draft' does not exist" in message
    assert "Use --revision" in message

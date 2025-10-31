from __future__ import annotations

from pathlib import Path

import pytest

from automation.session_sync import SessionSync, SessionSyncConfig, SessionSyncError


def test_session_sync_wraps_manifest_errors(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    missing_manifest = tmp_path / "missing_manifest.json"
    config = SessionSyncConfig(
        repo_root=tmp_path,
        manifest_path=missing_manifest,
        check=True,
        skip_formatting=True,
        skip_agent_sync=False,
        skip_cleanup=True,
        run_npm_lint=False,
        run_pytest=False,
        enforce_manifest=None,
    )

    workflow = SessionSync(config)
    with pytest.raises(SessionSyncError) as excinfo:
        workflow.run()

    assert str(missing_manifest) in str(excinfo.value)
    assert "Failed to load agents manifest" in str(excinfo.value)

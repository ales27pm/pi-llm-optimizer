from __future__ import annotations

from pathlib import Path

import pytest

from automation.session_sync import SessionSync, SessionSyncConfig, SessionSyncError


def _seed_minimal_roadmap(repo_root: Path) -> None:
    automation_dir = repo_root / "automation"
    automation_dir.mkdir()
    (repo_root / "ROADMAP.md").write_text("# Placeholder\n", encoding="utf-8")
    automation_dir.joinpath("roadmap.yaml").write_text(
        "\n".join(
            [
                "metadata:",
                "  title: Test Roadmap",
                "  intro: Intro line",
                "sections:",
                "  - heading: Section One",
                "    items:",
                "      - title: Item",
                "        description: Something",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_session_sync_wraps_manifest_errors(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    _seed_minimal_roadmap(tmp_path)
    missing_manifest = tmp_path / "missing_manifest.json"
    config = SessionSyncConfig(
        repo_root=tmp_path,
        manifest_path=missing_manifest,
        check=True,
        skip_formatting=True,
        skip_roadmap=True,
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

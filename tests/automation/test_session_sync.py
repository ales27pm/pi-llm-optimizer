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


def test_session_sync_runs_roadmap_sync(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    _seed_minimal_roadmap(tmp_path)
    manifest = tmp_path / "automation" / "agents_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    config = SessionSyncConfig(
        repo_root=tmp_path,
        manifest_path=manifest,
        check=False,
        skip_formatting=True,
        skip_roadmap=False,
        skip_agent_sync=True,
        skip_cleanup=True,
        run_npm_lint=False,
        run_pytest=False,
        enforce_manifest=None,
    )

    summary = SessionSync(config).run()

    assert summary.roadmap is not None
    assert summary.roadmap.sections_rendered == 1
    assert summary.roadmap.items_rendered == 1
    assert summary.roadmap.tasks_rendered == 0
    assert summary.roadmap.changed is True
    expected = (
        "# Test Roadmap\n\n"
        "Intro line\n\n"
        "## Section One\n\n"
        "- **Item** – Something\n"
    )
    assert (tmp_path / "ROADMAP.md").read_text(encoding="utf-8") == expected

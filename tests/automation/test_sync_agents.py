from __future__ import annotations

import json
from pathlib import Path


from automation import sync_agents


def _write_manifest(repo_root: Path, data: dict) -> Path:
    manifest_path = repo_root / "agents_manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    return manifest_path


def _normalise_text(text: str) -> str:
    return text.rstrip() + "\n"


def test_apply_manifest_writes_content_from_source(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    target_dir = tmp_path / "automation_area"
    source_dir = tmp_path / "sources"
    target_dir.mkdir()
    source_dir.mkdir()

    source_file = source_dir / "automation_area.md"
    source_content = "# Protocol\nversion: 1.0.0\n"
    source_file.write_text(source_content, encoding="utf-8")

    manifest_data = {
        "version": 2,
        "agents": [
            {
                "path": "automation_area",
                "source": "sources/automation_area.md",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    manifest = sync_agents.load_manifest(manifest_path, tmp_path)
    result = sync_agents.apply_manifest(
        manifest,
        repo_root=tmp_path,
        write=True,
        enforce_manifest=True,
    )

    target_file = target_dir / sync_agents.AGENT_FILENAME
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == _normalise_text(source_content)
    assert result.reports[0].status is sync_agents.AgentSyncStatus.CREATED
    assert not result.stray_agents


def test_apply_manifest_check_mode_detects_outdated_agents(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    target_dir = tmp_path / "training"
    target_dir.mkdir()
    agent_file = target_dir / sync_agents.AGENT_FILENAME
    agent_file.write_text("old content\n", encoding="utf-8")

    manifest_data = {
        "version": 2,
        "agents": [
            {
                "path": "training",
                "content": "# Fresh\nversion: 1.0.0\n",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    manifest = sync_agents.load_manifest(manifest_path, tmp_path)
    result = sync_agents.apply_manifest(
        manifest,
        repo_root=tmp_path,
        write=False,
        enforce_manifest=True,
    )

    assert result.reports[0].status is sync_agents.AgentSyncStatus.WOULD_UPDATE
    # Ensure the file was not modified in check mode
    assert agent_file.read_text(encoding="utf-8") == "old content\n"


def test_apply_manifest_flags_stray_agents(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    tracked_dir = tmp_path / "tracked"
    stray_dir = tmp_path / "stray"
    tracked_dir.mkdir()
    stray_dir.mkdir()

    manifest_data = {
        "version": 2,
        "settings": {"enforceTrackedAgents": True},
        "agents": [
            {
                "path": "tracked",
                "content": "# Managed\nversion: 1.0.0\n",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    manifest = sync_agents.load_manifest(manifest_path, tmp_path)
    result = sync_agents.apply_manifest(
        manifest,
        repo_root=tmp_path,
        write=True,
        enforce_manifest=True,
    )

    stray_file = stray_dir / sync_agents.AGENT_FILENAME
    stray_file.write_text("# stray\n", encoding="utf-8")

    # Re-run to pick up the stray file
    result = sync_agents.apply_manifest(
        manifest,
        repo_root=tmp_path,
        write=True,
        enforce_manifest=True,
    )

    assert stray_file in result.stray_agents

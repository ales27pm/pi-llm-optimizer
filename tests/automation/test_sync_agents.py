from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_apply_manifest_raises_error_for_missing_source(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    target_dir = tmp_path / "automation_area"
    source_dir = tmp_path / "sources"
    target_dir.mkdir()
    source_dir.mkdir()

    manifest_data = {
        "version": 2,
        "agents": [
            {
                "path": "automation_area",
                "source": "sources/missing.md",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    with pytest.raises(sync_agents.AgentEntryError):
        sync_agents.load_manifest(manifest_path, tmp_path)


def test_apply_manifest_raises_error_for_invalid_source(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    target_dir = tmp_path / "automation_area"
    source_dir = tmp_path / "sources"
    target_dir.mkdir()
    source_dir.mkdir()

    invalid_source = source_dir / "invalid"
    invalid_source.mkdir()

    manifest_data = {
        "version": 2,
        "agents": [
            {
                "path": "automation_area",
                "source": "sources/invalid",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    with pytest.raises(sync_agents.AgentEntryError):
        sync_agents.load_manifest(manifest_path, tmp_path)


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


def test_apply_manifest_respects_excludes(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    stray_root = tmp_path / "stray"
    stray_root.mkdir()

    manifest_data = {
        "version": 2,
        "settings": {
            "enforceTrackedAgents": True,
            "excludes": ["stray/excluded_*/AGENTS.md"],
        },
        "agents": [
            {
                "path": "managed",
                "content": "# Managed\nversion: 1.0.0\n",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    manifest = sync_agents.load_manifest(manifest_path, tmp_path)
    sync_agents.apply_manifest(
        manifest,
        repo_root=tmp_path,
        write=True,
        enforce_manifest=True,
    )

    excluded_dir = stray_root / "excluded_area"
    excluded_dir.mkdir()
    excluded = excluded_dir / sync_agents.AGENT_FILENAME
    excluded.write_text("# excluded\n", encoding="utf-8")

    included_dir = stray_root / "included_area"
    included_dir.mkdir()
    included = included_dir / sync_agents.AGENT_FILENAME
    included.write_text("# tracked stray\n", encoding="utf-8")

    result = sync_agents.apply_manifest(
        manifest,
        repo_root=tmp_path,
        write=False,
        enforce_manifest=True,
    )

    assert included in result.stray_agents
    assert excluded not in result.stray_agents


def test_apply_manifest_supports_manifest_v1(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()

    manifest_data = {
        "version": 1,
        "agents": [
            {
                "path": "legacy",
                "content": "# Legacy\nversion: 1.0.0\n",
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

    target = legacy_dir / sync_agents.AGENT_FILENAME
    assert target.exists()
    assert target.read_text(encoding="utf-8") == _normalise_text("# Legacy\nversion: 1.0.0\n")
    assert result.reports[0].status is sync_agents.AgentSyncStatus.CREATED


def test_manifest_v1_requires_content(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "legacy").mkdir()

    manifest_data = {
        "version": 1,
        "agents": [
            {
                "path": "legacy",
            }
        ],
    }
    manifest_path = _write_manifest(tmp_path, manifest_data)

    with pytest.raises(sync_agents.AgentEntryError):
        sync_agents.load_manifest(manifest_path, tmp_path)

from __future__ import annotations

from pathlib import Path

import pytest

from automation.roadmap_sync import RoadmapSynchroniser, RoadmapSyncError


def _write_sample_yaml(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "metadata:",
                "  title: Demo Roadmap",
                "  intro: Welcome to the roadmap",
                "sections:",
                "  - heading: Section Alpha",
                "    items:",
                "      - title: Feature A",
                "        description: Covers scenario A",
                "        tasks:",
                "          - summary: Ship task A1",
                "            status: done",
                "      - title: Feature B",
                "        description: Covers scenario B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_roadmap_sync_generates_markdown(tmp_path: Path) -> None:
    repo_root = tmp_path
    automation_dir = repo_root / "automation"
    automation_dir.mkdir()
    source = automation_dir / "roadmap.yaml"
    target = repo_root / "ROADMAP.md"
    _write_sample_yaml(source)

    synchroniser = RoadmapSynchroniser(
        repo_root=repo_root,
        source_path=source,
        target_path=target,
        check=False,
    )
    summary = synchroniser.run()

    assert summary.sections_rendered == 1
    assert summary.items_rendered == 2
    assert summary.tasks_rendered == 1
    assert summary.changed is True
    expected = (
        "# Demo Roadmap\n\n"
        "Welcome to the roadmap\n\n"
        "## Section Alpha\n\n"
        "- **Feature A** – Covers scenario A\n"
        "  - [x] Ship task A1\n"
        "- **Feature B** – Covers scenario B\n"
    )
    assert target.read_text(encoding="utf-8") == expected


def test_roadmap_sync_check_detects_drift(tmp_path: Path) -> None:
    repo_root = tmp_path
    automation_dir = repo_root / "automation"
    automation_dir.mkdir()
    source = automation_dir / "roadmap.yaml"
    target = repo_root / "ROADMAP.md"
    _write_sample_yaml(source)

    synchroniser = RoadmapSynchroniser(
        repo_root=repo_root,
        source_path=source,
        target_path=target,
        check=False,
    )
    synchroniser.run()
    target.write_text("stale\n", encoding="utf-8")

    checker = RoadmapSynchroniser(
        repo_root=repo_root,
        source_path=source,
        target_path=target,
        check=True,
    )
    with pytest.raises(RoadmapSyncError):
        checker.run()

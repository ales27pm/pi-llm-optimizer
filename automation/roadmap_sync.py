#!/usr/bin/env python3
"""Synchronise the canonical ROADMAP.md from structured data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import yaml

LOGGER = logging.getLogger("roadmap_sync")


class RoadmapSyncError(RuntimeError):
    """Raised when the roadmap synchronisation workflow cannot complete."""


@dataclass(frozen=True)
class RoadmapTask:
    """A single checkbox entry within a roadmap item."""

    summary: str
    status: str

    def marker(self) -> str:
        if self.status == "done":
            return "x"
        if self.status == "todo":
            return " "
        raise RoadmapSyncError(f"Unsupported task status: {self.status}")


@dataclass(frozen=True)
class RoadmapItem:
    """A roadmap bullet containing a title, description, and optional tasks."""

    title: str
    description: str
    tasks: Sequence[RoadmapTask]


@dataclass(frozen=True)
class RoadmapSection:
    """A roadmap section grouping related items or free-form bullets."""

    heading: str
    items: Sequence[RoadmapItem]
    bullets: Sequence[str]


@dataclass(frozen=True)
class RoadmapSyncSummary:
    """Aggregate details describing a synchronisation run."""

    sections_rendered: int
    items_rendered: int
    tasks_rendered: int
    changed: bool


class RoadmapSynchroniser:
    """Orchestrate regeneration of the Markdown roadmap."""

    def __init__(
        self,
        *,
        repo_root: Path,
        source_path: Path,
        target_path: Path,
        check: bool,
    ) -> None:
        self.repo_root = repo_root
        self.source_path = source_path
        self.target_path = target_path
        self.check = check

    # Public API -----------------------------------------------------
    def run(self) -> RoadmapSyncSummary:
        LOGGER.info("Regenerating roadmap from %s", self.source_path)
        raw_config = self._load_config()
        metadata = raw_config.get("metadata")
        if not isinstance(metadata, dict):
            raise RoadmapSyncError("Roadmap metadata must be a mapping with title and intro fields.")

        title = self._require_str(metadata, "title")
        intro = self._normalise_lines(metadata.get("intro"))

        sections_data = raw_config.get("sections")
        if not isinstance(sections_data, list) or not sections_data:
            raise RoadmapSyncError("Roadmap must define at least one section.")

        sections: List[RoadmapSection] = []
        item_count = 0
        task_count = 0
        for entry in sections_data:
            if not isinstance(entry, dict):
                raise RoadmapSyncError("Each section definition must be a mapping.")
            heading = self._require_str(entry, "heading")
            items = self._parse_items(entry.get("items"))
            bullets = self._parse_bullets(entry.get("bullets"))
            if not items and not bullets:
                raise RoadmapSyncError(f"Section '{heading}' must define items or bullets.")
            sections.append(RoadmapSection(heading=heading, items=items, bullets=bullets))
            item_count += len(items)
            task_count += sum(len(item.tasks) for item in items)

        rendered = self._render_document(title, intro, sections)
        changed = self._write(rendered)
        return RoadmapSyncSummary(
            sections_rendered=len(sections),
            items_rendered=item_count,
            tasks_rendered=task_count,
            changed=changed,
        )

    # Internal helpers ----------------------------------------------
    def _load_config(self) -> dict:
        try:
            text = self.source_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise RoadmapSyncError(f"Roadmap source not found: {self.source_path}") from exc
        try:
            loaded = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise RoadmapSyncError(f"Failed to parse roadmap YAML: {exc}") from exc
        if not isinstance(loaded, dict):
            raise RoadmapSyncError("Roadmap configuration must be a mapping at the top level.")
        return loaded

    @staticmethod
    def _require_str(container: dict, key: str) -> str:
        value = container.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RoadmapSyncError(f"Field '{key}' must be a non-empty string.")
        return value.strip()

    def _parse_items(self, raw_items: object) -> List[RoadmapItem]:
        if raw_items is None:
            return []
        if not isinstance(raw_items, list):
            raise RoadmapSyncError("Section items must be provided as a list.")
        items: List[RoadmapItem] = []
        for idx, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                raise RoadmapSyncError("Each item must be a mapping with title and description.")
            title = self._require_str(raw_item, "title")
            description = self._require_str(raw_item, "description")
            tasks = self._parse_tasks(raw_item.get("tasks"), title, idx)
            items.append(RoadmapItem(title=title, description=description, tasks=tasks))
        return items

    def _parse_tasks(self, raw_tasks: object, title: str, index: int) -> List[RoadmapTask]:
        if raw_tasks is None:
            return []
        if not isinstance(raw_tasks, list):
            raise RoadmapSyncError(f"Tasks for '{title}' must be a list of mappings.")
        tasks: List[RoadmapTask] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                raise RoadmapSyncError(f"Task entries for '{title}' must be mappings.")
            summary = self._require_str(raw_task, "summary")
            status_raw = raw_task.get("status", "todo")
            if not isinstance(status_raw, str):
                raise RoadmapSyncError(f"Task status for '{title}' must be a string.")
            status = status_raw.strip().lower()
            if status not in {"todo", "done"}:
                raise RoadmapSyncError(
                    f"Task status for '{title}' must be 'todo' or 'done', got '{status_raw}'."
                )
            tasks.append(RoadmapTask(summary=summary, status=status))
        return tasks

    def _parse_bullets(self, raw_bullets: object) -> List[str]:
        if raw_bullets is None:
            return []
        if not isinstance(raw_bullets, list):
            raise RoadmapSyncError("Bullets must be expressed as a list of strings.")
        bullets: List[str] = []
        for bullet in raw_bullets:
            if not isinstance(bullet, str) or not bullet.strip():
                raise RoadmapSyncError("Roadmap bullet entries must be non-empty strings.")
            bullets.append(bullet.strip())
        return bullets

    @staticmethod
    def _normalise_lines(value: object) -> Sequence[str]:
        if value is None:
            return []
        if not isinstance(value, str):
            raise RoadmapSyncError("Intro text must be a string.")
        lines = [line.rstrip() for line in value.strip().splitlines()]
        return [line for line in lines if line]

    @staticmethod
    def _render_document(title: str, intro_lines: Sequence[str], sections: Sequence[RoadmapSection]) -> str:
        lines: List[str] = [f"# {title}", ""]
        for intro_line in intro_lines:
            lines.append(intro_line)
        if intro_lines:
            lines.append("")
        for idx, section in enumerate(sections):
            lines.append(f"## {section.heading}")
            lines.append("")
            for item in section.items:
                lines.append(f"- **{item.title}** – {item.description}")
                for task in item.tasks:
                    lines.append(f"  - [{task.marker()}] {task.summary}")
            for bullet in section.bullets:
                lines.append(f"- {bullet}")
            if idx != len(sections) - 1:
                lines.append("")
        rendered = "\n".join(lines).rstrip() + "\n"
        return rendered

    def _write(self, content: str) -> bool:
        target = self.target_path
        try:
            existing = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        if self.check:
            if existing is None:
                raise RoadmapSyncError("ROADMAP.md is missing; run the sync workflow to regenerate it.")
            if existing != content:
                raise RoadmapSyncError(
                    "ROADMAP.md is out of date. Run automation/update_and_cleanup.sh to regenerate it."
                )
            return False
        if existing == content:
            LOGGER.info("Roadmap already up to date; no changes written.")
            return False
        target.write_text(content, encoding="utf-8")
        LOGGER.info("Updated %s", self._describe_path(target))
        return True

    def _describe_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)


__all__ = [
    "RoadmapSynchroniser",
    "RoadmapSyncError",
    "RoadmapSyncSummary",
]

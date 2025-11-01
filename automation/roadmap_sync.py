#!/usr/bin/env python3
"""Synchronise the canonical ROADMAP.md from structured data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml
from jinja2 import Environment
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

LOGGER = logging.getLogger("roadmap_sync")


class RoadmapSyncError(RuntimeError):
    """Raised when the roadmap synchronisation workflow cannot complete."""


class RoadmapTask(BaseModel):
    """A single checkbox entry within a roadmap item."""

    summary: str
    status: str = Field("todo")

    class Config:
        extra = "forbid"

    @validator("summary")
    def _ensure_summary(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("summary must be a non-empty string")
        return value.strip()

    @validator("status", pre=True)
    def _normalise_status(cls, value: str | None) -> str:
        if value is None:
            return "todo"
        if not isinstance(value, str):
            raise ValueError("status must be a string")
        normalised = value.strip().lower()
        if normalised not in {"todo", "done"}:
            raise ValueError("status must be 'todo' or 'done'")
        return normalised


class RoadmapItem(BaseModel):
    """A roadmap bullet containing a title, description, and optional tasks."""

    title: str
    description: str
    tasks: list[RoadmapTask] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @validator("title")
    def _ensure_title(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("title must be a non-empty string")
        return value.strip()

    @validator("description")
    def _ensure_description(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("description must be a non-empty string")
        return value.strip()


class RoadmapSection(BaseModel):
    """A roadmap section grouping related items or free-form bullets."""

    heading: str
    items: list[RoadmapItem] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @validator("heading")
    def _ensure_heading(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("heading must be a non-empty string")
        return value.strip()

    @validator("bullets", pre=True)
    def _default_bullets(cls, value: Sequence[str] | None) -> Sequence[str]:
        return [] if value is None else value

    @validator("bullets", each_item=True)
    def _normalise_bullet(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("bullets must contain non-empty strings")
        return value.strip()

    @root_validator
    def _ensure_content(cls, values: dict) -> dict:
        items: list[RoadmapItem] = values.get("items", [])
        bullets: list[str] = values.get("bullets", [])
        if not items and not bullets:
            raise ValueError("section must define at least one item or bullet")
        return values


class RoadmapMetadata(BaseModel):
    """Document metadata used to render the roadmap header."""

    title: str
    intro: str | None = None

    class Config:
        extra = "forbid"

    @validator("title")
    def _ensure_title(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("title must be a non-empty string")
        return value.strip()

    @validator("intro")
    def _normalise_intro(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("intro must be a string")
        stripped = value.strip()
        return stripped or None

    @property
    def intro_lines(self) -> list[str]:
        if not self.intro:
            return []
        lines = [line.rstrip() for line in self.intro.splitlines()]
        return [line for line in lines if line]


class RoadmapConfig(BaseModel):
    """Structured roadmap configuration parsed from YAML."""

    metadata: RoadmapMetadata
    sections: list[RoadmapSection]

    class Config:
        extra = "forbid"

    @validator("sections")
    def _ensure_sections(cls, value: list[RoadmapSection]) -> list[RoadmapSection]:
        if not value:
            raise ValueError("at least one section must be defined")
        return value


@dataclass(frozen=True)
class RoadmapSyncSummary:
    """Aggregate details describing a synchronisation run."""

    sections_rendered: int
    items_rendered: int
    tasks_rendered: int
    changed: bool


_TEMPLATE_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
_ROADMAP_TEMPLATE = _TEMPLATE_ENV.from_string(
    """
# {{ metadata.title }}

{% for line in metadata.intro_lines %}
{{ line }}
{% endfor %}{% if metadata.intro_lines %}

{% endif %}{% for section in sections %}
## {{ section.heading }}

{% for item in section.items %}
- **{{ item.title }}** – {{ item.description }}
{% for task in item.tasks %}
  - [{{ 'x' if task.status == 'done' else ' ' }}] {{ task.summary }}
{% endfor %}
{% endfor %}
{% for bullet in section.bullets %}
- {{ bullet }}
{% endfor %}{% if not loop.last %}

{% endif %}{% endfor %}
""".strip()
)


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
        config = self._load_config()
        rendered = self._render_document(config)
        changed = self._write(rendered)
        return RoadmapSyncSummary(
            sections_rendered=len(config.sections),
            items_rendered=sum(len(section.items) for section in config.sections),
            tasks_rendered=sum(len(item.tasks) for section in config.sections for item in section.items),
            changed=changed,
        )

    # Internal helpers ----------------------------------------------
    def _load_config(self) -> RoadmapConfig:
        try:
            text = self.source_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise RoadmapSyncError(f"Roadmap source not found: {self.source_path}") from exc
        try:
            loaded = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise RoadmapSyncError(f"Failed to parse roadmap YAML: {exc}") from exc
        try:
            return RoadmapConfig.parse_obj(loaded)
        except ValidationError as exc:
            raise RoadmapSyncError(self._format_validation_error(exc)) from exc

    @staticmethod
    def _format_validation_error(exc: ValidationError) -> str:
        details = ", ".join(
            f"{RoadmapSynchroniser._format_location(err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        return f"Invalid roadmap configuration: {details}"

    @staticmethod
    def _format_location(location: Sequence[object]) -> str:
        return ".".join(str(part) for part in location if part is not None)

    @staticmethod
    def _render_document(config: RoadmapConfig) -> str:
        rendered = _ROADMAP_TEMPLATE.render(metadata=config.metadata, sections=config.sections).rstrip()
        return f"{rendered}\n"

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

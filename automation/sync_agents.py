#!/usr/bin/env python3
"""Synchronise nested ``AGENTS.md`` files across the repository.

This module consumes ``automation/agents_manifest.json`` to generate scoped agent
protocol files. Version 2 of the manifest introduces metadata-rich entries with
shared content sources, enforcement rules, and dry-run support. The CLI is
idempotent and safe for CI usage while also powering the higher-level session
sync workflow.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Mapping, MutableMapping
from typing import List, Optional, Sequence

MANIFEST_NAME = "agents_manifest.json"
AGENT_FILENAME = "AGENTS.md"


class AgentEntryError(RuntimeError):
    """Raised when the manifest cannot be interpreted safely."""


class AgentSyncStatus(Enum):
    """Possible outcomes when comparing desired and current agent files."""

    UNCHANGED = "unchanged"
    UPDATED = "updated"
    CREATED = "created"
    WOULD_UPDATE = "would_update"
    WOULD_CREATE = "would_create"


@dataclass(frozen=True)
class ManifestSettings:
    """Global configuration for agent synchronisation."""

    enforce_tracked_agents: bool = False
    excludes: Sequence[str] = ()


@dataclass(frozen=True)
class AgentSpec:
    """Concrete instruction for rendering a scoped agent protocol."""

    relative_dir: Path
    target_file: Path
    source_file: Optional[Path]
    content: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class Manifest:
    """Structured representation of the manifest file."""

    version: int
    agents: Sequence[AgentSpec]
    settings: ManifestSettings
    manifest_path: Path


@dataclass(frozen=True)
class SyncReport:
    """Outcome for a single agent file after applying the manifest."""

    spec: AgentSpec
    status: AgentSyncStatus


@dataclass(frozen=True)
class SyncResult:
    """Aggregate outcome for a synchronisation run."""

    reports: Sequence[SyncReport]
    stray_agents: Sequence[Path]

    @property
    def updated(self) -> Sequence[SyncReport]:
        return [
            report
            for report in self.reports
            if report.status in {AgentSyncStatus.CREATED, AgentSyncStatus.UPDATED}
        ]

    @property
    def pending_updates(self) -> Sequence[SyncReport]:
        return [
            report
            for report in self.reports
            if report.status in {AgentSyncStatus.WOULD_CREATE, AgentSyncStatus.WOULD_UPDATE}
        ]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[sync_agents] %(levelname)s %(message)s")


def discover_repo_root(start_path: Path) -> Path:
    current = start_path.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise AgentEntryError("Unable to locate repository root (missing .git directory).")


def _normalise_content(raw: str) -> str:
    return raw.rstrip() + "\n"


def _validate_directory(path: Path) -> None:
    if not path.exists():
        raise AgentEntryError(f"Manifest path '{path}' does not exist in the repository.")
    if not path.is_dir():
        raise AgentEntryError(f"Manifest path '{path}' is not a directory.")


def _load_settings(data: Mapping[str, object]) -> ManifestSettings:
    enforce = bool(data.get("enforceTrackedAgents", False))
    excludes_raw = data.get("excludes", [])
    if excludes_raw is None:
        excludes: Sequence[str] = ()
    elif isinstance(excludes_raw, list) and all(isinstance(item, str) for item in excludes_raw):
        excludes = tuple(excludes_raw)
    else:
        raise AgentEntryError("Manifest 'settings.excludes' must be a list of strings.")
    return ManifestSettings(enforce_tracked_agents=enforce, excludes=excludes)


def _read_source_file(repo_root: Path, source_path: str, manifest_path: Path) -> Path:
    if not source_path:
        raise AgentEntryError("Manifest entry 'source' must be a non-empty string when provided.")
    candidate = (repo_root / source_path).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise AgentEntryError(
            f"Source file '{source_path}' for manifest {manifest_path} escapes the repository root."
        ) from exc
    if not candidate.exists():
        raise AgentEntryError(f"Source file '{source_path}' listed in manifest does not exist.")
    if not candidate.is_file():
        raise AgentEntryError(f"Source path '{source_path}' in manifest must reference a file.")
    return candidate


def _parse_agent_entry(
    raw_entry: MutableMapping[str, object],
    repo_root: Path,
    manifest_path: Path,
    manifest_version: int,
) -> AgentSpec:
    path_value = raw_entry.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise AgentEntryError("Manifest entry missing non-empty 'path'.")
    relative_dir = Path(path_value)
    target_dir = (repo_root / relative_dir).resolve()
    try:
        target_dir.relative_to(repo_root)
    except ValueError as exc:
        raise AgentEntryError(
            f"Manifest path '{path_value}' points outside of the repository root."
        ) from exc
    _validate_directory(target_dir)
    target_file = target_dir / AGENT_FILENAME

    content_value = raw_entry.get("content")
    source_value = raw_entry.get("source")
    if manifest_version >= 2:
        if source_value and content_value:
            raise AgentEntryError(
                f"Manifest entry for '{path_value}' cannot define both 'source' and 'content'."
            )
        if source_value:
            if not isinstance(source_value, str):
                raise AgentEntryError("Manifest entry 'source' must be a string when provided.")
            source_file = _read_source_file(repo_root, source_value, manifest_path)
            desired_content = _normalise_content(source_file.read_text(encoding="utf-8"))
        elif isinstance(content_value, str) and content_value.strip():
            source_file = None
            desired_content = _normalise_content(content_value)
        else:
            raise AgentEntryError(
                f"Manifest entry for '{path_value}' requires either 'source' or populated 'content'."
            )
    else:
        if not isinstance(content_value, str) or not content_value.strip():
            raise AgentEntryError(
                f"Manifest entry for '{path_value}' is missing content for version {manifest_version}."
            )
        source_file = None
        desired_content = _normalise_content(content_value)

    metadata = {
        key: value
        for key, value in raw_entry.items()
        if key not in {"path", "content", "source"}
    }
    return AgentSpec(
        relative_dir=relative_dir,
        target_file=target_file,
        source_file=source_file,
        content=desired_content,
        metadata=metadata,
    )


def load_manifest(manifest_path: Path, repo_root: Path) -> Manifest:
    if not manifest_path.exists():
        raise AgentEntryError(
            f"Manifest file not found: {manifest_path}. Ensure automation/agents_manifest.json exists."
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AgentEntryError(f"Unable to parse {manifest_path}: {exc}.") from exc
    if not isinstance(data, dict):
        raise AgentEntryError("Manifest root must be a JSON object.")

    version = data.get("version")
    if version not in {1, 2}:
        raise AgentEntryError("Unsupported manifest version. Expected version=1 or version=2.")

    agents_raw = data.get("agents")
    if not isinstance(agents_raw, list):
        raise AgentEntryError("Manifest field 'agents' must be a list.")

    settings_raw = data.get("settings", {})
    if not isinstance(settings_raw, dict):
        raise AgentEntryError("Manifest field 'settings' must be an object when provided.")
    settings = _load_settings(settings_raw)

    agents: List[AgentSpec] = []
    for raw_entry in agents_raw:
        if not isinstance(raw_entry, MutableMapping):
            raise AgentEntryError("Each manifest entry must be an object.")
        agents.append(_parse_agent_entry(raw_entry, repo_root, manifest_path, version))

    return Manifest(version=version, agents=tuple(agents), settings=settings, manifest_path=manifest_path)


def _collect_existing_agent_files(repo_root: Path, excludes: Sequence[str]) -> Sequence[Path]:
    tracked: List[Path] = []
    for candidate in repo_root.rglob(AGENT_FILENAME):
        if candidate == repo_root / AGENT_FILENAME:
            continue
        try:
            relative = candidate.relative_to(repo_root)
        except ValueError:  # pragma: no cover - defensive guard
            continue
        if any(fnmatch.fnmatch(str(relative), pattern) for pattern in excludes):
            logging.debug("Ignoring agent file %s due to excludes.", relative)
            continue
        tracked.append(candidate)
    return tracked


def apply_manifest(
    manifest: Manifest,
    *,
    repo_root: Path,
    write: bool,
    enforce_manifest: bool,
) -> SyncResult:
    reports: List[SyncReport] = []
    for spec in manifest.agents:
        target = spec.target_file
        desired_content = spec.content
        if target.exists():
            existing_content = target.read_text(encoding="utf-8")
            if existing_content == desired_content:
                status = AgentSyncStatus.UNCHANGED
                logging.debug("Agent %s already up to date.", target.relative_to(repo_root))
            else:
                if write:
                    logging.info("Updating %s", target.relative_to(repo_root))
                    try:
                        target.write_text(desired_content, encoding="utf-8")
                    except OSError as exc:
                        raise AgentEntryError(
                            f"Failed to write agent file {target.relative_to(repo_root)}: {exc}"
                        ) from exc
                    status = AgentSyncStatus.UPDATED
                else:
                    logging.warning(
                        "Agent %s would be updated (run without --check to apply).",
                        target.relative_to(repo_root),
                    )
                    status = AgentSyncStatus.WOULD_UPDATE
        else:
            if not write:
                logging.warning(
                    "Agent %s would be created (run without --check to apply).",
                    target.relative_to(repo_root),
                )
                status = AgentSyncStatus.WOULD_CREATE
            else:
                logging.info("Creating %s", target.relative_to(repo_root))
                try:
                    target.write_text(desired_content, encoding="utf-8")
                except OSError as exc:
                    raise AgentEntryError(
                        f"Failed to write agent file {target.relative_to(repo_root)}: {exc}"
                    ) from exc
                status = AgentSyncStatus.CREATED
        reports.append(SyncReport(spec=spec, status=status))

    tracked_targets = {spec.target_file.resolve() for spec in manifest.agents}
    excludes = manifest.settings.excludes
    existing_agents = _collect_existing_agent_files(repo_root, excludes)
    stray_agents = [path for path in existing_agents if path.resolve() not in tracked_targets]

    if stray_agents and enforce_manifest:
        stray_list = ", ".join(str(path.relative_to(repo_root)) for path in stray_agents)
        logging.error("Untracked agent protocols detected: %s", stray_list)
    elif stray_agents:
        stray_list = ", ".join(str(path.relative_to(repo_root)) for path in stray_agents)
        logging.warning("Untracked agent protocols detected (ignored): %s", stray_list)

    return SyncResult(reports=tuple(reports), stray_agents=tuple(stray_agents))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronise nested AGENTS.md files based on the automation manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to the agents manifest. Defaults to automation/agents_manifest.json.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root. The script walks up from the manifest to find .git when omitted.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run mode. Exit non-zero if any files are out of sync.",
    )
    parser.add_argument(
        "--enforce-manifest",
        action="store_true",
        help="Fail when agent files exist outside of the manifest (in addition to manifest settings).",
    )
    return parser.parse_args(argv)


def _summarise_reports(result: SyncResult) -> str:
    counts: MutableMapping[AgentSyncStatus, int] = {status: 0 for status in AgentSyncStatus}
    for report in result.reports:
        counts[report.status] += 1
    fragments = [f"{status.value}={counts[status]}" for status in AgentSyncStatus]
    if result.stray_agents:
        fragments.append(f"stray={len(result.stray_agents)}")
    return ", ".join(fragments)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parent / MANIFEST_NAME

    repo_root = args.repo_root
    if repo_root is None:
        repo_root = discover_repo_root(manifest_path.parent)
    else:
        repo_root = repo_root.resolve()

    configure_logging(args.verbose)

    try:
        manifest = load_manifest(manifest_path, repo_root)
        enforce_manifest = args.enforce_manifest or manifest.settings.enforce_tracked_agents
        result = apply_manifest(
            manifest,
            repo_root=repo_root,
            write=not args.check,
            enforce_manifest=enforce_manifest,
        )
    except AgentEntryError as exc:
        logging.error("%s", exc)
        return 1

    summary = _summarise_reports(result)
    if args.check:
        if result.pending_updates:
            for report in result.pending_updates:
                logging.error(
                    "Agent %s is out of date.",
                    report.spec.target_file.relative_to(repo_root),
                )
            return 2
        if result.stray_agents and (args.enforce_manifest or manifest.settings.enforce_tracked_agents):
            return 3
        logging.info("All agent files are in sync (%s).", summary)
        return 0

    if result.stray_agents and (args.enforce_manifest or manifest.settings.enforce_tracked_agents):
        return 3

    logging.info("AGENTS.md synchronization complete (%s).", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

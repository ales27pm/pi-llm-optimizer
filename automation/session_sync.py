#!/usr/bin/env python3
"""High-level repository maintenance workflow for end-of-session syncs."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from . import sync_agents

LOGGER = logging.getLogger("session_sync")


class SessionSyncError(RuntimeError):
    """Raised when the maintenance workflow cannot be completed."""


@dataclass(frozen=True)
class SessionSyncConfig:
    """User-configurable options for the session synchronisation workflow."""

    repo_root: Path
    manifest_path: Path
    check: bool = False
    skip_formatting: bool = False
    skip_agent_sync: bool = False
    skip_cleanup: bool = False
    run_npm_lint: bool = False
    run_pytest: bool = False
    enforce_manifest: Optional[bool] = None


@dataclass(frozen=True)
class FormattingSummary:
    """Information captured after running Prettier."""

    processed_files: Sequence[Path]
    mode: str


@dataclass(frozen=True)
class CleanupSummary:
    """Metrics describing filesystem cleanup."""

    pycache_directories_removed: int
    temp_files_removed: int


@dataclass(frozen=True)
class SessionSyncSummary:
    """Roll-up of the maintenance workflow."""

    formatting: Optional[FormattingSummary]
    agent_sync: Optional[sync_agents.SyncResult]
    cleanup: Optional[CleanupSummary]


class SessionSync:
    """Coordinate formatting, protocol generation, and workspace cleanup."""

    def __init__(self, config: SessionSyncConfig) -> None:
        self.config = config
        self.repo_root = config.repo_root
        self.manifest_path = config.manifest_path

    # Public API ---------------------------------------------------------
    def run(self) -> SessionSyncSummary:
        formatting_summary: Optional[FormattingSummary] = None
        agent_summary: Optional[sync_agents.SyncResult] = None
        cleanup_summary: Optional[CleanupSummary] = None

        if not self.config.skip_formatting:
            formatting_summary = self._format_markdown()

        if not self.config.skip_agent_sync:
            agent_summary = self._sync_agent_protocols()

        if not self.config.skip_cleanup:
            cleanup_summary = self._cleanup_workspace()

        if self.config.run_npm_lint:
            self._run_subprocess(["npm", "run", "lint"], "npm lint")

        if self.config.run_pytest:
            self._run_subprocess(["python", "-m", "pytest"], "pytest")

        return SessionSyncSummary(
            formatting=formatting_summary,
            agent_sync=agent_summary,
            cleanup=cleanup_summary,
        )

    # Internal helpers ---------------------------------------------------
    def _format_markdown(self) -> FormattingSummary:
        LOGGER.info("Formatting tracked Markdown files...")
        git = shutil.which("git")
        npx = shutil.which("npx")
        if git is None:
            raise SessionSyncError("Git is required to discover tracked markdown files.")
        if npx is None:
            raise SessionSyncError("npx is required to run Prettier. Install Node.js tooling.")

        command = [
            git,
            "ls-files",
            "*.md",
        ]
        result = subprocess.run(
            command,
            cwd=self.repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if not files:
            LOGGER.info("No markdown files detected; skipping Prettier.")
            return FormattingSummary(processed_files=(), mode="check" if self.config.check else "write")

        prettier_command: List[str] = [npx, "--yes", "prettier", "--log-level", "warn"]
        mode = "check" if self.config.check else "write"
        prettier_command.append(f"--{mode}")
        prettier_command.extend(str(path) for path in files)

        try:
            self._run_subprocess(prettier_command, f"prettier ({mode})")
        except SessionSyncError as exc:
            if self.config.check:
                raise
            raise SessionSyncError(f"Prettier formatting failed: {exc}") from exc
        return FormattingSummary(processed_files=tuple(self.repo_root / path for path in files), mode=mode)

    def _sync_agent_protocols(self) -> sync_agents.SyncResult:
        LOGGER.info("Synchronising scoped agent protocols...")
        manifest = sync_agents.load_manifest(self.manifest_path, self.repo_root)
        enforce_manifest = (
            self.config.enforce_manifest
            if self.config.enforce_manifest is not None
            else manifest.settings.enforce_tracked_agents
        )
        result = sync_agents.apply_manifest(
            manifest,
            repo_root=self.repo_root,
            write=not self.config.check,
            enforce_manifest=enforce_manifest,
        )
        if self.config.check and result.pending_updates:
            paths = ", ".join(str(report.spec.target_file.relative_to(self.repo_root)) for report in result.pending_updates)
            raise SessionSyncError(
                f"Agent protocols are out of sync: {paths}. Run without --check to regenerate files."
            )
        if enforce_manifest and result.stray_agents:
            stray = ", ".join(str(path.relative_to(self.repo_root)) for path in result.stray_agents)
            raise SessionSyncError(
                f"Untracked agent protocols detected: {stray}. Add them to the manifest or remove them."
            )
        return result

    def _cleanup_workspace(self) -> CleanupSummary:
        LOGGER.info("Removing cached Python bytecode directories...")
        pycache_dirs = list(self.repo_root.rglob("__pycache__"))
        removed_pycache = 0
        for directory in pycache_dirs:
            try:
                shutil.rmtree(directory)
                removed_pycache += 1
            except OSError as exc:
                raise SessionSyncError(f"Failed to remove {directory}: {exc}") from exc

        LOGGER.info("Removing temporary artefacts (*.tmp, *~, .DS_Store)...")
        temp_patterns = ("*.tmp", "*~", ".DS_Store")
        removed_files = 0
        for pattern in temp_patterns:
            for file_path in self.repo_root.rglob(pattern):
                try:
                    file_path.unlink()
                    removed_files += 1
                except OSError as exc:
                    raise SessionSyncError(f"Failed to remove {file_path}: {exc}") from exc

        return CleanupSummary(
            pycache_directories_removed=removed_pycache,
            temp_files_removed=removed_files,
        )

    def _run_subprocess(self, command: Sequence[str], step: str) -> None:
        LOGGER.info("Running %s...", step)
        try:
            subprocess.run(command, cwd=self.repo_root, check=True)
        except FileNotFoundError as exc:
            raise SessionSyncError(f"Required command for {step} not found: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            raise SessionSyncError(f"{step} failed with exit code {exc.returncode}") from exc


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the end-of-session maintenance workflow.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root (defaults to CWD).")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the agents manifest (defaults to automation/agents_manifest.json).",
    )
    parser.add_argument("--check", action="store_true", help="Dry-run mode. Fail when changes would be written.")
    parser.add_argument("--skip-formatting", action="store_true", help="Disable Prettier invocation.")
    parser.add_argument("--skip-agent-sync", action="store_true", help="Skip agent protocol synchronisation.")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip cache and temporary file cleanup.")
    parser.add_argument("--run-npm-lint", action="store_true", help="Execute `npm run lint` as part of the workflow.")
    parser.add_argument("--run-pytest", action="store_true", help="Execute `python -m pytest` as part of the workflow.")
    parser.add_argument(
        "--enforce-manifest",
        action="store_true",
        help="Force enforcement of the manifest, even if disabled in settings.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest or (repo_root / "automation" / sync_agents.MANIFEST_NAME)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[session_sync] %(levelname)s %(message)s")

    config = SessionSyncConfig(
        repo_root=repo_root,
        manifest_path=manifest_path,
        check=args.check,
        skip_formatting=args.skip_formatting,
        skip_agent_sync=args.skip_agent_sync,
        skip_cleanup=args.skip_cleanup,
        run_npm_lint=args.run_npm_lint,
        run_pytest=args.run_pytest,
        enforce_manifest=True if args.enforce_manifest else None,
    )

    try:
        summary = SessionSync(config).run()
    except SessionSyncError as exc:
        LOGGER.error("%s", exc)
        return 1

    if summary.formatting:
        LOGGER.info(
            "Prettier completed in %s mode for %d files.",
            summary.formatting.mode,
            len(summary.formatting.processed_files),
        )
    if summary.agent_sync:
        LOGGER.info(
            "Agent synchronisation complete: %s",
            ", ".join(report.status.value for report in summary.agent_sync.reports),
        )
    if summary.cleanup:
        LOGGER.info(
            "Cleanup removed %d __pycache__ directories and %d temp files.",
            summary.cleanup.pycache_directories_removed,
            summary.cleanup.temp_files_removed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

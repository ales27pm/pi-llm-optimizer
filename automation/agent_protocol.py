"""Helpers for loading the repository's agent protocol specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "AgentProtocolError",
    "AgentProtocolMetadata",
    "DEFAULT_PROTOCOL_PATH",
    "load_agent_protocol_metadata",
]


class AgentProtocolError(RuntimeError):
    """Raised when the agent protocol file is missing or malformed."""


@dataclass(frozen=True)
class AgentProtocolMetadata:
    """Structured metadata parsed from ``AGENTS.md``."""

    title: str
    version: str
    path: Path


DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "AGENTS.md"


def load_agent_protocol_metadata(path: Optional[Path] = None) -> AgentProtocolMetadata:
    """Return the protocol metadata extracted from ``AGENTS.md``.

    Parameters
    ----------
    path:
        Optional path override for locating the protocol file. When omitted the
        repository root ``AGENTS.md`` is inspected.
    """

    protocol_path = path or DEFAULT_PROTOCOL_PATH
    if not protocol_path.exists():
        raise AgentProtocolError(f"Agent protocol file not found at {protocol_path}.")

    raw_text = protocol_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise AgentProtocolError("Agent protocol file is empty.")

    header = lines[0]
    if not header.startswith("# "):
        raise AgentProtocolError("Agent protocol header must begin with '# '.")
    title = header[2:].strip()
    if not title:
        raise AgentProtocolError("Agent protocol title is missing.")

    version_line = next((line for line in lines[1:] if line.lower().startswith("version:")), None)
    if version_line is None:
        raise AgentProtocolError("Agent protocol must define a 'version:' metadata field immediately after the header.")

    version = version_line.split(":", 1)[1].strip()
    if not version:
        raise AgentProtocolError("Agent protocol version metadata cannot be empty.")

    return AgentProtocolMetadata(title=title, version=version, path=protocol_path)

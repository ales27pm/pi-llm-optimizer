"""Utilities for constructing and executing Pi-LLM optimisation pipeline commands.

The automation scripts in this repository are intentionally lightweight wrappers
around standalone entry points such as ``desktop_distill/teacher_label.py`` or
``rpi4/bench/pi_bench.py``.  While this keeps each script focused, it makes it
harder to build higher-level tools (interactive UIs, API servers, etc.) that can
compose the full workflow.

This module provides a typed facade around those entry points.  Each pipeline
stage exposes a :class:`dataclasses.dataclass` describing the required and
optional parameters.  Instances can validate basic invariants and generate a
process command-line invocation.  The resulting commands are safe to pass to
``subprocess`` and are suitable for reuse by terminal user interfaces or custom
automation.

The goal is to centralise the mechanics of command construction so that new
clients (such as the interactive UI implemented in ``automation/ui_app.py``)
can focus on UX concerns while remaining testable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

Command = List[str]


def _python_executable() -> str:
    """Return the interpreter that should execute pipeline scripts."""

    return sys.executable or "python3"


def _coerce_path(value: Path | str) -> str:
    """Convert ``Path`` objects into POSIX strings for subprocess usage."""

    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    return str(Path(value).expanduser().resolve())


def _extend(command: Command, flag: str, value: Optional[str] | Optional[int] | Optional[float]) -> None:
    """Append ``flag`` and ``value`` to the command if ``value`` is provided."""

    if value is None:
        return
    if isinstance(value, Path):
        value = _coerce_path(value)
    command.extend([flag, str(value)])


def _extend_flag(command: Command, flag: str, enabled: bool) -> None:
    """Append a boolean flag when ``enabled`` is ``True``."""

    if enabled:
        command.append(flag)


@dataclass(slots=True)
class TeacherLabelConfig:
    """Configuration for :mod:`desktop_distill.teacher_label`."""

    model: str
    input_path: Path
    output_path: Path
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    device_map: Optional[str] = None
    batch_size: int = 1
    skip_existing: bool = False

    def build_command(self, python_executable: Optional[str] = None) -> Command:
        """Return the CLI command to execute the teacher labelling script."""

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        python = python_executable or _python_executable()
        command: Command = [python, _coerce_path(Path("desktop_distill/teacher_label.py"))]
        _extend(command, "--model", self.model)
        _extend(command, "--input", self.input_path)
        _extend(command, "--output", self.output_path)
        _extend(command, "--max_new_tokens", self.max_new_tokens)
        _extend_flag(command, "--do_sample", self.do_sample)
        _extend(command, "--temperature", self.temperature)
        _extend(command, "--device_map", self.device_map)
        _extend(command, "--batch_size", self.batch_size)
        _extend_flag(command, "--skip_existing", self.skip_existing)
        return command


@dataclass(slots=True)
class TrainStudentConfig:
    """Configuration for :mod:`desktop_distill.train_student`."""

    dataset: Path
    base_model: str
    output_dir: Path
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    use_dora: bool = False
    qlora: bool = False
    seed: Optional[int] = None
    logging_steps: Optional[int] = None

    extra_args: Sequence[str] = field(default_factory=list)

    def build_command(self, python_executable: Optional[str] = None) -> Command:
        """Return the CLI command for the student training script."""

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")

        python = python_executable or _python_executable()
        command: Command = [python, _coerce_path(Path("desktop_distill/train_student.py"))]
        _extend(command, "--dataset", self.dataset)
        _extend(command, "--base_model", self.base_model)
        _extend(command, "--output_dir", self.output_dir)
        _extend(command, "--num_epochs", self.num_epochs)
        _extend(command, "--batch_size", self.batch_size)
        _extend(command, "--gradient_accumulation_steps", self.gradient_accumulation_steps)
        _extend(command, "--learning_rate", self.learning_rate)
        _extend_flag(command, "--use_dora", self.use_dora)
        _extend_flag(command, "--qlora", self.qlora)
        _extend(command, "--seed", self.seed)
        _extend(command, "--logging_steps", self.logging_steps)
        if self.extra_args:
            command.extend(list(self.extra_args))
        return command


@dataclass(slots=True)
class ExportConfig:
    """Configuration for :mod:`desktop_distill.export_gguf`."""

    model_path: Path
    output_dir: Path
    quant_type: str = "q4_k_m"
    hf_token: Optional[str] = None

    def build_command(self, python_executable: Optional[str] = None) -> Command:
        python = python_executable or _python_executable()
        command: Command = [python, _coerce_path(Path("desktop_distill/export_gguf.py"))]
        _extend(command, "--model", self.model_path)
        _extend(command, "--outdir", self.output_dir)
        _extend(command, "--qtype", self.quant_type)
        if self.hf_token:
            _extend(command, "--hf-token", self.hf_token)
        return command


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for :mod:`rpi4.bench.pi_bench`."""

    model: Path
    iterations: int = 3
    min_tokps: Optional[float] = None
    csv_path: Optional[Path] = None

    def build_command(self, python_executable: Optional[str] = None) -> Command:
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        python = python_executable or _python_executable()
        command: Command = [python, _coerce_path(Path("rpi4/bench/pi_bench.py"))]
        _extend(command, "--model", self.model)
        _extend(command, "--iterations", self.iterations)
        _extend(command, "--min-tokps", self.min_tokps)
        _extend(command, "--csv", self.csv_path)
        return command


def env_with_hf_token(token: Optional[str]) -> dict[str, str]:
    """Return an environment mapping with ``HF_TOKEN`` when provided."""

    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    return env


__all__ = [
    "TeacherLabelConfig",
    "TrainStudentConfig",
    "ExportConfig",
    "BenchmarkConfig",
    "env_with_hf_token",
]


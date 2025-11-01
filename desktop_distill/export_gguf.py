"""
export_gguf.py
================

Convert a HuggingFace model directory or model id into the GGUF format
and quantize it for use with the llama.cpp runtime.  This script wraps
the `convert_hf_to_gguf.py` converter and the `llama-quantize` binary
provided by the llama.cpp project.  It produces a directory containing
the quantized `.gguf` file, tokenizer and prompt cache.

Compared to the previous implementation, the script now supports
automatically downloading remote HuggingFace models, improved logging
for debugging failed conversions and richer tokenizer asset handling.

Usage::

    python export_gguf.py --model path/to/model --outdir gguf_artifacts --qtype q4_k_m

You must have llama.cpp cloned at `~/llama.cpp` (or configured via the
`LLAMA_CPP_DIR` environment variable) with its build artifacts
available.  The script looks for `convert_hf_to_gguf.py` and
`llama-quantize` in that directory.  If the converter is not found, it
will emit an error.

"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_DIR = Path.home() / "llama.cpp"
LLAMA_CPP_ENV_VAR = "LLAMA_CPP_DIR"
TOKENIZER_FILENAMES: Sequence[str] = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer_cache.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)
TOKENIZER_DIRECTORIES: Sequence[str] = ("tokenizer",)


def _format_cmd(cmd: Sequence[object]) -> str:
    return " ".join(str(part) for part in cmd)


def run_cmd(cmd: Sequence[object], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute a shell command, logging stdout/stderr and raising on failure."""

    display_cmd = _format_cmd(cmd)
    logger.info("Running command: %s", display_cmd)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.stdout:
        logger.debug("%s stdout:%s%s", display_cmd, os.linesep, result.stdout.strip())
    if result.stderr:
        logger.debug("%s stderr:%s%s", display_cmd, os.linesep, result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command '{display_cmd}' failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    return result


def _get_snapshot_download():
    """Return huggingface_hub.snapshot_download, raising if unavailable."""

    try:
        hub_module = importlib.import_module("huggingface_hub")
    except ImportError as exc:  # pragma: no cover - exercised in unit tests
        raise ImportError(
            "huggingface_hub is required to download remote models. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    snapshot_download = getattr(hub_module, "snapshot_download", None)
    if snapshot_download is None:  # pragma: no cover - defensive guard
        raise ImportError(
            "huggingface_hub.snapshot_download is not available. Upgrade huggingface_hub to a recent version."
        )
    return snapshot_download

class ModelResolutionError(RuntimeError):
    """Raised when a remote HuggingFace repository cannot be resolved."""

    def __init__(self, repo_id: str, reason: str, *, hints: Sequence[str]) -> None:
        self.repo_id = repo_id
        self.reason = reason
        self.hints = tuple(hints)
        hint_block = "\n".join(f"  - {hint}" for hint in self.hints)
        message = (
            f"Failed to resolve HuggingFace repository '{repo_id}': {reason}\n"
            f"Remediation steps:\n{hint_block}"
        )
        super().__init__(message)


@dataclass
class _ProgressState:
    last_fraction: float = -0.1
    last_logged_bytes: int = 0


class DownloadProgressReporter:
    """Bridge huggingface_hub progress updates into structured logs."""

    def __init__(self, repo_id: str, *, min_fraction: float = 0.05, min_bytes: int = 10_000_000) -> None:
        self.repo_id = repo_id
        self._min_fraction = min_fraction
        self._min_bytes = min_bytes
        self._states: Dict[str, _ProgressState] = {}

    def make_tqdm_class(self):
        try:
            from tqdm.auto import tqdm as base_tqdm  # type: ignore
        except ImportError:  # pragma: no cover - tqdm is an optional dependency
            logger.debug("tqdm is not available; download progress logging disabled")
            return None

        reporter = self

        class LoggingTqdm(base_tqdm):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                reporter._start(self.desc or "download", self.total)

            def update(self, n=1):  # type: ignore[override]
                super().update(n)
                reporter._progress(self.desc or "download", self.n, self.total)

            def close(self):  # type: ignore[override]
                try:
                    reporter._complete(self.desc or "download", self.n, self.total)
                finally:
                    super().close()

        return LoggingTqdm

    def _state_for(self, target: str) -> _ProgressState:
        if target not in self._states:
            self._states[target] = _ProgressState()
        return self._states[target]

    def _log(self, event: str, target: str, downloaded: int, total: Optional[int], *, extra: Optional[dict] = None) -> None:
        payload: Dict[str, object] = {
            "event": event,
            "repo": self.repo_id,
            "target": target,
            "downloaded_bytes": downloaded,
            "timestamp": time.time(),
        }
        if total is not None:
            payload["total_bytes"] = total
            if total:
                payload["percent"] = round(downloaded / total * 100, 2)
        else:
            payload["total_bytes"] = None
        if extra:
            payload |= extra
        logger.info(json.dumps(payload, sort_keys=True))

    def _start(self, target: str, total: Optional[int]) -> None:
        self._log("download_start", target, 0, total)

    def _progress(self, target: str, downloaded: int, total: Optional[int]) -> None:
        state = self._state_for(target)
        should_log = False
        if total:
            fraction = downloaded / total if total else 0
            if fraction >= state.last_fraction + self._min_fraction or downloaded == total:
                state.last_fraction = fraction
                should_log = True
        elif downloaded - state.last_logged_bytes >= self._min_bytes or downloaded == total:
            should_log = True
        if should_log:
            state.last_logged_bytes = downloaded
            self._log("download_progress", target, downloaded, total)

    def _complete(self, target: str, downloaded: int, total: Optional[int]) -> None:
        self._log("download_complete", target, downloaded, total)

@contextmanager
def _resolve_model_path(model: str, *, revision: Optional[str], token: Optional[str]) -> Iterator[Path]:
    """Yield a local directory containing the model weights."""

    candidate = Path(model)
    if candidate.exists():
        yield candidate.resolve()
        return

    snapshot_download = _get_snapshot_download()
    reporter = DownloadProgressReporter(model)
    tqdm_class = reporter.make_tqdm_class()

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_dir = Path(tmp_dir) / "model"
        download_dir.mkdir(parents=True, exist_ok=True)
        snapshot_kwargs = {
            "repo_id": model,
            "revision": revision,
            "token": token,
            "local_dir": str(download_dir),
            "local_dir_use_symlinks": False,
        }
        if tqdm_class is not None:
            snapshot_kwargs["tqdm_class"] = tqdm_class

        try:
            snapshot_download(**snapshot_kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # pragma: no cover - exercised via dedicated tests
            raise _convert_snapshot_error(model, exc, revision=revision) from exc

        yield download_dir


def _convert_snapshot_error(model: str, exc: Exception, *, revision: Optional[str]) -> ModelResolutionError:
    """Convert huggingface_hub exceptions into actionable guidance."""

    reason = str(exc) or exc.__class__.__name__
    hints: list[str] = [
        "Verify the model id on https://huggingface.co and ensure it is spelled correctly.",
        "If the repository is private, provide an access token via --hf-token or the HF_TOKEN environment variable.",
        "Confirm your network/proxy settings allow outbound HTTPS connections to huggingface.co.",
    ]

    try:
        from huggingface_hub.utils import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError:  # pragma: no cover - huggingface_hub already imported earlier
        EntryNotFoundError = HfHubHTTPError = RepositoryNotFoundError = RevisionNotFoundError = ()  # type: ignore[assignment]

    try:
        if isinstance(exc, RepositoryNotFoundError):  # type: ignore[arg-type]
            reason = "Repository was not found or access is denied."
            hints.insert(0, "Check that the repository exists and that your account has accepted any gating terms.")
        elif isinstance(exc, RevisionNotFoundError):  # type: ignore[arg-type]
            rev = revision or "<default>"
            reason = f"Revision '{rev}' does not exist in the repository."
            hints.insert(0, "Use --revision to target a valid branch, tag or commit hash.")
        elif isinstance(exc, EntryNotFoundError):  # type: ignore[arg-type]
            reason = "The repository is missing expected model files."
            hints.insert(0, "Ensure the repository contains the model weights and configuration files.")
        elif isinstance(exc, HfHubHTTPError):  # type: ignore[arg-type]
            if status_code := getattr(getattr(exc, "response", None), "status_code", None):
                reason = f"HTTP {status_code} error while downloading the repository."
            hints.append("Retry the export; transient HuggingFace outages can cause HTTP errors.")
    except Exception:  # pragma: no cover - defensive logging of unexpected errors
        logger.error(
            "Unexpected exception while converting snapshot error for model '%s' (revision: %s):\n%s",
            model,
            revision,
            traceback.format_exc(),
        )
        raise

    payload = {
        "event": "model_resolution",
        "status": "error",
        "repo": model,
        "reason": reason,
        "exception_type": exc.__class__.__name__,
    }
    logger.error(json.dumps(payload, sort_keys=True))

    return ModelResolutionError(model, reason, hints=hints)


def _determine_converter_dir(provided: Optional[Path]) -> Path:
    if provided is not None:
        return provided
    if env_dir := os.getenv(LLAMA_CPP_ENV_VAR):
        return Path(env_dir)
    return DEFAULT_LLAMA_CPP_DIR


def _determine_output_filename(model_identifier: str, model_path: Path, qtype: str) -> str:
    if Path(model_identifier).exists():
        stem = model_path.name or "model"
    else:
        stem = model_path.name
        if not stem or stem == "model":
            identifier = model_identifier.strip().strip("/")
            if identifier:
                stem = identifier.replace("/", "_")
            else:
                stem = "model"
    safe_stem = stem.replace(" ", "_").replace("/", "_")
    return f"{safe_stem}-{qtype}.gguf"


def _iter_tokenizer_assets(model_path: Path) -> Iterable[Path]:
    for filename in TOKENIZER_FILENAMES:
        candidate = model_path / filename
        if candidate.exists():
            yield candidate
    for dirname in TOKENIZER_DIRECTORIES:
        directory = model_path / dirname
        if directory.is_dir():
            yield directory


def _copy_tokenizer_assets(model_path: Path, outdir: Path) -> None:
    for asset in _iter_tokenizer_assets(model_path):
        destination = outdir / asset.name
        if asset.is_dir():
            shutil.copytree(asset, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(asset, destination)
        logger.info("Copied %s -> %s", asset, destination)


def export_gguf(
    model: str,
    outdir: Path,
    qtype: str = "q4_k_m",
    converter_dir: Optional[Path] = None,
    *,
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    preserve_tmp_dir: bool = False,
) -> None:
    """
    Convert a HuggingFace model to GGUF and quantize it.

    :param model: Path or HF model id of the fine tuned model.
    :param outdir: Directory where the GGUF and artifacts will be saved.
    :param qtype: Quantization type, e.g. q4_k_m, q3_k_m, q5_k_m.
    :param converter_dir: Optional directory containing llama.cpp and its tools.
    :param revision: Optional revision/branch/tag for HuggingFace models.
    :param hf_token: Optional HuggingFace token for private repositories.
    :param preserve_tmp_dir: Preserve the temporary working directory on failure for debugging.
    """

    resolved_converter_dir = _determine_converter_dir(converter_dir)
    convert_script = resolved_converter_dir / "convert_hf_to_gguf.py"
    quant_bin = resolved_converter_dir / "build/bin/llama-quantize"
    if not convert_script.exists():
        raise FileNotFoundError(f"Converter script not found: {convert_script}")
    if not quant_bin.exists():
        raise FileNotFoundError(f"Quantize binary not found: {quant_bin}")

    outdir.mkdir(parents=True, exist_ok=True)
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    tmp_dir_ctx: Optional[tempfile.TemporaryDirectory] = None
    if preserve_tmp_dir:
        tmp_path = Path(tempfile.mkdtemp(prefix="export_gguf_"))
    else:
        tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir_ctx.name)

    try:
        with _resolve_model_path(model, revision=revision, token=hf_token) as model_path:
            f16_path = tmp_path / "model-f16.gguf"
            cmd_convert: list[object] = [
                "python",
                str(convert_script),
                "--outtype",
                "f16",
                "--outfile",
                str(f16_path),
                str(model_path),
            ]
            run_cmd(cmd_convert)

            q_filename = _determine_output_filename(model, model_path, qtype)
            q_path = outdir / q_filename
            cmd_quant: list[object] = [
                str(quant_bin),
                str(f16_path),
                str(q_path),
                qtype,
            ]
            run_cmd(cmd_quant)

            _copy_tokenizer_assets(model_path, outdir)
            logger.info("Quantized model available at %s", q_path)
    except Exception:
        if preserve_tmp_dir:
            logger.warning("Error occurred; temporary directory preserved at: %s", tmp_path)
        else:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()
                tmp_dir_ctx = None
        raise
    else:
        if not preserve_tmp_dir and tmp_dir_ctx is not None:
            tmp_dir_ctx.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a HF model to GGUF and quantize it")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF model id of the model to export",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for the GGUF artifact",
    )
    parser.add_argument(
        "--qtype",
        type=str,
        default="q4_k_m",
        help="Quantization type (default: q4_k_m)",
    )
    parser.add_argument(
        "--converter-dir",
        type=Path,
        default=None,
        help="Directory containing llama.cpp",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HuggingFace revision (branch/tag/commit)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for private repositories",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--preserve-tmp-dir",
        action="store_true",
        help="Preserve the temporary working directory on failure for debugging",
    )
    args = parser.parse_args()

    requested_level = args.log_level.upper()
    resolved_level = logging.getLevelName(requested_level)
    if isinstance(resolved_level, int):
        logging.basicConfig(level=resolved_level)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.warning("Invalid log level '%s'; defaulting to INFO", args.log_level)

    export_gguf(
        args.model,
        args.outdir,
        args.qtype,
        args.converter_dir,
        revision=args.revision,
        hf_token=args.hf_token,
        preserve_tmp_dir=args.preserve_tmp_dir,
    )
    print(f"Exported GGUF to {args.outdir} (quant={args.qtype})")


if __name__ == "__main__":
    main()

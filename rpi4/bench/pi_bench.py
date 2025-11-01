from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_csv import BenchmarkCSVWriter, BenchmarkRow
from history_utils import DEFAULT_HISTORY_LIMIT, append_history
from throughput_regressor import ValidationError, validate

LLAMA_BIN = Path.home() / "llama.cpp/build/bin/llama-cli"
DEFAULT_HISTORY_PATH = Path("rpi4/bench/out/bench_history.json")
DEFAULT_WARMUP_TOKENS = 16
DEFAULT_DECODE_TOKENS = 64

LOGGER = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        LOGGER.warning("Unable to resolve git commit hash: %s", exc)
        return None
    return result.stdout.strip() or None


def _device_model() -> str | None:
    model_path = Path("/proc/device-tree/model")
    try:
        model = model_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return model.replace("\x00", "").strip() or None


def _kernel_version() -> str:
    return platform.uname().release


def _row_to_dict(row: BenchmarkRow, iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "timestamp": row.timestamp.astimezone(dt.timezone.utc).isoformat(),
        "tokens_per_second": row.tokens_per_second,
        "init_ms": row.init_ms,
        "decode_ms_per_token": row.decode_ms_per_token,
        "embed_ms": row.embed_ms,
    }


def _build_record(
    *,
    rows: list[BenchmarkRow],
    summary: tuple[float, float],
    args: argparse.Namespace,
    csv_path: Path,
    json_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    min_tokps, avg_tokps = summary
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "timestamp": now.isoformat(),
        "model": str(Path(args.model).expanduser()),
        "prompt": args.prompt,
        "context_length": args.ctx,
        "batch_size": args.batch,
        "threads": args.threads,
        "kv_cache_type": args.kv_type,
        "warmup_tokens": metadata["warmup_tokens"],
        "decode_tokens": metadata["decode_tokens"],
        "iterations": len(rows),
        "minimum_required_tokps": args.min_tokps,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": {
            "minimum_observed_tokps": min_tokps,
            "average_tokps": avg_tokps,
        },
        "samples": [
            _row_to_dict(row, idx + 1)
            for idx, row in enumerate(rows)
        ],
        "metadata": metadata,
    }


def _run_llama(args: list[str]) -> None:
    LOGGER.debug("Executing llama-cli with args: %s", " ".join(args))
    subprocess.run(
        [
            str(LLAMA_BIN),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def bench(
    model_path: str,
    prompt: str,
    ctx: int,
    batch: int,
    threads: int,
    kv_type: str,
    warmup: int = DEFAULT_WARMUP_TOKENS,
    decode_tokens: int = DEFAULT_DECODE_TOKENS,
) -> tuple[float, float, float]:
    """Run a benchmark and return (init_ms, tokps, embed_ms)."""
    # Measure initialization/prefill
    t0 = time.perf_counter()
    _run_llama([
        "-m", model_path,
        "-c", str(ctx),
        "-b", str(batch),
        "-t", str(threads),
        "--kv-type", kv_type,
        "-p", prompt,
        "-n", "0",
    ])
    t1 = time.perf_counter()
    init_ms = (t1 - t0) * 1e3

    # Measure decode throughput
    t2 = time.perf_counter()
    _run_llama([
        "-m", model_path,
        "-c", str(ctx),
        "-b", str(batch),
        "-t", str(threads),
        "--kv-type", kv_type,
        "-p", prompt,
        "-n", str(warmup + decode_tokens),
    ])
    t3 = time.perf_counter()
    tokps = (warmup + decode_tokens) / ((t3 - t2))

    # Measure embedding time
    t4 = time.perf_counter()
    _run_llama([
        "-m", model_path,
        "--embedding",
        "-p", prompt,
        "-n", "0",
    ])
    t5 = time.perf_counter()
    embed_ms = (t5 - t4) * 1e3
    return init_ms, tokps, embed_ms


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[pi_bench] %(message)s")
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp model on a Raspberry Pi"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Test.",
        help="Prompt to use for benchmarking",
    )
    parser.add_argument("--ctx", type=int, default=1024, help="Context length")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument(
        "--kv-type",
        type=str,
        default="q8_0",
        help="KV cache quantisation type",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--min-tokps",
        type=float,
        default=0.25,
        help="Minimum acceptable tokens/s",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("rpi4/bench/out/bench.csv"),
        help="Path to write CSV report",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to write JSON benchmark history",
    )
    parser.add_argument(
        "--json-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help="Maximum number of entries to retain in the JSON history",
    )
    args = parser.parse_args()

    rows = []
    rates = []
    warmup_tokens = DEFAULT_WARMUP_TOKENS
    decode_tokens = DEFAULT_DECODE_TOKENS
    metadata = {
        "device_model": _device_model(),
        "kernel": _kernel_version(),
        "commit": _git_commit(),
        "llama_binary": str(LLAMA_BIN),
        "warmup_tokens": warmup_tokens,
        "decode_tokens": decode_tokens,
    }
    csv_path = args.csv.expanduser().resolve()
    json_path = args.json.expanduser().resolve()

    for iteration in range(args.iterations):
        LOGGER.info("Starting benchmark iteration %d/%d", iteration + 1, args.iterations)
        init_ms, tokps, embed_ms = bench(
            args.model,
            args.prompt,
            args.ctx,
            args.batch,
            args.threads,
            args.kv_type,
            warmup_tokens,
            decode_tokens,
        )
        rows.append(BenchmarkRow(
            dt.datetime.now(dt.timezone.utc),
            tokps,
            init_ms,
            1000.0 / tokps,
            embed_ms,
        ))
        rates.append(tokps)

    # Validate throughput
    try:
        summary = validate(rates, args.min_tokps)
    except ValidationError as exc:
        LOGGER.error("Throughput validation failed: %s", exc)
        raise SystemExit(1) from exc
    # Write CSV
    BenchmarkCSVWriter().write(rows, csv_path)

    record = _build_record(
        rows=rows,
        summary=(summary.minimum_observed, summary.average_tokens_per_second),
        args=args,
        csv_path=csv_path,
        json_path=json_path,
        metadata=metadata,
    )
    history = append_history(json_path, record, limit=args.json_limit)

    LOGGER.info("Wrote CSV report to %s", csv_path)
    LOGGER.info("Updated benchmark history at %s (%d entries)", json_path, len(history))

    print(json.dumps({
        "record": record,
        "history_path": str(json_path),
        "history_size": len(history),
    }))


if __name__ == "__main__":
    main()

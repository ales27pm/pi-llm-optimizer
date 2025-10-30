from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path
from typing import Tuple

from benchmark_csv import BenchmarkCSVWriter, BenchmarkRow
from throughput_regressor import validate

LLAMA_BIN = Path.home() / "llama.cpp/build/bin/llama-cli"


def _run_llama(args: list[str]) -> None:
    subprocess.run([
        str(LLAMA_BIN),
        *args,
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def bench(model_path: str, prompt: str, ctx: int, batch: int, threads: int, kv_type: str,
          warmup: int = 16, decode_tokens: int = 64) -> Tuple[float, float, float]:
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
    args = parser.parse_args()

    rows = []
    rates = []
    for _ in range(args.iterations):
        init_ms, tokps, embed_ms = bench(
            args.model,
            args.prompt,
            args.ctx,
            args.batch,
            args.threads,
            args.kv_type,
        )
        rows.append(BenchmarkRow(
            dt.datetime.utcnow(),
            tokps,
            init_ms,
            1000.0 / tokps,
            embed_ms,
        ))
        rates.append(tokps)

    # Validate throughput
    summary = validate(rates, args.min_tokps)
    # Write CSV
    BenchmarkCSVWriter().write(rows, args.csv)
    print(
        {
            "min_tokps": summary.minimum_observed,
            "avg_tokps": summary.average_tokens_per_second,
            "csv": str(args.csv),
        }
    )


if __name__ == "__main__":
    main()
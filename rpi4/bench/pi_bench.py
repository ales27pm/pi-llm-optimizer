import json, subprocess, time
from pathlib import Path
from datetime import datetime
from benchmark_csv import BenchmarkRow, BenchmarkCSVWriter
from throughput_regressor import validate

LLAMA_BIN = Path.home()/"llama.cpp/build/bin/llama-cli"

def _run_llama(args: list[str]) -> None:
    subprocess.run([str(LLAMA_BIN), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def bench(model_path: str, prompt: str, ctx: int=1024, batch: int=64, threads: int=4, kv_type: str="q8_0", warmup: int=16, decode_tokens: int=64):
    t0 = time.perf_counter(); _run_llama(["-m", model_path, "-c", str(ctx), "-b", str(batch), "-t", str(threads), "--kv-type", kv_type, "-p", prompt, "-n", "0"]); t1 = time.perf_counter()
    init_ms = (t1 - t0) * 1e3
    t2 = time.perf_counter(); _run_llama(["-m", model_path, "-c", str(ctx), "-b", str(batch), "-t", str(threads), "--kv-type", kv_type, "-p", prompt, "-n", str(warmup+decode_tokens)]); t3 = time.perf_counter()
    total_ms = (t3 - t2) * 1e3; step_ms = total_ms / max(1, warmup+decode_tokens); tokps = 1000.0 / step_ms
    t4 = time.perf_counter(); _run_llama(["-m", model_path, "--embedding", "-p", prompt, "-n", "0"]); t5 = time.perf_counter()
    embed_ms = (t5 - t4) * 1e3
    return init_ms, tokps, embed_ms

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Pi local LLMs are possible.")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kv-type", default="q8_0")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--min-tokps", type=float, default=0.25)
    ap.add_argument("--csv", type=Path, default=Path("rpi4/bench/out/bench.csv"))
    args = ap.parse_args()

    rows, rates = [], []
    for _ in range(args.iterations):
        init_ms, tokps, embed_ms = bench(args.model, args.prompt, args.ctx, args.batch, args.threads, args.kv_type)
        rows.append(BenchmarkRow(datetime.utcnow(), tokps, init_ms, 1000.0/tokps, embed_ms))
        rates.append(tokps)

    summary = validate(rates, args.min_tokps)
    BenchmarkCSVWriter().write(rows, args.csv)
    print(json.dumps({"min_tokps": summary.minimum_observed, "avg_tokps": summary.average_tokens_per_second, "csv": str(args.csv)}, indent=2))

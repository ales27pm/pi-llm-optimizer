# Automation & Dashboard Overview

The `automation` package bundles the Textual-based pipeline UI, Codex session
sync utilities and roadmap tooling.  The UI wraps the command builders defined
in `automation/pipeline_ops.py`, providing validated forms for each stage of the
Pi-LLM workflow.

## Launching the Textual UI

Install the dependencies and run the app:

```bash
pip install -r automation/requirements.txt
python -m automation.ui_app
```

Use `--panel` to focus on a specific workflow tab at start-up.  For example,
`python -m automation.ui_app --panel train` opens the student-training form
immediately.

## Benchmark Dashboard Mode

Pass `--dashboard` to collapse the UI to the benchmark tab and stream historical
metrics from the Raspberry Pi bench runs:

```bash
python -m automation.ui_app --dashboard --benchmark-refresh 30
```

* `--benchmark-history` points the dashboard at a JSON file (defaults to
  `rpi4/bench/out/bench_history.json`).
* `--benchmark-refresh` defines the polling cadence in seconds (minimum 1 s,
  default 30 s).

The dashboard renders the latest entries from the history file as a sparkline
and data table.  Each run captures the device model, kernel version, git commit,
minimum/average token rates and per-iteration timings as emitted by
`rpi4/bench/pi_bench.py`.

Ensure you execute the benchmark script with `--json` so new runs append to the
history file:

```bash
python rpi4/bench/pi_bench.py --model <path/to/model.gguf> --json rpi4/bench/out/bench_history.json
```

By default the benchmark script retains the most recent 200 runs; adjust via
`--json-limit` if you need a longer or shorter window.

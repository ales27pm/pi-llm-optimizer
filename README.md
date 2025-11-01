# Pi‑LLM Optimizer

This repository contains everything you need to train, quantize and deploy a
compact large‑language model on a Raspberry Pi 4.  It was built as a
companion to the `CoreML‑DOLPHIN3.0` project and demonstrates how to
distill a larger *teacher* model (e.g. the Dolphin 8 B model) into a much
smaller *student* model that fits into 4 GB of RAM and runs locally on
ARM64 hardware.  The training pipeline runs on your desktop (Ubuntu/x86_64) and
produces a GGUF artifact that is then transferred to the Pi for inference
using [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

* **Knowledge Distillation** – Use a powerful teacher model to generate
  labelled examples from a lightweight dataset.  The
  `desktop_distill/teacher_label.py` script concatenates `system_hint`,
  `user` and optional JSON schema instructions, supports batched
  generation, and can skip previously answered prompts so incremental
  labelling runs are fast.
* **Student Training** – Fine-tune a small base model (e.g. TinyLlama
  1.1 B or Qwen 2.5 B) with LoRA or DoRA adapters, gradient checkpointing,
  mixed precision and optional QLoRA (k‑bit) training.  The
  `desktop_distill/train_student.py` script validates dataset integrity,
  configures tokenizer padding automatically and merges adapters back
  into the base model on completion.
* **GGUF Export** – Convert your fine‑tuned HuggingFace model to the
  GGUF format and quantize it (e.g. `q4_k_m`) for maximum runtime
  efficiency.  The `desktop_distill/export_gguf.py` script resolves local
  or remote models (via `huggingface_hub`), logs llama.cpp commands,
  preserves tokenizer assets and optionally keeps temporary directories
  for debugging failed runs.
* **Automated Workflow & UI** – A Makefile, `automation/e2e.sh` script
  and reusable command builders in `automation/pipeline_ops.py`
  orchestrate the entire process: labelling, training, export,
  deployment and on-device benchmarking.  A Textual-powered dashboard
  (`automation/ui_app.py`) reuses the same builders to provide a guided
  terminal UI.
* **Codex Session Sync Automation** – `automation/session_sync.py`
  centralises Prettier formatting, scoped agent protocol regeneration,
  optional lint/test execution, and workspace cleanup for the internal
  @codex assistant environment. The orchestration runs through
  `automation/update_and_cleanup.sh`, which @codex maintainers must run
  at the end of every session, and is not part of the standard project
  toolchain for external contributors.
* **Dataset Documentation** – `dataset/qf_corpus_blueprint/scripts/dataset_card.py`
  summarises register and dialect coverage, hashes corpora and emits
  provenance-rich dataset cards through a reusable library and CLI.
* **Raspberry Pi Runtime** – Scripts in the `rpi4/` directory build
  llama.cpp on ARM64 (with OpenBLAS), download a prebuilt model if
  desired, run the GGUF model interactively, fetch embeddings, and
  benchmark throughput to guard against regressions.

## Quickstart

### 1. Prepare the Dataset

Create a dataset of user prompts and metadata in JSONL format under
`dataset/`.  Each line must contain at least a `user` field, and may
optionally include:

* `system_hint` – system‑level instructions or context
* `requires_json` – whether the answer should be valid JSON
* `tool_schema` – a JSON schema string to constrain the output when
  `requires_json` is true

Run the labeller on your desktop (add `--skip_existing` for incremental reruns):

```bash
python desktop_distill/teacher_label.py \
    --model openai/dolphin  \
    --input dataset/dataset_quebecois_distill.jsonl \
    --output dataset/labelled.jsonl
```

The output file will contain the teacher’s response in the `assistant`
field for each record.

### 2. Fine Tune the Student

Use the labelled dataset to train a smaller model.  You can specify a
base model from HuggingFace (e.g. `qwen/Qwen2.5-1.5B-Instruct`) and
tune LoRA or DoRA adapters, optionally enabling QLoRA (requires
`bitsandbytes` and a CUDA GPU):

```bash
python desktop_distill/train_student.py \
    --dataset dataset/labelled.jsonl \
    --base_model qwen/Qwen2.5-1.5B-Instruct \
    --output_dir trained_student \
    --use_dora \
    --qlora_preset ampere-balanced \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5
```

This will save a merged model in `trained_student/` ready for export.

#### Recommended QLoRA presets

Use the `--qlora_preset` flag (or select a preset in the Textual UI) to
load a proven configuration that matches your GPU generation.  Presets
apply tuned LoRA hyperparameters and 4‑bit quantisation settings while
still allowing you to override individual flags if needed.  The
regression suite in `tests/test_train_student_qlora.py` exercises each
matrix entry below.

| Preset | Target GPU families | Quant type | Compute dtype | Double quant | LoRA rank | LoRA α | LoRA dropout | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ampere-balanced` | NVIDIA Ampere/Hopper (RTX30+/A100/A800/H100) | `nf4` | `bfloat16` | ✅ | 64 | 128 | 0.05 | Throughput baseline for bf16-capable cards. |
| `turing-safe` | NVIDIA Turing (RTX20/T4) | `nf4` | `float16` | ✅ | 32 | 64 | 0.10 | fp16 compute keeps compatibility with older GPUs. |
| `ada-memory` | NVIDIA Ada (RTX40 24 GB+) | `fp4` | `bfloat16` | ❌ | 16 | 32 | 0.05 | Minimises memory by targeting attention + MLP projection matrices. |

If a preset requests bf16 compute on hardware that lacks support, the
trainer automatically falls back to fp16 and logs a warning.

### 3. Export to GGUF

Convert the merged model to GGUF and quantize it to the desired
bit‑width (default `q4_k_m`).  Remote HuggingFace models can be
downloaded on demand, and private repositories are supported via
`--hf-token` or the `HF_TOKEN`/`HUGGINGFACEHUB_API_TOKEN` environment
variables:

```bash
python desktop_distill/export_gguf.py \
    --model trained_student \
    --outdir gguf_artifacts \
    --qtype q4_k_m
```

You will find a `.gguf` file in `gguf_artifacts/` along with the
tokenizer and a prompt cache.

### 4. Deploy to the Pi

Set up your Pi once:

```bash
cd rpi4
bash setup_pi.sh
```

Then copy the GGUF file, tokenizer and prompts to your Pi (see
`automation/e2e.sh` or the Makefile for an example).  You can run the
model with:

```bash
bash rpi4/run_decoder.sh ~/models/your-student.q4_k_m.gguf
```

For embeddings use:

```bash
bash rpi4/run_encoder.sh ~/models/your-student.q4_k_m.gguf "Bonjour le monde"
```

### 5. Benchmark

Run on‑device benchmarks to track performance:

```bash
python rpi4/bench/pi_bench.py \
    --model ~/models/your-student.q4_k_m.gguf \
    --iterations 3 \
    --min-tokps 0.25 \
    --csv rpi4/bench/out/bench.csv
```

This will produce a CSV of init/decode/embedding timings and validate
the minimum token rate.

## Interactive Terminal UI

Prefer a guided experience?  Install the UI dependencies and launch the
Textual dashboard to drive each pipeline stage from an intuitive interface:

```bash
pip install -r automation/requirements.txt
python -m automation.ui_app
```

The dashboard exposes dedicated panels for dataset labelling, student training,
GGUF export and Raspberry Pi benchmarking.  Each panel validates your inputs,
shows the exact command that will be executed (via
`automation/pipeline_ops.py`), and streams live logs so you can track progress
in real time.

## Command Builders & Automation

If you need to integrate the pipeline into another tool, import the
dataclasses in `automation/pipeline_ops.py`.  Each builder validates
common footguns (e.g. non-positive batch sizes) and returns the exact
CLI command the scripts expect.  They are safe to pass directly into
`subprocess` and power the Textual UI.

For headless environments use the Makefile or `automation/e2e.sh`.  Both
expect environment variables defined in `.env` (see table below) to know
where to fetch datasets, store checkpoints and deploy artifacts on the
Pi.

## Environment Variables

Copy `.env.example` to `.env` and edit the values to configure remote
deployment:

| Variable      | Description                                           |
|---------------|-------------------------------------------------------|
| `PI_HOST`     | Hostname or IP of your Raspberry Pi                  |
| `PI_USER`     | Username used on the Pi (e.g. `pi`)                  |
| `PI_DIR`      | Remote directory to store models and runtime data    |
| `MODEL_ID`    | HF model id of the teacher (defaults to Dolphin)     |
| `STUDENT_ID`  | HF model id of the base student model                |
| `HF_TOKEN`    | Optional token for private HuggingFace models        |

## Running Tests

Pytest suites in `tests/` cover the GGUF exporter, command builders and
the dataset blueprint utilities under
`dataset/qf_corpus_blueprint/scripts/`.  Run them locally before
contributing:

```bash
pip install -r automation/requirements.txt  # provides pytest & Textual deps
pytest
```

## Dataset CI

Pull requests that modify anything under `dataset/` automatically trigger
the **Dataset Card Validation** workflow.  GitHub Actions runs:

```bash
python dataset/qf_corpus_blueprint/scripts/dataset_card.py \
    --data dataset/qf_corpus_blueprint/examples/sample.jsonl \
    --output build/dataset_card.json \
    --split-name analysis \
    --license CC-BY-4.0 \
    --schema-version 1.0.0 \
    --creation-date 2024-01-01 \
    --validate
```

The command produces a dataset card for the Québec French blueprint and
validates it against `schema/dataset.card.schema.json`.  Each matrix job
uploads its rendered `build/dataset_card.json` as an artifact named
`dataset-card-<python-version>` so other jobs or reviewers can inspect
the metadata produced by every supported runtime.  The workflow runs the
validator on Python 3.10 and 3.11 to match the supported runtime
matrix.

Troubleshooting tips for failed runs:

* **Missing `jsonschema`** – Install the validator dependency locally
  with `pip install jsonschema` before re-running the command.
* **Schema validation errors** – Inspect the workflow logs for the
  failing field path (e.g. `register_distribution/Casual`).  Fix the
  offending dataset records or adjust the blueprint helpers, then rerun
  the script.
* **Non-deterministic card output** – Ensure you supply a stable
  `--creation-date` when generating cards locally; the CI workflow fixes
  it to `2024-01-01` to keep artifacts reproducible.

The Python scripts target CPython 3.10+.  Static type hints are
included throughout the automation helpers to aid IDE integration.

## Maintenance Workflow

The end-of-session sync tooling primarily supports the @codex agent
that maintains this repository. Within that workflow the
`automation/update_and_cleanup.sh` wrapper **must** run at the end of
every @codex session to keep generated guidance, formatting and caches
in sync. Maintainers outside that automation can run the same refresh
script whenever they want to replicate the codex housekeeping
(formatting, agent regeneration and cache cleanup), but it is not a
required step for day-to-day project development.

To invoke the workflow manually, run:

```bash
./automation/update_and_cleanup.sh
```

The wrapper delegates to `automation/session_sync.py`, which coordinates
Prettier formatting, manifest-driven protocol generation, optional
`npm run lint`/`pytest` execution, and filesystem cleanup whenever the
@codex automation runs its end-of-session maintenance. Canonical agent
guidance lives under `automation/agent_sources/` and is compiled into
tracked `AGENTS.md` files via `automation/agents_manifest.json`.
Adjust the manifest instead of editing generated files by hand so
updates remain reproducible across sessions.

Environment toggles (set as `SESSION_SYNC_*` variables) map directly to
the Python orchestrator:
* `SESSION_SYNC_CHECK=1` or `./automation/update_and_cleanup.sh --check` – dry run; exit non-zero when formatting or agent files are stale.
* `SESSION_SYNC_SKIP_FORMATTING=1`, `SESSION_SYNC_SKIP_AGENT_SYNC=1`,
  `SESSION_SYNC_SKIP_CLEANUP=1` – selectively disable steps when
  iterating on specific areas.
* `SESSION_SYNC_RUN_NPM_LINT=1`, `SESSION_SYNC_RUN_PYTEST=1` – fold the
  JavaScript lint check and pytest suite into the sync run.
* `SESSION_SYNC_ENFORCE_MANIFEST=1` – fail when stray `AGENTS.md` files
  are detected outside the manifest (enabled by default via manifest
  settings).

## Notes

* Running large models may require a GPU; the training scripts support
  gradient checkpointing, bf16/fp16 selection and QLoRA to reduce memory
  usage on commodity hardware.
* The provided JSON schemas under `schemas/` help the assistant
  generate valid structured responses for tool calls.
* `AGENTS.md` documents each automation module and is the best starting
  point when extending the pipeline.

This repository is offered under the MIT License.  Contributions and
improvements are welcome!

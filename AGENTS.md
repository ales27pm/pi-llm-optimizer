# Agent Definitions

This document describes the main roles and behaviours of the agents
implemented in this repository.  Agents are modular pieces of code that
perform specific tasks in the pipeline, such as labelling the dataset,
training a student model, exporting to GGUF and running the model on
device.  Understanding these roles makes it easier to extend or
customise the system.

## 🧠 `teacher_label.py`

`desktop_distill/teacher_label.py` is responsible for **labelling the
dataset** using a large teacher model.  It reads JSONL rows containing
user prompts, optional system hints and JSON schemas, constructs the
full prompt, and queries a HuggingFace `pipeline` to generate a
response.  The script writes the original record plus a newly populated
`assistant` field to disk so that student training can begin.

Updated capabilities worth noting when extending or reusing this agent:

* Concatenates `system_hint`, `user` and JSON instructions (when
  `requires_json` is true) using double newlines so the student sees
  the same structure during training.
* Supports streaming multiple prompts per forward pass via
  `--batch_size`.
* Respects existing answers with the `--skip_existing` guard; skipped
  rows are logged with identifiers for quick inspection.
* Allows generation parameters such as `--max_new_tokens`,
  `--temperature`, `--do_sample`, and `--device_map` to be configured
  from the CLI or higher-level automation.

> **Tip:** Maintain separate JSONL files for different speaking styles
> (e.g. `dataset_quebecois_distill.jsonl` for instructional data and
> `dataset_quebecois_conversation.jsonl` for dialogues) and label each
> file independently.  This keeps topic-specific prompts together while
> allowing incremental labelling runs thanks to `--skip_existing`.

## 🎓 `train_student.py`

`desktop_distill/train_student.py` fine-tunes a **student model** on
the labelled dataset.  The script builds LoRA or DoRA adapters,
optionally enables QLoRA (4-bit) training with `bitsandbytes`, and
merges the adapters back into the base model on completion.  It
preprocesses records by concatenating system hints, user prompts and
assistant replies into a single causal language modelling sequence.

Key implementation details:

* Validates the dataset eagerly to guarantee every record contains an
  `assistant` response.
* Configures the tokenizer on-the-fly, guaranteeing a pad token exists
  and using double-newline separators that mirror `teacher_label.py`.
* Enables gradient checkpointing automatically on CUDA devices and
  selects fp16 or bf16 based on GPU support unless QLoRA is active.
  No user configuration is required—the script manages this
  automatically.
* Exposes adapter hyperparameters (`--lora_rank`, `--lora_alpha`,
  `--lora_dropout`, `--target_modules`) alongside scheduling options so
  new variants can be explored without code changes.
* Merges and saves the fully materialised model and tokenizer to the
  output directory, casting to `float16` when trained with QLoRA.

## 🧪 `export_gguf.py`

`desktop_distill/export_gguf.py` converts a fine-tuned HuggingFace
model into the **GGUF** format and quantizes it using the
`llama-quantize` binary from llama.cpp.  Enhancements captured in the
current implementation include:

* Automatic resolution of either local model directories or remote
  HuggingFace repos via `huggingface_hub.snapshot_download`, with
  optional `--revision` and `--hf-token` arguments.
* Detailed command logging and error propagation to aid debugging,
  including a `--preserve-tmp-dir` flag that retains intermediate
  artifacts after failures.
* Smarter naming of the resulting `.gguf` file based on the model id
  and quantization type.
* Comprehensive copying of tokenizer assets (files *and* directories)
  into the output folder so llama.cpp has everything it needs.

When executed inside CI (see `.github/workflows/gguf-build.yml`) the
script produces a ready-to-download quantized model artifact.

## 🧰 Raspberry Pi Scripts

The `rpi4/` directory houses the on-device automation:

* **setup_pi.sh** – installs dependencies, enables zram and builds
  llama.cpp with OpenBLAS support on ARM64.
* **get_model.sh** – downloads a pre-quantized TinyLlama or Qwen model
  as a bootstrap option.
* **run_decoder.sh** – launches interactive inference with configurable
  context window, batching, KV cache, prompt cache and grammar inputs.
* **run_encoder.sh** – computes embeddings for downstream search or
  retrieval tasks.
* **bench/pi_bench.py** – orchestrates llama.cpp benchmarks and
  validates minimum token throughput using
  `rpi4/bench/throughput_regressor.py`.

Supporting modules such as `rpi4/bench/benchmark_csv.py` format timing
data and persist CSV outputs for regressions tracking.

## 🧭 Pipeline Helpers & UI

`automation/pipeline_ops.py` exposes dataclass-backed builders that
assemble the exact CLI invocations for every stage (teacher labelling,
student training, GGUF export and benchmarking).  These helpers
validate common mistakes (e.g. non-positive batch sizes) and are reused
by the Textual dashboard in `automation/ui_app.py`.  The UI streams
logs live, highlights required parameters, and is styled via
`automation/ui_app.tcss`.  The training panel mirrors the CLI flags for
LoRA/DoRA and QLoRA toggles while letting the Python script continue to
manage gradient checkpointing automatically based on device support.

For headless automation, `automation/e2e.sh` sequences the entire flow
using environment variables declared in `.env`.  The `Makefile` mirrors
the same primitives and is a convenient entry point for CI.

## 🗂️ Dataset Blueprint

The `dataset/qf_corpus_blueprint/scripts/dataset_card.py` module builds
rich dataset cards from JSONL corpora.  It computes register and dialect
distributions, aggregates provenance metadata (including tool versions)
and exposes a CLI for reproducible documentation.  Companion utilities
in `dataset/qf_corpus_blueprint/scripts/jsonl_utils.py` handle JSONL
parsing so both the CLI and tests can share the same loaders.

## ✅ Tests

The `tests/` directory contains pytest suites that exercise dataset
helpers, GGUF export tooling and pipeline command builders.  Use
`pytest` to validate contributions and ensure CLI contracts remain
stable.

Refer to `README.md` for end-to-end usage examples and environment
setup guidance.
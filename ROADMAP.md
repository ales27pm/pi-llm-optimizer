# Project Roadmap

The roadmap captures the team priorities for Pi‑LLM Optimizer as we
iterate on the end‑to‑end distillation, quantization and deployment
experience.  Items are grouped by timeframe and focus on work that
extends the capabilities already documented in `README.md` and
`AGENTS.md`.

## ✅ Recently Landed

- **Dataset Blueprint Tooling** – Added the dataset card generator and
  JSONL utilities so every corpus ships with reproducible documentation.
- **Automation Command Builders** – Unified CLI assembly for labelling,
  training, export and benchmarking through `automation/pipeline_ops.py`
  and the Textual dashboard.
- **UI Parity with CLI** – Removed stale gradient‑checkpoint flags and
  ensured the Textual panels mirror the authoritative script options.
- **Batch Labelling Improvements** – Teacher labelling now supports
  batched prompts, skip‑existing safeguards and structured JSON
  instructions.

## ⏱️ Near Term (0‑2 Sprints)

- **Dataset Blueprint CI Integration** – Run
  `dataset_card.py --validate` in CI to block regressions and publish
  dataset card artifacts alongside training runs.
- **Expanded QLoRA Coverage** – Document recommended quantization
  configs, add regression tests for 4‑bit adapters and expose presets in
  the UI and command builders.
- **Remote Model Export UX** – Add progress reporting and richer error
  diagnostics when resolving remote HuggingFace repos during GGUF
  export.
- **Benchmark Dashboards** – Emit structured metrics from
  `rpi4/bench/pi_bench.py` and surface summaries within the Textual UI.

## 🔭 Mid Term (Quarter)

- **Pipeline Profiles** – Ship reusable YAML profiles that capture common
  end‑to‑end flows (e.g. TinyLlama tutoring, Qwen bilingual assistant)
  and hydrate the command builders automatically.
- **Multi‑Node Training Experiments** – Investigate DeepSpeed or FSDP for
  large teacher fine‑tuning while keeping the student path lightweight.
- **Enhanced Dataset Tooling** – Add schema validation helpers for
  tool‑augmented conversations and surface dialect coverage warnings in
  the Textual UI.
- **Pi Runtime Packaging** – Produce a Debian package that installs the
  llama.cpp runtime, helper scripts and systemd units for background
  decoding services.

## 🧭 Long Term (6+ Months)

- **Model Zoo Publishing** – Host a curated catalog of distilled models
  with reproducible dataset cards, training configs and GGUF exports.
- **Edge Accelerator Support** – Prototype Metal / CoreML export paths
  for Apple Silicon along with Vulkan backends for other ARM devices.
- **Data Privacy Tooling** – Integrate automated redaction checks and
  audit logging for sensitive training data.
- **Community Contribution Path** – Document contribution templates and
  governance for external dataset and model submissions.

## 📌 How to Contribute

- Track roadmap issues under the `#roadmap` GitHub label.
- Propose changes via pull requests that link to the relevant roadmap
  item.
- Keep documentation (`README.md`, `AGENTS.md`, dataset cards) updated as
  features land so the roadmap stays an accurate forward‑looking plan.

# Project Roadmap

The roadmap captures the team priorities for Pi‑LLM Optimizer as we
iterate on the end‑to‑end distillation, quantization and deployment
experience. Items are grouped by timeframe and focus on work that
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
  - [ ] Create a dedicated GitHub Actions workflow that invokes
        `dataset/qf_corpus_blueprint/scripts/dataset_card.py --validate` on
        every pull request touching the dataset.
  - [ ] Archive the generated dataset card as a build artifact using the
        workflow so downstream jobs can inspect the rendered card.
  - [ ] Add a "Dataset CI" subsection to `README.md` describing the new
        workflow and troubleshooting steps for failed validations.
- **Expanded QLoRA Coverage** – Document recommended quantization
  configs, add regression tests for 4-bit adapters and expose presets in
  the UI and command builders.
  - [ ] Capture a matrix of tested quantization settings in
        `desktop_distill/train_student.py` docstrings and mirror it in
        `README.md`.
  - [ ] Extend the pytest suite with fixtures that exercise QLoRA
        adapters end-to-end using a small synthetic dataset.
  - [ ] Add preset dropdowns to `automation/ui_app.py` so operators can
        select proven QLoRA configurations without manual flag editing.
- **Remote Model Export UX** – Add progress reporting and richer error
  diagnostics when resolving remote HuggingFace repos during GGUF
  export.
  - [ ] Integrate download progress callbacks into
        `desktop_distill/export_gguf.py` and surface them in CLI logs.
  - [ ] Implement structured exception handling that outputs actionable
        remediation hints when remote resolution fails.
  - [ ] Write integration tests under `tests/` that mock failed
        downloads and assert user-friendly messages are emitted.
- **Benchmark Dashboards** – Emit structured metrics from
  `rpi4/bench/pi_bench.py` and surface summaries within the Textual UI.
  - [ ] Refactor `rpi4/bench/pi_bench.py` to emit JSON summaries alongside
        CSV output for consumption by dashboards.
  - [ ] Update `automation/ui_app.py` to visualize recent benchmark runs
        in a dedicated panel with trend lines.
  - [ ] Document how to launch the dashboard mode inside
        `automation/ui_app.tcss`/`README.md`, including expected data
        refresh cadence.

## 🔭 Mid Term (Quarter)

- **Pipeline Profiles** – Ship reusable YAML profiles that capture common
  end‑to‑end flows (e.g. TinyLlama tutoring, Qwen bilingual assistant)
  and hydrate the command builders automatically.
  - [ ] Define a YAML schema (with JSON Schema validation) covering
        teacher/student ids, dataset sources and automation presets.
  - [ ] Implement profile loading in `automation/pipeline_ops.py`,
        falling back to existing CLI arguments when profiles are absent.
  - [ ] Add profile selection controls to the Textual UI and document
        sample profiles under `automation/profiles/`.
- **Multi‑Node Training Experiments** – Investigate DeepSpeed or FSDP for
  large teacher fine‑tuning while keeping the student path lightweight.
  - [ ] Prototype a DeepSpeed configuration for the teacher model using
        a constrained dataset to validate scaling assumptions.
  - [ ] Benchmark FSDP against LoRA-only baselines and summarize token
        throughput in `CODEBASE_ANALYSIS.md`.
  - [ ] Document hardware prerequisites and trade-offs for each strategy
        in `README.md`'s advanced training section.
- **Enhanced Dataset Tooling** – Add schema validation helpers for
  tool‑augmented conversations and surface dialect coverage warnings in
  the Textual UI.
  - [ ] Extend `dataset/qf_corpus_blueprint/scripts/jsonl_utils.py` with
        schema validation hooks that can be shared by CLI and tests.
  - [ ] Render dialect coverage summaries in the Textual UI with warning
        banners when corpora fall below thresholds.
  - [ ] Author documentation for new validation rules, including
        remediation suggestions, in the dataset blueprint README.
- **Pi Runtime Packaging** – Produce a Debian package that installs the
  llama.cpp runtime, helper scripts and systemd units for background
  decoding services.
  - [ ] Create `rpi4/package/` with Debian packaging metadata and
        reproducible build scripts.
  - [ ] Automate package builds via GitHub Actions, attaching `.deb`
        artifacts to releases.
  - [ ] Provide installation and service management instructions in the
        Raspberry Pi setup guide.

## 🧭 Long Term (6+ Months)

- **Model Zoo Publishing** – Host a curated catalog of distilled models
  with reproducible dataset cards, training configs and GGUF exports.
  - [ ] Stand up a static site (e.g. GitHub Pages) fed from a structured
        `model_catalog.json` maintained in-repo.
  - [ ] Automate GGUF upload and metadata publishing as part of release
        workflows.
  - [ ] Provide dataset card templates and release checklists for new
        models within the documentation site.
- **Edge Accelerator Support** – Prototype Metal / CoreML export paths
  for Apple Silicon along with Vulkan backends for other ARM devices.
  - [ ] Evaluate llama.cpp Metal builds and document required flags for
        macOS/iOS targets.
  - [ ] Investigate Vulkan compute support on Raspberry Pi-class devices
        and capture findings in `CODEBASE_ANALYSIS.md`.
  - [ ] Add abstraction layers in the runtime scripts so accelerator
        selection is configurable per device profile.
- **Data Privacy Tooling** – Integrate automated redaction checks and
  audit logging for sensitive training data.
  - [ ] Incorporate PII redaction passes into the dataset labelling
        pipeline with opt-in CLI flags.
  - [ ] Store redaction reports alongside dataset cards for auditing.
  - [ ] Establish governance documentation describing privacy review
        expectations for contributors.
- **Community Contribution Path** – Document contribution templates and
  governance for external dataset and model submissions.
  - [ ] Draft contribution templates for datasets, training recipes and
        runtime scripts under `.github/ISSUE_TEMPLATE/`.
  - [ ] Define review checklists and escalation paths in `CONTRIBUTING.md`.
  - [ ] Host quarterly community sync notes summarizing roadmap changes
        and published contributions.

## 📌 How to Contribute

- Track roadmap issues under the `#roadmap` GitHub label.
- Propose changes via pull requests that link to the relevant roadmap
  item.
- Keep documentation (`README.md`, `AGENTS.md`, dataset cards) updated as
  features land so the roadmap stays an accurate forward‑looking plan.

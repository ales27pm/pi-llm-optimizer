# Pi-LLM Optimizer Codebase Analysis

This document summarizes the structure, responsibilities, and key flows within the Pi-LLM Optimizer repository.

## Top-Level Layout

- `README.md` – High-level overview and quickstart for dataset creation, training, GGUF export, Pi deployment, and benchmarking. 【F:README.md†L1-L120】【F:README.md†L120-L207】
- `Makefile` / `automation/e2e.sh` – Orchestrate teacher labelling, student training, GGUF export, deployment, and on-device benchmarking.
- `desktop_distill/` – Desktop-side scripts for teacher data labelling, student model fine-tuning, and GGUF export. 【F:desktop_distill/teacher_label.py†L1-L126】【F:desktop_distill/train_student.py†L1-L205】【F:desktop_distill/export_gguf.py†L1-L124】
- `rpi4/` – Raspberry Pi runtime scripts, including setup automation, runtime launchers, and benchmarking utilities. 【F:rpi4/run_decoder.sh†L1-L33】【F:rpi4/bench/pi_bench.py†L1-L96】
- `dataset/` – Example datasets and documentation for Québécois prompts, conversations, and schema-driven tool use.
- `schemas/` – JSON schemas referenced when `requires_json` is true in dataset records.

## Desktop Distillation Pipeline

1. **Teacher Labelling (`desktop_distill/teacher_label.py`)**
   - Loads JSONL records with `user`, optional `system_hint`, `requires_json`, and `tool_schema`, builds prompts, and generates completions with a HF pipeline. 【F:desktop_distill/teacher_label.py†L13-L110】
   - Writes augmented JSONL with an `assistant` field per example. 【F:desktop_distill/teacher_label.py†L112-L124】

2. **Student Training (`desktop_distill/train_student.py`)**
   - Loads labelled data, verifies `assistant` presence, tokenizes concatenated system/user/assistant text, and prepares for causal LM training. 【F:desktop_distill/train_student.py†L42-L126】
   - Supports LoRA/DoRA adapters, gradient checkpointing, and optional QLoRA via PEFT utilities, merging adapters back into base model after training. 【F:desktop_distill/train_student.py†L128-L210】

3. **GGUF Export (`desktop_distill/export_gguf.py`)**
   - Invokes llama.cpp converters to build f16 GGUF, quantizes to configured type, and copies tokenizer assets to the output directory. 【F:desktop_distill/export_gguf.py†L25-L108】

## Raspberry Pi Runtime

- `setup_pi.sh` configures dependencies, builds llama.cpp with OpenBLAS, and prepares the Pi environment.
- `run_decoder.sh` launches interactive inference with configurable context, batch, thread, KV cache, prompt cache, grammar, and system prompt inputs. 【F:rpi4/run_decoder.sh†L1-L33】
- `run_encoder.sh` mirrors decoder script for embedding extraction.
- `rpi4/bench/pi_bench.py` coordinates llama.cpp CLI invocations to measure init latency, decode throughput, and embedding latency, writing CSV results and enforcing throughput thresholds via `throughput_regressor.validate`. 【F:rpi4/bench/pi_bench.py†L1-L96】【F:rpi4/bench/throughput_regressor.py†L1-L48】
- `rpi4/bench/benchmark_csv.py` formats benchmark rows and writes CSV files with ISO8601 timestamps. 【F:rpi4/bench/benchmark_csv.py†L1-L40】

## Automation and Deployment

- Make targets and `automation/e2e.sh` compose the full workflow from labelling through Pi benchmarking, using environment variables defined in `.env`. 【F:README.md†L129-L167】
- `.github/workflows/gguf-build.yml` (mentioned in README) automates GGUF export on tagged releases.

## Dataset and Schema Assets

- Example Québécois dataset files under `dataset/` illustrate both distillation prompts and conversational data for natural language coverage. 【F:dataset/DATASET_README.md†L1-L120】
- JSON schemas in `schemas/` define tool outputs when `requires_json` is flagged, enabling structured responses during teacher labelling and student training.

## Key Takeaways

- The repository delivers an end-to-end flow: prompt labelling ➜ student fine-tuning with adapter support ➜ GGUF export ➜ Pi deployment and benchmarking.
- Desktop scripts rely on HuggingFace Transformers, Datasets, and PEFT for model handling, while Pi scripts depend on llama.cpp binaries.
- Dataset design supports system hints, JSON tool schemas, and bilingual prompts to align the student model with desired behaviour.

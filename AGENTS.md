# Codex Agents & Playbooks

This repo uses a small set of *deterministic* agents you can run locally (no cloud).

## Agents
- **architect**: validates goals, chooses teacher/student, distill plan.
- **data-smith**: builds synthetic QC dataset, enforces JSON schemas.
- **trainer**: executes QLoRA/DoRA runs, logs metrics.
- **exporter**: merges adapters, converts HF→GGUF, quantizes Q4_K_M.
- **pi-runner**: configures llama.cpp on Pi, runs benches, ensures JSON validity.

## Playbooks
1. plan_distill.md – inputs, teacher, student, datasets, KPIs.
2. build_dataset.md – how to expand & label with Dolphin.
3. train_student.md – training flags & schedules.
4. export_student.md – GGUF export, quant sweep.
5. run_on_pi.md – runtime flags, grammar decoding, prompt-cache.

# pi-llm-optimizer

On-device LLM distillation and optimization suite for Raspberry Pi 4 (4 GB).

**Core principles**
- 100% on-device inference (no remote streaming).
- Teacher: Dolphin 8B on Ubuntu desktop; Student: 1.1–1.5B for Pi.
- LLM2Vec: one model for **encode** (embeddings) and **decode** (chat).
- Local KB with SQLite + pgvector; JSON/GBNF tool calls.

## Quickstart
```bash
# Desktop (train + export)
python desktop_distill/train_student.py --help
python desktop_distill/export_gguf.py --help

# Raspberry Pi 4 (run)
bash rpi4/setup_pi.sh
bash rpi4/get_model.sh
bash rpi4/run_decoder.sh $HOME/models/model-q4_k_m.gguf
```

## Layout
- `desktop_distill/` – distillation (teacher→student), merge, GGUF export.
- `rpi4/` – Pi build/run scripts, benches, sweeps.
- `dataset/` – Québecois + reasoning prompts (teacher labels to fill).
- `schemas/` – JSON Schemas for strict tool responses.
- `AGENTS.md` – Codex agents (builder, tester, packager) and playbooks.

## License
MIT

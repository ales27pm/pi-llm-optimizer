#!/usr/bin/env bash
set -euo pipefail
LLAMA="$HOME/llama.cpp/build/bin"
MODEL="${1:-$HOME/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
CTX="${CTX:-1024}"
BATCH="${BATCH:-64}"
THREADS="${THREADS:-$(nproc)}"
KVTYPE="${KVTYPE:-q8_0}"
GRAMMAR="${GRAMMAR:-$PWD/schemas/email.schema.json}"
SYS_PROMPT_PATH="${SYS_PROMPT_PATH:-$PWD/dataset/system_minimal.txt}"
PCACHE="${PCACHE:-$HOME/.cache/llm_pc.bin}"

"$LLAMA/llama-cli" -m "$MODEL" -c "$CTX" -b "$BATCH" -t "$THREADS" --kv-type "$KVTYPE" --prompt-cache "$PCACHE" --interactive

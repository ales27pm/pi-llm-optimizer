#!/usr/bin/env bash
# Run a GGUF model interactively on the Raspberry Pi.  You can
# optionally override environment variables to adjust context length,
# batch size, number of threads, KV cache quantisation, grammar file
# and system prompt.

set -euo pipefail

LLAMA_DIR="$HOME/llama.cpp/build/bin"
MODEL=${1:-$HOME/models/$(ls $HOME/models | grep \.gguf$ | head -n1)}

CTX=${CTX:-1024}
BATCH=${BATCH:-64}
THREADS=${THREADS:-$(nproc)}
KVTYPE=${KVTYPE:-q8_0}
GRAMMAR=${GRAMMAR:-}
PROMPT_CACHE=${PROMPT_CACHE:-$HOME/.cache/llm_prompt_cache.bin}
SYS_PROMPT_FILE=${SYS_PROMPT_FILE:-}

CMD=("$LLAMA_DIR/llama-cli" -m "$MODEL" -c "$CTX" -b "$BATCH" -t "$THREADS" --kv-type "$KVTYPE" --prompt-cache "$PROMPT_CACHE" --interactive)
if [ -n "$GRAMMAR" ]; then
  CMD+=(--grammar "$GRAMMAR")
fi
if [ -n "$SYS_PROMPT_FILE" ]; then
  CMD+=(--file "$SYS_PROMPT_FILE")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
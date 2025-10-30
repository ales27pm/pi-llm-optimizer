#!/usr/bin/env bash
# Generate sentence embeddings from a GGUF model.  The first argument
# is the model path (default: first .gguf in $HOME/models).  The
# optional second argument is the text to embed; if omitted you will
# be prompted interactively.

set -euo pipefail

LLAMA_DIR="$HOME/llama.cpp/build/bin"
MODEL=${1:-$HOME/models/$(ls $HOME/models | grep \.gguf$ | head -n1)}
TEXT=${2:-}

if [ -z "$TEXT" ]; then
  read -rp "Enter text to embed: " TEXT
fi

"$LLAMA_DIR/llama-cli" -m "$MODEL" --embedding -p "$TEXT" -n 0
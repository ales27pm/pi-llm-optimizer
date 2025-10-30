#!/usr/bin/env bash
# Download a small GGUF model suitable for the Raspberry Pi.  You can
# choose between TinyLlama 1.1 B Chat and Qwen2.5 1.5 B Instruct.  The
# downloaded model will be saved in the directory specified as the
# first argument (default: $HOME/models).

set -euo pipefail

MODEL_DIR=${1:-$HOME/models}
mkdir -p "$MODEL_DIR"

echo "Select model to download:"
echo " 1) TinyLlama-1.1B-Chat (Q4_K_M)"
echo " 2) Qwen2.5-1.5B-Instruct (Q4_K_M)"
read -rp "Choice [1/2]: " CH

case "$CH" in
  1)
    URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
    OUT="$MODEL_DIR/tinyllama-1.1b-chat-q4_k_m.gguf"
    ;;
  2)
    URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    OUT="$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ;;
  *)
    echo "Invalid selection"; exit 1;;
esac

echo "Downloading $URL to $OUT"
curl -L "$URL" -o "$OUT"
echo "[OK] Saved to $OUT"
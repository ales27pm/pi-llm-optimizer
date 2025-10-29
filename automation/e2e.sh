#!/usr/bin/env bash
set -euo pipefail
# End-to-end pipeline: distill → export GGUF → deploy to Pi → run + bench.
# Requires env vars: PI_HOST, PI_USER, PI_DIR (e.g. /home/pi/models), MODEL_ID, STUDENT_ID
# Optional: DATASET (path to JSONL), HF_TOKEN

: "${PI_HOST:?set PI_HOST}"
: "${PI_USER:?set PI_USER}"
: "${PI_DIR:?set PI_DIR}"
: "${MODEL_ID:=cognitivecomputations/Dolphin3.0-Llama3.1-8B}"
: "${STUDENT_ID:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${DATASET:=dataset/dataset_quebecois_distill.jsonl}"
: "${HF_TOKEN:=}"

PY=python3
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# 1) Train student with teacher labels (distillation)
$PY "$ROOT/desktop_distill/train_student.py" \
  --teacher "$MODEL_ID" \
  --student "$STUDENT_ID" \
  --dataset "$ROOT/$DATASET" \
  --out "$ROOT/out/student" \
  ${HF_TOKEN:+--hf-token "$HF_TOKEN"}

# 2) Export to GGUF + quantize
$PY "$ROOT/desktop_distill/export_gguf.py" \
  --model "$ROOT/out/student" \
  --out "$ROOT/out/gguf" \
  --qtype q4_k_m \
  ${HF_TOKEN:+--hf-token "$HF_TOKEN"}
GGUF=$(ls -1 "$ROOT/out/gguf"/*.gguf | head -n1)

# 3) Copy to Pi
rsync -av --progress "$GGUF" "$PI_USER@$PI_HOST:$PI_DIR/model-q4_k_m.gguf"

# 4) Remote setup + run + bench
ssh -o BatchMode=yes "$PI_USER@$PI_HOST" bash -lc "'
  mkdir -p $PI_DIR && cd $PI_DIR
  if [ ! -d "~/llama.cpp" ]; then
    echo [*] First-time setup; building llama.cpp...
    bash -lc "$(cat ~/pi-llm-optimizer/rpi4/setup_pi.sh)"
  fi
  echo [*] Running decode sanity...
  "~/llama.cpp/build/bin/llama-cli" -m $PI_DIR/model-q4_k_m.gguf -n 16 -p "Bonjour!" || true
'"

# 5) Local bench helper
$PY "$ROOT/rpi4/bench/pi_bench.py" --model "$GGUF" --iterations 3 --csv "$ROOT/rpi4/bench/out/bench.csv"

echo "[OK] E2E done. GGUF=$GGUF"

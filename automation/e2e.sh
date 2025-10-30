#!/usr/bin/env bash
# End‑to‑end automation script.
#
# This script orchestrates the entire workflow from dataset labelling
# through training, GGUF export and deployment to a Raspberry Pi.  It
# assumes that environment variables are defined in a `.env` file in
# the repository root (see `.env.example` for a template).

set -euo pipefail

# Load environment variables
if [[ -f .env ]]; then
  # shellcheck source=/dev/null
  source .env
else
  echo "Error: .env file not found.  Copy .env.example to .env and edit the values." >&2
  exit 1
fi

echo "[1/5] Labelling dataset using teacher model ${MODEL_ID}"
python desktop_distill/teacher_label.py \
  --model "${MODEL_ID}" \
  --input dataset/dataset_quebecois_distill.jsonl \
  --output dataset/labelled.jsonl

echo "[2/5] Training student model ${STUDENT_ID}"
python desktop_distill/train_student.py \
  --dataset dataset/labelled.jsonl \
  --base_model "${STUDENT_ID}" \
  --output_dir trained_student \
  --use_dora \
  --qlora \
  --num_epochs 1 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5

echo "[3/5] Exporting to GGUF with quantization (q4_k_m)"
python desktop_distill/export_gguf.py \
  --model trained_student \
  --outdir gguf_artifacts \
  --qtype q4_k_m

echo "[4/5] Copying GGUF artifacts to ${PI_USER}@${PI_HOST}:${PI_DIR}"
ssh "${PI_USER}@${PI_HOST}" "mkdir -p \"${PI_DIR}\""
scp gguf_artifacts/*.gguf "${PI_USER}@${PI_HOST}:${PI_DIR}/"
scp gguf_artifacts/tokenizer* "${PI_USER}@${PI_HOST}:${PI_DIR}/"

echo "[5/5] Running benchmark on the Pi"
ssh "${PI_USER}@${PI_HOST}" "bash -l -c 'cd ${PI_DIR} && ~/llama.cpp/build/bin/llama-cli -m $(ls *.gguf | head -n1) -c 1024 -b 64 -t 4 --prompt-cache prompt_cache.bin -p "Test" -n 10'"

echo "All steps completed successfully.  You can now run the model interactively on your Pi."
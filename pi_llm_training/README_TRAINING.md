# Pi LLM Optimizer — Training Toolkit

This folder contains everything you need to label your dataset with a **Dolphin** teacher and train a **TinyLlama** student using **QLoRA + DoRA**, then export to GGUF for your Raspberry Pi 4.

## 0) Create a virtual environment and install dependencies
```bash
cd pi_llm_training
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Label the dataset with the teacher
```bash
cd scripts
python label_with_teacher.py --in-file ../train.jsonl --out-file ../train_labeled.jsonl \
  --teacher-model cognitivecomputations/Dolphin3.0-Llama3.1-8B --load-in-4bit \
  --max-new-tokens 256 --batch-size 2

python label_with_teacher.py --in-file ../val.jsonl --out-file ../val_labeled.jsonl \
  --teacher-model cognitivecomputations/Dolphin3.0-Llama3.1-8B --load-in-4bit \
  --max-new-tokens 256 --batch-size 2
```

## 2) (Optional) Flatten to plain text prompts
```bash
python prepare_dataset.py --in-file ../train_labeled.jsonl --out-file ../train_text.jsonl --drop-unlabeled
python prepare_dataset.py --in-file ../val_labeled.jsonl   --out-file ../val_text.jsonl   --drop-unlabeled
```

## 3) Train using QLoRA + DoRA
```bash
python train_qlora.py --config ../configs/qlora_tinyllama.yaml
```

## 4) Merge adapters and export to GGUF
```bash
# Merge LoRA weights
python - <<'PYCODE'
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_dir = Path("runs/tinyllama_qlora_dora")

base_model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu")
peft_model = PeftModel.from_pretrained(base_model, model_dir)
merged = peft_model.merge_and_unload()
merged.save_pretrained(model_dir / "merged")
PYCODE

# Export to GGUF using llama.cpp tooling
/path/to/llama.cpp/convert-hf-to-gguf.py --outfile runs/tinyllama_qlora_dora/merged.gguf runs/tinyllama_qlora_dora/merged
/path/to/llama.cpp/quantize runs/tinyllama_qlora_dora/merged.gguf runs/tinyllama_qlora_dora/tinyllama.Q4_K_M.gguf Q4_K_M
```

## Notes
- The teacher can run in 4-bit NF4 mode to fit on an 8 GB GPU (or CPU fallback).
- The TinyLlama student (1.1B params) keeps inference light enough for embedded devices.
- LoRA/DoRA parameters are tuned for efficiency (`r=64`, `alpha=16`).
- Export with `Q4_K_M` for a good balance of latency and quality on Raspberry Pi 4.
- Replace the dataset paths with your curated Québec French training/validation files.

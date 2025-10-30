# Pi‑LLM Optimizer

This repository contains everything you need to train, quantize and deploy a
compact large‑language model on a Raspberry Pi 4.  It was built as a
companion to the `CoreML‑DOLPHIN3.0` project and demonstrates how to
distill a larger *teacher* model (e.g. the Dolphin 8 B model) into a much
smaller *student* model that fits into 4 GB of RAM and runs locally on
ARM64 hardware.  The training pipeline runs on your desktop (Ubuntu/x86_64) and
produces a GGUF artifact that is then transferred to the Pi for inference
using [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

* **Knowledge Distillation** – Use a powerful teacher model to generate
  labelled examples from a lightweight dataset.  The `desktop_distill/teacher_label.py`
  script will populate the `assistant` field of a JSONL file using your
  teacher model (e.g. Dolphin) and optional JSON schemas.
* **Student Training** – Fine tune a small base model (e.g. TinyLlama
  1.1 B or Qwen 2.5 B) with LoRA or DoRA adapters, gradient checkpointing and
  optional QLoRA (k‑bit) training.  The script
  `desktop_distill/train_student.py` supports mixed precision and merging
  adapters back into the base model.
* **GGUF Export** – Convert your fine‑tuned HuggingFace model to the
  GGUF format and quantize it (e.g. `q4_k_m`) for maximum runtime
  efficiency.  The `desktop_distill/export_gguf.py` script uses
  `convert_hf_to_gguf.py` and `llama‑quantize` from llama.cpp to perform
  conversion and quantization.
* **Automated Workflow** – A Makefile and `automation/e2e.sh` script
  orchestrate the entire process: labelling, training, export,
  deployment and on‑device benchmarking.  A GitHub Action
  (`.github/workflows/gguf-build.yml`) demonstrates how to export a GGUF
  artifact whenever you push a version tag.
* **Raspberry Pi Runtime** – Scripts in the `rpi4/` directory build
  llama.cpp on ARM64 (with OpenBLAS), download a pre‑built model if
  desired, run the GGUF model interactively, or fetch embeddings.  Bench
  scripts measure throughput on the Pi to guard against performance
  regressions.

## Quickstart

### 1. Prepare the Dataset

Create a dataset of user prompts and metadata in JSONL format under
`dataset/`.  Each line must contain at least a `user` field, and may
optionally include:

* `system_hint` – system‑level instructions or context
* `requires_json` – whether the answer should be valid JSON
* `tool_schema` – a JSON schema string to constrain the output when
  `requires_json` is true

Run the labeller on your desktop:

```bash
python desktop_distill/teacher_label.py \
    --model openai/dolphin  \
    --input dataset/dataset_quebecois_distill.jsonl \
    --output dataset/labelled.jsonl
```

The output file will contain the teacher’s response in the `assistant`
field for each record.

### 2. Fine Tune the Student

Use the labelled dataset to train a smaller model.  You can specify a
base model from HuggingFace (e.g. `qwen/Qwen2.5-1.5B-Instruct`) and
tune LoRA or DoRA adapters:

```bash
python desktop_distill/train_student.py \
    --dataset dataset/labelled.jsonl \
    --base_model qwen/Qwen2.5-1.5B-Instruct \
    --output_dir trained_student \
    --use_dora \
    --qlora \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5
```

This will save a merged model in `trained_student/` ready for export.

### 3. Export to GGUF

Convert the merged model to GGUF and quantize it to the desired
bit‑width (default `q4_k_m`):

```bash
python desktop_distill/export_gguf.py \
    --model trained_student \
    --outdir gguf_artifacts \
    --qtype q4_k_m
```

You will find a `.gguf` file in `gguf_artifacts/` along with the
tokenizer and a prompt cache.

### 4. Deploy to the Pi

Set up your Pi once:

```bash
cd rpi4
bash setup_pi.sh
```

Then copy the GGUF file, tokenizer and prompts to your Pi (see
`automation/e2e.sh` or the Makefile for an example).  You can run the
model with:

```bash
bash rpi4/run_decoder.sh ~/models/your-student.q4_k_m.gguf
```

For embeddings use:

```bash
bash rpi4/run_encoder.sh ~/models/your-student.q4_k_m.gguf "Bonjour le monde"
```

### 5. Benchmark

Run on‑device benchmarks to track performance:

```bash
python rpi4/bench/pi_bench.py \
    --model ~/models/your-student.q4_k_m.gguf \
    --iterations 3 \
    --min-tokps 0.25 \
    --csv rpi4/bench/out/bench.csv
```

This will produce a CSV of init/decode/embedding timings and validate
the minimum token rate.

## Interactive Terminal UI

Prefer a guided experience?  Install the UI dependencies and launch the
Textual dashboard to drive each pipeline stage from an intuitive interface:

```bash
pip install -r automation/requirements.txt
python -m automation.ui_app
```

The dashboard exposes dedicated panels for dataset labelling, student training,
GGUF export and Raspberry Pi benchmarking.  Each panel validates your inputs,
shows the exact command that will be executed, and streams live logs so you can
track progress in real time.

## Environment Variables

Copy `.env.example` to `.env` and edit the values to configure remote
deployment:

| Variable      | Description                                           |
|---------------|-------------------------------------------------------|
| `PI_HOST`     | Hostname or IP of your Raspberry Pi                  |
| `PI_USER`     | Username used on the Pi (e.g. `pi`)                  |
| `PI_DIR`      | Remote directory to store models and runtime data     |
| `MODEL_ID`    | HF model id of the teacher (defaults to Dolphin)      |
| `STUDENT_ID`  | HF model id of the base student model                |
| `HF_TOKEN`    | Optional token for private HuggingFace models        |

## Notes

* Running large models may require a GPU; the training scripts support
  gradient checkpointing and k‑bit training to reduce memory usage.
* The provided JSON schemas under `schemas/` help the assistant
  generate valid structured responses for tool calls.
* See `AGENTS.md` for an overview of agent roles and best practices
  when designing an on‑device assistant.

This repository is offered under the MIT License.  Contributions and
improvements are welcome!

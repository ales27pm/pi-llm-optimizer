# Agent Definitions

This document describes the main roles and behaviours of the agents
implemented in this repository.  Agents are modular pieces of code that
perform specific tasks in the pipeline, such as labelling the dataset,
training a student model, exporting to GGUF and running the model on
device.  Understanding these roles makes it easier to extend or
customise the system.

## 🧠 `teacher_label.py`

This script is responsible for **labelling the dataset** using a
large teacher model.  It reads a JSONL file with user prompts,
optional system hints and JSON schemas, constructs a textual prompt
for each record and queries the teacher model to generate a response.
The resulting file contains the original record with an additional
`assistant` field holding the teacher’s reply.  You should run this
once for each dataset you wish to use for distillation.

Key behaviours:

* Concatenates `system_hint` and `user` fields to form the prompt.
* Appends a JSON instruction and schema if `requires_json` is true.
* Uses HuggingFace’s `pipeline` API for generation with configurable
  `max_new_tokens`, temperature and sampling.
* Strips the prompt from the output so only the generated text is
  stored in `assistant`.

> **Note:** If you maintain multiple dataset files (for example,
> `dataset_quebecois_distill.jsonl` and the more conversational
> `dataset_quebecois_conversation.jsonl`), run the labelling script
> separately on each.  The conversation file contains everyday
> Québécois dialogues that emphasise natural question‑asking and
> colloquial language.  Label both files to create a comprehensive
> training corpus.

## 🎓 `train_student.py`

After labelling, the next agent fine tunes a **student model** on the
teacher‑generated dataset.  The script supports LoRA and DoRA
adapters, gradient checkpointing and optional k‑bit (QLoRA) training to
reduce memory usage.  Once training completes the adapters are merged
back into the base model and the full model is saved to disk.

Important features:

* Loads a base model and tokenizer from HuggingFace.
* Preprocesses the dataset by concatenating system, user and assistant
  text separated by double newlines.
* Builds LoRA/DoRA adapters with configurable rank, alpha and dropout.
* Enables gradient checkpointing and QLoRA when requested.
* Uses `transformers.Trainer` with a cosine scheduler by default.

You can experiment with different base models and training settings to
strike a balance between quality and size.

## 🧪 `export_gguf.py`

This agent converts a fine‑tuned HuggingFace model into the **GGUF**
format and quantizes it to the desired bit‑width.  It wraps the
`convert_hf_to_gguf.py` script from `llama.cpp` and the
`llama‑quantize` binary, producing a model that can be loaded by
`llama.cpp` on the Raspberry Pi.  When run in a GitHub Action (see
`.github/workflows/gguf-build.yml`), it automatically exports and
uploads the GGUF artifact whenever a new git tag is pushed.

## 🧰 Raspberry Pi Scripts

Under `rpi4/` you will find scripts that perform specific tasks on
the Raspberry Pi:

* **setup_pi.sh** – installs dependencies, enables zram and builds
  `llama.cpp` on ARM64 with OpenBLAS.
* **get_model.sh** – downloads a pre‑quantized small model from
  HuggingFace (TinyLlama 1.1 B or Qwen 2.5 B) as a starting point.
* **run_decoder.sh** – runs the GGUF model interactively, with
  optional grammar constraints, prompt caching and custom context
  lengths.
* **run_encoder.sh** – generates embeddings from the GGUF model for
  use with a vector database.
* **bench/pi_bench.py** – measures decode throughput and embedding
  latency; fails if a minimum token rate is not met.

## 🏗️ Makefile & Automation

The `Makefile` exposes common tasks as single commands (e.g.
`make distill`, `make export`, `make deploy`), while
`automation/e2e.sh` chains the entire workflow together.  Combined
with environment variables defined in `.env`, these scripts allow
fully automated training, export and deployment from your desktop to
the Pi.

See the repository README for usage examples and the GitHub Action
file for an example of continuous integration.
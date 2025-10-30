# Simple Makefile for common project tasks

ifndef .env
ifneq (,$(wildcard .env))
include .env
export
endif
endif

PYTHON ?= python

# Label the dataset using the teacher model
distill:
	@echo "Labeling dataset with teacher $(MODEL_ID)"
	$(PYTHON) desktop_distill/teacher_label.py \
		--model $(MODEL_ID) \
		--input dataset/dataset_quebecois_distill.jsonl \
		--output dataset/labelled.jsonl

# Train the student model using LoRA/DoRA and QLoRA if enabled
train:
	@echo "Training student model $(STUDENT_ID) on labelled dataset"
	$(PYTHON) desktop_distill/train_student.py \
		--dataset dataset/labelled.jsonl \
		--base_model $(STUDENT_ID) \
		--output_dir trained_student \
		--use_dora \
		--qlora \
		--num_epochs 1 \
		--batch_size 2 \
		--gradient_accumulation_steps 8 \
		--learning_rate 2e-5

# Export the trained model to GGUF with quantization
export:
	@echo "Exporting student model to GGUF"
	$(PYTHON) desktop_distill/export_gguf.py \
		--model trained_student \
		--outdir gguf_artifacts \
		--qtype q4_k_m

# Deploy the model artifacts to the Raspberry Pi
deploy:
	@echo "Deploying GGUF artifacts to $(PI_USER)@$(PI_HOST):$(PI_DIR)"
	ssh $(PI_USER)@$(PI_HOST) "mkdir -p $(PI_DIR)"
	scp gguf_artifacts/*.gguf $(PI_USER)@$(PI_HOST):$(PI_DIR)/
	scp gguf_artifacts/tokenizer* $(PI_USER)@$(PI_HOST):$(PI_DIR)/

# Run the model on the Pi interactively
run:
	@echo "Running model on the Pi"
	ssh $(PI_USER)@$(PI_HOST) "bash -l -c 'cd $(PI_DIR) && ~/llama.cpp/build/bin/llama-cli -m $(PI_DIR)/$(shell ls gguf_artifacts/*.gguf | xargs -n1 basename) -c 1024 -b 64 -t 4 --prompt-cache $(PI_DIR)/prompt_cache.bin --interactive'"

# Benchmark the model on the Pi
bench:
	@echo "Benchmarking model on the Pi"
	$(PYTHON) rpi4/bench/pi_bench.py \
		--model gguf_artifacts/$(shell ls gguf_artifacts/*.gguf | xargs -n1 basename) \
		--iterations 3 \
		--min-tokps 0.25 \
		--csv rpi4/bench/out/bench.csv

.PHONY: distill train export deploy run bench
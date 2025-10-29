# Automations

all: distill export deploy run bench

include .env

.PHONY: distill
distill:
	python3 desktop_distill/train_student.py \
	  --teacher "$(MODEL_ID)" --student "$(STUDENT_ID)" \
	  --dataset "$(DATASET)" --out out/student ${HF_TOKEN:+--hf-token $(HF_TOKEN)}

.PHONY: export
export:
	python3 desktop_distill/export_gguf.py --model out/student --out out/gguf --qtype q4_k_m ${HF_TOKEN:+--hf-token $(HF_TOKEN)}

.PHONY: deploy
deploy:
	rsync -av out/gguf/*.gguf $(PI_USER)@$(PI_HOST):$(PI_DIR)/model-q4_k_m.gguf

.PHONY: run
run:
	ssh $(PI_USER)@$(PI_HOST) "~/llama.cpp/build/bin/llama-cli -m $(PI_DIR)/model-q4_k_m.gguf -n 16 -p 'Bonjour'"

.PHONY: bench
bench:
	python3 rpi4/bench/pi_bench.py --model out/gguf/*.gguf --iterations 3 --csv rpi4/bench/out/bench.csv

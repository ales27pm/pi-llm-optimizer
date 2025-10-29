#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git build-essential cmake libopenblas-dev python3-pip zram-tools

# 50% zram to soften OOM on 4 GB
printf "ALGO=lz4\nPERCENT=50\nPRIORITY=100\n" | sudo tee /etc/default/zram-config >/dev/null
sudo systemctl enable --now zram-config

# llama.cpp (ARM/NEON + OpenBLAS)
if [ ! -d "$HOME/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$HOME/llama.cpp"
fi
cd "$HOME/llama.cpp"
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -DGGML_OPENBLAS=ON ..
make -j"$(nproc)"

echo "[OK] Built llama.cpp → $HOME/llama.cpp/build/bin"

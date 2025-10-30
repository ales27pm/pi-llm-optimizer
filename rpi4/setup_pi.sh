#!/usr/bin/env bash
# Bootstrap a Raspberry Pi 4 for running llama.cpp models.
#
# This script installs dependencies, enables zram to mitigate out of
# memory issues on 4 GB boards, clones the llama.cpp repository and
# builds it with NEON and OpenBLAS support.

set -euo pipefail

sudo apt-get update
sudo apt-get install -y git build-essential cmake libopenblas-dev python3-pip zram-tools

# Configure half of RAM as compressed zram swap to reduce OOM risk
printf "ALGO=lz4\nPERCENT=50\nPRIORITY=100\n" | sudo tee /etc/default/zram-config >/dev/null
sudo systemctl enable --now zram-config

# Clone llama.cpp if it doesn't exist
if [ ! -d "$HOME/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$HOME/llama.cpp"
fi
cd "$HOME/llama.cpp"
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -DGGML_OPENBLAS=ON ..
make -j"$(nproc)"

echo "[OK] Built llama.cpp with NEON/OpenBLAS.  Binaries are in $HOME/llama.cpp/build/bin"
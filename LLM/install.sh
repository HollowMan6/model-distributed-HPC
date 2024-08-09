#!/bin/bash

set -e

module purge
module load CrayEnv cray-python rocm/5.6.1 buildtools PrgEnv-gnu/8.4.0
python -m venv venv
source venv/bin/activate
export GPU_ARCHS="gfx90a"
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6


if [ ! -d "$PWD/vllm/bin" ]; then
  ml tykky
  mkdir -p vllm
  conda-containerize new --prefix $PWD/vllm env.yml
fi
export PATH="$PWD/vllm/bin:$PATH"

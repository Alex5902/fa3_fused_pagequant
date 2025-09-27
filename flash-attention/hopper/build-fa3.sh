#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00
#SBATCH -J build-fa3

set -euo pipefail

module load cuda/12.4 || true   # load if available

conda activate fa3-fpq
pip install -U pip wheel ninja

# clean previous mixed caches
rm -rf ~/.flashattn/nvidia \
       ~/fa3/flash-attention/third_party/nvidia \
       ~/fa3/flash-attention/hopper/build

export TORCH_CUDA_ARCH_LIST=90a
cd ~/fa3/flash-attention/hopper
python -m pip install -v .

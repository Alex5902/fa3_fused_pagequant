#!/bin/bash
#SBATCH -J fa3_sanity_a6000
#SBATCH -p 48-4
#SBATCH --gres=gpu:rtxA6000:1      # or: --gres=gpu:1 + --constraint=rtxA6000
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH --chdir=/home/alex.ia/fa3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Conda env
source ~/.bashrc
conda activate fa3-fpq

# CUDA toolkit (matches what you used to build on the login node)
module load cuda/12.6
export CUDA_HOME=${CUDA_HOME:-/apps/cuda/cuda-12.6}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX A6000 (SM86)

echo "Hostname: $(hostname)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME: ${CUDA_HOME}"
nvcc --version || true

python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not visible to PyTorch on this node")
PY

nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Pre-build the CUDA extension (fast if cached; fails early if toolchain mismatch)
python - <<'PY'
from harness import fused_dequant_ext
print("Built fused_dequant_producer OK (compute node)")
PY

# Run tests/benches
python -m kv_formats.sanity_tile_check
python -m harness.unit_tests
python -m harness.microbench --M 1024 --dk 128 --dv 128

#!/bin/bash
#SBATCH -J fa3_proto_a6000
#SBATCH -p 48-4
#SBATCH --gres=gpu:rtxA6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH --chdir=/home/alex.ia/fa3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

source ~/.bashrc
conda activate fa3-fpq

module load cuda/12.6
export CUDA_HOME=${CUDA_HOME:-/apps/cuda/cuda-12.6}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

echo "Hostname: $(hostname)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME: ${CUDA_HOME}"
nvcc --version || true
nvidia-smi || true

# Pre-build both extensions so any toolchain issues fail early
python - <<'PY'
from harness import fa3_proto_ext, fused_dequant_ext
print("Built fa3_proto & fused_dequant_producer OK (compute node)")
PY

# Optional: one-time blocking for crisp stacktraces if anything trips
export CUDA_LAUNCH_BLOCKING=1

# Run the proto test (producer→SMEM→consumer on K; compares to PyTorch)
python -m harness.proto_test

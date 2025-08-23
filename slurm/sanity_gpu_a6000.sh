#!/bin/bash
#SBATCH -J fa3_sanity_a6000
#SBATCH -p 48-4
#SBATCH --gres=gpu:rtxA6000:1          # or: --gres=gpu:1 + --constraint=rtxA6000
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH --chdir=/home/alex.ia/fa3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Make conda available on compute nodes
source ~/.bashrc
conda activate fa3-fpq

echo "Hostname: $(hostname)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Optional: load a CUDA module if your site requires it (uncomment and pick the right one)
# module load cuda/12.2

python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("NOTE: You’re on a GPU node but PyTorch CUDA isn’t installed in this env.")
    print("Install it once on a login node: pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision")
PY

nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Run from repo root (set by --chdir). Use module style for robustness.
python -m kv_formats.sanity_tile_check
python -m harness.unit_tests
python -m harness.microbench --M 1024 --dk 128 --dv 128

#!/bin/bash -l
#SBATCH -J fa3_preflight_check
#SBATCH -p 48-4          # Use the same partition as your main job
#SBATCH --gpus=1         # We need a GPU for this check
#SBATCH -t 00:05:00      # 5 minutes is more than enough
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/home/alex.ia/fa3
#SBATCH -o logs/preflight-%j.out
#SBATCH -e logs/preflight-%j.err

echo "===== PRE-FLIGHT CHECK ON COMPUTE NODE ====="
hostname
echo ""

echo "--- 1. Checking GPU ---"
nvidia-smi
echo ""

echo "--- 2. Checking Module System ---"
module purge
module load cuda/12.6
echo "CUDA module loaded."
nvcc --version
echo ""

echo "--- 3. Checking Conda Environment ---"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fa3-fpq
echo "Conda environment activated."
echo "which python -> $(which python)"
echo ""

echo "--- 4. Checking PyTorch and CUDA version from Python ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')"
echo ""

echo "===== PRE-FLIGHT CHECK COMPLETE ====="
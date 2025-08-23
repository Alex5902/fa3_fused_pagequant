#!/bin/bash
#SBATCH -J fa3_sanity_cpu
#SBATCH -A <your_account>           # <-- fill if your cluster requires accounts
#SBATCH -p <cpu_partition>          # e.g., cpu, short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Load shell so "conda" is available (adjust path if needed)
source ~/.bashrc
conda activate fa3-fpq

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"  # the repo root where you submit from

# 1) basic quantizer checks
python kv_formats/sanity_tile_check.py

# 2) correctness vs FP16 reference
python harness/unit_tests.py

# 3) microbench (CPU run)
python harness/microbench.py --M 512 --dk 128 --dv 128

#!/bin/bash
#PBS -N fa3_correctness
#PBS -q rt_HG
#PBS -P gce51019
#PBS -l select=1
#PBS -l walltime=00:05:00

set -euo pipefail

# --- PATHS ---
ROOT=/groups/gce51019/alex.ia/fa3
LOGDIR="$ROOT/logs"
# The 'hopper' directory contains the fa3_cuda package with __init__.py
HOPPER_PACKAGE_ROOT="$ROOT/flash-attention/hopper"
ENV=/groups/gce51019/alex.ia/envs/fa3-fpq
MINICONDA=/groups/gce51019/alex.ia/miniconda

echo "== JOB INFO =="
echo "Job ID: ${PBS_JOBID:-}"
echo "Running on: $(hostname)"
date

# --- ENVIRONMENT SETUP ---
echo "== Activating Conda Environment =="
. "$MINICONDA/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "== Loading Modules (ABCI) =="
module purge
module load cuda/12.8

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"

echo "== Verifying Environment =="
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version
nvidia-smi -L

# Make the in-place built extension visible to Python
# This allows `import fa3_cuda` to work.
export PYTHONPATH="${HOPPER_PACKAGE_ROOT}:${PYTHONPATH-}"

# CRITICAL: Add Torch’s internal lib dir to the library path
# This helps the .so find libraries like libc10.so and libtorch_cuda.so
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'):${LD_LIBRARY_PATH-}"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# This will make the program crash *exactly* on the line that causes the error.
export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING is set to 1"

echo "== Running Correctness Test =="
python -u "$ROOT/harness/fa3_correctness.py"
# compute-sanitizer --tool memcheck \
#   --target-processes all \
#   --print-limit 100 \
#   python -u "$ROOT/harness/fa3_correctness.py"


echo "== Job Finished Successfully =="
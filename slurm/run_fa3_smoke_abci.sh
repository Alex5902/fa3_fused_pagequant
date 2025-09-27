#!/bin/sh
#PBS -N fa3_smoke
#PBS -q rt_HG
#PBS -P gce51019
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -m n
#PBS -j oe
#PBS -o /groups/gce51019/alex.ia/fa3/logs/
#PBS -e /groups/gce51019/alex.ia/fa3/logs/

set -euo pipefail

# --- paths ---
ROOT=/groups/gce51019/alex.ia/fa3
LOGDIR="$ROOT/logs"
HOPPER="$ROOT/flash-attention/hopper"
ENV=/groups/gce51019/alex.ia/envs/fa3-fpq
MINICONDA=/groups/gce51019/alex.ia/miniconda

mkdir -p "$LOGDIR"

echo "== ENV =="; hostname; date
echo "PBS_JOBID=${PBS_JOBID:-}"
echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-}"

# --- Conda env ---
. "$MINICONDA/etc/profile.d/conda.sh"
conda activate "$ENV"

# --- Modules (ABCI) ---
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module purge || true
module load cuda/12.8 || module load cuda/12.6

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"

echo "CUDA_HOME=$CUDA_HOME"
nvcc --version || true
nvidia-smi -L || nvidia-smi || true

# Make the in-place built extension visible to Python
export PYTHONPATH="$HOPPER:${PYTHONPATH-}"

# IMPORTANT: add Torch’s lib dir so libc10.so, libtorch_cuda.so, etc. resolve
export LD_LIBRARY_PATH="$(
python - <<'PY'
import pathlib, torch
print(pathlib.Path(torch.__file__).parent / "lib")
PY
):${LD_LIBRARY_PATH-}"

echo "which python: $(which python)"
python -V

python - <<'PY'
import importlib, torch
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
m = importlib.import_module("fa3_cuda")
print("loaded module:", getattr(m, "__file__", "<builtin>"))

# tiny CUDA op to exercise the stack
x = torch.randn(8, 8, device="cuda", dtype=torch.float16)
y = (x @ x.t()).sum().item()
print("tiny CUDA op ok; checksum:", round(y, 4))
print("OK")
PY

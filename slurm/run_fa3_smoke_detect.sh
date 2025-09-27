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

ROOT=/groups/gce51019/alex.ia/fa3
LOGDIR="$ROOT/logs"
HOPPER="$ROOT/flash-attention/hopper"
ENV=/groups/gce51019/alex.ia/envs/fa3-fpq
MINICONDA=/groups/gce51019/alex.ia/miniconda

mkdir -p "$LOGDIR"

echo "== ENV =="; hostname; date
echo "PBS_JOBID=${PBS_JOBID:-}"; echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-}"

# --- Conda env ---
. "$MINICONDA/etc/profile.d/conda.sh"
conda activate "$ENV"

# --- CUDA modules (ABCI) ---
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module purge || true
module load cuda/12.8 || module load cuda/12.6

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"

echo "CUDA_HOME=$CUDA_HOME"
nvcc --version || true
nvidia-smi -L || nvidia-smi || true

# Make the in-place build visible
export PYTHONPATH="$HOPPER:${PYTHONPATH-}"

# Locate the built .so (first match)
SO="$(ls -1 "$HOPPER"/fa3_cuda*.so 2>/dev/null | head -n1 || true)"
if [ -z "${SO}" ]; then
  echo "ERROR: Could not find fa3_cuda*.so under $HOPPER"; exit 2
fi
echo "Found extension: $SO"

# Find which PyInit_* symbol it actually exports
INIT_SYM="$(nm -D "$SO" 2>/dev/null | awk '/PyInit_/ {print $3}' | sort -u | head -n1 || true)"
echo "Exported init symbol: ${INIT_SYM:-<none>}"

echo "which python: $(which python)"; python -V

python - <<PY
import importlib.util, importlib.machinery, pathlib, sys, torch, os
so_path = pathlib.Path(r"$SO")
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))

# Try to discover the correct module init name from the library
init_name = "${INIT_SYM}"
modname = None
if init_name.startswith("PyInit_"):
    modname = init_name[len("PyInit_"):]
else:
    # Fallback guesses
    for guess in ("fa3_cuda","flash_attn_3_cuda","flash_attn_cuda"):
        try:
            spec = importlib.util.spec_from_file_location(guess, so_path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)  # type: ignore
            modname = guess
            break
        except Exception as e:
            print(f"try import {guess}: {e!r}")

if modname is None:
    # Final attempt: if we *do* have a PyInit_* symbol, try that exact name
    if init_name.startswith("PyInit_"):
        modname = init_name[len("PyInit_"):]
        spec = importlib.util.spec_from_file_location(modname, so_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore
    else:
        raise RuntimeError("Could not determine module init name; check build.")

# Load with the discovered name
spec = importlib.util.spec_from_file_location(modname, so_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)  # type: ignore
print("loaded as:", modname, "from:", m.__file__)

# Alias to 'fa3_cuda' so downstream 'import fa3_cuda' works in this process
sys.modules["fa3_cuda"] = m
print("alias 'fa3_cuda' →", modname)
print("OK")
PY

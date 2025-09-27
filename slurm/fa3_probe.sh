#!/bin/bash
#PBS -N fa3_probe
#PBS -q rt_HG
#PBS -P gce51019
#PBS -l select=1
#PBS -l walltime=00:20:00

set -euo pipefail

# --- PATHS (match your correctness script) ---
ROOT=/groups/gce51019/alex.ia/fa3
HOPPER_PACKAGE_ROOT="$ROOT/flash-attention/hopper"   # contains fa3_cuda package
ENV=/groups/gce51019/alex.ia/envs/fa3-fpq
MINICONDA=/groups/gce51019/alex.ia/miniconda

echo "== JOB INFO =="
echo "Job ID: ${PBS_JOBID:-}"
echo "Running on: $(hostname)"
date

echo "== Activating Conda Environment =="
. "$MINICONDA/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "== Loading Modules (ABCI) =="
module purge
module load cuda/12.8

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"

# Make the in-place built extension visible to Python
export PYTHONPATH="${HOPPER_PACKAGE_ROOT}:${PYTHONPATH-}"

# Add Torch's internal lib dir so the .so can find libc10/libtorch_cuda
export LD_LIBRARY_PATH="$(python - <<'PY'
import pathlib, torch, sys
p = pathlib.Path(torch.__file__).parent / "lib"
print(str(p))
PY
):${LD_LIBRARY_PATH-}"

echo "== ENV CHECK =="
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version
nvidia-smi -L
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | cut -c-200
python - <<'PY'
import os, sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, " torch.cuda:", torch.version.cuda)
print("GPU0:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))
PY
echo

# ---------- A) DISPATCH / SCHEMA PROBE ----------
echo "== A) DISPATCH / SCHEMA PROBE =="
python - <<'PY'
import torch, fa3_cuda  # noqa: F401
ns = torch.ops.flash_attn_3
print("namespace has:", [k for k in dir(ns) if not k.startswith('_')])
print("repr(fwd):", ns.fwd)

# 1) Dump dispatch table if available
try:
    from torch._C import _dispatch_dump_table
    print("\n--- _dispatch_dump_table(flash_attn_3::fwd) ---")
    print(_dispatch_dump_table("flash_attn_3::fwd"))
except Exception as e:
    print("_dispatch_dump_table unavailable:", e)

# 2) List JIT schemas if available
try:
    from torch._C import _jit_get_all_schemas
    schemas = [s for s in _jit_get_all_schemas() if s.name == "flash_attn_3::fwd"]
    print("\n--- _jit_get_all_schemas (filtered) ---")
    if schemas:
        for s in schemas: print(s)
    else:
        print("(no schemas found for flash_attn_3::fwd)")
except Exception as e:
    print("_jit_get_all_schemas unavailable:", e)

# 3) Trigger a bogus kw to see what kwargs the op accepts (error usually lists them)
try:
    ns.fwd(torch.empty(1), torch.empty(1), torch.empty(1), bogus_kw=123)
except Exception as e:
    print("\n--- bogus_kw error (often lists accepted kwargs) ---")
    print(e)
PY
echo

# Locate the built .so
SO_PATH="$(python - <<'PY'
import fa3_cuda as m, os
print(m.__file__)
PY
)"
echo "fa3_cuda .so path: ${SO_PATH}"
echo

# ---------- C) BINARY SYMBOL / STRING CHECKS ----------
echo "== C) SYMBOLS via nm -D (int8/int8kv/dequant/fused) =="
if command -v nm >/dev/null 2>&1; then
  (nm -D "${SO_PATH}" 2>/dev/null | grep -Ei 'int8|int8kv|dequant|fused' | head -n 80) || echo "(no matching dynamic symbols or stripped)"
else
  echo "nm not found"
fi
echo

echo "== C) STRINGS scan (int8/int8kv/dequant/fused) =="
if command -v strings >/dev/null 2>&1; then
  (strings "${SO_PATH}" 2>/dev/null | grep -Ei 'int8|int8kv|dequant|fused' | head -n 80) || echo "(no matching strings)"
else
  echo "strings not found"
fi
echo

echo "== C) CUOBJDUMP arch listing =="
if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --list "${SO_PATH}" 2>/dev/null | sed -n '1,120p'
else
  echo "cuobjdump not found"
fi
echo

echo "== DONE =="
echo "Review: "
echo " - In A) check if fwd schema lists k_q/v_q and k_scales/v_scales (or variant names)."
echo " - In C) ensure symbols/strings mention int8/int8kv/dequant/fused; otherwise rebuild with INT8KV enabled."

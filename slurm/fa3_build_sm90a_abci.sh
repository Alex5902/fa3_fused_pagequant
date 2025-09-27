#!/bin/bash -l
#PBS -N fa3_build_final
#PBS -q rt_HG                  
#PBS -P gce51019
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=24:00:00

# This line is good, but we add echo checkpoints to see where it might fail
set -euo pipefail

# ---------- PATHS ----------
export ROOT_DIR="/groups/gce51019/alex.ia/fa3"
export PROJECT_DIR="$ROOT_DIR/flash-attention"
export HOPPER_DIR="$PROJECT_DIR/hopper" 
export ENV_DIR="/groups/gce51019/alex.ia/envs/fa3-fpq"
export MINICONDA_DIR="/groups/gce51019/alex.ia/miniconda"
export LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "== Paths configured =="

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# ---------- BUILD KNOBS ----------
export MAX_JOBS=${MAX_JOBS:-16}
export TORCH_CUDA_BUILD_VERBOSE=1
export TORCH_CUDA_ARCH_LIST="90a"
export FLASH_ATTENTION_FORCE_BUILD=1

# Target Hopper only; drop SM80 to prune variants
export FLASH_ATTENTION_DISABLE_SM80=1
export FLASH_ATTENTION_DISABLE_SM8x=1

# DTypes: keep fp16, bf16, int8; drop fp8 to prune a lot
export FA3_ENABLE_DTYPES="fp16,bf16,int8"
export FA3_DISABLE_FP8=1
export FLASHATTENTION_DISABLE_FP8=1

# Keep only the head dims you care about
export FA3_HEAD_DIMS="64,128"
# Explicitly disable the rest (avoid accidental instantiations)
export FLASHATTENTION_DISABLE_HDIM96=1
export FLASHATTENTION_DISABLE_HDIM192=1
export FLASHATTENTION_DISABLE_HDIM256=1
# Also disable “different V head dim” specializations
export FLASHATTENTION_DISABLE_HDIMDIFF64=1
export FLASHATTENTION_DISABLE_HDIMDIFF192=1

# INT8 QKV path (for quantization research)
export FA3_ENABLE_QKV=1

# Trim scheduler/features that explode templates
export FA3_ENABLE_SPLIT=0          # no split kernels
export FA3_ENABLE_PAGED=0          # no paged KV cache (turn on only if you truly need it)
export FA3_ENABLE_PACKGQA=0        # reduces variants; enable only if needed

# Softcap not typically needed for quantization benchmarks
export FA3_ENABLE_SOFTCAP=0
export FLASH_ATTENTION_DISABLE_SOFTCAP=1

# Forward-only (drops a *ton* of code)
export FA3_DROP_BWD=1
export FLASHATTENTION_DISABLE_BACKWARD=1
# Also pass to compilers (keeps translation units small)
export CXXFLAGS="${CXXFLAGS:-} -DFA3_BUILD_FWD_ONLY -DFLASHATTENTION_DISABLE_BACKWARD"
export NVCCFLAGS="${NVCCFLAGS:-} -DFA3_BUILD_FWD_ONLY -DFLASHATTENTION_DISABLE_BACKWARD"
export TORCH_NVCC_FLAGS="${TORCH_NVCC_FLAGS:-} -DFA3_BUILD_FWD_ONLY -DFLASHATTENTION_DISABLE_BACKWARD"

# If you previously hit “relocation truncated to fit / text too far”:
# keep large code model for the shared link step
export LDSHARED="/usr/bin/g++ -shared -mcmodel=large"

echo "== Build knobs configured =="

# ---- Step 1: Activate Conda Environment ----
echo "== Sourcing and activating Conda environment =="
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
echo "== Conda environment activated. Python is at: $(which python) =="

# ---- Step 2: Repair PyTorch Installation ----
echo "== Repairing PyTorch installation to fix missing libraries... =="
# The --force-reinstall flag will remove the broken ~orch directory and correctly install all .so files.
# We specify the exact CUDA 12.1 version for stability.
python -m pip install --force-reinstall torch==2.3.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "== PyTorch repair complete. Verifying installation... =="
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available()"

# ---- Toolchain (ABCI has CUDA modules) ----
echo "== Loading CUDA module =="
module purge
module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"
echo "== CUDA module loaded. CUDA_HOME=$CUDA_HOME =="
nvcc --version

export TORCH_LIBDIR="$(python - <<'PY'
import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIBDIR:${LD_LIBRARY_PATH-}"

# ---- Regenerate kernel instantiations ----
echo "== Navigating to hopper directory to regenerate kernels =="
cd "$HOPPER_DIR"
echo "Current directory: $(pwd)"
rm -rf instantiations
echo "== Running kernel generator script =="
python generate_kernels.py -o instantiations 

# --- PRUNE UNWANTED INSTANTIATIONS ---

# 1) keep Hopper only (kill Ampere/Ada)
find instantiations -type f \( -name '*_sm80.cu' -o -name '*_sm86.cu' -o -name '*_sm89.cu' \) -print -delete

# 2) forward-only: drop all backward files & softcap (since you set disable-softcap)
find instantiations -type f -name 'flash_bwd_*' -print -delete
find instantiations -type f -name '*softcap*' -print -delete
find instantiations -type f -name '*softcapall*' -print -delete

# kill all “batch” aggregators so we only compile the individual kernels we kept
find instantiations -type f -name 'flash_fwd_hdimall_*'  -print -delete
find instantiations -type f -name 'flash_fwd_hdimdiff_*' -print -delete
find instantiations -type f -name 'flash_bwd_hdim*_softcapall_sm*.cu' -print -delete

# 3) head dims: keep only 64/128
find instantiations -type f -name 'flash_fwd_hdim*.cu' \
  ! -name '*hdim64*' ! -name '*hdim128*' -print -delete

# different V-dim variants (not needed for your quant work)
find instantiations -type f -name 'flash_fwd_hdim64_256_*'  -print -delete
find instantiations -type f -name 'flash_fwd_hdim64_512_*'  -print -delete
find instantiations -type f -name 'flash_fwd_hdim192_128_*' -print -delete

# 4) dtype: drop fp8 if you don’t want it
find instantiations -type f -name '*_e4m3_*' -print -delete

# 5) optional features you disabled
[ "${FA3_ENABLE_PAGED:-0}" = "0" ]   && find instantiations -type f -name '*_paged*'   -print -delete
[ "${FA3_ENABLE_PACKGQA:-0}" = "0" ] && find instantiations -type f -name '*_packgqa*' -print -delete
[ "${FA3_ENABLE_SPLIT:-0}" = "0" ]   && find instantiations -type f -name '*_split*'   -print -delete

# 6) sanity: show what remains
echo "Remaining instantiations:"
ls -1 instantiations | sed -n '1,50p'
echo "== Kernel instantiation pruning complete. =="

# ---- Build deps ----
echo "== Installing build dependencies =="
python -m pip install -U pip wheel ninja packaging setuptools

# ---- Clean stale build artifacts ----
echo "== Cleaning stale artifacts =="
python setup.py clean
rm -rf build hopper/build .ninja_*

echo "== Searching for and removing any old .so files in the source tree... =="
find "$PROJECT_DIR" -name "*.so" -print -delete
echo "== Old .so files removed. Starting build... =="

# ---- Final Build Step ----
LOG_FILE="$LOG_DIR/build_output_${PBS_JOBID:-manual}.log"
echo "== Starting final build. Output will be logged to: $LOG_FILE =="

# Using a subshell with `set -o pipefail` is safer
(set -o pipefail; python setup.py build_ext --inplace 2>&1 | tee "$LOG_FILE")

echo "== Build command finished. Checking for errors in log... =="
# Check for common failure keywords in the detailed log
if grep -q -i -E "error:|fatal error:|undefined reference|ninja: build stopped" "$LOG_FILE"; then
    echo "!!!!!! BUILD FAILED. See errors below. !!!!!!"
    grep -i -n -E "error:|fatal error:|undefined reference|ninja: build stopped" "$LOG_FILE"
    exit 1
else
    echo "++++++ BUILD SUCCEEDED. ++++++"
fi

echo "== Verifying built library =="
ls -lh "$HOPPER_DIR"/*.so

echo "== Import smoke test =="
export PYTHONPATH="$HOPPER_DIR:$PYTHONPATH"
python - <<'PY'
import os, sys, importlib.util
print("sys.version:", sys.version)
print("PYTHONPATH head:", sys.path[0])
try:
    import fa3_cuda
    print("Loaded:", getattr(fa3_cuda, "__file__", "<no file>"))
    # Optional: probe that the symbol set we expect is there
    import ctypes, subprocess, shlex
    so = getattr(fa3_cuda, "__file__", "")
    if so:
        print("ldd check:")
        print(subprocess.run(["ldd", so], text=True, capture_output=True).stdout)
    print("OK: import worked")
except Exception as e:
    import traceback; traceback.print_exc(); sys.exit(1)
PY

echo "== SCRIPT FINISHED SUCCESSFULLY =="
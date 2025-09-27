#!/bin/bash -l
#PBS -N fa3_build_final_direct
#PBS -q rt_HG
#PBS -P gce51019
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=02:00:00

set -euxo pipefail

# ---------- 1. PATHS & BASIC SETUP ----------
export ROOT_DIR="/groups/gce51019/alex.ia/fa3"
export PROJECT_DIR="$ROOT_DIR/flash-attention"
export HOPPER_DIR="$PROJECT_DIR/hopper"
export ENV_DIR="/groups/gce51019/alex.ia/envs/fa3-fpq"
export MINICONDA_DIR="/groups/gce51019/alex.ia/miniconda"
export LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
export BUILD_LOG_FILE="$LOG_DIR/build_output_${PBS_JOBID:-manual}.log"
echo "== Paths configured. Build log will be at: $BUILD_LOG_FILE =="

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MAX_JOBS=${PBS_NCPUS:-16}
export TORCH_CUDA_BUILD_VERBOSE=1
export TORCH_CUDA_ARCH_LIST="90a"
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# ---------- 2. MINIMAL BUILD CONFIGURATION ----------
echo "== Applying MINIMAL BUILD configuration =="
export FLASH_ATTENTION_DISABLE_SM80=TRUE 
export FA3_DROP_BWD="True"
export FA3_ENABLE_DTYPES="bf16,fp16,int8"
export FA3_HEAD_DIMS="64,128"
# export FA3_ENABLE_QKV="True"
export FA3_ENABLE_SPLIT="False"
export FA3_ENABLE_PAGED="True"
export FA3_ENABLE_SOFTCAP="True"
# export FA3_ENABLE_PACKGQA="True"

export FLASHATTENTION_DISABLE_FP8=1
export FA3_DISABLE_FP8=1
export FLASH_ATTENTION_FORCE_NO_FP8=1


# ---------- 3. ENVIRONMENT & TOOLCHAIN SETUP ----------
echo "== Activating Conda environment =="
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
echo "Python is at: $(which python)"

echo "== Loading CUDA module (ABCI) =="
module purge
module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"
nvcc --version

export TORCH_LIBDIR="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIBDIR:${LD_LIBRARY_PATH-}"

export CXXFLAGS="${CXXFLAGS:-} -DFA3_BUILD_FWD_ONLY -DFLASHATTENTION_DISABLE_BACKWARD"
export NVCCFLAGS="${NVCCFLAGS:-} -DFA3_BUILD_FWD_ONLY -DFLASHATTENTION_DISABLE_BACKWARD"

# ---------- 4. DIRECT BUILD PROCESS ----------
echo "== Navigating to build directory: $HOPPER_DIR =="
cd "$HOPPER_DIR"

echo "== Installing build dependencies (ninja only) =="
python -m pip install -U ninja

echo "== Cleaning all stale build artifacts =="
rm -rf build dist *.egg-info
find . -name "*.so" -print -delete
echo "== Stale artifacts cleaned. =="

# ---- THE FIX: Use the original, direct build command which avoids pip's complexities ----
echo "== Starting DIRECT build with 'build_ext --inplace'. Logged to $BUILD_LOG_FILE =="
(
  set -o pipefail
  python setup.py build_ext --inplace 2>&1 | tee "$BUILD_LOG_FILE"
)

echo "== Build command finished. Checking for errors... =="
if grep -q -i -E "error:|fatal error:|undefined reference|ninja: build stopped" "$BUILD_LOG_FILE"; then
    echo "!!!!!! BUILD FAILED. Critical errors found in log. !!!!!!"
    grep -i -C 5 -E "error:|fatal error:|undefined reference|ninja: build stopped" "$BUILD_LOG_FILE"
    exit 1
else
    echo "++++++ BUILD SUCCEEDED. ++++++"
fi


# ---------- 5. VERIFICATION ----------
echo "== Verifying built library location =="
# This should now show the .so file in the current (hopper) directory
ls -lh "$HOPPER_DIR"/*.so

echo "== Preparing to run test by adding build directory to PYTHONPATH =="
export PYTHONPATH="$HOPPER_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "== Running final correctness test =="
python -u "$ROOT_DIR/harness/fa3_correctness.py"

echo "== SCRIPT FINISHED SUCCESSFULLY =="
#!/bin/bash -l
#SBATCH -J fa3_dbg_sm90a
#SBATCH -p 48-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --chdir=/home/alex.ia/fa3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs ~/fa3/wheels

echo "== ENV =="; hostname; date
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

# ---------- DIAG KNOBS ----------
export MAX_JOBS=1                          # one TU at a time, clearer failures
export TORCH_CUDA_BUILD_VERBOSE=1
export TORCH_CUDA_ARCH_LIST=90a            # H100 path
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CXXFLAGS="${CXXFLAGS-} -fmax-errors=1 -fdiagnostics-color=always"
export FA3_HOPPER_ONLY=${FA3_HOPPER_ONLY:-0}
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
# ---------------------------------

# ---- Conda env ----
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate fa3-fpq

echo "which python: $(which python)"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("sys.executable", sys.executable)
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
PY
python -m pip -V

# It's safer to uninstall the broken ones first.
python -m pip uninstall -y torchaudio torchvision
# Now, install torch again to pull in the correct dependencies.
python -m pip install --force-reinstall "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128

echo "which python after fix: $(which python)"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("sys.executable", sys.executable)
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
PY

# ---- Toolchain (cluster has CUDA 12.6) ----
module purge || true
module load cuda/12.6

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH-}"

echo "CUDA_HOME=$CUDA_HOME"; nvcc --version || true; gcc --version | head -n1 || true
# warn (don’t fail) if nvcc CUDA != torch.version.cuda
python - <<'PY'
import re, subprocess, sys, torch
nv = subprocess.check_output(["nvcc","--version"], text=True, errors="ignore")
m = re.search(r"release (\d+\.\d+)", nv)
nvcc_ver = m.group(1) if m else "unknown"
torch_ver = torch.version.cuda or "unknown"
print(f"nvcc CUDA={nvcc_ver}  vs  torch.version.cuda={torch_ver}")
if not (torch_ver != "unknown" and nvcc_ver.startswith(torch_ver)):
    print("WARNING: nvcc/torch CUDA mismatch; build may fail.")
PY

# ---- Build deps (env pip) ----
python -m pip install -U pip wheel ninja packaging setuptools

# ---- Fix the build error ----
# --- Forcefully reinstall a known compatible version of setuptools ---
# echo "Forcibly reinstalling a known compatible version of setuptools..."
# python -m pip uninstall -y setuptools
# python -m pip install -U "setuptools>=75,<81"
echo "Setuptools version after fix:"
python -m pip show setuptools | grep Version

# ---- Clean stale FA bits ----
rm -rf \
  "$HOME/fa3/flash-attention/third_party/nvidia" \
  "$HOME/fa3/flash-attention/hopper/build" \
  "$HOME/fa3/flash-attention/hopper/.ninja_*" || true

# ---- Re-populate vendor submodules & prefer them on include path ----
cd "$HOME/fa3/flash-attention"
git submodule sync --recursive || true
git submodule update --init --recursive || true

# Put vendor CUTLASS/CUTE ahead of repo csrc/ headers to avoid API skew
export CPATH="$HOME/fa3/flash-attention/third_party/nvidia/cutlass/include:$HOME/fa3/flash-attention/third_party/nvidia/cute/include:${CPATH-}"

# ---- Patch duplicate inline/attribute issue (no-op if nothing matches) ----
echo "Scanning & patching duplicate __inline__/__attribute__ sequences (if any)…"
mapfile -t PATCH_FILES < <(
  grep -RIl "__inline__ *__attribute__\(\(always_inline\)\).*__inline__ *__attribute__\(\(always_inline\)\)" \
    "$HOME/fa3/flash-attention/hopper" 2>/dev/null || true
)
if ((${#PATCH_FILES[@]})); then
  for f in "${PATCH_FILES[@]}"; do
    echo "  patching $f"
    perl -0777 -pe 's/__inline__\s+__attribute__\(\(always_inline\)\)\s+__attribute__\(\(device\)\)\s+__inline__\s+__attribute__\(\(always_inline\)\)/__inline__ __attribute__((always_inline)) __attribute__((device))/g' -i "$f"
  done
else
  echo "  no files to patch"
fi

PRJ="$HOME/fa3/flash-attention/hopper"

# ---- Minimal sanity check (double backticks only) ----
echo "Sanity-check: no *double* backticks in sources…"
if grep -RIn --include='*.{cu,cuh,hpp,h,cc,cpp}' '``' "$PRJ" >/dev/null; then
  echo "ERROR: suspicious double-backticks found:"
  grep -RIn --include='*.{cu,cuh,hpp,h,cc,cpp}' '``' "$PRJ" | sed -n '1,40p'
  exit 2
fi
echo "OK."

cd "$PRJ"

# ---- Hopper-only toggle: drop sm_80 gencodes during diagnosis ----
if [ "$FA3_HOPPER_ONLY" = "1" ]; then
  echo "Hopper-only mode: removing sm_80 gencodes from setup.py for this build…"
  cp setup.py setup.py.bak
  python - <<'PY'
import re, pathlib
p = pathlib.Path("setup.py")
s = p.read_text()
s = re.sub(r"('-gencode',\s*'arch=compute_80,code=sm_80',?)", "", s)
p.write_text(s)
print("sm_80 flags removed")
PY
fi

# ---- DIAGNOSTIC BUILD (in-place, not wheel) ----
python setup.py clean

LOG="$HOME/fa3/logs/hopper_build_${SLURM_JOB_ID:-manual}.log"
echo "Building in-place with verbose output…  (log: $LOG)"
(set -o pipefail; python setup.py build_ext --inplace -v 2>&1 | tee "$LOG")

echo "== FIRST ERRORS (if any) =="
grep -nE "error:|fatal error:|undefined reference|ninja: build stopped" "$LOG" || true

echo "== DONE DIAG BUILD =="
ls -lh ./*.so || true

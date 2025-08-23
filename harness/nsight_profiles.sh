#!/usr/bin/env bash
set -e
if ! command -v ncu >/dev/null 2>&1; then
  echo "Nsight Compute (ncu) not found in PATH. Skipping."
  exit 0
fi
ncu --target-processes all \
    --set full \
    --section "SpeedOfLight,MemoryWorkloadAnalysis,LaunchStatistics,SharedMemory" \
    python harness/microbench.py --M 512 --dk 128 --dv 128

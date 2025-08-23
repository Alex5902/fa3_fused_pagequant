# FA-3–Fused Page-Quant (FP8/INT8) — skeleton

This repo gives you a working scaffold to:
1) Quantize K/V per FA-3-aligned fragments (CPU/GPU safe),
2) Simulate separate vs fused dequant in Python,
3) Later wire the real fused path into FlashAttention-3 producer warpgroups.

## Quick start (local, free)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
python kv_formats/sanity_tile_check.py
python harness/unit_tests.py
python harness/microbench.py

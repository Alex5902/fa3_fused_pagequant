# harness/fused_dequant_ext.py
import torch
from torch.utils.cpp_extension import load

_fused = load(
    name="fused_dequant_producer",
    sources=["kernels/fused_dequant_producer.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

def dequant_tiles(q_int8, scales_fp16, zps_int8=None, frag_m=16, frag_k=64):
    M, K = q_int8.shape
    out = torch.empty((M, K), dtype=torch.float16, device=q_int8.device)
    _fused.dequant_tiles(q_int8, scales_fp16, zps_int8, frag_m, frag_k, out)
    return out

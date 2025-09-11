# harness/fa3_proto_ext.py
import torch
from torch.utils.cpp_extension import load

_fa3p = load(
    name="fa3_proto",
    sources=["kernels/fa3_proto.cu"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-gencode=arch=compute_86,code=sm_86",  # A6000
    ],
    verbose=False,
)

def proto_scores(Q_fp16, K_int8, sK_fp16, zK_int8=None, *, frag_m=16, frag_k=64, tile_q=0, tile_k=0):
    out = torch.empty((frag_m, frag_m), dtype=torch.float16, device=Q_fp16.device)
    _fa3p.proto_scores(Q_fp16, K_int8, sK_fp16, zK_int8, frag_m, frag_k, tile_q, tile_k, out)
    return out

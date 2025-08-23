import torch
from kv_formats.quantize_pages_fp8_int8 import (
    quantize_int8_per_frag, dequant_int8_per_frag,
    quantize_fp8_per_frag, dequant_fp8_per_frag
)

def attn_ref(q, k, v):
    # q: [M,D], k: [M,D], v: [M,Dv]
    scale = (q.shape[-1] ** -0.5)
    scores = (q @ k.transpose(-1, -2)) * scale
    p = torch.softmax(scores, dim=-1)
    return p @ v

def run_one(M=256, dk=128, dv=128, frag_M=16, frag_K=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    q = torch.randn(M, dk, dtype=torch.float16, device=device)
    k = torch.randn(M, dk, dtype=torch.float16, device=device)
    v = torch.randn(M, dv, dtype=torch.float16, device=device)

    out_ref = attn_ref(q, k, v).half()

    # INT8 per-frag
    kq, sk, zk = quantize_int8_per_frag(k, frag_M, frag_K)
    vq, sv, zv = quantize_int8_per_frag(v, frag_M, frag_K)
    k_i = dequant_int8_per_frag(kq, sk, zk, frag_M, frag_K)
    v_i = dequant_int8_per_frag(vq, sv, zv, frag_M, frag_K)
    out_i = attn_ref(q, k_i, v_i).half()

    # FP8-like per-frag
    kf_q, kf_s = quantize_fp8_per_frag(k, frag_M, frag_K)
    vf_q, vf_s = quantize_fp8_per_frag(v, frag_M, frag_K)
    k_f = dequant_fp8_per_frag(kf_q, kf_s, frag_M, frag_K)
    v_f = dequant_fp8_per_frag(vf_q, vf_s, frag_M, frag_K)
    out_f = attn_ref(q, k_f, v_f).half()

    mse_i = (out_i.float() - out_ref.float()).pow(2).mean().item()
    mse_f = (out_f.float() - out_ref.float()).pow(2).mean().item()
    print(f"Device={device}  M={M} dk={dk} dv={dv}")
    print(f"INT8 per-frag  MSE={mse_i:.3e}")
    print(f"FP8-like       MSE={mse_f:.3e}")

    # Loose demo threshold; tighten later.
    assert mse_i < 5e-2 and mse_f < 5e-2

if __name__ == "__main__":
    run_one()

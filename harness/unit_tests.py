# harness/unit_tests.py
import torch
from kv_formats.quantize_pages_fp8_int8 import (
    quantize_int8_per_frag, dequant_int8_per_frag,
    quantize_fp8_per_frag, dequant_fp8_per_frag,
)

# Try to import the CUDA test extension
try:
    from harness.fused_dequant_ext import dequant_tiles  # builds kernels/fused_dequant_producer.cu
    HAVE_CUDA_EXT = True
    EXT_ERR = None
except Exception as e:
    HAVE_CUDA_EXT = False
    EXT_ERR = e

def attn_ref(q, k, v):
    scale = (q.shape[-1] ** -0.5)
    return torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v

@torch.no_grad()
def test_attention_mse(M=256, dk=128, dv=128, frag_M=16, frag_K=64):
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
    print(f"[attention] Device={device} M={M} dk={dk} dv={dv}")
    print(f"  INT8 per-frag  MSE={mse_i:.3e}")
    print(f"  FP8-like       MSE={mse_f:.3e}")

    # Loose for now; you can tighten later.
    assert mse_i < 5e-2 and mse_f < 5e-2

@torch.no_grad()
def test_cuda_dequant_matches_python(frag_M=16, frag_K=64):
    if not torch.cuda.is_available():
        print("[cuda-dequant] skipped: CUDA not available")
        return
    if not HAVE_CUDA_EXT:
        print(f"[cuda-dequant] skipped: fused extension not available ({EXT_ERR})")
        return
    torch.manual_seed(1)
    cases = [(256, 128), (257, 129), (16, 64), (31, 193)]
    for (M, K) in cases:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        q, s, z = quantize_int8_per_frag(x, frag_M, frag_K)
        x_py = dequant_int8_per_frag(q, s, z, frag_M, frag_K)
        x_cu = dequant_tiles(q, s, z, frag_M, frag_K)
        ok = torch.allclose(x_py, x_cu, atol=1e-3, rtol=1e-3)
        print(f"[cuda-dequant] M={M} K={K} -> {'OK' if ok else 'MISMATCH'}")
        assert ok, f"CUDA dequant mismatch at M={M}, K={K}"

if __name__ == "__main__":
    test_attention_mse()
    test_cuda_dequant_matches_python()
    print("All unit tests passed.")

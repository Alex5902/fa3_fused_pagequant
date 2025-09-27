# file: /groups/gce51019/alex.ia/fa3/harness/fa3_correctness.py
import os
os.environ.setdefault("FA3_FORCE_FUSED_INT8KV", "1")  # ok if your build honors it

import torch
import fa3_cuda  # registers torch.ops.flash_attn_3.fwd

FRAG_M, FRAG_K = 16, 64

def info(name, t):
    print(f"{name:>12}: shape={tuple(t.shape)}, stride={t.stride()}, "
          f"dtype={t.dtype}, contig={t.is_contiguous()}, ptr=0x{t.data_ptr():x}")

def quantize_per_fragment_fp16_scales(x_fp16: torch.Tensor):
    """
    Quantize (B,H,S,D) to INT8 with per-(FRAG_M x FRAG_K) scales.
    IMPORTANT: return scales as FP16 (matches C++ __half*).
    """
    B,H,S,D = x_fp16.shape
    assert S % FRAG_M == 0 and D % FRAG_K == 0
    tiles = x_fp16.view(B,H, S//FRAG_M, FRAG_M, D//FRAG_K, FRAG_K)
    scales = (tiles.abs().amax(dim=(-1,-3)) / 127.0).clamp_min(1e-8)  # (B,H,S/FRAG_M,D/FRAG_K)
    q_tiles = torch.round(tiles / scales.unsqueeze(-1).unsqueeze(-3)).clamp(-128,127).to(torch.int8)
    x_q = q_tiles.view(B,H,S,D).contiguous()
    # return FP16 scales to match kernel's __half*
    return x_q, scales.contiguous().to(torch.float16)

def build_payload(q,k,v):
    B,H,S,D = q.shape
    nseq = S // FRAG_M
    ndim = D // FRAG_K
    BH = B*H

    kq, ks4d = quantize_per_fragment_fp16_scales(k)
    vq, vs4d = quantize_per_fragment_fp16_scales(v)

    # Use head-major 4D layout [B*H, nseq, ndim] with tight per-head stride
    ks = ks4d
    vs = vs4d
    kz = torch.zeros_like(ks, dtype=torch.int8).contiguous()
    vz = torch.zeros_like(vs, dtype=torch.int8).contiguous()

    print("\n--- Payload tensors ---")
    info("k_q", kq); info("v_q", vq)
    info("k_scales4d", ks); info("v_scales4d", vs)
    info("k_zps4d", kz); info("v_zps4d", vz)
    return kq, vq, ks, vs, kz, vz

def build_keyword_args(q, k, v, kq, vq, ks, vs, kz, vz):
    """
    Builds a dictionary of keyword arguments for the fwd pass.
    This is much more robust than positional arguments.
    """
    return {
        # Required inputs
        "q": q, "k": k, "v": v,

        # INT8 KV Cache inputs
        "k_q_": kq,
        "v_q_": vq,
        "k_scales_": ks,
        "v_scales_": vs,
        "k_zps_": kz,
        "v_zps_": vz,
        "frag_m_": FRAG_M,
        "frag_k_": FRAG_K,

        # Other attention parameters
        "is_causal": True,
        # Add any other parameters you might need, e.g., softmax_scale
        # "softmax_scale": 1.0 / (D**0.5)
    }

def main():
    device = "cuda"; dtype = torch.float16
    B,H,S,D = 1, 2, 1024, 128

    q = torch.randn(B,H,S,D, device=device, dtype=dtype).contiguous()
    k = torch.randn(B,H,S,D, device=device, dtype=dtype).contiguous()
    v = torch.randn(B,H,S,D, device=device, dtype=dtype).contiguous()

    # Baseline sanity
    _ = torch.ops.flash_attn_3.fwd(q, k, v, is_causal=True); torch.cuda.synchronize()
    print("✅ Baseline (FP16) OK")

    kq, vq, ks, vs, kz, vz = build_payload(q,k,v)

    print("\nCalling fused (positionals)…")
    args = build_keyword_args(q, k, v, kq, vq, ks, vs, kz, vz)

    print("--- Python: About to call torch.ops.flash_attn_3.fwd ---")
    out, _, _, _ = torch.ops.flash_attn_3.fwd(**args)
    # The program will hang on the line above, so the next line will not be printed.
    print("--- Python: torch.ops.flash_attn_3.fwd call has returned ---")
    
    torch.cuda.synchronize()
    print("✅ Fused INT8KV call returned without crash.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ fused failed:", e)


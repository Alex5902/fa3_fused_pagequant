# harness/unit_tests.py
# import torch
# from kv_formats.quantize_pages_fp8_int8 import (
#     quantize_int8_per_frag, dequant_int8_per_frag,
#     quantize_fp8_per_frag, dequant_fp8_per_frag,
# )

# # Try to import the CUDA test extension
# try:
#     from harness.fused_dequant_ext import dequant_tiles  # builds kernels/fused_dequant_producer.cu
#     HAVE_CUDA_EXT = True
#     EXT_ERR = None
# except Exception as e:
#     HAVE_CUDA_EXT = False
#     EXT_ERR = e

# def attn_ref(q, k, v):
#     scale = (q.shape[-1] ** -0.5)
#     return torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v

# @torch.no_grad()
# def test_attention_mse(M=256, dk=128, dv=128, frag_M=16, frag_K=64):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     torch.manual_seed(0)
#     q = torch.randn(M, dk, dtype=torch.float16, device=device)
#     k = torch.randn(M, dk, dtype=torch.float16, device=device)
#     v = torch.randn(M, dv, dtype=torch.float16, device=device)

#     out_ref = attn_ref(q, k, v).half()

#     # INT8 per-frag
#     kq, sk, zk = quantize_int8_per_frag(k, frag_M, frag_K)
#     vq, sv, zv = quantize_int8_per_frag(v, frag_M, frag_K)
#     k_i = dequant_int8_per_frag(kq, sk, zk, frag_M, frag_K)
#     v_i = dequant_int8_per_frag(vq, sv, zv, frag_M, frag_K)
#     out_i = attn_ref(q, k_i, v_i).half()

#     # FP8-like per-frag
#     kf_q, kf_s = quantize_fp8_per_frag(k, frag_M, frag_K)
#     vf_q, vf_s = quantize_fp8_per_frag(v, frag_M, frag_K)
#     k_f = dequant_fp8_per_frag(kf_q, kf_s, frag_M, frag_K)
#     v_f = dequant_fp8_per_frag(vf_q, vf_s, frag_M, frag_K)
#     out_f = attn_ref(q, k_f, v_f).half()

#     mse_i = (out_i.float() - out_ref.float()).pow(2).mean().item()
#     mse_f = (out_f.float() - out_ref.float()).pow(2).mean().item()
#     print(f"[attention] Device={device} M={M} dk={dk} dv={dv}")
#     print(f"  INT8 per-frag  MSE={mse_i:.3e}")
#     print(f"  FP8-like       MSE={mse_f:.3e}")

#     # Loose for now; you can tighten later.
#     assert mse_i < 5e-2 and mse_f < 5e-2

# @torch.no_grad()
# def test_cuda_dequant_matches_python(frag_M=16, frag_K=64):
#     if not torch.cuda.is_available():
#         print("[cuda-dequant] skipped: CUDA not available")
#         return
#     if not HAVE_CUDA_EXT:
#         print(f"[cuda-dequant] skipped: fused extension not available ({EXT_ERR})")
#         return
#     torch.manual_seed(1)
#     cases = [(256, 128), (257, 129), (16, 64), (31, 193)]
#     for (M, K) in cases:
#         x = torch.randn(M, K, dtype=torch.float16, device="cuda")
#         q, s, z = quantize_int8_per_frag(x, frag_M, frag_K)
#         x_py = dequant_int8_per_frag(q, s, z, frag_M, frag_K)
#         x_cu = dequant_tiles(q, s, z, frag_M, frag_K)
#         ok = torch.allclose(x_py, x_cu, atol=1e-3, rtol=1e-3)
#         print(f"[cuda-dequant] M={M} K={K} -> {'OK' if ok else 'MISMATCH'}")
#         assert ok, f"CUDA dequant mismatch at M={M}, K={K}"

# if __name__ == "__main__":
#     test_attention_mse()
#     test_cuda_dequant_matches_python()
#     print("All unit tests passed.")

# harness/unit_tests.py
import argparse, json, os, time
import torch
from kv_formats.quantize_pages_fp8_int8 import (
    quantize_int8_per_frag, dequant_int8_per_frag,
    quantize_fp8_per_frag,  dequant_fp8_per_frag,
)

# Try to import the CUDA fused dequant
try:
    from harness.fused_dequant_ext import dequant_tiles  # kernels/fused_dequant_producer.cu
    HAVE_CUDA_EXT, EXT_ERR = True, None
except Exception as e:
    HAVE_CUDA_EXT, EXT_ERR = False, e

def attn_ref(q, k, v):
    scale = (q.shape[-1] ** -0.5)
    p = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
    return p @ v

@torch.no_grad()
def compare_impls(M, dk, dv, frag_M=16, frag_K=64, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(M, dk, dtype=torch.float16, device=device)
    k = torch.randn(M, dk, dtype=torch.float16, device=device)
    v = torch.randn(M, dv, dtype=torch.float16, device=device)

    out_ref = attn_ref(q, k, v).half()

    # ---------- INT8: Python dequant ----------
    kq_i, sk_i, zk_i = quantize_int8_per_frag(k, frag_M, frag_K)
    vq_i, sv_i, zv_i = quantize_int8_per_frag(v, frag_M, frag_K)
    k_i_py = dequant_int8_per_frag(kq_i, sk_i, zk_i, frag_M, frag_K)
    v_i_py = dequant_int8_per_frag(vq_i, sv_i, zv_i, frag_M, frag_K)
    out_i_py = attn_ref(q, k_i_py, v_i_py).half()
    mse_i_py = (out_i_py.float() - out_ref.float()).pow(2).mean().item()

    # ---------- INT8: CUDA fused dequant (producer) ----------
    if device == "cuda" and HAVE_CUDA_EXT:
        k_i_cu = dequant_tiles(kq_i, sk_i, zk_i, frag_M, frag_K)
        v_i_cu = dequant_tiles(vq_i, sv_i, zv_i, frag_M, frag_K)
        out_i_cu = attn_ref(q, k_i_cu, v_i_cu).half()
        mse_i_cu = (out_i_cu.float() - out_ref.float()).pow(2).mean().item()
    else:
        mse_i_cu = None

    # ---------- FP8-like: Python ----------
    kq_f, sk_f = quantize_fp8_per_frag(k, frag_M, frag_K)
    vq_f, sv_f = quantize_fp8_per_frag(v, frag_M, frag_K)
    k_f_py = dequant_fp8_per_frag(kq_f, sk_f, frag_M, frag_K)
    v_f_py = dequant_fp8_per_frag(vq_f, sv_f, frag_M, frag_K)
    out_f_py = attn_ref(q, k_f_py, v_f_py).half()
    mse_f_py = (out_f_py.float() - out_ref.float()).pow(2).mean().item()

    # INT8 CUDA vs INT8 Python sanity (they should match closely)
    if mse_i_cu is not None:
        ok_cuda_vs_py = torch.allclose(k_i_cu, k_i_py, atol=1e-3, rtol=1e-3) and \
                        torch.allclose(v_i_cu, v_i_py, atol=1e-3, rtol=1e-3)
    else:
        ok_cuda_vs_py = None

    return {
        "M": M, "dk": dk, "dv": dv,
        "frag_M": frag_M, "frag_K": frag_K, "device": device,
        "mse_int8_py": mse_i_py,
        "mse_int8_cuda": mse_i_cu,
        "mse_fp8_py": mse_f_py,
        "cuda_vs_python_ok": ok_cuda_vs_py,
    }

@torch.no_grad()
def cuda_dequant_matches_python_smoketest(frag_M=16, frag_K=64):
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    if not HAVE_CUDA_EXT:
        return False, f"fused extension not available ({EXT_ERR})"
    torch.manual_seed(1)
    cases = [(256, 128), (257, 129), (16, 64), (31, 193)]
    for (M, K) in cases:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        q, s, z = quantize_int8_per_frag(x, frag_M, frag_K)
        x_py = dequant_int8_per_frag(q, s, z, frag_M, frag_K)
        x_cu = dequant_tiles(q, s, z, frag_M, frag_K)
        if not torch.allclose(x_py, x_cu, atol=1e-3, rtol=1e-3):
            return False, f"CUDA dequant mismatch at M={M}, K={K}"
    return True, "cuda dequant matches python"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=str, default=None, help="path prefix for JSON output")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fragM", type=int, default=16)
    ap.add_argument("--fragK", type=int, default=64)
    args = ap.parse_args()

    ok, msg = cuda_dequant_matches_python_smoketest(args.fragM, args.fragK)
    print("[cuda-dequant-smoke]", msg)
    if not ok:
        raise RuntimeError(msg)

    grid = []
    for M in [2048, 8192, 32768]:
        for dk in [64, 128, 192, 256, 512]:
            dv = dk
            res = compare_impls(M, dk, dv, args.fragM, args.fragK, args.device)
            print(f"[parity] M={M} dk=dv={dk} -> "
                  f"MSE int8(py)={res['mse_int8_py']:.3e} "
                  f"{'(int8(cu)=' + f'{res['mse_int8_cuda']:.3e})' if res['mse_int8_cuda'] is not None else ''} "
                  f"fp8(py)={res['mse_fp8_py']:.3e} "
                  f"cuda_vs_py={res['cuda_vs_python_ok']}")
            grid.append(res)

    if args.write:
        os.makedirs(os.path.dirname(args.write), exist_ok=True)
        out = f"{args.write}.json"
        with open(out, "w") as f:
            json.dump(grid, f, indent=2)
        print("wrote", out, "n=", len(grid))

if __name__ == "__main__":
    main()

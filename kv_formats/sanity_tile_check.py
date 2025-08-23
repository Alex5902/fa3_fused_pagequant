import torch
from kv_formats.quantize_pages_fp8_int8 import (
    quantize_int8_per_frag, quantize_fp8_per_frag,
    dequant_int8_per_frag, dequant_fp8_per_frag
)

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    M, K = 128, 128
    frag_M, frag_K = 16, 64
    x = torch.randn(M, K, dtype=torch.float16, device=device)

    # INT8
    qi, si, zi = quantize_int8_per_frag(x, frag_M, frag_K)
    xi = dequant_int8_per_frag(qi, si, zi, frag_M, frag_K)
    mse_i = (xi.float() - x.float()).pow(2).mean().item()

    # FP8-like
    qf, sf = quantize_fp8_per_frag(x, frag_M, frag_K)
    xf = dequant_fp8_per_frag(qf, sf, frag_M, frag_K)
    mse_f = (xf.float() - x.float()).pow(2).mean().item()

    print(f"Device: {device}")
    print("INT8 payload:", qi.shape, "scales:", si.shape, "MSE:", f"{mse_i:.3e}")
    print("FP8 payload:", qf.shape, "scales:", sf.shape, "MSE:", f"{mse_f:.3e}")

if __name__ == "__main__":
    run()

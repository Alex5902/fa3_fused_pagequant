import time, argparse, torch
from kv_formats.quantize_pages_fp8_int8 import (
    quantize_int8_per_frag, dequant_int8_per_frag
)

def attn(q,k,v):
    s = (q.shape[-1] ** -0.5)
    return torch.softmax(q @ k.T * s, dim=-1) @ v

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t0) / iters

def fused_sim(q, k_q, v_q, sk, sv, zk, zv, frag_M=16, frag_K=64):
    # Simulate "fused" by dequantizing fragments and accumulating (CPU/GPU friendly).
    M, dk = q.shape
    dv = v_q.shape[1]
    out = torch.zeros(M, dv, dtype=torch.float16, device=q.device)
    # Dequant full once per tile along M. (Close enough for microbench.)
    k = dequant_int8_per_frag(k_q, sk, zk, frag_M, frag_K)
    v = dequant_int8_per_frag(v_q, sv, zv, frag_M, frag_K)
    return attn(q, k, v)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--dk", type=int, default=128)
    p.add_argument("--dv", type=int, default=128)
    p.add_argument("--fragM", type=int, default=16)
    p.add_argument("--fragK", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    q = torch.randn(args.M, args.dk, dtype=torch.float16, device=device)
    k = torch.randn(args.M, args.dk, dtype=torch.float16, device=device)
    v = torch.randn(args.M, args.dv, dtype=torch.float16, device=device)

    # Baseline FP16
    tA = bench(lambda: attn(q, k, v))
    print(f"A) FP16 baseline: {tA*1e3:.2f} ms/iter")

    # Separate dequant (simulate quantized store + full dequant buffer)
    kq, sk, zk = quantize_int8_per_frag(k, args.fragM, args.fragK)
    vq, sv, zv = quantize_int8_per_frag(v, args.fragM, args.fragK)
    def sep():
        kd = dequant_int8_per_frag(kq, sk, zk, args.fragM, args.fragK)
        vd = dequant_int8_per_frag(vq, sv, zv, args.fragM, args.fragK)
        return attn(q, kd, vd)
    tB = bench(sep)
    print(f"B) Separate dequant: {tB*1e3:.2f} ms/iter")

    # "Fused" sim (still dequants, but structured to be easy to replace by kernel)
    tC = bench(lambda: fused_sim(q, kq, vq, sk, sv, zk, zv, args.fragM, args.fragK))
    print(f"C) Fused (sim): {tC*1e3:.2f} ms/iter")

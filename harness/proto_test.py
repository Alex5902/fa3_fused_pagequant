# harness/proto_test.py
import torch
from harness.fa3_proto_ext import proto_scores
from kv_formats.quantize_pages_fp8_int8 import quantize_int8_per_frag, dequant_int8_per_frag

@torch.no_grad()
def main(M=256, Dk=128, frag_m=16, frag_k=64):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    Q = torch.randn(M, Dk, dtype=torch.float16, device=dev)
    K = torch.randn(M, Dk, dtype=torch.float16, device=dev)

    # Quantize K per (mFrag, kFrag) to match kernel's scale indexing
    Kq, sK, zK = quantize_int8_per_frag(K, frag_m, frag_k)

    # >>> Dequantize with the same Python path the CUDA producer matches <<<
    Kd = dequant_int8_per_frag(Kq, sK, zK, frag_m, frag_k)  # fp16

    mF_seq  = (M  + frag_m - 1) // frag_m
    kF_feat = (Dk + frag_k - 1) // frag_k

    tests = [(0,0), (mF_seq-1, 0), (0, mF_seq-1)]
    for tile_q, tile_k in tests:
        # CUDA path (producer→SMEM→consumer)
        S_cu = proto_scores(Q, Kq, sK, zK,
                            frag_m=frag_m, frag_k=frag_k,
                            tile_q=tile_q, tile_k=tile_k).float()

        # Reference: Q · (Kd)^T over the same tile pair
        q_rows = slice(tile_q*frag_m, min(M, (tile_q+1)*frag_m))
        k_rows = slice(tile_k*frag_m, min(M, (tile_k+1)*frag_m))

        S_ref_full = (Q[q_rows].float() @ Kd[k_rows].float().T)
        S_ref = torch.zeros(frag_m, frag_m, dtype=torch.float32, device=dev)
        S_ref[:S_ref_full.shape[0], :S_ref_full.shape[1]] = S_ref_full

        ok = torch.allclose(S_cu, S_ref, atol=2e-2, rtol=1e-3)
        print(f"[proto] (tile_q={tile_q}, tile_k={tile_k}) -> "
              f"{'OK' if ok else 'MISMATCH'} | max_abs={(S_cu - S_ref).abs().max().item():.3e}")
        assert ok, "proto block mismatch"

    print("FA-3 proto test passed.")

if __name__ == "__main__":
    main()
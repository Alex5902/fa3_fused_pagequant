import torch
from typing import Tuple

def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _frag_view(x: torch.Tensor, frag_M: int, frag_K: int):
    """Pad and view x[M,K] as [mF, frag_M, kF, frag_K]."""
    assert x.ndim == 2, "expected [M, K]"
    M, K = x.shape
    padM = (frag_M - M % frag_M) % frag_M
    padK = (frag_K - K % frag_K) % frag_K
    if padM or padK:
        x = torch.nn.functional.pad(x, (0, padK, 0, padM))
    M2, K2 = x.shape
    return x.view(M2 // frag_M, frag_M, K2 // frag_K, frag_K), M2, K2

@torch.no_grad()
def quantize_int8_per_frag(
    x: torch.Tensor, frag_M: int = 16, frag_K: int = 64, symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    INT8 quant per (mFrag, kFrag). Returns (q[M,K] int8, scales[mF,kF] fp16, zps[mF,kF] int8).
    Dequant: x ~= (q - zp) * scale
    """
    x = x.to(torch.float32)
    fv, M2, K2 = _frag_view(x, frag_M, frag_K)  # [mF, frag_M, kF, frag_K]
    if symmetric:
        s = fv.abs().amax(dim=(1,3), keepdim=True).clamp(min=1e-8) / 127.0
        zp = torch.zeros_like(s)
        q = (fv / s).round().clamp(-127, 127).to(torch.int8)
    else:
        mn = fv.amin(dim=(1,3), keepdim=True)
        mx = fv.amax(dim=(1,3), keepdim=True)
        s = ((mx - mn) / 255.0).clamp(min=1e-8)
        zp = (-mn / s).round().clamp(0, 255)
        q = (fv / s + zp).round().clamp(0, 255).to(torch.int8)
    scales = s.squeeze(-1).squeeze(1).to(torch.float16).contiguous()  # [mF,kF]
    zps = zp.squeeze(-1).squeeze(1).to(torch.int16)                   # temp int16
    zps = zps.clamp(-128, 127).to(torch.int8).contiguous()
    return q.view(M2, K2).contiguous(), scales, zps

@torch.no_grad()
def quantize_fp8_per_frag(
    x: torch.Tensor, frag_M: int = 16, frag_K: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple FP8-like symmetric quant (we store an int8 payload and a scale).
    Range target ~127 similar to INT8, but labeled FP8 for later E4M3/E5M2 swap.
    Returns (q[M,K] int8, scales[mF,kF] fp16).
    Dequant: x ~= q * scale
    """
    x = x.to(torch.float32)
    fv, M2, K2 = _frag_view(x, frag_M, frag_K)
    s = fv.abs().amax(dim=(1,3), keepdim=True).clamp(min=1e-8) / 127.0
    q = (fv / s).round().clamp(-127, 127).to(torch.int8)
    scales = s.squeeze(-1).squeeze(1).to(torch.float16).contiguous()
    return q.view(M2, K2).contiguous(), scales

@torch.no_grad()
def dequant_int8_per_frag(q: torch.Tensor, scales: torch.Tensor, zps: torch.Tensor,
                          frag_M: int = 16, frag_K: int = 64) -> torch.Tensor:
    """Reconstruct fp16 from per-fragment INT8."""
    qv, M2, K2 = _frag_view(q.to(torch.int8), frag_M, frag_K)
    s = scales[:, None, :, None].to(torch.float32)  # broadcast to [mF,1,kF,1]
    zp = zps[:, None, :, None].to(torch.float32)
    x = (qv.to(torch.float32) - zp) * s
    return x.view(M2, K2).to(torch.float16).contiguous()

@torch.no_grad()
def dequant_fp8_per_frag(q: torch.Tensor, scales: torch.Tensor,
                         frag_M: int = 16, frag_K: int = 64) -> torch.Tensor:
    """Reconstruct fp16 from per-fragment FP8-like int8+scale representation."""
    qv, M2, K2 = _frag_view(q.to(torch.int8), frag_M, frag_K)
    s = scales[:, None, :, None].to(torch.float32)
    x = qv.to(torch.float32) * s
    return x.view(M2, K2).to(torch.float16).contiguous()

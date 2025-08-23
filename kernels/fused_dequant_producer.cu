#include <cuda.h>
#include <cuda_fp16.h>
#include "smem_layout.h"

// NOTE: This file is NOT a standalone program.
// These helpers are meant to be called from FA-3's producer warpgroups.

__device__ inline __half hmul(__half a, __half b) { return __hmul(a, b); }

__device__ void dequant_tile_to_smem_int8(
    const int8_t* __restrict__ q, const __half scale, const int8_t zp,
    __half* __restrict__ smem, int frag_M, int frag_K, int ld, int skew) {

  int t = threadIdx.x + blockDim.x * threadIdx.y;
  int stride = blockDim.x * blockDim.y;

  for (int i = t; i < frag_M * frag_K; i += stride) {
    int r = i / frag_K;
    int c = i % frag_K;
    int v = (int)q[i] - (int)zp;
    __half x = hmul(__int2half_rn(v), scale);
    uint32_t off = fa3_smem_offset(r, c, ld, skew);
    *reinterpret_cast<__half*>((char*)smem + off) = x;
  }
}

__device__ void dequant_tile_to_smem_fp8_like(
    const int8_t* __restrict__ q, const __half scale,
    __half* __restrict__ smem, int frag_M, int frag_K, int ld, int skew) {

  int t = threadIdx.x + blockDim.x * threadIdx.y;
  int stride = blockDim.x * blockDim.y;

  for (int i = t; i < frag_M * frag_K; i += stride) {
    int r = i / frag_K;
    int c = i % frag_K;
    __half x = hmul(__int2half_rn((int)q[i]), scale); // FP8-like (int8 + scale)
    uint32_t off = fa3_smem_offset(r, c, ld, skew);
    *reinterpret_cast<__half*>((char*)smem + off) = x;
  }
}

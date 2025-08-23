#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

// Replace this with FA-3's exact SMEM swizzle (row-skew/ldmatrix-friendly).
// Here it's a simple row-major + skew placeholder.
__device__ inline uint32_t fa3_smem_offset(int r, int c, int ld, int skew) {
  return static_cast<uint32_t>((r * (ld + skew) + c) * sizeof(__half));
}

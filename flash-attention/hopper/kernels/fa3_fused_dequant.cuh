#pragma once
#include <stdint.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "smem_layout.h"
#include <cuda_runtime.h>  

#define WARP_SIZE 32
__device__ __forceinline__ int fa3_lane_id(){ return threadIdx.x & (WARP_SIZE-1); }

// 16B aligned 4xint loader
struct __align__(16) fa3_int4x1 { int x, y, z, w; };

__device__ __forceinline__ fa3_int4x1 fa3_load16b(const int8_t* gptr) {
  return *reinterpret_cast<const fa3_int4x1*>(gptr);
}

// 16 int8 -> 16 Elem with scale + zero-point
template <typename Elem>
__device__ __forceinline__
void fa3_dequant_pack16_to_elem(const fa3_int4x1& v, float s, float zp, Elem* dst) {
  const signed char* p = reinterpret_cast<const signed char*>(&v);
  cutlass::NumericConverter<Elem, float> convert;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float f = (static_cast<int>(p[i]) - zp) * s;
    dst[i] = convert(f);
  }
}

// Overload for int4 (CUDA built-in)
template <typename Elem>
__device__ __forceinline__
void fa3_dequant_pack16_to_elem(const int4& v, float s, float zp, Elem* dst) {
  fa3_dequant_pack16_to_elem(*reinterpret_cast<const fa3_int4x1*>(&v), s, zp, dst);
}

// Back-compat shim (__half*)
__device__ __forceinline__
void fa3_dequant_pack16_to_half(const fa3_int4x1& v, float s, float zp, __half* dst) {
  fa3_dequant_pack16_to_elem<cutlass::half_t>(
      v, s, zp, reinterpret_cast<cutlass::half_t*>(dst));
}

// Tested producer from your harness: writes through FA-3 layout helper
template<int FRAG_M, int FRAG_K, typename Elem>
__device__ __forceinline__
void fa3_produce_tile_int8_to_smem(const int8_t* __restrict__ q_tile,
                                   int ldq,
                                   float s, float zp,
                                   Elem* __restrict__ smem_stage_base,
                                   int M_valid, int K_valid,
                                   bool row_is_v) {
  const int lane = fa3_lane_id();
  constexpr int VEC = 16;
  constexpr int STEPS_K = FRAG_K / VEC;
  cutlass::NumericConverter<Elem, float> convert;

  #pragma unroll
  for (int pass = 0; pass < ((FRAG_M + WARP_SIZE - 1) / WARP_SIZE); ++pass) {
    int r = pass * WARP_SIZE + lane;
    if (r >= FRAG_M) break;

    bool row_ok = (r < M_valid);
    const int8_t* row_g = row_ok ? (q_tile + r * ldq) : nullptr;

    #pragma unroll
    for (int sk = 0; sk < STEPS_K; ++sk) {
      int col = sk * VEC;
      Elem tmp[VEC];

      if (row_ok && col < K_valid) {
        int remain = K_valid - col;
        if (remain >= VEC && (((reinterpret_cast<uintptr_t>(row_g + col)) & 0xF) == 0)) {
          int4 v = *reinterpret_cast<const int4*>(row_g + col);
          fa3_dequant_pack16_to_elem<Elem>(v, s, zp, tmp);
        } else {
          #pragma unroll
          for (int i = 0; i < VEC; ++i) {
            float qf = (i < remain) ? static_cast<float>(row_g[col + i]) : 0.f;
            tmp[i] = convert((qf - zp) * s);
          }
        }
      } else {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) tmp[i] = convert(0.f);
      }

      fa3_smem_store16_interleaved_tile<FRAG_M, FRAG_K>(
          smem_stage_base, /*r=*/r, /*c_start=*/col, tmp, /*row_is_v=*/row_is_v);
    }
  }
}

// Legacy wrapper to __half*
template<int FRAG_M, int FRAG_K>
__device__ __forceinline__
void fa3_produce_tile_int8_to_smem(const int8_t* __restrict__ q_tile,
                                   int ldq, float s, float zp,
                                   __half* __restrict__ smem_stage_base,
                                   int M_valid, int K_valid, bool row_is_v) {
  fa3_produce_tile_int8_to_smem<FRAG_M, FRAG_K, cutlass::half_t>(
      q_tile, ldq, s, zp,
      reinterpret_cast<cutlass::half_t*>(smem_stage_base),
      M_valid, K_valid, row_is_v);
}

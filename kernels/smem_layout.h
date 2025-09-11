#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

// -------- Architecture-dependent interleave (compile-time for device; default 8 on host) --------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  static constexpr int FA3_INTERLEAVE = 16;   // SM90 (Hopper): 16-wide is WGMMA-friendly
#else
  static constexpr int FA3_INTERLEAVE = 8;    // SM80/SM86 (Ampere): 8-wide is ldmatrix-friendly
#endif

// -----------------------------
// Baseline row-major (+ optional skew) in BYTES (debug/reference)
// -----------------------------
__device__ __forceinline__
uint32_t fa3_smem_offset_rowmajor_bytes(int r, int c, int ld, int skew = 0) {
  return static_cast<uint32_t>((r * (ld + skew) + c) * int(sizeof(__half)));
}

// -----------------------------
// Row-major INTERLEAVED layout (BYTES):
//   columns are partitioned into groups of I (8/16), then [r,I] are the inner coords.
//   idx_elems = group_c * (ld * I) + r * I + inner_c
// -----------------------------
template<int I = FA3_INTERLEAVE>
__device__ __forceinline__
uint32_t fa3_smem_offset_rowmajor_interleaved_bytes(int r, int c, int ld) {
  int group = c / I;
  int inner = c - group * I;      // c % I
  int idx   = group * (ld * I) + r * I + inner;
  return static_cast<uint32_t>(idx * int(sizeof(__half)));
}

// -----------------------------
// Col-major INTERLEAVED layout (BYTES):
//   rows are partitioned into groups of I, then [I,ld] are inner coords.
//   idx_elems = group_r * (ld * I) + inner_r * ld + c
// -----------------------------
template<int I = FA3_INTERLEAVE>
__device__ __forceinline__
uint32_t fa3_smem_offset_colmajor_interleaved_bytes(int r, int c, int ld) {
  int group = r / I;
  int inner = r - group * I;      // r % I
  int idx   = group * (ld * I) + inner * ld + c;
  return static_cast<uint32_t>(idx * int(sizeof(__half)));
}

// -----------------------------
// K/V wrappers (BYTES)
// -----------------------------
__device__ __forceinline__
uint32_t fa3_smem_offset_k_bytes(int r, int c, int ld) { // "B-like" for QK^T
  return fa3_smem_offset_colmajor_interleaved_bytes<>(r, c, ld);
}
__device__ __forceinline__
uint32_t fa3_smem_offset_v_bytes(int r, int c, int ld) { // "A-like" for P·V
  return fa3_smem_offset_rowmajor_interleaved_bytes<>(r, c, ld);
}

// Tiny helper so we don't need a device lambda
__device__ __forceinline__
uint32_t fa3_map_bytes(bool row_is_v, int r, int c, int ld) {
  return row_is_v ? fa3_smem_offset_v_bytes(r, c, ld)
                  : fa3_smem_offset_k_bytes(r, c, ld);
}

// -----------------------------
// Store 16 halfs to SMEM with interleaved layout
//   smem_base: base __half* of the tile
//   ld       : logical leading dimension of the tile
//   r        : row in the tile
//   c_start  : starting column (ideally multiple of 16)
//   vals[16] : values in registers
//   row_is_v : true=>use V (row-major interleaved), false=>use K (col-major interleaved)
// -----------------------------
__device__ __forceinline__
void fa3_smem_store16_interleaved(__half* smem_base, int ld, int r, int c_start,
                                  const __half vals[16], bool row_is_v)
{
  constexpr int I = FA3_INTERLEAVE;
  int c0 = c_start;
  int remain = 16;
  int src = 0;

  while (remain > 0) {
    int inner = c0 % I;
    int cap   = I - inner;
    int take  = (cap < remain ? cap : remain);

    // Base pointer for (r, c0)
    uint32_t byte_off0 = fa3_map_bytes(row_is_v, r, c0, ld);
    __half* dst0 = reinterpret_cast<__half*>(
        reinterpret_cast<char*>(smem_base) + byte_off0);

    #pragma unroll
    for (int i = 0; i < take; ++i) {
      dst0[i] = vals[src + i];   // contiguous within the interleave group
    }

    c0    += take;
    src   += take;
    remain -= take;
  }
}

// ---- TILE-LOCAL interleaved layouts (byte offsets) ----
// Row-major interleave: groups of I along columns, stride = FRAG_M
template<int FRAG_M, int FRAG_K, int I = FA3_INTERLEAVE>
__device__ __forceinline__
uint32_t fa3_smem_offset_rowmajor_interleaved_bytes_tile(int r, int c) {
  int group = c / I;
  int inner = c - group * I;
  int idx   = group * (FRAG_M * I) + r * I + inner;
  return static_cast<uint32_t>(idx * int(sizeof(__half)));
}

// Col-major interleave: groups of I along rows, stride = FRAG_K
template<int FRAG_M, int FRAG_K, int I = FA3_INTERLEAVE>
__device__ __forceinline__
uint32_t fa3_smem_offset_colmajor_interleaved_bytes_tile(int r, int c) {
  int group = r / I;
  int inner = r - group * I;
  int idx   = group * (FRAG_K * I) + inner * FRAG_K + c;
  return static_cast<uint32_t>(idx * int(sizeof(__half)));
}

// Wrappers for K/V (tile-local)
template<int FRAG_M, int FRAG_K, int I = FA3_INTERLEAVE>
__device__ __forceinline__
uint32_t fa3_smem_offset_k_bytes_tile(int r, int c) { // K: "B-like" for QK^T
  return fa3_smem_offset_colmajor_interleaved_bytes_tile<FRAG_M,FRAG_K,I>(r, c);
}
template<int FRAG_M, int FRAG_K, int I = FA3_INTERLEAVE>
__device__ __forceinline__
uint32_t fa3_smem_offset_v_bytes_tile(int r, int c) { // V: "A-like" for P·V
  return fa3_smem_offset_rowmajor_interleaved_bytes_tile<FRAG_M,FRAG_K,I>(r, c);
}

// Store 16 halfs using tile-local mapping
template<int FRAG_M, int FRAG_K, int I = FA3_INTERLEAVE>
__device__ __forceinline__
void fa3_smem_store16_interleaved_tile(__half* smem_base, int r, int c_start,
                                       const __half vals[16], bool row_is_v)
{
  int c0 = c_start, remain = 16, src = 0;
  while (remain > 0) {
    int inner = c0 % I;
    int take  = min(I - inner, remain);

    uint32_t byte_off0 = row_is_v
        ? fa3_smem_offset_v_bytes_tile<FRAG_M,FRAG_K,I>(r, c0)
        : fa3_smem_offset_k_bytes_tile<FRAG_M,FRAG_K,I>(r, c0);

    __half* dst0 = reinterpret_cast<__half*>(
        reinterpret_cast<char*>(smem_base) + byte_off0);

    #pragma unroll
    for (int i = 0; i < take; ++i) dst0[i] = vals[src + i];

    c0 += take; src += take; remain -= take;
  }
}


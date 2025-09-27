#pragma once
#include <cuda_runtime.h> 
#include <cuda_fp16.h>
#include <cuda_bf16.h>  
#include <stdint.h>

// -------- Architecture-dependent interleave (compile-time for device; default 8 on host) --------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  #define FA3_DEV_INTERLEAVE 16
#else
  #define FA3_DEV_INTERLEAVE 8
#endif

// -----------------------------
// Baseline row-major (+ optional skew) in BYTES (debug/reference)
// -----------------------------
template <typename T>
__device__ __forceinline__
uint32_t fa3_smem_offset_rowmajor_bytes(int r, int c, int ld, int skew = 0) {
  return static_cast<uint32_t>((r * (ld + skew) + c) * int(sizeof(T)));
}

// -----------------------------
// Row-major INTERLEAVED layout (BYTES):
//   columns are partitioned into groups of I (8/16), then [r,I] are the inner coords.
//   idx_elems = group_c * (ld * I) + r * I + inner_c
// -----------------------------
template<typename T, int I>
__device__ __forceinline__
uint32_t fa3_smem_offset_rowmajor_interleaved_bytes(int r, int c, int ld) {
  int group = c / I;
  int inner = c - group * I;      // c % I
  int idx   = group * (ld * I) + r * I + inner;
  return static_cast<uint32_t>(idx * int(sizeof(T)));
}

// -----------------------------
// Col-major INTERLEAVED layout (BYTES):
//   rows are partitioned into groups of I, then [I,ld] are inner coords.
//   idx_elems = group_r * (ld * I) + inner_r * ld + c
// -----------------------------
template<typename T, int I>
__device__ __forceinline__
uint32_t fa3_smem_offset_colmajor_interleaved_bytes(int r, int c, int ld) {
  int group = r / I;
  int inner = r - group * I;      // r % I
  int idx   = group * (ld * I) + inner * ld + c;
  return static_cast<uint32_t>(idx * int(sizeof(T)));
}

// -----------------------------
// K/V wrappers (BYTES)
// -----------------------------
template <typename T, int I> __device__ __forceinline__
uint32_t fa3_smem_offset_k_bytes(int r, int c, int ld) {
  return fa3_smem_offset_colmajor_interleaved_bytes<T,I>(r, c, ld);
}
template <typename T, int I> __device__ __forceinline__
uint32_t fa3_smem_offset_v_bytes(int r, int c, int ld) {
  return fa3_smem_offset_rowmajor_interleaved_bytes<T,I>(r, c, ld);
}

// Tiny helper so we don't need a device lambda
template <typename T, int I>
__device__ __forceinline__
uint32_t fa3_map_bytes(bool row_is_v, int r, int c, int ld) {
  return row_is_v ? fa3_smem_offset_v_bytes<T,I>(r, c, ld)
                  : fa3_smem_offset_k_bytes<T,I>(r, c, ld);
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
template <typename T, int I>
__device__ __forceinline__
void fa3_smem_store16_interleaved_impl(T* smem_base, int ld, int r, int c_start,
                                       const T (&vals)[16], bool row_is_v) {
  int c0 = c_start;
  int remain = 16;
  int src = 0;

  while (remain > 0) {
    int inner = c0 % I;
    int cap   = I - inner;
    int take  = (cap < remain ? cap : remain);

    // Base pointer for (r, c0)
    uint32_t byte_off0 = fa3_map_bytes<T,I>(row_is_v, r, c0, ld);
    T* dst0 = reinterpret_cast<T*>(
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

// Safe wrapper that picks I per device arch at *call* site (avoids default-template trap)
template <typename T>
__device__ __forceinline__
void fa3_smem_store16_interleaved(T* smem_base, int ld, int r, int c_start,
                                  const T (&vals)[16], bool row_is_v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  fa3_smem_store16_interleaved_impl<T,16>(smem_base, ld, r, c_start, vals, row_is_v);
#else
  fa3_smem_store16_interleaved_impl<T,8 >(smem_base, ld, r, c_start, vals, row_is_v);
#endif
}

// Pointer form (if caller has T* instead of T[16])
template <typename T>
__device__ __forceinline__
void fa3_smem_store16_interleaved(T* smem_base, int ld, int r, int c_start,
                                  const T* vals_ptr, bool row_is_v) {
  T tmp[16];
  #pragma unroll
  for (int i=0;i<16;++i) tmp[i] = vals_ptr[i];
  fa3_smem_store16_interleaved(smem_base, ld, r, c_start, tmp, row_is_v);
}

// ---- TILE-LOCAL interleaved layouts (byte offsets) ----
// Row-major interleave: groups of I along columns, stride = FRAG_M
template<typename T, int FRAG_M, int FRAG_K, int I>
__device__ __forceinline__
uint32_t fa3_smem_offset_rowmajor_interleaved_bytes_tile(int r, int c) {
  int group = c / I;
  int inner = c - group * I;
  int idx   = group * (FRAG_M * I) + r * I + inner;
  return static_cast<uint32_t>(idx * int(sizeof(T)));
}

// Col-major interleave: groups of I along rows, stride = FRAG_K
template<typename T, int FRAG_M, int FRAG_K, int I>
__device__ __forceinline__
uint32_t fa3_smem_offset_colmajor_interleaved_bytes_tile(int r, int c) {
  int group = r / I;
  int inner = r - group * I;
  int idx   = group * (FRAG_K * I) + inner * FRAG_K + c;
  return static_cast<uint32_t>(idx * int(sizeof(T)));
}

// Wrappers for K/V (tile-local)
template<typename T, int FRAG_M, int FRAG_K, int I>
__device__ __forceinline__
uint32_t fa3_smem_offset_k_bytes_tile(int r, int c) { // K: "B-like" for QK^T
  return fa3_smem_offset_colmajor_interleaved_bytes_tile<T,FRAG_M,FRAG_K,I>(r, c);
}
template<typename T, int FRAG_M, int FRAG_K, int I>
__device__ __forceinline__
uint32_t fa3_smem_offset_v_bytes_tile(int r, int c) { // V: "A-like" for P·V
  return fa3_smem_offset_rowmajor_interleaved_bytes_tile<T,FRAG_M,FRAG_K,I>(r, c);
}

// Store 16 halfs using tile-local mapping
template<typename T, int FRAG_M, int FRAG_K, int I>
__device__ __forceinline__
void fa3_smem_store16_interleaved_tile(T* smem_base, int r, int c_start,
                                       const T (&vals)[16], bool row_is_v)
{
  int c0 = c_start, remain = 16, src = 0;
  while (remain > 0) {
    int inner = c0 % I;
    // int take  = min(I - inner, remain);
    int take = (I - inner < remain ? I - inner : remain);

    uint32_t byte_off0 = row_is_v
        ? fa3_smem_offset_v_bytes_tile<T,FRAG_M,FRAG_K,I>(r, c0)
        : fa3_smem_offset_k_bytes_tile<T,FRAG_M,FRAG_K,I>(r, c0);
    T* dst0 = reinterpret_cast<T*>(
        reinterpret_cast<char*>(smem_base) + byte_off0);

    #pragma unroll
    for (int i = 0; i < take; ++i) dst0[i] = vals[src + i];

    c0 += take; src += take; remain -= take;
  }
}

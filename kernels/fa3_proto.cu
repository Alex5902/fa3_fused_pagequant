// kernels/fa3_proto.cu
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "smem_layout.h"

#define WARP_SIZE 32
__device__ __forceinline__ int lane_id(){ return threadIdx.x & (WARP_SIZE-1); }

// ---------------------------
// Minimal helpers (match your fused producer math)
// ---------------------------
struct int4x1 { int x, y, z, w; };

__device__ __forceinline__ void load_16b(const int8_t* gptr, int4x1 &v) {
  v = *reinterpret_cast<const int4x1*>(gptr);
}

__device__ __forceinline__
void dequant_pack16_to_half(const int4x1 &v, float s, float zp, __half *dst) {
  const signed char* p = reinterpret_cast<const signed char*>(&v);
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float f = (static_cast<int>(p[i]) - zp) * s;
    dst[i] = __float2half_rn(f);
  }
}

// ---------------------------
// Fused producer (INT8 -> FP16 tile in SMEM), edge-safe, tile-local mapping
// ---------------------------
template<int FRAG_M, int FRAG_K>
__device__ __forceinline__
void produce_tile_int8_to_smem(const int8_t* __restrict__ q_tile,
                               int ldq,
                               float s, float zp,
                               __half* __restrict__ smem,
                               int /*unused_stride*/,
                               int M_valid, int K_valid,
                               bool row_is_v) {
  const int lane = lane_id();
  constexpr int VEC = 16;
  constexpr int STEPS_K = FRAG_K / VEC;

  #pragma unroll
  for (int pass = 0; pass < ((FRAG_M + WARP_SIZE - 1) / WARP_SIZE); ++pass) {
    int r = pass * WARP_SIZE + lane;
    if (r >= FRAG_M) break;

    bool row_ok = (r < M_valid);
    const int8_t* row_g = row_ok ? (q_tile + r * ldq) : nullptr;

    #pragma unroll
    for (int sk = 0; sk < STEPS_K; ++sk) {
      int col = sk * VEC;
      __half tmp[VEC];

      if (row_ok && col < K_valid) {
        int remain = K_valid - col;
        bool can_vec = (remain >= VEC) &&
                       (((reinterpret_cast<uintptr_t>(row_g + col)) & 0xF) == 0);
        if (can_vec) {
          int4x1 v16; load_16b(row_g + col, v16);
          dequant_pack16_to_half(v16, s, zp, tmp);
        } else {
          #pragma unroll
          for (int i = 0; i < VEC; ++i) {
            float qf = (i < remain) ? static_cast<float>(row_g[col + i]) : 0.f;
            tmp[i] = __float2half_rn((qf - zp) * s);
          }
        }
      } else {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) tmp[i] = __float2half_rn(0.f);
      }

      fa3_smem_store16_interleaved_tile<FRAG_M, FRAG_K>(
          /*smem_base=*/smem, /*r=*/r, /*c_start=*/col, /*vals=*/tmp,
          /*row_is_v=*/row_is_v);
    }
  }
}

// ---------------------------
// Kernel: build one Scores block S[q_rows, k_rows] = Q[q]*K[k]^T
// Uses fused producer to stream K tile into SMEM with K-layout,
// then consumers dot Q rows against K rows across the feature dim in FRAG_K chunks.
// ---------------------------
template<int FRAG_M, int FRAG_K>
__global__ void k_proto_scores(
    const __half* __restrict__ Q,     // [M, Dk]
    const int8_t* __restrict__ Kq,    // [M, Dk] int8
    const __half* __restrict__ sK,    // [mF_seq, kF_feat] per-tile scale
    const int8_t* __restrict__ zK,    // [mF_seq, kF_feat] per-tile zp (or nullptr)
    __half* __restrict__ Sblk,        // [FRAG_M, FRAG_M] output block (row-major)
    int M, int Dk,
    int tile_q, int tile_k)
{
  const int mF_seq  = (M  + FRAG_M - 1) / FRAG_M;
  const int kF_feat = (Dk + FRAG_K - 1) / FRAG_K;

  const int q_row0 = tile_q * FRAG_M;
  const int k_row0 = tile_k * FRAG_M;

  // shared K tile
  extern __shared__ __half smem[];
  __half* smemK = smem; // FRAG_M x FRAG_K

  // Each lane owns rows: r = lane, lane+32, ...
  const int lane = lane_id();
  float acc_row[FRAG_M]; // one row vs all k-rows
  #pragma unroll
  for (int j = 0; j < FRAG_M; ++j) acc_row[j] = 0.f;

  // Loop over feature tiles (Dk in chunks of FRAG_K)
  for (int t = 0; t < kF_feat; ++t) {
    int c0 = t * FRAG_K;
    int Dk_valid = max(0, min(FRAG_K, Dk - c0));
    int M_valid  = max(0, min(FRAG_M, M  - k_row0));

    // Producer: dequantize K[k_rows, c0:c0+FRAG_K] into SMEM (K layout)
    if (threadIdx.x < WARP_SIZE) {
      const int idx_sk = tile_k * kF_feat + t; // (mFrag_seq, kFrag_feat)
      float s  = __half2float(sK[idx_sk]);
      float zp = zK ? float(zK[idx_sk]) : 0.f;
      const int8_t* K_tile = Kq + k_row0 * Dk + c0;
      produce_tile_int8_to_smem<FRAG_M, FRAG_K>(
          K_tile, Dk, s, zp, smemK, FRAG_K, M_valid, Dk_valid, /*row_is_v=*/false);
    }
    __syncthreads();

    // Consumers: for our assigned Q row(s), dot against each of the FRAG_M K rows in SMEM
    for (int r = lane; r < FRAG_M; r += WARP_SIZE) {
      if (q_row0 + r >= M) continue;
      const __half* qrow = Q + (q_row0 + r) * Dk + c0;

      #pragma unroll
      for (int j = 0; j < FRAG_M; ++j) {
        if (k_row0 + j >= M) continue;
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < FRAG_K; ++c) {
          if (c >= Dk_valid) break;
          // K_smem[j, c] with K (col-major interleaved) mapping
          uint32_t offK = fa3_smem_offset_k_bytes_tile<FRAG_M, FRAG_K>(j, c);
          __half Kj = *reinterpret_cast<const __half*>(
            reinterpret_cast<const char*>(smemK) + offK);
          acc += __half2float(qrow[c]) * __half2float(Kj);
        }
        acc_row[j] += acc;
      }
    }
    __syncthreads();
  }

  // Write our rows of the Scores block (no softmax here; this is just Q·Kᵀ for this tile pair)
  for (int r = lane; r < FRAG_M; r += WARP_SIZE) {
    if (q_row0 + r >= M) continue;
    __half* out_row = Sblk + r * FRAG_M;
    #pragma unroll
    for (int j = 0; j < FRAG_M; ++j) {
      if (k_row0 + j >= M) continue;
      out_row[j] = __float2half(acc_row[j]);
    }
  }
}

// ---------------------------
// Launcher / pybind
// ---------------------------
static void launch_proto_scores(at::Tensor Q, at::Tensor Kq,
                                at::Tensor sK, c10::optional<at::Tensor> zK_opt,
                                int frag_m, int frag_k,
                                int tile_q, int tile_k,
                                at::Tensor Sblk) {
  TORCH_CHECK(Q.is_cuda() && Kq.is_cuda() && Sblk.is_cuda(), "cuda tensors required");
  TORCH_CHECK(Q.dtype() == at::kHalf, "Q must be fp16");
  TORCH_CHECK(Kq.dtype() == at::kChar, "Kq must be int8");
  TORCH_CHECK(sK.is_cuda() && sK.dtype() == at::kHalf, "sK must be fp16 cuda");
  TORCH_CHECK(Sblk.dtype() == at::kHalf &&
              Sblk.size(0) == frag_m && Sblk.size(1) == frag_m,
              "Sblk must be [frag_m, frag_m] fp16");

  const int M  = Q.size(0);
  const int Dk = Q.size(1);

  const int mF_seq  = (M  + frag_m - 1) / frag_m;
  const int kF_feat = (Dk + frag_k - 1) / frag_k;
  TORCH_CHECK(sK.numel() == mF_seq * kF_feat, "sK shape mismatch");

  const int8_t* zK_ptr = nullptr;
  at::Tensor zK;
  if (zK_opt.has_value()) {
    zK = *zK_opt;
    TORCH_CHECK(zK.is_cuda() && zK.dtype() == at::kChar, "zK must be int8 cuda");
    TORCH_CHECK(zK.numel() == mF_seq * kF_feat, "zK shape mismatch");
    zK_ptr = zK.data_ptr<int8_t>();
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(1,1,1);
  dim3 block(WARP_SIZE,1,1);
  const int smem_bytes = frag_m * frag_k * sizeof(__half); // K tile only

  if (frag_m == 16 && frag_k == 64) {
    k_proto_scores<16,64><<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
      Kq.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(sK.data_ptr<at::Half>()),
      zK_ptr,
      reinterpret_cast<__half*>(Sblk.data_ptr<at::Half>()),
      M, Dk, tile_q, tile_k);
  } else if (frag_m == 16 && frag_k == 128) {
    k_proto_scores<16,128><<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
      Kq.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(sK.data_ptr<at::Half>()),
      zK_ptr,
      reinterpret_cast<__half*>(Sblk.data_ptr<at::Half>()),
      M, Dk, tile_q, tile_k);
  } else {
    TORCH_CHECK(false, "Unsupported (frag_m, frag_k)");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("proto_scores", &launch_proto_scores,
        "FA3 proto: fused-producer K tile + Q·K^T block (no softmax)");
}

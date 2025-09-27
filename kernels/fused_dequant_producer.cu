// // kernels/fused_dequant_producer.cu
// #include <cuda.h>
// #include <cuda_fp16.h>
// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include "smem_layout.h"

// #define WARP_SIZE 32

// // ---------------------------
// // 1) Low-level helpers
// // ---------------------------

// __device__ __forceinline__ int lane_id() {
//   return threadIdx.x & (WARP_SIZE - 1);
// }

// template<int FRAG_M, int FRAG_K>
// __device__ __forceinline__ int smem_offset_rowmajor(int r, int c, int smem_stride) {
//   // Default row-major store to shared; replace with your FA-3 swizzle if desired.
//   return r * smem_stride + c;
// }

// // Vectorized int8 load of 16 bytes (one int4)
// struct int4x1 { int x, y, z, w; };

// __device__ __forceinline__ void load_16b(const int8_t* gptr, int4x1 &v) {
//   v = *reinterpret_cast<const int4x1*>(gptr); // aligned 16B assumed if you align your pages
// }

// // Unpack 16 int8s from int4x1 into 16 floats (sign-extended), convert+scale → 16 halfs.
// __device__ __forceinline__
// void dequant_pack16_to_half(const int4x1 &v, float s, float zp, __half *dst) {
//   const signed char* p = reinterpret_cast<const signed char*>(&v);
//   #pragma unroll
//   for (int i = 0; i < 16; ++i) {
//     float f = (static_cast<int>(p[i]) - zp) * s;
//     dst[i] = __float2half_rn(f);
//   }
// }

// // ---------------------------
// // 2) Device micro-kernel:
// //    INT8 per-frag → FP16 tiles to SMEM
// // ---------------------------
// //
// // One warp produces exactly one (FRAG_M x FRAG_K) tile.
// // - q_tile:  base of the quantized tile (global memory)
// // - ldq:     leading dim (K) of q in elements
// // - s:       scale for this (mFrag, kFrag) as float (or half->float)
// // - zp:      zero-point for this (mFrag, kFrag) (0 for symmetric)
// // - smem:    base of SMEM tile
// // - smem_stride: stride in elements (K-side) inside SMEM
// //
// // Store layout: row-major by default; replace smem_offset_rowmajor with your FA-3 swizzle.
// //
// template<int FRAG_M, int FRAG_K>
// __device__ __forceinline__
// void produce_tile_int8_to_smem(const int8_t* __restrict__ q_tile,
//                                int ldq,
//                                float s, float zp,
//                                __half* __restrict__ smem,
//                                int smem_stride,
//                                int M_valid, int K_valid,
//                                bool row_is_v) {
//   const int lane = lane_id();

//   constexpr int VEC = 16;                // 16 int8 per vector step
//   constexpr int STEPS_K = FRAG_K / VEC;  // number of 16B chunks per row

//   #pragma unroll
//   for (int pass = 0; pass < ((FRAG_M + WARP_SIZE - 1) / WARP_SIZE); ++pass) {
//     int r = pass * WARP_SIZE + lane;
//     if (r >= FRAG_M) break;

//     bool row_ok = (r < M_valid);
//     const int8_t* row_g = row_ok ? (q_tile + r * ldq) : nullptr;

//     #pragma unroll
//     for (int sk = 0; sk < STEPS_K; ++sk) {
//       int col = sk * VEC;

//       __half tmp[VEC];

//       if (row_ok && col < K_valid) {
//         int remain = K_valid - col;

//         // Use fast 16B vector load only if we have 16 valid elems AND the address is 16B-aligned
//         bool can_vec = (remain >= VEC) &&
//                        (((reinterpret_cast<uintptr_t>(row_g + col)) & 0xF) == 0);

//         if (can_vec) {
//           int4x1 v16;
//           load_16b(row_g + col, v16);
//           dequant_pack16_to_half(v16, s, zp, tmp);
//         } else {
//           // Tail or misaligned: scalar fallback + zero-pad
//           #pragma unroll
//           for (int i = 0; i < VEC; ++i) {
//             float qf = (i < remain) ? static_cast<float>(row_g[col + i]) : 0.f;
//             tmp[i] = __float2half_rn((qf - zp) * s);
//           }
//         }
//       } else {
//         // Row or col out of range: write zeros
//         #pragma unroll
//         for (int i = 0; i < VEC; ++i) tmp[i] = __float2half_rn(0.f);
//       }

//       // Interleaved store into SMEM: row_is_v=true => V layout; false => K layout
//       fa3_smem_store16_interleaved_tile<FRAG_M, FRAG_K>(
//           /*smem_base=*/smem,
//           /*r=*/r,
//           /*c_start=*/col,
//           /*vals=*/tmp,
//           /*row_is_v=*/row_is_v);
//     }
//   }
// }

// // ---------------------------
// // 3) Standalone kernel for testing:
// //    Converts entire [M,K] in tiles and writes to global out[M,K]
// // ---------------------------

// template<int FRAG_M, int FRAG_K>
// __global__ void k_dequant_tiles_rowmajor(
//     const int8_t* __restrict__ q,
//     const __half* __restrict__ scales,
//     const int8_t* __restrict__ zps,
//     __half* __restrict__ out,
//     int M, int K)
// {
//   const int mF = (M + FRAG_M - 1) / FRAG_M;
//   const int kF = (K + FRAG_K - 1) / FRAG_K;

//   const int tile_m = blockIdx.y;
//   const int tile_k = blockIdx.x;
//   if (tile_m >= mF || tile_k >= kF) return;

//   const int row0 = tile_m * FRAG_M;
//   const int col0 = tile_k * FRAG_K;

//   // Valid extents inside this tile
//   int M_valid = max(0, min(FRAG_M, M - row0));
//   int K_valid = max(0, min(FRAG_K, K - col0));

//   const int idx_sk = tile_m * kF + tile_k;
//   float s  = __half2float(scales[idx_sk]);
//   float zp = static_cast<float>(zps ? zps[idx_sk] : 0);

//   const int8_t* q_tile = q + row0 * K + col0;
//   __half* out_tile = out + row0 * K + col0;

//   extern __shared__ __half smem[];
//   const int smem_stride = FRAG_K;

//   // Produce into SMEM using the SAME layout the copy-out expects (treat as V-layout for test)
//   const bool row_is_v = true;
//   produce_tile_int8_to_smem<FRAG_M, FRAG_K>(
//       q_tile, K, s, zp, smem, smem_stride,
//       M_valid, K_valid,
//       /*row_is_v=*/row_is_v);

//   __syncthreads();

//   // Interleaved copy-out (already correct in your file)
//   const int lane = lane_id();
//   #pragma unroll
//   for (int rpass = 0; rpass < ((FRAG_M + WARP_SIZE - 1) / WARP_SIZE); ++rpass) {
//     int r = rpass * WARP_SIZE + lane;
//     if (r >= FRAG_M) break;
//     int g_row = row0 + r;
//     if (g_row >= M) continue;

//     __half* g_row_ptr = out_tile + r * K;
//     for (int c = 0; c < FRAG_K; ++c) {
//       int g_col = col0 + c;
//       if (g_col < K) {
//         uint32_t byte_off = row_is_v
//           ? fa3_smem_offset_v_bytes_tile<FRAG_M, FRAG_K>(r, c)
//           : fa3_smem_offset_k_bytes_tile<FRAG_M, FRAG_K>(r, c);
//         __half* sm_ptr = reinterpret_cast<__half*>(
//             reinterpret_cast<char*>(smem) + byte_off);
//         g_row_ptr[c] = *sm_ptr;
//       }
//     }
//   }
// }

// // ---------------------------
// // 4) PyTorch bindings
// // ---------------------------

// static void launch_dequant_tiles(at::Tensor q, at::Tensor scales, c10::optional<at::Tensor> zps_opt,
//                                  int frag_m, int frag_k, at::Tensor out)
// {
//   TORCH_CHECK(q.is_cuda() && out.is_cuda(), "q/out must be CUDA");
//   TORCH_CHECK(q.dtype() == at::kChar, "q must be int8 (Char)");
//   TORCH_CHECK(out.dtype() == at::kHalf, "out must be float16");
//   TORCH_CHECK(scales.dtype() == at::kHalf && scales.is_cuda(), "scales must be fp16 CUDA");

//   const int M = q.size(0);
//   const int K = q.size(1);
//   TORCH_CHECK(out.size(0) == M && out.size(1) == K, "out shape mismatch");
//   const int mF = (M + frag_m - 1) / frag_m;
//   const int kF = (K + frag_k - 1) / frag_k;
//   TORCH_CHECK(scales.numel() == mF * kF, "scales shape must be [mF,kF]");

//   const int smem_bytes = frag_m * frag_k * sizeof(__half);

//   dim3 grid(kF, mF, 1);
//   dim3 block(WARP_SIZE, 1, 1);

//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   const int8_t* zps_ptr = nullptr;
//   at::Tensor zps;
//   if (zps_opt.has_value()) {
//     zps = *zps_opt;
//     TORCH_CHECK(zps.is_cuda() && zps.dtype() == at::kChar, "zps must be int8 CUDA");
//     TORCH_CHECK(zps.numel() == mF * kF, "zps shape must be [mF,kF]");
//     zps_ptr = zps.data_ptr<int8_t>();
//   }

//   // Instantiate for your default fragment sizes; you can add more specializations if you need.
//   if (frag_m == 16 && frag_k == 64) {
//     k_dequant_tiles_rowmajor<16,64><<<grid, block, smem_bytes, stream>>>(
//       q.data_ptr<int8_t>(),
//       reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
//       zps_ptr,
//       reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
//       M, K
//     );
//   } else if (frag_m == 16 && frag_k == 128) {
//     k_dequant_tiles_rowmajor<16,128><<<grid, block, smem_bytes, stream>>>(
//       q.data_ptr<int8_t>(),
//       reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
//       zps_ptr,
//       reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
//       M, K
//     );
//   } else {
//     TORCH_CHECK(false, "Unsupported (frag_m, frag_k) = (", frag_m, ", ", frag_k, ")");
//   }
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("dequant_tiles",
//         &launch_dequant_tiles,
//         "INT8 per-frag -> FP16 tiles (row-major out) [test harness]");
// }

// harness/kernels/fused_dequant_producer.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "kernels/smem_layout.h"

template <typename scalar_t>
__global__ void dequant_tiles_kernel(
    const int8_t* __restrict__ q,
    const scalar_t* __restrict__ scales,      // (nFragM, nFragK) as half/bfloat16
    const int8_t* __restrict__ zps,           // (nFragM, nFragK) or nullptr
    half* __restrict__ out,
    int M, int K, int frag_m, int frag_k, int nFragM, int nFragK)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..K)
    if (row >= M || col >= K) return;

    int frag_i = row / frag_m;
    int frag_j = col / frag_k;
    int frag_idx = frag_i * nFragK + frag_j;

    float s = static_cast<float>(scales[frag_idx]);
    int zp = zps ? static_cast<int>(zps[frag_idx]) : 0;

    int idx = row * K + col;
    int qv = static_cast<int>(q[idx]);
    float val = (static_cast<float>(qv - zp)) * s;   // dequantize
    out[idx] = __float2half(val);
}

void dequant_tiles_launcher(
    torch::Tensor q_int8,       // (M,K) int8
    torch::Tensor scales,       // (ceil(M/frag_m), ceil(K/frag_k)) half/bfloat16
    c10::optional<torch::Tensor> zps_opt, // same shape as scales or None
    int frag_m, int frag_k,
    torch::Tensor out_fp16)     // (M,K) half
{
    TORCH_CHECK(q_int8.is_cuda() && out_fp16.is_cuda() && scales.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(q_int8.dtype() == torch::kInt8, "q must be int8");
    TORCH_CHECK(out_fp16.dtype() == torch::kHalf, "out must be fp16");

    auto M = q_int8.size(0);
    auto K = q_int8.size(1);
    TORCH_CHECK(out_fp16.sizes() == q_int8.sizes(), "out shape mismatch");

    int nFragM = (M + frag_m - 1) / frag_m;
    int nFragK = (K + frag_k - 1) / frag_k;
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == nFragM && scales.size(1) == nFragK,
                "scales shape should be (ceil(M/frag_m), ceil(K/frag_k))");

    const int8_t* zps_ptr = nullptr;
    torch::Tensor zps;
    if (zps_opt.has_value() && zps_opt->defined()) {
        zps = *zps_opt;
        TORCH_CHECK(zps.is_cuda() && zps.dtype() == torch::kInt8, "zps must be int8 CUDA");
        TORCH_CHECK(zps.sizes() == scales.sizes(), "zps shape must match scales");
        zps_ptr = zps.data_ptr<int8_t>();
    }

    dim3 block(32, 8);
    dim3 grid((K + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    auto stream = c10::cuda::getCurrentCUDAStream();

    // support half or bfloat16 scales
    if (scales.dtype() == torch::kHalf) {
        dequant_tiles_kernel<at::Half><<<grid, block, 0, stream>>>(
            q_int8.data_ptr<int8_t>(),
            reinterpret_cast<const at::Half*>(scales.data_ptr<at::Half>()),
            zps_ptr,
            reinterpret_cast<half*>(out_fp16.data_ptr<at::Half>()),
            M, K, frag_m, frag_k, nFragM, nFragK);
    } else if (scales.dtype() == torch::kBFloat16) {
        dequant_tiles_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            q_int8.data_ptr<int8_t>(),
            reinterpret_cast<const at::BFloat16*>(scales.data_ptr<at::BFloat16>()),
            zps_ptr,
            reinterpret_cast<half*>(out_fp16.data_ptr<at::Half>()),
            M, K, frag_m, frag_k, nFragM, nFragK);
    } else {
        TORCH_CHECK(false, "scales must be fp16 or bf16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_tiles", &dequant_tiles_launcher, "Dequantize per-frag tiles (INT8->FP16)");
}

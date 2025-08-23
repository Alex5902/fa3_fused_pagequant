#pragma once
#include <cstdint>

enum KvDType : int32_t { KV_FP16=0, KV_FP8_E4M3=1, KV_FP8_E5M2=2, KV_INT8=3 };

struct KvPageHeader {
  uint32_t n_tokens;
  uint32_t head_group_id;
  uint32_t d_k, d_v;
  uint32_t frag_M, frag_K;     // must match FA-3 tiling
  uint32_t dtype;              // KvDType
  uint32_t flags;              // bit0: has_zp
  uint64_t scales_K;           // device ptr to [mF*kF] fp16
  uint64_t scales_V;           // device ptr to [mF*kF] fp16
  uint64_t zp_K;               // optional device ptr [mF*kF] int8
  uint64_t zp_V;               // optional device ptr [mF*kF] int8
  uint64_t data_K;             // device ptr quantized payload
  uint64_t data_V;             // device ptr quantized payload
};

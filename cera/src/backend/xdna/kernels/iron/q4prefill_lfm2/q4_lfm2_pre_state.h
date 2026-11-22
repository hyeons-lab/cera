#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int B_SUB = 8;
constexpr int B_CORE = 16;
constexpr int ROWS = 16;
constexpr int QB = 32;
constexpr int CHUNK_K = 1024;
constexpr int NB_CHUNK = CHUNK_K / QB;  // 32
constexpr int XQ_SUB_BYTES = B_SUB * CHUNK_K;  // 8192
constexpr int HEAD_DIM = 64;
constexpr int GQA_Q_HEADS = 4;  // per column (32 Q heads / 8 cols)
constexpr int KV_CHUNK_T = 32;
constexpr int KV_HALF_F32 = 1024;  // 32 * 64 bf16 = 4096 B
constexpr int D_CONV = 2;
constexpr int L_CACHE = 3;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float ATTN_SCALE = 0.125f;  // 1.0f / sqrtf(64.0f)
constexpr float ATTN_SCALE_LOG2E = 0.125f * 1.4426950408889634f;

// Shared L1 scratch union across mutually exclusive stages (~14.1 KiB max, no separate ModeBuf)
union ScratchUnion {
  struct {
    alignas(64) float x_slice[B_SUB][256];  // 8192 B
    alignas(64) float norm_w[256];          // 1024 B
    alignas(64) float sum_sq[B_SUB];        //   32 B
    alignas(64) uint8_t xq_out[2048];       // 2048 B
    alignas(64) bfloat16 xd_out[64];        //  128 B
  } norm;                                   // 11424 B
  struct {
    alignas(64) float y_acc[8 * B_CORE * ROWS];  // 8192 B
    alignas(64) bfloat16 xd_out[64];             //  128 B
  } gemm;                                        // 8320 B
  struct {
    alignas(64) float rope_cs[B_SUB][64];         // 2048 B
    alignas(64) bfloat16 k_buf[B_SUB][HEAD_DIM];  // 1024 B
    alignas(64) bfloat16 v_buf[B_SUB][HEAD_DIM];  // 1024 B
    alignas(64) float q_norm_w[HEAD_DIM];         //  256 B
    alignas(64) float k_norm_w[HEAD_DIM];         //  256 B
    uint8_t _pad_after_rope[10496 - 4608];         // reserve up to q_bf16 (align with attn.q_bf16)
    alignas(64) bfloat16 q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];  // 4096 B (disjoint from attn fields!)
  } rope;                                         // 14592 B
  struct {
    alignas(64) float o_acc[GQA_Q_HEADS][B_SUB][HEAD_DIM];  // 8192 B
    alignas(64) float row_l[GQA_Q_HEADS][B_SUB];            //  128 B (contiguous directly after o_acc)
    alignas(64) float row_m[GQA_Q_HEADS][B_SUB];            //  128 B
    union {
      alignas(64) float scores[2][B_SUB][KV_CHUNK_T];       // 2048 B
      alignas(64) bfloat16 p_bf16[2 * 4 * 64];              // 1024 B
    };
    alignas(64) bfloat16 q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];  // 4096 B
  } attn;                                                   // 14592 B
  struct {
    alignas(64) float state[D_CONV][64];  //  512 B (contiguous directly before bx!)
    alignas(64) float bx[32][64];         // 8192 B
    alignas(64) float wt[L_CACHE][64];    //  768 B
    alignas(64) float conv_out[8 * 64];   // 2048 B
  } conv;                                 // 11520 B
};

extern "C" {
extern ScratchUnion g_scratch;
extern int32_t g_params[16];
}

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v);
aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x);
void q4_scale_f32_16(float *__restrict dst, const float *__restrict src, float s, int n);
void q4_fill_f32_16(float *__restrict dst, float val, int n);
void q4_fa_qk_dual(const bfloat16 *__restrict q0_base, const bfloat16 *__restrict q1_base, const bfloat16 *__restrict kv_mat, int32_t kb_max);
void q4_fa_pv_d16(const bfloat16 *__restrict kv_mat, int32_t kb_max, int32_t h_base);
float q4_fa_exp2_row32(const float *__restrict sc, float m_val, bfloat16 *__restrict p_col);

/// 2-limb BF16-emulated f32 vector multiply (~15-bit mantissa) for softmax/exp2f without 4-limb spills.
static inline __attribute__((always_inline)) aie::vector<float, 16> q4_fast_fmul2_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  aie::vector<bfloat16, 16> ah = aie::accum<accfloat, 16>(a).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> bh = aie::accum<accfloat, 16>(b).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> al = aie::accum<accfloat, 16>(
      aie::sub(a, aie::accum<accfloat, 16>(ah).template to_vector<float>())).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> bl = aie::accum<accfloat, 16>(
      aie::sub(b, aie::accum<accfloat, 16>(bh).template to_vector<float>())).template to_vector<bfloat16>();
  aie::accum<accfloat, 16> acc = aie::mul<accfloat>(ah, bh);
  acc = aie::mac(acc, ah, bl);
  acc = aie::mac(acc, al, bh);
  return acc.template to_vector<float>();
}

#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int B_SUB = 8;
constexpr int B_CORE = 16;
constexpr int ROWS = 16;
constexpr int QB = 32;
constexpr int CHUNK_K = 1024;
constexpr int NB_CHUNK = CHUNK_K / QB;
constexpr int XQ_SUB_BYTES = B_SUB * CHUNK_K;
constexpr int HEAD_DIM = 64;
constexpr int GQA_Q_HEADS = 4;
constexpr int KV_CHUNK_T = 32;
constexpr int KV_HALF_F32 = 1024;
constexpr int D_CONV = 2;
constexpr int L_CACHE = 3;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float ATTN_SCALE = 0.125f;
constexpr float ATTN_SCALE_LOG2E = 0.125f * 1.4426950408889634f;

// Shared L1 scratch (20480 B -> data_size = 21504)
union ScratchUnion {
  struct {
    alignas(64) float x_slice[B_SUB][256];
    alignas(64) float norm_w[256];
    alignas(64) float sum_sq[B_SUB];
    uint8_t _pad_after_norm[9472 - 9248];
    alignas(64) uint8_t xq_out[2048];
    alignas(64) bfloat16 xd_out[64];
  } norm;
  struct {
    alignas(64) float y_acc[16 * B_CORE * ROWS];  // 0..16383
    union {
      alignas(64) bfloat16 xd_out[64];            // 16384..16511 (FFN)
      alignas(64) bfloat16 conv_xd[2][2][512];    // 16384..20479 (ShortConv)
    };
  } gemm;
  struct {
    uint8_t _pad_y[16384];                       // 0..16383
    alignas(64) bfloat16 k_buf[B_SUB][HEAD_DIM];  // 16384..17407
    alignas(64) float tmp_n[HEAD_DIM];            // 17408..17663
  } rope;
  struct {
    alignas(64) float o_acc[GQA_Q_HEADS][HEAD_DIM / 8][B_SUB * 8];  // 0..8191
    alignas(64) float row_l[GQA_Q_HEADS][2 * B_SUB];        // 8192..8447
    alignas(64) float row_m[GQA_Q_HEADS][2 * B_SUB];        // 8448..8703
    alignas(64) float scores[KV_CHUNK_T][16];               // 8704..10751
    alignas(64) bfloat16 p_bf16[2 * 4 * 64];                // 10752..11775
    alignas(64) int32_t kbuf[16];                           // 11776..11839
    alignas(64) float abuf[16];                             // 11840..11903
    uint8_t _pad_q1[12288 - 11904];
    alignas(64) bfloat16 q1_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8]; // 12288..16383
    alignas(64) bfloat16 q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];  // 16384..20479
    alignas(64) bfloat16 v_t[2048];                                           // 4096 B pre-transposed V
  } attn;
  struct {
    alignas(64) float state[D_CONV][64];
    alignas(64) float bx[32][64];
    alignas(64) float wt[L_CACHE][64];
    alignas(64) uint8_t xq_out[2048];
    alignas(64) bfloat16 xd_out[64];
    alignas(64) float conv_out[8 * 64];
  } conv;
};

extern "C" {
extern ScratchUnion g_scratch;
extern int32_t g_params[16];
extern float *g_x_ptr;
}

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v);
void q4_elem_mul(float *__restrict dst, const float *__restrict a, const float *__restrict b, float s, int n);
void q4_fill_f32_16(float *__restrict dst, float val, int n);
void q4_fa_qk_quad(const bfloat16 *__restrict q0a, const bfloat16 *__restrict q1a, const bfloat16 *__restrict q0b, const bfloat16 *__restrict q1b, const bfloat16 *__restrict kv_mat, int32_t kb_max, float (*__restrict sc0)[16], float (*__restrict sc1)[16]);
void q4_fa_pv_d16(const bfloat16 *__restrict kv_mat, int32_t kb_max, int32_t h_tag);
void q4_fa_rescale_2h(int32_t h_tag, aie::vector<float, 16> diff, uint32_t need);
void q4_fa_softmax_2h(int32_t h_tag, uint32_t mask16, const float (*__restrict sc_mat)[16]);
extern "C" void q4_silu_mul_512(float *__restrict gp);
extern "C" void quantize_q8_2blk_8tok(
    const float *__restrict src,
    int32_t row_stride,
    int32_t col2_off,
    uint8_t *__restrict q_dst0,
    uint8_t *__restrict q_dst1,
    bfloat16 *__restrict d_dst0);

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

static_assert(sizeof(ScratchUnion) <= 24576, "ScratchUnion size");
static_assert(__builtin_offsetof(ScratchUnion, attn.q_bf16) - __builtin_offsetof(ScratchUnion, attn.q1_bf16) == 4096, "q1/q offset");

aie::vector<float, 32> q4_exp2f_32(aie::vector<float, 16> x0, aie::vector<float, 16> x1);

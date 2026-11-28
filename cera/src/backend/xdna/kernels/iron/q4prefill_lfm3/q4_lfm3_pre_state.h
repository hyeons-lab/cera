#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int B_SUB = 8;
constexpr int B_CORE = 16;
constexpr int ROWS = 16;
constexpr int QB = 32;
constexpr int CHUNK_K = 1024;
constexpr int NB_CHUNK = CHUNK_K / QB;       // 32
constexpr int XQ_SUB_BYTES = B_SUB * CHUNK_K;  // 8192
constexpr int HEAD_DIM = 128;
constexpr int GQA_Q_HEADS = 2;  // per column (16 Q heads / 8 cols)
constexpr int KV_CHUNK_T = 32;
constexpr int KV_HALF_F32 = 2048;  // 32 * 128 bf16 = 8192 B
constexpr int D_CONV = 6;
constexpr int L_CACHE = 7;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float ATTN_SCALE = 0.08838834764831845f;  // 1.0f / sqrtf(128.0f)
constexpr float ATTN_SCALE_LOG2E = 0.08838834764831845f * 1.4426950408889634f;

// Shared L1 scratch union (20480 B -> data_size = 21056)
union ScratchUnion {
  struct {
    alignas(64) float x_slice[B_SUB][256];  // 0..8191
    alignas(64) float norm_w[256];          // 8192..9215
    alignas(64) float sum_sq[B_SUB];        // 9216..9247
    uint8_t _pad_after_norm[9472 - 9248];   // 9248..9471
    alignas(64) uint8_t xq_out[2048];       // 9472..11519
    alignas(64) bfloat16 xd_out[64];        // 11520..11647
  } norm;
  struct {
    alignas(64) float y_acc[16 * B_CORE * ROWS];  // 0..16383
    alignas(64) bfloat16 xd_out[64];              // 16384..16511
  } gemm;
  struct {
    alignas(64) bfloat16 rope_cs_bf[B_SUB][HEAD_DIM];  // 0..2047
    uint8_t _pad_after_cs[4096 - 2048];                // 2048..4095
    alignas(64) bfloat16 k_buf[B_SUB][HEAD_DIM];   // 4096..6143
    alignas(64) bfloat16 v_buf[B_SUB][HEAD_DIM];   // 6144..8191
    alignas(64) float q_norm_w[HEAD_DIM];          // 8192..8703
    alignas(64) float k_norm_w[HEAD_DIM];          // 8704..9215
    alignas(64) float tmp_n[HEAD_DIM];             // 9216..9727
    alignas(64) bfloat16 tmp_bf[HEAD_DIM];         // 9728..9983
    uint8_t _pad_after_rope[12288 - 9984];         // 9984..12287
    alignas(64) bfloat16 q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];   // 12288..16383
    alignas(64) bfloat16 q1_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];  // 16384..20479
  } rope;
  struct {
    alignas(64) float o_acc[GQA_Q_HEADS][HEAD_DIM / 8][B_SUB * 8];  // 0..8191 (2048 f32)
    alignas(64) float row_l[GQA_Q_HEADS][2 * B_SUB];                // 8192..8319 (+2048 f32)
    alignas(64) float row_m[GQA_Q_HEADS][2 * B_SUB];                // 8320..8447 (+2080 f32)
    union {
      alignas(64) float scores[KV_CHUNK_T][16];                     // 8448..10495 (2048 B)
      alignas(64) bfloat16 v_t[1024];                               // 8448..10495 (2048 B)
    };
    alignas(64) uint32_t fa_mask16;                                 // 10496..10499
    int32_t fa_kb_max;                                              // 10500..10503
    uint8_t _pad_before_p[10624 - 10504];                           // 10504..10623
    alignas(64) bfloat16 p_bf16[GQA_Q_HEADS * 4 * 64];              // 10624..11647 (+2656 f32)
    alignas(64) float gate_sig[2][GQA_Q_HEADS][B_SUB];              // 11648..11775
    uint8_t _pad_after_attn[12288 - 11776];                         // 11776..12287
    alignas(64) bfloat16 q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];   // 12288..16383
    alignas(64) bfloat16 q1_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_SUB * 8];  // 16384..20479
  } attn;
  struct {
    alignas(64) float ring[D_CONV + 32 + D_CONV][64];  // 0..11263 (0..5: state, 6..37: bx)
    alignas(64) float wt[L_CACHE][64];                 // 11264..13055
    alignas(64) float conv_out[B_SUB * 64];            // 13056..15103
  } conv;
};

constexpr int ATTN_ST_L_OFF = 2048;
constexpr int ATTN_ST_M_OFF = 2080;
constexpr int ATTN_ST_P_OFF = 2656;

static_assert(sizeof(ScratchUnion) == 20480, "ScratchUnion size");
static_assert(__builtin_offsetof(ScratchUnion, rope.k_buf) == 4096, "rope k_buf offset");
static_assert(__builtin_offsetof(ScratchUnion, rope.v_buf) - __builtin_offsetof(ScratchUnion, rope.k_buf) == 2048, "v/k offset");
static_assert(__builtin_offsetof(ScratchUnion, rope.q_norm_w) == 8192, "rope q_norm_w offset");
static_assert(__builtin_offsetof(ScratchUnion, rope.tmp_bf) == 9728, "rope tmp_bf offset");
static_assert(__builtin_offsetof(ScratchUnion, rope.q_bf16) == 12288, "rope q_bf16 offset");
static_assert(__builtin_offsetof(ScratchUnion, conv.wt) == 11264, "conv wt offset");
static_assert(__builtin_offsetof(ScratchUnion, conv.conv_out) == 13056, "conv_out offset");
static_assert(__builtin_offsetof(ScratchUnion, attn.q1_bf16) - __builtin_offsetof(ScratchUnion, attn.q_bf16) == 4096, "q1/q offset");
static_assert(__builtin_offsetof(ScratchUnion, attn.row_l) == ATTN_ST_L_OFF * 4, "row_l offset");
static_assert(__builtin_offsetof(ScratchUnion, attn.row_m) == ATTN_ST_M_OFF * 4, "row_m offset");
static_assert(__builtin_offsetof(ScratchUnion, attn.p_bf16) == ATTN_ST_P_OFF * 4, "p_bf16 offset");
static_assert(__builtin_offsetof(ScratchUnion, attn.gate_sig) >= 11648, "gate_sig outside xq/xd");

extern "C" {
extern ScratchUnion g_scratch;
extern int32_t g_params[16];
extern float *g_x_ptr;
void q4_silu_mul_512(float *__restrict gp);
void quantize_q8_2blk_8tok(
    const float *__restrict src,
    int32_t row_stride,
    int32_t col2_off,
    uint8_t *__restrict q_dst0,
    uint8_t *__restrict q_dst1,
    bfloat16 *__restrict d_dst0);
}

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

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v);
void q4_sigmoid_f32_16(float *__restrict p);
void q4_elem_mul(float *__restrict dst, const float *__restrict a, const float *__restrict b, float s, int n);
void q4_fill_f32_16(float *__restrict dst, float val, int n);
void q4_fa_qk_quad(
    const bfloat16 *__restrict q0a,
    const bfloat16 *__restrict q1a,
    const bfloat16 *__restrict q0b,
    const bfloat16 *__restrict q1b,
    const bfloat16 *__restrict kv_mat,
    int32_t kb_max,
    float (*__restrict sc0)[16],
    float (*__restrict sc1)[16]);
void q4_fa_softmax_2h(
    int32_t h_idx,
    uint32_t mask16,
    const float (*__restrict sc_mat)[16],
    bfloat16 *__restrict p_dst);
void q4_fa_pack_v_32(const bfloat16 *__restrict v_mat_d32, int32_t kb_max);
void q4_fa_pv_d32(int32_t kb_max, int32_t d32);
void q4_fa_rescale_2h(int32_t h_idx, aie::vector<float, 16> diff, uint32_t need);
extern "C" void q4_lfm3_fa_op_impl(const float *__restrict kv_pkt, int32_t row_arg, int32_t iter_idx);

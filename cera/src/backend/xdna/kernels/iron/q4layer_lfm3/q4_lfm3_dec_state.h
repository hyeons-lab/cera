#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr int32_t N_ROWS = 4;
static constexpr int32_t N_COLS = 8;
static constexpr int32_t ROWS = 16;
static constexpr int32_t QB = 32;
static constexpr int32_t CHUNK_K = 2048;
static constexpr int32_t NB_CHUNK = CHUNK_K / QB;
static constexpr int32_t HEAD_DIM = 128;
static constexpr int32_t KV_CHUNK_T = 32;
static constexpr int32_t X_CHUNK_F32 = 20;
static constexpr int32_t X_ELEMS = 640;
static constexpr int32_t W_ELEMS = 4608;
static constexpr int32_t Y_CORE_ELEMS = 64;

struct DecState {
  alignas(64) int32_t params[16];
  alignas(64) float norm_w[64];
  alignas(64) float post_norm_w[64];
  alignas(64) float x_slice[64];
  alignas(64) float res_acc[64];
  alignas(64) float y_gemm[4][16];
  alignas(64) float ffn_gate[4][16];
  alignas(64) uint8_t xq_out[64];
  alignas(64) bfloat16 xd_out[8];
  float sum_sq;
  union {
    alignas(64) float xq_buf[4 * X_ELEMS];
    struct {
      alignas(64) float _xq0_ffn[X_ELEMS];
      alignas(64) float ffn_gates_hi[3][4][16];
    };
    struct {
      alignas(64) float _xq0[X_ELEMS];
      alignas(64) float conv_wt[7][64];
      alignas(64) float conv_state[6][64];
      alignas(64) float conv_bx[64];
      alignas(64) float conv_c[64];
      alignas(64) bfloat16 q_bf16[HEAD_DIM];
      alignas(64) float rope_cs[HEAD_DIM];
      alignas(64) float q_norm_w[HEAD_DIM];
      alignas(64) float k_norm_w[HEAD_DIM];
      alignas(64) float scores[KV_CHUNK_T];
      alignas(64) float o_acc[HEAD_DIM];
      float row_m;
      float row_l;
      float gate_sig;
    };
  };
};

extern "C" {
extern DecState g_dec;
extern const int32_t dec_lane_idx[16];
}

aie::vector<float, 16> dec_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> dec_inv_16(aie::vector<float, 16> v);
aie::vector<float, 16> dec_exp2f_16(aie::vector<float, 16> x);

static inline aie::vector<bfloat16, 16> dec_f32_to_bf16_16(aie::vector<float, 16> v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>();
}

static inline aie::vector<float, 16> dec_bf16_to_f32_16(aie::vector<bfloat16, 16> v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.template to_vector<float>();
}

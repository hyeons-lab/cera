#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int B_CORE = 8;
constexpr int ROWS = 16;
constexpr int QB = 32;
constexpr int CHUNK_K = 2048;
constexpr int NB_CHUNK = CHUNK_K / QB;  // 64
constexpr int XQ_BYTES = B_CORE * CHUNK_K;  // 16384
constexpr int HEAD_DIM = 64;
constexpr int GQA_Q_HEADS = 4;  // per column (32 Q heads / 8 cols)
constexpr int KV_CHUNK_T = 64;
constexpr int SILU_HALF_F32 = 2304;  // 9216 bytes
constexpr float LOG2E = 1.4426950408889634f;

// Shared L1 scratch union across mutually exclusive stages (~11.5 KiB max)
union ScratchUnion {
  struct {
    alignas(64) float x_slice[B_CORE][256];
    alignas(64) float norm_w[256];
    alignas(64) float sum_sq[B_CORE];
    alignas(64) uint8_t xq_out[2048];
    alignas(64) bfloat16 xd_out[64];
  } norm;
  struct {
    alignas(64) float y_acc[8 * B_CORE * ROWS];  // 1024 floats = 4096 B
  } gemm;
  struct {
    alignas(64) float rope_cs[B_CORE][64];
    alignas(64) float k_buf[B_CORE][HEAD_DIM];
    alignas(64) float v_buf[B_CORE][HEAD_DIM];
  } rope;
  struct {
    alignas(64) float o_acc[GQA_Q_HEADS][B_CORE][HEAD_DIM];  // 8192 B
    alignas(64) float scores[B_CORE][KV_CHUNK_T];            // 2048 B
    alignas(64) bfloat16 p_bf16[B_CORE][KV_CHUNK_T];         // 1024 B
    alignas(64) float row_m[GQA_Q_HEADS][B_CORE];            // 128 B
    alignas(64) float row_l[GQA_Q_HEADS][B_CORE];            // 128 B
  } attn;
  struct {
    alignas(64) float a_buf[B_CORE][64];  // 8 * 64 = 2048 B
    alignas(64) uint8_t xq_out[2048];
    alignas(64) bfloat16 xd_out[64];
  } elem;
};

extern "C" {
extern ScratchUnion g_scratch;
extern int32_t g_params[16];
extern bfloat16 g_q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_CORE * 8];
extern const int32_t lane_idx[16];
}

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v);
aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x);
void q4_scale_f32_16(float *__restrict dst, const float *__restrict src, float s, int n);

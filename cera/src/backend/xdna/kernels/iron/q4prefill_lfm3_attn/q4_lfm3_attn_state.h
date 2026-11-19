#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int B_CORE = 8;
constexpr int ROWS = 16;
constexpr int QB = 32;
constexpr int CHUNK_K = 2048;
constexpr int NB_CHUNK = CHUNK_K / QB;  // 64
constexpr int XQ_BYTES = B_CORE * CHUNK_K;  // 16384
constexpr int HEAD_DIM = 128;
constexpr int GQA_Q_HEADS = 2;  // per column (16 Q heads / 8 cols)
constexpr int KV_CHUNK_T = 32;
constexpr int SILU_HALF_F32 = 2304;  // 9216 bytes
constexpr float LOG2E = 1.4426950408889634f;
constexpr float ATTN_SCALE = 0.08838834764831845f;  // 1.0f / sqrtf(128.0f)

// Shared L1 scratch union across mutually exclusive stages (<= 11,424 B max)
union ScratchUnion {
  struct {
    alignas(64) float x_slice[B_CORE][256];  // 8192 B
    alignas(64) float norm_w[256];           // 1024 B
    alignas(64) float sum_sq[B_CORE];        //   32 B
    alignas(64) uint8_t xq_out[2048];        // 2048 B
    alignas(64) bfloat16 xd_out[64];         //  128 B
  } norm;                                    // 11424 B
  struct {
    alignas(64) float y_acc[8 * B_CORE * ROWS];  // 4096 B
  } gemm;
  struct {
    alignas(64) float rope_cs[B_CORE][128];        // 4096 B
    alignas(64) bfloat16 k_buf[B_CORE][HEAD_DIM];  // 2048 B
    alignas(64) bfloat16 v_buf[B_CORE][HEAD_DIM];  // 2048 B
    alignas(64) float q_norm_w[HEAD_DIM];          //  512 B
    alignas(64) float k_norm_w[HEAD_DIM];          //  512 B
  } rope;                                          // 9216 B
  struct {
    alignas(64) float o_acc[GQA_Q_HEADS][B_CORE][HEAD_DIM];  // 8192 B
    alignas(64) float row_l[GQA_Q_HEADS][B_CORE];            //   64 B
    alignas(64) float row_m[GQA_Q_HEADS][B_CORE];            //   64 B
    alignas(64) float gate_sig[GQA_Q_HEADS][B_CORE];         //   64 B
    alignas(64) float scores[B_CORE][KV_CHUNK_T];            // 1024 B
    alignas(64) bfloat16 p_bf16[4 * 64];                     //  512 B
  } attn;                                                    // 9856 B
  struct {
    alignas(64) float a_buf[B_CORE][64];  // 2048 B
    alignas(64) uint8_t xq_out[2048];     // 2048 B
    alignas(64) bfloat16 xd_out[64];      //  128 B
  } elem;
};

extern "C" {
extern ScratchUnion g_scratch;
extern int32_t g_params[16];
extern bfloat16 g_q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_CORE * 8];
}

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v);
aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x);
void q4_scale_f32_16(float *__restrict dst, const float *__restrict src, float s, int n);
void q4_fill_f32_16(float *__restrict dst, float val, int n);

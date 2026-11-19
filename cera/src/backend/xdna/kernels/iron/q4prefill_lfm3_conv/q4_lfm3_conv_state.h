#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int B_CORE = 8;
constexpr int ROWS = 16;
constexpr int QB = 32;
constexpr int CHUNK_K = 2048;
constexpr int NB_CHUNK = CHUNK_K / QB;  // 64
constexpr int XQ_BYTES = B_CORE * CHUNK_K;  // 16384
constexpr int D_CONV = 6;
constexpr int L_CACHE = 7;
constexpr float LOG2E = 1.4426950408889634f;

// Shared L1 scratch union across mutually exclusive stages (~11.5 KiB max)
union ScratchUnion {
  struct {
    alignas(64) float x_slice[B_CORE][256];  // 8192 B
    alignas(64) float norm_w[256];           // 1024 B
    alignas(64) float sum_sq[B_CORE];        //   32 B
    alignas(64) uint8_t xq_out[2048];        // 2048 B
    alignas(64) bfloat16 xd_out[64];         //  128 B
  } norm;
  struct {
    alignas(64) float y_acc[8 * B_CORE * ROWS];  // 1024 floats = 4096 B
  } gemm;
  struct {
    alignas(64) float a_buf[B_CORE][64];  // 2048 B
    alignas(64) uint8_t xq_out[2048];     // 2048 B
    alignas(64) bfloat16 xd_out[64];      //  128 B
  } elem;
  struct {
    alignas(64) float state[D_CONV][64];     // 1536 B
    alignas(64) float wt[L_CACHE][64];       // 1792 B
    alignas(64) float bx[32][64];            // 8192 B
  } conv;
};

extern "C" {
extern ScratchUnion g_scratch;
extern int32_t g_params[16];
extern float g_conv_out[16 * 64];  // 4096 B
}

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b);
aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v);
aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x);

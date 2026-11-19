#include <aie_api/aie.hpp>
#include <stdint.h>

constexpr int M_CORE = 32;
constexpr int N_CORE = 32;
constexpr int K_CHUNK = 64;

// Accumulates C[32, 32] (row-major f32) += X[32, 64] (row-major bf16) * W[K/8][N/8][8_k][8_n] (bf16).
// When acc_flag == 0, overwrites y_out; when acc_flag != 0, adds into y_out.
extern "C" void gemm_bf16_tile(
    const uint32_t *__restrict x_u32,
    const uint32_t *__restrict w_u32,
    float *__restrict y_out,
    int32_t acc_flag) {
  const bfloat16 *__restrict x = (const bfloat16 *)x_u32;
  const bfloat16 *__restrict w = (const bfloat16 *)w_u32;

  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16, accfloat>;

  for (int m4 = 0; m4 < M_CORE / 4; ++m4) {
    MMUL acc0, acc1, acc2, acc3;
    const bfloat16 *px_row = x + (m4 * 4) * K_CHUNK;

    for (int k16 = 0; k16 < K_CHUNK / 16; ++k16) chess_prepare_for_pipelining {
      const bfloat16 *px = px_row + k16 * 16;
      aie::vector<bfloat16, 16> r0 = aie::load_v<16>(px + 0 * K_CHUNK);
      aie::vector<bfloat16, 16> r1 = aie::load_v<16>(px + 1 * K_CHUNK);
      aie::vector<bfloat16, 16> r2 = aie::load_v<16>(px + 2 * K_CHUNK);
      aie::vector<bfloat16, 16> r3 = aie::load_v<16>(px + 3 * K_CHUNK);

      aie::vector<bfloat16, 32> a_ev = aie::concat(
          aie::concat(r0.template extract<8>(0), r1.template extract<8>(0)),
          aie::concat(r2.template extract<8>(0), r3.template extract<8>(0)));
      aie::vector<bfloat16, 32> a_od = aie::concat(
          aie::concat(r0.template extract<8>(1), r1.template extract<8>(1)),
          aie::concat(r2.template extract<8>(1), r3.template extract<8>(1)));

      const bfloat16 *pw0 = w + (2 * k16) * (4 * 64);
      const bfloat16 *pw1 = pw0 + (4 * 64);

      aie::vector<bfloat16, 64> b00 = aie::load_v<64>(pw0 + 0 * 64);
      aie::vector<bfloat16, 64> b01 = aie::load_v<64>(pw0 + 1 * 64);
      aie::vector<bfloat16, 64> b02 = aie::load_v<64>(pw0 + 2 * 64);
      aie::vector<bfloat16, 64> b03 = aie::load_v<64>(pw0 + 3 * 64);

      if (k16 == 0) {
        acc0.mul(a_ev, b00);
        acc1.mul(a_ev, b01);
        acc2.mul(a_ev, b02);
        acc3.mul(a_ev, b03);
      } else {
        acc0.mac(a_ev, b00);
        acc1.mac(a_ev, b01);
        acc2.mac(a_ev, b02);
        acc3.mac(a_ev, b03);
      }

      aie::vector<bfloat16, 64> b10 = aie::load_v<64>(pw1 + 0 * 64);
      aie::vector<bfloat16, 64> b11 = aie::load_v<64>(pw1 + 1 * 64);
      aie::vector<bfloat16, 64> b12 = aie::load_v<64>(pw1 + 2 * 64);
      aie::vector<bfloat16, 64> b13 = aie::load_v<64>(pw1 + 3 * 64);

      acc0.mac(a_od, b10);
      acc1.mac(a_od, b11);
      acc2.mac(a_od, b12);
      acc3.mac(a_od, b13);
    }

    aie::vector<float, 32> v0 = acc0.template to_vector<float>();
    aie::vector<float, 32> v1 = acc1.template to_vector<float>();
    aie::vector<float, 32> v2 = acc2.template to_vector<float>();
    aie::vector<float, 32> v3 = acc3.template to_vector<float>();

    float *py_base = y_out + (m4 * 4) * N_CORE;
    for (int r = 0; r < 4; ++r) {
      float *py = py_base + r * N_CORE;
      aie::vector<float, 16> lo =
          aie::concat(v0.template extract<8>(r), v1.template extract<8>(r));
      aie::vector<float, 16> hi =
          aie::concat(v2.template extract<8>(r), v3.template extract<8>(r));
      if (acc_flag) {
        lo = aie::add(lo, aie::load_v<16>(py));
        hi = aie::add(hi, aie::load_v<16>(py + 16));
      }
      aie::store_v(py, lo);
      aie::store_v(py + 16, hi);
    }
  }
}

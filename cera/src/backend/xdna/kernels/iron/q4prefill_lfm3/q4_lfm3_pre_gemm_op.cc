#include "q4_lfm3_pre_state.h"

extern "C" void q4_lfm3_pre_gemm_op(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t tile_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t add_prev = flags & 1;
  const int32_t xd_mode = (flags >> 1) & 1;
  const bfloat16 *__restrict xd_base = (const bfloat16 *)(x_slot + 2 * XQ_SUB_BYTES);
  const bfloat16 *__restrict wd_base = (const bfloat16 *)(w_tile + NB_CHUNK * 256);
  float *__restrict y_base = g_scratch.gemm.y_acc + tile_idx * (B_SUB * ROWS);

  using mmul_t = aie::mmul<4, 16, 16, int8, int4>;

#pragma clang loop unroll(disable)
  for (int sub = 0; sub < 2; ++sub) {
    const uint8_t *__restrict xq_ptr = x_slot + sub * XQ_SUB_BYTES;
    const bfloat16 *__restrict xd_ptr = xd_base + sub * (xd_mode ? 32 : 512);
    aie::accum<accfloat, 64> facc0 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> facc1 = aie::zeros<accfloat, 64>();
    const int8_t *__restrict xp0 = (const int8_t *)xq_ptr;
    const int8_t *__restrict xp1 = (const int8_t *)(xq_ptr + 128);
    const uint8_t *__restrict bp = w_tile;
    const bfloat16 *__restrict pws = wd_base;

#pragma clang loop unroll(disable)
    for (int lb = 0; lb < NB_CHUNK; ++lb) chess_prepare_for_pipelining {
      aie::vector<int4, 256> b0 = aie::load_v<256>((const int4 *)bp);
      aie::vector<int4, 256> b1 = aie::load_v<256>((const int4 *)(bp + 128));
      aie::vector<bfloat16, 16> ws = aie::load_v<16>(pws);
      aie::vector<bfloat16, 32> ws32 = aie::concat(ws, ws);
      aie::vector<bfloat16, 64> ws64 = aie::concat(ws32, ws32);

      int c_blk = xd_mode ? ((lb >> 1) & 7) : (lb >> 3);
      int r_blk = xd_mode ? (((lb >> 4) << 1) | (lb & 1)) : (lb & 7);
      const bfloat16 *xd_b = xd_ptr + c_blk * 128 + r_blk * 8;

      {
        mmul_t m0;
        m0.mul(aie::load_v<64>(xp0), b0);
        m0.mac(aie::load_v<64>(xp0 + 64), b1);
        aie::vector<bfloat16, 64> r_bf16 = aie::to_float<bfloat16>(m0.template to_vector<int16>());
        aie::vector<bfloat16, 32> xd01 = aie::concat(
            aie::broadcast<bfloat16, 16>(xd_b[0]),
            aie::broadcast<bfloat16, 16>(xd_b[1]));
        aie::vector<bfloat16, 32> xd23 = aie::concat(
            aie::broadcast<bfloat16, 16>(xd_b[2]),
            aie::broadcast<bfloat16, 16>(xd_b[3]));
        aie::vector<bfloat16, 64> sc64 = aie::mul(ws64, aie::concat(xd01, xd23)).template to_vector<bfloat16>();
        facc0 = aie::mac(facc0, r_bf16, sc64);
      }

      {
        mmul_t m1;
        m1.mul(aie::load_v<64>(xp1), b0);
        m1.mac(aie::load_v<64>(xp1 + 64), b1);
        aie::vector<bfloat16, 64> r_bf16 = aie::to_float<bfloat16>(m1.template to_vector<int16>());
        aie::vector<bfloat16, 32> xd45 = aie::concat(
            aie::broadcast<bfloat16, 16>(xd_b[4]),
            aie::broadcast<bfloat16, 16>(xd_b[5]));
        aie::vector<bfloat16, 32> xd67 = aie::concat(
            aie::broadcast<bfloat16, 16>(xd_b[6]),
            aie::broadcast<bfloat16, 16>(xd_b[7]));
        aie::vector<bfloat16, 64> sc64 = aie::mul(ws64, aie::concat(xd45, xd67)).template to_vector<bfloat16>();
        facc1 = aie::mac(facc1, r_bf16, sc64);
      }

      xp0 += 256;
      xp1 += 256;
      bp += 256;
      pws += ROWS;
    }

    float *__restrict y_ptr = y_base + sub * (8 * B_SUB * ROWS);
    aie::vector<float, 64> fv0 = facc0.template to_vector<float>();
    aie::vector<float, 64> fv1 = facc1.template to_vector<float>();
    if (add_prev) {
      fv0 = aie::add(fv0, aie::load_v<64>(y_ptr));
      fv1 = aie::add(fv1, aie::load_v<64>(y_ptr + 64));
    }
    aie::store_v(y_ptr, fv0);
    aie::store_v(y_ptr + 64, fv1);
  }
}

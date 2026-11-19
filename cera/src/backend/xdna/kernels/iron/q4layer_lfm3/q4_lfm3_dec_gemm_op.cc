#include "q4_lfm3_dec_state.h"

extern "C" void q4_lfm3_dec_gemm_op(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t out_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t add_prev = flags & 1;
  const int32_t dst_sel = (flags >> 1) & 3;

  aie::accum<accfloat, 16> facc = aie::zeros<accfloat, 16>();
  using mmul_t = aie::mmul<4, 16, 16, int8, int4>;
  aie::vector<int8, 64> a0 = aie::zeros<int8, 64>();
  aie::vector<int8, 64> a1 = aie::zeros<int8, 64>();
  aie::vector<int8, 64> a2 = aie::zeros<int8, 64>();
  aie::vector<int8, 64> a3 = aie::zeros<int8, 64>();

  const uint8_t *__restrict x_blk = x_slot;
  const uint8_t *__restrict whalf = w_tile;
#pragma clang loop unroll(disable)
  for (int kh = 0; kh < 2; ++kh) {
    const uint8_t *__restrict bp = whalf;
    const bfloat16 *__restrict pws = (const bfloat16 *)(whalf + 32 * 256);
    for (int c_idx = 0; c_idx < 16; ++c_idx) chess_prepare_for_pipelining {
      const int8_t *__restrict xp = (const int8_t *)x_blk;
      const bfloat16 *__restrict xdp = (const bfloat16 *)(x_blk + 64);

      aie::vector<int4, 256> b00 = aie::load_v<256>((const int4 *)bp);
      aie::vector<int4, 256> b01 = aie::load_v<256>((const int4 *)(bp + 128));
      aie::vector<int4, 256> b10 = aie::load_v<256>((const int4 *)(bp + 256));
      aie::vector<int4, 256> b11 = aie::load_v<256>((const int4 *)(bp + 384));
      aie::vector<bfloat16, 16> ws0 = aie::load_v<16>(pws);
      aie::vector<bfloat16, 16> ws1 = aie::load_v<16>(pws + ROWS);

      a0.insert(0, aie::load_v<16>(xp));
      a1.insert(0, aie::load_v<16>(xp + 16));
      a2.insert(0, aie::load_v<16>(xp + 32));
      a3.insert(0, aie::load_v<16>(xp + 48));

      mmul_t m0, m1;
      m0.mul(a0, b00);
      m0.mac(a1, b01);
      m1.mul(a2, b10);
      m1.mac(a3, b11);

      aie::vector<bfloat16, 16> r0 = aie::to_float<bfloat16>(m0.template to_vector<int16>().template extract<16>(0));
      aie::vector<bfloat16, 16> r1 = aie::to_float<bfloat16>(m1.template to_vector<int16>().template extract<16>(0));
      aie::vector<bfloat16, 16> sc0 = aie::mul(ws0, aie::broadcast<bfloat16, 16>(xdp[0])).template to_vector<bfloat16>();
      aie::vector<bfloat16, 16> sc1 = aie::mul(ws1, aie::broadcast<bfloat16, 16>(xdp[1])).template to_vector<bfloat16>();
      facc = aie::mac(facc, r0, sc0);
      facc = aie::mac(facc, r1, sc1);

      x_blk += 80;
      bp += 512;
      pws += 2 * ROWS;
    }
    whalf += 9216;
  }

  aie::vector<float, 16> fv = facc.template to_vector<float>();
  float *__restrict dst =
      (dst_sel == 0) ? &g_dec.y_gemm[out_idx][0] :
      (dst_sel == 1) ? &g_dec.ffn_gate[out_idx][0] :
                       (g_dec.conv_c + out_idx * 16);
  if (add_prev) {
    fv = aie::add(fv, aie::load_v<16>(dst));
  }
  aie::store_v(dst, fv);
}

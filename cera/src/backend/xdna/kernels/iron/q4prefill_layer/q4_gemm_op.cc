#include "q4_layer_state.h"

extern "C" void q4_layer_gemm_op(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t tile_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  int32_t add_prev = flags & 1;
  int32_t xd_mode = (flags >> 1) & 1;
  const uint8_t *__restrict xq_ptr = x_slot;
  const bfloat16 *__restrict xd_ptr = (const bfloat16 *)(x_slot + XQ_BYTES);
  const bfloat16 *__restrict wd_base = (const bfloat16 *)(w_tile + NB_CHUNK * 256);
  float *__restrict y_ptr = g_scratch.gemm.y_acc + tile_idx * (B_CORE * ROWS);

  using mmul_t = aie::mmul<4, 16, 16, int8, int4>;
  // Process tokens 0..3 then 4..7 sequentially so only one v64accfloat is live.
  for (int grp = 0; grp < 2; ++grp) {
    aie::accum<accfloat, 64> facc = aie::zeros<accfloat, 64>();
    const int8_t *__restrict xp = (const int8_t *)(xq_ptr + grp * 128);
    const uint8_t *__restrict bp = w_tile;
    const bfloat16 *__restrict pws = wd_base;

    for (int lb = 0; lb < NB_CHUNK; ++lb) chess_prepare_for_pipelining {
      aie::vector<int4, 256> b0 = aie::load_v<256>((const int4 *)bp);
      aie::vector<int4, 256> b1 = aie::load_v<256>((const int4 *)(bp + 128));
      aie::vector<bfloat16, 16> ws = aie::load_v<16>(pws);
      aie::vector<bfloat16, 32> ws32 = aie::concat(ws, ws);
      aie::vector<bfloat16, 64> ws64 = aie::concat(ws32, ws32);

      aie::vector<int8, 64> a0 = aie::load_v<64>(xp);
      aie::vector<int8, 64> a1 = aie::load_v<64>(xp + 64);
      mmul_t m;
      m.mul(a0, b0);
      m.mac(a1, b1);
      aie::vector<bfloat16, 64> r_bf16 = aie::to_float<bfloat16>(m.template to_vector<int16>());

      int c_blk = xd_mode ? ((lb >> 1) & 7) : (lb >> 3);
      int r_blk = xd_mode ? (((lb >> 4) << 1) | (lb & 1)) : (lb & 7);
      const bfloat16 *xd_b = xd_ptr + c_blk * 256 + r_blk * 8 + grp * 4;

      aie::vector<bfloat16, 32> xd01 = aie::concat(
          aie::broadcast<bfloat16, 16>(xd_b[0]),
          aie::broadcast<bfloat16, 16>(xd_b[1]));
      aie::vector<bfloat16, 32> xd23 = aie::concat(
          aie::broadcast<bfloat16, 16>(xd_b[2]),
          aie::broadcast<bfloat16, 16>(xd_b[3]));
      aie::vector<bfloat16, 64> xd64 = aie::concat(xd01, xd23);
      aie::vector<bfloat16, 64> sc64 = aie::mul(ws64, xd64).template to_vector<bfloat16>();
      facc = aie::mac(facc, r_bf16, sc64);

      xp += 256;
      bp += 256;
      pws += ROWS;
    }

    aie::vector<float, 64> fv = facc.template to_vector<float>();
    float *gdst = y_ptr + grp * 64;
    if (add_prev) {
      fv = aie::add(fv, aie::load_v<64>(gdst));
    }
    aie::store_v(gdst, fv);
  }
}

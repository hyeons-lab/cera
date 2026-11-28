#include "q4_lfm2_dec_state.h"

extern "C" void q4_lfm2_dec_gemm_op(
    const uint8_t *__restrict w_tile,
    int32_t out_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t add_prev = flags & 1;
  const int32_t dst_sel = (flags >> 1) & 3;

  const aie::vector<bfloat16, 32> magic_v =
      aie::broadcast<int16, 32>(0x4b40).template cast_to<bfloat16>();
  const aie::accum<accfloat, 64> magic_f(aie::concat(magic_v, magic_v));
  const aie::accum<acc32, 64> magic_i = magic_f.template cast_to<acc32>();
  aie::accum<accfloat, 16> facc = aie::zeros<accfloat, 16>();
  using mmul_t = aie::mmul<4, 16, 16, int8, int4>;
  aie::vector<int8, 64> a0 = aie::zeros<int8, 64>();
  aie::vector<int8, 64> a1 = aie::zeros<int8, 64>();

  const int32_t x_slot = (flags >> 3) & 3;
  const int32_t g_slot = (flags >> 5) & 3;
  const uint8_t *__restrict x_blk = (const uint8_t *)(g_dec.xq_buf + x_slot * X_ELEMS);
  const uint8_t *__restrict bp = w_tile;
#pragma clang loop unroll(disable)
  for (int kh = 0; kh < 2; ++kh) {
    const bfloat16 *__restrict pws = (const bfloat16 *)(bp + 8192);
    for (int lb = 0; lb < 32; ++lb) chess_prepare_for_pipelining chess_loop_count(32) {
      int odd = lb & 1;
      const int8 *__restrict xp = (const int8 *)(x_blk + odd * 32);
      const bfloat16 xd = *((const bfloat16 *)(x_blk + 64) + odd);
      a0.insert(0, aie::load_v<16>(xp));
      a1.insert(0, aie::load_v<16>(xp + 16));
      aie::vector<int4, 256> b0 = aie::load_v<256>((const int4 *)bp);
      aie::vector<int4, 256> b1 = aie::load_v<256>((const int4 *)(bp + 128));
      aie::vector<bfloat16, 16> ws = aie::load_v<16>(pws);
      mmul_t m0;
      m0.mul(a0, b0);
      m0.mac(a1, b1);
      aie::accum<acc32, 64> vi = ::add_conf(m0.to_accum().template grow<64>(), magic_i, 0, 0, 0, 0);
      aie::accum<accfloat, 64> vf = ::sub_conf(vi.template cast_to<accfloat>(), magic_f, 0, 0, 0);
      aie::vector<bfloat16, 16> r0 = vf.template to_vector<bfloat16>().template extract<16>(0);
      aie::vector<bfloat16, 16> sc0 = aie::mul(ws, aie::broadcast<bfloat16, 16>(xd)).template to_vector<bfloat16>();
      facc = aie::mac(facc, r0, sc0);
      x_blk += odd * 80;
      bp += 256;
      pws += ROWS;
    }
    bp += 1024;
  }

  aie::vector<float, 16> fv = facc.template to_vector<float>();
  float *__restrict dst =
      (dst_sel == 0) ? &g_dec.y_gemm[out_idx][0] :
      (dst_sel == 1) ? ((g_slot == 0) ? &g_dec.ffn_gate[out_idx][0] : &g_dec.ffn_gates_hi[g_slot - 1][out_idx][0]) :
                       (g_dec.conv_c + out_idx * 16);
  if (add_prev) {
    fv = aie::add(fv, aie::load_v<16>(dst));
  }
  aie::store_v(dst, fv);
}

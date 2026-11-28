#include "q4_lfm3_pre_state.h"

static __attribute__((always_inline)) void gemm_half_step(
    const int8_t *__restrict xp,
    aie::vector<int4, 256> b0,
    aie::vector<int4, 256> b1,
    aie::vector<bfloat16, 32> ws32,
    aie::vector<bfloat16, 32> z4,
    aie::vector<bfloat16, 32> magic_v,
    aie::accum<accfloat, 64> &facc) {
  using mmul_t = aie::mmul<4, 16, 16, int8, int4>;
  mmul_t m;
  m.mul(aie::load_v<64>(xp), b0);
  m.mac(aie::load_v<64>(xp + 64), b1);
  aie::accum<accfloat, 64> mf(aie::concat(magic_v, magic_v));
  aie::accum<acc32, 64> vi = ::add_conf(m.to_accum(), mf.template cast_to<acc32>(), 0, 0, 0, 0);
  aie::accum<accfloat, 64> vf = ::sub_conf(vi.template cast_to<accfloat>(), mf, 0, 0, 0);
  aie::vector<bfloat16, 64> vfb = vf.template to_vector<bfloat16>();
  auto [xd0, xd1] = aie::interleave_zip(z4, z4, 8);
  aie::vector<bfloat16, 32> sc0 = aie::mul(ws32, xd0).template to_vector<bfloat16>();
  aie::vector<bfloat16, 32> sc1 = aie::mul(ws32, xd1).template to_vector<bfloat16>();
  facc = aie::mac(facc, vfb, aie::concat(sc0, sc1));
}

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
  float *__restrict y_base =
      g_scratch.gemm.y_acc + (tile_idx >> 3) * (16 * B_SUB * ROWS) + (tile_idx & 7) * (B_SUB * ROWS);
  const uint32_t m_a = xd_mode ? 0x0eU : 0x18U;
  const uint32_t s_a = xd_mode ? 3U : 1U;
  const uint32_t m_b = xd_mode ? 0x10U : 0U;
  const uint32_t m_c = xd_mode ? 1U : 7U;
  const aie::vector<bfloat16, 32> magic_v =
      aie::broadcast<int16, 32>(0x4b40).template cast_to<bfloat16>();

#pragma clang loop unroll(disable)
  for (int sub = 0; sub < 2; ++sub) {
    const int8_t *__restrict xp = (const int8_t *)(x_slot + sub * XQ_SUB_BYTES);
    const bfloat16 *__restrict xd_ptr = xd_base + sub * (xd_mode ? 32 : 512);
    aie::accum<accfloat, 64> facc0 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> facc1 = aie::zeros<accfloat, 64>();
    const uint8_t *__restrict bp = w_tile;
    const bfloat16 *__restrict pws = wd_base;
    aie::vector<bfloat16, 32> z0 = aie::zeros<bfloat16, 32>();

#pragma clang loop unroll(disable)
    for (int lb = 0; lb < NB_CHUNK; ++lb) chess_prepare_for_pipelining chess_loop_count(32) {
      uint32_t xd_off = ((lb & m_a) << s_a) | ((lb & m_b) >> 3) | (lb & m_c);
      aie::vector<int4, 256> b0 = aie::load_v<256>((const int4 *)bp);
      aie::vector<int4, 256> b1 = aie::load_v<256>((const int4 *)(bp + 128));
      aie::vector<bfloat16, 16> ws = aie::load_v<16>(pws);
      aie::vector<bfloat16, 32> ws32 = aie::concat(ws, ws);
      z0.insert(0, aie::load_v<8>(xd_ptr + xd_off * 8));
      auto [z1, _u1] = aie::interleave_zip(z0, z0, 1);
      auto [z2, _u2] = aie::interleave_zip(z1, z1, 2);
      auto [z4a, z4b] = aie::interleave_zip(z2, z2, 4);

      gemm_half_step(xp, b0, b1, ws32, z4a, magic_v, facc0);
      gemm_half_step(xp + 128, b0, b1, ws32, z4b, magic_v, facc1);

      xp += 256;
      bp += 256;
      pws += ROWS;
    }

    float *__restrict y_ptr0 = y_base + sub * (8 * B_SUB * ROWS);
    aie::vector<float, 64> fv0 = facc0.template to_vector<float>();
    aie::vector<float, 64> fv1 = facc1.template to_vector<float>();
    if (add_prev) {
      fv0 = aie::add(fv0, aie::load_v<64>(y_ptr0));
      fv1 = aie::add(fv1, aie::load_v<64>(y_ptr0 + 64));
    }
    aie::store_v(y_ptr0, fv0);
    aie::store_v(y_ptr0 + 64, fv1);
  }
}

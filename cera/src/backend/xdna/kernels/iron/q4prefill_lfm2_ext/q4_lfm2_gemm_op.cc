#include "q4_lfm2_pre_state.h"

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

extern "C" void q4_lfm2_pre_gemm_op(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t tile_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t add_prev = flags & 1;
  const int32_t xd_mode = (flags >> 1) & 1;
  const int32_t use_cxd = (flags >> 2) & 1;
  const bfloat16 *__restrict xd_base = (const bfloat16 *)(x_slot + 2 * XQ_SUB_BYTES);
  const bfloat16 *__restrict wd_base = (const bfloat16 *)(w_tile + NB_CHUNK * 256);
  const int32_t pt = (flags & 8) ? ((tile_idx & 3) | ((tile_idx & 8) >> 1) | ((tile_idx & 4) << 1)) : tile_idx;
  float *__restrict y_base =
      g_scratch.gemm.y_acc + (pt >> 3) * (16 * B_SUB * ROWS) + (pt & 7) * (B_SUB * ROWS);
  const uint32_t m_a = xd_mode ? 0x0eU : 0x18U;
  const uint32_t s_a = xd_mode ? 3U : 1U;
  const uint32_t m_b = xd_mode ? 0x10U : 0U;
  const uint32_t m_c = xd_mode ? 1U : 7U;
  const aie::vector<bfloat16, 32> magic_v =
      aie::broadcast<int16, 32>(0x4b40).template cast_to<bfloat16>();

#pragma clang loop unroll(disable)
  for (int sub = 0; sub < 2; ++sub) {
    const int8_t *__restrict xp = (const int8_t *)(x_slot + sub * XQ_SUB_BYTES);
    const bfloat16 *__restrict xd_ptr =
        use_cxd ? &g_scratch.gemm.conv_xd[add_prev][sub][0]
                : (xd_base + sub * (xd_mode ? 32 : 512));
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

static __attribute__((noinline, minsize)) float abs_max_32(const float *__restrict r, const float *__restrict r2) {
  const aie::vector<int32_t, 16> abs_mask = aie::broadcast<int32_t, 16>(0x7fffffff);
  aie::vector<int32_t, 16> v = aie::max(
      aie::bit_and(aie::load_v<16>(r).template cast_to<int32_t>(), abs_mask),
      aie::bit_and(aie::load_v<16>(r2).template cast_to<int32_t>(), abs_mask));
#pragma clang loop unroll(disable)
  for (int sh = 32; sh >= 4; sh >>= 1) {
    v = aie::max(v, aie::vector<int32_t, 16>(::shift_bytes(v, v, sh)));
  }
  return v.template cast_to<float>().get(0);
}

extern "C" __attribute__((noinline, minsize)) void quantize_q8_2blk_8tok(
    const float *__restrict src,
    int32_t row_stride,
    int32_t col2_off,
    uint8_t *__restrict q_dst0,
    uint8_t *__restrict q_dst1,
    bfloat16 *__restrict d_dst0) {
  const int32_t blk2_off = col2_off * 2;
  bfloat16 *__restrict d_dst1 = d_dst0 + 8;
  alignas(32) float sc_buf[16];
#pragma clang loop unroll(disable)
  for (int t = 0; t < 8; ++t) {
    const float *__restrict r = src + t * row_stride;
    const float *__restrict r2 = r + blk2_off;
    sc_buf[t] = abs_max_32(r, r + col2_off);
    sc_buf[8 + t] = abs_max_32(r2, r2 + col2_off);
  }
  aie::vector<float, 16> dv = q4_fmul_16(aie::load_v<16>(sc_buf), aie::broadcast<float, 16>(1.0f / 127.0f));
  aie::vector<bfloat16, 16> dv_b = aie::accum<accfloat, 16>(dv).template to_vector<bfloat16>();
  aie::store_v(d_dst0, dv_b.template extract<8>(0));
  aie::store_v(d_dst1, dv_b.template extract<8>(1));
  aie::vector<bfloat16, 16> idv_b = aie::accum<accfloat, 16>(q4_inv_16(dv)).template to_vector<bfloat16>();
#pragma clang loop unroll(disable)
  for (int t = 0; t < 8; ++t) chess_prepare_for_pipelining chess_loop_count(8) {
    const float *__restrict r = src + t * row_stride;
    const float *__restrict r2 = r + blk2_off;
    aie::vector<bfloat16, 32> r01 = aie::concat(
        aie::accum<accfloat, 16>(aie::load_v<16>(r)).template to_vector<bfloat16>(),
        aie::accum<accfloat, 16>(aie::load_v<16>(r + col2_off)).template to_vector<bfloat16>());
    aie::vector<bfloat16, 32> r23 = aie::concat(
        aie::accum<accfloat, 16>(aie::load_v<16>(r2)).template to_vector<bfloat16>(),
        aie::accum<accfloat, 16>(aie::load_v<16>(r2 + col2_off)).template to_vector<bfloat16>());
    aie::vector<float, 32> p01 = aie::mul<accfloat>(r01, aie::broadcast<bfloat16, 32>(idv_b[t])).template to_vector<float>();
    aie::vector<float, 32> p23 = aie::mul<accfloat>(r23, aie::broadcast<bfloat16, 32>(idv_b[8 + t])).template to_vector<float>();
    aie::vector<int8, 32> q01 = aie::to_fixed<int8>(p01, 0);
    aie::vector<int8, 32> q23 = aie::to_fixed<int8>(p23, 0);
    int off = (t >> 2) * 128 + (t & 3) * 16;
    aie::store_v((int8_t *)(q_dst0 + off), q01.template extract<16>(0));
    aie::store_v((int8_t *)(q_dst0 + off + 64), q01.template extract<16>(1));
    aie::store_v((int8_t *)(q_dst1 + off), q23.template extract<16>(0));
    aie::store_v((int8_t *)(q_dst1 + off + 64), q23.template extract<16>(1));
  }
}

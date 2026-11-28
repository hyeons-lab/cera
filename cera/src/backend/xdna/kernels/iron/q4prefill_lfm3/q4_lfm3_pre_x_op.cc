#include "q4_lfm3_pre_state.h"

__attribute__((noinline, minsize)) aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v) {
  return aie::inv(aie::max(v, aie::broadcast<float, 16>(1e-35f)));
}

__attribute__((noinline, minsize)) void q4_sigmoid_f32_16(float *__restrict p) {
  aie::vector<bfloat16, 16> half_b = aie::broadcast<bfloat16, 16>((bfloat16)0.5f);
  aie::accum<accfloat, 16> hx = aie::mul<accfloat>(
      aie::accum<accfloat, 16>(aie::load_v<16>(p)).template to_vector<bfloat16>(),
      half_b);
  aie::vector<bfloat16, 16> th(::tanh(hx.to_native()));
  aie::accum<accfloat, 16> half_acc;
  half_acc.from_vector(aie::broadcast<float, 16>(0.5f));
  aie::store_v(p, aie::mac(half_acc, th, half_b).template to_vector<float>());
}

__attribute__((noinline, cold, minsize)) void q4_fa_rescale_2h(
    int32_t h_idx,
    aie::vector<float, 16> diff,
    uint32_t need) {
  aie::vector<bfloat16, 16> eb(::exp2(aie::accum<accfloat, 16>(diff).to_native()));
  aie::vector<float, 16> alpha16 = aie::accum<accfloat, 16>(eb).template to_vector<float>();
  float *__restrict rlp = &g_scratch.attn.row_l[h_idx][0];
  aie::store_v(rlp, q4_fmul_16(aie::load_v<16>(rlp), alpha16));
  alignas(32) float abuf[16];
  aie::store_v(abuf, alpha16);
  float *__restrict b0 = &g_scratch.attn.o_acc[h_idx][0][0];
  float *__restrict b1 = g_x_ptr + h_idx * 1024;
#pragma clang loop unroll(disable)
  for (int idx = 0; idx < 16; ++idx) {
    if (!((need >> idx) & 1U)) continue;
    float *__restrict p = ((idx < 8) ? b0 : b1) + (idx & 7) * 8;
    float a = abuf[idx];
#pragma clang loop unroll(disable)
    for (int d16 = 0; d16 < (HEAD_DIM / 16); ++d16, p += 128) {
      aie::vector<float, 16> o = q4_fmul_16(
          aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + 64)),
          aie::broadcast<float, 16>(a));
      aie::store_v(p, o.template extract<8>(0));
      aie::store_v(p + 64, o.template extract<8>(1));
    }
  }
}

extern "C" __attribute__((noinline)) void q4_silu_mul_512(float *__restrict gp) {
  aie::vector<bfloat16, 16> half_b = aie::broadcast<bfloat16, 16>((bfloat16)0.5f);
  aie::vector<bfloat16, 16> one_b = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  aie::accum<accfloat, 16> one_acc;
  one_acc.from_vector(aie::broadcast<float, 16>(1.0f));
  float *__restrict g_p = gp;
  const float *__restrict u_p = gp + 512;
  for (int i = 0; i < 32; ++i) chess_prepare_for_pipelining chess_loop_count(32) {
    aie::vector<bfloat16, 16> gj = aie::accum<accfloat, 16>(aie::load_v<16>(g_p)).template to_vector<bfloat16>();
    aie::accum<accfloat, 16> hg = aie::mul<accfloat>(gj, half_b);
    aie::vector<bfloat16, 16> th(::tanh(hg.to_native()));
    aie::vector<bfloat16, 16> sv = aie::mac(one_acc, th, one_b).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> uj = aie::accum<accfloat, 16>(aie::load_v<16>(u_p)).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> hgu = aie::mul<accfloat>(hg.template to_vector<bfloat16>(), uj).template to_vector<bfloat16>();
    aie::store_v(g_p, aie::mul<accfloat>(hgu, sv).template to_vector<float>());
    g_p += 16;
    u_p += 16;
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
  alignas(32) union {
    float f[16];
    bfloat16 b[32];
  } sc_buf;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 16; ++i) {
    const float *__restrict r = src + (i & 7) * row_stride + (i >> 3) * blk2_off;
    sc_buf.f[i] = abs_max_32(r, r + col2_off);
  }
  aie::vector<float, 16> dv = q4_fmul_16(aie::load_v<16>(sc_buf.f), aie::broadcast<float, 16>(1.0f / 127.0f));
  // All callers pass 32B-aligned xd_out with d_dst1 == d_dst0 + 8 contiguous.
  aie::store_v(d_dst0, aie::accum<accfloat, 16>(dv).template to_vector<bfloat16>());
  aie::store_v(sc_buf.b, aie::accum<accfloat, 16>(q4_inv_16(dv)).template to_vector<bfloat16>());
#pragma clang loop unroll(disable)
  for (int i = 0; i < 16; ++i) {
    const int b = i >> 3;
    const int t = i & 7;
    const float *__restrict r = src + t * row_stride + b * blk2_off;
    aie::vector<bfloat16, 32> r01 = aie::concat(
        aie::accum<accfloat, 16>(aie::load_v<16>(r)).template to_vector<bfloat16>(),
        aie::accum<accfloat, 16>(aie::load_v<16>(r + col2_off)).template to_vector<bfloat16>());
    aie::vector<float, 32> p01 = aie::mul<accfloat>(r01, aie::broadcast<bfloat16, 32>(sc_buf.b[i])).template to_vector<float>();
    aie::vector<int8, 32> q01 = aie::to_fixed<int8>(p01, 0);
    uint8_t *__restrict qd = (b ? q_dst1 : q_dst0) + (t >> 2) * 128 + (t & 3) * 16;
    aie::store_v((int8_t *)qd, q01.template extract<16>(0));
    aie::store_v((int8_t *)(qd + 64), q01.template extract<16>(1));
  }
}

extern "C" {
void q4_lfm3_pre_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2);

void q4_lfm3_pre_x_op(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  q4_lfm3_pre_x_op_impl(x_pkt, op, a0, a1, a2);
}
}

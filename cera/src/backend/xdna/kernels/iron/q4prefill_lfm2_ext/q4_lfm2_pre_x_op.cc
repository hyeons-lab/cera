// Size-minimized (-Oz) math/activation helpers shared by FlashAttention and FFN SiLU.
#include "q4_lfm2_pre_state.h"
__attribute__((always_inline)) static inline aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x0) {
  aie::vector<bfloat16, 16> eb(::exp2(aie::accum<accfloat, 16>(x0).to_native()));
  return aie::accum<accfloat, 16>(eb).template to_vector<float>();
}

__attribute__((noinline, cold, minsize)) void q4_fa_rescale_2h(
    int32_t h_tag,
    aie::vector<float, 16> diff,
    uint32_t need) {
  const int32_t it = h_tag & 3;
  aie::vector<float, 16> alpha16 = q4_exp2f_16(diff);
  float *__restrict rlp = &g_scratch.attn.row_l[it][0];
  aie::store_v(rlp, q4_fmul_16(aie::load_v<16>(rlp), alpha16));
  float *__restrict abuf = (float *)g_scratch.attn.p_bf16;
  aie::store_v(abuf, alpha16);
  float *__restrict b0 = &g_scratch.attn.o_acc[it][0][0];
  float *__restrict b1 = g_x_ptr + it * 512;
#pragma clang loop unroll(disable)
  for (int idx = 0; idx < 16; ++idx) {
    if (!((need >> idx) & 1U)) continue;
    float *__restrict p = ((idx < 8) ? b0 : b1) + (idx & 7) * 8;
    aie::vector<float, 16> sv = aie::broadcast<float, 16>(abuf[idx]);
#pragma clang loop unroll(disable)
    for (int d16 = 0; d16 < (HEAD_DIM / 16); ++d16, p += 128) {
      aie::vector<float, 16> o = q4_fmul_16(aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + 64)), sv);
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

__attribute__((noinline, minsize)) aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v) { return aie::inv(aie::max(v, aie::broadcast<float, 16>(1e-35f))); }

// Hot FlashAttention microkernels compiled at -Os/-O2 with pre-packed V and hardware NLF.
#include "q4_lfm2_pre_state.h"

namespace {

__attribute__((always_inline)) static inline void load_8x32_rowstride(
    const bfloat16 *__restrict p,
    aie::vector<bfloat16, 64> &k0,
    aie::vector<bfloat16, 64> &k1,
    aie::vector<bfloat16, 64> &k2,
    aie::vector<bfloat16, 64> &k3) {
  aie::vector<int32_t, 16> r0 = aie::load_v<16>((const int32_t *)(p + 0 * HEAD_DIM));
  aie::vector<int32_t, 16> r1 = aie::load_v<16>((const int32_t *)(p + 1 * HEAD_DIM));
  aie::vector<int32_t, 16> r2 = aie::load_v<16>((const int32_t *)(p + 2 * HEAD_DIM));
  aie::vector<int32_t, 16> r3 = aie::load_v<16>((const int32_t *)(p + 3 * HEAD_DIM));
  aie::vector<int32_t, 16> r4 = aie::load_v<16>((const int32_t *)(p + 4 * HEAD_DIM));
  aie::vector<int32_t, 16> r5 = aie::load_v<16>((const int32_t *)(p + 5 * HEAD_DIM));
  aie::vector<int32_t, 16> r6 = aie::load_v<16>((const int32_t *)(p + 6 * HEAD_DIM));
  aie::vector<int32_t, 16> r7 = aie::load_v<16>((const int32_t *)(p + 7 * HEAD_DIM));
  auto [ac03, bd03] = aie::interleave_unzip(aie::concat(r0, r1), aie::concat(r2, r3), 4);
  auto [ac47, bd47] = aie::interleave_unzip(aie::concat(r4, r5), aie::concat(r6, r7), 4);
  auto [a, c] = aie::interleave_unzip(ac03, ac47, 4);
  auto [b, d] = aie::interleave_unzip(bd03, bd47, 4);
  k0 = a.template cast_to<bfloat16>();
  k1 = b.template cast_to<bfloat16>();
  k2 = c.template cast_to<bfloat16>();
  k3 = d.template cast_to<bfloat16>();
}

__attribute__((noinline)) static void q4_fa_exp_pack(
    const float (*__restrict sc_mat)[16],
    const float *__restrict m_ptr,
    uint32_t mask16,
    bfloat16 *__restrict p_dst,
    float *__restrict l_row) {
  aie::vector<float, 16> nm = aie::load_v<16>(m_ptr);
  aie::vector<bfloat16, 16> z_bf = aie::broadcast<bfloat16, 16>((bfloat16)0.0f);
  aie::mask<16> keep = aie::mask<16>::from_uint32(mask16);
  aie::accum<accfloat, 16> l16 = aie::zeros<accfloat, 16>();
  aie::vector<bfloat16, 16> ones16 = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const float *__restrict sp = &sc_mat[0][0];
  bfloat16 *__restrict ph0 = p_dst;
  bfloat16 *__restrict ph1 = p_dst + 256;
  aie::vector<bfloat16, 32> prev0, prev1;
  for (int kb2 = 0; kb2 < 8; ++kb2) chess_prepare_for_pipelining chess_loop_count(8) {
    aie::vector<bfloat16, 16> e0 = aie::select(z_bf, aie::vector<bfloat16, 16>(::exp2(aie::accum<accfloat, 16>(aie::sub(aie::load_v<16>(sp + 0), nm)).to_native())), keep);
    aie::vector<bfloat16, 16> e1 = aie::select(z_bf, aie::vector<bfloat16, 16>(::exp2(aie::accum<accfloat, 16>(aie::sub(aie::load_v<16>(sp + 16), nm)).to_native())), keep);
    aie::vector<bfloat16, 16> e2 = aie::select(z_bf, aie::vector<bfloat16, 16>(::exp2(aie::accum<accfloat, 16>(aie::sub(aie::load_v<16>(sp + 32), nm)).to_native())), keep);
    aie::vector<bfloat16, 16> e3 = aie::select(z_bf, aie::vector<bfloat16, 16>(::exp2(aie::accum<accfloat, 16>(aie::sub(aie::load_v<16>(sp + 48), nm)).to_native())), keep);
    sp += 64;
    l16 = aie::mac(l16, e0, ones16); l16 = aie::mac(l16, e1, ones16);
    l16 = aie::mac(l16, e2, ones16); l16 = aie::mac(l16, e3, ones16);
    auto [h0, h1] = aie::interleave_unzip(aie::concat(e0, e1), aie::concat(e2, e3), 8);
    if (kb2 & 1) {
      aie::store_v(ph0, aie::transpose(aie::concat(prev0, h0), 8, 8)); ph0 += 64;
      aie::store_v(ph1, aie::transpose(aie::concat(prev1, h1), 8, 8)); ph1 += 64;
    } else {
      prev0 = h0;
      prev1 = h1;
    }
  }
  float *__restrict sp_l = (float *)(const_cast<float (*)[16]>(sc_mat));
  aie::store_v(sp_l, l16.template to_vector<float>());
  volatile uintptr_t *pv = (volatile uintptr_t *)(sp_l + 16);
  pv[0] = (uintptr_t)sp_l;
  const float *rp = (const float *)pv[0];
  aie::store_v(l_row, aie::add(aie::load_v<16>(l_row), aie::load_v<16>(rp)));
}

}  // namespace

__attribute__((noinline)) void q4_fa_pack_v(const bfloat16 *__restrict v_mat, int32_t kb_max) {
  bfloat16 *__restrict base = g_scratch.attn.v_t;
  for (int kb = 0; kb < kb_max; ++kb) chess_prepare_for_pipelining chess_loop_range(1, 4) {
    const bfloat16 *__restrict vp = v_mat + kb * 8 * HEAD_DIM;
    for (int d32 = 0; d32 < 2; ++d32) chess_prepare_for_pipelining chess_loop_count(2) {
      aie::vector<bfloat16, 64> v0, v1, v2, v3;
      load_8x32_rowstride(vp + d32 * 32, v0, v1, v2, v3);
      bfloat16 *__restrict d0 = base + (d32 * 2 + 0) * 512 + kb * 128;
      bfloat16 *__restrict d1 = d0 + 512;
      aie::store_v(d0 + 0, aie::transpose(v0, 8, 8));
      aie::store_v(d0 + 64, aie::transpose(v1, 8, 8));
      aie::store_v(d1 + 0, aie::transpose(v2, 8, 8));
      aie::store_v(d1 + 64, aie::transpose(v3, 8, 8));
    }
  }
}

__attribute__((always_inline)) static inline void store_scores_2h(
    aie::accum<accfloat, 64> acc0,
    aie::accum<accfloat, 64> acc1,
    float *__restrict sd) {
  aie::vector<float, 64> v0 = acc0.template to_vector<float>();
  aie::vector<float, 64> v1 = acc1.template to_vector<float>();
  aie::vector<float, 32> t00 = aie::transpose(v0.template extract<32>(0), 4, 8);
  aie::vector<float, 32> t01 = aie::transpose(v0.template extract<32>(1), 4, 8);
  aie::vector<float, 32> t10 = aie::transpose(v1.template extract<32>(0), 4, 8);
  aie::vector<float, 32> t11 = aie::transpose(v1.template extract<32>(1), 4, 8);
#pragma clang loop unroll(full)
  for (int h = 0; h < 2; ++h) {
    auto [a0, a1] = aie::interleave_zip(t00.template extract<16>(h), t01.template extract<16>(h), 4);
    auto [b0, b1] = aie::interleave_zip(t10.template extract<16>(h), t11.template extract<16>(h), 4);
    auto [s0, s1] = aie::interleave_zip(a0, b0, 8);
    auto [s2, s3] = aie::interleave_zip(a1, b1, 8);
    aie::store_v(sd + (h * 4 + 0) * 16, s0);
    aie::store_v(sd + (h * 4 + 1) * 16, s1);
    aie::store_v(sd + (h * 4 + 2) * 16, s2);
    aie::store_v(sd + (h * 4 + 3) * 16, s3);
  }
}

__attribute__((noinline)) void q4_fa_qk_quad(
    const bfloat16 *__restrict q0a,
    const bfloat16 *__restrict q1a,
    const bfloat16 *__restrict q0b,
    const bfloat16 *__restrict q1b,
    const bfloat16 *__restrict kv_mat,
    int32_t kb_max,
    float (*__restrict sc0)[16],
    float (*__restrict sc1)[16]) {
  for (int kb = 0; kb < kb_max; ++kb) chess_prepare_for_pipelining chess_loop_range(1, 4) {
    aie::accum<accfloat, 64> acc0 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> acc1 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> acc2 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> acc3 = aie::zeros<accfloat, 64>();
    const bfloat16 *__restrict kp0 = kv_mat + kb * 8 * HEAD_DIM;
    for (int d32 = 0; d32 < (HEAD_DIM / 32); ++d32) {
      aie::vector<bfloat16, 64> ks[4];
      load_8x32_rowstride(kp0 + d32 * 32, ks[0], ks[1], ks[2], ks[3]);
      const int d8 = d32 * 4;
      for (int i = 0; i < 4; ++i) chess_prepare_for_pipelining chess_loop_count(4) {
        v64bfp16ebs8 kf = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(ks[i]));
        const int off = (d8 + i) * 64;
        acc0 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q0a + off))), kf, acc0, 0, 0, 0);
        acc1 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q1a + off))), kf, acc1, 0, 0, 0);
        acc2 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q0b + off))), kf, acc2, 0, 0, 0);
        acc3 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q1b + off))), kf, acc3, 0, 0, 0);
      }
    }
    store_scores_2h(acc0, acc1, &sc0[kb * 8][0]);
    store_scores_2h(acc2, acc3, &sc1[kb * 8][0]);
  }
}

__attribute__((noinline)) void q4_fa_pv_d16(
    const bfloat16 *__restrict,
    int32_t kb_max,
    int32_t h_tag) {
  const bfloat16 *__restrict p0 = &g_scratch.attn.p_bf16[0];
  const bfloat16 *__restrict p1 = &g_scratch.attn.p_bf16[256];
  const int32_t it = h_tag & 3;
  float *__restrict b0 = &g_scratch.attn.o_acc[it][0][0];
  float *__restrict b1 = g_x_ptr + it * 512;
#pragma clang loop unroll(disable)
  for (int d16 = 0; d16 < (HEAD_DIM / 16); ++d16) chess_loop_count(4) {
    float *__restrict dst0 = b0 + d16 * 128;
    float *__restrict dst1 = b1 + d16 * 128;
    aie::accum<accfloat, 64> a0_0(aie::load_v<64>(dst0));
    aie::accum<accfloat, 64> a1_0(aie::load_v<64>(dst0 + 64));
    aie::accum<accfloat, 64> a0_1(aie::load_v<64>(dst1));
    aie::accum<accfloat, 64> a1_1(aie::load_v<64>(dst1 + 64));
    const bfloat16 *__restrict vp = &g_scratch.attn.v_t[d16 * 512];
    const bfloat16 *__restrict pp0 = p0;
    const bfloat16 *__restrict pp1 = p1;
    for (int kb = 0; kb < kb_max; ++kb, vp += 128, pp0 += 64, pp1 += 64) chess_prepare_for_pipelining chess_loop_range(1, 4) {
      v64bfp16ebs8 pf0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pp0)));
      v64bfp16ebs8 pf1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(pp1)));
      v64bfp16ebs8 vt0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(vp)));
      a0_0 = ::mac_8x8_8x8T_conf(pf0, vt0, a0_0, 0, 0, 0);
      a0_1 = ::mac_8x8_8x8T_conf(pf1, vt0, a0_1, 0, 0, 0);
      v64bfp16ebs8 vt1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(vp + 64)));
      a1_0 = ::mac_8x8_8x8T_conf(pf0, vt1, a1_0, 0, 0, 0);
      a1_1 = ::mac_8x8_8x8T_conf(pf1, vt1, a1_1, 0, 0, 0);
    }
    aie::store_v(dst0, a0_0.template to_vector<float>());
    aie::store_v(dst0 + 64, a1_0.template to_vector<float>());
    aie::store_v(dst1, a0_1.template to_vector<float>());
    aie::store_v(dst1 + 64, a1_1.template to_vector<float>());
  }
}

__attribute__((noinline)) void q4_fa_softmax_2h(int32_t h_tag, uint32_t mask16, const float (*__restrict sc_mat)[16]) {
  const int32_t it = h_tag & 3;
  aie::vector<int32_t, 16> zi = aie::zeros<int32_t, 16>();
  aie::vector<int32_t, 16> mask7f = aie::broadcast<int32_t, 16>(0x7fffffff);
  aie::vector<int32_t, 16> ninf_i = aie::broadcast<int32_t, 16>(-1900671691);
#define Q4_FA_ORD(v) aie::select((v), aie::sub(mask7f, (v)), aie::lt((v), zi))
  aie::vector<int32_t, 16> k_chk0 = ninf_i;
  aie::vector<int32_t, 16> k_chk1 = ninf_i;
  const int32_t *__restrict sp = (const int32_t *)&sc_mat[0][0];
  for (int k = 0; k < (KV_CHUNK_T / 2); ++k) chess_prepare_for_pipelining chess_loop_count(16) {
    aie::vector<int32_t, 16> s0 = aie::load_v<16>(sp);
    aie::vector<int32_t, 16> s1 = aie::load_v<16>(sp + 16);
    sp += 32;
    k_chk0 = aie::max(k_chk0, Q4_FA_ORD(s0));
    k_chk1 = aie::max(k_chk1, Q4_FA_ORD(s1));
  }
  aie::vector<int32_t, 16> k_chk = aie::max(k_chk0, k_chk1);
  auto rmask = aie::mask<16>::from_uint32(mask16);
  k_chk = aie::select(ninf_i, k_chk, rmask);
  aie::vector<float, 16> old_m = aie::load_v<16>(&g_scratch.attn.row_m[it][0]);
  aie::vector<int32_t, 16> k_old = Q4_FA_ORD(old_m.template cast_to<int32_t>());
  aie::vector<float, 16> new_m = Q4_FA_ORD(aie::max(k_old, k_chk)).template cast_to<float>();
#undef Q4_FA_ORD
  aie::store_v(&g_scratch.attn.row_m[it][0], new_m);
  aie::vector<float, 16> rl = aie::load_v<16>(&g_scratch.attn.row_l[it][0]);
  uint32_t need = aie::lt(k_old, k_chk).to_uint32() & aie::lt(zi, rl.template cast_to<int32_t>()).to_uint32() & mask16;
  if (need) {
    q4_fa_rescale_2h(h_tag, aie::sub(old_m, new_m), need);
  }
  q4_fa_exp_pack(sc_mat, &g_scratch.attn.row_m[it][0], mask16, g_scratch.attn.p_bf16, &g_scratch.attn.row_l[it][0]);
}

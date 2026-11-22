#include "q4_lfm2_pre_state.h"

namespace {

__attribute__((always_inline)) static inline void load_8x16_rowstride(
    const bfloat16 *__restrict p,
    aie::vector<bfloat16, 64> &a,
    aie::vector<bfloat16, 64> &b) {
  aie::vector<int32_t, 8> r0 = aie::load_v<8>((const int32_t *)(p + 0 * HEAD_DIM));
  aie::vector<int32_t, 8> r1 = aie::load_v<8>((const int32_t *)(p + 1 * HEAD_DIM));
  aie::vector<int32_t, 8> r2 = aie::load_v<8>((const int32_t *)(p + 2 * HEAD_DIM));
  aie::vector<int32_t, 8> r3 = aie::load_v<8>((const int32_t *)(p + 3 * HEAD_DIM));
  aie::vector<int32_t, 8> r4 = aie::load_v<8>((const int32_t *)(p + 4 * HEAD_DIM));
  aie::vector<int32_t, 8> r5 = aie::load_v<8>((const int32_t *)(p + 5 * HEAD_DIM));
  aie::vector<int32_t, 8> r6 = aie::load_v<8>((const int32_t *)(p + 6 * HEAD_DIM));
  aie::vector<int32_t, 8> r7 = aie::load_v<8>((const int32_t *)(p + 7 * HEAD_DIM));
  auto [a01, b01] = aie::interleave_unzip(aie::concat(r0, r1), aie::concat(r2, r3), 4);
  auto [a45, b45] = aie::interleave_unzip(aie::concat(r4, r5), aie::concat(r6, r7), 4);
  a = aie::concat(a01.template cast_to<bfloat16>(), a45.template cast_to<bfloat16>());
  b = aie::concat(b01.template cast_to<bfloat16>(), b45.template cast_to<bfloat16>());
}

}  // namespace

__attribute__((noinline)) void q4_fa_qk_dual(
    const bfloat16 *__restrict q0_base,
    const bfloat16 *__restrict q1_base,
    const bfloat16 *__restrict kv_mat,
    int32_t kb_max) {
#pragma clang loop unroll(disable)
  for (int kb = 0; kb < kb_max; ++kb) {
    aie::accum<accfloat, 64> acc0 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> acc1 = aie::zeros<accfloat, 64>();
    const bfloat16 *__restrict kp0 = kv_mat + kb * 8 * HEAD_DIM;
    for (int d16 = 0; d16 < (HEAD_DIM / 16); ++d16) {
      aie::vector<bfloat16, 64> k0, k1;
      load_8x16_rowstride(kp0 + d16 * 16, k0, k1);
      v64bfp16ebs8 kf0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(k0));
      v64bfp16ebs8 kf1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(k1));
      const int d8 = d16 * 2;
      acc0 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q0_base + d8 * 64))), kf0, acc0, 0, 0, 0);
      acc1 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q1_base + d8 * 64))), kf0, acc1, 0, 0, 0);
      acc0 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q0_base + (d8 + 1) * 64))), kf1, acc0, 0, 0, 0);
      acc1 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q1_base + (d8 + 1) * 64))), kf1, acc1, 0, 0, 0);
    }
    aie::vector<float, 64> sv0 = acc0.template to_vector<float>();
    aie::vector<float, 64> sv1 = acc1.template to_vector<float>();
    float *__restrict sd0 = &g_scratch.attn.scores[0][0][kb * 8];
    float *__restrict sd1 = &g_scratch.attn.scores[1][0][kb * 8];
#pragma clang loop unroll(full)
    for (int t = 0; t < B_SUB; ++t) {
      aie::store_v(sd0 + t * KV_CHUNK_T, sv0.template extract<8>(t));
      aie::store_v(sd1 + t * KV_CHUNK_T, sv1.template extract<8>(t));
    }
  }
}

__attribute__((noinline)) void q4_fa_pv_d16(
    const bfloat16 *__restrict kv_mat,
    int32_t kb_max,
    int32_t h_base) {
  const bfloat16 *__restrict p0 = &g_scratch.attn.p_bf16[0];
  const bfloat16 *__restrict p1 = &g_scratch.attn.p_bf16[256];
#pragma clang loop unroll(disable)
  for (int d16 = 0; d16 < (HEAD_DIM / 16); ++d16) {
    aie::accum<accfloat, 64> a00 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> a01 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> a10 = aie::zeros<accfloat, 64>();
    aie::accum<accfloat, 64> a11 = aie::zeros<accfloat, 64>();
    for (int kb = 0; kb < kb_max; ++kb) {
      aie::vector<bfloat16, 64> v0, v1;
      load_8x16_rowstride(kv_mat + kb * 8 * HEAD_DIM + d16 * 16, v0, v1);
      v64bfp16ebs8 vt0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::transpose(v0, 8, 8)));
      v64bfp16ebs8 vt1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::transpose(v1, 8, 8)));
      v64bfp16ebs8 pf0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(p0 + kb * 64)));
      v64bfp16ebs8 pf1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(p1 + kb * 64)));
      a00 = ::mac_8x8_8x8T_conf(pf0, vt0, a00, 0, 0, 0);
      a01 = ::mac_8x8_8x8T_conf(pf0, vt1, a01, 0, 0, 0);
      a10 = ::mac_8x8_8x8T_conf(pf1, vt0, a10, 0, 0, 0);
      a11 = ::mac_8x8_8x8T_conf(pf1, vt1, a11, 0, 0, 0);
    }
    aie::vector<float, 64> ov00 = a00.template to_vector<float>();
    aie::vector<float, 64> ov01 = a01.template to_vector<float>();
    aie::vector<float, 64> ov10 = a10.template to_vector<float>();
    aie::vector<float, 64> ov11 = a11.template to_vector<float>();
    float *__restrict dst0 = &g_scratch.attn.o_acc[h_base][0][d16 * 16];
    float *__restrict dst1 = dst0 + B_SUB * HEAD_DIM;
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      aie::vector<float, 16> r0 = aie::concat(ov00.template extract<8>(t), ov01.template extract<8>(t));
      aie::vector<float, 16> r1 = aie::concat(ov10.template extract<8>(t), ov11.template extract<8>(t));
      aie::store_v(dst0 + t * HEAD_DIM, aie::add(aie::load_v<16>(dst0 + t * HEAD_DIM), r0));
      aie::store_v(dst1 + t * HEAD_DIM, aie::add(aie::load_v<16>(dst1 + t * HEAD_DIM), r1));
    }
  }
}

__attribute__((noinline, minsize)) float q4_fa_exp2_row32(
    const float *__restrict sc,
    float m_val,
    bfloat16 *__restrict p_col) {
  aie::vector<float, 16> mn = aie::broadcast<float, 16>(m_val);
  aie::vector<float, 16> ev0 = q4_exp2f_16(aie::sub(aie::load_v<16>(sc), mn));
  aie::vector<float, 16> ev1 = q4_exp2f_16(aie::sub(aie::load_v<16>(sc + 16), mn));
  float l_sum = aie::reduce_add(aie::add(ev0, ev1));
  aie::vector<bfloat16, 16> pb0 = aie::accum<accfloat, 16>(ev0).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> pb1 = aie::accum<accfloat, 16>(ev1).template to_vector<bfloat16>();
  aie::store_v(p_col + 0 * 64, pb0.template extract<8>(0));
  aie::store_v(p_col + 1 * 64, pb0.template extract<8>(1));
  aie::store_v(p_col + 2 * 64, pb1.template extract<8>(0));
  aie::store_v(p_col + 3 * 64, pb1.template extract<8>(1));
  return l_sum;
}

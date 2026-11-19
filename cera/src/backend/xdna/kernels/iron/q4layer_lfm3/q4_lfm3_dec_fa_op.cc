#include "q4_lfm3_dec_state.h"

namespace {
__attribute__((noinline)) static float fa_exp2_scalar(float x) {
  return dec_exp2f_16(aie::broadcast<float, 16>(x)).get(0);
}

__attribute__((noinline)) static float fa_qk_chunk(
    const bfloat16 *__restrict k_base,
    int32_t t0,
    int32_t t1) {
  aie::vector<bfloat16, 32> q0 = aie::load_v<32>(g_dec.q_bf16 + 0);
  aie::vector<bfloat16, 32> q1 = aie::load_v<32>(g_dec.q_bf16 + 32);
  aie::vector<bfloat16, 32> q2 = aie::load_v<32>(g_dec.q_bf16 + 64);
  aie::vector<bfloat16, 32> q3 = aie::load_v<32>(g_dec.q_bf16 + 96);

#pragma clang loop unroll(disable)
  for (int kt = t0; kt < t1; ++kt) chess_prepare_for_pipelining {
    const bfloat16 *__restrict kp = k_base + kt * HEAD_DIM;
    aie::vector<bfloat16, 32> k0 = aie::load_v<32>(kp + 0);
    aie::vector<bfloat16, 32> k1 = aie::load_v<32>(kp + 32);
    aie::vector<bfloat16, 32> k2 = aie::load_v<32>(kp + 64);
    aie::vector<bfloat16, 32> k3 = aie::load_v<32>(kp + 96);
    aie::accum<accfloat, 16> acc = aie::mul(q0.template extract<16>(0), k0.template extract<16>(0));
    acc = aie::mac(acc, q0.template extract<16>(1), k0.template extract<16>(1));
    acc = aie::mac(acc, q1.template extract<16>(0), k1.template extract<16>(0));
    acc = aie::mac(acc, q1.template extract<16>(1), k1.template extract<16>(1));
    acc = aie::mac(acc, q2.template extract<16>(0), k2.template extract<16>(0));
    acc = aie::mac(acc, q2.template extract<16>(1), k2.template extract<16>(1));
    acc = aie::mac(acc, q3.template extract<16>(0), k3.template extract<16>(0));
    acc = aie::mac(acc, q3.template extract<16>(1), k3.template extract<16>(1));
    g_dec.scores[kt] = aie::reduce_add(acc.template to_vector<float>());
  }

  float m_chk = aie::reduce_max(aie::load_v<16>(g_dec.scores));
  if (t1 > 16) {
    float hm = aie::reduce_max(aie::load_v<16>(g_dec.scores + 16));
    if (hm > m_chk) m_chk = hm;
  }
  return m_chk;
}

__attribute__((noinline)) static void fa_scale_o(float alpha) {
  aie::vector<float, 16> va = aie::broadcast<float, 16>(alpha);
#pragma clang loop unroll(disable)
  for (int d = 0; d < HEAD_DIM; d += 16) {
    aie::store_v(g_dec.o_acc + d, dec_fmul_16(aie::load_v<16>(g_dec.o_acc + d), va));
  }
}

__attribute__((noinline)) static void fa_v_half(
    const bfloat16 *__restrict v_base,
    const bfloat16 *__restrict p_bf,
    int32_t t0,
    int32_t t1,
    int32_t d_off,
    int32_t add_prev) {
  float *__restrict dst = g_dec.o_acc + d_off;
  aie::accum<accfloat, 16> o0, o1, o2, o3;
  if (add_prev) {
    o0.from_vector(aie::load_v<16>(dst + 0));
    o1.from_vector(aie::load_v<16>(dst + 16));
    o2.from_vector(aie::load_v<16>(dst + 32));
    o3.from_vector(aie::load_v<16>(dst + 48));
  } else {
    o0 = aie::zeros<accfloat, 16>();
    o1 = aie::zeros<accfloat, 16>();
    o2 = aie::zeros<accfloat, 16>();
    o3 = aie::zeros<accfloat, 16>();
  }
#pragma clang loop unroll(disable)
  for (int kt = t0; kt < t1; ++kt) chess_prepare_for_pipelining {
    aie::vector<bfloat16, 16> pv = aie::broadcast<bfloat16, 16>(p_bf[kt]);
    const bfloat16 *__restrict vp = v_base + kt * HEAD_DIM + d_off;
    aie::vector<bfloat16, 32> v0 = aie::load_v<32>(vp + 0);
    aie::vector<bfloat16, 32> v1 = aie::load_v<32>(vp + 32);
    o0 = aie::mac(o0, pv, v0.template extract<16>(0));
    o1 = aie::mac(o1, pv, v0.template extract<16>(1));
    o2 = aie::mac(o2, pv, v1.template extract<16>(0));
    o3 = aie::mac(o3, pv, v1.template extract<16>(1));
  }
  aie::store_v(dst + 0, o0.template to_vector<float>());
  aie::store_v(dst + 16, o1.template to_vector<float>());
  aie::store_v(dst + 32, o2.template to_vector<float>());
  aie::store_v(dst + 48, o3.template to_vector<float>());
}
}  // namespace

extern "C" void q4_lfm3_dec_fa_op(
    const uint8_t *__restrict w_pkt,
    int32_t row,
    int32_t chunk_idx,
    int32_t mode) {
  if (row >= 2) return;
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);

  const int32_t is_cur_kv = mode & 1;
  const int32_t is_first = (mode >> 1) & 1;

  if (is_first) {
    aie::vector<float, 16> zf = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
    for (int d = 0; d < HEAD_DIM; d += 16) aie::store_v(g_dec.o_acc + d, zf);
    g_dec.row_m = -1e30f;
    g_dec.row_l = 0.0f;
  }

  int32_t t0 = 0;
  int32_t t1 = 1;
  const bfloat16 *__restrict k_base;
  const bfloat16 *__restrict v_base;
  if (is_cur_kv) {
    k_base = (const bfloat16 *)(w_pkt + 128 * 4);
    v_base = (const bfloat16 *)(w_pkt + 192 * 4);
  } else {
    const int32_t pos = g_dec.params[0];
    const int32_t n_past = g_dec.params[3];
    const int32_t swa = g_dec.params[5];
    const int32_t kv_start_chunk = g_dec.params[8];
    const int32_t lo = (swa > 0 && (pos + 1) > swa) ? (pos + 1 - swa) : 0;
    const int32_t t_base = (kv_start_chunk + chunk_idx) * KV_CHUNK_T;
    t0 = (lo > t_base) ? (lo - t_base) : 0;
    const int32_t avail = n_past - t_base;
    t1 = (avail < KV_CHUNK_T) ? avail : KV_CHUNK_T;
    if (t0 >= t1) return;
    k_base = (const bfloat16 *)w_pkt;
    v_base = (const bfloat16 *)(w_pkt + 2304 * 4);
  }

  if (t0 > 0 || t1 < KV_CHUNK_T) {
    aie::vector<float, 16> vminf = aie::broadcast<float, 16>(-1e30f);
    aie::store_v(g_dec.scores, vminf);
    aie::store_v(g_dec.scores + 16, vminf);
  }

  float m_chk = fa_qk_chunk(k_base, t0, t1);
  float old_m = g_dec.row_m;
  int32_t has_prev = (g_dec.row_l > 0.0f);
  float new_m = (m_chk > old_m || !has_prev) ? m_chk : old_m;
  int32_t need_scale = has_prev && (m_chk > old_m);
  float alpha = need_scale ? fa_exp2_scalar(old_m - new_m) : 1.0f;

  aie::vector<float, 16> vn = aie::broadcast<float, 16>(new_m);
  alignas(64) bfloat16 p_bf[32];
  float sum_p = 0.0f;
  const int n_halves = (t1 > 16) ? 2 : 1;
#pragma clang loop unroll(disable)
  for (int h = 0; h < n_halves; ++h) {
    aie::vector<float, 16> p = dec_exp2f_16(aie::sub(aie::load_v<16>(g_dec.scores + h * 16), vn));
    sum_p += aie::reduce_add(p);
    aie::store_v(p_bf + h * 16, dec_f32_to_bf16_16(p));
  }

  if (need_scale) fa_scale_o(alpha);
  fa_v_half(v_base, p_bf, t0, t1, 0, has_prev);
  fa_v_half(v_base, p_bf, t0, t1, 64, has_prev);

  g_dec.row_l = (need_scale
      ? dec_fmul_16(aie::broadcast<float, 16>(g_dec.row_l), aie::broadcast<float, 16>(alpha)).get(0)
      : g_dec.row_l) + sum_p;
  g_dec.row_m = new_m;
}

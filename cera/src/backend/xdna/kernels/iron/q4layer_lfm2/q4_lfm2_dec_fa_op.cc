#include "q4_lfm2_dec_state.h"

namespace {
__attribute__((noinline)) static float fa_exp2_scalar(float x) {
  return dec_exp2f_16(aie::broadcast<float, 16>(x)).get(0);
}

__attribute__((always_inline)) static inline aie::vector<int32_t, 16> dec_fa_ord16(
    aie::vector<int32_t, 16> s,
    aie::vector<int32_t, 16> zi,
    aie::vector<int32_t, 16> mask7f) {
  return aie::select(s, aie::sub(mask7f, s), aie::lt(s, zi));
}

__attribute__((noinline, minsize)) static float fa_max32() {
  aie::vector<int32_t, 16> zi = aie::zeros<int32_t, 16>();
  aie::vector<int32_t, 16> mask7f = aie::broadcast<int32_t, 16>(0x7fffffff);
  aie::vector<int32_t, 16> k0 = dec_fa_ord16(aie::load_v<16>(g_dec.scores).template cast_to<int32_t>(), zi, mask7f);
  aie::vector<int32_t, 16> k1 = dec_fa_ord16(aie::load_v<16>(g_dec.scores + 16).template cast_to<int32_t>(), zi, mask7f);
  aie::vector<int32_t, 16> v = aie::max(k0, k1);
#pragma clang loop unroll(disable)
  for (int sh = 32; sh >= 4; sh >>= 1) {
    v = aie::max(v, aie::vector<int32_t, 16>(::shift_bytes(v, v, sh)));
  }
  int32_t kv = v.get(0);
  kv ^= (kv < 0 ? 0x7fffffff : 0);
  float fv;
  __builtin_memcpy(&fv, &kv, 4);
  return fv;
}

__attribute__((noinline, minsize)) static float fa_qk_tail(
    const bfloat16 *__restrict k_base,
    int32_t t0,
    int32_t t1) {
  aie::vector<bfloat16, 32> q0 = aie::load_v<32>(g_dec.q_bf16 + 0);
  aie::vector<bfloat16, 32> q1 = aie::load_v<32>(g_dec.q_bf16 + 32);
#pragma clang loop unroll(disable)
  for (int kt = t0; kt < t1; ++kt) {
    const bfloat16 *__restrict kp = k_base + kt * HEAD_DIM;
    aie::accum<accfloat, 32> acc = aie::mul(q0, aie::load_v<32>(kp + 0));
    acc = aie::mac(acc, q1, aie::load_v<32>(kp + 32));
    g_dec.scores[kt] = aie::reduce_add(acc.template to_vector<float>());
  }
  return fa_max32();
}

__attribute__((noinline)) static float fa_qk_chunk(
    const bfloat16 *__restrict k_base,
    int32_t t0,
    int32_t t1) {
  if (t0 != 0 || t1 != KV_CHUNK_T) {
    return fa_qk_tail(k_base, t0, t1);
  }
  aie::vector<bfloat16, 32> q0 = aie::load_v<32>(g_dec.q_bf16 + 0);
  aie::vector<bfloat16, 32> q1 = aie::load_v<32>(g_dec.q_bf16 + 32);
  for (int kb = 0; kb < 16; ++kb) chess_prepare_for_pipelining chess_loop_count(16) {
    const bfloat16 *__restrict kp = k_base + kb * 2 * HEAD_DIM;
    aie::accum<accfloat, 32> a0 = aie::mul(q0, aie::load_v<32>(kp + 0));
    a0 = aie::mac(a0, q1, aie::load_v<32>(kp + 32));
    aie::accum<accfloat, 32> a1 = aie::mul(q0, aie::load_v<32>(kp + 64));
    a1 = aie::mac(a1, q1, aie::load_v<32>(kp + 96));
    auto [e0, o0] = aie::interleave_unzip(a0.template to_vector<float>(), a1.template to_vector<float>(), 1);
    aie::vector<float, 32> s = aie::add(e0, o0);
    auto [e1, o1] = aie::interleave_unzip(s, s, 1);
    s = aie::add(e1, o1);
    auto [e2, o2] = aie::interleave_unzip(s, s, 1);
    s = aie::add(e2, o2);
    auto [e3, o3] = aie::interleave_unzip(s, s, 1);
    s = aie::add(e3, o3);
    auto [e4, o4] = aie::interleave_unzip(s, s, 1);
    s = aie::add(e4, o4);
    g_dec.scores[kb * 2 + 0] = s.get(0);
    g_dec.scores[kb * 2 + 1] = s.get(1);
  }
  return fa_max32();
}

__attribute__((noinline)) static float dec_fa_exp2_32(
    float m_val,
    bfloat16 *__restrict p_bf) {
  aie::vector<float, 16> vm = aie::broadcast<float, 16>(m_val);
  aie::vector<float, 16> vmin = aie::broadcast<float, 16>(-126.0f);
  aie::vector<float, 16> magic = aie::broadcast<float, 16>(12582912.0f);
  aie::vector<float, 16> bias = aie::broadcast<float, 16>(98431.0f);
  aie::vector<float, 16> x0 = aie::max(aie::sub(aie::load_v<16>(g_dec.scores + 0), vm), vmin);
  aie::vector<float, 16> x1 = aie::max(aie::sub(aie::load_v<16>(g_dec.scores + 16), vm), vmin);
  aie::vector<float, 16> kf0 = aie::sub(aie::add(x0, magic), magic);
  aie::vector<float, 16> kf1 = aie::sub(aie::add(x1, magic), magic);
  aie::vector<bfloat16, 32> fb = aie::concat(
      aie::accum<accfloat, 16>(aie::sub(x0, kf0)).template to_vector<bfloat16>(),
      aie::accum<accfloat, 16>(aie::sub(x1, kf1)).template to_vector<bfloat16>());
  aie::vector<bfloat16, 32> eab = aie::concat(
      aie::filter_even(aie::add(kf0, bias).template cast_to<bfloat16>(), 1),
      aie::filter_even(aie::add(kf1, bias).template cast_to<bfloat16>(), 1));
  aie::vector<bfloat16, 32> pb = aie::broadcast<bfloat16, 32>((bfloat16)0.0555041087f);
  aie::accum<accfloat, 32> acc;
  acc.from_vector(aie::broadcast<float, 32>(0.2402265069f));
  pb = aie::mac(acc, pb, fb).template to_vector<bfloat16>();
  acc.from_vector(aie::broadcast<float, 32>(0.6931471805f));
  pb = aie::mac(acc, pb, fb).template to_vector<bfloat16>();
  acc.from_vector(aie::broadcast<float, 32>(1.0f));
  pb = aie::mac(acc, pb, fb).template to_vector<bfloat16>();
  aie::accum<accfloat, 32> ea = aie::mul<accfloat>(pb, eab);
  aie::store_v(p_bf, ea.template to_vector<bfloat16>());
  aie::vector<float, 32> ef = ea.template to_vector<float>();
  return aie::reduce_add(aie::add(ef.template extract<16>(0), ef.template extract<16>(1)));
}

__attribute__((noinline, minsize)) static void fa_scale_o(float alpha) {
  aie::vector<float, 16> va = aie::broadcast<float, 16>(alpha);
#pragma clang loop unroll(disable)
  for (int d = 0; d < HEAD_DIM; d += 16) {
    aie::store_v(g_dec.o_acc + d, dec_fmul_16(aie::load_v<16>(g_dec.o_acc + d), va));
  }
}

__attribute__((noinline)) static void fa_pv_chunk(
    const bfloat16 *__restrict v_base,
    const bfloat16 *__restrict p_bf,
    int32_t t0,
    int32_t t1,
    int32_t add_prev) {
  aie::accum<accfloat, 32> o0, o1;
  if (add_prev) {
    o0.from_vector(aie::load_v<32>(g_dec.o_acc + 0));
    o1.from_vector(aie::load_v<32>(g_dec.o_acc + 32));
  } else {
    o0 = aie::zeros<accfloat, 32>();
    o1 = aie::zeros<accfloat, 32>();
  }
  if (t0 == 0 && t1 == KV_CHUNK_T) {
    aie::accum<accfloat, 32> o0_b = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> o1_b = aie::zeros<accfloat, 32>();
    for (int kt = 0; kt < KV_CHUNK_T; kt += 2) chess_prepare_for_pipelining chess_loop_count(16) {
      const bfloat16 *__restrict vp0 = v_base + kt * HEAD_DIM;
      const bfloat16 *__restrict vp1 = vp0 + HEAD_DIM;
      aie::vector<bfloat16, 32> pv0 = aie::broadcast<bfloat16, 32>(p_bf[kt]);
      aie::vector<bfloat16, 32> pv1 = aie::broadcast<bfloat16, 32>(p_bf[kt + 1]);
      o0 = aie::mac(o0, pv0, aie::load_v<32>(vp0 + 0));
      o1 = aie::mac(o1, pv0, aie::load_v<32>(vp0 + 32));
      o0_b = aie::mac(o0_b, pv1, aie::load_v<32>(vp1 + 0));
      o1_b = aie::mac(o1_b, pv1, aie::load_v<32>(vp1 + 32));
    }
    o0 = aie::add(o0, o0_b);
    o1 = aie::add(o1, o1_b);
  } else {
    for (int kt = t0; kt < t1; ++kt) chess_prepare_for_pipelining chess_loop_range(1, 31) {
      aie::vector<bfloat16, 32> pv = aie::broadcast<bfloat16, 32>(p_bf[kt]);
      const bfloat16 *__restrict vp = v_base + kt * HEAD_DIM;
      o0 = aie::mac(o0, pv, aie::load_v<32>(vp + 0));
      o1 = aie::mac(o1, pv, aie::load_v<32>(vp + 32));
    }
  }
  aie::store_v(g_dec.o_acc + 0, o0.template to_vector<float>());
  aie::store_v(g_dec.o_acc + 32, o1.template to_vector<float>());
}
}  // namespace

extern "C" __attribute__((minsize)) void q4_lfm2_dec_fa_op(
    const uint8_t *__restrict w_pkt,
    int32_t _row,
    int32_t chunk_idx,
    int32_t mode) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t is_cur_kv = mode & 1;
  if (is_cur_kv && g_dec.params[10]) return;

  int32_t *__restrict st = (int32_t *)(g_dec.kv_bf16 + 32);
  if (mode & 4) {
    const int32_t t0 = st[0];
    const int32_t t1 = st[1];
    if (t0 < t1) {
      fa_pv_chunk((const bfloat16 *)w_pkt, g_dec.kv_bf16, t0, t1, st[2]);
    }
    return;
  }

  if ((mode & 2) && ((g_dec.params[6] <= 2) || (g_dec.params[6] == 6) || !g_dec.params[9])) {
    aie::vector<float, 16> zf = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
    for (int d = 0; d < HEAD_DIM; d += 16) aie::store_v(g_dec.o_acc + d, zf);
    g_dec.row_m = -1e30f;
    g_dec.row_l = 0.0f;
  }

  int32_t t0 = 0;
  int32_t t1 = 1;
  const bfloat16 *__restrict k_base;
  const bfloat16 *__restrict v_base = nullptr;
  if (is_cur_kv) {
    k_base = (const bfloat16 *)(w_pkt + 128 * 4);
    v_base = (const bfloat16 *)(w_pkt + 192 * 4);
  } else {
    const int32_t pos = g_dec.params[0];
    const int32_t n_past = g_dec.params[3];
    const int32_t swa = g_dec.params[5];
    const int32_t lo = (swa > 0 && (pos + 1) > swa) ? (pos + 1 - swa) : 0;
    const int32_t t_base = g_dec.params[8] * (2 * KV_CHUNK_T) + chunk_idx * KV_CHUNK_T;
    t0 = (lo > t_base) ? (lo - t_base) : 0;
    const int32_t avail = n_past - t_base;
    t1 = (avail < KV_CHUNK_T) ? avail : KV_CHUNK_T;
    if (t0 >= t1) {
      st[0] = 0;
      st[1] = 0;
      return;
    }
    k_base = (const bfloat16 *)w_pkt;
  }

  if (t0 > 0 || t1 < KV_CHUNK_T) {
    aie::vector<float, 16> vminf = aie::broadcast<float, 16>(-1e30f);
    aie::store_v(g_dec.scores, vminf);
    aie::store_v(g_dec.scores + 16, vminf);
  }

  float m_chk = fa_qk_chunk(k_base, t0, t1);
  float old_m = g_dec.row_m;
  uint32_t rl_bits = 0;
  __builtin_memcpy(&rl_bits, &g_dec.row_l, 4);
  int32_t has_prev = (rl_bits & 0x7FFFFFFFU) != 0;
  int32_t oi = 0, mi = 0;
  __builtin_memcpy(&oi, &old_m, 4);
  __builtin_memcpy(&mi, &m_chk, 4);
  oi ^= (oi < 0 ? 0x7fffffff : 0);
  mi ^= (mi < 0 ? 0x7fffffff : 0);
  int32_t m_higher = (oi < mi);
  float new_m = (m_higher || !has_prev) ? m_chk : old_m;
  int32_t need_scale = has_prev && m_higher;
  float alpha = need_scale ? fa_exp2_scalar(old_m - new_m) : 1.0f;

  float sum_p = dec_fa_exp2_32(new_m, g_dec.kv_bf16);

  if (need_scale) fa_scale_o(alpha);
  float scaled_l = need_scale
      ? dec_fmul_16(aie::broadcast<float, 16>(g_dec.row_l), aie::broadcast<float, 16>(alpha)).get(0)
      : g_dec.row_l;
  g_dec.row_l = scaled_l + sum_p;
  g_dec.row_m = new_m;

  if (is_cur_kv) {
    fa_pv_chunk(v_base, g_dec.kv_bf16, t0, t1, has_prev);
  } else {
    st[0] = t0;
    st[1] = t1;
    st[2] = has_prev;
  }
}

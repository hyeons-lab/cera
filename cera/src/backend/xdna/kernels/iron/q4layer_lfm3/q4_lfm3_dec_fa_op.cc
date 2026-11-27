#include "q4_lfm3_dec_state.h"

namespace {
__attribute__((always_inline)) static inline aie::vector<int32_t, 16> dec_fa_ord16(
    aie::vector<int32_t, 16> s,
    aie::vector<int32_t, 16> zi,
    aie::vector<int32_t, 16> mask7f) {
  return aie::select(s, aie::sub(mask7f, s), aie::lt(s, zi));
}

__attribute__((always_inline)) static inline float fa_max32() {
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

__attribute__((noinline)) static float fa_qk_tail(
    const bfloat16 *__restrict k_base,
    int32_t t0,
    int32_t t1) {
  aie::vector<float, 16> vminf = aie::broadcast<float, 16>(-1e30f);
  aie::store_v(g_dec.scores, vminf);
  aie::store_v(g_dec.scores + 16, vminf);
  aie::vector<bfloat16, 32> q0 = aie::load_v<32>(g_dec.q_bf16 + 0);
  aie::vector<bfloat16, 32> q1 = aie::load_v<32>(g_dec.q_bf16 + 32);
  aie::vector<bfloat16, 32> q2 = aie::load_v<32>(g_dec.q_bf16 + 64);
  aie::vector<bfloat16, 32> q3 = aie::load_v<32>(g_dec.q_bf16 + 96);
#pragma clang loop unroll(disable)
  for (int kt = t0; kt < t1; ++kt) {
    const bfloat16 *__restrict kp = k_base + kt * HEAD_DIM;
    aie::accum<accfloat, 32> acc = aie::mul(q0, aie::load_v<32>(kp + 0));
    acc = aie::mac(acc, q1, aie::load_v<32>(kp + 32));
    acc = aie::mac(acc, q2, aie::load_v<32>(kp + 64));
    acc = aie::mac(acc, q3, aie::load_v<32>(kp + 96));
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
  aie::vector<bfloat16, 32> q2 = aie::load_v<32>(g_dec.q_bf16 + 64);
  aie::vector<bfloat16, 32> q3 = aie::load_v<32>(g_dec.q_bf16 + 96);
  for (int kb = 0; kb < 16; ++kb) chess_prepare_for_pipelining chess_loop_count(16) {
    const bfloat16 *__restrict kp = k_base + kb * 2 * HEAD_DIM;
    aie::accum<accfloat, 32> a0 = aie::mul(q0, aie::load_v<32>(kp + 0));
    a0 = aie::mac(a0, q1, aie::load_v<32>(kp + 32));
    a0 = aie::mac(a0, q2, aie::load_v<32>(kp + 64));
    a0 = aie::mac(a0, q3, aie::load_v<32>(kp + 96));
    aie::accum<accfloat, 32> a1 = aie::mul(q0, aie::load_v<32>(kp + 128));
    a1 = aie::mac(a1, q1, aie::load_v<32>(kp + 160));
    a1 = aie::mac(a1, q2, aie::load_v<32>(kp + 192));
    a1 = aie::mac(a1, q3, aie::load_v<32>(kp + 224));
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
    bfloat16 *p_bf) {
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

__attribute__((noinline)) static void fa_v_tail(
    const bfloat16 *__restrict v_base,
    const bfloat16 *__restrict p_bf,
    int32_t t0,
    int32_t t1,
    float *__restrict dst,
    int32_t add_prev) {
#pragma clang loop unroll(disable)
  for (int d = 0; d < HEAD_DIM; d += 32) {
    aie::accum<accfloat, 32> o;
    if (add_prev) o.from_vector(aie::load_v<32>(dst + d));
    else o = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
    for (int kt = t0; kt < t1; ++kt) {
      o = aie::mac(o, aie::broadcast<bfloat16, 32>(p_bf[kt]), aie::load_v<32>(v_base + kt * HEAD_DIM + d));
    }
    aie::store_v(dst + d, o.template to_vector<float>());
  }
}

__attribute__((noinline)) static void fa_v_128(
    const bfloat16 *__restrict v_base,
    const bfloat16 *__restrict p_bf,
    int32_t t0,
    int32_t t1,
    int32_t add_prev) {
  float *__restrict dst = g_dec.o_acc;
  if (t0 != 0 || t1 != KV_CHUNK_T) {
    fa_v_tail(v_base, p_bf, t0, t1, dst, add_prev);
    return;
  }
  aie::accum<accfloat, 32> o0, o1, o2, o3;
  if (add_prev) {
    o0.from_vector(aie::load_v<32>(dst + 0));
    o1.from_vector(aie::load_v<32>(dst + 32));
    o2.from_vector(aie::load_v<32>(dst + 64));
    o3.from_vector(aie::load_v<32>(dst + 96));
  } else {
    o0 = aie::zeros<accfloat, 32>();
    o1 = aie::zeros<accfloat, 32>();
    o2 = aie::zeros<accfloat, 32>();
    o3 = aie::zeros<accfloat, 32>();
  }
  for (int kt = 0; kt < KV_CHUNK_T; ++kt) chess_prepare_for_pipelining chess_loop_count(32) {
    const bfloat16 *__restrict vp = v_base + kt * HEAD_DIM;
    aie::vector<bfloat16, 32> pv = aie::broadcast<bfloat16, 32>(p_bf[kt]);
    o0 = aie::mac(o0, pv, aie::load_v<32>(vp + 0));
    o1 = aie::mac(o1, pv, aie::load_v<32>(vp + 32));
    o2 = aie::mac(o2, pv, aie::load_v<32>(vp + 64));
    o3 = aie::mac(o3, pv, aie::load_v<32>(vp + 96));
  }
  aie::store_v(dst + 0, o0.template to_vector<float>());
  aie::store_v(dst + 32, o1.template to_vector<float>());
  aie::store_v(dst + 64, o2.template to_vector<float>());
  aie::store_v(dst + 96, o3.template to_vector<float>());
}
}  // namespace

extern "C" void q4_lfm3_dec_fa_op(
    const uint8_t *__restrict w_pkt,
    int32_t row,
    int32_t chunk_idx,
    int32_t mode) {
  (void)row;
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);

  const int32_t is_cur_kv = mode & 1;
  const int32_t is_first = (mode >> 1) & 1;

  if (is_first) dec_reset_fa_state();

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

  float m_chk = fa_qk_chunk(k_base, t0, t1);
  float old_m = g_dec.row_m;
  uint32_t rl_bits = 0;
  __builtin_memcpy(&rl_bits, &g_dec.row_l, 4);
  int32_t has_prev = (rl_bits & 0x7FFFFFFFU) != 0;
  float max_m = aie::max(aie::broadcast<float, 16>(old_m), aie::broadcast<float, 16>(m_chk)).get(0);
  float new_m = has_prev ? max_m : m_chk;
  int32_t oi = 0, ni = 0;
  __builtin_memcpy(&oi, &old_m, 4);
  __builtin_memcpy(&ni, &new_m, 4);
  int32_t need_scale = has_prev && (oi != ni);
  float alpha = need_scale
      ? dec_exp2f_16(aie::sub(aie::broadcast<float, 16>(old_m), aie::broadcast<float, 16>(new_m))).get(0)
      : 1.0f;

  bfloat16 *p_bf = reinterpret_cast<bfloat16 *>(g_dec.scores);
  float sum_p = dec_fa_exp2_32(new_m, p_bf);

  if (need_scale) dec_scale_o(alpha);
  fa_v_128(v_base, p_bf, t0, t1, has_prev);

  g_dec.row_l = aie::add(
      dec_fmul_16(aie::broadcast<float, 16>(g_dec.row_l), aie::broadcast<float, 16>(alpha)),
      aie::broadcast<float, 16>(sum_p)).get(0);
  g_dec.row_m = new_m;
}

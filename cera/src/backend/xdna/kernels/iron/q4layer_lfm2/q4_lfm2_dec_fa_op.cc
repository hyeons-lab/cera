#include "q4_lfm2_dec_state.h"

namespace {
constexpr float kScale = 0.125f;  // 1 / sqrt(64)
constexpr float kLog2e = 1.4426950408889634f;

#define fa_vmul_16 dec_fmul_16

__attribute__((always_inline)) static inline float fa_fmul_s(float a, float b) {
  return fa_vmul_16(aie::broadcast<float, 16>(a), aie::broadcast<float, 16>(b)).get(0);
}

__attribute__((always_inline)) static inline float fa_max_s(float a, float b) {
  return aie::max(aie::broadcast<float, 16>(a), aie::broadcast<float, 16>(b)).get(0);
}

__attribute__((always_inline)) static inline bool fa_nz_f32(float x) {
  uint32_t b;
  __builtin_memcpy(&b, &x, 4);
  return (b & 0x7FFFFFFFu) != 0;
}

__attribute__((noinline)) static float fa_exp2_scalar(float x) {
  return dec_exp2f_16(aie::broadcast<float, 16>(x)).get(0);
}

__attribute__((noinline)) static void fa_qk_chunk(const bfloat16 *__restrict k_base) {
  aie::vector<bfloat16, 16> q0 = aie::load_v<16>(g_dec.q_bf16 + 0);
  aie::vector<bfloat16, 16> q1 = aie::load_v<16>(g_dec.q_bf16 + 16);
  aie::vector<bfloat16, 16> q2 = aie::load_v<16>(g_dec.q_bf16 + 32);
  aie::vector<bfloat16, 16> q3 = aie::load_v<16>(g_dec.q_bf16 + 48);
  for (int kt = 0; kt < KV_CHUNK_T; ++kt) chess_prepare_for_pipelining {
    const bfloat16 *__restrict kp = k_base + kt * HEAD_DIM;
    aie::accum<accfloat, 16> acc = aie::mul(q0, aie::load_v<16>(kp + 0));
    acc = aie::mac(acc, q1, aie::load_v<16>(kp + 16));
    acc = aie::mac(acc, q2, aie::load_v<16>(kp + 32));
    acc = aie::mac(acc, q3, aie::load_v<16>(kp + 48));
    g_dec.scores[kt] = aie::reduce_add(acc.template to_vector<float>());
  }
}

__attribute__((noinline)) static void fa_pv_chunk(
    const bfloat16 *__restrict v_base,
    float alpha,
    int32_t t0,
    int32_t t1) {
  aie::vector<float, 16> va = aie::broadcast<float, 16>(alpha);
  aie::accum<accfloat, 16> o0(dec_fmul_16(aie::load_v<16>(g_dec.o_acc + 0), va));
  aie::accum<accfloat, 16> o1(dec_fmul_16(aie::load_v<16>(g_dec.o_acc + 16), va));
  aie::accum<accfloat, 16> o2(dec_fmul_16(aie::load_v<16>(g_dec.o_acc + 32), va));
  aie::accum<accfloat, 16> o3(dec_fmul_16(aie::load_v<16>(g_dec.o_acc + 48), va));
  for (int kt = t0; kt < t1; ++kt) chess_prepare_for_pipelining {
    const bfloat16 *__restrict vp = v_base + kt * HEAD_DIM;
    aie::vector<bfloat16, 16> vpk = dec_f32_to_bf16_16(aie::broadcast<float, 16>(g_dec.scores[kt]));
    o0 = aie::mac(o0, aie::load_v<16>(vp + 0), vpk);
    o1 = aie::mac(o1, aie::load_v<16>(vp + 16), vpk);
    o2 = aie::mac(o2, aie::load_v<16>(vp + 32), vpk);
    o3 = aie::mac(o3, aie::load_v<16>(vp + 48), vpk);
  }
  aie::store_v(g_dec.o_acc + 0, o0.template to_vector<float>());
  aie::store_v(g_dec.o_acc + 16, o1.template to_vector<float>());
  aie::store_v(g_dec.o_acc + 32, o2.template to_vector<float>());
  aie::store_v(g_dec.o_acc + 48, o3.template to_vector<float>());
}
}  // namespace

extern "C" __attribute__((minsize)) void q4_lfm2_dec_fa_op(
    const uint8_t *__restrict w_pkt,
    int32_t _row,
    int32_t chunk_idx,
    int32_t mode) {
  const int32_t is_cur_kv = mode & 1;
  if (is_cur_kv && g_dec.params[10]) return;
  const int32_t is_first = ((mode >> 1) & 1) && ((g_dec.params[6] <= 2) || !g_dec.params[9]);

  if (is_first) {
    aie::vector<float, 16> zf = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
    for (int d = 0; d < HEAD_DIM; d += 16) aie::store_v(g_dec.o_acc + d, zf);
    g_dec.row_m = -1e30f;
    g_dec.row_l = 0.0f;
  }

  if (is_cur_kv) {
    const bfloat16 *__restrict kc = (const bfloat16 *)(w_pkt + 128 * 4);
    const bfloat16 *__restrict vc = (const bfloat16 *)(w_pkt + 192 * 4);
    float s;
    {
      aie::accum<accfloat, 16> acc = aie::mul(aie::load_v<16>(g_dec.q_bf16 + 0), aie::load_v<16>(kc + 0));
      acc = aie::mac(acc, aie::load_v<16>(g_dec.q_bf16 + 16), aie::load_v<16>(kc + 16));
      acc = aie::mac(acc, aie::load_v<16>(g_dec.q_bf16 + 32), aie::load_v<16>(kc + 32));
      acc = aie::mac(acc, aie::load_v<16>(g_dec.q_bf16 + 48), aie::load_v<16>(kc + 48));
      s = fa_fmul_s(aie::reduce_add(acc.template to_vector<float>()), kScale);
    }
    float old_m = g_dec.row_m;
    float new_m = fa_max_s(s, old_m);
    float alpha = fa_nz_f32(g_dec.row_l) ? fa_exp2_scalar(fa_fmul_s(old_m - new_m, kLog2e)) : 0.0f;
    float p = fa_exp2_scalar(fa_fmul_s(s - new_m, kLog2e));
    g_dec.scores[0] = p;
    fa_pv_chunk(vc, alpha, 0, 1);
    g_dec.row_l = fa_fmul_s(g_dec.row_l, alpha) + p;
    g_dec.row_m = new_m;
    return;
  }

  const int32_t pos = g_dec.params[0];
  const int32_t n_past = g_dec.params[3];
  const int32_t swa = g_dec.params[5];
  const int32_t lo = (swa > 0 && (pos + 1) > swa) ? (pos + 1 - swa) : 0;
  const int32_t t_base = (g_dec.params[8] + chunk_idx) * KV_CHUNK_T;
  const int32_t t0 = (lo > t_base) ? (lo - t_base) : 0;
  const int32_t avail = n_past - t_base;
  const int32_t t1 = (avail < KV_CHUNK_T) ? avail : KV_CHUNK_T;
  if (t0 >= t1) return;

  const bfloat16 *__restrict k_base = (const bfloat16 *)w_pkt;
  const bfloat16 *__restrict v_base = (const bfloat16 *)(w_pkt + 2304 * 4);

  fa_qk_chunk(k_base);

  float m_chk;
  const bool partial = (t0 > 0) || (t1 < KV_CHUNK_T);
  {
    aie::vector<float, 16> vscale = aie::broadcast<float, 16>(kScale);
    aie::vector<float, 16> s0 = fa_vmul_16(aie::load_v<16>(g_dec.scores + 0), vscale);
    aie::vector<float, 16> s1 = fa_vmul_16(aie::load_v<16>(g_dec.scores + 16), vscale);
    aie::store_v(g_dec.scores + 0, s0);
    aie::store_v(g_dec.scores + 16, s1);
    if (partial) {
#pragma clang loop unroll(disable)
      for (int kt = 0; kt < t0; ++kt) g_dec.scores[kt] = -1e30f;
#pragma clang loop unroll(disable)
      for (int kt = t1; kt < KV_CHUNK_T; ++kt) g_dec.scores[kt] = -1e30f;
      s0 = aie::load_v<16>(g_dec.scores + 0);
      s1 = aie::load_v<16>(g_dec.scores + 16);
    }
    m_chk = aie::reduce_max(aie::max(s0, s1));
  }

  float old_m = g_dec.row_m;
  float new_m = fa_max_s(m_chk, old_m);
  float alpha = fa_nz_f32(g_dec.row_l) ? fa_exp2_scalar(fa_fmul_s(old_m - new_m, kLog2e)) : 0.0f;

  float sum_p;
  {
    aie::vector<float, 16> vn = aie::broadcast<float, 16>(new_m);
    aie::vector<float, 16> vl2e = aie::broadcast<float, 16>(kLog2e);
    aie::vector<float, 16> p0 = dec_exp2f_16(fa_vmul_16(aie::sub(aie::load_v<16>(g_dec.scores + 0), vn), vl2e));
    aie::vector<float, 16> p1 = dec_exp2f_16(fa_vmul_16(aie::sub(aie::load_v<16>(g_dec.scores + 16), vn), vl2e));
    aie::store_v(g_dec.scores + 0, p0);
    aie::store_v(g_dec.scores + 16, p1);
    if (partial) {
#pragma clang loop unroll(disable)
      for (int kt = 0; kt < t0; ++kt) g_dec.scores[kt] = 0.0f;
#pragma clang loop unroll(disable)
      for (int kt = t1; kt < KV_CHUNK_T; ++kt) g_dec.scores[kt] = 0.0f;
      p0 = aie::load_v<16>(g_dec.scores + 0);
      p1 = aie::load_v<16>(g_dec.scores + 16);
    }
    sum_p = aie::reduce_add(aie::add(p0, p1));
  }

  fa_pv_chunk(v_base, alpha, t0, t1);

  g_dec.row_l = fa_fmul_s(g_dec.row_l, alpha) + sum_p;
  g_dec.row_m = new_m;
}

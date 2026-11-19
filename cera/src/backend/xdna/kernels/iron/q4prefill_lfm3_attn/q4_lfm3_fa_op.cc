#include "q4_lfm3_attn_state.h"

static void q4_lfm3_fa_op(
    const float *__restrict kv_pkt,
    int32_t row,
    int32_t chunk_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t is_cur_kv = flags & 1;
  if (is_cur_kv && g_params[10]) return;
  const int32_t is_first = ((flags >> 1) & 1) && ((g_params[6] <= 3) || !g_params[9]);

  if (is_first) {
    q4_fill_f32_16(&g_scratch.attn.o_acc[0][0][0], 0.0f, GQA_Q_HEADS * B_CORE * (HEAD_DIM + 1));
    q4_fill_f32_16(&g_scratch.attn.row_m[0][0], -1e30f, 16);
  }

  const int32_t pos0 = g_params[0];
  const int32_t n_valid = g_params[1];
  const int32_t n_past = g_params[3];
  const int32_t is_causal = g_params[4];
  const int32_t swa_window = g_params[5];
  const int32_t kv_start_chunk = g_params[8];

  const int32_t chunk_gpos0 = is_cur_kv ? pos0 : ((kv_start_chunk + chunk_idx) * KV_CHUNK_T);
  int32_t valid_len = is_cur_kv ? n_valid : (n_past - chunk_gpos0);
  if (valid_len <= 0) return;
  if (valid_len > KV_CHUNK_T) valid_len = KV_CHUNK_T;

  const bfloat16 *__restrict k_mat = (const bfloat16 *)kv_pkt;
  const bfloat16 *__restrict v_mat = (const bfloat16 *)(kv_pkt + SILU_HALF_F32);
  aie::vector<float, 16> log2e_v = aie::broadcast<float, 16>(LOG2E);
  bfloat16 *__restrict p_flat = g_scratch.attn.p_bf16;
  alignas(64) float tmp8x8[64];

  using mmul888 = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
  const int32_t kb_max = (valid_len + 7) >> 3;

#pragma clang loop unroll(disable)
  for (int h = 0; h < GQA_Q_HEADS; ++h) {
    const bfloat16 *__restrict q_base = g_q_bf16 + h * (HEAD_DIM / 8) * 64;
#pragma clang loop unroll(disable)
    for (int kb = 0; kb < kb_max; ++kb) {
      aie::accum<accfloat, 64> acc;
      const bfloat16 *__restrict kp0 = k_mat + kb * 8 * HEAD_DIM;
      for (int d8 = 0; d8 < (HEAD_DIM / 8); ++d8) chess_prepare_for_pipelining {
        const bfloat16 *__restrict kp = kp0 + d8 * 8;
        aie::vector<bfloat16, 64> kv = aie::concat(
            aie::concat(aie::load_v<8>(kp + 0 * HEAD_DIM), aie::load_v<8>(kp + 1 * HEAD_DIM)),
            aie::concat(aie::load_v<8>(kp + 2 * HEAD_DIM), aie::load_v<8>(kp + 3 * HEAD_DIM)),
            aie::concat(aie::load_v<8>(kp + 4 * HEAD_DIM), aie::load_v<8>(kp + 5 * HEAD_DIM)),
            aie::concat(aie::load_v<8>(kp + 6 * HEAD_DIM), aie::load_v<8>(kp + 7 * HEAD_DIM)));
        aie::vector<bfloat16, 64> qv = aie::load_v<64>(q_base + d8 * 64);
        v64bfp16ebs8 aq = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(qv));
        v64bfp16ebs8 bk = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(kv));
        acc = ::mac_8x8_8x8T_conf(aq, bk, acc, d8 == 0, 0, 0);
      }
      aie::store_v(tmp8x8, acc.template to_vector<float>());
      float *__restrict sd = &g_scratch.attn.scores[0][kb * 8];
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_CORE; ++t) {
        aie::store_v(sd + t * KV_CHUNK_T, aie::load_v<8>(tmp8x8 + t * 8));
      }
    }

    q4_scale_f32_16(&g_scratch.attn.scores[0][0], &g_scratch.attn.scores[0][0], ATTN_SCALE, B_CORE * KV_CHUNK_T);
    q4_fill_f32_16((float *)p_flat, 0.0f, kb_max * 32);

#pragma clang loop unroll(disable)
    for (int t = 0; t < B_CORE; ++t) {
      int32_t q_local = row * B_CORE + t;
      int32_t lo = 0;
      int32_t hi = valid_len;
      if (q_local >= n_valid) {
        hi = 0;
      } else if (is_causal) {
        int32_t g_q = pos0 + q_local;
        int32_t c_hi = (g_q - chunk_gpos0) + 1;
        if (c_hi < hi) hi = c_hi;
        if (swa_window > 0) {
          int32_t c_lo = (g_q + 1 - swa_window) - chunk_gpos0;
          if (c_lo > lo) lo = c_lo;
        }
      }
      if (hi <= lo) continue;

      float *__restrict sc = &g_scratch.attn.scores[t][0];
      if ((lo > 0) || (hi < KV_CHUNK_T)) {
#pragma clang loop unroll(disable)
        for (int k = 0; k < KV_CHUNK_T; ++k) {
          if (k < lo || k >= hi) sc[k] = -1e30f;
        }
      }

      aie::vector<float, 16> s0 = aie::load_v<16>(sc);
      aie::vector<float, 16> s1 = aie::load_v<16>(sc + 16);
      float m_chk = aie::reduce_max(aie::max(s0, s1));
      float old_m = g_scratch.attn.row_m[h][t];
      float alpha = 1.0f;
      if (aie::gt(aie::broadcast<float, 16>(m_chk), aie::broadcast<float, 16>(old_m)).test(0)) {
        alpha = q4_exp2f_16(q4_fmul_16(aie::broadcast<float, 16>(old_m - m_chk), log2e_v)).get(0);
        old_m = m_chk;
        g_scratch.attn.row_m[h][t] = old_m;
        q4_scale_f32_16(&g_scratch.attn.o_acc[h][t][0], &g_scratch.attn.o_acc[h][t][0], alpha, HEAD_DIM);
      }
      aie::vector<float, 16> m_new_v = aie::broadcast<float, 16>(old_m);
      float l_sum = 0.0f;
#pragma clang loop unroll(disable)
      for (int k0 = 0; k0 < KV_CHUNK_T; k0 += 16) {
        aie::vector<float, 16> diff = aie::sub(aie::load_v<16>(sc + k0), m_new_v);
        aie::vector<float, 16> ev = q4_exp2f_16(q4_fmul_16(diff, log2e_v));
        l_sum += aie::reduce_add(ev);
        aie::accum<accfloat, 16> pa;
        pa.from_vector(ev);
        aie::vector<bfloat16, 16> pb = pa.template to_vector<bfloat16>();
        int kb0 = k0 >> 3;
        aie::store_v(p_flat + kb0 * 64 + t * 8, pb.template extract<8>(0));
        aie::store_v(p_flat + (kb0 + 1) * 64 + t * 8, pb.template extract<8>(1));
      }
      g_scratch.attn.row_l[h][t] = q4_fmul_16(aie::broadcast<float, 16>(g_scratch.attn.row_l[h][t]), aie::broadcast<float, 16>(alpha)).get(0) + l_sum;
    }

#pragma clang loop unroll(disable)
    for (int db8 = 0; db8 < (HEAD_DIM / 8); ++db8) {
      mmul888 accv(aie::zeros<accfloat, 64>());
      for (int kb = 0; kb < kb_max; ++kb) chess_prepare_for_pipelining {
        const bfloat16 *__restrict vp = v_mat + kb * 8 * HEAD_DIM + db8 * 8;
        aie::vector<bfloat16, 64> bv = aie::concat(
            aie::concat(aie::load_v<8>(vp + 0 * HEAD_DIM), aie::load_v<8>(vp + 1 * HEAD_DIM)),
            aie::concat(aie::load_v<8>(vp + 2 * HEAD_DIM), aie::load_v<8>(vp + 3 * HEAD_DIM)),
            aie::concat(aie::load_v<8>(vp + 4 * HEAD_DIM), aie::load_v<8>(vp + 5 * HEAD_DIM)),
            aie::concat(aie::load_v<8>(vp + 6 * HEAD_DIM), aie::load_v<8>(vp + 7 * HEAD_DIM)));
        aie::vector<bfloat16, 64> ap = aie::load_v<64>(p_flat + kb * 64);
        accv.mac(ap, bv);
      }
      aie::store_v(tmp8x8, accv.template to_vector<float>());
      float *__restrict dst = &g_scratch.attn.o_acc[h][0][db8 * 8];
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_CORE; ++t) {
        aie::store_v(dst + t * HEAD_DIM, aie::add(aie::load_v<8>(dst + t * HEAD_DIM), aie::load_v<8>(tmp8x8 + t * 8)));
      }
    }
  }
}

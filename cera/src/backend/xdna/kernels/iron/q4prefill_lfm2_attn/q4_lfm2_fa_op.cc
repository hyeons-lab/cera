#include "q4_lfm2_attn_state.h"

extern "C" void q4_lfm2_fa_op(
    const float *__restrict kv_pkt,
    int32_t row,
    int32_t chunk_idx,
    int32_t flags) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t is_cur_kv = flags & 1;
  const int32_t is_first = (flags >> 1) & 1;

  aie::vector<float, 16> neg_inf = aie::broadcast<float, 16>(-1e30f);
  aie::vector<float, 16> zero_f = aie::zeros<float, 16>();
  if (is_first) {
    float *__restrict oa = &g_scratch.attn.o_acc[0][0][0];
#pragma clang loop unroll(disable)
    for (int i = 0; i < GQA_Q_HEADS * B_CORE * HEAD_DIM; i += 16) {
      aie::store_v(oa + i, zero_f);
    }
    float *__restrict rm = &g_scratch.attn.row_m[0][0];
    float *__restrict rl = &g_scratch.attn.row_l[0][0];
#pragma clang loop unroll(disable)
    for (int i = 0; i < GQA_Q_HEADS * B_CORE; i += 16) {
      aie::store_v(rm + i, neg_inf);
      aie::store_v(rl + i, zero_f);
    }
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
  aie::vector<float, 16> scale_v = aie::broadcast<float, 16>(ATTN_SCALE);
  aie::vector<float, 16> log2e_v = aie::broadcast<float, 16>(LOG2E);
  aie::vector<int32_t, 16> lidx = aie::load_v<16>(lane_idx);
  bfloat16 *__restrict p_flat = g_scratch.attn.p_bf16;

  using mmul884 = aie::mmul<8, 8, 4, bfloat16, bfloat16, accfloat>;
  const int32_t kb_max = (valid_len + 7) >> 3;
  const int32_t kb4_max = kb_max * 2;

#pragma clang loop unroll(disable)
  for (int h = 0; h < GQA_Q_HEADS; ++h) {
#pragma clang loop unroll(disable)
    for (int kb4 = 0; kb4 < kb4_max; ++kb4) {
      mmul884 acc;
      for (int db = 0; db < (HEAD_DIM / 8); ++db) {
        aie::vector<bfloat16, 64> a_q = aie::load_v<64>(g_q_bf16 + (h * (HEAD_DIM / 8) + db) * 64);
        alignas(64) bfloat16 b_buf[32];
#pragma clang loop unroll(disable)
        for (int kt = 0; kt < 4; ++kt) {
          const bfloat16 *__restrict kp = k_mat + (kb4 * 4 + kt) * HEAD_DIM + db * 8;
#pragma clang loop unroll(disable)
          for (int dd = 0; dd < 8; ++dd) b_buf[dd * 4 + kt] = kp[dd];
        }
        aie::vector<bfloat16, 32> b_vec = aie::load_v<32>(b_buf);
        if (db == 0) acc.mul(a_q, b_vec);
        else acc.mac(a_q, b_vec);
      }
      alignas(64) float c_buf[32];
      aie::vector<float, 32> cv = acc.template to_vector<float>();
      aie::store_v(c_buf, q4_fmul_16(cv.template extract<16>(0), scale_v));
      aie::store_v(c_buf + 16, q4_fmul_16(cv.template extract<16>(1), scale_v));
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_CORE; ++t) {
        g_scratch.attn.scores[t][kb4 * 4 + 0] = c_buf[t * 4 + 0];
        g_scratch.attn.scores[t][kb4 * 4 + 1] = c_buf[t * 4 + 1];
        g_scratch.attn.scores[t][kb4 * 4 + 2] = c_buf[t * 4 + 2];
        g_scratch.attn.scores[t][kb4 * 4 + 3] = c_buf[t * 4 + 3];
      }
    }

    aie::vector<bfloat16, 32> zero_b = aie::zeros<bfloat16, 32>();
#pragma clang loop unroll(disable)
    for (int i = 0; i < kb_max * 64; i += 32) {
      aie::store_v(p_flat + i, zero_b);
    }

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

      aie::vector<int32_t, 16> lo_v = aie::broadcast<int32_t, 16>(lo);
      aie::vector<int32_t, 16> hi_v = aie::broadcast<int32_t, 16>(hi);
      aie::vector<float, 16> mij_v = neg_inf;
#pragma clang loop unroll(disable)
      for (int k0 = 0; k0 < hi; k0 += 16) {
        aie::vector<int32_t, 16> kidx = aie::add(lidx, aie::broadcast<int32_t, 16>(k0));
        auto valid = aie::ge(kidx, lo_v) & aie::lt(kidx, hi_v);
        aie::vector<float, 16> sv = aie::select(neg_inf, aie::load_v<16>(&g_scratch.attn.scores[t][k0]), valid);
        mij_v = aie::max(mij_v, sv);
      }
      aie::vector<float, 16> m_old_v = aie::broadcast<float, 16>(g_scratch.attn.row_m[h][t]);
      aie::vector<float, 16> m_new_v = aie::max(m_old_v, aie::broadcast<float, 16>(aie::reduce_max(mij_v)));
      aie::vector<float, 16> alpha_v = q4_exp2f_16(q4_fmul_16(aie::sub(m_old_v, m_new_v), log2e_v));
      float alpha = alpha_v.get(0);
      float l_sum = 0.0f;
      alignas(64) bfloat16 p_tmp[16];
#pragma clang loop unroll(disable)
      for (int k0 = 0; k0 < hi; k0 += 16) {
        aie::vector<int32_t, 16> kidx = aie::add(lidx, aie::broadcast<int32_t, 16>(k0));
        auto valid = aie::ge(kidx, lo_v) & aie::lt(kidx, hi_v);
        aie::vector<float, 16> diff = aie::sub(aie::load_v<16>(&g_scratch.attn.scores[t][k0]), m_new_v);
        aie::vector<float, 16> ev = aie::select(zero_f, q4_exp2f_16(q4_fmul_16(diff, log2e_v)), valid);
        l_sum += aie::reduce_add(ev);
        aie::accum<accfloat, 16> pa;
        pa.from_vector(ev);
        aie::store_v(p_tmp, pa.template to_vector<bfloat16>());
        int kb0 = k0 >> 3;
        const uint32_t *__restrict s32 = (const uint32_t *)p_tmp;
        uint32_t *__restrict d0 = (uint32_t *)(p_flat + kb0 * 64 + t * 8);
        uint32_t *__restrict d1 = (uint32_t *)(p_flat + (kb0 + 1) * 64 + t * 8);
        d0[0] = s32[0];
        d0[1] = s32[1];
        d0[2] = s32[2];
        d0[3] = s32[3];
        d1[0] = s32[4];
        d1[1] = s32[5];
        d1[2] = s32[6];
        d1[3] = s32[7];
      }
      g_scratch.attn.row_m[h][t] = m_new_v.get(0);
      float l_scaled = q4_fmul_16(aie::broadcast<float, 16>(g_scratch.attn.row_l[h][t]), alpha_v).get(0);
      g_scratch.attn.row_l[h][t] = l_scaled + l_sum;
      q4_scale_f32_16(&g_scratch.attn.o_acc[h][t][0], &g_scratch.attn.o_acc[h][t][0], alpha, HEAD_DIM);
    }

#pragma clang loop unroll(disable)
    for (int db4 = 0; db4 < (HEAD_DIM / 4); ++db4) {
      mmul884 accv;
#pragma clang loop unroll(disable)
      for (int kb = 0; kb < kb_max; ++kb) {
        alignas(64) bfloat16 b_v_buf[32];
        const uint32_t *__restrict vp32 = (const uint32_t *)(v_mat + kb * 8 * HEAD_DIM + db4 * 4);
        uint32_t *__restrict bv32 = (uint32_t *)b_v_buf;
#pragma clang loop unroll(disable)
        for (int kt = 0; kt < 8; ++kt) {
          bv32[2 * kt + 0] = vp32[kt * (HEAD_DIM / 2) + 0];
          bv32[2 * kt + 1] = vp32[kt * (HEAD_DIM / 2) + 1];
        }
        aie::vector<bfloat16, 64> a_p = aie::load_v<64>(p_flat + kb * 64);
        aie::vector<bfloat16, 32> b_v = aie::load_v<32>(b_v_buf);
        if (kb == 0) accv.mul(a_p, b_v);
        else accv.mac(a_p, b_v);
      }
      alignas(64) float ov_buf[32];
      aie::store_v(ov_buf, accv.template to_vector<float>());
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_CORE; ++t) {
        g_scratch.attn.o_acc[h][t][db4 * 4 + 0] += ov_buf[t * 4 + 0];
        g_scratch.attn.o_acc[h][t][db4 * 4 + 1] += ov_buf[t * 4 + 1];
        g_scratch.attn.o_acc[h][t][db4 * 4 + 2] += ov_buf[t * 4 + 2];
        g_scratch.attn.o_acc[h][t][db4 * 4 + 3] += ov_buf[t * 4 + 3];
      }
    }
  }
}

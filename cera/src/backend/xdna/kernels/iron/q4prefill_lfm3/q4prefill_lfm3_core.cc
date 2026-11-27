// Unified Fused B=64 LFM3 Q4_0 Prefill Core (32 cores: 4 rows x 8 cols, npu2).
// Modes via g_params[6]:
//   0 = ShortConv (d_conv=6, l_cache=7) + SwiGLU FFN (7680, padded to 8192)
//   1 = Causal / SWA GQA (16 Q heads, 4 KV heads, head_dim=128) + Single-Pass FA + in-L1 Q RoPE
//   2 = Phase A (QKV + Gate + Q/K Norm + RoPE -> SLOT_FA_Q + kvo_buf + SLOT_ATTN_GATE)
//   3 = Bidir Global Phase B (Single-Pass: Reload q0/q1 + Gate -> FA -> O -> FFN tail)
//   4 = FA Partial (Reload q0/q1 + SLOT_FA_STATE -> FA -> Drain SLOT_FA_STATE)
//   5 = FA Final (Reload q0/q1 + SLOT_FA_STATE + Gate -> FA + cur_kv -> O -> FFN tail)
//   6 = FA Partial blk==0 (Reload q0/q1 only, skip SLOT_FA_STATE preload -> FA -> Drain SLOT_FA_STATE)

#include "q4_lfm3_pre_state.h"

#define FN_SZ __attribute__((noinline, minsize))

extern "C" {
ScratchUnion g_scratch;
alignas(64) int32_t g_params[16];
float *g_x_ptr;
}

FN_SZ aie::vector<float, 16> q4_fmul_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  return q4_fast_fmul2_16(a, b);
}

FN_SZ void q4_elem_mul(
    float *__restrict dst,
    const float *__restrict a,
    const float *__restrict b,
    float s,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> bv = b ? aie::load_v<16>(b + i) : aie::broadcast<float, 16>(s);
    aie::store_v(dst + i, q4_fast_fmul2_16(aie::load_v<16>(a + i), bv));
  }
}

FN_SZ void q4_fill_f32_16(float *__restrict dst, float val, int n) {
  aie::vector<float, 16> v = aie::broadcast<float, 16>(val);
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::store_v(dst + i, v);
  }
}

namespace {

#define fmul_16 q4_fmul_16
#define inv_16 q4_inv_16

FN_SZ static void copy_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::store_v(dst + i, aie::load_v<16>(src + i));
  }
}

FN_SZ static float sum_sq_f32(const float *__restrict p, int n) {
  aie::accum<accfloat, 16> ssa = aie::zeros<accfloat, 16>();
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<bfloat16, 16> v = aie::accum<accfloat, 16>(aie::load_v<16>(p + i)).template to_vector<bfloat16>();
    ssa = aie::mac(ssa, v, v);
  }
  aie::vector<float, 16> ss = ssa.template to_vector<float>();
#pragma clang loop unroll(disable)
  for (int sh = 32; sh >= 4; sh >>= 1) {
    ss = aie::add(ss, aie::vector<float, 16>(::shift_bytes(ss, ss, sh)));
  }
  return ss.get(0);
}

static inline float read_eps() {
  float eps = 1e-5f;
  __builtin_memcpy(&eps, &g_params[7], 4);
  return eps;
}

FN_SZ static void apply_rmsnorm_n(
    const float *__restrict src,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float post_scale,
    int n,
    float *__restrict dst) {
  aie::vector<float, 16> mean = fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len));
  float r = ::invsqrt(mean.get(0) + read_eps());
  float scale = fmul_16(aie::broadcast<float, 16>(r), aie::broadcast<float, 16>(post_scale)).get(0);
  q4_elem_mul(dst, src, nullptr, scale, n);
  q4_elem_mul(dst, dst, nw, 0.0f, n);
}

static void f32_to_bf16_n(
    bfloat16 *__restrict dst,
    const float *__restrict src,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(src + i));
    aie::store_v(dst + i, a.template to_vector<bfloat16>());
  }
}

FN_SZ static float q4_rmax_16(aie::vector<float, 16> v) {
#pragma clang loop unroll(disable)
  for (int sh = 32; sh >= 4; sh >>= 1) {
    v = aie::max(v, aie::vector<float, 16>(::shift_bytes(v, v, sh)));
  }
  return v.get(0);
}

static void apply_rmsnorm_neox_rope_128(
    const float *__restrict in_h,
    const float *__restrict cs,
    bfloat16 *__restrict out_h,
    int32_t is_q) {
  float *__restrict tmp_n = g_scratch.rope.tmp_n;
  const float *__restrict norm_w = is_q ? g_scratch.rope.q_norm_w : g_scratch.rope.k_norm_w;
  float ss = sum_sq_f32(in_h, HEAD_DIM);
  apply_rmsnorm_n(in_h, norm_w, ss, 1.0f / (float)HEAD_DIM, is_q ? ATTN_SCALE_LOG2E : 1.0f, HEAD_DIM, tmp_n);
  int32_t stride_db = is_q ? (B_SUB * 8) : 8;
  const float *__restrict c = cs;
  const float *__restrict s = cs + 64;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 16) {
    int db = i >> 3;
    aie::vector<bfloat16, 16> x0 = aie::accum<accfloat, 16>(aie::load_v<16>(tmp_n + i)).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> x1 = aie::accum<accfloat, 16>(aie::load_v<16>(tmp_n + 64 + i)).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> c0 = aie::accum<accfloat, 16>(aie::load_v<16>(c + i)).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> s0 = aie::accum<accfloat, 16>(aie::load_v<16>(s + i)).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> b0 = aie::msc(aie::mul<accfloat>(x0, c0), x1, s0).template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> b1 = aie::mac(aie::mul<accfloat>(x0, s0), x1, c0).template to_vector<bfloat16>();
    aie::store_v(out_h + db * stride_db, b0.template extract<8>(0));
    aie::store_v(out_h + (db + 1) * stride_db, b0.template extract<8>(1));
    aie::store_v(out_h + (db + 8) * stride_db, b1.template extract<8>(0));
    aie::store_v(out_h + (db + 9) * stride_db, b1.template extract<8>(1));
  }
}

static void q4_rmsnorm_q8_pair_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t flags) {
  const int32_t skip_norm = flags & 1;
  const int32_t f32_out = (flags >> 1) & 1;
#pragma clang loop unroll(disable)
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict row_ptr = x_pkt + local_t * 2048;
    if (!skip_norm) {
      g_scratch.norm.sum_sq[t] = sum_sq_f32(row_ptr, 2048);
    }
    copy_f32_16(&g_scratch.norm.x_slice[t][0], row_ptr + col * 256, 256);
  }
  if (pair_idx == 3) {
    if (!skip_norm) {
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_SUB; ++t) {
        float *__restrict xs = &g_scratch.norm.x_slice[t][0];
        apply_rmsnorm_n(xs, &g_scratch.norm.norm_w[0], g_scratch.norm.sum_sq[t], 1.0f / 2048.0f, 1.0f, 256, xs);
      }
    }
    if (!f32_out) {
#pragma clang loop unroll(disable)
      for (int b = 0; b < 8; b += 2) {
        quantize_q8_2blk_8tok(
            &g_scratch.norm.x_slice[0][b * 32],
            256,
            16,
            g_scratch.norm.xq_out + b * 256,
            g_scratch.norm.xq_out + (b + 1) * 256,
            g_scratch.norm.xd_out + b * 8);
      }
    }
  }
}

static void q4_rope_step_impl(
    float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t mode) {
  if (mode == 3) {
    bfloat16 *dst = (pair_idx >= 2) ? g_scratch.attn.q1_bf16 : g_scratch.attn.q_bf16;
    if ((col >> 2) == (pair_idx & 1)) {
      copy_f32_16((float *)dst, x_pkt + (col & 3) * 1024, 1024);
    }
    return;
  }
  if (mode == 4) {
    if (g_params[9]) {
      if ((col >> 1) == pair_idx) {
        copy_f32_16(&g_scratch.attn.o_acc[0][0][0], x_pkt + (col & 1) * 2048, 2048);
      }
      if (pair_idx == 3) {
        copy_f32_16(x_pkt, &g_scratch.attn.o_acc[0][0][0], 2048);
      }
    }
    return;
  }
  const int32_t is_q = (mode == 0) || (mode == 5);
  const int32_t nh = is_q ? GQA_Q_HEADS : 1;
  const int32_t sh = is_q ? 2 : 0;
  const int32_t t_st = is_q ? 8 : HEAD_DIM;
  bfloat16 *__restrict base =
      (mode == 5) ? g_scratch.rope.q1_bf16 :
      (mode == 0) ? g_scratch.rope.q_bf16 :
      (mode == 1) ? &g_scratch.rope.k_buf[0][0] :
                    &g_scratch.rope.v_buf[0][0];

#pragma clang loop unroll(disable)
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict in_p = x_pkt + ((local_t * 512 + col * 64) << sh);
    if (mode == 2) {
      f32_to_bf16_n(base + t * HEAD_DIM, in_p, HEAD_DIM);
    } else {
      const float *__restrict cs = &g_scratch.rope.rope_cs[t][0];
#pragma clang loop unroll(disable)
      for (int h = 0; h < nh; ++h) {
        apply_rmsnorm_neox_rope_128(
            in_p + h * HEAD_DIM,
            cs,
            base + h * ((HEAD_DIM / 8) * B_SUB * 8) + t * t_st,
            is_q);
      }
    }
  }
}

static void q4_res_w_impl(const float *__restrict w_pkt, int32_t row, int32_t stage) {
  if (row != stage) return;
  const float *__restrict src = w_pkt;
  float *__restrict dst = &g_scratch.norm.x_slice[0][0];
#pragma clang loop unroll(disable)
  for (int t = 0; t < B_SUB; ++t) {
    const float *__restrict s = src + t * 288;
    float *__restrict d = dst + t * 256;
#pragma clang loop unroll(disable)
    for (int i = 0; i < 256; i += 16) {
      aie::store_v(d + i, aie::add(aie::load_v<16>(d + i), aie::load_v<16>(s + i)));
    }
  }
}


static void q4_silu_fused_drain_impl(float *__restrict y_out, int32_t arg) {
  const int32_t b = arg & 1;
  const int32_t j = (arg >> 1) & 1;
  const int32_t sub = (arg >> 2) & 1;
  float *__restrict gp =
      g_scratch.gemm.y_acc + j * (16 * B_SUB * ROWS) + sub * (8 * B_SUB * ROWS);
  if (!b) {
    q4_silu_mul_512(gp);
    bfloat16 *xd0 = g_scratch.gemm.xd_out + sub * 32 + j * 16;
    quantize_q8_2blk_8tok(gp, 16, 128, (uint8_t *)y_out, (uint8_t *)(gp + 512), xd0);
  } else {
    copy_f32_16(y_out, gp + 512, 64);
  }
}

static inline int32_t read_n_valid_clamped() {
  int32_t n_v = g_params[1];
  if (n_v < 1) n_v = 1;
  if (n_v > 64) n_v = 64;
  return n_v;
}

static void q4_conv1d_w_step_impl(
    const float *__restrict w_pkt,
    int32_t row,
    int32_t stage) {
  const int32_t r_off = row * 64;
  if (stage == 0 || stage == 15) {
    float *__restrict dst = (stage == 0) ? &g_scratch.conv.state[0][0] : &g_scratch.conv.wt[0][0];
    int nk = (stage == 0) ? D_CONV : L_CACHE;
#pragma clang loop unroll(disable)
    for (int k = 0; k < nk; ++k) {
      copy_f32_16(dst + k * 64, w_pkt + k * 256 + r_off, 64);
    }
  } else {
    if (stage >= 16) {
      if (stage == 16) {
        int32_t n_v = read_n_valid_clamped();
        int32_t si = (n_v <= 32) ? (n_v - D_CONV) : (32 - D_CONV);
        // Shift ascending so when idx < 0, D_CONV + idx = n_v + k > k (disjoint rows).
#pragma clang loop unroll(disable)
        for (int k = 0; k < D_CONV; ++k) {
          int32_t idx = si + k;
          const float *src = (idx < 0) ? &g_scratch.conv.state[D_CONV + idx][0]
                                       : &g_scratch.conv.bx[idx][0];
          copy_f32_16(&g_scratch.conv.state[k][0], src, 64);
        }
      }
      stage -= 15;
    }
    if (stage <= 8) {
      int32_t t_base = ((stage - 1) & 3) * 8;
      bool is_mul = stage >= 5;
#pragma clang loop unroll(disable)
      for (int t = 0; t < 8; ++t) {
        float *__restrict bxp = &g_scratch.conv.bx[t_base + t][0];
        const float *__restrict src = w_pkt + t * 288 + r_off;
        if (!is_mul) {
          copy_f32_16(bxp, src, 64);
        } else {
          q4_elem_mul(bxp, bxp, src, 0.0f, 64);
        }
      }
    } else {
      int32_t t_base = (stage - 9) * 8;
#pragma clang loop unroll(disable)
      for (int t = 0; t < 8; ++t) {
        int32_t ti = t_base + t;
        const float *__restrict cp = w_pkt + t * 288 + r_off;
        float *__restrict op = g_scratch.conv.conv_out + t * 64;
#pragma clang loop unroll(disable)
        for (int i = 0; i < 64; i += 16) {
          aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
#pragma clang loop unroll(disable)
          for (int k = 0; k < L_CACHE; ++k) {
            int32_t src_ti = ti + k - D_CONV;
            const float *sp = (src_ti < 0) ? &g_scratch.conv.state[D_CONV + src_ti][i]
                                           : &g_scratch.conv.bx[src_ti][i];
            acc = aie::mac(
                acc,
                aie::accum<accfloat, 16>(aie::load_v<16>(sp)).template to_vector<bfloat16>(),
                aie::accum<accfloat, 16>(aie::load_v<16>(&g_scratch.conv.wt[k][i])).template to_vector<bfloat16>());
          }
          aie::store_v(op + i, acc.template to_vector<float>());
        }
        q4_elem_mul(op, op, cp, 0.0f, 64);
      }
    }
  }
}

__attribute__((minsize)) static void q4_flash_attn_q8_prep(int32_t op_tag) {
  const int32_t sp_sub = (op_tag == 13);
  float *__restrict rlp = &g_scratch.attn.row_l[0][0];
  if (!sp_sub) {
    aie::store_v(rlp, inv_16(aie::load_v<16>(rlp)));
    aie::store_v(rlp + 16, inv_16(aie::load_v<16>(rlp + 16)));
  }
  float *__restrict tmp = (float *)g_scratch.attn.q_bf16;
  float *__restrict oa = sp_sub ? g_x_ptr : &g_scratch.attn.o_acc[0][0][0];
  uint8_t *__restrict xqo = g_scratch.norm.xq_out;
  bfloat16 *__restrict xdo = g_scratch.norm.xd_out;
  float *__restrict sc = &g_scratch.attn.gate_sig[sp_sub][0][0];
  aie::store_v(
      sc,
      fmul_16(
          aie::concat(aie::load_v<8>(rlp + sp_sub * 8), aie::load_v<8>(rlp + 16 + sp_sub * 8)),
          aie::load_v<16>(sc)));
#pragma clang loop unroll(disable)
  for (int h = 0; h < GQA_Q_HEADS; ++h, oa += 1024, xqo += 1024, xdo += 32) {
    const float *__restrict sh = sc + h * B_SUB;
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      float *__restrict dst = tmp + t * HEAD_DIM;
#pragma clang loop unroll(disable)
      for (int d8 = 0; d8 < (HEAD_DIM / 8); ++d8) {
        aie::store_v(dst + d8 * 8, aie::load_v<8>(oa + d8 * 64 + t * 8));
      }
      q4_elem_mul(dst, dst, nullptr, sh[t], HEAD_DIM);
    }
    quantize_q8_2blk_8tok(tmp, HEAD_DIM, 16, xqo, xqo + 256, xdo);
    quantize_q8_2blk_8tok(tmp + 64, HEAD_DIM, 16, xqo + 512, xqo + 768, xdo + 16);
  }
}

static uint32_t fa_mask_prep_2h(
    int32_t row_q0,
    int32_t chunk_gpos0,
    int32_t chunk_valid,
    int32_t *__restrict lo16,
    int32_t *__restrict hi16,
    int32_t &all_full) {
  const int32_t pos0 = g_params[0], n_valid = g_params[1];
  const int32_t is_causal = g_params[4], swa = g_params[5];
  uint32_t mask16 = 0;
  all_full = 1;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 16; ++i) {
    const int32_t s = i >> 3;
    const int32_t t = i & 7;
    const int32_t rem_q = (s ? (n_valid - 32) : n_valid) - row_q0;
    int32_t lo = 0, hi = (t < rem_q) ? chunk_valid : 0;
    if (is_causal && hi > 0) {
      int32_t c_hi = (pos0 + s * 32 + row_q0 + t - chunk_gpos0) + 1;
      if (c_hi < hi) hi = c_hi;
      if (swa > 0) {
        int32_t c_lo = c_hi - swa;
        if (c_lo > lo) lo = c_lo;
      }
    }
    if (hi <= lo) {
      lo = 0;
      hi = 0;
      all_full = 0;
    } else {
      mask16 |= (1U << i);
      if (lo != 0 || hi != KV_CHUNK_T) all_full = 0;
    }
    lo16[i] = lo;
    hi16[i] = hi;
  }
  return mask16;
}

}  // namespace

extern "C" {
__attribute__((minsize)) void q4_lfm3_fa_op_impl(
    const float *__restrict kv_pkt,
    int32_t row_arg,
    int32_t iter_idx) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  const int32_t pos0 = g_params[0];
  const int32_t n_valid = g_params[1];
  const int32_t kv_ch = g_params[2];
  const int32_t n_past = g_params[3];
  const int32_t lmode = g_params[6];
  const int32_t kv_start_chunk = g_params[8];
  const int32_t row_q0 = (row_arg & 3) * B_SUB;
  const int32_t chunk_iter = iter_idx >> 1;
  const int32_t is_v = iter_idx & 1;

  if ((iter_idx == 0) && ((lmode != 4 && lmode != 5) || !g_params[9])) {
    q4_fill_f32_16(&g_scratch.attn.o_acc[0][0][0], 0.0f, ATTN_ST_M_OFF);
    q4_fill_f32_16(&g_scratch.attn.row_m[0][0], -1e30f, GQA_Q_HEADS * 2 * B_SUB);
    q4_fill_f32_16(g_x_ptr, 0.0f, GQA_Q_HEADS * HEAD_DIM * B_SUB);
  }

  const bfloat16 *__restrict kv_mat = (const bfloat16 *)kv_pkt;
  if (!is_v) {
    g_scratch.attn.fa_mask16 = 0;
    if (chunk_iter >= kv_ch && g_params[10]) return;
    const int32_t cur_off = (chunk_iter - kv_ch) * 32;
    const int32_t chunk_gpos0 =
        (chunk_iter < kv_ch) ? ((kv_start_chunk + chunk_iter) * KV_CHUNK_T) : (pos0 + cur_off);
    int32_t chunk_valid = (chunk_iter < kv_ch) ? (n_past - chunk_gpos0) : (n_valid - cur_off);
    if (chunk_valid <= 0) return;
    if (chunk_valid > KV_CHUNK_T) chunk_valid = KV_CHUNK_T;

    int32_t *__restrict lo16 = (int32_t *)g_scratch.attn.p_bf16;
    int32_t *__restrict hi16 = lo16 + 16;
    int32_t all_full = 1;
    uint32_t mask16 = fa_mask_prep_2h(row_q0, chunk_gpos0, chunk_valid, lo16, hi16, all_full);
    if (!mask16) return;
    g_scratch.attn.fa_mask16 = mask16;
    const int32_t kb_max = (chunk_valid + 7) >> 3;
    g_scratch.attn.fa_kb_max = kb_max;

    float (*__restrict sc0)[16] = g_scratch.attn.scores;
    float (*__restrict sc1)[16] = (float (*)[16])(g_x_ptr + 2048);
    const bfloat16 *__restrict q0 = g_scratch.attn.q_bf16;
    const bfloat16 *__restrict q1 = g_scratch.attn.q1_bf16;
    q4_fa_qk_quad(q0, q1, q0 + 1024, q1 + 1024, kv_mat, kb_max, sc0, sc1);

    if (!all_full) {
      aie::vector<int32_t, 16> lo_v = aie::load_v<16>(lo16);
      aie::vector<int32_t, 16> hi_v = aie::load_v<16>(hi16);
      aie::vector<float, 16> ninf = aie::broadcast<float, 16>(-1e30f);
#pragma clang loop unroll(disable)
      for (int k = 0; k < KV_CHUNK_T; ++k) {
        aie::vector<int32_t, 16> kv = aie::broadcast<int32_t, 16>(k);
        auto keep = aie::ge(kv, lo_v) & aie::lt(kv, hi_v);
        aie::store_v(&sc0[k][0], aie::select(ninf, aie::load_v<16>(&sc0[k][0]), keep));
        aie::store_v(&sc1[k][0], aie::select(ninf, aie::load_v<16>(&sc1[k][0]), keep));
      }
    }
    q4_fa_softmax_2h(0, mask16, sc0, g_scratch.attn.p_bf16);
    q4_fa_softmax_2h(1, mask16, sc1, (bfloat16 *)(g_x_ptr + 2560));
  } else {
    uint32_t mask16 = g_scratch.attn.fa_mask16;
    if (!mask16) return;
    int32_t kb_max = g_scratch.attn.fa_kb_max;
#pragma clang loop unroll(disable)
    for (int d32 = 0; d32 < (HEAD_DIM / 32); ++d32) {
      q4_fa_pack_v_32(kv_mat + d32 * 32, kb_max);
      q4_fa_pv_d32(kb_max, d32);
    }
  }
}

__attribute__((minsize)) void q4_lfm3_pre_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  if (op >= 16) {
    q4_lfm3_fa_op_impl(w_pkt, op - 16, a1);
    return;
  }
  if (op == 0 || op == 4) {
    if (op == 0) {
      copy_f32_16((float *)g_params, w_pkt + 2048, 16);
    }
    copy_f32_16(&g_scratch.norm.norm_w[0], w_pkt + a0 * 256, 256);
  } else if (op == 1) {
    if (a1 == 0) {
      copy_f32_16(g_scratch.rope.q_norm_w, w_pkt + 2048, 2 * HEAD_DIM);
    }
    if ((a0 >> 1) == a1) {
      copy_f32_16(&g_scratch.rope.rope_cs[0][0], w_pkt + (a0 & 1) * 1024, 1024);
    }
  } else if (op == 2) {
#pragma clang loop unroll(disable)
    for (int s = 0; s < 2; ++s) {
      float *__restrict gl = &g_scratch.attn.gate_sig[s][0][0];
      const float *__restrict base = w_pkt + s * 512 + a0 * 2;
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_SUB; ++t) {
        gl[t] = base[t * 16];
        gl[B_SUB + t] = base[t * 16 + 1];
      }
      q4_sigmoid_f32_16(gl);
    }
  } else if (op == 5) {
    if (g_params[9] && a0 == a1) {
      copy_f32_16(&g_scratch.attn.o_acc[0][0][0], w_pkt, 2112);
    }
  } else if (op == 6) {
    q4_res_w_impl(w_pkt, a0, a1);
  } else {
    q4_conv1d_w_step_impl(w_pkt, a0, a1);
  }
}

__attribute__((minsize)) void q4_lfm3_pre_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  g_x_ptr = (float *)x_pkt;
  if (op == 0) {
    q4_rmsnorm_q8_pair_impl(x_pkt, a0, a1, a2);
  } else {
    q4_rope_step_impl((float *)x_pkt, a0, a1, a2);
  }
}

__attribute__((minsize)) void q4_lfm3_pre_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  if (op == 9) {
    if (arg < 8) {
      q4_silu_fused_drain_impl(y_out, arg);
      return;
    }
    op = 5;
  }
  if (op == 3 || op == 13) {
    if (arg == 0) q4_flash_attn_q8_prep(op);
    op = 0;
  }
  const float *__restrict src;
  int32_t valid_f32 = 64;
  if (op == 5) {
    src = (const float *)g_scratch.gemm.xd_out;
    valid_f32 = 32;
  } else if (op == 0) {
    src = (const float *)(g_scratch.norm.xq_out + arg * 256);
    if (arg >= 8) valid_f32 = 32;
  } else if (op == 8) {
    src = g_scratch.conv.conv_out + arg * 64;
    if (arg >= 8) {
      int32_t n_v = read_n_valid_clamped();
      int32_t k = arg - 8;
      if (n_v > 32) {
        int32_t idx = (n_v - 32) - D_CONV + k;
        src = (idx < 0) ? &g_scratch.conv.state[D_CONV + idx][0] : &g_scratch.conv.bx[idx][0];
      } else {
        src = &g_scratch.conv.state[k][0];
      }
    }
  } else if (op == 1) {
    int32_t p4 = arg >> 4;
    const float *__restrict base = g_scratch.gemm.y_acc +
        (p4 >> 1) * 2048 + (arg & 8) * 128 + (p4 & 1) * 512 + (arg & 7) * 16;
#pragma clang loop unroll(disable)
    for (int tl = 0; tl < 4; ++tl) {
      aie::store_v(y_out + tl * 16, aie::load_v<16>(base + tl * 128));
    }
    return;
  } else if (op == 11) {
    src = g_scratch.gemm.y_acc + (arg >> 1) * 1024 + (arg & 1) * 64;
  } else {
    const int32_t goff = ((op == 2) + ((op == 7) * 3)) << 10;
    const float *__restrict base = (op == 6) ? g_x_ptr : ((const float *)&g_scratch + goff);
    src = base + arg * 64;
  }
  copy_f32_16(y_out, src, valid_f32);
  if (valid_f32 < 64) q4_fill_f32_16(y_out + 32, 0.0f, 32);
}

}  // extern "C"

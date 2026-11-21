// Unified Fused B=64 LFM3 Q4_0 Prefill Core (32 cores: 4 rows x 8 cols, npu2).
// Modes via g_params[6]:
//   0 = ShortConv (d_conv=6, l_cache=7) + SwiGLU FFN (8192)
//   1 = Causal / SWA GQA (16 Q heads, 4 KV heads, head_dim=128) + QK-norm + NeoX RoPE + Gate + SwiGLU FFN
//   2 = Phase A (QKV + Gate + Q RoPE -> SLOT_FA_Q + kvo_buf + SLOT_ATTN_GATE)
//   3 = Bidir Global Phase B (Reload SLOT_FA_Q + Gate -> FA -> O -> FFN tail)
//   4 = FA Partial (Reload SLOT_FA_Q + SLOT_FA_STATE -> FA -> Drain SLOT_FA_STATE)
//   5 = FA Final (Reload SLOT_FA_Q + SLOT_FA_STATE + Gate -> FA + cur_kv -> O -> FFN tail)

#include "q4_lfm3_pre_state.h"

#define FN_SZ __attribute__((noinline, minsize))

extern "C" {
ScratchUnion g_scratch;
alignas(64) int32_t g_params[16];
}

FN_SZ aie::vector<float, 16> q4_fmul_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  aie::vector<bfloat16, 16> ah = aie::accum<accfloat, 16>(a).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> bh = aie::accum<accfloat, 16>(b).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> al = aie::accum<accfloat, 16>(
      aie::sub(a, aie::accum<accfloat, 16>(ah).template to_vector<float>())).template to_vector<bfloat16>();
  aie::vector<bfloat16, 16> bl = aie::accum<accfloat, 16>(
      aie::sub(b, aie::accum<accfloat, 16>(bh).template to_vector<float>())).template to_vector<bfloat16>();
  aie::accum<accfloat, 16> acc = aie::mul<accfloat>(ah, bh);
  acc = aie::mac(acc, ah, bl);
  acc = aie::mac(acc, al, bh);
  return acc.template to_vector<float>();
}

FN_SZ aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v) {
  return aie::inv(aie::max(v, aie::broadcast<float, 16>(1e-35f)));
}

static inline aie::vector<float, 16> q4_invsqrt_16(aie::vector<float, 16> v) {
  return aie::invsqrt(v);
}

FN_SZ aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x) {
  x = aie::max(x, aie::broadcast<float, 16>(-126.0f));
  x = aie::min(x, aie::broadcast<float, 16>(126.0f));
  aie::vector<float, 16> magic = aie::broadcast<float, 16>(12582912.0f);
  aie::vector<float, 16> kf = aie::sub(aie::add(x, magic), magic);
  aie::vector<bfloat16, 16> fb =
      aie::accum<accfloat, 16>(aie::sub(x, kf)).template to_vector<bfloat16>();
  aie::vector<float, 16> u = aie::add(kf, aie::broadcast<float, 16>(98431.0f));
  aie::vector<bfloat16, 16> eab = aie::filter_even(u.template cast_to<bfloat16>(), 1);
  aie::vector<bfloat16, 16> pb = aie::broadcast<bfloat16, 16>((bfloat16)0.0555041087f);
  {
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::broadcast<float, 16>(0.2402265069f));
    pb = aie::mac(acc, pb, fb).template to_vector<bfloat16>();
    acc.from_vector(aie::broadcast<float, 16>(0.6931471805f));
    pb = aie::mac(acc, pb, fb).template to_vector<bfloat16>();
    acc.from_vector(aie::broadcast<float, 16>(1.0f));
    pb = aie::mac(acc, pb, fb).template to_vector<bfloat16>();
  }
  return aie::mul<accfloat>(pb, eab).template to_vector<float>();
}

FN_SZ static void q4_elem_mul(
    float *__restrict dst,
    const float *__restrict a,
    const float *__restrict b,
    float s,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> bv = b ? aie::load_v<16>(b + i) : aie::broadcast<float, 16>(s);
    aie::store_v(dst + i, q4_fmul_16(aie::load_v<16>(a + i), bv));
  }
}

FN_SZ void q4_scale_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    float s,
    int n) {
  q4_elem_mul(dst, src, nullptr, s, n);
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
#define exp2f_16 q4_exp2f_16

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
    aie::vector<float, 16> v = aie::load_v<16>(p + i);
    ssa = aie::mac(ssa, v, v);
  }
  return aie::reduce_add(ssa.template to_vector<float>());
}

FN_SZ static void apply_rmsnorm_n(
    const float *__restrict src,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float eps,
    float post_scale,
    int n,
    float *__restrict dst) {
  aie::vector<float, 16> mean = fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len));
  aie::vector<float, 16> inv_r = q4_invsqrt_16(aie::add(mean, aie::broadcast<float, 16>(eps)));
  float scale = fmul_16(inv_r, aie::broadcast<float, 16>(post_scale)).get(0);
  q4_elem_mul(dst, src, nullptr, scale, n);
  q4_elem_mul(dst, dst, nw, 0.0f, n);
}

FN_SZ static aie::vector<float, 16> q4_sigmoid_16(aie::vector<float, 16> x) {
  aie::vector<float, 16> one_v = aie::broadcast<float, 16>(1.0f);
  aie::vector<float, 16> ml2e = aie::broadcast<float, 16>(-LOG2E);
  return inv_16(aie::add(one_v, exp2f_16(fmul_16(x, ml2e))));
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

FN_SZ static void quantize_q8_block_8tok(
    const float *__restrict src,
    int32_t row_stride,
    int32_t col2_off,
    uint8_t *__restrict xq_out,
    bfloat16 *__restrict xd_out) {
#pragma clang loop unroll(disable)
  for (int t = 0; t < B_SUB; ++t) {
    const float *r = src + t * row_stride;
    aie::vector<float, 16> x0 = aie::load_v<16>(r);
    aie::vector<float, 16> x1 = aie::load_v<16>(r + col2_off);
    aie::vector<float, 16> z = aie::zeros<float, 16>();
    float amax = q4_rmax_16(aie::max(aie::max(x0, aie::sub(z, x0)), aie::max(x1, aie::sub(z, x1))));
    aie::vector<float, 16> dv =
        fmul_16(aie::broadcast<float, 16>(amax), aie::broadcast<float, 16>(1.0f / 127.0f));
    aie::accum<accfloat, 16> da;
    da.from_vector(dv);
    xd_out[t] = da.template to_vector<bfloat16>().get(0);
    float id = inv_16(dv).get(0);
    aie::vector<float, 16> s0 = fmul_16(aie::load_v<16>(r), aie::broadcast<float, 16>(id));
    aie::vector<float, 16> s1 = fmul_16(aie::load_v<16>(r + col2_off), aie::broadcast<float, 16>(id));
    aie::vector<int8, 32> q32 = aie::to_fixed<int8>(aie::concat(s0, s1), 0);
    int grp = t >> 2;
    int rt = t & 3;
    aie::store_v((int8_t *)(xq_out + grp * 128 + rt * 16), q32.template extract<16>(0));
    aie::store_v((int8_t *)(xq_out + grp * 128 + 64 + rt * 16), q32.template extract<16>(1));
  }
}

__attribute__((minsize)) static void apply_rmsnorm_neox_rope_128(
    const float *__restrict in_h,
    const float *__restrict norm_w,
    float eps,
    float post_scale,
    const float *__restrict cs,
    bfloat16 *__restrict out_h,
    int32_t stride_db) {
  float *__restrict tmp_n = g_scratch.rope.tmp_n;
  float ss = sum_sq_f32(in_h, HEAD_DIM);
  apply_rmsnorm_n(in_h, norm_w, ss, 1.0f / (float)HEAD_DIM, eps, post_scale, HEAD_DIM, tmp_n);
  const float *__restrict c = cs;
  const float *__restrict s = cs + 64;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 16) {
    int db = i >> 3;
    aie::accum<accfloat, 16> a0;
    a0.from_vector(aie::sub(
        fmul_16(aie::load_v<16>(tmp_n + i), aie::load_v<16>(c + i)),
        fmul_16(aie::load_v<16>(tmp_n + 64 + i), aie::load_v<16>(s + i))));
    aie::vector<bfloat16, 16> b0 = a0.template to_vector<bfloat16>();
    aie::store_v(out_h + db * stride_db, b0.template extract<8>(0));
    aie::store_v(out_h + (db + 1) * stride_db, b0.template extract<8>(1));
    aie::accum<accfloat, 16> a1;
    a1.from_vector(aie::add(
        fmul_16(aie::load_v<16>(tmp_n + i), aie::load_v<16>(s + i)),
        fmul_16(aie::load_v<16>(tmp_n + 64 + i), aie::load_v<16>(c + i))));
    aie::vector<bfloat16, 16> b1 = a1.template to_vector<bfloat16>();
    aie::store_v(out_h + (db + 8) * stride_db, b1.template extract<8>(0));
    aie::store_v(out_h + (db + 9) * stride_db, b1.template extract<8>(1));
  }
}

static inline float read_eps() {
  float eps;
  __builtin_memcpy(&eps, &g_params[7], 4);
  return eps;
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
    float eps = read_eps();
    if (!skip_norm) {
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_SUB; ++t) {
        float *__restrict xs = &g_scratch.norm.x_slice[t][0];
        apply_rmsnorm_n(xs, &g_scratch.norm.norm_w[0], g_scratch.norm.sum_sq[t], 1.0f / 2048.0f, eps, 1.0f, 256, xs);
      }
    }
    if (!f32_out) {
#pragma clang loop unroll(disable)
      for (int b = 0; b < 8; ++b) {
        quantize_q8_block_8tok(
            &g_scratch.norm.x_slice[0][b * 32],
            256,
            16,
            g_scratch.norm.xq_out + b * 256,
            g_scratch.norm.xd_out + b * 8);
      }
    }
  }
}

__attribute__((minsize)) static void q4_rope_step_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t mode) {
  if (mode == 3) {
    if ((col >> 2) == pair_idx) {
      copy_f32_16((float *)g_scratch.attn.q_bf16, x_pkt + (col & 3) * 1024, 1024);
    }
    return;
  }
  if (mode == 4) {
    if (g_params[9] && ((col >> 1) == pair_idx)) {
      copy_f32_16(&g_scratch.attn.o_acc[0][0][0], x_pkt + (col & 1) * 2112, 2080);
    }
    return;
  }
  float eps = read_eps();
  const int32_t is_q = (mode == 0);
  const int32_t nh = is_q ? GQA_Q_HEADS : 1;
  const int32_t pkt_st = is_q ? 2048 : 512;
  const int32_t col_st = is_q ? 256 : 64;
  const int32_t db_st = is_q ? (B_SUB * 8) : 8;
  const int32_t t_st = is_q ? 8 : HEAD_DIM;
  const float *__restrict nw = is_q ? g_scratch.rope.q_norm_w : g_scratch.rope.k_norm_w;
  bfloat16 *__restrict base =
      is_q ? g_scratch.rope.q_bf16 : ((mode == 1) ? &g_scratch.rope.k_buf[0][0] : &g_scratch.rope.v_buf[0][0]);

#pragma clang loop unroll(disable)
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict in_p = x_pkt + local_t * pkt_st + col * col_st;
    if (mode == 2) {
      f32_to_bf16_n(base + t * HEAD_DIM, in_p, HEAD_DIM);
    } else {
      const float *__restrict cs = &g_scratch.rope.rope_cs[t][0];
#pragma clang loop unroll(disable)
      for (int h = 0; h < nh; ++h) {
        apply_rmsnorm_neox_rope_128(
            in_p + h * HEAD_DIM,
            nw,
            eps,
            is_q ? ATTN_SCALE_LOG2E : 1.0f,
            cs,
            base + h * ((HEAD_DIM / 8) * B_SUB * 8) + t * t_st,
            db_st);
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
  const int32_t sub = (arg >> 1) & 1;
  const int32_t j = (arg >> 2) & 1;
  float *__restrict gp = g_scratch.gemm.y_acc + sub * (8 * B_SUB * ROWS);
  if (!b) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < 512; i += 16) {
      aie::store_v(gp + i, fmul_16(fmul_16(aie::load_v<16>(gp + i), aie::load_v<16>(gp + 512 + i)), q4_sigmoid_16(aie::load_v<16>(gp + i))));
    }
  }
  quantize_q8_block_8tok(
      gp + b * 256,
      16,
      128,
      (uint8_t *)y_out,
      g_scratch.gemm.xd_out + sub * 32 + (j * 2 + b) * 8);
}

static inline int32_t read_n_valid_clamped() {
  int32_t n_v = g_params[1];
  if (n_v < 1) n_v = 1;
  if (n_v > 64) n_v = 64;
  return n_v;
}

__attribute__((minsize)) static void q4_conv1d_w_step_impl(
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
        copy_f32_16(&g_scratch.conv.state[0][0], &g_scratch.conv.bx[si][0], D_CONV * 64);
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
            acc = aie::mac(acc, aie::load_v<16>(&g_scratch.conv.bx[ti + k - D_CONV][i]), aie::load_v<16>(&g_scratch.conv.wt[k][i]));
          }
          aie::store_v(op + i, acc.template to_vector<float>());
        }
        q4_elem_mul(op, op, cp, 0.0f, 64);
      }
    }
  }
}

__attribute__((minsize)) static void q4_flash_attn_q8_prep() {
  float *__restrict rlp = &g_scratch.attn.row_l[0][0];
  float *__restrict gs = &g_scratch.attn.gate_sig[0][0];
  float *__restrict oa = &g_scratch.attn.o_acc[0][0][0];
  aie::vector<float, 16> rl = aie::max(aie::load_v<16>(rlp), aie::broadcast<float, 16>(1e-35f));
  aie::store_v(rlp, fmul_16(aie::load_v<16>(gs), inv_16(rl)));
#pragma clang loop unroll(disable)
  for (int ht = 0; ht < GQA_Q_HEADS * B_SUB; ++ht) {
    q4_scale_f32_16(oa + ht * HEAD_DIM, oa + ht * HEAD_DIM, rlp[ht], HEAD_DIM);
  }
#pragma clang loop unroll(disable)
  for (int b = 0; b < 8; ++b) {
    quantize_q8_block_8tok(
        oa + (b >> 2) * (B_SUB * HEAD_DIM) + (b & 3) * 32,
        HEAD_DIM,
        16,
        g_scratch.norm.xq_out + b * 256,
        g_scratch.norm.xd_out + b * 8);
  }
}

FN_SZ static void fa_mask_2row32(int32_t t, int32_t lo, int32_t hi) {
  float *__restrict s0 = &g_scratch.attn.scores[0][t][0];
  float *__restrict s1 = &g_scratch.attn.scores[1][t][0];
#pragma clang loop unroll(disable)
  for (int k = 0; k < KV_CHUNK_T; ++k) {
    if (k < lo || k >= hi) {
      s0[k] = -1e30f;
      s1[k] = -1e30f;
    }
  }
}

static inline void fa_rescale_one(int32_t h, int32_t t, float old_m, float new_m) {
  float *__restrict rl = &g_scratch.attn.row_l[h][t];
  float rl_prev = *rl;
  uint16_t rl_hi = 0;
  __builtin_memcpy(&rl_hi, reinterpret_cast<const uint8_t *>(rl) + 2, 2);
  if (rl_hi != 0) {
    float alpha = exp2f_16(aie::broadcast<float, 16>(old_m - new_m)).get(0);
    *rl = fmul_16(aie::broadcast<float, 16>(rl_prev), aie::broadcast<float, 16>(alpha)).get(0);
    q4_scale_f32_16(&g_scratch.attn.o_acc[h][t][0], &g_scratch.attn.o_acc[h][t][0], alpha, HEAD_DIM);
  }
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
  const int32_t is_causal = g_params[4];
  const int32_t swa_window = g_params[5];
  const int32_t kv_start_chunk = g_params[8];
  const int32_t sub = row_arg >> 2;
  const int32_t chunk_iter = iter_idx >> 1;
  const int32_t is_v = iter_idx & 1;

  const int32_t is_first = (iter_idx == 0) && ((g_params[6] <= 3) || !g_params[9]);
  if (is_first) {
    q4_fill_f32_16(&g_scratch.attn.o_acc[0][0][0], 0.0f, GQA_Q_HEADS * B_SUB * (HEAD_DIM + 1));
    q4_fill_f32_16(&g_scratch.attn.row_m[0][0], -1e30f, GQA_Q_HEADS * B_SUB);
  }

  const int32_t n_valid_sub = sub ? (n_valid - 32) : n_valid;
  const int32_t row = row_arg & 3;
  const int32_t row_q0 = row * B_SUB;
  const int32_t rem_q = n_valid_sub - row_q0;
  if (rem_q <= 0) return;

  const int32_t pos0_sub = pos0 + sub * 32;
  const int32_t cur_off = (chunk_iter - kv_ch) * 32;
  const int32_t chunk_gpos0 =
      (chunk_iter < kv_ch) ? ((kv_start_chunk + chunk_iter) * KV_CHUNK_T) : (pos0 + cur_off);
  int32_t valid_len = (chunk_iter < kv_ch) ? (n_past - chunk_gpos0) : (n_valid - cur_off);
  if (valid_len <= 0) return;
  if (valid_len > KV_CHUNK_T) valid_len = KV_CHUNK_T;

  if (is_causal) {
    const int32_t q_last = pos0_sub + row_q0 + ((rem_q < B_SUB) ? rem_q : B_SUB) - 1;
    if (chunk_gpos0 > q_last) return;
    if (swa_window > 0 && (chunk_gpos0 + valid_len) <= (pos0_sub + row_q0 + 1 - swa_window)) return;
    const int32_t max_k = (q_last - chunk_gpos0) + 1;
    if (max_k < valid_len) valid_len = max_k;
  }

  const bfloat16 *__restrict kv_mat = (const bfloat16 *)kv_pkt;
  const int32_t kb_max = (valid_len + 7) >> 3;

  if (!is_v) {
    q4_fa_qk_dual(
        g_scratch.attn.q_bf16,
        g_scratch.attn.q_bf16 + (HEAD_DIM / 8) * 64,
        kv_mat,
        kb_max);

    uint32_t active_mask = 0;
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      int32_t q_local = row_q0 + t;
      int32_t lo = 0;
      int32_t hi = valid_len;
      if (q_local >= n_valid_sub) {
        hi = 0;
      } else if (is_causal) {
        int32_t g_q = pos0_sub + q_local;
        int32_t c_hi = (g_q - chunk_gpos0) + 1;
        if (c_hi < hi) hi = c_hi;
        if (swa_window > 0) {
          int32_t c_lo = (g_q + 1 - swa_window) - chunk_gpos0;
          if (c_lo > lo) lo = c_lo;
        }
      }
      if (hi <= lo) continue;
      active_mask |= (1U << t);
      if ((lo > 0) || (hi < KV_CHUNK_T)) fa_mask_2row32(t, lo, hi);
    }

    alignas(64) bfloat16 p0_tmp[256];
#pragma clang loop unroll(disable)
    for (int h = 0; h < GQA_Q_HEADS; ++h) {
      bfloat16 *__restrict p_dst = h ? &g_scratch.attn.p_bf16[256] : p0_tmp;
      q4_fill_f32_16((float *)p_dst, 0.0f, 128);
      if (active_mask) {
        float *__restrict rm = &g_scratch.attn.row_m[h][0];
        float *__restrict rl = &g_scratch.attn.row_l[h][0];
#pragma clang loop unroll(disable)
        for (int t = 0; t < B_SUB; ++t) {
          if (!((active_mask >> t) & 1U)) continue;
          const float *__restrict sc = &g_scratch.attn.scores[h][t][0];
          float m_chk = q4_rmax_16(aie::max(aie::load_v<16>(sc), aie::load_v<16>(sc + 16)));
          float old_m = rm[t];
          if (aie::gt(aie::broadcast<float, 16>(m_chk), aie::broadcast<float, 16>(old_m)).test(0)) {
            rm[t] = m_chk;
            fa_rescale_one(h, t, old_m, m_chk);
          }
          rl[t] += q4_fa_exp2_row32(sc, rm[t], p_dst + t * 8);
        }
      }
      if (h == 0) copy_f32_16((float *)&g_scratch.attn.p_bf16[0], (const float *)p0_tmp, 128);
    }
  } else {
    q4_fa_pv_d16(kv_mat, kb_max);
  }
}

__attribute__((minsize)) void q4_lfm3_pre_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
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
    float *__restrict gl = &g_scratch.attn.gate_sig[0][0];
    const float *__restrict base = w_pkt + a1 * 512 + a0 * 2;
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      gl[t] = base[t * 16];
      gl[B_SUB + t] = base[t * 16 + 1];
    }
    aie::store_v(gl, q4_sigmoid_16(aie::load_v<16>(gl)));
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
  if (op == 0) {
    q4_rmsnorm_q8_pair_impl(x_pkt, a0, a1, a2);
  } else {
    q4_rope_step_impl(x_pkt, a0, a1, a2);
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
  if (op == 3) {
    if (arg == 0) q4_flash_attn_q8_prep();
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
    src = g_scratch.conv.conv_out + ((arg < 8) ? arg * 64 : 0);
    if (arg >= 8) {
      int32_t n_v = read_n_valid_clamped();
      const float *st = (n_v > 32) ? &g_scratch.conv.bx[n_v - 32 - D_CONV][0] : &g_scratch.conv.state[0][0];
      src = st + (arg - 8) * 64;
    }
  } else if (op == 1) {
    int32_t p4 = arg >> 4;
    int32_t tok = arg & 15;
    const float *__restrict base = g_scratch.gemm.y_acc +
        (tok >> 3) * 1024 + p4 * 512 + ((tok >> 2) & 1) * 64 + (tok & 3) * 16;
#pragma clang loop unroll(disable)
    for (int tl = 0; tl < 4; ++tl) {
      aie::store_v(y_out + tl * 16, aie::load_v<16>(base + tl * 128));
    }
    return;
  } else if (op == 11) {
    src = g_scratch.gemm.y_acc + (arg >> 1) * 1024 + (arg & 1) * 64;
  } else {
    const float *__restrict base =
        (op == 2) ? (const float *)g_scratch.rope.k_buf :
        (op == 7) ? (const float *)g_scratch.attn.q_bf16 :
        (op == 10) ? &g_scratch.attn.o_acc[0][0][0] :
                     &g_scratch.norm.x_slice[0][0];
    src = base + arg * 64;
    if (op == 10 && arg == 32) valid_f32 = 32;
  }
  copy_f32_16(y_out, src, valid_f32);
  if (valid_f32 < 64) q4_fill_f32_16(y_out + valid_f32, 0.0f, 64 - valid_f32);
}

}  // extern "C"

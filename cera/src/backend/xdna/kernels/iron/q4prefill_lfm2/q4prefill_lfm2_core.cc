// Unified Fused B=64 LFM2 / LFM2.5 Q4_0 Prefill Core (32 cores: 4 rows x 8 cols, npu2).
// Mode 0 (g_params[6]==0): ShortConv (d_conv=2) + SwiGLU FFN (8192).
// Mode 1 (g_params[6]==1): GQA (32 Q heads, 8 KV heads, head_dim=64) + QK-norm + NeoX RoPE + SwiGLU FFN.

#include "q4_lfm2_pre_state.h"

#define FN_SZ __attribute__((noinline, minsize))

extern "C" {
ScratchUnion g_scratch;
alignas(64) int32_t g_params[16];
}

FN_SZ aie::vector<float, 16> q4_fmul_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  return aie::mul(a, b).template to_vector<float>();
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
  aie::vector<float, 16> f = aie::sub(x, kf);
  aie::vector<float, 16> u = aie::add(kf, aie::broadcast<float, 16>(98431.0f));
  aie::accum<accfloat, 16> ea;
  ea.from_vector(aie::filter_even(u.template cast_to<bfloat16>(), 1));
  static const float coeffs[5] = {
      0.0096181291f, 0.0555041087f, 0.2402265069f, 0.6931471805f, 1.0f};
  aie::vector<float, 16> p = aie::broadcast<float, 16>(0.0013333558f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 5; ++i)
    p = aie::add(q4_fast_fmul2_16(p, f), aie::broadcast<float, 16>(coeffs[i]));
  return q4_fast_fmul2_16(p, ea.template to_vector<float>());
}

FN_SZ void q4_scale_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    float s,
    int n) {
  aie::vector<float, 16> sv = aie::broadcast<float, 16>(s);
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::store_v(dst + i, q4_fmul_16(aie::load_v<16>(src + i), sv));
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

FN_SZ static void q4_mul_f32_16(
    float *__restrict dst,
    const float *__restrict a,
    const float *__restrict b,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::store_v(dst + i, q4_fmul_16(aie::load_v<16>(a + i), aie::load_v<16>(b + i)));
  }
}

static void apply_rmsnorm_n(
    const float *__restrict src,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float eps,
    int n,
    float *__restrict dst) {
  aie::vector<float, 16> mean = fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len));
  float inv_r = q4_invsqrt_16(aie::add(mean, aie::broadcast<float, 16>(eps))).get(0);
  q4_scale_f32_16(dst, src, inv_r, n);
  q4_mul_f32_16(dst, dst, nw, n);
}

static inline aie::vector<float, 16> q4_sigmoid_16(aie::vector<float, 16> x) {
  aie::vector<float, 16> one_v = aie::broadcast<float, 16>(1.0f);
  aie::vector<float, 16> ml2e = aie::broadcast<float, 16>(-LOG2E);
  return inv_16(aie::add(one_v, exp2f_16(fmul_16(x, ml2e))));
}

FN_SZ static void f32_to_bf16_n(
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

static __attribute__((noinline)) void quantize_q8_block_8tok(
    const float *__restrict src,
    int32_t row_stride,
    int32_t col2_off,
    uint8_t *__restrict xq_out,
    bfloat16 *__restrict xd_out) {
#pragma clang loop unroll(disable)
  for (int t = 0; t < B_SUB; ++t) {
    const float *r = src + t * row_stride;
    float amax = aie::reduce_max(
        aie::max(aie::abs(aie::load_v<16>(r)), aie::abs(aie::load_v<16>(r + col2_off))));
    aie::vector<float, 16> dv =
        fmul_16(aie::broadcast<float, 16>(amax), aie::broadcast<float, 16>(1.0f / 127.0f));
    aie::accum<accfloat, 16> da;
    da.from_vector(dv);
    xd_out[t] = da.template to_vector<bfloat16>().get(0);
    aie::vector<float, 16> idv = inv_16(dv);
    aie::vector<float, 16> s0 = fmul_16(aie::load_v<16>(r), idv);
    aie::vector<float, 16> s1 = fmul_16(aie::load_v<16>(r + col2_off), idv);
    aie::vector<int8, 32> q32 = aie::to_fixed<int8>(aie::concat(s0, s1), 0);
    int grp = t >> 2;
    int rt = t & 3;
    aie::store_v((int8_t *)(xq_out + grp * 128 + rt * 16), q32.template extract<16>(0));
    aie::store_v((int8_t *)(xq_out + grp * 128 + 64 + rt * 16), q32.template extract<16>(1));
  }
}

FN_SZ static void apply_rmsnorm_neox_rope_64(
    const float *__restrict in_h,
    const float *__restrict norm_w,
    float eps,
    const float *__restrict cs,
    bfloat16 *__restrict out_h,
    int32_t stride_db) {
  float *__restrict tmp_n = (float *)g_scratch.rope._pad_after_rope;
  float ss = sum_sq_f32(in_h, HEAD_DIM);
  apply_rmsnorm_n(in_h, norm_w, ss, 1.0f / (float)HEAD_DIM, eps, HEAD_DIM, tmp_n);
  const float *__restrict c = cs;
  const float *__restrict s = cs + 32;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 32; i += 16) {
    aie::vector<float, 16> x0 = aie::load_v<16>(tmp_n + i);
    aie::vector<float, 16> x1 = aie::load_v<16>(tmp_n + 32 + i);
    aie::vector<float, 16> cv = aie::load_v<16>(c + i);
    aie::vector<float, 16> sv = aie::load_v<16>(s + i);
    aie::accum<accfloat, 16> a0, a1;
    a0.from_vector(aie::sub(fmul_16(x0, cv), fmul_16(x1, sv)));
    a1.from_vector(aie::add(fmul_16(x0, sv), fmul_16(x1, cv)));
    aie::vector<bfloat16, 16> b0 = a0.template to_vector<bfloat16>();
    aie::vector<bfloat16, 16> b1 = a1.template to_vector<bfloat16>();
    int db = i >> 3;
    aie::store_v(out_h + db * stride_db, b0.template extract<8>(0));
    aie::store_v(out_h + (db + 1) * stride_db, b0.template extract<8>(1));
    aie::store_v(out_h + (db + 4) * stride_db, b1.template extract<8>(0));
    aie::store_v(out_h + (db + 5) * stride_db, b1.template extract<8>(1));
  }
}

FN_SZ static float read_eps() {
  float eps = 1e-5f;
  __builtin_memcpy(&eps, &g_params[7], 4);
  return eps;
}

FN_SZ static void q4_rmsnorm_q8_pair_impl(
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
        apply_rmsnorm_n(xs, &g_scratch.norm.norm_w[0], g_scratch.norm.sum_sq[t], 1.0f / 2048.0f, eps, 256, xs);
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

FN_SZ static void q4_rope_step_impl(
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
      copy_f32_16(&g_scratch.attn.o_acc[0][0][0], x_pkt + (col & 1) * 2112, 2112);
    }
    return;
  }
  float eps = read_eps();
  const int32_t is_q = (mode == 0);
  const int32_t nh = is_q ? GQA_Q_HEADS : 1;
  const int32_t pkt_st = is_q ? 2048 : 512;
  const int32_t col_st = is_q ? 256 : HEAD_DIM;
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
        apply_rmsnorm_neox_rope_64(in_p + h * HEAD_DIM, nw, eps, cs, base + h * 512 + t * t_st, db_st);
      }
    }
  }
}

FN_SZ static void q4_res_w_impl(const float *__restrict w_pkt, int32_t row, int32_t stage) {
  if (row != (stage & 3)) return;
  const float *__restrict src = w_pkt;
  float *__restrict dst = &g_scratch.norm.x_slice[0][0];
  int add = stage >> 2;
#pragma clang loop unroll(disable)
  for (int t = 0; t < B_SUB; ++t) {
    const float *__restrict s = src + t * 288;
    float *__restrict d = dst + t * 256;
#pragma clang loop unroll(disable)
    for (int i = 0; i < 256; i += 16) {
      aie::vector<float, 16> sv = aie::load_v<16>(s + i);
      aie::store_v(d + i, add ? aie::add(aie::load_v<16>(d + i), sv) : sv);
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
      aie::vector<float, 16> sv = q4_sigmoid_16(aie::load_v<16>(gp + i));
      aie::store_v(gp + i, fmul_16(fmul_16(aie::load_v<16>(gp + i), sv), aie::load_v<16>(gp + 512 + i)));
    }
  }
  quantize_q8_block_8tok(
      gp + b * 256,
      16,
      128,
      (uint8_t *)y_out,
      g_scratch.gemm.xd_out + sub * 32 + (j * 2 + b) * 8);
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
    if (stage == 16) {
      int32_t n_v = g_params[1];
      if (n_v < 1) n_v = 1;
      int32_t si = (n_v <= 32) ? (n_v - 2) : 30;
      copy_f32_16(&g_scratch.conv.state[0][0], &g_scratch.conv.bx[si][0], D_CONV * 64);
      stage = 1;
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
          q4_mul_f32_16(bxp, bxp, src, 64);
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
        q4_mul_f32_16(op, op, cp, 64);
      }
    }
  }
}

static void q4_flash_attn_q8_prep() {
  float *__restrict rlp = &g_scratch.attn.row_l[0][0];
  float *__restrict oa = &g_scratch.attn.o_acc[0][0][0];
#pragma clang loop unroll(disable)
  for (int i = 0; i < GQA_Q_HEADS * B_SUB; i += 16) {
    aie::store_v(rlp + i, inv_16(aie::load_v<16>(rlp + i)));
  }
#pragma clang loop unroll(disable)
  for (int ht = 0; ht < GQA_Q_HEADS * B_SUB; ++ht) {
    q4_scale_f32_16(oa + ht * HEAD_DIM, oa + ht * HEAD_DIM, rlp[ht], HEAD_DIM);
  }
#pragma clang loop unroll(disable)
  for (int b = 0; b < 8; ++b) {
    quantize_q8_block_8tok(
        oa + (b >> 1) * (B_SUB * HEAD_DIM) + (b & 1) * 32,
        HEAD_DIM,
        16,
        g_scratch.norm.xq_out + b * 256,
        g_scratch.norm.xd_out + b * 8);
  }
}

}  // namespace

extern "C" {
void q4_lfm2_fa_op(const float *__restrict kv_pkt, int32_t row_arg, int32_t iter_idx);

__attribute__((minsize)) void q4_lfm2_pre_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  if (op >= 16) {
    q4_lfm2_fa_op(w_pkt, op - 16, a1);
  } else if (op == 0 || op == 4) {
    if (op == 0) {
      copy_f32_16((float *)g_params, w_pkt + 2048, 16);
    }
    copy_f32_16(&g_scratch.norm.norm_w[0], w_pkt + a0 * 256, 256);
  } else if (op == 1) {
    copy_f32_16(&g_scratch.rope.rope_cs[0][0], w_pkt + a0 * (B_SUB * HEAD_DIM), B_SUB * HEAD_DIM);
    copy_f32_16(g_scratch.rope.q_norm_w, w_pkt + 2048, 2 * HEAD_DIM);
  } else if (op == 6) {
    q4_res_w_impl(w_pkt, a0, a1);
  } else {
    q4_conv1d_w_step_impl(w_pkt, a0, a1);
  }
}

__attribute__((minsize)) void q4_lfm2_pre_x_op_impl(
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

__attribute__((minsize)) void q4_lfm2_pre_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
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
  if (op == 0) {
    src = (const float *)(g_scratch.norm.xq_out + arg * 256);
    if (arg >= 8) valid_f32 = 32;
  } else if (op == 5) {
    src = (const float *)g_scratch.gemm.xd_out;
    valid_f32 = 32;
  } else if (op == 8) {
    src = g_scratch.conv.conv_out + ((arg < 8) ? arg * 64 : 0);
    if (arg >= 8) {
      int32_t n_v = g_params[1];
      if (n_v < 1) n_v = 1;
      if (n_v > 64) n_v = 64;
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
  } else {
    const float *__restrict base =
        (op == 2) ? (const float *)g_scratch.rope.k_buf :
        (op == 7) ? (const float *)g_scratch.attn.q_bf16 :
        (op == 10) ? &g_scratch.attn.o_acc[0][0][0] :
                     &g_scratch.norm.x_slice[0][0];
    src = base + arg * 64;
  }
  copy_f32_16(y_out, src, valid_f32);
  if (valid_f32 < 64) q4_fill_f32_16(y_out + valid_f32, 0.0f, 64 - valid_f32);
}

}  // extern "C"

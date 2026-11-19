// Fused B=32 LFM2 / LFM2.5 Q4_0 Attention prefill kernel core (32 cores: 4 rows x 8 cols, npu2).
// Causal GQA (32 Q heads, 8 KV heads, head_dim=64) + QK-norm + NeoX RoPE + SwiGLU FFN (8192).

#include "q4_lfm2_attn_state.h"

extern "C" {
ScratchUnion g_scratch;
alignas(64) int32_t g_params[16];
alignas(64) bfloat16 g_q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_CORE * 8];
alignas(64) const int32_t lane_idx[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
}

__attribute__((noinline)) aie::vector<float, 16> q4_fmul_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  return aie::mul(a, b).template to_vector<float>();
}

__attribute__((noinline)) aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v) {
  return aie::inv(v);
}

static inline aie::vector<float, 16> q4_invsqrt_16(aie::vector<float, 16> v) {
  return aie::invsqrt(v);
}

__attribute__((noinline)) aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x) {
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
    p = aie::add(q4_fmul_16(p, f), aie::broadcast<float, 16>(coeffs[i]));
  return q4_fmul_16(p, ea.template to_vector<float>());
}

__attribute__((noinline)) void q4_scale_f32_16(
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

namespace {

#define fmul_16 q4_fmul_16
#define inv_16 q4_inv_16
#define exp2f_16 q4_exp2f_16

__attribute__((noinline)) static void copy_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::store_v(dst + i, aie::load_v<16>(src + i));
  }
}

__attribute__((noinline)) static aie::vector<float, 16> q4_sigmoid_16(aie::vector<float, 16> x) {
  aie::vector<float, 16> one_v = aie::broadcast<float, 16>(1.0f);
  aie::vector<float, 16> ml2e = aie::broadcast<float, 16>(-LOG2E);
  return inv_16(aie::add(one_v, exp2f_16(fmul_16(x, ml2e))));
}

__attribute__((noinline)) static float sum_sq_f32(const float *__restrict p, int n) {
  aie::accum<accfloat, 16> ssa = aie::zeros<accfloat, 16>();
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(p + i);
    ssa = aie::mac(ssa, v, v);
  }
  return aie::reduce_add(ssa.template to_vector<float>());
}

__attribute__((noinline)) static void apply_rmsnorm_n(
    const float *__restrict src,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float eps,
    int n,
    float *__restrict dst) {
  aie::vector<float, 16> mean = fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len));
  aie::vector<float, 16> inv_r = q4_invsqrt_16(aie::add(mean, aie::broadcast<float, 16>(eps)));
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = fmul_16(aie::load_v<16>(src + i), inv_r);
    aie::store_v(dst + i, fmul_16(v, aie::load_v<16>(nw + i)));
  }
}

static inline void f32_to_bf16_n(
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

__attribute__((noinline)) static void quantize_q8_block_8tok(
    const float *__restrict src,
    int32_t row_stride,
    uint8_t *__restrict xq_out,
    bfloat16 *__restrict xd_out) {
  aie::vector<float, 16> inv_qmax = aie::broadcast<float, 16>(1.0f / 127.0f);
  aie::vector<float, 16> tiny = aie::broadcast<float, 16>(1e-35f);

#pragma clang loop unroll(disable)
  for (int t = 0; t < B_CORE; ++t) {
    const float *r = src + t * row_stride;
    aie::vector<float, 16> v0 = aie::load_v<16>(r);
    aie::vector<float, 16> v1 = aie::load_v<16>(r + 16);
    float amax = aie::reduce_max(aie::max(aie::abs(v0), aie::abs(v1)));
    aie::vector<float, 16> dv = fmul_16(aie::broadcast<float, 16>(amax), inv_qmax);
    aie::accum<accfloat, 16> da;
    da.from_vector(dv);
    xd_out[t] = da.template to_vector<bfloat16>().get(0);
    aie::vector<float, 16> idv = inv_16(aie::max(dv, tiny));
    aie::vector<float, 16> s0 = fmul_16(v0, idv);
    aie::vector<float, 16> s1 = fmul_16(v1, idv);
    aie::vector<int8, 32> q32 = aie::to_fixed<int8>(aie::concat(s0, s1), 0);
    int grp = t >> 2;
    int rt = t & 3;
    aie::store_v((int8_t *)(xq_out + grp * 128 + rt * 16), q32.template extract<16>(0));
    aie::store_v((int8_t *)(xq_out + grp * 128 + 64 + rt * 16), q32.template extract<16>(1));
  }
}

__attribute__((noinline)) static void drain_xq_xd(
    float *__restrict y_out,
    const uint8_t *__restrict xq,
    const bfloat16 *__restrict xd,
    int32_t part_idx) {
  const float *__restrict src = (part_idx < 4) ? (const float *)(xq + part_idx * 512) : (const float *)xd;
  int valid_f32 = (part_idx < 4) ? 128 : 32;
  copy_f32_16(y_out, src, valid_f32);
  aie::vector<float, 16> z = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
  for (int i = valid_f32; i < 128; i += 16) aie::store_v(y_out + i, z);
}

__attribute__((noinline)) static void apply_rmsnorm_neox_rope_64(
    const float *__restrict in_h,
    const float *__restrict norm_w,
    float eps,
    const float *__restrict cs,
    bfloat16 *__restrict out_h,
    int32_t stride_db) {
  alignas(64) float tmp_n[HEAD_DIM];
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

__attribute__((noinline)) static void q4_rmsnorm_q8_pair_impl(
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
    float eps = 1e-5f;
    __builtin_memcpy(&eps, &g_params[7], 4);
    if (!skip_norm) {
#pragma clang loop unroll(disable)
      for (int t = 0; t < B_CORE; ++t) {
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
            g_scratch.norm.xq_out + b * 256,
            g_scratch.norm.xd_out + b * 8);
      }
    }
  }
}

__attribute__((noinline)) static void q4_rope_step_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t mode) {
  float eps = 1e-5f;
  __builtin_memcpy(&eps, &g_params[7], 4);

  if (mode == 0) {
#pragma clang loop unroll(disable)
    for (int local_t = 0; local_t < 2; ++local_t) {
      int t = pair_idx * 2 + local_t;
      const float *__restrict cs = &g_scratch.rope.rope_cs[t][0];
      const float *__restrict q_row = x_pkt + local_t * 2048 + col * 256;
#pragma clang loop unroll(disable)
      for (int h = 0; h < GQA_Q_HEADS; ++h) {
        bfloat16 *__restrict qh = g_q_bf16 + h * ((HEAD_DIM / 8) * B_CORE * 8) + t * 8;
        apply_rmsnorm_neox_rope_64(q_row + h * HEAD_DIM, g_scratch.rope.q_norm_w, eps, cs, qh, B_CORE * 8);
      }
    }
  } else {
    const int32_t kv_h = col;
    bfloat16 *__restrict base = (mode == 1) ? &g_scratch.rope.k_buf[0][0] : &g_scratch.rope.v_buf[0][0];
#pragma clang loop unroll(disable)
    for (int local_t = 0; local_t < 2; ++local_t) {
      int t = pair_idx * 2 + local_t;
      bfloat16 *__restrict dst = base + t * HEAD_DIM;
      const float *__restrict in_p = x_pkt + local_t * 512 + kv_h * HEAD_DIM;
      if (mode == 1) {
        apply_rmsnorm_neox_rope_64(in_p, g_scratch.rope.k_norm_w, eps, &g_scratch.rope.rope_cs[t][0], dst, 8);
      } else {
        f32_to_bf16_n(dst, in_p, HEAD_DIM);
      }
    }
  }
}

__attribute__((noinline)) static void q4_res_pair_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t which,
    int32_t pair_idx) {
#pragma clang loop unroll(disable)
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict src = x_pkt + local_t * 2048 + col * 256;
    float *__restrict dst = &g_scratch.norm.x_slice[t][0];
    if (which == 0) {
      copy_f32_16(dst, src, 256);
    } else {
#pragma clang loop unroll(disable)
      for (int i = 0; i < 256; i += 16) {
        aie::store_v(dst + i, aie::add(aie::load_v<16>(dst + i), aie::load_v<16>(src + i)));
      }
    }
  }
}

__attribute__((noinline)) static void q4_silu_round_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pass_in_kc,
    int32_t flags) {
  const int32_t is_up = flags & 1;

  if (!is_up) {
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_CORE; ++t) {
      copy_f32_16(&g_scratch.elem.a_buf[t][0], x_pkt + t * 512 + col * 64, 64);
    }
    return;
  }

#pragma clang loop unroll(disable)
  for (int t = 0; t < B_CORE; ++t) {
    float *__restrict g_ptr = &g_scratch.elem.a_buf[t][0];
    const float *__restrict u_ptr = x_pkt + t * 512 + col * 64;
#pragma clang loop unroll(disable)
    for (int i = 0; i < 64; i += 16) {
      aie::vector<float, 16> s = q4_sigmoid_16(aie::load_v<16>(g_ptr + i));
      aie::vector<float, 16> gu = fmul_16(aie::load_v<16>(g_ptr + i), s);
      aie::store_v(g_ptr + i, fmul_16(gu, aie::load_v<16>(u_ptr + i)));
    }
  }
#pragma clang loop unroll(disable)
  for (int b = 0; b < 2; ++b) {
    quantize_q8_block_8tok(
        &g_scratch.elem.a_buf[0][b * 32],
        64,
        g_scratch.elem.xq_out + b * 256,
        g_scratch.elem.xd_out + (pass_in_kc * 2 + b) * 8);
  }
}

__attribute__((noinline)) static void q4_flash_attn_drain_impl(float *__restrict y_out, int32_t tile_idx) {
  if (tile_idx == 0) {
    float *__restrict rlp = &g_scratch.attn.row_l[0][0];
#pragma clang loop unroll(disable)
    for (int i = 0; i < GQA_Q_HEADS * B_CORE; i += 16) {
      aie::vector<float, 16> rl = aie::max(aie::load_v<16>(rlp + i), aie::broadcast<float, 16>(1e-35f));
      aie::store_v(rlp + i, inv_16(rl));
    }
  }
  const int32_t h = tile_idx >> 2;
  const int32_t d0 = (tile_idx & 3) * 16;
#pragma clang loop unroll(disable)
  for (int t = 0; t < B_CORE; ++t) {
    q4_scale_f32_16(y_out + t * 16, &g_scratch.attn.o_acc[h][t][d0], g_scratch.attn.row_l[h][t], 16);
  }
}

}  // namespace

extern "C" {

void q4_lfm2_attn_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  if (op == 0 || op == 4) {
    if (op == 0) {
      aie::store_v((float *)g_params, aie::load_v<16>(w_pkt + 4096));
    }
    if (a1 >= 0) {
      copy_f32_16(&g_scratch.norm.norm_w[0], w_pkt + a1 * 2048 + a0 * 256, 256);
    }
  } else if (op == 1) {
    copy_f32_16(&g_scratch.rope.rope_cs[0][0], w_pkt + a0 * (B_CORE * HEAD_DIM), B_CORE * HEAD_DIM);
    copy_f32_16(g_scratch.rope.q_norm_w, w_pkt + 4096, HEAD_DIM);
    copy_f32_16(g_scratch.rope.k_norm_w, w_pkt + 4224, HEAD_DIM);
  }
}

void q4_lfm2_attn_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  if (op == 0) {
    q4_rmsnorm_q8_pair_impl(x_pkt, a0, a1, a2);
  } else if (op == 1) {
    q4_rope_step_impl(x_pkt, a0, a1, a2);
  } else if (op == 2) {
    q4_res_pair_impl(x_pkt, a0, a1, a2);
  } else {
    q4_silu_round_impl(x_pkt, a0, a1, a2);
  }
}

void q4_lfm2_attn_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  if (op == 0) {
    drain_xq_xd(y_out, g_scratch.norm.xq_out, g_scratch.norm.xd_out, arg);
  } else if (op == 3) {
    q4_flash_attn_drain_impl(y_out, arg);
  } else if (op == 5) {
    drain_xq_xd(y_out, g_scratch.elem.xq_out, g_scratch.elem.xd_out, (arg == 0) ? 0 : 4);
  } else {
    const float *__restrict base =
        (op == 1) ? g_scratch.gemm.y_acc :
        (op == 2) ? (const float *)g_scratch.rope.k_buf :
                    &g_scratch.norm.x_slice[0][0];
    copy_f32_16(y_out, base + arg * 128, 128);
  }
}

}  // extern "C"

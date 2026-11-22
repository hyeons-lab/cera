#include "q4_lfm2_dec_state.h"

extern "C" {
alignas(64) DecState g_dec = {};
alignas(64) const int32_t dec_lane_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
}

__attribute__((noinline, minsize)) aie::vector<float, 16> dec_fmul_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  aie::accum<accfloat, 16> acc = aie::mul(a, b);
  return acc.template to_vector<float>();
}

static inline __attribute__((always_inline)) aie::vector<float, 16> fast_fmul2_16(
    aie::vector<float, 16> a, aie::vector<float, 16> b) {
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
#define dec_fmul_16 fast_fmul2_16

__attribute__((noinline, minsize)) aie::vector<float, 16> dec_inv_16(aie::vector<float, 16> v) {
  return aie::inv(v);
}

__attribute__((noinline)) aie::vector<float, 16> dec_exp2f_16(aie::vector<float, 16> x) {
  static const float kExp2C[5] = {
      0.0096130371f,
      0.0555038452f,
      0.2402267456f,
      0.6931471825f,
      1.0f,
  };
  x = aie::max(x, aie::broadcast<float, 16>(-126.0f));
  aie::vector<float, 16> n = aie::to_float<float>(aie::to_fixed<int32>(x, 0));
  aie::vector<float, 16> f = aie::sub(x, n);
  aie::vector<float, 16> p = aie::broadcast<float, 16>(0.0013427734f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 5; ++i) {
    p = aie::add(dec_fmul_16(p, f), aie::broadcast<float, 16>(kExp2C[i]));
  }
  aie::vector<int32, 16> ni = aie::to_fixed<int32>(n, 0);
  aie::vector<int32, 16> exp_bits = aie::upshift(aie::add(ni, aie::broadcast<int32, 16>(127)), 23);
  return dec_fmul_16(p, aie::vector_cast<float>(exp_bits));
}

namespace {

static constexpr float kQScaleLog2e = 0.125f * 1.4426950408889634f;

__attribute__((minsize)) static inline aie::vector<float, 16> dec_invsqrt_16(aie::vector<float, 16> v) {
  return aie::invsqrt(v);
}

__attribute__((noinline)) static aie::vector<float, 16> dec_sigmoid_16(aie::vector<float, 16> x) {
  aie::vector<float, 16> neg_l2e = aie::broadcast<float, 16>(-1.4426950408889634f);
  aie::vector<float, 16> e = dec_exp2f_16(dec_fmul_16(x, neg_l2e));
  return dec_inv_16(aie::add(aie::broadcast<float, 16>(1.0f), e));
}

__attribute__((noinline, minsize)) static void copy_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) aie::store_v(dst + i, aie::load_v<16>(src + i));
}

__attribute__((noinline, minsize)) static float sum_sq_f32(const float *__restrict x, int n) {
  aie::vector<float, 16> s = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(x + i);
    s = aie::add(s, dec_fmul_16(v, v));
  }
  return aie::reduce_add(s);
}

__attribute__((noinline, minsize)) static void apply_rmsnorm_n(
    const float *__restrict src,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float eps,
    float post_scale,
    int n,
    float *__restrict dst) {
  aie::vector<float, 16> mean = dec_fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len));
  aie::vector<float, 16> inv_r = dec_invsqrt_16(aie::add(mean, aie::broadcast<float, 16>(eps)));
  inv_r = dec_fmul_16(inv_r, aie::broadcast<float, 16>(post_scale));
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = dec_fmul_16(aie::load_v<16>(src + i), inv_r);
    aie::store_v(dst + i, dec_fmul_16(v, aie::load_v<16>(nw + i)));
  }
}

__attribute__((noinline, minsize)) static void f32_to_bf16_n(
    bfloat16 *__restrict dst,
    const float *__restrict src,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::store_v(dst + i, dec_f32_to_bf16_16(aie::load_v<16>(src + i)));
  }
}

__attribute__((noinline)) static void quantize_q8_block32_x2(
    const float *__restrict src,
    uint8_t *__restrict xq_out,
    bfloat16 *__restrict xd_out) {
  aie::vector<float, 16> inv_qmax = aie::broadcast<float, 16>(1.0f / 127.0f);
  aie::vector<float, 16> tiny = aie::broadcast<float, 16>(1e-35f);
#pragma clang loop unroll(disable)
  for (int b = 0; b < 2; ++b) {
    const float *r = src + b * 32;
    aie::vector<float, 16> mv = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
    for (int h = 0; h < 2; ++h) {
      mv = aie::max(mv, aie::abs(aie::load_v<16>(r + h * 16)));
    }
    float amax = aie::reduce_max(mv);
    aie::vector<float, 16> dv = dec_fmul_16(aie::broadcast<float, 16>(amax), inv_qmax);
    xd_out[b] = dec_f32_to_bf16_16(dv).get(0);
    aie::vector<float, 16> idv = dec_inv_16(aie::max(dv, tiny));
#pragma clang loop unroll(disable)
    for (int h = 0; h < 2; ++h) {
      aie::vector<int8, 16> q = aie::to_fixed<int8>(dec_fmul_16(aie::load_v<16>(r + h * 16), idv), 0);
      aie::store_v((int8_t *)(xq_out + b * 32 + h * 16), q);
    }
  }
}

__attribute__((noinline, minsize)) static void apply_rmsnorm_neox_rope_64(
    const float *__restrict in_h,
    const float *__restrict norm_w,
    float eps,
    float post_scale,
    const float *__restrict cs,
    bfloat16 *__restrict out_h) {
  alignas(64) float tmp_n[HEAD_DIM];
  float ss = sum_sq_f32(in_h, HEAD_DIM);
  apply_rmsnorm_n(in_h, norm_w, ss, 1.0f / (float)HEAD_DIM, eps, post_scale, HEAD_DIM, tmp_n);
  const float *__restrict c = cs;
  const float *__restrict s = cs + 32;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 32; i += 16) {
    aie::vector<float, 16> x0 = aie::load_v<16>(tmp_n + i);
    aie::vector<float, 16> x1 = aie::load_v<16>(tmp_n + 32 + i);
    aie::vector<float, 16> cv = aie::load_v<16>(c + i);
    aie::vector<float, 16> sv = aie::load_v<16>(s + i);
    aie::store_v(out_h + i, dec_f32_to_bf16_16(aie::sub(dec_fmul_16(x0, cv), dec_fmul_16(x1, sv))));
    aie::store_v(out_h + 32 + i, dec_f32_to_bf16_16(aie::add(dec_fmul_16(x0, sv), dec_fmul_16(x1, cv))));
  }
}

static inline float read_eps() {
  float eps = 1e-5f;
  __builtin_memcpy(&eps, &g_dec.params[7], 4);
  return eps;
}

__attribute__((noinline, minsize)) static void q4_dec_rms_q8_step(
    const float *__restrict w_pkt,
    int32_t core_id,
    int32_t flags) {
  const int32_t save_res = flags & 1;
  const int32_t skip_norm = (flags >> 1) & 1;
  const float ss = skip_norm ? 0.0f : sum_sq_f32(w_pkt, CHUNK_K);
  copy_f32_16(g_dec.x_slice, w_pkt + core_id * 64, 64);
  if (save_res) {
#pragma clang loop unroll(disable)
    for (int p = 0; p < 4; ++p) {
      copy_f32_16(g_dec.res_acc + p * 16, w_pkt + p * 512 + core_id * 16, 16);
    }
  }
  if (!skip_norm) {
    apply_rmsnorm_n(g_dec.x_slice, g_dec.norm_w, ss, 1.0f / (float)CHUNK_K, read_eps(), 1.0f, 64, g_dec.x_slice);
  }
  quantize_q8_block32_x2(g_dec.x_slice, g_dec.xq_out, g_dec.xd_out);
}

__attribute__((noinline, minsize)) static void q4_dec_rope_step(
    const float *__restrict w_pkt,
    int32_t core_id,
    int32_t mode) {
  const int32_t c = core_id >> 2;
  const int32_t r = core_id & 3;
  if (mode == 2) {
    copy_f32_16((float *)g_dec.q_bf16, w_pkt + r * 64, 32);
    return;
  }
  if (mode == 3) {
    if (g_dec.params[9]) {
      const float *__restrict src = w_pkt + r * 128;
      copy_f32_16(g_dec.o_acc, src, 64);
      g_dec.row_m = src[64];
      g_dec.row_l = src[65];
    }
    return;
  }
  const float eps = read_eps();
  apply_rmsnorm_neox_rope_64(w_pkt + core_id * HEAD_DIM, g_dec.q_norm_w, eps, kQScaleLog2e, g_dec.rope_cs, g_dec.q_bf16);
  if (r == 2) {
    apply_rmsnorm_neox_rope_64(w_pkt + 2560 + c * HEAD_DIM, g_dec.k_norm_w, eps, 1.0f, g_dec.rope_cs, g_dec.kv_bf16);
  } else if (r == 3) {
    f32_to_bf16_n(g_dec.kv_bf16, w_pkt + 3072 + c * HEAD_DIM, HEAD_DIM);
  }
}

__attribute__((noinline, minsize)) static void q4_dec_conv_step() {
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 16) {
    aie::vector<float, 16> bx = dec_fmul_16(aie::load_v<16>(&g_dec.ffn_gate[0][0] + i), aie::load_v<16>(&g_dec.y_gemm[0][0] + i));
    aie::vector<float, 16> acc = dec_fmul_16(bx, aie::load_v<16>(&g_dec.conv_wt[2][i]));
#pragma clang loop unroll(disable)
    for (int k = 0; k < 2; ++k) {
      acc = aie::add(acc, dec_fmul_16(aie::load_v<16>(&g_dec.conv_state[k][i]), aie::load_v<16>(&g_dec.conv_wt[k][i])));
    }
    aie::store_v(g_dec.x_slice + i, dec_fmul_16(acc, aie::load_v<16>(g_dec.conv_c + i)));
    aie::store_v(g_dec.conv_bx + i, bx);
  }
  copy_f32_16(&g_dec.conv_state[0][0], &g_dec.conv_state[1][0], 64);
  copy_f32_16(&g_dec.conv_state[1][0], g_dec.conv_bx, 64);
}

__attribute__((noinline, minsize)) static void q4_dec_silu_mul_step(int32_t g_slot) {
  const float *__restrict gp =
      (g_slot == 0) ? &g_dec.ffn_gate[0][0] : &g_dec.ffn_gates_hi[g_slot - 1][0][0];
#pragma clang loop unroll(disable)
  for (int p = 0; p < 4; ++p) {
    aie::vector<float, 16> gv = aie::load_v<16>(gp + p * 16);
    aie::vector<float, 16> uv = aie::load_v<16>(&g_dec.y_gemm[p][0]);
    aie::store_v(&g_dec.y_gemm[p][0], dec_fmul_16(dec_fmul_16(gv, dec_sigmoid_16(gv)), uv));
  }
}

}  // namespace

extern "C" {

__attribute__((minsize)) void q4_lfm2_dec_w_op_impl(
    const float *__restrict w_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  if (op == 0) {
    copy_f32_16(g_dec.norm_w, w_pkt + a0 * 64, 64);
    if (a1) {
      aie::store_v((float *)g_dec.params, aie::load_v<16>(w_pkt + 4096));
      copy_f32_16(g_dec.rope_cs, w_pkt + 4112, HEAD_DIM);
      copy_f32_16(g_dec.q_norm_w, w_pkt + 4240, HEAD_DIM);
      copy_f32_16(g_dec.k_norm_w, w_pkt + 4368, HEAD_DIM);
    }
  } else if (op == 3 || op == 4) {
    const int n = (op == 3) ? 3 : 2;
    float *__restrict dst = (op == 3) ? &g_dec.conv_wt[0][0] : &g_dec.conv_state[0][0];
#pragma clang loop unroll(disable)
    for (int k = 0; k < n; ++k) copy_f32_16(dst + k * 64, w_pkt + k * 256 + a0 * 64, 64);
  } else if (op == 5) {
    const int n_chk = a1 ? 4 : 1;
    const int stride = a1 ? 36 : 64;
#pragma clang loop unroll(disable)
    for (int kc = 0; kc < n_chk; ++kc) {
      const float *__restrict wp = w_pkt + kc * 1152;
      float *__restrict dp = g_dec.xq_buf + kc * X_ELEMS;
#pragma clang loop unroll(disable)
      for (int i = 0; i < 32; ++i) {
        const float *__restrict s = wp + i * stride;
        float *__restrict d = dp + i * 20;
        aie::store_v(d + 0, aie::load_v<4>(s + 0));
        aie::store_v(d + 4, aie::load_v<4>(s + 4));
        aie::store_v(d + 8, aie::load_v<4>(s + 8));
        aie::store_v(d + 12, aie::load_v<4>(s + 12));
        d[16] = s[16];
      }
    }
  } else if (op == 6) {
    q4_dec_rms_q8_step(w_pkt, a0, a2);
  } else if (op == 7) {
#pragma clang loop unroll(disable)
    for (int p = 0; p < 4; ++p) {
      copy_f32_16(g_dec.res_acc + p * 16, w_pkt + p * 512 + a0 * 16, 16);
    }
  } else {
    q4_dec_rope_step(w_pkt, a0, op - 8);
  }
}

__attribute__((noinline, minsize)) static void drain_q8_y_out(float *__restrict y_out) {
  copy_f32_16(y_out, (const float *)g_dec.xq_out, 16);
  y_out[16] = *(const float *)g_dec.xd_out;
#pragma clang loop unroll(disable)
  for (int i = 17; i < 64; ++i) y_out[i] = 0.0f;
}

__attribute__((minsize)) void q4_lfm2_dec_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  if (op == 0 || op == 6 || op == 7) {
    if (op == 6) {
      q4_dec_silu_mul_step(arg);
      quantize_q8_block32_x2(&g_dec.y_gemm[0][0], g_dec.xq_out, g_dec.xd_out);
    } else if (op == 7) {
      q4_dec_conv_step();
      quantize_q8_block32_x2(g_dec.x_slice, g_dec.xq_out, g_dec.xd_out);
    }
    drain_q8_y_out(y_out);
  } else if (op == 4) {
    float denom = aie::max(aie::broadcast<float, 16>(g_dec.row_l), aie::broadcast<float, 16>(1e-35f)).get(0);
    aie::vector<float, 16> vsc = dec_inv_16(aie::broadcast<float, 16>(denom));
#pragma clang loop unroll(disable)
    for (int d = 0; d < HEAD_DIM; d += 16) {
      aie::store_v(g_dec.o_acc + d, dec_fmul_16(aie::load_v<16>(g_dec.o_acc + d), vsc));
    }
    quantize_q8_block32_x2(g_dec.o_acc, g_dec.xq_out, g_dec.xd_out);
    drain_q8_y_out(y_out);
  } else if (op == 9) {
    if (arg == 0) {
      if (g_dec.params[6] == 2) {
        aie::vector<float, 16> zf = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
        for (int d = 0; d < HEAD_DIM; d += 16) aie::store_v(g_dec.o_acc + d, zf);
        g_dec.row_m = -1e30f;
        g_dec.row_l = 0.0f;
      }
      copy_f32_16(y_out, g_dec.o_acc, 64);
    } else {
      y_out[0] = g_dec.row_m;
      y_out[1] = g_dec.row_l;
    }
  } else if (op == 11) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < 64; i += 16) {
      aie::store_v(
          y_out + i,
          aie::add(aie::load_v<16>(&g_dec.y_gemm[0][0] + i), aie::load_v<16>(g_dec.res_acc + i)));
    }
  } else {
    const float *__restrict src =
        (op == 1) ? &g_dec.y_gemm[0][0] :
        (op == 2) ? g_dec.x_slice :
        (op == 3) ? &g_dec.conv_state[arg][0] :
        (op == 8) ? (const float *)g_dec.q_bf16 :
                    (const float *)g_dec.kv_bf16;
    copy_f32_16(y_out, src, 64);
  }
}

}  // extern "C"

// Fused B=32 LFM2 / LFM2.5 Q4_0 ShortConv prefill kernel core (32 cores: 4 rows x 8 cols, npu2).
// Mode 0 only: RMSNorm+Q8 -> B/C/X IN_PROJ -> 3-tap causal conv + conv_state (d_conv=2)
// -> Q8 + OUT_PROJ + res1 -> ffn_norm+Q8 -> FFN SiLU*Up (8192) -> Down + res2.

#include "q4_lfm2_conv_state.h"

extern "C" {
ScratchUnion g_scratch;
int32_t g_params[16];
alignas(64) float g_conv_out[16 * 64];
}

__attribute__((always_inline)) inline aie::vector<float, 16> fmul_16(
    aie::vector<float, 16> a,
    aie::vector<float, 16> b) {
  return aie::mul(a, b).template to_vector<float>();
}

aie::vector<float, 16> q4_fmul_16(aie::vector<float, 16> a, aie::vector<float, 16> b) {
  return fmul_16(a, b);
}

aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v) {
  return aie::inv(v);
}

aie::vector<float, 16> q4_exp2f_16(aie::vector<float, 16> x) {
  x = aie::max(x, aie::broadcast<float, 16>(-126.0f));
  x = aie::min(x, aie::broadcast<float, 16>(126.0f));
  aie::vector<int32_t, 16> ki = aie::to_fixed<int32_t>(x, 0);
  aie::vector<float, 16> kf = aie::to_float<float>(ki, 0);
  aie::vector<float, 16> f = aie::sub(x, kf);

  constexpr float c0 = 1.0f;
  constexpr float c1 = 0.69314718f;
  constexpr float c2 = 0.24022651f;
  constexpr float c3 = 0.05550411f;
  constexpr float c4 = 0.00961813f;

  aie::vector<float, 16> p = aie::add(fmul_16(aie::broadcast<float, 16>(c4), f), aie::broadcast<float, 16>(c3));
  p = aie::add(fmul_16(p, f), aie::broadcast<float, 16>(c2));
  p = aie::add(fmul_16(p, f), aie::broadcast<float, 16>(c1));
  p = aie::add(fmul_16(p, f), aie::broadcast<float, 16>(c0));

  aie::vector<int32_t, 16> exp_bits = aie::upshift(aie::add(ki, aie::broadcast<int32_t, 16>(127)), 23);
  aie::vector<float, 16> scale = aie::vector_cast<float>(exp_bits);
  return fmul_16(p, scale);
}

namespace {

__attribute__((noinline)) static void copy_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    int n) {
  for (int i = 0; i < n; i += 16) chess_prepare_for_pipelining {
    aie::store_v(dst + i, aie::load_v<16>(src + i));
  }
}

// Quantize one 32-element block across all 8 tokens into [grp=0..1][half=0..1][rt=0..3][16] Q8 + bf16 scales.
__attribute__((noinline)) static void quantize_q8_block_8tok(
    const float *__restrict src,
    int32_t row_stride,
    uint8_t *__restrict xq_out,
    bfloat16 *__restrict xd_out) {
  aie::vector<float, 16> inv_qmax = aie::broadcast<float, 16>(1.0f / 127.0f);
  aie::vector<float, 16> zero = aie::zeros<float, 16>();

  for (int t = 0; t < B_CORE; ++t) {
    const float *r = src + t * row_stride;
    aie::vector<float, 16> v0 = aie::load_v<16>(r);
    aie::vector<float, 16> v1 = aie::load_v<16>(r + 16);
    float amax = aie::reduce_max(aie::max(aie::abs(v0), aie::abs(v1)));
    aie::vector<float, 16> dv = fmul_16(aie::broadcast<float, 16>(amax), inv_qmax);
    aie::accum<accfloat, 16> da;
    da.from_vector(dv);
    aie::vector<bfloat16, 16> dbf = da.template to_vector<bfloat16>();
    xd_out[t] = dbf.get(0);
    aie::accum<accfloat, 16> dr_acc;
    dr_acc.from_vector(dbf);
    aie::vector<float, 16> d_rnd = dr_acc.template to_vector<float>();
    aie::vector<float, 16> idv = aie::select(zero, q4_inv_16(d_rnd), aie::lt(zero, d_rnd));
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
  for (int i = valid_f32; i < 128; i += 16) aie::store_v(y_out + i, z);
}

static void q4_rmsnorm_q8_pair_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t flags) {
  const int32_t skip_norm = flags & 1;
  const int32_t f32_out = (flags >> 1) & 1;
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict row_ptr = x_pkt + local_t * 2048;
    if (!skip_norm) {
      aie::accum<accfloat, 16> ssa = aie::zeros<accfloat, 16>();
      for (int d = 0; d < 2048; d += 16) chess_prepare_for_pipelining {
        aie::vector<float, 16> v = aie::load_v<16>(row_ptr + d);
        ssa = aie::mac(ssa, v, v);
      }
      g_scratch.norm.sum_sq[t] = aie::reduce_add(ssa.template to_vector<float>());
    }
    copy_f32_16(&g_scratch.norm.x_slice[t][0], row_ptr + col * 256, 256);
  }
  if (pair_idx == 3) {
    float eps = 1e-5f;
    __builtin_memcpy(&eps, &g_params[7], 4);
    if (!skip_norm) {
      aie::vector<float, 16> inv_n = aie::broadcast<float, 16>(1.0f / 2048.0f);
      aie::vector<float, 16> eps_v = aie::broadcast<float, 16>(eps);
      for (int t = 0; t < B_CORE; ++t) {
        aie::vector<float, 16> mean = fmul_16(aie::broadcast<float, 16>(g_scratch.norm.sum_sq[t]), inv_n);
        aie::vector<float, 16> inv_rv = aie::invsqrt(aie::add(mean, eps_v));
        float *__restrict xs = &g_scratch.norm.x_slice[t][0];
        const float *__restrict nw = &g_scratch.norm.norm_w[0];
        for (int i = 0; i < 256; i += 16) chess_prepare_for_pipelining {
          aie::vector<float, 16> xv = fmul_16(aie::load_v<16>(xs + i), inv_rv);
          aie::store_v(xs + i, fmul_16(xv, aie::load_v<16>(nw + i)));
        }
      }
    }
    if (!f32_out) {
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

static void q4_res_pair_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t which,
    int32_t pair_idx) {
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict src = x_pkt + local_t * 2048 + col * 256;
    float *__restrict dst = &g_scratch.norm.x_slice[t][0];
    if (which == 0) {
      copy_f32_16(dst, src, 256);
    } else {
      for (int i = 0; i < 256; i += 16) chess_prepare_for_pipelining {
        aie::store_v(dst + i, aie::add(aie::load_v<16>(dst + i), aie::load_v<16>(src + i)));
      }
    }
  }
}

static void q4_silu_round_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pass_in_kc,
    int32_t flags) {
  const int32_t is_up = flags & 1;

  if (!is_up) {
    for (int t = 0; t < B_CORE; ++t) {
      copy_f32_16(&g_scratch.elem.a_buf[t][0], x_pkt + t * 512 + col * 64, 64);
    }
    return;
  }

  aie::vector<float, 16> one_v = aie::broadcast<float, 16>(1.0f);
  aie::vector<float, 16> ml2e = aie::broadcast<float, 16>(-LOG2E);
  for (int t = 0; t < B_CORE; ++t) {
    float *__restrict g_ptr = &g_scratch.elem.a_buf[t][0];
    const float *__restrict u_ptr = x_pkt + t * 512 + col * 64;
    for (int i = 0; i < 64; i += 16) chess_prepare_for_pipelining {
      aie::vector<float, 16> g = aie::load_v<16>(g_ptr + i);
      aie::vector<float, 16> u = aie::load_v<16>(u_ptr + i);
      aie::vector<float, 16> sig = q4_inv_16(aie::add(one_v, q4_exp2f_16(fmul_16(g, ml2e))));
      aie::vector<float, 16> silu = fmul_16(g, sig);
      aie::store_v(g_ptr + i, fmul_16(silu, u));
    }
  }
  for (int b = 0; b < 2; ++b) {
    quantize_q8_block_8tok(
        &g_scratch.elem.a_buf[0][b * 32],
        64,
        g_scratch.elem.xq_out + b * 256,
        g_scratch.elem.xd_out + (pass_in_kc * 2 + b) * 8);
  }
}

static void q4_conv1d_w_step_impl(
    const float *__restrict w_pkt,
    int32_t row,
    int32_t stage) {
  const int32_t r_off = row * 64;
  if (stage == 0) {
    for (int k = 0; k < D_CONV; ++k) {
      copy_f32_16(&g_scratch.conv.state[k][0], w_pkt + k * 256 + r_off, 64);
    }
  } else if (stage == 1 || stage == 2) {
    int32_t t_base = (stage - 1) * 16;
    for (int t = 0; t < 16; ++t) {
      copy_f32_16(&g_scratch.conv.bx[t_base + t][0], w_pkt + t * 256 + r_off, 64);
    }
  } else if (stage == 3 || stage == 4) {
    int32_t t_base = (stage - 3) * 16;
    for (int t = 0; t < 16; ++t) {
      float *__restrict bxp = &g_scratch.conv.bx[t_base + t][0];
      const float *__restrict xp = w_pkt + t * 256 + r_off;
      for (int i = 0; i < 64; i += 16) chess_prepare_for_pipelining {
        aie::store_v(bxp + i, fmul_16(aie::load_v<16>(bxp + i), aie::load_v<16>(xp + i)));
      }
    }
  } else {
    int32_t t_base = (stage - 5) * 16;
    for (int t = 0; t < 16; ++t) {
      int32_t ti = t_base + t;
      const float *__restrict cp = w_pkt + t * 256 + r_off;
      float *__restrict op = g_conv_out + t * 64;
      for (int i = 0; i < 64; i += 16) {
        aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
        for (int k = 0; k < L_CACHE; ++k) {
          int32_t src_ti = ti + k - D_CONV;
          const float *__restrict sp = (src_ti < 0) ? &g_scratch.conv.state[D_CONV + src_ti][i]
                                                    : &g_scratch.conv.bx[src_ti][i];
          acc = aie::mac(acc, aie::load_v<16>(sp), aie::load_v<16>(&g_scratch.conv.wt[k][i]));
        }
        aie::store_v(op + i, fmul_16(acc.template to_vector<float>(), aie::load_v<16>(cp + i)));
      }
    }
  }
}

__attribute__((noinline)) static void q4_conv1d_finalize_state() {
  int32_t n_v = g_params[1];
  if (n_v < 1) n_v = 1;
  if (n_v > 32) n_v = 32;
  for (int k = 0; k < D_CONV; ++k) {
    int32_t idx = n_v - D_CONV + k;
    const float *__restrict src =
        (idx < 0) ? &g_scratch.conv.state[D_CONV + idx][0] : &g_scratch.conv.bx[idx][0];
    copy_f32_16(g_conv_out + k * 64, src, 64);
  }
}

}  // namespace

extern "C" {

void q4_lfm2_conv_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  if (op == 0) {
    const int32_t *__restrict ip = (const int32_t *)(w_pkt + 4096);
    for (int i = 0; i < 16; ++i) g_params[i] = ip[i];
    if (a1 >= 0) {
      copy_f32_16(&g_scratch.norm.norm_w[0], w_pkt + a1 * 2048 + a0 * 256, 256);
    }
  } else if (op == 4) {
    copy_f32_16(&g_scratch.norm.norm_w[0], w_pkt + a1 * 2048 + a0 * 256, 256);
  } else if (op == 3) {
    int32_t r_off = a0 * 64;
    for (int k = 0; k < L_CACHE; ++k) {
      copy_f32_16(&g_scratch.conv.wt[k][0], w_pkt + k * 256 + r_off, 64);
    }
  } else {
    q4_conv1d_w_step_impl(w_pkt, a0, a1);
  }
}

void q4_lfm2_conv_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  if (op == 0) {
    q4_rmsnorm_q8_pair_impl(x_pkt, a0, a1, a2);
  } else if (op == 2) {
    q4_res_pair_impl(x_pkt, a0, a1, a2);
  } else {
    q4_silu_round_impl(x_pkt, a0, a1, a2);
  }
}

void q4_lfm2_conv_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  if (op == 0) {
    drain_xq_xd(y_out, g_scratch.norm.xq_out, g_scratch.norm.xd_out, arg);
  } else if (op == 1) {
    copy_f32_16(y_out, g_scratch.gemm.y_acc + arg * (B_CORE * ROWS), 128);
  } else if (op == 4 || op == 6) {
    copy_f32_16(y_out, &g_scratch.norm.x_slice[arg >> 1][(arg & 1) * 128], 128);
  } else if (op == 5) {
    drain_xq_xd(y_out, g_scratch.elem.xq_out, g_scratch.elem.xd_out, (arg == 0) ? 0 : 4);
  } else {
    // op == 8: arg = 0..7 -> o_chunk[16][64], arg == 8 -> finalize & drain 2 conv state frames.
    if (arg == 8) q4_conv1d_finalize_state();
    int32_t idx = (arg < 8) ? arg : 0;
    copy_f32_16(y_out, g_conv_out + idx * 128, 128);
  }
}

}  // extern "C"

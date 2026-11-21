#include "q4_lfm3_dec_state.h"

extern "C" {
alignas(64) DecState g_dec = {};
alignas(64) const int32_t dec_lane_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
}

__attribute__((noinline, minsize)) aie::vector<float, 16> dec_fmul_16(
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

__attribute__((noinline)) aie::vector<float, 16> dec_inv_16(aie::vector<float, 16> v) {
  return aie::inv(v);
}

__attribute__((noinline)) aie::vector<float, 16> dec_exp2f_16(aie::vector<float, 16> x) {
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

namespace {

static inline aie::vector<float, 16> dec_invsqrt_16(aie::vector<float, 16> v) {
  return aie::invsqrt(v);
}

__attribute__((noinline)) static aie::vector<float, 16> dec_sigmoid_16(aie::vector<float, 16> x) {
  aie::vector<float, 16> neg_l2e = aie::broadcast<float, 16>(-1.4426950408889634f);
  aie::vector<float, 16> e = dec_exp2f_16(dec_fmul_16(x, neg_l2e));
  return dec_inv_16(aie::add(aie::broadcast<float, 16>(1.0f), e));
}

__attribute__((noinline)) static void copy_f32_16(
    float *__restrict dst,
    const float *__restrict src,
    int n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) aie::store_v(dst + i, aie::load_v<16>(src + i));
}

__attribute__((noinline)) static float sum_sq_f32(const float *__restrict x, int n) {
  aie::vector<float, 16> s = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(x + i);
    s = aie::add(s, dec_fmul_16(v, v));
  }
  return aie::reduce_add(s);
}

__attribute__((noinline)) static void apply_rmsnorm_n(
    const float *__restrict src,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float eps,
    int n,
    float post_scale,
    float *__restrict dst) {
  aie::vector<float, 16> mean = dec_fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len));
  aie::vector<float, 16> inv_r = dec_fmul_16(
      dec_invsqrt_16(aie::add(mean, aie::broadcast<float, 16>(eps))),
      aie::broadcast<float, 16>(post_scale));
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = dec_fmul_16(aie::load_v<16>(src + i), inv_r);
    aie::store_v(dst + i, dec_fmul_16(v, aie::load_v<16>(nw + i)));
  }
}

__attribute__((noinline)) static void f32_to_bf16_n(
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

__attribute__((noinline)) static void apply_rmsnorm_neox_rope_128(
    const float *__restrict in_h,
    const float *__restrict norm_w,
    float eps,
    const float *__restrict cs,
    float post_scale,
    bfloat16 *__restrict out_h) {
  alignas(64) float tmp_n[HEAD_DIM];
  float ss = sum_sq_f32(in_h, HEAD_DIM);
  apply_rmsnorm_n(in_h, norm_w, ss, 1.0f / (float)HEAD_DIM, eps, HEAD_DIM, post_scale, tmp_n);
  const float *__restrict c = cs;
  const float *__restrict s = cs + 64;
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 16) {
    aie::vector<float, 16> x0 = aie::load_v<16>(tmp_n + i);
    aie::vector<float, 16> x1 = aie::load_v<16>(tmp_n + 64 + i);
    aie::vector<float, 16> cv = aie::load_v<16>(c + i);
    aie::vector<float, 16> sv = aie::load_v<16>(s + i);
    aie::store_v(out_h + i, dec_f32_to_bf16_16(aie::sub(dec_fmul_16(x0, cv), dec_fmul_16(x1, sv))));
    aie::store_v(out_h + 64 + i, dec_f32_to_bf16_16(aie::add(dec_fmul_16(x0, sv), dec_fmul_16(x1, cv))));
  }
}

static inline float read_eps() {
  float eps = 1e-5f;
  __builtin_memcpy(&eps, &g_dec.params[7], 4);
  return eps;
}

__attribute__((noinline)) static void accum_chunk_slice(
    const float *__restrict x_pkt,
    int32_t core_id,
    int32_t chunk) {
  if (chunk == 0) g_dec.sum_sq = 0.0f;
  g_dec.sum_sq += sum_sq_f32(x_pkt, 512);
  if (chunk == (core_id >> 3)) {
    copy_f32_16(g_dec.x_slice, x_pkt + (core_id & 7) * 64, 64);
  }
}

__attribute__((noinline)) static void q4_dec_rms_q8_step(
    const float *__restrict x_pkt,
    int32_t core_id,
    int32_t chunk,
    int32_t flags) {
  const int32_t save_res = flags & 1;
  const int32_t skip_norm = (flags >> 1) & 1;
  accum_chunk_slice(x_pkt, core_id, chunk);
  if (save_res && chunk == (core_id >> 3)) copy_f32_16(g_dec.res_acc, g_dec.x_slice, 64);
  if (chunk == 3) {
    if (!skip_norm) {
      apply_rmsnorm_n(g_dec.x_slice, g_dec.norm_w, g_dec.sum_sq, 1.0f / 2048.0f, read_eps(), 64, 1.0f, g_dec.x_slice);
    }
    quantize_q8_block32_x2(g_dec.x_slice, g_dec.xq_out, g_dec.xd_out);
  }
}

__attribute__((noinline)) static void q4_dec_post_res_step(
    const float *__restrict x_pkt,
    int32_t core_id,
    int32_t chunk,
    int32_t update_res) {
  accum_chunk_slice(x_pkt, core_id, chunk);
  if (chunk == 3) {
    apply_rmsnorm_n(g_dec.x_slice, g_dec.post_norm_w, g_dec.sum_sq, 1.0f / 2048.0f, read_eps(), 64, 1.0f, g_dec.x_slice);
#pragma clang loop unroll(disable)
    for (int i = 0; i < 64; i += 16) {
      aie::store_v(g_dec.x_slice + i, aie::add(aie::load_v<16>(g_dec.x_slice + i), aie::load_v<16>(g_dec.res_acc + i)));
    }
    if (update_res) copy_f32_16(g_dec.res_acc, g_dec.x_slice, 64);
  }
}

__attribute__((noinline)) static void q4_dec_rope_step(
    const float *__restrict x_pkt,
    int32_t core_id,
    int32_t chunk,
    int32_t mode) {
  const int32_t c = core_id >> 2;
  const int32_t r = core_id & 3;
  const float eps = read_eps();
  if (mode == 0) {
    if (r >= 2) return;
    const int32_t q_h = c * 2 + r;
    if (chunk == (q_h >> 2)) {
      constexpr float kQScaleLog2e = 0.08838834764831845f * 1.4426950408889634f;
      apply_rmsnorm_neox_rope_128(x_pkt + (q_h & 3) * HEAD_DIM, g_dec.q_norm_w, eps, g_dec.rope_cs, kQScaleLog2e, g_dec.q_bf16);
    }
  } else {
    if (c & 1) return;
    const int32_t kv_h = c >> 1;
    if (r == 2 && chunk == 0) {
      apply_rmsnorm_neox_rope_128(x_pkt + kv_h * HEAD_DIM, g_dec.k_norm_w, eps, g_dec.rope_cs, 1.0f, g_dec.q_bf16);
    } else if (r == 3 && chunk == 1) {
      f32_to_bf16_n(g_dec.q_bf16, x_pkt + kv_h * HEAD_DIM, HEAD_DIM);
    }
  }
}

__attribute__((noinline)) static void q4_dec_conv_step() {
  const float *__restrict b_ptr = &g_dec.ffn_gate[0][0];
  const float *__restrict x_ptr = &g_dec.y_gemm[0][0];
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 16) {
    aie::vector<float, 16> bx = dec_fmul_16(aie::load_v<16>(b_ptr + i), aie::load_v<16>(x_ptr + i));
    aie::vector<float, 16> acc = dec_fmul_16(bx, aie::load_v<16>(&g_dec.conv_wt[6][i]));
#pragma clang loop unroll(disable)
    for (int k = 0; k < 6; ++k) {
      acc = aie::add(acc, dec_fmul_16(aie::load_v<16>(&g_dec.conv_state[k][i]), aie::load_v<16>(&g_dec.conv_wt[k][i])));
    }
    aie::store_v(g_dec.x_slice + i, dec_fmul_16(acc, aie::load_v<16>(g_dec.conv_c + i)));
    aie::store_v(g_dec.conv_bx + i, bx);
  }
#pragma clang loop unroll(disable)
  for (int k = 0; k < 5; ++k) copy_f32_16(&g_dec.conv_state[k][0], &g_dec.conv_state[k + 1][0], 64);
  copy_f32_16(&g_dec.conv_state[5][0], g_dec.conv_bx, 64);
}

__attribute__((noinline)) static void q4_dec_silu_mul_step(int32_t n_passes) {
#pragma clang loop unroll(disable)
  for (int p = 0; p < n_passes; ++p) {
    aie::vector<float, 16> gv = aie::load_v<16>(&g_dec.ffn_gate[p][0]);
    aie::vector<float, 16> uv = aie::load_v<16>(&g_dec.y_gemm[p][0]);
    aie::store_v(&g_dec.y_gemm[p][0], dec_fmul_16(dec_fmul_16(gv, dec_sigmoid_16(gv)), uv));
  }
  if (n_passes == 3) {
    aie::store_v(&g_dec.y_gemm[3][0], aie::zeros<float, 16>());
  }
}

}  // namespace

extern "C" {

void q4_lfm3_dec_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  if (op == 0) {
    copy_f32_16(g_dec.norm_w, w_pkt + a0 * 64, 64);
    copy_f32_16(g_dec.post_norm_w, w_pkt + 2048 + a0 * 64, 64);
    if (a1) {
      aie::store_v((float *)g_dec.params, aie::load_v<16>(w_pkt + 4096));
    }
  } else if (op == 1) {
    copy_f32_16(g_dec.rope_cs, w_pkt, HEAD_DIM);
    copy_f32_16(g_dec.q_norm_w, w_pkt + 4096, HEAD_DIM);
    copy_f32_16(g_dec.k_norm_w, w_pkt + 4224, HEAD_DIM);
  } else if (op == 2) {
    g_dec.gate_sig = dec_sigmoid_16(aie::broadcast<float, 16>(w_pkt[a0 & 15])).get(0);
  } else {
    const int n = (op == 3) ? 7 : 6;
    float *__restrict dst = (op == 3) ? &g_dec.conv_wt[0][0] : &g_dec.conv_state[0][0];
#pragma clang loop unroll(disable)
    for (int k = 0; k < n; ++k) copy_f32_16(dst + k * 64, w_pkt + k * 256 + a0 * 64, 64);
  }
}

void q4_lfm3_dec_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  if (op == 0) {
    q4_dec_rms_q8_step(x_pkt, a0, a1, a2);
  } else if (op == 1) {
    q4_dec_post_res_step(x_pkt, a0, a1, a2);
  } else if (op == 2) {
    q4_dec_rope_step(x_pkt, a0, a1, a2);
  } else if (op == 4) {
    if (a1 == (a0 >> 3)) copy_f32_16(g_dec.res_acc, x_pkt + (a0 & 7) * 64, 64);
  } else if (op == 5) {
    const int32_t r = a0 & 3;
    if (r < 2) copy_f32_16((float *)g_dec.q_bf16, x_pkt + r * 64, 64);
  } else {
    const int32_t r = a0 & 3;
    if (r >= 2) return;
    if (g_dec.params[9]) {
      if (a1 == 0) {
        copy_f32_16(g_dec.o_acc, x_pkt + r * 64, 64);
        copy_f32_16(g_dec.o_acc + 64, x_pkt + 256 + r * 64, 64);
      } else {
        g_dec.row_m = x_pkt[r * 64 + 0];
        g_dec.row_l = x_pkt[r * 64 + 1];
      }
    } else if (a1 == 0) {
      aie::vector<float, 16> zf = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
      for (int d = 0; d < HEAD_DIM; d += 16) aie::store_v(g_dec.o_acc + d, zf);
    } else {
      g_dec.row_m = -1e30f;
      g_dec.row_l = 0.0f;
    }
  }
}

__attribute__((noinline)) static void drain_q8_y_out(float *__restrict y_out) {
  copy_f32_16(y_out, (const float *)g_dec.xq_out, 16);
  y_out[16] = *(const float *)g_dec.xd_out;
#pragma clang loop unroll(disable)
  for (int i = 17; i < 64; ++i) y_out[i] = 0.0f;
}

void q4_lfm3_dec_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  if (op == 0) {
    drain_q8_y_out(y_out);
  } else if (op == 4) {
    if (arg == 0) {
      float denom = (g_dec.row_l > 1e-35f) ? g_dec.row_l : 1e-35f;
      aie::vector<float, 16> vinv = dec_inv_16(aie::broadcast<float, 16>(denom));
      aie::vector<float, 16> vsc = dec_fmul_16(aie::broadcast<float, 16>(g_dec.gate_sig), vinv);
#pragma clang loop unroll(disable)
      for (int d = 0; d < HEAD_DIM; d += 16) {
        aie::store_v(g_dec.o_acc + d, dec_fmul_16(aie::load_v<16>(g_dec.o_acc + d), vsc));
      }
    }
    quantize_q8_block32_x2(g_dec.o_acc + arg * 64, g_dec.xq_out, g_dec.xd_out);
    drain_q8_y_out(y_out);
  } else if (op == 7 || op == 8) {
    if (op == 8 && arg == 0) {
      aie::vector<float, 16> zf = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
      for (int d = 0; d < HEAD_DIM; d += 16) aie::store_v(g_dec.o_acc + d, zf);
      g_dec.row_m = -1e30f;
      g_dec.row_l = 0.0f;
    }
    if (arg < 2) {
      copy_f32_16(y_out, g_dec.o_acc + arg * 64, 64);
    } else {
      y_out[0] = g_dec.row_m;
      y_out[1] = g_dec.row_l;
    }
  } else if (op == 9 || op == 6) {
    if (op == 6) q4_dec_silu_mul_step(arg);
    else q4_dec_conv_step();
    const float *__restrict q_src = (op == 6) ? &g_dec.y_gemm[0][0] : g_dec.x_slice;
    quantize_q8_block32_x2(q_src, g_dec.xq_out, g_dec.xd_out);
    drain_q8_y_out(y_out);
  } else {
    if (op == 10) q4_dec_silu_mul_step(arg);
    const float *__restrict src =
        (op == 1 || op == 10) ? &g_dec.y_gemm[0][0] :
        (op == 2) ? g_dec.x_slice :
        (op == 3) ? &g_dec.conv_state[arg][0] :
                    (const float *)g_dec.q_bf16;
    copy_f32_16(y_out, src, 64);
  }
}

}  // extern "C"

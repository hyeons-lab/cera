// Fused B=32 Q4 prefill transformer-layer AIE2P kernel implementation.
// Supports:
//   1. q4_rmsnorm_*      : RMSNorm (or raw Q8 quantize when skip_norm=1) -> swizzled Q8/bf16-scale X_SLOT
//   2. q4_prefill_*      : Q4_0 x Q8 prefill mmul<4,16,16> (k_chunk=2048, K<=8192)
//   3. q4_rope_*         : RoPE (Norm or NeoX) + bf16 K/V drain + persistent L1 q_bf16 pack
//   4. q4_flash_attn_*   : Tiled FlashAttention over bo_kv + cur_kv (causal or bidirectional) via mmul<8,8,4>
//   5. q4_silu_*         : SiLU(gate) * up -> swizzled Q8/bf16-scale X_SLOT (xd_mode=1)
//   6. q4_res_*          : Elementwise residual add (a + b)

#include "q4_layer_state.h"

extern "C" {
alignas(64) ScratchUnion g_scratch;
// Persistent L1 state across stages within one layer run
int32_t g_params[16];
// q_bf16 packed as [4 heads][8 d_blk (8 dims each)][8 tokens][8 d] = 2048 bf16 = 4096 B
alignas(64) bfloat16 g_q_bf16[GQA_Q_HEADS * (HEAD_DIM / 8) * B_CORE * 8];
alignas(64) const int32_t lane_idx[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
}

__attribute__((noinline)) aie::vector<float, 16> q4_fmul_16(
    aie::vector<float, 16> a, aie::vector<float, 16> b) {
  return aie::mul(a, b).template to_vector<float>();
}

__attribute__((noinline)) aie::vector<float, 16> q4_inv_16(aie::vector<float, 16> v) {
  return aie::inv(v);
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
  static const float coeffs[5] = {0.0096181291f, 0.0555041087f, 0.2402265069f, 0.6931471805f, 1.0f};
  aie::vector<float, 16> p = aie::broadcast<float, 16>(0.0013333558f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 5; ++i)
    p = aie::add(q4_fmul_16(p, f), aie::broadcast<float, 16>(coeffs[i]));
  return q4_fmul_16(p, ea.template to_vector<float>());
}

__attribute__((noinline)) void q4_scale_f32_16(
    float *__restrict dst, const float *__restrict src, float s, int n) {
  aie::vector<float, 16> sv = aie::broadcast<float, 16>(s);
  for (int i = 0; i < n; i += 16) chess_prepare_for_pipelining {
    aie::store_v(dst + i, q4_fmul_16(aie::load_v<16>(src + i), sv));
  }
}

namespace {

#define fmul_16 q4_fmul_16
#define inv_16 q4_inv_16
#define exp2f_16 q4_exp2f_16
#define scale_f32_16 q4_scale_f32_16

__attribute__((noinline)) static void copy_f32_16(
    float *__restrict dst, const float *__restrict src, int n) {
  for (int i = 0; i < n; i += 16) chess_prepare_for_pipelining {
    aie::store_v(dst + i, aie::load_v<16>(src + i));
  }
}

__attribute__((noinline)) static void quantize_q8_block_8tok(
    const float *__restrict base,
    int stride,
    uint8_t *__restrict xq_blk,
    bfloat16 *__restrict xd_blk) {
  aie::vector<float, 16> inv127 = aie::broadcast<float, 16>(1.0f / 127.0f);
  aie::vector<float, 16> zero = aie::zeros<float, 16>();
  for (int t = 0; t < B_CORE; ++t) {
    const float *__restrict row_p = base + t * stride;
    aie::vector<float, 16> v0 = aie::load_v<16>(row_p);
    aie::vector<float, 16> v1 = aie::load_v<16>(row_p + 16);
    float amax = aie::reduce_max(aie::max(aie::abs(v0), aie::abs(v1)));
    aie::vector<float, 16> dv = fmul_16(aie::broadcast<float, 16>(amax), inv127);
    aie::accum<accfloat, 16> da;
    da.from_vector(dv);
    xd_blk[t] = da.template to_vector<bfloat16>().get(0);
    aie::vector<float, 16> idv = aie::select(zero, inv_16(dv), aie::lt(zero, dv));
    aie::vector<float, 16> s0 = fmul_16(v0, idv);
    aie::vector<float, 16> s1 = fmul_16(v1, idv);
    aie::vector<int8, 32> q32 = aie::to_fixed<int8>(aie::concat(s0, s1), 0);
    int grp = t >> 2;
    int rt = t & 3;
    aie::store_v((int8_t *)(xq_blk + grp * 128 + rt * 16), q32.template extract<16>(0));
    aie::store_v((int8_t *)(xq_blk + grp * 128 + 64 + rt * 16), q32.template extract<16>(1));
  }
}

__attribute__((noinline)) static void drain_xq_xd(
    float *__restrict y_out,
    const uint8_t *xq,
    const bfloat16 *xd,
    int32_t part_idx) {
  const float *__restrict src = (part_idx < 4) ? (const float *)(xq + part_idx * 512) : (const float *)xd;
  int valid_f32 = (part_idx < 4) ? 128 : 32;
  copy_f32_16(y_out, src, valid_f32);
  aie::vector<float, 16> z = aie::zeros<float, 16>();
  for (int i = valid_f32; i < 128; i += 16) aie::store_v(y_out + i, z);
}

__attribute__((noinline)) static void load_a_buf(const float *__restrict x_pkt, int32_t col) {
  for (int t = 0; t < B_CORE; ++t) {
    copy_f32_16(&g_scratch.elem.a_buf[t][0], x_pkt + t * 512 + col * 64, 64);
  }
}

__attribute__((noinline)) static void apply_rope_head(
    const float *__restrict in_h,
    const float *__restrict cs,
    int32_t rope_mode,
    float *__restrict out_h) {
  const float *__restrict c = cs;
  const float *__restrict s = cs + 32;
  if (rope_mode == 0) {
    for (int i = 0; i < 32; i += 16) {
      aie::vector<float, 32> xy = aie::load_v<32>(in_h + i * 2);
      aie::vector<float, 16> x0 = aie::filter_even(xy, 1);
      aie::vector<float, 16> x1 = aie::filter_odd(xy, 1);
      aie::vector<float, 16> cv = aie::load_v<16>(c + i);
      aie::vector<float, 16> sv = aie::load_v<16>(s + i);
      aie::vector<float, 16> o0 = aie::sub(fmul_16(x0, cv), fmul_16(x1, sv));
      aie::vector<float, 16> o1 = aie::add(fmul_16(x0, sv), fmul_16(x1, cv));
      auto z = aie::interleave_zip(o0, o1, 1);
      aie::store_v(out_h + i * 2, aie::concat(z.first, z.second));
    }
  } else {
    for (int i = 0; i < 32; i += 16) {
      aie::vector<float, 16> x0 = aie::load_v<16>(in_h + i);
      aie::vector<float, 16> x1 = aie::load_v<16>(in_h + 32 + i);
      aie::vector<float, 16> cv = aie::load_v<16>(c + i);
      aie::vector<float, 16> sv = aie::load_v<16>(s + i);
      aie::store_v(out_h + i, aie::sub(fmul_16(x0, cv), fmul_16(x1, sv)));
      aie::store_v(out_h + 32 + i, aie::add(fmul_16(x0, sv), fmul_16(x1, cv)));
    }
  }
}

static void q4_rmsnorm_init_impl(const float *__restrict meta_pkt, int32_t col, int32_t load_norm) {
  const int32_t *__restrict ip = (const int32_t *)(meta_pkt + 4096);
  for (int i = 0; i < 16; ++i) g_params[i] = ip[i];
  if (load_norm >= 0) {
    copy_f32_16(&g_scratch.norm.norm_w[0], meta_pkt + load_norm * 2048 + col * 256, 256);
  }
}

static void q4_rmsnorm_q8_pair_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t skip_norm) {
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
    for (int b = 0; b < 8; ++b) {
      quantize_q8_block_8tok(
          &g_scratch.norm.x_slice[0][b * 32],
          256,
          g_scratch.norm.xq_out + b * 256,
          g_scratch.norm.xd_out + b * 8);
    }
  }
}

static void q4_prefill_drain_impl(float *__restrict y_out, int32_t tile_idx) {
  copy_f32_16(y_out, g_scratch.gemm.y_acc + tile_idx * (B_CORE * ROWS), 128);
}

static void q4_rope_load_cs_impl(const float *__restrict x_pkt, int32_t row) {
  copy_f32_16(&g_scratch.rope.rope_cs[0][0], x_pkt + row * 512, B_CORE * 64);
}

static void q4_rope_q_pkt_impl(const float *__restrict w_pkt, int32_t row, int32_t half_pair) {
  int32_t rope_mode = g_params[5];
  float *__restrict rot = &g_scratch.rope.k_buf[0][0];
  alignas(64) bfloat16 rot_b[64];
  for (int t = 0; t < B_CORE; ++t) {
    int tok = row * B_CORE + t;
    const float *__restrict tok_ptr = w_pkt + tok * 128;
    const float *__restrict cs = g_scratch.rope.rope_cs[t];
    for (int h_loc = 0; h_loc < 2; ++h_loc) {
      int h = half_pair * 2 + h_loc;
      apply_rope_head(tok_ptr + h_loc * HEAD_DIM, cs, rope_mode, rot);
      for (int i = 0; i < 64; i += 16) {
        aie::accum<accfloat, 16> a;
        a.from_vector(aie::load_v<16>(rot + i));
        aie::store_v(rot_b + i, a.template to_vector<bfloat16>());
      }
      const uint32_t *__restrict src32 = (const uint32_t *)rot_b;
      for (int db = 0; db < 8; ++db) {
        uint32_t *__restrict dst32 = (uint32_t *)(g_q_bf16 + (h * 8 + db) * 64 + t * 8);
        const uint32_t *__restrict s32 = src32 + db * 4;
        dst32[0] = s32[0];
        dst32[1] = s32[1];
        dst32[2] = s32[2];
        dst32[3] = s32[3];
      }
    }
  }
}

static void q4_rope_kv_pkt_impl(const float *__restrict w_pkt, int32_t row) {
  int32_t rope_mode = g_params[5];
  for (int t = 0; t < B_CORE; ++t) {
    int tok = row * B_CORE + t;
    apply_rope_head(
        w_pkt + tok * HEAD_DIM,
        g_scratch.rope.rope_cs[t],
        rope_mode,
        g_scratch.rope.k_buf[t]);
    copy_f32_16(&g_scratch.rope.v_buf[t][0], w_pkt + 2048 + tok * HEAD_DIM, HEAD_DIM);
  }
}

static void q4_rope_kv_drain_impl(float *__restrict y_out, int32_t part_idx) {
  const float *__restrict base = (part_idx < 2)
      ? &g_scratch.rope.k_buf[(part_idx & 1) * 4][0]
      : &g_scratch.rope.v_buf[(part_idx & 1) * 4][0];
  bfloat16 *__restrict dst = (bfloat16 *)y_out;
  for (int i = 0; i < 4 * HEAD_DIM; i += 16) chess_prepare_for_pipelining {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(base + i));
    aie::store_v(dst + i, a.template to_vector<bfloat16>());
  }
}

static void q4_flash_attn_drain_impl(float *__restrict y_out, int32_t tile_idx) {
  int h = tile_idx >> 2;
  int d0 = (tile_idx & 3) * 16;
  alignas(64) float il_arr[16];
  aie::vector<float, 16> zero_f = aie::zeros<float, 16>();
  aie::vector<float, 16> lv = zero_f;
  for (int t = 0; t < B_CORE; ++t) lv.set(g_scratch.attn.row_l[h][t], t);
  aie::store_v(il_arr, aie::select(zero_f, inv_16(lv), aie::lt(zero_f, lv)));
  for (int t = 0; t < B_CORE; ++t) {
    scale_f32_16(y_out + t * 16, &g_scratch.attn.o_acc[h][t][d0], il_arr[t], 16);
  }
}

static void q4_res_pkt_impl(const float *__restrict x_pkt, int32_t col, int32_t is_b) {
  if (!is_b) {
    load_a_buf(x_pkt, col);
  } else {
    for (int t = 0; t < B_CORE; ++t) {
      const float *__restrict src = x_pkt + t * 512 + col * 64;
      float *__restrict dst = &g_scratch.elem.a_buf[t][0];
      for (int i = 0; i < 64; i += 16) {
        aie::store_v(dst + i, aie::add(aie::load_v<16>(dst + i), aie::load_v<16>(src + i)));
      }
    }
  }
}

static void q4_res_drain_impl(float *__restrict y_out, int32_t tile4) {
  int d0 = tile4 * 16;
  for (int t = 0; t < B_CORE; ++t) {
    copy_f32_16(y_out + t * 16, &g_scratch.elem.a_buf[t][d0], 16);
  }
}

static void q4_silu_round_impl(const float *__restrict x_pkt, int32_t col, int32_t round_j, int32_t is_up) {
  if (!is_up) {
    load_a_buf(x_pkt, col);
  } else {
    aie::vector<float, 16> one = aie::broadcast<float, 16>(1.0f);
    aie::vector<float, 16> nlog2e = aie::broadcast<float, 16>(-LOG2E);
    for (int half = 0; half < 2; ++half) {
      for (int t = 0; t < B_CORE; ++t) {
        const float *__restrict src = x_pkt + t * 512 + col * 64 + half * 32;
        float *__restrict dst = &g_scratch.elem.a_buf[t][half * 32];
        for (int d = 0; d < 32; d += 16) {
          aie::vector<float, 16> x = aie::load_v<16>(dst + d);
          aie::vector<float, 16> u = aie::load_v<16>(src + d);
          aie::vector<float, 16> denom = aie::add(one, exp2f_16(fmul_16(x, nlog2e)));
          aie::vector<float, 16> sx = fmul_16(x, inv_16(denom));
          aie::store_v(dst + d, fmul_16(sx, u));
        }
      }
      int b_idx = round_j * 2 + half;
      quantize_q8_block_8tok(
          &g_scratch.elem.a_buf[0][half * 32],
          64,
          g_scratch.elem.xq_out + b_idx * 256,
          g_scratch.elem.xd_out + b_idx * 8);
    }
  }
}

}  // namespace

extern "C" {

void q4_layer_w_op(
    const float *__restrict w_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1) {
  if (op == 0) {
    q4_rmsnorm_init_impl(w_pkt, a0, a1);
  } else if (op == 1) {
    q4_rope_q_pkt_impl(w_pkt, a0, a1);
  } else {
    q4_rope_kv_pkt_impl(w_pkt, a0);
  }
}

void q4_layer_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  if (op == 0) {
    q4_rmsnorm_q8_pair_impl(x_pkt, a0, a1, a2);
  } else if (op == 1) {
    q4_rope_load_cs_impl(x_pkt, a0);
  } else if (op == 2) {
    q4_res_pkt_impl(x_pkt, a0, a1);
  } else {
    q4_silu_round_impl(x_pkt, a0, a1, a2);
  }
}

void q4_layer_y_op_impl(
    float *__restrict y_out,
    int32_t op,
    int32_t arg) {
  if (op == 0) {
    drain_xq_xd(y_out, g_scratch.norm.xq_out, g_scratch.norm.xd_out, arg);
  } else if (op == 1) {
    q4_prefill_drain_impl(y_out, arg);
  } else if (op == 2) {
    q4_rope_kv_drain_impl(y_out, arg);
  } else if (op == 3) {
    q4_flash_attn_drain_impl(y_out, arg);
  } else if (op == 4) {
    q4_res_drain_impl(y_out, arg);
  } else {
    drain_xq_xd(y_out, g_scratch.elem.xq_out, g_scratch.elem.xd_out, arg);
  }
}

}  // extern "C"

// Unified Fused B=64 LFM2 / LFM2.5 >32K Q4_0 Prefill Core (32 cores: 4 rows x 8 cols, npu2).
// Mode 0 (g_params[6]==0): ShortConv (d_conv=2, param-slot) + SwiGLU FFN (8192).
// Mode 2 (g_params[6]==2): Phase A (QKV + QK-norm + NeoX RoPE -> SLOT_FA_Q + kvo_buf).
// Modes 3/4 (g_params[6]==3/4): FA Partial (blk==0 / blk>=1) -> Drain SLOT_FA_STATE + SLOT_FA_LM.
// Modes 5/6 (g_params[6]==5/6): FA Final (+ cur_kv -> O) with FFN (Mode 5) or without FFN (Mode 6).
// Mode 7 (g_params[6]==7): ShortConv (d_conv=2, param-slot) without FFN.

#include "q4_lfm2_pre_state.h"
void q4_fa_pack_v(const bfloat16 *, int32_t);

#define FN_SZ __attribute__((noinline, minsize))

extern "C" {
ScratchUnion g_scratch;
alignas(64) int32_t g_params[16];
float *g_x_ptr;
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

#define inv_16 q4_inv_16

FN_SZ static void gather4_128(float *__restrict dst, const float *__restrict src) {
#pragma clang loop unroll(disable)
  for (int tl = 0; tl < 4; ++tl) aie::store_v(dst + tl * 16, aie::load_v<16>(src + tl * 128));
}

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
    float *__restrict buf,
    const float *__restrict nw,
    float ss,
    float inv_len,
    float post_scale,
    int n) {
  float mean = q4_fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(inv_len)).get(0);
  float r = ::invsqrt(mean + read_eps());
  float inv_r = q4_fmul_16(aie::broadcast<float, 16>(r), aie::broadcast<float, 16>(post_scale)).get(0);
  q4_elem_mul(buf, buf, nullptr, inv_r, n);
  q4_elem_mul(buf, buf, nw, 0.0f, n);
}
__attribute__((minsize)) static void q4_rmsnorm_q8_pair_impl(
    const float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t) {
#pragma clang loop unroll(disable)
  for (int local_t = 0; local_t < 2; ++local_t) {
    int t = pair_idx * 2 + local_t;
    const float *__restrict row_ptr = x_pkt + local_t * 2048;
    g_scratch.norm.sum_sq[t] = sum_sq_f32(row_ptr, 2048);
    copy_f32_16(&g_scratch.norm.x_slice[t][0], row_ptr + col * 256, 256);
  }
  if (pair_idx == 3) {
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      apply_rmsnorm_n(&g_scratch.norm.x_slice[t][0], &g_scratch.norm.norm_w[0], g_scratch.norm.sum_sq[t], 1.0f / 2048.0f, 1.0f, 256);
    }
#pragma clang loop unroll(disable)
    for (int b = 0; b < 8; b += 2) {
      quantize_q8_2blk_8tok(
          &g_scratch.norm.x_slice[0][b * 32],
          256, 16,
          g_scratch.norm.xq_out + b * 256,
          g_scratch.norm.xq_out + (b + 1) * 256,
          g_scratch.norm.xd_out + b * 8);
    }
  }
}

static_assert(
    __builtin_offsetof(ScratchUnion, attn.row_l) - __builtin_offsetof(ScratchUnion, attn.o_acc) == 8192,
    "o_acc..row_l contiguous");
static_assert(
    __builtin_offsetof(ScratchUnion, attn.q1_bf16) - __builtin_offsetof(ScratchUnion, attn.row_l) == 4096,
    "lm..q1 staging gap");
static_assert(sizeof(((ScratchUnion *)0)->attn.v_t) == 4096, "v_t staging size");

__attribute__((minsize)) static void q4_rope_step_impl(
    float *__restrict x_pkt,
    int32_t col,
    int32_t pair_idx,
    int32_t mode) {
  const int32_t is_q = mode >> 1;
  if (!is_q && !g_params[9]) return;
  const int32_t sh = 1 + is_q;
  if ((col >> sh) == pair_idx) {
    const int32_t sub = mode & 1;
    const float *__restrict s = x_pkt + (col & (is_q ? 3 : 1)) * (is_q ? 1024 : 2048);
    float *__restrict d = is_q
        ? (float *)(sub ? g_scratch.attn.q1_bf16 : g_scratch.attn.q_bf16)
        : (sub ? (float *)&g_scratch.attn.row_l[0][0] : &g_scratch.attn.o_acc[0][0][0]);
    if (mode == 1) copy_f32_16((float *)g_scratch.attn.v_t, s + 1024, 1024);
    copy_f32_16(d, s, mode ? 1024 : 2048);
  }
  if (mode == 1 && pair_idx == 3) {
    copy_f32_16(x_pkt, (const float *)&g_scratch.attn.row_l[0][0], 1024);
    copy_f32_16(x_pkt + 1024, (const float *)g_scratch.attn.v_t, 1024);
  }
}

static void q4_kvq_rope_in_l1(const float *__restrict w_pkt, int32_t row, int32_t is_q, int32_t sub) {
  const float *__restrict cs_base = w_pkt + row * (B_SUB * HEAD_DIM);
  const float *__restrict nw = w_pkt + 2048 + (is_q ? 0 : HEAD_DIM);
  const float pscale = is_q ? ATTN_SCALE_LOG2E : 1.0f;
  const int32_t nh = is_q ? 4 : 1;
  const int32_t db_st = is_q ? (B_SUB * 8) : 8;
  float *__restrict out_sub = g_scratch.gemm.y_acc + sub * 1024;
  bfloat16 *__restrict y_b = (bfloat16 *)out_sub;
#pragma clang loop unroll(disable)
  for (int h = 0; h < nh; ++h) {
    const float *__restrict in_b = g_scratch.gemm.y_acc + (h >> 1) * 2048 + sub * 1024 + (h & 1) * 512;
    bfloat16 *__restrict out_h_base = (h == 0) ? &g_scratch.rope.k_buf[0][0] : (y_b + h * 512);
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      const float *__restrict bt = in_b + t * 16;
      float *__restrict tn = g_scratch.rope.tmp_n;
      gather4_128(tn, bt);
      float ss = sum_sq_f32(tn, HEAD_DIM);
      apply_rmsnorm_n(tn, nw, ss, 1.0f / (float)HEAD_DIM, pscale, HEAD_DIM);
      const float *__restrict cs = cs_base + t * HEAD_DIM;
      bfloat16 *__restrict out_h = out_h_base + t * (is_q ? 8 : HEAD_DIM);
#pragma clang loop unroll(disable)
      for (int i = 0; i < 32; i += 16) {
        aie::vector<bfloat16, 16> x0 = aie::accum<accfloat, 16>(aie::load_v<16>(tn + i)).template to_vector<bfloat16>();
        aie::vector<bfloat16, 16> x1 = aie::accum<accfloat, 16>(aie::load_v<16>(tn + 32 + i)).template to_vector<bfloat16>();
        aie::vector<bfloat16, 16> c0 = aie::accum<accfloat, 16>(aie::load_v<16>(cs + i)).template to_vector<bfloat16>();
        aie::vector<bfloat16, 16> s0 = aie::accum<accfloat, 16>(aie::load_v<16>(cs + 32 + i)).template to_vector<bfloat16>();
        aie::vector<bfloat16, 16> b0 = aie::msc(aie::mul<accfloat>(x0, c0), x1, s0).template to_vector<bfloat16>();
        aie::vector<bfloat16, 16> b1 = aie::mac(aie::mul<accfloat>(x0, s0), x1, c0).template to_vector<bfloat16>();
        int db = i >> 3;
        aie::store_v(out_h + db * db_st, b0.template extract<8>(0));
        aie::store_v(out_h + (db + 1) * db_st, b0.template extract<8>(1));
        aie::store_v(out_h + (db + 4) * db_st, b1.template extract<8>(0));
        aie::store_v(out_h + (db + 5) * db_st, b1.template extract<8>(1));
      }
    }
    if (is_q && h == 0) copy_f32_16(out_sub, (float *)&g_scratch.rope.k_buf[0][0], 256);
  }
  if (is_q) copy_f32_16((float *)(sub ? g_scratch.attn.q1_bf16 : g_scratch.attn.q_bf16), out_sub, 1024);
}

__attribute__((minsize)) static void q4_conv_xd_w_impl(const float *__restrict w_pkt, int32_t row, int32_t stage) {
  if ((row >> 1) != stage) return;
  const bfloat16 *__restrict src_b = (const bfloat16 *)(w_pkt + (row & 1) * 16);
#pragma clang loop unroll(disable)
  for (int i2 = 0; i2 < 32; ++i2) {
    const int kc = i2 >> 4;
    const uint32_t base_off = (((i2 & 12U) << 2) | ((i2 & 3U) << 1)) * 8;
    aie::vector<bfloat16, 32> v = aie::load_v<32>(src_b + i2 * 128);
    aie::store_v(&g_scratch.gemm.conv_xd[kc][0][base_off], v.template extract<16>(0));
    aie::store_v(&g_scratch.gemm.conv_xd[kc][1][base_off], v.template extract<16>(1));
  }
}

__attribute__((minsize)) static void q4_res_w_impl(const float *__restrict w_pkt, int32_t row, int32_t stage) {
  if (row != (stage & 3)) return;
  float *__restrict y_sub = g_scratch.gemm.y_acc + (stage >> 2) * 1024;
#pragma clang loop unroll(disable)
  for (int t = 0; t < B_SUB; ++t) {
    const float *__restrict s = w_pkt + t * 288;
    float *__restrict d = y_sub + t * 16;
#pragma clang loop unroll(disable)
    for (int ti = 0; ti < 8; ++ti, s += 16, d += 128) {
      aie::store_v(d, aie::add(aie::load_v<16>(d), aie::load_v<16>(s)));
      aie::store_v(d + 2048, aie::add(aie::load_v<16>(d + 2048), aie::load_v<16>(s + 128)));
    }
  }
}

__attribute__((minsize)) static void q4_silu_fused_drain_impl(float *__restrict y_out, int32_t arg) {
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
          q4_elem_mul(bxp, bxp, src, 0.0f, 64);
        }
      }
    } else {
      int32_t t_base = (stage - 9) * 8;
      auto bf16 = [](const float *p) { return aie::accum<accfloat, 16>(aie::load_v<16>(p)).template to_vector<bfloat16>(); };
      for (int i = 0; i < 64; i += 16) {
        const float *__restrict wp = &g_scratch.conv.wt[0][i];
        const float *__restrict bp = &g_scratch.conv.bx[t_base][i];
        const float *p_m2 = (t_base >= 2) ? (bp - 128) : &g_scratch.conv.state[D_CONV + t_base - 2][i];
        const float *p_m1 = (t_base >= 1) ? (bp - 64) : &g_scratch.conv.state[D_CONV + t_base - 1][i];
        aie::vector<bfloat16, 16> w0 = bf16(wp), w1 = bf16(wp + 64), w2 = bf16(wp + 128);
        aie::vector<bfloat16, 16> x_m2 = bf16(p_m2), x_m1 = bf16(p_m1);
        for (int t = 0; t < 8; ++t, bp += 64) chess_prepare_for_pipelining chess_loop_count(8) {
          aie::vector<bfloat16, 16> x_0 = bf16(bp);
          aie::accum<accfloat, 16> acc = aie::mul<accfloat>(x_m2, w0);
          acc = aie::mac(acc, x_m1, w1);
          acc = aie::mac(acc, x_0, w2);
          x_m2 = x_m1;
          x_m1 = x_0;
          aie::store_v(
              g_scratch.conv.conv_out + t * 64 + i,
              q4_fast_fmul2_16(acc.template to_vector<float>(), aie::load_v<16>(w_pkt + t * 288 + r_off + i)));
        }
      }
      int32_t q = stage - 9;
      quantize_q8_2blk_8tok(
          g_scratch.conv.conv_out, 64, 16,
          g_scratch.conv.xq_out + q * 512,
          g_scratch.conv.xq_out + q * 512 + 256,
          g_scratch.conv.xd_out + q * 16);
    }
  }
}

__attribute__((minsize)) static void q4_flash_attn_q8_prep(int32_t sub) {
  float *__restrict rlp = &g_scratch.attn.row_l[0][0];
  float *__restrict tmp = (float *)g_scratch.attn.q_bf16;
  const int32_t sp_sub = sub;
  if (!sp_sub) {
#pragma clang loop unroll(disable)
    for (int i = 0; i < GQA_Q_HEADS * 2 * B_SUB; i += 16) {
      aie::store_v(rlp + i, inv_16(aie::load_v<16>(rlp + i)));
    }
  }
  const float *__restrict rh = rlp + sp_sub * 8;
  const float *__restrict oa = sp_sub ? g_x_ptr : &g_scratch.attn.o_acc[0][0][0];
  uint8_t *__restrict xqo = g_scratch.norm.xq_out;
  bfloat16 *__restrict xdo = g_scratch.norm.xd_out;
#pragma clang loop unroll(disable)
  for (int h = 0; h < GQA_Q_HEADS; ++h, oa += 512, rh += 16, xqo += 512, xdo += 16) {
#pragma clang loop unroll(disable)
    for (int t = 0; t < B_SUB; ++t) {
      float *__restrict dst = tmp + t * HEAD_DIM;
#pragma clang loop unroll(disable)
      for (int d8 = 0; d8 < 8; ++d8) {
        aie::store_v(dst + d8 * 8, aie::load_v<8>(oa + d8 * 64 + t * 8));
      }
      q4_elem_mul(dst, dst, nullptr, rh[t], HEAD_DIM);
    }
    quantize_q8_2blk_8tok(tmp, HEAD_DIM, 16, xqo, xqo + 256, xdo);
  }
}

FN_SZ static uint32_t fa_mask_check_slot(
    int32_t cur_sub,
    int32_t row_q0,
    int32_t chunk_gpos0,
    int32_t chunk_valid,
    float *__restrict sc_col,
    int32_t &all_full) {
  const int32_t pos0 = g_params[0], n_valid = g_params[1];
  int32_t q_off = cur_sub * 32 + row_q0;
  int32_t r_v = n_valid - q_off;
  if (r_v > B_SUB) r_v = B_SUB;
  int32_t b_hi = (pos0 + q_off - chunk_gpos0) + 1;
  int32_t vl = b_hi + r_v - 1;
  if (chunk_valid < vl) vl = chunk_valid;
  if (r_v <= 0 || vl <= 0) { all_full = 0; return 0; }
  bool fl = (vl == KV_CHUNK_T) && (b_hi >= KV_CHUNK_T);
  if (!fl || r_v != B_SUB) all_full = 0;
  if (fl) return (1U << r_v) - 1U;
  uint32_t am = 0;
#pragma clang loop unroll(disable)
  for (int t = 0; t < r_v; ++t) {
    int32_t hi = b_hi + t;
    if (vl < hi) hi = vl;
    if (hi <= 0) continue;
    am |= (1U << t);
    if (sc_col) {
      float *__restrict sc = sc_col + t;
#pragma clang loop unroll(disable)
      for (int k = hi; k < KV_CHUNK_T; ++k) {
        sc[k * 16] = -1e30f;
      }
    }
  }
  return am;
}

}  // namespace

extern "C" {
__attribute__((minsize)) static void q4_lfm2_fa_op(
    const float *__restrict kv_pkt,
    int32_t row_arg,
    int32_t iter_idx) {
  const int32_t pos0 = g_params[0];
  const int32_t n_valid = g_params[1];
  const int32_t kv_ch = g_params[2];
  const int32_t n_past = g_params[3];
  const int32_t kv_start_chunk = g_params[8];
  const int32_t row_q0 = (row_arg & 3) * B_SUB;

  if (iter_idx == 0 && !g_params[9]) {
    q4_fill_f32_16(&g_scratch.attn.o_acc[0][0][0], 0.0f, GQA_Q_HEADS * B_SUB * (HEAD_DIM + 2));
    q4_fill_f32_16(&g_scratch.attn.row_m[0][0], -1e30f, GQA_Q_HEADS * 2 * B_SUB);
    q4_fill_f32_16(g_x_ptr, 0.0f, GQA_Q_HEADS * B_SUB * HEAD_DIM);
  }

  const int32_t cur_off = (iter_idx - kv_ch) * 32;
  const int32_t chunk_gpos0 =
      (iter_idx < kv_ch) ? ((kv_start_chunk + iter_idx) * KV_CHUNK_T) : (pos0 + cur_off);
  int32_t chunk_valid = (iter_idx < kv_ch) ? (n_past - chunk_gpos0) : (n_valid - cur_off);
  if (chunk_valid <= 0) return;
  if (chunk_valid > KV_CHUNK_T) chunk_valid = KV_CHUNK_T;

  const bfloat16 *__restrict k_mat = (const bfloat16 *)kv_pkt;
  const bfloat16 *__restrict v_mat = (const bfloat16 *)(kv_pkt + KV_HALF_F32);
  const int32_t kb_max = (chunk_valid + 7) >> 3;

  int32_t all_full = 1;
  uint32_t mask16 = 0;
#pragma clang loop unroll(disable)
  for (int s = 0; s < 2; ++s) {
    mask16 |= (fa_mask_check_slot(s, row_q0, chunk_gpos0, chunk_valid, nullptr, all_full) << (s * 8));
  }
  if (!mask16) return;

  q4_fa_pack_v(v_mat, kb_max);
  float (*__restrict scores1)[16] = (float (*)[16])(g_x_ptr + 3072);
#pragma clang loop unroll(disable)
  for (int pair = 0; pair < 2; ++pair) {
    const int32_t it0 = pair * 2;
    const bfloat16 *__restrict q0a = g_scratch.attn.q_bf16 + (it0 << 9);
    const bfloat16 *__restrict q1a = q0a - 2048;
    q4_fa_qk_quad(q0a, q1a, q0a + 512, q1a + 512, k_mat, kb_max, g_scratch.attn.scores, scores1);
#pragma clang loop unroll(disable)
    for (int sub_it = 0; sub_it < 2; ++sub_it) {
      float (*__restrict sc_ptr)[16] = sub_it ? scores1 : g_scratch.attn.scores;
      if (!all_full) {
#pragma clang loop unroll(disable)
        for (int s = 0; s < 2; ++s) {
          fa_mask_check_slot(s, row_q0, chunk_gpos0, chunk_valid, &sc_ptr[0][s * 8], all_full);
        }
      }
      int32_t h_tag = 4 | (it0 + sub_it);
      q4_fa_softmax_2h(h_tag, mask16, sc_ptr);
      q4_fa_pv_d16(v_mat, kb_max, h_tag);
    }
  }
}

__attribute__((minsize)) void q4_lfm2_pre_w_op(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  if (op >= 16) {
    q4_lfm2_fa_op(w_pkt, op - 16, a1);
  } else if (op == 0 || op == 4) {
    if (op == 0) {
      copy_f32_16((float *)g_params, w_pkt + 2048, 16);
    }
    copy_f32_16(&g_scratch.norm.norm_w[0], w_pkt + a0 * 256, 256);
  } else if (op == 1 || op == 2) {
    q4_kvq_rope_in_l1(w_pkt, a0, op == 2, a1);
  } else if (op == 3) {
    q4_conv_xd_w_impl(w_pkt, a0, a1);
  } else if (op == 6) {
    q4_res_w_impl(w_pkt, a0, a1);
  } else if (op == 7) {
    copy_f32_16(&g_scratch.attn.row_l[0][0], w_pkt + a0 * 128, 128);
  } else {
    q4_conv1d_w_step_impl(w_pkt, a0, a1);
  }
}

__attribute__((minsize)) void q4_lfm2_pre_x_op(
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

__attribute__((minsize)) void q4_lfm2_pre_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  if (op == 9) {
    if (arg < 8) {
      q4_silu_fused_drain_impl(y_out, arg);
      return;
    }
    op = 5;
  }
  if (op == 3 || op == 4) {
    if (arg == 0) q4_flash_attn_q8_prep(op == 4);
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
    int32_t n_v = read_n_valid_clamped();
    int32_t k = arg - 8;
    if (n_v > 32) {
      int32_t idx = (n_v - 32) - D_CONV + k;
      src = (idx < 0) ? &g_scratch.conv.state[D_CONV + idx][0] : &g_scratch.conv.bx[idx][0];
    } else {
      src = &g_scratch.conv.state[k][0];
    }
  } else if (op == 1) {
    int32_t p4 = arg >> 4;
    int32_t tok = arg & 15;
    const float *__restrict base = g_scratch.gemm.y_acc +
        (p4 >> 1) * 2048 + (tok >> 3) * 1024 + (p4 & 1) * 512 + ((tok >> 2) & 1) * 64 + (tok & 3) * 16;
    gather4_128(y_out, base);
    return;
  } else if (op == 2 && arg >= 8) {
    int32_t i = arg - 8;
    const float *src_v = g_scratch.gemm.y_acc + 512 + (i >> 2) * 1024 + (i & 3) * 32;
    bfloat16 *dst = (bfloat16 *)y_out;
#pragma clang loop unroll(disable)
    for (int p = 0; p < 8; ++p) {
      aie::store_v(dst + p * 16, aie::accum<accfloat, 16>(aie::load_v<16>(src_v + (p >> 2) * 16 + (p & 3) * 128)).template to_vector<bfloat16>());
    }
    return;
  } else {
    const float *__restrict base = (op == 6) ? g_x_ptr
        : ((op == 7) ? (const float *)g_scratch.attn.q1_bf16 : &g_scratch.attn.o_acc[0][0][0]);
    src = base + ((op == 7) ? (arg ^ 16) : arg) * 64;
  }
  copy_f32_16(y_out, src, valid_f32);
  if (valid_f32 < 64) q4_fill_f32_16(y_out + 32, 0.0f, 32);
}

}  // extern "C"

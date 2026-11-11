// Fused-layer Q4 kernel for AIE2P (cera/src/backend/xdna/kernels, cera XDNA backend).
//
// One worker runs a whole transformer layer's stages back to back in a single
// NPU run: each stage's f32 activation slot is quantized on the core
// (Q8_0-style, one f32 scale per 32 elements) and the stage's Q4 GEMV tiles
// are consumed from the same stream, so the host never steps between GEMVs
// and the xclbin is never switched inside a token. The tile math is the
// q4gemv kernel's (int8 x int4 mmul, bf16 per-block row scales, split-K
// accumulation); t encodes the tile and chunk as tile * TSTRIDE + chunk.
//
// q4_step modes (t):
//   <= -200: 16-token block online-softmax GQA attention step over k tokens
//            (mode = -t - 200: bit 0 = init running max/sum/acc + split q,
//            bit 1 = current f32 token + normalize by 1/sum)
//   <= -110: drain 32 f32 attention outputs at offset (-t - 110) * 32
//   -100: RMSNorm + quantize a 2048-wide activation + weight pair; if
//         k >= ROPE_CS (wid = k - ROPE_CS), saves worker's 128 residual floats
//         and 64 RoPE cos/sin floats; else (wid = k), saves worker's 128 FFN
//         residual floats and arms on-core gate->SiLU(gate)*up fusion
//   -3: quantize k f32 activations (at offset 0 if k == NORM_N else ACT_CHUNK)
//       and arm residual epilogue from res_x
//   -2: quantize k f32 activations at offset 0
//   >= 0: weight tile chunk; 16 outputs go to gate_local (gate_mode == 1) or
//         out[16 * (t / TSTRIDE) ..] (with SiLU(gate_local) * up when
//         gate_mode == 2, accumulated when (t % TSTRIDE) != 0)

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef KMAX
#define KMAX 8192
#endif
#ifndef ACT_CHUNK
#define ACT_CHUNK 4608
#endif
#define TSTRIDE 8
#define ROWS 16
#define QBLK 32
#define BLOCK_W_BYTES 256
#define BLOCK_S_BYTES (ROWS * 2)
#define NORM_N 2048
#define NORM_EPS 1e-5f
#define ROPE_CS 4096
#define SILU_U_OFF 2304
#define RES_N 128

alignas(64) static int8_t x_q[KMAX];
static float x_d[KMAX / QBLK];
alignas(64) static float res_x[RES_N];
alignas(64) static float saved_rope[64];
alignas(64) static float gate_local[512];
alignas(64) static float q_local[RES_N];
alignas(64) static bfloat16 q_bf16[384];
alignas(64) static float attn_acc[RES_N];
alignas(64) static float attn_ml[64];
alignas(64) static float scratch_f32[16];
alignas(64) static bfloat16 p_bf16[16 * 48];
alignas(64) static const int32_t lane_idx[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
static int32_t res_pos = -1;
static int32_t rope_pos = -1;
static int32_t k_rope_off = 0;
static int32_t attn_rem = 0;
static int32_t gate_mode = 0;
static int32_t gate_idx = 0;

// AIE2P emulates 16-lane f32 multiply in software using nine bf16xbf16 vector
// multiplies (~208 bytes inline per call) and has no callee-saved vector
// registers; keeping fmul_16 and inv_16 out-of-line avoids code duplication
// across the 16 KB program memory budget.
static __attribute__((noinline)) aie::vector<float, 16> fmul_16(
    aie::vector<float, 16> a, aie::vector<float, 16> b) {
  return aie::mul(a, b).template to_vector<float>();
}

static __attribute__((noinline)) aie::vector<float, 16> inv_16(aie::vector<float, 16> v) {
  return aie::inv(v);
}

// Software f32 2^x on 16 lanes (after mlir-aie aie_kernels/aie2p/exp2f_vec.cc,
// Apache-2.0 WITH LLVM-exception): 2^x = p(f) * 2^k, k = round(x), f = x - k
// in [-0.5, 0.5], holding ~2.4e-6 relative error where hardware
// aie::exp2<bfloat16> is 6-49%.
static __attribute__((noinline)) aie::vector<float, 16> exp2f_16(aie::vector<float, 16> x) {
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
    p = aie::add(aie::mul(p, f).template to_vector<float>(), aie::broadcast<float, 16>(coeffs[i]));
  return aie::mul(p, ea.template to_vector<float>()).template to_vector<float>();
}

static __attribute__((noinline)) void copy_f32_16x(float *dst, const float *src, int32_t n) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; i += 16)
    aie::store_v(dst + i, aie::load_v<16>(src + i));
}

static __attribute__((noinline)) void split3_bf16(aie::vector<float, 16> x, bfloat16 *dst) {
#pragma clang loop unroll(disable)
  for (int s = 0; s < 48; s += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(x);
    aie::vector<bfloat16, 16> b = a.template to_vector<bfloat16>();
    aie::store_v(dst + s, b);
    a.from_vector(b);
    x = aie::sub(x, a.template to_vector<float>());
  }
}

static inline aie::accum<accfloat, 16> mac3_bf16(
    aie::accum<accfloat, 16> acc, const bfloat16 *w3, aie::vector<bfloat16, 16> v) {
  return aie::mac(
      aie::mac(aie::mac(acc, aie::load_v<16>(w3), v), aie::load_v<16>(w3 + 16), v),
      aie::load_v<16>(w3 + 32),
      v);
}

// Plain (mode == 0) or RMSNorm (mode == 1) folded into the quantize pass for a
// k-wide f32 activation chunk (k and base are multiples of QBLK = 32).
static __attribute__((noinline)) void prep_quantize_x_f32(
    const float *slot, int32_t k, int32_t base, int32_t mode) {
  res_pos = -1;
  rope_pos = -1;
  gate_mode = 0;
  float norm_s = 1.0f;
  if (mode == 1) {
    aie::accum<accfloat, 16> ssa = aie::zeros<accfloat, 16>();
    for (int i = 0; i < NORM_N; i += 16) chess_prepare_for_pipelining {
        aie::vector<float, 16> v = aie::load_v<16>(slot + i);
        ssa = aie::mac(ssa, v, v);
      }
    float ss = aie::reduce_add(ssa.template to_vector<float>());
    aie::vector<float, 16> mean =
        fmul_16(aie::broadcast<float, 16>(ss), aie::broadcast<float, 16>(1.0f / NORM_N));
    norm_s = aie::invsqrt(aie::add(mean, aie::broadcast<float, 16>(NORM_EPS))).get(0);
  }
  const float *wf = slot + NORM_N;
#pragma clang loop unroll(disable)
  for (int b = 0; b < (k >> 5); ++b) {
    const float *xf = slot + b * QBLK;
    if (mode > 0) {
      aie::vector<float, 16> scale = aie::broadcast<float, 16>(norm_s);
#pragma clang loop unroll(disable)
      for (int j = 0; j < QBLK; j += 16) {
        aie::store_v(
            res_x + j,
            fmul_16(fmul_16(aie::load_v<16>(xf + j), scale), aie::load_v<16>(wf + b * QBLK + j)));
      }
      xf = res_x;
    }
    float amax = aie::reduce_max(aie::max(aie::abs(aie::load_v<16>(xf)), aie::abs(aie::load_v<16>(xf + 16))));
    aie::vector<float, 16> dv = fmul_16(aie::broadcast<float, 16>(amax), aie::broadcast<float, 16>(1.0f / 127.0f));
    x_d[(base >> 5) + b] = dv.get(0);
    aie::vector<float, 16> zero = aie::zeros<float, 16>();
    aie::vector<float, 16> idv = aie::select(zero, inv_16(dv), aie::lt(zero, dv));
    aie::vector<float, 16> s0 = fmul_16(aie::load_v<16>(xf), idv);
    aie::vector<float, 16> s1 = fmul_16(aie::load_v<16>(xf + 16), idv);
    aie::store_v(x_q + base + b * QBLK, aie::to_fixed<int8>(aie::concat(s0, s1), 0));
  }
}

static __attribute__((noinline)) void attn_step(const float *slot, float *out, int32_t k, int32_t mode) {
  if (mode < 0) {
    aie::store_v(out, aie::load_v<32>(attn_acc + (-mode - 1) * 32));
    return;
  }
  int32_t n_tok = k;
  if (!(mode & 2)) {
    n_tok = attn_rem < k ? attn_rem : k;
    attn_rem -= n_tok;
    if (n_tok == 0 && mode == 0)
      return;
  }
#pragma clang loop unroll(disable)
  for (int h = 0; h < 2; ++h) {
    float *qp = q_local + h * 64;
    float *acc = attn_acc + h * 64;
    float *ml = attn_ml + h * 32;
    bfloat16 *qb = q_bf16 + h * 192;
    if (mode & 1) {
      aie::store_v(ml, aie::broadcast<float, 16>(-1e30f));
      aie::store_v(ml + 16, aie::zeros<float, 16>());
#pragma clang loop unroll(disable)
      for (int d = 0; d < 64; d += 32)
        aie::store_v(acc + d, aie::zeros<float, 32>());
#pragma clang loop unroll(disable)
      for (int d = 0; d < 4; ++d) {
        aie::vector<float, 16> qv =
            fmul_16(aie::load_v<16>(qp + d * 16), aie::broadcast<float, 16>(0.18033688011112042f));
        aie::store_v(qp + d * 16, qv);
        split3_bf16(qv, qb + d * 48);
      }
    }
    if (mode & 2) {
      aie::accum<accfloat, 16> sa = aie::zeros<accfloat, 16>();
#pragma clang loop unroll(disable)
      for (int d = 0; d < 64; d += 16)
        sa = aie::mac(sa, aie::load_v<16>(qp + d), aie::load_v<16>(slot + d));
      aie::vector<float, 16> sv = aie::broadcast<float, 16>(aie::reduce_add(sa.template to_vector<float>()));
      aie::vector<float, 16> mv = aie::load_v<16>(ml);
      aie::vector<float, 16> mnv = aie::max(sv, mv);
      aie::vector<float, 16> ev = exp2f_16(aie::sub(aie::interleave_zip(mv, sv, 1).first, mnv));
      aie::vector<float, 16> cv = aie::broadcast<float, 16>(ev.get(0));
      aie::vector<float, 16> pv = aie::broadcast<float, 16>(ev.get(1));
      aie::vector<float, 16> lv = aie::add(pv, fmul_16(aie::load_v<16>(ml + 16), cv));
      aie::vector<float, 16> ilv = inv_16(lv);
      cv = fmul_16(cv, ilv);
      pv = fmul_16(pv, ilv);
#pragma clang loop unroll(disable)
      for (int d = 0; d < 64; d += 16) {
        aie::vector<float, 16> av = fmul_16(aie::load_v<16>(acc + d), cv);
        av = aie::add(av, fmul_16(pv, aie::load_v<16>(slot + SILU_U_OFF + d)));
        aie::store_v(acc + d, av);
      }
    } else {
#pragma clang loop unroll(disable)
      for (int t0 = 0; t0 < n_tok; t0 += 16) {
        int32_t blk = n_tok - t0 < 16 ? n_tok - t0 : 16;
        aie::store_v(scratch_f32, aie::broadcast<float, 16>(-1e30f));
#pragma clang loop unroll(disable)
        for (int j = 0; j < blk; ++j) {
          aie::accum<accfloat, 16> sa = aie::zeros<accfloat, 16>();
          const bfloat16 *kb = (const bfloat16 *)slot + (t0 + j) * 64;
#pragma clang loop unroll(disable)
          for (int d = 0; d < 4; ++d)
            sa = mac3_bf16(sa, qb + d * 48, aie::load_v<16>(kb + d * 16));
          scratch_f32[j] = aie::reduce_add(sa.template to_vector<float>());
        }
        aie::vector<float, 16> sv = aie::load_v<16>(scratch_f32);
        aie::vector<float, 16> bmax = aie::broadcast<float, 16>(aie::reduce_max(sv));
        aie::vector<float, 16> mv = aie::load_v<16>(ml);
        bool rescale = aie::lt(mv, bmax).test(0);
        aie::vector<float, 16> mnv = aie::max(bmax, mv);
        aie::store_v(ml, mnv);
        aie::vector<float, 16> ev = exp2f_16(aie::sub(sv, mnv));
        auto valid = aie::lt(aie::load_v<16>(lane_idx), aie::broadcast<int32, 16>(blk));
        ev = aie::select(aie::zeros<float, 16>(), ev, valid);
        aie::store_v(scratch_f32, ev);
        aie::vector<float, 16> lv = aie::load_v<16>(ml + 16);
        aie::vector<float, 16> cv;
        if (rescale) {
          cv = exp2f_16(aie::sub(mv, mnv));
          lv = fmul_16(lv, cv);
        }
        lv = aie::add(lv, aie::broadcast<float, 16>(aie::reduce_add(ev)));
        aie::store_v(ml + 16, lv);

#pragma clang loop unroll(disable)
        for (int j = 0; j < blk; ++j)
          split3_bf16(aie::broadcast<float, 16>(scratch_f32[j]), p_bf16 + j * 48);
        const bfloat16 *vb0 = (const bfloat16 *)slot + SILU_U_OFF * 2 + t0 * 64;
#pragma clang loop unroll(disable)
        for (int d = 0; d < 64; d += 16) {
          aie::vector<float, 16> av = aie::load_v<16>(acc + d);
          if (rescale)
            av = fmul_16(av, cv);
          aie::accum<accfloat, 16> va;
          va.from_vector(av);
#pragma clang loop unroll(disable)
          for (int j = 0; j < blk; ++j)
            va = mac3_bf16(va, p_bf16 + j * 48, aie::load_v<16>(vb0 + j * 64 + d));
          aie::store_v(acc + d, va.template to_vector<float>());
        }
      }
    }
  }
}

static __attribute__((noinline)) void gemv_tile(const uint8_t *tile, float *out, int32_t k, int32_t t) {
  const int32_t tile_t = t / TSTRIDE;
  const bool accumulate = (t % TSTRIDE) != 0;
  const int32_t blk0 = (t % TSTRIDE) * (k >> 5);
  using MMUL = aie::mmul<4, 16, 16, int8, int4>;
  aie::accum<accfloat, ROWS> facc = aie::zeros<accfloat, ROWS>();
  const uint8_t *p = tile;
  const uint8_t *ps = tile + (k >> 5) * BLOCK_W_BYTES;
  for (int b = 0; b < (k >> 5); ++b) chess_prepare_for_pipelining {
      aie::vector<int8, 64> a0 = aie::zeros<int8, 64>();
      aie::vector<int8, 64> a1 = aie::zeros<int8, 64>();
      a0.insert(0, aie::load_v<16>(x_q + (blk0 + b) * QBLK));
      a1.insert(0, aie::load_v<16>(x_q + (blk0 + b) * QBLK + 16));
      aie::vector<int4, 256> b0 = aie::load_v<256>((const int4 *)p);
      aie::vector<int4, 256> b1 = aie::load_v<256>((const int4 *)(p + 128));
      MMUL m;
      m.mul(a0, b0);
      m.mac(a1, b1);
      aie::vector<int32, 16> r = m.template to_vector<int32>().template extract<16>(0);
      aie::vector<float, 16> rf = aie::to_float(r, 0);
      aie::vector<bfloat16, 16> ws = aie::load_v<16>((const bfloat16 *)ps);
      aie::accum<accfloat, 16> wsa;
      wsa.from_vector(ws);
      aie::vector<float, 16> scale = aie::mul(wsa.template to_vector<float>(), x_d[blk0 + b]).template to_vector<float>();
      facc = aie::mac(facc, rf, scale);
      p += BLOCK_W_BYTES;
      ps += BLOCK_S_BYTES;
    }
  aie::vector<float, ROWS> fv = facc.template to_vector<float>();
  if (gate_mode == 1) {
    aie::store_v(gate_local + gate_idx, fv);
    gate_idx += ROWS;
    if (gate_idx == 512) {
      gate_mode = 2;
      gate_idx = 0;
    }
    return;
  }
  if (gate_mode == 2) {
    aie::vector<float, ROWS> g = aie::load_v<ROWS>(gate_local + gate_idx);
    aie::vector<float, ROWS> e = exp2f_16(fmul_16(g, aie::broadcast<float, ROWS>(-1.4426950408889634f)));
    aie::vector<float, ROWS> sig = inv_16(aie::add(e, aie::broadcast<float, ROWS>(1.0f)));
    fv = fmul_16(fmul_16(g, sig), fv);
    gate_idx += ROWS;
    if (gate_idx == 512)
      gate_mode = 0;
  } else if (accumulate) {
    fv = aie::add(aie::load_v<ROWS>(out + ROWS * tile_t), fv);
  } else if (res_pos >= 0) {
    fv = aie::add(fv, aie::load_v<ROWS>(res_x + res_pos));
    res_pos += ROWS;
  }
  aie::store_v(out + ROWS * tile_t, fv);
  if (!accumulate && rope_pos >= 0 && rope_pos <= RES_N && tile_t == 1) {
    aie::vector<float, 32> v = aie::load_v<32>(out);
    aie::vector<float, 16> x0 = aie::filter_even(v, 1);
    aie::vector<float, 16> x1 = aie::filter_odd(v, 1);
    const float *cs = saved_rope + (rope_pos < RES_N ? (rope_pos & 32) : k_rope_off);
    aie::vector<float, 16> c = aie::load_v<16>(cs);
    aie::vector<float, 16> s = aie::load_v<16>(cs + 16);
    aie::vector<float, 16> y0 = aie::sub(fmul_16(x0, c), fmul_16(x1, s));
    aie::vector<float, 16> y1 = aie::add(fmul_16(x1, c), fmul_16(x0, s));
    auto pr = aie::interleave_zip(y0, y1, 1);
    aie::vector<float, 32> y = aie::concat(pr.first, pr.second);
    aie::store_v(out, y);
    if (rope_pos < RES_N)
      aie::store_v(q_local + rope_pos, y);
    rope_pos += 32;
  }
}

extern "C" {

void q4_step(const uint8_t *slot, float *out, int32_t k, int32_t t) {
  if (t <= -110) {
    attn_step((const float *)slot, out, k, t <= -200 ? -t - 200 : t + 109);
  } else if (t == -100) {
    prep_quantize_x_f32((const float *)slot, NORM_N, 0, 1);
    if (k >= ROPE_CS) {
      int32_t wid = k - ROPE_CS;
      attn_rem = ((const int32_t *)slot)[ROPE_CS + RES_N];
      if (attn_rem >= 0) {
        copy_f32_16x(res_x, (const float *)slot + wid * RES_N, RES_N);
        copy_f32_16x(saved_rope, (const float *)slot + ROPE_CS, 64);
        k_rope_off = (wid & 1) * 32;
        rope_pos = 0;
      }
    } else {
      copy_f32_16x(res_x, (const float *)slot + k * RES_N, RES_N);
      gate_mode = 1;
      gate_idx = 0;
    }
  } else if (t <= -2) {
    prep_quantize_x_f32((const float *)slot, k, t == -3 && k != NORM_N ? ACT_CHUNK : 0, 0);
    if (t == -3)
      res_pos = 0;
  } else {
    gemv_tile(slot, out, k, t);
  }
}

} // extern "C"

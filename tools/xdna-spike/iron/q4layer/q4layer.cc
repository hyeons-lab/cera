// Fused-layer Q4 kernel for AIE2P (tools/xdna-spike, backend plan step 2b).
//
// One worker runs a whole transformer layer's GEMV stages back to back in a
// single NPU run: each stage's f32 activation slot is quantized on the core
// (Q8_0-style, one f32 scale per 32 elements) and the stage's Q4 GEMV tiles
// are consumed from the same stream, so the host never steps between GEMVs
// and the xclbin is never switched inside a token. The tile math is the
// q4gemv kernel's (int8 x int4 mmul, bf16 per-block row scales, split-K
// accumulation); t encodes the tile and chunk as tile * TSTRIDE + chunk.
//
// q4_step modes (t):
//   -1: load a pre-quantized x (int8 + f32 scales) from the slot
//   -2: quantize an f32 activation chunk at x offset 0
//   <= -3: quantize an f32 activation chunk at x offset (-t - 3) * ACT_CHUNK
//   >= 0: weight tile chunk; 16 outputs go to out[16 * (t / TSTRIDE) ..],
//         accumulated when (t % TSTRIDE) != 0

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

alignas(64) static int8_t x_q[KMAX];
static float x_d[KMAX / QBLK];

static void load_x(const uint8_t *slot, int32_t k) {
  const int8_t *xq = (const int8_t *)slot;
  for (int i = 0; i < k; i += 64)
    aie::store_v(x_q + i, aie::load_v<64>(xq + i));
  const float *xd = (const float *)(slot + k);
  for (int b = 0; b < k / QBLK; ++b)
    x_d[b] = xd[b];
}

// Quantizes k f32 activations (k a multiple of 32) into x_q/x_d at element
// offset base, Q8_0 rule: d = amax / 127, q = round(x / d), saturated to
// int8 (the scalar version cost ~150 ns per element; this is vectorized).
// The float->int conversion rounds to nearest, where the host reference
// rounds half away from zero: they can differ by one quantum on exact .5
// ties, which the GEMV tolerance absorbs.
static void quantize_x_f32(const float *xf, int32_t k, int32_t base) {
  for (int b = 0; b < k / QBLK; ++b) {
    aie::vector<float, QBLK> v = aie::load_v<QBLK>(xf + b * QBLK);
    float amax = aie::reduce_max(aie::abs(v));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    x_d[base / QBLK + b] = d;
    aie::vector<float, QBLK> s = aie::mul(v, aie::broadcast<float, QBLK>(id)).template to_vector<float>();
    aie::store_v(x_q + base + b * QBLK, aie::to_fixed<int8>(s, 0));
  }
}

static void gemv_tile(const uint8_t *tile, float *out, int32_t k, int32_t t) {
  const int32_t tile_t = t / TSTRIDE;
  const bool accumulate = (t % TSTRIDE) != 0;
  const int32_t blk0 = (t % TSTRIDE) * (k / QBLK);
  using MMUL = aie::mmul<4, 16, 16, int8, int4>;
  aie::accum<accfloat, ROWS> facc = aie::zeros<accfloat, ROWS>();
  const uint8_t *p = tile;
  const uint8_t *ps = tile + (k / QBLK) * BLOCK_W_BYTES;
  for (int b = 0; b < k / QBLK; ++b) chess_prepare_for_pipelining {
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
  if (accumulate) {
    aie::vector<float, ROWS> prev = aie::load_v<ROWS>(out + ROWS * tile_t);
    aie::store_v(out + ROWS * tile_t, aie::add(prev, facc.template to_vector<float>()));
  } else {
    aie::store_v(out + ROWS * tile_t, facc.template to_vector<float>());
  }
}

extern "C" {

void q4_step(const uint8_t *slot, float *out, int32_t k, int32_t t) {
  if (t == -1)
    load_x(slot, k);
  else if (t == -2)
    quantize_x_f32((const float *)slot, k, 0);
  else if (t <= -3)
    quantize_x_f32((const float *)slot, k, (-t - 3) * ACT_CHUNK);
  else
    gemv_tile(slot, out, k, t);
}

} // extern "C"

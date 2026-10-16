// Q4_0-style GEMV tile kernel for AIE2P (tools/xdna-spike, Phase 0b spike).
//
// y[n] = sum_b (sum_{k in block b} W[n][k] * x[k]) * dw[n][b] * dx[b]
//
// Weights are 4-bit signed (Q4_0's q - 8) with one bf16 scale per 32-weight
// block per row; activations are int8 with one f32 scale per 32-element block
// (Q8_0-style). A tile holds 16 output rows x K columns, packed by the host as
//   K/32 x 256 bytes: per block, two int4 B matrices (16 K x 16 N, row-major,
//                     so B[k][n] = W[n][k]), one per 16-column half
//   K/32 x 32 bytes:  per block, 16 bf16 row scales dw[n][b]
// for 288 bytes per block, 4.5 bits per weight (same as Q4_0). Weights come
// first so every 128-byte int4 load stays 64-byte aligned: AIE vector loads
// silently read from the aligned-down address when given a misaligned pointer.
//
// The int8 x int4 product uses aie::mmul<4, 16, 16, int8, int4> with only row
// 0 of A populated (x); rows 1-3 are zero. That wastes 3/4 of the MACs, which
// is fine: the kernel is bandwidth-bound by a wide margin.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef KMAX
#define KMAX 2048
#endif
#ifndef KCHUNKS
#define KCHUNKS 1
#endif
#define ROWS 16
#define QBLK 32
#define BLOCK_W_BYTES 256
#define BLOCK_S_BYTES (ROWS * 2)

alignas(64) static int8_t x_q[KMAX];
static float x_d[KMAX / QBLK];

// The first element of each worker's stream carries x: K int8 values, then
// K/32 f32 scales. Copy it into core-local memory so the stream buffers stay
// free for weight tiles.
static void load_x(const uint8_t *slot, int32_t k) {
  const int8_t *xq = (const int8_t *)slot;
  for (int i = 0; i < k; i += 64)
    aie::store_v(x_q + i, aie::load_v<64>(xq + i));
  const float *xd = (const float *)(slot + k);
  for (int b = 0; b < k / QBLK; ++b)
    x_d[b] = xd[b];
}

// One chunk of a tile: 16 rows x k columns. t counts chunks across the whole
// output element: tile index t / KCHUNKS, chunk t % KCHUNKS. Chunk 0 stores
// its result; later chunks add theirs into out (split-K accumulation, so a
// tile stream element never has to hold more than 2048 columns).
static void gemv_tile(const uint8_t *tile, float *out, int32_t k, int32_t t) {
  const int32_t tile_t = t / KCHUNKS;
  const bool accumulate = (t % KCHUNKS) != 0;
  const int32_t blk0 = (t % KCHUNKS) * (k / QBLK);  // this chunk's first x block
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

// Single entry point so x_q and x_d live in one object: t < 0 loads x from
// the slot, otherwise the slot is a weight tile whose 16 outputs go to
// out[16 t ..].
void q4_step(const uint8_t *slot, float *out, int32_t k, int32_t t) {
  if (t < 0)
    load_x(slot, k);
  else
    gemv_tile(slot, out, k, t);
}

} // extern "C"

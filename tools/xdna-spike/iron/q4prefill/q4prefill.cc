// Q4_0-style prefill GEMM kernel for AIE2P (tools/xdna-spike, Gate 1).
//
// Computes Y[B_CORE, N_col] = X[B_CORE, K] @ W[N_col, K]^T on each of 32
// cores (4 token-rows x 8 weight-cols).
//
// Weights reuse the exact blocks-then-scales Q4_0 repack from q4gemv/repack.rs:
//   16 rows x 2048 cols = 18,432 bytes per chunk:
//     - 64 blocks x 256 bytes of two 16x16 int4 B matrices (k-major [k][n])
//     - 64 blocks x 32 bytes of 16 bf16 row scales dw[n][b]
//
// Activations X for B_CORE tokens x 2048 cols are packed as:
//   - x_q: 64 blocks x (B_CORE/4) groups x 2 halves x (4 x 16 int8) = B_CORE * 2048 bytes
//   - x_d: 64 blocks x B_CORE bf16 scales = B_CORE * 128 bytes
//   - hdr: 16 int32s (64 bytes): [0]=n_passes, [1]=pass_tiles, [2]=k_chunks
//
// Inner loop uses aie::mmul<4, 16, 16, int8, int4> (no scalar/fp32 scale
// multiply in the hot loop: converts int32 -> bf16 and accumulates via native
// 64-lane bf16 -> accfloat aie::mac).

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef B_CORE
#define B_CORE 8
#endif
#define ROWS 16
#define QBLK 32
#define CHUNK_K 2048
#define NB_CHUNK (CHUNK_K / QBLK)
#define BLOCK_W_BYTES 256
#define XQ_BYTES (B_CORE * CHUNK_K)
#define XD_OFF XQ_BYTES
#define XD_BYTES (B_CORE * NB_CHUNK * 2)
#define MAX_ACC_TILES 8

#if B_CORE != 8
#error "q4prefill.cc currently requires B_CORE == 8 (uses 2x v64accfloat = 128 accfloat regs)"
#endif

alignas(64) static float y_acc[MAX_ACC_TILES * B_CORE * ROWS];

static __attribute__((noinline)) void compute_tile_to(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    float *__restrict dst,
    int32_t add_prev) {
  aie::tile::current().set_rounding(aie::rounding_mode::conv_even);
  using MMUL = aie::mmul<4, 16, 16, int8, int4>;

  // Process tokens 0..3 then tokens 4..7 in separate loops so only one
  // v64accfloat accumulator is live at a time (eliminates stack spills).
  for (int grp = 0; grp < 2; ++grp) {
    aie::accum<accfloat, 64> facc = aie::zeros<accfloat, 64>();
    const int8_t *__restrict px = (const int8_t *)(x_slot + grp * 128);
    const bfloat16 *__restrict pxd = (const bfloat16 *)(x_slot + XD_OFF) + grp * 4;
    const uint8_t *__restrict pw = w_tile;
    const bfloat16 *__restrict pws = (const bfloat16 *)(w_tile + NB_CHUNK * BLOCK_W_BYTES);

    for (int b = 0; b < NB_CHUNK; ++b) chess_prepare_for_pipelining {
        aie::vector<int4, 256> b0 = aie::load_v<256>((const int4 *)pw);
        aie::vector<int4, 256> b1 = aie::load_v<256>((const int4 *)(pw + 128));
        aie::vector<bfloat16, 16> ws = aie::load_v<16>(pws);
        aie::vector<bfloat16, 32> ws32 = aie::concat(ws, ws);
        aie::vector<bfloat16, 64> ws64 = aie::concat(ws32, ws32);

        aie::vector<int8, 64> a0 = aie::load_v<64>(px);
        aie::vector<int8, 64> a1 = aie::load_v<64>(px + 64);
        MMUL m;
        m.mul(a0, b0);
        m.mac(a1, b1);
        aie::vector<int16, 64> rv = m.template to_vector<int16>();
        aie::vector<bfloat16, 64> r_bf16 = aie::to_float<bfloat16>(rv);

        aie::vector<bfloat16, 32> xd01 = aie::concat(
            aie::broadcast<bfloat16, 16>(pxd[0]),
            aie::broadcast<bfloat16, 16>(pxd[1]));
        aie::vector<bfloat16, 32> xd23 = aie::concat(
            aie::broadcast<bfloat16, 16>(pxd[2]),
            aie::broadcast<bfloat16, 16>(pxd[3]));
        aie::vector<bfloat16, 64> xd64 = aie::concat(xd01, xd23);
        aie::accum<accfloat, 64> sa = aie::mul(ws64, xd64);
        aie::vector<bfloat16, 64> sc64 = sa.template to_vector<bfloat16>();
        facc = aie::mac(facc, r_bf16, sc64);

        px += 256;
        pxd += B_CORE;
        pw += BLOCK_W_BYTES;
        pws += ROWS;
      }

    aie::vector<float, 64> fv = facc.template to_vector<float>();
    float *gdst = dst + grp * 64;
    if (add_prev) {
      fv = aie::add(fv, aie::load_v<64>(gdst));
    }
    aie::store_v(gdst, fv);
  }
}

extern "C" {

void q4_prefill_tile(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    float *__restrict out,
    int32_t acc_idx) {
  if (acc_idx < 0) {
    compute_tile_to(x_slot, w_tile, out, 0);
  } else {
    float *acc_ptr = y_acc + acc_idx * (B_CORE * ROWS);
    compute_tile_to(x_slot, w_tile, acc_ptr, 1);
    aie::store_v(out, aie::load_v<64>(acc_ptr));
    aie::store_v(out + 64, aie::load_v<64>(acc_ptr + 64));
  }
}

void q4_prefill_acc_impl(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t tile_idx,
    int32_t add_prev) {
  compute_tile_to(x_slot, w_tile, y_acc + tile_idx * (B_CORE * ROWS), add_prev);
}

} // extern "C"

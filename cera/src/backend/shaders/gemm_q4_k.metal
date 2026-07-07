#include <metal_stdlib>
using namespace metal;

// GEMM for Q4_K weights using simdgroup matrix multiply-accumulate.
//
// Same tiled `kernel_mul_mm` framework as gemm_q4_0 (64×32 output tile, 8×8
// simdgroup ops, 128 threads / 4 simdgroups), specialized to the Q4_K
// super-block: nl = QK_K/16 = 16 sixteen-element tiles per 256-element block.
// `dequantize_q4_K` decodes tile `il` (0..15) to match cera's
// `dequantize_q4_k_m_block` (quant.rs) — so the batched prefill GEMM produces the
// same result as the per-token `gemv_q4_k` path.
//
// Dispatch: (ceil(n/32), ceil(m/64)) TGs × 128 threads. Threadgroup memory: 8 KB.

#define BLOCK_SIZE_M  64
#define BLOCK_SIZE_N  32
#define BLOCK_SIZE_K  32
#define THREAD_MAT_M   4
#define THREAD_MAT_N   2
#define THREAD_PER_ROW 2
#define THREAD_PER_COL 4
#define SG_MAT_SIZE   64
#define Q4K_NL        16

struct GemmParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
    uint _pad;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

// (scale, min) for sub-block `j + k` — port of `decode_q4km_scales` (quant.rs:236),
// the same `get_scale_min_k4` used by llama.cpp's dequantize.
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] >> 6) << 4)),
                          uchar((q[j + 4 + k] >>  4) | ((q[j + k]     >> 6) << 4))};
}

// Decode the 16-element tile `il` (0..15) of a Q4_K super-block into `reg`.
// Tile il covers output elements [16*il, 16*il+16): sub-block
// `2*(il/4) + (il%4)/2`, qs base `32*(il/4) + 16*(il%2)`, low nibble for il%4<2
// else high. Matches `out[64j+l] = d*sc[2j]*(qs[32j+l]&0xF) - dmin*mn[2j]` and the
// high-nibble sibling in `dequantize_q4_k_m_block`.
void dequantize_q4_K(device const block_q4_K * xb, short il, thread half4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    // High nibbles (il>=2) use mask 0xF0 with d/16 so dl*(q&0xF0) == d*sc*(q>>4).
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float mn  = float(xb->dmin);
    const float dl  = d  * float(sc[0]);
    const float ml  = mn * float(sc[1]);
    const ushort mask = il < 2 ? 0x0F : 0xF0;

    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
    reg = (half4x4) reg_f;
}

kernel void gemm_q4_k(
    const device uchar * src0 [[buffer(0)]],
    const device float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant GemmParams & params [[buffer(3)]],
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint m = params.m;
    const uint k = params.k;
    const uint n = params.n;
    const uint x_stride = params.x_stride;
    const uint y_stride = params.y_stride;

    const uint nb = k / 256;
    const uint row_bytes = nb * 144;
    const short nl = Q4K_NL;

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;

    const short n_rows = min((int)m - r0 * BLOCK_SIZE_M, BLOCK_SIZE_M);
    const short n_cols = min((int)n - r1 * BLOCK_SIZE_N, BLOCK_SIZE_N);

    const short thread_row = min((short)(tiitg / THREAD_PER_ROW), (short)(n_rows - 1));
    const short thread_col = min((short)(tiitg / THREAD_PER_COL), (short)(n_cols - 1));

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    device const block_q4_K * x = (device const block_q4_K *)(src0
        + row_bytes * (r0 * BLOCK_SIZE_M + thread_row)) + il / nl;

    device const float * y = src1
        + x_stride * (r1 * BLOCK_SIZE_N + thread_col)
        + (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL));

    for (uint loop_k = 0; loop_k < k; loop_k += BLOCK_SIZE_K) {
        half4x4 temp_a;
        dequantize_q4_K(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8)
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8)
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32 * 8 * (tiitg % THREAD_PER_COL) + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup const float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(8)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M / 8) * SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N / 8) * SG_MAT_SIZE;
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= (int)m && (r1 + 1) * BLOCK_SIZE_N <= (int)n) {
        // Fast path: full tile, no accumulate — direct simdgroup store.
        device float * C = dst
            + (BLOCK_SIZE_M * r0 + 32 * (sgitg & 1))
            + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * y_stride;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * y_stride * (i / 4), y_stride);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem)
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * BLOCK_SIZE_M * (i / 4), BLOCK_SIZE_M);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = dst + (r0 * BLOCK_SIZE_M) + (r1 * BLOCK_SIZE_N + j) * y_stride;
                threadgroup float  * S  = ((threadgroup float *) shmem) + (j * BLOCK_SIZE_M);
                device float4 * D4 = (device float4 *) D;
                threadgroup float4 * S4 = (threadgroup float4 *) S;
                int i = 0;
                for (; i < n_rows / 4; i++) {
                    *(D4 + i) = *(S4 + i);
                }
                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(S + i);
                }
            }
        }
    }
}

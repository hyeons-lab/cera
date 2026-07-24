#include <metal_stdlib>
using namespace metal;

// Q4_1 GEMV: y[m] = dequant(A_q4_1[m, k]) × x[k]  (+= for the _accum variant)
//
// Same 2-rows-per-threadgroup / 32-thread shape as gemv_q4_0.metal. Q4_1 differs
// from Q4_0 in the block format: an explicit minimum `m` sits next to the scale
// `d`, and nibbles are not recentered — `w = d·q + m`, `q ∈ [0,15]`. Because this
// kernel dequantizes and dots in float, the `+m` offset just rides along in the
// per-element weight; the `m·Σx` term falls out of the dot for free (no separate
// correction, unlike the int8 CPU path).
//
// Layout bonus: a Q4_1 block is 20 bytes (2+2+16), a whole number of 4-byte
// words, and rows are `nb·20` bytes — so every block starts word-aligned. That
// removes the arbitrary-byte-offset unpacking gemv_q4_0.metal needs for its
// 18-byte block: word 0 is `d | (m<<16)`, words 1..4 are the 16 nibble bytes.

struct Params {
    uint m;
    uint k;
};

constant constexpr uint ROWS_PER_TG = 2;
constant constexpr uint BLOCK_BYTES = 20;

inline float decode_f16(uint lo, uint hi) {
    uint bits = lo | (hi << 8);
    return float(as_type<half>(ushort(bits)));
}

inline float process_block_q4_1(
    uint row, uint bi, uint row_bytes,
    const device uint* a,
    thread const float* xl
) {
    // Block is word-aligned (row_bytes and 20 are both multiples of 4), so a
    // single word index addresses it: word 0 = d|m, words 1..4 = 16 qs bytes.
    uint word_off = (row * row_bytes + bi * BLOCK_BYTES) / 4;

    uint dm = a[word_off];
    float delta = decode_f16(dm & 0xFFu, (dm >> 8) & 0xFFu);
    float mmin = decode_f16((dm >> 16) & 0xFFu, (dm >> 24) & 0xFFu);

    float sum = 0.0;
    for (uint w_idx = 0; w_idx < 4; w_idx++) {
        uint word = a[word_off + 1 + w_idx];
        uint base = w_idx * 4;
        for (uint b_idx = 0; b_idx < 4; b_idx++) {
            uint byte = (word >> (b_idx * 8)) & 0xFFu;
            float lo = float(byte & 0xFu) * delta + mmin;
            float hi = float((byte >> 4) & 0xFu) * delta + mmin;
            sum += lo * xl[base + b_idx];
            sum += hi * xl[base + b_idx + 16];
        }
    }
    return sum;
}

kernel void gemv_q4_1(
    const device uint* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    uint tgi = tg_id.x + tg_id.y * 65535u;
    uint row_base = tgi * ROWS_PER_TG;
    if (row_base >= m) {
        return;
    }

    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;

    float sums[ROWS_PER_TG] = {0};
    uint bi = tid.x;
    while (bi < nb) {
        uint col_base = bi * 32;
        float xl[32];
        for (uint i = 0; i < 32; i++) {
            xl[i] = x[col_base + i];
        }
        for (uint r = 0; r < ROWS_PER_TG; r++) {
            if (row_base + r < m) {
                sums[r] += process_block_q4_1(row_base + r, bi, row_bytes, a, xl);
            }
        }
        bi += 32;
    }

    for (uint r = 0; r < ROWS_PER_TG; r++) {
        float total = simd_sum(sums[r]);
        if (tid.x == 0 && row_base + r < m) {
            y[row_base + r] = total;
        }
    }
}

kernel void gemv_q4_1_accum(
    const device uint* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    uint tgi = tg_id.x + tg_id.y * 65535u;
    uint row_base = tgi * ROWS_PER_TG;
    if (row_base >= m) {
        return;
    }

    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;

    float sums[ROWS_PER_TG] = {0};
    uint bi = tid.x;
    while (bi < nb) {
        uint col_base = bi * 32;
        float xl[32];
        for (uint i = 0; i < 32; i++) {
            xl[i] = x[col_base + i];
        }
        for (uint r = 0; r < ROWS_PER_TG; r++) {
            if (row_base + r < m) {
                sums[r] += process_block_q4_1(row_base + r, bi, row_bytes, a, xl);
            }
        }
        bi += 32;
    }

    for (uint r = 0; r < ROWS_PER_TG; r++) {
        float total = simd_sum(sums[r]);
        if (tid.x == 0 && row_base + r < m) {
            y[row_base + r] += total;
        }
    }
}

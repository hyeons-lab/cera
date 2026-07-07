#include <metal_stdlib>
using namespace metal;

// Q4_0 GEMV: y[m] = dequant(A_q4_0[m, k]) × x[k]
//
// 32 threads per threadgroup (one simdgroup) processing 2 output rows —
// reuses loaded x values across rows, single simd_sum per row for reduction.
//
// Dispatch: ceil(m / 2) threadgroups, 32 threads each.

struct Params {
    uint m;
    uint k;
};

// Q4_0 block: 2 bytes f16 scale + 16 bytes of packed nibbles = 18 bytes, 32 elements
// NOTE: caller MUST pad the weight buffer by at least 8 trailing bytes.
// process_block() reads up to a[word_off + 5] (24 bytes from block_byte) to handle
// any byte alignment of the 18-byte block within the uint-indexed buffer.
constant constexpr uint ROWS_PER_TG = 2;
constant constexpr uint BLOCK_BYTES = 18;

inline float decode_f16(uint lo, uint hi) {
    uint bits = lo | (hi << 8);
    return float(as_type<half>(ushort(bits)));
}

inline float process_block(
    uint row, uint bi, uint row_bytes,
    const device uint* a,
    thread const float* xl
) {
    uint block_byte = row * row_bytes + bi * BLOCK_BYTES;
    uint word_off = block_byte / 4;
    uint byte_rem = block_byte % 4;

    uint w0 = a[word_off];
    uint w1 = a[word_off + 1];
    uint w2 = a[word_off + 2];
    uint w3 = a[word_off + 3];
    uint w4 = a[word_off + 4];

    uint scale_bits;
    if (byte_rem == 0)      scale_bits = w0 & 0xFFFFu;
    else if (byte_rem == 1) scale_bits = (w0 >> 8) & 0xFFFFu;
    else if (byte_rem == 2) scale_bits = (w0 >> 16) & 0xFFFFu;
    else                    scale_bits = ((w0 >> 24) & 0xFFu) | ((w1 & 0xFFu) << 8);
    float delta = decode_f16(scale_bits & 0xFFu, (scale_bits >> 8) & 0xFFu);

    uint nib_start = byte_rem + 2;
    uint n0, n1, n2, n3;
    if (nib_start == 2) {
        n0 = (w0 >> 16) | (w1 << 16);
        n1 = (w1 >> 16) | (w2 << 16);
        n2 = (w2 >> 16) | (w3 << 16);
        n3 = (w3 >> 16) | (w4 << 16);
    } else if (nib_start == 3) {
        n0 = (w0 >> 24) | (w1 << 8);
        n1 = (w1 >> 24) | (w2 << 8);
        n2 = (w2 >> 24) | (w3 << 8);
        n3 = (w3 >> 24) | (w4 << 8);
    } else if (nib_start == 4) {
        n0 = w1; n1 = w2; n2 = w3; n3 = w4;
    } else {
        n0 = (w1 >> 8) | (w2 << 24);
        n1 = (w2 >> 8) | (w3 << 24);
        n2 = (w3 >> 8) | (w4 << 24);
        n3 = (w4 >> 8) | (a[word_off + 5] << 24);
    }

    float sum = 0.0;
    uint ns[4] = { n0, n1, n2, n3 };
    for (uint w_idx = 0; w_idx < 4; w_idx++) {
        uint word = ns[w_idx];
        uint base = w_idx * 4;
        for (uint b_idx = 0; b_idx < 4; b_idx++) {
            uint byte = (word >> (b_idx * 8)) & 0xFFu;
            float lo = (float(byte & 0xFu) - 8.0) * delta;
            float hi = (float((byte >> 4) & 0xFu) - 8.0) * delta;
            sum += lo * xl[base + b_idx];
            sum += hi * xl[base + b_idx + 16];
        }
    }
    return sum;
}

kernel void gemv_q4_0(
    const device uint* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    // Linearize the 2-D dispatch grid (host uses sz2d(min(groups,65535),
    // ceil(groups/65535))) so m/ROWS_PER_TG > 65535 — e.g. a Qwen/Gemma-vocab
    // Q4_0 logit projection — maps to distinct threadgroups instead of aliasing.
    uint tgi = tg_id.x + tg_id.y * 65535u;
    uint row_base = tgi * ROWS_PER_TG;

    // Rounding groups up to a full second grid row (ceil(groups/65535)*65535)
    // launches surplus threadgroups whose row_base >= m. Bail before the block
    // loop so process_block() never reads weight bytes for an out-of-range row
    // (an out-of-bounds device read → GPU command-buffer fault). Uniform across
    // the threadgroup (row_base derives from tg_id), so simd_sum stays balanced.
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
            // Odd m: the last valid threadgroup has row_base == m-1, so r==1 is
            // out of range — guard the read (writeback is guarded separately).
            if (row_base + r < m) {
                sums[r] += process_block(row_base + r, bi, row_bytes, a, xl);
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

// Fused gate+up GEMV: computes BOTH y_gate = W_gate × x and y_up = W_up × x
// in one dispatch (same x, same m, k shape). Halves GEMV dispatches for FFN.
kernel void gemv_q4_0_gate_up(
    const device uint* a_gate [[buffer(0)]],
    const device uint* a_up [[buffer(1)]],
    const device float* x [[buffer(2)]],
    device float* y_gate [[buffer(3)]],
    device float* y_up [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    uint row = tg_id;
    if (row >= m) return;

    uint nb = k / 32;
    uint row_bytes = nb * BLOCK_BYTES;

    float sum_gate = 0.0;
    float sum_up = 0.0;

    uint bi = tid;
    while (bi < nb) {
        uint col_base = bi * 32;
        float xl[32];
        for (uint i = 0; i < 32; i++) {
            xl[i] = x[col_base + i];
        }
        sum_gate += process_block(row, bi, row_bytes, a_gate, xl);
        sum_up += process_block(row, bi, row_bytes, a_up, xl);
        bi += 32;
    }

    float total_gate = simd_sum(sum_gate);
    float total_up = simd_sum(sum_up);
    if (tid == 0) {
        y_gate[row] = total_gate;
        y_up[row] = total_up;
    }
}

// Same but accumulates (y += W × x) — used to fuse residual adds into the final
// GEMV of each block. Separate kernel avoids a runtime branch in the common path.
kernel void gemv_q4_0_accum(
    const device uint* a [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    uint m = params.m;
    uint k = params.k;
    // Linearize the 2-D dispatch grid (host uses sz2d(min(groups,65535),
    // ceil(groups/65535))) so m/ROWS_PER_TG > 65535 — e.g. a Qwen/Gemma-vocab
    // Q4_0 logit projection — maps to distinct threadgroups instead of aliasing.
    uint tgi = tg_id.x + tg_id.y * 65535u;
    uint row_base = tgi * ROWS_PER_TG;

    // See gemv_q4_0: surplus threadgroups (row_base >= m) must bail before the
    // block loop so process_block() never issues an out-of-bounds device read.
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
            // Odd m: r==1 of the last valid threadgroup is out of range.
            if (row_base + r < m) {
                sums[r] += process_block(row_base + r, bi, row_bytes, a, xl);
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


#include "common_decls.tmpl"

// Q8_0 GEMV: y[m] = dequant(A_q8_0[m, k]) * x[k].
//
// Q8_0 block layout (34 bytes per 32 elements):
//   bytes 0-1:  f16 scale
//   bytes 2-33: 32 signed i8 quants
//
// One workgroup handles 8 rows. Threads stride over Q8 blocks, then
// subgroup-reduce one partial sum per output row.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec2<u32>;

const ROWS_PER_WG: u32 = 8u;
const Q8_BLOCK_BYTES: u32 = 34u;

@compute @workgroup_size(32, 1, 1)
fn gemv_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let k = params.y;
    let tid = lid.x;
    let row_base = get_wid(wid) * ROWS_PER_WG;

    let nb = k / 32u;
    let row_bytes = nb * Q8_BLOCK_BYTES;

    var sums: array<f32, 8>;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        sums[r] = 0.0;
    }

    var bi = tid;
    while bi < nb {
        let col_base = bi * 32u;

        var xl: array<f32, 32>;
        for (var i = 0u; i < 32u; i += 1u) {
            xl[i] = x[col_base + i];
        }

        for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
            let row = row_base + r;
            if row < m {
                sums[r] += process_block_q8_0(row, bi, row_bytes, &xl);
            }
        }

        bi += 32u;
    }

    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        let total = subgroupAdd(sums[r]);
        if tid == 0u && row_base + r < m {
            y[row_base + r] = total;
        }
    }
}

fn get_u32_at(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let shift = (byte_offset & 3u) * 8u;
    let lo = a[word_idx];
    if shift == 0u {
        return lo;
    }
    let hi = a[word_idx + 1u];
    return (lo >> shift) | (hi << (32u - shift));
}

fn process_block_q8_0(row: u32, bi: u32, row_bytes: u32, xl: ptr<function, array<f32, 32>>) -> f32 {
    let block_byte = row * row_bytes + bi * Q8_BLOCK_BYTES;
    let scale_bits = get_u32_at(block_byte) & 0xFFFFu;
    let scale = unpack2x16float(scale_bits).x;

    var sum = 0.0;
    for (var i = 0u; i < 32u; i += 4u) {
        let packed = get_u32_at(block_byte + 2u + i);
        sum += f32(bitcast<i32>((packed & 0x000000FFu) << 24u) >> 24u) * (*xl)[i + 0u];
        sum += f32(bitcast<i32>((packed & 0x0000FF00u) << 16u) >> 24u) * (*xl)[i + 1u];
        sum += f32(bitcast<i32>((packed & 0x00FF0000u) << 8u) >> 24u) * (*xl)[i + 2u];
        sum += f32(bitcast<i32>(packed & 0xFF000000u) >> 24u) * (*xl)[i + 3u];
    }

    return sum * scale;
}

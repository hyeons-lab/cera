// Batched Q8_0 GEMM: output[token, row] = sum_k weight[row, k] * x[token, k].
//
// This mirrors the simple batched Q4_0 kernel shape: one workgroup computes
// 8 output rows for one token. It is intentionally conservative and exists to
// keep Q8_0 prefill on the batched path instead of falling back to per-token
// decode.
//
// Bind group 0:
//   @binding(0) a: array<u32>     (weights, Q8_0 packed: M rows x nb*34 bytes)
//   @binding(1) x: array<f32>     (activations, N tokens x x_stride floats)
//   @binding(2) y: array<f32>     (output,      N tokens x y_stride floats)
//   @binding(3) params: array<u32, 6>
//        (m, k, n, x_stride, y_stride, _pad)
//
// Dispatch: (ceil(m/8), n, 1) workgroups of 32 threads each.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 6>;

const ROWS_PER_WG: u32 = 8u;
const Q8_BLOCK_BYTES: u32 = 34u;

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

fn process_block_q8_0(row: u32, bi: u32, row_bytes: u32, token_base: u32) -> f32 {
    let block_byte = row * row_bytes + bi * Q8_BLOCK_BYTES;
    let scale_bits = get_u32_at(block_byte) & 0xFFFFu;
    let scale = unpack2x16float(scale_bits).x;
    let x_base = token_base + bi * 32u;

    var sum = 0.0;
    for (var i = 0u; i < 32u; i += 4u) {
        let packed = get_u32_at(block_byte + 2u + i);
        sum += f32(bitcast<i32>((packed & 0x000000FFu) << 24u) >> 24u) * x[x_base + i + 0u];
        sum += f32(bitcast<i32>((packed & 0x0000FF00u) << 16u) >> 24u) * x[x_base + i + 1u];
        sum += f32(bitcast<i32>((packed & 0x00FF0000u) << 8u) >> 24u) * x[x_base + i + 2u];
        sum += f32(bitcast<i32>(packed & 0xFF000000u) >> 24u) * x[x_base + i + 3u];
    }

    return sum * scale;
}

@compute @workgroup_size(32, 1, 1)
fn gemm_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params[0];
    let k = params[1];
    let x_stride = params[3];
    let y_stride = params[4];

    let tid = lid.x;
    let token = wid.y;
    let row_base = wid.x * ROWS_PER_WG;
    let nb = k / 32u;
    let row_bytes = nb * Q8_BLOCK_BYTES;
    let token_base = token * x_stride;

    var sums: array<f32, 8>;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        sums[r] = 0.0;
    }

    var bi = tid;
    while bi < nb {
        for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
            let row = row_base + r;
            if row < m {
                sums[r] += process_block_q8_0(row, bi, row_bytes, token_base);
            }
        }
        bi += 32u;
    }

    let y_base = token * y_stride;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        let total = subgroupAdd(sums[r]);
        if tid == 0u && row_base + r < m {
            y[y_base + row_base + r] = total;
        }
    }
}

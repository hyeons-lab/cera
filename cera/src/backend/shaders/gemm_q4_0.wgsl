// Batched Q4_0 GEMM: output[token, row] = Σ_k weight[row, k] * x[token, k].
//
// Each workgroup computes 4 output rows for ONE token. The kernel body
// reuses gemv_q4_0_fast's half-block dot product with workgroup-memory
// reduction across the 32-thread workgroup.
//
// This is a no-frills batched form — for a real GEMM we'd want a tiled
// shared-memory variant that loads each weight block once and reuses it
// across multiple tokens. WGSL doesn't have simdgroup matrix ops (the
// trick the Metal `gemm_q4_0.metal` leans on), so the right tile shape
// for WGSL is a separate exploration; this version exists so PR 2.C-full
// can wire forward_prefill against a working batched signature today.
//
// Bind group 0:
//   @binding(0) a: array<u32>     (weights, Q4_0 packed: M rows × nb*18 bytes)
//   @binding(1) x: array<f32>     (activations, N tokens × x_stride floats)
//   @binding(2) y: array<f32>     (output,      N tokens × y_stride floats)
//   @binding(3) params: array<u32, 6>
//        (m, k, n, x_stride, y_stride, _pad)
//
// Dispatch: (ceil(m/4), n, 1) workgroups of 32 threads each.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 6>;

const NR: u32 = 4u;
const NQ: u32 = 16u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 128>;

fn half_block_dot(blk_byte: u32, sumy: f32, yl: ptr<function, array<f32, 16>>, il: u32) -> f32 {
    let word_off = blk_byte / 4u;
    let byte_rem = blk_byte % 4u;

    var scale_bits: u32;
    if byte_rem == 0u {
        scale_bits = a[word_off] & 0xFFFFu;
    } else if byte_rem == 1u {
        scale_bits = (a[word_off] >> 8u) & 0xFFFFu;
    } else if byte_rem == 2u {
        scale_bits = (a[word_off] >> 16u) & 0xFFFFu;
    } else {
        scale_bits = ((a[word_off] >> 24u) & 0xFFu) | ((a[word_off + 1u] & 0xFFu) << 8u);
    }
    let d = unpack2x16float(scale_bits).x;

    let qs_byte = blk_byte + 2u + il;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var qi = 0u; qi < 8u; qi += 2u) {
        let byte_pos = qs_byte + qi;
        let w_off = byte_pos / 4u;
        let w_rem = byte_pos % 4u;
        var q: u32;
        if w_rem <= 2u {
            q = (a[w_off] >> (w_rem * 8u)) & 0xFFFFu;
        } else {
            q = ((a[w_off] >> 24u) & 0xFFu) | ((a[w_off + 1u] & 0xFFu) << 8u);
        }

        acc0 += (*yl)[qi + 0u] * f32(q & 0x000Fu);
        acc1 += (*yl)[qi + 1u] * f32(q & 0x0F00u);
        acc2 += (*yl)[qi + 8u] * f32(q & 0x00F0u);
        acc3 += (*yl)[qi + 9u] * f32(q & 0xF000u);
    }

    return d * (sumy * -8.0 + acc0 + acc1 + acc2 + acc3);
}

@compute @workgroup_size(32, 1, 1)
fn gemm_q4_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params[0];
    let k = params[1];
    // params[2] (n) is implicit in the dispatch grid.
    let x_stride = params[3];
    let y_stride = params[4];

    let nb = k / 32u;
    let row_bytes = nb * 18u;
    let r0 = wid.x * NR;
    let token = wid.y;
    let tid = lid.x;

    let ix = tid / 2u;
    let il = (tid & 1u) * 8u;

    var sumf: array<f32, 4>;
    sumf[0] = 0.0;
    sumf[1] = 0.0;
    sumf[2] = 0.0;
    sumf[3] = 0.0;

    var yl: array<f32, 16>;
    let x_base = token * x_stride;
    var yb_off: u32 = x_base + ix * 32u + il;

    var ib = ix;
    while ib < nb {
        var sumy0: f32 = 0.0;
        var sumy1: f32 = 0.0;
        for (var i = 0u; i < 8u; i += 2u) {
            sumy0 += x[yb_off + i + 0u] + x[yb_off + i + 1u];
            yl[i + 0u] = x[yb_off + i + 0u];
            yl[i + 1u] = x[yb_off + i + 1u] / 256.0;
            sumy1 += x[yb_off + i + 16u] + x[yb_off + i + 17u];
            yl[i + 8u] = x[yb_off + i + 16u] / 16.0;
            yl[i + 9u] = x[yb_off + i + 17u] / 4096.0;
        }
        let sumy_total = sumy0 + sumy1;

        for (var r = 0u; r < NR; r += 1u) {
            let blk_byte = (r0 + r) * row_bytes + ib * 18u;
            sumf[r] += half_block_dot(blk_byte, sumy_total, &yl, il);
        }

        yb_off += 32u * NQ;
        ib += NQ;
    }

    for (var r = 0u; r < NR; r += 1u) {
        partials[r * WG_SIZE + tid] = sumf[r];
    }
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            for (var r = 0u; r < NR; r += 1u) {
                let idx = r * WG_SIZE + tid;
                partials[idx] += partials[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if tid == 0u {
        let y_base = token * y_stride;
        if r0 + 0u < m { y[y_base + r0 + 0u] = partials[0u * WG_SIZE]; }
        if r0 + 1u < m { y[y_base + r0 + 1u] = partials[1u * WG_SIZE]; }
        if r0 + 2u < m { y[y_base + r0 + 2u] = partials[2u * WG_SIZE]; }
        if r0 + 3u < m { y[y_base + r0 + 3u] = partials[3u * WG_SIZE]; }
    }
}

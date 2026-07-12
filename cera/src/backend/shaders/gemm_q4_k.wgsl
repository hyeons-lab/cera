// Batched Q4_K (Q4_K_M) GEMM: output[token, row] = Σ_k dequant(W_q4k[row, k]) * x[token, k].
//
// Mirrors the conservative batched shape of gemm_q8_0.wgsl (one workgroup
// computes ROWS_PER_WG output rows for one token; the token axis is wid.y).
// WGSL has no portable simdgroup-matrix ops, so this is a batched GEMV-shaped
// kernel, not the Metal simdgroup gemm_q4_k. It keeps Q4_K prefill on the
// batched path instead of the per-token fallback. Dequant matches cera's
// `dequantize_q4_k_m_block` (quant.rs) / gemv_q4_k.wgsl within f32 accumulation
// order (argmax preserved).
//
// Q4_K super-block: 256 elems, 144 bytes: d f16 @0, dmin f16 @2,
//   scales[12] @4 (6-bit packed sub-scales+mins), qs[128] @16.
//   out[64j+l]    = d*sc[2j]   * (qs[32j+l] & 0xF) - dmin*mn[2j]
//   out[64j+l+32] = d*sc[2j+1] * (qs[32j+l] >> 4 ) - dmin*mn[2j+1]
//
// Bind group 0:
//   @binding(0) a: array<u32>  (weights, Q4_K packed: M rows × nb*144 bytes)
//   @binding(1) x: array<f32>  (activations, N tokens × x_stride floats)
//   @binding(2) y: array<f32>  (output,      N tokens × y_stride floats)
//   @binding(3) params: array<u32, 6> (m, k, n, x_stride, y_stride, _pad)
//
// Dispatch: (ceil(m/ROWS_PER_WG), n, 1) workgroups of 32 threads. The row tile
// uses wid.x directly (wid.y carries the token axis); the host asserts
// ceil(m/ROWS_PER_WG) <= 65535. Finalized with a workgroup-memory reduction so
// it stays correct on adapters with narrower subgroups.

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32, 6>;

const QK_K: u32 = 256u;
const Q4K_BYTES: u32 = 144u;
const ROWS_PER_WG: u32 = 8u;
const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 256>; // ROWS_PER_WG * WG_SIZE

fn rb(off: u32) -> u32 {
    return (a[off / 4u] >> ((off % 4u) * 8u)) & 0xFFu;
}

fn rf16(off: u32) -> f32 {
    let lo = rb(off);
    let hi = rb(off + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

// 6-bit sub-block scale / min unpack — port of `decode_q4km_scales` (quant.rs).
fn q4k_sc(sb: u32, sub: u32) -> u32 {
    if sub < 4u {
        return rb(sb + sub) & 63u;
    }
    return (rb(sb + sub + 4u) & 0x0Fu) | ((rb(sb + sub - 4u) >> 6u) << 4u);
}

fn q4k_mn(sb: u32, sub: u32) -> u32 {
    if sub < 4u {
        return rb(sb + sub + 4u) & 63u;
    }
    return (rb(sb + sub + 4u) >> 4u) | ((rb(sb + sub) >> 6u) << 4u);
}

// Full 256-element dot of super-block `bi` of row `row` against the token's
// activations, read directly from `x` at `x_base` (the super-block's base
// offset within the token row). Reading `x` directly instead of staging all 256
// activations into a per-thread array keeps register pressure low (the 8 rows in
// a workgroup re-read the same block, but those reads hit the L2 cache).
fn process_block_q4_k(row: u32, bi: u32, row_bytes: u32, x_base: u32) -> f32 {
    let blk = row * row_bytes + bi * Q4K_BYTES;
    let d = rf16(blk);
    let dmin = rf16(blk + 2u);
    let sb = blk + 4u;
    let qs = blk + 16u;

    var sum = 0.0;
    for (var j = 0u; j < 4u; j += 1u) {
        let scale_lo = d * f32(q4k_sc(sb, 2u * j));
        let min_lo = dmin * f32(q4k_mn(sb, 2u * j));
        let scale_hi = d * f32(q4k_sc(sb, 2u * j + 1u));
        let min_hi = dmin * f32(q4k_mn(sb, 2u * j + 1u));
        let base = 64u * j;
        for (var l = 0u; l < 32u; l += 1u) {
            let qb = rb(qs + 32u * j + l);
            let lo = f32(qb & 0x0Fu);
            let hivar = f32(qb >> 4u);
            sum += (scale_lo * lo - min_lo) * x[x_base + base + l];
            sum += (scale_hi * hivar - min_hi) * x[x_base + base + l + 32u];
        }
    }
    return sum;
}

@compute @workgroup_size(32, 1, 1)
fn gemm_q4_k(
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
    let nb = k / QK_K;
    let row_bytes = nb * Q4K_BYTES;
    let token_base = token * x_stride;

    var sums: array<f32, 8>;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        sums[r] = 0.0;
    }

    var bi = tid;
    while bi < nb {
        let x_base = token_base + bi * QK_K;
        for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
            let row = row_base + r;
            if row < m {
                sums[r] += process_block_q4_k(row, bi, row_bytes, x_base);
            }
        }
        bi += WG_SIZE;
    }

    let y_base = token * y_stride;
    for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
        partials[r * WG_SIZE + tid] = sums[r];
    }
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
                let idx = r * WG_SIZE + tid;
                partials[idx] += partials[idx + stride];
            }
        }
        workgroupBarrier();
    }

    if tid == 0u {
        for (var r = 0u; r < ROWS_PER_WG; r += 1u) {
            if row_base + r < m {
                y[y_base + row_base + r] = partials[r * WG_SIZE];
            }
        }
    }
}

// Kernels for running an LFM2 trunk bidirectionally (the d1 decision models). WGSL twin of
// `bidirectional.metal`.
//
// The causal prefill keeps a rolling window of the last two inputs per channel, which only looks
// backwards. A bidirectional trunk reads the whole sequence at once, so its gated convolution is
// a centred 3-tap filter over rows that are already projected.
//
// The in-projection of a row is (B, C, x) side by side. The convolution reads B*x of three
// neighbouring rows, so the projections of a long sequence are split into B*x and C first, into
// buffers that hold every row (a [rows, 3 * hs] buffer would not fit the binding limit of a
// long prompt on every adapter).
//
// Both kernels cover rows * hs elements one thread each, over a 2-D grid of workgroups of 256
// (a single dimension tops out at 65535 workgroups, 16.7M elements).

const GRID_X: u32 = 32768u;

// ── bidir_split ──────────────────────────────────────────────────────────────
// For the `rows` rows of `proj` (stride 3 * hs): bx[row_off + r] = B * x, c[row_off + r] = C.
//
// Bind group 0:
//   @binding(0) proj:   array<f32>  rows x 3 * hs
//   @binding(1) bx:     array<f32>  (row_off + rows) x hs, read_write
//   @binding(2) c:      array<f32>  (row_off + rows) x hs, read_write
//   @binding(3) params: array<u32, 4> (rows, hs, row_off, 0)

@group(0) @binding(0) var<storage, read> split_proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> split_bx: array<f32>;
@group(0) @binding(2) var<storage, read_write> split_c: array<f32>;
@group(0) @binding(3) var<storage, read> split_params: array<u32, 4>;

@compute @workgroup_size(256, 1, 1)
fn bidir_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = split_params[0];
    let hs = split_params[1];
    let row_off = split_params[2];
    let idx = (wid.y * GRID_X + wid.x) * 256u + lid.x;
    if idx >= rows * hs {
        return;
    }
    let r = idx / hs;
    let ch = idx % hs;
    let base = r * 3u * hs;
    let dst = (row_off + r) * hs + ch;
    split_bx[dst] = split_proj[base + ch] * split_proj[base + 2u * hs + ch];
    split_c[dst] = split_proj[base + hs + ch];
}

// ── bidir_conv_centered ──────────────────────────────────────────────────────
// c[t] = c[t] * (bx[t-1] * w0 + bx[t] * w1 + bx[t+1] * w2), zero padded at both ends, in place.
//
// The window never crosses the end of a media prefix: the last prefix row does not read the
// first text row (a prefix is a function of the media alone), while the first text row does read
// the last prefix row. `taps` is [hs x 3].
//
// Bind group 0:
//   @binding(0) bx:     array<f32>  n x hs
//   @binding(1) taps:   array<f32>  hs x 3
//   @binding(2) c:      array<f32>  n x hs, read_write (C in, the gated output out)
//   @binding(3) params: array<u32, 4> (n, hs, prefix, 0)

@group(0) @binding(0) var<storage, read> conv_bx: array<f32>;
@group(0) @binding(1) var<storage, read> conv_taps: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv_c: array<f32>;
@group(0) @binding(3) var<storage, read> conv_params: array<u32, 4>;

@compute @workgroup_size(256, 1, 1)
fn bidir_conv_centered(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = conv_params[0];
    let hs = conv_params[1];
    let prefix = conv_params[2];
    let idx = (wid.y * GRID_X + wid.x) * 256u + lid.x;
    if idx >= n * hs {
        return;
    }
    let t = idx / hs;
    let ch = idx % hs;
    var prev = 0.0;
    if t > 0u {
        prev = conv_bx[idx - hs];
    }
    var next = 0.0;
    if t + 1u < n && t + 1u != prefix {
        next = conv_bx[idx + hs];
    }
    conv_c[idx] = conv_c[idx] * (prev * conv_taps[ch * 3u] + conv_bx[idx] * conv_taps[ch * 3u + 1u] + next * conv_taps[ch * 3u + 2u]);
}

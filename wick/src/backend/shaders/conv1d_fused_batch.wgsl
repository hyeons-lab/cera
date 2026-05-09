// Batched fused conv1d for N tokens in a single dispatch.
//
// One thread per channel walks all N tokens sequentially because the
// rolling buffer state must advance in token order. This collapses
// N separate dispatch round-trips per conv layer into one.
//
// Per-token work (in register / thread-private state):
//   x = proj[token, 0:hs]
//   c = proj[token, hs:2*hs]
//   b = proj[token, 2*hs:3*hs]
//   bx = x * b
//   conv_out = Σ_k rb[k] * w[k] + bx * w[d_conv]
//   rb shifts left, rb[d_conv - 1] = bx
//   output[token, ch] = c * conv_out
//
// Constraints: kernel_size ≤ 4, d_conv ≤ 3 (LFM2 uses ks=4, d_conv=3).
//
// Bind group 0:
//   @binding(0) proj: array<f32>     (read; n_tokens × proj_stride, packed (x,c,b))
//   @binding(1) rbuffer: array<f32>  (read-write; rolling state, d_conv × hs)
//   @binding(2) weight: array<f32>   (read; hs × kernel_size)
//   @binding(3) output: array<f32>   (read-write; n_tokens × out_stride)
//   @binding(4) params: array<u32, 6>
//        (hidden_size, kernel_size, d_conv, n_tokens, proj_stride, out_stride)
//
// Dispatch: (ceil(hidden_size / 256), 1, 1) workgroups of 256 threads.

@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> rbuffer: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 6>;

@compute @workgroup_size(256, 1, 1)
fn conv1d_fused_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    let hs = params[0];
    let ks = params[1];
    let d_conv = params[2];
    let n_tokens = params[3];
    let proj_stride = params[4];
    let out_stride = params[5];

    if ch >= hs { return; }

    // Pre-load weights for this channel (size ≤ 4).
    var w_local: array<f32, 4>;
    var k = 0u;
    loop {
        if k >= ks || k >= 4u { break; }
        w_local[k] = weight[ch * ks + k];
        k += 1u;
    }

    // Pre-load rolling buffer state for this channel (size ≤ 3).
    var rb: array<f32, 3>;
    k = 0u;
    loop {
        if k >= d_conv || k >= 3u { break; }
        rb[k] = rbuffer[k * hs + ch];
        k += 1u;
    }

    // Walk N tokens sequentially.
    var t = 0u;
    while t < n_tokens {
        let base = t * proj_stride;
        let x_val = proj[base + ch];
        let c_val = proj[base + hs + ch];
        let b_val = proj[base + 2u * hs + ch];
        let bx = x_val * b_val;

        // Conv accumulation over rolling buffer.
        var sum: f32 = 0.0;
        var kk = 0u;
        loop {
            if kk >= d_conv || kk >= 3u { break; }
            sum += rb[kk] * w_local[kk];
            kk += 1u;
        }
        sum += bx * w_local[d_conv];

        // Shift rolling buffer in registers; append bx at the tail.
        if d_conv > 1u {
            var s = 0u;
            loop {
                if s >= d_conv - 1u || s >= 2u { break; }
                rb[s] = rb[s + 1u];
                s += 1u;
            }
        }
        if d_conv > 0u {
            rb[d_conv - 1u] = bx;
        }

        output[t * out_stride + ch] = c_val * sum;
        t += 1u;
    }

    // Write final rolling buffer back to global memory.
    k = 0u;
    loop {
        if k >= d_conv || k >= 3u { break; }
        rbuffer[k * hs + ch] = rb[k];
        k += 1u;
    }
}

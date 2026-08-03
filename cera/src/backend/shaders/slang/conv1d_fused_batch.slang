// Batched fused conv1d: N tokens in a single dispatch. One thread per channel
// walks all N tokens sequentially, because the rolling buffer state has to
// advance in token order. This collapses N dispatch round-trips per conv layer
// into one.
//
// Per-token work, in thread-private state:
//
//   x = proj[token, 0:hs], c = proj[token, hs:2*hs], b = proj[token, 2*hs:3*hs]
//   bx  = x * b
//   sum = sum_k rb[k] * w[k] + bx * w[d_conv]
//   rb shifts left one slot and bx is appended
//   output[token, ch] = c * sum
//
// One Slang source for both GPU backends, ported from the handwritten pair it
// replaces (`shaders/conv1d_fused_batch.wgsl`, `.metal`), preserving both
// binding contracts exactly so no call site changes. The two twins agree on
// element type, bindings and entry name and neither has a reduction or subgroup
// op, so the whole body is written once with no `__target_switch`.
//
// They did not agree on everything, and this body is not a straight copy of
// either:
//   - Loop spelling: the WGSL twin has no C-style `for`, so it writes the same
//     loops with `loop {}` / `while`.
//   - Weight preload bound: `k < ks` (WGSL) vs `k <= d_conv` (Metal). These
//     coincide because every host derives `d_conv = kernel_size - 1`; this body
//     applies both, so a params buffer that broke the invariant cannot read past
//     the channel's weight row.
//   - The `ks > 4 || d_conv > 3` early-out: the WGSL twin has it, the
//     handwritten Metal twin does not (it clamps inside each loop condition but
//     still does an unguarded `w[d_conv]`, so it would read out of bounds on
//     d_conv > 3). This body keeps the guard, which makes the generated MSL
//     strictly safer than the kernel it replaces.
//
//   binding/buffer 0: proj     f32, read, [n_tokens x proj_stride], packed (x, c, b)
//   binding/buffer 1: rbuffer  f32, read-write, rolling state [d_conv x hs]
//   binding/buffer 2: weight   f32, read, conv weights [hs x kernel_size]
//   binding/buffer 3: output   f32, read-write, [n_tokens x out_stride]
//   binding/buffer 4: params   (hs, kernel_size, d_conv, n_tokens, proj_stride,
//                               out_stride)
//
// Constraints: kernel_size <= 4, d_conv <= 3, so `w_local[d_conv]` and
// `rb[d_conv - 1]` cannot run off the fixed-size register arrays. That range is
// enforced at load by `validate_conv_kernel_size` in `model/lfm2.rs`, which
// rejects an out-of-range `lfm2.shortconv.l_cache` rather than letting this
// kernel receive params it cannot honor; the early-out below is the second line
// of defence, not the only one. Shipped LFM2 GGUFs set `l_cache = 3`, giving
// ks=3 and d_conv=2; 4 and 3 are the maxima the arrays are sized for, not the
// usual values.
//
// Every loop over `w_local` / `rb` is written as a literal-trip-count loop with
// a predicated body (`for k in 0..3 { if (k < d_conv) ... }`) plus
// `[ForceUnroll]`, rather than the compound `k < d_conv && k < 3` the two
// handwritten twins use. That shape is load-bearing, not stylistic: the constant
// trip count is what lets the compiler keep these arrays in registers instead of
// spilling them to thread-local memory, and Slang lowers `&&` into a branchy
// short-circuit that defeats the unroll. Writing the bound the handwritten way
// measured 0.72x against the handwritten kernel on an M1 Max; this way measures
// slightly faster than it.
//
// Dispatch: (ceil(hs / 256), 1, 1) workgroups of 256 threads.

[[vk::binding(0)]] StructuredBuffer<float>   proj_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> rbuf     : register(u1);
[[vk::binding(2)]] StructuredBuffer<float>   w_buf    : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> out_buf  : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint>    par_buf  : register(t4);

[shader("compute")]
[numthreads(256, 1, 1)]
void conv1d_fused_batch(uint3 gid : SV_DispatchThreadID) {
    uint ch          = gid.x;
    uint hs          = par_buf[0];
    uint ks          = par_buf[1];
    uint d_conv      = par_buf[2];
    uint n_tokens    = par_buf[3];
    uint proj_stride = par_buf[4];
    uint out_stride  = par_buf[5];

    if (ch >= hs) {
        return;
    }
    // Static guard: `w_local` and `rb` are sized for LFM2's ks=4 / d_conv=3.
    // Bail before `w_local[d_conv]` or `rb[d_conv - 1u]` can run off the end.
    if (ks > 4u || d_conv > 3u) {
        return;
    }

    // Pre-load this channel's conv weights (at most 4) and rolling state (at
    // most 3) once, so the per-token loop stays in thread-private state.
    // `k < ks` as well as `k <= d_conv`: the body reads taps 0..d_conv, and
    // ks == d_conv + 1 for every model that ships (each host derives
    // `d_conv = kernel_size - 1`), but a params buffer with a shorter weight row
    // would otherwise read into the next channel's row. The explicit zero
    // initializer covers that same degenerate case: without it the skipped taps
    // are read back uninitialized in the MSL emission while the WGSL emission
    // zero-fills function-scope `var`, so the two targets would answer
    // differently. Both are defensive only: `ks == d_conv + 1` holds by
    // construction, so no fixture can reach either case (`CONV_SHAPES` cannot
    // express it, and the CPU references index `weight[ch * ks + d_conv]`), and
    // they are pinned by this comment rather than by a test.
    float w_local[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    [ForceUnroll]
    for (uint k = 0u; k < 4u; ++k) {
        if (k <= d_conv && k < ks) {
            w_local[k] = w_buf[ch * ks + k];
        }
    }
    float rb[3] = { 0.0f, 0.0f, 0.0f };
    [ForceUnroll]
    for (uint k = 0u; k < 3u; ++k) {
        if (k < d_conv) {
            rb[k] = rbuf[k * hs + ch];
        }
    }

    for (uint t = 0u; t < n_tokens; ++t) {
        uint base = t * proj_stride;
        // Load all three components up front, as both handwritten twins do, so
        // the three reads issue together instead of stalling on `c` after the
        // conv arithmetic. Deferring the `c` load to the output write measured
        // 0.64x against the handwritten kernel on an M1 Max.
        float x_val = proj_buf[base + ch];
        float c_val = proj_buf[base + hs + ch];
        float b_val = proj_buf[base + 2u * hs + ch];
        float bx = x_val * b_val;

        float sum = 0.0f;
        [ForceUnroll]
        for (uint k = 0u; k < 3u; ++k) {
            if (k < d_conv) {
                sum += rb[k] * w_local[k];
            }
        }
        // The current-input tap and the append below index by `d_conv`
        // directly. Rewriting them as predicated constant-index loops (which
        // removes the last runtime-variable indices into `w_local` / `rb`)
        // measured slightly slower, around 0.97x where direct indexing holds
        // just above 1.0x, so the extra predicated iterations cost more here
        // than the dynamic index does.
        sum += bx * w_local[d_conv];

        // Roll the state left one slot and append bx, all in registers.
        [ForceUnroll]
        for (uint k = 0u; k < 2u; ++k) {
            if (k + 1u < d_conv) {
                rb[k] = rb[k + 1u];
            }
        }
        if (d_conv > 0u) {
            rb[d_conv - 1u] = bx;
        }

        out_buf[t * out_stride + ch] = c_val * sum;
    }

    // Publish the final rolling state back to global memory.
    [ForceUnroll]
    for (uint k = 0u; k < 3u; ++k) {
        if (k < d_conv) {
            rbuf[k * hs + ch] = rb[k];
        }
    }
}

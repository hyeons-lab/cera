#include <metal_stdlib>
using namespace metal;

// Fused: bx = x * b → conv1d(bx, state) → output = c * conv_out
// Combines 3 dispatches into 1 per token per conv layer.
// Dispatch: ceil(hidden_size / 256) threadgroups × 256 threads.
//
// `proj` is the packed conv projection [x | c | b], each `hs` floats wide, the
// same layout `conv1d_fused_batch.metal` and the WGSL twin read. x, b and c used
// to be three separate buffer bindings, but every call site bound the same
// conv-projection buffer three times at offsets 0 / 2*hs / hs, so the split
// carried no information. Reading the three components at in-shader offsets
// instead puts both backends on one binding contract.

struct Params {
    uint hidden_size;
    uint kernel_size;
    uint d_conv;
    uint _pad;
};

kernel void conv1d_fused(
    const device float* proj [[buffer(0)]],    // [x | c | b], hs floats each
    device float* rbuffer [[buffer(1)]],       // rolling conv state
    const device float* weight [[buffer(2)]],  // conv weight
    device float* output [[buffer(3)]],        // result written here
    constant Params& params [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    uint hs = params.hidden_size;
    uint ks = params.kernel_size;
    uint d_conv = params.d_conv;
    if (ch >= hs) return;
    // No kernel-size guard here on purpose. Every index below is dynamic
    // (`rbuffer[k * hs + ch]`, `weight[ch * ks + k]`) with no fixed-size
    // register array to overrun, so this kernel is correct for any d_conv. A
    // `ks > 4 || d_conv > 3` early-out would only turn a correct result into a
    // silently skipped write, leaving the caller's output buffer stale, so
    // neither this kernel nor the WGSL twin carries one. `conv1d_fused_batch` is
    // the kernel that genuinely needs such a bound, for its `w[4]` / `rb[3]`
    // arrays; note the handwritten one there does not carry it, and is unguarded
    // on `w[d_conv]` as a result. `2 <= kernel_size <= 4` is enforced at load by
    // `validate_conv_kernel_size` in `model/lfm2.rs`, so the loop below cannot be
    // driven by a malformed GGUF.

    // Step 1: bx = x * b. All three components are read up front so the
    // loads issue together rather than stalling on `c` after the conv.
    float x_val = proj[ch];
    float c_val = proj[hs + ch];
    float b_val = proj[2u * hs + ch];
    float bx = x_val * b_val;

    // Step 2: conv1d with rolling buffer
    float sum = 0.0f;
    for (uint k_idx = 0u; k_idx < d_conv; k_idx++) {
        sum += rbuffer[k_idx * hs + ch] * weight[ch * ks + k_idx];
    }
    sum += bx * weight[ch * ks + d_conv];

    // Update rolling buffer: shift left, append bx
    if (d_conv > 1u) {
        for (uint k_idx = 0u; k_idx < d_conv - 1u; k_idx++) {
            rbuffer[k_idx * hs + ch] = rbuffer[(k_idx + 1u) * hs + ch];
        }
    }
    if (d_conv > 0u) {
        rbuffer[(d_conv - 1u) * hs + ch] = bx;
    }

    // Step 3: output = c * conv_out
    output[ch] = c_val * sum;
}

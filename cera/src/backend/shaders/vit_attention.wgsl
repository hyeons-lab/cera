// Bidirectional (non-causal, no-RoPE) multi-head self-attention for a ViT.
//
// One workgroup per (query token, head) computes:
//   scores[key] = dot(Q[q,h], K[key,h]) * scale         (all keys)
//   p          = softmax(scores)                          (over all keys)
//   out[q,h,d] = Σ_key p[key] * V[key,h,d]
//
// Q/K/V/out are each [tokens, n_head*head_dim] row-major; head `h` occupies
// columns [h*head_dim, (h+1)*head_dim). No causal mask, no KV cache, no RoPE —
// this matches the CPU ViT attention in `vision_encoder.rs`.
//
// `scores` lives in workgroup memory sized MAX_TOKENS; the Rust caller must
// guarantee tokens ≤ MAX_TOKENS (true for LFM2-VL: image_max_pixels/patch²
// ≤ 1024) and otherwise fall back to CPU.
//
// Dispatch: (tokens, n_head, 1) workgroups of 256 threads.
//
// Bind group 0:
//   @binding(0) q: array<f32>      (read)
//   @binding(1) k: array<f32>      (read)
//   @binding(2) v: array<f32>      (read)
//   @binding(3) out: array<f32>    (read-write)
//   @binding(4) params: vec4<u32>  (tokens, n_head, head_dim, scale_bits)

const MAX_TOKENS: u32 = 1024u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: vec4<u32>;

var<workgroup> scores: array<f32, MAX_TOKENS>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn vit_attention(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let tokens = params.x;
    let n_head = params.y;
    let head_dim = params.z;
    let scale = bitcast<f32>(params.w);
    let q_idx = wid.x;
    let h = wid.y;
    let dim = n_head * head_dim;
    let q_off = q_idx * dim + h * head_dim;

    // Phase A: scores[key] = dot(Q[q,h], K[key,h]) * scale.
    var key = tid;
    while key < tokens {
        let k_off = key * dim + h * head_dim;
        var s: f32 = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            s += q[q_off + d] * k[k_off + d];
        }
        scores[key] = s * scale;
        key += 256u;
    }
    workgroupBarrier();

    // Phase B: max over scores (numerical stability).
    var lmax: f32 = -3.402823e+38;
    key = tid;
    while key < tokens {
        lmax = max(lmax, scores[key]);
        key += 256u;
    }
    red[tid] = lmax;
    workgroupBarrier();
    if tid < 128u { red[tid] = max(red[tid], red[tid + 128u]); }
    workgroupBarrier();
    if tid < 64u { red[tid] = max(red[tid], red[tid + 64u]); }
    workgroupBarrier();
    if tid < 32u { red[tid] = max(red[tid], red[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { red[tid] = max(red[tid], red[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { red[tid] = max(red[tid], red[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { red[tid] = max(red[tid], red[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { red[tid] = max(red[tid], red[tid + 2u]); }
    workgroupBarrier();
    if tid < 1u { red[tid] = max(red[tid], red[tid + 1u]); }
    workgroupBarrier();
    let mx = red[0];
    workgroupBarrier();

    // Phase C: exp(scores - max) in place, and sum.
    var lsum: f32 = 0.0;
    key = tid;
    while key < tokens {
        let e = exp(scores[key] - mx);
        scores[key] = e;
        lsum += e;
        key += 256u;
    }
    red[tid] = lsum;
    workgroupBarrier();
    if tid < 128u { red[tid] += red[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { red[tid] += red[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { red[tid] += red[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { red[tid] += red[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { red[tid] += red[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { red[tid] += red[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { red[tid] += red[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { red[tid] += red[tid + 1u]; }
    workgroupBarrier();
    let inv_sum = 1.0 / red[0];
    workgroupBarrier();

    // Phase D: out[q,h,d] = Σ_key p[key] * V[key,h,d].
    var d = tid;
    while d < head_dim {
        var acc: f32 = 0.0;
        for (var key2 = 0u; key2 < tokens; key2 = key2 + 1u) {
            acc += scores[key2] * v[key2 * dim + h * head_dim + d];
        }
        out[q_off + d] = acc * inv_sum;
        d += 256u;
    }
}

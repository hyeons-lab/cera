// Batched per-head Q/K rmsnorm + RoPE for N tokens.
//
// One workgroup per (token, head) pair. For workgroup `tg`:
//   token = tg / heads_per_token
//   head  = tg % heads_per_token   where heads_per_token = n_heads + n_kv_heads
//   pos   = start_pos + token
//   if head < n_heads      → operate on Q[token, head]   with `q_norm_w`
//   else                   → operate on K[token, head - n_heads] with `k_norm_w`
//
// Each chosen head buffer (`head_dim` floats) is rmsnorm'd in place using
// the appropriate weight, then RoPE is applied with the per-token `pos`.
//
// Dispatch: (n_tokens * (n_heads + n_kv_heads), 1, 1) workgroups of 256 threads.
// `head_dim` must be ≤ 256 (existing LFM2 configs use 64 or 128); larger
// would require splitting the workgroup-shared scratch.
//
// Bind group 0:
//   @binding(0) q_batch: array<f32>    (read-write, n_tokens × q_stride floats)
//   @binding(1) k_batch: array<f32>    (read-write, n_tokens × k_stride floats)
//   @binding(2) q_norm_w: array<f32>   (read, head_dim floats)
//   @binding(3) k_norm_w: array<f32>   (read, head_dim floats)
//   @binding(4) params: array<u32, 10> (start_pos, n_tokens, n_heads, n_kv_heads,
//                                       head_dim, eps_bits, freq_base_bits,
//                                       rope_type, q_stride, k_stride)

@group(0) @binding(0) var<storage, read_write> q_batch: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_batch: array<f32>;
@group(0) @binding(2) var<storage, read> q_norm_w: array<f32>;
@group(0) @binding(3) var<storage, read> k_norm_w: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 10>;

var<workgroup> shared_sum: array<f32, 256>;

fn workgroup_sum_reduce(tid: u32) {
    if tid < 128u { shared_sum[tid] += shared_sum[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { shared_sum[tid] += shared_sum[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_sum[tid] += shared_sum[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_sum[tid] += shared_sum[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_sum[tid] += shared_sum[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_sum[tid] += shared_sum[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_sum[tid] += shared_sum[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_sum[tid] += shared_sum[tid + 1u]; }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn qk_norm_rope_batch(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let start_pos = params[0];
    // params[1] (n_tokens) is implicit in the dispatch grid.
    let n_heads = params[2];
    let n_kv_heads = params[3];
    let head_dim = params[4];
    let eps = bitcast<f32>(params[5]);
    let freq_base = bitcast<f32>(params[6]);
    let rope_type = params[7];
    let q_stride = params[8];
    let k_stride = params[9];

    let heads_per_token = n_heads + n_kv_heads;
    let token = wid.x / heads_per_token;
    let head = wid.x % heads_per_token;
    let pos = start_pos + token;
    let half_dim = head_dim / 2u;

    // Pick which buffer + weight + base offset this workgroup operates on.
    let is_q = head < n_heads;
    var base: u32;
    if is_q {
        base = token * q_stride + head * head_dim;
    } else {
        let kh = head - n_heads;
        base = token * k_stride + kh * head_dim;
    }

    // ─── Phase 1: per-head rmsnorm in place ────────────────────────────────
    // Sum of squares.
    var partial: f32 = 0.0;
    var i = tid;
    while i < head_dim {
        let v = select(k_batch[base + i], q_batch[base + i], is_q);
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();
    workgroup_sum_reduce(tid);
    let inv_rms = 1.0 / sqrt(shared_sum[0] / f32(head_dim) + eps);

    // Normalize + scale by per-element weight, write back in place.
    i = tid;
    while i < head_dim {
        let w = select(k_norm_w[i], q_norm_w[i], is_q);
        if is_q {
            q_batch[base + i] = q_batch[base + i] * inv_rms * w;
        } else {
            k_batch[base + i] = k_batch[base + i] * inv_rms * w;
        }
        i += 256u;
    }
    workgroupBarrier();

    // ─── Phase 2: RoPE — pairs of (cos, sin) rotations ─────────────────────
    // theta_d = pos * freq_base^(-2d / head_dim). Compute once per d via pow.
    var d = tid;
    while d < half_dim {
        let theta = f32(pos) * pow(freq_base, -2.0 * f32(d) / f32(head_dim));
        let cos_a = cos(theta);
        let sin_a = sin(theta);
        var i0: u32;
        var i1: u32;
        if rope_type == 0u {
            i0 = base + d;
            i1 = base + d + half_dim;
        } else {
            i0 = base + 2u * d;
            i1 = base + 2u * d + 1u;
        }
        if is_q {
            let x0 = q_batch[i0];
            let x1 = q_batch[i1];
            q_batch[i0] = x0 * cos_a - x1 * sin_a;
            q_batch[i1] = x0 * sin_a + x1 * cos_a;
        } else {
            let x0 = k_batch[i0];
            let x1 = k_batch[i1];
            k_batch[i0] = x0 * cos_a - x1 * sin_a;
            k_batch[i1] = x0 * sin_a + x1 * cos_a;
        }
        d += 256u;
    }
}

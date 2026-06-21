// Query-tiled bidirectional (non-causal, no-RoPE) multi-head self-attention
// for a ViT — a flash-attention-style rewrite of `vit_attention.wgsl`.
//
// One workgroup handles Q_TILE consecutive query tokens for one head, with one
// thread per query. K/V are streamed in K_TILE-row blocks into shared memory
// and reused across all Q_TILE queries (vs the scalar kernel, which re-reads
// all of K/V once per query). Softmax is computed online (running max/sum +
// rescale) so a single pass over the keys suffices and no per-query `scores`
// scratch is needed.
//
// Q/K/V/out are each [tokens, n_head*head_dim] row-major; head `h` occupies
// columns [h*head_dim, (h+1)*head_dim). No causal mask, no KV cache, no RoPE —
// matches the CPU ViT attention in `vision_encoder.rs`.
//
// Constraint: head_dim ≤ MAX_HEAD_DIM (64). The Rust caller falls back to the
// scalar `vit_attention` kernel when head_dim exceeds this (the shared K/V
// tiles and per-thread Q/accumulator arrays are sized for MAX_HEAD_DIM).
//
// Dispatch: (ceil(tokens / Q_TILE), n_head, 1) workgroups of Q_TILE threads.
//
// Bind group 0 (identical to vit_attention.wgsl):
//   @binding(0) q: array<f32>      (read)
//   @binding(1) k: array<f32>      (read)
//   @binding(2) v: array<f32>      (read)
//   @binding(3) out: array<f32>    (read-write)
//   @binding(4) params: vec4<u32>  (tokens, n_head, head_dim, scale_bits)

const Q_TILE: u32 = 256u;
const K_TILE: u32 = 32u;
const MAX_HEAD_DIM: u32 = 64u;
const NEG_INF: f32 = -3.402823e+38;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: vec4<u32>;

var<workgroup> k_tile: array<f32, K_TILE * MAX_HEAD_DIM>;
var<workgroup> v_tile: array<f32, K_TILE * MAX_HEAD_DIM>;

@compute @workgroup_size(Q_TILE, 1, 1)
fn vit_attention_tiled(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let tokens = params.x;
    let n_head = params.y;
    let head_dim = params.z;
    let scale = bitcast<f32>(params.w);
    let h = wid.y;
    let dim = n_head * head_dim;

    let q_idx = wid.x * Q_TILE + tid;
    let valid = q_idx < tokens;

    // This thread's Q row in registers (reused across every key block).
    var q_reg: array<f32, MAX_HEAD_DIM>;
    if valid {
        let q_off = q_idx * dim + h * head_dim;
        for (var d = 0u; d < head_dim; d += 1u) {
            q_reg[d] = q[q_off + d];
        }
    }

    // Online-softmax running state + output accumulator (all in registers).
    var m: f32 = NEG_INF;
    var l: f32 = 0.0;
    var acc: array<f32, MAX_HEAD_DIM>;
    for (var d = 0u; d < head_dim; d += 1u) {
        acc[d] = 0.0;
    }

    for (var k_base = 0u; k_base < tokens; k_base += K_TILE) {
        let k_count = min(K_TILE, tokens - k_base);

        // Cooperative load: all Q_TILE threads stage this K/V block into shared
        // memory (head `h`'s columns only).
        for (var idx = tid; idx < k_count * head_dim; idx += Q_TILE) {
            let i = idx / head_dim;
            let d = idx % head_dim;
            let src = (k_base + i) * dim + h * head_dim + d;
            k_tile[idx] = k[src];
            v_tile[idx] = v[src];
        }
        workgroupBarrier();

        if valid {
            for (var i = 0u; i < k_count; i += 1u) {
                let base = i * head_dim;
                var s: f32 = 0.0;
                for (var d = 0u; d < head_dim; d += 1u) {
                    s += q_reg[d] * k_tile[base + d];
                }
                s *= scale;

                // Online softmax: fold key `i` into the running max/sum/acc.
                let m_new = max(m, s);
                let corr = exp(m - m_new);
                let p = exp(s - m_new);
                l = l * corr + p;
                for (var d = 0u; d < head_dim; d += 1u) {
                    acc[d] = acc[d] * corr + p * v_tile[base + d];
                }
                m = m_new;
            }
        }
        workgroupBarrier(); // guard k_tile/v_tile before the next block overwrites them
    }

    if valid {
        let inv = 1.0 / l;
        let o_off = q_idx * dim + h * head_dim;
        for (var d = 0u; d < head_dim; d += 1u) {
            out[o_off + d] = acc[d] * inv;
        }
    }
}

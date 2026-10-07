// Split-K FlashAttention, phase 2 of 2: combine the partial results of `flash_attention_split.wgsl`.
//
// For each head, with per-split (max m_s, sum l_s, unnormalized acc_s):
//   M = max_s m_s;  out = (sum_s exp(m_s - M) acc_s) / (sum_s exp(m_s - M) l_s)
// A split that saw no keys has m_s = -inf-ish and l_s = 0, so it contributes nothing. This is the exact
// online-softmax merge, so the result equals the single-pass kernel's up to roundoff.
//
// Grid (n_heads, 1, 1); 64 threads, one per output dimension (head_dim == 64).
//
// Bind group 0:
//   @binding(0) part: array<f32>     (n_heads x SPLITS x 66, read)
//   @binding(1) out: array<f32>      (all heads concatenated, read-write)
//   @binding(2) params: array<u32,8> (as in the split kernel)

@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32, 8>;

const SPLITS: u32 = 16u;
const HEAD_DIM: u32 = 64u;
const REC: u32 = 66u;

@compute @workgroup_size(64, 1, 1)
fn flash_attention_merge(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let d = lid.x;
    if params[4] == 0u {
        out[head * HEAD_DIM + d] = 0.0;
        return;
    }
    var m = -3.402823e+38;
    for (var s = 0u; s < SPLITS; s += 1u) {
        m = max(m, part[(head * SPLITS + s) * REC]);
    }
    var num = 0.0;
    var den = 0.0;
    for (var s = 0u; s < SPLITS; s += 1u) {
        let rec = (head * SPLITS + s) * REC;
        let w = exp(part[rec] - m);
        num += w * part[rec + 2u + d];
        den += w * part[rec + 1u];
    }
    out[head * HEAD_DIM + d] = num / den;
}

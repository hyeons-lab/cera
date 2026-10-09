// Split-K FlashAttention for one query vector, phase 1 of 2: the decode path at long context.
//
// `flash_attention.wgsl` runs one workgroup per head, so a 16-head model launches 16 workgroups and each
// walks the whole KV cache serially, with the value sum done by only head_dim of its 256 threads. At
// 1,800 tokens of context that left most of an Adreno 830 idle (attn_core took 4.7 ms per token against
// 0.9 ms at 210 tokens). This kernel splits the keys across `SPLITS` workgroups per head and uses every
// thread for the value sum; `flash_attention_merge.wgsl` combines the partial results.
//
// Grid (n_heads, SPLITS, 1); 256 threads. Split `s` takes tiles `s, s + SPLITS, s + 2 SPLITS, ...` of
// TILE = 256 timesteps (one per thread for the scores), keeping an online softmax across its tiles, so
// any context length works with the one fixed grid and a split with no tiles writes an empty partial
// (max = -inf, sum = 0, acc = 0). The sequence length is read from `params` at run time, which is what
// lets the decode command buffers be built once.
//
// Value sum: thread (part, dp) with part = tid / 32 and dp = tid % 32 owns the dimension pair
// (2 dp, 2 dp + 1) and the 32 timesteps part * 32 .. part * 32 + 31 of each tile, one u32 load each;
// the 8 parts are added at the end. Requires head_dim == 64 (checked host-side).
//
// Output `part[(head * SPLITS + split) * 66 + {0: max, 1: sum, 2..66: acc}]` (f32), unnormalized.
//
// Bind group 0:
//   @binding(0) q: array<f32>        (all heads concatenated, read)
//   @binding(1) k_cache: array<vec4<u32>> (seq_len x kv_dim packed halves, read; 8 halves per element)
//   @binding(2) v_cache: array<u32>  (seq_len x kv_dim packed halves, read)
//   @binding(3) part: array<f32>     (n_heads x SPLITS x 66, read-write)
//   @binding(4) params: array<u32,8> (n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale_bits, _, _)
//
// Barriers sit at kernel scope and the reductions are inlined, as in `flash_attention.wgsl`.

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 8>;

const TILE: u32 = 256u;
const SPLITS: u32 = 16u;
const HEAD_DIM: u32 = 64u;
const REC: u32 = 66u;
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> q_shared: array<f32, 64>;
var<workgroup> tile_scores: array<f32, 256>;
var<workgroup> pdot: array<f32, 1024>;     // [256 timesteps][4 lanes] partial dot products
var<workgroup> red: array<f32, 256>;
var<workgroup> part_acc: array<f32, 512>; // [8 parts][64 dims]
var<workgroup> st: array<f32, 4>;         // [0]=running max, [1]=running sum, [2]=new max, [3]=correction

@compute @workgroup_size(256, 1, 1)
fn flash_attention_split(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let split = wid.y;
    let tid = lid.x;
    let n_heads = params[0];
    let n_kv_heads = params[1];
    let kv_dim = params[3];
    let seq_len = params[4];
    let scale = bitcast<f32>(params[5]);

    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let kv_h_offset = kv_head * HEAD_DIM;
    let q_offset = head * HEAD_DIM;

    if tid < HEAD_DIM {
        q_shared[tid] = q[q_offset + tid];
    }
    if tid == 0u {
        st[0] = NEG_INF;
        st[1] = 0.0;
    }
    let part_id = tid >> 5u;
    let dp = tid & 31u;
    var acc_x = 0.0;
    var acc_y = 0.0;
    workgroupBarrier();

    var base = split * TILE;
    while base < seq_len {
        // ── scores: four lanes per timestep, each a 16-dim slice of the key row, so a wave reads
        // contiguous 128-byte rows with 16-byte loads instead of one lane per row, 32 scalar loads each ──
        let lane = tid & 3u;
        let row_in_pass = tid >> 2u;
        for (var rp = 0u; rp < 4u; rp += 1u) {
            let tl = rp * 64u + row_in_pass;
            let tt = base + tl;
            var pd = 0.0;
            if tt < seq_len {
                let h = tt * kv_dim + kv_h_offset + lane * 16u;
                let k0 = k_cache[h >> 3u];
                let k1 = k_cache[(h >> 3u) + 1u];
                let q0 = lane * 16u;
                let a = unpack2x16float(k0.x);
                let b = unpack2x16float(k0.y);
                let c = unpack2x16float(k0.z);
                let d = unpack2x16float(k0.w);
                let e = unpack2x16float(k1.x);
                let f = unpack2x16float(k1.y);
                let g = unpack2x16float(k1.z);
                let hh = unpack2x16float(k1.w);
                pd = q_shared[q0] * a.x + q_shared[q0 + 1u] * a.y
                   + q_shared[q0 + 2u] * b.x + q_shared[q0 + 3u] * b.y
                   + q_shared[q0 + 4u] * c.x + q_shared[q0 + 5u] * c.y
                   + q_shared[q0 + 6u] * d.x + q_shared[q0 + 7u] * d.y
                   + q_shared[q0 + 8u] * e.x + q_shared[q0 + 9u] * e.y
                   + q_shared[q0 + 10u] * f.x + q_shared[q0 + 11u] * f.y
                   + q_shared[q0 + 12u] * g.x + q_shared[q0 + 13u] * g.y
                   + q_shared[q0 + 14u] * hh.x + q_shared[q0 + 15u] * hh.y;
            }
            pdot[tl * 4u + lane] = pd;
        }
        workgroupBarrier();
        let t = base + tid;
        var score = NEG_INF;
        if t < seq_len {
            score = (pdot[tid * 4u] + pdot[tid * 4u + 1u] + pdot[tid * 4u + 2u] + pdot[tid * 4u + 3u]) * scale;
        }
        tile_scores[tid] = score;

        // ── tile max ──
        red[tid] = score;
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
        let tmax = red[0];

        if tid == 0u {
            let nm = max(st[0], tmax);
            st[2] = nm;
            st[3] = exp(st[0] - nm); // first tile: exp(-inf) = 0
        }
        workgroupBarrier();
        let nm = st[2];
        let corr = st[3];

        var p = 0.0;
        if t < seq_len {
            p = exp(tile_scores[tid] - nm);
        }
        tile_scores[tid] = p;

        // ── tile sum ──
        red[tid] = p;
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
        let tsum = red[0];

        // ── this thread's share of the value sum: dimension pair `dp`, timesteps part*32 .. +31 ──
        var sx = 0.0;
        var sy = 0.0;
        let v_word = (kv_h_offset >> 1u) + dp;
        for (var jj = 0u; jj < 32u; jj += 1u) {
            let j = part_id * 32u + jj;
            let tt = base + j;
            if tt < seq_len {
                let pair = unpack2x16float(v_cache[(tt * kv_dim >> 1u) + v_word]);
                let pj = tile_scores[j];
                sx += pj * pair.x;
                sy += pj * pair.y;
            }
        }
        acc_x = acc_x * corr + sx;
        acc_y = acc_y * corr + sy;
        if tid == 0u {
            st[1] = st[1] * corr + tsum;
            st[0] = nm;
        }
        // Before the next tile reuses tile_scores / red and reads st.
        workgroupBarrier();
        base += SPLITS * TILE;
    }

    // Add the 8 parts' accumulators per dimension.
    part_acc[part_id * 64u + dp * 2u] = acc_x;
    part_acc[part_id * 64u + dp * 2u + 1u] = acc_y;
    workgroupBarrier();
    let rec = (head * SPLITS + split) * REC;
    if tid < HEAD_DIM {
        var a = 0.0;
        for (var pp = 0u; pp < 8u; pp += 1u) {
            a += part_acc[pp * 64u + tid];
        }
        part[rec + 2u + tid] = a;
    }
    if tid == 0u {
        part[rec] = st[0];
        part[rec + 1u] = st[1];
    }
}

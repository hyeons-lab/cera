// Split-K FlashAttention for one query vector with head_dim == 128, phase 1 of 2: the decode path.
//
// Twin of `flash_attention_split.wgsl` specialized for head_dim == 128 (LFM3).
// Grid (n_heads, SPLITS, 1); 256 threads. Split `s` takes tiles `s, s + SPLITS, ...` of TILE = 256
// starting from the tile covering `min_pos` (`params[6]`).
//
// Scores: 8 lanes per timestep (`lane = tid & 7`), each reading 2 `vec4<u32>` = 16 dims, 8 passes = 256 timesteps.
// Value sum: 4 parts (`part_id = tid >> 6`), `dp = tid & 63` (64 dim pairs = 128 dims), 64 timesteps per part.
// Output `part[(head * SPLITS + split) * 130 + {0: max, 1: sum, 2..130: acc}]` (f32), unnormalized.
//
// Bind group 0:
//   @binding(0) q: array<f32>             (all heads concatenated, read)
//   @binding(1) k_cache: array<vec4<u32>> (seq_len x kv_dim packed halves, read; 8 halves per element)
//   @binding(2) v_cache: array<u32>       (seq_len x kv_dim packed halves, read)
//   @binding(3) part: array<f32>          (n_heads x SPLITS x 130, read-write)
//   @binding(4) params: array<u32,8>      (n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale_bits, min_pos, _)

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 8>;

const TILE: u32 = 256u;
const SPLITS: u32 = 16u;
const HEAD_DIM: u32 = 128u;
const REC: u32 = 130u;
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> q_shared: array<f32, 128>;
var<workgroup> tile_scores: array<f32, 256>;
var<workgroup> pdot: array<f32, 2048>;     // [256 timesteps][8 lanes] partial dot products
var<workgroup> red: array<f32, 256>;
var<workgroup> part_acc: array<f32, 512>; // [4 parts][128 dims]
var<workgroup> st: array<f32, 4>;         // [0]=running max, [1]=running sum, [2]=new max, [3]=correction

@compute @workgroup_size(256, 1, 1)
fn flash_attention_split_hd128(
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
    let min_pos = params[6];

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
    let part_id = tid >> 6u;
    let dp = tid & 63u;
    var acc_x = 0.0;
    var acc_y = 0.0;
    workgroupBarrier();

    var base = (min_pos & ~(TILE - 1u)) + split * TILE;
    while base < seq_len {
        let lane = tid & 7u;
        let row_in_pass = tid >> 3u;
        for (var rp = 0u; rp < 8u; rp += 1u) {
            let tl = rp * 32u + row_in_pass;
            let tt = base + tl;
            var pd = 0.0;
            if tt >= min_pos && tt < seq_len {
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
            pdot[tl * 8u + lane] = pd;
        }
        workgroupBarrier();
        let t = base + tid;
        var score = NEG_INF;
        if t >= min_pos && t < seq_len {
            let pbase = tid * 8u;
            score = (pdot[pbase] + pdot[pbase + 1u] + pdot[pbase + 2u] + pdot[pbase + 3u]
                   + pdot[pbase + 4u] + pdot[pbase + 5u] + pdot[pbase + 6u] + pdot[pbase + 7u]) * scale;
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
            st[3] = exp(st[0] - nm);
        }
        workgroupBarrier();
        let nm = st[2];
        let corr = st[3];

        var p = 0.0;
        if t >= min_pos && t < seq_len {
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

        // ── this thread's share of the value sum: dimension pair `dp`, timesteps part_id*64 .. +63 ──
        var sx = 0.0;
        var sy = 0.0;
        let v_word = (kv_h_offset >> 1u) + dp;
        for (var jj = 0u; jj < 64u; jj += 1u) {
            let j = part_id * 64u + jj;
            let tt = base + j;
            if tt >= min_pos && tt < seq_len {
                let pair = unpack2x16float(v_cache[((tt * kv_dim) >> 1u) + v_word]);
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
        workgroupBarrier();
        base += SPLITS * TILE;
    }

    // Add the 4 parts' accumulators per dimension.
    part_acc[part_id * 128u + dp * 2u] = acc_x;
    part_acc[part_id * 128u + dp * 2u + 1u] = acc_y;
    workgroupBarrier();
    let rec = (head * SPLITS + split) * REC;
    if tid < HEAD_DIM {
        part[rec + 2u + tid] = part_acc[tid] + part_acc[128u + tid] + part_acc[256u + tid] + part_acc[384u + tid];
    }
    if tid == 0u {
        part[rec] = st[0];
        part[rec + 1u] = st[1];
    }
}

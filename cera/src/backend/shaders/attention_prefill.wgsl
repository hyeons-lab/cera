// Batched FlashAttention prefill: N queries × n_heads in one dispatch.
//
// Online-softmax (Dao 2022), single tiled pass over the KV cache with bounded
// workgroup memory and NO external scores scratch buffer — 5 bindings. This is
// the decode `flash_attention.wgsl` kernel extended along the query dimension:
// one workgroup per (head, 8-query group), a per-query causal window, and
// batched Q/out strides. The 8 queries in a workgroup share each K/V tile
// (read once from DRAM, L2-shared across queries); the single-query shape
// this replaced re-read the whole KV cache per query — 12 GB of KV traffic
// for a 512-token prefill on LFM2-350M, 15% of prefill time on Adreno 830.
//
// Matches a plain batched-softmax reference up to floating-point roundoff
// (online vs batched accumulation order, 32-key tiles); the wgpu-side test
// asserts it against a CPU ground truth.
//
// Because the scores are never materialized (only an 8×32 tile lives in
// workgroup memory), the storage bindings do not scale with seq_len — this
// removes the `[n_queries, n_heads, max_seq]` scores slab that previously forced
// host-side query-tiling to stay under the adapter storage-binding limit.
// Contexts long enough that the *KV* binding itself (`max_seq × kv_dim`)
// overflows the limit remain future paged-KV work, guarded host-side.
//
// The tree reductions are inlined (not helper functions) and all barriers sit at
// kernel scope — naga's SPIR-V path (lavapipe/Vulkan) miscompiles
// `workgroupBarrier()` reached through a function call inside a loop, so every
// barrier stays in the entry point. The 8 queries reduce concurrently (lanes
// work their query's slice); no query loop wraps any barrier.
//
// Constraints (asserted host-side in encode_attention_prefill):
//   - head_dim <= 128 (bounds `q_shared` and `acc`).
//   - caller MUST pass `max_seq >= start_pos + n_queries`; K/V must hold valid
//     entries for positions `[0, start_pos + n_queries)`. As a defensive belt the
//     shader clamps `seq_len = min(pos_q + 1, max_seq)` — an under-sized `max_seq`
//     yields truncated (incorrect) attention rather than an OOB read.
// GQA: kv_head = head / (n_heads / n_kv_heads).
//
// Bind group 0:
//   @binding(0) q_batch:   array<f32>    n_queries × q_stride floats
//   @binding(1) k_cache:   array<u32>    seq_len × kv_dim packed halves
//   @binding(2) v_cache:   array<u32>    seq_len × kv_dim packed halves
//   @binding(3) out_batch: array<f32>    n_queries × out_stride floats (rw)
//   @binding(4) params:    array<u32, 12>
//
// K/V are LE f16 halves packed 2-per-u32 (see `flash_attention.wgsl`):
// kv_dim and head_dim are even (asserted host-side), accumulation stays f32.
//        ( n_heads, n_kv_heads, head_dim, kv_dim, max_seq, scale_bits,
//          start_pos, <unused>, q_stride, out_stride, q_base, n_sub )
//
// params[7] is NOT read by the shader — the per-dispatch query count comes from
// params[11] (`n_sub`, authoritative) — edge workgroups mask queries past it.
// `q_base` is the index, within `q_batch` / `out_batch`, of the first query in
// this dispatch, so a caller may still split the query batch across dispatches;
// `q_global = q_base + q_idx` addresses `q_batch` / `out_batch` and sets the
// causal position. `q_base = 0` is the single-dispatch case (the default now that
// the scores slab is gone).
//
// Dispatch: (n_heads, ceil(n_sub / 8), 1) workgroups of 256 threads.

@group(0) @binding(0) var<storage, read> q_batch: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<u32>;
@group(0) @binding(2) var<storage, read> v_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_batch: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32, 12>;

const TILE: u32 = 32u;
const Q_PER_WG: u32 = 8u;
const LANES: u32 = 32u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> q_shared: array<f32, Q_PER_WG * MAX_HEAD_DIM>;
var<workgroup> acc: array<f32, Q_PER_WG * MAX_HEAD_DIM>; // per-query output accumulator
var<workgroup> tile_scores: array<f32, Q_PER_WG * TILE>;
var<workgroup> red: array<f32, Q_PER_WG * TILE>; // reduction scratch
// Running online-softmax state per query, broadcast via workgroup memory.
// [q*4+0]=running max, [q*4+1]=running sum, [q*4+2]=tile new max, [q*4+3]=correction.
var<workgroup> st: array<f32, Q_PER_WG * 4>;

@compute @workgroup_size(256, 1, 1)
fn attention_prefill(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    let q_grp = wid.y;
    let tid = lid.x;
    let q = tid / LANES;
    let lane = tid % LANES;

    let n_heads = params[0];
    let n_kv_heads = params[1];
    let head_dim = params[2];
    let kv_dim = params[3];
    let max_seq = params[4];
    let scale = bitcast<f32>(params[5]);
    let start_pos = params[6];
    // params[7] is unused (historical batch size; authoritative count is params[11]).
    let q_stride = params[8];
    let out_stride = params[9];
    let q_base = params[10];
    let n_sub = params[11];

    // Local query index within this dispatch; queries past n_sub are dead:
    // they run the loop (barrier uniformity) on clamped inputs and skip stores.
    let q_local = q_grp * Q_PER_WG + q;
    let q_live = q_local < n_sub;
    let q_global = q_base + q_local;
    // Dead queries read Q row q_base (always valid: q_base < batch size... but
    // a fully-OOB edge workgroup has q_base + 0 valid only if the batch is
    // non-empty, which the host guarantees with n == 0 early-out).
    let q_read = select(q_base, q_global, q_live);

    // Per-query causal window: attend over [0..pos_q]. Clamp against max_seq so a
    // caller passing inconsistent params can only cause silent window truncation,
    // never an OOB read of k_cache / v_cache.
    let pos_q = start_pos + q_read;
    let seq_len = min(pos_q + 1u, max_seq);

    let group_size = n_heads / n_kv_heads;
    let kv_head = head / group_size;
    let kv_h_offset = kv_head * head_dim;
    let q_offset = q_read * q_stride + head * head_dim;
    let out_offset = q_global * out_stride + head * head_dim;

    // Load Q into workgroup memory and zero the output accumulator.
    // Threads stride by LANES over head_dim (<= 128, so <= 4 iters).
    var d = lane;
    loop {
        if d >= head_dim { break; }
        q_shared[q * MAX_HEAD_DIM + d] = q_batch[q_offset + d];
        acc[q * MAX_HEAD_DIM + d] = 0.0;
        d += LANES;
    }
    if lane == 0u {
        st[q * 4u] = NEG_INF; // running max
        st[q * 4u + 1u] = 0.0; // running sum
    }
    workgroupBarrier();

    // Tile loop bound: the max seq_len across LIVE queries (dead slots
    // contribute 0 so edge workgroups don't spin over empty tiles).
    // Uniform: same q_grp shape for all threads in the group. Unrolled over
    // the 8 queries (constant indices keep naga happy).
    let lbase = q_grp * Q_PER_WG;
    var max_seq_all = 0u;
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 0u + 1u, max_seq), lbase + 0u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 1u + 1u, max_seq), lbase + 1u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 2u + 1u, max_seq), lbase + 2u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 3u + 1u, max_seq), lbase + 3u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 4u + 1u, max_seq), lbase + 4u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 5u + 1u, max_seq), lbase + 5u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 6u + 1u, max_seq), lbase + 6u < n_sub));
    max_seq_all = max(max_seq_all, select(0u, min(start_pos + q_base + lbase + 7u + 1u, max_seq), lbase + 7u < n_sub));

    var base = 0u;
    loop {
        if base >= max_seq_all { break; }
        // ── score for timestep t = base + lane (one per lane, per query) ──
        let t = base + lane;
        var score = NEG_INF;
        if t < seq_len {
            var dot = 0.0;
            // k_base is even (kv_dim and kv_h_offset are multiples of the
            // even head_dim), so stepping dd by 2 walks whole u32 words.
            let k_base = t * kv_dim + kv_h_offset;
            for (var dd = 0u; dd < head_dim; dd += 2u) {
                let pair = unpack2x16float(k_cache[(k_base + dd) >> 1u]);
                dot += q_shared[q * MAX_HEAD_DIM + dd] * pair.x
                    + q_shared[q * MAX_HEAD_DIM + dd + 1u] * pair.y;
            }
            score = dot * scale;
        }
        tile_scores[q * TILE + lane] = score;

        // ── tile max: 8 concurrent 32-wide tree reductions over `red` ──
        red[tid] = score;
        workgroupBarrier();
        if lane < 16u { red[tid] = max(red[tid], red[tid + 16u]); }
        workgroupBarrier();
        if lane < 8u { red[tid] = max(red[tid], red[tid + 8u]); }
        workgroupBarrier();
        if lane < 4u { red[tid] = max(red[tid], red[tid + 4u]); }
        workgroupBarrier();
        if lane < 2u { red[tid] = max(red[tid], red[tid + 2u]); }
        workgroupBarrier();
        if lane < 1u { red[tid] = max(red[tid], red[tid + 1u]); }
        workgroupBarrier();
        let tmax = red[q * LANES];

        // new running max + correction factor (published by lane 0 of each query)
        if lane == 0u {
            let nm = max(st[q * 4u], tmax);
            st[q * 4u + 2u] = nm;
            st[q * 4u + 3u] = exp(st[q * 4u] - nm); // first tile: exp(-inf) = 0
        }
        workgroupBarrier();
        let nm = st[q * 4u + 2u];
        let corr = st[q * 4u + 3u];

        // p = exp(score - nm); reuse tile_scores to hold the exponentials.
        var p = 0.0;
        if t < seq_len {
            p = exp(tile_scores[q * TILE + lane] - nm);
        }
        tile_scores[q * TILE + lane] = p;

        // ── tile sum: 8 concurrent 32-wide tree reductions over `red` ──
        red[tid] = p;
        workgroupBarrier();
        if lane < 16u { red[tid] += red[tid + 16u]; }
        workgroupBarrier();
        if lane < 8u { red[tid] += red[tid + 8u]; }
        workgroupBarrier();
        if lane < 4u { red[tid] += red[tid + 4u]; }
        workgroupBarrier();
        if lane < 2u { red[tid] += red[tid + 2u]; }
        workgroupBarrier();
        if lane < 1u { red[tid] += red[tid + 1u]; }
        workgroupBarrier();
        let tsum = red[q * LANES];

        // rescale the accumulator by the correction and add this tile's V.
        var dd = lane;
        loop {
            if dd >= head_dim { break; }
            var a = acc[q * MAX_HEAD_DIM + dd] * corr;
            let vd = kv_h_offset + dd;
            // vd's lane is loop-invariant; hoist it out of the row walk.
            let vd_hi = (vd & 1u) == 1u;
            for (var jj = 0u; jj < TILE; jj += 1u) {
                let tt = base + jj;
                if tt < seq_len {
                    let pair = unpack2x16float(v_cache[(tt * kv_dim + vd) >> 1u]);
                    a += tile_scores[q * TILE + jj] * select(pair.x, pair.y, vd_hi);
                }
            }
            acc[q * MAX_HEAD_DIM + dd] = a;
            dd += LANES;
        }
        if lane == 0u {
            st[q * 4u + 1u] = st[q * 4u + 1u] * corr + tsum;
            st[q * 4u] = nm;
        }
        // Barrier before the next tile reuses tile_scores/red and reads acc/st.
        workgroupBarrier();
        base += TILE;
    }

    // Normalize and store. Live queries with an empty window (seq_len == 0,
    // only if max_seq == 0) write zeros instead of NaN; dead queries skip.
    if q_live {
        let inv = st[q * 4u + 1u];
        var dd = lane;
        loop {
            if dd >= head_dim { break; }
            let v = acc[q * MAX_HEAD_DIM + dd];
            if inv == 0.0 {
                out_batch[out_offset + dd] = 0.0;
            } else {
                out_batch[out_offset + dd] = v / inv;
            }
            dd += LANES;
        }
    }
}

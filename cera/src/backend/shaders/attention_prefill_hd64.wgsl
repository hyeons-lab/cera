// Causal (and bidirectional-prefix) flash attention prefill for head_dim 64, reading the packed-f16
// KV cache. The batched prefill of `attention_prefill.wgsl` for the one head size the LFM2 family
// uses, written as the register-tiled kernel of `attention_flash_hd64.wgsl` (which serves the f32,
// bidirectional vision tower): about 0.2 TFLOPS on an Adreno 830 against 0.07 for the scalar
// kernel, whose 8-query workgroups re-read every K/V tile once per 8 queries and reduce each
// softmax tile through a barrier tree.
//
// ## Shape
//
// A workgroup of 128 threads (8 query groups x 16 columns) owns 32 queries of one head and sweeps
// the keys 32 at a time, keeping the operands in workgroup memory and the results in 4 x 4
// register tiles:
//
//   S = Q K^T   thread (ty, tx): 4 queries x 2 keys
//   P = softmax online, in the log2 domain (Q is pre-scaled by scale * log2(e), so `exp2`)
//   O += P V    thread (ty, tx): 4 queries x 4 dims
//
// The loops are unrolled by hand and every per-thread value is a named register for the reason the
// f32 kernel gives: wgpu bounds each loop with a 64-bit counter, which stops the Metal compiler
// unrolling it and leaves loop-indexed arrays in memory.
//
// ## What differs from the f32 kernel
//
//   * K and V are the model's packed-f16 cache (two halves per u32, LE; four dims are one
//     vec2<u32>), widened to f32 with `unpack2x16float` as they are staged. Accumulation is f32.
//   * The key window of a query is its causal prefix (`pos + 1`, with `pos = start_pos + row`),
//     or in a bidirectional pass the whole call's rows, except that a query inside the media
//     prefix reads only the prefix. A workgroup sweeps only the keys its last live query reads,
//     so a causal pass does about half the work of a full one.
//   * Queries are addressed like `attention_prefill.wgsl`: `q_base` is the first query of this
//     dispatch within the Q and output batches and `n_sub` the number of live queries, so the host
//     can split a long call into dispatches short enough for a phone's hang detector. A query
//     past `n_sub` reads a clamped row and writes nothing.
//
// ## Bindings (group 0), the contract of `attention_prefill.wgsl` with vector element types
//
//   0 q_batch    array<vec4<f32>>   n_queries x q_stride floats       (q_stride % 4 == 0)
//   1 k_cache    array<vec2<u32>>   max_seq x kv_dim packed halves
//   2 v_cache    array<vec2<u32>>   max_seq x kv_dim packed halves
//   3 out_batch  array<vec4<f32>>   n_queries x out_stride floats, read_write (out_stride % 4 == 0)
//   4 params     array<u32, 14>     n_heads, n_kv_heads, head_dim (64), kv_dim, max_seq, scale_bits,
//                                   start_pos, total_rows, q_stride, out_stride, q_base, n_sub,
//                                   bidir, prefix_rows
//
// `total_rows` (params[7]) is the whole call's key rows, read only when `bidir` is set.
//
// Dispatch: (ceil(n_sub / 32), n_heads, 1) workgroups of 128 threads. Workgroup memory is about
// 30 KB. A live query with an empty window (`max_seq` 0) writes zeros.

const QT: u32 = 32u;
const KT: u32 = 32u;
const HD: u32 = 64u;
const NEG: f32 = -1.0e30;
const LOG2E: f32 = 1.4426950408889634;

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> v: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> params: array<u32, 14>;

// Q^T: [d][32 queries], as 8 vec4 (4 queries each) per d.
var<workgroup> qs: array<vec4<f32>, 512>;
// K^T: [d][32 keys], as 16 vec2 (2 keys each) per d.
var<workgroup> kt: array<vec2<f32>, 1024>;
// V: [32 keys][64 d] as 17 vec4 per key (one of padding).
var<workgroup> vs: array<vec4<f32>, 544>;
// scores, then probabilities: [32 keys][32 queries].
var<workgroup> ps: array<f32, 1024>;
var<workgroup> pm: array<f32, 128>;
var<workgroup> corr: array<f32, 32>;
var<workgroup> lsum: array<f32, 128>;

// Four consecutive dims of a packed-f16 cache row, as f32.
fn half4(w: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
}

// How many keys the query at global position `pos` reads: its causal prefix, or in a bidirectional
// pass the whole call (the media prefix, for a query inside it). Clamped to the cache's live rows.
fn key_window(pos: u32, bidir: u32, prefix: u32, total: u32, max_seq: u32) -> u32 {
    var w = pos + 1u;
    if bidir != 0u {
        w = select(total, prefix, pos < prefix);
    }
    return min(w, max_seq);
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let tx = t % 16u;
    let ty = t / 16u;
    let n_head = params[0];
    let n_kv = params[1];
    let kv_dim4 = params[3] / 4u;
    let max_seq = params[4];
    let scale = bitcast<f32>(params[5]);
    let start_pos = params[6];
    let total = params[7];
    let q_stride4 = params[8] / 4u;
    let out_stride4 = params[9] / 4u;
    let q_base = params[10];
    let n_sub = params[11];
    let bidir = params[12];
    let prefix = params[13];
    let head = wid.y;
    let kv_head = head / (n_head / n_kv);
    let q0 = wid.x * QT;
    // The last live row of this tile bounds the keys the workgroup sweeps (a window never shrinks
    // with the row, and a query past n_sub is clamped onto the last live row).
    let last_row = min(q0 + QT - 1u, n_sub - 1u);
    let wg_limit = key_window(start_pos + q_base + last_row, bidir, prefix, total, max_seq);

    // Q tile -> qs, 4 queries x 4 dims per thread, pre-scaled into the log2 domain.
    {
        let qg = t % 8u;
        let dv = t / 8u;
        let s = scale * LOG2E;
        let r0 = q[(q_base + min(q0 + qg * 4u + 0u, n_sub - 1u)) * q_stride4 + head * 16u + dv] * s;
        let r1 = q[(q_base + min(q0 + qg * 4u + 1u, n_sub - 1u)) * q_stride4 + head * 16u + dv] * s;
        let r2 = q[(q_base + min(q0 + qg * 4u + 2u, n_sub - 1u)) * q_stride4 + head * 16u + dv] * s;
        let r3 = q[(q_base + min(q0 + qg * 4u + 3u, n_sub - 1u)) * q_stride4 + head * 16u + dv] * s;
        qs[(4u * dv + 0u) * 8u + qg] = vec4<f32>(r0.x, r1.x, r2.x, r3.x);
        qs[(4u * dv + 1u) * 8u + qg] = vec4<f32>(r0.y, r1.y, r2.y, r3.y);
        qs[(4u * dv + 2u) * 8u + qg] = vec4<f32>(r0.z, r1.z, r2.z, r3.z);
        qs[(4u * dv + 3u) * 8u + qg] = vec4<f32>(r0.w, r1.w, r2.w, r3.w);
    }

    // per-thread: the 4 queries of the QK/PV tile
    let lim0 = key_window(start_pos + q_base + min(q0 + ty * 4u + 0u, n_sub - 1u), bidir, prefix, total, max_seq);
    let lim1 = key_window(start_pos + q_base + min(q0 + ty * 4u + 1u, n_sub - 1u), bidir, prefix, total, max_seq);
    let lim2 = key_window(start_pos + q_base + min(q0 + ty * 4u + 2u, n_sub - 1u), bidir, prefix, total, max_seq);
    let lim3 = key_window(start_pos + q_base + min(q0 + ty * 4u + 3u, n_sub - 1u), bidir, prefix, total, max_seq);

    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    // softmax role: query `sq`, keys [sp*8, sp*8+8) of each block
    let sq = t % 32u;
    let sp = t / 32u;
    var m = NEG;
    var l = 0.0;

    for (var kb = 0u; kb < wg_limit; kb += KT) {
        // K^T block: 2 keys x 4 dims per unit, 2 units per thread
        {

            let u = t + 128u * 0u;
            let kp = u % 16u;
            let dv = u / 16u;
            let ra = kb + kp * 2u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if ra < wg_limit {
                a = half4(k[ra * kv_dim4 + kv_head * 16u + dv]);
            }
            if ra + 1u < wg_limit {
                b = half4(k[(ra + 1u) * kv_dim4 + kv_head * 16u + dv]);
            }
            kt[(4u * dv + 0u) * 16u + kp] = vec2<f32>(a.x, b.x);
            kt[(4u * dv + 1u) * 16u + kp] = vec2<f32>(a.y, b.y);
            kt[(4u * dv + 2u) * 16u + kp] = vec2<f32>(a.z, b.z);
            kt[(4u * dv + 3u) * 16u + kp] = vec2<f32>(a.w, b.w);
                }
        {

            let u = t + 128u * 1u;
            let kp = u % 16u;
            let dv = u / 16u;
            let ra = kb + kp * 2u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if ra < wg_limit {
                a = half4(k[ra * kv_dim4 + kv_head * 16u + dv]);
            }
            if ra + 1u < wg_limit {
                b = half4(k[(ra + 1u) * kv_dim4 + kv_head * 16u + dv]);
            }
            kt[(4u * dv + 0u) * 16u + kp] = vec2<f32>(a.x, b.x);
            kt[(4u * dv + 1u) * 16u + kp] = vec2<f32>(a.y, b.y);
            kt[(4u * dv + 2u) * 16u + kp] = vec2<f32>(a.z, b.z);
            kt[(4u * dv + 3u) * 16u + kp] = vec2<f32>(a.w, b.w);
                }

        workgroupBarrier();

        // S = Q K^T: this thread's 4 queries x 2 keys
        var s0 = vec2<f32>(0.0);
        var s1 = vec2<f32>(0.0);
        var s2 = vec2<f32>(0.0);
        var s3 = vec2<f32>(0.0);
        for (var d = 0u; d < HD; d += 8u) {
            {
                let qv = qs[(d + 0u) * 8u + ty];
                let kv = kt[(d + 0u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 1u) * 8u + ty];
                let kv = kt[(d + 1u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 2u) * 8u + ty];
                let kv = kt[(d + 2u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 3u) * 8u + ty];
                let kv = kt[(d + 3u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 4u) * 8u + ty];
                let kv = kt[(d + 4u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 5u) * 8u + ty];
                let kv = kt[(d + 5u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 6u) * 8u + ty];
                let kv = kt[(d + 6u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
            {
                let qv = qs[(d + 7u) * 8u + ty];
                let kv = kt[(d + 7u) * 16u + tx];
                s0 += qv.x * kv;
                s1 += qv.y * kv;
                s2 += qv.z * kv;
                s3 += qv.w * kv;
            }
        }
        let key0 = kb + tx * 2u;
        let pb0 = (tx * 2u + 0u) * 32u + ty * 4u;
        let pb1 = (tx * 2u + 1u) * 32u + ty * 4u;
        ps[pb0] = select(NEG, s0.x, key0 < lim0);
        ps[pb0 + 1u] = select(NEG, s1.x, key0 < lim1);
        ps[pb0 + 2u] = select(NEG, s2.x, key0 < lim2);
        ps[pb0 + 3u] = select(NEG, s3.x, key0 < lim3);
        ps[pb1] = select(NEG, s0.y, key0 + 1u < lim0);
        ps[pb1 + 1u] = select(NEG, s1.y, key0 + 1u < lim1);
        ps[pb1 + 2u] = select(NEG, s2.y, key0 + 1u < lim2);
        ps[pb1 + 3u] = select(NEG, s3.y, key0 + 1u < lim3);
        workgroupBarrier();

        // V block (K^T is no longer read) while the softmax runs
        {

            let idx = t + 128u * 0u;
            let kj = idx / 16u;
            let dv = idx % 16u;
            let row = kb + kj;
            var x = vec4<f32>(0.0);
            if row < wg_limit {
                x = half4(v[row * kv_dim4 + kv_head * 16u + dv]);
            }
            vs[kj * 17u + dv] = x;
                }
        {

            let idx = t + 128u * 1u;
            let kj = idx / 16u;
            let dv = idx % 16u;
            let row = kb + kj;
            var x = vec4<f32>(0.0);
            if row < wg_limit {
                x = half4(v[row * kv_dim4 + kv_head * 16u + dv]);
            }
            vs[kj * 17u + dv] = x;
                }
        {

            let idx = t + 128u * 2u;
            let kj = idx / 16u;
            let dv = idx % 16u;
            let row = kb + kj;
            var x = vec4<f32>(0.0);
            if row < wg_limit {
                x = half4(v[row * kv_dim4 + kv_head * 16u + dv]);
            }
            vs[kj * 17u + dv] = x;
                }
        {

            let idx = t + 128u * 3u;
            let kj = idx / 16u;
            let dv = idx % 16u;
            let row = kb + kj;
            var x = vec4<f32>(0.0);
            if row < wg_limit {
                x = half4(v[row * kv_dim4 + kv_head * 16u + dv]);
            }
            vs[kj * 17u + dv] = x;
                }


        // online softmax, 4 threads per query, 8 keys each
        var sv: array<f32, 8>;
        var pmax = NEG;
        let x0 = ps[(sp * 8u + 0u) * 32u + sq];
        pmax = max(pmax, x0);
        let x1 = ps[(sp * 8u + 1u) * 32u + sq];
        pmax = max(pmax, x1);
        let x2 = ps[(sp * 8u + 2u) * 32u + sq];
        pmax = max(pmax, x2);
        let x3 = ps[(sp * 8u + 3u) * 32u + sq];
        pmax = max(pmax, x3);
        let x4 = ps[(sp * 8u + 4u) * 32u + sq];
        pmax = max(pmax, x4);
        let x5 = ps[(sp * 8u + 5u) * 32u + sq];
        pmax = max(pmax, x5);
        let x6 = ps[(sp * 8u + 6u) * 32u + sq];
        pmax = max(pmax, x6);
        let x7 = ps[(sp * 8u + 7u) * 32u + sq];
        pmax = max(pmax, x7);
        pm[sp * 32u + sq] = pmax;
        workgroupBarrier();
        let m_new = max(max(m, max(pm[sq], pm[32u + sq])), max(pm[64u + sq], pm[96u + sq]));
        let c = exp2(m - m_new);
        var psum = 0.0;
        let p0 = exp2(x0 - m_new);
        ps[(sp * 8u + 0u) * 32u + sq] = p0;
        psum += p0;
        let p1 = exp2(x1 - m_new);
        ps[(sp * 8u + 1u) * 32u + sq] = p1;
        psum += p1;
        let p2 = exp2(x2 - m_new);
        ps[(sp * 8u + 2u) * 32u + sq] = p2;
        psum += p2;
        let p3 = exp2(x3 - m_new);
        ps[(sp * 8u + 3u) * 32u + sq] = p3;
        psum += p3;
        let p4 = exp2(x4 - m_new);
        ps[(sp * 8u + 4u) * 32u + sq] = p4;
        psum += p4;
        let p5 = exp2(x5 - m_new);
        ps[(sp * 8u + 5u) * 32u + sq] = p5;
        psum += p5;
        let p6 = exp2(x6 - m_new);
        ps[(sp * 8u + 6u) * 32u + sq] = p6;
        psum += p6;
        let p7 = exp2(x7 - m_new);
        ps[(sp * 8u + 7u) * 32u + sq] = p7;
        psum += p7;
        l = l * c + psum;
        m = m_new;
        if sp == 0u {
            corr[sq] = c;
        }
        workgroupBarrier();

        // O = O * corr + P V
        o0 *= corr[ty * 4u + 0u];
        o1 *= corr[ty * 4u + 1u];
        o2 *= corr[ty * 4u + 2u];
        o3 *= corr[ty * 4u + 3u];
        for (var j = 0u; j < KT; j += 8u) {
            {
                let pv = vec4<f32>(ps[(j + 0u) * 32u + ty * 4u], ps[(j + 0u) * 32u + ty * 4u + 1u], ps[(j + 0u) * 32u + ty * 4u + 2u], ps[(j + 0u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 0u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 1u) * 32u + ty * 4u], ps[(j + 1u) * 32u + ty * 4u + 1u], ps[(j + 1u) * 32u + ty * 4u + 2u], ps[(j + 1u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 1u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 2u) * 32u + ty * 4u], ps[(j + 2u) * 32u + ty * 4u + 1u], ps[(j + 2u) * 32u + ty * 4u + 2u], ps[(j + 2u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 2u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 3u) * 32u + ty * 4u], ps[(j + 3u) * 32u + ty * 4u + 1u], ps[(j + 3u) * 32u + ty * 4u + 2u], ps[(j + 3u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 3u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 4u) * 32u + ty * 4u], ps[(j + 4u) * 32u + ty * 4u + 1u], ps[(j + 4u) * 32u + ty * 4u + 2u], ps[(j + 4u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 4u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 5u) * 32u + ty * 4u], ps[(j + 5u) * 32u + ty * 4u + 1u], ps[(j + 5u) * 32u + ty * 4u + 2u], ps[(j + 5u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 5u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 6u) * 32u + ty * 4u], ps[(j + 6u) * 32u + ty * 4u + 1u], ps[(j + 6u) * 32u + ty * 4u + 2u], ps[(j + 6u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 6u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
            {
                let pv = vec4<f32>(ps[(j + 7u) * 32u + ty * 4u], ps[(j + 7u) * 32u + ty * 4u + 1u], ps[(j + 7u) * 32u + ty * 4u + 2u], ps[(j + 7u) * 32u + ty * 4u + 3u]);
                let vv = vs[(j + 7u) * 17u + tx];
                o0 += pv.x * vv;
                o1 += pv.y * vv;
                o2 += pv.z * vv;
                o3 += pv.w * vv;
            }
        }
        workgroupBarrier();
    }

    lsum[sp * 32u + sq] = l;
    workgroupBarrier();
    let r0 = q0 + ty * 4u;
    let s0 = lsum[ty * 4u] + lsum[32u + ty * 4u] + lsum[64u + ty * 4u] + lsum[96u + ty * 4u];
    let s1 = lsum[ty * 4u + 1u] + lsum[32u + ty * 4u + 1u] + lsum[64u + ty * 4u + 1u] + lsum[96u + ty * 4u + 1u];
    let s2 = lsum[ty * 4u + 2u] + lsum[32u + ty * 4u + 2u] + lsum[64u + ty * 4u + 2u] + lsum[96u + ty * 4u + 2u];
    let s3 = lsum[ty * 4u + 3u] + lsum[32u + ty * 4u + 3u] + lsum[64u + ty * 4u + 3u] + lsum[96u + ty * 4u + 3u];
    // An empty window leaves the sum at 0: write zeros, not 0 * inf.
    let inv0 = select(0.0, 1.0 / s0, s0 > 0.0);
    let inv1 = select(0.0, 1.0 / s1, s1 > 0.0);
    let inv2 = select(0.0, 1.0 / s2, s2 > 0.0);
    let inv3 = select(0.0, 1.0 / s3, s3 > 0.0);
    if r0 + 0u < n_sub { out[(q_base + r0 + 0u) * out_stride4 + head * 16u + tx] = o0 * inv0; }
    if r0 + 1u < n_sub { out[(q_base + r0 + 1u) * out_stride4 + head * 16u + tx] = o1 * inv1; }
    if r0 + 2u < n_sub { out[(q_base + r0 + 2u) * out_stride4 + head * 16u + tx] = o2 * inv2; }
    if r0 + 3u < n_sub { out[(q_base + r0 + 3u) * out_stride4 + head * 16u + tx] = o3 * inv3; }
}

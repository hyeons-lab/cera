// Flash attention for head_dim 64: f32 Q/K/V, grouped-query heads, and an optional
// bidirectional window with a media prefix. A register-tiled rewrite of the scalar
// `vit_attention_tiled.wgsl` (about 0.17 TFLOPS on an M1 Max) that reaches about 1.5.
//
// ## Shape
//
// Attention is two GEMMs with a softmax between them, so it is written like
// `mul_mat_reg_tile.wgsl`: a workgroup of 128 threads (8 query groups x 16 columns) owns 32
// queries of one head and sweeps the keys 32 at a time, keeping the operands in workgroup
// memory and the results in 4 x 4 register tiles.
//
//   S = Q K^T   thread (ty, tx): 4 queries x 2 keys   (reads a vec4 of Q^T and a vec2 of K^T per d)
//   P = softmax online, in the log2 domain (Q is pre-scaled by scale * log2(e), so `exp2`)
//   O += P V    thread (ty, tx): 4 queries x 4 dims   (reads a vec4 of P and a vec4 of V per key)
//
// ## Why it is written out
//
// wgpu bounds every loop with a 64-bit counter, which stops the Metal compiler unrolling it and
// leaves per-thread arrays indexed by the loop in memory. The inner loops are therefore
// unrolled by hand (8 d-steps of QK, 8 keys of PV, the 8 softmax keys, the staging copies) and
// every per-thread value is a named register. Written with loops the same kernel ran at
// 0.7 TFLOPS. Two other things the hard way: a dynamic component write into a workgroup vec4
// (`qs[i][c] = x`) produced NaNs, so the transposed tiles are built in registers with static
// components; and a 16-way max reduction per softmax thread was slower than four.
//
// ## Bindings (group 0)
//
//   0 q       array<vec4<f32>>   [tokens, n_head * 64]
//   1 k       array<vec4<f32>>   [tokens, n_kv_head * 64]
//   2 v       array<vec4<f32>>   [tokens, n_kv_head * 64]
//   3 out     array<vec4<f32>>   [tokens, n_head * 64], read_write
//   4 params  array<u32, 8>      tokens, n_head, head_dim (64), scale_bits, n_kv_head,
//                                bidir, prefix_rows, first_query_tile
//
// A query reads the first `window(row)` keys: all `tokens` of them, or with `bidir` set and a
// non-zero `prefix_rows`, a query inside the prefix reads only the prefix (a media prefix is a
// function of the media alone). A causal window is not offered: the causal prefill reads a
// packed f16 cache and has its own kernel (`attention_prefill.wgsl`).
//
// Dispatch: (ceil(tokens / 32), n_head, 1) workgroups of 128 threads, or any run of query tiles
// of that grid starting at `first_query_tile`. Workgroup memory is about
// 30 KB, so it is the one workgroup resident per core on Apple GPUs; the staging is cheap enough
// that this has not mattered. On an Adreno 830 it runs at about 0.22 TFLOPS and is correct to 1e-6;
// a call of seconds there loses the device, so the host issues long calls in short pieces.

const QT: u32 = 32u;
const KT: u32 = 32u;
const HD: u32 = 64u;
const NEG: f32 = -1.0e30;
const LOG2E: f32 = 1.4426950408889634;

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
// tokens, n_head, head_dim, scale_bits, n_kv_head, bidir, prefix_rows, _
@group(0) @binding(4) var<storage, read> params: array<u32, 8>;

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

fn window(row: u32, tokens: u32, bidir: u32, prefix: u32) -> u32 {
    if bidir != 0u {
        return select(tokens, prefix, row < prefix);
    }
    return tokens;
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let tx = t % 16u;
    let ty = t / 16u;
    let tokens = params[0];
    let n_head = params[1];
    let scale = bitcast<f32>(params[3]);
    let n_kv = params[4];
    let bidir = params[5];
    let prefix = params[6];
    let head = wid.y;
    let kv_head = head / (n_head / n_kv);
    let q_stride4 = n_head * 16u;
    let kv_stride4 = n_kv * 16u;
    // `params[7]` is the first query tile this dispatch covers, so one attention can be issued as
    // several short dispatches (a mobile GPU kills a dispatch that runs for seconds)
    let q0 = (wid.x + params[7]) * QT;
    let last_row = min(q0 + QT - 1u, tokens - 1u);
    let wg_limit = window(last_row, tokens, bidir, prefix);

    // Q tile -> qs, 4 queries x 4 dims per thread, pre-scaled into the log2 domain.
    {
        let qg = t % 8u;
        let dv = t / 8u;
        let s = scale * LOG2E;
        let r0 = q[min(q0 + qg * 4u + 0u, tokens - 1u) * q_stride4 + head * 16u + dv] * s;
        let r1 = q[min(q0 + qg * 4u + 1u, tokens - 1u) * q_stride4 + head * 16u + dv] * s;
        let r2 = q[min(q0 + qg * 4u + 2u, tokens - 1u) * q_stride4 + head * 16u + dv] * s;
        let r3 = q[min(q0 + qg * 4u + 3u, tokens - 1u) * q_stride4 + head * 16u + dv] * s;
        qs[(4u * dv + 0u) * 8u + qg] = vec4<f32>(r0.x, r1.x, r2.x, r3.x);
        qs[(4u * dv + 1u) * 8u + qg] = vec4<f32>(r0.y, r1.y, r2.y, r3.y);
        qs[(4u * dv + 2u) * 8u + qg] = vec4<f32>(r0.z, r1.z, r2.z, r3.z);
        qs[(4u * dv + 3u) * 8u + qg] = vec4<f32>(r0.w, r1.w, r2.w, r3.w);
    }

    // per-thread: the 4 queries of the QK/PV tile
    let lim0 = window(min(q0 + ty * 4u + 0u, tokens - 1u), tokens, bidir, prefix);
    let lim1 = window(min(q0 + ty * 4u + 1u, tokens - 1u), tokens, bidir, prefix);
    let lim2 = window(min(q0 + ty * 4u + 2u, tokens - 1u), tokens, bidir, prefix);
    let lim3 = window(min(q0 + ty * 4u + 3u, tokens - 1u), tokens, bidir, prefix);

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
                a = k[ra * kv_stride4 + kv_head * 16u + dv];
            }
            if ra + 1u < wg_limit {
                b = k[(ra + 1u) * kv_stride4 + kv_head * 16u + dv];
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
                a = k[ra * kv_stride4 + kv_head * 16u + dv];
            }
            if ra + 1u < wg_limit {
                b = k[(ra + 1u) * kv_stride4 + kv_head * 16u + dv];
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
                x = v[row * kv_stride4 + kv_head * 16u + dv];
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
                x = v[row * kv_stride4 + kv_head * 16u + dv];
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
                x = v[row * kv_stride4 + kv_head * 16u + dv];
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
                x = v[row * kv_stride4 + kv_head * 16u + dv];
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
    let inv0 = 1.0 / (lsum[ty * 4u] + lsum[32u + ty * 4u] + lsum[64u + ty * 4u] + lsum[96u + ty * 4u]);
    let inv1 = 1.0 / (lsum[ty * 4u + 1u] + lsum[32u + ty * 4u + 1u] + lsum[64u + ty * 4u + 1u] + lsum[96u + ty * 4u + 1u]);
    let inv2 = 1.0 / (lsum[ty * 4u + 2u] + lsum[32u + ty * 4u + 2u] + lsum[64u + ty * 4u + 2u] + lsum[96u + ty * 4u + 2u]);
    let inv3 = 1.0 / (lsum[ty * 4u + 3u] + lsum[32u + ty * 4u + 3u] + lsum[64u + ty * 4u + 3u] + lsum[96u + ty * 4u + 3u]);
    if r0 + 0u < tokens { out[(r0 + 0u) * q_stride4 + head * 16u + tx] = o0 * inv0; }
    if r0 + 1u < tokens { out[(r0 + 1u) * q_stride4 + head * 16u + tx] = o1 * inv1; }
    if r0 + 2u < tokens { out[(r0 + 2u) * q_stride4 + head * 16u + tx] = o2 * inv2; }
    if r0 + 3u < tokens { out[(r0 + 3u) * q_stride4 + head * 16u + tx] = o3 * inv3; }
}

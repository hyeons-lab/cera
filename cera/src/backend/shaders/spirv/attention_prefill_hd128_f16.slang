// Causal (with optional sliding window) and bidirectional-prefix flash attention prefill for
// head_dim == 128 with fp16 math, 32 queries per workgroup, reading the packed-f16 KV cache.
// Mirrors `attention_prefill_hd64_f16.slang`'s 5-binding / 14-word parameter block contract on a
// grid of `(ceil(n_sub / 32), n_heads)` 256-thread workgroups.
//
// Thread layout (256 threads = ty 0..7 x tx 0..31):
//   * QK^T: ty (0..7) = 4 queries, tx (0..31) = 1 key in the 32-key tile.
//   * Softmax: sp = t / 32 (0..7, 4 keys each), sq = t % 32 (0..31, 1 query each).
//   * PV: ty (0..7) = 4 queries, tx (0..31) = 4 output dims (`float4`, 32 * 4 = 128 dims).
//
// Workgroup storage: qs 8192 + kt 8192 + vs 8448 + ps 4096 + ph 2048 + pm 1024 + corr 128
// = 32,128 bytes (fits under the 32 KiB limit by sharing `pm` for final `l` sum).

[[vk::binding(0, 0)]] StructuredBuffer<float4> q_batch;
[[vk::binding(1, 0)]] StructuredBuffer<uint2> k_cache;   // packed f16, 4 dims per element
[[vk::binding(2, 0)]] StructuredBuffer<uint2> v_cache;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> out_batch;
[[vk::binding(4, 0)]] StructuredBuffer<uint> params;

static const float LOG2E = 1.4426950408889634;
static const float NEG = -1.0e30;

groupshared half4 qs[1024];     // [128 dims][8 query groups], 4 queries each
groupshared half kt[4096];      // [128 dims][32 keys]
groupshared half4 vs[1056];     // [32 keys][33 half4] (1 padding per row)
groupshared float ps[1024];     // [32 keys][32 queries]
groupshared half4 ph[256];      // [32 keys][8 query groups], 4 queries each
groupshared float pm[256];      // [8 key slices][32 queries] (reused at end for lsum)
groupshared float corr[32];

uint key_window(uint pos, uint bidir, uint prefix, uint total, uint max_seq) {
    uint w = pos + 1u;
    if (bidir != 0u) { w = pos < prefix ? prefix : total; }
    return min(w, max_seq);
}

uint win_min(uint lim, uint bidir, uint prefix) {
    return (bidir == 0u && prefix > 0u && lim > prefix) ? (lim - prefix) : 0u;
}

half4 half4_of(uint2 w) { return reinterpret<half4>(w); }

[numthreads(256, 1, 1)]
void main(uint3 wid : SV_GroupID, uint t : SV_GroupThreadID) {
    uint tx = t % 32u;
    uint ty = t / 32u;
    uint n_head = params[0];
    uint n_kv = params[1];
    uint kv_dim4 = params[3] / 4u;
    uint max_seq = params[4];
    float scale = asfloat(params[5]);
    uint start_pos = params[6];
    uint total = params[7];
    uint q_stride4 = params[8] / 4u;
    uint out_stride4 = params[9] / 4u;
    uint q_base = params[10];
    uint n_sub = params[11];
    uint bidir = params[12];
    uint prefix = params[13];
    uint head = wid.y;
    uint kv_head = head / (n_head / n_kv);
    uint q0 = wid.x * 32u;
    uint first_row = min(q0, n_sub - 1u);
    uint last_row = min(q0 + 31u, n_sub - 1u);
    uint wg_limit = key_window(start_pos + q_base + last_row, bidir, prefix, total, max_seq);
    uint wg_first_lim = key_window(start_pos + q_base + first_row, bidir, prefix, total, max_seq);
    uint wg_min = win_min(wg_first_lim, bidir, prefix);
    uint kb_start = wg_min & ~31u;

    {
        uint qg = t % 8u;
        uint dv = t / 8u;
        float s = scale * LOG2E;
        float4 r0 = q_batch[(q_base + min(q0 + qg * 4u + 0u, n_sub - 1u)) * q_stride4 + head * 32u + dv] * s;
        float4 r1 = q_batch[(q_base + min(q0 + qg * 4u + 1u, n_sub - 1u)) * q_stride4 + head * 32u + dv] * s;
        float4 r2 = q_batch[(q_base + min(q0 + qg * 4u + 2u, n_sub - 1u)) * q_stride4 + head * 32u + dv] * s;
        float4 r3 = q_batch[(q_base + min(q0 + qg * 4u + 3u, n_sub - 1u)) * q_stride4 + head * 32u + dv] * s;
        qs[(4u * dv + 0u) * 8u + qg] = half4(half(r0.x), half(r1.x), half(r2.x), half(r3.x));
        qs[(4u * dv + 1u) * 8u + qg] = half4(half(r0.y), half(r1.y), half(r2.y), half(r3.y));
        qs[(4u * dv + 2u) * 8u + qg] = half4(half(r0.z), half(r1.z), half(r2.z), half(r3.z));
        qs[(4u * dv + 3u) * 8u + qg] = half4(half(r0.w), half(r1.w), half(r2.w), half(r3.w));
    }

    uint lim0 = key_window(start_pos + q_base + min(q0 + ty * 4u + 0u, n_sub - 1u), bidir, prefix, total, max_seq);
    uint lim1 = key_window(start_pos + q_base + min(q0 + ty * 4u + 1u, n_sub - 1u), bidir, prefix, total, max_seq);
    uint lim2 = key_window(start_pos + q_base + min(q0 + ty * 4u + 2u, n_sub - 1u), bidir, prefix, total, max_seq);
    uint lim3 = key_window(start_pos + q_base + min(q0 + ty * 4u + 3u, n_sub - 1u), bidir, prefix, total, max_seq);
    uint min0 = win_min(lim0, bidir, prefix);
    uint min1 = win_min(lim1, bidir, prefix);
    uint min2 = win_min(lim2, bidir, prefix);
    uint min3 = win_min(lim3, bidir, prefix);

    float4 o0 = float4(0.0); float4 o1 = float4(0.0); float4 o2 = float4(0.0); float4 o3 = float4(0.0);
    uint sq = t % 32u;
    uint sp = t / 32u;
    float m = NEG;
    float l = 0.0;

    for (uint kb = kb_start; kb < wg_limit; kb += 32u) {
        [unroll] for (uint z = 0u; z < 4u; z++) {
            uint u = t + 256u * z;
            uint key = u / 32u;
            uint dv = u % 32u;
            uint ra = kb + key;
            half4 a = half4(0.0h);
            if (ra < wg_limit) { a = half4_of(k_cache[ra * kv_dim4 + kv_head * 32u + dv]); }
            kt[(4u * dv + 0u) * 32u + key] = a.x;
            kt[(4u * dv + 1u) * 32u + key] = a.y;
            kt[(4u * dv + 2u) * 32u + key] = a.z;
            kt[(4u * dv + 3u) * 32u + key] = a.w;
        }
        GroupMemoryBarrierWithGroupSync();

        float4 sf = float4(0.0);
        [loop] for (uint db = 0u; db < 128u; db += 8u) {
            half4 s = half4(0.0h);
            [unroll] for (uint k = 0u; k < 8u; k++) {
                half4 qv = qs[(db + k) * 8u + ty];
                half kv = kt[(db + k) * 32u + tx];
                s += qv * kv;
            }
            sf += float4(s);
        }
        uint key_g = kb + tx;
        uint pb = tx * 32u + ty * 4u;
        ps[pb + 0u] = (key_g >= min0 && key_g < lim0) ? sf.x : NEG;
        ps[pb + 1u] = (key_g >= min1 && key_g < lim1) ? sf.y : NEG;
        ps[pb + 2u] = (key_g >= min2 && key_g < lim2) ? sf.z : NEG;
        ps[pb + 3u] = (key_g >= min3 && key_g < lim3) ? sf.w : NEG;
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint z = 0u; z < 4u; z++) {
            uint idx = t + 256u * z;
            uint kj = idx / 32u;
            uint dv = idx % 32u;
            uint row = kb + kj;
            half4 x = half4(0.0h);
            if (row < wg_limit) { x = half4_of(v_cache[row * kv_dim4 + kv_head * 32u + dv]); }
            vs[kj * 33u + dv] = x;
        }

        float pmax = NEG;
        float x0 = ps[(sp * 4u + 0u) * 32u + sq]; pmax = max(pmax, x0);
        float x1 = ps[(sp * 4u + 1u) * 32u + sq]; pmax = max(pmax, x1);
        float x2 = ps[(sp * 4u + 2u) * 32u + sq]; pmax = max(pmax, x2);
        float x3 = ps[(sp * 4u + 3u) * 32u + sq]; pmax = max(pmax, x3);
        pm[sp * 32u + sq] = pmax;
        GroupMemoryBarrierWithGroupSync();

        float m_new = m;
        [unroll] for (uint i = 0u; i < 8u; i++) {
            m_new = max(m_new, pm[i * 32u + sq]);
        }
        float c = exp2(m - m_new);
        float psum = 0.0;
        float p0 = exp2(x0 - m_new); psum += p0;
        float p1 = exp2(x1 - m_new); psum += p1;
        float p2 = exp2(x2 - m_new); psum += p2;
        float p3 = exp2(x3 - m_new); psum += p3;
        ps[(sp * 4u + 0u) * 32u + sq] = p0;
        ps[(sp * 4u + 1u) * 32u + sq] = p1;
        ps[(sp * 4u + 2u) * 32u + sq] = p2;
        ps[(sp * 4u + 3u) * 32u + sq] = p3;
        l = l * c + psum;
        m = m_new;
        if (sp == 0u) { corr[sq] = c; }
        GroupMemoryBarrierWithGroupSync();

        {
            uint key = t / 8u;
            uint qg = t % 8u;
            ph[t] = half4(
                half(ps[key * 32u + qg * 4u + 0u]),
                half(ps[key * 32u + qg * 4u + 1u]),
                half(ps[key * 32u + qg * 4u + 2u]),
                half(ps[key * 32u + qg * 4u + 3u])
            );
        }
        GroupMemoryBarrierWithGroupSync();

        o0 *= corr[ty * 4u + 0u];
        o1 *= corr[ty * 4u + 1u];
        o2 *= corr[ty * 4u + 2u];
        o3 *= corr[ty * 4u + 3u];
        [loop] for (uint jb = 0u; jb < 32u; jb += 4u) {
            half4 a0 = half4(0.0h); half4 a1 = half4(0.0h); half4 a2 = half4(0.0h); half4 a3 = half4(0.0h);
            [unroll] for (uint k = 0u; k < 4u; k++) {
                half4 pv = ph[(jb + k) * 8u + ty];
                half4 vv = vs[(jb + k) * 33u + tx];
                a0 += pv.x * vv; a1 += pv.y * vv; a2 += pv.z * vv; a3 += pv.w * vv;
            }
            o0 += float4(a0); o1 += float4(a1); o2 += float4(a2); o3 += float4(a3);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    pm[sp * 32u + sq] = l;
    GroupMemoryBarrierWithGroupSync();
    uint r0 = q0 + ty * 4u;
    float s0 = pm[ty * 4u + 0u] + pm[32u + ty * 4u + 0u] + pm[64u + ty * 4u + 0u] + pm[96u + ty * 4u + 0u]
             + pm[128u + ty * 4u + 0u] + pm[160u + ty * 4u + 0u] + pm[192u + ty * 4u + 0u] + pm[224u + ty * 4u + 0u];
    float s1 = pm[ty * 4u + 1u] + pm[32u + ty * 4u + 1u] + pm[64u + ty * 4u + 1u] + pm[96u + ty * 4u + 1u]
             + pm[128u + ty * 4u + 1u] + pm[160u + ty * 4u + 1u] + pm[192u + ty * 4u + 1u] + pm[224u + ty * 4u + 1u];
    float s2 = pm[ty * 4u + 2u] + pm[32u + ty * 4u + 2u] + pm[64u + ty * 4u + 2u] + pm[96u + ty * 4u + 2u]
             + pm[128u + ty * 4u + 2u] + pm[160u + ty * 4u + 2u] + pm[192u + ty * 4u + 2u] + pm[224u + ty * 4u + 2u];
    float s3 = pm[ty * 4u + 3u] + pm[32u + ty * 4u + 3u] + pm[64u + ty * 4u + 3u] + pm[96u + ty * 4u + 3u]
             + pm[128u + ty * 4u + 3u] + pm[160u + ty * 4u + 3u] + pm[192u + ty * 4u + 3u] + pm[224u + ty * 4u + 3u];
    float i0 = s0 > 0.0 ? 1.0 / s0 : 0.0;
    float i1 = s1 > 0.0 ? 1.0 / s1 : 0.0;
    float i2 = s2 > 0.0 ? 1.0 / s2 : 0.0;
    float i3 = s3 > 0.0 ? 1.0 / s3 : 0.0;
    if (r0 + 0u < n_sub) { out_batch[(q_base + r0 + 0u) * out_stride4 + head * 32u + tx] = o0 * i0; }
    if (r0 + 1u < n_sub) { out_batch[(q_base + r0 + 1u) * out_stride4 + head * 32u + tx] = o1 * i1; }
    if (r0 + 2u < n_sub) { out_batch[(q_base + r0 + 2u) * out_stride4 + head * 32u + tx] = o2 * i2; }
    if (r0 + 3u < n_sub) { out_batch[(q_base + r0 + 3u) * out_stride4 + head * 32u + tx] = o3 * i3; }
}

// Append f32 rows to the packed-f16 KV cache: one thread per u32 word packs
// two f32 halves (`dst[off + w] = pack(src[2w], src[2w+1])`). One thread per
// WORD (not per float) so no two threads share a word. n is always even
// (kv_dim is even, counts are multiples of it — asserted host-side).
//
// The f32->f16 conversion is manual bit ops, not a `half` cast: WGSL has no
// f16 storage without SHADER_F16 (withheld on Adreno) and no half values at
// all in naga's core profile, so a `half` intermediate would not survive to
// the WGSL output. Round-half-up, subnormals flush to zero, overflow clamps
// to max-half (never Inf — an Inf K would NaN the softmax).
//
//   binding/buffer 0: src     f32, read (n)
//   binding/buffer 1: dst     u32, read-write (cache slab, packed halves)
//   binding/buffer 2: params  (off_words, n_floats, 0, 0)
//
// `[[vk::binding(n)]]` drives the WGSL binding index and `register(u0/t1/t2)` the
// Metal buffer index; they are numbered independently because the two backends
// count slots from different spaces, and here they agree.

[[vk::binding(0)]] StructuredBuffer<float> src_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<uint> dst_buf : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint4>  par_buf : register(t2);

uint to_f16_bits(float v) {
    uint b = asuint(v);
    uint s = (b >> 16u) & 0x8000u;
    int e = int((b >> 23u) & 0xFFu) - 112;
    uint m = b & 0x7FFFFFu;
    uint h;
    if (e >= 31) {
        h = 0x7BFFu;
    } else if (e <= 0) {
        h = 0u;
    } else {
        // (m + 0x1000) >> 13 can yield 0x400 on round-up; OR-ing that into
        // e << 10 carries correctly into (e+1) << 10.
        h = (uint(e) << 10u) | ((m + 0x1000u) >> 13u);
    }
    return s | h;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void kv_append(uint3 gid : SV_DispatchThreadID) {
    uint w  = gid.x;
    uint off = par_buf[0].x;
    uint n   = par_buf[0].y;
    if (w >= (n >> 1u)) {
        return;
    }
    uint i0 = w << 1u;
    dst_buf[off + w] = to_f16_bits(src_buf[i0]) | (to_f16_bits(src_buf[i0 + 1u]) << 16u);
}

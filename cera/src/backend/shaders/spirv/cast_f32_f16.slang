// Elementwise f32 -> f16 cast. Grid: ceil(len / 256); [numthreads(256, 1, 1)].

struct Params {
    uint len;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<half> dst;
[[vk::binding(2, 0)]] StructuredBuffer<Params> paramsBuf;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i < paramsBuf[0].len) {
        dst[i] = half(src[i]);
    }
}

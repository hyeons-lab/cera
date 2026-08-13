// Polar half-spectrum → complex, for the GPU ISTFT front stage.
//
// The detokenizer emits each frame as `[log_abs_0..log_abs_{bins-1},
// angle_0..angle_{bins-1}]` (the CPU `istft_to_pcm` reads exactly this layout).
// This kernel maps that to the interleaved real/imag half-spectrum the iDFT
// matmul consumes: `re_j = exp(log_abs_j)·cos(angle_j)`, `im_j =
// exp(log_abs_j)·sin(angle_j)`. Input and output share the per-frame stride
// `2·bins`, so the re block overwrites the log_abs block position and the im
// block overwrites the angle block position (in a distinct output buffer).
//
//   binding 0: spectrum  f32, read  (n_frames · 2·bins)
//   binding 1: out       f32, write (n_frames · 2·bins), [re | im] per frame
//   binding 2: params    (n_frames, bins)
//
// Dispatch: one thread per (frame, bin), ceil(n_frames·bins / 256) groups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   spec_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> out_buf  : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint2>   par_buf  : register(t2);

[shader("compute")]
[numthreads(256, 1, 1)]
void exp_polar(uint3 gid : SV_DispatchThreadID) {
    uint bins = par_buf[0].y;
    uint total = par_buf[0].x * bins;
    uint i = gid.x;
    if (i >= total) {
        return;
    }
    uint frame = i / bins;
    uint j = i % bins;
    uint base = frame * 2u * bins;
    float log_abs = spec_buf[base + j];
    float angle = spec_buf[base + bins + j];
    float mag = exp(log_abs);
    out_buf[base + j] = mag * cos(angle);
    out_buf[base + bins + j] = mag * sin(angle);
}

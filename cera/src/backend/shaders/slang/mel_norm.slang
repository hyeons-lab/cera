// Per-feature normalization of the log-mel spectrogram, with the mel-major to
// time-major transpose folded into the store. One workgroup per mel bin:
//
//   dst[t][m] = (mel[m][t] - mean_m) * inv_std_m   for t <  effective_n_len
//   dst[t][m] = 0                                  for t >= effective_n_len
//
// Mirrors the tail of `audio_preprocessor::log_mel_spectrogram` exactly:
//
// - The statistics come from the **effective** frame count only
//   (`n_samples / hop`, clamped to `n_frames`); frames past it are part of the
//   output but always zero. They are the trailing center-padding, and the C++
//   reference zeroes them rather than trimming the frame count.
// - The variance is the **unbiased** estimator (denominator `eff - 1`), with
//   `eps` added before the sqrt, not after.
// - `eff <= 1` zeroes the whole row. A single live frame centers to exactly 0
//   and its unbiased variance is undefined, so this is the reference's behaviour
//   and not a special case invented here.
//
// The transpose is free: the normalize pass already touches every element, so
// writing `dst[t * n_mel + m]` instead of `dst[m * n_frames + t]` costs a
// strided store and saves a whole dispatch plus an `[n_frames][n_mel]` scratch.
//
// Reductions run in f32 where the CPU uses f64. Both are the two-pass form
// (mean, then sum of squared deviations), which is the numerically stable one;
// the naive `E[x^2] - E[x]^2` shortcut would not survive the switch.
//
// The reduction divergence is `layernorm_batch`'s: metal reduces with a
// two-stage `simd_sum` (via `WaveActiveSum`), the portable path with a
// shared-memory tree, kept apart by `__target_switch` in `block_sum`. The
// barrier between the two reductions is present on both targets, since only the
// short `(mel - mean)^2` loop separates the first reduction's `scratch` reads
// from the second's writes.
//
//   binding/buffer 0: mel     f32, read  [n_mel][n_frames], mel-major
//   binding/buffer 1: dst     f32, write [n_frames][n_mel], time-major
//   binding/buffer 2: params  (n_mel, n_frames, effective_n_len, eps_bits)
//
// Dispatch: (n_mel, 1, 1) workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   mel_buf : register(t0);
[[vk::binding(1)]] RWStructuredBuffer<float> dst_buf : register(u1);
[[vk::binding(2)]] StructuredBuffer<uint4>   par_buf : register(t2);

// Shared scratch: metal uses 8 slots (one per simdgroup), the tree uses all 256.
groupshared float scratch[256];

static const uint WG = 256u;

/// Sum of `v` across the whole workgroup. Result valid on every thread. Metal
/// reduces with a two-stage `simd_sum` (via `WaveActiveSum`), the portable path
/// with a shared-memory tree.
float block_sum(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        float sg = WaveActiveSum(v);
        if ((tid & 31u) == 0u) { scratch[tid >> 5u] = sg; }
        GroupMemoryBarrierWithGroupSync();
        float lane = (tid < 8u) ? scratch[tid] : 0.0f;
        float total = WaveActiveSum(lane);
        if (tid == 0u) { scratch[0] = total; }
        GroupMemoryBarrierWithGroupSync();
        result = scratch[0];
        break;
    }
    default:
    {
        scratch[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) { scratch[tid] += scratch[tid + s]; }
            GroupMemoryBarrierWithGroupSync();
        }
        result = scratch[0];
        break;
    }
    }
    return result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mel_norm(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID) {
    uint tid      = lid.x;
    uint mi       = wid.x;
    uint n_mel    = par_buf[0].x;
    uint n_frames = par_buf[0].y;
    uint eff      = par_buf[0].z;
    float eps     = asfloat(par_buf[0].w);
    uint row      = mi * n_frames;

    // Uniform across the workgroup, so returning here skips no barrier some
    // other thread still waits on.
    if (eff <= 1u) {
        for (uint t = tid; t < n_frames; t += WG) {
            dst_buf[t * n_mel + mi] = 0.0f;
        }
        return;
    }

    // Pass 1: mean over the live frames.
    float partial = 0.0f;
    for (uint t = tid; t < eff; t += WG) {
        partial += mel_buf[row + t];
    }
    float mean = block_sum(tid, partial) / float(eff);

    // Separate the two reductions' use of `scratch` (kept on both targets).
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: unbiased variance over the same frames.
    partial = 0.0f;
    for (uint t = tid; t < eff; t += WG) {
        float d = mel_buf[row + t] - mean;
        partial += d * d;
    }
    float inv_std = 1.0f / sqrt(block_sum(tid, partial) / float(eff - 1u) + eps);

    // Pass 3: normalize the live frames, zero the tail, transpose on the way out.
    for (uint t = tid; t < n_frames; t += WG) {
        float v = (t < eff) ? (mel_buf[row + t] - mean) * inv_std : 0.0f;
        dst_buf[t * n_mel + mi] = v;
    }
}

// Reduction epilogue for Split-K GEMV partials:
//   y[row] = sum_{s=0..k_splits-1} partial_y[s * m + row]
//
// Bindings:
//   0 = partial_y (f32), 1 = y (rw f32), 2 = params (uint4: m, k, k_splits, unused).
// Dispatch: (ceil(m / 64), 1, 1).

[[vk::binding(0, 0)]] StructuredBuffer<float>   partial_y;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(2, 0)]] StructuredBuffer<uint4>   params;

[shader("compute")]
[WaveSize(32)]
[numthreads(64, 1, 1)]
void main(
    uint3 lid : SV_GroupThreadID,
    uint3 wid : SV_GroupID
) {
    uint m = params[0].x;
    uint k_splits = params[0].z;
    uint row = wid.x * 64u + lid.x;
    if (row >= m) {
        return;
    }
    float sum = 0.0f;
    for (uint s = 0u; s < k_splits; s++) {
        sum += partial_y[s * m + row];
    }
    y[row] = sum;
}

// slang-entries: moe_gemv_q4_0
//
// Expert-indexed Q4_0 GEMV: one row of one *routed* expert's weight matrix,
// dotted with one token's activation.
//
// The reason this is not just `gemv_q4_0` with a buffer offset is that the
// offset is not known on the host. Routing happens on the GPU (`moe_route`), so
// the expert id lives in a device buffer; binding the right slice would mean
// reading it back, and a blocking round trip costs ~1.3-1.5ms whatever it
// carries. At 22 routed layers that is ~30ms *per token* of pure stall, which
// would dominate the ~2ms of arithmetic the model actually needs. So the kernel
// reads `sel_expert` itself and does the slice arithmetic in-shader, which is
// the same indirection llama.cpp's `mul_mat_id` uses.
//
// ## One entry per matrix, and one threadgroup per (row, entry)
//
// An "entry" is one (token, slot) pair: with `n_used = 4` a token has four of
// them, each bound to a different expert. The dispatch grid is
// (rows, n_entries), so every threadgroup reads its expert's weight row exactly
// once. For decode (`n_tokens = 1`) that is the whole story and the traffic is
// optimal: `n_used x m x row_bytes`.
//
// For prefill it is *not* optimal, and deliberately so. Tokens in a chunk that
// route to the same expert do not share the weight row here, so the matrix is
// re-read once per (token, slot) instead of once per expert. Fixing that means
// grouping tokens by expert and running a tiled GEMM per group (a bucket sort
// plus `mul_mat_id`-shaped tiling); it is the right end state and it is a much
// larger change. This kernel gets the architecture running on GPU with the same
// token-at-a-time shape the CPU backend already has, so all three backends can
// be pinned to each other first. See the devlog for the grouped-GEMM follow-up.
//
// Bindings:
//   t0: w           u32, the *stacked* expert weight, `[n_expert][m][k]` Q4_0.
//                   Bound at the tensor's own base; the expert slice offset is
//                   applied in-shader.
//   t1: x           f32, activations, `[rows_in][k]` row-major
//   u2: y           f32, `[n_entries][m]` row-major
//   t3: sel_expert  u32, `[n_entries]` from `moe_route`
//   t4: params      p[0] = (m, k, n_used, n_entries)
//                   p[1] = (expert_stride_bytes, x_by_entry, _, _)
//
// `x_by_entry` selects which row of `x` an entry reads. The gate and up
// projections consume the *token's* hidden state, shared by all of that token's
// slots (`x_by_entry = 0`, row = entry / n_used). The down projection consumes
// the per-slot SwiGLU product, which is distinct for every entry
// (`x_by_entry = 1`, row = entry). Getting this backwards makes every slot of a
// token compute the same expert output, which still produces plausible text.
//
// Dispatch: (m, n_entries) threadgroups of 32.

[[vk::binding(0)]] StructuredBuffer<uint>    w          : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   x          : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> y          : register(u2);
[[vk::binding(3)]] StructuredBuffer<uint>    sel_expert : register(t3);
[[vk::binding(4)]] StructuredBuffer<uint4>   params     : register(t4);

static const uint WG = 32u;

/// Q4_0 block: an f16 scale followed by 16 bytes of packed nibbles, 32 weights
/// in 18 bytes. Never a struct: 18 is not a multiple of 4, so any struct view
/// would be padded and the row stride would stop matching the file layout.
static const uint BLOCK_BYTES = 18u;

groupshared float scratch[WG];

/// Sum of `v` across the threadgroup. Result valid on thread 0 only, which is
/// all the caller needs. Metal reduces with `simd_sum` (the threadgroup is
/// exactly one simdgroup); the portable path walks a shared-memory tree.
float block_sum(uint tid, float v) {
    float result;
    __target_switch {
    case metal:
    {
        result = WaveActiveSum(v);
        break;
    }
    default:
    {
        scratch[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s >>= 1) {
            if (tid < s) {
                scratch[tid] += scratch[tid + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        result = scratch[0];
        break;
    }
    }
    return result;
}

/// `byte_off` must be 2-byte aligned. Every Q4_0 block start (and every even
/// byte inside `qs`) is: blocks are 18 bytes, rows are a whole number of
/// blocks, and an expert slice is a whole number of rows.
uint load_u16(uint byte_off) {
    uint word = w[byte_off >> 2u];
    return ((byte_off & 2u) != 0u) ? (word >> 16u) : (word & 0xFFFFu);
}

float load_f16(uint byte_off) {
    return f16tof32(load_u16(byte_off));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void moe_gemv_q4_0(uint3 lid : SV_GroupThreadID, uint3 grp : SV_GroupID) {
    uint m = params[0].x;
    uint k = params[0].y;
    uint n_used = max(params[0].z, 1u);
    uint n_entries = params[0].w;
    uint expert_stride = params[1].x;
    bool x_by_entry = params[1].y != 0u;

    uint row = grp.x;
    uint entry = grp.y;
    // Uniform across the threadgroup, so the `block_sum` tree below still sees
    // all 32 threads on the portable path.
    if (row >= m || entry >= n_entries) {
        return;
    }

    uint x_row = x_by_entry ? entry : (entry / n_used);
    uint x_base = x_row * k;

    uint nb = k / 32u;
    uint row_bytes = nb * BLOCK_BYTES;
    uint base = sel_expert[entry] * expert_stride + row * row_bytes;

    float sum = 0.0f;
    for (uint bi = lid.x; bi < nb; bi += WG) {
        uint blk = base + bi * BLOCK_BYTES;
        uint col = x_base + bi * 32u;

        // Accumulate the unsigned nibble products and activation sum first,
        // then apply the -8 zero-point offset and block scale once per block.
        float acc = 0.0f;
        float xsum = 0.0f;
        [unroll]
        for (uint i = 0u; i < 8u; ++i) {
            uint q16 = load_u16(blk + 2u + i * 2u);
            uint idx = col + i * 2u;
            float x0 = x[idx];
            float x1 = x[idx + 1u];
            float x16 = x[idx + 16u];
            float x17 = x[idx + 17u];
            xsum += (x0 + x1) + (x16 + x17);
            acc += float(q16 & 0xFu) * x0
                 + float((q16 >> 8u) & 0xFu) * x1
                 + float((q16 >> 4u) & 0xFu) * x16
                 + float(q16 >> 12u) * x17;
        }
        sum += (acc - 8.0f * xsum) * load_f16(blk);
    }

    float total = block_sum(lid.x, sum);
    if (lid.x == 0u) {
        y[entry * m + row] = total;
    }
}

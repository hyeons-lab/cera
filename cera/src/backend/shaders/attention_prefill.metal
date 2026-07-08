#include <metal_stdlib>
using namespace metal;

// Iter 6 — Query-block (QPT) templating on top of Iter 5 (hd-specialized MMA).
//
// The body is `template<uint HD_CONST, uint QPT>`. HD_CONST specializes the
// head_dim inner-loop bounds (see Iter 5). QPT is the number of query rows a
// threadgroup owns; it must be a multiple of 8 (the simdgroup-matrix tile
// height). QT = QPT / 8 is the number of query-row-tiles each threadgroup
// processes per K/V chunk.
//
// Why QPT matters: the outer dispatch launches ceil(n / QPT) * n_heads
// threadgroups, and each threadgroup streams *all* preceding K/V from device
// memory in C-sized chunks. So a K/V column near the prompt start is re-read
// from device by every query-group that comes after it. Total K/V device bytes
// per head ≈ 2·hd·n²/QPT — inversely proportional to the query block. Iters
// #20/#22/#23/#129 tuned the attention *compute* (MMA, fp16 staging, constexpr
// hd) but never changed QPT (fixed at 8). Raising QPT amortizes each K/V device
// read over more queries, cutting the O(n²) bandwidth that dominates long
// prompts. The cost is threadgroup memory (scores/out_tg grow with QPT) and
// thus occupancy — measured, not assumed.
//
// Entry points:
//   `attention_prefill`        — runtime fallback (HD_CONST=0, QPT=8)
//   `attention_prefill_hd64`   — LFM2-VL-450M / LFM2.5-VL-450M   (hd=64,  QPT=8)
//   `attention_prefill_hd128`  — LFM2.5-VL-1.6B / LFM2.5-Audio-1.5B (hd=128, QPT=8)
//   `attention_prefill_hd64_q16` — hd=64, QPT=16 (18.2 KB shmem)
//   `attention_prefill_hd64_q32` — hd=64, QPT=32 (28.4 KB shmem)
// Host-side dispatch (`encode_attention_prefill_batch`) picks the variant by
// (head_dim, prefill_qpt). hd=128 can't grow past QPT=8: QPT=16 needs 32.2 KB,
// over M1's 32 KB threadgroup-memory cap.
//
// Iter 4/5 background (carried over):
//   `C` is 64; Q, K, V stage as half in threadgroup memory; the score matrix
//   and output accumulator remain fp32 (standard flash-attention precision
//   shape). Mixed-precision MMA overloads (confirmed supported on M1+):
//     - QK^T:  simdgroup_multiply_accumulate(float8x8, half8x8, half8x8, float8x8)
//     - AV:    simdgroup_multiply_accumulate(float8x8, float8x8, half8x8, float8x8)
//   Softmax reduces over 64 cells per query but the simdgroup is only 32 lanes
//   wide, so each lane owns two cells (l and l+32).
//
// Constraint (unchanged): head_dim % 8 == 0, head_dim <= 256. QPT % 8 == 0.

constant constexpr uint C = 64;
constant constexpr uint NW = 32;
constant constexpr uint NSG = 8;
constant constexpr uint N_THREADS = NSG * NW; // 256

struct PrefillAttnParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint start_pos;
    uint n_queries;
    uint scale_bits;
    uint q_stride;
    uint out_stride;
};

// Templated helper. HD_CONST>0 folds `hd` to a literal (Iter 5). QPT is the
// query-block height (multiple of 8); QT=QPT/8 query-row-tiles per chunk.
template<uint HD_CONST, uint QPT>
inline void attention_prefill_impl(
    const device float* q_batch,
    const device half*  k_cache,
    const device half*  v_cache,
    device float*       out_batch,
    constant PrefillAttnParams& params,
    threadgroup char*   shmem,
    uint tid,
    uint tg_idx
) {
    const uint n_heads = params.n_heads;
    const uint n_kv_heads = params.n_kv_heads;
    const uint hd = (HD_CONST > 0) ? HD_CONST : params.head_dim;
    const uint QT = QPT / 8u;  // query-row-tiles per threadgroup
    const uint kv_dim = params.kv_dim;
    const uint start_pos = params.start_pos;
    const uint n_queries = params.n_queries;
    const float scale = as_type<float>(params.scale_bits);

    const uint head = tg_idx % n_heads;
    const uint q_group = tg_idx / n_heads;
    const uint q_base = q_group * QPT;
    const uint group_size = n_heads / n_kv_heads;
    const uint kv_head = head / group_size;
    const uint kv_h_off = kv_head * hd;

    const uint n_q = min(QPT, n_queries - q_base);
    const uint max_seq = start_pos + q_base + n_q;

    // TG memory layout:
    //   q_tg    : half  [QPT × hd]
    //   kv_tile : half  [C × hd]            (K first, overwritten by V)
    //   scores  : float [QPT × C]
    //   out_tg  : float [QPT × hd]          (running softmax-weighted V sum)
    //   state   : float [QPT × 2]           (per-query max, sum)
    //   rescales: float [QPT]
    // The `half*` → `float*` cast is safe because (QPT + C) * hd * 2 bytes is a
    // multiple of 4: (QPT + C) is even (both multiples of 8) and hd is divisible
    // by 8, so the product is a multiple of 4.
    threadgroup half*  q_tg     = (threadgroup half*)(shmem);
    threadgroup half*  kv_tile  = q_tg + QPT * hd;
    threadgroup float* scores   = (threadgroup float*)(kv_tile + C * hd);
    threadgroup float* out_tg   = scores + QPT * C;
    threadgroup float* state    = out_tg + QPT * hd;
    threadgroup float* rescales = state + QPT * 2;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;

    // --- Load Q + init output accumulators (cooperative) ---
    //
    // out_tg is zeroed for the full QPT × hd (not just n_q × hd): Step B's
    // pre-rescale reads all QPT rows, threadgroup memory isn't zero on entry,
    // and NaN × 0 = NaN would leave unused rows as NaN. q_tg for q >= n_q stays
    // uninitialized; any garbage the scoring MMA produces on those rows is
    // scrubbed by the softmax else-branch (writes 0 to every cell of rows
    // q >= n_q) before the V MMA reads `scores`.
    for (uint idx = tid; idx < n_q * hd; idx += N_THREADS) {
        uint q = idx / hd;
        uint d = idx % hd;
        q_tg[q * hd + d] = half(q_batch[(q_base + q) * params.q_stride + head * hd + d]);
    }
    for (uint idx = tid; idx < QPT * hd; idx += N_THREADS) {
        out_tg[idx] = 0.0f;
    }
    if (tid < QPT) {
        state[tid * 2 + 0] = -INFINITY;
        state[tid * 2 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Outer chunk loop (online softmax) ---
    for (uint c0 = 0; c0 < max_seq; c0 += C) {
        const uint c_end = min(c0 + C, max_seq);
        const uint c_len = c_end - c0;

        // --- Load K tile into TG memory (cooperative, half-precision) ---
        //
        // MMA reads the full C×hd tile regardless of c_len, so tail rows
        // (t >= c_len on the last chunk) must be zeroed to avoid
        // 0 × uninitialized = NaN propagation.
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = k_cache[(c0 + t) * kv_dim + kv_h_off + d];
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA QK scoring (all 8 SGs, QT row-tiles each) ---
        //
        // Score matrix [QPT × C] decomposes into QT row-tiles × 8 col-tiles.
        // SG `simd_id` owns col-tile `simd_id` and sweeps all QT row-tiles, so
        // one K tile load in shmem feeds every query row (that's the point of a
        // larger QPT — the device K read is shared across QPT queries).
        {
            const uint hd_tiles = hd / 8u;
            const uint t_tile = simd_id;
            for (uint q_tile = 0u; q_tile < QT; q_tile++) {
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8  q_mat;
                simdgroup_half8x8  k_mat;

                for (uint d_tile = 0u; d_tile < hd_tiles; d_tile++) {
                    // Q[q_tile*8 .. +8, d_tile*8 .. +8], stride = hd, no transpose.
                    simdgroup_load(q_mat, q_tg + q_tile * 8u * hd + d_tile * 8u, hd);

                    // K with transpose=true loads K^T[d_tile*8..+8, t_tile*8..+8].
                    // `origin` is `ulong2` per MSL's simdgroup_load signature on
                    // this toolchain (uint2 fails compilation).
                    simdgroup_load(k_mat,
                                   kv_tile + t_tile * 8u * hd + d_tile * 8u,
                                   hd,
                                   /*origin*/ ulong2(0, 0),
                                   /*transpose*/ true);

                    simdgroup_multiply_accumulate(acc, q_mat, k_mat, acc);
                }

                simdgroup_store(acc, scores + q_tile * 8u * C + t_tile * 8u, C);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Scale + causal mask (per-lane fix-up of the score matrix) ---
        //
        // The MMA produced raw QK. Apply `* scale` and the triangular mask
        // in one cooperative pass over n_q × c_len entries.
        for (uint idx = tid; idx < n_q * c_len; idx += N_THREADS) {
            uint q = idx / c_len;
            uint t = idx % c_len;
            uint seq_len_q = start_pos + q_base + q + 1;
            float s = scores[q * C + t];
            if (c0 + t >= seq_len_q) {
                s = -INFINITY;
            } else {
                s = s * scale;
            }
            scores[q * C + t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Overwrite kv_tile with V values (half-precision) ---
        for (uint idx = tid; idx < c_len * hd; idx += N_THREADS) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = v_cache[(c0 + t) * kv_dim + kv_h_off + d];
        }
        for (uint idx = tid + c_len * hd; idx < C * hd; idx += N_THREADS) {
            kv_tile[idx] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Per-query softmax (one SG per query, strided over QPT) ---
        //
        // NSG=8 simdgroups, QPT queries: SG `simd_id` handles queries
        // {simd_id, simd_id+8, ...}. Simdgroup has 32 lanes but C=64 cells per
        // query, so lane `l` owns cells `l` and `l + 32`; lane-local max/sum
        // folds both cells before the cross-lane simd_max / simd_sum.
        for (uint q = simd_id; q < QPT; q += NSG) {
            if (q < n_q) {
                const uint idx0 = simd_lane;
                const uint idx1 = simd_lane + NW;

                float s0 = (idx0 < c_len) ? scores[q * C + idx0] : -INFINITY;
                float s1 = (idx1 < c_len) ? scores[q * C + idx1] : -INFINITY;

                float chunk_max = simd_max(max(s0, s1));

                float prev_max = state[q * 2 + 0];
                float new_max = max(prev_max, chunk_max);
                float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;

                float e0 = 0.0f;
                float e1 = 0.0f;
                if (idx0 < c_len) {
                    e0 = exp(s0 - new_max);
                    scores[q * C + idx0] = e0;
                } else {
                    scores[q * C + idx0] = 0.0f;
                }
                if (idx1 < c_len) {
                    e1 = exp(s1 - new_max);
                    scores[q * C + idx1] = e1;
                } else {
                    scores[q * C + idx1] = 0.0f;
                }
                float chunk_sum = simd_sum(e0 + e1);

                if (simd_lane == 0u) {
                    state[q * 2 + 0] = new_max;
                    state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
                    rescales[q] = rescale;
                }
            } else {
                // Unused query row (q >= n_q): zero both cells per lane so the
                // scores row is fully zero (MMA then produces 0 × V = 0), and
                // zero rescales[q] so the pre-MMA rescale of out_tg is also 0.
                scores[q * C + simd_lane]        = 0.0f;
                scores[q * C + simd_lane + NW]   = 0.0f;
                if (simd_lane == 0u) {
                    rescales[q] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA V accumulation ---
        //
        // Pre-rescale out_tg by per-query rescales[q] (cooperative over all QPT
        // rows), then fuse V MMA with the add via po pre-loaded with rescaled
        // out_tg:  po_new = scores × V + po  where po = rescales[q] · out_tg
        for (uint idx = tid; idx < QPT * hd; idx += N_THREADS) {
            uint q = idx / hd;
            out_tg[idx] = out_tg[idx] * rescales[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SG owns a dim-tile (8 output cols); round-robin across the hd/8
        // tiles, for each of the QT query-row-tiles.
        const uint dim_tiles = hd / 8u;
        const uint inner_tiles = C / 8u;  // C=64 → 8 MMA inner iterations
        for (uint q_tile = 0u; q_tile < QT; q_tile++) {
            for (uint d_off = simd_id; d_off < dim_tiles; d_off += NSG) {
                simdgroup_float8x8 po;
                simdgroup_load(po, out_tg + q_tile * 8u * hd + d_off * 8u, hd);

                simdgroup_float8x8 s_mat;
                simdgroup_half8x8  v_mat;
                for (uint t_in = 0u; t_in < inner_tiles; t_in++) {
                    simdgroup_load(s_mat, scores + q_tile * 8u * C + t_in * 8u, C);
                    simdgroup_load(v_mat,
                                   kv_tile + t_in * 8u * hd + d_off * 8u,
                                   hd);
                    simdgroup_multiply_accumulate(po, s_mat, v_mat, po);
                }
                simdgroup_store(po, out_tg + q_tile * 8u * hd + d_off * 8u, hd);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Final normalization + write-out ---
    for (uint q = 0; q < n_q; q++) {
        float inv_sum = 1.0f / state[q * 2 + 1];
        for (uint d = tid; d < hd; d += N_THREADS) {
            uint out_idx = (q_base + q) * params.out_stride + head * hd + d;
            out_batch[out_idx] = out_tg[q * hd + d] * inv_sum;
        }
    }
}

// === Entry points =========================================================
//
// Shared body; the runtime kernel (`attention_prefill`) handles any head_dim,
// the specialized kernels constant-propagate (head_dim, QPT).

kernel void attention_prefill(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    attention_prefill_impl<0, 8>(q_batch, k_cache, v_cache, out_batch, params,
                                 shmem, tid, tg_idx);
}

kernel void attention_prefill_hd64(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    attention_prefill_impl<64, 8>(q_batch, k_cache, v_cache, out_batch, params,
                                  shmem, tid, tg_idx);
}

kernel void attention_prefill_hd128(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    attention_prefill_impl<128, 8>(q_batch, k_cache, v_cache, out_batch, params,
                                   shmem, tid, tg_idx);
}

kernel void attention_prefill_hd64_q16(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    attention_prefill_impl<64, 16>(q_batch, k_cache, v_cache, out_batch, params,
                                   shmem, tid, tg_idx);
}

kernel void attention_prefill_hd64_q32(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    attention_prefill_impl<64, 32>(q_batch, k_cache, v_cache, out_batch, params,
                                   shmem, tid, tg_idx);
}

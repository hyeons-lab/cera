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
template<uint HD_CONST, uint QPT, uint C_CHUNK = 64>
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

    const uint n_threads = (C_CHUNK == 32) ? 128u : 256u;
    const uint n_sg = n_threads / 32u;

    // TG memory layout:
    //   q_tg    : half  [QPT × hd]
    //   kv_tile : half  [C_CHUNK × hd]            (K first, overwritten by V)
    //   scores  : float [QPT × C_CHUNK]
    //   out_tg  : float [QPT × hd]          (running softmax-weighted V sum)
    //   state   : float [QPT × 2]           (per-query max, sum)
    //   rescales: float [QPT]
    threadgroup half*  q_tg     = (threadgroup half*)(shmem);
    threadgroup half*  kv_tile  = q_tg + QPT * hd;
    threadgroup float* scores   = (threadgroup float*)(kv_tile + C_CHUNK * hd);
    threadgroup float* out_tg   = scores + QPT * C_CHUNK;
    threadgroup float* state    = out_tg + QPT * hd;
    threadgroup float* rescales = state + QPT * 2;

    const uint simd_lane = tid & 31u;
    const uint simd_id = tid >> 5u;

    // --- Load Q + init output accumulators (cooperative) ---
    for (uint idx = tid; idx < n_q * hd; idx += n_threads) {
        uint q = idx / hd;
        uint d = idx % hd;
        q_tg[q * hd + d] = half(q_batch[(q_base + q) * params.q_stride + head * hd + d]);
    }
    for (uint idx = tid; idx < QPT * hd; idx += n_threads) {
        out_tg[idx] = 0.0f;
    }
    if (tid < QPT) {
        state[tid * 2 + 0] = -INFINITY;
        state[tid * 2 + 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Outer chunk loop (online softmax) ---
    for (uint c0 = 0; c0 < max_seq; c0 += C_CHUNK) {
        const uint c_end = min(c0 + C_CHUNK, max_seq);
        const uint c_len = c_end - c0;

        // --- Load K tile into TG memory (cooperative, half-precision) ---
        for (uint idx = tid; idx < c_len * hd; idx += n_threads) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = k_cache[(c0 + t) * kv_dim + kv_h_off + d];
        }
        for (uint idx = tid + c_len * hd; idx < C_CHUNK * hd; idx += n_threads) {
            kv_tile[idx] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA QK scoring (all n_sg SGs, QT row-tiles each) ---
        {
            const uint hd_tiles = hd / 8u;
            const uint t_tile = simd_id;
            for (uint q_tile = 0u; q_tile < QT; q_tile++) {
                simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8  q_mat;
                simdgroup_half8x8  k_mat;

                for (uint d_tile = 0u; d_tile < hd_tiles; d_tile++) {
                    simdgroup_load(q_mat, q_tg + q_tile * 8u * hd + d_tile * 8u, hd);
                    simdgroup_load(k_mat,
                                   kv_tile + t_tile * 8u * hd + d_tile * 8u,
                                   hd,
                                   /*origin*/ ulong2(0, 0),
                                   /*transpose*/ true);

                    simdgroup_multiply_accumulate(acc, q_mat, k_mat, acc);
                }

                simdgroup_store(acc, scores + q_tile * 8u * C_CHUNK + t_tile * 8u, C_CHUNK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Scale + causal mask ---
        for (uint idx = tid; idx < n_q * c_len; idx += n_threads) {
            uint q = idx / c_len;
            uint t = idx % c_len;
            uint seq_len_q = start_pos + q_base + q + 1;
            float s = scores[q * C_CHUNK + t];
            if (c0 + t >= seq_len_q) {
                s = -INFINITY;
            } else {
                s = s * scale;
            }
            scores[q * C_CHUNK + t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Overwrite kv_tile with V values (half-precision) ---
        for (uint idx = tid; idx < c_len * hd; idx += n_threads) {
            uint t = idx / hd;
            uint d = idx % hd;
            kv_tile[t * hd + d] = v_cache[(c0 + t) * kv_dim + kv_h_off + d];
        }
        for (uint idx = tid + c_len * hd; idx < C_CHUNK * hd; idx += n_threads) {
            kv_tile[idx] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Per-query softmax ---
        for (uint q = simd_id; q < QPT; q += n_sg) {
            if (q < n_q) {
                if (C_CHUNK == 32) {
                    const uint idx0 = simd_lane;
                    float s0 = (idx0 < c_len) ? scores[q * 32 + idx0] : -INFINITY;
                    float chunk_max = simd_max(s0);
                    float prev_max = state[q * 2 + 0];
                    float new_max = max(prev_max, chunk_max);
                    float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;
                    float e0 = (idx0 < c_len) ? exp(s0 - new_max) : 0.0f;
                    scores[q * 32 + idx0] = e0;
                    float chunk_sum = simd_sum(e0);
                    if (simd_lane == 0u) {
                        state[q * 2 + 0] = new_max;
                        state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
                        rescales[q] = rescale;
                    }
                } else {
                    const uint idx0 = simd_lane;
                    const uint idx1 = simd_lane + NW;

                    float s0 = (idx0 < c_len) ? scores[q * C_CHUNK + idx0] : -INFINITY;
                    float s1 = (idx1 < c_len) ? scores[q * C_CHUNK + idx1] : -INFINITY;

                    float chunk_max = simd_max(max(s0, s1));

                    float prev_max = state[q * 2 + 0];
                    float new_max = max(prev_max, chunk_max);
                    float rescale = (prev_max > -INFINITY) ? exp(prev_max - new_max) : 0.0f;

                    float e0 = 0.0f;
                    float e1 = 0.0f;
                    if (idx0 < c_len) {
                        e0 = exp(s0 - new_max);
                        scores[q * C_CHUNK + idx0] = e0;
                    } else {
                        scores[q * C_CHUNK + idx0] = 0.0f;
                    }
                    if (idx1 < c_len) {
                        e1 = exp(s1 - new_max);
                        scores[q * C_CHUNK + idx1] = e1;
                    } else {
                        scores[q * C_CHUNK + idx1] = 0.0f;
                    }
                    float chunk_sum = simd_sum(e0 + e1);

                    if (simd_lane == 0u) {
                        state[q * 2 + 0] = new_max;
                        state[q * 2 + 1] = state[q * 2 + 1] * rescale + chunk_sum;
                        rescales[q] = rescale;
                    }
                }
            } else {
                scores[q * C_CHUNK + simd_lane] = 0.0f;
                if (C_CHUNK > 32) {
                    scores[q * C_CHUNK + simd_lane + NW] = 0.0f;
                }
                if (simd_lane == 0u) {
                    rescales[q] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- MMA V accumulation ---
        for (uint idx = tid; idx < QPT * hd; idx += n_threads) {
            uint q = idx / hd;
            out_tg[idx] = out_tg[idx] * rescales[q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint dim_tiles = hd / 8u;
        const uint inner_tiles = C_CHUNK / 8u;
        for (uint q_tile = 0u; q_tile < QT; q_tile++) {
            for (uint d_off = simd_id; d_off < dim_tiles; d_off += n_sg) {
                simdgroup_float8x8 po;
                simdgroup_load(po, out_tg + q_tile * 8u * hd + d_off * 8u, hd);

                simdgroup_float8x8 s_mat;
                simdgroup_half8x8  v_mat;
                for (uint t_in = 0u; t_in < inner_tiles; t_in++) {
                    simdgroup_load(s_mat, scores + q_tile * 8u * C_CHUNK + t_in * 8u, C_CHUNK);
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
        for (uint d = tid; d < hd; d += n_threads) {
            uint out_idx = (q_base + q) * params.out_stride + head * hd + d;
            out_batch[out_idx] = out_tg[q * hd + d] * inv_sum;
        }
    }
}

// === Entry points =========================================================

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
    attention_prefill_impl<0, 8, 64>(q_batch, k_cache, v_cache, out_batch, params,
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
    attention_prefill_impl<64, 8, 64>(q_batch, k_cache, v_cache, out_batch, params,
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
    attention_prefill_impl<128, 8, 64>(q_batch, k_cache, v_cache, out_batch, params,
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
    attention_prefill_impl<64, 16, 64>(q_batch, k_cache, v_cache, out_batch, params,
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
    attention_prefill_impl<64, 32, 64>(q_batch, k_cache, v_cache, out_batch, params,
                                       shmem, tid, tg_idx);
}

kernel void attention_prefill_hd128_q16(
    const device float* q_batch [[buffer(0)]],
    const device half*  k_cache [[buffer(1)]],
    const device half*  v_cache [[buffer(2)]],
    device float* out_batch [[buffer(3)]],
    constant PrefillAttnParams& params [[buffer(4)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]]
) {
    attention_prefill_impl<128, 16, 32>(q_batch, k_cache, v_cache, out_batch, params,
                                        shmem, tid, tg_idx);
}

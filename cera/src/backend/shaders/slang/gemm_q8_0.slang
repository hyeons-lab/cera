// Q8_0 GEMM: `dst = src0 * src1`, one Slang source for both Metal and WGSL.
//
// This is the first *production-shaped* kernel in the Slang tree. `softmax.slang`
// showed a reduction can be shared; `coopmat_probe.slang` showed `linalg::CoopMat`
// reaches Metal's `simdgroup_matrix` at all. This one asks the question that
// actually decides whether a migration is worth doing: can Slang express the
// hand-tuned simdgroup GEMM (`shaders/gemm_q8_0.metal`) without giving up its
// layout, its threadgroup budget, or its speed?
//
// ## What is shared and what is not
//
// The two branches share the binding contract and `GemmParams`. They share
// nothing else, and that is not a failure of the port: the Metal path is 8x8
// simdgroup tiling, and the portable path on wgpu is a register-tiled kernel
// (`shaders/mul_mat_reg_tile.wgsl`) with an entirely different blocking
// strategy. `__target_switch` is what makes that honest rather than a lie by
// `#ifdef`: the untaken branch is eliminated before entry-point capability
// validation, so the `case metal:` body may use a capability WGSL cannot
// express at all.
//
// The `default:` branch here is a plain correctness reference, one dot product
// per output element with no tiling and no reuse. It exists so the WGSL half of
// `tests/slang_multitarget_parity.rs` has something to check and so the emitted
// WGSL is a legal shader; it is *not* a replacement for `mul_mat_reg_tile.wgsl`
// and nothing dispatches it. Do not benchmark it and conclude anything.
//
// ## Performance: matching the hand-tuned kernel took metal-specific intrinsics
//
// A naive port that expressed the hot path in portable Slang ran at ~0.64x the
// handwritten kernel. Profiling (not guessing: three plausible causes were
// measured and rejected) traced almost all of it to one thing. `src0` is bound
// as `StructuredBuffer<uint>` because WGSL has no byte-addressable storage, so a
// portable Q8_0 dequant reconstructs every quant with a word load, a shift, a
// mask and a manual sign-extend. That per-byte arithmetic, not the MMAs, was the
// gap, and it inflated register pressure enough to cut occupancy too.
//
// The fix is to drop to Metal for the memory access only, via `__target_intrinsic`
// (`load_i8x4_direct`, `load_f16_direct`, `stage_input_f2x4` below). Each reads
// or stores exactly as the handwritten kernel does: a `packed_char4` for four
// quants at once, a `half*` deref for the scale, a `float2x4` for the input
// stage. All are called only inside `case metal:`, so the WGSL branch never sees
// them and keeps its portable scalar path (verified: the emitted WGSL contains
// no `packed_char4`/`float2x4`/`device char`). With those plus the unroll
// strategy below, the generated kernel reaches ~0.93x, bit-identical, at higher
// occupancy than the handwritten one.
//
// The residual ~7% is *not* the `CoopMat` abstraction, though an earlier version
// of this comment said so. Disassembling to native AGX shows the kernel body
// makes no `linalg_*` calls at all; the cost is address arithmetic. The native
// compiler folds load displacements into the instruction immediate for the
// handwritten kernel (4 base registers, 21 immediates) and not for this one (11
// bases, 2 immediates, at half the unroll factor, so compare those counts only
// via the rewritten kernel). `build_support/msl_postpass.rs` rewrites the
// emitted MSL to recover the folding, reaching ~0.98x. Read that file before
// changing anything below.
//
// Two levers were measured and *rejected*, recorded so they are not retried.
// The first is the `simdgroup_barrier(mem_flags::mem_none)` operand-load hint,
// expressible via `__target_intrinsic` but neutral on Apple silicon in both the
// dequant-bound and scheduling-bound regimes. The second is unrolling the `ik`
// loop in Slang, recorded at that loop below.
//
// ## Remaining divergence from `shaders/gemm_q8_0.metal`
//
// The ragged-tile epilogue bounces through `sb` in two 16-column rounds instead
// of reinterpreting the whole 8 KB scratch as one 64x32 float tile. Slang has no
// way to type-pun groupshared memory, and declaring a third array would have
// pushed the threadgroup budget from 8 KB to 16 KB and halved occupancy.
// Interior tiles take the direct-store fast path and never reach it.
//
// ## Layout (mirrors the handwritten kernel exactly)
//
// Dispatch: (ceil(n/32), ceil(m/64)) threadgroups x 128 threads (4 simdgroups).
// Threadgroup memory: 8 KB, as 4 KB of half weights + 4 KB of float input.
// Each threadgroup computes a 64x32 output tile; each simdgroup owns 32x16 of
// it as eight 8x8 accumulators.

using namespace linalg;

struct GemmParams {
    uint m;
    uint k;
    uint n;
    uint x_stride;
    uint y_stride;
    uint _pad;
};

[[vk::binding(0)]] StructuredBuffer<uint>       src0   : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>      src1   : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float>    dst    : register(u2);
[[vk::binding(3)]] StructuredBuffer<GemmParams> params : register(t3);

static const uint BLOCK_SIZE_M   = 64u;
static const uint BLOCK_SIZE_N   = 32u;
static const uint BLOCK_SIZE_K   = 32u;
static const uint THREAD_PER_ROW = 2u;
static const uint THREAD_PER_COL = 4u;
static const uint SG_MAT_SIZE    = 64u;
static const uint THREADS        = 128u;

/// Q8_0 block: an f16 scale followed by 32 int8 quants. Never a struct: 34 is
/// not a multiple of 4, so any struct view of it would be padded and the row
/// stride would silently stop matching the file layout.
static const uint Q8_BLOCK_BYTES = 34u;

groupshared half  sa[BLOCK_SIZE_M * BLOCK_SIZE_K]; // 2048 half  = 4 KB
groupshared float sb[BLOCK_SIZE_N * BLOCK_SIZE_K]; // 1024 float = 4 KB

// ---------------------------------------------------------------------------
// Q8_0 byte access
//
// `src0` is bound as `StructuredBuffer<uint>` rather than a byte buffer because
// that is the one view both targets agree on: WGSL has no byte-addressable
// storage type, and Metal's `device uchar*` has no Slang spelling. Blocks start
// at multiples of 34, so a block is 2-byte aligned but not 4-byte aligned, and
// every read has to select its lane out of the containing word.
// ---------------------------------------------------------------------------

uint load_byte(uint byte_off) {
    uint w = src0[byte_off >> 2u];
    return (w >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

/// Sign-extend without a branch: bias into unsigned, subtract the bias back.
int load_i8(uint byte_off) {
    return int(load_byte(byte_off) ^ 0x80u) - 128;
}

/// Metal-only vectorized signed-byte read: four consecutive Q8_0 quants as one
/// `packed_char4` load, sign-extended to `int4`. `src0` is bound as
/// `StructuredBuffer<uint>` for WGSL's sake, but on Metal that lowers to a
/// `device uint*`, and the quants are signed bytes inside it. The portable
/// `load_i8` reconstructs each byte with a word load, a shift, a mask and a
/// manual sign-extend; profiling showed that per-byte arithmetic, not the MMAs,
/// was the whole ~0.64x gap (it also inflated register pressure enough to cut
/// occupancy). `packed_char4` is 1-byte aligned and lets the hardware
/// sign-extend, which removes that arithmetic. It does *not* reduce the load
/// count, though an earlier version of this comment claimed it cut 16 scalar
/// loads per k-tile to four: AIR shows 16 scalar `load i8` per k-tile here and
/// in the handwritten kernel alike. Only ever called inside `case metal:`, so
/// the WGSL branch never references it and keeps the portable path. `$0`/`$1`
/// are the buffer and the byte offset.
__target_intrinsic(metal, "int4(*(const device packed_char4*)((const device char*)$0 + $1))")
int4 load_i8x4_direct(StructuredBuffer<uint> buf, uint byte_off);

/// Metal-only direct f16 scale read, the analog of `load_i8x4_direct` for the
/// block's leading half. `byte_off` is 2-byte aligned (blocks start at multiples
/// of 34), so this matches the handwritten `*(const device half*)blk`. The
/// portable `load_f16` reconstructs it with a word load, a conditional shift and
/// `f16tof32`; on Metal a `half*` deref is a single load the hardware widens.
__target_intrinsic(metal, "(*(const device half*)((const device char*)$0 + $1))")
half load_f16_direct(StructuredBuffer<uint> buf, uint byte_off);

/// Metal-only vectorized input staging: copy 8 consecutive floats from `src1`
/// into `sb` as one `float2x4` (32-byte) load and store, exactly the handwritten
/// kernel's `*(threadgroup float2x4*)(...) = *(device float2x4*)y`. The portable
/// path does eight scalar load/store pairs because Slang cannot cast a
/// groupshared pointer to a vector type; those scalar transactions were part of
/// the residual staging gap. `sb_off` (a multiple of 8) and `src_off` are both
/// 32-byte aligned by construction, the same precondition the handwritten store
/// relies on. `$0..$3` are sb, its offset, src1, its offset; the array parameter
/// size is BLOCK_SIZE_N * BLOCK_SIZE_K.
__target_intrinsic(metal, "(*(threadgroup float2x4*)(&$0[$1]) = *(const device float2x4*)(&$2[$3]))")
void stage_input_f2x4(float sb[1024], uint sb_off, StructuredBuffer<float> src, uint src_off);

/// `byte_off` must be 2-byte aligned, which every Q8_0 scale is.
float load_f16(uint byte_off) {
    uint w = src0[byte_off >> 2u];
    uint h = ((byte_off & 2u) != 0u) ? (w >> 16u) : (w & 0xFFFFu);
    return f16tof32(h);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gemm_q8_0(uint3 gid : SV_GroupID, uint tiitg : SV_GroupIndex) {
    const uint m        = params[0].m;
    const uint k        = params[0].k;
    const uint n        = params[0].n;
    const uint x_stride = params[0].x_stride;
    const uint y_stride = params[0].y_stride;

    const uint nb        = k / 32u;
    const uint row_bytes = nb * Q8_BLOCK_BYTES;

    const uint r0 = gid.y; // output row block
    const uint r1 = gid.x; // output column block

    __target_switch {
    case metal:
    {
        typealias MatA = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixA>;
        typealias MatB = CoopMat<half,  MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixB>;
        typealias MatC = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixAccumulator>;

        // Metal forms simdgroups from consecutive flat thread indices, so this
        // equals `simdgroup_index_in_threadgroup` for a 1D 128-thread group.
        const uint sgitg = tiitg >> 5u;

        const uint n_rows = min(m - r0 * BLOCK_SIZE_M, BLOCK_SIZE_M);
        const uint n_cols = min(n - r1 * BLOCK_SIZE_N, BLOCK_SIZE_N);

        // Clamped so an overhanging tile re-reads a live row rather than running
        // off the buffer. The staging index below deliberately uses the
        // *unclamped* thread id, so those duplicate rows land in the slots the
        // MMA will read and be discarded by the epilogue's bounds check.
        const uint thread_row = min(tiitg / THREAD_PER_ROW, n_rows - 1u);
        const uint thread_col = min(tiitg / THREAD_PER_COL, n_cols - 1u);

        // Which 16-element half of the 32-element Q8_0 block this thread owns.
        // The handwritten kernel carries llama.cpp's general `nl` stepping here;
        // with 32-element blocks and THREAD_PER_ROW == 2 it reduces to a
        // constant `il` and a one-block advance per k tile.
        const uint il = tiitg % THREAD_PER_ROW;

        uint x_byte = row_bytes * (r0 * BLOCK_SIZE_M + thread_row);
        uint y_off  = x_stride * (r1 * BLOCK_SIZE_N + thread_col)
                    + (BLOCK_SIZE_K / THREAD_PER_COL) * (tiitg % THREAD_PER_COL);

        MatC mc[8];
        [ForceUnroll]
        for (uint i = 0u; i < 8u; i++) {
            mc[i] = MatC(0.0f);
        }

        for (uint loop_k = 0u; loop_k < k; loop_k += BLOCK_SIZE_K) {
            // Dequantize this thread's 16 weights before the barrier: the read
            // is from device memory and does not depend on anyone else's stores.
            const float d = float(load_f16_direct(src0, x_byte));
            half temp_a[16];
            const uint q_base = x_byte + 2u + 16u * il;
            [ForceUnroll]
            for (uint g = 0u; g < 4u; g++) {
                const int4 q = load_i8x4_direct(src0, q_base + g * 4u);
                temp_a[g * 4u + 0u] = half(float(q.x) * d);
                temp_a[g * 4u + 1u] = half(float(q.y) * d);
                temp_a[g * 4u + 2u] = half(float(q.z) * d);
                temp_a[g * 4u + 3u] = half(float(q.w) * d);
            }

            GroupMemoryBarrierWithGroupSync();

            // Stage A in simdgroup-matrix-native order: each 8x8 tile occupies
            // SG_MAT_SIZE contiguous halves, so the MMA loads below read with a
            // stride of 8 and no gather.
            [ForceUnroll]
            for (uint i = 0u; i < 16u; i++) {
                const uint slot = SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8u)
                                               + (tiitg % THREAD_PER_ROW) * 16u
                                               + (i / 8u) * 8u)
                                + (tiitg / THREAD_PER_ROW) % 8u + (i & 7u) * 8u;
                sa[slot] = temp_a[i];
            }

            // Eight contiguous floats per thread, staged as one float2x4 to
            // match the handwritten kernel (see `stage_input_f2x4`).
            const uint sb_base = 32u * 8u * (tiitg % THREAD_PER_COL)
                               + 8u * (tiitg / THREAD_PER_COL);
            stage_input_f2x4(sb, sb_base, src1, y_off);

            x_byte += Q8_BLOCK_BYTES;
            y_off  += BLOCK_SIZE_K;

            GroupMemoryBarrierWithGroupSync();

            uint lsma = 4u * SG_MAT_SIZE * (sgitg % 2u);
            uint lsmb = 2u * SG_MAT_SIZE * (sgitg / 2u);

            // `ik` stays rolled *here* on purpose: unrolling it in Slang gives
            // each iteration's `ma`/`mb` distinct registers (all four live at
            // once), which drops occupancy 896 -> 704, and it measures 0.89x.
            // The inner loops below are still unrolled for constant indices and
            // MMA scheduling.
            //
            // This is not the same lever as the `#pragma unroll(4)` that
            // `build_support/msl_postpass.rs` adds to the emitted MSL. That
            // pragma does not unroll in AIR; it attaches loop metadata the
            // native AGX translator honors, which `[ForceUnroll]` here cannot
            // produce. Do not "fix" the inconsistency by unrolling in Slang.
            for (uint ik = 0u; ik < BLOCK_SIZE_K / 8u; ik++) {
                MatB ma[4];
                MatA mb[2];

                [ForceUnroll]
                for (uint i = 0u; i < 4u; i++) {
                    ma[i] = MatB.Load<CoopMatMatrixLayout.RowMajor>(sa, lsma + SG_MAT_SIZE * i, 8);
                }
                [ForceUnroll]
                for (uint i = 0u; i < 2u; i++) {
                    mb[i] = MatA.Load<CoopMatMatrixLayout.RowMajor>(sb, lsmb + SG_MAT_SIZE * i, 8);
                }

                // Operand order matches the handwritten kernel: the *input* tile
                // is the A operand and the *weight* tile is B, which is why the
                // accumulator's row axis is a dst column and its column axis a
                // dst row. The epilogue strides below depend on that.
                [ForceUnroll]
                for (uint i = 0u; i < 8u; i++) {
                    mc[i] = coopMatMulAdd<float, false>(mb[i / 4u], ma[i % 4u], mc[i]);
                }

                lsma += (BLOCK_SIZE_M / 8u) * SG_MAT_SIZE;
                lsmb += (BLOCK_SIZE_N / 8u) * SG_MAT_SIZE;
            }
        }

        if ((r0 + 1u) * BLOCK_SIZE_M <= m && (r1 + 1u) * BLOCK_SIZE_N <= n) {
            // Full tile: no bounds check needed, store straight to dst.
            const uint c_off = (BLOCK_SIZE_M * r0 + 32u * (sgitg & 1u))
                             + (BLOCK_SIZE_N * r1 + 16u * (sgitg >> 1u)) * y_stride;
            [ForceUnroll]
            for (uint i = 0u; i < 8u; i++) {
                mc[i].Store<CoopMatMatrixLayout.RowMajor>(
                    dst, c_off + 8u * (i % 4u) + 8u * y_stride * (i / 4u), y_stride);
            }
        } else {
            // Ragged tile. `sb` is dead once the k loop ends and holds exactly
            // 16 columns of the 64-row tile, so the two column halves go out in
            // two rounds. See the divergence note in the header.
            for (uint round = 0u; round < 2u; round++) {
                GroupMemoryBarrierWithGroupSync();

                if ((sgitg >> 1u) == round) {
                    const uint base = 32u * (sgitg & 1u);
                    [ForceUnroll]
                    for (uint i = 0u; i < 8u; i++) {
                        mc[i].Store<CoopMatMatrixLayout.RowMajor>(
                            sb, base + 8u * (i % 4u) + 8u * BLOCK_SIZE_M * (i / 4u), BLOCK_SIZE_M);
                    }
                }

                GroupMemoryBarrierWithGroupSync();

                // 16 columns x 64 rows = 1024 elements over 128 threads. The
                // row index is the fast axis, so a simdgroup's writes to dst
                // stay contiguous.
                for (uint idx = tiitg; idx < 16u * BLOCK_SIZE_M; idx += THREADS) {
                    const uint j = idx / BLOCK_SIZE_M;
                    const uint r = idx % BLOCK_SIZE_M;
                    const uint gj = round * 16u + j;
                    if (gj < n_cols && r < n_rows) {
                        dst[(r0 * BLOCK_SIZE_M + r)
                            + (r1 * BLOCK_SIZE_N + gj) * y_stride] = sb[j * BLOCK_SIZE_M + r];
                    }
                }
            }
        }
        break;
    }
    default:
    {
        // Correctness reference only. One dot product per output element, no
        // tiling, no staging, no reuse. See the header before drawing any
        // conclusion from its speed.
        for (uint idx = tiitg; idx < BLOCK_SIZE_M * BLOCK_SIZE_N; idx += THREADS) {
            const uint row = r0 * BLOCK_SIZE_M + (idx % BLOCK_SIZE_M);
            const uint col = r1 * BLOCK_SIZE_N + (idx / BLOCK_SIZE_M);
            if (row >= m || col >= n) {
                continue;
            }

            const uint wbase = row * row_bytes;
            const uint ybase = col * x_stride;

            float acc = 0.0f;
            for (uint b = 0u; b < nb; b++) {
                const uint blk = wbase + b * Q8_BLOCK_BYTES;
                const float d = load_f16(blk);
                for (uint e = 0u; e < 32u; e++) {
                    acc += d * float(load_i8(blk + 2u + e)) * src1[ybase + b * 32u + e];
                }
            }
            dst[row + col * y_stride] = acc;
        }
        break;
    }
    }
}

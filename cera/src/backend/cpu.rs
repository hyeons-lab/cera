// CPU compute backend — naive scalar implementations.
//
// All functions operate on raw f32 slices. No Tensor abstraction in the hot path.

// ── Thread pool configuration ──────────────────────────────────────────────

/// Warm the compute pools and size rayon's global pool to performance cores.
///
/// The GEMV/GEMM row hot path runs on the persistent
/// [`super::threadpool::RowPool`]s, sized from the detected topology —
/// `CERA_THREADS` overrides the prefill width, `CERA_DECODE_THREADS` the
/// decode width (see `super::calibrate`). `RAYON_NUM_THREADS` governs only
/// the residual rayon sites (dequantization, VL preprocessing); it does
/// **not** constrain the RowPools.
///
/// P-cores only, because efficiency cores have lower clock speed and share
/// memory bandwidth: including them creates straggler threads on synchronized
/// dispatches — measured as a 12% decode regression on M1 Max (58.6 vs 66.4
/// tok/s) back on the rayon path.
///
/// Must be called once before any rayon work (e.g., early in `main()`).
/// Returns the number of threads configured.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn configure_thread_pool() -> usize {
    // Warm both row-parallel pools — spawns and pins their workers now rather
    // than lazily on the first GEMV. The prefill pool's width is the headline
    // thread count; the decode pool is intentionally narrower (memory-bound).
    let _ = super::threadpool::RowPool::decode().num_threads();
    let pool_threads = super::threadpool::RowPool::prefill().num_threads();

    // Also size rayon's global pool for the remaining `par_chunks_mut` sites
    // (prefill GEMM, VL preprocessing), unless the user pinned
    // RAYON_NUM_THREADS.
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        let n = super::cpu_features::performance_core_count();
        // build_global() succeeds at most once per process; if rayon was
        // already initialized (test harness, dependency), the P-core cap
        // doesn't apply to those residual sites — surface that.
        if let Err(err) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            tracing::warn!(
                "cera: rayon global pool already initialized; P-core cap ({n}) not applied to residual rayon sites: {err}"
            );
        }
    }

    pool_threads
}

/// On `wasm32` the row hot path runs on rayon (wasm-bindgen-rayon web
/// workers — see [`par_rows`]), and the pool is built by JS calling
/// `initThreadPool`, never here. Deliberately does NOT touch rayon: querying
/// `current_num_threads()` before `initThreadPool` would force-instantiate
/// the default global registry (1 thread on wasm, since std spawn fails) and
/// make `initThreadPool`'s own `build_global` fail — permanently locking a
/// threaded build to a single thread.
#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
pub fn configure_thread_pool() -> usize {
    1
}

/// No-op stub for builds without the `parallel` feature. Returns `1`
/// because single-threaded is the only choice.
#[cfg(not(feature = "parallel"))]
pub fn configure_thread_pool() -> usize {
    1
}

use crate::quant::{
    BlockQ4_0, BlockQ4_1, BlockQ4KM, BlockQ5K, BlockQ8_0, vec_dot_q4_0_f32, vec_dot_q4_1_f32,
    vec_dot_q4_k_m_f32, vec_dot_q5_k_f32, vec_dot_q8_0_f32,
};
#[cfg(not(target_arch = "aarch64"))]
use crate::quant::{BlockQ6K, vec_dot_q6_k_f32};
use crate::tensor::DType;
use std::mem::size_of;

// ── Matrix multiplication ───────────────────────────────────────────────────

/// Dense f32 matrix multiply: standard `C = A · B`, row-major.
///
/// Concretely: `c[i*n + j] = Σ_p a[i*k + p] · b[p*n + j]` for
/// `i ∈ [0, m), j ∈ [0, n), p ∈ [0, k)`. So `b` must be in
/// `[k × n]` row-major layout (rows are inputs, cols are outputs)
/// — **not** `[n × k]`. This is *not* `A · Bᵀ`; for that
/// orientation, transpose `b` at load time or use one of the
/// `gemv_*` helpers that consumes `[rows × cols]` weight tensors
/// directly.
///
/// `c` accumulates (the loop is `c += a · b`), so `c` must be
/// pre-zeroed (or pre-filled with a broadcast bias if you want to
/// fold the bias-add into the gemm).
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Quantized Q4_0 × f32 matmul: `C[m,n] = dequant(A_q4_0)[m,k] * B[k,n]`.
///
/// `a_quant` is raw Q4_0 bytes, row-major with `m` rows of `k` elements each.
/// Each row is k/32 blocks of 18 bytes.
pub fn matmul_q4_0_f32(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * size_of::<BlockQ4_0>();
    debug_assert_eq!(a_quant.len(), m * bytes_per_row);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let row_start = i * bytes_per_row;
        for j in 0..n {
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let block_offset = row_start + bi * size_of::<BlockQ4_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(block_offset) as *const BlockQ4_0) };
                let col_start = bi * 32;
                let b_slice: Vec<f32> = (0..32).map(|l| b[(col_start + l) * n + j]).collect();
                sum += vec_dot_q4_0_f32(block, &b_slice);
            }
            c[i * n + j] = sum;
        }
    }
}

/// Quantized Q8_0 × f32 matmul: `C[m,n] = dequant(A_q8)[m,k] * B[k,n]`.
///
/// `a_quant` is raw Q8_0 bytes, row-major with `m` rows of `k` elements each.
/// Each row is k/32 blocks of 34 bytes.
pub fn matmul_q8_0_f32(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * size_of::<BlockQ8_0>();
    debug_assert_eq!(a_quant.len(), m * bytes_per_row);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let row_start = i * bytes_per_row;
        for j in 0..n {
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let block_offset = row_start + bi * size_of::<BlockQ8_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(block_offset) as *const BlockQ8_0) };
                // Extract the 32-element column slice from B
                let col_start = bi * 32;
                let b_slice: Vec<f32> = (0..32).map(|l| b[(col_start + l) * n + j]).collect();
                sum += vec_dot_q8_0_f32(block, &b_slice);
            }
            c[i * n + j] = sum;
        }
    }
}

/// Quantized Q4_K_M × f32 matmul: `C[m,n] = dequant(A_q4km)[m,k] * B[k,n]`.
///
/// `a_quant` is raw Q4_K_M bytes, row-major with `m` rows of `k` elements each.
/// Each row is k/256 blocks of 144 bytes.
pub fn matmul_q4km_f32(a_quant: &[u8], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let bytes_per_row = blocks_per_row * size_of::<BlockQ4KM>();
    debug_assert_eq!(a_quant.len(), m * bytes_per_row);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for i in 0..m {
        let row_start = i * bytes_per_row;
        for j in 0..n {
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let block_offset = row_start + bi * size_of::<BlockQ4KM>();
                let block = unsafe { &*(a_quant.as_ptr().add(block_offset) as *const BlockQ4KM) };
                let col_start = bi * 256;
                let b_slice: Vec<f32> = (0..256).map(|l| b[(col_start + l) * n + j]).collect();
                sum += vec_dot_q4_k_m_f32(block, &b_slice);
            }
            c[i * n + j] = sum;
        }
    }
}

// ── GEMV (matrix-vector multiply) ──────────────────────────────────────────

/// Row-parallel `for_each`: applies `f` to each element of `y`. Under
/// `parallel` this dispatches through the persistent
/// [`super::threadpool::RowPool`] (see that module for why not rayon), where
/// `min_rows` gates how many workers participate — each participating worker
/// gets at least that many rows, but the dynamic steal units within a worker
/// are smaller. Otherwise it runs serially.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn par_rows(y: &mut [f32], min_rows: usize, f: impl Fn((usize, &mut f32)) + Sync + Send) {
    super::threadpool::RowPool::decode().dispatch_rows(y, 1, min_rows, |row, slice| {
        f((row, &mut slice[0]));
    });
}

/// On `wasm32` std threads can't spawn, so a `RowPool` would silently degrade
/// to a single worker. Route through rayon instead — the threaded wasm builds
/// back it with web workers via `wasm-bindgen-rayon`'s `initThreadPool`.
#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
pub fn par_rows(y: &mut [f32], min_rows: usize, f: impl Fn((usize, &mut f32)) + Sync + Send) {
    use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
    // Rayon's fork-join barrier rides web workers here — far pricier per task
    // than the native pool's steal units — so floor the chunk size at the
    // rayon-era 512 rather than the RowPool-tuned `min_rows` (default 128),
    // preserving the pre-RowPool wasm task granularity.
    const WASM_MIN_CHUNK_ROWS: usize = 512;
    let chunk_size = (y.len() / crate::par::current_num_threads())
        .max(min_rows)
        .max(WASM_MIN_CHUNK_ROWS);
    y.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * chunk_size;
            for (j, yi) in chunk.iter_mut().enumerate() {
                f((base + j, yi));
            }
        });
}

#[cfg(not(feature = "parallel"))]
pub fn par_rows(y: &mut [f32], _min_rows: usize, f: impl Fn((usize, &mut f32))) {
    for (i, yi) in y.iter_mut().enumerate() {
        f((i, yi));
    }
}

/// Like [`par_rows`] but each "row" is `n` contiguous f32 elements (GEMM
/// output). `f` receives `(row_index, &mut [f32; n])`.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub fn par_rows_n(
    y: &mut [f32],
    n: usize,
    min_rows: usize,
    f: impl Fn((usize, &mut [f32])) + Sync + Send,
) {
    debug_assert_ne!(n, 0, "par_rows_n: n must be > 0");
    if n == 0 || y.is_empty() {
        return;
    }
    super::threadpool::RowPool::prefill().dispatch_rows(y, n, min_rows, |row, row_slice| {
        f((row, row_slice));
    });
}

/// See the `wasm32` note on [`par_rows`].
#[cfg(all(feature = "parallel", target_arch = "wasm32"))]
pub fn par_rows_n(
    y: &mut [f32],
    n: usize,
    min_rows: usize,
    f: impl Fn((usize, &mut [f32])) + Sync + Send,
) {
    debug_assert_ne!(n, 0, "par_rows_n: n must be > 0");
    if n == 0 || y.is_empty() {
        return;
    }
    use crate::par::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
    let m = y.len() / n;
    let rows_per_chunk = (m / crate::par::current_num_threads()).max(min_rows.max(1));
    let elems_per_chunk = rows_per_chunk * n;
    y.par_chunks_mut(elems_per_chunk)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_row = ci * rows_per_chunk;
            for (j, row) in chunk.chunks_mut(n).enumerate() {
                f((base_row + j, row));
            }
        });
}

#[cfg(not(feature = "parallel"))]
pub fn par_rows_n(y: &mut [f32], n: usize, _min_rows: usize, f: impl Fn((usize, &mut [f32]))) {
    debug_assert_ne!(n, 0, "par_rows_n: n must be > 0");
    if n == 0 || y.is_empty() {
        return;
    }
    for (j, row) in y.chunks_mut(n).enumerate() {
        f((j, row));
    }
}

#[allow(clippy::ptr_arg)]
/// Q4_0 GEMV: `y[m] = A_q4_0[m,k] @ x[k]`.
///
/// On aarch64, uses integer dot product with caller-provided Q8_0 scratch buffers
/// to avoid per-call heap allocation. The scratch buffers are resized as needed.
pub fn gemv_q4_0_f32(
    a_quant: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scales: &mut Vec<f32>,
    q8_quants: &mut Vec<i8>,
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 32, 0, "Q4_0 GEMV: k must be divisible by 32");

    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::backend::simd::neon::gemv_q4_0_f32_neon(
                a_quant, x, y, m, k, q8_scales, q8_quants,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // x86 int8: quantize the activation to Q8_0 once, then keep the whole
        // dot product in int8 instead of widening every weight to f32 —
        // `dpbusd` on the VNNI arm, the `maddubs` emulation on the AVX2 one.
        // Same shape as the aarch64 branch above; `q8_scales`/`q8_quants`
        // are the caller's reusable scratch, which is why they are threaded
        // through this signature at all.
        // `vnni_int8_available()`, not a hand-rolled tier compare: this predicate
        // and the one selecting the quantizer must not drift, and since the AVX2
        // int8 kernels landed they are two different predicates.
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if vnni_int8_available() {
            q8_scales.resize(blocks_per_row, 0.0);
            q8_quants.resize(k, 0);
            // SAFETY: the tier predicate above proved the kernel's feature set,
            // and the scratch was just sized to the lengths it asserts.
            // `quantize_f32_to_q8_0_into`, not a per-tier quantizer: it already
            // dispatches, and naming it here is what makes decode and prefill
            // provably quantize through the same function.
            unsafe {
                quantize_f32_to_q8_0_into(x, q8_scales, q8_quants);
                crate::backend::simd::avx512_vnni::gemv_q4_0_q8_0(
                    a_quant, q8_scales, q8_quants, y, m, k,
                );
            }
            return;
        }

        // No VNNI but AVX2: the emulated int8 GEMV. Decode has to take the same
        // arithmetic as batched prefill or the parity bar breaks — see
        // `avx2_int8::gemv_q4k_f32` for the measurement behind that, and
        // `tests/avx2_decode_prefill_identity.rs` for the guard.
        #[cfg(target_arch = "x86_64")]
        if avx2_int8_available() {
            q8_scales.resize(blocks_per_row, 0.0);
            q8_quants.resize(k, 0);
            // SAFETY: the tier predicate above proved the kernel's feature set,
            // and the scratch was just sized to the lengths it asserts.
            // `quantize_f32_to_q8_0_into`, not a per-tier quantizer: it already
            // dispatches, and naming it here is what makes decode and prefill
            // provably quantize through the same function.
            unsafe {
                quantize_f32_to_q8_0_into(x, q8_scales, q8_quants);
                crate::backend::simd::avx2_int8::gemv_q4_0_q8_0(
                    a_quant, q8_scales, q8_quants, y, m, k,
                );
            }
            return;
        }

        let _ = (q8_scales, q8_quants);
        // The AVX-512 f32 row-dot dispatch that used to sit here is gone. The
        // int8 arms above return for every tier from `Avx2` up, so it was
        // unreachable on every shipping x86 CPU — including at `Scalar`, where
        // its own tier guard is false. Two reasons it had to go rather than
        // merely being shadowed: the int8 GEMV measured faster even on an
        // AVX-512 host (31.5 -> 41.6 tok/s decode at `CERA_CPU_TIER=avx512`),
        // and decode must run the same arithmetic as batched prefill or the
        // parity bar breaks. The kernels themselves
        // (`simd::avx512::row_dot_{q4_0,q8_0}_f32_avx512`) are kept and still
        // unit-tested, so restoring the dispatch is a small change if the int8
        // path ever needs narrowing.
        let compute_row = |(i, yi): (usize, &mut f32)| {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let offset = row_start + bi * size_of::<BlockQ4_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ4_0) };
                sum += vec_dot_q4_0_f32(block, &x[bi * 32..(bi + 1) * 32]);
            }
            *yi = sum;
        };

        if m >= gemv_par_threshold() {
            par_rows(y, gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }
}

/// Default minimum output dimension to parallelize a GEMV — below this the
/// per-dispatch barrier costs more than the split saves.
pub const GEMV_PAR_THRESHOLD_DEFAULT: usize = 256;

/// Minimum output dimension to use parallel GEMV, resolved once. `CERA_PAR_THRESHOLD`
/// overrides [`GEMV_PAR_THRESHOLD_DEFAULT`] for tuning the parallel/serial cutoff
/// per device (small decode GEMVs may not pay for the threadpool barrier).
#[cfg(feature = "parallel")]
pub fn gemv_par_threshold() -> usize {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        super::cpu_features::env_usize("CERA_PAR_THRESHOLD").unwrap_or(GEMV_PAR_THRESHOLD_DEFAULT)
    })
}

/// Without `parallel`, GEMVs never parallelize — the threshold is effectively
/// infinite, so callers always take the serial path.
#[cfg(not(feature = "parallel"))]
pub fn gemv_par_threshold() -> usize {
    usize::MAX
}

/// Default minimum columns per worker for the batched-prefill activation
/// pre-quantization (`quantize_columns`) RowPool fan-out.
///
/// One value with one meaning: a worker is never handed fewer than this many
/// columns, and — since a single worker's minimum is the smallest split worth
/// making — fewer than this many columns total run serially on the caller. The
/// split matters because the batched GEMM already parallelizes over output rows,
/// so a *serial* pre-quant is the Amdahl term that caps multi-core prefill
/// (measured as a regression on an 8-core Android big.LITTLE once the parallel
/// GEMM shrank per core). A column's work (a strided gather + `dim/32` Q8_0
/// block-quantizes) is lighter than a GEMV output row, so this sits well below
/// [`GEMV_PAR_THRESHOLD_DEFAULT`].
///
/// Consumed by the no-`blas` prefill `quantize_columns` on both int8-GEMM
/// targets: aarch64 NEON and x86_64 int8 (VNNI or AVX2).
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(feature = "blas")
))]
pub const PREQUANT_PAR_MIN_COLS_DEFAULT: usize = 32;

/// Minimum columns per worker for the prefill pre-quant fan-out, resolved once.
/// `CERA_PREQUANT_MIN_COLS` overrides [`PREQUANT_PAR_MIN_COLS_DEFAULT`] for
/// per-device tuning — the fan-out is a measured win/loss knob on big.LITTLE, so
/// it gets a runtime override like its siblings `gemv_par_threshold` /
/// `gemv_min_rows`, rather than needing a recompile to sweep.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(feature = "blas")
))]
pub fn prequant_par_min_cols() -> usize {
    use std::sync::OnceLock;
    static MIN_COLS: OnceLock<usize> = OnceLock::new();
    *MIN_COLS.get_or_init(|| {
        super::cpu_features::env_usize("CERA_PREQUANT_MIN_COLS")
            .unwrap_or(PREQUANT_PAR_MIN_COLS_DEFAULT)
    })
}

/// Default minimum rows a decode-GEMV worker is given before another worker is
/// added. With the persistent chunk-stealing pool the per-chunk cost is low, so
/// this can be small — smaller lets narrow projections (e.g. GQA K/V, ≤ kv_dim
/// rows) parallelize instead of falling to the serial `active == 1` path.
pub const GEMV_MIN_ROWS_DEFAULT: usize = 128;

/// Minimum rows per worker for decode GEMVs, resolved once. `CERA_MIN_ROWS`
/// overrides [`GEMV_MIN_ROWS_DEFAULT`] for per-device tuning.
#[cfg(feature = "parallel")]
pub fn gemv_min_rows() -> usize {
    use std::sync::OnceLock;
    static MIN_ROWS: OnceLock<usize> = OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        super::cpu_features::env_usize("CERA_MIN_ROWS").unwrap_or(GEMV_MIN_ROWS_DEFAULT)
    })
}

#[cfg(not(feature = "parallel"))]
pub fn gemv_min_rows() -> usize {
    GEMV_MIN_ROWS_DEFAULT
}

/// Portable Q8_0 quantizer. Mirrors the NEON/AVX-512 kernels exactly — same
/// `amax / 127` scale, same f16 round-trip of `d`, same round-to-nearest-even —
/// so a host that falls back here produces bit-identical blocks to one that
/// doesn't.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn quantize_f32_to_q8_0_scalar(x: &[f32], scales: &mut [f32], quants: &mut [i8]) {
    for (bi, blk) in x.chunks(32).enumerate() {
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        // Non-finite guard, matching `quantize_f32_to_q8_0_avx512` — see the
        // comment there. Keeps this reference byte-identical to the SIMD kernel
        // for denormal / NaN blocks instead of saturating the opposite way.
        let id = match 1.0 / d {
            r if d != 0.0 && r.is_finite() => r,
            _ => 0.0,
        };
        scales[bi] = half::f16::from_f32(d).to_f32();
        for (t, &v) in blk.iter().enumerate() {
            quants[bi * 32 + t] = (v * id).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }
}

/// Quantize `x` to Q8_0 blocks into caller-owned scratch, dispatched to the best
/// kernel for the host (NEON, AVX-512+VNNI, else scalar).
///
/// The arch-neutral entry point for the batched-prefill helpers: the per-arch
/// kernels each sit behind their own `target_feature`, so they cannot be named
/// interchangeably at a call site.
/// `#[doc(hidden)] pub` for the same reason as `gemm_preq_dispatch` — the
/// decode/prefill identity test has to quantize its own activation column.
#[doc(hidden)]
pub fn quantize_f32_to_q8_0_into(x: &[f32], scales: &mut [f32], quants: &mut [i8]) {
    // `assert!`, not `debug_assert!`: this is a safe function that hands its
    // arguments to `unsafe` SIMD kernels which write `x.len()/32` scales and
    // `x.len()` quants with no bounds checking of their own. A short buffer is
    // an out-of-bounds write, and a `debug_assert` would let exactly that
    // through in the release builds that matter. The hot caller
    // (`quantize_columns`) reaches here once per 32-element block, so this is
    // three integer compares against a 32-float gather and quantize.
    assert_eq!(
        x.len() % 32,
        0,
        "quantize_f32_to_q8_0_into: x.len() must be divisible by 32"
    );
    assert!(
        scales.len() >= x.len() / 32 && quants.len() >= x.len(),
        "quantize_f32_to_q8_0_into: scales/quants too small for x.len()={} \
         (need {} scales, {} quants; got {} and {})",
        x.len(),
        x.len() / 32,
        x.len(),
        scales.len(),
        quants.len()
    );

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::backend::simd::neon::quantize_f32_to_q8_0_neon(x, scales, quants);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // NOT `int8_gemm_available`: this quantizer is built from `_mm512_*`
        // intrinsics, so the predicate has to answer "can this host execute
        // AVX-512", not "does this host have some int8 GEMM". Since the AVX2
        // int8 kernels landed those differ, and the wider predicate would hand
        // AVX-512 code to an AVX2-only host — a SIGILL, not a wrong answer.
        //
        // Not `vnni_int8_available()` either. `quantize_f32_to_q8_0_avx512`
        // declares `avx512f,avx512vl,avx2` and uses no `dpbusd`, so requiring
        // VNNI would deny it to every Skylake-X-class host — which, since the
        // int8 arms now shadow the f32 row-dot, is a host that quantizes on
        // every projection. Hence the dedicated `avx512_quantizer_available()`.
        //
        // Do not read that as a measured win: A/B'd on LFM2.5-230M-Q4_K_M at
        // `CERA_CPU_TIER=avx512`, pool pinned at 16, n=20, the two arms are
        // indistinguishable (decode p50 75.7 vs 76.1, stddev ~7; prefill 212 vs
        // 203, stddev ~58). The quantize is O(k) against a GEMV's O(m*k), so
        // that is the expected result. This is a correctness-of-predicate fix —
        // the old gate demanded an instruction set the kernel never uses — and
        // it may matter more on a real Skylake-X, where AVX2 is not being
        // executed by a Zen 5.
        //
        // Still deliberately not done: an AVX2 quantizer. Below the `Avx512`
        // tier every int8 prefill AND decode pays *scalar* quantization, since
        // the only vectorized quantizer here is the `_mm512_*` one. That is a
        // known, accepted cost — deferred, not overlooked.
        //
        // Which arm runs does not change the bytes:
        // `quantize_q8_0_scalar_matches_avx512` pins the scalar fallback to the
        // AVX-512 kernel
        // bit-for-bit. That is what lets decode and prefill share this one
        // dispatcher without the parity bar noticing which arm ran, and what
        // lets the AVX2 GEMM consume scalar-quantized activations.
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512_quantizer_available() {
            unsafe {
                crate::backend::simd::avx512_vnni::quantize_f32_to_q8_0_avx512(x, scales, quants);
            }
            return;
        }
        quantize_f32_to_q8_0_scalar(x, scales, quants);
    }
}

/// Whether `quantize_f32_to_q8_0_avx512` is callable on this host.
///
/// Deliberately not `vnni_int8_available()`: the quantizer needs
/// `avx512f,avx512vl,avx2` and no VNNI, and it is reached from the int8 GEMV and
/// from prefill's `quantize_columns` alike. `avx512vl` is not implied by the
/// `Avx512` tier, so it is checked as a raw feature; the tier compare is what
/// keeps `CERA_CPU_TIER` a working downgrade lever.
///
/// `#[doc(hidden)] pub` so `avx512_quantizer_gate.rs` can pin it at a *forced*
/// tier. That binary exists because an in-process test cannot do the job twice
/// over: on a VNNI host `>= Avx512` and `>= Avx512Vnni` are both true, so a
/// re-narrowed predicate is invisible; and gating such a test on raw CPUID
/// instead would fire it under a deliberate `CERA_CPU_TIER` downgrade. Forcing
/// the tier in a dedicated process resolves both. Not part of the supported
/// API.
#[doc(hidden)]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn avx512_quantizer_available() -> bool {
    let f = crate::backend::cpu_features::cpu_features();
    f.tier >= crate::backend::cpu_features::CpuTier::Avx512 && f.avx512vl && f.avx2
}

/// Whether the AVX-512 VNNI int8 kernels are callable on this host.
///
/// Distinct from [`int8_gemm_available`] because the two answer different
/// questions: this one selects a *kernel*, while `int8_gemm_available` asks only
/// whether *some* int8 GEMM exists. The AVX-512 activation quantizer is gated by
/// neither — it has its own [`avx512_quantizer_available`], because it needs no
/// VNNI.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn vnni_int8_available() -> bool {
    crate::backend::cpu_features::cpu_features().tier
        >= crate::backend::cpu_features::CpuTier::Avx512Vnni
}

/// Whether the VNNI-free AVX2 int8 kernels are callable on this host.
///
/// The `Avx2` tier already implies `avx2` + `fma`, which is the whole
/// requirement: the emulation is built from `_mm256_maddubs_epi16` and
/// `_mm256_madd_epi16`, both AVX2, and needs nothing above it. No `avx512` crate
/// feature either, so this holds on an `--no-default-features` build too.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_int8_available() -> bool {
    crate::backend::cpu_features::cpu_features().tier >= crate::backend::cpu_features::CpuTier::Avx2
}

/// Whether this host has an int8 batched GEMM for the pre-quantized prefill path.
///
/// **Runtime**, not just compile-time. On x86 every tier from `Avx2` up now
/// satisfies it, backed by two implementations of one kernel body: VNNI runs
/// `vpdpbusd` directly, `Avx2` and `Avx512` emulate it (see the `avx2_int8`
/// module). `batched_gemm_supports` consults this, so a
/// host that answers `false` never reaches `gemm_preq` — which matters because
/// the prefill callers ignore that function's return value and reuse one output
/// buffer across layers.
pub fn int8_gemm_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Not `|| vnni_...`: the VNNI tier is strictly above `Avx2` in the
        // ordering, so the AVX2 predicate already covers it.
        avx2_int8_available()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

// ── Q4_0 weight repacking (8-row interleave, prefill only) ───────────────────
//
// The repacked prefill GEMM (`simd::*::gemm_q4_0_8x8_q8_0`) interleaves 8 weight
// rows so the 8 int32 lanes of one `dpbusd` are 8 output rows, which removes the
// per-column hsum the standard-layout kernel pays. `repack_q4_0_8x8` builds that
// layout once at load; `q4_0_repack_supported` decides whether a given weight
// qualifies; `gemm_preq_repacked_q4_0_dispatch` runs the kernel. Prefill only — the
// win is amortizing the removed reduction across prefill columns; decode (n=1)
// is dispatch-bound and keeps the standard mmap layout.

/// Repack `m x k` standard Q4_0 weight rows into the 8-row-interleaved layout
/// consumed by `gemm_q4_0_8x8_q8_0`, plus the per-(super-row, block) f32 row
/// scales. Requires `m % 8 == 0` and `k % 32 == 0` (see [`q4_0_repack_supported`]).
///
/// Returns `(packed, scales)`. In `packed`, byte `i` of k-group `g`'s 16 bytes
/// (at `(sr*nb + b)*128 + g*16 + i`) holds row `8*sr + i/4` in its low nibble
/// and row `8*sr + 4 + i/4` in its high nibble, both at k-element `4*g + i%4` —
/// exactly what the kernel's low/high nibble unpack reassembles into a
/// lane-per-row weight vector. `scales[(sr*nb + b)*8 + r]` is row `8*sr + r`'s
/// f32 scale for block `b`. The *nibble* footprint is unchanged — 128 bytes per
/// super-row block (8× a `BlockQ4_0`'s 16 `qs` bytes) — but the scales are
/// re-stored as f32 (32 bytes) where the source held f16 (16 bytes), so the
/// repacked copy is a little larger than the source, and it is kept alongside
/// it (decode reads the mmap).
///
/// x86-only: the only consumer, `gemm_q4_0_8x8_q8_0`, is an x86 int8 kernel.
/// `allow(dead_code)` under `blas`: `with_repack` (its non-test caller) is
/// `cfg(not(blas))`, so a non-test `--features blas` build has no caller.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(feature = "blas", allow(dead_code))]
pub(crate) fn repack_q4_0_8x8(src: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(
        m.is_multiple_of(8),
        "repack_q4_0_8x8: m must be a multiple of 8"
    );
    assert!(
        k.is_multiple_of(32),
        "repack_q4_0_8x8: k must be a multiple of 32"
    );
    let nb = k / 32;
    let bsz = size_of::<crate::quant::BlockQ4_0>();
    assert_eq!(
        src.len(),
        m * nb * bsz,
        "repack_q4_0_8x8: src is {} bytes, need {} for {m}x{k}",
        src.len(),
        m * nb * bsz,
    );
    let sr_count = m / 8;
    let mut packed = vec![0u8; sr_count * nb * 128];
    let mut scales = vec![0.0f32; sr_count * nb * 8];
    // Nibble of `row`, block `b`, element `e` (0..32) in the standard layout:
    // e<16 is the low nibble of qs[e], e>=16 the high nibble of qs[e-16].
    let nibble = |row: usize, b: usize, e: usize| -> u8 {
        let qs = (row * nb + b) * bsz + 2; // skip the 2-byte f16 scale
        if e < 16 {
            src[qs + e] & 0x0F
        } else {
            src[qs + e - 16] >> 4
        }
    };
    for sr in 0..sr_count {
        for b in 0..nb {
            for r in 0..8 {
                let off = ((8 * sr + r) * nb + b) * bsz;
                scales[(sr * nb + b) * 8 + r] =
                    half::f16::from_le_bytes([src[off], src[off + 1]]).to_f32();
            }
            for g in 0..8usize {
                for i in 0..16usize {
                    let e = 4 * g + (i % 4);
                    let lo = nibble(8 * sr + i / 4, b, e);
                    let hi = nibble(8 * sr + 4 + i / 4, b, e);
                    packed[(sr * nb + b) * 128 + g * 16 + i] = lo | (hi << 4);
                }
            }
        }
    }
    (packed, scales)
}

/// Whether a Q4_0 weight of shape `m x k` should be repacked for prefill on this
/// host: needs the x86 int8 kernels, a whole number of 8-row super-rows, and
/// Q8_0-block-aligned `k`. A weight that fails this keeps the standard layout and
/// the standard kernel — correctness is identical, only prefill is slower.
#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
pub(crate) fn q4_0_repack_supported(m: usize, k: usize) -> bool {
    m.is_multiple_of(8) && k.is_multiple_of(32) && int8_gemm_available()
}

/// Run the repacked-Q4_0 prefill GEMM on whichever x86 int8 tier this host has
/// (VNNI, or the AVX2 emulation). Returns `true` when a kernel ran. A `false`
/// return means nothing was computed — `q4_0_repack_supported` gates the repack
/// at load, so it cannot happen for a weight that was actually repacked, but the
/// caller must still treat it as the invariant break it is (the output buffer is
/// reused across layers).
#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_preq_repacked_q4_0_dispatch(
    packed: &[u8],
    scales: &[f32],
    b_scales: &[f32],
    b_quants: &[i8],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let nb = k / 32;
    assert!(
        k.is_multiple_of(32) && m.is_multiple_of(8),
        "gemm_preq_repacked_q4_0_dispatch: need k%32==0 and m%8==0, got m={m} k={k}"
    );
    assert!(
        packed.len() >= (m / 8) * nb * 128 && scales.len() >= (m / 8) * nb * 8,
        "gemm_preq_repacked_q4_0_dispatch: repacked weights too small for {m}x{k}"
    );
    assert!(
        b_quants.len() >= n * k && b_scales.len() >= n * nb && out.len() == m * n,
        "gemm_preq_repacked_q4_0_dispatch: activation/output buffers wrong for {m}x{n}x{k}"
    );

    #[cfg(feature = "avx512")]
    if vnni_int8_available() {
        // SAFETY: `vnni_int8_available()` proved the VNNI tier; the kernel
        // re-asserts its own length invariants in debug.
        unsafe {
            crate::backend::simd::avx512_vnni::gemm_q4_0_8x8_q8_0(
                packed, scales, b_scales, b_quants, out, m, n, k,
            );
        }
        return true;
    }
    if avx2_int8_available() {
        // SAFETY: `avx2_int8_available()` proved avx2+fma; same as above.
        unsafe {
            crate::backend::simd::avx2_int8::gemm_q4_0_8x8_q8_0(
                packed, scales, b_scales, b_quants, out, m, n, k,
            );
        }
        return true;
    }
    false
}

/// Repack `m x k` standard Q4_K_M weight rows into the 8-row-interleaved layout
/// consumed by `gemm_q4_k_8x8_q8_0`, plus the per-(super-row, 32-block, row)
/// scale/min products. Requires `m % 8 == 0` and `k % 256 == 0`.
///
/// Returns `(packed, dsc, dmn)`. A Q4_K super-block is 256 values in 8 sub-blocks
/// of 32, so its `k/32` 32-element blocks index the packed nibble buffer exactly
/// like Q4_0's do: `packed[(sr*nb32 + block)*128 + g*16 + i]` holds row `8*sr +
/// i/4` (low nibble) and row `8*sr + 4 + i/4` (high nibble) at that 32-block's
/// element `4*g + i%4`. `dsc`/`dmn` bake the per-row products the kernel would
/// otherwise recompute: `dsc[(sr*nb32 + block)*8 + r] = d_row * sc_row[s]` and
/// `dmn[...] = dmin_row * mn_row[s]`, where `block = bi*8 + s` selects super-block
/// `bi` and its sub-block `s`. The *nibble* footprint is unchanged (16 bytes per
/// 32-block per row, either layout); the scales grow to two baked f32 products
/// per (row, 32-block) — 8 bytes — vs the ~2 bytes the source spends there (its
/// 16-byte `d`+`dmin`+packed-scales header amortized over the super-block's 8
/// sub-blocks), so the repacked copy is ~1.3× the source and is kept alongside it
/// (decode reads the mmap).
///
/// x86-only / `allow(dead_code)` under `blas` for the same reasons as
/// [`repack_q4_0_8x8`].
#[cfg(target_arch = "x86_64")]
#[cfg_attr(feature = "blas", allow(dead_code))]
pub(crate) fn repack_q4_k_8x8(src: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert!(
        m.is_multiple_of(8),
        "repack_q4_k_8x8: m must be a multiple of 8"
    );
    assert!(
        k.is_multiple_of(256),
        "repack_q4_k_8x8: k must be a multiple of 256"
    );
    let sb = k / 256; // super-blocks per row
    let nb32 = k / 32; // 32-element blocks per row
    let bsz = size_of::<crate::quant::BlockQ4KM>();
    assert_eq!(
        src.len(),
        m * sb * bsz,
        "repack_q4_k_8x8: src is {} bytes, need {} for {m}x{k}",
        src.len(),
        m * sb * bsz,
    );
    let sr_count = m / 8;
    let mut packed = vec![0u8; sr_count * nb32 * 128];
    let mut dsc = vec![0.0f32; sr_count * nb32 * 8];
    let mut dmn = vec![0.0f32; sr_count * nb32 * 8];
    // Field offsets within `BlockQ4KM`, derived from the struct rather than
    // hard-coded, so this repack stays correct if the block layout ever moves.
    const D_OFF: usize = std::mem::offset_of!(crate::quant::BlockQ4KM, d);
    const DMIN_OFF: usize = std::mem::offset_of!(crate::quant::BlockQ4KM, dmin);
    const SC_OFF: usize = std::mem::offset_of!(crate::quant::BlockQ4KM, scales);
    const QS_OFF: usize = std::mem::offset_of!(crate::quant::BlockQ4KM, qs);
    // Nibble of `row`, 32-block `block` (0..nb32), element `e` (0..32). `block`
    // selects super-block `bi = block/8` and sub-block `s = block%8`; sub-block
    // `s` lives in the low (s even) or high (s odd) nibbles of the super-block's
    // `qs[(s/2)*32 + e]`. Mirrors the standard kernel's chunk unpack.
    let nibble = |row: usize, block: usize, e: usize| -> u8 {
        let bi = block / 8;
        let s = block % 8;
        let qs = (row * sb + bi) * bsz + QS_OFF;
        let byte = src[qs + (s / 2) * 32 + e];
        if s.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        }
    };
    for sr in 0..sr_count {
        for bi in 0..sb {
            for r in 0..8 {
                let off = ((8 * sr + r) * sb + bi) * bsz;
                let d = half::f16::from_le_bytes([src[off + D_OFF], src[off + D_OFF + 1]]).to_f32();
                let dmin = half::f16::from_le_bytes([src[off + DMIN_OFF], src[off + DMIN_OFF + 1]])
                    .to_f32();
                let scales_bytes: &[u8; 12] =
                    src[off + SC_OFF..off + SC_OFF + 12].try_into().unwrap();
                let (sc, mn) = crate::quant::decode_q4km_scales(scales_bytes);
                for s in 0..8 {
                    let block = bi * 8 + s;
                    dsc[(sr * nb32 + block) * 8 + r] = d * sc[s] as f32;
                    dmn[(sr * nb32 + block) * 8 + r] = dmin * mn[s] as f32;
                }
            }
            for s in 0..8usize {
                let block = bi * 8 + s;
                for g in 0..8usize {
                    for i in 0..16usize {
                        let e = 4 * g + (i % 4);
                        let lo = nibble(8 * sr + i / 4, block, e);
                        let hi = nibble(8 * sr + 4 + i / 4, block, e);
                        packed[(sr * nb32 + block) * 128 + g * 16 + i] = lo | (hi << 4);
                    }
                }
            }
        }
    }
    (packed, dsc, dmn)
}

/// Whether a Q4_K_M weight of shape `m x k` should be repacked for prefill on
/// this host. Like [`q4_0_repack_supported`], but K-quants need whole
/// super-blocks (`k % 256 == 0`).
#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
pub(crate) fn q4_k_repack_supported(m: usize, k: usize) -> bool {
    m.is_multiple_of(8) && k.is_multiple_of(256) && int8_gemm_available()
}

/// Run the repacked-Q4_K prefill GEMM on whichever x86 int8 tier this host has.
/// Returns `true` when a kernel ran; see [`gemm_preq_repacked_q4_0_dispatch`] for
/// why a `false` here is still an invariant break the caller must handle.
#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_preq_repacked_q4_k_dispatch(
    packed: &[u8],
    dsc: &[f32],
    dmn: &[f32],
    b_scales: &[f32],
    b_quants: &[i8],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let nb32 = k / 32;
    assert!(
        k.is_multiple_of(256) && m.is_multiple_of(8),
        "gemm_preq_repacked_q4_k_dispatch: need k%256==0 and m%8==0, got m={m} k={k}"
    );
    assert!(
        packed.len() >= (m / 8) * nb32 * 128
            && dsc.len() >= (m / 8) * nb32 * 8
            && dmn.len() >= (m / 8) * nb32 * 8,
        "gemm_preq_repacked_q4_k_dispatch: repacked weights too small for {m}x{k}"
    );
    assert!(
        b_quants.len() >= n * k && b_scales.len() >= n * nb32 && out.len() == m * n,
        "gemm_preq_repacked_q4_k_dispatch: activation/output buffers wrong for {m}x{n}x{k}"
    );

    #[cfg(feature = "avx512")]
    if vnni_int8_available() {
        // SAFETY: `vnni_int8_available()` proved the VNNI tier; the kernel
        // re-asserts its own length invariants in debug.
        unsafe {
            crate::backend::simd::avx512_vnni::gemm_q4_k_8x8_q8_0(
                packed, dsc, dmn, b_scales, b_quants, out, m, n, k,
            );
        }
        return true;
    }
    if avx2_int8_available() {
        // SAFETY: `avx2_int8_available()` proved avx2+fma; same as above.
        unsafe {
            crate::backend::simd::avx2_int8::gemm_q4_k_8x8_q8_0(
                packed, dsc, dmn, b_scales, b_quants, out, m, n, k,
            );
        }
        return true;
    }
    false
}

/// Batched pre-quantized GEMM: `out[m,n] = A_q[m,k] @ B_q8_0[k,n]`.
/// Returns `false` when no kernel on this host can compute `dtype`.
///
/// The arch dispatch for `transformer::gemm_preq`, kept here so the model layer
/// names one function rather than one per architecture.
#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
// `#[doc(hidden)] pub` rather than `pub(crate)` so a dedicated integration-test
// binary can drive it. The invariant worth testing here — decode and batched
// prefill produce the same bits — is only interesting at a *forced* CPU tier,
// and `CERA_CPU_TIER` is read once per process into a `OnceLock`, so it cannot
// be exercised from a unit test sharing a process with 300 others. Not part of
// the supported API; same pattern as `transformer::oracle_dump`.
#[doc(hidden)]
pub fn gemm_preq_dispatch(
    dtype: DType,
    data: &[u8],
    b_scales: &[f32],
    b_quants: &[i8],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    // `assert!`, not `debug_assert!`: this is a safe `pub` fn (see the note
    // above) and every kernel below indexes unchecked off these lengths. A
    // release build with an inconsistent m/n/k would read out of bounds — UB
    // reached from safe code. O(1) against an O(m*n*k) GEMM.
    let blocks = k / dtype.block_size();
    assert!(
        k.is_multiple_of(dtype.block_size()),
        "gemm_preq_dispatch: k={k} is not a multiple of {:?}'s block size",
        dtype
    );
    assert!(
        data.len() >= m * blocks * dtype.block_bytes(),
        "gemm_preq_dispatch: weights are {} bytes, need {} for {m}x{k} {:?}",
        data.len(),
        m * blocks * dtype.block_bytes(),
        dtype
    );
    // `out` is `==`, not `>=`: the kernels derive their strip/row index from
    // `out.len()` rather than from `m`, so an over-long output buffer walks past
    // row `m` and reads weights out of bounds. `>=` here would read as a
    // guarantee it does not provide — and would also undercut the `data` assert
    // above, whose sufficiency depends on the row count being exactly `m`.
    assert!(
        b_quants.len() >= n * k && b_scales.len() >= n * (k / 32) && out.len() == m * n,
        "gemm_preq_dispatch: activation/output buffers wrong for {m}x{n}x{k} \
         (quants {}, scales {}, out {} — out must be exactly {})",
        b_quants.len(),
        b_scales.len(),
        out.len(),
        m * n
    );

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use crate::backend::simd::neon;
        match dtype {
            DType::Q4_0 => {
                neon::gemm_q4_0_q8_0_neon(data, b_scales, b_quants, out, m, n, k);
                true
            }
            DType::Q8_0 => {
                neon::gemm_q8_0_q8_0_neon(data, b_scales, b_quants, out, m, n, k);
                true
            }
            // K-quants are dotprod-only and 256-aligned, so these can decline
            // at runtime even though the dtype is known.
            DType::Q4KM => neon::gemm_q4_k_q8_0_neon(data, b_scales, b_quants, out, m, n, k),
            DType::Q6K => neon::gemm_q6_k_q8_0_neon(data, b_scales, b_quants, out, m, n, k),
            _ => false,
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // The two x86 tiers run the *same* kernel bodies — `int8_gemm_kernels!`
        // instantiates them once for VNNI and once for AVX2 under identical
        // names — so the dtype allowlist is written once here. Spelling it out
        // per tier is how a newly supported dtype ends up wired on one tier and
        // silently declined on the other.
        #[cfg(target_arch = "x86_64")]
        macro_rules! x86_int8_gemm {
            ($m:path) => {{
                use $m as kern;
                match dtype {
                    DType::Q4_0 => {
                        kern::gemm_q4_0_q8_0(data, b_scales, b_quants, out, m, n, k);
                        true
                    }
                    DType::Q8_0 => {
                        kern::gemm_q8_0_q8_0(data, b_scales, b_quants, out, m, n, k);
                        true
                    }
                    DType::Q4KM => {
                        kern::gemm_q4_k_q8_0(data, b_scales, b_quants, out, m, n, k);
                        true
                    }
                    DType::Q6K => {
                        kern::gemm_q6_k_q8_0(data, b_scales, b_quants, out, m, n, k);
                        true
                    }
                    _ => false,
                }
            }};
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if vnni_int8_available() {
            // SAFETY: `vnni_int8_available()` proved the VNNI tier; the kernels
            // re-assert their own length invariants in debug.
            return unsafe { x86_int8_gemm!(crate::backend::simd::avx512_vnni) };
        }

        // No VNNI: the same kernels, with `dpbusd` emulated on AVX2. Reached by
        // every Zen 1-3 / pre-Ice-Lake host, and by Skylake-X (AVX-512, no
        // VNNI). Needs no `avx512` crate feature.
        #[cfg(target_arch = "x86_64")]
        if avx2_int8_available() {
            // SAFETY: `avx2_int8_available()` proved avx2+fma; the kernels
            // re-assert their own length invariants in debug.
            return unsafe { x86_int8_gemm!(crate::backend::simd::avx2_int8) };
        }
        false
    }
}

/// Quantize f32 vector to Q8_0 format for use with `gemv_q4_0_with_q8`.
/// Returns (scales, quants). On aarch64, uses NEON-vectorized quantization.
#[cfg(target_arch = "aarch64")]
pub fn quantize_f32_to_q8_0(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    assert_eq!(
        x.len() % 32,
        0,
        "quantize_f32_to_q8_0: x.len() must be divisible by 32"
    );
    let n_blocks = x.len() / 32;
    let mut scales = vec![0.0f32; n_blocks];
    let mut quants = vec![0i8; x.len()];
    unsafe {
        crate::backend::simd::neon::quantize_f32_to_q8_0_neon(x, &mut scales, &mut quants);
    }
    (scales, quants)
}

/// GEMV with pre-quantized Q8_0 input. Dispatches to Q4_0 or Q8_0 integer path.
/// For other dtypes, falls back to the regular f32 path.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn gemv_with_preq(
    dtype: DType,
    a_quant: &[u8],
    x_scales: &[f32],
    x_quants: &[i8],
    x_f32: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
) {
    match dtype {
        DType::Q4_0 => gemv_q4_0_with_q8(a_quant, x_scales, x_quants, y, m, k),
        DType::Q8_0 => unsafe {
            crate::backend::simd::neon::gemv_q8_0_q8_0_neon(a_quant, x_scales, x_quants, y, m, k)
        },
        DType::Q6K => unsafe {
            crate::backend::simd::neon::gemv_q6k_q8_0_neon(a_quant, x_scales, x_quants, y, m, k)
        },
        // NOTE: no Q4KM arm here on purpose. Routing Q4_K through a pre-quantized
        // dispatcher measured a consistent ~5% *regression* vs re-quantizing in
        // `gemv_dispatch` (interleaved A/B, LFM2.5-350M-Q4_K_M decode) — the
        // per-call Q8_0 quantization is cheap next to the GEMV, and the shared-
        // buffer path loses activation cache locality. Q4_K falls through below.
        _ => gemv_dispatch(dtype, a_quant, x_f32, y, m, k, None),
    }
}

/// Q4_0 GEMV with pre-quantized Q8_0 input. Avoids re-quantizing x when
/// the same input is used for multiple weight matrices (e.g., ffn_gate + ffn_up).
#[cfg(target_arch = "aarch64")]
pub fn gemv_q4_0_with_q8(
    a_quant: &[u8],
    x_scales: &[f32],
    x_quants: &[i8],
    y: &mut [f32],
    m: usize,
    k: usize,
) {
    unsafe {
        crate::backend::simd::neon::gemv_q4_0_q8_0_neon(a_quant, x_scales, x_quants, y, m, k);
    }
}

#[allow(clippy::ptr_arg)]
/// Q8_0 GEMV: `y[m] = A_q8_0[m,k] @ x[k]`.
/// On aarch64, uses integer dot product (quantize x to Q8_0, then Q8_0 × Q8_0
/// with vdotq_s32 — ~4x fewer instructions than f32 widening path).
pub fn gemv_q8_0_f32(
    a_quant: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scales: &mut Vec<f32>,
    q8_quants: &mut Vec<i8>,
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 32, 0, "Q8_0 GEMV: k must be divisible by 32");

    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * size_of::<BlockQ8_0>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::backend::simd::neon::gemv_q8_0_f32_neon(
                a_quant, x, y, m, k, q8_scales, q8_quants,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // x86 VNNI int8 path — see the note in `gemv_q4_0_f32`.
        // `vnni_int8_available()`, not a hand-rolled tier compare: this predicate
        // and the one selecting the quantizer must not drift, and since the AVX2
        // int8 kernels landed they are two different predicates.
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if vnni_int8_available() {
            q8_scales.resize(blocks_per_row, 0.0);
            q8_quants.resize(k, 0);
            // SAFETY: the tier predicate above proved the kernel's feature set,
            // and the scratch was just sized to the lengths it asserts.
            // `quantize_f32_to_q8_0_into`, not a per-tier quantizer: it already
            // dispatches, and naming it here is what makes decode and prefill
            // provably quantize through the same function.
            unsafe {
                quantize_f32_to_q8_0_into(x, q8_scales, q8_quants);
                crate::backend::simd::avx512_vnni::gemv_q8_0_q8_0(
                    a_quant, q8_scales, q8_quants, y, m, k,
                );
            }
            return;
        }

        // No VNNI but AVX2: the emulated int8 GEMV. Decode has to take the same
        // arithmetic as batched prefill or the parity bar breaks — see
        // `avx2_int8::gemv_q4k_f32` for the measurement behind that, and
        // `tests/avx2_decode_prefill_identity.rs` for the guard.
        #[cfg(target_arch = "x86_64")]
        if avx2_int8_available() {
            q8_scales.resize(blocks_per_row, 0.0);
            q8_quants.resize(k, 0);
            // SAFETY: the tier predicate above proved the kernel's feature set,
            // and the scratch was just sized to the lengths it asserts.
            // `quantize_f32_to_q8_0_into`, not a per-tier quantizer: it already
            // dispatches, and naming it here is what makes decode and prefill
            // provably quantize through the same function.
            unsafe {
                quantize_f32_to_q8_0_into(x, q8_scales, q8_quants);
                crate::backend::simd::avx2_int8::gemv_q8_0_q8_0(
                    a_quant, q8_scales, q8_quants, y, m, k,
                );
            }
            return;
        }

        let _ = (q8_scales, q8_quants);
        // The AVX-512 f32 row-dot that used to sit here is gone for the same
        // measured reason it is gone from `gemv_q4_0_f32` — see the note there.
        let compute_row = |(i, yi): (usize, &mut f32)| {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let offset = row_start + bi * size_of::<BlockQ8_0>();
                let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ8_0) };
                sum += vec_dot_q8_0_f32(block, &x[bi * 32..(bi + 1) * 32]);
            }
            *yi = sum;
        };

        if m >= gemv_par_threshold() {
            par_rows(y, gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }
}

/// Q6_K GEMV: `y[m] = A_q6k[m,k] @ x[k]`. Parallelized across rows.
/// On aarch64, quantizes x to Q8_0 then uses integer Q6_K × Q8_0 dot product with vdotq_s32.
#[allow(clippy::ptr_arg)]
#[allow(unused_variables)]
pub fn gemv_q6k_f32(
    a_quant: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scales: &mut Vec<f32>,
    q8_quants: &mut Vec<i8>,
) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 256, 0, "Q6_K GEMV: k must be divisible by 256");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::backend::simd::neon::gemv_q6k_f32_neon(
                a_quant, x, y, m, k, q8_scales, q8_quants,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * size_of::<BlockQ6K>();
        debug_assert_eq!(a_quant.len(), m * row_bytes);

        let compute_row = |(i, yi): (usize, &mut f32)| {
            let row_start = i * row_bytes;
            let mut sum = 0.0f32;
            for bi in 0..blocks_per_row {
                let offset = row_start + bi * size_of::<BlockQ6K>();
                let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ6K) };
                sum += vec_dot_q6_k_f32(block, &x[bi * 256..(bi + 1) * 256]);
            }
            *yi = sum;
        };

        if m >= gemv_par_threshold() {
            par_rows(y, gemv_min_rows(), compute_row);
        } else {
            y.iter_mut().enumerate().for_each(compute_row);
        }
    }
}

/// Q4_K_M GEMV: `y[m] = A_q4km[m,k] @ x[k]`. Parallelized across rows.
pub fn gemv_q4km_f32(a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 256, 0, "Q4_K_M GEMV: k must be divisible by 256");
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * size_of::<BlockQ4KM>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    let compute_row = |(i, yi): (usize, &mut f32)| {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        for bi in 0..blocks_per_row {
            let offset = row_start + bi * size_of::<BlockQ4KM>();
            let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ4KM) };
            sum += vec_dot_q4_k_m_f32(block, &x[bi * 256..(bi + 1) * 256]);
        }
        *yi = sum;
    };

    if m >= gemv_par_threshold() {
        par_rows(y, gemv_min_rows(), compute_row);
    } else {
        y.iter_mut().enumerate().for_each(compute_row);
    }
}

/// Q5_K GEMV: `y[m] = A_q5km[m,k] @ x[k]`. Parallelized across rows.
pub fn gemv_q5km_f32(a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 256, 0, "Q5_K GEMV: k must be divisible by 256");
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * size_of::<BlockQ5K>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    let compute_row = |(i, yi): (usize, &mut f32)| {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        for bi in 0..blocks_per_row {
            let offset = row_start + bi * size_of::<BlockQ5K>();
            let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ5K) };
            sum += vec_dot_q5_k_f32(block, &x[bi * 256..(bi + 1) * 256]);
        }
        *yi = sum;
    };

    if m >= gemv_par_threshold() {
        par_rows(y, gemv_min_rows(), compute_row);
    } else {
        y.iter_mut().enumerate().for_each(compute_row);
    }
}

/// Q4_1 GEMV: `y[m] = A_q4_1[m,k] @ x[k]`.
///
/// Scalar only. Q4_1 is a legacy ggml format with no SIMD or GPU kernels in
/// this tree — it appears almost exclusively as a stray `ffn_down` inside
/// otherwise-Q4_0 files, so it exists to make those files load and produce
/// correct output, not to be fast. Anything performance-sensitive should use
/// the Q4_0 or Q8_0 build of the same model.
pub fn gemv_q4_1_f32(a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    debug_assert_eq!(k % 32, 0, "Q4_1 GEMV: k must be divisible by 32");
    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * size_of::<BlockQ4_1>();
    debug_assert_eq!(a_quant.len(), m * row_bytes);

    let compute_row = |(i, yi): (usize, &mut f32)| {
        let row_start = i * row_bytes;
        let mut sum = 0.0f32;
        for bi in 0..blocks_per_row {
            let offset = row_start + bi * size_of::<BlockQ4_1>();
            // SAFETY: row `i` spans `a_quant[i*row_bytes ..][..row_bytes]`, and
            // the debug_assert above pins `a_quant.len()` to `m * row_bytes`.
            let block = unsafe { &*(a_quant.as_ptr().add(offset) as *const BlockQ4_1) };
            sum += vec_dot_q4_1_f32(block, &x[bi * 32..(bi + 1) * 32]);
        }
        *yi = sum;
    };

    if m >= gemv_par_threshold() {
        par_rows(y, gemv_min_rows(), compute_row);
    } else {
        y.iter_mut().enumerate().for_each(compute_row);
    }
}

/// F32 GEMV: `y[m] = A_f32[m,k] @ x[k]`.
pub fn gemv_f32(a: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);
    let a_f32: &[f32] = bytemuck::cast_slice(a);
    debug_assert_eq!(a_f32.len(), m * k);

    for i in 0..m {
        let row = &a_f32[i * k..(i + 1) * k];
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

/// Dispatch GEMV based on dtype: `y[m] = W[m,k] @ x[k]`.
/// For Q4_0, pass scratch buffers to avoid per-call allocation.
pub fn gemv_dispatch(
    dtype: DType,
    data: &[u8],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    q8_scratch: Option<(&mut Vec<f32>, &mut Vec<i8>)>,
) {
    // The K-quant arms below all say the same thing: run `$f` with the caller's
    // Q8_0 scratch when it lent us one, otherwise with a pair of local `Vec`s.
    // Written out, that is 12-18 lines per (dtype x tier) pair and six pairs;
    // the repetition is how the NEON, VNNI and AVX2 arms drift apart.
    //
    // `q8_scratch` is moved by the `Some` arm, which is sound only because each
    // expansion `return`s: the move sits on a diverging path, so a later
    // expansion still sees it live.
    // Cfg'd for the same reason `int8_gemm_kernels!` is: an uninvoked
    // `macro_rules!` is an `unused macro definition` warning on every target
    // with no SIMD K-quant GEMV (wasm32, riscv64), and the clippy leg that
    // would catch it runs on x86 only.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    macro_rules! kq_gemv {
        ($f:path) => {{
            match q8_scratch {
                Some((scales, quants)) => unsafe { $f(data, x, y, m, k, scales, quants) },
                None => {
                    let mut s = Vec::new();
                    let mut q = Vec::new();
                    unsafe { $f(data, x, y, m, k, &mut s, &mut q) }
                }
            }
            return;
        }};
    }

    match dtype {
        DType::Q4_0 => {
            if let Some((scales, quants)) = q8_scratch {
                gemv_q4_0_f32(data, x, y, m, k, scales, quants);
            } else {
                let mut s = Vec::new();
                let mut q = Vec::new();
                gemv_q4_0_f32(data, x, y, m, k, &mut s, &mut q);
            }
        }
        DType::Q8_0 => {
            if let Some((scales, quants)) = q8_scratch {
                gemv_q8_0_f32(data, x, y, m, k, scales, quants);
            } else {
                let mut s = Vec::new();
                let mut q = Vec::new();
                gemv_q8_0_f32(data, x, y, m, k, &mut s, &mut q);
            }
        }
        DType::Q4_1 => gemv_q4_1_f32(data, x, y, m, k),
        DType::F32 => gemv_f32(data, x, y, m, k),
        DType::Q6K => {
            #[cfg(target_arch = "aarch64")]
            kq_gemv!(crate::backend::simd::neon::gemv_q6k_f32_neon);
            #[cfg(not(target_arch = "aarch64"))]
            {
                // VNNI hosts share arithmetic with the batched GEMM (the GEMV
                // *is* the GEMM at n = 1), which the parity tests' tight naive
                // bar depends on.
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                if vnni_int8_available() {
                    kq_gemv!(crate::backend::simd::avx512_vnni::gemv_q6k_f32);
                }

                // Same invariant one tier down: the AVX2 int8 GEMV is the AVX2
                // GEMM at n = 1, so decode and prefill stay identical there too.
                #[cfg(target_arch = "x86_64")]
                if avx2_int8_available() {
                    kq_gemv!(crate::backend::simd::avx2_int8::gemv_q6k_f32);
                }
                let mut s = Vec::new();
                let mut q = Vec::new();
                gemv_q6k_f32(data, x, y, m, k, &mut s, &mut q);
            }
        }
        DType::Q4KM => {
            #[cfg(target_arch = "aarch64")]
            kq_gemv!(crate::backend::simd::neon::gemv_q4k_f32_neon);
            #[cfg(not(target_arch = "aarch64"))]
            {
                // See the Q6K arm: int8 GEMV shared with the batched GEMM.
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                if vnni_int8_available() {
                    kq_gemv!(crate::backend::simd::avx512_vnni::gemv_q4k_f32);
                }

                // Same invariant one tier down: the AVX2 int8 GEMV is the AVX2
                // GEMM at n = 1, so decode and prefill stay identical there too.
                #[cfg(target_arch = "x86_64")]
                if avx2_int8_available() {
                    kq_gemv!(crate::backend::simd::avx2_int8::gemv_q4k_f32);
                }
                gemv_q4km_f32(data, x, y, m, k);
            }
        }
        DType::Q5KM => gemv_q5km_f32(data, x, y, m, k),
        _ => panic!("gemv_dispatch: unsupported dtype {:?}", dtype),
    }
}

// ── Normalization ───────────────────────────────────────────────────────────

/// RMS normalization in-place: x = x / rms(x) * weight.
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();

    // Accumulate sum of squares in f64 to match ggml's ggml_float (double) precision.
    // This avoids f32 rounding that compounds across layers.
    let mut sum_sq = 0.0f64;
    for &v in x.iter() {
        sum_sq += (v as f64) * (v as f64);
    }
    let mean = sum_sq / n as f64;
    let rms = (mean + eps as f64).sqrt();
    let inv_rms = (1.0 / rms) as f32;

    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// ── Exp approximation ──────────────────────────────────────────────────────

/// Polynomial exp approximation matching ggml's `ggml_v_expf` (ARM optimized routine).
/// Maximum error: 1.45358 + 0.5 ULPs.
/// Inputs above 88.38 flush to infinity, below -103.97 flush to zero.
#[inline(always)]
fn ggml_expf(x: f32) -> f32 {
    // Bit-exact constants from ggml's hex float literals.
    const R: f32 = f32::from_bits(0x4B400000); // 0x1.8p23       = 12582912.0
    const LOG2E: f32 = f32::from_bits(0x3FB8AA3B); // 0x1.715476p+0  = log2(e)
    const LN2_HI: f32 = f32::from_bits(0x3F317200); // 0x1.62e4p-1    = ln(2) high
    const LN2_LO: f32 = f32::from_bits(0x35BFBE8E); // 0x1.7f7d1cp-20 = ln(2) low
    const C1: f32 = f32::from_bits(0x3F7FFFF6); // 0x1.ffffecp-1  ≈ 1/1!
    const C2: f32 = f32::from_bits(0x3EFFFEDB); // 0x1.fffdb6p-2  ≈ 1/2!
    const C3: f32 = f32::from_bits(0x3E2AAF33); // 0x1.555e66p-3  ≈ 1/3!
    const C4: f32 = f32::from_bits(0x3D2B9F17); // 0x1.573e2ep-5  ≈ 1/4!
    const C5: f32 = f32::from_bits(0x3C072010); // 0x1.0e4020p-7  ≈ 1/5!

    // n = round(x / ln2) via magic number trick
    let z = R + x * LOG2E;
    let n = z - R;

    // Cody-Waite range reduction: b = x - n*ln2
    let b = x - n * LN2_HI - n * LN2_LO;

    // 2^n via integer bit manipulation
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));

    // Polynomial approximation of exp(b) - 1 (Estrin's scheme)
    let u = b * b;
    let j = C1 * b + (C2 + C3 * b + (C4 + C5 * b) * u) * u;

    // Combine: result = k * (1 + j) = 2^n * exp(b)
    let abs_n = f32::from_bits(n.to_bits() & 0x7FFF_FFFF);

    if abs_n <= 126.0 {
        k + j * k
    } else if abs_n > 192.0 {
        if n > 0.0 { f32::INFINITY } else { 0.0 }
    } else {
        let d = if n <= 0.0 { 0x82000000u32 } else { 0u32 };
        let s1 = f32::from_bits(d.wrapping_add(0x7f000000));
        let s2 = f32::from_bits(e.wrapping_sub(d));
        (s2 + s2 * j) * s1
    }
}

// ── Activation functions ────────────────────────────────────────────────────

/// SiLU (Swish) activation in-place: x = x * sigmoid(x).
/// Uses ggml's polynomial exp approximation to match ggml's NEON silu path.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + ggml_expf(-*v));
    }
}

/// ReLU activation in-place: `x = max(x, 0)`. Used between the
/// LFM2A conv subsampling stem layers; trivial enough to inline,
/// but kept here so it's grep-able and can be SIMD-replaced later
/// without touching call sites.
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Fused SiLU activation + element-wise multiply: gate = silu(gate) * up.
/// Single pass instead of separate silu_inplace + mul_inplace.
pub fn silu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = *g / (1.0 + ggml_expf(-*g)) * u;
    }
}

/// Sigmoid activation in-place: `x = 1 / (1 + exp(-x))`. Uses
/// `ggml_expf` for the inner exponential to match the SiLU /
/// softmax precision pattern.
pub fn sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + ggml_expf(-*v));
    }
}

/// Gated Linear Unit split-and-gate. Reads a 2N-element `input`;
/// writes the N-element result `output[i] = a[i] * sigmoid(b[i])`
/// where `a = input[..N]` and `b = input[N..]`. Writes to a
/// separate `output` buffer (not in-place over `input`) — name
/// omits the `_inplace` suffix to reflect that, matching the
/// `conv1d` precedent in this file.
///
/// Inlines the sigmoid rather than calling `sigmoid_inplace` so
/// the gate fuses with the multiply in one pass over each
/// element — same shape `silu_mul_inplace` uses for the SiLU
/// gate.
///
/// The Conformer audio encoder's per-block conv module starts
/// with `conv_pw1` projecting to `2 * channels`, then this GLU
/// gate halves it back to `channels`. Output channel count =
/// input channel count / 2.
pub fn glu_split(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len() % 2, 0);
    let half = input.len() / 2;
    debug_assert_eq!(output.len(), half);
    let (a, b) = input.split_at(half);
    for i in 0..half {
        let gate = 1.0 / (1.0 + ggml_expf(-b[i]));
        output[i] = a[i] * gate;
    }
}

// ── Softmax ─────────────────────────────────────────────────────────────────

/// Softmax in-place over a 1D slice.
/// Uses ggml's polynomial exp approximation and f64 accumulation to match ggml exactly.
pub fn softmax_inplace(x: &mut [f32]) {
    // Find max for numerical stability
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Exponentiate using ggml's polynomial exp and sum with f64 (matches ggml_float)
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = ggml_expf(*v - max);
        sum += *v as f64;
    }

    // Normalize
    let inv_sum = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

// ── LayerNorm + GELU (Conformer audio encoder kernels) ────────────────────

/// Affine LayerNorm in-place: `x = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias`.
///
/// Distinct from `rmsnorm`: subtracts the mean (LayerNorm vs RMSNorm) and
/// adds an explicit `bias` term. Used by the Conformer audio encoder which
/// follows Whisper's affine-LayerNorm convention rather than LFM2's
/// RMSNorm. f64 accumulation matches `rmsnorm`'s precision approach.
pub fn layer_norm_inplace(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), bias.len());
    let n = x.len();
    if n == 0 {
        return;
    }

    // Mean + variance in f64.
    let mut sum = 0.0f64;
    for &v in x.iter() {
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut var_sum = 0.0f64;
    for &v in x.iter() {
        let d = v as f64 - mean;
        var_sum += d * d;
    }
    let var = var_sum / n as f64;
    let inv_std = (1.0 / (var + eps as f64).sqrt()) as f32;
    let mean_f32 = mean as f32;

    for i in 0..n {
        x[i] = (x[i] - mean_f32) * inv_std * weight[i] + bias[i];
    }
}

/// erf-form GELU activation in-place:
/// `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, using a polynomial
/// approximation of `erf` (max abs error ~1.5e-7) — close enough that
/// downstream f32 activations carry the precision floor, but not the
/// "exact GELU" of higher-precision libm impls.
///
/// Used by the Conformer audio encoder's MLP adapter (`mm.a.mlp`).
/// LFM2's main path uses SiLU (`silu_inplace`); the audio encoder is
/// the only consumer of GELU today, so this lives in the
/// encoder-kernels section rather than next to SiLU.
pub fn gelu_erf_inplace(x: &mut [f32]) {
    // sqrt(2)^-1 ≈ 0.7071067811865476
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erff(*v * INV_SQRT_2));
    }
}

/// tanh-approximation GELU activation in-place:
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
/// This is what `ggml_gelu` (i.e. llama.cpp's default GELU) computes,
/// and what every CLIP-family ViT trained with `clip.use_gelu = true`
/// in the GGUF metadata expects. Differs from
/// [`gelu_erf_inplace`] (the exact erf-form) by ~1e-3 relative around
/// |x| ≈ 1; that gap accumulates over many MLP layers, so picking the
/// wrong variant degrades downstream output noticeably even though
/// each individual call looks fine.
pub fn gelu_inplace(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEF: f32 = 0.044_715;
    for v in x.iter_mut() {
        let xv = *v;
        let inner = SQRT_2_OVER_PI * (xv + COEF * xv * xv * xv);
        *v = 0.5 * xv * (1.0 + inner.tanh());
    }
}

/// Approximation of `erf(x)` for `f32`. Abramowitz & Stegun 7.1.26
/// form, max abs error ~1.5e-7. `f32::erf` isn't in stable `std`, and
/// adding `libm` for one function would break the "no extra math deps
/// where cera has a hand-rolled equivalent" pattern (`ggml_expf` set
/// the precedent). Uses `ggml_expf` for the inner exponential to stay
/// consistent with that pattern.
#[inline(always)]
fn erff(x: f32) -> f32 {
    // Constants from Abramowitz & Stegun 7.1.26 (truncated to f32
    // precision; original tabulated values have more digits but they
    // round at f32 anyway).
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_7;
    const A3: f32 = 1.421_413_7;
    const A4: f32 = -1.453_152;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + P * abs_x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * ggml_expf(-abs_x * abs_x);
    sign * y
}

/// 1D convolution along the time dimension. Writes to a separate
/// `output` buffer (not in-place over `input`) — name omits the
/// `_inplace` suffix to reflect that.
///
/// Generic enough to cover standard, depthwise, and grouped conv1d
/// via the `groups` argument:
///
/// - `groups = 1`: standard conv — every output channel sees every
///   input channel.
/// - `groups = in_channels` and `out_channels = in_channels`: pure
///   depthwise conv — one kernel per channel, no cross-channel
///   mixing.
/// - `groups = in_channels` and `out_channels = in_channels × M`
///   for some integer multiplier `M`: depthwise with channel
///   multiplier (each input channel produces `M` output channels).
/// - Any other `groups` value that divides both `in_channels` and
///   `out_channels`: grouped conv — each group sees
///   `in_channels / groups` input channels.
///
/// Layout:
/// - `input`:  `[in_channels × t_in]`, row-major (channel-major).
/// - `weight`: `[out_channels × (in_channels / groups) × kernel_size]`,
///   row-major. Matches the GGUF `[O, I/G, K]` shape that the Conformer
///   stem (`a.conv1d.{i}.weight`) and per-block depthwise conv
///   (`a.blk.{i}.conv_dw.weight`) ship.
/// - `bias`:   `Some(&[out_channels])` to add a per-output-channel
///   bias; `None` for a bias-less layer (no allocation needed).
/// - `output`: `[out_channels × t_out]`, written by this fn. Caller
///   sizes it; the fn computes `t_out` from the standard formula and
///   returns it for sanity assertion.
///   `t_out = ((t_in + 2*pad - kernel_size) / stride) + 1`.
///
/// Padding is symmetric (same on both ends); causal/asymmetric padding
/// is the caller's job (zero-pad `input` before calling).
///
/// Algorithm is the textbook im2col-free direct convolution. Not SIMD
/// optimized — the encoder runs once per audio chunk so per-chunk
/// throughput dominates over per-frame latency. Metal acceleration
/// lands in a follow-up PR using the same shape signature.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    t_in: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    groups: usize,
) -> usize {
    debug_assert!(stride > 0, "stride must be > 0");
    debug_assert!(groups > 0, "groups must be > 0");
    debug_assert!(kernel_size > 0, "kernel_size must be > 0");
    debug_assert_eq!(input.len(), in_channels * t_in);
    debug_assert!(in_channels.is_multiple_of(groups));
    debug_assert!(out_channels.is_multiple_of(groups));
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_channels);
    }
    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;
    debug_assert_eq!(weight.len(), out_channels * in_per_group * kernel_size);

    let padded_t_in = t_in + 2 * pad;
    debug_assert!(padded_t_in >= kernel_size, "kernel exceeds padded input");
    let t_out = (padded_t_in - kernel_size) / stride + 1;
    debug_assert_eq!(output.len(), out_channels * t_out);

    for g in 0..groups {
        for oc_local in 0..out_per_group {
            let oc = g * out_per_group + oc_local;
            let bias_v = bias.map_or(0.0, |b| b[oc]);

            // Pre-bias every output position for this channel — also
            // doubles as the zero-init pass since the accumulator
            // below uses `+=` over multiple input channels.
            let out_row_start = oc * t_out;
            output[out_row_start..out_row_start + t_out].fill(bias_v);

            for ic_local in 0..in_per_group {
                let ic = g * in_per_group + ic_local;
                // Weight row layout: [oc, ic_local, k]
                let weight_row_start = oc * in_per_group * kernel_size + ic_local * kernel_size;

                for ot in 0..t_out {
                    let mut acc = 0.0f32;
                    for k in 0..kernel_size {
                        // Position in the (conceptually padded) input.
                        let padded_pos = ot * stride + k;
                        // Translate back to the unpadded input. Out-of-
                        // bounds samples are zero (= no contribution).
                        if padded_pos >= pad && padded_pos < padded_t_in - pad {
                            let it = padded_pos - pad;
                            let w = weight[weight_row_start + k];
                            let x = input[ic * t_in + it];
                            acc += w * x;
                        }
                    }
                    output[out_row_start + ot] += acc;
                }
            }
        }
    }

    t_out
}

/// 2D convolution. Generalizes `conv1d` to two spatial axes;
/// covers all the modes the LFM2A conv subsampling stem needs:
/// - Regular conv (`groups = 1`).
/// - Depthwise conv (`groups = in_channels`). The depthwise fast
///   path further requires `out_channels == in_channels`; a
///   depthwise channel-multiplier (`out_channels = in_channels * M`,
///   `M > 1`) is supported by the naive 7-loop fallback.
/// - Pointwise conv (`kh = kw = 1`).
///
/// Three fast paths land before the naive 7-loop fallback:
///
/// 1. **Pointwise** (`kh = kw = 1`, stride 1, no pad, groups 1):
///    dispatch directly to a gemm — pointwise conv is mathematically
///    a per-position matmul.
/// 2. **Regular k×k** (`groups == 1`, kernel > 1×1): im2col the
///    input into `[in_ch * kh * kw, plane_out]`, then dispatch the
///    resulting `[out_ch × kk·ic] @ [kk·ic × plane_out]` gemm.
/// 3. **Depthwise** (`groups == in_channels == out_channels`):
///    parallelize across channels with rayon (under the `parallel`
///    feature); each thread holds its own `[kh*kw, plane_out]`
///    im2col scratch and runs a flat per-channel matmul.
///
/// Paths 1 and 2 use `gemm_with_bias_broadcast`, which dispatches
/// to BLAS (Apple Accelerate / OpenBLAS) under the `blas` feature
/// and falls back to scalar `matmul_f32` otherwise.
///
/// See `tests/bench_conv2d.rs` for measured numbers on the LFM2A
/// audio encoder stem (320× cumulative speedup vs the naive baseline
/// in the BLAS build, 42× in the default-feature scalar build).
/// Inputs that don't match any fast path fall through to the naive
/// 7-loop — the LFM2A stem itself doesn't hit it.
///
/// Layouts (all row-major, channel-major outer):
/// - `input`: `[in_channels, h_in, w_in]`.
/// - `weight`: `[out_channels, in_per_group, kh, kw]` where
///   `in_per_group = in_channels / groups`.
/// - `bias`: `[out_channels]` if present.
/// - `output`: `[out_channels, h_out, w_out]` where
///   `h_out = (h_in + 2 * pad_h - kh) / stride_h + 1` and
///   `w_out = (w_in + 2 * pad_w - kw) / stride_w + 1`.
///
/// Returns `(h_out, w_out)`. Out-of-bounds reads from the conceptual
/// pad zone contribute zero (no explicit pad buffer materialized).
///
/// Dilation is fixed at 1 — the LFM2A stem doesn't use dilated
/// convs. Add a `(dil_h, dil_w)` parameter when a caller actually
/// needs it.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    groups: usize,
) -> (usize, usize) {
    debug_assert!(stride_h > 0, "stride_h must be > 0");
    debug_assert!(stride_w > 0, "stride_w must be > 0");
    debug_assert!(groups > 0, "groups must be > 0");
    debug_assert!(kh > 0, "kh must be > 0");
    debug_assert!(kw > 0, "kw must be > 0");
    debug_assert_eq!(input.len(), in_channels * h_in * w_in);
    debug_assert!(in_channels.is_multiple_of(groups));
    debug_assert!(out_channels.is_multiple_of(groups));
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_channels);
    }
    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;
    debug_assert_eq!(weight.len(), out_channels * in_per_group * kh * kw);

    // Checked size math. On 64-bit the bounds are astronomical
    // and these checks compile away when LLVM proves the inputs
    // sane; on 32-bit they catch silent wraps that would otherwise
    // mis-size the scratch allocations and panic later inside the
    // per-row indexing.
    let two_pad_h = pad_h
        .checked_mul(2)
        .expect("conv2d: 2 * pad_h overflowed usize");
    let two_pad_w = pad_w
        .checked_mul(2)
        .expect("conv2d: 2 * pad_w overflowed usize");
    let padded_h = h_in
        .checked_add(two_pad_h)
        .expect("conv2d: h_in + 2 * pad_h overflowed usize");
    let padded_w = w_in
        .checked_add(two_pad_w)
        .expect("conv2d: w_in + 2 * pad_w overflowed usize");
    debug_assert!(padded_h >= kh, "kh exceeds padded h_in");
    debug_assert!(padded_w >= kw, "kw exceeds padded w_in");
    let h_out = (padded_h - kh) / stride_h + 1;
    let w_out = (padded_w - kw) / stride_w + 1;
    let plane_in = h_in
        .checked_mul(w_in)
        .expect("conv2d: h_in * w_in overflowed usize");
    let plane_out = h_out
        .checked_mul(w_out)
        .expect("conv2d: h_out * w_out overflowed usize");
    let kernel_plane = kh
        .checked_mul(kw)
        .expect("conv2d: kh * kw overflowed usize");
    let total_out = out_channels
        .checked_mul(plane_out)
        .expect("conv2d: out_channels * plane_out overflowed usize");
    debug_assert_eq!(output.len(), total_out);

    // Compute `output[m × n] = weight[m × k] @ input[k × n] +
    // bias_broadcast`. Used by both the pointwise and im2col fast
    // paths below. Dispatches to BLAS (Apple Accelerate AMX or
    // OpenBLAS) when the `blas` feature is on; falls back to the
    // scalar `matmul_f32` with the bias-prefill trick otherwise.
    fn gemm_with_bias_broadcast(
        output: &mut [f32],
        weight: &[f32],
        input: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        #[cfg(feature = "blas")]
        {
            // sgemm overwrites C (alpha=1, beta=0). Bias is added
            // in a separate per-channel sweep — small relative to
            // the gemm cost.
            crate::backend::blas::sgemm_rowmajor_nn(m, n, k, weight, input, output);
            if let Some(b) = bias {
                for oc in 0..m {
                    let bias_v = b[oc];
                    for v in output[oc * n..(oc + 1) * n].iter_mut() {
                        *v += bias_v;
                    }
                }
            }
        }
        #[cfg(not(feature = "blas"))]
        {
            // matmul_f32 accumulates onto C — pre-fill with the
            // broadcast bias so the bias add lands for free.
            for oc in 0..m {
                let bias_v = bias.map_or(0.0, |b| b[oc]);
                output[oc * n..(oc + 1) * n].fill(bias_v);
            }
            matmul_f32(weight, input, output, m, n, k);
        }
    }

    // Fast path 1: pointwise convs (1x1, stride 1, no pad, groups 1).
    // Mathematically a per-position matmul; the naive 7-loop below
    // has terrible cache behavior for this case (~5s on the LFM2A
    // 30s-audio stem layer.3 vs ~80ms via matmul_f32). Pre-fill
    // output with the broadcast bias so the accumulating matmul
    // lands the bias add for free. weight `[out_ch × in_ch × 1 × 1]`
    // is already exactly `[m × k]` for matmul_f32(weight, input, out).
    let pointwise = kh == 1
        && kw == 1
        && stride_h == 1
        && stride_w == 1
        && pad_h == 0
        && pad_w == 0
        && groups == 1;
    if pointwise {
        gemm_with_bias_broadcast(
            output,
            weight,
            input,
            bias,
            out_channels,
            plane_out,
            in_channels,
        );
        return (h_out, w_out);
    }

    // Fast path 2: regular (non-grouped) k×k convs. Im2col the input
    // into `[in_ch * kh * kw, plane_out]`, then dispatch the
    // [out_ch × kk·ic] @ [kk·ic × plane_out] gemm to matmul_f32.
    // weight is already laid out as [out_ch × in_ch × kh × kw] —
    // matmul reads it as [m × k] with k = in_ch * kh * kw, matching
    // our im2col row decomposition `(ic * kh + ki) * kw + kj`.
    //
    // Memory: im2col is `kh * kw * in_ch * plane_out * 4` bytes.
    // For the LFM2A stem layer.0 at 30s (in_ch=1, plane_out=60K):
    // 2.16 MB — acceptable. Skip this path for depthwise / grouped
    // convs (the per-group matmuls would be too small to amortize
    // the im2col allocation; the naive path is plenty fast there).
    if groups == 1 {
        let cols = kernel_plane
            .checked_mul(in_channels)
            .expect("conv2d: kernel_plane * in_channels overflowed usize");
        let im2col_len = cols
            .checked_mul(plane_out)
            .expect("conv2d: im2col buffer size overflowed usize");
        let mut im2col = vec![0.0f32; im2col_len];
        for ic in 0..in_channels {
            let in_plane = ic * plane_in;
            for ki in 0..kh {
                for kj in 0..kw {
                    let row_idx = (ic * kh + ki) * kw + kj;
                    let im_row_start = row_idx * plane_out;
                    for oh in 0..h_out {
                        let pad_row = oh * stride_h + ki;
                        if pad_row < pad_h || pad_row >= h_in + pad_h {
                            // Whole row is zero — leave the
                            // pre-zeroed im2col untouched.
                            continue;
                        }
                        let ih = pad_row - pad_h;
                        let in_row_start = in_plane + ih * w_in;
                        let out_row_start = im_row_start + oh * w_out;
                        for ow in 0..w_out {
                            let pad_col = ow * stride_w + kj;
                            if pad_col >= pad_w && pad_col < w_in + pad_w {
                                let iw = pad_col - pad_w;
                                im2col[out_row_start + ow] = input[in_row_start + iw];
                            }
                        }
                    }
                }
            }
        }

        gemm_with_bias_broadcast(output, weight, &im2col, bias, out_channels, plane_out, cols);
        return (h_out, w_out);
    }

    // Fast path 3: depthwise convs (groups == in_channels ==
    // out_channels). Each channel is independent — parallelize
    // across channels with rayon under the `parallel` feature.
    // Per-channel work: build a `[kh * kw, plane_out]` im2col,
    // then dispatch a flat `[1 × kk] @ [kk × plane_out]` matmul
    // — eliminates the per-multiply bounds-check branch that
    // hurts the naive 7-loop on the LFM2A stem.
    //
    // Memory: under `parallel`, one im2col scratch per worker
    // thread via `for_each_init` (rayon-only API; the sequential
    // shim in `crate::par` doesn't expose it, so the wasm /
    // single-threaded path uses one scratch reused across the
    // sequential `chunks_mut` loop). For the LFM2A 30s layer.2:
    // 540KB per worker.
    let depthwise = groups == in_channels && out_channels == in_channels;
    if depthwise {
        let im2col_len = kernel_plane
            .checked_mul(plane_out)
            .expect("conv2d: depthwise im2col buffer size overflowed usize");
        // Per-channel work, factored into a closure so both the
        // parallel and sequential branches below share a body.
        let do_channel = |im2col: &mut [f32], ic: usize, out_chunk: &mut [f32]| {
            let bias_v = bias.map_or(0.0, |b| b[ic]);
            out_chunk.fill(bias_v);
            // No per-channel `im2col.fill(0.0)` needed: the per-thread
            // scratch is zero-initialized once at allocation, and the
            // "skip" branches in the pad-bounds checks below depend
            // only on (ki, kj, oh, ow, stride, pad, h_in, w_in) — not
            // on `ic`. So the same im2col slots are written every
            // channel and the same slots are skipped every channel,
            // meaning the skipped (padded-zone) cells retain their
            // initial 0 across the entire run.
            let in_plane = ic * plane_in;
            for ki in 0..kh {
                for kj in 0..kw {
                    let row_idx = ki * kw + kj;
                    let im_row_start = row_idx * plane_out;
                    for oh in 0..h_out {
                        let pad_row = oh * stride_h + ki;
                        if pad_row < pad_h || pad_row >= h_in + pad_h {
                            continue;
                        }
                        let ih = pad_row - pad_h;
                        let in_row_start = in_plane + ih * w_in;
                        let out_row_start = im_row_start + oh * w_out;
                        for ow in 0..w_out {
                            let pad_col = ow * stride_w + kj;
                            if pad_col >= pad_w && pad_col < w_in + pad_w {
                                let iw = pad_col - pad_w;
                                im2col[out_row_start + ow] = input[in_row_start + iw];
                            }
                        }
                    }
                }
            }

            // Per-channel kernel: a contiguous `kernel_plane`-wide
            // slice of `weight`. matmul_f32 accumulates onto the
            // bias-prefilled `out_chunk` so the bias add lands for
            // free.
            let w_start = ic * kernel_plane;
            matmul_f32(
                &weight[w_start..w_start + kernel_plane],
                im2col,
                out_chunk,
                1,
                plane_out,
                kernel_plane,
            );
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            output.par_chunks_mut(plane_out).enumerate().for_each_init(
                || vec![0.0f32; im2col_len],
                |im2col, (ic, out_chunk)| do_channel(im2col, ic, out_chunk),
            );
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut im2col = vec![0.0f32; im2col_len];
            for (ic, out_chunk) in output.chunks_mut(plane_out).enumerate() {
                do_channel(&mut im2col, ic, out_chunk);
            }
        }
        return (h_out, w_out);
    }

    for g in 0..groups {
        for oc_local in 0..out_per_group {
            let oc = g * out_per_group + oc_local;
            let bias_v = bias.map_or(0.0, |b| b[oc]);

            // Pre-bias every output position for this channel — also
            // doubles as the zero-init pass since the per-input-channel
            // accumulator below uses `+=`.
            let oc_offset = oc * plane_out;
            output[oc_offset..oc_offset + plane_out].fill(bias_v);

            for ic_local in 0..in_per_group {
                let ic = g * in_per_group + ic_local;
                // Weight layout per (oc, ic_local): [kh × kw], row-major.
                let w_oc_ic = (oc * in_per_group + ic_local) * kernel_plane;
                let in_plane = ic * plane_in;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0f32;
                        for ki in 0..kh {
                            let pad_row = oh * stride_h + ki;
                            // Translate back to the unpadded input.
                            // Skip rows entirely outside the unpadded
                            // window (their contribution is zero).
                            if pad_row < pad_h || pad_row >= h_in + pad_h {
                                continue;
                            }
                            let ih = pad_row - pad_h;
                            for kj in 0..kw {
                                let pad_col = ow * stride_w + kj;
                                if pad_col < pad_w || pad_col >= w_in + pad_w {
                                    continue;
                                }
                                let iw = pad_col - pad_w;
                                let w = weight[w_oc_ic + ki * kw + kj];
                                let x = input[in_plane + ih * w_in + iw];
                                acc += w * x;
                            }
                        }
                        output[oc_offset + oh * w_out + ow] += acc;
                    }
                }
            }
        }
    }

    (h_out, w_out)
}

// ── Attention score/value computation ───────────────────────────────────────

/// Compute attention scores for one head: `scores[t] = dot(q_head, k_cache_row_t) * scale`.
/// `k_cache` has stride `kv_dim` between timesteps; each key starts at offset `kv_h_offset`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attn_scores(
    q_head: &[f32],
    k_cache: &[f32],
    scores: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    seq_len: usize,
) {
    debug_assert!(q_head.len() >= head_dim);
    debug_assert!(scores.len() >= seq_len);
    if seq_len > 0 {
        debug_assert!(k_cache.len() >= (seq_len - 1) * kv_dim + kv_h_offset + head_dim);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        attn_scores_neon(
            q_head,
            k_cache,
            scores,
            kv_dim,
            kv_h_offset,
            head_dim,
            scale,
            seq_len,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    for t in 0..seq_len {
        let mut dot = 0.0f32;
        let k_off = t * kv_dim + kv_h_offset;
        for d in 0..head_dim {
            dot += q_head[d] * k_cache[k_off + d];
        }
        scores[t] = dot * scale;
    }
}

/// Compute weighted sum of V cache for one head: `attn_out[d] = sum_t(scores[t] * v[t,d])`.
/// `v_cache` has stride `kv_dim` between timesteps; each value starts at offset `kv_h_offset`.
#[allow(clippy::needless_range_loop)]
pub fn attn_values(
    scores: &[f32],
    v_cache: &[f32],
    attn_out: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    seq_len: usize,
) {
    debug_assert!(scores.len() >= seq_len);
    debug_assert!(attn_out.len() >= head_dim);
    if seq_len > 0 {
        debug_assert!(v_cache.len() >= (seq_len - 1) * kv_dim + kv_h_offset + head_dim);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        attn_values_neon(
            scores,
            v_cache,
            attn_out,
            kv_dim,
            kv_h_offset,
            head_dim,
            seq_len,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        attn_out[..head_dim].fill(0.0);
        for t in 0..seq_len {
            let s = scores[t];
            let v_base = t * kv_dim + kv_h_offset;
            for d in 0..head_dim {
                attn_out[d] += s * v_cache[v_base + d];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[target_feature(enable = "neon")]
unsafe fn attn_scores_neon(
    q_head: &[f32],
    k_cache: &[f32],
    scores: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    seq_len: usize,
) {
    use std::arch::aarch64::*;
    // Safety: caller ensures buffer bounds; intrinsics require unsafe in Edition 2024.
    unsafe {
        let q_ptr = q_head.as_ptr();
        let k_ptr = k_cache.as_ptr();

        // Pre-load Q vectors once (constant across all timesteps).
        // Max 32 float32x4 = head_dim up to 128. Stack array avoids heap alloc.
        const MAX_Q_VECS: usize = 32;
        let n_q_vecs = head_dim / 4;
        debug_assert!(n_q_vecs <= MAX_Q_VECS, "head_dim > 128 not supported");
        let mut q_vecs = [vdupq_n_f32(0.0); MAX_Q_VECS];
        for i in 0..n_q_vecs {
            q_vecs[i] = vld1q_f32(q_ptr.add(i * 4));
        }

        for t in 0..seq_len {
            let k_off = t * kv_dim + kv_h_offset;
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);

            let mut d = 0usize;
            let mut qi = 0usize;
            while d + 8 <= head_dim {
                let k0 = vld1q_f32(k_ptr.add(k_off + d));
                let k1 = vld1q_f32(k_ptr.add(k_off + d + 4));
                sum0 = vfmaq_f32(sum0, q_vecs[qi], k0);
                sum1 = vfmaq_f32(sum1, q_vecs[qi + 1], k1);
                d += 8;
                qi += 2;
            }
            if d + 4 <= head_dim {
                let k0 = vld1q_f32(k_ptr.add(k_off + d));
                sum0 = vfmaq_f32(sum0, q_vecs[qi], k0);
                d += 4;
            }
            let mut total = vaddvq_f32(vaddq_f32(sum0, sum1));
            while d < head_dim {
                total += *q_ptr.add(d) * *k_ptr.add(k_off + d);
                d += 1;
            }
            scores[t] = total * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "neon")]
unsafe fn attn_values_neon(
    scores: &[f32],
    v_cache: &[f32],
    attn_out: &mut [f32],
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    seq_len: usize,
) {
    use std::arch::aarch64::*;
    // Safety: caller ensures buffer bounds; intrinsics require unsafe in Edition 2024.
    unsafe {
        let v_ptr = v_cache.as_ptr();
        let out_ptr = attn_out.as_mut_ptr();

        // Accumulate in registers (not memory) across all timesteps, store once at end.
        // Max 32 float32x4 = head_dim up to 128.
        const MAX_ACC_VECS: usize = 32;
        let n_vec = head_dim / 4;
        let n_tail = head_dim % 4;
        debug_assert!(n_vec <= MAX_ACC_VECS, "head_dim > 128 not supported");
        let mut acc = [vdupq_n_f32(0.0); MAX_ACC_VECS];

        for t in 0..seq_len {
            let s = vdupq_n_f32(scores[t]);
            let v_base = t * kv_dim + kv_h_offset;
            for i in 0..n_vec {
                let v = vld1q_f32(v_ptr.add(v_base + i * 4));
                acc[i] = vfmaq_f32(acc[i], s, v);
            }
        }

        // Store accumulators to output
        for i in 0..n_vec {
            vst1q_f32(out_ptr.add(i * 4), acc[i]);
        }
        // Scalar tail
        let tail_start = n_vec * 4;
        for dd in 0..n_tail {
            let mut val = 0.0f32;
            for t in 0..seq_len {
                val += scores[t] * *v_ptr.add(t * kv_dim + kv_h_offset + tail_start + dd);
            }
            *out_ptr.add(tail_start + dd) = val;
        }
    }
}

// ── Flash attention (tiled, online softmax) ────────────────────────────────

const FLASH_TILE_KV: usize = 32;

/// Tiled flash attention for one KV head group (GQA).
///
/// Processes `group_size` query heads against a single KV head's cache. For
/// each query position, tiles over the KV cache with `FLASH_TILE_KV`-sized
/// chunks, using online softmax (running max + sum) so the full score vector
/// is never materialized.
///
/// **Layouts:**
/// - `q_mat`: `[hs, n]` stride-n (the batched projection output). Q for head
///   h, token j, dim d lives at `q_mat[(h * head_dim + d) * q_stride + j]`.
///   Gathered into a local contiguous array per query.
/// - `k_cache` / `v_cache`: `[total_seq, kv_dim]`, stride `kv_dim`. Position
///   t, dim d of KV head kv_h is at `cache[t * kv_dim + kv_h_offset + d]`.
/// - `out`: contiguous `[group_size, n_queries, head_dim]`. Element
///   `out[(g * n_queries + j) * head_dim + d]` is dim d of query j, group
///   member g. Caller is responsible for scatter-copying back to stride-n
///   layout if needed.
///
/// **Causal masking:** query at position `start_pos + j` attends only to KV
/// positions `0 .. start_pos + j` (inclusive). Tiles beyond the causal limit
/// are skipped entirely; individual positions within a boundary tile are
/// masked to `-INF` before the softmax update.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn flash_attention_gqa_cpu(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    // NEON kernel requires head_dim to be a multiple of 4 and <= 128.
    // Fall back to scalar for unsupported dimensions.
    #[cfg(target_arch = "aarch64")]
    {
        if head_dim % 4 == 0 && head_dim <= 128 {
            unsafe {
                flash_attention_gqa_neon(
                    q_mat,
                    k_cache,
                    v_cache,
                    out,
                    n_heads_start,
                    group_size,
                    n_queries,
                    q_stride,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    start_pos,
                );
            }
            return;
        }
    }
    // x86 AVX-512 kernel: 16-wide zmm, needs head_dim a multiple of 16. Checked
    // before the AVX2 path so an AVX-512 host uses the wider kernel; a head_dim
    // that is a multiple of 8 but not 16 falls through to AVX2 below.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if head_dim.is_multiple_of(16) && head_dim <= 256 && is_x86_feature_detected!("avx512f") {
            unsafe {
                flash_attention_gqa_avx512(
                    q_mat,
                    k_cache,
                    v_cache,
                    out,
                    n_heads_start,
                    group_size,
                    n_queries,
                    q_stride,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    start_pos,
                );
            }
            return;
        }
    }
    // x86 AVX2+FMA kernel: needs head_dim a multiple of 8 (one ymm = 8 f32) and
    // <= 256 (the acc/q register-array bound). Runtime-detected so a Haswell
    // baseline build still falls back to scalar on a host without AVX2. Q/K/V
    // are all f32 here (the KV cache is f32 on CPU), so this is plain FMA, not
    // an int8 kernel — no tier gate beyond the CPUID check.
    #[cfg(target_arch = "x86_64")]
    {
        if head_dim.is_multiple_of(8)
            && head_dim <= 256
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                flash_attention_gqa_avx2(
                    q_mat,
                    k_cache,
                    v_cache,
                    out,
                    n_heads_start,
                    group_size,
                    n_queries,
                    q_stride,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    start_pos,
                );
            }
            return;
        }
    }
    flash_attention_gqa_scalar(
        q_mat,
        k_cache,
        v_cache,
        out,
        n_heads_start,
        group_size,
        n_queries,
        q_stride,
        kv_dim,
        kv_h_offset,
        head_dim,
        scale,
        start_pos,
    );
}

#[allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]
fn flash_attention_gqa_scalar(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    // Stack-allocated scratch to avoid heap alloc contention in parallel
    // dispatch. 256 covers all known model head_dims (64, 128, 160, 256).
    // The NEON kernel falls back to this scalar path for head_dim > 128.
    assert!(
        head_dim <= 256,
        "flash_attention_gqa_scalar: head_dim {head_dim} > 256"
    );
    let mut q_buf = [0.0f32; 256];
    let mut acc_buf = [0.0f32; 256];
    let q_local = &mut q_buf[..head_dim];
    let acc = &mut acc_buf[..head_dim];
    let mut tile_scores = [0.0f32; FLASH_TILE_KV];

    for g in 0..group_size {
        let h = n_heads_start + g;
        let h_off = h * head_dim;

        for j in 0..n_queries {
            let max_kv = start_pos + j + 1;

            for d in 0..head_dim {
                q_local[d] = q_mat[(h_off + d) * q_stride + j];
            }

            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f64;
            acc.fill(0.0);

            for kv_start in (0..max_kv).step_by(FLASH_TILE_KV) {
                let kv_end = (kv_start + FLASH_TILE_KV).min(max_kv);
                let tile_len = kv_end - kv_start;

                for ti in 0..tile_len {
                    let k_off = (kv_start + ti) * kv_dim + kv_h_offset;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_local[d] * k_cache[k_off + d];
                    }
                    tile_scores[ti] = dot * scale;
                }

                let tile_max = tile_scores[..tile_len]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let new_max = running_max.max(tile_max);

                let rescale = if running_max > f32::NEG_INFINITY {
                    ggml_expf(running_max - new_max)
                } else {
                    0.0
                };

                let mut tile_sum = 0.0f64;
                for ti in 0..tile_len {
                    tile_scores[ti] = ggml_expf(tile_scores[ti] - new_max);
                    tile_sum += tile_scores[ti] as f64;
                }

                for d in 0..head_dim {
                    acc[d] *= rescale;
                }
                for ti in 0..tile_len {
                    let s = tile_scores[ti];
                    let v_off = (kv_start + ti) * kv_dim + kv_h_offset;
                    for d in 0..head_dim {
                        acc[d] += s * v_cache[v_off + d];
                    }
                }

                running_sum = running_sum * rescale as f64 + tile_sum;
                running_max = new_max;
            }

            let inv_sum = (1.0 / running_sum) as f32;
            let out_off = (g * n_queries + j) * head_dim;
            for d in 0..head_dim {
                out[out_off + d] = acc[d] * inv_sum;
            }
        }
    }
}

/// AVX2+FMA flash attention, structurally mirroring `flash_attention_gqa_neon`:
/// the QK dot and the weighted-V accumulate are vectorized (256-bit lanes), the
/// online-softmax bookkeeping stays scalar and calls the identical `ggml_expf`,
/// so the only numeric divergence from the scalar path is the summation order
/// of the two dot products — the same divergence NEON already has, well inside
/// the parity suite's cosine>0.99 flash bar.
///
/// Requires `head_dim % 8 == 0` and `head_dim <= 256` (both enforced by the
/// dispatcher). Q is gathered from the stride-`q_stride` column layout into a
/// contiguous buffer once per query, then loaded into registers.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available (the dispatcher CPUID-checks),
/// and that the buffers are sized for the head range, `q_stride`, `kv_dim`, and
/// `start_pos + n_queries` (same contract as the scalar/NEON kernels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn flash_attention_gqa_avx2(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        /// Horizontal sum of a `__m256` (matches `simd::hsum_avx`).
        #[target_feature(enable = "avx2")]
        unsafe fn hsum256(v: __m256) -> f32 {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let s128 = _mm_add_ps(lo, hi);
            let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
            let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
            _mm_cvtss_f32(s32)
        }

        // Buffer-sizing tripwires, mirroring `flash_attention_gqa_neon`. The
        // dispatcher checks `head_dim`, but the caller still owns the q/k/v/out
        // sizing contract; without these a violation is silent UB.
        debug_assert!(
            q_mat.len() >= ((n_heads_start + group_size) * head_dim - 1) * q_stride + n_queries,
            "q_mat too small for the given head range and q_stride"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || k_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "k_cache too small"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || v_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "v_cache too small"
        );
        debug_assert!(
            out.len() >= group_size * n_queries * head_dim,
            "out buffer too small for contiguous [group_size, n_queries, head_dim] output"
        );

        // 256 / 8 = 32 vectors max (head_dim <= 256, multiple of 8).
        const MAX_VECS: usize = 32;
        let n_vecs = head_dim / 8;

        let q_ptr = q_mat.as_ptr();
        let k_ptr = k_cache.as_ptr();
        let v_ptr = v_cache.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let mut q_vecs = [_mm256_setzero_ps(); MAX_VECS];
        let mut acc_vecs = [_mm256_setzero_ps(); MAX_VECS];
        let mut q_gather = [0.0f32; 256];
        let mut tile_scores = [0.0f32; FLASH_TILE_KV];

        for g in 0..group_size {
            let h = n_heads_start + g;
            let h_off = h * head_dim;

            for j in 0..n_queries {
                let max_kv = start_pos + j + 1;

                // Gather Q[h, j] from the stride-q_stride column layout into a
                // contiguous buffer, then into registers. One gather per query,
                // amortized over max_kv KV positions.
                for d in 0..head_dim {
                    q_gather[d] = *q_ptr.add((h_off + d) * q_stride + j);
                }
                for i in 0..n_vecs {
                    q_vecs[i] = _mm256_loadu_ps(q_gather.as_ptr().add(i * 8));
                    acc_vecs[i] = _mm256_setzero_ps();
                }

                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f64;

                for kv_start in (0..max_kv).step_by(FLASH_TILE_KV) {
                    let kv_end = (kv_start + FLASH_TILE_KV).min(max_kv);
                    let tile_len = kv_end - kv_start;

                    // QK dot products for the tile. Two independent lane
                    // accumulators to hide the FMA latency (mirrors NEON).
                    for ti in 0..tile_len {
                        let k_off = (kv_start + ti) * kv_dim + kv_h_offset;
                        let mut s0 = _mm256_setzero_ps();
                        let mut s1 = _mm256_setzero_ps();
                        let mut i = 0;
                        while i + 2 <= n_vecs {
                            let k0 = _mm256_loadu_ps(k_ptr.add(k_off + i * 8));
                            let k1 = _mm256_loadu_ps(k_ptr.add(k_off + i * 8 + 8));
                            s0 = _mm256_fmadd_ps(q_vecs[i], k0, s0);
                            s1 = _mm256_fmadd_ps(q_vecs[i + 1], k1, s1);
                            i += 2;
                        }
                        if i < n_vecs {
                            let k0 = _mm256_loadu_ps(k_ptr.add(k_off + i * 8));
                            s0 = _mm256_fmadd_ps(q_vecs[i], k0, s0);
                        }
                        tile_scores[ti] = hsum256(_mm256_add_ps(s0, s1)) * scale;
                    }

                    // Online softmax: tile max (scalar, identical to the
                    // scalar/NEON kernels so the exp reduction order matches).
                    let mut tile_max = f32::NEG_INFINITY;
                    for ti in 0..tile_len {
                        if tile_scores[ti] > tile_max {
                            tile_max = tile_scores[ti];
                        }
                    }
                    let new_max = running_max.max(tile_max);

                    let rescale = if running_max > f32::NEG_INFINITY {
                        ggml_expf(running_max - new_max)
                    } else {
                        0.0
                    };

                    let mut tile_sum = 0.0f64;
                    for ti in 0..tile_len {
                        tile_scores[ti] = ggml_expf(tile_scores[ti] - new_max);
                        tile_sum += tile_scores[ti] as f64;
                    }

                    // Rescale accumulator by the online-softmax factor.
                    let rescale_v = _mm256_set1_ps(rescale);
                    for i in 0..n_vecs {
                        acc_vecs[i] = _mm256_mul_ps(acc_vecs[i], rescale_v);
                    }

                    // acc += score * V.
                    for ti in 0..tile_len {
                        let s = _mm256_set1_ps(tile_scores[ti]);
                        let v_base = (kv_start + ti) * kv_dim + kv_h_offset;
                        for i in 0..n_vecs {
                            let v = _mm256_loadu_ps(v_ptr.add(v_base + i * 8));
                            acc_vecs[i] = _mm256_fmadd_ps(s, v, acc_vecs[i]);
                        }
                    }

                    running_sum = running_sum * rescale as f64 + tile_sum;
                    running_max = new_max;
                }

                let inv_sum = (1.0 / running_sum) as f32;
                let inv_sum_v = _mm256_set1_ps(inv_sum);
                let out_off = (g * n_queries + j) * head_dim;
                for i in 0..n_vecs {
                    let r = _mm256_mul_ps(acc_vecs[i], inv_sum_v);
                    _mm256_storeu_ps(out_ptr.add(out_off + i * 8), r);
                }
            }
        }
    }
}

/// AVX-512 flash attention — the 512-bit twin of `flash_attention_gqa_avx2`,
/// same structure and same scalar online-softmax, 16-wide zmm lanes. Used when
/// `head_dim % 16 == 0` and the host has AVX-512F; the dispatcher falls back to
/// the AVX2 kernel for `head_dim` that is a multiple of 8 but not 16.
///
/// # Safety
/// Caller must ensure AVX-512F is available (the dispatcher CPUID-checks) and
/// that the buffers satisfy the same sizing contract as the scalar kernel.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn flash_attention_gqa_avx512(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        // Buffer-sizing tripwires, mirroring `flash_attention_gqa_neon`. The
        // dispatcher checks `head_dim`, but the caller still owns the q/k/v/out
        // sizing contract; without these a violation is silent UB.
        debug_assert!(
            q_mat.len() >= ((n_heads_start + group_size) * head_dim - 1) * q_stride + n_queries,
            "q_mat too small for the given head range and q_stride"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || k_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "k_cache too small"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || v_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "v_cache too small"
        );
        debug_assert!(
            out.len() >= group_size * n_queries * head_dim,
            "out buffer too small for contiguous [group_size, n_queries, head_dim] output"
        );

        // 256 / 16 = 16 vectors max (head_dim <= 256, multiple of 16).
        const MAX_VECS: usize = 16;
        let n_vecs = head_dim / 16;

        let q_ptr = q_mat.as_ptr();
        let k_ptr = k_cache.as_ptr();
        let v_ptr = v_cache.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let mut q_vecs = [_mm512_setzero_ps(); MAX_VECS];
        let mut acc_vecs = [_mm512_setzero_ps(); MAX_VECS];
        let mut q_gather = [0.0f32; 256];
        let mut tile_scores = [0.0f32; FLASH_TILE_KV];

        for g in 0..group_size {
            let h = n_heads_start + g;
            let h_off = h * head_dim;

            for j in 0..n_queries {
                let max_kv = start_pos + j + 1;

                for d in 0..head_dim {
                    q_gather[d] = *q_ptr.add((h_off + d) * q_stride + j);
                }
                for i in 0..n_vecs {
                    q_vecs[i] = _mm512_loadu_ps(q_gather.as_ptr().add(i * 16));
                    acc_vecs[i] = _mm512_setzero_ps();
                }

                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f64;

                for kv_start in (0..max_kv).step_by(FLASH_TILE_KV) {
                    let kv_end = (kv_start + FLASH_TILE_KV).min(max_kv);
                    let tile_len = kv_end - kv_start;

                    for ti in 0..tile_len {
                        let k_off = (kv_start + ti) * kv_dim + kv_h_offset;
                        let mut s0 = _mm512_setzero_ps();
                        let mut s1 = _mm512_setzero_ps();
                        let mut i = 0;
                        while i + 2 <= n_vecs {
                            let k0 = _mm512_loadu_ps(k_ptr.add(k_off + i * 16));
                            let k1 = _mm512_loadu_ps(k_ptr.add(k_off + i * 16 + 16));
                            s0 = _mm512_fmadd_ps(q_vecs[i], k0, s0);
                            s1 = _mm512_fmadd_ps(q_vecs[i + 1], k1, s1);
                            i += 2;
                        }
                        if i < n_vecs {
                            let k0 = _mm512_loadu_ps(k_ptr.add(k_off + i * 16));
                            s0 = _mm512_fmadd_ps(q_vecs[i], k0, s0);
                        }
                        tile_scores[ti] = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1)) * scale;
                    }

                    let mut tile_max = f32::NEG_INFINITY;
                    for ti in 0..tile_len {
                        if tile_scores[ti] > tile_max {
                            tile_max = tile_scores[ti];
                        }
                    }
                    let new_max = running_max.max(tile_max);

                    let rescale = if running_max > f32::NEG_INFINITY {
                        ggml_expf(running_max - new_max)
                    } else {
                        0.0
                    };

                    let mut tile_sum = 0.0f64;
                    for ti in 0..tile_len {
                        tile_scores[ti] = ggml_expf(tile_scores[ti] - new_max);
                        tile_sum += tile_scores[ti] as f64;
                    }

                    let rescale_v = _mm512_set1_ps(rescale);
                    for i in 0..n_vecs {
                        acc_vecs[i] = _mm512_mul_ps(acc_vecs[i], rescale_v);
                    }

                    for ti in 0..tile_len {
                        let s = _mm512_set1_ps(tile_scores[ti]);
                        let v_base = (kv_start + ti) * kv_dim + kv_h_offset;
                        for i in 0..n_vecs {
                            let v = _mm512_loadu_ps(v_ptr.add(v_base + i * 16));
                            acc_vecs[i] = _mm512_fmadd_ps(s, v, acc_vecs[i]);
                        }
                    }

                    running_sum = running_sum * rescale as f64 + tile_sum;
                    running_max = new_max;
                }

                let inv_sum = (1.0 / running_sum) as f32;
                let inv_sum_v = _mm512_set1_ps(inv_sum);
                let out_off = (g * n_queries + j) * head_dim;
                for i in 0..n_vecs {
                    let r = _mm512_mul_ps(acc_vecs[i], inv_sum_v);
                    _mm512_storeu_ps(out_ptr.add(out_off + i * 16), r);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn flash_attention_gqa_neon(
    q_mat: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_heads_start: usize,
    group_size: usize,
    n_queries: usize,
    q_stride: usize,
    kv_dim: usize,
    kv_h_offset: usize,
    head_dim: usize,
    scale: f32,
    start_pos: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        debug_assert!(
            q_mat.len() >= ((n_heads_start + group_size) * head_dim - 1) * q_stride + n_queries,
            "q_mat too small for the given head range and q_stride"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || k_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "k_cache too small"
        );
        debug_assert!(
            (start_pos + n_queries == 0)
                || v_cache.len() >= (start_pos + n_queries - 1) * kv_dim + kv_h_offset + head_dim,
            "v_cache too small"
        );
        debug_assert!(
            out.len() >= group_size * n_queries * head_dim,
            "out buffer too small for contiguous [group_size, n_queries, head_dim] output"
        );

        let q_ptr = q_mat.as_ptr();
        let k_ptr = k_cache.as_ptr();
        let v_ptr = v_cache.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let n_vecs = head_dim / 4;
        debug_assert!(
            head_dim % 4 == 0 && n_vecs <= 32,
            "head_dim must be a multiple of 4 and <= 128"
        );

        const MAX_VECS: usize = 32;
        let mut q_vecs = [vdupq_n_f32(0.0); MAX_VECS];
        let mut acc_vecs = [vdupq_n_f32(0.0); MAX_VECS];
        let mut tile_scores = [0.0f32; FLASH_TILE_KV];

        for g in 0..group_size {
            let h = n_heads_start + g;
            let h_off = h * head_dim;

            for j in 0..n_queries {
                let max_kv = start_pos + j + 1;

                // Gather Q[h, j] from stride-n layout into NEON registers
                for i in 0..n_vecs {
                    let d = i * 4;
                    let q = [
                        *q_ptr.add((h_off + d) * q_stride + j),
                        *q_ptr.add((h_off + d + 1) * q_stride + j),
                        *q_ptr.add((h_off + d + 2) * q_stride + j),
                        *q_ptr.add((h_off + d + 3) * q_stride + j),
                    ];
                    q_vecs[i] = vld1q_f32(q.as_ptr());
                }

                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f64;
                for i in 0..n_vecs {
                    acc_vecs[i] = vdupq_n_f32(0.0);
                }

                for kv_start in (0..max_kv).step_by(FLASH_TILE_KV) {
                    let kv_end = (kv_start + FLASH_TILE_KV).min(max_kv);
                    let tile_len = kv_end - kv_start;

                    // QK dot products for the tile
                    for ti in 0..tile_len {
                        let k_off = (kv_start + ti) * kv_dim + kv_h_offset;
                        let mut sum0 = vdupq_n_f32(0.0);
                        let mut sum1 = vdupq_n_f32(0.0);
                        let mut i = 0;
                        while i + 2 <= n_vecs {
                            let k0 = vld1q_f32(k_ptr.add(k_off + i * 4));
                            let k1 = vld1q_f32(k_ptr.add(k_off + i * 4 + 4));
                            sum0 = vfmaq_f32(sum0, q_vecs[i], k0);
                            sum1 = vfmaq_f32(sum1, q_vecs[i + 1], k1);
                            i += 2;
                        }
                        if i < n_vecs {
                            let k0 = vld1q_f32(k_ptr.add(k_off + i * 4));
                            sum0 = vfmaq_f32(sum0, q_vecs[i], k0);
                        }
                        tile_scores[ti] = vaddvq_f32(vaddq_f32(sum0, sum1)) * scale;
                    }

                    // Online softmax: tile max
                    let mut tile_max = f32::NEG_INFINITY;
                    for ti in 0..tile_len {
                        if tile_scores[ti] > tile_max {
                            tile_max = tile_scores[ti];
                        }
                    }
                    let new_max = running_max.max(tile_max);

                    let rescale = if running_max > f32::NEG_INFINITY {
                        ggml_expf(running_max - new_max)
                    } else {
                        0.0
                    };

                    // Exp scores and sum
                    let mut tile_sum = 0.0f64;
                    for ti in 0..tile_len {
                        tile_scores[ti] = ggml_expf(tile_scores[ti] - new_max);
                        tile_sum += tile_scores[ti] as f64;
                    }

                    // Rescale accumulator
                    let rescale_v = vdupq_n_f32(rescale);
                    for i in 0..n_vecs {
                        acc_vecs[i] = vmulq_f32(acc_vecs[i], rescale_v);
                    }

                    // Accumulate weighted V: acc += score * V
                    for ti in 0..tile_len {
                        let s = vdupq_n_f32(tile_scores[ti]);
                        let v_base = (kv_start + ti) * kv_dim + kv_h_offset;
                        for i in 0..n_vecs {
                            let v = vld1q_f32(v_ptr.add(v_base + i * 4));
                            acc_vecs[i] = vfmaq_f32(acc_vecs[i], s, v);
                        }
                    }

                    running_sum = running_sum * rescale as f64 + tile_sum;
                    running_max = new_max;
                }

                // Normalize and write contiguous output
                let inv_sum = (1.0 / running_sum) as f32;
                let inv_sum_v = vdupq_n_f32(inv_sum);
                let out_off = (g * n_queries + j) * head_dim;
                for i in 0..n_vecs {
                    let result = vmulq_f32(acc_vecs[i], inv_sum_v);
                    vst1q_f32(out_ptr.add(out_off + i * 4), result);
                }
            }
        }
    }
}

// ── TurboQuant NEON attention ───────────────────────────────────────────────

/// NEON-optimized TurboQuant attention scores for one KV head, multiple query heads.
///
/// Replaces the scalar bucket-sum + QJL loops with NEON intrinsics.
/// For head_dim=128: processes 32 polar bytes and 16 JL bytes per timestep.
///
/// # Safety
/// Caller must ensure all buffer lengths match head_dim, seq_len, and group_size.
/// Requires aarch64 NEON.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_scores_turboquant_neon(
    q_rot_all: &[f32], // [n_heads * head_dim] pre-rotated queries
    q_jl_all: &[f32],  // [n_heads * head_dim] pre-JL-projected queries
    polar_data: &[u8], // packed 2-bit data for this KV head
    jl_data: &[u8],    // packed 1-bit data for this KV head
    norms_f32: &[f32], // pre-converted f32 norms
    residual_norms_f32: &[f32],
    q_jl_total_sums: &[f32], // pre-computed sum(q_jl) per head
    group_start: usize,      // first query head index in the group
    group_size: usize,       // number of query heads in the group
    scores_flat: &mut [f32], // [group_size * seq_len] output, row-major by head
    head_dim: usize,
    centroids: &[f32; 4],
    scale: f32,
    qjl_scale: f32,
    seq_len: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let polar_bytes = head_dim / 4;
        let jl_bytes = head_dim / 8;
        let c_arr = *centroids;

        // Comment #13: Pre-unpack centroid f32x4 vectors per timestep,
        // shared across all query heads in the GQA group.
        // Max head_dim=128 → 32 polar bytes → 32 float32x4 centroids.
        // Comment #18: Same for QJL masks — 16 jl bytes × 2 halves = 32 float32x4.
        const MAX_VECS: usize = 32;
        let n_cent_vecs = polar_bytes; // one float32x4 per packed byte
        let n_mask_vecs = jl_bytes * 2; // two float32x4 per jl byte (lo/hi)
        debug_assert!(n_cent_vecs <= MAX_VECS);
        debug_assert!(n_mask_vecs <= MAX_VECS);
        let mut cent_vecs = [vdupq_n_f32(0.0); MAX_VECS];
        let mut mask_vecs = [vdupq_n_f32(0.0); MAX_VECS];

        for t in 0..seq_len {
            let p_base = t * polar_bytes;
            let j_base = t * jl_bytes;
            let norm = norms_f32[t];
            let residual_norm = residual_norms_f32[t];

            // Unpack centroids once per timestep (hoisted from head loop)
            for (i, cv) in cent_vecs.iter_mut().enumerate().take(n_cent_vecs) {
                let b = *polar_data.get_unchecked(p_base + i);
                *cv = select_centroids_4(b, &c_arr);
            }

            // Unpack QJL masks once per timestep (hoisted from head loop, Comment #18)
            for i in 0..jl_bytes {
                let b = *jl_data.get_unchecked(j_base + i) as u32;
                mask_vecs[i * 2] = bits_to_f32_mask_lo(b);
                mask_vecs[i * 2 + 1] = bits_to_f32_mask_hi(b);
            }

            // Process each query head in the GQA group
            for g in 0..group_size {
                let h = group_start + g;
                let q_rot = &q_rot_all[h * head_dim..];
                let q_jl = &q_jl_all[h * head_dim..];

                // PolarQuant dot: FMA pre-unpacked centroids with query
                let mut dot_acc0 = vdupq_n_f32(0.0);
                let mut dot_acc1 = vdupq_n_f32(0.0);
                let mut ci = 0usize;
                let mut q_off = 0usize;
                while ci + 4 <= n_cent_vecs {
                    let qv0 = vld1q_f32(q_rot.as_ptr().add(q_off));
                    let qv1 = vld1q_f32(q_rot.as_ptr().add(q_off + 4));
                    let qv2 = vld1q_f32(q_rot.as_ptr().add(q_off + 8));
                    let qv3 = vld1q_f32(q_rot.as_ptr().add(q_off + 12));
                    dot_acc0 = vfmaq_f32(dot_acc0, qv0, cent_vecs[ci]);
                    dot_acc1 = vfmaq_f32(dot_acc1, qv1, cent_vecs[ci + 1]);
                    dot_acc0 = vfmaq_f32(dot_acc0, qv2, cent_vecs[ci + 2]);
                    dot_acc1 = vfmaq_f32(dot_acc1, qv3, cent_vecs[ci + 3]);
                    ci += 4;
                    q_off += 16;
                }
                while ci < n_cent_vecs {
                    let qv = vld1q_f32(q_rot.as_ptr().add(q_off));
                    dot_acc0 = vfmaq_f32(dot_acc0, qv, cent_vecs[ci]);
                    ci += 1;
                    q_off += 4;
                }
                let polar_dot = vaddvq_f32(vaddq_f32(dot_acc0, dot_acc1)) * norm;

                // QJL: pos_sum only, total_sum pre-computed, masks pre-unpacked
                let total_sum = *q_jl_total_sums.get_unchecked(h);
                let mut pos_acc0 = vdupq_n_f32(0.0);
                let mut pos_acc1 = vdupq_n_f32(0.0);
                let mut mi = 0usize;
                let mut jl_q_off = 0usize;
                while mi + 4 <= n_mask_vecs {
                    let q0 = vld1q_f32(q_jl.as_ptr().add(jl_q_off));
                    let q1 = vld1q_f32(q_jl.as_ptr().add(jl_q_off + 4));
                    let q2 = vld1q_f32(q_jl.as_ptr().add(jl_q_off + 8));
                    let q3 = vld1q_f32(q_jl.as_ptr().add(jl_q_off + 12));
                    pos_acc0 = vfmaq_f32(pos_acc0, q0, mask_vecs[mi]);
                    pos_acc1 = vfmaq_f32(pos_acc1, q1, mask_vecs[mi + 1]);
                    pos_acc0 = vfmaq_f32(pos_acc0, q2, mask_vecs[mi + 2]);
                    pos_acc1 = vfmaq_f32(pos_acc1, q3, mask_vecs[mi + 3]);
                    mi += 4;
                    jl_q_off += 16;
                }
                while mi < n_mask_vecs {
                    let q = vld1q_f32(q_jl.as_ptr().add(jl_q_off));
                    pos_acc0 = vfmaq_f32(pos_acc0, q, mask_vecs[mi]);
                    mi += 1;
                    jl_q_off += 4;
                }
                let pos_sum = vaddvq_f32(vaddq_f32(pos_acc0, pos_acc1));
                let signed_sum = 2.0 * pos_sum - total_sum;
                // residual_norm is stored in unit-normalized key space, so
                // the correction must be rescaled by the original key norm
                // to match polar_dot (which was multiplied by norm above).
                let correction = norm * residual_norm * qjl_scale * signed_sum;

                scores_flat[g * seq_len + t] = (polar_dot + correction) * scale;
            }
        }
    }
}

/// NEON weighted sum of compressed values for a GQA group.
///
/// For each query head in `[group_start, group_start + group_size)`:
///   `out[h*head_dim + d] = Σ_t (scores[g*seq_len + t] * norms_f32[t]) * centroid[indices[t, d]]`
///
/// Writes the **rotated-space** accumulator to `attn_out`; the caller is
/// responsible for applying `rht_inverse` to each head after this function
/// returns. Caller must also ensure `head_dim <= 128` (stack accumulator
/// limit) and that all buffer lengths are consistent.
///
/// # Safety
/// All slices must be large enough: `polar_data.len() >= seq_len * head_dim/4`,
/// `norms_f32.len() >= seq_len`, `scores.len() >= group_size * seq_len`,
/// `attn_out.len() >= (group_start + group_size) * head_dim`.
/// Requires aarch64 NEON.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub unsafe fn attn_values_turboquant_neon(
    polar_data: &[u8],    // packed 2-bit data for this KV head
    norms_f32: &[f32],    // pre-converted f32 norms for this KV head
    scores: &[f32],       // [group_size * seq_len], row-major by head
    attn_out: &mut [f32], // [n_heads * head_dim] — writes group_size heads
    group_start: usize,
    group_size: usize,
    head_dim: usize,
    seq_len: usize,
    centroids: &[f32; 4],
) {
    use std::arch::aarch64::*;
    unsafe {
        let polar_bytes = head_dim / 4;
        debug_assert!(
            polar_bytes <= 32,
            "head_dim > 128 not supported by NEON path"
        );

        // Stack-allocated accumulator: up to 32 float32x4 = 128 floats.
        // One set per group member; reused across the group loop.
        const MAX_VECS: usize = 32;

        for g in 0..group_size {
            let h = group_start + g;
            let head_scores = scores.as_ptr().add(g * seq_len);

            // Initialize accumulator to zero.
            let mut acc = [vdupq_n_f32(0.0); MAX_VECS];

            // Accumulate weighted centroid vectors across all timesteps.
            for t in 0..seq_len {
                let w = *head_scores.add(t) * *norms_f32.get_unchecked(t);
                let w_vec = vdupq_n_f32(w);
                let base = t * polar_bytes;
                for i in 0..polar_bytes {
                    let b = *polar_data.get_unchecked(base + i);
                    let c_vec = select_centroids_4(b, centroids);
                    acc[i] = vfmaq_f32(acc[i], w_vec, c_vec);
                }
            }

            // Store the per-head accumulator to attn_out. The caller applies
            // rht_inverse afterwards — it's cheap (O(head_dim log head_dim))
            // and doesn't benefit from being inline here.
            let out_ptr = attn_out.as_mut_ptr().add(h * head_dim);
            for i in 0..polar_bytes {
                vst1q_f32(out_ptr.add(i * 4), acc[i]);
            }
        }
    }
}

/// Select 4 centroid values from a packed 2-bit byte.
/// Returns float32x4 with centroids[idx0], centroids[idx1], centroids[idx2], centroids[idx3].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn select_centroids_4(byte: u8, c: &[f32; 4]) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let vals: [f32; 4] = [
            *c.get_unchecked((byte & 0x03) as usize),
            *c.get_unchecked(((byte >> 2) & 0x03) as usize),
            *c.get_unchecked(((byte >> 4) & 0x03) as usize),
            *c.get_unchecked(((byte >> 6) & 0x03) as usize),
        ];
        vld1q_f32(vals.as_ptr())
    }
}

/// Expand lower 4 bits of a byte to f32 mask: bit i → 0.0 or 1.0.
/// Returns float32x4 for bits 0,1,2,3.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bits_to_f32_mask_lo(byte: u32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let vals: [f32; 4] = [
            (byte & 1) as f32,
            ((byte >> 1) & 1) as f32,
            ((byte >> 2) & 1) as f32,
            ((byte >> 3) & 1) as f32,
        ];
        vld1q_f32(vals.as_ptr())
    }
}

/// Expand upper 4 bits of a byte to f32 mask: bit i → 0.0 or 1.0.
/// Returns float32x4 for bits 4,5,6,7.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bits_to_f32_mask_hi(byte: u32) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let vals: [f32; 4] = [
            ((byte >> 4) & 1) as f32,
            ((byte >> 5) & 1) as f32,
            ((byte >> 6) & 1) as f32,
            ((byte >> 7) & 1) as f32,
        ];
        vld1q_f32(vals.as_ptr())
    }
}

// ── Positional encoding ─────────────────────────────────────────────────────

/// RoPE pair layout.
///
/// - `Neox` (GPT-NeoX / split-halves): rotates `(x[i], x[i + head_dim/2])`.
///   Correct for un-permuted Qwen2/Qwen3/LFM2 GGUF weights — llama.cpp applies
///   `GGML_ROPE_TYPE_NEOX` to these.
/// - `Norm` (original LLaMA / interleaved): rotates adjacent pairs
///   `(x[2i], x[2i+1])`. Correct for un-permuted LLaMA/Mistral/Granite GGUF
///   weights — llama.cpp applies `GGML_ROPE_TYPE_NORM` to these.
///
/// The per-pair angle schedule (`theta_base * theta_scale^i`) is identical
/// across both layouts; only the element pairing differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Neox,
    Norm,
}

/// Apply Rotary Position Embedding (RoPE) to Q and K vectors.
///
/// `q` and `k` are [n_heads * head_dim] and [n_kv_heads * head_dim] respectively.
/// Applies rotation based on position `pos` and frequency base `freq_base`.
pub fn rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(k.len(), n_kv_heads * head_dim);

    // Apply to Q heads
    for h in 0..n_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut q[offset..offset + head_dim], pos, head_dim, freq_base);
    }

    // Apply to K heads
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        apply_rope_to_head(&mut k[offset..offset + head_dim], pos, head_dim, freq_base);
    }
}

/// Apply RoPE rotation to a single head vector.
/// Uses iterative theta multiplication to match ggml's `ggml_rope_cache_init`.
///
/// Exposed `pub` so integration tests can use it as the oracle when
/// verifying [`apply_rope_delta_to_head`] — the two functions must
/// produce equivalent results when composed per the additive-rotation
/// identity `R(p + δ) = R(δ) · R(p)`.
pub fn apply_rope_to_head(head: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..half_dim {
        let (sin_t, cos_t) = theta.sin_cos();

        let x0 = head[i];
        let x1 = head[i + half_dim];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[i + half_dim] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

/// Compose an additional RoPE rotation onto an already-rotated head
/// vector (Q or K). Given a head that was previously rotated for
/// position `p_old` — so `head = R(p_old) · raw` — calling this with
/// `delta_pos = p_new - p_old` leaves the head rotated for position
/// `p_new`, since 2D rotations compose additively in each dim-pair
/// plane: `R(p_new) = R(p_new - p_old) · R(p_old)`.
///
/// `delta_pos` is signed — negative values unwind the rotation
/// (`sin_cos` handles negatives directly, no sign-flip bookkeeping
/// needed). Used by the `n_keep` context shift (`InferenceState::shift_kv_with_rope`)
/// to re-rotate K cells whose absolute position has moved after a
/// middle-range drain. Same split-halves pair layout + iterative
/// theta schedule as [`apply_rope_to_head`] so the two compose
/// cleanly.
pub fn apply_rope_delta_to_head(head: &mut [f32], delta_pos: i32, head_dim: usize, freq_base: f32) {
    let half_dim = head_dim / 2;
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = delta_pos as f32;
    for i in 0..half_dim {
        let (sin_t, cos_t) = theta.sin_cos();

        let x0 = head[i];
        let x1 = head[i + half_dim];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[i + half_dim] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

/// NORM-layout (interleaved-pair) RoPE for Q and K. Sibling of [`rope`], but
/// rotates adjacent pairs `(x[2i], x[2i+1])` instead of split halves. Correct for
/// un-permuted LLaMA/Mistral/Granite GGUF weights (llama.cpp `GGML_ROPE_TYPE_NORM`).
/// `freq_factors` is the optional Llama-3 RoPE scaling (`rope_freqs.weight`);
/// `None` ⇒ plain RoPE (Mistral/Granite, and Qwen never reach this path).
#[allow(clippy::too_many_arguments)]
pub fn rope_norm(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(k.len(), n_kv_heads * head_dim);

    for h in 0..n_heads {
        let offset = h * head_dim;
        apply_rope_norm_to_head(
            &mut q[offset..offset + head_dim],
            pos,
            head_dim,
            freq_base,
            freq_factors,
        );
    }
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        apply_rope_norm_to_head(
            &mut k[offset..offset + head_dim],
            pos,
            head_dim,
            freq_base,
            freq_factors,
        );
    }
}

/// NORM-layout counterpart of [`apply_rope_to_head`]. Rotates adjacent pairs
/// `(head[2i], head[2i+1])` with the same iterative theta schedule. `freq_factors`
/// (Llama-3 RoPE scaling, `rope_freqs.weight`) optionally divides each pair's
/// angle; `None` ⇒ plain RoPE.
pub fn apply_rope_norm_to_head(
    head: &mut [f32],
    pos: usize,
    head_dim: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) {
    rope_norm_pairs(head, pos as f32, head_dim, freq_base, freq_factors);
}

/// NORM-layout counterpart of [`apply_rope_delta_to_head`]: composes an
/// additional rotation of `delta_pos` onto an already-NORM-rotated head, used by
/// the `n_keep` context shift for NORM-rope models. Same additive-rotation
/// identity as the NEOX delta, on interleaved pairs. Must receive the same
/// `freq_factors` as the forward pass for the composition identity to hold.
pub fn apply_rope_norm_delta_to_head(
    head: &mut [f32],
    delta_pos: i32,
    head_dim: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) {
    rope_norm_pairs(head, delta_pos as f32, head_dim, freq_base, freq_factors);
}

/// Shared kernel for NORM (interleaved-pair) RoPE: rotates each adjacent pair
/// `(head[2i], head[2i+1])` by `(theta_start * theta_scale^i) / freq_factors[i]`.
/// The absolute (`pos`) and delta (`delta_pos`) entry points differ only in
/// `theta_start`. `freq_factors` is the optional Llama-3 per-pair scaling
/// (`rope_freqs.weight`, length `head_dim/2`); `None` ⇒ all factors 1.0. Matches
/// ggml's `rope_yarn(theta_base / ff, ...)`.
fn rope_norm_pairs(
    head: &mut [f32],
    theta_start: f32,
    head_dim: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta_base = theta_start;
    for (i, pair) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let ff = freq_factors.map_or(1.0, |f| f[i]);
        let theta = theta_base / ff;
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = pair[0];
        let x1 = pair[1];
        pair[0] = x0 * cos_t - x1 * sin_t;
        pair[1] = x0 * sin_t + x1 * cos_t;
        theta_base *= theta_scale;
    }
}

// ── Convolution ─────────────────────────────────────────────────────────────

/// Depthwise 1D convolution.
///
/// `input`:  `[seq_len, channels]`
/// `weight`: `[channels, kernel_size]` (one kernel per channel)
/// `bias`:   optional `[channels]`
/// `output`: `[seq_len, channels]` (same padding via zero-pad)
pub fn conv1d_depthwise(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    channels: usize,
    kernel_size: usize,
    seq_len: usize,
) {
    debug_assert_eq!(input.len(), seq_len * channels);
    debug_assert_eq!(weight.len(), channels * kernel_size);
    debug_assert_eq!(output.len(), seq_len * channels);

    let pad = kernel_size / 2; // causal or symmetric padding

    for t in 0..seq_len {
        for c in 0..channels {
            let mut sum = if let Some(b) = bias { b[c] } else { 0.0 };

            for ki in 0..kernel_size {
                let input_t = t as isize + ki as isize - pad as isize;
                if input_t >= 0 && (input_t as usize) < seq_len {
                    sum += input[input_t as usize * channels + c] * weight[c * kernel_size + ki];
                }
            }
            output[t * channels + c] = sum;
        }
    }
}

// ── Element-wise operations ─────────────────────────────────────────────────

/// Element-wise addition: a += b.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += *b;
    }
}

/// Element-wise multiplication: a *= b.
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a *= *b;
    }
}

/// Scalar multiplication: a *= s. Used by Granite's embedding / residual /
/// logit scalar multipliers (`ggml_scale`).
pub fn scale_inplace(a: &mut [f32], s: f32) {
    for x in a.iter_mut() {
        *x *= s;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The SIMD flash-attention kernels must agree with the scalar reference
    /// across every supported `head_dim`, group size, and prompt length.
    ///
    /// The scalar kernel is the oracle: `flash_attention_gqa_cpu` dispatches to
    /// the SIMD kernels on x86, so a bug there would silently corrupt every
    /// dense-transformer prefill on this host. The SIMD kernels are NOT
    /// bit-identical to scalar (the QK dot and V accumulate sum in a different
    /// lane order — the same divergence NEON already carries), so the bar is a
    /// tight relative one, not equality: max |simd - scalar| / (|scalar| + eps)
    /// well under the parity suite's cosine>0.99 flash bound.
    ///
    /// The `*_matches_scalar_across_head_dims` tests call each kernel *directly*
    /// by name, so their `head_dim` list varies `n_vecs` for tail coverage (the
    /// `if i < n_vecs` QK-dot tail fires at odd counts: head_dim/8 = 9 at 72,
    /// head_dim/16 = 5 at 80), NOT the dispatch path. Routing itself — the
    /// `% 16 → avx512, else % 8 → avx2` selection and the CPUID gates — is
    /// covered separately by `dispatcher_routes_to_a_correct_kernel`, and both
    /// `start_pos` regimes (fresh vs continuation prefill) are covered throughout.
    #[cfg(target_arch = "x86_64")]
    mod flash_attention_simd {
        use super::*;

        /// Deterministic pseudo-random f32 in roughly [-1, 1].
        fn lcg(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        /// Relative L2 deviation `||b - a|| / ||a||`. This is the vector-level
        /// metric the flash parity bar (cosine) is built on; a per-element
        /// relative error is the wrong tool here because the softmax-weighted
        /// output has legitimately near-zero elements that make any small
        /// absolute wobble look enormous in relative terms.
        fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
            let num: f64 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| ((x - y) as f64).powi(2))
                .sum();
            let den: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
            (num.sqrt() / (den.sqrt() + 1e-12)) as f32
        }

        /// Build one GQA config and assert the chosen `kernel` agrees with the
        /// scalar reference. `start_pos` is the absolute position of the first
        /// query: 0 is a fresh prefill, >0 a continuation prefill where queries
        /// attend to `start_pos` prior tokens already in the KV cache (the
        /// multi-turn / prefix-cache-reuse regime). `kernel` is `"avx2"`,
        /// `"avx512"`, or `"dispatch"` — the last routes through
        /// `flash_attention_gqa_cpu`, exercising the CPUID + head_dim routing
        /// rather than a kernel by name.
        fn check(head_dim: usize, start_pos: usize, kernel: &str) {
            // One KV head with a group of query heads, a prompt long enough to
            // span multiple FLASH_TILE_KV tiles (32), and a non-tile-aligned
            // length so the tile tail is exercised.
            let group_size = 3;
            let n_kv_heads = 2;
            let n_heads = n_kv_heads * group_size;
            let n = 70usize; // > 2*FLASH_TILE_KV, not a multiple of 32
            let kv_len = start_pos + n; // KV cache holds prior context + queries
            let kv_dim = n_kv_heads * head_dim;
            let q_dim = n_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut st = 0x1234_5678_9abc_def0u64 ^ (head_dim as u64) ^ ((start_pos as u64) << 40);
            // q_mat is [q_dim, n] column-major (stride n).
            let q: Vec<f32> = (0..q_dim * n).map(|_| lcg(&mut st)).collect();
            let k: Vec<f32> = (0..kv_len * kv_dim).map(|_| lcg(&mut st)).collect();
            let v: Vec<f32> = (0..kv_len * kv_dim).map(|_| lcg(&mut st)).collect();

            // Compare per KV head, matching the dispatcher's per-head calls.
            for kv_h in 0..n_kv_heads {
                let n_heads_start = kv_h * group_size;
                let kv_h_offset = kv_h * head_dim;
                let mut out_ref = vec![0.0f32; group_size * n * head_dim];
                // The scalar kernel is the oracle for the very kernels the
                // dispatcher would pick — call it directly, bypassing dispatch.
                flash_attention_gqa_scalar(
                    &q,
                    &k,
                    &v,
                    &mut out_ref,
                    n_heads_start,
                    group_size,
                    n,
                    n,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    start_pos,
                );

                let mut out_simd = vec![0.0f32; group_size * n * head_dim];
                match kernel {
                    "avx2" => unsafe {
                        flash_attention_gqa_avx2(
                            &q,
                            &k,
                            &v,
                            &mut out_simd,
                            n_heads_start,
                            group_size,
                            n,
                            n,
                            kv_dim,
                            kv_h_offset,
                            head_dim,
                            scale,
                            start_pos,
                        );
                    },
                    #[cfg(feature = "avx512")]
                    "avx512" => unsafe {
                        flash_attention_gqa_avx512(
                            &q,
                            &k,
                            &v,
                            &mut out_simd,
                            n_heads_start,
                            group_size,
                            n,
                            n,
                            kv_dim,
                            kv_h_offset,
                            head_dim,
                            scale,
                            start_pos,
                        );
                    },
                    // Routes through the real dispatcher (CPUID + head_dim
                    // selection), so the "% 16 → avx512, else % 8 → avx2, else
                    // scalar" routing and the feature-detection gates are covered.
                    "dispatch" => flash_attention_gqa_cpu(
                        &q,
                        &k,
                        &v,
                        &mut out_simd,
                        n_heads_start,
                        group_size,
                        n,
                        n,
                        kv_dim,
                        kv_h_offset,
                        head_dim,
                        scale,
                        start_pos,
                    ),
                    other => panic!("unknown kernel {other}"),
                }

                let rel = rel_l2(&out_ref, &out_simd);
                // FMA-vs-separate-mul-add rounding on a head_dim-length dot,
                // propagated through the online softmax, lands ~1e-6. 1e-4 is
                // two orders of margin yet still ~O(1) below what a transposed
                // index or wrong offset (cosine collapse) would produce.
                assert!(
                    rel < 1e-4,
                    "{kernel} head_dim={head_dim} start_pos={start_pos} kv_h={kv_h}: \
                     relative L2 deviation {rel:e} exceeds 1e-4 vs scalar reference"
                );
            }
        }

        /// start_pos values: 0 (fresh prefill) and 37 (continuation prefill —
        /// non-tile-aligned so the causal `max_kv = start_pos + j + 1` bound and
        /// the start_pos-dependent K/V offsets are exercised, not just the
        /// start_pos=0 fast path).
        const START_POS: [usize; 2] = [0, 37];

        #[test]
        fn avx2_matches_scalar_across_head_dims() {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
                if std::env::var("CERA_REQUIRE_SIMD")
                    .unwrap_or_default()
                    .split(',')
                    .any(|f| f.trim() == "avx2")
                {
                    panic!("CERA_REQUIRE_SIMD=avx2 but avx2/fma not detected");
                }
                eprintln!("[flash-avx2] SKIP: avx2/fma not detected");
                return;
            }
            // 72 is a multiple of 8 but not 16 and gives an odd n_vecs=9,
            // exercising the QK-dot loop tail.
            for hd in [64usize, 72, 128, 256] {
                for sp in START_POS {
                    check(hd, sp, "avx2");
                }
            }
        }

        #[cfg(feature = "avx512")]
        #[test]
        fn avx512_matches_scalar_across_head_dims() {
            if !is_x86_feature_detected!("avx512f") {
                if std::env::var("CERA_REQUIRE_SIMD")
                    .unwrap_or_default()
                    .split(',')
                    .any(|f| f.trim() == "avx512")
                {
                    panic!("CERA_REQUIRE_SIMD=avx512 but avx512f not detected");
                }
                eprintln!("[flash-avx512] SKIP: avx512f not detected");
                return;
            }
            // 80 gives an odd n_vecs=5 (80/16), exercising the AVX-512 loop tail.
            for hd in [64usize, 80, 128, 256] {
                for sp in START_POS {
                    check(hd, sp, "avx512");
                }
            }
        }

        /// The dispatcher (`flash_attention_gqa_cpu`) itself: its `% 16 → avx512,
        /// else % 8 → avx2, else scalar` routing plus the CPUID gates. Whatever
        /// it selects on this host must match the scalar reference. head_dim 64
        /// takes the widest path the host offers; 72 forces the
        /// multiple-of-8-not-16 → AVX2 fallthrough even on an AVX-512 host — the
        /// branch no real model's head_dim hits, so nothing else covers it.
        #[test]
        fn dispatcher_routes_to_a_correct_kernel() {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
                // Below AVX2 the dispatcher falls through to scalar, which is the
                // oracle — the comparison would be scalar-vs-scalar, vacuous.
                if std::env::var("CERA_REQUIRE_SIMD")
                    .unwrap_or_default()
                    .split(',')
                    .any(|f| f.trim() == "avx2")
                {
                    panic!("CERA_REQUIRE_SIMD=avx2 but avx2/fma not detected");
                }
                eprintln!("[flash-dispatch] SKIP: avx2/fma not detected");
                return;
            }
            for hd in [64usize, 72] {
                for sp in START_POS {
                    check(hd, sp, "dispatch");
                }
            }
        }
    }

    /// The `gemm_preq_dispatch` length guards must actually fire.
    ///
    /// Those asserts are the entire justification for `gemm_preq_dispatch`
    /// being a safe `pub` fn that hands unchecked lengths to `unsafe` kernels —
    /// and nothing pinned them: replacing all three with tautologies passed the
    /// whole suite. `out` is the subtle one. The kernels derive their strip/row
    /// index from `out.len()`, not from `m`, so an *over-long* `out` walks past
    /// row `m` and reads weights out of bounds — which is why the contract is
    /// `==` and not `>=`, and why this case gets its own test.
    mod gemm_preq_guards {
        use super::*;
        use crate::tensor::DType;

        /// A well-formed Q8_0 call: 2 rows, 1 column, k = 64.
        fn args() -> (Vec<u8>, Vec<f32>, Vec<i8>, Vec<f32>) {
            let (m, n, k) = (2usize, 1usize, 64usize);
            let nb = k / 32;
            (
                vec![0u8; m * nb * DType::Q8_0.block_bytes()],
                vec![0.0f32; n * nb],
                vec![0i8; n * k],
                vec![0.0f32; m * n],
            )
        }

        #[test]
        #[should_panic(expected = "out must be exactly")]
        fn over_long_out_is_rejected() {
            let (data, bs, bq, mut out) = args();
            out.push(0.0);
            gemm_preq_dispatch(DType::Q8_0, &data, &bs, &bq, &mut out, 2, 1, 64);
        }

        #[test]
        #[should_panic(expected = "out must be exactly")]
        fn short_out_is_rejected() {
            let (data, bs, bq, mut out) = args();
            out.pop();
            gemm_preq_dispatch(DType::Q8_0, &data, &bs, &bq, &mut out, 2, 1, 64);
        }

        #[test]
        #[should_panic(expected = "weights are")]
        fn short_weights_are_rejected() {
            let (mut data, bs, bq, mut out) = args();
            data.truncate(DType::Q8_0.block_bytes());
            gemm_preq_dispatch(DType::Q8_0, &data, &bs, &bq, &mut out, 2, 1, 64);
        }

        #[test]
        #[should_panic(expected = "not a multiple")]
        fn unaligned_k_is_rejected() {
            let (data, bs, bq, mut out) = args();
            gemm_preq_dispatch(DType::Q8_0, &data, &bs, &bq, &mut out, 2, 1, 60);
        }

        /// The well-formed call must NOT panic, or the four above would pass
        /// against a guard that rejects everything.
        #[test]
        fn well_formed_call_is_accepted() {
            let (data, bs, bq, mut out) = args();
            gemm_preq_dispatch(DType::Q8_0, &data, &bs, &bq, &mut out, 2, 1, 64);
        }
    }

    /// The repacked-Q4_0 dispatch must agree with the standard-layout dispatch
    /// for the same weight — one level above the kernel equivalence test in
    /// `simd.rs`. This drives the *plumbing*: `repack_q4_0_8x8` →
    /// `gemm_preq_repacked_q4_0_dispatch` (its tier selection and length asserts)
    /// vs `gemm_preq_dispatch(Q4_0, …)`, so a mis-routed tier or a wrong buffer
    /// hand-off is caught here even though both underlying kernels are correct.
    ///
    /// `n = 13` exercises the column tile plus a remainder on both tiers; `m =
    /// 16` is two super-rows. Skips on a host without the x86 int8 kernels
    /// (where `q4_0_repack_supported` would decline the repack in production).
    #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
    #[test]
    fn repacked_q4_0_dispatch_matches_standard_dispatch() {
        use crate::tensor::DType;
        if !int8_gemm_available() {
            return;
        }
        let (m, n, k) = (16usize, 13usize, 128usize);
        let nb = k / 32;

        // Synthetic Q4_0 weights: nonzero f16 scale + random nibbles per block.
        let mut st = 0xd15e_a5edu64;
        let mut lcg = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            (st >> 33) as u32
        };
        let mut data = Vec::with_capacity(m * nb * DType::Q4_0.block_bytes());
        for _ in 0..m * nb {
            let d = half::f16::from_f32(0.01 + 0.04 * (lcg() as f32 / u32::MAX as f32));
            data.extend_from_slice(&d.to_bits().to_le_bytes());
            for _ in 0..16 {
                data.push(lcg() as u8);
            }
        }

        // Random activations, quantized to Q8_0 in the column-major GEMM layout.
        let mut b_scales = vec![0.0f32; n * nb];
        let mut b_quants = vec![0i8; n * k];
        for j in 0..n {
            let col: Vec<f32> = (0..k)
                .map(|_| lcg() as f32 / u32::MAX as f32 * 2.0 - 1.0)
                .collect();
            quantize_f32_to_q8_0_into(
                &col,
                &mut b_scales[j * nb..(j + 1) * nb],
                &mut b_quants[j * k..(j + 1) * k],
            );
        }

        let mut want = vec![0.0f32; m * n];
        assert!(gemm_preq_dispatch(
            DType::Q4_0,
            &data,
            &b_scales,
            &b_quants,
            &mut want,
            m,
            n,
            k
        ));

        let (packed, scales) = repack_q4_0_8x8(&data, m, k);
        let mut got = vec![0.0f32; m * n];
        assert!(gemm_preq_repacked_q4_0_dispatch(
            &packed, &scales, &b_scales, &b_quants, &mut got, m, n, k
        ));

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "repacked vs standard dispatch [{},{}]: {g} vs {w}",
                i / n,
                i % n,
            );
        }
    }

    /// The repacked-Q4_K dispatch must agree with the standard-layout dispatch —
    /// the Q4_K twin of `repacked_q4_0_dispatch_matches_standard_dispatch`, and
    /// the only test that drives `repack_q4_k_8x8` →
    /// `gemm_preq_repacked_q4_k_dispatch` end to end (tier selection + length
    /// asserts + baked-scale hand-off). `k = 256` is one super-block; `n = 13`
    /// hits the column tile plus a remainder; `m = 16` is two super-rows.
    #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
    #[test]
    fn repacked_q4_k_dispatch_matches_standard_dispatch() {
        use crate::tensor::DType;
        if !int8_gemm_available() {
            return;
        }
        let (m, n, k) = (16usize, 13usize, 256usize);
        let sb = k / 256;
        let nb = k / 32;

        // Synthetic Q4_K_M weights: controlled f16 d/dmin (random bits can be
        // NaN/inf), random packed scales and nibbles.
        let mut st = 0x9e37_79b9u64;
        let mut lcg = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            (st >> 33) as u32
        };
        let bsz = size_of::<crate::quant::BlockQ4KM>();
        let mut data = vec![0u8; m * sb * bsz];
        for chunk in data.chunks_mut(bsz) {
            let d = half::f16::from_f32(0.01 + 0.04 * (lcg() as f32 / u32::MAX as f32));
            let dmin = half::f16::from_f32(0.02 + 0.03 * (lcg() as f32 / u32::MAX as f32));
            chunk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
            chunk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            for b in chunk[4..].iter_mut() {
                *b = lcg() as u8;
            }
        }

        // Random activations, quantized to Q8_0 in the column-major GEMM layout.
        let mut b_scales = vec![0.0f32; n * nb];
        let mut b_quants = vec![0i8; n * k];
        for j in 0..n {
            let col: Vec<f32> = (0..k)
                .map(|_| lcg() as f32 / u32::MAX as f32 * 2.0 - 1.0)
                .collect();
            quantize_f32_to_q8_0_into(
                &col,
                &mut b_scales[j * nb..(j + 1) * nb],
                &mut b_quants[j * k..(j + 1) * k],
            );
        }

        let mut want = vec![0.0f32; m * n];
        assert!(gemm_preq_dispatch(
            DType::Q4KM,
            &data,
            &b_scales,
            &b_quants,
            &mut want,
            m,
            n,
            k
        ));

        let (packed, dsc, dmn) = repack_q4_k_8x8(&data, m, k);
        let mut got = vec![0.0f32; m * n];
        assert!(gemm_preq_repacked_q4_k_dispatch(
            &packed, &dsc, &dmn, &b_scales, &b_quants, &mut got, m, n, k
        ));

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "repacked Q4_K vs standard dispatch [{},{}]: {g} vs {w}",
                i / n,
                i % n,
            );
        }
    }

    /// `gemv_dispatch` must reach the right kernel for every int8 dtype.
    ///
    /// The kernels themselves are unit-tested in `simd.rs`, but nothing drove
    /// the *dispatcher* on x86: the tier arms added for AVX2 (and the K-quant
    /// `kq_gemv!` expansions) were reached only by the `#[ignore]`d real-model
    /// parity suite. A dtype mis-wire — a Q6K arm calling `gemv_q4k_f32` — would
    /// have produced garbage logits with nothing in `cargo test` objecting.
    ///
    /// Covers the four dtypes with int8 kernels. `Q4_1`, `Q5KM` and `F32` take
    /// scalar arms this change does not touch and are not driven here.
    ///
    /// The reference dequantizes the weight and does the dot in f32, so it is
    /// independent of every int8 path under test. The bound is loose on purpose:
    /// this asserts "the right kernel ran", not "the arithmetic is exact" —
    /// exactness is the job of the tests next to each kernel. A wrong kernel is
    /// off by orders of magnitude, not by 2%.
    #[test]
    fn gemv_dispatch_matches_dequantized_reference() {
        use crate::tensor::DType;

        // k must satisfy every dtype's block alignment at once: 256 for the
        // K-quants, 32 for Q4_0/Q8_0.
        let (m, k) = (7usize, 256usize);
        let mut st = 0x5eed_1234u64;
        let mut next = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as u32
        };

        let x: Vec<f32> = (0..k)
            .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
            .collect();

        for dtype in [DType::Q4_0, DType::Q8_0, DType::Q4KM, DType::Q6K] {
            let nb = k / dtype.block_size();
            let bb = dtype.block_bytes();
            let mut data: Vec<u8> = (0..m * nb * bb).map(|_| (next() % 256) as u8).collect();
            // Random bytes in a scale field decode to inf/NaN, which would make
            // the reference itself meaningless (NaN fails this bound rather than
            // passing it, so the test would be flaky, not vacuous). Everything
            // else — nibbles, 6-bit scales, qh — stays fully random.
            for (bi, blk) in data.chunks_mut(bb).enumerate() {
                let d = half::f16::from_f32(0.01 + 0.004 * (bi % 7) as f32);
                match dtype {
                    // scale first
                    DType::Q4_0 | DType::Q8_0 | DType::Q4KM => {
                        blk[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                    }
                    // scale last
                    DType::Q6K => {
                        let n = blk.len();
                        blk[n - 2..].copy_from_slice(&d.to_bits().to_le_bytes());
                    }
                    _ => unreachable!(),
                }
                if dtype == DType::Q4KM {
                    let dmin = half::f16::from_f32(0.02 + 0.003 * (bi % 5) as f32);
                    blk[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
                }
            }

            // f32 reference, independent of every int8 kernel.
            let mut w = vec![0.0f32; m * k];
            match dtype {
                DType::Q4_0 => crate::quant::dequantize_q4_0_matrix(&data, m, k, &mut w),
                DType::Q8_0 => crate::quant::dequantize_q8_0_matrix(&data, m, k, &mut w),
                DType::Q4KM => crate::quant::dequantize_q4_k_m_matrix(&data, m, k, &mut w),
                DType::Q6K => crate::quant::dequantize_q6_k_matrix(&data, m, k, &mut w),
                _ => unreachable!(),
            }
            let want: Vec<f32> = (0..m)
                .map(|i| (0..k).map(|j| w[i * k + j] * x[j]).sum())
                .collect();

            // Both scratch modes: production decode always lends a buffer
            // (`model/weights.rs`), so `None` alone would leave the arm that
            // actually ships untested. The results must agree — the scratch is
            // an allocation optimization, not a numeric one.
            let mut got = vec![0.0f32; m];
            gemv_dispatch(dtype, &data, &x, &mut got, m, k, None);

            let mut scratch_s = vec![7.0f32; 1];
            let mut scratch_q = vec![7i8; 1];
            let mut got_scratch = vec![0.0f32; m];
            gemv_dispatch(
                dtype,
                &data,
                &x,
                &mut got_scratch,
                m,
                k,
                Some((&mut scratch_s, &mut scratch_q)),
            );
            for (i, (a, b)) in got.iter().zip(&got_scratch).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{dtype:?} row {i}: lending scratch changed the result \
                     ({a} vs {b}) — the two `kq_gemv!` arms have diverged"
                );
            }

            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1.0);
            for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - wv).abs() <= 0.02 * scale,
                    "{dtype:?} row {i}: dispatch gave {g}, dequantized reference {wv} \
                     — a wrong kernel, not a rounding difference"
                );
            }
        }
    }

    #[test]
    fn test_matmul_f32_identity() {
        // 2x2 identity matrix × [1,2; 3,4] = [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_f32_3x2_times_2x4() {
        // A = [[1,2],[3,4],[5,6]], B = [[1,2,3,4],[5,6,7,8]]
        // C = [[11,14,17,20],[23,30,37,44],[35,46,57,68]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 12];
        matmul_f32(&a, &b, &mut c, 3, 4, 2);
        assert_eq!(
            c,
            vec![
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_rmsnorm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;

        // rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps)
        let rms = (7.5f32 + eps).sqrt();
        let expected: Vec<f32> = vec![1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];

        rmsnorm(&mut x, &weight, eps);

        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "rmsnorm[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_rmsnorm_with_weight() {
        let mut x = vec![2.0, 2.0];
        let weight = vec![3.0, 0.5];
        let eps = 1e-5;

        let rms = (4.0f32 + eps).sqrt(); // sqrt((4+4)/2 + eps)
        let inv_rms = 1.0 / rms;
        let expected = [2.0 * inv_rms * 3.0, 2.0 * inv_rms * 0.5];

        rmsnorm(&mut x, &weight, eps);

        for (i, (&got, &exp)) in x.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "rmsnorm[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0, 1.0, -1.0, 5.0];
        silu_inplace(&mut x);

        // silu(0) = 0, silu(1) = 1/(1+e^-1) ≈ 0.7311, silu(-1) ≈ -0.2689, silu(5) ≈ 4.9665
        assert!((x[0] - 0.0).abs() < 1e-5);
        assert!((x[1] - 0.7311).abs() < 1e-3);
        assert!((x[2] - (-0.2689)).abs() < 1e-3);
        assert!((x[3] - 4.9665).abs() < 1e-3);
    }

    #[test]
    fn test_silu_mul_inplace() {
        let mut gate = vec![0.0, 1.0, -1.0, 5.0];
        let up = vec![2.0, 3.0, 0.5, 1.0];

        // Reference: silu(gate) * up
        let mut gate_ref = gate.clone();
        silu_inplace(&mut gate_ref);
        mul_inplace(&mut gate_ref, &up);

        silu_mul_inplace(&mut gate, &up);

        for (i, (&got, &expected)) in gate.iter().zip(gate_ref.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "silu_mul mismatch at {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_sigmoid() {
        let mut x = vec![0.0f32, 2.0, -2.0, 10.0, -10.0];
        sigmoid_inplace(&mut x);
        // sigmoid(0) = 0.5; sigmoid(±large) → {1, 0}; sigmoid(2) ≈ 0.881.
        assert!((x[0] - 0.5).abs() < 1e-5, "sigmoid(0) = {}", x[0]);
        assert!((x[1] - 0.880_797).abs() < 1e-3, "sigmoid(2) = {}", x[1]);
        assert!((x[2] - 0.119_203).abs() < 1e-3, "sigmoid(-2) = {}", x[2]);
        assert!(x[3] > 0.999_5, "sigmoid(10) = {}", x[3]);
        assert!(x[4] < 5e-4, "sigmoid(-10) = {}", x[4]);
    }

    #[test]
    fn test_glu_split() {
        // input = [a0, a1, b0, b1] → output[i] = a[i] * sigmoid(b[i]).
        // Pick b values with known sigmoids: b0 = 0 → sigmoid = 0.5;
        // b1 = large → sigmoid = ~1.
        let input = vec![3.0, 7.0, 0.0, 100.0];
        let mut output = vec![0.0; 2];
        glu_split(&input, &mut output);
        assert!((output[0] - 1.5).abs() < 1e-5, "got {}", output[0]); // 3 * 0.5
        assert!((output[1] - 7.0).abs() < 1e-3, "got {}", output[1]); // 7 * ~1
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);

        // Sum should be 1.0
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Values should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);

        // Check known values: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
        assert!((x[0] - 0.0900).abs() < 1e-3);
        assert!((x[1] - 0.2447).abs() < 1e-3);
        assert!((x[2] - 0.6652).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_zero_mean_unit_var_after_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        layer_norm_inplace(&mut x, &weight, &bias, 1e-5);

        // After LayerNorm with weight=1, bias=0: mean ≈ 0, std ≈ 1.
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        assert!(mean.abs() < 1e-5, "mean = {mean}");
        assert!((var.sqrt() - 1.0).abs() < 1e-3, "std = {}", var.sqrt());
    }

    #[test]
    fn test_layer_norm_applies_affine() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        // Use weight=2, bias=10 to verify the affine post-norm transform.
        let weight = vec![2.0; 4];
        let bias = vec![10.0; 4];
        layer_norm_inplace(&mut x, &weight, &bias, 1e-5);

        // After norm without affine, values would have mean 0, std 1.
        // With weight=2, bias=10: mean shifts to 10, std becomes 2.
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!((mean - 10.0).abs() < 1e-3, "mean = {mean} expected ~10");
    }

    #[test]
    fn test_gelu_erf_known_values() {
        // GELU(0) = 0
        // GELU(1) ≈ 0.8413 (from PyTorch torch.nn.functional.gelu)
        // GELU(-1) ≈ -0.1587
        // GELU(2) ≈ 1.9545
        let mut x = vec![0.0f32, 1.0, -1.0, 2.0];
        gelu_erf_inplace(&mut x);
        assert!(x[0].abs() < 1e-4, "gelu(0) = {}", x[0]);
        assert!((x[1] - 0.8413).abs() < 5e-3, "gelu(1) = {}", x[1]);
        assert!((x[2] + 0.1587).abs() < 5e-3, "gelu(-1) = {}", x[2]);
        assert!((x[3] - 1.9545).abs() < 5e-3, "gelu(2) = {}", x[3]);
    }

    #[test]
    fn test_gelu_tanh_known_values() {
        // tanh-approx GELU values from PyTorch's
        // F.gelu(x, approximate="tanh") at f64 precision. Differs
        // from the erf-form by ~1e-3 around |x|≈1 — picking up that
        // gap is exactly the point of having two kernels.
        //   gelu_tanh(0)  = 0
        //   gelu_tanh(1)  ≈ 0.84119198
        //   gelu_tanh(-1) ≈ -0.15880802
        //   gelu_tanh(2)  ≈ 1.95459783
        // Tolerance 1e-4 catches a constant-precision regression
        // (e.g. someone "fixing" SQRT_2_OVER_PI to a wrong digit) —
        // tighter than the original 5e-3 while still leaving room
        // for the f32 vs reference-f64 rounding floor.
        let mut x = vec![0.0f32, 1.0, -1.0, 2.0];
        gelu_inplace(&mut x);
        // f32 references rounded from PyTorch f64 (the underlying
        // formula is rational-arithmetic, so f32 precision is the
        // floor here). Tolerance 1e-4 catches a constant-precision
        // regression while leaving room for the rounding floor.
        assert!(x[0].abs() < 1e-6, "gelu_tanh(0) = {}", x[0]);
        assert!((x[1] - 0.841_192).abs() < 1e-4, "gelu_tanh(1) = {}", x[1]);
        assert!((x[2] + 0.158_808).abs() < 1e-4, "gelu_tanh(-1) = {}", x[2]);
        assert!((x[3] - 1.954_598).abs() < 1e-4, "gelu_tanh(2) = {}", x[3]);
    }

    #[test]
    fn test_conv1d_standard_identity_kernel() {
        // 1×1 kernel with weight=1 = identity (per-channel passthrough).
        // Input: 2 channels × 3 timesteps.
        let input = vec![
            1.0, 2.0, 3.0, // channel 0
            4.0, 5.0, 6.0, // channel 1
        ];
        // Weight shape [out=2, in=2, k=1]: identity per-channel mapping
        // (weight[0,0,0]=1, weight[1,1,0]=1, others=0).
        let weight = vec![
            1.0, 0.0, // out 0: in 0=1, in 1=0
            0.0, 1.0, // out 1: in 0=0, in 1=1
        ];
        let mut output = vec![0.0; 2 * 3];
        let t_out = conv1d(&input, &weight, None, &mut output, 2, 2, 3, 1, 1, 0, 1);
        assert_eq!(t_out, 3);
        assert_eq!(output, input);
    }

    #[test]
    fn test_conv1d_depthwise_per_channel() {
        // Depthwise conv with kernel=3, stride=1, pad=1 (same-size output).
        // 2 channels, weight is one kernel per channel.
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];
        // groups=2 (depthwise), weight shape [out=2, in/groups=1, k=3].
        // With pad=1, output[t] = sum_k input[t+k-1] * kernel[k].
        // Kernel = [1, 0, 0] taps input[t-1] only → values shift RIGHT
        // by 1 (each output position takes the value to its left).
        // Kernel = [0, 0, 1] taps input[t+1] only → values shift LEFT
        // by 1 (each output position takes the value to its right).
        let weight = vec![
            1.0, 0.0, 0.0, // ch 0 kernel — right-shift / delay
            0.0, 0.0, 1.0, // ch 1 kernel — left-shift / advance
        ];
        let mut output = vec![0.0; 2 * 4];
        let t_out = conv1d(&input, &weight, None, &mut output, 2, 2, 4, 3, 1, 1, 2);
        assert_eq!(t_out, 4);
        // Channel 0 input [1,2,3,4] right-shift → [0, 1, 2, 3] (pad fills the leading zero).
        assert_eq!(&output[0..4], &[0.0, 1.0, 2.0, 3.0]);
        // Channel 1 input [5,6,7,8] left-shift → [6, 7, 8, 0] (pad fills the trailing zero).
        assert_eq!(&output[4..8], &[6.0, 7.0, 8.0, 0.0]);
    }

    #[test]
    fn test_conv1d_strided() {
        // 1 channel, kernel=2, stride=2, pad=0. Halves the timesteps.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 6 timesteps
        let weight = vec![1.0, 1.0]; // sum-pair kernel
        let mut output = vec![0.0; 3]; // (6-2)/2 + 1 = 3
        let t_out = conv1d(&input, &weight, None, &mut output, 1, 1, 6, 2, 2, 0, 1);
        assert_eq!(t_out, 3);
        // Sum pairs: [1+2, 3+4, 5+6] = [3, 7, 11]
        assert_eq!(output, vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0]; // 1×1 identity
        let bias = vec![5.0];
        let mut output = vec![0.0; 3];
        conv1d(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            1,
            1,
            3,
            1,
            1,
            0,
            1,
        );
        // Input + bias.
        assert_eq!(output, vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_conv2d_pointwise_identity() {
        // 1×1 kernel with identity per-channel weights = passthrough.
        // Input: 2 channels × 2×3 (h × w).
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // channel 0 (h=2, w=3)
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // channel 1
        ];
        // Weight [oc=2, ic=2, kh=1, kw=1]: identity per-channel.
        let weight = vec![
            1.0, 0.0, // oc 0: ic 0=1, ic 1=0
            0.0, 1.0, // oc 1: ic 0=0, ic 1=1
        ];
        let mut output = vec![0.0; 2 * 2 * 3];
        let (h_out, w_out) = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            2,
            2,
            2,
            3,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
        );
        assert_eq!((h_out, w_out), (2, 3));
        assert_eq!(output, input);
    }

    #[test]
    fn test_conv2d_strided_with_pad() {
        // Stride 2x2, pad 1x1, kernel 3x3, single channel — common
        // first-layer subsampling pattern (LFM2A stem layer.0 shape).
        // Input: 1 × 4 × 4. Output dims: (4 + 2*1 - 3)/2 + 1 = 2.
        let input: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        // Mean-pool kernel (1/9 each cell).
        let weight = vec![1.0 / 9.0; 9];
        let mut output = vec![0.0; 2 * 2];
        let (h_out, w_out) = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            4,
            4,
            3,
            3,
            2,
            2,
            1,
            1,
            1,
        );
        assert_eq!((h_out, w_out), (2, 2));
        // For pad=1 + 4×4 input with mean-pool 3×3 stride-2 kernel,
        // each output covers a 3×3 window centered on (oh*2, ow*2)
        // in the unpadded input space, with out-of-bounds rows/cols
        // contributing zero. Spot-check the top-left output:
        // window covers (-1, -1)..(1, 1) in unpadded coords; valid
        // cells are input[0..2, 0..2] = [1, 2, 5, 6]. Sum = 14;
        // mean = 14/9.
        assert!((output[0] - 14.0 / 9.0).abs() < 1e-6, "got {}", output[0]);
    }

    #[test]
    fn test_conv2d_depthwise_per_channel_independence() {
        // groups = in_channels = 2 → depthwise. Each input channel
        // is convolved with its own kernel, no cross-channel sums.
        // Verify channel 1's data doesn't leak into channel 0's
        // output and vice versa.
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // ch 0 (h=2, w=2)
            10.0, 20.0, 30.0, 40.0, // ch 1
        ];
        // Two depthwise kernels, each 1×1: ch 0 weight = 2.0 (double),
        // ch 1 weight = 0.5 (half).
        let weight = vec![2.0, 0.5];
        let mut output = vec![0.0; 2 * 2 * 2];
        let (h_out, w_out) = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            0,
            0,
            2,
        );
        assert_eq!((h_out, w_out), (2, 2));
        assert_eq!(&output[0..4], &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(&output[4..8], &[5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_conv2d_with_bias() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 × 2 × 2
        let weight = vec![1.0]; // 1×1 identity
        let bias = vec![10.0];
        let mut output = vec![0.0; 4];
        let (h_out, w_out) = conv2d(
            &input,
            &weight,
            Some(&bias),
            &mut output,
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
        );
        assert_eq!((h_out, w_out), (2, 2));
        assert_eq!(output, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_conv2d_pad_zero_contribution() {
        // 3×3 kernel with all weights = 1, pad = 1, single 1×1 input.
        // Output is 1×1: only the input center contributes; the 8
        // surrounding pad cells contribute zero. Result = input value.
        let input = vec![7.0];
        let weight = vec![1.0; 9];
        let mut output = vec![0.0; 1];
        let (h_out, w_out) = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            1,
            1,
            1,
            1,
            3,
            3,
            1,
            1,
            1,
            1,
            1,
        );
        assert_eq!((h_out, w_out), (1, 1));
        assert_eq!(output[0], 7.0);
    }

    #[test]
    fn test_conv2d_grouped_non_depthwise_fallback() {
        // groups = 2 with in_ch = 4, out_ch = 4 (in_per_group =
        // out_per_group = 2). This is the only conv2d shape that
        // misses all three fast paths and falls through to the
        // naive 7-loop. Verify the fallback still computes a
        // correct grouped conv: each group sees only its own input
        // channels and produces only its own output channels.
        //
        // Input: 4 ch × 1 × 1 (single spatial position).
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Weight [oc=4, in_per_group=2, kh=1, kw=1]:
        //   group 0 (oc 0,1) sees in 0,1: identity within group
        //   group 1 (oc 2,3) sees in 2,3: identity within group
        let weight = vec![
            1.0, 0.0, // oc 0: in_grp 0 = 1, in_grp 1 = 0
            0.0, 1.0, // oc 1: in_grp 0 = 0, in_grp 1 = 1
            1.0, 0.0, // oc 2 (grp 1): in_grp 0 = 1, in_grp 1 = 0
            0.0, 1.0, // oc 3 (grp 1): in_grp 0 = 0, in_grp 1 = 1
        ];
        let mut output = vec![0.0; 4];
        let (h_out, w_out) = conv2d(
            &input,
            &weight,
            None,
            &mut output,
            4,
            4,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            2,
        );
        assert_eq!((h_out, w_out), (1, 1));
        // group 0 maps in 0→oc 0, in 1→oc 1.
        // group 1 maps in 2→oc 2, in 3→oc 3.
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ggml_expf() {
        // ggml_expf should approximate exp() within ~1.5 ULPs
        let test_vals = [0.0f32, 1.0, -1.0, 2.0, -5.0, -10.0, -50.0, 80.0];
        for &x in &test_vals {
            let got = ggml_expf(x);
            let expected = x.exp();
            let rel_err = if expected.abs() > 1e-10 {
                ((got - expected) / expected).abs()
            } else {
                (got - expected).abs()
            };
            assert!(
                rel_err < 1e-5,
                "ggml_expf({x}) = {got}, expected {expected}, rel_err = {rel_err}"
            );
        }
        // Edge cases
        assert!(ggml_expf(100.0).is_infinite() || ggml_expf(100.0) > 1e30);
        assert!(ggml_expf(-200.0) < 1e-30);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not overflow
        let mut x = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rope_basic() {
        // Basic test: pos=0 should not rotate (cos(0)=1, sin(0)=0)
        let mut q = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, dim=4
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        rope(&mut q, &mut k, 0, 1, 1, 4, 10000.0);

        // At pos=0, theta=0 for all dims, so cos=1, sin=0 → no change
        for i in 0..4 {
            assert!((q[i] - q_orig[i]).abs() < 1e-5, "q[{i}] changed at pos=0");
            assert!((k[i] - k_orig[i]).abs() < 1e-5, "k[{i}] changed at pos=0");
        }
    }

    #[test]
    fn test_rope_rotates() {
        // At pos > 0, values should change
        let mut q = vec![1.0, 0.0, 0.0, 0.0]; // 1 head, dim=4
        let mut k = vec![1.0, 0.0, 0.0, 0.0];

        rope(&mut q, &mut k, 10, 1, 1, 4, 10000.0);

        // q should have been rotated — not identical anymore
        assert!((q[0] - 1.0).abs() > 1e-3 || (q[2]).abs() > 1e-3);
    }

    #[test]
    fn test_rope_norm_basic() {
        // pos=0 → no rotation, same as NEOX.
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        rope_norm(&mut q, &mut k, 0, 1, 1, 4, 10000.0, None);

        for i in 0..4 {
            assert!((q[i] - q_orig[i]).abs() < 1e-5, "q[{i}] changed at pos=0");
            assert!((k[i] - k_orig[i]).abs() < 1e-5, "k[{i}] changed at pos=0");
        }
    }

    #[test]
    fn test_rope_norm_rotates_adjacent_pairs() {
        // NORM rotates (x[2i], x[2i+1]). With head=[1,0,1,0] and the first
        // pair's angle = pos*1 = 1 rad, the first pair must become
        // (cos 1, sin 1); split-halves (NEOX) would instead pair (x0,x2).
        let head_dim = 4;
        let freq_base = 10000.0_f32;
        let pos = 1usize;
        let mut head = vec![1.0, 0.0, 1.0, 0.0];
        apply_rope_norm_to_head(&mut head, pos, head_dim, freq_base, None);

        let theta0 = pos as f32; // i=0
        assert!((head[0] - theta0.cos()).abs() < 1e-5);
        assert!((head[1] - theta0.sin()).abs() < 1e-5);
        // Second pair angle = pos * freq_base^(-2/head_dim).
        let theta1 = pos as f32 * freq_base.powf(-2.0 / head_dim as f32);
        assert!((head[2] - theta1.cos()).abs() < 1e-5);
        assert!((head[3] - theta1.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_rope_norm_freq_factors_divide_theta() {
        // Llama-3 RoPE scaling divides each pair's angle by freq_factors[i]
        // (ggml: theta_base / ff). factor 2.0 on pair 0 halves its rotation angle
        // vs the plain-RoPE case.
        let head_dim = 4;
        let freq_base = 10000.0_f32;
        let pos = 1usize;
        let ff = [2.0_f32, 4.0]; // head_dim/2 entries
        let mut head = vec![1.0, 0.0, 1.0, 0.0];
        apply_rope_norm_to_head(&mut head, pos, head_dim, freq_base, Some(&ff));

        // Pair 0: angle = pos / ff[0] = 1/2.
        let theta0 = pos as f32 / ff[0];
        assert!((head[0] - theta0.cos()).abs() < 1e-5);
        assert!((head[1] - theta0.sin()).abs() < 1e-5);
        // Pair 1: angle = (pos * freq_base^(-2/head_dim)) / ff[1].
        let theta1 = pos as f32 * freq_base.powf(-2.0 / head_dim as f32) / ff[1];
        assert!((head[2] - theta1.cos()).abs() < 1e-5);
        assert!((head[3] - theta1.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_rope_norm_delta_composition() {
        // R(p_new) == R(p_new - p_old) ∘ R(p_old) for the NORM layout too.
        let head_dim = 8;
        let freq_base = 1_000_000.0_f32;
        let raw = vec![0.3, -1.1, 2.0, 0.5, -0.7, 1.3, 0.9, -0.2];

        let mut direct = raw.clone();
        apply_rope_norm_to_head(&mut direct, 13, head_dim, freq_base, None);

        let mut composed = raw.clone();
        apply_rope_norm_to_head(&mut composed, 5, head_dim, freq_base, None);
        apply_rope_norm_delta_to_head(&mut composed, 13 - 5, head_dim, freq_base, None);

        for i in 0..head_dim {
            assert!(
                (direct[i] - composed[i]).abs() < 1e-4,
                "norm rope delta mismatch at {i}: {} vs {}",
                direct[i],
                composed[i]
            );
        }
    }

    #[test]
    fn test_conv1d_depthwise_identity() {
        // Kernel [0, 1, 0] is identity
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // seq=3, channels=2
        let weight = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]; // 2 channels, kernel=3
        let mut output = vec![0.0; 6];

        conv1d_depthwise(&input, &weight, None, &mut output, 2, 3, 3);

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_conv1d_depthwise_with_bias() {
        let input = vec![1.0, 2.0]; // seq=1, channels=2
        let weight = vec![1.0, 1.0]; // 2 channels, kernel=1
        let bias = vec![10.0, 20.0];
        let mut output = vec![0.0; 2];

        conv1d_depthwise(&input, &weight, Some(&bias), &mut output, 2, 1, 1);

        assert_eq!(output, vec![11.0, 22.0]);
    }

    #[test]
    fn test_add_inplace() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        add_inplace(&mut a, &b);
        assert_eq!(a, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_inplace() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        mul_inplace(&mut a, &b);
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_par_rows_n_basic() {
        // 3 rows × 2 columns, each row doubles its index
        let mut out = vec![0.0f32; 6];
        par_rows_n(&mut out, 2, 1, |(i, row)| {
            row[0] = i as f32;
            row[1] = i as f32 * 2.0;
        });
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0, 2.0, 4.0]);
    }

    #[test]
    fn test_par_rows_n_empty() {
        let mut out: Vec<f32> = vec![];
        par_rows_n(&mut out, 3, 1, |(_i, _row)| {
            panic!("should not be called");
        });
    }

    /// Reference scalar attention scores for testing.
    #[allow(clippy::too_many_arguments)]
    fn attn_scores_scalar(
        q: &[f32],
        k_cache: &[f32],
        scores: &mut [f32],
        kv_dim: usize,
        kv_h_off: usize,
        head_dim: usize,
        scale: f32,
        seq_len: usize,
    ) {
        for t in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_cache[t * kv_dim + kv_h_off + d];
            }
            scores[t] = dot * scale;
        }
    }

    /// Reference scalar attention values for testing.
    fn attn_values_scalar(
        scores: &[f32],
        v_cache: &[f32],
        out: &mut [f32],
        kv_dim: usize,
        kv_h_off: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for t in 0..seq_len {
                val += scores[t] * v_cache[t * kv_dim + kv_h_off + d];
            }
            out[d] = val;
        }
    }

    #[test]
    fn test_attn_scores_matches_scalar() {
        let head_dim = 64;
        let kv_dim = 128; // 2 KV heads × 64
        let kv_h_off = 64; // second KV head
        let seq_len = 10;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let k_cache: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 7 + 3) % 31) as f32 * 0.04 - 0.6)
            .collect();

        let mut expected = vec![0.0f32; seq_len];
        attn_scores_scalar(
            &q,
            &k_cache,
            &mut expected,
            kv_dim,
            kv_h_off,
            head_dim,
            scale,
            seq_len,
        );

        let mut actual = vec![0.0f32; seq_len];
        attn_scores(
            &q,
            &k_cache,
            &mut actual,
            kv_dim,
            kv_h_off,
            head_dim,
            scale,
            seq_len,
        );

        for t in 0..seq_len {
            let diff = (expected[t] - actual[t]).abs();
            assert!(
                diff < 1e-5,
                "attn_scores mismatch at t={t}: expected={}, actual={}, diff={diff}",
                expected[t],
                actual[t]
            );
        }
    }

    #[test]
    fn test_attn_values_matches_scalar() {
        let head_dim = 64;
        let kv_dim = 128;
        let kv_h_off = 0;
        let seq_len = 10;

        let scores: Vec<f32> = (0..seq_len)
            .map(|i| (i as f32 + 1.0) / seq_len as f32)
            .collect();
        let v_cache: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 11 + 5) % 29) as f32 * 0.03 - 0.4)
            .collect();

        let mut expected = vec![0.0f32; head_dim];
        attn_values_scalar(
            &scores,
            &v_cache,
            &mut expected,
            kv_dim,
            kv_h_off,
            head_dim,
            seq_len,
        );

        let mut actual = vec![0.0f32; head_dim];
        attn_values(
            &scores,
            &v_cache,
            &mut actual,
            kv_dim,
            kv_h_off,
            head_dim,
            seq_len,
        );

        for d in 0..head_dim {
            let diff = (expected[d] - actual[d]).abs();
            assert!(
                diff < 1e-4,
                "attn_values mismatch at d={d}: expected={}, actual={}, diff={diff}",
                expected[d],
                actual[d]
            );
        }
    }

    #[test]
    fn test_attn_scores_seq_len_zero() {
        let mut scores = vec![];
        attn_scores(&[0.0; 64], &[], &mut scores, 64, 0, 64, 0.125, 0);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_flash_attention_matches_naive() {
        // Compare flash attention output against the naive
        // attn_scores + softmax_inplace + attn_values pipeline.
        //
        // Setup: 4 query heads, 2 KV heads (group_size=2), head_dim=64,
        // 8 query tokens, start_pos=4 (so total seq_len up to 12).
        let n_heads = 4;
        let n_kv_heads = 2;
        let group_size = n_heads / n_kv_heads;
        let head_dim = 64;
        let hs = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let n = 8;
        let start_pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_seq = start_pos + n; // 12

        // Random Q in [hs, n] stride-n layout
        let mut q_mat = vec![0.0f32; hs * n];
        let mut seed: u64 = 0xCAFE_BABE;
        for v in q_mat.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as i32 as f32) * 1e-9;
        }

        // Random K/V cache in [total_seq, kv_dim] layout
        let mut k_cache = vec![0.0f32; total_seq * kv_dim];
        for v in k_cache.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as i32 as f32) * 1e-9;
        }
        let mut v_cache = vec![0.0f32; total_seq * kv_dim];
        for v in v_cache.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((seed >> 33) as i32 as f32) * 1e-9;
        }

        // ── Flash attention ────────────────────────────────────────
        // Kernel writes contiguous [group_size, n, head_dim] per KV head.
        // Scatter-copy back to [hs, n] stride-n for comparison with naive.
        let chunk_size = group_size * n * head_dim;
        let mut flash_raw = vec![0.0f32; n_kv_heads * chunk_size];
        for kv_h in 0..n_kv_heads {
            let chunk = &mut flash_raw[kv_h * chunk_size..(kv_h + 1) * chunk_size];
            flash_attention_gqa_cpu(
                &q_mat,
                &k_cache,
                &v_cache,
                chunk,
                kv_h * group_size,
                group_size,
                n,
                n, // q_stride
                kv_dim,
                kv_h * head_dim,
                head_dim,
                scale,
                start_pos,
            );
        }
        let mut flash_out = vec![0.0f32; hs * n];
        for kv_h in 0..n_kv_heads {
            for g in 0..group_size {
                let h = kv_h * group_size + g;
                let src_base = kv_h * chunk_size + g * n * head_dim;
                for j in 0..n {
                    for d in 0..head_dim {
                        flash_out[(h * head_dim + d) * n + j] =
                            flash_raw[src_base + j * head_dim + d];
                    }
                }
            }
        }

        // ── Naive reference ────────────────────────────────────────
        let mut naive_out = vec![0.0f32; hs * n];
        for j in 0..n {
            let seq_len = start_pos + j + 1; // causal: attend to 0..seq_len
            for h in 0..n_heads {
                let kv_h = h / group_size;
                let kv_h_offset = kv_h * head_dim;

                // Gather Q[h, j] from stride-n layout
                let mut q_head = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    q_head[d] = q_mat[(h * head_dim + d) * n + j];
                }

                // Scores
                let mut scores = vec![0.0f32; seq_len];
                attn_scores(
                    &q_head,
                    &k_cache,
                    &mut scores,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    scale,
                    seq_len,
                );

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted values
                let mut attn_out = vec![0.0f32; head_dim];
                attn_values(
                    &scores,
                    &v_cache,
                    &mut attn_out,
                    kv_dim,
                    kv_h_offset,
                    head_dim,
                    seq_len,
                );

                // Scatter-write to stride-n output
                for d in 0..head_dim {
                    naive_out[(h * head_dim + d) * n + j] = attn_out[d];
                }
            }
        }

        // ── Compare ────────────────────────────────────────────────
        let mut max_diff = 0.0f32;
        for i in 0..hs * n {
            max_diff = max_diff.max((flash_out[i] - naive_out[i]).abs());
        }
        assert!(
            max_diff < 1e-4,
            "flash vs naive max_diff = {max_diff} (expected < 1e-4)"
        );
    }

    /// Cost of one `RowPool` dispatch with essentially no work in it — the
    /// synchronization tax decode pays per GEMV.
    ///
    /// Worth having permanently because it settles an argument that recurs:
    /// decode issues ~113 dispatches per token (Llama-3.2-1B: 16 layers x 7
    /// GEMVs + logits), so "fuse the GEMVs to cut barriers" sounds compelling
    /// until the barrier is measured. On a Ryzen AI MAX+ 395 it is ~2 us
    /// spinning at the default 12 threads (~3 us at 16) — call it 0.2-0.4 ms
    /// of a 19 ms token, so 1-2%. Halving the dispatch count buys under 1%.
    ///
    /// The same run under `CERA_SPIN=0` reports ~240 us at 12 threads (~330 us
    /// at 16): a 100x jump. Parking and re-waking workers is the expensive
    /// path, and spin-before-park is what keeps the barrier off the critical
    /// path — which also means the spin is not free power-wise. Anyone tempted
    /// to shrink `SPIN_BEFORE_PARK` should look at that number first.
    ///
    /// Run with:
    /// `cargo test -p cera --release --lib backend::cpu::tests::microbench_dispatch -- --ignored --nocapture`
    // This measures native `RowPool` dispatch, and `threadpool::RowPool` is
    // gated off wasm32 (`par_rows` itself has a wasm impl over web workers), so
    // `parallel` alone would fail to compile on a threaded wasm build.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    #[test]
    #[ignore]
    fn microbench_dispatch() {
        use std::time::Instant;

        // The probe below already forces lazy pool init, so the warm-up loop is
        // for caller pinning and cache/steal-loop warmth, not init.
        let threads = crate::backend::threadpool::RowPool::decode().num_threads();

        let mut warm = vec![0.0f32; 4096];
        for _ in 0..100 {
            par_rows(&mut warm, gemv_min_rows(), |(i, v)| *v = i as f32);
        }

        eprintln!("\n=== RowPool dispatch cost ({threads} decode threads) ===");
        for rows in [512usize, 2048, 8192] {
            let mut y = vec![0.0f32; rows];
            let iters = 2000;
            // Trivial body on purpose: this measures the barrier and steal
            // loop, not the kernel.
            let t = Instant::now();
            for _ in 0..iters {
                par_rows(&mut y, gemv_min_rows(), |(_i, v)| *v += 1.0);
            }
            let per = t.elapsed().as_secs_f64() / iters as f64;
            // Observe `y` so the optimizer cannot elide the trivial body and
            // leave us timing an empty loop.
            std::hint::black_box(&y);
            eprintln!(
                "  rows={rows:<5} {:>7.1} us/dispatch  ->  {:>6.2} ms/token at 113 dispatches",
                per * 1e6,
                per * 1e3 * 113.0
            );
        }
    }

    /// Microbenchmark: measure GEMV throughput and effective memory bandwidth
    /// for the Q4_0 × Q8_0 pre-quantized kernel at FFN gate shape.
    ///
    /// Run with:
    /// `cargo test -p cera --release --lib backend::cpu::tests::microbench_gemv_q4_0 -- --ignored --nocapture`
    #[cfg(all(target_arch = "aarch64", feature = "parallel"))]
    #[test]
    #[ignore]
    fn microbench_gemv_q4_0() {
        use std::time::Instant;

        // The GEMV row loop runs on the persistent decode RowPool — rayon no
        // longer applies here. Touching the pool up front warms it (including
        // any calibration sweep) *before* the timed region, and reports the
        // width actually used.
        let n_threads = crate::backend::threadpool::RowPool::decode().num_threads();
        let m = 6912; // FFN gate rows
        let k = 2048; // hidden_size
        let iters = 200;

        // Random Q4_0 weight
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * size_of::<crate::quant::BlockQ4_0>();
        let mut weight = vec![0u8; m * row_bytes];
        let mut s: u64 = 0xdead_beef;
        for b in weight.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (s >> 33) as u8;
        }

        // Random input, pre-quantized to Q8_0
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 31) % 127) as f32 * 0.01 - 0.5)
            .collect();
        let (x_scales, x_quants) = quantize_f32_to_q8_0(&x);
        let mut y = vec![0.0f32; m];

        // Warmup
        gemv_q4_0_with_q8(&weight, &x_scales, &x_quants, &mut y, m, k);

        let t0 = Instant::now();
        for _ in 0..iters {
            gemv_q4_0_with_q8(&weight, &x_scales, &x_quants, &mut y, m, k);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;

        let weight_bytes = m * row_bytes;
        let input_bytes = x_scales.len() * 4 + x_quants.len();
        let total_bytes = weight_bytes + input_bytes;
        let bw_gbps = (total_bytes as f64 / per_call) / 1e9;

        eprintln!("\n=== GEMV Q4_0×Q8_0 microbench (m={m}, k={k}) ===");
        eprintln!("  per-call: {:.1} µs", per_call * 1e6);
        eprintln!("  weight:   {:.2} MB", weight_bytes as f64 / 1e6);
        eprintln!("  bandwidth: {:.1} GB/s", bw_gbps);
        eprintln!("  decode pool threads: {n_threads}");

        // Also measure a large GEMV (output projection shape)
        let m_large = 65536;
        let mut weight_large = vec![0u8; m_large * row_bytes];
        for b in weight_large.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (s >> 33) as u8;
        }
        let mut y_large = vec![0.0f32; m_large];
        gemv_q4_0_with_q8(
            &weight_large,
            &x_scales,
            &x_quants,
            &mut y_large,
            m_large,
            k,
        );

        let t0 = Instant::now();
        for _ in 0..20 {
            gemv_q4_0_with_q8(
                &weight_large,
                &x_scales,
                &x_quants,
                &mut y_large,
                m_large,
                k,
            );
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / 20.0;
        let weight_bytes_large = m_large * row_bytes;
        let bw_large = ((weight_bytes_large + input_bytes) as f64 / per_call) / 1e9;

        eprintln!("\n=== GEMV Q4_0×Q8_0 large (m={m_large}, k={k}) ===");
        eprintln!("  per-call: {:.1} µs", per_call * 1e6);
        eprintln!("  weight:   {:.2} MB", weight_bytes_large as f64 / 1e6);
        eprintln!("  bandwidth: {:.1} GB/s", bw_large);
    }
}

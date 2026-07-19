// Architecture-independent transformer machinery shared by the dense text models
// (`llama.rs`: Qwen2/Qwen3/LLaMA/Mistral/Granite) and `lfm2.rs`. Holds the weight
// plumbing (`WeightRef`, `resolve_weight`, `gemv`/`gemv_preq`, `dequantize_row*`,
// `quantize_to_scratch`) and the per-token kernels (`forward_attn_block`,
// `forward_ffn_block`).
//
// LFM2 shares the `WeightRef` type, the plumbing helpers, and `forward_ffn_block`.
// Its attention stays in `lfm2.rs` because of the TurboQuant KV-compression branches
// (compressed key/value caches + the GQA-batched TQ path), which don't belong in this
// generic helper; likewise LFM2's batched/BLAS prefill is model-specific.

use anyhow::{Context, Result};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, LayerState};
use crate::tensor::DType;

// ── Oracle dump sink (test-only correctness gate) ───────────────────────────
//
// When enabled, records the full-tensor `sum` of named sub-step activations in
// call order, so a test can compare them against per-node `sum` checksums
// captured from llama.cpp (see `cera/tests/oracle_text.rs` and
// `scripts/oracle/`). Off (and free) unless `oracle_dump::begin()` is called.
// Records every occurrence (once per token during prefill) so the test can sum
// all-position nodes and take the last occurrence for last-position nodes.
//
// `#[doc(hidden)] pub` so the integration test (`tests/oracle_text.rs`, a
// separate crate) can drive it; not part of the supported public API.
//
// Callers that must allocate to build a node name (e.g. `format!("l_out-{i}")`)
// should guard with `is_active()` so disabled inference pays nothing beyond a
// cheap thread-local bool read.
#[doc(hidden)]
pub mod oracle_dump {
    use std::cell::RefCell;

    thread_local! {
        static SINK: RefCell<Option<Vec<(String, f64)>>> = const { RefCell::new(None) };
    }

    /// Start collecting (clears any prior buffer).
    pub fn begin() {
        SINK.with(|s| *s.borrow_mut() = Some(Vec::new()));
    }

    /// Stop collecting and return the recorded `(name, sum)` occurrences.
    pub fn take() -> Vec<(String, f64)> {
        SINK.with(|s| s.borrow_mut().take().unwrap_or_default())
    }

    /// Whether collection is active. Lets hot-path callers skip building node
    /// names (and the record call) when the dump is off.
    #[inline]
    pub fn is_active() -> bool {
        SINK.with(|s| s.borrow().is_some())
    }

    /// Record the sum of `data` under `name` if collection is active.
    #[inline]
    pub(crate) fn record(name: &str, data: &[f32]) {
        SINK.with(|s| {
            if let Some(buf) = s.borrow_mut().as_mut() {
                buf.push((name.to_string(), data.iter().map(|&x| x as f64).sum()));
            }
        });
    }
}

// ── Pre-resolved weight reference ───────────────────────────────────────────

/// Pre-resolved reference to a quantized weight in the mmap. Computed once at
/// load time to avoid HashMap lookups during inference. Semantics match
/// `lfm2::WeightRef`.
#[derive(Debug, Clone)]
pub(crate) struct WeightRef {
    pub start: usize,
    pub size: usize,
    pub dtype: DType,
    pub m: usize,
    pub k: usize,
}

/// Resolve a tensor name to a pre-computed byte range in the mmap.
pub(crate) fn resolve_weight(gguf: &GgufFile, name: &str) -> Result<WeightRef> {
    let info = gguf
        .tensors
        .get(name)
        .with_context(|| format!("tensor not found: {name}"))?;

    // info.offset is already absolute (data_offset + raw_offset from GGUF)
    let start =
        usize::try_from(info.offset).with_context(|| format!("tensor {name} offset overflow"))?;

    let size = info.size_bytes;
    let dtype = info.dtype;

    // GGUF shape: [inner_dim, outer_dim] → in memory: outer_dim rows of inner_dim elements
    let k = info.shape.first().copied().unwrap_or(1); // inner dim (elements per row)
    let m = if info.shape.len() > 1 {
        info.shape[1]
    } else {
        1
    }; // outer dim (number of rows)

    Ok(WeightRef {
        start,
        size,
        dtype,
        m,
        k,
    })
}

/// Get the raw bytes for a pre-resolved weight.
#[inline]
pub(crate) fn weight_data<'a>(gguf: &'a GgufFile, wref: &WeightRef) -> &'a [u8] {
    &gguf.mmap_data()[wref.start..wref.start + wref.size]
}

/// GEMV dispatch without scratch buffers.
pub(crate) fn gemv(gguf: &GgufFile, wref: &WeightRef, x: &[f32], y: &mut [f32]) {
    let data = weight_data(gguf, wref);
    cpu::gemv_dispatch(wref.dtype, data, x, y, wref.m, wref.k, None);
}

/// GEMV with pre-quantized Q8_0 input (skips re-quantizing x for each weight
/// matrix). For Q4_0/Q8_0/Q6K weights the integer dot-product path is used;
/// other dtypes fall back to the f32 path.
#[cfg(target_arch = "aarch64")]
pub(crate) fn gemv_preq(
    gguf: &GgufFile,
    wref: &WeightRef,
    x_f32: &[f32],
    q8s: &[f32],
    q8q: &[i8],
    y: &mut [f32],
) {
    let data = weight_data(gguf, wref);
    cpu::gemv_with_preq(wref.dtype, data, q8s, q8q, x_f32, y, wref.m, wref.k);
}

/// Quantize `x` to Q8_0 into the state's reusable scratch buffers.
#[cfg(target_arch = "aarch64")]
pub(crate) fn quantize_to_scratch(x: &[f32], state: &mut InferenceState) {
    assert_eq!(
        x.len() % 32,
        0,
        "quantize_to_scratch: x.len() must be divisible by 32"
    );
    let nb = x.len() / 32;
    state.scratch.q8_scales.resize(nb, 0.0);
    state.scratch.q8_quants.resize(x.len(), 0);
    unsafe {
        crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
            x,
            &mut state.scratch.q8_scales,
            &mut state.scratch.q8_quants,
        );
    }
}

// ── Batched-GEMM prefill helpers ────────────────────────────────────────────
//
// Shared by the dense-transformer (`llama.rs`) and LFM2 (`lfm2.rs`) CPU prefill
// paths, which read each weight matrix once for all N prompt tokens instead of
// the per-token GEMV loop. `try_blas_prefill_gemm` dequantizes the weight and
// runs an f32 SGEMM (any target, `blas` feature); `gemm_preq`/`quantize_columns`
// are the NEON fallback that pre-quantizes the input columns to Q8_0 and uses
// the integer-dot kernels (aarch64, no `blas`).

/// The weight dtypes the batched prefill GEMM can consume.
///
/// **This is the single source of truth for the LFM2 fast path.** The LFM2 gates and
/// both implementations (`gemm_preq`, `try_blas_prefill_gemm`) must agree, or a model
/// silently loses batched prefill — which is exactly what happened: the gates admitted
/// only `Q4_0 | Q8_0`, and a `Q4_K_M` file (which is *not* uniformly Q4_K — it mixes
/// Q4_K, Q6_K, and often Q5_K) matched none of them, so **every layer fell back to the
/// per-token GEMV loop, silently**. Add a dtype here only once *both* implementations
/// handle it.
///
/// `llama.rs` deliberately does **not** call this: its gate stays Q4_0/Q8_0-only until
/// there is a dense-transformer fixture that actually exercises a widened path (see the
/// comment at its gate). Its dtype set is a strict subset of this one, so it can only
/// be over-conservative, never wrong — but do not assume widening this function widens
/// llama too.
///
/// The K-quant arm is **runtime**-gated, not just dtype-gated: the Q4_K/Q6_K int8
/// GEMMs exist only in `dotprod` form. If this admitted them on a CPU without
/// FEAT_DotProd, `gemm_preq` would decline and the matmul would be *silently
/// skipped* — and because the callers reuse one output buffer across layers, that is
/// not even zeros, it is the *previous layer's* activations. Under `blas` the question
/// is moot: that path dequantizes to f32 and SGEMMs, so it handles any dtype it can
/// dequantize.
///
/// `k` is the weight's inner dimension: K-quant superblocks are 256 wide, so a
/// `k` that is not a multiple of 256 cannot be handled (GGUF should never produce
/// one — a row that short could not have been K-quantized in the first place — but
/// "the format guarantees it" is precisely how the last two silent fallbacks got
/// written, so it is checked rather than assumed).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
pub(crate) fn batched_gemm_supports(dtype: DType, k: usize) -> bool {
    match dtype {
        // Not unconditional: on x86 the int8 GEMM needs VNNI, and a non-VNNI
        // host must stay on the per-token GEMV fallback. Under `blas` the
        // question is moot — that path dequantizes and SGEMMs.
        DType::Q4_0 | DType::Q8_0 => {
            cfg!(feature = "blas") || crate::backend::cpu::int8_gemm_available()
        }
        DType::Q4KM | DType::Q6K => k_quant_gemm_available() && k % 256 == 0,
        _ => false,
    }
}

/// Whether the K-quant batched GEMM can actually run here — see
/// [`batched_gemm_supports`].
///
/// Cfg'd to the targets that have a batched path at all (the caller gates carry the
/// same cfg). Without it this is dead code on wasm and on any target without a
/// batched path, which the CI lint job (`cargo clippy --workspace --all-targets --
/// -D warnings`, ubuntu, no `blas`) turns into a hard error — an aarch64 dev
/// machine cannot reproduce that. It *is* called on x86_64 now (where it returns
/// `false`, there being no VNNI K-quant GEMM), so this is a lint cfg, not a
/// statement about which targets reach it.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
fn k_quant_gemm_available() -> bool {
    // BLAS dequantizes the weight and SGEMMs, so it needs no int8 kernel.
    #[cfg(feature = "blas")]
    {
        true
    }
    #[cfg(all(not(feature = "blas"), target_arch = "aarch64"))]
    {
        crate::backend::simd::neon::k_quant_gemm_available()
    }
    // No BLAS and no NEON: there is no batched path at all on this target (the
    // caller gates are themselves cfg'd off), so the answer is moot but must be
    // `false` rather than optimistic.
    // No BLAS and no NEON K-quant kernel (x86's VNNI path covers Q4_0/Q8_0
    // only), so there is no batched K-quant path on this target.
    #[cfg(all(not(feature = "blas"), not(target_arch = "aarch64")))]
    {
        false
    }
}

/// Report — once per offending dtype — that a weight knocked prefill off the
/// batched GEMM path.
///
/// A gate that declines in silence is the bug, not the missing kernel. This cost
/// ~4x prefill on CPU (T1) and ~340x the submits on GPU (T8) before anyone noticed,
/// both times because the fallback said nothing. If prefill is slow and this is
/// quiet, the dtypes are not the reason.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "blas"))]
pub(crate) fn warn_unbatchable(tensor: &str, dtype: DType) {
    use std::sync::Mutex;
    // A Vec, not a HashSet: `DType` is not `Hash`, the set holds a handful of
    // entries at most, and deriving `Hash` on a core enum to dedupe a warning
    // would be the tail wagging the dog.
    static SEEN: Mutex<Vec<DType>> = Mutex::new(Vec::new());
    let mut guard = match SEEN.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(), // a poisoned warn-dedupe set must not kill inference
    };
    if !guard.contains(&dtype) {
        guard.push(dtype);
        tracing::warn!(
            "prefill fell back to the per-token path: `{tensor}` is {dtype:?}, which is \
             not supported on the batched path for this model. Prefill will be several \
             times slower than it should be."
        );
    }
}

/// Prefill GEMM through BLAS: dequantize `wref` into `dequant_scratch[..m*k]`,
/// then SGEMM `out[m, n] = weight[m, k] @ b[k, n]` in row-major (`b`/`out` are
/// row-major `[k|m, n]`, stride `n`). Returns `true` for the supported dtypes;
/// callers gate on dtype upfront so the `false` arm is defensive.
#[cfg(feature = "blas")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_blas_prefill_gemm(
    gguf: &GgufFile,
    wref: &WeightRef,
    b: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    dequant_scratch: &mut Vec<f32>,
) -> bool {
    debug_assert_eq!(wref.m, m, "try_blas_prefill_gemm: weight m mismatch");
    debug_assert_eq!(wref.k, k, "try_blas_prefill_gemm: weight k mismatch");
    let data = weight_data(gguf, wref);
    if dequant_scratch.len() < m * k {
        dequant_scratch.resize(m * k, 0.0);
    }
    let dequant = &mut dequant_scratch[..m * k];
    match wref.dtype {
        DType::Q4_0 => crate::quant::dequantize_q4_0_matrix(data, m, k, dequant),
        DType::Q8_0 => crate::quant::dequantize_q8_0_matrix(data, m, k, dequant),
        DType::Q4KM => crate::quant::dequantize_q4_k_m_matrix(data, m, k, dequant),
        DType::Q6K => crate::quant::dequantize_q6_k_matrix(data, m, k, dequant),
        _ => return false,
    }
    crate::backend::blas::sgemm_rowmajor_nn(m, n, k, dequant, b, out);
    true
}

/// Batched GEMM with pre-quantized Q8_0 input columns (the no-BLAS fallback).
/// Dispatches on the weight dtype to whichever int8 kernel this target has —
/// aarch64 NEON or x86_64 AVX-512 VNNI; returns `true` when a kernel ran.
/// A `false` return means nothing was computed and the caller's output buffer
/// still holds whatever was in it, so callers must gate rather than ignore it.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(feature = "blas")
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_preq(
    gguf: &GgufFile,
    wref: &WeightRef,
    b_scales: &[f32],
    b_quants: &[i8],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    debug_assert_eq!(wref.m, m, "gemm_preq: weight m mismatch");
    debug_assert_eq!(wref.k, k, "gemm_preq: weight k mismatch");
    // The NEON kernels assume Q8_0 block alignment (k a multiple of 32) and read
    // exactly n*k quants / n*(k/32) scales; enforce both at the wrapper boundary
    // so misuse (a non-32 k or an under-sized scratch) fails loudly in debug
    // rather than silently truncating (k/32) or producing wrong results.
    debug_assert_eq!(k % 32, 0, "gemm_preq: k ({k}) must be a multiple of 32");
    debug_assert!(
        b_scales.len() >= n * (k / 32) && b_quants.len() >= n * k,
        "gemm_preq: input scratch too small (need {} scales / {} quants for n={n}, k={k})",
        n * (k / 32),
        n * k,
    );
    let data = weight_data(gguf, wref);
    // The input scratch may be sized to the largest GEMM k-dim and shared across
    // projections with differing k; the NEON kernels require exactly `n*k`
    // quants / `n*(k/32)` scales, so slice to this GEMM's k. The buffer is
    // always ≥ the needed length (a no-op for exactly-sized callers), and
    // `quantize_columns` packs column j at the matching `k`-strided offset.
    let b_scales = &b_scales[..n * (k / 32)];
    let b_quants = &b_quants[..n * k];
    let ran = cpu::gemm_preq_dispatch(wref.dtype, data, b_scales, b_quants, out, m, n, k);
    if !ran {
        // Reaching here means a caller gated on `batched_gemm_supports` and got a
        // different answer than the dispatcher — i.e. the two drifted apart. That is
        // not a benign "fall back to the slow path": the callers of `gemm_preq`
        // **ignore this return value** and reuse one output buffer across layers, so
        // an uncomputed GEMM leaves the *previous* layer's activations in `out`.
        report_uncomputed_gemm(wref.dtype, k);
    }
    ran
}

/// A batched GEMM was requested for a weight no kernel here can compute.
///
/// This must never happen — `batched_gemm_supports` gates it — so treat it as the
/// invariant break it is. It is *not* a benign "fall back to the slow path": the
/// callers of `gemm_preq` **ignore its return value**, and they reuse a single output
/// buffer across layers, so an uncomputed GEMM leaves the previous layer's activations
/// in `out` and inference produces confident garbage. Panic in debug; in release, at
/// least say so loudly rather than silently corrupting the forward pass.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(feature = "blas")
))]
fn report_uncomputed_gemm(dtype: DType, k: usize) {
    debug_assert!(
        false,
        "gemm_preq: no batched kernel ran for {dtype:?} (k={k}), but `batched_gemm_supports` \
         admitted it — the gate and the kernel table have drifted. `out` is now stale."
    );
    tracing::error!(
        "gemm_preq: no batched kernel for {dtype:?} (k={k}); the matmul was NOT computed \
         and the output buffer holds stale data"
    );
}

/// Quantize all `n` columns of a column-major `[dim × n]` matrix to Q8_0
/// (no-`blas` fallback). `col` is a scratch column of length ≥ `dim`;
/// `scales`/`quants` receive the packed `[n][dim/32]` / `[n][dim]` layout the
/// batched int8 GEMM kernels consume — the same layout on NEON and VNNI.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(feature = "blas")
))]
pub(crate) fn quantize_columns(
    mat: &[f32],
    dim: usize,
    n: usize,
    col: &mut [f32],
    scales: &mut [f32],
    quants: &mut [i8],
) {
    // Q8_0 packs 32-element blocks; `dim` must divide evenly (else the tail is
    // silently dropped by `dim / 32`). Assert alignment + scratch capacity at the
    // top so misuse is caught before the unsafe NEON quantizer runs.
    debug_assert_eq!(
        dim % 32,
        0,
        "quantize_columns: dim ({dim}) must be a multiple of 32"
    );
    debug_assert!(
        mat.len() >= dim * n
            && col.len() >= dim
            && scales.len() >= n * (dim / 32)
            && quants.len() >= n * dim,
        "quantize_columns: scratch too small for dim={dim}, n={n}",
    );
    let nb = dim / 32;

    // Fan the per-column quantization out over the **RowPool**, not rayon. Each
    // column is independent (disjoint `scales`/`quants` slices), so left serial
    // this is the Amdahl term that caps multi-core prefill — the batched GEMM
    // downstream parallelizes over rows, so a serial pre-quant does not shrink
    // per core (a measured multi-core regression on Android big.LITTLE).
    //
    // It must ride the same persistent pool the GEMM uses: rayon's fork-join
    // barrier costs a futex wake + core migration *per dispatch*, and this runs
    // once per projection, so a rayon fan-out here was measured ~2× *slower*
    // than serial on Tensor G5 (see `backend::threadpool` docs). `par_rows_n`
    // dispatches on the RowPool, where a dispatch is an atomic store.
    //
    // `par_rows_n` splits `scales` into one `nb`-wide row per column; `quants`
    // and `mat` are reached through raw pointers, each column touching a
    // disjoint `quants` span — the same disjoint-`&mut`-via-`usize` handoff the
    // K-quant GEMM uses. Below the threshold the caller's `col` scratch path
    // runs (and `dispatch_rows` itself degrades to caller-serial anyway).
    #[cfg(feature = "parallel")]
    {
        // Resolve once so the entry gate and the per-worker granularity are
        // provably the same value (the "one value, one meaning" contract).
        let min_cols = cpu::prequant_par_min_cols();
        if n >= min_cols {
            let mat_ptr = mat.as_ptr() as usize;
            let quants_ptr = quants.as_mut_ptr() as usize;
            cpu::par_rows_n(&mut scales[..n * nb], nb, min_cols, move |(j, sc)| {
                let mat = mat_ptr as *const f32;
                // SAFETY: column `j` exclusively owns `quants[j*dim .. (j+1)*dim]`
                // (columns are disjoint), and `mat` is read-only. Quantize one
                // Q8_0 block at a time out of a stack gather buffer, so there is
                // no per-worker heap scratch to thread through the pool.
                let qcol = (quants_ptr as *mut i8).wrapping_add(j * dim);
                let mut blk = [0.0f32; 32];
                for b in 0..nb {
                    for (t, bt) in blk.iter_mut().enumerate() {
                        *bt = unsafe { *mat.add((b * 32 + t) * n + j) };
                    }
                    // SAFETY: column `j` exclusively owns this 32-quant span.
                    let qs = unsafe { core::slice::from_raw_parts_mut(qcol.add(b * 32), 32) };
                    cpu::quantize_f32_to_q8_0_into(&blk, &mut sc[b..b + 1], qs);
                }
            });
            return;
        }
    }

    for j in 0..n {
        for i in 0..dim {
            col[i] = mat[i * n + j];
        }
        cpu::quantize_f32_to_q8_0_into(
            &col[..dim],
            &mut scales[j * nb..(j + 1) * nb],
            &mut quants[j * dim..(j + 1) * dim],
        );
    }
}

/// Dequantize a single row from a quantized matrix into `out`.
pub(crate) fn dequantize_row_into(
    gguf: &GgufFile,
    wref: &WeightRef,
    row_idx: usize,
    out: &mut [f32],
) {
    assert!(
        row_idx < wref.m,
        "dequantize_row: row_idx {row_idx} out of range (m={})",
        wref.m
    );
    let data = weight_data(gguf, wref);
    // `row_bytes` divides by the block size; a `k` that isn't a whole number of
    // blocks would truncate the stride and silently drop each row's tail (the
    // downstream `dequantize_*_row` only `debug_assert`s the length, so release
    // builds would dequantize garbage). Well-formed GGUF K-quant rows are always
    // a multiple of 256, so this only fires on a malformed file — fail loudly
    // rather than corrupt the row.
    let block_size = wref.dtype.block_size();
    assert_eq!(
        wref.k % block_size,
        0,
        "dequantize_row: k ({}) is not a multiple of the {:?} block size ({block_size})",
        wref.k,
        wref.dtype,
    );
    let row_bytes = wref.k / block_size * wref.dtype.block_bytes();
    let row_start = row_idx * row_bytes;
    let row_data = &data[row_start..row_start + row_bytes];

    match wref.dtype {
        DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, out),
        DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, out),
        DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, out),
        DType::Q4KM => crate::quant::dequantize_q4_k_m_row(row_data, out),
        DType::Q5KM => crate::quant::dequantize_q5_k_row(row_data, out),
        DType::F32 => {
            let floats: &[f32] = bytemuck::cast_slice(row_data);
            out.copy_from_slice(floats);
        }
        _ => panic!("unsupported embedding dtype: {:?}", wref.dtype),
    }
}

/// Dequantize a single row to an owned `Vec<f32>` (embedding lookup).
pub(crate) fn dequantize_row(gguf: &GgufFile, wref: &WeightRef, row_idx: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; wref.k];
    dequantize_row_into(gguf, wref, row_idx, &mut out);
    out
}

/// Dequantize a full `[m, k]` weight matrix to an owned row-major `Vec<f32>`.
/// Used by the GPU loaders to upload non-quantized-kernel dtypes as F32.
/// The metal loader references weights via mmap offsets and never dequantizes,
/// so this is dead under `metal` alone (live under `gpu`).
#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn dequantize_weight(gguf: &GgufFile, wref: &WeightRef) -> Vec<f32> {
    let mut out = vec![0.0f32; wref.m * wref.k];
    for row in 0..wref.m {
        let row_out = &mut out[row * wref.k..(row + 1) * wref.k];
        dequantize_row_into(gguf, wref, row, row_out);
    }
    out
}

// ── Generic per-layer kernels ───────────────────────────────────────────────

/// Pre-resolved attention weight refs for a transformer layer.
pub(crate) struct AttnWeights<'a> {
    pub attn_q: &'a WeightRef,
    pub attn_k: &'a WeightRef,
    pub attn_v: &'a WeightRef,
    pub attn_output: &'a WeightRef,
}

/// Optional per-arch knobs for the attention helper.
///
/// - `qkv_bias`: Q/K/V bias vectors added right after each projection GEMV.
///   Present for Qwen2, `None` for Qwen3.
/// - `qk_norm`: per-head RMSNorm weights for Q and K, applied BEFORE RoPE
///   (head_dim each). Present for Qwen3, `None` for Qwen2.
pub(crate) struct AttnExtras<'a> {
    pub qkv_bias: Option<(&'a [f32], &'a [f32], &'a [f32])>,
    pub qk_norm: Option<(&'a [f32], &'a [f32])>,
}

/// Static per-layer dimensions for the attention helper.
#[derive(Clone, Copy)]
pub(crate) struct AttnDims<'a> {
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// RoPE pair layout: `Neox` for Qwen2/Qwen3, `Norm` for LLaMA/Mistral/Granite.
    pub rope_type: cpu::RopeType,
    /// Softmax scale override. `None` ⇒ `1/sqrt(head_dim)` (the default). Granite
    /// 3.x sets this to its `attention.scale` multiplier.
    pub attn_scale: Option<f32>,
    /// Llama-3 RoPE frequency-scaling factors (`rope_freqs.weight`, `head_dim/2`),
    /// applied only on the NORM path; `None` ⇒ plain RoPE.
    pub rope_freqs: Option<&'a [f32]>,
}

/// Run one attention block for a single token. Writes the post-output-projection
/// result into `state.scratch.out[..hidden_size]`. The pre-normed hidden state
/// `hidden` is expected to already be RMSNorm'd by the caller (and, on aarch64,
/// pre-quantized into `state.scratch.q8_*`). KV append + attention go through
/// the f32 `LayerState::Attention` cache exactly as LFM2's f32 path does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_attn_block(
    gguf: &GgufFile,
    layer: usize,
    weights: &AttnWeights,
    extras: &AttnExtras,
    dims: AttnDims<'_>,
    hidden: &[f32],
    pos: usize,
    state: &mut InferenceState,
) {
    let head_dim = dims.head_dim;
    let n_heads = dims.n_heads;
    let n_kv_heads = dims.n_kv_heads;
    let hidden_size = dims.hidden_size;
    let kv_dim = n_kv_heads * head_dim;
    // Q projection width = attention output width. Equals hidden_size for most
    // models, but Qwen3 decouples head_dim so q_dim can exceed it.
    let q_dim = n_heads * head_dim;

    // Cloned once (cheap Arc bump) so the base-weight scratch buffers can stay
    // mutably borrowed while we read the adapter (a disjoint field).
    let lora = state.lora.clone();

    let q = &mut state.scratch.q[..q_dim];
    let k = &mut state.scratch.k[..kv_dim];
    let v = &mut state.scratch.v[..kv_dim];

    // Q, K, V projections. On aarch64 the hidden state was pre-quantized to
    // Q8_0 at the layer level, so the integer dot-product path is used.
    #[cfg(target_arch = "aarch64")]
    {
        gemv_preq(
            gguf,
            weights.attn_q,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            q,
        );
        gemv_preq(
            gguf,
            weights.attn_k,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            k,
        );
        gemv_preq(
            gguf,
            weights.attn_v,
            hidden,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            v,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemv(gguf, weights.attn_q, hidden, q);
        gemv(gguf, weights.attn_k, hidden, k);
        gemv(gguf, weights.attn_v, hidden, v);
    }

    // Qwen2 bias: applied right after each Q/K/V projection.
    if let Some((q_bias, k_bias, v_bias)) = extras.qkv_bias {
        cpu::add_inplace(q, q_bias);
        cpu::add_inplace(k, k_bias);
        cpu::add_inplace(v, v_bias);
    }

    // LoRA: add `scale·B·(A·hidden)` to each of Q/K/V (input is the normed
    // hidden; the delta is applied before RoPE, matching the base projection).
    if let Some(lora) = &lora {
        crate::lora::apply_attn_qkv(lora, layer, hidden, q, k, v, &mut state.scratch.lora_tmp);
    }

    // Qwen3 per-head QK norm: RMSNorm each head slice with shared weights,
    // applied BEFORE RoPE (mirrors LFM2's mandatory QK-norm).
    if let Some((q_norm, k_norm)) = extras.qk_norm {
        for h in 0..n_heads {
            cpu::rmsnorm(
                &mut q[h * head_dim..(h + 1) * head_dim],
                q_norm,
                dims.rms_norm_eps,
            );
        }
        for h in 0..n_kv_heads {
            cpu::rmsnorm(
                &mut k[h * head_dim..(h + 1) * head_dim],
                k_norm,
                dims.rms_norm_eps,
            );
        }
    }

    // RoPE — layout per arch (NEOX split-halves for Qwen2/Qwen3, NORM
    // interleaved for LLaMA/Mistral/Granite).
    match dims.rope_type {
        cpu::RopeType::Neox => cpu::rope(q, k, pos, n_heads, n_kv_heads, head_dim, dims.rope_theta),
        cpu::RopeType::Norm => cpu::rope_norm(
            q,
            k,
            pos,
            n_heads,
            n_kv_heads,
            head_dim,
            dims.rope_theta,
            dims.rope_freqs,
        ),
    }

    // Append K, V to the f32 cache.
    if let LayerState::Attention {
        key_cache,
        value_cache,
        ..
    } = &mut state.layers[layer]
    {
        key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
        value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
    }

    // GQA: grouped query attention over the full KV cache.
    let group_size = n_heads / n_kv_heads;
    // Default softmax scale 1/sqrt(head_dim); Granite overrides via attn_scale.
    let scale = dims
        .attn_scale
        .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    {
        let (k_cache, v_cache) = match &state.layers[layer] {
            LayerState::Attention {
                key_cache,
                value_cache,
                ..
            } => (key_cache.as_slice(), value_cache.as_slice()),
            _ => panic!("expected Attention state for layer {layer}"),
        };
        let seq_len = k_cache.len() / kv_dim;
        let attn_out = &mut state.scratch.attn_out[..q_dim];
        let q = &state.scratch.q[..q_dim];
        let scores = &mut state.scratch.scores;
        scores.resize(seq_len, 0.0);
        for h in 0..n_heads {
            let kv_h = h / group_size;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let kv_h_offset = kv_h * head_dim;
            cpu::attn_scores(
                q_head,
                k_cache,
                scores,
                kv_dim,
                kv_h_offset,
                head_dim,
                scale,
                seq_len,
            );
            cpu::softmax_inplace(scores);
            cpu::attn_values(
                scores,
                v_cache,
                &mut attn_out[h * head_dim..(h + 1) * head_dim],
                kv_dim,
                kv_h_offset,
                head_dim,
                seq_len,
            );
        }
    }

    // Output projection: attn_out (n_heads * head_dim) → out (hidden_size).
    let out = &mut state.scratch.out[..hidden_size];
    gemv(
        gguf,
        weights.attn_output,
        &state.scratch.attn_out[..q_dim],
        out,
    );
    // LoRA on the output projection (input is the attention output).
    if let Some(lora) = &lora
        && let Some(t) = lora.get(layer, crate::lora::LoraTarget::AttnOutput)
    {
        crate::lora::apply_decode(
            t,
            &state.scratch.attn_out[..q_dim],
            out,
            &mut state.scratch.lora_tmp,
        );
    }
}

/// Pre-resolved FFN weight refs for a transformer layer.
pub(crate) struct FfnWeights<'a> {
    pub ffn_gate: &'a WeightRef,
    pub ffn_up: &'a WeightRef,
    pub ffn_down: &'a WeightRef,
}

/// Run one SwiGLU FFN block for a single token: `ffn_input` is the already
/// RMSNorm'd (and, on aarch64, pre-quantized) hidden state. Writes the result
/// into `state.scratch.out[..hidden_size]`. Identical to LFM2's FFN.
pub(crate) fn forward_ffn_block(
    gguf: &GgufFile,
    layer: usize,
    weights: &FfnWeights,
    hidden_size: usize,
    intermediate_size: usize,
    ffn_input: &[f32],
    state: &mut InferenceState,
) {
    let lora = state.lora.clone();
    #[cfg(target_arch = "aarch64")]
    {
        gemv_preq(
            gguf,
            weights.ffn_gate,
            ffn_input,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.gate[..intermediate_size],
        );
        gemv_preq(
            gguf,
            weights.ffn_up,
            ffn_input,
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.up[..intermediate_size],
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemv(
            gguf,
            weights.ffn_gate,
            ffn_input,
            &mut state.scratch.gate[..intermediate_size],
        );
        gemv(
            gguf,
            weights.ffn_up,
            ffn_input,
            &mut state.scratch.up[..intermediate_size],
        );
    }

    // LoRA on gate/up — BEFORE the SwiGLU mul (which reads both), input is the
    // normed FFN input.
    if let Some(lora) = &lora {
        if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnGate) {
            crate::lora::apply_decode(
                t,
                ffn_input,
                &mut state.scratch.gate[..intermediate_size],
                &mut state.scratch.lora_tmp,
            );
        }
        if let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnUp) {
            crate::lora::apply_decode(
                t,
                ffn_input,
                &mut state.scratch.up[..intermediate_size],
                &mut state.scratch.lora_tmp,
            );
        }
    }

    cpu::silu_mul_inplace(
        &mut state.scratch.gate[..intermediate_size],
        &state.scratch.up[..intermediate_size],
    );

    #[cfg(target_arch = "aarch64")]
    {
        let nb = intermediate_size / 32;
        state.scratch.q8_scales.resize(nb, 0.0);
        state.scratch.q8_quants.resize(intermediate_size, 0);
        unsafe {
            crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                &state.scratch.gate[..intermediate_size],
                &mut state.scratch.q8_scales,
                &mut state.scratch.q8_quants,
            );
        }
        gemv_preq(
            gguf,
            weights.ffn_down,
            &state.scratch.gate[..intermediate_size],
            &state.scratch.q8_scales,
            &state.scratch.q8_quants,
            &mut state.scratch.out[..hidden_size],
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    gemv(
        gguf,
        weights.ffn_down,
        &state.scratch.gate[..intermediate_size],
        &mut state.scratch.out[..hidden_size],
    );

    // LoRA on the down projection (input is the SwiGLU product in `gate`).
    if let Some(lora) = &lora
        && let Some(t) = lora.get(layer, crate::lora::LoraTarget::FfnDown)
    {
        crate::lora::apply_decode(
            t,
            &state.scratch.gate[..intermediate_size],
            &mut state.scratch.out[..hidden_size],
            &mut state.scratch.lora_tmp,
        );
    }
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    not(feature = "blas"),
    feature = "parallel"
))]
mod tests {
    use super::*;

    /// Parallel `quantize_columns` must produce byte-identical output to the
    /// serial per-column reference. There is no cross-column reduction, so the
    /// only way the fan-out can differ is a wiring bug (a column written to the
    /// wrong `scales`/`quants` slice); this asserts it away at a column count
    /// above `prequant_par_min_cols()`, so the parallel branch is the one exercised.
    #[test]
    fn quantize_columns_parallel_matches_serial() {
        let dim = 256usize;
        let n = 64usize; // ≥ prequant_par_min_cols() → the parallel branch runs.
        let nb = dim / 32;

        // Deterministic column-major [dim × n] activation matrix.
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut lcg = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let mat: Vec<f32> = (0..dim * n).map(|_| lcg()).collect();

        let mut col = vec![0.0f32; dim];
        let mut scales = vec![0.0f32; n * nb];
        let mut quants = vec![0i8; n * dim];
        quantize_columns(&mat, dim, n, &mut col, &mut scales, &mut quants);

        // Serial reference: gather each column and quantize it in isolation.
        let mut ref_scales = vec![0.0f32; n * nb];
        let mut ref_quants = vec![0i8; n * dim];
        let mut rc = vec![0.0f32; dim];
        for j in 0..n {
            for (i, ci) in rc.iter_mut().enumerate() {
                *ci = mat[i * n + j];
            }
            unsafe {
                crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                    &rc,
                    &mut ref_scales[j * nb..(j + 1) * nb],
                    &mut ref_quants[j * dim..(j + 1) * dim],
                );
            }
        }

        assert_eq!(
            quants, ref_quants,
            "parallel quantize_columns quants differ"
        );
        assert_eq!(
            scales, ref_scales,
            "parallel quantize_columns scales differ"
        );
    }
}

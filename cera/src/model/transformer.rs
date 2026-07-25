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

use anyhow::{Context, Result, ensure};

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

/// The 8-row-interleaved payload of a weight repacked at load for the prefill
/// GEMM. One variant per repackable dtype; each holds the packed nibbles plus
/// that dtype's baked scales (Q4_0: one f32 row scale per block; Q4_K: per-row
/// `d·sc` and `dmin·mn` products). Owned (not a view into the mmap) because the
/// layout differs from GGUF's, and kept *alongside* the mmap weights — prefill
/// only, decode keeps the standard mmap layout — so it costs roughly one extra
/// weight-sized copy for each repacked weight.
#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
#[derive(Clone)]
pub(crate) enum Repacked {
    Q40 {
        packed: Vec<u8>,
        scales: Vec<f32>,
    },
    Q4K {
        packed: Vec<u8>,
        dsc: Vec<f32>,
        dmn: Vec<f32>,
    },
}

/// A weight's `m x k` body repacked into the layout the prefill GEMM
/// (`cpu::gemm_preq_repacked_*_dispatch`) consumes.
///
/// Gated to the one config that reads it — the x86 no-BLAS prefill path — so it
/// does not read as dead code where `gemm_preq`'s repacked branch is compiled out.
#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
#[derive(Clone)]
pub(crate) struct RepackedWeight {
    pub kind: Repacked,
    pub m: usize,
    pub k: usize,
}

#[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
impl std::fmt::Debug for RepackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the buffers — they can be hundreds of MiB.
        let (tag, packed_len) = match &self.kind {
            Repacked::Q40 { packed, .. } => ("Q4_0", packed.len()),
            Repacked::Q4K { packed, .. } => ("Q4_K", packed.len()),
        };
        f.debug_struct("RepackedWeight")
            .field("kind", &tag)
            .field("m", &self.m)
            .field("k", &self.k)
            .field("packed_len", &packed_len)
            .finish()
    }
}

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
    /// Set by [`WeightRef::with_repack`] for Q4_0 / Q4_K projection weights on
    /// hosts with the x86 int8 kernels. `None` when unset (other dtypes, ragged
    /// row counts, or weights that never hit the batched GEMM). The field exists
    /// only on the target/feature combo whose `gemm_preq` reads it.
    #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
    pub repacked: Option<std::sync::Arc<RepackedWeight>>,
}

impl WeightRef {
    /// Repack this weight for the prefill GEMM if it qualifies, returning the
    /// (possibly augmented) ref. Call at projection-weight resolution sites; do
    /// **not** call for embeddings / the output projection, whose prefill GEMM
    /// runs at `n = 1` (last column only), where the repacked layout gives
    /// nothing and the extra copy is pure waste.
    #[allow(unused_variables, unused_mut)]
    pub(crate) fn with_repack(mut self, gguf: &GgufFile) -> Self {
        #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
        {
            let kind = if self.dtype == DType::Q4_0 && cpu::q4_0_repack_supported(self.m, self.k) {
                let (packed, scales) =
                    cpu::repack_q4_0_8x8(weight_data(gguf, &self), self.m, self.k);
                Some(Repacked::Q40 { packed, scales })
            } else if self.dtype == DType::Q4KM && cpu::q4_k_repack_supported(self.m, self.k) {
                let (packed, dsc, dmn) =
                    cpu::repack_q4_k_8x8(weight_data(gguf, &self), self.m, self.k);
                Some(Repacked::Q4K { packed, dsc, dmn })
            } else {
                None
            };
            if let Some(kind) = kind {
                self.repacked = Some(std::sync::Arc::new(RepackedWeight {
                    kind,
                    m: self.m,
                    k: self.k,
                }));
            }
        }
        self
    }
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

    // A tensor whose GGML type cannot be mapped is recorded as F32 with
    // `size_bytes == 0` so `inspect` can still list it (see `gguf.rs`). Catch
    // that here rather than handing a zero-length slice to a kernel that will
    // index it and panic — the caller gets the actual type name instead.
    // Reports the numeric id alongside the name, matching `GgufFile::tensor_range`:
    // `ggml_type_name` returns "???" for an id it does not know, so the name alone
    // would say nothing at all about a type newer than this build.
    ensure!(
        info.size_bytes > 0,
        "tensor {name} has unsupported GGML type {} ({}) — cera cannot run this file",
        info.ggml_type_id,
        crate::gguf::ggml_type_name(info.ggml_type_id)
    );

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
        #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
        repacked: None,
    })
}

/// Get the raw bytes for a pre-resolved weight.
#[inline]
pub(crate) fn weight_data<'a>(gguf: &'a GgufFile, wref: &WeightRef) -> &'a [u8] {
    &gguf.mmap_data()[wref.start..wref.start + wref.size]
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    /// The K-quant super-block guard must actually decline.
    ///
    /// `batched_gemm_supports` had no test coverage at all: deleting
    /// `&& k % 256 == 0` passed the entire suite, because the only thing that
    /// exercised this function was the `#[ignore]`d real-model parity suite.
    /// With the guard gone, a K-quant layer whose `k` is not a super-block
    /// multiple reaches `gemm_preq_dispatch` and hits its block-alignment
    /// assert — a release panic on a real model, where the intended behaviour
    /// is a clean fall back to per-token GEMV.
    #[test]
    fn k_quant_batched_gemm_requires_whole_superblocks() {
        for k in [32usize, 128, 255, 257, 384] {
            assert!(
                !batched_gemm_supports(DType::Q4KM, k),
                "Q4KM admitted k={k}, which is not a multiple of 256"
            );
            assert!(
                !batched_gemm_supports(DType::Q6K, k),
                "Q6K admitted k={k}, which is not a multiple of 256"
            );
        }
        // The positive direction, so the test cannot pass by the gate being
        // stuck closed. Aligned `k` is admitted exactly when a kernel exists.
        let expect = k_quant_gemm_available();
        for k in [256usize, 512, 2048] {
            assert_eq!(
                batched_gemm_supports(DType::Q4KM, k),
                expect,
                "Q4KM at aligned k={k} disagrees with k_quant_gemm_available()"
            );
        }
    }

    /// Dtypes with no int8 kernel must decline whatever the host.
    #[test]
    fn unsupported_dtypes_are_never_batched() {
        for dtype in [DType::Q5KM, DType::F16, DType::F32] {
            assert!(
                !batched_gemm_supports(dtype, 256),
                "{dtype:?} was admitted to the batched path with no kernel to run it"
            );
        }
    }

    /// Q4_1 is batched exactly when `k_quant_gemm_available()` holds — the shared
    /// predicate is true for the int8 dotprod/AVX2 kernels *and* the BLAS dequant+SGEMM
    /// path, which is why the test asserts against that predicate rather than naming one
    /// backend. Unlike the K-quants there is no 256-alignment requirement, since Q4_1
    /// blocks are 32 wide. `k = 96` (not a multiple of 256) proves the distinction: a
    /// K-quant would decline there, Q4_1 must not.
    #[test]
    fn q4_1_is_batched_when_the_int8_path_is_available() {
        let expect = k_quant_gemm_available();
        for k in [32usize, 96, 256, 2048] {
            assert_eq!(
                batched_gemm_supports(DType::Q4_1, k),
                expect,
                "Q4_1 at k={k} disagrees with k_quant_gemm_available()"
            );
        }
    }
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
/// This is the single source of truth: both `lfm2.rs` and `llama.rs` gate on it,
/// so widening it widens every caller at once. `llama.rs` used to keep a narrower
/// Q4_0/Q8_0 allowlist of its own, which is why its gate now reads as a plain
/// call — that duplicate list is gone, not merely satisfied.
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
/// `#[doc(hidden)] pub` so `int8_gemm_gate.rs` can assert on the gate itself,
/// not just on the predicate it consults. That binary forces
/// `CERA_CPU_TIER=scalar` in a dedicated process, which a unit test cannot do —
/// and asserting only on `cpu::int8_gemm_available()` left every arm of this
/// function replaceable with `true` without a single test failing. Not part of
/// the supported API.
#[doc(hidden)]
pub fn batched_gemm_supports(dtype: DType, k: usize) -> bool {
    match dtype {
        // Not unconditional: x86 needs avx2+fma at minimum, so a Scalar-tier
        // host must stay on the per-token GEMV fallback. This used to require
        // VNNI; the AVX2 kernels (`dpbusd` emulated with `maddubs`) lowered the
        // bar to every tier from `Avx2` up, which is why the predicate is a tier
        // comparison and not a VNNI check. Under `blas` the question is moot —
        // that path dequantizes and SGEMMs.
        DType::Q4_0 | DType::Q8_0 => {
            cfg!(feature = "blas") || crate::backend::cpu::int8_gemm_available()
        }
        // Q4_1's int8 GEMM reuses the K-quants' dotprod col-sum machinery (its `m`
        // term needs `Σ(activation)` exactly as their `dmin` term does), so it shares
        // their availability predicate — dotprod on aarch64, AVX2-int8 on x86, or the
        // dequant+SGEMM BLAS path. Unlike the K-quants there is no 256-alignment
        // requirement: Q4_1 blocks are 32 wide.
        DType::Q4_1 => k_quant_gemm_available(),
        DType::Q4KM | DType::Q6K => k_quant_gemm_available() && k.is_multiple_of(256),
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
/// machine cannot reproduce that. It *is* called on x86_64, where it now answers
/// for the x86 K-quant GEMM kernels (VNNI and AVX2 alike) — so this is a lint
/// cfg, not a statement about which targets reach it.
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
    // x86: the K-quant GEMM shares its availability condition with the
    // Q4_0/Q8_0 int8 kernels. Both are emitted by the same macro and both are
    // instantiated at the VNNI and AVX2 tiers, so this needs neither VNNI nor
    // the `avx512` crate feature — just avx2+fma.
    #[cfg(all(not(feature = "blas"), target_arch = "x86_64"))]
    {
        crate::backend::cpu::int8_gemm_available()
    }
    // No BLAS, no NEON, no x86 int8: no batched K-quant path on this target.
    #[cfg(all(
        not(feature = "blas"),
        not(target_arch = "aarch64"),
        not(target_arch = "x86_64")
    ))]
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
        DType::Q4_1 => crate::quant::dequantize_q4_1_matrix(data, m, k, dequant),
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
/// aarch64 NEON, or x86_64 int8 (VNNI or the AVX2 emulation). Returns `true`
/// when a kernel ran.
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
    // Same treatment for `out`, and for a sharper reason than tidiness: the
    // kernels derive their row/strip index from `out.len()`, not from `m`, so an
    // over-long buffer walks past row `m` and reads weights out of bounds.
    //
    // One in-tree caller hands us exactly that: LFM2's short-conv `in_proj`
    // GEMM passes `m = 3*hs` into `proj_mat`, which is sized
    // `max(3*hs, hs + 2*kv_dim) * n` because it is shared with the attention
    // projection. That exceeds `m*n` whenever `kv_dim > hs`. No shipping GQA
    // config does that — kv_dim is always the smaller one — so it is latent
    // rather than live, but the fix belongs here, where every caller passes
    // through, rather than at the one site that happens to trip it.
    let out = &mut out[..m * n];

    // A repacked weight (8-row interleave, built once at load) takes the
    // dedicated prefill kernel — no per-column hsum. Only present on x86 hosts
    // with the int8 kernels, and only for weights that pass the dtype's
    // `*_repack_supported`, so this is a no-op fall-through everywhere else.
    #[cfg(all(target_arch = "x86_64", not(feature = "blas")))]
    if let Some(rp) = &wref.repacked {
        debug_assert_eq!(rp.m, m, "gemm_preq: repacked m mismatch");
        debug_assert_eq!(rp.k, k, "gemm_preq: repacked k mismatch");
        let ran = match &rp.kind {
            Repacked::Q40 { packed, scales } => cpu::gemm_preq_repacked_q4_0_dispatch(
                packed, scales, b_scales, b_quants, out, m, n, k,
            ),
            Repacked::Q4K { packed, dsc, dmn } => cpu::gemm_preq_repacked_q4_k_dispatch(
                packed, dsc, dmn, b_scales, b_quants, out, m, n, k,
            ),
        };
        if !ran {
            report_uncomputed_gemm(wref.dtype, k);
        }
        return ran;
    }

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
/// batched int8 GEMM kernels consume — the same layout on NEON, VNNI, and AVX2.
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
        DType::Q4_1 => crate::quant::dequantize_q4_1_row(row_data, out),
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

// ── Decode-time GQA attention ───────────────────────────────────────────────

/// The KV cache one decode attention pass reads. A layer stores exactly one
/// representation, so this is an enum rather than both pairs plus a `use_f16`
/// discriminant — the "only one of these is populated" invariant is then
/// unrepresentable instead of a promise a caller has to keep.
pub(crate) enum KvView<'a> {
    F32 {
        k: &'a [f32],
        v: &'a [f32],
    },
    /// Widened to f32 on read by the `*_f16` kernels.
    F16 {
        k: &'a [u16],
        v: &'a [u16],
    },
}

/// Shapes for one decode attention pass (a single query position attending over
/// `seq_len` cached positions).
///
/// `n_heads` must be a positive multiple of `n_kv_heads` (the GQA invariant).
/// Both model families enforce it when they parse the GGUF — `LlamaConfig` and
/// `Lfm2Config` each `ensure!` it at load — so this is a contract for future
/// constructors, not a runtime risk on the shipped paths. It matters because
/// `group_size` is a truncating divide: a non-multiple would place the last
/// heads' `kv_h_offset` past the end of a KV row, which the attention kernels
/// would read straight through in release builds.
pub(crate) struct DecodeAttnDims {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub seq_len: usize,
}

impl DecodeAttnDims {
    /// Query heads per KV head (GQA fan-in).
    #[inline]
    fn group_size(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Row stride of the KV cache.
    #[inline]
    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// Score-MACs (`n_heads * seq_len * head_dim`) below which the head loop runs on
/// the calling thread, so a degenerate shape (one head, a cache one position
/// deep) doesn't pay for a dispatch it cannot fill. Override with
/// `CERA_DECODE_ATTN_PAR_MIN_WORK`; `env_usize` keeps only values `>= 1`, so use
/// `=1` (not `=0`) to force the pool arm — `0` is rejected and leaves the
/// default in place.
///
/// This is a floor, not a measured crossover — a shallow-depth sweep on a
/// 16-core host (forcing each arm with the env override) found the pool arm
/// ahead at *every* depth tried, on both a 9-head 135M model and a 32-head 1B:
///
/// ```text
///   prompt depth    16     32     64    128    256
///   SmolLM-135M   +10.7%  +5.4%  +6.7% +15.6% +24.4%
///   Llama-3.2-1B   +3.2%  +3.8%  +5.2%  +6.0%  +8.9%
/// ```
///
/// The lowest-work run there (SmolLM at depth 16) starts at 9216 score-MACs and
/// still wins, so the gate sits just under that. Everything at or above it was
/// measured faster on the pool; below it is unmeasured territory where the whole
/// pass costs microseconds either way.
const DECODE_ATTN_PAR_MIN_WORK_DEFAULT: usize = 8_192;

fn decode_attn_par_min_work() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        crate::backend::cpu_features::env_usize("CERA_DECODE_ATTN_PAR_MIN_WORK")
            .unwrap_or(DECODE_ATTN_PAR_MIN_WORK_DEFAULT)
    })
}

/// One query head: scores over the cache, softmax, value accumulation.
fn decode_attn_head(
    q: &[f32],
    kv: &KvView<'_>,
    d: &DecodeAttnDims,
    h: usize,
    scores: &mut [f32],
    head_out: &mut [f32],
) {
    let q_head = &q[h * d.head_dim..(h + 1) * d.head_dim];
    let kv_h_offset = (h / d.group_size()) * d.head_dim;
    match kv {
        KvView::F16 { k, v } => {
            cpu::attn_scores_f16(
                q_head,
                k,
                scores,
                d.kv_dim(),
                kv_h_offset,
                d.head_dim,
                d.scale,
                d.seq_len,
            );
            cpu::softmax_inplace(scores);
            cpu::attn_values_f16(
                scores,
                v,
                head_out,
                d.kv_dim(),
                kv_h_offset,
                d.head_dim,
                d.seq_len,
            );
        }
        KvView::F32 { k, v } => {
            cpu::attn_scores(
                q_head,
                k,
                scores,
                d.kv_dim(),
                kv_h_offset,
                d.head_dim,
                d.scale,
                d.seq_len,
            );
            cpu::softmax_inplace(scores);
            cpu::attn_values(
                scores,
                v,
                head_out,
                d.kv_dim(),
                kv_h_offset,
                d.head_dim,
                d.seq_len,
            );
        }
    }
}

/// Decode-time grouped-query attention over the whole KV cache, writing
/// `attn_out[q_dim]`. Shared by the dense transformers and LFM2's non-TurboQuant
/// path — the two head loops were identical.
///
/// The heads run on the decode pool when all three of: more than one head, work
/// at or above [`decode_attn_par_min_work`] (the default constant, unless
/// `CERA_DECODE_ATTN_PAR_MIN_WORK` moves it), and a decode pool wider than one
/// worker. Otherwise the loop stays on the calling thread.
/// `scratch` is then laid out as `n_heads` rows of `seq_len + head_dim`: each
/// head's own score buffer followed by its own output. Fusing the two into one
/// arena is what makes the loop parallelizable at all — the serial version
/// reuses a single score buffer across heads, which becomes a write-write race
/// the moment two heads run at once, and the dispatch API hands each row exactly
/// one mutable slice. The tail copy back into `attn_out` is `q_dim` floats,
/// negligible against the `n_heads * seq_len * head_dim` reductions above it.
///
/// That layout costs memory: `scratch` grows from `seq_len` floats to
/// `n_heads * (seq_len + head_dim)`, an `n_heads`-fold jump — ~4 MB at 32k
/// context on a 32-head model, against ~128 KB for the serial buffer. It is one
/// per-session allocation, not per-layer, and each worker touches only its own
/// row, so locality is unaffected; but on a memory-budgeted target that factor
/// is the price of the fan-out.
///
/// Bit-identical to the serial loop either way: heads are independent, so which
/// worker runs which head cannot change a result.
pub(crate) fn decode_attention(
    q: &[f32],
    kv: &KvView<'_>,
    d: &DecodeAttnDims,
    attn_out: &mut [f32],
    scratch: &mut Vec<f32>,
) {
    debug_assert!(
        d.n_kv_heads > 0 && d.n_heads.is_multiple_of(d.n_kv_heads),
        "GQA invariant: n_heads ({}) must be a positive multiple of n_kv_heads ({})",
        d.n_heads,
        d.n_kv_heads
    );
    let work = d
        .n_heads
        .saturating_mul(d.seq_len)
        .saturating_mul(d.head_dim);
    // Build the fused arena only when a dispatch can actually spread the heads.
    // With one usable worker — no `parallel` feature, a single-core host,
    // `CERA_DECODE_THREADS=1`, a pool degraded by a failed spawn — the arena and
    // the gather below are pure overhead for a loop that runs on this thread
    // regardless.
    let fan_out =
        d.n_heads > 1 && work >= decode_attn_par_min_work() && cpu::decode_par_threads() > 1;
    if !fan_out {
        scratch.resize(d.seq_len, 0.0);
        for h in 0..d.n_heads {
            let head_out = &mut attn_out[h * d.head_dim..(h + 1) * d.head_dim];
            decode_attn_head(q, kv, d, h, scratch.as_mut_slice(), head_out);
        }
        return;
    }

    let stride = d.seq_len.saturating_add(d.head_dim);
    // Saturating, not wrapping: a wrap here would silently hand out a *short*
    // arena that the row dispatch and the gather below would then index as if
    // it were `n_heads * stride` long. Saturating turns the same absurd shape
    // into a failed allocation instead. (No real config comes close — this is
    // the cheap way to keep a garbage `DecodeAttnDims` from becoming UB.)
    //
    // Not cleared: every element of every row is fully written below
    // (`attn_scores*` fills `[..seq_len]`, `attn_values*` fills the head output),
    // so `resize` only has to zero the bytes it grows by.
    scratch.resize(d.n_heads.saturating_mul(stride), 0.0);
    // One head per row *and* per steal unit: there are only `n_heads` of them and
    // each is heavy, so the default steal floor would hand them all to a couple
    // of workers.
    cpu::par_rows_n_chunked_decode(scratch, stride, 1, 1, |(h, row)| {
        let (scores, head_out) = row.split_at_mut(d.seq_len);
        decode_attn_head(q, kv, d, h, scores, head_out);
    });
    for h in 0..d.n_heads {
        let src = h * stride + d.seq_len;
        attn_out[h * d.head_dim..(h + 1) * d.head_dim]
            .copy_from_slice(&scratch[src..src + d.head_dim]);
    }
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

    // Append K, V to the cache (f16 or f32). `kv_f16` is read before the
    // mutable layer borrow; the f32 `scratch` fields are a disjoint borrow.
    let use_f16 = state.kv_f16;
    if let LayerState::Attention {
        key_cache,
        value_cache,
        key_cache_f16,
        value_cache_f16,
        ..
    } = &mut state.layers[layer]
    {
        if use_f16 {
            key_cache_f16.extend(
                state.scratch.k[..kv_dim]
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits()),
            );
            value_cache_f16.extend(
                state.scratch.v[..kv_dim]
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits()),
            );
        } else {
            key_cache.extend_from_slice(&state.scratch.k[..kv_dim]);
            value_cache.extend_from_slice(&state.scratch.v[..kv_dim]);
        }
    }

    // GQA: grouped query attention over the full KV cache. The head→KV-head
    // mapping is derived inside `decode_attention` from `n_kv_heads`.
    // Default softmax scale 1/sqrt(head_dim); Granite overrides via attn_scale.
    let scale = dims
        .attn_scale
        .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
    {
        // Bind both representations; only the active one is non-empty. The
        // `use_f16` choice is made once below, when building the `KvView` — the
        // head loop itself no longer carries the discriminant.
        let (k_cache, v_cache, k_cache_f16, v_cache_f16) = match &state.layers[layer] {
            LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                ..
            } => (
                key_cache.as_slice(),
                value_cache.as_slice(),
                key_cache_f16.as_slice(),
                value_cache_f16.as_slice(),
            ),
            _ => panic!("expected Attention state for layer {layer}"),
        };
        let seq_len = if use_f16 {
            k_cache_f16.len() / kv_dim
        } else {
            k_cache.len() / kv_dim
        };
        let attn_out = &mut state.scratch.attn_out[..q_dim];
        let q = &state.scratch.q[..q_dim];
        let kv = if use_f16 {
            KvView::F16 {
                k: k_cache_f16,
                v: v_cache_f16,
            }
        } else {
            KvView::F32 {
                k: k_cache,
                v: v_cache,
            }
        };
        decode_attention(
            q,
            &kv,
            &DecodeAttnDims {
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                seq_len,
            },
            attn_out,
            &mut state.scratch.scores,
        );
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

// Not gated on `parallel`: `decode_attention` compiles either way, and the
// serial branch is the *only* branch without the feature — precisely the
// configuration that most needs the coverage.
#[cfg(test)]
mod decode_attn_tests {
    use super::*;

    /// Reference: the serial head loop `decode_attention` replaced, driven
    /// through the same per-head primitive so the only thing under test is the
    /// fan-out plumbing — head→row mapping, the fused `scores | out` arena, and
    /// the gather back into `attn_out`.
    fn serial_reference(q: &[f32], kv: &KvView<'_>, d: &DecodeAttnDims) -> Vec<f32> {
        let mut out = vec![0.0f32; d.n_heads * d.head_dim];
        let mut scores = vec![0.0f32; d.seq_len];
        for h in 0..d.n_heads {
            let head_out = &mut out[h * d.head_dim..(h + 1) * d.head_dim];
            decode_attn_head(q, kv, d, h, &mut scores, head_out);
        }
        out
    }

    /// Mirror of `decode_attention`'s `fan_out` gate, so a test can assert which
    /// branch it actually exercised.
    fn would_fan_out(d: &DecodeAttnDims) -> bool {
        // Saturating, matching `decode_attention` exactly — a mirror that
        // computes the work term differently is not a mirror.
        let work = d
            .n_heads
            .saturating_mul(d.seq_len)
            .saturating_mul(d.head_dim);
        d.n_heads > 1 && work >= decode_attn_par_min_work() && cpu::decode_par_threads() > 1
    }

    /// `expect_fan_out` is what the shape should do *on a host that can fan out*
    /// — i.e. with the `parallel` feature and a multi-worker decode pool. Without
    /// those, `decode_attention` always takes the serial branch and the
    /// assertion is skipped, because there is nothing else it could do. Checking
    /// this is what stops `decode_attention_parallel_matches_serial` from
    /// quietly degrading into comparing the serial branch against itself if the
    /// gate default is raised or the shapes here drift below it.
    fn run_case(
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        expect_fan_out: bool,
    ) {
        let kv_dim = n_kv_heads * head_dim;
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        let mut lcg = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| lcg()).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|_| lcg()).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|_| lcg()).collect();
        let k_f16: Vec<u16> = k
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v_f16: Vec<u16> = v
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        for use_f16 in [false, true] {
            let kv = if use_f16 {
                KvView::F16 {
                    k: &k_f16,
                    v: &v_f16,
                }
            } else {
                KvView::F32 { k: &k, v: &v }
            };
            let d = DecodeAttnDims {
                n_heads,
                n_kv_heads,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                seq_len,
            };
            // Skipped when the gate is overridden, since the override moves the
            // very threshold this is asserting against — otherwise anyone who
            // exports `CERA_DECODE_ATTN_PAR_MIN_WORK` to tune sees the suite
            // fail in a test that is working exactly as intended.
            let gate_overridden = std::env::var_os("CERA_DECODE_ATTN_PAR_MIN_WORK").is_some();
            if cfg!(feature = "parallel") && !gate_overridden && cpu::decode_par_threads() > 1 {
                assert_eq!(
                    would_fan_out(&d),
                    expect_fan_out,
                    "case (n_heads={n_heads}, seq_len={seq_len}) took the wrong \
                     branch — this test would not be checking what it claims"
                );
            }
            let want = serial_reference(&q, &kv, &d);
            let mut got = vec![0.0f32; n_heads * head_dim];
            // Deliberately undersized: `decode_attention` must grow and re-lay-out
            // whatever it is handed, exactly as it does when depth advances.
            let mut scratch = vec![0.0f32; 1];
            decode_attention(&q, &kv, &d, &mut got, &mut scratch);
            assert_eq!(
                got, want,
                "decode_attention differs (n_heads={n_heads}, seq_len={seq_len}, use_f16={use_f16})"
            );
        }
    }

    /// Above the work gate the heads run on the decode pool; the result must be
    /// bit-identical to the serial loop (heads are independent, so scheduling
    /// cannot change a value).
    #[test]
    fn decode_attention_parallel_matches_serial() {
        // 8 * 256 * 64 = 131072 MACs ≥ DECODE_ATTN_PAR_MIN_WORK_DEFAULT.
        run_case(8, 2, 64, 256, true);
        // GQA with group_size 1 (MHA) and an odd head count, still above the gate.
        run_case(7, 7, 64, 512, true);
    }

    /// Below the gate the same call must take the serial branch and agree too —
    /// this is the branch that reuses one score buffer across heads.
    ///
    /// Each case trips a different one of the three `fan_out` conditions, so the
    /// serial path is reached the way each guard would reach it in production:
    /// under the work gate, and with a single head (a shape that would dispatch
    /// one row and gain nothing). The third guard, a one-worker decode pool, is
    /// not reachable from a test — the pool is a process-global built once from
    /// the environment — so it is covered by inspection only.
    #[test]
    fn decode_attention_serial_branch_matches_reference() {
        // 8 * 8 * 64 = 4096 MACs, under DECODE_ATTN_PAR_MIN_WORK_DEFAULT.
        run_case(8, 2, 64, 8, false);
        // n_heads == 1: over the work gate, but nothing to spread.
        run_case(1, 1, 64, 4096, false);
    }

    /// Depth grows by one position per token, so the fused arena is re-laid-out
    /// on every call and crosses the gate mid-sequence. Walking depths through
    /// the boundary catches a stale-stride or stale-contents bug that a single
    /// fixed depth would miss.
    ///
    /// Run for both cache representations: f16 reads the same arena through a
    /// different kernel pair, so a stride bug could show up in one and not the
    /// other. One `scratch` spans the whole walk, as in a real session.
    #[test]
    fn decode_attention_across_growing_depth() {
        let n_heads = 8;
        let n_kv_heads = 2;
        let head_dim = 64;
        let kv_dim = n_kv_heads * head_dim;
        let max_len = 160;
        let mut st = 0x853C_49E6_748F_EA9Bu64;
        let mut lcg = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| lcg()).collect();
        let k: Vec<f32> = (0..max_len * kv_dim).map(|_| lcg()).collect();
        let v: Vec<f32> = (0..max_len * kv_dim).map(|_| lcg()).collect();
        let k_f16: Vec<u16> = k
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let v_f16: Vec<u16> = v
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        for use_f16 in [false, true] {
            let mut scratch = Vec::new();
            for seq_len in 1..=max_len {
                let n = seq_len * kv_dim;
                let kv = if use_f16 {
                    KvView::F16 {
                        k: &k_f16[..n],
                        v: &v_f16[..n],
                    }
                } else {
                    KvView::F32 {
                        k: &k[..n],
                        v: &v[..n],
                    }
                };
                let d = DecodeAttnDims {
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                    seq_len,
                };
                let want = serial_reference(&q, &kv, &d);
                let mut got = vec![0.0f32; n_heads * head_dim];
                decode_attention(&q, &kv, &d, &mut got, &mut scratch);
                assert_eq!(
                    got, want,
                    "decode_attention differs at seq_len={seq_len} (use_f16={use_f16})"
                );
            }
        }
    }
}

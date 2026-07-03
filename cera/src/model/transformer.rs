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

/// Prefill GEMM through BLAS: dequantize `wref` into `dequant_scratch[..m*k]`,
/// then SGEMM `out[m, n] = weight[m, k] @ b[k, n]` in row-major (`b`/`out` are
/// row-major `[k|m, n]`, stride `n`). Returns `true` for the supported dtypes
/// (Q4_0 / Q8_0); callers gate on dtype upfront so the `false` arm is defensive.
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
        _ => return false,
    }
    crate::backend::blas::sgemm_rowmajor_nn(m, n, k, dequant, b, out);
    true
}

/// Batched GEMM with pre-quantized Q8_0 input columns (NEON fallback, no BLAS).
/// Dispatches on the weight dtype; returns `true` when a kernel ran.
#[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
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
    match wref.dtype {
        DType::Q4_0 => unsafe {
            crate::backend::simd::neon::gemm_q4_0_q8_0_neon(data, b_scales, b_quants, out, m, n, k);
            true
        },
        DType::Q8_0 => unsafe {
            crate::backend::simd::neon::gemm_q8_0_q8_0_neon(data, b_scales, b_quants, out, m, n, k);
            true
        },
        _ => false,
    }
}

/// Quantize all `n` columns of a column-major `[dim × n]` matrix to Q8_0 (NEON
/// fallback only). `col` is a scratch column of length ≥ `dim`; `scales`/`quants`
/// receive the packed `[n][dim/32]` / `[n][dim]` layout the NEON GEMM kernels
/// consume.
#[cfg(all(target_arch = "aarch64", not(feature = "blas")))]
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
        col.len() >= dim && scales.len() >= n * (dim / 32) && quants.len() >= n * dim,
        "quantize_columns: scratch too small for dim={dim}, n={n}",
    );
    let nb = dim / 32;
    for j in 0..n {
        for i in 0..dim {
            col[i] = mat[i * n + j];
        }
        unsafe {
            crate::backend::simd::neon::quantize_f32_to_q8_0_neon(
                &col[..dim],
                &mut scales[j * nb..(j + 1) * nb],
                &mut quants[j * dim..(j + 1) * dim],
            );
        }
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
    let row_bytes = wref.k / wref.dtype.block_size() * wref.dtype.block_bytes();
    let row_start = row_idx * row_bytes;
    let row_data = &data[row_start..row_start + row_bytes];

    match wref.dtype {
        DType::Q6K => crate::quant::dequantize_q6_k_row(row_data, out),
        DType::Q8_0 => crate::quant::dequantize_q8_0_row(row_data, out),
        DType::Q4_0 => crate::quant::dequantize_q4_0_row(row_data, out),
        DType::Q4KM => crate::quant::dequantize_q4_k_m_row(row_data, out),
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
#[cfg(any(feature = "gpu", all(feature = "metal", target_os = "macos")))]
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
    weights: &FfnWeights,
    hidden_size: usize,
    intermediate_size: usize,
    ffn_input: &[f32],
    state: &mut InferenceState,
) {
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
}

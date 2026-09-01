//! Backend-agnostic weight accessor surface for the wgpu loader.
//!
//! The wgpu transformer model (`gpu_lfm2.rs`) was originally written against
//! the concrete `Lfm2Model`. To serve the plain dense transformers
//! (Qwen2/Qwen3/LLaMA/Mistral/Granite) on the same GPU code path, the loader
//! takes a `&dyn GpuWeightSource` instead, and both `Lfm2Model` and
//! `LlamaModel` implement it.
//!
//! Everything here is host-side metadata + small F32 weights + `WeightRef`
//! handles into the GGUF mmap — no wgpu types — so the trait stays in the core
//! model layer. It is compiled under `feature = "gpu"` or Apple `feature =
//! "metal"`, because the Metal loader consumes it too.
//!
//! The trait is not all that is shared. The routed mixture-of-experts section at
//! the bottom of this file holds the expert-stack layout arithmetic and the two
//! kernel bounds that *both* GPU backends load against, for the reason stated
//! there: a wrong expert stride does not fault, so two backends deriving it
//! separately is a defect neither one's tests would name.
//!
//! The GPU *attention* block is already architecture-generic (Q/K/V GEMV →
//! optional per-head QK-norm → RoPE → GQA attention → output projection); the
//! only arch-specific inputs are exactly the accessors below: which weight
//! refs exist, the optional QK-norm / QKV-bias / untied-output tensors, the
//! RoPE layout + Llama-3 frequency factors, and (via `config().scalars`) the
//! Granite scalar multipliers.

use anyhow::Result;

pub use crate::backend::cpu::RopeType;
use crate::gguf::GgufFile;
use crate::model::ModelConfig;
use crate::model::transformer::WeightRef;

/// Read-only weight + config surface the wgpu loader needs to upload a model.
///
/// `Option`-returning accessors encode per-arch presence: an LFM2 conv layer
/// has no `attn_*` refs (returns `None`); a plain transformer layer has no
/// `conv_*` refs. Callers branch on the block type and only touch the refs
/// they expect to be `Some`.
pub trait GpuWeightSource {
    fn config(&self) -> &ModelConfig;
    fn gguf(&self) -> &GgufFile;

    // ── Small pre-dequantized F32 weights ──────────────────────────────────
    fn output_norm_weight(&self) -> &[f32];
    fn attn_norm_weight(&self, layer: usize) -> &[f32];
    fn ffn_norm_weight(&self, layer: usize) -> &[f32];
    /// Qwen3 per-head Q/K RMSNorm weights (`None` for archs without QK-norm).
    fn attn_q_norm_weight(&self, layer: usize) -> Option<&[f32]>;
    fn attn_k_norm_weight(&self, layer: usize) -> Option<&[f32]>;
    /// LFM2 depthwise conv kernel (`None` for plain transformer layers).
    fn conv_weight(&self, layer: usize) -> Option<&[f32]>;
    /// Qwen2 Q/K/V projection biases (`None` for archs without QKV bias).
    fn attn_q_bias(&self, layer: usize) -> Option<&[f32]>;
    fn attn_k_bias(&self, layer: usize) -> Option<&[f32]>;
    fn attn_v_bias(&self, layer: usize) -> Option<&[f32]>;
    /// Llama-3 RoPE frequency factors (`rope_freqs.weight`, `head_dim/2`);
    /// `None` ⇒ plain RoPE.
    fn rope_freqs(&self) -> Option<&[f32]>;

    /// Token embedding tensor metadata (`token_embd.weight`).
    fn embedding_tensor(&self) -> Result<crate::tensor::Tensor> {
        self.gguf().get_tensor("token_embd.weight")
    }

    /// Token embedding tensor raw byte slice.
    fn embedding_tensor_data(&self) -> Result<std::borrow::Cow<'_, [u8]>> {
        self.gguf()
            .tensor_data("token_embd.weight")
            .map(std::borrow::Cow::Borrowed)
    }

    // ── Raw quantized-weight access (GGUF mmap handles) ─────────────────────
    // The metal loader maps weights by absolute `wref.start` offset into its own
    // mmap buffer, so it needs neither the byte slice nor a full dequantize.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn weight_bytes(&self, wref: &WeightRef) -> std::borrow::Cow<'_, [u8]>;
    // The metal loader references weights via mmap byte offsets and never
    // dequantizes a full matrix, so this accessor is dead under `metal` alone
    // (live under `gpu`, where non-kernel dtypes are uploaded as F32).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32>;

    // ── Per-layer / global weight refs ─────────────────────────────────────
    /// Separate output projection (`output.weight`) when present; `None` ⇒ the
    /// embedding table is reused for the logit projection (tied embeddings).
    fn output_ref(&self) -> Option<&WeightRef>;
    /// The layer's dense SwiGLU projections.
    ///
    /// Fallible because a mixture-of-experts layer has no single FFN weight to
    /// return: its projections are per-expert and only selected after routing.
    /// A backend that has expert kernels asks [`Self::moe_refs`] first and only
    /// falls through to these for the dense layers; one that does not gets an
    /// error rather than an arbitrary expert, so forgetting to handle MoE fails
    /// loudly at upload instead of quietly running every token through expert 0.
    fn ffn_gate_ref(&self, layer: usize) -> Result<&WeightRef>;
    fn ffn_up_ref(&self, layer: usize) -> Result<&WeightRef>;
    fn ffn_down_ref(&self, layer: usize) -> Result<&WeightRef>;

    /// The routed expert set for `layer`, or `None` when the layer is dense.
    ///
    /// A MoE file is not uniformly MoE (`lfm2moe` runs dense leading blocks),
    /// so this is per-layer, and `None` is the answer for every layer of every
    /// dense model, which is why it defaults rather than being implemented
    /// everywhere.
    ///
    /// Consumed by both GPU loaders (`upload_moe` in `metal_lfm2.rs` and
    /// `gpu_lfm2.rs`), so unlike several accessors above it needs no conditional
    /// `dead_code` allow: whichever feature compiles this module also has a
    /// caller for it.
    fn moe_refs(&self, _layer: usize) -> Option<&crate::model::lfm2::MoeFfnRefs> {
        None
    }
    fn conv_in_proj_ref(&self, layer: usize) -> Option<&WeightRef>;
    fn conv_out_proj_ref(&self, layer: usize) -> Option<&WeightRef>;
    fn attn_q_ref(&self, layer: usize) -> Option<&WeightRef>;
    fn attn_k_ref(&self, layer: usize) -> Option<&WeightRef>;
    fn attn_v_ref(&self, layer: usize) -> Option<&WeightRef>;
    fn attn_output_ref(&self, layer: usize) -> Option<&WeightRef>;

    // ── RoPE layout + prefill capability ───────────────────────────────────
    fn rope_type(&self) -> RopeType;
    /// Whether the batched-prefill GPU path is wired for this model. LFM2 has
    /// the batched shaders; the dense transformers currently prefill via the
    /// per-token decode loop (correctness-first; batched prefill for them is a
    /// follow-up), so they return `false`. The metal backend always batches
    /// prefill, so it never queries this — dead under `metal` alone.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn supports_batched_prefill(&self) -> bool;
}

// ── Routed mixture-of-experts loading, shared by both GPU backends ──────────
//
// The two backends' `upload_moe` functions are otherwise separate on purpose:
// each speaks its own buffer API and its own error vocabulary. What lives here
// is the part that is neither, the arithmetic that turns a routed layer's
// per-expert `WeightRef`s into the single number both expert GEMV kernels
// address them with. It is shared rather than mirrored because a wrong stride
// does not fault: it reads a neighbouring expert's weights and still produces
// fluent text, so two copies drifting apart is a defect no test in either
// backend would name.

/// Largest `n_expert` the routing kernel can hold, matching `MAX_EXPERTS` in
/// `shaders/slang/moe_route.slang`, where it sizes the groupshared probability
/// scratch. Both backends check it at load so an oversized model is a named
/// error instead of a groupshared overrun.
pub(crate) const MOE_MAX_EXPERTS: u32 = 256;

/// Largest `n_expert_used` the routing kernel can hold, matching `MAX_USED` in
/// `shaders/slang/moe_route.slang`. The kernel clamps rather than overruns, but
/// a clamp would silently drop experts, so both backends reject here instead.
pub(crate) const MOE_MAX_EXPERT_USED: u32 = 16;

/// One stacked expert projection's validated layout.
///
/// `rows`/`inner` describe a *single* expert's slice; `expert_stride` is the
/// byte distance to the next one.
pub(crate) struct StackedExperts {
    // Read by the wgpu loader only. Metal takes its shapes from the
    // `MetalWeight` it uploads for expert 0 instead, so on a metal-only build
    // these are genuinely dead rather than conditionally so.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub rows: usize,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub inner: usize,
    pub expert_stride: u32,
    /// The whole stack's extent, `n_expert * expert_stride`. Carried rather than
    /// left to the caller to recompute because the checked multiply that
    /// produces it is the overflow guard described below; multiplying again at
    /// the call site would be the unchecked version of the same product. Only
    /// wgpu needs the value (Metal's kernel indexes into the whole-file mmap, so
    /// only the guard matters there), hence the same conditional allow as above.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub total_bytes: u32,
}

/// Validate one projection's per-expert refs and derive the byte stride between
/// consecutive experts.
///
/// `backend` names the caller in the Q4_0 rejection, which is the one message
/// here that is about a backend's kernels rather than about the file.
///
/// The stride is computed from the tensor's own dimensions and then checked
/// against the offsets `GgufFile::tensor_meta_expert` derived for every expert.
/// That catches a transcription error in the formula here. It is not an
/// independent confirmation that the file is evenly stacked, despite what the
/// mismatch error says: `tensor_meta_expert` derives each expert's offset as
/// `range.start + e * tensor_data_size(&[ne0, ne1], dtype)`, the same product
/// from the same shape, so evenly stacked is its definition rather than its
/// finding.
pub(crate) fn stacked_expert_layout(
    refs: &[WeightRef],
    layer: usize,
    what: &str,
    backend: &str,
) -> Result<StackedExperts> {
    use anyhow::Context;

    let first = refs
        .first()
        .with_context(|| format!("layer {layer}: {what} has no experts"))?;
    anyhow::ensure!(
        first.dtype == crate::tensor::DType::Q4_0,
        "layer {layer}: the {backend} expert GEMV kernel is Q4_0-only, {what} is {:?}",
        first.dtype,
    );
    // Checked here rather than left to the stride derivation below, which
    // truncates `k / block_size` and would surface a ragged `k` as an "experts
    // are not evenly stacked" offset mismatch, naming a problem the file does
    // not have.
    anyhow::ensure!(
        first.k.is_multiple_of(first.dtype.block_size()),
        "layer {layer}: {what} inner dim k={} is not divisible by the {:?} block size {}",
        first.k,
        first.dtype,
        first.dtype.block_size(),
    );
    let stride = first
        .m
        .checked_mul(first.k / first.dtype.block_size())
        .and_then(|blocks| blocks.checked_mul(first.dtype.block_bytes()))
        .and_then(|bytes| u32::try_from(bytes).ok())
        .with_context(|| format!("layer {layer}: {what} expert stride overflows"))?;
    // One stride fitting u32 is not enough. The kernel addresses a byte as
    // `sel_expert[entry] * expert_stride + row * row_bytes + bi * 18 + 17`, all
    // in u32, and since `expert_stride == m * row_bytes` that whole tail sums to
    // exactly `expert_stride - 1` at its maximum. The largest address touched is
    // therefore `n_expert * stride - 1`, so it is the *whole stacked extent*
    // that has to fit, not the last expert's base. Past 4 GiB the arithmetic
    // wraps onto another expert's weights, which does not fault and still
    // produces fluent text, the same failure the stride cross-check below exists
    // to catch.
    let n = refs.len();
    let total_bytes = u32::try_from(n)
        .ok()
        .and_then(|experts| experts.checked_mul(stride))
        .with_context(|| {
            format!(
                "layer {layer}: {what} stacked across {n} experts at a {stride}-byte stride \
                 overflows the u32 byte offset the expert GEMV addresses them with"
            )
        })?;
    refs.iter().enumerate().try_for_each(|(e, r)| {
        anyhow::ensure!(
            r.dtype == first.dtype && r.m == first.m && r.k == first.k,
            "layer {layer}: {what} expert {e} has shape {}x{} {:?}, expert 0 has {}x{} {:?}",
            r.m,
            r.k,
            r.dtype,
            first.m,
            first.k,
            first.dtype,
        );
        anyhow::ensure!(
            r.start == first.start + (e as u64) * (stride as u64),
            "layer {layer}: {what} expert {e} starts at byte {} but a {stride}-byte stride from \
             expert 0 puts it at {}; the experts are not evenly stacked",
            r.start,
            first.start + (e as u64) * (stride as u64),
        );
        Ok(())
    })?;
    Ok(StackedExperts {
        rows: first.m,
        inner: first.k,
        expert_stride: stride,
        total_bytes,
    })
}

#[cfg(test)]
mod moe_bound_tests {
    use super::{MOE_MAX_EXPERT_USED, MOE_MAX_EXPERTS};

    /// The two loader bounds are the *shader's* array sizes restated in Rust,
    /// and nothing but this test connects them.
    ///
    /// They are not a policy the host is free to pick: `MAX_EXPERTS` sizes
    /// `groupshared sh_prob[]` and `MAX_USED` sizes the per-thread winners
    /// array, both fixed at shader compile time. Shrinking either constant in
    /// the `.slang` source while the loaders still admit the old maximum is a
    /// groupshared overrun on a real model with no other test failing, so read
    /// them back out of the source the shader is generated from.
    ///
    /// Lives beside the constants rather than in either backend so it runs
    /// under `gpu` and `metal` alike; both backends use the same two.
    #[test]
    fn loader_bounds_match_the_routing_kernel() {
        const SRC: &str = include_str!("../backend/shaders/slang/moe_route.slang");

        let decl = |name: &str| -> u32 {
            SRC.lines()
                .filter_map(|l| l.trim().strip_prefix("static const uint "))
                .find_map(|rest| {
                    rest.strip_prefix(name)?
                        .trim_start()
                        .strip_prefix("= ")?
                        .trim_end_matches(';')
                        .trim_end_matches('u')
                        .parse()
                        .ok()
                })
                .unwrap_or_else(|| {
                    panic!(
                        "moe_route.slang declares no `static const uint {name}`; it was renamed \
                         or removed, and this test is the only thing pinning the loaders' bound \
                         to it"
                    )
                })
        };

        assert_eq!(
            decl("MAX_EXPERTS"),
            MOE_MAX_EXPERTS,
            "the routing kernel's groupshared probability array and the loaders' expert-count \
             bound disagree"
        );
        assert_eq!(
            decl("MAX_USED"),
            MOE_MAX_EXPERT_USED,
            "the routing kernel's per-thread winners array and the loaders' active-expert bound \
             disagree"
        );
    }
}

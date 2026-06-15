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
//! model layer. It is only consumed behind `feature = "gpu"`.
//!
//! The GPU *attention* block is already architecture-generic (Q/K/V GEMV →
//! optional per-head QK-norm → RoPE → GQA attention → output projection); the
//! only arch-specific inputs are exactly the accessors below: which weight
//! refs exist, the optional QK-norm / QKV-bias / untied-output tensors, the
//! RoPE layout + Llama-3 frequency factors, and (via `config().scalars`) the
//! Granite scalar multipliers.

use crate::backend::cpu::RopeType;
use crate::gguf::GgufFile;
use crate::model::ModelConfig;
use crate::model::transformer::WeightRef;

/// Read-only weight + config surface the wgpu loader needs to upload a model.
///
/// `Option`-returning accessors encode per-arch presence: an LFM2 conv layer
/// has no `attn_*` refs (returns `None`); a plain transformer layer has no
/// `conv_*` refs. Callers branch on the block type and only touch the refs
/// they expect to be `Some`.
pub(crate) trait GpuWeightSource {
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

    // ── Raw quantized-weight access (GGUF mmap handles) ─────────────────────
    fn weight_bytes(&self, wref: &WeightRef) -> &[u8];
    fn dequantize_weight(&self, wref: &WeightRef) -> Vec<f32>;

    // ── Per-layer / global weight refs ─────────────────────────────────────
    /// Separate output projection (`output.weight`) when present; `None` ⇒ the
    /// embedding table is reused for the logit projection (tied embeddings).
    fn output_ref(&self) -> Option<&WeightRef>;
    fn ffn_gate_ref(&self, layer: usize) -> &WeightRef;
    fn ffn_up_ref(&self, layer: usize) -> &WeightRef;
    fn ffn_down_ref(&self, layer: usize) -> &WeightRef;
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
    /// follow-up), so they return `false`.
    fn supports_batched_prefill(&self) -> bool;
}

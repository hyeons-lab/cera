//! CPU mirrors of the `constant …& params` structs used by the Metal LFM2,
//! dense-transformer, and audio-decoder inference paths (`metal.rs`,
//! `metal_lfm2.rs`, `metal_audio_decoder.rs`).
//!
//! This mirrors the `constant … & params` structs across the Metal text, audio, *and*
//! ViT vision-encoder (`vision_encoder_gpu.rs` / `MetalVitOps`) paths, uploading each via
//! [`MetalParams::set`] with a `size_of_val`-derived length. The untyped `[u32; N]`
//! uploads this module exists to replace are gone, so the NaN class below is closed
//! crate-wide.
//!
//! # Why these are named types and not `[u32; N]`
//!
//! A Metal kernel takes its scalar arguments as a `constant Params&` bound with
//! `set_bytes(slot, len, ptr)`. Nothing checks `len` against the shader's struct.
//! Upload too few bytes and the kernel reads past the end of the upload — which is
//! undefined behaviour, not a crash.
//!
//! That is not hypothetical. `qk_norm_rope.metal`'s `Params` grew from 7 to 9 fields
//! and gained a `freq_factors` buffer; the audio decoder kept uploading 7 fields with a
//! hardcoded length. Nothing failed to compile. The kernel read the two new flags out of
//! whatever followed the upload and, on a garbage `has_freq_factors`, divided by an
//! unbound buffer — NaN, and silent audio. Note the tail of that: it produced NaN *this
//! time*. Read a different garbage byte and the kernel returns plausible-but-wrong
//! numbers instead, and every test goes green over a real miscompute.
//!
//! # What guards what
//!
//! Two directions of drift, and they need different guards:
//!
//! - **Rust-side** (someone edits a struct here): the `const _: () = assert!(size_of…)`
//!   next to each type is a compile-time break.
//! - **Shader-side** (someone adds a field to the `.metal`): `size_of` cannot see the
//!   shader, so it catches nothing — and this is the direction that actually caused the
//!   NaN. `tests/metal_params_layout.rs` closes it: it parses the MSL source and asserts
//!   each struct here is the same *width* as its counterpart. That is what catches the
//!   NaN class — a field added to the shader makes it wider than the upload. It does not
//!   verify field *order* (the Rust side only exposes `size_of`), so a same-width
//!   reorder would slip through; keep the order matching by hand. It needs no GPU, so it
//!   runs anywhere the `metal` feature compiles.
//!
//! **Keep every struct below field-identical to its MSL counterpart**, `_pad` included.

use metal::{Buffer, ComputeCommandEncoderRef};

/// Upload a params struct to a kernel's `constant` binding.
///
/// The length always comes from `size_of_val`, never a literal — a hardcoded length is
/// the bug this module exists to prevent.
pub trait MetalParams: Sized {
    /// Bind this struct at `slot`.
    fn set(&self, enc: &ComputeCommandEncoderRef, slot: u64) {
        enc.set_bytes(
            slot,
            std::mem::size_of_val(self) as u64,
            self as *const Self as *const _,
        );
    }
}

// ── RoPE / QK-norm ──────────────────────────────────────────────────────────────

/// Mirror of `Params` in `shaders/qk_norm_rope.metal` (binding 4).
///
/// Use [`Self::bind`]: it sets binding 4 with a `size_of_val`-derived length *and*
/// binding 5 in the same call, so neither "wrong length" nor "forgot the freq_factors
/// buffer" is expressible at a call site.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkNormRopeParams {
    pub pos: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps_bits: u32,
    pub freq_base_bits: u32,
    /// 0 = NeoX (pairs at `[i, i+half]`); 1 = interleaved/NORM (pairs at `[2i, 2i+1]`).
    pub rope_type: u32,
    /// 1 ⇒ divide each pair's angle by `freq_factors[d]` (Llama-3 long-context scaling).
    /// When 0, the buffer passed to [`Self::bind`] is never read — a 1-element dummy is fine.
    pub has_freq_factors: u32,
    /// 1 ⇒ per-head RMS-norm of Q/K before RoPE (LFM2 / Qwen3 / the audio decoder);
    /// 0 ⇒ RoPE only (LLaMA / Qwen2 / Mistral / Granite).
    pub has_qk_norm: u32,
}

const _: () = assert!(size_of::<QkNormRopeParams>() == 36); // 9 × uint, qk_norm_rope.metal

impl QkNormRopeParams {
    /// Bind the params (buffer 4) and the `freq_factors` array (buffer 5).
    ///
    /// `freq_factors` must always be a live buffer even when `has_freq_factors == 0`:
    /// the kernel declares the binding unconditionally, and leaving slot 5 unbound is
    /// what produced NaN. Pass a 1-element `[1.0]` dummy in that case — `1.0`, not `0.0`,
    /// so that flipping the flag on can't divide by zero.
    pub fn bind(&self, enc: &ComputeCommandEncoderRef, freq_factors: &Buffer) {
        self.set(enc, 4);
        enc.set_buffer(5, Some(freq_factors), 0);
    }
}
impl MetalParams for QkNormRopeParams {}

/// Mirror of `BatchParams` in `shaders/qk_norm_rope_batch.metal` (binding 4).
///
/// The batched prefill sibling of [`QkNormRopeParams`]: same kernel body, but over `n`
/// tokens with per-token Q/K strides. Its own type because the layouts genuinely differ.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkNormRopeBatchParams {
    pub start_pos: u32,
    pub n_tokens: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps_bits: u32,
    pub freq_base_bits: u32,
    pub rope_type: u32,
    pub q_stride: u32,
    pub k_stride: u32,
    pub has_freq_factors: u32,
    pub has_qk_norm: u32,
}

const _: () = assert!(size_of::<QkNormRopeBatchParams>() == 48); // 12 × uint

impl QkNormRopeBatchParams {
    /// Bind the params (buffer 4) and the `freq_factors` array (buffer 5).
    /// See [`QkNormRopeParams::bind`] — slot 5 must always be live.
    pub fn bind(&self, enc: &ComputeCommandEncoderRef, freq_factors: &Buffer) {
        self.set(enc, 4);
        enc.set_buffer(5, Some(freq_factors), 0);
    }
}
impl MetalParams for QkNormRopeBatchParams {}

/// Mirror of `KParams` in `shaders/kv_shift.metal` (binding 2).
///
/// Same bug as [`QkNormRopeParams`], from the other side: `kv_shift.metal` grew
/// `rope_type` / `has_freq_factors` and a `freq_factors` buffer at binding 3. The shipped
/// dispatch was updated; `tests/metal_kv_shift_oracle.rs` kept a private copy of the old
/// 8-field layout and never bound slot 3 — so the oracle, the test whose entire job is to
/// police this kernel, was itself dispatching it wrong and comparing against NaN.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvShiftKParams {
    pub n_keep: u32,
    pub shift: u32,
    pub new_seq_len: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub freq_base_bits: u32,
    /// `-(shift as i32)`: the rotation delta applied to each retained cell.
    pub delta_pos: i32,
    /// 0 = NeoX, 1 = NORM/interleaved. Must match the layout the forward pass used —
    /// shifting a NORM model with the NeoX layout pairs the wrong elements.
    pub rope_type: u32,
    /// 1 ⇒ divide each pair's angle by `freq_factors[d]`. See [`QkNormRopeParams`].
    pub has_freq_factors: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<KvShiftKParams>() == 40); // 10 × 4B (incl. _pad), kv_shift.metal

impl KvShiftKParams {
    /// Bind the params (buffer 2) and the `freq_factors` array (buffer 3).
    ///
    /// Slot 3 must always be live even when `has_freq_factors == 0` — see
    /// [`QkNormRopeParams::bind`] for why, and pass a `[1.0]` dummy.
    pub fn bind(&self, enc: &ComputeCommandEncoderRef, freq_factors: &Buffer) {
        self.set(enc, 2);
        enc.set_buffer(3, Some(freq_factors), 0);
    }
}
impl MetalParams for KvShiftKParams {}

/// Mirror of `Params` in `shaders/rope.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeParams {
    pub pos: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub freq_base_bits: u32,
}
const _: () = assert!(size_of::<RopeParams>() == 20);
impl MetalParams for RopeParams {}

// ── GEMM / GEMV ─────────────────────────────────────────────────────────────────

/// Mirror of `GemmParams` in `shaders/gemm_f32.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmF32Params {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
const _: () = assert!(size_of::<GemmF32Params>() == 12);
impl MetalParams for GemmF32Params {}

/// Mirror of `GemmParams` in `shaders/gemm_q4_0.metal`, `gemm_q8_0.metal` and
/// `gemm_q4_k.metal` — all three declare the identical layout.
///
/// The last field is the shader's `_pad`, and it is genuinely padding: **none of the
/// three kernels reads it**, so they always plain-store and never accumulate. Callers
/// must not smuggle an `accumulate` flag through it. `MetalLfm2Model::encode_gemm`
/// enforces that by routing every accumulating call to the GEMV fallback before it can
/// reach these kernels — see the `accumulate` guard there. Contrast [`GemvBatchParams`],
/// whose equivalent slot is a real `accum` flag the kernel honours.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantGemmParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub x_stride: u32,
    pub y_stride: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<QuantGemmParams>() == 24);
impl MetalParams for QuantGemmParams {}

/// Mirror of `BatchParams` in `shaders/gemv_q4_0_batch.metal` / `gemv_q8_0_batch.metal`.
///
/// Same shape as [`QuantGemmParams`] but the final field is a live `accum` flag, not
/// padding: these kernels *do* honour it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvBatchParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub x_stride: u32,
    pub y_stride: u32,
    /// 1 ⇒ `y += A·x` instead of `y = A·x`.
    pub accum: u32,
}
const _: () = assert!(size_of::<GemvBatchParams>() == 24);
impl MetalParams for GemvBatchParams {}

/// Mirror of `ParamsQKV` in `shaders/gemv_q4_0_fast.metal` (fused Q/K/V projection).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvQkvParams {
    pub m_q: u32,
    pub m_kv: u32,
    pub k: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<GemvQkvParams>() == 16);
impl MetalParams for GemvQkvParams {}

/// Mirror of `RMSParams` in `shaders/gemv_q4_0_fast.metal` (fused rmsnorm + gate/up).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvRmsParams {
    pub m: u32,
    pub k: u32,
    pub eps_bits: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<GemvRmsParams>() == 16);
impl MetalParams for GemvRmsParams {}

/// Mirror of `SplitKParams` in `shaders/gemv_q4_0_fast.metal` (split-K GEMV).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvSplitKParams {
    pub m: u32,
    pub k: u32,
    pub n_splits: u32,
}
const _: () = assert!(size_of::<GemvSplitKParams>() == 12);
impl MetalParams for GemvSplitKParams {}

// ── Attention ───────────────────────────────────────────────────────────────────

/// Mirror of `Params` in `shaders/flash_attention.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttnParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub seq_len: u32,
    pub scale_bits: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}
const _: () = assert!(size_of::<FlashAttnParams>() == 32);
impl MetalParams for FlashAttnParams {}

/// Mirror of `SplitParams` in `shaders/attention_splitk.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitAttnParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub seq_len: u32,
    pub scale_bits: u32,
    pub n_splits: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<SplitAttnParams>() == 32);
impl MetalParams for SplitAttnParams {}

/// Mirror of `PrefillAttnParams` in `shaders/attention_prefill.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillAttnParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub start_pos: u32,
    pub n_queries: u32,
    pub scale_bits: u32,
    pub q_stride: u32,
    pub out_stride: u32,
}
const _: () = assert!(size_of::<PrefillAttnParams>() == 36);
impl MetalParams for PrefillAttnParams {}

// ── Element-wise / norms / conv ─────────────────────────────────────────────────

/// Mirror of `Params` in `shaders/elementwise.metal` — shared by `memcpy_f32`,
/// `add_inplace`, `cast_f32_to_f16`, `mul_out` and `silu_mul_inplace`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementwiseParams {
    pub n: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<ElementwiseParams>() == 8);
impl MetalParams for ElementwiseParams {}

impl ElementwiseParams {
    /// The common case: `n` elements, zero padding.
    pub fn new(n: u32) -> Self {
        Self { n, _pad: 0 }
    }
}

/// Mirror of `ScaleParams` in `shaders/elementwise.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleParams {
    pub n: u32,
    pub scale_bits: u32,
}
const _: () = assert!(size_of::<ScaleParams>() == 8);
impl MetalParams for ScaleParams {}

/// Mirror of `Params` in `shaders/bias_add.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiasAddParams {
    pub total: u32,
    pub dim: u32,
}
const _: () = assert!(size_of::<BiasAddParams>() == 8);
impl MetalParams for BiasAddParams {}

/// Mirror of `Params` in `shaders/rmsnorm_batch.metal` — shared by `rmsnorm_batch` and
/// `add_rmsnorm_batch`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormBatchParams {
    pub n: u32,
    pub eps_bits: u32,
    pub src_stride: u32,
    pub dst_stride: u32,
    pub res_scale_bits: u32,
}
const _: () = assert!(size_of::<RmsNormBatchParams>() == 20);
impl MetalParams for RmsNormBatchParams {}

/// Mirror of `Params` in `shaders/conv1d_fused_batch.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dBatchParams {
    pub hidden_size: u32,
    pub kernel_size: u32,
    pub d_conv: u32,
    pub n_tokens: u32,
    pub proj_stride: u32,
    pub out_stride: u32,
}
const _: () = assert!(size_of::<Conv1dBatchParams>() == 24);
impl MetalParams for Conv1dBatchParams {}

/// Mirror of `CopyParams` in `shaders/kv_shift.metal` (`memcpy_f16_offsets`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCopyParams {
    pub n_elements: u32,
    pub src_offset_elements: u32,
    pub dst_offset_elements: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<KvCopyParams>() == 16);
impl MetalParams for KvCopyParams {}

// ── ViT vision encoder ────────────────────────────────────────────────────────────

/// Mirror of `Params` in `shaders/vit_linear.metal` (the dense-weight ViT GEMM).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VitLinearParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<VitLinearParams>() == 16);
impl MetalParams for VitLinearParams {}

/// Mirror of `Params` in `shaders/vit_attention.metal` **and** `VitAttnParams` in
/// `shaders/vit_attention_mma.metal` — the scalar and flash-MMA ViT attention kernels
/// declare the identical layout, so one type guards both (two `metal_params_layout`
/// cases). `scale_bits` is `(1/sqrt(head_dim)).to_bits()`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VitAttnParams {
    pub tokens: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub scale_bits: u32,
}
const _: () = assert!(size_of::<VitAttnParams>() == 16);
impl MetalParams for VitAttnParams {}

/// Mirror of `Params` in `shaders/layernorm_batch.metal` (the ViT LayerNorm).
///
/// Distinct from [`RmsNormBatchParams`]: LayerNorm has no residual-scale field, so it is
/// four uints, not five. `src_stride`/`dst_stride` are both `dim` in the ViT caller.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerNormBatchParams {
    pub n: u32,
    pub eps_bits: u32,
    pub src_stride: u32,
    pub dst_stride: u32,
}
const _: () = assert!(size_of::<LayerNormBatchParams>() == 16);
impl MetalParams for LayerNormBatchParams {}

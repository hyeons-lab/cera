//! GPU forward pass for the LFM2A Conformer audio encoder.
//!
//! The CPU encoder ([`super::audio_encoder::audio_encoder_forward`]) runs every
//! linear layer through a per-timestep `gemv` and every attention score through a
//! scalar triple loop, over 17 blocks, which is the slowest stage of the audio-in
//! pipeline. This module batches the whole post-mel forward pass on the GPU.
//!
//! Same shape as the vision encoder's GPU path: the math is written once against
//! the [`AudioEncoderGpuOps`] trait and [`encode_audio_mel_gpu`] is
//! backend-agnostic, so a second backend is a host-side wiring change rather than
//! a kernel port. Only the Metal implementation ships here; the wgpu one follows.
//! Every kernel it will need is already generated for WGSL and exported as
//! `backend::wgpu::shaders::*`, but nothing dispatches those yet: they are
//! checked only for generation faults (no subgroup ops, no `enable f16`, entry
//! points present), not against a numeric reference. The numeric gate in this
//! change is Metal-only.
//!
//! Numerical reference is the CPU encoder: see `tests/audio_encoder_metal_parity.rs`.
//!
//! What stays on the CPU:
//!
//! - **The log-mel front-end.** [`AudioGpuEncode::encode_pcm`] takes raw PCM and
//!   calls [`crate::model::audio_preprocessor::log_mel_spectrogram`] before
//!   handing the spectrogram to the GPU. Moving the STFT and mel filterbank onto
//!   the GPU is a separate change, and this entry point is the seam for it:
//!   callers pass PCM either way, so nothing above this module moves.
//! - **The sinusoidal relative-position table** ([`super::audio_encoder::relative_pos_emb`]),
//!   built once per chunk and uploaded. It is `O(t)` transcendentals against the
//!   encoder's `O(t²·n_embd)` of real work, and it is CPU in the shipping Metal
//!   detokenizer for the same reason.

use anyhow::Result;

use super::audio_encoder::{
    AudioEncoderConfig, AudioEncoderWeights, ConformerLayerWeights, ConvStemWeights, POS_EMB_DIM,
    relative_pos_emb,
};
use crate::model::weights::MmapWeight;

/// Longest post-stem sequence the `audio_xl_attention` kernel supports: its
/// `scores` scratch is workgroup-resident and sized `MAX_TOKENS`.
///
/// (Plain code span, not an intra-doc link: the shader constants are behind the
/// `metal` / `gpu` features, and a link to one would break the featureless
/// rustdoc build.)
///
/// The stem downsamples time by 8×, so this admits ~8192 mel frames ≈ 82 s of
/// audio in one chunk. [`encode_audio_mel_gpu`] checks it *before* uploading
/// anything and returns an error above it, so the caller falls back to the CPU
/// encoder instead of the kernel writing past its scratch.
///
/// This value MUST match the `MAX_TOKENS` literal in
/// `backend/shaders/slang/audio_xl_attention.slang`. The generated WGSL and MSL
/// bake it into a workgroup array size and cannot take a runtime define, so the
/// link is enforced by
/// `const_sync_tests::max_audio_tokens_matches_attention_shader_scratch` rather
/// than by the compiler.
pub const MAX_AUDIO_TOKENS: usize = 1024;

/// Largest `head_dim` the attention kernel's groupshared Q+bias staging arrays
/// hold. LFM2A is 64 (`n_embd` 512 / 8 heads); the guard exists so a wider
/// variant falls back to the CPU rather than corrupting output.
///
/// Kept in lockstep with `MAX_HEAD_DIM` in `audio_xl_attention.slang` by the same
/// const-sync test as [`MAX_AUDIO_TOKENS`].
pub const MAX_AUDIO_HEAD_DIM: usize = 128;

// Co-located with the constants on purpose: this check must run in default CI
// (`#[cfg(test)]` only, no GPU/feature gate), unlike the feature-gated `tests`
// module at the end of the file, so it cannot be folded into it.
#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod const_sync_tests {
    use super::{MAX_AUDIO_HEAD_DIM, MAX_AUDIO_TOKENS};

    /// Fails loudly if either cap is bumped without updating the matching
    /// workgroup-array size in the attention shader, the compile-time link the
    /// `.slang`'s literals would otherwise lack. Reads the `.slang` source, not
    /// the generated targets: the generator is what has to agree with the host,
    /// and the source is the file a person edits. Needs no GPU.
    #[test]
    fn max_audio_tokens_matches_attention_shader_scratch() {
        let src = include_str!("../backend/shaders/slang/audio_xl_attention.slang");
        let tokens_decl = format!("static const uint MAX_TOKENS = {MAX_AUDIO_TOKENS}u;");
        let head_decl = format!("static const uint MAX_HEAD_DIM = {MAX_AUDIO_HEAD_DIM}u;");
        assert!(
            src.contains(&tokens_decl),
            "audio_xl_attention.slang MAX_TOKENS != MAX_AUDIO_TOKENS ({MAX_AUDIO_TOKENS}); \
             update the shader's `scores` array size to match"
        );
        assert!(
            src.contains(&head_decl),
            "audio_xl_attention.slang MAX_HEAD_DIM != MAX_AUDIO_HEAD_DIM ({MAX_AUDIO_HEAD_DIM}); \
             update the shader's `qu`/`qv` array sizes to match"
        );
    }
}

/// The message both capacity refusals carry.
///
/// `encode_pcm` refuses early (before the STFT) and `conv_stem_gpu` refuses
/// again on the path that actually reaches the kernel. Two call sites, one
/// spelling: the parity suite asserts on this text, and a wording edit that
/// landed in only one of them would keep passing.
fn over_capacity_msg(t_out: usize) -> String {
    format!(
        "post-stem length {t_out} exceeds MAX_AUDIO_TOKENS ({MAX_AUDIO_TOKENS}); \
         caller should fall back to the CPU encoder"
    )
}

/// Per-stem-layer `(depthwise, stride, pad)`, hardcoded to the LFM2A C++
/// reference exactly as [`super::audio_encoder::conv_stem_forward`] has them.
/// `depthwise` selects `groups = in_ch` over `groups = 1`.
pub const STEM_LAYER_MODES: [(bool, usize, usize); 5] = [
    (false, 2, 1), // layer.0: regular 3x3 s2 p1, 1 -> 256
    (true, 2, 1),  // layer.2: depthwise 3x3 s2 p1, 256 ch
    (false, 1, 0), // layer.3: pointwise 1x1, 256 -> 256
    (true, 2, 1),  // layer.5: depthwise 3x3 s2 p1, 256 ch
    (false, 1, 0), // layer.6: pointwise 1x1, 256 -> 256
];

/// ReLU follows positional stem layers 0, 2 and 4 (GGUF indices 1, 4 and 7,
/// which carry no parameters). Same table as the CPU stem.
pub const STEM_RELU_AFTER: [bool; 5] = [true, false, true, false, true];

/// Geometry of one convolution, in the form the `conv2d_direct` kernel takes.
///
/// `pad_h`/`pad_w` are the **low-side** pad only and the output dims are carried
/// explicitly rather than re-derived from a symmetric pad, so an asymmetric split
/// stays expressible. See [`Self::padded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dSpec {
    pub in_ch: usize,
    pub out_ch: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub groups: usize,
}

impl Conv2dSpec {
    /// Build a spec from independent low/high padding on each axis, computing the
    /// output dims the way the CPU convolutions do.
    ///
    /// Both audio call sites go through here:
    ///
    /// - the conv stem, with `pad_lo == pad_hi` on both axes;
    /// - each Conformer block's depthwise conv1d, as `kh = 1`, `h_in = 1`,
    ///   `pad_h = 0`. `cpu::conformer_conv_module_forward` splits its
    ///   `kernel_size - 1` total pad as `pad_left = total / 2` with the remainder
    ///   on the right, which is symmetric only for odd kernels. Taking lo and hi
    ///   separately reproduces the even case instead of shifting the convolution
    ///   half a tap.
    ///
    /// Returns `None` when the convolution is not runnable: a zero channel
    /// count, `groups`, stride or kernel dimension, a channel count not divisible
    /// by `groups`, or a kernel larger than the padded input on either axis
    /// (which would underflow the output-dim math).
    #[allow(clippy::too_many_arguments)]
    pub fn padded(
        in_ch: usize,
        out_ch: usize,
        h_in: usize,
        w_in: usize,
        (kh, kw): (usize, usize),
        (stride_h, stride_w): (usize, usize),
        (pad_h_lo, pad_h_hi): (usize, usize),
        (pad_w_lo, pad_w_hi): (usize, usize),
        groups: usize,
    ) -> Option<Self> {
        if in_ch == 0 || out_ch == 0 || groups == 0 {
            return None;
        }
        if stride_h == 0 || stride_w == 0 || kh == 0 || kw == 0 {
            return None;
        }
        if !in_ch.is_multiple_of(groups) || !out_ch.is_multiple_of(groups) {
            return None;
        }
        let padded_h = h_in.checked_add(pad_h_lo)?.checked_add(pad_h_hi)?;
        let padded_w = w_in.checked_add(pad_w_lo)?.checked_add(pad_w_hi)?;
        if padded_h < kh || padded_w < kw {
            return None;
        }
        // No `h_out == 0` check: the `padded < k` guard above already makes
        // `(padded - k) / stride + 1` at least 1, so one here would be vacuous.
        let h_out = (padded_h - kh) / stride_h + 1;
        let w_out = (padded_w - kw) / stride_w + 1;
        Some(Self {
            in_ch,
            out_ch,
            h_in,
            w_in,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h: pad_h_lo,
            pad_w: pad_w_lo,
            h_out,
            w_out,
            groups,
        })
    }

    /// Element count of this convolution's output buffer.
    pub fn out_len(&self) -> usize {
        self.out_ch * self.h_out * self.w_out
    }
}

/// Backend-agnostic GPU op interface for the Conformer audio encoder.
///
/// All ops operate on row-major f32 buffers. In-place ops (`bias_add`, the
/// activations, `add`, `scaled_add`, `chan_affine_silu`) mutate the GPU contents
/// behind `&Self::Buf`; producing ops allocate and return a fresh buffer.
///
/// Deliberately separate from [`crate::model::vision_encoder_gpu::VitGpuOps`]
/// rather than layered on it. The two overlap on the generic half (upload,
/// linear, layernorm, bias_add, add) but diverge on the half that matters: this
/// encoder's attention carries a Transformer-XL relative-position bias and its
/// convolutions are real 2D convs, neither of which the ViT trait can express.
/// Coupling them would mean every future audio op widening the vision trait.
pub trait AudioEncoderGpuOps {
    /// Opaque GPU buffer handle (e.g. `metal::Buffer`).
    type Buf;

    /// A linear-layer weight, ready for [`Self::linear`]. Backends may keep
    /// quantized weights packed and run a quantized GEMM straight from the bytes;
    /// other dtypes are dequantized to f32. Distinct from [`Self::Buf`] so
    /// backends can carry the dtype/packing alongside the GPU buffer.
    type Weight;

    /// Upload `data` to a new GPU buffer.
    fn upload(&self, data: &[f32]) -> Self::Buf;

    /// Read `len` f32s back from a GPU buffer (blocking).
    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32>;

    /// Upload a (possibly quantized) linear weight `[out_dim, in_dim]` row-major
    /// (the [`MmapWeight`] layout).
    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight;

    /// `y[rows, out_dim] = x[rows, in_dim] · wᵀ` where `w` is the
    /// `[out_dim, in_dim]` weight uploaded via [`Self::upload_weight`].
    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf;

    /// In-place broadcast bias: `x[r*dim + j] += bias[j]` for all `rows` rows.
    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize);

    /// Out-of-place affine LayerNorm over the last dim, returning a new buffer.
    /// `(src - mean) * inv_std * weight + bias` per row.
    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf;

    /// In-place ReLU over `len` elements.
    fn relu(&self, x: &Self::Buf, len: usize);

    /// In-place SiLU over `len` elements.
    fn silu(&self, x: &Self::Buf, len: usize);

    /// In-place **erf-form** GELU over `len` elements: the variant the audio
    /// adapter was trained against, not the tanh approximation the ViT uses.
    fn gelu_erf(&self, x: &Self::Buf, len: usize);

    /// In-place residual add: `dst[i] += src[i]` over `len` elements.
    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize);

    /// In-place scaled residual add: `dst[i] += scale * src[i]`. The Conformer's
    /// macaron FFNs accumulate at half weight.
    fn scaled_add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize, scale: f32);

    /// Direct 2D convolution per `spec`. `weight` is
    /// `[out_ch][in_per_group][kh][kw]` and `bias` is `[out_ch]`, both dense f32;
    /// returns a fresh `[out_ch][h_out][w_out]` buffer.
    fn conv2d(
        &self,
        input: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        spec: &Conv2dSpec,
    ) -> Self::Buf;

    /// Swap the outer two axes of `[a][b][k]`, keeping the `k`-wide inner block
    /// contiguous: returns `[b][a][k]`.
    fn transpose_blocked(&self, src: &Self::Buf, a: usize, b: usize, k: usize) -> Self::Buf;

    /// GLU split: returns `[rows][n]` with `dst[r][c] = src[r][c] * sigmoid(src[r][n+c])`
    /// from a `[rows][2n]` source.
    fn glu_split(&self, src: &Self::Buf, rows: usize, n: usize) -> Self::Buf;

    /// In-place per-channel affine + SiLU over channel-major `[channels][t]`:
    /// `x[c][i] = silu(x[c][i] * w[c] + b[c])`.
    fn chan_affine_silu(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        b: &Self::Buf,
        channels: usize,
        t: usize,
    );

    /// Conformer self-attention with Transformer-XL relative-position bias.
    /// `q`/`k`/`v` are `[tokens, n_head*head_dim]`, `p` is
    /// `[2*tokens-1, n_head*head_dim]`, `bias_u`/`bias_v` are `[n_head*head_dim]`.
    /// Returns `[tokens, n_head*head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn xl_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        p: &Self::Buf,
        bias_u: &Self::Buf,
        bias_v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf;
}

/// One conv-stem layer uploaded to GPU buffers, plus the geometry the loader
/// recovered from its GGUF shape.
struct GpuStemLayer<O: AudioEncoderGpuOps> {
    weight: O::Buf,
    bias: O::Buf,
    /// `(kh, kw)` from the GGUF shape `[kw, kh, in_per_group, out_ch]`.
    kernel: (usize, usize),
    in_per_group: usize,
    out_ch: usize,
}

/// One Conformer block's weights, uploaded to GPU buffers. Linear weights are
/// `O::Weight` (possibly quantized); norm/bias vectors are plain f32 `O::Buf`.
struct GpuConformerBlock<O: AudioEncoderGpuOps> {
    // FFN-1 (half residual).
    ffn_norm_w: O::Buf,
    ffn_norm_b: O::Buf,
    ffn_up_w: O::Weight,
    ffn_up_b: O::Buf,
    ffn_down_w: O::Weight,
    ffn_down_b: O::Buf,

    // Self-attention with relative-position bias.
    ln1_w: O::Buf,
    ln1_b: O::Buf,
    attn_q_w: O::Weight,
    attn_q_b: O::Buf,
    attn_k_w: O::Weight,
    attn_k_b: O::Buf,
    attn_v_w: O::Weight,
    attn_v_b: O::Buf,
    attn_o_w: O::Weight,
    attn_o_b: O::Buf,
    pos_bias_u: O::Buf,
    pos_bias_v: O::Buf,
    linear_pos_w: O::Weight,

    // Convolution module.
    norm_conv_w: O::Buf,
    norm_conv_b: O::Buf,
    conv_pw1_w: O::Weight,
    conv_pw1_b: O::Buf,
    conv_dw_w: O::Buf,
    conv_dw_b: O::Buf,
    /// Depthwise kernel width, from the block's `conv_dw` GGUF shape.
    conv_dw_k: usize,
    conv_norm_w: O::Buf,
    conv_norm_b: O::Buf,
    conv_pw2_w: O::Weight,
    conv_pw2_b: O::Buf,

    // FFN-2 (half residual).
    ffn_norm_1_w: O::Buf,
    ffn_norm_1_b: O::Buf,
    ffn_up_1_w: O::Weight,
    ffn_up_1_b: O::Buf,
    ffn_down_1_w: O::Weight,
    ffn_down_1_b: O::Buf,

    // Final per-block LayerNorm.
    ln2_w: O::Buf,
    ln2_b: O::Buf,
}

/// Check one Conformer block's tensors against the config the forward pass
/// hardcodes, returning its depthwise kernel width.
///
/// Separate from [`GpuAudioWeights::build`] so the upload reads as an upload:
/// these three tables are the reviewable unit, and inlining them in the closure
/// that also writes a 38-field struct literal buried them.
fn validate_block(il: usize, b: &ConformerLayerWeights, cfg: &AudioEncoderConfig) -> Result<usize> {
    let n_embd = cfg.n_embd;
    // The forward pass hardcodes n_embd / n_ff / POS_EMB_DIM as the
    // GEMM dims, so a mismatched mmproj would be an out-of-bounds
    // read on the GPU (silent garbage, and past the point where the
    // caller can still fall back). The CPU encoder gets a slice
    // panic here; this path has to check for itself.
    let linears: [(&str, &MmapWeight, usize, usize); 10] = [
        ("ffn_up", &b.ffn_up_w, cfg.n_ff, n_embd),
        ("ffn_down", &b.ffn_down_w, n_embd, cfg.n_ff),
        ("ffn_up_1", &b.ffn_up_1_w, cfg.n_ff, n_embd),
        ("ffn_down_1", &b.ffn_down_1_w, n_embd, cfg.n_ff),
        ("attn_q", &b.attn_q_w, n_embd, n_embd),
        ("attn_k", &b.attn_k_w, n_embd, n_embd),
        ("attn_v", &b.attn_v_w, n_embd, n_embd),
        ("attn_o", &b.attn_o_w, n_embd, n_embd),
        ("conv_pw1", &b.conv_pw1_w, 2 * n_embd, n_embd),
        ("conv_pw2", &b.conv_pw2_w, n_embd, n_embd),
    ];
    for (name, weight, rows, cols) in linears {
        anyhow::ensure!(
            weight.rows == rows && weight.cols == cols,
            "audio encoder block {il}: {name} is [{}, {}], expected [{rows}, {cols}]",
            weight.rows,
            weight.cols,
        );
    }
    anyhow::ensure!(
        b.linear_pos_w.rows == n_embd && b.linear_pos_w.cols == POS_EMB_DIM,
        "audio encoder block {il}: linear_pos is [{}, {}], expected \
                 [{n_embd}, {POS_EMB_DIM}]",
        b.linear_pos_w.rows,
        b.linear_pos_w.cols,
    );
    // Norm weights and biases are broadcast by index, so a short one
    // is an out-of-bounds read rather than a shape error.
    let vectors: [(&str, usize, usize); 25] = [
        ("ffn_norm_w", b.ffn_norm_w.len(), n_embd),
        ("ffn_norm_b", b.ffn_norm_b.len(), n_embd),
        ("ffn_norm_1_w", b.ffn_norm_1_w.len(), n_embd),
        ("ffn_norm_1_b", b.ffn_norm_1_b.len(), n_embd),
        ("ln1_w", b.ln1_w.len(), n_embd),
        ("ln1_b", b.ln1_b.len(), n_embd),
        ("ln2_w", b.ln2_w.len(), n_embd),
        ("ln2_b", b.ln2_b.len(), n_embd),
        ("norm_conv_w", b.norm_conv_w.len(), n_embd),
        ("norm_conv_b", b.norm_conv_b.len(), n_embd),
        ("pos_bias_u", b.pos_bias_u.len(), n_embd),
        ("pos_bias_v", b.pos_bias_v.len(), n_embd),
        // `conv_norm_w/b` is the per-channel affine inside the conv
        // module, NOT the `norm_conv_w/b` LayerNorm above it. The CPU
        // encoder's docs flag the two as easy to confuse, and
        // `chan_affine_silu` indexes this one by channel, so a short
        // one reads out of bounds.
        ("conv_norm_w", b.conv_norm_w.len(), n_embd),
        ("conv_norm_b", b.conv_norm_b.len(), n_embd),
        // Biases, all consumed positionally by `bias_add` / `conv2d`.
        ("attn_q_b", b.attn_q_b.len(), n_embd),
        ("attn_k_b", b.attn_k_b.len(), n_embd),
        ("attn_v_b", b.attn_v_b.len(), n_embd),
        ("attn_o_b", b.attn_o_b.len(), n_embd),
        ("ffn_up_b", b.ffn_up_b.len(), cfg.n_ff),
        ("ffn_down_b", b.ffn_down_b.len(), n_embd),
        ("ffn_up_1_b", b.ffn_up_1_b.len(), cfg.n_ff),
        ("ffn_down_1_b", b.ffn_down_1_b.len(), n_embd),
        ("conv_pw1_b", b.conv_pw1_b.len(), 2 * n_embd),
        ("conv_pw2_b", b.conv_pw2_b.len(), n_embd),
        ("conv_dw_b", b.conv_dw_b.len(), n_embd),
    ];
    for (name, got, want) in vectors {
        anyhow::ensure!(
            got == want,
            "audio encoder block {il}: {name} has {got} values, expected {want}",
        );
    }
    // Both the 2D `[k, channels]` form LFM2A stores and the 3D
    // `[k, 1, channels]` form other loaders use put kernel_size
    // first, matching `audio_encoder_forward`.
    let conv_dw_k = *b.conv_dw_shape.first().unwrap_or(&0);
    anyhow::ensure!(
        conv_dw_k > 0 && conv_dw_k * cfg.n_embd == b.conv_dw_w.len(),
        "audio encoder block {il}: conv_dw shape {:?} disagrees with its \
                 {} weights at n_embd {}",
        b.conv_dw_shape,
        b.conv_dw_w.len(),
        cfg.n_embd,
    );
    Ok(conv_dw_k)
}

/// All audio-encoder weights uploaded to GPU buffers.
///
/// Built once via [`GpuAudioWeights::build`] and reused across chunks: the
/// upload (the mmproj dequantized where the backend has no packed GEMM) is the
/// expensive part and must not happen per utterance.
pub struct GpuAudioWeights<O: AudioEncoderGpuOps> {
    cfg: AudioEncoderConfig,
    stem: Vec<GpuStemLayer<O>>,
    pre_encode_out_w: O::Weight,
    pre_encode_out_b: O::Buf,
    /// Column count of `pre_encode_out` (`channels · freq` after the stem),
    /// carried from the loaded weight so the flatten can be checked against it.
    pre_encode_in_dim: usize,
    blocks: Vec<GpuConformerBlock<O>>,
    adapter_norm_w: O::Buf,
    adapter_norm_b: O::Buf,
    adapter_up_w: O::Weight,
    adapter_up_b: O::Buf,
    adapter_down_w: O::Weight,
    adapter_down_b: O::Buf,
    /// Adapter intermediate width (`mm.a.mlp.1` rows), derived from the tensor
    /// shape rather than assumed, matching the CPU encoder's loader.
    adapter_intermediate: usize,
}

impl<O: AudioEncoderGpuOps> GpuAudioWeights<O> {
    /// Upload every encoder weight via `ops`. Run once per loaded model.
    ///
    /// Fails rather than uploading a model this path cannot run correctly: the
    /// stem layer count, the attention head geometry, and the adapter dims are
    /// all checked here, where the error can name the tensor, instead of
    /// surfacing as a wrong-shaped dispatch later.
    pub fn build(ops: &O, w: &AudioEncoderWeights) -> Result<Self> {
        let cfg = w.config.clone();
        anyhow::ensure!(cfg.n_head > 0, "audio encoder config has n_head = 0");
        anyhow::ensure!(
            cfg.n_embd.is_multiple_of(cfg.n_head),
            "audio encoder n_embd ({}) is not divisible by n_head ({})",
            cfg.n_embd,
            cfg.n_head,
        );
        let head_dim = cfg.n_embd / cfg.n_head;
        anyhow::ensure!(
            head_dim <= MAX_AUDIO_HEAD_DIM,
            "audio encoder head_dim ({head_dim}) exceeds the attention kernel's \
             MAX_AUDIO_HEAD_DIM ({MAX_AUDIO_HEAD_DIM}); caller should use the CPU encoder",
        );
        anyhow::ensure!(
            w.layers.len() == cfg.n_layer,
            "audio encoder config.n_layer ({}) != loaded blocks ({})",
            cfg.n_layer,
            w.layers.len(),
        );
        anyhow::ensure!(
            w.conv_stem.layers.len() == STEM_LAYER_MODES.len(),
            "audio encoder conv stem has {} layers, expected {}",
            w.conv_stem.layers.len(),
            STEM_LAYER_MODES.len(),
        );

        let stem = Self::build_stem(ops, &w.conv_stem)?;

        // `in_per_group * groups == in_ch` depends only on the stem's channel
        // counts, never on the input length, so it belongs here rather than on
        // every encode: failing at load lets `try_metal_audio_encoder` decline and
        // fall back to the CPU encoder, where failing mid-encode cannot.
        let stem_in_ch = std::iter::once(1usize).chain(stem.iter().map(|l| l.out_ch));
        for ((pos, layer), in_ch) in stem.iter().enumerate().zip(stem_in_ch) {
            let (depthwise, ..) = STEM_LAYER_MODES[pos];
            let groups = if depthwise { in_ch } else { 1 };
            anyhow::ensure!(
                layer.in_per_group * groups == in_ch,
                "audio conv stem layer {pos}: in_per_group ({}) * groups ({groups}) != in_ch ({in_ch})",
                layer.in_per_group,
            );
        }

        anyhow::ensure!(
            w.conv_stem.pre_encode_out_w.rows == cfg.n_embd
                && w.conv_stem.pre_encode_out_b.len() == cfg.n_embd,
            "audio encoder pre_encode_out is [{}, {}] with {} bias values, expected {} rows",
            w.conv_stem.pre_encode_out_w.rows,
            w.conv_stem.pre_encode_out_w.cols,
            w.conv_stem.pre_encode_out_b.len(),
            cfg.n_embd,
        );

        let adapter_intermediate = w.mlp_adapter.up_w.rows;
        anyhow::ensure!(
            w.mlp_adapter.norm_w.len() == cfg.n_embd
                && w.mlp_adapter.norm_b.len() == cfg.n_embd
                && w.mlp_adapter.up_b.len() == adapter_intermediate
                && w.mlp_adapter.down_b.len() == cfg.llm_hidden_size,
            "audio encoder MLP adapter vectors disagree with the config \
             (norm {}/{}, up_b {}, down_b {}; n_embd {}, intermediate {}, llm_hidden {})",
            w.mlp_adapter.norm_w.len(),
            w.mlp_adapter.norm_b.len(),
            w.mlp_adapter.up_b.len(),
            w.mlp_adapter.down_b.len(),
            cfg.n_embd,
            adapter_intermediate,
            cfg.llm_hidden_size,
        );
        anyhow::ensure!(
            w.mlp_adapter.up_w.cols == cfg.n_embd
                && w.mlp_adapter.down_w.cols == adapter_intermediate
                && w.mlp_adapter.down_w.rows == cfg.llm_hidden_size,
            "audio encoder MLP adapter shapes disagree with the config \
             (up [{}, {}], down [{}, {}], n_embd {}, llm_hidden_size {})",
            w.mlp_adapter.up_w.rows,
            w.mlp_adapter.up_w.cols,
            w.mlp_adapter.down_w.rows,
            w.mlp_adapter.down_w.cols,
            cfg.n_embd,
            cfg.llm_hidden_size,
        );

        let blocks = w
            .layers
            .iter()
            .enumerate()
            .map(|(il, b)| {
                let conv_dw_k = validate_block(il, b, &cfg)?;
                Ok(GpuConformerBlock {
                    ffn_norm_w: ops.upload(&b.ffn_norm_w),
                    ffn_norm_b: ops.upload(&b.ffn_norm_b),
                    ffn_up_w: ops.upload_weight(&b.ffn_up_w),
                    ffn_up_b: ops.upload(&b.ffn_up_b),
                    ffn_down_w: ops.upload_weight(&b.ffn_down_w),
                    ffn_down_b: ops.upload(&b.ffn_down_b),
                    ln1_w: ops.upload(&b.ln1_w),
                    ln1_b: ops.upload(&b.ln1_b),
                    attn_q_w: ops.upload_weight(&b.attn_q_w),
                    attn_q_b: ops.upload(&b.attn_q_b),
                    attn_k_w: ops.upload_weight(&b.attn_k_w),
                    attn_k_b: ops.upload(&b.attn_k_b),
                    attn_v_w: ops.upload_weight(&b.attn_v_w),
                    attn_v_b: ops.upload(&b.attn_v_b),
                    attn_o_w: ops.upload_weight(&b.attn_o_w),
                    attn_o_b: ops.upload(&b.attn_o_b),
                    pos_bias_u: ops.upload(&b.pos_bias_u),
                    pos_bias_v: ops.upload(&b.pos_bias_v),
                    linear_pos_w: ops.upload_weight(&b.linear_pos_w),
                    norm_conv_w: ops.upload(&b.norm_conv_w),
                    norm_conv_b: ops.upload(&b.norm_conv_b),
                    conv_pw1_w: ops.upload_weight(&b.conv_pw1_w),
                    conv_pw1_b: ops.upload(&b.conv_pw1_b),
                    conv_dw_w: ops.upload(&b.conv_dw_w),
                    conv_dw_b: ops.upload(&b.conv_dw_b),
                    conv_dw_k,
                    conv_norm_w: ops.upload(&b.conv_norm_w),
                    conv_norm_b: ops.upload(&b.conv_norm_b),
                    conv_pw2_w: ops.upload_weight(&b.conv_pw2_w),
                    conv_pw2_b: ops.upload(&b.conv_pw2_b),
                    ffn_norm_1_w: ops.upload(&b.ffn_norm_1_w),
                    ffn_norm_1_b: ops.upload(&b.ffn_norm_1_b),
                    ffn_up_1_w: ops.upload_weight(&b.ffn_up_1_w),
                    ffn_up_1_b: ops.upload(&b.ffn_up_1_b),
                    ffn_down_1_w: ops.upload_weight(&b.ffn_down_1_w),
                    ffn_down_1_b: ops.upload(&b.ffn_down_1_b),
                    ln2_w: ops.upload(&b.ln2_w),
                    ln2_b: ops.upload(&b.ln2_b),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            cfg,
            stem,
            pre_encode_out_w: ops.upload_weight(&w.conv_stem.pre_encode_out_w),
            pre_encode_out_b: ops.upload(&w.conv_stem.pre_encode_out_b),
            pre_encode_in_dim: w.conv_stem.pre_encode_out_w.cols,
            blocks,
            adapter_norm_w: ops.upload(&w.mlp_adapter.norm_w),
            adapter_norm_b: ops.upload(&w.mlp_adapter.norm_b),
            adapter_up_w: ops.upload_weight(&w.mlp_adapter.up_w),
            adapter_up_b: ops.upload(&w.mlp_adapter.up_b),
            adapter_down_w: ops.upload_weight(&w.mlp_adapter.down_w),
            adapter_down_b: ops.upload(&w.mlp_adapter.down_b),
            adapter_intermediate,
        })
    }

    /// Upload the five stem convolutions, decoding each GGUF shape
    /// `[kw, kh, in_per_group, out_ch]` into the geometry the forward pass needs.
    fn build_stem(ops: &O, stem: &ConvStemWeights) -> Result<Vec<GpuStemLayer<O>>> {
        stem.layers
            .iter()
            .enumerate()
            .map(|(pos, layer)| {
                anyhow::ensure!(
                    layer.shape.len() == 4,
                    "audio conv stem layer {pos} ({}): expected a 4-dim weight shape, got {:?}",
                    layer.name,
                    layer.shape,
                );
                let (kw, kh, in_per_group, out_ch) = (
                    layer.shape[0],
                    layer.shape[1],
                    layer.shape[2],
                    layer.shape[3],
                );
                anyhow::ensure!(
                    out_ch * in_per_group * kh * kw == layer.weight.len(),
                    "audio conv stem layer {pos} ({}): shape {:?} disagrees with its {} weights",
                    layer.name,
                    layer.shape,
                    layer.weight.len(),
                );
                anyhow::ensure!(
                    layer.bias.len() == out_ch,
                    "audio conv stem layer {pos} ({}): {} bias values for {out_ch} channels",
                    layer.name,
                    layer.bias.len(),
                );
                Ok(GpuStemLayer {
                    weight: ops.upload(&layer.weight),
                    bias: ops.upload(&layer.bias),
                    kernel: (kh, kw),
                    in_per_group,
                    out_ch,
                })
            })
            .collect()
    }

    /// Walk the stem's geometry to get the post-stem sequence length for
    /// `n_frames` mel frames, without touching the GPU.
    ///
    /// Lets [`AudioGpuEncode::encode_pcm`] check the attention kernel's capacity
    /// *before* computing the spectrogram, so an over-long chunk costs nothing on
    /// its way to the CPU fallback.
    ///
    /// It cannot disagree with the stem that actually runs because both go
    /// through the one `stem_specs` walk (plain code span: it is private, and an
    /// intra-doc link to it fails `rustdoc::private_intra_doc_links` on this
    /// public item); `conv_stem_gpu` re-checks the
    /// cap itself rather than trusting this, so the guard holds even if a caller
    /// skips the early check.
    ///
    /// `None` means the stem cannot run at this input size at all (the kernel
    /// exceeds the padded input on some layer).
    pub fn predict_t_out(&self, n_frames: usize) -> Option<usize> {
        Some(self.stem_specs(n_frames)?.last()?.h_out)
    }

    /// The geometry of all five stem convolutions for `n_frames` mel frames.
    ///
    /// One walk, used both by [`Self::predict_t_out`] (to check the attention
    /// kernel's capacity before anything is uploaded) and by the forward pass
    /// (to dispatch). Deriving them twice and reconciling the results is how the
    /// capacity guard silently becomes a no-op: it protects a fixed-size
    /// workgroup array, so a prediction that drifts from the dispatch is an
    /// out-of-bounds write, not a wrong answer.
    ///
    /// `None` means the stem cannot run at this input size (some layer's kernel
    /// exceeds its padded input).
    fn stem_specs(&self, n_frames: usize) -> Option<Vec<Conv2dSpec>> {
        self.stem
            .iter()
            .zip(&STEM_LAYER_MODES)
            .try_fold(
                (
                    Vec::with_capacity(STEM_LAYER_MODES.len()),
                    1usize,
                    n_frames,
                    self.cfg.n_mel_bins,
                ),
                |(mut specs, ch, h, w), (layer, &(depthwise, stride, pad))| {
                    let spec = Conv2dSpec::padded(
                        ch,
                        layer.out_ch,
                        h,
                        w,
                        layer.kernel,
                        (stride, stride),
                        (pad, pad),
                        (pad, pad),
                        if depthwise { ch } else { 1 },
                    )?;
                    specs.push(spec);
                    Some((specs, spec.out_ch, spec.h_out, spec.w_out))
                },
            )
            .map(|(specs, ..)| specs)
    }
}

/// Run the conv subsampling stem and the `pre_encode_out` projection, producing
/// the `[t_out × n_embd]` sequence the Conformer stack consumes.
///
/// `Ok(None)` means there is nothing to encode (`n_frames == 0`). Split out from
/// [`encode_audio_mel_gpu`] so the stem has its own parity boundary against
/// `cpu::conv_stem_forward`, reachable through [`encoder_input_gpu`]. A
/// divergence here and one 17 blocks later are very different bugs, and an
/// end-to-end check alone cannot tell them apart.
fn conv_stem_gpu<O: AudioEncoderGpuOps>(
    ops: &O,
    gpu_w: &GpuAudioWeights<O>,
    mel: &[f32],
    n_frames: usize,
) -> Result<Option<(O::Buf, usize)>> {
    let cfg = &gpu_w.cfg;
    anyhow::ensure!(
        mel.len() == n_frames * cfg.n_mel_bins,
        "conv_stem_gpu: mel.len() {} != n_frames * n_mel_bins ({n_frames} * {})",
        mel.len(),
        cfg.n_mel_bins,
    );
    if n_frames == 0 {
        return Ok(None);
    }

    // Capacity check before any upload: the geometry walk is pure arithmetic, so
    // an over-long chunk costs nothing on its way to the CPU fallback. These are
    // the same specs the dispatch loop below uses, so the guard cannot drift
    // from the work it guards.
    let specs = gpu_w.stem_specs(n_frames).ok_or_else(|| {
        anyhow::anyhow!("conv_stem_gpu: stem cannot run on {n_frames} mel frames")
    })?;
    let last = specs
        .last()
        .ok_or_else(|| anyhow::anyhow!("conv_stem_gpu: conv stem has no layers"))?;
    let (t, f_out) = (last.h_out, last.w_out);
    anyhow::ensure!(t > 0 && t <= MAX_AUDIO_TOKENS, "{}", over_capacity_msg(t));

    let cur = gpu_w.stem.iter().zip(&specs).zip(STEM_RELU_AFTER).fold(
        ops.upload(mel),
        |input, ((layer, spec), relu)| {
            let out = ops.conv2d(&input, &layer.weight, &layer.bias, spec);
            if relu {
                ops.relu(&out, spec.out_len());
            }
            out
        },
    );

    let cur_ch = last.out_ch;
    let plane = cur_ch * f_out;
    anyhow::ensure!(
        plane == gpu_w.pre_encode_in_dim,
        "audio conv stem produced {cur_ch}×{f_out} = {plane} features but pre_encode_out \
         expects {}",
        gpu_w.pre_encode_in_dim,
    );

    // Permute (channel, time, freq) → (time, channel·freq), then project.
    let flat = ops.transpose_blocked(&cur, cur_ch, t, f_out);
    let x = ops.linear(&flat, &gpu_w.pre_encode_out_w, t, cfg.n_embd, plane);
    ops.bias_add(&x, &gpu_w.pre_encode_out_b, t, cfg.n_embd);
    Ok(Some((x, t)))
}

/// The conv stem's output, read back to the host: `(encoder_in [t_out × n_embd],
/// t_out)`, directly comparable to [`super::audio_encoder::conv_stem_forward`].
///
/// Exists for the parity suite. The forward pass keeps the buffer on the GPU.
pub fn encoder_input_gpu<O: AudioEncoderGpuOps>(
    ops: &O,
    gpu_w: &GpuAudioWeights<O>,
    mel: &[f32],
    n_frames: usize,
) -> Result<(Vec<f32>, usize)> {
    match conv_stem_gpu(ops, gpu_w, mel, n_frames)? {
        Some((x, t)) => Ok((ops.download(&x, t * gpu_w.cfg.n_embd), t)),
        None => Ok((Vec::new(), 0)),
    }
}

/// Run the Conformer encoder + MLP adapter on the GPU. Backend-agnostic: `ops`
/// provides the kernels, `gpu_w` the uploaded weights.
///
/// `mel` is `[n_frames × n_mel_bins]` row-major (time-major outer, freq inner),
/// the same layout [`super::audio_preprocessor::log_mel_spectrogram`] emits and
/// [`super::audio_encoder::audio_encoder_forward`] takes. Output is identical in
/// shape to that function's: `(embeddings [t_out × llm_hidden_size], t_out)`.
///
/// Returns an error (rather than falling back silently) when the chunk exceeds
/// the attention kernel's capacity, so the caller decides how to degrade.
pub fn encode_audio_mel_gpu<O: AudioEncoderGpuOps>(
    ops: &O,
    gpu_w: &GpuAudioWeights<O>,
    mel: &[f32],
    n_frames: usize,
) -> Result<(Vec<f32>, usize)> {
    let cfg = &gpu_w.cfg;
    let (n_embd, n_ff, n_head, eps) = (cfg.n_embd, cfg.n_ff, cfg.n_head, cfg.eps);
    let head_dim = n_embd / n_head;

    let Some((mut x, t)) = conv_stem_gpu(ops, gpu_w, mel, n_frames)? else {
        return Ok((Vec::new(), 0));
    };

    // ── Stage 2: relative-position embedding, built on the CPU once per chunk ──
    let seq_len = 2 * t - 1;
    let pos_emb = ops.upload(&relative_pos_emb(t));

    let x_len = t * n_embd;

    // ── Stage 3: Conformer block stack ──
    for blk in &gpu_w.blocks {
        // FFN-½ #1.
        macaron_ffn(
            ops,
            &x,
            (&blk.ffn_norm_w, &blk.ffn_norm_b),
            (&blk.ffn_up_w, &blk.ffn_up_b),
            (&blk.ffn_down_w, &blk.ffn_down_b),
            t,
            n_embd,
            n_ff,
            eps,
        );

        // Self-attention with relative-position bias.
        let normed = ops.layernorm(&x, &blk.ln1_w, &blk.ln1_b, eps, t, n_embd);
        let q = ops.linear(&normed, &blk.attn_q_w, t, n_embd, n_embd);
        ops.bias_add(&q, &blk.attn_q_b, t, n_embd);
        let k = ops.linear(&normed, &blk.attn_k_w, t, n_embd, n_embd);
        ops.bias_add(&k, &blk.attn_k_b, t, n_embd);
        let v = ops.linear(&normed, &blk.attn_v_w, t, n_embd, n_embd);
        ops.bias_add(&v, &blk.attn_v_b, t, n_embd);
        // `linear_pos` has no bias term in the reference.
        let p = ops.linear(&pos_emb, &blk.linear_pos_w, seq_len, n_embd, POS_EMB_DIM);
        let attn = ops.xl_attention(
            &q,
            &k,
            &v,
            &p,
            &blk.pos_bias_u,
            &blk.pos_bias_v,
            t,
            n_head,
            head_dim,
        );
        let proj = ops.linear(&attn, &blk.attn_o_w, t, n_embd, n_embd);
        ops.bias_add(&proj, &blk.attn_o_b, t, n_embd);
        ops.add(&x, &proj, x_len);

        // Convolution module.
        let normed = ops.layernorm(&x, &blk.norm_conv_w, &blk.norm_conv_b, eps, t, n_embd);
        let pw1 = ops.linear(&normed, &blk.conv_pw1_w, t, 2 * n_embd, n_embd);
        ops.bias_add(&pw1, &blk.conv_pw1_b, t, 2 * n_embd);
        let glu = ops.glu_split(&pw1, t, n_embd);
        // Depthwise conv1d wants channel-major; `K = 1` makes this a plain
        // transpose.
        let ch_major = ops.transpose_blocked(&glu, t, n_embd, 1);
        let pad_total = blk.conv_dw_k - 1;
        let pad_lo = pad_total / 2;
        let conv_spec = Conv2dSpec::padded(
            n_embd,
            n_embd,
            1,
            t,
            (1, blk.conv_dw_k),
            (1, 1),
            (0, 0),
            (pad_lo, pad_total - pad_lo),
            n_embd,
        )
        .ok_or_else(|| {
            anyhow::anyhow!(
                "audio conv module: degenerate depthwise conv (t {t}, k {})",
                blk.conv_dw_k,
            )
        })?;
        debug_assert_eq!(conv_spec.w_out, t, "conv module pad math drifted");
        let conv = ops.conv2d(&ch_major, &blk.conv_dw_w, &blk.conv_dw_b, &conv_spec);
        ops.chan_affine_silu(&conv, &blk.conv_norm_w, &blk.conv_norm_b, n_embd, t);
        let time_major = ops.transpose_blocked(&conv, n_embd, t, 1);
        let pw2 = ops.linear(&time_major, &blk.conv_pw2_w, t, n_embd, n_embd);
        ops.bias_add(&pw2, &blk.conv_pw2_b, t, n_embd);
        ops.add(&x, &pw2, x_len);

        // FFN-½ #2.
        macaron_ffn(
            ops,
            &x,
            (&blk.ffn_norm_1_w, &blk.ffn_norm_1_b),
            (&blk.ffn_up_1_w, &blk.ffn_up_1_b),
            (&blk.ffn_down_1_w, &blk.ffn_down_1_b),
            t,
            n_embd,
            n_ff,
            eps,
        );

        // Final per-block LayerNorm. No residual.
        x = ops.layernorm(&x, &blk.ln2_w, &blk.ln2_b, eps, t, n_embd);
    }

    // ── Stage 4: MLP adapter → LLM hidden size ──
    let n_ff_adapter = gpu_w.adapter_intermediate;
    let llm_hidden = cfg.llm_hidden_size;
    let normed = ops.layernorm(
        &x,
        &gpu_w.adapter_norm_w,
        &gpu_w.adapter_norm_b,
        eps,
        t,
        n_embd,
    );
    let mid = ops.linear(&normed, &gpu_w.adapter_up_w, t, n_ff_adapter, n_embd);
    ops.bias_add(&mid, &gpu_w.adapter_up_b, t, n_ff_adapter);
    ops.gelu_erf(&mid, t * n_ff_adapter);
    let out = ops.linear(&mid, &gpu_w.adapter_down_w, t, llm_hidden, n_ff_adapter);
    ops.bias_add(&out, &gpu_w.adapter_down_b, t, llm_hidden);

    Ok((ops.download(&out, t * llm_hidden), t))
}

/// One Conformer macaron feed-forward sub-block, accumulated into `x` at half
/// weight. Used twice per block (before attention and after the conv module)
/// with only the weights differing, exactly as on the CPU.
#[allow(clippy::too_many_arguments)]
fn macaron_ffn<O: AudioEncoderGpuOps>(
    ops: &O,
    x: &O::Buf,
    norm: (&O::Buf, &O::Buf),
    up: (&O::Weight, &O::Buf),
    down: (&O::Weight, &O::Buf),
    t: usize,
    n_embd: usize,
    n_ff: usize,
    eps: f32,
) {
    let normed = ops.layernorm(x, norm.0, norm.1, eps, t, n_embd);
    let mid = ops.linear(&normed, up.0, t, n_ff, n_embd);
    ops.bias_add(&mid, up.1, t, n_ff);
    ops.silu(&mid, t * n_ff);
    let out = ops.linear(&mid, down.0, t, n_embd, n_ff);
    ops.bias_add(&out, down.1, t, n_embd);
    ops.scaled_add(x, &out, t * n_embd, 0.5);
}

// ── Native Metal backend ─────────────────────────────────────────────────────

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use crate::backend::metal::params::{
    AudioXlAttnParams, Batch2dParams, BiasAddParams, Conv2dDirectParams, ElementwiseParams,
    LayerNormBatchParams, MetalParams, ScaleParams, TransposeBlockedParams,
};
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use crate::backend::metal::{MetalLinear, MetalLinearWeight};

/// Native-Metal implementation of [`AudioEncoderGpuOps`].
///
/// Each op runs in its own command buffer and blocks on `wait_until_completed`,
/// so `download` always sees current data (unified memory on Apple Silicon).
///
/// The trait is deliberately independent of `VitGpuOps` (see its doc), but the
/// *mechanics* are not encoder-specific: the command-buffer runner lives on
/// [`crate::backend::metal::MetalContext::run_kernel`] and the linear tier on
/// [`MetalLinear`], both shared with `MetalVitOps`.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub struct MetalAudioOps {
    ctx: crate::backend::metal::MetalContext,
    linear: MetalLinear,
    p_bias: metal::ComputePipelineState,
    p_layernorm: metal::ComputePipelineState,
    p_relu: metal::ComputePipelineState,
    p_silu: metal::ComputePipelineState,
    p_gelu_erf: metal::ComputePipelineState,
    p_add: metal::ComputePipelineState,
    p_scaled_add: metal::ComputePipelineState,
    p_conv2d: metal::ComputePipelineState,
    p_transpose: metal::ComputePipelineState,
    p_glu: metal::ComputePipelineState,
    p_chan_affine: metal::ComputePipelineState,
    p_attn: metal::ComputePipelineState,
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl MetalAudioOps {
    pub fn new(ctx: crate::backend::metal::MetalContext) -> Result<Self> {
        use crate::backend::metal::shaders;
        Ok(Self {
            linear: MetalLinear::new(&ctx)?,
            p_bias: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add")?,
            p_layernorm: ctx.create_pipeline(shaders::LAYERNORM_BATCH, "layernorm_batch")?,
            p_relu: ctx.create_pipeline(shaders::ACTIVATIONS, "relu_inplace")?,
            p_silu: ctx.create_pipeline(shaders::ACTIVATIONS, "silu_inplace")?,
            p_gelu_erf: ctx.create_pipeline(shaders::ACTIVATIONS, "gelu_erf_inplace")?,
            p_add: ctx.create_pipeline(shaders::ELEMENTWISE_SLANG, "add_inplace")?,
            p_scaled_add: ctx.create_pipeline(shaders::ELEMENTWISE_SLANG, "scaled_add_inplace")?,
            p_conv2d: ctx.create_pipeline(shaders::CONV2D_DIRECT, "conv2d_direct")?,
            p_transpose: ctx.create_pipeline(shaders::TRANSPOSE_BLOCKED, "transpose_blocked")?,
            p_glu: ctx.create_pipeline(shaders::GLU_SPLIT, "glu_split")?,
            p_chan_affine: ctx.create_pipeline(shaders::CHAN_AFFINE_SILU, "chan_affine_silu")?,
            p_attn: ctx.create_pipeline(shaders::AUDIO_XL_ATTENTION, "audio_xl_attention")?,
            ctx,
        })
    }

    /// An uninitialized `len`-element f32 buffer. Inherent rather than a trait
    /// method: every producing op needs it, but it is allocation plumbing, not
    /// something the backend-agnostic driver ever calls. `MetalVitOps` calls
    /// `ctx.create_buffer` inline for the same reason.
    fn alloc(&self, len: usize) -> metal::Buffer {
        self.ctx.create_buffer((len * 4) as u64)
    }

    /// One thread per element, 256 to a threadgroup: the dispatch shape every
    /// element-wise kernel in this module shares.
    fn run_flat<P: MetalParams>(
        &self,
        pipe: &metal::ComputePipelineState,
        bufs: &[&metal::Buffer],
        params: &P,
        len: usize,
    ) {
        self.ctx.run_kernel(
            pipe,
            bufs,
            params,
            metal::MTLSize::new((len as u64).div_ceil(256), 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl AudioEncoderGpuOps for MetalAudioOps {
    type Buf = metal::Buffer;
    type Weight = MetalLinearWeight;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        self.ctx.upload_f32(data)
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        self.ctx.read_f32(buf, len)
    }

    /// The LFM2A mmproj ships every linear as Q4_0, so this takes the packed
    /// simdgroup-GEMM path in practice.
    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight {
        self.ctx.upload_linear_weight(w)
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        self.linear.forward(&self.ctx, x, w, rows, out_dim, in_dim)
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        let total = rows * dim;
        let params = BiasAddParams {
            total: total as u32,
            dim: dim as u32,
        };
        self.run_flat(&self.p_bias, &[x, bias], &params, total);
    }

    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        let dst = self.alloc(rows * dim);
        let params = LayerNormBatchParams {
            n: dim as u32,
            eps_bits: eps.to_bits(),
            src_stride: dim as u32,
            dst_stride: dim as u32,
        };
        self.ctx.run_kernel(
            &self.p_layernorm,
            &[src, &dst, weight, bias],
            &params,
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        dst
    }

    fn relu(&self, x: &Self::Buf, len: usize) {
        self.run_flat(&self.p_relu, &[x], &ElementwiseParams::new(len as u32), len);
    }

    fn silu(&self, x: &Self::Buf, len: usize) {
        self.run_flat(&self.p_silu, &[x], &ElementwiseParams::new(len as u32), len);
    }

    fn gelu_erf(&self, x: &Self::Buf, len: usize) {
        self.run_flat(
            &self.p_gelu_erf,
            &[x],
            &ElementwiseParams::new(len as u32),
            len,
        );
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        self.run_flat(
            &self.p_add,
            &[dst, src],
            &ElementwiseParams::new(len as u32),
            len,
        );
    }

    fn scaled_add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize, scale: f32) {
        let params = ScaleParams {
            n: len as u32,
            scale_bits: scale.to_bits(),
        };
        self.run_flat(&self.p_scaled_add, &[dst, src], &params, len);
    }

    fn conv2d(
        &self,
        input: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        spec: &Conv2dSpec,
    ) -> Self::Buf {
        let total = spec.out_len();
        let out = self.alloc(total);
        let params = Conv2dDirectParams {
            in_ch: spec.in_ch as u32,
            out_ch: spec.out_ch as u32,
            h_in: spec.h_in as u32,
            w_in: spec.w_in as u32,
            kh: spec.kh as u32,
            kw: spec.kw as u32,
            stride_h: spec.stride_h as u32,
            stride_w: spec.stride_w as u32,
            pad_h: spec.pad_h as u32,
            pad_w: spec.pad_w as u32,
            h_out: spec.h_out as u32,
            w_out: spec.w_out as u32,
            groups: spec.groups as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.run_flat(&self.p_conv2d, &[input, weight, bias, &out], &params, total);
        out
    }

    fn transpose_blocked(&self, src: &Self::Buf, a: usize, b: usize, k: usize) -> Self::Buf {
        let total = a * b * k;
        let dst = self.alloc(total);
        let params = TransposeBlockedParams {
            a: a as u32,
            b: b as u32,
            k: k as u32,
            _pad: 0,
        };
        self.run_flat(&self.p_transpose, &[src, &dst], &params, total);
        dst
    }

    fn glu_split(&self, src: &Self::Buf, rows: usize, n: usize) -> Self::Buf {
        let total = rows * n;
        let dst = self.alloc(total);
        let params = Batch2dParams::new(rows as u32, n as u32);
        self.run_flat(&self.p_glu, &[src, &dst], &params, total);
        dst
    }

    fn chan_affine_silu(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        b: &Self::Buf,
        channels: usize,
        t: usize,
    ) {
        let total = channels * t;
        let params = Batch2dParams::new(channels as u32, t as u32);
        self.run_flat(&self.p_chan_affine, &[x, w, b], &params, total);
    }

    fn xl_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        p: &Self::Buf,
        bias_u: &Self::Buf,
        bias_v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf {
        let out = self.alloc(tokens * n_head * head_dim);
        let params = AudioXlAttnParams {
            tokens: tokens as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            scale_bits: (1.0f32 / (head_dim as f32).sqrt()).to_bits(),
        };
        self.ctx.run_kernel(
            &self.p_attn,
            &[q, k, v, p, bias_u, bias_v, &out],
            &params,
            metal::MTLSize::new(tokens as u64, n_head as u64, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        out
    }
}

// ── Cached, object-safe encoder for the live session path ────────────────────

/// Object-safe GPU audio encoder cached in a [`crate::session::Session`].
///
/// Takes **raw PCM**, not a mel spectrogram, so that moving the log-mel
/// front-end onto the GPU later is entirely below this boundary and no caller
/// changes. Implementors are `Send + Sync` so the engine can share one across
/// sessions.
pub trait AudioGpuEncode: Send + Sync {
    /// Encode mono PCM at [`super::audio_encoder::SAMPLE_RATE`] into per-frame
    /// LLM-hidden-size embeddings. Output matches
    /// [`super::audio_encoder::encode_audio_pcm`]: `(embeddings, t_out)`.
    fn encode_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)>;
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
struct MetalAudioEncoder {
    ops: MetalAudioOps,
    weights: GpuAudioWeights<MetalAudioOps>,
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl AudioGpuEncode for MetalAudioEncoder {
    fn encode_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        // Decide capacity from the frame count alone, before computing the
        // spectrogram. A refusal sends the caller to the CPU encoder, which runs
        // its own `log_mel_spectrogram`; computing it here first would mean
        // paying for the STFT twice, and the refusal case is by definition the
        // longest utterances, where it is most expensive.
        let n_frames = crate::model::audio_preprocessor::n_frames_for(pcm.len());
        if n_frames == 0 {
            return Ok((Vec::new(), 0));
        }
        let t_out = self.weights.predict_t_out(n_frames).ok_or_else(|| {
            anyhow::anyhow!("audio encoder: conv stem cannot run on {n_frames} mel frames")
        })?;
        anyhow::ensure!(
            t_out > 0 && t_out <= MAX_AUDIO_TOKENS,
            "{}",
            over_capacity_msg(t_out)
        );

        let (mel, mel_frames) =
            crate::model::audio_preprocessor::log_mel_spectrogram(pcm, self.weights.cfg.n_mel_bins);
        anyhow::ensure!(
            mel_frames == n_frames,
            "audio encoder: n_frames_for predicted {n_frames} frames but the \
             spectrogram has {mel_frames}",
        );
        encode_audio_mel_gpu(&self.ops, &self.weights, &mel, n_frames)
    }
}

/// Build a cached GPU audio encoder for `weights`, honoring `backend`.
///
/// Returns `None` for `Cpu`, when the chosen backend's feature isn't compiled,
/// or when the device/context can't be created. The caller then uses the CPU
/// encoder. `Auto` prefers Metal. wgpu is not wired yet (its kernels ship and are
/// parity-tested, but the ops impl does not exist), so `Gpu` yields `None`.
pub fn build_gpu_audio_encoder(
    weights: &AudioEncoderWeights,
    backend: crate::engine::BackendPreference,
) -> Option<std::sync::Arc<dyn AudioGpuEncode>> {
    use crate::engine::BackendPreference as BP;
    match backend {
        BP::Cpu | BP::Gpu => None,
        BP::Metal | BP::Auto => try_metal_audio_encoder(weights),
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn try_metal_audio_encoder(
    weights: &AudioEncoderWeights,
) -> Option<std::sync::Arc<dyn AudioGpuEncode>> {
    let ctx = crate::backend::metal::MetalContext::new().ok()?;
    let ops = MetalAudioOps::new(ctx).ok()?;
    let gpu_w = match GpuAudioWeights::build(&ops, weights) {
        Ok(w) => w,
        Err(e) => {
            // A model this path cannot run correctly is a fall-back-to-CPU, not a
            // load failure, but it must be visible, since the only other symptom
            // is "audio encode is slow".
            tracing::warn!("audio encoder: Metal backend unavailable for this model: {e:#}");
            return None;
        }
    };
    tracing::info!("audio encoder: using native Metal backend");
    Some(std::sync::Arc::new(MetalAudioEncoder {
        ops,
        weights: gpu_w,
    }))
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn try_metal_audio_encoder(
    _weights: &AudioEncoderWeights,
) -> Option<std::sync::Arc<dyn AudioGpuEncode>> {
    None
}

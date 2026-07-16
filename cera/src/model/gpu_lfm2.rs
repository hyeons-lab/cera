// GPU-accelerated LFM2 forward pass using wgpu compute shaders.
//
// All weights are dequantized to f32 at load time and uploaded to GPU buffers.
// The full forward pass runs in a single CommandEncoder per token — only the
// logits vector is read back to CPU.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::Result;

use crate::backend::cpu::RopeType;
use crate::backend::wgpu::{GpuContext, GpuTensor, KvShiftParams, shaders};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvPrefixCache, LayerSnapshot, StateSnapshot};
use crate::lora::{LoraAdapterWeights, LoraTarget};
use crate::model::gpu_weight_source::GpuWeightSource;
use crate::model::transformer::WeightRef;
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use crate::tensor::DType;

/// Maximum N for a single batched-prefill dispatch. Mirrors the Metal
/// backend's `MAX_PREFILL_TOKENS = 512`. Prompts longer than this are
/// chunked at the host side; each chunk shares the same prefill batch
/// scratch, so the worst-case scratch footprint is bounded.
const MAX_PREFILL_TOKENS: usize = 512;

// Tile geometry for the register-tiled matmul pipeline. The shader
// receives these via preprocessor #defines below; keeping a single
// source of truth here means dispatch geometry can never drift out of
// sync with the kernel.
const MUL_MAT_TILE_WG_M: u32 = 8;

/// Rows emitted per workgroup by `gemv_f32` / `gemv_f32_accum` — MUST match the
/// `NR` constant in `gemv_f32.wgsl`. Used to size the LoRA dispatch grids.
const GEMV_F32_ROWS_PER_WG: u32 = 8;
const MUL_MAT_TILE_WG_N: u32 = 32;
// Each thread computes an 8×4 register tile. shmem for the K-quant variant is
// `(TILE_K+1)·WG_M·TILE_M + TILE_K·WG_N·TILE_N` f32 = (33·64 + 32·128)·4 ≈ 24.8 KB,
// which EXCEEDS the default WebGPU 16 KB workgroup-storage limit — it fits only
// because we request the adapter's actual limits (`GpuContext::new` uses
// `adapter.limits()`; PowerVR/Adreno/Mali expose 32 KB). Raising TILE_N or TILE_M
// further must re-check `max_compute_workgroup_storage_size`, or pipeline creation
// fails at runtime on the target GPU.
const MUL_MAT_TILE_M: u32 = 8;
const MUL_MAT_TILE_N: u32 = 4;
const MUL_MAT_TILE_K: u32 = 32;

/// Build a `mul_mat_reg_tile` pipeline for the requested variant.
///
/// `use_vec` enables vec4 loads/stores (requires the matrix dimensions and
/// effective row strides used by each dispatch to be multiples of 4).
/// `src0_loader` selects the shmem dequant path — one of
/// `"INIT_SRC0_SHMEM_{Q4_0,Q8_0,Q4_K,Q6_K}"`. The rest of the kernel is
/// dtype-agnostic: the loader decodes weights to f32 in shared memory once per
/// k-tile and the register-tiled inner loop reuses them across all
/// `WORKGROUP_SIZE_N` token columns. That reuse is the entire reason this kernel
/// beats the batched-GEMV-shaped `gemm_*` kernels, which re-dequantize per token.
fn build_mul_mat_pipeline(
    ctx: &GpuContext,
    label: &str,
    use_vec: bool,
    src0_loader: &str,
) -> wgpu::ComputePipeline {
    let wg_m = format!("{MUL_MAT_TILE_WG_M}u");
    let wg_n = format!("{MUL_MAT_TILE_WG_N}u");
    let tile_m = format!("{MUL_MAT_TILE_M}u");
    let tile_n = format!("{MUL_MAT_TILE_N}u");
    let tile_k = format!("{MUL_MAT_TILE_K}u");
    let variant = if use_vec { "VEC" } else { "SCALAR" };
    ctx.create_pipeline_with_defines(
        shaders::MUL_MAT_REG_TILE,
        "main",
        label,
        &[
            (variant, ""),
            ("SRC0_INNER_TYPE", "u32"),
            ("SRC1_INNER_TYPE", "f32"),
            (src0_loader, ""),
            ("INIT_SRC1_SHMEM_FLOAT", ""),
            ("WORKGROUP_SIZE_M", &wg_m),
            ("WORKGROUP_SIZE_N", &wg_n),
            ("TILE_M", &tile_m),
            ("TILE_N", &tile_n),
            ("TILE_K", &tile_k),
        ],
    )
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn lcm_u64(a: u64, b: u64) -> u64 {
    (a / gcd_u64(a, b)) * b
}

/// Rows per tile for the tiled LM-head GEMV, so each tile's weight sub-binding
/// fits `max_binding` and starts at a `offset_alignment`-aligned byte offset.
/// `elem_size` is the weight element size (4 for f32, 2 for f16).
fn gemv_tile_rows(m: u32, k: u32, max_binding: u64, offset_alignment: u64, elem_size: u64) -> u32 {
    const ROWS_PER_WG: u64 = 8;

    let row_bytes = u64::from(k) * elem_size;
    // Round to a whole u32: the weight is bound as `array<u32>`, so the true
    // binding size is padded up (matches `encode_gemv_f16`'s tiled/non-tiled
    // decision). Keeps the two "fits one binding" checks in agreement.
    let full_bytes = (u64::from(m) * row_bytes).div_ceil(4) * 4;
    if full_bytes <= max_binding {
        return m;
    }

    let max_rows = (max_binding / row_bytes) as u32;
    assert!(
        max_rows > 0,
        "GPU max storage binding size {} is too small for one GEMV row of {} bytes",
        max_binding,
        row_bytes
    );

    let offset_alignment = offset_alignment.max(elem_size.max(4));
    let row_alignment = (offset_alignment / gcd_u64(row_bytes, offset_alignment)).max(1) as u32;
    let tile_alignment = lcm_u64(u64::from(row_alignment), ROWS_PER_WG) as u32;
    let tile_rows = if max_rows >= tile_alignment {
        max_rows - (max_rows % tile_alignment)
    } else if max_rows >= row_alignment {
        max_rows - (max_rows % row_alignment)
    } else {
        max_rows
    };
    assert!(
        tile_rows > 0 && (u64::from(tile_rows) * row_bytes) % offset_alignment == 0,
        "GPU storage binding alignment {} cannot be satisfied for GEMV rows of {} bytes",
        offset_alignment,
        row_bytes
    );
    tile_rows
}

/// A weight matrix on GPU — tracks buffer + dtype + pre-allocated params for dispatch.
struct GpuWeight {
    tensor: GpuTensor,
    /// Pre-allocated params buffer with [m, k, row_base, 0] — eliminates per-dispatch allocation.
    params_buf: wgpu::Buffer,
    /// Pre-created bind group for this weight's primary GEMV dispatch.
    /// Created after all scratch buffers are allocated, to avoid per-token
    /// create_bind_group overhead (~16 µs each, 300×/token = 4.8 ms).
    cached_bg: Option<wgpu::BindGroup>,
}

/// GPU buffer handles for one layer's weights.
/// Q4_0/Q8_0 weights are uploaded quantized; f32 norms uploaded as-is.
struct GpuLayerWeights {
    attn_norm: wgpu::Buffer,
    ffn_norm: wgpu::Buffer,
    ffn_gate: GpuWeight,
    ffn_up: GpuWeight,
    ffn_down: GpuWeight,
    // Conv-specific
    conv_in_proj: Option<GpuWeight>,
    conv_out_proj: Option<GpuWeight>,
    conv_weight: Option<wgpu::Buffer>,
    // Attention-specific
    attn_q: Option<GpuWeight>,
    attn_k: Option<GpuWeight>,
    attn_v: Option<GpuWeight>,
    attn_output: Option<GpuWeight>,
    attn_q_norm: Option<wgpu::Buffer>,
    attn_k_norm: Option<wgpu::Buffer>,
    // Qwen2 Q/K/V projection biases (f32), added after each projection GEMV.
    // `None` for archs without QKV bias.
    attn_q_bias: Option<wgpu::Buffer>,
    attn_k_bias: Option<wgpu::Buffer>,
    attn_v_bias: Option<wgpu::Buffer>,
}

/// Compute pipelines for all shader entry points.
#[allow(dead_code)]
struct GpuPipelines {
    gemv_f32: wgpu::ComputePipeline,
    /// `gemv_f32` compiled with `F16_A` — reads the weight matrix as f16 (2 per
    /// u32) instead of f32. Serves the f16 LM head (embedding / output.weight)
    /// on the logit-projection path; activations and accumulation stay f32.
    gemv_f16: wgpu::ComputePipeline,
    /// `y[row] += dot(A[row,:], x)` — the accumulate epilogue for the LoRA
    /// up-projection (`out += B_scaled·tmp`).
    gemv_f32_accum: wgpu::ComputePipeline,
    /// Batched NT GEMM `C[M×N] = Lhs[M×K]·Rhs[N×K]ᵀ` (overwrite) — the LoRA
    /// down-projection (`Tmp[n×rank] = X·Aᵀ`) in the batched prefill path.
    gemm_f32_nt: wgpu::ComputePipeline,
    /// Accumulate variant `C += Lhs·Rhsᵀ` — the LoRA up-projection epilogue
    /// (`Y[n×d] += Tmp·B_batchedᵀ`) in the batched prefill path.
    gemm_f32_nt_accum: wgpu::ComputePipeline,
    gemv_q4_0: wgpu::ComputePipeline,
    gemv_q4_0_fast: wgpu::ComputePipeline,
    gemv_q4_k: wgpu::ComputePipeline,
    gemv_q5_k: wgpu::ComputePipeline,
    gemv_q6_k: wgpu::ComputePipeline,
    gemv_q8_0: wgpu::ComputePipeline,
    add_inplace: wgpu::ComputePipeline,
    /// Residual add with a scalar on the addend (`a += s*b`). Used for the
    /// attention/FFN residual adds so Granite's residual multiplier folds in;
    /// `s = 1.0` for every other arch.
    scaled_add_inplace: wgpu::ComputePipeline,
    /// In-place scale by a constant (`a *= s`). Granite logit/residual scalars.
    scale_f32: wgpu::ComputePipeline,
    mul_inplace: wgpu::ComputePipeline,
    silu_mul_inplace: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    per_head_rmsnorm: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    /// n_keep context shift: re-rotate retained K cells by `R(-shift)` into
    /// scratch (the memcpy halves use `copy_buffer_to_buffer`). See `shift_kv`.
    kv_shift: wgpu::ComputePipeline,
    flash_attention: wgpu::ComputePipeline,
    conv1d_fused: wgpu::ComputePipeline,
    argmax_f32: wgpu::ComputePipeline,
    // ── Batched-prefill pipelines ─────────────────────────────────────
    rmsnorm_batch: wgpu::ComputePipeline,
    add_rmsnorm_batch: wgpu::ComputePipeline,
    qk_norm_rope_batch: wgpu::ComputePipeline,
    conv1d_fused_batch: wgpu::ComputePipeline,
    /// Broadcast bias add for batched prefill (`x[t*dim+j] += bias[j]`). Qwen2
    /// QKV bias; absent on every other arch.
    bias_add: wgpu::ComputePipeline,

    mul_mat_reg_tile_q4_0_vec: wgpu::ComputePipeline,
    mul_mat_reg_tile_q4_0_scalar: wgpu::ComputePipeline,
    mul_mat_reg_tile_q8_0: wgpu::ComputePipeline,
    mul_mat_reg_tile_q4_k: wgpu::ComputePipeline,
    mul_mat_reg_tile_q6_k: wgpu::ComputePipeline,
    attention_prefill: wgpu::ComputePipeline,
}

/// One LoRA target's low-rank factors uploaded to GPU. The apply is two GEMV
/// dispatches: `tmp = A·x` (`gemv_f32`, `m=rank`) then `out += B_scaled·tmp`
/// (`gemv_f32_accum`, `m=d`). `scale = alpha/rank` is pre-folded into
/// `b_scaled` at upload, so the runtime path has no separate scale pass.
struct WgpuLoraTarget {
    /// Down-projection `A`, `[rank × k]` row-major (f32).
    a: wgpu::Buffer,
    /// Up-projection `scale · B`, `[d × rank]` row-major (f32). For the
    /// residual-fed targets (attn-output / ffn-down) this also folds the model's
    /// `residual_mult`, because the **decode** path adds this delta straight into
    /// the post-residual hidden state (see [`WgpuLoraAdapter::upload`]).
    b_scaled: wgpu::Buffer,
    /// Up-projection `scale · B` **without** the `residual_mult` fold, for the
    /// batched-prefill path. There the LoRA delta is accumulated into the
    /// projection scratch *before* the fused residual add (`add_rmsnorm_batch` /
    /// `scaled_add_inplace`) scales it by `residual_mult` — so folding
    /// `residual_mult` here too would double-apply it (Granite only; identical to
    /// `b_scaled` for every other arch, where `residual_mult == 1.0`). Matches the
    /// CPU `lora::apply_prefill`, which uses a scale-only `B` and lets the model's
    /// residual scale wrap the delta.
    b_batched: wgpu::Buffer,
    /// `[rank, k, 0, 0]` params for the `A·x` GEMV.
    a_params: wgpu::Buffer,
    /// `[d, rank, 0, 0]` params for the `B_scaled·tmp` GEMV.
    b_params: wgpu::Buffer,
    rank: u32,
    #[allow(dead_code)]
    k: u32,
    d: u32,
}

/// A LoRA adapter uploaded to GPU: per-layer, per-target (in `LoraTarget::index`
/// order) low-rank factors. Built from a CPU [`LoraAdapterWeights`] via
/// [`WgpuLoraAdapter::upload`] and cached on the model (Arc-pointer-keyed LRU).
struct WgpuLoraAdapter {
    layers: Vec<[Option<WgpuLoraTarget>; 9]>,
}

impl WgpuLoraAdapter {
    /// Upload every `(layer, target)` factor pair to GPU buffers, folding
    /// `scale` into `B` as it goes. Adapters are tiny (rank ≤ ~64), so the f32
    /// copy through `upload_f32` is negligible.
    ///
    /// `residual_mult` is the model's residual multiplier (`scalars.residual`,
    /// 1.0 for all archs except Granite). The base attn-output / ffn-down
    /// projections feed the residual `scaled_add_inplace`, which scales their
    /// result by `residual_mult` before the residual add — so those two targets'
    /// LoRA delta must carry the same factor (folded into `B` here). The other
    /// seven targets (incl. the shortconv projections, whose out_proj folds into
    /// the residual via a plain `add_inplace`, not the scaled path) use `scale`
    /// alone.
    fn upload(ctx: &GpuContext, w: &LoraAdapterWeights, residual_mult: f32) -> Self {
        let mut layers = Vec::with_capacity(w.n_layers());
        for layer in 0..w.n_layers() {
            let mut targets: [Option<WgpuLoraTarget>; 9] = Default::default();
            for target in LoraTarget::ALL {
                let Some(t) = w.get(layer, target) else {
                    continue;
                };
                let rank = t.rank as u32;
                let k = t.k as u32;
                let d = t.d as u32;
                // Fold scale into B at upload → no runtime scale dispatch. For the
                // residual-fed targets, also fold `residual_mult` (matches the
                // base projection's residual `scaled_add_inplace`; no-op unless
                // Granite).
                let b_factor = match target {
                    LoraTarget::AttnOutput | LoraTarget::FfnDown => t.scale * residual_mult,
                    _ => t.scale,
                };
                let b_scaled_data: Vec<f32> = t.b.iter().map(|&x| x * b_factor).collect();
                let b_scaled = ctx.upload_f32(&b_scaled_data, "lora_b_scaled");
                // Batched-prefill B: scale only (no residual_mult fold — the fused
                // residual add scales the delta afterward). Byte-identical to
                // `b_scaled` unless this is a residual-fed target on Granite
                // (`residual_mult != 1`); in the common case share the buffer (a
                // cheap `Arc` clone) instead of a duplicate upload.
                let b_batched = if b_factor == t.scale {
                    b_scaled.clone()
                } else {
                    ctx.upload_f32(
                        &t.b.iter().map(|&x| x * t.scale).collect::<Vec<f32>>(),
                        "lora_b_batched",
                    )
                };
                targets[target.index()] = Some(WgpuLoraTarget {
                    a: ctx.upload_f32(&t.a, "lora_a"),
                    b_scaled,
                    b_batched,
                    a_params: ctx
                        .upload_storage(bytemuck::cast_slice(&[rank, k, 0, 0]), "lora_a_p"),
                    b_params: ctx
                        .upload_storage(bytemuck::cast_slice(&[d, rank, 0, 0]), "lora_b_p"),
                    rank,
                    k,
                    d,
                });
            }
            layers.push(targets);
        }
        Self { layers }
    }
}

/// GPU-resident inference state (KV cache + conv rolling buffers).
#[allow(dead_code)]
struct GpuState {
    /// Per attention layer: (key_cache, value_cache) buffers, pre-allocated.
    kv_caches: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    /// Per conv layer: rolling buffer.
    conv_buffers: Vec<Option<wgpu::Buffer>>,
    seq_len: AtomicUsize,
    max_seq_len: usize,
    /// Pre-dequantized embedding rows (CPU-side cache for fast lookup).
    embedding_f32: Vec<f32>,
}

/// Scratch KV/conv caches for [`GpuLfm2Model::hidden_states`], mirroring the
/// generation caches' shapes. Allocated **lazily** on first `hidden_states` call
/// (via `OnceLock`) so a generation-only load never pays the extra VRAM.
/// Selected over the generation caches by `use_hs_scratch`.
struct HsScratch {
    kv: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
    conv: Vec<Option<wgpu::Buffer>>,
}

/// Clears `GpuLfm2Model::active_lora` when dropped, so a leaked `Some` can't
/// send a later base-model forward through the adapter. Mirrors the Metal
/// `LoraGuard`.
struct LoraGuard<'a>(&'a Mutex<Option<Arc<WgpuLoraAdapter>>>);

impl Drop for LoraGuard<'_> {
    fn drop(&mut self) {
        *self.0.lock().expect("active_lora poisoned") = None;
    }
}

/// GPU-accelerated LFM2 model.
///
/// NOTE: This model is stateful — KV caches and conv rolling buffers live on
/// the GPU and persist across forward() calls. This is inherent to GPU backends
/// (GPU-resident state can't live in the CPU-side InferenceState). Consequence:
/// one GpuLfm2Model instance = one session for throughput. The internal
/// `infer_lock` makes the backend self-defending: two `Session`s sharing this
/// `Arc<dyn Model>` and running `forward()` / `forward_prefill()` concurrently
/// will serialize cleanly on the lock instead of racing on per-instance scratch
/// buffers + GPU KV caches. For genuine throughput across concurrent Sessions,
/// create multiple model instances.
pub struct GpuLfm2Model {
    ctx: GpuContext,
    config: ModelConfig,
    pipelines: GpuPipelines,
    // GPU weight buffers
    embedding: wgpu::Buffer,
    #[allow(dead_code)]
    embedding_params: wgpu::Buffer,
    /// Separate output projection (`output.weight`), dequantized to f32, when
    /// the model has untied embeddings. `None` ⇒ the logit projection reuses
    /// `embedding` (tied embeddings — LFM2, Qwen, Llama-3.2, Granite).
    output_weight: Option<wgpu::Buffer>,
    output_norm: wgpu::Buffer,
    layers: Vec<GpuLayerWeights>,
    /// RoPE pair layout for this model (`Neox` LFM2/Qwen, `Norm` Llama family).
    rope_type: RopeType,
    /// Granite 3.x scalar multipliers (identity for every other arch). The
    /// embedding multiplier is pre-folded into `gpu_state.embedding_f32`; the
    /// residual/attention/logit multipliers are applied during the forward pass.
    scalars: ScalarMultipliers,
    /// Whether the batched-prefill GPU path is enabled (LFM2 only today; the
    /// dense transformers prefill via the per-token decode loop).
    batched_prefill: bool,
    /// Latches once the "no batched GEMM for this dtype" warning has been emitted,
    /// so a long generation doesn't repeat it on every `forward_prefill`.
    batched_fallback_warned: AtomicBool,
    /// Llama-3 RoPE frequency factors (`rope_freqs.weight`), or a 1-element
    /// dummy when the model uses plain RoPE. Always bound (binding 3) on the
    /// decode rope dispatch; `has_freq_factors` in `rope_params` gates its use.
    rope_freqs_buf: wgpu::Buffer,
    has_freq_factors: bool,
    // GPU scratch buffers (reused across layers)
    hidden_buf: wgpu::Buffer,    // [hidden_size]
    normed_buf: wgpu::Buffer,    // [hidden_size]
    ffn_input_buf: wgpu::Buffer, // [hidden_size]
    gate_buf: wgpu::Buffer,      // [intermediate_size]
    up_buf: wgpu::Buffer,        // [intermediate_size]
    out_buf: wgpu::Buffer,       // [hidden_size]
    q_buf: wgpu::Buffer,         // [hidden_size]
    k_buf: wgpu::Buffer,         // [max_kv_dim]
    v_buf: wgpu::Buffer,         // [max_kv_dim]
    /// Scratch for the n_keep KV shift: holds the re-rotated retained K (and,
    /// in a second pass, the moved V) for one layer before it is copied back
    /// into the cache. Sized `[max_seq_len × max_kv_dim]` f32. See `shift_kv`.
    kv_shift_scratch: wgpu::Buffer,
    attn_out_buf: wgpu::Buffer, // [hidden_size]
    logits_buf: wgpu::Buffer,   // [vocab_size]
    /// 4 bytes — receives argmax(logits) as a single u32. Cached so
    /// `forward_greedy` doesn't allocate per call. The `download_u32`
    /// readback over this 4-byte buffer is the wasm-async-friendly
    /// replacement for downloading `vocab_size * 4` bytes of logits.
    argmax_out_buf: wgpu::Buffer,
    /// Pre-uploaded `vec2<u32>{ vocab_size, 0 }` for the argmax shader.
    /// Held to keep the buffer alive for the cached `argmax_bg`'s
    /// reference; not directly read after construction.
    #[allow(dead_code)]
    argmax_params: wgpu::Buffer,
    /// Cached bind group for the argmax kernel — bindings never change
    /// (logits_buf, argmax_out_buf, argmax_params), so build it once.
    argmax_bg: wgpu::BindGroup,
    // Pre-allocated shader params (avoids upload_storage per dispatch).
    rmsnorm_hs_params: wgpu::Buffer,     // [hs, eps_bits, 0, 0]
    elementwise_hs_params: wgpu::Buffer, // [hs, 0]
    elementwise_is_params: wgpu::Buffer, // [intermediate_size, 0]
    /// `[n_heads*head_dim, 0]` — Q bias add length (= hs when head_dim=hs/n_heads).
    elementwise_qdim_params: wgpu::Buffer,
    /// `[n_kv_heads*head_dim, 0]` — K/V bias add length.
    elementwise_kvdim_params: wgpu::Buffer,
    /// `[hs, residual_scale_bits]` — addend scalar for the attention/FFN
    /// residual `scaled_add_inplace` (Granite residual multiplier; 1.0 else).
    residual_add_params: wgpu::Buffer,
    /// `[vocab_size, (1/logit_scale)_bits]` — Granite logit-scale divide, applied
    /// via `scale_f32` after the LM head. `None` when logit_scale == 1.0.
    logit_scale_params: Option<wgpu::Buffer>,
    conv1d_params: wgpu::Buffer,        // [hs, kernel_size, d_conv, 0]
    per_head_norm_params: wgpu::Buffer, // [head_dim, eps_bits, 0, 0]
    // [pos, n_heads, n_kv_heads, head_dim, theta_bits, rope_type, has_freq_factors]
    // — 7 u32, updated per token; must stay in sync with rope.wgsl's params array.
    rope_params: wgpu::Buffer,
    attn_params: wgpu::Buffer, // [n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale, 0, 0] — updated per token
    gemv_tile_params: Vec<wgpu::Buffer>, // [rows, k, row_base, 0] per output-projection tile
    // Conv scratch
    conv_proj_buf: wgpu::Buffer, // [3 × hidden_size]
    conv_gate_buf: wgpu::Buffer, // [hidden_size] — fused conv writes here, out_proj reads
    // ── Batched-prefill scratch (sized to MAX_PREFILL_TOKENS rows) ────────
    // Mirrors MetalLfm2Model's prefill_*_buf set. Used only by the batched
    // prefill path; the per-token forward path keeps using the scalar
    // scratch buffers above.
    /// `[MAX_PREFILL_TOKENS × hidden_size]` — running residual-stream
    /// activation across layers. Last token's slice is the final input
    /// to the output norm/projection.
    prefill_batch_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × hidden_size]` — post-rmsnorm activations,
    /// also reused as the attention output sink and as the conv1d output.
    prefill_normed_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × 3 × hidden_size]` — sized to fit the
    /// largest batched projection. For attention layers it's split into
    /// Q (offset 0, stride hs); the K/V projections land in the gate/up
    /// scratches because `mul_mat_reg_tile` writes contiguous token rows. For conv
    /// layers the full `3 × hs` slab is the in-projection target.
    prefill_proj_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × intermediate_size]` — FFN gate output;
    /// also reused as scratch for K projections and per-(layer,FFN)
    /// add-residual targets.
    prefill_gate_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × intermediate_size]` — FFN up output;
    /// also reused as scratch for V projections.
    prefill_up_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × n_heads × max_seq_len]` — per-(query,
    /// head) scratch slab consumed by `attention_prefill.wgsl`.
    /// Allocated once; sized to the worst case per the model config.
    prefill_scores_buf: wgpu::Buffer,
    // GPU state
    gpu_state: GpuState,
    /// Serializes Model trait calls on this instance. Without it, two
    /// `Session`s sharing this `Arc<dyn Model>` and running `forward()` /
    /// `forward_prefill()` concurrently would race on the per-instance
    /// scratch buffers (`hidden_buf`, `q_buf`, `k_buf`, etc.) and on the
    /// GPU KV caches in `gpu_state`. Mirrors the equivalent guard on
    /// `MetalLfm2Model`. Lock cost is ~50 ns uncontended (negligible vs
    /// wgpu dispatch); the wgpu queue already serializes GPU work — this
    /// just synchronizes the CPU-side bookkeeping that stages each
    /// command encoder and reads back logits.
    infer_lock: Mutex<()>,
    /// Lazily-allocated scratch KV/conv for [`Self::hidden_states`] (see
    /// `HsScratch`). Built on first use via [`Self::hs_scratch`] so a
    /// generation-only load pays no extra KV VRAM. Selected over the generation
    /// caches by `use_hs_scratch`, which is only toggled while holding
    /// `infer_lock`, so `Relaxed` ordering suffices.
    hs_scratch: OnceLock<HsScratch>,
    use_hs_scratch: AtomicBool,
    /// Caller-supplied identifier (typically the GGUF file path) used to
    /// namespace prefix-cache disk files. Prefixed with `"wgpu:"` before
    /// being fed to `model_fingerprint` so wgpu's f32 disk-cache files
    /// don't collide with Metal's f16 nor CPU's f32 ones at the same
    /// model path. CPU's f32 layout matches wgpu's, but the CPU model's
    /// own internal state shape (InferenceState-backed) differs from
    /// the GPU-resident state, so cross-loading isn't safe even when
    /// the byte format would line up — the prefix tag enforces backend
    /// separation cleanly.
    model_id: String,
    /// Two-tier prefix cache (warm in-memory + cold on-disk via
    /// FlatBuffers). Replaced wholesale by `Model::configure_cache`.
    /// Defaults to `KvCacheConfig::default()` (warm-only) at
    /// construction time so warm hits work without explicit config.
    prefix_cache: Mutex<KvPrefixCache>,
    /// GPU-uploaded LoRA adapters, keyed by the source CPU adapter's Arc
    /// identity (via `Arc::ptr_eq`, NOT the raw pointer — a freed adapter's
    /// address can be reused, so pointer identity alone would ABA-alias). LRU,
    /// cap 3, so hot-swapping between a few adapters doesn't re-upload every
    /// forward. Mutated only under `infer_lock`.
    lora_lru: Mutex<Vec<(Arc<LoraAdapterWeights>, Arc<WgpuLoraAdapter>)>>,
    /// The adapter to apply for the in-flight forward, staged by `resolve_lora`
    /// and read by the per-layer encoders. Cleared by the returned `LoraGuard`
    /// on drop so a leaked `Some` can't send a later base-model forward through
    /// the adapter.
    active_lora: Mutex<Option<Arc<WgpuLoraAdapter>>>,
    /// Rank-width f32 scratch for the LoRA `tmp = A·x` intermediate. Sized to
    /// `MAX_LORA_RANK` so any accepted adapter fits without reallocation.
    lora_tmp: wgpu::Buffer,
    /// Batched-prefill scratch for the LoRA down-projection result
    /// (`Tmp[n_tokens × rank]`, token-major). Sized `MAX_LORA_RANK ×
    /// min(max_seq_len, MAX_PREFILL_TOKENS)` f32 so it holds the whole rank
    /// output for the largest prefill chunk. Filled by `gemm_f32_nt`, consumed by
    /// `gemm_f32_nt_accum`.
    lora_tmp_batched: wgpu::Buffer,
    /// Reusable pool of 16-byte `[M,N,K,0]` params buffers for the batched-LoRA
    /// GEMM dispatches, plus the next-free index (reset to 0 per prefill). The
    /// batched prefill encodes every LoRA GEMM into ONE command buffer, so each
    /// dispatch needs its OWN params buffer (a single shared one would be
    /// last-write-wins across the submit); pooling reuses them across prefills so
    /// only adapter-active prefill pays, and only once (grows to the high-water
    /// mark, then zero allocation). Locked under `infer_lock` — no contention.
    lora_params_pool: Mutex<(Vec<wgpu::Buffer>, usize)>,
}

impl GpuLfm2Model {
    /// Construct without a model identifier. Equivalent to
    /// `from_gguf_with_id(gguf, context_size, "")`. Warm prefix cache
    /// works after `Model::configure_cache`; disk cache (when
    /// configured) would namespace-collide between path-less loads of
    /// different models.
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF
    /// path) used to namespace prefix-cache disk files. The id is
    /// prefixed with `"wgpu:"` before being fed to `model_fingerprint`
    /// so different backends (cpu / metal / wgpu) sharing a
    /// `--cache-dir` don't collide on file names — see CPU's `"cpu:"`
    /// in PR #119 for the same pattern.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let cpu_model = super::lfm2::Lfm2Model::from_gguf(gguf, context_size)?;
        Self::from_weight_source(&cpu_model, context_size, model_id)
    }

    /// Construct an LFM2 GPU model with an externally-built [`GpuContext`].
    /// The wasm/WebGPU entry point — callers build the context with
    /// `GpuContext::new_async().await` (browser init is async) and hand it in.
    pub fn from_gguf_with_ctx(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
        ctx: GpuContext,
    ) -> Result<Self> {
        let cpu_model = super::lfm2::Lfm2Model::from_gguf(gguf, context_size)?;
        Self::from_weight_source_with_ctx(&cpu_model, context_size, model_id, ctx)
    }

    /// Construct a GPU model for a dense transformer (Qwen2/Qwen3/LLaMA/
    /// Mistral/Granite) — the `LlamaModel` family. Mirrors `from_gguf_with_id`
    /// but feeds the shared loader a `LlamaModel` weight source instead of
    /// `Lfm2Model`. The GPU forward path is arch-generic; per-arch behavior
    /// (NEOX/NORM rope, QK-norm, QKV bias, untied output, Granite scalars) is
    /// driven by the `GpuWeightSource` accessors + `config`.
    pub fn from_llama_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let cpu_model =
            super::llama::LlamaModel::from_gguf_with_id(gguf, context_size, model_id.clone())?;
        Self::from_weight_source(&cpu_model, context_size, model_id)
    }

    /// Generalized GPU loader over any [`GpuWeightSource`]. Uploads weights,
    /// builds pipelines + scratch, and wires the arch-specific knobs. The
    /// concrete CPU model (`Lfm2Model` / `LlamaModel`) is only borrowed here
    /// for its weights/metadata; it is dropped on return.
    fn from_weight_source(
        src: &dyn GpuWeightSource,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        // Native: build the GPU context synchronously. wasm callers must use
        // `from_*_with_ctx` with a context built via `GpuContext::new_async`
        // (WebGPU init only resolves on the JS event loop).
        let ctx = GpuContext::new()?;
        Self::from_weight_source_with_ctx(src, context_size, model_id, ctx)
    }

    /// Like [`Self::from_weight_source`] but with an externally-constructed
    /// [`GpuContext`]. This is the wasm entry point: the context is built
    /// asynchronously (`GpuContext::new_async().await`) before construction,
    /// since the rest of loading (weight upload + pipeline build) is sync GPU
    /// work that does no readback and runs fine on the wasm main thread.
    fn from_weight_source_with_ctx(
        src: &dyn GpuWeightSource,
        context_size: usize,
        model_id: String,
        ctx: GpuContext,
    ) -> Result<Self> {
        // The CPU loader already caps max_seq_len to context_size internally,
        // so the second .min() below is redundant but kept for clarity.
        let mut config = src.config().clone();
        let max_seq_len = context_size.min(config.max_seq_len);
        config.max_seq_len = max_seq_len;
        let hs = config.hidden_size;
        let is = config.intermediate_size;
        // head_dim is decoupled from hidden/n_heads (Qwen3 sets it explicitly),
        // so size Q/K/V/attn-out buffers by config.head_dim, not hs/n_heads.
        let head_dim = config.head_dim;
        let q_dim = config.n_heads * head_dim;
        let max_kv_dim = config.kv_heads_per_layer.iter().copied().max().unwrap_or(0) * head_dim;
        let rope_type = src.rope_type();
        let scalars = config.scalars;
        let batched_prefill = src.supports_batched_prefill();

        tracing::info!(
            "GPU model: {} layers, hs={hs}, is={is}, vocab={}",
            config.n_layers,
            config.vocab_size
        );

        // Create pipelines
        let pipelines = GpuPipelines {
            gemv_f32: ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32", "gemv_f32"),
            gemv_f16: ctx.create_pipeline_with_defines(
                shaders::GEMV_F32,
                "gemv_f32",
                "gemv_f16",
                &[("F16_A", "1")],
            ),
            gemv_f32_accum: ctx.create_pipeline(
                shaders::GEMV_F32,
                "gemv_f32_accum",
                "gemv_f32_accum",
            ),
            gemm_f32_nt: ctx.create_pipeline(shaders::GEMM_F32, "gemm_f32_nt", "gemm_f32_nt"),
            gemm_f32_nt_accum: ctx.create_pipeline(
                shaders::GEMM_F32,
                "gemm_f32_nt_accum",
                "gemm_f32_nt_accum",
            ),
            gemv_q4_0: ctx.create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0", "gemv_q4_0"),
            gemv_q4_0_fast: ctx.create_pipeline(
                shaders::GEMV_Q4_0_FAST,
                "gemv_q4_0_fast",
                "gemv_q4_0_fast",
            ),
            gemv_q4_k: ctx.create_pipeline(shaders::GEMV_Q4_K, "gemv_q4_k", "gemv_q4_k"),
            gemv_q5_k: ctx.create_pipeline(shaders::GEMV_Q5_K, "gemv_q5_k", "gemv_q5_k"),
            gemv_q6_k: ctx.create_pipeline(shaders::GEMV_Q6_K, "gemv_q6_k", "gemv_q6_k"),
            gemv_q8_0: ctx.create_pipeline(shaders::GEMV_Q8_0, "gemv_q8_0", "gemv_q8_0"),
            add_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace", "add"),
            scaled_add_inplace: ctx.create_pipeline(
                shaders::ELEMENTWISE,
                "scaled_add_inplace",
                "scaled_add",
            ),
            scale_f32: ctx.create_pipeline(shaders::SCALE_F32, "scale_f32", "scale_f32"),
            mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_inplace", "mul"),
            silu_mul_inplace: ctx.create_pipeline(
                shaders::ELEMENTWISE,
                "silu_mul_inplace",
                "silu_mul",
            ),
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm"),
            per_head_rmsnorm: ctx.create_pipeline(
                shaders::PER_HEAD_RMSNORM,
                "per_head_rmsnorm",
                "per_head_rmsnorm",
            ),
            softmax: ctx.create_pipeline(shaders::SOFTMAX, "softmax", "softmax"),
            rope: ctx.create_pipeline(shaders::ROPE, "rope", "rope"),
            kv_shift: ctx.create_pipeline(shaders::KV_SHIFT, "kv_shift", "kv_shift"),
            flash_attention: ctx.create_pipeline(
                shaders::FLASH_ATTENTION,
                "flash_attention",
                "flash_attention",
            ),
            conv1d_fused: ctx.create_pipeline(
                shaders::CONV1D_FUSED,
                "conv1d_fused",
                "conv1d_fused",
            ),
            argmax_f32: ctx.create_pipeline(shaders::ARGMAX_F32, "argmax_f32", "argmax_f32"),
            rmsnorm_batch: ctx.create_pipeline(
                shaders::RMSNORM_BATCH,
                "rmsnorm_batch",
                "rmsnorm_batch",
            ),
            add_rmsnorm_batch: ctx.create_pipeline(
                shaders::RMSNORM_BATCH,
                "add_rmsnorm_batch",
                "add_rmsnorm_batch",
            ),
            qk_norm_rope_batch: ctx.create_pipeline(
                shaders::QK_NORM_ROPE_BATCH,
                "qk_norm_rope_batch",
                "qk_norm_rope_batch",
            ),
            conv1d_fused_batch: ctx.create_pipeline(
                shaders::CONV1D_FUSED_BATCH,
                "conv1d_fused_batch",
                "conv1d_fused_batch",
            ),
            bias_add: ctx.create_pipeline(shaders::BIAS_ADD, "bias_add", "bias_add"),

            mul_mat_reg_tile_q4_0_vec: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_q4_0_vec",
                true,
                "INIT_SRC0_SHMEM_Q4_0",
            ),
            mul_mat_reg_tile_q4_0_scalar: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_q4_0_scalar",
                false,
                "INIT_SRC0_SHMEM_Q4_0",
            ),
            // Every quantized weight goes through the register-tiled kernel (weight
            // reuse across the token tile), NOT the batched-GEMV-shaped gemm_* kernels:
            // those re-dequantize the weight once per token, so they buy submit count
            // and no compute, and measured *slower* than the per-token fallback they
            // were meant to replace.
            mul_mat_reg_tile_q8_0: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_q8_0",
                false,
                "INIT_SRC0_SHMEM_Q8_0",
            ),
            mul_mat_reg_tile_q4_k: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_q4_k",
                false,
                "INIT_SRC0_SHMEM_Q4_K",
            ),
            mul_mat_reg_tile_q6_k: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_q6_k",
                false,
                "INIT_SRC0_SHMEM_Q6_K",
            ),
            attention_prefill: ctx.create_pipeline(
                shaders::ATTENTION_PREFILL,
                "attention_prefill",
                "attention_prefill",
            ),
        };

        // Upload weights: Q4_0/Q8_0/Q6K stay quantized, others dequantized to f32.
        let emb_tensor = src.gguf().get_tensor("token_embd.weight")?;
        // The GPU `embedding` buffer feeds the (tied) logit projection and must
        // stay UNSCALED. The CPU-side `embedding_f32` cache feeds the input
        // embedding lookup; Granite's embedding multiplier is pre-folded into it
        // (no-op for every other arch). Keeping the two copies separate means a
        // tied-embedding Granite gets the scale on input only, exactly like the
        // CPU LlamaModel (`scale_inplace` after `dequantize_row`).
        let embedding_raw = emb_tensor.to_f32_vec();
        // The GPU `embedding` buffer feeds only the (tied) logit projection via
        // `encode_gemv_f16`, so it is stored as f16 (half the VRAM of the largest
        // tensor). The separate CPU `embedding_f32` cache below stays f32 for the
        // input-embedding lookup.
        let embedding = ctx.upload_f32_as_f16(&embedding_raw, "token_embd.weight");
        let mut embedding_f32 = embedding_raw;
        if scalars.embedding != 1.0 {
            for v in embedding_f32.iter_mut() {
                *v *= scalars.embedding;
            }
        }
        let embedding_params = ctx.upload_storage(
            bytemuck::cast_slice(&[
                config.vocab_size as u32,
                config.hidden_size as u32,
                0u32,
                0u32,
            ]),
            "emb_params",
        );
        let output_norm = ctx.upload_f32(src.output_norm_weight(), "output_norm");

        let upload_weight = |wref: &WeightRef, name: &str| -> GpuWeight {
            let (buf, dtype) = if matches!(
                wref.dtype,
                DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q5KM | DType::Q6K
            ) {
                // Q4_0/Q8_0/Q4KM have native GEMV+GEMM kernels; Q6K and Q5KM have
                // native GEMV kernels (`gemv_q6_k` / `gemv_q5_k`) used by decode
                // and the per-token prefill fallback. All stay quantized on the
                // GPU rather than dequantizing to f32: ~7× less VRAM for Q4KM
                // (144 B / 256 elems = 0.5625 B/elem vs 4 B/elem), ~5.8× for Q5KM
                // (176 B), and ~4.9× for Q6K (210 B).
                //
                // The shaders bind this buffer as `array<u32>` and do u32 reads.
                // `upload_storage`/`create_buffer_init` round the buffer size up
                // to COPY_BUFFER_ALIGNMENT (4 B) and zero the tail, so a row whose
                // byte length isn't a multiple of 4 is still safe to index as u32.
                // Q5KM (`nb*176` B) is already 4-aligned; Q6K (`nb*210`), Q4_0
                // (18 B/block), and Q8_0 (34 B/block) are not, and rely on that
                // round-up guarantee.
                let data = src.weight_bytes(wref);
                (ctx.upload_storage(data, name), wref.dtype)
            } else {
                // TODO: Upload as F16 to save bandwidth (requires F16-aware matmul shaders
                // in Phase B.1). For now we dequantize the remaining quant types to F32.
                let f32_data = src.dequantize_weight(wref);
                (ctx.upload_f32(&f32_data, name), DType::F32)
            };
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[wref.m as u32, wref.k as u32, 0u32, 0u32]),
                &format!("{name}.params"),
            );
            GpuWeight {
                tensor: GpuTensor {
                    buffer: buf,
                    dtype,
                    shape: vec![wref.m, wref.k],
                },
                params_buf,
                cached_bg: None,
            }
        };

        // Optional per-head QK-norm (Qwen3) and QKV bias (Qwen2) upload helpers.
        let upload_opt_f32 = |data: Option<&[f32]>, name: &str| -> Option<wgpu::Buffer> {
            data.map(|d| ctx.upload_f32(d, name))
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let attn_norm = ctx.upload_f32(src.attn_norm_weight(i), &format!("l{i}.anorm"));
            let ffn_norm = ctx.upload_f32(src.ffn_norm_weight(i), &format!("l{i}.fnorm"));

            let ffn_gate = upload_weight(src.ffn_gate_ref(i), &format!("l{i}.ffn_gate"));
            let ffn_up = upload_weight(src.ffn_up_ref(i), &format!("l{i}.ffn_up"));
            let ffn_down = upload_weight(src.ffn_down_ref(i), &format!("l{i}.ffn_down"));

            let is_conv = config.block_types[i] == BlockType::GatedConv;

            let (conv_in_proj, conv_out_proj, conv_weight) = if is_conv {
                let ip = src.conv_in_proj_ref(i).expect("conv layer missing in_proj");
                let op = src
                    .conv_out_proj_ref(i)
                    .expect("conv layer missing out_proj");
                (
                    Some(upload_weight(ip, &format!("l{i}.conv_ip"))),
                    Some(upload_weight(op, &format!("l{i}.conv_op"))),
                    Some(ctx.upload_f32(
                        src.conv_weight(i).expect("conv layer missing conv weight"),
                        &format!("l{i}.conv_w"),
                    )),
                )
            } else {
                (None, None, None)
            };

            // Attention weights. Plain transformers have every attention layer;
            // LFM2 has them only on attention blocks. QK-norm (Qwen3) and QKV
            // bias (Qwen2) are uploaded only when the source carries them.
            let (attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm) = if !is_conv {
                (
                    Some(upload_weight(
                        src.attn_q_ref(i).expect("attn layer missing q"),
                        &format!("l{i}.attn_q"),
                    )),
                    Some(upload_weight(
                        src.attn_k_ref(i).expect("attn layer missing k"),
                        &format!("l{i}.attn_k"),
                    )),
                    Some(upload_weight(
                        src.attn_v_ref(i).expect("attn layer missing v"),
                        &format!("l{i}.attn_v"),
                    )),
                    Some(upload_weight(
                        src.attn_output_ref(i).expect("attn layer missing output"),
                        &format!("l{i}.attn_o"),
                    )),
                    upload_opt_f32(src.attn_q_norm_weight(i), &format!("l{i}.qn")),
                    upload_opt_f32(src.attn_k_norm_weight(i), &format!("l{i}.kn")),
                )
            } else {
                (None, None, None, None, None, None)
            };

            let attn_q_bias = upload_opt_f32(src.attn_q_bias(i), &format!("l{i}.qb"));
            let attn_k_bias = upload_opt_f32(src.attn_k_bias(i), &format!("l{i}.kb"));
            let attn_v_bias = upload_opt_f32(src.attn_v_bias(i), &format!("l{i}.vb"));

            layers.push(GpuLayerWeights {
                attn_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                conv_in_proj,
                conv_out_proj,
                conv_weight,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
            });
        }

        // Untied output projection (`output.weight`), dequantized then stored as
        // f16 like the embedding table (feeds only `encode_gemv_f16`). `None` ⇒
        // tied embeddings (reuse `embedding`).
        let output_weight = src
            .output_ref()
            .map(|wref| ctx.upload_f32_as_f16(&src.dequantize_weight(wref), "output.weight"));

        // Create scratch buffers
        let f = |size: usize, name: &str| ctx.create_storage_rw((size * 4) as u64, name);
        let hidden_buf = f(hs, "hidden");
        let normed_buf = f(hs, "normed");
        let ffn_input_buf = f(hs, "ffn_input");
        let gate_buf = f(is, "gate");
        let up_buf = f(is, "up");
        let out_buf = f(hs, "out");
        // Q and the attention output are sized by n_heads*head_dim (= q_dim),
        // which exceeds hs when head_dim is decoupled (Qwen3). The out_proj maps
        // q_dim → hs. K/V are sized by max_kv_heads*head_dim.
        let q_buf = f(q_dim, "q");
        let k_buf = f(max_kv_dim, "k");
        let v_buf = f(max_kv_dim, "v");
        // KV-shift scratch: one retained K/V layer slab, sized to the worst case
        // (`max_seq_len × max_kv_dim`). No `.max(1)` guard — `k_buf`/`v_buf` above
        // already allocate `max_kv_dim` floats, so an attention-free
        // (`max_kv_dim == 0`) config would fail there first; LFM2 always has
        // attention layers, so `max_kv_dim` is never 0 in practice anyway.
        let kv_shift_scratch = f(max_seq_len * max_kv_dim, "kv_shift_scratch");
        let attn_out_buf = f(q_dim, "attn_out");
        let logits_buf = f(config.vocab_size, "logits");
        let conv_proj_buf = f(3 * hs, "conv_proj");
        let conv_gate_buf = f(hs, "conv_gate");

        // Batched-prefill scratch. Sized for the worst case of
        // `MAX_PREFILL_TOKENS` rows; chunking on the host side keeps
        // larger prompts within this footprint.
        let max_pref = max_seq_len.min(MAX_PREFILL_TOKENS);
        // Per-token column counts. `q_dim`/`max_kv_dim` can exceed `hs` when
        // head_dim is decoupled (Qwen3), so the scratch buffers that hold Q
        // (proj), the attention output (normed), and K/V (gate/up) must be
        // sized by the max of every role each buffer plays across the layer.
        // gate/up additionally carry hs-wide block outputs (attn/conv out_proj,
        // FFN down) and the hs-stride residual — include `hs` so the sizing is
        // self-evidently complete and not silently reliant on `is >= hs`.
        let prefill_batch_buf = f(hs * max_pref, "prefill_batch");
        let prefill_normed_buf = f(hs.max(q_dim) * max_pref, "prefill_normed");
        let prefill_proj_buf = f((3 * hs).max(q_dim) * max_pref, "prefill_proj");
        let prefill_gate_buf = f(is.max(max_kv_dim).max(hs) * max_pref, "prefill_gate");
        let prefill_up_buf = f(is.max(max_kv_dim).max(hs) * max_pref, "prefill_up");
        // attention_prefill scratch: per-(query, head, time) f32 slab.
        let prefill_scores_buf = f(max_pref * config.n_heads * max_seq_len, "prefill_scores");

        // Initialize GPU KV caches + conv buffers
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let mut kv_caches = Vec::with_capacity(config.n_layers);
        let mut conv_buffers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            if config.block_types[i] == BlockType::Attention {
                let kv_dim = config.kv_heads_per_layer[i] * head_dim;
                let k_cache = f(max_seq_len * kv_dim, &format!("l{i}.k_cache"));
                let v_cache = f(max_seq_len * kv_dim, &format!("l{i}.v_cache"));
                kv_caches.push(Some((k_cache, v_cache)));
                conv_buffers.push(None);
            } else {
                kv_caches.push(None);
                let cb = f(d_conv * hs, &format!("l{i}.conv_buf"));
                conv_buffers.push(Some(cb));
            }
        }

        let gpu_state = GpuState {
            kv_caches,
            conv_buffers,
            seq_len: AtomicUsize::new(0),
            max_seq_len,
            embedding_f32,
        };

        // Pre-allocate shader params buffers (avoids upload_storage per dispatch).
        let rmsnorm_hs_params = ctx.upload_storage(
            bytemuck::cast_slice(&[hs as u32, config.rms_norm_eps.to_bits(), 0u32, 0u32]),
            "rmsnorm_hs_params",
        );
        let elementwise_hs_params =
            ctx.upload_storage(bytemuck::cast_slice(&[hs as u32, 0u32]), "ew_hs_params");
        let elementwise_is_params =
            ctx.upload_storage(bytemuck::cast_slice(&[is as u32, 0u32]), "ew_is_params");
        // QKV-bias add lengths (Qwen2). q_dim == hs unless head_dim is decoupled.
        let kv_dim_bias = config.n_kv_heads * head_dim;
        let elementwise_qdim_params = ctx.upload_storage(
            bytemuck::cast_slice(&[q_dim as u32, 0u32]),
            "ew_qdim_params",
        );
        let elementwise_kvdim_params = ctx.upload_storage(
            bytemuck::cast_slice(&[kv_dim_bias as u32, 0u32]),
            "ew_kvdim_params",
        );
        // Residual add scalar (Granite residual multiplier; 1.0 elsewhere).
        let residual_add_params = ctx.upload_storage(
            bytemuck::cast_slice(&[hs as u32, scalars.residual.to_bits()]),
            "residual_add_params",
        );
        // Granite logit divide: scale by 1/logit_scale. None when identity.
        let logit_scale_params = (scalars.logit != 1.0).then(|| {
            ctx.upload_storage(
                bytemuck::cast_slice(&[config.vocab_size as u32, (1.0 / scalars.logit).to_bits()]),
                "logit_scale_params",
            )
        });
        let kernel_size = config.conv_kernel_size.unwrap_or(3) as u32;
        let d_conv = kernel_size - 1;
        let head_dim_u32 = head_dim as u32;
        let conv1d_params = ctx.upload_storage(
            bytemuck::cast_slice(&[hs as u32, kernel_size, d_conv, 0u32]),
            "conv1d_params",
        );
        let per_head_norm_params = ctx.upload_storage(
            bytemuck::cast_slice(&[head_dim_u32, config.rms_norm_eps.to_bits(), 0u32, 0u32]),
            "ph_norm_params",
        );
        // rope_params is updated per token via queue.write_buffer — needs COPY_DST.
        // 7 u32: pos, n_heads, n_kv_heads, head_dim, freq_base_bits, rope_type,
        // has_freq_factors.
        let rope_params = ctx.create_storage_rw(7 * 4, "rope_params");
        // Llama-3 RoPE frequency factors (binding 3 of the rope dispatch).
        // Always bound; a 1-element dummy when the model uses plain RoPE.
        let has_freq_factors = src.rope_freqs().is_some();
        let rope_freqs_buf = match src.rope_freqs() {
            Some(rf) => ctx.upload_f32(rf, "rope_freqs"),
            None => ctx.upload_f32(&[1.0f32], "rope_freqs_dummy"),
        };
        let attn_params = ctx.create_storage_rw(8 * 4, "attn_params");
        // The LM head is stored as f16 (2 bytes/elem) — see `encode_gemv_f16`.
        let lm_head_tile_rows = gemv_tile_rows(
            config.vocab_size as u32,
            hs as u32,
            ctx.max_storage_buffer_binding_size,
            ctx.min_storage_buffer_offset_alignment,
            2,
        );
        let lm_head_tile_count = (config.vocab_size as u32).div_ceil(lm_head_tile_rows);
        let mut gemv_tile_params = Vec::with_capacity(lm_head_tile_count as usize);
        for i in 0..lm_head_tile_count {
            gemv_tile_params.push(ctx.create_storage_rw(4 * 4, &format!("gemv_tile_params.{i}")));
        }

        // Argmax I/O buffers. `argmax_params` is uploaded once with
        // vocab_size; `argmax_out_buf` is a 4-byte sink. Bind group is
        // built after `pipelines` exists below.
        let argmax_out_buf = ctx.create_storage_rw(4, "argmax_out");
        let argmax_params = ctx.upload_storage(
            bytemuck::cast_slice(&[config.vocab_size as u32, 0u32]),
            "argmax_params",
        );
        let argmax_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("argmax_bg"),
            layout: &pipelines.argmax_f32.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: argmax_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: argmax_params.as_entire_binding(),
                },
            ],
        });

        // Build the prefix cache before constructing `Self` so we can
        // borrow `&config` here without conflicting with the upcoming
        // move of `config` into the struct literal.
        let prefix_cache = Mutex::new(KvPrefixCache::new(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &format!("wgpu:{model_id}"),
        ));

        // LoRA `tmp = A·x` scratch, sized to the max supported rank so any
        // accepted adapter fits.
        let lora_tmp = ctx.create_storage_rw((crate::lora::MAX_LORA_RANK * 4) as u64, "lora_tmp");
        // Batched LoRA down-projection scratch: MAX_LORA_RANK × max_pref f32s.
        let lora_tmp_batched = ctx.create_storage_rw(
            (crate::lora::MAX_LORA_RANK * max_pref * 4) as u64,
            "lora_tmp_batched",
        );

        let mut model = Self {
            ctx,
            config,
            pipelines,
            embedding,
            embedding_params,
            output_weight,
            output_norm,
            layers,
            rope_type,
            scalars,
            batched_prefill,
            batched_fallback_warned: AtomicBool::new(false),
            rope_freqs_buf,
            has_freq_factors,
            hidden_buf,
            normed_buf,
            ffn_input_buf,
            gate_buf,
            up_buf,
            out_buf,
            q_buf,
            k_buf,
            v_buf,
            kv_shift_scratch,
            attn_out_buf,
            logits_buf,
            argmax_out_buf,
            argmax_params,
            argmax_bg,
            rmsnorm_hs_params,
            elementwise_hs_params,
            elementwise_is_params,
            elementwise_qdim_params,
            elementwise_kvdim_params,
            residual_add_params,
            logit_scale_params,
            conv1d_params,
            per_head_norm_params,
            rope_params,
            attn_params,
            gemv_tile_params,
            conv_proj_buf,
            conv_gate_buf,
            prefill_batch_buf,
            prefill_normed_buf,
            prefill_proj_buf,
            prefill_gate_buf,
            prefill_up_buf,
            prefill_scores_buf,
            gpu_state,
            infer_lock: Mutex::new(()),
            hs_scratch: OnceLock::new(),
            use_hs_scratch: AtomicBool::new(false),
            prefix_cache,
            model_id,
            lora_lru: Mutex::new(Vec::new()),
            active_lora: Mutex::new(None),
            lora_tmp,
            lora_tmp_batched,
            lora_params_pool: Mutex::new((Vec::new(), 0)),
        };
        model.cache_bind_groups();
        Ok(model)
    }

    /// Resolve `state.lora` to a GPU-uploaded adapter and stage it in
    /// `active_lora` for the encoders to read, returning a guard that clears
    /// `active_lora` on drop. Uploads are cached in an Arc-pointer-keyed LRU
    /// (cap 3) so hot-swapping between a few adapters doesn't re-upload every
    /// forward. Must be called while holding `infer_lock` (it mutates the
    /// per-model `active_lora`/`lora_lru`).
    fn resolve_lora(&self, state: &InferenceState) -> LoraGuard<'_> {
        let resolved = state.lora.as_ref().map(|adapter| {
            let mut lru = self.lora_lru.lock().expect("lora_lru poisoned");
            if let Some(pos) = lru.iter().position(|(cpu, _)| Arc::ptr_eq(cpu, adapter)) {
                // Hit: mark most-recently-used by moving the entry to the end
                // (the vec is ordered least- → most-recently-used).
                let (cpu, gpu) = lru.remove(pos);
                lru.push((cpu, gpu.clone()));
                gpu
            } else {
                // Miss: upload, insert, evict the least-recently-used if over cap.
                let gpu = Arc::new(WgpuLoraAdapter::upload(
                    &self.ctx,
                    adapter,
                    self.scalars.residual,
                ));
                lru.push((adapter.clone(), gpu.clone()));
                if lru.len() > 3 {
                    lru.remove(0);
                }
                gpu
            }
        });
        *self.active_lora.lock().expect("active_lora poisoned") = resolved;
        LoraGuard(&self.active_lora)
    }

    /// Build the two bind groups for one LoRA target's apply:
    /// `bg_a` = (A, input, lora_tmp, a_params) for `gemv_f32` (`tmp = A·x`), and
    /// `bg_b` = (B_scaled, lora_tmp, output, b_params) for `gemv_f32_accum`
    /// (`out += B_scaled·tmp`). Decode-path offsets are 0, so whole-buffer
    /// bindings suffice (`x[col]`/`y[row]` read/write from the front).
    fn lora_target_bgs(
        &self,
        t: &WgpuLoraTarget,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        // `bg_a` binds to the `gemv_f32` pipeline, `bg_b` to `gemv_f32_accum`.
        // wgpu treats the two pipelines as exclusive even though their layouts
        // are structurally identical, so each bind group must be created from
        // its own pipeline's layout.
        let layout_a = self.pipelines.gemv_f32.get_bind_group_layout(0);
        let layout_b = self.pipelines.gemv_f32_accum.get_bind_group_layout(0);
        let bg_a = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lora_a"),
                layout: &layout_a,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: t.a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.lora_tmp.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: t.a_params.as_entire_binding(),
                    },
                ],
            });
        let bg_b = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lora_b"),
                layout: &layout_b,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: t.b_scaled.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.lora_tmp.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: t.b_params.as_entire_binding(),
                    },
                ],
            });
        (bg_a, bg_b)
    }

    /// Append the `(layer, target)` LoRA delta into an already-open compute pass
    /// if the active adapter touches it: two dispatches, `tmp = A·x` then
    /// `out += B_scaled·tmp`. WebGPU serializes storage reads/writes between
    /// dispatches in the same pass, so the shared `lora_tmp` scratch is safe to
    /// reuse across back-to-back hooks. The caller must have pre-built the bind
    /// groups (via `lora_target_bgs`) before opening the pass — bind groups
    /// borrow `self` immutably, which conflicts with the mutable pass borrow.
    fn dispatch_lora_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        t: &WgpuLoraTarget,
        bg_a: &wgpu::BindGroup,
        bg_b: &wgpu::BindGroup,
    ) {
        // tmp = A·x — m = rank rows. `gemv_f32`/`gemv_f32_accum` each emit `NR`
        // rows per workgroup, so the group count is `rows / NR`.
        let a_groups = t.rank.div_ceil(GEMV_F32_ROWS_PER_WG);
        self.dispatch_into(
            pass,
            &self.pipelines.gemv_f32,
            bg_a,
            crate::backend::wgpu::gemv_row_workgroups(a_groups),
        );
        // out += B_scaled·tmp — m = d rows.
        let b_groups = t.d.div_ceil(GEMV_F32_ROWS_PER_WG);
        self.dispatch_into(
            pass,
            &self.pipelines.gemv_f32_accum,
            bg_b,
            crate::backend::wgpu::gemv_row_workgroups(b_groups),
        );
    }

    /// Look up the active adapter's `(layer, target)` factors, if present.
    fn lora_target(
        lora: Option<&Arc<WgpuLoraAdapter>>,
        layer: usize,
        target: LoraTarget,
    ) -> Option<&WgpuLoraTarget> {
        lora?.layers.get(layer)?[target.index()].as_ref()
    }

    /// A pooled 16-byte `[M,N,K,0]` params buffer for a batched-LoRA GEMM,
    /// written with `data` and reused across prefills. The counter advances per
    /// call so each dispatch in a prefill's single command buffer gets a distinct
    /// buffer (a shared one would be last-write-wins across the submit); the pool
    /// grows to the high-water mark then never allocates again. Reset the counter
    /// (`lora_params_pool.1 = 0`) at the start of each batched prefill.
    fn next_lora_params(&self, data: &[u32; 4]) -> wgpu::Buffer {
        let mut pool = self
            .lora_params_pool
            .lock()
            .expect("lora_params_pool poisoned");
        let (bufs, next) = &mut *pool;
        let idx = *next;
        *next += 1;
        if bufs.len() <= idx {
            bufs.push(self.ctx.create_storage_rw(16, "lora_batched_params"));
        }
        let buf = bufs[idx].clone();
        self.ctx
            .queue
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    /// Batched-prefill LoRA delta for one target, applied in-batch across all `n`
    /// tokens: `Y[n×d] += B_batched · (A · X[n×k])`, computed as two NT GEMMs
    /// (`gemm_f32_nt` / `gemm_f32_nt_accum`) that match the token-major batch
    /// buffer layout (`X[tok*k + i]`, `Y[tok*d + o]`).
    ///
    /// `t.b_batched` carries only the `alpha/rank` scale (not `residual_mult`) —
    /// the caller applies the LoRA before the fused residual add, so the model's
    /// residual scale wraps the delta (matches `lora::apply_prefill`). Each GEMM
    /// runs as its own compute pass, so wgpu's inter-pass resource barriers keep
    /// the shared `lora_tmp_batched` write-then-read ordered (GEMM1 writes it,
    /// GEMM2 reads it) and back-to-back hooks reusing the scratch stay correct.
    fn encode_lora_batched(
        &self,
        enc: &mut wgpu::CommandEncoder,
        t: &WgpuLoraTarget,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) {
        // GEMM 1: Tmp[n × rank] = X[n × k] · Aᵀ  (A is [rank × k] row-major).
        // One workgroup per output element; total = n·rank workgroups.
        let total1 = n * t.rank;
        let p1: [u32; 4] = [n, t.rank, t.k, 0];
        let p1_buf = self.next_lora_params(&p1);
        let bg1 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lora_batched_a"),
                layout: &self.pipelines.gemm_f32_nt.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: t.a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.lora_tmp_batched.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p1_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.gemm_f32_nt,
            &bg1,
            crate::backend::wgpu::gemv_row_workgroups(total1),
            "lora_batched_a",
        );

        // GEMM 2: Y[n × d] += Tmp[n × rank] · Bᵀ  (B_batched is [d × rank] row-major).
        let total2 = n * t.d;
        let p2: [u32; 4] = [n, t.d, t.rank, 0];
        let p2_buf = self.next_lora_params(&p2);
        let bg2 = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lora_batched_b"),
                layout: &self.pipelines.gemm_f32_nt_accum.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lora_tmp_batched.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: t.b_batched.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p2_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.gemm_f32_nt_accum,
            &bg2,
            crate::backend::wgpu::gemv_row_workgroups(total2),
            "lora_batched_b",
        );
    }

    /// Batched-prefill counterpart of the decode `dispatch_lora_into`: apply the
    /// `(layer, target)` LoRA delta across all `n` tokens if the active adapter
    /// touches it. `input`/`output` are token-major batch buffers at offset 0.
    #[allow(clippy::too_many_arguments)]
    fn encode_lora_hook_batched(
        &self,
        enc: &mut wgpu::CommandEncoder,
        lora: Option<&Arc<WgpuLoraAdapter>>,
        layer: usize,
        target: LoraTarget,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) {
        if let Some(t) = Self::lora_target(lora, layer, target) {
            self.encode_lora_batched(enc, t, input, output, n);
        }
    }

    /// Create a GEMV bind group for a given (weight, input, output) triple.
    fn make_gemv_bg(
        &self,
        w: &GpuWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let (pipeline, _, _) = self.gemv_pipeline_rows_label(w);
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w.tensor.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: w.params_buf.as_entire_binding(),
                    },
                ],
            })
    }

    fn gemv_pipeline_rows_label(
        &self,
        w: &GpuWeight,
    ) -> (&wgpu::ComputePipeline, u32, &'static str) {
        // rows-per-workgroup MUST match each shader's `NR`/`ROWS_PER_WG`
        // constant: gemv_q4_0_fast=4, gemv_q8_0=8, gemv_q4_k=2, gemv_q5_k=2,
        // gemv_q6_k=2, gemv_f32=8. A mismatch over-dispatches and the shaders
        // bounds-check only writes, not weight reads, so a too-small value reads
        // past the weight buffer.
        match w.tensor.dtype {
            DType::Q4_0 => (&self.pipelines.gemv_q4_0_fast, 4, "gemv_q4"),
            DType::Q8_0 => (&self.pipelines.gemv_q8_0, 8, "gemv_q8"),
            DType::Q4KM => (&self.pipelines.gemv_q4_k, 2, "gemv_q4k"),
            DType::Q6K => (&self.pipelines.gemv_q6_k, 2, "gemv_q6"),
            // gemv_q5_k uses NR=2 rows per workgroup (must match the shader).
            DType::Q5KM => (&self.pipelines.gemv_q5_k, 2, "gemv_q5k"),
            _ => (&self.pipelines.gemv_f32, 8, "gemv_f32"),
        }
    }

    fn gemv_workgroups(&self, w: &GpuWeight) -> (u32, u32, u32) {
        let (_, rows_per_wg, _) = self.gemv_pipeline_rows_label(w);
        let row_groups = (w.tensor.shape[0] as u32).div_ceil(rows_per_wg);
        // Flatten into (x, y) so m > MAX_WG*rows_per_wg rows still map to distinct
        // row groups; the shaders recover the flat index via `get_wid`.
        crate::backend::wgpu::gemv_row_workgroups(row_groups)
    }

    fn dispatch_gemv_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        w: &GpuWeight,
        bind_group: &wgpu::BindGroup,
    ) {
        let (pipeline, _, _) = self.gemv_pipeline_rows_label(w);
        self.dispatch_into(pass, pipeline, bind_group, self.gemv_workgroups(w));
    }

    /// Pre-create bind groups for all per-layer GEMV dispatches.
    /// Eliminates ~150 create_bind_group calls per token (~2.4 ms CPU).
    fn cache_bind_groups(&mut self) {
        let cfg = &self.config;
        for i in 0..cfg.n_layers {
            // FFN
            let gate_bg = self.make_gemv_bg(
                &self.layers[i].ffn_gate,
                &self.ffn_input_buf,
                &self.gate_buf,
            );
            self.layers[i].ffn_gate.cached_bg = Some(gate_bg);
            let up_bg =
                self.make_gemv_bg(&self.layers[i].ffn_up, &self.ffn_input_buf, &self.up_buf);
            self.layers[i].ffn_up.cached_bg = Some(up_bg);
            let down_bg =
                self.make_gemv_bg(&self.layers[i].ffn_down, &self.gate_buf, &self.out_buf);
            self.layers[i].ffn_down.cached_bg = Some(down_bg);

            if cfg.block_types[i] == BlockType::GatedConv {
                if let Some(ref w) = self.layers[i].conv_in_proj {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.conv_proj_buf);
                    self.layers[i].conv_in_proj.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].conv_out_proj {
                    let bg = self.make_gemv_bg(w, &self.conv_gate_buf, &self.out_buf);
                    self.layers[i].conv_out_proj.as_mut().unwrap().cached_bg = Some(bg);
                }
            } else {
                if let Some(ref w) = self.layers[i].attn_q {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.q_buf);
                    self.layers[i].attn_q.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].attn_k {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.k_buf);
                    self.layers[i].attn_k.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].attn_v {
                    let bg = self.make_gemv_bg(w, &self.normed_buf, &self.v_buf);
                    self.layers[i].attn_v.as_mut().unwrap().cached_bg = Some(bg);
                }
                if let Some(ref w) = self.layers[i].attn_output {
                    let bg = self.make_gemv_bg(w, &self.attn_out_buf, &self.out_buf);
                    self.layers[i].attn_output.as_mut().unwrap().cached_bg = Some(bg);
                }
            }
        }
    }

    // ── GPU dispatch helpers ────────────────────────────────────────────

    /// Encode a compute pass into the given encoder (batched, no submit).
    fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
        label: &str,
    ) {
        let ts = self.ctx.begin_profile_span(label);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: ts,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
    }

    /// Dispatch into an existing compute pass (no pass creation overhead).
    fn dispatch_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    /// Submit encoder and wait for GPU to finish.
    fn submit_and_wait(&self, enc: wgpu::CommandEncoder) {
        self.ctx.submit_encoder(enc);
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    fn new_encoder(&self) -> wgpu::CommandEncoder {
        self.ctx.device.create_command_encoder(&Default::default())
    }

    // ── Encode helpers (add passes to an existing encoder) ────────────

    /// Encode GEMV dispatch — uses cached bind group if available, else creates one.
    #[allow(dead_code)]
    fn encode_gemv_weight(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &GpuWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        let (pipeline, _, label) = self.gemv_pipeline_rows_label(w);
        // Use cached BG if available (pre-created at init for known
        // weight/input/output triples — saves ~16µs per dispatch).
        let fresh_bg;
        let bg = if let Some(ref cached) = w.cached_bg {
            cached
        } else {
            fresh_bg = self.make_gemv_bg(w, input, output);
            &fresh_bg
        };
        self.encode(enc, pipeline, bg, self.gemv_workgroups(w), label);
    }

    /// Encode the LM-head GEMV (logit projection). The weight (`embedding` tied /
    /// `output.weight` untied) is stored as f16 — see `gemv_f16` — so activations
    /// stay f32 while the largest GPU tensor takes half the VRAM.
    fn encode_gemv_f16(
        &self,
        enc: &mut wgpu::CommandEncoder,
        weight: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        m: u32,
        k: u32,
    ) {
        // Compare the true binding size: the f16 buffer is u32-addressed, so its
        // `as_entire_binding` size is rounded up to a whole u32 (matching the
        // `upload_f32_as_f16` padding and the tiled round-up). Without this, an
        // adapter whose `max_binding` is not itself a multiple of 4 could take
        // the non-tiled path and then fail binding validation.
        let weight_bytes = (u64::from(m) * u64::from(k) * 2).div_ceil(4) * 4;
        let max_binding = self.ctx.max_storage_buffer_binding_size;
        if weight_bytes > max_binding {
            self.encode_gemv_f16_tiled(enc, weight, input, output, m, k);
            return;
        }

        // Use pre-allocated params (m=vocab_size, k=hs are constant).
        let params_buf = &self.embedding_params;
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.gemv_f16.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        let groups = m.div_ceil(8);
        self.encode(
            enc,
            &self.pipelines.gemv_f16,
            &bg,
            crate::backend::wgpu::gemv_row_workgroups(groups),
            "gemv_f16",
        );
    }

    /// Encode the f16 LM-head GEMV in row tiles for adapters with small
    /// max_storage_buffer_binding_size limits. The tied embedding/output
    /// projection can exceed those limits even though each row slice is legal.
    fn encode_gemv_f16_tiled(
        &self,
        enc: &mut wgpu::CommandEncoder,
        weight: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        m: u32,
        k: u32,
    ) {
        let row_bytes = u64::from(k) * 2;
        let max_binding = self.ctx.max_storage_buffer_binding_size;
        let tile_rows = gemv_tile_rows(
            m,
            k,
            max_binding,
            self.ctx.min_storage_buffer_offset_alignment,
            2, // f16 weight element size
        );

        let layout = self.pipelines.gemv_f16.get_bind_group_layout(0);
        let mut row_start = 0u32;
        let mut tile_idx = 0usize;
        while row_start < m {
            let rows = (m - row_start).min(tile_rows);
            let weight_offset = u64::from(row_start) * row_bytes;
            let params_buf = self
                .gemv_tile_params
                .get(tile_idx)
                .expect("preallocated LM-head GEMV tile params");
            self.ctx.queue.write_buffer(
                params_buf,
                0,
                bytemuck::cast_slice(&[rows, k, row_start, 0u32]),
            );
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: weight,
                                offset: weight_offset,
                                // Bind as `array<u32>` (f16 packed 2/u32), so the
                                // size must be a whole number of u32s. A final tile
                                // whose `rows*k` is odd (only when `k` is odd) would
                                // otherwise be 2-mod-4 and drop its last f16 pair;
                                // round up (the buffer is 4-byte padded at upload).
                                size: wgpu::BufferSize::new(
                                    (u64::from(rows) * row_bytes).div_ceil(4) * 4,
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });
            let groups = rows.div_ceil(8);
            self.encode(
                enc,
                &self.pipelines.gemv_f16,
                &bg,
                crate::backend::wgpu::gemv_row_workgroups(groups),
                "gemv_f16_tiled",
            );
            row_start += rows;
            tile_idx += 1;
        }
    }

    fn encode_rmsnorm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        _n: u32,
        _eps: f32,
    ) {
        // Use pre-allocated params buffer (n and eps are always hs and config.rms_norm_eps).
        let params_buf = &self.rmsnorm_hs_params;
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(enc, &self.pipelines.rmsnorm, &bg, (1, 1, 1), "rmsnorm");
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_attention(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_dim: u32,
        seq_len: u32,
        scale: f32,
    ) {
        // flash_attention.wgsl sizes `q_shared` / `acc` at MAX_HEAD_DIM (128) f32,
        // so head_dim must fit. Every current LFM2 / dense model uses head_dim ∈
        // {64, 128}, but assert loudly rather than let a larger head_dim silently
        // corrupt the output via clamped out-of-bounds workgroup-array writes —
        // an explicit contract instead of silent garbage.
        assert!(
            head_dim <= 128,
            "wgpu flash_attention supports head_dim <= 128 (q_shared/acc are \
             sized 128); got {head_dim}"
        );
        // The kernel derives `group_size = n_heads / n_kv_heads` and
        // `kv_head = head / group_size`, then reads a head_dim-wide slice at
        // `kv_head * head_dim` within each kv_dim-strided KV row. Enforce the GQA
        // invariants it assumes so a malformed config fails fast here instead of
        // dividing by zero or reading out of bounds on the GPU.
        assert!(
            n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "wgpu flash_attention requires n_kv_heads > 0 and n_heads divisible by \
             n_kv_heads; got n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        );
        assert_eq!(
            kv_dim,
            n_kv_heads * head_dim,
            "wgpu flash_attention requires kv_dim == n_kv_heads * head_dim; got \
             kv_dim={kv_dim}, n_kv_heads={n_kv_heads}, head_dim={head_dim}"
        );
        let params: [u32; 8] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            seq_len,
            scale.to_bits(),
            0,
            0,
        ];
        self.ctx
            .queue
            .write_buffer(&self.attn_params, 0, bytemuck::cast_slice(&params));
        let params_buf = &self.attn_params;
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.flash_attention.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.flash_attention,
            &bg,
            (n_heads, 1, 1),
            "flash_attention",
        );
    }

    // encode_per_head_rmsnorm, encode_rope, encode_elementwise, encode_conv1d
    // removed — logic inlined into batched forward pass.

    /// Encode an f32-granular buffer→buffer copy. ALL THREE size args
    /// (`src_off_floats`, `dst_off_floats`, `len_floats`) are counts of f32
    /// elements, not bytes — the helper scales each to bytes internally. Keeping
    /// a single unit at the call sites removes the foot-gun where an offset is
    /// byte-counted but the length is float-counted (or vice-versa), which would
    /// land the copy at the wrong offset and silently corrupt the KV cache.
    ///
    /// Associated (no `self`) so the unit contract is directly unit-testable;
    /// every buffer→buffer copy in this file (decode, prefill, KV-shift, the
    /// last-token epilogue) routes through here for one consistent convention.
    fn encode_copy(
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_off_floats: u64,
        dst: &wgpu::Buffer,
        dst_off_floats: u64,
        len_floats: u64,
    ) {
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        enc.copy_buffer_to_buffer(
            src,
            src_off_floats * f32_bytes,
            dst,
            dst_off_floats * f32_bytes,
            len_floats * f32_bytes,
        );
    }

    /// Per-layer dispatch loop for the n_keep KV shift, called by
    /// `Model::shift_kv` once `retained > 0` is established. For each attention
    /// layer: (1) re-rotate the retained K cells by `R(-shift)` into
    /// `kv_shift_scratch` via the `kv_shift` kernel, (2) copy the rotated K back
    /// into the cache at the `n_keep` offset, (3) ferry V through the same
    /// scratch to its new offset (V isn't RoPE'd, but its source/destination
    /// ranges overlap the same way K's do, so it can't move in place either).
    ///
    /// The two copies reuse `copy_buffer_to_buffer`; wgpu's automatic usage
    /// tracking inserts the WAR/RAW barriers between the compute pass and the
    /// copies (and across layers that share the one scratch buffer), so the
    /// single command encoder stays correct without manual synchronization.
    fn encode_kv_shift_layers(&self, n_keep: usize, shift: usize, retained: usize) {
        debug_assert!(retained > 0, "encode_kv_shift_layers requires retained > 0");
        let cfg = &self.config;
        let head_dim = cfg.head_dim;
        let freq_base_bits = cfg.rope_theta.to_bits();

        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv_shift"),
            });

        for layer_idx in 0..cfg.n_layers {
            if cfg.block_types[layer_idx] != BlockType::Attention {
                continue;
            }
            let n_kv_heads = cfg.kv_heads_per_layer[layer_idx];
            let kv_dim = n_kv_heads * head_dim;
            let (k_cache, v_cache) = self.gpu_state.kv_caches[layer_idx]
                .as_ref()
                .expect("attention layer missing GPU kv_caches entry");

            // Per-layer params: `n_kv_heads`/`kv_dim` can vary per layer (GQA),
            // so a fresh tiny storage buffer per layer is simpler — and cheaper
            // to reason about — than reusing one buffer with `write_buffer`
            // (whose writes wouldn't interleave with the in-encoder dispatches).
            // KV-shift fires only on context overflow, so the allocation is rare.
            let params = KvShiftParams {
                n_keep: n_keep as u32,
                shift: shift as u32,
                retained: retained as u32,
                n_kv_heads: n_kv_heads as u32,
                head_dim: head_dim as u32,
                freq_base_bits,
                rope_type: self.rope_type as u32,
                has_freq_factors: u32::from(self.has_freq_factors),
            };
            let params_buf = self.ctx.upload_storage(
                bytemuck::cast_slice(&params.to_u32_array()),
                "kv_shift_params",
            );

            // ── K: re-rotate retained cells into scratch (compact order) ──
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("kv_shift"),
                    layout: &self.pipelines.kv_shift.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: k_cache.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.kv_shift_scratch.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                        // Bound even when `has_freq_factors` is false: the kernel
                        // only reads it on the Llama-3 path, but every binding in
                        // the layout must be set. `rope_freqs_buf` is a `[1.0]`
                        // dummy for plain-RoPE models.
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rope_freqs_buf.as_entire_binding(),
                        },
                    ],
                });
            // One thread per (retained cell, kv head, RoPE pair). The grid is
            // 2-D-flattened via `dispatch_dims` (shared with the oracle test and
            // unit-tested in `backend::wgpu`) because the retained context can
            // push the workgroup count past the 65535 per-dimension limit; the
            // kernel recovers the flat index via `get_wid`. `encode` adds the GPU
            // profiling span + debug label.
            self.encode(
                &mut enc,
                &self.pipelines.kv_shift,
                &bg,
                params.dispatch_dims(),
                "kv_shift",
            );
            // Copy rotated K back into the cache at the new n_keep-aligned offset.
            let n_floats = (retained * kv_dim) as u64;
            Self::encode_copy(
                &mut enc,
                &self.kv_shift_scratch,
                0,
                k_cache,
                (n_keep * kv_dim) as u64,
                n_floats,
            );

            // ── V: ferry through scratch to the new offset (no rotation) ──
            Self::encode_copy(
                &mut enc,
                v_cache,
                ((n_keep + shift) * kv_dim) as u64,
                &self.kv_shift_scratch,
                0,
                n_floats,
            );
            Self::encode_copy(
                &mut enc,
                &self.kv_shift_scratch,
                0,
                v_cache,
                (n_keep * kv_dim) as u64,
                n_floats,
            );
        }

        // KV-shift is a rare, synchronous boundary (context overflow) — block so
        // the subsequent prefill reads the fully-shifted cache.
        self.submit_and_wait(enc);
    }
}

impl GpuLfm2Model {
    /// Lock-free body of [`Model::forward`]. Callers must already hold
    /// `infer_lock` — enter via the trait's `forward()` for a single
    /// token, or `forward_prefill` for the hot prefill loop. The
    /// `std::sync::Mutex` guarding the Model trait surface is not
    /// reentrant, so calling `Model::forward` from inside this body
    /// would deadlock.
    fn forward_inner(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.forward_inner_compute(tokens, pos, state);
        self.ctx
            .download_f32(&self.logits_buf, self.config.vocab_size)
    }

    /// Lazily build (once) the hidden-states scratch caches — same shapes as the
    /// generation caches. Called under `infer_lock` from `hidden_states` before
    /// `use_hs_scratch` is set, so `active_kv`/`active_conv` always find it built.
    fn hs_scratch(&self) -> &HsScratch {
        self.hs_scratch.get_or_init(|| {
            let cfg = &self.config;
            let head_dim = cfg.head_dim;
            let hs = cfg.hidden_size;
            let d_conv = cfg.conv_kernel_size.unwrap_or(3) - 1;
            let max_seq_len = self.gpu_state.max_seq_len;
            let f = |size: usize, name: &str| self.ctx.create_storage_rw((size * 4) as u64, name);
            let mut kv = Vec::with_capacity(cfg.n_layers);
            let mut conv = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                if cfg.block_types[i] == BlockType::Attention {
                    let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                    let k = f(max_seq_len * kv_dim, &format!("hs.l{i}.k"));
                    let v = f(max_seq_len * kv_dim, &format!("hs.l{i}.v"));
                    kv.push(Some((k, v)));
                    conv.push(None);
                } else {
                    kv.push(None);
                    conv.push(Some(f(d_conv * hs, &format!("hs.l{i}.conv"))));
                }
            }
            HsScratch { kv, conv }
        })
    }

    /// The attention KV cache for layer `i` — the hidden-states scratch cache
    /// when [`Self::hidden_states`] is running (`use_hs_scratch`), else the
    /// generation cache. Panics on a conv layer (no KV).
    #[inline]
    fn active_kv(&self, i: usize) -> &(wgpu::Buffer, wgpu::Buffer) {
        let caches = if self.use_hs_scratch.load(Ordering::Relaxed) {
            &self
                .hs_scratch
                .get()
                .expect("hs_scratch built before use_hs_scratch is set")
                .kv
        } else {
            &self.gpu_state.kv_caches
        };
        caches[i].as_ref().unwrap()
    }

    /// The conv rolling buffer for layer `i` — scratch vs generation.
    #[inline]
    fn active_conv(&self, i: usize) -> &wgpu::Buffer {
        let bufs = if self.use_hs_scratch.load(Ordering::Relaxed) {
            &self
                .hs_scratch
                .get()
                .expect("hs_scratch built before use_hs_scratch is set")
                .conv
        } else {
            &self.gpu_state.conv_buffers
        };
        bufs[i].as_ref().unwrap()
    }

    /// Computes one forward pass and leaves the resulting logits in
    /// `self.logits_buf` on the GPU **without** reading them back. Caller
    /// chooses how to consume the logits — full readback for sampling
    /// (`forward_inner`) or a single-`u32` argmax readback for greedy
    /// decoding (`forward_greedy_inner`). This split lets the wasm-async
    /// path avoid the vocab-sized blocking download every step.
    fn forward_inner_compute(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) {
        assert_eq!(tokens.len(), 1, "GPU forward expects single token");
        let token_id = tokens[0] as usize;
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;

        self.ctx.reset_profiler();

        // Bounds check: KV cache capacity
        assert!(
            self.gpu_state.seq_len.load(Ordering::Relaxed) < self.gpu_state.max_seq_len,
            "GPU seq_len {} exceeds max_seq_len {}",
            self.gpu_state.seq_len.load(Ordering::Relaxed),
            self.gpu_state.max_seq_len,
        );

        // 1. Embedding lookup from CPU cache (4KB upload per token)
        let emb_offset = token_id * hs;
        self.ctx.queue.write_buffer(
            &self.hidden_buf,
            0,
            bytemuck::cast_slice(&self.gpu_state.embedding_f32[emb_offset..emb_offset + hs]),
        );

        // Active LoRA adapter (cheap Arc clone; `None` on the base-model path).
        // Read once so every hook in this forward shares one lock acquisition.
        let lora = self
            .active_lora
            .lock()
            .expect("active_lora poisoned")
            .clone();

        // 2. Per-layer loop — one encoder per layer (block + FFN merged), each
        // submitted independently. That is 16 submits + 1 for the head below, and
        // the per-token GPU I/O counters will report ~19 submits/token.
        //
        // THAT COUNT IS NOT A BUG, AND MERGING THESE INTO ONE COMMAND BUFFER MAKES
        // DECODE SLOWER. Measured, LFM2 Q4_K_M / Q4_0, one submit per token:
        //
        //     Mac (wgpu/Metal)   62.0 -> 45.3 tok/s
        //     Adreno 840         12.4 ->  8.6 tok/s
        //
        // Decode is GPU-execution-bound, not submit-bound: ~15-18 ms of GPU work per
        // token against only ~1.6-2.4 ms of CPU encode. Submitting each layer as it
        // is encoded lets the GPU start layer i while the CPU is still building bind
        // groups for layer i+1. Batch them and the GPU instead sits idle through the
        // whole encode phase, which is pure loss — the submits themselves are cheap
        // on both platforms. The overlap is GPU-vs-CPU; it is NOT an attempt to
        // overlap layers with each other (they are strictly serial through
        // `hidden_buf` and cannot overlap).
        //
        // If you want faster decode, cut GPU work per token — not the submit count.
        // T5b has already profiled it (`CERA_GPU_PROFILE=1`): decode is memory-bound
        // inside the quantized GEMVs, which sustain only ~25 GB/s against the f16
        // GEMV's 106 GB/s on the same GPU. Fix those loads. See `BASELINE.md`.
        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];
            let mut enc = self.new_encoder();

            if cfg.block_types[i] == BlockType::GatedConv {
                let kernel_size = cfg.conv_kernel_size.unwrap_or(3) as u32;
                let _d_conv = kernel_size - 1;
                let conv_buf = self.active_conv(i);

                // Pre-create BGs for conv block (using pre-allocated params).
                let norm_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.normed_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: lw.attn_norm.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.rmsnorm_hs_params.as_entire_binding(),
                            },
                        ],
                    });
                let in_w = lw.conv_in_proj.as_ref().unwrap();
                let in_bg_tmp;
                let in_bg = match in_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        in_bg_tmp = self.make_gemv_bg(in_w, &self.normed_buf, &self.conv_proj_buf);
                        &in_bg_tmp
                    }
                };
                // LoRA conv in_proj delta (`conv_proj_buf += scale·B·(A·normed)`),
                // added into the full 3·hidden projection before the fused conv
                // reads the B/C/x gates. Bind groups built before the pass opens.
                let in_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::ShortconvInProj);
                let in_lora_bgs = in_lora.map(|t| {
                    (
                        t,
                        self.lora_target_bgs(t, &self.normed_buf, &self.conv_proj_buf),
                    )
                });
                // Pass 1: rmsnorm + in_proj (after hidden→normed copy).
                Self::encode_copy(
                    &mut enc,
                    &self.hidden_buf,
                    0,
                    &self.normed_buf,
                    0,
                    hs as u64,
                );
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_pre"),
                        timestamp_writes: self.ctx.begin_profile_span("conv_pre"),
                    });
                    self.dispatch_into(&mut pass, &self.pipelines.rmsnorm, &norm_bg, (1, 1, 1));
                    self.dispatch_gemv_into(&mut pass, in_w, in_bg);
                    if let Some((t, (bg_a, bg_b))) = &in_lora_bgs {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                }

                // Pre-create BGs for passes 2 and 3. The fused conv shader reads
                // x/c/b directly from `conv_proj_buf` at offsets 0/hs/2*hs and
                // writes output to `conv_gate_buf` (where the post-conv out_proj
                // gemv reads from) — replaces the prior mul1 + conv1d + mul2
                // sequence and the three encoder copies that fed it.
                let conv_p = &self.conv1d_params;
                let conv_fused_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.conv1d_fused.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.conv_proj_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: conv_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: lw.conv_weight.as_ref().unwrap().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.conv_gate_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: conv_p.as_entire_binding(),
                            },
                        ],
                    });
                let out_w = lw.conv_out_proj.as_ref().unwrap();
                let out_bg_tmp;
                let out_bg = match out_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        out_bg_tmp = self.make_gemv_bg(out_w, &self.conv_gate_buf, &self.out_buf);
                        &out_bg_tmp
                    }
                };
                // LoRA conv out_proj delta (`out_buf += scale·B·(A·conv_gate)`),
                // added before the plain `add_inplace` folds `out_buf` into the
                // residual — scale-only (not residual_mult), matching the CPU path.
                let out_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::ShortconvOutProj);
                let out_lora_bgs = out_lora.map(|t| {
                    (
                        t,
                        self.lora_target_bgs(t, &self.conv_gate_buf, &self.out_buf),
                    )
                });
                let add_p = &self.elementwise_hs_params;
                let add_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.add_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.hidden_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: add_p.as_entire_binding(),
                            },
                        ],
                    });

                // Pass 2: fused conv block (bx = x*b → conv → c*conv_out).
                // One dispatch replaces the prior mul1 + conv1d + mul2 trio
                // plus three encoder copies that extracted x/c/b from the
                // proj buffer into separate per-channel buffers.
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_mid"),
                        timestamp_writes: self.ctx.begin_profile_span("conv_mid"),
                    });
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.conv1d_fused,
                        &conv_fused_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                }

                // Pass 3: out_proj + add.
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("conv_post"),
                        timestamp_writes: self.ctx.begin_profile_span("conv_post"),
                    });
                    self.dispatch_gemv_into(&mut pass, out_w, out_bg);
                    if let Some((t, (bg_a, bg_b))) = &out_lora_bgs {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.add_inplace,
                        &add_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                }
            } else {
                // Attention block — batched into 2 compute passes (separated by KV cache copies).
                let head_dim = cfg.head_dim as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[i] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;
                let q_dim = n_heads * head_dim;

                // Pre-create all BGs before opening passes.
                let norm_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.normed_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: lw.attn_norm.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.rmsnorm_hs_params.as_entire_binding(),
                            },
                        ],
                    });
                let q_w = lw.attn_q.as_ref().unwrap();
                let q_bg_tmp;
                let q_bg = match q_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        q_bg_tmp = self.make_gemv_bg(q_w, &self.normed_buf, &self.q_buf);
                        &q_bg_tmp
                    }
                };
                let k_w = lw.attn_k.as_ref().unwrap();
                let k_bg_tmp;
                let k_bg = match k_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        k_bg_tmp = self.make_gemv_bg(k_w, &self.normed_buf, &self.k_buf);
                        &k_bg_tmp
                    }
                };
                let v_w = lw.attn_v.as_ref().unwrap();
                let v_bg_tmp;
                let v_bg = match v_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        v_bg_tmp = self.make_gemv_bg(v_w, &self.normed_buf, &self.v_buf);
                        &v_bg_tmp
                    }
                };
                // QK-norm (Qwen3) — only when the layer carries per-head norm
                // weights. Built as `Option` so non-Qwen3 archs skip the dispatch.
                let per_head_norm_bg = |buf: &wgpu::Buffer, norm: &wgpu::Buffer| {
                    self.ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &self.pipelines.per_head_rmsnorm.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: norm.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: self.per_head_norm_params.as_entire_binding(),
                                },
                            ],
                        })
                };
                let qn_bg = lw
                    .attn_q_norm
                    .as_ref()
                    .map(|w| per_head_norm_bg(&self.q_buf, w));
                let kn_bg = lw
                    .attn_k_norm
                    .as_ref()
                    .map(|w| per_head_norm_bg(&self.k_buf, w));

                // QKV bias (Qwen2) — added right after each projection GEMV,
                // before QK-norm/RoPE. `Option` so bias-less archs skip it.
                let bias_bg = |buf: &wgpu::Buffer, bias: &wgpu::Buffer, params: &wgpu::Buffer| {
                    self.ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &self.pipelines.add_inplace.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: bias.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: params.as_entire_binding(),
                                },
                            ],
                        })
                };
                let qb_bg = lw
                    .attn_q_bias
                    .as_ref()
                    .map(|b| bias_bg(&self.q_buf, b, &self.elementwise_qdim_params));
                let kb_bg = lw
                    .attn_k_bias
                    .as_ref()
                    .map(|b| bias_bg(&self.k_buf, b, &self.elementwise_kvdim_params));
                let vb_bg = lw
                    .attn_v_bias
                    .as_ref()
                    .map(|b| bias_bg(&self.v_buf, b, &self.elementwise_kvdim_params));

                let rope_data: [u32; 7] = [
                    pos as u32,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cfg.rope_theta.to_bits(),
                    self.rope_type as u32,
                    self.has_freq_factors as u32,
                ];
                self.ctx
                    .queue
                    .write_buffer(&self.rope_params, 0, bytemuck::cast_slice(&rope_data));
                let rope_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.rope.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.q_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.k_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.rope_params.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.rope_freqs_buf.as_entire_binding(),
                            },
                        ],
                    });

                let max_pairs = std::cmp::max(n_heads, n_kv_heads) * (head_dim / 2);

                // LoRA Q/K/V deltas: `+= scale·B·(A·normed)` on the raw
                // projections, before QK-norm/RoPE (additive, so it commutes
                // with the Qwen2 bias-add below). Bind groups are built here
                // (immutable `self` borrow) so they can be dispatched inside the
                // `attn_pre` pass, which mutably borrows `enc`.
                let q_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::AttnQ);
                let k_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::AttnK);
                let v_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::AttnV);
                let q_lora_bgs =
                    q_lora.map(|t| (t, self.lora_target_bgs(t, &self.normed_buf, &self.q_buf)));
                let k_lora_bgs =
                    k_lora.map(|t| (t, self.lora_target_bgs(t, &self.normed_buf, &self.k_buf)));
                let v_lora_bgs =
                    v_lora.map(|t| (t, self.lora_target_bgs(t, &self.normed_buf, &self.v_buf)));

                // Copy hidden → normed, then pass 1: norm + QKV + per-head norm + rope.
                Self::encode_copy(
                    &mut enc,
                    &self.hidden_buf,
                    0,
                    &self.normed_buf,
                    0,
                    hs as u64,
                );
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_pre"),
                        timestamp_writes: self.ctx.begin_profile_span("attn_pre"),
                    });
                    self.dispatch_into(&mut pass, &self.pipelines.rmsnorm, &norm_bg, (1, 1, 1));
                    self.dispatch_gemv_into(&mut pass, q_w, q_bg);
                    self.dispatch_gemv_into(&mut pass, k_w, k_bg);
                    self.dispatch_gemv_into(&mut pass, v_w, v_bg);
                    // LoRA Q/K/V deltas on the raw projections (before bias/norm/rope).
                    if let Some((t, (bg_a, bg_b))) = q_lora_bgs.as_ref() {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                    if let Some((t, (bg_a, bg_b))) = k_lora_bgs.as_ref() {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                    if let Some((t, (bg_a, bg_b))) = v_lora_bgs.as_ref() {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                    // QKV bias (Qwen2): add right after the projections.
                    if let Some(bg) = qb_bg.as_ref() {
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.add_inplace,
                            bg,
                            (q_dim.div_ceil(256), 1, 1),
                        );
                    }
                    if let Some(bg) = kb_bg.as_ref() {
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.add_inplace,
                            bg,
                            (kv_dim.div_ceil(256), 1, 1),
                        );
                    }
                    if let Some(bg) = vb_bg.as_ref() {
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.add_inplace,
                            bg,
                            (kv_dim.div_ceil(256), 1, 1),
                        );
                    }
                    // QK-norm (Qwen3): per-head RMSNorm before RoPE.
                    if let Some(bg) = qn_bg.as_ref() {
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.per_head_rmsnorm,
                            bg,
                            (n_heads, 1, 1),
                        );
                    }
                    if let Some(bg) = kn_bg.as_ref() {
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.per_head_rmsnorm,
                            bg,
                            (n_kv_heads, 1, 1),
                        );
                    }
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.rope,
                        &rope_bg,
                        (max_pairs.div_ceil(256), 1, 1),
                    );
                }

                // KV cache copies (encoder-level), then pass 2: attention + out_proj + add.
                let (k_cache, v_cache) = self.active_kv(i);
                let seq_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
                let kv_offset_floats = (seq_len * kv_dim as usize) as u64;
                Self::encode_copy(
                    &mut enc,
                    &self.k_buf,
                    0,
                    k_cache,
                    kv_offset_floats,
                    kv_dim as u64,
                );
                Self::encode_copy(
                    &mut enc,
                    &self.v_buf,
                    0,
                    v_cache,
                    kv_offset_floats,
                    kv_dim as u64,
                );

                let attn_seq_len = (seq_len + 1) as u32;
                // Granite overrides the softmax scale with its attention
                // multiplier; every other arch uses 1/sqrt(head_dim).
                let scale = self
                    .scalars
                    .attn
                    .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
                // Attention BG (changes per token due to seq_len).
                self.encode_attention(
                    &mut enc,
                    &self.q_buf,
                    k_cache,
                    v_cache,
                    &self.attn_out_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    attn_seq_len,
                    scale,
                );
                // out_proj + add — batch into one pass.
                let out_w = lw.attn_output.as_ref().unwrap();
                let out_bg_tmp;
                let out_bg = match out_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        out_bg_tmp = self.make_gemv_bg(out_w, &self.attn_out_buf, &self.out_buf);
                        &out_bg_tmp
                    }
                };
                // Residual add: `scaled_add_inplace` folds Granite's residual
                // multiplier into the addend (1.0 for every other arch).
                let add_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.scaled_add_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.hidden_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.residual_add_params.as_entire_binding(),
                            },
                        ],
                    });
                // LoRA attn-output delta: input is the attention output (o_proj
                // input), added into the post-residual hidden state. The
                // `residual_mult` fold at upload matches the base o_proj's
                // `scaled_add_inplace` scaling.
                let o_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::AttnOutput);
                let o_lora_bgs = o_lora.map(|t| {
                    (
                        t,
                        self.lora_target_bgs(t, &self.attn_out_buf, &self.hidden_buf),
                    )
                });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("attn_post"),
                        timestamp_writes: self.ctx.begin_profile_span("attn_post"),
                    });
                    self.dispatch_gemv_into(&mut pass, out_w, out_bg);
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.scaled_add_inplace,
                        &add_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                    if let Some((t, (bg_a, bg_b))) = o_lora_bgs.as_ref() {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                }
            }

            // FFN — same encoder as block above.
            Self::encode_copy(
                &mut enc,
                &self.hidden_buf,
                0,
                &self.ffn_input_buf,
                0,
                hs as u64,
            );
            // FFN: batch 6 dispatches into ONE compute pass.
            // Pre-create bind groups before opening the pass.
            let norm_params = &self.rmsnorm_hs_params;
            let norm_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.rmsnorm.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.ffn_input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: lw.ffn_norm.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: norm_params.as_entire_binding(),
                        },
                    ],
                });
            let gate_bg_tmp;
            let gate_bg = match lw.ffn_gate.cached_bg.as_ref() {
                Some(bg) => bg,
                None => {
                    gate_bg_tmp =
                        self.make_gemv_bg(&lw.ffn_gate, &self.ffn_input_buf, &self.gate_buf);
                    &gate_bg_tmp
                }
            };
            let up_bg_tmp;
            let up_bg = match lw.ffn_up.cached_bg.as_ref() {
                Some(bg) => bg,
                None => {
                    up_bg_tmp = self.make_gemv_bg(&lw.ffn_up, &self.ffn_input_buf, &self.up_buf);
                    &up_bg_tmp
                }
            };
            let silu_params = &self.elementwise_is_params;
            let silu_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.silu_mul_inplace.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.gate_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.up_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: silu_params.as_entire_binding(),
                        },
                    ],
                });
            let down_bg_tmp;
            let down_bg = match lw.ffn_down.cached_bg.as_ref() {
                Some(bg) => bg,
                None => {
                    down_bg_tmp = self.make_gemv_bg(&lw.ffn_down, &self.gate_buf, &self.out_buf);
                    &down_bg_tmp
                }
            };
            // Residual add: `scaled_add_inplace` folds Granite's residual
            // multiplier into the addend (1.0 for every other arch).
            let add_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.scaled_add_inplace.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.hidden_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.out_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.residual_add_params.as_entire_binding(),
                        },
                    ],
                });

            // LoRA gate/up deltas on the raw projections (before silu_mul), and
            // the ffn-down delta into the post-residual hidden state (input is
            // the silu_mul result in `gate_buf`). All three run for conv layers
            // too — only the FFN is shared by both block types. `residual_mult`
            // is folded into ffn-down's B at upload.
            let gate_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::FfnGate);
            let up_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::FfnUp);
            let down_lora = Self::lora_target(lora.as_ref(), i, LoraTarget::FfnDown);
            let gate_lora_bgs = gate_lora.map(|t| {
                (
                    t,
                    self.lora_target_bgs(t, &self.ffn_input_buf, &self.gate_buf),
                )
            });
            let up_lora_bgs = up_lora.map(|t| {
                (
                    t,
                    self.lora_target_bgs(t, &self.ffn_input_buf, &self.up_buf),
                )
            });
            let down_lora_bgs =
                down_lora.map(|t| (t, self.lora_target_bgs(t, &self.gate_buf, &self.hidden_buf)));

            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ffn"),
                    timestamp_writes: self.ctx.begin_profile_span("ffn"),
                });
                // rmsnorm
                self.dispatch_into(&mut pass, &self.pipelines.rmsnorm, &norm_bg, (1, 1, 1));
                // gate + up GEMVs
                self.dispatch_gemv_into(&mut pass, &lw.ffn_gate, gate_bg);
                self.dispatch_gemv_into(&mut pass, &lw.ffn_up, up_bg);
                // LoRA gate/up deltas on the raw projections, before silu_mul.
                if let Some((t, (bg_a, bg_b))) = gate_lora_bgs.as_ref() {
                    self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                }
                if let Some((t, (bg_a, bg_b))) = up_lora_bgs.as_ref() {
                    self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                }
                // silu_mul
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.silu_mul_inplace,
                    &silu_bg,
                    ((lw.ffn_gate.tensor.shape[0] as u32).div_ceil(256), 1, 1),
                );
                // down GEMV
                self.dispatch_gemv_into(&mut pass, &lw.ffn_down, down_bg);
                // residual add
                self.dispatch_into(
                    &mut pass,
                    &self.pipelines.scaled_add_inplace,
                    &add_bg,
                    (hs32.div_ceil(256), 1, 1),
                );
                // LoRA ffn-down delta into the post-residual hidden state.
                if let Some((t, (bg_a, bg_b))) = down_lora_bgs.as_ref() {
                    self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                }
            }
            self.ctx.submit_encoder(enc);
        }

        // 3. Output norm + projection. Untied models project through
        // `output.weight`; tied models reuse the embedding table.
        let mut enc = self.new_encoder();
        self.encode_rmsnorm(
            &mut enc,
            &self.hidden_buf,
            &self.output_norm,
            hs32,
            cfg.rms_norm_eps,
        );
        let out_proj = self.output_weight.as_ref().unwrap_or(&self.embedding);
        self.encode_gemv_f16(
            &mut enc,
            out_proj,
            &self.hidden_buf,
            &self.logits_buf,
            cfg.vocab_size as u32,
            hs32,
        );
        // Granite divides the logits by `logits_scaling` (identity elsewhere).
        if let Some(params) = self.logit_scale_params.as_ref() {
            let scale_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("logit_scale_bg"),
                    layout: &self.pipelines.scale_f32.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.logits_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("logit_scale"),
                timestamp_writes: None,
            });
            self.dispatch_into(
                &mut pass,
                &self.pipelines.scale_f32,
                &scale_bg,
                ((cfg.vocab_size as u32).div_ceil(256), 1, 1),
            );
            drop(pass);
        }
        self.submit_and_wait(enc);

        // 4. Update seq_len + profile bookkeeping. Logits are now in
        // `logits_buf` on the GPU; the caller decides how to consume
        // them (full readback vs. argmax-then-u32-readback).
        self.gpu_state.seq_len.fetch_add(1, Ordering::Relaxed);
        state.seq_len += 1;
        self.ctx.finish_profiler();
    }

    /// Greedy single-token forward: runs the same kernels as
    /// [`forward_inner`] but replaces the vocab-sized logits download
    /// with a 4-byte argmax readback. Cuts per-token PCIe/USB-C
    /// readback from `vocab_size * 4` bytes to `4` bytes — the
    /// wasm-async-friendly path, since a 4-byte map_async still
    /// blocks the JS event loop briefly but doesn't transfer megabytes.
    /// Encode the argmax compute pass into `enc`. Shared by the sync and async
    /// greedy paths so the kernel / bind-group / dispatch live in one place.
    fn encode_argmax_pass(&self, enc: &mut wgpu::CommandEncoder) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("argmax"),
            timestamp_writes: self.ctx.begin_profile_span("argmax"),
        });
        pass.set_pipeline(&self.pipelines.argmax_f32);
        pass.set_bind_group(0, &self.argmax_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn forward_greedy_inner(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        self.forward_inner_compute(tokens, pos, state);

        // Encode + submit the argmax pass on its own. Could be folded
        // into the output-projection encoder for one fewer submission,
        // but that's a `forward_inner_compute` refactor we're keeping
        // out of this PR.
        self.ctx.reset_profiler();
        let mut enc = self.new_encoder();
        self.encode_argmax_pass(&mut enc);
        self.submit_and_wait(enc);
        self.ctx.finish_profiler();

        let out = self.ctx.download_u32(&self.argmax_out_buf, 1);
        out[0]
    }

    /// Async-path prefill step: run the forward and update the KV cache
    /// *without* the argmax + readback. Used for every prompt token except the
    /// last, whose argmax seeds decoding — so an N-token prompt does one GPU→CPU
    /// round-trip instead of N. Synchronous: only the readback needs to be
    /// async. Pins `gpu_state.seq_len` to `pos` so the RoPE position (driven by
    /// `pos`) and the KV-write slot (driven by `gpu_state.seq_len`) cannot
    /// drift (mirrors `forward_prefill`).
    pub fn forward_prefill_step(&self, token: u32, pos: usize, state: &mut InferenceState) {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        let _lora_guard = self.resolve_lora(state);
        self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
        self.forward_inner_compute(&[token], pos, state);
    }

    /// Async (wasm/WebGPU) greedy decode step. Runs the full forward + argmax
    /// on the GPU, then reads back the single argmax token id without blocking
    /// — the wasm-compatible analog of `Self::forward_greedy_inner`.
    ///
    /// The blocking version's `submit_and_wait`'s `device.poll(Maintain::Wait)`
    /// is a no-op on the WebGPU backend (the browser owns the queue), so we
    /// submit the argmax pass directly; the readback is ordered after it on the
    /// same queue. The GPU compute + submit run under `infer_lock` (serialising
    /// shared scratch + GPU state against any other forward, like the sync
    /// `Model` methods); the lock is released before the `.await` (a per-call
    /// staging buffer makes the readback self-contained, so this is safe and
    /// avoids holding a `std::sync::Mutex` across `.await`). Single-token only.
    pub async fn forward_greedy_async(
        &self,
        token: u32,
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<u32> {
        let pending = {
            let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
            let _lora_guard = self.resolve_lora(state);
            // Keep the KV-write slot in lockstep with the RoPE position.
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            self.forward_inner_compute(&[token], pos, state);

            self.ctx.reset_profiler();
            let mut enc = self.new_encoder();
            self.encode_argmax_pass(&mut enc);
            self.ctx.submit_encoder(enc);
            self.ctx.finish_profiler();

            self.ctx
                .begin_download(&self.argmax_out_buf, std::mem::size_of::<u32>() as u64)
        };

        let bytes = pending.recv().await?;
        let mut out = [0u32; 1];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes);
        Ok(out[0])
    }

    /// Adapter name and backend of the underlying [`GpuContext`], for
    /// surfacing which GPU/backend the model is actually running on (e.g. in
    /// the wasm `WebGpuSession.adapter` getter).
    pub fn gpu_info(&self) -> (&str, &str) {
        (&self.ctx.adapter_name, &self.ctx.backend)
    }
}

// === Batched prefill — encode helpers + main method ========================
//
// Mirror `MetalLfm2Model::prefill_layers_and_logits` (metal_lfm2.rs:2906).
// Uses the five batched shaders landed in PRs #154 + #156:
//   rmsnorm_batch / add_rmsnorm_batch (PR #154)
//   qk_norm_rope_batch                (PR #154)
//   conv1d_fused_batch                (PR #154)
//   mul_mat_reg_tile                  (PR #162)
//   attention_prefill                 (PR #156)
//
// Scope:
//   * `forward_prefill_batched_locked` accepts any `start_pos`, so the
//     dispatcher chunks long prompts through it in
//     `min(max_seq_len, MAX_PREFILL_TOKENS)` chunks (each chunk advances
//     `start_pos`; conv rolling state and KV cache writes carry across).
//   * `1 <= n <= MAX_PREFILL_TOKENS` per call (asserted).
//   * `start_pos + n <= max_seq_len` (asserted).
//   * Every matmul weight must have a batched GEMM kernel. All four supported
//     dtypes (Q4_0, Q8_0, Q4KM, Q6K) now run `mul_mat_reg_tile`, differing only
//     in the shmem dequant loader. Any other dtype (F32, Q5KM, …) makes
//     `unbatchable_matmul_weight` return the offending tensor and the whole
//     prompt falls through to the per-token GEMV loop at the dispatcher.
//
// Extending the batched path to further dtypes (e.g. a Q5KM loader, or an
// f32 `gemm_f32` fallback) can land in a follow-up PR without disturbing this
// contract.
//
// Per-dispatch overhead note: each `encode_*` helper builds a fresh
// `wgpu::BindGroup` and uploads a small params buffer per call. The CPU
// cost is ~1 % of total prefill time at the workloads measured in PR #157;
// promoting the params buffers to model-resident state and caching the
// bind groups for fixed prefill scratch buffers is a clean follow-up
// optimization. Kept simple here so the refactor is reviewable.
//
// `prefill_scores_buf` size note: this scratch is sized to
// `MAX_PREFILL_TOKENS × n_heads × max_seq_len × 4` bytes (256 MB on
// LFM2-VL-450M / 512 MB on LFM2.5-VL-1.6B at the default 8192 context).
// On native macOS this is fine (M1+ unified memory). For wasm / WebGPU
// tier-1 (256 MB max storage buffer) this becomes load-bearing — the
// proper fix is a two-pass online softmax in `attention_prefill.wgsl`
// that doesn't materialize the full scores matrix; queued as a follow-up
// shader PR.

impl GpuLfm2Model {
    /// The first matmul weight that has no batched prefill kernel, as
    /// `(layer, tensor name, dtype)` — or `None` when every weight has one, which
    /// is the precondition for `forward_prefill_batched_locked` to take the batched
    /// path. All four supported dtypes (Q4_0, Q8_0, Q4KM, Q6K) run the same
    /// register-tiled kernel, differing only in the shmem dequant loader.
    ///
    /// Returns the *offender*, not a bare `bool`, because one unsupported tensor
    /// silently drops the whole prompt onto the per-token loop — ~340x the submits
    /// (measured: 8728 vs 25 on a 512-token prefill). A `false` that names nothing
    /// is how a `Q4_K_M` model — which is *not* uniformly Q4_K; it carries a
    /// handful of Q6_K tensors — sat on the slow path unnoticed. Cheap
    /// `O(n_layers)` walk; called once per `forward_prefill`.
    fn unbatchable_matmul_weight(&self) -> Option<(usize, &'static str, DType)> {
        for (li, lw) in self.layers.iter().enumerate() {
            let weights: [(&'static str, Option<&GpuWeight>); 9] = [
                ("ffn_gate", Some(&lw.ffn_gate)),
                ("ffn_up", Some(&lw.ffn_up)),
                ("ffn_down", Some(&lw.ffn_down)),
                ("attn_q", lw.attn_q.as_ref()),
                ("attn_k", lw.attn_k.as_ref()),
                ("attn_v", lw.attn_v.as_ref()),
                ("attn_output", lw.attn_output.as_ref()),
                ("conv_in_proj", lw.conv_in_proj.as_ref()),
                ("conv_out_proj", lw.conv_out_proj.as_ref()),
            ];
            for (name, w) in weights {
                let Some(w) = w else { continue };
                let dt = w.tensor.dtype;
                if !matches!(dt, DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q6K) {
                    return Some((li, name, dt));
                }
            }
        }
        None
    }

    /// Encode `rmsnorm_batch`: dst[t, i] = src[t, i] * inv_rms(src[t]) * w[i]
    /// for t in 0..n. Workgroup per token. Uses the binding layout shared
    /// with `add_rmsnorm_batch`; naga drops binding 4 from the
    /// auto-inferred layout for this entry point.
    fn encode_rmsnorm_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        n: u32,
        hs: u32,
    ) {
        // params[4] (res_scale) is unused by the no-residual `rmsnorm_batch`
        // entry point; pass 1.0 to keep the shared 5-u32 layout valid.
        let params: [u32; 5] = [
            hs,
            self.config.rms_norm_eps.to_bits(),
            hs,
            hs,
            1.0f32.to_bits(),
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "rmsnorm_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.rmsnorm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.rmsnorm_batch,
            &bg,
            (n, 1, 1),
            "rmsnorm_batch",
        );
    }

    /// Encode `add_rmsnorm_batch`: src[t,i] += residual[t,i]; dst[t,i] =
    /// src[t,i] * inv_rms(src[t]) * w[i]. One pass; src is read-write.
    #[allow(clippy::too_many_arguments)]
    fn encode_add_rmsnorm_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        residual: &wgpu::Buffer,
        n: u32,
        hs: u32,
    ) {
        // params[4] folds Granite's residual multiplier into the addend
        // (`scalars.residual`; 1.0 for every other arch ⇒ plain residual add).
        let params: [u32; 5] = [
            hs,
            self.config.rms_norm_eps.to_bits(),
            hs,
            hs,
            self.scalars.residual.to_bits(),
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "add_rmsnorm_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.add_rmsnorm_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: residual.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.add_rmsnorm_batch,
            &bg,
            (n, 1, 1),
            "add_rmsnorm_batch",
        );
    }

    /// Encode batched 2D matmul: y = weight * x.
    /// Batched prefill supports quantized Q4_0, Q8_0, Q4KM and Q6K weights. F32
    /// weights are not a production path in this model. `x_stride` and `y_stride`
    /// are measured in f32 elements between consecutive token vectors.
    #[allow(clippy::too_many_arguments)] // tile geometry + strides; splitting hurts clarity
    fn encode_mul_mat_reg_tile(
        &self,
        enc: &mut wgpu::CommandEncoder,
        w: &GpuWeight,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: u32,
        k: u32,
        x_stride: u32,
        y_stride: u32,
    ) {
        debug_assert!(
            matches!(
                w.tensor.dtype,
                DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q6K
            ),
            "encode_mul_mat_reg_tile only supports Q4_0/Q8_0/Q4KM/Q6K weights"
        );
        let m = w.tensor.shape[0] as u32;
        let (pipeline, wg_m, wg_n, label) = match w.tensor.dtype {
            DType::Q4_0 => {
                let use_vec = m % 4 == 0 && k % 4 == 0 && x_stride % 4 == 0 && y_stride % 4 == 0;
                let pipeline = if use_vec {
                    &self.pipelines.mul_mat_reg_tile_q4_0_vec
                } else {
                    &self.pipelines.mul_mat_reg_tile_q4_0_scalar
                };
                let wg_m = m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
                let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);
                (pipeline, wg_m, wg_n, "mul_mat_tile")
            }
            DType::Q8_0 => {
                // Same register-tiled geometry as Q4_0 — only the shmem dequant
                // differs. Scalar (not vec4): the packed-byte loader can't guarantee
                // the multiple-of-4 m/k/stride the vec4 path needs.
                let wg_m = m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
                let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);
                (
                    &self.pipelines.mul_mat_reg_tile_q8_0,
                    wg_m,
                    wg_n,
                    "mul_mat_q8_0",
                )
            }
            DType::Q4KM => {
                // Same register-tiled geometry as Q4_0/Q6_K — only the shmem dequant
                // differs. Scalar (not vec4): the packed-byte loader can't guarantee
                // the multiple-of-4 m/k/stride the vec4 path needs.
                let wg_m = m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
                let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);
                (
                    &self.pipelines.mul_mat_reg_tile_q4_k,
                    wg_m,
                    wg_n,
                    "mul_mat_q4k",
                )
            }
            DType::Q6K => {
                // Same register-tiled geometry as Q4_0 — only the shmem dequant
                // differs. Scalar (not vec4) because Q6_K rows are 210-byte blocks:
                // the vec4 path needs m/k/strides all multiples of 4, which the
                // packed-byte loader does not guarantee.
                let wg_m = m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
                let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);
                (
                    &self.pipelines.mul_mat_reg_tile_q6_k,
                    wg_m,
                    wg_n,
                    "mul_mat_q6k",
                )
            }
            // Unreachable in practice: the batched path is only entered when
            // `unbatchable_matmul_weight()` returned `None`, i.e. every weight is
            // Q4_0/Q8_0/Q4KM/Q6K. The debug_assert above documents the same
            // precondition; this arm is the release-mode backstop.
            _ => unreachable!("batched prefill only supports Q4_0/Q8_0/Q4KM/Q6K"),
        };

        // Matches `mul_mat_reg_tile`'s 5-field `MulMatParams`. This was 6 words while
        // the Q8_0 arm still dispatched `gemm_q8_0`, whose `params: array<u32, 6>` is
        // fixed-size — the buffer had to be sized to the union of both layouts. Every
        // dtype now goes through the register-tiled kernel, so the union is gone.
        let params: [u32; 5] = [m, k, n, x_stride, y_stride];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "mul_mat_tile_params");

        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: w.tensor.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: y.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });

        self.encode(enc, pipeline, &bg, (wg_m, wg_n, 1), label);
    }

    /// Encode `bias_add`: broadcast a `dim`-length bias across all `n` token
    /// rows of `buf` (`buf[t*dim + j] += bias[j]`). Qwen2 QKV bias; the batch
    /// path packs Q/K/V densely (stride == dim), so the shader's `i % dim`
    /// indexing lands on the right element.
    fn encode_bias_add_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        buf: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        n: u32,
        dim: u32,
    ) {
        let total = n * dim;
        let params: [u32; 2] = [total, dim];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "bias_add_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.bias_add.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bias.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.bias_add,
            &bg,
            (total.div_ceil(256), 1, 1),
            "bias_add_batch",
        );
    }

    /// Encode `qk_norm_rope_batch`: in-place rmsnorm + RoPE on Q (n × n_heads
    /// × head_dim) and K (n × n_kv_heads × head_dim) at positions
    /// `start_pos + token_idx`.
    #[allow(clippy::too_many_arguments)]
    fn encode_qk_norm_rope_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_batch: &wgpu::Buffer,
        k_batch: &wgpu::Buffer,
        q_norm_w: Option<&wgpu::Buffer>,
        k_norm_w: Option<&wgpu::Buffer>,
        start_pos: u32,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        q_stride: u32,
        k_stride: u32,
    ) {
        // QK-norm (per-head rmsnorm before RoPE) only applies to archs that
        // carry per-head norm weights (Qwen3/LFM2). Dense transformers
        // (llama/qwen2/mistral/granite) run rope-only; the kernel still needs
        // valid buffers bound at slots 2/3, so use `rope_freqs_buf` as a dummy.
        //
        // Require BOTH norms present to enable QK-norm: with only one present the
        // shader (has_qk_norm=1) would normalize the other head type against the
        // 1-element dummy buffer — a silent OOB read. Every QK-norm arch carries
        // both, so the assert documents that invariant rather than guarding a
        // live case.
        debug_assert_eq!(
            q_norm_w.is_some(),
            k_norm_w.is_some(),
            "QK-norm weights must be both present or both absent",
        );
        let has_qk_norm = q_norm_w.is_some() && k_norm_w.is_some();
        let q_norm = q_norm_w.unwrap_or(&self.rope_freqs_buf);
        let k_norm = k_norm_w.unwrap_or(&self.rope_freqs_buf);
        let params: [u32; 12] = [
            start_pos,
            n,
            n_heads,
            n_kv_heads,
            head_dim,
            self.config.rms_norm_eps.to_bits(),
            self.config.rope_theta.to_bits(),
            self.rope_type as u32,
            q_stride,
            k_stride,
            self.has_freq_factors as u32,
            has_qk_norm as u32,
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "qk_norm_rope_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.qk_norm_rope_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: q_norm.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: k_norm.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.rope_freqs_buf.as_entire_binding(),
                    },
                ],
            });
        let tg_count = n * (n_heads + n_kv_heads);
        self.encode(
            enc,
            &self.pipelines.qk_norm_rope_batch,
            &bg,
            (tg_count, 1, 1),
            "qk_norm_rope_batch",
        );
    }

    /// Encode `conv1d_fused_batch`. One thread per channel walks all n
    /// tokens sequentially; rolling-buffer state is in `rbuffer` and is
    /// updated in place.
    #[allow(clippy::too_many_arguments)]
    fn encode_conv1d_fused_batch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        proj: &wgpu::Buffer,
        rbuffer: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
        hs: u32,
    ) {
        let kernel_size = self.config.conv_kernel_size.unwrap_or(3) as u32;
        let d_conv = kernel_size - 1;
        let params: [u32; 6] = [hs, kernel_size, d_conv, n, 3 * hs, hs];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "conv1d_fused_batch_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.conv1d_fused_batch.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: proj.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rbuffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        let groups = hs.div_ceil(256);
        self.encode(
            enc,
            &self.pipelines.conv1d_fused_batch,
            &bg,
            (groups, 1, 1),
            "conv1d_fused_batch",
        );
    }

    /// Encode `attention_prefill`. Reads Q from `q_batch`, K/V from the
    /// model's KV caches, writes per-(token, head) output to `out_batch`.
    /// `scores_buf` is a per-(query, head, time) scratch slab.
    #[allow(clippy::too_many_arguments)]
    fn encode_attention_prefill(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q_batch: &wgpu::Buffer,
        k_cache: &wgpu::Buffer,
        v_cache: &wgpu::Buffer,
        out_batch: &wgpu::Buffer,
        n: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        kv_dim: u32,
        max_seq: u32,
        start_pos: u32,
        q_stride: u32,
        out_stride: u32,
        scale: f32,
    ) {
        let params: [u32; 12] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            max_seq,
            scale.to_bits(),
            start_pos,
            n,
            q_stride,
            out_stride,
            0,
            0,
        ];
        let p_buf = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&params), "attention_prefill_params");
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.attention_prefill.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.prefill_scores_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        self.encode(
            enc,
            &self.pipelines.attention_prefill,
            &bg,
            (n_heads, n, 1),
            "attention_prefill",
        );
    }

    /// Batched prefill — single-pass over `n` tokens for all layers, then
    /// final output norm + LM head on the last token only.
    ///
    /// Preconditions (caller-enforced):
    ///   * `start_pos == 0`. (Continuation prefills go through the
    ///     per-token loop.)
    ///   * `1 <= tokens.len() <= MAX_PREFILL_TOKENS`.
    ///   * All matmul weights on every layer are Q4_0
    ///     (`all_matmul_weights_q4_0() == true`).
    ///   * Caller already holds `infer_lock`.
    ///
    /// Mirrors `MetalLfm2Model::prefill_layers_and_logits`
    /// (metal_lfm2.rs:2906); the Metal version is the canonical
    /// reference for the dispatch order + buffer assignment.
    fn forward_prefill_batched_locked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        debug_assert!(!tokens.is_empty());
        let n = tokens.len();
        // Bounds checks — make a misuse fail deterministically rather
        // than show up later as a wgpu validation error during a buffer
        // copy or as silent out-of-bounds attention reads.
        assert!(
            start_pos + n <= self.gpu_state.max_seq_len,
            "prefill start_pos {start_pos} + n {n} exceeds max_seq_len {}",
            self.gpu_state.max_seq_len,
        );
        debug_assert!(
            n <= self.gpu_state.max_seq_len.min(MAX_PREFILL_TOKENS),
            "n {n} exceeds chunk capacity (max_seq_len = {}, MAX_PREFILL_TOKENS = {MAX_PREFILL_TOKENS})",
            self.gpu_state.max_seq_len,
        );
        // `start_pos > 0` is supported for chunked prefills — the
        // dispatcher walks through chunks of up to
        // `min(max_seq_len, MAX_PREFILL_TOKENS)` and increments
        // `start_pos` per chunk.

        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let is = cfg.intermediate_size;

        // Reset profiler spans + seq_len mirror so this chunk owns its
        // own profile output and starts clean. Conv buffer zeroing is
        // the dispatcher's responsibility (happens once per fresh
        // prefill, regardless of which path runs and how many chunks).
        self.ctx.reset_profiler();
        self.gpu_state.seq_len.store(start_pos, Ordering::Relaxed);

        // Active LoRA adapter (staged by `resolve_lora`); `None` on the base path.
        // Each in-batch hook is a no-op unless the adapter touches that target.
        let lora = self
            .active_lora
            .lock()
            .expect("active_lora poisoned")
            .clone();

        // ─── Stage embeddings into prefill_batch_buf ──────────────────────
        // CPU-side gather + one queue.write_buffer (the `embedding_f32`
        // table is pre-dequantized at load time and lives on the host).
        let mut staged: Vec<f32> = Vec::with_capacity(n * hs);
        for &t in tokens {
            let off = (t as usize) * hs;
            staged.extend_from_slice(&self.gpu_state.embedding_f32[off..off + hs]);
        }
        self.ctx
            .queue
            .write_buffer(&self.prefill_batch_buf, 0, bytemuck::cast_slice(&staged));

        // Reset the batched-LoRA params pool cursor — only when an adapter is
        // active (the base path encodes no LoRA dispatches, so it needn't touch
        // the pool lock). This call encodes into one command buffer + one submit,
        // so `next_lora_params` hands out a distinct pooled buffer per GEMM
        // dispatch starting from 0.
        if lora.is_some() {
            self.lora_params_pool
                .lock()
                .expect("lora_params_pool poisoned")
                .1 = 0;
        }

        let mut enc = self.new_encoder();
        let n_u = n as u32;
        let hs_u = hs as u32;
        let is_u = is as u32;

        for layer in 0..cfg.n_layers {
            let lw = &self.layers[layer];

            // ─── Phase 1: rmsnorm (or fused add_rmsnorm with prev FFN
            //              residual) → prefill_normed_buf ─────────────────
            if layer > 0 {
                // Fuse: batch_buf += prev_layer_ffn_down (`prefill_up_buf`),
                // then rmsnorm into `prefill_normed_buf`.
                //
                // Metal aliases dst === residual on `prefill_normed_buf`;
                // wgpu 24's binding-aliasing validator rejects that
                // pattern (binding 1 read_write + binding 4 read on the
                // same buffer in one dispatch). Route FFN down to
                // `prefill_up_buf` so dst and residual stay distinct.
                self.encode_add_rmsnorm_batch(
                    &mut enc,
                    &self.prefill_batch_buf,
                    &self.prefill_normed_buf,
                    &lw.attn_norm,
                    &self.prefill_up_buf,
                    n_u,
                    hs_u,
                );
            } else {
                self.encode_rmsnorm_batch(
                    &mut enc,
                    &self.prefill_batch_buf,
                    &self.prefill_normed_buf,
                    &lw.attn_norm,
                    n_u,
                    hs_u,
                );
            }

            if cfg.block_types[layer] == BlockType::GatedConv {
                let conv_buf = self.gpu_state.conv_buffers[layer].as_ref().unwrap();
                let w_in = lw.conv_in_proj.as_ref().unwrap();
                let w_out = lw.conv_out_proj.as_ref().unwrap();
                let conv_weight = lw.conv_weight.as_ref().unwrap();

                // Phase 2: in_proj batched GEMM (3*hs columns per token).
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_in,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    3 * hs_u,
                );
                // LoRA conv in_proj — must run before the fused conv1d overwrites
                // `prefill_normed_buf` (which still holds the rmsnorm output that
                // feeds the LoRA `A`).
                self.encode_lora_hook_batched(
                    &mut enc,
                    lora.as_ref(),
                    layer,
                    LoraTarget::ShortconvInProj,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                );

                // Phase 3: fused conv1d (1 dispatch over all N tokens;
                // rolling buffer state walks sequentially per channel).
                self.encode_conv1d_fused_batch(
                    &mut enc,
                    &self.prefill_proj_buf,
                    conv_buf,
                    conv_weight,
                    &self.prefill_normed_buf,
                    n_u,
                    hs_u,
                );

                // Phase 4: out_proj GEMM → prefill_gate_buf (residual
                // scratch; FFN's add_rmsnorm_batch will fuse the add).
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_out,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    hs_u,
                );
                // LoRA conv out_proj — input is the post-conv gated output (now in
                // `prefill_normed_buf`), accumulated into the residual scratch.
                self.encode_lora_hook_batched(
                    &mut enc,
                    lora.as_ref(),
                    layer,
                    LoraTarget::ShortconvOutProj,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                );
            } else {
                // Attention layer.
                //
                // Use `cfg.head_dim`, NOT `hs / n_heads`: Qwen3 decouples
                // head_dim (attention.key_length), so `q_dim = n_heads*head_dim`
                // and `kv_dim = n_kv_heads*head_dim` can both exceed `hs`. The
                // prefill scratch buffers are sized for that worst case at
                // construction; Q lives in `prefill_proj_buf` with stride
                // `q_dim`, the attention output in `prefill_normed_buf` with
                // the same stride, and out_proj maps `q_dim → hs`.
                let head_dim = cfg.head_dim as u32;
                let n_kv_heads = cfg.kv_heads_per_layer[layer] as u32;
                let kv_dim = n_kv_heads * head_dim;
                let n_heads = cfg.n_heads as u32;
                let q_dim = n_heads * head_dim;
                let (k_cache, v_cache) = self.gpu_state.kv_caches[layer].as_ref().unwrap();

                let w_q = lw.attn_q.as_ref().unwrap();
                let w_k = lw.attn_k.as_ref().unwrap();
                let w_v = lw.attn_v.as_ref().unwrap();
                let w_o = lw.attn_output.as_ref().unwrap();

                // Phase A: Q/K/V batched GEMMs.
                //   Q  → prefill_proj_buf, stride q_dim
                //   K  → prefill_gate_buf, stride kv_dim
                //   V  → prefill_up_buf,   stride kv_dim
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_q,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    q_dim,
                );
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_k,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    kv_dim,
                );
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_v,
                    &self.prefill_normed_buf,
                    &self.prefill_up_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    kv_dim,
                );

                // LoRA Q/K/V deltas: `+= scale·B·(A·normed)` on the raw
                // projections, before QK-norm/RoPE (and the Qwen2 bias) — mirrors
                // the decode hooks and the CPU `apply_attn_qkv`. Q → proj_buf,
                // K → gate_buf, V → up_buf, all token-major (input is the shared
                // attn_norm output in `prefill_normed_buf`).
                self.encode_lora_hook_batched(
                    &mut enc,
                    lora.as_ref(),
                    layer,
                    LoraTarget::AttnQ,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                );
                self.encode_lora_hook_batched(
                    &mut enc,
                    lora.as_ref(),
                    layer,
                    LoraTarget::AttnK,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                );
                self.encode_lora_hook_batched(
                    &mut enc,
                    lora.as_ref(),
                    layer,
                    LoraTarget::AttnV,
                    &self.prefill_normed_buf,
                    &self.prefill_up_buf,
                    n_u,
                );

                // Phase A2: QKV bias (Qwen2) — broadcast-add the bias vector
                // across all N token rows, right after each projection and
                // before QK-norm/RoPE. Absent on every other arch.
                if let Some(b) = lw.attn_q_bias.as_ref() {
                    self.encode_bias_add_batch(&mut enc, &self.prefill_proj_buf, b, n_u, q_dim);
                }
                if let Some(b) = lw.attn_k_bias.as_ref() {
                    self.encode_bias_add_batch(&mut enc, &self.prefill_gate_buf, b, n_u, kv_dim);
                }
                if let Some(b) = lw.attn_v_bias.as_ref() {
                    self.encode_bias_add_batch(&mut enc, &self.prefill_up_buf, b, n_u, kv_dim);
                }

                // Phase B: batched per-head Q/K rmsnorm (QK-norm, Qwen3/LFM2
                // only) + RoPE. Pass `None` norms for archs without QK-norm so
                // the kernel runs rope-only.
                self.encode_qk_norm_rope_batch(
                    &mut enc,
                    &self.prefill_proj_buf,
                    &self.prefill_gate_buf,
                    lw.attn_q_norm.as_ref(),
                    lw.attn_k_norm.as_ref(),
                    start_pos as u32,
                    n_u,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    q_dim,
                    kv_dim,
                );

                // Phase C: bulk-write K/V into the cache. The KV cache is
                // `max_seq_len × kv_dim` f32; write `n × kv_dim` floats starting
                // at row `start_pos × kv_dim`. `encode_copy` is a no-shader
                // memcpy that scales these f32 counts to bytes internally.
                let kv_off_floats = (start_pos * kv_dim as usize) as u64;
                let kv_chunk_floats = (n * kv_dim as usize) as u64;
                Self::encode_copy(
                    &mut enc,
                    &self.prefill_gate_buf,
                    0,
                    k_cache,
                    kv_off_floats,
                    kv_chunk_floats,
                );
                Self::encode_copy(
                    &mut enc,
                    &self.prefill_up_buf,
                    0,
                    v_cache,
                    kv_off_floats,
                    kv_chunk_floats,
                );

                // Phase D: batched causal attention. Q stride and the output
                // stride are both `q_dim` (concatenated head outputs). Granite
                // overrides the softmax scale via `scalars.attn`; every other
                // arch uses 1/sqrt(head_dim).
                let max_seq_for_kv = (start_pos + n) as u32;
                let attn_scale = self
                    .scalars
                    .attn
                    .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
                self.encode_attention_prefill(
                    &mut enc,
                    &self.prefill_proj_buf,
                    k_cache,
                    v_cache,
                    &self.prefill_normed_buf,
                    n_u,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    max_seq_for_kv,
                    start_pos as u32,
                    q_dim,
                    q_dim,
                    attn_scale,
                );

                // Phase E: output projection (`q_dim → hs`) → prefill_gate_buf
                // (residual scratch; FFN's add_rmsnorm_batch fuses the add).
                self.encode_mul_mat_reg_tile(
                    &mut enc,
                    w_o,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    q_dim,
                    q_dim,
                    hs_u,
                );

                // LoRA attn-output delta into gate_buf, BEFORE the FFN's fused
                // `add_rmsnorm_batch` below scales it by `scalars.residual` (so
                // `residual_mult` wraps the delta — hence `b_batched` carries scale
                // only). Input is the attention output (o_proj input) in
                // `prefill_normed_buf`.
                self.encode_lora_hook_batched(
                    &mut enc,
                    lora.as_ref(),
                    layer,
                    LoraTarget::AttnOutput,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                );
            }

            // ─── Phase 7: FFN ──────────────────────────────────────────────
            // Fused add(prefill_gate_buf residual) + ffn_norm.
            self.encode_add_rmsnorm_batch(
                &mut enc,
                &self.prefill_batch_buf,
                &self.prefill_normed_buf,
                &lw.ffn_norm,
                &self.prefill_gate_buf,
                n_u,
                hs_u,
            );
            // gate + up GEMMs.
            self.encode_mul_mat_reg_tile(
                &mut enc,
                &lw.ffn_gate,
                &self.prefill_normed_buf,
                &self.prefill_gate_buf,
                n_u,
                hs_u,
                hs_u,
                is_u,
            );
            self.encode_mul_mat_reg_tile(
                &mut enc,
                &lw.ffn_up,
                &self.prefill_normed_buf,
                &self.prefill_up_buf,
                n_u,
                hs_u,
                hs_u,
                is_u,
            );
            // LoRA gate/up deltas on the raw projections, before silu_mul. Input
            // is the ffn_norm output in `prefill_normed_buf`; outputs token-major
            // (gate → gate_buf, up → up_buf). Applies to every layer (conv + attn).
            self.encode_lora_hook_batched(
                &mut enc,
                lora.as_ref(),
                layer,
                LoraTarget::FfnGate,
                &self.prefill_normed_buf,
                &self.prefill_gate_buf,
                n_u,
            );
            self.encode_lora_hook_batched(
                &mut enc,
                lora.as_ref(),
                layer,
                LoraTarget::FfnUp,
                &self.prefill_normed_buf,
                &self.prefill_up_buf,
                n_u,
            );
            // silu_mul over the full N × is buffer.
            {
                let total = n_u * is_u;
                let params: [u32; 2] = [total, 0];
                let p_buf = self
                    .ctx
                    .upload_storage(bytemuck::cast_slice(&params), "silu_mul_batch_params");
                let bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.silu_mul_inplace.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.prefill_gate_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.prefill_up_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: p_buf.as_entire_binding(),
                            },
                        ],
                    });
                self.encode(
                    &mut enc,
                    &self.pipelines.silu_mul_inplace,
                    &bg,
                    (total.div_ceil(256), 1, 1),
                    "silu_mul_batch",
                );
            }
            // FFN down → prefill_up_buf (next layer's residual scratch).
            // The next layer's add_rmsnorm_batch reads from this buffer
            // as `residual`; using `prefill_up_buf` (rather than
            // `prefill_normed_buf` which Metal uses) keeps the dst and
            // residual bindings on distinct buffers — see the Phase 1
            // comment above for the wgpu validation reason. The buffer
            // is is×N, plenty of room for hs×N writes.
            self.encode_mul_mat_reg_tile(
                &mut enc,
                &lw.ffn_down,
                &self.prefill_gate_buf,
                &self.prefill_up_buf,
                n_u,
                is_u,
                is_u,
                hs_u,
            );
            // LoRA ffn-down delta into prefill_up_buf, BEFORE the next layer's
            // fused `add_rmsnorm_batch` (or the final `scaled_add_inplace`) scales
            // it by `scalars.residual`. Input is the silu_mul(gate,up) result in
            // `prefill_gate_buf`.
            self.encode_lora_hook_batched(
                &mut enc,
                lora.as_ref(),
                layer,
                LoraTarget::FfnDown,
                &self.prefill_gate_buf,
                &self.prefill_up_buf,
                n_u,
            );
        }

        // ─── Final residual add: batch_buf += residual_scale·prefill_up_buf ─
        // Last layer's FFN down residual lives in `prefill_up_buf`; add it back
        // into the running residual stream. `scaled_add_inplace` folds Granite's
        // residual multiplier into the addend (1.0 ⇒ plain add elsewhere).
        {
            let total = n_u * hs_u;
            let params: [u32; 2] = [total, self.scalars.residual.to_bits()];
            let p_buf = self
                .ctx
                .upload_storage(bytemuck::cast_slice(&params), "final_add_params");
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.scaled_add_inplace.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.prefill_batch_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.prefill_up_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: p_buf.as_entire_binding(),
                        },
                    ],
                });
            self.encode(
                &mut enc,
                &self.pipelines.scaled_add_inplace,
                &bg,
                (total.div_ceil(256), 1, 1),
                "final_add",
            );
        }

        // ─── Final output: norm + LM head, last token only ────────────────
        // Copy batch_buf[(n-1)*hs..n*hs] → hidden_buf (single-token
        // scratch), then rmsnorm + output projection through the existing
        // single-token helpers.
        let last_off_floats = ((n - 1) * hs) as u64;
        Self::encode_copy(
            &mut enc,
            &self.prefill_batch_buf,
            last_off_floats,
            &self.hidden_buf,
            0,
            hs as u64,
        );
        self.encode_rmsnorm(
            &mut enc,
            &self.hidden_buf,
            &self.output_norm,
            hs_u,
            cfg.rms_norm_eps,
        );
        // Output projection. Untied models (`output.weight`) project through
        // it; tied models reuse the embedding table. Mirrors the decode path.
        let out_proj = self.output_weight.as_ref().unwrap_or(&self.embedding);
        self.encode_gemv_f16(
            &mut enc,
            out_proj,
            &self.hidden_buf,
            &self.logits_buf,
            cfg.vocab_size as u32,
            hs_u,
        );
        // Granite divides the logits by `logits_scaling` (identity elsewhere).
        if let Some(params) = self.logit_scale_params.as_ref() {
            let scale_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("logit_scale_bg"),
                    layout: &self.pipelines.scale_f32.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.logits_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params.as_entire_binding(),
                        },
                    ],
                });
            self.encode(
                &mut enc,
                &self.pipelines.scale_f32,
                &scale_bg,
                ((cfg.vocab_size as u32).div_ceil(256), 1, 1),
                "logit_scale",
            );
        }

        self.submit_and_wait(enc);

        // Update seq_len mirrors after the GPU work completes.
        self.gpu_state
            .seq_len
            .store(start_pos + n, Ordering::Relaxed);
        state.seq_len = start_pos + n;
        self.ctx.finish_profiler();

        self.ctx.download_f32(&self.logits_buf, cfg.vocab_size)
    }
}

impl GpuLfm2Model {
    /// Lock-free body of `Model::snapshot_state`. Callers that already
    /// hold `infer_lock` (e.g. `forward_prefill`'s prefix-cache write
    /// step) call this directly to avoid a recursive `Mutex::lock()`
    /// deadlock — `std::sync::Mutex` is not reentrant.
    ///
    /// Snapshot layout (mirrors Metal's pattern but with f32 KV instead
    /// of f16): per attention layer, download the live `seq_len * kv_dim`
    /// floats from K and V; per conv layer, download the full
    /// `d_conv * hidden_size` rolling buffer. f32 → bytes via
    /// `bytemuck::cast_slice` on the contiguous `Vec<f32>` from
    /// `download_f32` (source-aligned, safe).
    fn snapshot_state_locked(&self) -> StateSnapshot {
        let seq_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
        let cfg = &self.config;
        // Use config.head_dim, NOT hidden_size/n_heads: Qwen3 decouples head_dim
        // (attention.key_length), so the KV cache is sized by config.head_dim. The
        // stale formula under-counts the snapshot/restore floats and corrupts the
        // KV cache on a prefix-cache hit. Matches the from_weight_source alloc.
        let head_dim = cfg.head_dim;
        let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;

        // `download_f32` now slices the staging buffer to exactly
        // `count * 4` bytes, so the returned `Vec<f32>` length
        // equals `count` directly — no truncation needed. The
        // closure is kept as the single calling site so a future
        // regression in `download_f32` re-introduces a single edit
        // point, not N call sites.
        let download_exact =
            |buf: &wgpu::Buffer, count: usize| -> Vec<f32> { self.ctx.download_f32(buf, count) };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if cfg.block_types[i] == BlockType::Attention {
                let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                let count = seq_len * kv_dim;
                let (k_buf, v_buf) = self.gpu_state.kv_caches[i]
                    .as_ref()
                    .expect("attention layer must have KV buffers");
                let k_floats = download_exact(k_buf, count);
                let v_floats = download_exact(v_buf, count);
                layers.push(LayerSnapshot::Attention {
                    k_data: bytemuck::cast_slice(&k_floats).to_vec(),
                    v_data: bytemuck::cast_slice(&v_floats).to_vec(),
                });
            } else {
                let count = d_conv * cfg.hidden_size;
                let conv_buf = self.gpu_state.conv_buffers[i]
                    .as_ref()
                    .expect("conv layer must have rolling buffer");
                let floats = download_exact(conv_buf, count);
                layers.push(LayerSnapshot::Conv {
                    buffer: bytemuck::cast_slice(&floats).to_vec(),
                });
            }
        }
        StateSnapshot { layers, seq_len }
    }

    /// Lock-free body of `Model::restore_state`. See
    /// [`Self::snapshot_state_locked`] for the locking contract.
    /// Writes raw bytes via `queue.write_buffer` at offset 0 — wgpu's
    /// `COPY_BUFFER_ALIGNMENT` is 4, which f32 byte counts always
    /// satisfy. The remainder of the pre-allocated cache (past
    /// `seq_len * kv_dim`) is left as-is; the kernels only read up
    /// to the seq_len reported by the atomic, so stale tail data
    /// can't influence subsequent forwards.
    fn restore_state_locked(&self, snapshot: &StateSnapshot) {
        let cfg = &self.config;
        for (i, layer_snap) in snapshot.layers.iter().enumerate() {
            match layer_snap {
                LayerSnapshot::Attention { k_data, v_data } => {
                    assert_eq!(
                        cfg.block_types[i],
                        BlockType::Attention,
                        "snapshot layer {i} attention vs state config"
                    );
                    let (k_buf, v_buf) = self.gpu_state.kv_caches[i]
                        .as_ref()
                        .expect("attention layer must have KV buffers");
                    self.ctx.queue.write_buffer(k_buf, 0, k_data);
                    self.ctx.queue.write_buffer(v_buf, 0, v_data);
                }
                LayerSnapshot::Conv { buffer } => {
                    assert_eq!(
                        cfg.block_types[i],
                        BlockType::GatedConv,
                        "snapshot layer {i} conv vs state config"
                    );
                    let conv_buf = self.gpu_state.conv_buffers[i]
                        .as_ref()
                        .expect("conv layer must have rolling buffer");
                    self.ctx.queue.write_buffer(conv_buf, 0, buffer);
                }
                LayerSnapshot::AttentionCompressed { .. } => {
                    // Unreachable in normal operation: wgpu doesn't
                    // configure TurboQuant compression. `model_id`
                    // is `"wgpu:..."` vs CPU's `"cpu:..."`, separating
                    // their on-disk namespaces. Panic on the hard
                    // error path so an accidental cross-namespace
                    // load surfaces fast instead of corrupting state.
                    panic!(
                        "GpuLfm2Model::restore_state_locked received \
                         a TurboQuant-compressed snapshot at layer {i}; \
                         wgpu does not support TurboQuant. This indicates \
                         a cross-backend cache-namespace leak."
                    );
                }
            }
        }
        self.gpu_state
            .seq_len
            .store(snapshot.seq_len, Ordering::Relaxed);
    }

    /// Zero every conv layer's GPU rolling buffer. Called on a fresh
    /// prefill (`start_pos == 0`) cache MISS so stale conv state
    /// from a prior generation can't leak into the new run. Cache
    /// HITs go through `restore_state_locked` which overwrites the
    /// buffers from the snapshot, so this only fires on the cold
    /// path. Mirrors `MetalLfm2Model::zero_conv_buffers_locked`.
    ///
    /// Conv layers always read the entire rolling buffer regardless
    /// of `seq_len`, so the seq_len atomic reset alone isn't enough
    /// to fence stale state. Without this an FFI / long-lived
    /// process that reuses the same `GpuLfm2Model` across multiple
    /// `Session`s would drift on conv state.
    ///
    /// Uses wgpu's native `clear_buffer` so the zero fill happens
    /// GPU-side — no CPU-allocated zero buffer, no CPU→GPU upload.
    /// One encoder, one submit, regardless of layer count.
    fn zero_conv_buffers_locked(&self) {
        let cfg = &self.config;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zero_conv_buffers"),
            });
        for i in 0..cfg.n_layers {
            if cfg.block_types[i] == BlockType::GatedConv
                && let Some(conv_buf) = self.gpu_state.conv_buffers[i].as_ref()
            {
                // `None` size = clear entire buffer.
                enc.clear_buffer(conv_buf, 0, None);
            }
        }
        self.ctx.submit_encoder(enc);
    }
}

impl Model for GpuLfm2Model {
    fn supports_hidden_states(&self) -> bool {
        true
    }

    /// Per-token post-final-norm hidden states, row-major `[n * hidden_size]`
    /// (llama.cpp `--pooling none`). Reuses `forward_inner_compute` per token
    /// (which drives the KV offset + attention window from `gpu_state.seq_len`):
    /// routes KV to the dedicated `HsScratch` caches via `use_hs_scratch` and
    /// drives `seq_len` from 0, so it's a fresh-context extraction on scratch KV
    /// that never touches the generation caches — the GPU analog of the CPU
    /// path's separate scratch state. Reads back the in-place post-`output_norm`
    /// `hidden_buf`; the logits it also computes are ignored. `state` is read
    /// only to stage the active LoRA adapter (wgpu keeps KV on the model, not in
    /// `state`). A drop-guard restores the generation `seq_len` and clears the
    /// flag on any exit, including a mid-run panic.
    ///
    /// Like [`Self::forward`], this is the **synchronous** native path: it blocks
    /// on `download_f32` per token. The browser/WASM GPU path is the async
    /// `WebGpuSession`, which never routes through this method — so the blocking
    /// readback here is a native-only concern, identical to `forward`.
    fn hidden_states(&self, tokens: &[u32], state: &mut InferenceState) -> Vec<f32> {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        // Stage the caller's adapter for the per-token layer encoders; the guard
        // clears it on the way out.
        let _lora_guard = self.resolve_lora(state);
        assert!(
            !tokens.is_empty(),
            "hidden_states requires at least one token"
        );
        let hs = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        assert!(
            tokens.len() <= self.gpu_state.max_seq_len,
            "hidden_states chunk ({}) exceeds max_seq_len ({})",
            tokens.len(),
            self.gpu_state.max_seq_len
        );

        // Build scratch (once) and zero its conv rolling buffers so each
        // extraction starts from a clean convolution state.
        let scratch = self.hs_scratch();
        let mut enc = self.new_encoder();
        for buf in scratch.conv.iter().flatten() {
            enc.clear_buffer(buf, 0, None);
        }
        self.submit_and_wait(enc);

        // Route KV to the scratch caches and drive `gpu_state.seq_len` from 0 so
        // the fresh-context prefill walks positions 0..n on the scratch KV. The
        // drop-guard restores the generation `seq_len` and clears the flag on ANY
        // exit (incl. a mid-run panic), so generation is never corrupted.
        let saved_seq = self.gpu_state.seq_len.load(Ordering::Relaxed);
        struct HsGuard<'a> {
            flag: &'a AtomicBool,
            seq: &'a AtomicUsize,
            saved: usize,
        }
        impl Drop for HsGuard<'_> {
            fn drop(&mut self) {
                self.seq.store(self.saved, Ordering::Relaxed);
                self.flag.store(false, Ordering::Relaxed);
            }
        }
        self.gpu_state.seq_len.store(0, Ordering::Relaxed);
        self.use_hs_scratch.store(true, Ordering::Relaxed);
        let _hs_guard = HsGuard {
            flag: &self.use_hs_scratch,
            seq: &self.gpu_state.seq_len,
            saved: saved_seq,
        };

        // `forward_inner_compute` needs a `&mut InferenceState` for its `seq_len`
        // bookkeeping only (wgpu KV lives on the model), so a throwaway suffices.
        // 1-token scratch state; the `Model::hidden_states` trait signature
        // returns `Vec<f32>` (not `Result`), so this can't propagate — but the
        // allocation is trivially small (~kv_dim floats/layer), so OOM here is
        // effectively impossible. `expect` documents that.
        let mut dummy = InferenceState::for_prefill(&self.config, 1)
            .expect("hidden_states: 1-token scratch InferenceState allocation failed");
        let mut out = Vec::with_capacity(tokens.len() * hs);
        for (pos, &token) in tokens.iter().enumerate() {
            let token_id = token as usize;
            assert!(
                token_id < vocab,
                "token_id {token_id} out of range (vocab_size={vocab})"
            );
            self.forward_inner_compute(&[token], pos, &mut dummy);
            out.extend_from_slice(&self.ctx.download_f32(&self.hidden_buf, hs));
        }
        out
    }

    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        // Stage the caller's adapter for the per-layer encoders; the guard
        // clears it on the way out.
        let _lora_guard = self.resolve_lora(state);
        self.forward_inner(tokens, pos, state)
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        let _lora_guard = self.resolve_lora(state);
        self.forward_greedy_inner(tokens, pos, state)
    }

    // forward_embedding and forward_from_embedding use default impls
    // (unimplemented). Audio generation requires Metal backend for now.
    // wgpu support would need refactoring forward() to split the layer
    // dispatch from the logit projection, plus a hidden_buf download path.

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        // Stage the caller's adapter so both prefill paths apply it: the
        // batched-GEMM path runs the in-batch LoRA hooks (two NT GEMMs per
        // target), and the sequential fallback loop runs the decode hooks. The
        // guard clears `active_lora` on the way out.
        let _lora_guard = self.resolve_lora(state);
        let lora_active = state.lora.is_some();
        // Reset internal seq_len so repeated generate() calls (bench) work.
        self.gpu_state.seq_len.store(start_pos, Ordering::Relaxed);

        // Fresh prefill (`start_pos == 0`) runs the batched-GEMM path — including
        // with a LoRA active, which now applies in-batch (two NT GEMMs per target)
        // rather than forcing the per-token fallback. The prefix cache is still
        // bypassed with an adapter, though: cached KV is base-model-only, so
        // restoring it and adapting only the tail would corrupt the result (and
        // inserting adapter-modified KV would poison the cache for later base runs).
        if start_pos == 0 {
            // Cache lookup only for base-model prefills.
            let hit = (!lora_active)
                .then(|| {
                    self.prefix_cache
                        .lock()
                        .expect("prefix_cache mutex poisoned")
                        .find_longest_prefix(tokens)
                })
                .flatten();
            if let Some((snapshot, prefix_len)) = hit {
                // Strict-prefix hits only. A `prefix_len == tokens.len()`
                // hit would force `use_len = tokens.len() - 1`, but the
                // restored state already reflects "after all tokens" —
                // re-running the last token would advance the conv
                // rolling buffer one position past where it should be
                // and overwrite already-correct attention KV cells.
                // The conv layer state isn't seq_len-gated, so the
                // off-by-one would corrupt logits.
                if prefix_len < tokens.len() && prefix_len > 0 {
                    let use_len = prefix_len;
                    self.restore_state_locked(&snapshot);
                    // `restore_state_locked` set `gpu_state.seq_len`
                    // to `snapshot.seq_len == prefix_len`, which
                    // matches `use_len` in this strict-prefix path.
                    // (Kept explicit so future use_len-vs-prefix_len
                    // splits don't drift.)
                    self.gpu_state.seq_len.store(use_len, Ordering::Relaxed);
                    state.seq_len = use_len;
                    // Skip the per-token vocab-sized download_f32 for
                    // every prefill step except the last — only the
                    // final logits are returned to the caller.
                    // `prefix_len < tokens.len()` is enforced above, so
                    // `remaining` is always >= 1 here.
                    let remaining = &tokens[use_len..];
                    let last = remaining.len() - 1;
                    let mut logits = Vec::new();
                    for (j, &token) in remaining.iter().enumerate() {
                        if j == last {
                            logits = self.forward_inner(&[token], use_len + j, state);
                        } else {
                            self.forward_inner_compute(&[token], use_len + j, state);
                        }
                    }
                    self.prefix_cache
                        .lock()
                        .expect("prefix_cache mutex poisoned")
                        .insert(tokens, self.snapshot_state_locked());
                    return logits;
                }
            }
            // Cache miss on a fresh prefill: zero the GPU conv
            // rolling buffers so stale state from a prior
            // generation can't leak in. Cache hits skip this
            // (`restore_state_locked` rewrites the buffers from
            // the snapshot). Mirrors the equivalent fix on Metal.
            self.zero_conv_buffers_locked();

            // Try the batched prefill path. Preconditions:
            //   * fresh prefill (start_pos == 0, already checked above)
            //   * non-empty
            //   * every matmul weight has a batched quantized kernel
            //     (Q4_0/Q8_0/Q4KM/Q6K); any other dtype falls through to the
            //     per-token loop
            //   * the model wires the batched-prefill path (`batched_prefill`).
            //     LFM2 and the dense transformers (llama/qwen2/qwen3/mistral/
            //     granite) all support it.
            //
            // The dtype fallback is *loud*: it costs ~340x the GPU submits, so it
            // must never again be something a model quietly sits on for months.
            //
            // Long prompts are chunked through the batched path in
            // MAX_PREFILL_TOKENS-sized chunks so the scratch buffers stay
            // bounded. Each chunk advances `start_pos`; conv rolling
            // state and KV cache writes carry across chunks naturally.
            let unbatchable = self.unbatchable_matmul_weight();
            if let Some((layer, name, dtype)) = unbatchable
                && !tokens.is_empty()
                && self.batched_prefill
                && !self.batched_fallback_warned.swap(true, Ordering::Relaxed)
            {
                tracing::warn!(
                    layer,
                    tensor = name,
                    ?dtype,
                    "no batched prefill GEMM for this dtype — falling back to the \
                     per-token loop, which issues ~340x the GPU submits and makes \
                     prefill no faster than decode. Add a batched kernel for {dtype:?} \
                     to put this model back on the fast path.",
                );
            }
            if !tokens.is_empty() && self.batched_prefill && unbatchable.is_none() {
                // Chunk size respects both the static MAX_PREFILL_TOKENS
                // budget AND the model's actual `max_seq_len` — otherwise
                // a caller with `--context-size < 512` would dispatch
                // batched chunks larger than the KV cache and OOB on the
                // copy_buffer_to_buffer write.
                let chunk_size = self.gpu_state.max_seq_len.min(MAX_PREFILL_TOKENS);
                let mut logits = Vec::new();
                let mut pos = 0usize;
                while pos < tokens.len() {
                    let end = (pos + chunk_size).min(tokens.len());
                    // `start_pos + pos` rather than `pos`: defensive against
                    // a future caller passing non-zero start_pos through
                    // this branch (today the outer `if start_pos == 0`
                    // gate makes them equal).
                    logits = self.forward_prefill_batched_locked(
                        &tokens[pos..end],
                        start_pos + pos,
                        state,
                    );
                    pos = end;
                }
                // Only cache base-model KV — an adapted run's KV must never be reused.
                if !lora_active {
                    self.prefix_cache
                        .lock()
                        .expect("prefix_cache mutex poisoned")
                        .insert(tokens, self.snapshot_state_locked());
                }
                return logits;
            }
        }

        // Cache miss (or continuation prefill): full prefill loop.
        // Sequential single-token forward via the lock-free body — calling
        // `self.forward()` here would re-acquire the (non-reentrant)
        // `infer_lock` we already hold and deadlock.
        //
        // For every step except the last, drive the GPU via
        // `forward_inner_compute` so the per-token vocab-sized
        // `download_f32` is skipped — only the final iteration's
        // logits make it back to the caller. At p=4096 this drops
        // 4095 vocab-sized blocking readbacks (vocab × 4 bytes ×
        // 4095 = ~1 GB at vocab=65536). Empty `tokens` makes
        // `last` underflow — guarded by `if !tokens.is_empty()`.
        let mut logits = Vec::new();
        if !tokens.is_empty() {
            let last = tokens.len() - 1;
            for (i, &token) in tokens.iter().enumerate() {
                if i == last {
                    logits = self.forward_inner(&[token], start_pos + i, state);
                } else {
                    self.forward_inner_compute(&[token], start_pos + i, state);
                }
            }
        }
        // Skip the cache insert with a LoRA active: the snapshot's KV reflects
        // the adapter, not the base model, and would poison later base runs.
        if start_pos == 0 && !lora_active {
            self.prefix_cache
                .lock()
                .expect("prefix_cache mutex poisoned")
                .insert(tokens, self.snapshot_state_locked());
        }
        logits
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        *self
            .prefix_cache
            .lock()
            .expect("prefix_cache mutex poisoned") =
            KvPrefixCache::new(config, &self.config, &format!("wgpu:{}", self.model_id));
    }

    /// Public Model trait surface for `_locked` snapshot/restore so
    /// external state-management callers (FFI / parity harness)
    /// can drive the prefix cache directly without going through
    /// `forward_prefill`. Mirrors `MetalLfm2Model`'s overrides.
    fn snapshot_state(&self) -> StateSnapshot {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        self.snapshot_state_locked()
    }

    fn restore_state(&self, snapshot: &StateSnapshot) {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        self.restore_state_locked(snapshot);
    }

    fn supports_kv_shift(&self) -> bool {
        // Mirror of CPU `Lfm2Model` / Metal `MetalLfm2Model` — the wgpu backend
        // implements the GPU-side shift via the `kv_shift` WGSL kernel +
        // `copy_buffer_to_buffer`. See `Self::shift_kv`.
        true
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        let _guard = self.infer_lock.lock().expect("infer_lock poisoned");
        assert!(shift > 0, "shift must be > 0");
        let cur_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
        // This bounds check (and the caller's `Session::can_shift` gate) authorize
        // the shift off counters that are equal only by the maintained
        // `gpu_state.seq_len == state.seq_len == current_pos` invariant. Assert the
        // two mirrors agree HERE so a future path that desyncs them trips loudly at
        // the source, instead of silently computing `new_seq_len` from the wrong
        // base or panicking on the bounds assert below with a confusing message.
        debug_assert_eq!(
            state.seq_len, cur_len,
            "seq_len mirrors out of sync: state.seq_len={} gpu_state.seq_len={cur_len}",
            state.seq_len,
        );
        assert!(
            n_keep + shift <= cur_len,
            "shift range out of bounds: n_keep={n_keep} + shift={shift} > seq_len={cur_len}",
        );
        // The wgpu KV cache is dense f32 — TurboQuant compression is a CPU-only
        // feature, so a compressed state should never reach this backend. Match
        // the Metal gate so a caller branching on the flag surfaces fast.
        assert!(
            !state.is_compressed(),
            "shift_kv called on a TurboQuant-compressed state; \
             shifting compressed caches is not supported on the wgpu backend"
        );

        let new_seq_len = cur_len - shift;
        let retained = new_seq_len - n_keep;

        // Edge case: a shift that drops EVERY non-keep cell
        // (`cur_len == n_keep + shift`) leaves nothing to re-rotate or copy.
        // Skip the per-layer GPU work — a 0-element dispatch / 0-byte
        // `copy_buffer_to_buffer` is a wgpu validation error — and only update
        // the seq_len mirrors below. Reachable exactly as on Metal: `n_keep=32`,
        // `max_seq_len=256`, an append to `cur_len=256` needs `shift=224 =
        // cur_len - n_keep`.
        if retained > 0 {
            self.encode_kv_shift_layers(n_keep, shift, retained);
        }

        // Decrement both seq_len mirrors: the GPU-side `AtomicUsize` drives the
        // forward path's KV write offsets + bounds checks; `state.seq_len` is the
        // value the Session reads.
        self.gpu_state.seq_len.store(new_seq_len, Ordering::Relaxed);
        state.seq_len = new_seq_len;
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use crate::backend::wgpu::GpuContext;

    /// Acquire a GPU context or skip. Under `CERA_REQUIRE_GPU` (the lavapipe CI
    /// job) a missing adapter is a hard failure, mirroring the oracle tests, so
    /// the contract below cannot pass by silently skipping.
    fn gpu_ctx_or_skip() -> Option<GpuContext> {
        match GpuContext::new() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                let required = std::env::var("CERA_REQUIRE_GPU").unwrap_or_default();
                assert!(
                    required.is_empty(),
                    "CERA_REQUIRE_GPU is set but no GPU adapter is available: {e}"
                );
                eprintln!("skipping: no GPU adapter ({e})");
                None
            }
        }
    }

    /// The tiled LM-head GEMV binds weight row-slices at byte offset
    /// `row_start * k * elem_size`; every such offset must be a multiple of the
    /// adapter's storage-buffer offset alignment and every tile must fit
    /// `max_binding`. `gemv_tile_rows` picks the rows-per-tile that guarantees
    /// both, for f16 (elem=2, the LM head) and f32 (elem=4). Pure host math.
    #[test]
    fn gemv_tile_rows_fits_and_aligns() {
        use super::gemv_tile_rows;
        // Whole matrix fits one binding → single tile.
        assert_eq!(gemv_tile_rows(1000, 512, 1 << 30, 256, 2), 1000);
        assert_eq!(gemv_tile_rows(1000, 512, 1 << 30, 256, 4), 1000);

        // `k` values chosen so a row is NOT a multiple of the offset alignment,
        // so the row-alignment rounding actually does work: k=100 (row_bytes 200
        // for f16, gcd 8 with 256) and k=99 — odd, so the f16 row_bytes 198 is
        // 2-mod-4, exercising the non-whole-u32 row case the round-up in
        // `encode_gemv_f16_tiled` guards.
        let m = 131_072u32;
        let align = 256u64;
        for &k in &[100u32, 99u32] {
            for &elem in &[2u64, 4u64] {
                let row_bytes = u64::from(k) * elem;
                for &max_binding in &[1u64 << 20, 4 << 20, 512 << 10] {
                    let rows = gemv_tile_rows(m, k, max_binding, align, elem);
                    assert!(rows > 0, "k={k} elem={elem} max={max_binding}");
                    assert!(
                        u64::from(rows) * row_bytes <= max_binding,
                        "tile exceeds max_binding (k={k}, elem={elem}, max={max_binding})",
                    );
                    // Offsets are multiples of `rows * row_bytes`, so that product
                    // must be a multiple of the offset alignment.
                    assert_eq!(
                        (u64::from(rows) * row_bytes) % align,
                        0,
                        "tile byte size not offset-aligned (k={k}, elem={elem}, max={max_binding})",
                    );
                    // The final tile's binding size is rounded up to a whole u32
                    // (`array<u32>` view). Offset + rounded size must still land
                    // inside the 4-byte-padded weight buffer.
                    let final_rows = m % rows;
                    if final_rows > 0 {
                        let offset = u64::from(m - final_rows) * row_bytes;
                        let bound = (u64::from(final_rows) * row_bytes).div_ceil(4) * 4;
                        let padded_buf = (u64::from(m) * row_bytes).div_ceil(4) * 4;
                        assert_eq!(offset % 4, 0, "offset not u32-aligned (k={k}, elem={elem})");
                        assert!(
                            offset + bound <= padded_buf,
                            "final tile rounded binding overruns padded buffer (k={k}, elem={elem})",
                        );
                    }
                }
            }
        }

        // Same head, same binding: f16 packs at least as many rows per tile as f32.
        let mb = 1u64 << 20;
        assert!(gemv_tile_rows(m, 100, mb, align, 2) >= gemv_tile_rows(m, 100, mb, align, 4));
    }

    /// `encode_copy` treats all three args as f32-element COUNTS: source float
    /// offset `S`, destination float offset `D`, length `L` must move
    /// `src[S..S+L]` into `dst[D..D+L]` — i.e. each count is scaled to bytes
    /// internally. The NON-ZERO offsets are the point: a regression that
    /// byte-counts (or fails to scale) an offset lands the copy at the wrong row,
    /// and this catches it hermetically on a GPU-less runner via lavapipe — the
    /// hot decode/prefill append paths all route their nonzero offsets through
    /// this same helper, so this is the value-level guard they otherwise lacked.
    #[test]
    fn encode_copy_scales_float_offsets_to_bytes() {
        let Some(ctx) = gpu_ctx_or_skip() else {
            return;
        };
        let src: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let src_buf = ctx.upload_f32(&src, "encode_copy_src");
        // create_storage_rw is zero-initialized.
        let dst_buf = ctx.create_storage_rw((16 * 4) as u64, "encode_copy_dst");

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Copy 4 floats from src[2..6] to dst[5..9] using FLOAT offsets.
        super::GpuLfm2Model::encode_copy(&mut enc, &src_buf, 2, &dst_buf, 5, 4);
        ctx.queue.submit(Some(enc.finish()));

        let got = ctx.download_f32(&dst_buf, 16);
        let mut want = vec![0.0f32; 16];
        want[5..9].copy_from_slice(&src[2..6]); // [2.0, 3.0, 4.0, 5.0]
        assert_eq!(
            got, want,
            "encode_copy must scale float offsets/length to bytes \
             (src_off=2, dst_off=5, len=4 → dst[5..9] == src[2..6])"
        );
    }
}

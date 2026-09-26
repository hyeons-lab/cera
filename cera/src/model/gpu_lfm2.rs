// GPU-accelerated LFM2 forward pass using wgpu compute shaders.
//
// All weights are dequantized to f32 at load time and uploaded to GPU buffers.
// The full forward pass runs in a single CommandEncoder per token — only the
// logits vector is read back to CPU.
//
// # Compute passes are a perf lever, and a scarce one
//
// Batch as many dispatches into one compute pass as correctness allows. The same
// dispatches measured **2.65x** more expensive split across N passes than
// batched into one (M1 Max), and a pass boundary costs GPU time — a pipeline
// drain — not just CPU encode. Decode issued 58 passes/token before the conv
// block's three were merged into one; that alone was +17% decode.
//
// Two things force a boundary, and only two:
//
// - **An `encode_copy`.** A buffer-to-buffer copy is an *encoder* operation and
//   cannot be issued inside a pass, so every copy ends one and starts another.
//   Most of these exist because a kernel is in-place (`rmsnorm` normalizes its
//   buffer, so the caller stages a scratch copy first); an out-of-place variant
//   removes the copy and the boundary with it.
// - **A readback.**
//
// Dependencies do *not*: WebGPU orders dispatches within a compute pass and
// makes each one's writes visible to the next, which is what the `ffn` and
// `conv` blocks rely on to run their whole dependent chain in one pass.
//
// `io_stats::passes` counts them and `gpu_lfm2_decode_passes.rs` holds decode to
// a budget, because this regresses invisibly — the output is identical either
// way.
//
// ## Profiling
//
// GPU timestamps are attached **per pass**, so production's merged passes
// (`layer_conv`, `layer_attn`, `tail`) report as single spans. Profile runs
// (`CERA_GPU_PROFILE=1`, gated on `GpuContext::profiling`) split each merged
// block — `conv_mixer`/`conv_ffn`, `attn_core`/`attn_ffn`,
// `tail_norm`/`tail_lm_head`/`tail_sample` — for per-stage attribution.
// Split them only while the profiler is enabled, never unconditionally:
// a pass boundary is not free. (Mid-pass `write_timestamp` would be the
// finer tool, but on the Adreno 830 it corrupts even the pass-boundary
// timestamps, so pass splitting is the attribution mechanism.)

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use half::f16;

use crate::CeraError;
use crate::backend::cpu::RopeType;
use crate::backend::wgpu::{DevicePollExt, GpuContext, GpuTensor, KvShiftParams, shaders};
use crate::gguf::GgufFile;
use crate::kv_cache::{InferenceState, KvCompression, KvPrefixCache, LayerSnapshot, StateSnapshot};
use crate::lora::{LoraAdapterWeights, LoraTarget};
use crate::model::gpu_turboquant::{TqGpuCache, TqMode, describe_kv_mode};
use crate::model::gpu_weight_source::{
    GpuWeightSource, MOE_MAX_EXPERT_USED, MOE_MAX_EXPERTS, stacked_expert_layout,
};
use crate::model::transformer::WeightRef;
use crate::model::weights::MmapWeight;
use crate::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use crate::tensor::DType;

/// Maximum N for a single batched-prefill dispatch. Mirrors the Metal
/// backend's `MAX_PREFILL_TOKENS = 2048`. Prompts longer than this are
/// chunked at the host side; each chunk shares the same prefill batch
/// scratch, so the worst-case scratch footprint is bounded.
const MAX_PREFILL_TOKENS: usize = 2048;

/// Maximum token batch size for full-vocabulary all-logits speculative verification.
const MAX_ALL_LOGITS_TOKENS: usize = 64;

// Tile geometry for the register-tiled matmul pipeline. The shader
// receives these via preprocessor #defines below; keeping a single
// source of truth here means dispatch geometry can never drift out of
// sync with the kernel.
pub(crate) const MUL_MAT_TILE_WG_M: u32 = 16;

/// Rows emitted per workgroup by `gemv_f32` / `gemv_f32_accum` — MUST match the
/// `NR` constant in `gemv_f32.wgsl`. Used to size the LoRA dispatch grids.
const GEMV_F32_ROWS_PER_WG: u32 = 8;
pub(crate) const MUL_MAT_TILE_WG_N: u32 = 16;
// Each thread computes a 4×4 register tile held in four named `vec4<f32>`s;
// `mul_mat_reg_tile.wgsl` hand-unrolls for exactly that shape, so TILE_M and
// TILE_N are not free parameters — see the accumulator note in the shader.
// 256 threads per workgroup covering a 64×64 output tile.
//
// shmem is `TILE_K·(TILE_ROWS+4) + TILE_K·(TILE_COLS+4)` f32 = (16·68)·2·4 =
// 8704 B ≈ 8.5 KiB, inside the 16 KiB `max_compute_workgroup_storage_size` that
// WebGPU guarantees on every adapter — so this runs on a spec-minimum device
// (notably a browser via cera-wasm), not only where the adapter reports more.
// The `const _` below enforces it.
//
// TILE_K=16 over 32 is a deliberate, measured platform trade, end-to-end prefill
// p50, interleaved: on a Pixel 9 Pro XL (Mali-G715) it is +8-10% at p=128/512/
// 1024 (p=2048 not measured there); on an M1 Max it is +4-8% at p=512/1024/2048
// and **-15% at p=128**, where two column tiles is too little work to hide the
// doubled barrier count. Taken because mobile is the constrained target, the
// absolute latency trade favours it (p=128 costs ~22 ms, p=512 saves ~31 ms),
// and 8.5 KiB is what clears the WebGPU floor above. The p=128 regression was
// accepted, not missed.
//
// One Q4_0-specific quirk of TILE_K=16, noted so it is not rediscovered as a
// bug: its loader stages 8 consecutive k per thread, so a 64×16 src0 tile is
// 1024 elements against 256 threads × 8 = 2048, and threads 128..255 idle
// through staging (at TILE_K=32 all 256 participated). Measured cost: none —
// Q4_0 runs 1353 GFLOP/s on an M1 Max, ahead of f32 (1269), Q4_K (1275) and
// Q6_K (1311). Staging is a small share of a k-tile's work.
//
// 16×16 also won the workgroup sweep on both parts (8×32, 32×8, 16×8, 8×16, 8×8
// were worse on each; 32×16 and 16×32 were tried on the M1 Max only). Re-run
// `cera/examples/wgpu_gemm_bench.rs` on BOTH before changing any of it — and
// confirm end-to-end, because the microbench only measures n=512, exactly the
// shape that made TILE_K=16 look like a free win on Apple too.
pub(crate) const MUL_MAT_TILE_M: u32 = 4;
pub(crate) const MUL_MAT_TILE_N: u32 = 4;
pub(crate) const MUL_MAT_TILE_K: u32 = 16;

// The two invariants `mul_mat_reg_tile.wgsl` and its Q4_0 loader depend on, as
// compile-time checks rather than comments. Violating either produces silently
// wrong numbers, not a shader compile error: a non-4 thread tile makes each
// thread compute a fraction of the tile the host dispatched for, and a TILE_K
// that is not a multiple of 8 lets the Q4_0 loader's 8-element run straddle a
// block boundary and write past the staged tile.
const _: () = assert!(
    MUL_MAT_TILE_M == 4 && MUL_MAT_TILE_N == 4,
    "mul_mat_reg_tile.wgsl hand-unrolls a 4x4 thread tile; re-unroll it before \
     changing MUL_MAT_TILE_M/N"
);
const _: () = assert!(
    MUL_MAT_TILE_K.is_multiple_of(8),
    "the Q4_0 shmem loader stages 8 consecutive k per thread and indexes within \
     one 32-element block; TILE_K must be a multiple of 8"
);
// 16384 B is WebGPU's guaranteed `max_compute_workgroup_storage_size`. Staying
// inside it is what lets this pipeline build on a spec-minimum adapter instead
// of only where `adapter.limits()` reports more; `GpuContext` has no fallback
// path, so exceeding it is a hard failure at load, not a slow path.
//
// This has to be a compile-time check because nothing else catches it:
// **native wgpu 24 does not validate this limit**. Measured directly — a device
// created with `wgpu::Limits::default()` (max 16384) accepted compute pipelines
// declaring 17408 B and even 32768 B of workgroup storage without a validation
// error. Browsers (Dawn) do enforce it, so an over-budget kernel is invisible on
// every desktop and CI run and only fails once it reaches WebGPU. A runtime test
// on native would be vacuous; this assert is not.
const _: () = assert!(
    (MUL_MAT_TILE_K * (MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M + 4)
        + MUL_MAT_TILE_K * (MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N + 4))
        * 4
        <= 16384,
    "reg-tile shmem must stay within WebGPU's guaranteed 16 KiB workgroup-storage \
     limit so the pipeline builds on a spec-minimum adapter"
);
// The other guaranteed limit this geometry sits against, and for the same
// reason: WebGPU promises only 256 for `max_compute_invocations_per_workgroup`
// (and 256 for `max_compute_workgroup_size_x`), which 16×16 hits exactly.
// `GpuContext::new` requests `adapter.limits()` — 1024 on an M1 Max — so a wider
// workgroup would build and run on every desktop and CI adapter and fail only in
// a browser. Note the sweep candidates named above: 32×16 and 16×32 are 512
// threads and would break a spec-minimum device.
const _: () = assert!(
    MUL_MAT_TILE_WG_M * MUL_MAT_TILE_WG_N <= 256,
    "reg-tile workgroup must stay within WebGPU's guaranteed 256 invocations per \
     workgroup so the pipeline builds on a spec-minimum adapter"
);

/// Build a `mul_mat_reg_tile` pipeline for the requested src0 dtype.
///
/// `src0_loader` selects the shmem dequant path, one of
/// `"INIT_SRC0_SHMEM_{Q4_0,Q8_0,Q4_K,Q5_K,Q6_K,FLOAT}"`, and `src0_inner` is the
/// element type the shader binds src0 as: `"u32"` for the quantized loaders
/// (they byte-address a packed `array<u32>` and decode) and `"f32"` for the
/// dense FLOAT loader (reads `array<f32>` weights directly, no dequant). The
/// rest of the kernel is dtype-agnostic: the loader decodes weights to f32 in
/// shared memory once per k-tile and the register-tiled inner loop reuses them
/// across all `TILE_COLS` token columns. That reuse is the entire reason this
/// kernel beats the batched-GEMV-shaped `gemm_*` kernels, which re-dequantize
/// per token.
fn build_mul_mat_pipeline(
    ctx: &GpuContext,
    label: &str,
    src0_loader: &str,
    src0_inner: &str,
) -> wgpu::ComputePipeline {
    let wg_m = format!("{MUL_MAT_TILE_WG_M}u");
    let wg_n = format!("{MUL_MAT_TILE_WG_N}u");
    let tile_m = format!("{MUL_MAT_TILE_M}u");
    let tile_n = format!("{MUL_MAT_TILE_N}u");
    let tile_k = format!("{MUL_MAT_TILE_K}u");
    ctx.create_pipeline_with_defines(
        shaders::MUL_MAT_REG_TILE,
        "main",
        label,
        &[
            ("SRC0_INNER_TYPE", src0_inner),
            (src0_loader, ""),
            ("WORKGROUP_SIZE_M", &wg_m),
            ("WORKGROUP_SIZE_N", &wg_n),
            ("TILE_M", &tile_m),
            ("TILE_N", &tile_n),
            ("TILE_K", &tile_k),
        ],
    )
}

/// Whether to build reg-tile GEMM pipelines from slangc SPIR-V fed straight to
/// the driver instead of the naga-compiled WGSL. Default ON wherever the device
/// accepts SPIR-V passthrough (Vulkan only; Metal/DX12/WebGPU fall back to naga,
/// see `GpuContext::supports_spirv_passthrough`). The slang kernels are
/// bit-identical to naga and avoid naga-30's
/// PowerVR codegen regression (~1.35x Q4_0 / ~1.53x Q8_0 prefill); on GPUs without
/// that regression they are still correct, just possibly perf-neutral. Only the
/// ported reg-tile loaders (Q4_0, Q8_0, and the K-quants Q4_K/Q5_K/Q6_K) are
/// affected; the dense f16/f32 loader stays on naga.
///
/// `CERA_WGPU_SPIRV_PASSTHROUGH=0` forces the naga WGSL path (escape hatch for a
/// driver that misbehaves on the raw SPIR-V); `=1` is the explicit-on default.
fn use_spirv_passthrough(ctx: &GpuContext) -> bool {
    std::env::var("CERA_WGPU_SPIRV_PASSTHROUGH").as_deref() != Ok("0")
        && ctx.supports_spirv_passthrough()
}

/// Whether eligible Q4_0 weights upload in the resident stream layout —
/// the pre-transposed (q, d) repack (see [`repack_q4_0_stream`]) — instead
/// of raw GGUF blocks. Same bytes, transposed, so the resident copy costs
/// nothing over raw; both phases read it directly (decode via
/// `gemv_q4_0_stream`, prefill via `gemm_stream_q4_0`), which deletes the
/// old permanent twin (a full second copy that OOM'd phones past 512 MB of
/// Q4_0) and the rotating twin (a per-GEMM on-GPU repack into shared
/// scratch, ~15% of prefill). Needs passthrough (the (q, d) kernels are
/// SPIR-V-only, Vulkan-only); everywhere else weights stay raw and ride the
/// pre-existing kernels. `CERA_WGPU_STREAM_GEMM=0` forces the raw layout
/// (escape hatch / A-B).
fn use_stream_layout(ctx: &GpuContext) -> bool {
    // Deliberately NOT gated on `ctx.shader_f16`: wgpu hides SHADER_F16 on
    // drivers that fully support Float16 (Adreno 830 reports f16=false here
    // while llama's fp16 OpenCL kernels run fine on the same silicon), and a
    // raw SPIR-V module doesn't need wgpu's blessing anyway. Float16
    // arithmetic is ubiquitous on the Vulkan GPUs this can reach (Adreno,
    // Mali, PowerVR, lavapipe); a driver without it fails loudly at pipeline
    // creation, and `CERA_WGPU_STREAM_GEMM=0` is the escape hatch.
    std::env::var("CERA_WGPU_STREAM_GEMM").as_deref() != Ok("0") && use_spirv_passthrough(ctx)
}

/// Whether one weight qualifies for the resident stream layout: Q4_0 with
/// `k % 32 == 0` (the repack's precondition). Ineligible weights upload
/// raw and ride the raw kernels in both phases.
fn stream_layout_eligible(dtype: DType, k: usize) -> bool {
    dtype == DType::Q4_0 && k.is_multiple_of(32)
}

/// Whether the k-slice-64 streaming GEMM twin is enabled: on by default
/// (+28-33% over the k-slice-32 kernel on Adreno 830, bit-exact),
/// `CERA_WGPU_GEMM_K64=0` forces the k-slice-32 kernel everywhere.
/// Read once per process so kernel selection per prefill call is static
/// (the stream-GEMM bind-group cache pairs slots with pipelines).
fn use_gemm_k64() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CERA_WGPU_GEMM_K64").as_deref() != Ok("0"))
}

/// Repack Q4_0 GGUF bytes into the streaming-GEMM layout: feature-major
/// nibble ushorts and split scales, both packed two per u32 (see
/// `gemm_stream_q4_0.slang`). Same total bytes as the GGUF, transposed.
/// Returns `(q_packed, d_packed)`.
///
/// Layout: `q[(k/8)*m]` with `q[k8*m + row] = ushort(w k8*8+0..3) |
/// ushort(w k8*8+4..7)<<16` (4 nibbles per ushort, low to high);
/// `d[ceil(k/64)*m]` with `d[k64*m + row] = half(scale 2*k64) |
/// half(scale 2*k64+1)<<16` (zero-padded lane when k/32 is odd).
fn repack_q4_0_stream(data: &[u8], m: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(k % 32, 0, "streaming repack needs k % 32 == 0, got k={k}");
    let blocks_per_row = k / 32;
    debug_assert_eq!(data.len(), m * blocks_per_row * 18);
    let nibble = |qs: &[u8], w: usize| -> u16 {
        if w < 16 {
            (qs[w] & 0xF) as u16
        } else {
            (qs[w - 16] >> 4) as u16
        }
    };
    let mut q = vec![0u32; m * k / 8];
    let mut d = vec![0u32; m * blocks_per_row.div_ceil(2)];
    for row in 0..m {
        for b in 0..blocks_per_row {
            let base = (row * blocks_per_row + b) * 18;
            let scale_bits = u16::from_le_bytes([data[base], data[base + 1]]) as u32;
            let qs = &data[base + 2..base + 18];
            // Scale pair: block 2p in the low half, 2p+1 in the high half.
            let pair = b / 2;
            if b % 2 == 0 {
                d[pair * m + row] = scale_bits;
            } else {
                d[pair * m + row] |= scale_bits << 16;
            }
            // Nibble ushorts: 8 per block, packed 2 per u32.
            for u in 0..8 {
                let mut v = 0u16;
                for t in 0..4 {
                    v |= nibble(qs, u * 4 + t) << (t * 4);
                }
                let qi = (b * 8 + u) / 2;
                if u % 2 == 0 {
                    q[qi * m + row] = v as u32;
                } else {
                    q[qi * m + row] |= (v as u32) << 16;
                }
            }
        }
    }
    (q, d)
}

/// Salt for synthetic *weight* vectors: every microbench and stream-kernel
/// test quantizes the same weights, so their inputs (and any input bug)
/// stay identical.
const SYNTH_SALT_WEIGHTS: u32 = 0x5EED;
/// Salt for synthetic *input* vectors (`x` / `B`): distinct from the
/// weights salt so inputs and weights are never accidentally correlated.
const SYNTH_SALT_INPUTS: u32 = 0xB0B;

/// Deterministic synthetic vector for bench/test fixtures: hash noise in
/// `[-1, 1)`, shared by the Q4_0 microbenches and the stream-kernel tests
/// so their inputs (and any input bug) stay identical.
fn synth_q4_0_vec(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
            ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

/// Row-major f32 to Q4_0 blocks (18 bytes per 32 elems), GGUF layout:
/// f16 scale + 16 bytes, byte i holding w[i] (low nibble) / w[i+16].
fn quantize_q4_0_synth(weights: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let nb = k / 32;
    let mut out = Vec::with_capacity(m * nb * 18);
    for row in 0..m {
        for b in 0..nb {
            let start = row * k + b * 32;
            let block = &weights[start..start + 32];
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = if amax == 0.0 { 1.0 } else { amax / 7.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            let id = 1.0 / d;
            for qi in 0..16 {
                let lo = ((block[qi] * id + 8.5) as i32).clamp(0, 15) as u8;
                let hi = ((block[16 + qi] * id + 8.5) as i32).clamp(0, 15) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

/// CPU reference: Q4_0 GEMV (`y = Wx`, `W` row-major `m x k`).
fn cpu_gemv_q4_0_ref(raw: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    let mut y = vec![0.0f32; m];
    let mut row_f32 = vec![0.0f32; k];
    for (r, y_r) in y.iter_mut().enumerate() {
        crate::quant::dequantize_q4_0_row(&raw[r * nb * 18..(r + 1) * nb * 18], &mut row_f32);
        *y_r = row_f32.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    y
}

/// CPU reference: Q4_0 GEMM (`Y = WB`, `W` row-major `m x k`, `B`
/// row-major `n x k`) into packed `[n][m]` (`y[c * m + r]`), matching the
/// stream kernels' output layout.
fn cpu_gemm_q4_0_ref(raw: &[u8], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    let mut y = vec![0.0f32; n * m];
    let mut row_f32 = vec![0.0f32; k];
    for r in 0..m {
        crate::quant::dequantize_q4_0_row(&raw[r * nb * 18..(r + 1) * nb * 18], &mut row_f32);
        for c in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += row_f32[t] * b[c * k + t];
            }
            y[c * m + r] = acc;
        }
    }
    y
}

/// Largest accepted `--spv` file. Shipped kernels are kilobytes; anything
/// past this is a mistyped path (a model file, a log), not a shader.
const SPV_MAX_BYTES: usize = 64 << 20;

/// SPIR-V magic word (`0x07230203`, little-endian on disk).
const SPV_MAGIC: u32 = 0x0723_0203;

/// Pure size leg of [`check_spv_bytes`]: word alignment + the cap. Split
/// out so the exact cap boundary is pinnable without a 64 MiB vec.
fn check_spv_size(len: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        len.is_multiple_of(4),
        "SPIR-V size {len} is not a multiple of 4"
    );
    anyhow::ensure!(
        len <= SPV_MAX_BYTES,
        "SPIR-V size {len} exceeds the {SPV_MAX_BYTES}-byte cap"
    );
    Ok(())
}

/// Validate raw `--spv` bytes before they reach the driver: word alignment,
/// the size cap, and the 5-word SPIR-V header shape (magic, a 1.x version,
/// a nonzero id bound, zero schema). Pure (no GPU needed) so a mistyped
/// path fails with a message naming the file instead of risking device
/// loss on attacker-influenced bytes (e.g. a shared CI artifact dir).
///
/// Header-shape only, not full module validity: magic + version + bound +
/// schema prove the file *starts like* a module, not that the driver will
/// accept it. (SPIR-V has no header-declared total length to cross-check
/// against the file size — the 5 words are all there is.)
fn check_spv_bytes(path: &str, bytes: &[u8]) -> anyhow::Result<Vec<u32>> {
    check_spv_size(bytes.len()).map_err(|e| anyhow::anyhow!("{path}: {e:#}"))?;
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    anyhow::ensure!(
        words.first() == Some(&SPV_MAGIC),
        "{path}: bad SPIR-V magic (expected 0x07230203)"
    );
    anyhow::ensure!(
        words.len() >= 5,
        "{path}: SPIR-V too short for the 5-word header ({} words)",
        words.len()
    );
    anyhow::ensure!(
        words[1] >> 16 == 1,
        "{path}: bad SPIR-V version 0x{:08x} (want 1.x)",
        words[1]
    );
    // words[2] is the generator id: any value (including 0) is legal.
    anyhow::ensure!(
        words[3] >= 1,
        "{path}: bad SPIR-V id bound {} (must be >= 1)",
        words[3]
    );
    anyhow::ensure!(
        words[4] == 0,
        "{path}: bad SPIR-V schema {} (reserved, must be 0)",
        words[4]
    );
    Ok(words)
}

/// Load one experimental `--spv` variant against a baseline pipeline's
/// bind-group layout.
fn load_spv_variant(
    ctx: &GpuContext,
    base_pipe: &wgpu::ComputePipeline,
    path: &str,
) -> anyhow::Result<(String, wgpu::ComputePipeline)> {
    use anyhow::Context;
    use std::io::Read;
    // Bounded read: `take` caps the allocation at one byte past the cap, so
    // a mistyped path (a multi-GB model file, a log) fails on the size check
    // below instead of OOMing first — with no metadata/read TOCTOU.
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("reading experimental SPIR-V {path}"))?
        .take(SPV_MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading experimental SPIR-V {path}"))?;
    let words = check_spv_bytes(path, &bytes)?;
    let layout = base_pipe.get_bind_group_layout(0);
    let pipe_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bench_variant_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    // SAFETY: `check_spv_bytes` above established word alignment, the size
    // cap, and the 5-word header shape (magic, 1.x version, nonzero id
    // bound, zero schema) — header-shape only, NOT full module validity.
    // The file is slangc output compiled from the same-shape source the
    // caller is A/Bing, and deeper spirv-val clean is on the caller
    // (bench-only path, never shipped): crafted bytes with a valid header
    // still reach the driver unverified, risking device loss at best.
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(path),
                spirv: Some(std::borrow::Cow::Owned(words)),
                entry_points: std::borrow::Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main"),
                    workgroup_size: (0, 0, 0),
                }]),
                dxil: None,
                hlsl: None,
                metallib: None,
                msl: None,
                glsl: None,
                wgsl: None,
            })
    };
    let pipe = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(path),
            layout: Some(&pipe_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                // Hard `false`, matching the baked baselines (a variant
                // measured with zero-init ON times a different kernel):
                // same-shape slangc output is safe uninitialized — every
                // kernel here writes scratch before reading it.
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
    let stem = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    Ok((stem, pipe))
}

/// Governor soak duration for [`soak_and_measure`]: tens of ms of
/// continuous load to ramp the Adreno governor off its idle clock.
const SOAK_MS: u64 = 500;

/// Clock soak + timed run for the microbenches: the Adreno governor idles
/// near 222 MHz and needs tens of ms of continuous load to ramp; a short
/// warmup times the ramp, not the kernel. Soak [`SOAK_MS`], then time
/// `iters` dispatches via `run`. Returns ms/iter and the current GPU clock.
///
/// The `run` closures block on `poll_wait`, so this must run on an executor
/// with no other tasks (today: `pollster::block_on` from a sync entry).
fn soak_and_measure(run: impl Fn(u32) -> std::time::Duration, iters: u32) -> (f64, String) {
    // `iters == 0` would print `inf` ms (f64 division) instead of failing;
    // both harness entries `ensure!(iters >= 1)`, so a zero here is a
    // caller bug that must fail fast in release too.
    assert!(iters >= 1, "soak_and_measure: iters must be >= 1");
    let soak_start = std::time::Instant::now();
    while soak_start.elapsed() < std::time::Duration::from_millis(SOAK_MS) {
        run(10);
    }
    let mhz = gpu_cur_freq_mhz()
        .map(|f| format!("{f}MHz"))
        .unwrap_or_else(|| "n/a".to_string());
    let dt = run(iters);
    (dt.as_secs_f64() * 1e3 / iters as f64, mhz)
}

/// Whether the LM head uploads a flat-planes Q6_K twin for decode GEMV
/// (see [`repack_q6_k_flat`]): passthrough-only (the flat kernel is
/// SPIR-V-only, Vulkan-only, and needs subgroups). `CERA_WGPU_FLAT_Q6K=0`
/// forces the interleaved layout everywhere (escape hatch / A-B).
fn use_flat_q6k(ctx: &GpuContext) -> bool {
    std::env::var("CERA_WGPU_FLAT_Q6K").as_deref() != Ok("0")
        && use_spirv_passthrough(ctx)
        && ctx.has_subgroup
}

/// Repack Q6_K GGUF bytes into flat planes for `gemv_q6_k_flat`: all rows'
/// low bytes, then all high bytes, scales, and super-scales. The 210-byte
/// interleaved stride misaligns every block (210 % 4 == 2), forcing the
/// raw kernel's funnel-shift loads; the planes are 4-aligned throughout,
/// so the flat kernel reads quants with single word loads and no shifts
/// (llama.cpp's `mul_mv_q6_K_f32_flat` layout, single-buffer form).
///
/// Layout for an (m, k) table with nb = k/256 blocks per row, same m*nb*210
/// bytes as the GGUF, transposed: `ql[m*nb*128]`, `qh[m*nb*64]`,
/// `scales[m*nb*16]` (signed bytes, preserved), `d[m*nb*2]` (f16 bits).
/// Panics unless k % 256 == 0 and the length matches.
fn repack_q6_k_flat(data: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "flat Q6_K repack needs k % 256 == 0, got k={k}");
    let nb = k / 256;
    assert_eq!(data.len(), m * nb * 210, "flat Q6_K repack length mismatch");
    let (ql_plane, qh_plane, s_plane) = (m * nb * 128, m * nb * 64, m * nb * 16);
    let mut out = vec![0u8; m * nb * 210];
    let (ql_out, rest) = out.split_at_mut(ql_plane);
    let (qh_out, rest) = rest.split_at_mut(qh_plane);
    let (s_out, d_out) = rest.split_at_mut(s_plane);
    for row in 0..m {
        for b in 0..nb {
            let base = (row * nb + b) * 210;
            let (ql, rest) = data[base..base + 210].split_at(128);
            let (qh, rest) = rest.split_at(64);
            let (s, d) = rest.split_at(16);
            debug_assert_eq!(d.len(), 2);
            let o = row * nb + b;
            ql_out[o * 128..(o + 1) * 128].copy_from_slice(ql);
            qh_out[o * 64..(o + 1) * 64].copy_from_slice(qh);
            s_out[o * 16..(o + 1) * 16].copy_from_slice(s);
            d_out[o * 2..(o + 1) * 2].copy_from_slice(d);
        }
    }
    out
}

/// Upload a table (`token_embd.weight`, `output.weight`) as f16, converting
/// row by row from the mmap so no host f32 copy of the table ever exists
/// (that copy cost vocab×hidden×4 B — 1 GB on a 128k-vocab model). Rows go
/// up in 256-row writes; the table stays unscaled (callers that need the
/// embedding multiplier apply it at gather time, input only).
fn upload_mmap_table_as_f16(ctx: &GpuContext, table: &MmapWeight, label: &str) -> wgpu::Buffer {
    const ROWS_PER_WRITE: usize = 256;
    let vocab = table.rows as u64;
    let hs = table.cols as u64;
    let buf = ctx.create_storage_rw(vocab * hs * 2, label);
    let mut row = vec![0.0f32; table.cols];
    let mut chunk: Vec<f16> = Vec::with_capacity(ROWS_PER_WRITE * table.cols);
    for start in (0..table.rows).step_by(ROWS_PER_WRITE) {
        chunk.clear();
        for r in start..(start + ROWS_PER_WRITE).min(table.rows) {
            table.dequantize_row(r, &mut row);
            chunk.extend(row.iter().map(|&x| f16::from_f32(x)));
        }
        ctx.queue
            .write_buffer(&buf, start as u64 * hs * 2, bytemuck::cast_slice(&chunk));
    }
    buf
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

/// KV cache slab bytes: rows × cols f16 halves packed 2-per-u32 (2 bytes
/// per element — half the old f32 slab). Byte-identical to a native f16 slab.
///
/// Widened to `u64` before multiplying: this backend also builds for `wasm32`,
/// where `usize` is 32 bits and a large `max_seq_len x kv_dim x 2` wraps,
/// silently sizing the cache to the wrapped remainder.
fn kv_slab_bytes(rows: usize, cols: usize) -> u64 {
    rows as u64 * cols as u64 * 2
}

/// Packing invariant: kv_dim (and head_dim) must be even so every u32 word
/// holds exactly one dim-pair and no kernel ever writes a partial word.
/// RoPE pairing already requires an even head_dim; this backstops it where
/// the packed cache is sized.
fn assert_kv_dim_packable(kv_dim: usize, head_dim: usize) {
    assert!(
        kv_dim.is_multiple_of(2) && head_dim.is_multiple_of(2),
        "packed-f16 KV cache needs even kv_dim/head_dim, got kv_dim={kv_dim} head_dim={head_dim}"
    );
}

fn packed_f16_binding(buffer: &wgpu::Buffer, len_floats: u64) -> wgpu::BindingResource<'_> {
    let bytes = len_floats
        .checked_mul(2)
        .expect("packed-f16 storage binding size overflow");
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(bytes.max(4)),
    })
}

/// Panic if an f32 storage binding of `len_floats` elements would exceed the
/// adapter's `max_storage_buffer_binding_size`. Shared by every live-range
/// attention binding so the byte math lives in one place; the multiply
/// saturates, so an overflow can only over-report and still trip the assert
/// rather than wrap to a small value that slips past it.
fn assert_packed_binding_fits(len_floats: u64, max_binding: u64, what: &str) {
    let bytes = len_floats.saturating_mul(2);
    assert!(
        bytes <= max_binding,
        "wgpu {what} binding is {bytes} bytes, exceeding adapter \
         max_storage_buffer_binding_size {max_binding}; context paging is required"
    );
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
        tile_rows > 0 && (u64::from(tile_rows) * row_bytes).is_multiple_of(offset_alignment),
        "GPU storage binding alignment {} cannot be satisfied for GEMV rows of {} bytes",
        offset_alignment,
        row_bytes
    );
    tile_rows
}

/// A weight matrix on GPU — tracks buffer + dtype + pre-allocated params for dispatch.
#[derive(Clone)]
struct GpuWeight {
    tensor: GpuTensor,
    /// Resident stream layout (see [`repack_q4_0_stream`]): feature-major
    /// nibbles and split scales, u32-packed. When `resident_stream` is set,
    /// these ARE the weight (no raw upload exists) and `tensor.buffer` is
    /// the q half; otherwise both are `None` and `tensor.buffer` is raw.
    stream_q: Option<wgpu::Buffer>,
    stream_d: Option<wgpu::Buffer>,
    /// Whether this weight uploaded in the resident stream layout instead
    /// of raw. Decode and prefill branch on this to pick the (q, d)
    /// kernels; set at upload from [`stream_layout_eligible`].
    resident_stream: bool,
    /// Whether `tensor.buffer` holds `repack_q6_k_flat` planes instead of
    /// GGUF-interleaved Q6_K blocks. Only the LM head's GEMV view sets
    /// this (see [`LmHead`]); decode picks the flat kernel off it.
    flat_q6k: bool,
    /// Pre-allocated params buffer with [m, k, row_base, 0] — eliminates per-dispatch allocation.
    params_buf: wgpu::Buffer,
    /// Pre-created bind group for this weight's primary GEMV dispatch.
    /// Created after all scratch buffers are allocated, to avoid per-token
    /// create_bind_group overhead (~16 µs each, 300×/token = 4.8 ms).
    cached_bg: Option<wgpu::BindGroup>,
}

/// How the logit projection is computed.
///
/// Both variants compute the same product. They differ in what the weight costs
/// to store and to read, and the LM head is the largest tensor in a small model
/// — for LFM2.5-230M it is the Q6_K `token_embd.weight`, 55 MB of a ~180 MB
/// model, read in full on every single token.
enum LmHead {
    /// The weight exactly as GGUF stores it, through the same quantized GEMV
    /// kernels the layer projections use.
    ///
    /// Preferred whenever the dtype has a kernel. Dequantizing to f16 instead
    /// costs 2.4x the bytes for Q6_K (2 B/elem vs 210 B/256), and this GEMV is
    /// bandwidth-bound, so those bytes are the runtime: 1439 -> 859 us on
    /// LFM2.5-230M/M1 Max, with 79 MB less VRAM held.
    ///
    /// Accuracy is a wash, not a win — worth stating because the reverse is easy
    /// to assume. The f16 copy does round the dequantized weights a second time,
    /// but measured against the CPU reference both paths sit at cosine 0.99977
    /// and differ from each other by only 1.7e-3 max: the gap to CPU is
    /// dominated by the 14 layers upstream, not by the LM head's weights.
    ///
    /// It also keeps the weight inside one storage binding more often. The f16
    /// copy here is exactly 128 MiB, which is precisely the common
    /// `max_storage_buffer_binding_size` — any larger vocab or hidden size tips
    /// it over and into `encode_gemv_f16_tiled`.
    ///
    /// `main` is the interleaved upload every consumer can read; `flat` is
    /// the optional `repack_q6_k_flat` twin (Q6_K LM head on passthrough
    /// only) that the GEMV projection reads instead. The batched-prefill
    /// GEMM keeps reading `main`: its reg-tile loader expects interleaved
    /// blocks. Boxed: two inline weights trip `large_enum_variant`.
    Quantized {
        main: GpuWeight,
        flat: Option<Box<GpuWeight>>,
    },
    /// A dequantized f16 copy, for dtypes with no quantized GEMV kernel (F32,
    /// F16, BF16 sources) or a weight too large for one binding even quantized.
    F16 {
        weight: wgpu::Buffer,
        /// `[m, k, 0, 0]` for the non-tiled dispatch.
        params: wgpu::Buffer,
    },
}

/// A layer's feed-forward weights on GPU: one dense SwiGLU, or a routed expert
/// set.
///
/// Mirrors `lfm2::FfnRefs` and the Metal backend's `MetalFfn`, and exists for
/// the reason given there: `lfm2moe` runs dense leading blocks and routes the
/// rest, so "which kind is this" is a per-layer question inside a single model.
/// A sum type makes exactly-one-of-two a fact the encoders match on, rather than
/// two sets of `Option` fields they would have to keep consistent by hand.
///
/// Both variants are boxed, where the Metal backend boxes only the routed one.
/// The difference is that a wgpu handle is wide: `GpuWeight` carries a shape
/// `Vec` and a cached bind group on top of its scalars, so `GpuDenseFfn`
/// measures 312 bytes and `GpuMoeFfn` 216, against tens each on Metal. Box only
/// one and the *other* stays inline, which is the comparison clippy's
/// `large_enum_variant` makes: 312 against a boxed 8 is 304, and 216 against 8
/// is 208, so either single-box choice still trips its 200-byte default. Boxing
/// both leaves 8 against 8. Note 208 clears 200 by only eight bytes, so shrinking
/// `GpuMoeFfn` could make boxing `Dense` alone legal again.
enum GpuFfn {
    Dense(Box<GpuDenseFfn>),
    Moe(Box<GpuMoeFfn>),
}

struct GpuDenseFfn {
    gate: GpuWeight,
    up: GpuWeight,
    down: GpuWeight,
}

/// Borrowed inputs to [`GpuLfm2Model::encode_attn_pre_into`]: the decode attn
/// pre-chain (norm + QKV + bias + QK-norm + RoPE) runs identically in the
/// merged `layer_attn` pass and the TurboQuant `attn_pre` split, so it takes
/// one struct instead of fifteen parameters. Per-head-norm, bias, and rope
/// groups come from `lw`; the QKV LoRA tuples are borrowed from arm locals.
struct AttnPreDecode<'a> {
    lw: &'a GpuLayerWeights,
    norm_bg: &'a wgpu::BindGroup,
    q_w: &'a GpuWeight,
    q_bg: &'a wgpu::BindGroup,
    k_w: &'a GpuWeight,
    k_bg: &'a wgpu::BindGroup,
    v_w: &'a GpuWeight,
    v_bg: &'a wgpu::BindGroup,
    q_lora: &'a Option<(&'a WgpuLoraTarget, (wgpu::BindGroup, wgpu::BindGroup))>,
    k_lora: &'a Option<(&'a WgpuLoraTarget, (wgpu::BindGroup, wgpu::BindGroup))>,
    v_lora: &'a Option<(&'a WgpuLoraTarget, (wgpu::BindGroup, wgpu::BindGroup))>,
    q_dim: u32,
    kv_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    max_pairs: u32,
}

/// One projection of a routed FFN, with every expert's slice in one buffer.
///
/// Not a [`GpuWeight`]: that type carries a params buffer and a cached bind
/// group for the dense GEMV dispatch, and neither survives the trip here. The
/// expert kernel takes its shape through its own params layout and picks the
/// slice on the device, so what it needs from the host is the base buffer and
/// the stride.
struct GpuMoeWeight {
    /// The stacked tensor, `[n_expert][m][k]` Q4_0, bound whole. The expert
    /// slice offset is applied in-shader from `sel_expert`.
    buffer: wgpu::Buffer,
    /// Rows and inner dimension of a *single* expert's slice.
    m: u32,
    k: u32,
    /// Byte distance between consecutive experts' slices, derived and checked by
    /// `gpu_weight_source::stacked_expert_layout`. That check catches a
    /// transcription error in the formula, not a file that is unevenly stacked:
    /// see the function for why those are not the same thing.
    expert_stride: u32,
}

/// One routed feed-forward block's weights.
///
/// The three expert projections stay *stacked*, exactly as on Metal: the CPU
/// path splits the rank-3 GGUF tensor into `n_expert` separate 2-D refs and
/// picks one after routing, which it can do because routing has already happened
/// on the same core. On GPU the selection lives in a device buffer, so the slice
/// has to be chosen inside the kernel, and a stride is the only form that
/// survives the trip.
struct GpuMoeFfn {
    /// Router projection (`ffn_gate_inp.weight`), f32, `[n_expert][hidden]`
    /// row-major. Bound as the right-hand side of the shared `gemm_f32_nt`
    /// rather than through a GEMV, so it is a plain buffer: its shape is
    /// `(n_expert, hidden)`, both of which are validated against the scratch and
    /// the config at load.
    router: wgpu::Buffer,
    /// Per-expert selection bias (`exp_probs_b.bias`), `n_expert` f32.
    bias: wgpu::Buffer,
    gate: GpuMoeWeight,
    up: GpuMoeWeight,
    down: GpuMoeWeight,
    /// The model's single routed-FFN scratch, shared by every routed layer.
    scratch: Arc<MoeScratch>,
}

/// GPU buffer handles for one layer's weights.
/// Q4_0/Q8_0 weights are uploaded quantized; f32 norms uploaded as-is.
struct GpuLayerWeights {
    attn_norm: wgpu::Buffer,
    ffn_norm: wgpu::Buffer,
    ffn: GpuFfn,
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

    // Cached bind groups for zero-allocation decode loop
    attn_norm_bg: Option<wgpu::BindGroup>,
    ffn_norm_bg: Option<wgpu::BindGroup>,
    rope_bg: Option<wgpu::BindGroup>,
    conv_fused_bg: Option<wgpu::BindGroup>,
    conv_add_bg: Option<wgpu::BindGroup>,
    attn_out_add_bg: Option<wgpu::BindGroup>,
    silu_bg: Option<wgpu::BindGroup>,
    ffn_swiglu_bg: Option<wgpu::BindGroup>,
    attn_bg: Option<wgpu::BindGroup>,
    /// `kv_append` groups (src k/v scratch -> cache slab at the per-token
    /// offset in `kv_append_params`). `None` on conv layers and under
    /// hs_scratch (built inline there, like `attn_bg`).
    k_append_bg: Option<wgpu::BindGroup>,
    v_append_bg: Option<wgpu::BindGroup>,
    qn_bg: Option<wgpu::BindGroup>,
    kn_bg: Option<wgpu::BindGroup>,
    qb_bg: Option<wgpu::BindGroup>,
    kb_bg: Option<wgpu::BindGroup>,
    vb_bg: Option<wgpu::BindGroup>,
    ffn_add_bg: Option<wgpu::BindGroup>,
}

/// Device-side working set for the routed FFN, allocated once for the model and
/// shared by every routed layer.
///
/// Sized for the largest batch the prefill path hands the kernels, which decode
/// then reuses as the `n = 1` case. Everything indexed *by entry* is
/// `n_tokens * n_expert_used` rows, not `n_tokens`.
///
/// Held by `Arc` from every routed layer rather than as an `Option` on the
/// model, for the reason the Metal backend gives: it makes "a routed layer
/// always has its scratch" a fact the type carries, so the encoder has no absent
/// case to either panic on or silently skip the FFN for.
struct MoeScratch {
    // The four dimensions the buffers below were sized from, and the single
    // source for them everywhere else: `upload_moe` validates each routed
    // layer's weights against these, and `moe_ffn_steps` reads them from here to
    // fill the kernel params and size its dispatch grids. The bounds they must
    // satisfy (`MOE_MAX_EXPERTS`, `MOE_MAX_EXPERT_USED`, and `MAX_WG` for the
    // grids) are enforced once, where the scratch is built, before anything is
    // sized from them.
    //
    // Not quite every dimension: `z` is also sized by the hidden size, which is
    // not recorded here because `upload_moe` ties the weights to it directly
    // (`down.m == hs`). A buffer sized from anything outside this list is
    // covered by neither, so add the dimension here when adding such a buffer.
    /// Experts per routed layer, sizing `logits`.
    n_expert: u32,
    /// Experts per token, the multiplier turning tokens into entries.
    n_expert_used: u32,
    /// Per-expert feed-forward width, the row stride of `gate` and `up`.
    expert_ff_len: u32,
    /// Entries the buffers below were sized for, i.e. the largest
    /// `n_tokens * n_expert_used` any dispatch may ask for. Kept because it also
    /// bounds a *dispatch* dimension: `moe_gemv_q4_0`'s grid is
    /// `(rows, n_entries)` and neither axis can be folded, so entries beyond
    /// [`crate::backend::wgpu::MAX_WG`] would be silently dropped rather than
    /// clamped. Checked at load, where it is a named error.
    max_entries: u32,
    /// `[n_tokens][n_expert]` router logits, pre-sigmoid.
    logits: wgpu::Buffer,
    /// `[n_entries]` chosen expert ids (u32).
    sel_expert: wgpu::Buffer,
    /// `[n_entries]` renormalized unbiased probabilities (f32).
    sel_weight: wgpu::Buffer,
    /// `[n_entries][expert_ff_len]` gate projection, overwritten in place with
    /// the SwiGLU product that the down projection then consumes.
    gate: wgpu::Buffer,
    /// `[n_entries][expert_ff_len]` up projection.
    up: wgpu::Buffer,
    /// `[n_entries][hidden]` per-entry expert outputs, before the weighted
    /// combine folds each token's entries together.
    z: wgpu::Buffer,
}

/// One dispatch of the routed FFN: which kernel, bound to what, over which grid.
///
/// The block is built as a list of these rather than encoded straight into a
/// command encoder, because a wgpu bind group cannot be created while a compute
/// pass is open. Building them all first is what lets the whole routed block
/// share one pass with the layer's rmsnorm instead of forcing a boundary per
/// kernel; see the module header on what a pass boundary costs.
///
/// The params buffers each step binds are not kept here, and do not have to
/// outlive the encode: wgpu refcounts the resources a `BindGroup` binds, and a
/// recorded dispatch holds the bind group, so the buffers survive to the submit
/// even where the caller drops the whole step list first (which the prefill arm
/// does, one layer at a time, into an encoder submitted after the loop). The
/// same pattern is already load-bearing in the dense batched path, whose params
/// buffers are scoped to the block that encodes them.
struct MoeStep<'a> {
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroups: (u32, u32, u32),
}

/// Upload one routed feed-forward block, stacking the expert projections into
/// one buffer each and deriving the per-expert byte stride.
///
/// Every shape here is checked against [`MoeScratch`]'s dimensions rather than
/// the counts this layer's own `MoeFfnRefs` carries; see `MoeScratch`'s fields
/// for why those are the authoritative copy.
///
/// The per-expert stride, and the validation behind it, come from
/// `gpu_weight_source::stacked_expert_layout`, shared with the Metal loader: a
/// wrong stride does not fault, it reads a neighbouring expert's weights and
/// still produces fluent text, so two backends deriving it separately is a
/// defect neither one's tests would name. What stays here is the part that is
/// this backend's: the storage-binding cap, and the upload itself.
fn upload_moe(
    ctx: &GpuContext,
    src: &dyn GpuWeightSource,
    scratch: Option<&Arc<MoeScratch>>,
    hidden_size: usize,
    layer: usize,
    moe: &crate::model::lfm2::MoeFfnRefs,
) -> Result<GpuMoeFfn> {
    use anyhow::Context;

    let scratch = scratch
        .with_context(|| {
            format!(
                "layer {layer} has routed expert weights but the model config carries no \
                 mixture-of-experts parameters to size their scratch with"
            )
        })?
        .clone();
    let n_expert = scratch.n_expert;
    let expert_ff_len = scratch.expert_ff_len;

    anyhow::ensure!(
        moe.exp_probs_b.len() == n_expert as usize,
        "layer {layer}: selection bias has {} entries for {n_expert} experts",
        moe.exp_probs_b.len(),
    );
    // `moe_route` emits ids in `0..n_expert`, and `moe_gemv_q4_0` turns each one
    // into `id * expert_stride` against a tensor stacked `refs.len()` deep, so a
    // layer with fewer expert tensors than the router has experts would address
    // past the stacked buffer.
    //
    // Both this and the bias-length check above are vacuous against today's only
    // `GpuWeightSource::moe_refs` implementation, which builds the ref lists as
    // `(0..n_expert)` and checks the bias itself. They are kept because this
    // function consumes a *trait*: it is the boundary where a second
    // implementation would arrive, and the cost is two integer comparisons at
    // load.
    anyhow::ensure!(
        moe.gate.len() == n_expert as usize
            && moe.up.len() == n_expert as usize
            && moe.down.len() == n_expert as usize,
        "layer {layer}: routed FFN has {n_expert} experts but {} gate / {} up / {} down expert \
         tensors; the routing kernel emits ids the GEMV would index past the stacked weights",
        moe.gate.len(),
        moe.up.len(),
        moe.down.len(),
    );
    anyhow::ensure!(
        moe.router.dtype == DType::F32,
        "layer {layer}: wgpu MoE routing needs an F32 router projection, found {:?}",
        moe.router.dtype,
    );

    // Stack one projection: every expert's bytes concatenated into a single
    // buffer, plus the byte stride from one expert to the next.
    //
    // Concatenated rather than sliced whole out of the mmap, even though
    // `stacked_expert_layout` proves the experts are contiguous there.
    // `weight_bytes` is a trait method whose contract is "the bytes of *this*
    // ref", and reading past one ref's extent would be reaching through it to
    // the mmap the only current implementation happens to be backed by. The cost
    // is one transient host copy per projection at load.
    let stack = |refs: &[WeightRef], what: &str| -> Result<GpuMoeWeight> {
        let layout = stacked_expert_layout(refs, layer, what, "wgpu")?;
        let stride = layout.expert_stride;
        let total = layout.total_bytes;
        // The kernel binds the whole stack as one `array<u32>`, so the adapter's
        // per-binding cap is a hard limit on the model rather than something the
        // dispatch can tile around. There is no equivalent of the f16 LM head's
        // `gemv_tile_rows` here: that one re-binds a row range per dispatch
        // because its kernel takes a `row_base`, and the expert GEMV has no such
        // parameter, it resolves its own slice from `sel_expert`. Reported as a
        // load error naming the adapter's limit; the alternative is a wgpu
        // validation failure mid-dispatch. Metal has no such cap, so this check
        // has no counterpart there and stays out of the shared helper.
        anyhow::ensure!(
            u64::from(total) <= ctx.max_storage_buffer_binding_size,
            "layer {layer}: {what} stacks {} experts into {total} bytes, over this adapter's \
             {} byte storage-binding limit; the expert GEMV binds the whole stack at once",
            refs.len(),
            ctx.max_storage_buffer_binding_size,
        );
        let bytes = refs
            .iter()
            .fold(Vec::with_capacity(total as usize), |mut acc: Vec<u8>, r| {
                acc.extend_from_slice(&src.weight_bytes(r));
                acc
            });
        Ok(GpuMoeWeight {
            buffer: ctx.upload_storage(&bytes, &format!("l{layer}.{what}")),
            m: u32::try_from(layout.rows)
                .with_context(|| format!("layer {layer}: {what} has too many rows"))?,
            k: u32::try_from(layout.inner)
                .with_context(|| format!("layer {layer}: {what} inner dim too large"))?,
            expert_stride: stride,
        })
    };

    let gate = stack(&moe.gate, "ffn_gate_exps")?;
    let up = stack(&moe.up, "ffn_up_exps")?;
    let down = stack(&moe.down, "ffn_down_exps")?;
    let hs = u32::try_from(hidden_size)
        .with_context(|| format!("layer {layer}: hidden size {hidden_size} too large"))?;
    anyhow::ensure!(
        gate.m == expert_ff_len && up.m == expert_ff_len && down.k == expert_ff_len,
        "layer {layer}: expert width {expert_ff_len} disagrees with the projection shapes \
         (gate.m={}, up.m={}, down.k={})",
        gate.m,
        up.m,
        down.k,
    );
    // Every shape the kernels index with, against the two numbers just tied to
    // the scratch's own. `down.m` is the sharp one: `MoeScratch::z` holds
    // `n_entries * hidden_size` floats and the down GEMV writes
    // `z[entry * down.m + row]`, so `down.m > hs` is an out-of-bounds device
    // write, and `down.m < hs` silently desyncs that row stride from the one
    // `moe_combine` reads it back with. `gate.k`/`up.k` index the activation
    // rows, and the router's shape bounds both the logits buffer and the GEMM
    // that fills it, so a file disagreeing on any of them is a load error rather
    // than a wrong answer.
    anyhow::ensure!(
        down.m == hs
            && gate.k == hs
            && up.k == hs
            && moe.router.k == hidden_size
            && moe.router.m == n_expert as usize,
        "layer {layer}: routed FFN shapes disagree with hidden size {hs} / expert count \
         {n_expert} (down.m={}, gate.k={}, up.k={}, router {}x{}); the wgpu expert kernels \
         index every one of these against those two numbers",
        down.m,
        gate.k,
        up.k,
        moe.router.m,
        moe.router.k,
    );
    // `moe_gemv_q4_0` takes its row index straight from `grp.x`, with no
    // `get_wid`-style folding available (the Y axis already carries the entry),
    // so a projection taller than the per-dimension workgroup cap would drop
    // every row above it and still return plausible output.
    anyhow::ensure!(
        gate.m.max(down.m) <= crate::backend::wgpu::MAX_WG,
        "layer {layer}: expert projections are {} rows, over the {} workgroups-per-dimension \
         cap the expert GEMV dispatches one row per workgroup against",
        gate.m.max(down.m),
        crate::backend::wgpu::MAX_WG,
    );

    // The router is the last routed binding without a named size error. It is
    // small on every real model (`n_expert x hidden` f32, ~256 KiB here), but it
    // is the one that scales with hidden size, and the alternative to checking
    // is a bare wgpu validation failure on the first forward. The bias needs no
    // check: `n_expert` is already bounded to `MOE_MAX_EXPERTS` floats.
    let router_bytes = (moe.router.m as u64)
        .checked_mul(moe.router.k as u64)
        .and_then(|n| n.checked_mul(4))
        .with_context(|| format!("layer {layer}: router projection size overflows u64"))?;
    anyhow::ensure!(
        router_bytes <= ctx.max_storage_buffer_binding_size,
        "layer {layer}: router projection needs {router_bytes} bytes, over this adapter's {} \
         byte storage-binding limit",
        ctx.max_storage_buffer_binding_size,
    );

    Ok(GpuMoeFfn {
        router: ctx.upload_f32(
            &src.dequantize_weight(&moe.router),
            &format!("l{layer}.ffn_gate_inp"),
        ),
        bias: ctx.upload_f32(&moe.exp_probs_b, &format!("l{layer}.exp_probs_b")),
        gate,
        up,
        down,
        scratch,
    })
}

/// A recorded prefill command: a compute dispatch or a buffer copy.
/// `encode_prefill_batched_locked` records each layer into a `Vec` of these
/// and `emit_prefill_cmds` turns them into passes — grouping dispatches
/// between copies into shared passes instead of one pass per dispatch. Each
/// pass costs ~130 µs of `CommandEncoder::finish` plus a GPU pipeline drain
/// (see `GpuIoStats::passes`), so the 257-pass prefill spent 33 ms in
/// `finish()` alone; merged it is ~24 passes.
///
/// Correctness: dispatches within a compute pass execute in order and wgpu
/// inserts per-dispatch barriers on conflicting usages (see
/// `State::flush_bindings`), so RAW hazards need no pass boundary — the
/// same guarantee the decode `ffn` span already relies on.
enum PrefillCmd<'a> {
    Dispatch {
        pipeline: &'a wgpu::ComputePipeline,
        bg: wgpu::BindGroup,
        grid: (u32, u32, u32),
        label: &'static str,
    },
    Copy {
        src: &'a wgpu::Buffer,
        src_off_floats: u64,
        dst: &'a wgpu::Buffer,
        dst_off_floats: u64,
        len_floats: u64,
    },
}

/// `(n, all_logits, cursor, slots)` — see `stream_gemm_bg_cache`.
type StreamGemmBgCache = (
    u32,
    bool,
    usize,
    Vec<Option<(wgpu::BindGroup, wgpu::BindGroup)>>,
);

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
    /// Q4_0 decode GEMV over the resident stream layout. `None` unless
    /// `use_stream_layout` (passthrough-only; SPIR-V has no WGSL twin).
    gemv_q4_0_stream: Option<wgpu::ComputePipeline>,
    gemv_q4_k: wgpu::ComputePipeline,
    gemv_q5_k: wgpu::ComputePipeline,
    gemv_q6_k: wgpu::ComputePipeline,
    /// Q6_K decode GEMV over the flat-planes LM-head twin. `None` unless
    /// `use_flat_q6k` (passthrough-only; SPIR-V has no WGSL twin).
    gemv_q6_k_flat: Option<wgpu::ComputePipeline>,
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
    ffn_swiglu_q4_0: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    /// Out-of-place twin (`rmsnorm_out` entry): `dst = norm(src)`. Decode-only;
    /// lets each layer's block and FFN share one compute pass by normalizing
    /// straight out of `hidden` instead of via a blit-split scratch copy.
    rmsnorm_out: wgpu::ComputePipeline,
    per_head_rmsnorm: wgpu::ComputePipeline,
    /// One decode K/V row into its cache slot. Decode-only; lets the attn
    /// block merge pre+post into one pass.
    kv_append: wgpu::ComputePipeline,
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

    mul_mat_reg_tile_q4_0: wgpu::ComputePipeline,
    /// Register-tiled Q4_0 GEMM over the resident stream layout: the prefill
    /// fallthrough for resident weights the streaming kernel declines (n <
    /// 32, strided B). `None` unless `use_stream_layout`.
    mul_mat_reg_tile_q4_0_stream: Option<wgpu::ComputePipeline>,
    mul_mat_reg_tile_q8_0: wgpu::ComputePipeline,
    mul_mat_reg_tile_q4_k: wgpu::ComputePipeline,
    mul_mat_reg_tile_q5_k: wgpu::ComputePipeline,
    mul_mat_reg_tile_q6_k: wgpu::ComputePipeline,
    /// Dense-f32 reg-tile GEMM. Serves every weight stored as f32 on the GPU:
    /// F16/BF16/F32 sources, plus quant types with no reg-tile loader (like
    /// Q4_1/Q2_K), which `upload_weight` dequantizes on the CPU and uploads as
    /// f32. Having it means a single unsupported-dtype tensor no longer drops the
    /// whole model onto the per-token prefill loop.
    mul_mat_reg_tile_f32: wgpu::ComputePipeline,
    /// Streaming fp16 Q4_0 prefill GEMM (llama-transcribed). `None` unless
    /// `use_stream_layout`; Q4_0 prefill falls back to the reg-tile
    /// kernels without it.
    gemm_stream_q4_0: Option<wgpu::ComputePipeline>,
    /// K-slice-64 twin of `gemm_stream_q4_0`: same interface and grid, each
    /// k-slice covers 64 k. Dispatched when k % 64 == 0 (see `use_gemm_k64`).
    /// Same `None`-unless-streaming convention as `gemm_stream_q4_0`.
    gemm_stream_q4_0_k64: Option<wgpu::ComputePipeline>,
    /// Transpose + f32->f16 cast feeding the streaming GEMM's B16 scratch.
    /// Same `None`-unless-streaming convention as `gemm_stream_q4_0`.
    transpose_cast_f16: Option<wgpu::ComputePipeline>,
    attention_prefill: wgpu::ComputePipeline,
    // ── Routed mixture-of-experts (`lfm2moe`) ─────────────────────────────
    // Built for every model, dense or routed: pipeline creation is a shader
    // compile, so making it conditional would trade a fixed load-time cost for a
    // branch on every construction path. Dense models never dispatch them.
    /// `lfm2moe` routing: sigmoid + biased top-k over the router logits.
    moe_route: wgpu::ComputePipeline,
    /// `lfm2moe` expert-indexed Q4_0 GEMV; the expert id comes from a device
    /// buffer, not the host.
    moe_gemv_q4_0: wgpu::ComputePipeline,
    /// `lfm2moe` weighted sum of a token's expert outputs.
    moe_combine: wgpu::ComputePipeline,
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
    layers: Vec<[Option<WgpuLoraTarget>; crate::lora::LORA_TARGET_COUNT]>,
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
        // Unreachable through `Session`, which refuses an adapter carrying
        // routed-FFN deltas (`supports_moe_lora` is false here). Asserted anyway
        // as a belt-and-braces check for a caller that bypasses `Session`: this
        // backend now *has* routed layers, so a delta reaching the upload would
        // be uploaded and then never applied rather than being impossible.
        // Compiles out in release; the gate itself is in `session.rs`.
        debug_assert!(
            !w.has_moe_deltas(),
            "adapter carries routed-FFN deltas, which this backend has no hooks for; \
             Session::attach_lora_adapters is meant to have rejected it"
        );
        for layer in 0..w.n_layers() {
            let mut targets: [Option<WgpuLoraTarget>; crate::lora::LORA_TARGET_COUNT] =
                Default::default();
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
    /// Per attention layer: (key_cache, value_cache) packed-f16 buffers.
    ///
    /// Allocated **lazily**, on the first `active_kv` (see
    /// [`GpuLfm2Model::f16_kv`]). A model is loaded before the session that
    /// configures its KV compression exists, so allocating the full
    /// `max_seq_len × kv_dim` slabs up front and freeing them once a
    /// TurboQuant session arrives would create exactly the transient memory peak
    /// compression exists to avoid. Under TurboQuant this `OnceLock` is never
    /// initialized and the packed buffers in `GpuLfm2Model::tq` hold the cache
    /// instead.
    kv_caches: OnceLock<Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>>,
    /// Per conv layer: rolling buffer.
    conv_buffers: Vec<Option<wgpu::Buffer>>,
    seq_len: AtomicUsize,
    max_seq_len: usize,
    /// Token embedding table (`token_embd.weight`) as an mmap handle into
    /// the GGUF — no copy. Input-embedding lookup dequantizes rows on the
    /// fly (`dequantize_row`, one row per token); this replaced a
    /// pre-dequantized f32 host copy that cost vocab×hidden×4 B (512 MB on
    /// a 2.6B) for the lifetime of the model. The retained mapping is
    /// reclaimable file pages, and only the table's own range stays hot.
    embedding: MmapWeight,
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
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// GPU-accelerated LFM2 model.
///
/// KV and convolution buffers persist on the model between calls. Only one
/// live Session may own this state; a second Session returns `CeraError::Busy`.
/// The per-call `infer_lock` still protects scratch and raw-call bookkeeping.
/// Use independently loaded models for concurrent GPU conversations. Direct raw
/// methods remain caller-managed and must not interfere with a live Session.
pub struct GpuLfm2Model {
    ctx: GpuContext,
    config: ModelConfig,
    pipelines: GpuPipelines,
    // GPU weight buffers
    /// The logit projection: `output.weight` when the model has untied
    /// embeddings, otherwise the tied `token_embd.weight`. See [`LmHead`] for
    /// why the two variants exist.
    ///
    /// Note this is the *projection* copy only. The input-embedding lookup
    /// reads rows from the mmap'd table (`gpu_state.embedding`) and applies
    /// the embedding multiplier at gather time — that split is what lets a
    /// tied-embedding Granite scale the input without also scaling the
    /// logits.
    lm_head: LmHead,
    output_norm: wgpu::Buffer,
    layers: Vec<GpuLayerWeights>,
    /// RoPE pair layout for this model (`Neox` LFM2/Qwen, `Norm` Llama family).
    rope_type: RopeType,
    /// Granite 3.x scalar multipliers (identity for every other arch). The
    /// embedding multiplier folds into gathered rows at lookup time; the
    /// residual/attention/logit multipliers are applied during the forward pass.
    scalars: ScalarMultipliers,
    /// Optional physical loop interval for looped architectures (e.g. Nanbeige).
    loop_norm_interval: Option<usize>,
    /// Whether the batched-prefill GPU path is enabled (LFM2 only today; the
    /// dense transformers prefill via the per-token decode loop).
    batched_prefill: bool,
    /// Latches once the "no batched GEMM for this dtype" warning has been emitted,
    /// so a long generation doesn't repeat it on every `forward_prefill`.
    batched_fallback_warned: AtomicBool,
    /// Latches once the "ignoring a routed-FFN adapter" error has been logged,
    /// so a long generation does not repeat it per token. See `resolve_lora`.
    moe_lora_dropped_warned: AtomicBool,
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
    /// 4-byte `MAP_READ` sink the argmax result is copied into inside the
    /// output projection's submission, so reading it back costs a map callback
    /// rather than a second GPU round trip. Owned rather than the shared
    /// `download_*` staging buffer: the copy is encoded well before the map, and
    /// a concurrent download would otherwise clobber it in between.
    argmax_readback_buf: wgpu::Buffer,
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
    // 7 u32, updated per token; must stay in sync with the params array in
    // `shaders/slang/rope.slang`'s wgsl branch.
    rope_params: wgpu::Buffer,
    attn_params: wgpu::Buffer, // [n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale, 0, 0] — updated per token
    kv_append_params: wgpu::Buffer, // [dst_off_words, n_floats, 0, 0] — updated per token
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
    /// Transposed-f16 activation scratch for the streaming Q4_0 GEMM:
    /// `[max_k × MAX_PREFILL_TOKENS_padded]` halfs, rewritten by
    /// `transpose_cast_f16` ahead of every streaming dispatch. `None` unless
    /// `use_stream_gemm`.
    stream_b16_buf: Option<wgpu::Buffer>,
    /// `[MAX_PREFILL_TOKENS × intermediate_size]` — FFN gate output;
    /// also reused as scratch for K projections and per-(layer,FFN)
    /// add-residual targets.
    prefill_gate_buf: wgpu::Buffer,
    /// `[MAX_PREFILL_TOKENS × intermediate_size]` — FFN up output;
    /// also reused as scratch for V projections.
    prefill_up_buf: wgpu::Buffer,
    /// `[MAX_SPEC_TOKENS × vocab_size]` - batched speculative verification logits buffer.
    prefill_all_logits_buf: wgpu::Buffer,
    // GPU state
    gpu_state: GpuState,
    /// Serializes individual raw calls and their shared scratch access.
    /// Session lifetime isolation is enforced separately by `session_gate`;
    /// this mutex alone cannot protect interleaved conversation histories.
    infer_lock: Mutex<()>,
    /// Reserves live KV/conv state for one Session until its destruction.
    session_gate: super::ModelSessionGate,
    /// Lazily-allocated scratch KV/conv for [`Self::hidden_states`] (see
    /// `HsScratch`). Built on first use via [`Self::hs_scratch`] so a
    /// generation-only load pays no extra KV VRAM. Selected over the generation
    /// caches by `use_hs_scratch`, which is only toggled while holding
    /// `infer_lock`, so `Relaxed` ordering suffices.
    hs_scratch: OnceLock<HsScratch>,
    use_hs_scratch: AtomicBool,
    /// GPU-resident TurboQuant KV cache, built by
    /// [`Model::configure_kv_compression`] when a session asks for it. `None` ⇒
    /// the packed-f16 KV path. Written once, under `infer_lock`.
    tq: OnceLock<TqGpuCache>,
    /// The compression mode this model has been configured for: `Some(mode)` for
    /// TurboQuant, `None` for packed-f16 KV. Distinct from `tq` because a request the
    /// backend can't serve (single-sided TurboQuant, unsupported `head_dim`)
    /// records f32 here while leaving `tq` empty. Set *after* the cache is built,
    /// so a failed (OOM) allocation leaves the model still reconfigurable; the
    /// whole of `configure_kv_compression` holds `infer_lock`, so the ordering is
    /// unobservable to other threads.
    kv_mode: OnceLock<Option<TqMode>>,
    /// Prefix-cache namespace tag for the mode this model resolved to
    /// (`KvCompression::cache_tag`). Empty until configured, which is the correct
    /// tag for the f32 default.
    kv_cache_tag: OnceLock<String>,
    /// Caller namespace plus loaded-byte identity when disk caching is compiled
    /// in, used to namespace prefix-cache disk files. Prefixed with `"wgpu:"` before
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
    prefill_params_pool: Mutex<(Vec<wgpu::Buffer>, usize)>,
    /// Cached bind groups for the streaming prefill GEMMs, in call order:
    /// `(n, all_logits, cursor, slots)` where each slot is the
    /// `(transpose_bg, gemm_bg)` pair for one `encode_gemm_stream_q4_0`
    /// call. A prefill creates ~184 of these fresh (~16 µs each to create,
    /// plus ~115 µs each of first-use validation inside
    /// `CommandEncoder::finish` — the 33 ms prefill `finish()` is almost
    /// entirely this). The bound buffers are stable across prefills —
    /// weights/scratch never move, and the pooled params buffers are handed
    /// out in deterministic call order after the per-prefill cursor reset —
    /// so only the *contents* need refreshing (which `next_prefill_params`
    /// still does on every call). Keyed by `n` because n < 32 routes to
    /// reg-tile and shifts the pool layout, and by `all_logits` because it
    /// swaps the lm-head output buffer; `start_pos` never appears in these
    /// bind groups. Single entry: chunked prefills reuse one `n` for every
    /// full chunk. Locked under `infer_lock`.
    stream_gemm_bg_cache: Mutex<StreamGemmBgCache>,
}

/// Where a decode step's initial hidden state comes from.
///
/// The two arms are the difference between a text token and an image patch.
/// Text has an id that indexes the embedding table; an image arrives from the
/// mmproj's projector as a hidden-size vector with no id behind it. Only the
/// first step of the forward pass differs, so this is a seed selector rather
/// than a second code path.
#[derive(Clone, Copy)]
enum HiddenSeed<'a> {
    /// Look the row up in the embedding table.
    Token(u32),
    /// Upload this hidden-size vector as-is.
    Embedding(&'a [f32]),
}

/// What the decode tail appends after the output projection.
///
/// Both greedy paths want the argmax dispatch in that encoder rather than one of
/// their own — a submit costs a GPU round trip regardless of how little it
/// carries. They differ on the readback: the blocking path stages it into the
/// model's `argmax_readback_buf` in the same submission, so reading it costs a
/// map instead of a second round trip, while the async path reads through
/// `begin_download`'s per-call buffer (which it can hold across an `.await`
/// without another caller clobbering it) and would gain nothing but a dead copy
/// from staging as well.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TailArgmax {
    /// Nothing — the caller wants the full logits buffer.
    None,
    /// Argmax dispatch only.
    Dispatch,
    /// Argmax dispatch plus the copy that stages its result for readback.
    DispatchAndStage,
}

/// Where the decode tail stops.
///
/// The audio path wants the hidden state rather than logits, so it stops after
/// the output norm and before the projection. **After** the norm, not before:
/// the CPU model's `run_layers` ends with `rmsnorm(hidden, output_norm_weight)`
/// and `forward_embedding` returns that, so the vector the depthformer is
/// calibrated against is the normed one. Stopping a step earlier gets a vector
/// that is off by the norm's per-element weight, which is a rotation and not
/// just a rescale, so nothing downstream reads as merely quieter.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeTail {
    /// Output norm, LM head, and whatever the [`TailArgmax`] asks for on top.
    /// Logits end up in `logits_buf`.
    Logits(TailArgmax),
    /// Output norm only, leaving the normed hidden state in `hidden_buf`.
    Hidden,
    /// Output norm only, leaving the normed hidden state in `hidden_buf` and returning
    /// the unsubmitted command encoder for single-pass download staging.
    HiddenUnsubmitted,
    /// Output norm and logits/argmax, returning the unsubmitted command encoder.
    LogitsUnsubmitted(TailArgmax),
}

impl GpuLfm2Model {
    /// Construct without a model identifier. Equivalent to
    /// `from_gguf_with_id(gguf, context_size, "")`. Warm prefix caching works;
    /// cold caching is disabled even when a directory is configured.
    pub fn from_gguf(gguf: GgufFile, context_size: usize) -> Result<Self> {
        Self::from_gguf_with_id(gguf, context_size, String::new())
    }

    /// Construct with an explicit model identifier (typically the GGUF
    /// path) used to namespace prefix-cache disk files. An empty ID disables
    /// cold caching. A nonempty ID is bound to the loaded GGUF bytes. The id is
    /// prefixed with `"wgpu:"` before being fed to `model_fingerprint`
    /// so different backends (cpu / metal / wgpu) sharing a
    /// `--cache-dir` don't collide on file names — see CPU's `"cpu:"`
    /// in PR #119 for the same pattern.
    pub fn from_gguf_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        let ctx = GpuContext::new()?;
        Self::from_gguf_with_ctx(gguf, context_size, model_id, ctx)
    }

    /// Construct a GPU model with an externally-built [`GpuContext`].
    /// The wasm/WebGPU entry point: callers build the context with
    /// `GpuContext::new_async().await` (browser init is async) and hand it in.
    /// Supports LFM2/LFM2-MoE (`lfm2`/`lfm2moe`) and dense transformers (`llama`, `qwen2`, `qwen3`, `granite`, `minicpm`, `minicpm5`, `nanbeige`, with classic Mistral served under `llama`).
    pub fn from_gguf_with_ctx(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
        ctx: GpuContext,
    ) -> Result<Self> {
        let arch = gguf.architecture().unwrap_or("").to_lowercase();
        match arch.as_str() {
            "llama" | "qwen2" | "qwen3" | "granite" | "minicpm" | "minicpm5" | "nanbeige"
            | "phi3" | "phi" => {
                // No CPU repacks: the GPU loader only resolves metadata from
                // this model (see `with_repack_if`).
                let cpu_model = super::llama::LlamaModel::from_gguf_with_id_no_repack(
                    gguf,
                    context_size,
                    model_id.clone(),
                )?;
                if let Some(sw) = cpu_model.sliding_window() {
                    tracing::warn!(
                        "Model specifies sliding window attention ({sw} tokens), which is not accelerated on WebGPU; full dense attention will be applied"
                    );
                }
                if cpu_model.has_projection_or_ffn_biases() {
                    tracing::warn!(
                        "Model specifies projection or FFN biases, which are not accelerated on WebGPU; biases will be omitted in GPU forward passes"
                    );
                }
                Self::from_weight_source_with_ctx(&cpu_model, context_size, model_id, ctx)
            }
            "lfm2" | "lfm2moe" => {
                // No CPU repacks: the GPU loader only resolves metadata from
                // this model (see `with_repack_if`).
                let cpu_model = super::lfm2::Lfm2Model::from_gguf_with_id_no_repack(
                    gguf,
                    context_size,
                    model_id.clone(),
                )?;
                Self::from_weight_source_with_ctx(&cpu_model, context_size, model_id, ctx)
            }
            other => anyhow::bail!("unsupported architecture for GPU: {other}"),
        }
    }

    /// Access the underlying GPU context (device, queue, adapter).
    pub fn ctx(&self) -> &GpuContext {
        &self.ctx
    }

    /// Construct a GPU model for a dense transformer (Qwen2/Qwen3/LLaMA/
    /// Mistral/Granite/MiniCPM): the `LlamaModel` family. Mirrors `from_gguf_with_id`
    /// but feeds the shared loader a `LlamaModel` weight source instead of
    /// `Lfm2Model`. The GPU forward path is arch-generic; per-arch behavior
    /// (NEOX/NORM rope, QK-norm, QKV bias, untied output, Granite/MiniCPM scalars) is
    /// driven by the `GpuWeightSource` accessors + `config`.
    pub fn from_llama_with_id(
        gguf: GgufFile,
        context_size: usize,
        model_id: String,
    ) -> Result<Self> {
        // No CPU repacks: the GPU loader only resolves metadata from this
        // model (see `with_repack_if`).
        let cpu_model = super::llama::LlamaModel::from_gguf_with_id_no_repack(
            gguf,
            context_size,
            model_id.clone(),
        )?;
        if let Some(sw) = cpu_model.sliding_window() {
            tracing::warn!(
                "Model specifies sliding window attention ({sw} tokens), which is not accelerated on WebGPU; full dense attention will be applied"
            );
        }
        if cpu_model.has_projection_or_ffn_biases() {
            tracing::warn!(
                "Model specifies projection or FFN biases, which are not accelerated on WebGPU; biases will be omitted in GPU forward passes"
            );
        }
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

    /// Construct a GPU model for a DSpark draft sidecar model.
    pub fn from_dspark_with_ctx(
        dspark: std::sync::Arc<crate::model::dspark::DSparkDraftModel>,
        context_size: usize,
        model_id: String,
        ctx: GpuContext,
    ) -> Result<Self> {
        let dspark_cfg = dspark.config.to_model_config(context_size);
        let dspark_src = crate::model::dspark::DSparkGpuWeightSource {
            config: dspark_cfg,
            dspark,
        };
        Self::from_weight_source_with_ctx(&dspark_src, context_size, model_id, ctx)
    }

    /// Like `from_weight_source` but with an externally-constructed
    /// [`GpuContext`]. This is the wasm entry point: the context is built
    /// asynchronously (`GpuContext::new_async().await`) before construction,
    /// since the rest of loading (weight upload + pipeline build) is sync GPU
    /// work that does no readback and runs fine on the wasm main thread.
    pub fn from_weight_source_with_ctx(
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
        let loop_norm_interval = src.loop_norm_interval();
        // The routed FFN's combine step adds its output into the residual stream
        // unscaled, matching what the dense path's `scaled_add_inplace` does when
        // `residual == 1.0`. No routed architecture also carries Granite's
        // sublayer multiplier today, so rather than thread the scale through
        // `moe_combine` for a combination that does not exist, refuse it: a
        // silent drop here scales every routed layer's contribution wrongly and
        // still produces fluent text.
        anyhow::ensure!(
            config.moe.is_none() || scalars.residual == 1.0,
            "mixture-of-experts with a residual multiplier ({}) is not supported on the wgpu \
             backend; the routed FFN combine adds into the residual unscaled",
            scalars.residual,
        );

        tracing::info!(
            "GPU model: {} layers, hs={hs}, is={is}, vocab={}",
            config.n_layers,
            config.vocab_size
        );

        // Resident stream layout (one decision for pipelines and upload
        // alike): eligible Q4_0 weights upload pre-transposed. Same bytes as
        // raw, so there is no budget and no per-model size gate.
        let stream_layout = use_stream_layout(&ctx);
        if stream_layout {
            tracing::info!(
                "streaming layout: Q4_0 weights upload pre-transposed; \
                 decode and prefill read the resident (q, d) directly"
            );
        }

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
            gemv_q4_0_fast: if ctx.supports_spirv_passthrough() && ctx.has_subgroup {
                // Vulkan: subgroup-reduction twin via passthrough (naga
                // cannot parse `enable subgroups`, so the WGSL twin below
                // carries the tree instead). Same bindings, same NR=8.
                ctx.gemv_q4_0_fast_passthrough()
            } else {
                ctx.create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast", "gemv_q4_0_fast")
            },
            gemv_q4_0_stream: if stream_layout {
                tracing::debug!("gemv_q4_0_stream: SPIR-V passthrough (slang)");
                Some(ctx.gemv_q4_0_stream_passthrough())
            } else {
                None
            },
            gemv_q4_k: ctx.create_pipeline(shaders::GEMV_Q4_K, "gemv_q4_k", "gemv_q4_k"),
            gemv_q5_k: ctx.create_pipeline(shaders::GEMV_Q5_K, "gemv_q5_k", "gemv_q5_k"),
            gemv_q6_k: if ctx.supports_spirv_passthrough() && ctx.has_subgroup {
                // Vulkan: 64-thread subgroup twin via passthrough. Same
                // bindings, NR=1 (WGSL fallback is NR=2 — the rows gate in
                // `gemv_pipeline_rows_label` mirrors this condition).
                ctx.gemv_q6_k_passthrough()
            } else {
                ctx.create_pipeline(shaders::GEMV_Q6_K, "gemv_q6_k", "gemv_q6_k")
            },
            gemv_q6_k_flat: if use_flat_q6k(&ctx) {
                tracing::debug!("gemv_q6_k_flat: SPIR-V passthrough (slang)");
                Some(ctx.gemv_q6_k_flat_passthrough())
            } else {
                None
            },
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
            ffn_swiglu_q4_0: ctx.create_pipeline(
                shaders::FFN_SWIGLU_Q4_0,
                "ffn_swiglu_q4_0",
                "ffn_swiglu_q4_0",
            ),
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm", "rmsnorm"),
            rmsnorm_out: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm_out", "rmsnorm_out"),
            kv_append: ctx.create_pipeline(shaders::KV_APPEND, "kv_append", "kv_append"),
            per_head_rmsnorm: ctx.create_pipeline(
                shaders::PER_HEAD_RMSNORM,
                "per_head_rmsnorm",
                "per_head_rmsnorm",
            ),
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

            mul_mat_reg_tile_q4_0: if use_spirv_passthrough(&ctx) {
                tracing::debug!("mul_mat_reg_tile_q4_0: SPIR-V passthrough (slang)");
                ctx.mul_mat_reg_tile_q4_0_passthrough()
            } else {
                build_mul_mat_pipeline(&ctx, "mul_mat_q4_0", "INIT_SRC0_SHMEM_Q4_0", "u32")
            },
            mul_mat_reg_tile_q4_0_stream: if stream_layout {
                tracing::debug!("mul_mat_reg_tile_q4_0_stream: SPIR-V passthrough (slang)");
                Some(ctx.mul_mat_reg_tile_q4_0_stream_passthrough())
            } else {
                None
            },
            gemm_stream_q4_0: if stream_layout {
                tracing::debug!("gemm_stream_q4_0: SPIR-V passthrough (slang)");
                Some(ctx.gemm_stream_q4_0_passthrough())
            } else {
                None
            },
            gemm_stream_q4_0_k64: if stream_layout {
                tracing::debug!("gemm_stream_q4_0_k64: SPIR-V passthrough (slang)");
                Some(ctx.gemm_stream_q4_0_k64_passthrough())
            } else {
                None
            },
            transpose_cast_f16: if stream_layout {
                tracing::debug!("transpose_cast_f16: SPIR-V passthrough (slang)");
                Some(ctx.transpose_cast_f16_passthrough())
            } else {
                None
            },
            // Every quantized weight goes through the register-tiled kernel (weight
            // reuse across the token tile), NOT the batched-GEMV-shaped gemm_* kernels:
            // those re-dequantize the weight once per token, so they buy submit count
            // and no compute, and measured *slower* than the per-token fallback they
            // were meant to replace.
            mul_mat_reg_tile_q8_0: if use_spirv_passthrough(&ctx) {
                tracing::debug!("mul_mat_reg_tile_q8_0: SPIR-V passthrough (slang)");
                ctx.mul_mat_reg_tile_q8_0_passthrough()
            } else {
                build_mul_mat_pipeline(&ctx, "mul_mat_q8_0", "INIT_SRC0_SHMEM_Q8_0", "u32")
            },
            mul_mat_reg_tile_q4_k: if use_spirv_passthrough(&ctx) {
                tracing::debug!("mul_mat_reg_tile_q4_k: SPIR-V passthrough (slang)");
                ctx.mul_mat_reg_tile_q4_k_passthrough()
            } else {
                build_mul_mat_pipeline(&ctx, "mul_mat_q4_k", "INIT_SRC0_SHMEM_Q4_K", "u32")
            },
            mul_mat_reg_tile_q5_k: if use_spirv_passthrough(&ctx) {
                tracing::debug!("mul_mat_reg_tile_q5_k: SPIR-V passthrough (slang)");
                ctx.mul_mat_reg_tile_q5_k_passthrough()
            } else {
                build_mul_mat_pipeline(&ctx, "mul_mat_q5_k", "INIT_SRC0_SHMEM_Q5_K", "u32")
            },
            mul_mat_reg_tile_q6_k: if use_spirv_passthrough(&ctx) {
                tracing::debug!("mul_mat_reg_tile_q6_k: SPIR-V passthrough (slang)");
                ctx.mul_mat_reg_tile_q6_k_passthrough()
            } else {
                build_mul_mat_pipeline(&ctx, "mul_mat_q6_k", "INIT_SRC0_SHMEM_Q6_K", "u32")
            },
            // Dense-f32 fallback loader. Stays on naga: the Slang SPIR-V
            // passthrough is Vulkan-only and covers just the quantized loaders
            // (the dense f32 loader was never naga-regressed).
            mul_mat_reg_tile_f32: build_mul_mat_pipeline(
                &ctx,
                "mul_mat_f32",
                "INIT_SRC0_SHMEM_FLOAT",
                "f32",
            ),
            attention_prefill: ctx.create_pipeline(
                shaders::ATTENTION_PREFILL,
                "attention_prefill",
                "attention_prefill",
            ),
            moe_route: ctx.create_pipeline(shaders::MOE_ROUTE, "moe_route", "moe_route"),
            moe_gemv_q4_0: ctx.create_pipeline(
                shaders::MOE_GEMV_Q4_0,
                "moe_gemv_q4_0",
                "moe_gemv_q4_0",
            ),
            moe_combine: ctx.create_pipeline(shaders::MOE_COMBINE, "moe_combine", "moe_combine"),
        };

        // Upload weights: Q4_0/Q8_0/Q6K stay quantized, others dequantized to f32.
        // The input-embedding lookup reads rows from this mmap handle (kept
        // past load) and dequantizes on the fly — no host copy of the table.
        // The (tied) logit projection reads the same raw bytes below and must
        // stay UNSCALED; Granite's embedding multiplier is applied at gather
        // time instead (no-op for every other arch), exactly like the CPU
        // LlamaModel (`scale_inplace` after `dequantize_row`).
        let gguf_arc = Arc::new(src.gguf().clone());
        let emb_table = MmapWeight::from_gguf(&gguf_arc, "token_embd.weight")?;

        // The logit-projection weight, as GGUF stores it: `output.weight` when
        // untied, else the tied embedding table.
        let (lm_head_dtype, lm_head_bytes) = match src.output_ref() {
            Some(wref) => (wref.dtype, src.weight_bytes(wref)),
            None => (emb_table.dtype, src.embedding_tensor_data()?),
        };
        // Resident stream layout for the logit projection (Q4_0 only): the
        // vocab projection is ~19% of prefill FLOPs, so it gets the repack
        // like every other eligible weight. Repacked on the host here; the
        // GPU upload happens inside the Quantized arm below so the F16
        // fallback never pays for buffers it drops.
        let lm_head_repack =
            if stream_layout && stream_layout_eligible(lm_head_dtype, config.hidden_size) {
                Some(repack_q4_0_stream(
                    &lm_head_bytes,
                    config.vocab_size,
                    config.hidden_size,
                ))
            } else {
                None
            };
        // Compare the size the buffers will actually be *bound* at, not the
        // raw GGUF length: `upload_storage` rounds up to
        // COPY_BUFFER_ALIGNMENT, and Q6_K (210 B/block), Q4_0 (18 B) and Q8_0
        // (34 B) are all 2 mod 4, so an odd block count does round up. Same
        // reasoning as `encode_gemv_f16`'s tiled/non-tiled check — on an
        // adapter whose `max_binding` is not itself a multiple of 4,
        // comparing the raw length could pick this path and then fail
        // binding validation. Resident binds q and d separately, so the
        // bound is the larger half, not the raw total.
        let lm_head_bound_bytes = if let Some((q, d)) = lm_head_repack.as_ref() {
            // u32 counts times 4: already a multiple of 4, no round-up.
            q.len().max(d.len()) as u64 * 4
        } else {
            (lm_head_bytes.len() as u64).div_ceil(4) * 4
        };
        // `[m, k, 0, 0]`; identical for both variants, so it is built once.
        let lm_head_params = ctx.upload_storage(
            bytemuck::cast_slice(&[
                config.vocab_size as u32,
                config.hidden_size as u32,
                0u32,
                0u32,
            ]),
            "lm_head.params",
        );
        // Take the quantized path only if the whole weight also fits one storage
        // binding — the GEMV kernels bind it entire, whereas the f16 path can
        // fall back to `encode_gemv_f16_tiled`. Quantized is the smaller of the
        // two, so this rejects only weights the f16 path would have had to tile
        // anyway.
        let lm_head = if Self::has_quantized_gemv(lm_head_dtype)
            && lm_head_bound_bytes <= ctx.max_storage_buffer_binding_size
        {
            // Resident weights keep only the (q, d) repack: `tensor.buffer`
            // is the q half (see `GpuWeight`), and there is no raw upload.
            let (main_buffer, stream_q, stream_d, resident_stream) = match lm_head_repack {
                Some((q, d)) => {
                    let qb = ctx.upload_storage(bytemuck::cast_slice(&q), "lm_head.stream_q");
                    let db = ctx.upload_storage(bytemuck::cast_slice(&d), "lm_head.stream_d");
                    (qb.clone(), Some(qb), Some(db), true)
                }
                None => (
                    ctx.upload_storage(&lm_head_bytes, "lm_head"),
                    None,
                    None,
                    false,
                ),
            };
            // Flat-planes Q6_K twin for decode GEMV (passthrough only): the
            // same bytes as the interleaved upload, de-interleaved, so the
            // flat kernel reads aligned words with no funnel shifts. The
            // interleaved buffer stays as `main` for the prefill GEMM's
            // reg-tile loader, which expects GGUF block order.
            let flat = if lm_head_dtype == DType::Q6K && use_flat_q6k(&ctx) {
                let flat_bytes =
                    repack_q6_k_flat(&lm_head_bytes, config.vocab_size, config.hidden_size);
                Some(Box::new(GpuWeight {
                    tensor: GpuTensor {
                        buffer: ctx.upload_storage(&flat_bytes, "lm_head.flat_q6k"),
                        dtype: lm_head_dtype,
                        shape: vec![config.vocab_size, config.hidden_size],
                    },
                    stream_q: None,
                    stream_d: None,
                    resident_stream: false,
                    flat_q6k: true,
                    params_buf: lm_head_params.clone(),
                    cached_bg: None,
                }))
            } else {
                None
            };
            LmHead::Quantized {
                main: GpuWeight {
                    tensor: GpuTensor {
                        buffer: main_buffer,
                        dtype: lm_head_dtype,
                        shape: vec![config.vocab_size, config.hidden_size],
                    },
                    stream_q,
                    stream_d,
                    resident_stream,
                    flat_q6k: false,
                    params_buf: lm_head_params,
                    cached_bg: None,
                },
                flat,
            }
        } else {
            // No quantized GEMV for this dtype (or it needs tiling): dequantize
            // and keep an f16 copy, which is still half the VRAM of f32.
            let f16_weight = match src.output_ref() {
                // Untied head: same row-wise conversion as the tied arm, so
                // no host f32 copy of the vocab-sized table ever exists.
                // `output.weight` is the key `output_ref` resolves (llama
                // family; LFM2 always ties and takes the arm below).
                Some(_) => {
                    let out_table = MmapWeight::from_gguf(&gguf_arc, "output.weight")?;
                    upload_mmap_table_as_f16(&ctx, &out_table, "output.weight")
                }
                None => upload_mmap_table_as_f16(&ctx, &emb_table, "token_embd.weight"),
            };
            LmHead::F16 {
                weight: f16_weight,
                params: lm_head_params,
            }
        };

        let output_norm = ctx.upload_f32(src.output_norm_weight(), "output_norm");

        let mut uploaded_weights: std::collections::HashMap<(u64, usize), GpuWeight> =
            std::collections::HashMap::new();

        let mut upload_weight = |wref: &WeightRef, name: &str| -> GpuWeight {
            let key = (wref.start, wref.size);
            if let Some(existing) = uploaded_weights.get(&key) {
                return existing.clone();
            }
            let (buf, dtype, stream, resident_stream) = if matches!(
                wref.dtype,
                DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q5KM | DType::Q6K
            ) {
                // All five have a native quantized GEMV (decode) and a batched
                // GEMM loader (`mul_mat_reg_tile_*`, prefill), so none falls to
                // the per-token prefill path. They stay quantized on the GPU
                // rather than dequantizing to f32: ~7× less VRAM for Q4KM
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
                // Resident stream layout: eligible Q4_0 uploads the (q, d)
                // repack INSTEAD of raw (same bytes, transposed) and both
                // phases read it directly — no second copy, no per-GEMM
                // repack. `tensor.buffer` is the q half (see `GpuWeight`).
                let resident = stream_layout && stream_layout_eligible(wref.dtype, wref.k);
                if resident {
                    let (q, d) = repack_q4_0_stream(&data, wref.m, wref.k);
                    let qb =
                        ctx.upload_storage(bytemuck::cast_slice(&q), &format!("{name}.stream_q"));
                    let db =
                        ctx.upload_storage(bytemuck::cast_slice(&d), &format!("{name}.stream_d"));
                    (qb.clone(), wref.dtype, Some((qb, db)), true)
                } else {
                    let buf = ctx.upload_storage(&data, name);
                    (buf, wref.dtype, None, false)
                }
            } else {
                // Every other dtype (F16/BF16/F32 sources, Q4_1, Q2_K, ...) is
                // dequantized to F32 here. F32 has both a decode GEMV (`gemv_f32`)
                // and a batched prefill loader (`mul_mat_reg_tile_f32`), so these
                // weights ride the fast batched path too rather than forcing the
                // whole model onto the per-token loop.
                //
                // TODO: Upload as F16 to halve this weight bandwidth (needs an
                // F16-aware reg-tile loader); a perf optimization, not correctness.
                let f32_data = src.dequantize_weight(wref);
                (ctx.upload_f32(&f32_data, name), DType::F32, None, false)
            };
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[wref.m as u32, wref.k as u32, 0u32, 0u32]),
                &format!("{name}.params"),
            );
            let (stream_q, stream_d) = match stream {
                Some((q, d)) => (Some(q), Some(d)),
                None => (None, None),
            };
            let weight = GpuWeight {
                tensor: GpuTensor {
                    buffer: buf,
                    dtype,
                    shape: vec![wref.m, wref.k],
                },
                stream_q,
                stream_d,
                resident_stream,
                // Flat planes are an LM-head-only twin (see `LmHead`); layer
                // weights always read the interleaved upload.
                flat_q6k: false,
                params_buf,
                cached_bg: None,
            };
            uploaded_weights.insert(key, weight.clone());
            weight
        };

        // Optional per-head QK-norm (Qwen3) and QKV bias (Qwen2) upload helpers.
        let upload_opt_f32 = |data: Option<&[f32]>, name: &str| -> Option<wgpu::Buffer> {
            data.map(|d| ctx.upload_f32(d, name))
        };

        // The largest batch the prefill path hands the kernels in one dispatch,
        // and the token count every buffer sized per-token below multiplies by:
        // the routed-FFN scratch here, and further down `lora_tmp_batched` and
        // the five `prefill_*_buf`. Any of them smaller than a chunk is an
        // out-of-bounds device write, not a load error, so it is bound once.
        let max_pref = max_seq_len.min(MAX_PREFILL_TOKENS);

        // Routed-FFN scratch: one allocation for the whole model, shared by every
        // routed layer, so decode reuses it as the n = 1 case rather than
        // allocating per call.
        let moe_scratch = config
            .moe
            .as_ref()
            .map(|m| -> Result<Arc<MoeScratch>> {
                use anyhow::Context;

                // Every scratch buffer is bound whole by the kernel that reads
                // it, so each one is capped by the adapter's per-binding limit,
                // not just by `create_storage_rw`'s `max_buffer_size` assert.
                // `z` is the one that actually reaches for it: it is
                // `entries x hidden_size` floats, and nothing else here bounds
                // the hidden size. Named error at load rather than a wgpu
                // validation failure on the first forward.
                let buf = |n: usize, name: &str| -> Result<wgpu::Buffer> {
                    let bytes = n as u64 * 4;
                    anyhow::ensure!(
                        bytes <= ctx.max_storage_buffer_binding_size,
                        "mixture-of-experts scratch `{name}` needs {bytes} bytes, over this \
                         adapter's {} byte storage-binding limit",
                        ctx.max_storage_buffer_binding_size,
                    );
                    Ok(ctx.create_storage_rw(bytes, name))
                };
                // The only enforcement of the kernels' expert-count limits in
                // this backend (Metal has its own copy of this check, on the
                // same constants), and it has to run before the sizing below: a
                // GGUF declaring a wild `n_expert_used` would otherwise drive
                // `entries * expert_ff_len` into the allocator and fail as an
                // allocation rather than with the named error written for it.
                // `upload_moe` validates each layer's *shapes* against the
                // scratch, but it does not re-check these bounds, so do not move
                // or weaken this without reading that function.
                anyhow::ensure!(
                    (1..=MOE_MAX_EXPERTS as usize).contains(&m.n_expert)
                        && (1..=MOE_MAX_EXPERT_USED as usize).contains(&m.n_expert_used)
                        && m.n_expert_used <= m.n_expert,
                    "wgpu MoE routing supports 1..={MOE_MAX_EXPERTS} experts and \
                     1..={MOE_MAX_EXPERT_USED} active (and no more active than available), \
                     model declares {} and {}",
                    m.n_expert,
                    m.n_expert_used,
                );
                let entries = max_pref * m.n_expert_used;
                let dim = |v: usize, what: &str| -> Result<u32> {
                    u32::try_from(v).with_context(|| {
                        format!("mixture-of-experts {what} {v} does not fit the kernels' u32")
                    })
                };
                // `expert_ff_len` is a file-declared number with no bound above
                // (unlike the two expert counts, which the kernels cap), so the
                // row products are checked: an overflow here wraps in release and
                // under-allocates a buffer the kernels then write past.
                let cells = |rows: usize, cols: usize, what: &str| -> Result<usize> {
                    rows.checked_mul(cols).with_context(|| {
                        format!("mixture-of-experts {what} scratch size overflows")
                    })
                };
                let max_entries = dim(entries, "entry count")?;
                anyhow::ensure!(
                    max_entries <= crate::backend::wgpu::MAX_WG,
                    "a {max_pref}-token chunk over {} experts per token is {max_entries} \
                     entries, over the {} workgroups-per-dimension cap the expert GEMV \
                     dispatches one entry per workgroup against",
                    m.n_expert_used,
                    crate::backend::wgpu::MAX_WG,
                );
                // The routed SwiGLU is the third and last grid this scratch
                // sizes, and the only one that is not one workgroup per row or
                // per entry: `silu_mul_inplace` takes 256 elements each and
                // indexes by a bare `gid.x`, so its extent is the whole
                // `entries x expert_ff_len` slab divided by 256. Checked in the
                // same place as the other two, so all three of the scratch's
                // dispatch bounds are one screenful apart rather than one of
                // them surfacing as a wgpu validation failure mid-prefill.
                let silu_groups = cells(entries, m.expert_ff_len, "SwiGLU")?.div_ceil(256);
                anyhow::ensure!(
                    silu_groups <= crate::backend::wgpu::MAX_WG as usize,
                    "the routed SwiGLU over {max_entries} entries of width {} needs \
                     {silu_groups} workgroups, over this backend's {} per-dimension cap",
                    m.expert_ff_len,
                    crate::backend::wgpu::MAX_WG,
                );
                Ok(Arc::new(MoeScratch {
                    n_expert: dim(m.n_expert, "expert count")?,
                    n_expert_used: dim(m.n_expert_used, "active expert count")?,
                    expert_ff_len: dim(m.expert_ff_len, "expert width")?,
                    max_entries,
                    logits: buf(cells(max_pref, m.n_expert, "logits")?, "moe.logits")?,
                    sel_expert: buf(entries, "moe.sel_expert")?,
                    sel_weight: buf(entries, "moe.sel_weight")?,
                    gate: buf(cells(entries, m.expert_ff_len, "gate")?, "moe.gate")?,
                    up: buf(cells(entries, m.expert_ff_len, "up")?, "moe.up")?,
                    z: buf(cells(entries, hs, "output")?, "moe.z")?,
                }))
            })
            .transpose()?;

        let mut uploaded_norms: std::collections::HashMap<usize, (wgpu::Buffer, wgpu::Buffer)> =
            std::collections::HashMap::new();
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let phys_idx = src
                .loop_norm_interval()
                .map(|n_phys| i % n_phys)
                .unwrap_or(i);
            let (attn_norm, ffn_norm) = uploaded_norms
                .entry(phys_idx)
                .or_insert_with(|| {
                    (
                        ctx.upload_f32(src.attn_norm_weight(i), &format!("l{i}.anorm")),
                        ctx.upload_f32(src.ffn_norm_weight(i), &format!("l{i}.fnorm")),
                    )
                })
                .clone();

            let ffn = match src.moe_refs(i) {
                None => GpuFfn::Dense(Box::new(GpuDenseFfn {
                    gate: upload_weight(src.ffn_gate_ref(i)?, &format!("l{i}.ffn_gate")),
                    up: upload_weight(src.ffn_up_ref(i)?, &format!("l{i}.ffn_up")),
                    down: upload_weight(src.ffn_down_ref(i)?, &format!("l{i}.ffn_down")),
                })),
                Some(m) => GpuFfn::Moe(Box::new(upload_moe(
                    &ctx,
                    src,
                    moe_scratch.as_ref(),
                    hs,
                    i,
                    m,
                )?)),
            };

            let is_conv = config.block_types[i] == BlockType::GatedConv;

            let (conv_in_proj, conv_out_proj, conv_weight) = if is_conv {
                let ip = src
                    .conv_in_proj_ref(i)
                    .ok_or_else(|| anyhow!("conv layer missing in_proj"))?;
                let op = src
                    .conv_out_proj_ref(i)
                    .ok_or_else(|| anyhow!("conv layer missing out_proj"))?;
                (
                    Some(upload_weight(ip, &format!("l{i}.conv_ip"))),
                    Some(upload_weight(op, &format!("l{i}.conv_op"))),
                    Some(
                        ctx.upload_f32(
                            src.conv_weight(i)
                                .ok_or_else(|| anyhow!("conv layer missing conv weight"))?,
                            &format!("l{i}.conv_w"),
                        ),
                    ),
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
                        src.attn_q_ref(i)
                            .ok_or_else(|| anyhow!("attn layer missing q"))?,
                        &format!("l{i}.attn_q"),
                    )),
                    Some(upload_weight(
                        src.attn_k_ref(i)
                            .ok_or_else(|| anyhow!("attn layer missing k"))?,
                        &format!("l{i}.attn_k"),
                    )),
                    Some(upload_weight(
                        src.attn_v_ref(i)
                            .ok_or_else(|| anyhow!("attn layer missing v"))?,
                        &format!("l{i}.attn_v"),
                    )),
                    Some(upload_weight(
                        src.attn_output_ref(i)
                            .ok_or_else(|| anyhow!("attn layer missing output"))?,
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
                ffn,
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
                attn_norm_bg: None,
                ffn_norm_bg: None,
                rope_bg: None,
                conv_fused_bg: None,
                conv_add_bg: None,
                attn_out_add_bg: None,
                silu_bg: None,
                ffn_swiglu_bg: None,
                attn_bg: None,
                k_append_bg: None,
                v_append_bg: None,
                qn_bg: None,
                kn_bg: None,
                qb_bg: None,
                kb_bg: None,
                vb_bg: None,
                ffn_add_bg: None,
            });
        }

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
        // (`max_seq_len × max_kv_dim`). Stays f32 (×4, NOT `kv_slab_bytes`):
        // `kv_shift` re-rotates into f32 scratch, then a `kv_append` dispatch
        // packs the result back into the cache. No `.max(1)` guard —
        // `k_buf`/`v_buf` above already allocate `max_kv_dim` floats, so an
        // attention-free (`max_kv_dim == 0`) config would fail there first;
        // LFM2 always has attention layers, so `max_kv_dim` is never 0 in
        // practice anyway.
        let kv_shift_scratch = ctx.create_storage_rw(
            max_seq_len as u64 * max_kv_dim as u64 * 4,
            "kv_shift_scratch",
        );
        let attn_out_buf = f(q_dim, "attn_out");
        let logits_buf = f(config.vocab_size, "logits");
        let conv_proj_buf = f(3 * hs, "conv_proj");
        let conv_gate_buf = f(hs, "conv_gate");

        // Batched-prefill scratch. Sized for the worst case of
        // `MAX_PREFILL_TOKENS` rows (`max_pref`, bound above the weight upload
        // because the routed-FFN scratch is sized from it too); chunking on the
        // host side keeps larger prompts within this footprint.
        //
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
        // Streaming-GEMM B16 scratch: [max_k × max_pref_padded] halfs. max_k
        // covers every Q4_0 matmul K (hs for most projections, `is` for FFN
        // down, q_dim for attention out). Only when the streaming path is on.
        let stream_b16_buf = if stream_layout {
            let max_k = is.max(q_dim).max(hs);
            let n_pad = max_pref.next_multiple_of(32);
            Some(ctx.create_storage_rw((max_k * n_pad * 2) as u64, "stream_b16"))
        } else {
            None
        };
        let max_all_logits = max_seq_len.min(MAX_ALL_LOGITS_TOKENS);
        let prefill_all_logits_buf = f(config.vocab_size * max_all_logits, "prefill_all_logits");

        // Conv rolling buffers are always needed and are tiny (`d_conv × hs`), so
        // they stay eager. The packed-f16 KV caches are context-scaled and mode-dependent,
        // so they are built on first use — see `GpuState::kv_caches`.
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let d_conv = kernel_size - 1;
        let mut conv_buffers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            if config.block_types[i] == BlockType::Attention {
                conv_buffers.push(None);
            } else {
                let cb = f(d_conv * hs, &format!("l{i}.conv_buf"));
                conv_buffers.push(Some(cb));
            }
        }

        let gpu_state = GpuState {
            kv_caches: OnceLock::new(),
            conv_buffers,
            seq_len: AtomicUsize::new(0),
            max_seq_len,
            embedding: emb_table,
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
        let kv_append_params = ctx.create_storage_rw(4 * 4, "kv_append_params");
        // Row-tile params for `encode_gemv_f16_tiled`, which only the f16 LM head
        // can reach — the quantized variant binds its weight entire or is not
        // chosen at all. Empty on that path rather than allocated and unused.
        // The `2` is the f16 element size, matching what those tiles bind.
        let gemv_tile_params = if matches!(lm_head, LmHead::F16 { .. }) {
            let tile_rows = gemv_tile_rows(
                config.vocab_size as u32,
                hs as u32,
                ctx.max_storage_buffer_binding_size,
                ctx.min_storage_buffer_offset_alignment,
                2,
            );
            let tile_count = (config.vocab_size as u32).div_ceil(tile_rows);
            (0..tile_count)
                .map(|i| ctx.create_storage_rw(4 * 4, &format!("gemv_tile_params.{i}")))
                .collect()
        } else {
            Vec::new()
        };

        // Argmax I/O buffers. `argmax_params` is uploaded once with
        // vocab_size; `argmax_out_buf` holds up to MAX_PREFILL_TOKENS u32 token IDs. Bind group is
        // built after `pipelines` exists below.
        let argmax_out_buf =
            ctx.create_storage_rw((MAX_PREFILL_TOKENS * 4).max(64) as u64, "argmax_out");
        let argmax_readback_buf =
            ctx.create_readback_buffer((MAX_PREFILL_TOKENS * 4).max(64) as u64, "argmax_readback");
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
        // The temporary CPU source is dropped after upload. Resolve the named
        // identity now without retaining another CPU copy solely for hashing.
        let model_id = super::cache_identity::for_gpu_source(src, &model_id);
        let prefix_cache = Mutex::new(KvPrefixCache::for_model(
            crate::kv_cache::KvCacheConfig::default(),
            &config,
            &model_id,
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
            lm_head,
            output_norm,
            layers,
            rope_type,
            scalars,
            loop_norm_interval,
            batched_prefill,
            batched_fallback_warned: AtomicBool::new(false),
            moe_lora_dropped_warned: AtomicBool::new(false),
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
            argmax_readback_buf,
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
            kv_append_params,
            gemv_tile_params,
            conv_proj_buf,
            conv_gate_buf,
            prefill_batch_buf,
            prefill_normed_buf,
            prefill_proj_buf,
            stream_b16_buf,
            prefill_gate_buf,
            prefill_up_buf,
            prefill_all_logits_buf,
            gpu_state,
            infer_lock: Mutex::new(()),
            session_gate: super::ModelSessionGate::default(),
            hs_scratch: OnceLock::new(),
            use_hs_scratch: AtomicBool::new(false),
            tq: OnceLock::new(),
            kv_mode: OnceLock::new(),
            kv_cache_tag: OnceLock::new(),
            prefix_cache,
            model_id,
            lora_lru: Mutex::new(Vec::new()),
            active_lora: Mutex::new(None),
            lora_tmp,
            lora_tmp_batched,
            lora_params_pool: Mutex::new((Vec::new(), 0)),
            prefill_params_pool: Mutex::new((Vec::new(), 0)),
            stream_gemm_bg_cache: Mutex::new((u32::MAX, false, 0, Vec::new())),
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
        // `Session::attach_lora_adapters` refuses an adapter carrying routed-FFN
        // deltas, because this backend applies none of them (see
        // `supports_moe_lora`). That gate is not the only way in:
        // `InferenceState::lora` is a public field and `Model::forward` takes
        // the state directly, so a caller driving the trait (the parity harness,
        // an FFI embedder, a test) reaches here without passing it.
        //
        // Dropping the whole adapter is the conservative arm. Applying its
        // attention half while silently dropping the router and per-expert
        // deltas is exactly the fluent-but-wrong outcome the gate exists to
        // prevent, and it is the harder of the two to notice. Logged once per
        // model, since a generation loop would otherwise repeat it per token.
        let usable = state.lora.as_ref().filter(|adapter| {
            let ok = !adapter.has_moe_deltas();
            if !ok && !self.moe_lora_dropped_warned.swap(true, Ordering::Relaxed) {
                tracing::error!(
                    "ignoring a LoRA adapter that carries mixture-of-experts deltas: this \
                     backend has no routed-FFN hooks. Attach through `Session`, which refuses \
                     it with `CeraError::LoraUnsupportedByBackend` instead of ignoring it."
                );
            }
            ok
        });
        let resolved = usable.map(|adapter| {
            let mut lru = self.lora_lru.lock().unwrap_or_else(|e| e.into_inner());
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
        *self.active_lora.lock().unwrap_or_else(|e| e.into_inner()) = resolved;
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
            .unwrap_or_else(|e| e.into_inner());
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

    /// Pull a pooled prefill parameter buffer for one batched dispatch. Sized and
    /// cached so decode and speculative verification perform 0 dynamic GPU memory allocations.
    fn next_prefill_params(&self, data: &[u8]) -> wgpu::Buffer {
        let mut pool = self
            .prefill_params_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (bufs, next) = &mut *pool;
        let idx = *next;
        *next += 1;
        let aligned_size = (data.len().max(16).div_ceil(4) * 4) as u64;
        if bufs.len() <= idx || bufs[idx].size() < aligned_size {
            if bufs.len() <= idx {
                bufs.push(
                    self.ctx
                        .create_storage_rw(aligned_size.max(64), "prefill_batched_params"),
                );
            } else {
                bufs[idx] = self
                    .ctx
                    .create_storage_rw(aligned_size.max(64), "prefill_batched_params");
            }
        }
        let buf = bufs[idx].clone();
        self.ctx.queue.write_buffer(&buf, 0, data);
        buf
    }

    /// Batched-prefill LoRA delta for one target, applied in-batch across all `n`
    /// tokens: `Y[n×d] += B_batched · (A · X[n×k])`, computed as two NT GEMMs
    /// (`gemm_f32_nt` / `gemm_f32_nt_accum`) that match the token-major batch
    /// buffer layout (`X[tok*k + i]`, `Y[tok*d + o]`).
    ///
    /// `t.b_batched` carries only the `alpha/rank` scale (not `residual_mult`) —
    /// the caller applies the LoRA before the fused residual add, so the model's
    /// residual scale wraps the delta (matches `lora::apply_prefill`). Both GEMMs
    /// share the merged prefill pass: wgpu's intra-pass dispatch ordering plus
    /// its automatic barriers keep the shared `lora_tmp_batched` write-then-read
    /// ordered (GEMM1 writes it, GEMM2 reads it), as the `PrefillCmd` doc notes.
    fn encode_lora_batched<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.gemm_f32_nt,
            bg1,
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.gemm_f32_nt_accum,
            bg2,
            crate::backend::wgpu::gemv_row_workgroups(total2),
            "lora_batched_b",
        );
    }

    /// Batched-prefill counterpart of the decode `dispatch_lora_into`: apply the
    /// `(layer, target)` LoRA delta across all `n` tokens if the active adapter
    /// touches it. `input`/`output` are token-major batch buffers at offset 0.
    #[allow(clippy::too_many_arguments)]
    fn encode_lora_hook_batched<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
        lora: Option<&Arc<WgpuLoraAdapter>>,
        layer: usize,
        target: LoraTarget,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) {
        if let Some(t) = Self::lora_target(lora, layer, target) {
            self.encode_lora_batched(cmds, t, input, output, n);
        }
    }

    /// Create a `kv_append` bind group: src scratch row -> cache slab, slot
    /// from the shared per-token `kv_append_params`.
    fn make_kv_append_bg(&self, src: &wgpu::Buffer, cache: &wgpu::Buffer) -> wgpu::BindGroup {
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kv_append"),
                layout: &self.pipelines.kv_append.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.kv_append_params.as_entire_binding(),
                    },
                ],
            })
    }

    /// Create a GEMV bind group for a given (weight, input, output) triple.
    fn make_gemv_bg(
        &self,
        w: &GpuWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let (pipeline, _, _) = self.gemv_pipeline_rows_label(w);
        // Resident stream weights bind (q, d) as two buffers; `tensor.buffer`
        // is the q half (see `GpuWeight`).
        let mut entries = Vec::with_capacity(5);
        if w.resident_stream {
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: w.tensor.buffer.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: w
                    .stream_d
                    .as_ref()
                    .expect("resident-stream weight without stream_d")
                    .as_entire_binding(),
            });
        } else {
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: w.tensor.buffer.as_entire_binding(),
            });
        }
        let (b_in, b_out, b_params) = if w.resident_stream {
            (2, 3, 4)
        } else {
            (1, 2, 3)
        };
        entries.push(wgpu::BindGroupEntry {
            binding: b_in,
            resource: input.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: b_out,
            resource: output.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: b_params,
            resource: w.params_buf.as_entire_binding(),
        });
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    }

    fn gemv_pipeline_rows_label(
        &self,
        w: &GpuWeight,
    ) -> (&wgpu::ComputePipeline, u32, &'static str) {
        // rows-per-workgroup MUST match each shader's `NR`/`ROWS_PER_WG`
        // constant: gemv_q4_0_fast=8, gemv_q4_0_stream=16, gemv_q8_0=8,
        // gemv_q4_k=2, gemv_q5_k=2, gemv_q6_k=1 on the SPIR-V passthrough
        // (2 on the WGSL fallback — gated below), gemv_q6_k_flat=8,
        // gemv_f32=8. Too large and
        // rows are silently dropped, in every kernel here. Too small
        // over-dispatches, and what that costs is per kernel:
        // `gemv_q4_0_fast` is the one that reads past the weight buffer,
        // since it alone guards writes but not weight reads. The rest guard
        // the read path too (`gemv_q5_k` returns early for a whole
        // workgroup, the others skip per row), so they only burn dispatches.
        if w.resident_stream {
            // Resident layout implies passthrough implies the pipeline
            // exists (same `stream_layout` gate at load).
            let pipe = self
                .pipelines
                .gemv_q4_0_stream
                .as_ref()
                .expect("resident-stream weight without a gemv_q4_0_stream pipeline");
            return (pipe, 16, "gemv_q4s");
        }
        match w.tensor.dtype {
            DType::Q4_0 => (&self.pipelines.gemv_q4_0_fast, 8, "gemv_q4"),
            DType::Q8_0 => (&self.pipelines.gemv_q8_0, 8, "gemv_q8"),
            DType::Q4KM => (&self.pipelines.gemv_q4_k, 2, "gemv_q4k"),
            // The Q6_K SPIR-V twin runs NR=1 while the WGSL fallback runs
            // NR=2; the pipeline was picked under this same condition at
            // construction, so mirror it here. Flat-planes twins (LM head
            // on passthrough) run NR=8 on the flat kernel.
            DType::Q6K => {
                if w.flat_q6k {
                    let pipe = self
                        .pipelines
                        .gemv_q6_k_flat
                        .as_ref()
                        .expect("flat Q6_K weight without a gemv_q6_k_flat pipeline");
                    return (pipe, 8, "gemv_q6f");
                }
                let rows = if self.ctx.supports_spirv_passthrough() && self.ctx.has_subgroup {
                    1
                } else {
                    2
                };
                (&self.pipelines.gemv_q6_k, rows, "gemv_q6")
            }
            DType::Q5KM => (&self.pipelines.gemv_q5_k, 2, "gemv_q5k"),
            _ => (&self.pipelines.gemv_f32, 8, "gemv_f32"),
        }
    }

    /// Whether `dtype` has a native quantized GEMV kernel — i.e. whether
    /// [`Self::gemv_pipeline_rows_label`] maps it to something other than the
    /// f32 fallback.
    ///
    /// Immediately above on purpose: adding a kernel there without adding the
    /// dtype here silently leaves the LM head being dequantized to f16, which
    /// costs throughput and VRAM and nothing fails.
    fn has_quantized_gemv(dtype: DType) -> bool {
        matches!(
            dtype,
            DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q5KM | DType::Q6K
        )
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

    /// Pre-create bind groups for all per-layer dispatches (GEMVs, norms, RoPE, elementwise ops).
    /// Eliminates ~250 create_bind_group calls per token.
    fn cache_bind_groups(&mut self) {
        let cfg = &self.config;
        for i in 0..cfg.n_layers {
            // Out-of-place norms (`rmsnorm_out` entry, bindings 3..6): both read
            // `hidden` directly, so no hidden->scratch blit splits the layer's
            // passes. The bind group pins buffers, not contents — the attn norm
            // reads the block input, the FFN norm the post-block residual.
            let attn_norm_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.rmsnorm_out.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.hidden_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.normed_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.layers[i].attn_norm.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.rmsnorm_hs_params.as_entire_binding(),
                        },
                    ],
                });

            let ffn_norm_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.rmsnorm_out.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.hidden_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.ffn_input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.layers[i].ffn_norm.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.rmsnorm_hs_params.as_entire_binding(),
                        },
                    ],
                });

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
                            resource: self.elementwise_is_params.as_entire_binding(),
                        },
                    ],
                });

            let ffn_add_bg = self
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

            let (conv_fused_bg, conv_add_bg) = if cfg.block_types[i] == BlockType::GatedConv {
                let conv_buf = self.active_conv(i);
                let conv_p = &self.conv1d_params;
                let bg_fused = self
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
                                resource: self.layers[i]
                                    .conv_weight
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
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
                let add_p = &self.elementwise_hs_params;
                let bg_add = self
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
                (Some(bg_fused), Some(bg_add))
            } else {
                (None, None)
            };

            let (rope_bg, attn_out_add_bg) = if cfg.block_types[i] != BlockType::GatedConv {
                let bg_rope = self
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
                let bg_add = self
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
                (Some(bg_rope), Some(bg_add))
            } else {
                (None, None)
            };

            let layer = &mut self.layers[i];
            layer.attn_norm_bg = Some(attn_norm_bg);
            layer.ffn_norm_bg = Some(ffn_norm_bg);
            layer.silu_bg = Some(silu_bg);
            layer.ffn_add_bg = Some(ffn_add_bg);
            layer.conv_fused_bg = conv_fused_bg;
            layer.conv_add_bg = conv_add_bg;
            layer.rope_bg = rope_bg;
            layer.attn_out_add_bg = attn_out_add_bg;

            // FFN. Only the dense arm caches: the routed one builds its bind
            // groups per call in `moe_ffn_steps`, so there is nothing here for it
            // to look up.
            if let GpuFfn::Dense(d) = &self.layers[i].ffn {
                let gate_bg = self.make_gemv_bg(&d.gate, &self.ffn_input_buf, &self.gate_buf);
                let up_bg = self.make_gemv_bg(&d.up, &self.ffn_input_buf, &self.up_buf);
                let down_bg = self.make_gemv_bg(&d.down, &self.gate_buf, &self.out_buf);
                let ffn_swiglu_bg = if d.gate.tensor.dtype == DType::Q4_0
                    && d.up.tensor.dtype == DType::Q4_0
                {
                    Some(
                        self.ctx
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("ffn_swiglu_q4_0"),
                                layout: &self.pipelines.ffn_swiglu_q4_0.get_bind_group_layout(0),
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: d.gate.tensor.buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: d.up.tensor.buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: self.ffn_input_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: self.gate_buf.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: d.gate.params_buf.as_entire_binding(),
                                    },
                                ],
                            }),
                    )
                } else {
                    None
                };
                if let GpuFfn::Dense(d) = &mut self.layers[i].ffn {
                    d.gate.cached_bg = Some(gate_bg);
                    d.up.cached_bg = Some(up_bg);
                    d.down.cached_bg = Some(down_bg);
                }
                self.layers[i].ffn_swiglu_bg = ffn_swiglu_bg;
            }

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

                if let Some(kv) = self.f16_kv().get(i).and_then(|opt| opt.as_ref()) {
                    let (k_cache, v_cache) = kv;
                    let attn_bg = self
                        .ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("flash_attention"),
                            layout: &self.pipelines.flash_attention.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.q_buf.as_entire_binding(),
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
                                    resource: self.attn_out_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: self.attn_params.as_entire_binding(),
                                },
                            ],
                        });
                    let k_bg = self.make_kv_append_bg(&self.k_buf, k_cache);
                    let v_bg = self.make_kv_append_bg(&self.v_buf, v_cache);
                    self.layers[i].attn_bg = Some(attn_bg);
                    self.layers[i].k_append_bg = Some(k_bg);
                    self.layers[i].v_append_bg = Some(v_bg);
                }

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
                self.layers[i].qn_bg = self.layers[i]
                    .attn_q_norm
                    .as_ref()
                    .map(|w| per_head_norm_bg(&self.q_buf, w));
                self.layers[i].kn_bg = self.layers[i]
                    .attn_k_norm
                    .as_ref()
                    .map(|w| per_head_norm_bg(&self.k_buf, w));

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
                self.layers[i].qb_bg = self.layers[i]
                    .attn_q_bias
                    .as_ref()
                    .map(|b| bias_bg(&self.q_buf, b, &self.elementwise_qdim_params));
                self.layers[i].kb_bg = self.layers[i]
                    .attn_k_bias
                    .as_ref()
                    .map(|b| bias_bg(&self.k_buf, b, &self.elementwise_kvdim_params));
                self.layers[i].vb_bg = self.layers[i]
                    .attn_v_bias
                    .as_ref()
                    .map(|b| bias_bg(&self.v_buf, b, &self.elementwise_kvdim_params));
            }
        }

        // The LM head runs once per token over fixed buffers, so its bind group
        // is as cacheable as the per-layer ones. (The f16 variant builds its own;
        // it is a single dispatch and not worth another field.)
        // Built then stored, rather than one `&mut` match: `make_gemv_bg` takes
        // `&self`, so it cannot be called while `self.lm_head` is borrowed
        // mutably. The bind group has to exist before the field is touched.
        // Cached against the GEMV view (the flat twin when present): that
        // is the weight `encode_lm_head_into` dispatches. The prefill GEMM
        // builds its own bind groups against `main`.
        let lm_head_bg = match &self.lm_head {
            LmHead::Quantized { main, flat } => {
                let view = flat.as_deref().unwrap_or(main);
                Some(self.make_gemv_bg(view, &self.hidden_buf, &self.logits_buf))
            }
            LmHead::F16 { .. } => None,
        };
        if let (Some(bg), LmHead::Quantized { main, flat }) = (lm_head_bg, &mut self.lm_head) {
            let view = flat.as_deref_mut().unwrap_or(main);
            view.cached_bg = Some(bg);
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
        {
            let mut pass = self.ctx.begin_pass(enc, label);
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

    /// Record a prefill dispatch (see `PrefillCmd`).
    fn push_prefill_dispatch<'a>(
        cmds: &mut Vec<PrefillCmd<'a>>,
        pipeline: &'a wgpu::ComputePipeline,
        bg: wgpu::BindGroup,
        grid: (u32, u32, u32),
        label: &'static str,
    ) {
        cmds.push(PrefillCmd::Dispatch {
            pipeline,
            bg,
            grid,
            label,
        });
    }

    /// Chunk length past which `emit_prefill_cmds` splits `mul_mat_*`
    /// dispatches into their own passes (the Adreno guard): a merged pass
    /// mixing kinds loses the device on 2.6B shapes past 128 tokens. The
    /// pass-split integration test pins the boundary (128 vs 129) with its
    /// own literals — it cannot see this private const.
    const ADRENO_PASS_SPLIT_TOKENS: u32 = 128;

    /// Whether a prefill dispatch label belongs to the Adreno-split kind
    /// (`mul_mat_*`, including non-passthrough members by design), for the
    /// Adreno kind split in `emit_prefill_cmds`. Empirical split, not a
    /// SPIR-V/WGSL split: `mul_mat_*` dispatches run in their own passes
    /// past [`ADRENO_PASS_SPLIT_TOKENS`] tokens, and everything else merges
    /// as before (including `transpose_cast_f16` and `gemm_stream_*`, which
    /// both ship SPIR-V twins and are both proven to mix safely).
    /// `mul_mat_f32` has no SPIR-V twin at all (harmless over-split:
    /// F32-weight prefills are rare and stay correct). Mechanism unknown;
    /// do not reclassify labels without re-running the 2.6B n>128 Adreno
    /// soak.
    fn is_adreno_split_label(label: &str) -> bool {
        label.starts_with("mul_mat_")
    }

    /// Emit recorded prefill commands: dispatches grouped into shared passes
    /// (split at copies, which are encoder-level), one label per layer
    /// segment. When the timestamp profiler is active each dispatch keeps
    /// its own pass so per-op attribution survives — production runs merged.
    ///
    /// Adreno guard (`chunk_n > ADRENO_PASS_SPLIT_TOKENS`): a merged pass
    /// mixing `mul_mat_*` with other dispatches loses the device on 2.6B
    /// shapes past 128 tokens (zero validation errors, all maps fail after). Verified
    /// safe: same mix at n ≤ 128, either kind merged alone at n = 129,
    /// and stream+WGSL merged at n = 512, so this is an empirical
    /// `mul_mat_*`-vs-rest split, not a SPIR-V/WGSL one (mechanism
    /// unknown; see `is_adreno_split_label`). Long chunks split runs at kind
    /// boundaries too; short chunks merge exactly as before.
    fn emit_prefill_cmds(
        &self,
        enc: &mut wgpu::CommandEncoder,
        cmds: &mut Vec<PrefillCmd<'_>>,
        pass_label: &str,
        chunk_n: u32,
    ) {
        if self.ctx.profiler.is_some() {
            for cmd in cmds.drain(..) {
                match cmd {
                    PrefillCmd::Dispatch {
                        pipeline,
                        bg,
                        grid,
                        label,
                    } => {
                        self.encode(enc, pipeline, &bg, grid, label);
                    }
                    PrefillCmd::Copy {
                        src,
                        src_off_floats,
                        dst,
                        dst_off_floats,
                        len_floats,
                    } => {
                        Self::encode_copy(
                            enc,
                            src,
                            src_off_floats,
                            dst,
                            dst_off_floats,
                            len_floats,
                        );
                    }
                }
            }
            return;
        }
        // One pass per dispatch-run (a copy ends the run; past 128 tokens a
        // `mul_mat_*`↔other transition ends it too — see the Adreno guard
        // above). Drained into an owned vec first so each segment's pass
        // borrow is scoped to its own block — holding one `ComputePass`
        // across loop iterations unifies its borrow region and conflicts
        // with the copies.
        let split_kinds = chunk_n > Self::ADRENO_PASS_SPLIT_TOKENS;
        let owned: Vec<PrefillCmd> = std::mem::take(cmds);
        let mut i = 0usize;
        let mut seg = 0u32;
        while i < owned.len() {
            // Extend over dispatches of the run's kind (without the guard
            // every dispatch shares one kind, reproducing the old shape).
            let mut j = i;
            if let PrefillCmd::Dispatch { label, .. } = &owned[i] {
                let kind0 = split_kinds && Self::is_adreno_split_label(label);
                j += 1;
                while j < owned.len() {
                    match &owned[j] {
                        PrefillCmd::Dispatch { label, .. }
                            if !split_kinds || Self::is_adreno_split_label(label) == kind0 =>
                        {
                            j += 1;
                        }
                        _ => break,
                    }
                }
            }
            if j > i {
                let mut pass = self.ctx.begin_pass(enc, &format!("{pass_label}s{seg}"));
                seg += 1;
                for d in &owned[i..j] {
                    if let PrefillCmd::Dispatch {
                        pipeline, bg, grid, ..
                    } = d
                    {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                    }
                }
            }
            if j < owned.len() {
                // Either a copy (emit it, consume it) or a kind boundary
                // (owned[j] starts the next run — do NOT consume it).
                if let PrefillCmd::Copy {
                    src,
                    src_off_floats,
                    dst,
                    dst_off_floats,
                    len_floats,
                } = &owned[j]
                {
                    Self::encode_copy(enc, src, *src_off_floats, dst, *dst_off_floats, *len_floats);
                    j += 1;
                }
            }
            i = j;
        }
    }

    /// Submit encoder and wait for GPU to finish.
    fn submit_and_wait(&self, enc: wgpu::CommandEncoder) {
        let host_prof = std::env::var("CERA_GPU_HOST_PROFILE").as_deref() == Ok("1");
        let t_submit = host_prof.then(std::time::Instant::now);
        self.ctx.submit_encoder(enc);
        let t_stall = host_prof.then(std::time::Instant::now);
        self.ctx.device.poll_wait();
        if let (Some(t_submit), Some(t_stall)) = (t_submit, t_stall) {
            eprintln!(
                "[GPU-HOST] submit={:.0}µs stall={:.0}µs",
                t_stall.duration_since(t_submit).as_secs_f64() * 1e6,
                t_stall.elapsed().as_secs_f64() * 1e6,
            );
        }
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

    /// Bind `buffers` at consecutive bindings from 0.
    ///
    /// All three MoE kernels number their bindings that way, with the params
    /// block last, so passing the buffers in the order the shader declares them
    /// is the whole contract.
    fn moe_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
    }

    /// One expert-indexed GEMV: `y[entry] = W[sel_expert[entry]] · x[row(entry)]`
    /// for every entry, in one dispatch.
    ///
    /// `x_by_entry` picks the activation row per entry: `false` for the gate and
    /// up projections, whose input is the token's hidden state and so is shared
    /// by all of that token's slots, and `true` for the down projection, whose
    /// input is the per-slot SwiGLU product.
    #[allow(clippy::too_many_arguments)]
    fn moe_gemv_step(
        &self,
        w: &GpuMoeWeight,
        sel_expert: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n_used: u32,
        n_entries: u32,
        x_by_entry: bool,
        label: &'static str,
    ) -> MoeStep<'_> {
        let params = self.ctx.upload_storage(
            bytemuck::cast_slice(&[
                w.m,
                w.k,
                n_used,
                n_entries,
                w.expert_stride,
                u32::from(x_by_entry),
                0,
                0,
            ]),
            label,
        );
        MoeStep {
            pipeline: &self.pipelines.moe_gemv_q4_0,
            bind_group: self.moe_bind_group(
                &self.pipelines.moe_gemv_q4_0,
                label,
                &[&w.buffer, x, y, sel_expert, &params],
            ),
            // Genuinely two-dimensional: one workgroup per (row, entry), with
            // neither axis foldable into the other. Both are bounded below
            // `MAX_WG` at load (`upload_moe` for the rows, the scratch for the
            // entries), which is what makes this safe to dispatch unfolded.
            workgroups: (w.m, n_entries, 1),
        }
    }

    /// The routed feed-forward block for one MoE layer, over `n` tokens, as the
    /// sequence of dispatches it decomposes into.
    ///
    /// `x` is the `ffn_norm` output, `[n][hidden]` token-major; `out` receives the
    /// block's output at the same layout. `accumulate` picks the convention of the
    /// calling site, which is not decided by the phase: `true` adds into whatever
    /// `out` holds, `false` overwrites it. Decode accumulates straight into the
    /// residual stream, as the dense path's residual add does. Prefill overwrites
    /// its FFN-output scratch and lets the *next* layer's `add_rmsnorm_batch` fold
    /// the residual in, which is exactly what the dense prefill path does with the
    /// same buffer.
    ///
    /// Mirrors `lfm2::forward_moe_ffn` step for step, and like that function it
    /// deliberately does not share code with the dense FFN path: the sequence is
    /// the same but every buffer is indexed by (token, slot) rather than token,
    /// and the projections are slices of a stacked tensor chosen on the device.
    /// Nothing pins the two to each other, so an arithmetic change in the dense
    /// block has to be mirrored here by hand; the oracle suite pins *this* to the
    /// CPU implementation, which is the direction that matters.
    ///
    /// LoRA is absent on purpose. The router and the experts are all LoRA targets
    /// on CPU, and no GPU backend uploads per-expert factors yet, so an adapter
    /// carrying them is refused by `Session::attach_lora_adapters` (via
    /// [`Model::supports_moe_lora`]) rather than silently dropped here.
    fn moe_ffn_steps(
        &self,
        moe: &GpuMoeFfn,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: u32,
        accumulate: bool,
    ) -> Vec<MoeStep<'_>> {
        let scratch = &moe.scratch;
        let hs = self.config.hidden_size as u32;
        let ff = scratch.expert_ff_len;
        let n_used = scratch.n_expert_used;
        let entries = n * n_used;
        // Both callers are internal and already bounded (decode passes 1;
        // prefill asserts its chunk against the same cap the scratch was sized
        // from), so this states the invariant rather than defending a reachable
        // input, which is why it compiles out in release.
        debug_assert!(
            entries <= scratch.max_entries,
            "routed FFN asked for {n} tokens ({entries} entries) against scratch sized for {}; \
             every per-entry buffer below would be indexed past its end",
            scratch.max_entries,
        );

        // Router logits, `[n][n_expert] = X · Wᵀ`. The router is f32, so this is
        // the same NT GEMM the batched LoRA down-projection uses rather than one
        // of the quantized GEMV paths, and it covers decode (n = 1) and prefill
        // with one dispatch shape instead of a per-token loop.
        let router_params = self.ctx.upload_storage(
            bytemuck::cast_slice(&[n, scratch.n_expert, hs, 0u32]),
            "moe_router_params",
        );
        let route_params = self.ctx.upload_storage(
            bytemuck::cast_slice(&[scratch.n_expert, n_used, n, 0u32]),
            "moe_route_params",
        );
        let silu_total = entries * ff;
        let silu_params = self
            .ctx
            .upload_storage(bytemuck::cast_slice(&[silu_total, 0u32]), "moe_silu_params");
        let combine_params = self.ctx.upload_storage(
            bytemuck::cast_slice(&[hs, n_used, n, u32::from(accumulate)]),
            "moe_combine_params",
        );

        vec![
            MoeStep {
                pipeline: &self.pipelines.gemm_f32_nt,
                bind_group: self.moe_bind_group(
                    &self.pipelines.gemm_f32_nt,
                    "moe_router",
                    &[x, &moe.router, &scratch.logits, &router_params],
                ),
                workgroups: crate::backend::wgpu::gemv_row_workgroups(n * scratch.n_expert),
            },
            // Sigmoid + biased top-k → (expert id, unbiased weight) per entry.
            MoeStep {
                pipeline: &self.pipelines.moe_route,
                bind_group: self.moe_bind_group(
                    &self.pipelines.moe_route,
                    "moe_route",
                    &[
                        &scratch.logits,
                        &moe.bias,
                        &scratch.sel_expert,
                        &scratch.sel_weight,
                        &route_params,
                    ],
                ),
                // One threadgroup per token, and `n` is capped by the prefill
                // chunk size well below `MAX_WG`.
                workgroups: (n, 1, 1),
            },
            self.moe_gemv_step(
                &moe.gate,
                &scratch.sel_expert,
                x,
                &scratch.gate,
                n_used,
                entries,
                false,
                "moe_gate",
            ),
            self.moe_gemv_step(
                &moe.up,
                &scratch.sel_expert,
                x,
                &scratch.up,
                n_used,
                entries,
                false,
                "moe_up",
            ),
            // SwiGLU over every entry at once: `gate = silu(gate) * up`.
            MoeStep {
                pipeline: &self.pipelines.silu_mul_inplace,
                bind_group: self.moe_bind_group(
                    &self.pipelines.silu_mul_inplace,
                    "moe_silu",
                    &[&scratch.gate, &scratch.up, &silu_params],
                ),
                workgroups: (silu_total.div_ceil(256), 1, 1),
            },
            // The down projection reads the per-entry SwiGLU product, so its
            // activation row is the entry itself, not the token.
            self.moe_gemv_step(
                &moe.down,
                &scratch.sel_expert,
                &scratch.gate,
                &scratch.z,
                n_used,
                entries,
                true,
                "moe_down",
            ),
            MoeStep {
                pipeline: &self.pipelines.moe_combine,
                bind_group: self.moe_bind_group(
                    &self.pipelines.moe_combine,
                    "moe_combine",
                    &[&scratch.z, &scratch.sel_weight, out, &combine_params],
                ),
                workgroups: (hs.div_ceil(256), n, 1),
            },
        ]
    }

    /// Encode the logit projection.
    ///
    /// One dispatch either way; the variant decides which kernel reads which
    /// form of the weight. See [`LmHead`].
    fn encode_lm_head_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        match &self.lm_head {
            LmHead::Quantized { main, flat } => {
                // GEMV reads the flat twin when present (interleaved stays
                // for the prefill GEMM, which needs GGUF block order).
                let w = flat.as_deref().unwrap_or(main);
                let bg_tmp;
                let bg = match w.cached_bg.as_ref() {
                    Some(bg) => bg,
                    None => {
                        bg_tmp = self.make_gemv_bg(w, input, output);
                        &bg_tmp
                    }
                };
                self.dispatch_gemv_into(pass, w, bg);
            }
            LmHead::F16 { weight, params } => {
                self.encode_gemv_f16_into(pass, weight, params, input, output)
            }
        }
    }

    /// Logit-scale (`scale_f32` over `logits_buf`) bind group. Granite-only
    /// (`logit_scale_params` is `None` elsewhere); factored out because the
    /// decode tail, the dspark path, and the prefill epilogue all built it
    /// inline, and the merged tail passes need it before opening the pass.
    fn logit_scale_bg(&self, params: &wgpu::Buffer) -> wgpu::BindGroup {
        self.ctx
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
            })
    }

    /// `m`/`k` are always the LM head's `vocab_size`/`hidden_size`, so they come
    /// from the config rather than the caller — the params buffer is built from
    /// the same two values at load, and passing them separately invited drift.
    fn encode_gemv_f16(
        &self,
        enc: &mut wgpu::CommandEncoder,
        weight: &wgpu::Buffer,
        params: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        let mut pass = self.ctx.begin_pass(enc, "gemv_f16");
        self.encode_gemv_f16_into(&mut pass, weight, params, input, output);
    }

    fn encode_gemv_f16_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        weight: &wgpu::Buffer,
        params: &wgpu::Buffer,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        let m = self.config.vocab_size as u32;
        let k = self.config.hidden_size as u32;
        // Compare the true binding size: the f16 buffer is u32-addressed, so its
        // `as_entire_binding` size is rounded up to a whole u32 (matching the
        // `upload_f32_as_f16` padding and the tiled round-up). Without this, an
        // adapter whose `max_binding` is not itself a multiple of 4 could take
        // the non-tiled path and then fail binding validation.
        let weight_bytes = (u64::from(m) * u64::from(k) * 2).div_ceil(4) * 4;
        let max_binding = self.ctx.max_storage_buffer_binding_size;
        if weight_bytes > max_binding {
            self.encode_gemv_f16_tiled_into(pass, weight, input, output, m, k);
            return;
        }

        // Pre-allocated params (m=vocab_size, k=hs are constant).
        let params_buf = params;
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
        self.dispatch_into(
            pass,
            &self.pipelines.gemv_f16,
            &bg,
            crate::backend::wgpu::gemv_row_workgroups(groups),
        );
    }

    /// Encode the f16 LM-head GEMV in row tiles for adapters with small
    /// max_storage_buffer_binding_size limits. The tied embedding/output
    /// projection can exceed those limits even though each row slice is legal.
    fn encode_gemv_f16_tiled_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
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
            let Some(params_buf) = self.gemv_tile_params.get(tile_idx) else {
                tracing::error!(
                    "tile_idx {tile_idx} exceeds preallocated LM-head GEMV tile params count"
                );
                break;
            };
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
            self.dispatch_into(
                pass,
                &self.pipelines.gemv_f16,
                &bg,
                crate::backend::wgpu::gemv_row_workgroups(groups),
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
        let mut pass = self.ctx.begin_pass(enc, "rmsnorm");
        self.encode_rmsnorm_into(&mut pass, x, weight);
    }

    fn encode_rmsnorm_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
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
        self.dispatch_into(pass, &self.pipelines.rmsnorm, &bg, (1, 1, 1));
    }

    /// FFN decode chain into an already-open pass. Dense: `rmsnorm_out`
    /// (straight out of `hidden`), gate/up GEMVs, silu_mul, down, residual
    /// add, with LoRA deltas. MoE: `rmsnorm_out` then the routed steps.
    /// Called from both block arms so each layer's block and FFN share one
    /// compute pass instead of two split by a hidden->scratch blit.
    ///
    /// Bind groups built here are dropped at return; the encoder retains what
    /// it recorded, so this is safe (and the common case hits `cached_bg`
    /// without building anything).
    #[allow(clippy::too_many_arguments)]
    fn encode_ffn_decode_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        lw: &GpuLayerWeights,
        lora: &Option<Arc<WgpuLoraAdapter>>,
        layer: usize,
        hs32: u32,
    ) {
        let norm_bg = lw.ffn_norm_bg.as_ref().unwrap();
        if let GpuFfn::Moe(moe) = &lw.ffn {
            self.dispatch_into(pass, &self.pipelines.rmsnorm_out, norm_bg, (1, 1, 1));
            let steps = self.moe_ffn_steps(moe, &self.ffn_input_buf, &self.hidden_buf, 1, true);
            steps.iter().for_each(|s| {
                self.dispatch_into(pass, s.pipeline, &s.bind_group, s.workgroups);
            });
            return;
        }
        let GpuFfn::Dense(dense) = &lw.ffn else {
            unreachable!()
        };
        let gate_bg_tmp;
        let gate_bg = match dense.gate.cached_bg.as_ref() {
            Some(bg) => bg,
            None => {
                gate_bg_tmp = self.make_gemv_bg(&dense.gate, &self.ffn_input_buf, &self.gate_buf);
                &gate_bg_tmp
            }
        };
        let up_bg_tmp;
        let up_bg = match dense.up.cached_bg.as_ref() {
            Some(bg) => bg,
            None => {
                up_bg_tmp = self.make_gemv_bg(&dense.up, &self.ffn_input_buf, &self.up_buf);
                &up_bg_tmp
            }
        };
        let silu_bg = lw.silu_bg.as_ref().unwrap();
        let down_bg_tmp;
        let down_bg = match dense.down.cached_bg.as_ref() {
            Some(bg) => bg,
            None => {
                down_bg_tmp = self.make_gemv_bg(&dense.down, &self.gate_buf, &self.out_buf);
                &down_bg_tmp
            }
        };
        let add_bg = lw.ffn_add_bg.as_ref().unwrap();

        // LoRA gate/up deltas on the raw projections (before silu_mul), and
        // the ffn-down delta into the post-residual hidden state (input is
        // the silu_mul result in `gate_buf`). `residual_mult` is folded into
        // ffn-down's B at upload.
        let gate_lora = Self::lora_target(lora.as_ref(), layer, LoraTarget::FfnGate);
        let up_lora = Self::lora_target(lora.as_ref(), layer, LoraTarget::FfnUp);
        let down_lora = Self::lora_target(lora.as_ref(), layer, LoraTarget::FfnDown);
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

        // rmsnorm
        self.dispatch_into(pass, &self.pipelines.rmsnorm_out, norm_bg, (1, 1, 1));
        // gate + up GEMVs
        self.dispatch_gemv_into(pass, &dense.gate, gate_bg);
        self.dispatch_gemv_into(pass, &dense.up, up_bg);
        // LoRA gate/up deltas on the raw projections, before silu_mul.
        if let Some((t, (bg_a, bg_b))) = gate_lora_bgs.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
        if let Some((t, (bg_a, bg_b))) = up_lora_bgs.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
        // silu_mul
        self.dispatch_into(
            pass,
            &self.pipelines.silu_mul_inplace,
            silu_bg,
            ((dense.gate.tensor.shape[0] as u32).div_ceil(256), 1, 1),
        );
        // down GEMV
        self.dispatch_gemv_into(pass, &dense.down, down_bg);
        // residual add
        self.dispatch_into(
            pass,
            &self.pipelines.scaled_add_inplace,
            add_bg,
            (hs32.div_ceil(256), 1, 1),
        );
        // LoRA ffn-down delta into the post-residual hidden state.
        if let Some((t, (bg_a, bg_b))) = down_lora_bgs.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
    }

    /// Decode attn pre-chain into an already-open pass: `rmsnorm_out`, QKV
    /// GEMVs, LoRA deltas, QKV bias, QK-norm, RoPE. Shared by the merged
    /// `layer_attn` pass and the TurboQuant `attn_pre` split.
    fn encode_attn_pre_into(&self, pass: &mut wgpu::ComputePass<'_>, pre: &AttnPreDecode) {
        let lw = pre.lw;
        self.dispatch_into(pass, &self.pipelines.rmsnorm_out, pre.norm_bg, (1, 1, 1));
        // Unfused Q/K/V: the fused QKV kernel predates the WS3
        // subgroup upgrade and measured ~2% slower on Adreno 830,
        // so it was removed; these use the subgroup GEMV twins.
        self.dispatch_gemv_into(pass, pre.q_w, pre.q_bg);
        self.dispatch_gemv_into(pass, pre.k_w, pre.k_bg);
        self.dispatch_gemv_into(pass, pre.v_w, pre.v_bg);
        // LoRA Q/K/V deltas on the raw projections (before bias/norm/rope).
        if let Some((t, (bg_a, bg_b))) = pre.q_lora.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
        if let Some((t, (bg_a, bg_b))) = pre.k_lora.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
        if let Some((t, (bg_a, bg_b))) = pre.v_lora.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
        // QKV bias (Qwen2): add right after the projections.
        if let Some(bg) = lw.qb_bg.as_ref() {
            self.dispatch_into(
                pass,
                &self.pipelines.add_inplace,
                bg,
                (pre.q_dim.div_ceil(256), 1, 1),
            );
        }
        if let Some(bg) = lw.kb_bg.as_ref() {
            self.dispatch_into(
                pass,
                &self.pipelines.add_inplace,
                bg,
                (pre.kv_dim.div_ceil(256), 1, 1),
            );
        }
        if let Some(bg) = lw.vb_bg.as_ref() {
            self.dispatch_into(
                pass,
                &self.pipelines.add_inplace,
                bg,
                (pre.kv_dim.div_ceil(256), 1, 1),
            );
        }
        // QK-norm (Qwen3): per-head RMSNorm before RoPE.
        if let Some(bg) = lw.qn_bg.as_ref() {
            self.dispatch_into(
                pass,
                &self.pipelines.per_head_rmsnorm,
                bg,
                (pre.n_heads, 1, 1),
            );
        }
        if let Some(bg) = lw.kn_bg.as_ref() {
            self.dispatch_into(
                pass,
                &self.pipelines.per_head_rmsnorm,
                bg,
                (pre.n_kv_heads, 1, 1),
            );
        }
        self.dispatch_into(
            pass,
            &self.pipelines.rope,
            lw.rope_bg.as_ref().unwrap(),
            (pre.max_pairs.div_ceil(256), 1, 1),
        );
    }

    /// Decode attn out_proj tail into an already-open pass: O GEMV, residual
    /// add, LoRA output delta. Shared by the merged `layer_attn` pass and the
    /// TurboQuant split (whose attention runs in its own encode passes).
    fn encode_attn_out_into(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        out_w: &GpuWeight,
        out_bg: &wgpu::BindGroup,
        add_bg: &wgpu::BindGroup,
        o_lora_bgs: &Option<(&WgpuLoraTarget, (wgpu::BindGroup, wgpu::BindGroup))>,
        hs32: u32,
    ) {
        self.dispatch_gemv_into(pass, out_w, out_bg);
        self.dispatch_into(
            pass,
            &self.pipelines.scaled_add_inplace,
            add_bg,
            (hs32.div_ceil(256), 1, 1),
        );
        if let Some((t, (bg_a, bg_b))) = o_lora_bgs.as_ref() {
            self.dispatch_lora_into(pass, t, bg_a, bg_b);
        }
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
    /// layer: (1) re-rotate the retained K cells by `R(-shift)` into f32
    /// `kv_shift_scratch` via the `kv_shift` kernel, (2) pack the rotated K
    /// back into the cache at the `n_keep` offset via `kv_append`, (3) ferry
    /// V through the same scratch to its new offset (V isn't RoPE'd, but its
    /// source/destination ranges overlap the same way K's do, so it can't
    /// move in place either).
    ///
    /// The V ferry reuses `copy_buffer_to_buffer` (packed bytes move as
    /// bytes); wgpu's automatic usage tracking inserts the WAR/RAW barriers
    /// between the compute passes and the copies (and across layers that
    /// share the one scratch buffer), so the single command encoder stays
    /// correct without manual synchronization.
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
            // Uncompressed cache only. `Session::can_shift` keeps a compressed
            // cache out — it needs both `supports_kv_shift` (false while
            // `self.tq` is set) and `!state.is_compressed()`, the latter also
            // covering a request this backend downgraded but the state-side
            // cache compressed anyway. `shift_kv` re-asserts the same
            // condition as a backstop.
            let (k_cache, v_cache) = self.f16_kv()[layer_idx]
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
            // Pack the rotated K back into the cache at the new n_keep-aligned
            // offset. A blit cannot convert f32 scratch -> packed halves, so
            // this is a `kv_append` dispatch (one thread per word).
            let n_floats = (retained * kv_dim) as u32;
            let append_params: [u32; 4] = [(n_keep * kv_dim / 2) as u32, n_floats, 0, 0];
            let append_params_buf = self.ctx.upload_storage(
                bytemuck::cast_slice(&append_params),
                "kv_shift_append_params",
            );
            let append_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("kv_shift_append"),
                    layout: &self.pipelines.kv_append.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.kv_shift_scratch.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: k_cache.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: append_params_buf.as_entire_binding(),
                        },
                    ],
                });
            self.encode(
                &mut enc,
                &self.pipelines.kv_append,
                &append_bg,
                ((n_floats / 2).div_ceil(256), 1, 1),
                "kv_shift_append",
            );

            // ── V: ferry through scratch to the new offset (no rotation) ──
            // Packed bytes move as bytes: counts/offsets are in u32 words
            // (words × 4 = bytes, same arithmetic `encode_copy` applies to
            // its float counts).
            let n_words = (retained * kv_dim / 2) as u64;
            Self::encode_copy(
                &mut enc,
                v_cache,
                ((n_keep + shift) * kv_dim / 2) as u64,
                &self.kv_shift_scratch,
                0,
                n_words,
            );
            Self::encode_copy(
                &mut enc,
                &self.kv_shift_scratch,
                0,
                v_cache,
                (n_keep * kv_dim / 2) as u64,
                n_words,
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
                    assert_kv_dim_packable(kv_dim, head_dim);
                    let bytes = kv_slab_bytes(max_seq_len, kv_dim);
                    let k = self.ctx.create_storage_rw(bytes, &format!("hs.l{i}.k"));
                    let v = self.ctx.create_storage_rw(bytes, &format!("hs.l{i}.v"));
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

    /// The packed-f16 generation KV caches, allocated on first use.
    ///
    /// Never reached while TurboQuant is active: every KV write and attention
    /// read on that path goes through `self.tq`, so the `OnceLock` stays empty
    /// and the slabs are never allocated. The assert makes a mis-gated call
    /// site fail loudly instead of quietly allocating the memory compression was
    /// meant to save (and then reading a cache nothing writes).
    fn f16_kv(&self) -> &Vec<Option<(wgpu::Buffer, wgpu::Buffer)>> {
        // A real `assert!`, not `debug_assert!`: release is precisely the build
        // where a mis-gated call site's `max_seq_len x kv_dim` allocation
        // matters, and this is not a hot path.
        assert!(
            self.tq.get().is_none(),
            "packed-f16 KV cache requested while TurboQuant is active"
        );
        self.gpu_state.kv_caches.get_or_init(|| {
            let cfg = &self.config;
            let head_dim = cfg.head_dim;
            let max_seq_len = self.gpu_state.max_seq_len;
            let mut kv = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                if cfg.block_types[i] == BlockType::Attention {
                    let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                    assert_kv_dim_packable(kv_dim, head_dim);
                    let bytes = kv_slab_bytes(max_seq_len, kv_dim);
                    kv.push(Some((
                        self.ctx.create_storage_rw(bytes, &format!("l{i}.k_cache")),
                        self.ctx.create_storage_rw(bytes, &format!("l{i}.v_cache")),
                    )));
                } else {
                    kv.push(None);
                }
            }
            kv
        })
    }

    /// The attention KV cache for layer `i` — the hidden-states scratch cache
    /// when [`Self::hidden_states`] is running (`use_hs_scratch`), else the
    /// generation cache. Panics on a conv layer (no KV).
    ///
    /// f32 only. `hidden_states` deliberately runs uncompressed (it is a
    /// one-shot full-precision pass on its own scratch caches), which is why the
    /// scratch arm is reachable even under TurboQuant.
    #[inline]
    fn active_kv(&self, i: usize) -> &(wgpu::Buffer, wgpu::Buffer) {
        let caches = if self.use_hs_scratch.load(Ordering::Relaxed) {
            &self
                .hs_scratch
                .get()
                .expect("hs_scratch built before use_hs_scratch is set")
                .kv
        } else {
            self.f16_kv()
        };
        caches[i].as_ref().unwrap()
    }

    /// Prefix-cache namespace for this model instance.
    ///
    /// The KV-compression mode is part of it, not just the model path: a
    /// compressed snapshot and an f32 one have different layouts, and the disk
    /// tier is shared by every session over the same model. Without the mode in
    /// the namespace, a TurboQuant session's entry permanently shadows the f32
    /// entry for the same prefix — the lookup-time mode filter turns the longest
    /// match into a miss and never falls back to a shorter compatible one, so the
    /// f32 session stays cold on *every* subsequent run, not just once. Same trick
    /// the `"wgpu:"` / `"cpu:"` / `"metal:"` prefixes already use to keep backends
    /// apart.
    ///
    /// Called from `configure_cache` and again from `configure_kv_compression`
    /// (which rebuilds the cache) because the engine configures the cache before
    /// the session configures compression.
    fn cache_namespace(&self) -> String {
        // Not yet configured behaves as f32 (the empty tag) — the mode-setting path
        // rebuilds the cache, so an early `configure_cache` can't leave a stale tag.
        let tag = self.kv_cache_tag.get().map(String::as_str).unwrap_or("");
        format!("wgpu:{tag}{}", self.model_id)
    }

    /// The GPU-resident TurboQuant cache, when the session configured one and
    /// this pass is a generation pass. `hidden_states` runs uncompressed on its
    /// own scratch caches, so it always sees `None` here.
    #[inline]
    fn tq_cache(&self) -> Option<&TqGpuCache> {
        if self.use_hs_scratch.load(Ordering::Relaxed) {
            return None;
        }
        self.tq.get()
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
        self.forward_inner_compute_tail(tokens, pos, state, DecodeTail::Logits(TailArgmax::None));
    }

    /// Run one decode step from a caller-supplied hidden vector rather than a
    /// token id, leaving logits in `logits_buf`.
    ///
    /// This is the vision path. An image becomes a run of hidden-size
    /// embeddings from the mmproj's projector, with no token id that could
    /// produce them, so the embedding-lookup step has nothing to look up.
    /// Everything after that step is identical, which is why this only
    /// re-seeds `hidden_buf` instead of duplicating the layer dispatch.
    fn forward_inner_compute_from_embedding(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) {
        self.forward_inner_compute_tail_seeded(
            HiddenSeed::Embedding(embedding),
            pos,
            state,
            DecodeTail::Logits(TailArgmax::None),
        );
    }

    /// Append `n_tokens` embedding frames to the KV cache, reading nothing
    /// back.
    ///
    /// This is the entry point the browser needs. `Model::forward_from_embedding`
    /// and [`Model::forward_prefill_from_embeddings`] both end in a blocking
    /// `download_f32`, which on wasm waits forever: the buffer-map callback it
    /// waits for is delivered by the JS event loop, and the thread calling it is
    /// the one that would have to return for that loop to run. Appending an
    /// image wants the KV cache updated and nothing else, so the readback is not
    /// merely unaffordable there, it is unnecessary.
    ///
    /// Logits for the last frame are left in `logits_buf` for a caller that does
    /// want them to fetch on its own terms (blocking natively, or via
    /// `begin_download` on wasm).
    pub fn seed_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        self.seed_embeddings_locked(embeddings, n_tokens, start_pos, state);
    }

    /// [`Self::seed_embeddings`] with `infer_lock` and the LoRA guard already
    /// held by the caller, so the two entry points cannot disagree about how a
    /// frame run is seeded.
    fn seed_embeddings_locked(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) {
        let hidden_size = self.config.hidden_size;
        assert!(n_tokens > 0, "seed_embeddings requires at least one frame");
        assert_eq!(
            embeddings.len(),
            n_tokens * hidden_size,
            "embeddings.len() ({}) != n_tokens ({}) * hidden_size ({})",
            embeddings.len(),
            n_tokens,
            hidden_size
        );

        // Same fresh-prefill reset as the token path and the Metal twin: at
        // position zero the GPU-resident counter and the conv rolling buffers
        // still hold whatever a previous generate() left, and the embeddings
        // path has no prefix-cache restore to overwrite them.
        if start_pos == 0 {
            self.gpu_state.seq_len.store(0, Ordering::Relaxed);
            self.zero_conv_buffers_locked();
        }

        for i in 0..n_tokens {
            let frame = &embeddings[i * hidden_size..(i + 1) * hidden_size];
            // `state.seq_len`, not `start_pos + i`: the compute tail advances it
            // per frame, and matching `forward_from_embedding` here is what
            // keeps a spliced image landing where the caller's state says.
            let pos = state.seq_len;
            self.forward_inner_compute_from_embedding(frame, pos, state);
        }
    }

    /// As [`Self::forward_inner_compute`], but the caller chooses where the tail
    /// stops (see [`DecodeTail`]) and, when it stops at logits, whether the
    /// greedy argmax — and its readback copy — ride in the *same* encoder as the
    /// output projection. See [`TailArgmax`] for why those two are separable.
    ///
    /// Worth the extra parameter. The argmax used to get its own encoder and its
    /// own `submit_and_wait`, which cost ~1.3 ms per token against the kernel's
    /// own ~0.13 ms of GPU time: a submit costs a GPU round trip no matter how
    /// little work it carries, so the second one was paying full stall price for
    /// a single-workgroup dispatch. Folding it in leaves one stall per decode
    /// step instead of two.
    fn forward_inner_compute_tail(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
        tail: DecodeTail,
    ) -> Option<wgpu::CommandEncoder> {
        assert_eq!(tokens.len(), 1, "GPU forward expects single token");
        self.forward_inner_compute_tail_seeded(HiddenSeed::Token(tokens[0]), pos, state, tail)
    }

    /// As [`Self::forward_inner_compute_tail`], but the initial hidden state
    /// comes from a [`HiddenSeed`] rather than always from an embedding-table
    /// lookup. Splitting on the seed keeps one copy of the layer dispatch:
    /// the token and image paths differ only in how `hidden_buf` is filled.
    fn forward_inner_compute_tail_seeded(
        &self,
        seed: HiddenSeed<'_>,
        pos: usize,
        state: &mut InferenceState,
        tail: DecodeTail,
    ) -> Option<wgpu::CommandEncoder> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let hs32 = hs as u32;
        let t_entry = std::time::Instant::now();

        self.ctx.reset_profiler();

        // Bounds check: KV cache capacity
        assert!(
            self.gpu_state.seq_len.load(Ordering::Relaxed) < self.gpu_state.max_seq_len,
            "GPU seq_len {} exceeds max_seq_len {}",
            self.gpu_state.seq_len.load(Ordering::Relaxed),
            self.gpu_state.max_seq_len,
        );

        // 1. Seed the hidden state (4KB upload per step). A token reads its
        //    row out of the mmap'd embedding table (dequantized on the fly);
        //    an image embedding is already a hidden-size vector and uploads
        //    directly.
        match seed {
            HiddenSeed::Token(token) => {
                let mut row = vec![0.0f32; hs];
                self.gpu_state
                    .embedding
                    .dequantize_row(token as usize, &mut row);
                if self.scalars.embedding != 1.0 {
                    for v in row.iter_mut() {
                        *v *= self.scalars.embedding;
                    }
                }
                self.ctx
                    .queue
                    .write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(&row));
            }
            HiddenSeed::Embedding(embedding) => {
                assert_eq!(
                    embedding.len(),
                    hs,
                    "GPU forward_from_embedding expects one hidden-size vector"
                );
                self.ctx
                    .queue
                    .write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(embedding));
            }
        }

        // Active LoRA adapter (cheap Arc clone; `None` on the base-model path).
        // Read once so every hook in this forward shares one lock acquisition.
        let lora = self
            .active_lora
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        let head_dim = cfg.head_dim as u32;
        let n_heads = cfg.n_heads as u32;
        let n_kv_heads = cfg
            .kv_heads_per_layer
            .iter()
            .copied()
            .find(|&h| h > 0)
            .unwrap_or(cfg.n_kv_heads) as u32;
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

        let seq_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
        let scale = self
            .scalars
            .attn
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
        let kv_dim = n_kv_heads * head_dim;
        let attn_params: [u32; 8] = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            (seq_len + 1) as u32,
            scale.to_bits(),
            0,
            0,
        ];
        self.ctx
            .queue
            .write_buffer(&self.attn_params, 0, bytemuck::cast_slice(&attn_params));
        // `kv_append` slot for this token: row `seq_len` of every layer's
        // cache slab (kv_dim is uniform across attn layers — same value as
        // `attn_params[3]`). Offset in u32 words (2 halves each), count in
        // floats; kv_dim is even (asserted at cache alloc).
        let kv_append_params: [u32; 4] = [(seq_len as u32) * kv_dim / 2, kv_dim, 0, 0];
        self.ctx.queue.write_buffer(
            &self.kv_append_params,
            0,
            bytemuck::cast_slice(&kv_append_params),
        );

        // Stage the TurboQuant shader params for every layer in one write, ahead
        // of the per-layer encoders below. One decode row, appended at the
        // current seq_len; Q and the attention output are both `q_dim`-strided.
        if let Some(tq) = self.tq_cache() {
            tq.write_params(
                &self.ctx,
                cfg,
                1,
                self.gpu_state.seq_len.load(Ordering::Relaxed),
                scale,
            );
        }

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
        //
        // NOTE (WS3 audit): the design above is STALE — this function now builds
        // ONE encoder for all layers and issues ONE submit per token (see the
        // `submit_and_wait` at the tail branches). Re-measured on Adreno 840,
        // LFM2.5-VL-450M Q4_0, `CERA_GPU_HOST_PROFILE=1`: encode ~3.4 ms, submit
        // ~0.5 ms, GPU stall ~8.5 ms, map ~0.04 ms per decode token. The encode
        // phase is real again (bind-group caching landed but 41 passes/token of
        // recording remain), so the overlap argument above may apply once more.
        //
        // NOTE (GPU gap work): re-tested on Adreno 830, 2.6B Q4_0 — a two-chunk
        // split (fire layers 0-14, encode 15-29 during their execution)
        // regressed decode 28.0 -> 25.2 tok/s. `finish()` carries ~1 ms of
        // fixed per-encoder cost (no pass-count scaling: 62 profile-mode
        // passes finish no slower than 31), so the second submit's fixed
        // cost eats the overlap saving now that encode is lean (cached bind
        // groups). One submit stands; do not re-split without re-measuring.
        let host_prof_pre = std::env::var("CERA_GPU_HOST_PROFILE").as_deref() == Ok("1");
        let t_pre = std::time::Instant::now();
        if host_prof_pre {
            eprintln!(
                "[GPU-HOST] pre={:.0}µs",
                t_pre.duration_since(t_entry).as_secs_f64() * 1e6,
            );
        }
        let mut enc = self.new_encoder();
        for i in 0..cfg.n_layers {
            let lw = &self.layers[i];

            if cfg.block_types[i] == BlockType::GatedConv {
                let kernel_size = cfg.conv_kernel_size.unwrap_or(3) as u32;
                let _d_conv = kernel_size - 1;
                let norm_bg = lw.attn_norm_bg.as_ref().unwrap();
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

                let conv_fused_bg = lw.conv_fused_bg.as_ref().unwrap();
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
                let add_bg = lw.conv_add_bg.as_ref().unwrap();

                // The whole conv layer in ONE compute pass: rmsnorm (out of
                // `hidden`, straight into `normed`), in_proj, the fused conv,
                // out_proj + residual add, then the FFN chain. These were two
                // passes (`conv` + `ffn`) split by a hidden->scratch blit — a
                // pass boundary is not free (2.65x, measured M1 Max) and the
                // blit is gone now that both norms read `hidden` directly.
                // WebGPU orders dispatches within a pass and makes each one's
                // writes visible to the next. Profile runs split mixer from
                // FFN for attribution; production keeps the merged pass.
                let profiling = self.ctx.profiling();
                {
                    let mut pass = self.ctx.begin_pass(
                        &mut enc,
                        if profiling {
                            "conv_mixer"
                        } else {
                            "layer_conv"
                        },
                    );
                    self.dispatch_into(&mut pass, &self.pipelines.rmsnorm_out, norm_bg, (1, 1, 1));
                    self.dispatch_gemv_into(&mut pass, in_w, in_bg);
                    if let Some((t, (bg_a, bg_b))) = &in_lora_bgs {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.conv1d_fused,
                        conv_fused_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                    self.dispatch_gemv_into(&mut pass, out_w, out_bg);
                    if let Some((t, (bg_a, bg_b))) = &out_lora_bgs {
                        self.dispatch_lora_into(&mut pass, t, bg_a, bg_b);
                    }
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.add_inplace,
                        add_bg,
                        (hs32.div_ceil(256), 1, 1),
                    );
                    // FFN — same pass (dense and MoE both), except on profile
                    // runs, where it gets its own pass below.
                    if !profiling {
                        self.encode_ffn_decode_into(&mut pass, lw, &lora, i, hs32);
                    }
                }
                if profiling {
                    let mut pass = self.ctx.begin_pass(&mut enc, "conv_ffn");
                    self.encode_ffn_decode_into(&mut pass, lw, &lora, i, hs32);
                }
            } else {
                // Attention block — one `layer_attn` pass (uncompressed KV),
                // or the `attn_pre`/`attn_post` split under TurboQuant.
                let q_dim = n_heads * head_dim;

                let norm_bg = lw.attn_norm_bg.as_ref().unwrap();
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

                // Pre-chain inputs shared by both shapes below.
                let pre = AttnPreDecode {
                    lw,
                    norm_bg,
                    q_w,
                    q_bg,
                    k_w,
                    k_bg,
                    v_w,
                    v_bg,
                    q_lora: &q_lora_bgs,
                    k_lora: &k_lora_bgs,
                    v_lora: &v_lora_bgs,
                    q_dim,
                    kv_dim,
                    n_heads,
                    n_kv_heads,
                    max_pairs,
                };
                // out_proj + add — both shapes. Bind groups built before the
                // pass opens.
                let out_w = lw.attn_output.as_ref().unwrap();
                let out_bg_tmp;
                let out_bg = match out_w.cached_bg.as_ref() {
                    Some(b) => b,
                    None => {
                        out_bg_tmp = self.make_gemv_bg(out_w, &self.attn_out_buf, &self.out_buf);
                        &out_bg_tmp
                    }
                };
                let add_bg = lw.attn_out_add_bg.as_ref().unwrap();
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

                if self.tq_cache().is_some() {
                    // TurboQuant split: attention runs in its own encode
                    // passes, so pre and post stay split.
                    {
                        let mut pass = self.ctx.begin_pass(&mut enc, "attn_pre");
                        self.encode_attn_pre_into(&mut pass, &pre);
                    }
                    // Compressed path: the two f32 memcpys become encode
                    // dispatches, and attention reads the packed cache. Params for
                    // every layer were staged once before this encoder (see
                    // `forward_inner_compute`).
                    let tq = self.tq_cache().unwrap();
                    tq.encode_kv(&self.ctx, &mut enc, i, &self.k_buf, &self.v_buf, 1);
                    tq.rotate_queries(&self.ctx, &mut enc, i, &self.q_buf, 1, n_heads as usize);
                    tq.attention(
                        &self.ctx,
                        &mut enc,
                        i,
                        &self.attn_out_buf,
                        1,
                        n_heads as usize,
                    );
                    let profiling = self.ctx.profiling();
                    {
                        let mut pass = self.ctx.begin_pass(
                            &mut enc,
                            if profiling {
                                "attn_post_out"
                            } else {
                                "attn_post"
                            },
                        );
                        self.encode_attn_out_into(
                            &mut pass,
                            out_w,
                            out_bg,
                            add_bg,
                            &o_lora_bgs,
                            hs32,
                        );
                        if !profiling {
                            self.encode_ffn_decode_into(&mut pass, lw, &lora, i, hs32);
                        }
                    }
                    if profiling {
                        let mut pass = self.ctx.begin_pass(&mut enc, "attn_post_ffn");
                        self.encode_ffn_decode_into(&mut pass, lw, &lora, i, hs32);
                    }
                } else {
                    // Whole attn layer in ONE pass: the KV cache write is a
                    // `kv_append` dispatch now, not an encoder-level blit, so
                    // nothing splits pre from post. WebGPU orders dispatches
                    // within a pass; flash_attention reads the row appended
                    // two dispatches earlier.
                    let flash_bg_tmp;
                    let flash_bg = if self.use_hs_scratch.load(Ordering::Relaxed) {
                        let (k_buf, v_buf) = self.active_kv(i);
                        flash_bg_tmp =
                            self.ctx
                                .device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("flash_attention_hs_bg"),
                                    layout: &self
                                        .pipelines
                                        .flash_attention
                                        .get_bind_group_layout(0),
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: self.q_buf.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: k_buf.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 2,
                                            resource: v_buf.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 3,
                                            resource: self.attn_out_buf.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 4,
                                            resource: self.attn_params.as_entire_binding(),
                                        },
                                    ],
                                });
                        &flash_bg_tmp
                    } else {
                        lw.attn_bg.as_ref().unwrap()
                    };
                    let k_app_tmp;
                    let v_app_tmp;
                    let (k_app_bg, v_app_bg) = if self.use_hs_scratch.load(Ordering::Relaxed) {
                        let (k_cache, v_cache) = self.active_kv(i);
                        k_app_tmp = self.make_kv_append_bg(&self.k_buf, k_cache);
                        v_app_tmp = self.make_kv_append_bg(&self.v_buf, v_cache);
                        (&k_app_tmp, &v_app_tmp)
                    } else {
                        (
                            lw.k_append_bg.as_ref().unwrap(),
                            lw.v_append_bg.as_ref().unwrap(),
                        )
                    };
                    let profiling = self.ctx.profiling();
                    {
                        let mut pass = self.ctx.begin_pass(
                            &mut enc,
                            if profiling { "attn_core" } else { "layer_attn" },
                        );
                        self.encode_attn_pre_into(&mut pass, &pre);
                        // One thread per packed word: kv_dim/2 threads.
                        let append_grid = ((kv_dim / 2).div_ceil(256), 1, 1);
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.kv_append,
                            k_app_bg,
                            append_grid,
                        );
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.kv_append,
                            v_app_bg,
                            append_grid,
                        );
                        self.dispatch_into(
                            &mut pass,
                            &self.pipelines.flash_attention,
                            flash_bg,
                            (n_heads, 1, 1),
                        );
                        self.encode_attn_out_into(
                            &mut pass,
                            out_w,
                            out_bg,
                            add_bg,
                            &o_lora_bgs,
                            hs32,
                        );
                        if !profiling {
                            self.encode_ffn_decode_into(&mut pass, lw, &lora, i, hs32);
                        }
                    }
                    if profiling {
                        let mut pass = self.ctx.begin_pass(&mut enc, "attn_ffn");
                        self.encode_ffn_decode_into(&mut pass, lw, &lora, i, hs32);
                    }
                }
            }

            if self
                .loop_norm_interval
                .is_some_and(|n_phys| (i + 1) % n_phys == 0)
                && (i + 1) < cfg.n_layers
            {
                self.encode_rmsnorm(
                    &mut enc,
                    &self.hidden_buf,
                    &self.output_norm,
                    hs32,
                    cfg.rms_norm_eps,
                );
            }
        }

        // 3. Output norm + projection. Untied models project through
        // `output.weight`; tied models reuse the embedding table.
        //
        // The norm runs for both tails: it is the last step of the CPU model's
        // `run_layers`, so it is inside what `forward_embedding` returns, not
        // part of the projection that `DecodeTail::Hidden` is declining. Only
        // the projection and the argmax below are conditional.
        //
        // The Logits tail runs norm + projection + optional scale + optional
        // argmax in ONE pass: a sequential RAW chain, and WebGPU orders
        // dispatches within a pass (same guarantee the merged `conv` pass
        // relies on). Profile runs split the three stages for attribution;
        // production saves the per-pass host overhead on every token.
        let norm_only = matches!(tail, DecodeTail::Hidden | DecodeTail::HiddenUnsubmitted);
        if norm_only {
            self.encode_rmsnorm(
                &mut enc,
                &self.hidden_buf,
                &self.output_norm,
                hs32,
                cfg.rms_norm_eps,
            );
        }
        match tail {
            DecodeTail::Hidden => {
                self.submit_and_wait(enc);
                self.gpu_state.seq_len.fetch_add(1, Ordering::Relaxed);
                state.seq_len += 1;
                self.ctx.finish_profiler();
                None
            }
            DecodeTail::HiddenUnsubmitted => {
                self.gpu_state.seq_len.fetch_add(1, Ordering::Relaxed);
                state.seq_len += 1;
                self.ctx.finish_profiler();
                Some(enc)
            }
            DecodeTail::Logits(argmax) | DecodeTail::LogitsUnsubmitted(argmax) => {
                // Granite divides the logits by `logits_scaling` (identity
                // elsewhere). Bind group built before the pass opens.
                let scale_bg = self
                    .logit_scale_params
                    .as_ref()
                    .map(|params| self.logit_scale_bg(params));
                let profiling = self.ctx.profiling();
                {
                    let mut pass = self
                        .ctx
                        .begin_pass(&mut enc, if profiling { "tail_norm" } else { "tail" });
                    self.encode_rmsnorm_into(&mut pass, &self.hidden_buf, &self.output_norm);
                    if !profiling {
                        self.encode_lm_head_into(&mut pass, &self.hidden_buf, &self.logits_buf);
                        if let Some(scale_bg) = scale_bg.as_ref() {
                            self.dispatch_into(
                                &mut pass,
                                &self.pipelines.scale_f32,
                                scale_bg,
                                ((cfg.vocab_size as u32).div_ceil(256), 1, 1),
                            );
                        }
                        if argmax != TailArgmax::None {
                            self.encode_argmax_into(&mut pass);
                        }
                    }
                }
                if profiling {
                    {
                        let mut pass = self.ctx.begin_pass(&mut enc, "tail_lm_head");
                        self.encode_lm_head_into(&mut pass, &self.hidden_buf, &self.logits_buf);
                    }
                    {
                        let mut pass = self.ctx.begin_pass(&mut enc, "tail_sample");
                        if let Some(scale_bg) = scale_bg.as_ref() {
                            self.dispatch_into(
                                &mut pass,
                                &self.pipelines.scale_f32,
                                scale_bg,
                                ((cfg.vocab_size as u32).div_ceil(256), 1, 1),
                            );
                        }
                        if argmax != TailArgmax::None {
                            self.encode_argmax_into(&mut pass);
                        }
                    }
                }
                if argmax == TailArgmax::DispatchAndStage {
                    // Stage the 4-byte result in this same submission.
                    enc.copy_buffer_to_buffer(
                        &self.argmax_out_buf,
                        0,
                        &self.argmax_readback_buf,
                        0,
                        4,
                    );
                }
                self.gpu_state.seq_len.fetch_add(1, Ordering::Relaxed);
                state.seq_len += 1;
                self.ctx.finish_profiler();
                if matches!(tail, DecodeTail::LogitsUnsubmitted(_)) {
                    Some(enc)
                } else {
                    self.submit_and_wait(enc);
                    None
                }
            }
        }
    }

    /// Encode the argmax dispatch into an open pass. Shared by the sync and
    /// async greedy paths so the kernel / bind-group / dispatch live in one
    /// place.
    fn encode_argmax_into(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.pipelines.argmax_f32);
        pass.set_bind_group(0, &self.argmax_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Greedy single-token forward: runs the same kernels as
    /// [`forward_inner`] but replaces the vocab-sized logits download
    /// with a 4-byte argmax readback. Cuts per-token PCIe/USB-C
    /// readback from `vocab_size * 4` bytes to `4` bytes — the
    /// wasm-async-friendly path, since a 4-byte map_async still
    /// blocks the JS event loop briefly but doesn't transfer megabytes.
    fn forward_greedy_inner(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        // The argmax rides along in the output projection's encoder, so a decode
        // step is one submit-and-stall, not two.
        let host_prof = std::env::var("CERA_GPU_HOST_PROFILE").as_deref() == Ok("1");
        let t_compute = std::time::Instant::now();
        self.forward_inner_compute_tail(
            tokens,
            pos,
            state,
            DecodeTail::Logits(TailArgmax::DispatchAndStage),
        );
        let t_map = std::time::Instant::now();
        let argmax = self.ctx.read_mapped_u32(&self.argmax_readback_buf, 1);
        let Some(&tok) = argmax.first() else {
            // The map failure is recorded first-wins in the readback slot
            // (and loud on stderr); return a dummy the session never
            // samples — its post-greedy drain surfaces the record as a
            // typed `Backend` error instead of unwinding out of the core
            // library. All production `forward_greedy` callers drain.
            return 0;
        };
        if host_prof {
            eprintln!(
                "[GPU-HOST] compute={:.0}µs map={:.0}µs",
                t_map.duration_since(t_compute).as_secs_f64() * 1e6,
                t_map.elapsed().as_secs_f64() * 1e6,
            );
        }
        tok
    }

    /// Async-path prefill step: run the forward and update the KV cache
    /// *without* the argmax + readback. Used for every prompt token except the
    /// last, whose argmax seeds decoding — so an N-token prompt does one GPU→CPU
    /// round-trip instead of N. Synchronous: only the readback needs to be
    /// async. Pins `gpu_state.seq_len` to `pos` so the RoPE position (driven by
    /// `pos`) and the KV-write slot (driven by `gpu_state.seq_len`) cannot
    /// drift (mirrors `forward_prefill`).
    pub fn forward_prefill_step(&self, token: u32, pos: usize, state: &mut InferenceState) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
        if pos == 0 {
            self.zero_conv_buffers_locked();
        }
        self.forward_inner_compute(&[token], pos, state);
    }

    /// Async (wasm/WebGPU) greedy decode step. Runs the full forward + argmax
    /// on the GPU, then reads back the single argmax token id without blocking
    /// — the wasm-compatible analog of `Self::forward_greedy_inner`.
    ///
    /// The argmax rides along in the output projection's encoder, as on the
    /// blocking path; that encoder's `submit_and_wait` reduces to a plain submit
    /// here, because `device.poll(Maintain::Wait)` is a no-op on the WebGPU
    /// backend (the browser owns the queue). The readback is ordered after it on
    /// the same queue. The GPU compute + submit run under `infer_lock` (serialising
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
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            // Keep the KV-write slot in lockstep with the RoPE position.
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            let enc = self
                .forward_inner_compute_tail(
                    &[token],
                    pos,
                    state,
                    DecodeTail::LogitsUnsubmitted(TailArgmax::Dispatch),
                )
                .ok_or_else(|| anyhow::anyhow!("LogitsUnsubmitted returned None"))?;

            Ok::<_, anyhow::Error>(self.ctx.begin_download_with_encoder(
                enc,
                &self.argmax_out_buf,
                std::mem::size_of::<u32>() as u64,
            ))
        }?;

        let bytes = pending.recv().await?;
        if bytes.len() < 4 {
            anyhow::bail!(
                "GPU argmax readback buffer truncated (expected 4 bytes, got {})",
                bytes.len()
            );
        }
        let token = bytemuck::pod_read_unaligned::<u32>(&bytes[..4]);
        Ok(token)
    }

    /// Async (wasm/WebGPU) decode step returning the full logits row, for
    /// callers that sample rather than take the argmax.
    ///
    /// [`Self::forward_greedy_async`]'s sibling, and the reason sampling can
    /// work on this backend at all: that one reduces the step to a token id on
    /// the GPU, so temperature, top-k and top-p have nothing left to act on by
    /// the time anything reaches the host. This reads the row back instead and
    /// lets the caller's [`crate::sampler::Sampler`] do the work, which keeps
    /// one sampler implementation across every backend rather than growing a
    /// second one in WGSL.
    ///
    /// Costs a vocab-sized readback per token where the greedy path costs four
    /// bytes, so callers should keep using that one when the request is greedy
    /// (`temperature <= 0` or `top_k == 1`). Same locking discipline as the
    /// greedy path: compute under `infer_lock`, release before the `.await`.
    pub async fn forward_logits_async(
        &self,
        token: u32,
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let vocab = self.config.vocab_size;
        let expected_bytes = vocab * std::mem::size_of::<f32>();
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            // Keep the KV-write slot in lockstep with the RoPE position.
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            let enc = self
                .forward_inner_compute_tail(
                    &[token],
                    pos,
                    state,
                    DecodeTail::LogitsUnsubmitted(TailArgmax::None),
                )
                .ok_or_else(|| anyhow::anyhow!("LogitsUnsubmitted returned None"))?;

            Ok::<_, anyhow::Error>(self.ctx.begin_download_with_encoder(
                enc,
                &self.logits_buf,
                expected_bytes as u64,
            ))
        }?;

        let bytes = pending.recv().await?;
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU logits readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        // Copy into an aligned Vec rather than casting the byte slice: the
        // readback's buffer carries no f32 alignment guarantee.
        let mut out = vec![0f32; vocab];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..expected_bytes]);
        Ok(out)
    }

    /// Async decode step returning the hidden state embedding on GPU, for vocoder conditioning.
    pub async fn forward_embedding_async(
        &self,
        token: u32,
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let expected_bytes = hidden_size * std::mem::size_of::<f32>();
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            let enc = self
                .forward_inner_compute_tail(&[token], pos, state, DecodeTail::HiddenUnsubmitted)
                .ok_or_else(|| anyhow::anyhow!("HiddenUnsubmitted returned None"))?;
            Ok::<_, anyhow::Error>(self.ctx.begin_download_with_encoder(
                enc,
                &self.hidden_buf,
                expected_bytes as u64,
            ))
        }?;
        let bytes = pending.recv().await?;
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU hidden readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        let mut out = vec![0f32; hidden_size];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..expected_bytes]);
        Ok(out)
    }

    /// Async decode step returning the hidden state embedding when seeded by an audio embedding.
    pub async fn forward_hidden_from_embedding_async(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let expected_bytes = hidden_size * std::mem::size_of::<f32>();
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            let enc = self
                .forward_inner_compute_tail_seeded(
                    HiddenSeed::Embedding(embedding),
                    pos,
                    state,
                    DecodeTail::HiddenUnsubmitted,
                )
                .ok_or_else(|| anyhow::anyhow!("HiddenUnsubmitted returned None"))?;
            Ok::<_, anyhow::Error>(self.ctx.begin_download_with_encoder(
                enc,
                &self.hidden_buf,
                expected_bytes as u64,
            ))
        }?;
        let bytes = pending.recv().await?;
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU hidden readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        let mut out = vec![0f32; hidden_size];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..expected_bytes]);
        Ok(out)
    }

    /// Decode step computing hidden state for a token and keeping it in `hidden_buf` on the GPU with no host readback.
    pub fn forward_hidden_gpu(
        &self,
        token: u32,
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<&wgpu::Buffer> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
        self.forward_inner_compute_tail(&[token], pos, state, DecodeTail::Hidden);
        Ok(&self.hidden_buf)
    }

    /// Decode step computing hidden state and keeping it in `hidden_buf` on the GPU with no host readback.
    pub fn forward_hidden_from_embedding_gpu(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<&wgpu::Buffer> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
        self.forward_inner_compute_tail_seeded(
            HiddenSeed::Embedding(embedding),
            pos,
            state,
            DecodeTail::Hidden,
        );
        Ok(&self.hidden_buf)
    }

    /// Access the GPU hidden state buffer directly for zero-copy downstream pipeline stages.
    pub fn hidden_buffer(&self) -> &wgpu::Buffer {
        &self.hidden_buf
    }

    /// Async decode step returning logits when seeded by an audio embedding.
    pub async fn forward_logits_from_embedding_async(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let expected_bytes = vocab_size * std::mem::size_of::<f32>();
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            let enc = self
                .forward_inner_compute_tail_seeded(
                    HiddenSeed::Embedding(embedding),
                    pos,
                    state,
                    DecodeTail::LogitsUnsubmitted(TailArgmax::None),
                )
                .ok_or_else(|| anyhow::anyhow!("LogitsUnsubmitted returned None"))?;
            Ok::<_, anyhow::Error>(self.ctx.begin_download_with_encoder(
                enc,
                &self.logits_buf,
                expected_bytes as u64,
            ))
        }?;
        let bytes = pending.recv().await?;
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU logits readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        let mut out = vec![0f32; vocab_size];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..expected_bytes]);
        Ok(out)
    }

    /// Async decode step returning the argmax token directly when seeded by an audio embedding.
    pub async fn forward_greedy_from_embedding_async(
        &self,
        embedding: &[f32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Result<u32> {
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
            let enc = self
                .forward_inner_compute_tail_seeded(
                    HiddenSeed::Embedding(embedding),
                    pos,
                    state,
                    DecodeTail::LogitsUnsubmitted(TailArgmax::Dispatch),
                )
                .ok_or_else(|| anyhow::anyhow!("LogitsUnsubmitted returned None"))?;
            Ok::<_, anyhow::Error>(self.ctx.begin_download_with_encoder(
                enc,
                &self.argmax_out_buf,
                std::mem::size_of::<u32>() as u64,
            ))
        }?;
        let bytes = pending.recv().await?;
        if bytes.len() < 4 {
            anyhow::bail!(
                "GPU argmax readback buffer truncated (expected 4 bytes, got {})",
                bytes.len()
            );
        }
        let token = bytemuck::pod_read_unaligned::<u32>(&bytes[..4]);
        Ok(token)
    }

    /// Projects a hidden state vector to the vocab-sized logits using the GPU-resident LM head
    /// and performs a parallel GPU argmax reduction, returning the top token ID with minimal readback.
    pub async fn lm_head_argmax_async(&self, hidden: &[f32]) -> Result<u32> {
        let hs = self.config.hidden_size;
        anyhow::ensure!(
            hidden.len() == hs,
            "dspark_hidden length ({}) != hidden_size ({})",
            hidden.len(),
            hs
        );
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            self.ctx
                .queue
                .write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(hidden));
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dspark_lm_head_argmax"),
                });
            // One merged tail pass (projection + optional scale + argmax),
            // same as the decode tail above.
            let scale_bg = self
                .logit_scale_params
                .as_ref()
                .map(|params| self.logit_scale_bg(params));
            {
                let mut pass = self.ctx.begin_pass(&mut enc, "tail");
                self.encode_lm_head_into(&mut pass, &self.hidden_buf, &self.logits_buf);
                if let Some(scale_bg) = scale_bg.as_ref() {
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.scale_f32,
                        scale_bg,
                        ((self.config.vocab_size as u32).div_ceil(256), 1, 1),
                    );
                }
                self.encode_argmax_into(&mut pass);
            }
            self.ctx.begin_download_with_encoder(
                enc,
                &self.argmax_out_buf,
                std::mem::size_of::<u32>() as u64,
            )
        };
        let bytes = pending.recv().await?;
        if bytes.len() < std::mem::size_of::<u32>() {
            anyhow::bail!("short read ({}) for argmax token readback", bytes.len());
        }
        let token = bytemuck::pod_read_unaligned::<u32>(&bytes[..4]);
        Ok(token)
    }

    /// Adapter name and backend of the underlying [`GpuContext`], for
    /// surfacing which GPU/backend the model is actually running on (e.g. in
    /// the wasm `WebGpuSession.adapter` getter).
    pub fn gpu_info(&self) -> (&str, &str) {
        (&self.ctx.adapter_name, &self.ctx.backend)
    }

    /// The KV-cache mode this model actually resolved to, as a human-readable
    /// label: `"turboquant(seed=N)"` or `"uncompressed"`.
    ///
    /// This reports the *effective* mode, not the requested one, and that
    /// distinction is the whole point: `configure_kv_compression` downgrades a
    /// request it can't serve (single-sided TurboQuant, or a `head_dim` the
    /// kernels reject) to uncompressed KV with only a `tracing::warn!`. A caller
    /// that can't see the log — a browser via `cera-wasm`, notably — otherwise
    /// has no way to tell a compressed session from a silently uncompressed one.
    ///
    /// Reads `"uncompressed"` before `configure_kv_compression` has run, which is
    /// the correct label for the f32 default.
    pub fn kv_mode_label(&self) -> String {
        describe_kv_mode(self.kv_mode.get().unwrap_or(&None))
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
//   * Every matmul weight must have a batched GEMM kernel. The five quantized
//     dtypes (Q4_0, Q8_0, Q4KM, Q5KM, Q6K) run `mul_mat_reg_tile` with a per-dtype
//     shmem dequant loader; F32 runs the same kernel with the direct-read
//     `INIT_SRC0_SHMEM_FLOAT` loader. `upload_weight` dequantizes every other
//     dtype (Q4_1, Q2_K, F16/BF16 sources) to F32, so the on-GPU weight dtype set
//     is closed to those six and `unbatchable_matmul_weight` returns `None` for
//     every real model, so the per-token bail is now unreachable, kept only as a
//     defensive backstop.
//
// Per-dispatch overhead note: each `encode_*` helper builds a fresh
// `wgpu::BindGroup` and uploads a small params buffer per call. The CPU
// cost is ~1 % of total prefill time at the workloads measured in PR #157;
// promoting the params buffers to model-resident state and caching the
// bind groups for fixed prefill scratch buffers is a clean follow-up
// optimization. Kept simple here so the refactor is reviewable.
//
// Prefill attention is an online-softmax (FlashAttention) kernel
// (`attention_prefill.wgsl`) that never materializes the scores matrix — only a
// TILE-sized tile lives in workgroup memory — so no storage binding scales with
// `n_queries × n_heads × max_seq`. The only seq_len-scaling binding left is the
// contiguous K/V cache (`max_seq × kv_dim`); contexts long enough that *it*
// overflows the storage-binding limit need key-tiled / paged KV, a follow-up.

impl GpuLfm2Model {
    /// The first matmul weight that has no batched prefill kernel, as
    /// `(layer, tensor name, dtype)` — or `None` when every weight has one, which
    /// is the precondition for `forward_prefill_batched_locked` to take the batched
    /// path. The five quantized dtypes (Q4_0, Q8_0, Q4KM, Q5KM, Q6K) and F32 all
    /// run the same register-tiled kernel, differing only in the shmem loader.
    ///
    /// In practice this now always returns `None`: `upload_weight` stores every
    /// dtype without a quantized reg-tile loader dequantized to F32, and F32 has
    /// one too, so the on-GPU weight dtype set is closed to the six admitted
    /// here. It is kept as a cheap defensive backstop (and to name the offender
    /// loudly if that invariant is ever broken) rather than a live fallback
    /// trigger.
    ///
    /// Returns the *offender*, not a bare `bool`, because one unsupported tensor
    /// silently drops the whole prompt onto the per-token loop — ~340x the submits
    /// (measured: 8728 vs 25 on a 512-token prefill). A `false` that names nothing
    /// is how a `Q4_K_M` model — which is *not* uniformly Q4_K; it carries a
    /// handful of Q6_K tensors — sat on the slow path unnoticed. Cheap
    /// `O(n_layers)` walk; called once per `forward_prefill`.
    fn unbatchable_matmul_weight(&self) -> Option<(usize, &'static str, DType)> {
        for (li, lw) in self.layers.iter().enumerate() {
            // Only the FFN differs by layer kind. A routed layer's experts never
            // reach the reg-tile GEMM (`upload_moe` admits Q4_0 only, and the
            // routed prefill path runs the same expert GEMV decode does), so
            // they contribute nothing here; the rest of the layer still does,
            // which is why this narrows the three FFN entries rather than
            // skipping the layer.
            let (gate, up, down) = match &lw.ffn {
                GpuFfn::Dense(d) => (Some(&d.gate), Some(&d.up), Some(&d.down)),
                GpuFfn::Moe(_) => (None, None, None),
            };
            let weights: [(&'static str, Option<&GpuWeight>); 9] = [
                ("ffn_gate", gate),
                ("ffn_up", up),
                ("ffn_down", down),
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
                if !matches!(
                    dt,
                    DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q5KM | DType::Q6K | DType::F32
                ) {
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
    fn encode_rmsnorm_batch<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
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
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.rmsnorm_batch,
            bg,
            (n, 1, 1),
            "rmsnorm_batch",
        );
    }

    /// Encode `scaled_add_inplace`: a[i] += scale * b[i] for i in 0..total.
    fn encode_scaled_add_inplace_batch<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        total: u32,
        scale: f32,
        label: &'static str,
    ) {
        let params: [u32; 2] = [total, scale.to_bits()];
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.scaled_add_inplace.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.scaled_add_inplace,
            bg,
            (total.div_ceil(256), 1, 1),
            label,
        );
    }

    /// Encode `add_rmsnorm_batch`: src[t,i] += residual[t,i]; dst[t,i] =
    /// src[t,i] * inv_rms(src[t]) * w[i]. One pass; src is read-write.
    #[allow(clippy::too_many_arguments)]
    fn encode_add_rmsnorm_batch<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
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
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.add_rmsnorm_batch,
            bg,
            (n, 1, 1),
            "add_rmsnorm_batch",
        );
    }

    /// Encode batched 2D matmul: y = weight * x.
    /// Batched prefill supports the five quantized reg-tile dtypes (Q4_0, Q8_0,
    /// Q4KM, Q5KM, Q6K) plus F32. `upload_weight` stores every other dtype
    /// (F16/BF16/F32 sources, Q4_1, Q2_K, ...) dequantized to F32, so the F32
    /// arm is the fallback that keeps any single unsupported-dtype tensor from
    /// dropping the whole model onto the per-token loop. `x_stride`/`y_stride`
    /// are measured in f32 elements between consecutive token vectors.
    #[allow(clippy::too_many_arguments)] // tile geometry + strides; splitting hurts clarity
    /// Streaming fp16 Q4_0 GEMM (`gemm_stream_q4_0`). Same contract as
    /// `encode_mul_mat_reg_tile` for the Q4_0 case: `y[n, m] = x[n, k] @
    /// w[m, k]^T` with token-major strides. Two passes into the same
    /// encoder (no extra submit): `transpose_cast_f16` first rewrites `x`
    /// into the B16 scratch as f16 k-major, then the GEMM reads the
    /// resident repack (`stream_q`/`stream_d`) plus B16. Grid
    /// `(ceil(m/256), ceil(n/32))` — each fiber covers 1 row x 32 columns
    /// in 256-thread workgroups; columns past `n` idle inside the fiber, so
    /// callers route n < 32 to the reg-tile kernel instead. Requires packed
    /// B (`x_stride == k`); strided-B callers stay on reg-tile.
    fn encode_gemm_stream_q4_0<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
        w: &GpuWeight,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: u32,
        k: u32,
        y_stride: u32,
        all_logits: bool,
    ) {
        let m = w.tensor.shape[0] as u32;
        // K-slice-64 twin when k cooperates (all LFM2/Qwen shapes): same
        // interface and grid, +28-33% on Adreno 830, bit-exact. Selection is
        // static per call (k is fixed per weight, the hatch process-static),
        // so the bind-group cache below stays paired.
        let k64 = k.is_multiple_of(64) && use_gemm_k64();
        let (gemm, gemm_label) = if k64 {
            (
                self.pipelines
                    .gemm_stream_q4_0_k64
                    .as_ref()
                    .expect("k64 streaming GEMM dispatched without a pipeline"),
                "gemm_stream_q4_0_k64",
            )
        } else {
            (
                self.pipelines
                    .gemm_stream_q4_0
                    .as_ref()
                    .expect("streaming GEMM dispatched without a pipeline"),
                "gemm_stream_q4_0",
            )
        };
        let transpose = self
            .pipelines
            .transpose_cast_f16
            .as_ref()
            .expect("streaming GEMM dispatched without a transpose pipeline");
        let b16 = self
            .stream_b16_buf
            .as_ref()
            .expect("streaming GEMM dispatched without B16 scratch");
        let (sq, sd) = match (&w.stream_q, &w.stream_d) {
            (Some(q), Some(d)) => (q, d),
            _ => panic!("streaming GEMM dispatched without resident (q, d) buffers"),
        };
        debug_assert_eq!(k % 32, 0);
        // Resident (q, d) counts (u32s): nq = m*(k/8),
        // nd = m*ceil((k/32)/2). The upload covers exactly these.
        let nq = m.checked_mul(k / 8).expect("repack q count exceeds u32");
        let nd = m
            .checked_mul((k / 32).div_ceil(2))
            .expect("repack d count exceeds u32");
        debug_assert!(u64::from(nq) * 4 <= sq.size());
        debug_assert!(u64::from(nd) * 4 <= sd.size());
        let n_pad = n.next_multiple_of(32);
        // Params contents are refreshed every call (pooled buffers are
        // stable, but each prefill rewrites them). The bind groups below
        // bind the pooled buffer *objects*, which is what makes them
        // cacheable across prefills.
        let t_params: [u32; 4] = [n, n_pad, k, 0];
        let t_buf = self.next_prefill_params(bytemuck::cast_slice(&t_params));
        let params: [u32; 5] = [m, k, n, n_pad, y_stride];
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));

        // Bind-group cache hit: same (n, all_logits) as the cached prefill,
        // so this call's slot holds the identical pair. Miss (first
        // prefill, or n/all_logits changed): create, store, and use. The
        // cursor advances in call order — reset per prefill next to the
        // params pool — so slot `i` always pairs with the same pooled
        // buffers.
        let (t_bg, bg) = {
            let mut cache = self
                .stream_gemm_bg_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (key_n, key_al, cursor, slots) = &mut *cache;
            if (*key_n, *key_al) != (n, all_logits) {
                *key_n = n;
                *key_al = all_logits;
                *cursor = 0;
                slots.clear();
            }
            let idx = *cursor;
            *cursor += 1;
            if slots.len() <= idx {
                slots.resize_with(idx + 1, || None);
            }
            if slots[idx].is_none() {
                let fresh_t = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &transpose.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: x.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b16.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: t_buf.as_entire_binding(),
                            },
                        ],
                    });
                let fresh_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &gemm.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: sq.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: sd.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: b16.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: y.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: p_buf.as_entire_binding(),
                            },
                        ],
                    });
                slots[idx] = Some((fresh_t, fresh_bg));
            }
            slots[idx].as_ref().unwrap().clone()
        };

        // Both recorded in order; the emitter groups them into the layer's
        // shared pass (transpose-then-GEMM RAW on B16 needs no boundary —
        // same guarantee as the decode `ffn` span). Profile mode keeps one
        // pass per dispatch, labeled as below.
        Self::push_prefill_dispatch(
            cmds,
            transpose,
            t_bg,
            (n_pad / 32, k.div_ceil(32), 1),
            "transpose_cast_f16",
        );
        Self::push_prefill_dispatch(
            cmds,
            gemm,
            bg,
            (m.div_ceil(256), n.div_ceil(32), 1),
            gemm_label,
        );
    }

    #[allow(clippy::too_many_arguments)] // internal prefill plumbing; all 10 load-bearing
    fn encode_mul_mat_reg_tile<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
        w: &GpuWeight,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: u32,
        k: u32,
        x_stride: u32,
        y_stride: u32,
        all_logits: bool,
    ) {
        // Streaming fp16 fast path (Q4_0, n >= 32, k % 32 == 0, packed B):
        // ~10x the reg-tile kernel on Adreno. Anything it declines — other
        // dtypes, short rows, strided B, missing repack — falls through to
        // reg-tile below.
        if w.tensor.dtype == DType::Q4_0
            && n >= 32
            && k.is_multiple_of(32)
            && x_stride == k
            && self.pipelines.gemm_stream_q4_0.is_some()
            && self.pipelines.transpose_cast_f16.is_some()
            && w.stream_q.is_some()
            && w.stream_d.is_some()
        {
            self.encode_gemm_stream_q4_0(cmds, w, x, y, n, k, y_stride, all_logits);
            return;
        }
        debug_assert!(
            matches!(
                w.tensor.dtype,
                DType::Q4_0 | DType::Q8_0 | DType::Q4KM | DType::Q5KM | DType::Q6K | DType::F32
            ),
            "encode_mul_mat_reg_tile only supports Q4_0/Q8_0/Q4KM/Q5KM/Q6K/F32 weights"
        );
        let m = w.tensor.shape[0] as u32;
        // Every dtype shares one register-tiled geometry — only the shmem dequant
        // loader differs, and the kernel is dtype-agnostic past it. Resident
        // stream weights have no raw upload, so they ride the (q, d) loader
        // variant with a 5-binding group; raw Q4_0 keeps the 4-binding one.
        let (pipeline, label): (&wgpu::ComputePipeline, &str) = match w.tensor.dtype {
            DType::Q4_0 if w.resident_stream => (
                self.pipelines
                    .mul_mat_reg_tile_q4_0_stream
                    .as_ref()
                    .expect("resident-stream weight without a reg-tile stream pipeline"),
                "mul_mat_tile_stream",
            ),
            DType::Q4_0 => (&self.pipelines.mul_mat_reg_tile_q4_0, "mul_mat_tile"),
            DType::Q8_0 => (&self.pipelines.mul_mat_reg_tile_q8_0, "mul_mat_q8_0"),
            DType::Q4KM => (&self.pipelines.mul_mat_reg_tile_q4_k, "mul_mat_q4k"),
            DType::Q5KM => (&self.pipelines.mul_mat_reg_tile_q5_k, "mul_mat_q5k"),
            DType::Q6K => (&self.pipelines.mul_mat_reg_tile_q6_k, "mul_mat_q6k"),
            DType::F32 => (&self.pipelines.mul_mat_reg_tile_f32, "mul_mat_f32"),
            // Unreachable in practice: the batched path is only entered when
            // `unbatchable_matmul_weight()` returned `None`, i.e. every weight is
            // one of the six admitted dtypes. The debug_assert above documents
            // the same precondition; this arm is the release-mode backstop.
            _ => unreachable!("batched prefill only supports Q4_0/Q8_0/Q4KM/Q5KM/Q6K/F32"),
        };
        let wg_m = m.div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M);
        let wg_n = n.div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N);

        // Matches `mul_mat_reg_tile`'s 5-field `MulMatParams`. This was 6 words while
        // the Q8_0 arm still dispatched `gemm_q8_0`, whose `params: array<u32, 6>` is
        // fixed-size — the buffer had to be sized to the union of both layouts. Every
        // dtype now goes through the register-tiled kernel, so the union is gone.
        let params: [u32; 5] = [m, k, n, x_stride, y_stride];
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));

        let w_q = w.tensor.buffer.as_entire_binding();
        let w_d = w.stream_d.as_ref().map(|d| d.as_entire_binding());
        let x_binding = x.as_entire_binding();
        let y_binding = y.as_entire_binding();
        let p_binding = p_buf.as_entire_binding();
        // Resident: (q, d, x, y, params). Raw: (w, x, y, params).
        let entries = if w.resident_stream {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_q,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_d.expect("resident-stream weight without stream_d"),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_binding,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y_binding,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p_binding,
                },
            ]
        } else {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_q,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_binding,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_binding,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_binding,
                },
            ]
        };
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });

        Self::push_prefill_dispatch(cmds, pipeline, bg, (wg_m, wg_n, 1), label);
    }

    /// Encode `bias_add`: broadcast a `dim`-length bias across all `n` token
    /// rows of `buf` (`buf[t*dim + j] += bias[j]`). Qwen2 QKV bias; the batch
    /// path packs Q/K/V densely (stride == dim), so the shader's `i % dim`
    /// indexing lands on the right element.
    fn encode_bias_add_batch<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
        buf: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        n: u32,
        dim: u32,
    ) {
        let total = n * dim;
        let params: [u32; 2] = [total, dim];
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.bias_add,
            bg,
            (total.div_ceil(256), 1, 1),
            "bias_add_batch",
        );
    }

    /// Encode `qk_norm_rope_batch`: in-place rmsnorm + RoPE on Q (n × n_heads
    /// × head_dim) and K (n × n_kv_heads × head_dim) at positions
    /// `start_pos + token_idx`.
    #[allow(clippy::too_many_arguments)]
    fn encode_qk_norm_rope_batch<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
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
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.qk_norm_rope_batch,
            bg,
            (tg_count, 1, 1),
            "qk_norm_rope_batch",
        );
    }

    /// Encode `conv1d_fused_batch`. One thread per channel walks all n
    /// tokens sequentially; rolling-buffer state is in `rbuffer` and is
    /// updated in place.
    #[allow(clippy::too_many_arguments)]
    fn encode_conv1d_fused_batch<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
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
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.conv1d_fused_batch,
            bg,
            (groups, 1, 1),
            "conv1d_fused_batch",
        );
    }

    /// Encode a batched `kv_append` (f32 rows -> packed-f16 cache): the
    /// prefill KV write. Offsets/counts mirror the decode path (`off_words`
    /// in u32 words, `n_floats` in floats, always even), but params come
    /// from the pooled prefill pool — the shared decode `kv_append_params`
    /// holds one token's slot, not this chunk's.
    fn encode_kv_append_prefill<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
        src: &wgpu::Buffer,
        cache: &wgpu::Buffer,
        off_words: u32,
        n_floats: u32,
    ) {
        debug_assert!(
            n_floats.is_multiple_of(2),
            "kv_append needs an even float count, got {n_floats}"
        );
        let params: [u32; 4] = [off_words, n_floats, 0, 0];
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("kv_append_prefill"),
                layout: &self.pipelines.kv_append.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.kv_append,
            bg,
            ((n_floats / 2).div_ceil(256), 1, 1),
            "kv_append",
        );
    }

    /// Encode `attention_prefill` (batched FlashAttention). Reads Q from
    /// `q_batch`, K/V from the model's KV caches, writes per-(token, head) output
    /// to `out_batch`. Online-softmax over a tiled pass — no scores scratch slab.
    #[allow(clippy::too_many_arguments)]
    fn encode_attention_prefill<'a>(
        &'a self,
        cmds: &mut Vec<PrefillCmd<'a>>,
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
        // No queries → nothing to dispatch (a 0-workgroup dispatch is a validation
        // error on some backends).
        if n == 0 {
            return;
        }

        // attention_prefill.wgsl sizes `q_shared` / `acc` at MAX_HEAD_DIM (128)
        // f32, so head_dim must fit — the same contract as the decode flash kernel.
        assert!(
            head_dim <= 128,
            "wgpu attention_prefill supports head_dim <= 128 (q_shared/acc are \
             sized 128); got {head_dim}"
        );
        // GQA invariants the kernel assumes (group_size = n_heads / n_kv_heads,
        // kv_head = head / group_size, a head_dim-wide slice at kv_head * head_dim
        // within each kv_dim-strided KV row). Fail fast on a malformed config.
        assert!(
            n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads),
            "wgpu attention_prefill requires n_kv_heads > 0 and n_heads divisible \
             by n_kv_heads; got n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        );
        assert_eq!(
            kv_dim,
            n_kv_heads * head_dim,
            "wgpu attention_prefill requires kv_dim == n_kv_heads * head_dim; got \
             kv_dim={kv_dim}, n_kv_heads={n_kv_heads}, head_dim={head_dim}"
        );

        // The online-softmax kernel never materializes the scores matrix, so the
        // only seq_len-scaling binding is the contiguous K/V cache
        // (`max_seq × kv_dim`). Guard it once: if this fires the context itself is
        // too long for a contiguous KV binding — the remaining case that needs
        // key-tiled / paged attention. Saturating multiply so an overflow pins to
        // u64::MAX and trips the assert instead of wrapping to a too-short range.
        let kv_live_floats = u64::from(max_seq).saturating_mul(u64::from(kv_dim));
        assert_packed_binding_fits(
            kv_live_floats,
            self.ctx.max_storage_buffer_binding_size,
            "attention_prefill live KV",
        );

        // Single dispatch over the whole query batch; `q_base = 0`. The kernel
        // still honors `q_base`, so a caller could sub-batch queries, but with the
        // scores slab gone there is no binding-size reason to. 8 queries share
        // each workgroup (and each K/V tile stream); params[11] is the
        // authoritative batch size edge workgroups mask against.
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
            0, // q_base
            n, // n_sub (authoritative; single dispatch so == batch size)
        ];
        let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
                        resource: packed_f16_binding(k_cache, kv_live_floats),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: packed_f16_binding(v_cache, kv_live_floats),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_batch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
        Self::push_prefill_dispatch(
            cmds,
            &self.pipelines.attention_prefill,
            bg,
            (n_heads, n.div_ceil(8), 1),
            "attention_prefill",
        );
    }

    /// Batched prefill — single-pass over `n` tokens for all layers, then
    /// final output norm + LM head on the last token only.
    ///
    /// Preconditions (caller-enforced):
    ///   * `1 <= tokens.len() <= MAX_PREFILL_TOKENS`.
    ///   * Every matmul weight has a batched kernel
    ///     (`unbatchable_matmul_weight() == None`).
    ///   * Caller already holds `infer_lock`.
    ///
    /// `start_pos` may be non-zero: `Model::forward_prefill_chunked` splits a
    /// prompt into ubatch-sized chunks and this runs once per chunk with an
    /// advancing position. Conv rolling state and KV writes carry across chunks
    /// naturally. Do NOT re-add a `start_pos == 0` gate on the caller side — it
    /// silently dropped every chunk after the first onto the per-token loop.
    ///
    /// Mirrors `MetalLfm2Model::prefill_layers_and_logits`
    /// (metal_lfm2.rs:2906); the Metal version is the canonical
    /// reference for the dispatch order + buffer assignment.
    fn encode_prefill_batched_locked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        _state: &mut InferenceState,
        all_logits: bool,
        need_logits: bool,
    ) -> wgpu::CommandEncoder {
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
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        // Stage the TurboQuant shader params for every layer in one write, before
        // the encoders below are submitted. `n` rows appended at `start_pos`; Q
        // lives in `prefill_proj_buf` and the attention output in
        // `prefill_normed_buf`, both `q_dim`-strided.
        if let Some(tq) = self.tq_cache() {
            let scale = self
                .scalars
                .attn
                .unwrap_or_else(|| 1.0 / (cfg.head_dim as f32).sqrt());
            tq.write_params(&self.ctx, cfg, n, start_pos, scale);
        }

        // ─── Stage embeddings into prefill_batch_buf ──────────────────────
        // CPU-side gather + one queue.write_buffer. Rows come from the
        // mmap'd table and dequantize on the fly (one reusable row buffer,
        // ~µs per token); the embedding multiplier folds in here, input
        // only — the projection copy stays unscaled.
        let mut staged: Vec<f32> = Vec::with_capacity(n * hs);
        let mut row = vec![0.0f32; hs];
        let emb_scale = self.scalars.embedding;
        for &t in tokens {
            self.gpu_state
                .embedding
                .dequantize_row(t as usize, &mut row);
            if emb_scale != 1.0 {
                for v in row.iter_mut() {
                    *v *= emb_scale;
                }
            }
            staged.extend_from_slice(&row);
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
                .unwrap_or_else(|e| e.into_inner())
                .1 = 0;
        }
        self.prefill_params_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1 = 0;
        // Reset the stream-GEMM bind-group cache cursor alongside the params
        // pool: both are handed out in call order, so slot `i` always pairs
        // with the same pooled params buffers.
        self.stream_gemm_bg_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .2 = 0;

        let mut enc = self.new_encoder();
        // Recorded prefill commands, emitted per layer (see `PrefillCmd`).
        let mut cmds: Vec<PrefillCmd> = Vec::new();
        let n_u = n as u32;
        let hs_u = hs as u32;
        let is_u = is as u32;

        for layer in 0..cfg.n_layers {
            let lw = &self.layers[layer];

            // ─── Phase 1: rmsnorm (or fused add_rmsnorm with prev FFN
            //              residual) → prefill_normed_buf ─────────────────
            if layer > 0 {
                let is_loop_boundary = self
                    .loop_norm_interval
                    .is_some_and(|n_phys| layer % n_phys == 0);
                if is_loop_boundary {
                    // Loop boundary: add previous layer FFN down residual to prefill_batch_buf,
                    // apply loop_norm using output_norm into prefill_up_buf, copy back,
                    // then apply current layer attn_norm into prefill_normed_buf.
                    self.encode_scaled_add_inplace_batch(
                        &mut cmds,
                        &self.prefill_batch_buf,
                        &self.prefill_up_buf,
                        n_u * hs_u,
                        self.scalars.residual,
                        "loop_add",
                    );
                    self.encode_rmsnorm_batch(
                        &mut cmds,
                        &self.prefill_batch_buf,
                        &self.prefill_up_buf,
                        &self.output_norm,
                        n_u,
                        hs_u,
                    );
                    cmds.push(PrefillCmd::Copy {
                        src: &self.prefill_up_buf,
                        src_off_floats: 0,
                        dst: &self.prefill_batch_buf,
                        dst_off_floats: 0,
                        len_floats: (n * hs) as u64,
                    });
                    self.encode_rmsnorm_batch(
                        &mut cmds,
                        &self.prefill_batch_buf,
                        &self.prefill_normed_buf,
                        &lw.attn_norm,
                        n_u,
                        hs_u,
                    );
                } else {
                    // Fuse: batch_buf += prev_layer_ffn_down (`prefill_up_buf`),
                    // then rmsnorm into `prefill_normed_buf`.
                    //
                    // Metal aliases dst === residual on `prefill_normed_buf`;
                    // wgpu 24's binding-aliasing validator rejects that
                    // pattern (binding 1 read_write + binding 4 read on the
                    // same buffer in one dispatch). Route FFN down to
                    // `prefill_up_buf` so dst and residual stay distinct.
                    self.encode_add_rmsnorm_batch(
                        &mut cmds,
                        &self.prefill_batch_buf,
                        &self.prefill_normed_buf,
                        &lw.attn_norm,
                        &self.prefill_up_buf,
                        n_u,
                        hs_u,
                    );
                }
            } else {
                self.encode_rmsnorm_batch(
                    &mut cmds,
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
                    &mut cmds,
                    w_in,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    3 * hs_u,
                    all_logits,
                );
                // LoRA conv in_proj — must run before the fused conv1d overwrites
                // `prefill_normed_buf` (which still holds the rmsnorm output that
                // feeds the LoRA `A`).
                self.encode_lora_hook_batched(
                    &mut cmds,
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
                    &mut cmds,
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
                    &mut cmds,
                    w_out,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    hs_u,
                    all_logits,
                );
                // LoRA conv out_proj — input is the post-conv gated output (now in
                // `prefill_normed_buf`), accumulated into the residual scratch.
                self.encode_lora_hook_batched(
                    &mut cmds,
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

                let w_q = lw.attn_q.as_ref().unwrap();
                let w_k = lw.attn_k.as_ref().unwrap();
                let w_v = lw.attn_v.as_ref().unwrap();
                let w_o = lw.attn_output.as_ref().unwrap();

                // Phase A: Q/K/V batched GEMMs.
                //   Q  → prefill_proj_buf, stride q_dim
                //   K  → prefill_gate_buf, stride kv_dim
                //   V  → prefill_up_buf,   stride kv_dim
                self.encode_mul_mat_reg_tile(
                    &mut cmds,
                    w_q,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    q_dim,
                    all_logits,
                );
                self.encode_mul_mat_reg_tile(
                    &mut cmds,
                    w_k,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    kv_dim,
                    all_logits,
                );
                self.encode_mul_mat_reg_tile(
                    &mut cmds,
                    w_v,
                    &self.prefill_normed_buf,
                    &self.prefill_up_buf,
                    n_u,
                    hs_u,
                    hs_u,
                    kv_dim,
                    all_logits,
                );

                // LoRA Q/K/V deltas: `+= scale·B·(A·normed)` on the raw
                // projections, before QK-norm/RoPE (and the Qwen2 bias) — mirrors
                // the decode hooks and the CPU `apply_attn_qkv`. Q → proj_buf,
                // K → gate_buf, V → up_buf, all token-major (input is the shared
                // attn_norm output in `prefill_normed_buf`).
                self.encode_lora_hook_batched(
                    &mut cmds,
                    lora.as_ref(),
                    layer,
                    LoraTarget::AttnQ,
                    &self.prefill_normed_buf,
                    &self.prefill_proj_buf,
                    n_u,
                );
                self.encode_lora_hook_batched(
                    &mut cmds,
                    lora.as_ref(),
                    layer,
                    LoraTarget::AttnK,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                );
                self.encode_lora_hook_batched(
                    &mut cmds,
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
                    self.encode_bias_add_batch(&mut cmds, &self.prefill_proj_buf, b, n_u, q_dim);
                }
                if let Some(b) = lw.attn_k_bias.as_ref() {
                    self.encode_bias_add_batch(&mut cmds, &self.prefill_gate_buf, b, n_u, kv_dim);
                }
                if let Some(b) = lw.attn_v_bias.as_ref() {
                    self.encode_bias_add_batch(&mut cmds, &self.prefill_up_buf, b, n_u, kv_dim);
                }

                // Phase B: batched per-head Q/K rmsnorm (QK-norm, Qwen3/LFM2
                // only) + RoPE. Pass `None` norms for archs without QK-norm so
                // the kernel runs rope-only.
                self.encode_qk_norm_rope_batch(
                    &mut cmds,
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

                // Phase C: bulk-write K/V into the cache, then Phase D: batched
                // causal attention. Q stride and the output stride are both
                // `q_dim` (concatenated head outputs). Granite overrides the
                // softmax scale via `scalars.attn`; every other arch uses
                // 1/sqrt(head_dim).
                let attn_scale = self
                    .scalars
                    .attn
                    .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
                if let Some(tq) = self.tq_cache() {
                    // Compressed path. The chunk's K/V are compressed into the
                    // cache first, so the causal attention below reads this
                    // chunk's own positions plus the history earlier chunks wrote
                    // — the reason prefill can't stay on the f32 path once the
                    // cache is compressed.
                    tq.encode_kv(
                        &self.ctx,
                        &mut enc,
                        layer,
                        &self.prefill_gate_buf,
                        &self.prefill_up_buf,
                        n,
                    );
                    tq.rotate_queries(
                        &self.ctx,
                        &mut enc,
                        layer,
                        &self.prefill_proj_buf,
                        n,
                        n_heads as usize,
                    );
                    tq.attention(
                        &self.ctx,
                        &mut enc,
                        layer,
                        &self.prefill_normed_buf,
                        n,
                        n_heads as usize,
                    );
                } else {
                    // The KV cache is `max_seq_len × kv_dim` packed f16
                    // halves; pack `n × kv_dim` f32 floats starting at row
                    // `start_pos × kv_dim`. A blit cannot convert, so these
                    // are `kv_append` dispatches (which the emitter folds
                    // into the layer's pass — one fewer split than the old
                    // copies).
                    let (k_cache, v_cache) = self.active_kv(layer);
                    let kv_off_words = (start_pos * kv_dim as usize / 2) as u32;
                    let kv_chunk_floats = (n * kv_dim as usize) as u32;
                    self.encode_kv_append_prefill(
                        &mut cmds,
                        &self.prefill_gate_buf,
                        k_cache,
                        kv_off_words,
                        kv_chunk_floats,
                    );
                    self.encode_kv_append_prefill(
                        &mut cmds,
                        &self.prefill_up_buf,
                        v_cache,
                        kv_off_words,
                        kv_chunk_floats,
                    );

                    let max_seq_for_kv = (start_pos + n) as u32;
                    self.encode_attention_prefill(
                        &mut cmds,
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
                }

                // Phase E: output projection (`q_dim → hs`) → prefill_gate_buf
                // (residual scratch; FFN's add_rmsnorm_batch fuses the add).
                self.encode_mul_mat_reg_tile(
                    &mut cmds,
                    w_o,
                    &self.prefill_normed_buf,
                    &self.prefill_gate_buf,
                    n_u,
                    q_dim,
                    q_dim,
                    hs_u,
                    all_logits,
                );

                // LoRA attn-output delta into gate_buf, BEFORE the FFN's fused
                // `add_rmsnorm_batch` below scales it by `scalars.residual` (so
                // `residual_mult` wraps the delta — hence `b_batched` carries scale
                // only). Input is the attention output (o_proj input) in
                // `prefill_normed_buf`.
                self.encode_lora_hook_batched(
                    &mut cmds,
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
                &mut cmds,
                &self.prefill_batch_buf,
                &self.prefill_normed_buf,
                &lw.ffn_norm,
                &self.prefill_gate_buf,
                n_u,
                hs_u,
            );
            // A routed layer replaces every dispatch from here to the end of the
            // block. Its output goes to `prefill_up_buf` and does *not*
            // accumulate, which is the same convention the dense arm's final
            // `mul_mat_reg_tile` below uses: the next layer's
            // `add_rmsnorm_batch` (or, after the last layer, the final
            // `scaled_add_inplace`) folds it into the residual stream.
            //
            // No LoRA hooks, for the reason the decode path gives: an adapter
            // that could want one here is rejected before it reaches the model.
            let dense = match &lw.ffn {
                GpuFfn::Moe(moe) => {
                    let steps = self.moe_ffn_steps(
                        moe,
                        &self.prefill_normed_buf,
                        &self.prefill_up_buf,
                        n_u,
                        false,
                    );
                    for s in steps {
                        Self::push_prefill_dispatch(
                            &mut cmds,
                            s.pipeline,
                            s.bind_group,
                            s.workgroups,
                            "ffn_moe_batch",
                        );
                    }
                    self.emit_prefill_cmds(&mut enc, &mut cmds, &format!("prefill_l{layer}"), n_u);
                    continue;
                }
                GpuFfn::Dense(d) => d,
            };
            // gate + up GEMMs.
            self.encode_mul_mat_reg_tile(
                &mut cmds,
                &dense.gate,
                &self.prefill_normed_buf,
                &self.prefill_gate_buf,
                n_u,
                hs_u,
                hs_u,
                is_u,
                all_logits,
            );
            self.encode_mul_mat_reg_tile(
                &mut cmds,
                &dense.up,
                &self.prefill_normed_buf,
                &self.prefill_up_buf,
                n_u,
                hs_u,
                hs_u,
                is_u,
                all_logits,
            );
            // LoRA gate/up deltas on the raw projections, before silu_mul. Input
            // is the ffn_norm output in `prefill_normed_buf`; outputs token-major
            // (gate → gate_buf, up → up_buf). Applies to every layer (conv + attn).
            self.encode_lora_hook_batched(
                &mut cmds,
                lora.as_ref(),
                layer,
                LoraTarget::FfnGate,
                &self.prefill_normed_buf,
                &self.prefill_gate_buf,
                n_u,
            );
            self.encode_lora_hook_batched(
                &mut cmds,
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
                let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params));
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
                Self::push_prefill_dispatch(
                    &mut cmds,
                    &self.pipelines.silu_mul_inplace,
                    bg,
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
                &mut cmds,
                &dense.down,
                &self.prefill_gate_buf,
                &self.prefill_up_buf,
                n_u,
                is_u,
                is_u,
                hs_u,
                all_logits,
            );
            // LoRA ffn-down delta into prefill_up_buf, BEFORE the next layer's
            // fused `add_rmsnorm_batch` (or the final `scaled_add_inplace`) scales
            // it by `scalars.residual`. Input is the silu_mul(gate,up) result in
            // `prefill_gate_buf`.
            self.encode_lora_hook_batched(
                &mut cmds,
                lora.as_ref(),
                layer,
                LoraTarget::FfnDown,
                &self.prefill_gate_buf,
                &self.prefill_up_buf,
                n_u,
            );
            self.emit_prefill_cmds(&mut enc, &mut cmds, &format!("prefill_l{layer}"), n_u);
        }

        // ─── Final residual add: batch_buf += residual_scale·prefill_up_buf ─
        // Last layer's FFN down residual lives in `prefill_up_buf`; add it back
        // into the running residual stream. `scaled_add_inplace` folds Granite's
        // residual multiplier into the addend (1.0 ⇒ plain add elsewhere).
        self.encode_scaled_add_inplace_batch(
            &mut cmds,
            &self.prefill_batch_buf,
            &self.prefill_up_buf,
            n_u * hs_u,
            self.scalars.residual,
            "final_add",
        );

        // ─── Final output: norm + LM head ────────────────────────────────
        if !need_logits && !all_logits {
            // Intermediate prefill chunk: skip final output norm + LM head entirely.
            self.emit_prefill_cmds(&mut enc, &mut cmds, "prefill_head", n_u);
        } else if !all_logits {
            // Last token only (standard prefill path). Flush the final_add
            // before the direct-encoder copy below (order matters).
            self.emit_prefill_cmds(&mut enc, &mut cmds, "prefill_head", n_u);
            let last_off_floats = ((n - 1) * hs) as u64;
            Self::encode_copy(
                &mut enc,
                &self.prefill_batch_buf,
                last_off_floats,
                &self.hidden_buf,
                0,
                hs as u64,
            );
            // One merged tail pass (norm + projection + optional scale),
            // same as the decode tail.
            let scale_bg = self
                .logit_scale_params
                .as_ref()
                .map(|params| self.logit_scale_bg(params));
            {
                let mut pass = self.ctx.begin_pass(&mut enc, "tail");
                self.encode_rmsnorm_into(&mut pass, &self.hidden_buf, &self.output_norm);
                self.encode_lm_head_into(&mut pass, &self.hidden_buf, &self.logits_buf);
                if let Some(scale_bg) = scale_bg.as_ref() {
                    self.dispatch_into(
                        &mut pass,
                        &self.pipelines.scale_f32,
                        scale_bg,
                        ((cfg.vocab_size as u32).div_ceil(256), 1, 1),
                    );
                }
            }
        } else {
            // All n tokens: batched RMSNorm + batched GEMM projection in ONE dispatch!
            let vocab = cfg.vocab_size;
            self.encode_rmsnorm_batch(
                &mut cmds,
                &self.prefill_batch_buf,
                &self.prefill_normed_buf,
                &self.output_norm,
                n_u,
                hs_u,
            );
            match &self.lm_head {
                LmHead::Quantized { main: w, .. } => {
                    self.encode_mul_mat_reg_tile(
                        &mut cmds,
                        w,
                        &self.prefill_normed_buf,
                        &self.prefill_all_logits_buf,
                        n_u,
                        hs_u,
                        hs_u,
                        vocab as u32,
                        all_logits,
                    );
                }
                LmHead::F16 { weight, params } => {
                    // Flush the normed rmsnorm before the direct-encoder
                    // per-token loop (order matters).
                    self.emit_prefill_cmds(&mut enc, &mut cmds, "prefill_head", n_u);
                    for j in 0..n {
                        let tok_off_floats = (j * hs) as u64;
                        Self::encode_copy(
                            &mut enc,
                            &self.prefill_normed_buf,
                            tok_off_floats,
                            &self.hidden_buf,
                            0,
                            hs as u64,
                        );
                        self.encode_gemv_f16(
                            &mut enc,
                            weight,
                            params,
                            &self.hidden_buf,
                            &self.logits_buf,
                        );
                        Self::encode_copy(
                            &mut enc,
                            &self.logits_buf,
                            0,
                            &self.prefill_all_logits_buf,
                            (j * vocab) as u64,
                            vocab as u64,
                        );
                    }
                }
            }
            if self.scalars.logit != 1.0 {
                // Flush the batched GEMM before the direct-encoder scale.
                self.emit_prefill_cmds(&mut enc, &mut cmds, "prefill_head", n_u);
                let total = n_u * (vocab as u32);
                let params_data: [u32; 2] = [total, (1.0 / self.scalars.logit).to_bits()];
                let p_buf = self.next_prefill_params(bytemuck::cast_slice(&params_data));
                let scale_bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("logit_scale_bg"),
                        layout: &self.pipelines.scale_f32.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.prefill_all_logits_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: p_buf.as_entire_binding(),
                            },
                        ],
                    });
                self.encode(
                    &mut enc,
                    &self.pipelines.scale_f32,
                    &scale_bg,
                    (total.div_ceil(256), 1, 1),
                    "logit_scale",
                );
            }
        }

        // Flush anything still recorded (all_logits Quantized path with
        // logit == 1.0 leaves the tail GEMM here; every other path flushed
        // above — emitting an empty vec is a no-op).
        self.emit_prefill_cmds(&mut enc, &mut cmds, "prefill_head", n_u);

        enc
    }

    fn forward_prefill_batched_locked(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
        all_logits: bool,
        need_logits: bool,
    ) -> Vec<f32> {
        let n = tokens.len();
        let host_prof = std::env::var("CERA_GPU_HOST_PROFILE").as_deref() == Ok("1");
        let passes_before = crate::backend::wgpu::io_stats::snapshot().passes;
        let t_enc = std::time::Instant::now();
        let enc =
            self.encode_prefill_batched_locked(tokens, start_pos, state, all_logits, need_logits);
        if host_prof {
            let passes = crate::backend::wgpu::io_stats::snapshot().passes - passes_before;
            eprintln!(
                "[GPU-HOST] prefill_encode={:.0}µs passes={passes} n={n}",
                t_enc.elapsed().as_secs_f64() * 1e6,
            );
        }
        self.submit_and_wait(enc);
        self.gpu_state
            .seq_len
            .store(start_pos + n, Ordering::Relaxed);
        state.seq_len = start_pos + n;
        self.ctx.finish_profiler();
        if !need_logits && !all_logits {
            Vec::new()
        } else if !all_logits {
            self.ctx
                .download_f32(&self.logits_buf, self.config.vocab_size)
        } else {
            self.ctx
                .download_f32(&self.prefill_all_logits_buf, n * self.config.vocab_size)
        }
    }

    /// Async (wasm/WebGPU) batched prefill step returning all logits rows `[n x vocab_size]`,
    /// used for zero-allocation speculative verification on WebGPU.
    pub async fn forward_prefill_logits_all_async(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > MAX_ALL_LOGITS_TOKENS {
            anyhow::bail!("batch size {n} exceeds MAX_ALL_LOGITS_TOKENS ({MAX_ALL_LOGITS_TOKENS})");
        }
        if !self.batched_prefill || self.unbatchable_matmul_weight().is_some() {
            anyhow::bail!("batched prefill verification not supported for this model");
        }
        let vocab = self.config.vocab_size;
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            let enc = self.encode_prefill_batched_locked(tokens, start_pos, state, true, true);
            self.gpu_state
                .seq_len
                .store(start_pos + n, Ordering::Relaxed);
            state.seq_len = start_pos + n;
            self.ctx.begin_download_with_encoder(
                enc,
                &self.prefill_all_logits_buf,
                (n * vocab * std::mem::size_of::<f32>()) as u64,
            )
        };

        let bytes = pending.recv().await?;
        let expected_bytes = n * vocab * std::mem::size_of::<f32>();
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU prefill logits readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        let mut out = vec![0.0f32; n * vocab];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..expected_bytes]);
        Ok(out)
    }

    /// Async (wasm/WebGPU) batched verification step returning argmax token IDs `[n]`,
    /// reducing the readback from `n * vocab * 4` bytes down to `n * 4` bytes (e.g. 16 bytes for 4 tokens).
    pub async fn forward_prefill_argmax_all_async(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Result<Vec<u32>> {
        let n = tokens.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > MAX_ALL_LOGITS_TOKENS {
            anyhow::bail!("batch size {n} exceeds MAX_ALL_LOGITS_TOKENS ({MAX_ALL_LOGITS_TOKENS})");
        }
        if !self.batched_prefill || self.unbatchable_matmul_weight().is_some() {
            anyhow::bail!("batched prefill verification not supported for this model");
        }
        let vocab = self.config.vocab_size;
        let pending = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            let _lora_guard = self.resolve_lora(state);
            let mut enc = self.encode_prefill_batched_locked(tokens, start_pos, state, true, true);
            self.gpu_state
                .seq_len
                .store(start_pos + n, Ordering::Relaxed);
            state.seq_len = start_pos + n;

            // Run argmax on each row of prefill_all_logits_buf on GPU.
            // Copy row j to logits_buf (offset 0) to avoid WebGPU 256-byte storage buffer
            // alignment constraints when vocab_size * 4 is not a multiple of 256.
            for j in 0..n {
                Self::encode_copy(
                    &mut enc,
                    &self.prefill_all_logits_buf,
                    (j * vocab) as u64,
                    &self.logits_buf,
                    0,
                    vocab as u64,
                );
                let params_buf =
                    self.next_prefill_params(bytemuck::cast_slice(&[vocab as u32, j as u32]));
                let bg = self
                    .ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("argmax_batch_row_bg"),
                        layout: &self.pipelines.argmax_f32.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.logits_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.argmax_out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });
                let mut pass = self.ctx.begin_pass(&mut enc, "argmax_batch_row");
                pass.set_pipeline(&self.pipelines.argmax_f32);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            self.ctx.begin_download_with_encoder(
                enc,
                &self.argmax_out_buf,
                std::mem::size_of_val(tokens) as u64,
            )
        };

        let bytes = pending.recv().await?;
        let expected_bytes = std::mem::size_of_val(tokens);
        if bytes.len() < expected_bytes {
            anyhow::bail!(
                "GPU prefill argmax readback buffer truncated (expected {expected_bytes} bytes, got {})",
                bytes.len()
            );
        }
        let mut out = vec![0u32; n];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..expected_bytes]);
        Ok(out)
    }

    /// Truncate the on-GPU KV cache sequence length and state for speculative rollback.
    pub fn truncate_kv_direct(&self, state: &mut InferenceState, len: usize) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.gpu_state.seq_len.store(len, Ordering::Relaxed);
        state.seq_len = len;
    }
}

impl GpuLfm2Model {
    /// Lock-free body of `Model::snapshot_state`. Callers that already
    /// hold `infer_lock` (e.g. `forward_prefill`'s prefix-cache write
    /// step) call this directly to avoid a recursive `Mutex::lock()`
    /// deadlock — `std::sync::Mutex` is not reentrant.
    ///
    /// Snapshot layout (mirrors Metal's pattern, including f16 KV): per
    /// attention layer, download the live `seq_len * kv_dim / 2` packed
    /// words from K and V and emit `AttentionF16`; per conv layer,
    /// download the full `d_conv * hidden_size` rolling buffer. Words →
    /// bytes via `bytemuck::cast_slice` on the contiguous `Vec<u32>`
    /// from `download_u32` (source-aligned, safe).
    /// Snapshot GPU state into the prefix cache, skipping the snapshot
    /// entirely when the cache is disabled. Building it unconditionally costs
    /// a blocking readback per layer (~20+ readbacks) just to have `insert`
    /// throw it away — measured as pure overhead on every prefill with
    /// `--no-cache`.
    fn maybe_snapshot_prefix_locked(&self, tokens: &[u32]) {
        let mut cache = self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.stores_entries() {
            let snap = self.snapshot_state_locked();
            cache.insert(tokens, snap);
        }
    }

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

        let tq = self.tq_cache();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if cfg.block_types[i] == BlockType::Attention {
                if let Some(tq) = tq {
                    // Compressed cache: emit the same `TQK1`/`TQV1` blobs the CPU
                    // backend writes, so the two are mutually loadable.
                    let (keys, values) = tq.snapshot_layer(&self.ctx, i, seq_len);
                    layers.push(LayerSnapshot::AttentionCompressed { keys, values });
                    continue;
                }
                let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                // Packed halves: download words, emit the same `AttentionF16`
                // LE-bytes the CPU/Metal backends write (a packed u32 IS two
                // LE u16s), so snapshots stay mutually loadable.
                let n_words = seq_len * kv_dim / 2;
                let (k_buf, v_buf) = self.active_kv(i);
                let k_words = self.ctx.download_u32(k_buf, n_words);
                let v_words = self.ctx.download_u32(v_buf, n_words);
                layers.push(LayerSnapshot::AttentionF16 {
                    k_data: bytemuck::cast_slice(&k_words).to_vec(),
                    v_data: bytemuck::cast_slice(&v_words).to_vec(),
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
        StateSnapshot::new(layers, seq_len)
    }

    /// Lock-free body of `Model::restore_state`. See
    /// [`Self::snapshot_state_locked`] for the locking contract.
    /// Writes raw bytes via `queue.write_buffer` at offset 0 — wgpu's
    /// `COPY_BUFFER_ALIGNMENT` is 4, which packed-word byte counts
    /// always satisfy. The remainder of the pre-allocated cache (past
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
                    assert!(
                        self.tq_cache().is_none(),
                        "f32 Attention snapshot restored into a TurboQuant-configured \
                         wgpu model at layer {i}; the lookup gate in forward_prefill \
                         must reject a mode-mismatched snapshot"
                    );
                    // Legacy f32 entry (pre-packed-cache prefix files, or a
                    // CPU-written f32 entry): pack to halves on the host.
                    // The cache holds an even float count (kv_dim is even),
                    // so every u32 packs exactly one pair.
                    let pack_f32 = |data: &[u8]| -> Vec<u8> {
                        assert!(
                            data.len().is_multiple_of(8),
                            "f32 snapshot layer {i} has {} bytes, not a multiple of 8",
                            data.len()
                        );
                        let floats: &[f32] = bytemuck::cast_slice(data);
                        let mut words = Vec::with_capacity(floats.len() / 2);
                        for pair in floats.as_chunks::<2>().0 {
                            let lo = half::f16::from_f32(pair[0]).to_bits() as u32;
                            let hi = half::f16::from_f32(pair[1]).to_bits() as u32;
                            words.push(lo | (hi << 16));
                        }
                        bytemuck::cast_slice::<u32, u8>(&words).to_vec()
                    };
                    let (k_buf, v_buf) = self.active_kv(i);
                    if !k_data.is_empty() {
                        self.ctx.queue.write_buffer(k_buf, 0, &pack_f32(k_data));
                    }
                    if !v_data.is_empty() {
                        self.ctx.queue.write_buffer(v_buf, 0, &pack_f32(v_data));
                    }
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
                    if !buffer.is_empty() {
                        self.ctx.queue.write_buffer(conv_buf, 0, buffer);
                    }
                }
                LayerSnapshot::AttentionCompressed { keys, values } => {
                    assert_eq!(
                        cfg.block_types[i],
                        BlockType::Attention,
                        "snapshot layer {i} attention vs state config"
                    );
                    // Reaching here without a compressed cache means the
                    // lookup-time mode gate was bypassed: the compressed blobs
                    // have no f32 slot to land in, so restoring would leave the
                    // kernels reading whatever was in the packed cache before.
                    let tq = self.tq_cache().unwrap_or_else(|| {
                        panic!(
                            "GpuLfm2Model::restore_state_locked received a \
                             TurboQuant-compressed snapshot at layer {i} but this \
                             model is not TurboQuant-configured; callers must gate \
                             on `StateSnapshot::is_compressed`"
                        )
                    });
                    // Cross-check the decoded length against the snapshot's own
                    // `seq_len`, which is what `gpu_state.seq_len` is set from
                    // below: a disagreement would leave the kernels reading
                    // compressed slots nothing wrote.
                    let restored =
                        tq.restore_layer(&self.ctx, i, keys, values)
                            .unwrap_or_else(|| {
                                panic!(
                                    "invalid or shape-mismatched TurboQuant blob in \
                                 snapshot at layer {i}"
                                )
                            });
                    assert_eq!(
                        restored, snapshot.seq_len,
                        "layer {i}: restored TurboQuant seq_len {restored} disagrees \
                         with the snapshot's {}",
                        snapshot.seq_len
                    );
                }
                LayerSnapshot::AttentionF16 { k_data, v_data } => {
                    assert_eq!(
                        cfg.block_types[i],
                        BlockType::Attention,
                        "snapshot layer {i} attention vs state config"
                    );
                    assert!(
                        self.tq_cache().is_none(),
                        "f16 Attention snapshot restored into a TurboQuant-configured \
                         wgpu model at layer {i}; the lookup gate in forward_prefill \
                         must reject a mode-mismatched snapshot"
                    );
                    // Native format: packed-halves LE bytes land verbatim
                    // (same bytes this backend, CPU-f16, and Metal write).
                    let (k_buf, v_buf) = self.active_kv(i);
                    if !k_data.is_empty() {
                        self.ctx.queue.write_buffer(k_buf, 0, k_data);
                    }
                    if !v_data.is_empty() {
                        self.ctx.queue.write_buffer(v_buf, 0, v_data);
                    }
                }
                LayerSnapshot::Mamba2 { .. }
                | LayerSnapshot::ParallelAttentionMamba2 { .. }
                | LayerSnapshot::DeltaNet { .. } => {
                    panic!(
                        "GpuLfm2Model::restore_state_locked received an unsupported recurrent snapshot at layer {i}; \
                         Mamba2 and DeltaNet are not supported on GpuLfm2Model."
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

    /// Resets GPU-side session state (rolling conv buffers and sequence counter).
    /// Used by WebGPU sessions to perform in-place resets without reloading weights.
    pub fn reset_session_state(&self) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.gpu_state.seq_len.store(0, Ordering::Relaxed);
        self.zero_conv_buffers_locked();
    }

    /// Asynchronously captures a resumable snapshot of GPU session state
    /// (attention KV buffers, rolling conv buffers, and sequence counter).
    ///
    /// Unlike the blocking `snapshot_state_locked` helper, this method
    /// dispatches GPU staging buffer readbacks under `infer_lock`, releases
    /// the lock, and asynchronously awaits buffer mapping without blocking
    /// the browser event loop.
    pub async fn snapshot_session_state_async(&self) -> Result<StateSnapshot, anyhow::Error> {
        let (seq_len, pending_layers) = {
            let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
            if self.tq_cache().is_some() {
                anyhow::bail!("async session snapshot is not supported with GPU TurboQuant cache");
            }
            let seq_len = self.gpu_state.seq_len.load(Ordering::Relaxed);
            let cfg = &self.config;
            let head_dim = cfg.head_dim;
            let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
            let d_conv = kernel_size.saturating_sub(1);

            let mut pending_layers = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                if cfg.block_types[i] == BlockType::Attention {
                    let kv_dim = cfg.kv_heads_per_layer[i] * head_dim;
                    let count = seq_len * kv_dim;
                    let size = (count * std::mem::size_of::<f32>()) as u64;
                    let (k_buf, v_buf) = self.active_kv(i);
                    let (k_pending, v_pending) = if size > 0 {
                        (
                            Some(self.ctx.begin_download(k_buf, size)),
                            Some(self.ctx.begin_download(v_buf, size)),
                        )
                    } else {
                        (None, None)
                    };
                    pending_layers.push((true, k_pending, v_pending));
                } else {
                    let count = d_conv * cfg.hidden_size;
                    let size = (count * std::mem::size_of::<f32>()) as u64;
                    let conv_buf = self.gpu_state.conv_buffers[i].as_ref().ok_or_else(|| {
                        anyhow::anyhow!("conv layer {i} must have rolling buffer")
                    })?;
                    let pending = if size > 0 {
                        Some(self.ctx.begin_download(conv_buf, size))
                    } else {
                        None
                    };
                    pending_layers.push((false, pending, None));
                }
            }
            (seq_len, pending_layers)
        };

        let mut layers = Vec::with_capacity(pending_layers.len());
        for (is_attention, p1, p2) in pending_layers {
            if is_attention {
                let k_data = if let Some(p) = p1 {
                    p.recv().await?
                } else {
                    Vec::new()
                };
                let v_data = if let Some(p) = p2 {
                    p.recv().await?
                } else {
                    Vec::new()
                };
                layers.push(LayerSnapshot::Attention { k_data, v_data });
            } else {
                let buffer = if let Some(p) = p1 {
                    p.recv().await?
                } else {
                    Vec::new()
                };
                layers.push(LayerSnapshot::Conv { buffer });
            }
        }
        Ok(StateSnapshot::new(layers, seq_len))
    }

    /// Restores GPU session state (attention KV buffers, rolling conv buffers,
    /// and sequence counter) from a snapshot.
    ///
    /// Acquires `infer_lock` and updates VRAM buffers via non-blocking queue
    /// writes.
    pub fn restore_session_state(&self, snapshot: &StateSnapshot) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.restore_state_locked(snapshot);
    }

    /// Returns true if this GPU model is configured with TurboQuant KV cache compression.
    #[inline]
    pub fn is_compressed(&self) -> bool {
        self.tq_cache().is_some()
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod recovery;

impl Model for GpuLfm2Model {
    #[cfg(not(target_arch = "wasm32"))]
    fn try_reset_kv(
        &self,
        state: &mut InferenceState,
        compression: &crate::kv_cache::KvCompression,
        max_seq_len: usize,
    ) -> Result<(), crate::session::CeraError> {
        self.reset_kv_checked(state, compression, max_seq_len)
    }

    fn acquire_session(&self) -> Result<Option<super::ModelSessionLease>, CeraError> {
        self.session_gate.try_acquire().map(Some)
    }

    fn supports_all_logits(&self) -> bool {
        self.batched_prefill && self.unbatchable_matmul_weight().is_none()
    }

    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            self.gpu_state.seq_len.store(start_pos, Ordering::Relaxed);
            return self.forward_inner(tokens, start_pos, state);
        }
        assert!(
            n <= MAX_ALL_LOGITS_TOKENS,
            "forward_prefill_logits_all token count ({n}) exceeds MAX_ALL_LOGITS_TOKENS ({MAX_ALL_LOGITS_TOKENS})"
        );
        self.forward_prefill_batched_locked(tokens, start_pos, state, true, true)
    }

    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.gpu_state.seq_len.store(len, Ordering::Relaxed);
        state.seq_len = len;
    }

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
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
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
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Stage the caller's adapter for the per-layer encoders; the guard
        // clears it on the way out.
        let _lora_guard = self.resolve_lora(state);
        self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
        self.forward_inner(tokens, pos, state)
    }

    fn forward_greedy(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> u32 {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        self.gpu_state.seq_len.store(pos, Ordering::Relaxed);
        self.forward_greedy_inner(tokens, pos, state)
    }

    fn take_decode_error(&self) -> Option<CeraError> {
        // Blocking readbacks zero-fill on map failure (see
        // `GpuContext::warn_readback_zeros`); drain the sticky record so
        // the session fails the generation instead of sampling the zeros.
        // Sticky-until-taken also covers prefill readbacks: a prefill fault
        // is still pending at the next session check (the session checks
        // after `append_tokens` too, so no bad token is emitted first).
        self.ctx.take_readback_fault()
    }

    fn supports_embedding_input(&self) -> bool {
        true
    }

    fn forward_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        // `state.seq_len`, not the `pos` argument, matching the CPU model:
        // embeddings are appended at the cache's current end, and callers
        // splicing an image into a prompt track position through the state.
        let pos = state.seq_len;
        self.forward_inner_compute_from_embedding(embedding, pos, state);
        self.ctx
            .download_f32(&self.logits_buf, self.config.vocab_size)
    }

    /// The audio path's counterpart to [`Model::forward`]: same layer stack and
    /// same output norm, but it stops before the projection, because what
    /// consumes the result is the depthformer rather than a sampler.
    ///
    /// Without this the whole audio pipeline was unreachable on this backend. A
    /// `cera-cli --features gpu` build with no Metal auto-selects WGPU for the
    /// LLM, and `generate_audio` calls straight into here, so the default in
    /// `model/mod.rs` panicked before a single frame was produced — including
    /// for anyone who had picked `CERA_AUDIO_GPU=wgpu` to get the WGPU
    /// detokenizer.
    fn forward_embedding(
        &self,
        tokens: &[u32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        // `state.seq_len` over the `pos` argument for the same reason as
        // `forward_from_embedding` above, and matching the CPU model, which
        // ignores its own `pos` here too.
        let pos = state.seq_len;
        self.forward_inner_compute_tail(tokens, pos, state, DecodeTail::Hidden);
        self.ctx
            .download_f32(&self.hidden_buf, self.config.hidden_size)
    }

    /// [`Model::forward_embedding`] seeded by a hidden vector rather than a
    /// token id. This is how an audio frame's codes are fed back into the LLM:
    /// the frame has no token id that could produce its embedding.
    fn forward_hidden_from_embedding(
        &self,
        embedding: &[f32],
        _pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        let pos = state.seq_len;
        self.forward_inner_compute_tail_seeded(
            HiddenSeed::Embedding(embedding),
            pos,
            state,
            DecodeTail::Hidden,
        );
        self.ctx
            .download_f32(&self.hidden_buf, self.config.hidden_size)
    }

    /// Overridden so an image costs **one** logits readback instead of one per
    /// patch token.
    ///
    /// The default in `model/mod.rs` loops [`Model::forward_from_embedding`],
    /// and every one of those ends in a blocking `download_f32`. Two problems,
    /// and the second is fatal rather than merely slow:
    ///
    /// - Natively it pulls a full vocab-sized vector per frame and discards all
    ///   but the last, which for a many-token image is most of the work.
    /// - On wasm it deadlocks. `download_f32` blocks in `mpsc::recv` waiting on
    ///   the buffer-map callback, and `poll_wait()` is a no-op there because
    ///   WebGPU is driven by the JS event loop, which cannot run while this
    ///   thread is the one blocking it. See `GpuContext::begin_download`, whose
    ///   docs spell out that the blocking helpers have no wasm analog.
    ///
    /// Seeding is therefore split from the readback: frames go through
    /// `seed_embeddings_locked` (private, hence not linked), which leaves
    /// logits on the GPU, and
    /// only the final frame's are brought back. Callers that want no readback
    /// at all (appending an image is one: it needs the KV cache, not logits)
    /// should call [`Self::seed_embeddings`] instead, which is the only form
    /// that is safe to call on wasm.
    fn forward_prefill_from_embeddings(
        &self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _lora_guard = self.resolve_lora(state);
        self.seed_embeddings_locked(embeddings, n_tokens, start_pos, state);
        self.ctx
            .download_f32(&self.logits_buf, self.config.vocab_size)
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Stage the caller's adapter so both prefill paths apply it: the
        // batched-GEMM path runs the in-batch LoRA hooks (two NT GEMMs per
        // target), and the sequential fallback loop runs the decode hooks. The
        // guard clears `active_lora` on the way out.
        let _lora_guard = self.resolve_lora(state);
        // Ask the *resolved* adapter, not `state.lora`. `resolve_lora` drops an
        // adapter carrying routed-FFN deltas entirely, and such a run is a pure
        // base-model prefill whose KV is cacheable. Reading `state.lora` here
        // would disable the prefix cache for both the lookup and the insert, so
        // every prefill in that session would run cold and none would ever
        // populate the cache.
        let lora_active = self
            .active_lora
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        // Reset internal seq_len so repeated generate() calls (bench) work.
        self.gpu_state.seq_len.store(start_pos, Ordering::Relaxed);

        // Fresh-prefill-only work: the prefix-cache lookup and zeroing the conv
        // rolling buffers. The batched-GEMM path itself runs at ANY `start_pos`
        // (see below) — including with a LoRA active, which applies in-batch (two
        // NT GEMMs per target) rather than forcing the per-token fallback. The
        // prefix cache is still bypassed with an adapter: cached KV is
        // base-model-only, so restoring it and adapting only the tail would
        // corrupt the result (and inserting adapter-modified KV would poison the
        // cache for later base runs).
        if start_pos == 0 {
            // Cache lookup only for base-model prefills.
            let hit = (!lora_active)
                .then(|| {
                    self.prefix_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .find_longest_prefix(tokens)
                })
                .flatten()
                // Compression-mode gate. `cache_namespace` now folds the mode and
                // seed into the disk fingerprint, so a cross-mode entry should be
                // unreachable — this is a defensive backstop, not the primary
                // guard, because the alternative on a namespace bug is
                // `restore_state_locked` panicking or restoring a cache the
                // kernels misread. Keep both: the filter is nearly free, and it
                // degrades a namespacing regression to a cold prefill.
                .filter(|(snapshot, _)| snapshot.is_compressed() == self.tq_cache().is_some());
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
                    self.maybe_snapshot_prefix_locked(tokens);
                    return logits;
                }
            }
            // Cache miss on a fresh prefill: zero the GPU conv
            // rolling buffers so stale state from a prior
            // generation can't leak in. Cache hits skip this
            // (`restore_state_locked` rewrites the buffers from
            // the snapshot). Mirrors the equivalent fix on Metal.
            //
            // A continuation (`start_pos > 0`) must NOT reach here: its conv
            // rolling state is exactly what the previous chunk left behind.
            self.zero_conv_buffers_locked();
        }

        // Try the batched prefill path. Preconditions:
        //   * non-empty
        //   * every matmul weight has a batched reg-tile kernel: the five
        //     quantized dtypes (Q4_0/Q8_0/Q4KM/Q5KM/Q6K) plus F32, which
        //     `upload_weight` normalizes every other dtype to, so this holds for
        //     every real model
        //   * the model wires the batched-prefill path (`batched_prefill`).
        //     LFM2 and the dense transformers (llama/qwen2/qwen3/mistral/
        //     granite) all support it.
        //
        // Deliberately NOT gated on `start_pos == 0`. `forward_prefill_chunked`
        // (model/mod.rs) splits a prompt into ubatch-sized chunks and calls this
        // once per chunk with an advancing `start_pos`; gating the batched path
        // on a fresh prefill silently dropped every chunk after the first onto
        // the per-token loop, so any prompt longer than one ubatch ran most of
        // itself at decode speed (measured: p=1024 took 8730 GPU submits, of
        // which ~8700 were the second chunk going token-by-token). Metal's
        // `forward_prefill` has always run its batched inner path for any
        // `start_pos`; this matches it. Only the prefix-cache lookup/insert and
        // the conv zeroing are fresh-prefill-only.
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
                let is_last = end >= tokens.len();
                logits = self.forward_prefill_batched_locked(
                    &tokens[pos..end],
                    start_pos + pos,
                    state,
                    false,
                    is_last,
                );
                pos = end;
            }
            // Only cache base-model KV — an adapted run's KV must never be
            // reused — and only for a prefill that started at 0, since the cache
            // key is the whole prefix. A continuation chunk's `tokens` is a
            // fragment, not a prefix.
            if start_pos == 0 && !lora_active {
                self.maybe_snapshot_prefix_locked(tokens);
            }
            return logits;
        }

        // Per-token fallback. Reached only when the batched path above declined:
        // empty input, a model without `batched_prefill`, or an unbatchable dtype.
        // Continuation chunks no longer land here — they take the batched path.
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
            self.maybe_snapshot_prefix_locked(tokens);
        }
        logits
    }

    fn configure_cache(&self, config: crate::kv_cache::KvCacheConfig) {
        let id = self.cache_namespace();
        *self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner()) =
            KvPrefixCache::for_model(config, &self.config, &self.model_id, &id);
    }

    #[cfg(test)]
    fn warm_cache_usage(&self) -> Option<(usize, u64)> {
        let cache = self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
        Some((cache.warm_count(), cache.warm_bytes()))
    }

    fn clear_warm_cache(&self) {
        self.prefix_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear_warm();
    }

    fn clear_cache(&self) {
        self.prefix_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Public Model trait surface for `_locked` snapshot/restore so
    /// external state-management callers (FFI / parity harness)
    /// can drive the prefix cache directly without going through
    /// `forward_prefill`. Mirrors `MetalLfm2Model`'s overrides.
    fn snapshot_state(&self) -> StateSnapshot {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.snapshot_state_locked()
    }

    fn restore_state(&self, snapshot: &StateSnapshot) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.restore_state_locked(snapshot);
    }

    fn supports_moe_lora(&self) -> bool {
        // The routed FFN runs here, but without LoRA hooks: `moe_ffn_steps`
        // applies no delta to the router or to any expert projection. Uploading
        // per-expert factors means a fourth stacked tensor per projection and a
        // rank-indexed variant of the expert GEMV, which is a port of its own.
        // Until then this stays false so `Session` refuses the adapter outright
        // instead of applying the attention half and silently dropping the rest.
        false
    }

    /// Dense-target hooks via `resolve_lora` (routed-FFN targets excluded, as
    /// above).
    fn supports_lora(&self) -> bool {
        true
    }

    fn turboquant_supported(&self) -> bool {
        // Gated on `head_dim`: the compressed kernels need a power-of-two
        // `head_dim` that is <= 128 and a multiple of 32. Reporting the real
        // capability here lets the CLI warn and fall back to f32 up front rather
        // than have `configure_kv_compression` silently ignore the request.
        crate::model::gpu_turboquant::head_dim_supported(self.config.head_dim)
    }

    fn configure_kv_compression(&self, compression: &KvCompression) -> Result<(), CeraError> {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        let want = TqMode::from_compression(compression, self.config.head_dim);

        // Requests the compressed path can't serve fall back to f32 rather than
        // erroring, matching the CPU's silent fallback for a non-power-of-two
        // head_dim. Warn so a silently-downgraded request is visible.
        if want.is_none() && matches!(compression, KvCompression::TurboQuant { .. }) {
            tracing::warn!(
                target: "cera::gpu",
                head_dim = self.config.head_dim,
                "TurboQuant requested but not supported for this configuration on \
                 the wgpu backend (needs keys+values compression and a \
                 power-of-two head_dim <= 128 that is a multiple of 32); \
                 falling back to packed-f16 KV"
            );
        }

        // First call wins. The compressed and packed-f16 caches have different layouts
        // and only the configured one is ever allocated, so a mode change after
        // the fact can't be honored — reject it instead of handing the kernels a
        // cache they don't match.
        if let Some(&configured) = self.kv_mode.get() {
            if configured != want {
                return Err(CeraError::KvCompressionConflict {
                    configured: describe_kv_mode(&configured),
                    requested: describe_kv_mode(&want),
                });
            }
            // Same mode → no-op (this is the `Session::reset` path). The two
            // records must agree: `kv_mode` is what we promised, `tq.mode` is what
            // was actually built.
            debug_assert_eq!(
                self.tq.get().map(|t| t.mode),
                configured,
                "kv_mode and the built TurboQuant cache disagree"
            );
            return Ok(());
        }

        if let Some(mode) = want {
            let q_cap = self.gpu_state.max_seq_len.min(MAX_PREFILL_TOKENS);
            let cache = TqGpuCache::new(
                &self.ctx,
                &self.config,
                self.gpu_state.max_seq_len,
                q_cap,
                mode,
            )?;
            // `set` can only fail if another thread won the race, which
            // `infer_lock` rules out.
            assert!(self.tq.set(cache).is_ok(), "tq cache set race");
        }
        let _ = self.kv_mode.set(want);
        // Tag with the mode the cache will actually hold, not the one requested, so
        // a downgraded request shares the f32 namespace it is now writing into.
        // Every downgrade — an unsupported `head_dim` as well as the GPU-only
        // restrictions (single-sided TurboQuant) — leaves `want` as `None` and lands
        // in the f32 arm below; `resolved_for` is then a no-op, since
        // `want.is_some()` already implies the `head_dim` it re-checks.
        let _ = self.kv_cache_tag.set(if want.is_some() {
            compression.resolved_for(&self.config).cache_tag()
        } else {
            KvCompression::None.cache_tag()
        });

        // Re-namespace the prefix cache now that the mode is known. The engine
        // calls `configure_cache` at load time, before any session exists, so the
        // cache it built is tagged for the default (f32) mode; leaving it that way
        // would let a compressed snapshot land in the f32 disk namespace and
        // shadow it. Only needed when the tag actually changes — the f32 case is
        // already correctly namespaced, and rebuilding would discard its warm tier
        // for nothing. The warm tier is empty here regardless (no forward has run
        // on this instance yet).
        let tag_changed = self.kv_cache_tag.get().is_some_and(|t| !t.is_empty());
        if tag_changed {
            let id = self.cache_namespace();
            let mut cache = self.prefix_cache.lock().unwrap_or_else(|e| e.into_inner());
            let cache_config = cache.config.clone();
            *cache = KvPrefixCache::for_model(cache_config, &self.config, &self.model_id, &id);
        }
        Ok(())
    }

    fn supports_kv_shift(&self) -> bool {
        // Mirror of CPU `Lfm2Model` / Metal `MetalLfm2Model` — the wgpu backend
        // implements the GPU-side shift via the `kv_shift` WGSL kernel +
        // `copy_buffer_to_buffer`. See `Self::shift_kv`.
        //
        // Not implemented for the compressed cache: the shift re-rotates stored K
        // by a RoPE delta, which needs the raw vectors, and TurboQuant only keeps
        // 2-bit rotated indices. Report `false` so `Session` warns accurately
        // instead of promising a shift the overflow path will refuse.
        self.tq.get().is_none()
    }

    fn shift_kv(&self, state: &mut InferenceState, n_keep: usize, shift: usize) {
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
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
        // Shifting a compressed cache is not implemented. `Session::can_shift`
        // gates it on `state.is_compressed()` — true when *either* side is
        // packed — which is the condition re-asserted here. `supports_kv_shift`
        // above is narrower: it tracks `self.tq`, which stays empty for a
        // request this backend downgraded but the state-side cache compressed.
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

/// Current GPU shader clock in MHz via devfreq sysfs, for microbench
/// diagnostics. Picks the first `kgsl`/`gpu` node that is not a bus
/// monitor (Adreno exposes `3d00000.qcom,kgsl-3d0`). `None` off-Android
/// or when the nodes are unreadable — the benches print `n/a` there.
fn gpu_cur_freq_mhz() -> Option<u64> {
    for entry in std::fs::read_dir("/sys/class/devfreq").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains("busmon") || !(name.contains("kgsl") || name.contains("gpu")) {
            continue;
        }
        let text = std::fs::read_to_string(entry.path().join("cur_freq")).ok()?;
        let hz: u64 = text.trim().parse().ok()?;
        return Some(hz / 1_000_000);
    }
    None
}

/// Largest single microbench dim. Kills `u32` overflow in the `n_pad`
/// round-up and the `*4` byte math (legit shapes are in the hundreds).
const BENCH_MAX_DIM: u64 = 1 << 24;
/// Largest estimated worst-case transient allocation for one microbench
/// shape, in bytes (host synth + packed + twins + device buffers — see the
/// estimators below). A typo'd giant shape must fail with a message, not
/// abort in the allocator. Sized so the CLI's own default `gemv-bench`
/// shapes validate: the 128000x2048 LM-head row estimates ~1.53 GiB
/// under the deliberately over-counting estimator.
const BENCH_MAX_BYTES: u64 = 1 << 31;

/// Stream-`(q, d)` twin bytes for an `(m, k)` weight matrix: `q` holds
/// `m*k/8` u32s, `d` holds `m*ceil(nb/2)` u32s (`nb = k/32`), with a
/// zero-padded lane when `nb` is odd — so the twins are exactly `packed`
/// only at even block counts, and `packed + 2*m` bytes otherwise.
/// Checked `u64` math — `None` on overflow.
fn stream_twin_bytes(m: u64, k: u64) -> Option<u64> {
    let nb = k.div_ceil(32);
    let q = m.checked_mul(k.div_ceil(8))?.checked_mul(4)?;
    let d = m.checked_mul(nb.div_ceil(2))?.checked_mul(4)?;
    q.checked_add(d)
}

/// Estimated worst-case transient bytes for one GEMV `(m, k)` shape: synth
/// f32 weights (`m*k*4`), `raw` + stream twins on host and device
/// (`2*packed + 2*twins`; the twins match `packed` at even block counts),
/// `x` at 2× (host + device), and `y` at 3× (`expected` + downloaded
/// `got` + device). Checked `u64` math — `None` on overflow. Deliberately
/// an over-estimate: the cap is a typo guard, not a precise allocator
/// model.
fn gemv_bench_bytes(m: u32, k: u32) -> Option<u64> {
    let (m, k) = (u64::from(m), u64::from(k));
    let synth = m.checked_mul(k)?.checked_mul(4)?;
    let packed = m.checked_mul(k.div_ceil(32))?.checked_mul(18)?;
    let twins = stream_twin_bytes(m, k)?;
    let weights = synth
        .checked_add(packed.checked_mul(2)?)?
        .checked_add(twins.checked_mul(2)?)?;
    let x_io = k.checked_mul(4)?.checked_mul(2)?;
    let y_io = m.checked_mul(4)?.checked_mul(3)?;
    let io = x_io.checked_add(y_io)?;
    weights.checked_add(io)
}

/// Estimated worst-case transient bytes for one GEMM `(m, n, k)` shape:
/// synth f32 weights (`m*k*4`) plus host `raw` and stream twins on host
/// and device (`packed + 2*twins` — GEMM never uploads `raw`; the CPU
/// reference keeps it), B row-major f32 (`n*k*4`), the transposed f16 twin
/// (`k*n_pad*2`, host + device), and the outputs at 4× (`m*n_pad*4`:
/// `expected` + downloaded `got` + the `base_out` clone variants score
/// against + device). Checked `u64` math — `None` on overflow (including
/// the `n_pad` round-up).
fn gemm_bench_bytes(m: u32, n: u32, k: u32) -> Option<u64> {
    let (m, n, k) = (u64::from(m), u64::from(n), u64::from(k));
    let n_pad = n.checked_next_multiple_of(32)?;
    let synth = m.checked_mul(k)?.checked_mul(4)?;
    let packed = m.checked_mul(k.div_ceil(32))?.checked_mul(18)?;
    let twins = stream_twin_bytes(m, k)?;
    let weights = synth
        .checked_add(packed)?
        .checked_add(twins.checked_mul(2)?)?;
    let b = n.checked_mul(k)?.checked_mul(4)?;
    let b16 = k.checked_mul(n_pad)?.checked_mul(2)?.checked_mul(2)?;
    let out = m.checked_mul(n_pad)?.checked_mul(4)?.checked_mul(4)?;
    weights.checked_add(b)?.checked_add(b16)?.checked_add(out)
}

/// Validate one GEMV `(m, k)` bench shape: dims must be positive (a zero
/// dim reaches GPU dispatches parameterized with 0 — a device-loss-shaped
/// footgun), bounded (see the consts above), `k` a Q4_0 block multiple
/// (the quantizer asserts it), and the estimated transient within budget.
///
/// Called by the CLI after parsing AND at the `pub` harness entries below
/// (which a direct library caller reaches without the CLI) — the one home
/// for shape validation, before any context creation or allocation.
///
/// Internal CLI harness, not semver-stable.
#[doc(hidden)]
pub fn validate_gemv_shape(m: u32, k: u32) -> anyhow::Result<()> {
    anyhow::ensure!(
        m >= 1 && k >= 1,
        "m={m} k={k}: dims must be >= 1 (a zero dim reaches GPU dispatches)"
    );
    anyhow::ensure!(
        u64::from(m) <= BENCH_MAX_DIM && u64::from(k) <= BENCH_MAX_DIM,
        "m={m} k={k}: dims must be <= {BENCH_MAX_DIM}"
    );
    anyhow::ensure!(
        k.is_multiple_of(32),
        "m={m} k={k}: k must be a multiple of 32"
    );
    let bytes = gemv_bench_bytes(m, k)
        .ok_or_else(|| anyhow::anyhow!("m={m} k={k}: shape overflows u64 byte math"))?;
    anyhow::ensure!(
        bytes <= BENCH_MAX_BYTES,
        "m={m} k={k}: estimated transient {bytes} bytes exceeds the {BENCH_MAX_BYTES}-byte cap"
    );
    Ok(())
}

/// Validate one GEMM `(m, n, k)` bench shape: the GEMV bounds on m/k plus
/// n's own bounds (the kernel's fiber width needs `n >= 32`) and the
/// GEMM transient budget. Same two call sites as [`validate_gemv_shape`].
///
/// Internal CLI harness, not semver-stable.
#[doc(hidden)]
pub fn validate_gemm_shape(m: u32, n: u32, k: u32) -> anyhow::Result<()> {
    // Same legs as the GEMV validator, restated (not delegated) so every
    // message names the full `m n k` shape in one format.
    anyhow::ensure!(
        m >= 1 && n >= 1 && k >= 1,
        "m={m} n={n} k={k}: dims must be >= 1 (a zero dim reaches GPU dispatches)"
    );
    anyhow::ensure!(
        u64::from(m) <= BENCH_MAX_DIM
            && u64::from(n) <= BENCH_MAX_DIM
            && u64::from(k) <= BENCH_MAX_DIM,
        "m={m} n={n} k={k}: dims must be <= {BENCH_MAX_DIM}"
    );
    anyhow::ensure!(
        k.is_multiple_of(32),
        "m={m} n={n} k={k}: k must be a multiple of 32"
    );
    anyhow::ensure!(n >= 32, "m={m} n={n} k={k}: n must be >= 32 (fiber width)");
    let bytes = gemm_bench_bytes(m, n, k)
        .ok_or_else(|| anyhow::anyhow!("m={m} n={n} k={k}: shape overflows u64 byte math"))?;
    anyhow::ensure!(
        bytes <= BENCH_MAX_BYTES,
        "m={m} n={n} k={k}: estimated transient {bytes} bytes exceeds the {BENCH_MAX_BYTES}-byte cap"
    );
    Ok(())
}

/// Device-side Q4_0 GEMV microbench: `fast` (raw blocks) vs `stream`
/// (resident (q, d)) across `(m, k)` shapes. Synthetic weights (quantized
/// on host), `iters` timed dispatches per kernel in one submit, parity vs
/// a CPU dequant+dot reference. Prints ms/iter, effective GB/s, and max
/// abs diff per shape. The `stream` kernel needs SPIR-V passthrough and is
/// skipped without it; `fast` falls back to its WGSL twin there.
/// Behind the `cera gemv-bench` CLI (kernel iteration without full-model
/// runs); not part of any test suite.
///
/// Experimental variants (via `spv_paths`, each `path` or `path@nr`) must
/// keep the baseline's contract: same 4 bindings (w, x, y, params), same
/// params `[m, k, 0, 0]`, entry `main`. The grid adapts to the variant's
/// rows-per-workgroup (`@nr`, default 8), so NR variants A/B in one run.
///
/// Internal CLI harness, not semver-stable.
#[doc(hidden)]
pub fn gemv_q4_0_microbench(
    shapes: &[(u32, u32)],
    iters: u32,
    kernels: &[&str],
    spv_paths: &[String],
) -> anyhow::Result<()> {
    pollster::block_on(gemv_q4_0_microbench_async(
        shapes, iters, kernels, spv_paths,
    ))
}

async fn gemv_q4_0_microbench_async(
    shapes: &[(u32, u32)],
    iters: u32,
    kernels: &[&str],
    spv_paths: &[String],
) -> anyhow::Result<()> {
    use crate::backend::wgpu::shaders;
    use anyhow::Context;

    // Validate before touching the GPU: the `pub` sync wrapper below
    // delegates here, so a direct library caller reaches this without the
    // CLI's pre-validation (shapes, iters, and kernel selection all have
    // host-side checks here).
    anyhow::ensure!(iters >= 1, "iters must be >= 1");
    anyhow::ensure!(
        kernels.contains(&"fast") || kernels.contains(&"stream"),
        "kernels must select at least one of fast,stream (got {kernels:?})"
    );
    for &(m, k) in shapes {
        validate_gemv_shape(m, k)?;
    }

    let ctx = GpuContext::new_async().await?;
    let passthrough = ctx.supports_spirv_passthrough() && ctx.has_subgroup;
    println!(
        "gemv-bench: backend={} passthrough={}",
        ctx.adapter_name, passthrough
    );
    let fast_pipe = if passthrough {
        ctx.gemv_q4_0_fast_passthrough()
    } else {
        ctx.create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast", "gemv_q4_0_fast")
    };
    let stream_pipe = if passthrough {
        Some(ctx.gemv_q4_0_stream_passthrough())
    } else {
        None
    };

    // Experimental variants: same bind-group layout as `fast` (contract in
    // the doc comment), module swapped for the file's SPIR-V. Each spec is
    // `path` or `path@nr` (rows-per-workgroup, default 8).
    if !spv_paths.is_empty() {
        anyhow::ensure!(
            passthrough,
            "--spv variants need SPIR-V passthrough (Vulkan + subgroups)"
        );
    }
    let mut variants: Vec<(String, wgpu::ComputePipeline, u32)> = Vec::new();
    for spec in spv_paths {
        let (path, nr) = match spec.rsplit_once('@') {
            Some((p, n)) => {
                let nr: u32 = n
                    .parse()
                    .with_context(|| format!("{spec}: bad @nr (want path or path@nr)"))?;
                anyhow::ensure!(nr >= 1, "{spec}: @nr must be >= 1");
                (p, nr)
            }
            None => (spec.as_str(), 8),
        };
        let (stem, pipe) = load_spv_variant(&ctx, &fast_pipe, path)?;
        variants.push((stem, pipe, nr));
    }

    for (shape_idx, &(m, k)) in shapes.iter().enumerate() {
        // Validated at entry (`4*m*k <= BENCH_MAX_BYTES`, so `m*k <= 2^29`):
        // every product below fits `usize` on 64- and 32-bit, and every
        // allocation is inside the transient budget.
        let (m_us, k_us) = (m as usize, k as usize);
        let weights = synth_q4_0_vec(m_us * k_us, SYNTH_SALT_WEIGHTS);
        let raw = quantize_q4_0_synth(&weights, m_us, k_us);
        let (q, d) = repack_q4_0_stream(&raw, m_us, k_us);
        let x = synth_q4_0_vec(k_us, SYNTH_SALT_INPUTS);
        let expected = cpu_gemv_q4_0_ref(&raw, &x, m_us, k_us);

        let raw_buf = ctx.upload_storage(&raw, "bench.raw");
        let q_buf = ctx.upload_storage(bytemuck::cast_slice(&q), "bench.q");
        let d_buf = ctx.upload_storage(bytemuck::cast_slice(&d), "bench.d");
        let x_buf = ctx.upload_f32(&x, "bench.x");
        let y_buf = ctx.create_storage_rw((m as u64) * 4, "bench.y");
        let params_buf =
            ctx.upload_storage(bytemuck::cast_slice(&[m, k, 0u32, 0u32]), "bench.params");
        let mk_bg = |pipe: &wgpu::ComputePipeline, entries: Vec<wgpu::BindGroupEntry>| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        // The 4-entry `fast` contract, shared by `--spv` variants (each
        // needs its own bind group against its own pipeline layout).
        let fast_entries = || {
            vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ]
        };
        let fast_bg = mk_bg(&fast_pipe, fast_entries());
        let stream_bg = stream_pipe.as_ref().map(|pipe| {
            mk_bg(
                pipe,
                vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: d_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: y_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            )
        });

        // rows-per-WG must match each kernel (fast NR=8, stream ROWS_PER_WG=16).
        let run = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, rows_per_wg: u32, n: u32| {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            let grid = (m.div_ceil(rows_per_wg), 1, 1);
            for _ in 0..n {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(grid.0, grid.1, grid.2);
            }
            let t = std::time::Instant::now();
            ctx.submit_encoder(enc);
            ctx.device.poll_wait();
            t.elapsed()
        };

        let bytes_per_iter = (m as u64) * (k as u64) / 2; // Q4_0 weight bytes
        println!("shape m={m} k={k} ({:.2} MB)", bytes_per_iter as f64 / 1e6);
        let variant_bgs: Vec<wgpu::BindGroup> = variants
            .iter()
            .map(|(_, pipe, _)| mk_bg(pipe, fast_entries()))
            .collect();
        let mut cases: Vec<(&str, &wgpu::ComputePipeline, &wgpu::BindGroup, u32)> = Vec::new();
        if kernels.contains(&"fast") {
            cases.push(("fast", &fast_pipe, &fast_bg, 8));
        }
        if kernels.contains(&"stream") {
            if let (Some(pipe), Some(bg)) = (stream_pipe.as_ref(), stream_bg.as_ref()) {
                cases.push(("stream", pipe, bg, 16));
            } else {
                println!("  stream: skipped (no passthrough)");
            }
        }
        for ((stem, pipe, nr), bg) in variants.iter().zip(variant_bgs.iter()) {
            cases.push((stem.as_str(), pipe, bg, *nr));
        }
        for (case_idx, (name, pipe, bg, rows_per_wg)) in cases.iter().enumerate() {
            // The first case of the process pays one-time init (pipeline /
            // driver setup) even after the soak below — up to 3x slow once,
            // then stable. Run it twice, keep the second.
            let rounds = if shape_idx == 0 && case_idx == 0 {
                2
            } else {
                1
            };
            let (mut ms, mut gbps, mut max_diff, mut mhz) = (0.0, 0.0, 0.0f32, "n/a".to_string());
            for _ in 0..rounds {
                let (ms_new, mhz_new) = soak_and_measure(|n| run(pipe, bg, *rows_per_wg, n), iters);
                ms = ms_new;
                mhz = mhz_new;
                gbps = bytes_per_iter as f64 / (ms / 1e3) / 1e9;
                let got = ctx.download_f32_async(&y_buf, m_us).await?;
                max_diff = expected
                    .iter()
                    .zip(got.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
            }
            println!(
                "  {name}: {ms:.3} ms/iter, {gbps:.1} GB/s, maxdiff={max_diff:.3e} gpuclk={mhz}"
            );
        }
    }
    Ok(())
}

/// Microbenchmark the Q4_0 prefill GEMM (`gemm_stream_q4_0`) plus optional
/// experimental SPIR-V variants loaded from disk.
///
/// Synthetic weights, no model file: times `iters` dispatches per kernel per
/// shape in one submit and checks parity against a CPU reference. B is
/// transposed + cast to f16 on the host, exactly as `transpose_cast_f16`
/// would lay it out ([k][n_pad] halves, zero-padded past n).
///
/// Experimental variants (via `spv_paths`) must keep the baseline's contract:
/// same 5 bindings (q, d, b16, dst, params), same grid
/// `(m/256, n_pad/32)`, same `[numthreads(256,1,1)]`, same params
/// `[m, k, n_valid, n_pad, y_stride]`, entry `main`.
///
/// Internal CLI harness, not semver-stable.
#[doc(hidden)]
pub fn gemm_q4_0_microbench(
    shapes: &[(u32, u32, u32)],
    iters: u32,
    spv_paths: &[String],
    spv_ny: u32,
) -> anyhow::Result<()> {
    pollster::block_on(gemm_q4_0_microbench_async(shapes, iters, spv_paths, spv_ny))
}

async fn gemm_q4_0_microbench_async(
    shapes: &[(u32, u32, u32)],
    iters: u32,
    spv_paths: &[String],
    spv_ny: u32,
) -> anyhow::Result<()> {
    // Validate before touching the GPU (see the GEMV entry): shapes, iters,
    // and the variant fiber width (`div_ceil(ny)` below panics on 0).
    anyhow::ensure!(iters >= 1, "iters must be >= 1");
    anyhow::ensure!(
        spv_ny == 32 || spv_ny == 64,
        "spv_ny must be 32 or 64 (got {spv_ny})"
    );
    for &(m, n, k) in shapes {
        validate_gemm_shape(m, n, k)?;
    }

    let ctx = GpuContext::new_async().await?;
    let passthrough = ctx.supports_spirv_passthrough() && ctx.has_subgroup;
    println!(
        "gemm-bench: backend={} passthrough={}",
        ctx.adapter_name, passthrough
    );
    anyhow::ensure!(
        passthrough,
        "gemm-bench needs SPIR-V passthrough (Vulkan + subgroups)"
    );
    let base_pipe = ctx.gemm_stream_q4_0_passthrough();

    // Experimental variants: same bind-group layout as the baseline (contract
    // above), module swapped for the file's SPIR-V.
    let mut variants: Vec<(String, wgpu::ComputePipeline)> = Vec::new();
    for path in spv_paths {
        variants.push(load_spv_variant(&ctx, &base_pipe, path)?);
    }

    for (shape_idx, &(m, n, k)) in shapes.iter().enumerate() {
        // Validated at entry (dims, `k % 32`, `n >= 32`, transient budget),
        // so every product below fits `usize` on 64- and 32-bit and every
        // allocation is inside the budget. The `checked_` pad stays as the
        // computation itself, not as validation.
        let (m_us, n_us, k_us) = (m as usize, n as usize, k as usize);
        let n_pad = n
            .checked_next_multiple_of(32)
            .ok_or_else(|| anyhow::anyhow!("n={n} overflows u32 when padded to 32"))?
            as usize;
        let weights = synth_q4_0_vec(m_us * k_us, SYNTH_SALT_WEIGHTS);
        let raw = quantize_q4_0_synth(&weights, m_us, k_us);
        let (q, d) = repack_q4_0_stream(&raw, m_us, k_us);
        // B row-major f32 [n][k], then host transpose+cast to [k][n_pad] f16.
        let b = synth_q4_0_vec(n_us * k_us, SYNTH_SALT_INPUTS);
        let mut b16 = vec![0u16; k_us * n_pad];
        for kk in 0..k_us {
            for nn in 0..n_us {
                b16[kk * n_pad + nn] = half::f16::from_f32(b[nn * k_us + kk]).to_bits();
            }
        }
        let expected = cpu_gemm_q4_0_ref(&raw, &b, m_us, n_us, k_us);

        let q_buf = ctx.upload_storage(bytemuck::cast_slice(&q), "bench.q");
        let d_buf = ctx.upload_storage(bytemuck::cast_slice(&d), "bench.d");
        let b_buf = ctx.upload_storage(bytemuck::cast_slice(&b16), "bench.b16");
        let y_buf = ctx.create_storage_rw((n_pad as u64) * (m as u64) * 4, "bench.y");
        let params_buf = ctx.upload_storage(
            bytemuck::cast_slice(&[m, k, n, n_pad as u32, m]),
            "bench.params",
        );
        let mk_bg = |entries: Vec<wgpu::BindGroupEntry>| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &base_pipe.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let bg = mk_bg(vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: d_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ]);

        // One pass, `it` back-to-back dispatches — matches production, where
        // GEMMs share passes, so no inter-pass barrier bubble pollutes timing.
        // `ny` is the variant's fiber width (baseline always 32).
        let run = |pipe: &wgpu::ComputePipeline, it: u32, ny: u32| {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            let grid = (m.div_ceil(256), (n_pad as u32).div_ceil(ny), 1);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..it {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
            }
            let t = std::time::Instant::now();
            ctx.submit_encoder(enc);
            ctx.device.poll_wait();
            t.elapsed()
        };

        let flops_per_iter = 2.0 * m as f64 * n as f64 * k as f64;
        println!(
            "shape m={m} n={n} k={k} ({:.2} GFLOP/iter)",
            flops_per_iter / 1e9
        );
        let mut cases: Vec<(&str, &wgpu::ComputePipeline)> = vec![("stream", &base_pipe)];
        for (name, pipe) in &variants {
            cases.push((name, pipe));
        }
        // f16 fiber accumulation over k=2048 with unit-scale synthetic inputs
        // genuinely disagrees with the f32 CPU reference by O(10) (same for
        // the shipped kernel — production parity is greedy-identical
        // generation, not this check). So variants are scored against the
        // baseline's own output; the CPU diff is a sanity anchor, not a gate.
        let mut base_out: Vec<f32> = Vec::new();
        for (idx, (name, pipe)) in cases.iter().enumerate() {
            let ny = if idx == 0 { 32 } else { spv_ny };
            // First case of the process pays one-time init even after the
            // soak (see gemv-bench) — run it twice, keep the second.
            let rounds = if shape_idx == 0 && idx == 0 { 2 } else { 1 };
            let (mut ms, mut tflops, mut mhz, mut got) = (0.0, 0.0, "n/a".to_string(), Vec::new());
            for _ in 0..rounds {
                let (ms_new, mhz_new) = soak_and_measure(|n| run(pipe, n, ny), iters);
                ms = ms_new;
                mhz = mhz_new;
                tflops = flops_per_iter / (ms / 1e3) / 1e12;
                got = ctx.download_f32_async(&y_buf, n_pad * m_us).await?;
            }
            if idx == 0 {
                base_out = got.clone();
            }
            // Packed columns [n][m]; ignore the [n, n_pad) padding.
            let mut max_diff = 0.0f32;
            let mut max_cpu = 0.0f32;
            for c in 0..n_us {
                for r in 0..m_us {
                    let i = c * m_us + r;
                    max_diff = max_diff.max((base_out[i] - got[i]).abs());
                    max_cpu = max_cpu.max((expected[i] - got[i]).abs());
                }
            }
            if idx == 0 {
                println!(
                    "  {name}: {ms:.3} ms/iter, {tflops:.2} TFLOP/s, cpu_diff={max_cpu:.3e} gpuclk={mhz}"
                );
            } else {
                println!(
                    "  {name}: {ms:.3} ms/iter, {tflops:.2} TFLOP/s, vs_base={max_diff:.3e} cpu_diff={max_cpu:.3e} gpuclk={mhz}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use crate::backend::wgpu::{DevicePollExt, GpuContext};

    /// `--spv` bytes are validated before reaching the driver: word
    /// alignment, the size cap, and the 5-word header shape each fail with
    /// the file named. Pure host check — runs without a GPU.
    #[test]
    fn spv_bytes_reject_mistyped_paths() {
        use super::{SPV_MAGIC, SPV_MAX_BYTES, check_spv_bytes, check_spv_size};
        // Minimal valid header: magic, 1.0, generator 0, bound 1, schema 0.
        let words = [SPV_MAGIC, 0x0001_0000, 0, 1, 0];
        let mut good = vec![0u8; 20];
        for (i, w) in words.iter().enumerate() {
            good[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(check_spv_bytes("k.spv", &good).unwrap(), words.to_vec());
        // Text file: word-aligned but wrong magic.
        let err = check_spv_bytes("note.txt", b"helloworld12").unwrap_err();
        assert!(err.to_string().contains("bad SPIR-V magic"), "{err}");
        assert!(err.to_string().contains("note.txt"), "{err}");
        // Truncated binary: not a multiple of 4.
        let err = check_spv_bytes("cut.spv", &[0u8; 7]).unwrap_err();
        assert!(err.to_string().contains("not a multiple of 4"), "{err}");
        // Empty file: aligned, but no magic word.
        let err = check_spv_bytes("empty.spv", &[]).unwrap_err();
        assert!(err.to_string().contains("bad SPIR-V magic"), "{err}");
        // Magic but too short for the 5-word header.
        let mut short = vec![0u8; 12];
        short[0..4].copy_from_slice(&SPV_MAGIC.to_le_bytes());
        let err = check_spv_bytes("short.spv", &short).unwrap_err();
        assert!(err.to_string().contains("too short"), "{err}");
        // Header legs, each broken alone: version major != 1, bound 0,
        // schema != 0.
        for (i, word, leg) in [
            (1, 0x0002_0000u32, "version"),
            (3, 0u32, "bound"),
            (4, 1u32, "schema"),
        ] {
            let mut bad = good.clone();
            bad[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
            let err = check_spv_bytes("bad.spv", &bad).unwrap_err();
            assert!(err.to_string().contains(leg), "{leg}: {err}");
        }
        // Exact cap boundary, pinned without a 64 MiB vec.
        assert!(check_spv_size(SPV_MAX_BYTES).is_ok());
        let err = check_spv_size(SPV_MAX_BYTES + 4).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    /// Bench shape validation: zero dims, misaligned `k`/`n` legs, giant
    /// dims, and the byte-budget boundary each fail with the shape named.
    /// Pure host checks — run without a GPU.
    #[test]
    fn bench_shapes_reject_zero_giant_and_misaligned() {
        use super::{validate_gemm_shape, validate_gemv_shape};
        // Zero dims reach GPU dispatches: rejected on every leg.
        assert!(validate_gemv_shape(0, 64).is_err());
        assert!(validate_gemv_shape(64, 0).is_err());
        assert!(validate_gemm_shape(0, 64, 64).is_err());
        assert!(validate_gemm_shape(64, 0, 64).is_err());
        assert!(validate_gemm_shape(64, 64, 0).is_err());
        // Misaligned legs: k must be a Q4_0 block multiple, n the fiber width.
        assert!(validate_gemv_shape(64, 100).is_err());
        assert!(validate_gemm_shape(64, 16, 64).is_err());
        // Giant dims: over the dim cap, or inside it but over the byte budget.
        assert!(validate_gemv_shape(1 << 24, 1 << 24).is_err());
        // Dim-cap legs: the GEMV leg gets a partner value that passes
        // every later gate (multiple of 32, ~239 MB against the 2 GiB
        // budget), so deleting the leg flips exactly this assert. No
        // GEMM partner exists (any over-cap dim forces the byte budget
        // over: each trip region lower-bounds one transient term above
        // the cap), so the GEMM leg pins the message instead.
        assert!(validate_gemv_shape(1, (1 << 24) + 32).is_err());
        let err = validate_gemm_shape(64, u32::MAX - 1, 64).unwrap_err();
        assert!(err.to_string().contains("dims must be <="), "{err}");
        assert!(validate_gemm_shape(1 << 20, 1 << 20, 64).is_err());
        // The hole a pure product cap admits: (65536, 65536) needs ~16 GiB
        // of synth weights alone — must fail with a message, not abort in
        // the allocator.
        let err = validate_gemv_shape(65536, 65536).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
        let err = validate_gemm_shape(65536, 64, 65536).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
        // Sane shapes pass.
        assert!(validate_gemv_shape(256, 1024).is_ok());
        assert!(validate_gemm_shape(256, 64, 1024).is_ok());
        // The CLI's own default `gemv-bench` shapes must validate,
        // including the 128000x2048 LM-head row (~1.53 GiB estimated).
        for (m, k) in [
            (512, 2048),
            (2048, 2048),
            (6144, 2048),
            (2048, 6144),
            (10752, 2048),
            (2048, 10752),
            (128000, 2048),
        ] {
            assert!(
                validate_gemv_shape(m, k).is_ok(),
                "default shape {m},{k} rejected"
            );
        }
        // The CLI's own default `gemm-bench` shapes must validate too
        // (mirrors the `GemmBench` clap `default_value`).
        for (m, n, k) in [
            (10752, 128, 2048),
            (10752, 512, 2048),
            (2048, 128, 2048),
            (2048, 512, 2048),
            (6144, 512, 2048),
        ] {
            assert!(
                validate_gemm_shape(m, n, k).is_ok(),
                "default shape {m},{n},{k} rejected"
            );
        }
    }

    /// The Adreno kind table, pinned host-side: every `mul_mat_*` label
    /// `encode_mul_mat_reg_tile` emits (including `mul_mat_f32`, which has
    /// no SPIR-V twin) splits into its own passes past 128 tokens, while
    /// the proven-safe mixers and ordinary batch labels merge. The
    /// pass-count integration test only proves *a* split exists, so a
    /// narrowing edit that keeps its fixture green would silently change
    /// on-device grouping, so this table pins the contract instead. Do not
    /// reclassify without the 2.6B n>128 Adreno soak.
    #[test]
    fn adreno_split_label_table() {
        use super::GpuLfm2Model;
        for label in [
            "mul_mat_tile",
            "mul_mat_tile_stream",
            "mul_mat_q8_0",
            "mul_mat_q4k",
            "mul_mat_q5k",
            "mul_mat_q6k",
            "mul_mat_f32",
        ] {
            assert!(
                GpuLfm2Model::is_adreno_split_label(label),
                "{label} must stay in the Adreno-split kind"
            );
        }
        for label in [
            "gemm_stream_q4_0",
            "gemm_stream_q4_0_k64",
            "transpose_cast_f16",
            "rmsnorm_batch",
            "attention_prefill",
            "kv_append",
        ] {
            assert!(
                !GpuLfm2Model::is_adreno_split_label(label),
                "{label} must stay out of the Adreno-split kind"
            );
        }
    }

    /// The transient-byte estimators, pinned by value: every live buffer
    /// the harnesses hold at peak (host twins, downloads, the `base_out`
    /// clone, device buffers) must be counted, or the typo-guard cap
    /// admits shapes that abort in the allocator. Both block-count
    /// parities are pinned: the `d` twin's padded lane makes odd counts
    /// slightly larger than `packed`.
    #[test]
    fn bench_byte_estimators_count_every_live_buffer() {
        use super::{gemm_bench_bytes, gemv_bench_bytes, stream_twin_bytes};
        // Twins at odd nb (k=32, nb=1): q 32*4*4 = 512, d 32*1*4 = 128 —
        // 640 vs packed 576 (the padded `d` lane).
        assert_eq!(stream_twin_bytes(32, 32), Some(512 + 128));
        // Twins at even nb (k=64, nb=2): q 32*8*4 = 1024, d 32*1*4 = 128 —
        // exactly packed (1152).
        assert_eq!(stream_twin_bytes(32, 64), Some(1024 + 128));
        // GEMV (32, 32): synth 4096; raw 576 host + device; twins 640
        // host + device; x 32*4 at 2×; y 32*4 at 3× (expected + got +
        // device).
        assert_eq!(
            gemv_bench_bytes(32, 32),
            Some(4096 + 2 * 576 + 2 * 640 + 2 * 128 + 3 * 128)
        );
        // GEMV (32, 64): synth 8192; raw and twins 1152 each, host +
        // device; x 64*4 at 2×; y 32*4 at 3×.
        assert_eq!(
            gemv_bench_bytes(32, 64),
            Some(8192 + 2 * 1152 + 2 * 1152 + 2 * 256 + 3 * 128)
        );
        // GEMM (32, 32, 32): synth 4096; host raw 576 (never uploaded);
        // twins 640 host + device; B 32*32*4; b16 32*32*2 at 2×; out
        // 32*32*4 at 4× (expected + got + base_out + device).
        assert_eq!(
            gemm_bench_bytes(32, 32, 32),
            Some(4096 + 576 + 2 * 640 + 4096 + 2 * 2048 + 4 * 4096)
        );
        // Genuine u64 overflow (here the synth product) is None, not
        // wrap. (Merely huge shapes like n = u32::MAX - 1 return a huge
        // estimate and fail at the budget check instead.)
        assert_eq!(gemv_bench_bytes(u32::MAX, u32::MAX), None);
    }

    /// The `pub` harness entries validate on the host before touching the
    /// GPU: bad shapes, `iters == 0`, an empty kernel selection, and a bad
    /// `spv_ny` all fail here, not on a device. (No GPU needed — validation
    /// precedes context creation.)
    #[test]
    fn microbench_entries_reject_bad_inputs_without_gpu() {
        use super::{gemm_q4_0_microbench, gemv_q4_0_microbench};
        // Assert the validation message, not mere `is_err()`: both entries
        // create the GPU context after validation, so on a GPU-less host
        // every input errors at context creation and bare `is_err()`
        // passes vacuously. The message pins validation-first order too
        // (a reorder surfaces the context error instead).
        let err = gemv_q4_0_microbench(&[(0, 64)], 1, &["fast"], &[]).unwrap_err();
        assert!(err.to_string().contains("dims must be"), "{err:?}");
        let err = gemv_q4_0_microbench(&[(65536, 65536)], 1, &["fast"], &[]).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err:?}");
        let err = gemv_q4_0_microbench(&[(64, 64)], 0, &["fast"], &[]).unwrap_err();
        assert!(err.to_string().contains("iters must be"), "{err:?}");
        let err = gemv_q4_0_microbench(&[(64, 64)], 1, &[], &[]).unwrap_err();
        assert!(err.to_string().contains("kernels must select"), "{err:?}");
        let err = gemm_q4_0_microbench(&[(64, 16, 64)], 1, &[], 32).unwrap_err();
        assert!(err.to_string().contains("n must be"), "{err:?}");
        let err = gemm_q4_0_microbench(&[(64, 64, 64)], 1, &[], 0).unwrap_err();
        assert!(err.to_string().contains("spv_ny must be"), "{err:?}");
    }

    /// `soak_and_measure` ms/iter math, pinned host-side with a fake `run`
    /// (no GPU: the closure never touches one). Takes ~`SOAK_MS` wall.
    #[test]
    fn soak_and_measure_reports_ms_per_iter() {
        use super::soak_and_measure;
        let (ms, mhz) = soak_and_measure(|n| std::time::Duration::from_millis(10 * n as u64), 4);
        assert!((ms - 10.0).abs() < 1e-6, "ms/iter math: {ms}");
        // Pin the frequency-suffix contract, not mere non-emptiness (both
        // branches of the helper produce non-empty strings, so `is_empty`
        // could never fail): `n/a` off-Android, `{f}MHz` on it.
        assert!(
            mhz == "n/a" || mhz.ends_with("MHz"),
            "freq suffix contract: {mhz}"
        );
    }

    /// `repack_q6_k_flat` must de-interleave every block into its plane at
    /// the offsets the flat kernel indexes: `ql[(row*nb+b)*128 + i]`,
    /// `qh[…*64 + i]`, `scales[…*16 + i]`, `d[…*2 .. +2]`, planes packed
    /// back to back. Pure host math — runs without a GPU.
    #[test]
    fn flat_repack_q6_k_layout() {
        use super::repack_q6_k_flat;
        let (m, k) = (3usize, 512usize); // nb = 2 blocks/row
        let nb = 2usize;
        let mut data = vec![0u8; m * nb * 210];
        for row in 0..m {
            for blk in 0..nb {
                let base = (row * nb + blk) * 210;
                let tag = (row * nb + blk) as u8;
                for i in 0..128 {
                    data[base + i] = tag.wrapping_add(i as u8);
                }
                for i in 0..64 {
                    data[base + 128 + i] = tag.wrapping_add(0x40).wrapping_add(i as u8);
                }
                for i in 0..16 {
                    data[base + 192 + i] = tag.wrapping_add(0x80).wrapping_add(i as u8);
                }
                data[base + 208] = tag.wrapping_add(0xC0);
                data[base + 209] = tag.wrapping_add(0xE0);
            }
        }
        let out = repack_q6_k_flat(&data, m, k);
        assert_eq!(out.len(), m * nb * 210);
        let (ql, rest) = out.split_at(m * nb * 128);
        let (qh, rest) = rest.split_at(m * nb * 64);
        let (s, d) = rest.split_at(m * nb * 16);
        assert_eq!(d.len(), m * nb * 2);
        for row in 0..m {
            for blk in 0..nb {
                let base = (row * nb + blk) * 210;
                let o = row * nb + blk;
                assert_eq!(&ql[o * 128..(o + 1) * 128], &data[base..base + 128]);
                assert_eq!(&qh[o * 64..(o + 1) * 64], &data[base + 128..base + 192]);
                assert_eq!(&s[o * 16..(o + 1) * 16], &data[base + 192..base + 208]);
                assert_eq!(&d[o * 2..(o + 1) * 2], &data[base + 208..base + 210]);
            }
        }
    }

    /// `repack_q4_0_stream` must place every nibble and scale where the
    /// streaming kernel indexes it: `q[(kb*4+p)*m + row]` holds weights
    /// `kb*32+p*8+s*4+j` (s = u32 lane, j = nibble), `d[(kb/2)*m + row]`
    /// holds scales `2p` (low) and `2p+1` (high). Pure host math — runs
    /// without a GPU.
    #[test]
    fn stream_repack_q4_0_layout() {
        use super::repack_q4_0_stream;
        let (m, k) = (3usize, 64usize);
        // 2 blocks/row; weight w of (row, blk) gets nibble (row + blk*32 + w) % 16.
        let mut data = vec![0u8; m * 2 * 18];
        let mut scale_bits = vec![0u16; m * 2];
        for row in 0..m {
            for blk in 0..2 {
                let base = (row * 2 + blk) * 18;
                let sb = 0x3C00 + (row * 2 + blk) as u16; // distinct f16 bits
                scale_bits[row * 2 + blk] = sb;
                data[base..base + 2].copy_from_slice(&sb.to_le_bytes());
                for w in 0..32 {
                    let nib = ((row * 64 + blk * 32 + w) % 16) as u8;
                    if w < 16 {
                        data[base + 2 + w] |= nib;
                    } else {
                        data[base + 2 + w - 16] |= nib << 4;
                    }
                }
            }
        }
        let (q, d) = repack_q4_0_stream(&data, m, k);
        assert_eq!(q.len(), m * k / 8);
        assert_eq!(d.len(), m); // 2 blocks -> 1 scale pair per row
        // Scales: pair holds block 2p (low) and 2p+1 (high).
        for row in 0..m {
            assert_eq!(d[row] & 0xFFFF, scale_bits[row * 2] as u32, "row {row} lo");
            assert_eq!(d[row] >> 16, scale_bits[row * 2 + 1] as u32, "row {row} hi");
        }
        // Nibbles through the kernel's own indexing.
        for row in 0..m {
            for kb in 0..2 {
                for p in 0..4 {
                    let word = q[(kb * 4 + p) * m + row];
                    for s in 0..2 {
                        let w4 = if s == 0 { word & 0xFFFF } else { word >> 16 };
                        for j in 0..4 {
                            let got = (w4 >> (j * 4)) & 0xF;
                            let w = kb * 32 + p * 8 + s * 4 + j;
                            let want = ((row * 64 + w) % 16) as u32;
                            assert_eq!(got, want, "row {row} kb {kb} p {p} s {s} j {j} (w={w})");
                        }
                    }
                }
            }
        }
    }

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

    /// GPU context gated on SPIR-V passthrough, or `None` to skip.
    /// Fail-closed under `CERA_REQUIRE_*`, like every passthrough test.
    fn passthrough_ctx_or_skip(label: &str) -> Option<GpuContext> {
        let ctx = gpu_ctx_or_skip()?;
        let passthrough = ctx.supports_spirv_passthrough() && ctx.has_subgroup;
        crate::backend::wgpu::require_passthrough_or_skip(&ctx, passthrough, label);
        if !passthrough {
            return None;
        }
        Some(ctx)
    }

    /// Bind the standard stream-kernel `(q, d, x, y, params)` group at
    /// bindings 0-4. Shared by the stream-kernel tests so a layout fix
    /// lands once. Binding 2 is the dense RHS: vector `x` for GEMV/reg-tile,
    /// B/B16 for GEMM.
    fn bind_stream_qdxyp(
        ctx: &GpuContext,
        pipe: &wgpu::ComputePipeline,
        q: &wgpu::Buffer,
        d: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        y: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Max absolute elementwise difference over flat slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
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

    /// Resident stream layout eligibility: Q4_0 with `k % 32 == 0` (the
    /// repack's precondition). There is no size gate — the repack is the
    /// same bytes transposed, so small and large models alike qualify.
    #[test]
    fn stream_layout_eligibility() {
        use super::stream_layout_eligible;
        use crate::tensor::DType;
        assert!(stream_layout_eligible(DType::Q4_0, 32));
        assert!(stream_layout_eligible(DType::Q4_0, 2048));
        assert!(stream_layout_eligible(DType::Q4_0, 10752));
        assert!(!stream_layout_eligible(DType::Q4_0, 100));
        assert!(!stream_layout_eligible(DType::Q4_0, 2056));
        assert!(!stream_layout_eligible(DType::Q8_0, 2048));
        assert!(!stream_layout_eligible(DType::Q4KM, 2048));
        assert!(!stream_layout_eligible(DType::F32, 2048));
    }

    /// The input-embedding gather reads rows from the mmap'd table
    /// (`MmapWeight::dequantize_row`) instead of a pre-dequantized f32 host
    /// copy: the rows must be bit-identical to `to_f32_vec` slices, or every
    /// GPU prefill/decode drifts from the old path. Needs the 230M model
    /// locally; skips without it. Pure host math — runs without a GPU.
    #[test]
    fn embedding_gather_matches_f32_table() {
        use super::MmapWeight;
        let home = std::env::var("HOME").expect("HOME unset");
        let path = std::path::PathBuf::from(home)
            .join(".leap/models/LFM2.5-230M-Q4_0/LFM2.5-230M-Q4_0.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let gguf = std::sync::Arc::new(crate::gguf::GgufFile::open(&path).unwrap());
        let table = MmapWeight::from_gguf(&gguf, "token_embd.weight").unwrap();
        let full = gguf.get_tensor("token_embd.weight").unwrap().to_f32_vec();
        let (vocab, hs) = (table.rows, table.cols);
        assert_eq!(full.len(), vocab * hs);
        let mut row = vec![0.0f32; hs];
        for &t in &[0usize, 1, 7, 42, vocab / 2, vocab - 1] {
            table.dequantize_row(t, &mut row);
            assert_eq!(
                row,
                full[t * hs..(t + 1) * hs],
                "row {t} differs from the f32 table"
            );
        }
    }

    /// Same contract for the untied logit projection: the F16 head upload
    /// converts `output.weight` row by row from the mmap, so those rows must
    /// be bit-identical to `to_f32_vec` slices. Needs the TinyStories model
    /// locally; skips without it. Pure host math — runs without a GPU.
    #[test]
    fn untied_head_gather_matches_f32_table() {
        use super::MmapWeight;
        let home = std::env::var("HOME").expect("HOME unset");
        let path = std::path::PathBuf::from(home)
            .join(".leap/models/TinyStories-LLaMA2-20M-GQA.Q8_0.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let gguf = std::sync::Arc::new(crate::gguf::GgufFile::open(&path).unwrap());
        assert!(
            gguf.tensors.contains_key("output.weight"),
            "test model must be untied"
        );
        let table = MmapWeight::from_gguf(&gguf, "output.weight").unwrap();
        let full = gguf.get_tensor("output.weight").unwrap().to_f32_vec();
        let (vocab, hs) = (table.rows, table.cols);
        assert_eq!(full.len(), vocab * hs);
        let mut row = vec![0.0f32; hs];
        for &t in &[0usize, 1, 7, 42, vocab / 2, vocab - 1] {
            table.dequantize_row(t, &mut row);
            assert_eq!(
                row,
                full[t * hs..(t + 1) * hs],
                "row {t} differs from the f32 table"
            );
        }
    }

    /// `upload_mmap_table_as_f16` uploads every row's f16 conversion at the
    /// right offsets — including a partial trailing chunk. Synthetic Q8_0
    /// table (257 rows: one full 256-row write + 1 leftover), device
    /// round-trip against a plain-Rust reference. Skips without a wgpu
    /// adapter (hard-fails under `CERA_REQUIRE_GPU`, like the oracles).
    #[test]
    fn f16_table_upload_round_trip() {
        use super::{MmapWeight, upload_mmap_table_as_f16};
        let Some(ctx) = gpu_ctx_or_skip() else {
            return;
        };
        const ROWS: usize = 257;
        const COLS: usize = 64;
        // 2 Q8_0 blocks/row; coprime strides so a wrong row/block base or a
        // dropped trailing chunk cannot cancel out in the reference.
        let mut raw = Vec::with_capacity(ROWS * 2 * 34);
        let mut expected = Vec::with_capacity(ROWS * COLS);
        for r in 0..ROWS {
            for b in 0..2 {
                let scale = 0.01 + ((r * 3 + b * 11) % 97) as f32 * 0.001;
                let delta = half::f16::from_f32(scale);
                raw.extend_from_slice(&delta.to_bits().to_le_bytes());
                for c in 0..32 {
                    let q = ((r * 7 + b * 13 + c * 5) % 251) as i8;
                    raw.push(q as u8);
                    let v = q as f32 * crate::quant::f16_to_f32(delta.to_bits());
                    expected.push(half::f16::from_f32(v).to_bits());
                }
            }
        }
        let table = MmapWeight::from_owned_bytes(raw, crate::tensor::DType::Q8_0, ROWS, COLS);
        let buf = upload_mmap_table_as_f16(&ctx, &table, "test_f16_table");
        let got_f32 = ctx.download_f16_as_f32(&buf, ROWS * COLS);
        assert_eq!(got_f32.len(), expected.len());
        for (i, (&got, &exp)) in got_f32.iter().zip(expected.iter()).enumerate() {
            // f16→f32→f16 is the identity on finite values, so a bits
            // comparison pins the upload plumbing exactly.
            assert_eq!(half::f16::from_f32(got).to_bits(), exp, "elem {i} differs");
        }
    }

    /// The resident-stream decode kernel (`gemv_q4_0_stream`) matches the
    /// CPU dequant+dot reference on synthetic Q4_0: exact-multiple rows plus
    /// a ragged row count (exercises the per-row guards). f32 accumulation,
    /// so the tolerance only absorbs summation-order differences.
    #[test]
    fn stream_gemv_matches_cpu() {
        use super::{
            SYNTH_SALT_INPUTS, SYNTH_SALT_WEIGHTS, cpu_gemv_q4_0_ref, quantize_q4_0_synth,
            repack_q4_0_stream, synth_q4_0_vec,
        };
        let Some(ctx) = passthrough_ctx_or_skip("gemv_q4_0_stream") else {
            return;
        };
        for (m, k) in [(256usize, 256usize), (200, 128)] {
            let weights = synth_q4_0_vec(m * k, SYNTH_SALT_WEIGHTS);
            let raw = quantize_q4_0_synth(&weights, m, k);
            let (q, d) = repack_q4_0_stream(&raw, m, k);
            let x = synth_q4_0_vec(k, SYNTH_SALT_INPUTS);
            let expected = cpu_gemv_q4_0_ref(&raw, &x, m, k);
            let pipe = ctx.gemv_q4_0_stream_passthrough();
            let q_buf = ctx.upload_storage(bytemuck::cast_slice(&q), "test.q");
            let d_buf = ctx.upload_storage(bytemuck::cast_slice(&d), "test.d");
            let x_buf = ctx.upload_f32(&x, "test.x");
            let y_buf = ctx.create_storage_rw(m as u64 * 4, "test.y");
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[m as u32, k as u32, 0u32, 0u32]),
                "test.params",
            );
            let bg = bind_stream_qdxyp(&ctx, &pipe, &q_buf, &d_buf, &x_buf, &y_buf, &params_buf);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((m as u32).div_ceil(16), 1, 1);
            }
            ctx.submit_encoder(enc);
            ctx.device.poll_wait();
            let got = ctx.download_f32(&y_buf, m);
            assert_eq!(got.len(), m);
            let max_diff = max_abs_diff(&expected, &got);
            assert!(
                max_diff < 1e-2,
                "m={m} k={k}: max_diff={max_diff:.3e} exceeds f32-order noise"
            );
        }
    }

    /// The resident-stream prefill kernels (`gemm_stream_q4_0` k-slice-32 and
    /// `_k64`) match the CPU reference within f16-fiber tolerance, and each
    /// other bitwise (production dispatches k64 whenever k % 64 == 0 as the
    /// bit-exact faster twin). Shapes cover n_pad padding (n=48),
    /// multi-col-groups, the odd k/32 scale lane (k=96), and a small k64
    /// twin (k=64). The B16 input comes from the production
    /// `transpose_cast_f16` kernel (dispatched here, then byte-compared
    /// against the host layout), so the test pins the production chain,
    /// not the GEMM given a host-fabricated B16.
    #[test]
    fn stream_gemm_matches_cpu_and_twins_agree() {
        use super::{
            SYNTH_SALT_INPUTS, SYNTH_SALT_WEIGHTS, cpu_gemm_q4_0_ref, quantize_q4_0_synth,
            repack_q4_0_stream, synth_q4_0_vec,
        };
        let Some(ctx) = passthrough_ctx_or_skip("gemm_stream_q4_0") else {
            return;
        };
        for (m, n, k) in [(256usize, 32usize, 128usize), (512, 48, 96), (128, 64, 64)] {
            let n_pad = n.next_multiple_of(32);
            // Alternate per-block amplitude (×1/×0.125): uniform synth
            // weights give every block the same scale, so a (q, d) lane
            // mismatch is near-invisible on them (0.14 vs the 0.5 bound)
            // — the modulation makes scale-pairing bugs fail loudly.
            let mut weights = synth_q4_0_vec(m * k, SYNTH_SALT_WEIGHTS);
            for (i, w) in weights.iter_mut().enumerate() {
                if (i / 32) % 2 == 1 {
                    *w *= 0.125;
                }
            }
            let raw = quantize_q4_0_synth(&weights, m, k);
            let (q, d) = repack_q4_0_stream(&raw, m, k);
            // B row-major f32 [n][k], plus the host transpose+cast to
            // [k][n_pad] f16 the production kernel must reproduce.
            let b = synth_q4_0_vec(n * k, SYNTH_SALT_INPUTS);
            let mut b16 = vec![0u16; k * n_pad];
            for kk in 0..k {
                for nn in 0..n {
                    b16[kk * n_pad + nn] = half::f16::from_f32(b[nn * k + kk]).to_bits();
                }
            }
            let expected = cpu_gemm_q4_0_ref(&raw, &b, m, n, k);
            // Dispatch the production transpose kernel (params `[n, n_pad,
            // k, 0]`, grid `(n_pad/32, k/32)`, cf.
            // `encode_gemm_stream_q4_0`) over the f32 B.
            let x_buf = ctx.upload_f32(&b, "test.x");
            let t_params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[n as u32, n_pad as u32, k as u32, 0u32]),
                "test.t_params",
            );
            let t_pipe = ctx.transpose_cast_f16_passthrough();
            let b16_buf = ctx.create_storage_rw((k * n_pad * 2) as u64, "test.b16");
            let t_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &t_pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b16_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: t_params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut t_enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = t_enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&t_pipe);
                pass.set_bind_group(0, &t_bg, &[]);
                pass.dispatch_workgroups((n_pad / 32) as u32, (k as u32).div_ceil(32), 1);
            }
            ctx.submit_encoder(t_enc);
            ctx.device.poll_wait();
            // Direct oracle: the kernel's B16 must equal the host layout
            // byte for byte, padding included.
            let got_b16 = ctx.download_u32(&b16_buf, k * n_pad / 2);
            assert_eq!(
                bytemuck::cast_slice::<u32, u16>(&got_b16),
                b16.as_slice(),
                "m={m} n={n} k={k}: transpose_cast_f16 output differs from host layout"
            );
            let q_buf = ctx.upload_storage(bytemuck::cast_slice(&q), "test.q");
            let d_buf = ctx.upload_storage(bytemuck::cast_slice(&d), "test.d");
            // The GEMM reads the kernel-produced B16: end-to-end chain.
            let b_buf = b16_buf;
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[m as u32, k as u32, n as u32, n_pad as u32, m as u32]),
                "test.params",
            );
            // Fresh output per run: sharing one `y_buf` across the k32/k64
            // runs would let a k64 under-write read stale k32 values and
            // pass the twin check vacuously.
            let run = |pipe: &wgpu::ComputePipeline| {
                let y_buf = ctx.create_storage_rw(n_pad as u64 * m as u64 * 4, "test.y");
                let bg = bind_stream_qdxyp(&ctx, pipe, &q_buf, &d_buf, &b_buf, &y_buf, &params_buf);
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, &bg, &[]);
                    // `n_pad.div_ceil(32)` where production spells
                    // `n.div_ceil(32)`: provably the same grid
                    // (`n_pad/32 == ⌈n/32⌉`), so don't "fix" the spelling.
                    pass.dispatch_workgroups(
                        (m as u32).div_ceil(256),
                        (n_pad as u32).div_ceil(32),
                        1,
                    );
                }
                ctx.submit_encoder(enc);
                ctx.device.poll_wait();
                ctx.download_f32(&y_buf, n_pad * m)
            };
            let got32 = run(&ctx.gemm_stream_q4_0_passthrough());
            // k64 only where production selects it (k % 64 == 0).
            // Production also requires `use_gemm_k64()` (the
            // `CERA_WGPU_GEMM_K64=0` hatch disables the twin), but this is
            // a kernel-level agreement check, so it runs regardless.
            if k % 64 == 0 {
                let got64 = run(&ctx.gemm_stream_q4_0_k64_passthrough());
                assert_eq!(
                    got64, got32,
                    "k64 twin differs bitwise at m={m} n={n} k={k}"
                );
            }
            // Packed columns [n][m]; ignore the [n, n_pad) padding. f16 fiber
            // accumulation, so the CPU bound is loose by design: layout bugs
            // fail at O(1-10) — demonstrated, not reasoned (column rotation
            // in `stream_gemm_tolerance_catches_layout_scale_errors`,
            // scale pairing in
            // `stream_gemm_tolerance_catches_scale_pairing_errors`).
            let mut max_diff = 0.0f32;
            for c in 0..n {
                for r in 0..m {
                    max_diff = max_diff.max((expected[c * m + r] - got32[c * m + r]).abs());
                }
            }
            assert!(
                max_diff < 0.5,
                "m={m} n={n} k={k}: max_diff={max_diff:.3e} exceeds f16-fiber noise"
            );
        }
    }

    /// The stream-GEMM 0.5 bound catches layout-scale errors: a
    /// column-rotated reference — exactly what a B-transpose layout bug
    /// produces (`y'[c] = y[c-1]`) — fails the comparison at O(1-10) on
    /// every test shape. Host-only (pins the bound's tightness, so a
    /// future loosening fails here); the lavapipe CI leg executes the
    /// kernel side this bound guards.
    #[test]
    fn stream_gemm_tolerance_catches_layout_scale_errors() {
        use super::{
            SYNTH_SALT_INPUTS, SYNTH_SALT_WEIGHTS, cpu_gemm_q4_0_ref, quantize_q4_0_synth,
            synth_q4_0_vec,
        };
        for (m, n, k) in [(256usize, 32usize, 128usize), (512, 48, 96), (128, 64, 64)] {
            let weights = synth_q4_0_vec(m * k, SYNTH_SALT_WEIGHTS);
            let raw = quantize_q4_0_synth(&weights, m, k);
            let b = synth_q4_0_vec(n * k, SYNTH_SALT_INPUTS);
            let expected = cpu_gemm_q4_0_ref(&raw, &b, m, n, k);
            // Simulate the layout bug: every column reads its predecessor.
            let mut buggy = vec![0.0f32; expected.len()];
            for c in 0..n {
                for r in 0..m {
                    buggy[c * m + r] = expected[((c + n - 1) % n) * m + r];
                }
            }
            let max_diff = max_abs_diff(&expected, &buggy);
            assert!(
                max_diff >= 0.5,
                "m={m} n={n} k={k}: layout-scale error only reached {max_diff:.3e}; \
                 the 0.5 bound would miss it"
            );
        }
    }

    /// The stream-GEMM 0.5 bound also catches scale-pairing errors: every
    /// block decoded with its neighbor's scale — exactly what a (q, d)
    /// lane mismatch produces — fails at O(1-10) on every test shape.
    /// Host-only, same style as the rotation probe above.
    ///
    /// The weights alternate per-block amplitude (×1/×0.125): uniform synth
    /// weights give every block the same scale, so ANY scale permutation is
    /// near-invisible on them (0.14 here before the modulation). The kernel
    /// test above modulates identically, so the bound catches the bug
    /// class where it manifests in both places.
    #[test]
    fn stream_gemm_tolerance_catches_scale_pairing_errors() {
        use super::{
            SYNTH_SALT_INPUTS, SYNTH_SALT_WEIGHTS, cpu_gemm_q4_0_ref, quantize_q4_0_synth,
            synth_q4_0_vec,
        };
        for (m, n, k) in [(256usize, 32usize, 128usize), (512, 48, 96), (128, 64, 64)] {
            let mut weights = synth_q4_0_vec(m * k, SYNTH_SALT_WEIGHTS);
            for (i, w) in weights.iter_mut().enumerate() {
                if (i / 32) % 2 == 1 {
                    *w *= 0.125;
                }
            }
            let raw = quantize_q4_0_synth(&weights, m, k);
            let b = synth_q4_0_vec(n * k, SYNTH_SALT_INPUTS);
            let expected = cpu_gemm_q4_0_ref(&raw, &b, m, n, k);
            // Simulate the pairing bug: swap the f16 scale (bytes 0..2 of
            // each 18-byte block) between adjacent blocks in every row.
            let nb = k / 32;
            let mut buggy_raw = raw.clone();
            for r in 0..m {
                for blk in (0..nb).step_by(2) {
                    if blk + 1 >= nb {
                        break;
                    }
                    let a = r * nb * 18 + blk * 18;
                    let bb = a + 18;
                    buggy_raw.swap(a, bb);
                    buggy_raw.swap(a + 1, bb + 1);
                }
            }
            let buggy = cpu_gemm_q4_0_ref(&buggy_raw, &b, m, n, k);
            let max_diff = max_abs_diff(&expected, &buggy);
            assert!(
                max_diff >= 0.5,
                "m={m} n={n} k={k}: scale-pairing error only reached {max_diff:.3e}; \
                 the 0.5 bound would miss it"
            );
        }
    }

    /// The stream-GEMV 1e-2 bound catches nibble-order errors: every block
    /// decoded with swapped nibbles — exactly what a low/high lane mixup
    /// produces — fails far above f32-order noise on both test shapes.
    /// Host-only (pins the bound's tightness, so a future loosening fails
    /// here); the lavapipe CI leg executes the kernel side this bound guards.
    #[test]
    fn stream_gemv_tolerance_catches_nibble_errors() {
        use super::{
            SYNTH_SALT_INPUTS, SYNTH_SALT_WEIGHTS, cpu_gemv_q4_0_ref, quantize_q4_0_synth,
            synth_q4_0_vec,
        };
        for (m, k) in [(256usize, 256usize), (200, 128)] {
            let weights = synth_q4_0_vec(m * k, SYNTH_SALT_WEIGHTS);
            let raw = quantize_q4_0_synth(&weights, m, k);
            let x = synth_q4_0_vec(k, SYNTH_SALT_INPUTS);
            let expected = cpu_gemv_q4_0_ref(&raw, &x, m, k);
            // Simulate the nibble bug: swap the low/high 4-bit quants of
            // every byte (bytes 2..18 of each 18-byte block).
            let mut buggy_raw = raw.clone();
            for blk in buggy_raw.chunks_mut(18) {
                for b in &mut blk[2..18] {
                    *b = (*b).rotate_left(4);
                }
            }
            let buggy = cpu_gemv_q4_0_ref(&buggy_raw, &x, m, k);
            let max_diff = max_abs_diff(&expected, &buggy);
            assert!(
                max_diff >= 1e-2,
                "m={m} k={k}: nibble-order error only reached {max_diff:.3e}; \
                 the 1e-2 bound would miss it"
            );
        }
    }

    /// The resident-stream register-tiled fallthrough
    /// (`mul_mat_reg_tile_q4_0_stream`, production's path for n < 32 /
    /// strided B) matches the CPU reference. f32 accumulation like the raw
    /// twin, so the tolerance only absorbs summation-order differences.
    /// Shapes cover packed strides plus one padded case (`x_stride = k+16`,
    /// `y_stride = m+8`), pinning the stride legs of the kernel's address
    /// math a packed-only test would leave unexercised.
    #[test]
    fn stream_reg_tile_matches_cpu() {
        use super::{
            MUL_MAT_TILE_M, MUL_MAT_TILE_N, MUL_MAT_TILE_WG_M, MUL_MAT_TILE_WG_N,
            SYNTH_SALT_INPUTS, SYNTH_SALT_WEIGHTS, cpu_gemm_q4_0_ref, quantize_q4_0_synth,
            repack_q4_0_stream, synth_q4_0_vec,
        };
        let Some(ctx) = passthrough_ctx_or_skip("mul_mat_reg_tile_q4_0_stream") else {
            return;
        };
        // (m, n, k, x_stride, y_stride); packed legs spell out the strides.
        for (m, n, k, xs, ys) in [
            (128usize, 8usize, 128usize, 128usize, 128usize),
            (128, 72, 128, 128, 128),
            (100, 8, 64, 64, 100),
            (96, 16, 96, 112, 104),
        ] {
            let weights = synth_q4_0_vec(m * k, SYNTH_SALT_WEIGHTS);
            let raw = quantize_q4_0_synth(&weights, m, k);
            let (q, d) = repack_q4_0_stream(&raw, m, k);
            // X rows are `xs` wide (valid values in `[..k]`, zeros past);
            // the CPU reference runs on the packed valid region.
            let mut x = vec![0.0f32; n * xs];
            let packed = synth_q4_0_vec(n * k, SYNTH_SALT_INPUTS);
            for c in 0..n {
                x[c * xs..c * xs + k].copy_from_slice(&packed[c * k..(c + 1) * k]);
            }
            let expected = cpu_gemm_q4_0_ref(&raw, &packed, m, n, k);
            let pipe = ctx.mul_mat_reg_tile_q4_0_stream_passthrough();
            let q_buf = ctx.upload_storage(bytemuck::cast_slice(&q), "test.q");
            let d_buf = ctx.upload_storage(bytemuck::cast_slice(&d), "test.d");
            let x_buf = ctx.upload_f32(&x, "test.x");
            let y_buf = ctx.create_storage_rw(n as u64 * ys as u64 * 4, "test.y");
            let params_buf = ctx.upload_storage(
                bytemuck::cast_slice(&[m as u32, k as u32, n as u32, xs as u32, ys as u32]),
                "test.params",
            );
            let bg = bind_stream_qdxyp(&ctx, &pipe, &q_buf, &d_buf, &x_buf, &y_buf, &params_buf);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(
                    (m as u32).div_ceil(MUL_MAT_TILE_WG_M * MUL_MAT_TILE_M),
                    (n as u32).div_ceil(MUL_MAT_TILE_WG_N * MUL_MAT_TILE_N),
                    1,
                );
            }
            ctx.submit_encoder(enc);
            ctx.device.poll_wait();
            // Compare the valid `[n][m]` region, skipping `y_stride`
            // padding (whose contents the kernel does not define).
            let got = ctx.download_f32(&y_buf, n * ys);
            assert_eq!(got.len(), n * ys);
            let mut max_diff = 0.0f32;
            for c in 0..n {
                for r in 0..m {
                    max_diff = max_diff.max((expected[c * m + r] - got[c * ys + r]).abs());
                }
            }
            assert!(
                max_diff < 1e-2,
                "m={m} n={n} k={k} xs={xs} ys={ys}: max_diff={max_diff:.3e} exceeds f32-order noise"
            );
        }
    }
}

//! Shader-level parity for the wgpu batched-prefill GEMM (`mul_mat_reg_tile`).
//!
//! These tests drive the reg-tile loaders on synthetic weights with **no GGUF
//! and no network**, so they run anywhere a GPU exists and pin each dequant
//! loader directly to its CPU reference. The only other coverage is the
//! end-to-end `gpu_lfm2_prefill_equivalence` differential, which needs a real
//! GGUF fixture. All five quantized loaders (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) have
//! shader-level parity here, and every one of them also exercises the slangc
//! SPIR-V passthrough kernel wherever the backend accepts it (Vulkan); this is
//! the only automated coverage of the passthrough path.
//!
//! The `qh` 5th-bit plane is the whole risk in the Q5_K loader: it is indexed by
//! `l` (= y%32) alone — the same 32 bytes serve all four sub-blocks, consuming
//! bit `sub` — where every other field is indexed by the sub-block. `synth_q5_k`
//! keeps `qh`/`scales`/`qs` on mutually-coprime strides so a wrong bit selector
//! or a wrong base index cannot cancel out against the reference.
#![cfg(feature = "gpu")]

use cera::backend::wgpu::{DevicePollExt, GpuContext, shaders};
use cera::quant::{
    BlockQ5K, dequantize_q4_0_matrix, dequantize_q4_k_m_matrix, dequantize_q5_k_block,
    dequantize_q6_k_matrix, dequantize_q8_0_matrix,
};
use half::f16;

// Mirror the `pub(crate)` production tile geometry from `model::gpu_lfm2`
// (`MUL_MAT_TILE_*`). `build_mul_mat_pipeline` sets exactly these defines, so
// the pipeline compiled here is byte-for-byte the production Q5_K kernel. A
// full workgroup covers WG_M*TILE_M = 64 rows by WG_N*TILE_N = 64 columns.
const WG_M: u32 = 16;
const WG_N: u32 = 16;
const TILE_M: u32 = 4;
const TILE_N: u32 = 4;
const TILE_K: u32 = 16;
const TILE_ROWS_COVERED: u32 = WG_M * TILE_M; // 64
const TILE_COLS_COVERED: u32 = WG_N * TILE_N; // 64

/// Build `m` rows of synthetic Q5_K weights: the packed bytes as the loader
/// reads them, plus the f32 values the CPU reference decodes from the same
/// blocks. `k` must be a multiple of 256 (one super-block).
fn synth_q5_k(m: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(
        k % 256,
        0,
        "k must be a whole number of 256-elem super-blocks"
    );
    let nb = (k / 256) as usize;
    let mut raw = Vec::with_capacity(m as usize * nb * 176);
    let mut w_f32 = vec![0.0f32; m as usize * k as usize];
    for row in 0..m as usize {
        for b in 0..nb {
            let mut blk = BlockQ5K {
                d: f16::from_f32(0.01 + (row as f32 * 0.003).sin() * 0.002).to_bits(),
                dmin: f16::from_f32(0.004 + (row as f32 * 0.002).cos() * 0.001).to_bits(),
                scales: [0u8; 12],
                qh: [0u8; 32],
                qs: [0u8; 128],
            };
            for i in 0..12 {
                blk.scales[i] = ((row * 5 + b * 3 + i) & 0xFF) as u8;
            }
            for i in 0..32 {
                blk.qh[i] = ((row * 11 + b * 7 + i) & 0xFF) as u8;
            }
            for i in 0..128 {
                blk.qs[i] = ((row * 37 + b * 13 + i) & 0xFF) as u8;
            }
            let dq = dequantize_q5_k_block(&blk);
            let off = row * k as usize + b * 256;
            w_f32[off..off + 256].copy_from_slice(&dq);
            // Serialize in `BlockQ5K` field order: d, dmin, scales, qh, qs.
            raw.extend_from_slice(&blk.d.to_le_bytes());
            raw.extend_from_slice(&blk.dmin.to_le_bytes());
            raw.extend_from_slice(&blk.scales);
            raw.extend_from_slice(&blk.qh);
            raw.extend_from_slice(&blk.qs);
        }
    }
    (raw, w_f32)
}

/// `m` rows of synthetic Q4_K weights (144 B / 256-elem super-block: d f16,
/// dmin f16, scales[12], qs[128]) plus the f32 the CPU loader decodes from the
/// same bytes. Mutually-coprime strides on the scale and quant bytes keep a
/// wrong sub-block index or nibble half from cancelling against the reference.
/// `k` must be a multiple of 256.
fn synth_q4_k(m: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(
        k % 256,
        0,
        "k must be a whole number of 256-elem super-blocks"
    );
    let nb = (k / 256) as usize;
    let mut raw = Vec::with_capacity(m as usize * nb * 144);
    for row in 0..m as usize {
        for b in 0..nb {
            let d = f16::from_f32(0.01 + (row as f32 * 0.003).sin() * 0.002);
            let dmin = f16::from_f32(0.004 + (row as f32 * 0.002).cos() * 0.001);
            raw.extend_from_slice(&d.to_bits().to_le_bytes());
            raw.extend_from_slice(&dmin.to_bits().to_le_bytes());
            for i in 0..12 {
                raw.push(((row * 5 + b * 3 + i) & 0xFF) as u8);
            }
            for i in 0..128 {
                raw.push(((row * 37 + b * 13 + i) & 0xFF) as u8);
            }
        }
    }
    let mut w_f32 = vec![0.0f32; m as usize * k as usize];
    dequantize_q4_k_m_matrix(&raw, m as usize, k as usize, &mut w_f32);
    (raw, w_f32)
}

/// `m` rows of synthetic Q6_K weights (210 B / 256-elem super-block: ql[128],
/// qh[64], scales[16] as signed i8, d f16) plus the f32 the CPU loader decodes
/// from them. The scale bytes sweep the full i8 range so a wrong sign-extension
/// of the sub-scale is caught; coprime strides on ql/qh keep the (n,l,j)
/// inversion honest. `k` must be a multiple of 256.
fn synth_q6_k(m: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(
        k % 256,
        0,
        "k must be a whole number of 256-elem super-blocks"
    );
    let nb = (k / 256) as usize;
    let mut raw = Vec::with_capacity(m as usize * nb * 210);
    for row in 0..m as usize {
        for b in 0..nb {
            for i in 0..128 {
                raw.push(((row * 37 + b * 13 + i) & 0xFF) as u8); // ql
            }
            for i in 0..64 {
                raw.push(((row * 11 + b * 7 + i) & 0xFF) as u8); // qh
            }
            for i in 0..16 {
                raw.push(((row * 23 + b * 17 + i) & 0xFF) as u8); // scales (i8)
            }
            let d = f16::from_f32(0.006 + (row as f32 * 0.002).cos() * 0.001);
            raw.extend_from_slice(&d.to_bits().to_le_bytes());
        }
    }
    let mut w_f32 = vec![0.0f32; m as usize * k as usize];
    dequantize_q6_k_matrix(&raw, m as usize, k as usize, &mut w_f32);
    (raw, w_f32)
}

/// Upload `raw` (packed src0 weights, `m` rows) plus synthetic activations,
/// dispatch the already-built reg-tile `pipeline`, and assert the GPU result
/// matches `w_f32 * xᵀ` over the whole output. The loader decodes weights to f32
/// (no f16 rounding of the weight), so the only divergence from the sequential
/// CPU sum is float reassociation in the tiled GPU accumulation; a dequant,
/// stride, or ragged-tile bug moves an element far above the tolerance. Shared
/// by the Q5_K, Q4_0, and Q8_0 cases, naga and SPIR-V passthrough alike.
// The 8 params (ctx, pipeline, raw + reference weights, m, k, n, label) are all
// intrinsic to a GEMM check; bundling them into a struct would be noise in a
// test helper.
#[allow(clippy::too_many_arguments)]
fn dispatch_and_check(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    raw: &[u8],
    w_f32: &[f32],
    m: u32,
    k: u32,
    n: u32,
    label: &str,
) {
    // Varied, non-degenerate activations; distinct per token column.
    let x: Vec<f32> = (0..(n * k))
        .map(|idx| {
            let t = (idx / k) as f32;
            let i = (idx % k) as f32;
            (i * 0.013 + t * 0.07).sin() * 0.5
        })
        .collect();

    // Reference: dst is laid out with y_stride = m, i.e. dst[t*m + r].
    let mut y_ref = vec![0.0f32; (n * m) as usize];
    for t in 0..n as usize {
        for r in 0..m as usize {
            let mut s = 0.0f32;
            for i in 0..k as usize {
                s += w_f32[r * k as usize + i] * x[t * k as usize + i];
            }
            y_ref[t * m as usize + r] = s;
        }
    }

    let src0 = ctx.upload_storage(raw, "src0");
    let src1 = ctx.upload_f32(&x, "x");
    let dst = ctx.create_storage_rw((n * m) as u64 * 4, "dst");
    // MulMatParams: [m, k, n, x_stride, y_stride]; x is n×k, y is n×m.
    let params = ctx.upload_storage(bytemuck::cast_slice(&[m, k, n, k, m]), "params");

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let grid = (
        m.div_ceil(TILE_ROWS_COVERED),
        n.div_ceil(TILE_COLS_COVERED),
        1,
    );
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.device.poll_wait();

    let y_gpu = ctx.download_f32(&dst, (n * m) as usize);

    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    for (a, b) in y_gpu.iter().zip(&y_ref) {
        assert!(
            a.is_finite(),
            "{label}: non-finite GPU output for {m}x{k}x{n}"
        );
        max_abs = max_abs.max((a - b).abs());
        max_ref = max_ref.max(b.abs());
    }
    let tol = 2e-3 * max_ref.max(1.0);
    assert!(
        max_abs <= tol,
        "{label} GEMM parity {m}x{k}x{n}: max_abs_err={max_abs:.3e} > tol={tol:.3e} \
         (max_ref={max_ref:.3e})",
    );
}

/// Build the production naga reg-tile pipeline for one src0 loader define, with
/// the exact `MUL_MAT_TILE_*` geometry mirrored above, so the compiled kernel is
/// byte-for-byte the one `model::gpu_lfm2` builds via `build_mul_mat_pipeline`.
fn naga_reg_tile(
    ctx: &GpuContext,
    label: &str,
    src0_loader: &str,
    src0_inner: &str,
) -> wgpu::ComputePipeline {
    let wg_m = format!("{WG_M}u");
    let wg_n = format!("{WG_N}u");
    let tile_m = format!("{TILE_M}u");
    let tile_n = format!("{TILE_N}u");
    let tile_k = format!("{TILE_K}u");
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

/// Drive the three K-quant reg-tile loaders (Q4_K, Q5_K, Q6_K) through the naga
/// path (every backend) and, where the device accepts SPIR-V passthrough
/// (Vulkan), the slangc kernel too, asserting both match the CPU dequant
/// reference. Mirrors `check_q4_0_q8_0`: all naga first, then one
/// passthrough-or-guard block so `CERA_REQUIRE_PASSTHROUGH` is honored once.
fn check_k_quants(ctx: &GpuContext, m: u32, k: u32, n: u32) {
    let (raw4, w4) = synth_q4_k(m, k);
    let naga4 = naga_reg_tile(ctx, "mul_mat_q4_k_parity", "INIT_SRC0_SHMEM_Q4_K", "u32");
    dispatch_and_check(ctx, &naga4, &raw4, &w4, m, k, n, "Q4_K naga");

    let (raw5, w5) = synth_q5_k(m, k);
    let naga5 = naga_reg_tile(ctx, "mul_mat_q5_k_parity", "INIT_SRC0_SHMEM_Q5_K", "u32");
    dispatch_and_check(ctx, &naga5, &raw5, &w5, m, k, n, "Q5_K naga");

    let (raw6, w6) = synth_q6_k(m, k);
    let naga6 = naga_reg_tile(ctx, "mul_mat_q6_k_parity", "INIT_SRC0_SHMEM_Q6_K", "u32");
    dispatch_and_check(ctx, &naga6, &raw6, &w6, m, k, n, "Q6_K naga");

    if ctx.supports_spirv_passthrough() {
        let pt4 = ctx.mul_mat_reg_tile_q4_k_passthrough();
        dispatch_and_check(ctx, &pt4, &raw4, &w4, m, k, n, "Q4_K passthrough");
        let pt5 = ctx.mul_mat_reg_tile_q5_k_passthrough();
        dispatch_and_check(ctx, &pt5, &raw5, &w5, m, k, n, "Q5_K passthrough");
        let pt6 = ctx.mul_mat_reg_tile_q6_k_passthrough();
        dispatch_and_check(ctx, &pt6, &raw6, &w6, m, k, n, "Q6_K passthrough");
    } else {
        // Mirror CERA_REQUIRE_GPU: on a Vulkan CI leg (lavapipe) this must not
        // silently skip, or the passthrough kernels would report green untested.
        assert!(
            std::env::var("CERA_REQUIRE_PASSTHROUGH")
                .unwrap_or_default()
                .is_empty(),
            "CERA_REQUIRE_PASSTHROUGH is set but the backend ({}) does not take \
             SPIR-V passthrough",
            ctx.backend
        );
        eprintln!(
            "[wgpu-mul-mat-parity] SKIP K-quant passthrough on non-Vulkan backend ({})",
            ctx.backend
        );
    }
}

/// `m` rows of synthetic Q4_0 weights (18 B / 32-elem block: an f16 scale then
/// 16 packed nibble pairs) plus the f32 the CPU loader decodes from the same
/// bytes. Mutually-coprime strides on the nibble bytes keep a wrong base index
/// or nibble half from cancelling against the reference.
fn synth_q4_0(m: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 32, 0, "k must be a whole number of 32-elem Q4_0 blocks");
    let nb = (k / 32) as usize;
    let mut raw = Vec::with_capacity(m as usize * nb * 18);
    for row in 0..m as usize {
        for b in 0..nb {
            let d = f16::from_f32(0.008 + (row as f32 * 0.003).sin() * 0.002);
            raw.extend_from_slice(&d.to_bits().to_le_bytes());
            for i in 0..16 {
                raw.push(((row * 37 + b * 13 + i) & 0xFF) as u8);
            }
        }
    }
    let mut w_f32 = vec![0.0f32; m as usize * k as usize];
    dequantize_q4_0_matrix(&raw, m as usize, k as usize, &mut w_f32);
    (raw, w_f32)
}

/// `m` rows of synthetic Q8_0 weights (34 B / 32-elem block: an f16 scale then
/// 32 signed int8 quants) plus the f32 the CPU loader decodes from them. The
/// quants sweep the full i8 range so a wrong sign-extension is caught.
fn synth_q8_0(m: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % 32, 0, "k must be a whole number of 32-elem Q8_0 blocks");
    let nb = (k / 32) as usize;
    let mut raw = Vec::with_capacity(m as usize * nb * 34);
    for row in 0..m as usize {
        for b in 0..nb {
            let d = f16::from_f32(0.006 + (row as f32 * 0.002).cos() * 0.001);
            raw.extend_from_slice(&d.to_bits().to_le_bytes());
            for i in 0..32 {
                raw.push(((row * 29 + b * 17 + i) & 0xFF) as u8);
            }
        }
    }
    let mut w_f32 = vec![0.0f32; m as usize * k as usize];
    dequantize_q8_0_matrix(&raw, m as usize, k as usize, &mut w_f32);
    (raw, w_f32)
}

/// Drive the Q4_0/Q8_0 reg-tile loader through the naga path (every backend)
/// and, where the device accepts SPIR-V passthrough (Vulkan), the slangc kernel
/// too, asserting both match the CPU dequant reference. This is the only
/// automated coverage of the Slang passthrough kernels: it runs them on real
/// (software, in CI) Vulkan and pins them to `dequantize_q{4_0,8_0}_matrix`.
fn check_q4_0_q8_0(ctx: &GpuContext, m: u32, k: u32, n: u32) {
    let (raw4, w4) = synth_q4_0(m, k);
    let naga4 = naga_reg_tile(ctx, "mul_mat_q4_0_parity", "INIT_SRC0_SHMEM_Q4_0", "u32");
    dispatch_and_check(ctx, &naga4, &raw4, &w4, m, k, n, "Q4_0 naga");

    let (raw8, w8) = synth_q8_0(m, k);
    let naga8 = naga_reg_tile(ctx, "mul_mat_q8_0_parity", "INIT_SRC0_SHMEM_Q8_0", "u32");
    dispatch_and_check(ctx, &naga8, &raw8, &w8, m, k, n, "Q8_0 naga");

    if ctx.supports_spirv_passthrough() {
        let pt4 = ctx.mul_mat_reg_tile_q4_0_passthrough();
        dispatch_and_check(ctx, &pt4, &raw4, &w4, m, k, n, "Q4_0 passthrough");
        let pt8 = ctx.mul_mat_reg_tile_q8_0_passthrough();
        dispatch_and_check(ctx, &pt8, &raw8, &w8, m, k, n, "Q8_0 passthrough");
    } else {
        // Mirror CERA_REQUIRE_GPU: on a Vulkan CI leg (lavapipe) this must not
        // silently skip, or the passthrough kernels would report green untested.
        assert!(
            std::env::var("CERA_REQUIRE_PASSTHROUGH")
                .unwrap_or_default()
                .is_empty(),
            "CERA_REQUIRE_PASSTHROUGH is set but the backend ({}) does not take \
             SPIR-V passthrough",
            ctx.backend
        );
        eprintln!(
            "[wgpu-mul-mat-parity] SKIP passthrough on non-Vulkan backend ({})",
            ctx.backend
        );
    }
}

/// Dense f32 weights for the FLOAT loader. The loader binds src0 as
/// `array<f32>` and reads directly, so the `raw` bytes ARE the reference f32
/// values (no packing). Varied and non-degenerate, distinct per (row, col).
fn synth_f32(m: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
    let w: Vec<f32> = (0..(m * k))
        .map(|idx| {
            let r = (idx / k) as f32;
            let c = (idx % k) as f32;
            (r * 0.019 - c * 0.011).cos() * 0.7
        })
        .collect();
    (bytemuck::cast_slice(&w).to_vec(), w)
}

/// Dense-f32 reg-tile loader: the batched-prefill fallback that `upload_weight`
/// routes every non-quant dtype through (F16/BF16/F32 sources, Q4_1, Q2_K, all
/// dequantized to f32 on the GPU). Binds src0 as `array<f32>`
/// (`SRC0_INNER_TYPE = f32`) and reads through `INIT_SRC0_SHMEM_FLOAT`, matching
/// `gpu_lfm2`'s `mul_mat_reg_tile_f32` pipeline. Naga only: unlike the quantized
/// loaders, f32 has no Slang SPIR-V passthrough kernel (the passthrough covers
/// just the quantized loaders; the dense loader was never naga-regressed).
fn check_f32(ctx: &GpuContext, m: u32, k: u32, n: u32) {
    let (raw, w) = synth_f32(m, k);
    let pipe = naga_reg_tile(ctx, "mul_mat_f32_parity", "INIT_SRC0_SHMEM_FLOAT", "f32");
    dispatch_and_check(ctx, &pipe, &raw, &w, m, k, n, "F32 naga");
}

fn setup() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but no GPU is available: {e}"
            );
            eprintln!("[wgpu-mul-mat-parity] SKIP (no GPU): {e}");
            None
        }
    }
}

#[test]
fn k_quants_gemm_clean_tile() {
    let Some(ctx) = setup() else { return };
    // Exact multiples of the 64×64 tile: no ragged masking.
    check_k_quants(&ctx, 128, 256, 64);
}

#[test]
fn k_quants_gemm_ragged_tile() {
    let Some(ctx) = setup() else { return };
    // Ragged in both m (>128, not a multiple of 64) and n (<64): exercises the
    // edge masking in the loaders and the kernel's partial-tile store, plus a
    // 2-super-block k so the qh plane (Q5_K) and the (n,l,j) inversion (Q6_K)
    // are decoded across sub-block boundaries.
    check_k_quants(&ctx, 200, 512, 40);
}

#[test]
fn q4_0_q8_0_gemm_clean_tile() {
    let Some(ctx) = setup() else { return };
    // Exact multiples of the 64×64 tile: no ragged masking.
    check_q4_0_q8_0(&ctx, 128, 256, 64);
}

#[test]
fn q4_0_q8_0_gemm_ragged_tile() {
    let Some(ctx) = setup() else { return };
    // Ragged in both m (>128, not a multiple of 64) and n (<64): exercises the
    // loader edge masking and the kernel's partial-tile store.
    check_q4_0_q8_0(&ctx, 200, 512, 40);
}

#[test]
fn f32_gemm_clean_tile() {
    let Some(ctx) = setup() else { return };
    // Exact multiples of the 64×64 tile: no ragged masking.
    check_f32(&ctx, 128, 256, 64);
}

#[test]
fn f32_gemm_ragged_tile() {
    let Some(ctx) = setup() else { return };
    // Ragged in both m (>128, not a multiple of 64) and n (<64): exercises the
    // FLOAT loader's per-element edge masking and the partial-tile store.
    check_f32(&ctx, 200, 512, 40);
}

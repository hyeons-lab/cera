//! Shader-level parity for the wgpu batched-prefill GEMM (`mul_mat_reg_tile`).
//!
//! Q5_K is the first dtype with a shader-level parity test on this kernel —
//! there was none for the Q4_0/Q8_0/Q4_K/Q6_K loaders, only the end-to-end
//! `gpu_lfm2_prefill_equivalence` differential, which needs a real GGUF fixture
//! (and the fixture set carries no Q5_K model, so a Q5_K end-to-end test would
//! silently skip in CI). This drives the loader on synthetic weights with **no
//! GGUF and no network**, so it runs anywhere a GPU exists and pins the new
//! Q5_K dequant loader directly to the CPU `dequantize_q5_k_block` reference.
//!
//! The `qh` 5th-bit plane is the whole risk in the Q5_K loader: it is indexed by
//! `l` (= y%32) alone — the same 32 bytes serve all four sub-blocks, consuming
//! bit `sub` — where every other field is indexed by the sub-block. `synth_q5_k`
//! keeps `qh`/`scales`/`qs` on mutually-coprime strides so a wrong bit selector
//! or a wrong base index cannot cancel out against the reference.
#![cfg(feature = "gpu")]

use cera::backend::wgpu::{DevicePollExt, GpuContext, shaders};
use cera::quant::{BlockQ5K, dequantize_q5_k_block};
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

/// Run `mul_mat_reg_tile` with the Q5_K loader for `y[t,r] = sum_i W[r,i]*x[t,i]`
/// and compare to the CPU reference over the whole output. `m`/`n` are chosen by
/// the caller to be a clean tile or a ragged one.
fn check(ctx: &GpuContext, m: u32, k: u32, n: u32) {
    let (raw, w_f32) = synth_q5_k(m, k);

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

    let src0 = ctx.upload_storage(&raw, "q5k_src0");
    let src1 = ctx.upload_f32(&x, "q5k_x");
    let dst = ctx.create_storage_rw((n * m) as u64 * 4, "q5k_dst");
    // MulMatParams: [m, k, n, x_stride, y_stride]; x is n×k, y is n×m.
    let params = ctx.upload_storage(bytemuck::cast_slice(&[m, k, n, k, m]), "q5k_params");

    let wg_m = format!("{WG_M}u");
    let wg_n = format!("{WG_N}u");
    let tile_m = format!("{TILE_M}u");
    let tile_n = format!("{TILE_N}u");
    let tile_k = format!("{TILE_K}u");
    let pipeline = ctx.create_pipeline_with_defines(
        shaders::MUL_MAT_REG_TILE,
        "main",
        "mul_mat_q5_k_parity",
        &[
            ("SRC0_INNER_TYPE", "u32"),
            ("INIT_SRC0_SHMEM_Q5_K", ""),
            ("WORKGROUP_SIZE_M", &wg_m),
            ("WORKGROUP_SIZE_N", &wg_n),
            ("TILE_M", &tile_m),
            ("TILE_N", &tile_n),
            ("TILE_K", &tile_k),
        ],
    );

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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.device.poll_wait();

    let y_gpu = ctx.download_f32(&dst, (n * m) as usize);

    // The loader decodes weights to f32 (no f16 rounding of the weight, unlike
    // the Metal simdgroup GEMM), so the only divergence is float reassociation
    // between the tiled GPU accumulation and the sequential CPU sum. A `qh`
    // decode error moves an element by 16*scale — orders above this bound.
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    for (a, b) in y_gpu.iter().zip(&y_ref) {
        assert!(a.is_finite(), "non-finite GPU output for {m}x{k}x{n}");
        max_abs = max_abs.max((a - b).abs());
        max_ref = max_ref.max(b.abs());
    }
    let tol = 2e-3 * max_ref.max(1.0);
    assert!(
        max_abs <= tol,
        "Q5_K GEMM parity {m}x{k}x{n}: max_abs_err={max_abs:.3e} > tol={tol:.3e} \
         (max_ref={max_ref:.3e})",
    );
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
fn q5_k_gemm_clean_tile() {
    let Some(ctx) = setup() else { return };
    // Exact multiples of the 64×64 tile: no ragged masking.
    check(&ctx, 128, 256, 64);
}

#[test]
fn q5_k_gemm_ragged_tile() {
    let Some(ctx) = setup() else { return };
    // Ragged in both m (>128, not a multiple of 64) and n (<64): exercises the
    // edge masking in the loader and the kernel's partial-tile store, plus a
    // 2-super-block k so the qh plane is decoded across sub-block boundaries.
    check(&ctx, 200, 512, 40);
}

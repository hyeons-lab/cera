//! Numerical parity: native-Metal vs CPU `encode_image` for the LFM2-VL ViT
//! vision encoder, on the real quantized mmproj.
//!
//! Unlike `vit_encode_bench` (which only *times* the Metal path), this asserts
//! the Metal encoder's output matches the CPU reference element-for-element. It
//! is the regression gate the typed-kernel-params migration (#266) needed: a
//! wrong param slot, field order, or a transpose in any migrated `MetalVitOps`
//! upload would decorrelate the two vectors, which the cosine check below
//! catches even when magnitudes happen to line up.
//!
//! Both backends run the identical algorithm on the identical GGUF weights, so
//! the only expected divergence is f32 accumulation-order drift plus the
//! MMA attention kernel's `half` Q/KV tiles (f32 accumulators). That keeps
//! cosine ≈ 1 and relative RMS small; the thresholds are calibrated with
//! headroom below.
//!
//! Gating: `#[ignore]` + `CERA_TEST_DOWNLOAD=1`, sharing the cached
//! LFM2.5-VL-450M GGUFs with `vit_encode_bench` / `vl_clip_parity`. Skips
//! (loudly, not vacuously) when the env var is unset or no Metal device is
//! present.
//!
//! ```sh
//! CERA_TEST_DOWNLOAD=1 cargo test -p cera --release \
//!     --features remote,metal,vl-preprocess --test vit_metal_parity -- --ignored --nocapture
//! ```

#![cfg(all(feature = "remote", feature = "metal", feature = "vl-preprocess"))]

mod common;

use cera::engine::{BackendPreference, CeraEngine, EngineConfig, ModelFiles};
use cera::manifest::InferenceType;
use cera::model::vision_encoder_gpu::build_gpu_vision_encoder;

const MAIN_URL: &str =
    "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-Q4_0.gguf";
const MAIN_FILE: &str = "LFM2.5-VL-450M-Q4_0.gguf";
const MMPROJ_URL: &str = "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-Q8_0.gguf";
const MMPROJ_FILE: &str = "mmproj-LFM2.5-VL-450m-Q8_0.gguf";

/// Cosine similarity in f64 to avoid catastrophic cancellation on long vectors.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `‖got − reference‖ / ‖reference‖` — relative magnitude of the error vector.
fn relative_rms(reference: &[f32], got: &[f32]) -> f64 {
    let (mut se, mut energy) = (0f64, 0f64);
    for (&r, &g) in reference.iter().zip(got) {
        let d = r as f64 - g as f64;
        se += d * d;
        energy += (r as f64) * (r as f64);
    }
    (se / energy.max(1e-12)).sqrt()
}

#[test]
#[ignore = "downloads ~310 MB and needs a Metal device; set CERA_TEST_DOWNLOAD=1 and pass --ignored"]
fn vit_metal_matches_cpu() {
    if std::env::var("CERA_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: CERA_TEST_DOWNLOAD not set");
        return;
    }

    let main = common::download::ensure_cached(MAIN_URL, MAIN_FILE);
    let mmproj = common::download::ensure_cached(MMPROJ_URL, MMPROJ_FILE);
    let mut files = ModelFiles::text(&main);
    files.multimodal_projector = Some(mmproj);
    files.inference_type = Some(InferenceType::LlamaCppImageToText);
    let engine = CeraEngine::from_files(
        files,
        EngineConfig {
            context_size: 256,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("load VL bundle");

    let cpu_enc = engine.vision_encoder().expect("vision encoder").clone();

    // A missing Metal device is a skip, not a pass — but say so loudly.
    let Some(metal_enc) = build_gpu_vision_encoder(cpu_enc.as_ref(), BackendPreference::Metal)
    else {
        eprintln!("skipping: no Metal vision encoder (device or feature unavailable)");
        return;
    };

    // Sweep source sizes → different patch grids so both attention paths run:
    // the MMA flash kernel (head_dim % 8 == 0 && ≤ 128) plus partial tiles at
    // token counts that aren't multiples of the tile geometry.
    let mut worst_cos = 1.0f64;
    let mut worst_rms = 0.0f64;
    for side in [256u32, 384, 512] {
        use image::{ImageBuffer, Rgb};
        // Deterministic, non-uniform content: a solid color underconstrains
        // attention (every key identical), which would hide a real attn bug.
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(side, side, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode png");

        let pre = cera::model::vision_preprocessor::preprocess_image(&png, &cpu_enc.config)
            .expect("preprocess");
        let (gw, gh) = (pre.grid_w, pre.grid_h);

        let cpu_out = cpu_enc
            .encode_image(&pre.pixels, gw, gh)
            .expect("cpu encode");
        let metal_out = metal_enc
            .encode_image(&pre.pixels, gw, gh)
            .expect("metal encode");

        assert_eq!(
            cpu_out.len(),
            metal_out.len(),
            "{side}px: CPU produced {} floats, Metal {}",
            cpu_out.len(),
            metal_out.len()
        );
        assert!(!cpu_out.is_empty(), "{side}px: empty encode output");

        for (i, (&c, &m)) in cpu_out.iter().zip(&metal_out).enumerate() {
            assert!(
                c.is_finite() && m.is_finite(),
                "{side}px: non-finite at index {i}: cpu={c}, metal={m}"
            );
        }

        let cos = cosine(&cpu_out, &metal_out);
        let rms = relative_rms(&cpu_out, &metal_out);
        worst_cos = worst_cos.min(cos);
        worst_rms = worst_rms.max(rms);
        eprintln!(
            "  {side}px → grid {gw}x{gh} ({} floats): cosine={cos:.6} rel_rms={rms:.4}",
            cpu_out.len()
        );

        // Cosine catches structural bugs (transpose / wrong slot / field-order
        // swap) that decorrelate the vectors — a garbage encode lands near 0.
        // Measured on LFM2.5-VL-450M across these three grids: 0.9988-0.9992
        // (worst 0.998804 at 384px), stable run-to-run since Metal f16 is
        // deterministic. 0.995 leaves headroom for other Apple GPU families
        // while still failing loudly on any real divergence.
        assert!(
            cos > 0.995,
            "{side}px: cosine {cos:.6} < 0.995 — Metal ViT structurally diverged from CPU"
        );
        // Relative RMS bounds per-element magnitude drift. The MMA attention
        // kernel's f16 Q/KV tiles set the realistic floor: measured 0.041-0.049
        // (worst 0.0489). 0.08 keeps ~1.6x headroom while still failing on a
        // scale / norm / GELU bug (which would push it past 0.1).
        assert!(
            rms < 0.08,
            "{side}px: relative RMS {rms:.4} > 0.08 — Metal ViT magnitudes diverged from CPU"
        );
    }
    eprintln!("  worst over sweep: cosine={worst_cos:.6} rel_rms={worst_rms:.4}");
}

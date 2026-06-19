//! Encode-only micro-benchmark: CPU vs wgpu vs native-Metal `encode_image` for
//! the LFM2-VL ViT vision encoder, on the real quantized mmproj.
//!
//! This times ONLY the ViT forward pass (`encode_image`) — image preprocessing
//! and the LLM prefill/decode are excluded — so it isolates the GPU-vs-CPU
//! encode tradeoff the GPU path is meant to win. It is a measurement tool, not a
//! regression gate: it prints timings and asserts nothing, because on small
//! patch grids the GPU path can legitimately lose to the parallel CPU `gemv`
//! (per-op command-buffer / dispatch overhead), which is exactly the open
//! question this benchmark exists to answer.
//!
//! Run with (release matters — debug CPU `gemv` is far slower than shipped):
//!
//! ```sh
//! CERA_TEST_DOWNLOAD=1 cargo test -p cera --release \
//!     --features remote,metal,gpu --test vit_encode_bench -- --ignored --nocapture
//! ```
//!
//! Drop `metal` or `gpu` to bench only the backends you care about; the missing
//! one prints "unavailable". The weight upload/dequant (`build_gpu_vision_encoder`)
//! happens once per backend, OUTSIDE the timing loop, so only the per-image
//! encode is measured.

#![cfg(all(feature = "remote", feature = "vl-preprocess"))]

mod common;

use std::time::Instant;

use cera::engine::{BackendPreference, CeraEngine, EngineConfig, ModelFiles};
use cera::manifest::InferenceType;
use cera::model::vision_encoder_gpu::build_gpu_vision_encoder;

const MAIN_URL: &str =
    "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-Q4_0.gguf";
const MAIN_FILE: &str = "LFM2.5-VL-450M-Q4_0.gguf";
const MMPROJ_URL: &str = "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-Q8_0.gguf";
const MMPROJ_FILE: &str = "mmproj-LFM2.5-VL-450m-Q8_0.gguf";

/// Median + mean wall-clock (ms) of `f` over `runs` timed iterations, after
/// `warmup` untimed iterations (to fault in pipelines / warm caches).
fn time_ms(warmup: usize, runs: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("NaN timing sample"));
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    (median, mean)
}

#[test]
#[ignore = "downloads ~310 MB and needs a GPU; set CERA_TEST_DOWNLOAD=1 and pass --ignored"]
fn vit_encode_cpu_vs_gpu_bench() {
    if std::env::var("CERA_TEST_DOWNLOAD").is_err() {
        eprintln!("skipping: CERA_TEST_DOWNLOAD not set");
        return;
    }

    let main = common::download::ensure_cached(MAIN_URL, MAIN_FILE);
    let mmproj = common::download::ensure_cached(MMPROJ_URL, MMPROJ_FILE);
    let mut files = ModelFiles::text(&main);
    files.multimodal_projector = Some(mmproj);
    files.inference_type = Some(InferenceType::LlamaCppImageToText);
    // Build with the CPU backend: we only need the parsed CPU encoder here; the
    // GPU encoders are built explicitly per backend below so we can time each.
    let engine = CeraEngine::from_files(
        files,
        EngineConfig {
            context_size: 512,
            backend: BackendPreference::Cpu,
            ..Default::default()
        },
    )
    .expect("load VL bundle");
    let cpu_enc = engine.vision_encoder().expect("vision encoder").clone();

    // Build each GPU backend once (dequant + upload are one-time setup, not part
    // of the per-image encode we're timing). `None` = feature off or no device.
    let wgpu_enc = build_gpu_vision_encoder(cpu_enc.as_ref(), BackendPreference::Gpu);
    let metal_enc = build_gpu_vision_encoder(cpu_enc.as_ref(), BackendPreference::Metal);

    let warmup = 2;
    let runs = 10;
    eprintln!(
        "\nViT encode bench — {} layers, n_embd={}, {warmup} warmup + {runs} runs\n\
         wgpu: {}   Metal: {}",
        cpu_enc.config.n_layer,
        cpu_enc.config.n_embd,
        if wgpu_enc.is_some() { "yes" } else { "n/a" },
        if metal_enc.is_some() { "yes" } else { "n/a" },
    );

    // Sweep source image sizes → small / mid / max patch grids. The preprocessor
    // resolves each to a grid within [image_min_pixels, image_max_pixels], so the
    // actual grid is printed per row. The small grid tests the earlier concern
    // that per-op GPU overhead could lose to the parallel CPU path at low token
    // counts; the max grid (1024 patches) is the worst case for the CPU gemv.
    for side in [256u32, 384, 512] {
        use image::{ImageBuffer, Rgb};
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(side, side, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode png");
        let pre = cera::model::vision_preprocessor::preprocess_image_with_opts(
            &png,
            &cpu_enc.config,
            None,
        )
        .expect("preprocess");
        let (gw, gh) = (pre.grid_w, pre.grid_h);

        let (cpu_med, _) = time_ms(warmup, runs, || {
            cpu_enc
                .encode_image(&pre.pixels, gw, gh)
                .expect("cpu encode");
        });
        eprintln!("\n  {side}px → grid {gw}x{gh} = {} patches", gw * gh);
        eprintln!("    CPU    : median {cpu_med:8.2} ms");

        for (label, enc) in [("wgpu ", &wgpu_enc), ("Metal", &metal_enc)] {
            let Some(gpu) = enc else { continue };
            // Trial encode first: a too-large grid (> MAX_VIT_TOKENS) or device
            // error should report cleanly rather than panic inside the timer.
            if let Err(e) = gpu.encode_image(&pre.pixels, gw, gh) {
                eprintln!("    {label}  : encode failed: {e:#}");
                continue;
            }
            let (med, _) = time_ms(warmup, runs, || {
                gpu.encode_image(&pre.pixels, gw, gh).expect("gpu encode");
            });
            eprintln!(
                "    {label}  : median {med:8.2} ms   ({:.2}x vs CPU)",
                cpu_med / med,
            );
        }
    }
    eprintln!();
}

//! The GPU vision tower on a real image: wall time per encode, and its distance from the CPU encoder.
//!
//! Builds the GPU encoder from the mmproj, encodes the image repeatedly (the first run includes
//! pipeline compilation and buffer allocation, so it is reported separately), then compares the last
//! output with the CPU encoder's f32 result. `CERA_GPU_PROFILE=1` adds per-kernel GPU times.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example vit_gpu_probe --features gpu,mmap,vl-preprocess
//! adb push target/aarch64-linux-android/release/examples/vit_gpu_probe /data/local/tmp/cmp-cera/
//! adb shell 'cd /data/local/tmp/cmp-cera && ./vit_gpu_probe mmproj.gguf image.jpg [runs]'
//! ```

#[cfg(all(feature = "gpu", feature = "mmap", feature = "vl-preprocess"))]
#[path = "common/vit_probe_ref.rs"]
mod vit_probe_ref;

#[cfg(all(feature = "gpu", feature = "mmap", feature = "vl-preprocess"))]
fn main() {
    use std::sync::Arc;
    use std::time::Instant;

    use cera::engine::BackendPreference;
    use cera::gguf::GgufFile;
    use cera::model::vision_encoder::VisionEncoderWeights;
    use cera::model::vision_encoder_gpu::build_gpu_vision_encoder;
    use cera::model::vision_preprocessor::preprocess_image_with_opts;

    let mut args = std::env::args().skip(1);
    let mmproj = args
        .next()
        .expect("usage: vit_gpu_probe MMPROJ IMAGE [RUNS]");
    let image = args
        .next()
        .expect("usage: vit_gpu_probe MMPROJ IMAGE [RUNS]");
    let runs: usize = args.next().map_or(8, |v| v.parse().expect("runs"));
    let reference_out = vit_probe_ref::reference_out();
    let gguf = Arc::new(GgufFile::open(std::path::Path::new(&mmproj)).expect("open mmproj"));
    let weights = VisionEncoderWeights::from_gguf(&gguf).expect("vision weights");
    let bytes = std::fs::read(&image).expect("read image");
    let pre = preprocess_image_with_opts(&bytes, &weights.config, None).expect("preprocess");
    println!(
        "{}x{} px, {} patches",
        pre.target_w,
        pre.target_h,
        pre.grid_w * pre.grid_h
    );

    // The reference child (see `f32_reference`): the CPU f32 embedding, written out raw, no GPU.
    if let Some(path) = reference_out {
        let out = weights
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .expect("cpu encode");
        vit_probe_ref::write_reference(path.as_ref(), &out);
        return;
    }

    let t = Instant::now();
    let gpu = build_gpu_vision_encoder(&weights, BackendPreference::Gpu).expect("no GPU encoder");
    println!(
        "GPU encoder built in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    // CERA_PROBE_WARM_LONG=<px>: encode a smaller version of the image first (long side capped at <px>),
    // to see how much of the first encode's extra time is size-independent one-time setup.
    if let Some(px) = std::env::var("CERA_PROBE_WARM_LONG")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        let small =
            preprocess_image_with_opts(&bytes, &weights.config, Some(px)).expect("preprocess");
        let t = Instant::now();
        gpu.encode_image(&small.pixels, small.grid_w, small.grid_h)
            .expect("encode");
        println!(
            "warm-up encode at {}x{} px: {:.1} ms",
            small.target_w,
            small.target_h,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut out = Vec::new();
    let mut times = Vec::new();
    for i in 0..runs {
        let t = Instant::now();
        out = gpu
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .expect("encode");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("run {i}: {ms:.1} ms");
        times.push(ms);
    }
    if times.len() > 1 {
        let mut rest = times[1..].to_vec();
        rest.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "median of runs after the first: {:.1} ms",
            rest[rest.len() / 2]
        );
    }

    let reference = vit_probe_ref::f32_reference(&mmproj, &image);
    assert_eq!(
        reference.len(),
        out.len(),
        "the reference embedding has another size"
    );
    let (mut dot, mut a2, mut b2, mut d2) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in out.iter().zip(&reference) {
        let (g, w) = (g as f64, w as f64);
        dot += g * w;
        a2 += g * g;
        b2 += w * w;
        d2 += (g - w) * (g - w);
    }
    println!(
        "GPU vs CPU f32: cosine {:.6}, relative RMS {:.3}% over {} floats",
        dot / (a2.sqrt() * b2.sqrt()),
        (d2 / b2).sqrt() * 100.0,
        out.len()
    );
}

#[cfg(not(all(feature = "gpu", feature = "mmap", feature = "vl-preprocess")))]
fn main() {
    eprintln!("build with --features gpu,mmap,vl-preprocess");
}

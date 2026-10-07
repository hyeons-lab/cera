//! The CPU vision tower on a real image: speed, and how far the int8 path is from f32.
//!
//! Encodes the image repeatedly with the int8 path on (`CERA_VIT_PROFILE=1` prints the per-phase
//! breakdown), then once with it off, and reports the relative RMS error and cosine between them.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example vit_cpu_probe --features mmap
//! adb push target/aarch64-linux-android/release/examples/vit_cpu_probe /data/local/tmp/cmp-cera/
//! adb shell 'cd /data/local/tmp/cmp-cera && CERA_VIT_PROFILE=1 ./vit_cpu_probe mmproj.gguf image.jpg'
//! ```

#[cfg(feature = "mmap")]
#[path = "common/vit_probe_ref.rs"]
mod vit_probe_ref;

#[cfg(feature = "mmap")]
fn main() {
    use std::sync::Arc;
    use std::time::Instant;

    use cera::gguf::GgufFile;
    use cera::model::vision_encoder::VisionEncoderWeights;
    use cera::model::vision_preprocessor::preprocess_image;

    let mut args = std::env::args().skip(1);
    let mmproj = args.next().expect("usage: vit_cpu_probe MMPROJ.gguf IMAGE");
    let image = args.next().expect("usage: vit_cpu_probe MMPROJ.gguf IMAGE");
    let reference_out = vit_probe_ref::reference_out();
    let t = Instant::now();
    let gguf = Arc::new(GgufFile::open(std::path::Path::new(&mmproj)).expect("open mmproj"));
    let weights = VisionEncoderWeights::from_gguf(&gguf).expect("vision weights");
    println!(
        "weights loaded in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let bytes = std::fs::read(&image).expect("read image");
    let pre = preprocess_image(&bytes, &weights.config).expect("preprocess");
    println!(
        "image {}x{} -> grid {}x{}",
        pre.target_w, pre.target_h, pre.grid_w, pre.grid_h
    );

    // The reference child (see `f32_reference`): one encode with the int8 path off, written out raw.
    if let Some(path) = reference_out {
        let t = Instant::now();
        let out = weights
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .expect("encode f32");
        println!("f32 run: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        vit_probe_ref::write_reference(path.as_ref(), &out);
        return;
    }

    let mut int8 = Vec::new();
    for i in 0..4 {
        let t = Instant::now();
        let out = weights
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .expect("encode");
        println!("int8 run {i}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        int8 = out;
    }
    let f32_out = vit_probe_ref::f32_reference(&mmproj, &image);
    assert_eq!(
        f32_out.len(),
        int8.len(),
        "the reference embedding has another size"
    );

    let (mut dot, mut a2, mut b2, mut d2) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in int8.iter().zip(&f32_out) {
        let (g, w) = (g as f64, w as f64);
        dot += g * w;
        a2 += g * g;
        b2 += w * w;
        d2 += (g - w) * (g - w);
    }
    println!(
        "{} embedding floats: int8 vs f32 cosine {:.6}, relative RMS {:.3}%",
        int8.len(),
        dot / (a2.sqrt() * b2.sqrt()),
        (d2 / b2).sqrt() * 100.0
    );
}

#[cfg(not(feature = "mmap"))]
fn main() {
    eprintln!("build with --features mmap");
}

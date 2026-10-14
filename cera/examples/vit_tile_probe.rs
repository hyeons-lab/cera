//! Per-chunk vision embeddings for a real image, to compare against llama.cpp's `mtmd` on the same
//! files.
//!
//! Preprocesses the image through `preprocess_image_layout` (tiles plus thumbnail for a large image,
//! one image otherwise), encodes every chunk on the CPU, and prints each chunk's statistics in the
//! format of llama.cpp's `MTMD_DEBUG_EMBEDDINGS` output (mean, std, min, max and the first 16 values of
//! token 0). The last chunk's embeddings are also written in that dump's layout (`[i32 n_tokens]
//! [i32 n_embd][f32 data]`) when a path is given, for a full comparison with
//! `MTMD_DEBUG_EMBEDDINGS=<path>`, which keeps only the last chunk.
//!
//! Run with `CERA_VIT_INT8=0` to compare against llama.cpp's f32 activations.
//!
//! ```text
//! CERA_VIT_INT8=0 cargo run --release -p cera --example vit_tile_probe --features mmap,vl-preprocess -- \
//!     mmproj.gguf image.jpg [last_chunk.bin]
//! ```

#[cfg(all(feature = "mmap", feature = "vl-preprocess"))]
fn main() {
    use std::sync::Arc;

    use cera::gguf::GgufFile;
    use cera::model::vision_encoder::VisionEncoderWeights;
    use cera::model::vision_preprocessor::{PreprocessedLayout, preprocess_image_layout};

    let mut args = std::env::args().skip(1);
    let mmproj = args
        .next()
        .expect("usage: vit_tile_probe MMPROJ IMAGE [DUMP]");
    let image = args
        .next()
        .expect("usage: vit_tile_probe MMPROJ IMAGE [DUMP]");
    let dump = args.next();
    let gguf = Arc::new(GgufFile::open(std::path::Path::new(&mmproj)).expect("open mmproj"));
    let weights = VisionEncoderWeights::from_gguf(&gguf).expect("vision weights");
    let bytes = std::fs::read(&image).expect("read image");
    // CERA_PROBE_SAVE_DECODED=<png>: write the decoded source as a lossless PNG, so another engine can
    // start from the same pixels and only the resize and tiling differ.
    if let Ok(path) = std::env::var("CERA_PROBE_SAVE_DECODED") {
        image::load_from_memory(&bytes)
            .expect("decode")
            .into_rgb8()
            .save(&path)
            .expect("save decoded png");
    }
    let layout = preprocess_image_layout(&bytes, &weights.config, None).expect("preprocess");

    let chunks: Vec<(String, &cera::model::PreprocessedImage)> = match &layout {
        PreprocessedLayout::Single(p) => vec![("image".into(), p)],
        PreprocessedLayout::Tiled(t) => {
            println!("tiled: {} columns x {} rows", t.cols, t.rows);
            let mut v: Vec<(String, &cera::model::PreprocessedImage)> = Vec::new();
            for (i, tile) in t.tiles.iter().enumerate() {
                v.push((
                    format!("tile row {} col {}", i / t.cols + 1, i % t.cols + 1),
                    tile,
                ));
            }
            v.push(("thumbnail".into(), &t.thumbnail));
            v
        }
    };
    let n_chunks = chunks.len();
    // CERA_PROBE_SAVE_LAST=<png>: write the last chunk's pixels back out as an 8-bit PNG (undoing the
    // mean/std normalization, which round-trips exactly), so another engine can embed the very same
    // pixels.
    if let Ok(path) = std::env::var("CERA_PROBE_SAVE_LAST") {
        let (_, pre) = chunks.last().expect("a chunk");
        let (w, h) = (pre.target_w, pre.target_h);
        let cfg = &weights.config;
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let px: Vec<u8> = (0..3)
                    .map(|c| {
                        let v = pre.pixels[c * w * h + y * w + x] * cfg.image_std[c]
                            + cfg.image_mean[c];
                        (v * 255.0).round().clamp(0.0, 255.0) as u8
                    })
                    .collect();
                img.put_pixel(x as u32, y as u32, image::Rgb([px[0], px[1], px[2]]));
            }
        }
        img.save(&path).expect("save png");
    }
    for (i, (name, pre)) in chunks.into_iter().enumerate() {
        let emb = weights
            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
            .expect("encode");
        let n_embd = weights.config.projection_dim;
        let n_tokens = emb.len() / n_embd;
        let (mut sum, mut sum_sq) = (0f64, 0f64);
        let (mut min, mut max) = (emb[0], emb[0]);
        for &v in &emb {
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            min = min.min(v);
            max = max.max(v);
        }
        let mean = sum / emb.len() as f64;
        let std = (sum_sq / emb.len() as f64 - mean * mean).sqrt();
        println!(
            "{name} ({}x{} px, {n_tokens} tokens): mean={mean:.6} std={std:.6} min={min:.6} max={max:.6}",
            pre.target_w, pre.target_h
        );
        let first: Vec<String> = emb[..16].iter().map(|v| format!("{v:.6}")).collect();
        println!("  token 0 first 16: {}", first.join(" "));
        if i + 1 == n_chunks
            && let Some(path) = &dump
        {
            let mut out = Vec::with_capacity(8 + emb.len() * 4);
            out.extend_from_slice(&(n_tokens as i32).to_le_bytes());
            out.extend_from_slice(&(n_embd as i32).to_le_bytes());
            for v in &emb {
                out.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(path, out).expect("write dump");
        }
    }
}

#[cfg(not(all(feature = "mmap", feature = "vl-preprocess")))]
fn main() {
    eprintln!("build with --features mmap,vl-preprocess");
}

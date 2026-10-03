//! The Hexagon ViT against the CPU one, on a device, at several image sizes.
//!
//! The NPU vision encoder runs the same blocks as the CPU encoder, so the same
//! pixels must give nearly the same embeddings. Sizes matter: a projection's
//! quantized activations take 36 bytes per element of VTCM, so the 3072-wide
//! feed-forward overflows the 8 MB VTCM past about 57 tokens unless the kernel
//! is told to walk the rows in chunks, and the fused Q/K/V kernel cannot chunk
//! at all. This checks every size end to end.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_vit_probe --features hexagon,mmap
//! adb push target/aarch64-linux-android/release/examples/hexagon_vit_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && ADSP_LIBRARY_PATH=$PWD ./hexagon_vit_probe /data/local/tmp/mmproj-LFM2-VL-450M-Q8_0.gguf'
//! ```

#[cfg(feature = "hexagon")]
fn main() {
    use std::sync::Arc;

    use cera::gguf::GgufFile;
    use cera::model::vision_encoder::VisionEncoderWeights;
    use cera::model::vision_encoder_gpu::VisionGpuEncode;
    use cera::model::vision_encoder_hexagon::try_new_hexagon_vision_encoder;

    let path = std::env::args()
        .nth(1)
        .expect("usage: hexagon_vit_probe MMPROJ.gguf");
    let gguf = Arc::new(GgufFile::open(std::path::Path::new(&path)).expect("open mmproj"));
    let weights = VisionEncoderWeights::from_gguf(&gguf).expect("vision weights");
    let npu = try_new_hexagon_vision_encoder(&weights).expect("Hexagon vision encoder");
    let p = weights.config.patch_size;
    println!(
        "{} blocks, {} wide, patch {p}, scale factor {}",
        weights.config.n_layer, weights.config.n_embd, weights.config.scale_factor
    );

    let rel = |got: &[f32], want: &[f32]| {
        let (mut dot, mut a2, mut b2, mut d2) = (0f64, 0f64, 0f64, 0f64);
        for (&g, &w) in got.iter().zip(want) {
            let (g, w) = (g as f64, w as f64);
            dot += g * w;
            a2 += g * g;
            b2 += w * w;
            d2 += (g - w) * (g - w);
        }
        (dot / (a2.sqrt() * b2.sqrt()), (d2 / b2).sqrt())
    };
    // Block by block on one size, so the first block that differs names the bug.
    {
        let (gw, gh) = (8usize, 8usize);
        let n = gw * p * gh * p * 3;
        let pixels: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.0007;
                0.6 * x.sin() + 0.3 * (x * 3.7).cos() + 0.1 * ((i % 97) as f32 / 97.0 - 0.5)
            })
            .collect();
        println!("tokens after k blocks, {gw}x{gh}:");
        for k in [0usize, 1, 2, 3, 4, 6, 8, 12] {
            let want = weights
                .debug_blocks(&pixels, gw, gh, k)
                .expect("CPU blocks");
            match npu.debug_blocks(&pixels, gw, gh, k) {
                Ok(got) => {
                    let (c, r) = rel(&got, &want);
                    println!("  k={k:<2} cosine {c:.6}  rel-rms {r:.3e}");
                }
                Err(e) => println!("  k={k:<2} NPU error: {e}"),
            }
        }
    }

    // The first block's stages: the first one that differs is the culprit.
    {
        let (gw, gh) = (8usize, 8usize);
        let n = gw * p * gh * p * 3;
        let pixels: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.0007;
                0.6 * x.sin() + 0.3 * (x * 3.7).cos() + 0.1 * ((i % 97) as f32 / 97.0 - 0.5)
            })
            .collect();
        let want = weights
            .debug_first_block(&pixels, gw, gh)
            .expect("CPU block");
        match npu.debug_first_block(&pixels, gw, gh) {
            Err(e) => println!("first block: NPU error: {e}"),
            Ok(got) => {
                println!("first block stages ({gw}x{gh}):");
                for (name, g, w) in [
                    ("x0 (input)", &got.x0, &want.x0),
                    ("q", &got.q, &want.q),
                    ("k", &got.k, &want.k),
                    ("v", &got.v, &want.v),
                    ("attention output", &got.attn_out, &want.attn_out),
                    ("output projection", &got.attn_proj, &want.attn_proj),
                    ("layernorm 2", &got.ln2, &want.ln2),
                    ("ffn gelu(up)", &got.ffn_mid, &want.ffn_mid),
                    ("ffn down", &got.ffn_out, &want.ffn_out),
                    ("block output", &got.tokens, &want.tokens),
                ] {
                    let (c, r) = rel(g, w);
                    println!("  {name:<18} cosine {c:.6}  rel-rms {r:.3e}");
                }
                // Where in the sequence is the feed-forward's error?
                let width = got.ffn_out.len() / (gw * gh);
                let per_token: Vec<String> = (0..gw * gh)
                    .step_by(8)
                    .map(|t| {
                        let rows = t * width..((t + 8).min(gw * gh)) * width;
                        let (_, r) = rel(&got.ffn_out[rows.clone()], &want.ffn_out[rows]);
                        format!("{t}:{r:.2}")
                    })
                    .collect();
                println!(
                    "  ffn down rel-rms by 8-token group: {}",
                    per_token.join(" ")
                );
            }
        }
    }

    let sf = weights.config.scale_factor;
    // (grid width, grid height): multiples of the scale factor, small to large.
    for &(gw, gh) in &[
        (8, 8),
        (8, 10),
        (12, 12),
        (16, 12),
        (16, 16),
        (20, 16),
        (24, 24),
        (32, 24),
    ] {
        let (gw, gh) = (gw / sf * sf, gh / sf * sf);
        if gw * gh == 0 {
            continue;
        }
        // A smooth deterministic pattern, normalized the way the encoder expects.
        let n = gw * p * gh * p * 3;
        let pixels: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.0007;
                0.6 * x.sin() + 0.3 * (x * 3.7).cos() + 0.1 * ((i % 97) as f32 / 97.0 - 0.5)
            })
            .collect();
        let want = weights.encode_image(&pixels, gw, gh).expect("CPU encode");
        match npu.encode_image(&pixels, gw, gh) {
            Err(e) => println!("{gw:>2}x{gh:<2} ({:>4} patches): NPU error: {e}", gw * gh),
            Ok(got) => {
                let (mut dot, mut a2, mut b2, mut d2) = (0f64, 0f64, 0f64, 0f64);
                for (&g, &w) in got.iter().zip(&want) {
                    let (g, w) = (g as f64, w as f64);
                    dot += g * w;
                    a2 += g * g;
                    b2 += w * w;
                    d2 += (g - w) * (g - w);
                }
                // The same pixels again, in this process: the whole forward is one
                // long DSP batch, the kind that once ran differently every time.
                let again = npu.encode_image(&pixels, gw, gh).expect("NPU repeat");
                let same = again.len() == got.len()
                    && again
                        .iter()
                        .zip(&got)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                // A fingerprint of the embedding bits, to compare across runs.
                let fingerprint = got.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                    (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
                });
                println!(
                    "{gw:>2}x{gh:<2} ({:>4} patches -> {:>4} tokens): cosine {:.6}  rel-rms {:.3e}  repeat {}  bits {fingerprint:016x}",
                    gw * gh,
                    want.len() / weights.config.projection_dim.max(1),
                    dot / (a2.sqrt() * b2.sqrt()),
                    (d2 / b2).sqrt(),
                    if same { "identical" } else { "DIFFERS" }
                );
            }
        }
    }
}

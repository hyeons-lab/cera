//! Image decode + dynamic-resolution resize + normalize for VL
//! input.
//!
//! Takes raw PNG / JPEG bytes and produces an
//! aspect-preserving-resized `[3 × H × W]` f32 NCHW tensor — the
//! layout
//! [`crate::model::vision_encoder::VisionEncoderWeights::encode_image`]
//! expects. The output `(W, H)` are picked by
//! [`calc_size_preserved_ratio`] to land within the encoder's
//! `[image_min_pixels, image_max_pixels]` band while keeping the
//! original aspect ratio and being divisible by
//! `patch_size · scale_factor` (so the patch grid + 2× pixel
//! shuffle work out cleanly).
//!
//! Hardcoded per `InferenceType::LlamaCppImageToText`:
//! - mean / std: from `cfg.image_mean` / `cfg.image_std` (read at
//!   load time from `clip.vision.image_{mean,std}` GGUF metadata).
//! - resize filter: bilinear (`Triangle`) — matches llama.cpp's
//!   `RESIZE_ALGO_BILINEAR` for `PROJECTOR_TYPE_LFM2`.
//! - pixel bounds: `cfg.image_min_pixels` / `cfg.image_max_pixels`
//!   (LFM2-VL: 65 536 / 262 144 pixels = 256² / 512² square
//!   baselines, but inputs need not be square).
//!
//! Gated behind the `vl-preprocess` feature so embedded targets
//! that only do text or raw-PCM audio input can drop the `image`
//! crate dep.

#![cfg(feature = "vl-preprocess")]

use crate::model::vision_encoder::VisionEncoderConfig;
use crate::session::CeraError;

/// Bytes the preprocessor produces — the f32 NCHW tensor plus the
/// dynamic patch grid that the encoder needs to interpret it.
/// `pixels.len() == 3 · target_h · target_w`.
#[derive(Debug)]
pub struct PreprocessedImage {
    /// `[3 · target_h · target_w]` f32 NCHW (R/G/B, `c·H·W + y·W +
    /// x` indexing).
    pub pixels: Vec<f32>,
    /// Resized image width in pixels (always a multiple of
    /// `cfg.patch_size · cfg.scale_factor`).
    pub target_w: usize,
    /// Resized image height in pixels (always a multiple of
    /// `cfg.patch_size · cfg.scale_factor`).
    pub target_h: usize,
    /// Patch grid width = `target_w / cfg.patch_size`.
    pub grid_w: usize,
    /// Patch grid height = `target_h / cfg.patch_size`.
    pub grid_h: usize,
}

/// Pick the smallest aspect-preserving resize of `(width, height)`
/// that lands within `[min_pixels, max_pixels]` and is divisible by
/// `align_size` on both axes. Mirrors llama.cpp's
/// `img_tool::calc_size_preserved_ratio` (lines 144-168 of
/// `tools/mtmd/mtmd-image.cpp`):
///
/// ```text
/// align_size = patch_size · scale_factor          (e.g. 16·2=32)
/// w_bar = max(align, round_to_multiple(width,  align))
/// h_bar = max(align, round_to_multiple(height, align))
/// if h_bar · w_bar > max_pixels:
///     β = sqrt(width · height / max_pixels)        ← scale down
///     w_bar = max(align, floor_to_multiple(width  / β, align))
///     h_bar = max(align, floor_to_multiple(height / β, align))
/// elif h_bar · w_bar < min_pixels:
///     β = sqrt(min_pixels / (width · height))      ← scale up
///     w_bar = ceil_to_multiple(width  · β, align)
///     h_bar = ceil_to_multiple(height · β, align)
/// ```
///
/// The asymmetry (round → floor on overshoot, ceil on undershoot)
/// is deliberate: rounding can leave you slightly over the
/// max-pixel cap, so the corrective branch must floor.
pub fn calc_size_preserved_ratio(
    width: usize,
    height: usize,
    align_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    debug_assert!(align_size > 0);
    debug_assert!(min_pixels <= max_pixels);
    // `area` divides into `beta` in the scale-up branch — guard
    // against `width == 0 || height == 0` (the `image` crate
    // rejects zero-pixel inputs upstream, but `calc_size_preserved_ratio`
    // is `pub` and could be called directly). Falling through with
    // `align_size × align_size` is the smallest aligned grid the
    // encoder can consume, which is the right "give up gracefully"
    // answer for an empty input.
    if width == 0 || height == 0 {
        return (align_size, align_size);
    }
    let round_by = |x: f64| ((x / align_size as f64).round() as usize) * align_size;
    let floor_by = |x: f64| ((x / align_size as f64).floor() as usize) * align_size;
    let ceil_by = |x: f64| ((x / align_size as f64).ceil() as usize) * align_size;

    let mut w_bar = align_size.max(round_by(width as f64));
    let mut h_bar = align_size.max(round_by(height as f64));

    let area = (width as f64) * (height as f64);
    // `saturating_mul` keeps the comparison correct on 32-bit
    // targets (wasm32) where huge inputs could overflow `usize`.
    // Saturation pushes the value to `usize::MAX`, which routes us
    // into the "scale down" branch — the safe direction.
    let area_check = h_bar.saturating_mul(w_bar);
    if area_check > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        w_bar = align_size.max(floor_by((width as f64) / beta));
        h_bar = align_size.max(floor_by((height as f64) / beta));
    } else if area_check < min_pixels {
        let beta = (min_pixels as f64 / area).sqrt();
        w_bar = ceil_by((width as f64) * beta);
        h_bar = ceil_by((height as f64) * beta);
    }
    (w_bar, h_bar)
}

/// Hard cap on a decoded image's width / height, applied as an
/// `image::Limits` dimension bound so a malformed or hostile file that
/// declares enormous dimensions is rejected before its pixel buffer is
/// allocated. 16384 px per side comfortably covers real photographs
/// while bounding the worst case; the `image` crate's default 512 MiB
/// `max_alloc` is the secondary backstop. See the decode site in
/// [`preprocess_image_with_opts`].
const MAX_DECODE_DIM: u32 = 16_384;

/// Decode + dynamic-resolution resize + normalize an image into a
/// [`PreprocessedImage`] the encoder consumes. `bytes` may be PNG
/// or JPEG (auto-detected via `image::guess_format`); other
/// formats fall through to a typed `Backend` error from the
/// underlying `image` crate.
pub fn preprocess_image(
    bytes: &[u8],
    cfg: &VisionEncoderConfig,
) -> Result<PreprocessedImage, CeraError> {
    preprocess_image_with_opts(bytes, cfg, None)
}

/// Like [`preprocess_image`], but with an optional caller-supplied
/// cap (`max_long_size`) on the longest side of the **encoded** image.
///
/// When `Some(n)`, the resize target chosen by
/// [`calc_size_preserved_ratio`] is shrunk (aspect-preserving,
/// re-aligned to `patch_size · scale_factor`) so its longer side is at
/// most `n` pixels, and the image is then resampled **once**, straight
/// from its native dimensions to that target — there is no cascaded
/// downscale-then-upscale. The cap only ever *shrinks* the target (the
/// `long > cap` guard never upscales) and **takes precedence over
/// `cfg.image_min_pixels`**: a small `n` is an explicit request to
/// trade detail for cost, clamped only at one aligned patch block.
/// `None` (or `0`, or a target already within the cap) behaves
/// identically to [`preprocess_image`].
///
/// `max_long_size` caps the encoded resolution, not the *decode*: a
/// huge source image is still fully decoded (bounded by the
/// dimension/alloc limits applied below) before the target shrink, so
/// the cap is a quality/encode-cost knob, not a decode-memory bound.
pub fn preprocess_image_with_opts(
    bytes: &[u8],
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedImage, CeraError> {
    if bytes.is_empty() {
        return Err(CeraError::EmptyInput);
    }

    // Decode with explicit dimension limits. `bytes` may come from
    // untrusted callers (the FFI `appendImage` surface is reachable
    // from Kotlin/Swift/Flutter), so bound the declared dimensions to
    // reject decompression bombs before the full pixel buffer is
    // allocated. `max_long_size` is applied post-decode (it caps the
    // encode target, not the decode), so it cannot bound this — the
    // dimension limit must. The `image` crate's default `max_alloc`
    // (512 MiB) still applies on top as a secondary backstop.
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| CeraError::Backend(format!("image format detection failed: {e}")))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_DIM);
    limits.max_image_height = Some(MAX_DECODE_DIM);
    reader.limits(limits);
    let img = reader
        .decode()
        .map_err(|e| CeraError::Backend(format!("image decode failed: {e}")))?;

    // Pick the resize target via llama.cpp's algorithm from the NATIVE
    // decoded dims. align_size = patch_size · scale_factor guarantees
    // both grid_w and grid_h are even and the 2× pixel-shuffle works
    // out.
    let align = cfg.patch_size * cfg.scale_factor;
    let (mut target_w, mut target_h) = calc_size_preserved_ratio(
        img.width() as usize,
        img.height() as usize,
        align,
        cfg.image_min_pixels,
        cfg.image_max_pixels,
    );

    // Optional caller cap on the longest side of the encoded target.
    // Applied to the TARGET (not a pre-resize of the input) so the
    // single `resize_exact` below goes straight from native dims to the
    // final target — one resample, no cascaded downscale-then-upscale.
    // Shrinks only (`long > cap`), preserves aspect, re-aligns by
    // flooring, and clamps to at least one aligned block so the patch
    // grid stays valid. Deliberately takes precedence over
    // `image_min_pixels` (the caller is trading detail for cost).
    if let Some(cap) = max_long_size.filter(|&c| c > 0).map(|c| c as usize) {
        let long = target_w.max(target_h);
        if long > cap {
            let beta = cap as f64 / long as f64; // < 1.0 — shrink only
            let floor_align = |x: f64| align.max(((x / align as f64).floor() as usize) * align);
            target_w = floor_align(target_w as f64 * beta);
            target_h = floor_align(target_h as f64 * beta);
        }
    }

    debug_assert_eq!(target_w % cfg.patch_size, 0);
    debug_assert_eq!(target_h % cfg.patch_size, 0);

    // Bilinear (Triangle) resize — matches llama.cpp's
    // `RESIZE_ALGO_BILINEAR` for `PROJECTOR_TYPE_LFM2`. Resize the
    // `DynamicImage` (in its native pixel format) before converting
    // to RGB — a 4096×4096 RGBA8 input would otherwise allocate a
    // ~50 MB intermediate `to_rgb8` buffer just to throw it away on
    // resize. Resizing first means we only `into_rgb8` the
    // post-resize buffer, which is bounded by `image_max_pixels`.
    let rgb = if img.width() == target_w as u32 && img.height() == target_h as u32 {
        img.into_rgb8()
    } else {
        img.resize_exact(
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::Triangle,
        )
        .into_rgb8()
    };

    // Normalize: NCHW f32, `(rgb / 255 - mean) / std` per channel.
    let h = rgb.height() as usize;
    let w = rgb.width() as usize;
    debug_assert_eq!(h, target_h);
    debug_assert_eq!(w, target_w);
    let mut pixels = vec![0f32; 3 * h * w];
    let raw = rgb.as_raw(); // [h * w * 3] u8 in row-major HWC
    for c in 0..3 {
        let mean = cfg.image_mean[c];
        let std_inv = 1.0 / cfg.image_std[c];
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * 3 + c;
                let dst = c * h * w + y * w + x;
                let pixel = raw[src] as f32 / 255.0;
                pixels[dst] = (pixel - mean) * std_inv;
            }
        }
    }
    Ok(PreprocessedImage {
        pixels,
        target_w: w,
        target_h: h,
        grid_w: w / cfg.patch_size,
        grid_h: h / cfg.patch_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn synth_cfg() -> VisionEncoderConfig {
        VisionEncoderConfig {
            n_layer: 12,
            n_embd: 768,
            n_ff: 3072,
            n_head: 12,
            eps: 1e-6,
            image_size: 4,
            patch_size: 2,
            n_trained_patches: 4,
            projection_dim: 1024,
            scale_factor: 2,
            // Pick non-trivial mean / std so a mean/std swap or
            // channel reorder shows up loudly in the assertions.
            image_mean: [0.5, 0.4, 0.3],
            image_std: [0.2, 0.25, 0.5],
            // Bound the `synth_cfg` resize at exactly 4×4 = 16
            // pixels so the original lossless solid-red test
            // continues to land at 4×4 deterministically.
            image_min_pixels: 16,
            image_max_pixels: 16,
        }
    }

    /// `calc_size_preserved_ratio` round-trips llama.cpp's
    /// reference behaviour for the LFM2-VL pug case.
    #[test]
    fn calc_size_preserved_ratio_pug_shape() {
        // 1024×771 image, align=32, [min, max] = [65 536, 262 144].
        // Expected (576, 416) — matches mtmd-cli's verbose output
        // for the committed pug fixture.
        let (w, h) = calc_size_preserved_ratio(1024, 771, 32, 65_536, 262_144);
        assert_eq!((w, h), (576, 416));
        // Patch grid: 36 × 26 = 936 patches → 18 × 13 = 234
        // image tokens (after 2× pixel shuffle).
        assert_eq!((w / 16) * (h / 16), 936);
    }

    /// Tiny input — must scale up to clear `min_pixels`.
    #[test]
    fn calc_size_preserved_ratio_scales_up_small_input() {
        let (w, h) = calc_size_preserved_ratio(100, 100, 32, 65_536, 262_144);
        assert!(
            w * h >= 65_536,
            "scaled-up area {w}×{h} = {} should ≥ min_pixels (65 536)",
            w * h
        );
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    /// Banner — must scale down preserving aspect.
    #[test]
    fn calc_size_preserved_ratio_clamps_huge_input() {
        let (w, h) = calc_size_preserved_ratio(4096, 1024, 32, 65_536, 262_144);
        assert!(
            w * h <= 262_144,
            "scaled-down area {w}×{h} = {} should ≤ max_pixels (262 144)",
            w * h
        );
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        // 4:1 input aspect should produce a wide output.
        let aspect = w as f32 / h as f32;
        assert!(
            (3.5..=4.5).contains(&aspect),
            "expected ~4:1 aspect, got {aspect}"
        );
    }

    /// Already-aligned input within [min, max] band — output
    /// equals input.
    #[test]
    fn calc_size_preserved_ratio_passes_through_when_in_band() {
        let (w, h) = calc_size_preserved_ratio(256, 256, 32, 65_536, 262_144);
        assert_eq!((w, h), (256, 256));
    }

    /// Zero dimensions short-circuit to `(align, align)` instead of
    /// dividing by zero in the scale-up branch.
    #[test]
    fn calc_size_preserved_ratio_zero_dims_returns_align() {
        assert_eq!(
            calc_size_preserved_ratio(0, 100, 32, 65_536, 262_144),
            (32, 32)
        );
        assert_eq!(
            calc_size_preserved_ratio(100, 0, 32, 65_536, 262_144),
            (32, 32)
        );
        assert_eq!(
            calc_size_preserved_ratio(0, 0, 32, 65_536, 262_144),
            (32, 32)
        );
    }

    /// Synthesise a 4×4 solid red PNG, run through the
    /// preprocessor, and assert per-channel normalisation lands
    /// where expected. Red = (1.0, 0.0, 0.0) post-÷255, so:
    ///   R: (1.0 - 0.5) / 0.2  =  2.5
    ///   G: (0.0 - 0.4) / 0.25 = -1.6
    ///   B: (0.0 - 0.3) / 0.5  = -0.6
    #[test]
    fn preprocess_solid_red_normalises_per_channel() {
        let cfg = synth_cfg();
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(4, 4, |_, _| Rgb([255u8, 0, 0]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode test png");

        let pre = preprocess_image(&bytes, &cfg).expect("preprocess");
        assert_eq!(pre.target_w, 4);
        assert_eq!(pre.target_h, 4);
        assert_eq!(pre.grid_w, 2);
        assert_eq!(pre.grid_h, 2);
        let out = pre.pixels;
        assert_eq!(out.len(), 3 * 4 * 4);
        let n = 4 * 4;
        for &v in &out[0..n] {
            assert!((v - 2.5).abs() < 1e-5, "R channel: {v}");
        }
        for &v in &out[n..2 * n] {
            assert!((v - (-1.6)).abs() < 1e-5, "G channel: {v}");
        }
        for &v in &out[2 * n..3 * n] {
            assert!((v - (-0.6)).abs() < 1e-5, "B channel: {v}");
        }
    }

    #[test]
    fn preprocess_empty_bytes_errors() {
        let cfg = synth_cfg();
        match preprocess_image(&[], &cfg) {
            Err(CeraError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    /// `max_long_size` cap downscales a large input before the model
    /// clamp, while `None` leaves it at native resolution. Uses a wide
    /// pixel band so the cap (not `image_max_pixels`) is what binds.
    #[test]
    fn preprocess_max_long_size_caps_long_side() {
        let cfg = VisionEncoderConfig {
            // align = patch_size · scale_factor = 2, so targets stay
            // multiples of 2 and the assertions below are exact.
            patch_size: 2,
            scale_factor: 1,
            // Wide band: an 800×400 image (320k px) sits inside it, so
            // without a cap the model resize is a pass-through and the
            // cap is the only thing that can change the output size.
            image_min_pixels: 4,
            image_max_pixels: 1_000_000,
            ..synth_cfg()
        };
        // 800×400 solid image (2:1 aspect).
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(800, 400, |_, _| Rgb([128u8, 64, 32]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode test png");

        // No cap → native 800×400 (within band, multiple of 2).
        let uncapped = preprocess_image_with_opts(&bytes, &cfg, None).expect("uncapped");
        assert_eq!((uncapped.target_w, uncapped.target_h), (800, 400));

        // Cap=100 → long side downscaled to 100, aspect preserved → 100×50.
        let capped = preprocess_image_with_opts(&bytes, &cfg, Some(100)).expect("capped");
        assert_eq!((capped.target_w, capped.target_h), (100, 50));

        // A cap larger than the image is a no-op (no upscale here).
        let big_cap = preprocess_image_with_opts(&bytes, &cfg, Some(4000)).expect("big cap");
        assert_eq!((big_cap.target_w, big_cap.target_h), (800, 400));

        // Cap of 0 is treated as "no cap".
        let zero_cap = preprocess_image_with_opts(&bytes, &cfg, Some(0)).expect("zero cap");
        assert_eq!((zero_cap.target_w, zero_cap.target_h), (800, 400));
    }

    /// The cap takes precedence over `image_min_pixels` and shrinks the
    /// target *below* the model's floor without any upscale-back — the
    /// regression guard for the old cascaded downscale→upscale bug.
    #[test]
    fn preprocess_max_long_size_takes_precedence_over_min_pixels() {
        let cfg = VisionEncoderConfig {
            // align = 16 · 2 = 32 (a realistic LFM2-VL grid).
            patch_size: 16,
            scale_factor: 2,
            // Real LFM2-VL band: 256² floor, 512² ceiling.
            image_min_pixels: 65_536,
            image_max_pixels: 262_144,
            ..synth_cfg()
        };
        // 256×256 sits exactly on the min_pixels floor → uncapped is a
        // pass-through at 256×256.
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(256, 256, |_, _| Rgb([200u8, 100, 50]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode test png");

        let uncapped = preprocess_image_with_opts(&bytes, &cfg, None).expect("uncapped");
        assert_eq!((uncapped.target_w, uncapped.target_h), (256, 256));

        // Cap=128 forces the target to 128×128 (= 16384 px, *below* the
        // 65536 min_pixels floor). The old impl pre-resized to 128 then
        // let the min_pixels branch upscale it back to 256; the fixed
        // impl shrinks the target and resizes once, so it must stay at
        // 128×128 and below the floor.
        let capped = preprocess_image_with_opts(&bytes, &cfg, Some(128)).expect("capped");
        assert_eq!((capped.target_w, capped.target_h), (128, 128));
        assert!(
            capped.target_w * capped.target_h < cfg.image_min_pixels,
            "cap must take precedence over min_pixels (no upscale-back); got {}×{}",
            capped.target_w,
            capped.target_h,
        );
    }

    /// Resize path: small JPEG (8×8) → resized to 4×4 (the
    /// synth_cfg pixel band). Verifies the auto-detect dispatch +
    /// the resize branch fires when input dims don't match.
    #[test]
    fn preprocess_jpeg_resizes_to_target() {
        let cfg = synth_cfg();
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(8, 8, |_, _| Rgb([255u8, 0, 0]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Jpeg,
            )
            .expect("encode test jpeg");

        let pre = preprocess_image(&bytes, &cfg).expect("preprocess");
        assert_eq!(pre.target_w, 4);
        assert_eq!(pre.target_h, 4);
        assert_eq!(pre.pixels.len(), 3 * 4 * 4);
        let n = 4 * 4;
        let r_avg = pre.pixels[0..n].iter().sum::<f32>() / (n as f32);
        assert!((r_avg - 2.5).abs() < 0.1, "R channel mean: {r_avg}");
    }
}

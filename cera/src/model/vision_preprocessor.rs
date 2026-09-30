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
#[derive(Debug, Clone, PartialEq)]
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

pub use crate::model::PixelFormat;

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
/// most `n` pixels — **except** that each dimension is floored at one
/// aligned block (`patch_size · scale_factor`), so when
/// `n < patch_size · scale_factor` the encoded long side rounds up to
/// that minimum rather than going below it. The image is then resampled
/// **once**, straight from its native dimensions to that target — there
/// is no cascaded downscale-then-upscale. The cap only ever *shrinks*
/// the target (the `long > cap` guard never upscales) and **takes
/// precedence over `cfg.image_min_pixels`**: a small `n` is an explicit
/// request to trade detail for cost, clamped only at one aligned patch
/// block.
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
    // final target: one resample, no cascaded downscale-then-upscale.
    // Shrinks only (`long > cap`), preserves aspect, re-aligns by
    // flooring, and clamps to at least one aligned block so the patch
    // grid stays valid. Deliberately takes precedence over
    // `image_min_pixels` (the caller is trading detail for cost).
    if let Some(cap) = max_long_size.filter(|&c| c > 0).map(|c| c as usize) {
        let long = target_w.max(target_h);
        if long > cap {
            let beta = cap as f64 / long as f64; // < 1.0: shrink only
            let floor_align = |x: f64| align.max(((x / align as f64).floor() as usize) * align);
            target_w = floor_align(target_w as f64 * beta);
            target_h = floor_align(target_h as f64 * beta);
        }
    }

    debug_assert_eq!(target_w % cfg.patch_size, 0);
    debug_assert_eq!(target_h % cfg.patch_size, 0);

    // Fast path: reuse decoded ImageRgb8 raw buffer when dimensions already match target.
    let rgb_bytes = match img {
        image::DynamicImage::ImageRgb8(rgb) => {
            if rgb.width() as usize == target_w && rgb.height() as usize == target_h {
                rgb.into_raw()
            } else {
                resize_bilinear_rgb(
                    rgb.as_raw(),
                    rgb.width() as usize,
                    rgb.height() as usize,
                    PixelFormat::Rgb8,
                    target_w,
                    target_h,
                )?
            }
        }
        image::DynamicImage::ImageRgba8(rgba) => resize_bilinear_rgb(
            rgba.as_raw(),
            rgba.width() as usize,
            rgba.height() as usize,
            PixelFormat::Rgba8,
            target_w,
            target_h,
        )?,
        other => {
            let rgb = other.to_rgb8();
            resize_bilinear_rgb(
                rgb.as_raw(),
                rgb.width() as usize,
                rgb.height() as usize,
                PixelFormat::Rgb8,
                target_w,
                target_h,
            )?
        }
    };

    let pixels = normalize_rgb8_to_nchw_f32(
        &rgb_bytes,
        target_w,
        target_h,
        &cfg.image_mean,
        &cfg.image_std,
    );

    Ok(PreprocessedImage {
        pixels,
        target_w,
        target_h,
        grid_w: target_w / cfg.patch_size,
        grid_h: target_h / cfg.patch_size,
    })
}

/// Preprocess an uncompressed raw pixel buffer (e.g. from an Android Bitmap
/// or camera frame) into a [`PreprocessedImage`] ready for vision encoding.
///
/// Bypasses all image decompression overhead and applies fast bilinear
/// resampling and SIMD/parallel NCHW normalization.
pub fn preprocess_raw_pixels(
    pixels: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedImage, CeraError> {
    if pixels.is_empty() || width == 0 || height == 0 {
        return Err(CeraError::EmptyInput);
    }

    let bpp = format.bytes_per_pixel();
    let min_src_len = width
        .checked_mul(height)
        .and_then(|px| px.checked_mul(bpp))
        .ok_or_else(|| CeraError::Backend("image dimensions overflow usize".into()))?;
    if pixels.len() < min_src_len {
        return Err(CeraError::Backend(format!(
            "preprocess_raw_pixels: buffer length {} is smaller than required {} ({}x{} @ {} bpp)",
            pixels.len(),
            min_src_len,
            width,
            height,
            bpp,
        )));
    }

    let align = cfg.patch_size * cfg.scale_factor;
    let (mut target_w, mut target_h) = calc_size_preserved_ratio(
        width,
        height,
        align,
        cfg.image_min_pixels,
        cfg.image_max_pixels,
    );

    if let Some(cap) = max_long_size.filter(|&c| c > 0).map(|c| c as usize) {
        let long = target_w.max(target_h);
        if long > cap {
            let beta = cap as f64 / long as f64;
            let floor_align = |x: f64| align.max(((x / align as f64).floor() as usize) * align);
            target_w = floor_align(target_w as f64 * beta);
            target_h = floor_align(target_h as f64 * beta);
        }
    }

    debug_assert_eq!(target_w % cfg.patch_size, 0);
    debug_assert_eq!(target_h % cfg.patch_size, 0);

    let rgb_bytes = if width == target_w && height == target_h && format == PixelFormat::Rgb8 {
        pixels[..3 * target_w * target_h].to_vec()
    } else {
        resize_bilinear_rgb(pixels, width, height, format, target_w, target_h)?
    };

    let norm_pixels = normalize_rgb8_to_nchw_f32(
        &rgb_bytes,
        target_w,
        target_h,
        &cfg.image_mean,
        &cfg.image_std,
    );

    Ok(PreprocessedImage {
        pixels: norm_pixels,
        target_w,
        target_h,
        grid_w: target_w / cfg.patch_size,
        grid_h: target_h / cfg.patch_size,
    })
}

/// Resample an uncompressed pixel buffer in any supported [`PixelFormat`]
/// to an interleaved 24-bit RGB8 buffer of dimensions `target_w x target_h`
/// using bilinear interpolation.
///
/// Precomputes horizontal sample coordinates and interpolation weights once,
/// and parallelizes row-wise across available threads when the `parallel`
/// feature is enabled.
pub fn resize_bilinear_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    format: PixelFormat,
    target_w: usize,
    target_h: usize,
) -> Result<Vec<u8>, CeraError> {
    if src_w == 0 || src_h == 0 || target_w == 0 || target_h == 0 {
        return Err(CeraError::EmptyInput);
    }
    let bpp = format.bytes_per_pixel();
    let min_src_len = src_w
        .checked_mul(src_h)
        .and_then(|px| px.checked_mul(bpp))
        .ok_or_else(|| CeraError::Backend("image dimensions overflow usize".into()))?;
    if src.len() < min_src_len {
        return Err(CeraError::Backend(format!(
            "resize_bilinear_rgb: source buffer length {} is smaller than required {} ({}x{} @ {} bpp)",
            src.len(),
            min_src_len,
            src_w,
            src_h,
            bpp,
        )));
    }

    if src_w == target_w && src_h == target_h {
        if format == PixelFormat::Rgb8 {
            return Ok(src[..min_src_len].to_vec());
        }
        let (r_off, g_off, b_off) = format.channel_offsets();
        let mut dst = vec![0u8; target_w * target_h * 3];
        for i in 0..target_w * target_h {
            let s = &src[i * bpp..];
            dst[i * 3] = s[r_off];
            dst[i * 3 + 1] = s[g_off];
            dst[i * 3 + 2] = s[b_off];
        }
        return Ok(dst);
    }

    let (r_off, g_off, b_off) = format.channel_offsets();
    let src_stride = src_w * bpp;
    let dst_stride = target_w * 3;
    let total_dst = target_h
        .checked_mul(dst_stride)
        .ok_or_else(|| CeraError::Backend("target buffer size overflow usize".into()))?;
    let mut dst = vec![0u8; total_dst];

    let x_scale = src_w as f32 / target_w as f32;
    let max_x = (src_w.saturating_sub(1)) as f32;
    let x_table: Vec<(usize, usize, f32, f32)> = if src_w == 1 {
        vec![(0, 0, 1.0, 0.0); target_w]
    } else {
        (0..target_w)
            .map(|dx| {
                let sx = ((dx as f32 + 0.5) * x_scale - 0.5).clamp(0.0, max_x);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(src_w.saturating_sub(1));
                let wx1 = sx - x0 as f32;
                let wx0 = 1.0 - wx1;
                (x0, x1, wx0, wx1)
            })
            .collect()
    };

    let y_scale = src_h as f32 / target_h as f32;
    let max_y = (src_h.saturating_sub(1)) as f32;

    let sample_row = |dy: usize, row_dst: &mut [u8]| {
        let (y0, y1, wy0, wy1) = if src_h == 1 {
            (0, 0, 1.0, 0.0)
        } else {
            let sy = ((dy as f32 + 0.5) * y_scale - 0.5).clamp(0.0, max_y);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(src_h.saturating_sub(1));
            let wy1 = sy - y0 as f32;
            let wy0 = 1.0 - wy1;
            (y0, y1, wy0, wy1)
        };

        let row0 = &src[y0 * src_stride..(y0 + 1) * src_stride];
        let row1 = &src[y1 * src_stride..(y1 + 1) * src_stride];

        for (dx, &(x0, x1, wx0, wx1)) in x_table.iter().enumerate() {
            let p00 = &row0[x0 * bpp..];
            let p01 = &row0[x1 * bpp..];
            let p10 = &row1[x0 * bpp..];
            let p11 = &row1[x1 * bpp..];

            let p00_r = p00[r_off] as f32;
            let p00_g = p00[g_off] as f32;
            let p00_b = p00[b_off] as f32;

            let p01_r = p01[r_off] as f32;
            let p01_g = p01[g_off] as f32;
            let p01_b = p01[b_off] as f32;

            let p10_r = p10[r_off] as f32;
            let p10_g = p10[g_off] as f32;
            let p10_b = p10[b_off] as f32;

            let p11_r = p11[r_off] as f32;
            let p11_g = p11[g_off] as f32;
            let p11_b = p11[b_off] as f32;

            let top_r = wx0 * p00_r + wx1 * p01_r;
            let bot_r = wx0 * p10_r + wx1 * p11_r;
            let top_g = wx0 * p00_g + wx1 * p01_g;
            let bot_g = wx0 * p10_g + wx1 * p11_g;
            let top_b = wx0 * p00_b + wx1 * p01_b;
            let bot_b = wx0 * p10_b + wx1 * p11_b;

            let out_idx = dx * 3;
            row_dst[out_idx] = (wy0 * top_r + wy1 * bot_r + 0.5).clamp(0.0, 255.0) as u8;
            row_dst[out_idx + 1] = (wy0 * top_g + wy1 * bot_g + 0.5).clamp(0.0, 255.0) as u8;
            row_dst[out_idx + 2] = (wy0 * top_b + wy1 * bot_b + 0.5).clamp(0.0, 255.0) as u8;
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        dst.par_chunks_mut(dst_stride)
            .enumerate()
            .for_each(|(dy, row_dst)| {
                sample_row(dy, row_dst);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (dy, row_dst) in dst.chunks_mut(dst_stride).enumerate() {
            sample_row(dy, row_dst);
        }
    }

    Ok(dst)
}

/// Normalize an interleaved RGB8 buffer into a planar NCHW f32 tensor.
///
/// Output layout: `[3 x height x width]`, where:
/// - Channel 0 (R): `0 .. width * height`
/// - Channel 1 (G): `width * height .. 2 * width * height`
/// - Channel 2 (B): `2 * width * height .. 3 * width * height`
///
/// Uses NEON SIMD vectorization on aarch64 targets with fallback to
/// auto-vectorized loops, and parallelizes across worker threads when
/// `parallel` feature is active.
pub fn normalize_rgb8_to_nchw_f32(
    rgb: &[u8],
    width: usize,
    height: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    let n_pixels = width * height;
    let mut out = vec![0f32; 3 * n_pixels];
    let (r_plane, rest) = out.split_at_mut(n_pixels);
    let (g_plane, b_plane) = rest.split_at_mut(n_pixels);

    let scale_r = 1.0 / (255.0 * std[0]);
    let bias_r = -mean[0] / std[0];
    let scale_g = 1.0 / (255.0 * std[1]);
    let bias_g = -mean[1] / std[1];
    let scale_b = 1.0 / (255.0 * std[2]);
    let bias_b = -mean[2] / std[2];

    let process_chunk =
        |raw_chunk: &[u8], out_r: &mut [f32], out_g: &mut [f32], out_b: &mut [f32]| {
            let chunk_len = out_r.len();
            assert_eq!(
                raw_chunk.len(),
                chunk_len * 3,
                "raw_chunk length {} does not match required {} (3 * {})",
                raw_chunk.len(),
                chunk_len * 3,
                chunk_len
            );

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let vscale_r = vdupq_n_f32(scale_r);
                let vbias_r = vdupq_n_f32(bias_r);
                let vscale_g = vdupq_n_f32(scale_g);
                let vbias_g = vdupq_n_f32(bias_g);
                let vscale_b = vdupq_n_f32(scale_b);
                let vbias_b = vdupq_n_f32(bias_b);

                let mut i = 0;
                while i + 16 <= chunk_len {
                    let ptr = raw_chunk.as_ptr().add(i * 3);
                    let loaded = vld3q_u8(ptr);

                    let process_16 = |u8_vec: uint8x16_t,
                                      scale: float32x4_t,
                                      bias: float32x4_t,
                                      out_ptr: *mut f32| {
                        let u16_low = vmovl_u8(vget_low_u8(u8_vec));
                        let u32_0 = vmovl_u16(vget_low_u16(u16_low));
                        let u32_1 = vmovl_u16(vget_high_u16(u16_low));
                        let f0 = vmlaq_f32(bias, vcvtq_f32_u32(u32_0), scale);
                        let f1 = vmlaq_f32(bias, vcvtq_f32_u32(u32_1), scale);
                        vst1q_f32(out_ptr, f0);
                        vst1q_f32(out_ptr.add(4), f1);

                        let u16_high = vmovl_u8(vget_high_u8(u8_vec));
                        let u32_2 = vmovl_u16(vget_low_u16(u16_high));
                        let u32_3 = vmovl_u16(vget_high_u16(u16_high));
                        let f2 = vmlaq_f32(bias, vcvtq_f32_u32(u32_2), scale);
                        let f3 = vmlaq_f32(bias, vcvtq_f32_u32(u32_3), scale);
                        vst1q_f32(out_ptr.add(8), f2);
                        vst1q_f32(out_ptr.add(12), f3);
                    };

                    process_16(loaded.0, vscale_r, vbias_r, out_r.as_mut_ptr().add(i));
                    process_16(loaded.1, vscale_g, vbias_g, out_g.as_mut_ptr().add(i));
                    process_16(loaded.2, vscale_b, vbias_b, out_b.as_mut_ptr().add(i));

                    i += 16;
                }

                while i < chunk_len {
                    let src_idx = i * 3;
                    let r = raw_chunk[src_idx] as f32;
                    let g = raw_chunk[src_idx + 1] as f32;
                    let b = raw_chunk[src_idx + 2] as f32;
                    out_r[i] = r * scale_r + bias_r;
                    out_g[i] = g * scale_g + bias_g;
                    out_b[i] = b * scale_b + bias_b;
                    i += 1;
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                for i in 0..chunk_len {
                    let src_idx = i * 3;
                    let r = raw_chunk[src_idx] as f32;
                    let g = raw_chunk[src_idx + 1] as f32;
                    let b = raw_chunk[src_idx + 2] as f32;
                    out_r[i] = r * scale_r + bias_r;
                    out_g[i] = g * scale_g + bias_g;
                    out_b[i] = b * scale_b + bias_b;
                }
            }
        };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let num_threads = crate::par::current_num_threads();
        let chunk_size = ((n_pixels / num_threads).max(1024) / 16) * 16;
        if num_threads > 1 && n_pixels >= 4096 && chunk_size > 0 {
            r_plane
                .par_chunks_mut(chunk_size)
                .zip(g_plane.par_chunks_mut(chunk_size))
                .zip(b_plane.par_chunks_mut(chunk_size))
                .enumerate()
                .for_each(|(chunk_idx, ((chunk_r, chunk_g), chunk_b))| {
                    let pixel_start = chunk_idx * chunk_size;
                    let chunk_len = chunk_r.len();
                    let raw_chunk = &rgb[pixel_start * 3..(pixel_start + chunk_len) * 3];
                    process_chunk(raw_chunk, chunk_r, chunk_g, chunk_b);
                });
            return out;
        }
    }

    process_chunk(rgb, r_plane, g_plane, b_plane);
    out
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

    #[test]
    fn test_resize_bilinear_rgb_all_formats() {
        let (w, h) = (8, 8);
        let (tw, th) = (4, 4);
        let (r, g, b) = (200u8, 100u8, 50u8);

        let rgb_data: Vec<u8> = (0..w * h).flat_map(|_| [r, g, b]).collect();
        let rgba_data: Vec<u8> = (0..w * h).flat_map(|_| [r, g, b, 255]).collect();
        let bgr_data: Vec<u8> = (0..w * h).flat_map(|_| [b, g, r]).collect();
        let bgra_data: Vec<u8> = (0..w * h).flat_map(|_| [b, g, r, 255]).collect();

        for (fmt, data) in [
            (PixelFormat::Rgb8, &rgb_data),
            (PixelFormat::Rgba8, &rgba_data),
            (PixelFormat::Bgr8, &bgr_data),
            (PixelFormat::Bgra8, &bgra_data),
        ] {
            let out = resize_bilinear_rgb(data, w, h, fmt, tw, th).expect("resize");
            assert_eq!(out.len(), tw * th * 3);
            for &px in out.as_chunks::<3>().0 {
                assert_eq!(px, [r, g, b], "format {fmt:?} channel mismatch");
            }
        }
    }

    #[test]
    fn test_resize_bilinear_rgb_degenerate_1x1() {
        let src = [100u8, 150u8, 200u8];
        let out = resize_bilinear_rgb(&src, 1, 1, PixelFormat::Rgb8, 4, 4).expect("resize 1x1");
        assert_eq!(out.len(), 4 * 4 * 3);
        for &chunk in out.as_chunks::<3>().0 {
            assert_eq!(chunk, [100, 150, 200]);
        }
    }

    #[test]
    fn test_normalize_rgb8_to_nchw_f32_parity() {
        let (w, h) = (64, 64);
        let n_pixels = w * h;
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        let mut raw = Vec::with_capacity(n_pixels * 3);
        for i in 0..n_pixels {
            raw.push((i % 256) as u8);
            raw.push(((i * 7) % 256) as u8);
            raw.push(((i * 13) % 256) as u8);
        }

        let out = normalize_rgb8_to_nchw_f32(&raw, w, h, &mean, &std);
        assert_eq!(out.len(), 3 * n_pixels);

        for c in 0..3 {
            let m = mean[c];
            let s = std[c];
            let plane = &out[c * n_pixels..(c + 1) * n_pixels];
            for i in 0..n_pixels {
                let pixel_byte = raw[i * 3 + c];
                let expected = (pixel_byte as f32 / 255.0 - m) / s;
                let actual = plane[i];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "channel {c} pixel {i}: actual {actual} vs expected {expected}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "does not match required")]
    fn test_normalize_rgb8_to_nchw_f32_undersized_panics() {
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let short_raw = vec![0u8; 10];
        normalize_rgb8_to_nchw_f32(&short_raw, 4, 4, &mean, &std);
    }

    #[test]
    fn test_preprocess_raw_pixels_solid_color() {
        let cfg = synth_cfg();
        let (w, h) = (8, 8);
        let rgba_data: Vec<u8> = (0..w * h).flat_map(|_| [255u8, 0, 0, 255]).collect();
        let pre = preprocess_raw_pixels(&rgba_data, w, h, PixelFormat::Rgba8, &cfg, None)
            .expect("preprocess raw");
        assert_eq!(pre.target_w, 4);
        assert_eq!(pre.target_h, 4);
        assert_eq!(pre.grid_w, 2);
        assert_eq!(pre.grid_h, 2);

        let n = 4 * 4;
        for &v in &pre.pixels[0..n] {
            assert!((v - 2.5).abs() < 1e-5, "R channel: {v}");
        }
        for &v in &pre.pixels[n..2 * n] {
            assert!((v - (-1.6)).abs() < 1e-5, "G channel: {v}");
        }
        for &v in &pre.pixels[2 * n..3 * n] {
            assert!((v - (-0.6)).abs() < 1e-5, "B channel: {v}");
        }
    }

    #[test]
    fn test_preprocess_raw_pixels_validation() {
        let cfg = synth_cfg();
        assert!(matches!(
            preprocess_raw_pixels(&[], 4, 4, PixelFormat::Rgb8, &cfg, None),
            Err(CeraError::EmptyInput)
        ));
        assert!(matches!(
            preprocess_raw_pixels(&[0u8; 10], 4, 4, PixelFormat::Rgb8, &cfg, None),
            Err(CeraError::Backend(_))
        ));
    }
}

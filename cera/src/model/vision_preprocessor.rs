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
/// This always produces **one** image, never tiles: for the layout a large image
/// needs, use [`preprocess_image_layout`].
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
    let (rgb, w, h) = decode_rgb8(bytes)?;
    preprocess_single_rgb8(&rgb, w, h, cfg, max_long_size)
}

/// Like [`preprocess_image_with_opts`], but follows the LFM2-VL reference processor for large
/// images: one that is more than twice the single-image pixel budget becomes a grid of 512 px
/// tiles plus a thumbnail ([`PreprocessedLayout::Tiled`]), anything else a single image.
/// A caller cap (`max_long_size`) is an explicit request for a smaller image, so with one set
/// the result is always [`PreprocessedLayout::Single`] and matches
/// [`preprocess_image_with_opts`].
pub fn preprocess_image_layout(
    bytes: &[u8],
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedLayout, CeraError> {
    let (rgb, w, h) = decode_rgb8(bytes)?;
    preprocess_layout_rgb8(&rgb, w, h, cfg, max_long_size)
}

/// Decode PNG/JPEG bytes to an interleaved RGB8 buffer with its dimensions.
fn decode_rgb8(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), CeraError> {
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
    // Alpha is dropped, not composited, as `resize_bilinear_rgb` always did.
    let rgb = img.into_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok((rgb.into_raw(), w, h))
}

/// The single-image resize target for a `w x h` source: llama.cpp's
/// `calc_size_preserved_ratio` from the NATIVE dims, then the optional caller cap on the longest
/// side. `align_size = patch_size · scale_factor` keeps both grid dims even so the 2x pixel
/// shuffle works out.
fn single_target(
    w: usize,
    h: usize,
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> (usize, usize) {
    let align = cfg.patch_size * cfg.scale_factor;
    let (mut target_w, mut target_h) =
        calc_size_preserved_ratio(w, h, align, cfg.image_min_pixels, cfg.image_max_pixels);

    // Optional caller cap on the longest side of the encoded target.
    // Applied to the TARGET (not a pre-resize of the input) so the
    // single resample goes straight from native dims to the final
    // target: no cascaded downscale-then-upscale. Shrinks only
    // (`long > cap`), preserves aspect, re-aligns by flooring, and
    // clamps to at least one aligned block so the patch grid stays
    // valid. Deliberately takes precedence over `image_min_pixels`
    // (the caller is trading detail for cost).
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
    (target_w, target_h)
}

/// Resize an RGB8 `w x h` image to `target_w x target_h` (reusing the buffer when it already is that
/// size) and normalize it into a [`PreprocessedImage`].
fn finish_image(
    rgb: &[u8],
    w: usize,
    h: usize,
    target_w: usize,
    target_h: usize,
    cfg: &VisionEncoderConfig,
) -> Result<PreprocessedImage, CeraError> {
    let resized;
    let rgb_bytes: &[u8] = if w == target_w && h == target_h {
        &rgb[..3 * w * h]
    } else {
        resized = resize_pillow_bilinear_rgb8(rgb, w, h, target_w, target_h)?;
        &resized
    };
    let pixels = normalize_rgb8_to_nchw_f32(
        rgb_bytes,
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

fn preprocess_single_rgb8(
    rgb: &[u8],
    w: usize,
    h: usize,
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedImage, CeraError> {
    let (target_w, target_h) = single_target(w, h, cfg, max_long_size);
    finish_image(rgb, w, h, target_w, target_h, cfg)
}

fn preprocess_layout_rgb8(
    rgb: &[u8],
    w: usize,
    h: usize,
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedLayout, CeraError> {
    if w == 0 || h == 0 {
        return Err(CeraError::EmptyInput);
    }
    let capped = max_long_size.is_some_and(|c| c > 0);
    if capped || !lfm2_should_tile(w, h, cfg) {
        return Ok(PreprocessedLayout::Single(preprocess_single_rgb8(
            rgb,
            w,
            h,
            cfg,
            max_long_size,
        )?));
    }
    let (cols, rows) = lfm2_tile_grid(w, h);
    // The whole image goes to `tile * grid` once, then is cropped into tiles, as the reference does
    // (`slice_image`): resizing each tile from its own source region would differ at the seams.
    let refined =
        resize_pillow_bilinear_rgb8(rgb, w, h, LFM2_TILE_SIZE * cols, LFM2_TILE_SIZE * rows)?;
    let stride = LFM2_TILE_SIZE * cols * 3;
    let mut tiles = Vec::with_capacity(cols * rows);
    let mut tile_rgb = vec![0u8; LFM2_TILE_SIZE * LFM2_TILE_SIZE * 3];
    for row in 0..rows {
        for col in 0..cols {
            for y in 0..LFM2_TILE_SIZE {
                let src = (row * LFM2_TILE_SIZE + y) * stride + col * LFM2_TILE_SIZE * 3;
                tile_rgb[y * LFM2_TILE_SIZE * 3..(y + 1) * LFM2_TILE_SIZE * 3]
                    .copy_from_slice(&refined[src..src + LFM2_TILE_SIZE * 3]);
            }
            tiles.push(finish_image(
                &tile_rgb,
                LFM2_TILE_SIZE,
                LFM2_TILE_SIZE,
                LFM2_TILE_SIZE,
                LFM2_TILE_SIZE,
                cfg,
            )?);
        }
    }
    // The thumbnail is the single-image resize of the original, after the tiles.
    let thumbnail = preprocess_single_rgb8(rgb, w, h, cfg, None)?;
    Ok(PreprocessedLayout::Tiled(TiledImage {
        cols,
        rows,
        tiles,
        thumbnail,
    }))
}

/// Preprocess an uncompressed raw pixel buffer (e.g. from an Android Bitmap
/// or camera frame) into a [`PreprocessedImage`] ready for vision encoding.
///
/// Bypasses all image decompression overhead and applies the same resampling
/// and SIMD/parallel NCHW normalization as the decoded-image path. Always one
/// image; see [`preprocess_raw_layout`] for the tiled layout.
pub fn preprocess_raw_pixels(
    pixels: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedImage, CeraError> {
    let rgb = raw_to_rgb8(pixels, width, height, format)?;
    preprocess_single_rgb8(&rgb, width, height, cfg, max_long_size)
}

/// [`preprocess_raw_pixels`] with the tiled layout for large images; see
/// [`preprocess_image_layout`].
pub fn preprocess_raw_layout(
    pixels: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
    cfg: &VisionEncoderConfig,
    max_long_size: Option<u32>,
) -> Result<PreprocessedLayout, CeraError> {
    let rgb = raw_to_rgb8(pixels, width, height, format)?;
    preprocess_layout_rgb8(&rgb, width, height, cfg, max_long_size)
}

/// Validate a raw pixel buffer and return it as interleaved RGB8 (borrowed when it already is).
fn raw_to_rgb8(
    pixels: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
) -> Result<std::borrow::Cow<'_, [u8]>, CeraError> {
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
    if format == PixelFormat::Rgb8 {
        return Ok(std::borrow::Cow::Borrowed(&pixels[..3 * width * height]));
    }
    let (r_off, g_off, b_off) = format.channel_offsets();
    let mut rgb = vec![0u8; width * height * 3];
    for (i, px) in pixels[..min_src_len].chunks_exact(bpp).enumerate() {
        rgb[i * 3] = px[r_off];
        rgb[i * 3 + 1] = px[g_off];
        rgb[i * 3 + 2] = px[b_off];
    }
    Ok(std::borrow::Cow::Owned(rgb))
}

/// LFM2-VL tiling constants, from the model's `processor_config.json` (the same values llama.cpp's
/// `mtmd_image_preprocessor_lfm2` hard-codes).
pub const LFM2_TILE_SIZE: usize = 512;
const LFM2_MIN_TILES: usize = 2;
const LFM2_MAX_TILES: usize = 10;
const LFM2_MAX_PIXELS_TOLERANCE: f64 = 2.0;

/// Whether the reference processor tiles a `w x h` image: its aligned area exceeds the
/// single-image pixel budget times the 2.0 tolerance. Alignment rounds half to even (C's
/// `nearbyint`), as llama.cpp's `should_tile` does.
pub fn lfm2_should_tile(w: usize, h: usize, cfg: &VisionEncoderConfig) -> bool {
    let align = (cfg.patch_size * cfg.scale_factor) as f64;
    let round_by = |x: usize| ((x as f64 / align).round_ties_even() as usize) * align as usize;
    let h_bar = cfg.patch_size.max(round_by(h));
    let w_bar = cfg.patch_size.max(round_by(w));
    (h_bar as f64) * (w_bar as f64) > cfg.image_max_pixels as f64 * LFM2_MAX_PIXELS_TOLERANCE
}

/// The tile grid `(cols, rows)` for a `w x h` image: of the grids of 2 to 10 tiles, the one whose
/// aspect ratio is closest to the image's, preferring the larger grid on a tie when the image fills
/// more than half of it. A port of llama.cpp's `find_closest_aspect_ratio` (single-precision, as
/// there).
pub fn lfm2_tile_grid(w: usize, h: usize) -> (usize, usize) {
    let aspect = w as f32 / h as f32;
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for n in LFM2_MIN_TILES..=LFM2_MAX_TILES {
        for cols in 1..=n {
            for rows in 1..=n {
                let tiles = cols * rows;
                if (LFM2_MIN_TILES..=LFM2_MAX_TILES).contains(&tiles)
                    && !ratios.contains(&(cols, rows))
                {
                    ratios.push((cols, rows));
                }
            }
        }
    }
    ratios.sort_by_key(|&(c, r)| c * r);
    let area = (w * h) as f32;
    let mut best = (1usize, 1usize);
    let mut best_diff = f32::MAX;
    for &(c, r) in &ratios {
        let diff = (aspect - c as f32 / r as f32).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (c, r);
        } else if diff == best_diff {
            let target_area = (LFM2_TILE_SIZE * LFM2_TILE_SIZE * c * r) as f32;
            if area > 0.5 * target_area {
                best = (c, r);
            }
        }
    }
    best
}

/// What a source image becomes: one image, or (for a large image, see [`lfm2_should_tile`]) a
/// grid of tiles followed by a thumbnail.
#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessedLayout {
    Single(PreprocessedImage),
    Tiled(TiledImage),
}

/// The tiled layout: `tiles` in row-major order (`cols x rows`, 512 px each) and the thumbnail,
/// the aspect-preserving single-image resize of the whole picture. The prompt is
/// `<|img_row_R_col_C|>` + each tile's tokens in that order, then `<|img_thumbnail|>` + the
/// thumbnail's tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct TiledImage {
    pub cols: usize,
    pub rows: usize,
    pub tiles: Vec<PreprocessedImage>,
    pub thumbnail: PreprocessedImage,
}

/// Largest working buffer (bytes) one Pillow resize may allocate: 1 GiB, far above any image the model
/// takes (a 10-tile grid is 5,120 x 2,560) and well inside a 32-bit `usize`.
const PILLOW_MAX_BUFFER_BYTES: usize = 1 << 30;

/// Pillow-compatible bilinear resize of an interleaved RGB8 image.
///
/// A port of llama.cpp's `resize_pillow` (itself Pillow's `Resample.c`, which is what the
/// reference processor's `resize` calls): two separable passes, horizontal then vertical, with the
/// triangle filter **widened by the scale factor when shrinking**, so a downscale averages the
/// source pixels it covers instead of sampling two of them, and 22-bit fixed-point weights
/// normalized per output pixel. Upscaling is the usual bilinear. Differs from
/// [`resize_bilinear_rgb`], which always takes the two nearest source pixels.
pub fn resize_pillow_bilinear_rgb8(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    target_w: usize,
    target_h: usize,
) -> Result<Vec<u8>, CeraError> {
    if src_w == 0 || src_h == 0 || target_w == 0 || target_h == 0 {
        return Err(CeraError::EmptyInput);
    }
    if target_w > 65_536 || target_h > 65_536 {
        return Err(CeraError::Backend(format!(
            "resize target {target_w}x{target_h} is out of range (max 65536)"
        )));
    }
    let need = src_w
        .checked_mul(src_h)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| CeraError::Backend("image dimensions overflow usize".into()))?;
    if src.len() < need {
        return Err(CeraError::Backend(format!(
            "resize_pillow_bilinear_rgb8: source is {} bytes, need {need} ({src_w}x{src_h} RGB)",
            src.len()
        )));
    }
    // The two working buffers: the horizontal pass keeps every source row at the target width, the
    // vertical pass produces the output. The 65,536 cap on the target and the source-length check do
    // not bound their product (a 1 x 400,000 source resized to 65,536 wide would ask for 78 GB, and
    // wrap `usize` on a 32-bit target), so refuse what is out of range rather than allocate it.
    for (px_w, px_h) in [(target_w, src_h), (target_w, target_h)] {
        px_w.checked_mul(px_h)
            .and_then(|px| px.checked_mul(3))
            .filter(|&b| b <= PILLOW_MAX_BUFFER_BYTES)
            .ok_or_else(|| {
                CeraError::Backend(format!(
                    "resize {src_w}x{src_h} -> {target_w}x{target_h} needs a working buffer over \
                     {PILLOW_MAX_BUFFER_BYTES} bytes"
                ))
            })?;
    }
    // `None` while the image is still the source: an axis already at its target size is skipped.
    let mut cur: Option<Vec<u8>> = None;
    let mut w = src_w;
    if target_w != src_w {
        let k = pillow_bilinear_kernel(src_w, target_w);
        cur = Some(pillow_horizontal(&src[..need], src_w, src_h, target_w, &k));
        w = target_w;
    }
    if target_h != src_h {
        let k = pillow_bilinear_kernel(src_h, target_h);
        let input = cur.as_deref().unwrap_or(&src[..need]);
        cur = Some(pillow_vertical(input, w, target_h, &k));
    }
    Ok(cur.unwrap_or_else(|| src[..need].to_vec()))
}

const PILLOW_PRECISION_BITS: u32 = 22;

struct PillowKernel {
    ksize: usize,
    /// `(first input index, input count)` per output index.
    bounds: Vec<(usize, usize)>,
    /// `ksize` fixed-point weights per output index.
    weights: Vec<i32>,
}

/// Filter taps for resampling `in_size` to `out_size` samples (one dimension), as `resize_pillow`
/// precomputes them.
fn pillow_bilinear_kernel(in_size: usize, out_size: usize) -> PillowKernel {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale; // the bilinear filter's support is 1.0
    let ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;
    let fxp = (1u64 << PILLOW_PRECISION_BITS) as f64;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = vec![0i32; out_size * ksize];
    let mut pre = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = (((center + support + 0.5) as i64).min(in_size as i64) - xmin).max(0);
        let mut ww = 0.0;
        for x in 0..xmax {
            let d = ((x + xmin) as f64 - center + 0.5) * ss;
            let w = (1.0 - d.abs()).max(0.0);
            pre[x as usize] = w;
            ww += w;
        }
        for x in 0..ksize {
            let w = if (x as i64) < xmax && ww != 0.0 {
                pre[x] / ww
            } else if (x as i64) < xmax {
                pre[x]
            } else {
                0.0
            };
            // Pillow adds +/- 0.5 and truncates toward zero (a plain round would round twice).
            weights[xx * ksize + x] = (w * fxp + if w < 0.0 { -0.5 } else { 0.5 }) as i32;
        }
        bounds.push((xmin as usize, xmax as usize));
    }
    PillowKernel {
        ksize,
        bounds,
        weights,
    }
}

#[inline]
fn pillow_clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn pillow_horizontal(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    k: &PillowKernel,
) -> Vec<u8> {
    let mut out = vec![0u8; out_w * in_h * 3];
    let row = |yy: usize, dst: &mut [u8]| {
        let src_row = &src[yy * in_w * 3..(yy + 1) * in_w * 3];
        for xx in 0..out_w {
            let (xmin, xcnt) = k.bounds[xx];
            let taps = &k.weights[xx * k.ksize..xx * k.ksize + xcnt];
            let half = 1i32 << (PILLOW_PRECISION_BITS - 1);
            let (mut s0, mut s1, mut s2) = (half, half, half);
            for (x, &wt) in taps.iter().enumerate() {
                let p = &src_row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                s0 += p[0] as i32 * wt;
                s1 += p[1] as i32 * wt;
                s2 += p[2] as i32 * wt;
            }
            dst[xx * 3] = pillow_clip8(s0 >> PILLOW_PRECISION_BITS);
            dst[xx * 3 + 1] = pillow_clip8(s1 >> PILLOW_PRECISION_BITS);
            dst[xx * 3 + 2] = pillow_clip8(s2 >> PILLOW_PRECISION_BITS);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(out_w * 3)
            .enumerate()
            .for_each(|(yy, dst)| row(yy, dst));
    }
    #[cfg(not(feature = "parallel"))]
    for (yy, dst) in out.chunks_mut(out_w * 3).enumerate() {
        row(yy, dst);
    }
    out
}

fn pillow_vertical(src: &[u8], in_w: usize, out_h: usize, k: &PillowKernel) -> Vec<u8> {
    let row_elems = in_w * 3;
    let mut out = vec![0u8; row_elems * out_h];
    let row = |yy: usize, dst: &mut [u8]| {
        let (ymin, ycnt) = k.bounds[yy];
        let taps = &k.weights[yy * k.ksize..yy * k.ksize + ycnt];
        let half = 1i32 << (PILLOW_PRECISION_BITS - 1);
        let mut acc = vec![half; row_elems];
        for (y, &wt) in taps.iter().enumerate() {
            let s = &src[(ymin + y) * row_elems..(ymin + y + 1) * row_elems];
            for (a, &v) in acc.iter_mut().zip(s) {
                *a += v as i32 * wt;
            }
        }
        for (d, a) in dst.iter_mut().zip(&acc) {
            *d = pillow_clip8(a >> PILLOW_PRECISION_BITS);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(row_elems)
            .enumerate()
            .for_each(|(yy, dst)| row(yy, dst));
    }
    #[cfg(not(feature = "parallel"))]
    for (yy, dst) in out.chunks_mut(row_elems).enumerate() {
        row(yy, dst);
    }
    out
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

    /// The LFM2.5-VL-450M vision config the tiling rules are specified against: patch 16, a 2x2
    /// merge (so 32 px alignment) and a 65,536 to 262,144 px single-image budget (64 to 256 tokens).
    fn lfm2_cfg() -> VisionEncoderConfig {
        VisionEncoderConfig {
            patch_size: 16,
            image_size: 512,
            n_trained_patches: 256,
            image_min_pixels: 65_536,
            image_max_pixels: 262_144,
            ..synth_cfg()
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

    // ── Pillow-compatible resize, tiling ─────────────────────────────

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| ((i * 37 + (i * i) % 251 + 13) % 256) as u8)
            .collect()
    }

    /// A resize whose working buffers would be enormous is refused, not allocated: the target cap
    /// and the source-length check do not bound `target_w * src_h`.
    #[test]
    fn pillow_resize_refuses_oversized_working_buffers() {
        // 1 x 400,000 to 65,536 wide: the horizontal pass alone is 78 GB.
        let src = vec![7u8; 400_000 * 3];
        assert!(matches!(
            resize_pillow_bilinear_rgb8(&src, 1, 400_000, 65_536, 4),
            Err(CeraError::Backend(_))
        ));
        // The model's largest real case is fine: a 10-tile grid, 5,120 x 2,560.
        let small = vec![9u8; 64 * 32 * 3];
        assert!(resize_pillow_bilinear_rgb8(&small, 64, 32, 5_120, 2_560).is_ok());
    }

    /// Empty and extreme-aspect inputs never reach the grid search with a degenerate ratio, and the
    /// grid it returns is always a valid one (2 to 10 tiles).
    #[test]
    fn tile_layout_handles_empty_and_extreme_shapes() {
        let cfg = lfm2_cfg();
        assert!(matches!(
            preprocess_layout_rgb8(&[], 0, 100_000, &cfg, None),
            Err(CeraError::EmptyInput)
        ));
        assert!(matches!(
            preprocess_layout_rgb8(&[], 512, 0, &cfg, None),
            Err(CeraError::EmptyInput)
        ));
        for (w, h) in [(1, 20_000), (20_000, 1), (3, 5_000), (4_000, 7)] {
            let (c, r) = lfm2_tile_grid(w, h);
            assert!((2..=10).contains(&(c * r)), "{w}x{h} -> {c}x{r}");
        }
    }

    /// The resize must reproduce Pillow's (and llama.cpp's port of it) arithmetic exactly. The
    /// expected values come from an independent implementation of the same algorithm (exact integer
    /// math in Python), covering a shrink, a grow, a mix, one axis only, and the 1024x771 to 576x416
    /// thumbnail resize.
    #[test]
    fn pillow_resize_matches_the_reference_implementation() {
        // (src w, src h, dst w, dst h, FNV-1a digest of the output, its first 12 bytes).
        type Case = (usize, usize, usize, usize, u64, [u8; 12]);
        let cases: &[Case] = &[
            (
                7,
                5,
                3,
                9,
                0x3686_d435_da0d_2a2c,
                [65, 108, 153, 104, 160, 113, 116, 95, 166, 113, 86, 145],
            ),
            (
                100,
                60,
                30,
                20,
                0x40d7_ccde_abc8_94e8,
                [120, 153, 149, 163, 148, 119, 155, 103, 125, 156, 115, 134],
            ),
            (
                20,
                30,
                45,
                70,
                0x13d8_0aa1_20b2_d684,
                [13, 51, 91, 33, 72, 113, 86, 128, 172, 126, 171, 217],
            ),
            (
                64,
                64,
                64,
                16,
                0x9e01_5940_7ef5_ef39,
                [80, 93, 87, 157, 130, 111, 133, 139, 119, 73, 195, 135],
            ),
            (
                33,
                17,
                33,
                40,
                0xddea_fb30_024e_b1ac,
                [13, 51, 91, 133, 177, 223, 15, 65, 117, 171, 227, 29],
            ),
            (
                1024,
                771,
                576,
                416,
                0x0c53_8a27_4337_33da,
                [81, 117, 148, 81, 150, 130, 105, 105, 157, 165, 93, 119],
            ),
        ];
        for &(sw, sh, tw, th, hash, first) in cases {
            let src = pattern(sw * sh * 3);
            let out = resize_pillow_bilinear_rgb8(&src, sw, sh, tw, th).unwrap();
            assert_eq!(out.len(), tw * th * 3, "{sw}x{sh} -> {tw}x{th} length");
            assert_eq!(&out[..12], &first, "{sw}x{sh} -> {tw}x{th} first pixels");
            assert_eq!(fnv1a(&out), hash, "{sw}x{sh} -> {tw}x{th} digest");
        }
    }

    /// Shrinking averages the source pixels it covers (the filter widens by the scale factor), so a
    /// single bright pixel still shows up in the smaller image. The two-tap `resize_bilinear_rgb`
    /// only ever reads the two source pixels nearest each output sample, and for 16 -> 4 it never
    /// reads source pixel 3, so it loses the pixel entirely.
    #[test]
    fn pillow_resize_keeps_every_source_pixel_when_shrinking() {
        let mut src = vec![0u8; 16 * 3];
        src[3 * 3..3 * 3 + 3].fill(240);
        let pillow = resize_pillow_bilinear_rgb8(&src, 16, 1, 4, 1).unwrap();
        assert!(
            pillow.iter().any(|&v| v > 20),
            "the bright pixel vanished: {pillow:?}"
        );
        // Weight is conserved: the output mean tracks the input mean (240 / 16 = 15).
        let mean = pillow.iter().map(|&v| v as f32).sum::<f32>() / pillow.len() as f32;
        assert!((mean - 15.0).abs() < 2.0, "mean {mean}");
        let two_tap = resize_bilinear_rgb(&src, 16, 1, PixelFormat::Rgb8, 4, 1).unwrap();
        assert!(
            two_tap.iter().all(|&v| v == 0),
            "two-tap read pixel 3: {two_tap:?}"
        );
    }

    #[test]
    fn pillow_resize_identity_and_errors() {
        let src = pattern(5 * 4 * 3);
        assert_eq!(resize_pillow_bilinear_rgb8(&src, 5, 4, 5, 4).unwrap(), src);
        assert!(resize_pillow_bilinear_rgb8(&src, 5, 4, 0, 4).is_err());
        assert!(resize_pillow_bilinear_rgb8(&src[..10], 5, 4, 3, 3).is_err());
    }

    /// Tiling starts once the 32-aligned area exceeds twice the single-image budget (524,288 px):
    /// 724x724 aligns to 736x736 and tiles, 700x700 aligns to 704x704 and does not. Alignment rounds
    /// half to even, as the reference does.
    #[test]
    fn lfm2_should_tile_follows_the_aligned_area() {
        let cfg = lfm2_cfg();
        assert!(!lfm2_should_tile(512, 385, &cfg));
        assert!(!lfm2_should_tile(700, 700, &cfg));
        assert!(lfm2_should_tile(724, 724, &cfg));
        assert!(lfm2_should_tile(1024, 771, &cfg));
        // 16 is the round-half-even case: 16/32 = 0.5 rounds to 0, floored at one patch (16).
        assert!(!lfm2_should_tile(16, 16, &cfg));
    }

    #[test]
    fn lfm2_tile_grid_picks_the_closest_aspect_ratio() {
        assert_eq!(lfm2_tile_grid(1024, 771), (3, 2));
        assert_eq!(lfm2_tile_grid(771, 1024), (2, 3));
        // An exact 2:1 tie between 2x1 and 4x2 goes to the larger grid when the image fills over
        // half of it.
        assert_eq!(lfm2_tile_grid(2000, 1000), (4, 2));
        assert_eq!(lfm2_tile_grid(2000, 2000), (3, 3));
        // A square ties between 2x2 and 3x3 (and more): the larger grid wins only once the image fills over
        // half of it, so a 1000x1000 photo is 4 tiles and a 1200x1200 one is 9.
        assert_eq!(lfm2_tile_grid(1000, 1000), (2, 2));
        assert_eq!(lfm2_tile_grid(1200, 1200), (3, 3));
        // Never fewer than 2 tiles or more than 10.
        for (w, h) in [(1500, 400), (400, 1500), (3000, 3000), (900, 1400)] {
            let (c, r) = lfm2_tile_grid(w, h);
            assert!((2..=10).contains(&(c * r)), "{w}x{h} -> {c}x{r}");
        }
    }

    fn png_of(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y) % 256) as u8,
            ])
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    /// 1024x771 is the worked example from the reference: a 3x2 grid of 512 px tiles (256 tokens
    /// each) plus a 576x416 thumbnail (234 tokens), and a caller cap turns tiling off.
    #[test]
    fn large_image_becomes_tiles_plus_thumbnail() {
        let cfg = lfm2_cfg();
        let bytes = png_of(1024, 771);
        match preprocess_image_layout(&bytes, &cfg, None).unwrap() {
            PreprocessedLayout::Tiled(t) => {
                assert_eq!((t.cols, t.rows), (3, 2));
                assert_eq!(t.tiles.len(), 6);
                for tile in &t.tiles {
                    assert_eq!((tile.target_w, tile.target_h), (512, 512));
                    assert_eq!(tile.pixels.len(), 3 * 512 * 512);
                }
                assert_eq!((t.thumbnail.target_w, t.thumbnail.target_h), (576, 416));
            }
            other => panic!("expected a tiled layout, got {other:?}"),
        }
        assert!(matches!(
            preprocess_image_layout(&bytes, &cfg, Some(512)).unwrap(),
            PreprocessedLayout::Single(_)
        ));
        // A small image stays single and equals the single-image path.
        let small = png_of(512, 385);
        let PreprocessedLayout::Single(one) = preprocess_image_layout(&small, &cfg, None).unwrap()
        else {
            panic!("a 512x385 image must not tile");
        };
        assert_eq!(one, preprocess_image_with_opts(&small, &cfg, None).unwrap());
    }

    /// Tiles are cropped from one resize of the whole image (as the reference does), not resized
    /// from their own source regions: adjacent tiles must meet without a seam.
    #[test]
    fn tiles_are_crops_of_one_refined_image() {
        let cfg = lfm2_cfg();
        let (w, h) = (1024usize, 771usize);
        let rgb = pattern(w * h * 3);
        let PreprocessedLayout::Tiled(t) = preprocess_layout_rgb8(&rgb, w, h, &cfg, None).unwrap()
        else {
            panic!("expected tiles");
        };
        // The thumbnail is the ordinary single-image preprocess of the whole picture.
        assert_eq!(
            t.thumbnail,
            preprocess_single_rgb8(&rgb, w, h, &cfg, None).unwrap()
        );
        let refined = resize_pillow_bilinear_rgb8(&rgb, w, h, 3 * 512, 2 * 512).unwrap();
        // Tile (row 1, col 2) is the refined image's crop at (x=1024, y=512), channel-major (NCHW).
        // Every pixel of the tile is compared, so a wrong row stride or a repeated row cannot pass.
        let (tile_row, tile_col) = (1usize, 2usize);
        let tile = &t.tiles[tile_row * 3 + tile_col];
        let side = LFM2_TILE_SIZE;
        for y in 0..side {
            for x in 0..side {
                for c in 0..3 {
                    let src = (((tile_row * side + y) * 3 * side) + tile_col * side + x) * 3 + c;
                    let want = (refined[src] as f32 / 255.0 - cfg.image_mean[c]) / cfg.image_std[c];
                    let got = tile.pixels[c * side * side + y * side + x];
                    assert!((got - want).abs() < 1e-5, "y{y} x{x} c{c}: {got} vs {want}");
                }
            }
        }
    }
}

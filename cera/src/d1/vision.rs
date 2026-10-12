//! Images to prefix embeddings.
//!
//! An image is read the way LFM2-VL reads one: a large image is cut into a grid of up to ten
//! 512 px tiles plus a thumbnail, a small one is read whole. Each crop goes through a SigLIP2
//! tower that accepts any patch grid ("NaFlex": the position table is resized to the crop's
//! grid), then a 2x2 pixel unshuffle and a two-layer GELU projector bring every 2x2 block of
//! patches to one embedding of the trunk's width. A crop of `h x w` patches gives
//! `(h / 2) * (w / 2)` embeddings, and several images are concatenated in the order given.
//!
//! The resize is part of the model: it is PyTorch's antialiased bilinear on `uint8`, horizontal
//! pass first, each pass rounded back to `uint8`, with 16-bit fixed-point weights. Any other
//! resampler moves pixels by one level here and there, which moves the answers.

use anyhow::{Context, Result, bail, ensure};

use crate::backend::cpu;
use crate::gguf::GgufFile;
use crate::model::pii::matmul_nt_f32;
use crate::par::*;

/// Side of a tile, in pixels.
const TILE: usize = 512;
/// Pixels per patch side.
pub(crate) const PATCH: usize = 16;
/// Resolution of the position table: 16 x 16 patches.
pub(crate) const POSITION_SIDE: usize = 16;
/// Side factor of the pixel unshuffle.
const UNSHUFFLE: usize = 2;
/// LFM2-VL's pixel budget for one crop, in the units its smart resize works in.
const MAX_PIXELS: usize = 256 * 1024;
const MIN_PIXELS: usize = 64 * 1024;
/// Tiles of a grid: at least 2 and at most 10 (a single tile is the thumbnail itself).
const MIN_TILES: usize = 2;
const MAX_TILES: usize = 10;

/// An RGB image, rows top to bottom, three bytes per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgb {
    /// A solid image, mostly for tests.
    pub fn filled(width: usize, height: usize, rgb: [u8; 3]) -> Self {
        Self {
            width,
            height,
            data: rgb
                .iter()
                .copied()
                .cycle()
                .take(width * height * 3)
                .collect(),
        }
    }
}

/// The decode dimension bound both image readers enforce (same bound as the VL
/// preprocessor).
#[cfg(feature = "vl-preprocess")]
fn decode_limits() -> image::Limits {
    use crate::model::vision_preprocessor::MAX_DECODE_DIM;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_DIM);
    limits.max_image_height = Some(MAX_DECODE_DIM);
    limits
}

/// An image reader over the bytes with its format detected.
#[cfg(feature = "vl-preprocess")]
fn image_reader(bytes: &[u8]) -> Result<image::ImageReader<std::io::Cursor<&[u8]>>> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("detecting the image format")
}

/// The declared dimensions of a PNG or JPEG, read from the header without
/// allocating the pixels, so the budget refuses an over-budget image before the
/// decode transient lands.
///
/// # Errors
///
/// Fails on bytes that are not a supported image, an image past the dimension bound,
/// or when the `vl-preprocess` feature is off.
#[cfg(feature = "vl-preprocess")]
pub(crate) fn image_dims(bytes: &[u8]) -> Result<(usize, usize)> {
    use image::ImageDecoder as _;
    let decoder = image_reader(bytes)?.into_decoder()?;
    let (width, height) = decoder.dimensions();
    decode_limits().check_dimensions(width, height)?;
    Ok((width as usize, height as usize))
}

/// The declared dimensions of a PNG or JPEG.
///
/// # Errors
///
/// Always: this build has no image decoder (the `vl-preprocess` feature is off).
#[cfg(not(feature = "vl-preprocess"))]
pub(crate) fn image_dims(_bytes: &[u8]) -> Result<(usize, usize)> {
    bail!("this build cannot decode images (the `vl-preprocess` feature is off)")
}

/// Decode a PNG or JPEG into RGB.
///
/// Declared dimensions are bounded before the pixel buffer is allocated, so a hostile
/// file cannot ask for a gigapixel allocation (same bound as the VL preprocessor).
///
/// # Errors
///
/// Fails on bytes that are not a supported image, an image past the dimension bound,
/// or when the `vl-preprocess` feature is off.
#[cfg(feature = "vl-preprocess")]
pub fn decode_image(bytes: &[u8]) -> Result<Rgb> {
    let mut reader = image_reader(bytes)?;
    reader.limits(decode_limits());
    let img = reader.decode().context("decoding the image")?.to_rgb8();
    Ok(Rgb {
        width: img.width() as usize,
        height: img.height() as usize,
        data: img.into_raw(),
    })
}

/// Decode a PNG or JPEG into RGB.
///
/// # Errors
///
/// Always: this build has no image decoder (the `vl-preprocess` feature is off).
#[cfg(not(feature = "vl-preprocess"))]
pub fn decode_image(_bytes: &[u8]) -> Result<Rgb> {
    bail!("this build cannot decode images (the `vl-preprocess` feature is off)")
}

/// The bytes of a `data:<type>;base64,<payload>` URL, or of bare base64.
///
/// # Errors
///
/// Fails on a data URL that is not base64, or a payload with a character outside the alphabet.
pub fn decode_data_url(url: &str) -> Result<Vec<u8>> {
    let payload = match url.strip_prefix("data:") {
        Some(rest) => {
            let (meta, payload) = rest.split_once(',').context("a data URL needs a comma")?;
            ensure!(
                meta.ends_with(";base64"),
                "only base64 data URLs are accepted"
            );
            payload
        }
        None => url,
    };
    let mut out = Vec::with_capacity(payload.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in payload.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => bail!("invalid base64 character {:?}", c as char),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// How an image is cut: a grid of tiles (when it is large) and a thumbnail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Tiles across and down; `(1, 1)` when the image is read whole.
    pub grid: (usize, usize),
    /// The thumbnail's height and width, multiples of 32.
    pub thumbnail: (usize, usize),
    /// Whether the image is cut into tiles at all.
    pub tiled: bool,
}

/// LFM2-VL's smart resize, tile grid and thumbnail for a `width x height` image.
///
/// # Errors
///
/// Fails on an empty image.
pub fn layout(width: usize, height: usize) -> Result<Layout> {
    ensure!(width >= 1 && height >= 1, "empty image");
    const FACTOR: usize = 32;
    let (wf, hf) = (width as f64, height as f64);
    let nearest = |v: f64| (v / FACTOR as f64).round_ties_even() as usize * FACTOR;
    let mut h = FACTOR.max(nearest(hf));
    let mut w = FACTOR.max(nearest(wf));
    if h * w > MAX_PIXELS {
        let beta = (hf * wf / MAX_PIXELS as f64).sqrt();
        h = FACTOR.max((hf / beta / FACTOR as f64).floor() as usize * FACTOR);
        w = FACTOR.max((wf / beta / FACTOR as f64).floor() as usize * FACTOR);
    } else if h * w < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (hf * wf)).sqrt();
        h = (hf * beta / FACTOR as f64).ceil() as usize * FACTOR;
        w = (wf * beta / FACTOR as f64).ceil() as usize * FACTOR;
    }
    let large = (16usize.max(nearest(hf))) * (16usize.max(nearest(wf))) > MAX_PIXELS * 2;
    let mut grid = (1, 1);
    if large {
        let mut ratios: Vec<(usize, usize)> = Vec::new();
        for n in MIN_TILES..=MAX_TILES {
            for x in 1..=n {
                for y in 1..=n {
                    if (MIN_TILES..=MAX_TILES).contains(&(x * y)) && !ratios.contains(&(x, y)) {
                        ratios.push((x, y));
                    }
                }
            }
        }
        ratios.sort_by_key(|&(x, y)| (x * y, x, y));
        let mut best = f64::INFINITY;
        let aspect = wf / hf;
        for (rx, ry) in ratios {
            let diff = (aspect - rx as f64 / ry as f64).abs();
            let area_wins = wf * hf > 0.5 * (TILE * TILE * rx * ry) as f64;
            if diff < best || (diff == best && area_wins) {
                grid = (rx, ry);
                best = diff;
            }
        }
    }
    Ok(Layout {
        grid,
        thumbnail: (h, w),
        tiled: large,
    })
}

/// Antialiased triangle weights of one axis, as PyTorch computes them for bilinear
/// `antialias=True`: the output sample `i` reads the input samples `start..start + weights.len()`.
fn axis_weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let support = scale.max(1.0);
    (0..out_size)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as isize).max(0) as usize;
            let hi = (((center + support + 0.5) as usize).min(in_size)).max(lo + 1);
            let mut w: Vec<f64> = (lo..hi)
                .map(|x| (1.0 - ((x as f64 + 0.5 - center) / support).abs()).max(0.0))
                .collect();
            let sum: f64 = w.iter().sum();
            for v in &mut w {
                *v /= sum;
            }
            (lo, w)
        })
        .collect()
}

/// The same weights as signed 16-bit-range integers with the shared binary point PyTorch
/// picks for the axis: the largest precision at which the biggest weight still fits in 15 bits.
fn fixed_point_weights(in_size: usize, out_size: usize) -> (u32, Vec<(usize, Vec<i64>)>) {
    let weights = axis_weights(in_size, out_size);
    let max_w = weights
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(0.0f64, f64::max);
    let mut precision = 0u32;
    while precision < 22 && ((0.5 + max_w * f64::from(1u32 << (precision + 1))) as i64) < (1 << 15)
    {
        precision += 1;
    }
    let fixed = weights
        .into_iter()
        .map(|(lo, w)| {
            let w = w
                .into_iter()
                .map(|v| (v * f64::from(1u32 << precision) + 0.5).floor() as i64)
                .collect();
            (lo, w)
        })
        .collect();
    (precision, fixed)
}

fn round_shift(acc: i64, precision: u32) -> u8 {
    let round = if precision > 0 {
        1i64 << (precision - 1).min(62)
    } else {
        0
    };
    ((acc + round) >> precision).clamp(0, 255) as u8
}

/// Resize with PyTorch's antialiased bilinear on `uint8`: width first, each pass rounded to
/// `uint8`, 16-bit fixed-point weights.
pub fn resize(src: &Rgb, height: usize, width: usize) -> Rgb {
    let (sw, sh) = (src.width, src.height);
    // horizontal pass: [sh, width]
    let (hp, hw) = fixed_point_weights(sw, width);
    let mut mid = vec![0u8; sh * width * 3];
    mid.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let src_row = &src.data[y * sw * 3..(y + 1) * sw * 3];
            for (x, (lo, w)) in hw.iter().enumerate() {
                let mut acc = [0i64; 3];
                for (k, wk) in w.iter().enumerate() {
                    let px = &src_row[(lo + k) * 3..(lo + k) * 3 + 3];
                    for c in 0..3 {
                        acc[c] += wk * i64::from(px[c]);
                    }
                }
                for c in 0..3 {
                    row[x * 3 + c] = round_shift(acc[c], hp);
                }
            }
        });
    // vertical pass: [height, width]
    let (vp, vw) = fixed_point_weights(sh, height);
    let mut out = vec![0u8; height * width * 3];
    out.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let (lo, w) = &vw[y];
            for x in 0..width * 3 {
                let mut acc = 0i64;
                for (k, wk) in w.iter().enumerate() {
                    acc += wk * i64::from(mid[(lo + k) * width * 3 + x]);
                }
                row[x] = round_shift(acc, vp);
            }
        });
    Rgb {
        width,
        height,
        data: out,
    }
}

/// The crops of an image, in the order their embeddings are laid out: the tiles row by row,
/// then the thumbnail.
pub fn crops(image: &Rgb) -> Result<Vec<Rgb>> {
    let plan = layout(image.width, image.height)?;
    let mut out = Vec::new();
    if plan.tiled {
        let (gw, gh) = plan.grid;
        let big = resize(image, gh * TILE, gw * TILE);
        for r in 0..gh {
            for c in 0..gw {
                let mut data = Vec::with_capacity(TILE * TILE * 3);
                for y in 0..TILE {
                    let from = ((r * TILE + y) * big.width + c * TILE) * 3;
                    data.extend_from_slice(&big.data[from..from + TILE * 3]);
                }
                out.push(Rgb {
                    width: TILE,
                    height: TILE,
                    data,
                });
            }
        }
    }
    out.push(resize(image, plan.thumbnail.0, plan.thumbnail.1));
    Ok(out)
}

/// A crop as patches: `[rows * cols, 16 * 16 * 3]`, each patch row by row, pixel by pixel,
/// channel last, values `(v - 127.5) / 127.5`.
fn patchify(crop: &Rgb) -> (Vec<f32>, usize, usize) {
    let (rows, cols) = (crop.height / PATCH, crop.width / PATCH);
    let width = PATCH * PATCH * 3;
    let mut out = vec![0f32; rows * cols * width];
    for py in 0..rows {
        for px in 0..cols {
            let dst = &mut out[(py * cols + px) * width..][..width];
            for r in 0..PATCH {
                let from = ((py * PATCH + r) * crop.width + px * PATCH) * 3;
                for (d, v) in dst[r * PATCH * 3..(r + 1) * PATCH * 3]
                    .iter_mut()
                    .zip(&crop.data[from..from + PATCH * 3])
                {
                    *d = (f32::from(*v) - 127.5) / 127.5;
                }
            }
        }
    }
    (out, rows, cols)
}

#[derive(Debug)]
struct Block {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    down_w: Vec<f32>,
    down_b: Vec<f32>,
}

/// The SigLIP2 tower and the projector into the trunk's width.
#[derive(Debug)]
pub struct VisionTower {
    width: usize,
    heads: usize,
    ffn: usize,
    eps: f32,
    out_width: usize,
    projector_hidden: usize,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// `[POSITION_SIDE * POSITION_SIDE, width]`.
    position: Vec<f32>,
    blocks: Vec<Block>,
    post_w: Vec<f32>,
    post_b: Vec<f32>,
    mm1_w: Vec<f32>,
    mm1_b: Vec<f32>,
    mm2_w: Vec<f32>,
    mm2_b: Vec<f32>,
}

impl VisionTower {
    /// Whether a GGUF carries a vision tower.
    pub fn is_present(gguf: &GgufFile) -> bool {
        gguf.get_u32("d1.vision.block_count").is_some()
    }

    /// Read the tower from `d1.vision.*` settings and `d1.v.*` tensors.
    ///
    /// # Errors
    ///
    /// Fails when a setting or a tensor is missing or has the wrong size, a
    /// width is zero or mismatched, or the present epsilon is not positive
    /// and finite.
    pub fn from_gguf(gguf: &GgufFile, out_width: usize) -> Result<Self> {
        let setting = |key: &str| super::setting_u32(gguf, key);
        let width = setting("d1.vision.embedding_length")?;
        let blocks = setting("d1.vision.block_count")?;
        let heads = setting("d1.vision.attention.head_count")?;
        let ffn = setting("d1.vision.feed_forward_length")?;
        let projector_hidden = setting("d1.vision.projector_hidden_length")?;
        ensure!(
            setting("d1.vision.patch_size")? == PATCH
                && setting("d1.vision.position_side")? == POSITION_SIDE,
            "the vision tower is not 16 px patches on a 16 x 16 position table"
        );
        ensure!(heads >= 1, "the vision tower has no heads");
        ensure!(width >= 1, "the vision tower has no width");
        ensure!(width % heads == 0, "{heads} heads do not divide {width}");
        // the feed-forward width only feeds block tensors, so a block-less tower
        // needs no width; the projector width always feeds tensors below
        ensure!(
            blocks == 0 || ffn >= 1,
            "the vision tower has no feed-forward width"
        );
        ensure!(
            projector_hidden >= 1,
            "the vision tower has no projector width"
        );
        let eps = gguf.get_f32("d1.vision.layer_norm_epsilon").unwrap_or(1e-6);
        // old files without the key still default; a present-but-bad value would
        // NaN every vision prefix, so refuse it here
        ensure!(
            eps.is_finite() && eps > 0.0,
            "the vision tower epsilon ({eps}) is not positive and finite"
        );
        // element counts multiply file-declared widths, so they route
        // through the checked product: the plain multiply wraps on 32-bit
        let t = |name: &str, factors: &[usize]| {
            let elements = super::checked_len(name, factors)?;
            super::load_f32(gguf, "vision tower", name, elements)
        };
        // no pre-size: `blocks` is a file-declared count, and reserving it up front
        // turns a corrupt GGUF into an aborted process instead of a load error
        let mut layers = Vec::new();
        for i in 0..blocks {
            let b = |s: &str, n: &[usize]| t(&format!("d1.v.blk.{i}.{s}"), n);
            layers.push(Block {
                ln1_w: b("ln1.weight", &[width])?,
                ln1_b: b("ln1.bias", &[width])?,
                q_w: b("attn_q.weight", &[width, width])?,
                q_b: b("attn_q.bias", &[width])?,
                k_w: b("attn_k.weight", &[width, width])?,
                k_b: b("attn_k.bias", &[width])?,
                v_w: b("attn_v.weight", &[width, width])?,
                v_b: b("attn_v.bias", &[width])?,
                o_w: b("attn_out.weight", &[width, width])?,
                o_b: b("attn_out.bias", &[width])?,
                ln2_w: b("ln2.weight", &[width])?,
                ln2_b: b("ln2.bias", &[width])?,
                up_w: b("ffn_up.weight", &[ffn, width])?,
                up_b: b("ffn_up.bias", &[ffn])?,
                down_w: b("ffn_down.weight", &[width, ffn])?,
                down_b: b("ffn_down.bias", &[width])?,
            });
        }
        let merged = super::checked_len("d1.v merged width", &[width, UNSHUFFLE, UNSHUFFLE])?;
        Ok(Self {
            width,
            heads,
            ffn,
            eps,
            out_width,
            projector_hidden,
            patch_w: t("d1.v.patch_embd.weight", &[width, PATCH, PATCH, 3])?,
            patch_b: t("d1.v.patch_embd.bias", &[width])?,
            position: t(
                "d1.v.position_embd.weight",
                &[POSITION_SIDE, POSITION_SIDE, width],
            )?,
            blocks: layers,
            post_w: t("d1.v.post_ln.weight", &[width])?,
            post_b: t("d1.v.post_ln.bias", &[width])?,
            mm1_w: t("d1.v.mm.1.weight", &[projector_hidden, merged])?,
            mm1_b: t("d1.v.mm.1.bias", &[projector_hidden])?,
            mm2_w: t("d1.v.mm.2.weight", &[out_width, projector_hidden])?,
            mm2_b: t("d1.v.mm.2.bias", &[out_width])?,
        })
    }

    /// The position table resized to a `rows x cols` patch grid, `[rows * cols, width]`:
    /// antialiased bilinear, as `Siglip2VisionEmbeddings.resize_positional_embeddings` does.
    fn positions(&self, rows: usize, cols: usize) -> Vec<f32> {
        let d = self.width;
        // width first: [POSITION_SIDE, cols, d]
        let cw = axis_weights(POSITION_SIDE, cols);
        let mut mid = vec![0f32; POSITION_SIDE * cols * d];
        for y in 0..POSITION_SIDE {
            for (x, (lo, w)) in cw.iter().enumerate() {
                let dst = &mut mid[(y * cols + x) * d..][..d];
                for (k, wk) in w.iter().enumerate() {
                    let src = &self.position[(y * POSITION_SIDE + lo + k) * d..][..d];
                    for (o, v) in dst.iter_mut().zip(src) {
                        *o += *wk as f32 * v;
                    }
                }
            }
        }
        // then height: [rows, cols, d]
        let rw = axis_weights(POSITION_SIDE, rows);
        let mut out = vec![0f32; rows * cols * d];
        for (y, (lo, w)) in rw.iter().enumerate() {
            for x in 0..cols {
                let dst = &mut out[(y * cols + x) * d..][..d];
                for (k, wk) in w.iter().enumerate() {
                    let src = &mid[((lo + k) * cols + x) * d..][..d];
                    for (o, v) in dst.iter_mut().zip(src) {
                        *o += *wk as f32 * v;
                    }
                }
            }
        }
        out
    }

    /// Full (unmasked) multi-head attention over `n` tokens of `q`, `k`, `v` (`[n, width]`).
    fn attend(&self, q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
        let d = self.width;
        let hd = d / self.heads;
        let scale = (hd as f32).powf(-0.5);
        let mut out = vec![0f32; n * d];
        out.par_chunks_mut(d).enumerate().for_each(|(t, row)| {
            let mut scores = vec![0f32; n];
            for h in 0..self.heads {
                let qh = &q[t * d + h * hd..][..hd];
                for (j, s) in scores.iter_mut().enumerate() {
                    *s = cpu::dot_f32(qh, &k[j * d + h * hd..][..hd]) * scale;
                }
                super::softmax_exact(&mut scores);
                let dst = &mut row[h * hd..(h + 1) * hd];
                for (j, s) in scores.iter().enumerate() {
                    let w = *s;
                    for (o, x) in dst.iter_mut().zip(&v[j * d + h * hd..][..hd]) {
                        *o += w * x;
                    }
                }
            }
        });
        out
    }

    /// The tower over one crop: `[rows * cols, width]` after the final LayerNorm.
    fn tower(&self, crop: &Rgb) -> (Vec<f32>, usize, usize) {
        let d = self.width;
        let (patches, rows, cols) = patchify(crop);
        let n = rows * cols;
        let mut x = vec![0f32; n * d];
        matmul_nt_f32(
            n,
            d,
            PATCH * PATCH * 3,
            &patches,
            &self.patch_w,
            Some(&self.patch_b),
            &mut x,
        );
        for (a, b) in x.iter_mut().zip(self.positions(rows, cols)) {
            *a += b;
        }
        let project = |input: &[f32], w: &[f32], b: &[f32], out_rows: usize, k: usize| {
            let mut out = vec![0f32; n * out_rows];
            matmul_nt_f32(n, out_rows, k, input, w, Some(b), &mut out);
            out
        };
        for blk in &self.blocks {
            let mut normed = x.clone();
            for row in normed.chunks_exact_mut(d) {
                cpu::layer_norm_inplace(row, &blk.ln1_w, &blk.ln1_b, self.eps);
            }
            let q = project(&normed, &blk.q_w, &blk.q_b, d, d);
            let k = project(&normed, &blk.k_w, &blk.k_b, d, d);
            let v = project(&normed, &blk.v_w, &blk.v_b, d, d);
            let ctx = self.attend(&q, &k, &v, n);
            let o = project(&ctx, &blk.o_w, &blk.o_b, d, d);
            for (a, b) in x.iter_mut().zip(&o) {
                *a += b;
            }
            let mut normed = x.clone();
            for row in normed.chunks_exact_mut(d) {
                cpu::layer_norm_inplace(row, &blk.ln2_w, &blk.ln2_b, self.eps);
            }
            let mut mid = project(&normed, &blk.up_w, &blk.up_b, self.ffn, d);
            cpu::gelu_inplace(&mut mid);
            let down = project(&mid, &blk.down_w, &blk.down_b, d, self.ffn);
            for (a, b) in x.iter_mut().zip(&down) {
                *a += b;
            }
        }
        for row in x.chunks_exact_mut(d) {
            cpu::layer_norm_inplace(row, &self.post_w, &self.post_b, self.eps);
        }
        (x, rows, cols)
    }

    /// 2x2 pixel unshuffle then the projector: `[rows, cols, width]` to
    /// `[(rows / 2) * (cols / 2), out_width]`. A merged token is the 2x2 block of patches laid
    /// out row by row, channel last.
    fn project(&self, tokens: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let d = self.width;
        let (oy, ox) = (rows / UNSHUFFLE, cols / UNSHUFFLE);
        let merged = d * UNSHUFFLE * UNSHUFFLE;
        let mut grouped = vec![0f32; oy * ox * merged];
        for y in 0..oy {
            for x in 0..ox {
                let dst = &mut grouped[(y * ox + x) * merged..][..merged];
                for dy in 0..UNSHUFFLE {
                    for dx in 0..UNSHUFFLE {
                        let from = ((y * UNSHUFFLE + dy) * cols + x * UNSHUFFLE + dx) * d;
                        let to = (dy * UNSHUFFLE + dx) * d;
                        dst[to..to + d].copy_from_slice(&tokens[from..from + d]);
                    }
                }
            }
        }
        let n = oy * ox;
        let mut mid = vec![0f32; n * self.projector_hidden];
        matmul_nt_f32(
            n,
            self.projector_hidden,
            merged,
            &grouped,
            &self.mm1_w,
            Some(&self.mm1_b),
            &mut mid,
        );
        super::gelu_exact_inplace(&mut mid);
        let mut out = vec![0f32; n * self.out_width];
        matmul_nt_f32(
            n,
            self.out_width,
            self.projector_hidden,
            &mid,
            &self.mm2_w,
            Some(&self.mm2_b),
            &mut out,
        );
        out
    }

    /// The prefix embeddings of `images`, every image's tiles then its thumbnail, images in
    /// the order given: `[rows * out_width]`.
    ///
    /// # Errors
    ///
    /// Fails on an empty image or a crop whose patch grid cannot be unshuffled.
    pub fn encode(&self, images: &[Rgb]) -> Result<Vec<f32>> {
        let mut prefix = Vec::new();
        for image in images {
            for crop in crops(image)? {
                let (tokens, rows, cols) = self.tower(&crop);
                ensure!(
                    rows % UNSHUFFLE == 0 && cols % UNSHUFFLE == 0,
                    "a {rows} x {cols} patch grid cannot be unshuffled"
                );
                prefix.extend(self.project(&tokens, rows, cols));
            }
        }
        Ok(prefix)
    }

    /// How many prefix rows `images` will take, without running the tower.
    ///
    /// # Errors
    ///
    /// Fails on an empty image.
    pub fn prefix_rows(images: &[(usize, usize)]) -> Result<usize> {
        let mut rows = 0;
        for &(width, height) in images {
            let plan = layout(width, height)?;
            let tiles = if plan.tiled {
                plan.grid.0 * plan.grid.1
            } else {
                0
            };
            let per_tile = (TILE / PATCH / UNSHUFFLE).pow(2);
            let thumb =
                plan.thumbnail.0 / PATCH / UNSHUFFLE * (plan.thumbnail.1 / PATCH / UNSHUFFLE);
            rows += tiles * per_tile + thumb;
        }
        Ok(rows)
    }
}

/// A flat PNG of `w` by `h` pixels for tests, shared by the image tests here and the
/// pipeline tests in `super` (one encoder, so the pixel layout is asserted once).
#[cfg(test)]
#[cfg(feature = "vl-preprocess")]
pub(crate) fn png_bytes_for_test(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([10u8, 10, 10]));
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

/// A zero-block vision tower GGUF for tests, shared by the loader tests here and the
/// pipeline tests in `super` (one builder, so a new required tensor is added once).
#[cfg(test)]
pub(crate) fn tower_builder(
    heads: u32,
    bias_len: usize,
    patch: u32,
    side: u32,
    width: usize,
) -> crate::gguf::GgufBuilder {
    crate::gguf::GgufBuilder::new()
        .kv_u32("d1.vision.block_count", 0)
        .kv_u32("d1.vision.embedding_length", width as u32)
        .kv_u32("d1.vision.feed_forward_length", width as u32)
        .kv_u32("d1.vision.attention.head_count", heads)
        .kv_u32("d1.vision.patch_size", patch)
        .kv_u32("d1.vision.position_side", side)
        .kv_u32("d1.vision.projector_hidden_length", 64)
        .tensor_f32(
            "d1.v.patch_embd.weight",
            &[width * 16 * 16 * 3],
            &vec![0.0; width * 16 * 16 * 3],
        )
        .tensor_f32("d1.v.patch_embd.bias", &[bias_len], &vec![0.0; bias_len])
        .tensor_f32(
            "d1.v.position_embd.weight",
            &[16 * 16 * width],
            &vec![0.0; 16 * 16 * width],
        )
        .tensor_f32("d1.v.post_ln.weight", &[width], &vec![0.0; width])
        .tensor_f32("d1.v.post_ln.bias", &[width], &vec![0.0; width])
        .tensor_f32(
            "d1.v.mm.1.weight",
            &[64 * width * 4],
            &vec![0.0; 64 * width * 4],
        )
        .tensor_f32("d1.v.mm.1.bias", &[64], &vec![0.0; 64])
        .tensor_f32("d1.v.mm.2.weight", &[64 * width], &vec![0.0; 64 * width])
        .tensor_f32("d1.v.mm.2.bias", &[width], &vec![0.0; width])
}

/// A zero-block vision tower GGUF with the standard test geometry: 1 head,
/// 64-wide, 16 px patches on a 16 x 16 table. Scalar-guard legs override via
/// chained `.kv_u32(...)` (builder keys are last-wins); only legs that vary
/// tensor bytes keep the positional [`tower_builder`].
#[cfg(test)]
pub(crate) fn tower_builder_default() -> crate::gguf::GgufBuilder {
    tower_builder(1, 64, 16, 16, 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_follow_the_smart_resize() {
        // 640 x 480 is over the pixel budget once rounded, so it is scaled into it
        let small = layout(640, 480).unwrap();
        assert_eq!(small.thumbnail, (416, 576));
        assert!(!small.tiled);
        assert_eq!(small.grid, (1, 1));
        // 2048 x 1536 is large: a 3 x 2 grid, and the same thumbnail
        let big = layout(2048, 1536).unwrap();
        assert!(big.tiled);
        assert_eq!(big.grid, (3, 2));
        assert_eq!(big.thumbnail, (416, 576));
        // a tiny image is scaled up to the minimum budget
        let tiny = layout(40, 30).unwrap();
        assert!(tiny.thumbnail.0 * tiny.thumbnail.1 >= MIN_PIXELS);
        assert!(layout(0, 10).is_err());
    }

    #[test]
    fn prefix_rows_count_tiles_and_thumbnail() {
        // one 416 x 576 thumbnail: 13 x 18 merged tokens
        assert_eq!(VisionTower::prefix_rows(&[(640, 480)]).unwrap(), 13 * 18);
        // 6 tiles of 16 x 16 merged tokens, plus the thumbnail
        assert_eq!(
            VisionTower::prefix_rows(&[(2048, 1536)]).unwrap(),
            6 * 16 * 16 + 13 * 18
        );
    }

    #[test]
    fn resizing_a_flat_image_keeps_it_flat() {
        let flat = Rgb::filled(37, 23, [10, 200, 77]);
        for (h, w) in [(11, 9), (23, 37), (64, 96)] {
            let out = resize(&flat, h, w);
            assert_eq!((out.height, out.width), (h, w));
            assert!(
                out.data
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .all(|p| *p == [10, 200, 77])
            );
        }
    }

    #[test]
    fn resizing_to_the_same_size_is_the_identity() {
        let data: Vec<u8> = (0..16 * 12 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let img = Rgb {
            width: 16,
            height: 12,
            data,
        };
        assert_eq!(resize(&img, 12, 16), img);
    }

    #[test]
    fn fixed_point_weights_use_the_widest_precision_that_fits() {
        let (precision, weights) = fixed_point_weights(8, 8);
        assert_eq!(precision, 14);
        assert!(
            weights
                .iter()
                .all(|(_, w)| w.iter().all(|v| *v < (1 << 15)))
        );
        // identity: one weight of exactly 1.0 per output
        assert_eq!(weights[3].1[0], 1 << 14);
    }

    #[test]
    fn a_patch_is_read_row_by_row_with_the_channel_last() {
        let mut img = Rgb::filled(32, 16, [0, 0, 0]);
        // pixel (x = 17, y = 1) is in patch (0, 1), row 1, column 1
        let at = (img.width + 17) * 3;
        img.data[at..at + 3].copy_from_slice(&[255, 127, 0]);
        let (patches, rows, cols) = patchify(&img);
        assert_eq!((rows, cols), (1, 2));
        let width = PATCH * PATCH * 3;
        let probe = &patches[width + (PATCH + 1) * 3..][..3];
        assert_eq!(probe, [1.0, (127.0 - 127.5) / 127.5, -1.0]);
        assert_eq!(patches[0], -1.0);
    }

    #[test]
    fn the_unshuffle_gathers_each_two_by_two_block_row_by_row() {
        // a stand-in tower: identity-width projector is not needed, only the gathering
        let d = 3;
        let (rows, cols) = (4, 4);
        let tokens: Vec<f32> = (0..rows * cols * d).map(|i| i as f32).collect();
        let tower = VisionTower {
            width: d,
            heads: 1,
            ffn: 1,
            eps: 1e-6,
            out_width: 12,
            projector_hidden: 12,
            patch_w: vec![],
            patch_b: vec![],
            position: vec![],
            blocks: vec![],
            post_w: vec![],
            post_b: vec![],
            // identity projector (GELU is applied between, so use a large positive input)
            mm1_w: (0..12 * 12).map(|i| f32::from(i / 12 == i % 12)).collect(),
            mm1_b: vec![0.0; 12],
            mm2_w: (0..12 * 12).map(|i| f32::from(i / 12 == i % 12)).collect(),
            mm2_b: vec![0.0; 12],
        };
        let out = tower.project(&tokens, rows, cols);
        assert_eq!(out.len(), 4 * 12);
        // merged token (0, 1) is patches (0,2) (0,3) (1,2) (1,3): the order is dy, dx, channel
        let want: Vec<f32> = [(0, 2), (0, 3), (1, 2), (1, 3)]
            .iter()
            .flat_map(|&(y, x)| (0..d).map(move |c| ((y * cols + x) * d + c) as f32))
            .collect();
        let got = &out[12..24];
        for (g, w) in got.iter().zip(&want) {
            // two GELUs of a positive value: close to the value, never equal
            assert!(
                (g - w).abs() <= 0.02 * w.abs() + 0.05,
                "{got:?} vs {want:?}"
            );
        }
        assert!(got[3] < got[4] && got[4] < got[5], "channels stay ordered");
    }

    #[test]
    fn base64_payloads_decode() {
        assert_eq!(decode_data_url("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(
            decode_data_url("data:image/png;base64,aGk+Pz8/").unwrap(),
            b"hi>???"
        );
        assert_eq!(decode_data_url("aGk\n=").unwrap(), b"hi");
        assert!(decode_data_url("data:image/png,aGk=").is_err());
        assert!(decode_data_url("a$b").is_err());
    }

    #[test]
    fn gelu_variants_differ_where_they_should() {
        let mut tanh = [0.0f32, 1.0];
        cpu::gelu_inplace(&mut tanh);
        assert!(tanh[0].abs() < 1e-9);
        assert!((tanh[1] - 0.841_192).abs() < 1e-5);
        let mut exact = [1.0f32];
        crate::d1::gelu_exact_inplace(&mut exact);
        assert!((exact[0] - 0.841_344_8).abs() < 1e-5);
    }

    #[cfg(feature = "vl-preprocess")]
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let lsb = crc & 1;
                crc >>= 1;
                if lsb == 1 {
                    crc ^= 0xEDB8_8320;
                }
            }
        }
        !crc
    }

    /// A PNG declaring `declared_w` px wide with a valid CRC, so only the dimension
    /// bound fires.
    #[cfg(feature = "vl-preprocess")]
    fn png_with_declared_width(w: u32, h: u32, declared_w: u32) -> Vec<u8> {
        let mut png = super::png_bytes_for_test(w, h);
        png[16..20].copy_from_slice(&declared_w.to_be_bytes());
        let crc = crc32(&png[12..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
        png
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn a_huge_declared_width_is_refused_before_its_pixels_allocate() {
        let png = super::png_bytes_for_test(64, 64);
        assert!(decode_image(&png).is_ok());
        let png = png_with_declared_width(64, 64, 20_000);
        let err = decode_image(&png).unwrap_err();
        // the dimension bound fires, not a CRC or truncation error: pin the typed
        // variant, not the third-party wording
        let typed = err.downcast_ref::<image::ImageError>();
        assert!(
            matches!(typed, Some(image::ImageError::Limits(_))),
            "expected the dimension bound, got: {err:?}"
        );
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn image_dims_read_the_header_without_the_pixels() {
        let png = super::png_bytes_for_test(64, 48);
        assert_eq!(image_dims(&png).unwrap(), (64, 48));
        // the dimension bound fires on the header dims, with the typed refusal
        let huge = png_with_declared_width(64, 48, 20_000);
        let err = image_dims(&huge).unwrap_err();
        let typed = err.downcast_ref::<image::ImageError>();
        assert!(
            matches!(typed, Some(image::ImageError::Limits(_))),
            "expected the dimension bound, got: {err:?}"
        );
        // non-image bytes refuse as an undeterminable format, not the dimension
        // bound: pin the typed variant, not the third-party wording
        let err = image_dims(b"not an image").unwrap_err();
        let typed = err.downcast_ref::<image::ImageError>();
        assert!(
            matches!(typed, Some(image::ImageError::Unsupported(_))),
            "expected an undeterminable-format refusal, got: {err:?}"
        );
        // the JPEG side of the documented pair: same header read, other container
        let rgb = image::RgbImage::from_pixel(64, 48, image::Rgb([10u8, 10, 10]));
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        assert_eq!(image_dims(&jpeg.into_inner()).unwrap(), (64, 48));
    }

    #[test]
    fn encode_returns_the_rows_prefix_rows_predicts() {
        let tower = VisionTower::from_gguf(&tower_builder_default().build(), 64).unwrap();
        for (w, h) in [(640, 480), (2048, 1536)] {
            let rgb = Rgb::filled(w, h, [10, 200, 77]);
            let prefix = tower.encode(std::slice::from_ref(&rgb)).unwrap();
            assert_eq!(
                prefix.len() / 64,
                VisionTower::prefix_rows(&[(w, h)]).unwrap()
            );
        }
    }

    #[test]
    fn loader_guards_refuse_a_broken_tower() {
        VisionTower::from_gguf(&tower_builder_default().build(), 64).unwrap();
        let err = VisionTower::from_gguf(
            &tower_builder_default()
                .kv_u32("d1.vision.patch_size", 32)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("not 16 px patches"),
            "unexpected message: {err}"
        );
        let err = VisionTower::from_gguf(
            &tower_builder_default()
                .kv_u32("d1.vision.attention.head_count", 0)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no heads"),
            "unexpected message: {err}"
        );
        // the width conjunct feeds from the embedding_length key; the chained key
        // overrides the shared width (builder keys are last-wins), and the shared
        // tensors are unread since the guard fires before any load
        let nowide = tower_builder_default()
            .kv_u32("d1.vision.embedding_length", 0)
            .build();
        let err = VisionTower::from_gguf(&nowide, 64).expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no width"),
            "unexpected message: {err}"
        );
        // a present-but-bad epsilon would NaN every vision prefix
        for bad in [f32::NAN, -1.0, f32::INFINITY] {
            let badeps = tower_builder_default()
                .kv_f32("d1.vision.layer_norm_epsilon", bad)
                .build();
            let err =
                VisionTower::from_gguf(&badeps, 64).expect_err("expected the guard to refuse");
            assert!(
                err.to_string().contains("is not positive and finite"),
                "unexpected message: {err}"
            );
        }
        // the side conjunct: 16 px patches on an 8 x 8 table still refuse
        let err = VisionTower::from_gguf(
            &tower_builder_default()
                .kv_u32("d1.vision.position_side", 8)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("16 x 16 position table"),
            "unexpected message: {err}"
        );
        let err = VisionTower::from_gguf(
            &tower_builder_default()
                .kv_u32("d1.vision.attention.head_count", 3)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("do not divide"),
            "unexpected message: {err}"
        );
        // the short bias is tensor bytes, not a scalar, so this leg keeps the
        // positional builder
        let err = VisionTower::from_gguf(&tower_builder(1, 63, 16, 16, 64).build(), 64)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("has 63 values, expected 64"),
            "unexpected message: {err}"
        );
        let err = VisionTower::from_gguf(&crate::gguf::GgufBuilder::new().build(), 64)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string()
                .contains("missing d1.vision.embedding_length"),
            "unexpected message: {err}"
        );
        // a zero feed-forward width with blocks would load 0-element tensors (the
        // shared builder carries no blocks, so the leg adds one)
        let noffn = tower_builder_default()
            .kv_u32("d1.vision.block_count", 1)
            .kv_u32("d1.vision.feed_forward_length", 0)
            .build();
        let err = VisionTower::from_gguf(&noffn, 64).expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no feed-forward width"),
            "unexpected message: {err}"
        );
        // ... while a block-less tower needs no width, so it loads without one
        VisionTower::from_gguf(
            &tower_builder_default()
                .kv_u32("d1.vision.feed_forward_length", 0)
                .build(),
            64,
        )
        .unwrap();
        // a zero projector width would load 0-element projector tensors
        let noph = tower_builder_default()
            .kv_u32("d1.vision.projector_hidden_length", 0)
            .build();
        let err = VisionTower::from_gguf(&noph, 64).expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no projector width"),
            "unexpected message: {err}"
        );
    }
}

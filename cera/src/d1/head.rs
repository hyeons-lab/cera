//! The decision head.
//!
//! It takes the trunk's final hidden states (one row per token), adds the question-type
//! embedding, runs `block_count` pre-norm transformer layers over every token, and scores the
//! state at each option marker with one shared MLP:
//!
//! ```text
//! x = h + type_embedding[type]
//! repeat block_count times:
//!     x = x + Attention(LayerNorm(x))        # bidirectional, no position encoding
//!     x = x + Down(ReLU(Up(LayerNorm(x))))
//! score = Out(GELU(Linear(LayerNorm(x[marker]))))
//! ```
//!
//! This is `torch.nn.TransformerEncoderLayer(d, d / 64, 4 d, dropout 0, norm_first)` followed by
//! `Sequential(LayerNorm, Linear, GELU, Linear(d, 1))`, with LayerNorm epsilon 1e-5.
//!
//! Only the scores at the markers are read, so the last layer computes its queries and its
//! feed-forward for the marker rows alone; every earlier layer needs all rows because the next
//! layer attends over them.

use anyhow::{Context, Result, ensure};

use std::sync::Arc;

use crate::backend::cpu;
use crate::engine::BackendPreference;
use crate::gguf::GgufFile;
use crate::model::pii::matmul_nt_f32;
use crate::model::vision_encoder_gpu::{
    VitStack, VitStackActivation, VitStackBlock, VitStackSpec, build_vit_stack_native,
};
use crate::model::weights::MmapWeight;
use crate::par::*;

/// Number of question types (`choice`, `score`, `noul`).
const TYPES: usize = 3;
/// Per-head width of the attention (`d / heads` is 64 for every d1 head).
const HEAD_DIM: usize = 64;

struct Layer {
    attn_norm_w: Vec<f32>,
    attn_norm_b: Vec<f32>,
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    ffn_norm_b: Vec<f32>,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    down_w: Vec<f32>,
    down_b: Vec<f32>,
}

/// The head's weights, read once as f32.
pub struct Head {
    n_embd: usize,
    n_heads: usize,
    ffn: usize,
    eps: f32,
    types: Vec<f32>,
    layers: Vec<Layer>,
    cls_norm_w: Vec<f32>,
    cls_norm_b: Vec<f32>,
    cls_w: Vec<f32>,
    cls_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: f32,
    /// The transformer layers on a GPU, when [`Head::accelerate`] found one.
    gpu: Option<Arc<dyn VitStack>>,
}

fn tensor(gguf: &GgufFile, name: &str, elements: usize) -> Result<Vec<f32>> {
    let t = gguf
        .get_tensor(name)
        .with_context(|| format!("the head needs the tensor `{name}`"))?
        .to_f32_vec();
    ensure!(
        t.len() == elements,
        "`{name}` has {} values, expected {elements}",
        t.len()
    );
    Ok(t)
}

impl Head {
    /// Whether a GGUF carries a d1 head.
    pub fn is_present(gguf: &GgufFile) -> bool {
        gguf.get_u32("d1.head.block_count").is_some()
    }

    /// Read the head from `d1.*` metadata and tensors.
    ///
    /// # Errors
    ///
    /// Fails when a setting or a tensor is missing or has the wrong size.
    pub fn from_gguf(gguf: &GgufFile, n_embd: usize) -> Result<Self> {
        let setting = |key: &str| {
            gguf.get_u32(key)
                .map(|v| v as usize)
                .with_context(|| format!("missing {key}"))
        };
        let blocks = setting("d1.head.block_count")?;
        let n_heads = setting("d1.head.attention.head_count")?;
        let ffn = setting("d1.head.feed_forward_length")?;
        ensure!(
            n_heads * HEAD_DIM == n_embd,
            "the head has {n_heads} heads of {HEAD_DIM}, not {n_embd} wide"
        );
        let eps = gguf.get_f32("d1.head.layer_norm_epsilon").unwrap_or(1e-5);
        let d = n_embd;
        let mut layers = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let t = |suffix: &str, elements: usize| {
                tensor(gguf, &format!("d1.blk.{i}.{suffix}"), elements)
            };
            layers.push(Layer {
                attn_norm_w: t("attn_norm.weight", d)?,
                attn_norm_b: t("attn_norm.bias", d)?,
                qkv_w: t("attn_qkv.weight", 3 * d * d)?,
                qkv_b: t("attn_qkv.bias", 3 * d)?,
                out_w: t("attn_output.weight", d * d)?,
                out_b: t("attn_output.bias", d)?,
                ffn_norm_w: t("ffn_norm.weight", d)?,
                ffn_norm_b: t("ffn_norm.bias", d)?,
                up_w: t("ffn_up.weight", ffn * d)?,
                up_b: t("ffn_up.bias", ffn)?,
                down_w: t("ffn_down.weight", d * ffn)?,
                down_b: t("ffn_down.bias", d)?,
            });
        }
        let out_b = tensor(gguf, "d1.cls.output.bias", 1)?[0];
        Ok(Self {
            n_embd,
            n_heads,
            ffn,
            eps,
            types: tensor(gguf, "d1.question_type.weight", TYPES * d)?,
            layers,
            cls_norm_w: tensor(gguf, "d1.cls.norm.weight", d)?,
            cls_norm_b: tensor(gguf, "d1.cls.norm.bias", d)?,
            cls_w: tensor(gguf, "d1.cls.weight", d * d)?,
            cls_b: tensor(gguf, "d1.cls.bias", d)?,
            out_w: tensor(gguf, "d1.cls.output.weight", d)?,
            out_b,
            gpu: None,
        })
    }

    /// Run the transformer layers on the GPU `backend` names, when it is Metal (see
    /// [`build_vit_stack_native`]: on wgpu the host is faster). The scorer (a LayerNorm, one linear
    /// and a GELU on a handful of rows) stays on the host. A pass the device cannot take (more
    /// rows than its buffers hold) runs on the host as before.
    pub fn accelerate(&mut self, gguf: &Arc<GgufFile>, backend: BackendPreference) {
        if backend == BackendPreference::Cpu || self.layers.is_empty() {
            return;
        }
        let from_file = |i: usize, name: &str, rows: usize, cols: usize| -> Result<MmapWeight> {
            let tensor = format!("d1.blk.{i}.{name}.weight");
            let w = MmapWeight::from_gguf(gguf, &tensor)
                .with_context(|| format!("the head needs the tensor `{tensor}`"))?;
            ensure!(
                w.rows == rows && w.cols == cols,
                "`{tensor}` is {} x {}, expected {rows} x {cols}",
                w.rows,
                w.cols
            );
            Ok(w)
        };
        match self.stack_spec(from_file) {
            Ok(spec) => self.gpu = build_vit_stack_native(&spec, backend),
            Err(e) => tracing::warn!("the d1 head stays on the CPU: {e:#}"),
        }
    }

    /// Whether the transformer layers run on a GPU.
    pub fn is_accelerated(&self) -> bool {
        self.gpu.is_some()
    }

    /// The layers as a GPU stack takes them. The packed `in_proj` becomes three linears (the
    /// query, key and value rows of one tensor, copied out whole so a quantized tensor stays
    /// quantized); `weight(layer, name)` supplies a layer's linear weight.
    fn stack_spec(
        &self,
        weight: impl Fn(usize, &str, usize, usize) -> Result<MmapWeight>,
    ) -> Result<VitStackSpec> {
        let (d, ff) = (self.n_embd, self.ffn);
        // rows `from..to` of a row-major tensor, as a weight of their own
        let rows_of = |w: &MmapWeight, from: usize, to: usize| -> Result<MmapWeight> {
            ensure!(
                w.cols.is_multiple_of(w.dtype.block_size()),
                "a {:?} row is not a whole number of blocks",
                w.dtype
            );
            let row_bytes = w.cols / w.dtype.block_size() * w.dtype.block_bytes();
            Ok(MmapWeight::from_owned_bytes(
                w.data()[from * row_bytes..to * row_bytes].to_vec(),
                w.dtype,
                to - from,
                w.cols,
            ))
        };
        let mut blocks = Vec::with_capacity(self.layers.len());
        for (i, l) in self.layers.iter().enumerate() {
            let qkv = weight(i, "attn_qkv", 3 * d, d)?;
            blocks.push(VitStackBlock {
                ln1_w: l.attn_norm_w.clone(),
                ln1_b: l.attn_norm_b.clone(),
                q: rows_of(&qkv, 0, d)?,
                q_b: l.qkv_b[..d].to_vec(),
                k: rows_of(&qkv, d, 2 * d)?,
                k_b: l.qkv_b[d..2 * d].to_vec(),
                v: rows_of(&qkv, 2 * d, 3 * d)?,
                v_b: l.qkv_b[2 * d..].to_vec(),
                o: weight(i, "attn_output", d, d)?,
                o_b: l.out_b.clone(),
                ln2_w: l.ffn_norm_w.clone(),
                ln2_b: l.ffn_norm_b.clone(),
                up: weight(i, "ffn_up", ff, d)?,
                up_b: l.up_b.clone(),
                down: weight(i, "ffn_down", d, ff)?,
                down_b: l.down_b.clone(),
            });
        }
        Ok(VitStackSpec {
            width: d,
            heads: self.n_heads,
            ffn: ff,
            eps: self.eps,
            activation: VitStackActivation::Relu,
            blocks,
            post: None,
        })
    }

    /// The score at each of `markers`, for a question of type `question_type`
    /// (0 choice, 1 score, 2 noul), over the trunk's hidden states `hidden`
    /// (`[n, n_embd]` row-major).
    ///
    /// # Errors
    ///
    /// Fails on a marker outside the sequence or a hidden buffer of the wrong size.
    pub fn scores(
        &self,
        hidden: &[f32],
        n: usize,
        question_type: usize,
        markers: &[usize],
    ) -> Result<Vec<f32>> {
        let d = self.n_embd;
        ensure!(
            question_type < TYPES,
            "question type {question_type} is out of range"
        );
        ensure!(hidden.len() == n * d, "hidden states are not [{n}, {d}]");
        ensure!(
            markers.iter().all(|&m| m < n) && !markers.is_empty(),
            "an option marker is outside the sequence"
        );
        let mut x = hidden.to_vec();
        let type_row = &self.types[question_type * d..(question_type + 1) * d];
        for row in x.chunks_exact_mut(d) {
            for (v, t) in row.iter_mut().zip(type_row) {
                *v += t;
            }
        }
        let mut on_gpu = None;
        if let Some(gpu) = &self.gpu {
            match gpu.run(&x, n) {
                Ok(y) => on_gpu = Some(y),
                Err(e) => tracing::warn!("the d1 head runs on the CPU for {n} rows: {e:#}"),
            }
        }
        let rows: Vec<&[f32]> = if let Some(y) = &on_gpu {
            // the GPU ran every layer over every row
            markers.iter().map(|&m| &y[m * d..(m + 1) * d]).collect()
        } else {
            // every layer but the last continues over all rows; the last keeps the markers alone
            for (li, layer) in self.layers.iter().enumerate() {
                let only = (li + 1 == self.layers.len()).then_some(markers);
                self.layer(layer, &mut x, n, only);
            }
            if self.layers.is_empty() {
                markers.iter().map(|&m| &x[m * d..(m + 1) * d]).collect()
            } else {
                x.chunks_exact(d).collect()
            }
        };
        let mut scores = Vec::with_capacity(markers.len());
        for row in rows {
            let mut g = row.to_vec();
            cpu::layer_norm_inplace(&mut g, &self.cls_norm_w, &self.cls_norm_b, self.eps);
            let mut h = vec![0f32; d];
            matmul_nt_f32(1, d, d, &g, &self.cls_w, Some(&self.cls_b), &mut h);
            for v in &mut h {
                *v = gelu(*v);
            }
            let dot: f32 = h.iter().zip(&self.out_w).map(|(a, b)| a * b).sum();
            scores.push(dot + self.out_b);
        }
        Ok(scores)
    }

    /// One pre-norm layer over `x` (`[n, d]`). With `only` the layer's outputs are computed for
    /// those rows alone and written to the first `only.len()` rows of `x`.
    fn layer(&self, l: &Layer, x: &mut Vec<f32>, n: usize, only: Option<&[usize]>) {
        let d = self.n_embd;
        // attention
        let mut normed = x.clone();
        for row in normed.chunks_exact_mut(d) {
            cpu::layer_norm_inplace(row, &l.attn_norm_w, &l.attn_norm_b, self.eps);
        }
        let mut qkv = vec![0f32; n * 3 * d];
        matmul_nt_f32(n, 3 * d, d, &normed, &l.qkv_w, Some(&l.qkv_b), &mut qkv);
        let query_rows: Vec<usize> = match only {
            Some(rows) => rows.to_vec(),
            None => (0..n).collect(),
        };
        let ctx = if only.is_none() {
            self.attend_all(&qkv, n)
        } else {
            self.attend(&qkv, n, &query_rows)
        };
        let m = query_rows.len();
        let mut attn_out = vec![0f32; m * d];
        matmul_nt_f32(m, d, d, &ctx, &l.out_w, Some(&l.out_b), &mut attn_out);
        // residual into the rows that continue
        let mut y: Vec<f32> = Vec::with_capacity(m * d);
        for (i, &row) in query_rows.iter().enumerate() {
            y.extend(
                x[row * d..(row + 1) * d]
                    .iter()
                    .zip(&attn_out[i * d..(i + 1) * d])
                    .map(|(a, b)| a + b),
            );
        }
        // feed-forward
        let mut normed = y.clone();
        for row in normed.chunks_exact_mut(d) {
            cpu::layer_norm_inplace(row, &l.ffn_norm_w, &l.ffn_norm_b, self.eps);
        }
        let mut mid = vec![0f32; m * self.ffn];
        matmul_nt_f32(m, self.ffn, d, &normed, &l.up_w, Some(&l.up_b), &mut mid);
        for v in &mut mid {
            *v = v.max(0.0);
        }
        let mut down = vec![0f32; m * d];
        matmul_nt_f32(m, d, self.ffn, &mid, &l.down_w, Some(&l.down_b), &mut down);
        for (a, b) in y.iter_mut().zip(&down) {
            *a += b;
        }
        *x = y;
    }

    /// [`Self::attend`] for every row as a query, which is what the first layer needs.
    ///
    /// The row-at-a-time form re-reads all of K and V for every query, so on a long prompt it
    /// is bound by memory traffic, not arithmetic. This hands blocks of queries to the blocked
    /// flash-attention kernel the trunk's CPU prefill uses, which streams each K/V tile once per
    /// block of queries.
    fn attend_all(&self, qkv: &[f32], n: usize) -> Vec<f32> {
        // the kernel reads Q as `[dim, queries]` columns, so each block's Q is transposed once
        const BLOCK: usize = 256;
        let d = self.n_embd;
        let heads = self.n_heads;
        let blocks = n.div_ceil(BLOCK);
        let scale = (HEAD_DIM as f32).powf(-0.5);
        let mut q_cols = vec![0f32; blocks * d * BLOCK];
        q_cols
            .par_chunks_mut(d * BLOCK)
            .enumerate()
            .for_each(|(b, cols)| {
                let rows = BLOCK.min(n - b * BLOCK);
                for r in 0..rows {
                    let q = &qkv[(b * BLOCK + r) * 3 * d..][..d];
                    for (c, &v) in q.iter().enumerate() {
                        cols[c * rows + r] = v;
                    }
                }
            });
        // one chunk per (block, head); K and V are read in place from the packed projections
        let mut parts = vec![0f32; blocks * heads * BLOCK * HEAD_DIM];
        parts
            .par_chunks_mut(BLOCK * HEAD_DIM)
            .enumerate()
            .for_each(|(task, out)| {
                let (b, h) = (task / heads, task % heads);
                let rows = BLOCK.min(n - b * BLOCK);
                cpu::flash_attention_gqa_cpu_opt(
                    &q_cols[b * d * BLOCK..][..d * rows],
                    &qkv[d..],
                    &qkv[2 * d..],
                    &mut out[..rows * HEAD_DIM],
                    h,
                    1,
                    rows,
                    rows,
                    3 * d,
                    h * HEAD_DIM,
                    HEAD_DIM,
                    scale,
                    // without a causal mask the kernel reads `start_pos + rows` keys: all of them
                    n - rows,
                    false,
                );
            });
        let mut ctx = vec![0f32; n * d];
        let (parts, _) = parts.as_chunks::<{ BLOCK * HEAD_DIM }>();
        for (task, part) in parts.iter().enumerate() {
            let (b, h) = (task / heads, task % heads);
            for r in 0..BLOCK.min(n - b * BLOCK) {
                ctx[(b * BLOCK + r) * d + h * HEAD_DIM..][..HEAD_DIM]
                    .copy_from_slice(&part[r * HEAD_DIM..][..HEAD_DIM]);
            }
        }
        ctx
    }

    /// Bidirectional multi-head attention for `query_rows` over all `n` rows of `qkv`
    /// (`[n, 3 d]`, the query, key and value projections side by side). Returns
    /// `[query_rows.len(), d]`.
    fn attend(&self, qkv: &[f32], n: usize, query_rows: &[usize]) -> Vec<f32> {
        let d = self.n_embd;
        let heads = self.n_heads;
        let scale = (HEAD_DIM as f32).powf(-0.5);
        let mut out = vec![0f32; query_rows.len() * d];
        out.par_chunks_mut(d).enumerate().for_each(|(qi, out_row)| {
            let q_row = query_rows[qi];
            let mut scores = vec![0f32; n];
            for h in 0..heads {
                let q = &qkv[q_row * 3 * d + h * HEAD_DIM..][..HEAD_DIM];
                for (j, s) in scores.iter_mut().enumerate() {
                    let k = &qkv[j * 3 * d + d + h * HEAD_DIM..][..HEAD_DIM];
                    *s = cpu::dot_f32(q, k) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let dst = &mut out_row[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                for (j, s) in scores.iter().enumerate() {
                    let w = s / sum;
                    let v = &qkv[j * 3 * d + 2 * d + h * HEAD_DIM..][..HEAD_DIM];
                    for (o, v) in dst.iter_mut().zip(v) {
                        *o += w * v;
                    }
                }
            }
        });
        out
    }
}

/// The exact (erf) GELU, `torch.nn.GELU()`'s default.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// `erf` to f32 precision (Abramowitz & Stegun 7.1.26 is too coarse for parity; this is the
/// W. J. Cody rational approximation as used by the C library, evaluated in f64).
fn libm_erf(x: f32) -> f32 {
    erf64(f64::from(x)) as f32
}

fn erf64(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1)) converges for all x but
    // loses precision for large |x|; use the continued fraction of erfc there.
    let ax = x.abs();
    let value = if ax == 0.0 {
        0.0
    } else if ax < 2.5 {
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        let mut n = 0.0f64;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else if ax > 6.0 {
        1.0
    } else {
        // erfc(x) = exp(-x^2) / (x sqrt(pi)) * 1 / (1 + 1/(2x^2) / (1 + 2/(2x^2) / (1 + ...)))
        let x2 = ax * ax;
        let mut frac = 0.0f64;
        for k in (1..=60).rev() {
            frac = (k as f64 / 2.0) / (ax + frac);
        }
        let erfc = (-x2).exp() / ((ax + frac) * std::f64::consts::PI.sqrt());
        1.0 - erfc
    };
    if x < 0.0 { -value } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (-1.0, -0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (5.0, 0.999_999_999_998_462_5),
        ] {
            assert!((erf64(x) - want).abs() < 1e-13, "erf({x}) = {}", erf64(x));
        }
    }

    #[test]
    fn gelu_is_the_exact_form() {
        assert!(gelu(0.0).abs() < 1e-9);
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
        assert!((gelu(10.0) - 10.0).abs() < 1e-6);
    }

    fn head(layers: usize) -> Head {
        let d = HEAD_DIM; // one head
        let ones = |n: usize| vec![1.0f32; n];
        let zeros = |n: usize| vec![0.0f32; n];
        let ffn = 4 * d;
        Head {
            n_embd: d,
            n_heads: 1,
            ffn,
            eps: 1e-5,
            types: zeros(TYPES * d),
            layers: (0..layers)
                .map(|_| Layer {
                    attn_norm_w: ones(d),
                    attn_norm_b: zeros(d),
                    qkv_w: zeros(3 * d * d),
                    qkv_b: zeros(3 * d),
                    out_w: zeros(d * d),
                    out_b: zeros(d),
                    ffn_norm_w: ones(d),
                    ffn_norm_b: zeros(d),
                    up_w: zeros(ffn * d),
                    up_b: zeros(ffn),
                    down_w: zeros(d * ffn),
                    down_b: zeros(d),
                })
                .collect(),
            cls_norm_w: ones(d),
            cls_norm_b: zeros(d),
            cls_w: zeros(d * d),
            cls_b: zeros(d),
            out_w: ones(d),
            out_b: 0.25,
            gpu: None,
        }
    }

    /// The blocked kernel the first layer uses computes what the row-at-a-time form does,
    /// across a block boundary (300 rows is one full block of 256 and a short one).
    #[test]
    fn blocked_attention_matches_the_row_form() {
        let mut h = head(0);
        h.n_embd = 2 * HEAD_DIM;
        h.n_heads = 2;
        for n in [1usize, 5, 256, 300, 513] {
            let qkv: Vec<f32> = (0..n * 3 * h.n_embd)
                .map(|i| ((i as f32 * 0.173).sin() + (i as f32 * 0.011).cos()) * 0.7)
                .collect();
            let rows: Vec<usize> = (0..n).collect();
            let want = h.attend(&qkv, n, &rows);
            let got = h.attend_all(&qkv, n);
            assert_eq!(got.len(), want.len());
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 2e-4, "n={n}: worst {worst}");
        }
    }

    fn noise(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((((i + seed) * 1_103_515_245 + 12_345) % 2000) as f32 / 1000.0 - 1.0) * scale)
            .collect()
    }

    /// A head with two real layers: 128 wide, two heads of 64, deterministic weights.
    fn busy_head() -> Head {
        let (d, ffn) = (2 * HEAD_DIM, 256usize);
        let layer = |s: usize| Layer {
            attn_norm_w: noise(d, s, 0.1).iter().map(|v| v + 1.0).collect(),
            attn_norm_b: noise(d, s + 1, 0.1),
            qkv_w: noise(3 * d * d, s + 2, 0.15),
            qkv_b: noise(3 * d, s + 3, 0.1),
            out_w: noise(d * d, s + 4, 0.15),
            out_b: noise(d, s + 5, 0.1),
            ffn_norm_w: noise(d, s + 6, 0.1).iter().map(|v| v + 1.0).collect(),
            ffn_norm_b: noise(d, s + 7, 0.1),
            up_w: noise(ffn * d, s + 8, 0.15),
            up_b: noise(ffn, s + 9, 0.1),
            down_w: noise(d * ffn, s + 10, 0.15),
            down_b: noise(d, s + 11, 0.1),
        };
        Head {
            n_embd: d,
            n_heads: 2,
            ffn,
            eps: 1e-5,
            types: noise(TYPES * d, 7, 0.2),
            layers: vec![layer(100), layer(200)],
            cls_norm_w: noise(d, 8, 0.1).iter().map(|v| v + 1.0).collect(),
            cls_norm_b: noise(d, 9, 0.1),
            cls_w: noise(d * d, 10, 0.15),
            cls_b: noise(d, 11, 0.1),
            out_w: noise(d, 12, 0.3),
            out_b: 0.1,
            gpu: None,
        }
    }

    /// The layers on a GPU score what the host layers do, over a row count that is not a
    /// multiple of the attention tile and with markers anywhere (the host skips all but the
    /// markers in the last layer, the GPU computes every row).
    #[test]
    fn the_gpu_layers_match_the_host_layers() {
        let mut head = busy_head();
        let d = head.n_embd;
        let owned = |w: &[f32], rows: usize, cols: usize| {
            MmapWeight::from_owned_bytes(
                w.iter().flat_map(|v| v.to_le_bytes()).collect(),
                crate::tensor::DType::F32,
                rows,
                cols,
            )
        };
        let spec = head
            .stack_spec(|i, name, rows, cols| {
                let l = &head.layers[i];
                let w = match name {
                    "attn_qkv" => &l.qkv_w,
                    "attn_output" => &l.out_w,
                    "ffn_up" => &l.up_w,
                    "ffn_down" => &l.down_w,
                    other => panic!("{other}"),
                };
                assert_eq!(w.len(), rows * cols);
                Ok(owned(w, rows, cols))
            })
            .unwrap();
        let Some(gpu) = build_vit_stack_native(&spec, BackendPreference::Auto) else {
            eprintln!("SKIPPED: no Metal device is available");
            return;
        };
        for n in [3usize, 40, 300, 777] {
            let hidden = noise(n * d, 5, 1.0);
            let markers = [0, n / 3, n - 1];
            for kind in 0..TYPES {
                head.gpu = None;
                let want = head.scores(&hidden, n, kind, &markers).unwrap();
                head.gpu = Some(gpu.clone());
                let got = head.scores(&hidden, n, kind, &markers).unwrap();
                for (w, g) in want.iter().zip(&got) {
                    assert!(
                        (w - g).abs() < 2e-3 * (1.0 + w.abs()),
                        "n={n} type={kind}: {w} vs {g}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_weights_leave_only_the_scorer_bias() {
        let h = head(2);
        let hidden: Vec<f32> = (0..3 * HEAD_DIM).map(|i| (i as f32).sin()).collect();
        let scores = h.scores(&hidden, 3, 0, &[0, 2]).unwrap();
        assert_eq!(scores, vec![0.25, 0.25]);
    }

    #[test]
    fn a_head_without_layers_scores_the_marker_rows() {
        let h = head(0);
        let hidden = vec![0.0f32; 4 * HEAD_DIM];
        assert_eq!(h.scores(&hidden, 4, 2, &[1, 3]).unwrap().len(), 2);
    }

    #[test]
    fn bad_inputs_are_refused() {
        let h = head(1);
        let hidden = vec![0.0f32; 2 * HEAD_DIM];
        assert!(h.scores(&hidden, 2, 3, &[0]).is_err(), "type out of range");
        assert!(h.scores(&hidden, 2, 0, &[2]).is_err(), "marker outside");
        assert!(h.scores(&hidden, 2, 0, &[]).is_err(), "no markers");
        assert!(
            h.scores(&hidden[..HEAD_DIM], 2, 0, &[0]).is_err(),
            "short hidden"
        );
    }
}

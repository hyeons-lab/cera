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

use anyhow::{Result, ensure};

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
pub(crate) const HEAD_DIM: usize = 64;

#[derive(Debug)]
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

impl std::fmt::Debug for Head {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Head")
            .field("n_embd", &self.n_embd)
            .field("n_heads", &self.n_heads)
            .field("ffn", &self.ffn)
            .field("eps", &self.eps)
            .field("layers", &self.layers)
            .field("is_accelerated", &self.is_accelerated())
            .finish()
    }
}

/// Rows `from..to` of a row-major tensor, as a weight of their own. The slice is by whole
/// blocks, so a quantized tensor stays quantized.
fn rows_of(w: &MmapWeight, from: usize, to: usize) -> Result<MmapWeight> {
    ensure!(
        w.cols.is_multiple_of(w.dtype.block_size()),
        "a {:?} row is not a whole number of blocks",
        w.dtype
    );
    ensure!(
        from <= to && to <= w.rows,
        "rows {from}..{to} are outside {} rows",
        w.rows
    );
    let row_bytes = w.cols / w.dtype.block_size() * w.dtype.block_bytes();
    let end = (to as u64) * (row_bytes as u64);
    ensure!(
        end <= w.data().len() as u64,
        "rows {from}..{to} need {end} bytes, the tensor holds {}",
        w.data().len()
    );
    Ok(MmapWeight::from_owned_bytes(
        w.data()[from * row_bytes..to * row_bytes].to_vec(),
        w.dtype,
        to - from,
        w.cols,
    ))
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
    /// Fails when a setting or a tensor is missing or has the wrong size, or the
    /// present epsilon is not positive and finite.
    pub fn from_gguf(gguf: &GgufFile, n_embd: usize) -> Result<Self> {
        let setting = |key: &str| super::setting_u32(gguf, key);
        let blocks = setting("d1.head.block_count")?;
        let n_heads = setting("d1.head.attention.head_count")?;
        let ffn = setting("d1.head.feed_forward_length")?;
        ensure!(n_embd >= 1, "the head is 0 wide");
        // checked: `n_heads` is file-declared, and the multiply wraps on 32-bit
        // (67108865 heads * 64 reads back as 64 wide), passing a hostile count
        // that panics in `attend` instead of refusing here
        ensure!(
            n_heads.checked_mul(HEAD_DIM) == Some(n_embd),
            "the head has {n_heads} heads of {HEAD_DIM}, not {n_embd} wide"
        );
        // the feed-forward width only feeds block tensors, so a block-less head
        // needs no width (mirrors the vision tower's guard)
        ensure!(
            blocks == 0 || ffn >= 1,
            "the head has no feed-forward width"
        );
        let eps = gguf.get_f32("d1.head.layer_norm_epsilon").unwrap_or(1e-5);
        // old files without the key still default; a present-but-bad value would
        // silently degenerate every score (+inf zeroes each norm to its bias), so refuse it here
        ensure!(
            eps.is_finite() && eps > 0.0,
            "the head epsilon ({eps}) is not positive and finite"
        );
        let d = n_embd;
        // no pre-size: `blocks` is a file-declared count, and reserving it up front
        // turns a corrupt GGUF into an aborted process instead of a load error
        let mut layers = Vec::new();
        for i in 0..blocks {
            // element counts multiply file-declared widths, so they route
            // through the checked product: the plain multiply wraps on 32-bit
            let t = |suffix: &str, factors: &[usize]| {
                let name = format!("d1.blk.{i}.{suffix}");
                let elements = super::checked_len(&name, factors)?;
                super::load_f32(gguf, "head", &name, elements)
            };
            layers.push(Layer {
                attn_norm_w: t("attn_norm.weight", &[d])?,
                attn_norm_b: t("attn_norm.bias", &[d])?,
                qkv_w: t("attn_qkv.weight", &[3, d, d])?,
                qkv_b: t("attn_qkv.bias", &[3, d])?,
                out_w: t("attn_output.weight", &[d, d])?,
                out_b: t("attn_output.bias", &[d])?,
                ffn_norm_w: t("ffn_norm.weight", &[d])?,
                ffn_norm_b: t("ffn_norm.bias", &[d])?,
                up_w: t("ffn_up.weight", &[ffn, d])?,
                up_b: t("ffn_up.bias", &[ffn])?,
                down_w: t("ffn_down.weight", &[d, ffn])?,
                down_b: t("ffn_down.bias", &[d])?,
            });
        }
        let out_b = super::load_f32(gguf, "head", "d1.cls.output.bias", 1)?[0];
        Ok(Self {
            n_embd,
            n_heads,
            ffn,
            eps,
            types: super::load_f32(
                gguf,
                "head",
                "d1.question_type.weight",
                super::checked_len("d1.question_type.weight", &[TYPES, d])?,
            )?,
            layers,
            cls_norm_w: super::load_f32(gguf, "head", "d1.cls.norm.weight", d)?,
            cls_norm_b: super::load_f32(gguf, "head", "d1.cls.norm.bias", d)?,
            cls_w: super::load_f32(
                gguf,
                "head",
                "d1.cls.weight",
                super::checked_len("d1.cls.weight", &[d, d])?,
            )?,
            cls_b: super::load_f32(gguf, "head", "d1.cls.bias", d)?,
            out_w: super::load_f32(gguf, "head", "d1.cls.output.weight", d)?,
            out_b,
            gpu: None,
        })
    }

    /// Run the transformer layers on the GPU `backend` names, when it is Metal (see
    /// [`build_vit_stack_native`]: on wgpu the host is faster). The scorer (a LayerNorm, one linear
    /// and a GELU on a handful of rows) stays on the host. A pass the device cannot take (more
    /// rows than its buffers hold) runs on the host as before. `gguf` must be the file the head
    /// was loaded from (weights are matched by shape only).
    pub fn accelerate(&mut self, gguf: &Arc<GgufFile>, backend: BackendPreference) {
        if backend == BackendPreference::Cpu || self.layers.is_empty() {
            return;
        }
        let from_file = |i: usize, name: &str, rows: usize, cols: usize| -> Result<MmapWeight> {
            crate::d1::block_weight(gguf, "d1.blk", "head", i, name, rows, cols)
        };
        match self.stack_spec(from_file) {
            // A backend that cannot build must not clear a working stack: only `Some` sticks.
            Ok(spec) => {
                if let Some(stack) = build_vit_stack_native(&spec, backend) {
                    self.gpu = Some(stack);
                }
            }
            Err(e) => {
                crate::d1::warn_cpu_fallback(format_args!("the d1 head stays on the CPU: {e:#}"))
            }
        }
    }

    /// Whether the transformer layers run on a GPU.
    pub fn is_accelerated(&self) -> bool {
        self.gpu.is_some()
    }

    /// The layers as a GPU stack takes them. The packed `attn_qkv` (the source checkpoint's
    /// `self_attn.in_proj`) becomes three linears (the query, key and value rows of one tensor,
    /// copied out whole so a quantized tensor stays quantized); `weight(layer, name)` supplies
    /// a layer's linear weight.
    fn stack_spec(
        &self,
        weight: impl Fn(usize, &str, usize, usize) -> Result<MmapWeight>,
    ) -> Result<VitStackSpec> {
        let (d, ff) = (self.n_embd, self.ffn);
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
    /// (pass [`QuestionType::index`](crate::d1::prompt::QuestionType::index);
    /// 0 choice, 1 score, 2 noul today), over the trunk's hidden states `hidden`
    /// (`[n, n_embd]` row-major).
    ///
    /// # Errors
    ///
    /// Fails on a question type outside the head's table, a marker outside the
    /// sequence, or a hidden buffer of the wrong size.
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
                Err(e) => crate::d1::warn_cpu_fallback(format_args!(
                    "the d1 head runs on the CPU for {n} rows: {e:#}"
                )),
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
            super::gelu_exact_inplace(&mut h);
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

    /// [`Self::attend`] for every row as a query, which is what every layer but the last needs.
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
        // Block-padded scratch by design: the `parallel`-off shim only offers fixed
        // `par_chunks_mut`, so exact-sized blocks would need indexed parallel writes the
        // facade deliberately lacks. The waste is one partial block of zeroing plus one
        // Q-sized gather pass, small next to the kernel work.
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
                super::softmax_exact(&mut scores);
                let dst = &mut out_row[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                for (j, s) in scores.iter().enumerate() {
                    let w = *s;
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

/// A zero-block head GGUF for tests, shared by the loader tests here and the pipeline
/// tests in `super` (one core builder, so a new required tensor is added once).
#[cfg(test)]
pub(crate) fn head_builder(heads: u32, types_len: usize) -> crate::gguf::GgufBuilder {
    head_builder_core(
        heads,
        &vec![0.0; types_len],
        &vec![0.0; 64],
        &vec![0.0; 64 * 64],
        &vec![0.0; 64],
    )
}

/// The tensor list both head fixtures share; only the payloads differ per fixture.
#[cfg(test)]
fn head_builder_core(
    heads: u32,
    types: &[f32],
    norm_w: &[f32],
    cls_w: &[f32],
    out_w: &[f32],
) -> crate::gguf::GgufBuilder {
    crate::gguf::GgufBuilder::new()
        .kv_u32("d1.head.block_count", 0)
        .kv_u32("d1.head.attention.head_count", heads)
        .kv_u32("d1.head.feed_forward_length", 256)
        .tensor_f32("d1.question_type.weight", &[types.len()], types)
        .tensor_f32("d1.cls.norm.weight", &[norm_w.len()], norm_w)
        .tensor_f32("d1.cls.norm.bias", &[64], &vec![0.0; 64])
        .tensor_f32("d1.cls.weight", &[cls_w.len()], cls_w)
        .tensor_f32("d1.cls.bias", &[64], &vec![0.0; 64])
        .tensor_f32("d1.cls.output.weight", &[out_w.len()], out_w)
        .tensor_f32("d1.cls.output.bias", &[1], &[0.0])
}

/// A zero-block head GGUF whose scores track the question-type row: distinct type
/// rows (row `r` holds `r` at channel 0), a unit cls norm, and one-hot cls and
/// output weights. The pipeline wiring leg answers two types through it and
/// asserts the vectors differ, so a hardcoded row fails.
#[cfg(test)]
pub(crate) fn head_builder_typed() -> crate::gguf::GgufBuilder {
    let mut types = vec![0.0f32; TYPES * 64];
    for r in 0..TYPES {
        types[r * 64] = r as f32;
    }
    let mut one_hot = vec![0.0f32; 64 * 64];
    one_hot[0] = 1.0;
    let mut out_hot = vec![0.0f32; 64];
    out_hot[0] = 1.0;
    head_builder_core(1, &types, &vec![1.0; 64], &one_hot, &out_hot)
}

/// A zero-block head GGUF with the standard test geometry: 1 head, 64-wide.
/// Scalar-guard legs override via chained `.kv_u32(...)` (builder keys are
/// last-wins); only legs that vary tensor bytes keep the positional
/// [`head_builder`].
#[cfg(test)]
pub(crate) fn head_builder_default() -> crate::gguf::GgufBuilder {
    head_builder(1, 3 * 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d1::noise;
    use crate::model::vision_encoder_gpu::build_vit_stack;

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

    /// `rows_of` slices a quantized tensor by whole blocks: a Q8_0 tensor split into thirds
    /// keeps each third's bytes and shape, and a row that is not a whole number of blocks is
    /// refused.
    #[test]
    fn rows_of_keeps_quantized_blocks_whole() {
        use crate::tensor::DType;
        let (rows, cols) = (6usize, 64usize);
        let row_bytes = cols / DType::Q8_0.block_size() * DType::Q8_0.block_bytes();
        let mut data = vec![0u8; rows * row_bytes];
        for (r, row) in data.chunks_exact_mut(row_bytes).enumerate() {
            row.fill(r as u8);
        }
        let w = MmapWeight::from_owned_bytes(data.clone(), DType::Q8_0, rows, cols);
        for (third, from) in [0usize, 2, 4].iter().enumerate() {
            let part = rows_of(&w, *from, from + 2).unwrap();
            assert_eq!((part.rows, part.cols, part.dtype), (2, cols, DType::Q8_0));
            assert_eq!(
                part.data(),
                &data[from * row_bytes..(from + 2) * row_bytes],
                "third {third}"
            );
        }
        let bad = MmapWeight::from_owned_bytes(vec![0u8; 48 * 4], DType::Q8_0, 1, 48);
        assert!(rows_of(&bad, 0, 1).is_err());
        // bounds: past-the-end and inverted ranges refuse rather than panicking the slice
        assert!(rows_of(&w, 0, rows + 1).is_err());
        assert!(rows_of(&w, 4, 2).is_err());
        // a short payload with consistent dims refuses on bytes, not on rows (`MmapWeight`
        // has no `Debug`, so `expect_err` is unavailable; match instead)
        let short =
            MmapWeight::from_owned_bytes(vec![0u8; rows * row_bytes - 1], DType::Q8_0, rows, cols);
        let err = match rows_of(&short, 0, rows) {
            Ok(_) => panic!("short payload must refuse"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("need"), "{err}");
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

    /// `accelerate` only lets `Some` stick: a backend that cannot build (here `Gpu`, which
    /// the native router declines without touching a device) must not clear a working stack,
    /// and `Cpu` is a no-op.
    #[test]
    fn accelerate_keeps_last_good_stack() {
        struct KeepStack;
        impl VitStack for KeepStack {
            fn run(&self, _x: &[f32], _n: usize) -> Result<Vec<f32>> {
                unreachable!("the fake stack is never run")
            }
        }

        let head = busy_head();
        let (d, ffn) = (head.n_embd, head.ffn);
        // the layer tensors `accelerate` reads (`d1.blk.{i}.{name}.weight`; dims are
        // `[cols, rows]` in GGUF order)
        let mut gguf = crate::gguf::GgufBuilder::new();
        for i in 0..head.layers.len() {
            for (name, rows, cols) in [
                ("attn_qkv", 3 * d, d),
                ("attn_output", d, d),
                ("ffn_up", ffn, d),
                ("ffn_down", d, ffn),
            ] {
                gguf = gguf.tensor_f32(
                    format!("d1.blk.{i}.{name}.weight"),
                    &[cols, rows],
                    &vec![0.0f32; rows * cols],
                );
            }
        }
        let gguf = Arc::new(gguf.build());

        let mut kept = busy_head();
        kept.gpu = Some(Arc::new(KeepStack));
        kept.accelerate(&gguf, BackendPreference::Gpu);
        assert!(
            kept.is_accelerated(),
            "a backend that cannot build must not clear a working stack"
        );
        let mut fresh = busy_head();
        fresh.accelerate(&gguf, BackendPreference::Cpu);
        assert!(!fresh.is_accelerated());
    }

    /// `accelerate` on a head without layers is a no-op: nothing to move, no device touched.
    #[test]
    fn accelerate_without_layers_is_a_noop() {
        let gguf = Arc::new(crate::gguf::GgufBuilder::new().build());
        let mut h = head(0);
        assert!(h.layers.is_empty());
        h.accelerate(&gguf, BackendPreference::Gpu);
        assert!(!h.is_accelerated());
    }

    /// A `Relu` stack spec over `head`'s layers, for the GPU-vs-host parity legs.
    fn relu_spec(head: &Head) -> VitStackSpec {
        let owned = |w: &[f32], rows: usize, cols: usize| {
            MmapWeight::from_owned_f32(w.to_vec(), rows, cols)
        };
        head.stack_spec(|i, name, rows, cols| {
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
        .unwrap()
    }

    /// Assert a GPU stack scores what the host layers do, for each row count and type.
    fn assert_gpu_layers_match_host(head: &mut Head, gpu: &Arc<dyn VitStack>, ns: &[usize]) {
        let d = head.n_embd;
        for &n in ns {
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

    /// The layers on a GPU score what the host layers do, over a row count that is not a
    /// multiple of the attention tile and with markers anywhere (the host skips all but the
    /// markers in the last layer, the GPU computes every row).
    #[test]
    fn the_gpu_layers_match_the_host_layers() {
        let mut head = busy_head();
        let spec = relu_spec(&head);
        let Some(gpu) = build_vit_stack_native(&spec, BackendPreference::Auto) else {
            // Metal-only stack: CERA_REQUIRE_GPU is set on non-Metal GPU hosts too (the
            // lavapipe leg), where skipping is correct; only the Metal flag forces this test.
            assert!(
                std::env::var("CERA_REQUIRE_METAL").as_deref() != Ok("1"),
                "CERA_REQUIRE_METAL=1 is set but no Metal device is available"
            );
            eprintln!("SKIPPED: no Metal device is available");
            return;
        };
        assert_gpu_layers_match_host(&mut head, &gpu, &[3, 40, 300, 777]);
    }

    /// The wgpu ReLU path scores what the host layers do: `WgpuVitOps::relu` has no production
    /// caller (the head builds Metal-only), so without this leg a wrong bind order or param
    /// word would surface only if head routing ever enables wgpu.
    #[test]
    fn the_wgpu_relu_layers_match_the_host_layers() {
        let mut head = busy_head();
        let spec = relu_spec(&head);
        let Some(gpu) = build_vit_stack(&spec, BackendPreference::Gpu) else {
            assert!(
                std::env::var("CERA_REQUIRE_GPU")
                    .unwrap_or_default()
                    .is_empty(),
                "CERA_REQUIRE_GPU is set but no wgpu adapter is available"
            );
            eprintln!("SKIPPED: no wgpu adapter is available");
            return;
        };
        assert_gpu_layers_match_host(&mut head, &gpu, &[3, 40, 300]);
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
        // zero weights score every marker row at the scorer bias
        assert_eq!(h.scores(&hidden, 4, 2, &[1, 3]).unwrap(), vec![0.25, 0.25]);
    }

    #[test]
    fn scores_follow_the_marker_rows_not_the_first_rows() {
        let mut h = head(0);
        h.cls_w[0] = 1.0; // one-hot: the score tracks row[0] through the norm
        let mut hidden = vec![0.0f32; 2 * HEAD_DIM];
        for (c, v) in hidden[HEAD_DIM..].iter_mut().enumerate() {
            *v = c as f32; // row 1 ramps, row 0 is constant
        }
        assert_eq!(h.scores(&hidden, 2, 0, &[0]).unwrap(), vec![0.25]);
        assert_ne!(
            h.scores(&hidden, 2, 0, &[1]).unwrap(),
            h.scores(&hidden, 2, 0, &[0]).unwrap()
        );
    }

    #[test]
    fn scores_read_the_row_of_the_question_type() {
        let mut h = head(0);
        h.cls_w[0] = 1.0; // one-hot: the score tracks row[0] through the norm
        for r in 0..TYPES {
            h.types[r * HEAD_DIM] = r as f32; // distinct rows, so the wrong row scores wrong
        }
        let hidden = vec![0.0f32; 2 * HEAD_DIM];
        let choice = h.scores(&hidden, 2, 0, &[0]).unwrap();
        let score = h.scores(&hidden, 2, 1, &[0]).unwrap();
        let noul = h.scores(&hidden, 2, 2, &[0]).unwrap();
        assert_ne!(choice, score);
        assert_ne!(score, noul);
        assert_ne!(choice, noul);
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

    #[test]
    fn loader_guards_refuse_a_broken_head() {
        Head::from_gguf(&head_builder_default().build(), 64).unwrap();
        let err = Head::from_gguf(
            &head_builder_default()
                .kv_u32("d1.head.attention.head_count", 2)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("not 64 wide"),
            "unexpected message: {err}"
        );
        // the short types table is tensor bytes, not a scalar, so this leg
        // keeps the positional builder
        let err = Head::from_gguf(&head_builder(1, 3 * 64 - 1).build(), 64)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("has 191 values, expected 192"),
            "unexpected message: {err}"
        );
        let err = Head::from_gguf(&crate::gguf::GgufBuilder::new().build(), 64)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("missing d1.head.block_count"),
            "unexpected message: {err}"
        );
        // zero heads pass the width product at width zero, so only this guard
        // says 0 wide (without it the failure is a tensor length mismatch)
        let err = Head::from_gguf(
            &head_builder_default()
                .kv_u32("d1.head.attention.head_count", 0)
                .build(),
            0,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("0 wide"),
            "unexpected message: {err}"
        );
        // 67108865 heads * 64 wraps to 64 wide on 32-bit: the checked product
        // refuses on every width (on 64-bit the plain multiply refuses too, so
        // this leg pins the message, and the wrap only on narrow targets)
        let err = Head::from_gguf(
            &head_builder_default()
                .kv_u32("d1.head.attention.head_count", 67108865)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("not 64 wide"),
            "unexpected message: {err}"
        );
        // a zero feed-forward width with blocks refuses up front, not deep in
        // the tensor reads; a block-less head needs no width, so it loads
        let err = Head::from_gguf(
            &head_builder_default()
                .kv_u32("d1.head.block_count", 1)
                .kv_u32("d1.head.feed_forward_length", 0)
                .build(),
            64,
        )
        .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no feed-forward width"),
            "unexpected message: {err}"
        );
        Head::from_gguf(
            &head_builder_default()
                .kv_u32("d1.head.feed_forward_length", 0)
                .build(),
            64,
        )
        .unwrap();
        // a present-but-bad epsilon would silently degenerate every score
        for bad in [f32::NAN, -1.0, f32::INFINITY] {
            let err = Head::from_gguf(
                &head_builder_default()
                    .kv_f32("d1.head.layer_norm_epsilon", bad)
                    .build(),
                64,
            )
            .expect_err("expected the guard to refuse");
            assert!(
                err.to_string().contains("is not positive and finite"),
                "unexpected message: {err}"
            );
        }
    }
}

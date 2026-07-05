//! LoRA (Low-Rank Adaptation) adapters.
//!
//! An adapter adds a low-rank delta to selected weight matrices: for a base
//! projection `y = W·x`, the adapted output is
//!
//! ```text
//! y = W·x + scale · B·(A·x)
//! ```
//!
//! where `A` is `[rank × k]` (down-projection, input width `k`), `B` is
//! `[d × rank]` (up-projection, output width `d`), and `scale = alpha / rank`.
//! Applying it at runtime (rather than merging into `W`) keeps the base weights
//! quantized and untouched, so adapters can be hot-swapped / unloaded per session.
//!
//! This module is the **loader + math**: it parses adapter files (GGUF from
//! llama.cpp's `convert_lora_to_gguf`, or PEFT `.safetensors`) into f32 factors
//! and exposes the pure apply helpers. Wiring an adapter into the model forward
//! passes lives in the backends (a later PR).
//!
//! Factors are stored **row-major, pre-dequantized to f32** (`a[i·k + j]` is
//! `A[i][j]`, `b[i·rank + j]` is `B[i][j]`). Adapters are tiny (rank ≤ ~64), so
//! f32 keeps the correction exact and gives one shared apply path across every
//! backend with no dtype dispatch.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::gguf::GgufFile;

/// The standard linear-projection targets a v1 adapter can modify: the four
/// attention projections and the three FFN projections. (LFM2's gated-conv
/// `in_proj`/`out_proj` are not v1 targets.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoraTarget {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl LoraTarget {
    /// All seven targets, in `index()` order.
    pub const ALL: [LoraTarget; 7] = [
        LoraTarget::AttnQ,
        LoraTarget::AttnK,
        LoraTarget::AttnV,
        LoraTarget::AttnOutput,
        LoraTarget::FfnGate,
        LoraTarget::FfnUp,
        LoraTarget::FfnDown,
    ];

    /// Dense array index (0..7) for `LoraLayer::targets`.
    pub fn index(self) -> usize {
        match self {
            LoraTarget::AttnQ => 0,
            LoraTarget::AttnK => 1,
            LoraTarget::AttnV => 2,
            LoraTarget::AttnOutput => 3,
            LoraTarget::FfnGate => 4,
            LoraTarget::FfnUp => 5,
            LoraTarget::FfnDown => 6,
        }
    }

    /// The GGUF base-weight stem, e.g. `attn_q` in `blk.N.attn_q.weight`.
    fn gguf_stem(self) -> &'static str {
        match self {
            LoraTarget::AttnQ => "attn_q",
            LoraTarget::AttnK => "attn_k",
            LoraTarget::AttnV => "attn_v",
            LoraTarget::AttnOutput => "attn_output",
            LoraTarget::FfnGate => "ffn_gate",
            LoraTarget::FfnUp => "ffn_up",
            LoraTarget::FfnDown => "ffn_down",
        }
    }

    /// The GGUF stem → target. `None` for stems we don't adapt in v1.
    fn from_gguf_stem(stem: &str) -> Option<LoraTarget> {
        LoraTarget::ALL.into_iter().find(|t| t.gguf_stem() == stem)
    }

    /// The PEFT sub-module name, e.g. `self_attn.q_proj`.
    fn from_peft_module(module: &str) -> Option<LoraTarget> {
        match module {
            "self_attn.q_proj" => Some(LoraTarget::AttnQ),
            "self_attn.k_proj" => Some(LoraTarget::AttnK),
            "self_attn.v_proj" => Some(LoraTarget::AttnV),
            "self_attn.o_proj" => Some(LoraTarget::AttnOutput),
            "mlp.gate_proj" => Some(LoraTarget::FfnGate),
            "mlp.up_proj" => Some(LoraTarget::FfnUp),
            "mlp.down_proj" => Some(LoraTarget::FfnDown),
            _ => None,
        }
    }
}

/// One target's low-rank factors, pre-dequantized to f32 (row-major).
pub struct LoraTargetWeights {
    /// Down-projection `A`, `[rank × k]` row-major.
    pub a: Vec<f32>,
    /// Up-projection `B`, `[d × rank]` row-major.
    pub b: Vec<f32>,
    /// Low rank `r`.
    pub rank: usize,
    /// Input width (base projection's input dim).
    pub k: usize,
    /// Output width (base projection's output dim).
    pub d: usize,
    /// `alpha / rank` — folded into the apply.
    pub scale: f32,
}

impl LoraTargetWeights {
    fn new(
        a: Vec<f32>,
        rank_a: usize,
        k: usize,
        b: Vec<f32>,
        d: usize,
        rank_b: usize,
        alpha: f32,
    ) -> Result<Self> {
        ensure!(
            rank_a == rank_b,
            "LoRA rank mismatch between A ({rank_a}) and B ({rank_b})"
        );
        ensure!(rank_a > 0 && k > 0 && d > 0, "LoRA dims must be non-zero");
        ensure!(
            a.len() == rank_a * k,
            "LoRA A size {} != rank*k {}",
            a.len(),
            rank_a * k
        );
        ensure!(
            b.len() == d * rank_a,
            "LoRA B size {} != d*rank {}",
            b.len(),
            d * rank_a
        );
        Ok(Self {
            a,
            b,
            rank: rank_a,
            k,
            d,
            scale: alpha / rank_a as f32,
        })
    }
}

/// The (up to seven) target deltas for one transformer layer.
#[derive(Default)]
pub struct LoraLayer {
    targets: [Option<LoraTargetWeights>; 7],
}

/// A loaded LoRA adapter: per-layer low-rank deltas plus scaling.
pub struct LoraAdapterWeights {
    layers: Vec<LoraLayer>,
    default_scale: f32,
}

impl LoraAdapterWeights {
    /// The delta for `(layer, target)`, or `None` if the adapter doesn't touch it.
    pub fn get(&self, layer: usize, target: LoraTarget) -> Option<&LoraTargetWeights> {
        self.layers.get(layer)?.targets[target.index()].as_ref()
    }

    /// Number of layers the adapter spans (one past the highest layer index seen).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// `alpha / rank` reported by the adapter (or derived), for diagnostics.
    pub fn default_scale(&self) -> f32 {
        self.default_scale
    }

    /// Total number of `(layer, target)` deltas present.
    pub fn target_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.targets.iter().filter(|t| t.is_some()).count())
            .sum()
    }

    // ── GGUF ────────────────────────────────────────────────────────────────

    /// Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
    /// a file. Tensors are named `blk.{N}.{stem}.weight.lora_a` / `.lora_b`;
    /// `alpha` is read from the `adapter.lora.alpha` metadata (falling back to
    /// `rank`, i.e. `scale = 1`). Native only — WASM uses [`Self::from_gguf_bytes`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_gguf(path: &Path) -> Result<Arc<Self>> {
        let gguf = GgufFile::open(path).with_context(|| format!("open adapter {path:?}"))?;
        Self::from_gguf_file(&gguf)
    }

    /// Load a GGUF adapter from in-memory bytes (no filesystem — WASM).
    pub fn from_gguf_bytes(bytes: Arc<[u8]>) -> Result<Arc<Self>> {
        let gguf = GgufFile::from_bytes(bytes).context("parse adapter GGUF bytes")?;
        Self::from_gguf_file(&gguf)
    }

    fn from_gguf_file(gguf: &GgufFile) -> Result<Arc<Self>> {
        // llama.cpp's convention (`adapter.lora.alpha`); missing ⇒ scale 1.0.
        let alpha_meta = gguf.get_f32("adapter.lora.alpha");

        let mut builder = AdapterBuilder::new();
        for name in gguf.tensors.keys() {
            let Some((layer, target, is_a)) = parse_gguf_lora_name(name) else {
                continue;
            };
            let (_, rows, cols, _) = gguf.tensor_meta(name)?;
            let data = gguf.get_tensor(name)?.to_f32_vec();
            builder.add_factor(layer, target, is_a, data, rows, cols);
        }
        builder.finish(alpha_meta)
    }

    // ── safetensors (PEFT) ────────────────────────────────────────────────────

    /// Load a PEFT `.safetensors` adapter from a file. Tensors are named
    /// `base_model.model.model.layers.{N}.{module}.lora_A.weight` /
    /// `lora_B.weight`. PEFT stores `alpha` in a sibling `adapter_config.json`,
    /// not in the tensor file — pass it via `alpha` (`None` ⇒ `scale = 1`).
    /// Native only — WASM uses [`Self::from_safetensors_bytes`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_safetensors(path: &Path, alpha: Option<f32>) -> Result<Arc<Self>> {
        let bytes = std::fs::read(path).with_context(|| format!("read adapter {path:?}"))?;
        Self::from_safetensors_bytes(&bytes, alpha)
    }

    /// Load a PEFT safetensors adapter from in-memory bytes.
    pub fn from_safetensors_bytes(bytes: &[u8], alpha: Option<f32>) -> Result<Arc<Self>> {
        let st = SafeTensors::parse(bytes)?;
        let mut builder = AdapterBuilder::new();
        for (name, entry) in st.tensors() {
            let Some((layer, target, is_a)) = parse_peft_lora_name(name) else {
                continue;
            };
            // PEFT weights are row-major `[out, in]`: `lora_A` is `[rank, k]`,
            // `lora_B` is `[d, rank]` — same (rows, cols) convention as GGUF.
            let (rows, cols) = entry
                .shape2()
                .with_context(|| format!("tensor {name} not 2-D"))?;
            let data = st.dequantize(entry, bytes)?;
            builder.add_factor(layer, target, is_a, data, rows, cols);
        }
        // PEFT keeps alpha out-of-band; default to alpha == rank (scale 1).
        builder.finish(alpha)
    }
}

/// Accumulates loose A/B factors keyed by (layer, target), then validates + pairs
/// them into a `LoraAdapterWeights`.
#[derive(Default)]
struct AdapterBuilder {
    /// (layer, target_index) → (A?, B?) as `(data, rows, cols)`.
    factors: std::collections::HashMap<(usize, usize), FactorPair>,
    max_layer: usize,
}

#[derive(Default)]
struct FactorPair {
    a: Option<(Vec<f32>, usize, usize)>,
    b: Option<(Vec<f32>, usize, usize)>,
}

impl AdapterBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn add_factor(
        &mut self,
        layer: usize,
        target: LoraTarget,
        is_a: bool,
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    ) {
        self.max_layer = self.max_layer.max(layer);
        let slot = self.factors.entry((layer, target.index())).or_default();
        if is_a {
            slot.a = Some((data, rows, cols));
        } else {
            slot.b = Some((data, rows, cols));
        }
    }

    fn finish(self, alpha: Option<f32>) -> Result<Arc<LoraAdapterWeights>> {
        ensure!(!self.factors.is_empty(), "adapter contains no LoRA tensors");
        let n_layers = self.max_layer + 1;
        let mut layers: Vec<LoraLayer> = (0..n_layers).map(|_| LoraLayer::default()).collect();

        // A single global scale: alpha (or the first pair's rank if absent).
        let mut default_scale = 1.0f32;
        let mut scale_set = false;

        for ((layer, target_idx), pair) in self.factors {
            let (a, rank_a, k) = pair
                .a
                .with_context(|| format!("layer {layer} target {target_idx}: missing lora_a"))?;
            let (b, d, rank_b) = pair
                .b
                .with_context(|| format!("layer {layer} target {target_idx}: missing lora_b"))?;
            let alpha = alpha.unwrap_or(rank_a as f32);
            let tw = LoraTargetWeights::new(a, rank_a, k, b, d, rank_b, alpha)
                .with_context(|| format!("layer {layer} target {target_idx}"))?;
            if !scale_set {
                default_scale = tw.scale;
                scale_set = true;
            }
            layers[layer].targets[target_idx] = Some(tw);
        }

        Ok(Arc::new(LoraAdapterWeights {
            layers,
            default_scale,
        }))
    }
}

/// Parse a GGUF LoRA tensor name → `(layer, target, is_a)`.
/// e.g. `blk.12.attn_q.weight.lora_a` → `(12, AttnQ, true)`.
fn parse_gguf_lora_name(name: &str) -> Option<(usize, LoraTarget, bool)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;
    let (stem, suffix) = rest.split_once(".weight.")?;
    let is_a = match suffix {
        "lora_a" => true,
        "lora_b" => false,
        _ => return None,
    };
    let target = LoraTarget::from_gguf_stem(stem)?;
    Some((layer, target, is_a))
}

/// Parse a PEFT safetensors LoRA tensor name → `(layer, target, is_a)`.
/// e.g. `base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight`
/// → `(7, AttnQ, true)`.
fn parse_peft_lora_name(name: &str) -> Option<(usize, LoraTarget, bool)> {
    // Find the `layers.{N}.` segment (prefix depth varies by export tooling).
    let idx = name.find("layers.")?;
    let after = &name[idx + "layers.".len()..];
    let (layer_str, rest) = after.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;
    // rest = `{module}.lora_{A,B}.weight`
    let rest = rest.strip_suffix(".weight")?;
    let (module, ab) = rest.rsplit_once('.')?;
    let is_a = match ab {
        "lora_A" => true,
        "lora_B" => false,
        _ => return None,
    };
    let target = LoraTarget::from_peft_module(module)?;
    Some((layer, target, is_a))
}

// ── apply (pure math) ────────────────────────────────────────────────────────

/// Decode-path apply: `y += scale · B·(A·x)`, in place. `x` is length `k`, `y`
/// length `d`; `tmp` is scratch resized to `rank`. Alloc-free given a reused `tmp`.
pub fn apply_decode(t: &LoraTargetWeights, x: &[f32], y: &mut [f32], tmp: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), t.k);
    debug_assert_eq!(y.len(), t.d);
    tmp.clear();
    tmp.resize(t.rank, 0.0);
    // tmp = scale · (A · x)   (A is [rank × k] row-major; fold scale into the
    // small r-vector, the cheapest place).
    for (row, tmp_r) in t.a.chunks_exact(t.k).zip(tmp.iter_mut()) {
        let acc: f32 = row.iter().zip(x).map(|(w, &xi)| w * xi).sum();
        *tmp_r = acc * t.scale;
    }
    // y += B · tmp   (B is [d × rank] row-major)
    for (row, yi) in t.b.chunks_exact(t.rank).zip(y.iter_mut()) {
        let acc: f32 = row.iter().zip(tmp.iter()).map(|(w, &ti)| w * ti).sum();
        *yi += acc;
    }
}

// ── minimal safetensors reader ────────────────────────────────────────────────

/// A parsed safetensors header entry.
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

impl StEntry {
    fn shape2(&self) -> Result<(usize, usize)> {
        ensure!(self.shape.len() == 2, "expected 2-D, got {:?}", self.shape);
        Ok((self.shape[0], self.shape[1]))
    }
}

/// A minimal safetensors reader: `u64-LE header length + JSON header + tensor
/// bytes`. Only the tiny LoRA factors are decoded, so this stays simple.
struct SafeTensors {
    entries: Vec<(String, StEntry)>,
    data_start: usize,
}

impl SafeTensors {
    fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 8, "safetensors: truncated header length");
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .context("safetensors: header length overflow")?;
        ensure!(
            header_end <= bytes.len(),
            "safetensors: header exceeds file"
        );
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])
            .context("safetensors: bad JSON header")?;
        let obj = header
            .as_object()
            .context("safetensors: header is not an object")?;

        let mut entries = Vec::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .with_context(|| format!("{name}: missing dtype"))?
                .to_string();
            let shape = v
                .get("shape")
                .and_then(|s| s.as_array())
                .with_context(|| format!("{name}: missing shape"))?
                .iter()
                .map(|n| n.as_u64().map(|u| u as usize))
                .collect::<Option<Vec<_>>>()
                .with_context(|| format!("{name}: bad shape"))?;
            let offsets = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .with_context(|| format!("{name}: missing data_offsets"))?;
            ensure!(
                offsets.len() == 2,
                "{name}: data_offsets must be [begin, end]"
            );
            let begin = offsets[0].as_u64().context("bad data_offset")? as usize;
            let end = offsets[1].as_u64().context("bad data_offset")? as usize;
            entries.push((
                name.clone(),
                StEntry {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            ));
        }
        Ok(Self {
            entries,
            data_start: header_end,
        })
    }

    fn tensors(&self) -> impl Iterator<Item = (&str, &StEntry)> {
        self.entries.iter().map(|(n, e)| (n.as_str(), e))
    }

    /// Decode one entry's bytes → f32 (F32 / F16 / BF16).
    fn dequantize(&self, e: &StEntry, bytes: &[u8]) -> Result<Vec<f32>> {
        let start = self
            .data_start
            .checked_add(e.begin)
            .context("safetensors: offset overflow")?;
        let end = self
            .data_start
            .checked_add(e.end)
            .context("safetensors: offset overflow")?;
        ensure!(
            end <= bytes.len() && start <= end,
            "safetensors: tensor slice out of range"
        );
        let raw = &bytes[start..end];
        let n: usize = e.shape.iter().product();
        match e.dtype.as_str() {
            "F32" => {
                ensure!(raw.len() == n * 4, "F32 byte count mismatch");
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect())
            }
            "F16" => {
                ensure!(raw.len() == n * 2, "F16 byte count mismatch");
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                    .collect())
            }
            "BF16" => {
                ensure!(raw.len() == n * 2, "BF16 byte count mismatch");
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                    .collect())
            }
            other => bail!("unsupported safetensors dtype for LoRA: {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_name_parse() {
        assert_eq!(
            parse_gguf_lora_name("blk.12.attn_q.weight.lora_a"),
            Some((12, LoraTarget::AttnQ, true))
        );
        assert_eq!(
            parse_gguf_lora_name("blk.0.ffn_down.weight.lora_b"),
            Some((0, LoraTarget::FfnDown, false))
        );
        // Non-lora / unknown target / malformed → None.
        assert_eq!(parse_gguf_lora_name("blk.3.attn_q.weight"), None);
        assert_eq!(parse_gguf_lora_name("blk.3.attn_norm.weight.lora_a"), None);
        assert_eq!(parse_gguf_lora_name("token_embd.weight"), None);
    }

    #[test]
    fn peft_name_parse() {
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight"),
            Some((7, LoraTarget::AttnQ, true))
        );
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.31.mlp.up_proj.lora_B.weight"),
            Some((31, LoraTarget::FfnUp, false))
        );
        assert_eq!(
            parse_peft_lora_name("model.layers.2.self_attn.o_proj.lora_A.weight"),
            Some((2, LoraTarget::AttnOutput, true))
        );
        // Not a lora tensor / unknown module.
        assert_eq!(
            parse_peft_lora_name("base_model.model.model.layers.0.input_layernorm.weight"),
            None
        );
    }

    /// Build a minimal PEFT safetensors buffer with one q_proj adapter on layer 0.
    fn synth_safetensors(rank: usize, k: usize, d: usize, a_val: f32, b_val: f32) -> Vec<u8> {
        let a: Vec<f32> = vec![a_val; rank * k];
        let b: Vec<f32> = vec![b_val; d * rank];
        let a_bytes: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
        let a_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let b_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
        let header = serde_json::json!({
            a_name: { "dtype": "F32", "shape": [rank, k], "data_offsets": [0, a_bytes.len()] },
            b_name: { "dtype": "F32", "shape": [d, rank], "data_offsets": [a_bytes.len(), a_bytes.len() + b_bytes.len()] },
        });
        let header_str = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_str);
        out.extend_from_slice(&a_bytes);
        out.extend_from_slice(&b_bytes);
        out
    }

    #[test]
    fn safetensors_load_shapes_and_scale() {
        let (rank, k, d) = (8, 64, 128);
        let buf = synth_safetensors(rank, k, d, 0.5, 0.25);
        // alpha = 16 → scale = alpha/rank = 2.0
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(16.0)).unwrap();
        assert_eq!(adapter.n_layers(), 1);
        assert_eq!(adapter.target_count(), 1);
        let t = adapter.get(0, LoraTarget::AttnQ).expect("q_proj present");
        assert_eq!((t.rank, t.k, t.d), (rank, k, d));
        assert_eq!(t.a.len(), rank * k);
        assert_eq!(t.b.len(), d * rank);
        assert!((t.scale - 2.0).abs() < 1e-6, "scale {}", t.scale);
        // No adapter on an untouched target.
        assert!(adapter.get(0, LoraTarget::FfnDown).is_none());
        // Default alpha (None) ⇒ scale 1.0.
        let a2 = LoraAdapterWeights::from_safetensors_bytes(&buf, None).unwrap();
        assert!((a2.get(0, LoraTarget::AttnQ).unwrap().scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_math_and_noop() {
        let (rank, k, d) = (2, 3, 4);
        // A = all 1.0, B = all 0.0 → delta is zero (no-op) regardless of scale.
        let buf = synth_safetensors(rank, k, d, 1.0, 0.0);
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(4.0)).unwrap();
        let t = adapter.get(0, LoraTarget::AttnQ).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];
        let before = y.clone();
        let mut tmp = Vec::new();
        apply_decode(t, &x, &mut y, &mut tmp);
        assert_eq!(y, before, "B=0 must be a no-op");

        // Now B = 1.0: delta_o = scale * sum_r( sum_j A[r][j] x[j] ) with A=1 →
        // A·x = sum(x) per rank; tmp[r] = scale*sum(x); B·tmp = rank*scale*sum(x).
        let buf = synth_safetensors(rank, k, d, 1.0, 1.0);
        let adapter = LoraAdapterWeights::from_safetensors_bytes(&buf, Some(4.0)).unwrap();
        let t = adapter.get(0, LoraTarget::AttnQ).unwrap();
        let mut y = vec![0.0; d];
        apply_decode(t, &x, &mut y, &mut tmp);
        let sum_x: f32 = x.iter().sum();
        let expected = rank as f32 * (t.scale * sum_x); // scale = 4/2 = 2
        for &yi in &y {
            assert!((yi - expected).abs() < 1e-5, "{yi} != {expected}");
        }
    }

    #[test]
    fn empty_adapter_errors() {
        // A safetensors buffer with no LoRA tensors → typed error, not a panic.
        let header = serde_json::json!({
            "some.other.weight": { "dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16] },
        });
        let hs = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hs);
        buf.extend_from_slice(&[0u8; 16]);
        assert!(LoraAdapterWeights::from_safetensors_bytes(&buf, None).is_err());
    }
}

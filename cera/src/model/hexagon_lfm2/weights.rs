//! Shared constructor machinery: tensor sources, the weight and KV planners,
//! and the copy-into-`weights_buf` pass.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{
    DenseExtras, HexagonDenseFfn, HexagonFfn, HexagonMoeFfn, HexagonStackedWeight, HexagonWeight,
};
use crate::backend::hexagon::{
    HtpDataType, RpcmemBuffer, TILE_SIZE_Q4_0, TILE_SIZE_Q4_K, TILE_SIZE_Q6_K, TILE_SIZE_Q8_0,
    hexagon_warn, quantize_f32_to_q8_0, repack_q4_0, repack_q4_1, repack_q4_k, repack_q6_k,
    repack_q8_0, repacked_matrix_size_q4_0, repacked_matrix_size_q4_1, repacked_matrix_size_q4_k,
    repacked_matrix_size_q6_k, repacked_matrix_size_q8_0, requant_q5_k_to_q8_0,
};
use crate::gguf::GgufFile;
use crate::model::ModelConfig;
use crate::model::gpu_weight_source::GpuWeightSource;
use crate::model::transformer::WeightRef;
use crate::session::CeraError;

// ---------------------------------------------------------------------------
// Shared constructor machinery. The three constructors (`from_gguf_on`,
// `from_qwen35_model_on`, `from_dense_weight_source_on`) differ in where their
// tensors live and in how a layer is laid out; planning helpers, KV-state
// planning and the copy-into-`weights_buf` pass are the same and live here.
// Each constructor only supplies a `TensorSource` that maps GGUF tensor names
// onto its own storage, and picks the layout order of its layers.
// ---------------------------------------------------------------------------

/// `(repacked byte size, wire dtype, block bytes, tile size)` of one weight.
pub(super) type SlicePlan = (usize, HtpDataType, usize, usize);

/// Split `blk.{layer}.{rest}` into `(layer, rest)`.
fn split_layer_tensor(name: &str) -> Option<(usize, &str)> {
    let (idx, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((idx.parse().ok()?, rest))
}

fn unknown_tensor(name: &str) -> CeraError {
    CeraError::Backend(format!("no source for Hexagon tensor {name}"))
}

/// Where a constructor's tensors come from, addressed by GGUF tensor name. The
/// shared planner asks for shapes and the shared copy pass for data, so a
/// constructor only maps names onto its own storage (a GGUF file, the Qwen 3.5
/// refs, a dense weight source).
pub(super) trait TensorSource {
    /// `(dtype, ne0, ne1)` of the 2D weight `name` (`ne0` = K, `ne1` = rows).
    fn weight_shape(&self, name: &str) -> Result<(crate::tensor::DType, usize, usize), CeraError>;
    /// Wire plan of one expert slice of the stacked weight `name`.
    fn stacked_slice_plan(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<SlicePlan, CeraError>;
    /// F32 vector (norm, bias, conv taps) `name`.
    fn vector(&self, name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CeraError>;
    /// Repack the 2D weight `name` into `buf` at `offset`.
    fn copy_weight(
        &self,
        name: &str,
        offset: usize,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError>;
    /// Repack every expert slice of the stacked weight `name` into `buf`.
    fn copy_stacked(
        &self,
        name: &str,
        sw: &HexagonStackedWeight,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError>;
}

/// Routing dimensions of one routed FFN layer.
#[derive(Clone, Copy)]
pub(super) struct MoeDims {
    pub(super) n_expert: usize,
    pub(super) n_expert_used: usize,
    pub(super) expert_ff_len: usize,
}

/// One planned tensor copy into `weights_buf`, recorded by [`WeightPlanner`]
/// at the moment the tensor's space is reserved, so the copy pass is a flat
/// loop over the plan and never re-derives tensor names.
pub(super) enum CopyOp {
    /// Repack the 2D weight `name` at `offset`.
    Weight { name: String, offset: usize },
    /// Copy the F32 vector `name` verbatim at `offset`; `len` is the reserved
    /// slot length in elements.
    Vector {
        name: String,
        offset: usize,
        len: usize,
    },
    /// Repack every expert slice of the stacked weight `name`.
    Stacked {
        name: String,
        weight: HexagonStackedWeight,
    },
    /// Short-conv taps `[hidden, 3]`: three per-tap planes at `taps` (manual
    /// decode chain) plus the GGUF layout verbatim at `ssm` (SsmConv).
    ConvTaps {
        name: String,
        hidden_size: usize,
        taps: [usize; 3],
        ssm: usize,
    },
    /// Host-built constant vector written verbatim.
    Constant {
        label: &'static str,
        offset: usize,
        values: Vec<f32>,
    },
}

/// Running plan of `weights_buf`: 256-aligned offsets, tiled wire formats, and
/// the copy list that fills them.
pub(super) struct WeightPlanner<'a> {
    pub(super) src: &'a dyn TensorSource,
    pub(super) total: usize,
    pub(super) copies: Vec<CopyOp>,
}

impl<'a> WeightPlanner<'a> {
    pub(super) fn new(src: &'a dyn TensorSource) -> Self {
        Self {
            src,
            total: 0,
            copies: Vec::new(),
        }
    }

    /// Reserve an F32 vector of `len` elements, filled from the source
    /// vector `name`.
    pub(super) fn vector(&mut self, name: &str, len: usize) -> usize {
        let offset = plan_offset(&mut self.total, len * 4);
        self.copies.push(CopyOp::Vector {
            name: name.to_string(),
            offset,
            len,
        });
        offset
    }

    /// Reserve the short-conv tap planes and SsmConv taps of layer `name`:
    /// `(per-tap plane offsets, SsmConv offset)`.
    pub(super) fn conv_taps(&mut self, name: &str, hidden_size: usize) -> ([usize; 3], usize) {
        let taps = [
            plan_offset(&mut self.total, hidden_size * 4),
            plan_offset(&mut self.total, hidden_size * 4),
            plan_offset(&mut self.total, hidden_size * 4),
        ];
        let ssm = plan_offset(&mut self.total, 3 * hidden_size * 4);
        self.copies.push(CopyOp::ConvTaps {
            name: name.to_string(),
            hidden_size,
            taps,
            ssm,
        });
        (taps, ssm)
    }

    /// Reserve a host-built constant vector holding `values`.
    pub(super) fn constant(&mut self, label: &'static str, values: Vec<f32>) -> usize {
        let offset = plan_offset(&mut self.total, values.len() * 4);
        self.copies.push(CopyOp::Constant {
            label,
            offset,
            values,
        });
        offset
    }

    /// Plan the 2D weight `name` for a `[in_dim -> out_dim]` matmul.
    pub(super) fn weight(
        &mut self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<HexagonWeight, CeraError> {
        let (dtype, ne0, ne1) = self.src.weight_shape(name)?;
        // `ne1 >` out_dim is legitimate (a padded-vocab LM head keeps its
        // padding rows); fewer rows than the matmul reads never is.
        if ne0 != in_dim || ne1 < out_dim {
            return Err(CeraError::Backend(format!(
                "{name} is [{ne0}, {ne1}], expected [{in_dim}, at least {out_dim}]"
            )));
        }
        let w = plan_weight(&mut self.total, dtype, ne0, ne1, name, in_dim, out_dim)?;
        self.copies.push(CopyOp::Weight {
            name: name.to_string(),
            offset: w.offset,
        });
        Ok(w)
    }

    /// Plan the stacked expert weight `name` (`n_expert` slices).
    pub(super) fn stacked(
        &mut self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
        n_expert: usize,
    ) -> Result<HexagonStackedWeight, CeraError> {
        let slice_plan = self.src.stacked_slice_plan(name, in_dim, out_dim)?;
        let weight = plan_stacked(&mut self.total, slice_plan, name, in_dim, out_dim, n_expert)?;
        self.copies.push(CopyOp::Stacked {
            name: name.to_string(),
            weight,
        });
        Ok(weight)
    }

    /// Plan an optional F32 bias vector, checking its length.
    pub(super) fn bias(
        &mut self,
        name: &str,
        bias: Option<&[f32]>,
        expected: usize,
    ) -> Result<Option<usize>, CeraError> {
        let Some(b) = bias else { return Ok(None) };
        if b.len() != expected {
            return Err(CeraError::Backend(format!(
                "{name} has {} elements, expected {expected}",
                b.len()
            )));
        }
        Ok(Some(self.vector(name, expected)))
    }

    /// Plan a bias that must be present (the Q/K/V triple); an absent one is
    /// an error, never a silent offset 0 aliasing the first weight.
    pub(super) fn bias_required(
        &mut self,
        name: &str,
        bias: &[f32],
        expected: usize,
    ) -> Result<usize, CeraError> {
        self.bias(name, Some(bias), expected)?
            .ok_or_else(|| CeraError::Backend(format!("{name} is required but absent")))
    }

    /// Plan layer `i`'s FFN: routed experts when `moe` is set, else dense
    /// gate/up/down with the optional `[gate, up, down]` biases.
    pub(super) fn ffn(
        &mut self,
        i: usize,
        hidden: usize,
        inter: usize,
        moe: Option<MoeDims>,
        biases: [Option<&[f32]>; 3],
    ) -> Result<HexagonFfn, CeraError> {
        if let Some(MoeDims {
            n_expert,
            n_expert_used,
            expert_ff_len,
        }) = moe
        {
            let router = self.weight(&format!("blk.{i}.ffn_gate_inp.weight"), hidden, n_expert)?;
            let exp_probs_b_offset = self.vector(&format!("blk.{i}.exp_probs_b.bias"), n_expert);
            let gate = self.stacked(
                &format!("blk.{i}.ffn_gate_exps.weight"),
                hidden,
                expert_ff_len,
                n_expert,
            )?;
            let up = self.stacked(
                &format!("blk.{i}.ffn_up_exps.weight"),
                hidden,
                expert_ff_len,
                n_expert,
            )?;
            let down = self.stacked(
                &format!("blk.{i}.ffn_down_exps.weight"),
                expert_ff_len,
                hidden,
                n_expert,
            )?;
            Ok(HexagonFfn::Moe(HexagonMoeFfn {
                router,
                exp_probs_b_offset,
                gate,
                up,
                down,
                n_expert,
                n_expert_used,
                expert_ff_len,
            }))
        } else {
            let gate = self.weight(&format!("blk.{i}.ffn_gate.weight"), hidden, inter)?;
            let up = self.weight(&format!("blk.{i}.ffn_up.weight"), hidden, inter)?;
            let down = self.weight(&format!("blk.{i}.ffn_down.weight"), inter, hidden)?;
            let [gate_b, up_b, down_b] = biases;
            let gate_bias = self.bias(&format!("blk.{i}.ffn_gate.bias"), gate_b, inter)?;
            let up_bias = self.bias(&format!("blk.{i}.ffn_up.bias"), up_b, inter)?;
            let down_bias = self.bias(&format!("blk.{i}.ffn_down.bias"), down_b, hidden)?;
            Ok(HexagonFfn::Dense(HexagonDenseFfn {
                gate,
                up,
                down,
                gate_bias,
                up_bias,
                down_bias,
            }))
        }
    }
}

/// Running plan of `kv_state_buf`: per-layer KV slabs and recurrent states.
pub(super) struct KvPlanner {
    pub(super) total: usize,
    pub(super) kv_dtype: HtpDataType,
    pub(super) max_seq_len: usize,
}

impl KvPlanner {
    pub(super) fn new(kv_dtype: HtpDataType, max_seq_len: usize) -> Self {
        Self {
            total: 0,
            kv_dtype,
            max_seq_len,
        }
    }

    /// K and V slabs of one attention layer: `(k_offset, v_offset)`.
    pub(super) fn attention(&mut self, kv_dim: usize) -> (usize, usize) {
        let slab = kv_slab_bytes(self.kv_dtype, self.max_seq_len, kv_dim);
        let k_offset = align256(self.total);
        let v_offset = align256(k_offset + slab);
        self.total = v_offset + slab;
        (k_offset, v_offset)
    }

    /// One channel-interleaved `[C, 2]` short-conv state slab (see
    /// `HexagonConvLayer::state_offset`); `state_size` is one `[C]` plane.
    pub(super) fn conv(&mut self, state_size: usize) -> usize {
        let offset = align256(self.total);
        self.total = align256(offset + 2 * state_size);
        offset
    }

    /// Conv and SSM states of one DeltaNet layer: `(conv_offset, ssm_offset)`.
    pub(super) fn deltanet(
        &mut self,
        conv_dim: usize,
        d_conv: usize,
        dt_rank: usize,
        d_state: usize,
    ) -> (usize, usize) {
        let conv_size = align256(conv_dim * d_conv.saturating_sub(1) * 4);
        let ssm_size = align256(dt_rank * d_state * d_state * 4);
        let conv_offset = align256(self.total);
        let ssm_offset = align256(conv_offset + conv_size);
        self.total = align256(ssm_offset + ssm_size);
        (conv_offset, ssm_offset)
    }
}

/// Largest per-layer KV width of `config`.
pub(super) fn max_kv_dim(config: &ModelConfig, head_dim: usize) -> usize {
    config
        .kv_heads_per_layer
        .iter()
        .map(|&h| h * head_dim)
        .max()
        .unwrap_or(head_dim)
}

/// Everything the shared copy pass needs: the planner's totals and copy list.
pub(super) struct WeightCopy<'a> {
    pub(super) src: &'a dyn TensorSource,
    pub(super) weights_total: usize,
    pub(super) kv_total: usize,
    pub(super) copies: &'a [CopyOp],
}

impl WeightCopy<'_> {
    /// Allocate `weights_buf` and `kv_state_buf`, run every planned copy from
    /// `src` into the former, and flush it.
    pub(super) fn run(
        &self,
        driver: &Arc<crate::backend::hexagon::FastRpcDriver>,
    ) -> Result<(RpcmemBuffer, RpcmemBuffer), CeraError> {
        let mut buf = RpcmemBuffer::alloc(Arc::clone(driver), self.weights_total, true)?;
        let kv_state_buf = alloc_zeroed_state(driver, self.kv_total)?;
        let src = self.src;

        for op in self.copies {
            match op {
                CopyOp::Weight { name, offset } => src.copy_weight(name, *offset, &mut buf)?,
                CopyOp::Vector { name, offset, len } => {
                    let v = src.vector(name)?;
                    // `<=`: LFM2 q/k norm slots are q_dim / kv_dim wide while
                    // the tensor is head_dim long. Longer would spill into the
                    // next tensor's region.
                    if v.len() > *len {
                        return Err(CeraError::Backend(format!(
                            "{name} has {} elements, more than its {len}-element slot",
                            v.len()
                        )));
                    }
                    copy_f32_into(&v, name, *offset, &mut buf)?;
                }
                CopyOp::Stacked { name, weight } => src.copy_stacked(name, weight, &mut buf)?,
                CopyOp::ConvTaps {
                    name,
                    hidden_size,
                    taps,
                    ssm,
                } => {
                    let hidden_size = *hidden_size;
                    let conv = src.vector(name)?;
                    let expected = hidden_size.checked_mul(3).ok_or_else(|| {
                        CeraError::Backend(
                            "hidden_size overflow calculating conv weight size".into(),
                        )
                    })?;
                    if conv.len() != expected {
                        return Err(CeraError::Backend(format!(
                            "{name} size {} != hidden_size * 3 ({hidden_size} * 3)",
                            conv.len()
                        )));
                    }
                    // Per-tap planes for the manual decode chain; the SsmConv
                    // taps use the GGUF `[3, C]` layout (already oldest-first
                    // per channel) verbatim.
                    for (tap, offset) in taps.iter().enumerate() {
                        let plane: Vec<f32> =
                            (0..hidden_size).map(|ch| conv[ch * 3 + tap]).collect();
                        copy_f32_into(&plane, name, *offset, &mut buf)?;
                    }
                    copy_f32_into(&conv, name, *ssm, &mut buf)?;
                }
                CopyOp::Constant {
                    label,
                    offset,
                    values,
                } => copy_f32_into(values, label, *offset, &mut buf)?,
            }
        }

        buf.flush_cpu_cache(0, self.weights_total);
        Ok((buf, kv_state_buf))
    }
}

/// [`TensorSource`] over a GGUF file (the LFM2 / LFM2-MoE loader).
pub(super) struct GgufSource<'a> {
    pub(super) gguf: &'a GgufFile,
}

impl TensorSource for GgufSource<'_> {
    fn weight_shape(&self, name: &str) -> Result<(crate::tensor::DType, usize, usize), CeraError> {
        let t = self
            .gguf
            .tensors
            .get(name)
            .ok_or_else(|| CeraError::Backend(format!("missing tensor {name}")))?;
        let rows = t
            .shape
            .first()
            .copied()
            .ok_or_else(|| CeraError::Backend(format!("tensor {name} has no dimensions")))?;
        Ok((t.dtype, rows, t.shape.get(1).copied().unwrap_or(1)))
    }

    fn stacked_slice_plan(
        &self,
        name: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<SlicePlan, CeraError> {
        let t = self
            .gguf
            .tensors
            .get(name)
            .ok_or_else(|| CeraError::Backend(format!("missing tensor {name}")))?;
        match t.dtype {
            crate::tensor::DType::Q4_0 => Ok((
                repacked_matrix_size_q4_0(in_dim, out_dim)?,
                HtpDataType::Q4_0,
                18,
                TILE_SIZE_Q4_0,
            )),
            crate::tensor::DType::Q8_0 => Ok((
                repacked_matrix_size_q8_0(in_dim, out_dim)?,
                HtpDataType::Q8_0,
                34,
                TILE_SIZE_Q8_0,
            )),
            other => Err(CeraError::Backend(format!(
                "unsupported quant format {other:?} for Hexagon stacked weight {name}"
            ))),
        }
    }

    fn vector(&self, name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CeraError> {
        let t = self
            .gguf
            .get_tensor(name)
            .map_err(|e| CeraError::Backend(format!("missing tensor {name}: {e}")))?;
        Ok(std::borrow::Cow::Owned(t.to_f32_vec()))
    }

    fn copy_weight(
        &self,
        name: &str,
        offset: usize,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError> {
        let (dtype, ne0, ne1) = self.weight_shape(name)?;
        let raw = self
            .gguf
            .tensor_data(name)
            .map_err(|e| CeraError::Backend(e.to_string()))?;
        repack_weight(
            dtype,
            raw,
            ne0,
            ne1,
            &mut buf.as_mut_slice()[offset..],
            name,
        )
    }

    fn copy_stacked(
        &self,
        name: &str,
        sw: &HexagonStackedWeight,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError> {
        let gguf = self.gguf;
        for e in 0..sw.n_expert {
            // `tensor_meta_expert` returns (start, bytes, rows, cols, dtype)
            // with rows = out_dim and cols = in_dim; the repackers take
            // ne0 = in_dim (K) and ne1 = out_dim (rows).
            let (raw_start, raw_size, rows, cols, dtype) = gguf
                .tensor_meta_expert(name, e)
                .map_err(|err| CeraError::Backend(format!("{name} expert {e}: {err}")))?;
            let dst_range = stacked_expert_dst_range(name, e, sw, rows, cols, dtype)?;
            if dst_range.end > buf.size() {
                return Err(CeraError::Backend(format!(
                    "{name} expert {e}: destination range {dst_range:?} exceeds weights buffer ({} bytes)",
                    buf.size()
                )));
            }
            let raw_end = raw_start.checked_add(raw_size).ok_or_else(|| {
                CeraError::Backend(format!("{name} expert {e}: source range overflows"))
            })?;
            let raw_data = gguf.mmap_data().get(raw_start..raw_end).ok_or_else(|| {
                CeraError::Backend(format!(
                    "{name} expert {e}: source range {raw_start}..{raw_end} is past the mapped file"
                ))
            })?;
            let dst_slice = &mut buf.as_mut_slice()[dst_range];
            match dtype {
                crate::tensor::DType::Q4_0 => {
                    repack_q4_0(raw_data, cols, rows, dst_slice)
                        .map_err(|err| CeraError::Backend(format!("{name} expert {e}: {err}")))?;
                }
                crate::tensor::DType::Q8_0 => {
                    repack_q8_0(raw_data, cols, rows, dst_slice)
                        .map_err(|err| CeraError::Backend(format!("{name} expert {e}: {err}")))?;
                }
                other => {
                    return Err(CeraError::Backend(format!(
                        "unsupported quant format {other:?} for Hexagon stacked weight {name}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// [`TensorSource`] over a loaded Qwen 3.5 model's weight refs.
pub(super) struct Qwen35Source<'a> {
    pub(super) cpu: &'a crate::model::qwen35::Qwen35Model,
}

impl Qwen35Source<'_> {
    pub(super) fn wref(&self, name: &str) -> Result<&WeightRef, CeraError> {
        use crate::model::qwen35::LayerKindRefs;
        match name {
            "output.weight" => {
                return self
                    .cpu
                    .output_ref
                    .as_ref()
                    .ok_or_else(|| unknown_tensor(name));
            }
            "token_embd.weight" => return Ok(&self.cpu.embd_ref),
            _ => {}
        }
        let (i, rest) = split_layer_tensor(name).ok_or_else(|| unknown_tensor(name))?;
        let layer = self.cpu.layers.get(i).ok_or_else(|| unknown_tensor(name))?;
        let found = match (rest, &layer.kind) {
            ("ffn_gate.weight", _) => Some(&layer.ffn_gate),
            ("ffn_up.weight", _) => Some(&layer.ffn_up),
            ("ffn_down.weight", _) => Some(&layer.ffn_down),
            ("attn_q.weight", LayerKindRefs::Attention(a)) => Some(&a.attn_q),
            ("attn_k.weight", LayerKindRefs::Attention(a)) => Some(&a.attn_k),
            ("attn_v.weight", LayerKindRefs::Attention(a)) => Some(&a.attn_v),
            ("attn_output.weight", LayerKindRefs::Attention(a)) => Some(&a.attn_output),
            ("attn_qkv.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.wqkv),
            ("attn_gate.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.wqkv_gate),
            ("ssm_beta.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_beta),
            ("ssm_alpha.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_alpha),
            ("ssm_out.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_out),
            _ => None,
        };
        found.ok_or_else(|| unknown_tensor(name))
    }
}

impl TensorSource for Qwen35Source<'_> {
    fn weight_shape(&self, name: &str) -> Result<(crate::tensor::DType, usize, usize), CeraError> {
        let w = self.wref(name)?;
        Ok((w.dtype, w.k, w.m))
    }

    fn stacked_slice_plan(&self, name: &str, _: usize, _: usize) -> Result<SlicePlan, CeraError> {
        Err(unknown_tensor(name))
    }

    fn vector(&self, name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CeraError> {
        use crate::model::qwen35::LayerKindRefs;
        if name == "output_norm.weight" {
            return Ok(std::borrow::Cow::Borrowed(&self.cpu.output_norm_weight));
        }
        let (i, rest) = split_layer_tensor(name).ok_or_else(|| unknown_tensor(name))?;
        let layer = self.cpu.layers.get(i).ok_or_else(|| unknown_tensor(name))?;
        let found: Option<&[f32]> = match (rest, &layer.kind) {
            ("attn_norm.weight", _) => Some(&layer.attn_norm),
            // Qwen 3.5 names its pre-FFN norm `attn_post_norm`.
            ("ffn_norm.weight", _) => Some(&layer.attn_post_norm),
            ("attn_q_norm.weight", LayerKindRefs::Attention(a)) => Some(&a.attn_q_norm),
            ("attn_k_norm.weight", LayerKindRefs::Attention(a)) => Some(&a.attn_k_norm),
            ("ssm_conv1d.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_conv1d),
            ("ssm_conv1d.bias", LayerKindRefs::DeltaNet(d)) => d.ssm_conv1d_bias.as_deref(),
            ("ssm_dt.bias", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_dt),
            ("ssm_a", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_a),
            ("ssm_norm.weight", LayerKindRefs::DeltaNet(d)) => Some(&d.ssm_norm),
            _ => None,
        };
        found
            .map(std::borrow::Cow::Borrowed)
            .ok_or_else(|| unknown_tensor(name))
    }

    fn copy_weight(
        &self,
        name: &str,
        offset: usize,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError> {
        let w = self.wref(name)?;
        let raw = self.cpu.weight_bytes(w);
        repack_weight(
            w.dtype,
            raw,
            w.k,
            w.m,
            &mut buf.as_mut_slice()[offset..],
            name,
        )
    }

    fn copy_stacked(
        &self,
        name: &str,
        _: &HexagonStackedWeight,
        _: &mut RpcmemBuffer,
    ) -> Result<(), CeraError> {
        Err(unknown_tensor(name))
    }
}

/// [`TensorSource`] over a dense-transformer [`GpuWeightSource`] plus its
/// [`DenseExtras`] biases.
pub(super) struct DenseSource<'a> {
    pub(super) src: &'a dyn GpuWeightSource,
    pub(super) extras: &'a DenseExtras,
}

impl DenseSource<'_> {
    pub(super) fn wref(&self, name: &str) -> Result<std::borrow::Cow<'_, WeightRef>, CeraError> {
        use std::borrow::Cow;
        let src = self.src;
        match name {
            "output.weight" => {
                return src
                    .output_ref()
                    .map(Cow::Borrowed)
                    .ok_or_else(|| unknown_tensor(name));
            }
            "token_embd.weight" => {
                return crate::model::transformer::resolve_weight(src.gguf(), name)
                    .map(Cow::Owned)
                    .map_err(|e| {
                        CeraError::Backend(format!(
                            "missing token_embd.weight for tied LM head: {e}"
                        ))
                    });
            }
            _ => {}
        }
        let (i, rest) = split_layer_tensor(name).ok_or_else(|| unknown_tensor(name))?;
        let missing = |what: &str| CeraError::Backend(format!("missing {what} ref for layer {i}"));
        fn ffn_ref<'r>(
            r: anyhow::Result<&'r WeightRef>,
            what: &str,
            i: usize,
        ) -> Result<std::borrow::Cow<'r, WeightRef>, CeraError> {
            r.map(std::borrow::Cow::Borrowed)
                .map_err(|e| CeraError::Backend(format!("missing {what} ref for layer {i}: {e}")))
        }
        match rest {
            "attn_q.weight" => src
                .attn_q_ref(i)
                .map(Cow::Borrowed)
                .ok_or_else(|| missing("attn_q")),
            "attn_k.weight" => src
                .attn_k_ref(i)
                .map(Cow::Borrowed)
                .ok_or_else(|| missing("attn_k")),
            "attn_v.weight" => src
                .attn_v_ref(i)
                .map(Cow::Borrowed)
                .ok_or_else(|| missing("attn_v")),
            "attn_output.weight" => src
                .attn_output_ref(i)
                .map(Cow::Borrowed)
                .ok_or_else(|| missing("attn_output")),
            "ffn_gate.weight" => ffn_ref(src.ffn_gate_ref(i), "ffn_gate", i),
            "ffn_up.weight" => ffn_ref(src.ffn_up_ref(i), "ffn_up", i),
            "ffn_down.weight" => ffn_ref(src.ffn_down_ref(i), "ffn_down", i),
            "ffn_gate_inp.weight" => src
                .moe_refs(i)
                .map(|m| Cow::Borrowed(&m.router))
                .ok_or_else(|| missing("moe")),
            _ => Err(unknown_tensor(name)),
        }
    }

    /// Expert weight refs of the stacked tensor `name` (`blk.{i}.ffn_*_exps.weight`).
    fn expert_refs(&self, name: &str) -> Result<&[WeightRef], CeraError> {
        let (i, rest) = split_layer_tensor(name).ok_or_else(|| unknown_tensor(name))?;
        let moe = self
            .src
            .moe_refs(i)
            .ok_or_else(|| CeraError::Backend(format!("missing moe_refs for layer {i}")))?;
        match rest {
            "ffn_gate_exps.weight" => Ok(&moe.gate),
            "ffn_up_exps.weight" => Ok(&moe.up),
            "ffn_down_exps.weight" => Ok(&moe.down),
            _ => Err(unknown_tensor(name)),
        }
    }
}

impl TensorSource for DenseSource<'_> {
    fn weight_shape(&self, name: &str) -> Result<(crate::tensor::DType, usize, usize), CeraError> {
        let w = self.wref(name)?;
        Ok((w.dtype, w.k, w.m))
    }

    fn stacked_slice_plan(&self, name: &str, _: usize, _: usize) -> Result<SlicePlan, CeraError> {
        let first = self
            .expert_refs(name)?
            .first()
            .ok_or_else(|| CeraError::Backend(format!("{name} has no experts")))?;
        wref_weight_plan(first, &format!("{name}.0"))
    }

    fn vector(&self, name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CeraError> {
        use std::borrow::Cow;
        let src = self.src;
        if name == "output_norm.weight" {
            return Ok(Cow::Borrowed(src.output_norm_weight()));
        }
        let (i, rest) = split_layer_tensor(name).ok_or_else(|| unknown_tensor(name))?;
        let extras = self.extras;
        let found: Option<&[f32]> = match rest {
            "attn_norm.weight" => Some(src.attn_norm_weight(i)),
            "ffn_norm.weight" => Some(src.ffn_norm_weight(i)),
            "attn_q_norm.weight" => src.attn_q_norm_weight(i),
            "attn_k_norm.weight" => src.attn_k_norm_weight(i),
            "attn_post_norm.weight" => src.attn_post_norm_weight(i),
            "ffn_post_norm.weight" => src.ffn_post_norm_weight(i),
            "attn_q.bias" => src.attn_q_bias(i),
            "attn_k.bias" => src.attn_k_bias(i),
            "attn_v.bias" => src.attn_v_bias(i),
            "attn_output.bias" => extras.attn_output_bias.get(i).and_then(|b| b.as_deref()),
            "ffn_gate.bias" => extras.ffn_gate_bias.get(i).and_then(|b| b.as_deref()),
            "ffn_up.bias" => extras.ffn_up_bias.get(i).and_then(|b| b.as_deref()),
            "ffn_down.bias" => extras.ffn_down_bias.get(i).and_then(|b| b.as_deref()),
            "exp_probs_b.bias" => src.moe_refs(i).map(|m| m.exp_probs_b.as_slice()),
            _ => None,
        };
        found.map(Cow::Borrowed).ok_or_else(|| unknown_tensor(name))
    }

    fn copy_weight(
        &self,
        name: &str,
        offset: usize,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError> {
        let w = self.wref(name)?;
        let raw = self.src.weight_bytes(&w);
        repack_weight(
            w.dtype,
            &raw,
            w.k,
            w.m,
            &mut buf.as_mut_slice()[offset..],
            name,
        )
    }

    fn copy_stacked(
        &self,
        name: &str,
        sw: &HexagonStackedWeight,
        buf: &mut RpcmemBuffer,
    ) -> Result<(), CeraError> {
        let refs = self.expert_refs(name)?;
        for e in 0..sw.n_expert {
            let w = refs.get(e).ok_or_else(|| {
                CeraError::Backend(format!(
                    "{name} has {} experts, expected {}",
                    refs.len(),
                    sw.n_expert
                ))
            })?;
            let raw = self.src.weight_bytes(w);
            let offset = sw.offset + e * sw.expert_stride;
            repack_weight(
                w.dtype,
                &raw,
                w.k,
                w.m,
                &mut buf.as_mut_slice()[offset..],
                &format!("{name}.{e}"),
            )?;
        }
        Ok(())
    }
}

/// Reserve `size` bytes in a 256-aligned running total, returning the offset.
pub(super) fn plan_offset(total: &mut usize, size: usize) -> usize {
    let offset = (*total + 255) & !255;
    *total = offset + size;
    offset
}

/// Destination byte range of expert `e` inside the weights buffer, after
/// checking the tensor's per-expert shape and dtype against the plan that
/// sized `sw`. `rows` is the GGUF ne1 (out_dim) and `cols` the GGUF ne0
/// (in_dim), the order `GgufFile::tensor_meta_expert` reports them in.
fn stacked_expert_dst_range(
    name: &str,
    e: usize,
    sw: &HexagonStackedWeight,
    rows: usize,
    cols: usize,
    dtype: crate::tensor::DType,
) -> Result<std::ops::Range<usize>, CeraError> {
    if cols != sw.in_dim || rows != sw.out_dim {
        return Err(CeraError::Backend(format!(
            "{name} expert {e}: tensor shape (in {cols}, out {rows}) does not match the planned (in {}, out {})",
            sw.in_dim, sw.out_dim
        )));
    }
    let planned = match dtype {
        crate::tensor::DType::Q4_0 => Some(HtpDataType::Q4_0),
        crate::tensor::DType::Q8_0 => Some(HtpDataType::Q8_0),
        _ => None,
    };
    if planned != Some(sw.wire_dtype) {
        return Err(CeraError::Backend(format!(
            "{name} expert {e}: dtype {dtype:?} does not match the planned wire dtype {:?}",
            sw.wire_dtype
        )));
    }
    let start = e
        .checked_mul(sw.expert_stride)
        .and_then(|off| sw.offset.checked_add(off))
        .ok_or_else(|| CeraError::Backend(format!("{name} expert {e}: offset overflows")))?;
    let end = start
        .checked_add(sw.expert_stride)
        .ok_or_else(|| CeraError::Backend(format!("{name} expert {e}: end offset overflows")))?;
    let limit = sw.offset.saturating_add(sw.size);
    if end > limit {
        return Err(CeraError::Backend(format!(
            "{name} expert {e}: range {start}..{end} exceeds the planned stacked region ending at {limit}"
        )));
    }
    Ok(start..end)
}

/// Wire layout planned for one 2D weight matrix: `(repacked byte size, wire
/// dtype, nb0 block bytes, tile size)`. Q5_K and F32 have no wire format and
/// plan Q8_0 bytes; [`repack_weight`] requantizes them before the repack.
///
/// Q4_1 rides the `Q4K` wire type (the 640-byte tile layout Q4_1 and Q4_K
/// share) but keeps its own 20-byte GGML block size in `nb0`, while real Q4_K
/// carries 144. The DSP is not known to read `nb0` for repacked weights, so
/// the two coexist; if a Q4_1 model misbehaves on device, `block_bytes` for
/// Q4_1 (20 vs 144) is the first thing to try. Pinned by
/// `test_wire_plan_formats`; needs validation on the S25 Ultra with a Q4_1 model.
pub(super) fn wire_plan(
    dtype: crate::tensor::DType,
    ne0: usize,
    ne1: usize,
    name: &str,
) -> Result<SlicePlan, CeraError> {
    use crate::tensor::DType;
    Ok(match dtype {
        DType::Q8_0 => (
            repacked_matrix_size_q8_0(ne0, ne1)?,
            HtpDataType::Q8_0,
            34,
            TILE_SIZE_Q8_0,
        ),
        DType::Q4_0 => (
            repacked_matrix_size_q4_0(ne0, ne1)?,
            HtpDataType::Q4_0,
            18,
            TILE_SIZE_Q4_0,
        ),
        DType::Q4_1 => (
            repacked_matrix_size_q4_1(ne0, ne1)?,
            HtpDataType::Q4K,
            20,
            TILE_SIZE_Q4_K,
        ),
        DType::Q4KM => (
            repacked_matrix_size_q4_k(ne0, ne1)?,
            HtpDataType::Q4K,
            144,
            TILE_SIZE_Q4_K,
        ),
        DType::Q6K => (
            repacked_matrix_size_q6_k(ne0, ne1)?,
            HtpDataType::Q6K,
            210,
            TILE_SIZE_Q6_K,
        ),
        DType::Q5KM | DType::F32 => (
            repacked_matrix_size_q8_0(ne0, ne1)?,
            HtpDataType::Q8_0,
            34,
            TILE_SIZE_Q8_0,
        ),
        other => {
            return Err(CeraError::Backend(format!(
                "unsupported quant format {other:?} for Hexagon weight {name}"
            )));
        }
    })
}

/// `(repacked byte size, wire dtype, block bytes, tile size)` of a weight ref.
fn wref_weight_plan(wref: &WeightRef, name: &str) -> Result<SlicePlan, CeraError> {
    wire_plan(wref.dtype, wref.k, wref.m, name)
}

/// Reserve one weight matrix (`ne0` = K, `ne1` = rows of the stored tensor) in
/// the weights buffer and describe its wire format. `in_dim` / `out_dim` are the
/// matmul dims the forward pass uses.
fn plan_weight(
    total: &mut usize,
    dtype: crate::tensor::DType,
    ne0: usize,
    ne1: usize,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<HexagonWeight, CeraError> {
    let (size, wire_dtype, block_bytes, tile_size) = wire_plan(dtype, ne0, ne1, name)?;
    let offset = plan_offset(total, size);
    Ok(HexagonWeight {
        offset,
        in_dim,
        out_dim,
        wire_dtype,
        block_bytes,
        tile_size,
    })
}

/// Reserve `n_expert` stacked expert slices, each `slice_plan.0` bytes
/// (256-aligned stride), with the checked total the copy step bounds against.
fn plan_stacked(
    total: &mut usize,
    slice_plan: SlicePlan,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    n_expert: usize,
) -> Result<HexagonStackedWeight, CeraError> {
    let (slice_size, wire_dtype, block_bytes, tile_size) = slice_plan;
    let expert_stride = align256(slice_size);
    let total_bytes = n_expert.checked_mul(expert_stride).ok_or_else(|| {
        CeraError::Backend(format!(
            "stacked weight {name}: n_expert ({n_expert}) * expert_stride ({expert_stride}) overflows usize"
        ))
    })?;
    let offset = plan_offset(total, total_bytes);
    Ok(HexagonStackedWeight {
        offset,
        size: total_bytes,
        in_dim,
        out_dim,
        expert_stride,
        n_expert,
        wire_dtype,
        tile_size,
        block_bytes,
    })
}

/// Repack `raw` (`dtype` blocks, GGUF row-major, `ne0` = K, `ne1` = rows) into
/// the tiled wire layout planned by [`wire_plan`].
fn repack_weight(
    dtype: crate::tensor::DType,
    raw: &[u8],
    ne0: usize,
    ne1: usize,
    dst: &mut [u8],
    name: &str,
) -> Result<(), CeraError> {
    use crate::tensor::DType;
    let ctx = |e: CeraError| CeraError::Backend(format!("{name}: {e}"));
    match dtype {
        DType::Q8_0 => repack_q8_0(raw, ne0, ne1, dst).map_err(ctx),
        DType::Q4_0 => repack_q4_0(raw, ne0, ne1, dst).map_err(ctx),
        DType::Q4_1 => repack_q4_1(raw, ne0, ne1, dst).map_err(ctx),
        DType::Q4KM => repack_q4_k(raw, ne0, ne1, dst).map_err(ctx),
        DType::Q6K => repack_q6_k(raw, ne0, ne1, dst).map_err(ctx),
        DType::Q5KM => {
            let q8 = requant_q5_k_to_q8_0(raw, ne0, ne1).map_err(ctx)?;
            repack_q8_0(&q8, ne0, ne1, dst).map_err(ctx)
        }
        DType::F32 => {
            warn_f32_requant(name);
            // The mmap slice carries no alignment guarantee, so decode by
            // value instead of casting.
            let f32_vals: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let q8 = quantize_f32_to_q8_0(&f32_vals, ne0, ne1).map_err(ctx)?;
            repack_q8_0(&q8, ne0, ne1, dst).map_err(ctx)
        }
        other => Err(CeraError::Backend(format!(
            "unsupported quant format {other:?} for Hexagon weight {name}"
        ))),
    }
}

/// 256-byte alignment: the DMA / HVX buffer granule of every planned region.
pub(super) const fn align256(x: usize) -> usize {
    (x + 255) & !255
}

/// Bytes of one K or V cache slab (`max_seq_len` positions of `kv_dim` values in
/// the cache wire type), 256-aligned.
fn kv_slab_bytes(kv_dtype: HtpDataType, max_seq_len: usize, kv_dim: usize) -> usize {
    align256(if kv_dtype == HtpDataType::Q8_0 {
        max_seq_len * kv_dim.div_ceil(32) * 34
    } else {
        max_seq_len * kv_dim * 2
    })
}

/// A zero-initialized shared buffer for KV cache and recurrent state.
pub(super) fn alloc_zeroed_state(
    driver: &Arc<crate::backend::hexagon::FastRpcDriver>,
    size: usize,
) -> Result<RpcmemBuffer, CeraError> {
    let buf = RpcmemBuffer::alloc(Arc::clone(driver), size, true)?;
    unsafe {
        std::ptr::write_bytes(buf.as_mut_ptr(), 0, size);
    }
    buf.flush_cpu_cache(0, size);
    Ok(buf)
}

/// Copy an f32 slice into `buf` at `offset`, bounds-checked.
fn copy_f32_into(
    slice: &[f32],
    name: &str,
    offset: usize,
    buf: &mut RpcmemBuffer,
) -> Result<(), CeraError> {
    let byte_size = std::mem::size_of_val(slice);
    if offset.saturating_add(byte_size) > buf.size() {
        return Err(CeraError::Backend(format!(
            "tensor {name} byte size ({byte_size}) exceeds weights buffer capacity at offset {offset}"
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            slice.as_ptr() as *const u8,
            buf.as_mut_ptr().add(offset),
            byte_size,
        );
    }
    Ok(())
}

/// F32 matrices (the lfm2moe router `ffn_gate_inp`, Qwen 3.5 `ssm_alpha` /
/// `ssm_beta`) are requantized to Q8_0 at load. The DSP weight wire formats are
/// the tiled quantized layouts (`Q8_0`, `Q4_0`, `Q4K`, `Q6K`); there is no F32
/// matmul weight layout in the host driver or the firmware contract, so an F32
/// weight cannot be carried without a new DSP kernel. The cost is Q8_0 rounding
/// on these small matrices (router probabilities shift by about 1/127 relative
/// per block, which can flip a near-tie top-k choice), so warn once.
fn warn_f32_requant(name: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        hexagon_warn!(
            "no F32 matmul weight format; requantizing F32 tensor {name} (and any other F32 weight) to Q8_0 at load time"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::hexagon_lfm2::EmbeddingTable;
    use crate::model::hexagon_lfm2::test_support::embd_gguf;

    #[test]
    fn test_stacked_weight_stride_and_alignment() {
        let pad32 = |d: usize| d.div_ceil(32) * 32;
        let in_dim = 2048;
        let out_dim = 1792;
        let n_expert = 32;
        let pad_k = pad32(in_dim);
        let pad_m = pad32(out_dim);
        let block_bytes = 18; // Q4_0
        let tile_size = 32 * block_bytes;
        let tiled_row_bytes = (pad_k / 32) * tile_size;
        let expert_stride = (pad_m / 32) * tiled_row_bytes;
        let total_size = n_expert * expert_stride;

        // Verify expert stride is 256-byte aligned for HTP DMA / HVX
        assert_eq!(expert_stride % 256, 0);
        assert_eq!(total_size % 256, 0);
        assert_eq!(expert_stride, (out_dim / 32) * (in_dim / 32) * 32 * 18);
    }

    fn stacked_fixture() -> HexagonStackedWeight {
        HexagonStackedWeight {
            offset: 1024,
            size: 3 * 512,
            in_dim: 64,
            out_dim: 32,
            expert_stride: 512,
            n_expert: 3,
            wire_dtype: HtpDataType::Q4_0,
            tile_size: TILE_SIZE_Q4_0,
            block_bytes: 18,
        }
    }

    #[test]
    fn test_stacked_expert_dst_range_bounds() {
        use crate::tensor::DType;
        let sw = stacked_fixture();
        let r = stacked_expert_dst_range("w", 0, &sw, 32, 64, DType::Q4_0).unwrap();
        assert_eq!(r, 1024..1536);
        let r = stacked_expert_dst_range("w", 2, &sw, 32, 64, DType::Q4_0).unwrap();
        assert_eq!(r, 2048..2560);
        // Expert past the planned region.
        let err = stacked_expert_dst_range("w", 3, &sw, 32, 64, DType::Q4_0).unwrap_err();
        assert!(err.to_string().contains("exceeds the planned"), "{err}");
    }

    #[test]
    fn test_stacked_expert_dst_range_rejects_shape_and_dtype_mismatch() {
        use crate::tensor::DType;
        let sw = stacked_fixture();
        // Swapped (rows, cols): the historical caller bug.
        let err = stacked_expert_dst_range("w", 0, &sw, 64, 32, DType::Q4_0).unwrap_err();
        assert!(
            err.to_string().contains("does not match the planned"),
            "{err}"
        );
        let err = stacked_expert_dst_range("w", 0, &sw, 32, 96, DType::Q4_0).unwrap_err();
        assert!(
            err.to_string().contains("does not match the planned"),
            "{err}"
        );
        let err = stacked_expert_dst_range("w", 0, &sw, 32, 64, DType::Q8_0).unwrap_err();
        assert!(err.to_string().contains("wire dtype"), "{err}");
        let err = stacked_expert_dst_range("w", 0, &sw, 32, 64, DType::F16).unwrap_err();
        assert!(err.to_string().contains("wire dtype"), "{err}");
    }

    /// Row lookup equals the row of the fully dequantized table the model used
    /// to keep resident, for every embedding dtype, with and without a multiplier.
    #[test]
    fn test_embedding_table_rows_match_full_dequantization() {
        use crate::tensor::DType;
        let (k, rows) = (64usize, 5usize);
        let pattern =
            |n: usize| -> Vec<u8> { (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect() };
        // F16 scale words must stay finite: force the exponent byte of every
        // block scale to a small normal value.
        let quant_bytes = |block_bytes: usize, n_blocks: usize| -> Vec<u8> {
            let mut b = pattern(block_bytes * n_blocks);
            for blk in 0..n_blocks {
                b[blk * block_bytes] = 0x00;
                b[blk * block_bytes + 1] = 0x2C;
            }
            b
        };
        let cases: [(u32, DType, Vec<u8>); 4] = [
            (0, DType::F32, {
                (0..k * rows)
                    .flat_map(|i| (0.01 * i as f32 - 1.5).to_le_bytes())
                    .collect()
            }),
            (1, DType::F16, {
                (0..k * rows)
                    .flat_map(|i| crate::quant::f32_to_f16(0.01 * i as f32 - 1.5).to_le_bytes())
                    .collect()
            }),
            (2, DType::Q4_0, quant_bytes(18, k / 32 * rows)),
            (8, DType::Q8_0, quant_bytes(34, k / 32 * rows)),
        ];
        for (ggml_type, dtype, data) in cases {
            let gguf = embd_gguf(ggml_type, k, rows, &data);
            let full = gguf.get_tensor("token_embd.weight").unwrap().to_f32_vec();
            assert_eq!(full.len(), k * rows, "{dtype:?}");
            for scale in [1.0f32, 2.5] {
                let table = EmbeddingTable::new(&gguf, rows, k, scale).unwrap();
                for token in 0..rows {
                    let mut row = vec![0.0f32; k];
                    table.row_into(token, &mut row);
                    let want: Vec<f32> = full[token * k..(token + 1) * k]
                        .iter()
                        .map(|x| x * scale)
                        .collect();
                    assert_eq!(row, want, "{dtype:?} token {token} scale {scale}");
                }
            }
        }
    }

    /// The table keeps only the mapping: a borrowed GGUF is stripped of its
    /// metadata and tensor tables, a shared one is not copied at all, and both
    /// still read every row.
    #[test]
    fn test_embedding_table_holds_only_the_mapping() {
        let (k, rows) = (32usize, 4usize);
        let data: Vec<u8> = (0..k * rows)
            .flat_map(|i| (0.5 * i as f32).to_le_bytes())
            .collect();
        let gguf = embd_gguf(0, k, rows, &data);
        assert!(!gguf.tensors.is_empty());
        let borrowed = EmbeddingTable::new(&gguf, rows, k, 1.0).unwrap();
        assert!(borrowed.gguf.tensors.is_empty() && borrowed.gguf.metadata.is_empty());

        let shared = Arc::new(gguf);
        let table = EmbeddingTable::shared(&shared, rows, k, 1.0).unwrap();
        assert!(Arc::ptr_eq(&table.gguf, &shared), "shared must not copy");
        for t in [&borrowed, &table] {
            let mut row = vec![0.0f32; k];
            t.row_into(rows - 1, &mut row);
            let want: Vec<f32> = ((rows - 1) * k..rows * k).map(|i| 0.5 * i as f32).collect();
            assert_eq!(row, want);
        }
    }

    #[test]
    fn test_embedding_table_rejects_bad_shapes() {
        let (k, rows) = (32usize, 4usize);
        let data: Vec<u8> = (0..k * rows)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let gguf = embd_gguf(0, k, rows, &data);
        // Hidden size mismatch.
        assert!(EmbeddingTable::new(&gguf, rows, k * 2, 1.0).is_err());
        // Vocabulary larger than the stored rows.
        assert!(EmbeddingTable::new(&gguf, rows + 1, k, 1.0).is_err());
        // Vocabulary smaller than the rows (padded table) is fine.
        assert!(EmbeddingTable::new(&gguf, rows - 1, k, 1.0).is_ok());
    }

    /// Wire layout per dtype, pinned: a change here changes what the DSP reads.
    /// Q4_1 (`Q4K` wire type, 20-byte nb0) is flagged for S25 Ultra validation.
    #[test]
    fn test_wire_plan_formats() {
        use crate::tensor::DType;
        let plan = |d| wire_plan(d, 64, 32, "w").unwrap();
        let (sz, dt, bb, tile) = plan(DType::Q8_0);
        assert_eq!((dt, bb, tile), (HtpDataType::Q8_0, 34, TILE_SIZE_Q8_0));
        assert_eq!(sz, repacked_matrix_size_q8_0(64, 32).unwrap());
        let (sz, dt, bb, tile) = plan(DType::Q4_0);
        assert_eq!((dt, bb, tile), (HtpDataType::Q4_0, 18, TILE_SIZE_Q4_0));
        assert_eq!(sz, repacked_matrix_size_q4_0(64, 32).unwrap());
        let (sz, dt, bb, tile) = plan(DType::Q4_1);
        assert_eq!((dt, bb, tile), (HtpDataType::Q4K, 20, TILE_SIZE_Q4_K));
        assert_eq!(sz, repacked_matrix_size_q4_1(64, 32).unwrap());
        let (_, dt, bb, tile) = plan(DType::Q4KM);
        assert_eq!((dt, bb, tile), (HtpDataType::Q4K, 144, TILE_SIZE_Q4_K));
        let (_, dt, bb, tile) = plan(DType::Q6K);
        assert_eq!((dt, bb, tile), (HtpDataType::Q6K, 210, TILE_SIZE_Q6_K));
        // No wire format for Q5_K / F32: planned (and later repacked) as Q8_0.
        for d in [DType::Q5KM, DType::F32] {
            let (sz, dt, bb, tile) = plan(d);
            assert_eq!((dt, bb, tile), (HtpDataType::Q8_0, 34, TILE_SIZE_Q8_0));
            assert_eq!(sz, repacked_matrix_size_q8_0(64, 32).unwrap());
        }
        assert!(wire_plan(DType::F16, 64, 32, "w").is_err());
    }

    /// An F32 matrix requantizes to the same bytes as the Q8_0 path, including
    /// from an odd (unaligned) byte offset, which an mmap slice can have.
    #[test]
    fn test_repack_weight_f32_matches_q8_0_and_tolerates_unaligned() {
        let (ne0, ne1) = (64usize, 32usize);
        let vals: Vec<f32> = (0..ne0 * ne1)
            .map(|i| ((i * 7) % 31) as f32 * 0.05 - 0.6)
            .collect();
        let q8 = quantize_f32_to_q8_0(&vals, ne0, ne1).unwrap();
        let size = repacked_matrix_size_q8_0(ne0, ne1).unwrap();
        let mut want = vec![0u8; size];
        repack_q8_0(&q8, ne0, ne1, &mut want).unwrap();

        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut got = vec![0u8; size];
        repack_weight(crate::tensor::DType::F32, &bytes, ne0, ne1, &mut got, "w").unwrap();
        assert_eq!(got, want);

        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&bytes);
        let mut got = vec![0u8; size];
        repack_weight(
            crate::tensor::DType::F32,
            &shifted[1..],
            ne0,
            ne1,
            &mut got,
            "w",
        )
        .unwrap();
        assert_eq!(got, want);
    }

    /// Source serving one fixed vector; every other method is unreachable
    /// for a plan holding only [`CopyOp::Vector`].
    struct OneVector(Vec<f32>);

    impl TensorSource for OneVector {
        fn weight_shape(
            &self,
            name: &str,
        ) -> Result<(crate::tensor::DType, usize, usize), CeraError> {
            Err(unknown_tensor(name))
        }
        fn stacked_slice_plan(
            &self,
            name: &str,
            _: usize,
            _: usize,
        ) -> Result<SlicePlan, CeraError> {
            Err(unknown_tensor(name))
        }
        fn vector(&self, _name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CeraError> {
            Ok(std::borrow::Cow::Borrowed(&self.0))
        }
        fn copy_weight(&self, name: &str, _: usize, _: &mut RpcmemBuffer) -> Result<(), CeraError> {
            Err(unknown_tensor(name))
        }
        fn copy_stacked(
            &self,
            name: &str,
            _: &HexagonStackedWeight,
            _: &mut RpcmemBuffer,
        ) -> Result<(), CeraError> {
            Err(unknown_tensor(name))
        }
    }

    fn run_vector_copy(len: usize, vec_len: usize) -> Result<(), CeraError> {
        let src = OneVector(vec![1.0; vec_len]);
        let copies = [CopyOp::Vector {
            name: "blk.0.attn_q_norm.weight".into(),
            offset: 0,
            len,
        }];
        WeightCopy {
            src: &src,
            weights_total: 4096,
            kv_total: 256,
            copies: &copies,
        }
        .run(&crate::backend::hexagon::sys::fake::driver())
        .map(|_| ())
    }

    /// A vector longer than its reserved slot would spill into the next
    /// tensor's region: refused, naming the tensor. Shorter (LFM2 q/k norm
    /// slots are wider than the head_dim tensor) and equal still copy.
    #[test]
    fn vector_longer_than_its_slot_is_rejected() {
        let err = run_vector_copy(4, 5).unwrap_err().to_string();
        assert!(err.contains("blk.0.attn_q_norm.weight"), "{err}");
        assert!(err.contains("slot"), "{err}");
        run_vector_copy(4, 4).unwrap();
        run_vector_copy(8, 4).unwrap();
    }

    /// Source reporting a fixed `(dtype, ne0, ne1)` for every weight.
    struct FixedShape((crate::tensor::DType, usize, usize));

    impl TensorSource for FixedShape {
        fn weight_shape(
            &self,
            _name: &str,
        ) -> Result<(crate::tensor::DType, usize, usize), CeraError> {
            Ok(self.0)
        }
        fn stacked_slice_plan(
            &self,
            name: &str,
            _: usize,
            _: usize,
        ) -> Result<SlicePlan, CeraError> {
            Err(unknown_tensor(name))
        }
        fn vector(&self, name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CeraError> {
            Err(unknown_tensor(name))
        }
        fn copy_weight(&self, name: &str, _: usize, _: &mut RpcmemBuffer) -> Result<(), CeraError> {
            Err(unknown_tensor(name))
        }
        fn copy_stacked(
            &self,
            name: &str,
            _: &HexagonStackedWeight,
            _: &mut RpcmemBuffer,
        ) -> Result<(), CeraError> {
            Err(unknown_tensor(name))
        }
    }

    /// The planner refuses a weight whose K differs from the matmul's or
    /// that has fewer rows than it reads, naming the tensor and the expected
    /// shape; extra rows (a padded-vocab LM head) are fine.
    #[test]
    fn weight_shape_is_validated_against_the_matmul() {
        let (in_dim, out_dim) = (64, 32);
        let plan = |ne0: usize, ne1: usize| {
            let src = FixedShape((crate::tensor::DType::F32, ne0, ne1));
            WeightPlanner::new(&src)
                .weight("blk.0.ffn_up.weight", in_dim, out_dim)
                .map(|_| ())
        };
        let err = plan(in_dim + 1, out_dim).unwrap_err().to_string();
        assert!(err.contains("expected ["), "{err}");
        assert!(err.contains("blk.0.ffn_up.weight"), "{err}");
        let err = plan(in_dim, out_dim - 1).unwrap_err().to_string();
        assert!(err.contains("expected ["), "{err}");
        plan(in_dim, out_dim).unwrap();
        plan(in_dim, out_dim + 8).unwrap();
    }

    /// A bias of the wrong length is refused; absent stays `None`.
    #[test]
    fn bias_length_is_validated() {
        let src = FixedShape((crate::tensor::DType::F32, 0, 0));
        let mut planner = WeightPlanner::new(&src);
        let err = planner
            .bias("blk.0.attn_q.bias", Some(&[0.0; 7]), 8)
            .unwrap_err()
            .to_string();
        assert!(err.contains("blk.0.attn_q.bias"), "{err}");
        assert!(err.contains("expected 8"), "{err}");
        assert!(planner.bias("b", None, 8).unwrap().is_none());
        assert!(planner.bias("b", Some(&[0.0; 8]), 8).unwrap().is_some());
    }

    /// Conv taps whose size is not `hidden_size * 3` fail the copy pass.
    #[test]
    fn conv_taps_size_is_validated() {
        let hs = 8;
        let run = |len: usize| {
            let src = OneVector(vec![1.0; len]);
            let mut planner = WeightPlanner::new(&src);
            planner.conv_taps("blk.0.shortconv.conv.weight", hs);
            WeightCopy {
                src: &src,
                weights_total: planner.total,
                kv_total: 256,
                copies: &planner.copies,
            }
            .run(&crate::backend::hexagon::sys::fake::driver())
            .map(|_| ())
        };
        let err = run(hs * 3 - 1).unwrap_err().to_string();
        assert!(err.contains("blk.0.shortconv.conv.weight"), "{err}");
        assert!(err.contains("hidden_size * 3"), "{err}");
        run(hs * 3).unwrap();
    }
}

//! Shared DSP op emission for the encoder/decoder models (Whisper, ViT, audio detokenizer).
//!
//! These five primitives (`linear_m`, `layer_norm`, `gelu`, `add_residual`,
//! `cpy_f32_to_f16`) were copy-pasted per model and drifted (argument order,
//! error mapping, token tiling). They now live here once. Same-typed
//! parameters travel in structs with named fields ([`TokenShape`],
//! [`LayerNormArgs`]) so a swapped `(dim, n_tokens)` or `(eps, n_tokens)` is a
//! compile error instead of a silent wrong-shape op.
//!
//! Every function is generic over [`OpSink`], which [`HexagonQueueSession`]
//! implements. Host tests drive the same code through a recording sink; no
//! device is needed to check the emitted op sequence.

use super::{
    HTP_TENSOR_COMPUTE, HTP_TENSOR_REPACK, HTP_TENSOR_WEIGHT, HexagonQueueSession,
    HexagonWeightDesc, HexagonWeightFormat, HtpDataType, HtpOpCode, MulMatF32Shape, RpcmemBuffer,
    build_binary_kernel_params, build_hmx_mm_kernel_params, build_layer_norm_params,
    build_mul_mat_f32_kernel_params, build_mul_mat_kernel_params, build_softmax_kernel_params,
    build_ssm_conv_kernel_params, build_unary_kernel_params, mm_hmx_nb1, mm_is_hmx_eligible,
};
use crate::session::CeraError;

/// VTCM budget handed to every kernel-param builder (8 MB on Snapdragon 8 Elite).
pub(crate) const VTCM_BUDGET: usize = 8 * 1024 * 1024;

/// Destination for emitted tensors and ops.
///
/// [`HexagonQueueSession`] is the production implementation; tests supply a
/// recorder. `Buf` is the buffer handle a tensor is placed in.
pub(crate) trait OpSink {
    type Buf;

    #[allow(clippy::too_many_arguments)]
    fn add_tensor(
        &mut self,
        buf: &Self::Buf,
        offset: usize,
        size: usize,
        flags: u32,
        dtype: u32,
        ne: [u32; 4],
        nb: [u32; 4],
    ) -> Result<u16, CeraError>;

    fn enqueue_op(
        &mut self,
        opcode: u32,
        src: &[u16],
        dst: &[u16],
        params: [i32; 16],
        kernel_params: [i32; 32],
    ) -> Result<(), CeraError>;

    fn dsp_threads(&self) -> u32;

    /// Op-group boundary, called once at the end of every helper below. A
    /// helper registers tensors once and reuses their indices across its own
    /// ops, so the only safe place to flush (`CERA_HEXAGON_STEP`) is between
    /// helpers, never between the ops inside one.
    fn end_group(&mut self) -> Result<(), CeraError>;
}

impl OpSink for HexagonQueueSession {
    type Buf = RpcmemBuffer;

    fn add_tensor(
        &mut self,
        buf: &RpcmemBuffer,
        offset: usize,
        size: usize,
        flags: u32,
        dtype: u32,
        ne: [u32; 4],
        nb: [u32; 4],
    ) -> Result<u16, CeraError> {
        HexagonQueueSession::add_tensor(self, buf, offset, size, flags, dtype, ne, nb)
    }

    fn enqueue_op(
        &mut self,
        opcode: u32,
        src: &[u16],
        dst: &[u16],
        params: [i32; 16],
        kernel_params: [i32; 32],
    ) -> Result<(), CeraError> {
        HexagonQueueSession::enqueue_op(self, opcode, src, dst, params, kernel_params)
    }

    fn dsp_threads(&self) -> u32 {
        HexagonQueueSession::dsp_threads(self)
    }

    fn end_group(&mut self) -> Result<(), CeraError> {
        HexagonQueueSession::end_group(self)
    }
}

/// Row-major activation shape: `n_tokens` rows of `dim` elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenShape {
    pub dim: usize,
    pub n_tokens: usize,
}

/// How a token axis is split across DSP ops.
///
/// `Tiles(n)` splits into runs of at most `n` tokens (the VTCM cap the
/// Whisper encoder needs, see `docs/HEXAGON_NPU.md`). `Whole` emits one op
/// over all tokens, which is what the ViT and audio decoders do today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenTile {
    Whole,
    Tiles(usize),
}

/// `(start, len)` token runs covering `0..n_tokens` under `tile`.
fn token_runs(n_tokens: usize, tile: TokenTile) -> impl Iterator<Item = (usize, usize)> {
    let step = match tile {
        TokenTile::Whole => n_tokens.max(1),
        TokenTile::Tiles(n) => n.max(1),
    };
    (0..n_tokens)
        .step_by(step)
        .map(move |start| (start, (n_tokens - start).min(step)))
}

fn op_err(name: &str, e: CeraError) -> CeraError {
    CeraError::Backend(format!("dispatch {name}: {e}"))
}

/// Row-major F32 activation tensor `[dim, rows]` at `offset`.
fn add_f32_rows<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    dim: usize,
    rows: usize,
) -> Result<u16, CeraError> {
    let bytes = dim * rows * 4;
    session.add_tensor(
        buf,
        offset,
        bytes,
        HTP_TENSOR_COMPUTE,
        HtpDataType::F32 as u32,
        [dim as u32, rows as u32, 1, 1],
        [4, (dim * 4) as u32, bytes as u32, bytes as u32],
    )
}

/// One F32 weight row `[len]` (norm scale/shift or bias).
fn add_f32_vector<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    len: usize,
) -> Result<u16, CeraError> {
    let bytes = len * 4;
    session.add_tensor(
        buf,
        offset,
        bytes,
        HTP_TENSOR_WEIGHT,
        HtpDataType::F32 as u32,
        [len as u32, 1, 1, 1],
        [4, bytes as u32, bytes as u32, bytes as u32],
    )
}

/// Broadcast `dst[i] += vec[i]` over a `[len, rows]` destination.
fn enqueue_broadcast_add<S: OpSink>(
    session: &mut S,
    dst_ti: u16,
    vec_ti: u16,
    len: usize,
    name: &str,
) -> Result<(), CeraError> {
    let kparams =
        build_binary_kernel_params(len, len, 1, 1, 1, 4, VTCM_BUDGET, session.dsp_threads());
    session
        .enqueue_op(
            HtpOpCode::Add as u32,
            &[dst_ti, vec_ti],
            &[dst_ti],
            [0i32; 16],
            kparams,
        )
        .map_err(|e| op_err(name, e))
}

/// Opt-in HMX matmul for [`linear_m_with`]: the VTCM budget the HMX chunking is
/// solved against. A caller passes it only for a device whose DSP has HMX.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HmxMm {
    pub vtcm_budget: usize,
}

/// Linear layer `dst[tokens, rows] = x[tokens, cols] . Wt (+ bias)`.
///
/// `bias_offset` is `None` for bias-free projections. With
/// [`TokenTile::Tiles`] the matmul and bias add are emitted per tile.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_m<S: OpSink>(
    session: &mut S,
    x: &S::Buf,
    x_offset: usize,
    weights: &S::Buf,
    w_desc: HexagonWeightDesc,
    bias_offset: impl Into<Option<usize>>,
    dst: &S::Buf,
    dst_offset: usize,
    n_tokens: usize,
    tile: TokenTile,
) -> Result<(), CeraError> {
    linear_m_with(
        session,
        x,
        x_offset,
        weights,
        w_desc,
        bias_offset,
        dst,
        dst_offset,
        n_tokens,
        tile,
        None,
    )
}

/// [`linear_m`] that may run the matmul on the HMX matrix unit.
///
/// With `hmx` set, a whole-batch (`TokenTile::Whole`) projection whose dims are 32
/// aligned and whose batch has at least [`mm_is_hmx_eligible`]'s minimum rows runs the
/// DSP's HMX matmul, as the LFM2 prefill does: the same repacked weights feed both
/// kernels and only the dim-1 weight stride differs. Anything else, or a chunking
/// that does not fit the budget, keeps the HVX matmul.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_m_with<S: OpSink>(
    session: &mut S,
    x: &S::Buf,
    x_offset: usize,
    weights: &S::Buf,
    w_desc: HexagonWeightDesc,
    bias_offset: impl Into<Option<usize>>,
    dst: &S::Buf,
    dst_offset: usize,
    n_tokens: usize,
    tile: TokenTile,
    hmx: Option<HmxMm>,
) -> Result<(), CeraError> {
    let (w_dtype, block_bytes, tile_size) = match w_desc.format {
        HexagonWeightFormat::RepackedQ8_0 => (HtpDataType::Q8_0, 34usize, 32 * 34usize),
        HexagonWeightFormat::RepackedQ4_0 => (HtpDataType::Q4_0, 18usize, 32 * 18usize),
    };
    let (cols, rows) = (w_desc.cols, w_desc.rows);
    let tiled_row_bytes = cols.div_ceil(32) * tile_size;
    let w_tot = rows.div_ceil(32) * tiled_row_bytes;
    // One whole-batch run only: the weight tensor's stride is registered once and the
    // HMX kernel reads it differently from the HVX one.
    let hmx_kparams = match (hmx, tile) {
        (Some(h), TokenTile::Whole) if mm_is_hmx_eligible(w_dtype, cols, rows, n_tokens) => {
            build_hmx_mm_kernel_params(
                w_dtype,
                cols,
                rows,
                n_tokens.next_multiple_of(32),
                n_tokens,
                session.dsp_threads(),
                h.vtcm_budget,
            )
        }
        _ => None,
    };
    let w_nb1 = if hmx_kparams.is_some() {
        mm_hmx_nb1(w_dtype, cols)
    } else {
        tiled_row_bytes
    };
    let w_ti = session.add_tensor(
        weights,
        w_desc.offset,
        w_desc.size_bytes,
        HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
        w_dtype as u32,
        [cols as u32, rows as u32, 1, 1],
        [block_bytes as u32, w_nb1 as u32, w_tot as u32, w_tot as u32],
    )?;
    let b_ti = match bias_offset.into() {
        Some(b_off) => Some(add_f32_vector(session, weights, b_off, rows)?),
        None => None,
    };

    for (start, run) in token_runs(n_tokens, tile) {
        let x_ti = add_f32_rows(session, x, x_offset + start * cols * 4, cols, run)?;
        let dst_ti = add_f32_rows(session, dst, dst_offset + start * rows * 4, rows, run)?;
        let kparams = hmx_kparams.unwrap_or_else(|| {
            build_mul_mat_kernel_params(
                w_dtype,
                cols,
                run as u32,
                1,
                rows * 4,
                session.dsp_threads(),
                VTCM_BUDGET,
            )
        });
        session
            .enqueue_op(
                HtpOpCode::MulMat as u32,
                &[w_ti, x_ti],
                &[dst_ti],
                [0i32; 16],
                kparams,
            )
            .map_err(|e| op_err("linear_m", e))?;
        if let Some(b_ti) = b_ti {
            enqueue_broadcast_add(session, dst_ti, b_ti, rows, "linear_m bias_add")?;
        }
    }
    session.end_group().map_err(|e| op_err("linear_m", e))
}

/// Arguments for [`layer_norm`]: `dst = norm(src) * weight + bias`.
pub(crate) struct LayerNormArgs<'a, B> {
    pub src: &'a B,
    pub src_offset: usize,
    pub dst: &'a B,
    pub dst_offset: usize,
    pub weights: &'a B,
    pub w_offset: usize,
    pub b_offset: usize,
    pub shape: TokenShape,
    pub eps: f32,
    pub tile: TokenTile,
}

/// LayerNorm: `Norm`, then `Mul` by the scale row, then `Add` of the shift row.
pub(crate) fn layer_norm<S: OpSink>(
    session: &mut S,
    a: LayerNormArgs<'_, S::Buf>,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = a.shape;
    let w_ti = add_f32_vector(session, a.weights, a.w_offset, dim)?;
    let b_ti = add_f32_vector(session, a.weights, a.b_offset, dim)?;

    for (start, run) in token_runs(n_tokens, a.tile) {
        let src_ti = add_f32_rows(session, a.src, a.src_offset + start * dim * 4, dim, run)?;
        let dst_ti = add_f32_rows(session, a.dst, a.dst_offset + start * dim * 4, dim, run)?;
        let kparams =
            build_unary_kernel_params(dim, run, 0, VTCM_BUDGET, session.dsp_threads(), false);
        session
            .enqueue_op(
                HtpOpCode::Norm as u32,
                &[src_ti],
                &[dst_ti],
                build_layer_norm_params(a.eps),
                kparams,
            )
            .map_err(|e| op_err("layer_norm norm", e))?;

        let mul_kparams =
            build_binary_kernel_params(dim, dim, 1, 1, 1, 4, VTCM_BUDGET, session.dsp_threads());
        session
            .enqueue_op(
                HtpOpCode::Mul as u32,
                &[dst_ti, w_ti],
                &[dst_ti],
                [0i32; 16],
                mul_kparams,
            )
            .map_err(|e| op_err("layer_norm mul", e))?;
        enqueue_broadcast_add(session, dst_ti, b_ti, dim, "layer_norm add")?;
    }
    session.end_group().map_err(|e| op_err("layer_norm", e))
}

/// In-place elementwise unary op over `[dim, n_tokens]` (`buf = op(buf)`).
fn unary_inplace<S: OpSink>(
    session: &mut S,
    opcode: HtpOpCode,
    name: &str,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace_with(session, opcode, name, [0i32; 16], buf, offset, shape, tile)
}

/// [`unary_inplace`] for ops that take parameters (`Scale`: scale and bias).
#[allow(clippy::too_many_arguments)]
fn unary_inplace_with<S: OpSink>(
    session: &mut S,
    opcode: HtpOpCode,
    name: &str,
    params: [i32; 16],
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    for (start, run) in token_runs(n_tokens, tile) {
        let ti = add_f32_rows(session, buf, offset + start * dim * 4, dim, run)?;
        let kparams =
            build_unary_kernel_params(dim, run, 0, VTCM_BUDGET, session.dsp_threads(), false);
        session
            .enqueue_op(opcode as u32, &[ti], &[ti], params, kparams)
            .map_err(|e| op_err(name, e))?;
    }
    session.end_group().map_err(|e| op_err(name, e))
}

/// In-place `buf = buf * scale + bias`.
pub(crate) fn scale_offset<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
    (scale, bias): (f32, f32),
) -> Result<(), CeraError> {
    let mut params = [0i32; 16];
    params[0] = scale.to_bits() as i32;
    params[1] = bias.to_bits() as i32;
    unary_inplace_with(
        session,
        HtpOpCode::Scale,
        "scale_offset",
        params,
        buf,
        offset,
        shape,
        tile,
    )
}

/// `2 * sqrt(2 / pi)`, the sigmoid form of the tanh GELU's constant.
const GELU_TANH_SIGMOID_SCALE: f32 = 1.595_769_2;

/// Rows of scratch the tanh GELU works through at a time.
pub(crate) const GELU_TMP_ROWS: usize = 64;

/// Rows of `dim` f32 elements that fit in `bytes` of GELU scratch. An error,
/// not a clamp, when not even one fits: a row wider than the scratch would be
/// written past its end by the DSP, and the shapes come from model files.
pub(crate) fn gelu_tmp_rows(bytes: usize, dim: usize) -> Result<usize, CeraError> {
    if dim == 0 || dim.saturating_mul(4) > bytes {
        return Err(CeraError::Backend(format!(
            "GELU row of {dim} elements does not fit the {bytes} B scratch"
        )));
    }
    Ok(bytes / (dim * 4))
}

/// In-place GELU, tanh form (`gelu_pytorch_tanh`, what the CPU ViT and Whisper
/// use; the CPU audio adapter uses the erf form, which this stays within about
/// 5e-4 of, absolute). The DSP's own `UNARY_GELU` is not this: it is the quick approximation
/// `x * sigmoid(1.702 x)`, up to about 2% per element away, which a wide
/// down-projection turned into a 20% error in the vision transformer, so no
/// helper here exposes it.
///
/// Computed as
/// `x * sigmoid(2 sqrt(2/pi) * x * (1 + 0.044715 x^2))`, since
/// `0.5 (1 + tanh z) = sigmoid(2 z)`. The DSP has no tanh GELU, so this is
/// seven ops over `tmp`, a scratch region of `tmp_rows` rows of `shape.dim`
/// elements that the rows are processed through, `tmp_rows` at a time.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gelu_tanh<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    tmp: &S::Buf,
    tmp_offset: usize,
    tmp_rows: usize,
    shape: TokenShape,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    for (start, run) in token_runs(n_tokens, TokenTile::Tiles(tmp_rows)) {
        let x_off = offset + start * dim * 4;
        let rows = TokenShape { dim, n_tokens: run };
        fn whole<B>(buf: &B, off: usize, dim: usize, run: usize) -> View<'_, B> {
            View::new(buf, off, [dim, run, 1], [4, dim * 4, run * dim * 4])
        }
        // tmp = x; tmp = x * x
        copy_view(
            session,
            whole(buf, x_off, dim, run),
            whole(tmp, tmp_offset, dim, run),
        )?;
        mul_inplace(session, tmp, tmp_offset, buf, x_off, rows, TokenTile::Whole)?;
        // tmp = x * (1 + 0.044715 x^2)
        scale_offset(
            session,
            tmp,
            tmp_offset,
            rows,
            TokenTile::Whole,
            (0.044_715, 1.0),
        )?;
        mul_inplace(session, tmp, tmp_offset, buf, x_off, rows, TokenTile::Whole)?;
        // x = x * sigmoid(2 sqrt(2/pi) * tmp)
        scale_offset(
            session,
            tmp,
            tmp_offset,
            rows,
            TokenTile::Whole,
            (GELU_TANH_SIGMOID_SCALE, 0.0),
        )?;
        sigmoid(session, tmp, tmp_offset, rows, TokenTile::Whole)?;
        mul_inplace(session, buf, x_off, tmp, tmp_offset, rows, TokenTile::Whole)?;
    }
    Ok(())
}

/// In-place natural logarithm: `buf = ln(buf)` (the inputs must be positive).
pub(crate) fn log_inplace<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace(
        session,
        HtpOpCode::UnaryLog,
        "log",
        buf,
        offset,
        shape,
        tile,
    )
}

/// In-place SiLU: `buf = silu(buf)`.
pub(crate) fn silu<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace(
        session,
        HtpOpCode::UnarySilu,
        "silu",
        buf,
        offset,
        shape,
        tile,
    )
}

/// In-place per-channel scale: `buf[row, :] *= vec[:]` for every row. The
/// vector lives in `vec_buf` at `vec_offset` (an F32 weight row of `dim`).
pub(crate) fn mul_row_bcast<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    vec_buf: &S::Buf,
    vec_offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    let vec_ti = add_f32_vector(session, vec_buf, vec_offset, dim)?;
    for (start, run) in token_runs(n_tokens, tile) {
        let ti = add_f32_rows(session, buf, offset + start * dim * 4, dim, run)?;
        let kparams =
            build_binary_kernel_params(dim, dim, 1, 1, 1, 4, VTCM_BUDGET, session.dsp_threads());
        session
            .enqueue_op(
                HtpOpCode::Mul as u32,
                &[ti, vec_ti],
                &[ti],
                [0i32; 16],
                kparams,
            )
            .map_err(|e| op_err("mul_row_bcast", e))?;
    }
    session.end_group().map_err(|e| op_err("mul_row_bcast", e))
}

/// In-place sigmoid: `buf = sigmoid(buf)`.
pub(crate) fn sigmoid<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace(
        session,
        HtpOpCode::UnarySigmoid,
        "sigmoid",
        buf,
        offset,
        shape,
        tile,
    )
}

/// In-place square root: `buf = sqrt(buf)` (the inputs must not be negative).
pub(crate) fn sqrt<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace(session, HtpOpCode::Sqrt, "sqrt", buf, offset, shape, tile)
}

/// In-place hyperbolic tangent: `buf = tanh(buf)`.
pub(crate) fn tanh<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace(
        session,
        HtpOpCode::UnaryTanh,
        "tanh",
        buf,
        offset,
        shape,
        tile,
    )
}

/// `Argmax` over one row of `vocab` F32 values at `in_offset`: the index of the largest, as an
/// I32 at `out_offset`. Greedy sampling on the DSP reads 4 bytes instead of the whole row.
pub(crate) fn argmax_row<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    in_offset: usize,
    out_offset: usize,
    vocab: usize,
) -> Result<(), CeraError> {
    let in_ti = session.add_tensor(
        buf,
        in_offset,
        vocab * 4,
        HTP_TENSOR_COMPUTE,
        HtpDataType::F32 as u32,
        [vocab as u32, 1, 1, 1],
        [
            4,
            (vocab * 4) as u32,
            (vocab * 4) as u32,
            (vocab * 4) as u32,
        ],
    )?;
    let out_ti = session.add_tensor(
        buf,
        out_offset,
        4,
        HTP_TENSOR_COMPUTE,
        HtpDataType::I32 as u32,
        [1, 1, 1, 1],
        [4, 4, 4, 4],
    )?;
    session
        .enqueue_op(
            HtpOpCode::Argmax as u32,
            &[in_ti],
            &[out_ti],
            [0i32; 16],
            [0i32; 32],
        )
        .map_err(|e| op_err("argmax", e))?;
    session.end_group().map_err(|e| op_err("argmax", e))
}

/// In-place ReLU: `buf = max(buf, 0)`.
pub(crate) fn relu<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    unary_inplace(
        session,
        HtpOpCode::UnaryRelu,
        "relu",
        buf,
        offset,
        shape,
        tile,
    )
}

/// In-place bias add: `buf[row, :] += vec[:]` for every row.
pub(crate) fn add_row_bcast<S: OpSink>(
    session: &mut S,
    buf: &S::Buf,
    offset: usize,
    vec_buf: &S::Buf,
    vec_offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    let vec_ti = add_f32_vector(session, vec_buf, vec_offset, dim)?;
    for (start, run) in token_runs(n_tokens, tile) {
        let ti = add_f32_rows(session, buf, offset + start * dim * 4, dim, run)?;
        enqueue_broadcast_add(session, ti, vec_ti, dim, "add_row_bcast")?;
    }
    session.end_group().map_err(|e| op_err("add_row_bcast", e))
}

/// In-place elementwise product: `dst *= src` over `[dim, n_tokens]`.
pub(crate) fn mul_inplace<S: OpSink>(
    session: &mut S,
    dst: &S::Buf,
    dst_offset: usize,
    src: &S::Buf,
    src_offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    for (start, run) in token_runs(n_tokens, tile) {
        let dst_ti = add_f32_rows(session, dst, dst_offset + start * dim * 4, dim, run)?;
        let src_ti = add_f32_rows(session, src, src_offset + start * dim * 4, dim, run)?;
        let kparams =
            build_binary_kernel_params(dim, dim, run, 1, 1, 4, VTCM_BUDGET, session.dsp_threads());
        session
            .enqueue_op(
                HtpOpCode::Mul as u32,
                &[dst_ti, src_ti],
                &[dst_ti],
                [0i32; 16],
                kparams,
            )
            .map_err(|e| op_err("mul_inplace", e))?;
    }
    session.end_group().map_err(|e| op_err("mul_inplace", e))
}

/// A source for [`concat_time_inner`]: `rows` positions of `dim` channels,
/// the position step `nb0` and the channel step `nb1` in bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConcatSrc<'a, B> {
    pub buf: &'a B,
    pub offset: usize,
    pub rows: usize,
    pub nb0: usize,
    pub nb1: usize,
}

/// Concatenate two sequences along time into a channel-major `[rows, dim]`
/// destination (one channel's positions contiguous), the layout `SsmConv`
/// reads. `Concat` along dim 0: each source is read through its own strides,
/// so a time-major `[rows, dim]` activation (position step `dim * 4`, channel
/// step 4) transposes on the way in.
pub(crate) fn concat_time_inner<S: OpSink>(
    session: &mut S,
    s0: ConcatSrc<'_, S::Buf>,
    s1: ConcatSrc<'_, S::Buf>,
    dst: &S::Buf,
    dst_offset: usize,
    dim: usize,
) -> Result<(), CeraError> {
    let span = |s: &ConcatSrc<'_, S::Buf>| {
        s.rows.saturating_sub(1) * s.nb0 + dim.saturating_sub(1) * s.nb1 + 4
    };
    let add = |session: &mut S, s: &ConcatSrc<'_, S::Buf>| {
        let span = span(s);
        session.add_tensor(
            s.buf,
            s.offset,
            span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [s.rows as u32, dim as u32, 1, 1],
            [s.nb0 as u32, s.nb1 as u32, span as u32, span as u32],
        )
    };
    let s0_ti = add(session, &s0)?;
    let s1_ti = add(session, &s1)?;
    let dst_rows = s0.rows + s1.rows;
    let dst_span = dst_rows * dim * 4;
    let dst_ti = session.add_tensor(
        dst,
        dst_offset,
        dst_span,
        HTP_TENSOR_COMPUTE,
        HtpDataType::F32 as u32,
        [dst_rows as u32, dim as u32, 1, 1],
        [4, (dst_rows * 4) as u32, dst_span as u32, dst_span as u32],
    )?;
    let mut params = [0i32; 16];
    params[0] = 0; // concat along dim 0
    session
        .enqueue_op(
            HtpOpCode::Concat as u32,
            &[s0_ti, s1_ti],
            &[dst_ti],
            params,
            [0i32; 32],
        )
        .map_err(|e| op_err("concat_time_inner", e))?;
    session
        .end_group()
        .map_err(|e| op_err("concat_time_inner", e))
}

/// Depthwise 1-D convolution through `SsmConv`:
/// `y[t, c] = sum_j x[t + j, c] * w[j, c]` over a channel-major input of
/// `d_conv - 1 + n_t` positions (the caller supplies the padding) and a
/// `[d_conv, d_inner]` tap matrix, producing a time-major `[n_t, d_inner]`
/// output. `vtcm_budget` caps the VTCM the kernel may plan for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssm_conv<S: OpSink>(
    session: &mut S,
    taps: &S::Buf,
    taps_offset: usize,
    x: &S::Buf,
    x_offset: usize,
    dst: &S::Buf,
    dst_offset: usize,
    d_conv: usize,
    d_inner: usize,
    n_t: usize,
    vtcm_budget: usize,
) -> Result<(), CeraError> {
    let ncs = d_conv - 1 + n_t;
    let x_bytes = ncs * d_inner * 4;
    let x_ti = session.add_tensor(
        x,
        x_offset,
        x_bytes,
        HTP_TENSOR_COMPUTE,
        HtpDataType::F32 as u32,
        [ncs as u32, d_inner as u32, 1, 1],
        [4, (ncs * 4) as u32, x_bytes as u32, x_bytes as u32],
    )?;
    let w_bytes = d_conv * d_inner * 4;
    let w_ti = session.add_tensor(
        taps,
        taps_offset,
        w_bytes,
        HTP_TENSOR_WEIGHT,
        HtpDataType::F32 as u32,
        [d_conv as u32, d_inner as u32, 1, 1],
        [4, (d_conv * 4) as u32, w_bytes as u32, w_bytes as u32],
    )?;
    let dst_bytes = d_inner * n_t * 4;
    let dst_ti = session.add_tensor(
        dst,
        dst_offset,
        dst_bytes,
        HTP_TENSOR_COMPUTE,
        HtpDataType::F32 as u32,
        [d_inner as u32, n_t as u32, 1, 1],
        [4, (d_inner * 4) as u32, dst_bytes as u32, dst_bytes as u32],
    )?;
    let kparams = build_ssm_conv_kernel_params(
        d_conv,
        d_inner,
        n_t,
        1,
        ncs,
        session.dsp_threads(),
        vtcm_budget,
    );
    session
        .enqueue_op(
            HtpOpCode::SsmConv as u32,
            &[x_ti, w_ti],
            &[dst_ti],
            [0i32; 16],
            kparams,
        )
        .map_err(|e| op_err("ssm_conv", e))?;
    session.end_group().map_err(|e| op_err("ssm_conv", e))
}

/// A strided F32 view of up to three dims, for ops that take shaped tensors
/// (batched matmul, softmax, copies). Strides are in bytes and `nb[0]` is
/// always 4 unless the view transposes (`nb[0] > 4`).
pub(crate) struct View<'a, B> {
    pub buf: &'a B,
    pub offset: usize,
    pub ne: [usize; 3],
    pub nb: [usize; 3],
}

// Manual impls: a view only holds a reference, so it is `Copy` whatever `B` is.
impl<B> Clone for View<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B> Copy for View<'_, B> {}

impl<'a, B> View<'a, B> {
    pub(crate) fn new(buf: &'a B, offset: usize, ne: [usize; 3], nb: [usize; 3]) -> Self {
        Self {
            buf,
            offset,
            ne,
            nb,
        }
    }

    /// Bytes from the first element to one past the last.
    fn span(&self) -> usize {
        (0..3)
            .map(|i| self.ne[i].saturating_sub(1) * self.nb[i])
            .sum::<usize>()
            + 4
    }

    fn add<S: OpSink<Buf = B>>(&self, session: &mut S, flags: u32) -> Result<u16, CeraError> {
        let span = self.span();
        let [nb0, nb1, nb2] = self.nb.map(|v| v as u32);
        session.add_tensor(
            self.buf,
            self.offset,
            span,
            flags,
            HtpDataType::F32 as u32,
            [self.ne[0] as u32, self.ne[1] as u32, self.ne[2] as u32, 1],
            [nb0, nb1, nb2, span as u32],
        )
    }
}

/// Batched F32 matmul on the HVX path: for every batch `b`,
/// `dst[n, m, b] = sum_k a[k, n, b] * x[k, m, b]`.
pub(crate) fn matmul_f32<S: OpSink>(
    session: &mut S,
    a: View<'_, S::Buf>,
    x: View<'_, S::Buf>,
    dst: View<'_, S::Buf>,
) -> Result<(), CeraError> {
    let kparams = build_mul_mat_f32_kernel_params(
        MulMatF32Shape {
            ne00: a.ne[0],
            ne02: a.ne[2],
            ne03: 1,
            src0_nb1: a.nb[1],
            ne11: x.ne[1],
            ne12: x.ne[2],
            ne13: 1,
            dst_nb1: dst.nb[1],
        },
        session.dsp_threads(),
        VTCM_BUDGET,
    )
    .ok_or_else(|| {
        op_err(
            "matmul_f32",
            CeraError::Backend(
                "activations do not fit the VTCM, or the batch shapes do not line up".into(),
            ),
        )
    })?;
    let a_ti = a.add(session, HTP_TENSOR_COMPUTE)?;
    let x_ti = x.add(session, HTP_TENSOR_COMPUTE)?;
    let dst_ti = dst.add(session, HTP_TENSOR_COMPUTE)?;
    session
        .enqueue_op(
            HtpOpCode::MulMat as u32,
            &[a_ti, x_ti],
            &[dst_ti],
            [0i32; 16],
            kparams,
        )
        .map_err(|e| op_err("matmul_f32", e))?;
    session.end_group().map_err(|e| op_err("matmul_f32", e))
}

/// `dst = softmax(scale * x + mask)` along dim 0. The mask (F32) is read
/// through its own strides, one row per `x` row and one plane per head, so a
/// shifted view of a larger matrix works as a relative-position bias.
pub(crate) fn softmax<S: OpSink>(
    session: &mut S,
    x: View<'_, S::Buf>,
    mask: Option<View<'_, S::Buf>>,
    dst: View<'_, S::Buf>,
    scale: f32,
) -> Result<(), CeraError> {
    let kparams = build_softmax_kernel_params(
        [x.ne[0], x.ne[1], x.ne[2], 1],
        mask.as_ref().map(|m| [m.ne[0], m.ne[2], 1]),
        scale,
        session.dsp_threads(),
    );
    let x_ti = x.add(session, HTP_TENSOR_COMPUTE)?;
    let mask_ti = match &mask {
        Some(m) => Some(m.add(session, HTP_TENSOR_COMPUTE)?),
        None => None,
    };
    let dst_ti = dst.add(session, HTP_TENSOR_COMPUTE)?;
    let mut params = [0i32; 16];
    params[0] = scale.to_bits() as i32;
    let src: Vec<u16> = std::iter::once(x_ti).chain(mask_ti).collect();
    session
        .enqueue_op(HtpOpCode::Softmax as u32, &src, &[dst_ti], params, kparams)
        .map_err(|e| op_err("softmax", e))?;
    session.end_group().map_err(|e| op_err("softmax", e))
}

/// `dst = src` through the DSP's generic same-type copy, which reads and
/// writes through each view's strides (a transpose when the strides differ).
pub(crate) fn copy_view<S: OpSink>(
    session: &mut S,
    src: View<'_, S::Buf>,
    dst: View<'_, S::Buf>,
) -> Result<(), CeraError> {
    let src_ti = src.add(session, HTP_TENSOR_COMPUTE)?;
    let dst_ti = dst.add(session, HTP_TENSOR_COMPUTE)?;
    session
        .enqueue_op(
            HtpOpCode::Cpy as u32,
            &[src_ti],
            &[dst_ti],
            [0i32; 16],
            [0i32; 32],
        )
        .map_err(|e| op_err("copy_view", e))?;
    session.end_group().map_err(|e| op_err("copy_view", e))
}

/// Out-of-place row-broadcast add: `dst[row, :] = src[row, :] + vec[:]`.
pub(crate) fn add_row_bcast_to<S: OpSink>(
    session: &mut S,
    src: View<'_, S::Buf>,
    vec_buf: &S::Buf,
    vec_offset: usize,
    dst: View<'_, S::Buf>,
) -> Result<(), CeraError> {
    let dim = src.ne[0];
    let vec_ti = add_f32_vector(session, vec_buf, vec_offset, dim)?;
    let src_ti = src.add(session, HTP_TENSOR_COMPUTE)?;
    let dst_ti = dst.add(session, HTP_TENSOR_COMPUTE)?;
    let kparams =
        build_binary_kernel_params(dim, dim, 1, 1, 1, 4, VTCM_BUDGET, session.dsp_threads());
    session
        .enqueue_op(
            HtpOpCode::Add as u32,
            &[src_ti, vec_ti],
            &[dst_ti],
            [0i32; 16],
            kparams,
        )
        .map_err(|e| op_err("add_row_bcast_to", e))?;
    session
        .end_group()
        .map_err(|e| op_err("add_row_bcast_to", e))
}

/// In-place residual add: `dst += src`.
pub(crate) fn add_residual<S: OpSink>(
    session: &mut S,
    dst: &S::Buf,
    dst_offset: usize,
    src: &S::Buf,
    src_offset: usize,
    shape: TokenShape,
    tile: TokenTile,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    for (start, run) in token_runs(n_tokens, tile) {
        let dst_ti = add_f32_rows(session, dst, dst_offset + start * dim * 4, dim, run)?;
        let src_ti = add_f32_rows(session, src, src_offset + start * dim * 4, dim, run)?;
        let kparams =
            build_binary_kernel_params(dim, dim, run, 1, 1, 4, VTCM_BUDGET, session.dsp_threads());
        session
            .enqueue_op(
                HtpOpCode::Add as u32,
                &[dst_ti, src_ti],
                &[dst_ti],
                [0i32; 16],
                kparams,
            )
            .map_err(|e| op_err("add_residual", e))?;
    }
    session.end_group().map_err(|e| op_err("add_residual", e))
}

/// `Cpy` that narrows an F32 `[dim, n_tokens]` activation to F16. Never
/// tiled: a plain copy has no VTCM working set.
pub(crate) fn cpy_f32_to_f16<S: OpSink>(
    session: &mut S,
    src: &S::Buf,
    src_offset: usize,
    dst: &S::Buf,
    dst_offset: usize,
    shape: TokenShape,
) -> Result<(), CeraError> {
    let TokenShape { dim, n_tokens } = shape;
    let src_ti = add_f32_rows(session, src, src_offset, dim, n_tokens)?;
    let dst_bytes = dim * n_tokens * 2;
    let dst_ti = session.add_tensor(
        dst,
        dst_offset,
        dst_bytes,
        HTP_TENSOR_COMPUTE,
        HtpDataType::F16 as u32,
        [dim as u32, n_tokens as u32, 1, 1],
        [2, (dim * 2) as u32, dst_bytes as u32, dst_bytes as u32],
    )?;
    session
        .enqueue_op(
            HtpOpCode::Cpy as u32,
            &[src_ti],
            &[dst_ti],
            [0i32; 16],
            [0i32; 32],
        )
        .map_err(|e| op_err("cpy_f32_to_f16", e))?;
    session.end_group().map_err(|e| op_err("cpy_f32_to_f16", e))
}

#[cfg(test)]
pub(crate) mod testing {
    //! Recording [`OpSink`] shared by the model op-emission tests.
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct RecTensor {
        pub buf: &'static str,
        pub offset: usize,
        pub size: usize,
        pub flags: u32,
        pub dtype: u32,
        pub ne: [u32; 4],
        pub nb: [u32; 4],
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct RecOp {
        pub opcode: u32,
        pub src: Vec<u16>,
        pub dst: Vec<u16>,
        pub params: [i32; 16],
        pub kparams: [i32; 32],
    }

    #[derive(Default)]
    pub(crate) struct RecordingSink {
        pub tensors: Vec<RecTensor>,
        pub ops: Vec<RecOp>,
        /// Fail the enqueue with this ordinal (0-based) to test error mapping.
        pub fail_op_at: Option<usize>,
        /// `ops.len()` at each `end_group` call, in order.
        pub group_ends: Vec<usize>,
    }

    impl RecordingSink {
        pub(crate) fn opcodes(&self) -> Vec<u32> {
            self.ops.iter().map(|o| o.opcode).collect()
        }

        /// Tensor referenced as src operand `i` of op `op`.
        pub(crate) fn src(&self, op: usize, i: usize) -> &RecTensor {
            &self.tensors[self.ops[op].src[i] as usize]
        }

        pub(crate) fn dst(&self, op: usize) -> &RecTensor {
            &self.tensors[self.ops[op].dst[0] as usize]
        }
    }

    /// Ops emitted by a 130-token `layer_norm` under `tile`: pins what a
    /// model's tile constant does (tiled = per-tile ops, whole = one triple).
    pub(crate) fn layer_norm_op_count(tile: TokenTile) -> usize {
        let mut s = RecordingSink::default();
        layer_norm(
            &mut s,
            LayerNormArgs {
                src: &"src",
                src_offset: 0,
                dst: &"dst",
                dst_offset: 0,
                weights: &"w",
                w_offset: 0,
                b_offset: 128,
                shape: TokenShape {
                    dim: 48,
                    n_tokens: 130,
                },
                eps: 1e-5,
                tile,
            },
        )
        .unwrap();
        s.ops.len()
    }

    impl OpSink for RecordingSink {
        type Buf = &'static str;

        fn add_tensor(
            &mut self,
            buf: &&'static str,
            offset: usize,
            size: usize,
            flags: u32,
            dtype: u32,
            ne: [u32; 4],
            nb: [u32; 4],
        ) -> Result<u16, CeraError> {
            self.tensors.push(RecTensor {
                buf,
                offset,
                size,
                flags,
                dtype,
                ne,
                nb,
            });
            Ok((self.tensors.len() - 1) as u16)
        }

        fn enqueue_op(
            &mut self,
            opcode: u32,
            src: &[u16],
            dst: &[u16],
            params: [i32; 16],
            kernel_params: [i32; 32],
        ) -> Result<(), CeraError> {
            if self.fail_op_at == Some(self.ops.len()) {
                return Err(CeraError::Backend("injected".into()));
            }
            self.ops.push(RecOp {
                opcode,
                src: src.to_vec(),
                dst: dst.to_vec(),
                params,
                kparams: kernel_params,
            });
            Ok(())
        }

        fn dsp_threads(&self) -> u32 {
            8
        }

        fn end_group(&mut self) -> Result<(), CeraError> {
            self.group_ends.push(self.ops.len());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    const OP_MUL: u32 = HtpOpCode::Mul as u32;
    const OP_ADD: u32 = HtpOpCode::Add as u32;
    const OP_ARGMAX: u32 = HtpOpCode::Argmax as u32;
    const OP_NORM: u32 = HtpOpCode::Norm as u32;
    const OP_MULMAT: u32 = HtpOpCode::MulMat as u32;
    const OP_SILU: u32 = HtpOpCode::UnarySilu as u32;
    const OP_CPY: u32 = HtpOpCode::Cpy as u32;
    /// A tiled policy for the tile-walk tests (the value Whisper uses).
    const TILE_64: TokenTile = TokenTile::Tiles(64);

    fn desc(rows: usize, cols: usize) -> HexagonWeightDesc {
        HexagonWeightDesc {
            offset: 4096,
            size_bytes: 12345,
            format: HexagonWeightFormat::RepackedQ8_0,
            rows,
            cols,
        }
    }

    #[test]
    fn token_runs_cover_exactly() {
        let runs: Vec<_> = token_runs(130, TokenTile::Tiles(64)).collect();
        assert_eq!(runs, vec![(0, 64), (64, 64), (128, 2)]);
        assert_eq!(
            token_runs(130, TokenTile::Whole).collect::<Vec<_>>(),
            vec![(0, 130)]
        );
        assert_eq!(token_runs(0, TokenTile::Whole).count(), 0);
        assert_eq!(token_runs(0, TokenTile::Tiles(64)).count(), 0);
        assert_eq!(token_runs(3, TokenTile::Tiles(0)).count(), 3);
    }

    #[test]
    fn linear_m_tiles_matmul_and_bias_per_tile() {
        let mut s = RecordingSink::default();
        linear_m(
            &mut s,
            &"x",
            100,
            &"w",
            desc(96, 64),
            512usize,
            &"y",
            200,
            130,
            TokenTile::Tiles(64),
        )
        .unwrap();
        assert_eq!(
            s.opcodes(),
            vec![OP_MULMAT, OP_ADD, OP_MULMAT, OP_ADD, OP_MULMAT, OP_ADD]
        );
        // Weight first, bias second, then per tile x and dst.
        assert_eq!(s.tensors[0].flags, HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK);
        assert_eq!(s.tensors[0].ne, [64, 96, 1, 1]);
        assert_eq!(s.tensors[1].offset, 512);
        assert_eq!(s.tensors[1].ne, [96, 1, 1, 1]);
        // Tile 1 (tokens 64..128): x advances by 64 * cols * 4, dst by 64 * rows * 4.
        assert_eq!(s.src(2, 1).offset, 100 + 64 * 64 * 4);
        assert_eq!(s.src(2, 1).ne, [64, 64, 1, 1]);
        assert_eq!(s.dst(2).offset, 200 + 64 * 96 * 4);
        assert_eq!(s.dst(2).ne, [96, 64, 1, 1]);
        // Tail tile of 2 tokens.
        assert_eq!(s.src(4, 1).ne, [64, 2, 1, 1]);
        assert_eq!(s.dst(4).size, 2 * 96 * 4);
        // Bias add targets the tile's dst tensor and the shared bias tensor.
        assert_eq!(s.ops[5].src, vec![s.ops[4].dst[0], 1]);
    }

    /// `kparams[6]` is the DSP's `n_hmx` word: 1 on the HMX matmul, 0 on the HVX one.
    const KPARAMS_N_HMX: usize = 6;

    #[test]
    fn linear_m_with_hmx_runs_the_hmx_matmul_with_its_own_weight_stride() {
        let hmx = Some(HmxMm {
            vtcm_budget: 8 * 1024 * 1024,
        });
        let run = |tokens: usize, rows: usize, cols: usize, tile: TokenTile, hmx| {
            let mut s = RecordingSink::default();
            linear_m_with(
                &mut s,
                &"x",
                0,
                &"w",
                desc(rows, cols),
                None,
                &"y",
                0,
                tokens,
                tile,
                hmx,
            )
            .unwrap();
            s
        };

        let s = run(280, 96, 64, TokenTile::Whole, hmx);
        assert_eq!(s.opcodes(), vec![OP_MULMAT]);
        assert_eq!(s.ops[0].kparams[KPARAMS_N_HMX], 1, "HMX kernel params");
        // The same weight bytes, addressed per N tile instead of per N row of K tiles.
        assert_eq!(
            s.tensors[0].nb[1] as usize,
            mm_hmx_nb1(HtpDataType::Q8_0, 64)
        );

        let hvx = run(280, 96, 64, TokenTile::Whole, None);
        assert_eq!(hvx.ops[0].kparams[KPARAMS_N_HMX], 0, "no option, HVX");
        assert_eq!(
            hvx.tensors[0].nb[1] as usize,
            64usize.div_ceil(32) * 32 * 34
        );

        // Too few rows for the matrix unit, unaligned dims and a tiled batch stay HVX.
        for (tokens, rows, cols, tile) in [
            (4, 96, 64, TokenTile::Whole),
            (280, 96, 48, TokenTile::Whole),
            (280, 90, 64, TokenTile::Whole),
            (280, 96, 64, TokenTile::Tiles(64)),
        ] {
            let s = run(tokens, rows, cols, tile, hmx);
            assert!(
                s.ops.iter().all(|o| o.kparams[KPARAMS_N_HMX] == 0),
                "{tokens} tokens {rows}x{cols} {tile:?}"
            );
        }
    }

    #[test]
    fn argmax_row_emits_one_argmax_over_an_f32_row_to_an_i32_scalar() {
        let mut s = RecordingSink::default();
        argmax_row(&mut s, &"b", 100, 200, 128).unwrap();
        assert_eq!(s.opcodes(), vec![OP_ARGMAX]);
        assert_eq!(s.group_ends, vec![1]);
        let (src, dst) = (s.src(0, 0), s.dst(0));
        assert_eq!(src.buf, "b");
        assert_eq!(src.offset, 100);
        assert_eq!(
            (src.ne, src.nb, src.size),
            ([128, 1, 1, 1], [4, 512, 512, 512], 128 * 4)
        );
        assert_eq!(src.dtype, HtpDataType::F32 as u32);
        assert_eq!(dst.offset, 200);
        assert_eq!((dst.ne, dst.nb, dst.size), ([1, 1, 1, 1], [4, 4, 4, 4], 4));
        assert_eq!(dst.dtype, HtpDataType::I32 as u32);
    }

    #[test]
    fn linear_m_whole_without_bias_is_one_matmul() {
        let mut s = RecordingSink::default();
        linear_m(
            &mut s,
            &"x",
            0,
            &"w",
            desc(32, 32),
            None,
            &"y",
            0,
            130,
            TokenTile::Whole,
        )
        .unwrap();
        assert_eq!(s.opcodes(), vec![OP_MULMAT]);
        assert_eq!(s.src(0, 1).ne, [32, 130, 1, 1]);
        // ne11 (token count) reaches the kernel params unchanged.
        let expect =
            build_mul_mat_kernel_params(HtpDataType::Q8_0, 32, 130, 1, 32 * 4, 8, VTCM_BUDGET);
        assert_eq!(s.ops[0].kparams, expect);
    }

    #[test]
    fn layer_norm_emits_norm_mul_add_per_tile_with_eps() {
        let mut s = RecordingSink::default();
        layer_norm(
            &mut s,
            LayerNormArgs {
                src: &"src",
                src_offset: 8,
                dst: &"dst",
                dst_offset: 16,
                weights: &"w",
                w_offset: 1024,
                b_offset: 2048,
                shape: TokenShape {
                    dim: 48,
                    n_tokens: 65,
                },
                eps: 1e-5,
                tile: TILE_64,
            },
        )
        .unwrap();
        assert_eq!(
            s.opcodes(),
            vec![OP_NORM, OP_MUL, OP_ADD, OP_NORM, OP_MUL, OP_ADD]
        );
        assert_eq!(s.ops[0].params, build_layer_norm_params(1e-5));
        assert_eq!(s.ops[3].params, build_layer_norm_params(1e-5));
        // Scale then shift rows are registered once, ahead of the tiles.
        assert_eq!((s.tensors[0].offset, s.tensors[1].offset), (1024, 2048));
        assert_eq!(s.src(0, 0).ne, [48, 64, 1, 1]);
        assert_eq!(s.src(3, 0).ne, [48, 1, 1, 1]);
        assert_eq!(s.src(3, 0).offset, 8 + 64 * 48 * 4);
        assert_eq!(s.dst(3).offset, 16 + 64 * 48 * 4);
    }

    #[test]
    fn gelu_and_residual_and_cpy_shapes() {
        let mut s = RecordingSink::default();
        let shape = TokenShape {
            dim: 7,
            n_tokens: 3,
        };
        silu(&mut s, &"b", 0, shape, TokenTile::Whole).unwrap();
        add_residual(&mut s, &"d", 0, &"s", 0, shape, TokenTile::Whole).unwrap();
        cpy_f32_to_f16(&mut s, &"a", 0, &"h", 0, shape).unwrap();
        assert_eq!(s.opcodes(), vec![OP_SILU, OP_ADD, OP_CPY]);
        // In-place ops write their own source tensor.
        assert_eq!(s.ops[0].src, s.ops[0].dst);
        assert_eq!(s.ops[1].dst, vec![s.ops[1].src[0]]);
        // dim is ne0 and n_tokens is ne1, never the reverse.
        assert_eq!(s.tensors[0].ne, [7, 3, 1, 1]);
        let cpy_dst = s.dst(2);
        assert_eq!(cpy_dst.dtype, HtpDataType::F16 as u32);
        assert_eq!(cpy_dst.ne, [7, 3, 1, 1]);
        assert_eq!(cpy_dst.size, 7 * 3 * 2);
        assert_eq!(cpy_dst.nb, [2, 14, 42, 42]);
    }

    #[test]
    fn silu_is_in_place_and_mul_row_bcast_scales_by_the_vector() {
        let mut s = RecordingSink::default();
        let shape = TokenShape {
            dim: 16,
            n_tokens: 5,
        };
        silu(&mut s, &"b", 64, shape, TokenTile::Whole).unwrap();
        mul_row_bcast(&mut s, &"b", 64, &"v", 4096, shape, TokenTile::Whole).unwrap();
        assert_eq!(s.opcodes(), vec![HtpOpCode::UnarySilu as u32, OP_MUL]);
        assert_eq!(s.ops[0].src, s.ops[0].dst);
        // The vector tensor is registered once, as a weight row of `dim`.
        let v = &s.tensors[s.ops[1].src[1] as usize];
        assert_eq!((v.buf, v.offset, v.ne), ("v", 4096, [16, 1, 1, 1]));
        assert_eq!(v.flags, HTP_TENSOR_WEIGHT);
        assert_eq!(s.src(1, 0).ne, [16, 5, 1, 1]);
        // Same row-broadcast kernel params the bias add uses.
        assert_eq!(
            s.ops[1].kparams,
            build_binary_kernel_params(16, 16, 1, 1, 1, 4, VTCM_BUDGET, 8)
        );
        assert_eq!(s.ops[1].dst, vec![s.ops[1].src[0]]);
    }

    #[test]
    fn enqueue_failure_names_the_dispatch() {
        let mut s = RecordingSink {
            fail_op_at: Some(1),
            ..Default::default()
        };
        let err = layer_norm(
            &mut s,
            LayerNormArgs {
                src: &"s",
                src_offset: 0,
                dst: &"d",
                dst_offset: 0,
                weights: &"w",
                w_offset: 0,
                b_offset: 0,
                shape: TokenShape {
                    dim: 4,
                    n_tokens: 2,
                },
                eps: 1e-5,
                tile: TokenTile::Whole,
            },
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dispatch layer_norm mul"), "{msg}");
        assert!(msg.contains("injected"), "{msg}");
    }

    #[test]
    fn every_helper_ends_exactly_one_group() {
        let mut s = RecordingSink::default();
        let shape = TokenShape {
            dim: 8,
            n_tokens: 130,
        };
        linear_m(
            &mut s,
            &"x",
            0,
            &"w",
            desc(8, 8),
            0usize,
            &"y",
            0,
            130,
            TokenTile::Tiles(64),
        )
        .unwrap();
        assert_eq!(s.group_ends, vec![6]);
        silu(&mut s, &"b", 0, shape, TokenTile::Tiles(64)).unwrap();
        assert_eq!(s.group_ends, vec![6, 9]);
        add_residual(&mut s, &"d", 0, &"s", 0, shape, TokenTile::Whole).unwrap();
        assert_eq!(s.group_ends, vec![6, 9, 10]);
        cpy_f32_to_f16(&mut s, &"a", 0, &"h", 0, shape).unwrap();
        assert_eq!(s.group_ends, vec![6, 9, 10, 11]);
    }

    /// The shaped-tensor helpers also end exactly one group each (the step-mode
    /// flush must not land between a helper's own ops).
    #[test]
    fn view_and_conv_helpers_end_exactly_one_group() {
        let mut s = RecordingSink::default();
        let shape = TokenShape {
            dim: 8,
            n_tokens: 4,
        };
        let v = |off| View::new(&"b", off, [8, 4, 1], [4, 32, 128]);
        let mut groups = 0;
        let mut check = |s: &RecordingSink| {
            groups += 1;
            assert_eq!(s.group_ends.len(), groups, "one group per helper");
        };
        sigmoid(&mut s, &"b", 0, shape, TokenTile::Whole).unwrap();
        check(&s);
        add_row_bcast(&mut s, &"b", 0, &"v", 0, shape, TokenTile::Whole).unwrap();
        check(&s);
        mul_inplace(&mut s, &"b", 0, &"b", 512, shape, TokenTile::Whole).unwrap();
        check(&s);
        matmul_f32(&mut s, v(0), v(1024), v(2048)).unwrap();
        check(&s);
        softmax(&mut s, v(0), Some(v(512)), v(1024), 0.5).unwrap();
        check(&s);
        copy_view(&mut s, v(0), v(1024)).unwrap();
        check(&s);
        add_row_bcast_to(&mut s, v(0), &"v", 0, v(1024)).unwrap();
        check(&s);
        ssm_conv(&mut s, &"w", 0, &"b", 0, &"b", 4096, 3, 8, 4, VTCM_BUDGET).unwrap();
        check(&s);
        let src = |off| ConcatSrc {
            buf: &"b",
            offset: off,
            rows: 4,
            nb0: 4,
            nb1: 16,
        };
        concat_time_inner(&mut s, src(0), src(512), &"b", 2048, 8).unwrap();
        check(&s);
        // The softmax carries the mask as its second source and the scale in
        // both the op params and the kernel params.
        let sm = s
            .ops
            .iter()
            .find(|o| o.opcode == HtpOpCode::Softmax as u32)
            .unwrap();
        assert_eq!(sm.src.len(), 2);
        assert_eq!(f32::from_bits(sm.params[0] as u32), 0.5);
        assert_eq!(f32::from_bits(sm.kparams[18] as u32), 0.5);
    }

    /// `CERA_HEXAGON_STEP` on the real queue: the helpers reuse tensor
    /// indices across their own ops (bias tensor across tiles, norm dst across
    /// Norm/Mul/Add), so stepping must flush between helpers, not between ops.
    /// Flushing after every `enqueue_op` failed here with "src[0] index 0 out
    /// of bounds (registered 0)".
    #[test]
    fn step_mode_flushes_between_helpers_not_between_ops() {
        use super::super::sys::fake;
        for step in [true, false] {
            fake::reset();
            let mut q = HexagonQueueSession::new(fake::driver()).expect("fake queue session");
            q.set_step_mode(step);
            let buf = RpcmemBuffer::alloc(fake::driver(), 1 << 20, false).unwrap();
            linear_m(
                &mut q,
                &buf,
                0,
                &buf,
                desc(96, 64),
                512usize,
                &buf,
                200_000,
                130,
                TokenTile::Tiles(64),
            )
            .unwrap_or_else(|e| panic!("step={step}: linear_m: {e}"));
            // 3 tiles x (MulMat + bias Add), one group.
            assert_eq!(q.ops_len(), if step { 0 } else { 6 }, "step={step}");
            layer_norm(
                &mut q,
                LayerNormArgs {
                    src: &buf,
                    src_offset: 0,
                    dst: &buf,
                    dst_offset: 300_000,
                    weights: &buf,
                    w_offset: 1024,
                    b_offset: 2048,
                    shape: TokenShape {
                        dim: 48,
                        n_tokens: 65,
                    },
                    eps: 1e-5,
                    tile: TILE_64,
                },
            )
            .unwrap_or_else(|e| panic!("step={step}: layer_norm: {e}"));
            assert_eq!(q.ops_len(), if step { 0 } else { 12 }, "step={step}");
        }
    }

    /// One tiny pre-norm encoder block as the Whisper encoder emits it:
    /// LN -> QKV-style linear -> F16 cast -> residual -> LN -> fc1 -> GELU -> fc2 -> residual.
    /// Pins the golden opcode sequence of the shared primitives together.
    #[test]
    fn tiny_block_golden_opcode_sequence() {
        let mut s = RecordingSink::default();
        let shape = TokenShape {
            dim: 32,
            n_tokens: 70,
        };
        let t = TILE_64;
        let ln = |s: &mut RecordingSink| {
            layer_norm(
                s,
                LayerNormArgs {
                    src: &"h",
                    src_offset: 0,
                    dst: &"n",
                    dst_offset: 0,
                    weights: &"w",
                    w_offset: 0,
                    b_offset: 128,
                    shape,
                    eps: 1e-5,
                    tile: t,
                },
            )
            .unwrap();
        };
        ln(&mut s);
        linear_m(
            &mut s,
            &"n",
            0,
            &"w",
            desc(32, 32),
            256usize,
            &"q",
            0,
            70,
            t,
        )
        .unwrap();
        cpy_f32_to_f16(&mut s, &"q", 0, &"kv", 0, shape).unwrap();
        add_residual(&mut s, &"h", 0, &"q", 0, shape, t).unwrap();
        ln(&mut s);
        linear_m(
            &mut s,
            &"n",
            0,
            &"w",
            desc(64, 32),
            256usize,
            &"m",
            0,
            70,
            t,
        )
        .unwrap();
        silu(
            &mut s,
            &"m",
            0,
            TokenShape {
                dim: 64,
                n_tokens: 70,
            },
            t,
        )
        .unwrap();
        linear_m(&mut s, &"m", 0, &"w", desc(32, 64), None, &"q", 0, 70, t).unwrap();
        add_residual(&mut s, &"h", 0, &"q", 0, shape, t).unwrap();

        let per_tile = |ops: &[u32]| ops.repeat(2);
        let mut expect = Vec::new();
        expect.extend(per_tile(&[OP_NORM, OP_MUL, OP_ADD]));
        expect.extend(per_tile(&[OP_MULMAT, OP_ADD]));
        expect.push(OP_CPY);
        expect.extend(per_tile(&[OP_ADD]));
        expect.extend(per_tile(&[OP_NORM, OP_MUL, OP_ADD]));
        expect.extend(per_tile(&[OP_MULMAT, OP_ADD]));
        expect.extend(per_tile(&[OP_SILU]));
        expect.extend(per_tile(&[OP_MULMAT]));
        expect.extend(per_tile(&[OP_ADD]));
        assert_eq!(s.opcodes(), expect);
    }

    /// The tanh GELU is seven ops per row tile through a scratch region, and its
    /// two `Scale` ops carry the constants in their parameters.
    #[test]
    fn gelu_tanh_emits_seven_ops_per_tile_with_the_scale_constants() {
        use HtpOpCode::*;
        let mut s = RecordingSink::default();
        let shape = TokenShape {
            dim: 96,
            n_tokens: 130,
        };
        gelu_tanh(&mut s, &"x", 0, &"t", 4096, 64, shape).unwrap();
        let tile = [Cpy, Mul, Scale, Mul, Scale, UnarySigmoid, Mul].map(|o| o as u32);
        let want: Vec<u32> = (0..3).flat_map(|_| tile).collect();
        assert_eq!(
            s.opcodes(),
            want,
            "130 rows through a 64-row scratch: 3 tiles"
        );
        let scales: Vec<(f32, f32)> = s
            .ops
            .iter()
            .filter(|o| o.opcode == Scale as u32)
            .map(|o| {
                (
                    f32::from_bits(o.params[0] as u32),
                    f32::from_bits(o.params[1] as u32),
                )
            })
            .collect();
        assert_eq!(
            scales[..2],
            [(0.044_715, 1.0), (GELU_TANH_SIGMOID_SCALE, 0.0)]
        );
        // The last tile is the 2-row remainder, and it works in the scratch.
        let last_copy = s.ops.iter().rposition(|o| o.opcode == Cpy as u32).unwrap();
        assert_eq!(s.dst(last_copy).ne[1], 2);
        assert_eq!(s.dst(last_copy).offset, 4096);
    }

    /// The sequence computes the tanh GELU (not the DSP's quick GELU, which is
    /// off by up to about 2% per element): done on the host in `f32`, it matches
    /// the CPU's `gelu_approx_f32` over the range activations take.
    #[test]
    fn the_gelu_tanh_sequence_matches_the_cpu_tanh_gelu() {
        let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
        let sequence = |x: f32| {
            let t = x * x;
            let t = t * 0.044_715 + 1.0;
            let t = t * x;
            let t = t * GELU_TANH_SIGMOID_SCALE;
            x * sigmoid(t)
        };
        let quick = |x: f32| x * sigmoid(1.702 * x);
        let (mut worst, mut worst_quick) = (0.0f32, 0.0f32);
        let mut x = -9.0f32;
        while x <= 9.0 {
            let want = crate::backend::cpu::gelu_approx_f32(x);
            worst = worst.max((sequence(x) - want).abs());
            worst_quick = worst_quick.max((quick(x) - want).abs());
            x += 0.01;
        }
        assert!(worst < 2e-5, "tanh sequence off by {worst}");
        // The audio adapter's CPU reference is the erf form; the NPU's tanh
        // stays within 1e-3 of it (measured 4.7e-4 at the worst point).
        let mut worst_erf = 0.0f32;
        let mut x = -9.0f32;
        while x <= 9.0 {
            let mut e = [x];
            crate::backend::cpu::gelu_erf_inplace(&mut e);
            worst_erf = worst_erf.max((sequence(x) - e[0]).abs());
            x += 0.01;
        }
        assert!(worst_erf < 1e-3, "tanh sequence is {worst_erf} from erf");
        assert!(
            worst_quick > 1e-2,
            "the quick GELU is {worst_quick} off: why this exists"
        );
    }

    /// A row width that overflows when multiplied by the element size is
    /// refused, not wrapped into an accepted one (widths come from model files).
    #[test]
    fn a_gelu_row_width_that_overflows_is_refused() {
        assert!(gelu_tmp_rows(1 << 20, usize::MAX / 2).is_err());
        assert!(gelu_tmp_rows(1 << 20, usize::MAX).is_err());
        // Widths whose product with the element size wraps to 0 and to 4 bytes.
        assert!(gelu_tmp_rows(1 << 20, 1 << 62).is_err());
        assert!(gelu_tmp_rows(1 << 20, (1 << 62) + 1).is_err());
        assert!(gelu_tmp_rows(1 << 20, (1 << 63) + 1).is_err());
        // A zero width, and a row that is exactly as wide as the scratch.
        assert!(gelu_tmp_rows(4096, 0).is_err());
        assert_eq!(gelu_tmp_rows(4096, 1024).unwrap(), 1);
        assert!(gelu_tmp_rows(4096, 1025).is_err());
        assert_eq!(gelu_tmp_rows(4096, 16).unwrap(), 64);
    }

    /// Execute the recorded elementwise ops on host memory, so a test can check
    /// what an op sequence computes rather than only which ops it emits.
    fn interpret(s: &RecordingSink, mem: &mut std::collections::HashMap<&'static str, Vec<f32>>) {
        let at = |t: &testing::RecTensor, i0: usize, i1: usize| {
            (t.offset + i0 * t.nb[0] as usize + i1 * t.nb[1] as usize) / 4
        };
        for (n, op) in s.ops.iter().enumerate() {
            let dst = s.dst(n).clone();
            for i1 in 0..dst.ne[1] as usize {
                for i0 in 0..dst.ne[0] as usize {
                    let a = {
                        let t = s.src(n, 0);
                        mem[t.buf][at(t, i0, i1)]
                    };
                    let v = match HtpOpCode::from_u32(op.opcode).unwrap() {
                        HtpOpCode::Cpy => a,
                        HtpOpCode::Mul => {
                            let t = s.src(n, 1);
                            a * mem[t.buf][at(t, i0, i1)]
                        }
                        HtpOpCode::Scale => {
                            a * f32::from_bits(op.params[0] as u32)
                                + f32::from_bits(op.params[1] as u32)
                        }
                        HtpOpCode::UnarySigmoid => 1.0 / (1.0 + (-a).exp()),
                        other => panic!("not interpreted: {other:?}"),
                    };
                    let i = at(&dst, i0, i1);
                    mem.get_mut(dst.buf).unwrap()[i] = v;
                }
            }
        }
    }

    /// The emitted ops, run on the host, compute the CPU's tanh GELU in place
    /// (130 rows through a 64-row scratch, so the remainder tile is covered).
    #[test]
    fn the_emitted_gelu_tanh_ops_compute_the_cpu_gelu() {
        let (dim, n_tokens, tmp_rows) = (96usize, 130usize, 64usize);
        let mut s = RecordingSink::default();
        let shape = TokenShape { dim, n_tokens };
        let tmp_off = 512;
        gelu_tanh(&mut s, &"x", 0, &"t", tmp_off, tmp_rows, shape).unwrap();
        let xs: Vec<f32> = (0..dim * n_tokens)
            .map(|i| (i as f32 * 0.037).sin() * 6.0)
            .collect();
        let mut mem = std::collections::HashMap::new();
        mem.insert("x", xs.clone());
        mem.insert("t", vec![0.0f32; tmp_off / 4 + dim * tmp_rows]);
        interpret(&s, &mut mem);
        for (i, (&got, &x)) in mem["x"].iter().zip(&xs).enumerate() {
            let want = crate::backend::cpu::gelu_approx_f32(x);
            assert!(
                (got - want).abs() < 2e-5,
                "element {i}: {got} vs {want} at x={x}"
            );
        }
    }
}

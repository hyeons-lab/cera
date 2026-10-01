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
    HexagonWeightDesc, HexagonWeightFormat, HtpDataType, HtpOpCode, RpcmemBuffer,
    build_binary_kernel_params, build_layer_norm_params, build_mul_mat_kernel_params,
    build_unary_kernel_params,
};
use crate::session::CeraError;

/// VTCM budget handed to every kernel-param builder (8 MB on Snapdragon 8 Elite).
const VTCM_BUDGET: usize = 8 * 1024 * 1024;

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
    let (w_dtype, block_bytes, tile_size) = match w_desc.format {
        HexagonWeightFormat::RepackedQ8_0 => (HtpDataType::Q8_0, 34usize, 32 * 34usize),
        HexagonWeightFormat::RepackedQ4_0 => (HtpDataType::Q4_0, 18usize, 32 * 18usize),
    };
    let (cols, rows) = (w_desc.cols, w_desc.rows);
    let tiled_row_bytes = cols.div_ceil(32) * tile_size;
    let w_tot = rows.div_ceil(32) * tiled_row_bytes;
    let w_ti = session.add_tensor(
        weights,
        w_desc.offset,
        w_desc.size_bytes,
        HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
        w_dtype as u32,
        [cols as u32, rows as u32, 1, 1],
        [
            block_bytes as u32,
            tiled_row_bytes as u32,
            w_tot as u32,
            w_tot as u32,
        ],
    )?;
    let b_ti = match bias_offset.into() {
        Some(b_off) => Some(add_f32_vector(session, weights, b_off, rows)?),
        None => None,
    };

    for (start, run) in token_runs(n_tokens, tile) {
        let x_ti = add_f32_rows(session, x, x_offset + start * cols * 4, cols, run)?;
        let dst_ti = add_f32_rows(session, dst, dst_offset + start * rows * 4, rows, run)?;
        let kparams = build_mul_mat_kernel_params(
            w_dtype,
            cols,
            run as u32,
            1,
            rows * 4,
            session.dsp_threads(),
            VTCM_BUDGET,
        );
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

/// In-place GELU: `buf = gelu(buf)`.
pub(crate) fn gelu<S: OpSink>(
    session: &mut S,
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
            .enqueue_op(
                HtpOpCode::UnaryGelu as u32,
                &[ti],
                &[ti],
                [0i32; 16],
                kparams,
            )
            .map_err(|e| op_err("gelu", e))?;
    }
    session.end_group().map_err(|e| op_err("gelu", e))
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
    const OP_NORM: u32 = HtpOpCode::Norm as u32;
    const OP_MULMAT: u32 = HtpOpCode::MulMat as u32;
    const OP_GELU: u32 = HtpOpCode::UnaryGelu as u32;
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
        gelu(&mut s, &"b", 0, shape, TokenTile::Whole).unwrap();
        add_residual(&mut s, &"d", 0, &"s", 0, shape, TokenTile::Whole).unwrap();
        cpy_f32_to_f16(&mut s, &"a", 0, &"h", 0, shape).unwrap();
        assert_eq!(s.opcodes(), vec![OP_GELU, OP_ADD, OP_CPY]);
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
        gelu(&mut s, &"b", 0, shape, TokenTile::Tiles(64)).unwrap();
        assert_eq!(s.group_ends, vec![6, 9]);
        add_residual(&mut s, &"d", 0, &"s", 0, shape, TokenTile::Whole).unwrap();
        assert_eq!(s.group_ends, vec![6, 9, 10]);
        cpy_f32_to_f16(&mut s, &"a", 0, &"h", 0, shape).unwrap();
        assert_eq!(s.group_ends, vec![6, 9, 10, 11]);
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
        gelu(
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
        expect.extend(per_tile(&[OP_GELU]));
        expect.extend(per_tile(&[OP_MULMAT]));
        expect.extend(per_tile(&[OP_ADD]));
        assert_eq!(s.opcodes(), expect);
    }
}

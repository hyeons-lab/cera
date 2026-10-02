//! Op emitters: one `dispatch_*` associated function per DSP operation.

use super::*;

impl HexagonLfmModel {
    /// Contiguous f32 vector descriptor (`[dim,1,1,1]`): the one spelling of
    /// the vec shape+strides all elementwise dispatches share.
    fn add_f32_vec(
        session: &mut HexagonQueueSession,
        buf: &RpcmemBuffer,
        offset: usize,
        dim: usize,
        flags: u32,
    ) -> Result<u16, CeraError> {
        session.add_tensor(
            buf,
            offset,
            dim * 4,
            flags,
            HtpDataType::F32 as u32,
            [dim as u32, 1, 1, 1],
            [4, (dim * 4) as u32, (dim * 4) as u32, (dim * 4) as u32],
        )
    }

    /// Enqueue with op-name context: the one spelling of the
    /// `enqueue_op` + `dispatch_*:` label every dispatch shares. Every
    /// `dispatch_*` registers its own tensors and enqueues exactly one op
    /// through here, so this is also the op-group boundary that
    /// `CERA_HEXAGON_STEP` flushes at (`end_group`).
    fn enqueue_labeled(
        session: &mut HexagonQueueSession,
        label: &str,
        opcode: u32,
        src: &[u16],
        dst: &[u16],
        params: [i32; 16],
        kernel_params: [i32; 32],
    ) -> Result<(), CeraError> {
        session
            .enqueue_op(opcode, src, dst, params, kernel_params)
            .and_then(|()| session.end_group())
            .map_err(|e| CeraError::Backend(format!("{label}: {e}")))
    }

    pub(super) fn dispatch_argmax(
        session: &mut HexagonQueueSession,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
        vocab_size: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let in_ti = session.add_tensor(
            in_act,
            in_offset,
            n_rows * vocab_size * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [vocab_size as u32, n_rows as u32, 1, 1],
            [
                4,
                (vocab_size * 4) as u32,
                (n_rows * vocab_size * 4) as u32,
                (n_rows * vocab_size * 4) as u32,
            ],
        )?;
        let out_ti = session.add_tensor(
            out_act,
            out_offset,
            n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_rows as u32, 1, 1, 1],
            [
                4,
                (n_rows * 4) as u32,
                (n_rows * 4) as u32,
                (n_rows * 4) as u32,
            ],
        )?;
        Self::enqueue_labeled(
            session,
            "dispatch_argmax",
            HtpOpCode::Argmax as u32,
            &[in_ti],
            &[out_ti],
            [0i32; 16],
            [0i32; 32],
        )?;
        Ok(())
    }

    pub(super) fn dispatch_mul(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src0_flags: u32,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        src1_flags: u32,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, src0_flags)?;
        let src1_ti = Self::add_f32_vec(session, src1, src1_offset, dim, src1_flags)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            1,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_mul",
            HtpOpCode::Mul as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Row-wise multiply with strided inputs (llama's strided-view MUL):
    /// `dst[r, c] = a[r, c] * b[r, c]` over `[dim, n_rows]`, reading A/B
    /// with byte row strides `a_row_stride`/`b_row_stride`. The DSP reads the
    /// strides from the tensor descriptors. Used for the conv `b * x` and
    /// gate products straight out of the strided `in_proj` thirds.
    pub(super) fn dispatch_mul_m_strided(
        session: &mut HexagonQueueSession,
        a_buf: &RpcmemBuffer,
        a_offset: usize,
        a_flags: u32,
        b_buf: &RpcmemBuffer,
        b_offset: usize,
        b_flags: u32,
        dst_buf: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        n_rows: usize,
        a_row_stride: usize,
        b_row_stride: usize,
    ) -> Result<(), CeraError> {
        let row = dim * 4;
        let span = |stride: usize| n_rows.saturating_sub(1) * stride + row;
        let a_span = span(a_row_stride);
        let b_span = span(b_row_stride);
        let dst_bytes = row * n_rows;
        let ne = [dim as u32, n_rows as u32, 1, 1];
        let a_ti = session.add_tensor(
            a_buf,
            a_offset,
            a_span,
            a_flags,
            HtpDataType::F32 as u32,
            ne,
            [4, a_row_stride as u32, a_span as u32, a_span as u32],
        )?;
        let b_ti = session.add_tensor(
            b_buf,
            b_offset,
            b_span,
            b_flags,
            HtpDataType::F32 as u32,
            ne,
            [4, b_row_stride as u32, b_span as u32, b_span as u32],
        )?;
        let dst_ti = session.add_tensor(
            dst_buf,
            dst_offset,
            dst_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            [4, row as u32, dst_bytes as u32, dst_bytes as u32],
        )?;
        let params = [0i32; 16];
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            n_rows,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_mul_m_strided",
            HtpOpCode::Mul as u32,
            &[a_ti, b_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Dim-0 CONCAT of two 2D f32 tensors:
    /// `[s0_rows, dim] + [s1_rows, dim] -> [s0_rows + s1_rows, dim]`.
    /// `params[0]` is the concat dim; kparams are zero (the DSP sizes VTCM
    /// itself). The second source may be a transposed view (`s1_nb0 >
    /// s1_nb1`), which takes the DSP's specialized 2D-transposed worker:
    /// the conv state prepend (`[s0; s1] + bx-as-[m, hs]`).
    pub(super) fn dispatch_concat_2d(
        session: &mut HexagonQueueSession,
        s0_buf: &RpcmemBuffer,
        s0_offset: usize,
        s0_rows: usize,
        s1_buf: &RpcmemBuffer,
        s1_offset: usize,
        s1_rows: usize,
        s1_nb0: usize,
        s1_nb1: usize,
        dst_buf: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let s0_span = s0_rows * dim * 4;
        let s1_span = s1_rows.saturating_sub(1) * s1_nb0 + dim.saturating_sub(1) * s1_nb1 + 4;
        let dst_rows = s0_rows + s1_rows;
        let dst_span = dst_rows * dim * 4;
        let s0_ti = session.add_tensor(
            s0_buf,
            s0_offset,
            s0_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [s0_rows as u32, dim as u32, 1, 1],
            [4, (s0_rows * 4) as u32, s0_span as u32, s0_span as u32],
        )?;
        let s1_ti = session.add_tensor(
            s1_buf,
            s1_offset,
            s1_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [s1_rows as u32, dim as u32, 1, 1],
            [s1_nb0 as u32, s1_nb1 as u32, s1_span as u32, s1_span as u32],
        )?;
        let dst_ti = session.add_tensor(
            dst_buf,
            dst_offset,
            dst_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [dst_rows as u32, dim as u32, 1, 1],
            [4, (dst_rows * 4) as u32, dst_span as u32, dst_span as u32],
        )?;
        let mut params = [0i32; 16];
        params[0] = 0; // concat dim
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_concat_2d",
            HtpOpCode::Concat as u32,
            &[s0_ti, s1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_add(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, HTP_TENSOR_COMPUTE)?;
        let src1_ti = Self::add_f32_vec(session, src1, src1_offset, dim, HTP_TENSOR_COMPUTE)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            1,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_add",
            HtpOpCode::Add as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_add_with_flags(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src0_flags: u32,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        src1_flags: u32,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, src0_flags)?;
        let src1_ti = Self::add_f32_vec(session, src1, src1_offset, dim, src1_flags)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            1,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_add_with_flags",
            HtpOpCode::Add as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_mul_scalar(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1_scalar: &RpcmemBuffer,
        src1_scalar_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        Self::dispatch_scalar_binary(
            session,
            HtpOpCode::Mul,
            "dispatch_mul_scalar",
            src0,
            src0_offset,
            src1_scalar,
            src1_scalar_offset,
            dst,
            dst_offset,
            dim,
        )
    }

    /// `dst[i] = src0[i] <op> scalar` with the scalar a single f32 read from
    /// `src1_scalar`. Mul (routed-expert combine) and Div (top-k weight
    /// renormalization) share this builder: both are the same binary kernel.
    fn dispatch_scalar_binary(
        session: &mut HexagonQueueSession,
        opcode: HtpOpCode,
        label: &str,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1_scalar: &RpcmemBuffer,
        src1_scalar_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, HTP_TENSOR_COMPUTE)?;
        let src1_ti = session.add_tensor(
            src1_scalar,
            src1_scalar_offset,
            4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [1, 1, 1, 1],
            [4, 4, 4, 4],
        )?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams =
            build_binary_kernel_params(dim, 1, 1, 1, 1, 4, 8 * 1024 * 1024, session.dsp_threads());
        Self::enqueue_labeled(
            session,
            label,
            opcode as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Same-shape elementwise binary op over two `dim`-element vectors.
    fn dispatch_binary_vec(
        session: &mut HexagonQueueSession,
        opcode: HtpOpCode,
        label: &str,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = Self::add_f32_vec(session, src0, src0_offset, dim, HTP_TENSOR_COMPUTE)?;
        let src1_ti = Self::add_f32_vec(session, src1, src1_offset, dim, HTP_TENSOR_COMPUTE)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            1,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            label,
            opcode as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            [0i32; 16],
            kparams,
        )
    }

    /// Fill each token's renormalization slot with the divisor floor. The slots
    /// live in the scratch buffer and nothing else writes them, so this runs
    /// once at load.
    pub(super) fn init_moe_renorm_scratch(scratch: &RpcmemBuffer, so: &ScratchOffsets) {
        if so.moe_renorm == 0 {
            return;
        }
        for token in 0..PREFILL_MAX_ROWS {
            unsafe {
                *(scratch
                    .as_mut_ptr()
                    .add(so.moe_renorm + token * MOE_RENORM_SLOT_BYTES + 4)
                    as *mut f32) = MOE_DENOM_FLOOR;
            }
        }
        scratch.flush_cpu_cache(so.moe_renorm, PREFILL_MAX_ROWS * MOE_RENORM_SLOT_BYTES);
    }

    /// Emit the sum of one token's `n_used` selected weights and the clamped
    /// divisor `max(sum, MOE_DENOM_FLOOR)`, returning the divisor's scratch
    /// offset. `max` is `sum + relu(floor - sum)`: Add, Sub and UnaryRelu are
    /// already in the op set, so no new DSP op (Max/Clamp/SumRows) is needed.
    pub(super) fn dispatch_moe_renorm_denominator(
        session: &mut HexagonQueueSession,
        scratch: &RpcmemBuffer,
        so: &ScratchOffsets,
        token_idx: usize,
        n_used: usize,
    ) -> Result<usize, CeraError> {
        let weight = |e: usize| so.moe_selected_weights + (token_idx * n_used + e) * 4;
        let slot = so.moe_renorm + token_idx * MOE_RENORM_SLOT_BYTES;
        let (sum, floor, below, relu, denom) = (slot, slot + 4, slot + 8, slot + 12, slot + 16);
        if n_used == 1 {
            Self::dispatch_cpy(session, scratch, weight(0), scratch, sum, 1)?;
        } else {
            Self::dispatch_binary_vec(
                session,
                HtpOpCode::Add,
                "dispatch_moe_renorm_sum",
                scratch,
                weight(0),
                scratch,
                weight(1),
                scratch,
                sum,
                1,
            )?;
            for e in 2..n_used {
                Self::dispatch_binary_vec(
                    session,
                    HtpOpCode::Add,
                    "dispatch_moe_renorm_sum",
                    scratch,
                    sum,
                    scratch,
                    weight(e),
                    scratch,
                    sum,
                    1,
                )?;
            }
        }
        Self::dispatch_binary_vec(
            session,
            HtpOpCode::Sub,
            "dispatch_moe_renorm_floor_gap",
            scratch,
            floor,
            scratch,
            sum,
            scratch,
            below,
            1,
        )?;
        Self::dispatch_unary(
            session,
            scratch,
            below,
            scratch,
            relu,
            1,
            HtpOpCode::UnaryRelu,
        )?;
        Self::dispatch_binary_vec(
            session,
            HtpOpCode::Add,
            "dispatch_moe_renorm_denom",
            scratch,
            sum,
            scratch,
            relu,
            scratch,
            denom,
            1,
        )?;
        Ok(denom)
    }

    pub(super) fn dispatch_unary(
        session: &mut HexagonQueueSession,
        in_buf: &RpcmemBuffer,
        in_offset: usize,
        out_buf: &RpcmemBuffer,
        out_offset: usize,
        dim: usize,
        opcode: HtpOpCode,
    ) -> Result<(), CeraError> {
        let in_ti = Self::add_f32_vec(session, in_buf, in_offset, dim, HTP_TENSOR_COMPUTE)?;
        let out_ti = Self::add_f32_vec(session, out_buf, out_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams =
            build_unary_kernel_params(dim, 1, 0, 8 * 1024 * 1024, session.dsp_threads(), false);
        Self::enqueue_labeled(
            session,
            "dispatch_unary",
            opcode as u32,
            &[in_ti],
            &[out_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// [`Self::dispatch_unary`] over `[dim, n_rows]`: one row per token, so the
    /// DSP streams rows through VTCM instead of holding a flat `dim * n_rows`
    /// vector, which fails with `VtcmTooSmall` once a prefill chunk is a few
    /// hundred rows wide.
    pub(super) fn dispatch_unary_rows(
        session: &mut HexagonQueueSession,
        in_buf: &RpcmemBuffer,
        in_offset: usize,
        out_buf: &RpcmemBuffer,
        out_offset: usize,
        dim: usize,
        n_rows: usize,
        opcode: HtpOpCode,
    ) -> Result<(), CeraError> {
        let bytes = dim * n_rows * 4;
        let ne = [dim as u32, n_rows as u32, 1, 1];
        let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];
        let in_ti = session.add_tensor(
            in_buf,
            in_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let out_ti = session.add_tensor(
            out_buf,
            out_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let params = [0i32; 16];
        let kparams = build_unary_kernel_params(
            dim,
            n_rows,
            0,
            8 * 1024 * 1024,
            session.dsp_threads(),
            false,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_unary_rows",
            opcode as u32,
            &[in_ti],
            &[out_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_argsort(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src_ti = Self::add_f32_vec(session, src, src_offset, dim, HTP_TENSOR_COMPUTE)?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            dim * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [dim as u32, 1, 1, 1],
            [4, (dim * 4) as u32, (dim * 4) as u32, (dim * 4) as u32],
        )?;
        let mut params = [0i32; 16];
        params[0] = 1; // GGML_SORT_ORDER_DESC = 1 (descending)
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_argsort",
            HtpOpCode::Argsort as u32,
            &[src_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_get_rows(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        n_rows_in: usize,
        n_rows_out: usize,
    ) -> Result<(), CeraError> {
        let src0_ti = session.add_tensor(
            src0,
            src0_offset,
            n_rows_in * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [1, n_rows_in as u32, 1, 1],
            [4, 4, (n_rows_in * 4) as u32, (n_rows_in * 4) as u32],
        )?;
        let src1_ti = session.add_tensor(
            src1,
            src1_offset,
            n_rows_out * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_rows_out as u32, 1, 1, 1],
            [
                4,
                (n_rows_out * 4) as u32,
                (n_rows_out * 4) as u32,
                (n_rows_out * 4) as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            n_rows_out * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [1, n_rows_out as u32, 1, 1],
            [4, 4, (n_rows_out * 4) as u32, (n_rows_out * 4) as u32],
        )?;
        let params = [0i32; 16];
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_get_rows",
            HtpOpCode::GetRows as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    fn dispatch_mul_mat_id(
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        w: &HexagonStackedWeight,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        in_rows: usize,
        ids_act: &RpcmemBuffer,
        ids_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
        n_expert_used: usize,
        dsp_threads: u32,
        vtcm_budget: usize,
    ) -> Result<(), CeraError> {
        let pad32 = |x: usize| (x + 31) & !31;
        let k = w.in_dim;
        let m = w.out_dim;
        let ne0 = pad32(k);
        let ne1 = pad32(m);
        let tiled_row_bytes = (ne0 / 32) * w.tile_size;
        let w_nb1 = tiled_row_bytes;
        let nb0 = w.block_bytes as u32;
        let nb1 = w_nb1 as u32;
        let nb2 = w.expert_stride as u32;
        let nb3 = (w.n_expert * w.expert_stride) as u32;

        let src0_ti = session.add_tensor(
            weights,
            w.offset,
            w.size,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w.wire_dtype as u32,
            [ne0 as u32, ne1 as u32, w.n_expert as u32, 1],
            [nb0, nb1, nb2, nb3],
        )?;

        let src1_ti = session.add_tensor(
            in_act,
            in_offset,
            k * in_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [k as u32, in_rows as u32, 1, 1],
            [
                4,
                (k * 4) as u32,
                (k * in_rows * 4) as u32,
                (k * in_rows * 4) as u32,
            ],
        )?;

        let src2_ti = session.add_tensor(
            ids_act,
            ids_offset,
            n_expert_used * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_expert_used as u32, 1, 1, 1],
            [
                4,
                (n_expert_used * 4) as u32,
                (n_expert_used * 4) as u32,
                (n_expert_used * 4) as u32,
            ],
        )?;

        let dst_ti = session.add_tensor(
            out_act,
            out_offset,
            m * n_expert_used * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [m as u32, n_expert_used as u32, 1, 1],
            [
                4,
                (m * 4) as u32,
                (m * n_expert_used * 4) as u32,
                (m * n_expert_used * 4) as u32,
            ],
        )?;

        // The same host-precomputed HVX matmul params as `dispatch_mul_mat`:
        // the DSP rejects an all-zero set (`InvalParams`). `in_rows` activation
        // rows per expert slice, one batch.
        let kparams = build_mul_mat_kernel_params(
            w.wire_dtype,
            k,
            in_rows as u32,
            1,
            m * 4,
            dsp_threads,
            vtcm_budget,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat_id",
            HtpOpCode::MulMatId as u32,
            &[src0_ti, src1_ti, src2_ti],
            &[dst_ti],
            [0i32; 16],
            kparams,
        )?;
        Ok(())
    }

    /// The buffer holding a stacked expert weight: its layer's expert buffer
    /// when the experts are paged, else the shared weights buffer.
    fn stacked_buf(&self, w: &HexagonStackedWeight) -> &RpcmemBuffer {
        match (w.group, &self.pager) {
            (Some(group), Some(pager)) => {
                let buf = pager.buf(group);
                // `emit_ffn_block` pages the layer in first; an unmapped
                // buffer here would fault the whole batch on the DSP.
                debug_assert!(buf.is_mapped(), "expert group {group} used while unmapped");
                buf
            }
            _ => &self.weights_buf,
        }
    }

    /// Map layer `moe`'s expert buffer before ops that read it are queued.
    ///
    /// Making room flushes the pending batch and waits for the DSP, since a
    /// buffer cannot be unmapped under a batch that reads it.
    pub(super) fn page_in_experts(
        &self,
        session: &mut HexagonQueueSession,
        moe: &HexagonMoeFfn,
    ) -> Result<(), CeraError> {
        let (Some(group), Some(pager)) = (moe.gate.group, &self.pager) else {
            return Ok(());
        };
        pager.page_in(group, || {
            session.flush()?;
            session.quiesce()
        })
    }

    pub(super) fn dispatch_moe_token(
        &self,
        session: &mut HexagonQueueSession,
        moe: &HexagonMoeFfn,
        scratch: &RpcmemBuffer,
        in_act_offset: usize,
        out_act_offset: usize,
        token_idx: usize,
    ) -> Result<(), CeraError> {
        let so = &self.scratch_offsets;
        let hs = self.config.hidden_size;
        let n_exp = moe.n_expert;
        let n_used = moe.n_expert_used;
        let ff = moe.expert_ff_len;

        let router_logits_off = so.moe_router_logits + token_idx * n_exp * 4;
        let probs_off = so.moe_probs + token_idx * n_exp * 4;
        let biased_probs_off = so.moe_biased_probs + token_idx * n_exp * 4;
        let selected_ids_off = so.moe_selected_ids + token_idx * n_exp * 4;
        let selected_weights_off = so.moe_selected_weights + token_idx * n_used * 4;
        let gate_off = so.moe_gate + token_idx * n_used * ff * 4;
        let up_off = so.moe_up + token_idx * n_used * ff * 4;
        let swiglu_off = so.moe_swiglu + token_idx * n_used * ff * 4;
        let down_off = so.moe_down + token_idx * n_used * hs * 4;
        let temp_weighted_off = so.moe_temp_weighted + token_idx * hs * 4;

        // 1. Router GEMV: [hs, 1] * [hs, n_exp] -> [n_exp, 1]
        self.dispatch_mul_mat_m(
            session,
            &self.weights_buf,
            &moe.router,
            scratch,
            in_act_offset,
            scratch,
            router_logits_off,
            1,
        )?;

        // 2. Sigmoid activation: probs = sigmoid(router_logits)
        Self::dispatch_unary(
            session,
            scratch,
            router_logits_off,
            scratch,
            probs_off,
            n_exp,
            HtpOpCode::UnarySigmoid,
        )?;

        // 3. Biased probs: biased_probs = probs + exp_probs_b
        Self::dispatch_add_with_flags(
            session,
            scratch,
            probs_off,
            HTP_TENSOR_COMPUTE,
            &self.weights_buf,
            moe.exp_probs_b_offset,
            HTP_TENSOR_WEIGHT,
            scratch,
            biased_probs_off,
            n_exp,
        )?;

        // 4. Top-k selection: argsort in descending order
        Self::dispatch_argsort(
            session,
            scratch,
            biased_probs_off,
            scratch,
            selected_ids_off,
            n_exp,
        )?;

        // 5. Extract unbiased weights for the selected top-k experts
        Self::dispatch_get_rows(
            session,
            scratch,
            probs_off,
            scratch,
            selected_ids_off,
            scratch,
            selected_weights_off,
            n_exp,
            n_used,
        )?;

        // 5b. Renormalization divisor for the selected weights: the CPU
        // reference (`select_experts`, llama.cpp `build_moe_ffn`) divides them
        // by their sum clamped to 2^-14. Sum and clamp run here; the division
        // is applied once to the combined output in step 11.
        let denom_off =
            Self::dispatch_moe_renorm_denominator(session, scratch, so, token_idx, n_used)?;

        // 6. Gate projection: [hs, 1] * stacked_gate -> [ff, n_used]
        Self::dispatch_mul_mat_id(
            session,
            self.stacked_buf(&moe.gate),
            &moe.gate,
            scratch,
            in_act_offset,
            1,
            scratch,
            selected_ids_off,
            scratch,
            gate_off,
            n_used,
            session.dsp_threads(),
            self.vtcm_budget,
        )?;

        // 7. Up projection: [hs, 1] * stacked_up -> [ff, n_used]
        Self::dispatch_mul_mat_id(
            session,
            self.stacked_buf(&moe.up),
            &moe.up,
            scratch,
            in_act_offset,
            1,
            scratch,
            selected_ids_off,
            scratch,
            up_off,
            n_used,
            session.dsp_threads(),
            self.vtcm_budget,
        )?;

        // 8. FFN activation: swiglu([ff, n_used])
        Self::dispatch_glu(
            session,
            scratch,
            gate_off,
            scratch,
            up_off,
            scratch,
            swiglu_off,
            ff,
            n_used,
            self.activation,
        )?;

        // 9. Down projection: [ff, n_used] * stacked_down -> [hs, n_used]
        Self::dispatch_mul_mat_id(
            session,
            self.stacked_buf(&moe.down),
            &moe.down,
            scratch,
            swiglu_off,
            n_used,
            scratch,
            selected_ids_off,
            scratch,
            down_off,
            n_used,
            session.dsp_threads(),
            self.vtcm_budget,
        )?;

        // 10. Combine expert outputs: sum_{e=0..n_used} (down[e] * weight[e])
        for e in 0..n_used {
            let expert_down_off = down_off + e * hs * 4;
            let weight_scalar_off = selected_weights_off + e * 4;
            if e == 0 {
                Self::dispatch_mul_scalar(
                    session,
                    scratch,
                    expert_down_off,
                    scratch,
                    weight_scalar_off,
                    scratch,
                    out_act_offset,
                    hs,
                )?;
            } else {
                Self::dispatch_mul_scalar(
                    session,
                    scratch,
                    expert_down_off,
                    scratch,
                    weight_scalar_off,
                    scratch,
                    temp_weighted_off,
                    hs,
                )?;
                Self::dispatch_add(
                    session,
                    scratch,
                    out_act_offset,
                    scratch,
                    temp_weighted_off,
                    scratch,
                    out_act_offset,
                    hs,
                )?;
            }
        }

        // 11. Renormalize: sum_e(down[e] * w[e]) / max(sum_e w[e], 2^-14)
        // equals sum_e(down[e] * w[e] / denom), the CPU reference combine.
        Self::dispatch_scalar_binary(
            session,
            HtpOpCode::Div,
            "dispatch_moe_renorm_div",
            scratch,
            out_act_offset,
            scratch,
            denom_off,
            scratch,
            out_act_offset,
            hs,
        )?;

        Ok(())
    }

    /// In-place row-broadcast elementwise op over `n_rows` rows of `dim` f32s:
    /// `act[r, :] = act[r, :] <op> vec[:]`, with `vec` an F32 vector in the
    /// weights buffer. This is the ROW_BCAST binary kernel (`ne11 == 1`), the
    /// same shape the encoder models' bias add uses; projection biases use Add
    /// and the Granite residual multiplier uses Mul with a constant vector.
    fn dispatch_row_bcast(
        session: &mut HexagonQueueSession,
        opcode: HtpOpCode,
        label: &str,
        act: &RpcmemBuffer,
        act_offset: usize,
        vec_buf: &RpcmemBuffer,
        vec_offset: usize,
        dim: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let bytes = dim * n_rows * 4;
        let act_ti = session.add_tensor(
            act,
            act_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [dim as u32, n_rows as u32, 1, 1],
            [4, (dim * 4) as u32, bytes as u32, bytes as u32],
        )?;
        let vec_ti = Self::add_f32_vec(session, vec_buf, vec_offset, dim, HTP_TENSOR_WEIGHT)?;
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            1,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            label,
            opcode as u32,
            &[act_ti, vec_ti],
            &[act_ti],
            [0i32; 16],
            kparams,
        )
    }

    /// Add the F32 bias at `bias_offset` of the weights buffer to `n_rows` rows.
    pub(super) fn dispatch_bias_add(
        &self,
        session: &mut HexagonQueueSession,
        act: &RpcmemBuffer,
        act_offset: usize,
        bias_offset: usize,
        dim: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        Self::dispatch_row_bcast(
            session,
            HtpOpCode::Add,
            "dispatch_bias_add",
            act,
            act_offset,
            &self.weights_buf,
            bias_offset,
            dim,
            n_rows,
        )
    }

    /// Scale a block output by the residual multiplier before its residual add
    /// (no-op unless the model carries one).
    pub(super) fn dispatch_residual_scale(
        &self,
        session: &mut HexagonQueueSession,
        act: &RpcmemBuffer,
        act_offset: usize,
        dim: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let Some(vec_offset) = self.dense.residual_vec_offset else {
            return Ok(());
        };
        Self::dispatch_row_bcast(
            session,
            HtpOpCode::Mul,
            "dispatch_residual_scale",
            act,
            act_offset,
            &self.weights_buf,
            vec_offset,
            dim,
            n_rows,
        )
    }

    /// M-row (prefill) Add: `dst[m, :] = a[m, :] + b[m, :]` over `n_rows`
    /// contiguous rows of `dim` f32s.
    pub(super) fn dispatch_add_m(
        session: &mut HexagonQueueSession,
        src0: &RpcmemBuffer,
        src0_offset: usize,
        src1: &RpcmemBuffer,
        src1_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let bytes = dim * n_rows * 4;
        let ne = [dim as u32, n_rows as u32, 1, 1];
        let nb = [4, (dim * 4) as u32, bytes as u32, bytes as u32];
        let src0_ti = session.add_tensor(
            src0,
            src0_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let src1_ti = session.add_tensor(
            src1,
            src1_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let params = [0i32; 16];
        let kparams = build_binary_kernel_params(
            dim,
            dim,
            n_rows,
            1,
            1,
            4,
            8 * 1024 * 1024,
            session.dsp_threads(),
        );
        Self::enqueue_labeled(
            session,
            "dispatch_add_m",
            HtpOpCode::Add as u32,
            &[src0_ti, src1_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_mul_mat(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        w: &HexagonWeight,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
    ) -> Result<(), CeraError> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        // Tiled wire format: dims padded to 32, row stride = K tiles wide.
        let ne0 = in_dim.div_ceil(32) * 32;
        let ne1 = out_dim.div_ceil(32) * 32;
        let tiled_row_bytes = (ne0 / 32) * w.tile_size;
        let w_ti = session.add_tensor(
            weights,
            w.offset,
            (ne1 / 32) * tiled_row_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w.wire_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                w.block_bytes as u32,
                tiled_row_bytes as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
            ],
        )?;
        let in_ti = session.add_tensor(
            in_act,
            in_offset,
            in_dim * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [in_dim as u32, 1, 1, 1],
            [
                4,
                (in_dim * 4) as u32,
                (in_dim * 4) as u32,
                (in_dim * 4) as u32,
            ],
        )?;
        let out_ti = session.add_tensor(
            out_act,
            out_offset,
            out_dim * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [out_dim as u32, 1, 1, 1],
            [
                4,
                (out_dim * 4) as u32,
                (out_dim * 4) as u32,
                (out_dim * 4) as u32,
            ],
        )?;
        let wtype = w.wire_dtype;
        let params = [0i32; 16];
        let kparams = build_mul_mat_kernel_params(
            wtype,
            in_dim,
            1,
            1,
            out_dim * 4,
            session.dsp_threads(),
            self.vtcm_budget,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat",
            HtpOpCode::MulMat as u32,
            &[w_ti, in_ti],
            &[out_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-row (prefill) GEMM: `out[m, :] = in[m, :] @ W` over `n_rows`
    /// contiguous activation rows (HVX path; weights stream from DDR once
    /// per op, so chunk rows to fit VTCM).
    pub(super) fn dispatch_mul_mat_m(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        w: &HexagonWeight,
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offset: usize,
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let in_dim = w.in_dim;
        let out_dim = w.out_dim;
        // Tiled wire format: dims padded to 32, row stride = K tiles wide.
        let ne0 = in_dim.div_ceil(32) * 32;
        let ne1 = out_dim.div_ceil(32) * 32;
        let wtype = w.wire_dtype;
        // HMX first (M >= 5 prefill), HVX fallback: mirrors ggml's
        // HMX-then-HVX selection. The same repacked weights feed both, but
        // the dim-1 stride differs: HVX walks N rows of K tiles while HMX
        // addresses N-tile starts as `nc * nb[1]`, so HMX needs the
        // tiled row size (`ggml_hexagon_tiled_row_size`).
        let hmx_kparams = if self.use_hmx && mm_is_hmx_eligible(wtype, ne0, ne1, n_rows) {
            build_hmx_mm_kernel_params(
                wtype,
                ne0,
                ne1,
                n_rows.next_multiple_of(32),
                n_rows,
                session.dsp_threads(),
                self.vtcm_budget,
            )
        } else {
            None
        };
        let tiled_row_bytes = (ne0 / 32) * w.tile_size;
        let w_nb1 = if hmx_kparams.is_some() {
            mm_hmx_nb1(wtype, ne0)
        } else {
            tiled_row_bytes
        };
        let w_ti = session.add_tensor(
            weights,
            w.offset,
            (ne1 / 32) * tiled_row_bytes,
            HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
            w.wire_dtype as u32,
            [ne0 as u32, ne1 as u32, 1, 1],
            [
                w.block_bytes as u32,
                w_nb1 as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
                ((ne1 / 32) * tiled_row_bytes) as u32,
            ],
        )?;
        let in_ti = session.add_tensor(
            in_act,
            in_offset,
            in_dim * n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [in_dim as u32, n_rows as u32, 1, 1],
            [
                4,
                (in_dim * 4) as u32,
                (in_dim * n_rows * 4) as u32,
                (in_dim * n_rows * 4) as u32,
            ],
        )?;
        let out_ti = session.add_tensor(
            out_act,
            out_offset,
            out_dim * n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [out_dim as u32, n_rows as u32, 1, 1],
            [
                4,
                (out_dim * 4) as u32,
                (out_dim * n_rows * 4) as u32,
                (out_dim * n_rows * 4) as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = hmx_kparams.unwrap_or_else(|| {
            build_mul_mat_kernel_params(
                wtype,
                in_dim,
                n_rows as u32,
                1,
                out_dim * 4,
                session.dsp_threads(),
                self.vtcm_budget,
            )
        });
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat_m",
            HtpOpCode::MulMat as u32,
            &[w_ti, in_ti],
            &[out_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Fused multi-projection GEMM (MUL_MAT_NX): `dst[i][m, :] = in[m, :] @
    /// W[i]` for N weights sharing one activation. The DSP quantizes the
    /// shared activation once instead of N times (llama's QKV and gate/up
    /// fusion). Sources are weights-first, activation last (`[w0..wN, x]`);
    /// N weight inputs plus one activation fit the 10-source / 4-dst op
    /// descriptor for N <= 4.
    ///
    /// Falls back to N single dispatches when fusion is unsupported: N
    /// outside 2..=4, mixed K (`in_dim`) or wire dtype, HMX-eligibility
    /// mismatch across the set, Q6_K on the HVX path (no fused HVX
    /// kernel), or HMX chunking overflow (which retries HVX first, like
    /// the single path). Kernel parameters are W0's single-matmul
    /// parameters with `n_weights` set, so NX fits VTCM exactly when the
    /// W0 single would; W0 must carry the largest N (`out_dim`) so the
    /// m=1 dst scratch covers every output.
    pub(super) fn dispatch_mul_mat_nx(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        ws: &[&HexagonWeight],
        in_act: &RpcmemBuffer,
        in_offset: usize,
        out_act: &RpcmemBuffer,
        out_offsets: &[usize],
        n_rows: usize,
    ) -> Result<(), CeraError> {
        let unfused = |this: &Self, session: &mut HexagonQueueSession| -> Result<(), CeraError> {
            for (w, &off) in ws.iter().zip(out_offsets.iter()) {
                this.dispatch_mul_mat_m(
                    session, weights, w, in_act, in_offset, out_act, off, n_rows,
                )?;
            }
            Ok(())
        };
        let n = ws.len();
        let Some(w0) = ws.first() else {
            return Ok(());
        };
        let pad32 = |d: usize| d.div_ceil(32) * 32;
        let wtype = w0.wire_dtype;
        let k = w0.in_dim;
        let hmx0 = self.use_hmx && mm_is_hmx_eligible(wtype, pad32(k), pad32(w0.out_dim), n_rows);
        let fusable = (2..=4).contains(&n)
            && out_offsets.len() == n
            && ws.iter().all(|w| w.in_dim == k && w.wire_dtype == wtype)
            && ws.iter().all(|w| w.out_dim <= w0.out_dim)
            && ws.iter().all(|w| {
                (self.use_hmx && mm_is_hmx_eligible(wtype, pad32(k), pad32(w.out_dim), n_rows))
                    == hmx0
            })
            && (hmx0 || wtype != HtpDataType::Q6K);
        if !fusable {
            if crate::backend::hexagon::queue::debug_enabled() {
                eprintln!("cera-hexagon: NX fallback: {n} unfused matmuls");
            }
            unfused(self, session)?;
            return Ok(());
        }
        // HMX first, HVX fallback: the same selection as singles, with
        // `n_weights` set. HVX forces QUANT_ROW: NX has no block kernel.
        let hmx_built = if hmx0 {
            build_hmx_mm_kernel_params(
                wtype,
                pad32(k),
                pad32(w0.out_dim),
                n_rows.next_multiple_of(32),
                n_rows,
                session.dsp_threads(),
                self.vtcm_budget,
            )
        } else {
            None
        };
        let kparams = if let Some(mut kp) = hmx_built {
            kp[17] = n as i32; // n_weights
            kp
        } else {
            // HVX path (ineligible or HMX chunking overflow): Q6_K has no
            // fused HVX kernel either way.
            if wtype == HtpDataType::Q6K {
                unfused(self, session)?;
                return Ok(());
            }
            let mut kp = build_mul_mat_kernel_params(
                wtype,
                k,
                n_rows as u32,
                1,
                w0.out_dim * 4,
                session.dsp_threads(),
                self.vtcm_budget,
            );
            kp[0] = 5; // HTP_MM_KERNEL_HVX_QUANT_ROW
            kp[17] = n as i32; // n_weights
            kp
        };
        let hmx_path = kparams[6] == 1; // n_hmx
        let tiled_row_bytes = (pad32(k) / 32) * w0.tile_size;
        let w_nb1 = if hmx_path {
            mm_hmx_nb1(wtype, pad32(k))
        } else {
            tiled_row_bytes
        };
        let mut srcs = Vec::with_capacity(n + 1);
        let mut dsts = Vec::with_capacity(n);
        for w in ws {
            let ne1 = pad32(w.out_dim);
            srcs.push(session.add_tensor(
                weights,
                w.offset,
                (ne1 / 32) * tiled_row_bytes,
                HTP_TENSOR_WEIGHT | HTP_TENSOR_REPACK,
                w.wire_dtype as u32,
                [pad32(k) as u32, ne1 as u32, 1, 1],
                [
                    w.block_bytes as u32,
                    w_nb1 as u32,
                    ((ne1 / 32) * tiled_row_bytes) as u32,
                    ((ne1 / 32) * tiled_row_bytes) as u32,
                ],
            )?);
        }
        srcs.push(session.add_tensor(
            in_act,
            in_offset,
            k * n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [k as u32, n_rows as u32, 1, 1],
            [
                4,
                (k * 4) as u32,
                (k * n_rows * 4) as u32,
                (k * n_rows * 4) as u32,
            ],
        )?);
        for (w, &off) in ws.iter().zip(out_offsets.iter()) {
            dsts.push(session.add_tensor(
                out_act,
                off,
                w.out_dim * n_rows * 4,
                HTP_TENSOR_COMPUTE,
                HtpDataType::F32 as u32,
                [w.out_dim as u32, n_rows as u32, 1, 1],
                [
                    4,
                    (w.out_dim * 4) as u32,
                    (w.out_dim * n_rows * 4) as u32,
                    (w.out_dim * n_rows * 4) as u32,
                ],
            )?);
        }
        let params = [0i32; 16];
        Self::enqueue_labeled(
            session,
            "dispatch_mul_mat_nx",
            HtpOpCode::MulMatNx as u32,
            &srcs,
            &dsts,
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_glu(
        session: &mut HexagonQueueSession,
        gate: &RpcmemBuffer,
        gate_offset: usize,
        up: &RpcmemBuffer,
        up_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        row_dim: usize,
        n_rows: usize,
        activation: FfnActivation,
    ) -> Result<(), CeraError> {
        let bytes = row_dim * n_rows * 4;
        let ne = [row_dim as u32, n_rows as u32, 1, 1];
        let nb = [4, (row_dim * 4) as u32, bytes as u32, bytes as u32];
        let gate_ti = session.add_tensor(
            gate,
            gate_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let up_ti = session.add_tensor(
            up,
            up_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            ne,
            nb,
        )?;
        let params = [0i32; 16];
        // No host precompute: the DSP sizes threads/VTCM itself (llama
        // passes zero kparams).
        let kparams = [0i32; 32];
        let (label, opcode) = match activation {
            FfnActivation::Swiglu => ("dispatch_swiglu", HtpOpCode::GluSwiglu as u32),
            FfnActivation::Geglu => ("dispatch_geglu", HtpOpCode::GluGeglu as u32),
        };
        Self::enqueue_labeled(
            session,
            label,
            opcode,
            &[gate_ti, up_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_rope(
        session: &mut HexagonQueueSession,
        act: &RpcmemBuffer,
        act_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        head_dim: usize,
        n_heads: usize,
        max_seq_len: usize,
        rope_theta: f32,
        mode: u32,
    ) -> Result<(), CeraError> {
        let total_bytes = head_dim * n_heads * 4;
        let act_ti = session.add_tensor(
            act,
            act_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                total_bytes as u32,
                total_bytes as u32,
            ],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [1, 1, 1, 1],
            [4, 4, 4, 4],
        )?;
        let params = build_rope_params(head_dim, mode, max_seq_len as u32, rope_theta, 1.0);
        // Dims are [head_dim, n_heads, 1]: heads are dim 1, tokens dim 2.
        let nrows = n_heads;
        let n_threads = session.dsp_threads().min(nrows as u32).max(1);
        let kparams = build_rope_kernel_params(head_dim, nrows, n_heads, 1, n_threads);
        Self::enqueue_labeled(
            session,
            "dispatch_rope",
            HtpOpCode::Rope as u32,
            &[act_ti, pos_ti],
            &[act_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-token (prefill) RoPE over `[head_dim, n_heads, n_tokens]`
    /// (heads dim 1, tokens dim 2), rotated in place by the `n_tokens`
    /// positions in `pos_buf`.
    pub(super) fn dispatch_rope_m(
        session: &mut HexagonQueueSession,
        act: &RpcmemBuffer,
        act_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_tokens: usize,
        max_seq_len: usize,
        rope_theta: f32,
        mode: u32,
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let total_bytes = q_dim * n_tokens * 4;
        let act_ti = session.add_tensor(
            act,
            act_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, n_tokens as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (q_dim * 4) as u32,
                total_bytes as u32,
            ],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            n_tokens * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_tokens as u32, 1, 1, 1],
            [
                4,
                (n_tokens * 4) as u32,
                (n_tokens * 4) as u32,
                (n_tokens * 4) as u32,
            ],
        )?;
        let params = build_rope_params(head_dim, mode, max_seq_len as u32, rope_theta, 1.0);
        let nrows = n_heads * n_tokens;
        let n_threads = session.dsp_threads().min(nrows as u32).max(1);
        let kparams = build_rope_kernel_params(head_dim, nrows, n_heads, n_tokens, n_threads);
        Self::enqueue_labeled(
            session,
            "dispatch_rope_m",
            HtpOpCode::Rope as u32,
            &[act_ti, pos_ti],
            &[act_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Single-row (decode) SetRows: appends one K (or V) row into the f16
    /// cache at the absolute slot in the positions vector. Values and
    /// cache are flat 2D (`[kv_dim, rows]`): the DSP worker iterates
    /// `ne02 * rows` DMA steps, so the old 3D per-head view cost a
    /// per-head round-trip (6x at 8 KV heads).
    pub(super) fn dispatch_set_rows_typed(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        cache: &RpcmemBuffer,
        cache_offset: usize,
        head_dim: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        kv_dtype: HtpDataType,
    ) -> Result<(), CeraError> {
        let kv_dim = head_dim * n_kv_heads;
        let src_bytes = kv_dim * 4;
        let cache_bytes = kv_cache_bytes(kv_dtype, kv_dim, max_seq_len);
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [kv_dim as u32, 1, 1, 1],
            [4, (kv_dim * 4) as u32, src_bytes as u32, src_bytes as u32],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [1, 1, 1, 1],
            [4, 4, 4, 4],
        )?;
        let cache_ti = session.add_tensor(
            cache,
            cache_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            kv_dtype as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                kv_elem_nb0(kv_dtype),
                kv_row_stride(kv_dtype, kv_dim) as u32,
                cache_bytes as u32,
                cache_bytes as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = build_set_rows_kernel_params(1, 1, 1, 1, kv_dim, true, session.dsp_threads());
        Self::enqueue_labeled(
            session,
            "dispatch_set_rows",
            HtpOpCode::SetRows as u32,
            &[src_ti, pos_ti],
            &[cache_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// M-row (prefill) SetRows: appends `n_rows` K (or V) rows from the
    /// flat `[kv_dim, n_rows]` values view into the interleaved cache
    /// (`[kv_dim, max_seq]`, all heads contiguous per position) at the
    /// `n_rows` absolute slots in the positions vector.
    pub(super) fn dispatch_set_rows_m_typed(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        pos_buf: &RpcmemBuffer,
        pos_offset: usize,
        cache: &RpcmemBuffer,
        cache_offset: usize,
        head_dim: usize,
        n_kv_heads: usize,
        n_rows: usize,
        max_seq_len: usize,
        kv_dtype: HtpDataType,
    ) -> Result<(), CeraError> {
        let kv_dim = head_dim * n_kv_heads;
        let src_bytes = kv_dim * n_rows * 4;
        let cache_bytes = kv_cache_bytes(kv_dtype, kv_dim, max_seq_len);
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [kv_dim as u32, n_rows as u32, 1, 1],
            [4, (kv_dim * 4) as u32, src_bytes as u32, src_bytes as u32],
        )?;
        let pos_ti = session.add_tensor(
            pos_buf,
            pos_offset,
            n_rows * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::I32 as u32,
            [n_rows as u32, 1, 1, 1],
            [
                4,
                (n_rows * 4) as u32,
                (n_rows * 4) as u32,
                (n_rows * 4) as u32,
            ],
        )?;
        let cache_ti = session.add_tensor(
            cache,
            cache_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            kv_dtype as u32,
            [kv_dim as u32, max_seq_len as u32, 1, 1],
            [
                kv_elem_nb0(kv_dtype),
                kv_row_stride(kv_dtype, kv_dim) as u32,
                cache_bytes as u32,
                cache_bytes as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams =
            build_set_rows_kernel_params(n_rows, 1, 1, 1, kv_dim, true, session.dsp_threads());
        Self::enqueue_labeled(
            session,
            "dispatch_set_rows_m",
            HtpOpCode::SetRows as u32,
            &[src_ti, pos_ti],
            &[cache_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_rms_norm_mul(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        weight: &RpcmemBuffer,
        weight_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        eps: f32,
        head_dim: usize,
        n_heads: usize,
    ) -> Result<(), CeraError> {
        let total_bytes = head_dim * n_heads * 4;
        let src_ti = session.add_tensor(
            src,
            src_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                total_bytes as u32,
                total_bytes as u32,
            ],
        )?;
        let weight_ti = session.add_tensor(
            weight,
            weight_offset,
            head_dim * 4,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            total_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, 1, 1],
            [
                4,
                (head_dim * 4) as u32,
                total_bytes as u32,
                total_bytes as u32,
            ],
        )?;
        let params = build_rms_norm_params(eps);
        let kparams = build_unary_kernel_params(
            head_dim,
            n_heads,
            head_dim,
            8 * 1024 * 1024,
            session.dsp_threads(),
            true,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_rms_norm_mul",
            HtpOpCode::RmsNormMul as u32,
            &[src_ti, weight_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_flash_attn_ext_typed(
        session: &mut HexagonQueueSession,
        q: &RpcmemBuffer,
        q_offset: usize,
        k_cache: &RpcmemBuffer,
        k_offset: usize,
        v_cache: &RpcmemBuffer,
        v_offset: usize,
        mask: &RpcmemBuffer,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        max_seq_len: usize,
        scale: f32,
        kv_dtype: HtpDataType,
        softcap: f32,
    ) -> Result<(usize, usize, usize), CeraError> {
        let q_bytes = head_dim * n_heads * 4;
        let kv_dim = head_dim * n_kv_heads;
        let cache_bytes = kv_cache_bytes(kv_dtype, kv_dim, max_seq_len);
        let q_ti = session.add_tensor(
            q,
            q_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, n_heads as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                q_bytes as u32,
            ],
        )?;
        let k_ti = session.add_tensor(
            k_cache,
            k_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                kv_elem_nb0(kv_dtype),
                kv_row_stride(kv_dtype, kv_dim) as u32,
                kv_row_stride(kv_dtype, head_dim) as u32,
                cache_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v_cache,
            v_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                kv_elem_nb0(kv_dtype),
                kv_row_stride(kv_dtype, kv_dim) as u32,
                kv_row_stride(kv_dtype, head_dim) as u32,
                cache_bytes as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, 1, n_heads as u32, 1],
            [
                4,
                (head_dim * 4) as u32,
                (head_dim * 4) as u32,
                q_bytes as u32,
            ],
        )?;
        let mask_bytes = seq_len * 2;
        let mask_ti = session.add_tensor(
            mask,
            0,
            mask_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [seq_len as u32, 1, 1, 1],
            [2, mask_bytes as u32, mask_bytes as u32, mask_bytes as u32],
        )?;
        let mut params = [0i32; 16];
        params[0] = scale.to_bits() as i32;
        let kparams = build_flash_attn_kernel_params_with_softcap(
            head_dim,
            n_heads,
            n_kv_heads,
            1,
            seq_len,
            scale,
            session.dsp_threads(),
            true,
            softcap,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_flash_attn_ext",
            HtpOpCode::FlashAttnExt as u32,
            &[q_ti, k_ti, v_ti, mask_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok((k_ti as usize, v_ti as usize, mask_ti as usize))
    }

    /// Flush pending ops if debug_barriers is set.
    #[inline]
    pub(super) fn debug_barrier(
        &self,
        session: &mut HexagonQueueSession,
        label: &str,
    ) -> Result<(), CeraError> {
        if self.debug_barriers {
            session
                .flush()
                .map_err(|e| CeraError::Backend(format!("{label} flush failed: {e}")))?;
        }
        Ok(())
    }

    /// M-token (prefill) FlashAttention: `n_tokens` queries in `[head_dim,
    /// n_tokens, n_heads]` over the `[head_dim, seq_len, n_kv_heads]` valid
    /// KV prefix, biased by the `[seq_len, n_tokens]` causal mask (query
    /// rows via dim-1 stride). The output follows the ggml permute(0, 2, 1,
    /// 3) convention: `[head_dim, n_heads, n_tokens]` (the firmware indexes
    /// head via dim-1 and token via dim-2, ignoring `dst->ne`).
    pub(super) fn dispatch_flash_attn_m(
        &self,
        session: &mut HexagonQueueSession,
        q: &RpcmemBuffer,
        q_offset: usize,
        k_cache: &RpcmemBuffer,
        k_offset: usize,
        v_cache: &RpcmemBuffer,
        v_offset: usize,
        mask: &RpcmemBuffer,
        mask_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        n_tokens: usize,
        seq_len: usize,
        max_seq_len: usize,
        scale: f32,
    ) -> Result<(), CeraError> {
        let q_dim = head_dim * n_heads;
        let kv_dim = head_dim * n_kv_heads;
        let q_bytes = q_dim * n_tokens * 4;
        let cache_bytes = kv_cache_bytes(self.kv_dtype, kv_dim, max_seq_len);
        let q_ti = session.add_tensor(
            q,
            q_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_tokens as u32, n_heads as u32, 1],
            [4, (q_dim * 4) as u32, (head_dim * 4) as u32, q_bytes as u32],
        )?;
        // Permuted views of the interleaved `[kv_dim, max_seq]` cache: head
        // h, position p starts at `p * kv_dim + h * head_dim`.
        let k_ti = session.add_tensor(
            k_cache,
            k_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            self.kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                kv_elem_nb0(self.kv_dtype),
                kv_row_stride(self.kv_dtype, kv_dim) as u32,
                kv_row_stride(self.kv_dtype, head_dim) as u32,
                cache_bytes as u32,
            ],
        )?;
        let v_ti = session.add_tensor(
            v_cache,
            v_offset,
            cache_bytes,
            HTP_TENSOR_COMPUTE,
            self.kv_dtype as u32,
            [head_dim as u32, seq_len as u32, n_kv_heads as u32, 1],
            [
                kv_elem_nb0(self.kv_dtype),
                kv_row_stride(self.kv_dtype, kv_dim) as u32,
                kv_row_stride(self.kv_dtype, head_dim) as u32,
                cache_bytes as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            q_bytes,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [head_dim as u32, n_heads as u32, n_tokens as u32, 1],
            [4, (head_dim * 4) as u32, (q_dim * 4) as u32, q_bytes as u32],
        )?;
        let mask_ti = session.add_tensor(
            mask,
            mask_offset,
            seq_len * n_tokens * 2,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F16 as u32,
            [seq_len as u32, n_tokens as u32, 1, 1],
            [
                2,
                (seq_len * 2) as u32,
                (seq_len * n_tokens * 2) as u32,
                (seq_len * n_tokens * 2) as u32,
            ],
        )?;
        let mut params = [0i32; 16];
        params[0] = scale.to_bits() as i32;
        let softcap = self.attn_logit_softcapping.unwrap_or(0.0);
        // HMX first (DK % 8, M >= 5 at small head_dim), HVX fallback.
        let kparams = if self.use_hmx
            && fa_is_hmx_eligible(head_dim, n_tokens)
            && let Some(hmx) = build_hmx_fa_kernel_params_with_softcap(
                head_dim,
                n_heads,
                n_kv_heads,
                n_tokens,
                seq_len,
                scale,
                session.dsp_threads(),
                self.vtcm_budget,
                softcap,
            ) {
            hmx
        } else {
            build_flash_attn_kernel_params_with_softcap(
                head_dim,
                n_heads,
                n_kv_heads,
                n_tokens,
                seq_len,
                scale,
                session.dsp_threads(),
                true,
                softcap,
            )
        };
        Self::enqueue_labeled(
            session,
            "dispatch_flash_attn_m",
            HtpOpCode::FlashAttnExt as u32,
            &[q_ti, k_ti, v_ti, mask_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    pub(super) fn dispatch_cpy(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
    ) -> Result<(), CeraError> {
        let src_ti = Self::add_f32_vec(session, src, src_offset, dim, HTP_TENSOR_COMPUTE)?;
        let dst_ti = Self::add_f32_vec(session, dst, dst_offset, dim, HTP_TENSOR_COMPUTE)?;
        let params = [0i32; 16];
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_cpy",
            HtpOpCode::Cpy as u32,
            &[src_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// Strided 2D copy (assembly/transpose/scatter primitive): copies the
    /// `[ne0, ne1]` f32 tile between two strided descriptors (same index
    /// space, independent strides). A transpose carries the transposed
    /// shape on the source side; a scatter (interleaved destination)
    /// strides the destination side. The firmware resolves strides
    /// device-side (no kparams). NOTE: strided sides take the firmware's
    /// scalar per-element path: fine for state-sized (hs-scale) tiles,
    /// prohibitive for m*hs transposes (use CONCAT's transposed worker).
    pub(super) fn dispatch_cpy_2d(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        src_ne0: usize,
        src_ne1: usize,
        src_nb0: usize,
        src_nb1: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dst_nb0: usize,
        dst_nb1: usize,
    ) -> Result<(), CeraError> {
        let span = |nb0: usize, nb1: usize| {
            src_ne0.saturating_sub(1) * nb0 + src_ne1.saturating_sub(1) * nb1 + 4
        };
        let src_span = span(src_nb0, src_nb1);
        let dst_span = span(dst_nb0, dst_nb1);
        let src_ti = session.add_tensor(
            src,
            src_offset,
            src_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [src_ne0 as u32, src_ne1 as u32, 1, 1],
            [
                src_nb0 as u32,
                src_nb1 as u32,
                src_span as u32,
                src_span as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            dst_span,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [src_ne0 as u32, src_ne1 as u32, 1, 1],
            [
                dst_nb0 as u32,
                dst_nb1 as u32,
                dst_span as u32,
                dst_span as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = [0i32; 32];
        Self::enqueue_labeled(
            session,
            "dispatch_cpy_2d",
            HtpOpCode::Cpy as u32,
            &[src_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }

    /// SsmConv dispatch: `y[c, m] = sum_t x[m + t, c] * w[t, c]` over the
    /// `[ncs, C]` input window (`ncs = d_conv - 1 + n_t`: prior states plus
    /// new inputs, time-major), `[d_conv, C]` oldest-first taps, producing
    /// channel-major `[C, n_t]` (already `[M, C]` row-major in memory: dst
    /// dim-1 stride is the token stride, so no transpose is needed).
    pub(super) fn dispatch_ssm_conv(
        &self,
        session: &mut HexagonQueueSession,
        weights: &RpcmemBuffer,
        weights_offset: usize,
        conv_x: &RpcmemBuffer,
        conv_x_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        d_conv: usize,
        d_inner: usize,
        n_t: usize,
    ) -> Result<(), CeraError> {
        let ncs = d_conv - 1 + n_t;
        let x_ti = session.add_tensor(
            conv_x,
            conv_x_offset,
            ncs * d_inner * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [ncs as u32, d_inner as u32, 1, 1],
            [
                4,
                (ncs * 4) as u32,
                (ncs * d_inner * 4) as u32,
                (ncs * d_inner * 4) as u32,
            ],
        )?;
        let w_ti = session.add_tensor(
            weights,
            weights_offset,
            d_conv * d_inner * 4,
            HTP_TENSOR_WEIGHT,
            HtpDataType::F32 as u32,
            [d_conv as u32, d_inner as u32, 1, 1],
            [
                4,
                (d_conv * 4) as u32,
                (d_conv * d_inner * 4) as u32,
                (d_conv * d_inner * 4) as u32,
            ],
        )?;
        let dst_ti = session.add_tensor(
            dst,
            dst_offset,
            d_inner * n_t * 4,
            HTP_TENSOR_COMPUTE,
            HtpDataType::F32 as u32,
            [d_inner as u32, n_t as u32, 1, 1],
            [
                4,
                (d_inner * 4) as u32,
                (d_inner * n_t * 4) as u32,
                (d_inner * n_t * 4) as u32,
            ],
        )?;
        let params = [0i32; 16];
        let kparams = build_ssm_conv_kernel_params(
            d_conv,
            d_inner,
            n_t,
            1,
            ncs,
            session.dsp_threads(),
            self.vtcm_budget,
        );
        Self::enqueue_labeled(
            session,
            "dispatch_ssm_conv",
            HtpOpCode::SsmConv as u32,
            &[x_ti, w_ti],
            &[dst_ti],
            params,
            kparams,
        )?;
        Ok(())
    }
}

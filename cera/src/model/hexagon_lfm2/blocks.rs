//! Per-block emitters (attention, FFN, short-conv, DeltaNet) composed from the
//! op emitters, for decode and prefill.

use super::*;

impl HexagonLfmModel {
    /// Softmax scale of the attention layers.
    pub(super) fn attn_scale(&self) -> f32 {
        self.dense
            .attn_scale
            .unwrap_or(1.0f32 / (self.config.head_dim as f32).sqrt())
    }

    /// Ping-pong scratch buffers of a layer: `(input activation, output
    /// activation, normed)`. Adjacent layers alternate so a layer never reads
    /// what the previous one is still writing.
    pub(super) fn layer_buffers(&self, layer_idx: usize) -> (usize, usize, usize) {
        let so = &self.scratch_offsets;
        if layer_idx.is_multiple_of(2) {
            (so.activation, so.activation_b, so.normed)
        } else {
            (so.activation_b, so.activation, so.normed_b)
        }
    }

    /// Looped architectures (Nanbeige) re-norm the residual stream with the
    /// output norm after every `interval` layers, except after the last.
    pub(super) fn emit_loop_norm(
        &self,
        session: &mut HexagonQueueSession,
        layer_idx: usize,
        act: usize,
        rows: usize,
    ) -> Result<(), CeraError> {
        let Some(interval) = self.dense.loop_norm_interval else {
            return Ok(());
        };
        if !(layer_idx + 1).is_multiple_of(interval) || layer_idx + 1 >= self.layers.len() {
            return Ok(());
        }
        let scratch = &self.scratch_buf;
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            act,
            &self.weights_buf,
            self.output_norm_offset,
            scratch,
            act,
            self.config.rms_norm_eps,
            self.config.hidden_size,
            rows,
        )
    }

    /// FFN sub-block of one layer, shared by every layer kind and both passes:
    /// pre-norm (skipped for post-norm architectures), dense or routed FFN with
    /// optional projection biases, post-norm, residual multiplier, and the
    /// residual add into `next_act`. `decode` selects the single-token kernels;
    /// otherwise `rows` prefill rows run through the batched ones.
    fn emit_ffn_block(
        &self,
        session: &mut HexagonQueueSession,
        ffn: &HexagonFfn,
        ffn_norm_offset: usize,
        post_norm_offset: Option<usize>,
        layer_idx: usize,
        next_act: usize,
        cur_normed: usize,
        rows: usize,
        decode: bool,
    ) -> Result<(), CeraError> {
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let hs = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let intermediate_size = self.config.intermediate_size;
        let ffn_in = if self.dense.post_norm {
            next_act
        } else {
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                next_act,
                &self.weights_buf,
                ffn_norm_offset,
                scratch,
                cur_normed,
                eps,
                hs,
                rows,
            )?;
            cur_normed
        };
        match ffn {
            HexagonFfn::Dense(dense) => {
                self.dispatch_mul_mat_nx(
                    session,
                    &self.weights_buf,
                    &[&dense.gate, &dense.up],
                    scratch,
                    ffn_in,
                    scratch,
                    &[so.ffn_gate, so.ffn_up],
                    rows,
                )?;
                if let Some(b) = dense.gate_bias {
                    self.dispatch_bias_add(
                        session,
                        scratch,
                        so.ffn_gate,
                        b,
                        intermediate_size,
                        rows,
                    )?;
                }
                if let Some(b) = dense.up_bias {
                    self.dispatch_bias_add(
                        session,
                        scratch,
                        so.ffn_up,
                        b,
                        intermediate_size,
                        rows,
                    )?;
                }
                Self::dispatch_glu(
                    session,
                    scratch,
                    so.ffn_gate,
                    scratch,
                    so.ffn_up,
                    scratch,
                    so.ffn_out,
                    intermediate_size,
                    rows,
                    self.activation,
                )?;
                if decode {
                    self.dispatch_mul_mat(
                        session,
                        &self.weights_buf,
                        &dense.down,
                        scratch,
                        so.ffn_out,
                        scratch,
                        cur_normed,
                    )?;
                } else {
                    self.dispatch_mul_mat_m(
                        session,
                        &self.weights_buf,
                        &dense.down,
                        scratch,
                        so.ffn_out,
                        scratch,
                        cur_normed,
                        rows,
                    )?;
                }
                if let Some(b) = dense.down_bias {
                    self.dispatch_bias_add(session, scratch, cur_normed, b, hs, rows)?;
                }
            }
            HexagonFfn::Moe(moe) => {
                self.page_in_experts(session, moe)?;
                // Routing is per token: one chain per row. Every row's chain
                // registers its own tensors, so a row boundary is a valid place
                // to flush; a long chunk would otherwise overflow the staging
                // buffer (each row adds roughly 10 KB of descriptors, so a chunk
                // of a few hundred rows is more than the 4 MiB staging buffer).
                for row in 0..rows {
                    if session.pending_bytes() > session.staging_capacity() / 2 {
                        session.flush()?;
                    }
                    self.dispatch_moe_token(
                        session,
                        moe,
                        scratch,
                        ffn_in + row * hs * 4,
                        cur_normed + row * hs * 4,
                        row,
                    )?;
                }
            }
        }
        if let Some(post_norm_offset) = post_norm_offset {
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                cur_normed,
                &self.weights_buf,
                post_norm_offset,
                scratch,
                cur_normed,
                eps,
                hs,
                rows,
            )?;
        }
        self.dispatch_residual_scale(session, scratch, cur_normed, hs, rows)?;
        if decode {
            self.dump_hidden(session, scratch, layer_idx, "ffn-out", cur_normed, hs);
            Self::dispatch_add(
                session, scratch, next_act, scratch, cur_normed, scratch, next_act, hs,
            )?;
        } else {
            Self::dispatch_add_m(
                session, scratch, next_act, scratch, cur_normed, scratch, next_act, hs, rows,
            )?;
            self.dump_hidden(
                session,
                scratch,
                layer_idx,
                "prefill post-ffn",
                next_act + (rows - 1) * hs * 4,
                hs,
            );
        }
        Ok(())
    }

    /// Ops of one attention layer for either pass. Shared: block norm, QKV
    /// projections and biases, the Qwen 3.5 output-gate split, Q/K norms, host
    /// RoPE, attention temperature, the gate multiply, the output bias,
    /// post-norm, residual multiplier and residual add. Per pass: the DSP RoPE,
    /// KV write, flash-attention and output-projection kernels (single-token
    /// versus `m`-row variants).
    pub(super) fn emit_attention_block(
        &self,
        session: &mut HexagonQueueSession,
        attn: &HexagonAttentionLayer,
        layer_idx: usize,
        cur_act: usize,
        next_act: usize,
        cur_normed: usize,
        pass: AttnPass<'_>,
    ) -> Result<(), CeraError> {
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let hs = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let max_seq_len = self.config.max_seq_len;
        let rope_theta = self.config.rope_theta;
        let attn_scale = self.attn_scale();
        let n_kv_heads = attn.kv_dim / head_dim;
        let q_dim = attn.q_dim;
        let kv_dim = attn.kv_dim;
        let decode = matches!(pass, AttnPass::Decode { .. });
        let (first_pos, rows, patches) = match pass {
            AttnPass::Decode { pos, patches } => (pos, 1, patches),
            AttnPass::Prefill { start_pos, m } => (start_pos, m, None),
        };
        let kv_len = first_pos + rows;

        // Block norm (post-norm architectures, Olmo 2/3, project the raw
        // residual stream and norm the block output instead).
        let attn_in = if self.dense.post_norm {
            cur_act
        } else {
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                cur_act,
                &self.weights_buf,
                attn.attn_norm_offset,
                scratch,
                cur_normed,
                eps,
                hs,
                rows,
            )?;
            cur_normed
        };
        self.debug_barrier(session, "attention attn_norm")?;

        // Projections: Q, K, V (fused NX: one shared-activation op).
        self.dispatch_mul_mat_nx(
            session,
            &self.weights_buf,
            &[&attn.attn_q, &attn.attn_k, &attn.attn_v],
            scratch,
            attn_in,
            scratch,
            &[so.q, so.k, so.v],
            rows,
        )?;
        self.debug_barrier(session, "attention QKV proj")?;
        if let Some([q_bias, k_bias, v_bias]) = attn.qkv_bias {
            self.dispatch_bias_add(session, scratch, so.q, q_bias, q_dim, rows)?;
            self.dispatch_bias_add(session, scratch, so.k, k_bias, kv_dim, rows)?;
            self.dispatch_bias_add(session, scratch, so.v, v_bias, kv_dim, rows)?;
        }
        if attn.has_q_gate {
            // Qwen 3.5 fuses the output gate into Q: per row `[head, (q | gate)]`.
            for mm in 0..rows {
                let row_full_off = so.q + mm * 2 * q_dim * 4;
                let row_q_off = so.conv_in + mm * q_dim * 4;
                let row_gate_off = so.conv_bx + mm * q_dim * 4;
                Self::dispatch_cpy_2d(
                    session,
                    scratch,
                    row_full_off,
                    head_dim,
                    n_heads,
                    4,
                    2 * head_dim * 4,
                    scratch,
                    row_q_off,
                    4,
                    head_dim * 4,
                )?;
                Self::dispatch_cpy_2d(
                    session,
                    scratch,
                    row_full_off + head_dim * 4,
                    head_dim,
                    n_heads,
                    4,
                    2 * head_dim * 4,
                    scratch,
                    row_gate_off,
                    4,
                    head_dim * 4,
                )?;
            }
            Self::dispatch_cpy(session, scratch, so.conv_in, scratch, so.q, rows * q_dim)?;
        }

        // Optional Q/K norm: per head, or over the whole vector (Olmo 2/3).
        if let Some(qn_offset) = attn.attn_q_norm_offset {
            let (norm_dim, norm_rows) = if attn.qk_norm_full {
                (q_dim, rows)
            } else {
                (head_dim, rows * n_heads)
            };
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                so.q,
                &self.weights_buf,
                qn_offset,
                scratch,
                so.q,
                eps,
                norm_dim,
                norm_rows,
            )?;
        }
        if let Some(kn_offset) = attn.attn_k_norm_offset {
            let (norm_dim, norm_rows) = if attn.qk_norm_full {
                (kv_dim, rows)
            } else {
                (head_dim, rows * n_kv_heads)
            };
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                so.k,
                &self.weights_buf,
                kn_offset,
                scratch,
                so.k,
                eps,
                norm_dim,
                norm_rows,
            )?;
        }
        self.debug_barrier(session, "attention QK norm")?;

        // RoPE on Q and K: DSP kernel by default, with a host-CPU route
        // (CERA_HEXAGON_CPU_ROPE=1, or a model that needs YaRN / frequency
        // factors / partial rotary) using the same cpu::rope the CPU backend uses.
        if self.cpu_rope {
            // Host-CPU RoPE reads DSP-produced Q/K in place, so this barrier is
            // mandatory even in fused mode.
            session
                .flush()
                .map_err(|e| CeraError::Backend(format!("attention pre-RoPE flush failed: {e}")))?;
            let q_bytes = q_dim * rows * 4;
            let k_bytes = kv_dim * rows * 4;
            scratch.invalidate_cpu_cache(so.q, q_bytes);
            scratch.invalidate_cpu_cache(so.k, k_bytes);
            let mut rope_scratch = self.rope_scratch.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                for mm in 0..rows {
                    let q = std::slice::from_raw_parts_mut(
                        scratch.as_mut_ptr().add(so.q + mm * q_dim * 4) as *mut f32,
                        q_dim,
                    );
                    let k = std::slice::from_raw_parts_mut(
                        scratch.as_mut_ptr().add(so.k + mm * kv_dim * 4) as *mut f32,
                        kv_dim,
                    );
                    host_rope(
                        &mut rope_scratch,
                        self.rope_type,
                        attn.yarn.as_ref(),
                        self.dense.rope_freqs.as_deref(),
                        q,
                        k,
                        first_pos + mm,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        self.dense.rope_dim.unwrap_or(head_dim),
                        rope_theta,
                    );
                }
            }
            scratch.flush_cpu_cache(so.q, q_bytes);
            scratch.flush_cpu_cache(so.k, k_bytes);
        } else {
            let rope_mode = self.rope_type.htp_mode();
            for (act, heads) in [(so.q, n_heads), (so.k, n_kv_heads)] {
                if decode {
                    Self::dispatch_rope(
                        session,
                        scratch,
                        act,
                        scratch,
                        so.pos,
                        head_dim,
                        heads,
                        max_seq_len,
                        rope_theta,
                        rope_mode,
                    )?;
                } else {
                    Self::dispatch_rope_m(
                        session,
                        scratch,
                        act,
                        scratch,
                        so.pos,
                        head_dim,
                        heads,
                        rows,
                        max_seq_len,
                        rope_theta,
                        rope_mode,
                    )?;
                }
            }
            self.debug_barrier(session, "attention RoPE")?;
        }

        // Attention temperature (Mistral 3): Q rows scaled after RoPE.
        if self.dense.attn_temp.is_some() {
            let factors: Vec<(usize, f32)> = (0..rows)
                .filter_map(|mm| {
                    attn_temp_q_scale(first_pos + mm, self.dense.attn_temp).map(|f| (mm, f))
                })
                .collect();
            Self::host_scale_rows(session, scratch, so.q, q_dim, &factors)?;
        }

        // Append the K/V rows to the cache (slots == positions: the pos vector).
        for (act, cache_offset) in [(so.k, attn.k_offset), (so.v, attn.v_offset)] {
            if decode {
                Self::dispatch_set_rows_typed(
                    session,
                    scratch,
                    act,
                    scratch,
                    so.pos,
                    &self.kv_state_buf,
                    cache_offset,
                    head_dim,
                    n_kv_heads,
                    max_seq_len,
                    self.kv_dtype,
                )?;
            } else {
                Self::dispatch_set_rows_m_typed(
                    session,
                    scratch,
                    act,
                    scratch,
                    so.pos,
                    &self.kv_state_buf,
                    cache_offset,
                    head_dim,
                    n_kv_heads,
                    rows,
                    max_seq_len,
                    self.kv_dtype,
                )?;
            }
        }
        self.debug_barrier(session, "attention SetRows")?;
        if !decode {
            self.dump_hidden(
                session,
                scratch,
                layer_idx,
                "(attn) prefill v-proj",
                so.v,
                rows * kv_dim,
            );
        }

        // Attention over the valid prefix (sliding-window layers use the
        // windowed mask).
        if decode {
            let mask_buf = match (&self.dense.mask_swa, attn.swa) {
                (Some(swa), true) => swa,
                _ => &self.mask_buf,
            };
            let op_idx = session.ops_len();
            let (k_ti, v_ti, mask_ti) = Self::dispatch_flash_attn_ext_typed(
                session,
                scratch,
                so.q,
                &self.kv_state_buf,
                attn.k_offset,
                &self.kv_state_buf,
                attn.v_offset,
                mask_buf,
                scratch,
                so.attn_out,
                head_dim,
                n_heads,
                n_kv_heads,
                kv_len,
                max_seq_len,
                attn_scale,
                self.kv_dtype,
                self.attn_logit_softcapping.unwrap_or(0.0),
            )?;
            if let Some(patches) = patches {
                patches.push(FlashAttnPatch {
                    op_idx,
                    k_ti,
                    v_ti,
                    mask_ti,
                    g: (n_heads / n_kv_heads.max(1)).max(1),
                });
            }
        } else {
            self.dispatch_flash_attn_m(
                session,
                scratch,
                so.q,
                &self.kv_state_buf,
                attn.k_offset,
                &self.kv_state_buf,
                attn.v_offset,
                scratch,
                if attn.swa && so.mask_swa != 0 {
                    so.mask_swa
                } else {
                    so.mask
                },
                scratch,
                so.attn_out,
                head_dim,
                n_heads,
                n_kv_heads,
                rows,
                kv_len,
                max_seq_len,
                attn_scale,
            )?;
        }
        self.debug_barrier(
            session,
            &format!("attention layer {layer_idx} flash attention"),
        )?;
        if !decode {
            self.dump_hidden(
                session,
                scratch,
                layer_idx,
                "(attn) prefill fa-out",
                so.attn_out,
                rows * q_dim,
            );
        }

        if attn.has_q_gate {
            Self::dispatch_unary_rows(
                session,
                scratch,
                so.conv_bx,
                scratch,
                so.conv_bx,
                q_dim,
                rows,
                HtpOpCode::UnarySigmoid,
            )?;
            // One row per token, not one flat `rows * q_dim` vector: the Mul
            // worker keeps a whole row in VTCM, so the flat form fails with
            // `VtcmTooSmall` once a Qwen 3.5 prefill chunk reaches 64 rows.
            Self::dispatch_mul_m_strided(
                session,
                scratch,
                so.attn_out,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.conv_bx,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.attn_out,
                q_dim,
                rows,
                q_dim * 4,
                q_dim * 4,
            )?;
        }

        // Output projection, bias, post-norm, residual multiplier, residual add.
        if decode {
            self.dispatch_mul_mat(
                session,
                &self.weights_buf,
                &attn.attn_output,
                scratch,
                so.attn_out,
                scratch,
                cur_normed,
            )?;
        } else {
            self.dispatch_mul_mat_m(
                session,
                &self.weights_buf,
                &attn.attn_output,
                scratch,
                so.attn_out,
                scratch,
                cur_normed,
                rows,
            )?;
        }
        if let Some(out_bias) = attn.out_bias {
            self.dispatch_bias_add(session, scratch, cur_normed, out_bias, hs, rows)?;
        }
        if let Some(post_norm_offset) = attn.attn_post_norm_offset {
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                cur_normed,
                &self.weights_buf,
                post_norm_offset,
                scratch,
                cur_normed,
                eps,
                hs,
                rows,
            )?;
        }
        self.dispatch_residual_scale(session, scratch, cur_normed, hs, rows)?;
        let last_row = (rows - 1) * hs * 4;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(attn) block-out",
            cur_normed + last_row,
            hs,
        );
        if decode {
            Self::dispatch_add(
                session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
            )?;
        } else {
            Self::dispatch_add_m(
                session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs, rows,
            )?;
        }
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(attn) post-block",
            next_act + last_row,
            hs,
        );

        self.emit_ffn_block(
            session,
            &attn.ffn,
            attn.ffn_norm_offset,
            attn.ffn_post_norm_offset,
            layer_idx,
            next_act,
            cur_normed,
            rows,
            decode,
        )
    }

    /// Decode-pass ops of one Conv layer (a single token at `pos`).
    pub(super) fn emit_conv_decode(
        &self,
        session: &mut HexagonQueueSession,
        conv: &HexagonConvLayer,
        layer_idx: usize,
        cur_act: usize,
        next_act: usize,
        cur_normed: usize,
    ) -> Result<(), CeraError> {
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let hs = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        // This layer's `b*x` row lives in a slot no other conv layer uses. The state writeback below
        // reads it with strided scalar loads, and when it shared one address with every other conv
        // layer in a long batch those loads returned the previous layer's row (see
        // `ScratchOffsets::conv_stage`).
        let bx = so.conv_stage_at(layer_idx);
        // Conv RMS norm
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            cur_act,
            &self.weights_buf,
            conv.attn_norm_offset,
            scratch,
            cur_normed,
            eps,
            hs,
            1,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) normed-act",
            cur_normed,
            hs,
        );

        // in_proj: hs -> 3 * hs (b, c, x)
        self.dispatch_mul_mat(
            session,
            &self.weights_buf,
            &conv.in_proj,
            scratch,
            cur_normed,
            scratch,
            so.conv_in,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) conv-in",
            so.conv_in,
            3 * hs,
        );

        if self.use_ssm_conv {
            // bx = b * x (M=1 rows are contiguous, as in the manual path)
            Self::dispatch_mul(
                session,
                scratch,
                so.conv_in,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.conv_in + 2 * hs * 4,
                HTP_TENSOR_COMPUTE,
                scratch,
                bx,
                hs,
            )?;
            // State prepend: CONCAT([s0; s1] + bx-as-[1, hs])
            // into conv_x `[3, hs]` (same op as prefill; the
            // s0/s1 slab is adjacent by construction).
            Self::dispatch_concat_2d(
                session,
                &self.kv_state_buf,
                conv.state_offset,
                2,
                scratch,
                bx,
                1,
                hs * 4,
                4,
                scratch,
                so.conv_x,
                hs,
            )?;
            // y = shortconv(conv_x), channel-major [C, 1]
            self.dispatch_ssm_conv(
                session,
                &self.weights_buf,
                conv.conv_ssm_offset,
                scratch,
                so.conv_x,
                scratch,
                so.conv_ssm_y,
                3,
                hs,
                1,
            )?;
            // [C, 1] -> [1, C] is a flat copy (same bytes)
            Self::dispatch_cpy(session, scratch, so.conv_ssm_y, scratch, so.conv_y, hs)?;
            self.dump_hidden(session, scratch, layer_idx, "(conv) ssm-y", so.conv_y, hs);
            // Update states: s0 = s1; s1 = bx. The odd->even
            // shift overlaps in the interleaved slab, so it
            // stages through conv_t0 (unused on this path).
            Self::dispatch_cpy_2d(
                session,
                &self.kv_state_buf,
                conv.state_offset + 4,
                hs,
                1,
                8,
                8,
                scratch,
                so.conv_t0,
                4,
                hs * 4,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                so.conv_t0,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset,
                8,
                8,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                bx,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset + 4,
                8,
                8,
            )?;
        } else {
            // bx = b * x
            Self::dispatch_mul(
                session,
                scratch,
                so.conv_in,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.conv_in + 2 * hs * 4,
                HTP_TENSOR_COMPUTE,
                scratch,
                bx,
                hs,
            )?;
            self.dump_hidden(session, scratch, layer_idx, "(conv) bx", bx, hs);

            // De-interleave [s0; s1] into conv_x rows 0-1
            // (conv_x is unused on the manual path); the MUL
            // worker only reads dense rows.
            Self::dispatch_cpy_2d(
                session,
                &self.kv_state_buf,
                conv.state_offset,
                hs,
                1,
                8,
                8,
                scratch,
                so.conv_x,
                4,
                hs * 4,
            )?;
            Self::dispatch_cpy_2d(
                session,
                &self.kv_state_buf,
                conv.state_offset + 4,
                hs,
                1,
                8,
                8,
                scratch,
                so.conv_x + hs * 4,
                4,
                hs * 4,
            )?;
            // Rolling conv: y = s0 * w0 + s1 * w1 + bx * w2
            Self::dispatch_mul(
                session,
                &self.weights_buf,
                conv.conv_w0_offset,
                HTP_TENSOR_WEIGHT,
                scratch,
                so.conv_x,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.conv_t0,
                hs,
            )?;
            Self::dispatch_mul(
                session,
                &self.weights_buf,
                conv.conv_w1_offset,
                HTP_TENSOR_WEIGHT,
                scratch,
                so.conv_x + hs * 4,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.conv_t1,
                hs,
            )?;
            Self::dispatch_mul(
                session,
                &self.weights_buf,
                conv.conv_w2_offset,
                HTP_TENSOR_WEIGHT,
                scratch,
                bx,
                HTP_TENSOR_COMPUTE,
                scratch,
                so.conv_y,
                hs,
            )?;
            Self::dispatch_add(
                session, scratch, so.conv_t0, scratch, so.conv_t1, scratch, so.conv_t0, hs,
            )?;
            Self::dispatch_add(
                session, scratch, so.conv_y, scratch, so.conv_t0, scratch, so.conv_y, hs,
            )?;

            // Update states: s0 = s1; s1 = bx. The odd->even
            // shift stages through conv_ssm_y (unused on the
            // manual path).
            Self::dispatch_cpy_2d(
                session,
                &self.kv_state_buf,
                conv.state_offset + 4,
                hs,
                1,
                8,
                8,
                scratch,
                so.conv_ssm_y,
                4,
                hs * 4,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                so.conv_ssm_y,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset,
                8,
                8,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                bx,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset + 4,
                8,
                8,
            )?;
        }

        // Gate: y = y * c
        Self::dispatch_mul(
            session,
            scratch,
            so.conv_in + hs * 4,
            HTP_TENSOR_COMPUTE,
            scratch,
            so.conv_y,
            HTP_TENSOR_COMPUTE,
            scratch,
            so.conv_y,
            hs,
        )?;
        self.dump_hidden(session, scratch, layer_idx, "(conv) gated-y", so.conv_y, hs);

        // out_proj: hs -> hs
        self.dispatch_mul_mat(
            session,
            &self.weights_buf,
            &conv.out_proj,
            scratch,
            so.conv_y,
            scratch,
            cur_normed,
        )?;

        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) block-out",
            cur_normed,
            hs,
        );

        // Residual add: next_act = cur_act + cur_normed
        Self::dispatch_add(
            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) post-block",
            next_act,
            hs,
        );

        self.emit_ffn_block(
            session,
            &conv.ffn,
            conv.ffn_norm_offset,
            None,
            layer_idx,
            next_act,
            cur_normed,
            1,
            true,
        )?;
        Ok(())
    }

    /// Prefill-pass ops of one Conv layer (`m` rows from `start_pos`).
    pub(super) fn emit_conv_prefill(
        &self,
        session: &mut HexagonQueueSession,
        conv: &HexagonConvLayer,
        layer_idx: usize,
        cur_act: usize,
        next_act: usize,
        cur_normed: usize,
        m: usize,
    ) -> Result<(), CeraError> {
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let hs = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        // Private slot for the rows the state writeback reads (see `emit_conv_decode`).
        let stage = so.conv_stage_at(layer_idx);
        // Block norm + in_proj over M rows.
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            cur_act,
            &self.weights_buf,
            conv.attn_norm_offset,
            scratch,
            cur_normed,
            eps,
            hs,
            m,
        )?;
        self.dispatch_mul_mat_m(
            session,
            &self.weights_buf,
            &conv.in_proj,
            scratch,
            cur_normed,
            scratch,
            so.conv_in,
            m,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) prefill conv_in r0",
            so.conv_in,
            3 * hs,
        );
        if m > 1 {
            self.dump_hidden(
                session,
                scratch,
                layer_idx,
                "(conv) prefill conv_in r1",
                so.conv_in + 3 * hs * 4,
                3 * hs,
            );
        }
        self.debug_barrier(session, "prefill conv/in-proj")?;
        // b * x straight out of the strided in_proj thirds (no
        // materializing copies); the DSP reads the row strides.
        Self::dispatch_mul_m_strided(
            session,
            scratch,
            so.conv_in,
            HTP_TENSOR_COMPUTE,
            scratch,
            so.conv_in + 2 * hs * 4,
            HTP_TENSOR_COMPUTE,
            scratch,
            so.conv_bx,
            hs,
            m,
            3 * hs * 4,
            3 * hs * 4,
        )?;
        // State prepend: CONCAT(state-as-[2, hs] + bx-as-[m,
        // hs]) into conv_x `[ncs, hs]`, time-inner: one op
        // replacing the s0/s1 scatter plus the bx transpose.
        Self::dispatch_concat_2d(
            session,
            &self.kv_state_buf,
            conv.state_offset,
            2,
            scratch,
            so.conv_bx,
            m,
            hs * 4,
            4,
            scratch,
            so.conv_x,
            hs,
        )?;
        self.debug_barrier(session, "prefill conv/scatter")?;
        self.dispatch_ssm_conv(
            session,
            &self.weights_buf,
            conv.conv_ssm_offset,
            scratch,
            so.conv_x,
            scratch,
            so.conv_ssm_y,
            3,
            hs,
            m,
        )?;
        self.debug_barrier(session, "prefill conv/ssm-only")?;
        // No transpose: the SsmConv worker writes token t's C
        // values at `t * C` (dst dim-1 stride is the token
        // stride), so `conv_ssm_y` already holds [M, C]
        // row-major. The gate and out_proj below read it
        // directly; the old strided copy was an identity that
        // took the firmware's scalar reshape path (10x wall
        // past m=128).
        // State writeback into the interleaved `[C, 2]` slots
        // (slot t at `state + c*8 + t*4`): last two bx rows
        // when m>=2, else shift + insert.
        if m >= 2 {
            // The last two rows are adjacent: one contiguous copy into this layer's slot.
            Self::dispatch_cpy(
                session,
                scratch,
                so.conv_bx + (m - 2) * hs * 4,
                scratch,
                stage,
                2 * hs,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                stage,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset,
                8,
                8,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                stage + hs * 4,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset + 4,
                8,
                8,
            )?;
        } else {
            Self::dispatch_cpy(session, scratch, so.conv_bx, scratch, stage, hs)?;
            // Shift via scratch temp: odd->even overlaps in
            // the state slab, and CPY has memcpy (not
            // memmove) semantics.
            Self::dispatch_cpy_2d(
                session,
                &self.kv_state_buf,
                conv.state_offset + 4,
                hs,
                1,
                8,
                8,
                scratch,
                so.conv_t0,
                4,
                hs * 4,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                so.conv_t0,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset,
                8,
                8,
            )?;
            Self::dispatch_cpy_2d(
                session,
                scratch,
                stage,
                hs,
                1,
                4,
                hs * 4,
                &self.kv_state_buf,
                conv.state_offset + 4,
                8,
                8,
            )?;
        }
        // Gate with the strided c third in place (no materialize).
        Self::dispatch_mul_m_strided(
            session,
            scratch,
            so.conv_ssm_y,
            HTP_TENSOR_COMPUTE,
            scratch,
            so.conv_in + hs * 4,
            HTP_TENSOR_COMPUTE,
            scratch,
            so.conv_ssm_y,
            hs,
            m,
            hs * 4,
            3 * hs * 4,
        )?;
        self.debug_barrier(session, "prefill conv/ssm")?;
        // out_proj + residual.
        self.dispatch_mul_mat_m(
            session,
            &self.weights_buf,
            &conv.out_proj,
            scratch,
            so.conv_ssm_y,
            scratch,
            cur_normed,
            m,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) prefill conv_out r0",
            cur_normed,
            hs,
        );
        if m > 1 {
            self.dump_hidden(
                session,
                scratch,
                layer_idx,
                "(conv) prefill conv_out r1",
                cur_normed + hs * 4,
                hs,
            );
        }
        Self::dispatch_add_m(
            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs, m,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(conv) prefill block-out",
            next_act + (m - 1) * hs * 4,
            hs,
        );
        self.debug_barrier(session, "prefill conv/out-proj")?;
        self.emit_ffn_block(
            session,
            &conv.ffn,
            conv.ffn_norm_offset,
            None,
            layer_idx,
            next_act,
            cur_normed,
            m,
            false,
        )?;
        Ok(())
    }

    /// Decode-pass ops of one DeltaNet layer (a single token at `pos`).
    pub(super) fn emit_deltanet_decode(
        &self,
        session: &mut HexagonQueueSession,
        dnet: &HexagonDeltaNetLayer,
        layer_idx: usize,
        cur_act: usize,
        next_act: usize,
        cur_normed: usize,
    ) -> Result<(), CeraError> {
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let hs = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let attempts0 = session.dispatch_attempts();
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            cur_act,
            &self.weights_buf,
            dnet.attn_norm_offset,
            scratch,
            cur_normed,
            eps,
            hs,
            1,
        )?;
        self.debug_barrier(session, "DeltaNet attn_norm")?;

        self.dispatch_mul_mat_nx(
            session,
            &self.weights_buf,
            &[&dnet.wqkv, &dnet.wqkv_gate, &dnet.ssm_beta, &dnet.ssm_alpha],
            scratch,
            cur_normed,
            scratch,
            &[so.conv_in, so.conv_y, so.conv_t0, so.conv_t1],
            1,
        )?;
        self.debug_barrier(session, "DeltaNet in_projections")?;

        // Invariant behind `mark_state_torn_if_dispatched`: the host mutates
        // recurrent state only after the batch was flushed to the DSP. (An
        // empty explicit flush is legal when step mode or an op cap already
        // dispatched the group, so this checks the weaker "nothing pending
        // and this block's ops reached a dispatch": either an earlier group
        // flush or this one advanced the attempt counter.)
        session.flush()?;
        debug_assert!(
            session.ops_len() == 0 && session.dispatch_attempts() != attempts0,
            "host recurrent step without a preceding flush"
        );

        self.step_deltanet_recurrence_row(dnet, scratch, &so, 0);

        self.dispatch_mul_mat(
            session,
            &self.weights_buf,
            &dnet.ssm_out,
            scratch,
            so.conv_ssm_y,
            scratch,
            cur_normed,
        )?;

        if let Some(post_norm_offset) = dnet.attn_post_norm_offset {
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                cur_normed,
                &self.weights_buf,
                post_norm_offset,
                scratch,
                cur_normed,
                eps,
                hs,
                1,
            )?;
        }

        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(deltanet) block-out",
            cur_normed,
            hs,
        );

        // Residual add: next_act = cur_act + cur_normed
        Self::dispatch_add(
            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs,
        )?;
        self.dump_hidden(
            session,
            scratch,
            layer_idx,
            "(deltanet) post-block",
            next_act,
            hs,
        );

        // FFN
        self.emit_ffn_block(
            session,
            &dnet.ffn,
            dnet.ffn_norm_offset,
            dnet.ffn_post_norm_offset,
            layer_idx,
            next_act,
            cur_normed,
            1,
            true,
        )?;
        Ok(())
    }

    /// Prefill-pass ops of one DeltaNet layer (`m` rows from `start_pos`).
    pub(super) fn emit_deltanet_prefill(
        &self,
        session: &mut HexagonQueueSession,
        dnet: &HexagonDeltaNetLayer,
        layer_idx: usize,
        cur_act: usize,
        next_act: usize,
        cur_normed: usize,
        m: usize,
    ) -> Result<(), CeraError> {
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let hs = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let attempts0 = session.dispatch_attempts();
        Self::dispatch_rms_norm_mul(
            session,
            scratch,
            cur_act,
            &self.weights_buf,
            dnet.attn_norm_offset,
            scratch,
            cur_normed,
            eps,
            hs,
            m,
        )?;
        self.debug_barrier(session, "prefill deltanet attn_norm")?;

        self.dispatch_mul_mat_nx(
            session,
            &self.weights_buf,
            &[&dnet.wqkv, &dnet.wqkv_gate, &dnet.ssm_beta, &dnet.ssm_alpha],
            scratch,
            cur_normed,
            scratch,
            &[so.conv_in, so.conv_y, so.conv_t0, so.conv_t1],
            m,
        )?;
        self.debug_barrier(session, "prefill deltanet in_projections")?;

        // See the decode site: host recurrent steps need a flushed batch.
        session.flush()?;
        debug_assert!(
            session.ops_len() == 0 && session.dispatch_attempts() != attempts0,
            "host recurrent step without a preceding flush"
        );

        if m > 1 {
            let in_bytes = m * dnet.conv_dim * 4;
            let y_bytes = m * dnet.dt_rank * dnet.d_state * 4;
            let dt_bytes = m * dnet.dt_rank * 4;
            scratch.invalidate_cpu_cache(so.conv_in, in_bytes);
            scratch.invalidate_cpu_cache(so.conv_y, y_bytes);
            scratch.invalidate_cpu_cache(so.conv_t0, dt_bytes);
            scratch.invalidate_cpu_cache(so.conv_t1, dt_bytes);
        }

        for row in 0..m {
            self.step_deltanet_recurrence_row(dnet, scratch, &so, row);
        }

        if m > 1 {
            let y_bytes = m * dnet.dt_rank * dnet.d_state * 4;
            scratch.flush_cpu_cache(so.conv_ssm_y, y_bytes);
        }

        self.dispatch_mul_mat_m(
            session,
            &self.weights_buf,
            &dnet.ssm_out,
            scratch,
            so.conv_ssm_y,
            scratch,
            cur_normed,
            m,
        )?;

        if let Some(post_norm_offset) = dnet.attn_post_norm_offset {
            Self::dispatch_rms_norm_mul(
                session,
                scratch,
                cur_normed,
                &self.weights_buf,
                post_norm_offset,
                scratch,
                cur_normed,
                eps,
                hs,
                m,
            )?;
        }

        Self::dispatch_add_m(
            session, scratch, cur_act, scratch, cur_normed, scratch, next_act, hs, m,
        )?;

        self.emit_ffn_block(
            session,
            &dnet.ffn,
            dnet.ffn_norm_offset,
            dnet.ffn_post_norm_offset,
            layer_idx,
            next_act,
            cur_normed,
            m,
            false,
        )?;
        Ok(())
    }
}

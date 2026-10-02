//! Host-CPU steps that run between DSP flushes: the Gated DeltaNet
//! recurrence, the per-position attention temperature scaling and the
//! activation dump. Everything else in this model is a DSP op emitter (see
//! `ops.rs`).

use super::*;

impl HexagonLfmModel {
    /// Host step between DSP flushes: scale `rows` of `row_len` f32s at
    /// `offset` (row index, factor) in place. Used for the attention
    /// temperature, whose factor is per position; only reached past
    /// `floor_scale`, so the flush cost never hits a short context.
    pub(super) fn host_scale_rows(
        session: &mut HexagonQueueSession,
        scratch: &RpcmemBuffer,
        offset: usize,
        row_len: usize,
        rows: &[(usize, f32)],
    ) -> Result<(), CeraError> {
        if rows.is_empty() {
            return Ok(());
        }
        session.flush().map_err(|e| {
            CeraError::Backend(format!("attention-temperature pre-scale flush failed: {e}"))
        })?;
        let last = rows.iter().map(|&(r, _)| r).max().unwrap_or(0);
        let bytes = (last + 1) * row_len * 4;
        scratch.invalidate_cpu_cache(offset, bytes);
        for &(r, factor) in rows {
            let row = unsafe {
                std::slice::from_raw_parts_mut(
                    scratch.as_mut_ptr().add(offset + r * row_len * 4) as *mut f32,
                    row_len,
                )
            };
            crate::backend::cpu::scale_inplace(row, factor);
        }
        scratch.flush_cpu_cache(offset, bytes);
        Ok(())
    }

    /// Debug helper: flush pending ops, then log RMS/max_abs of a scratch
    /// region. Active only with CERA_DUMP_ACT set. Mirrors the CPU backend's
    /// `[cera.hidden]` log points for cross-backend diffing.
    pub(super) fn dump_hidden(
        &self,
        session: &mut HexagonQueueSession,
        scratch: &RpcmemBuffer,
        layer_idx: usize,
        tag: &str,
        offset: usize,
        len: usize,
    ) {
        if !self.dump_act {
            return;
        }
        if let Err(e) = session.flush() {
            hexagon_error!("dump_hidden flush failed: {e}");
            return;
        }
        scratch.invalidate_cpu_cache(offset, len * 4);
        let act =
            unsafe { std::slice::from_raw_parts(scratch.as_ptr().add(offset) as *const f32, len) };
        let sum: f64 = act.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        let rms = (sum / len as f64).sqrt();
        let absmax = act.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!("cera-hexagon: layer {layer_idx} {tag}: rms={rms:e} max_abs={absmax:e}");
    }

    /// Debug (`CERA_DUMP_ACT`): flush, then log what the routed FFN of `layer_idx`
    /// computed for token row `row`: the experts the DSP selected, their
    /// weights and renormalization divisor, and each stage's RMS.
    pub(super) fn dump_moe_row(
        &self,
        session: &mut HexagonQueueSession,
        moe: &HexagonMoeFfn,
        layer_idx: usize,
        row: usize,
        out_offset: usize,
    ) {
        if let Err(e) = session.flush() {
            hexagon_error!("dump_moe_row flush failed: {e}");
            return;
        }
        let so = &self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let (n_exp, n_used, ff, hs) = (
            moe.n_expert,
            moe.n_expert_used,
            moe.expert_ff_len,
            self.config.hidden_size,
        );
        let f32s = |off: usize, n: usize| -> Vec<f32> {
            scratch.invalidate_cpu_cache(off, n * 4);
            unsafe { std::slice::from_raw_parts(scratch.as_ptr().add(off) as *const f32, n) }
                .to_vec()
        };
        let rms = |v: &[f32]| {
            (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let ids_off = so.moe_selected_ids + row * n_exp * 4;
        scratch.invalidate_cpu_cache(ids_off, n_exp * 4);
        let ids: Vec<i32> = unsafe {
            std::slice::from_raw_parts(scratch.as_ptr().add(ids_off) as *const i32, n_used)
        }
        .to_vec();
        let weights = f32s(so.moe_selected_weights + row * n_used * 4, n_used);
        let denom = f32s(so.moe_renorm + row * MOE_RENORM_SLOT_BYTES + 16, 1)[0];
        let probs = f32s(so.moe_probs + row * n_exp * 4, n_exp);
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe row {row}: ids={ids:?} weights={weights:?} denom={denom:e}"
        );
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe probs[ids]={:?}",
            ids.iter()
                .map(|&i| probs.get(i as usize).copied().unwrap_or(f32::NAN))
                .collect::<Vec<_>>()
        );
        let per = |base: usize, width: usize| -> Vec<String> {
            (0..n_used)
                .map(|e| {
                    format!(
                        "{:.4e}",
                        rms(&f32s(base + (row * n_used + e) * width * 4, width))
                    )
                })
                .collect()
        };
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe gate rms/expert={:?}",
            per(so.moe_gate, ff)
        );
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe up rms/expert={:?}",
            per(so.moe_up, ff)
        );
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe swiglu rms/expert={:?}",
            per(so.moe_swiglu, ff)
        );
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe down rms/expert={:?}",
            per(so.moe_down, hs)
        );
        let out = f32s(out_offset, hs);
        eprintln!(
            "cera-hexagon: layer {layer_idx} moe out rms={:.6e}",
            rms(&out)
        );
    }

    /// Host-CPU Gated DeltaNet step for one token row: advances the conv and
    /// SSM state in place in `kv_state_buf` and writes the gated output row.
    pub(super) fn step_deltanet_recurrence_row(
        &self,
        dnet: &HexagonDeltaNetLayer,
        scratch: &RpcmemBuffer,
        so: &ScratchOffsets,
        row_idx: usize,
    ) {
        let num_k_heads = dnet.n_group;
        let num_v_heads = dnet.dt_rank;
        let head_k_dim = dnet.d_state;
        let head_v_dim = dnet.d_state;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = dnet.conv_dim;
        let d_conv = dnet.d_conv;
        let eps = self.config.rms_norm_eps;

        let in_row_bytes = conv_dim * 4;
        let y_row_bytes = value_dim * 4;
        let dt_bytes = num_v_heads * 4;

        scratch.invalidate_cpu_cache(so.conv_in + row_idx * in_row_bytes, in_row_bytes);
        scratch.invalidate_cpu_cache(so.conv_y + row_idx * y_row_bytes, y_row_bytes);
        scratch.invalidate_cpu_cache(so.conv_t0 + row_idx * dt_bytes, dt_bytes);
        scratch.invalidate_cpu_cache(so.conv_t1 + row_idx * dt_bytes, dt_bytes);

        let qkv_mixed = unsafe {
            std::slice::from_raw_parts(
                scratch.as_ptr().add(so.conv_in + row_idx * in_row_bytes) as *const f32,
                conv_dim,
            )
        };
        let z = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.conv_y + row_idx * y_row_bytes) as *mut f32,
                value_dim,
            )
        };
        let beta_raw = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.conv_t0 + row_idx * dt_bytes) as *mut f32,
                num_v_heads,
            )
        };
        let alpha_raw = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.conv_t1 + row_idx * dt_bytes) as *mut f32,
                num_v_heads,
            )
        };
        let conv_out = unsafe {
            std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.conv_x + row_idx * in_row_bytes) as *mut f32,
                conv_dim,
            )
        };
        let core_out = unsafe {
            std::slice::from_raw_parts_mut(
                scratch
                    .as_mut_ptr()
                    .add(so.conv_ssm_y + row_idx * y_row_bytes) as *mut f32,
                value_dim,
            )
        };

        let conv_state_len = conv_dim * (d_conv.saturating_sub(1));
        let ssm_state_len = num_v_heads * head_v_dim * head_v_dim;
        let conv_state = unsafe {
            std::slice::from_raw_parts_mut(
                self.kv_state_buf.as_mut_ptr().add(dnet.conv_state_offset) as *mut f32,
                conv_state_len,
            )
        };
        let ssm_state = unsafe {
            std::slice::from_raw_parts_mut(
                self.kv_state_buf.as_mut_ptr().add(dnet.ssm_state_offset) as *mut f32,
                ssm_state_len,
            )
        };

        let ssm_conv1d = unsafe {
            std::slice::from_raw_parts(
                self.weights_buf.as_ptr().add(dnet.ssm_conv1d_offset) as *const f32,
                conv_dim * d_conv,
            )
        };
        let ssm_conv1d_bias = dnet.ssm_conv1d_bias_offset.map(|off| unsafe {
            std::slice::from_raw_parts(self.weights_buf.as_ptr().add(off) as *const f32, conv_dim)
        });
        let ssm_dt = unsafe {
            std::slice::from_raw_parts(
                self.weights_buf.as_ptr().add(dnet.ssm_dt_offset) as *const f32,
                num_v_heads,
            )
        };
        let ssm_a = unsafe {
            std::slice::from_raw_parts(
                self.weights_buf.as_ptr().add(dnet.ssm_a_offset) as *const f32,
                num_v_heads,
            )
        };
        let ssm_norm = unsafe {
            std::slice::from_raw_parts(
                self.weights_buf.as_ptr().add(dnet.ssm_norm_offset) as *const f32,
                dnet.d_state,
            )
        };

        crate::backend::cpu::sigmoid_inplace(beta_raw);
        for (h, a) in alpha_raw.iter_mut().enumerate().take(num_v_heads) {
            let alpha_biased = *a + ssm_dt[h];
            let alpha_sp = crate::backend::cpu::softplus(alpha_biased);
            let gate_h = (alpha_sp * ssm_a[h]).clamp(-80.0, 0.0);
            *a = gate_h.exp();
        }

        crate::backend::cpu::mamba2_conv1d_step(
            qkv_mixed,
            conv_state,
            ssm_conv1d,
            ssm_conv1d_bias,
            conv_dim,
            d_conv,
            conv_out,
        );

        let (q_part, rest) = conv_out.split_at_mut(key_dim);
        let (k_part, v_part) = rest.split_at_mut(key_dim);

        let q_scale = 1.0 / (head_k_dim as f32).sqrt();
        for kh in 0..num_k_heads {
            let q_head = &mut q_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let sum_sq = crate::backend::cpu::dot_f32(q_head, q_head);
            let l2 = sum_sq.sqrt().max(eps);
            let factor = q_scale / l2;
            crate::backend::cpu::scale_inplace(q_head, factor);

            let k_head = &mut k_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let sum_sq_k = crate::backend::cpu::dot_f32(k_head, k_head);
            let l2_k = sum_sq_k.sqrt().max(eps);
            let factor_k = 1.0 / l2_k;
            crate::backend::cpu::scale_inplace(k_head, factor_k);
        }

        let s_dim = head_v_dim;
        let mut sk_buf = [0.0f32; 256];
        let mut d_buf = [0.0f32; 256];
        let sk = &mut sk_buf[..s_dim];
        let d = &mut d_buf[..s_dim];
        let heads_per_group = (num_v_heads / num_k_heads.max(1)).max(1);

        for h in 0..num_v_heads {
            let kh = (h / heads_per_group).min(num_k_heads.saturating_sub(1));
            let q = &q_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let k = &k_part[kh * head_k_dim..(kh + 1) * head_k_dim];
            let v = &v_part[h * head_v_dim..(h + 1) * head_v_dim];
            let dec = alpha_raw[h];
            let b = beta_raw[h];

            let state_offset = h * s_dim * s_dim;
            let s_mat = &mut ssm_state[state_offset..state_offset + s_dim * s_dim];

            if !dec.is_finite() || dec <= 0.0 {
                s_mat.fill(0.0);
            } else if (dec - 1.0).abs() > 1e-7 {
                crate::backend::cpu::scale_inplace(s_mat, dec);
            }

            sk.fill(0.0);
            for i in 0..s_dim {
                let ki = k[i];
                if ki == 0.0 {
                    continue;
                }
                let row = &s_mat[i * s_dim..(i + 1) * s_dim];
                for (sk_elem, &r) in sk.iter_mut().zip(row.iter()) {
                    *sk_elem += r * ki;
                }
            }

            for ((dj, &vj), &skj) in d.iter_mut().zip(v.iter()).zip(sk.iter()) {
                *dj = b * (vj - skj);
            }

            let o_head = &mut core_out[h * head_v_dim..(h + 1) * head_v_dim];
            o_head.fill(0.0);
            for i in 0..s_dim {
                let ki = k[i];
                let qi = q[i];
                let row_offset = i * s_dim;
                let row = &mut s_mat[row_offset..row_offset + s_dim];
                for ((r, &dj), o) in row.iter_mut().zip(d.iter()).zip(o_head.iter_mut()) {
                    let updated = *r + ki * dj;
                    *r = updated;
                    *o += updated * qi;
                }
            }
        }

        crate::backend::cpu::silu_inplace(z);
        for h in 0..num_v_heads {
            let o_head = &mut core_out[h * head_v_dim..(h + 1) * head_v_dim];
            let z_head = &z[h * head_v_dim..(h + 1) * head_v_dim];

            crate::backend::cpu::rmsnorm(o_head, ssm_norm, eps);
            crate::backend::cpu::mul_inplace(o_head, z_head);
        }

        scratch.flush_cpu_cache(so.conv_ssm_y + row_idx * y_row_bytes, y_row_bytes);
    }
}

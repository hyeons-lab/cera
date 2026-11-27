//! Native AMD XDNA2 NPU audio decoder (Depthformer + Detokenizer).
//!
//! Offloads all linear projections (depth_linear, depthformer QKV/out/SwiGLU/to_logits,
//! detokenizer ShortConv in/out projections, attention Q/K/V/O/gate, SwiGLU FFN, and STFT
//! linear head) to the shared [`XdnaGemmRunner`] bf16 GEMM kernel on XDNA2 AIE tiles, keeping
//! streaming convolutional/KV state and deterministic PRNG sampling coherent across frames.
//!
//! Supports both:
//! - `LFM2.5-Audio` (`lfm2a`): 8-codebook autoregressive depthformer (`Interleaved` RoPE) +
//!   non-causal 6x upsampled LFM2 detokenizer.
//! - `LFM3-Audio` / `LFM3-Omni` (`lfm3a_gen`): 12-codebook 3-step MaskGIT bidirectional
//!   depthformer (`ShortConv` + `NeoX` RoPE attention + CTRL repetition penalty) + causal
//!   FSQ LFM3 detokenizer (`1->6` upsample or direct `6 * n_bins` STFT head).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::backend::cpu::{
    add_inplace, conv1d, dot_f32, rmsnorm, silu_mul_inplace, softmax_inplace,
};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{
    AudioAccelerator, AudioDecoderWeights, DecoderConfig, DepthformerConfig, DetokenizerConfig,
    DetokenizerWeights, LFM3A_REP_PENALTY, LFM3A_REP_PENALTY_MAX_FRAMES, apply_rope_interleaved,
    apply_rope_neox, detok_embed_codes, istft_to_pcm, next_atomic_xorshift64_f32_01,
    non_zero_rng_seed, seed_to_rng_state, upsample,
};
use crate::model::weights::MmapWeight;
use crate::model::xdna_towers::{XdnaGemmRunner, XdnaGemmWeight};
use crate::model::{record_first_fault, take_fault};
use crate::sampler::argmax;
use crate::session::CeraError;

// ── Uploaded weights ──────────────────────────────────────────────────────────

struct XdnaCodebookWeights {
    embedding: MmapWeight,
    norm: Vec<f32>,
    w_to_logits: XdnaGemmWeight,
    n_vocab: usize,
    n_embd: usize,
}

struct XdnaDepthformerLayer {
    operator_norm: Vec<f32>,
    w_qkv: XdnaGemmWeight,
    qkv_rows: usize,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    w_o: XdnaGemmWeight,
    o_rows: usize,
    o_cols: usize,
    ffn_norm: Vec<f32>,
    w_1: XdnaGemmWeight,
    w_2: XdnaGemmWeight,
    w_3: XdnaGemmWeight,
}

struct XdnaDfState {
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    n_past: usize,
    code_history: VecDeque<Vec<i32>>,
}

impl XdnaDfState {
    fn new(cfg: &DepthformerConfig) -> Self {
        let max_positions = cfg.max_seq_len.max(8);
        let kv_len = max_positions * cfg.n_head_kv * cfg.n_embd_head;
        let kv = (0..cfg.n_layer)
            .map(|_| (vec![0.0f32; kv_len], vec![0.0f32; kv_len]))
            .collect();
        Self {
            kv,
            n_past: 0,
            code_history: VecDeque::with_capacity(LFM3A_REP_PENALTY_MAX_FRAMES),
        }
    }

    fn reset_kv(&mut self) {
        for (k, v) in &mut self.kv {
            k.fill(0.0);
            v.fill(0.0);
        }
        self.n_past = 0;
    }

    fn push_code_history(&mut self, codes: Vec<i32>) {
        if self.code_history.len() >= LFM3A_REP_PENALTY_MAX_FRAMES {
            self.code_history.pop_front();
        }
        self.code_history.push_back(codes);
    }
}

struct XdnaDepthformer {
    lfm3a_gen: bool,
    depthformer_config: DepthformerConfig,
    decoder_config: DecoderConfig,
    depth_linear_rows: usize,
    depth_linear_cols: usize,
    depth_linear_b: Vec<f32>,
    audio_embedding_norm: Vec<f32>,
    w_depth_linear: XdnaGemmWeight,
    layers: Vec<XdnaDepthformerLayer>,
    codebooks: Vec<XdnaCodebookWeights>,
    state: Mutex<XdnaDfState>,
    rng_state: AtomicU64,
}

impl XdnaDepthformer {
    fn new(runner: &XdnaGemmRunner, weights: &AudioDecoderWeights) -> Result<Self> {
        let w_depth_linear = runner.upload_mmap_weight(&weights.depth_linear_w)?;
        let mut layers = Vec::with_capacity(weights.depthformer_layers.len());
        for lw in &weights.depthformer_layers {
            layers.push(XdnaDepthformerLayer {
                operator_norm: lw.operator_norm.clone(),
                w_qkv: runner.upload_mmap_weight(&lw.wqkv)?,
                qkv_rows: lw.wqkv.rows,
                q_norm: lw.q_norm.clone(),
                k_norm: lw.k_norm.clone(),
                w_o: runner.upload_mmap_weight(&lw.wo)?,
                o_rows: lw.wo.rows,
                o_cols: lw.wo.cols,
                ffn_norm: lw.ffn_norm.clone(),
                w_1: runner.upload_mmap_weight(&lw.w1)?,
                w_2: runner.upload_mmap_weight(&lw.w2)?,
                w_3: runner.upload_mmap_weight(&lw.w3)?,
            });
        }
        let mut codebooks = Vec::with_capacity(weights.depth_embeddings.len());
        for cb in &weights.depth_embeddings {
            codebooks.push(XdnaCodebookWeights {
                embedding: cb.embedding.clone(),
                norm: cb.norm.clone(),
                w_to_logits: runner.upload_mmap_weight(&cb.to_logits)?,
                n_vocab: cb.to_logits.rows,
                n_embd: cb.to_logits.cols,
            });
        }
        let state = Mutex::new(XdnaDfState::new(&weights.depthformer_config));
        let rng_state = AtomicU64::new(non_zero_rng_seed(rand::random::<u64>()));
        Ok(Self {
            lfm3a_gen: weights.is_lfm3a_gen(),
            depthformer_config: weights.depthformer_config.clone(),
            decoder_config: weights.decoder_config.clone(),
            depth_linear_rows: weights.depth_linear_w.rows,
            depth_linear_cols: weights.depth_linear_w.cols,
            depth_linear_b: weights.depth_linear_b.clone(),
            audio_embedding_norm: weights.audio_embedding.norm.clone(),
            w_depth_linear,
            layers,
            codebooks,
            state,
            rng_state,
        })
    }

    fn reset(&self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.reset_kv();
        st.code_history.clear();
    }

    fn seed_rng(&self, seed: u64) {
        self.rng_state
            .store(seed_to_rng_state(seed), Ordering::Relaxed);
    }

    fn next_f32_01(&self) -> f32 {
        next_atomic_xorshift64_f32_01(&self.rng_state)
    }

    /// Project LLM hidden state into either a shared `[n_embd]` vector or per-codebook `[n_cb, n_embd]` matrix.
    fn project_llm_input(
        &self,
        runner: &XdnaGemmRunner,
        embedding: &[f32],
        n_cb: usize,
    ) -> Result<(Vec<f32>, bool)> {
        let n_embd = self.depthformer_config.n_embd;
        let llm_dim = self.depth_linear_cols;
        let dl_rows = self.depth_linear_rows;
        anyhow::ensure!(n_embd > 0, "XDNA depthformer requires n_embd > 0");
        let shared_llm_proj = dl_rows == n_embd;
        let required_rows = if shared_llm_proj {
            n_embd
        } else {
            n_cb.saturating_mul(n_embd)
        };
        anyhow::ensure!(
            dl_rows >= required_rows,
            "XDNA depthformer depth_linear rows ({dl_rows}) < required ({required_rows}) for {n_cb} codebooks"
        );

        let mut llm_in = vec![0.0f32; llm_dim];
        let copy_len = embedding.len().min(llm_dim);
        llm_in[..copy_len].copy_from_slice(&embedding[..copy_len]);

        let mut llm_proj_buf =
            runner.matmul_f32(&llm_in, &self.w_depth_linear, 1, dl_rows, llm_dim)?;
        anyhow::ensure!(
            llm_proj_buf.len() >= required_rows,
            "XDNA depthformer projection output len ({}) < required ({required_rows})",
            llm_proj_buf.len()
        );
        for (dst, &b) in llm_proj_buf.iter_mut().zip(self.depth_linear_b.iter()) {
            *dst += b;
        }
        Ok((llm_proj_buf, shared_llm_proj))
    }

    /// Run one autoregressive LFM2A frame (`8` codebooks).
    fn sample_frame_lfm2a(
        &self,
        runner: &XdnaGemmRunner,
        embedding: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<[i32; 8]> {
        let dec = &self.decoder_config;
        let cfg = &self.depthformer_config;
        let n_cb = dec.n_codebook.min(8);
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let n_kv = cfg.n_head_kv;
        let hd = cfg.n_embd_head;
        anyhow::ensure!(
            n_embd > 0 && n_head > 0 && n_kv > 0 && hd > 0,
            "XDNA depthformer requires positive attention and embedding dimensions"
        );
        anyhow::ensure!(
            self.codebooks.len() >= n_cb,
            "XDNA depthformer codebooks ({}) < {n_cb}",
            self.codebooks.len()
        );
        let q_dim = n_head * hd;
        let kv_dim = n_kv * hd;
        let group_size = (n_head / n_kv).max(1);
        let scale = 1.0f32 / (hd as f32).sqrt();

        // Project LLM hidden -> codebook inputs in a single XDNA GEMM call.
        let (all_llm_proj, shared_llm_proj) = self.project_llm_input(runner, embedding, n_cb)?;

        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.reset_kv();

        let mut codes = [0i32; 8];
        let mut prev_token: i32 = -1;
        let mut cur = vec![0.0f32; n_embd];
        let mut normed = vec![0.0f32; n_embd];
        let mut emb_row = vec![0.0f32; n_embd];
        let mut scores = vec![0.0f32; cfg.max_seq_len.max(8)];
        let mut attn_out = vec![0.0f32; q_dim];

        for (j, code_slot) in codes.iter_mut().enumerate().take(n_cb) {
            let pos = st.n_past;
            let proj_src = if shared_llm_proj {
                &all_llm_proj[..n_embd]
            } else {
                &all_llm_proj[j * n_embd..(j + 1) * n_embd]
            };
            cur.copy_from_slice(proj_src);

            if j > 0 && prev_token >= 0 {
                let prev_cb = &self.codebooks[j - 1];
                let tok = prev_token as usize;
                if tok < prev_cb.embedding.rows {
                    prev_cb.embedding.dequantize_row(tok, &mut emb_row);
                    add_inplace(&mut cur, &emb_row);
                }
            }

            for (il, lw) in self.layers.iter().enumerate() {
                normed.copy_from_slice(&cur);
                rmsnorm(&mut normed, &lw.operator_norm, cfg.rms_norm_eps);

                let mut qkv = runner.matmul_f32(&normed, &lw.w_qkv, 1, lw.qkv_rows, n_embd)?;
                for h in 0..n_head {
                    let s = &mut qkv[h * hd..(h + 1) * hd];
                    rmsnorm(s, &lw.q_norm, cfg.rms_norm_eps);
                    apply_rope_interleaved(s, pos, hd, cfg.rope_freq_base);
                }
                for h in 0..n_kv {
                    let s = &mut qkv[q_dim + h * hd..q_dim + (h + 1) * hd];
                    rmsnorm(s, &lw.k_norm, cfg.rms_norm_eps);
                    apply_rope_interleaved(s, pos, hd, cfg.rope_freq_base);
                }

                let (k_cache, v_cache) = &mut st.kv[il];
                for h in 0..n_kv {
                    let cache_off = pos * kv_dim + h * hd;
                    k_cache[cache_off..cache_off + hd]
                        .copy_from_slice(&qkv[q_dim + h * hd..q_dim + (h + 1) * hd]);
                    v_cache[cache_off..cache_off + hd].copy_from_slice(
                        &qkv[q_dim + kv_dim + h * hd..q_dim + kv_dim + (h + 1) * hd],
                    );
                }

                let seq_len = pos + 1;
                attn_out.fill(0.0);
                for h in 0..n_head {
                    let kv_h = h / group_size;
                    let qh = &qkv[h * hd..(h + 1) * hd];
                    let sc = &mut scores[..seq_len];
                    for (tt, s) in sc.iter_mut().enumerate().take(seq_len) {
                        let ko = tt * kv_dim + kv_h * hd;
                        *s = dot_f32(qh, &k_cache[ko..ko + hd]) * scale;
                    }
                    softmax_inplace(sc);
                    let oh = &mut attn_out[h * hd..(h + 1) * hd];
                    for (tt, &w) in sc.iter().enumerate().take(seq_len) {
                        let vo = tt * kv_dim + kv_h * hd;
                        for (dst, &val) in oh.iter_mut().zip(&v_cache[vo..vo + hd]) {
                            *dst += w * val;
                        }
                    }
                }

                let proj = runner.matmul_f32(&attn_out, &lw.w_o, 1, lw.o_rows, lw.o_cols)?;
                add_inplace(&mut cur, &proj);

                normed.copy_from_slice(&cur);
                rmsnorm(&mut normed, &lw.ffn_norm, cfg.rms_norm_eps);
                let mut gate = runner.matmul_f32(&normed, &lw.w_1, 1, cfg.ffn_dim, n_embd)?;
                let up = runner.matmul_f32(&normed, &lw.w_3, 1, cfg.ffn_dim, n_embd)?;
                silu_mul_inplace(&mut gate, &up);
                let down = runner.matmul_f32(&gate, &lw.w_2, 1, n_embd, cfg.ffn_dim)?;
                add_inplace(&mut cur, &down);
            }
            st.n_past += 1;

            let cb = &self.codebooks[j];
            normed.copy_from_slice(&cur);
            rmsnorm(&mut normed, &cb.norm, dec.rms_norm_eps);
            let mut logits =
                runner.matmul_f32(&normed, &cb.w_to_logits, 1, cb.n_vocab, cb.n_embd)?;

            let sampled = if logits.is_empty() {
                0
            } else if !temperature.is_finite() || temperature <= 0.0 || top_k <= 1 {
                argmax(&logits) as i32
            } else {
                let inv_temp = 1.0 / temperature;
                for l in &mut logits {
                    *l *= inv_temp;
                }
                softmax_inplace(&mut logits);
                let n_logits = logits.len();
                let k = top_k.min(n_logits).max(1);
                let mut indices: Vec<usize> = (0..n_logits).collect();
                if k < n_logits {
                    indices.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
                }
                let top_indices = &indices[..k];
                let sum: f32 = top_indices.iter().map(|&i| logits[i]).sum();
                let mut r = self.next_f32_01() * sum;
                let mut picked = *top_indices.last().unwrap_or(&0);
                for &i in top_indices {
                    r -= logits[i];
                    if r <= 0.0 {
                        picked = i;
                        break;
                    }
                }
                picked as i32
            };

            *code_slot = sampled;
            prev_token = sampled;
        }

        Ok(codes)
    }

    /// Run one 12-codebook `lfm3a_gen` MaskGIT depthformer frame via batched XDNA GEMMs.
    fn sample_frame_lfm3a(
        &self,
        runner: &XdnaGemmRunner,
        embedding: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<Vec<i32>> {
        let base_t = if temperature.is_finite() && temperature > 0.0 {
            temperature
        } else {
            0.0
        };
        let step_temps = [
            base_t * (1.0 - (1.0f32 / 3.0).powi(3)),
            base_t * (1.0 - (2.0f32 / 3.0).powi(3)),
            0.0,
        ];
        self.sample_frame_lfm3a_with_temps(runner, embedding, step_temps, top_k)
    }

    fn sample_frame_lfm3a_with_temps(
        &self,
        runner: &XdnaGemmRunner,
        embedding: &[f32],
        step_temps: [f32; 3],
        top_k: usize,
    ) -> Result<Vec<i32>> {
        let cfg = &self.decoder_config;
        let df_cfg = &self.depthformer_config;
        let n_cb = cfg.n_codebook;
        if n_cb == 0 {
            return Ok(Vec::new());
        }
        let n_vocab = cfg.n_vocab;
        let mask_id = n_vocab.saturating_sub(1) as i32;
        let n_embd = df_cfg.n_embd;
        let n_head = df_cfg.n_head;
        let n_kv = df_cfg.n_head_kv;
        let hd = df_cfg.n_embd_head;
        anyhow::ensure!(
            n_embd > 0 && n_head > 0 && n_kv > 0 && hd > 0,
            "XDNA lfm3a depthformer requires positive attention and embedding dimensions"
        );
        anyhow::ensure!(
            self.codebooks.len() >= n_cb,
            "XDNA lfm3a depthformer codebooks ({}) < {n_cb}",
            self.codebooks.len()
        );
        let eps = df_cfg.rms_norm_eps;
        let (llm_proj_buf, shared_llm_proj) = self.project_llm_input(runner, embedding, n_cb)?;

        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut codes = vec![mask_id; n_cb];
        let mut unmasked = vec![false; n_cb];
        let unmasked_per_step = [4usize, 4, n_cb.saturating_sub(8)];
        let mut seq = vec![0.0f32; n_cb * n_embd];
        let mut tok_row = vec![0.0f32; n_embd];
        let mut norm_mat = vec![0.0f32; n_cb * n_embd];

        for (step, &k_unmask) in unmasked_per_step.iter().enumerate() {
            // 1. Build `[n_cb, n_embd]` sequence input for this MaskGIT step.
            for j in 0..n_cb {
                let dst = &mut seq[j * n_embd..(j + 1) * n_embd];
                if shared_llm_proj {
                    dst.copy_from_slice(&llm_proj_buf);
                } else {
                    dst.copy_from_slice(&llm_proj_buf[j * n_embd..(j + 1) * n_embd]);
                }
                let cb_emb = &self.codebooks[j].embedding;
                let tok_idx = (codes[j].max(0) as usize).min(cb_emb.rows.saturating_sub(1));
                cb_emb.dequantize_row(tok_idx, &mut tok_row);
                add_inplace(dst, &tok_row);
            }

            // 2. Run bidirectional LFM3 depthformer stack over all `n_cb` tokens simultaneously.
            for lw in &self.layers {
                let is_shortconv = lw.k_norm.is_empty();
                norm_mat.copy_from_slice(&seq);
                for s in 0..n_cb {
                    rmsnorm(
                        &mut norm_mat[s * n_embd..(s + 1) * n_embd],
                        &lw.operator_norm,
                        eps,
                    );
                }

                if is_shortconv {
                    let kernel_size = (lw.q_norm.len() / n_embd).max(1);
                    let pad_total = kernel_size - 1;
                    let pad_left = pad_total / 2;
                    let t_padded = n_cb + pad_total;
                    let bcx_mat =
                        runner.matmul_f32(&norm_mat, &lw.w_qkv, n_cb, lw.qkv_rows, n_embd)?;

                    let mut bx_padded = vec![0.0f32; n_embd * t_padded];
                    let mut c_gate = vec![0.0f32; n_cb * n_embd];
                    for s in 0..n_cb {
                        let bcx = &bcx_mat[s * 3 * n_embd..(s + 1) * 3 * n_embd];
                        let (b_part, rest) = bcx.split_at(n_embd);
                        let (c_part, x_part) = rest.split_at(n_embd);
                        c_gate[s * n_embd..(s + 1) * n_embd].copy_from_slice(c_part);
                        for ch in 0..n_embd {
                            bx_padded[ch * t_padded + pad_left + s] = b_part[ch] * x_part[ch];
                        }
                    }

                    let mut conv_out = vec![0.0f32; n_embd * n_cb];
                    conv1d(
                        &bx_padded,
                        &lw.q_norm,
                        None,
                        &mut conv_out,
                        n_embd,
                        n_embd,
                        t_padded,
                        kernel_size,
                        1,
                        0,
                        n_embd,
                    );

                    for s in 0..n_cb {
                        let c_row = &c_gate[s * n_embd..(s + 1) * n_embd];
                        let y_in = &mut norm_mat[s * n_embd..(s + 1) * n_embd];
                        for ch in 0..n_embd {
                            y_in[ch] = conv_out[ch * n_cb + s] * c_row[ch];
                        }
                    }
                    let proj = runner.matmul_f32(&norm_mat, &lw.w_o, n_cb, lw.o_rows, lw.o_cols)?;
                    add_inplace(&mut seq, &proj);
                } else {
                    let q_dim = n_head * hd;
                    let k_dim = n_kv * hd;
                    let qkv_dim = q_dim + 2 * k_dim;
                    let gate_rows = lw.qkv_rows.saturating_sub(qkv_dim);
                    let group_size = (n_head / n_kv.max(1)).max(1);
                    let scale = 1.0f32 / (hd as f32).sqrt();

                    let qkv_mat =
                        runner.matmul_f32(&norm_mat, &lw.w_qkv, n_cb, lw.qkv_rows, n_embd)?;

                    let mut all_q = vec![0.0f32; n_cb * q_dim];
                    let mut all_k = vec![0.0f32; n_cb * k_dim];
                    let mut all_v = vec![0.0f32; n_cb * k_dim];
                    let mut all_gate = vec![0.0f32; n_cb * gate_rows];

                    for s in 0..n_cb {
                        let qkv = &qkv_mat[s * lw.qkv_rows..(s + 1) * lw.qkv_rows];
                        let q_row = &mut all_q[s * q_dim..(s + 1) * q_dim];
                        let k_row = &mut all_k[s * k_dim..(s + 1) * k_dim];
                        let v_row = &mut all_v[s * k_dim..(s + 1) * k_dim];
                        q_row.copy_from_slice(&qkv[..q_dim]);
                        k_row.copy_from_slice(&qkv[q_dim..q_dim + k_dim]);
                        v_row.copy_from_slice(&qkv[q_dim + k_dim..qkv_dim]);
                        if gate_rows > 0 {
                            all_gate[s * gate_rows..(s + 1) * gate_rows]
                                .copy_from_slice(&qkv[qkv_dim..]);
                        }
                        for h in 0..n_head {
                            let qh = &mut q_row[h * hd..(h + 1) * hd];
                            rmsnorm(qh, &lw.q_norm, eps);
                            apply_rope_neox(qh, s, hd, df_cfg.rope_freq_base);
                        }
                        for h_kv in 0..n_kv {
                            let kh = &mut k_row[h_kv * hd..(h_kv + 1) * hd];
                            rmsnorm(kh, &lw.k_norm, eps);
                            apply_rope_neox(kh, s, hd, df_cfg.rope_freq_base);
                        }
                    }

                    let mut attn_out_mat = vec![0.0f32; n_cb * q_dim];
                    let mut scores = vec![0.0f32; n_cb];
                    for q_pos in 0..n_cb {
                        let attn_out = &mut attn_out_mat[q_pos * q_dim..(q_pos + 1) * q_dim];
                        for h in 0..n_head {
                            let kv_h = h / group_size;
                            let qh = &all_q[q_pos * q_dim + h * hd..q_pos * q_dim + (h + 1) * hd];
                            for k_pos in 0..n_cb {
                                let kh = &all_k
                                    [k_pos * k_dim + kv_h * hd..k_pos * k_dim + (kv_h + 1) * hd];
                                scores[k_pos] = dot_f32(qh, kh) * scale;
                            }
                            softmax_inplace(&mut scores);
                            let out_h = &mut attn_out[h * hd..(h + 1) * hd];
                            for k_pos in 0..n_cb {
                                let vh = &all_v
                                    [k_pos * k_dim + kv_h * hd..k_pos * k_dim + (kv_h + 1) * hd];
                                let w = scores[k_pos];
                                for d in 0..hd {
                                    out_h[d] += w * vh[d];
                                }
                            }
                        }
                        if gate_rows > 0 {
                            let g_row = &all_gate[q_pos * gate_rows..(q_pos + 1) * gate_rows];
                            if gate_rows == n_head {
                                for h in 0..n_head {
                                    let g = 1.0f32 / (1.0f32 + (-g_row[h]).exp());
                                    for v in &mut attn_out[h * hd..(h + 1) * hd] {
                                        *v *= g;
                                    }
                                }
                            } else {
                                for (v, &g) in attn_out.iter_mut().zip(g_row.iter()) {
                                    *v *= 1.0f32 / (1.0f32 + (-g).exp());
                                }
                            }
                        }
                    }
                    let proj =
                        runner.matmul_f32(&attn_out_mat, &lw.w_o, n_cb, lw.o_rows, lw.o_cols)?;
                    add_inplace(&mut seq, &proj);
                }

                norm_mat.copy_from_slice(&seq);
                for s in 0..n_cb {
                    rmsnorm(
                        &mut norm_mat[s * n_embd..(s + 1) * n_embd],
                        &lw.ffn_norm,
                        eps,
                    );
                }
                let mut gate_mat =
                    runner.matmul_f32(&norm_mat, &lw.w_1, n_cb, df_cfg.ffn_dim, n_embd)?;
                let up_mat = runner.matmul_f32(&norm_mat, &lw.w_3, n_cb, df_cfg.ffn_dim, n_embd)?;
                silu_mul_inplace(&mut gate_mat, &up_mat);
                let down_mat =
                    runner.matmul_f32(&gate_mat, &lw.w_2, n_cb, n_embd, df_cfg.ffn_dim)?;
                add_inplace(&mut seq, &down_mat);
            }

            if self.audio_embedding_norm.len() == n_embd {
                for s in 0..n_cb {
                    rmsnorm(
                        &mut seq[s * n_embd..(s + 1) * n_embd],
                        &self.audio_embedding_norm,
                        eps,
                    );
                }
            }

            // 3. Score remaining masked codebooks and sample.
            let t_step = step_temps[step];
            let mut cand_tokens = vec![0i32; n_cb];
            let mut confidences = vec![f32::NEG_INFINITY; n_cb];

            for j in 0..n_cb {
                if unmasked[j] {
                    continue;
                }
                let cb = &self.codebooks[j];
                let mut normed_row = seq[j * n_embd..(j + 1) * n_embd].to_vec();
                rmsnorm(&mut normed_row, &cb.norm, cfg.rms_norm_eps);
                let mut logits =
                    runner.matmul_f32(&normed_row, &cb.w_to_logits, 1, cb.n_vocab, cb.n_embd)?;

                if !st.code_history.is_empty() && !logits.is_empty() {
                    let vocab_cap = logits.len().min(n_vocab);
                    let mut seen = vec![false; vocab_cap];
                    for past_frame in &st.code_history {
                        if let Some(&past_code) = past_frame.get(j)
                            && past_code >= 0
                        {
                            let idx = past_code as usize;
                            if idx < vocab_cap && !seen[idx] {
                                seen[idx] = true;
                                if logits[idx] > 0.0 {
                                    logits[idx] /= LFM3A_REP_PENALTY;
                                } else {
                                    logits[idx] *= LFM3A_REP_PENALTY;
                                }
                            }
                        }
                    }
                }

                if j > 0 && (mask_id as usize) < logits.len() {
                    logits[mask_id as usize] = f32::NEG_INFINITY;
                }

                let max_logit = logits
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                let log_z = if max_logit.is_finite() {
                    let sum_exp: f64 = logits
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .map(|v| ((v - max_logit) as f64).exp())
                        .sum();
                    max_logit + (sum_exp.max(1e-30).ln() as f32)
                } else {
                    0.0
                };

                if logits.is_empty() {
                    cand_tokens[j] = 0;
                    confidences[j] = f32::NEG_INFINITY;
                } else if !t_step.is_finite() || t_step <= 0.0 || top_k == 1 {
                    let best = argmax(&logits);
                    cand_tokens[j] = best as i32;
                    confidences[j] = logits[best as usize] - log_z;
                } else {
                    let raw_logits_snap = logits.clone();
                    if top_k > 0 && top_k < logits.len() {
                        let mut order: Vec<usize> = (0..logits.len()).collect();
                        order.select_nth_unstable_by(top_k - 1, |&a, &b| {
                            logits[b].total_cmp(&logits[a])
                        });
                        for &idx in &order[top_k..] {
                            logits[idx] = f32::NEG_INFINITY;
                        }
                    }
                    let mut best_idx = 0usize;
                    let mut best_score = f32::NEG_INFINITY;
                    for (v_idx, &logit) in logits.iter().enumerate() {
                        if !logit.is_finite() {
                            continue;
                        }
                        let u = self.next_f32_01().clamp(1e-10, 1.0 - 1e-7);
                        let gumbel = -(-u.ln()).ln();
                        let noisy = logit + t_step * gumbel;
                        if noisy > best_score {
                            best_score = noisy;
                            best_idx = v_idx;
                        }
                    }
                    let u_conf = self.next_f32_01().clamp(1e-10, 1.0 - 1e-7);
                    let gumbel_conf = -(-u_conf.ln()).ln();
                    let anneal = 1.0 - ((step + 1) as f32 / unmasked_per_step.len() as f32);
                    cand_tokens[j] = best_idx as i32;
                    confidences[j] =
                        (raw_logits_snap[best_idx] - log_z) + 0.5 * t_step * gumbel_conf * anneal;
                }

                if step == 0 && j == 0 && cand_tokens[0] == mask_id {
                    return Ok(vec![mask_id; n_cb]);
                }
            }

            let mut remaining: Vec<usize> = (0..n_cb).filter(|&j| !unmasked[j]).collect();
            remaining.sort_by(|&a, &b| {
                confidences[b]
                    .total_cmp(&confidences[a])
                    .then_with(|| a.cmp(&b))
            });
            for &j in remaining.iter().take(k_unmask) {
                codes[j] = cand_tokens[j];
                unmasked[j] = true;
            }
            if unmasked.first().copied() == Some(true) && codes.first().copied() == Some(mask_id) {
                return Ok(vec![mask_id; n_cb]);
            }
        }

        if codes.first().copied() != Some(mask_id) {
            st.push_code_history(codes.clone());
        }
        Ok(codes)
    }
}

struct XdnaDetokLayer {
    is_conv: bool,
    operator_norm: Vec<f32>,
    attn_post_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    ffn_post_norm: Option<Vec<f32>>,
    w_ffn_w1: XdnaGemmWeight,
    w_ffn_w2: XdnaGemmWeight,
    w_ffn_w3: XdnaGemmWeight,
    conv_in_proj: Option<XdnaGemmWeight>,
    conv_out_proj: Option<XdnaGemmWeight>,
    conv_weight: Option<Vec<f32>>,
    w_q: Option<XdnaGemmWeight>,
    w_k: Option<XdnaGemmWeight>,
    w_v: Option<XdnaGemmWeight>,
    w_o: Option<XdnaGemmWeight>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    w_attn_gate: Option<(XdnaGemmWeight, usize)>,
}

struct XdnaDetokState {
    conv_bufs: Vec<Vec<f32>>,
    attn_kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    n_past: usize,
}

impl XdnaDetokState {
    fn new(cfg: &DetokenizerConfig) -> Self {
        let mut conv_bufs = Vec::with_capacity(cfg.layer_is_conv.len());
        let mut attn_kv = Vec::with_capacity(cfg.layer_is_conv.len());
        let kv_size = cfg.swa_window_size * cfg.n_head_kv * cfg.n_embd_head;
        for &is_conv in &cfg.layer_is_conv {
            if is_conv {
                conv_bufs.push(vec![0.0f32; cfg.d_conv * cfg.n_embd]);
                attn_kv.push(None);
            } else {
                conv_bufs.push(Vec::new());
                attn_kv.push(Some((vec![0.0f32; kv_size], vec![0.0f32; kv_size])));
            }
        }
        Self {
            conv_bufs,
            attn_kv,
            n_past: 0,
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.conv_bufs {
            buf.fill(0.0);
        }
        for (k, v) in self.attn_kv.iter_mut().flatten() {
            k.fill(0.0);
            v.fill(0.0);
        }
        self.n_past = 0;
    }
}

struct XdnaDetokenizer {
    config: DetokenizerConfig,
    lfm3_gen: bool,
    output_norm: Vec<f32>,
    lin_rows: usize,
    lin_cols: usize,
    w_lin: XdnaGemmWeight,
    lin_b: Vec<f32>,
    layers: Vec<XdnaDetokLayer>,
    state: Mutex<XdnaDetokState>,
}

impl XdnaDetokenizer {
    fn new(runner: &XdnaGemmRunner, weights: &DetokenizerWeights) -> Result<Self> {
        let lfm3_gen = weights.is_lfm3a_gen();
        anyhow::ensure!(
            weights.config.n_embd > 0,
            "XDNA detokenizer config requires n_embd > 0"
        );
        anyhow::ensure!(
            weights.config.layer_is_conv.len() == weights.layers.len(),
            "XDNA detokenizer layer_is_conv len ({}) != layers len ({})",
            weights.config.layer_is_conv.len(),
            weights.layers.len()
        );
        if weights.config.layer_is_conv.iter().any(|&c| !c) {
            anyhow::ensure!(
                weights.config.swa_window_size > 0,
                "XDNA detokenizer with attention layers requires swa_window_size > 0"
            );
            anyhow::ensure!(
                weights.config.n_head > 0
                    && weights.config.n_head_kv > 0
                    && weights.config.n_embd_head > 0,
                "XDNA detokenizer with attention layers requires positive head dimensions"
            );
        }
        let mut layers = Vec::with_capacity(weights.layers.len());
        for (il, lw) in weights.layers.iter().enumerate() {
            let is_conv = weights.config.layer_is_conv[il];
            let (conv_in_proj, conv_out_proj, w_q, w_k, w_v, w_o, w_attn_gate) = if is_conv {
                (
                    lw.conv_in_proj
                        .as_ref()
                        .map(|w| runner.upload_mmap_weight(w))
                        .transpose()?,
                    lw.conv_out_proj
                        .as_ref()
                        .map(|w| runner.upload_mmap_weight(w))
                        .transpose()?,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            } else {
                (
                    None,
                    None,
                    lw.wq
                        .as_ref()
                        .map(|w| runner.upload_mmap_weight(w))
                        .transpose()?,
                    lw.wk
                        .as_ref()
                        .map(|w| runner.upload_mmap_weight(w))
                        .transpose()?,
                    lw.wv
                        .as_ref()
                        .map(|w| runner.upload_mmap_weight(w))
                        .transpose()?,
                    lw.wo
                        .as_ref()
                        .map(|w| runner.upload_mmap_weight(w))
                        .transpose()?,
                    lw.attn_gate
                        .as_ref()
                        .map(|w| Ok::<_, anyhow::Error>((runner.upload_mmap_weight(w)?, w.rows)))
                        .transpose()?,
                )
            };
            layers.push(XdnaDetokLayer {
                is_conv,
                operator_norm: lw.operator_norm.clone(),
                attn_post_norm: lw.attn_post_norm.clone(),
                ffn_norm: lw.ffn_norm.clone(),
                ffn_post_norm: lw.ffn_post_norm.clone(),
                w_ffn_w1: runner.upload_mmap_weight(&lw.ffn_w1)?,
                w_ffn_w2: runner.upload_mmap_weight(&lw.ffn_w2)?,
                w_ffn_w3: runner.upload_mmap_weight(&lw.ffn_w3)?,
                conv_in_proj,
                conv_out_proj,
                conv_weight: lw.conv_weight.clone(),
                w_q,
                w_k,
                w_v,
                w_o,
                q_norm: lw.q_norm.clone(),
                k_norm: lw.k_norm.clone(),
                w_attn_gate,
            });
        }
        let w_lin = runner.upload_mmap_weight(&weights.lin_w)?;
        let state = Mutex::new(XdnaDetokState::new(&weights.config));
        Ok(Self {
            config: weights.config.clone(),
            lfm3_gen,
            output_norm: weights.output_norm.clone(),
            lin_rows: weights.lin_w.rows,
            lin_cols: weights.lin_w.cols,
            w_lin,
            lin_b: weights.lin_b.clone(),
            layers,
            state,
        })
    }

    fn reset(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).reset();
    }

    fn detokenize_to_spectrum(
        &self,
        runner: &XdnaGemmRunner,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;
        anyhow::ensure!(n_embd > 0, "XDNA detokenizer requires n_embd > 0");

        let embedding = detok_embed_codes(cpu_weights, codes);
        let single_frame_bins = (cfg.n_fft / 2 + 1) * 2;
        let tokens = if self.lin_rows == 6 * single_frame_bins {
            embedding
        } else {
            upsample(&embedding, n_embd, 6)
        };
        anyhow::ensure!(
            !tokens.is_empty() && tokens.len().is_multiple_of(n_embd),
            "XDNA detokenizer embedded token buffer len ({}) is invalid for n_embd={n_embd}",
            tokens.len()
        );
        let n_tokens = tokens.len() / n_embd;
        let mut hidden = tokens;
        let mut norm_mat = vec![0.0f32; n_tokens * n_embd];
        let mut conv_y_mat = vec![0.0f32; n_tokens * n_embd];
        let mut op_out_mat = vec![0.0f32; n_tokens * n_embd];

        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let n_past_base = st.n_past;

        for (il, lw) in self.layers.iter().enumerate() {
            norm_mat.copy_from_slice(&hidden);
            for t in 0..n_tokens {
                rmsnorm(
                    &mut norm_mat[t * n_embd..(t + 1) * n_embd],
                    &lw.operator_norm,
                    cfg.rms_norm_eps,
                );
            }

            if lw.is_conv {
                if let (Some(in_proj), Some(conv_w), Some(out_proj)) =
                    (&lw.conv_in_proj, &lw.conv_weight, &lw.conv_out_proj)
                {
                    let bcx_mat =
                        runner.matmul_f32(&norm_mat, in_proj, n_tokens, 3 * n_embd, n_embd)?;
                    let conv_buf = &mut st.conv_bufs[il];
                    let d_conv = cfg.d_conv;
                    let kernel_size = d_conv + 1;
                    let mut bx = vec![0.0f32; n_embd];
                    for t in 0..n_tokens {
                        let bcx = &bcx_mat[t * 3 * n_embd..(t + 1) * 3 * n_embd];
                        let b = &bcx[..n_embd];
                        let c = &bcx[n_embd..2 * n_embd];
                        let x = &bcx[2 * n_embd..];
                        for ch in 0..n_embd {
                            bx[ch] = b[ch] * x[ch];
                        }
                        let y_row = &mut conv_y_mat[t * n_embd..(t + 1) * n_embd];
                        for ch in 0..n_embd {
                            let mut sum = 0.0f32;
                            for k in 0..d_conv {
                                sum += conv_buf[k * n_embd + ch] * conv_w[ch * kernel_size + k];
                            }
                            sum += bx[ch] * conv_w[ch * kernel_size + d_conv];
                            y_row[ch] = c[ch] * sum;
                        }
                        if d_conv > 1 {
                            for k in 0..d_conv - 1 {
                                let src = (k + 1) * n_embd;
                                let dst = k * n_embd;
                                conv_buf.copy_within(src..src + n_embd, dst);
                            }
                        }
                        if d_conv > 0 {
                            conv_buf[(d_conv - 1) * n_embd..d_conv * n_embd].copy_from_slice(&bx);
                        }
                    }
                    op_out_mat =
                        runner.matmul_f32(&conv_y_mat, out_proj, n_tokens, n_embd, n_embd)?;
                } else {
                    op_out_mat.copy_from_slice(&norm_mat);
                }
            } else if let (
                Some(wq),
                Some(wk),
                Some(wv),
                Some(wo),
                Some(q_norm),
                Some(k_norm),
                Some(kv),
            ) = (
                &lw.w_q,
                &lw.w_k,
                &lw.w_v,
                &lw.w_o,
                &lw.q_norm,
                &lw.k_norm,
                st.attn_kv.get_mut(il).and_then(|o| o.as_mut()),
            ) {
                let n_head = cfg.n_head;
                let n_kv = cfg.n_head_kv;
                let hd = cfg.n_embd_head;
                anyhow::ensure!(
                    cfg.swa_window_size > 0 && n_head > 0 && n_kv > 0 && hd > 0,
                    "XDNA detokenizer attention layer requires positive swa_window_size and head dimensions"
                );
                let q_dim = n_head * hd;
                let kv_stride = n_kv * hd;
                let group_size = (n_head / n_kv.max(1)).max(1);
                let scale = 1.0f32 / (hd as f32).sqrt();

                let mut all_q = runner.matmul_f32(&norm_mat, wq, n_tokens, q_dim, n_embd)?;
                let mut all_k = runner.matmul_f32(&norm_mat, wk, n_tokens, kv_stride, n_embd)?;
                let all_v = runner.matmul_f32(&norm_mat, wv, n_tokens, kv_stride, n_embd)?;

                let (gate_rows, mut all_gate) = if let Some((gw, gr)) = &lw.w_attn_gate {
                    (
                        *gr,
                        runner.matmul_f32(&norm_mat, gw, n_tokens, *gr, n_embd)?,
                    )
                } else {
                    (0, Vec::new())
                };

                for t in 0..n_tokens {
                    let pos = n_past_base + t;
                    let q = &mut all_q[t * q_dim..(t + 1) * q_dim];
                    let k = &mut all_k[t * kv_stride..(t + 1) * kv_stride];
                    let v = &all_v[t * kv_stride..(t + 1) * kv_stride];
                    for h in 0..n_head {
                        rmsnorm(&mut q[h * hd..(h + 1) * hd], q_norm, cfg.rms_norm_eps);
                        apply_rope_neox(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
                    }
                    for h in 0..n_kv {
                        rmsnorm(&mut k[h * hd..(h + 1) * hd], k_norm, cfg.rms_norm_eps);
                        apply_rope_neox(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
                    }
                    if !self.lfm3_gen {
                        let wp = pos % cfg.swa_window_size;
                        kv.0[wp * kv_stride..(wp + 1) * kv_stride].copy_from_slice(k);
                        kv.1[wp * kv_stride..(wp + 1) * kv_stride].copy_from_slice(v);
                    }
                }

                let mut attn_out_mat = vec![0.0f32; n_tokens * q_dim];
                for t in 0..n_tokens {
                    if self.lfm3_gen {
                        let wp = (n_past_base + t) % cfg.swa_window_size;
                        kv.0[wp * kv_stride..(wp + 1) * kv_stride]
                            .copy_from_slice(&all_k[t * kv_stride..(t + 1) * kv_stride]);
                        kv.1[wp * kv_stride..(wp + 1) * kv_stride]
                            .copy_from_slice(&all_v[t * kv_stride..(t + 1) * kv_stride]);
                    }
                    let q = &all_q[t * q_dim..(t + 1) * q_dim];
                    let eff_end = if self.lfm3_gen {
                        n_past_base + t + 1
                    } else {
                        n_past_base + n_tokens
                    };
                    let kv_start = eff_end.saturating_sub(cfg.swa_window_size);
                    let kv_len = eff_end - kv_start;
                    let attn_out = &mut attn_out_mat[t * q_dim..(t + 1) * q_dim];
                    let mut scores = vec![0.0f32; kv_len];
                    for h in 0..n_head {
                        let kv_h = h / group_size;
                        let qh = &q[h * hd..(h + 1) * hd];
                        for (tt, s) in scores.iter_mut().enumerate() {
                            let cp = (kv_start + tt) % cfg.swa_window_size;
                            let ko = cp * kv_stride + kv_h * hd;
                            *s = dot_f32(qh, &kv.0[ko..ko + hd]) * scale;
                        }
                        softmax_inplace(&mut scores);
                        let out = &mut attn_out[h * hd..(h + 1) * hd];
                        for (tt, &s) in scores.iter().enumerate() {
                            let cp = (kv_start + tt) % cfg.swa_window_size;
                            let vo = cp * kv_stride + kv_h * hd;
                            for (dst, &val) in out.iter_mut().zip(&kv.1[vo..vo + hd]) {
                                *dst += s * val;
                            }
                        }
                    }
                    if gate_rows > 0 {
                        let g_row = &mut all_gate[t * gate_rows..(t + 1) * gate_rows];
                        if gate_rows == n_head {
                            for h in 0..n_head {
                                let s = 1.0f32 / (1.0f32 + (-g_row[h]).exp());
                                for v in &mut attn_out[h * hd..(h + 1) * hd] {
                                    *v *= s;
                                }
                            }
                        } else {
                            for (av, &g) in attn_out.iter_mut().zip(g_row.iter()) {
                                *av *= 1.0f32 / (1.0f32 + (-g).exp());
                            }
                        }
                    }
                }
                op_out_mat = runner.matmul_f32(&attn_out_mat, wo, n_tokens, n_embd, q_dim)?;
            } else {
                continue;
            }

            if let Some(pn) = &lw.attn_post_norm {
                for t in 0..n_tokens {
                    rmsnorm(
                        &mut op_out_mat[t * n_embd..(t + 1) * n_embd],
                        pn,
                        cfg.rms_norm_eps,
                    );
                }
            }
            add_inplace(&mut hidden, &op_out_mat);

            // Batched SwiGLU FFN on XDNA.
            norm_mat.copy_from_slice(&hidden);
            for t in 0..n_tokens {
                rmsnorm(
                    &mut norm_mat[t * n_embd..(t + 1) * n_embd],
                    &lw.ffn_norm,
                    cfg.rms_norm_eps,
                );
            }
            let mut gate_mat =
                runner.matmul_f32(&norm_mat, &lw.w_ffn_w1, n_tokens, cfg.ffn_dim, n_embd)?;
            let up_mat =
                runner.matmul_f32(&norm_mat, &lw.w_ffn_w3, n_tokens, cfg.ffn_dim, n_embd)?;
            silu_mul_inplace(&mut gate_mat, &up_mat);
            let mut down_mat =
                runner.matmul_f32(&gate_mat, &lw.w_ffn_w2, n_tokens, n_embd, cfg.ffn_dim)?;
            if let Some(pn) = &lw.ffn_post_norm {
                for t in 0..n_tokens {
                    rmsnorm(
                        &mut down_mat[t * n_embd..(t + 1) * n_embd],
                        pn,
                        cfg.rms_norm_eps,
                    );
                }
            }
            add_inplace(&mut hidden, &down_mat);
        }

        st.n_past += n_tokens;

        for t in 0..n_tokens {
            rmsnorm(
                &mut hidden[t * n_embd..(t + 1) * n_embd],
                &self.output_norm,
                cfg.rms_norm_eps,
            );
        }

        let mut spectrum =
            runner.matmul_f32(&hidden, &self.w_lin, n_tokens, self.lin_rows, self.lin_cols)?;
        for t in 0..n_tokens {
            add_inplace(
                &mut spectrum[t * self.lin_rows..(t + 1) * self.lin_rows],
                &self.lin_b,
            );
        }
        Ok(spectrum)
    }
}

// ── Public XDNA AudioAccelerator ─────────────────────────────────────────────

/// AMD XDNA2 NPU audio decoder & detokenizer implementing [`AudioAccelerator`].
pub struct XdnaAudioDecoder {
    runner: XdnaGemmRunner,
    depthformer: Option<XdnaDepthformer>,
    detokenizer: Option<XdnaDetokenizer>,
    last_error: Mutex<Option<CeraError>>,
    session_active: AtomicBool,
}

impl XdnaAudioDecoder {
    /// Construct an XDNA audio decoder from parsed decoder and/or detokenizer weights.
    pub fn from_parts(
        runner: XdnaGemmRunner,
        depthformer_weights: Option<&AudioDecoderWeights>,
        detok_weights: Option<&DetokenizerWeights>,
    ) -> Result<Self> {
        anyhow::ensure!(
            depthformer_weights.is_some() || detok_weights.is_some(),
            "XDNA audio decoder requires depthformer or detokenizer weights"
        );
        let _ = runner.take_err();
        let depthformer = depthformer_weights
            .map(|w| XdnaDepthformer::new(&runner, w))
            .transpose()?;
        let detokenizer = detok_weights
            .map(|w| XdnaDetokenizer::new(&runner, w))
            .transpose()?;
        if let Some(e) = runner.take_err() {
            return Err(e);
        }
        Ok(Self {
            runner,
            depthformer,
            detokenizer,
            last_error: Mutex::new(None),
            session_active: AtomicBool::new(false),
        })
    }

    fn record_error(&self, err: CeraError) {
        record_first_fault(&self.last_error, err);
    }

    /// Whether the XDNA detokenizer is loaded.
    pub fn has_detokenizer(&self) -> bool {
        self.detokenizer.is_some()
    }
}

impl AudioAccelerator for XdnaAudioDecoder {
    fn sample_audio_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8] {
        let Some(ref df) = self.depthformer else {
            return [0; 8];
        };
        if df.lfm3a_gen {
            match df.sample_frame_lfm3a(&self.runner, embedding, temperature, top_k) {
                Ok(v) => {
                    let mut out = [0i32; 8];
                    for (dst, &src) in out.iter_mut().zip(v.iter()) {
                        *dst = src;
                    }
                    return out;
                }
                Err(e) => {
                    self.record_error(CeraError::Backend(format!("XDNA depthformer: {e:#}")));
                    return [0; 8];
                }
            }
        }
        match df.sample_frame_lfm2a(&self.runner, embedding, temperature, top_k) {
            Ok(codes) => codes,
            Err(e) => {
                self.record_error(CeraError::Backend(format!("XDNA depthformer: {e:#}")));
                [0; 8]
            }
        }
    }

    fn supports_lfm3a_gen_depthformer(&self) -> bool {
        self.depthformer.as_ref().is_some_and(|df| df.lfm3a_gen)
    }

    fn sample_audio_frame_lfm3a(
        &self,
        embedding: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Option<Vec<i32>> {
        let df = self.depthformer.as_ref()?;
        if !df.lfm3a_gen {
            return None;
        }
        match df.sample_frame_lfm3a(&self.runner, embedding, temperature, top_k) {
            Ok(codes) => Some(codes),
            Err(e) => {
                self.record_error(CeraError::Backend(format!("XDNA lfm3a depthformer: {e:#}")));
                None
            }
        }
    }

    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
        let Some(ref detok) = self.detokenizer else {
            self.record_error(CeraError::Backend(
                "XDNA audio detokenizer weights were not loaded".to_string(),
            ));
            return Vec::new();
        };
        match detok.detokenize_to_spectrum(&self.runner, cpu_weights, codes) {
            Ok(spec) => spec,
            Err(e) => {
                self.record_error(CeraError::Backend(format!("XDNA detokenizer: {e:#}")));
                Vec::new()
            }
        }
    }

    fn reset_depthformer(&self) {
        if let Some(ref df) = self.depthformer {
            df.reset();
        }
    }

    fn seed_rng(&self, seed: u64) {
        if let Some(ref df) = self.depthformer {
            df.seed_rng(seed);
        }
    }

    fn reset_detokenizer(&self) {
        if let Some(ref detok) = self.detokenizer {
            detok.reset();
        }
    }

    fn supports_depthformer(&self) -> bool {
        self.depthformer.is_some()
    }

    fn supports_lfm3a_gen_detokenizer(&self) -> bool {
        self.detokenizer.is_some()
    }

    fn depthformer_default_on(&self) -> bool {
        true
    }

    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        istft_to_pcm(spectrum, n_fft, hop_length)
    }

    fn take_audio_error(&self) -> Option<CeraError> {
        let own_err = take_fault(&self.last_error);
        let runner_err = self
            .runner
            .take_err()
            .map(|e| CeraError::Backend(format!("XDNA runner: {e:#}")));
        own_err.or(runner_err)
    }

    fn try_acquire_session(&self) -> bool {
        self.session_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    fn release_session(&self) {
        self.session_active.store(false, Ordering::Release);
    }
}

/// Probe and build an [`XdnaAudioDecoder`] for a vocoder / audio mmproj GGUF.
pub fn try_xdna_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioAccelerator>> {
    let df_weights = AudioDecoderWeights::from_gguf(gguf).ok();
    let detok_weights = DetokenizerWeights::from_gguf(gguf).ok();
    if df_weights.is_none() && detok_weights.is_none() {
        return None;
    }
    let runner = match XdnaGemmRunner::open_shared() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("audio decoder: XDNA NPU unavailable ({e:#}); falling back");
            return None;
        }
    };
    match XdnaAudioDecoder::from_parts(runner, df_weights.as_ref(), detok_weights.as_ref()) {
        Ok(dec) => {
            tracing::info!("audio decoder: using AMD XDNA2 NPU backend");
            Some(Arc::new(dec))
        }
        Err(e) => {
            tracing::warn!("audio decoder: XDNA weight upload failed, falling back: {e:#}");
            None
        }
    }
}

// ── Unit tests (fake XDNA shim) ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::xdna::sys::fake::{self, Config as FakeConfig};
    use crate::gguf::GgufBuilder;
    use crate::model::audio_decoder::{
        AudioDecoderWeights, CodebookWeights, DecoderConfig, DepthformerConfig,
        DepthformerLayerWeights, DepthformerState, DetokLayerWeights, DetokenizerConfig,
        DetokenizerState, DetokenizerWeights, Lfm3FsqWeights, build_audio_accelerator,
        build_default_fsq_codebook, detokenize_to_spectrum, sample_audio_frame, sample_frame,
    };

    fn det_vec(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i.wrapping_mul(37).wrapping_add(seed.wrapping_mul(17))) % 251) as f32;
                ((x - 125.0) / 125.0) * scale
            })
            .collect()
    }

    fn f32_w(rows: usize, cols: usize, seed: usize, scale: f32) -> MmapWeight {
        MmapWeight::from_owned_f32(det_vec(rows * cols, seed, scale), rows, cols)
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (xf, yf) = (x as f64, y as f64);
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn synth_lfm2a_weights() -> (AudioDecoderWeights, DetokenizerWeights) {
        let n_embd = 64;
        let llm_dim = 64;
        let ffn_dim = 128;
        let n_head = 4;
        let n_kv = 2;
        let hd = 16;
        let n_vocab = 32;
        let n_cb = 8;
        let n_fft = 64;
        let spec_bins = (n_fft / 2 + 1) * 2;

        let df_cfg = DepthformerConfig {
            n_layer: 2,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            n_embd_head: hd,
            ffn_dim,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            max_seq_len: n_cb,
        };
        let dec_cfg = DecoderConfig {
            n_codebook: n_cb,
            n_vocab,
            n_embd: llm_dim,
            rms_norm_eps: 1e-5,
        };
        let qkv_rows = (n_head + 2 * n_kv) * hd;
        let df_layers = (0..2)
            .map(|il| DepthformerLayerWeights {
                operator_norm: vec![1.0; n_embd],
                wqkv: f32_w(qkv_rows, n_embd, 10 + il, 0.2),
                q_norm: vec![1.0; hd],
                k_norm: vec![1.0; hd],
                wo: f32_w(n_embd, n_embd, 20 + il, 0.2),
                ffn_norm: vec![1.0; n_embd],
                w1: f32_w(ffn_dim, n_embd, 30 + il, 0.2),
                w2: f32_w(n_embd, ffn_dim, 40 + il, 0.2),
                w3: f32_w(ffn_dim, n_embd, 50 + il, 0.2),
            })
            .collect();
        let depth_embeddings = (0..n_cb)
            .map(|j| CodebookWeights {
                embedding: f32_w(n_vocab, n_embd, 60 + j, 0.15),
                norm: vec![1.0; n_embd],
                to_logits: f32_w(n_vocab, n_embd, 70 + j, 0.25),
            })
            .collect();
        let dec_w = AudioDecoderWeights {
            lfm3a_gen: false,
            depthformer_config: df_cfg,
            decoder_config: dec_cfg,
            depthformer_layers: df_layers,
            depth_linear_w: f32_w(n_cb * n_embd, llm_dim, 80, 0.2),
            depth_linear_b: det_vec(n_cb * n_embd, 81, 0.02),
            depth_embeddings,
            audio_embedding: CodebookWeights {
                embedding: f32_w(n_cb * n_vocab, llm_dim, 90, 0.1),
                norm: Vec::new(),
                to_logits: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            },
            interleave: crate::audio_engine::InterleaveCadence { text: 1, audio: 2 },
        };

        let detok_cfg = DetokenizerConfig {
            n_layer: 2,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            n_embd_head: hd,
            ffn_dim,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10000.0,
            swa_window_size: 16,
            n_codes: n_cb,
            n_fft,
            hop_length: 16,
            sample_rate: 24000,
            layer_is_conv: vec![true, false],
        };
        let detok_w = DetokenizerWeights {
            config: detok_cfg,
            output_norm: vec![1.0; n_embd],
            emb_weight: f32_w(n_cb * n_vocab, n_embd, 100, 0.2),
            lin_w: f32_w(spec_bins, n_embd, 101, 0.2),
            lin_b: det_vec(spec_bins, 102, 0.02),
            layers: vec![
                DetokLayerWeights {
                    operator_norm: vec![1.0; n_embd],
                    ffn_norm: vec![1.0; n_embd],
                    attn_post_norm: None,
                    ffn_post_norm: None,
                    ffn_w1: f32_w(ffn_dim, n_embd, 110, 0.2),
                    ffn_w2: f32_w(n_embd, ffn_dim, 111, 0.2),
                    ffn_w3: f32_w(ffn_dim, n_embd, 112, 0.2),
                    conv_in_proj: Some(f32_w(3 * n_embd, n_embd, 113, 0.2)),
                    conv_out_proj: Some(f32_w(n_embd, n_embd, 114, 0.2)),
                    conv_weight: Some(det_vec(3 * n_embd, 115, 0.15)),
                    wq: None,
                    wk: None,
                    wv: None,
                    wo: None,
                    q_norm: None,
                    k_norm: None,
                    attn_gate: None,
                },
                DetokLayerWeights {
                    operator_norm: vec![1.0; n_embd],
                    ffn_norm: vec![1.0; n_embd],
                    attn_post_norm: None,
                    ffn_post_norm: None,
                    ffn_w1: f32_w(ffn_dim, n_embd, 120, 0.2),
                    ffn_w2: f32_w(n_embd, ffn_dim, 121, 0.2),
                    ffn_w3: f32_w(ffn_dim, n_embd, 122, 0.2),
                    conv_in_proj: None,
                    conv_out_proj: None,
                    conv_weight: None,
                    wq: Some(f32_w(n_head * hd, n_embd, 123, 0.2)),
                    wk: Some(f32_w(n_kv * hd, n_embd, 124, 0.2)),
                    wv: Some(f32_w(n_kv * hd, n_embd, 125, 0.2)),
                    wo: Some(f32_w(n_embd, n_head * hd, 126, 0.2)),
                    q_norm: Some(vec![1.0; hd]),
                    k_norm: Some(vec![1.0; hd]),
                    attn_gate: None,
                },
            ],
            lfm3_fsq: None,
        };
        (dec_w, detok_w)
    }

    fn synth_lfm3a_weights() -> (AudioDecoderWeights, DetokenizerWeights) {
        let n_cb = 12;
        let n_vocab = 257;
        let n_embd = 64;
        let llm_dim = 64;
        let ffn_dim = 128;
        let n_head = 4;
        let n_kv = 2;
        let hd = 16;
        let n_fft = 64;
        let spec_bins = (n_fft / 2 + 1) * 2;

        let df_cfg = DepthformerConfig {
            n_layer: 2,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            n_embd_head: hd,
            ffn_dim,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            max_seq_len: n_cb,
        };
        let dec_cfg = DecoderConfig {
            n_codebook: n_cb,
            n_vocab,
            n_embd: llm_dim,
            rms_norm_eps: 1e-5,
        };

        let shortconv = DepthformerLayerWeights {
            operator_norm: vec![1.0; n_embd],
            wqkv: f32_w(3 * n_embd, n_embd, 200, 0.15),
            q_norm: det_vec(3 * n_embd, 201, 0.1),
            k_norm: Vec::new(),
            wo: f32_w(n_embd, n_embd, 202, 0.15),
            ffn_norm: vec![1.0; n_embd],
            w1: f32_w(ffn_dim, n_embd, 203, 0.15),
            w2: f32_w(n_embd, ffn_dim, 204, 0.15),
            w3: f32_w(ffn_dim, n_embd, 205, 0.15),
        };
        let q_dim = n_head * hd;
        let kv_dim = n_kv * hd;
        let attn = DepthformerLayerWeights {
            operator_norm: vec![1.0; n_embd],
            wqkv: f32_w(q_dim + 2 * kv_dim + n_head, n_embd, 210, 0.15),
            q_norm: vec![1.0; hd],
            k_norm: vec![1.0; hd],
            wo: f32_w(n_embd, q_dim, 211, 0.15),
            ffn_norm: vec![1.0; n_embd],
            w1: f32_w(ffn_dim, n_embd, 212, 0.15),
            w2: f32_w(n_embd, ffn_dim, 213, 0.15),
            w3: f32_w(ffn_dim, n_embd, 214, 0.15),
        };

        let mut depth_embeddings = Vec::with_capacity(n_cb);
        for j in 0..n_cb {
            let mut logits_w = det_vec(n_vocab * n_embd, 220 + j, 0.01);
            let win_code = j + 5;
            let alt_code = j + 20;
            // Assign non-orthogonal feature weights across dims so logits have clear
            // separation above BF16 rounding (even after RMSNorm zero-centering).
            for d in 0..n_embd {
                let pat = if d % 2 == 0 { 1.2 } else { -0.4 };
                logits_w[win_code * n_embd + d] = pat;
                logits_w[alt_code * n_embd + d] = pat * 0.85;
            }
            depth_embeddings.push(CodebookWeights {
                embedding: f32_w(n_vocab, n_embd, 240 + j, 0.08),
                norm: (0..n_embd)
                    .map(|d| if d % 2 == 0 { 1.5 } else { 0.5 })
                    .collect(),
                to_logits: MmapWeight::from_owned_f32(logits_w, n_vocab, n_embd),
            });
        }
        let dec_w = AudioDecoderWeights {
            lfm3a_gen: true,
            depthformer_config: df_cfg,
            decoder_config: dec_cfg,
            depthformer_layers: vec![shortconv, attn],
            depth_linear_w: f32_w(n_embd, llm_dim, 260, 0.12),
            depth_linear_b: (0..n_embd)
                .map(|d| if d % 2 == 0 { 0.6 } else { -0.2 })
                .collect(),
            depth_embeddings,
            audio_embedding: CodebookWeights {
                embedding: f32_w(n_cb * n_vocab, llm_dim, 262, 0.05),
                norm: vec![1.0; n_embd],
                to_logits: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            },
            interleave: crate::audio_engine::InterleaveCadence { text: 6, audio: 12 },
        };

        let fsq_proj_dim = 64;
        let lfm3_fsq = Lfm3FsqWeights {
            fsq_codebook: build_default_fsq_codebook(),
            fsq_proj_dim,
            quant_first_out_w: det_vec(fsq_proj_dim * 16, 270, 0.05),
            quant_first_out_b: det_vec(fsq_proj_dim, 271, 0.01),
            quant_rest_out_w: det_vec(fsq_proj_dim * 16, 272, 0.05),
            quant_rest_out_b: det_vec(fsq_proj_dim, 273, 0.01),
            tfm_in_proj_w: f32_w(n_embd, fsq_proj_dim, 274, 0.08),
            tfm_in_proj_b: det_vec(n_embd, 275, 0.01),
        };
        let detok_cfg = DetokenizerConfig {
            n_layer: 2,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            n_embd_head: hd,
            ffn_dim,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 16,
            n_codes: n_cb,
            n_fft,
            hop_length: 16,
            sample_rate: 24000,
            layer_is_conv: vec![true, false],
        };
        let detok_w = DetokenizerWeights {
            config: detok_cfg,
            output_norm: vec![1.0; n_embd],
            emb_weight: MmapWeight::from_owned_f32(Vec::new(), 0, 0),
            lin_w: f32_w(6 * spec_bins, n_embd, 280, 0.15),
            lin_b: det_vec(6 * spec_bins, 281, 0.02),
            layers: vec![
                DetokLayerWeights {
                    operator_norm: vec![1.0; n_embd],
                    ffn_norm: vec![1.0; n_embd],
                    attn_post_norm: Some(vec![1.0; n_embd]),
                    ffn_post_norm: Some(vec![1.0; n_embd]),
                    ffn_w1: f32_w(ffn_dim, n_embd, 282, 0.15),
                    ffn_w2: f32_w(n_embd, ffn_dim, 283, 0.15),
                    ffn_w3: f32_w(ffn_dim, n_embd, 284, 0.15),
                    conv_in_proj: Some(f32_w(3 * n_embd, n_embd, 285, 0.15)),
                    conv_out_proj: Some(f32_w(n_embd, n_embd, 286, 0.15)),
                    conv_weight: Some(det_vec(3 * n_embd, 287, 0.1)),
                    wq: None,
                    wk: None,
                    wv: None,
                    wo: None,
                    q_norm: None,
                    k_norm: None,
                    attn_gate: None,
                },
                DetokLayerWeights {
                    operator_norm: vec![1.0; n_embd],
                    ffn_norm: vec![1.0; n_embd],
                    attn_post_norm: Some(vec![1.0; n_embd]),
                    ffn_post_norm: Some(vec![1.0; n_embd]),
                    ffn_w1: f32_w(ffn_dim, n_embd, 290, 0.15),
                    ffn_w2: f32_w(n_embd, ffn_dim, 291, 0.15),
                    ffn_w3: f32_w(ffn_dim, n_embd, 292, 0.15),
                    conv_in_proj: None,
                    conv_out_proj: None,
                    conv_weight: None,
                    wq: Some(f32_w(q_dim, n_embd, 293, 0.15)),
                    wk: Some(f32_w(kv_dim, n_embd, 294, 0.15)),
                    wv: Some(f32_w(kv_dim, n_embd, 295, 0.15)),
                    wo: Some(f32_w(n_embd, q_dim, 296, 0.15)),
                    q_norm: Some(vec![1.0; hd]),
                    k_norm: Some(vec![1.0; hd]),
                    attn_gate: Some(f32_w(n_head, n_embd, 297, 0.15)),
                },
            ],
            lfm3_fsq: Some(lfm3_fsq),
        };
        (dec_w, detok_w)
    }

    #[test]
    fn fake_shim_lfm2a_depthformer_and_detokenizer_lifecycle_and_shapes() {
        fake::with_shim(FakeConfig::default(), || {
            let (dec_w, detok_w) = synth_lfm2a_weights();
            let runner = XdnaGemmRunner::open_shared().expect("open_shared");
            let xdna = XdnaAudioDecoder::from_parts(runner, Some(&dec_w), Some(&detok_w))
                .expect("XdnaAudioDecoder lfm2a");
            assert!(xdna.supports_depthformer());
            assert!(xdna.depthformer_default_on());
            assert!(!xdna.supports_lfm3a_gen_depthformer());
            assert!(xdna.supports_lfm3a_gen_detokenizer());

            let embed = det_vec(dec_w.decoder_config.n_embd, 7, 0.5);
            let mut df_state = DepthformerState::new(&dec_w.depthformer_config);
            let cpu_codes = sample_audio_frame(&dec_w, &mut df_state, &embed, 0.0, 1);
            let npu_codes = xdna.sample_audio_frame(&embed, 0.0, 1);
            assert!(xdna.take_audio_error().is_none());
            assert_eq!(npu_codes.len(), 8);

            let mut dt_state = DetokenizerState::new(&detok_w.config);
            for frame_idx in 0..2 {
                let mut f_codes = cpu_codes;
                f_codes[0] = (f_codes[0] + frame_idx) % 16;
                let cpu_spec = detokenize_to_spectrum(&detok_w, &dec_w, &mut dt_state, &f_codes);
                let npu_spec = xdna.detokenize_to_spectrum(&detok_w, &f_codes);
                assert!(xdna.take_audio_error().is_none());
                assert_eq!(npu_spec.len(), cpu_spec.len());
                assert!(npu_spec.iter().all(|v| v.is_finite()));
            }

            drop(xdna);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_lfm3a_maskgit_depthformer_and_fsq_detok_lifecycle_and_shapes() {
        fake::with_shim(FakeConfig::default(), || {
            let (dec_w, detok_w) = synth_lfm3a_weights();
            let runner = XdnaGemmRunner::open_shared().expect("open_shared");
            let xdna = XdnaAudioDecoder::from_parts(runner, Some(&dec_w), Some(&detok_w))
                .expect("XdnaAudioDecoder lfm3a");
            assert!(xdna.supports_depthformer());
            assert!(xdna.supports_lfm3a_gen_depthformer());
            assert!(xdna.supports_lfm3a_gen_detokenizer());

            let embed = det_vec(dec_w.decoder_config.n_embd, 19, 0.4);
            let mut df_state = DepthformerState::new(&dec_w.depthformer_config);
            for _ in 0..2 {
                let npu_codes = xdna
                    .sample_audio_frame_lfm3a(&embed, 0.0, 1)
                    .expect("lfm3a codes");
                assert!(xdna.take_audio_error().is_none());
                assert_eq!(npu_codes.len(), 12);
            }
            xdna.reset_depthformer();
            let cpu_codes = sample_frame(&dec_w, &mut df_state, &embed, 0.0, 1);

            let mut dt_state = DetokenizerState::new(&detok_w.config);
            for frame_idx in 0..2 {
                let mut f_codes = cpu_codes.clone();
                f_codes[0] = (f_codes[0] + frame_idx) % 256;
                let cpu_spec = detokenize_to_spectrum(&detok_w, &dec_w, &mut dt_state, &f_codes);
                let npu_spec = xdna.detokenize_to_spectrum(&detok_w, &f_codes);
                assert!(xdna.take_audio_error().is_none());
                assert_eq!(npu_spec.len(), cpu_spec.len());
                assert!(npu_spec.iter().all(|v| v.is_finite()));
            }

            drop(xdna);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_detokenizer_rejects_invalid_config_and_zero_swa_attn() {
        fake::with_shim(FakeConfig::default(), || {
            let (dec_w, mut detok_w) = synth_lfm2a_weights();
            let runner = XdnaGemmRunner::open_shared().expect("open_shared");

            detok_w.config.layer_is_conv.pop();
            let err = match XdnaAudioDecoder::from_parts(runner.clone(), None, Some(&detok_w)) {
                Ok(_) => panic!("expected layer_is_conv mismatch error"),
                Err(e) => e,
            };
            assert!(err.to_string().contains("layer_is_conv"));

            let (_, mut bad_swa_detok) = synth_lfm2a_weights();
            bad_swa_detok.config.swa_window_size = 0;
            let err = match XdnaAudioDecoder::from_parts(runner.clone(), None, Some(&bad_swa_detok))
            {
                Ok(_) => panic!("expected swa_window_size=0 attention error"),
                Err(e) => e,
            };
            assert!(err.to_string().contains("swa_window_size"));

            let (_, mut bad_embd_detok) = synth_lfm2a_weights();
            bad_embd_detok.config.n_embd = 0;
            let err =
                match XdnaAudioDecoder::from_parts(runner.clone(), None, Some(&bad_embd_detok)) {
                    Ok(_) => panic!("expected n_embd=0 error"),
                    Err(e) => e,
                };
            assert!(err.to_string().contains("n_embd"));

            // Conv-only detokenizers do not touch attention KV and may set swa_window_size = 0.
            let (_, mut conv_only_detok) = synth_lfm2a_weights();
            conv_only_detok.config.swa_window_size = 0;
            conv_only_detok.config.layer_is_conv = vec![true];
            conv_only_detok.layers.truncate(1);
            let xdna = XdnaAudioDecoder::from_parts(runner, Some(&dec_w), Some(&conv_only_detok))
                .expect("conv-only detokenizer with swa_window_size=0 should construct");
            let spec = xdna.detokenize_to_spectrum(&conv_only_detok, &[1; 8]);
            assert!(xdna.take_audio_error().is_none());
            assert!(!spec.is_empty());

            drop(xdna);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_lfm2a_shared_projection_and_small_max_seq_len() {
        fake::with_shim(FakeConfig::default(), || {
            let (mut dec_w, _) = synth_lfm2a_weights();
            let n_embd = dec_w.depthformer_config.n_embd;
            let llm_dim = dec_w.decoder_config.n_embd;
            let n_vocab = dec_w.decoder_config.n_vocab as i32;
            dec_w.depthformer_config.max_seq_len = 2;
            dec_w.depth_linear_w = f32_w(n_embd, llm_dim, 80, 0.2);
            dec_w.depth_linear_b = det_vec(n_embd, 81, 0.02);

            let runner = XdnaGemmRunner::open_shared().expect("open_shared");
            let xdna = XdnaAudioDecoder::from_parts(runner.clone(), Some(&dec_w), None)
                .expect("shared projection lfm2a");
            let embed = det_vec(llm_dim, 7, 0.5);
            let codes = xdna.sample_audio_frame(&embed, 0.0, 1);
            assert!(xdna.take_audio_error().is_none());
            assert!(codes.iter().all(|&c| (0..n_vocab).contains(&c)));

            dec_w.depth_linear_w = f32_w(2 * n_embd, llm_dim, 80, 0.2);
            dec_w.depth_linear_b = det_vec(2 * n_embd, 81, 0.02);
            let bad_xdna = XdnaAudioDecoder::from_parts(runner, Some(&dec_w), None)
                .expect("construct with undersized depth_linear");
            let fallback = bad_xdna.sample_audio_frame(&embed, 0.0, 1);
            assert_eq!(fallback, [0; 8]);
            let err = bad_xdna
                .take_audio_error()
                .expect("undersized depth_linear must surface error");
            assert!(err.to_string().contains("depth_linear"));

            drop(xdna);
            drop(bad_xdna);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_lfm3a_repetition_penalty_oob_history_and_empty_logits() {
        fake::with_shim(FakeConfig::default(), || {
            let (mut dec_w, _) = synth_lfm3a_weights();
            let n_embd = dec_w.depthformer_config.n_embd;
            dec_w.depth_embeddings[0].to_logits = f32_w(4, n_embd, 70, 0.25);
            dec_w.depth_embeddings[1].to_logits = MmapWeight::from_owned_f32(Vec::new(), 0, n_embd);

            let runner = XdnaGemmRunner::open_shared().expect("open_shared");
            let xdna = XdnaAudioDecoder::from_parts(runner, Some(&dec_w), None)
                .expect("XdnaAudioDecoder lfm3a");
            let df = xdna.depthformer.as_ref().expect("depthformer");
            {
                let mut st = df.state.lock().unwrap_or_else(|e| e.into_inner());
                st.push_code_history(vec![200, 15, i32::MAX, -3]);
            }

            let embed = det_vec(dec_w.decoder_config.n_embd, 19, 0.4);
            let greedy = xdna
                .sample_audio_frame_lfm3a(&embed, 0.0, 1)
                .expect("greedy lfm3a with empty and smaller codebook logits");
            assert!(xdna.take_audio_error().is_none());
            assert_eq!(greedy.len(), 12);
            assert!((0..4).contains(&greedy[0]));

            let sampled = xdna
                .sample_audio_frame_lfm3a(&embed, 0.8, 4)
                .expect("stochastic lfm3a with empty and smaller codebook logits");
            assert!(xdna.take_audio_error().is_none());
            assert_eq!(sampled.len(), 12);
            assert!((0..4).contains(&sampled[0]));

            drop(xdna);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_take_audio_error_drains_runner_fault() {
        use crate::model::vision_encoder_gpu::VitGpuOps;
        use crate::model::xdna_towers::XdnaVitOps;

        fake::with_shim(FakeConfig::default(), || {
            let (dec_w, _) = synth_lfm2a_weights();
            let runner = XdnaGemmRunner::open_shared().expect("open_shared");
            let xdna = XdnaAudioDecoder::from_parts(runner.clone(), Some(&dec_w), None)
                .expect("XdnaAudioDecoder");
            assert!(xdna.take_audio_error().is_none());

            let _ = XdnaVitOps::new(runner).upload_weight_f32(&[], 1, 1);
            let err = xdna
                .take_audio_error()
                .expect("runner fault should surface through take_audio_error");
            assert!(
                err.to_string()
                    .contains("gemm_bf16 weight length 0 != 1 * 1"),
                "unexpected runner error: {err}"
            );
            assert!(xdna.take_audio_error().is_none());

            drop(xdna);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_build_audio_accelerator_routes_empty_gguf_none() {
        fake::with_shim(FakeConfig::default(), || {
            let empty = Arc::new(GgufBuilder::new().build());
            assert!(
                build_audio_accelerator(&empty, crate::engine::BackendPreference::Npu).is_none()
            );
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_audio_decoder_lfm2a_and_lfm3a_parity() {
        let runner = XdnaGemmRunner::open_shared().expect("open XDNA gemm runner");

        // 1. LFM2.5-Audio (lfm2a): 8-codebook causal depthformer + 6x upsampled detokenizer.
        {
            let (dec_w, detok_w) = synth_lfm2a_weights();
            let xdna = XdnaAudioDecoder::from_parts(runner.clone(), Some(&dec_w), Some(&detok_w))
                .expect("XdnaAudioDecoder lfm2a");
            let embed = det_vec(dec_w.decoder_config.n_embd, 7, 0.5);
            let mut df_state = DepthformerState::new(&dec_w.depthformer_config);
            let cpu_codes = sample_audio_frame(&dec_w, &mut df_state, &embed, 0.0, 1);
            let npu_codes = xdna.sample_audio_frame(&embed, 0.0, 1);
            assert!(xdna.take_audio_error().is_none());
            assert_eq!(npu_codes, cpu_codes, "NPU lfm2a depthformer greedy codes");

            let mut dt_state = DetokenizerState::new(&detok_w.config);
            for frame_idx in 0..3 {
                let mut f_codes = cpu_codes;
                f_codes[0] = (f_codes[0] + frame_idx) % 16;
                let cpu_spec = detokenize_to_spectrum(&detok_w, &dec_w, &mut dt_state, &f_codes);
                let npu_spec = xdna.detokenize_to_spectrum(&detok_w, &f_codes);
                assert!(xdna.take_audio_error().is_none());
                let cos = cosine_sim(&npu_spec, &cpu_spec);
                eprintln!("HW LFM2A detok frame {frame_idx}: cos={cos:.6}");
                assert!(cos > 0.999, "NPU lfm2a detok frame {frame_idx} cos={cos}");
            }
        }

        // 2. LFM3-Audio / Omni (lfm3a_gen): 12-codebook MaskGIT depthformer + FSQ detokenizer.
        {
            let (dec_w, detok_w) = synth_lfm3a_weights();
            let xdna = XdnaAudioDecoder::from_parts(runner, Some(&dec_w), Some(&detok_w))
                .expect("XdnaAudioDecoder lfm3a");
            let embed = det_vec(dec_w.decoder_config.n_embd, 19, 0.4);
            let mut df_state = DepthformerState::new(&dec_w.depthformer_config);

            for frame_idx in 0..2 {
                let cpu_codes = sample_frame(&dec_w, &mut df_state, &embed, 0.0, 1);
                let npu_codes = xdna
                    .sample_audio_frame_lfm3a(&embed, 0.0, 1)
                    .expect("NPU lfm3a codes");
                assert!(xdna.take_audio_error().is_none());
                assert_eq!(
                    npu_codes, cpu_codes,
                    "NPU lfm3a MaskGIT codes mismatch at frame {frame_idx}"
                );
            }

            let mut dt_state = DetokenizerState::new(&detok_w.config);
            let seed_codes = sample_frame(&dec_w, &mut df_state, &embed, 0.0, 1);
            for frame_idx in 0..3 {
                let mut f_codes = seed_codes.clone();
                f_codes[0] = (f_codes[0] + frame_idx) % 256;
                let cpu_spec = detokenize_to_spectrum(&detok_w, &dec_w, &mut dt_state, &f_codes);
                let npu_spec = xdna.detokenize_to_spectrum(&detok_w, &f_codes);
                assert!(xdna.take_audio_error().is_none());
                let cos = cosine_sim(&npu_spec, &cpu_spec);
                eprintln!("HW LFM3A detok frame {frame_idx}: cos={cos:.6}");
                assert!(cos > 0.999, "NPU lfm3a detok frame {frame_idx} cos={cos}");
            }
        }
    }
}

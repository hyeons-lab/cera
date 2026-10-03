use super::*;
use crate::backend::hexagon::{HtpOpDesc, HtpTensor};
use crate::model::{ModelConfig, ScalarMultipliers};

fn tiny_config() -> ModelConfig {
    ModelConfig {
        architecture: "lfm2".into(),
        n_layers: 1,
        hidden_size: 8,
        intermediate_size: 16,
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 4,
        vocab_size: 32,
        max_seq_len: 2048,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types: vec![BlockType::Attention],
        conv_kernel_size: None,
        ssm: None,
        kv_heads_per_layer: vec![2],
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}

#[test]
fn failed_chunk_aborts_and_skips_later_chunks() {
    let config = tiny_config();
    let mut state = InferenceState::from_config(&config).unwrap();
    // Three chunks; the scripted runner fails chunk 2.
    let tokens: Vec<u32> = (0..PREFILL_MAX_ROWS * 2 + 7)
        .map(|i| (i % 31 + 1) as u32)
        .collect();
    let mut ran: Vec<(usize, usize)> = Vec::new();
    let (consumed, logits) = run_scratch_chunks(&tokens, 0, &mut state, |chunk, pos, _state| {
        ran.push((chunk.len(), pos));
        if ran.len() == 2 {
            return None;
        }
        Some(vec![1.0f32; config.vocab_size])
    });
    // `consumed` stops at the failed chunk (the session advances over
    // exactly this prefix); the last good logits survive for direct
    // callers that can use them.
    assert_eq!(consumed, PREFILL_MAX_ROWS);
    assert_eq!(logits, Some(vec![1.0f32; config.vocab_size]));
    assert_eq!(ran.len(), 2, "chunk 3 ran after chunk 2 failed: {ran:?}");
    assert_eq!(ran[0], (PREFILL_MAX_ROWS, 0));
    assert_eq!(ran[1], (PREFILL_MAX_ROWS, PREFILL_MAX_ROWS));
}

#[test]
fn all_chunks_ok_returns_last_logits() {
    let config = tiny_config();
    let mut state = InferenceState::from_config(&config).unwrap();
    let tokens: Vec<u32> = (0..PREFILL_MAX_ROWS + 3)
        .map(|i| (i % 31 + 1) as u32)
        .collect();
    let (consumed, logits) = run_scratch_chunks(&tokens, 0, &mut state, |_chunk, pos, _state| {
        Some(vec![pos as f32; config.vocab_size])
    });
    // Final chunk's logits win; positions advance per chunk.
    assert_eq!(consumed, tokens.len());
    assert_eq!(
        logits,
        Some(vec![PREFILL_MAX_ROWS as f32; config.vocab_size])
    );
    // Empty prompt: `(0, None)` without invoking the runner.
    let (consumed, logits) = run_scratch_chunks(&[], 0, &mut state, |_, _, _| {
        panic!("runner invoked for empty tokens")
    });
    assert_eq!((consumed, logits), (0, None));
}

#[test]
fn prefill_tail_logits_maps_short_run_to_zeros() {
    // Short run → zeros even when a last-good chunk exists: returning
    // the stale logits would sample over a KV hole. Deleting the `else`
    // leg must fail this test.
    assert_eq!(
        prefill_tail_logits(5, 30, Some(vec![1.0f32; 4]), 4),
        vec![0.0f32; 4]
    );
    // Full run → the last chunk's logits untouched.
    assert_eq!(
        prefill_tail_logits(30, 30, Some(vec![2.0f32; 4]), 4),
        vec![2.0f32; 4]
    );
    // Empty input → zeros (no chunk ran, so `None`).
    assert_eq!(prefill_tail_logits(0, 0, None, 4), vec![0.0f32; 4]);
}

#[test]
fn run_scratch_chunks_embeddings_all_ok() {
    let config = tiny_config();
    let mut state = InferenceState::from_config(&config).unwrap();
    let n_tokens = PREFILL_MAX_ROWS * 2 + 5;
    let hs = config.hidden_size;
    let embeddings: Vec<f32> = (0..n_tokens * hs).map(|i| i as f32).collect();
    let (consumed, logits) = run_scratch_chunks_embeddings(
        &embeddings,
        n_tokens,
        hs,
        0,
        &mut state,
        |_chunk, pos, _state| Some(vec![pos as f32; config.vocab_size]),
    );
    assert_eq!(consumed, n_tokens);
    assert_eq!(
        logits,
        Some(vec![(PREFILL_MAX_ROWS * 2) as f32; config.vocab_size])
    );
}

#[test]
fn run_scratch_chunks_embeddings_aborts_on_failure() {
    let config = tiny_config();
    let mut state = InferenceState::from_config(&config).unwrap();
    let n_tokens = PREFILL_MAX_ROWS * 3;
    let hs = config.hidden_size;
    let embeddings: Vec<f32> = vec![0.5f32; n_tokens * hs];
    let mut ran = 0;
    let (consumed, logits) = run_scratch_chunks_embeddings(
        &embeddings,
        n_tokens,
        hs,
        0,
        &mut state,
        |_chunk, _pos, _state| {
            ran += 1;
            if ran == 2 {
                return None;
            }
            Some(vec![1.0f32; config.vocab_size])
        },
    );
    assert_eq!(consumed, PREFILL_MAX_ROWS);
    assert_eq!(ran, 2);
    assert_eq!(logits, Some(vec![1.0f32; config.vocab_size]));
}

#[test]
fn run_scratch_chunks_embeddings_empty() {
    let config = tiny_config();
    let mut state = InferenceState::from_config(&config).unwrap();
    let (consumed, logits) =
        run_scratch_chunks_embeddings(&[], 0, config.hidden_size, 0, &mut state, |_, _, _| {
            panic!("should not run")
        });
    assert_eq!(consumed, 0);
    assert_eq!(logits, None);
}

#[test]
fn test_scratch_offsets_alignment_and_non_overlapping() {
    let offsets = ScratchOffsets::new(1024, 1024, 256, 4096, 32000, 2048, None, None);
    let list = [
        ("activation", offsets.activation),
        ("activation_b", offsets.activation_b),
        ("normed", offsets.normed),
        ("normed_b", offsets.normed_b),
        ("q", offsets.q),
        ("k", offsets.k),
        ("v", offsets.v),
        ("attn_out", offsets.attn_out),
        ("conv_in", offsets.conv_in),
        ("conv_bx", offsets.conv_bx),
        ("conv_t0", offsets.conv_t0),
        ("conv_t1", offsets.conv_t1),
        ("conv_y", offsets.conv_y),
        ("conv_x", offsets.conv_x),
        ("conv_ssm_y", offsets.conv_ssm_y),
        ("ffn_gate", offsets.ffn_gate),
        ("ffn_up", offsets.ffn_up),
        ("ffn_out", offsets.ffn_out),
        ("logits", offsets.logits),
        ("argmax", offsets.argmax),
        ("pos", offsets.pos),
        ("mask", offsets.mask),
        ("total_size", offsets.total_size),
    ];

    // All offsets must be 4096-byte aligned.
    for (name, offset) in &list {
        assert_eq!(
            offset % 4096,
            0,
            "offset for {name} ({offset}) must be 4096-byte aligned"
        );
    }

    // Each section strictly proceeds the previous one without overlapping.
    for i in 0..list.len() - 1 {
        assert!(
            list[i].1 < list[i + 1].1,
            "offset for {} ({}) must be strictly less than next offset {} ({})",
            list[i].0,
            list[i].1,
            list[i + 1].0,
            list[i + 1].1
        );
    }
}

#[test]
fn test_moe_scratch_offsets_alignment_and_non_overlapping() {
    let moe_cfg = crate::model::MoeConfig {
        n_expert: 32,
        n_expert_used: 4,
        expert_ff_len: 1792,
        is_moe_layer: vec![true; 16],
    };
    let offsets = ScratchOffsets::new(1024, 1024, 256, 4096, 32000, 2048, Some(&moe_cfg), None);
    let list = [
        ("activation", offsets.activation),
        ("activation_b", offsets.activation_b),
        ("normed", offsets.normed),
        ("normed_b", offsets.normed_b),
        ("q", offsets.q),
        ("k", offsets.k),
        ("v", offsets.v),
        ("attn_out", offsets.attn_out),
        ("conv_in", offsets.conv_in),
        ("conv_bx", offsets.conv_bx),
        ("conv_t0", offsets.conv_t0),
        ("conv_t1", offsets.conv_t1),
        ("conv_y", offsets.conv_y),
        ("conv_x", offsets.conv_x),
        ("conv_ssm_y", offsets.conv_ssm_y),
        ("ffn_gate", offsets.ffn_gate),
        ("ffn_up", offsets.ffn_up),
        ("ffn_out", offsets.ffn_out),
        ("moe_router_logits", offsets.moe_router_logits),
        ("moe_probs", offsets.moe_probs),
        ("moe_biased_probs", offsets.moe_biased_probs),
        ("moe_selected_ids", offsets.moe_selected_ids),
        ("moe_selected_weights", offsets.moe_selected_weights),
        ("moe_gate", offsets.moe_gate),
        ("moe_up", offsets.moe_up),
        ("moe_swiglu", offsets.moe_swiglu),
        ("moe_down", offsets.moe_down),
        ("moe_temp_weighted", offsets.moe_temp_weighted),
        ("logits", offsets.logits),
        ("argmax", offsets.argmax),
        ("pos", offsets.pos),
        ("mask", offsets.mask),
        ("total_size", offsets.total_size),
    ];

    // All offsets must be 4096-byte aligned.
    for (name, offset) in &list {
        assert_eq!(
            offset % 4096,
            0,
            "offset for {name} ({offset}) must be 4096-byte aligned"
        );
    }

    // Each section strictly proceeds the previous one without overlapping.
    for i in 0..list.len() - 1 {
        assert!(
            list[i].1 < list[i + 1].1,
            "offset for {} ({}) must be strictly less than next offset {} ({})",
            list[i].0,
            list[i].1,
            list[i + 1].0,
            list[i + 1].1
        );
    }
}

#[test]
fn test_kv_q8_0_sizing_and_strides() {
    let (max_seq_len, kv_dim, head_dim) = (2048usize, 256usize, 64usize);
    let f16_slab = kv_cache_bytes(HtpDataType::F16, kv_dim, max_seq_len);
    let q8_slab = kv_cache_bytes(HtpDataType::Q8_0, kv_dim, max_seq_len);
    // 34 bytes per 32-element block against 64 for f16: ~53%.
    assert_eq!(f16_slab, 1_048_576);
    assert_eq!(q8_slab, 557_056);
    assert!(q8_slab < f16_slab);
    assert_eq!(kv_row_stride(HtpDataType::Q8_0, kv_dim), 272);
    assert_eq!(kv_row_stride(HtpDataType::Q8_0, head_dim), 68);
    assert_eq!(kv_row_stride(HtpDataType::F16, kv_dim), 512);
    assert_eq!(kv_elem_nb0(HtpDataType::Q8_0), 1);
    assert_eq!(kv_elem_nb0(HtpDataType::F16), 2);
    // A dim that is not a whole number of blocks rounds up to a block.
    assert_eq!(
        kv_row_stride(HtpDataType::Q8_0, 33),
        2 * crate::tensor::DType::Q8_0.block_bytes()
    );
}

/// Raw bytes of a slice of plain `repr(C)` descriptors.
fn descriptor_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: descriptors are `repr(C)` plain data; only read here.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[test]
fn test_decode_template_flash_attn_patching() {
    let tensor = |ti: u16, size: u32, ne: [u32; 4], nb: [u32; 4]| HtpTensor {
        data: 0,
        size,
        flags: 0,
        dtype: HtpDataType::F16 as u32,
        bi: 0,
        ti,
        ne,
        nb,
    };
    let tens = [
        tensor(0, 0, [64, 1, 4, 1], [2, 128, 512, 512]),
        tensor(1, 0, [64, 1, 4, 1], [2, 128, 512, 512]),
        tensor(2, 2, [1, 1, 1, 1], [2, 2, 2, 2]),
    ];
    let mut op = HtpOpDesc {
        opcode: HtpOpCode::FlashAttnExt as u32,
        flags: 0,
        params: [0; 16],
        kernel_params: [0; 32],
        src: [0; 10],
        dst: [0; 4],
        pad: [0; 2],
    };
    op.kernel_params =
        build_flash_attn_kernel_params_with_softcap(64, 16, 4, 1, 1, 0.125, 4, true, 0.0);

    // Serialize into the same layout `export_staged_batch` produces
    // (tensors, then ops), so the descriptor accessors address real bytes.
    let (tens_bytes, ops_bytes) = (
        tens.len() * std::mem::size_of::<HtpTensor>(),
        std::mem::size_of::<HtpOpDesc>(),
    );
    let mut raw_bytes = Vec::with_capacity(tens_bytes + ops_bytes);
    raw_bytes.extend_from_slice(descriptor_bytes(&tens));
    raw_bytes.extend_from_slice(descriptor_bytes(std::slice::from_ref(&op)));
    let mut staged = StagedBatch {
        raw_bytes,
        n_bufs: 0,
        n_tensors: tens.len() as u32,
        n_ops: 1,
        bufs_bytes: 0,
        tens_bytes,
        ops_bytes,
        prof_bytes: 0,
        total_bytes: tens_bytes + ops_bytes,
    };
    let patch = FlashAttnPatch {
        op_idx: 0,
        k_ti: 0,
        v_ti: 1,
        mask_ti: 2,
        g: 16 / 4,
    };

    // Decode step at pos = 15 (seq_len = 16), then pos = 64 (seq_len 65:
    // a second KV block).
    for seq_len in [16usize, 65] {
        apply_flash_attn_patches(&mut staged, &[patch], seq_len);
        let mask_bytes = (seq_len * 2) as u32;
        assert_eq!(staged.tensor(0).ne[1], seq_len as u32);
        assert_eq!(staged.tensor(1).ne[1], seq_len as u32);
        let mask = staged.tensor(2);
        assert_eq!(mask.ne[0], seq_len as u32);
        assert_eq!(mask.size, mask_bytes);
        assert_eq!(mask.nb[1..], [mask_bytes; 3]);
        assert_eq!(
            staged.op(0).kernel_params,
            build_flash_attn_kernel_params_with_softcap(64, 16, 4, 1, seq_len, 0.125, 4, true, 0.0),
            "seq_len {seq_len}"
        );
    }
}

#[test]
fn chunked_all_logits_splits_offsets_and_orders_rows() {
    let vocab = 2;
    let max = 4;
    let tokens: Vec<u32> = (0..(2 * max + 1) as u32).collect();
    let mut calls = Vec::new();
    let out = chunked_all_logits(&tokens, 10, max, vocab, |chunk, start| {
        calls.push((start, chunk.len()));
        Ok(chunk
            .iter()
            .flat_map(|&t| [t as f32, -(t as f32)])
            .collect())
    })
    .unwrap();
    assert_eq!(calls, vec![(10, 4), (14, 4), (18, 1)]);
    assert_eq!(out.len(), tokens.len() * vocab);
    for (i, row) in out.chunks(vocab).enumerate() {
        assert_eq!(row, [i as f32, -(i as f32)], "row {i} out of order");
    }
}

#[test]
fn chunked_all_logits_aborts_on_first_chunk_error() {
    let mut ran = 0;
    let r = chunked_all_logits(&[1, 2, 3, 4, 5], 0, 2, 1, |_, _| {
        ran += 1;
        if ran == 2 {
            Err(CeraError::Backend("boom".into()))
        } else {
            Ok(vec![0.0; 2])
        }
    });
    assert!(r.is_err());
    assert_eq!(ran, 2, "chunks after the failed one must not run");
}

#[test]
fn test_rope_mode_derivation() {
    let neox_mode = RopeType::Neox.htp_mode();
    let norm_mode = RopeType::Norm.htp_mode();
    assert_eq!(neox_mode, 2);
    assert_eq!(norm_mode, 0);
}

/// The FFN emit maps each activation to its own GLU opcode: drive the
/// production `dispatch_glu` and read the recorded op back.
#[test]
fn test_hexagon_glu_opcode_selection() {
    use crate::backend::hexagon::op_capture;
    for (activation, want) in [
        (FfnActivation::Swiglu, HtpOpCode::GluSwiglu),
        (FfnActivation::Geglu, HtpOpCode::GluGeglu),
    ] {
        let (driver, mut device) = op_capture::fresh_device();
        let buf = RpcmemBuffer::alloc(driver, 4096, false).unwrap();
        let session = device.queue_session_mut();
        HexagonLfmModel::dispatch_glu(session, &buf, 0, &buf, 1024, &buf, 2048, 8, 1, activation)
            .unwrap();
        session.flush().unwrap();
        let text = op_capture::take().join("");
        let ops: Vec<&str> = text.lines().filter_map(|l| l.split(' ').next()).collect();
        assert_eq!(ops, [want.name()], "{activation:?}");
    }
}

#[test]
fn test_hexagon_flash_attn_softcap_params() {
    let softcap = 50.0f32;
    let kparams =
        build_flash_attn_kernel_params_with_softcap(64, 16, 4, 1, 128, 0.125, 4, true, softcap);
    assert_eq!(kparams[5], softcap.to_bits() as i32);

    let zero_softcap = 0.0f32;
    let kparams_zero = build_flash_attn_kernel_params_with_softcap(
        64,
        16,
        4,
        1,
        128,
        0.125,
        4,
        true,
        zero_softcap,
    );
    assert_eq!(kparams_zero[5], 0);
}

#[test]
fn test_post_norm_offset_planning() {
    let mut total = 0usize;
    let hidden_size = 2048;
    let attn_norm_off = plan_offset(&mut total, hidden_size * 4);
    let q_off = plan_offset(&mut total, hidden_size * 4);
    let post_norm_off = plan_offset(&mut total, hidden_size * 4);
    assert_eq!(attn_norm_off, 0);
    assert_eq!(q_off, 8192);
    assert_eq!(post_norm_off, 16384);
    assert_eq!(post_norm_off % 256, 0);
}

#[test]
fn test_deltanet_scratch_offsets_alignment_and_non_overlapping() {
    let ssm_conv_dim = 2 * (4 * 128) + 16 * 128;
    let offsets = ScratchOffsets::new(1024, 1024, 256, 4096, 32000, 2048, None, Some(ssm_conv_dim));
    let list = [
        ("activation", offsets.activation),
        ("activation_b", offsets.activation_b),
        ("normed", offsets.normed),
        ("normed_b", offsets.normed_b),
        ("q", offsets.q),
        ("k", offsets.k),
        ("v", offsets.v),
        ("attn_out", offsets.attn_out),
        ("conv_in", offsets.conv_in),
        ("conv_bx", offsets.conv_bx),
        ("conv_t0", offsets.conv_t0),
        ("conv_t1", offsets.conv_t1),
        ("conv_y", offsets.conv_y),
        ("conv_x", offsets.conv_x),
        ("conv_ssm_y", offsets.conv_ssm_y),
        ("ffn_gate", offsets.ffn_gate),
        ("ffn_up", offsets.ffn_up),
        ("ffn_out", offsets.ffn_out),
        ("logits", offsets.logits),
        ("argmax", offsets.argmax),
        ("pos", offsets.pos),
        ("mask", offsets.mask),
        ("total_size", offsets.total_size),
    ];

    for (name, offset) in &list {
        assert_eq!(
            offset % 4096,
            0,
            "offset for {name} ({offset}) must be 4096-byte aligned"
        );
    }

    for i in 0..list.len() - 1 {
        assert!(
            list[i].1 < list[i + 1].1,
            "offset for {} ({}) must be strictly less than next offset {} ({})",
            list[i].0,
            list[i].1,
            list[i + 1].0,
            list[i + 1].1
        );
    }
}

/// The attention rule the CPU reference uses: query at position `p` sees
/// slot `j` iff `j <= p` and, with a window `w`, `j >= p + 1 - w`.
fn cpu_allows(p: usize, j: usize, window: Option<usize>) -> bool {
    j <= p && window.filter(|&w| w > 0).is_none_or(|w| j + w > p)
}

#[test]
fn test_prefill_mask_matches_cpu_window_rule() {
    for window in [None, Some(0), Some(1), Some(3), Some(64)] {
        for (start_pos, m) in [(0usize, 5usize), (4, 3), (10, 1), (7, 8)] {
            let kv_len = start_pos + m;
            let mut mask = vec![0xABCDu16; kv_len * m];
            fill_prefill_mask(&mut mask, start_pos, m, kv_len, window);
            for mm in 0..m {
                for j in 0..kv_len {
                    let want = if cpu_allows(start_pos + mm, j, window) {
                        0x0000
                    } else {
                        MASK_NEG_INF
                    };
                    assert_eq!(
                        mask[mm * kv_len + j],
                        want,
                        "window {window:?} start {start_pos} m {m} row {mm} slot {j}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_decode_swa_mask_is_last_prefill_row() {
    for (seq_len, window) in [(1usize, 4usize), (4, 4), (9, 4), (100, 7)] {
        let mut decode = vec![0u16; seq_len];
        fill_decode_swa_mask(&mut decode, seq_len, window);
        let mut prefill = vec![0u16; seq_len];
        fill_prefill_mask(&mut prefill, seq_len - 1, 1, seq_len, Some(window));
        assert_eq!(decode, prefill);
        let visible = decode.iter().filter(|&&v| v == 0).count();
        assert_eq!(visible, seq_len.min(window));
    }
}

#[test]
fn test_attn_temp_q_scale_matches_cpu_formula() {
    assert_eq!(attn_temp_q_scale(100, None), None);
    // Below the floor position (and for a non-positive scale) nothing scales.
    assert_eq!(attn_temp_q_scale(15, Some((0.1, 16))), None);
    assert_eq!(attn_temp_q_scale(100, Some((0.0, 16))), None);
    assert_eq!(attn_temp_q_scale(100, Some((0.1, 0))), None);
    let at_floor = attn_temp_q_scale(16, Some((0.1, 16))).unwrap();
    assert!((at_floor - ((2.0f32).ln() * 0.1 + 1.0)).abs() < 1e-6);
    let later = attn_temp_q_scale(48, Some((0.1, 16))).unwrap();
    assert!((later - ((4.0f32).ln() * 0.1 + 1.0)).abs() < 1e-6);
}

#[test]
fn test_host_rope_routes_like_the_cpu_reference() {
    use crate::backend::cpu;
    let (n_heads, n_kv, hd, theta, pos) = (2usize, 1usize, 8usize, 10_000.0f32, 5usize);
    let q0: Vec<f32> = (0..n_heads * hd).map(|i| 0.1 * i as f32 - 0.7).collect();
    let k0: Vec<f32> = (0..n_kv * hd).map(|i| 0.05 * i as f32 + 0.3).collect();
    let freqs: Vec<f32> = (0..hd / 2).map(|i| 1.0 + i as f32).collect();
    let yarn = cpu::YarnParams::new_with_log_mul(0.25, 1.0, 1.0, 32.0, 1.0, 64, 0.1);

    let run = |rope_type, y: Option<&cpu::YarnParams>, f: Option<&[f32]>| {
        let (mut q, mut k) = (q0.clone(), k0.clone());
        host_rope(
            &mut crate::model::llama::RopeGather::default(),
            rope_type,
            y,
            f,
            &mut q,
            &mut k,
            pos,
            n_heads,
            n_kv,
            hd,
            hd,
            theta,
        );
        (q, k)
    };

    // NEOX without YaRN ignores the factors (as the CPU path does).
    let (mut q, mut k) = (q0.clone(), k0.clone());
    cpu::rope(&mut q, &mut k, pos, n_heads, n_kv, hd, theta);
    assert_eq!(run(RopeType::Neox, None, Some(&freqs)), (q, k));

    // NORM applies the factors only without YaRN.
    let (mut q, mut k) = (q0.clone(), k0.clone());
    cpu::rope_norm(&mut q, &mut k, pos, n_heads, n_kv, hd, theta, Some(&freqs));
    assert_eq!(run(RopeType::Norm, None, Some(&freqs)), (q, k));
    let plain = run(RopeType::Norm, None, None);
    assert_ne!(plain.0, run(RopeType::Norm, None, Some(&freqs)).0);

    let (mut q, mut k) = (q0.clone(), k0.clone());
    cpu::rope_norm_yarn(&mut q, &mut k, pos, n_heads, n_kv, hd, theta, &yarn);
    assert_eq!(run(RopeType::Norm, Some(&yarn), Some(&freqs)), (q, k));

    let (mut q, mut k) = (q0.clone(), k0.clone());
    cpu::rope_neox_yarn(&mut q, &mut k, pos, n_heads, n_kv, hd, theta, &yarn);
    assert_eq!(run(RopeType::Neox, Some(&yarn), None), (q, k));

    // Partial rotary rotates a prefix of each head and leaves the tail alone.
    let n_rot = 4;
    let (mut q, mut k) = (q0.clone(), k0.clone());
    host_rope(
        &mut crate::model::llama::RopeGather::default(),
        RopeType::Neox,
        None,
        None,
        &mut q,
        &mut k,
        pos,
        n_heads,
        n_kv,
        hd,
        n_rot,
        theta,
    );
    for h in 0..n_heads {
        assert_eq!(
            &q[h * hd + n_rot..(h + 1) * hd],
            &q0[h * hd + n_rot..(h + 1) * hd]
        );
        assert_ne!(&q[h * hd..h * hd + n_rot], &q0[h * hd..h * hd + n_rot]);
    }
}

/// The host partial rotary equals `llama::rope_partial` (the CPU
/// reference) for both rope types, and reuses its gather buffers: seeded
/// with capacity 64, they keep it (a reallocation, e.g. a fresh `Vec`, would
/// drop it to the exact gather size, 8 and 4 here).
#[test]
fn test_host_rope_partial_matches_rope_partial_and_reuses_scratch() {
    use crate::model::llama::RopeGather;
    let (n_heads, n_kv, hd, n_rot, theta) = (2usize, 1usize, 8usize, 4usize, 10_000.0f32);
    let q0: Vec<f32> = (0..n_heads * hd).map(|i| 0.1 * i as f32 - 0.7).collect();
    let k0: Vec<f32> = (0..n_kv * hd).map(|i| 0.05 * i as f32 + 0.3).collect();
    let mut scratch = RopeGather {
        q: Vec::with_capacity(64),
        k: Vec::with_capacity(64),
    };
    for rope_type in [RopeType::Neox, RopeType::Norm] {
        for pos in [0usize, 5, 9] {
            let (mut q, mut k) = (q0.clone(), k0.clone());
            host_rope(
                &mut scratch,
                rope_type,
                None,
                None,
                &mut q,
                &mut k,
                pos,
                n_heads,
                n_kv,
                hd,
                n_rot,
                theta,
            );
            let (mut qe, mut ke) = (q0.clone(), k0.clone());
            crate::model::llama::rope_partial(
                &mut qe,
                &mut ke,
                pos,
                n_heads,
                n_kv,
                hd,
                n_rot,
                theta,
                rope_type,
                &mut RopeGather::default(),
            );
            assert_eq!((q, k), (qe, ke));
        }
    }
    assert_eq!((scratch.q.capacity(), scratch.k.capacity()), (64, 64));
}

#[test]
fn test_partial_rope_rejects_yarn_and_freq_factors() {
    assert!(ensure_partial_rope_plain(Some(4), 8, false, false).is_ok());
    assert!(ensure_partial_rope_plain(None, 8, true, true).is_ok());
    assert!(ensure_partial_rope_plain(Some(8), 8, true, true).is_ok());
    for (yarn, freqs) in [(true, false), (false, true), (true, true)] {
        let err = ensure_partial_rope_plain(Some(4), 8, yarn, freqs).unwrap_err();
        assert!(err.to_string().contains("partial rotary"), "{err}");
    }
}

#[test]
fn test_swa_mask_scratch_region_is_disjoint_and_aligned() {
    let max_seq = 256;
    let base = ScratchOffsets::new(64, 64, 32, 128, 1000, max_seq, None, None);
    let swa = base.with_swa_mask(max_seq);
    assert_eq!(base.mask_swa, 0);
    assert_eq!(swa.mask, base.mask);
    assert_eq!(swa.mask_swa % 4096, 0);
    assert!(swa.mask_swa >= base.total_size);
    assert_eq!(
        swa.total_size,
        (swa.mask_swa + PREFILL_MAX_ROWS * max_seq * 2 + 4095) & !4095
    );
}

#[test]
fn test_dense_semantics_default_is_identity() {
    let d = DenseSemantics::default();
    assert!(d.attn_scale.is_none() && d.residual_vec_offset.is_none());
    assert!(d.logit_scale.is_none() && d.rope_freqs.is_none());
    assert!(!d.post_norm && d.loop_norm_interval.is_none());
    assert!(!d.needs_host_step());
    let d = DenseSemantics {
        attn_temp: Some((0.1, 16)),
        ..Default::default()
    };
    assert!(d.needs_host_step());
}

fn moe_test_config(n_used: usize) -> crate::model::MoeConfig {
    crate::model::MoeConfig {
        n_expert: 32,
        n_expert_used: n_used,
        expert_ff_len: 64,
        is_moe_layer: vec![true; 2],
    }
}

/// The DSP op chain's arithmetic equals the CPU `select_experts`
/// renormalization (sum clamped to 2^-14, divide) on ordinary gates, and
/// stays within an ulp of it on a degenerate all-but-zero gate.
#[test]
fn test_moe_renorm_op_chain_matches_cpu_select_experts() {
    use crate::model::lfm2::select_experts;
    let mut state = 0x2545_f491u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    for n_used in [1usize, 2, 4, 8] {
        for scale in [1.0f32, 1e-3, 1e-6, 1e-9] {
            let probs: Vec<f32> = (0..32).map(|_| next() * scale).collect();
            let biases: Vec<f32> = (0..32).map(|_| next() * 0.1).collect();
            let mut selected = Vec::new();
            select_experts(&probs, &biases, n_used, &mut selected);
            let raw: Vec<f32> = selected.iter().map(|&(e, _)| probs[e]).collect();
            let got = moe_renorm_via_op_chain(&raw);
            let sum: f32 = raw.iter().sum();
            for (g, &(_, want)) in got.iter().zip(&selected) {
                if sum >= MOE_DENOM_FLOOR {
                    assert_eq!(*g, want, "n_used {n_used} scale {scale}");
                } else {
                    let tol = want.abs() * 4.0 * f32::EPSILON + f32::MIN_POSITIVE;
                    assert!((g - want).abs() <= tol, "{g} vs {want} (degenerate gate)");
                }
            }
        }
    }
    // A normalized gate sums to 1.
    let w = moe_renorm_via_op_chain(&[0.2, 0.3, 0.1, 0.4]);
    assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

/// Golden op sequence and operand offsets of the renormalization chain,
/// recorded through a real queue session on the fake FastRPC driver.
#[test]
fn test_moe_renorm_denominator_op_sequence_golden() {
    use crate::backend::hexagon::sys::fake;
    fake::reset();
    let driver = fake::driver();
    let so = ScratchOffsets::new(64, 64, 32, 128, 1000, 128, Some(&moe_test_config(4)), None);
    let scratch = RpcmemBuffer::alloc(Arc::clone(&driver), so.total_size, true).unwrap();
    let mut session = HexagonQueueSession::new(driver).unwrap();
    // Hermetic: an exported `CERA_HEXAGON_STEP` must not flush the batch
    // this test exports.
    session.set_step_mode(false);

    let token = 3;
    let denom =
        HexagonLfmModel::dispatch_moe_renorm_denominator(&mut session, &scratch, &so, token, 4)
            .unwrap();

    let slot = so.moe_renorm + token * MOE_RENORM_SLOT_BYTES;
    assert_eq!(denom, slot + 16);
    let (_, _, tens, ops) = session.export_batch();
    let opcodes: Vec<u32> = ops.iter().map(|o| o.opcode).collect();
    assert_eq!(
        opcodes,
        [
            HtpOpCode::Add,
            HtpOpCode::Add,
            HtpOpCode::Add,
            HtpOpCode::Sub,
            HtpOpCode::UnaryRelu,
            HtpOpCode::Add,
        ]
        .map(|o| o as u32)
    );
    let data = |op: usize, which: usize| tens[ops[op].src[which] as usize].data as usize;
    let dst = |op: usize| tens[ops[op].dst[0] as usize].data as usize;
    let w = |e: usize| so.moe_selected_weights + (token * 4 + e) * 4;
    // sum = w0 + w1; sum += w2; sum += w3
    assert_eq!((data(0, 0), data(0, 1), dst(0)), (w(0), w(1), slot));
    assert_eq!((data(1, 0), data(1, 1), dst(1)), (slot, w(2), slot));
    assert_eq!((data(2, 0), data(2, 1), dst(2)), (slot, w(3), slot));
    // gap = floor - sum (Sub, floor first); relu(gap); denom = sum + relu
    assert_eq!((data(3, 0), data(3, 1), dst(3)), (slot + 4, slot, slot + 8));
    assert_eq!((data(4, 0), dst(4)), (slot + 8, slot + 12));
    assert_eq!(
        (data(5, 0), data(5, 1), dst(5)),
        (slot, slot + 12, slot + 16)
    );
    // Every operand is a single f32.
    assert!(tens.iter().all(|t| t.ne == [1, 1, 1, 1] && t.size == 4));
    // The floor slot is seeded once at load.
    HexagonLfmModel::init_moe_renorm_scratch(&scratch, &so);
    let floor = unsafe { *(scratch.as_ptr().add(slot + 4) as *const f32) };
    assert_eq!(floor, MOE_DENOM_FLOOR);
}

fn knobs(pairs: &[(&str, &str)]) -> HexagonKnobs {
    HexagonKnobs::from_lookup(|k| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    })
}

/// The decode tensor cap defaults on, `0` turns it off, and a number sets it.
#[test]
fn test_batch_tensor_cap_knob() {
    let cap = |v: Option<&str>| match v {
        Some(v) => knobs(&[("CERA_HEXAGON_BATCH_TENSORS", v)]).batch_tensors,
        None => knobs(&[]).batch_tensors,
    };
    assert_eq!(cap(None), Some(MAX_TENSORS_PER_FLUSH));
    assert_eq!(cap(Some("0")), None);
    assert_eq!(cap(Some(" 24 ")), Some(24));
    assert_eq!(
        cap(Some("many")),
        Some(MAX_TENSORS_PER_FLUSH),
        "unparsable keeps the default"
    );
    const {
        assert!(
            MAX_TENSORS_PER_FLUSH <= 40,
            "40 is the largest cap seen to decode reproducibly on every model"
        );
    }
}

/// One boolean rule per knob kind: opt-in on for `1`/`true` only, default-on
/// off for `0`/`false` only; an unrecognized value keeps the default.
#[test]
fn test_knob_boolean_rule() {
    let d = knobs(&[]);
    assert!(!d.cpu_rope && !d.debug_barriers && !d.dump_act && !d.kv_q8);
    assert!(d.use_ssm_conv && d.use_hmx);
    assert_eq!(d.adpf_target_nanos, 10_000_000);
    assert_eq!(d.arch_override, None);
    assert_eq!(d.kv_dtype(), HtpDataType::F16);

    for on in ["1", "true", "TRUE", " True "] {
        let k = knobs(&[
            ("CERA_HEXAGON_CPU_ROPE", on),
            ("CERA_HEXAGON_BARRIERS", on),
            ("CERA_DUMP_ACT", on),
            ("CERA_HEXAGON_KV_Q8", on),
        ]);
        assert!(
            k.cpu_rope && k.debug_barriers && k.dump_act && k.kv_q8,
            "{on:?}"
        );
        assert_eq!(k.kv_dtype(), HtpDataType::Q8_0);
    }
    // Opt-in knobs ignore falsy and unrecognized spellings.
    for off in ["0", "false", "", "yes", "on", "2"] {
        let k = knobs(&[("CERA_HEXAGON_CPU_ROPE", off), ("CERA_DUMP_ACT", off)]);
        assert!(!k.cpu_rope && !k.dump_act, "{off:?}");
    }
    for off in ["0", "false", "FALSE", " 0 "] {
        let k = knobs(&[("CERA_HEXAGON_SSM_CONV", off), ("CERA_HEXAGON_HMX", off)]);
        assert!(!k.use_ssm_conv && !k.use_hmx, "{off:?}");
    }
    // Default-on knobs stay on for anything else.
    for keep in ["1", "true", "", "off", "no"] {
        let k = knobs(&[("CERA_HEXAGON_SSM_CONV", keep), ("CERA_HEXAGON_HMX", keep)]);
        assert!(k.use_ssm_conv && k.use_hmx, "{keep:?}");
    }
}

#[test]
fn test_knob_numeric_values() {
    assert_eq!(
        knobs(&[("CERA_HEXAGON_ADPF_TARGET_MS", "25")]).adpf_target_nanos,
        25_000_000
    );
    assert_eq!(
        knobs(&[("CERA_HEXAGON_ADPF_TARGET_MS", "junk")]).adpf_target_nanos,
        10_000_000
    );
    assert_eq!(
        knobs(&[("CERA_HEXAGON_ADPF_TARGET_MS", "18446744073709551615")]).adpf_target_nanos,
        i64::MAX
    );
    assert_eq!(
        knobs(&[("CERA_HEXAGON_ARCH", "79")]).arch_override,
        HexagonArch::from_u32(79)
    );
    assert_eq!(knobs(&[("CERA_HEXAGON_ARCH", "x")]).arch_override, None);
}

//! Committed kernel artifacts (see `kernels/README.md` for provenance,
//! regeneration, and the drift check).

/// Fused Llama-3.2-1B layer: all seven GEMV stages + GQA attention in one run.
pub const Q4LAYER_XCLBIN: &[u8] = include_bytes!("kernels/q4layer.xclbin");
/// Instruction words for [`Q4LAYER_XCLBIN`] (full 9-chunk / 648-token KV stream).
pub const Q4LAYER_INSTS: &[u8] = include_bytes!("kernels/insts_q4layer.bin");
/// Exact per-KV-chunk instruction streams (`c1..c8`, `72` cached tokens per chunk)
/// sharing [`Q4LAYER_XCLBIN`].
pub const Q4LAYER_INSTS_C1: &[u8] = include_bytes!("kernels/insts_q4layer_c1.bin");
pub const Q4LAYER_INSTS_C2: &[u8] = include_bytes!("kernels/insts_q4layer_c2.bin");
pub const Q4LAYER_INSTS_C3: &[u8] = include_bytes!("kernels/insts_q4layer_c3.bin");
pub const Q4LAYER_INSTS_C4: &[u8] = include_bytes!("kernels/insts_q4layer_c4.bin");
pub const Q4LAYER_INSTS_C5: &[u8] = include_bytes!("kernels/insts_q4layer_c5.bin");
pub const Q4LAYER_INSTS_C6: &[u8] = include_bytes!("kernels/insts_q4layer_c6.bin");
pub const Q4LAYER_INSTS_C7: &[u8] = include_bytes!("kernels/insts_q4layer_c7.bin");
pub const Q4LAYER_INSTS_C8: &[u8] = include_bytes!("kernels/insts_q4layer_c8.bin");

/// `(kv_chunks, inst_bytes)` buckets in ascending order (`72` tokens per chunk).
pub const Q4LAYER_KV_BUCKETS: [(usize, &[u8]); 9] = [
    (1, Q4LAYER_INSTS_C1),
    (2, Q4LAYER_INSTS_C2),
    (3, Q4LAYER_INSTS_C3),
    (4, Q4LAYER_INSTS_C4),
    (5, Q4LAYER_INSTS_C5),
    (6, Q4LAYER_INSTS_C6),
    (7, Q4LAYER_INSTS_C7),
    (8, Q4LAYER_INSTS_C8),
    (9, Q4LAYER_INSTS),
];

/// LM-head GEMV at the padded 131072x2048 shape.
pub const Q4HEAD_XCLBIN: &[u8] = include_bytes!("kernels/q4head.xclbin");
/// Instruction words for [`Q4HEAD_XCLBIN`].
pub const Q4HEAD_INSTS: &[u8] = include_bytes!("kernels/insts_q4head.bin");

/// Activation slot stride in `f32` elements inside `q4layer`'s `bo_y`.
pub const Q4LAYER_ACT_SLOT: usize = 8192;
/// Number of activation slots in `q4layer`'s `bo_y`.
pub const Q4LAYER_N_SLOTS: usize = 8;
/// Float offset of attention RMSNorm weights inside slot 0 of `bo_y`.
pub const Q4LAYER_ATTN_NORM_W: usize = 2048;
/// Float offset of packed RoPE cos/sin table inside slot 0 of `bo_y`.
pub const Q4LAYER_ROPE_CS: usize = 4096;
/// Float offset of FFN RMSNorm weights inside slot 4 of `bo_y`.
pub const Q4LAYER_FFN_NORM_W: usize = 2048;
/// Stream tile width in `f32` words (`18,432` bytes).
pub const Q4LAYER_TILE_F32: usize = 4608;
/// Half-tile offset in `f32` words (`9,216` bytes) separating K and V halves.
pub const Q4LAYER_SILU_HALF: usize = 2304;
/// Cached tokens per attention KV chunk.
pub const Q4LAYER_ATTN_CHUNK_T: usize = 72;
/// Total attention KV chunks in `q4layer`'s `bo_y` layout.
pub const Q4LAYER_ATTN_CHUNKS: usize = 9;
/// Number of GQA KV heads in `q4layer`.
pub const Q4LAYER_N_KV_HEADS: usize = 8;
/// Attention head dimension in `q4layer`.
pub const Q4LAYER_HEAD_DIM: usize = 64;
/// Maximum number of prior cached tokens supported by `q4layer`'s 9x72-token
/// bf16 KV stream (`pos < 648`, up to 648 total tokens including current).
pub const MAX_KV_CACHE_TOKENS: usize = Q4LAYER_ATTN_CHUNKS * Q4LAYER_ATTN_CHUNK_T;
/// Per-KV-head stride in `f32` words inside `bo_y`.
pub const Q4LAYER_KV_HEAD_STRIDE: usize = Q4LAYER_ATTN_CHUNKS * Q4LAYER_TILE_F32;
/// Total `f32` words reserved for the bf16 KV cache inside `bo_y`.
pub const Q4LAYER_KV_SLOT: usize = Q4LAYER_N_KV_HEADS * Q4LAYER_KV_HEAD_STRIDE;
/// Float offset where the bf16 KV cache region begins inside `bo_y`.
pub const Q4LAYER_KV_CACHE_BASE: usize = Q4LAYER_N_SLOTS * Q4LAYER_ACT_SLOT;
/// Total `f32` words in `q4layer`'s `bo_y` buffer.
pub const Q4LAYER_ACT_FLOATS: usize = Q4LAYER_KV_CACHE_BASE + Q4LAYER_KV_SLOT;
/// Total byte length of `q4layer`'s `bo_y` buffer.
pub const Q4LAYER_ACT_BYTES: usize = Q4LAYER_ACT_FLOATS * 4;
/// Word offset of `n_kv_cache` (`pos`) inside slot 0 of `bo_y`.
pub const Q4LAYER_ATTN_N_OFF: usize = Q4LAYER_ROPE_CS + 128;
/// Word offset of `skip_kv` inside slot 0 of `bo_y`.
pub const Q4LAYER_ATTN_SKIP_KV_OFF: usize = Q4LAYER_ROPE_CS + 131;

/// Certified minimum mean logit cosine similarity across teacher-forced decode
/// positions (`docs/XDNA_PHASE0_RESULTS.md` Section 4: `mean_cos >= 0.99`).
pub const PARITY_MIN_MEAN_COS: f64 = 0.99;
/// Diagnostic per-step minimum logit cosine similarity floor (`min_cos >= 0.97`,
/// certified Llama-3.2-1B-Instruct Q4_0 baseline `min_cos = 0.979014`).
pub const PARITY_MIN_STEP_COS: f64 = 0.97;
/// Certified maximum CPU top-2 logit margin on any top-1 disagreement (`<= 2.0`).
pub const PARITY_MAX_CPU_MARGIN: f32 = 2.0;

/// Select `(bucket_idx, skip_kv)` for `pos` prior cached tokens across
/// strictly ascending `bucket_chunks` capacities in `1..=Q4LAYER_ATTN_CHUNKS`.
pub fn select_q4layer_kv_bucket(bucket_chunks: &[usize], pos: usize) -> (usize, i32) {
    assert!(!bucket_chunks.is_empty(), "bucket_chunks must be non-empty");
    assert!(
        pos < MAX_KV_CACHE_TOKENS,
        "pos ({pos}) must be < MAX_KV_CACHE_TOKENS ({MAX_KV_CACHE_TOKENS})"
    );
    assert!(
        bucket_chunks
            .iter()
            .all(|&c| (1..=Q4LAYER_ATTN_CHUNKS).contains(&c)),
        "bucket_chunks entries must be in 1..={Q4LAYER_ATTN_CHUNKS}"
    );
    assert!(
        bucket_chunks.windows(2).all(|w| w[0] < w[1]),
        "bucket_chunks must be strictly ascending"
    );
    let needed_chunks = pos.max(1).div_ceil(Q4LAYER_ATTN_CHUNK_T);
    let bucket_idx = bucket_chunks
        .iter()
        .position(|&c| c >= needed_chunks)
        .expect("pos exceeds maximum bucket capacity in bucket_chunks");
    let skip_kv = i32::try_from(Q4LAYER_ATTN_CHUNKS - bucket_chunks[bucket_idx])
        .expect("skip_kv fits in i32");
    (bucket_idx, skip_kv)
}

/// Pack 32-pair `(cos, sin)` tables into the two-half 128-float layout expected
/// by `q4layer.cc` at [`Q4LAYER_ROPE_CS`].
pub fn pack_q4layer_rope_cs(
    rope_cos: &[f32; Q4LAYER_HEAD_DIM / 2],
    rope_sin: &[f32; Q4LAYER_HEAD_DIM / 2],
) -> [f32; 128] {
    let mut rope_cs = [0.0f32; 128];
    for h in 0..2 {
        let base = h * 64;
        rope_cs[base..base + 16].copy_from_slice(&rope_cos[..16]);
        rope_cs[base + 16..base + 32].copy_from_slice(&rope_sin[..16]);
        rope_cs[base + 32..base + 48].copy_from_slice(&rope_cos[16..32]);
        rope_cs[base + 48..base + 64].copy_from_slice(&rope_sin[16..32]);
    }
    rope_cs
}

/// Write little-endian `f32` values into `buf` starting at `float_idx`.
#[inline]
pub fn write_le_f32_slice(buf: &mut [u8], float_idx: usize, vals: &[f32]) {
    let off = float_idx
        .checked_mul(4)
        .expect("write_le_f32_slice float_idx byte offset overflow");
    let byte_len = vals
        .len()
        .checked_mul(4)
        .expect("write_le_f32_slice slice byte length overflow");
    let end = off
        .checked_add(byte_len)
        .expect("write_le_f32_slice end byte offset overflow");
    let dst = &mut buf[off..end];
    for (chunk, &v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(vals) {
        *chunk = v.to_le_bytes();
    }
}

/// Read little-endian `f32` values from `buf` starting at `float_idx`.
#[inline]
pub fn read_le_f32_slice(buf: &[u8], float_idx: usize, dst: &mut [f32]) {
    let off = float_idx
        .checked_mul(4)
        .expect("read_le_f32_slice float_idx byte offset overflow");
    let byte_len = dst
        .len()
        .checked_mul(4)
        .expect("read_le_f32_slice slice byte length overflow");
    let end = off
        .checked_add(byte_len)
        .expect("read_le_f32_slice end byte offset overflow");
    let src = &buf[off..end];
    for (out, chunk) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
        *out = f32::from_le_bytes(*chunk);
    }
}

/// Write a little-endian `i32` into `buf` at `word_idx`.
#[inline]
pub fn write_le_i32(buf: &mut [u8], word_idx: usize, val: i32) {
    let off = word_idx
        .checked_mul(4)
        .expect("write_le_i32 word_idx byte offset overflow");
    let end = off
        .checked_add(4)
        .expect("write_le_i32 end byte offset overflow");
    buf[off..end].copy_from_slice(&val.to_le_bytes());
}

/// Zero `bo_y` and stage static attention and FFN RMSNorm weight vectors at
/// [`Q4LAYER_ATTN_NORM_W`] and `4 * Q4LAYER_ACT_SLOT + Q4LAYER_FFN_NORM_W`.
pub fn init_q4layer_bo_y_norms(buf: &mut [u8], attn_norm: &[f32], ffn_norm: &[f32]) {
    assert_eq!(buf.len(), Q4LAYER_ACT_BYTES);
    assert_eq!(attn_norm.len(), Q4LAYER_ATTN_NORM_W);
    assert_eq!(ffn_norm.len(), Q4LAYER_FFN_NORM_W);
    buf.fill(0);
    write_le_f32_slice(buf, Q4LAYER_ATTN_NORM_W, attn_norm);
    write_le_f32_slice(buf, 4 * Q4LAYER_ACT_SLOT + Q4LAYER_FFN_NORM_W, ffn_norm);
}

/// Stage per-token RoPE table, cached token count (`pos`), and `skip_kv` into
/// slot 0 of `bo_y`.
#[inline]
pub fn write_q4layer_step_params(buf: &mut [u8], rope_cs: &[f32; 128], pos: usize, skip_kv: i32) {
    assert_eq!(buf.len(), Q4LAYER_ACT_BYTES);
    assert!(pos < MAX_KV_CACHE_TOKENS);
    assert!(
        (0..Q4LAYER_ATTN_CHUNKS as i32).contains(&skip_kv),
        "skip_kv ({skip_kv}) out of range 0..{Q4LAYER_ATTN_CHUNKS}"
    );
    let active_chunks = Q4LAYER_ATTN_CHUNKS - skip_kv as usize;
    assert!(
        active_chunks * Q4LAYER_ATTN_CHUNK_T >= pos.max(1),
        "skip_kv ({skip_kv}) leaves fewer KV tokens than pos ({pos})"
    );
    write_le_f32_slice(buf, Q4LAYER_ROPE_CS, rope_cs);
    write_le_i32(buf, Q4LAYER_ATTN_N_OFF, pos as i32);
    write_le_i32(buf, Q4LAYER_ATTN_SKIP_KV_OFF, skip_kv);
}

/// Convert one KV head's `(k_head, v_head)` at `pos` to bf16 and write into
/// `q4layer`'s chunked KV region in `bo_y`, returning the byte offset of the
/// written K row (V sits `Q4LAYER_SILU_HALF * 4` bytes past it).
#[inline]
pub fn write_q4layer_kv_bf16(
    buf: &mut [u8],
    kvh: usize,
    pos: usize,
    k_head: &[f32],
    v_head: &[f32],
) -> usize {
    assert_eq!(buf.len(), Q4LAYER_ACT_BYTES);
    assert!(kvh < Q4LAYER_N_KV_HEADS);
    assert!(pos < MAX_KV_CACHE_TOKENS);
    assert_eq!(k_head.len(), Q4LAYER_HEAD_DIM);
    assert_eq!(v_head.len(), Q4LAYER_HEAD_DIM);
    let c = pos / Q4LAYER_ATTN_CHUNK_T;
    let rem = pos % Q4LAYER_ATTN_CHUNK_T;
    let chunk_byte_off =
        (Q4LAYER_KV_CACHE_BASE + (kvh * Q4LAYER_ATTN_CHUNKS + c) * Q4LAYER_TILE_F32) * 4;
    let k_byte_off = chunk_byte_off + rem * Q4LAYER_HEAD_DIM * 2;
    let v_byte_off = k_byte_off + Q4LAYER_SILU_HALF * 4;
    for d in 0..Q4LAYER_HEAD_DIM {
        let kb = super::repack::f32_to_bf16(k_head[d]).to_le_bytes();
        let vb = super::repack::f32_to_bf16(v_head[d]).to_le_bytes();
        buf[k_byte_off + d * 2..k_byte_off + d * 2 + 2].copy_from_slice(&kb);
        buf[v_byte_off + d * 2..v_byte_off + d * 2 + 2].copy_from_slice(&vb);
    }
    k_byte_off
}

/// Parse instruction bytes as little-endian 32-bit words, rejecting empty or
/// non-4-byte-aligned streams instead of silently dropping trailing bytes.
pub fn try_inst_words(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "XDNA instruction stream must be non-empty and a multiple of 4 bytes, got {} bytes",
            bytes.len()
        ));
    }
    let (chunks, rem) = bytes.as_chunks::<4>();
    debug_assert!(rem.is_empty());
    Ok(chunks.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Instruction bytes as little-endian words, as the shim uploads them.
///
/// Panics if `bytes` is empty or not a whole number of 32-bit words; use
/// [`try_inst_words`] for untrusted inputs.
pub fn inst_words(bytes: &[u8]) -> Vec<u32> {
    try_inst_words(bytes).expect("XDNA instruction stream must be non-empty and 4-byte aligned")
}

/// Return the top `k` `(token_idx, logit)` pairs sorted descending by logit,
/// canonicalizing `-0.0` to `+0.0` and breaking equal-logit ties by ascending
/// token index so greedy top-1 matches `cera::sampler::argmax` across finite
/// logits and all-`NaN` inputs. Non-`NaN` values rank ahead of `NaN`, and `NaN`
/// entries break ties by ascending token index.
pub fn top_k_logits(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &(usize, f32), b: &(usize, f32)| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)),
    };
    let canonical = |v: f32| if v == 0.0 { 0.0 } else { v };
    if k <= 64 && k < logits.len() {
        let mut best: Vec<(usize, f32)> = Vec::with_capacity(k);
        for (idx, &raw) in logits.iter().enumerate() {
            let candidate = (idx, canonical(raw));
            if best.len() == k && cmp(&candidate, &best[k - 1]) != std::cmp::Ordering::Less {
                continue;
            }
            let pos =
                best.partition_point(|entry| cmp(entry, &candidate) == std::cmp::Ordering::Less);
            if best.len() == k {
                best.pop();
            }
            best.insert(pos, candidate);
        }
        return best;
    }
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().map(canonical).enumerate().collect();
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, cmp);
        pairs.truncate(k);
        pairs.shrink_to_fit();
    }
    pairs.sort_by(cmp);
    pairs
}

/// Cosine similarity in `f64` between two non-empty, finite, non-zero logit slices.
pub fn logit_cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 || !na.is_finite() || !nb.is_finite() {
        return None;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    cos.is_finite().then(|| cos.clamp(-1.0, 1.0))
}

/// Verify teacher-forced parity summary metrics against Section 4 of
/// `docs/XDNA_PHASE0_RESULTS.md`.
pub fn check_parity_thresholds(
    target_k: usize,
    mean_overlap: f64,
    mean_cos: f64,
    min_cos: f64,
    max_cpu_margin: f32,
) -> Result<(), String> {
    let min_expected_overlap = (target_k as f64 - 1.0).max(1.0);
    if !mean_overlap.is_finite()
        || !mean_cos.is_finite()
        || !min_cos.is_finite()
        || !max_cpu_margin.is_finite()
        || mean_overlap < min_expected_overlap
        || mean_overlap > target_k as f64
        || mean_cos < PARITY_MIN_MEAN_COS
        || mean_cos > 1.0
        || min_cos < PARITY_MIN_STEP_COS
        || min_cos > 1.0
        || max_cpu_margin < 0.0
        || max_cpu_margin > PARITY_MAX_CPU_MARGIN
    {
        return Err(format!(
            "parity below threshold (mean_overlap={mean_overlap:.2} [min {min_expected_overlap:.1}], mean_cos={mean_cos:.6} [min {PARITY_MIN_MEAN_COS:.2}], min_cos={min_cos:.6} [min {PARITY_MIN_STEP_COS:.2}], max_cpu_margin={max_cpu_margin:.4} [max {PARITY_MAX_CPU_MARGIN:.1}])"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_inst_words_rejects_empty_and_unaligned_and_parses_embedded() {
        assert!(try_inst_words(&[]).is_err());
        assert!(try_inst_words(&[1, 2, 3]).is_err());
        assert!(try_inst_words(&[1, 2, 3, 4, 5]).is_err());
        assert_eq!(
            try_inst_words(&[0x01, 0x02, 0x03, 0x04]).unwrap(),
            vec![0x0403_0201]
        );
        for &(_, inst_bytes) in &Q4LAYER_KV_BUCKETS {
            assert!(!inst_words(inst_bytes).is_empty());
        }
        assert!(!inst_words(Q4HEAD_INSTS).is_empty());
    }

    #[test]
    fn q4layer_bo_y_framing_places_norms_step_params_and_chunked_kv() {
        let mut bo_y = vec![0xFFu8; Q4LAYER_ACT_BYTES];
        let attn_norm = vec![1.25f32; Q4LAYER_ATTN_NORM_W];
        let ffn_norm = vec![2.5f32; Q4LAYER_FFN_NORM_W];
        init_q4layer_bo_y_norms(&mut bo_y, &attn_norm, &ffn_norm);

        let mut got_attn = vec![0.0f32; Q4LAYER_ATTN_NORM_W];
        let mut got_ffn = vec![0.0f32; Q4LAYER_FFN_NORM_W];
        read_le_f32_slice(&bo_y, Q4LAYER_ATTN_NORM_W, &mut got_attn);
        read_le_f32_slice(
            &bo_y,
            4 * Q4LAYER_ACT_SLOT + Q4LAYER_FFN_NORM_W,
            &mut got_ffn,
        );
        assert_eq!(got_attn, attn_norm);
        assert_eq!(got_ffn, ffn_norm);

        let mut cos = [0.0f32; Q4LAYER_HEAD_DIM / 2];
        let mut sin = [0.0f32; Q4LAYER_HEAD_DIM / 2];
        for i in 0..Q4LAYER_HEAD_DIM / 2 {
            cos[i] = i as f32;
            sin[i] = -(i as f32);
        }
        let rope_cs = pack_q4layer_rope_cs(&cos, &sin);
        assert_eq!(&rope_cs[0..16], &cos[0..16]);
        assert_eq!(&rope_cs[16..32], &sin[0..16]);
        assert_eq!(&rope_cs[32..48], &cos[16..32]);
        assert_eq!(&rope_cs[48..64], &sin[16..32]);
        assert_eq!(&rope_cs[64..128], &rope_cs[0..64]);

        let buckets: Vec<usize> = Q4LAYER_KV_BUCKETS.iter().map(|&(c, _)| c).collect();
        let (bucket_idx, skip_kv) = select_q4layer_kv_bucket(&buckets, 540);
        assert_eq!((bucket_idx, skip_kv), (7, 1));
        write_q4layer_step_params(&mut bo_y, &rope_cs, 540, skip_kv);
        let n_off = Q4LAYER_ATTN_N_OFF * 4;
        let skip_off = Q4LAYER_ATTN_SKIP_KV_OFF * 4;
        assert_eq!(
            i32::from_le_bytes(bo_y[n_off..n_off + 4].try_into().unwrap()),
            540
        );
        assert_eq!(
            i32::from_le_bytes(bo_y[skip_off..skip_off + 4].try_into().unwrap()),
            1
        );

        let k_head = [1.0f32; Q4LAYER_HEAD_DIM];
        let v_head = [-2.0f32; Q4LAYER_HEAD_DIM];
        let k_off = write_q4layer_kv_bf16(&mut bo_y, 7, 647, &k_head, &v_head);
        let v_off = k_off + Q4LAYER_SILU_HALF * 4;
        assert!(v_off + Q4LAYER_HEAD_DIM * 2 <= Q4LAYER_ACT_BYTES);
        assert_eq!(u16::from_le_bytes([bo_y[k_off], bo_y[k_off + 1]]), 0x3F80);
        assert_eq!(u16::from_le_bytes([bo_y[v_off], bo_y[v_off + 1]]), 0xC000);
    }

    #[test]
    #[should_panic(expected = "byte offset overflow")]
    fn write_le_f32_slice_rejects_wrapping_word_index() {
        let mut buf = [0u8; 16];
        write_le_f32_slice(&mut buf, (usize::MAX / 4) + 1, &[1.0]);
    }

    #[test]
    #[should_panic(expected = "byte offset overflow")]
    fn read_le_f32_slice_rejects_wrapping_word_index() {
        let buf = [0u8; 16];
        let mut dst = [0.0f32; 1];
        read_le_f32_slice(&buf, (usize::MAX / 4) + 1, &mut dst);
    }

    #[test]
    #[should_panic(expected = "byte offset overflow")]
    fn write_le_i32_rejects_wrapping_word_index() {
        let mut buf = [0u8; 16];
        write_le_i32(&mut buf, (usize::MAX / 4) + 1, 7);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum bucket capacity")]
    fn select_q4layer_kv_bucket_panics_when_pos_exceeds_bucket_table() {
        let _ = select_q4layer_kv_bucket(&[1, 2], 200);
    }

    #[test]
    #[should_panic(expected = "leaves fewer KV tokens")]
    fn write_q4layer_step_params_rejects_undersized_active_chunks() {
        let mut bo_y = vec![0u8; Q4LAYER_ACT_BYTES];
        let rope_cs = [0.0f32; 128];
        write_q4layer_step_params(&mut bo_y, &rope_cs, 100, 8);
    }

    #[test]
    fn top_k_logits_ranks_non_nan_above_nan_and_bounds_small_k_alloc() {
        let nan_pos = f32::NAN;
        let nan_neg = -f32::NAN;
        let nan_payload = f32::from_bits(0x7fc0_1234);

        let mixed = [nan_pos, -1.0f32, nan_neg, 2.5f32, 2.5f32, nan_payload];
        let ranked = top_k_logits(&mixed, 6);
        assert_eq!(ranked[0], (3, 2.5));
        assert_eq!(ranked[1], (4, 2.5));
        assert_eq!(ranked[2], (1, -1.0));
        assert_eq!(ranked[3].0, 0);
        assert!(ranked[3].1.is_nan());
        assert_eq!(ranked[4].0, 2);
        assert!(ranked[4].1.is_nan());
        assert_eq!(ranked[5].0, 5);
        assert!(ranked[5].1.is_nan());
        assert_eq!(
            top_k_logits(&mixed, 1)[0].0,
            crate::sampler::argmax(&mixed) as usize
        );

        let all_nan = [nan_neg, nan_payload, nan_pos];
        let all_nan_top = top_k_logits(&all_nan, 2);
        assert_eq!(all_nan_top.len(), 2);
        assert_eq!(all_nan_top[0].0, 0);
        assert_eq!(all_nan_top[1].0, 1);
        assert_eq!(all_nan_top[0].0, crate::sampler::argmax(&all_nan) as usize);

        let large = vec![0.5f32; 256];
        let small_top = top_k_logits(&large, 4);
        assert_eq!(small_top.len(), 4);
        assert_eq!(small_top.capacity(), 4);

        let medium_top = top_k_logits(&large, 80);
        assert_eq!(medium_top.len(), 80);
        assert_eq!(medium_top.capacity(), 80);

        assert!(top_k_logits(&mixed, 0).is_empty());
        assert!(top_k_logits(&[], 4).is_empty());
    }

    #[test]
    fn parity_helpers_canonicalize_signed_zero_ties_and_match_certified_baseline() {
        let ranked = top_k_logits(&[-0.0, 0.0, 1.5, 1.5, -2.0], 4);
        assert_eq!(ranked, vec![(2, 1.5), (3, 1.5), (0, 0.0), (1, 0.0)]);
        assert_eq!(ranked[2].1.to_bits(), 0.0f32.to_bits());

        assert_eq!(logit_cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None);
        assert_eq!(logit_cosine_similarity(&[f32::NAN, 1.0], &[1.0, 1.0]), None);
        let cos = logit_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).expect("unit cos");
        assert!((cos - 1.0).abs() < 1e-12);
        let cos_over = logit_cosine_similarity(&[1e30, 1e30], &[1e30, 1e30]).expect("clamped");
        assert!((-1.0..=1.0).contains(&cos_over));

        // Certified Llama-3.2-1B-Instruct-Q4_0 512-step hardware metrics
        // (`docs/XDNA_PHASE0_RESULTS.md` Section 4).
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.979_014, 1.0658).is_ok());
        assert!(check_parity_thresholds(10, 8.95, 0.995_958, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 10.5, 0.995_958, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.989_000, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 1.01, 0.979_014, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.969_000, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 1.01, 1.0658).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.979_014, -0.1).is_err());
        assert!(check_parity_thresholds(10, 9.40, 0.995_958, 0.979_014, 2.05).is_err());

        for w in Q4LAYER_KV_BUCKETS.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "KV bucket chunk counts must strictly increase"
            );
        }
        assert_eq!(Q4LAYER_KV_BUCKETS.last().unwrap().0, Q4LAYER_ATTN_CHUNKS);
    }
}

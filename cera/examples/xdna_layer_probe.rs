//! Runs the fused XDNA layer kernel on weights packed by cera's own repack
//! and checks every stage's output slot against a host simulation of the
//! same full transformer layer chain (RMSNorms, GEMVs, RoPE, GQA attention,
//! SiLU*up, and residual adds).
//! This is backend milestone (a): one transformer layer, NPU vs host.
//!
//! Usage: `xdna_layer_probe [iters]`

use cera::backend::xdna::XdnaDevice;
use cera::backend::xdna::kernels::{Q4LAYER_INSTS, Q4LAYER_XCLBIN, inst_words};
use cera::backend::xdna::repack::{
    f32_to_bf16, interleave_worker_stages, pack_f32_stage, stage_bytes,
};
use std::process::ExitCode;

const ACT_SLOT: usize = 8192;
const N_SLOTS: usize = 8;
const TILE_F32: usize = 4608;
const SILU_HALF: usize = 2304;
/// Slot 0 is [hidden | RMSNorm weights | RoPE cos/sin | n_kv_cache]: the kernel
/// norms + quantizes the hidden state once for (q, k, v).
const NORM_W: usize = 2048;
const ROPE_CS: usize = 4096;
const ATTN_N_OFF: usize = ROPE_CS + 128;
const ATTN_OUT: usize = 2048;
const HEAD_DIM: usize = 64;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 8;
const ATTN_CHUNK_T: usize = 72;
const ATTN_CHUNKS: usize = 9;
const KV_CACHE_BASE: usize = N_SLOTS * ACT_SLOT;
const KV_HEAD_STRIDE: usize = ATTN_CHUNKS * TILE_F32;
const KV_SLOT: usize = N_KV_HEADS * KV_HEAD_STRIDE;
const ACT_TOTAL: usize = KV_CACHE_BASE + KV_SLOT;
const ATTN_CACHE_T: usize = 540;
const NORM_EPS: f32 = 1e-5;
/// (input slot, rows, cols, output slot) for a Llama-3.2-1B layer.
const STAGES: [(usize, usize, usize, usize); 7] = [
    (0, 2048, 2048, 1),
    (0, 512, 2048, 2),
    (0, 512, 2048, 3),
    (1, 2048, 2048, 4),
    (4, 8192, 2048, 5),
    (4, 8192, 2048, 6),
    (5, 2048, 8192, 7),
];

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct Rng(u32);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        ((self.0 >> 8) & 0xFFFF) as f32 / 32768.0 - 1.0
    }
}

/// Quantize one f32 block with the spike's Q4_0 rule; returns (q-8 values,
/// bf16 scale bits as the kernel will read them, back-converted to f32).
fn quantize_block(vals: &[f32]) -> ([i8; 32], f32) {
    let mut mx = 0.0f32;
    let mut amax = 0.0f32;
    for &v in vals {
        if v.abs() > amax {
            amax = v.abs();
            mx = v;
        }
    }
    let d = mx / -8.0;
    let stored = bf16_to_f32(f32_to_bf16(d));
    let mut q = [0i8; 32];
    if d != 0.0 {
        for (i, &v) in vals.iter().enumerate() {
            q[i] = ((v / d + 8.5) as i32).clamp(0, 15) as i8 - 8;
        }
    }
    (q, stored)
}

fn main() -> ExitCode {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(5);
    let device = match XdnaDevice::open(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("xdna unavailable: {e}");
            return ExitCode::FAILURE;
        }
    };
    let kernel = match device.load_kernel(Q4LAYER_XCLBIN, &inst_words(Q4LAYER_INSTS)) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("xdna unavailable: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Deterministic f32 weights per stage, packed by the Rust repack.
    let total_w: usize = STAGES
        .iter()
        .map(|&(_, m, k, _)| stage_bytes(m, k).unwrap())
        .sum();
    let mut bo_w = match kernel.alloc_bo(total_w, 3) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut stage_weights: Vec<Vec<f32>> = Vec::new();
    let mut stage_packed: Vec<Vec<u8>> = Vec::new();
    {
        let mut rng = Rng(2024);
        for &(_, m, k, _) in &STAGES {
            let weights: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
            let mut buf = vec![0u8; stage_bytes(m, k).unwrap()];
            if let Err(e) = pack_f32_stage(&mut buf, &weights, m, k) {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
            stage_packed.push(buf);
            stage_weights.push(weights);
        }
        let w = bo_w.as_mut_slice();
        let mut off = 0usize;
        off += interleave_worker_stages(
            &mut w[off..],
            &[&stage_packed[0], &stage_packed[1], &stage_packed[2]],
        )
        .unwrap();
        w[off..off + stage_packed[3].len()].copy_from_slice(&stage_packed[3]);
        off += stage_packed[3].len();
        off +=
            interleave_worker_stages(&mut w[off..], &[&stage_packed[4], &stage_packed[5]]).unwrap();
        w[off..off + stage_packed[6].len()].copy_from_slice(&stage_packed[6]);
    }
    if let Err(e) = bo_w.sync(true) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    let mut bo_y = match kernel.alloc_bo(ACT_TOTAL * 4, 4) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut hidden = vec![0.0f32; 2048];
    let mut rng = Rng(7);
    for v in &mut hidden {
        *v = rng.next_f32();
    }
    let norm_w: Vec<f32> = (0..NORM_W).map(|_| 1.0 + 0.5 * rng.next_f32()).collect();
    let ffn_norm_w: Vec<f32> = (0..NORM_W).map(|_| 1.0 + 0.5 * rng.next_f32()).collect();
    let mut rope_cos = [0.0f32; HEAD_DIM / 2];
    let mut rope_sin = [0.0f32; HEAD_DIM / 2];
    for j in 0..HEAD_DIM / 2 {
        let freq = 1.0 / 500_000.0f32.powf((2 * j) as f32 / HEAD_DIM as f32);
        let angle = (ATTN_CACHE_T as f32) * freq;
        rope_cos[j] = angle.cos();
        rope_sin[j] = angle.sin();
    }
    let mut rope_cs = [0.0f32; 128];
    for h in 0..2 {
        let base = h * 64;
        rope_cs[base..base + 16].copy_from_slice(&rope_cos[..16]);
        rope_cs[base + 16..base + 32].copy_from_slice(&rope_sin[..16]);
        rope_cs[base + 32..base + 48].copy_from_slice(&rope_cos[16..32]);
        rope_cs[base + 48..base + 64].copy_from_slice(&rope_sin[16..32]);
    }
    let kv_cache_len = N_KV_HEADS * ATTN_CACHE_T * HEAD_DIM;
    let k_cache: Vec<f32> = (0..kv_cache_len)
        .map(|_| bf16_to_f32(f32_to_bf16(rng.next_f32())))
        .collect();
    let v_cache: Vec<f32> = (0..kv_cache_len)
        .map(|_| bf16_to_f32(f32_to_bf16(rng.next_f32())))
        .collect();
    {
        let y = bo_y.as_mut_slice();
        y.fill(0);
        for (i, v) in hidden.iter().chain(norm_w.iter()).enumerate() {
            y[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in rope_cs.iter().enumerate() {
            let off = (ROPE_CS + i) * 4;
            y[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let n_off = ATTN_N_OFF * 4;
        y[n_off..n_off + 4].copy_from_slice(&(ATTN_CACHE_T as i32).to_le_bytes());
        for kvh in 0..N_KV_HEADS {
            let src_base = kvh * ATTN_CACHE_T * HEAD_DIM;
            for t in 0..ATTN_CACHE_T {
                let c = t / ATTN_CHUNK_T;
                let rem = t % ATTN_CHUNK_T;
                let chunk_byte_off = (KV_CACHE_BASE + (kvh * ATTN_CHUNKS + c) * TILE_F32) * 4;
                let k_byte_off = chunk_byte_off + rem * HEAD_DIM * 2;
                let v_byte_off = k_byte_off + SILU_HALF * 4;
                for d in 0..HEAD_DIM {
                    let kb = f32_to_bf16(k_cache[src_base + t * HEAD_DIM + d]).to_le_bytes();
                    let vb = f32_to_bf16(v_cache[src_base + t * HEAD_DIM + d]).to_le_bytes();
                    y[k_byte_off + d * 2..k_byte_off + d * 2 + 2].copy_from_slice(&kb);
                    y[v_byte_off + d * 2..v_byte_off + d * 2 + 2].copy_from_slice(&vb);
                }
            }
        }
        let ffn_off = 4 * ACT_SLOT + NORM_W;
        for (i, v) in ffn_norm_w.iter().enumerate() {
            let off = (ffn_off + i) * 4;
            y[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    if let Err(e) = bo_y.sync(true) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    let mut bo_y_next = match kernel.alloc_bo(ACT_TOTAL * 4, 5) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    bo_y_next.as_mut_slice().fill(0);
    if let Err(e) = bo_y_next.sync(true) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    let run = match kernel.create_run(&[&bo_w, &bo_y, &bo_y_next]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = run.start() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    match run.wait(0) {
        Ok(4) => {}
        Ok(state) => {
            eprintln!("run ended in state {state}, expected 4 (completed)");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    }
    if let Err(e) = bo_y.sync(false) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = bo_y_next.sync(false) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let mut npu_slots: Vec<f32> = bo_y
        .as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    for (i, c) in bo_y_next
        .as_slice()
        .as_chunks::<4>()
        .0
        .iter()
        .take(2048)
        .enumerate()
    {
        npu_slots[7 * ACT_SLOT + i] = f32::from_le_bytes(*c);
    }

    // Host simulation of the same stage chain.
    let mut slots = npu_slots.clone();
    slots[..2048].copy_from_slice(&hidden);
    let mut bad_total = 0usize;
    for (si, &(in_slot, m, k, out_slot)) in STAGES.iter().enumerate() {
        let in_base = in_slot * ACT_SLOT + if in_slot == 1 { ATTN_OUT } else { 0 };
        let mut input: Vec<f32> = slots[in_base..in_base + k].to_vec();
        if in_slot == 0 || in_slot == 4 {
            // RMSNorm folded into the kernel's quantize pass.
            let nw = if in_slot == 0 { &norm_w } else { &ffn_norm_w };
            let ss: f32 = input.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / k as f32 + NORM_EPS).sqrt();
            for (v, w) in input.iter_mut().zip(nw) {
                *v = *v * inv * w;
            }
        }
        // Q8_0-style activation quantization, round half away from zero.
        let nb = k / 32;
        let mut xq = vec![0i8; k];
        let mut xd = vec![0f32; nb];
        for b in 0..nb {
            let mut amax = 0.0f32;
            for j in 0..32 {
                amax = amax.max(input[b * 32 + j].abs());
            }
            let d = amax / 127.0;
            xd[b] = d;
            if d != 0.0 {
                for j in 0..32 {
                    xq[b * 32 + j] = (input[b * 32 + j] / d).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        let weights = &stage_weights[si];
        let res_slot = if out_slot == 4 {
            Some(0)
        } else if out_slot == 7 {
            Some(4)
        } else {
            None
        };
        let mut raw_ref = vec![0.0f64; m];
        let mut raw_mag = vec![0.0f64; m];
        for r in 0..m {
            let mut reference = 0.0f64;
            let mut mag = 0.0f64;
            for b in 0..nb {
                let (wq, wd) = quantize_block(&weights[r * k + b * 32..r * k + b * 32 + 32]);
                let mut dot = 0i32;
                for j in 0..32 {
                    dot += wq[j] as i32 * xq[b * 32 + j] as i32;
                }
                let term = dot as f64 * wd as f64 * xd[b] as f64;
                reference += term;
                mag += term.abs();
            }
            if let Some(rs) = res_slot {
                let res = slots[rs * ACT_SLOT + r] as f64;
                reference += res;
                mag += res.abs();
            }
            raw_ref[r] = reference;
            raw_mag[r] = mag;
        }
        if out_slot == 1 || out_slot == 2 {
            for h in 0..m / HEAD_DIM {
                for j in 0..HEAD_DIM / 2 {
                    let i0 = h * HEAD_DIM + 2 * j;
                    let i1 = i0 + 1;
                    let (x0, x1) = (raw_ref[i0], raw_ref[i1]);
                    let (c, s) = (rope_cos[j] as f64, rope_sin[j] as f64);
                    raw_ref[i0] = x0 * c - x1 * s;
                    raw_ref[i1] = x0 * s + x1 * c;
                    let m_pair = raw_mag[i0].max(raw_mag[i1]);
                    raw_mag[i0] = m_pair;
                    raw_mag[i1] = m_pair;
                }
            }
        }
        if out_slot == 5 {
            // Stage 4 (gate) stays on-core in gate_local; save host reference in slot 6
            // so Stage 5 (up) can fold SiLU(gate) * up and verify against slot 5.
            for r in 0..m {
                slots[6 * ACT_SLOT + r] = raw_ref[r] as f32;
            }
            println!("  stage {si} ({m}x{k} -> L1 gate_local): on-core (verified via stage 5)");
            continue;
        }
        let verify_slot = if out_slot == 6 { 5 } else { out_slot };
        if out_slot == 6 {
            for r in 0..m {
                let g = slots[6 * ACT_SLOT + r] as f64;
                let silu_g = g / (1.0 + (-g).exp());
                raw_ref[r] *= silu_g;
                raw_mag[r] *= silu_g.abs().max(1.0);
            }
        }
        let mut bad = 0usize;
        let mut worst = 0.0f64;
        for r in 0..m {
            let reference = raw_ref[r];
            let mag = raw_mag[r];
            slots[verify_slot * ACT_SLOT + r] = reference as f32;
            let got = npu_slots[verify_slot * ACT_SLOT + r] as f64;
            let err = (got - reference).abs() / if mag > 0.0 { mag } else { 1.0 };
            if err <= 1e-4 {
                worst = worst.max(err);
            } else {
                bad += 1;
            }
        }
        println!(
            "  stage {si} ({m}x{k} -> slot {verify_slot}): {bad} of {m} rows off (worst ok {worst:.2e}) {}",
            if bad == 0 { "PASS" } else { "FAIL" }
        );
        bad_total += bad;

        if out_slot == 3 {
            let mut attn_bad = 0usize;
            let mut attn_worst = 0.0f64;
            let n_tok = ATTN_CACHE_T + 1;
            let scale = 1.0f64 / (HEAD_DIM as f64).sqrt();
            for qh in 0..N_HEADS {
                let kvh = qh / (N_HEADS / N_KV_HEADS);
                let q_slice = &slots[ACT_SLOT + qh * HEAD_DIM..ACT_SLOT + (qh + 1) * HEAD_DIM];
                let mut scores = vec![0.0f64; n_tok];
                let mut s_mags = vec![0.0f64; n_tok];
                let mut max_s = f64::NEG_INFINITY;
                for t in 0..n_tok {
                    let k_slice = if t < ATTN_CACHE_T {
                        let base = (kvh * ATTN_CACHE_T + t) * HEAD_DIM;
                        &k_cache[base..base + HEAD_DIM]
                    } else {
                        let base = 2 * ACT_SLOT + kvh * HEAD_DIM;
                        &slots[base..base + HEAD_DIM]
                    };
                    let mut dot = 0.0f64;
                    let mut s_mag = 0.0f64;
                    for (&q, &kv) in q_slice.iter().zip(k_slice) {
                        let prod = q as f64 * kv as f64;
                        dot += prod;
                        s_mag += prod.abs();
                    }
                    scores[t] = dot * scale;
                    s_mags[t] = s_mag * scale;
                    max_s = max_s.max(scores[t]);
                }
                let mut weights = vec![0.0f64; n_tok];
                let mut sum_w = 0.0f64;
                for t in 0..n_tok {
                    weights[t] = (scores[t] - max_s).exp();
                    sum_w += weights[t];
                }
                for d in 0..HEAD_DIM {
                    let mut acc = 0.0f64;
                    let mut mag = 0.0f64;
                    for t in 0..n_tok {
                        let v_val = if t < ATTN_CACHE_T {
                            v_cache[(kvh * ATTN_CACHE_T + t) * HEAD_DIM + d] as f64
                        } else {
                            slots[3 * ACT_SLOT + kvh * HEAD_DIM + d] as f64
                        };
                        let w = weights[t] / sum_w;
                        acc += w * v_val;
                        mag += w * v_val.abs() * (1.0 + s_mags[t]);
                    }
                    let idx = ACT_SLOT + ATTN_OUT + qh * HEAD_DIM + d;
                    slots[idx] = acc as f32;
                    let got = npu_slots[idx] as f64;
                    let err = (got - acc).abs() / if mag > 0.0 { mag } else { 1.0 };
                    if err <= 1e-4 {
                        attn_worst = attn_worst.max(err);
                    } else {
                        attn_bad += 1;
                    }
                }
            }
            println!(
                "  attn ({N_HEADS} heads, {n_tok} kv tokens -> slot 1+{ATTN_OUT}): {attn_bad} of 2048 values off (worst ok {attn_worst:.2e}) {}",
                if attn_bad == 0 { "PASS" } else { "FAIL" }
            );
            bad_total += attn_bad;
        }
    }
    if bad_total > 0 {
        eprintln!("xdna layer probe: FAIL ({bad_total} rows off)");
        return ExitCode::FAILURE;
    }

    // Timing: fresh start+wait per run, first three are warmup.
    let mut times = Vec::new();
    for i in 0..3 + iters {
        let t0 = std::time::Instant::now();
        if run.start().is_err() || run.wait(0).is_err() {
            eprintln!("timed run failed");
            return ExitCode::FAILURE;
        }
        if i >= 3 {
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    println!(
        "fused layer: p50 {:.1} us over {} runs; projection 16 layers + head (2816 us) = {:.2} ms/token -> {:.1} tok/s",
        p50,
        times.len(),
        (16.0 * p50 + 2816.0) / 1000.0,
        1e6 / (16.0 * p50 + 2816.0)
    );
    println!("xdna layer probe: PASS");
    ExitCode::SUCCESS
}

//! Teacher-forced parity oracle for `XdnaLlamaModel` vs CPU `LlamaModel`
//! on `Llama-3.2-1B-Instruct-Q4_0.gguf` (Section 4 of `docs/XDNA_PHASE0_RESULTS.md`).
//!
//! Usage:
//!   `cargo run --release -p cera --features xdna --example xdna_parity_oracle -- [model.gguf] [n_positions]`

use std::path::PathBuf;
use std::process::ExitCode;

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::Model;
use cera::model::llama::LlamaModel;
use cera::model::xdna_llama::XdnaLlamaModel;

fn top_k_indices(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let default_model =
        r"C:\Users\iamst\.leap\models\Llama-3.2-1B-Instruct-Q4_0\Llama-3.2-1B-Instruct-Q4_0.gguf";
    let model_path = PathBuf::from(args.get(1).map(String::as_str).unwrap_or(default_model));
    let n_forced: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);

    println!("Loading GGUF: {}", model_path.display());
    let gguf = match GgufFile::open(&model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open GGUF: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ctx_len = 1024;
    let cpu_model =
        match LlamaModel::from_gguf_with_id(gguf.clone(), ctx_len, "cpu-oracle".to_string()) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load CPU model: {e}");
                return ExitCode::FAILURE;
            }
        };
    let npu_model = match XdnaLlamaModel::from_gguf_with_id(gguf, ctx_len, "npu-oracle".to_string())
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load XDNA NPU model: {e}");
            return ExitCode::FAILURE;
        }
    };

    // 16-token Llama-3 instruct prompt prefix so 16 + 512 = 528 <= 540 positions
    // all run through the NPU decode path.
    let prompt_tokens: [u32; 16] = [
        128000, // <|begin_of_text|>
        128006, // <|start_header_id|>
        9125,   // system
        128007, // <|end_header_id|>
        271,    // \n\n
        38766,  // Cutting
        1303,   //  Knowledge
        33025,  //  Date
        2696,   // :
        25,     // :
        6790,   //  December
        220,    //
        2366,   // 2023
        18,     // 3
        198,    // \n
        128009, // <|eot_id|>
    ];

    let mut cpu_state = InferenceState::from_config(cpu_model.config()).unwrap();
    let mut npu_state = InferenceState::from_config(npu_model.config()).unwrap();

    let cpu_prefill_logits = cpu_model.forward_prefill(&prompt_tokens, 0, &mut cpu_state);
    let npu_prefill_logits = npu_model.forward_prefill(&prompt_tokens, 0, &mut npu_state);
    assert_eq!(
        cpu_prefill_logits, npu_prefill_logits,
        "CPU and NPU prefill logits must be bit-identical (both run CPU prefill)"
    );

    let mut next_tok = top_k_indices(&cpu_prefill_logits, 1)[0].0 as u32;
    let mut top1_exact = 0usize;
    let mut top1_disagreements: Vec<(usize, u32, u32, f32, f32)> = Vec::new();
    let mut top10_overlaps: Vec<usize> = Vec::with_capacity(n_forced);
    let mut cosines: Vec<f64> = Vec::with_capacity(n_forced);

    for step in 0..n_forced {
        let pos = prompt_tokens.len() + step;
        let in_tok = [next_tok];
        let cpu_logits = cpu_model.forward(&in_tok, pos, &mut cpu_state);
        let npu_logits = npu_model.forward(&in_tok, pos, &mut npu_state);

        if npu_logits.iter().any(|x| !x.is_finite()) {
            eprintln!("FAIL: non-finite logit at step {step} (pos {pos})");
            return ExitCode::FAILURE;
        }

        let cpu_top10 = top_k_indices(&cpu_logits, 10);
        let npu_top10 = top_k_indices(&npu_logits, 10);

        let cpu_top1 = cpu_top10[0].0 as u32;
        let npu_top1 = npu_top10[0].0 as u32;
        let cpu_margin = cpu_top10[0].1 - cpu_top10[1].1;

        if cpu_top1 == npu_top1 {
            top1_exact += 1;
        } else {
            let npu_logit_for_cpu_top1 = npu_logits[cpu_top1 as usize];
            let npu_top1_logit = npu_top10[0].1;
            top1_disagreements.push((
                pos,
                cpu_top1,
                npu_top1,
                cpu_margin,
                npu_top1_logit - npu_logit_for_cpu_top1,
            ));
        }

        let overlap = cpu_top10
            .iter()
            .filter(|&&(idx, _)| npu_top10.iter().any(|&(nidx, _)| nidx == idx))
            .count();
        top10_overlaps.push(overlap);
        cosines.push(cosine_sim(&cpu_logits, &npu_logits));

        // Teacher forcing: advance both with CPU's greedy token.
        next_tok = cpu_top1;
    }

    let mean_overlap: f64 = top10_overlaps.iter().map(|&x| x as f64).sum::<f64>() / n_forced as f64;
    let min_overlap = *top10_overlaps.iter().min().unwrap_or(&0);
    let ge9_count = top10_overlaps.iter().filter(|&&x| x >= 9).count();
    let mean_cos: f64 = cosines.iter().sum::<f64>() / n_forced as f64;
    let min_cos: f64 = cosines.iter().copied().fold(f64::INFINITY, f64::min);

    println!(
        "=== Teacher-Forced Parity ({n_forced} positions, prompt_len={}) ===",
        prompt_tokens.len()
    );
    println!(
        "  top-1 exact agreement: {top1_exact}/{n_forced} ({:.2}%)",
        100.0 * top1_exact as f64 / n_forced as f64
    );
    println!(
        "  top-10 overlap: mean {:.2}/10, min {min_overlap}/10, >=9/10 at {ge9_count}/{n_forced} ({:.2}%) positions",
        mean_overlap,
        100.0 * ge9_count as f64 / n_forced as f64
    );
    println!(
        "  logit cosine similarity: mean {:.6}, min {:.6}",
        mean_cos, min_cos
    );
    if !top1_disagreements.is_empty() {
        let max_cpu_margin = top1_disagreements
            .iter()
            .map(|x| x.3)
            .fold(0.0f32, f32::max);
        println!(
            "  top-1 disagreements ({}): max CPU top-2 margin = {:.4}",
            top1_disagreements.len(),
            max_cpu_margin
        );
        for &(pos, ct, nt, cm, nm) in top1_disagreements.iter().take(15) {
            println!(
                "    pos {pos}: cpu_top1={ct} vs npu_top1={nt} | cpu_top2_margin={cm:.4}, npu_diff={nm:.4}"
            );
        }
    }

    ExitCode::SUCCESS
}

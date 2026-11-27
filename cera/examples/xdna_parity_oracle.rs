//! Teacher-forced parity oracle for an XDNA NPU model vs the CPU model on the same GGUF:
//! `Llama-3.2-1B-Instruct-Q4_0.gguf` (Section 4 of `docs/XDNA_PHASE0_RESULTS.md`), or an
//! LFM2 / LFM2.5 / LFM3 Q4_0 GGUF.
//!
//! Usage:
//!   `cargo run --release -p cera --features xdna --example xdna_parity_oracle -- <model.gguf> [n_positions]`

use std::path::PathBuf;
use std::process::ExitCode;

use cera::backend::xdna::kernels::{
    MAX_KV_CACHE_TOKENS, ParityTally, check_parity_thresholds, logit_cosine_similarity,
    top_k_logits,
};
use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::tokenizer::BpeTokenizer;

const PROMPT_LEN: usize = 16;
/// LFM chat-format counterpart of the 16-token Llama-3 system-turn prompt.
const LFM_PROMPT: &str =
    "<|startoftext|><|im_start|>system\nCutting Knowledge Date: December 2023\n<|im_end|>\n";

fn print_usage() {
    println!(
        "Usage: xdna_parity_oracle <model.gguf> [n_positions] [prompt_len]\n  <model.gguf>   Path to a Llama-3.2-1B, LFM2/LFM2.5 or LFM3 Q4_0 GGUF\n  [n_positions]  Teacher-forced decode steps (default: 512)\n  [prompt_len]   Prompt length in 16..={} (default: 16)",
        MAX_KV_CACHE_TOKENS - 1
    );
}

fn parse_cli(args: &[String]) -> Result<Option<(PathBuf, usize, usize)>, String> {
    if args.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        return Ok(None);
    }
    if args.len() < 2 || args.len() > 4 {
        return Err(
            "expected `<model.gguf> [n_positions] [prompt_len]` (pass --help for usage)"
                .to_string(),
        );
    }
    let raw_model = &args[1];
    if raw_model.starts_with('-') {
        return Err(format!(
            "unknown flag `{raw_model}` (pass --help for usage)"
        ));
    }
    let n_forced = if let Some(raw_n) = args.get(2) {
        if raw_n.starts_with('-') {
            return Err(format!("unknown flag `{raw_n}` (pass --help for usage)"));
        }
        raw_n
            .parse::<usize>()
            .map_err(|_| format!("invalid n_positions `{raw_n}`: expected positive integer"))?
    } else {
        512
    };
    let prompt_len = if let Some(raw_p) = args.get(3) {
        if raw_p.starts_with('-') {
            return Err(format!("unknown flag `{raw_p}` (pass --help for usage)"));
        }
        raw_p
            .parse::<usize>()
            .map_err(|_| format!("invalid prompt_len `{raw_p}`: expected integer >= 16"))?
    } else {
        PROMPT_LEN
    };
    if !(16..MAX_KV_CACHE_TOKENS).contains(&prompt_len) {
        return Err(format!(
            "prompt_len must be in 16..={}, got {prompt_len}",
            MAX_KV_CACHE_TOKENS - 1
        ));
    }
    let max_forced = MAX_KV_CACHE_TOKENS - prompt_len;
    if n_forced == 0 || n_forced > max_forced {
        return Err(format!(
            "n_positions must be in 1..={max_forced} so prompt ({prompt_len}) + n_positions <= {MAX_KV_CACHE_TOKENS}, got {n_forced}"
        ));
    }
    Ok(Some((PathBuf::from(raw_model), n_forced, prompt_len)))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (model_path, n_forced, prompt_len) = match parse_cli(&args) {
        Ok(Some(v)) => v,
        Ok(None) => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("error: {msg}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    println!("Loading GGUF: {}", model_path.display());
    let gguf = match GgufFile::open(&model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open GGUF: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    let ctx_len = (prompt_len + n_forced + 64).max(1024);
    let is_llama = gguf.get_str("general.architecture") == Some("llama");
    let cpu_model = match cera::model::load_model(gguf.clone(), None, ctx_len) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load CPU model: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let npu_model = match cera::model::load_model_xdna(gguf.clone(), None, ctx_len) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load XDNA NPU model: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    // Repeat 16-token Llama-3 instruct prompt prefix (or `LFM_PROMPT`) up to `prompt_len` so
    // NPU prefill and decode can both be tested at arbitrary prompt lengths.
    let llama_prompt: [u32; PROMPT_LEN] = [
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
    let base_prompt = if is_llama {
        llama_prompt.to_vec()
    } else {
        match BpeTokenizer::from_gguf(&gguf) {
            Ok(tok) => tok.encode(LFM_PROMPT),
            Err(e) => {
                eprintln!("Failed to load tokenizer: {e:#}");
                return ExitCode::FAILURE;
            }
        }
    };
    let prompt_tokens: Vec<u32> = (0..prompt_len)
        .map(|i| {
            if i == 0 {
                base_prompt[0]
            } else {
                base_prompt[1 + ((i - 1) % (base_prompt.len() - 1))]
            }
        })
        .collect();

    let vocab_size = cpu_model.config().vocab_size;
    if vocab_size < 2 {
        eprintln!("Invalid model vocab_size {vocab_size}: expected at least 2");
        return ExitCode::FAILURE;
    }
    if let Some(&bad_tok) = prompt_tokens.iter().find(|&&t| (t as usize) >= vocab_size) {
        eprintln!(
            "Prompt token {bad_tok} exceeds model vocab_size {vocab_size}; expected Llama-3 tokenizer"
        );
        return ExitCode::FAILURE;
    }

    let mut cpu_state = match InferenceState::from_config(cpu_model.config()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to allocate CPU InferenceState: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut npu_state = match InferenceState::from_config(npu_model.config()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to allocate NPU InferenceState: {e}");
            return ExitCode::FAILURE;
        }
    };

    let t_cpu_pf0 = std::time::Instant::now();
    let cpu_prefill_logits = cpu_model.forward_prefill(&prompt_tokens, 0, &mut cpu_state);
    let cpu_pf_secs = t_cpu_pf0.elapsed().as_secs_f64();
    if let Some(err) = cpu_model.take_decode_error() {
        eprintln!("FAIL: CPU prefill fault: {err}");
        return ExitCode::FAILURE;
    }
    let t_npu_pf0 = std::time::Instant::now();
    let npu_prefill_logits = npu_model.forward_prefill(&prompt_tokens, 0, &mut npu_state);
    let npu_pf_secs = t_npu_pf0.elapsed().as_secs_f64();
    if let Some(err) = npu_model.take_decode_error() {
        eprintln!("FAIL: XDNA prefill fault: {err}");
        return ExitCode::FAILURE;
    }
    if cpu_prefill_logits.len() != npu_prefill_logits.len()
        || cpu_prefill_logits.iter().any(|x| !x.is_finite())
        || npu_prefill_logits.iter().any(|x| !x.is_finite())
    {
        eprintln!("FAIL: degenerate or non-finite prefill logits");
        return ExitCode::FAILURE;
    }
    let Some(prefill_cos) = logit_cosine_similarity(&cpu_prefill_logits, &npu_prefill_logits)
    else {
        eprintln!("FAIL: zero-norm prefill logits");
        return ExitCode::FAILURE;
    };
    let cpu_pf_top10 = top_k_logits(&cpu_prefill_logits, 10);
    let npu_pf_top10 = top_k_logits(&npu_prefill_logits, 10);
    let pf_overlap = cpu_pf_top10
        .iter()
        .filter(|&&(idx, _)| npu_pf_top10.iter().any(|&(nidx, _)| nidx == idx))
        .count();
    println!(
        "=== Prefill Parity (prompt_len={}) ===\n  top-1: cpu={} npu={} (match={}) | top-10 overlap: {pf_overlap}/10 | logit cos: {prefill_cos:.6}\n  speed: cpu={:.2} tok/s ({:.1} ms) | npu={:.2} tok/s ({:.1} ms)",
        prompt_tokens.len(),
        cpu_pf_top10[0].0,
        npu_pf_top10[0].0,
        cpu_pf_top10[0].0 == npu_pf_top10[0].0,
        prompt_tokens.len() as f64 / cpu_pf_secs,
        cpu_pf_secs * 1e3,
        prompt_tokens.len() as f64 / npu_pf_secs,
        npu_pf_secs * 1e3,
    );
    if prefill_cos < cera::backend::xdna::kernels::PARITY_MIN_STEP_COS {
        eprintln!(
            "FAIL: prefill logit cosine {prefill_cos:.6} below threshold {}",
            cera::backend::xdna::kernels::PARITY_MIN_STEP_COS
        );
        return ExitCode::FAILURE;
    }

    let mut next_tok = top_k_logits(&cpu_prefill_logits, 1)[0].0 as u32;
    let mut top1_exact = 0usize;
    let mut top1_disagreements: Vec<(usize, u32, u32, f32, f32)> = Vec::new();
    let mut top10_overlaps: Vec<usize> = Vec::with_capacity(n_forced);
    let mut cosines: Vec<f64> = Vec::with_capacity(n_forced);
    let mut tally = ParityTally::default();

    let mut cpu_dec_secs = 0.0f64;
    let mut npu_dec_secs = 0.0f64;

    for step in 0..n_forced {
        let pos = prompt_tokens.len() + step;
        let in_tok = [next_tok];
        let tc0 = std::time::Instant::now();
        let cpu_logits = cpu_model.forward(&in_tok, pos, &mut cpu_state);
        cpu_dec_secs += tc0.elapsed().as_secs_f64();
        if let Some(err) = cpu_model.take_decode_error() {
            eprintln!("FAIL: CPU decode fault at step {step} (pos {pos}): {err}");
            return ExitCode::FAILURE;
        }
        let tn0 = std::time::Instant::now();
        let npu_logits = npu_model.forward(&in_tok, pos, &mut npu_state);
        npu_dec_secs += tn0.elapsed().as_secs_f64();
        if let Some(err) = npu_model.take_decode_error() {
            eprintln!("FAIL: XDNA decode fault at step {step} (pos {pos}): {err}");
            return ExitCode::FAILURE;
        }

        if cpu_logits.iter().any(|x| !x.is_finite()) || npu_logits.iter().any(|x| !x.is_finite()) {
            eprintln!("FAIL: non-finite logit at step {step} (pos {pos})");
            return ExitCode::FAILURE;
        }

        let cpu_top10 = top_k_logits(&cpu_logits, 10);
        let npu_top10 = top_k_logits(&npu_logits, 10);

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
        let Some(cos) = logit_cosine_similarity(&cpu_logits, &npu_logits) else {
            eprintln!("FAIL: degenerate or zero-norm logit vector at step {step} (pos {pos})");
            return ExitCode::FAILURE;
        };
        cosines.push(cos);
        tally.record(pos, &cpu_logits, &npu_logits);

        // Teacher forcing: advance both with CPU's greedy token.
        next_tok = cpu_top1;
    }

    let target_k = 10.min(vocab_size);
    let mean_overlap: f64 = top10_overlaps.iter().map(|&x| x as f64).sum::<f64>() / n_forced as f64;
    let min_overlap = *top10_overlaps.iter().min().unwrap_or(&0);
    let ge9_count = top10_overlaps.iter().filter(|&&x| x >= 9).count();
    let mean_cos: f64 = cosines.iter().sum::<f64>() / n_forced as f64;
    let min_cos: f64 = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let max_cpu_margin = top1_disagreements
        .iter()
        .map(|x| x.3)
        .fold(0.0f32, f32::max);

    println!(
        "=== Teacher-Forced Parity ({n_forced} positions, prompt_len={}) ===",
        prompt_tokens.len()
    );
    println!(
        "  top-1 exact agreement: {top1_exact}/{n_forced} ({:.2}%)",
        100.0 * top1_exact as f64 / n_forced as f64
    );
    println!(
        "  top-{target_k} overlap: mean {:.2}/{target_k}, min {min_overlap}/{target_k}, >=9/{target_k} at {ge9_count}/{n_forced} ({:.2}%) positions",
        mean_overlap,
        100.0 * ge9_count as f64 / n_forced as f64
    );
    println!(
        "  logit cosine similarity: mean {:.6}, min {:.6}\n  decode speed: cpu={:.2} tok/s ({:.1} ms) | npu={:.2} tok/s ({:.1} ms)",
        mean_cos,
        min_cos,
        n_forced as f64 / cpu_dec_secs,
        cpu_dec_secs * 1e3,
        n_forced as f64 / npu_dec_secs,
        npu_dec_secs * 1e3,
    );
    println!("  {tally}");
    if !top1_disagreements.is_empty() {
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

    if let Err(msg) =
        check_parity_thresholds(target_k, mean_overlap, mean_cos, min_cos, max_cpu_margin)
    {
        eprintln!("FAIL: {msg}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

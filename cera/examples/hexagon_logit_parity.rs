//! How close are the NPU's logits to the CPU's, step by step?
//!
//! Loads the model on both backends, prefills the same prompt, lets the CPU decode greedily and
//! teacher-forces its tokens into the NPU, so every step compares the two on identical inputs
//! (text comparison alone drifts as soon as one near-tie flips). Per step it prints the cosine
//! between the two logit vectors, whether the argmax agrees, and the largest absolute difference;
//! the summary gives the minimum and mean cosine and the argmax agreement. The exit status is 1
//! when any step's cosine is not finite (the NPU returned NaN or all-zero logits).
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_logit_parity --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_logit_parity /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && ADSP_LIBRARY_PATH=$PWD ./hexagon_logit_parity model.gguf 32'
//! # a long context: STEPS, PROMPT (any text), CONTEXT size
//! ./hexagon_logit_parity model.gguf 16 "$(yes 'The quick brown fox jumps over the lazy dog.' | head -130)" 2048
//! ```
//!
//! Needs the skel next to the binary (see `skels/SOURCE.md`) when run from `adb shell`.

fn main() {
    use cera::{BackendPreference, CeraEngine, EngineConfig, SessionConfig};

    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: hexagon_logit_parity MODEL [STEPS] [PROMPT] [CONTEXT]");
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let prompt_text = args
        .next()
        .unwrap_or_else(|| "Explain in three sentences why the sky is blue.".into());

    let context_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let open = |backend| {
        CeraEngine::from_path(
            &model,
            EngineConfig {
                backend,
                context_size,
                ..EngineConfig::default()
            },
        )
        .expect("load model")
    };
    let cpu_engine = open(BackendPreference::Cpu);
    let npu_engine = open(BackendPreference::Hexagon);
    let mut cpu = cpu_engine
        .new_session(SessionConfig::default())
        .expect("cpu session");
    let mut npu = npu_engine
        .new_session(SessionConfig::default())
        .expect("npu session");
    let prompt = cpu.tokenizer().encode(&prompt_text);

    let argmax = |l: &[f32]| {
        l.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0
    };
    let cosine = |a: &[f32], b: &[f32]| {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (f64::from(x), f64::from(y));
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt())
    };
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    cpu.append_tokens(&prompt).expect("cpu prefill");
    npu.append_tokens(&prompt).expect("npu prefill");
    let mut cosines = Vec::new();
    let mut agree = 0usize;
    for step in 0..=steps {
        let c = cpu.last_logits().expect("cpu logits").to_vec();
        let n = npu.last_logits().expect("npu logits").to_vec();
        let (cos, same) = (cosine(&c, &n), argmax(&c) == argmax(&n));
        println!(
            "step {step:3}{}: cosine {cos:.6}  argmax {}  max|diff| {:.3e}",
            if step == 0 { " (prefill)" } else { "" },
            if same { "agree   " } else { "DIFFER  " },
            max_diff(&c, &n)
        );
        cosines.push(cos);
        agree += usize::from(same);
        if step == steps {
            break;
        }
        let next = argmax(&c) as u32;
        cpu.append_tokens(&[next]).expect("cpu decode");
        npu.append_tokens(&[next]).expect("npu decode");
    }
    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
    // `f64::min` drops a NaN, so an NPU that returns dead logits (a NaN cosine every step) would
    // print `cosine min inf`; carry the NaN through and fail the run instead
    let non_finite = cosines.iter().filter(|c| !c.is_finite()).count();
    let min = cosines.iter().cloned().fold(f64::INFINITY, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    println!(
        "{} prompt tokens, {} decode steps: cosine min {min:.6} mean {mean:.6}; argmax agrees on {agree}/{}",
        prompt.len(),
        steps,
        cosines.len()
    );
    if non_finite > 0 {
        eprintln!(
            "{non_finite} of {} steps have a non-finite cosine",
            cosines.len()
        );
        std::process::exit(1);
    }
}

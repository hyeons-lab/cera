//! Is NPU decode reproducible, and if not, where does it first differ?
//!
//! Runs the same prompt and the same continuation tokens through a model many
//! times, comparing every step's logits bit for bit with the first run, and
//! reports how often and at which step a run departs. Greedy decode on the
//! NPU was seen to give different text on identical input (the CPU never
//! does), and the knobs that serialize the decode batch (`CERA_HEXAGON_STEP`,
//! `CERA_HEXAGON_BARRIERS`) hid it, so this is the tool for bisecting it.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_decode_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_decode_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && ADSP_LIBRARY_PATH=$PWD ./hexagon_decode_probe model.gguf 64 40'
//! ```
//!
//! Arguments: the model, the number of decode steps (default 64) and the number
//! of repetitions (default 40), `cpu` as a fourth argument for the CPU control
//! and `greedy` as a fifth to run whole `generate` calls (the CLI's path, with
//! the DSP's argmax) and compare their token sequences instead, and an optional
//! sixth argument, the prompt text (default: a short story prompt).

fn main() {
    use cera::{BackendPreference, CeraEngine, EngineConfig, SessionConfig};

    let mut args = std::env::args().skip(1);
    let model = args.next().expect(
        "usage: hexagon_decode_probe MODEL [STEPS] [REPS] [cpu|hexagon] [logits|greedy] [PROMPT]",
    );
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40)
        .max(2);
    let backend = match args.next().as_deref() {
        Some("cpu") => BackendPreference::Cpu,
        _ => BackendPreference::Hexagon,
    };
    let greedy = args.next().as_deref() == Some("greedy");
    let prompt_text = args
        .next()
        .unwrap_or_else(|| "Write a short story about a pug dog who learns to fly. The pug".into());
    let engine = CeraEngine::from_path(
        &model,
        EngineConfig {
            backend,
            context_size: 1024,
            ..EngineConfig::default()
        },
    )
    .expect("load model");
    let mut session = engine
        .new_session(SessionConfig::default())
        .expect("session");
    let prompt = session.tokenizer().encode(&prompt_text);

    if greedy {
        // The path the CLI and apps take: the first token from the prefill
        // logits, every later one from `forward_greedy` (argmax on the DSP).
        struct Collect(Vec<u32>);
        impl cera::ModalitySink for Collect {
            fn on_text_tokens(&mut self, tokens: &[u32]) {
                self.0.extend_from_slice(tokens);
            }
            fn on_done(&mut self, _: cera::FinishReason) {}
        }
        let mut runs: Vec<Vec<u32>> = Vec::new();
        for _ in 0..reps {
            session.reset().expect("reset");
            session.append_tokens(&prompt).expect("prefill");
            let mut sink = Collect(Vec::new());
            session
                .generate(
                    &cera::GenerateOpts {
                        max_tokens: steps as u32,
                        temperature: 0.0,
                        ignore_eos: true,
                        ..cera::GenerateOpts::default()
                    },
                    &mut sink,
                )
                .expect("generate");
            runs.push(sink.0);
        }
        let mut distinct: Vec<(&Vec<u32>, usize)> = Vec::new();
        for r in &runs {
            match distinct.iter_mut().find(|(d, _)| *d == r) {
                Some((_, n)) => *n += 1,
                None => distinct.push((r, 1)),
            }
        }
        distinct.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let top = distinct[0].0;
        let fingerprint = top.iter().fold(1469598103934665603u64, |h, &t| {
            (h ^ u64::from(t)).wrapping_mul(1099511628211)
        });
        println!(
            "majority sequence fingerprint {fingerprint:016x} ({} prompt tokens)",
            prompt.len()
        );
        println!(
            "{} greedy runs of {} tokens: {} distinct sequences (runs per sequence: {:?})",
            reps,
            steps,
            distinct.len(),
            distinct.iter().map(|(_, n)| *n).collect::<Vec<_>>()
        );
        for (d, n) in distinct.iter().skip(1) {
            let at = d.iter().zip(top.iter()).position(|(a, b)| a != b);
            println!("  {n} run(s) leave the majority at token {at:?}");
        }
        return;
    }

    // Run 0 decodes greedily and fixes the continuation; every later run is
    // teacher-forced on it, so their steps stay comparable after a mismatch.
    let argmax = |l: &[f32]| {
        l.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0 as u32
    };
    session.reset().expect("reset");
    session.append_tokens(&prompt).expect("prefill");
    let mut reference: Vec<Vec<f32>> = vec![session.last_logits().expect("logits").to_vec()];
    let mut tokens = Vec::new();
    for _ in 0..steps {
        let next = argmax(reference.last().unwrap());
        tokens.push(next);
        session.append_tokens(&[next]).expect("decode");
        reference.push(session.last_logits().expect("logits").to_vec());
    }

    // One teacher-forced pass over the fixed continuation.
    let replay = |session: &mut cera::Session| -> Vec<Vec<f32>> {
        session.reset().expect("reset");
        session.append_tokens(&prompt).expect("prefill");
        let mut out = vec![session.last_logits().expect("logits").to_vec()];
        for &t in &tokens {
            session.append_tokens(&[t]).expect("decode");
            out.push(session.last_logits().expect("logits").to_vec());
        }
        out
    };
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let first_mismatch = |run: &[Vec<f32>], base: &[Vec<f32>]| {
        run.iter()
            .zip(base)
            .position(|(a, b)| a != b)
            .map(|step| (step, diff(&run[step], &base[step])))
    };

    // Run 1 is the first replay and the reference for the rest: run 0 decoded
    // greedily and built the decode template on its way, so it may legitimately
    // differ from a replay, which is a different question from run-to-run noise.
    let base = replay(&mut session);
    match first_mismatch(&reference, &base) {
        None => println!("cold greedy run == first replay (bit-identical)"),
        Some((step, d)) => println!(
            "cold greedy run differs from the first replay at step {step} (max |diff| {d:.3e})"
        ),
    }
    let mut first_bad = vec![0usize; steps + 2];
    let mut differing_runs = 0;
    let mut worst = 0.0f32;
    let fingerprint = |run: &[Vec<f32>]| {
        let mut h: u64 = 1469598103934665603;
        for l in run {
            for v in l {
                h = (h ^ u64::from(v.to_bits())).wrapping_mul(1099511628211);
            }
        }
        h
    };
    let mut distinct: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
    *distinct.entry(fingerprint(&base)).or_default() += 1;
    for _ in 2..reps {
        let run = replay(&mut session);
        *distinct.entry(fingerprint(&run)).or_default() += 1;
        if let Some((step, d)) = first_mismatch(&run, &base) {
            differing_runs += 1;
            first_bad[step] += 1;
            worst = worst.max(d);
        }
    }
    println!(
        "{} replays of {} decode steps after a {}-token prompt: {} differ from the first replay (max |logit diff| {:.3e})",
        reps - 2,
        steps,
        prompt.len(),
        differing_runs,
        worst
    );
    let mut sizes: Vec<usize> = distinct.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    println!(
        "distinct whole-run logit sequences among the {} replays: {} (runs per sequence: {sizes:?})",
        reps - 1,
        distinct.len()
    );
    let hist: Vec<String> = first_bad
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(s, n)| format!("step {s}: {n}"))
        .collect();
    println!(
        "first differing step (0 = prefill): {}",
        if hist.is_empty() {
            "none".into()
        } else {
            hist.join(", ")
        }
    );
}

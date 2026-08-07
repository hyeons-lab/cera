//! Regression harness for the entry path an FFI/library embedder takes.
//!
//! `backend::cpu::configure_thread_pool()` has exactly one production caller,
//! `cera-cli/src/main.rs` (plus this example's control arm). Every other
//! consumer, the UniFFI bindings in `cera-ffi` and any direct library user,
//! reaches inference through a `CeraEngine` constructor without it. The wasm
//! bindings are out of it entirely: their pool is built by JS calling
//! `initThreadPool`, and the fix below is a deliberate no-op on `wasm32`.
//!
//! That used to mean the two paths got different rayon pools, because rayon's
//! lazy global pool takes both its width and its CPU mask from whichever thread
//! first touches it. On the embedder path that thread had already been pinned
//! to a single core by `RowPool::pin_caller_once`, so rayon came up as one
//! worker on one core and everything left on it serialised. When this harness
//! was written that included the aarch64 i8mm prefill GEMM, which is what made
//! the gap so visible: measured on a Pixel 10 Pro Fold (LFM2.5-350M-Q4_K_M,
//! 512 prompt tokens) at 78-81 tok/s prefill against 108-189 for the CLI path.
//! Those kernels have since moved to the prefill `RowPool`, so what rides on
//! rayon now is model-load dequantization, the ViT patch embed and the audio
//! conv stem. Text prefill throughput will therefore no longer show the gap:
//! the thread dump is the part that still proves it, and the two arms should
//! agree on both.
//!
//! `CeraEngine::from_gguf` now calls `ensure_rayon_global_pool()`, which fixes
//! the width and mask independently of call order. This example exists to keep
//! that honest: run both arms and the thread dump and the throughput should
//! match. Run with `CERA_EMBEDDER_CONFIGURE=1` to opt into the CLI's
//! `configure_thread_pool()` call as the control arm.
//!
//! Linux and Android only, in substance. Elsewhere there is no affinity to
//! inherit (`pin_cores` is empty, `sched_setaffinity` is not called) and no
//! `/proc/self/task` to read, so both arms are identical by construction and
//! the thread dump reports itself unavailable. That is the platform being out
//! of scope, not the fix being a no-op.
//!
//! ```sh
//! # embedder arm
//! cargo run --release -p cera --example embedder_path -- model.gguf 512 16 8
//! # control arm: what cera-cli does
//! CERA_EMBEDDER_CONFIGURE=1 \
//!   cargo run --release -p cera --example embedder_path -- model.gguf 512 16 8
//! ```
//!
//! Arguments: `<model.gguf> [prompt_tokens] [max_tokens] [runs]`.

use cera::kv_cache::KvCacheConfig;
use cera::{
    BackendPreference, CeraEngine, EngineConfig, FinishReason, GenerateOpts, ModalitySink,
    SessionConfig,
};

struct NullSink;
impl ModalitySink for NullSink {
    fn on_done(&mut self, _reason: FinishReason) {}
}

/// Dump every thread's name and `Cpus_allowed_list`, grouped. A rayon pool that
/// inherited a pinned mask shows up as several threads sharing one core id.
fn dump_thread_affinity(label: &str) {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        println!("{label}: /proc/self/task unavailable");
        return;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let comm = std::fs::read_to_string(dir.join("comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        let mask = std::fs::read_to_string(dir.join("status"))
            .unwrap_or_default()
            .lines()
            .find(|l| l.starts_with("Cpus_allowed_list:"))
            .and_then(|l| l.split_whitespace().nth(1).map(str::to_string))
            .unwrap_or_default();
        *counts.entry((comm, mask)).or_default() += 1;
    }
    println!("{label}:");
    for ((comm, mask), n) in counts {
        println!("  {n:>2} x {comm:<24} cpus_allowed={mask}");
    }
}

const USAGE: &str = "usage: embedder_path <model.gguf> [prompt_tokens] [max_tokens] [runs]";

/// Parse an optional positional argument, naming it on failure. A misread
/// `prompt_tokens` silently changes what the two arms are being compared at,
/// so this reports which argument was bad rather than raising a bare
/// `ParseIntError`.
fn arg<T: std::str::FromStr>(v: Option<String>, name: &str, default: T) -> Result<T, String> {
    match v {
        None => Ok(default),
        Some(s) => s
            .parse()
            .map_err(|_| format!("{name}: expected a number, got `{s}`\n{USAGE}")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or(USAGE)?;
    let prompt_tokens: usize = arg(args.next(), "prompt_tokens", 512)?;
    let max_tokens: u32 = arg(args.next(), "max_tokens", 16)?;
    let runs: usize = arg(args.next(), "runs", 6)?;
    if prompt_tokens == 0 || max_tokens == 0 || runs == 0 {
        return Err(format!("prompt_tokens, max_tokens and runs must all be >= 1\n{USAGE}").into());
    }

    // The control arm: opt back into what cera-cli does. Accepts the same
    // spellings of "off" that `cpu_features::env_disabled` does, plus the
    // empty string, so `CERA_EMBEDDER_CONFIGURE=false` does not silently
    // select the arm it names.
    let configure = std::env::var("CERA_EMBEDDER_CONFIGURE").is_ok_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off"
        )
    });
    if configure {
        let n = cera::backend::cpu::configure_thread_pool();
        println!("called configure_thread_pool() -> {n} threads");
    } else {
        println!("did NOT call configure_thread_pool() (embedder path)");
    }

    // `EngineConfig`'s only other field is `#[cfg(feature = "remote")]`, so
    // the struct update covers a real field with that feature on and nothing
    // without it. Allowed rather than dropped, because dropping it stops the
    // example compiling under `--features remote`.
    #[allow(clippy::needless_update)]
    let engine_cfg = EngineConfig {
        context_size: 4096,
        backend: BackendPreference::Cpu,
        ..Default::default()
    };
    let engine = CeraEngine::from_path(&model, engine_cfg)?;

    // Every run replays the same prompt, so the KV prefix cache could serve
    // runs 2..N a warm restore and turn this into a measurement of the cache.
    // It should not today (a full hit is rejected, only strict prefixes
    // restore), but that is a side effect of a conv-buffer workaround rather
    // than a guarantee, and this harness reports prefill throughput. Default to
    // zero entries so the question cannot arise; `CERA_EMBEDDER_CACHE=1` puts
    // the default cache back, which is how you check whether a given set of
    // numbers was leaning on it.
    let cache_on = std::env::var("CERA_EMBEDDER_CACHE").is_ok_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off"
        )
    });
    println!("prefix cache: {}", if cache_on { "default" } else { "off" });
    if !cache_on {
        engine.configure_cache(KvCacheConfig {
            max_warm_entries: 0,
            max_warm_bytes: 0,
            max_cold_bytes: 0,
            cache_dir: None,
        });
    }

    // A prompt of a known token count, built from the model's own vocab so the
    // prefill width matches the CLI benchmark's `--prompt-tokens`.
    let bos = engine.tokenizer().bos_token();
    let mut tokens: Vec<u32> = Vec::with_capacity(prompt_tokens);
    if let Some(b) = bos {
        tokens.push(b);
    }
    let filler = engine
        .tokenizer()
        .encode("the quick brown fox jumps over a lazy dog ");
    if filler.is_empty() {
        return Err("tokenizer encoded the filler string to nothing; cannot build a prompt".into());
    }
    while tokens.len() < prompt_tokens {
        tokens.push(filler[tokens.len() % filler.len()]);
    }

    let opts = GenerateOpts {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ignore_eos: true,
        ..Default::default()
    };

    for i in 0..runs {
        let mut session = engine.new_session(SessionConfig::default())?;
        session.append_tokens(&tokens)?;
        let summary = session.generate(&opts, &mut NullSink)?;
        let pf = if summary.prompt_eval_ms > 0 {
            summary.prompt_eval_tokens as f64 * 1000.0 / summary.prompt_eval_ms as f64
        } else {
            f64::NAN
        };
        let dc = if summary.decode_ms > 0 {
            summary.tokens_generated as f64 * 1000.0 / summary.decode_ms as f64
        } else {
            f64::NAN
        };
        println!(
            "run {}/{}: prefill={pf:.0} decode={dc:.1} tok/s",
            i + 1,
            runs
        );
        if i == 0 {
            dump_thread_affinity("thread affinity after first generate");
        }
    }
    Ok(())
}

# cera

Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

> See the [project README](https://github.com/hyeons-lab/cera) for
> benchmarks and design notes.

`cera` is the core library: GGUF loading, a quantized CPU kernel stack
(AVX2/AVX-512, NEON dotprod/i8mm) with optional wgpu GPU and BLAS backends, a
stateful session API with prefix caching, and a streaming token sink. It powers
the [`cera-cli`](https://github.com/hyeons-lab/cera/tree/main/cera-cli) CLI, the
[`cera-ffi`](https://github.com/hyeons-lab/cera/tree/main/cera-ffi) mobile
bindings, and [`cera-wasm`](https://github.com/hyeons-lab/cera/tree/main/cera-wasm).

## Install

```toml
[dependencies]
cera = "0.3"
```

## Breaking changes in 0.3.0

0.3.0 adds public fields to two public structs, so it is a minor (not patch)
release — a `cargo update` from 0.2.x will not pull it in automatically.

- **`GenerateOpts` gained `ignore_eos: bool`** (run decode to exactly
  `max_tokens`, ignoring EOS/stop tokens — the `llama.cpp --ignore-eos`
  analog). Code that constructs `GenerateOpts` with an exhaustive struct
  literal must add the field; prefer functional-update syntax —
  `GenerateOpts { max_tokens: 256, ..Default::default() }` — which stays
  source-compatible across field additions. It defaults to `false`,
  preserving prior behavior.
- **`ModelMetadata` gained `add_eos_token: bool`** (mirrors GGUF
  `tokenizer.ggml.add_eos_token`, alongside the existing `add_bos_token`).
  This is an engine output type, so it only affects code that exhaustively
  pattern-matches or constructs it.

Also new (non-breaking): `BpeTokenizer::encode_special` (and the FFI
`encode_text_special` / wasm `encodeSpecial` wrappers) apply BOS/EOS to match
`llama.cpp`'s `llama_tokenize`, and `GenerateSummary::prompt_eval_ms` now
reports real prefill wall time paired with `prompt_eval_tokens`.

## Supported models

cera loads **GGUF** weights — either a raw `.gguf` file or a
[LeapBundles](https://huggingface.co/LiquidAI/LeapBundles) manifest that points
at one. Dispatch is on the GGUF `general.architecture` string:

| Architecture | Examples |
|--------------|----------|
| `lfm2` | Liquid LFM2 / LFM2.5 (the canonical LeapBundles family) |
| `qwen2`, `qwen3` | Qwen2 / Qwen2.5 / Qwen3 |
| `llama` | LLaMA 2/3, and classic Mistral 7B (ships as GGUF arch `llama`) |
| `granite` | IBM Granite 3.x |

Any other architecture errors out with `unsupported architecture: <name>` (this
includes the newer `mistral3`/`mistral4` layouts).

**Modalities:** text-to-text is fully supported for every architecture above.
**LFM2-Audio** (`lfm2-audio-v1`, text+audio in/out) also loads. **Vision (VL,
image-to-text)** is wired up end-to-end: `CeraEngine` auto-attaches the vision
mmproj encoder for VL bundles, and `Session::append_image` (or
`append_chat_with_images`) runs image → ViT → projector → soft-token prefill.
Verified against LFM2.5-VL-450M. The ViT encode runs on the GPU (native Metal or
wgpu, selected by `BackendPreference`) with a CPU fallback.

## Quick start

Load a local GGUF and stream tokens to stdout as they decode:

```rust
use cera::{CeraEngine, EngineConfig, FinishReason, GenerateOpts, ModalitySink, SessionConfig};
use cera::tokenizer::BpeTokenizer;

/// A `ModalitySink` receives decoded tokens as generation streams. Only
/// `on_done` is required; `on_text_tokens` defaults to a no-op.
struct Printer<'a> {
    tokenizer: &'a BpeTokenizer,
}

impl ModalitySink for Printer<'_> {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        print!("{}", self.tokenizer.decode(tokens));
    }
    fn on_done(&mut self, _reason: FinishReason) {}
}

fn main() -> Result<(), cera::CeraError> {
    // A `.gguf` file, a `.json` LeapBundles manifest, or a directory with one.
    let engine = CeraEngine::from_path("model.gguf", EngineConfig::default())?;

    let mut session = engine.new_session(SessionConfig::default())?;
    session.append_text("Once upon a time")?;

    let mut sink = Printer { tokenizer: engine.tokenizer() };
    let opts = GenerateOpts { max_tokens: 128, ..Default::default() };
    let summary = session.generate(&opts, &mut sink)?;

    eprintln!("\n[{} tokens, {:?}]", summary.tokens_generated, summary.finish_reason);
    Ok(())
}
```

`Session` keeps the KV cache alive across `append_text` / `generate` calls, so a
chat loop reuses the prefix cache instead of re-prefilling each turn. Render a
model's chat template with `cera::tokenizer::apply_chat_template`.

### Auto-downloading LeapBundles

With the `remote` feature, load a model straight from
[`huggingface.co/LiquidAI/LeapBundles`](https://huggingface.co/LiquidAI/LeapBundles)
by id and quant (cached locally, SHA-256 verified):

```rust
use cera::{CeraEngine, EngineConfig};
use cera::bundle::BundleRepo;

// `BundleRepo` caches downloaded manifests + model files under this directory.
let cfg = EngineConfig {
    bundle_repo: Some(BundleRepo::new("/path/to/cache")),
    ..Default::default()
};
let engine = CeraEngine::from_bundle_id("LFM2.5-1.2B-Instruct-GGUF", "Q4_0", cfg)?;
```

## Sampling

`GenerateOpts` exposes the usual knobs: `temperature`, `top_p`, `top_k`,
`min_p`, `repetition_penalty`, plus `stop_tokens` and an optional GBNF
`grammar` for constrained / JSON-shaped output. `temperature <= 0` (or
`top_k == 1`) selects deterministic greedy decoding; otherwise sampling is
stochastic. Min-p and repetition penalty apply on the stochastic path only.

## Tool calling

`cera::tools` renders tool schemas into the chat template and parses tool calls
back out, format-aware: `ToolFormat::detect(arch)` picks Pythonic (LFM2) vs
Hermes JSON (Qwen2.5/Qwen3) from the GGUF architecture.

Continuing from the Quick start (which sets up `engine`, `session`, and the
chat `messages`, and produces the decoded `reply_text`) — the schema below uses
the `serde_json` crate, which `cera` does not re-export, so add it to your
`Cargo.toml`:

```rust
use std::sync::Arc;
use cera::grammar::Grammar;
use cera::tools::{ToolDef, ToolFormat, tool_grammar, parse_tool_calls};
use cera::tokenizer::apply_chat_template_with_tools;

let tools = vec![ToolDef {
    name: "get_weather".into(),
    description: Some("Get the current weather for a city".into()),
    parameters: serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
    }),
}];
let format = ToolFormat::detect(&engine.model().config().architecture)
    .unwrap_or(ToolFormat::Lfm2Pythonic);

// Render tools into the prompt.
let prompt = apply_chat_template_with_tools(engine.tokenizer(), &messages, &tools, true)?;
session.append_text(&prompt)?;

// Optional: constrain to a valid call via grammar + lazy start-marker trigger.
let mut opts = GenerateOpts::default();
if let Some(trigger) = engine.tokenizer().special_token_id(format.call_start_marker()) {
    opts.grammar = Some(Arc::new(Grammar::parse(&tool_grammar(&tools, format)?)?));
    opts.grammar_trigger_tokens = vec![trigger];
}

// After generating, parse the reply. `ToolCall { name, arguments }`.
let calls = parse_tool_calls(&reply_text, format)?; // empty vec == answered in prose
```

The constrained path guarantees a well-formed call (valid function name, valid
argument names, correctly-typed values via JSON-Schema → GBNF); without it the
model decides freely whether and how to call a tool.

## LoRA adapters & hidden states

Load a LoRA adapter — a llama.cpp GGUF (from `convert_lora_to_gguf`) or a PEFT
`.safetensors` — and attach it to a `Session`. The delta is applied at inference
time (`y += scale·B·(A·x)`), **never merged into the weights**, so the base model
stays quantized and adapters hot-swap / unload per request. Runs on CPU, Metal,
and wgpu (batched-GEMM prefill + decode) and is dimension-checked at attach.

```rust
use cera::lora::LoraAdapterWeights;

let adapters = LoraAdapterWeights::from_safetensors(path, None)?; // or ::from_gguf(path)
session.attach_lora_adapters(adapters)?;   // hot-swap-able; applies to every forward
// ... generate / extract hidden states with the adapter active ...
session.remove_lora_adapters();
```

Pull the per-token last-layer hidden state (post-final-RMSNorm — the llama.cpp
`--pooling none` vector) straight out of the engine, reflecting the active
adapter. This is the classifier / embedding path (e.g. a section router: `LFM2.5`
+ a `route_section` LoRA + a small linear head over the mean-pooled state):

```rust
let hs = session.hidden_states_for_tokens(&tokens)?;      // [T * hidden_size], row-major
let pooled = session.hidden_states_mean_pooled(&tokens)?; // [hidden_size]
```

Both are also exposed over the FFI (`LoraAdapters` / `attachLora` /
`hiddenStatesMeanPooled`) and WASM bindings.

## Feature flags

Default-on features keep desktop/CLI builds full-featured; turn them off to
shrink the crate for `wasm32-unknown-unknown` or embedded targets
(`--no-default-features`).

| Feature | Default | What it adds |
|---------|:------:|--------------|
| `parallel` | ✅ | Multi-threaded CPU kernels (persistent affinity-pinned threadpool on native; rayon on wasm) |
| `std-fs` | ✅ | Filesystem access (paths, caches) |
| `mmap` | ✅ | Memory-mapped GGUF loading (⇒ `std-fs`) |
| `disk-cache` | ✅ | Cold KV-cache tier on disk (⇒ `std-fs`) |
| `vl-preprocess` | ✅ | Image input decode/resize for VL models |
| `avx512` | ✅ | x86-64 AVX-512 Q8_0/Q4_0 tier (needs Rust 1.89+) |
| `gpu` | — | wgpu compute backend |
| `metal` | — | Apple Metal backend (⇒ `mmap`) |
| `blas` | — | Opt-in GEMM accelerator |
| `remote` | — | `BundleRepo` HTTP download + SHA-256 (⇒ `std-fs`) |

MSRV: Rust 1.85 (edition 2024). The `avx512` tier needs 1.89+; disable it to
build on 1.85–1.88 (x86 then caps at AVX2).

## CPU threading & tuning

On native targets the CPU backend dispatches GEMV/GEMM rows through a
persistent, affinity-pinned worker pool (not a per-call fork-join), with dynamic
chunk-stealing so faster cores absorb more work on heterogeneous big.LITTLE
mobile. Decode runs on the detected performance cores; prefill uses all of them.
Two independent auto-caps bound the width: on heterogeneous big.LITTLE parts
(Linux/Android) detection keeps at most 6 big cores (both pools), and on the
homogeneous fallback — desktop/server/macOS, where sysfs detection is skipped
and every logical CPU counts as a "perf core" — decode is separately capped at
12 while prefill uses all. This fixes the multi-core decode collapse on Android
big.LITTLE and lets decode scale across the performance cores. Everything is
auto-detected per device — the environment variables below only override for
tuning (`CERA_THREADS` moves the detected count past either cap in both
directions):

| Variable | Default | Effect |
|----------|---------|--------|
| `CERA_DECODE_THREADS` | detected perf cores (≤6 heterogeneous, ≤12 homogeneous) | Decode worker count — a fixed `<n>`, or `auto`. A fixed value is clamped to the detected performance cores; the 12 cap applies only to the homogeneous `auto` path (heterogeneous big.LITTLE detection already caps big cores at 6). |
| `CERA_THREADS` | detected perf-core count | Override the detected performance-core count (moves the auto width for both pools). |
| `CERA_MIN_ROWS` | 128 | Minimum output rows a decode-GEMV worker takes before another joins. |
| `CERA_PAR_THRESHOLD` | 256 | Minimum output dimension before a GEMV parallelizes; smaller GEMVs stay serial. |
| `CERA_SPIN` | 100000 | Spin iterations before an idle worker parks. |
| `CERA_PIN` | on | `0` / `false` / `off` disables affinity pinning (for hosts that manage thread placement themselves). |
| `CERA_CPU_TIER` | auto | Force a lower CPU SIMD tier (downgrade only) — for parity testing on capable hardware. |

Affinity pinning applies on Linux/Android with a detected heterogeneous
topology; homogeneous hosts and macOS run unpinned.

## License

Apache-2.0 OR MIT.

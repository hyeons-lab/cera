# cera

Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

> The 0.6.2 API corrects chat lifecycle, streaming, checkpoint validation, structured output and audio processing. See the [0.6 API guide](../docs/API_0_6.md) for migration and compatibility limits, and [Releases](https://github.com/hyeons-lab/cera/releases) for published builds.

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
cera = "0.6"
```

## The 0.6 API

- **Chat coordinator (`cera::session::chat`)**: `Session::into_chat()` transfers execution into `SessionChat`, with delta-only prefill and validated template profiles. Ingest another turn only from `TurnComplete`. Interrupted generation requires reset or replacement before a new user turn; zero-token/no-progress calls can preserve `PromptReady` for retry. Recovery can restore, reset or leave the session unusable. Legacy `Session::append_user_message` is deprecated in favor of this API.
- **Language-Native Reactive Streaming**: Real-time token and text streaming via `SessionChat::stream_text` in Rust, `AsyncThrowingStream` in Swift, `Flow` in Kotlin, `Iterator[str]` in Python, and `Stream<String>` in Dart, with cancellation isolation across conversation turns.
- **JSON Schema compiler (`cera::grammar::json_schema_to_gbnf`)**: Compiles a [documented subset](../docs/API_0_6.md#json-schema-constraints) with required/optional properties, bounded arrays, local refs and restricted `allOf` composition. Use `GenerateOpts::with_json_schema`; still check for truncated output and validate constraints outside the supported subset.
- **First-Class Tool Calling**: Tool definition, schema validation, format detection (LFM2 Pythonic, Hermes/Qwen JSON), `chat.set_tools()`, and `chat.ingest_tool_response()`.
- **Session checkpoints (`cera::session::checkpoint`)**: CPU Session/Chat snapshots (`CERASCHK` / `CERACHAT`) validate structural identity, KV geometry and compression including the TurboQuant seed. Native Metal/wgpu checkpoints are rejected; browser WebGPU has a separate snapshot path. Recreate pre-0.6.2 f16/TurboQuant checkpoints with old fingerprints. File persistence uses atomic rename.
- **Unified Stateful Audio Pipeline (`cera::audio_pipeline::AudioPipeline`)**: Stateful streaming pipeline uniting Silero VAD v5, streaming hotword detection, and Whisper speech-to-text transcription. Features pre-roll ring buffering, max utterance duration chunking that preserves active VAD hidden states across continuation segments, automatic transcription, and wait-free cancellation across FFI boundaries.
- **Expanded Model Architectures**: Native support for Mamba-2 SSM and hybrid architectures, Gemma 2, Olmo 2, Gemma 4 (PLE and cross-layer KV sharing), Olmo 3 (sliding window and YaRN RoPE), MiniCPM, Nanbeige 4.2, Qwen 3.5 / Ornith 1.0, Ministral 3, Phi-3 / Phi-4-mini, and Ling 3.0 Tiny.
- **Prefix cache anchors (`cera::kv_cache::KvPrefixCache`)**: Warm memory and optional disk caching with caller-supplied anchor metadata. FlatBuffers v2 persistence retains existing snapshot precision; the engine does not automatically extract anchors or recompress cold entries.
- **DSpark: Neural Speculative Decoding ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608))**: Neural speculative drafting via lightweight sidecars (`cera::model::dspark`), parallel multi-token GPU verification on Metal and WebGPU, and batched LM-head verification.
- **TurboQuant KV-Cache Compression ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874))**: Pure-Rust PolarQuant + QJL compression achieving ~12x KV-cache memory reduction across CPU, Metal, and WebGPU backends.
- **Pure-Rust Silero VAD v5 (`cera::vad`)**: Native ONNX-free voice activity detection engine (`SileroVad`, `VadIterator`, `VadConfig`, `VadSampleRate`) operating on 512-sample streaming audio frames with automatic speech segment timestamping.
- **OpenAI Whisper ASR (`cera::model::whisper`)**: Pure-Rust Whisper speech-to-text inference with multi-language identification, timestamp support, and cooperative cancellation.

The additive API work also includes a [checked raw KV rewind example](examples/checked_rewind.rs)
and a [backend recovery matrix](../docs/internals/API_RESHAPE_RECOVERY.md).
CPU Llama/LFM2 support checked recovery. The runnable
[ingestion recovery example](examples/ingestion_recovery.rs) demonstrates cancellation,
restoration/reset diagnostics and retry. Select native device reset with `--metal`
or `--wgpu` and the matching Cargo feature; `--compressed` exercises TurboQuant.
The full conversational chat contract is available via `Session::into_chat()` and
demonstrated in the [chat example](examples/chat.rs).

## Breaking changes in 0.4.0

0.4.0 adds public fields and enum variants to public types, so it is a minor
(not patch) release; a `cargo update` from 0.3.x will not pull it in
automatically. The affected 0.4.0 types were not `#[non_exhaustive]`, so these break any code
that writes exhaustive struct literals or exhaustive `match`es. Code that keeps
the default settings sees no behavior change; the one exception is the new
`KvCompressionConflict` below, which turns a previously-silent mismatch into an
error.

- **`GenerateOpts` gained `spec: Option<SpecDecode>`**: opt-in greedy
  speculative decoding (see [Speculative decoding](#speculative-decoding)).
  Code that constructs `GenerateOpts` with an exhaustive struct literal must
  add the field; prefer functional-update syntax,
  `GenerateOpts { max_tokens: 256, ..Default::default() }`, which stays
  source-compatible across field additions. It defaults to `None`, preserving
  prior behavior.
- **`CeraError` gained `KvCompressionConflict`**: returned when a second
  session asks a model for a different KV-compression mode than the one it was
  built with, instead of handing back a cache the kernels do not match.
  Exhaustive `match`es over `CeraError` need a new arm. This is the one item
  here that changes runtime behavior: that call used to succeed silently and
  corrupt the prefix cache.
- **`CeraError` also gained `LoraUnsupportedByBackend`**: an adapter that is
  well-formed and dimensionally valid, but adapts a target the active backend has
  no hook for. Today that is exactly the routed mixture-of-experts targets (the
  router and the per-expert projections) on the two GPU backends, which the CPU
  backend applies. Another new arm for exhaustive `match`es. Note the FFI mirror
  `FfiError` appends its counterpart at the *end* of the enum: UniFFI serializes
  by declaration ordinal, so a mid-enum insertion would renumber every later
  variant for a prebuilt binding.
- **`ModelConfig` gained `moe: Option<MoeConfig>`**, `None` for every dense
  architecture, alongside the new public `MoeConfig` type. Struct literals of
  `ModelConfig` need the field; `..Default::default()` is not available here, so
  this is a compile error rather than a silent behavior change.
- **`LoraTarget` gained four routed-FFN variants** (`FfnGateInp` for the router,
  and `FfnGateExps` / `FfnUpExps` / `FfnDownExps` for the per-expert
  projections). Exhaustive `match`es need the arms, and `LoraTarget::ALL` is now
  `[LoraTarget; 13]` rather than `[LoraTarget; 9]`: prefer the public
  `LORA_TARGET_COUNT` over a hard-coded length in any array typed by it.
- **The f16 KV cache** (added for decode-at-depth) widened four public items in
  the `cera::kv_cache` module: `KvCompression` gained `F16`, `LayerSnapshot`
  gained `AttentionF16`, `LayerState::Attention` gained `key_cache_f16` and
  `value_cache_f16`, and `InferenceState` gained `kv_f16: bool`. Exhaustive
  `match`es and literals over any of them need updating; matching
  `LayerState::Attention { .. }` with a rest pattern is unaffected.
- **Behind the non-default `gpu` feature**, `backend::wgpu::io_stats::GpuIoStats`
  gained a `passes: u64` counter, and `GpuIoStats::per_token` returns
  `(f64, f64, f64, f64)` instead of `(f64, f64, f64)`. That one is a signature
  change rather than an exhaustiveness break, so callers must destructure the
  extra element. These are debug counters; nothing in the inference path uses
  them.

Also new (non-breaking): `SpecDecode` and the `cera::spec` module; the `Model`
trait gained a defaulted `truncate_kv` method, so a backend can override the
speculative-decoding KV rewind while existing implementors inherit the prior
behavior unchanged; TurboQuant KV-cache compression now runs on the wgpu and
native Metal backends, not just CPU. The rest of the release is CPU and GPU
performance and correctness work, including a batched LM-head projection that
amortizes the output matrix across all verified positions in speculative decode,
and a Q4_1 decode path that now matches the batched prefill GEMM bit-for-bit. See
the [benchmarks](https://github.com/hyeons-lab/cera/tree/main/benchmarks).

## Changes in 0.3.1

A patch release: CPU and GPU performance work, Q5_K/Q4_1 quantization support,
and a native wgpu flash-attention decode path. No changes to `CeraEngine`,
`Session`, `GenerateOpts`, or any other type in the public prelude.

One caveat for anyone reaching into backend internals: the dead WGSL kernels
`backend::wgpu::shaders::{GEMM_Q4_0, GEMM_Q8_0, ATTENTION}` were removed once
the register-tiled GEMM and flash-attention kernels superseded them. They were
shader **source text** behind the `gpu` feature, never part of the intended
API, so this ships as a patch rather than a minor bump, but a `^0.3` consumer
that named them will need to stop.

## Breaking changes in 0.3.0

0.3.0 adds public fields to two public structs, so it is a minor (not patch)
release; a `cargo update` from 0.2.x will not pull it in automatically.

- **`GenerateOpts` gained `ignore_eos: bool`** (run decode to exactly
  `max_tokens`, ignoring EOS/stop tokens, the `llama.cpp --ignore-eos`
  analog). Code that constructs `GenerateOpts` with an exhaustive struct
  literal must add the field; prefer functional-update syntax,
  `GenerateOpts { max_tokens: 256, ..Default::default() }`, which stays
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

cera loads **GGUF** weights, either a raw `.gguf` file or a
[LeapBundles](https://huggingface.co/LiquidAI/LeapBundles) manifest that points
at one. Dispatch is on the GGUF `general.architecture` string; selected families
are listed below:

| Architecture | Examples |
|--------------|----------|
| `lfm2` | Liquid LFM2 / LFM2.5 (the canonical LeapBundles family) |
| `lfm2moe` | Liquid LFM2.5-8B-A1B (routed mixture-of-experts) |
| `qwen2`, `qwen3` | Qwen2 / Qwen2.5 / Qwen3 |
| `qwen35` | Qwen 3.5 / Ornith 1.0 (interleaved Gated Delta Net hybrid) |
| `llama` | LLaMA 2/3, and classic Mistral 7B (ships as GGUF arch `llama`) |
| `granite` | IBM Granite 3.x, and the dense Granite 4.1 line (3b / 8b / 30b) |
| `minicpm`, `minicpm5` | MiniCPM / MiniCPM5 (MiniCPM-1B / 2B) |
| `nanbeige` | Nanbeige 4.2 (looped-layer dense transformer) |
| `mistral3`, `ministral3` | Mistral 3 / Ministral 3 (Ministral-3B / 8B, Mistral-Small-3) |
| `phi3`, `phi` | Microsoft Phi-3-mini, Phi-3.5-mini, and Phi-4-mini (fused QKV, packed SwiGLU FFN) |
| `bailingmoe3`, `bailingmoe` | Ling 3.0 Tiny (hybrid KDA linear, MLA latent attention, and MoE) |

The CPU loader also accepts Gemma 2/4, Olmo 2/3, Granite Hybrid, Falcon H1 and
Mamba-2 families. See [model loading](src/model/mod.rs) for architecture aliases
and backend admission rules. Tensor layouts and quantization must also match
the selected loader. An architecture outside that dispatch returns
`unsupported architecture: <name>`; the table above is not exhaustive.

**Modalities:** text-to-text is fully supported for every architecture above.
**LFM2-Audio** (`lfm2-audio-v1`, text+audio in/out) also loads. **Vision (VL,
image-to-text)** is wired up end-to-end: `CeraEngine` auto-attaches the vision
mmproj encoder for VL bundles, and `Session::append_image` (or
`append_chat_with_images`) runs image → ViT → projector → soft-token prefill.
Verified against LFM2.5-VL-450M. The ViT encode runs on the GPU (native Metal or
wgpu, selected by `BackendPreference`) with a CPU fallback.

## Sharing a loaded GPU model

Metal and wgpu permit one live `Session` per loaded model. A second session
returns `Busy` until the first is dropped; reset and cancellation keep ownership.
CPU models continue to support shared weights across concurrent sessions. Load
separate GPU models for simultaneous conversations. The
[session ownership walkthrough](../docs/internals/API_RESHAPE_GPU_SESSION_EXAMPLES.md) shows the public API, cleanup behavior
and executable CPU/Metal/wgpu checks.

LFM2-Audio transcription during a live GPU conversation uses a cached secondary
model built from retained weights. This preserves conversation KV and costs
additional model memory and first-use setup; the walkthrough includes a native
Dart consumer for transcription and seeded generation.

## Standalone Whisper

`WhisperModel::from_file` and `WhisperModel::from_bytes` return the model and its
tokenizer. Supply 16 kHz mono float PCM to `model.transcribe`:

```rust,no_run
use cera::{WhisperModel, WhisperTranscribeOpts};

fn transcribe_recording(path: &str, pcm_16k_mono: &[f32]) -> anyhow::Result<String> {
    let (model, tokenizer) = WhisperModel::from_file(path)?;
    model.transcribe(&tokenizer, pcm_16k_mono, &WhisperTranscribeOpts {
        language: Some("en".into()),
        ..Default::default()
    })
}
```

This synchronous CPU path handles one 30-second chunk. The
[Swift/Kotlin guide](../docs/internals/API_RESHAPE_WHISPER_EXAMPLES.md) uses the
standalone FFI object and its background decode method. The unified typed
Whisper/VAD/hotword loader remains planned separately from the generative API refactor.

## API refactor examples

The `ModelLoader` / `GenerativeModel` API is public in Rust and the generated
native bindings. The linked refactor walkthroughs also retain historical
validation details from before promotion.
[Run the Rust, Swift and Kotlin examples](../docs/internals/API_RESHAPE_EXAMPLES.md)
to load a fixture, create a session and generate, or exercise paired vision and
DSpark companions. The guide links complete executable sources and explains
which APIs are implemented. The quick start below shows both explicit loading
and the existing `CeraEngine` API.
The [audio walkthrough](../docs/internals/API_RESHAPE_AUDIO_EXAMPLE.md) loads an encoder/vocoder, ingests PCM and captures 24 kHz output using synthetic CPU weights.
The [remote companion examples](../docs/internals/API_RESHAPE_REMOTE_EXAMPLES.md) exercise discovery, integrity and retained CPU execution through loopback HTTP.
The [HF revision examples](../docs/internals/API_RESHAPE_HF_EXAMPLES.md) load changed GGUF revisions into separate cache entries, verify defaults and keep a CPU session live across another revision load. They also exercise failed metadata resolution and explicit commit checks.
The [SafeTensors conversion examples](../docs/internals/API_RESHAPE_CONVERSION_EXAMPLES.md) exercise real conversion, corruption repair and option-sensitive cached reloads. The examples verify checkpoint bytes, pin inputs to one upstream commit, and refresh changed revisions. Retained models execute after replacement, and sessions continue after their parent model is released.
The [persistent cache examples](../docs/internals/API_RESHAPE_CACHE_EXAMPLES.md) demonstrate same-path weight replacement, unchanged-weight disk reuse and retained live sessions.

## Quick start

This checkout includes explicit model loading alongside `CeraEngine` constructors:

```rust
use cera::{BackendPreference, LoadConfig, ModelLoader, ModelSource, SessionConfig};

let model = ModelLoader::new(ModelSource::path("model.gguf"))
    .config(LoadConfig { backend: BackendPreference::Cpu, ..LoadConfig::default() })
    .build_generative()?;
let engine = model.engine(); // Shares the loaded engine and weights.
let mut session = model.create_session(SessionConfig::default())?;
drop(model);
session.append_tokens(&engine.tokenizer().encode("Once upon a time"))?;
```

`ModelSource::bytes` and `reader` work without `mmap`; `path` and `files` require
`mmap`, and `bundle_id`/`hugging_face` additionally require `remote`. `LoadConfig`
is an alias of the existing `EngineConfig`. Use `.build()` for a dynamic
`ModelHandle`, then `.as_generative()` to share its generative model. Other
recognized kinds return `LoadError::KindMismatch` before generative assembly;
their existing standalone constructors remain available.

The complete [raw text-completion example](examples/explicit_loading.rs) tokenizes
the prompt and decodes generated tokens:

```bash
cargo run -p cera --example explicit_loading -- model.gguf "Once upon a time"
```

For conversational chat, use the dedicated coordinator (`Session::into_chat()`)
detailed below, which handles Jinja2 template formatting, incremental turn boundaries,
and KV cache retention automatically. Run the [conversational chat example](examples/chat.rs):

```bash
cargo run -p cera --example chat -- model.gguf
```

Load a local GGUF and collect streamed tokens. Decode the combined result so
UTF-8 sequences split across batches remain intact:

```rust
use cera::{CeraEngine, EngineConfig, FinishReason, GenerateOpts, ModalitySink, SessionConfig};

/// A `ModalitySink` receives decoded tokens as generation streams. Only
/// `on_done` is required; `on_text_tokens` defaults to a no-op.
#[derive(Default)]
struct TokenCollector {
    tokens: Vec<u32>,
}

impl ModalitySink for TokenCollector {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
    }
    fn on_done(&mut self, _reason: FinishReason) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A `.gguf` file, a `.json` LeapBundles manifest, or a directory with one.
    let engine = CeraEngine::from_path("model.gguf", EngineConfig::default())?;

    let mut session = engine.new_session(SessionConfig::default())?;
    session.append_text("Once upon a time")?;

    let mut sink = TokenCollector::default();
    let opts = GenerateOpts { max_tokens: 128, ..Default::default() };
    let summary = session.generate(&opts, &mut sink)?;
    println!("{}", engine.tokenizer().decode(&sink.tokens));

    eprintln!("\n[{} tokens, {:?}]", summary.tokens_generated, summary.finish_reason);
    Ok(())
}
```

`Session` retains live KV across `append_text` / `generate` calls. Continue by
appending new input to the same session. For incremental chat text, use
`SessionChat::stream_text`, which buffers incomplete UTF-8 sequences.

### Conversational chat coordinator (`Session::into_chat`)

For chat models, use `Session::into_chat()` instead of manual string formatting.
`SessionChat` discovers the model's chat template, tracks conversation phases,
enforces role alternation, evaluates only new tokens on continuation turns (delta-only prefill),
and retains live KV state across turns:

```rust
use cera::{CeraEngine, EngineConfig, GenerateOpts, Message, SessionConfig, SessionPhase};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = CeraEngine::from_path("model.gguf", EngineConfig::default())?;
    let session = engine.new_session(SessionConfig::default())?;

    // Transition the session into a chat coordinator.
    let mut chat = session.into_chat().map_err(|(_, err)| err)?;

    // Ingest conversation history or initial prompt.
    chat.ingest_messages(&[
        Message::system("You are a concise, helpful assistant."),
        Message::user("What is the capital of France?"),
    ])?;

    // Generate assistant reply (delta-only prefill + decode).
    let opts = GenerateOpts { max_tokens: 64, ..Default::default() };
    let reply = chat.complete(&opts)?;
    println!("Assistant: {}", reply.text);

    // Continue only after the profile's terminal marker ended the first turn.
    if chat.phase() == SessionPhase::TurnComplete {
        chat.ingest(&Message::user("And what is its population?"))?;
        let reply2 = chat.complete(&opts)?;
        println!("Assistant: {}", reply2.text);
    }

    // Reclaim the underlying session when raw completion access is needed.
    let mut session = chat.into_session();
    session.reset()?;
    Ok(())
}
```

Legacy `Session::append_user_message` is deprecated in favor of `Session::into_chat()`.
See [`cera/examples/chat.rs`](examples/chat.rs) for the complete runnable example.

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

## Speculative decoding

`GenerateOpts::spec` opts into greedy speculative decoding with **prompt-lookup
(n-gram) drafting**: no draft model, so no extra weight memory. The drafter
guesses the next tokens from the most recent earlier occurrence of the last
`ngram` tokens, and the target verifies up to `k` of them in a single forward.
A target forward reads every weight once, so verifying K drafted tokens in one
pass amortizes that read over the accepted run, which is why this targets the
memory-bandwidth wall in CPU decode-at-depth. As of #327 the verify path projects
all `1 + k` positions' logits in a single batched GEMM, so the LM head (the
largest matrix in the model) is read once per round rather than once per position;
a per-row fallback remains for head dtypes without a batched kernel. The 1.49x
measured on a repetitive prompt predates that change and did not include its gain.

```rust
use cera::SpecDecode;

let opts = GenerateOpts {
    temperature: 0.0,                       // greedy path only
    spec: Some(SpecDecode { ngram: 2, k: 6 }), // or SpecDecode::default()
    ..Default::default()
};
```

**Every emitted token is the target's own argmax**, a valid greedy decode, so
a poor draft lowers the acceptance rate without affecting correctness. It is not
guaranteed bit-identical to a *sequential* greedy run: the verifier forwards a
batch where a sequential loop forwards one token at a time, and the two
reduction orders can pick opposite sides of a near-tie. It engages only on the
plain greedy path (`temperature <= 0` or `top_k == 1`, no grammar), with a model
that reports `supports_all_logits()`, an uncompressed (f32/f16) KV cache, and no
audio decoder. CPU dense transformers and LFM2, native Metal models, and wgpu
models with batched prefill and compatible matrix weights advertise this
capability. Configurations excluded by these gates use normal decode. Backend
rewind support and the draft size still constrain execution; benchmark with the
intended model/backend rather than assuming a speedup.

The CLI exposes prompt-lookup knobs on `bench` (`--spec`, `--spec-ngram`,
`--spec-k`). `run` and `chat` expose separate draft-model options.

## Tool calling

`cera::tools` renders tool schemas into the chat template and parses tool calls
back out, format-aware: `ToolFormat::detect(arch)` picks Pythonic (LFM2) vs
Hermes JSON (Qwen2.5/Qwen3) from the GGUF architecture.

Using `engine`, the mutable `session`, and `TokenCollector` from the raw-token
quickstart, replace its prompt/generation block with the sequence below. It
resets existing raw context, renders a tool-aware prompt, generates and parses
the reply. Add `serde_json` to your `Cargo.toml`; `cera` does not re-export it:

```rust
use std::sync::Arc;
use cera::grammar::Grammar;
use cera::tools::{ToolDef, ToolFormat, tool_grammar, parse_tool_calls};
use cera::tokenizer::{ChatMessage, apply_chat_template_with_tools};

session.reset()?;
let messages = vec![ChatMessage {
    role: "user".into(),
    content: "What is the weather in Paris?".into(),
}];

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

// Optional: constrain tool-call syntax via grammar + lazy start-marker trigger.
let mut opts = GenerateOpts::default();
if let Some(trigger) = engine.tokenizer().special_token_id(format.call_start_marker()) {
    opts.grammar = Some(Arc::new(Grammar::parse(&tool_grammar(&tools, format)?)?));
    opts.grammar_trigger_tokens = vec![trigger];
}

let mut sink = TokenCollector::default();
session.generate(&opts, &mut sink)?;
let reply_text = engine.tokenizer().decode(&sink.tokens);

// Parse the generated reply. `ToolCall { name, arguments }`.
let calls = parse_tool_calls(&reply_text, format)?; // empty vec == answered in prose
```

The constrained path limits function names, argument names and value syntax to
the supported tool grammar. Its separate compiler does not enforce required
arguments, uniqueness or nested item/property schemas. A token limit or
cancellation can still truncate a call; parse and validate it before execution.
Without constraints, the model decides freely whether and how to call a tool.

## LoRA adapters & hidden states

Load a LoRA adapter, a llama.cpp GGUF (from `convert_lora_to_gguf`) or a PEFT
`.safetensors`, and attach it to a `Session`. The delta is applied at inference
time (`y += scale·B·(A·x)`), **never merged into the weights**, so the base model
stays quantized and adapters hot-swap / unload per request. Runs on CPU, Metal,
and wgpu (batched-GEMM prefill + decode) and is dimension-checked at attach. The
one gap is `lfm2moe`'s routed-FFN targets (the router and the per-expert
projections), which apply on CPU only; on a GPU backend such an adapter is
refused with `CeraError::LoraUnsupportedByBackend` rather than half-applied.

```rust
use cera::lora::LoraAdapterWeights;

let adapters = LoraAdapterWeights::from_safetensors(path, None)?; // or ::from_gguf(path)
session.attach_lora_adapters(adapters)?;   // hot-swap-able; applies to every forward
// ... generate / extract hidden states with the adapter active ...
session.remove_lora_adapters();
```

Pull the per-token last-layer hidden state (post-final-RMSNorm, the llama.cpp
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
| `gpu` | - | wgpu compute backend |
| `metal` | - | Apple Metal backend (⇒ `mmap`) |
| `blas` | - | Opt-in GEMM accelerator |
| `remote` | - | `BundleRepo` HTTP download + SHA-256 (⇒ `std-fs`) |

MSRV: Rust 1.94 (edition 2024; the NEON f16 `vcvt_f32_f16` KV-cache widen needs
1.94). The default-on `avx512` feature enables the AVX-512 tier on x86;
disabling it caps that tier at AVX2.

## CPU threading & tuning

On native targets the CPU backend dispatches GEMV/GEMM rows through a
persistent, affinity-pinned worker pool (not a per-call fork-join). Rows are
handed out by dynamic chunk-stealing, so a faster core simply claims more
chunks. On a part whose cores differ in speed each worker's chunk is also
*sized* to the `cpu_capacity` of the core it is pinned to, so that every
worker's chunk costs roughly the same wall-clock time and one slow core cannot
hold the rest at the dispatch barrier for a multiple of what its chunk cost.
Both mechanisms are inert on homogeneous hosts, and `CERA_PIN=0` turns the
sizing off along with the pinning it reads placement from.

On heterogeneous big.LITTLE parts (Linux/Android) detection separates the
performance cores from the efficiency ones and sizes both pools to the former,
which fixes the multi-core decode collapse there. The efficiency cores are
still recorded, so a deliberately widened pool can give every worker its own
core rather than falling off a cliff, but nothing is that wide by default.
Capacity-sized chunks make such a widened pool considerably less costly
(measured on a Tensor G5, `CERA_THREADS=8` prefill 116.5 to 142.0 tok/s) but
not free, so it remains an override rather than the default.

Elsewhere, desktop/server (where sysfs detection is skipped and every
logical CPU counts as a "perf core") and macOS (where the P-core count comes
from `hw.perflevel0`), prefill
uses all of them while **decode is sized from the loaded model** (see "How the
decode thread count is chosen" in the top-level README): small models that
spread a token across many small pool dispatches run narrow, large ones that
move more bytes per dispatch run wide. Where that sizing does not apply,
heterogeneous parts, or a host whose physical core count cannot be detected
(Windows, BSD, Intel macOS), the flat cap applies instead: the detected
perf-core count, capped at 12. Both pools are process-wide singletons, so the
decode width is sized from the **first** model loaded into a process and stays
there for any loaded after it; it does not re-size per load. Everything else is
auto-detected per device; the
environment variables below only override for tuning (`CERA_THREADS` moves the
detected performance-core count, which both RowPools and rayon's global pool
size from):

| Variable | Default | Effect |
|----------|---------|--------|
| `CERA_DECODE_THREADS` | `auto` | Decode worker count. A fixed `<n>` pins the width and overrides the automatic sizing (clamped to the detected performance cores); `auto` selects the model-based sizing below. |
| `CERA_DECODE_SIZING` | on | `0` / `false` / `off` disables model-aware decode sizing, falling back to the flat cap (detected perf cores, capped at 12). |
| `CERA_DECODE_NARROW` | `physical / 2`, capped at 12 | Decode width for barrier-bound models (below the bytes-per-dispatch threshold); never exceeds the wide arm. Setting it also forces sizing on where it would otherwise be declined (on a host whose physical core count is undetectable, both arms must be pinned). |
| `CERA_DECODE_WIDE` | `physical + physical / 4`, capped at 24 | Decode width for bandwidth-bound models, clamped to the detected cores. Setting it also forces sizing on where it would otherwise be declined (on a host whose physical core count is undetectable, both arms must be pinned). |
| `CERA_DECODE_BPD_KB` | 2500 | Bytes-per-dispatch threshold (decimal KB) separating the two arms above. Unlike the two widths, this does **not** force sizing on where it is declined; it moves the threshold, it does not pin a width. |
| `CERA_THREADS` | detected perf-core count | Override the detected performance-core count (moves the auto width for both pools). Clamped to the number of pinnable cores on hosts that have any, with a warning: past that the surplus workers run unpinned and contend with pinned ones that are spin-waiting, measured at 35x slower on a Tensor G4. Not clamped where nothing gets pinned anyway: hosts with no affinity, or `CERA_PIN=0`. |
| `CERA_PREFILL_THREADS` | detected perf-core count | Prefill pool width on its own, without moving decode. May reach past the performance cores up to every pinnable core, for sweeping a part where widening might pay. It does not on the parts measured so far, though it is no longer costly: the prefill pool stops pinning once it is wider than the performance cores, which took widening on a Tensor G5 from 211 to 148 tok/s down to 211 to about 200. Still a small loss, so the default stays narrow. Same no-clamp-without-pins rule as `CERA_THREADS`, including the `CERA_PIN=0` case. Note the two interact: `CERA_THREADS` truncates the pinnable-core list, so setting it as well lowers the ceiling this is clamped against. To sweep prefill past the perf cores, leave `CERA_THREADS` unset. |
| `CERA_MIN_ROWS` | 128 | Minimum output rows a decode-GEMV worker takes before another joins. |
| `CERA_PAR_THRESHOLD` | 256 | Minimum output dimension before a GEMV parallelizes; smaller GEMVs stay serial. |
| `CERA_SPIN` | 100000 | Spin iterations before an idle worker parks. |
| `CERA_PIN` | on | `0` / `false` / `off` disables affinity pinning (for hosts that manage thread placement themselves). |
| `RAYON_NUM_THREADS` | detected perf-core count (moved by `CERA_THREADS`) | Width of rayon's global pool, which covers the parallel sites outside the RowPools: dequantization (so, model load), the ViT patch embed, and the LFM2-Audio conv stem. It does **not** move text prefill or decode width; every GEMM and GEMV on that path runs on a `RowPool`, sized by `CERA_PREFILL_THREADS` for prefill and `CERA_DECODE_THREADS` for decode (both defaulting from `CERA_THREADS`). It can still move a VL or audio prefill, whose encoders fan out on rayon. Read by `cera` itself rather than left to rayon, so the pool is built eagerly with a known CPU mask instead of lazily inheriting the mask of whichever thread reached it first. |
| `CERA_RAYON_GLOBAL` | on | `0` / `false` / `off` stops `cera` claiming rayon's process-global pool, for a Rust host that wants to build it itself. Such a host can also just call `rayon::ThreadPoolBuilder::new().build_global()` before loading a model; `cera` then logs a warning and leaves it alone. |
| `CERA_CPU_TIER` | auto | Force a lower CPU SIMD tier (downgrade only), for parity testing on capable hardware. |
| `CERA_POOL_STATS` | off | `1` annotates each `cera bench` run with the pool's fan-out health: how many dispatches wanted more than one worker, and how many of those silently ran serially because the pool was already busy. The counts are exact; the accompanying work percentage mixes units across dispatch kinds, so read the counts. |
| `CERA_LM_HEAD_NO_GEMM` | unset | `1` puts the LM-head projection in `forward_prefill_logits_all` back on the per-row loop the batched GEMM replaced, so both halves of a speculative-decoding A/B run from one binary. Measurement lever only; both paths compute the same projection, to within f32 accumulation order. |

Affinity pinning applies on Linux/Android with a detected heterogeneous
topology; homogeneous hosts and macOS run unpinned.

None of this needs configuring from the host. Loading a model builds rayon's
global pool; the RowPools build themselves on first use (prefill on the first
GEMM, decode on the first decode GEMV, which is what lets decode size itself
from the loaded model). Two functions let a host move that work earlier if it
wants to: `cera::backend::cpu::ensure_rayon_global_pool()` builds just the
rayon pool, and `configure_thread_pool()` also warms the prefill RowPool and
returns its width, so a CLI can report a thread count before a model exists.
Both are optional and idempotent.

## License

Apache-2.0 OR MIT.

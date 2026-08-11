# Cera

**A Rust-native LLM inference engine.** Load a GGUF model and run it locally,
on your laptop's CPU, an Apple GPU, a cross-platform Vulkan/DX12 GPU, a phone,
or in the browser, from a single dependency-free core.

## Why Cera

- **No Python, no runtime.** Pure Rust. The CLI is a single binary; the
  library has zero required system dependencies on a default build.
- **Runs everywhere.** The same core drives a desktop CLI, Android/iOS apps
  (via UniFFI), and the browser (via WebAssembly). Pick CPU or GPU at runtime.
- **Loads standard GGUF.** Point it at a `.gguf` file, a
  [LeapBundles](https://huggingface.co/LiquidAI/LeapBundles) manifest, or a
  bundle id; it can auto-download and cache models from Hugging Face.
- **Multimodal.** Text, vision (image → text), and audio (in/out) models all
  load through the same session API.
- **Structured output.** Constrain generation to a GBNF grammar, or one flag
  for guaranteed-valid JSON.
- **Tool calling.** Give the model a set of tool schemas and parse the calls it
  makes back out: format-aware (LFM2 Pythonic, Hermes/Qwen JSON), with an
  optional constrained mode that guarantees a well-formed, correctly-typed call.
  From the CLI and every binding.
- **LoRA adapters & embeddings.** Load LoRA adapters at runtime (GGUF or PEFT
  safetensors), hot-swap / unload per session (applied on CPU, Metal, and wgpu,
  never merged) and pull per-token hidden states out of the engine for
  classifier and embedding heads. From every binding.

## Supported models

Dispatch is on the GGUF `general.architecture` string, so any GGUF matching one
of these architectures loads:

| Architecture | Models | Modalities |
|--------------|--------|------------|
| `lfm2` | Liquid **LFM2 / LFM2.5** (the canonical LeapBundles family) | text, vision, audio |
| `llama` | **LLaMA 2 / 3**, and classic **Mistral 7B** (ships as GGUF arch `llama`) | text |
| `qwen2`, `qwen3` | **Qwen2 / Qwen2.5 / Qwen3** | text |
| `granite` | **IBM Granite 3.x** | text |

Every architecture above runs on **all three compute backends** (CPU, Metal, and
wgpu), with single-token decode and prompt prefill on each. Prefill uses
batched-GEMM (each weight read once for the whole prompt) on every backend and
architecture, including CPU for both LFM2 and the dense transformers, with a
tiled flash-attention path that kicks in for long prompts.

### Modalities

- **Text → text**: every supported architecture.
- **Vision (image → text)**: LFM2-VL models. `CeraEngine` auto-attaches the
  vision encoder; `Session::append_image` runs image → ViT → projector → prefill.
  The ViT encoder runs on the GPU (Metal or wgpu) with a CPU fallback. Verified
  against LFM2.5-VL-450M.
- **Audio (in / out)**: LFM2-Audio (`lfm2-audio-v1`): feed PCM audio in, and
  (with a vocoder) decode audio out.

## Platforms & backends

Cera dispatches to the fastest available backend at runtime (`--device auto`),
or you can pin one:

| Backend | `--device` | Platforms | Notes |
|---------|-----------|-----------|-------|
| **CPU** | `cpu` | everywhere | Scalar reference + **NEON** (aarch64) / **AVX2** (x86_64) kernels; optional **Accelerate/OpenBLAS** via the `blas` feature |
| **Native Metal** | `metal` | macOS, iOS | Hand-written MSL shaders, single-encoder dispatch, GPU argmax |
| **wgpu** | `gpu` | macOS, Linux, Windows, browser | WGSL shaders over **Metal / Vulkan / DX12 / WebGPU** |

`--device auto` uses native Metal on macOS and iOS, and wgpu where a GPU is
available, falling back to CPU otherwise.

### Quantization

Weights run in **Q4_0**, **Q4_1**, **Q8_0**, **Q4_K**, **Q5_K**, and **Q6_K**,
plus dense **F32**. Activations are dynamically quantized to Q8_0 for fast
integer GEMV on CPU.

Those are GGML *tensor* types, and dispatch is per tensor, so what decides
whether a file runs is the mix inside it, not its `Q4_K_M`-style label. K-quant
downloads usually work, but the label is not a guarantee: llama.cpp substitutes
Q5_0 / Q5_1 / IQ4_NL for tensors whose rows are not a multiple of 256, and cera
has no kernel for those. `cera inspect --model <file>` lists the per-tensor types,
and an unsupported one fails at weight resolution naming the tensor rather than
failing anonymously, with the type name too, for the types cera knows about (an
IQ type reports its numeric id).

Backends are not uniform, so a file that runs on one may not run on another. On
native Metal, Q4_1 works as a projection weight but not as `token_embd` /
`output`; `--device auto` falls back to wgpu or CPU on its own, while
`--device metal` reports the gap. F16/BF16
tensors parse, and are dequantized on the LoRA, vision, and audio paths, but the
transformer weight and token-embedding paths have no kernel for them; an
F16-weight LLM is not a supported configuration.

## Language bindings

One Rust core, consumed from many places:

| Target | Crate / package | Consumers |
|--------|-----------------|-----------|
| **Rust** | [`cera`](cera/) | any Rust project (`cargo add cera`) |
| **CLI** | [`cera-cli`](cera-cli/) | the `cera` binary |
| **Kotlin / Swift / Python** | [`cera-ffi`](cera-ffi/) (UniFFI) | JVM, Apple platforms |
| **Android** | [`cera-ffi-kotlin`](cera-ffi-kotlin/) | Android apps (AAR) |
| **iOS / macOS** | [`Package.swift`](Package.swift) (SwiftPM XCFramework) | Apple apps (`.package(url:)`), Metal GPU (Auto: Metal → CPU) |
| **Flutter** | [`cera_ffi_flutter`](cera_ffi_flutter/) | cross-platform apps; ships the native library per platform |
| **Dart (no Flutter)** | [`cera_ffi`](cera_ffi/) | CLI / server; bring your own `cera-ffi` cdylib |
| **Browser / Node** | [`cera-wasm`](cera-wasm/) (`@hyeons-lab/cera-wasm`) | WebAssembly + WebGPU |

A complete SwiftUI example app (streaming chat + embeddings + LoRA) that consumes
the published Swift Package lives in [`examples/CeraChat`](examples/CeraChat/);
it doubles as a real-device Metal validation harness.

## Structured output (GBNF grammars)

Force the model's output to match a grammar, useful for JSON, tool calls, or any
schema. Cera ships a byte-level GBNF engine (mirroring llama.cpp's) that masks the
sampler each step so only grammar-valid tokens can be produced.

```bash
# Guaranteed-valid JSON (bundled grammar)
cera run -m model.gguf -p "List 3 colors as JSON" --json

# Any custom GBNF (inline or @file)
cera run -m model.gguf -p "..." --grammar @schema.gbnf
```

Supports literals, character classes, alternation, grouping, and repetition
(`* + ?` and bounded `{n,m}`). Available from **every binding**, not just the
CLI: in Rust set `GenerateOpts.grammar` to a compiled `Grammar` (`Grammar::parse(gbnf)?`),
while the Kotlin/Swift FFI (`GenerateOpts.grammar`) and browser/Node WASM
(`GenerateOpts.setGrammar(gbnf)`) take the GBNF string directly and compile it
natively, so mobile and web apps get the same guaranteed-valid output.

## Tool calling

Give the model a set of tools (OpenAI "function" schemas) and get the calls it
makes back out as structured data. Cera renders the tools into the model's chat
template and parses tool calls from the reply, **format-aware**, since the wire
format isn't interchangeable: LFM2/LFM2.5 emit **Pythonic** calls
(`[get_weather(city="Paris")]`) while Hermes-style Qwen2.5/Qwen3 emit **JSON**
(`<tool_call>{"name": …, "arguments": {…}}</tool_call>`). The format is detected
from the GGUF architecture.

```bash
# Let the model decide whether/how to call a tool (free text or a call)
cera run -m model.gguf -p "What's the weather in Paris?" \
  --tools '[{"name":"get_weather","description":"Get weather for a city",
             "parameters":{"type":"object",
               "properties":{"city":{"type":"string"}},"required":["city"]}}]'

# Force a well-formed, correctly-typed call (grammar-constrained)
cera run -m model.gguf -p "..." --tools @tools.json --constrain-tools
```

With `--constrain-tools` a **lazy grammar trigger** keeps generation free until
the model starts a tool call, then constrains the call to a valid function name,
valid argument names, and correctly-typed values (JSON-Schema → GBNF). In
`--tools` mode stdout is machine-readable: **only** the JSON array of calls
(`[]` when the model answered in prose); the assistant reply and timing stream to
stderr, so `… --tools tools.json | jq` just works.

Available from **every binding**, not just the CLI: Rust (`cera::tools`),
Kotlin/Swift (`applyChatTemplateWithTools`, `parseToolCalls`, `toolGrammar`,
`detectToolFormat`), and browser/Node WASM (the same names); see each crate's
README for the API surface.

## LoRA adapters & hidden states

**Runtime LoRA.** Load a LoRA adapter, a llama.cpp GGUF (from
`convert_lora_to_gguf`) or a PEFT `.safetensors`, and attach it to a session. The
delta is applied at inference time (`y += scale·B·(A·x)`), **never merged into the
weights**, so the base model stays quantized and adapters hot-swap / unload per
request at ~no cost. Runs on **CPU, Metal, and wgpu** (batched-GEMM prefill +
decode) and is dimension-checked against the model at attach.

**Hidden-states extraction.** Pull the per-token last-layer hidden state
(post-final-RMSNorm, the llama.cpp `--pooling none` vector) straight out of the
engine, reflecting the active adapter. This unblocks classifier / extractor /
embedding heads, e.g. a section router: `LFM2.5` + a `route_section` LoRA + a
small linear head over the mean-pooled hidden state.

Both are available from **every binding**:

```swift
// Swift / Kotlin (UniFFI): same shape on both
let adapters = try LoraAdapters.fromSafetensors(path: p, alpha: nil)
try session.attachLora(adapters: adapters)               // hot-swap-able
let pooled = try session.hiddenStatesMeanPooled(tokens: toks)  // [hidden_size]
```

```js
// Browser / Node (WASM)
const adapters = LoraAdapters.fromSafetensorsBytes(bytes, undefined);
session.attachLora(adapters);
const pooled = session.hiddenStatesMeanPooled(tokens); // Float32Array
```

## TurboQuant KV-cache compression

Cera includes the **first implementation of TurboQuant**
([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Google Research 2025) for
LFM2, compressing the KV cache to **~3 bits/key + ~2 bits/value (~12× vs f32)**
with near-lossless quality and **no calibration**. On a 1.6B LFM2 model at 4K
tokens that's ~192 MB → ~16 MB of KV, with decode staying within ±5% of f32.

Enable it on the CLI:

```bash
cera run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3 --device cpu
cera run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3 --device gpu     # wgpu
cera run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3 --device metal   # native Metal
```

Supported on **all three** backends (CPU, wgpu, and native Metal) for both
decode and chunked prefill. On the GPU backends the compressed cache lives in GPU
buffers and the prefix-cache snapshots are byte-compatible with the CPU's, so all
three write the same `TQK1`/`TQV1` blob format. Two GPU-specific restrictions:
`head_dim` must be a power of two ≤ 128 and a multiple of 32 (every supported
model is 64 or 128), and only the both-sides mode (`tq3`) is available; the
single-sided debug modes `tq3-keys` / `tq3-values` fall back to the backend's
uncompressed KV there (f32 on wgpu, f16 on Metal) with a warning. `--n-keep`
context shift is not supported with any TurboQuant mode on any backend.

See `cera/src/turboquant.rs` for the algorithm (PolarQuant + QJL).

## Quick start

```bash
# Install the CLI (CPU-only build)
cargo install cera-cli --locked

# ...or with a GPU backend
cargo install cera-cli --locked --features metal   # or: --features gpu

# ...or build from source
just release   # optimized LTO build → target/release/cera
```

```bash
# Generate from a local GGUF
cera run --model model.gguf --prompt "Explain quantization in one sentence."

# Auto-download a bundle from Hugging Face and generate
cera run --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0 --prompt "Hello"

# Interactive multi-turn chat (keeps the prefix cache warm across turns)
cera chat --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0

# Pick a GPU
cera run -m model.gguf -p "Hi" --device metal   # or: gpu, cpu, auto
```

Using the library directly (streaming tokens through a sink):

```rust
use cera::{CeraEngine, EngineConfig, GenerateOpts, SessionConfig};

let engine = CeraEngine::from_path("model.gguf", EngineConfig::default())?;
let mut session = engine.new_session(SessionConfig::default());
session.append_text("Once upon a time")?;

let opts = GenerateOpts { max_tokens: 128, ..Default::default() };
let summary = session.generate(&opts, &mut sink)?; // sink: your ModalitySink
```

See the [`cera` crate README](cera/README.md) for the full library API.

## CLI commands

| Command | Purpose |
|---------|---------|
| `run` | One-shot inference: text, optional grammar/JSON or tool calling (`--tools`), plus image/audio input for VL/Audio bundles |
| `chat` | Interactive multi-turn REPL with a persistent KV prefix cache |
| `embed` | Extract last-layer hidden-state embeddings: mean-pooled, or `--per-token` for the full matrix |
| `logits` | Dump next-token logits over the vocabulary (single prefill), handy for cross-backend parity checks |
| `inspect` | Dump a GGUF's metadata, tensor shapes, and resolved backend tier |
| `cpu` | Print the host's CPU backend tier + detected SIMD features (no model needed) |
| `tokenize` | Encode text to token IDs (e.g. to compare against Hugging Face) |
| `bench` | Measure decode/prefill throughput with p10/p50/p90/mean/stddev |
| `list-bundles` | List bundles available on `LiquidAI/LeapBundles` |
| `download-bundles` | Prefetch bundle manifests + model files without loading |

## Tuning

Defaults are chosen to be correct-or-harmless across a wide range of hosts, not
optimal on any one. The knobs below override them.

| Variable | Effect |
|----------|--------|
| `CERA_DECODE_THREADS=<n>` or `CERA_DECODE_THREADS=auto` | Worker count for decode (per-token GEMV). A fixed `<n>` overrides the automatic sizing below (clamped to the detected count); `auto` selects it. |
| `CERA_DECODE_SIZING=off` | Disable model-aware decode sizing (`0` / `false` / `off`, case-insensitive) and fall back to the flat cap (detected perf cores, capped at 12). Use this to A/B the sizing on a new machine. |
| `CERA_DECODE_NARROW=<n>` / `CERA_DECODE_WIDE=<n>` | Override the two widths the sizing picks between. Defaults derive from the physical core count: `physical/2` capped at 12, and `physical + physical/4` capped at 24; the narrow arm never exceeds the wide one. Setting either also forces sizing on where it would otherwise be declined; on a host whose physical core count is undetectable, both must be pinned. |
| `CERA_DECODE_BPD_KB=<n>` | Bytes-per-dispatch threshold (decimal KB) separating the narrow arm from the wide one. Default 2500. Unlike the two above, this does not force sizing on where it is declined; it moves the threshold, it does not pin a width. |
| `CERA_THREADS=<n>` | Overrides the detected performance-core count, which the decode/prefill pools are sized from. Clamped to the number of pinnable cores on hosts that have any, with a warning: past that the surplus workers run unpinned and contend with pinned ones that are spin-waiting (measured 35x slower on a Tensor G4). Not clamped where nothing gets pinned anyway, i.e. hosts with no affinity or `CERA_PIN=0`. |
| `CERA_PREFILL_THREADS=<n>` | Prefill pool width on its own, without moving decode. May reach past the performance cores up to every pinnable core, for sweeping a part where widening might pay; it does not on the parts measured so far. Subject to the same clamp rule as `CERA_THREADS`, and to `CERA_THREADS` itself, which lowers the ceiling it is clamped against. |
| `CERA_PIN=0` | Disables worker affinity pinning, for host apps that manage thread placement themselves. Also lifts the clamps above, since with nothing pinned there is no oversubscription cliff to guard against. |
| `CERA_POOL_STATS=1` | Annotates each `cera bench` run with pool fan-out health: how many dispatches wanted more than one worker, and how many of those silently ran serially because the pool was busy. The counts are exact; the work percentage mixes units across dispatch kinds, so read the counts. |
| `CERA_CPU_TIER=<tier>` | Caps the SIMD tier (e.g. `avx2`, `avx512`). May only downgrade, useful for A/B-ing a kernel path. |
| `CERA_LM_HEAD_NO_GEMM=1` | Puts the LM-head projection in `forward_prefill_logits_all` back on the per-row loop the batched GEMM replaced, so both halves of a speculative-decoding A/B run from the same binary. Measurement lever only: both paths compute the same projection, to within f32 accumulation order. |
| `RUST_LOG=<filter>` | Log level. Defaults to `warn`, which surfaces things like prefill falling back to the slow per-token path. |

### How the decode thread count is chosen

Decode width is sized from the **loaded model**, because there is no single best
value, and, less obviously, model *size* does not predict it. Measured on a
Ryzen AI MAX+ 395 (16 physical / 32 logical), decode tok/s change going from 12
to 20 workers:

| Model | Weights | Bytes/dispatch | 12 → 20 |
|-------|--------:|---------------:|--------:|
| TinyStories-20M Q8_0 | 21 MB | 0.84 MB | −47% |
| SmolLM-135M Q4_0 | 92 MB | 0.51 MB | −11% |
| LFM2.5-350M Q8_0 | 379 MB | 3.83 MB | **+29%** |
| SmolLM-360M Q8_0 | 386 MB | 1.50 MB | **−7%** |
| Llama-3.2-1B Q8_0 | 1321 MB | 10.24 MB | **+38%** |

Note the middle two: near-identical weight bytes, opposite answers. What
separates them is how many **pool dispatches** a token is split across; LFM2
issues 99 per token against SmolLM-360M's 257, so each dispatch amortizes 2.6×
more work against the same fixed barrier cost. Below ~2.5 MB per dispatch decode
is barrier-bound and wants fewer workers; above it decode is bandwidth-bound and
wants more.

cera computes that ratio from the GGUF at load and picks a narrow or wide width
accordingly. A/B against the old flat cap of 12 (same binary, interleaved,
`CERA_DECODE_SIZING=off` vs on, the five models above at two prompt depths)
gave **+19% mean over those 10 cells with no cell slower** (small models gain
too: they get *fewer* than 12 workers). Scored differently, against each model's
own best measured width across the full 12-model set, the rule reaches 98.1% of
peak versus 90.1% for the flat cap; one combo near the threshold
(LFM2.5-350M Q4_0) lands on the narrow arm when its measurement mildly preferred
the wide one, which is the cost that average already includes.

The pool itself is a process-wide singleton, built once on the first decode
dispatch, so the width comes from the **first** model loaded into a process and
stays there for any loaded after it; it does not re-size per load. (The thread
count was already inherited that way before this sizing existed.) Embedders
running several models in one process should pick the knobs below for whichever
model's decode matters, or load that one first.

This is calibrated on one x86 host. The sizing steps aside, leaving the
previous flat-cap behaviour, in two cases. On heterogeneous parts, because
decode there measured best across *all* big cores (that evidence is from ARM
big.LITTLE; x86 hybrid parts decline by the same argument, without the direct
measurement). And on hosts whose physical core count cannot be detected
(Windows, BSD, Intel macOS), where deriving a width from the *logical* count
would overshoot badly. Pinning a width forces it on anyway, for sweeping a
machine it has not been tuned against: one arm is enough on a heterogeneous
host, both are required where the physical core count is undetectable.

**Apple Silicon is sized, not declined.** M-series parts are heterogeneous in
hardware, but cera detects only their P-cores, so neither decline case applies:
an M4 Max gets 12 workers for large models (its full P-core count; there is no
SMT for the wide arm to spend) and 6 for small ones. That narrow arm is the
part of this that has *not* been measured off x86.

If you run one model repeatedly and care about decode latency, it is still
worth measuring: `cera bench --model <path>` with `CERA_DECODE_THREADS` set
sweeps a fixed width, and `CERA_DECODE_SIZING=off` gives you the old default to
compare against. Interleave the arms rather than sweeping; laptop clocks drift
enough to invent a trend that isn't there.

## Other features

- **Speculative decoding**: opt-in greedy speculation with prompt-lookup
  (n-gram) drafting: no draft model, so no extra weight memory. Verifying K
  drafted tokens in one forward amortizes the single weight-read over the
  accepted run, which is where bandwidth-bound decode gets its win. (As of #327
  the LM head is projected for all verified positions in one batched GEMM, so it
  too is read once per round.) Every emitted token is the target's own
  argmax (a poor draft costs acceptance rate, never
  correctness), though a near-tie can land differently than a sequential greedy
  run, since the verifier forwards a different batch shape. Today it engages on
  the **CPU dense (`llama`-family) path only**, on the plain greedy path with an
  uncompressed KV cache, and falls back transparently everywhere else. On the
  CLI it is exposed on `bench` (`--spec`).
- **Streaming & cancellation**: tokens (and audio frames) arrive through a
  `ModalitySink` as they decode; `Session::cancel()` interrupts long prompts
  responsively via chunked prefill.
- **Prefix caching**: warm (in-memory) and cold (on-disk) KV reuse across
  sessions, namespaced by model fingerprint, so repeated prompt prefixes skip
  re-prefill.
- **Chat templates**: Jinja2 (minijinja) rendering straight from GGUF metadata,
  including multimodal (image + text) messages.
- **Context shifting**: RoPE re-rotation with `n_keep` prefix pinning, on CPU
  and GPU, keeps generation going past the context window.
- **Built-in BPE tokenizer**: vocab, merges, and special tokens loaded directly
  from GGUF; no external tokenizer files.

## Architecture

Cera is a Cargo workspace. The core library does GGUF parsing, quantization, the
compute backends, the models, and the tokenizer; everything else is a thin
adapter over it.

- **[`cera`](cera/)**: core library
- **[`cera-cli`](cera-cli/)**: CLI binary (clap)
- **[`cera-ffi`](cera-ffi/)**: UniFFI bindings (Kotlin / Swift / Python)
- **[`cera-ffi-kotlin`](cera-ffi-kotlin/)**: Android packaging (AAR)
- **[`cera_ffi`](cera_ffi/)** · **[`cera_ffi_flutter`](cera_ffi_flutter/)**: Dart bindings, and the Flutter plugin that ships a native library for them
- **[`cera-wasm`](cera-wasm/)**: `wasm-bindgen` browser / Node bindings
- **[`cera-parity`](cera-parity/)**: cross-binding parity harness (runs one prompt through every binding and reports drift)

See the [`cera` crate README](cera/README.md) for the module layout, the model
trait, and the inference loop.

## Performance

Cera is competitive with (and on decode, often faster than) llama.cpp on the
LFM2 family. On an M1 Max with Q4_0 weights, the native Metal backend decodes
roughly **2× faster than llama.cpp** across tested VL and Audio models; prefill
leads at short prompts and trails at long ones.

On the cross-platform **wgpu** backend, decode is **1.78x faster than it was**
on the same model and machine (LFM2-VL-450M Q4_0, M1 Max: 63.4 to 112.8 tok/s),
from four changes: removing register spills in the quantized GEMV kernels,
merging the LFM2 conv block into one compute pass, deleting two per-token GPU
round trips that carried almost no work, and running the LM head on the weight as
GGUF stores it instead of a dequantized f16 copy (which also gives back ~79 MB of
VRAM on a 230M model). A fifth change since then, word-loading the Q6_K GEMV
instead of reading it byte-at-a-time, measured **+11.6%** decode (109.5 to
122.1 tok/s, ABBA-ordered) on **LFM2.5-230M-Q4_K_M**, a different model from the
1.78x row above, so the two do not multiply. That gain is entirely conditional on
the change before it: measured *without* #320, the same kernel was flat (110.0 vs
110.5 tok/s), because the only Q6_K matrix in the decode path was one `ffn_down`
of 42. #320 makes the tied Q6_K embedding the LM head, and that is what the
faster loads then pay off on. The per-kernel breakdown, and what did *not* work,
are in [`benchmarks/BASELINE.md`](benchmarks/BASELINE.md).

On CPU, rows dispatch through a persistent, affinity-pinned threadpool with
dynamic chunk-stealing rather than a per-GEMV fork-join. This fixes the
multi-core decode collapse on Android big.LITTLE and scales decode across the
performance cores (Tensor G5, LFM2 Q4_0, warm: CPU decode matches or beats
llama.cpp). Thread placement and count auto-detect per device; the `CERA_*`
override knobs are documented under
**[*CPU threading & tuning*](cera/README.md#cpu-threading--tuning)**.

Detailed methodology, per-model tables (decode + prefill vs llama.cpp), the
Accelerate/AMX BLAS results, and the backend optimization notes live in
**[`benchmarks/README.md`](benchmarks/README.md)**. Numbers there are tagged with
the commit and device they were measured on; some sections are older than
others, and the Android GPU row in particular predates the wgpu work above.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.

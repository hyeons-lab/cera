# Cera

**A Rust-native LLM inference engine.** Load a GGUF model and run it locally —
on your laptop's CPU, an Apple GPU, a cross-platform Vulkan/DX12 GPU, a phone,
or in the browser — from a single dependency-free core.

> **Note:** Cera is a learning experiment — built to explore LLM inference
> internals, GGUF parsing, quantization, and SIMD/GPU compute in Rust. It's
> functional and reasonably fast, but not intended for production use.

## Why Cera

- **No Python, no runtime.** Pure Rust. The CLI is a single binary; the
  library has zero required system dependencies on a default build.
- **Runs everywhere.** The same core drives a desktop CLI, Android/iOS apps
  (via UniFFI), and the browser (via WebAssembly). Pick CPU or GPU at runtime.
- **Loads standard GGUF.** Point it at a `.gguf` file, a
  [LeapBundles](https://huggingface.co/LiquidAI/LeapBundles) manifest, or a
  bundle id — it can auto-download and cache models from Hugging Face.
- **Multimodal.** Text, vision (image → text), and audio (in/out) models all
  load through the same session API.
- **Structured output.** Constrain generation to a GBNF grammar — or one flag
  for guaranteed-valid JSON.

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
batched-GEMM (each weight read once for the whole prompt) on the GPU backends for
every architecture, and on CPU for LFM2; CPU prefill for the dense transformers
is currently sequential per-token.

### Modalities

- **Text → text** — every supported architecture.
- **Vision (image → text)** — LFM2-VL models. `CeraEngine` auto-attaches the
  vision encoder; `Session::append_image` runs image → ViT → projector → prefill.
  The ViT encoder runs on the GPU (Metal or wgpu) with a CPU fallback. Verified
  against LFM2.5-VL-450M.
- **Audio (in / out)** — LFM2-Audio (`lfm2-audio-v1`): feed PCM audio in, and
  (with a vocoder) decode audio out.

## Platforms & backends

Cera dispatches to the fastest available backend at runtime (`--device auto`),
or you can pin one:

| Backend | `--device` | Platforms | Notes |
|---------|-----------|-----------|-------|
| **CPU** | `cpu` | everywhere | Scalar reference + **NEON** (aarch64) / **AVX2** (x86_64) kernels; optional **Accelerate/OpenBLAS** via the `blas` feature |
| **Native Metal** | `metal` | macOS | Hand-written MSL shaders, single-encoder dispatch, GPU argmax |
| **wgpu** | `gpu` | macOS, Linux, Windows, browser | WGSL shaders over **Metal / Vulkan / DX12 / WebGPU** |

`--device auto` uses native Metal on macOS and wgpu where a GPU is available,
falling back to CPU otherwise.

### Quantization

Weights load in **Q4_0**, **Q8_0**, **Q6_K**, and **Q4_K_M**; dense **F32 / F16 /
BF16** are also supported. Activations are dynamically quantized to Q8_0 for fast
integer GEMV on CPU.

## Language bindings

One Rust core, consumed from many places:

| Target | Crate / package | Consumers |
|--------|-----------------|-----------|
| **Rust** | [`cera`](cera/) | any Rust project (`cargo add cera`) |
| **CLI** | [`cera-cli`](cera-cli/) | the `cera` binary |
| **Kotlin / Swift / Python** | [`cera-ffi`](cera-ffi/) (UniFFI) | JVM, Apple platforms |
| **Android** | [`cera-ffi-kotlin`](cera-ffi-kotlin/) | Android apps (AAR) |
| **Flutter / Dart** | [`cera-ffi-flutter`](cera-ffi-flutter/) | cross-platform mobile |
| **Browser / Node** | [`cera-wasm`](cera-wasm/) (`@hyeons-lab/cera-wasm`) | WebAssembly + WebGPU |

## Structured output (GBNF grammars)

Force the model's output to match a grammar — useful for JSON, tool calls, or any
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
natively — so mobile and web apps get the same guaranteed-valid output.

## TurboQuant KV-cache compression

Cera includes the **first implementation of TurboQuant**
([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Google Research 2025) for
LFM2 — compressing the KV cache to **~3 bits/key + ~2 bits/value (~12× vs f32)**
with near-lossless quality and **no calibration**. On a 1.6B LFM2 model at 4K
tokens that's ~192 MB → ~16 MB of KV, with decode staying within ±5% of f32.

Enable it on the CLI (CPU backend):

```bash
cera run -m lfm2.gguf -p "Hello" --kv-cache-keys tq3 --device cpu
```

> Currently **CPU-only** (the Metal/wgpu backends fall back to f32 KV). See the
> [`cera` crate README](cera/README.md) and `cera/src/turboquant.rs` for the
> algorithm (PolarQuant + QJL) and the full compression/quality tables.

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
| `run` | One-shot inference — text, optional grammar/JSON, plus image/audio input for VL/Audio bundles |
| `chat` | Interactive multi-turn REPL with a persistent KV prefix cache |
| `inspect` | Dump a GGUF's metadata, tensor shapes, and resolved backend tier |
| `cpu` | Print the host's CPU backend tier + detected SIMD features (no model needed) |
| `tokenize` | Encode text to token IDs (e.g. to compare against Hugging Face) |
| `bench` | Measure decode/prefill throughput with p10/p50/p90/mean/stddev |
| `list-bundles` | List bundles available on `LiquidAI/LeapBundles` |
| `download-bundles` | Prefetch bundle manifests + model files without loading |

## Other features

- **Streaming & cancellation** — tokens (and audio frames) arrive through a
  `ModalitySink` as they decode; `Session::cancel()` interrupts long prompts
  responsively via chunked prefill.
- **Prefix caching** — warm (in-memory) and cold (on-disk) KV reuse across
  sessions, namespaced by model fingerprint, so repeated prompt prefixes skip
  re-prefill.
- **Chat templates** — Jinja2 (minijinja) rendering straight from GGUF metadata,
  including multimodal (image + text) messages.
- **Context shifting** — RoPE re-rotation with `n_keep` prefix pinning, on CPU
  and GPU, keeps generation going past the context window.
- **Built-in BPE tokenizer** — vocab, merges, and special tokens loaded directly
  from GGUF; no external tokenizer files.

## Architecture

Cera is a Cargo workspace. The core library does GGUF parsing, quantization, the
compute backends, the models, and the tokenizer; everything else is a thin
adapter over it.

- **[`cera`](cera/)** — core library
- **[`cera-cli`](cera-cli/)** — CLI binary (clap)
- **[`cera-ffi`](cera-ffi/)** — UniFFI bindings (Kotlin / Swift / Python)
- **[`cera-ffi-kotlin`](cera-ffi-kotlin/)** · **[`cera-ffi-flutter`](cera-ffi-flutter/)** — Android / Flutter packaging
- **[`cera-wasm`](cera-wasm/)** — `wasm-bindgen` browser / Node bindings
- **[`cera-parity`](cera-parity/)** — cross-binding parity harness (runs one prompt through every binding and reports drift)

See the [`cera` crate README](cera/README.md) for the module layout, the model
trait, and the inference loop.

## Performance

Cera is competitive with — and on decode, often faster than — llama.cpp on the
LFM2 family. On an M1 Max with Q4_0 weights, the native Metal backend decodes
roughly **2× faster than llama.cpp** across tested VL and Audio models; prefill
leads at short prompts and trails at long ones.

Detailed methodology, per-model tables (decode + prefill vs llama.cpp), the
Accelerate/AMX BLAS results, and the backend optimization notes live in
**[`benchmarks/README.md`](benchmarks/README.md)**.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.

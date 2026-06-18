# cera

Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

> **Note:** This project is a learning experiment — built to explore LLM
> inference internals, GGUF parsing, quantization, and SIMD/GPU compute in Rust.
> Not intended for production use. See the [project README](https://github.com/hyeons-lab/cera)
> for benchmarks and design notes.

`cera` is the core library: GGUF loading, a quantized CPU kernel stack
(AVX2/AVX-512, NEON dotprod/i8mm) with optional wgpu GPU and BLAS backends, a
stateful session API with prefix caching, and a streaming token sink. It powers
the [`cera-cli`](https://github.com/hyeons-lab/cera/tree/main/cera-cli) CLI, the
[`cera-ffi`](https://github.com/hyeons-lab/cera/tree/main/cera-ffi) mobile
bindings, and [`cera-wasm`](https://github.com/hyeons-lab/cera/tree/main/cera-wasm).

## Install

```toml
[dependencies]
cera = "0.1"
```

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
Verified against LFM2.5-VL-450M; the ViT encode runs on CPU (no GPU/Metal path
yet).

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

    let mut session = engine.new_session(SessionConfig::default());
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

## Feature flags

Default-on features keep desktop/CLI builds full-featured; turn them off to
shrink the crate for `wasm32-unknown-unknown` or embedded targets
(`--no-default-features`).

| Feature | Default | What it adds |
|---------|:------:|--------------|
| `parallel` | ✅ | Rayon-parallel kernels |
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

## License

Apache-2.0 OR MIT.

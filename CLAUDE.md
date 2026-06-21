# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
just build              # debug build
just release            # optimized release build (LTO thin, stripped)
just test               # run all tests
just fmt                # format code
just clippy             # lint
just ci                 # full CI check: fmt + clippy + test

# Single test or module
cargo test -p cera -- <test_name>
cargo test -p cera <module>::tests       # e.g. quant::tests, gguf::tests

# CLI commands (working features)
cargo run -p cera-cli -- inspect <path.gguf>
cargo run -p cera-cli -- tokenize <path.gguf> "text"
```

**Always run `cargo fmt` before committing.** CI enforces `cargo fmt --check` and will fail on unformatted code.

## Architecture

Five-crate Cargo workspace:

- **`cera`** — core library (all inference logic)
- **`cera-cli`** — binary (clap CLI that dispatches to `cera`)
- **`cera-ffi`** — UniFFI bindings exposing the engine to Kotlin/Swift/etc.
- **`cera-parity`** — cross-language parity harness (not published)
- **`cera-wasm`** — wasm-bindgen browser/Node bindings

### GGUF Parsing (`gguf.rs`)

GGUF files are memory-mapped via `memmap2`. Two access patterns:
- `get_tensor(name)` — copies data from mmap into an owned `Tensor`
- `tensor_data(name)` — returns a zero-copy `&[u8]` slice into the mmap

Both validate offsets with checked arithmetic (`checked_add`, `usize::try_from`).

### Tensor & Quantization (`tensor.rs`, `quant.rs`)

`DType` enum covers dense types (F32, F16, BF16) and quantized types (Q4_0, Q4KM, Q8_0). Each quantized format has:
- A block struct (e.g. `BlockQ4_0`, `BlockQ4KM`, `BlockQ8_0`)
- `dequantize_*()` — block/row to f32
- `vec_dot_*()` — dot product without full dequantization

### Compute Backends (`backend/`)

Four tiers with runtime dispatch:
1. **`cpu.rs`** — scalar reference implementations operating on raw `&[f32]` slices (no Tensor in the hot path)
2. **`simd.rs`** — NEON (aarch64) and AVX2 (x86_64) optimized `vec_dot` kernels with compile-time + runtime dispatch
3. **`wgpu.rs`** — cross-platform GPU backend (`gpu` feature). Functional: WGSL GEMV/GEMM kernels (incl. quantized `gemm_q8_0`/`gemm_q4_0`), the ViT vision encoder, and LFM2 prefill/decode
4. **`metal/`** — native Apple Metal backend (`metal` feature, macOS) with MSL kernels — the `Auto`-preferred GPU backend, used by LFM2 and the ViT vision encoder

### Models (`model/`)

`Model` trait with `forward()` and `config()`. `ModelConfig` supports per-layer `BlockType` (Attention or GatedConv) for hybrid architectures like LFM2. LLaMA and LFM2 model implementations are Phase 3-4 work.

### Tokenizer (`tokenizer.rs`)

Self-contained BPE tokenizer that loads vocab, merges, and special tokens directly from GGUF metadata. Chat template rendering via `minijinja`.

## Conventions

- Edition 2024, MSRV 1.85
- `.cargo/config.toml` sets native CPU feature flags per target architecture
- `wgpu` is wired into `cera/Cargo.toml` as an optional dependency behind the `gpu` feature (Metal backend behind `metal`)
- Error handling: `anyhow` with `ensure!` / `with_context()` / `bail!`
- Release profile: LTO thin, single codegen unit, stripped symbols

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
cargo run -p cera-cli -- inspect --model <path.gguf>
cargo run -p cera-cli -- tokenize --model <path.gguf> --text "text"
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

`DType` enum covers dense types (F32, F16, BF16) and quantized types (Q4_0, Q4_1, Q4KM, Q5KM, Q8_0, Q6K) — the `Q4KM`/`Q5KM` names are historical; they map from the `Q4_K`/`Q5_K` *tensor* types, not the `_M` file types. Each quantized format has:
- A block struct (e.g. `BlockQ4_0`, `BlockQ4KM`, `BlockQ8_0`)
- `dequantize_*()` — block/row to f32
- `vec_dot_*()` — dot product without full dequantization

### Compute Backends (`backend/`)

Four tiers with runtime dispatch:
1. **`cpu.rs`** — scalar reference implementations operating on raw `&[f32]` slices (no Tensor in the hot path)
2. **`simd.rs`** — NEON (aarch64) and AVX2 (x86_64) optimized `vec_dot` kernels with compile-time + runtime dispatch
3. **`wgpu.rs`** — cross-platform GPU backend (`gpu` feature). Functional: WGSL GEMV/GEMM kernels (incl. quantized `gemm_q8_0`/`gemm_q4_0`), the ViT vision encoder, and decode + batched prefill for both LFM2 and dense transformers (llama/qwen2/qwen3/granite)
4. **`metal/`** — native Apple Metal backend (`metal` feature, macOS + iOS) with MSL kernels — the `Auto`-preferred GPU backend, running decode + batched prefill for LFM2 and dense transformers plus the ViT vision encoder. The shipped `CeraFFI.xcframework` (SwiftPM) is Metal-enabled across all three arm64 slices (iOS device, iOS Simulator, native macOS)

### Models (`model/`)

`Model` trait with `forward()` and `config()`. `ModelConfig` supports per-layer `BlockType` (Attention or GatedConv) for hybrid architectures like LFM2. Both model families are implemented and inference-complete across all backends (CPU, wgpu, Metal). Decode works everywhere; prefill uses batched GEMM on every backend — all GPU backends and CPU for both LFM2 and the dense transformers (`llama.rs::forward_prefill`, sharing the GEMM helpers in `transformer.rs`), with a tiled flash-attention path for long prompts:
- **LFM2 / LFM2.5** (`lfm2` arch, hybrid attention + gated-conv) — plus vision (LFM2-VL) and audio (LFM2-Audio) modalities.
- **Dense transformers** (`llama` arch — also Qwen2/3, classic Mistral, Granite).

TurboQuant KV-cache compression runs on all three backends — CPU (`Lfm2Model`), wgpu (`GpuLfm2Model`), and native Metal (`MetalLfm2Model`) — for decode and chunked prefill, with cross-backend-compatible prefix-cache snapshots. Both GPU host mirrors (`model/gpu_turboquant.rs`, `model/metal_turboquant.rs`) are pinned to the CPU implementation by oracle suites (`tests/wgpu_turboquant_oracle.rs`, `tests/metal_turboquant_oracle.rs`) that assert byte equality on the packed layout. The GPU path needs `head_dim` a power of two ≤ 128 and a multiple of 32, and only supports compressing keys *and* values (the single-sided `tq3-keys` / `tq3-values` debug modes fall back to the backend's uncompressed KV — f32 on wgpu, f16 on Metal). Compression is chosen once per model instance: a second session asking for a different mode gets `CeraError::KvCompressionConflict` rather than a cache the kernels don't match. `n_keep` shift is unsupported on compressed caches everywhere: re-rotating packed K would need the raw vectors, and the gate trips whenever *either* side is compressed, so the single-sided modes are excluded too. The per-phase profiling paths (`CERA_PROFILE=timing`/`gpu`, `forward_prefill_profiled`) assert against a compressed cache instead of silently measuring the uncompressed kernels.

### Tokenizer (`tokenizer.rs`)

Self-contained BPE tokenizer that loads vocab, merges, and special tokens directly from GGUF metadata. Chat template rendering via `minijinja`.

## Conventions

- Edition 2024, MSRV 1.94 (the NEON f16 `vcvt_f32_f16` KV-cache widen needs 1.94; `avx512` `_mm512_*` needed 1.89)
- `.cargo/config.toml` sets native CPU feature flags per target architecture
- `wgpu` is wired into `cera/Cargo.toml` as an optional dependency behind the `gpu` feature (Metal backend behind `metal`)
- Error handling: `anyhow` with `ensure!` / `with_context()` / `bail!`
- Release profile: LTO thin, single codegen unit, stripped symbols

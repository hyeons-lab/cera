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

Six-crate Cargo workspace (plus multi-language SDK packages):

- **`cera`**: core library (all inference logic)
- **`cera-cli`**: binary (clap CLI that dispatches to `cera`)
- **`cera-client`**: Rust SDK re-exporting the `cera` engine API
- **`cera-ffi`**: UniFFI bindings exposing the engine to Kotlin/Swift/Python/Dart
- **`cera-parity`**: cross-language parity harness (not published)
- **`cera-wasm`**: `wasm-bindgen` browser/Node bindings (CPU + WebGPU + OPFS bundle store)
- **`cera-ffi-kotlin`**, **`cera_ffi`**, **`cera_ffi_flutter`**: Gradle (JVM + Android), pure-Dart, and Flutter SDK packages

### GGUF Parsing (`gguf.rs`)

GGUF files are memory-mapped via `memmap2` (or parsed header-only for streaming weight sources). Access patterns include:
- `get_tensor(name)`: copies data from mmap into an owned `Tensor`
- `tensor_data(name)`: returns a zero-copy `&[u8]` slice into the mmap
- `from_header_bytes(data, file_size)`: parses metadata and tensor descriptors from a header prefix while validating against `file_size`, releasing the header buffer afterwards so streaming backends (`OpfsGpuWeightSource`) hold zero raw weight bytes in memory

All readers validate offsets and tensor byte counts through shared checked arithmetic (`checked_add`, `usize::try_from`, `checked_tensor_byte_size`).

### Tensor & Quantization (`tensor.rs`, `quant.rs`)

`DType` enum covers dense types (F32, F16, BF16) and quantized types (Q4_0, Q4_1, Q4KM, Q5KM, Q8_0, Q6K); the `Q4KM`/`Q5KM` names are historical; they map from the `Q4_K`/`Q5_K` *tensor* types, not the `_M` file types. Each quantized format has:
- A block struct (e.g. `BlockQ4_0`, `BlockQ4KM`, `BlockQ8_0`)
- `dequantize_*()`: block/row to f32
- `vec_dot_*()`: dot product without full dequantization

### Compute Backends (`backend/`)

Five tiers with runtime dispatch:
1. **`cpu.rs`**: scalar reference implementations operating on raw `&[f32]` slices (no Tensor in the hot path)
2. **`simd/`**: NEON/dotprod/i8mm (aarch64), AVX2/AVX-512 (x86_64), and `simd128` (`wasm32`) optimized `vec_dot` and tiled GEMM kernels with compile-time + runtime dispatch
3. **`wgpu.rs`**: cross-platform GPU backend (`gpu` feature, Vulkan/Metal/DX12/BrowserWebGpu). Functional: WGSL/SPIR-V GEMV/GEMM kernels, the ViT vision encoder, and decode + batched prefill for LFM2, routed `lfm2moe`, and dense transformers (`llama`, `qwen2`, `qwen3`, `granite`, `minicpm`/`minicpm5`, `nanbeige`, `phi3`/`phi`, `mistral3`/`ministral3`)
4. **`metal/`**: native Apple Metal backend (`metal` feature, macOS + iOS) with MSL kernels; the `Auto`-preferred GPU backend on Apple silicon, running decode + batched prefill for LFM2, routed `lfm2moe`, and dense transformers, plus the ViT vision encoder. The shipped `CeraFFI.xcframework` (SwiftPM) is Metal-enabled across all three arm64 dynamic framework slices (`ios-arm64`, `ios-arm64-simulator`, `macos-arm64`)
5. **`hexagon/`**: Qualcomm Hexagon NPU backend (`hexagon` feature, Android `aarch64-linux-android` on v73/v75/v79/v81/v85 cDSP) with HVX kernels via FastRPC; preferred by `Auto` (and `Npu`/`Hexagon`) when available, supporting LFM2, routed `lfm2moe` (with paged routed experts), dense transformers, ViT vision encoder, and FastConformer audio encoders (Sortformer / Nemotron-3 diarization)

### Models (`model/`)

`Model` trait with `forward()` and `config()`. `ModelConfig` supports per-layer `BlockType` (Attention or GatedConv) for hybrid architectures like LFM2. Decode works everywhere; prefill uses batched GEMM across CPU, wgpu, Metal, and Hexagon:
- **LFM2 / LFM2.5** (`lfm2` arch, hybrid attention + gated-conv), plus vision (LFM2-VL), audio (LFM2-Audio), and bidirectional `d1` decision models.
- **Routed mixture-of-experts** (`lfm2moe` arch, LFM2 with per-layer expert FFNs; LFM2.5-8B-A1B, plus `bailingmoe3` / Ling 3.0 Tiny on CPU). Supported on CPU, wgpu, Metal, and Hexagon NPU.
- **LFM3** (`lfm3` arch): `LfmModel` with `Lfm3Extras` (`model/lfm2.rs`) on CPU, and `HexagonLfmModel` on the Hexagon NPU (op details and the S25 Ultra measurements are in `docs/HEXAGON_NPU.md`, row T3b). Over LFM2 it adds sandwich norms (`post_attention_norm`, `post_ffw_norm`), a per-head sigmoid gate on the attention output, per-layer sliding windows with their own RoPE base, partial rotary on global layers through `rope_freqs.weight` (`cpu::rope_neox_freq_factors`), a 7-tap conv, and an untied `output.weight` on some checkpoints. On CPU its attention runs the per-token path in prefill (the batched attention loop knows none of this; the NPU dispatches it per chunk); conv and FFN are batched. The wgpu and Metal loaders refuse the arch; the loader refuses bidirectional files; LoRA is refused, a TurboQuant request falls back to the uncompressed cache with a warning from `Session::new` (`kv_cache::turboquant_capable`), and the `n_keep` shift is unsupported (overflow returns `ContextOverflow`). Verified three ways: `tests/convert_lfm3_layout.rs` pins the CPU graph without python (a deterministic F32 conversion, tied and untied head, against pinned logits; this is what CI runs), `tests/lfm3_oracle_parity.rs` compares a synthetic f32 model with committed llama.cpp values (needs python3 with `numpy` and `gguf`; CI does not install them, so it skips there unless `CERA_REQUIRE_ORACLE=1`) plus opt-in real-checkpoint gates, and the Hexagon golden tests pin the emitted ops. `cera::convert` writes the `lfm3` GGUF (sandwich-norm and gate tensor names, the generated `rope_freqs.weight`, sliding-window metadata; also from the audio checkpoint's `lfm.` layout), checked tensor-for-tensor against llama.cpp's converter.
- **Dense & hybrid transformers** (`llama`, `qwen2`, `qwen3`, `qwen35`, `granite`, `minicpm`/`minicpm5`, `nanbeige`, `phi3`/`phi`, `mistral3`/`ministral3`, `gemma2`/`gemma4`, `olmo2`/`olmo3`).
- **Speech & audio models**: OpenAI Whisper (`model/whisper/`), Silero VAD v5 (`vad.rs`), Keyword Spotting (`hotword.rs`), and Streaming Sortformer / Nemotron-3 speaker diarization (`model/sortformer/`, `model/nemotron3_diar/`), composable via `AudioPipeline`.

The default KV cache is f16 (`SessionConfig::default()` is `KvCompression::F16`; the CLI's `--kv-cache-keys auto` does the same), resolved in `Session::new`: a model whose `f16_kv_supported()` is false (wgpu, Metal, Hexagon, which keep their own cache) is configured with `KvCompression::None`, and wasm32 defaults to `None`. `KvCompression::None` always means the backend's own full-precision cache.

TurboQuant KV-cache compression runs on all three backends (not for `lfm3`, whose sliding-window layers the compressed read does not mask, so it falls back to the uncompressed cache): CPU (`LfmModel`), wgpu (`GpuLfmModel`), and native Metal (`MetalLfmModel`), for decode and chunked prefill, with cross-backend-compatible prefix-cache snapshots. Both GPU host mirrors (`model/gpu_turboquant.rs`, `model/metal_turboquant.rs`) are pinned to the CPU implementation by oracle suites (`tests/wgpu_turboquant_oracle.rs`, `tests/metal_turboquant_oracle.rs`) that assert byte equality on the packed layout. The GPU path needs `head_dim` a power of two ≤ 128 and a multiple of 32, and only supports compressing keys *and* values (the single-sided `tq3-keys` / `tq3-values` debug modes fall back to the backend's uncompressed KV: packed f16 halves on wgpu (no SHADER_F16 storage on Adreno, so u32 words + `unpack2x16float`), native f16 on Metal). Compression is chosen once per model instance: a second session asking for a different mode gets `CeraError::KvCompressionConflict` rather than a cache the kernels don't match. `n_keep` shift is unsupported on compressed caches everywhere: re-rotating packed K would need the raw vectors, and the gate trips whenever *either* side is compressed, so the single-sided modes are excluded too. The per-phase profiling paths (`CERA_PROFILE=timing`/`gpu`, `forward_prefill_profiled`) assert against a compressed cache instead of silently measuring the uncompressed kernels.

### Tokenizer (`tokenizer.rs`)

Self-contained BPE tokenizer that loads vocab, merges, and special tokens directly from GGUF metadata. Chat template rendering via `minijinja`.

## Conventions

- Edition 2024, MSRV 1.94 (the NEON f16 `vcvt_f32_f16` KV-cache widen needs 1.94; `avx512` `_mm512_*` needed 1.89)
- `.cargo/config.toml` sets native CPU feature flags per target architecture
- `wgpu` is wired into `cera/Cargo.toml` as an optional dependency behind the `gpu` feature (Metal backend behind `metal`)
- Error handling: `anyhow` with `ensure!` / `with_context()` / `bail!`
- Release profile: LTO thin, single codegen unit, stripped symbols

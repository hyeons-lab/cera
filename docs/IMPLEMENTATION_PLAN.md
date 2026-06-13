# Cera — Implementation Plan

A Rust-native LLM inference engine. Load a GGUF, generate text, make it fast.

---

## Implementation Status (updated 2026-06-13)

V1 is complete, and the project has since grown well beyond the original
roadmap into **multimodal** territory (vision, audio, TTS) that this plan never
anticipated. The status legend below uses ✅ done · 🟡 partial · ⬜ not started.

**V1 (Phases 0–6):** ✅ pipeline complete, 🟡 Phase 4 architecture coverage
partial. CPU (SIMD + runtime feature dispatch + BLAS), wgpu GPU backend, Metal
backend, GGUF parser, BPE tokenizer, sampler, KV cache, generation engine, HF
bundle download, interactive chat TUI, and bench command are all shipped — the
end-to-end inference pipeline is done. The one open item is **Phase 4 model
coverage**: architecture dispatch currently handles `lfm2`, `qwen2`, `qwen3`
(Qwen runs on the LLaMA code path), but the bare `llama` arch string and the
Mistral / Gemma / Phi3 variants named in Phase 4 are **not yet wired** — the
shared code path exists but those arch strings aren't dispatched. GPU (wgpu) and
Metal forward passes currently support **LFM2 only**; Qwen runs on CPU.

**Beyond the V1 plan (off-roadmap, shipped):**
- LFM2-VL **vision** encoder + preprocessor (image → embeddings)
- LFM2-Audio **audio** encoder/decoder + preprocessor (PCM → embeddings, ASR)
- **TTS** generation
- **WASM** build (threaded via wasm-bindgen-rayon + wgpu-on-wasm) — this is V2.2
- Kotlin Multiplatform FFI (`cera-ffi-kotlin`: android + jvm) — this is V2.14

**V2 status at a glance:**

| Item | Status | Notes |
|------|--------|-------|
| V2.1 Server + continuous batching | ⬜ | no HTTP server, KV cache still contiguous (not paged) |
| V2.2 Browser / WASM | ✅ | `cera-wasm`, threads, wgpu-on-wasm |
| V2.3 Structured output (GBNF) | ⬜ | |
| V2.4 KV cache serialization (.lmkv) | ⬜ | |
| V2.5 Prefix caching (radix) | ⬜ | |
| V2.5b TurboQuant KV compression | ✅ | `cera/src/turboquant.rs` |
| V2.6 More quant formats | 🟡 | Q6_K added; Q2/Q3/Q5_K, IQ, GPTQ, AWQ, FP8 remain |
| V2.7 Per-shape kernel tuning | ⬜ | no `cera tune` / autotune |
| V2.8 Speculative decoding | ⬜ | |
| V2.9 LoRA adapters | ⬜ | |
| V2.10 MoE support | ⬜ | |
| V2.11 Multi-GPU | ⬜ | |
| V2.12 CUDA backend | ⬜ | |
| V2.13 Python (PyO3) bindings | ⬜ | |
| V2.14 Kotlin Multiplatform bindings | ✅ | `cera-ffi-kotlin` (android + jvm) |
| V2.17 Flutter / Dart bindings | 🟡 | `cera-ffi-flutter` — sync API + token streaming verified end-to-end; `*Async` stubbed |
| V2.15 Vision (LFM2-VL) | ✅ | off-roadmap; core shipped, CPU-only encode |
| V2.16 Audio + TTS (LFM2-Audio) | ✅ | off-roadmap; core shipped, Metal-only decode accel |
| V2.17 Flutter / Dart bindings | 🟡 | `cera-ffi-flutter` — sync engine API verified end-to-end; streaming/async stubbed |

**Tally:** original V2 — 3 done (2.2, 2.5b, 2.14), 2 partial (2.6, 2.17), 11 remaining.
Plus 2 off-roadmap multimodal tracks shipped (V2.15 Vision, V2.16 Audio/TTS).
The largest untouched buckets are the **production server stack** (2.1/2.4/2.5)
and **decode-speed work** (2.7/2.8).

---

## Guiding Principles

1. **Build for the hard case first.** LFM2's hybrid conv+attention architecture is more complex than LLaMA. If the abstractions handle LFM2, every pure-transformer model falls out for free.
2. **Two crates, not nine.** `cera` (library) and `cera-cli` (binary). Split later when API boundaries are stable. Every additional crate is compile-time overhead and API surface to maintain.
3. **No CUDA in v1.** wgpu gives us Vulkan, Metal, D3D12, and WebGPU from one set of WGSL shaders. Accept the 10-20% gap vs cuBLAS on datacenter GPUs. Add CUDA as a v2 backend if demand warrants.
4. **Two quant types, not twenty.** Q4_K_M and Q8_0 cover >90% of models people actually download. Each quant type requires a dequant kernel × every backend. Expand later.
5. **Own the tokenizer.** Write a minimal BPE implementation (~300 lines) instead of pulling in the HF `tokenizers` crate (15+ deps, doesn't compile to WASM). LFM2's byte-level BPE with 65K vocab is simple.
6. **Correctness first, then speed.** Naive implementations → verify against llama.cpp → then optimize with SIMD/GPU. Never optimize unverified code.

---

# V1 — "Load a GGUF, generate text, make it fast"

**Target: 6-8 weeks.** One developer + Claude Code.

**End state:** `cera run -m LFM2.5-1.2B-Q4_K_M.gguf -p "Hello"` generates coherent text at 15-30+ tok/s on CPU with SIMD, 40+ tok/s on GPU via wgpu. Supports LFM2 and LLaMA-family models. Single static binary, no Python, no runtime dependencies.

---

## Phase 0: Scaffold ✅
**Time: 1 day**

```
0.1  Create the workspace:

     cera/
     ├── Cargo.toml              # workspace root
     ├── cera/                  # library crate (everything lives here)
     │   ├── Cargo.toml
     │   └── src/
     │       ├── lib.rs
     │       ├── tensor.rs       # Tensor types, dtypes, storage
     │       ├── quant.rs        # Q4_0, Q4_K_M, Q8_0 block dequantization
     │       ├── gguf.rs         # GGUF file parser
     │       ├── tokenizer.rs    # Minimal BPE tokenizer
     │       ├── sampler.rs      # Sampling strategies
     │       ├── backend/
     │       │   ├── mod.rs      # Backend trait
     │       │   ├── cpu.rs      # CPU compute (SIMD)
     │       │   ├── simd.rs     # SIMD-optimized kernels (NEON, AVX2)
     │       │   └── wgpu.rs     # wgpu compute (GPU)
     │       ├── model/
     │       │   ├── mod.rs      # Model trait + loader dispatch
     │       │   ├── lfm2.rs     # LFM2 / LFM2.5
     │       │   └── llama.rs    # LLaMA / Mistral / Qwen / Gemma / Phi
     │       ├── kv_cache.rs     # KV cache (simple contiguous, then paged)
     │       └── engine.rs       # Top-level generate() orchestration
     └── cera-cli/              # CLI binary
         ├── Cargo.toml
         └── src/main.rs

0.2  Workspace Cargo.toml:
     - edition = "2024", rust-version = "1.85"
     - Feature flags: "wgpu" (optional GPU backend)
     - Workspace dependencies: anyhow, thiserror, tracing, byteorder,
       serde, serde_json, half, memmap2, clap, minijinja, bytemuck

0.3  .cargo/config.toml:
     - Release profile: LTO = "thin", codegen-units = 1
     - Target-specific RUSTFLAGS for native CPU features

0.4  justfile:
     - just build, just test, just run -- <args>, just bench, just ci

0.5  README.md, LICENSE (Apache-2.0 + MIT), .gitignore
```

---

## Phase 1: Tensor + CPU Compute ✅
**Time: 5-7 days**

```
1.1  tensor.rs — Core types:

     pub enum DType { F32, F16, BF16, I32, U8, Q4_0, Q4KM, Q8_0 }

     pub struct Tensor {
         data: Vec<u8>,       // raw bytes
         shape: Vec<usize>,
         dtype: DType,
     }

     Methods: shape(), dtype(), numel(), size_bytes(),
     to_f32_vec(), from_f32_vec(), as_f32_slice(), zeros_f32()

1.2  quant.rs — Q4_0, Q4_K_M and Q8_0:

     Q4_0 block (18 bytes):
       d: f16                  // scale
       qs: [u8; 16]            // 32 4-bit unsigned values, offset by -8

     Q8_0 block (34 bytes):
       delta: f16              // scale
       quants: [i8; 32]        // 32 signed 8-bit values

     Q4_K_M block (144 bytes):
       d: f16                  // super-block scale
       dmin: f16               // super-block min
       scales: [u8; 12]        // sub-block scales and mins (packed)
       qs: [u8; 128]           // 256 4-bit quants (128 bytes)

     Implement dequantize and vec_dot for each.

1.3  backend/cpu.rs — Naive reference implementations:
     fn matmul_f32, matmul_q4_0_f32, matmul_q8_0_f32, matmul_q4km_f32
     fn rmsnorm, silu_inplace, softmax_inplace
     fn rope, conv1d_depthwise
     fn add_inplace, mul_inplace

1.4  backend/simd.rs — SIMD-optimized vec_dot:
     NEON (aarch64) and AVX2 (x86_64) implementations
     with compile-time / runtime dispatch.
```

---

## Phase 2: GGUF Parser + Tokenizer ✅
**Time: 3-4 days**

```
2.1  gguf.rs — Parser:
     - Parse header: magic (0x46554747), version, tensor_count, kv_count
     - Parse KV metadata: all 13 GGUF value types
     - Parse tensor info: name, dims, dtype (with raw ggml_type_id), offset
     - Memory-map tensor data with memmap2 (zero-copy)
     - get_tensor(), tensor_data(), print_inspect()

2.2  cera inspect CLI command — dumps metadata + tensor info

2.3  tokenizer.rs — Minimal BPE:
     - Load vocab + merges from GGUF metadata
     - Byte-level BPE encode/decode
     - Special token detection from token_type array
     - Chat template rendering via minijinja

2.4  cera tokenize CLI command + Python comparison script
```

---

## Phase 3: LFM2 Forward Pass ✅
**Time: 7-10 days**

Build LFM2 FIRST. This is the hard case. LLaMA comes after, trivially.

```
3.1  Determine LFM2 GGUF tensor naming:
     BEFORE writing any model code, run `cera inspect` on the LFM2 GGUF
     and document every tensor name and shape.

     Known from real LFM2-VL-450M inspection:
     - Conv blocks: blk.N.shortconv.{in_proj,conv,out_proj}.weight
     - Attn blocks: blk.N.attn_{q,k,v}.weight, blk.N.attn_{q,k}_norm.weight,
       blk.N.attn_output.weight
     - All blocks: blk.N.attn_norm.weight, blk.N.ffn_{gate,up,down}.weight,
       blk.N.ffn_norm.weight
     - Global: token_embd.weight, token_embd_norm.weight
     - Note: lfm2.attention.head_count_kv is an i32 array (per-layer), not scalar

3.2  model/mod.rs — Model loading dispatch:

     pub struct ModelConfig { ... }
     pub enum BlockType { Attention, GatedConv }
     pub trait Model: Send { fn forward(...), fn config() }
     pub fn load_model(gguf: &GgufFile) -> Result<Box<dyn Model>>

3.3  kv_cache.rs — Simple contiguous KV cache (NOT paged yet).

3.4  model/lfm2.rs — LFM2 model struct + forward pass

3.5  sampler.rs — greedy, temperature, top_k, top_p, sample

3.6  engine.rs — Generation loop with prefill + decode

3.7  cera run CLI command

3.8  Correctness validation against llama.cpp
```

---

## Phase 4: LLaMA + Additional Architectures 🟡
**Time: 3-5 days**

```
4.1  model/llama.rs — LLaMA is all-attention blocks.   [done: shared path]
4.2  Architecture variants: mistral, qwen2, gemma, phi3  [partial: qwen2/qwen3 only]
4.3  Test each on a real GGUF. Greedy decoding matches llama.cpp.
```

> Status: the LLaMA code path is implemented and serves `qwen2`/`qwen3`. The bare
> `llama` arch string and the `mistral`/`gemma`/`phi3` variants are not yet
> dispatched in `model/mod.rs`. Verified vs llama.cpp for Qwen2/Qwen3 (NEOX RoPE).

---

## Phase 5: wgpu GPU Backend ✅
**Time: 10-14 days**

> Status: wgpu backend shipped (matmul, quantized GEMM/GEMV, rmsnorm, silu, rope,
> softmax, attention, conv1d, element-wise) plus a separate **Metal** backend and
> shader preprocessor. GPU forward pass currently supports **LFM2 only**; runs on
> wasm as well. Subgroup variants implemented with small-subgroup adapter support.

```
5.1  backend/wgpu.rs — Device init, buffer pool, weight upload.
5.2  WGSL shaders: matmul, quantized matmul, rmsnorm, silu, rope, softmax,
     attention, conv1d, element-wise ops
5.3  Subgroup-enhanced variants (feature-detect at init)
5.4  Full GPU forward pass: single CommandEncoder, read back logits only.
5.5  CLI: --device gpu/cpu/auto. Benchmark CPU vs GPU.

     Note: V1 shaders use fixed workgroup sizes. Per-shape kernel tuning
     (V2.7) adds profile-guided dispatch for decode GEMV — significant
     wins on AMD RDNA3 (see kernel-anvil results: 2.25x on 7900 XTX).
     Design shader dispatch to accept configurable workgroup params from
     the start so V2.7 is a config change, not a rewrite.
```

---

## Phase 6: Polish v1 for Release ✅
**Time: 3-5 days**

```
6.1  HuggingFace model download (bundle system: list/download/bundle cmds)
6.2  Interactive chat mode: cera chat (TUI)
6.3  Benchmark command: cera bench
6.4  Correctness: perplexity / parity harness vs llama.cpp
6.5  CI + static binary releases (Linux, macOS, Windows)
6.6  README with benchmarks, install instructions, supported models
```

> Note: model distribution is handled via the **bundle** system
> (`list-bundles`, `download-bundles`, `bundle`) rather than a bare
> `cera run -m <hf-id>` form.

---

# V1 Complete. Everything below is V2.

---

# V2 — Roadmap

Ordered by estimated impact. Many can be worked in parallel.

### V2.1: Server + Continuous Batching — 3-4 weeks ⬜
OpenAI-compatible HTTP server (axum + SSE), continuous batching scheduler, paged attention (replaces contiguous KV cache), request queue, Prometheus metrics, preemption.

### V2.2: Browser / WASM — 3-4 weeks ✅ DONE
WASM build (dual: threaded + single-threaded), wasm-bindgen-rayon for multi-threaded CPU, Web Worker architecture, OPFS model caching, JS API + npm package, Chrome enhanced (subgroups, dot4U8Packed, f16), Safari baseline (f16, standard WGSL), feature detection.

### V2.3: Structured Output — 1-2 weeks ⬜
GBNF grammar parser, JSON schema → grammar compiler, regex constraints, async FSM mask computation overlapped with forward pass.

### V2.4: KV Cache Serialization — 1-2 weeks ⬜
Serialize KV cache + conv buffers to .lmkv files, system prompt caching, conversation checkpointing, KV quantization for storage.

### V2.5: Prefix Caching (Radix Attention) — 1-2 weeks ⬜
Radix tree for in-memory prefix matching, LRU eviction, scheduler integration. 5-6x speedup on prefix-heavy workloads.

### V2.5b: TurboQuant KV Cache Compression — 1-2 weeks ✅ DONE
Google Research's data-oblivious KV cache compression (ICLR 2026). Compresses KV cache to 3-3.5 bits with zero accuracy loss.

### V2.6: More Quantization Formats — 1 week per format 🟡 PARTIAL (Q6_K done)
Q2_K through Q6_K, IQ quants, GPTQ, AWQ, FP8, in-situ quantization.

### V2.7: Per-Shape Kernel Tuning (GEMV/MMVQ) — 1-2 weeks ⬜
Profile-guided kernel optimization for quantized decode (batch=1 GEMV). Instead of using one-size-fits-all thread/block configs for all layers, profile each unique (quant_type, N, K) shape on the target GPU and apply optimal nwarps/rows_per_block at runtime. Inspired by [kernel-anvil](https://github.com/apollosenvy/kernel-anvil) which demonstrated 2.25x decode speedup on Qwen3.5-27B Q4_K_M (12→27 tok/s on RX 7900 XTX) by auto-tuning llama.cpp's MMVQ kernels per model shape. Key insight: a 1024-row GQA projection and a 17408-row FFN layer have very different optimal configs. The bottleneck classification (bandwidth-bound vs occupancy-limited vs compute-bound) determines the sweep strategy. For cera: implement shape-aware dispatch in wgpu compute shaders (WGSL workgroup size, rows per invocation) and optionally in CPU SIMD (loop tiling). Store per-model configs as JSON; profile on first run or via `cera tune` command.

### V2.8: Speculative Decoding — 1-2 weeks ⬜
Draft model + verification, self-speculative. 1.3-2x decode speedup.

### V2.9: LoRA Adapters — 1-2 weeks ⬜
Runtime LoRA loading, merge/unmerge, per-request LoRA selection.

### V2.10: MoE Support — 2-3 weeks ⬜
Top-K expert routing for Mixtral, LFM2-8B-A1B, LFM2-24B-A2B.

### V2.11: Multi-GPU — 3-4 weeks ⬜
Pipeline parallelism, tensor parallelism, CPU offloading.

### V2.12: CUDA Backend — 3-4 weeks ⬜
Optional cuBLAS + FlashAttention + CUDA graphs. Requires nvcc.

### V2.13: Python Bindings — 1-2 weeks ⬜
PyO3 bindings, `pip install cera-engine`.

### V2.14: Kotlin Multiplatform Bindings — 2-3 weeks ✅ DONE
C ABI via cbindgen + platform-native FFI per KMP target (cinterop, Panama FFM, PanamaPort, JS interop).

### V2.17: Flutter / Dart Bindings — 2-3 weeks 🟡 (sync + streaming working)
Expose the engine to Flutter/Dart, reusing the existing `cera-ffi` UniFFI
surface (the same C ABI that already backs Kotlin + Swift). The
`cera-ffi-flutter` Dart package ships the generated+patched bindings plus a
platform-aware native-library loader.

**Working (verified end-to-end):** the synchronous engine API round-trips real
inference — loaded a Qwen2-0.5B GGUF through `CeraEngine.fromPath` →
`newSession` → `appendText` → `generate` and got tokens back; `cpuBackendReport`
and structured `FfiError` propagation also confirmed. Delivered:
- `cera-ffi` gains an **`ffi-buffer`** cargo feature
  (`uniffi/scaffolding-ffi-buffer-fns`) — the Dart generator calls
  `uniffi_ffibuffer_*` trampolines UniFFI only emits under that flag.
- `tool/patch_generated_bindings.dart` — deterministic, idempotent post-gen
  fixups: corrects `rustbuffer`/`rust_future` symbol names + the `.ref.ptr`
  union field, rewrites native-lib resolution (`CERA_FFI_LIB` + platform name),
  synthesizes the `EngineConfig` record encoder, fixes the async-ctor return
  type, and stubs the unsupported callback-sink methods.
- `just dart-libs` / `dart-bindings` / `dart-bindings-check` recipes; committed
  generated bindings (analyze clean); `example/cera_generate.dart`.

**Streaming — WORKING (verified end-to-end).** `generate_streaming(opts, sink)`
delivers tokens to a Dart-implemented `ModalitySink`: a Qwen2-0.5B run produced
24 `onTextTokens` callbacks + one `onDone(FinishReasonMaxTokens)`. Getting there
required vendoring the generator (`third_party/uniffi-bindgen-dart/`, own
workspace) and four codegen fixes — to be upstreamed to
`nchapman/uniffi-bindgen-dart`:
1. **Callback-arg lowering** — sink args lower via `<Name>FfiCodec.lower`
   (registers the Dart impl + installs the vtable), not a raw object write — so
   a sink can be passed *into* Rust.
2. **Foreign-trait vtable-init symbol** — was `<name>_trait_callback_init` (no
   such export); now UniFFI's `uniffi_<ns>_fn_init_callback_vtable_<name>`.
3. **Vtable slot order** — the generator sorted methods alphabetically,
   misaligning slots vs Rust's declaration order; now preserved for callback
   traits (`onTextTokens, onAudioFrames, onDone`).
4. **RustBuffer callback-arg ABI** — the generator JSON-encoded complex callback
   args (`Pointer<Utf8>`), but stock UniFFI passes a **RustBuffer by value**.
   Added callback-specific FFI mappers (`map_callback_native/dart_ffi_type`,
   scoped to callback bridges — the non-ffibuffer runtime path is untouched) +
   RustBuffer decode via the existing `_UniFfiBinaryReader`/`_uniffiRead<T>`.
   `Vec<u32>`/`Vec<f32>`/enum now decode correctly. 223 vendored tests pass
   (incl. new callback-mapper tests).

**Remaining:** `*Async` methods (`generate_async`, `generate_streaming_async`,
`fromBundleIdAsync`) still need the async invocation ABI (separate, larger —
stubbed to throw); `BundleRepo.withProgress` (DownloadProgressSink) is wired but
unverified (stubbed); package prebuilt native libs per target (Android jniLibs /
iOS xcframework / desktop); expose a detokenizer over FFI; example Flutter app +
wire the Dart drift check into CI; then the upstream PR.

**Spike result (2026-06-13, `uniffi-bindgen-dart` 0.1.3):** Viable but not
turnkey. The generator builds against `uniffi_bindgen 0.31.1` (our exact
version) and emitted ~7,300 lines of Dart from the current `cera-ffi` dylib
with **zero Rust-side changes** — structs, enums, sync methods, and
`CeraEngine.transcribe` came out clean (UniFFI checksums matched). After adding
the `ffi` package dep and an SDK `^3.3.0` constraint, `dart analyze` drops to
**8 errors, 0 warnings**, and every error sits in the *advanced* FFI surface:
- callback / foreign-trait sinks — `DownloadProgressSink`, `ModalitySink`
  (download progress + audio-modality streaming) generate invalid casts;
- async constructor `fromBundleIdAsync` returns `CeraEngine` instead of
  `Future<CeraEngine>`;
- a `_UniFfiFfiBufferElement.pointer` getter bug in sequence handling.

So the bulk auto-generates, but cera leans hard on exactly the async +
streaming-callback features 0.1.3 mishandles. Paths forward:
1. **Narrow the Dart-exposed surface** — generate for the sync core, hand-write
   thin Dart shims for the streaming/async bits.
2. **Patch/contribute upstream** — the failures are isolated; `uniffi-bindgen-dart`
   is young (0.1.x) and the fixes look tractable.
3. **flutter_rust_bridge** — separate binding layer, but first-class async +
   `Stream` support (a better fit for token/audio streaming) at the cost of not
   reusing the UniFFI interface.

Recommendation: pursue (1)+(2) to stay aligned with the existing UniFFI
bindings; fall back to (3) if streaming UX becomes the priority.

---

## Multimodal (off original roadmap — added retroactively)

These tracks were not in the original V1/V2 plan but have been built out to
support the LFM2-VL and LFM2-Audio model families. Documented here so the
roadmap reflects what actually exists.

### V2.15: Vision (LFM2-VL) — ✅ core shipped
Image → text via a CLIP-family ViT encoder with a 2-layer MLP projector
(`PROJECTOR_TYPE_LFM2`). Shipped:
- `model/vision_encoder.rs` — ViT encoder weights + tensor mapping, loaded from
  the `multimodal_projector` GGUF in a VL bundle (`mmproj-*.gguf`). Verified
  against LFM2.5-VL-450M.
- `model/vision_preprocessor.rs` — PNG/JPEG decode → aspect-preserving dynamic
  resize → normalize → `[3×H×W]` NCHW tensor, with 2× pixel-unshuffle to match
  the encoder patch grid.
- Soft-token prefill into the LLM via `Session::append_chat_with_images`;
  CLI `cera run --image <path|url> [--image ...] [--prompt "…"]`, multi-image
  supported. `--prompt` is optional in image mode (image-only inputs are allowed).

Remaining:
- CPU-only encode path — no wgpu/Metal acceleration for the ViT yet.
- Single projector family (`LFM2`); other VL projector types not mapped.

### V2.16: Audio + TTS (LFM2-Audio) — ✅ core shipped
Full duplex: PCM in (ASR / audio understanding) and PCM out (speech
generation). Shipped:
- **Input:** `model/audio_preprocessor.rs` (PCM → log-mel, Slaney scale,
  librosa-compatible) → `model/audio_encoder.rs` (Conformer-style encoder,
  `PROJECTOR_TYPE_LFM2A`) → soft tokens via `Session::append_audio`.
  CLI `cera run --audio-in <wav> --system "Perform ASR."`; one-call ASR via the
  `CeraEngine.transcribe` UniFFI method.
- **Output:** `model/audio_decoder.rs` (DecoderModel samples 8 codes/frame +
  6-layer Depthformer backbone) → Detokenizer (codes → spectrogram → PCM via
  ISTFT/rustfft). Driven by `audio_engine.rs`, a generation loop with text↔audio
  modality switching. CLI
  `cera run --vocoder <gguf> --system "…" --prompt "…" --audio-out <wav>`
  (`--system` is required with `--vocoder`).
- **Acceleration:** `model/metal_audio_decoder.rs` moves the detokenizer
  backbone to Metal (~165ms→~10-15ms/frame target); ISTFT stays on CPU.

Remaining:
- Metal-only detokenizer acceleration — no wgpu path; CPU fallback is slow.
- Streaming/real-time output not yet exposed (batch WAV writer only).

---

### V2.17: Flutter / Dart Bindings — 2-3 weeks 🟡 (sync API working)
Expose the engine to Flutter/Dart, reusing the existing `cera-ffi` UniFFI
surface (the same C ABI that already backs Kotlin + Swift). The
`cera-ffi-flutter` Dart package ships the generated+patched bindings plus a
platform-aware native-library loader.

**Working (verified end-to-end):** the synchronous engine API round-trips real
inference — loaded a Qwen2-0.5B GGUF through `CeraEngine.fromPath` →
`newSession` → `appendText` → `generate` and got tokens back; `cpuBackendReport`
and structured `FfiError` propagation also confirmed. Delivered:
- `cera-ffi` gains an **`ffi-buffer`** cargo feature
  (`uniffi/scaffolding-ffi-buffer-fns`) — the Dart generator calls
  `uniffi_ffibuffer_*` trampolines UniFFI only emits under that flag.
- `tool/patch_generated_bindings.dart` — deterministic, idempotent post-gen
  fixups: corrects `rustbuffer`/`rust_future` symbol names + the `.ref.ptr`
  union field, rewrites native-lib resolution (`CERA_FFI_LIB` + platform name),
  synthesizes the `EngineConfig` record encoder, fixes the async-ctor return
  type, and stubs the unsupported callback-sink methods.
- `just dart-libs` / `dart-bindings` / `dart-bindings-check` recipes; committed
  generated bindings (analyze clean); `example/cera_generate.dart`.

**Remaining:** `Result`-returning methods (`transcribe`, the tokenizer
accessors, `storeDir`, `fromBundleId*`) and the streaming/progress + `*Async`
surface throw `UnsupportedError` — `uniffi-bindgen-dart` 0.1.3 implements neither
the RustCallStatus out-arg ABI nor callback-interface lowering; package prebuilt
native libs per target (Android jniLibs / iOS xcframework / desktop); expose a
detokenizer over FFI; example Flutter app + wire the Dart drift check into CI.

**Spike result (2026-06-13, `uniffi-bindgen-dart` 0.1.3):** Viable but not
turnkey. The generator builds against `uniffi_bindgen 0.31.1` (our exact
version) and emitted ~7,300 lines of Dart from the current `cera-ffi` dylib
with **zero Rust-side changes** — structs, enums, sync methods, and
`CeraEngine.transcribe` came out clean (UniFFI checksums matched). After adding
the `ffi` package dep and an SDK `^3.3.0` constraint, `dart analyze` drops to
**8 errors, 0 warnings**, and every error sits in the *advanced* FFI surface:
- callback / foreign-trait sinks — `DownloadProgressSink`, `ModalitySink`
  (download progress + audio-modality streaming) generate invalid casts;
- async constructor `fromBundleIdAsync` returns `CeraEngine` instead of
  `Future<CeraEngine>`;
- a `_UniFfiFfiBufferElement.pointer` getter bug in sequence handling.

So the bulk auto-generates, but cera leans hard on exactly the async +
streaming-callback features 0.1.3 mishandles. Paths forward:
1. **Narrow the Dart-exposed surface** — generate for the sync core, hand-write
   thin Dart shims for the streaming/async bits.
2. **Patch/contribute upstream** — the failures are isolated; `uniffi-bindgen-dart`
   is young (0.1.x) and the fixes look tractable.
3. **flutter_rust_bridge** — separate binding layer, but first-class async +
   `Stream` support (a better fit for token/audio streaming) at the cost of not
   reusing the UniFFI interface.

Recommendation: pursue (1)+(2) to stay aligned with the existing UniFFI
bindings; fall back to (3) if streaming UX becomes the priority.

---

## V2 Prioritization

**Local inference on laptop:** V1 is sufficient. Add V2.6 for more quants, V2.7 for per-shape tuning.

**Production API server:** V2.1 → V2.5 → V2.5b (TurboQuant) → V2.3

**Browser inference (differentiator):** V2.2 → V2.5b (TurboQuant) → V2.4 → V2.3

**Mobile / on-device apps:** V2.14 → V2.5b (TurboQuant) → V2.4 (KV serialization) → V2.3

**AMD GPU performance:** V2.7 (per-shape tuning) → V2.6 (more quants) → V2.8 (speculative)

**Long-context use cases (32K+):** V2.5b (TurboQuant) → V2.1 (paged attention) → V2.5 (prefix caching)

**Largest models:** V2.10 → V2.11 → V2.12

---

## Dependencies (V1)

```toml
[dependencies]
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"
byteorder = "1"
bytemuck = "1"
half = "2"
memmap2 = "0.9"
clap = { version = "4", features = ["derive"] }
minijinja = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"

# Optional GPU backend
wgpu = { version = "24", optional = true }

[features]
default = []
gpu = ["dep:wgpu"]
```

> **Note:** The `wgpu` dependency and `gpu = ["dep:wgpu"]` feature shown above are
> illustrative of the planned V2 layout. The current `cera/Cargo.toml` has `gpu = []`
> as a placeholder with no `wgpu` dependency wired in yet.

No `tokenizers`, no `rayon`, no `axum`, no `tokio`, no `wasm-bindgen`.
Add these in v2 modules that need them.

---

## Claude Code Session Plan (V1)

| Session | Phase | Goal |
|---------|-------|------|
| 1 | 0 | Scaffold workspace, all files created, compiles ✅ |
| 2 | 1a | Tensor types, Q4_0/Q4_K_M/Q8_0 dequantization, tests ✅ |
| 3 | 1b | Naive CPU matmul + all element-wise ops, tests ✅ |
| 4 | 1c | SIMD matmul (AVX2 + NEON), benchmarks ✅ |
| 5 | 2a | GGUF parser, inspect command, test with real file ✅ |
| 6 | 2b | BPE tokenizer, chat templates, test against HF ✅ |
| 7 | 3a | LFM2 model struct, from_gguf loading, tensor name mapping |
| 8 | 3b | LFM2 conv block forward, attention forward, KV cache |
| 9 | 3c | Full forward pass + sampling + generate loop. First text! |
| 10 | 3d | Debug until output matches llama.cpp reference |
| 11 | 4 | LLaMA model + 2-3 variants (Mistral, Qwen, Gemma) |
| 12 | 5a | wgpu init, naive matmul shader, test against CPU |
| 13 | 5b | Tiled matmul, quantized matmul, element-wise shaders |
| 14 | 5c | Attention + conv1d shaders, subgroup variants |
| 15 | 5d | Full GPU forward pass integration, benchmark |
| 16 | 6 | HF download, chat mode, bench command, CI, README |

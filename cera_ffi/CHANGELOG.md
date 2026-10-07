# Changelog

## 0.7.0

- Add Qualcomm Hexagon NPU backend and FastRPC runtime integration. The backend is implemented for LFM2/LFM2.5, LFM2-MoE, dense transformers and Qwen 3.5 hybrid models, plus the vision encoder and audio detokenizer. The dense, MoE and Qwen 3.5 paths are covered by host tests and still need validation on hardware; see `docs/HEXAGON_NPU.md` for per-architecture status. Set `CERA_DISABLE_HEXAGON=1` (alias `CERA_NO_HEXAGON`) to force the CPU/GPU path.
- Fix loading of real Qwen 3.5 GGUFs on the CPU path (and the Hexagon loader that reuses it): the loader looked for `blk.N.attn_post_norm.weight` or `ffn_norm.weight`, but llama.cpp exports name the tensor `post_attention_norm`, so files such as Qwen3.5-0.8B-Q4_K_M failed at load with a missing-tensor error. The older names still load.
- Add structured CPU topology discovery and dynamic worker threadpool resizing.
- Add `GenerateOpts.no_spec` and `SessionConfig.disable_spec` to opt out of speculative decoding per call or per session, and `Session.disable_spec` / `Session.enable_spec` to toggle it at runtime without detaching the drafter.
- Add `Session.append_raw_image` and `PixelFormat` for feeding uncompressed pixel buffers (for example an Android `Bitmap` or a camera frame) to the vision encoder without a PNG or JPEG round trip, and the `Session.image_max_long_size` getter to pair with the existing setter.
- Add `CeraEngine.configure_prefix_cache` to set the prefix-cache directory and warm entry limit at runtime.
- Add `BackendPreference.npu` (Dart `CeraBackend.npu`, CLI `--device npu`): a vendor-neutral NPU preference that tries each NPU backend compiled into the build and uses the first that loads, so callers do not have to name a vendor the device may not have. **Behavior change:** the `"npu"` label used to parse to `Hexagon`; it now parses to `Npu`, and `"hexagon"` / `"htp"` still force Hexagon. On a Hexagon build `npu` behaves as `hexagon` did, but a failure now lists each vendor tried. `BackendPreference::Npu` is a new variant on a public enum, so exhaustive matches need an arm. `Auto` is unchanged.
- Add `hexagon_probe` and `hexagon_install_skels` for detecting the Hexagon NPU and installing its DSP skeleton libraries from an app-owned directory.
- Further breaking changes in the `cera` crate. `RowPool::decode()` and `RowPool::prefill()` return `Arc<RowPool>` (were `&'static RowPool`). The `GEMV_Q4_0_QKV` shader-source consts are removed from `backend::wgpu::shaders` and `backend::metal::shaders`. `BackendPreference::Hexagon` is a new variant of `cera::BackendPreference` and of the FFI `BackendPreference` enum, so exhaustive `match`, Kotlin `when`, Swift `switch` and Dart `switch` over it need a new arm. New public fields `GenerateOpts.no_spec`, `SessionConfig.disable_spec`, `WeightRef.repack_loader_skipped` and `InferenceState.prefill_scratch` break struct-literal construction: supply them, use `..Default::default()` for `GenerateOpts` and `SessionConfig`, and build `WeightRef` and `InferenceState` through `WeightRef::new` and `InferenceState::from_config`. `Session::has_gpu_audio_decoder` is now `Session::has_audio_accelerator`, alongside the `CeraEngine` rename above. `transformer::Repacked` gained the variants `Q40Smmla`, `Q4KSmmla`, `Q6K` and `Q6KSmmla`, so exhaustive matches need new arms. `LayerWeightRefs.qkv_repacked` changed type to `Arc<OnceLock<Option<FusedAttnProjections>>>`.
- Align package and native artifact versions at 0.7.0.
- **Breaking (Rust API, `cera` crate):** the audio accelerator surface is renamed with no deprecated aliases: trait `AudioGpu` is now `AudioAccelerator`, `build_gpu_audio_decoder` is `build_audio_accelerator`, `Session::attach_gpu_audio_decoder` is `attach_audio_accelerator`, and `has_gpu_audio_decoder` is `has_audio_accelerator` (on both `Session` and `CeraEngine`). `FrameOutcome::Fault` and `BackendPreference::Hexagon` are new variants on public enums, so exhaustive matches need an arm.
- **Behavior change:** the Android 64-bit AAR now ships the Hexagon NPU and wgpu backends (`hexagon`, `gpu` features), so `BackendPreference.AUTO` probes Hexagon, then wgpu, then CPU there. The 32-bit ABIs stay CPU-only.
- Add `CeraEngine.audioProfile()` (UniFFI records `AudioProfile` / `TtsVoice`), the wasm `audioProfile` getters on `CeraEngine` and `WebGpuSession`, and `Cera.audioProfile` (Dart: `CeraAudioProfile`, `CeraTtsVoice`, `CeraAudioProfile.plain()`, `audioModeOf`, `ceraTextOnlySystemPrompt`): the TTS and interleaved system prompts, voices and sample texts an audio model needs, resolved by cera from the bundle manifest's optional `audio_profile`, else a built-in registry matched on the model and vocoder file names, else plain `Perform TTS.`. Web apps re-run `dart run cera_ffi:install_web` to get the worker that serves profiles: an older worker degrades silently to the plain profile, and a current worker on an older wasm does the same with a console warning. A malformed manifest `audio_profile` is ignored field by field and reported through `tracing` (and the browser console on web). Rust: `cera::AudioProfile`, `cera::TtsVoice`, `CeraEngine::audio_profile()` and the optional manifest `audio_profile` object (format documented in `cera::manifest`).
- **Behavior change:** `Cera.appendAudio` with no `systemPrompt` now applies the profile's interleaved prompt on a model with audio output, which on a model with voices includes the first voice (`Respond with interleaved text and audio. Use the US female voice.` on LFM2.5-Audio, where it used to be the bare prompt).
- **Breaking (Rust API, `cera` crate):** `ModelBytes` gains `audio_profile: Option<AudioProfile>`; struct-literal constructors must set it (`None` keeps the plain profile). Custom implementations of the Dart `Cera` interface must add the `audioProfile` getter.
- Add FFI methods `Session.disableSpec()` / `enableSpec()`, `appendRawImage` and `configurePrefixCache`.
- Fix `SessionConfig.disableSpec` being overridden when the engine attaches a draft sidecar, and let `enableSpec` restore the drafter.
- **Behavior change:** the KV cache defaults to f16 where the model honors it (CPU LFM2 and the dense transformers), as llama.cpp's default cache type: `SessionConfig::default()` is `KvCompression::F16`, an omitted FFI `kvCompression` takes that default (pass `KvCompression.None` to keep the backend's own full-precision cache), and the CLI's `--kv-cache-keys` defaults to `auto` (`f32` restores full precision). A model that does not honor f16 (wgpu, Metal, Hexagon, which keep their own cache) is configured with its own KV, and wasm32 keeps the uncompressed cache. On a Galaxy S25 Ultra, CPU decode at 1,795 tokens of context goes from 141.8 to 172.4 tok/s. The mode is part of the prefix-cache tag and the checkpoint fingerprint, so a checkpoint or disk prefix-cache entry written with the previous default does not restore into a default session; recreate it. Dart's `CeraKvCompression.none` is now sent as an explicit `KvCompressionNone`, so the Dart wrapper keeps its documented full-precision default.
- **Behavior change:** `appendImage` and `appendRawImage` tile a large image the way the LFM2-VL reference does when no cap is in effect (no session default and `maxLongSize` null, or `maxLongSize` 0): an image over about twice the single-image budget (roughly 724x724) becomes a grid of up to 10 tiles of 256 tokens plus a thumbnail, with the reference's row and column markers, so a 1024x771 photo is 1,795 prompt tokens where it used to be 252. Budget context accordingly: a 4:3 photo is about 1,800 tokens, a 5:2 panorama up to about 2,800, with one vision-tower pass per tile, and set `maxLongSize` to keep an image to one tile. Tiling needs a model that can hand out token embedding rows (CPU, wgpu and Hexagon do; native Metal does not and falls back to the single thumbnail image).
- **Behavior change:** on aarch64 hosts with dotprod and fp16, CPU decode keeps a four-row interleaved copy of each Q4_0 weight and reads it with kernels whose arithmetic is the repacked prefill's, so decode and prefill agree bit for bit (text-only decode on a Galaxy S25 Ultra is level with llama.cpp). It costs about 155 MB of resident memory on the 450M (+155 to +158 MB measured on a Galaxy S25 Ultra), and a few near-tie greedy choices can differ from the previous layout (the logits differ at about 1e-3 relative). `CERA_Q4_DEC4=0` disables it.
- **Behavior change:** VL image resizing uses a Pillow-compatible bilinear (the reference processor's), so preprocessed pixels and image embeddings differ slightly from earlier builds.
- **Behavior change:** the Hexagon NPU backend no longer caps each DSP batch at 32 tensors by default; the conv-state scratch fix that made long batches unreproducible is in, and decode runs one uncapped batch per token (157 to 160 tok/s instead of 128 to 130 on LFM2.5-VL-450M). `CERA_HEXAGON_BATCH_TENSORS=<n>` still sets a cap.

## 0.6.3

- Add LoRA composition, per-request seeds, and extraction overrides to the session API.
- Bundle the native engine fix for repacked-Q4_0 row-major prefill addressing on ARM64.
- Align package and native artifact versions at 0.6.3.

## 0.6.2

- Fix chat stopping, streamed UTF-8, moved-handle cancellation and terminal callbacks.
- Validate checkpoint geometry and compression identity; reject unsupported native GPU checkpoints without changing session state.
- Enforce supported JSON Schema constraints and reject unsupported intersections.
- Fix audio timing, wake transitions, retained samples and configuration precedence.
- Align package and native artifact versions at 0.6.2.

## 0.6.1

### Changed

- **Version Alignment & Documentation**: Bumped workspace patch version to 0.6.1 in lockstep with `cera_ffi_flutter` and the underlying native engine crates; updated README documentation and examples for chat coordination and reactive streaming.

## 0.6.0

### Added

- **Conversational Chat Coordinator (`ChatSession`)**: Transactional multi-turn conversational coordinator with prompt template discovery, reactive streaming, first-class tool calling, JSON schema compilation, and multimodal turns.
- **Unified Streaming Audio Pipeline (`FfiAudioPipeline`)**: Low-latency voice facade coordinating Silero VAD v5, Hotword keyword spotting, and Whisper speech-to-text with wait-free cancellation.
- **Session Checkpointing & Persistence**: Binary snapshot export and restore with structural integrity validation and WebGPU direct VRAM persistence.

### Changed

- **Version Alignment**: Bumped workspace version to 0.6.0 in lockstep with `cera_ffi_flutter` and the underlying native engine crates.

## 0.5.6

### Added

- **Keyword Spotting (KWS) Engine**: Pure-Rust streaming acoustic front-end (`LogMelFrontEnd`), self-describing GGUF model detector (`HotwordDetector`), and VAD-gated chunk iterator (`HotwordIterator`) exposed over UniFFI (`FfiHotwordConfig`, `FfiHotwordDetector`, `FfiHotwordIterator`, `FfiHotwordEvent`).
- **Whisper ASR UniFFI Bindings**: Exposed pure-Rust OpenAI Whisper speech-to-text inference across foreign language bindings (`FfiWhisperModel`, `FfiWhisperTranscribeOpts`) with asynchronous cancellation support.

### Changed

- **Version Alignment**: Bumped workspace patch version to 0.5.6 in lockstep with `cera_ffi_flutter` and the underlying native engine crates.

## 0.5.5

### Added

- **Speculative Decoding FFI Configuration**: Plumbed speculative decoding (`SpecDecode`) configuration over the UniFFI boundary.
- **In-Memory Model Loading**: Added support for loading models and adapters directly from in-memory byte buffers across platforms.

### Changed

- **Version Alignment**: Bumped workspace patch version to 0.5.5 in lockstep with `cera_ffi_flutter` and the underlying native engine crates.

## 0.5.4

### Fixed

- **Model and Adapter Loading**: Added fallback path loaders for LoRA adapters, Silero VAD, and Hybrid PII models when `cera` is compiled without default features (`mmap` disabled) on native targets.

### Changed

- **Version Alignment**: Bumped workspace patch version to 0.5.4 in lockstep with `cera_ffi_flutter` and the underlying native engine crates.

## 0.5.3

### Added

- **Streaming Text Chunks**: Added `generateStreamChunks` API for streaming multi-token text chunks, reducing cross-boundary messaging overhead.
- **Multimodal Envelopes**: Added `MultimodalEnvelope`, `ImageInput`, and `AudioInput` types for structured multimodal prompt delivery.
- **Android Download Service**: Added support for background model asset downloading via `AndroidDownloadService`.
- **Whisper ASR Integration**: Added native bindings for OpenAI Whisper speech recognition models.

## 0.5.2

### Changed

- **Version Alignment**: Bumped workspace patch version to 0.5.2 in lockstep with `cera_ffi_flutter` and the underlying native engine crates.

## 0.5.1

### Added

- **WebGPU Dense Transformer Support**: Direct GPU execution support for dense transformer architectures (`llama`, `qwen2`, `qwen3`, `granite`) in addition to `lfm2`/`lfm2.5`/`lfm2moe`.
- **License Formatting**: Cleaned dual Apache-2.0 / MIT license formatting for automated OSI recognition by `pana` on pub.dev.

## 0.5.0

### Added

- **Native Silero VAD v5 engine and bindings**: Added `FfiSileroVad`, `FfiVadSampleRate`, `FfiVadIterator`, `FfiVadConfig`, and `sileroVadDefaultConfig` for real-time speech activity detection, streaming audio chunk processing, and speech segment timestamping.
- **Hugging Face model repository support**: Direct loading from Hugging Face model repository specs and URLs with optional on-the-fly streaming quantization.
- **Multimodal audio & vision enhancements**: WebGPU Depthformer acceleration, 4 voice modes (SpeechToText, VoiceChat, TextToSpeech, TextOnly), microphone audio input, silence trimming, and vision encoder ViT optimizations.

## 0.4.0

First release. The Dart bindings previously lived inside the
`cera_ffi_flutter` package; they were split out here so that they can be used
without Flutter at all.

### Added

- **`Cera`, a portable asynchronous API that runs on every target including the
  web.** One surface over two transports: the Rust async runtime natively, a Web
  Worker running `cera-wasm` in a browser. Loading, chat templating, tokenizing
  and streaming generation, with generations serialized against one KV cache.
  The generated bindings stay synchronous and native-only; `Cera` exists because
  a browser cannot offer a synchronous `generate` at any price.
- **Web inference**, on WebGPU where the browser has it and on a wasm CPU build
  where it does not (58 tok/s against 1.4 tok/s on the same machine and model).
  `dart run cera_ffi:install_web` puts the runtime in an app's `web/`; no
  COOP/COEP headers are required.
- `CeraEngine.fromPathAsync` and `fromBytesAsync`, so loading a model no longer
  blocks the calling isolate.
- `dart:ffi` bindings for the Cera inference engine, generated from the compiled
  `cera-ffi` cdylib by a vendored `uniffi-bindgen-dart` and committed under
  `lib/src/generated/`.
- `CeraLibrary`, a platform-aware loader: an explicit path, then `CERA_FFI_LIB`,
  then the platform's normal search path.
- Plain-Dart examples under `example/`: chat, streaming, async, and download
  progress.

### Notes

- **This package resolves under plain `dart pub get`, which is the reason it
  exists.** pub will not publish a package declaring `flutter.plugin.platforms`
  without a Flutter SDK constraint, and declaring that constraint makes
  `dart pub get` refuse the package outright. A Flutter plugin therefore cannot
  also be a plain-Dart package, so the bindings live here and
  `cera_ffi_flutter` depends on them.
- No native library ships with this package. Flutter apps get one from
  `cera_ffi_flutter`; everyone else points `CERA_FFI_LIB` at a `cera-ffi` build.
  That build needs the `ffi-buffer` cargo feature, without which every call
  fails at `dlsym`.
- The bindings are still exported conditionally, and the web branch is still a
  generated stub with the same API and no `dart:ffi`: data types are real,
  engine entry points throw `UnsupportedError`. That is what lets a
  multi-platform app build at all, since an unconditional `dart:ffi` import
  fails the whole build rather than one branch.

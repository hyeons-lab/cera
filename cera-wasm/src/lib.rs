//! wasm-bindgen wrapper around the `cera` core inference engine.
//!
//! This crate produces the `.wasm` cdylib that browser / Node consumers
//! drive via the JS glue emitted by `wasm-bindgen-cli`. Native consumers
//! should use the `cera` crate directly — cera-wasm exists purely to
//! map `cera`'s Rust API onto the JS interop boundary.
//!
//! The lib body is `cfg(target_arch = "wasm32")`-gated: on native
//! targets this crate compiles to an empty cdylib, which keeps
//! workspace-wide commands (`cargo check --workspace`,
//! `cargo clippy --workspace`) honest without needing to special-case
//! the wasm wrapper.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

/// Remote bundle downloading + caching, over the Origin Private File
/// System. The web counterpart to `cera::bundle::BundleRepo`.
pub mod bundle;

// Pull `wasm-bindgen-rayon` into the link graph so its
// `#[wasm_bindgen]`-emitted `initThreadPool` export survives
// dead-code elimination and reaches the generated `cera_wasm.d.ts`.
// JS init pattern + COOP/COEP requirement live in
// `cera-wasm/README.md`'s "Multi-threaded build" section.
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

// Custom TypeScript declarations injected into the generated
// `cera_wasm.d.ts`. wasm-bindgen would otherwise type wasm-side
// `JsValue` parameters as `any`, losing IDE completion + type
// checking for the structured shapes we expect from JS callers.
//
// Each `extern "C" { #[wasm_bindgen(typescript_type = "...")]
// pub type T; }` block below declares a Rust-side opaque handle
// whose only purpose is to carry a custom TS type label. At
// runtime these are still plain `JsValue`s.
#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND: &'static str = r#"
/**
 * One entry in the chat-message array passed to
 * `Tokenizer.applyChatTemplate` / `applyChatTemplateWithTools`.
 * Mirrors the OpenAI / Anthropic SDK shape. For tool results, use
 * `role: "tool"` with the JSON result string as `content`.
 */
export interface ChatMessage {
    role: string;
    content: string;
}

/**
 * A tool the model may call, in the OpenAI "function" shape. Pass an
 * array of these (JSON-stringified) to `applyChatTemplateWithTools`
 * and `toolGrammar`. `parameters` is a JSON Schema object for the
 * arguments.
 */
export interface ToolDef {
    name: string;
    description?: string;
    parameters?: object;
}

/**
 * A tool call parsed from model output by `parseToolCalls`. `arguments` is
 * normally an object, but a malformed Hermes/Qwen reply may pass through a
 * non-object JSON value — narrow before assuming an object map.
 */
export interface ToolCall {
    name: string;
    // Usually an object map (see the doc comment above); typed `unknown`
    // because a malformed reply can pass through a non-object JSON value.
    arguments: unknown;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// Opaque type-label wrapper for `ChatMessage[]` — the
    /// argument shape `Tokenizer.applyChatTemplate` accepts.
    /// At the wasm boundary this is just a JsValue array; the
    /// generated .d.ts surfaces it as `ChatMessage[]` so JS/TS
    /// callers get IDE completion + type checking.
    #[wasm_bindgen(typescript_type = "ChatMessage[]")]
    pub type ChatMessageArray;
}

#[wasm_bindgen(typescript_custom_section)]
const TS_CAPABILITIES: &'static str = r#"
/**
 * Modality capability flags for a loaded model. Returned by
 * `CeraEngine.capabilities` and `Session.capabilities`. Mirrors
 * the `ModalityCapabilities` shape exposed by the JVM/Apple
 * bindings (cera-ffi) so cross-platform consumers can probe the
 * same fields regardless of which binding they're driving.
 *
 * These describe the bundle that was actually loaded. A model
 * opened with `fromGgufBytes` is text-only by construction, since
 * one GGUF cannot carry a vision tower or an audio encoder: load
 * the multimodal projector alongside it via `fromGgufParts` for
 * `imageIn` / `audioIn` to be true.
 */
export interface Capabilities {
    readonly textIn: boolean;
    readonly textOut: boolean;
    readonly imageIn: boolean;
    readonly audioIn: boolean;
    readonly audioOut: boolean;
}
"#;

/// TS declaration for the `ModelMetadata` record. See `TS_CAPABILITIES`
/// for why these interfaces are hand-declared rather than derived.
#[wasm_bindgen(typescript_custom_section)]
const TS_MODEL_METADATA: &'static str = r#"
/**
 * Summary of a loaded model, returned by `CeraEngine.metadata`.
 * Mirrors the `ModelMetadata` record the JVM/Apple bindings
 * (cera-ffi) expose, so cross-platform consumers read the same
 * field names from either binding.
 *
 * Every field is also available as an individual getter on
 * `CeraEngine`. This is the one-call form: each getter is a
 * separate wasm boundary crossing, so prefer this when you want
 * several of them at once (rendering a model-info panel, say).
 */
export interface ModelMetadata {
    readonly architecture: string;
    readonly maxSeqLen: number;
    readonly vocabSize: number;
    readonly hasChatTemplate: boolean;
    readonly quantization: string;
    readonly addBosToken: boolean;
    readonly addEosToken: boolean;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// Opaque type-label wrapper for the `Capabilities`
    /// interface declared in the TS custom section above. At the
    /// wasm boundary this is a plain JS object; the type label
    /// surfaces in `.d.ts` so callers get IDE completion +
    /// destructuring (`const { audioIn } = engine.capabilities`).
    #[wasm_bindgen(typescript_type = "Capabilities")]
    pub type Capabilities;

    /// Opaque type-label wrapper for the `ModelMetadata` interface above,
    /// same arrangement as `Capabilities`.
    #[wasm_bindgen(typescript_type = "ModelMetadata")]
    pub type ModelMetadata;
}

/// Build a JS-side `ModelMetadata` object from cera core's own struct.
fn metadata_to_js(md: &cera::ModelMetadata) -> ModelMetadata {
    let obj = js_sys::Object::new();
    // As in `capabilities_to_js`: `Reflect::set` cannot fail against a
    // freshly-made object, so the `Result` is discarded per field.
    let set = |key: &str, value: JsValue| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(key), &value);
    };
    set("architecture", JsValue::from_str(&md.architecture));
    set("maxSeqLen", JsValue::from_f64(md.max_seq_len as f64));
    set("vocabSize", JsValue::from_f64(md.vocab_size as f64));
    set("hasChatTemplate", JsValue::from_bool(md.has_chat_template));
    set("quantization", JsValue::from_str(&md.quantization));
    set("addBosToken", JsValue::from_bool(md.add_bos_token));
    set("addEosToken", JsValue::from_bool(md.add_eos_token));
    JsValue::from(obj).unchecked_into()
}

/// Describe the CPU backend tier this build resolved at runtime, e.g.
/// `"tier=wasm_simd128 [simd128]"`.
///
/// Diagnostic: it tells you whether the SIMD kernels are actually live,
/// which is the difference between roughly 1.4 and 0.64 tokens/second on the
/// wasm CPU path. A browser without the simd128 proposal, or a `.wasm` built
/// without `+simd128`, reports the scalar tier.
#[wasm_bindgen(js_name = cpuBackendReport)]
pub fn cpu_backend_report() -> String {
    cera::cpu_features().report()
}

/// Build a JS-side `Capabilities` object from a cera core
/// `ModalityCapabilities`. Used by both `CeraEngine.capabilities`
/// and `Session.capabilities` so the field set + naming stays in
/// lock-step.
fn capabilities_to_js(caps: cera::ModalityCapabilities) -> Capabilities {
    let obj = js_sys::Object::new();
    // `Reflect::set` only fails when the target isn't an object
    // — `Object::new()` always is, so the `Result` is structurally
    // unreachable here. Discard it inside the helper closure to
    // keep the per-field calls one line each.
    let set_bool = |key: &str, value: bool| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_bool(value));
    };
    set_bool("textIn", caps.text_in);
    set_bool("textOut", caps.text_out);
    set_bool("imageIn", caps.image_in);
    set_bool("audioIn", caps.audio_in);
    set_bool("audioOut", caps.audio_out);
    JsValue::from(obj).unchecked_into()
}

/// Returns the version of the `cera` core library this binding wraps.
///
/// Note this is **`cera`'s** version, not `cera-wasm`'s — JS callers
/// usually want to know what core lib is driving the engine, since
/// the wrapper crate version may evolve independently.
#[wasm_bindgen(js_name = ceraVersion)]
pub fn cera_version() -> String {
    cera::VERSION.to_string()
}

/// Map an `anyhow::Error` into a `JsError` preserving the full
/// `{:#}` chain. Centralised so every wrapper surface throws the
/// same shape.
fn map_err(err: anyhow::Error) -> JsError {
    JsError::new(&format!("{err:#}"))
}

/// Parsed view of a LeapBundles `*.json` manifest.
///
/// JS callers fetch the manifest bytes (e.g. via `fetch().arrayBuffer()`)
/// and pass them to `Manifest.parse`. The wrapper exposes the typed
/// fields cera already understands; the raw `serde_json::Value`
/// retained on the inner `cera::manifest::Manifest` is intentionally
/// **not** exposed here — JS callers can re-parse the JSON themselves
/// for forward-compat fields, and we don't want to commit to a
/// `serde-wasm-bindgen` round-trip on every getter.
#[wasm_bindgen]
pub struct Manifest {
    inner: cera::manifest::Manifest,
}

#[wasm_bindgen]
impl Manifest {
    /// Parse a JSON manifest from raw bytes. Throws a `JsError` on
    /// malformed JSON or when required fields are missing or wrongly
    /// typed (e.g. no `load_time_parameters.model`). Unknown
    /// `inference_type` values are **not** an error — they round-trip
    /// through `cera::manifest::InferenceType::Unknown(String)` and
    /// surface verbatim via the `inferenceType` getter, so JS callers
    /// can decide how to react instead of catching here.
    #[wasm_bindgen]
    pub fn parse(json_bytes: &[u8]) -> Result<Manifest, JsError> {
        cera::manifest::Manifest::from_bytes(json_bytes)
            .map(|inner| Manifest { inner })
            .map_err(map_err)
    }

    /// Raw `inference_type` string (e.g. `llama.cpp/text-to-text`).
    /// Round-trips through cera's enum, so unknown variants come back
    /// as their original string — no information loss.
    #[wasm_bindgen(getter, js_name = inferenceType)]
    pub fn inference_type(&self) -> String {
        self.inner.inference_type.as_str().to_string()
    }

    #[wasm_bindgen(getter, js_name = schemaVersion)]
    pub fn schema_version(&self) -> String {
        self.inner.schema_version.clone()
    }

    /// URL (or local path string) for the primary model GGUF.
    #[wasm_bindgen(getter, js_name = modelUrl)]
    pub fn model_url(&self) -> String {
        self.inner.files.model.clone()
    }

    /// URL of the multimodal projector GGUF if the manifest declares
    /// one (VL / audio models). `undefined` for plain text models.
    #[wasm_bindgen(getter, js_name = multimodalProjectorUrl)]
    pub fn multimodal_projector_url(&self) -> Option<String> {
        self.inner.files.multimodal_projector.clone()
    }

    /// URL of the audio-decoder GGUF for audio-out models.
    #[wasm_bindgen(getter, js_name = audioDecoderUrl)]
    pub fn audio_decoder_url(&self) -> Option<String> {
        self.inner.files.audio_decoder.clone()
    }

    /// URL of the audio-tokenizer checkpoint (typically `.safetensors`).
    #[wasm_bindgen(getter, js_name = audioTokenizerUrl)]
    pub fn audio_tokenizer_url(&self) -> Option<String> {
        self.inner.files.audio_tokenizer.clone()
    }

    /// Jinja chat template override from the manifest, if present.
    /// `undefined` means "use the template embedded in the GGUF
    /// metadata" (cera's standard fallback).
    #[wasm_bindgen(getter, js_name = chatTemplate)]
    pub fn chat_template(&self) -> Option<String> {
        self.inner.chat_template.clone()
    }
}

/// Map a `cera::CeraError` into a `JsError`. Uses `Display` (not `Debug`)
/// so JS callers see the same message a cera CLI consumer would. Kept
/// distinct from `map_err` (which handles `anyhow::Error`) so the call
/// sites stay readable — both helpers throw the same `JsError` shape on
/// the JS side.
fn map_cera_err(err: cera::CeraError) -> JsError {
    JsError::new(&err.to_string())
}

#[inline]
#[allow(dead_code)]
pub(crate) fn console_info(msg: &str) {
    web_sys::console::info_1(&wasm_bindgen::JsValue::from_str(msg));
}

#[inline]
#[allow(dead_code)]
pub(crate) fn console_warn(msg: &str) {
    web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(msg));
}

/// Loaded inference engine — wraps `cera::CeraEngine` with sync access
/// to model metadata and the tokenizer.
///
/// JS callers fetch the GGUF (e.g. via `fetch().arrayBuffer()`), pass
/// the bytes to `CeraEngine.fromGgufBytes`, and use the returned
/// handle to read model info or pull a `Tokenizer`. Session-based
/// inference (`generate`, streaming) is intentionally not exposed yet
/// — that shape needs an async/streaming design that lives in a
/// follow-up PR.
///
/// **Memory:** the loaded GGUF stays resident in wasm linear memory
/// for the lifetime of this object. Call `.free()` (auto-emitted by
/// wasm-bindgen) when done to release it; without that, the entire
/// model lives until the page unloads.
#[wasm_bindgen]
pub struct CeraEngine {
    inner: cera::CeraEngine,
}

#[wasm_bindgen]
impl CeraEngine {
    /// Load a model from in-memory GGUF bytes. `contextSize` defaults
    /// to 4096 if omitted; the actual KV-cache cap is the smaller of
    /// the requested size and the model's own `max_seq_len`.
    ///
    /// The backend is forced to CPU — wasm has no native GPU/Metal
    /// backend. Throws on parse failure, unsupported quantization,
    /// or unrecognized architecture.
    #[wasm_bindgen(js_name = fromGgufBytes)]
    pub fn from_gguf_bytes(
        bytes: Vec<u8>,
        context_size: Option<u32>,
    ) -> Result<CeraEngine, JsError> {
        // Spread `..Default::default()` so the wrapper picks up any
        // future EngineConfig fields (e.g. `bundle_repo` when the
        // `remote` feature is on) without a compile break — only the
        // two we actually want to override are spelled out.
        // `..default()` is intentionally retained for forward
        // compatibility — clippy's `needless_update` only fires under
        // feature combos where `EngineConfig` happens to collapse to
        // exactly the two listed fields (e.g. wasm32 minimal builds).
        // Adding a feature later that introduces a new field would
        // otherwise compile-break this constructor; the spread keeps
        // it future-proof.
        #[allow(clippy::needless_update)]
        let cfg = cera::EngineConfig {
            context_size: context_size.unwrap_or(4096) as usize,
            backend: cera::BackendPreference::Cpu,
            ..cera::EngineConfig::default()
        };
        cera::CeraEngine::from_bytes(bytes, cfg)
            .map(|inner| CeraEngine { inner })
            .map_err(map_cera_err)
    }

    /// Load a multi-file bundle: the model GGUF plus its multimodal
    /// projector ("mmproj"). This is the constructor a VL or audio model
    /// needs, and `fromGgufBytes` structurally cannot be: the vision tower
    /// and the audio encoder live in a *second* GGUF, and that one takes a
    /// single buffer.
    ///
    /// `mmproj` may be `null`, in which case this is exactly
    /// `fromGgufBytes` with an explicit context size.
    ///
    /// **Modality is inferred from the arguments, not just the header.**
    /// Every published LFM2-VL model reports `architecture = "lfm2"`, the
    /// same string a text model reports, because the vision half is entirely
    /// in the mmproj. So passing an `mmproj` alongside a text-arch model is
    /// taken as the statement of intent it is and loads as image-to-text;
    /// audio models already identify themselves and are unaffected. Pass
    /// `inferenceType` explicitly to override (`"llama.cpp/text-to-text"`,
    /// `"llama.cpp/image-to-text"`, `"llama.cpp/lfm2-audio-v1"`).
    ///
    /// A malformed or mismatched mmproj is **not** fatal: it warns and the
    /// bundle still serves text, with `capabilities.imageIn` staying false
    /// and `appendImage` throwing "no vision encoder attached". That mirrors
    /// the native loaders rather than failing a whole page load over a
    /// sidecar.
    ///
    /// **Memory:** both buffers stay resident in wasm linear memory for the
    /// engine's lifetime. A VL bundle is the model *plus* the tower.
    #[wasm_bindgen(js_name = fromGgufParts)]
    pub fn from_gguf_parts(
        bytes: Vec<u8>,
        mmproj: Option<Vec<u8>>,
        context_size: Option<u32>,
        inference_type: Option<String>,
    ) -> Result<CeraEngine, JsError> {
        // See `from_gguf_bytes` for why the spread is deliberate.
        #[allow(clippy::needless_update)]
        let cfg = cera::EngineConfig {
            context_size: context_size.unwrap_or(4096) as usize,
            backend: cera::BackendPreference::Cpu,
            ..cera::EngineConfig::default()
        };
        let parts = cera::ModelBytes {
            model: bytes.into(),
            multimodal_projector: mmproj.map(Into::into),
            audio_decoder: None,
            // `parse_str` maps anything unrecognized to `Unknown(s)`, which
            // `from_parts` rejects by name, better than silently falling
            // back to text when a caller fat-fingers the string.
            inference_type: inference_type
                .as_deref()
                .map(cera::manifest::InferenceType::parse_str),
            chat_template: None,
            generation_defaults: None,
        };
        cera::CeraEngine::from_parts(parts, cfg)
            .map(|inner| CeraEngine { inner })
            .map_err(map_cera_err)
    }

    /// Load a published LeapBundle by id and quantization, downloading
    /// through `repo` and reusing whatever it already cached.
    ///
    /// This is the browser equivalent of the native
    /// `CeraEngine::from_bundle_id`. The manifest picks up every file
    /// the bundle names, so a VL or audio bundle arrives complete: no
    /// separate mmproj argument and no guessing at the modality, unlike
    /// `fromGgufParts` which has only its arguments to go on.
    ///
    /// `onProgress(url, bytesDownloaded, totalBytes)` fires during
    /// downloads only; a fully cached bundle loads without calling it.
    /// `totalBytes` is `null` when the server doesn't say.
    ///
    /// **Memory:** every file lands in wasm linear memory and stays for
    /// the engine's lifetime. The bytes are never handed to JS on the
    /// way, so this costs one copy of the model rather than two.
    #[wasm_bindgen(js_name = fromBundleId)]
    pub async fn from_bundle_id(
        repo: &bundle::BundleRepo,
        bundle_id: String,
        quant: String,
        context_size: Option<u32>,
        on_progress: Option<js_sys::Function>,
    ) -> Result<CeraEngine, JsError> {
        let parts = bundle::load_bundle(repo, &bundle_id, &quant, on_progress.as_ref()).await?;
        Self::from_bundle_parts(parts, context_size)
    }

    /// Load a bundle from the URL of its manifest JSON, for bundles
    /// hosted somewhere other than `LiquidAI/LeapBundles`.
    ///
    /// Files the manifest names are fetched relative to it. Entries
    /// with a nested path are refused rather than guessed at: see
    /// `bundle::join_url`.
    #[wasm_bindgen(js_name = fromManifestUrl)]
    pub async fn from_manifest_url(
        repo: &bundle::BundleRepo,
        manifest_url: String,
        context_size: Option<u32>,
        on_progress: Option<js_sys::Function>,
    ) -> Result<CeraEngine, JsError> {
        let parts = bundle::load_manifest(repo, &manifest_url, on_progress.as_ref()).await?;
        Self::from_bundle_parts(parts, context_size)
    }

    /// Shared tail of the two bundle constructors. Not exported: it
    /// takes a Rust type, and the point of `ModelBytes` here is that the
    /// weights never cross the JS boundary.
    fn from_bundle_parts(
        parts: cera::ModelBytes,
        context_size: Option<u32>,
    ) -> Result<CeraEngine, JsError> {
        // See `from_gguf_bytes` for why the spread is deliberate.
        #[allow(clippy::needless_update)]
        let cfg = cera::EngineConfig {
            context_size: context_size.unwrap_or(4096) as usize,
            backend: cera::BackendPreference::Cpu,
            ..cera::EngineConfig::default()
        };
        cera::CeraEngine::from_parts(parts, cfg)
            .map(|inner| CeraEngine { inner })
            .map_err(map_cera_err)
    }

    /// Model architecture string from the GGUF metadata
    /// (e.g. `"lfm2"`, `"llama"`).
    #[wasm_bindgen(getter)]
    pub fn architecture(&self) -> String {
        self.inner.metadata().architecture.clone()
    }

    /// Maximum sequence length the model was trained for. Independent
    /// of the engine's `contextSize` config — that one is the KV
    /// cache cap, this is the model's positional encoding ceiling.
    #[wasm_bindgen(getter, js_name = maxSeqLen)]
    pub fn max_seq_len(&self) -> u32 {
        self.inner.metadata().max_seq_len
    }

    #[wasm_bindgen(getter, js_name = vocabSize)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.metadata().vocab_size
    }

    /// Quantization label from the GGUF (e.g. `"Q4_0"`, `"Q4_K_M"`).
    /// Useful for telling users what they actually loaded when the
    /// download URL doesn't make it obvious.
    #[wasm_bindgen(getter)]
    pub fn quantization(&self) -> String {
        self.inner.metadata().quantization.clone()
    }

    /// `true` when the loaded GGUF carries an embedded Jinja chat
    /// template. JS callers can use this to decide whether to render
    /// `Tokenizer.chatTemplate` themselves vs falling back to a
    /// hard-coded prompt format.
    #[wasm_bindgen(getter, js_name = hasChatTemplate)]
    pub fn has_chat_template(&self) -> bool {
        self.inner.metadata().has_chat_template
    }

    /// `true` when the GGUF declares `tokenizer.ggml.add_bos_token`.
    /// Callers that hand-build a token sequence from `Tokenizer.encode`
    /// should prepend `Tokenizer.bosToken` when this is `true` (and
    /// the model has a BOS) — cera's encoder returns the raw tokens
    /// without that prefix.
    #[wasm_bindgen(getter, js_name = addBosToken)]
    pub fn add_bos_token(&self) -> bool {
        self.inner.metadata().add_bos_token
    }

    /// `true` when the GGUF declares `tokenizer.ggml.add_eos_token`. Prefer
    /// `Tokenizer.encodeSpecial`, which applies this (and BOS) automatically.
    #[wasm_bindgen(getter, js_name = addEosToken)]
    pub fn add_eos_token(&self) -> bool {
        self.inner.metadata().add_eos_token
    }

    /// Modality capability flags reported by the loaded model.
    /// See the `Capabilities` interface in the generated `.d.ts`
    /// for the field shape.
    ///
    /// These reflect the bundle you actually loaded. A model opened with
    /// `fromGgufBytes` is text-only by construction and reports
    /// `{ textIn: true, textOut: true }` with everything else false, because
    /// a single GGUF cannot carry a vision tower or an audio encoder. To get
    /// `imageIn` or `audioIn`, load the mmproj too via `fromGgufParts`.
    ///
    /// A bundle whose mmproj failed to parse reports the flag as false and
    /// logs a warning, so this stays an accurate answer about what the
    /// engine can do rather than what the caller intended.
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> Capabilities {
        capabilities_to_js(self.inner.capabilities())
    }

    /// Requested context-window size (KV cache cap) the engine was
    /// configured with. Mirrors what `fromGgufBytes(bytes,
    /// contextSize)` resolved to — i.e. the value of `contextSize`
    /// you passed in, or `4096` if you omitted it. Unlike
    /// `cera-ffi`'s `EngineConfig::try_from`, the wasm load path
    /// has no `0` → `maxSeqLen` translation: a `contextSize` of `0`
    /// trips cera core's `context_size > 0` load assertion and
    /// `fromGgufBytes` throws.
    ///
    /// Note this is the **engine-level requested** cap, not a
    /// per-session ceiling. cera core clamps the model's
    /// `maxSeqLen` at load time to `min(contextSize,
    /// gguf_max_seq_len)`, so `engine.maxSeqLen` is already the
    /// effective ceiling — `contextSize` is informational ("what
    /// cap did I load with?") rather than a value to `Math.min`
    /// against `maxSeqLen` at call sites.
    #[wasm_bindgen(getter, js_name = contextSize)]
    pub fn context_size(&self) -> u32 {
        self.inner.config().context_size as u32
    }

    /// Everything `CeraEngine`'s individual metadata getters report, in one
    /// object. See the `ModelMetadata` interface in the generated `.d.ts`.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> ModelMetadata {
        metadata_to_js(self.inner.metadata())
    }

    /// Constructs a `GenerateOpts` seeded with the advisory defaults from the
    /// model manifest (if any), falling back to standard defaults for unmentioned fields.
    #[wasm_bindgen(js_name = defaultGenerateOpts)]
    pub fn default_generate_opts(&self) -> GenerateOpts {
        GenerateOpts {
            inner: cera::GenerateOpts::from_manifest(self.inner.manifest()),
        }
    }

    /// Transcribe mono `f32` PCM audio (roughly normalized to `[-1.0, 1.0]`)
    /// to text, using the model's own audio encoder and chat template.
    ///
    /// `sampleRate` is the rate of the samples you pass; cera resamples to
    /// whatever the encoder wants. A typical browser source is
    /// `AudioBuffer.getChannelData(0)` after decoding through
    /// `AudioContext.decodeAudioData`, whose `sampleRate` you read off the
    /// same `AudioBuffer`.
    ///
    /// Requires an audio bundle loaded through `fromGgufParts` with its
    /// mmproj; otherwise this throws `"modality not supported by this
    /// model"`. This runs a full prefill + decode, so it is *slow* on the
    /// wasm CPU backend for anything but short clips.
    #[wasm_bindgen]
    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<String, JsError> {
        self.inner
            .transcribe(pcm, sample_rate)
            .map_err(map_cera_err)
    }

    /// The tool-call format auto-detected from this model's architecture, or
    /// `undefined` when the architecture has no known tool convention.
    ///
    /// Engine-level counterpart to the free `detectToolFormat(architecture)`
    /// function: this one already knows the loaded model's architecture, so
    /// it cannot disagree with it.
    #[wasm_bindgen(js_name = toolFormat)]
    pub fn tool_format(&self) -> Option<ToolFormat> {
        cera::tools::ToolFormat::detect(&self.inner.model().config().architecture).map(Into::into)
    }

    /// The token id of `format`'s tool-call start marker (e.g.
    /// `<|tool_call_start|>`) in this model's vocab, for use as a lazy
    /// grammar trigger in `GenerateOpts.grammarTriggerTokens`.
    /// `undefined` when this tokenizer lacks that special token.
    #[wasm_bindgen(js_name = toolCallStartToken)]
    pub fn tool_call_start_token(&self, format: ToolFormat) -> Option<u32> {
        let fmt: cera::tools::ToolFormat = format.into();
        self.inner
            .tokenizer()
            .special_token_id(fmt.call_start_marker())
    }

    /// Returns a `Tokenizer` handle bound to this engine's vocab.
    /// Each call allocates a fresh JS object but the underlying
    /// tokenizer state is shared via `Arc` — cheap to call, JS
    /// callers can cache the result if they prefer one handle.
    #[wasm_bindgen(getter)]
    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer {
            inner: self.inner.tokenizer_arc(),
        }
    }

    /// Construct a new `Session` for this engine. The `config`
    /// freezes per-session knobs — sampler `seed`, `nKeep`
    /// pinned-prefix size, `ubatchSize` chunked-prefill batch,
    /// `maxSeqLen` KV cap. For the cera defaults
    /// (`maxSeqLen = null` → engine's effective cap, i.e.
    /// `min(engine.contextSize, model.maxSeqLen)`; `nKeep = 0`,
    /// `seed = null`, `ubatchSize = 512`), pass a freshly-
    /// constructed `new SessionConfig()`.
    ///
    /// `config` is **borrowed**, not consumed — JS callers can
    /// reuse the same `SessionConfig` across multiple `newSession`
    /// calls. Inner state is cloned per-session at the boundary.
    /// This mirrors how `Session.generate` borrows `GenerateOpts`.
    /// (wasm-bindgen doesn't support `Option<&T>` for wrapper
    /// types, so a default-config caller passes
    /// `new SessionConfig()` rather than omitting the arg.)
    ///
    /// The returned `Session` keeps its own `Arc` clones of the
    /// engine's model and tokenizer, so freeing the engine doesn't
    /// invalidate any in-flight sessions.
    #[wasm_bindgen(js_name = newSession)]
    pub fn new_session(&self, config: &SessionConfig) -> Result<Session, JsError> {
        let inner = self
            .inner
            .new_session(config.inner.clone())
            .map_err(map_cera_err)?;
        let hidden_size = inner.hidden_size() as u32;
        Ok(Session { inner, hidden_size })
    }
}

/// BPE tokenizer wrapper. Constructed via `CeraEngine.tokenizer`;
/// no standalone `from*` constructor (the GGUF metadata required to
/// build one is reachable only through the engine).
///
/// Round-trip note: `decode(encode(text))` is **not** guaranteed to
/// be byte-identical to `text` for inputs containing tokens that
/// don't survive BPE merge replay (rare in practice — BOS/EOS,
/// some byte-level edge cases). When you need exact reproduction,
/// keep the original string around.
#[wasm_bindgen]
pub struct Tokenizer {
    inner: std::sync::Arc<cera::tokenizer::BpeTokenizer>,
}

#[wasm_bindgen]
impl Tokenizer {
    /// Tokenize a UTF-8 string. Returns the token IDs as a
    /// `Uint32Array`. No BOS/EOS prefix — callers that want them
    /// should prepend `bosToken` / append `eosToken` manually, or use
    /// `encodeSpecial`.
    #[wasm_bindgen]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Encode with optional special markers — the analog of llama.cpp's
    /// `llama_tokenize(..., add_special)`. When `addSpecial` is true, BOS is
    /// prepended iff the GGUF declares `tokenizer.ggml.add_bos_token` and EOS
    /// appended iff it declares `tokenizer.ggml.add_eos_token`, so token counts
    /// match llama.cpp. With `addSpecial = false` this is exactly `encode`.
    #[wasm_bindgen(js_name = encodeSpecial)]
    pub fn encode_special(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.inner.encode_special(text, add_special)
    }

    /// Detokenize back to a UTF-8 string. Lossy for tokens whose
    /// byte sequences don't decode to valid UTF-8 — those are
    /// replaced with U+FFFD per `String::from_utf8_lossy`.
    #[wasm_bindgen]
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens)
    }

    #[wasm_bindgen(getter, js_name = vocabSize)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.vocab_size() as u32
    }

    /// Whether the GGUF asks for a BOS token to be prepended
    /// (`tokenizer.ggml.add_bos_token`).
    ///
    /// Needed to frame a prompt correctly without `encodeSpecial`: that
    /// helper prepends BOS *and* appends EOS together, and a chat-template
    /// prompt wants the first and not the second. Callers doing their own
    /// framing read this and prepend `bosToken` themselves.
    #[wasm_bindgen(getter, js_name = addBosToken)]
    pub fn add_bos_token(&self) -> bool {
        self.inner.add_bos_token()
    }

    /// BOS token ID, if the GGUF metadata declares one.
    #[wasm_bindgen(getter, js_name = bosToken)]
    pub fn bos_token(&self) -> Option<u32> {
        self.inner.bos_token()
    }

    /// EOS token ID, if the GGUF metadata declares one.
    #[wasm_bindgen(getter, js_name = eosToken)]
    pub fn eos_token(&self) -> Option<u32> {
        self.inner.eos_token()
    }

    /// Look up a special-token ID by its literal name (e.g.
    /// `"<|im_start|>"`, `"<|tool_calls_section_begin|>"`).
    /// Returns `undefined` when no entry exists for that name in
    /// the model's special-token registry.
    ///
    /// Lookup scope: only tokens flagged as control or
    /// user-defined in the GGUF metadata are registered for this
    /// lookup. cera reads `tokenizer.ggml.token_type` and admits
    /// tokens with type `3` (control) or type `4` (user-defined);
    /// regular vocab entries are not reachable via this method
    /// even though their names exist in `tokenizer.ggml.tokens`.
    /// Names accepted here are the literal vocab strings indexed
    /// by the special token's ID.
    ///
    /// Useful for constructing prompts with specific control
    /// tokens directly (chat-template-like flows) without
    /// round-tripping through `applyChatTemplate`. For BOS / EOS
    /// prefer `bosToken` / `eosToken` (named getters that don't
    /// risk a typo in the lookup string).
    ///
    /// Mirrors `CeraEngine.specialTokenId` from cera-ffi (where
    /// it lives engine-side); cera-wasm hangs it off `Tokenizer`
    /// to match the established `engine.tokenizer.<method>`
    /// access pattern.
    #[wasm_bindgen(js_name = specialTokenId)]
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.inner.special_token_id(name)
    }

    /// `true` when `id` is registered as a control or user-defined
    /// special token in the model's GGUF metadata
    /// (`tokenizer.ggml.token_type` types `3` / `4`). Useful for
    /// output filtering — e.g. dropping `<|im_end|>` from a
    /// `Session.generate` token-callback batch before joining the
    /// IDs into UI-rendered text — and for token-class
    /// classification in analysis tools.
    ///
    /// Out-of-range IDs (>= vocab size) and regular vocab tokens
    /// both return `false`. Companion to `specialTokenId` which
    /// goes the other direction (name → ID).
    #[wasm_bindgen(js_name = isSpecialToken)]
    pub fn is_special_token(&self, id: u32) -> bool {
        self.inner.is_special_token(id)
    }

    /// Raw embedded Jinja chat template from the GGUF metadata, if
    /// any. Most callers should use [`Self::apply_chat_template`]
    /// (`applyChatTemplate` in JS) instead — this getter is for
    /// inspection or for callers who want to render with a
    /// different Jinja runtime.
    #[wasm_bindgen(getter, js_name = chatTemplate)]
    pub fn chat_template(&self) -> Option<String> {
        self.inner.chat_template().map(str::to_owned)
    }

    /// Render the model's embedded Jinja chat template against a
    /// `[{ role, content }, ...]` array, returning the prompt
    /// string ready for `Tokenizer.encode` + `Session.appendTokens`.
    ///
    /// `addGenerationPrompt` defaults to `true` (the common case
    /// when sending to the model expecting a response). Set to
    /// `false` when you only want the conversation rendered without
    /// the trailing assistant-prompt suffix.
    ///
    /// Throws `JsError` on:
    /// - the model not carrying a chat template
    ///   (`engine.hasChatTemplate === false`),
    /// - malformed `messages` (not an array, or entries missing
    ///   `role`/`content` strings),
    /// - a Jinja render failure (template references an undefined
    ///   variable, etc.).
    #[wasm_bindgen(js_name = applyChatTemplate)]
    pub fn apply_chat_template(
        &self,
        messages: ChatMessageArray,
        add_generation_prompt: Option<bool>,
    ) -> Result<String, JsError> {
        // `ChatMessageArray` is a wasm-bindgen opaque type-label
        // wrapper around `JsValue` — the runtime check + parse
        // still happens in `parse_chat_messages`. The TS-side
        // win is purely surface (callers get `ChatMessage[]`
        // instead of `any`).
        let msgs = parse_chat_messages(messages.as_ref())?;
        cera::tokenizer::apply_chat_template(
            &self.inner,
            &msgs,
            add_generation_prompt.unwrap_or(true),
        )
        .map_err(map_err)
    }

    /// Like `applyChatTemplate`, but also injects a `tools` array so a
    /// tool-trained model renders its tool-definition block. `toolsJson` is a
    /// JSON string encoding an array of `ToolDef` (`[{name, description?,
    /// parameters?}]`). Throws on invalid `toolsJson` or a render failure.
    #[wasm_bindgen(js_name = applyChatTemplateWithTools)]
    pub fn apply_chat_template_with_tools(
        &self,
        messages: ChatMessageArray,
        tools_json: &str,
        add_generation_prompt: Option<bool>,
    ) -> Result<String, JsError> {
        let msgs = parse_chat_messages(messages.as_ref())?;
        let tools = parse_tool_defs(tools_json)?;
        cera::tokenizer::apply_chat_template_with_tools(
            &self.inner,
            &msgs,
            &tools,
            add_generation_prompt.unwrap_or(true),
        )
        .map_err(map_err)
    }
}

/// The tool-call wire format a model family uses. Get one from
/// `detectToolFormat(architecture)` or choose explicitly.
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// LFM2 / LFM2.5: Pythonic `[get_weather(city="Paris")]`.
    Lfm2Pythonic,
    /// Hermes / Qwen: JSON `{"name":…,"arguments":{…}}`.
    Hermes,
}

impl From<ToolFormat> for cera::tools::ToolFormat {
    fn from(f: ToolFormat) -> Self {
        match f {
            ToolFormat::Lfm2Pythonic => cera::tools::ToolFormat::Lfm2Pythonic,
            ToolFormat::Hermes => cera::tools::ToolFormat::Hermes,
        }
    }
}

impl From<cera::tools::ToolFormat> for ToolFormat {
    fn from(f: cera::tools::ToolFormat) -> Self {
        match f {
            cera::tools::ToolFormat::Lfm2Pythonic => ToolFormat::Lfm2Pythonic,
            cera::tools::ToolFormat::Hermes => ToolFormat::Hermes,
        }
    }
}

/// Detect the tool-call format for a model architecture string (`"lfm2"`,
/// `"qwen3"`, …). Returns `undefined` for architectures with no known
/// convention.
///
/// Prefer `CeraEngine.toolFormat` when you have an engine: it reads the
/// loaded model's own architecture, so it cannot be given the wrong string.
#[wasm_bindgen(js_name = detectToolFormat)]
pub fn detect_tool_format(architecture: &str) -> Option<ToolFormat> {
    cera::tools::ToolFormat::detect(architecture).map(Into::into)
}

/// Parse tool calls out of generated model text. Returns a JSON string
/// encoding an array of `ToolCall` (`[{name, arguments}]`) — `JSON.parse` it.
/// An empty array means the reply had no tool call.
#[wasm_bindgen(js_name = parseToolCalls)]
pub fn parse_tool_calls(text: &str, format: ToolFormat) -> Result<String, JsError> {
    let calls = cera::tools::parse_tool_calls(text, format.into()).map_err(map_err)?;
    serde_json::to_string(&calls).map_err(|e| JsError::new(&e.to_string()))
}

/// Build a GBNF grammar constraining output to a valid call for one of the
/// tools in `toolsJson` (a JSON array of `ToolDef`). Feed the result to
/// `GenerateOpts.setGrammar` and set `GenerateOpts.grammarTriggerTokens` for a lazy
/// tool-call trigger.
#[wasm_bindgen(js_name = toolGrammar)]
pub fn tool_grammar(tools_json: &str, format: ToolFormat) -> Result<String, JsError> {
    let tools = parse_tool_defs(tools_json)?;
    cera::tools::tool_grammar(&tools, format.into()).map_err(map_err)
}

/// Parse a JSON array of `ToolDef` into the cera core type. An empty/blank
/// string is treated as "no tools" (parity with the FFI empty-list contract),
/// so callers can pass `""` to render without tools.
fn parse_tool_defs(tools_json: &str) -> Result<Vec<cera::tools::ToolDef>, JsError> {
    if tools_json.trim().is_empty() {
        return Ok(Vec::new());
    }
    let defs: Vec<cera::tools::ToolDef> = serde_json::from_str(tools_json)
        .map_err(|e| JsError::new(&format!("invalid tools JSON: {e}")))?;
    // Each tool's `parameters` must be a JSON Schema object; a non-object
    // (e.g. `"parameters": null`) breaks chat templates that read
    // `tool.parameters.properties` and yields a zero-property grammar. Reject
    // it here, matching the CLI and UniFFI surfaces.
    for def in &defs {
        if !def.parameters.is_object() {
            return Err(JsError::new(&format!(
                "tool `{}` has non-object `parameters` (expected a JSON Schema object)",
                def.name
            )));
        }
    }
    Ok(defs)
}

/// Parse a JS-side `[{ role, content }, ...]` array into the cera
/// core type, using `js_sys::Reflect` directly rather than going
/// through `serde-wasm-bindgen`. Both approaches were measured —
/// they produce **the same wasm size** (the size growth from
/// `apply_chat_template` is dominated by minijinja's render path,
/// not the deserialiser). The manual `Reflect` walk is preferred
/// here because it keeps the dep graph smaller (one less crate to
/// audit + faster cold builds) for two flat string fields. If a
/// future surface needs rich nested deserialisation, revisit and
/// add `serde-wasm-bindgen` then.
fn parse_chat_messages(value: &JsValue) -> Result<Vec<cera::tokenizer::ChatMessage>, JsError> {
    let array = value
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsError::new("messages must be an array"))?;
    // `js_sys::Array::length` returns `u32` and that's the index
    // type `Array::get` takes — keep `len` in `u32` for the loop
    // and only widen to `usize` at the `Vec::with_capacity` call.
    let len = array.length();
    let mut msgs = Vec::with_capacity(len as usize);
    let role_key = JsValue::from_str("role");
    let content_key = JsValue::from_str("content");
    for i in 0..len {
        let entry = array.get(i);
        msgs.push(cera::tokenizer::ChatMessage {
            role: read_string_field(&entry, &role_key, "role", i)?,
            content: read_string_field(&entry, &content_key, "content", i)?,
        });
    }
    Ok(msgs)
}

/// Read a string-typed field off a JS object, distinguishing the
/// three failure modes a JS caller will commonly hit so the thrown
/// `JsError` actually points at the bug:
///   - `entry` is not an object (`Reflect::get` errors)
///   - the field is missing (`Reflect::get` returns `undefined`)
///   - the field is present but not a string
///
/// `js_sys::Reflect::get` only `Err`s on the first case (proxy
/// throws, target not Object); missing-property-on-an-Object
/// returns `Ok(JsValue::UNDEFINED)`, which would otherwise
/// silently fall through to a misleading "must be a string"
/// message. Splitting the cases keeps `messages[i].role missing`
/// distinguishable from `messages[i].role must be a string`.
fn read_string_field(
    entry: &JsValue,
    key: &JsValue,
    field_name: &str,
    index: u32,
) -> Result<String, JsError> {
    let value = js_sys::Reflect::get(entry, key)
        .map_err(|_| JsError::new(&format!("messages[{index}] is not an object")))?;
    if value.is_undefined() {
        return Err(JsError::new(&format!(
            "messages[{index}] missing '{field_name}' field"
        )));
    }
    value
        .as_string()
        .ok_or_else(|| JsError::new(&format!("messages[{index}].{field_name} must be a string")))
}

// ---------------------------------------------------------------------------
// Session + generate
// ---------------------------------------------------------------------------

/// Per-session knobs frozen at `CeraEngine.newSession(config)` time.
/// Constructed via `new SessionConfig()` in JS (returns the cera
/// defaults: `maxSeqLen=null` → engine's effective max, `nKeep=0`,
/// `seed=null`, `ubatchSize=512`, `kvCompression=null`).
///
/// Set `kvCompression` to a `TurboQuantConfig` to compress the
/// KV cache (~3 bits/elem for keys, ~2 bits/elem for values).
/// See the per-property doc for trade-offs.
#[wasm_bindgen]
#[derive(Default, Clone)]
pub struct SessionConfig {
    inner: cera::SessionConfig,
}

#[wasm_bindgen]
impl SessionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap on total tokens held in KV. `null` (the common case)
    /// defers to the engine's effective max — i.e.
    /// `min(engine.contextSize, model.maxSeqLen)`. Set to a
    /// smaller value here to further lower the cap; values larger
    /// than the engine's effective max are still capped at it.
    #[wasm_bindgen(getter, js_name = maxSeqLen)]
    pub fn max_seq_len(&self) -> Option<u32> {
        self.inner.max_seq_len
    }
    #[wasm_bindgen(setter, js_name = maxSeqLen)]
    pub fn set_max_seq_len(&mut self, v: Option<u32>) {
        self.inner.max_seq_len = v;
    }

    /// Number of leading tokens pinned in KV across context shifts —
    /// a system prompt or persistent prefix that should survive
    /// when the cache fills. `0` (default) disables the pin.
    #[wasm_bindgen(getter, js_name = nKeep)]
    pub fn n_keep(&self) -> u32 {
        self.inner.n_keep
    }
    #[wasm_bindgen(setter, js_name = nKeep)]
    pub fn set_n_keep(&mut self, v: u32) {
        self.inner.n_keep = v;
    }

    /// Deterministic sampler seed. `null` (default) uses a fresh
    /// random seed per session — set this to make a session's
    /// outputs reproducible across runs (useful for testing /
    /// demos / regression checks).
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> Option<u64> {
        self.inner.seed
    }
    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, v: Option<u64>) {
        self.inner.seed = v;
    }

    /// Chunked-prefill batch size (tokens per micro-batch during
    /// the prefill pass). Smaller values give finer-grained
    /// `Session.cancel()` checkpoints during long prompt eval at
    /// some perf cost. cera's default is `512`.
    #[wasm_bindgen(getter, js_name = ubatchSize)]
    pub fn ubatch_size(&self) -> u32 {
        self.inner.ubatch_size
    }
    #[wasm_bindgen(setter, js_name = ubatchSize)]
    pub fn set_ubatch_size(&mut self, v: u32) {
        self.inner.ubatch_size = v;
    }

    /// KV cache compression configuration. `null` (default) stores
    /// keys and values as f32 — best fidelity, biggest memory
    /// footprint. Set to a `TurboQuantConfig` to **request**
    /// TurboQuant compression — keys to ~3 bits/elem, values to
    /// ~2 bits/elem (plus a norm word per vector); the same `seed`
    /// reproduces the same per-layer Hadamard rotations
    /// deterministically.
    ///
    /// **Silent fallbacks to be aware of:**
    /// - TurboQuant only kicks in when the loaded model's
    ///   attention `head_dim` is a power of two (a constraint of
    ///   the Hadamard rotation). If it isn't, cera logs a warning
    ///   and falls back to the uncompressed f32 path even with
    ///   this set — there's no JS-visible error, just no
    ///   compression.
    /// - `nKeep` (context-shift) is incompatible with TurboQuant.
    ///   Setting both gets a warning at session creation and the
    ///   `nKeep` value is ignored on KV overflow (the cache
    ///   overflows hard instead of shifting). Pick one.
    /// - This config drives the CPU session. `WebGpuSession` takes
    ///   no `SessionConfig` — it accepts its own `kvCompression`
    ///   argument on `create` instead, and its `kvCompression`
    ///   getter reports the mode that actually took effect. Its
    ///   `head_dim` constraint is stricter than the CPU's: a power
    ///   of two that is also `<= 128` and a multiple of 32, and
    ///   keys *and* values must both be compressed (a single-sided
    ///   debug config falls back to f32 there).
    ///
    /// Setting this consumes the JS-side `TurboQuantConfig`
    /// handle (wasm-bindgen's `Option<T>` parameter shape). Read
    /// back via the getter — which returns a fresh handle that's
    /// a snapshot, not a live link — if you need to inspect the
    /// current config without affecting it.
    ///
    /// Assign a fresh config per session. Reusing an already-
    /// consumed handle does **not** throw in a release build:
    /// wasm-bindgen lowers it to pointer 0, which arrives as
    /// `None`, so the second session silently gets uncompressed
    /// KV. (A `--dev` build does throw "Attempt to use a moved
    /// value" — so this is a bug that only appears in release.)
    #[wasm_bindgen(getter, js_name = kvCompression)]
    pub fn kv_compression(&self) -> Option<TurboQuantConfig> {
        match &self.inner.kv_compression {
            // f16 KV isn't a TurboQuant config; the wasm binding doesn't expose
            // an f16 knob yet, so it reads back as "no TurboQuant" (None). A
            // dedicated wasm f16 API is a follow-up.
            cera::kv_cache::KvCompression::None | cera::kv_cache::KvCompression::F16 => None,
            cera::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                Some(TurboQuantConfig {
                    seed: *seed,
                    keys: *keys,
                    values: *values,
                })
            }
        }
    }
    #[wasm_bindgen(setter, js_name = kvCompression)]
    pub fn set_kv_compression(&mut self, v: Option<TurboQuantConfig>) {
        self.inner.kv_compression = to_kv_compression(v.as_ref());
    }
}

/// `Option<&TurboQuantConfig>` → the engine's `KvCompression`.
///
/// Shared by the CPU [`SessionConfig`] setter and `WebGpuSession::create` so the
/// two entry points cannot drift on what `null` means or on which fields get
/// carried across. `None` maps to `KvCompression::None`, i.e. the backend's
/// uncompressed KV — f32 on both the CPU and WebGPU paths.
fn to_kv_compression(v: Option<&TurboQuantConfig>) -> cera::kv_cache::KvCompression {
    match v {
        None => cera::kv_cache::KvCompression::None,
        Some(tqc) => cera::kv_cache::KvCompression::TurboQuant {
            seed: tqc.seed,
            keys: tqc.keys,
            values: tqc.values,
        },
    }
}

/// TurboQuant KV-cache compression configuration. Construct via
/// `new TurboQuantConfig(seed)` for the common production setup
/// (both `keys` and `values` compressed); flip the per-side
/// toggles for debugging (e.g. to isolate how much drift each
/// side contributes).
///
/// - **Keys**: 2-bit PolarQuant + 1-bit QJL residual
///   (3 bits/elem + a packed norm word per vector).
/// - **Values**: 2-bit PolarQuant only (2 bits/elem + a packed
///   norm word per vector).
///
/// `seed` drives the per-layer randomized Hadamard rotations —
/// the same seed produces the same rotations deterministically,
/// so a seeded session with TurboQuant on stays bitwise-
/// reproducible across runs.
#[wasm_bindgen]
#[derive(Clone)]
pub struct TurboQuantConfig {
    seed: u64,
    keys: bool,
    values: bool,
}

#[wasm_bindgen]
impl TurboQuantConfig {
    /// Construct with the common production setup: both keys and
    /// values compressed. Pass an explicit `seed` so the per-layer
    /// rotations are reproducible.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            keys: true,
            values: true,
        }
    }

    /// Hadamard-rotation seed. Same seed → same rotations →
    /// reproducible KV cache contents (necessary for bitwise-
    /// identical replay across sessions).
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u64 {
        self.seed
    }
    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, v: u64) {
        self.seed = v;
    }

    /// Compress the K side of the KV cache. Default `true`.
    /// Useful to flip off when debugging quality regressions to
    /// isolate K-side vs V-side contribution.
    #[wasm_bindgen(getter)]
    pub fn keys(&self) -> bool {
        self.keys
    }
    #[wasm_bindgen(setter)]
    pub fn set_keys(&mut self, v: bool) {
        self.keys = v;
    }

    /// Compress the V side of the KV cache. Default `true`.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> bool {
        self.values
    }
    #[wasm_bindgen(setter)]
    pub fn set_values(&mut self, v: bool) {
        self.values = v;
    }
}

/// Per-call generation options. Constructed via `new GenerateOpts()`
/// in JS (returns the cera defaults: `maxTokens=256`,
/// `temperature=0.7`, `topP=0.9`, `topK=40`, no stop tokens, flush
/// every 16 tokens or 50 ms).
///
/// `minP` and `repetitionPenalty` are honored in the stochastic path
/// (`temperature > 0` and `topK != 1`); greedy/argmax decoding ignores them.
#[wasm_bindgen]
#[derive(Default)]
pub struct GenerateOpts {
    inner: cera::GenerateOpts,
}

#[wasm_bindgen]
impl GenerateOpts {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter, js_name = maxTokens)]
    pub fn max_tokens(&self) -> u32 {
        self.inner.max_tokens
    }
    #[wasm_bindgen(setter, js_name = maxTokens)]
    pub fn set_max_tokens(&mut self, v: u32) {
        self.inner.max_tokens = v;
    }

    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.inner.temperature
    }
    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, v: f32) {
        self.inner.temperature = v;
    }

    #[wasm_bindgen(getter, js_name = topP)]
    pub fn top_p(&self) -> f32 {
        self.inner.top_p
    }
    #[wasm_bindgen(setter, js_name = topP)]
    pub fn set_top_p(&mut self, v: f32) {
        self.inner.top_p = v;
    }

    #[wasm_bindgen(getter, js_name = topK)]
    pub fn top_k(&self) -> u32 {
        self.inner.top_k
    }
    #[wasm_bindgen(setter, js_name = topK)]
    pub fn set_top_k(&mut self, v: u32) {
        self.inner.top_k = v;
    }

    /// Min-p (relative) nucleus cutoff: drop tokens below `minP * pMax`.
    /// `0.0` (default) disables it. Honored in the stochastic path.
    #[wasm_bindgen(getter, js_name = minP)]
    pub fn min_p(&self) -> f32 {
        self.inner.min_p
    }
    #[wasm_bindgen(setter, js_name = minP)]
    pub fn set_min_p(&mut self, v: f32) {
        self.inner.min_p = v;
    }

    /// Repetition penalty over tokens generated this call. `1.0` (default)
    /// disables it. Honored in the stochastic path.
    #[wasm_bindgen(getter, js_name = repetitionPenalty)]
    pub fn repetition_penalty(&self) -> f32 {
        self.inner.repetition_penalty
    }
    #[wasm_bindgen(setter, js_name = repetitionPenalty)]
    pub fn set_repetition_penalty(&mut self, v: f32) {
        self.inner.repetition_penalty = v;
    }

    /// Token IDs that, if produced, end decoding with
    /// `finishReason = "Stop"`. Empty by default.
    #[wasm_bindgen(getter, js_name = stopTokens)]
    pub fn stop_tokens(&self) -> Vec<u32> {
        self.inner.stop_tokens.clone()
    }
    #[wasm_bindgen(setter, js_name = stopTokens)]
    pub fn set_stop_tokens(&mut self, v: Vec<u32>) {
        self.inner.stop_tokens = v;
    }

    /// Ignore end-of-generation: EOS and `stopTokens` are not honored, so
    /// decode always runs to `maxTokens`. For benchmark loops that must
    /// cover an exact token count. `false` by default.
    #[wasm_bindgen(getter, js_name = ignoreEos)]
    pub fn ignore_eos(&self) -> bool {
        self.inner.ignore_eos
    }
    #[wasm_bindgen(setter, js_name = ignoreEos)]
    pub fn set_ignore_eos(&mut self, v: bool) {
        self.inner.ignore_eos = v;
    }

    /// Lazy-grammar trigger token IDs (tool calling). When non-empty and a
    /// grammar is set (`GenerateOpts.setGrammar`), the grammar stays inactive until
    /// the model emits one of these tokens (e.g. the tool-call start marker
    /// from `Tokenizer.specialTokenId`), then constrains the call and
    /// deactivates on completion. Empty (default) → the grammar is active from
    /// the first token.
    #[wasm_bindgen(getter, js_name = grammarTriggerTokens)]
    pub fn grammar_trigger_tokens(&self) -> Vec<u32> {
        self.inner.grammar_trigger_tokens.clone()
    }
    #[wasm_bindgen(setter, js_name = grammarTriggerTokens)]
    pub fn set_grammar_trigger_tokens(&mut self, v: Vec<u32>) {
        self.inner.grammar_trigger_tokens = v;
    }

    #[wasm_bindgen(getter, js_name = flushEveryTokens)]
    pub fn flush_every_tokens(&self) -> u32 {
        self.inner.flush_every_tokens
    }
    #[wasm_bindgen(setter, js_name = flushEveryTokens)]
    pub fn set_flush_every_tokens(&mut self, v: u32) {
        self.inner.flush_every_tokens = v;
    }

    #[wasm_bindgen(getter, js_name = flushEveryMs)]
    pub fn flush_every_ms(&self) -> u32 {
        self.inner.flush_every_ms
    }
    #[wasm_bindgen(setter, js_name = flushEveryMs)]
    pub fn set_flush_every_ms(&mut self, v: u32) {
        self.inner.flush_every_ms = v;
    }

    /// Constrain decoding to a GBNF grammar (source text, e.g. a JSON grammar).
    /// Each step masks the logits so only tokens the grammar accepts are
    /// sampled. Throws a `JsError` if the grammar fails to compile; replaces any
    /// grammar set by a prior call. A setter can't surface the parse error, so
    /// this is a method rather than a `grammar` property.
    #[wasm_bindgen(js_name = setGrammar)]
    pub fn set_grammar(&mut self, gbnf: &str) -> Result<(), JsError> {
        let grammar = cera::grammar::Grammar::parse(gbnf).map_err(map_err)?;
        self.inner.grammar = Some(std::sync::Arc::new(grammar));
        Ok(())
    }

    /// Remove any grammar constraint, returning to unconstrained decoding.
    #[wasm_bindgen(js_name = clearGrammar)]
    pub fn clear_grammar(&mut self) {
        self.inner.grammar = None;
    }

    /// Whether a grammar constraint is currently set.
    #[wasm_bindgen(getter, js_name = hasGrammar)]
    pub fn has_grammar(&self) -> bool {
        self.inner.grammar.is_some()
    }
}

/// Summary returned from a completed `Session.generate` call.
#[wasm_bindgen]
pub struct GenerateSummary {
    inner: cera::GenerateSummary,
}

#[wasm_bindgen]
impl GenerateSummary {
    #[wasm_bindgen(getter, js_name = tokensGenerated)]
    pub fn tokens_generated(&self) -> u32 {
        self.inner.tokens_generated
    }

    #[wasm_bindgen(getter, js_name = promptEvalTokens)]
    pub fn prompt_eval_tokens(&self) -> u32 {
        self.inner.prompt_eval_tokens
    }

    #[wasm_bindgen(getter, js_name = promptEvalMs)]
    pub fn prompt_eval_ms(&self) -> u32 {
        self.inner.prompt_eval_ms
    }

    #[wasm_bindgen(getter, js_name = decodeMs)]
    pub fn decode_ms(&self) -> u32 {
        self.inner.decode_ms
    }

    /// Why decode ended. One of `"MaxTokens"`, `"Stop"`,
    /// `"Cancelled"`, `"ContextFull"`, or `"Error(<message>)"` —
    /// the `Error(...)` form preserves the inner string verbatim
    /// (no surrounding quotes), so JS callers can log it directly.
    #[wasm_bindgen(getter, js_name = finishReason)]
    pub fn finish_reason(&self) -> String {
        // `format!("{:?}", reason)` would render `Error(String)` as
        // `Error("...")` (with the Debug-quoted inner string).
        // Match each variant explicitly so the public shape matches
        // the doc comment: `Error(plain inner message)` and bare
        // names for the payload-free variants.
        match &self.inner.finish_reason {
            cera::FinishReason::MaxTokens => "MaxTokens".to_string(),
            cera::FinishReason::Stop => "Stop".to_string(),
            cera::FinishReason::Cancelled => "Cancelled".to_string(),
            cera::FinishReason::ContextFull => "ContextFull".to_string(),
            cera::FinishReason::GrammarDeadEnd => "GrammarDeadEnd".to_string(),
            cera::FinishReason::Error(msg) => format!("Error({msg})"),
        }
    }
}

/// A loaded LoRA adapter, ready to attach to a [`Session`] via `attachLora`.
/// Load it once (from bytes — the browser has no filesystem) and reuse the
/// handle across sessions; the factors are reference-counted internally.
#[wasm_bindgen]
pub struct LoraAdapters {
    inner: std::sync::Arc<cera::lora::LoraAdapterWeights>,
}

#[wasm_bindgen]
impl LoraAdapters {
    /// Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
    /// bytes. `alpha` is read from the adapter's `adapter.lora.alpha` metadata.
    #[wasm_bindgen(js_name = fromGgufBytes)]
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<LoraAdapters, JsError> {
        let inner = cera::lora::LoraAdapterWeights::from_gguf_bytes(std::sync::Arc::from(bytes))
            .map_err(map_err)?;
        Ok(LoraAdapters { inner })
    }

    /// Load a PEFT `.safetensors` adapter from bytes. PEFT keeps `alpha` in a
    /// sibling `adapter_config.json`, so pass it explicitly (`undefined` ⇒
    /// scale = 1, i.e. `alpha == rank`).
    #[wasm_bindgen(js_name = fromSafetensorsBytes)]
    pub fn from_safetensors_bytes(
        bytes: &[u8],
        alpha: Option<f32>,
    ) -> Result<LoraAdapters, JsError> {
        let inner = cera::lora::LoraAdapterWeights::from_safetensors_bytes(bytes, alpha)
            .map_err(map_err)?;
        Ok(LoraAdapters { inner })
    }

    /// Number of `(layer, target)` low-rank deltas the adapter carries.
    #[wasm_bindgen(js_name = targetCount)]
    pub fn target_count(&self) -> u32 {
        self.inner.target_count() as u32
    }
}

/// Stateful generation handle. Built via `CeraEngine.newSession(config)`.
///
/// JS callers seed the conversation by calling `appendText` /
/// `appendTokens` and then drive decode with `generate(opts, cb)`.
/// The callback fires once per flush boundary (every
/// `flushEveryTokens` decoded tokens, or `flushEveryMs` ms,
/// whichever comes first) with the new tokens.
///
/// **Worker note:** `generate` is synchronous and will block the
/// thread it runs on for the duration of decode (potentially
/// seconds). On the browser main thread that freezes the page —
/// always call from a Web Worker. On Node it also blocks the JS
/// event loop (libuv's background I/O thread pool keeps running,
/// but JS callbacks queue): use `worker_threads` for server
/// processes that need to handle other requests during inference;
/// one-off scripts are fine to run sync.
///
/// **Cancellation:** since the worker thread is blocked inside
/// `generate`, the worker's own `onmessage` handler can't run —
/// incoming `postMessage({kind:'cancel'})` queues but doesn't
/// dispatch until `generate` returns, so a flag set by that
/// handler can't be updated mid-decode. To cancel during a
/// running `generate` call, either call `session.cancel()` from inside
/// the token callback based on state it can observe directly
/// (elapsed time, token budget, accumulated content), or use
/// cross-thread shared memory signalling (`SharedArrayBuffer` +
/// `Atomics`) — see `cera-wasm/README.md` for the full
/// `SharedArrayBuffer` pattern, which requires cross-origin
/// isolation in browsers.
#[wasm_bindgen]
pub struct Session {
    inner: cera::Session,
    /// Model hidden dimension, cached at construction so `hiddenSize()` is a
    /// plain field read. wasm-bindgen guards each exported method with an
    /// internal `RefCell` borrow, so an uncached getter called from inside a
    /// `generate` callback (which holds the `&mut self` borrow) would panic with
    /// a borrow error; the cached read avoids re-borrowing `self.inner`.
    hidden_size: u32,
}

#[wasm_bindgen]
impl Session {
    /// Tokenize `text` using the session's tokenizer and append the
    /// result to the KV cache. Equivalent to
    /// `appendTokens(tokenizer.encode(text))` but avoids the round
    /// trip through JS for the encoded buffer.
    #[wasm_bindgen(js_name = appendText)]
    pub fn append_text(&mut self, text: &str) -> Result<(), JsError> {
        self.inner.append_text(text).map_err(map_cera_err)
    }

    /// Append already-tokenized IDs to the KV cache. Use when you
    /// need control over BOS/EOS framing or you've cached tokens
    /// from a previous encode.
    #[wasm_bindgen(js_name = appendTokens)]
    pub fn append_tokens(&mut self, tokens: &[u32]) -> Result<(), JsError> {
        self.inner.append_tokens(tokens).map_err(map_cera_err)
    }

    /// Model hidden dimension `D` — reshape a `[T*D]` hidden-states buffer into
    /// `[T][D]` with this. Reads a cached field (set at construction), so — unlike
    /// the `&mut self` compute methods — it's safe to call from inside a `generate`
    /// callback without a wasm-bindgen borrow panic.
    #[wasm_bindgen(js_name = hiddenSize)]
    pub fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Per-token last-layer hidden states (post-final-RMSNorm — the llama.cpp
    /// `--pooling none` vector) for `tokens`, as a `Float32Array` of length
    /// `tokens.length * hiddenSize` (row-major; token `t` channel `c` at
    /// `t*hiddenSize + c`). The wasm boundary copies the buffer into the JS heap
    /// once. Side-effect-free — does not disturb the generation KV.
    #[wasm_bindgen(js_name = hiddenStatesForTokens)]
    pub fn hidden_states_for_tokens(
        &mut self,
        tokens: &[u32],
    ) -> Result<js_sys::Float32Array, JsError> {
        let hs = self
            .inner
            .hidden_states_for_tokens(tokens)
            .map_err(map_cera_err)?;
        Ok(js_sys::Float32Array::from(hs.as_slice()))
    }

    /// Tokenize `text` and return its per-token hidden states as a `Float32Array`.
    #[wasm_bindgen(js_name = hiddenStatesForText)]
    pub fn hidden_states_for_text(&mut self, text: &str) -> Result<js_sys::Float32Array, JsError> {
        let hs = self
            .inner
            .hidden_states_for_text(text)
            .map_err(map_cera_err)?;
        Ok(js_sys::Float32Array::from(hs.as_slice()))
    }

    /// Mean-pooled hidden state — a single `Float32Array` of length `hiddenSize`
    /// (the common classifier path: pool in Rust, ship `D` floats not `T*D`).
    #[wasm_bindgen(js_name = hiddenStatesMeanPooled)]
    pub fn hidden_states_mean_pooled(
        &mut self,
        tokens: &[u32],
    ) -> Result<js_sys::Float32Array, JsError> {
        let pooled = self
            .inner
            .hidden_states_mean_pooled(tokens)
            .map_err(map_cera_err)?;
        Ok(js_sys::Float32Array::from(pooled.as_slice()))
    }

    /// Attach a [`LoraAdapters`] to this session. Applied to every subsequent
    /// forward pass — generation **and** hidden-states extraction — until
    /// removed or replaced (hot-swap), and preserved across `reset()`. Throws if
    /// the adapter's dimensions don't match the loaded model. Only affects tokens
    /// processed after the call (doesn't retroactively re-adapt cached KV).
    #[wasm_bindgen(js_name = attachLora)]
    pub fn attach_lora(&mut self, adapters: &LoraAdapters) -> Result<(), JsError> {
        self.inner
            .attach_lora_adapters(adapters.inner.clone())
            .map_err(map_cera_err)
    }

    /// Remove any attached LoRA adapter, returning to base-model inference.
    #[wasm_bindgen(js_name = removeLora)]
    pub fn remove_lora(&mut self) {
        self.inner.remove_lora_adapters();
    }

    /// Whether a LoRA adapter is currently attached to this session.
    #[wasm_bindgen(js_name = hasLora)]
    pub fn has_lora(&self) -> bool {
        self.inner.has_lora_adapters()
    }

    /// Append PCM audio samples (mono `f32`, normalized to roughly
    /// `[-1.0, 1.0]`) at `sample_rate` Hz.
    ///
    /// Non-16kHz inputs are automatically linearly resampled to 16 kHz.
    /// `samples` arrives as `Float32Array` on the JS side. The
    /// wasm-bindgen boundary copies the typed-array contents into
    /// wasm linear memory once — there's no per-element boxing
    /// (contrast with Kotlin's `List<Float>` 4× memory overhead
    /// flagged in PR #78). The `&[f32]` Rust signature matches
    /// `appendTokens(&[u32])` and avoids the per-call `Vec`
    /// allocation that an owned parameter would require.
    ///
    /// Errors today are thrown as JS `Error`s; the message string
    /// is the underlying `cera::CeraError::Display` text (same as
    /// `appendText` / `appendTokens` produce):
    /// - `"empty input"` if `samples.length === 0` — fast-fail at
    ///   the wasm boundary, parity with `appendText` /
    ///   `appendTokens` empty-input rejection.
    /// - `"modality not supported by this model"` when
    ///   `session.capabilities.audioIn === false`. Load the bundle's
    ///   mmproj through `CeraEngine.fromGgufParts` to get an
    ///   audio-capable session; `fromGgufBytes` cannot produce one.
    /// - `"backend: Session::append_audio: no audio encoder attached..."`
    ///   when the bundle claimed audio but its mmproj failed to parse.
    #[wasm_bindgen(js_name = appendAudio)]
    pub fn append_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), JsError> {
        if samples.is_empty() {
            return Err(map_cera_err(cera::CeraError::EmptyInput));
        }
        self.inner
            .append_audio(samples, sample_rate)
            .map_err(map_cera_err)
    }

    /// Encode an image and append its embeddings to the KV cache.
    ///
    /// `bytes` is an encoded image file (PNG or JPEG), not raw pixels: pass
    /// a `Uint8Array` over a `fetch` response, a `File`/`Blob`
    /// `arrayBuffer()`, or a canvas `toBlob` result.
    ///
    /// `maxLongSize` caps the longest side of the **encoded** image in
    /// pixels, trading detail for speed and token count:
    ///
    /// - `null`/omitted: use the session default
    ///   (`setImageMaxLongSize`, itself unset by default).
    /// - `0`: force *no* cap for this call, overriding a session default.
    /// - `n`: cap at `n` pixels.
    ///
    /// Requires a VL bundle (`capabilities.imageIn === true`), which means
    /// loading via `CeraEngine.fromGgufParts` with the vision mmproj.
    /// Otherwise this throws `"modality not supported by this model"`.
    /// Building `cera-wasm` with `--no-default-features` (dropping the `vl`
    /// feature) produces the same error, since the image decoders are gone.
    #[wasm_bindgen(js_name = appendImage)]
    pub fn append_image(
        &mut self,
        bytes: &[u8],
        max_long_size: Option<u32>,
    ) -> Result<(), JsError> {
        // Mirrors cera-ffi's mapping so the two bindings agree on what a
        // `0` means. Delegating to `append_image` for `None` (rather than
        // always calling the `_with_opts` form) is what keeps the session
        // default reachable.
        match max_long_size {
            None => self.inner.append_image(bytes),
            Some(0) => self.inner.append_image_with_opts(bytes, None),
            Some(n) => self.inner.append_image_with_opts(bytes, Some(n)),
        }
        .map_err(map_cera_err)
    }

    /// Set the session-default cap on the longest side of an appended
    /// image, in pixels. `null` clears it (no cap).
    ///
    /// Applies to later `appendImage` calls that pass no explicit
    /// `maxLongSize`. A per-call value always wins.
    #[wasm_bindgen(js_name = setImageMaxLongSize)]
    pub fn set_image_max_long_size(&mut self, max_long_size: Option<u32>) {
        self.inner.set_image_max_long_size(max_long_size);
    }

    /// Current KV cache position (number of tokens currently held).
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> u32 {
        self.inner.position()
    }

    /// Modality capability flags reported by the model backing
    /// this session. Same shape as `CeraEngine.capabilities` —
    /// see that getter for the `Capabilities` field documentation
    /// and the synthetic-text caveat that applies to all
    /// `fromGgufBytes`-loaded models today.
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> Capabilities {
        capabilities_to_js(self.inner.capabilities())
    }

    /// Flip the cancel atomic, requesting that any in-flight
    /// `generate` call exit at its next checkpoint with
    /// `finishReason = "Cancelled"`. Safe to call from any thread
    /// (including a Worker that owns this session — though wasm
    /// without SharedArrayBuffer makes cross-thread sharing
    /// unusual).
    #[wasm_bindgen]
    pub fn cancel(&self) {
        self.inner.cancel()
    }

    /// Clear the cancel flag without dropping any session state.
    /// Use this after observing a cancellation signal — either a
    /// thrown cancellation error from `appendText` / `appendTokens`
    /// (mid-prefill cancellation surfaces as a thrown error) or
    /// `summary.finishReason === "Cancelled"` on the value
    /// returned from `generate` (cancellation during decode is
    /// reported via the finish reason, not a thrown error) — when
    /// you want to resume work on the same session without losing
    /// the accumulated KV cache.
    ///
    /// Compared to `reset()`:
    /// - `clearCancel`: keeps KV state, `position`, and the
    ///   sampler intact; only flips the cancel atomic back to
    ///   `false`. Use for "interrupted but continuing" flows.
    /// - `reset()`: drops KV cache, `position`, last logits, and
    ///   re-seeds the sampler. Use for "clear conversation"
    ///   flows.
    ///
    /// **Call sequencing:** invoke this *after* `generate` /
    /// `appendText` / `appendTokens` has returned. Even though
    /// the underlying cera method takes `&self`, wasm-bindgen's
    /// JS-side borrow check on the `Session` wrapper rejects any
    /// method call (including this `&self` one) while another
    /// method is still borrowing the same handle — calling
    /// `session.clearCancel()` from inside a `generate` token
    /// callback would throw "recursive use of an object". The
    /// `&self` Rust shape matters in the native binding
    /// (`cera-ffi`) where there's no JS-side borrow check; in
    /// wasm it just means there's no `&mut self` cost on the cera
    /// core side.
    #[wasm_bindgen(js_name = clearCancel)]
    pub fn clear_cancel(&self) {
        self.inner.clear_cancel()
    }

    /// Drop accumulated state and return the session to a freshly-
    /// opened shape. Clears the KV cache, `position`, the last
    /// logits, and the cancel flag, then re-seeds the sampler from
    /// the `SessionConfig.seed` originally passed to `newSession`.
    ///
    /// Use this for "clear conversation" UI actions — it skips the
    /// per-session setup cost that `engine.newSession(config)`
    /// would pay (model + tokenizer Arc clones, sampler ctor),
    /// while still leaving the session indistinguishable from a
    /// fresh one.
    ///
    /// Sampler re-seed semantics:
    /// - `SessionConfig.seed = some bigint` — deterministic
    ///   sessions stay deterministic across `reset()`; the next
    ///   `generate` produces the same first token sequence as the
    ///   original.
    /// - `SessionConfig.seed = null` — the sampler picks a new
    ///   random seed on each `reset()`, so successive
    ///   conversations decorrelate.
    ///
    /// Engine-level disk prefix cache (when configured on
    /// `CeraEngine`) is not touched — those entries are
    /// engine-scoped, not session-scoped.
    ///
    /// **Threading:** unlike `cancel()` (which only flips an
    /// atomic and is safe to call concurrently with anything),
    /// `reset()` takes `&mut self` and rebuilds non-atomic
    /// internal state (KV cache, sampler). Must be called on
    /// the owning thread, with no in-flight `generate` /
    /// `appendText` / `appendTokens` running. The wasm-bindgen
    /// borrow check enforces this within a single Worker; if
    /// you share a `Session` across Workers via
    /// `SharedArrayBuffer`-style schemes, it's on you to
    /// serialize calls.
    #[wasm_bindgen]
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.inner.reset().map_err(map_cera_err)?;
        Ok(())
    }

    /// Decode tokens until `opts.maxTokens`, a stop token, EOS, or
    /// `cancel()` fires. The `onTextTokens` callback is invoked once
    /// per flush boundary with a `Uint32Array` of the latest tokens
    /// (*not* the cumulative buffer — concatenate yourself if you
    /// want the full sequence).
    ///
    /// Returns the `GenerateSummary` once decode finishes. Throws
    /// `JsError` on backend failure (the summary's `finishReason`
    /// already covers logical end conditions like `"Stop"` or
    /// `"ContextFull"`).
    #[wasm_bindgen]
    pub fn generate(
        &mut self,
        opts: &GenerateOpts,
        on_text_tokens: &js_sys::Function,
        on_audio_frames: Option<js_sys::Function>,
    ) -> Result<GenerateSummary, JsError> {
        let mut sink = JsTextSink {
            on_text: on_text_tokens,
            on_audio: on_audio_frames.as_ref(),
        };
        self.inner
            .generate(&opts.inner, &mut sink)
            .map(|inner| GenerateSummary { inner })
            .map_err(map_cera_err)
    }
}

/// Internal `ModalitySink` implementation that trampolines text
/// tokens and audio frames to JS callbacks.
struct JsTextSink<'a> {
    on_text: &'a js_sys::Function,
    on_audio: Option<&'a js_sys::Function>,
}

impl<'a> cera::ModalitySink for JsTextSink<'a> {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        // `Uint32Array::from(&[u32])` allocates JS-owned memory and
        // copies the slice in.
        let array = js_sys::Uint32Array::from(tokens);
        if let Err(err) = self.on_text.call1(&JsValue::null(), &array) {
            wasm_bindgen::throw_val(err);
        }
    }

    fn on_audio_frames(&mut self, pcm: &[f32], sample_rate: u32) {
        if let Some(cb) = self.on_audio {
            let array = js_sys::Float32Array::from(pcm);
            let rate = JsValue::from_f64(sample_rate as f64);
            if let Err(err) = cb.call2(&JsValue::null(), &array, &rate) {
                wasm_bindgen::throw_val(err);
            }
        }
    }

    fn on_done(&mut self, _reason: cera::FinishReason) {
        // The `GenerateSummary` already carries the finish reason.
    }
}

// ── WebGPU (wgpu) GPU-accelerated LFM2 inference ──────────────────────────
//
// Gated behind the `wgpu` cargo feature (which enables `cera/gpu`). Exposes an
// async, GPU-backed text-generation surface for the browser. It deliberately
// bypasses the synchronous `cera::Session`: the WebGPU backend can only be
// driven from the JS event loop (no blocking GPU readback on the main thread),
// so the whole prefill + decode loop is `async` and reads logits back via
// `GpuContext::download_*_async`. Prototype scope: LFM2 only, greedy decode.
// See devlog 000169.
#[cfg(feature = "wgpu")]
mod webgpu {
    use super::{Capabilities, Tokenizer, capabilities_to_js, console_info, console_warn, map_err};
    use cera::model::Model;
    use cera::time::Instant;
    use std::sync::Arc;
    use wasm_bindgen::prelude::*;

    /// Stream one decoded piece to the JS callback. Any exception it throws is
    /// fatal and re-thrown across the wasm boundary (mirroring
    /// `JsTextSink::on_text_tokens`); `throw_val` preserves the original error
    /// object so it lands in the caller's `try { ... } catch` around `generate`
    /// rather than being silently swallowed mid-decode.
    fn emit(on_token: &js_sys::Function, piece: &str) {
        if let Err(err) = on_token.call1(&JsValue::null(), &JsValue::from_str(piece)) {
            wasm_bindgen::throw_val(err);
        }
    }

    /// GPU-accelerated LFM2 session for the browser. Holds a WebGPU-resident
    /// model + KV/conv state and streams decoded token text to a JS callback.
    #[wasm_bindgen]
    pub struct WebGpuSession {
        model: cera::model::gpu_lfm2::GpuLfm2Model,
        tokenizer: Arc<cera::tokenizer::BpeTokenizer>,
        state: cera::kv_cache::InferenceState,
        eos: Option<u32>,
        /// CPU vision-encoder weights, parsed from the mmproj passed to
        /// `createWithParts`. `None` for a text-only session. Kept even when
        /// `gpu_vision_encoder` is present: it is the documented fallback for
        /// an oversized patch grid or a GPU encode failure.
        vision_encoder: Option<Arc<cera::model::vision_encoder::VisionEncoderWeights>>,
        /// The same tower uploaded to the GPU, built once at construction so
        /// it stays off the per-image path. `None` when the upload failed, in
        /// which case encoding falls back to `vision_encoder`.
        gpu_vision_encoder: Option<Arc<dyn cera::model::vision_encoder_gpu::VisionGpuEncode>>,
        /// CPU audio-encoder weights, parsed from the mmproj passed from an audio bundle.
        audio_encoder: Option<Arc<cera::model::audio_encoder::AudioEncoderWeights>>,
        /// GPU audio encoder instance if available.
        gpu_audio_encoder: Option<Arc<dyn cera::model::audio_encoder_gpu::AudioGpuEncode>>,
        /// CPU audio decoder weights for vocoder output generation.
        audio_decoder: Option<Arc<cera::model::audio_decoder::AudioDecoderWeights>>,
        /// CPU detokenizer weights for vocoder output generation.
        detok_weights: Option<Arc<cera::model::audio_decoder::DetokenizerWeights>>,
        /// GPU detokenizer and ISTFT instance if available.
        gpu_audio_decoder: Option<Arc<cera::model::wgpu_audio_decoder::WgpuAudioDecoder>>,
        /// Session-default cap on an appended image's longest side.
        image_max_long_size: Option<u32>,
        /// Generation defaults from the bundle manifest.
        generation_defaults: Option<cera::manifest::GenerationDefaults>,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    }

    #[wasm_bindgen]
    pub struct WebGpuCancelHandle {
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[wasm_bindgen]
    impl WebGpuCancelHandle {
        #[wasm_bindgen]
        pub fn cancel(&self) {
            self.cancel.store(true, std::sync::atomic::Ordering::Release);
        }

        #[wasm_bindgen(js_name = clearCancel)]
        pub fn clear_cancel(&self) {
            self.cancel.store(false, std::sync::atomic::Ordering::Release);
        }
    }

    #[wasm_bindgen]
    impl WebGpuSession {
        #[wasm_bindgen(js_name = cancelHandle)]
        pub fn cancel_handle(&self) -> WebGpuCancelHandle {
            WebGpuCancelHandle {
                cancel: self.cancel.clone(),
            }
        }

        #[wasm_bindgen]
        pub fn cancel(&self) {
            self.cancel.store(true, std::sync::atomic::Ordering::Release);
        }

        #[wasm_bindgen(js_name = clearCancel)]
        pub fn clear_cancel(&self) {
            self.cancel.store(false, std::sync::atomic::Ordering::Release);
        }
        /// Async constructor: initialize WebGPU (`requestAdapter` /
        /// `requestDevice` resolve on the JS event loop), parse the in-memory
        /// GGUF, upload the model to the GPU, and build a fresh inference
        /// state. `contextSize` defaults to 4096. Throws if WebGPU is
        /// unavailable, the bytes aren't a valid LFM2 GGUF, or the device
        /// rejects the model.
        ///
        /// `kvCompression` is optional and defaults to `null` (uncompressed f32
        /// KV), so existing two-argument callers are unaffected. Pass a
        /// `TurboQuantConfig` to request TurboQuant on the GPU-resident cache:
        /// keys to ~3 bits/elem and values to ~2, plus a 4-byte norm word per
        /// (token, KV head) vector, so against f32's 32 bits the KV slabs shrink
        /// ~10.7x rather than the ~12.8x the bit rates alone suggest. Concretely,
        /// for LFM2-1.2B (6 attention layers, 8 KV heads x head_dim 64) that is
        /// 24 KiB per token down to 2.25 KiB — at a 16K context, 384 MiB
        /// (~403 MB) of GPU-side KV becomes 36 MiB (~38 MB).
        ///
        /// Both trailing parameters are optional, so to request compression while
        /// keeping the default `contextSize`, pass an explicit placeholder for
        /// argument 2. The generated signature is
        /// `context_size?: number | null`, and `undefined` and `null` both mean
        /// "use the default":
        ///
        /// ```js
        /// await WebGpuSession.create(bytes, undefined, new TurboQuantConfig(1234n));
        /// ```
        ///
        /// Do **not** collapse that to `create(bytes, tqConfig)`. TypeScript
        /// rejects it, but plain JS does not: in a release build the config
        /// object is coerced by `>>> 0` to a `contextSize` of 0 and the
        /// compression argument goes missing, yielding an unusable session with
        /// no compression and no error. (A `--dev` build throws on the argument
        /// type assertion instead, so this only bites in release.)
        ///
        /// Setting this **consumes** the JS-side `TurboQuantConfig` handle
        /// (wasm-bindgen's by-value `Option<T>` parameter shape), exactly like
        /// the `SessionConfig.kvCompression` setter. Build a fresh config per
        /// session: reusing one across two `create` calls — two sessions, or a
        /// retry after a failed load — does **not** throw in a release build.
        /// wasm-bindgen lowers an already-moved handle to pointer 0, which
        /// arrives in Rust as `None`, so the second session silently runs
        /// uncompressed. That makes handle reuse a third silent-downgrade cause
        /// alongside the two below, and the `kvCompression` getter is again the
        /// only way to notice. (A `wasm-pack --dev` build *does* throw "Attempt
        /// to use a moved value" here, so this misbehaves only in release.)
        ///
        /// The request is also **silently downgraded** when the WebGPU kernels
        /// can't serve it: they need a `head_dim` that is a power of two,
        /// `<= 128`, and a multiple of 32, and they only implement compressing
        /// keys *and* values together. Anything else falls back to uncompressed
        /// KV. The engine records that as a `tracing::warn!`, and `cera-wasm`
        /// installs no tracing subscriber, so **nothing reaches the browser
        /// console** — read the `kvCompression` getter to see what took effect.
        //
        // No rustdoc intra-doc links in the doc block above: wasm-bindgen copies
        // doc comments verbatim into `cera_wasm.d.ts`, so a `crate::`- or
        // `Self::`-qualified link ships to TS consumers as raw markdown pointing
        // at a Rust path they cannot follow. Plain code spans only.
        #[wasm_bindgen(js_name = create)]
        pub async fn create(
            bytes: Vec<u8>,
            context_size: Option<u32>,
            kv_compression: Option<crate::TurboQuantConfig>,
        ) -> Result<WebGpuSession, JsError> {
            Self::from_model_bytes(Arc::from(bytes), context_size, kv_compression).await
        }

        /// The body of `create`, over an `Arc<[u8]>` the caller already holds.
        ///
        /// Split out for `fromBundleId`, whose bytes come from the store as an
        /// `Arc<[u8]>` and must never be turned back into a `Vec` for the sake
        /// of a signature: that is a second full copy of the model.
        ///
        /// Not `#[wasm_bindgen]`: it takes a Rust type, which is the point.
        async fn from_model_bytes(
            gguf_bytes: Arc<[u8]>,
            context_size: Option<u32>,
            kv_compression: Option<crate::TurboQuantConfig>,
        ) -> Result<WebGpuSession, JsError> {
            let ctx = cera::backend::wgpu::GpuContext::new_async()
                .await
                .map_err(map_err)?;
            let ctx_size = context_size.unwrap_or(4096) as usize;
            let gguf = cera::gguf::GgufFile::from_bytes(gguf_bytes).map_err(map_err)?;
            // Build the tokenizer before the GGUF is moved into the model.
            let tokenizer =
                Arc::new(cera::tokenizer::BpeTokenizer::from_gguf(&gguf).map_err(map_err)?);
            let eos = tokenizer.eos_token();
            let model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_ctx(
                gguf,
                ctx_size,
                String::new(),
                ctx,
            )
            .map_err(map_err)?;
            // Builds the GPU-resident compressed cache. Must run before the first
            // forward pass — `Session::new` calls it at the same point.
            let compression = crate::to_kv_compression(kv_compression.as_ref());
            model
                .configure_kv_compression(&compression)
                .map_err(crate::map_cera_err)?;
            // Pass the mode even though `GpuLfm2Model` keeps its real KV on the
            // GPU and reads TurboQuant state from its own `tq` cache, never from
            // `InferenceState`. The reason is memory, not correctness:
            // `from_config_capped` replaces a side's f32 `key_cache`/`value_cache`
            // with a packed one as soon as that side is compressed. For
            // LFM2-1.2B at ctx 16K that trades 384 MiB of f32 for 43.5 MiB, an
            // 8.8x cut in wasm linear memory. Both are dead weight here — the GPU
            // reads neither — but in a browser tab the smaller dead weight is
            // worth one argument. Capping the allocation outright is still the
            // better fix; this just stops it dwarfing the win `create` advertises.
            //
            // Note the CPU-side figure is NOT the ~36 MiB quoted on `create`:
            // that is the GPU slab's 48 B per (token, KV head), whereas
            // `CompressedKeyCache`/`CompressedValueCache` also keep f32 mirrors of
            // the norms to avoid a per-call f16→f32 pass, costing 58 B. Same
            // compression, two layouts — don't reuse one's numbers for the other.
            //
            // The swap is per side, so a single-sided (debug) config still
            // reserves the full f32 vector for the uncompressed side while the
            // GPU downgrades to fully uncompressed — the memory argument above
            // only holds for the both-sides mode that production uses.
            //
            // Safe because the only consumers of the resulting
            // `state.is_compressed() == true` are `shift_kv` and
            // `Session::can_shift`, and `WebGpuSession` has no context-shift path
            // — it rejects an over-long prompt instead of shifting.
            let state = cera::kv_cache::InferenceState::from_config_with_compression(
                model.config(),
                &compression,
            )
            .map_err(crate::map_cera_err)?;
            Ok(WebGpuSession {
                model,
                tokenizer,
                state,
                eos,
                vision_encoder: None,
                gpu_vision_encoder: None,
                audio_encoder: None,
                gpu_audio_encoder: None,
                audio_decoder: None,
                detok_weights: None,
                gpu_audio_decoder: None,
                image_max_long_size: None,
                generation_defaults: None,
                cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            })
        }

        /// As `create`, but also attaches a vision tower from `mmproj`, giving
        /// the GPU session `appendImage`.
        ///
        /// Separate from `create` rather than an extra argument on it so an
        /// existing call keeps its meaning and its parameter order.
        ///
        /// The tower is uploaded to the GPU once here, not per image. If that
        /// upload fails the session still works and encodes on the CPU, which
        /// is slower but numerically equivalent; a malformed mmproj leaves
        /// `imageIn` false and `appendImage` throwing, exactly as on the CPU
        /// engine.
        #[wasm_bindgen(js_name = createWithParts)]
        pub async fn create_with_parts(
            bytes: Vec<u8>,
            mmproj: Vec<u8>,
            context_size: Option<u32>,
            kv_compression: Option<crate::TurboQuantConfig>,
        ) -> Result<WebGpuSession, JsError> {
            let mut session = Self::create(bytes, context_size, kv_compression).await?;
            session.attach_projector(Arc::from(mmproj))?;
            Ok(session)
        }

        /// Load a published LeapBundle by id and quantization onto the GPU,
        /// downloading through `repo` and reusing whatever it already cached.
        ///
        /// The GPU twin of `CeraEngine.fromBundleId`, and the one to prefer in
        /// a browser: the WebGPU backend measured ~58 tok/s against ~1.4 for
        /// the wasm CPU build on the same model, so reaching a bundle only
        /// through the CPU constructor is a ~40x cost.
        ///
        /// `onProgress(url, bytesDownloaded, totalBytes)` fires during
        /// downloads only; a fully cached bundle loads without calling it.
        /// `totalBytes` is `null` when the server doesn't say.
        ///
        /// **Memory:** the weights go from the store straight into wasm linear
        /// memory and stay for the session's lifetime. They are never handed to
        /// JS on the way, which is what keeps this clear of the ~2 GiB ceiling
        /// on a single JS `ArrayBuffer` that loading through `create` runs into,
        /// and costs one copy of the model rather than two.
        ///
        /// Throws for every reason `create` does, plus a bundle the GPU path
        /// cannot serve: it is LFM2-only. A caller wanting a fallback should catch
        /// and retry through `CeraEngine.fromBundleId`, which is what
        /// `cera_worker.js` does for `backend: 'auto'`.
        #[wasm_bindgen(js_name = fromBundleId)]
        pub async fn from_bundle_id(
            repo: &crate::bundle::BundleRepo,
            bundle_id: String,
            quant: String,
            context_size: Option<u32>,
            kv_compression: Option<crate::TurboQuantConfig>,
            on_progress: Option<js_sys::Function>,
        ) -> Result<WebGpuSession, JsError> {
            let parts =
                crate::bundle::load_bundle(repo, &bundle_id, &quant, on_progress.as_ref()).await?;
            let mut session =
                Self::from_model_bytes(parts.model, context_size, kv_compression).await?;
            session.generation_defaults = parts.generation_defaults;
            // A VL or audio bundle names its tower in the manifest, so unlike
            // `createWithParts` this path never has to be told about it.
            if let Some(mmproj) = parts.multimodal_projector {
                session.attach_projector(mmproj)?;
            }
            if let Some(voc_bytes) = parts.audio_decoder {
                session.attach_vocoder(voc_bytes)?;
            }
            Ok(session)
        }

        /// Parse `vocoder_bytes` as an audio decoder & detokenizer and attach it,
        /// giving this GPU session audio output synthesis via `AudioOutputDecoder`.
        fn attach_vocoder(&mut self, vocoder_bytes: Arc<[u8]>) -> Result<(), JsError> {
            let voc_gguf = cera::gguf::GgufFile::from_bytes(vocoder_bytes).map_err(map_err)?;
            let voc_arc = Arc::new(voc_gguf);
            let decoder_weights = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(
                &voc_arc,
            )
            .map_err(|e| JsError::new(&format!("failed to parse audio decoder weights: {e:#}")))?;
            let llm_hidden = cera::model::Model::config(&self.model).hidden_size;
            if decoder_weights.decoder_config.n_embd != llm_hidden {
                return Err(JsError::new(&format!(
                    "audio vocoder expects an LLM with hidden size {}, but the loaded model has {}",
                    decoder_weights.decoder_config.n_embd, llm_hidden
                )));
            }
            let detok_weights = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&voc_arc)
                .map_err(|e| {
                    JsError::new(&format!("failed to parse detokenizer weights: {e:#}"))
                })?;
            let ctx = self.model.ctx().clone();
            match cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_gguf_with_context(
                ctx, &voc_arc,
            ) {
                Ok(gad) => {
                    console_info(&format!(
                        "[cera-wasm] WgpuAudioDecoder loaded: supports_depthformer={}",
                        gad.supports_depthformer()
                    ));
                    self.gpu_audio_decoder = Some(Arc::new(gad));
                }
                Err(e) => {
                    console_info(&format!(
                        "[cera-wasm] WgpuAudioDecoder load failed ({e:#}), using CPU decoder"
                    ));
                    self.gpu_audio_decoder = None;
                }
            }
            self.audio_decoder = Some(Arc::new(decoder_weights));
            self.detok_weights = Some(Arc::new(detok_weights));
            Ok(())
        }

        /// Parse `mmproj` as a vision tower or audio encoder and attach it, giving
        /// this session `appendImage` or `appendAudio`.
        ///
        /// Shared by `createWithParts` and `fromBundleId` so the two cannot
        /// drift on the pairing check below.
        fn attach_projector(&mut self, mmproj: Arc<[u8]>) -> Result<(), JsError> {
            let proj_gguf = cera::gguf::GgufFile::from_bytes(Arc::clone(&mmproj)).map_err(map_err)?;
            let proj_arc = Arc::new(proj_gguf);
            let llm_hidden = cera::model::Model::config(&self.model).hidden_size;

            let is_audio = cera::model::audio_encoder::is_audio_encoder_gguf(&proj_arc);

            if is_audio {
                let weights = cera::model::audio_encoder::AudioEncoderWeights::from_gguf(&proj_arc)
                    .map_err(|e| {
                        JsError::new(&format!(
                            "failed to parse audio encoder mmproj weights: {e:#}"
                        ))
                    })?;
                let enc_hidden = weights.config.llm_hidden_size;
                if enc_hidden != llm_hidden {
                    return Err(JsError::new(&format!(
                        "audio encoder's llm_hidden_size ({enc_hidden}) does not match the \
                         model's hidden_size ({llm_hidden}); the mmproj must pair with \
                         the LLM it was trained against"
                    )));
                }
                let weights = Arc::new(weights);
                // Audio encoding on WebGPU falls back to the CPU encoder path.
                self.gpu_audio_encoder = None;
                self.audio_encoder = Some(weights);
                if self.audio_decoder.is_none() {
                    let _ = self.attach_vocoder(Arc::clone(&mmproj));
                }
                Ok(())
            } else {
                let weights =
                    cera::model::vision_encoder::VisionEncoderWeights::from_gguf(&proj_arc)
                        .map_err(|e| {
                            JsError::new(&format!(
                                "failed to parse vision encoder mmproj weights: {e:#}"
                            ))
                        })?;
                let proj_dim = weights.config.projection_dim;
                if proj_dim != llm_hidden {
                    return Err(JsError::new(&format!(
                        "vision encoder's projection_dim ({proj_dim}) does not match the \
                         model's hidden_size ({llm_hidden}); the mmproj must pair with \
                         the LLM it was trained against"
                    )));
                }
                let weights = Arc::new(weights);
                self.gpu_vision_encoder =
                    cera::model::vision_encoder_gpu::build_wgpu_vision_encoder_with_context(
                        self.model.ctx().clone(),
                        &weights,
                    );
                self.vision_encoder = Some(weights);
                Ok(())
            }
        }

        /// Number of tokens currently in the KV cache.
        ///
        /// The only reliable way for a caller to know whether this session is
        /// at the start of a sequence, which is what decides BOS framing. A
        /// counter kept outside cannot: a prompt can be rejected before any
        /// forward runs (leaving the cache untouched) or fail partway through
        /// decode (leaving it advanced), and the two need opposite answers.
        #[wasm_bindgen(getter)]
        pub fn position(&self) -> u32 {
            self.state.seq_len as u32
        }

        /// Feed `ids` into the KV cache without running any token generation.
        #[wasm_bindgen(js_name = appendTokens)]
        pub fn append_tokens(&mut self, ids: Vec<u32>) -> Result<(), JsError> {
            let max_seq_len = self.model.config().max_seq_len;
            let mut pos = self.state.seq_len;
            if pos.saturating_add(ids.len()) > max_seq_len {
                return Err(JsError::new(&format!(
                    "tokens of len {} plus {pos} already in context exceeds max sequence length {max_seq_len}",
                    ids.len()
                )));
            }
            for tok in ids {
                self.model.forward_prefill_step(tok, pos, &mut self.state);
                pos += 1;
            }
            self.state.seq_len = pos;
            Ok(())
        }

        /// Whether this session can accept images, i.e. whether it was built
        /// by `createWithParts` or `fromBundleId` with a usable vision mmproj.
        #[wasm_bindgen(getter, js_name = imageIn)]
        pub fn image_in(&self) -> bool {
            self.vision_encoder.is_some()
        }

        /// Whether this session can accept audio, i.e. whether it was built
        /// from an audio bundle with a usable audio mmproj.
        #[wasm_bindgen(getter, js_name = audioIn)]
        pub fn audio_in(&self) -> bool {
            self.audio_encoder.is_some()
        }

        /// Whether this session can produce audio output, i.e. whether it was built
        /// from an audio bundle with an audio vocoder sidecar attached.
        #[wasm_bindgen(getter, js_name = audioOut)]
        pub fn audio_out(&self) -> bool {
            self.audio_decoder.is_some() && self.detok_weights.is_some()
        }

        /// Modality capability flags for this session, same shape as
        /// `Session.capabilities` on the CPU path.
        #[wasm_bindgen(getter)]
        pub fn capabilities(&self) -> Capabilities {
            capabilities_to_js(cera::ModalityCapabilities {
                image_in: self.image_in(),
                audio_in: self.audio_in(),
                audio_out: self.audio_out(),
                ..cera::ModalityCapabilities::text_only()
            })
        }

        /// Set the session-default cap on an appended image's longest side in
        /// pixels; `null` clears it. A per-call `maxLongSize` still wins.
        #[wasm_bindgen(js_name = setImageMaxLongSize)]
        pub fn set_image_max_long_size(&mut self, max_long_size: Option<u32>) {
            self.image_max_long_size = max_long_size;
        }

        /// Encode an image (PNG or JPEG bytes) and append its embeddings to
        /// the KV cache, so a following `generate` / `generateTokens` sees it.
        ///
        /// `maxLongSize` follows the CPU session: `null` uses the session
        /// default, `0` forces no cap for this call, `n` caps at `n` pixels.
        ///
        /// Ordering is the caller's to manage, as with `generateTokens`:
        /// append the image where the chat template puts its `<image>` marker,
        /// which usually means framing the prompt in two halves around it.
        ///
        /// Throws when the session has no vision tower (`imageIn === false`),
        /// or when the image would overflow the context. Unlike the CPU
        /// session this cannot shift the KV cache to make room; that
        /// limitation is pre-existing and applies to prompts too.
        #[wasm_bindgen(js_name = appendImage)]
        pub async fn append_image(
            &mut self,
            bytes: &[u8],
            max_long_size: Option<u32>,
        ) -> Result<(), JsError> {
            let Some(encoder) = self.vision_encoder.as_ref() else {
                return Err(crate::map_cera_err(cera::CeraError::UnsupportedModality));
            };
            let cap = match max_long_size {
                None => self.image_max_long_size,
                Some(0) => None,
                Some(n) => Some(n),
            };

            let t_start = Instant::now();
            let pre = cera::model::vision_preprocessor::preprocess_image_with_opts(
                bytes,
                &encoder.config,
                cap,
            )
            .map_err(crate::map_cera_err)?;
            let t_pre = t_start.elapsed();

            let grid_tokens = pre.grid_w.saturating_mul(pre.grid_h);
            console_info(&format!(
                "[cera:worker] appendImage: 1/3 Preprocessed image ({}x{} grid = {grid_tokens} patches) in {:.1}ms",
                pre.grid_w,
                pre.grid_h,
                t_pre.as_secs_f64() * 1000.0
            ));

            // Prefer the GPU tower, but only within the attention kernel's
            // token capacity, and fall back rather than fail on a runtime GPU
            // error: the CPU encoder is always attached and numerically
            // equivalent. Same policy as `Session::append_image_with_opts`.
            let gpu = self
                .gpu_vision_encoder
                .as_ref()
                .filter(|_| grid_tokens <= cera::model::vision_encoder_gpu::MAX_VIT_TOKENS);

            let t_enc_start = Instant::now();
            let (img_tokens, backend_name) = match gpu {
                Some(g) => {
                    console_info(&format!(
                        "[cera:worker] appendImage: 2/3 Running WebGPU Vision Tower ({grid_tokens} patches, projection_dim={})...",
                        encoder.config.projection_dim
                    ));
                    match g
                        .encode_image_async(&pre.pixels, pre.grid_w, pre.grid_h)
                        .await
                    {
                        Ok(t) => (t, "WebGPU"),
                        Err(e) => {
                            console_warn(&format!(
                                "[cera:worker] appendImage: 2/3 WebGPU vision encode failed ({e:#}); falling back to WASM CPU encoder"
                            ));
                            (
                                encoder
                                    .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
                                    .map_err(map_err)?,
                                "WASM-CPU (fallback)",
                            )
                        }
                    }
                }
                None => {
                    if self.gpu_vision_encoder.is_some()
                        && grid_tokens > cera::model::vision_encoder_gpu::MAX_VIT_TOKENS
                    {
                        console_warn(&format!(
                            "[cera:worker] appendImage: 2/3 Image grid ({grid_tokens} patches) exceeds WebGPU ViT max ({}); running on WASM CPU (this may be slow)...",
                            cera::model::vision_encoder_gpu::MAX_VIT_TOKENS
                        ));
                    } else {
                        console_info(&format!(
                            "[cera:worker] appendImage: 2/3 Running WASM CPU Vision Tower ({grid_tokens} patches)..."
                        ));
                    }
                    (
                        encoder
                            .encode_image(&pre.pixels, pre.grid_w, pre.grid_h)
                            .map_err(map_err)?,
                        "WASM-CPU",
                    )
                }
            };
            let t_enc = t_enc_start.elapsed();
            console_info(&format!(
                "[cera:worker] appendImage: 2/3 Vision Tower encoded {grid_tokens} tokens on {backend_name} in {:.1}ms",
                t_enc.as_secs_f64() * 1000.0
            ));

            let hidden = cera::model::Model::config(&self.model).hidden_size;
            if hidden == 0 || !img_tokens.len().is_multiple_of(hidden) {
                return Err(JsError::new(&format!(
                    "vision encoder returned {} f32s, not a multiple of hidden_size \
                     ({hidden}), malformed image-token tensor",
                    img_tokens.len()
                )));
            }
            let n_tokens = img_tokens.len() / hidden;
            if n_tokens == 0 {
                return Err(crate::map_cera_err(cera::CeraError::EmptyInput));
            }

            // Check capacity before dispatching: this session has no
            // context-shift path, so an overflow has to be refused up front
            // rather than discovered halfway through the frames, which would
            // leave the cache holding half an image.
            let start = self.state.seq_len;
            let max = cera::model::Model::config(&self.model).max_seq_len;
            let end = start.checked_add(n_tokens).ok_or_else(|| {
                crate::map_cera_err(cera::CeraError::ContextOverflow {
                    max_seq_len: max as u32,
                    by: u32::MAX,
                })
            })?;
            if end > max {
                return Err(crate::map_cera_err(cera::CeraError::ContextOverflow {
                    max_seq_len: max as u32,
                    by: u32::try_from(end.saturating_sub(max)).unwrap_or(u32::MAX),
                }));
            }

            console_info(&format!(
                "[cera:worker] appendImage: 3/3 Seeding {n_tokens} embeddings into WebGPU LLM KV cache (pos {start} -> {end})..."
            ));
            let t_seed_start = Instant::now();
            self.model
                .seed_embeddings(&img_tokens, n_tokens, start, &mut self.state);
            let t_seed = t_seed_start.elapsed();

            debug_assert_eq!(
                self.state.seq_len, end,
                "seed_embeddings must advance seq_len by n_tokens"
            );

            let t_total = t_start.elapsed();
            console_info(&format!(
                "[cera:worker] appendImage: 3/3 KV cache seeded in {:.1}ms (total: {:.1}ms [prep: {:.1}ms, vit: {:.1}ms, kv: {:.1}ms])",
                t_seed.as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
                t_pre.as_secs_f64() * 1000.0,
                t_enc.as_secs_f64() * 1000.0,
                t_seed.as_secs_f64() * 1000.0
            ));
            Ok(())
        }

        /// Feed mono PCM audio into the conversation, to be processed by the
        /// next `generateTokens`.
        #[wasm_bindgen(js_name = appendAudio)]
        pub fn append_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), JsError> {
            let Some(encoder) = self.audio_encoder.as_ref() else {
                return Err(crate::map_cera_err(cera::CeraError::UnsupportedModality));
            };
            if samples.is_empty() {
                return Err(crate::map_cera_err(cera::CeraError::EmptyInput));
            }
            if !(1000..=192_000).contains(&sample_rate) {
                return Err(crate::map_cera_err(cera::CeraError::Backend(format!(
                    "unsupported audio sample rate: {sample_rate} Hz (expected 1000..=192000 Hz)"
                ))));
            }
            let target_sr = cera::model::audio_encoder::SAMPLE_RATE;
            let resampled: std::borrow::Cow<[f32]> = if sample_rate == target_sr {
                std::borrow::Cow::Borrowed(samples)
            } else {
                std::borrow::Cow::Owned(cera::model::audio_encoder::resample_linear(
                    samples,
                    sample_rate,
                    target_sr,
                ))
            };
            if resampled.is_empty() {
                return Err(crate::map_cera_err(cera::CeraError::EmptyInput));
            }
            let effective_samples = resampled.as_ref();
            let (embeddings, n_tokens) = match self.gpu_audio_encoder.as_ref() {
                Some(gpu) => match gpu.encode_pcm(effective_samples) {
                    Ok(out) => out,
                    Err(e) => {
                        tracing::warn!(
                            "gpu audio encode failed ({e:#}); falling back to CPU encoder"
                        );
                        cera::model::audio_encoder::encode_audio_pcm(
                            effective_samples,
                            encoder.as_ref(),
                        )
                    }
                },
                None => cera::model::audio_encoder::encode_audio_pcm(
                    effective_samples,
                    encoder.as_ref(),
                ),
            };
            if n_tokens == 0 {
                return Err(crate::map_cera_err(cera::CeraError::EmptyInput));
            }

            let start = self.state.seq_len;
            let max = cera::model::Model::config(&self.model).max_seq_len;
            let end = start.checked_add(n_tokens).ok_or_else(|| {
                crate::map_cera_err(cera::CeraError::ContextOverflow {
                    max_seq_len: max as u32,
                    by: u32::MAX,
                })
            })?;
            if end > max {
                return Err(crate::map_cera_err(cera::CeraError::ContextOverflow {
                    max_seq_len: max as u32,
                    by: u32::try_from(end.saturating_sub(max)).unwrap_or(u32::MAX),
                }));
            }

            self.model
                .seed_embeddings(&embeddings, n_tokens, start, &mut self.state);
            debug_assert_eq!(
                self.state.seq_len, end,
                "seed_embeddings must advance seq_len by n_tokens"
            );
            Ok(())
        }

        /// The KV-cache mode this session actually resolved to:
        /// `"turboquant(seed=N)"` or `"uncompressed"`.
        ///
        /// Read this after `create` to confirm a TurboQuant request was honored —
        /// a downgrade is silent in the browser, so this is the only JS-visible
        /// signal that compression is off. See `create` for what causes one.
        #[wasm_bindgen(getter, js_name = kvCompression)]
        pub fn kv_compression(&self) -> String {
            self.model.kv_mode_label()
        }

        /// Adapter + backend description of the WebGPU device (e.g.
        /// `"… (BrowserWebGpu)"`). Useful for confirming the GPU path is live.
        /// Note: headless Chrome's Dawn adapter reports an empty name.
        #[wasm_bindgen(getter)]
        pub fn adapter(&self) -> String {
            let (name, backend) = self.model.gpu_info();
            format!("{name} ({backend})")
        }

        /// Tokenize `prompt`, run prefill, then decode up to `maxTokens`
        /// tokens. `onToken(text)` is invoked for each decoded piece as it is
        /// produced; the full generated string is also returned. Stops early on
        /// the model's EOS token.
        ///
        /// Sampling follows the same rule as every other backend: greedy when
        /// `temperature <= 0` or `topK == 1`, stochastic otherwise. Omitted
        /// knobs fall back to `SamplerConfig`'s defaults, and `seed` makes a
        /// stochastic run reproducible. See `generateTokens` for what the
        /// stochastic path costs.
        // The four sampling knobs are passed flat rather than bundled, which
        // puts this one over clippy's limit. The CPU `Session::generate` takes
        // a `GenerateOpts`, but that type carries stop sequences, repetition
        // penalties and a context policy this path does not implement, so
        // accepting it here would advertise options the GPU decode loop
        // silently ignores. A knob that is visibly absent beats one that is
        // present and does nothing.
        #[allow(clippy::too_many_arguments)]
        #[wasm_bindgen]
        pub async fn generate(
            &mut self,
            prompt: &str,
            max_tokens: u32,
            temperature: Option<f32>,
            top_p: Option<f32>,
            top_k: Option<u32>,
            seed: Option<u64>,
            on_token: &js_sys::Function,
        ) -> Result<String, JsError> {
            let mut ids = self.tokenizer.encode(prompt);
            // Prepend BOS only at the start of a session, and only when the GGUF
            // declares `add_bos_token` — a model with a BOS id but
            // `add_bos_token = false` must not get a spurious leading BOS (which
            // would desync this path from cera's CLI/session paths and
            // llama.cpp). The on-GPU KV cache persists across `generate` calls,
            // so prepending on a continuation would inject a BOS at a nonzero
            // position mid-sequence. Also skip if the encoder already emitted it
            // (chat template / special token).
            if self.state.seq_len == 0
                && self.tokenizer.add_bos_token()
                && let Some(bos) = self.tokenizer.bos_token()
                && ids.first() != Some(&bos)
            {
                ids.insert(0, bos);
            }
            self.generate_ids(
                ids,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                on_token,
                None,
            )
            .await
        }

        /// The tokenizer this session's GGUF declares, for callers that need to
        /// frame a prompt themselves.
        ///
        /// Needed because rendering a chat template requires a tokenizer, and
        /// this session did not hand one out: a caller could feed it prompts
        /// but had no way to build one for a chat model.
        ///
        /// Note what is *not* the reason. `generate` encodes with the
        /// tokenizer's `encode`, which splits at special-token boundaries and
        /// emits `<|im_start|>` as its own id, so a rendered template fed
        /// straight to `generate` tokenizes correctly. Reach for
        /// `generateTokens` when you want to own the framing instead, which is
        /// what lets one caller frame identically across this session and the
        /// CPU one:
        ///
        /// ```js
        /// const tk = session.tokenizer;
        /// // A real array, not JSON: `applyChatTemplate` type-checks its
        /// // argument and rejects a string with "messages must be an array".
        /// const prompt = tk.applyChatTemplate(messages, true);
        /// // `encode`, not `encodeSpecial`: the latter also appends EOS when
        /// // the GGUF asks for one, which ends the turn before the model has
        /// // seen the generation header. It still lowers the template's
        /// // markers to their own ids, which is the part that matters.
        /// const ids = Array.from(tk.encode(prompt));
        /// if (tk.addBosToken && tk.bosToken != null && ids[0] !== tk.bosToken) {
        ///   ids.unshift(tk.bosToken);
        /// }
        /// await session.generateTokens(new Uint32Array(ids), 128, onToken);
        /// ```
        ///
        /// The returned handle shares this session's tokenizer rather than
        /// copying it, and is independent of the session's lifetime.
        #[wasm_bindgen(getter)]
        pub fn tokenizer(&self) -> Tokenizer {
            Tokenizer {
                inner: Arc::clone(&self.tokenizer),
            }
        }

        /// Prefill `tokens`, then decode up to `maxTokens`, exactly as
        /// `generate` does, but taking token ids the caller has already framed.
        ///
        /// **No BOS is prepended.** That is the whole point of this entry point:
        /// the caller owns framing, so a chat template's own leading special
        /// token is not competing with an injected one. `generate` is this
        /// method plus text encoding and the BOS rule.
        ///
        /// Like `generate`, this appends to the session's live KV cache rather
        /// than resetting it, so consecutive calls continue one conversation.
        ///
        /// **Sampling is not free here, unlike on the CPU.** Greedy decoding
        /// takes the argmax on the GPU and reads back four bytes; sampling has
        /// to read the whole logits row back so the sampler can see it, which
        /// is a vocab-sized transfer per token. The greedy path is kept for
        /// exactly that reason, so a caller that does not ask for sampling
        /// pays nothing for its availability.
        // Flat sampling knobs; see the note on `generate`.
        #[allow(clippy::too_many_arguments)]
        #[wasm_bindgen(js_name = generateTokens)]
        pub async fn generate_tokens(
            &mut self,
            tokens: Vec<u32>,
            max_tokens: u32,
            temperature: Option<f32>,
            top_p: Option<f32>,
            top_k: Option<u32>,
            seed: Option<u64>,
            on_token: &js_sys::Function,
            on_audio: Option<js_sys::Function>,
        ) -> Result<String, JsError> {
            self.generate_ids(
                tokens,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                on_token,
                on_audio.as_ref(),
            )
            .await
        }

        /// Shared body of `generate` / `generateTokens`: prefill, decode,
        /// UTF-8-safe streaming. Not exported; the two public entry points
        /// differ only in how `ids` is produced.
        // Flat sampling knobs; see the note on `generate`.
        #[allow(clippy::too_many_arguments)]
        async fn generate_ids(
            &mut self,
            ids: Vec<u32>,
            max_tokens: u32,
            temperature: Option<f32>,
            top_p: Option<f32>,
            top_k: Option<u32>,
            seed: Option<u64>,
            on_token: &js_sys::Function,
            on_audio: Option<&js_sys::Function>,
        ) -> Result<String, JsError> {
            if ids.is_empty() {
                return Ok(String::new());
            }

            // `WebGpuSession` is stateful: the on-GPU KV cache persists across
            // calls, so positions must continue from the current sequence
            // length, not restart at 0 (restarting corrupts RoPE and overwrites
            // live cache slots). `state.seq_len` is advanced by every forward.
            let max_seq_len = self.model.config().max_seq_len;
            let mut pos = self.state.seq_len;

            // The GPU forward asserts `seq_len < max_seq_len`, so a prompt that
            // doesn't fit in the remaining window would panic. Reject it with a
            // clear error instead.
            if pos + ids.len() > max_seq_len {
                return Err(JsError::new(&format!(
                    "prompt of {} tokens plus {pos} already in context exceeds \
                     the model's max sequence length of {max_seq_len}",
                    ids.len(),
                )));
            }

            // Prefill: feed every prompt token through the GPU to build the KV
            // cache. All but the last token skip the argmax + readback (their
            // predictions are unused), so an N-token prompt does a single
            // GPU→CPU round-trip instead of N. The last token's argmax is the
            // first generated token.
            let (last, prefix) = ids.split_last().expect("ids is non-empty");
            for &tok in prefix {
                self.model.forward_prefill_step(tok, pos, &mut self.state);
                pos += 1;
            }

            let (def_temp, def_top_p, def_top_k, def_min_p, def_rep_pen, audio_temp, audio_top_k) =
                match &self.generation_defaults {
                    Some(cera::manifest::GenerationDefaults::Text {
                        temperature,
                        top_p,
                        top_k,
                        min_p,
                        repetition_penalty,
                        ..
                    }) => (
                        *temperature,
                        *top_p,
                        *top_k,
                        *min_p,
                        *repetition_penalty,
                        0.8,
                        4,
                    ),
                    Some(cera::manifest::GenerationDefaults::Audio {
                        temperature,
                        top_p,
                        top_k,
                        min_p,
                        repetition_penalty,
                        audio_temperature,
                        audio_top_k,
                        ..
                    }) => (
                        *temperature,
                        *top_p,
                        *top_k,
                        *min_p,
                        *repetition_penalty,
                        audio_temperature.unwrap_or(0.8),
                        audio_top_k.map(|k| k as usize).unwrap_or(4),
                    ),
                    _ => (None, None, None, None, None, 0.8, 4),
                };

            // Greedy stays on the GPU-argmax path, which reads back four bytes
            // per token; sampling needs the whole logits row on the host, which
            // is a vocab-sized readback per token. Deciding once, here, keeps a
            // caller that did not ask for sampling on the cheap path.
            //
            // The rule matches `SamplerConfig`'s own definition of greedy, so
            // this backend agrees with the CPU on what `temperature: 0` means
            // rather than inventing a second threshold.
            let defaults = cera::sampler::SamplerConfig::default();
            let cfg = cera::sampler::SamplerConfig {
                temperature: temperature.or(def_temp).unwrap_or(defaults.temperature),
                top_p: top_p.or(def_top_p).unwrap_or(defaults.top_p),
                top_k: top_k
                    .or(def_top_k)
                    .map(|k| k as usize)
                    .unwrap_or(defaults.top_k),
                min_p: def_min_p.unwrap_or(defaults.min_p),
                repetition_penalty: def_rep_pen.unwrap_or(defaults.repetition_penalty),
                seed,
            };
            let mut sampler =
                (cfg.temperature > 0.0 && cfg.top_k != 1).then(|| cera::sampler::Sampler::new(cfg));

            let mut next = match sampler.as_mut() {
                Some(s) => {
                    let mut logits = self
                        .model
                        .forward_logits_async(*last, pos, &mut self.state)
                        .await
                        .map_err(map_err)?;
                    s.sample(&mut logits)
                }
                None => self
                    .model
                    .forward_greedy_async(*last, pos, &mut self.state)
                    .await
                    .map_err(map_err)?,
            };
            pos += 1;

            let mut decoder =
                if let (Some(dec), Some(detok)) = (&self.audio_decoder, &self.detok_weights) {
                    let gpu_ref: Option<&dyn cera::model::audio_decoder::AudioGpu> = self
                        .gpu_audio_decoder
                        .as_deref()
                        .map(|g| g as &dyn cera::model::audio_decoder::AudioGpu);
                    let use_gpu_df = self
                        .gpu_audio_decoder
                        .as_ref()
                        .is_some_and(|g| g.supports_depthformer());
                    console_info(&format!(
                        "[cera-wasm] AudioOutputDecoder: use_gpu_df={use_gpu_df}, gpu_audio_decoder={}",
                        self.gpu_audio_decoder.is_some()
                    ));
                    Some(
                        cera::audio_engine::AudioOutputDecoder::new(
                            dec,
                            detok,
                            gpu_ref,
                            audio_temp,
                            audio_top_k,
                            use_gpu_df,
                        )
                        .with_streaming(true),
                    )
                } else {
                    None
                };
            let is_tts = if let Ok(prompt_str) = std::str::from_utf8(&self.tokenizer.decode_bytes(&ids)) {
                prompt_str.contains("Perform TTS")
            } else {
                false
            };
            let is_interleaved = on_audio.is_some() && !is_tts;
            let mut modality_budget = if is_interleaved { 6usize } else { usize::MAX };
            let mut text_done = false;
            let mut trailing_audio_segments: usize = 0;
            const MAX_TRAILING_AUDIO_SEGMENTS: usize = 3;

            // Greedy decode loop. Stream raw bytes through a buffer and emit
            // only complete UTF-8: a multi-byte character can span several
            // byte-fallback tokens, so converting one token at a time would
            // corrupt non-ASCII output into U+FFFD replacement chars.
            let mut out = String::new();
            let mut pending = Vec::<u8>::new();
            let mut text_tokens_count = 0;
            let mut audio_frames_count = 0;
            let mut llm_hidden_passes = 0;

            let mut time_llm_text_ms = 0.0;
            let mut time_llm_audio_ms = 0.0;
            let mut time_depthformer_ms = 0.0;
            let mut time_vocoder_finish_ms = 0.0;

            let gen_start_time = js_sys::Date::now();

            self.cancel.store(false, std::sync::atomic::Ordering::Release);

            for _ in 0..max_tokens {
                if self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                if Some(next) == self.eos {
                    break;
                }

                if next == cera::audio_engine::TOKEN_AUDIO_START
                    && let Some(ref mut dec) = decoder
                {
                    console_info(
                        &format!("[cera-wasm] WebGpuSession hit audio start token ({}). Beginning sequential vocoder audio decoding...", next),
                    );
                    if !pending.is_empty() {
                        let valid = match std::str::from_utf8(&pending) {
                            Ok(s) => s.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if valid > 0 {
                            let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                            out.push_str(&piece);
                            emit(on_token, &piece);
                        }
                        pending.clear();
                    }

                    if pos < max_seq_len {
                        let t_emb0 = js_sys::Date::now();
                        let mut emb = self
                            .model
                            .forward_embedding_async(next, pos, &mut self.state)
                            .await
                            .map_err(map_err)?;
                        time_llm_audio_ms += js_sys::Date::now() - t_emb0;
                        pos += 1;

                        let mut frame_count = 0;
                        let can_use_gpu_buf = dec.supports_gpu_depthformer();
                        let mut use_gpu_buf = false;
                        loop {
                            if self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                                break;
                            }
                            if pos >= max_seq_len || audio_frames_count >= 4096 {
                                break;
                            }
                            let t_df0 = js_sys::Date::now();
                            let outcome = if use_gpu_buf && can_use_gpu_buf {
                                dec.decode_frame_from_gpu_hidden_async(self.model.hidden_buffer())
                                    .await
                                    .map_err(map_err)?
                            } else {
                                dec.decode_frame_async(&emb).await.map_err(map_err)?
                            };
                            time_depthformer_ms += js_sys::Date::now() - t_df0;

                            let audio_emb = match outcome {
                                cera::audio_engine::FrameOutcome::End => {
                                    console_info(&format!(
                                        "[cera-wasm] WebGpuSession vocoder emitted End code after {frame_count} frames"
                                    ));
                                    break;
                                }
                                cera::audio_engine::FrameOutcome::Codes {
                                    audio_embedding,
                                    pcm,
                                } => {
                                    frame_count += 1;
                                    audio_frames_count += 1;
                                    if !pcm.is_empty()
                                        && let Some(cb) = on_audio
                                    {
                                        let array = js_sys::Float32Array::from(pcm.as_slice());
                                        let rate_val = JsValue::from_f64(dec.sample_rate() as f64);
                                        let _ = cb.call2(&JsValue::null(), &array, &rate_val);
                                    }
                                    audio_embedding
                                }
                            };

                            let t_hid0 = js_sys::Date::now();
                            if can_use_gpu_buf {
                                self.model
                                    .forward_hidden_from_embedding_gpu(
                                        &audio_emb,
                                        pos,
                                        &mut self.state,
                                    )
                                    .map_err(map_err)?;
                                use_gpu_buf = true;
                            } else {
                                emb = self
                                    .model
                                    .forward_hidden_from_embedding_async(
                                        &audio_emb,
                                        pos,
                                        &mut self.state,
                                    )
                                    .await
                                    .map_err(map_err)?;
                            }
                            time_llm_audio_ms += js_sys::Date::now() - t_hid0;
                            llm_hidden_passes += 1;
                            pos += 1;
                        }
                    }
                    break;
                }

                if next == cera::audio_engine::TOKEN_TEXT_END {
                    text_done = true;
                }

                if next != cera::audio_engine::TOKEN_TEXT_END {
                    text_tokens_count += 1;
                    pending.extend_from_slice(&self.tokenizer.decode_bytes(&[next]));
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(s) => s.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid > 0 {
                        let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                        out.push_str(&piece);
                        emit(on_token, &piece);
                        pending.drain(..valid);
                    }
                }

                modality_budget = modality_budget.saturating_sub(1);

                // Interleaved audio generation: when modality budget hits 0 or text is done,
                // extract audio embedding from this token and decode up to 12 audio frames.
                if is_interleaved
                    && let Some(ref mut dec) = decoder
                    && (modality_budget == 0 || text_done)
                    && pos < max_seq_len
                {
                    if text_done {
                        trailing_audio_segments += 1;
                        if trailing_audio_segments > MAX_TRAILING_AUDIO_SEGMENTS {
                            break;
                        }
                    }

                    if !pending.is_empty() {
                        let valid = match std::str::from_utf8(&pending) {
                            Ok(s) => s.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if valid > 0 {
                            let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                            out.push_str(&piece);
                            emit(on_token, &piece);
                            pending.drain(..valid);
                        }
                    }

                    let t_emb0 = js_sys::Date::now();
                    let mut emb = self
                        .model
                        .forward_embedding_async(next, pos, &mut self.state)
                        .await
                        .map_err(map_err)?;
                    time_llm_audio_ms += js_sys::Date::now() - t_emb0;
                    pos += 1;

                    let mut audio_budget = 12usize;
                    let mut end_reached = false;
                    let can_use_gpu_buf = dec.supports_gpu_depthformer();
                    let mut use_gpu_buf = false;
                    loop {
                        if self.cancel.load(std::sync::atomic::Ordering::Relaxed) || pos >= max_seq_len {
                            break;
                        }
                        let t_df0 = js_sys::Date::now();
                        let outcome = if use_gpu_buf && can_use_gpu_buf {
                            dec.decode_frame_from_gpu_hidden_async(self.model.hidden_buffer())
                                .await
                                .map_err(map_err)?
                        } else {
                            dec.decode_frame_async(&emb).await.map_err(map_err)?
                        };
                        time_depthformer_ms += js_sys::Date::now() - t_df0;

                        let audio_emb = match outcome {
                            cera::audio_engine::FrameOutcome::End => {
                                if text_done {
                                    end_reached = true;
                                    break;
                                }
                                // Transition back to text by forwarding the audio end token
                                let t_trans0 = js_sys::Date::now();
                                next = match sampler.as_mut() {
                                    Some(s) => {
                                        let mut l = self
                                            .model
                                            .forward_logits_async(
                                                cera::audio_engine::TOKEN_TEXT_END,
                                                pos,
                                                &mut self.state,
                                            )
                                            .await
                                            .map_err(map_err)?;
                                        s.sample(&mut l)
                                    }
                                    None => self
                                        .model
                                        .forward_greedy_async(
                                            cera::audio_engine::TOKEN_TEXT_END,
                                            pos,
                                            &mut self.state,
                                        )
                                        .await
                                        .map_err(map_err)?,
                                };
                                time_llm_text_ms += js_sys::Date::now() - t_trans0;
                                pos += 1;
                                modality_budget = 6;
                                break;
                            }
                            cera::audio_engine::FrameOutcome::Codes {
                                audio_embedding,
                                pcm,
                            } => {
                                audio_frames_count += 1;
                                if !pcm.is_empty()
                                    && let Some(cb) = on_audio
                                {
                                    let array = js_sys::Float32Array::from(pcm.as_slice());
                                    let rate_val = JsValue::from_f64(dec.sample_rate() as f64);
                                    let _ = cb.call2(&JsValue::null(), &array, &rate_val);
                                }
                                audio_embedding
                            }
                        };
                        audio_budget = audio_budget.saturating_sub(1);

                        if audio_budget == 0 && !text_done {
                            // Transition back to text from the audio embedding
                            let t_trans0 = js_sys::Date::now();
                            next = match sampler.as_mut() {
                                Some(s) => {
                                    let mut l = self
                                        .model
                                        .forward_logits_from_embedding_async(
                                            &audio_emb,
                                            pos,
                                            &mut self.state,
                                        )
                                        .await
                                        .map_err(map_err)?;
                                    s.sample(&mut l)
                                }
                                None => self
                                    .model
                                    .forward_greedy_from_embedding_async(
                                        &audio_emb,
                                        pos,
                                        &mut self.state,
                                    )
                                    .await
                                    .map_err(map_err)?,
                            };
                            time_llm_text_ms += js_sys::Date::now() - t_trans0;
                            pos += 1;
                            modality_budget = 6;
                            break;
                        }

                        let t_hid0 = js_sys::Date::now();
                        if can_use_gpu_buf {
                            self.model
                                .forward_hidden_from_embedding_gpu(&audio_emb, pos, &mut self.state)
                                .map_err(map_err)?;
                            use_gpu_buf = true;
                        } else {
                            emb = self
                                .model
                                .forward_hidden_from_embedding_async(&audio_emb, pos, &mut self.state)
                                .await
                                .map_err(map_err)?;
                        }
                        time_llm_audio_ms += js_sys::Date::now() - t_hid0;
                        llm_hidden_passes += 1;
                        pos += 1;
                    }

                    if end_reached {
                        break;
                    }
                    continue;
                }

                // Stop before the next forward would overflow the context
                // window (the GPU forward asserts `seq_len < max_seq_len`). The
                // token just emitted is the last one this call can produce.
                if pos >= max_seq_len || text_tokens_count >= max_tokens as usize {
                    break;
                }
                let t_txt0 = js_sys::Date::now();
                next = match sampler.as_mut() {
                    Some(s) => {
                        let mut logits = self
                            .model
                            .forward_logits_async(next, pos, &mut self.state)
                            .await
                            .map_err(map_err)?;
                        s.sample(&mut logits)
                    }
                    None => self
                        .model
                        .forward_greedy_async(next, pos, &mut self.state)
                        .await
                        .map_err(map_err)?,
                };
                time_llm_text_ms += js_sys::Date::now() - t_txt0;
                pos += 1;
            }

            let mut total_samples = 0;
            if let Some(mut dec) = decoder
                && dec.audio_frames() > 0
                && let Some(cb) = on_audio
            {
                let t_fin0 = js_sys::Date::now();
                dec.finish_async(|pcm, rate| {
                    total_samples += pcm.len();
                    console_info(&format!(
                        "[cera-wasm] WebGpuSession vocoder finish produced {} PCM samples at {} Hz",
                        pcm.len(),
                        rate
                    ));
                    let array = js_sys::Float32Array::from(pcm);
                    let rate_val = JsValue::from_f64(rate as f64);
                    let _ = cb.call2(&JsValue::null(), &array, &rate_val);
                })
                .await
                .map_err(map_err)?;
                time_vocoder_finish_ms += js_sys::Date::now() - t_fin0;
                let final_samples = if dec.streamed_samples() > 0 {
                    dec.streamed_samples()
                } else {
                    total_samples
                };
                total_samples = final_samples;
                console_info(&format!(
                    "[cera-wasm] WebGpuSession audio generation finished with {} frames and {} total samples",
                    dec.audio_frames(),
                    total_samples
                ));
            }

            let total_gen_ms = js_sys::Date::now() - gen_start_time;
            let audio_dur_sec = total_samples as f64 / 24000.0;
            let rtf = if total_gen_ms > 0.0 {
                audio_dur_sec / (total_gen_ms / 1000.0)
            } else {
                0.0
            };

            let avg_txt = if text_tokens_count > 0 { time_llm_text_ms / text_tokens_count as f64 } else { 0.0 };
            let avg_aud = if llm_hidden_passes > 0 { time_llm_audio_ms / llm_hidden_passes as f64 } else { 0.0 };
            let avg_df = if audio_frames_count > 0 { time_depthformer_ms / audio_frames_count as f64 } else { 0.0 };

            console_info(&format!(
                "[cera:perf] Breakdown (total: {total_gen_ms:.1}ms, {audio_dur_sec:.2}s audio, RTF: {rtf:.2}x):\n\
                 - LLM Text Decode:      {time_llm_text_ms:.1}ms ({text_tokens_count} tokens, {avg_txt:.1}ms/tok)\n\
                 - LLM Audio Embedding:  {time_llm_audio_ms:.1}ms ({llm_hidden_passes} passes, {avg_aud:.1}ms/pass)\n\
                 - Depthformer (CPU/GPU):{time_depthformer_ms:.1}ms ({audio_frames_count} frames, {avg_df:.1}ms/frame)\n\
                 - Vocoder Detok+ISTFT:  {time_vocoder_finish_ms:.1}ms ({total_samples} PCM samples)"
            ));

            // Flush any trailing bytes (an incomplete multi-byte char at the
            // stop boundary — lossy as a last resort).
            if !pending.is_empty() {
                let piece = String::from_utf8_lossy(&pending).into_owned();
                out.push_str(&piece);
                emit(on_token, &piece);
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "wgpu")]
pub use webgpu::{WebGpuCancelHandle, WebGpuSession};

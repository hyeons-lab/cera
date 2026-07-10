//! `cera-ffi` — foreign-language bindings to [`cera`] via UniFFI.
//!
//! This crate exposes a subset of the `cera` inference engine to
//! Kotlin, Swift, Python, and any other language UniFFI supports. It
//! is structured around the **proc-macro** path (rather than a UDL
//! file) so the Rust types we expose are the source of truth and
//! annotations stay colocated with the code they describe.
//!
//! ## Current surface
//!
//! Engine-level:
//! - [`CeraEngine`] — model loader, session factory, and tokenizer
//!   accessor. Constructors: [`CeraEngine::from_path`] for a local
//!   GGUF, manifest, or directory; [`CeraEngine::from_bundle_id`]
//!   for LeapBundles-style remote loading;
//!   [`CeraEngine::from_bundle_id_async`] for the tokio-async variant.
//!   Tokenizer methods ([`CeraEngine::encode_text`],
//!   [`CeraEngine::decode_tokens`],
//!   [`CeraEngine::apply_chat_template`]) let foreign callers
//!   tokenize / detokenize / format messages without `Session`.
//! - [`ChatMessage`] — input record for `apply_chat_template`.
//! - [`BundleRepo`] — HTTP model cache; construct once per app and
//!   attach via [`EngineConfig::bundle_repo`] for remote loading.
//!   [`BundleRepo::with_progress`] takes a [`DownloadProgressSink`]
//!   foreign-trait callback for download progress UI.
//!   [`BundleRepo::cache_size`] / [`BundleRepo::clear_cache`] for
//!   on-disk usage queries + cleanup.
//! - [`EngineConfig`] + [`BackendPreference`] — load-time config.
//! - [`ModelMetadata`] + [`ModalityCapabilities`] — model-level info.
//!
//! Session-level:
//! - [`Session`] — stateful inference handle (one per conversation).
//!   `append_text` / `append_tokens` for input, synchronous
//!   [`Session::generate`] returning [`GenerateOutput`] (tokens +
//!   [`GenerateSummary`]), or [`Session::generate_streaming`] that
//!   delivers tokens + audio frames through a foreign [`ModalitySink`]
//!   as they're produced. Async twins
//!   [`Session::generate_async`] + [`Session::generate_streaming_async`]
//!   let foreign async runtimes — Kotlin coroutines, Swift `async`,
//!   Python `asyncio` — `.await` decode without stalling the caller.
//! - [`SessionConfig`] + [`KvCompression`] — per-session knobs.
//! - [`GenerateOpts`] + [`FinishReason`] — per-call decode config + exit reason.
//! - [`Session::cancel`] / [`Session::position`] for cooperative
//!   interrupt + progress monitoring across threads.
//! - [`ModalitySink`] — UniFFI foreign-trait callback for streaming
//!   decode output to Kotlin / Swift / Python implementations.
//!
//! Error:
//! - [`FfiError`] — typed error surface mirroring [`cera::CeraError`]
//!   one-to-one (`ContextOverflow { max_seq_len, by }`,
//!   `UnsupportedModality`, `UnsupportedInferenceType`, `Busy`,
//!   `Cancelled`, `EmptyInput`, `Io`), plus `Backend` for FFI-internal
//!   errors that have no cera analog (poisoned mutex, `JoinError`).
//!   The `From<CeraError>` conversion is exhaustive — new cera
//!   variants break compilation, never silently fall through to
//!   `Backend`.
//!
//! ## Not exposed yet
//!
//! Future PRs grow the surface per the roadmap in
//! `cera-ffi/README.md`. Highlights: remote URL loading through
//! `BundleRepo` (gated on the `remote` feature) and a parity harness
//! crate that cross-checks `cera-ffi` output against a reference
//! implementation.
//!
//! ## Design notes
//!
//! - **Wrapper types, not annotations on `cera` core.** Every
//!   UniFFI-exposed type is a wrapper defined in this crate with
//!   `From` conversions to/from the `cera` equivalent. The core crate
//!   stays UniFFI-agnostic.
//! - **`u64` on the wire, `usize` internally.** UniFFI records can't
//!   marshal `usize` (pointer-sized). Convert at the boundary.

use std::sync::Arc;

uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Typed error surface for `cera-ffi`. Mirrors [`cera::CeraError`] one-
/// to-one so foreign callers can pattern-match on error class (Kotlin
/// `when`, Swift `switch`, Python `match`) instead of string-sniffing
/// a generic message.
///
/// `Backend` is **not** a silent fallback for unmapped `cera::CeraError`
/// variants — the `From<CeraError>` impl is exhaustive, so adding a
/// new cera variant breaks compilation here. `Backend` exists solely
/// for FFI-internal errors that have no cera analog: `JoinError` from
/// a panicking `spawn_blocking` task, a poisoned `Session::inner`
/// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`.
///
/// Every variant carries the data needed to act on it:
/// `ContextOverflow` exposes `max_seq_len` and `by` so callers can
/// reset or truncate rather than re-reading the message;
/// `UnsupportedInferenceType` exposes the offending value;
/// `Io` preserves the underlying OS error message as a string since
/// `io::Error` isn't UniFFI-marshallable.
///
/// `#[error(...)]` format strings match `cera::CeraError` exactly for
/// every shared variant, so `Display` output is identical whether the
/// error originates from cera directly or routes through the FFI
/// wrapper. Pinned by `ffi_error_display_matches_cera_error_for_every_shared_variant`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    /// The loaded model doesn't support the modality the caller
    /// requested (e.g. `append_audio` on a text-only LLM).
    #[error("modality not supported by this model")]
    UnsupportedModality,

    /// The manifest's `inference_type` is one cera doesn't recognize
    /// at this version. Field carries the offending string.
    #[error("inference_type `{inference_type}` is not supported in this version of cera")]
    UnsupportedInferenceType { inference_type: String },

    /// A concurrent `generate*` call is already in flight on this
    /// session. Rust side guards with a mutex; this surfaces when the
    /// FFI detects contention.
    #[error("session is busy with another operation")]
    Busy,

    /// The caller (or the cancel-on-drop guard) flipped the cancel
    /// atomic mid-call. Surfaces from `append_text`, `append_tokens`,
    /// and `append_audio` when chunked prefill detects the cancel
    /// flag between micro-batches and aborts (see
    /// [`cera::Session::append_tokens`] for the chunked-prefill
    /// mechanism). Call [`Session::clear_cancel`] to reset the flag
    /// so the next call can proceed.
    ///
    /// `generate` reports cancellation via a different path: the
    /// call still returns `Ok` with a [`GenerateOutput`] whose
    /// `finish_reason` is set to `Cancelled`. Two paths because
    /// chunked prefill has nothing useful to return on cancel (no
    /// decoded tokens) while decode has accumulated tokens worth
    /// preserving.
    #[error("cancelled")]
    Cancelled,

    /// The context window is full and the session can't shift to make
    /// room (e.g. `n_keep == 0`, TurboQuant caches, or the active
    /// model doesn't support rope-shift). `max_seq_len` is the cap
    /// that was hit; `by` is the overshoot in tokens.
    #[error("context window ({max_seq_len}) exceeded by {by} tokens")]
    ContextOverflow { max_seq_len: u32, by: u32 },

    /// Input buffer was empty (e.g. `append_text("")`, or decode with
    /// no prefill state).
    #[error("empty input")]
    EmptyInput,

    /// Filesystem / mmap / network error surfaced from cera. The
    /// underlying `io::Error` isn't marshallable, so the message is
    /// flattened to a string. Callers that need the raw kind should
    /// parse the `detail` field or open an issue to request a typed
    /// field.
    ///
    /// Field is named `detail` rather than `message` because UniFFI's
    /// 0.31 Kotlin generator emits `class Io(val `message`) : FfiException()`
    /// AND `override val message` in the body when the field is literally
    /// named `message`, producing a "conflicting declarations" error
    /// (the constructor param collides with the inherited
    /// `Throwable.message` override). Renaming to `detail` sidesteps
    /// the collision.
    ///
    /// Format string matches `cera::CeraError::Io`'s `"io: {0}"` so
    /// foreign `.toString()` / `String(describing:)` gives the same
    /// output Rust consumers see.
    #[error("io: {detail}")]
    Io { detail: String },

    /// FFI-internal error with no cera analog: `JoinError` from a
    /// panicking `spawn_blocking` task, poisoned `Session::inner`
    /// mutex, 32-bit `u64 → usize` overflow in `EngineConfig::try_from`,
    /// or `cera::CeraError::Backend` routed through the `From` impl.
    /// Format string matches `cera::CeraError::Backend`'s
    /// `"backend: {0}"` — FFI-internal constructors that have already
    /// formatted a descriptive message (e.g. "generate_async join
    /// error: ...") still read cleanly with the `backend:` label.
    ///
    /// Field is named `detail` rather than `message` for the same
    /// `Throwable.message` collision reason as [`FfiError::Io`].
    #[error("backend: {detail}")]
    Backend { detail: String },

    /// The GBNF grammar string passed in `GenerateOpts.grammar` failed to
    /// compile. Grammar compilation happens in the FFI wrapper (the compiled
    /// grammar object can't cross the boundary, so callers pass the source text
    /// and it's parsed here). `detail` carries the parser's diagnostic.
    #[error("grammar: {detail}")]
    GrammarParse { detail: String },

    /// A token id passed to `hidden_states_for_tokens` (or another
    /// token-taking method) was `>= vocab_size`. Returned as a typed error
    /// rather than tripping the model-layer `assert!` (whose panic would
    /// unwind through the held session lock and poison it). Mirrors
    /// `cera::CeraError::InvalidToken`.
    #[error("token id {id} out of range (vocab_size {vocab_size})")]
    InvalidToken { id: u32, vocab_size: u32 },

    /// A LoRA adapter failed to load ([`LoraAdapters::from_gguf`] /
    /// [`LoraAdapters::from_safetensors`]) or was incompatible with the model at
    /// attach time (wrong dimensions). `detail` carries the diagnostic.
    #[error("lora: {detail}")]
    LoraParse { detail: String },

    /// A large model/KV allocation could not be satisfied — the device is out of
    /// memory for this model at this context size. Returned instead of aborting
    /// the process, so a caller can fall back (smaller model or context) or
    /// surface a clean error. Mirrors `cera::CeraError::OutOfMemory`.
    #[error("out of memory: could not allocate {requested_bytes} bytes")]
    OutOfMemory { requested_bytes: u64 },
}

impl From<cera::CeraError> for FfiError {
    fn from(e: cera::CeraError) -> Self {
        // Match exhaustively on the upstream enum so a future cera
        // variant-add breaks compilation here loudly rather than
        // silently routing through the `Backend` catch-all.
        match e {
            cera::CeraError::UnsupportedModality => FfiError::UnsupportedModality,
            cera::CeraError::UnsupportedInferenceType(s) => {
                FfiError::UnsupportedInferenceType { inference_type: s }
            }
            cera::CeraError::Busy => FfiError::Busy,
            cera::CeraError::Cancelled => FfiError::Cancelled,
            cera::CeraError::ContextOverflow { max_seq_len, by } => {
                FfiError::ContextOverflow { max_seq_len, by }
            }
            cera::CeraError::EmptyInput => FfiError::EmptyInput,
            cera::CeraError::InvalidToken { id, vocab_size } => {
                FfiError::InvalidToken { id, vocab_size }
            }
            cera::CeraError::Backend(s) => FfiError::Backend { detail: s },
            cera::CeraError::OutOfMemory { requested_bytes } => {
                FfiError::OutOfMemory { requested_bytes }
            }
            cera::CeraError::LoraDimMismatch(s) => FfiError::LoraParse { detail: s },
            cera::CeraError::Io(io_err) => FfiError::Io {
                detail: io_err.to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Config + enums
// ---------------------------------------------------------------------------

/// Compute-backend selector. Mirrors [`cera::BackendPreference`];
/// kept as a separate type so the `cera` crate doesn't carry UniFFI
/// annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum BackendPreference {
    /// Probe Metal → GPU → CPU at load time.
    Auto,
    Cpu,
    /// `wgpu` (Vulkan / Metal / DX12). Requires the `gpu` feature.
    Gpu,
    /// Native Metal. Requires the `metal` feature + macOS.
    Metal,
}

impl From<BackendPreference> for cera::BackendPreference {
    fn from(b: BackendPreference) -> Self {
        match b {
            BackendPreference::Auto => cera::BackendPreference::Auto,
            BackendPreference::Cpu => cera::BackendPreference::Cpu,
            BackendPreference::Gpu => cera::BackendPreference::Gpu,
            BackendPreference::Metal => cera::BackendPreference::Metal,
        }
    }
}

impl From<cera::BackendPreference> for BackendPreference {
    fn from(b: cera::BackendPreference) -> Self {
        match b {
            cera::BackendPreference::Auto => BackendPreference::Auto,
            cera::BackendPreference::Cpu => BackendPreference::Cpu,
            cera::BackendPreference::Gpu => BackendPreference::Gpu,
            cera::BackendPreference::Metal => BackendPreference::Metal,
        }
    }
}

/// Per-engine configuration at load time. Mirrors [`cera::EngineConfig`]
/// with `u64` fields (UniFFI doesn't marshal `usize`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct EngineConfig {
    /// KV-cache capacity in tokens. Capped by the model's own
    /// `max_seq_len`. Pass `0` to use the model's full declared
    /// `max_seq_len` (translated to `usize::MAX` internally, then
    /// capped by the loader).
    pub context_size: u64,
    pub backend: BackendPreference,
    /// Bundle repository for resolving `http(s)://` URLs in manifests
    /// (or for [`CeraEngine::from_bundle_id`]). `None` means "remote
    /// URLs will fail with an error"; set this to a [`BundleRepo`]
    /// rooted at a persistent cache directory to enable remote
    /// downloads. Construct the repo once + reuse it across engine
    /// loads so its HTTP client pool + on-disk cache are shared.
    pub bundle_repo: Option<Arc<BundleRepo>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        // Delegate to `cera::EngineConfig::default()` so the
        // defaults stay in one place. `usize → u64` is infallible on
        // every platform cera targets (`usize` is 32 or 64 bit; both
        // fit in u64).
        let core = cera::EngineConfig::default();
        Self {
            context_size: core.context_size as u64,
            backend: core.backend.into(),
            // `bundle_repo` defaults to None; foreign callers who want
            // remote-URL loading set it explicitly before passing the
            // config to `CeraEngine::from_path` / `from_bundle_id`.
            bundle_repo: None,
        }
    }
}

impl TryFrom<EngineConfig> for cera::EngineConfig {
    type Error = FfiError;

    fn try_from(c: EngineConfig) -> Result<Self, FfiError> {
        // Checked `u64 → usize` conversion. On 32-bit targets (Android
        // armv7 is still a supported ABI) `u64` can exceed `usize::MAX`
        // and a bare `as usize` would silently truncate — producing a
        // much smaller KV cache than the caller intended. Surface the
        // overflow as a typed error instead.
        let context_size = if c.context_size == 0 {
            // Sentinel for "use model default" — cera caps at model.max_seq_len.
            usize::MAX
        } else {
            usize::try_from(c.context_size).map_err(|_| FfiError::Backend {
                detail: format!(
                    "context_size {} exceeds usize::MAX on this target",
                    c.context_size
                ),
            })?
        };
        // Under the `remote` feature `cera::EngineConfig` carries a
        // `bundle_repo: Option<cera::bundle::BundleRepo>` field. Pull
        // the inner from our FFI `Arc<BundleRepo>` wrapper (cheap —
        // `cera::bundle::BundleRepo` is `Clone` and the two reqwest
        // clients inside share their connection pool via Arc-backed
        // refcounts).
        Ok(cera::EngineConfig {
            context_size,
            backend: c.backend.into(),
            bundle_repo: c.bundle_repo.map(|r| r.inner.clone()),
        })
    }
}

// ---------------------------------------------------------------------------
// Metadata + capabilities
// ---------------------------------------------------------------------------

/// Short summary of a loaded model. Mirrors [`cera::ModelMetadata`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelMetadata {
    pub architecture: String,
    pub max_seq_len: u32,
    pub vocab_size: u32,
    pub has_chat_template: bool,
    pub quantization: String,
    /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
    /// want to insert a BOS at the head of a raw prompt should honor it.
    pub add_bos_token: bool,
    /// SIMD backend tier the runtime resolved for this host (e.g.
    /// `"neon+dotprod"`, `"avx2"`, `"scalar"`). A host property, not
    /// model-specific — surfaced here so consumers fetching metadata also
    /// get backend diagnostics for telemetry / bug reports. For the full
    /// feature list, see [`cpu_backend_report`].
    pub cpu_backend: String,
}

impl From<&cera::ModelMetadata> for ModelMetadata {
    fn from(m: &cera::ModelMetadata) -> Self {
        ModelMetadata {
            architecture: m.architecture.clone(),
            max_seq_len: m.max_seq_len,
            vocab_size: m.vocab_size,
            has_chat_template: m.has_chat_template,
            quantization: m.quantization.clone(),
            add_bos_token: m.add_bos_token,
            cpu_backend: cera::cpu_tier().label().to_string(),
        }
    }
}

/// Modality support flags for a loaded model. Mirrors
/// [`cera::ModalityCapabilities`].
#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct ModalityCapabilities {
    pub text_in: bool,
    pub text_out: bool,
    pub image_in: bool,
    pub audio_in: bool,
    pub audio_out: bool,
}

impl From<cera::ModalityCapabilities> for ModalityCapabilities {
    fn from(c: cera::ModalityCapabilities) -> Self {
        ModalityCapabilities {
            text_in: c.text_in,
            text_out: c.text_out,
            image_in: c.image_in,
            audio_in: c.audio_in,
            audio_out: c.audio_out,
        }
    }
}

// ---------------------------------------------------------------------------
// ChatMessage (PR 13 — chat template input)
// ---------------------------------------------------------------------------

/// One message in a chat-template conversation. Mirrors
/// [`cera::tokenizer::ChatMessage`]. Pass a `Vec<ChatMessage>` to
/// [`CeraEngine::apply_chat_template`] to render the model's
/// chat-template (Jinja2 from GGUF metadata) into a prompt string
/// ready to feed into [`Session::append_text`].
///
/// `role` follows the OpenAI / chat-template convention — typically
/// one of `"system"`, `"user"`, `"assistant"`, occasionally
/// `"tool"`. cera-ffi doesn't validate the role string; whatever is
/// passed flows directly into the Jinja template. Whether an
/// unknown role errors or silently no-ops depends on the template's
/// own logic — many templates have an explicit error path for
/// unrecognized roles, but it's template-dependent rather than
/// enforced by [`CeraEngine::apply_chat_template`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl From<ChatMessage> for cera::tokenizer::ChatMessage {
    fn from(m: ChatMessage) -> Self {
        cera::tokenizer::ChatMessage {
            role: m.role,
            content: m.content,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------

/// The tool-call wire format a model family uses. Mirrors
/// [`cera::tools::ToolFormat`]. Get one from
/// [`CeraEngine::tool_format`] (auto-detected from the model) or set it
/// explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ToolFormat {
    /// LFM2 / LFM2.5: Pythonic `[get_weather(city="Paris")]` in
    /// `<|tool_call_start|>…<|tool_call_end|>`.
    Lfm2Pythonic,
    /// Hermes / Qwen: JSON `{"name":…,"arguments":{…}}` in
    /// `<tool_call>…</tool_call>`.
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

/// A tool the model may call. Mirrors [`cera::tools::ToolDef`], but the
/// JSON Schema for the arguments crosses the boundary as a JSON **string**
/// (`parameters_json`) since UniFFI has no arbitrary-JSON type. An empty
/// `parameters_json` means "no parameters".
#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema object for the arguments, as a JSON string (e.g.
    /// `{"type":"object","properties":{…},"required":[…]}`). Empty → none.
    pub parameters_json: String,
}

impl TryFrom<ToolDef> for cera::tools::ToolDef {
    type Error = FfiError;
    fn try_from(t: ToolDef) -> Result<Self, FfiError> {
        let parameters = if t.parameters_json.trim().is_empty() {
            serde_json::json!({ "type": "object", "properties": {} })
        } else {
            serde_json::from_str(&t.parameters_json).map_err(|e| FfiError::Backend {
                detail: format!("tool `{}` parameters_json is not valid JSON: {e}", t.name),
            })?
        };
        Ok(cera::tools::ToolDef {
            name: t.name,
            description: t.description,
            parameters,
        })
    }
}

/// A tool call parsed from model output. Mirrors [`cera::tools::ToolCall`];
/// `arguments_json` is the argument object encoded as a JSON string.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolCall {
    pub name: String,
    /// The argument object as a JSON string (e.g. `{"city":"Paris"}`).
    pub arguments_json: String,
}

impl From<cera::tools::ToolCall> for ToolCall {
    fn from(c: cera::tools::ToolCall) -> Self {
        ToolCall {
            name: c.name,
            arguments_json: serde_json::to_string(&c.arguments)
                .unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

fn to_core_tools(tools: Vec<ToolDef>) -> Result<Vec<cera::tools::ToolDef>, FfiError> {
    tools.into_iter().map(TryInto::try_into).collect()
}

/// Detect the tool-call format for a model architecture string (e.g.
/// `"lfm2"`, `"qwen3"`). Returns `None` for architectures with no known
/// convention — the caller may still choose a format explicitly.
#[uniffi::export]
pub fn detect_tool_format(architecture: String) -> Option<ToolFormat> {
    cera::tools::ToolFormat::detect(&architecture).map(Into::into)
}

/// Parse tool calls out of generated model text for the given `format`.
/// Returns an empty list when the reply contains no tool call (the model
/// answered in prose). Errors only when a call section is present but
/// unrecoverably malformed.
#[uniffi::export]
pub fn parse_tool_calls(text: String, format: ToolFormat) -> Result<Vec<ToolCall>, FfiError> {
    cera::tools::parse_tool_calls(&text, format.into())
        .map(|calls| calls.into_iter().map(Into::into).collect())
        .map_err(|e| FfiError::Backend {
            detail: format!("parse_tool_calls: {e}"),
        })
}

/// Build a GBNF grammar string constraining output to a valid call for one
/// of `tools`, in `format`. Put the result in `GenerateOpts.grammar` and set
/// `GenerateOpts.grammar_trigger_tokens` (see
/// [`CeraEngine::tool_call_start_token`]) for a lazy tool-call trigger.
#[uniffi::export]
pub fn tool_grammar(tools: Vec<ToolDef>, format: ToolFormat) -> Result<String, FfiError> {
    let core = to_core_tools(tools)?;
    cera::tools::tool_grammar(&core, format.into()).map_err(|e| FfiError::Backend {
        detail: format!("tool_grammar: {e}"),
    })
}

// ---------------------------------------------------------------------------
// BundleRepo
// ---------------------------------------------------------------------------

/// Remote model-bundle downloader + on-disk cache. Wraps
/// [`cera::bundle::BundleRepo`]; construct once per application with
/// a persistent `store_dir` and reuse across engine loads so the
/// HTTP client pool + downloaded-file cache are shared.
///
/// On Android the `store_dir` should typically be
/// `Context.getFilesDir()` (persistent), not `getCacheDir()` (OS-
/// purgeable under storage pressure). On iOS / macOS, the app's
/// Application Support or a dedicated subdirectory under Documents
/// is a reasonable baseline.
///
/// Cache layout mirrors the remote URL structure under
/// `<store_dir>/huggingface.co/<full path>`, so inspecting the
/// on-disk state with a file browser is straightforward and multiple
/// cera-powered apps on the same device can share the same cache
/// directory without conflicting.
#[derive(Debug, uniffi::Object)]
pub struct BundleRepo {
    inner: cera::bundle::BundleRepo,
}

#[uniffi::export]
impl BundleRepo {
    /// Create a new repo rooted at `store_dir`. The directory doesn't
    /// need to exist yet — it's created on the first download. Pass
    /// the same path to subsequent runs to reuse the cached bundles.
    #[uniffi::constructor]
    pub fn new(store_dir: String) -> Arc<Self> {
        Arc::new(Self {
            inner: cera::bundle::BundleRepo::new(store_dir),
        })
    }

    /// Create a new repo rooted at `store_dir` with a foreign
    /// [`DownloadProgressSink`] attached. The sink fires periodically
    /// during cache-miss downloads (every ~256 KB written + once at
    /// end-of-stream). Cache-hit resolves don't fire any callbacks.
    /// The same sink receives events for every file the repo
    /// downloads — distinguish per-file progress by the `url`
    /// argument on each callback.
    ///
    /// Construction-time attachment (rather than per-call) matches
    /// how mobile apps drive a single download-progress UI across
    /// multiple files in one logical bundle (manifest + GGUF + …):
    /// one repo, one sink, one progress bar. If you need to tear
    /// down the sink mid-app-lifecycle, drop the repo + construct a
    /// new one — Arc-based, so all in-flight calls finish on the
    /// old sink and new calls go to the new one.
    #[uniffi::constructor]
    pub fn with_progress(store_dir: String, progress: Arc<dyn DownloadProgressSink>) -> Arc<Self> {
        let adapter: Arc<dyn cera::bundle::DownloadProgress> =
            Arc::new(DownloadProgressAdapter { inner: progress });
        Arc::new(Self {
            inner: cera::bundle::BundleRepo::with_progress(store_dir, adapter),
        })
    }

    /// The directory this repo caches bundles under. Matches what was
    /// passed to [`BundleRepo::new`] / [`BundleRepo::with_progress`],
    /// useful for log / telemetry.
    pub fn store_dir(&self) -> String {
        self.inner.store_dir().to_string_lossy().into_owned()
    }

    /// Total bytes currently held in the cache. Returns `0` if the
    /// `store_dir` doesn't exist yet (no downloads have run).
    /// O(n) over the cache contents; for a multi-GB cache it's a
    /// real walk, not a constant-time query — UIs surfacing the
    /// value should run it off the main thread (e.g. via
    /// `withContext(Dispatchers.IO)` on Kotlin or
    /// `Task.detached` on Swift).
    ///
    /// Mobile apps use this to drive a "Storage: X MB used" line in
    /// settings or to gate a "Clear cache" button on actual
    /// non-zero usage.
    pub fn cache_size(&self) -> Result<u64, FfiError> {
        Ok(self.inner.cache_size()?)
    }

    /// Wipe every file the repo has cached, leaving `store_dir`
    /// itself in place so subsequent downloads land in the same
    /// path. Idempotent — calling on an empty repo or non-existent
    /// `store_dir` is a no-op success.
    ///
    /// Mobile apps trigger this from a "Clear downloaded models"
    /// settings action. Caller is responsible for serializing
    /// against in-flight downloads — typically trivial since the
    /// action is user-driven.
    pub fn clear_cache(&self) -> Result<(), FfiError> {
        Ok(self.inner.clear_cache()?)
    }
}

// ---------------------------------------------------------------------------
// DownloadProgressSink (foreign trait — PR 12)
// ---------------------------------------------------------------------------

/// Foreign-trait callback for download progress events from
/// [`BundleRepo::with_progress`]. Implementers (Kotlin class, Swift
/// class, Python subclass) drive a progress UI from these events.
///
/// All methods are required from foreign implementations (UniFFI
/// 0.31 foreign traits don't carry Rust default-impl fallbacks).
///
/// Threading: `on_progress` is invoked from the thread driving the
/// download. For sync `from_bundle_id` that's the caller's thread;
/// for `from_bundle_id_async` it's a tokio blocking worker. If your
/// progress UI requires marshalling onto a UI thread (`@MainActor`,
/// `runOnUiThread`, etc.), the implementer is responsible for the
/// dispatch.
#[uniffi::export(with_foreign)]
pub trait DownloadProgressSink: Send + Sync {
    /// Called periodically during a download. `bytes_downloaded` is
    /// monotonic across the same call's stream; `total_bytes` is the
    /// `Content-Length` reported by the server (may be `None` for
    /// chunked-transfer responses or when HEAD didn't surface a
    /// length). Same `url` value across all calls for one download
    /// — pattern-match on it to drive a per-file UI within a
    /// multi-file bundle download.
    ///
    /// Throttled by `cera-core` to ~256 KB granularity + one final
    /// callback at end-of-stream so the consumer always sees the
    /// final byte count.
    fn on_progress(&self, url: String, bytes_downloaded: u64, total_bytes: Option<u64>);
}

/// Adapter from the UniFFI foreign trait to cera-core's
/// [`cera::bundle::DownloadProgress`]. Same shape as
/// [`ForeignSinkAdapter`] for `ModalitySink` (PR 4): the foreign
/// arg's `&str` becomes an owned `String` because UniFFI can't
/// marshal a borrowed slice across the boundary.
struct DownloadProgressAdapter {
    inner: Arc<dyn DownloadProgressSink>,
}

// Manual Debug impl because `dyn DownloadProgressSink` is a UniFFI
// foreign-trait object — the foreign side (Kotlin / Swift / Python
// implementations) has no Rust Debug, so we can't blanket-derive.
// `cera::bundle::DownloadProgress` requires Debug for `BundleRepo`'s
// own derived Debug to work; printing the adapter as a typed handle
// is sufficient for any Rust-side log line that touches it.
impl std::fmt::Debug for DownloadProgressAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadProgressAdapter")
            .field("inner", &"<foreign DownloadProgressSink>")
            .finish()
    }
}

impl cera::bundle::DownloadProgress for DownloadProgressAdapter {
    fn on_progress(&self, url: &str, bytes_downloaded: u64, total_bytes: Option<u64>) {
        self.inner
            .on_progress(url.to_string(), bytes_downloaded, total_bytes);
    }
}

// ---------------------------------------------------------------------------
// CeraEngine
// ---------------------------------------------------------------------------

/// Owning handle to a loaded model. Mirrors [`cera::CeraEngine`];
/// `#[uniffi::Object]` requires `Arc<Self>` wrapping which matches how
/// the underlying engine is already used internally.
#[derive(uniffi::Object)]
pub struct CeraEngine {
    inner: cera::CeraEngine,
}

#[uniffi::export]
impl CeraEngine {
    /// Load a model from a local filesystem path. Accepts the same
    /// inputs as the native [`cera::CeraEngine::from_path`]: a bare
    /// `.gguf`, a LeapBundles `.json` manifest, or a directory
    /// containing exactly one `.json` manifest.
    ///
    /// If the manifest carries `http(s)://` URLs for its files,
    /// `config.bundle_repo` must be set — otherwise those URLs fail
    /// to resolve. For a pure-local workflow (bundle already on
    /// disk) leave `bundle_repo = None`.
    #[uniffi::constructor]
    pub fn from_path(path: String, config: EngineConfig) -> Result<Arc<Self>, FfiError> {
        let inner = cera::CeraEngine::from_path(&path, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load a model by LeapBundles ID + quantization selector, e.g.
    /// `from_bundle_id("LFM2-1.2B-GGUF", "Q4_0", config)`. Resolves
    /// to the matching `<bundle_id>/<quant>.json` manifest under
    /// `huggingface.co/LiquidAI/LeapBundles` and downloads whatever
    /// isn't already in `config.bundle_repo`'s on-disk cache.
    ///
    /// `config.bundle_repo` must be set; otherwise this returns an
    /// [`FfiError::Backend`] telling the caller to construct a
    /// [`BundleRepo`] and attach it. Idempotent across calls — the
    /// repo's cache deduplicates subsequent downloads.
    ///
    /// Blocking: this call fetches over the network on first run +
    /// opens / parses the GGUF. Foreign async runtimes should wrap
    /// the call in `spawn_blocking` / its equivalent. (An async
    /// counterpart matching `generate_async` could be added later;
    /// not in this PR.)
    #[uniffi::constructor]
    pub fn from_bundle_id(
        bundle_id: String,
        quant: String,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        let inner = cera::CeraEngine::from_bundle_id(&bundle_id, &quant, config.try_into()?)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Short summary of the loaded model (architecture, vocab size,
    /// max context, etc.). Returns a `Clone` of the stored metadata.
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata::from(self.inner.metadata())
    }

    /// What this model accepts as input / emits as output. Derived at
    /// load time from the manifest's `inference_type`.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.inner.capabilities().into()
    }

    /// Transcribe mono `f32` PCM audio (normalized to roughly `[-1.0, 1.0]`) to text using the
    /// model's trained `"Perform ASR."` chat mode. `sample_rate` must match the audio encoder's
    /// expected rate (resample beforehand if needed). Requires an audio-capable bundle; a text-only
    /// model returns an [`FfiError`] for unsupported modality.
    ///
    /// Blocking: runs a full prefill + greedy decode. Foreign async runtimes should wrap the call in
    /// `spawn_blocking` / its equivalent.
    pub fn transcribe(&self, pcm: Vec<f32>, sample_rate: u32) -> Result<String, FfiError> {
        Ok(self.inner.transcribe(&pcm, sample_rate)?)
    }

    /// Resolved context-window size (KV cache cap) the engine was
    /// configured with. Mirrors the `context_size` field of the
    /// [`EngineConfig`] passed to `from_path` / `from_bundle_id`,
    /// with the `0` → `model.max_seq_len` defaulting already
    /// applied so callers always see a meaningful number rather
    /// than the internal `usize::MAX` sentinel.
    ///
    /// Note this is the **engine-level** requested cap, not a
    /// per-session ceiling. cera core clamps the model's
    /// `max_seq_len` at load time to `min(requested_context,
    /// gguf_max_seq_len)` (see `cera/src/model/lfm2.rs`), so
    /// [`Self::metadata`]`.max_seq_len` is already the effective
    /// ceiling for any session built from this engine — `context_size`
    /// is informational ("what cap did this engine load with?")
    /// rather than a value callers should `min(...)` against.
    pub fn context_size(&self) -> u64 {
        let cs = self.inner.config().context_size;
        // EngineConfig::try_from maps a `0` request to `usize::MAX`
        // as a "use the model's own max" sentinel; resolve it back
        // to a real number for FFI consumers so they don't see a
        // 18-quintillion-token cap.
        if cs == usize::MAX {
            self.inner.metadata().max_seq_len as u64
        } else {
            cs as u64
        }
    }

    // ----- Tokenizer surface (PR 13) ---------------------------------
    //
    // Wraps `cera::tokenizer::BpeTokenizer` so foreign callers can
    // tokenize / detokenize / introspect the model's vocab without
    // going through `Session::append_text`. Useful for: pre-counting
    // prompt tokens before deciding to start a session, manual prompt
    // construction with explicit special tokens, decoding token IDs
    // returned from `generate` for incremental UI display.
    //
    // Tokenizer is shared across all sessions opened from this engine
    // (via Arc internally), so calling these methods concurrently with
    // a `generate*` is safe — they only read.

    /// Encode `text` into token IDs using the model's BPE tokenizer.
    /// Empty input returns an empty vec.
    pub fn encode_text(&self, text: String) -> Vec<u32> {
        self.inner.tokenizer().encode(&text)
    }

    /// Decode token IDs back to text. Out-of-vocab IDs are silently
    /// skipped (omitted from the decoded output) — `BpeTokenizer::decode`
    /// only appends bytes for IDs it has in `vocab.get(id)`. No
    /// substitution glyph, no error. Callers that want to detect
    /// invalid IDs should validate against `vocab_size()` first.
    pub fn decode_tokens(&self, tokens: Vec<u32>) -> String {
        self.inner.tokenizer().decode(&tokens)
    }

    /// Total vocabulary size — the number of distinct token IDs the
    /// model can emit. Sourced from the model's config (matches
    /// [`ModelMetadata::vocab_size`]) rather than the tokenizer's
    /// own count: in healthy models they match, but the model's
    /// config is the authoritative range for valid logit indices.
    pub fn vocab_size(&self) -> u32 {
        self.inner.metadata().vocab_size
    }

    /// Beginning-of-sequence token ID, if the model has one.
    /// LLaMA-family models typically do; some don't. Honor
    /// [`ModelMetadata::add_bos_token`] when deciding whether to
    /// prepend it manually to a prompt.
    pub fn bos_token(&self) -> Option<u32> {
        self.inner.tokenizer().bos_token()
    }

    /// End-of-sequence / end-of-text token ID, if the model has one.
    /// Used as a default stop-token by the sampler; callers can also
    /// pass it explicitly in [`GenerateOpts::stop_tokens`].
    pub fn eos_token(&self) -> Option<u32> {
        self.inner.tokenizer().eos_token()
    }

    /// Look up a special token by name (e.g. `<|im_start|>`,
    /// `<|im_end|>`, `<|tool_call|>`). Returns `None` if the token
    /// isn't defined in the tokenizer's vocab.
    pub fn special_token_id(&self, name: String) -> Option<u32> {
        self.inner.tokenizer().special_token_id(&name)
    }

    /// `true` when `id` is registered as a control or user-defined
    /// special token in the model's GGUF metadata
    /// (`tokenizer.ggml.token_type` types `3` / `4`). Useful for
    /// output filtering — e.g. dropping `<|im_end|>` from streamed
    /// tokens before rendering them to a UI — and for token-class
    /// classification in analysis tools.
    ///
    /// Out-of-range IDs (>= vocab size) and regular vocab tokens
    /// both return `false`. Companion to [`Self::special_token_id`]
    /// which goes the other direction (name → ID).
    pub fn is_special_token(&self, id: u32) -> bool {
        self.inner.tokenizer().is_special_token(id)
    }

    /// `true` if the model's tokenizer carries a chat template (a
    /// minijinja string from GGUF metadata). Foreign callers should
    /// check this before calling [`CeraEngine::apply_chat_template`].
    pub fn has_chat_template(&self) -> bool {
        self.inner.tokenizer().chat_template().is_some()
    }

    /// Render the model's chat template against a sequence of
    /// `ChatMessage`s. `add_generation_prompt = true` appends the
    /// model's "now it's the assistant's turn" suffix (typical when
    /// driving an interactive chat); `false` produces a transcript
    /// the model can keep continuing.
    ///
    /// Returns [`FfiError::Backend`] if the model has no chat
    /// template (check [`CeraEngine::has_chat_template`] first) or
    /// if the template fails to render against the supplied messages.
    pub fn apply_chat_template(
        &self,
        messages: Vec<ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String, FfiError> {
        let core_messages: Vec<cera::tokenizer::ChatMessage> =
            messages.into_iter().map(Into::into).collect();
        cera::tokenizer::apply_chat_template(
            self.inner.tokenizer(),
            &core_messages,
            add_generation_prompt,
        )
        .map_err(|e| FfiError::Backend {
            detail: format!("apply_chat_template: {e}"),
        })
    }

    /// Like [`CeraEngine::apply_chat_template`], but also passes a `tools`
    /// array so a tool-trained model renders its tool-definition block. Pass an
    /// empty `tools` for identical behavior to the plain call.
    pub fn apply_chat_template_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDef>,
        add_generation_prompt: bool,
    ) -> Result<String, FfiError> {
        let core_messages: Vec<cera::tokenizer::ChatMessage> =
            messages.into_iter().map(Into::into).collect();
        let core_tools = to_core_tools(tools)?;
        cera::tokenizer::apply_chat_template_with_tools(
            self.inner.tokenizer(),
            &core_messages,
            &core_tools,
            add_generation_prompt,
        )
        .map_err(|e| FfiError::Backend {
            detail: format!("apply_chat_template_with_tools: {e}"),
        })
    }

    /// The tool-call format auto-detected from this model's architecture, or
    /// `None` if the architecture has no known tool convention.
    pub fn tool_format(&self) -> Option<ToolFormat> {
        cera::tools::ToolFormat::detect(&self.inner.model().config().architecture).map(Into::into)
    }

    /// The token id of `format`'s tool-call start marker (e.g.
    /// `<|tool_call_start|>`) in this model's vocab, for use as a lazy grammar
    /// trigger in `GenerateOpts.grammar_trigger_tokens`. `None` if the model's
    /// tokenizer lacks that special token.
    pub fn tool_call_start_token(&self, format: ToolFormat) -> Option<u32> {
        let fmt: cera::tools::ToolFormat = format.into();
        self.inner
            .tokenizer()
            .special_token_id(fmt.call_start_marker())
    }
}

// ---------------------------------------------------------------------------
// Session types (PR 3)
// ---------------------------------------------------------------------------

/// KV-cache compression mode. Mirrors [`cera::kv_cache::KvCompression`].
/// `TurboQuant` is honored by the CPU backend only; Metal / GPU ignore
/// the setting and use the f32 path.
#[derive(Debug, Clone, Default, uniffi::Enum)]
pub enum KvCompression {
    /// No compression — f32 keys and values (default).
    #[default]
    None,
    /// TurboQuant compression. Both `keys` + `values` true is the
    /// production configuration; toggling them individually is
    /// primarily for debugging the drift contribution of each side.
    /// `seed` drives the per-layer randomized Hadamard rotations.
    TurboQuant { seed: u64, keys: bool, values: bool },
}

impl From<KvCompression> for cera::kv_cache::KvCompression {
    fn from(c: KvCompression) -> Self {
        match c {
            KvCompression::None => cera::kv_cache::KvCompression::None,
            KvCompression::TurboQuant { seed, keys, values } => {
                cera::kv_cache::KvCompression::TurboQuant { seed, keys, values }
            }
        }
    }
}

impl From<cera::kv_cache::KvCompression> for KvCompression {
    fn from(c: cera::kv_cache::KvCompression) -> Self {
        match c {
            cera::kv_cache::KvCompression::None => KvCompression::None,
            cera::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                KvCompression::TurboQuant { seed, keys, values }
            }
        }
    }
}

/// Per-session configuration. Mirrors [`cera::SessionConfig`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct SessionConfig {
    /// Cap on total tokens held in KV. `None` → model's default
    /// `max_seq_len`.
    pub max_seq_len: Option<u32>,
    /// KV cache compression mode.
    pub kv_compression: KvCompression,
    /// Pinned-prefix length for Phase-1.5 context shift on overflow.
    /// `0` disables shift; overflow returns `ContextOverflow` error.
    pub n_keep: u32,
    /// Deterministic sampling seed. `None` = fresh entropy per call.
    pub seed: Option<u64>,
    /// Chunked-prefill ubatch size. `0` = monolithic prefill.
    pub ubatch_size: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        // Delegate to `cera::SessionConfig::default()` so the defaults
        // stay in one place; `kv_compression` flows through the
        // wrapper's `From` impl (both directions live above).
        let core = cera::SessionConfig::default();
        Self {
            max_seq_len: core.max_seq_len,
            kv_compression: core.kv_compression.into(),
            n_keep: core.n_keep,
            seed: core.seed,
            ubatch_size: core.ubatch_size,
        }
    }
}

impl From<SessionConfig> for cera::SessionConfig {
    fn from(c: SessionConfig) -> Self {
        cera::SessionConfig {
            max_seq_len: c.max_seq_len,
            kv_compression: c.kv_compression.into(),
            n_keep: c.n_keep,
            seed: c.seed,
            ubatch_size: c.ubatch_size,
        }
    }
}

/// Per-call decode options. Mirrors [`cera::GenerateOpts`].
///
/// `flush_every_tokens` / `flush_every_ms` are accepted but have no
/// effect under the synchronous [`Session::generate`] — they're
/// meaningful once streaming (foreign-trait `ModalitySink`) lands
/// in a follow-up PR. Including them in the record now keeps the FFI
/// surface stable across that transition.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateOpts {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// Min-p (relative) nucleus cutoff: drop tokens below `min_p * p_max`. `0.0`
    /// disables it. Honored in the stochastic path.
    pub min_p: f32,
    /// Repetition penalty over tokens generated this call. `1.0` disables it.
    /// Honored in the stochastic path (greedy/argmax decoding is unaffected).
    pub repetition_penalty: f32,
    /// Early-stop IDs (EOS / instruction markers / end-of-turn).
    pub stop_tokens: Vec<u32>,
    /// Optional GBNF grammar **source text** constraining the output (e.g. a
    /// JSON grammar). When absent (the default), decoding is unconstrained. The
    /// grammar is compiled on the Rust side when generation starts; a malformed
    /// grammar is reported as a `GrammarParse` error.
    pub grammar: Option<String>,
    /// Lazy-grammar trigger token ids (tool calling). When non-empty and
    /// `grammar` is set, the grammar stays inactive until the model emits one
    /// of these tokens (e.g. the tool-call start marker from
    /// [`CeraEngine::tool_call_start_token`]), then constrains the call and
    /// deactivates on completion. Empty → `grammar` is active from the start.
    pub grammar_trigger_tokens: Vec<u32>,
    /// Ignored under synchronous generate; reserved for streaming.
    pub flush_every_tokens: u32,
    /// Ignored under synchronous generate; reserved for streaming.
    pub flush_every_ms: u32,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        let core = cera::GenerateOpts::default();
        Self {
            max_tokens: core.max_tokens,
            temperature: core.temperature,
            top_p: core.top_p,
            top_k: core.top_k,
            min_p: core.min_p,
            repetition_penalty: core.repetition_penalty,
            stop_tokens: core.stop_tokens,
            // Core default is no grammar; the compiled `Arc` has no FFI form, so
            // the mirrored field is the (absent) source string.
            grammar: None,
            grammar_trigger_tokens: core.grammar_trigger_tokens,
            flush_every_tokens: core.flush_every_tokens,
            flush_every_ms: core.flush_every_ms,
        }
    }
}

impl TryFrom<GenerateOpts> for cera::GenerateOpts {
    type Error = FfiError;

    /// Fallible because the GBNF `grammar` source is compiled here — a malformed
    /// grammar becomes [`FfiError::GrammarParse`] rather than silently decoding
    /// unconstrained.
    fn try_from(o: GenerateOpts) -> Result<Self, FfiError> {
        let grammar = match o.grammar {
            Some(src) => Some(Arc::new(cera::grammar::Grammar::parse(&src).map_err(
                |e| FfiError::GrammarParse {
                    detail: format!("{e:#}"),
                },
            )?)),
            None => None,
        };
        Ok(cera::GenerateOpts {
            max_tokens: o.max_tokens,
            temperature: o.temperature,
            top_p: o.top_p,
            top_k: o.top_k,
            min_p: o.min_p,
            repetition_penalty: o.repetition_penalty,
            stop_tokens: o.stop_tokens,
            grammar,
            grammar_trigger_tokens: o.grammar_trigger_tokens,
            flush_every_tokens: o.flush_every_tokens,
            flush_every_ms: o.flush_every_ms,
        })
    }
}

/// Why a decode loop exited. Mirrors [`cera::FinishReason`].
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FinishReason {
    MaxTokens,
    Stop,
    Cancelled,
    ContextFull,
    /// A grammar constraint left no token allowed at this step — decoding
    /// stopped because the grammar dead-ended. Only reachable when
    /// `GenerateOpts.grammar` is set.
    GrammarDeadEnd,
    Error {
        message: String,
    },
}

impl From<cera::FinishReason> for FinishReason {
    fn from(r: cera::FinishReason) -> Self {
        match r {
            cera::FinishReason::MaxTokens => FinishReason::MaxTokens,
            cera::FinishReason::Stop => FinishReason::Stop,
            cera::FinishReason::Cancelled => FinishReason::Cancelled,
            cera::FinishReason::ContextFull => FinishReason::ContextFull,
            cera::FinishReason::GrammarDeadEnd => FinishReason::GrammarDeadEnd,
            cera::FinishReason::Error(msg) => FinishReason::Error { message: msg },
        }
    }
}

/// Decode-run metadata. Mirrors [`cera::GenerateSummary`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateSummary {
    pub tokens_generated: u32,
    pub prompt_eval_tokens: u32,
    pub prompt_eval_ms: u32,
    pub decode_ms: u32,
    pub finish_reason: FinishReason,
}

impl From<cera::GenerateSummary> for GenerateSummary {
    fn from(s: cera::GenerateSummary) -> Self {
        Self {
            tokens_generated: s.tokens_generated,
            prompt_eval_tokens: s.prompt_eval_tokens,
            prompt_eval_ms: s.prompt_eval_ms,
            decode_ms: s.decode_ms,
            finish_reason: s.finish_reason.into(),
        }
    }
}

/// Bundle of everything a synchronous `generate` call produces:
/// the generated token IDs plus the decode summary. The two are
/// returned together so callers don't have to manage a separate
/// callback channel; streaming (per-chunk delivery) lands in PR 4.
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerateOutput {
    /// Generated token IDs, in order, not including any prompt
    /// tokens. Decode with [`cera::tokenizer::BpeTokenizer`] on the
    /// Rust side or (once exposed) through a tokenizer handle on the
    /// FFI side.
    pub tokens: Vec<u32>,
    pub summary: GenerateSummary,
}

// ---------------------------------------------------------------------------
// ModalitySink (foreign trait — PR 4)
// ---------------------------------------------------------------------------

/// Streaming sink for decode output. Foreign callers implement this
/// trait (Kotlin class, Swift class, Python subclass) and pass an
/// `Arc<dyn ModalitySink>` to [`Session::generate_streaming`] to
/// receive tokens + audio frames + the finish reason as they happen.
///
/// All methods are required from foreign implementations (UniFFI 0.28
/// foreign traits don't carry Rust's default-impl fallbacks). Callers
/// that don't care about a modality can provide an empty body.
///
/// Threading: every method is invoked on the same Rust thread running
/// `generate` — the decode thread. If the foreign runtime requires
/// marshalling onto a different thread (e.g. Swift's `@MainActor`) it
/// is the implementer's responsibility to dispatch the call there.
#[uniffi::export(with_foreign)]
pub trait ModalitySink: Send + Sync {
    /// Called with each chunk of generated token IDs. Ownership of the
    /// `Vec<u32>` is transferred to the callback, so implementations
    /// may retain or store it directly if needed — no clone required.
    fn on_text_tokens(&self, tokens: Vec<u32>);

    /// Called with each chunk of generated PCM audio samples. Not
    /// called for text-only models; LFM2-Audio-class models emit here.
    /// The `sample_rate` is the model's native output rate (typically
    /// 24000 for LFM2-Audio) and is stable across the whole generate
    /// call.
    fn on_audio_frames(&self, pcm: Vec<f32>, sample_rate: u32);

    /// Called exactly once per [`Session::generate_streaming`] call,
    /// as the last thing before the wrapper returns. Fires for both
    /// success (`MaxTokens`, `Stop`, `Cancelled`, `ContextFull`) and
    /// failure paths: on error the wrapper synthesizes a
    /// [`FinishReason::Error`] so foreign consumers have a reliable
    /// end-of-stream signal regardless of how the call exits.
    fn on_done(&self, reason: FinishReason);
}

/// Adapter from the UniFFI foreign trait to the internal
/// [`cera::ModalitySink`]. Forwards every call; unavoidable `Vec`
/// copy per chunk because UniFFI can't marshal a borrowed `&[u32]`
/// or `&[f32]` across the ABI boundary. Impact is bounded: the
/// decode loop emits chunks of at most a few tokens at a time, so
/// the allocation volume is orders of magnitude lower than the decode
/// itself. For audio the copy is larger but a single frame per decode
/// step is still small (a few hundred f32s).
///
/// `done_called` tracks whether the underlying `cera::Session::generate`
/// fired `on_done`. The FFI wrapper uses this to synthesize a
/// terminal `on_done(Error)` if core returns an error before getting
/// to its own `on_done` call (currently only possible on
/// `CeraError::EmptyInput`, but robust against future error paths).
/// Guards against double-firing if the core ever starts calling
/// `on_done` internally on error paths.
struct ForeignSinkAdapter {
    inner: Arc<dyn ModalitySink>,
    done_called: bool,
}

impl cera::ModalitySink for ForeignSinkAdapter {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.inner.on_text_tokens(tokens.to_vec());
    }
    fn on_audio_frames(&mut self, pcm: &[f32], sample_rate: u32) {
        self.inner.on_audio_frames(pcm.to_vec(), sample_rate);
    }
    fn on_done(&mut self, reason: cera::FinishReason) {
        self.done_called = true;
        self.inner.on_done(reason.into());
    }
}

// ---------------------------------------------------------------------------
// LoRA adapters
// ---------------------------------------------------------------------------

/// A loaded LoRA adapter, ready to attach to a [`Session`] via
/// [`Session::attach_lora`]. Load it once and share the handle across sessions —
/// it's reference-counted internally, so attaching to multiple sessions doesn't
/// re-parse or re-allocate the factors.
#[derive(uniffi::Object)]
pub struct LoraAdapters {
    inner: Arc<cera::lora::LoraAdapterWeights>,
}

#[uniffi::export]
impl LoraAdapters {
    /// Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
    /// a local path. `alpha` is read from the adapter's `adapter.lora.alpha`
    /// metadata (missing ⇒ scale = 1).
    #[uniffi::constructor]
    pub fn from_gguf(path: String) -> Result<Arc<Self>, FfiError> {
        let inner = cera::lora::LoraAdapterWeights::from_gguf(std::path::Path::new(&path))
            .map_err(|e| FfiError::LoraParse {
                detail: e.to_string(),
            })?;
        Ok(Arc::new(Self { inner }))
    }

    /// Load a PEFT `.safetensors` adapter from a local path. PEFT stores `alpha`
    /// in a sibling `adapter_config.json`, so pass it explicitly here (`None` ⇒
    /// scale = 1, i.e. `alpha == rank`).
    #[uniffi::constructor]
    pub fn from_safetensors(path: String, alpha: Option<f32>) -> Result<Arc<Self>, FfiError> {
        let inner =
            cera::lora::LoraAdapterWeights::from_safetensors(std::path::Path::new(&path), alpha)
                .map_err(|e| FfiError::LoraParse {
                    detail: e.to_string(),
                })?;
        Ok(Arc::new(Self { inner }))
    }

    /// Number of `(layer, target)` low-rank deltas the adapter carries — for
    /// diagnostics / logging.
    pub fn target_count(&self) -> u32 {
        self.inner.target_count() as u32
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// Stateful inference handle. Wraps [`cera::Session`] behind a
/// `Mutex` so UniFFI's `Arc<Session>` shape works with methods that
/// need `&mut self` on the inner session (prefill, generate, reset).
///
/// Call [`CeraEngine::new_session`] to open a session; the engine's
/// `Arc<Model>` and `Arc<BpeTokenizer>` are cloned into the new
/// session so it outlives the engine handle across FFI calls.
#[derive(uniffi::Object)]
pub struct Session {
    inner: std::sync::Mutex<cera::Session>,
    /// Cloned from the inner session at construction time. Shared
    /// atomic — `position()` / `cancel()` don't need to acquire the
    /// mutex, so they're safe to call from a different thread while
    /// `generate()` is running.
    position: Arc<std::sync::atomic::AtomicU32>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Stored at construction so `capabilities()` doesn't need a lock.
    capabilities: ModalityCapabilities,
    /// Model hidden dimension, cached at construction so `hidden_size()` is a
    /// lock-free read — safe to call from a `generate_streaming` sink callback
    /// (which runs while `generate` holds the mutex), same as `position()`.
    hidden_size: u32,
}

impl Session {
    /// Lock the inner session, converting `PoisonError` into
    /// `FfiError::Backend` instead of panicking. `expect` on a
    /// poisoned mutex would propagate as a panic across the FFI
    /// boundary — Kotlin / Swift / Python callers see that as an
    /// uncatchable abort of the host process, which is unusable in
    /// production. Returning an error lets callers decide whether to
    /// retry, reset, or surface the failure.
    ///
    /// A poisoned mutex here means a prior session method panicked
    /// while holding the lock — the session's internal state (KV
    /// cache, sampler, position counters) is therefore in an unknown
    /// state. The error message gives the caller enough context to
    /// decide whether to reset or drop the session entirely.
    fn lock_inner(&self) -> Result<std::sync::MutexGuard<'_, cera::Session>, FfiError> {
        self.inner.lock().map_err(|e| FfiError::Backend {
            detail: format!(
                "session mutex poisoned (a prior call panicked mid-lock; session state is \
                 inconsistent): {e}"
            ),
        })
    }
}

/// Flatten `f32`s into a little-endian byte buffer (`4 * len` bytes) so the wire
/// format is stable across host architectures; callers reinterpret 4-byte groups
/// as `f32`. On little-endian hosts (every UniFFI target in practice) the
/// in-memory `f32` bytes already ARE the LE wire format, so a single bulk copy
/// beats the per-element `to_le_bytes()` loop on the large `[T*D]` payloads.
fn f32_vec_to_le_bytes(v: &[f32]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice::<f32, u8>(v).to_vec()
    }
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for &x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        bytes
    }
}

#[uniffi::export]
impl Session {
    /// Append raw text to the context, running a prefill over just
    /// the new tokens. `EmptyInput` error if `text` is empty.
    pub fn append_text(&self, text: String) -> Result<(), FfiError> {
        self.lock_inner()?.append_text(&text)?;
        Ok(())
    }

    /// Append pre-tokenized IDs. Useful when the caller has its own
    /// tokenizer + chat-template pipeline.
    pub fn append_tokens(&self, tokens: Vec<u32>) -> Result<(), FfiError> {
        self.lock_inner()?.append_tokens(&tokens)?;
        Ok(())
    }

    /// Append PCM audio samples (mono `f32`, normalized to roughly
    /// `[-1.0, 1.0]`) at `sample_rate` Hz. The audio is encoded via
    /// the bundle's mmproj (`AudioEncoderWeights`) and prefilled
    /// into the LLM as soft tokens — see
    /// [`cera::Session::append_audio`] for the underlying flow.
    ///
    /// `CeraEngine::new_session` auto-attaches the encoder when the
    /// loaded bundle's `inference_type == LlamaCppLfm2AudioV1` and
    /// has `multimodal_projector` set in the manifest, so FFI
    /// consumers don't need any separate "load encoder" call.
    /// Bundles where the manifest omits `multimodal_projector`
    /// silently end up with no encoder attached (no log). Bundles
    /// where the file is named but fails to open or parse log a
    /// `tracing::warn!` at `CeraEngine` construction. Both cases
    /// surface here as a "no audio encoder attached" `Backend`
    /// error.
    ///
    /// `sample_rate` must be 16000 — resampling is out of scope.
    /// Callers should resample externally before passing samples in.
    ///
    /// **Marshaling cost**: UniFFI maps `Vec<f32>` to `List<Float>`
    /// in Kotlin and `[Float]` in Swift. The Kotlin side boxes each
    /// `Float` to `java.lang.Float`, a ~4× memory overhead vs the
    /// underlying `f32` wire bytes — negligible for O(seconds ×
    /// sample-rate) chunks but worth knowing if you're streaming
    /// continuous audio in tight loops.
    ///
    /// Errors:
    /// - `EmptyInput` either when `samples` is empty (fast-fail,
    ///   enforced here for parity with `append_text` /
    ///   `append_tokens`) **or** when the audio is too short to
    ///   produce any encoder frames (e.g. shorter than one
    ///   center-padded STFT window).
    /// - `UnsupportedModality` if the loaded model's
    ///   [`ModalityCapabilities::audio_in`] is `false`.
    /// - `Backend(...)` for sample-rate mismatch, encoder/LLM
    ///   `hidden_size` mismatch, or missing encoder. The latter
    ///   includes both "manifest didn't list a mmproj" (no warn
    ///   logged) and "mmproj listed but failed to open/parse"
    ///   (warn logged at `CeraEngine::from_path`).
    /// - `ContextOverflow` / `Cancelled` propagate from the
    ///   underlying prefill.
    pub fn append_audio(&self, samples: Vec<f32>, sample_rate: u32) -> Result<(), FfiError> {
        if samples.is_empty() {
            return Err(FfiError::EmptyInput);
        }
        self.lock_inner()?.append_audio(&samples, sample_rate)?;
        Ok(())
    }

    /// Model hidden dimension `D`. Reshape a raw `[T*D]` byte buffer from
    /// [`Self::hidden_states_for_tokens`] into `[T][D]` with this. Lock-free
    /// (cached at construction), so — like `position()` — it's safe to call from
    /// a `generate_streaming` sink callback.
    pub fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Per-token last-layer hidden states (post-final-RMSNorm — the llama.cpp
    /// `--pooling none` / `llama_get_embeddings_ith` vector) for `tokens`,
    /// returned as **little-endian f32 bytes**: `n_tokens * hidden_size * 4`
    /// bytes, row-major, token `t` channel `c` at `(t*D + c) * 4`.
    ///
    /// Bytes (UniFFI `Data` in Swift, `ByteArray` in Kotlin) rather than
    /// `List<Float>` to avoid Kotlin's per-element boxing on the potentially
    /// large `[T*D]` payload. Swift decodes via `Data.withUnsafeBytes`; reflects
    /// the active LoRA once that lands. Side-effect-free: does not disturb the
    /// session's generation KV.
    ///
    /// Like `append_*` / `generate`, this holds the session mutex for the
    /// duration of the compute, so it must NOT be called re-entrantly from
    /// within a `generate_streaming` sink callback (would self-deadlock).
    ///
    /// Errors: `EmptyInput` on empty input; `UnsupportedModality` if the backend
    /// doesn't implement hidden-state extraction; `InvalidToken` if any id is
    /// `>= vocab_size`.
    pub fn hidden_states_for_tokens(&self, tokens: Vec<u32>) -> Result<Vec<u8>, FfiError> {
        let hs = self.lock_inner()?.hidden_states_for_tokens(&tokens)?;
        Ok(f32_vec_to_le_bytes(&hs))
    }

    /// Like [`Self::hidden_states_for_tokens`] but tokenizes `text` first
    /// (Swift `hiddenStates(for:)`). Returns the same LE-f32 byte layout.
    pub fn hidden_states_for_text(&self, text: String) -> Result<Vec<u8>, FfiError> {
        let hs = self.lock_inner()?.hidden_states_for_text(&text)?;
        Ok(f32_vec_to_le_bytes(&hs))
    }

    /// Mean-pooled hidden state — a single `[hidden_size]` vector (the common
    /// classifier path: pool in Rust, ship `D` floats not `T*D`). Returned as
    /// `[Float]` / `List<Float>`; only `D` elements, so boxing is negligible.
    pub fn hidden_states_mean_pooled(&self, tokens: Vec<u32>) -> Result<Vec<f32>, FfiError> {
        Ok(self.lock_inner()?.hidden_states_mean_pooled(&tokens)?)
    }

    /// Attach a [`LoraAdapters`] to this session (generated as `attachLora` in
    /// Swift/Kotlin — this is the engine's equivalent of a `setLoraAdapters`
    /// call). It's applied to every subsequent forward pass — generation **and**
    /// hidden-states extraction — until removed or replaced (hot-swap), and is
    /// preserved across [`Self::reset`]. Returns [`FfiError::LoraParse`] if the
    /// adapter's dimensions don't match the loaded model. Only affects tokens
    /// processed after the call (doesn't retroactively re-adapt cached KV).
    pub fn attach_lora(&self, adapters: Arc<LoraAdapters>) -> Result<(), FfiError> {
        self.lock_inner()?
            .attach_lora_adapters(adapters.inner.clone())?;
        Ok(())
    }

    /// Remove any attached LoRA adapter, returning to base-model inference.
    pub fn remove_lora(&self) -> Result<(), FfiError> {
        self.lock_inner()?.remove_lora_adapters();
        Ok(())
    }

    /// Whether a LoRA adapter is currently attached to this session.
    pub fn has_lora(&self) -> Result<bool, FfiError> {
        Ok(self.lock_inner()?.has_lora_adapters())
    }

    /// Append an encoded image (PNG / JPEG bytes, auto-detected) to the
    /// context. The image is decoded, resized, normalized, and run
    /// through the bundle's vision mmproj (`VisionEncoderWeights`), then
    /// prefilled into the LLM as soft tokens — see
    /// [`cera::Session::append_image`] for the underlying flow.
    ///
    /// `CeraEngine::new_session` auto-attaches the vision encoder when
    /// the loaded bundle's `inference_type` is a VL type with
    /// `multimodal_projector` set in the manifest, so FFI consumers
    /// don't need a separate "load encoder" call. Bundles whose
    /// manifest omits the mmproj end up with no encoder attached (no
    /// log); bundles where it's named but fails to open/parse log a
    /// `tracing::warn!` at `CeraEngine` construction. Both surface here
    /// as a "no vision encoder attached" `Backend` error.
    ///
    /// `max_long_size` controls the per-call cap on the longest side of
    /// the *encoded* image, with three cases distinguished so the
    /// session default stays reachable through FFI:
    /// - `None` — defer to the session default set via
    ///   [`Self::set_image_max_long_size`] (no cap if none was set).
    /// - `Some(0)` — explicitly force *no cap* for this call, ignoring
    ///   the session default.
    /// - `Some(n)` (`n > 0`) — cap this call at `n`, overriding the
    ///   session default.
    ///
    /// When a cap applies, the resize target is shrunk
    /// (aspect-preserving) so its longer side is at most `n` pixels,
    /// floored at one aligned patch block (so a very small `n` can still
    /// round up to that minimum) — a quality/cost knob (smaller = fewer
    /// image tokens, faster, less detail). It only shrinks (never
    /// upscales) and takes precedence over the model's
    /// minimum-resolution floor. The cap bounds the *encode*, not the
    /// *decode* (a huge source image is still decoded, bounded by
    /// internal limits).
    ///
    /// **Placement matters.** Prefer driving multimodal turns through
    /// the chat template; calling this at the wrong stream position
    /// (outside the model's image-marker envelope) leaves the LLM
    /// unable to interpret the embeddings as visual content. See
    /// [`cera::Session::append_image`] for the marker recipe.
    ///
    /// Errors (capability is checked before emptiness, matching core):
    /// - `UnsupportedModality` if the loaded model's
    ///   [`ModalityCapabilities::image_in`] is `false`.
    /// - `EmptyInput` when `bytes` is empty (on a VL session).
    /// - `Backend(...)` for image decode failure, missing vision
    ///   encoder, or encoder/LLM `projection_dim` ≠ `hidden_size`
    ///   mismatch.
    /// - `ContextOverflow` / `Cancelled` propagate from the
    ///   underlying prefill.
    pub fn append_image(&self, bytes: Vec<u8>, max_long_size: Option<u32>) -> Result<(), FfiError> {
        // Delegate to the core methods (rather than always calling
        // `append_image_with_opts`) so the session default stays
        // reachable through FFI and core's capability-before-empty error
        // precedence is preserved: `None` -> session default, `Some(0)`
        // -> force no cap, `Some(n)` -> cap at `n`. The empty-bytes guard
        // lives in core (`preprocess_image_with_opts`), which runs after
        // the capability check, so a non-VL session still reports
        // `UnsupportedModality` rather than `EmptyInput` for empty input.
        let mut inner = self.lock_inner()?;
        match max_long_size {
            None => inner.append_image(&bytes),
            Some(0) => inner.append_image_with_opts(&bytes, None),
            Some(n) => inner.append_image_with_opts(&bytes, Some(n)),
        }?;
        Ok(())
    }

    /// Set a session-default cap on the longest side of an appended
    /// image, in pixels (`None` = no cap). Unlike the per-call
    /// `max_long_size` argument to [`Self::append_image`], this default
    /// is honored by every image-append path the session drives —
    /// including chat-template flows — so a host can configure the
    /// image-encode budget once. See [`Self::append_image`] for the cap
    /// semantics (shrinks the encoded target, never upscales, takes
    /// precedence over the model's minimum-resolution floor).
    pub fn set_image_max_long_size(&self, max_long_size: Option<u32>) -> Result<(), FfiError> {
        self.lock_inner()?.set_image_max_long_size(max_long_size);
        Ok(())
    }

    /// Run autoregressive decode and return all emitted tokens +
    /// a summary. Synchronous — the call blocks until the decode
    /// loop exits (`max_tokens`, EOS, `cancel()`, or error).
    ///
    /// For streaming (per-chunk delivery) and async, see the PR 4 /
    /// PR 5 follow-ups in `cera-ffi/README.md`.
    pub fn generate(&self, opts: GenerateOpts) -> Result<GenerateOutput, FfiError> {
        // Collector sink: captures every token the decode loop emits.
        // `on_done` is invoked once at the end regardless of exit
        // reason; we read the Result from `session.generate` to see
        // whether the run succeeded.
        struct CollectSink(Vec<u32>);
        impl cera::ModalitySink for CollectSink {
            fn on_text_tokens(&mut self, tokens: &[u32]) {
                self.0.extend_from_slice(tokens);
            }
            fn on_done(&mut self, _reason: cera::FinishReason) {}
        }
        let mut sink = CollectSink(Vec::new());
        // Compile the grammar (if any) before taking the session lock so a
        // malformed GBNF fails fast with `FfiError::GrammarParse`.
        let core: cera::GenerateOpts = opts.try_into()?;
        let summary = self.lock_inner()?.generate(&core, &mut sink)?;
        Ok(GenerateOutput {
            tokens: sink.0,
            summary: summary.into(),
        })
    }

    /// Run autoregressive decode, streaming every token (and audio
    /// frame, for audio-capable models) to a foreign [`ModalitySink`]
    /// as soon as it's produced. Returns only a [`GenerateSummary`] —
    /// token IDs are delivered through `sink.on_text_tokens`, not a
    /// return value.
    ///
    /// Synchronous: the call blocks on the decode thread and each
    /// `sink` method runs on that same thread before decoding
    /// continues. For async, see PR 5 in `cera-ffi/README.md`.
    ///
    /// **Callback reentrancy — deadlock hazard.** The session mutex is
    /// held for the entire call, and sink callbacks run while that
    /// lock is held. Calling back into methods that also take the
    /// mutex ([`Session::append_text`], [`Session::append_tokens`],
    /// [`Session::generate`], [`Session::generate_streaming`],
    /// [`Session::reset`]) from inside a sink method will deadlock.
    /// [`Session::cancel`] and [`Session::position`] are atomic-backed
    /// and safe to call from the sink or from any other thread.
    ///
    /// Cancellation: call [`Session::cancel`] from any thread (or from
    /// inside a sink callback on this thread) to terminate the loop at
    /// the next between-token check; `sink.on_done` fires with
    /// [`FinishReason::Cancelled`].
    ///
    /// End-of-stream guarantee: `sink.on_done` fires exactly once per
    /// call, even on error paths. If the underlying decode returns an
    /// error before reaching its own `on_done` call (e.g.,
    /// `EmptyInput` with no prefill logits), the wrapper synthesizes
    /// a terminal `on_done(FinishReason::Error { message })` so
    /// foreign consumers have a reliable end-of-stream signal
    /// regardless of how the call exits.
    pub fn generate_streaming(
        &self,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let mut adapter = ForeignSinkAdapter {
            inner: sink,
            done_called: false,
        };
        // Scope the lock so the synthesized on_done on the error path
        // doesn't run with the session mutex held — foreign sinks
        // already have to avoid session-reentrancy during success
        // callbacks; reusing that contract on the error path keeps
        // the hazard set minimal.
        // Compile the grammar first; a malformed GBNF (`FfiError::GrammarParse`)
        // routes through the same best-effort `on_done(Error)` path below as any
        // other pre-decode failure.
        let outcome = match cera::GenerateOpts::try_from(opts) {
            Ok(core) => match self.lock_inner() {
                Ok(mut guard) => guard.generate(&core, &mut adapter).map_err(FfiError::from),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match outcome {
            Ok(summary) => Ok(summary.into()),
            Err(err) => {
                if !adapter.done_called {
                    // `FinishReason::Error` only carries a message string,
                    // so flatten whichever typed FfiError variant via
                    // Display. Foreign callers still receive the full
                    // typed error from the return value; the sink's
                    // on_done(Error) is a best-effort terminal signal.
                    adapter.inner.on_done(FinishReason::Error {
                        message: err.to_string(),
                    });
                }
                Err(err)
            }
        }
    }

    /// Current KV position — how many tokens live in the cache.
    /// Atomic-backed; safe to call from a different thread while
    /// `generate()` is in flight.
    pub fn position(&self) -> u32 {
        self.position.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Signal in-flight `generate()` to exit with
    /// `FinishReason::Cancelled` at the next between-token check.
    /// Safe from any thread. No-op if no `generate()` is running.
    pub fn cancel(&self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Clear the cancel flag without dropping any session state.
    /// Use this after observing a cancellation signal — either
    /// [`FfiError::Cancelled`] from `append_text` / `append_tokens`
    /// / `append_audio` (mid-prefill cancellation surfaces
    /// typed), or `finish_reason = "Cancelled"` on the
    /// [`GenerateOutput`] returned from `generate` (cancellation
    /// during decode is reported as an `Ok` with that finish
    /// reason rather than an `Err`) — when you want to resume
    /// work on the same session without losing the accumulated
    /// KV cache.
    ///
    /// Compared to [`Self::reset`]:
    /// - `clear_cancel`: keeps KV state + position + sampler
    ///   intact; only flips the cancel atomic back to `false`.
    ///   Use for "interrupted but continuing" flows.
    /// - `reset`: drops KV cache + position + last logits +
    ///   re-seeds sampler. Use for "clear conversation" flows.
    ///
    /// Atomic-backed; no mutex acquire, infallible, safe from
    /// any thread (mirrors the shape of [`Self::cancel`]).
    pub fn clear_cancel(&self) {
        self.cancel
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drop cached state + resample the seed. After `reset()` the
    /// session behaves like a freshly-opened one (same
    /// model/tokenizer/config, no accumulated context).
    ///
    /// Returns `Result` so a poisoned-mutex case surfaces as an error
    /// instead of panicking across the FFI boundary.
    pub fn reset(&self) -> Result<(), FfiError> {
        self.lock_inner()?.reset()?;
        Ok(())
    }

    /// Capabilities reported by the loaded model. Cheap — reads a
    /// cached copy, no lock.
    pub fn capabilities(&self) -> ModalityCapabilities {
        self.capabilities
    }
}

// ---------------------------------------------------------------------------
// Async Session methods (PR 5)
// ---------------------------------------------------------------------------
//
// Foreign callers driving an async runtime (Kotlin coroutines, Swift
// `async`, Python `asyncio`) can `.await` these without bouncing into
// a sync context. Every method defers to its synchronous twin inside
// `tokio::task::spawn_blocking`, which moves the actual decode work
// onto a blocking worker thread — so the tokio async worker pool
// stays free to poll other futures while decoding is in flight.
//
// `self: Arc<Self>` rather than `&self` so the session handle can
// cross the `spawn_blocking` boundary (requires `'static`). UniFFI
// wraps `#[uniffi::Object]` types in `Arc` on the foreign side anyway,
// so this doesn't change the foreign API shape — it's still
// `session.generateAsync(opts)` on Kotlin / `session.generateAsync(opts)`
// on Swift / `session.generate_async(opts)` on Python.
//
// UniFFI's `tokio` feature starts an internal multi-thread tokio
// runtime the first time a `#[uniffi::export(async_runtime = "tokio")]`
// method is invoked. Spawned blocking tasks inherit that runtime's
// blocking worker pool (`tokio::runtime::Builder::new_multi_thread`
// default). We don't need to create or enter a runtime ourselves.

/// RAII guard that cancels the in-flight `spawn_blocking` decode on
/// future-drop. Addresses a subtle hazard of wrapping sync decode in
/// `tokio::task::spawn_blocking`: dropping the outer future drops the
/// `JoinHandle`, but tokio does **not** abort a `spawn_blocking` task
/// on handle-drop — the blocking worker keeps decoding, keeps holding
/// the session mutex, keeps mutating `Session::state`.
///
/// Without this guard, a foreign-side cancellation (Kotlin coroutine
/// scope exit, Swift `Task.cancel`, Python `asyncio.Task.cancel`) would
/// silently leak decode work into the background. The caller's next
/// `generate*` call would block on the still-held mutex or observe
/// state advanced by the "cancelled" call.
///
/// Two code paths, two mitigations — both fire together because the
/// guard can't know which path applies:
///
/// 1. **Running decode.** The task is executing `cera::Session::generate`
///    on a blocking worker. `session.cancel()` flips the cancel atomic;
///    the decode loop polls it between tokens and exits with
///    `FinishReason::Cancelled`. `JoinHandle::abort` has no effect
///    here — `spawn_blocking` tasks are opaque synchronous code with
///    no await points to interrupt.
///
/// 2. **Queued decode.** The task is in the blocking pool's queue
///    waiting for a worker (pool saturated, or just hasn't been
///    scheduled). `JoinHandle::abort` cancels queued-but-not-started
///    blocking tasks before their closure runs — the closure never
///    executes, so `cera::Session::generate` never starts, so the
///    session's cancel flag is never reset. Without this, the race is:
///    guard sets cancel → task eventually dequeues → decode's first
///    line clears cancel back to `false` (`cera/src/session.rs:603-605`)
///    → decode runs to completion despite the caller having dropped
///    the future.
///
/// Both operations are idempotent / harmless on the irrelevant path:
/// `session.cancel()` on a queued task is overridden by `abort`; an
/// already-completed task ignores both. `cera::Session::generate`
/// resets the cancel atomic on entry, so a spurious late-arriving
/// cancel from a guard that dropped just after the await resolved
/// (not reachable in practice — futures aren't preemptively dropped
/// between synchronous statements) wouldn't affect the next call.
struct AsyncCancelGuard {
    session: Arc<Session>,
    /// Abort handle for the `spawn_blocking` task. Calling `abort()`
    /// on a queued task removes it from the pool's queue; on a running
    /// task it's a no-op (no await point to unwind through). Kept as
    /// an `AbortHandle` rather than a `JoinHandle` so the guard can
    /// coexist with the outer `.await` on the same handle (we take
    /// `abort_handle()` before awaiting).
    abort: tokio::task::AbortHandle,
    /// `true` until the await successfully resolves. Dropping with
    /// `armed = true` means we're being dropped mid-await: fire both
    /// abort (for queued-but-not-started) and cancel (for in-flight).
    armed: bool,
}

impl Drop for AsyncCancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.abort.abort();
            self.session.cancel();
        }
    }
}

/// Lighter-weight sibling of [`AsyncCancelGuard`] for `spawn_blocking`
/// tasks that don't share mutable state with anything the caller can
/// signal. Used by [`CeraEngine::from_bundle_id_async`]: the underlying
/// `cera::CeraEngine::from_bundle_id` holds no cross-thread cancel flag
/// and the `reqwest::blocking` download can't be cooperatively
/// cancelled, so there's nothing like `Session::cancel` to call on
/// drop. All we can do is abort the queued task before its closure
/// runs — `AbortHandle::abort` (taken from the task's `JoinHandle`
/// via `JoinHandle::abort_handle()` so the guard doesn't fight the
/// outer `.await` for ownership of the handle) on a queued
/// `spawn_blocking` task is effective; on a running one it's a no-op
/// and the download finishes to cache (which is arguably a feature:
/// a dropped future's bandwidth isn't wasted, the next call finds
/// the bundle cached and returns instantly).
///
/// Drop logic is one conditional (`if armed { abort.abort() }`) —
/// structurally identical to [`AsyncCancelGuard`]'s. The
/// `async_cancel_guard_drop_fires_when_armed_only` ProbeGuard test in
/// the test module already exercises that exact branch shape; a
/// duplicate test for AbortOnDrop would add no coverage. End-to-end
/// "abort actually cancels a queued tokio task" is upstream tokio's
/// behavior to test, not ours.
struct AbortOnDrop {
    abort: tokio::task::AbortHandle,
    /// Set to `false` once the outer `.await` resolves; prevents
    /// `abort()` from running on a task that already completed.
    armed: bool,
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.abort.abort();
        }
    }
}

#[uniffi::export(async_runtime = "tokio")]
impl Session {
    /// Async variant of [`Session::generate`] — runs buffered decode
    /// (returning every emitted token + a summary) on a tokio blocking
    /// worker so the caller's async context isn't stalled by the
    /// synchronous decode loop.
    ///
    /// Cancellation: dropping the returned future (Kotlin coroutine
    /// scope exit, Swift `Task.cancel`, Python `asyncio.Task.cancel`)
    /// triggers both an abort of the queued `spawn_blocking` task (so
    /// a not-yet-started decode never runs) and a
    /// [`Session::cancel`] call (so an in-flight decode exits at its
    /// next between-token check with [`FinishReason::Cancelled`]).
    /// Either path releases the session mutex; subsequent calls see
    /// a clean session. You can also call [`Session::cancel`]
    /// directly from any thread to trigger the same in-flight exit
    /// without dropping the future. See `AsyncCancelGuard` for the
    /// full rationale.
    ///
    /// On error the wrapper performs the same poisoned-mutex handling
    /// as sync [`Session::generate`]. `JoinError` from a panic in the
    /// blocking closure surfaces as [`FfiError::Backend`] with a
    /// diagnostic prefix.
    pub async fn generate_async(
        self: Arc<Self>,
        opts: GenerateOpts,
    ) -> Result<GenerateOutput, FfiError> {
        let session_for_guard = Arc::clone(&self);
        let handle = tokio::task::spawn_blocking(move || self.generate(opts));
        let mut guard = AsyncCancelGuard {
            session: session_for_guard,
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result.map_err(|e| FfiError::Backend {
            detail: format!("generate_async join error: {e}"),
        })?
    }

    /// Async variant of [`Session::generate_streaming`] — delivers
    /// tokens and audio frames to the foreign [`ModalitySink`] as the
    /// decode loop produces them, from within a blocking worker so
    /// the caller's async runtime stays responsive.
    ///
    /// Sink callbacks run on the blocking worker thread that's
    /// executing the decode — **not** on the caller's async thread.
    /// The reentrancy hazard documented on
    /// [`Session::generate_streaming`] still applies: sink callbacks
    /// that call back into `append_text` / `generate*` / `reset` from
    /// inside the session will deadlock on the session mutex.
    /// [`Session::cancel`] and [`Session::position`] remain atomic-
    /// backed and safe to invoke from any thread (including from
    /// inside a callback).
    ///
    /// Cancellation: dropping the returned future fires the same
    /// abort + [`Session::cancel`] pair as [`Session::generate_async`]
    /// (see `AsyncCancelGuard`). For an in-flight decode, the loop
    /// exits with [`FinishReason::Cancelled`] and the sink's `on_done`
    /// fires on the blocking worker before the task completes —
    /// foreign consumers get the terminal signal even though they've
    /// already stopped awaiting. For a queued-but-not-started decode,
    /// abort cancels the task without ever running the closure; no
    /// sink callbacks fire for that case (the decode never began).
    pub async fn generate_streaming_async(
        self: Arc<Self>,
        opts: GenerateOpts,
        sink: Arc<dyn ModalitySink>,
    ) -> Result<GenerateSummary, FfiError> {
        let session_for_guard = Arc::clone(&self);
        let handle = tokio::task::spawn_blocking(move || self.generate_streaming(opts, sink));
        let mut guard = AsyncCancelGuard {
            session: session_for_guard,
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result.map_err(|e| FfiError::Backend {
            detail: format!("generate_streaming_async join error: {e}"),
        })?
    }
}

// Async CeraEngine constructors (PR 11).
#[uniffi::export(async_runtime = "tokio")]
impl CeraEngine {
    /// Async variant of [`CeraEngine::from_bundle_id`] — offloads the
    /// manifest + GGUF download and the engine construction onto a
    /// tokio blocking worker so the caller's async context isn't
    /// stalled. Foreign async runtimes (Kotlin coroutines, Swift
    /// `async`, Python `asyncio`) `.await` it directly.
    ///
    /// `config.bundle_repo` must be set (same constraint as the sync
    /// twin); construct a [`BundleRepo`] rooted at a persistent cache
    /// directory and attach it to the config before calling.
    ///
    /// Cancellation semantics (weaker than [`Session::generate_async`]):
    /// dropping the returned future drops the `AbortOnDrop` guard,
    /// which calls `AbortHandle::abort` on the spawned task. That
    /// cancels the task if it's still queued on tokio's blocking
    /// pool, so a not-yet-started download never runs. But if the
    /// task has started, abort is a no-op — the download is a
    /// `reqwest::blocking` call with no cooperative cancel point,
    /// and cera's engine-construction code (tokenizer build, model
    /// load, KV alloc) also isn't interruptible. In that case the
    /// task runs to completion and the engine is constructed then
    /// dropped; the downloaded bundle stays cached, so the caller's
    /// next attempt starts from that cache hit. Bandwidth isn't
    /// wasted, it's just shifted.
    ///
    /// `JoinError` from a panicking blocking closure surfaces as
    /// [`FfiError::Backend`] with a diagnostic prefix, same as
    /// [`Session::generate_async`].
    #[uniffi::constructor]
    pub async fn from_bundle_id_async(
        bundle_id: String,
        quant: String,
        config: EngineConfig,
    ) -> Result<Arc<Self>, FfiError> {
        // Convert the config synchronously. Any `TryFrom` error
        // (e.g. 32-bit `u64 → usize` overflow on the context size)
        // fails fast without spawning a blocking task.
        let cera_config: cera::EngineConfig = config.try_into()?;
        let handle = tokio::task::spawn_blocking(move || {
            cera::CeraEngine::from_bundle_id(&bundle_id, &quant, cera_config)
                .map_err(FfiError::from)
        });
        let mut guard = AbortOnDrop {
            abort: handle.abort_handle(),
            armed: true,
        };
        let join_result = handle.await;
        guard.armed = false;
        join_result
            .map_err(|e| FfiError::Backend {
                detail: format!("from_bundle_id_async join error: {e}"),
            })?
            .map(|inner| Arc::new(Self { inner }))
    }
}

// Session-level method on CeraEngine.
#[uniffi::export]
impl CeraEngine {
    /// Open a new [`Session`] sharing this engine's model + tokenizer
    /// by `Arc` clone. The returned session outlives `&self`; the
    /// engine keeps the shared state live for every session it hands
    /// out. Cheap — no model load, just config + state allocation.
    pub fn new_session(&self, config: SessionConfig) -> Result<Arc<Session>, FfiError> {
        let session = self.inner.new_session(config.into())?;
        let position = session.position_handle();
        let cancel = session.cancel_handle();
        let capabilities = session.capabilities().into();
        let hidden_size = u32::try_from(session.hidden_size()).unwrap_or(u32::MAX);
        Ok(Arc::new(Session {
            inner: std::sync::Mutex::new(session),
            position,
            cancel,
            capabilities,
            hidden_size,
        }))
    }
}

// ---------------------------------------------------------------------------
// Smoke test (from PR 1)
// ---------------------------------------------------------------------------

/// Version string of the `cera-ffi` crate. Useful as a smoke test
/// from the foreign-language side — if this is callable, the binding
/// pipeline works end-to-end.
#[uniffi::export]
pub fn cera_ffi_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// One-line CPU backend report for this host — the resolved SIMD tier plus the
/// detected feature flags, e.g. `cpu: tier=neon+dotprod [neon dotprod]`. A host
/// property independent of any loaded model; callable without an engine. Handy
/// for telemetry and bug reports (tells you which kernel path actually ran).
#[uniffi::export]
pub fn cpu_backend_report() -> String {
    cera::cpu_features().report()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        // Smoke test: proves the proc-macro expanded and the export is
        // callable. No shape check on the string — SemVer allows
        // pre-release + build-metadata suffixes (`0.1.0-alpha.1+deadbe`)
        // that a strict `x.y.z` split would reject.
        let v = cera_ffi_version();
        assert!(!v.is_empty(), "version string must not be empty");
    }

    #[test]
    fn engine_config_default_roundtrips_to_cera() {
        let ffi = EngineConfig::default();
        let core: cera::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, 4096);
        assert_eq!(core.backend, cera::BackendPreference::Auto);
        // bundle_repo defaults to None — foreign callers opt in by
        // attaching a BundleRepo before the try_into.
        assert!(core.bundle_repo.is_none());
    }

    /// `ChatMessage` round-trips its `role` + `content` fields
    /// through the cera-core conversion. `From<ChatMessage> for
    /// cera::tokenizer::ChatMessage` is a trivial field-copy; this
    /// test pins the field shape so a future cera-core rename
    /// breaks compilation here loudly instead of silently dropping
    /// data on the FFI boundary.
    #[test]
    fn chat_message_converts_to_cera_core() {
        let m = ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        };
        let core: cera::tokenizer::ChatMessage = m.clone().into();
        assert_eq!(core.role, "user");
        assert_eq!(core.content, "hello");
        // Original FFI value is unchanged (we cloned for the
        // conversion); proves the From doesn't mutate by ref.
        assert_eq!(m.role, "user");
    }

    /// Pick a temp path scoped to this process + test name so parallel
    /// test binaries and prior runs don't collide. `std::env::temp_dir()`
    /// honors `TMPDIR` on macOS and `/tmp` elsewhere; the process-id
    /// suffix is stable across the test's lifetime but unique per run.
    /// `remove_dir_all` on entry makes the existence assertion below
    /// deterministic even if a previous run's panic left the dir behind.
    fn unique_test_bundle_dir(test_name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cera-ffi-test-{}-{}",
            test_name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// `cache_size` round-trips through the FFI wrapper. Builds a
    /// small synthetic cache, verifies the FFI method returns the
    /// same byte count as the cera-core method (which has its own
    /// unit-test coverage in `cera/src/bundle/mod.rs`).
    #[test]
    fn bundle_repo_cache_size_forwards_to_cera_core() {
        use std::fs;
        let dir = unique_test_bundle_dir("size");
        fs::create_dir_all(dir.join("huggingface.co/test")).unwrap();
        fs::write(dir.join("huggingface.co/test/file"), vec![0u8; 2048]).unwrap();
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        assert_eq!(repo.cache_size().unwrap(), 2048);
        let _ = fs::remove_dir_all(&dir);
    }

    /// `clear_cache` round-trips through the FFI wrapper. After the
    /// clear, `cache_size` reports 0 and `store_dir` still exists
    /// (subsequent downloads can land there).
    #[test]
    fn bundle_repo_clear_cache_wipes_files_via_ffi() {
        use std::fs;
        let dir = unique_test_bundle_dir("clear");
        fs::create_dir_all(dir.join("huggingface.co/test")).unwrap();
        fs::write(dir.join("huggingface.co/test/file"), vec![0u8; 512]).unwrap();
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        assert_eq!(repo.cache_size().unwrap(), 512);
        repo.clear_cache().unwrap();
        assert!(dir.exists(), "store_dir must survive clear_cache");
        assert_eq!(repo.cache_size().unwrap(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// `BundleRepo::new` wraps a `cera::bundle::BundleRepo` without
    /// creating the store_dir on disk (that happens lazily on first
    /// download). Store_dir round-trips.
    #[test]
    fn bundle_repo_constructs_and_store_dir_roundtrips() {
        let dir = unique_test_bundle_dir("construct");
        let dir_str = dir.to_string_lossy().into_owned();
        let repo = BundleRepo::new(dir_str.clone());
        assert_eq!(repo.store_dir(), dir_str);
        // Directory creation is lazy; the path must not exist yet.
        assert!(
            !dir.exists(),
            "BundleRepo::new eagerly created {}",
            dir.display()
        );
    }

    /// Attaching a `BundleRepo` to `EngineConfig` plumbs through to
    /// `cera::EngineConfig::bundle_repo` — proves the From/TryFrom
    /// conversion handles the Arc<BundleRepo> → Option<cera::BundleRepo>
    /// path correctly.
    #[test]
    fn engine_config_carries_bundle_repo_through_try_from() {
        let dir = unique_test_bundle_dir("carry");
        let repo = BundleRepo::new(dir.to_string_lossy().into_owned());
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
            bundle_repo: Some(repo.clone()),
        };
        let core: cera::EngineConfig = ffi.try_into().unwrap();
        let core_repo = core.bundle_repo.expect("bundle_repo must be Some");
        assert_eq!(core_repo.store_dir(), repo.inner.store_dir());
    }

    /// `DownloadProgressAdapter` forwards `on_progress` calls from
    /// cera-core to a foreign-trait implementation (impl'd here as a
    /// recording Rust struct that mirrors how UniFFI codegens the
    /// foreign side). Verifies the URL / bytes / total round-trip
    /// without dropping data.
    #[test]
    fn download_progress_adapter_forwards() {
        use cera::bundle::DownloadProgress as _;
        use std::sync::Mutex;

        #[derive(Debug, Default)]
        struct Recorder {
            calls: Mutex<Vec<(String, u64, Option<u64>)>>,
        }
        impl DownloadProgressSink for Recorder {
            fn on_progress(&self, url: String, bytes: u64, total: Option<u64>) {
                self.calls.lock().unwrap().push((url, bytes, total));
            }
        }

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let adapter = DownloadProgressAdapter {
            inner: recorder.clone() as Arc<dyn DownloadProgressSink>,
        };

        // Drive the adapter as cera-core's download_to would.
        adapter.on_progress("https://example.com/a.gguf", 1024, Some(2048));
        adapter.on_progress("https://example.com/a.gguf", 2048, Some(2048));
        adapter.on_progress("https://example.com/no-length", 512, None);

        let calls = recorder.calls.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, "https://example.com/a.gguf");
        assert_eq!(calls[0].1, 1024);
        assert_eq!(calls[0].2, Some(2048));
        assert_eq!(calls[2].0, "https://example.com/no-length");
        assert_eq!(calls[2].1, 512);
        assert_eq!(calls[2].2, None);
    }

    #[test]
    fn engine_config_zero_context_size_means_max() {
        // `0` on the wire is the FFI's "use model default" signal;
        // translate to `usize::MAX` so cera caps at model.max_seq_len.
        let ffi = EngineConfig {
            context_size: 0,
            backend: BackendPreference::Cpu,
            bundle_repo: None,
        };
        let core: cera::EngineConfig = ffi.try_into().unwrap();
        assert_eq!(core.context_size, usize::MAX);
    }

    #[test]
    fn engine_config_oversize_context_errors_on_32bit_targets() {
        // On 32-bit targets, `u64::MAX` exceeds `usize::MAX` and the
        // checked conversion must surface an error rather than
        // silently truncating. On 64-bit targets `usize::MAX ==
        // u64::MAX`, so the conversion succeeds — skip the assert
        // there. This test proves the error path compiles + is
        // reachable under the narrow condition where it matters.
        let ffi = EngineConfig {
            context_size: u64::MAX,
            backend: BackendPreference::Cpu,
            bundle_repo: None,
        };
        let result: Result<cera::EngineConfig, FfiError> = ffi.try_into();
        #[cfg(target_pointer_width = "32")]
        {
            let err = result.expect_err("u64::MAX must fail on 32-bit");
            match err {
                FfiError::Backend { detail } => {
                    assert!(
                        detail.contains("exceeds usize::MAX"),
                        "unexpected: {detail}"
                    );
                }
                other => panic!("expected Backend, got: {other:?}"),
            }
        }
        #[cfg(target_pointer_width = "64")]
        {
            // On 64-bit `u64::MAX == usize::MAX`; the sentinel check
            // has already rejected `0`, so `u64::MAX` converts cleanly.
            let core = result.expect("u64::MAX fits usize::MAX on 64-bit");
            assert_eq!(core.context_size, usize::MAX);
        }
    }

    #[test]
    fn backend_preference_roundtrips() {
        for ffi in [
            BackendPreference::Auto,
            BackendPreference::Cpu,
            BackendPreference::Gpu,
            BackendPreference::Metal,
        ] {
            let core: cera::BackendPreference = ffi.into();
            let back: BackendPreference = core.into();
            assert_eq!(ffi, back, "{ffi:?} didn't round-trip");
        }
    }

    /// Every `cera::CeraError` variant maps to a specific `FfiError`
    /// variant (not the generic `Backend` catch-all) so foreign
    /// callers can pattern-match on class. If cera adds a new
    /// `CeraError` variant and forgets to update `From<CeraError>`,
    /// the exhaustive match in that impl breaks compilation loudly —
    /// this test just asserts the existing mapping is correct.
    #[test]
    fn cera_error_variants_map_to_typed_ffi_error_variants() {
        // Payload-free variants.
        assert!(matches!(
            FfiError::from(cera::CeraError::UnsupportedModality),
            FfiError::UnsupportedModality
        ));
        assert!(matches!(
            FfiError::from(cera::CeraError::Busy),
            FfiError::Busy
        ));
        assert!(matches!(
            FfiError::from(cera::CeraError::Cancelled),
            FfiError::Cancelled
        ));
        assert!(matches!(
            FfiError::from(cera::CeraError::EmptyInput),
            FfiError::EmptyInput
        ));

        // Payload-carrying variants preserve their fields.
        match FfiError::from(cera::CeraError::UnsupportedInferenceType(
            "audio-magic".into(),
        )) {
            FfiError::UnsupportedInferenceType { inference_type } => {
                assert_eq!(inference_type, "audio-magic");
            }
            other => panic!("expected UnsupportedInferenceType, got: {other:?}"),
        }

        match FfiError::from(cera::CeraError::ContextOverflow {
            max_seq_len: 4096,
            by: 17,
        }) {
            FfiError::ContextOverflow { max_seq_len, by } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(by, 17);
            }
            other => panic!("expected ContextOverflow, got: {other:?}"),
        }

        match FfiError::from(cera::CeraError::Backend("metal driver crashed".into())) {
            FfiError::Backend { detail } => {
                assert_eq!(detail, "metal driver crashed");
            }
            other => panic!("expected Backend, got: {other:?}"),
        }

        // Io flattens the OS error to a string.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let io_str = io_err.to_string();
        match FfiError::from(cera::CeraError::Io(io_err)) {
            FfiError::Io { detail } => {
                assert_eq!(detail, io_str);
            }
            other => panic!("expected Io, got: {other:?}"),
        }
    }

    /// Display (`thiserror`-derived) produces byte-identical message
    /// text to the equivalent `cera::CeraError` for every variant
    /// that has a cera analog. Foreign callers logging the error via
    /// `.toString()` / `String(describing:)` / `str()` see the same
    /// output whether the error originates from cera directly or
    /// routes through the FFI wrapper.
    ///
    /// Tests every shared variant including `Backend`, `Io`, and
    /// `UnsupportedInferenceType` — an earlier iteration of this test
    /// quietly excluded them, which masked a real drift where the FFI
    /// side had dropped the `"backend: "` and `"io: "` label prefixes.
    #[test]
    fn ffi_error_display_matches_cera_error_for_every_shared_variant() {
        // Prep the Io pair outside the vec since io::Error isn't
        // `Clone`: we need to consume one into `CeraError::Io` and
        // stash its pre-wrap display string for the FFI side.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let io_msg = io_err.to_string();

        let pairs: Vec<(FfiError, cera::CeraError)> = vec![
            (
                FfiError::UnsupportedModality,
                cera::CeraError::UnsupportedModality,
            ),
            (
                FfiError::UnsupportedInferenceType {
                    inference_type: "audio-magic".into(),
                },
                cera::CeraError::UnsupportedInferenceType("audio-magic".into()),
            ),
            (FfiError::Busy, cera::CeraError::Busy),
            (FfiError::Cancelled, cera::CeraError::Cancelled),
            (
                FfiError::ContextOverflow {
                    max_seq_len: 2048,
                    by: 5,
                },
                cera::CeraError::ContextOverflow {
                    max_seq_len: 2048,
                    by: 5,
                },
            ),
            (FfiError::EmptyInput, cera::CeraError::EmptyInput),
            (
                FfiError::Backend {
                    detail: "metal driver crashed".into(),
                },
                cera::CeraError::Backend("metal driver crashed".into()),
            ),
            (FfiError::Io { detail: io_msg }, cera::CeraError::Io(io_err)),
        ];
        for (ffi, core) in pairs {
            assert_eq!(
                ffi.to_string(),
                core.to_string(),
                "display mismatch for {ffi:?} vs {core:?}"
            );
        }
    }

    #[test]
    fn session_config_default_roundtrips_to_cera() {
        let ffi = SessionConfig::default();
        let core: cera::SessionConfig = ffi.into();
        let default_core = cera::SessionConfig::default();
        assert_eq!(core.max_seq_len, default_core.max_seq_len);
        assert_eq!(core.n_keep, default_core.n_keep);
        assert_eq!(core.seed, default_core.seed);
        assert_eq!(core.ubatch_size, default_core.ubatch_size);
    }

    #[test]
    fn kv_compression_none_roundtrips() {
        let ffi = KvCompression::None;
        let core: cera::kv_cache::KvCompression = ffi.into();
        assert!(matches!(core, cera::kv_cache::KvCompression::None));
    }

    #[test]
    fn kv_compression_turboquant_roundtrips() {
        let ffi = KvCompression::TurboQuant {
            seed: 42,
            keys: true,
            values: false,
        };
        let core: cera::kv_cache::KvCompression = ffi.into();
        match core {
            cera::kv_cache::KvCompression::TurboQuant { seed, keys, values } => {
                assert_eq!(seed, 42);
                assert!(keys);
                assert!(!values);
            }
            _ => panic!("expected TurboQuant variant"),
        }
    }

    #[test]
    fn generate_opts_default_roundtrips_to_cera() {
        let ffi = GenerateOpts::default();
        let core: cera::GenerateOpts = ffi.try_into().expect("default opts have no grammar");
        let default_core = cera::GenerateOpts::default();
        // Field-by-field so a future cera field-add breaks here loudly.
        assert_eq!(core.max_tokens, default_core.max_tokens);
        assert_eq!(core.temperature, default_core.temperature);
        assert_eq!(core.top_p, default_core.top_p);
        assert_eq!(core.top_k, default_core.top_k);
        assert_eq!(core.repetition_penalty, default_core.repetition_penalty);
        assert_eq!(core.stop_tokens, default_core.stop_tokens);
        assert!(core.grammar.is_none());
        assert_eq!(core.flush_every_tokens, default_core.flush_every_tokens);
        assert_eq!(core.flush_every_ms, default_core.flush_every_ms);
    }

    #[test]
    fn generate_opts_compiles_valid_grammar() {
        let ffi = GenerateOpts {
            grammar: Some(r#"root ::= "yes" | "no""#.to_string()),
            ..GenerateOpts::default()
        };
        let core: cera::GenerateOpts = ffi.try_into().expect("valid GBNF compiles");
        assert!(core.grammar.is_some());
    }

    #[test]
    fn generate_opts_rejects_malformed_grammar() {
        let ffi = GenerateOpts {
            // Unterminated string literal — the GBNF parser rejects it.
            grammar: Some(r#"root ::= "oops"#.to_string()),
            ..GenerateOpts::default()
        };
        let err = cera::GenerateOpts::try_from(ffi).unwrap_err();
        assert!(matches!(err, FfiError::GrammarParse { .. }), "got: {err:?}");
    }

    #[test]
    fn finish_reason_covers_every_variant() {
        use cera::FinishReason as Core;
        let cases = [
            (Core::MaxTokens, "MaxTokens"),
            (Core::Stop, "Stop"),
            (Core::Cancelled, "Cancelled"),
            (Core::ContextFull, "ContextFull"),
            (Core::GrammarDeadEnd, "GrammarDeadEnd"),
            (Core::Error("boom".into()), "Error"),
        ];
        for (core, tag) in cases {
            let ffi: FinishReason = core.into();
            match (&ffi, tag) {
                (FinishReason::MaxTokens, "MaxTokens") => {}
                (FinishReason::Stop, "Stop") => {}
                (FinishReason::Cancelled, "Cancelled") => {}
                (FinishReason::ContextFull, "ContextFull") => {}
                (FinishReason::GrammarDeadEnd, "GrammarDeadEnd") => {}
                (FinishReason::Error { message }, "Error") => assert_eq!(message, "boom"),
                _ => panic!("variant mismatch: {ffi:?} tagged {tag}"),
            }
        }
    }

    /// Exercises the ForeignSinkAdapter by implementing the FFI
    /// `ModalitySink` trait from Rust (what UniFFI codegens the foreign
    /// binding to look like on the Rust side) and driving it through
    /// the internal `cera::ModalitySink` impl. Confirms:
    /// - `on_text_tokens` forwards with the exact bytes.
    /// - `on_audio_frames` forwards with the exact bytes + rate.
    /// - `on_done` forwards and maps the FinishReason through `.into()`.
    /// - All three run without the foreign side needing `&mut self` —
    ///   interior mutability (Mutex/atomic) is the caller's burden,
    ///   mirroring what Kotlin/Swift will see.
    #[test]
    fn foreign_sink_adapter_forwards_every_method() {
        use cera::ModalitySink as CoreSink;
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            text: Mutex<Vec<u32>>,
            audio: Mutex<Vec<(Vec<f32>, u32)>>,
            done: Mutex<Option<FinishReason>>,
        }

        impl ModalitySink for Recorder {
            fn on_text_tokens(&self, tokens: Vec<u32>) {
                self.text.lock().unwrap().extend(tokens);
            }
            fn on_audio_frames(&self, pcm: Vec<f32>, sample_rate: u32) {
                self.audio.lock().unwrap().push((pcm, sample_rate));
            }
            fn on_done(&self, reason: FinishReason) {
                *self.done.lock().unwrap() = Some(reason);
            }
        }

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let mut adapter = ForeignSinkAdapter {
            inner: recorder.clone() as Arc<dyn ModalitySink>,
            done_called: false,
        };

        // Drive the adapter as cera's decode loop would.
        adapter.on_text_tokens(&[1, 2, 3]);
        adapter.on_text_tokens(&[4, 5]);
        adapter.on_audio_frames(&[0.1, 0.2, 0.3], 24_000);
        adapter.on_done(cera::FinishReason::MaxTokens);

        assert_eq!(&*recorder.text.lock().unwrap(), &[1, 2, 3, 4, 5]);
        let audio = recorder.audio.lock().unwrap();
        assert_eq!(audio.len(), 1);
        assert_eq!(audio[0].0, vec![0.1, 0.2, 0.3]);
        assert_eq!(audio[0].1, 24_000);
        assert!(matches!(
            &*recorder.done.lock().unwrap(),
            Some(FinishReason::MaxTokens)
        ));
        assert!(
            adapter.done_called,
            "adapter.done_called must flip after on_done"
        );
    }

    /// Before `adapter.on_done` has been forwarded, `done_called`
    /// stays `false`. Protects the error-synthesis branch in
    /// `Session::generate_streaming` — we can only safely synthesize
    /// a terminal `on_done(Error)` on failure when the inner
    /// `cera::Session::generate` hasn't already fired its own `on_done`.
    #[test]
    fn adapter_done_called_starts_false_and_guards_error_synthesis() {
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            calls: Mutex<usize>,
        }
        impl ModalitySink for Recorder {
            fn on_text_tokens(&self, _: Vec<u32>) {}
            fn on_audio_frames(&self, _: Vec<f32>, _: u32) {}
            fn on_done(&self, _: FinishReason) {
                *self.calls.lock().unwrap() += 1;
            }
        }

        let recorder: Arc<Recorder> = Arc::new(Recorder::default());
        let adapter = ForeignSinkAdapter {
            inner: recorder.clone() as Arc<dyn ModalitySink>,
            done_called: false,
        };

        // Simulate the error-branch logic: never forwarded on_done,
        // so the wrapper should synthesize one.
        assert!(!adapter.done_called);
        if !adapter.done_called {
            adapter.inner.on_done(FinishReason::Error {
                message: "simulated pre-decode error".into(),
            });
        }
        assert_eq!(*recorder.calls.lock().unwrap(), 1, "synthesized once");

        // And the double-fire guard: if done_called were already true,
        // the wrapper must skip synthesis.
        let adapter_already_done = ForeignSinkAdapter {
            inner: recorder.clone() as Arc<dyn ModalitySink>,
            done_called: true,
        };
        if !adapter_already_done.done_called {
            adapter_already_done
                .inner
                .on_done(FinishReason::Error { message: "".into() });
        }
        assert_eq!(
            *recorder.calls.lock().unwrap(),
            1,
            "still one — no double-fire"
        );
    }

    /// Mirrors the exact `if armed { ... }` branch in
    /// `AsyncCancelGuard::drop`. Can't build a real `cera::Session`
    /// in a unit test (no model to load), but the guard's logic is
    /// one conditional — a structurally identical probe guard gives
    /// the same coverage. End-to-end verification of
    /// "drop-future-cancels-decode" requires a real model and lands
    /// with PR 6's binding smoke tests / PR 7+'s parity harness.
    #[test]
    fn async_cancel_guard_drop_fires_when_armed_only() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ProbeGuard {
            fired: Arc<AtomicBool>,
            armed: bool,
        }
        impl Drop for ProbeGuard {
            fn drop(&mut self) {
                if self.armed {
                    self.fired.store(true, Ordering::Relaxed);
                }
            }
        }

        // Armed drop → fires.
        let armed_fired = Arc::new(AtomicBool::new(false));
        drop(ProbeGuard {
            fired: armed_fired.clone(),
            armed: true,
        });
        assert!(armed_fired.load(Ordering::Relaxed), "armed drop must fire");

        // Disarmed drop → does not fire (the await-resolved path).
        let disarmed_fired = Arc::new(AtomicBool::new(false));
        let mut g = ProbeGuard {
            fired: disarmed_fired.clone(),
            armed: true,
        };
        g.armed = false;
        drop(g);
        assert!(
            !disarmed_fired.load(Ordering::Relaxed),
            "disarmed drop must not fire"
        );
    }

    /// Replicates the `spawn_blocking(..).await.map_err(..)?` pattern
    /// used by `generate_async` / `generate_streaming_async` — the
    /// wrapper's entire logic. Proves:
    ///
    /// - Successful blocking-closure results propagate through the
    ///   await + ?-sugar unchanged.
    /// - A panic in the blocking closure surfaces as
    ///   `tokio::task::JoinError`, which our `map_err` folds into
    ///   `FfiError::Backend` with the documented prefix.
    ///
    /// Can't construct a real `Session` in a unit test (no model to
    /// load), so the actual async wrappers are exercised only via
    /// this shape-equivalent stand-in. The binding generation step
    /// (PR 6) and the parity harness (PR 7+) will end-to-end exercise
    /// the real methods.
    #[tokio::test]
    async fn spawn_blocking_pattern_propagates_ok_and_maps_join_error() {
        // Ok path: same shape as `generate_async`'s body — the sync
        // closure returns the final value, spawn_blocking + await +
        // map_err hands it back via `?`.
        let map_join = |e: tokio::task::JoinError| FfiError::Backend {
            detail: format!("test join error: {e}"),
        };

        let ok: u32 = tokio::task::spawn_blocking(|| 42u32)
            .await
            .map_err(map_join)
            .expect("tokio should not drop the blocking task");
        assert_eq!(ok, 42);

        // Panic path: the blocking closure panics, JoinError bubbles
        // out, map_err converts it. No `?` here so we can inspect the
        // error variant.
        let panicked = tokio::task::spawn_blocking(|| -> u32 {
            panic!("simulated decode panic");
        })
        .await
        .map_err(map_join);
        match panicked {
            Err(FfiError::Backend { detail }) => {
                assert!(
                    detail.contains("test join error"),
                    "expected prefix, got: {detail}"
                );
            }
            other => panic!("expected Err(Backend), got: {other:?}"),
        }
    }

    #[test]
    fn tool_def_json_marshals_to_core() {
        let ffi = ToolDef {
            name: "get_weather".into(),
            description: Some("weather".into()),
            parameters_json: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#
                .into(),
        };
        let core: cera::tools::ToolDef = ffi.try_into().expect("valid schema");
        assert_eq!(core.name, "get_weather");
        assert_eq!(core.parameters["properties"]["city"]["type"], "string");

        // Empty parameters_json → default empty object schema.
        let bare = ToolDef {
            name: "ping".into(),
            description: None,
            parameters_json: String::new(),
        };
        let core: cera::tools::ToolDef = bare.try_into().unwrap();
        assert_eq!(core.parameters["type"], "object");

        // Malformed schema → error, not panic.
        let bad = ToolDef {
            name: "x".into(),
            description: None,
            parameters_json: "{not json".into(),
        };
        let core: Result<cera::tools::ToolDef, _> = bad.try_into();
        assert!(core.is_err());
    }

    #[test]
    fn parse_tool_calls_ffi_returns_json_args() {
        let calls = parse_tool_calls(
            "<|tool_call_start|>[get_weather(city=\"Paris\")]<|tool_call_end|>".into(),
            ToolFormat::Lfm2Pythonic,
        )
        .expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments_json).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn tool_grammar_ffi_compiles() {
        let tools = vec![ToolDef {
            name: "get_weather".into(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"city":{"type":"string"}}}"#
                .into(),
        }];
        let gbnf = tool_grammar(tools, ToolFormat::Lfm2Pythonic).expect("grammar");
        assert!(cera::grammar::Grammar::parse(&gbnf).is_ok());
    }

    #[test]
    fn detect_tool_format_ffi() {
        assert_eq!(detect_tool_format("lfm2".into()), Some(ToolFormat::Lfm2Pythonic));
        assert_eq!(detect_tool_format("qwen3".into()), Some(ToolFormat::Hermes));
        assert_eq!(detect_tool_format("gpt2".into()), None);
    }
}

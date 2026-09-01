//! `CeraEngine` — the owning, loader-aware front door for the core crate.
//!
//! Previous versions exposed `cera::engine::generate()`, a one-shot helper
//! that owned model + tokenizer only for the duration of a single call.
//! That was retired in PR #27 (Phase 1.1) when `Session` became the
//! canonical stateful API. This module reclaims the `engine` name for
//! what the FFI / CLI / web demos all actually need: a handle that owns
//! the loaded model + tokenizer + manifest for a process's lifetime and
//! hands out cheap `Session<'_>` instances.
//!
//! `from_path` accepts three shapes — a bare `.gguf` (synthesized text
//! manifest), a `.json` LeapBundles manifest, or a directory containing
//! exactly one `.json` manifest. All three converge on an internal
//! `from_manifest` routine that dispatches on `InferenceType`. For
//! callers who have explicit file paths and don't want to fabricate a
//! manifest, `from_files(ModelFiles, cfg)` is the overload.
//!
//! Scope notes (Phase 1.2):
//! - Text models: loaded via the existing CPU / wgpu / Metal paths,
//!   selected by [`BackendPreference`] on [`EngineConfig`].
//! - Audio models (`llama.cpp/lfm2-audio-v1`): the primary text LLM is
//!   loaded the same way as text; the audio decoder + detokenizer +
//!   safetensors tokenizer are not consumed by the engine itself —
//!   they're surfaced via [`CeraEngine::manifest`] for callers (the
//!   CLI today) that drive `cera::audio_engine::generate_audio`
//!   directly. Unified `Session::append_audio` wiring lands in a
//!   follow-up.
//! - VL models (`llama.cpp/image-to-text`): the LLM half (plain
//!   `architecture = "lfm2"` GGUF) loads via the existing text path;
//!   the mmproj GGUF is mmaped and exposed via
//!   `Self::vision_encoder_gguf` for follow-up phases. Image input
//!   isn't wired yet — `Session::append_image` lands in a later
//!   phase. Today VL bundles work text-only.
//! - Remote manifests: `from_path` only resolves local paths in v1. A
//!   manifest whose `load_time_parameters.model` looks like an HTTP(S)
//!   URL is rejected with a typed error pointing at Phase 1.6's
//!   `BundleRepo` as the follow-up. Callers who already have the
//!   bundle on disk should point the manifest at the on-disk file.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gguf::GgufFile;
use crate::kv_cache::KvCacheConfig;
#[cfg(feature = "mmap")]
use crate::manifest::ManifestFiles;
use crate::manifest::{InferenceType, Manifest};
use crate::model::audio_encoder::AudioEncoderWeights;
use crate::model::vision_encoder::VisionEncoderWeights;
use crate::model::{self, Model};
use crate::session::{CeraError, ModalityCapabilities, Session, SessionConfig};
use crate::tokenizer::BpeTokenizer;

// ---------------------------------------------------------------------------
// Public configuration + metadata types
// ---------------------------------------------------------------------------

/// Which compute backend to use when loading a model.
///
/// `Auto` probes `metal → gpu → cpu` at load time with runtime fallback,
/// matching the existing CLI `--device auto` behavior. Explicit variants
/// error if their feature isn't compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    /// `wgpu` (Vulkan / Metal / DX12). Requires the `gpu` feature.
    Gpu,
    /// Native Metal. Requires the `metal` feature + macOS.
    Metal,
}

impl BackendPreference {
    /// Parse a case-insensitive string (`"auto"`, `"cpu"`, `"gpu"`, `"wgpu"`, `"metal"`).
    /// Returns `Err` on an unknown label.
    pub fn parse_str(s: &str) -> Result<Self, CeraError> {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" | "wgpu" => Ok(Self::Gpu),
            "metal" => Ok(Self::Metal),
            other => Err(CeraError::Backend(format!(
                "unknown backend preference `{other}` (use auto, cpu, gpu, or metal)"
            ))),
        }
    }
}

/// Per-engine configuration. Set at `from_path` / `from_files` time;
/// immutable for the engine's lifetime.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// KV cache capacity in tokens. Capped by the model's own `max_seq_len`.
    pub context_size: usize,
    /// Which compute backend to prefer.
    pub backend: BackendPreference,
    /// Optional speculative decoding draft model GGUF path (e.g. DSpark sidecar).
    pub draft_model: Option<PathBuf>,
    /// Optional repository used to resolve `http(s)://` URLs found in a
    /// manifest's `files` entries. When `None`, remote URLs fail with a
    /// clear error asking the caller to either set this field or
    /// pre-download the bundle. Requires the `remote` feature.
    #[cfg(feature = "remote")]
    pub bundle_repo: Option<crate::bundle::BundleRepo>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_size: 4096,
            backend: BackendPreference::Auto,
            draft_model: None,
            #[cfg(feature = "remote")]
            bundle_repo: None,
        }
    }
}

/// Explicit file paths + metadata for `CeraEngine::from_files`. Mirrors
/// [`ManifestFiles`] but with local paths instead of URLs and an optional
/// `inference_type` override (auto-detected from the GGUF header when
/// absent).
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// Required: primary GGUF path.
    pub model: PathBuf,
    /// Optional: multimodal projector GGUF (VL + audio models).
    pub multimodal_projector: Option<PathBuf>,
    /// Optional: audio-decoder GGUF (audio-out models).
    pub audio_decoder: Option<PathBuf>,
    /// Optional: audio tokenizer (usually a `.safetensors` checkpoint).
    pub audio_tokenizer: Option<PathBuf>,
    /// Optional: speculative decoding draft model GGUF (e.g. DSpark / DFlash).
    pub draft_model: Option<PathBuf>,
    /// Forward-compat: any additional named aux file.
    pub extras: std::collections::HashMap<String, PathBuf>,
    /// Explicit inference type. `None` → auto-detect from GGUF
    /// `general.architecture` metadata + aux-file heuristic.
    pub inference_type: Option<InferenceType>,
    /// Optional chat-template override. If set, replaces any template
    /// embedded in the GGUF.
    pub chat_template: Option<String>,
}

/// In-memory counterpart to [`ModelFiles`] for [`CeraEngine::from_parts`].
///
/// [`CeraEngine::from_bytes`] takes a single buffer and is therefore text-only:
/// a VL or audio bundle needs a *second* GGUF, the multimodal projector, and
/// there is nowhere to put it. That is the whole reason this type exists.
/// Targets with no filesystem (wasm above all) can only ever hand the engine
/// bytes, so without it they are locked out of every non-text modality.
///
/// Buffers are `Arc<[u8]>` because `GgufFile` keeps a zero-copy view into
/// them; cloning a `ModelBytes` shares the same allocation rather than
/// duplicating a multi-hundred-megabyte model.
#[derive(Clone)]
pub struct ModelBytes {
    /// Required: the primary model GGUF.
    pub model: Arc<[u8]>,
    /// The multimodal projector GGUF (the "mmproj"): the vision tower for a
    /// VL bundle, the audio encoder for an audio one. `None` is text-only.
    pub multimodal_projector: Option<Arc<[u8]>>,
    /// Optional vocoder GGUF for audio output decoding.
    pub audio_decoder: Option<Arc<[u8]>>,
    /// Optional audio tokenizer / detokenizer GGUF for speech synthesis.
    pub audio_tokenizer: Option<Arc<[u8]>>,
    /// Optional speculative decoding draft model GGUF (e.g. DSpark / DFlash).
    pub draft_model: Option<Arc<[u8]>>,
    /// Explicit inference type. `None` auto-detects from the primary GGUF's
    /// `general.architecture`, then upgrades text → VL when an mmproj is
    /// present (see [`CeraEngine::from_parts`] for why).
    pub inference_type: Option<InferenceType>,
    /// Chat-template override. When set, replaces the GGUF's own template.
    pub chat_template: Option<String>,
    /// Generation defaults from the bundle manifest (if loaded from a bundle).
    pub generation_defaults: Option<crate::manifest::GenerationDefaults>,
}

impl ModelBytes {
    /// Convenience: a text-only `ModelBytes` from a single buffer.
    pub fn text(model: impl Into<Arc<[u8]>>) -> Self {
        Self {
            model: model.into(),
            multimodal_projector: None,
            audio_decoder: None,
            audio_tokenizer: None,
            draft_model: None,
            inference_type: Some(InferenceType::LlamaCppTextToText),
            chat_template: None,
            generation_defaults: None,
        }
    }
}

impl ModelFiles {
    /// Convenience: construct a text-only `ModelFiles` from a single path.
    pub fn text(path: impl Into<PathBuf>) -> Self {
        Self {
            model: path.into(),
            multimodal_projector: None,
            audio_decoder: None,
            audio_tokenizer: None,
            draft_model: None,
            extras: std::collections::HashMap::new(),
            inference_type: Some(InferenceType::LlamaCppTextToText),
            chat_template: None,
        }
    }
}

/// Encoder weights a caller has already parsed, handed to
/// [`CeraEngine::from_gguf`] instead of being read from the manifest's paths.
///
/// Internal: the public route is [`CeraEngine::from_parts`]. Every path-based
/// constructor passes `default()` and keeps loading aux files eagerly.
#[derive(Default)]
struct AuxWeights {
    /// Parsed vision mmproj, for `LlamaCppImageToText` bundles.
    vision_mmproj: Option<Arc<GgufFile>>,
    /// Parsed audio encoder, for `LlamaCppLfm2AudioV1` bundles.
    audio_encoder: Option<Arc<AudioEncoderWeights>>,
    /// Parsed audio decoder (vocoder depthformer), for audio-out bundles.
    audio_decoder: Option<Arc<crate::model::audio_decoder::AudioDecoderWeights>>,
    /// Parsed detokenizer, for audio-out bundles.
    detok_weights: Option<Arc<crate::model::audio_decoder::DetokenizerWeights>>,
    /// Pre-parsed speculative decoding drafter (e.g. from in-memory bytes).
    drafter: Option<Arc<dyn crate::spec::Drafter>>,
}

/// Short summary of the loaded model. Matches the shape planned for the
/// UniFFI `ModelMetadata` record so FFI bindings can surface it without
/// re-deriving.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub architecture: String,
    pub max_seq_len: u32,
    pub vocab_size: u32,
    pub has_chat_template: bool,
    pub quantization: String,
    /// Mirror of GGUF `tokenizer.ggml.add_bos_token`. Consumers that
    /// want to insert a BOS at the head of a raw prompt should honor it —
    /// or, better, tokenize via `BpeTokenizer::encode_special`, which applies
    /// both this and `add_eos_token`.
    pub add_bos_token: bool,
    /// Mirror of GGUF `tokenizer.ggml.add_eos_token`. See `add_bos_token`.
    pub add_eos_token: bool,
}

// ---------------------------------------------------------------------------
// CeraEngine
// ---------------------------------------------------------------------------

/// Owning handle to a loaded model + tokenizer + manifest.
///
/// `model` and `tokenizer` are stored as `Arc` rather than `Box`/owned
/// so [`new_session`](Self::new_session) can hand out cheap
/// lifetime-free [`Session`] handles (see [`Session`]'s doc comment for
/// why the FFI story requires this).
pub struct CeraEngine {
    manifest: Manifest,
    model: Arc<dyn Model>,
    tokenizer: Arc<BpeTokenizer>,
    metadata: ModelMetadata,
    config: EngineConfig,
    /// Audio encoder weights, eagerly loaded from
    /// `manifest.files.multimodal_projector` at construction when the
    /// inference_type is audio. `None` for text-only / VL bundles, or
    /// when the mmproj file is missing / fails to parse (a warn is
    /// logged in that case so ops can spot it without text generation
    /// being affected). Auto-attached to every Session returned by
    /// [`Self::new_session`].
    audio_encoder: Option<Arc<AudioEncoderWeights>>,
    /// Vision-encoder mmproj GGUF, eagerly mmapped from
    /// `manifest.files.multimodal_projector` when the inference_type
    /// is `LlamaCppImageToText`. `None` for text / audio bundles, or
    /// when the mmproj fails to open (warned, not fatal — text-only
    /// chat against a VL bundle still works without it). Kept around
    /// alongside the typed `vision_encoder` for raw-bytes consumers
    /// that need direct GGUF metadata access.
    vision_encoder_gguf: Option<Arc<GgufFile>>,
    /// Typed vision-encoder weights, parsed from the mmproj GGUF.
    /// `Some` whenever `vision_encoder_gguf` is `Some` and the
    /// shape sanity checks pass; `None` when the GGUF was
    /// successfully mmapped but parsing failed (warned). This is
    /// the primary VL accessor — Phase 2's forward pass reads from
    /// it directly.
    vision_encoder: Option<Arc<VisionEncoderWeights>>,
    /// Cached GPU vision encoder, built at construction when `vision_encoder`
    /// is present and `cfg.backend` selects a GPU backend (and the device is
    /// available). Shared into every session via `new_session`; sessions fall
    /// back to the CPU `vision_encoder` when this is `None`.
    gpu_vision_encoder: Option<Arc<dyn crate::model::vision_encoder_gpu::VisionGpuEncode>>,
    /// Cached GPU audio encoder, built at construction when `audio_encoder` is
    /// present and `cfg.backend` selects a GPU backend (and the device is
    /// available). Shared into every session via `new_session`; sessions fall
    /// back to the CPU `audio_encoder` when this is `None`.
    gpu_audio_encoder: Option<Arc<dyn crate::model::audio_encoder_gpu::AudioGpuEncode>>,
    /// Audio decoder (depthformer) weights for audio output generation.
    audio_decoder: Option<Arc<crate::model::audio_decoder::AudioDecoderWeights>>,
    /// Detokenizer weights for audio output generation.
    detok_weights: Option<Arc<crate::model::audio_decoder::DetokenizerWeights>>,
    /// Optional speculative decoding drafter (e.g. DSpark sidecar draft model).
    drafter: Option<Arc<dyn crate::spec::Drafter>>,
}

impl CeraEngine {
    /// Load from a path that may be:
    /// - a bare `.gguf` file → internally synthesizes a text manifest,
    /// - a `.json` LeapBundles manifest → parsed + dispatched on `inference_type`,
    /// - a directory → scanned for exactly one `.json` manifest.
    ///
    /// Requires both `std-fs` (for directory + manifest I/O) and `mmap`
    /// (to mmap-open the GGUF). Both are default-on. Builds without
    /// them (e.g. wasm32) should use [`Self::from_reader`] or
    /// [`Self::from_bytes`] with externally-sourced bytes.
    #[cfg(feature = "mmap")]
    pub fn from_path<P: AsRef<Path>>(path: P, cfg: EngineConfig) -> Result<Self, CeraError> {
        let path = path.as_ref();
        if path.is_dir() {
            let manifest_path = find_single_manifest(path)?;
            Self::from_manifest_file(&manifest_path, cfg)
        } else if has_extension(path, "json") {
            Self::from_manifest_file(path, cfg)
        } else if has_extension(path, "gguf") {
            // Bare `.gguf` → peek at `general.architecture`. Text +
            // audio go through the synthetic-text manifest path; aux
            // files for audio (mmproj) are manifest-driven and
            // consumers who need them must load via manifest or
            // `from_files`. VL is refused at this gate because a
            // bare GGUF can't possibly carry the vision tower —
            // silently downgrading to text would surprise users who
            // later reach for `--image`. Today this arm is
            // unreachable (every published VL main GGUF reports
            // `architecture = "lfm2"` and auto-detect lands on
            // text); if Liquid ever ships `architecture = "lfm2vl"`
            // the typed error tells the user what to do. Unknown
            // arches fall back to text per auto-detect's existing
            // policy.
            let detected = auto_detect_inference_type(path)?;
            match detected {
                InferenceType::LlamaCppTextToText | InferenceType::LlamaCppLfm2AudioV1 => {
                    let manifest = Manifest::synthetic_text(path);
                    Self::from_manifest(manifest, cfg)
                }
                InferenceType::LlamaCppImageToText => Err(CeraError::Backend(format!(
                    "bare-GGUF VL load is not supported (file `{}`); load via a \
                     `.json` manifest, a directory containing one, \
                     `from_files`, or `from_bundle_id` so the vision mmproj \
                     can be attached",
                    path.display()
                ))),
                // `auto_detect_inference_type` defaults unknown arches to
                // Text, so this arm is unreachable today — matched for
                // exhaustiveness if the policy changes.
                InferenceType::Unknown(s) => Err(CeraError::UnsupportedInferenceType(s)),
            }
        } else {
            Err(CeraError::Backend(format!(
                "don't know how to load `{}` — expected a .gguf file, a .json manifest, or a directory containing one",
                path.display()
            )))
        }
    }

    /// Load from an in-memory byte buffer. Text-only; for multi-file loads
    /// (VL / audio) use [`Self::from_path`] with a manifest or
    /// [`Self::from_files`]. Documented as `<50 MB or testing only` —
    /// production paths should stream from disk.
    ///
    /// Unconditional — works in every feature configuration, including
    /// `--no-default-features`. Phase 3's `cera-wasm` crate uses this
    /// (plus [`Self::from_reader`]) to back its OPFS-loaded paths.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, cfg: EngineConfig) -> Result<Self, CeraError> {
        let arc_bytes: Arc<[u8]> = bytes.into();
        let gguf = GgufFile::from_bytes(arc_bytes)
            .map_err(|e| CeraError::Backend(format!("parsing GGUF bytes: {e}")))?;
        let manifest = Manifest::synthetic_text(Path::new("<bytes>"));
        Self::from_gguf(gguf, manifest, cfg, None, AuxWeights::default())
    }

    /// Load a multi-file bundle entirely from memory: the multi-file
    /// counterpart to [`Self::from_bytes`], as [`Self::from_files`] is to
    /// [`Self::from_path`].
    ///
    /// This is what makes VL and audio reachable without a filesystem. Both
    /// modalities need a second GGUF (the multimodal projector), every other
    /// multi-file constructor takes `PathBuf`s, and [`Self::from_bytes`] has
    /// nowhere to put a second buffer, so a target like wasm could previously
    /// load text and nothing else.
    ///
    /// Inference type resolves in three steps, most specific first:
    ///
    /// 1. an explicit `parts.inference_type`, honored as given;
    /// 2. otherwise the primary GGUF's `general.architecture`;
    /// 3. otherwise, if (2) said text-to-text *and* an mmproj was supplied
    ///    **and parsed**, image-to-text.
    ///
    /// Step 3 is not a guess dressed up as a default. Every published LFM2-VL
    /// bundle reports `architecture = "lfm2"`, identical to a text model: the
    /// vision tower lives entirely in the mmproj. Auto-detect therefore says
    /// "text" for every real VL model, and taking it at its word would make
    /// the mmproj argument silently inert, which is the failure mode
    /// `from_path` refuses a bare-GGUF VL load to avoid. Supplying an mmproj
    /// is an unambiguous statement of intent, so it wins. Audio is never
    /// inferred this way (its `lfm2-audio` arch is already distinctive) and
    /// callers who genuinely want text plus an ignored sidecar can say so
    /// with an explicit `inference_type`.
    ///
    /// Unconditional: no `mmap`, no `std-fs`. Aux parsing failures are
    /// non-fatal and warn, matching the path-based loaders, so a bundle with
    /// a broken mmproj still serves text. Note the interaction with step 3:
    /// an *inferred* modality downgrades when its mmproj will not parse, so
    /// `capabilities()` keeps describing what the engine can actually do,
    /// while an *explicit* one is left standing and surfaces the specific
    /// "no encoder attached" error at first use.
    pub fn from_parts(parts: ModelBytes, cfg: EngineConfig) -> Result<Self, CeraError> {
        let gguf = GgufFile::from_bytes(Arc::clone(&parts.model))
            .map_err(|e| CeraError::Backend(format!("parsing GGUF bytes: {e}")))?;

        // Parse the mmproj *before* resolving the modality, because the
        // upgrade below is only sound if the sidecar is real. Deciding
        // "VL" from the mere presence of a buffer and then discovering it is
        // garbage leaves `capabilities.image_in == true` on an engine with no
        // vision encoder: a claim that `append_image` immediately refuses.
        // An unusable mmproj must therefore not be evidence of anything.
        //
        // Skipped entirely when nothing downstream can consult it: an
        // explicit *text* type short-circuits the upgrade and routes no
        // aux weights, so parsing there would burn the work and warn about
        // a sidecar the caller deliberately opted out of. That opt-out is
        // a documented use ("text plus an ignored sidecar"), so it should
        // not be the noisy path.
        let mmproj_can_matter = match &parts.inference_type {
            // Inferred: the upgrade needs to know whether it is usable.
            None => true,
            // Explicit and multimodal: `aux` below needs it parsed.
            Some(InferenceType::LlamaCppImageToText | InferenceType::LlamaCppLfm2AudioV1) => true,
            Some(_) => false,
        };
        let mmproj = parts
            .multimodal_projector
            .as_ref()
            .filter(|_| mmproj_can_matter)
            .and_then(|bytes| parse_aux_gguf(bytes, "multimodal projector"));

        let inference_type = resolve_parts_inference_type(
            parts.inference_type.clone(),
            gguf.get_str("general.architecture").unwrap_or(""),
            mmproj.is_some(),
        );
        check_inference_type_supported(&inference_type)?;

        // Route the parsed mmproj to whichever encoder this modality wants.
        //
        // An *explicit* type is still honored even when the mmproj is
        // unusable: the caller declared the bundle's shape, so the modality
        // stays on and `append_image` / `append_audio` report the specific
        // "no encoder attached" error. That matches how a manifest-driven
        // load behaves. Only the inferred case downgrades, above.
        //
        // A text type ignores the mmproj entirely, which is what makes the
        // documented opt-out ("text plus an ignored sidecar") mean something.
        let (audio_decoder, detok_weights) = {
            let voc_arc = parts
                .audio_decoder
                .and_then(|b| GgufFile::from_bytes(b).ok())
                .map(Arc::new);
            let tok_arc = parts
                .audio_tokenizer
                .and_then(|b| GgufFile::from_bytes(b).ok())
                .map(Arc::new);

            let dec = if let Some(ref vg) = voc_arc {
                crate::model::audio_decoder::AudioDecoderWeights::from_gguf(vg)
                    .map_err(|e| {
                        tracing::warn!("failed to parse audio decoder weights: {e:#}");
                        e
                    })
                    .ok()
                    .map(Arc::new)
            } else if let Some(ref g) = mmproj {
                crate::model::audio_decoder::AudioDecoderWeights::from_gguf(g)
                    .ok()
                    .map(Arc::new)
            } else {
                None
            };

            // The vocoder owns the detokenizer, so it is tried first, and each
            // source is a real fallback rather than an `else if` arm: a
            // present-but-unparseable file used to end the chain and leave the
            // session with no detokenizer at all.
            //
            // Order matters beyond tidiness. `DetokenizerWeights::from_gguf`
            // accepts fallback tensor names (`token_embd.weight`, `dense_2.*`,
            // `token_embd_norm.weight`) so a combined GGUF still loads, and an
            // audio *tokenizer* GGUF carries every one of them -- it is a whole
            // separate `lfm2` model. Parsing it as a detokenizer therefore
            // succeeds while silently substituting the wrong model's tensors:
            // a 65536-row `token_embd.weight` stands in for the 16384-row
            // `emb.emb.weight` code table (8 codebooks x 2048), so audio codes
            // index a table that has nothing to do with them. The audio still
            // sounds broadly like speech, which is what let it go unnoticed,
            // but frames land wrong and some run an order of magnitude past
            // [-1, 1].
            let detok = [voc_arc.as_ref(), tok_arc.as_ref(), mmproj.as_ref()]
                .into_iter()
                .flatten()
                .find_map(|g| {
                    crate::model::audio_decoder::DetokenizerWeights::from_gguf(g)
                        .map_err(|e| {
                            tracing::warn!("failed to parse detokenizer weights: {e:#}");
                            e
                        })
                        .ok()
                        .map(Arc::new)
                });

            (dec, detok)
        };

        let mut aux = match (&inference_type, mmproj) {
            (InferenceType::LlamaCppImageToText, Some(g)) => AuxWeights {
                vision_mmproj: Some(g),
                audio_encoder: None,
                audio_decoder: None,
                detok_weights: None,
                drafter: None,
            },
            (InferenceType::LlamaCppLfm2AudioV1, Some(g)) => AuxWeights {
                vision_mmproj: None,
                audio_encoder: try_parse_audio_encoder(&g, None),
                audio_decoder: None,
                detok_weights: None,
                drafter: None,
            },
            _ => AuxWeights::default(),
        };
        aux.audio_decoder = audio_decoder;
        aux.detok_weights = detok_weights;
        aux.drafter = parts
            .draft_model
            .as_ref()
            .and_then(|b| parse_aux_gguf(b, "draft model"))
            .and_then(|draft_gguf| {
                let base_gguf_arc = Arc::new(gguf.clone());
                init_dspark_drafter(draft_gguf, &base_gguf_arc, "bytes")
            });

        let mut manifest = Manifest::synthetic(
            Path::new("<bytes>"),
            inference_type,
            parts
                .multimodal_projector
                .as_ref()
                .map(|_| "<mmproj-bytes>".to_string()),
        );
        manifest.chat_template = parts.chat_template;
        if let Some(defaults) = parts.generation_defaults {
            manifest.generation_defaults = defaults;
        }
        Self::from_gguf(gguf, manifest, cfg, None, aux)
    }

    /// Load from any `std::io::Read`. Streams the full GGUF into an
    /// owned buffer before parsing. Unconditional — works in every
    /// feature configuration.
    ///
    /// Intended backend for Phase 3's `cera-wasm` (an OPFS-backed
    /// `Read + Seek` shim) and for any consumer that has the bytes
    /// coming from a source other than a filesystem path (decrypted
    /// blob, network stream, archive entry).
    pub fn from_reader<R: Read>(reader: R, cfg: EngineConfig) -> Result<Self, CeraError> {
        let gguf = GgufFile::from_reader(reader)
            .map_err(|e| CeraError::Backend(format!("reading GGUF stream: {e}")))?;
        let manifest = Manifest::synthetic_text(Path::new("<reader>"));
        Self::from_gguf(gguf, manifest, cfg, None, AuxWeights::default())
    }

    /// Load from explicit file paths — skips manifest JSON parsing.
    /// `files.inference_type` decides the loader; `None` auto-detects
    /// from the GGUF header.
    ///
    /// Requires the `mmap` feature (default-on) because it mmap-opens
    /// the primary file. Callers without `mmap` should read the file
    /// manually and use [`Self::from_reader`].
    #[cfg(feature = "mmap")]
    pub fn from_files(files: ModelFiles, cfg: EngineConfig) -> Result<Self, CeraError> {
        let mut manifest = synthesize_manifest_from_files(&files)?;
        // Match `from_manifest_file`'s behavior: relative paths in
        // `multimodal_projector` / `audio_decoder` / etc. are resolved
        // relative to the primary model's directory. Absolute paths and
        // URLs pass through unchanged. Without this, downstream code
        // that expects manifest paths to be normalized (e.g.
        // `try_load_audio_encoder`) would see un-resolved relative
        // paths and could fail to open aux files that happen to live
        // next to the primary GGUF.
        resolve_all_manifest_files(&mut manifest, files.model.parent(), &cfg)?;
        // If the caller overrode the chat template, apply it by threading
        // it through the manifest; the text loader doesn't need to know.
        // (The tokenizer will still be built from the GGUF; template
        // precedence lives on the manifest for downstream consumers.)
        Self::from_manifest_with_primary(manifest, files.model.as_path(), cfg)
    }

    /// Load from a LeapBundles ID + quantization selector, e.g.
    /// `from_bundle_id("LFM2-1.2B-GGUF", "Q4_0", cfg)`.
    ///
    /// Resolves to
    /// `https://huggingface.co/LiquidAI/LeapBundles/resolve/main/{bundle_id}/{quant}.json`,
    /// downloads + caches it via `cfg.bundle_repo`, then loads the
    /// engine through the normal manifest path — which in turn fetches
    /// the GGUF (also via `bundle_repo`) since the manifest's model URL
    /// is an `http(s)://` reference to the model's own HF repo.
    ///
    /// `cfg.bundle_repo` **must** be set; otherwise this returns an
    /// error telling the caller to set it. Requires both `remote` and
    /// `mmap` features.
    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn from_bundle_id(
        bundle_id: &str,
        quant: &str,
        cfg: EngineConfig,
    ) -> Result<Self, CeraError> {
        let repo = cfg.bundle_repo.as_ref().ok_or_else(|| {
            CeraError::Backend(
                "`CeraEngine::from_bundle_id` requires `EngineConfig::bundle_repo` to be set — \
                 construct a `BundleRepo` rooted at your desired store directory and assign it \
                 before calling this constructor."
                    .to_string(),
            )
        })?;
        let want_dspark = quant.to_ascii_lowercase().contains("dspark");
        let clean_quant = quant.split(['+', ' ']).next().unwrap_or(quant).trim();
        let (mut manifest, manifest_dir) = if let Some(known) =
            crate::bundle::known_bundle_manifest(bundle_id, clean_quant)
        {
            (known, None)
        } else {
            let manifest_url = crate::bundle::leap_bundles_manifest_url(bundle_id, clean_quant)?;
            // No caller-supplied hash for manifest JSONs (LeapBundles schema
            // doesn't carry one, and the file is tiny: etag fallback is
            // sufficient). Manifest-level per-file hashes, when they land,
            // would be threaded through from inside `from_manifest_file`.
            let manifest_path = repo.resolve_url(&manifest_url, None)?;
            let m = Manifest::from_file(&manifest_path).map_err(|e| {
                CeraError::Backend(format!(
                    "parsing manifest `{}`: {e}",
                    manifest_path.display()
                ))
            })?;
            (m, manifest_path.parent().map(|p| p.to_path_buf()))
        };

        if want_dspark
            && manifest.files.draft_model.is_none()
            && let Some(draft_url) =
                crate::bundle::hf::known_companion_dspark_url(bundle_id, clean_quant)
        {
            manifest.files.draft_model = Some(draft_url);
        }

        resolve_all_manifest_files(&mut manifest, manifest_dir.as_deref(), &cfg)?;
        let primary = PathBuf::from(&manifest.files.model);
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    /// Load from a Hugging Face repository spec or URL, e.g.
    /// `from_hf("LiquidAI/LFM2.5-VL-3B-GGUF", Some("Q4_K_M"), cfg)` or
    /// `from_hf("https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF", None, cfg)`.
    ///
    /// Inspects the Hugging Face repository via its metadata API, selects
    /// the requested (or best default) quantization, automatically pairs
    /// any modality auxiliary files (e.g. `mmproj-*.gguf` for vision-language
    /// models, `audiodecoder-*.gguf` for audio models), downloads and caches
    /// them via `cfg.bundle_repo`, and loads the engine.
    ///
    /// `cfg.bundle_repo` **must** be set; otherwise this returns an
    /// error telling the caller to set it. Requires both `remote` and
    /// `mmap` features.
    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn from_hf(
        spec_or_url: &str,
        quant: Option<&str>,
        cfg: EngineConfig,
    ) -> Result<Self, CeraError> {
        Self::from_hf_with_strategy(spec_or_url, quant, None, cfg)
    }

    /// Load from a Hugging Face repository spec or URL with an explicit quantization strategy
    /// (e.g. `auto`, `fast-mse`, `hqq`, `quarot`).
    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn from_hf_with_strategy(
        spec_or_url: &str,
        quant: Option<&str>,
        quant_strategy: Option<&str>,
        cfg: EngineConfig,
    ) -> Result<Self, CeraError> {
        let repo = cfg.bundle_repo.as_ref().ok_or_else(|| {
            CeraError::Backend(
                "`CeraEngine::from_hf` requires `EngineConfig::bundle_repo` to be set — \
                 construct a `BundleRepo` rooted at your desired store directory and assign it \
                 before calling this constructor."
                    .to_string(),
            )
        })?;
        let manifest = crate::bundle::hf::inspect_and_resolve_manifest(
            spec_or_url,
            quant,
            quant_strategy,
            Some(repo.store_dir()),
            repo.progress(),
        )?;
        Self::from_manifest(manifest, cfg)
    }

    /// Alias for [`Self::from_hf`] when loading from a full Hugging Face URL.
    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn from_hf_url(
        url: &str,
        quant: Option<&str>,
        cfg: EngineConfig,
    ) -> Result<Self, CeraError> {
        Self::from_hf(url, quant, cfg)
    }

    // --- internal constructors ---

    /// Core assembly: take a pre-constructed `GgufFile` + parsed
    /// manifest, build the tokenizer, load the model, wrap in
    /// `CeraEngine`. All three public constructors funnel through here:
    ///
    /// - `from_bytes` / `from_reader` pass `path = None` — no on-disk
    ///   file to hand to backends.
    /// - `from_manifest_with_primary` (via `from_path` / `from_files`)
    ///   passes `Some(primary)` — Metal and wgpu's auto-dispatch may
    ///   reopen the file by path for their own mmap, so they need the
    ///   original filesystem path even though we also hand them the
    ///   already-parsed `GgufFile`.
    ///
    /// `aux` carries encoder weights the caller has already parsed. It is how
    /// the hermetic constructors reach a modality at all: the eager mmproj
    /// load below is gated on `path.is_some()` so a `from_bytes` call cannot
    /// surreptitiously open a filesystem path the manifest happens to name,
    /// and that gate would otherwise make a no-filesystem VL load impossible
    /// rather than merely unsupported. Path-based callers pass
    /// `AuxWeights::default()` and keep the existing eager-load behavior.
    fn from_gguf(
        gguf: GgufFile,
        manifest: Manifest,
        cfg: EngineConfig,
        path: Option<&Path>,
        aux: AuxWeights,
    ) -> Result<Self, CeraError> {
        // Give rayon's global pool a deterministic width and CPU mask before
        // anything can build it lazily. Only `cera-cli` calls
        // `configure_thread_pool`, so without this every library embedder (the
        // UniFFI bindings, the iOS/Android SDKs, direct users) inherits
        // whatever mask the first rayon touch happened to see. See
        // `backend::cpu::ensure_rayon_global_pool` for the measured cost.
        // Idempotent, so the constructors that funnel through here repeatedly
        // pay only the first one.
        crate::backend::cpu::ensure_rayon_global_pool();

        // Covers `from_bytes` / `from_reader`, which skip the pre-filter
        // in `from_manifest_with_primary`. Text LLMs AND LFM2-audio
        // models both load the primary GGUF through the same path;
        // audio aux files (decoder, mmproj, safetensors tokenizer) stay
        // on the manifest for the audio pipeline to pick up separately.
        check_inference_type_supported(&manifest.inference_type)?;

        let gguf_arc = Arc::new(gguf);
        let tokenizer = BpeTokenizer::from_gguf(&gguf_arc)
            .map_err(|e| CeraError::Backend(format!("loading tokenizer: {e}")))?;
        // Extract `general.file_type` BEFORE `load_text_model` consumes
        // the gguf — that's the only place this metadata exists, and
        // we need it for the metadata's quantization label.
        let quantization = gguf_arc
            .get_u32("general.file_type")
            .map(ftype_label)
            .unwrap_or_else(|| "unknown".to_string());
        let drafter = aux
            .drafter
            .or_else(|| try_load_drafter(&manifest, &cfg, &gguf_arc));
        let model_gguf = Arc::try_unwrap(gguf_arc).unwrap_or_else(|a| (*a).clone());
        // `load_text_model` returns `Box<dyn Model>`; convert to `Arc`
        // at the engine boundary. `Arc::from(Box<T>)` is documented on
        // `Arc` for exactly this sizing dance (including `T: ?Sized`).
        let model: Arc<dyn Model> = Arc::from(load_text_model(model_gguf, path, &cfg)?);
        let metadata = build_metadata(model.as_ref(), &tokenizer, &manifest, quantization);
        // Eager mmproj load for audio + VL bundles. Gated on
        // `path.is_some()` because hermetic constructors
        // (`from_bytes` / `from_reader`) shouldn't surreptitiously
        // open filesystem paths even if the manifest mentions one —
        // that would violate the no-filesystem contract. Failure
        // here is non-fatal: warn and leave the encoder unset so
        // text generation still works on a partly-broken bundle.
        //
        // A caller-supplied `aux` wins outright rather than merging: it is
        // the only route a hermetic caller has, and when both could apply
        // (`from_files` with pre-parsed weights) the explicit argument is the
        // more specific instruction.
        let audio_encoder = aux.audio_encoder.or_else(|| {
            if path.is_some() {
                try_load_audio_encoder(&manifest)
            } else {
                None
            }
        });
        let vision_encoder_gguf = aux.vision_mmproj.or_else(|| {
            if path.is_some() {
                try_load_vision_encoder_gguf(&manifest)
            } else {
                None
            }
        });
        // Typed weights only when the raw mmproj loaded — we don't
        // re-attempt the open here. Failure to parse leaves the
        // typed slot unset (warned in `try_parse_vision_encoder`)
        // but text-only chat keeps working. Pass the mmproj path
        // along so the warn log can identify which file failed.
        let vl_mmproj_path = manifest.files.multimodal_projector.as_deref();
        let vision_encoder = vision_encoder_gguf
            .as_ref()
            .and_then(|g| try_parse_vision_encoder(g, vl_mmproj_path));
        // Build a cached GPU vision encoder when the backend selects one and a
        // device is available. Uploading the mmproj to the GPU here (once)
        // keeps it off the per-image hot path. Falls back to `None` (CPU
        // encode) for `Cpu`, disabled features, or device-init failure.
        let gpu_vision_encoder = vision_encoder.as_ref().and_then(|w| {
            crate::model::vision_encoder_gpu::build_gpu_vision_encoder(w, cfg.backend)
        });
        let gpu_audio_encoder = audio_encoder
            .as_ref()
            .and_then(|w| crate::model::audio_encoder_gpu::build_gpu_audio_encoder(w, cfg.backend));
        let (eager_audio_decoder, eager_detok_weights) =
            if path.is_some() && aux.audio_decoder.is_none() {
                try_load_audio_decoder_and_detok(&manifest)
            } else {
                (aux.audio_decoder, aux.detok_weights)
            };

        let audio_decoder = eager_audio_decoder.filter(|dec| {
            let model_dim = model.config().hidden_size;
            let vocoder_dim = dec.decoder_config.n_embd;
            if vocoder_dim != model_dim {
                tracing::warn!(
                    "audio decoder embedding dimension ({vocoder_dim}) does not match model hidden size ({model_dim}); ignoring mismatched vocoder"
                );
                false
            } else {
                true
            }
        });
        let detok_weights = if audio_decoder.is_some() {
            eager_detok_weights
        } else {
            None
        };
        Ok(Self {
            manifest,
            model,
            tokenizer: Arc::new(tokenizer),
            metadata,
            config: cfg,
            audio_encoder,
            audio_decoder,
            detok_weights,
            vision_encoder_gguf,
            vision_encoder,
            gpu_vision_encoder,
            gpu_audio_encoder,
            drafter,
        })
    }

    #[cfg(feature = "mmap")]
    fn from_manifest_file(path: &Path, cfg: EngineConfig) -> Result<Self, CeraError> {
        let mut manifest = Manifest::from_file(path).map_err(|e| {
            CeraError::Backend(format!("parsing manifest `{}`: {e}", path.display()))
        })?;
        resolve_all_manifest_files(&mut manifest, path.parent(), &cfg)?;
        let primary = PathBuf::from(&manifest.files.model);
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    /// Opens the primary GGUF at `primary` and delegates assembly to
    /// [`Self::from_gguf`] with `Some(primary)` so Metal/GPU backends
    /// can reach the on-disk file.
    ///
    /// Requires `mmap` because it opens the primary via `GgufFile::open`.
    #[cfg(feature = "mmap")]
    fn from_manifest_with_primary(
        manifest: Manifest,
        primary: &Path,
        cfg: EngineConfig,
    ) -> Result<Self, CeraError> {
        // Pre-filter on inference_type so VL / Unknown manifests fail
        // fast without paying for the GGUF mmap + header parse. `from_gguf`
        // checks again for the in-memory constructors that skip this path.
        check_inference_type_supported(&manifest.inference_type)?;
        let gguf = GgufFile::open(primary)
            .map_err(|e| CeraError::Backend(format!("opening `{}`: {e}", primary.display())))?;
        Self::from_gguf(gguf, manifest, cfg, Some(primary), AuxWeights::default())
    }

    /// Convergence point for `from_path(.gguf)`. Re-resolves the primary
    /// from the synthetic manifest and dispatches through
    /// [`Self::from_manifest_with_primary`].
    #[cfg(feature = "mmap")]
    fn from_manifest(mut manifest: Manifest, cfg: EngineConfig) -> Result<Self, CeraError> {
        resolve_all_manifest_files(&mut manifest, None, &cfg)?;
        let primary = PathBuf::from(&manifest.files.model);
        Self::from_manifest_with_primary(manifest, &primary, cfg)
    }

    // --- accessors ---

    /// Create a new [`Session`] sharing ownership of the engine's model
    /// and tokenizer via `Arc` clones. The returned session outlives
    /// `&self`; the engine keeps the originals live for every session
    /// it handed out. The session's [`ModalityCapabilities`] is derived
    /// from the manifest's `inference_type`.
    pub fn new_session(&self, cfg: SessionConfig) -> Result<Session, CeraError> {
        let mut session = Session::new(
            Arc::clone(&self.model),
            Arc::clone(&self.tokenizer),
            self.capabilities(),
            cfg,
        )?;
        // Auto-attach the eagerly-loaded audio encoder so callers
        // can `session.append_audio(...)` directly without first
        // loading + attaching the mmproj GGUF. Encoder is shared
        // by Arc across every session this engine hands out.
        if let Some(encoder) = &self.audio_encoder {
            session.attach_audio_encoder(Arc::clone(encoder));
        }
        // Same auto-attach for the vision encoder — VL bundles
        // populate `vision_encoder` at engine construction; cloning
        // the Arc per session is cheap.
        if let Some(encoder) = &self.vision_encoder {
            session.attach_vision_encoder(Arc::clone(encoder));
        }
        // GPU vision encoder (if one was built); the session prefers it over
        // the CPU encoder for image input within the GPU kernel's capacity.
        if let Some(gpu) = &self.gpu_vision_encoder {
            session.attach_gpu_vision_encoder(Arc::clone(gpu));
        }
        // GPU audio encoder (if one was built); the session prefers it over the
        // CPU encoder for PCM input the GPU kernels can take.
        if let Some(gpu) = &self.gpu_audio_encoder {
            session.attach_gpu_audio_encoder(Arc::clone(gpu));
        }
        // Auto-attach vocoder for audio output generation when present.
        if let (Some(decoder), Some(detok)) = (&self.audio_decoder, &self.detok_weights) {
            session.attach_vocoder(Arc::clone(decoder), Arc::clone(detok));
        }
        // Auto-attach speculative decoding drafter when present.
        if let Some(drafter) = &self.drafter {
            session.attach_drafter(drafter.as_ref());
        }
        session.set_default_generate_opts(self.default_generate_opts());
        Ok(session)
    }

    /// Reserved special-token names, in priority order, that mark the audio
    /// insertion point inside a rendered chat template. The first one the
    /// tokenizer actually defines is used. Shared by [`Self::transcribe`] and
    /// the CLI's audio chat path so the two never drift.
    pub const AUDIO_MARKER_CANDIDATES: [&'static str; 4] = [
        "<|reserved_4|>",
        "<|reserved_5|>",
        "<|reserved_6|>",
        "<|reserved_7|>",
    ];

    /// Find the unique index of `marker_id` in `tokens` (single pass, no
    /// allocation). The caller slices `tokens[..idx]` / `tokens[idx + 1..]` for
    /// the prefix/suffix around the audio marker. Errors name both `marker_name`
    /// and `marker_id` so callers can act: "not found" means the template
    /// stripped/escaped the placeholder; "appears N times" means user text
    /// contained a literal marker, making the insertion point ambiguous.
    pub fn split_tokens_at_marker(
        tokens: &[u32],
        marker_id: u32,
        marker_name: &str,
    ) -> Result<usize, CeraError> {
        let mut found: Option<usize> = None;
        let mut count: usize = 0;
        for (i, &t) in tokens.iter().enumerate() {
            if t == marker_id {
                count += 1;
                if found.is_none() {
                    found = Some(i);
                }
            }
        }
        match (count, found) {
            (1, Some(idx)) => Ok(idx),
            (0, _) => Err(CeraError::Backend(format!(
                "audio marker token `{marker_name}` (id {marker_id}) not found in rendered \
                 chat-template tokens — the template may have stripped or escaped the placeholder"
            ))),
            (n, _) => Err(CeraError::Backend(format!(
                "audio marker token `{marker_name}` (id {marker_id}) appears {n} times in \
                 rendered tokens; expected exactly one insertion point (check that prompt/system \
                 text does not contain a literal `{marker_name}`)"
            ))),
        }
    }

    /// Transcribe mono `f32` PCM audio to text using the model's trained `"Perform ASR."` chat mode.
    ///
    /// Renders the chat template with a system `"Perform ASR."` turn and an audio-marker placeholder
    /// in the user turn, prefills `prefix tokens → audio → suffix tokens`, then greedily decodes and
    /// returns the trimmed transcription. Requires an audio-capable bundle (one whose mmproj / audio
    /// encoder is attached); on a text-only model `append_audio` returns
    /// [`CeraError::UnsupportedModality`].
    ///
    /// `sample_rate` must match the audio encoder's expected rate (resample beforehand if needed).
    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<String, CeraError> {
        use crate::session::{FinishReason, GenerateOpts, ModalitySink};
        use crate::tokenizer::{ChatMessage, apply_chat_template};

        let tok = self.tokenizer();
        // The audio insertion point is marked with a reserved special token, split out after render.
        let (marker_id, marker_name) = Self::AUDIO_MARKER_CANDIDATES
            .into_iter()
            .find_map(|name| tok.special_token_id(name).map(|id| (id, name)))
            .ok_or_else(|| {
                CeraError::Backend(
                    "no audio marker special token (<|reserved_4|>..7) in tokenizer".to_string(),
                )
            })?;

        let messages = [
            ChatMessage {
                role: "system".to_string(),
                content: "Perform ASR.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: marker_name.to_string(),
            },
        ];
        let formatted = apply_chat_template(tok, &messages, true)
            .map_err(|e| CeraError::Backend(format!("chat template render failed: {e}")))?;
        let toks = tok.encode(&formatted);

        let split = Self::split_tokens_at_marker(&toks, marker_id, marker_name)?;

        let mut session = self.new_session(SessionConfig::default())?;
        if split > 0 {
            session.append_tokens(&toks[..split])?;
        }
        session.append_audio(pcm, sample_rate)?;
        if split + 1 < toks.len() {
            session.append_tokens(&toks[split + 1..])?;
        }

        struct CollectSink {
            tokens: Vec<u32>,
        }
        impl ModalitySink for CollectSink {
            fn on_text_tokens(&mut self, tokens: &[u32]) {
                self.tokens.extend_from_slice(tokens);
            }
            fn on_done(&mut self, _reason: FinishReason) {}
        }

        let mut sink = CollectSink { tokens: Vec::new() };
        // Greedy decode for deterministic transcription. Keep the default
        // `max_tokens` (256) as the safety ceiling — generation stops on EOS;
        // a low hard cap (was 64) silently truncated longer transcriptions.
        let opts = GenerateOpts {
            temperature: 0.0,
            ..GenerateOpts::default()
        };
        session.generate(&opts, &mut sink)?;
        Ok(tok.decode(&sink.tokens).trim().to_string())
    }

    /// Borrow the eagerly-loaded audio encoder, if any. Most callers
    /// shouldn't need this — [`Self::new_session`] auto-attaches the
    /// encoder to every session — but the audio output pipeline and
    /// custom integration tests can use it for non-Session
    /// computations (encoding embeddings without an LLM forward).
    pub fn audio_encoder(&self) -> Option<&Arc<AudioEncoderWeights>> {
        self.audio_encoder.as_ref()
    }

    /// Borrow the typed vision-encoder weights parsed from the
    /// mmproj GGUF, if any. **Primary VL accessor** — Phase 2's
    /// forward pass reads from this. `Some` whenever the mmproj
    /// loaded AND the typed parse succeeded; `None` for non-VL
    /// bundles, hermetic constructors (which skip the eager
    /// load), or when parsing the mmproj failed at engine
    /// construction (warned via tracing).
    pub fn vision_encoder(&self) -> Option<&Arc<VisionEncoderWeights>> {
        self.vision_encoder.as_ref()
    }

    /// Whether a GPU vision encoder was built at construction (true when a VL
    /// mmproj loaded, `cfg.backend` selected a GPU backend, and the device was
    /// available). When false, image input falls back to the CPU encoder.
    /// Primarily for tests/diagnostics — sessions auto-select the GPU path.
    pub fn has_gpu_vision_encoder(&self) -> bool {
        self.gpu_vision_encoder.is_some()
    }

    /// Whether a GPU audio encoder was built at construction (true when an audio
    /// mmproj loaded, `cfg.backend` selected a supported GPU backend, and the
    /// device was available). When false, PCM input falls back to the CPU
    /// encoder. Primarily for tests/diagnostics; sessions auto-select the GPU
    /// path.
    pub fn has_gpu_audio_encoder(&self) -> bool {
        self.gpu_audio_encoder.is_some()
    }

    /// Borrow the raw mmapped vision-encoder mmproj GGUF, if any.
    /// `Some` for VL bundles loaded from filesystem paths via
    /// `from_path`, `from_files`, or `from_bundle_id`. Hermetic
    /// constructors (`from_bytes`, `from_reader`) skip the eager
    /// mmap to keep the no-filesystem contract, so they always
    /// return `None` here.
    ///
    /// **Escape hatch for raw-bytes consumers — hidden from
    /// public docs.** Most callers should reach for the typed
    /// [`Self::vision_encoder`] instead. This accessor stays
    /// around for tools that need direct GGUF metadata access
    /// (debug inspection, format-shape introspection, future
    /// `cera inspect-mmproj`-style commands), but it can
    /// disagree with `vision_encoder()` — if the mmap succeeded
    /// but the typed parse failed, this returns `Some` while
    /// `vision_encoder()` returns `None`. Callers gating VL
    /// support must use `vision_encoder()`. Today, image input
    /// still errors; text-only chat against a VL bundle works.
    #[doc(hidden)]
    pub fn vision_encoder_gguf(&self) -> Option<&Arc<GgufFile>> {
        self.vision_encoder_gguf.as_ref()
    }

    /// Modality capabilities reported by the loaded model, derived from
    /// the manifest's `inference_type`. Useful for FFI consumers that
    /// want to gate UI / API surfaces on what the model supports
    /// without constructing a [`Session`].
    pub fn capabilities(&self) -> ModalityCapabilities {
        ModalityCapabilities::from_inference_type(&self.manifest.inference_type)
    }

    /// Borrow the loaded model. Used by the audio pipeline today;
    /// unified `Session::append_audio` will subsume this in a follow-up.
    pub fn model(&self) -> &dyn Model {
        self.model.as_ref()
    }

    /// Shared refcounted handle to the loaded model. Used by callers
    /// (FFI wrappers, the audio pipeline, future trait impls) that
    /// need to keep the model alive independently of the engine.
    pub fn model_arc(&self) -> Arc<dyn Model> {
        Arc::clone(&self.model)
    }

    /// Borrow the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        self.tokenizer.as_ref()
    }

    /// Shared refcounted handle to the tokenizer.
    pub fn tokenizer_arc(&self) -> Arc<BpeTokenizer> {
        Arc::clone(&self.tokenizer)
    }

    /// Borrow the parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Borrow the metadata summary.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Borrow the engine config.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Returns default generation options for this engine, populated from the
    /// manifest's advisory sampling defaults (if loaded from a bundle manifest)
    /// or standard defaults.
    pub fn default_generate_opts(&self) -> crate::session::GenerateOpts {
        crate::session::GenerateOpts::from_manifest(&self.manifest)
    }

    /// Configure the model's KV prefix cache. Passthrough to
    /// `Model::configure_cache`; exposed here so callers that only hold
    /// a `CeraEngine` don't need to reach into `engine.model()`.
    pub fn configure_cache(&self, cfg: KvCacheConfig) {
        self.model.configure_cache(cfg);
    }

    /// Clear the in-memory warm KV prefix cache, preserving cold disk files.
    pub fn clear_warm_cache(&self) {
        self.model.clear_warm_cache();
    }

    /// Clear the model's KV prefix cache (both warm memory and cold disk tiers).
    pub fn clear_cache(&self) {
        self.model.clear_cache();
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

// Only used on `mmap` builds (by `from_path` + `find_single_manifest`).
#[cfg(feature = "mmap")]
fn has_extension(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

#[cfg(feature = "mmap")]
fn find_single_manifest(dir: &Path) -> Result<PathBuf, CeraError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| CeraError::Backend(format!("reading directory `{}`: {e}", dir.display())))?;
    let mut jsons: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|e| CeraError::Backend(format!("reading directory entry: {e}")))?;
        let path = entry.path();
        if path.is_file() && has_extension(&path, "json") {
            jsons.push(path);
        }
    }
    match jsons.len() {
        0 => Err(CeraError::Backend(format!(
            "no .json manifest in directory `{}`",
            dir.display()
        ))),
        1 => jsons.pop().ok_or_else(|| {
            CeraError::Backend(format!(
                "no .json manifest in directory `{}`",
                dir.display()
            ))
        }),
        n => {
            jsons.sort();
            let names: Vec<String> = jsons
                .iter()
                .filter_map(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
                .collect();
            Err(CeraError::Backend(format!(
                "{n} .json manifests in directory `{}` (expected exactly one): {}",
                dir.display(),
                names.join(", ")
            )))
        }
    }
}

/// Resolve every file reference in `manifest.files` to a local path,
/// rewriting the manifest's URL/path strings in place. A remote
/// `http(s)://` URL is downloaded + cached via `EngineConfig::bundle_repo`
/// (with the `remote` feature); a relative path is joined against
/// `manifest_dir` when provided; an absolute path is kept as-is.
///
/// Fields walked (in declaration order):
/// - `files.model` (required)
/// - `files.multimodal_projector` (optional — VL + audio bundles)
/// - `files.audio_decoder` (optional — audio-out bundles)
/// - `files.audio_tokenizer` (optional — audio-in bundles)
/// - `files.extras` (every entry — forward-compat aux roles)
///
/// Consumers downstream of the loader (audio pipeline, VL loader, etc.)
/// read back from `engine.manifest().files.*` and expect local paths,
/// so every URL must be rewritten before we hand the manifest on.
///
/// Gated on `mmap` — the callers (`from_manifest_file`, `from_manifest`)
/// are both `mmap`-only. `from_bytes` / `from_reader` skip path
/// resolution entirely since they receive bytes.
#[cfg(feature = "mmap")]
fn resolve_all_manifest_files(
    manifest: &mut Manifest,
    manifest_dir: Option<&Path>,
    cfg: &EngineConfig,
) -> Result<(), CeraError> {
    manifest.files.model = resolve_url_or_path(&manifest.files.model, manifest_dir, cfg)?
        .to_string_lossy()
        .into_owned();

    for slot in [
        &mut manifest.files.multimodal_projector,
        &mut manifest.files.audio_decoder,
        &mut manifest.files.audio_tokenizer,
        &mut manifest.files.draft_model,
    ] {
        if let Some(s) = slot.as_ref() {
            let resolved = resolve_url_or_path(s, manifest_dir, cfg)?;
            *slot = Some(resolved.to_string_lossy().into_owned());
        }
    }

    for value in manifest.files.extras.values_mut() {
        *value = resolve_url_or_path(value, manifest_dir, cfg)?
            .to_string_lossy()
            .into_owned();
    }

    Ok(())
}

#[cfg(feature = "mmap")]
fn resolve_url_or_path(
    value: &str,
    base_dir: Option<&Path>,
    cfg: &EngineConfig,
) -> Result<PathBuf, CeraError> {
    if is_remote_url(value) {
        #[cfg(feature = "remote")]
        {
            if let Some(repo) = cfg.bundle_repo.as_ref() {
                return repo.resolve_url(value, None);
            }
            return Err(CeraError::Backend(format!(
                "manifest references remote URL `{value}` — set `EngineConfig::bundle_repo` \
                 to a `BundleRepo` rooted at your desired store directory, or pre-download \
                 the bundle and pass a local file path."
            )));
        }
        #[cfg(not(feature = "remote"))]
        {
            let _ = cfg;
            return Err(CeraError::Backend(format!(
                "manifest references remote URL `{value}` — rebuild cera with the `remote` \
                 feature + set `EngineConfig::bundle_repo`, or pre-download the bundle \
                 and pass a local file path."
            )));
        }
    }
    if let Some(rest) = strip_file_scheme(value) {
        // `file://…` isn't portable via `Path::new` (Windows especially),
        // so reject until we take a real URI dependency. Users with a
        // `file://` URI in hand can drop the scheme before calling.
        return Err(CeraError::Backend(format!(
            "manifest references `file://` URI `{value}` — cera doesn't parse file URIs yet; \
             pass the local path directly (e.g. `{rest}`)."
        )));
    }
    let p = Path::new(value);
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else if let Some(base) = base_dir {
        Ok(base.join(p))
    } else {
        Ok(p.to_path_buf())
    }
}

#[cfg(feature = "mmap")]
fn is_remote_url(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Return the path-like tail of a `file://` URI, or `None` if the input
/// isn't a file URI. Case-insensitive on the scheme. Does NOT decode
/// percent-encoding or handle Windows drive letters; the caller errors
/// out rather than trying to interpret it.
#[cfg(feature = "mmap")]
fn strip_file_scheme(s: &str) -> Option<&str> {
    let lower = s.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("file://") {
        // Slice from the same byte offset in the original string so we
        // preserve case.
        let offset = s.len() - rest.len();
        Some(&s[offset..])
    } else {
        None
    }
}

/// Build a minimal `Manifest` from an explicit `ModelFiles`.
#[cfg(feature = "mmap")]
fn synthesize_manifest_from_files(files: &ModelFiles) -> Result<Manifest, CeraError> {
    let inference_type = match files.inference_type.clone() {
        Some(it) => it,
        None => auto_detect_inference_type(&files.model)?,
    };

    let model_str = files.model.to_string_lossy().into_owned();
    let mmproj = files
        .multimodal_projector
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let audio_decoder = files
        .audio_decoder
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let audio_tokenizer = files
        .audio_tokenizer
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let draft_model = files
        .draft_model
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    let mut extras_str = std::collections::HashMap::with_capacity(files.extras.len());
    for (k, v) in &files.extras {
        extras_str.insert(k.clone(), v.to_string_lossy().into_owned());
    }

    // Build a serde_json::Value that mirrors the typed shape so
    // `Manifest::raw` stays useful for consumers that inspect it.
    let mut load_params = serde_json::Map::new();
    load_params.insert("model".into(), serde_json::Value::String(model_str.clone()));
    if let Some(v) = &mmproj {
        load_params.insert(
            "multimodal_projector".into(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = &audio_decoder {
        load_params.insert("audio_decoder".into(), serde_json::Value::String(v.clone()));
    }
    if let Some(v) = &audio_tokenizer {
        load_params.insert(
            "audio_tokenizer".into(),
            serde_json::Value::String(v.clone()),
        );
    }
    if let Some(v) = &draft_model {
        load_params.insert("draft_model".into(), serde_json::Value::String(v.clone()));
    }
    for (k, v) in &extras_str {
        load_params.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    if let Some(t) = &files.chat_template {
        load_params.insert("chat_template".into(), serde_json::Value::String(t.clone()));
    }

    let mut raw_map = serde_json::Map::new();
    raw_map.insert(
        "inference_type".into(),
        serde_json::Value::String(inference_type.as_str().to_string()),
    );
    raw_map.insert(
        "schema_version".into(),
        serde_json::Value::String("1.0.0".into()),
    );
    raw_map.insert(
        "load_time_parameters".into(),
        serde_json::Value::Object(load_params),
    );

    let defaults_shape = inference_type_defaults_shape(&inference_type);
    Ok(Manifest {
        inference_type,
        schema_version: "1.0.0".into(),
        files: ManifestFiles {
            model: model_str,
            multimodal_projector: mmproj,
            audio_decoder,
            audio_tokenizer,
            draft_model,
            extras: extras_str,
        },
        chat_template: files.chat_template.clone(),
        // For `from_files` the caller hasn't provided sampling defaults;
        // surface a zero-info `Text` variant for text/VL models and an
        // empty `Audio` variant for audio models. Consumers who need
        // defaults should go through a real manifest.
        //
        // Key on the *resolved* `inference_type` — using `files.inference_type`
        // (pre-resolution) would hand the `Text` defaults shape to an
        // auto-detected audio model.
        generation_defaults: match defaults_shape {
            DefaultsShape::Text => crate::manifest::GenerationDefaults::Text {
                temperature: None,
                min_p: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
            },
            DefaultsShape::Audio => crate::manifest::GenerationDefaults::Audio {
                number_of_decoding_threads: None,
                audio_temperature: None,
                audio_top_k: None,
                temperature: None,
                min_p: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
            },
            DefaultsShape::Other => crate::manifest::GenerationDefaults::Other {
                raw: serde_json::Value::Null,
            },
        },
        raw: serde_json::Value::Object(raw_map),
    })
}

// Only used by `synthesize_manifest_from_files` (mmap-gated).
#[cfg(feature = "mmap")]
enum DefaultsShape {
    Text,
    Audio,
    Other,
}

#[cfg(feature = "mmap")]
fn inference_type_defaults_shape(it: &InferenceType) -> DefaultsShape {
    match it {
        InferenceType::LlamaCppLfm2AudioV1 => DefaultsShape::Audio,
        InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText => {
            DefaultsShape::Text
        }
        InferenceType::Unknown(_) => DefaultsShape::Other,
    }
}

/// Try to load the audio encoder weights from
/// `manifest.files.multimodal_projector` when the inference_type
/// is audio. Returns `None` (with a `warn!`) on any failure so the
/// engine can continue serving text generation against partly-
/// broken bundles. Path-only — the manifest's `multimodal_projector`
/// field is already a resolved filesystem path by the time we get
/// here (via `resolve_all_manifest_files`).
///
/// `mmap`-gated because the underlying `GgufFile::open` is. Wasm
/// builds without `mmap` get a stub that returns `None`; on those
/// targets aux files would have to come through `from_bytes` /
/// `from_reader`, which already skip auto-attach by contract.
#[cfg(feature = "mmap")]
fn try_load_audio_encoder(manifest: &Manifest) -> Option<Arc<AudioEncoderWeights>> {
    if !matches!(manifest.inference_type, InferenceType::LlamaCppLfm2AudioV1) {
        return None;
    }
    let mmproj_path = manifest.files.multimodal_projector.as_ref()?;
    let path = Path::new(mmproj_path);
    let gguf = match GgufFile::open(path) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                path = %path.display(),
                error = %format!("{e:#}"),
                "audio mmproj GGUF failed to open; audio input will surface \
                 as 'no audio encoder attached' until a working mmproj is supplied"
            );
            return None;
        }
    };
    try_parse_audio_encoder(&gguf, Some(&path.display().to_string()))
}

/// Parse an already-opened mmproj GGUF into typed `AudioEncoderWeights`.
/// Split out of [`try_load_audio_encoder`] so the in-memory constructors,
/// which have the GGUF but no path, share the same warn-and-continue policy.
fn try_parse_audio_encoder(
    gguf: &Arc<GgufFile>,
    path: Option<&str>,
) -> Option<Arc<AudioEncoderWeights>> {
    match AudioEncoderWeights::from_gguf(gguf) {
        Ok(w) => Some(Arc::new(w)),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                path = %path.unwrap_or("<in-memory>"),
                error = %format!("{e:#}"),
                "audio mmproj GGUF parsed but encoder weights failed to load; \
                 audio input will surface as 'no audio encoder attached'"
            );
            None
        }
    }
}

/// Parse aux GGUF bytes, warning rather than failing the whole load.
///
/// Matches the path-based loaders' policy: a bundle whose mmproj is corrupt
/// still serves text, and the modality surfaces its own "no encoder attached"
/// error when someone actually reaches for it.
fn parse_aux_gguf(bytes: &Arc<[u8]>, kind: &str) -> Option<Arc<GgufFile>> {
    match GgufFile::from_bytes(Arc::clone(bytes)) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                kind = %kind,
                error = %format!("{e:#}"),
                "in-memory {kind} GGUF failed to parse; continuing without this component"
            );
            None
        }
    }
}

/// Eager audio decoder & detokenizer GGUF loader for audio-out bundles.
#[cfg(feature = "mmap")]
fn try_load_audio_decoder_and_detok(
    manifest: &Manifest,
) -> (
    Option<Arc<crate::model::audio_decoder::AudioDecoderWeights>>,
    Option<Arc<crate::model::audio_decoder::DetokenizerWeights>>,
) {
    if !matches!(manifest.inference_type, InferenceType::LlamaCppLfm2AudioV1) {
        return (None, None);
    }
    let voc_path = manifest.files.audio_decoder.as_deref().map(Path::new);
    let tok_path = manifest.files.audio_tokenizer.as_deref().map(Path::new);

    let voc_gguf = voc_path.and_then(|p| match GgufFile::open_arc(p) {
        Ok(g) => Some(g),
        Err(e) => {
            tracing::warn!("failed to open audio_decoder GGUF `{}`: {e:#}", p.display());
            None
        }
    });

    let tok_gguf = tok_path.and_then(|p| match GgufFile::open_arc(p) {
        Ok(g) => Some(g),
        Err(e) => {
            tracing::warn!(
                "failed to open audio_tokenizer GGUF `{}`: {e:#}",
                p.display()
            );
            None
        }
    });

    let dec = if let Some(ref vg) = voc_gguf {
        crate::model::audio_decoder::AudioDecoderWeights::from_gguf(vg)
            .ok()
            .map(Arc::new)
    } else {
        None
    };

    // Vocoder first, then the audio tokenizer, for the reason spelled out on
    // the other detokenizer load above: a tokenizer GGUF parses as a
    // detokenizer through the fallback tensor names while carrying a different
    // model's weights.
    let detok = [voc_gguf.as_ref(), tok_gguf.as_ref()]
        .into_iter()
        .flatten()
        .find_map(|g| {
            crate::model::audio_decoder::DetokenizerWeights::from_gguf(g)
                .ok()
                .map(Arc::new)
        });

    (dec, detok)
}

#[cfg(not(feature = "mmap"))]
fn try_load_audio_decoder_and_detok(
    _manifest: &Manifest,
) -> (
    Option<Arc<crate::model::audio_decoder::AudioDecoderWeights>>,
    Option<Arc<crate::model::audio_decoder::DetokenizerWeights>>,
) {
    (None, None)
}

#[cfg(not(feature = "mmap"))]
fn try_load_audio_encoder(_manifest: &Manifest) -> Option<Arc<AudioEncoderWeights>> {
    None
}

/// Eager vision mmproj GGUF loader for VL bundles. Returns the raw
/// parsed GGUF; `VisionEncoderWeights` are constructed separately
/// in [`from_parts`].
#[cfg(feature = "mmap")]
fn try_load_vision_encoder_gguf(manifest: &Manifest) -> Option<Arc<GgufFile>> {
    let path = manifest.files.multimodal_projector.as_ref()?;
    let p = Path::new(path);
    match GgufFile::open_arc(p) {
        Ok(g) => Some(g),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                path = %path,
                error = %format!("{e:#}"),
                "failed to open vision mmproj GGUF; skipping vision encoder"
            );
            None
        }
    }
}

#[cfg(not(feature = "mmap"))]
fn try_load_vision_encoder_gguf(_manifest: &Manifest) -> Option<Arc<GgufFile>> {
    None
}

/// Common initializer for DSpark draft models sharing base embeddings and LM head.
pub fn init_dspark_drafter(
    draft_gguf: Arc<GgufFile>,
    base_gguf: &Arc<GgufFile>,
    source_label: &str,
) -> Option<Arc<dyn crate::spec::Drafter>> {
    let base_embd_ref = match crate::model::transformer::resolve_weight(
        base_gguf,
        "token_embd.weight",
    ) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                error = %format!("{e:#}"),
                "failed to resolve token_embd.weight in base model for draft pairing ({source_label})"
            );
            return None;
        }
    };

    let base_output_ref =
        crate::model::transformer::resolve_weight(base_gguf, "output.weight").ok();

    match crate::model::dspark::DSparkDraftModel::from_gguf(
        draft_gguf,
        Arc::clone(base_gguf),
        base_embd_ref,
        base_output_ref,
    ) {
        Ok(dspark) => Some(Arc::new(dspark)),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                error = %format!("{e:#}"),
                "failed to initialize DSpark draft model from {source_label}; falling back to non-draft inference"
            );
            None
        }
    }
}

/// Eager draft model loader for speculative decoding (e.g. DSpark draft sidecar).
#[cfg(feature = "mmap")]
fn try_load_drafter(
    manifest: &Manifest,
    cfg: &EngineConfig,
    base_gguf: &Arc<GgufFile>,
) -> Option<Arc<dyn crate::spec::Drafter>> {
    let draft_path = cfg
        .draft_model
        .as_deref()
        .or_else(|| manifest.files.draft_model.as_deref().map(Path::new))?;

    if !draft_path.exists() {
        tracing::warn!(
            target: "cera::engine",
            path = %draft_path.display(),
            "draft model file does not exist; skipping draft model"
        );
        return None;
    }

    let draft_gguf = match GgufFile::open_arc(draft_path) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                path = %draft_path.display(),
                error = %format!("{e:#}"),
                "failed to open draft model GGUF; falling back to non-draft inference"
            );
            return None;
        }
    };

    init_dspark_drafter(draft_gguf, base_gguf, &draft_path.to_string_lossy())
}

#[cfg(not(feature = "mmap"))]
fn try_load_drafter(
    _manifest: &Manifest,
    _cfg: &EngineConfig,
    _base_gguf: &Arc<GgufFile>,
) -> Option<Arc<dyn crate::spec::Drafter>> {
    None
}
/// Parse the eagerly-mmapped mmproj GGUF into typed
/// `VisionEncoderWeights`. Failure is non-fatal — text-only chat
/// against a VL bundle still works without typed weights — so a
/// parse error logs a warn and returns `None` rather than failing
/// the engine load. Phase-2 forward pass code reads
/// `engine.vision_encoder()` and surfaces an explicit
/// "no vision encoder attached" error if `None`.
fn try_parse_vision_encoder(
    gguf: &Arc<GgufFile>,
    path: Option<&str>,
) -> Option<Arc<VisionEncoderWeights>> {
    match VisionEncoderWeights::from_gguf(gguf) {
        Ok(w) => Some(Arc::new(w)),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                path = %path.unwrap_or("<in-memory>"),
                error = %format!("{e:#}"),
                "vision mmproj parsed-into-weights step failed; image \
                 input will surface as 'no vision encoder attached' \
                 once that path lands. Text-only chat against this \
                 bundle still works."
            );
            None
        }
    }
}

/// Shared gate for the set of `InferenceType`s the engine can actually
/// load today. Returns `Ok(())` for text, LFM2-audio, and VL; returns
/// `CeraError::UnsupportedInferenceType` only for unrecognised arches.
/// Unconditional so both the mmap-backed path (pre-file-open) and the
/// in-memory paths (`from_bytes` / `from_reader`) use the same rule.
///
/// Note: VL bundles load the LLM half (plain LFM2) but image input
/// (`Session::append_image`) is not yet wired — phase 1 of the VL
/// pipeline only opens the gate and mmaps the mmproj GGUF. Calling
/// generate against a VL bundle today behaves as text-only chat.
fn check_inference_type_supported(it: &InferenceType) -> Result<(), CeraError> {
    match it {
        InferenceType::LlamaCppTextToText
        | InferenceType::LlamaCppLfm2AudioV1
        | InferenceType::LlamaCppImageToText => Ok(()),
        InferenceType::Unknown(s) => Err(CeraError::UnsupportedInferenceType(s.clone())),
    }
}

/// Peek at the GGUF header and guess an inference type. Minimal mapping
/// for v1 — only `lfm2` is actually loadable today; the other arches
/// are listed so auto-detect doesn't silently confuse a future non-text
/// model for text.
#[cfg(feature = "mmap")]
fn auto_detect_inference_type(model_path: &Path) -> Result<InferenceType, CeraError> {
    let gguf = GgufFile::open(model_path).map_err(|e| {
        CeraError::Backend(format!(
            "opening `{}` for inference-type auto-detect: {e}",
            model_path.display()
        ))
    })?;
    Ok(inference_type_for_arch(
        gguf.get_str("general.architecture").unwrap_or(""),
    ))
}

/// Decide a [`ModelBytes`] bundle's inference type. See
/// [`CeraEngine::from_parts`] for the reasoning, in particular why an mmproj
/// upgrades a text detection to VL. Pure, so the policy is testable without a
/// GGUF on hand.
///
/// `mmproj_parsed` means the sidecar was supplied *and* parsed. Passing "was
/// supplied" instead would let a corrupt buffer talk the engine into
/// advertising a modality it cannot serve.
fn resolve_parts_inference_type(
    explicit: Option<InferenceType>,
    arch: &str,
    mmproj_parsed: bool,
) -> InferenceType {
    if let Some(it) = explicit {
        return it;
    }
    let detected = inference_type_for_arch(arch);
    match (&detected, mmproj_parsed) {
        (InferenceType::LlamaCppTextToText, true) => InferenceType::LlamaCppImageToText,
        _ => detected,
    }
}

/// The arch → `InferenceType` mapping, split out of
/// [`auto_detect_inference_type`] so the in-memory constructors can reuse it.
/// They already hold a parsed [`GgufFile`] and must not touch the filesystem,
/// which is the only thing the mmap-gated wrapper adds.
fn inference_type_for_arch(arch: &str) -> InferenceType {
    match arch {
        "lfm2" | "lfm2moe" | "llama" | "qwen2" | "qwen3" => InferenceType::LlamaCppTextToText,
        "lfm2vl" => InferenceType::LlamaCppImageToText,
        "lfm2-audio" => InferenceType::LlamaCppLfm2AudioV1,
        // Unknown arch → assume text. Callers who need a different
        // mapping can set `ModelFiles::inference_type` explicitly.
        _ => InferenceType::LlamaCppTextToText,
    }
}

/// Dispatch the text-model loader on [`BackendPreference`]. Single source
/// of truth for "how to turn a `GgufFile` + a preference into a
/// `Box<dyn Model>`" — the CLI used to carry this logic.
fn load_text_model(
    gguf: GgufFile,
    path: Option<&Path>,
    cfg: &EngineConfig,
) -> Result<Box<dyn Model>, CeraError> {
    // Forward hook: fail fast with a clear error if a future build ever
    // requires an ISA feature the host lacks. Today every backend has a
    // runtime fallback (aarch64 NEON without dotprod, x86 scalar), so this is
    // a no-op — but it keeps the check wired at the load boundary.
    crate::backend::cpu_features::cpu_features()
        .ensure_supported()
        .map_err(CeraError::Backend)?;

    match cfg.backend {
        BackendPreference::Auto => load_text_model_auto(gguf, path, cfg.context_size),
        BackendPreference::Cpu => model::load_model(gguf, path, cfg.context_size)
            .map_err(|e| CeraError::Backend(format!("CPU model load failed: {e}"))),
        #[cfg(feature = "gpu")]
        BackendPreference::Gpu => model::load_model_gpu(gguf, path, cfg.context_size)
            .map_err(|e| CeraError::Backend(format!("GPU model load failed: {e}"))),
        #[cfg(not(feature = "gpu"))]
        BackendPreference::Gpu => Err(CeraError::Backend(
            "GPU backend not available (compile with --features gpu)".into(),
        )),
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        BackendPreference::Metal => {
            let p = path.ok_or_else(|| {
                CeraError::Backend("Metal backend requires a file path (not from_bytes)".into())
            })?;
            model::load_model_metal(gguf, p, cfg.context_size)
                .map_err(|e| CeraError::Backend(format!("Metal model load failed: {e}")))
        }
        #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
        BackendPreference::Metal => Err(CeraError::Backend(
            "Metal backend not available (compile with --features metal on macOS or iOS)".into(),
        )),
    }
}

fn load_text_model_auto(
    gguf: GgufFile,
    path: Option<&Path>,
    context_size: usize,
) -> Result<Box<dyn Model>, CeraError> {
    // Without a path (today: `from_bytes` only), Metal is unreachable
    // (it requires a file) and wgpu-then-CPU fallback can't re-open the
    // source. Short-circuit to CPU so `from_bytes` stays robust — this
    // matches the documented "testing / <50 MB" intent of that
    // constructor. Callers who want GPU with in-memory bytes must
    // opt in explicitly via `BackendPreference::Gpu`.
    if path.is_none() {
        tracing::debug!("cera::engine: no path available (from_bytes); using CPU backend (auto)");
        return model::load_model(gguf, None, context_size)
            .map_err(|e| CeraError::Backend(format!("CPU model load failed: {e}")));
    }

    // Metal → wgpu → CPU. Mirrors the CLI's previous `load_model_auto`.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    if let Some(p) = path {
        match model::load_model_metal(clone_gguf_like(&gguf, p)?, p, context_size) {
            Ok(m) => {
                tracing::debug!("cera::engine: using native Metal backend (auto)");
                return Ok(m);
            }
            Err(e) => {
                tracing::debug!("cera::engine: Metal unavailable ({e}); trying next backend");
            }
        }
    }

    // Auto-dispatch's gpu retry needs `mmap` to re-open the GGUF
    // between attempts. With gpu enabled but mmap disabled (e.g. the
    // wasm wgpu build), the auto path skips gpu and falls through to
    // CPU; users wanting gpu must opt in via `BackendPreference::Gpu`,
    // which doesn't need the re-open helper.
    #[cfg(all(feature = "gpu", feature = "mmap"))]
    {
        if let Some(p) = path {
            let gguf_for_gpu = clone_gguf_like(&gguf, p)?;
            match model::load_model_gpu(gguf_for_gpu, Some(p), context_size) {
                Ok(m) => {
                    tracing::debug!("cera::engine: using wgpu GPU backend (auto)");
                    return Ok(m);
                }
                Err(e) => {
                    tracing::debug!("cera::engine: wgpu unavailable ({e}); falling back to CPU");
                }
            }
            // Re-open the file for CPU — original `gguf` may have been
            // consumed by the Metal attempt above.
            let gguf_for_cpu = GgufFile::open(p).map_err(|e| {
                CeraError::Backend(format!("reopening `{}` for CPU fallback: {e}", p.display()))
            })?;
            model::load_model(gguf_for_cpu, Some(p), context_size)
                .map_err(|e| CeraError::Backend(format!("CPU model load failed: {e}")))
        } else {
            model::load_model(gguf, path, context_size)
                .map_err(|e| CeraError::Backend(format!("CPU model load failed: {e}")))
        }
    }

    #[cfg(not(all(feature = "gpu", feature = "mmap")))]
    {
        tracing::debug!("cera::engine: using CPU backend (auto)");
        model::load_model(gguf, path, context_size)
            .map_err(|e| CeraError::Backend(format!("CPU model load failed: {e}")))
    }
}

/// Re-open a GGUF from its path. The Metal and wgpu loaders consume
/// `GgufFile` by value, so the auto-dispatch path has to freshly map
/// the file for each backend it tries. Requires `mmap` (the only
/// supported re-open path); the explicit `BackendPreference::Gpu`
/// arm uses the caller-supplied `GgufFile` directly and doesn't need
/// this helper.
#[cfg(all(
    feature = "mmap",
    any(
        all(feature = "metal", any(target_os = "macos", target_os = "ios")),
        feature = "gpu"
    )
))]
fn clone_gguf_like(_: &GgufFile, path: &Path) -> Result<GgufFile, CeraError> {
    GgufFile::open(path)
        .map_err(|e| CeraError::Backend(format!("reopening `{}`: {e}", path.display())))
}

fn build_metadata(
    model: &dyn Model,
    tokenizer: &BpeTokenizer,
    manifest: &Manifest,
    quantization: String,
) -> ModelMetadata {
    let cfg = model.config();
    // Reflect the effective template availability: a manifest override
    // OR a GGUF-embedded template (the common case for bare `.gguf`
    // loads). Consumers asking `metadata().has_chat_template` expect a
    // truthful answer, not just "does the manifest have one".
    let has_chat_template = manifest.chat_template.is_some() || tokenizer.chat_template().is_some();
    ModelMetadata {
        architecture: cfg.architecture.clone(),
        max_seq_len: cfg.max_seq_len as u32,
        vocab_size: cfg.vocab_size as u32,
        has_chat_template,
        quantization,
        add_bos_token: tokenizer.add_bos_token(),
        add_eos_token: tokenizer.add_eos_token(),
    }
}

/// Map a GGUF `general.file_type` value (the llama.cpp `LLAMA_FTYPE_*`
/// enum) to the canonical short label used in filenames and tooling
/// (`Q4_0`, `Q4_K_M`, `BF16`, etc.). Falls back to `ftype:N` for
/// unrecognized values rather than dropping information — when a new
/// quantization scheme appears, the number itself is enough for a
/// human to look up. Returns `"unknown"` when the GGUF doesn't carry
/// the field at all.
///
/// List mirrors llama.cpp's enum as of early 2026; extend as new
/// quants ship upstream.
fn ftype_label(ftype: u32) -> String {
    match ftype {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        7 => "Q8_0".into(),
        8 => "Q5_0".into(),
        9 => "Q5_1".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K_S".into(),
        12 => "Q3_K_M".into(),
        13 => "Q3_K_L".into(),
        14 => "Q4_K_S".into(),
        15 => "Q4_K_M".into(),
        16 => "Q5_K_S".into(),
        17 => "Q5_K_M".into(),
        18 => "Q6_K".into(),
        19 => "IQ2_XXS".into(),
        20 => "IQ2_XS".into(),
        21 => "Q2_K_S".into(),
        22 => "IQ3_XS".into(),
        23 => "IQ3_XXS".into(),
        24 => "IQ1_S".into(),
        25 => "IQ4_NL".into(),
        26 => "IQ3_S".into(),
        27 => "IQ3_M".into(),
        28 => "IQ2_S".into(),
        29 => "IQ2_M".into(),
        30 => "IQ4_XS".into(),
        31 => "IQ1_M".into(),
        32 => "BF16".into(),
        36 => "TQ1_0".into(),
        37 => "TQ2_0".into(),
        other => format!("ftype:{other}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_preference_default_is_auto() {
        assert_eq!(BackendPreference::default(), BackendPreference::Auto);
    }

    #[test]
    fn backend_preference_parse_str_known_labels() {
        assert_eq!(
            BackendPreference::parse_str("auto").unwrap(),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse_str("").unwrap(),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse_str("CPU").unwrap(),
            BackendPreference::Cpu
        );
        assert_eq!(
            BackendPreference::parse_str("gpu").unwrap(),
            BackendPreference::Gpu
        );
        assert_eq!(
            BackendPreference::parse_str("wgpu").unwrap(),
            BackendPreference::Gpu
        );
        assert_eq!(
            BackendPreference::parse_str("Metal").unwrap(),
            BackendPreference::Metal
        );
        assert!(BackendPreference::parse_str("nvidia").is_err());
    }

    #[test]
    fn engine_config_default_is_4k_auto() {
        let c = EngineConfig::default();
        assert_eq!(c.context_size, 4096);
        assert_eq!(c.backend, BackendPreference::Auto);
    }

    /// URL / manifest-resolution helpers are `mmap`-gated; grouping their
    /// tests in one gated module means new ones inherit the gate instead of
    /// each needing its own attribute.
    #[cfg(feature = "mmap")]
    mod resolution {
        use super::*;

        #[test]
        fn is_remote_url_covers_http_https() {
            assert!(is_remote_url("http://example.com/x.gguf"));
            assert!(is_remote_url("HTTPS://example.com/x.gguf"));
            assert!(!is_remote_url("/local/path.gguf"));
            assert!(!is_remote_url("./rel/path.gguf"));
            assert!(!is_remote_url("file:///local/path.gguf"));
        }

        #[test]
        fn has_extension_case_insensitive() {
            assert!(has_extension(Path::new("foo.gguf"), "gguf"));
            assert!(has_extension(Path::new("foo.GGUF"), "gguf"));
            assert!(has_extension(Path::new("foo.json"), "json"));
            assert!(!has_extension(Path::new("foo.txt"), "gguf"));
            assert!(!has_extension(Path::new("foo"), "gguf"));
        }

        #[test]
        fn resolve_url_or_path_rejects_remote_without_repo() {
            let cfg = EngineConfig::default();
            let e = resolve_url_or_path("https://hf.co/x.gguf", None, &cfg)
                .expect_err("remote URL must error without a BundleRepo");
            let msg = format!("{e}");
            // Without the `remote` feature or with `bundle_repo = None`, the
            // error should steer the user toward the fix.
            assert!(
                msg.contains("remote URL"),
                "error should mention remote URL; got `{msg}`"
            );
            #[cfg(feature = "remote")]
            assert!(
                msg.contains("bundle_repo"),
                "error under `remote` feature should point at the config field; got `{msg}`"
            );
            #[cfg(not(feature = "remote"))]
            assert!(
                msg.contains("`remote` feature"),
                "error without `remote` feature should point at enabling it; got `{msg}`"
            );
        }

        #[test]
        fn resolve_url_or_path_rejects_file_scheme() {
            let cfg = EngineConfig::default();
            let e = resolve_url_or_path("file:///models/x.gguf", None, &cfg)
                .expect_err("file:// URIs aren't supported yet");
            let msg = format!("{e}");
            assert!(
                msg.contains("file://") && msg.contains("cera doesn't parse file URIs"),
                "error should point at the file:// limitation; got `{msg}`"
            );
        }

        #[test]
        fn strip_file_scheme_preserves_case() {
            assert_eq!(
                strip_file_scheme("FILE:///Models/Foo.gguf"),
                Some("/Models/Foo.gguf")
            );
            assert_eq!(strip_file_scheme("file://./rel"), Some("./rel"));
            assert_eq!(strip_file_scheme("https://x/y"), None);
            assert_eq!(strip_file_scheme("/abs/path"), None);
        }

        #[test]
        fn resolve_url_or_path_joins_relative_against_base() {
            let cfg = EngineConfig::default();
            let base = PathBuf::from("/models/bundles");
            let got = resolve_url_or_path("LFM2-1.2B-Q4_0.gguf", Some(&base), &cfg).unwrap();
            assert_eq!(got, PathBuf::from("/models/bundles/LFM2-1.2B-Q4_0.gguf"));
        }

        #[test]
        fn resolve_url_or_path_keeps_absolute_unchanged() {
            let cfg = EngineConfig::default();
            let base = PathBuf::from("/models/bundles");
            let got = resolve_url_or_path("/opt/foo.gguf", Some(&base), &cfg).unwrap();
            assert_eq!(got, PathBuf::from("/opt/foo.gguf"));
        }

        /// Regression guard: the resolver must touch every file field, not
        /// just `files.model`. Previously `resolve_primary_model_path` only
        /// handled the primary, silently leaving audio / VL / extras
        /// fields as raw URLs — which then broke downstream consumers.
        #[test]
        fn resolve_all_manifest_files_walks_every_field() {
            use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};

            let base = PathBuf::from("/models/bundles");
            let mut extras = std::collections::HashMap::new();
            extras.insert("cover_art".to_string(), "cover.png".to_string());
            extras.insert("config".to_string(), "/abs/config.toml".to_string());

            let mut manifest = Manifest {
                inference_type: InferenceType::LlamaCppLfm2AudioV1,
                schema_version: "1.0.0".to_string(),
                files: ManifestFiles {
                    model: "model.gguf".to_string(),
                    multimodal_projector: Some("mmproj.gguf".to_string()),
                    audio_decoder: Some("decoder.gguf".to_string()),
                    audio_tokenizer: Some("tokenizer.safetensors".to_string()),
                    draft_model: None,
                    extras,
                },
                chat_template: None,
                generation_defaults: GenerationDefaults::Other {
                    raw: serde_json::Value::Null,
                },
                raw: serde_json::Value::Null,
            };

            let cfg = EngineConfig::default();
            resolve_all_manifest_files(&mut manifest, Some(&base), &cfg).unwrap();

            assert_eq!(manifest.files.model, "/models/bundles/model.gguf");
            assert_eq!(
                manifest.files.multimodal_projector.as_deref(),
                Some("/models/bundles/mmproj.gguf")
            );
            assert_eq!(
                manifest.files.audio_decoder.as_deref(),
                Some("/models/bundles/decoder.gguf")
            );
            assert_eq!(
                manifest.files.audio_tokenizer.as_deref(),
                Some("/models/bundles/tokenizer.safetensors")
            );
            assert_eq!(
                manifest.files.extras.get("cover_art").map(String::as_str),
                Some("/models/bundles/cover.png")
            );
            // Absolute extras stay absolute.
            assert_eq!(
                manifest.files.extras.get("config").map(String::as_str),
                Some("/abs/config.toml")
            );
        }

        #[test]
        fn resolve_all_manifest_files_none_optionals_stay_none() {
            use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};
            let mut manifest = Manifest {
                inference_type: InferenceType::LlamaCppTextToText,
                schema_version: "1.0.0".to_string(),
                files: ManifestFiles {
                    model: "/abs/model.gguf".to_string(),
                    multimodal_projector: None,
                    audio_decoder: None,
                    audio_tokenizer: None,
                    draft_model: None,
                    extras: std::collections::HashMap::new(),
                },
                chat_template: None,
                generation_defaults: GenerationDefaults::Other {
                    raw: serde_json::Value::Null,
                },
                raw: serde_json::Value::Null,
            };
            let cfg = EngineConfig::default();
            resolve_all_manifest_files(&mut manifest, None, &cfg).unwrap();
            assert!(manifest.files.multimodal_projector.is_none());
            assert!(manifest.files.audio_decoder.is_none());
            assert!(manifest.files.audio_tokenizer.is_none());
        }

        #[test]
        fn find_single_manifest_zero_and_many() {
            let dir = tempfile::tempdir().unwrap();
            let e0 = find_single_manifest(dir.path()).expect_err("empty dir must error");
            assert!(format!("{e0}").contains("no .json manifest"));

            std::fs::write(dir.path().join("a.json"), b"{}").unwrap();
            let got = find_single_manifest(dir.path()).unwrap();
            assert_eq!(got.file_name().unwrap(), "a.json");

            std::fs::write(dir.path().join("b.json"), b"{}").unwrap();
            let e2 =
                find_single_manifest(dir.path()).expect_err("two manifests must error (ambiguous)");
            let msg = format!("{e2}");
            assert!(msg.contains("2 .json manifests"), "{msg}");
            assert!(msg.contains("a.json") && msg.contains("b.json"), "{msg}");
        }

        #[test]
        fn synthesize_manifest_from_files_preserves_aux() {
            let files = ModelFiles {
                model: PathBuf::from("/m/model.gguf"),
                multimodal_projector: Some(PathBuf::from("/m/mmproj.gguf")),
                audio_decoder: Some(PathBuf::from("/m/ad.gguf")),
                audio_tokenizer: Some(PathBuf::from("/m/at.safetensors")),
                draft_model: None,
                extras: std::collections::HashMap::new(),
                inference_type: Some(InferenceType::LlamaCppLfm2AudioV1),
                chat_template: None,
            };
            let m = synthesize_manifest_from_files(&files).unwrap();
            assert_eq!(m.inference_type, InferenceType::LlamaCppLfm2AudioV1);
            assert_eq!(m.files.model, "/m/model.gguf");
            assert_eq!(
                m.files.multimodal_projector.as_deref(),
                Some("/m/mmproj.gguf")
            );
            assert_eq!(m.files.audio_decoder.as_deref(), Some("/m/ad.gguf"));
            assert_eq!(
                m.files.audio_tokenizer.as_deref(),
                Some("/m/at.safetensors")
            );
            assert!(matches!(
                m.generation_defaults,
                crate::manifest::GenerationDefaults::Audio { .. }
            ));
        }
    }

    #[test]
    fn model_files_text_helper_is_text_only() {
        let f = ModelFiles::text("/x/y.gguf");
        assert_eq!(f.model, PathBuf::from("/x/y.gguf"));
        assert!(f.multimodal_projector.is_none());
        assert_eq!(f.inference_type, Some(InferenceType::LlamaCppTextToText));
    }

    #[test]
    fn model_bytes_text_helper_is_text_only() {
        let b = ModelBytes::text(vec![0u8; 4]);
        assert!(b.multimodal_projector.is_none());
        assert_eq!(b.inference_type, Some(InferenceType::LlamaCppTextToText));
    }

    /// The load-bearing case. Every published LFM2-VL bundle reports
    /// `architecture = "lfm2"`, indistinguishable from a text model, so
    /// trusting auto-detect alone would make the mmproj argument silently
    /// inert and `append_image` fail with "no vision encoder attached" on a
    /// correctly-supplied bundle.
    #[test]
    fn mmproj_upgrades_a_text_arch_to_vl() {
        assert_eq!(
            resolve_parts_inference_type(None, "lfm2", true),
            InferenceType::LlamaCppImageToText
        );
    }

    /// `false` here means "no usable mmproj", covering both the absent case
    /// and the supplied-but-unparseable one. The latter is the important
    /// one: upgrading on a corrupt sidecar would leave `capabilities()`
    /// advertising an `image_in` that `append_image` immediately refuses.
    #[test]
    fn no_usable_mmproj_leaves_a_text_arch_as_text() {
        assert_eq!(
            resolve_parts_inference_type(None, "lfm2", false),
            InferenceType::LlamaCppTextToText
        );
    }

    /// The upgrade is text-only. An audio arch already identifies itself, so
    /// an mmproj must not push it onto the vision path.
    #[test]
    fn mmproj_does_not_retype_an_audio_arch() {
        assert_eq!(
            resolve_parts_inference_type(None, "lfm2-audio", true),
            InferenceType::LlamaCppLfm2AudioV1
        );
    }

    /// An explicit type is honored as given, which is how a caller opts out
    /// of the upgrade and asks for text plus an ignored sidecar.
    #[test]
    fn explicit_inference_type_overrides_the_mmproj_upgrade() {
        assert_eq!(
            resolve_parts_inference_type(Some(InferenceType::LlamaCppTextToText), "lfm2", true),
            InferenceType::LlamaCppTextToText
        );
    }

    /// An unknown arch *is* upgraded by a usable mmproj, same as a known
    /// text one. The arch is not what makes a bundle multimodal; a real
    /// vision encoder sitting next to it is.
    #[test]
    fn unknown_arch_is_upgraded_by_a_usable_mmproj() {
        // `inference_type_for_arch` maps unrecognised arches to text, so
        // without this upgrade a VL bundle on an arch cera does not know
        // by name would load text-only and drop its vision tower in
        // silence. Pinned because it is the whole reason the upgrade keys
        // off the sidecar rather than off the arch string.
        assert_eq!(
            resolve_parts_inference_type(None, "totally-made-up", true),
            InferenceType::LlamaCppImageToText
        );
    }

    #[test]
    fn synthetic_manifest_records_the_mmproj_and_type() {
        let m = Manifest::synthetic(
            Path::new("<bytes>"),
            InferenceType::LlamaCppImageToText,
            Some("<mmproj-bytes>".to_string()),
        );
        assert_eq!(m.inference_type, InferenceType::LlamaCppImageToText);
        assert_eq!(
            m.files.multimodal_projector.as_deref(),
            Some("<mmproj-bytes>")
        );
        assert_eq!(
            m.raw["inference_type"].as_str(),
            Some("llama.cpp/image-to-text")
        );
    }

    /// `synthetic_text` must keep producing exactly what it did before it was
    /// re-expressed on top of `synthetic`.
    #[test]
    fn synthetic_text_is_unchanged_by_the_generalization() {
        let m = Manifest::synthetic_text(Path::new("/m/model.gguf"));
        assert_eq!(m.inference_type, InferenceType::LlamaCppTextToText);
        assert!(m.files.multimodal_projector.is_none());
        assert_eq!(m.files.model, "/m/model.gguf");
        assert_eq!(
            m.raw["inference_type"].as_str(),
            Some("llama.cpp/text-to-text")
        );
        assert_eq!(
            m.raw["load_time_parameters"]["model"].as_str(),
            Some("/m/model.gguf")
        );
    }
}

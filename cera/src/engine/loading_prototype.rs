//! Explicit model loading, shared generative models and structured load errors.
//!
//! Re-exported through `cera` and `cera::engine`. Bytes/readers reuse the engine's
//! assembly point; multipart and filesystem input retain their parsed primary.
//! The internal filename preserves the loading contract regression fixtures.

use super::{AuxWeights, CeraEngine, EngineConfig, GgufFile, Manifest, ModelBytes};
use crate::{CeraError, Session, SessionConfig};
use std::io::Read;
use std::path::Path;
#[cfg(feature = "mmap")]
use std::path::PathBuf;
use std::sync::Arc;

mod retained;

/// Engine loading options, with the same defaults and feature gates as [`EngineConfig`].
/// Existing `EngineConfig` values can be passed directly to [`ModelLoader::config`].
pub type LoadConfig = EngineConfig;

/// Explicit model input. Construction performs no loading or network requests.
///
/// Filesystem forms require `mmap`; remote forms additionally require `remote`.
/// A reader may borrow data and need not implement `Send`, `Seek`, or `Clone`.
/// There is no implicit string conversion between a path and a remote repository.
pub enum ModelSource<'a> {
    /// A Leap bundle ID and quantization, resolved using the configured repository.
    #[cfg(all(feature = "remote", feature = "mmap"))]
    BundleId { id: String, quant: String },
    /// A Hugging Face spec with optional quantization and conversion strategy overrides.
    #[cfg(all(feature = "remote", feature = "mmap"))]
    HuggingFace {
        spec: String,
        quant: Option<String>,
        strategy: Option<String>,
    },
    /// A GGUF file, manifest file, or model directory.
    #[cfg(feature = "mmap")]
    Path(PathBuf),
    /// Primary and auxiliary model files, including inference/template metadata.
    #[cfg(feature = "mmap")]
    Files(super::ModelFiles),
    /// A primary GGUF retained in shared memory.
    Bytes(Arc<[u8]>),
    /// A synchronously consumed GGUF byte stream.
    Reader(Box<dyn Read + 'a>),
    /// Primary and auxiliary model bytes, including generation defaults.
    Parts(ModelBytes),
}

impl<'a> ModelSource<'a> {
    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn bundle_id(id: impl Into<String>, quant: impl Into<String>) -> Self {
        Self::BundleId {
            id: id.into(),
            quant: quant.into(),
        }
    }

    #[cfg(all(feature = "remote", feature = "mmap"))]
    pub fn hugging_face(
        spec: impl Into<String>,
        quant: Option<&str>,
        strategy: Option<&str>,
    ) -> Self {
        Self::HuggingFace {
            spec: spec.into(),
            quant: quant.map(str::to_owned),
            strategy: strategy.map(str::to_owned),
        }
    }

    #[cfg(feature = "mmap")]
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    #[cfg(feature = "mmap")]
    pub fn files(files: super::ModelFiles) -> Self {
        Self::Files(files)
    }

    pub fn bytes(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self::Bytes(bytes.into())
    }

    /// Retain a reader until build; buffer it once without seeking or rewinding.
    pub fn reader(reader: impl Read + 'a) -> Self {
        Self::Reader(Box::new(reader))
    }

    pub fn parts(parts: ModelBytes) -> Self {
        Self::Parts(parts)
    }
}

/// Model family recognized before generative backend/weight assembly.
/// Recognition does not imply that a typed loader for every family is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelKind {
    Generative,
    Encoder,
    Whisper,
    Vad,
    Hotword,
}

/// Failure while resolving, classifying or assembling a model.
/// Session operations continue to return [`CeraError`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
    #[error("expected {expected:?}, found {actual:?} (architecture {architecture:?})")]
    KindMismatch {
        expected: ModelKind,
        actual: ModelKind,
        architecture: String,
    },
    #[error("unsupported architecture {architecture:?}")]
    UnsupportedArchitecture { architecture: String },
    #[error("{error}")]
    Source {
        source_kind: &'static str,
        #[source]
        error: CeraError,
    },
    #[error("{error}")]
    Assembly {
        backend: crate::BackendPreference,
        #[source]
        error: CeraError,
    },
    #[error(transparent)]
    Engine(#[from] CeraError),
}

impl LoadError {
    #[cfg(feature = "mmap")]
    fn with_source(self, source_kind: &'static str) -> Self {
        match self {
            Self::Engine(error) => Self::Source { source_kind, error },
            other => other,
        }
    }
}

/// A loaded model with a dynamic kind. Additional variants require implemented facades.
#[non_exhaustive]
pub enum ModelHandle {
    Generative(GenerativeModel),
}

impl ModelHandle {
    /// Kind of this successfully loaded model.
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Generative(_) => ModelKind::Generative,
        }
    }

    /// Share the generative model, if present. It can outlive this handle.
    pub fn as_generative(&self) -> Option<GenerativeModel> {
        match self {
            Self::Generative(model) => Some(model.clone()),
        }
    }
}

/// Shared ownership of a loaded generative engine and its weights.
///
/// Cloning this value or obtaining its engine does not reload the source.
/// Sessions retain their engine resources independently; backend-specific shared
/// execution-state restrictions still apply, as for [`CeraEngine::new_session`].
#[derive(Clone)]
pub struct GenerativeModel {
    engine: Arc<CeraEngine>,
}

impl GenerativeModel {
    /// Create an existing core session with the supplied configuration.
    /// Dropping this model does not invalidate the session or its KV state.
    pub fn create_session(&self, config: SessionConfig) -> Result<Session, CeraError> {
        self.engine.new_session(config)
    }
}

/// Consuming builder for explicit model sources.
///
/// Both build methods resolve and classify the source before generative
/// assembly. Existing standalone encoder, Whisper, VAD and hotword loaders
/// remain available; this builder currently loads generative models only.
pub struct ModelLoader<'a> {
    source: ModelSource<'a>,
    config: LoadConfig,
}

impl<'a> ModelLoader<'a> {
    /// Start with the existing engine configuration defaults.
    pub fn new(source: ModelSource<'a>) -> Self {
        Self {
            source,
            config: LoadConfig::default(),
        }
    }

    /// Replace loading options, preserving all existing engine feature fields.
    pub fn config(mut self, config: LoadConfig) -> Self {
        self.config = config;
        self
    }

    /// Load a dynamic handle. Currently supports the generative variant.
    pub fn build(self) -> Result<ModelHandle, LoadError> {
        self.build_generative().map(ModelHandle::Generative)
    }

    /// Load a generative model, rejecting other recognized kinds before assembly.
    pub fn build_generative(self) -> Result<GenerativeModel, LoadError> {
        let backend = self.config.backend;
        let engine = match self.source {
            #[cfg(all(feature = "remote", feature = "mmap"))]
            ModelSource::BundleId { id, quant } => {
                let source = super::resolve_bundle_source(&id, &quant, &self.config)
                    .map_err(|error| LoadError::Engine(error).with_source("bundle"))?;
                Self::assemble_path(source, self.config, "bundle")?
            }
            #[cfg(all(feature = "remote", feature = "mmap"))]
            ModelSource::HuggingFace {
                spec,
                quant,
                strategy,
            } => {
                let source = super::resolve_hf_source(
                    &spec,
                    quant.as_deref(),
                    strategy.as_deref(),
                    &self.config,
                )
                .map_err(|error| LoadError::Engine(error).with_source("hf"))?;
                Self::assemble_path(source, self.config, "hf")?
            }
            #[cfg(feature = "mmap")]
            ModelSource::Path(path) => {
                let source =
                    super::resolve_path_source_checked(&path, &self.config, require_generative)
                        .map_err(|error| error.with_source("path"))?;
                Self::assemble_path(source, self.config, "path")?
            }
            #[cfg(feature = "mmap")]
            ModelSource::Files(files) => {
                let source =
                    super::resolve_files_source_checked(&files, &self.config, require_generative)
                        .map_err(|error| error.with_source("files"))?;
                Self::assemble_path(source, self.config, "files")?
            }
            ModelSource::Bytes(bytes) => {
                let gguf = GgufFile::from_bytes(bytes).map_err(|e| LoadError::Source {
                    source_kind: "bytes",
                    error: CeraError::Backend(format!("parsing GGUF bytes: {e}")),
                })?;
                require_generative(&gguf)?;
                CeraEngine::from_gguf(
                    gguf,
                    Manifest::synthetic_text(Path::new("<bytes>")),
                    self.config,
                    None,
                    AuxWeights::default(),
                )
                .map_err(|error| LoadError::Assembly { backend, error })?
            }
            ModelSource::Reader(reader) => {
                let gguf = GgufFile::from_reader(reader).map_err(|e| LoadError::Source {
                    source_kind: "reader",
                    error: CeraError::Backend(format!("reading GGUF stream: {e}")),
                })?;
                require_generative(&gguf)?;
                CeraEngine::from_gguf(
                    gguf,
                    Manifest::synthetic_text(Path::new("<reader>")),
                    self.config,
                    None,
                    AuxWeights::default(),
                )
                .map_err(|error| LoadError::Assembly { backend, error })?
            }
            ModelSource::Parts(parts) => {
                let gguf = GgufFile::from_bytes(Arc::clone(&parts.model)).map_err(|e| {
                    LoadError::Source {
                        source_kind: "parts",
                        error: CeraError::Backend(format!("parsing GGUF bytes: {e}")),
                    }
                })?;
                require_generative(&gguf)?;
                CeraEngine::from_parts_with_primary(gguf, parts, self.config)
                    .map_err(|error| LoadError::Assembly { backend, error })?
            }
        };
        Ok(GenerativeModel {
            engine: Arc::new(engine),
        })
    }

    #[cfg(feature = "mmap")]
    fn assemble_path(
        mut source: super::ResolvedPathSource,
        config: LoadConfig,
        source_kind: &'static str,
    ) -> Result<CeraEngine, LoadError> {
        let gguf = source
            .open_primary()
            .map_err(|error| LoadError::Source { source_kind, error })?;
        require_generative(&gguf)?;
        let backend = config.backend;
        CeraEngine::from_gguf(
            gguf,
            source.manifest,
            config,
            Some(&source.primary),
            AuxWeights::default(),
        )
        .map_err(|error| LoadError::Assembly { backend, error })
    }
}

fn require_generative(gguf: &GgufFile) -> Result<(), LoadError> {
    let architecture = gguf.get_str("general.architecture").unwrap_or("");
    let actual = if crate::is_whisper_gguf(gguf) {
        ModelKind::Whisper
    } else {
        match architecture {
            "lfm2" | "lfm2moe" | "llama" | "qwen2" | "qwen3" | "granite" => ModelKind::Generative,
            "bert" | "modernbert" => ModelKind::Encoder,
            "silero_vad" => ModelKind::Vad,
            "kws" => ModelKind::Hotword,
            _ => {
                return Err(LoadError::UnsupportedArchitecture {
                    architecture: architecture.into(),
                });
            }
        }
    };
    if actual != ModelKind::Generative {
        return Err(LoadError::KindMismatch {
            expected: ModelKind::Generative,
            actual,
            architecture: architecture.into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[path = "auxiliary/audio/fixture.rs"]
    mod audio_fixture;
    mod auxiliary;
    #[path = "auxiliary/fixture.rs"]
    mod companion_fixture;
    #[cfg(feature = "mmap")]
    mod filesystem;
    mod ownership;
    mod parsing;
    #[cfg(all(feature = "remote", feature = "mmap"))]
    mod remote;

    use super::*;
    use crate::convert::writer::{GGML_TYPE_F32, GgufWriter};
    use crate::manifest::{GenerationDefaults, InferenceType};
    use crate::{BackendPreference, GenerateOpts, ModalitySink};
    use std::io::{self, Cursor};

    fn cpu_config() -> LoadConfig {
        LoadConfig {
            backend: BackendPreference::Cpu,
            context_size: 32,
            ..LoadConfig::default()
        }
    }

    fn header(architecture: &str) -> Arc<[u8]> {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", architecture);
        let mut bytes = Vec::new();
        writer.write_header_and_tensor_info(&mut bytes).unwrap();
        bytes.into()
    }

    // One attention block with deterministic F32 weights: exercises actual
    // assembly and generation without a network download or a model cache.
    fn tiny_llama() -> Arc<[u8]> {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "llama");
        writer.add_string("general.name", "loading-contract-fixture");
        writer.add_string("tokenizer.chat_template", "embedded template");
        writer.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
        for (key, value) in [
            ("block_count", 1),
            ("embedding_length", 32),
            ("feed_forward_length", 32),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("context_length", 64),
            ("vocab_size", 2),
        ] {
            writer.add_u32(format!("llama.{key}"), value);
        }
        let mut payloads = Vec::new();
        for (name, dims) in [
            ("token_embd.weight", vec![32, 2]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![32, 32]),
            ("blk.0.attn_v.weight", vec![32, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![32, 32]),
            ("blk.0.ffn_up.weight", vec![32, 32]),
            ("blk.0.ffn_down.weight", vec![32, 32]),
        ] {
            let count: u64 = dims.iter().product();
            let data: Vec<u8> = (0..count)
                .flat_map(|i| {
                    let value = if name.contains("norm") {
                        1.0_f32
                    } else {
                        ((i % 17) as f32 - 8.0) / 32.0
                    };
                    value.to_le_bytes()
                })
                .collect();
            writer.add_tensor(name, dims, GGML_TYPE_F32, data.len());
            payloads.push(data);
        }
        let mut bytes = Vec::new();
        writer.write_header_and_tensor_info(&mut bytes).unwrap();
        for data in payloads {
            writer.write_tensor_data(&mut bytes, &data).unwrap();
        }
        bytes.into()
    }

    #[track_caller]
    fn error(result: Result<GenerativeModel, LoadError>) -> LoadError {
        match result {
            Ok(_) => panic!("expected load failure"),
            Err(error) => error,
        }
    }

    #[test]
    fn wrong_kind_fails_before_tokenizer_or_backend_assembly_for_every_memory_source() {
        for (arch, expected) in [
            ("bert", ModelKind::Encoder),
            ("modernbert", ModelKind::Encoder),
            ("whisper", ModelKind::Whisper),
            ("silero_vad", ModelKind::Vad),
            ("kws", ModelKind::Hotword),
        ] {
            let bytes = header(arch);
            // These headers deliberately lack tokenizer data and weights. A
            // late kind check would instead produce a tokenizer/backend error.
            for source in [
                ModelSource::bytes(bytes.clone()),
                ModelSource::reader(Cursor::new(bytes.clone())),
                ModelSource::parts(ModelBytes::text(bytes.clone())),
            ] {
                assert!(matches!(
                    error(ModelLoader::new(source).build_generative()),
                    LoadError::KindMismatch { actual, architecture, .. }
                        if actual == expected && architecture == arch
                ));
            }
        }
    }

    #[test]
    fn unknown_architecture_is_not_assumed_generative() {
        for arch in ["", "future-model"] {
            assert!(matches!(
                error(ModelLoader::new(ModelSource::bytes(header(arch))).build_generative()),
                LoadError::UnsupportedArchitecture { architecture } if architecture == arch
            ));
        }
    }

    #[test]
    fn whisper_fallback_metadata_uses_existing_detection() {
        let mut gguf = GgufFile::from_bytes(header("")).unwrap();
        gguf.metadata.insert(
            "whisper.audio.embedding_length".into(),
            crate::gguf::GgufValue::U32(384),
        );
        assert!(matches!(
            require_generative(&gguf),
            Err(LoadError::KindMismatch {
                actual: ModelKind::Whisper,
                ..
            })
        ));
    }

    #[test]
    fn malformed_input_and_reader_failure_remain_load_errors() {
        struct BrokenReader;
        impl Read for BrokenReader {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("fixture read failure"))
            }
        }
        for (kind, source) in [
            ("bytes", ModelSource::bytes(Vec::from(&b"invalid"[..]))),
            ("reader", ModelSource::reader(BrokenReader)),
            (
                "parts",
                ModelSource::parts(ModelBytes::text(Vec::from(&b"invalid"[..]))),
            ),
        ] {
            assert!(matches!(
                error(ModelLoader::new(source).build_generative()),
                LoadError::Source { source_kind, error: CeraError::Backend(_) } if source_kind == kind
            ));
        }
    }

    #[test]
    fn valid_primary_header_without_weights_reports_assembly_backend() {
        for source in [
            ModelSource::bytes(header("llama")),
            ModelSource::reader(Cursor::new(header("llama"))),
            ModelSource::parts(ModelBytes::text(header("llama"))),
        ] {
            let failure = error(
                ModelLoader::new(source)
                    .config(cpu_config())
                    .build_generative(),
            );
            assert!(std::error::Error::source(&failure).is_some());
            assert!(matches!(
                failure,
                LoadError::Assembly {
                    backend: BackendPreference::Cpu,
                    ..
                }
            ));
        }
    }

    #[derive(Default)]
    struct Tokens(Vec<u32>, usize);
    impl ModalitySink for Tokens {
        fn on_text_tokens(&mut self, tokens: &[u32]) {
            self.0.extend_from_slice(tokens);
        }

        fn on_done(&mut self, _: crate::FinishReason) {
            self.1 += 1;
        }
    }

    fn generate(session: &mut Session) -> Vec<u32> {
        session.append_tokens(&[0, 1]).unwrap();
        let mut sink = Tokens::default();
        let summary = session
            .generate(
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                    ..GenerateOpts::default()
                },
                &mut sink,
            )
            .unwrap();
        assert_eq!(summary.tokens_generated, 2);
        assert_eq!(sink.1, 1);
        sink.0
    }

    #[test]
    fn typed_and_dynamic_loading_preserve_tokens_and_session_lifetime() {
        let bytes = tiny_llama();
        let legacy = CeraEngine::from_bytes(bytes.clone(), cpu_config()).unwrap();
        let expected = generate(&mut legacy.new_session(SessionConfig::default()).unwrap());
        for source in [
            ModelSource::bytes(bytes.clone()),
            ModelSource::reader(Cursor::new(bytes.as_ref())),
            ModelSource::parts(ModelBytes::text(bytes.clone())),
        ] {
            let handle = ModelLoader::new(source)
                .config(cpu_config())
                .build()
                .unwrap();
            assert_eq!(handle.kind(), ModelKind::Generative);
            let model = handle.as_generative().unwrap();
            drop(handle);
            let mut session = model.create_session(SessionConfig::default()).unwrap();
            drop(model);
            assert_eq!(generate(&mut session), expected);
        }
    }

    #[test]
    fn bytes_share_weight_storage_and_parts_preserve_overrides() {
        let bytes = tiny_llama();
        let mut parts = ModelBytes::text(bytes.clone());
        parts.inference_type = Some(InferenceType::LlamaCppTextToText);
        parts.chat_template = Some("caller template".into());
        parts.multimodal_projector = Some(Arc::from(&b"ignored invalid projector"[..]));
        parts.generation_defaults = Some(GenerationDefaults::Text {
            temperature: Some(0.3),
            min_p: Some(0.15),
            top_p: Some(0.9),
            top_k: Some(7),
            repetition_penalty: Some(1.1),
        });
        let legacy = CeraEngine::from_parts(parts.clone(), cpu_config()).unwrap();
        let model = ModelLoader::new(ModelSource::parts(parts))
            .config(cpu_config())
            .build_generative()
            .unwrap();
        assert_eq!(
            model.engine.manifest().chat_template.as_deref(),
            Some("caller template")
        );
        assert_eq!(model.engine.manifest().raw, legacy.manifest().raw);
        assert_eq!(model.engine.default_generate_opts().temperature, 0.3);
        assert_eq!(model.engine.default_generate_opts().top_k, 7);
        assert_eq!(model.engine.config().context_size, 32);
        assert!(!model.engine.capabilities().image_in);
        drop(legacy);
        assert!(Arc::strong_count(&bytes) > 1);
        drop(model);
        assert_eq!(Arc::strong_count(&bytes), 1);
    }
}

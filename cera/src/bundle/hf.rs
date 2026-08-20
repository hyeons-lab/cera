//! Hugging Face model repository resolution and auxiliary file pairing.
//!
//! Extends Cera's bundle infrastructure beyond `LiquidAI/LeapBundles` to any
//! Hugging Face repository or file URL.
//!
//! Two main layers:
//!
//! 1. **Pure spec & classification logic** (always compiled):
//!    - [`HfSpec`]: Parses full URLs (`https://huggingface.co/...`) and repo IDs (`owner/repo[:quant][@rev]`).
//!    - [`classify_repo_siblings`]: Groups repo files into primary GGUFs, vision mmproj files, audio decoders, and tokenizers.
//!    - [`resolve_hf_manifest`]: Matches the requested or default quantization, auto-pairs modality aux files, and produces a dynamic [`Manifest`].
//!
//! 2. **Network inspection client** (gated under `#[cfg(feature = "remote")]`):
//!    - [`fetch_model_info`]: Queries `https://huggingface.co/api/models/{owner}/{repo}` with optional `HF_TOKEN` authentication.
//!    - [`inspect_and_resolve_manifest`]: End-to-end resolution from a spec string to a loadable [`Manifest`].

use std::collections::HashMap;

use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};
use crate::session::CeraError;

#[cfg(feature = "remote")]
use reqwest::blocking::Client;
#[cfg(feature = "remote")]
use std::time::Duration;

/// Canonical Hugging Face API base URL.
pub const HF_API_BASE: &str = "https://huggingface.co/api/models";

/// HTTP timeout for Hugging Face metadata requests (30s).
#[cfg(feature = "remote")]
const HF_API_TIMEOUT: Duration = Duration::from_secs(30);

/// Parsed representation of a Hugging Face model spec or URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSpec {
    pub owner: String,
    pub repo: String,
    pub revision: String,
    pub subpath: Option<String>,
    pub quant: Option<String>,
}

/// Helper to obtain the active Hugging Face base endpoint (default: `https://huggingface.co`).
pub fn hf_base_endpoint() -> String {
    let base =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    base.trim_end_matches('/').to_string()
}

/// Returns true if the host is a recognized Hugging Face domain or matches `HF_ENDPOINT`.
pub fn is_hf_or_endpoint_host(host_str: &str) -> bool {
    let host_lower = host_str.to_ascii_lowercase();
    let host_clean = host_lower.split(':').next().unwrap_or(&host_lower);
    if matches!(
        host_clean,
        "huggingface.co" | "www.huggingface.co" | "hf.co" | "www.hf.co"
    ) {
        return true;
    }
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        let ep_lower = endpoint.to_ascii_lowercase();
        let ep_after = ep_lower
            .strip_prefix("https://")
            .or_else(|| ep_lower.strip_prefix("http://"))
            .unwrap_or(&ep_lower);
        let ep_host = ep_after.split(['/', ':', '?', '#']).next().unwrap_or("");
        if !ep_host.is_empty() && host_clean == ep_host {
            return true;
        }
    }
    false
}

impl HfSpec {
    /// Parse a Hugging Face URL or repository identifier into a structured [`HfSpec`].
    ///
    /// Accepts:
    /// - Web tree URLs: `https://huggingface.co/LiquidAI/LFM2.5-VL-3B/tree/main`
    /// - Model URLs: `https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF`
    /// - Direct file URLs: `https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF/resolve/main/LFM2.5-VL-3B-Q4_K_M.gguf`
    /// - Repo IDs: `LiquidAI/LFM2.5-VL-3B-GGUF`
    /// - Repo IDs with quant: `LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M`
    /// - Repo IDs with revision: `LiquidAI/LFM2.5-VL-3B-GGUF@v1.0`
    /// - Combined: `LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M@main` or `@main:Q4_K_M`
    pub fn parse(input: &str) -> Result<Self, CeraError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(CeraError::Backend(
                "Hugging Face model spec cannot be empty".into(),
            ));
        }

        let lower = trimmed.to_ascii_lowercase();
        // Case 1: Full HTTP/HTTPS URL
        if lower.starts_with("http://") || lower.starts_with("https://") {
            return Self::parse_url(trimmed);
        }

        // Case 2: Repo identifier: `owner/repo[:quant][@rev]` or `owner/repo[@rev][:quant]`
        Self::parse_repo_id(trimmed)
    }

    fn parse_url(url: &str) -> Result<Self, CeraError> {
        let after_scheme = if let Some(idx) = url.find("://") {
            &url[idx + 3..]
        } else {
            return Err(CeraError::Backend(format!("invalid URL `{url}`")));
        };

        let (host, path) = after_scheme
            .split_once('/')
            .ok_or_else(|| CeraError::Backend(format!("URL `{url}` missing path components")))?;

        if !is_hf_or_endpoint_host(host) {
            return Err(CeraError::Backend(format!(
                "URL host `{host}` is not a recognized Hugging Face host or configured HF_ENDPOINT"
            )));
        }

        let path_no_qs = path
            .split(['?', '#'])
            .next()
            .unwrap_or("")
            .trim_matches('/');
        let segments: Vec<&str> = path_no_qs.split('/').filter(|s| !s.is_empty()).collect();

        if segments.len() < 2 {
            return Err(CeraError::Backend(format!(
                "URL `{url}` does not contain valid owner/repo path"
            )));
        }

        let owner = segments[0];
        let repo = segments[1];
        crate::bundle::cache_key::validate_path_segment("owner", owner)?;
        crate::bundle::cache_key::validate_path_segment("repo", repo)?;

        let mut revision = "main".to_string();
        let mut subpath = None;
        let mut quant = None;

        if segments.len() >= 4 && matches!(segments[2], "tree" | "resolve" | "blob" | "raw") {
            revision = segments[3].to_string();
            crate::bundle::cache_key::validate_path_segment("revision", &revision)?;

            if segments.len() > 4 {
                let file_parts = &segments[4..];
                for part in file_parts {
                    crate::bundle::cache_key::validate_path_segment("file segment", part)?;
                }
                let file_str = file_parts.join("/");
                if file_str.to_ascii_lowercase().ends_with(".gguf") {
                    quant = extract_quant_from_filename(&file_str);
                }
                subpath = Some(file_str);
            }
        }

        Ok(Self {
            owner: owner.to_string(),
            repo: repo.to_string(),
            revision,
            subpath,
            quant,
        })
    }

    fn parse_repo_id(input: &str) -> Result<Self, CeraError> {
        let (repo_part, quant, revision) = extract_quant_and_rev(input);

        let parts: Vec<&str> = repo_part.split('/').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(CeraError::Backend(format!(
                "invalid Hugging Face model identifier `{input}`; expected `owner/repo` (e.g. `LiquidAI/LFM2.5-VL-3B-GGUF`)"
            )));
        }

        let owner = parts[0];
        let repo = parts[1];
        crate::bundle::cache_key::validate_path_segment("owner", owner)?;
        crate::bundle::cache_key::validate_path_segment("repo", repo)?;

        let rev = if let Some(r) = revision {
            crate::bundle::cache_key::validate_path_segment("revision", &r)?;
            r
        } else {
            "main".to_string()
        };

        if let Some(q) = &quant {
            crate::bundle::cache_key::validate_path_segment("quant", q)?;
        }

        Ok(Self {
            owner: owner.to_string(),
            repo: repo.to_string(),
            revision: rev,
            subpath: None,
            quant,
        })
    }

    /// Build the API model-info URL for this repository.
    pub fn api_url(&self) -> String {
        let base = hf_base_endpoint();
        if self.revision != "main" {
            format!(
                "{base}/api/models/{}/{}?revision={}",
                self.owner, self.repo, self.revision
            )
        } else {
            format!("{base}/api/models/{}/{}", self.owner, self.repo)
        }
    }

    /// Build the raw download URL for a file in this repository.
    pub fn file_download_url(&self, rfilename: &str) -> String {
        let base = hf_base_endpoint();
        format!(
            "{base}/{}/{}/resolve/{}/{}",
            self.owner, self.repo, self.revision, rfilename
        )
    }
}

/// Helper to split `owner/repo:quant@rev` or `owner/repo@rev:quant`.
fn extract_quant_and_rev(input: &str) -> (&str, Option<String>, Option<String>) {
    let mut quant = None;
    let mut rev = None;
    let mut base = input;

    // Check for `@revision`
    if let Some((b, r)) = base.split_once('@') {
        base = b;
        if let Some((r_part, q)) = r.split_once(':') {
            rev = Some(r_part.to_string());
            quant = Some(q.to_string());
        } else {
            rev = Some(r.to_string());
        }
    }

    // Check for `:quant` if not already extracted
    if quant.is_none()
        && let Some((b, q)) = base.split_once(':')
    {
        base = b;
        if let Some((q_part, r)) = q.split_once('@') {
            quant = Some(q_part.to_string());
            if rev.is_none() {
                rev = Some(r.to_string());
            }
        } else {
            quant = Some(q.to_string());
        }
    }

    (base, quant, rev)
}

/// Sibling file in a Hugging Face repository model-info payload.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HfSibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
}

/// Summary of a Hugging Face model repository returned by the model-info endpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HfModelInfo {
    pub id: String,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

/// Individual GGUF file entry discovered in a repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFileEntry {
    pub rfilename: String,
    pub quant: String,
    pub size_bytes: Option<u64>,
}

/// Classified files within a Hugging Face model repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRepoContents {
    pub primary_ggufs: Vec<GgufFileEntry>,
    pub mmproj_ggufs: Vec<GgufFileEntry>,
    pub audio_decoders: Vec<GgufFileEntry>,
    pub vocoder_ggufs: Vec<GgufFileEntry>,
    pub tokenizer_ggufs: Vec<GgufFileEntry>,
    pub audio_tokenizers: Vec<String>,
    pub safetensors_files: Vec<String>,
    pub has_safetensors: bool,
    pub generation_config: Option<String>,
    pub tokenizer_config: Option<String>,
    pub chat_template_jinja: Option<String>,
}

/// Classify repository files from a model-info payload into primary GGUFs and modality aux files.
pub fn classify_repo_siblings(siblings: &[HfSibling]) -> HfRepoContents {
    let mut primary_ggufs = Vec::new();
    let mut mmproj_ggufs = Vec::new();
    let mut audio_decoders = Vec::new();
    let mut vocoder_ggufs = Vec::new();
    let mut tokenizer_ggufs = Vec::new();
    let mut audio_tokenizers = Vec::new();
    let mut safetensors_files = Vec::new();
    let mut generation_config = None;
    let mut tokenizer_config = None;
    let mut chat_template_jinja = None;

    for sib in siblings {
        let name = &sib.rfilename;
        let lower = name.to_ascii_lowercase();

        let base_file = lower.rsplit(['/', '\\']).next().unwrap_or(&lower);

        if lower.ends_with(".gguf") {
            let quant = extract_quant_from_filename(name).unwrap_or_else(|| "unknown".to_string());
            let entry = GgufFileEntry {
                rfilename: name.clone(),
                quant,
                size_bytes: sib.size,
            };

            if is_vision_mmproj(base_file) {
                mmproj_ggufs.push(entry);
            } else if is_audio_decoder(base_file) {
                audio_decoders.push(entry);
            } else if is_vocoder(base_file) {
                vocoder_ggufs.push(entry);
            } else if is_audio_tokenizer_gguf(base_file) {
                tokenizer_ggufs.push(entry);
            } else {
                primary_ggufs.push(entry);
            }
        } else if lower.ends_with(".safetensors") {
            safetensors_files.push(name.clone());
            if lower.contains("audiotokenizer") || lower.contains("tokenizer.safetensors") {
                audio_tokenizers.push(name.clone());
            }
        } else if lower == "generation_config.json" {
            generation_config = Some(name.clone());
        } else if lower == "tokenizer_config.json" {
            tokenizer_config = Some(name.clone());
        } else if lower == "chat_template.jinja" {
            chat_template_jinja = Some(name.clone());
        }
    }

    let has_safetensors = !safetensors_files.is_empty();

    HfRepoContents {
        primary_ggufs,
        mmproj_ggufs,
        audio_decoders,
        vocoder_ggufs,
        tokenizer_ggufs,
        audio_tokenizers,
        safetensors_files,
        has_safetensors,
        generation_config,
        tokenizer_config,
        chat_template_jinja,
    }
}

fn is_vision_mmproj(file: &str) -> bool {
    !is_audio_decoder(file)
        && !file.contains("audio")
        && (file.starts_with("mmproj")
            || file.contains("-mmproj")
            || file.contains("_mmproj")
            || file.contains("projector"))
}

fn is_audio_decoder(file: &str) -> bool {
    file.contains("audio_decoder")
        || file.contains("audio-decoder")
        || file.contains("audiodecoder")
}

fn is_vocoder(file: &str) -> bool {
    file.starts_with("vocoder") || file.contains("-vocoder") || file.contains("_vocoder")
}

fn is_audio_tokenizer_gguf(file: &str) -> bool {
    file.starts_with("tokenizer")
}

/// Check whether two quantization tags match (case-insensitive and hyphen/underscore/dot agnostic, zero-alloc).
#[inline]
pub(crate) fn quant_matches(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes().zip(b.bytes()).all(|(b1, b2)| {
        let c1 = if b1 == b'_' || b1 == b'.' {
            b'-'
        } else {
            b1.to_ascii_uppercase()
        };
        let c2 = if b2 == b'_' || b2 == b'.' {
            b'-'
        } else {
            b2.to_ascii_uppercase()
        };
        c1 == c2
    })
}

/// Extract quantization string (e.g. `Q4_K_M`, `Q4_0`, `Q8_0`, `F16`, `BF16`, `QAD-Q4_0`, `QAD-Q4_K_M`) from a filename.
pub fn extract_quant_from_filename(filename: &str) -> Option<String> {
    let base = filename.strip_suffix(".gguf").unwrap_or(filename);
    let name = base.rsplit(['/', '\\']).next().unwrap_or(base);

    const BASE_QUANTS: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q4_1", "Q5_K_M", "Q5_K_S", "Q5_K", "Q5_0", "Q5_1",
        "Q6_K", "Q8_0", "Q8_1", "Q8_K", "Q2_K", "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q3_K", "F16",
        "BF16", "F32",
    ];

    let upper = name.to_ascii_uppercase();
    let bytes = upper.as_bytes();

    // Pass 1: Prioritize explicit QAD variants (e.g. QAD-Q4_0, QAD_Q4_K_M, QAD.Q8_0)
    for &q in BASE_QUANTS {
        for (pos, _) in upper.match_indices(q) {
            let end = pos + q.len();
            let after_ok = end == upper.len() || matches!(bytes[end], b'-' | b'_' | b'.' | b'/');
            if !after_ok {
                continue;
            }

            // Check for immediate anchored QAD prefix: [-_./]QAD[-_.] or ^QAD[-_.]
            let has_immediate_qad = if pos >= 4
                && (&bytes[pos - 4..pos] == b"QAD-"
                    || &bytes[pos - 4..pos] == b"QAD_"
                    || &bytes[pos - 4..pos] == b"QAD.")
            {
                let qad_start = pos - 4;
                qad_start == 0 || matches!(bytes[qad_start - 1], b'-' | b'_' | b'.' | b'/')
            } else {
                false
            };

            if has_immediate_qad {
                return Some(format!("QAD-{q}"));
            }
        }
    }

    // Pass 2: Standard base quantization tags
    for &q in BASE_QUANTS {
        for (pos, _) in upper.match_indices(q) {
            let end = pos + q.len();
            let after_ok = end == upper.len() || matches!(bytes[end], b'-' | b'_' | b'.' | b'/');
            if !after_ok {
                continue;
            }

            let before_ok = pos == 0 || matches!(bytes[pos - 1], b'-' | b'_' | b'.' | b'/');
            if before_ok {
                return Some(q.to_string());
            }
        }
    }

    None
}

/// Preference ranking for automatic quantization selection when none is requested.
const QUANT_PREFERENCE_ORDER: &[&str] = &[
    "QAD-Q4_K_M",
    "QAD-Q4_0",
    "Q4_K_M",
    "Q4_0",
    "QAD-Q5_K_M",
    "Q5_K_M",
    "QAD-Q8_0",
    "Q8_0",
    "QAD-Q6_K",
    "Q6_K",
    "Q5_0",
    "Q4_1",
    "F16",
    "BF16",
];

/// Resolve a [`HfModelInfo`] payload to a fully populated [`Manifest`] with download URLs.
pub fn resolve_hf_manifest(
    spec: &HfSpec,
    info: &HfModelInfo,
    requested_quant: Option<&str>,
    generation_defaults: Option<GenerationDefaults>,
) -> Result<Manifest, CeraError> {
    let contents = classify_repo_siblings(&info.siblings);

    if contents.primary_ggufs.is_empty() {
        if contents.has_safetensors {
            return Err(CeraError::Backend(format!(
                "repository `{}/{}` has no primary .gguf files (found SafeTensors weights). Use streaming quantization.",
                spec.owner, spec.repo
            )));
        }
        return Err(CeraError::Backend(format!(
            "repository `{}/{}` has no primary .gguf model files",
            spec.owner, spec.repo
        )));
    }

    // 1. Select Primary Model GGUF
    let primary_entry = if let Some(ref sub) = spec.subpath {
        contents
            .primary_ggufs
            .iter()
            .find(|e| {
                e.rfilename == *sub
                    || e.rfilename.strip_suffix(sub).is_some_and(|prefix| {
                        prefix.is_empty() || prefix.ends_with('/') || prefix.ends_with('\\')
                    })
            })
            .ok_or_else(|| {
                CeraError::Backend(format!(
                    "requested file `{sub}` not found in `{}/{}`",
                    spec.owner, spec.repo
                ))
            })?
    } else {
        match requested_quant {
            Some(q) => {
                let norm_q = q.trim();
                contents
                    .primary_ggufs
                    .iter()
                    .find(|e| quant_matches(&e.quant, norm_q))
                    .ok_or_else(|| {
                        let available: Vec<_> = contents
                            .primary_ggufs
                            .iter()
                            .map(|e| e.quant.as_str())
                            .collect();
                        CeraError::Backend(format!(
                            "requested quant `{q}` not found in `{}/{}`. Available quants: {:?}",
                            spec.owner, spec.repo, available
                        ))
                    })?
            }
            None => QUANT_PREFERENCE_ORDER
                .iter()
                .find_map(|&pref| {
                    contents
                        .primary_ggufs
                        .iter()
                        .find(|e| quant_matches(&e.quant, pref))
                })
                .unwrap_or(&contents.primary_ggufs[0]),
        }
    };

    let model_url = spec.file_download_url(&primary_entry.rfilename);

    // 2. Modality Aux Pairing
    let mut multimodal_projector = None;
    let mut audio_decoder = None;
    let mut audio_tokenizer = None;

    // Vision matching
    let is_vl_pipeline = info.pipeline_tag.as_deref() == Some("image-text-to-text")
        || info
            .tags
            .iter()
            .any(|t| t == "image-text-to-text" || t == "lfm2-vl" || t == "lfm2.5-vl")
        || !contents.mmproj_ggufs.is_empty();

    if is_vl_pipeline && !contents.mmproj_ggufs.is_empty() {
        // Match mmproj by preference: 1. same quant as primary, 2. Q8_0, 3. F16/BF16, 4. first
        let mmproj_chosen = contents
            .mmproj_ggufs
            .iter()
            .find(|e| quant_matches(&e.quant, &primary_entry.quant))
            .or_else(|| {
                contents
                    .mmproj_ggufs
                    .iter()
                    .find(|e| quant_matches(&e.quant, "Q8_0"))
            })
            .or_else(|| {
                contents
                    .mmproj_ggufs
                    .iter()
                    .find(|e| quant_matches(&e.quant, "F16") || quant_matches(&e.quant, "BF16"))
            })
            .unwrap_or(&contents.mmproj_ggufs[0]);

        multimodal_projector = Some(spec.file_download_url(&mmproj_chosen.rfilename));
    }

    // Audio matching
    let is_audio_pipeline = info.pipeline_tag.as_deref() == Some("automatic-speech-recognition")
        || info
            .tags
            .iter()
            .any(|t| t.contains("audio") || t == "lfm2-audio")
        || !contents.audio_decoders.is_empty()
        || !contents.vocoder_ggufs.is_empty();

    let mut extras = HashMap::new();

    if is_audio_pipeline {
        if let Some(dec) = contents.audio_decoders.first() {
            audio_decoder = Some(spec.file_download_url(&dec.rfilename));
        }
        if let Some(tok_gguf) = contents.tokenizer_ggufs.first() {
            audio_tokenizer = Some(spec.file_download_url(&tok_gguf.rfilename));
        } else if let Some(tok) = contents.audio_tokenizers.first() {
            audio_tokenizer = Some(spec.file_download_url(tok));
        }
        if let Some(voc) = contents.vocoder_ggufs.first() {
            extras.insert(
                "vocoder".to_string(),
                spec.file_download_url(&voc.rfilename),
            );
        }
    }

    // Determine InferenceType
    let inference_type = if is_audio_pipeline || audio_decoder.is_some() {
        InferenceType::LlamaCppLfm2AudioV1
    } else if multimodal_projector.is_some() || is_vl_pipeline {
        InferenceType::LlamaCppImageToText
    } else {
        InferenceType::LlamaCppTextToText
    };

    let files = ManifestFiles {
        model: model_url,
        multimodal_projector,
        audio_decoder,
        audio_tokenizer,
        extras,
    };

    let defaults = match (inference_type.clone(), generation_defaults) {
        (
            InferenceType::LlamaCppLfm2AudioV1,
            Some(GenerationDefaults::Text {
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
            }),
        ) => GenerationDefaults::Audio {
            number_of_decoding_threads: None,
            audio_temperature: None,
            audio_top_k: None,
            temperature,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
        },
        (InferenceType::LlamaCppLfm2AudioV1, Some(audio @ GenerationDefaults::Audio { .. })) => {
            audio
        }
        (InferenceType::LlamaCppLfm2AudioV1, _) => GenerationDefaults::Audio {
            number_of_decoding_threads: None,
            audio_temperature: None,
            audio_top_k: None,
            temperature: None,
            min_p: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        },
        (
            InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText,
            Some(GenerationDefaults::Audio {
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
                ..
            }),
        ) => GenerationDefaults::Text {
            temperature,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
        },
        (
            InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText,
            Some(text @ GenerationDefaults::Text { .. }),
        ) => text,
        (InferenceType::LlamaCppTextToText | InferenceType::LlamaCppImageToText, _) => {
            GenerationDefaults::Text {
                temperature: None,
                min_p: None,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
            }
        }
        (_, Some(defaults)) => defaults,
        (_, None) => GenerationDefaults::Other {
            raw: serde_json::Value::Null,
        },
    };

    let mut raw_map = serde_json::Map::new();
    raw_map.insert(
        "inference_type".into(),
        serde_json::Value::String(inference_type.as_str().to_string()),
    );
    raw_map.insert(
        "schema_version".into(),
        serde_json::Value::String("1.0.0".into()),
    );

    Ok(Manifest {
        inference_type,
        schema_version: "1.0.0".into(),
        files,
        chat_template: None,
        generation_defaults: defaults,
        raw: serde_json::Value::Object(raw_map),
    })
}

/// Fetch model repository info from the Hugging Face API.
#[cfg(feature = "remote")]
pub fn fetch_model_info(spec: &HfSpec) -> Result<HfModelInfo, CeraError> {
    let client = Client::builder()
        .timeout(HF_API_TIMEOUT)
        .build()
        .map_err(|e| CeraError::Backend(format!("failed to build HTTP client for HF API: {e}")))?;

    let url = spec.api_url();
    let mut last_error = String::new();

    for attempt in 0..5 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(200 * (1 << (attempt - 1))));
        }

        let mut req = client.get(&url);
        if let Some(token) = get_hf_auth_token() {
            req = req.header("Authorization", format!("Bearer {token}"));
        }

        match req.send() {
            Ok(resp) => {
                let status = resp.status();
                if status == reqwest::StatusCode::NOT_FOUND {
                    return Err(CeraError::Backend(format!(
                        "Hugging Face repository `{}/{}` not found (404)",
                        spec.owner, spec.repo
                    )));
                }
                if status == reqwest::StatusCode::UNAUTHORIZED
                    || status == reqwest::StatusCode::FORBIDDEN
                {
                    return Err(CeraError::Backend(format!(
                        "Hugging Face repository `{}/{}` requires authentication (401/403). Set the `HF_TOKEN` environment variable.",
                        spec.owner, spec.repo
                    )));
                }
                if status.is_success() {
                    let body = resp.text().map_err(|e| {
                        CeraError::Backend(format!("failed to read HF API response body: {e}"))
                    })?;
                    return serde_json::from_str::<HfModelInfo>(&body).map_err(|e| {
                        CeraError::Backend(format!("failed to parse HF API model info JSON: {e}"))
                    });
                }
                if status.is_server_error()
                    || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status == reqwest::StatusCode::REQUEST_TIMEOUT
                {
                    last_error = format!("HTTP {status} from HF API");
                    continue;
                }
                return Err(CeraError::Backend(format!(
                    "Hugging Face API error for `{url}`: HTTP status {status}"
                )));
            }
            Err(e) => {
                last_error = format!("connection error for `{url}`: {e}");
            }
        }
    }

    Err(CeraError::Backend(format!(
        "failed to query HF API for `{url}` after retries: {last_error}"
    )))
}

#[cfg(feature = "remote")]
pub fn fetch_generation_defaults(spec: &HfSpec) -> Option<GenerationDefaults> {
    let client = Client::builder().timeout(HF_API_TIMEOUT).build().ok()?;
    let url = spec.file_download_url("generation_config.json");
    let token = get_hf_auth_token();

    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500 * (1 << (attempt - 1))));
        }

        let mut req = client.get(&url);
        if let Some(ref t) = token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }

        if let Ok(resp) = req.send() {
            let status = resp.status();
            if status.is_success() {
                if let Ok(body) = resp.text() {
                    return parse_generation_config_json(&body);
                }
            } else if status == reqwest::StatusCode::NOT_FOUND {
                return None;
            } else if status.is_server_error()
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status == reqwest::StatusCode::REQUEST_TIMEOUT
            {
                continue;
            } else {
                return None;
            }
        }
    }

    None
}

/// Parse a raw `generation_config.json` string into [`GenerationDefaults`].
pub fn parse_generation_config_json(body: &str) -> Option<GenerationDefaults> {
    let val: serde_json::Value = serde_json::from_str(body).ok()?;
    let obj = val.as_object()?;

    let temperature = obj
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let top_p = obj.get("top_p").and_then(|v| v.as_f64()).map(|f| f as f32);
    let top_k = obj
        .get("top_k")
        .and_then(|v| v.as_u64())
        .and_then(|u| u32::try_from(u).ok());
    let repetition_penalty = obj
        .get("repetition_penalty")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let min_p = obj.get("min_p").and_then(|v| v.as_f64()).map(|f| f as f32);

    Some(GenerationDefaults::Text {
        temperature,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
    })
}

/// Attempt to read Hugging Face token from environment variables or local token cache.
fn read_token_file(p: impl AsRef<std::path::Path>) -> Option<String> {
    std::fs::read_to_string(p).ok().and_then(|c| {
        let trimmed = c.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

/// Helper to obtain the default caching directory (`~/.cache/cera` or `%USERPROFILE%/.cache/cera`).
pub fn default_cache_dir() -> std::path::PathBuf {
    let base = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join(".cache").join("cera")
}

/// Retrieve Hugging Face authentication token from environment or local cache files.
pub fn get_hf_auth_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN")
        && !t.trim().is_empty()
    {
        return Some(t.trim().to_string());
    }
    if let Ok(t) = std::env::var("HUGGING_FACE_HUB_TOKEN")
        && !t.trim().is_empty()
    {
        return Some(t.trim().to_string());
    }
    if let Ok(hf_home) = std::env::var("HF_HOME")
        && let Some(token) = read_token_file(std::path::Path::new(&hf_home).join("token"))
    {
        return Some(token);
    }
    let home_dir = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from);
    if let Some(home) = home_dir
        && let Some(token) = read_token_file(home.join(".cache").join("huggingface").join("token"))
    {
        return Some(token);
    }
    None
}

/// End-to-end inspect and resolve a Hugging Face model spec or URL to a [`Manifest`].
#[cfg(feature = "remote")]
pub fn inspect_and_resolve_manifest(
    spec_or_url: &str,
    quant: Option<&str>,
    quant_strategy: Option<&str>,
    cache_dir: Option<&std::path::Path>,
    progress: Option<std::sync::Arc<dyn crate::bundle::DownloadProgress>>,
) -> Result<Manifest, CeraError> {
    let spec = HfSpec::parse(spec_or_url)?;
    let info = fetch_model_info(&spec)?;
    let contents = classify_repo_siblings(&info.siblings);

    let requested_quant = quant.or(spec.quant.as_deref());
    if contents.primary_ggufs.is_empty() && contents.has_safetensors {
        let target_quant = match requested_quant {
            Some(q) => crate::convert::TargetQuant::parse_str(q).ok_or_else(|| {
                CeraError::Backend(format!(
                    "unsupported quantization format `{q}`. Supported formats: Q4_K_M, Q5_K_M, Q6_K, Q8_0, Q4_0, F16, F32"
                ))
            })?,
            None => crate::convert::TargetQuant::Q4_K_M,
        };

        let strategy = quant_strategy
            .and_then(crate::convert::QuantStrategy::parse_str)
            .unwrap_or(crate::convert::QuantStrategy::Auto);

        let base_cache = cache_dir
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(default_cache_dir);

        let opts = crate::convert::QuantizeOptions {
            target_quant,
            strategy,
            cache_dir: base_cache,
            auth_token: get_hf_auth_token(),
            progress,
            cancel: None,
        };

        return crate::convert::stream_quantize_hf_repo(&spec, opts);
    }

    let gen_defaults = if contents.generation_config.is_some() {
        fetch_generation_defaults(&spec)
    } else {
        None
    };
    resolve_hf_manifest(&spec, &info, requested_quant, gen_defaults)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_web_url() {
        let spec = HfSpec::parse("https://huggingface.co/LiquidAI/LFM2.5-VL-3B/tree/main").unwrap();
        assert_eq!(spec.owner, "LiquidAI");
        assert_eq!(spec.repo, "LFM2.5-VL-3B");
        assert_eq!(spec.revision, "main");
        assert_eq!(spec.subpath, None);
    }

    #[test]
    fn parse_direct_file_url() {
        let spec = HfSpec::parse(
            "https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF/resolve/main/LFM2.5-VL-3B-Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(spec.owner, "LiquidAI");
        assert_eq!(spec.repo, "LFM2.5-VL-3B-GGUF");
        assert_eq!(spec.revision, "main");
        assert_eq!(spec.subpath.as_deref(), Some("LFM2.5-VL-3B-Q4_K_M.gguf"));
        assert_eq!(spec.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parse_repo_id_plain() {
        let spec = HfSpec::parse("LiquidAI/LFM2.5-VL-3B-GGUF").unwrap();
        assert_eq!(spec.owner, "LiquidAI");
        assert_eq!(spec.repo, "LFM2.5-VL-3B-GGUF");
        assert_eq!(spec.revision, "main");
        assert_eq!(spec.quant, None);
    }

    #[test]
    fn parse_repo_id_with_quant() {
        let spec = HfSpec::parse("LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M").unwrap();
        assert_eq!(spec.owner, "LiquidAI");
        assert_eq!(spec.repo, "LFM2.5-VL-3B-GGUF");
        assert_eq!(spec.revision, "main");
        assert_eq!(spec.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parse_repo_id_with_rev_and_quant() {
        let spec = HfSpec::parse("LiquidAI/LFM2.5-VL-3B-GGUF@v1.2:Q8_0").unwrap();
        assert_eq!(spec.owner, "LiquidAI");
        assert_eq!(spec.repo, "LFM2.5-VL-3B-GGUF");
        assert_eq!(spec.revision, "v1.2");
        assert_eq!(spec.quant.as_deref(), Some("Q8_0"));
    }

    #[test]
    fn classify_and_pair_vl_model() {
        let siblings = vec![
            HfSibling {
                rfilename: ".gitattributes".into(),
                size: None,
            },
            HfSibling {
                rfilename: "README.md".into(),
                size: None,
            },
            HfSibling {
                rfilename: "LFM2.5-VL-3B-Q4_K_M.gguf".into(),
                size: Some(1950000000),
            },
            HfSibling {
                rfilename: "LFM2.5-VL-3B-Q8_0.gguf".into(),
                size: Some(3200000000),
            },
            HfSibling {
                rfilename: "mmproj-LFM2.5-VL-3B-Q8_0.gguf".into(),
                size: Some(450000000),
            },
        ];

        let contents = classify_repo_siblings(&siblings);
        assert_eq!(contents.primary_ggufs.len(), 2);
        assert_eq!(contents.mmproj_ggufs.len(), 1);

        let spec = HfSpec::parse("LiquidAI/LFM2.5-VL-3B-GGUF").unwrap();
        let info = HfModelInfo {
            id: "LiquidAI/LFM2.5-VL-3B-GGUF".into(),
            siblings,
            tags: vec!["image-text-to-text".into()],
            pipeline_tag: Some("image-text-to-text".into()),
            config: None,
        };

        let manifest = resolve_hf_manifest(&spec, &info, Some("Q4_K_M"), None).unwrap();
        assert_eq!(manifest.inference_type, InferenceType::LlamaCppImageToText);
        assert!(manifest.files.model.contains("LFM2.5-VL-3B-Q4_K_M.gguf"));
        assert!(
            manifest
                .files
                .multimodal_projector
                .as_deref()
                .unwrap()
                .contains("mmproj-LFM2.5-VL-3B-Q8_0.gguf")
        );
    }

    #[test]
    fn parse_generation_config_sampling_params() {
        let json = r#"{
            "temperature": 0.2,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "min_p": 0.15
        }"#;
        let defaults = parse_generation_config_json(json).unwrap();
        match defaults {
            GenerationDefaults::Text {
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
            } => {
                assert_eq!(temperature, Some(0.2));
                assert_eq!(top_k, Some(50));
                assert_eq!(top_p, Some(0.9));
                assert_eq!(repetition_penalty, Some(1.05));
                assert_eq!(min_p, Some(0.15));
            }
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn test_hf_endpoint_mirror_urls() {
        let spec = HfSpec::parse("LiquidAI/LFM2.5-VL-3B-GGUF").unwrap();
        assert_eq!(
            spec.api_url(),
            "https://huggingface.co/api/models/LiquidAI/LFM2.5-VL-3B-GGUF"
        );
        assert_eq!(
            spec.file_download_url("model.gguf"),
            "https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF/resolve/main/model.gguf"
        );

        // Test with explicit revision
        let spec_rev = HfSpec::parse("LiquidAI/LFM2.5-VL-3B-GGUF@v1.0").unwrap();
        assert_eq!(
            spec_rev.api_url(),
            "https://huggingface.co/api/models/LiquidAI/LFM2.5-VL-3B-GGUF?revision=v1.0"
        );
    }

    #[test]
    fn test_qad_quant_extraction_and_resolution() {
        assert_eq!(
            extract_quant_from_filename("LFM2.5-2.6B-QAD-Q4_0.gguf").as_deref(),
            Some("QAD-Q4_0")
        );
        assert_eq!(
            extract_quant_from_filename("LFM2.5-2.6B-QAD_Q4_0.gguf").as_deref(),
            Some("QAD-Q4_0")
        );
        assert_eq!(
            extract_quant_from_filename("LFM2.5-2.6B.QAD.Q4_0.gguf").as_deref(),
            Some("QAD-Q4_0")
        );
        assert_eq!(
            extract_quant_from_filename("LFM2.5-2.6B-QAD-Q4_K_M.gguf").as_deref(),
            Some("QAD-Q4_K_M")
        );
        assert_eq!(
            extract_quant_from_filename("LFM2.5-2.6B-Q4_0.gguf").as_deref(),
            Some("Q4_0")
        );
        assert_eq!(
            extract_quant_from_filename("QADNet-model-Q4_0.gguf").as_deref(),
            Some("Q4_0")
        );

        assert_eq!(
            extract_quant_from_filename("LFM2.5-Q4_0-QAD-Q4_0.gguf").as_deref(),
            Some("QAD-Q4_0")
        );

        assert!(quant_matches("QAD-Q4_0", "qad_q4_0"));
        assert!(quant_matches("QAD_Q4_0", "QAD-Q4-0"));
        assert!(quant_matches("QAD-Q4_0", "QAD.Q4_0"));
        assert!(quant_matches("QAD.Q4_0", "qad_q4_0"));
        assert!(quant_matches("Q4_K_M", "q4-k-m"));
        assert!(!quant_matches("QAD-Q4_0", "Q4_0"));

        let siblings = vec![
            HfSibling {
                rfilename: "LFM2.5-2.6B-Q4_0.gguf".into(),
                size: Some(1500000000),
            },
            HfSibling {
                rfilename: "LFM2.5-2.6B-QAD-Q4_0.gguf".into(),
                size: Some(1500000000),
            },
            HfSibling {
                rfilename: "LFM2.5-2.6B-Q8_0.gguf".into(),
                size: Some(2800000000),
            },
        ];

        let spec = HfSpec::parse("LiquidAI/LFM2.5-2.6B-GGUF").unwrap();
        let info = HfModelInfo {
            id: "LiquidAI/LFM2.5-2.6B-GGUF".into(),
            siblings,
            tags: vec!["text-generation".into()],
            pipeline_tag: Some("text-generation".into()),
            config: None,
        };

        // 1. Explicit QAD-Q4_0 request
        let manifest_qad = resolve_hf_manifest(&spec, &info, Some("QAD-Q4_0"), None).unwrap();
        assert!(
            manifest_qad
                .files
                .model
                .contains("LFM2.5-2.6B-QAD-Q4_0.gguf")
        );

        // 2. Explicit QAD_Q4_0 underscore request
        let manifest_qad_underscore =
            resolve_hf_manifest(&spec, &info, Some("qad_q4_0"), None).unwrap();
        assert!(
            manifest_qad_underscore
                .files
                .model
                .contains("LFM2.5-2.6B-QAD-Q4_0.gguf")
        );

        // 3. Explicit standard Q4_0 request
        let manifest_ptq = resolve_hf_manifest(&spec, &info, Some("Q4_0"), None).unwrap();
        assert!(manifest_ptq.files.model.contains("LFM2.5-2.6B-Q4_0.gguf"));

        // 4. Default preference should prefer QAD-Q4_0 over standard Q4_0
        let manifest_default = resolve_hf_manifest(&spec, &info, None, None).unwrap();
        assert!(
            manifest_default
                .files
                .model
                .contains("LFM2.5-2.6B-QAD-Q4_0.gguf")
        );
    }

    #[test]
    fn test_audio_hf_manifest_normalizes_text_generation_config_to_audio() {
        let siblings = vec![
            HfSibling {
                rfilename: "LFM2.5-Audio-Q4_0.gguf".into(),
                size: Some(1500000000),
            },
            HfSibling {
                rfilename: "audio_decoder.gguf".into(),
                size: Some(300000000),
            },
        ];

        let spec = HfSpec::parse("LiquidAI/LFM2.5-Audio-GGUF").unwrap();
        let info = HfModelInfo {
            id: "LiquidAI/LFM2.5-Audio-GGUF".into(),
            siblings,
            tags: vec!["audio".into(), "text-to-speech".into()],
            pipeline_tag: Some("text-to-speech".into()),
            config: None,
        };

        // GenerationDefaults parsed from generation_config.json is GenerationDefaults::Text
        let gen_config_text = GenerationDefaults::Text {
            temperature: Some(0.3),
            min_p: Some(0.05),
            top_p: Some(0.95),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
        };

        let manifest = resolve_hf_manifest(&spec, &info, None, Some(gen_config_text)).unwrap();
        assert_eq!(manifest.inference_type, InferenceType::LlamaCppLfm2AudioV1);
        match manifest.generation_defaults {
            GenerationDefaults::Audio {
                number_of_decoding_threads,
                audio_temperature,
                audio_top_k,
                temperature,
                min_p,
                top_p,
                top_k,
                repetition_penalty,
            } => {
                assert_eq!(number_of_decoding_threads, None);
                assert_eq!(audio_temperature, None);
                assert_eq!(audio_top_k, None);
                assert_eq!(temperature, Some(0.3));
                assert_eq!(min_p, Some(0.05));
                assert_eq!(top_p, Some(0.95));
                assert_eq!(top_k, Some(40));
                assert_eq!(repetition_penalty, Some(1.1));
            }
            other => panic!("expected GenerationDefaults::Audio, got {other:?}"),
        }
    }
}

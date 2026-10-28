//! Streaming SafeTensors to GGUF quantization pipeline with HTTP retries & checkpoint resumption.

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "remote")]
use sha2::{Digest, Sha256};
#[cfg(feature = "remote")]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(feature = "remote")]
use std::sync::Arc;

#[cfg(feature = "remote")]
use crate::bundle::download;
#[cfg(feature = "remote")]
use crate::bundle::hf::fetch_model_snapshot;
#[cfg(feature = "remote")]
use crate::bundle::{DownloadProgress, HfSpec};
#[cfg(feature = "remote")]
use crate::convert::cache::{ConversionReceipt, ConversionRequest, RECEIPT_NAME};
#[cfg(feature = "remote")]
use crate::convert::checkpoint::{HashingWriter, hash_prefix};
use crate::convert::config::HfModelConfig;
use crate::convert::d1;
use crate::convert::quantize::{QuantStrategy, TargetQuant, quantize_tensor_data_with_strategy};
use crate::convert::safetensors::{
    SafeTensorsHeader, decode_safetensor_to_f32_into, gguf_tensor_dims,
};
use crate::convert::tensor_map::{TensorAction, classify_tensor, stack_expert_tensors};
#[cfg(test)]
use crate::convert::tensor_map::{split_lfm3_vl_gate_up, split_lfm3_vl_qkv};
use crate::convert::tokenizer::{HfTokenizerJson, VocabOptions};
use crate::convert::writer::GgufWriter;
#[cfg(feature = "remote")]
use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};
use crate::session::CeraError;

/// Options controlling model quantization and caching.
#[cfg(feature = "remote")]
#[derive(Clone)]
pub struct QuantizeOptions {
    pub target_quant: TargetQuant,
    pub strategy: QuantStrategy,
    pub cache_dir: PathBuf,
    pub auth_token: Option<String>,
    pub progress: Option<Arc<dyn DownloadProgress>>,
    pub cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub tensor_overrides: Vec<(String, TargetQuant)>,
}

#[cfg(feature = "remote")]
impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            target_quant: TargetQuant::Q4_K_M,
            strategy: QuantStrategy::Auto,
            cache_dir: crate::bundle::hf::default_cache_dir(),
            auth_token: crate::bundle::hf::get_hf_auth_token(),
            progress: None,
            cancel: None,
            tensor_overrides: Vec::new(),
        }
    }
}

static TEMP_FILE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn unique_temp_suffix() -> String {
    let seq = TEMP_FILE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("tmp.{}.{}", std::process::id(), seq)
}

/// RAII guard to automatically clean up temporary files on error or drop.
struct TempFileGuard {
    path: PathBuf,
    active: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, active: true }
    }

    fn disarm(mut self) {
        self.active = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Checkpoint metadata for resuming interrupted streaming quantization.
#[cfg(feature = "remote")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantCheckpoint {
    #[serde(default)]
    request: Option<ConversionRequest>,
    #[serde(default)]
    prefix_sha256: Option<String>,
    #[serde(default)]
    header_sha256: Option<String>,
    quant: String,
    strategy: String,
    completed_tensors: usize,
    total_tensors: usize,
    file_bytes: u64,
}

fn extract_chat_template(tokenizer_config: &serde_json::Value) -> Option<String> {
    tokenizer_config.get("chat_template").and_then(|t| match t {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => arr
            .iter()
            .find(|item| item.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| arr.first())
            .and_then(|item| {
                item.get("template")
                    .and_then(|s| s.as_str())
                    .map(str::to_string)
            }),
        _ => None,
    })
}

/// An optional file of a checkpoint directory: `None` only when it is absent. A file
/// that exists but cannot be read is an error, not a checkpoint without that file.
fn read_optional_file(path: &Path) -> Result<Option<Vec<u8>>, CeraError> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        // a dangling symlink (a pruned blob in a Hugging Face cache snapshot) reads as
        // not found, but the file is there: it is unreadable, not absent
        Err(e)
            if e.kind() == std::io::ErrorKind::NotFound && fs::symlink_metadata(path).is_err() =>
        {
            Ok(None)
        }
        Err(e) => Err(CeraError::Backend(format!(
            "failed to read `{}`: {e}",
            path.display()
        ))),
    }
}

fn read_optional_template(path: &Path) -> Result<Option<String>, CeraError> {
    read_optional_file(path)?
        .map(|bytes| template_from_utf8(bytes, &path.display().to_string()))
        .transpose()
}

fn template_from_utf8(bytes: Vec<u8>, what: &str) -> Result<String, CeraError> {
    // a byte order mark would be kept as a leading U+FEFF and rendered into every prompt,
    // where it stops the first special token from matching
    let text = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&bytes);
    std::str::from_utf8(text)
        .map(str::to_owned)
        .map_err(|e| CeraError::Backend(format!("`{what}` is not valid UTF-8: {e}")))
}

/// `tokenizer_config.json`, parsed. It supplies the special tokens, the `add_*_token`
/// overrides and usually the chat template, so a file that is present but unusable is
/// an error rather than a checkpoint without one.
fn parse_tokenizer_config(bytes: &[u8], what: &str) -> Result<serde_json::Value, CeraError> {
    // a UTF-8 byte order mark is common in files saved on Windows
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);
    serde_json::from_slice(bytes).map_err(|e| {
        CeraError::Backend(format!(
            "`{what}` is not valid JSON ({e}); NaN, Infinity and trailing commas are \
             not accepted"
        ))
    })
}

/// The chat template for a checkpoint: the `chat_template` field of
/// `tokenizer_config.json`, else the standalone `chat_template.jinja` that
/// Transformers 5 writes instead.
fn resolve_chat_template(
    tokenizer_config: Option<&serde_json::Value>,
    template_jinja: Option<String>,
) -> Option<String> {
    config_chat_template(tokenizer_config).or(template_jinja.filter(|t| !t.trim().is_empty()))
}

/// The non-blank `chat_template` of `tokenizer_config.json`. A blank one falls through to
/// the standalone file, and the standalone file is fetched only when this is `None`.
fn config_chat_template(tokenizer_config: Option<&serde_json::Value>) -> Option<String> {
    tokenizer_config
        .and_then(extract_chat_template)
        .filter(|t| !t.trim().is_empty())
}

/// llama.cpp's converter asserts that every token id is below `vocab_size`: the
/// embedding has `vocab_size` rows, so a token past them would index outside it.
fn ensure_vocab_fits(config: &HfModelConfig, tokenizer: &HfTokenizerJson) -> Result<(), CeraError> {
    let max_id = tokenizer.max_token_id() as usize;
    match config.vocab_size {
        Some(vocab_size) if config.uses_llama_cpp_vocab_layout() && max_id >= vocab_size => {
            Err(CeraError::Backend(format!(
                "tokenizer.json has a token id {max_id}, but config.json `vocab_size` is \
                 {vocab_size}"
            )))
        }
        _ => Ok(()),
    }
}

/// Vocabulary layout options for `config`'s architecture.
fn vocab_options<'a>(
    config: &HfModelConfig,
    tokenizer_config: Option<&'a serde_json::Value>,
) -> VocabOptions<'a> {
    VocabOptions {
        llama_cpp_layout: config.uses_llama_cpp_vocab_layout(),
        pad_to: config.vocab_size,
        force_add_bos: config.forced_add_bos_token(),
        tokenizer_config,
    }
}

fn transform_qwen35_ssm_a_inplace(f32_data: &mut [f32]) {
    for v in f32_data.iter_mut() {
        let clamped = v.clamp(-80.0, 80.0);
        *v = -clamped.exp();
    }
}

fn transform_bailingmoe_ssm_a_inplace(f32_data: &mut [f32]) {
    for v in f32_data.iter_mut() {
        let clamped = v.clamp(-80.0, 80.0);
        *v = clamped.exp();
    }
}

/// Stream and quantize a remote Hugging Face SafeTensors repository into a cached GGUF model.
///
/// # Errors
///
/// Besides network and format failures, this refuses (before fetching any weights) a
/// checkpoint it would write a wrong GGUF for: LFM2-MoE, an LFM2 config whose layout or
/// hyperparameters the loaders cannot use (see [`HfModelConfig::ensure_convertible`]), and
/// an LFM2 tokenizer with a token id past `vocab_size`. The optional files
/// `tokenizer_config.json` and `chat_template.jinja` are absent only on a 404; any other
/// failure to fetch or parse them is an error rather than a conversion without them.
#[cfg(feature = "remote")]
pub fn stream_quantize_hf_repo(
    spec: &HfSpec,
    opts: QuantizeOptions,
) -> Result<Manifest, CeraError> {
    let quant_str = opts.target_quant.as_str();
    let strat_str = opts.strategy.as_str();
    let rev_segment = if spec.revision == "main" {
        ""
    } else {
        spec.revision.as_str()
    };
    let strat_segment = if opts.strategy == QuantStrategy::Auto {
        ""
    } else {
        strat_str
    };
    let cache_leaf = match (rev_segment.is_empty(), strat_segment.is_empty()) {
        (true, true) => quant_str.to_string(),
        (false, true) => format!("{quant_str}@{rev_segment}"),
        (true, false) => format!("{quant_str}-{strat_segment}"),
        (false, false) => format!("{quant_str}-{strat_segment}@{rev_segment}"),
    };
    let store_dir = opts
        .cache_dir
        .join("huggingface.co")
        .join(&spec.owner)
        .join(&spec.repo)
        .join("quantized")
        .join(cache_leaf);

    let manifest_path = store_dir.join(format!("{quant_str}.json"));
    let target_gguf_path = store_dir.join("model.gguf");
    let tmp_gguf_path = store_dir.join("model.gguf.tmp");
    let ckpt_path = store_dir.join("model.gguf.checkpoint.json");
    let receipt_path = store_dir.join(RECEIPT_NAME);
    // Resolve even on a completed-cache hit: a branch or tag may have moved.
    // Failure leaves existing artifacts intact but cannot serve stale remote output.
    let (info, pinned_spec) = fetch_model_snapshot(spec, opts.auth_token.as_deref())?;
    let request = ConversionRequest::new(spec, &opts, &pinned_spec.revision);

    if let Some(manifest) = ConversionReceipt::verified_manifest(
        &receipt_path,
        &manifest_path,
        &target_gguf_path,
        &request,
    ) {
        return Ok(manifest);
    }

    // An interrupted repair must not leave an old completion record in place.
    match fs::remove_file(&receipt_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(CeraError::Backend(format!(
                "failed to invalidate conversion receipt `{}`: {e}",
                receipt_path.display()
            )));
        }
    }

    fs::create_dir_all(&store_dir).map_err(|e| {
        CeraError::Backend(format!(
            "failed to create cache dir `{}`: {e}",
            store_dir.display()
        ))
    })?;

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| CeraError::Backend(format!("failed to build reqwest client: {e}")))?;

    // 1. Fetch config.json, tokenizer.json, and optional generation_config.json
    let config_url = pinned_spec.file_download_url("config.json");
    let config_bytes = fetch_hf_file_bytes(&client, &config_url, opts.auth_token.as_deref())?;
    let config = HfModelConfig::parse_from_bytes(&config_bytes)?;
    config.ensure_convertible()?;
    if config.model_type.eq_ignore_ascii_case(d1::MODEL_TYPE) {
        // the d1 head, towers and batch-norm folds are written by the local conversion only;
        // streaming would write the trunk under the wrong names and no head
        return Err(CeraError::Backend(
            "d1-omni checkpoints convert from a local directory: download the repository and run \
             `cera convert --input <dir> --output <file.gguf>`"
                .into(),
        ));
    }

    let tokenizer_url = pinned_spec.file_download_url("tokenizer.json");
    let tokenizer_bytes = fetch_hf_file_bytes(&client, &tokenizer_url, opts.auth_token.as_deref())?;
    let tokenizer = HfTokenizerJson::parse_from_bytes(&tokenizer_bytes)?;
    ensure_vocab_fits(&config, &tokenizer)?;

    // Optional chat template & generation config
    let template_url = pinned_spec.file_download_url("tokenizer_config.json");
    let tokenizer_config =
        fetch_optional_hf_file_bytes(&client, &template_url, opts.auth_token.as_deref())?
            .map(|b| parse_tokenizer_config(&b, &template_url))
            .transpose()?;
    // the standalone file is only consulted when tokenizer_config.json has no template,
    // so a repo that never needed it is not failed by it
    let template_jinja = if config_chat_template(tokenizer_config.as_ref()).is_some() {
        None
    } else {
        let jinja_url = pinned_spec.file_download_url("chat_template.jinja");
        fetch_optional_hf_file_bytes(&client, &jinja_url, opts.auth_token.as_deref())?
            .map(|b| template_from_utf8(b, &jinja_url))
            .transpose()?
    };
    let chat_template = resolve_chat_template(tokenizer_config.as_ref(), template_jinja);

    let gen_url = pinned_spec.file_download_url("generation_config.json");
    let gen_defaults = fetch_hf_file_bytes(&client, &gen_url, opts.auth_token.as_deref())
        .ok()
        .and_then(|b| {
            std::str::from_utf8(&b)
                .ok()
                .and_then(crate::bundle::hf::parse_generation_config_json)
        })
        .unwrap_or(GenerationDefaults::Text {
            temperature: None,
            min_p: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        });

    // 3. Discover SafeTensors files (single or sharded)
    let mut safetensors_files: Vec<String> = info
        .siblings
        .iter()
        .filter(|s| {
            let lower = s.rfilename.to_ascii_lowercase();
            lower.ends_with(".safetensors") && !is_auxiliary_shard(&lower)
        })
        .map(|s| s.rfilename.clone())
        .collect();
    safetensors_files.sort();

    if safetensors_files.is_empty() {
        return Err(CeraError::Backend(format!(
            "repository `{}/{}` has no .safetensors files to quantize",
            spec.owner, spec.repo
        )));
    }

    // 4. Read headers of each SafeTensors shard
    let mut shard_headers = Vec::new();
    for file_name in &safetensors_files {
        let header_url = pinned_spec.file_download_url(file_name);
        // Read 8-byte header size first
        let len_bytes =
            fetch_hf_file_range(&client, &header_url, 0, 7, opts.auth_token.as_deref())?;
        let header_len_arr: [u8; 8] = len_bytes.try_into().map_err(|_| {
            CeraError::Backend(format!(
                "failed to read 8-byte header length from `{header_url}`"
            ))
        })?;
        let header_len_u64 = u64::from_le_bytes(header_len_arr);
        if header_len_u64 == 0 || header_len_u64 > 100_000_000 {
            return Err(CeraError::Backend(format!(
                "invalid SafeTensors header length {header_len_u64} in `{header_url}`"
            )));
        }
        let header_len = header_len_u64 as usize;
        let json_bytes = fetch_hf_file_range(
            &client,
            &header_url,
            8,
            8 + header_len - 1,
            opts.auth_token.as_deref(),
        )?;
        let json_str = std::str::from_utf8(&json_bytes).map_err(|e| {
            CeraError::Backend(format!(
                "invalid UTF-8 in SafeTensors header for `{header_url}`: {e}"
            ))
        })?;
        let header = SafeTensorsHeader::parse_from_json_str(json_str, 8 + header_len)?;
        shard_headers.push((file_name.clone(), header));
    }

    // 5. Initialize GgufWriter & register all tensor metadata
    let mut writer = GgufWriter::new();
    config.apply_to_gguf_writer(&mut writer, &spec.repo);
    tokenizer.apply_to_gguf_writer_with(
        &mut writer,
        chat_template.as_deref(),
        &vocab_options(&config, tokenizer_config.as_ref()),
    );

    let arch = config.gguf_architecture();
    let pending_tensors = plan_pending_tensors(
        shard_headers
            .iter()
            .enumerate()
            .map(|(idx, (_, header))| (idx, header)),
        &config,
        arch,
        false,
        opts.target_quant,
        &opts.tensor_overrides,
        &mut writer,
    )?;

    let total_tensors = pending_tensors.len();

    // Regenerate the header/layout without allocating a second metadata buffer.
    let mut header_hash = HashingWriter {
        inner: std::io::sink(),
        hasher: Sha256::new(),
    };
    let header_bytes = writer.write_header_and_tensor_info(&mut header_hash)?;
    let header_sha256 = download::hex_encode(&header_hash.hasher.finalize());
    let align = writer.alignment() as u64;
    let align_mask = align - 1;

    // 6. Verify the exact saved boundary, layout and bytes before resuming.
    // Keep the verified handle for truncation/writing; do not reopen by path.
    let mut resumed = None;
    if let Ok(ckpt_bytes) = fs::read(&ckpt_path)
        && let Ok(ckpt) = serde_json::from_slice::<QuantCheckpoint>(&ckpt_bytes)
        && ckpt.request.as_ref() == Some(&request)
        && ckpt.quant == quant_str
        && ckpt.strategy == strat_str
        && ckpt.header_sha256.as_ref() == Some(&header_sha256)
        && ckpt.total_tensors == total_tensors
        && ckpt.completed_tensors > 0
        && ckpt.completed_tensors < total_tensors
        && let Some(last) = writer.tensors().get(ckpt.completed_tensors - 1)
        && let Some(boundary) = header_bytes
            .checked_add(last.offset)
            .and_then(|n| n.checked_add((last.size_bytes as u64 + align_mask) & !align_mask))
        && ckpt.file_bytes == boundary
        && let Some(prefix_sha256) = ckpt.prefix_sha256.as_ref()
        && prefix_sha256.len() == 64
        && let Ok(mut file) = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tmp_gguf_path)
        && let Ok(hasher) = hash_prefix(&mut file, boundary)
        && prefix_sha256 == &download::hex_encode(&hasher.clone().finalize())
    {
        tracing::info!(
            "Resuming quantization of `{}/{}` ({quant_str}) from tensor {}/{}…",
            spec.owner,
            spec.repo,
            ckpt.completed_tensors,
            total_tensors
        );
        resumed = Some((file, ckpt, hasher));
    }

    let (out_file, mut current_bytes_written, start_tensor_idx, hasher) =
        if let Some((mut file, ckpt, hasher)) = resumed {
            file.set_len(ckpt.file_bytes).map_err(|e| {
                CeraError::Backend(format!(
                    "failed to truncate `{}` to resume boundary {} bytes: {e}",
                    tmp_gguf_path.display(),
                    ckpt.file_bytes
                ))
            })?;
            file.seek(SeekFrom::Start(ckpt.file_bytes)).map_err(|e| {
                CeraError::Backend(format!(
                    "failed to seek `{}` to resume boundary: {e}",
                    tmp_gguf_path.display()
                ))
            })?;
            (file, ckpt.file_bytes, ckpt.completed_tensors, hasher)
        } else {
            let _ = fs::remove_file(&ckpt_path);
            let file = File::create(&tmp_gguf_path).map_err(|e| {
                CeraError::Backend(format!(
                    "failed to create `{}`: {e}",
                    tmp_gguf_path.display()
                ))
            })?;
            (file, 0, 0, Sha256::new())
        };

    let mut buf_writer = HashingWriter {
        inner: BufWriter::with_capacity(1024 * 1024 * 4, out_file),
        hasher,
    };
    if start_tensor_idx == 0 {
        current_bytes_written = writer.write_header_and_tensor_info(&mut buf_writer)?;
        buf_writer.flush().map_err(|e| {
            CeraError::Backend(format!(
                "failed to flush `{}`: {e}",
                tmp_gguf_path.display()
            ))
        })?;
    }

    // Estimate total output file bytes accurately
    let total_estimated_bytes: u64 = current_bytes_written
        + pending_tensors
            .iter()
            .skip(start_tensor_idx)
            .map(|pt| {
                let raw =
                    TargetQuant::compute_tensor_bytes(pt.ggml_type, pt.expected_elements) as u64;
                (raw + align_mask) & !align_mask
            })
            .sum::<u64>();

    // 7. Stream tensors from SafeTensors shards, quantize on-the-fly, and write to GGUF
    let shard_download_urls: Vec<String> = shard_headers
        .iter()
        .map(|(name, _)| pinned_spec.file_download_url(name))
        .collect();

    let expected_experts = expected_expert_count(&config);
    let mut quant_buf = Vec::new();
    let mut f32_data = Vec::new();
    let mut raw_tensor_bytes = Vec::new();
    let mut last_split_source: Option<(SourceRef, Vec<f32>)> = None;

    for (i, pt) in pending_tensors.iter().enumerate().skip(start_tensor_idx) {
        if let Some(ref cancel) = opts.cancel
            && cancel.load(std::sync::atomic::Ordering::Relaxed)
        {
            let _ = buf_writer.flush();
            return Err(CeraError::Cancelled);
        }

        let tensor_url = &shard_download_urls[pt.action.primary_file_idx()];

        if let Some(p) = &opts.progress {
            p.on_progress(
                tensor_url,
                current_bytes_written,
                Some(total_estimated_bytes),
            );
        }

        materialize_pending_tensor_f32(
            pt,
            expected_experts,
            &mut last_split_source,
            &mut f32_data,
            |src, out| {
                fetch_remote_source_f32(
                    src,
                    &client,
                    &shard_download_urls,
                    opts.auth_token.as_deref(),
                    &mut raw_tensor_bytes,
                    out,
                )
            },
        )?;
        let num_elements = f32_data.len();
        if num_elements != pt.expected_elements {
            return Err(CeraError::Backend(format!(
                "tensor `{}` element count mismatch: expected {}, got {num_elements}",
                pt.gguf_name, pt.expected_elements
            )));
        }

        apply_arch_tensor_transforms(arch, &pt.gguf_name, &mut f32_data);

        // Quantize to target GGML type
        let target_size = TargetQuant::compute_tensor_bytes(pt.ggml_type, num_elements);
        quant_buf.resize(target_size, 0);

        quantize_tensor_data_with_strategy(&f32_data, pt.ggml_type, opts.strategy, &mut quant_buf)?;
        if matches!(pt.action, LocalSourceAction::StackExperts { .. }) {
            f32_data = Vec::new();
        }

        // Write to GGUF with alignment padding
        let written = writer.write_tensor_data(&mut buf_writer, &quant_buf)?;
        current_bytes_written += written as u64;

        // Persist checkpoint periodically
        if (i + 1) % 5 == 0 || i + 1 == total_tensors {
            buf_writer.flush().map_err(|e| {
                CeraError::Backend(format!(
                    "failed to flush `{}`: {e}",
                    tmp_gguf_path.display()
                ))
            })?;
            let ckpt = QuantCheckpoint {
                request: Some(request.clone()),
                prefix_sha256: Some(download::hex_encode(&buf_writer.hasher.clone().finalize())),
                header_sha256: Some(header_sha256.clone()),
                quant: quant_str.to_string(),
                strategy: strat_str.to_string(),
                completed_tensors: i + 1,
                total_tensors,
                file_bytes: current_bytes_written,
            };

            if let Ok(json) = serde_json::to_string(&ckpt) {
                let ckpt_tmp = store_dir.join(format!(
                    "model.gguf.checkpoint.json.{}",
                    unique_temp_suffix()
                ));
                if fs::write(&ckpt_tmp, json).is_ok() {
                    #[cfg(windows)]
                    let _ = fs::remove_file(&ckpt_path);
                    if fs::rename(&ckpt_tmp, &ckpt_path).is_err() {
                        let _ = fs::remove_file(&ckpt_path);
                        if fs::rename(&ckpt_tmp, &ckpt_path).is_err() {
                            let _ = fs::remove_file(&ckpt_tmp);
                        }
                    }
                }
            }
        }
    }

    if let Some(p) = &opts.progress {
        p.on_progress(
            &pinned_spec.file_download_url("model.gguf"),
            current_bytes_written,
            Some(current_bytes_written),
        );
    }

    buf_writer.flush().map_err(|e| {
        CeraError::Backend(format!(
            "failed to flush `{}`: {e}",
            tmp_gguf_path.display()
        ))
    })?;
    if let Ok(file) = buf_writer.inner.into_inner() {
        let _ = file.sync_all();
    }

    // 8. Compute final SHA-256 digest
    let final_digest = crate::bundle::download::sha256_file(&tmp_gguf_path).map_err(|e| {
        CeraError::Backend(format!(
            "failed to compute sha256 of `{}`: {e}",
            tmp_gguf_path.display()
        ))
    })?;

    // 9. Atomic Rename and SHA-256 sidecar
    let _ = fs::remove_file(&target_gguf_path);
    fs::rename(&tmp_gguf_path, &target_gguf_path).map_err(|e| {
        CeraError::Backend(format!(
            "failed to rename to `{}`: {e}",
            target_gguf_path.display()
        ))
    })?;

    // Remove completed checkpoint
    let _ = fs::remove_file(&ckpt_path);

    let sha_path = store_dir.join("model.gguf.sha256");
    let _ = fs::write(sha_path, format!("{final_digest}\n"));

    // 10. Generate local Manifest JSON
    let files = ManifestFiles {
        model: target_gguf_path.to_string_lossy().to_string(),
        multimodal_projector: None,
        audio_decoder: None,
        audio_tokenizer: None,
        draft_model: None,
        extras: std::collections::HashMap::new(),
    };

    let mut load_time_params = serde_json::json!({
        "model": "model.gguf"
    });
    if let Some(tmpl) = &chat_template {
        load_time_params["chat_template"] = serde_json::Value::String(tmpl.clone());
    }

    let raw_val = serde_json::json!({
        "inference_type": InferenceType::LlamaCppTextToText.as_str(),
        "schema_version": "1.0.0",
        "load_time_parameters": load_time_params,
        "generation_time_parameters": gen_defaults.to_json_value(),
        "quant": quant_str,
        "strategy": strat_str,
    });

    let manifest_json = serde_json::to_string_pretty(&raw_val)
        .map_err(|e| CeraError::Backend(format!("failed to serialize manifest json: {e}")))?;

    fs::write(&manifest_path, &manifest_json).map_err(|e| {
        CeraError::Backend(format!(
            "failed to write `{}`: {e}",
            manifest_path.display()
        ))
    })?;

    // Publish completion last. Without this record the next load reconverts.
    let receipt = ConversionReceipt::new(request, manifest_json.as_bytes(), final_digest);
    let receipt_json = serde_json::to_vec(&receipt)
        .map_err(|e| CeraError::Backend(format!("failed to serialize conversion receipt: {e}")))?;
    let receipt_tmp = store_dir.join(format!("{RECEIPT_NAME}.{}", unique_temp_suffix()));
    let receipt_guard = TempFileGuard::new(receipt_tmp.clone());
    fs::write(&receipt_tmp, receipt_json)
        .and_then(|()| fs::rename(&receipt_tmp, &receipt_path))
        .map_err(|e| {
            CeraError::Backend(format!(
                "failed to publish conversion receipt `{}`: {e}",
                receipt_path.display()
            ))
        })?;
    receipt_guard.disarm();

    let manifest = Manifest {
        inference_type: InferenceType::LlamaCppTextToText,
        schema_version: "1.0.0".into(),
        files,
        chat_template,
        generation_defaults: gen_defaults,
        raw: raw_val,
    };

    Ok(manifest)
}

/// Quantize a local SafeTensors directory or file to GGUF.
pub fn quantize_safetensors_to_gguf(
    input_path: &Path,
    output_gguf_path: &Path,
    quant: TargetQuant,
) -> Result<(), CeraError> {
    quantize_safetensors_to_gguf_with_strategy(
        input_path,
        output_gguf_path,
        quant,
        QuantStrategy::Auto,
        &[],
    )
}

/// Quantize a local SafeTensors directory or file to GGUF with per-tensor quantization overrides.
pub fn quantize_safetensors_to_gguf_with_overrides(
    input_path: &Path,
    output_gguf_path: &Path,
    quant: TargetQuant,
    overrides: &[(String, TargetQuant)],
) -> Result<(), CeraError> {
    quantize_safetensors_to_gguf_with_strategy(
        input_path,
        output_gguf_path,
        quant,
        QuantStrategy::Auto,
        overrides,
    )
}

/// Whether the four batch-norm spans cover the same channels, so folding them cannot
/// index past the short one.
fn fold_stats_match(stats: &[Vec<f32>]) -> bool {
    stats
        .first()
        .is_some_and(|first| stats.len() == 4 && stats.iter().all(|s| s.len() == first.len()))
}

/// Fold batch-norm statistics, refusing spans that disagree on channels with the span
/// lengths instead of indexing past the short one inside the fold.
fn fold_stats_checked(
    kind: d1::Fold,
    stats: &[Vec<f32>],
    gguf_name: &str,
) -> Result<Vec<f32>, CeraError> {
    if !fold_stats_match(stats) {
        let lens: Vec<usize> = stats.iter().map(Vec::len).collect();
        return Err(CeraError::Backend(format!(
            "batch-norm statistics disagree for `{gguf_name}`: {lens:?}"
        )));
    }
    Ok(d1::fold_batch_norm(
        kind, &stats[0], &stats[1], &stats[2], &stats[3],
    ))
}

/// Where a tensor's bytes sit in the checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceRef {
    file_idx: usize,
    dtype: String,
    data_start: usize,
    data_end: usize,
}

/// Deferred source action for materializing a planned GGUF tensor on demand during the write pass.
#[derive(Clone, Debug)]
enum LocalSourceAction {
    /// Read and decode a single tensor directly from a shard.
    Direct(SourceRef),
    /// Values for a tensor generated from the config instead of read from a shard.
    Generated(Vec<f32>),
    /// A tensor derived from several sources (a folded batch norm): the half it holds
    /// and the weight, bias, running mean and running variance it is computed from.
    Fold {
        kind: d1::Fold,
        sources: Vec<SourceRef>,
    },
    /// A row slice (`attn_q`, `attn_k`, or `attn_v`) of a fused LFM3-VL `self_attn.qkv.weight` tensor.
    SplitVlQkvSlice {
        source: SourceRef,
        block_idx: usize,
        shape: Vec<usize>,
        out_offset_elems: usize,
        out_len_elems: usize,
    },
    /// A row half (`ffn_gate` or `ffn_up`) of a fused LFM3-VL `feed_forward.gate_up.weight` tensor.
    SplitVlGateUpSlice {
        source: SourceRef,
        block_idx: usize,
        shape: Vec<usize>,
        is_up: bool,
    },
    /// Per-expert 2D tensors stacked in ascending `expert_idx` order into a 3D `[n_experts, rows, cols]` tensor.
    StackExperts {
        experts: Vec<(usize, SourceRef)>,
        rows: usize,
        cols: usize,
    },
}

impl LocalSourceAction {
    #[cfg(feature = "remote")]
    fn primary_file_idx(&self) -> usize {
        match self {
            Self::Direct(src)
            | Self::SplitVlQkvSlice { source: src, .. }
            | Self::SplitVlGateUpSlice { source: src, .. } => src.file_idx,
            Self::Generated(_) => 0,
            Self::Fold { sources, .. } => sources.first().map_or(0, |s| s.file_idx),
            Self::StackExperts { experts, .. } => experts.first().map_or(0, |(_, s)| s.file_idx),
        }
    }
}

#[derive(Clone, Debug)]
struct LocalPendingTensor {
    expected_elements: usize,
    ggml_type: u32,
    gguf_name: String,
    action: LocalSourceAction,
}

type PlannedSlice = (String, Vec<usize>, usize, LocalSourceAction);

fn expected_expert_count(config: &HfModelConfig) -> Option<usize> {
    config.expert_count()
}

fn plan_split_lfm3_vl_qkv(
    shape: &[usize],
    block_idx: usize,
    config: &HfModelConfig,
    source: SourceRef,
) -> Result<[PlannedSlice; 3], CeraError> {
    if shape.len() != 2 {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused qkv.weight must be 2D, got {shape:?}"
        )));
    }
    let n_head = config.num_attention_heads.unwrap_or(0);
    let n_head_kv = config
        .num_key_value_heads
        .or(config.num_attention_heads)
        .unwrap_or(0);
    let head_dim = config.lfm3_head_dim().unwrap_or(0);
    let (total_rows, cols) = (shape[0], shape[1]);
    let Some((q_rows, kv_rows, expected_rows)) =
        crate::convert::tensor_map::lfm3_vl_qkv_rows(n_head, n_head_kv, head_dim)
    else {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused qkv.weight: {n_head} heads and {n_head_kv} kv heads of \
             width {head_dim} overflow the row count"
        )));
    };
    if total_rows != expected_rows {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused qkv.weight shape {shape:?} does not match \
             q_rows={q_rows} + 2*kv_rows={kv_rows}"
        )));
    }
    let _total_elems = total_rows.checked_mul(cols).ok_or_else(|| {
        CeraError::Backend(format!("blk.{block_idx} fused qkv.weight shape overflow"))
    })?;
    let q_elems = q_rows * cols;
    let kv_elems = kv_rows * cols;
    let full_shape = shape.to_vec();
    Ok([
        (
            format!("blk.{block_idx}.attn_q.weight"),
            vec![q_rows, cols],
            q_elems,
            LocalSourceAction::SplitVlQkvSlice {
                source: source.clone(),
                block_idx,
                shape: full_shape.clone(),
                out_offset_elems: 0,
                out_len_elems: q_elems,
            },
        ),
        (
            format!("blk.{block_idx}.attn_k.weight"),
            vec![kv_rows, cols],
            kv_elems,
            LocalSourceAction::SplitVlQkvSlice {
                source: source.clone(),
                block_idx,
                shape: full_shape.clone(),
                out_offset_elems: q_elems,
                out_len_elems: kv_elems,
            },
        ),
        (
            format!("blk.{block_idx}.attn_v.weight"),
            vec![kv_rows, cols],
            kv_elems,
            LocalSourceAction::SplitVlQkvSlice {
                source,
                block_idx,
                shape: full_shape,
                out_offset_elems: q_elems + kv_elems,
                out_len_elems: kv_elems,
            },
        ),
    ])
}

fn plan_split_lfm3_vl_gate_up(
    shape: &[usize],
    block_idx: usize,
    source: SourceRef,
) -> Result<[PlannedSlice; 2], CeraError> {
    if shape.len() != 2 || !shape[0].is_multiple_of(2) {
        return Err(CeraError::Backend(format!(
            "blk.{block_idx} fused gate_up.weight must have even 2D rows, got {shape:?}"
        )));
    }
    let (total_rows, cols) = (shape[0], shape[1]);
    let total_elems = total_rows.checked_mul(cols).ok_or_else(|| {
        CeraError::Backend(format!(
            "blk.{block_idx} fused gate_up.weight shape overflow"
        ))
    })?;
    let half_rows = total_rows / 2;
    let half_elems = total_elems / 2;
    let full_shape = shape.to_vec();
    Ok([
        (
            format!("blk.{block_idx}.ffn_gate.weight"),
            vec![half_rows, cols],
            half_elems,
            LocalSourceAction::SplitVlGateUpSlice {
                source: source.clone(),
                block_idx,
                shape: full_shape.clone(),
                is_up: false,
            },
        ),
        (
            format!("blk.{block_idx}.ffn_up.weight"),
            vec![half_rows, cols],
            half_elems,
            LocalSourceAction::SplitVlGateUpSlice {
                source,
                block_idx,
                shape: full_shape,
                is_up: true,
            },
        ),
    ])
}

fn plan_stacked_experts(
    mut experts: Vec<(usize, Vec<usize>, SourceRef)>,
    expected_experts: Option<usize>,
    gguf_name: &str,
) -> Result<(Vec<usize>, usize, LocalSourceAction), CeraError> {
    if experts.is_empty() {
        return Err(CeraError::Backend(format!(
            "no expert tensors found for `{gguf_name}`"
        )));
    }
    experts.sort_by_key(|(idx, _, _)| *idx);
    if let Some(expected) = expected_experts
        && experts.len() != expected
    {
        return Err(CeraError::Backend(format!(
            "`{gguf_name}` has {} expert tensors, expected {expected}",
            experts.len()
        )));
    }
    let first_shape = &experts[0].1;
    if first_shape.len() != 2 {
        return Err(CeraError::Backend(format!(
            "`{gguf_name}` expert 0 has shape {first_shape:?}; expected 2D [rows, cols]"
        )));
    }
    let (rows, cols) = (first_shape[0], first_shape[1]);
    let slice_len = rows
        .checked_mul(cols)
        .ok_or_else(|| CeraError::Backend(format!("`{gguf_name}` expert dimensions overflow")))?;
    let total_len = slice_len.checked_mul(experts.len()).ok_or_else(|| {
        CeraError::Backend(format!("`{gguf_name}` stacked expert buffer size overflow"))
    })?;
    let mut ordered_sources = Vec::with_capacity(experts.len());
    for (expected_idx, (idx, shape, src)) in experts.into_iter().enumerate() {
        if idx != expected_idx {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` missing expert {expected_idx} (found expert {idx})"
            )));
        }
        if shape.as_slice() != [rows, cols] {
            return Err(CeraError::Backend(format!(
                "`{gguf_name}` expert {idx} shape {shape:?} does not match [{rows}, {cols}]"
            )));
        }
        ordered_sources.push((idx, src));
    }
    let stacked_shape = vec![ordered_sources.len(), rows, cols];
    Ok((
        stacked_shape,
        total_len,
        LocalSourceAction::StackExperts {
            experts: ordered_sources,
            rows,
            cols,
        },
    ))
}

fn plan_pending_tensors<'a>(
    shard_headers: impl IntoIterator<Item = (usize, &'a SafeTensorsHeader)>,
    config: &HfModelConfig,
    arch: &str,
    is_d1: bool,
    quant: TargetQuant,
    overrides: &[(String, TargetQuant)],
    writer: &mut GgufWriter,
) -> Result<Vec<LocalPendingTensor>, CeraError> {
    let mut pending_tensors = Vec::new();
    // the batch-norm statistics of the d1 conformer, by layer, folded after the scan
    let mut batch_norms: std::collections::BTreeMap<
        usize,
        std::collections::BTreeMap<d1::BatchNormStat, (SourceRef, usize)>,
    > = std::collections::BTreeMap::new();
    // per-expert tensors of an LFM3-MoE checkpoint, grouped by (block_idx, gguf_name) and stacked after the scan
    type ExpertSliceEntry = (usize, Vec<usize>, SourceRef);
    let mut stacked_experts: std::collections::BTreeMap<(usize, String), Vec<ExpertSliceEntry>> =
        std::collections::BTreeMap::new();

    for (file_idx, header) in shard_headers {
        for (raw_name, tensor_info) in &header.tensors {
            if config.skips_tensor(raw_name) {
                continue;
            }
            let data_start = header
                .header_size_bytes
                .checked_add(tensor_info.data_offsets.0)
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor offset start overflow for `{raw_name}`"))
                })?;
            let data_end = header
                .header_size_bytes
                .checked_add(tensor_info.data_offsets.1)
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor offset end overflow for `{raw_name}`"))
                })?;
            let source = SourceRef {
                file_idx,
                dtype: tensor_info.dtype.clone(),
                data_start,
                data_end,
            };

            if !is_d1 {
                match classify_tensor(raw_name, arch) {
                    TensorAction::Skip => continue,
                    TensorAction::StackExpert {
                        block_idx,
                        expert_idx,
                        gguf_name,
                    } => {
                        stacked_experts
                            .entry((block_idx, gguf_name))
                            .or_default()
                            .push((expert_idx, tensor_info.shape.clone(), source));
                        continue;
                    }
                    TensorAction::SplitVlQkv { block_idx } => {
                        for (gguf_name, slice_shape, n_elems, action) in
                            plan_split_lfm3_vl_qkv(&tensor_info.shape, block_idx, config, source)?
                        {
                            let ggml_type = quant.select_ggml_type_with_overrides(
                                &gguf_name,
                                slice_shape.len(),
                                n_elems,
                                slice_shape.last().copied().unwrap_or(1),
                                overrides,
                            );
                            let dims = gguf_tensor_dims(&gguf_name, &slice_shape);
                            writer.add_tensor(
                                &gguf_name,
                                dims,
                                ggml_type,
                                TargetQuant::compute_tensor_bytes(ggml_type, n_elems),
                            );
                            pending_tensors.push(LocalPendingTensor {
                                expected_elements: n_elems,
                                ggml_type,
                                gguf_name,
                                action,
                            });
                        }
                        continue;
                    }
                    TensorAction::SplitVlGateUp { block_idx } => {
                        for (gguf_name, slice_shape, n_elems, action) in
                            plan_split_lfm3_vl_gate_up(&tensor_info.shape, block_idx, source)?
                        {
                            let ggml_type = quant.select_ggml_type_with_overrides(
                                &gguf_name,
                                slice_shape.len(),
                                n_elems,
                                slice_shape.last().copied().unwrap_or(1),
                                overrides,
                            );
                            let dims = gguf_tensor_dims(&gguf_name, &slice_shape);
                            writer.add_tensor(
                                &gguf_name,
                                dims,
                                ggml_type,
                                TargetQuant::compute_tensor_bytes(ggml_type, n_elems),
                            );
                            pending_tensors.push(LocalPendingTensor {
                                expected_elements: n_elems,
                                ggml_type,
                                gguf_name,
                                action,
                            });
                        }
                        continue;
                    }
                    TensorAction::Map(gguf_name) => {
                        let num_elements: usize = tensor_info
                            .shape
                            .iter()
                            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                            .ok_or_else(|| {
                                CeraError::Backend(format!(
                                    "tensor shape overflow for `{raw_name}`"
                                ))
                            })?;
                        let ggml_type = quant.select_ggml_type_with_overrides(
                            &gguf_name,
                            tensor_info.shape.len(),
                            num_elements,
                            tensor_info.shape.last().copied().unwrap_or(1),
                            overrides,
                        );
                        let dims = gguf_tensor_dims(&gguf_name, &tensor_info.shape);
                        let out_bytes = TargetQuant::compute_tensor_bytes(ggml_type, num_elements);
                        writer.add_tensor(&gguf_name, dims, ggml_type, out_bytes);
                        pending_tensors.push(LocalPendingTensor {
                            expected_elements: num_elements,
                            ggml_type,
                            gguf_name,
                            action: LocalSourceAction::Direct(source),
                        });
                        continue;
                    }
                }
            }

            let num_elements: usize = tensor_info
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor shape overflow for `{raw_name}`"))
                })?;
            let (gguf_name, batch_norm) = match d1::tensor_plan(raw_name) {
                d1::TensorPlan::Skip => continue,
                d1::TensorPlan::Name(name) => (name, None),
                d1::TensorPlan::BatchNorm { layer, stat } => (String::new(), Some((layer, stat))),
            };
            if let Some((layer, stat)) = batch_norm {
                batch_norms
                    .entry(layer)
                    .or_default()
                    .insert(stat, (source, num_elements));
                continue;
            }
            let ggml_type = if d1::keeps_f32(&gguf_name) {
                crate::convert::writer::GGML_TYPE_F32
            } else {
                quant.select_ggml_type_with_overrides(
                    &gguf_name,
                    tensor_info.shape.len(),
                    num_elements,
                    tensor_info.shape.last().copied().unwrap_or(1),
                    overrides,
                )
            };

            let dims = d1::tensor_dims(&gguf_name, &tensor_info.shape);
            let out_bytes = TargetQuant::compute_tensor_bytes(ggml_type, num_elements);

            writer.add_tensor(&gguf_name, dims, ggml_type, out_bytes);

            pending_tensors.push(LocalPendingTensor {
                expected_elements: num_elements,
                ggml_type,
                gguf_name,
                action: LocalSourceAction::Direct(source),
            });
        }
    }

    let expected_experts = expected_expert_count(config);
    for ((_block_idx, gguf_name), expert_refs) in stacked_experts {
        let (stacked_shape, n_elems, action) =
            plan_stacked_experts(expert_refs, expected_experts, &gguf_name)?;
        let ggml_type = quant.select_ggml_type_with_overrides(
            &gguf_name,
            stacked_shape.len(),
            n_elems,
            stacked_shape.last().copied().unwrap_or(1),
            overrides,
        );
        let dims = gguf_tensor_dims(&gguf_name, &stacked_shape);
        writer.add_tensor(
            &gguf_name,
            dims,
            ggml_type,
            TargetQuant::compute_tensor_bytes(ggml_type, n_elems),
        );
        pending_tensors.push(LocalPendingTensor {
            expected_elements: n_elems,
            ggml_type,
            gguf_name,
            action,
        });
    }

    // each conformer layer's batch norm becomes a scale and a shift
    for (layer, stats) in &batch_norms {
        use d1::BatchNormStat::{Bias, Mean, Var, Weight};
        let mut sources = Vec::with_capacity(4);
        let mut elements = None;
        for stat in [Weight, Bias, Mean, Var] {
            let (source, n) = stats.get(&stat).ok_or_else(|| {
                CeraError::Backend(format!(
                    "d1-omni conformer layer {layer} has no batch-norm {stat:?} tensor"
                ))
            })?;
            if *elements.get_or_insert(*n) != *n {
                return Err(CeraError::Backend(format!(
                    "d1-omni conformer layer {layer}: the batch-norm tensors differ in length"
                )));
            }
            sources.push(source.clone());
        }
        let n = elements.unwrap_or(0);
        for (name, kind) in [("weight", d1::Fold::Scale), ("bias", d1::Fold::Shift)] {
            let gguf_name = format!("a.blk.{layer}.conv_norm.{name}");
            let ggml_type = crate::convert::writer::GGML_TYPE_F32;
            writer.add_tensor(
                &gguf_name,
                vec![n as u64],
                ggml_type,
                TargetQuant::compute_tensor_bytes(ggml_type, n),
            );
            pending_tensors.push(LocalPendingTensor {
                expected_elements: n,
                ggml_type,
                gguf_name,
                action: LocalSourceAction::Fold {
                    kind,
                    sources: sources.clone(),
                },
            });
        }
    }

    for generated in config.generated_tensors() {
        let num_elements = generated.data.len();
        let row_len = generated.dims.first().copied().unwrap_or(1) as usize;
        let ggml_type = quant.select_ggml_type_with_overrides(
            &generated.name,
            generated.dims.len(),
            num_elements,
            row_len,
            overrides,
        );
        writer.add_tensor(
            &generated.name,
            generated.dims,
            ggml_type,
            TargetQuant::compute_tensor_bytes(ggml_type, num_elements),
        );
        pending_tensors.push(LocalPendingTensor {
            expected_elements: num_elements,
            ggml_type,
            gguf_name: generated.name,
            action: LocalSourceAction::Generated(generated.data),
        });
    }

    Ok(pending_tensors)
}

fn materialize_pending_tensor_f32<F>(
    pt: &LocalPendingTensor,
    expected_experts: Option<usize>,
    last_split_source: &mut Option<(SourceRef, Vec<f32>)>,
    f32_data: &mut Vec<f32>,
    mut read_source: F,
) -> Result<(), CeraError>
where
    F: FnMut(&SourceRef, &mut Vec<f32>) -> Result<(), CeraError>,
{
    if !matches!(
        pt.action,
        LocalSourceAction::SplitVlQkvSlice { .. } | LocalSourceAction::SplitVlGateUpSlice { .. }
    ) {
        *last_split_source = None;
    }

    match &pt.action {
        LocalSourceAction::Direct(src) => {
            read_source(src, f32_data)?;
        }
        LocalSourceAction::Generated(values) => {
            // a tensor the checkpoint does not store but the GGUF needs
            f32_data.clear();
            f32_data.extend_from_slice(values);
        }
        LocalSourceAction::Fold { kind, sources } => {
            // a folded batch norm: read its weight, bias, running mean and running variance
            let mut stats: Vec<Vec<f32>> = Vec::with_capacity(sources.len());
            for src in sources {
                let mut values = Vec::new();
                read_source(src, &mut values)?;
                stats.push(values);
            }
            // a corrupt shard fails here with the span lengths, not inside the fold with
            // an index panic
            *f32_data = fold_stats_checked(*kind, &stats, &pt.gguf_name)?;
        }
        LocalSourceAction::SplitVlQkvSlice {
            source,
            block_idx,
            shape,
            out_offset_elems,
            out_len_elems,
        } => {
            let needs_load = last_split_source
                .as_ref()
                .is_none_or(|(cached_src, _)| cached_src != source);
            if needs_load {
                let mut decoded = Vec::new();
                read_source(source, &mut decoded)?;
                *last_split_source = Some((source.clone(), decoded));
            }
            let decoded = &last_split_source.as_ref().unwrap().1;
            let expected_len = shape[0].saturating_mul(shape[1]);
            let end_elems = out_offset_elems.saturating_add(*out_len_elems);
            if decoded.len() != expected_len || end_elems > decoded.len() {
                return Err(CeraError::Backend(format!(
                    "blk.{block_idx} fused qkv.weight element count mismatch: expected {expected_len}, got {}",
                    decoded.len()
                )));
            }
            f32_data.clear();
            f32_data.extend_from_slice(&decoded[*out_offset_elems..end_elems]);
            if end_elems >= decoded.len() {
                *last_split_source = None;
            }
        }
        LocalSourceAction::SplitVlGateUpSlice {
            source,
            block_idx,
            shape,
            is_up,
        } => {
            let needs_load = last_split_source
                .as_ref()
                .is_none_or(|(cached_src, _)| cached_src != source);
            if needs_load {
                let mut decoded = Vec::new();
                read_source(source, &mut decoded)?;
                *last_split_source = Some((source.clone(), decoded));
            }
            let decoded = &last_split_source.as_ref().unwrap().1;
            let expected_len = shape[0].saturating_mul(shape[1]);
            if decoded.len() != expected_len {
                return Err(CeraError::Backend(format!(
                    "blk.{block_idx} fused gate_up.weight element count mismatch: expected {expected_len}, got {}",
                    decoded.len()
                )));
            }
            let half_elems = expected_len / 2;
            let slice = if *is_up {
                &decoded[half_elems..]
            } else {
                &decoded[..half_elems]
            };
            f32_data.clear();
            f32_data.extend_from_slice(slice);
            if *is_up {
                *last_split_source = None;
            }
        }
        LocalSourceAction::StackExperts {
            experts,
            rows,
            cols,
        } => {
            let mut decoded_experts = Vec::with_capacity(experts.len());
            for (exp_idx, src) in experts {
                let mut values = Vec::new();
                read_source(src, &mut values)?;
                decoded_experts.push((*exp_idx, vec![*rows, *cols], values));
            }
            let (_, stacked_data) =
                stack_expert_tensors(&mut decoded_experts, expected_experts, &pt.gguf_name)?;
            *f32_data = stacked_data;
        }
    }

    Ok(())
}

fn apply_arch_tensor_transforms(arch: &str, gguf_name: &str, f32_data: &mut [f32]) {
    // Hugging Face Gemma and Gemma 2 checkpoints store RMSNorm weights as an offset
    // from 1.0: `x * (1.0 + weight)` (see Hugging Face transformers modeling_gemma.py
    // and modeling_gemma2.py). Standard GGUF runtime RMSNorm executes standard `x * weight`
    // without runtime offset addition. Therefore, standard GGUF converters
    // (including upstream llama.cpp conversion/gemma.py:67) fold the +1.0 offset
    // directly into the exported tensor weights during conversion so real checkpoints
    // produce correct activations at inference.
    if (arch == "gemma" || arch == "gemma2") && gguf_name.ends_with("norm.weight") {
        for v in f32_data.iter_mut() {
            *v += 1.0;
        }
    }

    if matches!(arch, "qwen35" | "qwen3_5" | "qwen3.5")
        && (gguf_name.ends_with(".ssm_a") || gguf_name.ends_with(".ssm_a.weight"))
    {
        transform_qwen35_ssm_a_inplace(f32_data);
    }

    if matches!(arch, "bailingmoe3" | "bailingmoe" | "bailingmoe2")
        && (gguf_name.ends_with(".ssm_a") || gguf_name.ends_with(".ssm_a.weight"))
    {
        transform_bailingmoe_ssm_a_inplace(f32_data);
    }
}

fn read_local_source_f32(
    src: &SourceRef,
    safetensors_files: &[PathBuf],
    current_file_idx: &mut usize,
    current_file: &mut Option<File>,
    raw_bytes: &mut Vec<u8>,
    f32_out: &mut Vec<f32>,
) -> Result<(), CeraError> {
    if src.file_idx != *current_file_idx {
        let f = File::open(&safetensors_files[src.file_idx]).map_err(|e| {
            CeraError::Backend(format!(
                "failed to open `{}`: {e}",
                safetensors_files[src.file_idx].display()
            ))
        })?;
        *current_file = Some(f);
        *current_file_idx = src.file_idx;
    }

    let file = current_file
        .as_mut()
        .ok_or_else(|| CeraError::Backend("no active safetensors shard file".into()))?;

    // The span comes from the shard's own header: bound it by the file before sizing a
    // buffer from it, or a crafted header allocates (and zero-fills) gigabytes.
    let file_len = file
        .metadata()
        .map_err(|e| CeraError::Backend(format!("failed to stat safetensors shard: {e}")))?
        .len();
    if src.data_end < src.data_start || src.data_end as u64 > file_len {
        return Err(CeraError::Backend(format!(
            "tensor data range {}..{} extends past the end of its {file_len}-byte shard",
            src.data_start, src.data_end
        )));
    }
    let byte_len = src.data_end - src.data_start;
    file.seek(SeekFrom::Start(src.data_start as u64))
        .map_err(|e| CeraError::Backend(format!("failed to seek in safetensors shard: {e}")))?;

    raw_bytes.resize(byte_len, 0);
    file.read_exact(raw_bytes).map_err(|e| {
        CeraError::Backend(format!(
            "failed to read tensor data from safetensors shard: {e}"
        ))
    })?;

    decode_safetensor_to_f32_into(raw_bytes, &src.dtype, f32_out)
}

#[cfg(feature = "remote")]
fn fetch_remote_source_f32(
    src: &SourceRef,
    client: &reqwest::blocking::Client,
    shard_download_urls: &[String],
    auth_token: Option<&str>,
    raw_bytes: &mut Vec<u8>,
    f32_out: &mut Vec<f32>,
) -> Result<(), CeraError> {
    let tensor_url = &shard_download_urls[src.file_idx];
    if src.data_start < src.data_end {
        fetch_hf_file_range_into(
            client,
            tensor_url,
            src.data_start,
            src.data_end - 1,
            auth_token,
            raw_bytes,
        )?;
    } else {
        raw_bytes.clear();
    }
    decode_safetensor_to_f32_into(raw_bytes, &src.dtype, f32_out)
}

/// Quantize a local SafeTensors directory or file to GGUF with explicit strategy and per-tensor overrides.
///
/// # Errors
///
/// The same refusals as `stream_quantize_hf_repo` (available with the `remote` feature). `tokenizer_config.json`,
/// `chat_template.jinja` and `tokenizer.json` are absent only when the file is not there:
/// one that is unreadable (including a dangling symlink), not UTF-8, or not valid JSON is an
/// error rather than a conversion without it.
pub fn quantize_safetensors_to_gguf_with_strategy(
    input_path: &Path,
    output_gguf_path: &Path,
    quant: TargetQuant,
    strategy: QuantStrategy,
    overrides: &[(String, TargetQuant)],
) -> Result<(), CeraError> {
    if !input_path.exists() {
        return Err(CeraError::Backend(format!(
            "input path `{}` does not exist",
            input_path.display()
        )));
    }

    let dir = if input_path.is_dir() {
        input_path.to_path_buf()
    } else {
        input_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };

    let config_path = dir.join("config.json");
    let tokenizer_path = dir.join("tokenizer.json");

    if !config_path.exists() {
        return Err(CeraError::Backend(format!(
            "config.json not found in `{}`",
            dir.display()
        )));
    }

    let config_bytes = fs::read(&config_path).map_err(|e| {
        CeraError::Backend(format!("failed to read `{}`: {e}", config_path.display()))
    })?;
    let config = HfModelConfig::parse_from_bytes(&config_bytes)?;
    config.ensure_convertible()?;
    let arch = config.gguf_architecture().to_string();
    let is_d1 = config.model_type.eq_ignore_ascii_case(d1::MODEL_TYPE);

    let mut writer = GgufWriter::new();
    let model_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    config.apply_to_gguf_writer(&mut writer, model_name);
    if is_d1 {
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes).map_err(|e| {
            CeraError::Backend(format!("failed to parse `{}`: {e}", config_path.display()))
        })?;
        d1::apply_metadata(&raw, &mut writer)?;
    }

    if let Some(tok_bytes) = read_optional_file(&tokenizer_path)? {
        let tokenizer = HfTokenizerJson::parse_from_bytes(&tok_bytes)?;
        ensure_vocab_fits(&config, &tokenizer)?;
        let tokenizer_config_path = dir.join("tokenizer_config.json");
        let tokenizer_config = read_optional_file(&tokenizer_config_path)?
            .map(|b| parse_tokenizer_config(&b, &tokenizer_config_path.display().to_string()))
            .transpose()?;
        let template_jinja = if config_chat_template(tokenizer_config.as_ref()).is_some() {
            None
        } else {
            read_optional_template(&dir.join("chat_template.jinja"))?
        };
        let chat_template = resolve_chat_template(tokenizer_config.as_ref(), template_jinja);
        tokenizer.apply_to_gguf_writer_with(
            &mut writer,
            chat_template.as_deref(),
            &vocab_options(&config, tokenizer_config.as_ref()),
        );
    }

    let mut safetensors_files = Vec::new();
    if input_path.is_file()
        && input_path.extension().and_then(|e| e.to_str()) == Some("safetensors")
    {
        safetensors_files.push(input_path.to_path_buf());
    } else {
        for entry in fs::read_dir(&dir)
            .map_err(|e| CeraError::Backend(format!("failed to read dir: {e}")))?
        {
            let entry =
                entry.map_err(|e| CeraError::Backend(format!("failed to read entry: {e}")))?;
            let p = entry.path();
            let is_safetensors =
                p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("safetensors");
            let filename_lower = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if is_safetensors && !is_auxiliary_shard(&filename_lower) {
                safetensors_files.push(p);
            }
        }
        safetensors_files.sort();
    }

    if safetensors_files.is_empty() {
        return Err(CeraError::Backend(format!(
            "no .safetensors files found in `{}`",
            dir.display()
        )));
    }

    let mut shard_headers = Vec::with_capacity(safetensors_files.len());
    for file_path in &safetensors_files {
        let mut file = File::open(file_path).map_err(|e| {
            CeraError::Backend(format!("failed to open `{}`: {e}", file_path.display()))
        })?;
        let header = SafeTensorsHeader::parse_from_reader(&mut file)?;
        shard_headers.push(header);
    }

    let pending_tensors = plan_pending_tensors(
        shard_headers.iter().enumerate(),
        &config,
        &arch,
        is_d1,
        quant,
        overrides,
        &mut writer,
    )?;

    let tmp_out = output_gguf_path.with_extension(unique_temp_suffix());
    let tmp_guard = TempFileGuard::new(tmp_out.clone());

    let out_file = File::create(&tmp_out).map_err(|e| {
        CeraError::Backend(format!("failed to create `{}`: {e}", tmp_out.display()))
    })?;
    let mut buf_writer = BufWriter::with_capacity(1024 * 1024 * 4, out_file);
    writer.write_header_and_tensor_info(&mut buf_writer)?;

    let expected_experts = expected_expert_count(&config);
    let mut quant_buf = Vec::new();
    let mut f32_data = Vec::new();
    let mut raw_bytes = Vec::new();
    let mut current_file_idx = usize::MAX;
    let mut current_file: Option<File> = None;
    let mut last_split_source: Option<(SourceRef, Vec<f32>)> = None;

    for pt in pending_tensors {
        materialize_pending_tensor_f32(
            &pt,
            expected_experts,
            &mut last_split_source,
            &mut f32_data,
            |src, out| {
                read_local_source_f32(
                    src,
                    &safetensors_files,
                    &mut current_file_idx,
                    &mut current_file,
                    &mut raw_bytes,
                    out,
                )
            },
        )?;
        let num_elements = f32_data.len();
        if num_elements != pt.expected_elements {
            return Err(CeraError::Backend(format!(
                "tensor element count mismatch: expected {}, got {num_elements}",
                pt.expected_elements
            )));
        }

        apply_arch_tensor_transforms(&arch, &pt.gguf_name, &mut f32_data);

        let target_size = TargetQuant::compute_tensor_bytes(pt.ggml_type, num_elements);
        quant_buf.resize(target_size, 0);

        quantize_tensor_data_with_strategy(&f32_data, pt.ggml_type, strategy, &mut quant_buf)?;
        if matches!(pt.action, LocalSourceAction::StackExperts { .. }) {
            f32_data = Vec::new();
        }
        writer.write_tensor_data(&mut buf_writer, &quant_buf)?;
    }

    buf_writer
        .flush()
        .map_err(|e| CeraError::Backend(format!("failed to flush `{}`: {e}", tmp_out.display())))?;
    if let Ok(file) = buf_writer.into_inner() {
        let _ = file.sync_all();
    }

    let _ = fs::remove_file(output_gguf_path);
    fs::rename(&tmp_out, output_gguf_path).map_err(|e| {
        CeraError::Backend(format!(
            "failed to rename to `{}`: {e}",
            output_gguf_path.display()
        ))
    })?;

    tmp_guard.disarm();
    Ok(())
}

#[cfg(feature = "remote")]
const MAX_RETRIES: u32 = 5;
#[cfg(feature = "remote")]
const BASE_RETRY_DELAY_MS: u64 = 1000;

/// Most a remote range fetch reserves up front (256 MiB); larger tensors grow as bytes arrive.
#[cfg(feature = "remote")]
const REMOTE_RESERVE_CAP: usize = 256 << 20;

#[cfg(feature = "remote")]
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

#[cfg(feature = "remote")]
fn fetch_hf_file_bytes(
    client: &reqwest::blocking::Client,
    url: &str,
    auth_token: Option<&str>,
) -> Result<Vec<u8>, CeraError> {
    fetch_optional_hf_file_bytes(client, url, auth_token)?.ok_or_else(|| {
        CeraError::Backend(format!(
            "HTTP {} when fetching `{url}`",
            reqwest::StatusCode::NOT_FOUND
        ))
    })
}

/// Like [`fetch_hf_file_bytes`], but a 404 is `Ok(None)`: the file is absent from the
/// repo. Every other failure (retries exhausted, 401/403, a dropped connection) stays
/// an error, so a file that exists but could not be read is never mistaken for one
/// that is not there.
#[cfg(feature = "remote")]
fn fetch_optional_hf_file_bytes(
    client: &reqwest::blocking::Client,
    url: &str,
    auth_token: Option<&str>,
) -> Result<Option<Vec<u8>>, CeraError> {
    let mut last_error = String::new();
    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            let backoff_ms = BASE_RETRY_DELAY_MS * (1u64 << (attempt - 1));
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
        }

        let mut req = client.get(url);
        if let Some(token) = auth_token {
            req = req.bearer_auth(token);
        }

        match req.send() {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let bytes = resp
                        .bytes()
                        .map_err(|e| CeraError::Backend(format!("reading `{url}` bytes: {e}")))?;
                    return Ok(Some(bytes.to_vec()));
                } else if status == reqwest::StatusCode::NOT_FOUND {
                    return Ok(None);
                } else if is_retryable_status(status) {
                    last_error = format!("HTTP {status} when fetching `{url}`");
                } else {
                    return Err(CeraError::Backend(format!(
                        "HTTP {status} when fetching `{url}`"
                    )));
                }
            }
            Err(e) => {
                last_error = format!("connection error for `{url}`: {e}");
            }
        }
    }

    Err(CeraError::Backend(format!(
        "failed to fetch `{url}` after {MAX_RETRIES} attempts: {last_error}"
    )))
}

#[cfg(feature = "remote")]
fn fetch_hf_file_range_into(
    client: &reqwest::blocking::Client,
    url: &str,
    start: usize,
    end: usize,
    auth_token: Option<&str>,
    out: &mut Vec<u8>,
) -> Result<(), CeraError> {
    out.clear();
    if start > end {
        return Ok(());
    }

    let range_val = format!("bytes={start}-{end}");

    let mut last_error = String::new();
    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            let backoff_ms = BASE_RETRY_DELAY_MS * (1u64 << (attempt - 1));
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
        }

        let mut req = client.get(url).header("Range", &range_val);
        if let Some(token) = auth_token {
            req = req.bearer_auth(token);
        }

        match req.send() {
            Ok(mut resp) => {
                let status = resp.status();
                if status == reqwest::StatusCode::PARTIAL_CONTENT {
                    let expected_range_len = end.saturating_sub(start) + 1;
                    // the range comes from the shard's own header: reserve a bounded head start
                    // and let `read_to_end` grow with the bytes that really arrive
                    out.reserve(expected_range_len.min(REMOTE_RESERVE_CAP));
                    let mut reader =
                        std::io::Read::take(&mut resp, (expected_range_len + 1) as u64);
                    match std::io::Read::read_to_end(&mut reader, out) {
                        Ok(n) => {
                            if n != expected_range_len {
                                last_error = format!(
                                    "truncated range body from `{url}`: expected {expected_range_len} bytes, got {n}"
                                );
                                out.clear();
                                continue;
                            }
                            return Ok(());
                        }
                        Err(e) => {
                            last_error = format!("failed reading range body from `{url}`: {e}");
                            out.clear();
                        }
                    }
                } else if status == reqwest::StatusCode::OK {
                    // Server returned entire body instead of range; guard against unbounded memory consumption
                    let expected_range_len = end.saturating_sub(start) + 1;
                    let max_allowed = (expected_range_len * 10).max(50 * 1024 * 1024);
                    if let Some(content_len) = resp.content_length()
                        && content_len > max_allowed as u64
                    {
                        return Err(CeraError::Backend(format!(
                            "server returned full HTTP 200 body ({content_len} bytes) instead of HTTP 206 Partial Content for range {start}-{end} on `{url}`"
                        )));
                    }
                    let mut reader = std::io::Read::take(resp, max_allowed as u64 + 1);
                    let mut body_bytes = Vec::new();
                    match std::io::Read::read_to_end(&mut reader, &mut body_bytes) {
                        Ok(n) => {
                            if n > max_allowed {
                                return Err(CeraError::Backend(format!(
                                    "server returned full HTTP 200 stream exceeding {max_allowed} bytes instead of HTTP 206 Partial Content for range {start}-{end} on `{url}`"
                                )));
                            }
                            if body_bytes.len() <= end {
                                last_error = format!(
                                    "HTTP 200 OK response from `{url}` too short ({} bytes) for range {start}-{end}",
                                    body_bytes.len()
                                );
                                continue;
                            }
                            out.extend_from_slice(&body_bytes[start..=end]);
                            return Ok(());
                        }
                        Err(e) => last_error = format!("failed reading body from `{url}`: {e}"),
                    }
                } else if is_retryable_status(status) {
                    last_error = format!("HTTP {status} for range request on `{url}`");
                } else {
                    return Err(CeraError::Backend(format!(
                        "HTTP {status} for range request on `{url}`"
                    )));
                }
            }
            Err(e) => {
                last_error = format!("connection error for range request on `{url}`: {e}");
            }
        }
    }

    Err(CeraError::Backend(format!(
        "HTTP range request `{range_val}` to `{url}` failed after {MAX_RETRIES} attempts: {last_error}"
    )))
}

#[cfg(feature = "remote")]
fn fetch_hf_file_range(
    client: &reqwest::blocking::Client,
    url: &str,
    start: usize,
    end: usize,
    auth_token: Option<&str>,
) -> Result<Vec<u8>, CeraError> {
    let mut buf = Vec::new();
    fetch_hf_file_range_into(client, url, start, end, auth_token, &mut buf)?;
    Ok(buf)
}

/// Whether a lower-cased checkpoint file belongs to a side model (projector, vision tower,
/// audio decoder or detokenizer, vocoder), not the language model being converted. The
/// local and the streaming converter filter shards with this one rule.
pub(crate) fn is_auxiliary_shard(lower: &str) -> bool {
    [
        "mmproj",
        "visual",
        "audio_decoder",
        "detokenizer",
        "vocoder",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_span_past_the_end_of_its_shard_is_refused_before_allocating() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("model.safetensors");
        fs::write(&shard, [0u8; 16]).unwrap();
        // the span comes from the shard's own header: usize::MAX bytes would abort on allocation
        let src = SourceRef {
            file_idx: 0,
            dtype: "F32".into(),
            data_start: 8,
            data_end: usize::MAX,
        };
        let (mut idx, mut file, mut raw, mut out) = (usize::MAX, None, Vec::new(), Vec::new());
        let err = read_local_source_f32(&src, &[shard], &mut idx, &mut file, &mut raw, &mut out)
            .unwrap_err();
        assert!(err.to_string().contains("past the end"), "{err}");
    }

    #[test]
    fn qkv_plan_refuses_head_counts_whose_rows_overflow() {
        let config = HfModelConfig::from_json_str(
            r#"{"num_attention_heads": 2, "head_dim": 64,
                "num_key_value_heads": 9223372036854775808, "hidden_size": 128}"#,
        )
        .unwrap();
        let source = SourceRef {
            file_idx: 0,
            dtype: "F32".into(),
            data_start: 0,
            data_end: 0,
        };
        let err = plan_split_lfm3_vl_qkv(&[128, 4], 0, &config, source).unwrap_err();
        assert!(err.to_string().contains("fused qkv"), "{err}");
    }

    #[test]
    fn expert_count_follows_the_metadata_writer_lookup() {
        let count =
            |json: &str| expected_expert_count(&HfModelConfig::from_json_str(json).unwrap());
        // a null top-level key falls through to `text_config`, as the writer's lookup does
        assert_eq!(
            count(r#"{"num_experts": null, "text_config": {"num_experts": 4}}"#),
            Some(4)
        );
        // `num_experts` anywhere beats `n_routed_experts` at the top level
        assert_eq!(
            count(r#"{"n_routed_experts": 4, "text_config": {"num_experts": 8}}"#),
            Some(8)
        );
        assert_eq!(
            count(r#"{"num_experts": 8, "n_routed_experts": 4}"#),
            Some(8)
        );
        // zero experts means a dense model: there is no count to enforce
        assert_eq!(count(r#"{"num_experts": 0}"#), None);
        assert_eq!(count(r#"{"hidden_size": 8}"#), None);
    }

    /// Side-model files (the LFM3 audio bundle ships a `detokenizer` next to its language
    /// model) are not shards of the model being converted. The converters' call sites are
    /// pinned by the detokenizer sibling in the lfm3 local and streaming fixtures; the
    /// full-path cases here are the forms only the streaming converter sees.
    #[test]
    fn side_model_files_are_not_language_model_shards() {
        use super::is_auxiliary_shard;
        for aux in [
            "mmproj-model.safetensors",
            "vision/visual.safetensors",
            "audio_decoder/model.safetensors",
            "detokenizer/model.safetensors",
            "vocoder-model.safetensors",
        ] {
            assert!(is_auxiliary_shard(aux), "{aux}");
        }
        for shard in ["model.safetensors", "model-00001-of-00002.safetensors"] {
            assert!(!is_auxiliary_shard(shard), "{shard}");
        }
    }

    #[test]
    fn a_template_file_is_absent_only_when_it_is_not_there() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chat_template.jinja");
        assert_eq!(read_optional_template(&path).unwrap(), None);
        fs::write(&path, "{{ x }}").unwrap();
        assert_eq!(
            read_optional_template(&path).unwrap().as_deref(),
            Some("{{ x }}")
        );
        // a byte order mark is not part of the template
        fs::write(&path, b"\xef\xbb\xbf{{ x }}").unwrap();
        assert_eq!(
            read_optional_template(&path).unwrap().as_deref(),
            Some("{{ x }}")
        );
        // present but unreadable as text: an error, not "no template"
        fs::write(&path, [0xff, 0xfe]).unwrap();
        assert!(read_optional_template(&path).is_err());
        // a directory in its place is not "not found" either
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(read_optional_template(&path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_is_unreadable_not_absent() {
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("tokenizer_config.json");
        std::os::unix::fs::symlink(dir.path().join("pruned-blob"), &link).unwrap();
        // a Hugging Face cache snapshot whose blob is gone
        assert!(read_optional_file(&link).is_err());
        assert_eq!(
            read_optional_file(&dir.path().join("never-there")).unwrap(),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_path_under_a_regular_file_is_an_error_not_absent() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        fs::write(&file, "x").unwrap();
        // ENOTDIR: the file may well be there, the path just cannot be walked
        assert!(read_optional_file(&file.join("chat_template.jinja")).is_err());
    }

    #[test]
    fn a_blank_template_is_no_template() {
        let blank = serde_json::json!({"chat_template": "  "});
        // a blank one in tokenizer_config.json falls through to the standalone file
        assert_eq!(config_chat_template(Some(&blank)), None);
        assert_eq!(
            resolve_chat_template(Some(&blank), Some("{{ x }}".into())).as_deref(),
            Some("{{ x }}")
        );
        // and a blank standalone file is nothing
        assert_eq!(resolve_chat_template(None, Some(" \n".into())), None);
        // a real one in tokenizer_config.json wins
        let real = serde_json::json!({"chat_template": "{{ y }}"});
        assert_eq!(
            resolve_chat_template(Some(&real), Some("{{ x }}".into())).as_deref(),
            Some("{{ y }}")
        );
    }

    #[test]
    fn tokenizer_config_tolerates_a_byte_order_mark_and_explains_other_failures() {
        let with_bom = [&b"\xef\xbb\xbf"[..], br#"{"bos_token": "<s>"}"#].concat();
        assert_eq!(
            parse_tokenizer_config(&with_bom, "t").unwrap()["bos_token"],
            "<s>"
        );
        let err = parse_tokenizer_config(b"{\"x\": Infinity}", "tok.json").unwrap_err();
        let err = err.to_string();
        // names the file, carries the parser's position, and hints at the cause
        assert!(err.contains("tok.json") && err.contains("line 1"), "{err}");
        assert!(err.contains("Infinity"), "{err}");
    }

    #[test]
    fn a_tokenizer_with_an_id_past_vocab_size_is_refused_for_lfm2_only() {
        let tokenizer = |max_id: u32| {
            HfTokenizerJson::parse_from_bytes(
                serde_json::json!({
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": max_id}, "merges": []},
                })
                .to_string()
                .as_bytes(),
            )
            .unwrap()
        };
        let config = |model_type: &str| {
            HfModelConfig::from_json_str(
                &serde_json::json!({"model_type": model_type, "vocab_size": 8}).to_string(),
            )
            .unwrap()
        };
        // the last row is id 7; id 8 is one past the embedding
        assert!(ensure_vocab_fits(&config("lfm2"), &tokenizer(7)).is_ok());
        assert!(ensure_vocab_fits(&config("lfm2"), &tokenizer(8)).is_err());
        // other architectures keep their unchecked behaviour
        assert!(ensure_vocab_fits(&config("llama"), &tokenizer(8)).is_ok());
    }

    #[test]
    fn test_extract_chat_template_string_and_array() {
        // 1. Direct string template
        let str_val: serde_json::Value = serde_json::json!({ "chat_template": "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}" });
        assert_eq!(
            extract_chat_template(&str_val),
            Some(
                "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
                    .to_string()
            )
        );

        // 2. Array with explicit default name
        let arr_default: serde_json::Value = serde_json::json!({
            "chat_template": [
                { "name": "tool_use", "template": "tools" },
                { "name": "default", "template": "default_template" }
            ]
        });
        assert_eq!(
            extract_chat_template(&arr_default),
            Some("default_template".to_string())
        );

        // 3. Array without default name falls back to first
        let arr_first: serde_json::Value = serde_json::json!({
            "chat_template": [
                { "name": "custom", "template": "custom_template" }
            ]
        });
        assert_eq!(
            extract_chat_template(&arr_first),
            Some("custom_template".to_string())
        );

        // 4. Missing or invalid
        let empty_val: serde_json::Value = serde_json::json!({});
        assert_eq!(extract_chat_template(&empty_val), None);
    }

    #[test]
    fn fold_stats_match_needs_four_equal_spans() {
        let even = vec![vec![0.0f32; 8]; 4];
        assert!(fold_stats_match(&even));
        let mut short = even.clone();
        short[2].truncate(7);
        assert!(!fold_stats_match(&short));
        assert!(!fold_stats_match(&even[..3]));
        assert!(!fold_stats_match(&[]));
        assert!(!fold_stats_match(&vec![vec![0.0f32; 8]; 5]));
    }

    #[test]
    fn the_fold_refuses_spans_that_disagree() {
        // distinct per-position spans: any permutation inside the wrapper changes the
        // output, and Shift exercises the bias/mean roles Scale ignores
        let spans = vec![
            vec![2.0f32; 4], // weight
            vec![0.5f32; 4], // bias
            vec![1.5f32; 4], // mean
            vec![3.0f32; 4], // var
        ];
        for kind in [d1::Fold::Scale, d1::Fold::Shift] {
            let folded = fold_stats_checked(kind, &spans, "t").unwrap();
            assert_eq!(
                folded,
                d1::fold_batch_norm(kind, &spans[0], &spans[1], &spans[2], &spans[3])
            );
        }
        let mut short = spans.clone();
        short[2].truncate(3);
        let err = fold_stats_checked(d1::Fold::Scale, &short, "blk.0.ffn").unwrap_err();
        assert!(
            err.to_string().contains("batch-norm statistics disagree"),
            "unexpected message: {err}"
        );
        assert!(
            err.to_string().contains("blk.0.ffn"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn lazy_assembly_matches_eager_for_vl_splits_and_stacked_experts() {
        let config = HfModelConfig::from_json_str(
            &serde_json::json!({
                "model_type": "lfm3",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "intermediate_size": 3,
                "vocab_size": 16,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "num_experts": 2,
            })
            .to_string(),
        )
        .unwrap();

        let qkv_data: Vec<f32> = (0..48).map(|v| v as f32 + 0.25).collect();
        let gate_up_data: Vec<f32> = (0..24).map(|v| (v as f32) * 1.5).collect();
        let exp0_w1: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let exp1_w1: Vec<f32> = (101..=108).map(|v| v as f32).collect();
        let exp0_w2: Vec<f32> = (201..=208).map(|v| v as f32).collect();
        let exp1_w2: Vec<f32> = (301..=308).map(|v| v as f32).collect();
        let exp0_w3: Vec<f32> = (401..=408).map(|v| v as f32).collect();
        let exp1_w3: Vec<f32> = (501..=508).map(|v| v as f32).collect();
        let vision_skip: Vec<f32> = vec![9.0; 16];

        let raw_entries: Vec<(&str, Vec<usize>, &[f32])> = vec![
            (
                "vision_tower.encoder.layers.0.weight",
                vec![4, 4],
                &vision_skip,
            ),
            (
                "language_model.layers.0.self_attn.qkv.weight",
                vec![12, 4],
                &qkv_data,
            ),
            (
                "language_model.layers.0.feed_forward.gate_up.weight",
                vec![6, 4],
                &gate_up_data,
            ),
            (
                "model.layers.1.feed_forward.experts.1.w1.weight",
                vec![2, 4],
                &exp1_w1,
            ),
            (
                "model.layers.1.feed_forward.experts.0.w1.weight",
                vec![2, 4],
                &exp0_w1,
            ),
            (
                "model.layers.1.feed_forward.experts.0.w2.weight",
                vec![4, 2],
                &exp0_w2,
            ),
            (
                "model.layers.1.feed_forward.experts.1.w2.weight",
                vec![4, 2],
                &exp1_w2,
            ),
            (
                "model.layers.1.feed_forward.experts.0.w3.weight",
                vec![2, 4],
                &exp0_w3,
            ),
            (
                "model.layers.1.feed_forward.experts.1.w3.weight",
                vec![2, 4],
                &exp1_w3,
            ),
        ];

        let mut header_map = serde_json::Map::new();
        let mut payload_bytes = Vec::new();
        for (name, shape, values) in &raw_entries {
            let start = payload_bytes.len();
            for v in *values {
                payload_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let end = payload_bytes.len();
            header_map.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_json = serde_json::Value::Object(header_map).to_string();
        let header_size = 8 + header_json.len();
        let header = SafeTensorsHeader::parse_from_json_str(&header_json, header_size).unwrap();

        let mut shard_bytes = Vec::with_capacity(header_size + payload_bytes.len());
        shard_bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        shard_bytes.extend_from_slice(header_json.as_bytes());
        shard_bytes.extend_from_slice(&payload_bytes);

        let mut writer = GgufWriter::new();
        // Header planning must succeed from metadata alone before reading any tensor bytes.
        let planned = plan_pending_tensors(
            [(0usize, &header)],
            &config,
            "lfm3",
            false,
            TargetQuant::F32,
            &[],
            &mut writer,
        )
        .unwrap();

        assert!(
            !planned.iter().any(|pt| pt.gguf_name.contains("vision")),
            "multimodal tensor should be skipped"
        );

        let mut read_counts = std::collections::BTreeMap::<(usize, usize), usize>::new();
        let mut last_split_source = None;
        let mut materialized = std::collections::BTreeMap::<String, Vec<f32>>::new();
        for pt in &planned {
            let mut f32_data = Vec::new();
            materialize_pending_tensor_f32(
                pt,
                expected_expert_count(&config),
                &mut last_split_source,
                &mut f32_data,
                |src, out| {
                    *read_counts
                        .entry((src.data_start, src.data_end))
                        .or_insert(0) += 1;
                    decode_safetensor_to_f32_into(
                        &shard_bytes[src.data_start..src.data_end],
                        &src.dtype,
                        out,
                    )
                },
            )
            .unwrap();
            assert_eq!(f32_data.len(), pt.expected_elements, "{}", pt.gguf_name);
            if pt.gguf_name == "blk.0.attn_v.weight" || pt.gguf_name == "blk.0.ffn_up.weight" {
                assert!(
                    last_split_source.is_none(),
                    "split cache should be dropped after final slice of {}",
                    pt.gguf_name
                );
            }
            materialized.insert(pt.gguf_name.clone(), f32_data);
        }

        // Each source span in the shard should be decoded only once across the entire pass.
        for (&span, &count) in &read_counts {
            assert_eq!(count, 1, "source span {span:?} was decoded {count} times");
        }

        let eager_qkv = split_lfm3_vl_qkv(&qkv_data, &[12, 4], 0, 4, 1, 2).unwrap();
        for (name, shape, data) in eager_qkv {
            assert_eq!(materialized.get(&name), Some(&data), "{name}");
            let info = writer.tensors().iter().find(|t| t.name == name).unwrap();
            assert_eq!(info.dims, gguf_tensor_dims(&name, &shape));
        }

        let eager_gu = split_lfm3_vl_gate_up(&gate_up_data, &[6, 4], 0).unwrap();
        for (name, shape, data) in eager_gu {
            assert_eq!(materialized.get(&name), Some(&data), "{name}");
            let info = writer.tensors().iter().find(|t| t.name == name).unwrap();
            assert_eq!(info.dims, gguf_tensor_dims(&name, &shape));
        }

        for (gguf_name, exp_shape, e0, e1) in [
            ("blk.1.ffn_gate_exps.weight", vec![2, 4], &exp0_w1, &exp1_w1),
            ("blk.1.ffn_down_exps.weight", vec![4, 2], &exp0_w2, &exp1_w2),
            ("blk.1.ffn_up_exps.weight", vec![2, 4], &exp0_w3, &exp1_w3),
        ] {
            let mut raw = vec![
                (1, exp_shape.clone(), e1.clone()),
                (0, exp_shape.clone(), e0.clone()),
            ];
            let (expected_shape, expected_data) =
                stack_expert_tensors(&mut raw, Some(2), gguf_name).unwrap();
            assert_eq!(
                materialized.get(gguf_name),
                Some(&expected_data),
                "{gguf_name}"
            );
            let info = writer
                .tensors()
                .iter()
                .find(|t| t.name == gguf_name)
                .unwrap();
            assert_eq!(info.dims, gguf_tensor_dims(gguf_name, &expected_shape));
        }
    }
}

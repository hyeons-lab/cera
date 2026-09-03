//! Streaming SafeTensors to GGUF quantization pipeline with HTTP retries & checkpoint resumption.

use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::bundle::{DownloadProgress, HfSpec, fetch_model_info};
use crate::convert::config::HfModelConfig;
use crate::convert::quantize::{QuantStrategy, TargetQuant, quantize_tensor_data_with_strategy};
use crate::convert::safetensors::{
    SafeTensorsHeader, decode_safetensor_to_f32_into, translate_hf_to_gguf_tensor_name,
};
use crate::convert::tokenizer::HfTokenizerJson;
use crate::convert::writer::GgufWriter;
use crate::manifest::{GenerationDefaults, InferenceType, Manifest, ManifestFiles};
use crate::session::CeraError;

/// Options controlling model quantization and caching.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantCheckpoint {
    quant: String,
    strategy: String,
    completed_tensors: usize,
    total_tensors: usize,
    file_bytes: u64,
}

/// Stream and quantize a remote Hugging Face SafeTensors repository into a cached GGUF model.
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

    // Check if previously converted and cached
    if manifest_path.exists()
        && target_gguf_path.exists()
        && let Ok(manifest_text) = fs::read_to_string(&manifest_path)
        && let Ok(mut manifest) = Manifest::from_bytes(manifest_text.as_bytes())
    {
        manifest.files.model = target_gguf_path.to_string_lossy().to_string();
        return Ok(manifest);
    }

    fs::create_dir_all(&store_dir).map_err(|e| {
        CeraError::Backend(format!(
            "failed to create cache dir `{}`: {e}",
            store_dir.display()
        ))
    })?;

    // 1. Fetch Repo Metadata from HF API (with automatic retries)
    let info = fetch_model_info(spec)?;

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| CeraError::Backend(format!("failed to build reqwest client: {e}")))?;

    // 1. Fetch config.json, tokenizer.json, and optional generation_config.json
    let config_url = spec.file_download_url("config.json");
    let config_bytes = fetch_hf_file_bytes(&client, &config_url, opts.auth_token.as_deref())?;
    let config = HfModelConfig::parse_from_bytes(&config_bytes)?;

    let tokenizer_url = spec.file_download_url("tokenizer.json");
    let tokenizer_bytes = fetch_hf_file_bytes(&client, &tokenizer_url, opts.auth_token.as_deref())?;
    let tokenizer = HfTokenizerJson::parse_from_bytes(&tokenizer_bytes)?;

    // Optional chat template & generation config
    let template_url = spec.file_download_url("tokenizer_config.json");
    let chat_template = fetch_hf_file_bytes(&client, &template_url, opts.auth_token.as_deref())
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("chat_template")
                .and_then(|t| t.as_str().map(str::to_string))
        });

    let gen_url = spec.file_download_url("generation_config.json");
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
            lower.ends_with(".safetensors")
                && !lower.contains("mmproj")
                && !lower.contains("visual")
                && !lower.contains("audio_decoder")
                && !lower.contains("vocoder")
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
        let header_url = spec.file_download_url(file_name);
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
    tokenizer.apply_to_gguf_writer(&mut writer, chat_template.as_deref());

    // Register all tensors in GGUF writer
    struct PendingTensor {
        shard_idx: usize,
        name: String,
        dtype: String,
        data_start: usize,
        data_end: usize,
        ggml_type: u32,
        expected_elements: usize,
    }

    let mut pending_tensors = Vec::new();

    for (shard_idx, (_file_name, header)) in shard_headers.iter().enumerate() {
        for (tensor_name, tensor_info) in &header.tensors {
            let gguf_name = translate_hf_to_gguf_tensor_name(tensor_name);
            let num_elements: usize = tensor_info
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor shape overflow for `{tensor_name}`"))
                })?;
            let ggml_type = opts.target_quant.select_ggml_type_with_overrides(
                &gguf_name,
                tensor_info.shape.len(),
                num_elements,
                &opts.tensor_overrides,
            );
            let out_bytes = TargetQuant::compute_tensor_bytes(ggml_type, num_elements);

            // In GGUF, dimensions are column-major (reversed shape: [cols, rows])
            let dims: Vec<u64> = tensor_info.shape.iter().rev().map(|&d| d as u64).collect();

            writer.add_tensor(&gguf_name, dims, ggml_type, out_bytes);

            let data_start = header
                .header_size_bytes
                .checked_add(tensor_info.data_offsets.0)
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor offset start overflow for `{tensor_name}`"))
                })?;
            let data_end = header
                .header_size_bytes
                .checked_add(tensor_info.data_offsets.1)
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor offset end overflow for `{tensor_name}`"))
                })?;

            pending_tensors.push(PendingTensor {
                shard_idx,
                name: gguf_name,
                dtype: tensor_info.dtype.clone(),
                data_start,
                data_end,
                ggml_type,
                expected_elements: num_elements,
            });
        }
    }

    let total_tensors = pending_tensors.len();

    // 6. Check for valid existing checkpoint to resume from
    let mut start_tensor_idx = 0;
    let mut should_write_header = true;
    let mut resume_bytes = 0u64;

    if tmp_gguf_path.exists()
        && ckpt_path.exists()
        && let Ok(ckpt_str) = fs::read_to_string(&ckpt_path)
        && let Ok(ckpt) = serde_json::from_str::<QuantCheckpoint>(&ckpt_str)
        && ckpt.quant == quant_str
        && ckpt.strategy == strat_str
        && ckpt.total_tensors == total_tensors
        && ckpt.completed_tensors < total_tensors
        && let Ok(meta) = fs::metadata(&tmp_gguf_path)
        && meta.len() >= ckpt.file_bytes
    {
        start_tensor_idx = ckpt.completed_tensors;
        should_write_header = false;
        resume_bytes = ckpt.file_bytes;
        tracing::info!(
            "Resuming quantization of `{}/{}` ({quant_str}) from tensor {}/{}…",
            spec.owner,
            spec.repo,
            start_tensor_idx,
            total_tensors
        );
    }

    // Open file (create fresh or append for resume)
    let (out_file, mut current_bytes_written) = if should_write_header {
        let _ = fs::remove_file(&ckpt_path);
        let file = File::create(&tmp_gguf_path).map_err(|e| {
            CeraError::Backend(format!(
                "failed to create `{}`: {e}",
                tmp_gguf_path.display()
            ))
        })?;
        (file, 0u64)
    } else {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tmp_gguf_path)
            .map_err(|e| {
                CeraError::Backend(format!(
                    "failed to open `{}` for resume: {e}",
                    tmp_gguf_path.display()
                ))
            })?;
        file.set_len(resume_bytes).map_err(|e| {
            CeraError::Backend(format!(
                "failed to truncate `{}` to resume boundary {resume_bytes} bytes: {e}",
                tmp_gguf_path.display()
            ))
        })?;
        let len = file.seek(SeekFrom::Start(resume_bytes)).map_err(|e| {
            CeraError::Backend(format!(
                "failed to seek `{}` to resume boundary: {e}",
                tmp_gguf_path.display()
            ))
        })?;
        (file, len)
    };

    let mut buf_writer = BufWriter::with_capacity(1024 * 1024 * 4, out_file);

    if should_write_header {
        let _data_start_offset = writer.write_header_and_tensor_info(&mut buf_writer)?;
        buf_writer.flush().map_err(|e| {
            CeraError::Backend(format!(
                "failed to flush `{}`: {e}",
                tmp_gguf_path.display()
            ))
        })?;
        current_bytes_written = fs::metadata(&tmp_gguf_path).map(|m| m.len()).unwrap_or(0);
    }

    // Estimate total output file bytes accurately
    let align = writer.alignment() as u64;
    let align_mask = align - 1;
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
        .map(|(name, _)| spec.file_download_url(name))
        .collect();

    let mut quant_buf = Vec::new();
    let mut f32_data = Vec::new();
    let mut raw_tensor_bytes = Vec::new();

    for (i, pt) in pending_tensors.iter().enumerate().skip(start_tensor_idx) {
        if let Some(ref cancel) = opts.cancel
            && cancel.load(std::sync::atomic::Ordering::Relaxed)
        {
            let _ = buf_writer.flush();
            return Err(CeraError::Cancelled);
        }

        let tensor_url = &shard_download_urls[pt.shard_idx];

        if let Some(p) = &opts.progress {
            p.on_progress(
                tensor_url,
                current_bytes_written,
                Some(total_estimated_bytes),
            );
        }

        // Fetch raw tensor bytes over HTTP Range with automatic retry and exponential backoff
        if pt.data_start < pt.data_end {
            fetch_hf_file_range_into(
                &client,
                tensor_url,
                pt.data_start,
                pt.data_end - 1,
                opts.auth_token.as_deref(),
                &mut raw_tensor_bytes,
            )?;
        } else {
            raw_tensor_bytes.clear();
        }

        // Decode BF16/F16/F32 to f32
        decode_safetensor_to_f32_into(&raw_tensor_bytes, &pt.dtype, &mut f32_data)?;
        let num_elements = f32_data.len();
        if num_elements != pt.expected_elements {
            return Err(CeraError::Backend(format!(
                "tensor `{}` element count mismatch: expected {}, got {num_elements}",
                pt.name, pt.expected_elements
            )));
        }

        // Quantize to target GGML type
        let target_size = TargetQuant::compute_tensor_bytes(pt.ggml_type, num_elements);
        quant_buf.resize(target_size, 0);

        quantize_tensor_data_with_strategy(&f32_data, pt.ggml_type, opts.strategy, &mut quant_buf)?;

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
            &spec.file_download_url("model.gguf"),
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
    if let Ok(file) = buf_writer.into_inner() {
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
    quantize_safetensors_to_gguf_with_overrides(input_path, output_gguf_path, quant, &[])
}

/// Quantize a local SafeTensors directory or file to GGUF with per-tensor quantization overrides.
pub fn quantize_safetensors_to_gguf_with_overrides(
    input_path: &Path,
    output_gguf_path: &Path,
    quant: TargetQuant,
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

    let mut writer = GgufWriter::new();
    let model_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    config.apply_to_gguf_writer(&mut writer, model_name);

    if tokenizer_path.exists() {
        let tok_bytes = fs::read(&tokenizer_path).map_err(|e| {
            CeraError::Backend(format!(
                "failed to read `{}`: {e}",
                tokenizer_path.display()
            ))
        })?;
        if let Ok(tokenizer) = HfTokenizerJson::parse_from_bytes(&tok_bytes) {
            let chat_template = dir
                .join("tokenizer_config.json")
                .exists()
                .then(|| fs::read(dir.join("tokenizer_config.json")).ok())
                .flatten()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| {
                    v.get("chat_template")
                        .and_then(|t| t.as_str().map(str::to_string))
                });
            tokenizer.apply_to_gguf_writer(&mut writer, chat_template.as_deref());
        }
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
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
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

    struct LocalPendingTensor {
        file_idx: usize,
        expected_elements: usize,
        dtype: String,
        data_start: usize,
        data_end: usize,
        ggml_type: u32,
    }

    let mut pending_tensors = Vec::new();

    for (file_idx, file_path) in safetensors_files.iter().enumerate() {
        let mut file = File::open(file_path).map_err(|e| {
            CeraError::Backend(format!("failed to open `{}`: {e}", file_path.display()))
        })?;
        let header = SafeTensorsHeader::parse_from_reader(&mut file)?;
        for (raw_name, tensor_info) in &header.tensors {
            let gguf_name = translate_hf_to_gguf_tensor_name(raw_name);
            let num_elements: usize = tensor_info
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| {
                    CeraError::Backend(format!("tensor shape overflow for `{raw_name}`"))
                })?;
            let ggml_type = quant.select_ggml_type_with_overrides(
                &gguf_name,
                tensor_info.shape.len(),
                num_elements,
                overrides,
            );

            let dims: Vec<u64> = tensor_info.shape.iter().rev().map(|&d| d as u64).collect();
            let out_bytes = TargetQuant::compute_tensor_bytes(ggml_type, num_elements);

            writer.add_tensor(&gguf_name, dims, ggml_type, out_bytes);

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

            pending_tensors.push(LocalPendingTensor {
                file_idx,
                expected_elements: num_elements,
                dtype: tensor_info.dtype.clone(),
                data_start,
                data_end,
                ggml_type,
            });
        }
    }

    let tmp_out = output_gguf_path.with_extension(unique_temp_suffix());
    let tmp_guard = TempFileGuard::new(tmp_out.clone());

    let out_file = File::create(&tmp_out).map_err(|e| {
        CeraError::Backend(format!("failed to create `{}`: {e}", tmp_out.display()))
    })?;
    let mut buf_writer = BufWriter::with_capacity(1024 * 1024 * 4, out_file);
    writer.write_header_and_tensor_info(&mut buf_writer)?;

    let mut quant_buf = Vec::new();
    let mut f32_data = Vec::new();
    let mut raw_bytes = Vec::new();
    let mut current_file_idx = usize::MAX;
    let mut current_file: Option<File> = None;

    for pt in pending_tensors {
        if pt.file_idx != current_file_idx {
            let f = File::open(&safetensors_files[pt.file_idx]).map_err(|e| {
                CeraError::Backend(format!(
                    "failed to open `{}`: {e}",
                    safetensors_files[pt.file_idx].display()
                ))
            })?;
            current_file = Some(f);
            current_file_idx = pt.file_idx;
        }

        let file = current_file
            .as_mut()
            .ok_or_else(|| CeraError::Backend("no active safetensors shard file".into()))?;

        let byte_len = pt.data_end.saturating_sub(pt.data_start);
        file.seek(SeekFrom::Start(pt.data_start as u64))
            .map_err(|e| CeraError::Backend(format!("failed to seek in safetensors shard: {e}")))?;

        raw_bytes.resize(byte_len, 0);
        file.read_exact(&mut raw_bytes).map_err(|e| {
            CeraError::Backend(format!(
                "failed to read tensor data from safetensors shard: {e}"
            ))
        })?;

        decode_safetensor_to_f32_into(&raw_bytes, &pt.dtype, &mut f32_data)?;
        let num_elements = f32_data.len();
        if num_elements != pt.expected_elements {
            return Err(CeraError::Backend(format!(
                "tensor element count mismatch: expected {}, got {num_elements}",
                pt.expected_elements
            )));
        }

        let target_size = TargetQuant::compute_tensor_bytes(pt.ggml_type, num_elements);
        quant_buf.resize(target_size, 0);

        quantize_tensor_data_with_strategy(
            &f32_data,
            pt.ggml_type,
            QuantStrategy::Auto,
            &mut quant_buf,
        )?;
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

const MAX_RETRIES: u32 = 5;
const BASE_RETRY_DELAY_MS: u64 = 1000;

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn fetch_hf_file_bytes(
    client: &reqwest::blocking::Client,
    url: &str,
    auth_token: Option<&str>,
) -> Result<Vec<u8>, CeraError> {
    let mut last_error = String::new();
    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            let delay_ms = BASE_RETRY_DELAY_MS * (1 << (attempt - 1));
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }

        let mut req = client.get(url);
        if let Some(token) = auth_token {
            req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"));
        }

        match req.send() {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    match resp.bytes() {
                        Ok(b) => return Ok(b.to_vec()),
                        Err(e) => last_error = format!("failed reading body from `{url}`: {e}"),
                    }
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
            let delay_ms = BASE_RETRY_DELAY_MS * (1 << (attempt - 1));
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }

        let mut req = client.get(url).header(reqwest::header::RANGE, &range_val);
        if let Some(token) = auth_token {
            req = req.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"));
        }

        match req.send() {
            Ok(mut resp) => {
                let status = resp.status();
                if status == reqwest::StatusCode::PARTIAL_CONTENT {
                    let expected_len = end.saturating_sub(start) + 1;
                    out.reserve(expected_len);
                    let mut reader = std::io::Read::take(&mut resp, (expected_len + 1) as u64);
                    match std::io::Read::read_to_end(&mut reader, out) {
                        Ok(n) => {
                            if n != expected_len {
                                last_error = format!(
                                    "truncated range body from `{url}`: expected {expected_len} bytes, got {n}"
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

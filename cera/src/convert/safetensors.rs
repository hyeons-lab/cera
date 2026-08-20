//! SafeTensors header parser, tensor name translation, and shard index resolution.

use crate::quant::{bf16_to_f32, f16_to_f32};
use crate::session::CeraError;
use serde::Deserialize;
use std::collections::BTreeMap;

/// Individual tensor header in a SafeTensors file.
#[derive(Debug, Clone, Deserialize)]
pub struct SafeTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Parsed header of a single `.safetensors` file.
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    pub header_size_bytes: usize,
    pub tensors: BTreeMap<String, SafeTensorInfo>,
}

impl SafeTensorsHeader {
    /// Parse the header from the start of a SafeTensors file/buffer.
    ///
    /// Returns the parsed header and the total bytes of header (8 + N).
    pub fn parse_from_bytes(bytes: &[u8]) -> Result<Self, CeraError> {
        if bytes.len() < 8 {
            return Err(CeraError::Backend(
                "invalid safetensors: buffer smaller than 8 bytes".into(),
            ));
        }

        let len_bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| {
            CeraError::Backend("failed to read 8-byte safetensors header length".into())
        })?;
        let header_len_u64 = u64::from_le_bytes(len_bytes);
        if header_len_u64 == 0 || header_len_u64 > 100_000_000 {
            return Err(CeraError::Backend(format!(
                "invalid safetensors header length: {header_len_u64}"
            )));
        }
        let header_len = header_len_u64 as usize;
        let total_header = match 8usize.checked_add(header_len) {
            Some(t) => t,
            None => {
                return Err(CeraError::Backend(
                    "safetensors header length overflows usize".into(),
                ));
            }
        };
        if bytes.len() < total_header {
            return Err(CeraError::Backend(format!(
                "invalid safetensors: buffer size ({}) smaller than 8 + header size ({header_len})",
                bytes.len()
            )));
        }

        let json_str = std::str::from_utf8(&bytes[8..total_header])
            .map_err(|e| CeraError::Backend(format!("invalid utf8 in safetensors header: {e}")))?;

        Self::parse_from_json_str(json_str, total_header)
    }

    /// Parse the header from a reader (e.g. `File`), reading only the header bytes.
    pub fn parse_from_reader<R: std::io::Read>(reader: &mut R) -> Result<Self, CeraError> {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes).map_err(|e| {
            CeraError::Backend(format!(
                "failed to read 8-byte safetensors header length: {e}"
            ))
        })?;
        let header_len_u64 = u64::from_le_bytes(len_bytes);
        if header_len_u64 == 0 || header_len_u64 > 100_000_000 {
            return Err(CeraError::Backend(format!(
                "invalid safetensors header length: {header_len_u64}"
            )));
        }
        let header_len = header_len_u64 as usize;
        let mut json_bytes = vec![0u8; header_len];
        reader.read_exact(&mut json_bytes).map_err(|e| {
            CeraError::Backend(format!(
                "failed to read {header_len} bytes of safetensors header: {e}"
            ))
        })?;
        let json_str = std::str::from_utf8(&json_bytes)
            .map_err(|e| CeraError::Backend(format!("invalid utf8 in safetensors header: {e}")))?;

        let total_header = match 8usize.checked_add(header_len) {
            Some(t) => t,
            None => {
                return Err(CeraError::Backend(
                    "safetensors header length overflows usize".into(),
                ));
            }
        };

        Self::parse_from_json_str(json_str, total_header)
    }

    /// Parse `SafeTensorsHeader` from raw JSON string and total header size.
    pub fn parse_from_json_str(json_str: &str, total_header: usize) -> Result<Self, CeraError> {
        let raw_map: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(json_str).map_err(|e| {
                CeraError::Backend(format!("failed to parse safetensors json header: {e}"))
            })?;

        let mut tensors = BTreeMap::new();
        for (name, val) in raw_map {
            if name == "__metadata__" {
                continue;
            }
            let info: SafeTensorInfo = serde_json::from_value(val).map_err(|e| {
                CeraError::Backend(format!("failed to parse tensor `{name}` metadata: {e}"))
            })?;
            tensors.insert(name, info);
        }

        Ok(Self {
            header_size_bytes: total_header,
            tensors,
        })
    }
}

/// Translate Hugging Face standard tensor names to standard GGUF tensor names.
pub fn translate_hf_to_gguf_tensor_name(hf_name: &str) -> String {
    // Direct global mappings
    if hf_name == "model.embed_tokens.weight"
        || hf_name == "transformer.wte.weight"
        || hf_name == "embeddings.word_embeddings.weight"
    {
        return "token_embd.weight".to_string();
    }
    if hf_name == "model.norm.weight"
        || hf_name == "transformer.ln_f.weight"
        || hf_name == "ln_f.weight"
    {
        return "output_norm.weight".to_string();
    }
    if hf_name == "lm_head.weight" {
        return "output.weight".to_string();
    }

    // Layer-level mappings
    let layer_rest = hf_name
        .strip_prefix("model.layers.")
        .or_else(|| hf_name.strip_prefix("transformer.h."))
        .or_else(|| hf_name.strip_prefix("layers."));
    if let Some((layer_idx, sub_name)) = layer_rest.and_then(|r| r.split_once('.')) {
        let gguf_suffix = match sub_name {
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.q_proj.bias" => "attn_q.bias",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.k_proj.bias" => "attn_k.bias",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.v_proj.bias" => "attn_v.bias",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.o_proj.bias" => "attn_output.bias",
            "self_attn.qkv_proj.weight" => "attn_qkv.weight",
            "self_attn.qkv_proj.bias" => "attn_qkv.bias",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.q_norm.bias" => "attn_q_norm.bias",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "self_attn.k_norm.bias" => "attn_k_norm.bias",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.gate_proj.bias" => "ffn_gate.bias",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.up_proj.bias" => "ffn_up.bias",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "mlp.down_proj.bias" => "ffn_down.bias",
            "input_layernorm.weight" => "attn_norm.weight",
            "input_layernorm.bias" => "attn_norm.bias",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "post_attention_layernorm.bias" => "ffn_norm.bias",
            "operator.conv.weight" => "shortconv.conv.weight",
            "operator.conv.bias" => "shortconv.conv.bias",
            "operator.in_proj.weight" => "shortconv.in_proj.weight",
            "operator.in_proj.bias" => "shortconv.in_proj.bias",
            "operator.out_proj.weight" => "shortconv.out_proj.weight",
            "operator.out_proj.bias" => "shortconv.out_proj.bias",
            _ => sub_name,
        };

        return format!("blk.{layer_idx}.{gguf_suffix}");
    }

    hf_name.to_string()
}

/// Convert raw SafeTensors tensor bytes (BF16, F16, F32) directly into an existing `f32` buffer.
pub fn decode_safetensor_to_f32_into(
    raw_bytes: &[u8],
    dtype: &str,
    out: &mut Vec<f32>,
) -> Result<(), CeraError> {
    out.clear();
    match dtype.to_ascii_uppercase().as_str() {
        "F32" => {
            if !raw_bytes.len().is_multiple_of(4) {
                return Err(CeraError::Backend(
                    "F32 tensor bytes not multiple of 4".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 4);
            let (chunks, _) = raw_bytes.as_chunks::<4>();
            out.extend(chunks.iter().map(|c| f32::from_le_bytes(*c)));
            Ok(())
        }
        "F64" => {
            if !raw_bytes.len().is_multiple_of(8) {
                return Err(CeraError::Backend(
                    "F64 tensor bytes not multiple of 8".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 8);
            let (chunks, _) = raw_bytes.as_chunks::<8>();
            out.extend(chunks.iter().map(|c| f64::from_le_bytes(*c) as f32));
            Ok(())
        }
        "BF16" => {
            if !raw_bytes.len().is_multiple_of(2) {
                return Err(CeraError::Backend(
                    "BF16 tensor bytes not multiple of 2".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 2);
            let (chunks, _) = raw_bytes.as_chunks::<2>();
            out.extend(chunks.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))));
            Ok(())
        }
        "F16" => {
            if !raw_bytes.len().is_multiple_of(2) {
                return Err(CeraError::Backend(
                    "F16 tensor bytes not multiple of 2".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 2);
            let (chunks, _) = raw_bytes.as_chunks::<2>();
            out.extend(chunks.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))));
            Ok(())
        }
        "I32" => {
            if !raw_bytes.len().is_multiple_of(4) {
                return Err(CeraError::Backend(
                    "I32 tensor bytes not multiple of 4".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 4);
            let (chunks, _) = raw_bytes.as_chunks::<4>();
            out.extend(chunks.iter().map(|c| i32::from_le_bytes(*c) as f32));
            Ok(())
        }
        "I64" => {
            if !raw_bytes.len().is_multiple_of(8) {
                return Err(CeraError::Backend(
                    "I64 tensor bytes not multiple of 8".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 8);
            let (chunks, _) = raw_bytes.as_chunks::<8>();
            out.extend(chunks.iter().map(|c| i64::from_le_bytes(*c) as f32));
            Ok(())
        }
        "U32" => {
            if !raw_bytes.len().is_multiple_of(4) {
                return Err(CeraError::Backend(
                    "U32 tensor bytes not multiple of 4".into(),
                ));
            }
            out.reserve(raw_bytes.len() / 4);
            let (chunks, _) = raw_bytes.as_chunks::<4>();
            out.extend(chunks.iter().map(|c| u32::from_le_bytes(*c) as f32));
            Ok(())
        }
        "U8" => {
            out.reserve(raw_bytes.len());
            out.extend(raw_bytes.iter().map(|&b| b as f32));
            Ok(())
        }
        "I8" => {
            out.reserve(raw_bytes.len());
            out.extend(raw_bytes.iter().map(|&b| b as i8 as f32));
            Ok(())
        }
        "BOOL" => {
            out.reserve(raw_bytes.len());
            out.extend(raw_bytes.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }));
            Ok(())
        }
        other => Err(CeraError::Backend(format!(
            "unsupported safetensors dtype: `{other}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_name_translation() {
        assert_eq!(
            translate_hf_to_gguf_tensor_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
        assert_eq!(
            translate_hf_to_gguf_tensor_name("lm_head.weight"),
            "output.weight"
        );
        assert_eq!(
            translate_hf_to_gguf_tensor_name("model.norm.weight"),
            "output_norm.weight"
        );
        assert_eq!(
            translate_hf_to_gguf_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            translate_hf_to_gguf_tensor_name("model.layers.15.mlp.down_proj.weight"),
            "blk.15.ffn_down.weight"
        );
    }
}

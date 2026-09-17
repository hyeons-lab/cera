//! Session checkpointing and binary persistence for inference and chat sessions.
//!
//! Provides self-contained serialization and restoration of live KV caches,
//! recurrent layer states, conversation history, prefill metrics, and coordinator
//! phases. Checkpoints validate model architectural compatibility using an FNV-1a
//! fingerprint before modifying any session state.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::kv_cache::StateSnapshot;
use crate::session::CeraError;
use crate::session::chat::SessionPhase;
use crate::tools::{ToolDef, ToolFormat};

#[cfg(test)]
mod tests;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_checkpoint_path(path: &Path) -> PathBuf {
    let count = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    match path.file_name() {
        Some(name) => path.with_file_name(format!(
            "{}.tmp.{}.{}",
            name.to_string_lossy(),
            std::process::id(),
            count
        )),
        None => path.with_extension(format!("tmp.{}.{}", std::process::id(), count)),
    }
}

const SESSION_CHECKPOINT_MAGIC: &[u8; 8] = b"CERASCHK";
const SESSION_CHECKPOINT_VERSION: u32 = 1;

const CHAT_CHECKPOINT_MAGIC: &[u8; 8] = b"CERACHAT";
const CHAT_CHECKPOINT_VERSION: u32 = 2;

/// A serializable, resumable snapshot of an inference session's state.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionCheckpoint {
    /// Model fingerprint to verify structural architecture compatibility upon restore.
    pub model_fingerprint: u64,
    /// Total tokens currently resident in the KV cache.
    pub position: usize,
    /// Maximum context capacity.
    pub max_seq_len: usize,
    /// Prefill token count metric.
    pub prefill_tokens: u32,
    /// Prefill duration in milliseconds.
    pub prefill_elapsed_ms: u64,
    /// Last step logits if available.
    pub last_logits: Option<Vec<f32>>,
    /// Token history for speculative decoding lookup.
    pub token_history: Vec<u32>,
    /// Underlying KV cache and recurrent layer states.
    pub kv_state: StateSnapshot,
}

impl SessionCheckpoint {
    /// Serialize this checkpoint into a compact binary representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut writer = BufferWriter::with_capacity(self.byte_size() + 128);
        writer.write_bytes(SESSION_CHECKPOINT_MAGIC);
        writer.write_u32(SESSION_CHECKPOINT_VERSION);
        writer.write_u64(self.model_fingerprint);
        writer.write_u64(self.position as u64);
        writer.write_u64(self.max_seq_len as u64);
        writer.write_u32(self.prefill_tokens);
        writer.write_u64(self.prefill_elapsed_ms);

        if let Some(logits) = &self.last_logits {
            writer.write_u8(1);
            writer.write_u32(logits.len() as u32);
            for &val in logits {
                writer.write_bytes(&val.to_le_bytes());
            }
        } else {
            writer.write_u8(0);
        }

        writer.write_u32(self.token_history.len() as u32);
        for &tok in &self.token_history {
            writer.write_u32(tok);
        }

        write_state_snapshot(&mut writer, &self.kv_state);
        writer.buf
    }

    /// Deserialize a checkpoint from a compact binary slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CeraError> {
        let mut reader = BufferReader::new(bytes);
        let magic = reader.read_bytes(8)?;
        if magic != SESSION_CHECKPOINT_MAGIC {
            return Err(CeraError::Format(
                "invalid session checkpoint magic bytes".to_string(),
            ));
        }
        let version = reader.read_u32()?;
        if version != SESSION_CHECKPOINT_VERSION {
            return Err(CeraError::Format(format!(
                "unsupported session checkpoint version {version}, expected {SESSION_CHECKPOINT_VERSION}"
            )));
        }
        let model_fingerprint = reader.read_u64()?;
        let position = reader.read_u64()? as usize;
        let max_seq_len = reader.read_u64()? as usize;
        let prefill_tokens = reader.read_u32()?;
        let prefill_elapsed_ms = reader.read_u64()?;

        let has_logits = reader.read_u8()?;
        let last_logits = if has_logits == 1 {
            let count = reader.read_u32()? as usize;
            let cap = count.min(reader.remaining_bytes() / 4);
            let mut logits = Vec::with_capacity(cap);
            for _ in 0..count {
                let b = reader.read_bytes(4)?;
                let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                logits.push(val);
            }
            Some(logits)
        } else {
            None
        };

        let history_len = reader.read_u32()? as usize;
        let cap = history_len.min(reader.remaining_bytes() / 4);
        let mut token_history = Vec::with_capacity(cap);
        for _ in 0..history_len {
            token_history.push(reader.read_u32()?);
        }

        let kv_state = read_state_snapshot(&mut reader)?;
        if reader.pos != reader.buf.len() {
            return Err(CeraError::Format(
                "unexpected trailing bytes in session checkpoint".into(),
            ));
        }

        Ok(Self {
            model_fingerprint,
            position,
            max_seq_len,
            prefill_tokens,
            prefill_elapsed_ms,
            last_logits,
            token_history,
            kv_state,
        })
    }

    /// Estimated memory footprint of this checkpoint in bytes.
    pub fn byte_size(&self) -> usize {
        let mut size = 8 + 4 + 8 + 8 + 8 + 4 + 8 + 1 + 4;
        if let Some(logits) = &self.last_logits {
            size += 4 + logits.len() * 4;
        }
        size += self.token_history.len() * 4;
        size += self.kv_state.byte_size() + 29 + self.kv_state.layers.len() * 9;
        size
    }

    /// Save this checkpoint directly to a filesystem path using an atomic rename.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), CeraError> {
        let bytes = self.to_bytes();
        let path = path.as_ref();
        let tmp_path = temp_checkpoint_path(path);
        std::fs::write(&tmp_path, bytes).map_err(CeraError::Io)?;
        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(CeraError::Io(e));
        }
        Ok(())
    }

    /// Load a checkpoint from a filesystem path.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, CeraError> {
        let bytes = std::fs::read(path).map_err(CeraError::Io)?;
        Self::from_bytes(&bytes)
    }
}

/// A serializable, resumable snapshot of a conversational chat coordinator.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatCheckpoint {
    /// Underlying session checkpoint.
    pub session_checkpoint: SessionCheckpoint,
    /// Conversational phase at time of checkpoint.
    pub phase: SessionPhase,
    /// Active tool format.
    pub tool_format: ToolFormat,
    /// Currently registered tools.
    pub tools: Vec<ToolDef>,
    /// Whether the terminal token was already committed to the KV cache.
    pub terminal_committed: Option<bool>,
}

impl ChatCheckpoint {
    /// Serialize this chat checkpoint into a compact binary representation.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CeraError> {
        let tools_json = serde_json::to_string(&self.tools)
            .map_err(|e| CeraError::Format(format!("failed to serialize tools to json: {e}")))?;
        let tools_bytes = tools_json.as_bytes();
        let session_bytes = self.session_checkpoint.to_bytes();

        // Exact capacity:
        // magic(8) + version(4) + phase(1) + format(1) + term(1) + tools_len(4) + tools_bytes + session_len(4) + session_bytes
        let capacity = 8 + 4 + 1 + 1 + 1 + 4 + tools_bytes.len() + 4 + session_bytes.len();
        let mut writer = BufferWriter::with_capacity(capacity);
        writer.write_bytes(CHAT_CHECKPOINT_MAGIC);
        writer.write_u32(CHAT_CHECKPOINT_VERSION);

        let phase_byte = match self.phase {
            SessionPhase::Idle => 0u8,
            SessionPhase::PromptReady => 1u8,
            SessionPhase::TurnComplete => 2u8,
            SessionPhase::Interrupted => 3u8,
            SessionPhase::RawContext => 4u8,
            SessionPhase::Unusable => 5u8,
        };
        writer.write_u8(phase_byte);

        let format_byte = match self.tool_format {
            ToolFormat::Lfm2Pythonic => 0u8,
            ToolFormat::Hermes => 1u8,
        };
        writer.write_u8(format_byte);

        let term_byte = match self.terminal_committed {
            None => 0u8,
            Some(false) => 1u8,
            Some(true) => 2u8,
        };
        writer.write_u8(term_byte);

        writer.write_u32(tools_bytes.len() as u32);
        writer.write_bytes(tools_bytes);

        writer.write_u32(session_bytes.len() as u32);
        writer.write_bytes(&session_bytes);

        Ok(writer.buf)
    }

    /// Deserialize a chat checkpoint from a compact binary slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CeraError> {
        let mut reader = BufferReader::new(bytes);
        let magic = reader.read_bytes(8)?;
        if magic != CHAT_CHECKPOINT_MAGIC {
            return Err(CeraError::Format(
                "invalid chat checkpoint magic bytes".to_string(),
            ));
        }
        let version = reader.read_u32()?;
        if version != 1 && version != CHAT_CHECKPOINT_VERSION {
            return Err(CeraError::Format(format!(
                "unsupported chat checkpoint version {version}, expected {CHAT_CHECKPOINT_VERSION}"
            )));
        }

        let phase_byte = reader.read_u8()?;
        let phase = match phase_byte {
            0 => SessionPhase::Idle,
            1 => SessionPhase::PromptReady,
            2 => SessionPhase::TurnComplete,
            3 => SessionPhase::Interrupted,
            4 => SessionPhase::RawContext,
            5 => SessionPhase::Unusable,
            other => {
                return Err(CeraError::Format(format!(
                    "unknown session phase code {other} in chat checkpoint"
                )));
            }
        };

        let format_byte = reader.read_u8()?;
        let tool_format = match format_byte {
            0 => ToolFormat::Lfm2Pythonic,
            1 => ToolFormat::Hermes,
            other => {
                return Err(CeraError::Format(format!(
                    "unknown tool format code {other} in chat checkpoint"
                )));
            }
        };

        let terminal_committed = if version >= 2 {
            let term_byte = reader.read_u8()?;
            match term_byte {
                0 => None,
                1 => Some(false),
                2 => Some(true),
                other => {
                    return Err(CeraError::Format(format!(
                        "unknown terminal_committed code {other} in chat checkpoint"
                    )));
                }
            }
        } else if phase == SessionPhase::TurnComplete {
            Some(false)
        } else {
            None
        };

        let tools_len = reader.read_u32()? as usize;
        let tools_bytes = reader.read_bytes(tools_len)?;
        let tools: Vec<ToolDef> = if tools_bytes.is_empty() {
            Vec::new()
        } else {
            serde_json::from_slice(tools_bytes).map_err(|e| {
                CeraError::Format(format!(
                    "failed to parse tools json in chat checkpoint: {e}"
                ))
            })?
        };

        let session_len = reader.read_u32()? as usize;
        let session_bytes = reader.read_bytes(session_len)?;
        let session_checkpoint = SessionCheckpoint::from_bytes(session_bytes)?;
        if reader.pos != reader.buf.len() {
            return Err(CeraError::Format(
                "unexpected trailing bytes in chat checkpoint".into(),
            ));
        }

        Ok(Self {
            session_checkpoint,
            phase,
            tool_format,
            tools,
            terminal_committed,
        })
    }

    /// Save this chat checkpoint directly to a filesystem path using an atomic rename.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), CeraError> {
        let bytes = self.to_bytes()?;
        let path = path.as_ref();
        let tmp_path = temp_checkpoint_path(path);
        std::fs::write(&tmp_path, bytes).map_err(CeraError::Io)?;
        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(CeraError::Io(e));
        }
        Ok(())
    }

    /// Load a chat checkpoint from a filesystem path.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, CeraError> {
        let bytes = std::fs::read(path).map_err(CeraError::Io)?;
        Self::from_bytes(&bytes)
    }
}

// ---------------------------------------------------------------------------
// Binary encoding and decoding helpers
// ---------------------------------------------------------------------------

struct BufferReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> BufferReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining_bytes(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8, CeraError> {
        if self.remaining_bytes() >= 1 {
            let b = self.buf[self.pos];
            self.pos += 1;
            Ok(b)
        } else {
            Err(CeraError::Format(
                "unexpected EOF reading u8 in checkpoint".into(),
            ))
        }
    }

    fn read_u32(&mut self) -> Result<u32, CeraError> {
        if self.remaining_bytes() >= 4 {
            let bytes = [
                self.buf[self.pos],
                self.buf[self.pos + 1],
                self.buf[self.pos + 2],
                self.buf[self.pos + 3],
            ];
            self.pos += 4;
            Ok(u32::from_le_bytes(bytes))
        } else {
            Err(CeraError::Format(
                "unexpected EOF reading u32 in checkpoint".into(),
            ))
        }
    }

    fn read_u64(&mut self) -> Result<u64, CeraError> {
        if self.remaining_bytes() >= 8 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
            self.pos += 8;
            Ok(u64::from_le_bytes(bytes))
        } else {
            Err(CeraError::Format(
                "unexpected EOF reading u64 in checkpoint".into(),
            ))
        }
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], CeraError> {
        let end = self.pos.checked_add(len).ok_or_else(|| {
            CeraError::Format("buffer length overflow in checkpoint reader".into())
        })?;
        if end <= self.buf.len() {
            let slice = &self.buf[self.pos..end];
            self.pos = end;
            Ok(slice)
        } else {
            Err(CeraError::Format(
                "unexpected EOF reading byte buffer in checkpoint".into(),
            ))
        }
    }
}

struct BufferWriter {
    buf: Vec<u8>,
}

impl BufferWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    fn write_u8(&mut self, val: u8) {
        self.buf.push(val);
    }

    fn write_u32(&mut self, val: u32) {
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_u64(&mut self, val: u64) {
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

fn write_layer_snapshot(writer: &mut BufferWriter, layer: &crate::kv_cache::LayerSnapshot) {
    use crate::kv_cache::LayerSnapshot;
    match layer {
        LayerSnapshot::Attention { k_data, v_data } => {
            writer.write_u8(0);
            writer.write_u32(k_data.len() as u32);
            writer.write_bytes(k_data);
            writer.write_u32(v_data.len() as u32);
            writer.write_bytes(v_data);
        }
        LayerSnapshot::Conv { buffer } => {
            writer.write_u8(1);
            writer.write_u32(buffer.len() as u32);
            writer.write_bytes(buffer);
        }
        LayerSnapshot::AttentionCompressed { keys, values } => {
            writer.write_u8(2);
            writer.write_u32(keys.len() as u32);
            writer.write_bytes(keys);
            writer.write_u32(values.len() as u32);
            writer.write_bytes(values);
        }
        LayerSnapshot::AttentionF16 { k_data, v_data } => {
            writer.write_u8(3);
            writer.write_u32(k_data.len() as u32);
            writer.write_bytes(k_data);
            writer.write_u32(v_data.len() as u32);
            writer.write_bytes(v_data);
        }
        LayerSnapshot::Mamba2 {
            conv_state,
            ssm_state,
        } => {
            writer.write_u8(4);
            writer.write_u32(conv_state.len() as u32);
            writer.write_bytes(conv_state);
            writer.write_u32(ssm_state.len() as u32);
            writer.write_bytes(ssm_state);
        }
        LayerSnapshot::DeltaNet {
            conv_state,
            ssm_state,
        } => {
            writer.write_u8(5);
            writer.write_u32(conv_state.len() as u32);
            writer.write_bytes(conv_state);
            writer.write_u32(ssm_state.len() as u32);
            writer.write_bytes(ssm_state);
        }
        LayerSnapshot::ParallelAttentionMamba2 {
            snap,
            conv_state,
            ssm_state,
        } => {
            writer.write_u8(6);
            write_layer_snapshot(writer, snap);
            writer.write_u32(conv_state.len() as u32);
            writer.write_bytes(conv_state);
            writer.write_u32(ssm_state.len() as u32);
            writer.write_bytes(ssm_state);
        }
    }
}

fn read_layer_snapshot(
    reader: &mut BufferReader<'_>,
) -> Result<crate::kv_cache::LayerSnapshot, CeraError> {
    use crate::kv_cache::LayerSnapshot;
    let tag = reader.read_u8()?;
    match tag {
        0 => {
            let k_len = reader.read_u32()? as usize;
            let k_data = reader.read_bytes(k_len)?.to_vec();
            let v_len = reader.read_u32()? as usize;
            let v_data = reader.read_bytes(v_len)?.to_vec();
            Ok(LayerSnapshot::Attention { k_data, v_data })
        }
        1 => {
            let buf_len = reader.read_u32()? as usize;
            let buffer = reader.read_bytes(buf_len)?.to_vec();
            Ok(LayerSnapshot::Conv { buffer })
        }
        2 => {
            let k_len = reader.read_u32()? as usize;
            let keys = reader.read_bytes(k_len)?.to_vec();
            let v_len = reader.read_u32()? as usize;
            let values = reader.read_bytes(v_len)?.to_vec();
            Ok(LayerSnapshot::AttentionCompressed { keys, values })
        }
        3 => {
            let k_len = reader.read_u32()? as usize;
            let k_data = reader.read_bytes(k_len)?.to_vec();
            let v_len = reader.read_u32()? as usize;
            let v_data = reader.read_bytes(v_len)?.to_vec();
            Ok(LayerSnapshot::AttentionF16 { k_data, v_data })
        }
        4 => {
            let conv_len = reader.read_u32()? as usize;
            let conv_state = reader.read_bytes(conv_len)?.to_vec();
            let ssm_len = reader.read_u32()? as usize;
            let ssm_state = reader.read_bytes(ssm_len)?.to_vec();
            Ok(LayerSnapshot::Mamba2 {
                conv_state,
                ssm_state,
            })
        }
        5 => {
            let conv_len = reader.read_u32()? as usize;
            let conv_state = reader.read_bytes(conv_len)?.to_vec();
            let ssm_len = reader.read_u32()? as usize;
            let ssm_state = reader.read_bytes(ssm_len)?.to_vec();
            Ok(LayerSnapshot::DeltaNet {
                conv_state,
                ssm_state,
            })
        }
        6 => {
            let snap = read_layer_snapshot(reader)?;
            let conv_len = reader.read_u32()? as usize;
            let conv_state = reader.read_bytes(conv_len)?.to_vec();
            let ssm_len = reader.read_u32()? as usize;
            let ssm_state = reader.read_bytes(ssm_len)?.to_vec();
            Ok(LayerSnapshot::ParallelAttentionMamba2 {
                snap: Box::new(snap),
                conv_state,
                ssm_state,
            })
        }
        other => Err(CeraError::Format(format!(
            "unknown layer snapshot type tag {other} in checkpoint"
        ))),
    }
}

fn write_state_snapshot(writer: &mut BufferWriter, state: &StateSnapshot) {
    writer.write_u64(state.seq_len as u64);
    writer.write_u32(state.anchor_depth);
    writer.write_u8(state.boundary_kind);
    writer.write_u64(state.semantic_hash);
    writer.write_u32(state.shift_offset);
    writer.write_u32(state.layers.len() as u32);
    for layer in &state.layers {
        write_layer_snapshot(writer, layer);
    }
}

fn read_state_snapshot(reader: &mut BufferReader<'_>) -> Result<StateSnapshot, CeraError> {
    let seq_len = reader.read_u64()? as usize;
    let anchor_depth = reader.read_u32()?;
    let boundary_kind = reader.read_u8()?;
    let semantic_hash = reader.read_u64()?;
    let shift_offset = reader.read_u32()?;
    let layer_count = reader.read_u32()? as usize;
    let cap = layer_count.min(reader.remaining_bytes() / 8).min(1024);
    let mut layers = Vec::with_capacity(cap);
    for _ in 0..layer_count {
        layers.push(read_layer_snapshot(reader)?);
    }
    Ok(StateSnapshot {
        layers,
        seq_len,
        anchor_depth,
        boundary_kind,
        semantic_hash,
        shift_offset,
    })
}

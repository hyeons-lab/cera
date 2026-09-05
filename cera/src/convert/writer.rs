//! GGUF v3 file serializer and streaming writer.

use crate::session::CeraError;
use std::collections::BTreeMap;
use std::io::{self, Write};

/// GGUF Type Constants
pub const GGUF_TYPE_UINT8: u32 = 0;
pub const GGUF_TYPE_INT8: u32 = 1;
pub const GGUF_TYPE_UINT16: u32 = 2;
pub const GGUF_TYPE_INT16: u32 = 3;
pub const GGUF_TYPE_UINT32: u32 = 4;
pub const GGUF_TYPE_INT32: u32 = 5;
pub const GGUF_TYPE_FLOAT32: u32 = 6;
pub const GGUF_TYPE_BOOL: u32 = 7;
pub const GGUF_TYPE_STRING: u32 = 8;
pub const GGUF_TYPE_ARRAY: u32 = 9;
pub const GGUF_TYPE_UINT64: u32 = 10;
pub const GGUF_TYPE_INT64: u32 = 11;
pub const GGUF_TYPE_FLOAT64: u32 = 12;

/// GGML Tensor Types
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q5_1: u32 = 7;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q8_1: u32 = 9;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_K: u32 = 15;
pub const GGML_TYPE_BF16: u32 = 30;

/// Default GGUF data alignment boundary (32 bytes).
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Typed GGUF metadata value to serialize.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    StringArray(Vec<String>),
    Float32Array(Vec<f32>),
    Int32Array(Vec<i32>),
}

/// Tensor descriptor registered before writing data.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    /// Dimensions in GGUF order (reverse of PyTorch shape: [cols, rows, ...]).
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub size_bytes: usize,
}

/// Serializer for GGUF v3 files.
#[derive(Debug)]
pub struct GgufWriter {
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorMeta>,
    alignment: usize,
    current_data_offset: u64,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    /// Create a new GgufWriter with default 32-byte alignment.
    pub fn new() -> Self {
        Self {
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            current_data_offset: 0,
        }
    }

    /// Set alignment (default 32).
    pub fn set_alignment(&mut self, alignment: usize) {
        let align = if alignment == 0 {
            GGUF_DEFAULT_ALIGNMENT
        } else {
            alignment
        };
        self.alignment = align;
        if align != GGUF_DEFAULT_ALIGNMENT {
            self.add_u32("general.alignment", align as u32);
        } else {
            self.metadata.remove("general.alignment");
        }
    }

    /// Get configured data alignment.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get a reference to a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Add a string metadata key-value pair.
    pub fn add_string(&mut self, key: impl Into<String>, val: impl Into<String>) {
        self.metadata
            .insert(key.into(), MetadataValue::String(val.into()));
    }

    /// Add a u32 metadata key-value pair.
    pub fn add_u32(&mut self, key: impl Into<String>, val: u32) {
        self.metadata.insert(key.into(), MetadataValue::Uint32(val));
    }

    /// Add a i32 metadata key-value pair.
    pub fn add_i32(&mut self, key: impl Into<String>, val: i32) {
        self.metadata.insert(key.into(), MetadataValue::Int32(val));
    }

    /// Add a f32 metadata key-value pair.
    pub fn add_f32(&mut self, key: impl Into<String>, val: f32) {
        self.metadata
            .insert(key.into(), MetadataValue::Float32(val));
    }

    /// Add a bool metadata key-value pair.
    pub fn add_bool(&mut self, key: impl Into<String>, val: bool) {
        self.metadata.insert(key.into(), MetadataValue::Bool(val));
    }

    /// Add a u64 metadata key-value pair.
    pub fn add_u64(&mut self, key: impl Into<String>, val: u64) {
        self.metadata.insert(key.into(), MetadataValue::Uint64(val));
    }

    /// Add an array of strings metadata key-value pair.
    pub fn add_string_array(&mut self, key: impl Into<String>, val: Vec<String>) {
        self.metadata
            .insert(key.into(), MetadataValue::StringArray(val));
    }

    /// Add an array of f32 metadata key-value pair.
    pub fn add_f32_array(&mut self, key: impl Into<String>, val: Vec<f32>) {
        self.metadata
            .insert(key.into(), MetadataValue::Float32Array(val));
    }

    /// Add an array of i32 metadata key-value pair.
    pub fn add_i32_array(&mut self, key: impl Into<String>, val: Vec<i32>) {
        self.metadata
            .insert(key.into(), MetadataValue::Int32Array(val));
    }

    /// Register a tensor and compute its aligned data offset.
    ///
    /// `dims` should be in GGUF order (reverse of PyTorch: [cols, rows, ...]).
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dims: Vec<u64>,
        ggml_type: u32,
        size_bytes: usize,
    ) -> &TensorMeta {
        let align = self.alignment as u64;
        let offset = if self.current_data_offset.is_multiple_of(align) {
            self.current_data_offset
        } else {
            self.current_data_offset + (align - (self.current_data_offset % align))
        };

        self.current_data_offset = offset + size_bytes as u64;

        let meta = TensorMeta {
            name: name.into(),
            dims,
            ggml_type,
            offset,
            size_bytes,
        };
        self.tensors.push(meta);
        let idx = self.tensors.len() - 1;
        &self.tensors[idx]
    }

    /// Get registered tensor metadata slice.
    pub fn tensors(&self) -> &[TensorMeta] {
        &self.tensors
    }

    /// Write GGUF header, metadata KV pairs, and tensor info table.
    ///
    /// Returns the exact byte position where the first tensor's data begins.
    pub fn write_header_and_tensor_info<W: Write>(&self, w: &mut W) -> Result<u64, CeraError> {
        let mut written_bytes: u64 = 0;

        // 1. Magic
        w.write_all(b"GGUF")?;
        written_bytes += 4;

        // 2. Version (v3)
        w.write_all(&3u32.to_le_bytes())?;
        written_bytes += 4;

        // 3. Tensor count
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        written_bytes += 8;

        // 4. Metadata KV count
        w.write_all(&(self.metadata.len() as u64).to_le_bytes())?;
        written_bytes += 8;

        // 5. Metadata KV pairs
        for (key, val) in &self.metadata {
            write_string(w, key, &mut written_bytes)?;
            write_metadata_value(w, val, &mut written_bytes)?;
        }

        // 6. Tensor Info table
        for t in &self.tensors {
            write_string(w, &t.name, &mut written_bytes)?;
            w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
            written_bytes += 4;
            for &d in &t.dims {
                w.write_all(&d.to_le_bytes())?;
                written_bytes += 8;
            }
            w.write_all(&t.ggml_type.to_le_bytes())?;
            written_bytes += 4;
            w.write_all(&t.offset.to_le_bytes())?;
            written_bytes += 8;
        }

        // 7. Align to data boundary
        let align = self.alignment as u64;
        let pad = if written_bytes.is_multiple_of(align) {
            0
        } else {
            align - (written_bytes % align)
        };

        if pad > 0 {
            write_zero_padding(w, pad as usize)?;
            written_bytes += pad;
        }

        Ok(written_bytes)
    }

    /// Write an individual tensor's payload bytes at its current position,
    /// adding alignment padding after if needed.
    pub fn write_tensor_data<W: Write>(&self, w: &mut W, data: &[u8]) -> Result<usize, CeraError> {
        w.write_all(data)?;
        let align = self.alignment;
        let rem = data.len() % align;
        let pad = if rem == 0 { 0 } else { align - rem };
        if pad > 0 {
            write_zero_padding(w, pad)?;
        }
        Ok(data.len() + pad)
    }
}

fn write_zero_padding<W: Write>(w: &mut W, mut remaining: usize) -> io::Result<()> {
    let zeros = [0u8; 256];
    while remaining > 0 {
        let chunk = remaining.min(zeros.len());
        w.write_all(&zeros[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn write_string<W: Write>(w: &mut W, s: &str, written: &mut u64) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    *written += 8 + bytes.len() as u64;
    Ok(())
}

fn write_metadata_value<W: Write>(
    w: &mut W,
    val: &MetadataValue,
    written: &mut u64,
) -> Result<(), CeraError> {
    match val {
        MetadataValue::Uint8(v) => {
            w.write_all(&GGUF_TYPE_UINT8.to_le_bytes())?;
            w.write_all(&[*v])?;
            *written += 4 + 1;
        }
        MetadataValue::Int8(v) => {
            w.write_all(&GGUF_TYPE_INT8.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
            *written += 4 + 1;
        }
        MetadataValue::Uint16(v) => {
            w.write_all(&GGUF_TYPE_UINT16.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 2;
        }
        MetadataValue::Int16(v) => {
            w.write_all(&GGUF_TYPE_INT16.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 2;
        }
        MetadataValue::Uint32(v) => {
            w.write_all(&GGUF_TYPE_UINT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 4;
        }
        MetadataValue::Int32(v) => {
            w.write_all(&GGUF_TYPE_INT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 4;
        }
        MetadataValue::Float32(v) => {
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 4;
        }
        MetadataValue::Bool(v) => {
            w.write_all(&GGUF_TYPE_BOOL.to_le_bytes())?;
            w.write_all(&[if *v { 1 } else { 0 }])?;
            *written += 4 + 1;
        }
        MetadataValue::String(s) => {
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            *written += 4;
            write_string(w, s, written)?;
        }
        MetadataValue::Uint64(v) => {
            w.write_all(&GGUF_TYPE_UINT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 8;
        }
        MetadataValue::Int64(v) => {
            w.write_all(&GGUF_TYPE_INT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 8;
        }
        MetadataValue::Float64(v) => {
            w.write_all(&GGUF_TYPE_FLOAT64.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            *written += 4 + 8;
        }
        MetadataValue::StringArray(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_STRING.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            *written += 4 + 4 + 8;
            for s in arr {
                write_string(w, s, written)?;
            }
        }
        MetadataValue::Float32Array(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            *written += 4 + 4 + 8;
            for &f in arr {
                w.write_all(&f.to_le_bytes())?;
                *written += 4;
            }
        }
        MetadataValue::Int32Array(arr) => {
            w.write_all(&GGUF_TYPE_ARRAY.to_le_bytes())?;
            w.write_all(&GGUF_TYPE_INT32.to_le_bytes())?;
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            *written += 4 + 4 + 8;
            for &i in arr {
                w.write_all(&i.to_le_bytes())?;
                *written += 4;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;

    #[test]
    fn test_gguf_writer_roundtrip() {
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "llama");
        writer.add_u32("llama.context_length", 4096);
        writer.add_f32("llama.rope.freq_base", 10000.0);
        writer.add_bool("general.is_test", true);
        writer.add_string_array(
            "tokenizer.ggml.tokens",
            vec!["<unk>".into(), "hello".into(), "world".into()],
        );

        // Register dummy tensor: 2x32 F32 = 64 floats = 256 bytes
        let tensor_bytes = vec![0x42u8; 256];
        writer.add_tensor(
            "blk.0.attn_q.weight",
            vec![32, 2],
            GGML_TYPE_F32,
            tensor_bytes.len(),
        );

        let mut buf = Vec::new();
        let data_start = writer.write_header_and_tensor_info(&mut buf).unwrap();
        assert_eq!(buf.len() as u64, data_start);
        assert_eq!(data_start % 32, 0);

        writer.write_tensor_data(&mut buf, &tensor_bytes).unwrap();

        // Verify with GgufFile parser
        let gguf = GgufFile::from_bytes(buf.into()).unwrap();
        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(gguf.get_u32("llama.context_length"), Some(4096));
        assert_eq!(gguf.get_bool("general.is_test"), Some(true));

        let tensor = gguf.get_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(tensor.shape(), &[32, 2]);
        let data = gguf.tensor_data("blk.0.attn_q.weight").unwrap();
        assert_eq!(data, &tensor_bytes[..]);
    }
}

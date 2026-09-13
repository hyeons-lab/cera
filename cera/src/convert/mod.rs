//! SafeTensors -> GGUF model conversion and on-the-fly streaming quantization.

pub mod config;
pub mod parity;
#[cfg(feature = "std-fs")]
pub mod pipeline;
pub mod quantize;
pub mod safetensors;
pub mod tokenizer;
pub mod writer;

pub use parity::{
    GgufParityReport, InferenceParityResult, MetadataDiff, MetadataParityStatus,
    TensorComparisonSummary, TensorParityEntry, compare_gguf_metadata, compare_gguf_tensors,
};
#[cfg(feature = "mmap")]
pub use parity::{audit_gguf_parity, compare_gguf_files, compare_inference_parity};
#[cfg(all(feature = "std-fs", feature = "remote"))]
pub use pipeline::{QuantizeOptions, stream_quantize_hf_repo};
#[cfg(feature = "std-fs")]
pub use pipeline::{
    quantize_safetensors_to_gguf, quantize_safetensors_to_gguf_with_overrides,
    quantize_safetensors_to_gguf_with_strategy,
};
pub use quantize::{
    QuantStrategy, TargetQuant, compute_cosine_similarity, compute_rmse, compute_snr_db,
    matches_tensor_pattern, parse_tensor_override, quantize_tensor_data,
    quantize_tensor_data_with_strategy,
};
pub use writer::GgufWriter;

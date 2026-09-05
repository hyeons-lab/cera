//! SafeTensors -> GGUF model conversion and on-the-fly streaming quantization.

pub mod config;
#[cfg(feature = "remote")]
pub mod pipeline;
pub mod quantize;
pub mod safetensors;
pub mod tokenizer;
pub mod writer;

#[cfg(feature = "remote")]
pub use pipeline::{
    QuantizeOptions, quantize_safetensors_to_gguf, quantize_safetensors_to_gguf_with_overrides,
    stream_quantize_hf_repo,
};
pub use quantize::{
    QuantStrategy, TargetQuant, compute_cosine_similarity, compute_rmse, compute_snr_db,
    matches_tensor_pattern, parse_tensor_override, quantize_tensor_data,
    quantize_tensor_data_with_strategy,
};
pub use writer::GgufWriter;

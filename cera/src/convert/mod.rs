//! SafeTensors -> GGUF model conversion and on-the-fly streaming quantization.

pub mod config;
#[cfg(feature = "remote")]
pub mod pipeline;
pub mod quantize;
pub mod safetensors;
pub mod tokenizer;
pub mod writer;

#[cfg(feature = "remote")]
pub use pipeline::{QuantizeOptions, quantize_safetensors_to_gguf, stream_quantize_hf_repo};
pub use quantize::{
    QuantStrategy, TargetQuant, compute_cosine_similarity, compute_rmse, compute_snr_db,
    quantize_tensor_data, quantize_tensor_data_with_strategy,
};
pub use writer::GgufWriter;

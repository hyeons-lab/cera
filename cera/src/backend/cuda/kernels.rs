// CUDA kernel parameter definitions and source registry.
//
// Strongly-typed parameters implementing DeviceRepr for safe kernel argument passing.
// Sources are embedded at compile-time via include_str! for self-contained binaries.

use cudarc::driver::DeviceRepr;

// Embedded CUDA source strings
pub const GEMV_Q4_0_SRC: &str = include_str!("../shaders/cuda/gemv_q4_0.cu");
pub const GEMV_Q8_0_SRC: &str = include_str!("../shaders/cuda/gemv_q8_0.cu");
pub const GEMM_Q4_0_SRC: &str = include_str!("../shaders/cuda/gemm_q4_0.cu");
pub const GEMM_Q8_0_SRC: &str = include_str!("../shaders/cuda/gemm_q8_0.cu");
pub const GATHER_EMBEDDING_SRC: &str = include_str!("../shaders/cuda/gather_embedding.cu");
pub const RMSNORM_SRC: &str = include_str!("../shaders/cuda/rmsnorm.cu");
pub const QK_NORM_ROPE_SRC: &str = include_str!("../shaders/cuda/qk_norm_rope.cu");
pub const ELEMENTWISE_SRC: &str = include_str!("../shaders/cuda/elementwise.cu");
pub const SOFTMAX_SRC: &str = include_str!("../shaders/cuda/softmax.cu");
pub const ATTENTION_SRC: &str = include_str!("../shaders/cuda/attention.cu");
pub const CONV1D_FUSED_SRC: &str = include_str!("../shaders/cuda/conv1d_fused.cu");
pub const ARGMAX_F32_SRC: &str = include_str!("../shaders/cuda/argmax.cu");

/// Parameters for matrix-vector multiplication kernels (GEMV).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemvParams {
    pub m: u32,
    pub k: u32,
}
unsafe impl DeviceRepr for GemvParams {}

/// Parameters for matrix-matrix multiplication kernels (GEMM).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _pad: u32,
}
unsafe impl DeviceRepr for GemmParams {}

/// Parameters for embedding gather kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GatherParams {
    pub token_id: u32,
    pub hidden_size: u32,
}
unsafe impl DeviceRepr for GatherParams {}

/// Parameters for Root Mean Square Normalization (RMSNorm).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    pub n: u32,
    pub eps: f32,
}
unsafe impl DeviceRepr for RmsNormParams {}

/// Parameters for fused Query-Key Normalization + Rotary Position Embedding (RoPE).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QkNormRopeParams {
    pub pos: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
    pub freq_base: f32,
    pub rope_type: u32,
    pub has_freq_factors: u32,
    pub has_qk_norm: u32,
}
unsafe impl DeviceRepr for QkNormRopeParams {}

/// Parameters for basic elementwise operations.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ElementwiseParams {
    pub n: u32,
    pub _pad: u32,
}
unsafe impl DeviceRepr for ElementwiseParams {}

/// Parameters for scaled elementwise additions or scalar scaling.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScaleParams {
    pub n: u32,
    pub scale: f32,
}
unsafe impl DeviceRepr for ScaleParams {}

/// Parameters for softmax operations.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftmaxParams {
    pub n: u32,
    pub _pad: u32,
}
unsafe impl DeviceRepr for SoftmaxParams {}

/// Parameters for single-token FlashAttention decode kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttentionParams {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub seq_len: u32,
    pub scale: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}
unsafe impl DeviceRepr for AttentionParams {}

/// Parameters for fused 1D gated convolution kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Conv1dParams {
    pub hs: u32,
    pub kernel_size: u32,
    pub d_conv: u32,
    pub _pad: u32,
}
unsafe impl DeviceRepr for Conv1dParams {}

/// Parameters for argmax reduction kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ArgmaxParams {
    pub n: u32,
    pub _pad: u32,
}
unsafe impl DeviceRepr for ArgmaxParams {}

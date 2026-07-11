pub mod cpu;
pub mod cpu_features;
pub mod simd;

// Not on wasm32: std threads can't spawn there, so the row hot path routes
// through rayon/wasm-bindgen-rayon instead (see `cpu::par_rows`).
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub mod threadpool;

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
mod calibrate;

#[cfg(feature = "blas")]
pub mod blas;

#[cfg(feature = "gpu")]
pub mod wgpu;

#[cfg(feature = "gpu")]
pub mod wgsl_pp;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub mod metal;

/// Compute operations supported by backends.
#[derive(Debug)]
pub enum Op {
    Linear {
        weight: usize,
        bias: Option<usize>,
    },
    RmsNorm {
        weight: usize,
        eps: f32,
    },
    Rope {
        pos: usize,
        freq_base: f32,
        head_dim: usize,
    },
    Silu,
    GatedMlp {
        gate: usize,
        up: usize,
        down: usize,
    },
    Attention {
        n_heads: usize,
        n_kv_heads: usize,
    },
    Conv1d {
        weight: usize,
        bias: Option<usize>,
        groups: usize,
    },
    Mul,
    Add,
    Softmax,
}

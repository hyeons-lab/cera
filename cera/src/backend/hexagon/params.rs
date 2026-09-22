//! Host-side parameter precomputations for Hexagon HTP operations.
//!
//! Hexagon DSP kernels avoid runtime hardware division and transcendental calls
//! by consuming precomputed integer division constants (Granlund and Montgomery FastDiv)
//! and layout descriptors generated on the host.

/// Precomputed integer division constants using Granlund and Montgomery's algorithm.
///
/// Permits the DSP to calculate `n / d` without hardware division via:
/// `((mulhi(n, mp) + n) >> l)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FastDivValues {
    pub mp: u32,
    pub l: u32,
}

/// Compute fast integer division constants for divisor `d`.
pub fn init_fastdiv(d: u32) -> FastDivValues {
    if d == 0 {
        return FastDivValues { mp: 0, l: 0 };
    }
    let mut l = 0;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - (d as u64))) / (d as u64) + 1) as u32;
    FastDivValues { mp, l }
}

/// Host-computed parameters for RMS norm.
pub fn build_rms_norm_params(eps: f32) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = eps.to_bits() as i32;
    params
}

/// Host-computed parameters for RoPE positional embeddings.
pub fn build_rope_params(
    pos: usize,
    n_dims: usize,
    mode: u32,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
) -> [i32; 16] {
    let mut params = [0i32; 16];
    params[0] = pos as i32;
    params[1] = n_dims as i32;
    params[2] = mode as i32;
    params[3] = n_ctx_orig as i32;
    params[4] = freq_base.to_bits() as i32;
    params[5] = freq_scale.to_bits() as i32;
    params
}

/// Build kernel parameters for matrix multiplication dispatch.
pub fn build_mul_mat_kernel_params(ne11: u32, ne12: u32, n_threads: u32) -> [i32; 32] {
    let mut kparams = [0i32; 32];
    let div_ne11 = init_fastdiv(ne11.max(1));
    let div_ne12 = init_fastdiv(ne12.max(1));
    let div_ne12_ne11 = init_fastdiv((ne11 * ne12).max(1));

    // Pack into kernel_params array matching upstream layout
    kparams[0] = n_threads as i32;
    kparams[1] = div_ne11.mp as i32;
    kparams[2] = div_ne11.l as i32;
    kparams[3] = div_ne12.mp as i32;
    kparams[4] = div_ne12.l as i32;
    kparams[5] = div_ne12_ne11.mp as i32;
    kparams[6] = div_ne12_ne11.l as i32;

    kparams
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastdiv_basic() {
        let divisors = [1, 2, 3, 5, 7, 10, 16, 32, 64, 128, 256, 1024, 4096];
        for &d in &divisors {
            let fd = init_fastdiv(d);
            for n in [0, 1, 2, d - 1, d, d + 1, d * 5 + 3, 65535, 1_000_000] {
                let hi = (((n as u64) * (fd.mp as u64)) >> 32) as u32;
                let q = (hi + n) >> fd.l;
                assert_eq!(q, n / d, "FastDiv mismatch for n={}, d={}", n, d);
            }
        }
    }
}

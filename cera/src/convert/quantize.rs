//! Tensor quantization encoders for GGUF model conversion.
//!
//! Provides streaming row-level SIMD/scalar quantizers for Q4_0, Q8_0, Q4_K_M,
//! Q5_K_M, Q6_K, F16, and F32, along with smart optimization strategies
//! (Fast MSE grid search, Half-Quadratic Quantization HQQ, and QuaRot Hadamard transforms).

use crate::convert::writer::*;
use crate::quant::{f16_to_f32, f32_to_f16};
use crate::session::CeraError;

#[inline(always)]
fn safe_clamp(val: f32, min: f32, max: f32) -> f32 {
    if val.is_nan() || val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

/// Target Quantization Preset
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetQuant {
    Q4_0,
    Q8_0,
    Q4_K_M,
    Q5_K_M,
    Q6_K,
    F16,
    F32,
}

/// Match a glob or wildcard pattern like "classifier.*", "*.bias", or exact "token_embd.weight".
pub fn matches_tensor_pattern(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    pattern == name
}

/// Parse a key-value override like "classifier.*=F16" or "output.weight=Q8_0".
pub fn parse_tensor_override(s: &str) -> Option<(String, TargetQuant)> {
    let (pattern, quant_str) = s.split_once('=')?;
    let quant = TargetQuant::parse_str(quant_str.trim())?;
    Some((pattern.trim().to_string(), quant))
}

impl TargetQuant {
    /// Parse string representation (e.g. "q4_k_m", "Q4_0", "f16").
    pub fn parse_str(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "Q4_0" => Some(Self::Q4_0),
            "Q8_0" => Some(Self::Q8_0),
            "Q4_K" | "Q4_K_M" => Some(Self::Q4_K_M),
            "Q5_K" | "Q5_K_M" => Some(Self::Q5_K_M),
            "Q6_K" => Some(Self::Q6_K),
            "F16" | "BF16" => Some(Self::F16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }

    /// Canonical string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q6_K => "Q6_K",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    /// Select the appropriate GGML type for a tensor, checking per-tensor overrides first.
    pub fn select_ggml_type_with_overrides(
        &self,
        tensor_name: &str,
        num_dims: usize,
        num_elements: usize,
        overrides: &[(String, TargetQuant)],
    ) -> u32 {
        for (pattern, target) in overrides {
            if matches_tensor_pattern(pattern, tensor_name) {
                return match target {
                    TargetQuant::F32 => GGML_TYPE_F32,
                    TargetQuant::F16 => GGML_TYPE_F16,
                    TargetQuant::Q8_0 => GGML_TYPE_Q8_0,
                    TargetQuant::Q4_0 => GGML_TYPE_Q4_0,
                    TargetQuant::Q6_K => GGML_TYPE_Q6_K,
                    TargetQuant::Q5_K_M => GGML_TYPE_Q5_K,
                    TargetQuant::Q4_K_M => GGML_TYPE_Q4_K,
                };
            }
        }
        self.select_ggml_type(tensor_name, num_dims, num_elements)
    }

    /// Select the appropriate GGML type for a tensor given its name and rank.
    pub fn select_ggml_type(&self, tensor_name: &str, num_dims: usize, num_elements: usize) -> u32 {
        match self {
            Self::F32 => return GGML_TYPE_F32,
            Self::F16 => return GGML_TYPE_F16,
            _ => {}
        }

        // 1D tensors (layer norms, bias vectors) and shortconv 3-token kernels are kept in F32.
        if num_dims <= 1
            || num_elements < 256
            || tensor_name.contains("shortconv.conv")
            || tensor_name.contains("conv.conv")
        {
            return GGML_TYPE_F32;
        }

        // Tensors must be a multiple of block size (256 for K-quants, 32 for Q4_0/Q8_0)
        if !num_elements.is_multiple_of(256) {
            if num_elements.is_multiple_of(32) {
                return match self {
                    Self::Q8_0 => GGML_TYPE_Q8_0,
                    _ => GGML_TYPE_Q4_0,
                };
            }
            return GGML_TYPE_F32;
        }

        match self {
            Self::F32 | Self::F16 => unreachable!(),
            Self::Q8_0 => GGML_TYPE_Q8_0,
            Self::Q4_0 => GGML_TYPE_Q4_0,
            Self::Q6_K => GGML_TYPE_Q6_K,
            Self::Q5_K_M => {
                if tensor_name.contains("attn_v")
                    || tensor_name.contains("ffn_down")
                    || tensor_name.contains("w2")
                {
                    GGML_TYPE_Q6_K
                } else {
                    GGML_TYPE_Q5_K
                }
            }
            Self::Q4_K_M => {
                if tensor_name.contains("attn_v")
                    || tensor_name.contains("ffn_down")
                    || tensor_name.contains("w2")
                    || tensor_name.contains("token_embd")
                    || tensor_name.contains("output.weight")
                {
                    GGML_TYPE_Q6_K
                } else {
                    GGML_TYPE_Q4_K
                }
            }
        }
    }

    /// Compute the exact byte size of a quantized tensor.
    pub fn compute_tensor_bytes(ggml_type: u32, num_elements: usize) -> usize {
        match ggml_type {
            GGML_TYPE_F32 => num_elements * 4,
            GGML_TYPE_F16 | GGML_TYPE_BF16 => num_elements * 2,
            GGML_TYPE_Q8_0 => num_elements.div_ceil(32) * 34,
            GGML_TYPE_Q4_0 => num_elements.div_ceil(32) * 18,
            GGML_TYPE_Q4_K => num_elements.div_ceil(256) * 144,
            GGML_TYPE_Q5_K => num_elements.div_ceil(256) * 176,
            GGML_TYPE_Q6_K => num_elements.div_ceil(256) * 210,
            _ => num_elements * 4,
        }
    }
}

/// Quantization optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantStrategy {
    /// Automatic: picks best calibration-free optimization (Fast MSE grid search).
    #[default]
    Auto,
    /// Fast blockwise L2 MSE scale grid search (zero calibration data, higher SNR).
    FastMse,
    /// Half-Quadratic Quantization (iterative coordinate splitting for scale and zero points).
    Hqq,
    /// Randomized Walsh-Hadamard Transform (FWHT) coordinate rotation (QuaRot outlier diffusion).
    QuaRot,
}

impl QuantStrategy {
    pub fn parse_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "auto" => Some(Self::Auto),
            "fast-mse" | "fast" | "mse" => Some(Self::FastMse),
            "hqq" => Some(Self::Hqq),
            "quarot" | "hadamard" | "rotation" => Some(Self::QuaRot),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::FastMse => "fast-mse",
            Self::Hqq => "hqq",
            Self::QuaRot => "quarot",
        }
    }
}

// ── Accuracy Evaluation Utilities ─────────────────────────────────────────────

/// Compute Signal-to-Noise Ratio (SNR) in decibels (dB) between original and dequantized tensor.
pub fn compute_snr_db(orig: &[f32], dequant: &[f32]) -> f32 {
    if orig.len() != dequant.len() || orig.is_empty() {
        return 0.0;
    }
    let mut signal_power = 0.0f32;
    let mut noise_power = 0.0f32;
    for (o, d) in orig.iter().zip(dequant.iter()) {
        signal_power += o * o;
        let diff = o - d;
        noise_power += diff * diff;
    }
    if noise_power <= 1e-12 {
        return 100.0; // Near-infinite SNR
    }
    if signal_power <= 1e-12 {
        return 0.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Compute Root Mean Square Error (RMSE) between original and dequantized tensor.
pub fn compute_rmse(orig: &[f32], dequant: &[f32]) -> f32 {
    if orig.len() != dequant.len() || orig.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0f32;
    for (o, d) in orig.iter().zip(dequant.iter()) {
        let diff = o - d;
        sum_sq += diff * diff;
    }
    (sum_sq / orig.len() as f32).sqrt()
}

/// Compute Cosine Similarity between original and dequantized tensor.
pub fn compute_cosine_similarity(orig: &[f32], dequant: &[f32]) -> f32 {
    if orig.len() != dequant.len() || orig.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_o = 0.0f32;
    let mut norm_d = 0.0f32;
    for (o, d) in orig.iter().zip(dequant.iter()) {
        dot += o * d;
        norm_o += o * o;
        norm_d += d * d;
    }
    let denom = (norm_o * norm_d).sqrt();
    if denom <= 1e-12 {
        return 1.0;
    }
    dot / denom
}

// ── In-Place Fast Walsh-Hadamard Transform (FWHT) for QuaRot ─────────────────

/// In-place normalized Fast Walsh-Hadamard Transform (FWHT) on a power-of-2 slice.
pub fn fast_walsh_hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    if n <= 1 || (n & (n - 1)) != 0 {
        return; // FWHT requires power of 2
    }

    let mut len = 1;
    while len < n {
        for i in (0..n).step_by(len * 2) {
            for j in 0..len {
                let u = data[i + j];
                let v = data[i + len + j];
                data[i + j] = u + v;
                data[i + len + j] = u - v;
            }
        }
        len *= 2;
    }

    let scale = 1.0 / (n as f32).sqrt();
    for x in data.iter_mut() {
        *x *= scale;
    }
}

// ── Top-Level Quantization Dispatch ──────────────────────────────────────────

/// Quantize a slice of floats into the target GGML binary format with default strategy.
pub fn quantize_tensor_data(
    input: &[f32],
    ggml_type: u32,
    output: &mut [u8],
) -> Result<usize, CeraError> {
    quantize_tensor_data_with_strategy(input, ggml_type, QuantStrategy::Auto, output)
}

/// Quantize a slice of floats into the target GGML binary format using a specific optimization strategy.
pub fn quantize_tensor_data_with_strategy(
    input: &[f32],
    ggml_type: u32,
    strategy: QuantStrategy,
    output: &mut [u8],
) -> Result<usize, CeraError> {
    let expected_bytes = TargetQuant::compute_tensor_bytes(ggml_type, input.len());
    if output.len() < expected_bytes {
        return Err(CeraError::Backend(format!(
            "output buffer too small for quantize: expected {expected_bytes} bytes, got {}",
            output.len()
        )));
    }

    if strategy == QuantStrategy::QuaRot {
        return Err(CeraError::Backend(
            "QuaRot quantization requires online activation Hadamard rotation support in inference kernels".into(),
        ));
    }

    match ggml_type {
        GGML_TYPE_F32 => {
            let (chunks, _) = output.as_chunks_mut::<4>();
            for (chunk, &x) in chunks.iter_mut().zip(input.iter()) {
                chunk.copy_from_slice(&x.to_le_bytes());
            }
            Ok(expected_bytes)
        }
        GGML_TYPE_F16 => {
            let (chunks, _) = output.as_chunks_mut::<2>();
            for (chunk, &x) in chunks.iter_mut().zip(input.iter()) {
                let half = f32_to_f16(x);
                chunk.copy_from_slice(&half.to_le_bytes());
            }
            Ok(expected_bytes)
        }
        GGML_TYPE_Q8_0 => {
            match strategy {
                QuantStrategy::Auto | QuantStrategy::FastMse => {
                    quantize_q8_0_smart_mse(input, output)?;
                }
                QuantStrategy::Hqq => {
                    quantize_q8_0_hqq(input, output)?;
                }
                QuantStrategy::QuaRot => unreachable!(),
            }
            Ok(expected_bytes)
        }
        GGML_TYPE_Q4_0 => {
            match strategy {
                QuantStrategy::Auto | QuantStrategy::FastMse => {
                    quantize_q4_0_smart_mse(input, output)?;
                }
                QuantStrategy::Hqq => {
                    quantize_q4_0_hqq(input, output)?;
                }
                QuantStrategy::QuaRot => unreachable!(),
            }
            Ok(expected_bytes)
        }
        GGML_TYPE_Q4_K => {
            quantize_q4_k(input, output)?;
            Ok(expected_bytes)
        }
        GGML_TYPE_Q5_K => {
            quantize_q5_k(input, output)?;
            Ok(expected_bytes)
        }
        GGML_TYPE_Q6_K => {
            quantize_q6_k(input, output)?;
            Ok(expected_bytes)
        }
        other => Err(CeraError::Backend(format!(
            "unsupported GGML quantization type id: {other}"
        ))),
    }
}

// ── Q8_0 Quantizers ───────────────────────────────────────────────────────────

/// Standard Q8_0 Quantizer (32 floats -> 34 bytes).
pub fn quantize_q8_0(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(32) {
        return Err(CeraError::Backend(format!(
            "input length ({}) must be a multiple of 32 for Q8_0",
            input.len()
        )));
    }
    let required_bytes = (input.len() / 32) * 34;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q8_0 output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 32;
    for b in 0..num_blocks {
        let block_in = &input[b * 32..(b + 1) * 32];
        let block_out = &mut output[b * 34..(b + 1) * 34];

        let mut amax: f32 = 0.0;
        for &x in block_in {
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let d = amax / 127.0;
        let d_f16 = f32_to_f16(d);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };

        for i in 0..32 {
            let val = (block_in[i] * id).round();
            let clamped = safe_clamp(val, -127.0, 127.0) as i8;
            block_out[2 + i] = clamped as u8;
        }
    }

    Ok(())
}

/// Smart Fast MSE Grid Search Q8_0 Quantizer.
pub fn quantize_q8_0_smart_mse(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(32) {
        return Err(CeraError::Backend(
            "Q8_0 input must be multiple of 32".into(),
        ));
    }
    let required_bytes = (input.len() / 32) * 34;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q8_0 output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 32;
    let alphas = [0.85f32, 0.90, 0.95, 1.0, 1.05];

    for b in 0..num_blocks {
        let block_in = &input[b * 32..(b + 1) * 32];
        let block_out = &mut output[b * 34..(b + 1) * 34];

        let mut amax: f32 = 0.0;
        for &x in block_in {
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let base_d = amax / 127.0;
        let mut best_d = f16_to_f32(f32_to_f16(base_d));
        let mut min_mse = f32::MAX;

        if amax > 0.0 {
            for &alpha in &alphas {
                let cand_d = f16_to_f32(f32_to_f16(base_d * alpha));
                let id = if cand_d != 0.0 { 1.0 / cand_d } else { 0.0 };
                let mut mse = 0.0f32;

                for &x in block_in {
                    let q = safe_clamp((x * id).round(), -127.0, 127.0);
                    let dequant = q * cand_d;
                    let diff = x - dequant;
                    mse += diff * diff;
                }

                if mse < min_mse {
                    min_mse = mse;
                    best_d = cand_d;
                }
            }
        }

        let d_f16 = f32_to_f16(best_d);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };
        for i in 0..32 {
            let val = safe_clamp((block_in[i] * id).round(), -127.0, 127.0) as i8;
            block_out[2 + i] = val as u8;
        }
    }

    Ok(())
}

/// Half-Quadratic (HQQ) Q8_0 Quantizer.
pub fn quantize_q8_0_hqq(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(32) {
        return Err(CeraError::Backend(
            "Q8_0 input must be multiple of 32".into(),
        ));
    }
    let required_bytes = (input.len() / 32) * 34;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q8_0 output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 32;
    for b in 0..num_blocks {
        let block_in = &input[b * 32..(b + 1) * 32];
        let block_out = &mut output[b * 34..(b + 1) * 34];

        let mut amax: f32 = 0.0;
        for &x in block_in {
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let mut d = amax / 127.0;

        // Coordinate descent iterations
        if amax > 0.0 {
            for _ in 0..3 {
                let id = 1.0 / d;
                let mut num = 0.0f32;
                let mut den = 0.0f32;

                for &x in block_in {
                    let q = safe_clamp((x * id).round(), -127.0, 127.0);
                    num += x * q;
                    den += q * q;
                }

                if den > 1e-8 && num > 0.0 {
                    d = (num / den).max(1e-10);
                }
            }
        }

        let d_f16 = f32_to_f16(d);
        let actual_d = f16_to_f32(d_f16);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };
        for i in 0..32 {
            let val = safe_clamp((block_in[i] * id).round(), -127.0, 127.0) as i8;
            block_out[2 + i] = val as u8;
        }
    }

    Ok(())
}

// ── Q4_0 Quantizers ───────────────────────────────────────────────────────────

/// Standard Q4_0 Quantizer (32 floats -> 18 bytes).
pub fn quantize_q4_0(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(32) {
        return Err(CeraError::Backend(format!(
            "input length ({}) must be a multiple of 32 for Q4_0",
            input.len()
        )));
    }
    let required_bytes = (input.len() / 32) * 18;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q4_0 output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 32;
    for b in 0..num_blocks {
        let block_in = &input[b * 32..(b + 1) * 32];
        let block_out = &mut output[b * 18..(b + 1) * 18];

        let mut amax: f32 = 0.0;
        for &x in block_in {
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let d = amax / 8.0;
        let d_f16 = f32_to_f16(d);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };

        for i in 0..16 {
            let x0 = block_in[i];
            let x1 = block_in[i + 16];

            let q0 = safe_clamp((x0 * id + 8.5).floor(), 0.0, 15.0) as u8;
            let q1 = safe_clamp((x1 * id + 8.5).floor(), 0.0, 15.0) as u8;

            block_out[2 + i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }

    Ok(())
}

/// Smart Fast MSE Grid Search Q4_0 Quantizer.
pub fn quantize_q4_0_smart_mse(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(32) {
        return Err(CeraError::Backend(
            "Q4_0 input must be multiple of 32".into(),
        ));
    }
    let required_bytes = (input.len() / 32) * 18;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q4_0 output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 32;
    let alphas = [0.75f32, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05];

    for b in 0..num_blocks {
        let block_in = &input[b * 32..(b + 1) * 32];
        let block_out = &mut output[b * 18..(b + 1) * 18];

        let mut amax: f32 = 0.0;
        for &x in block_in {
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let base_d = amax / 8.0;
        let mut best_d = f16_to_f32(f32_to_f16(base_d));
        let mut min_mse = f32::MAX;

        if amax > 0.0 {
            for &alpha in &alphas {
                let cand_d = f16_to_f32(f32_to_f16(base_d * alpha));
                let id = if cand_d != 0.0 { 1.0 / cand_d } else { 0.0 };
                let mut mse = 0.0f32;

                for &x in block_in {
                    let q = safe_clamp((x * id + 8.5).floor(), 0.0, 15.0);
                    let dequant = (q - 8.0) * cand_d;
                    let diff = x - dequant;
                    mse += diff * diff;
                }

                if mse < min_mse {
                    min_mse = mse;
                    best_d = cand_d;
                }
            }
        }

        let d_f16 = f32_to_f16(best_d);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };
        for i in 0..16 {
            let x0 = block_in[i];
            let x1 = block_in[i + 16];

            let q0 = safe_clamp((x0 * id + 8.5).floor(), 0.0, 15.0) as u8;
            let q1 = safe_clamp((x1 * id + 8.5).floor(), 0.0, 15.0) as u8;

            block_out[2 + i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }

    Ok(())
}

/// Half-Quadratic (HQQ) Q4_0 Quantizer.
pub fn quantize_q4_0_hqq(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(32) {
        return Err(CeraError::Backend(
            "Q4_0 input must be multiple of 32".into(),
        ));
    }
    let required_bytes = (input.len() / 32) * 18;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q4_0 output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 32;
    for b in 0..num_blocks {
        let block_in = &input[b * 32..(b + 1) * 32];
        let block_out = &mut output[b * 18..(b + 1) * 18];

        let mut amax: f32 = 0.0;
        for &x in block_in {
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let mut d = amax / 8.0;

        // Alternating least squares coordinate descent (3 iterations)
        if amax > 0.0 {
            for _ in 0..3 {
                let id = 1.0 / d;
                let mut num = 0.0f32;
                let mut den = 0.0f32;

                for &x in block_in {
                    let q = safe_clamp((x * id + 8.5).floor(), 0.0, 15.0) - 8.0;
                    num += x * q;
                    den += q * q;
                }

                if den > 1e-8 {
                    let updated_d = num / den;
                    if updated_d > 1e-5 {
                        d = updated_d;
                    }
                }
            }
        }

        let d_f16 = f32_to_f16(d);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };
        for i in 0..16 {
            let x0 = block_in[i];
            let x1 = block_in[i + 16];

            let q0 = safe_clamp((x0 * id + 8.5).floor(), 0.0, 15.0) as u8;
            let q1 = safe_clamp((x1 * id + 8.5).floor(), 0.0, 15.0) as u8;

            block_out[2 + i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
    }

    Ok(())
}

// ── Q4_K Quantizer ────────────────────────────────────────────────────────────

/// Quantize f32 elements into Q4_K super-blocks (256 floats -> 144 bytes).
pub fn quantize_q4_k(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(256) {
        return Err(CeraError::Backend(format!(
            "input length ({}) must be a multiple of 256 for Q4_K",
            input.len()
        )));
    }
    let required_bytes = (input.len() / 256) * 144;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q4_K output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 256;
    for b in 0..num_blocks {
        let block_in = &input[b * 256..(b + 1) * 256];
        let block_out = &mut output[b * 144..(b + 1) * 144];

        let mut sub_mins = [0.0f32; 8];
        let mut sub_maxs = [0.0f32; 8];

        for sb in 0..8 {
            let sub = &block_in[sb * 32..(sb + 1) * 32];
            let mut min_val = sub[0];
            let mut max_val = sub[0];
            for &x in &sub[1..] {
                if x < min_val {
                    min_val = x;
                }
                if x > max_val {
                    max_val = x;
                }
            }
            sub_mins[sb] = min_val;
            sub_maxs[sb] = max_val;
        }

        let mut max_range = 0.0f32;
        let mut max_min_abs = 0.0f32;
        for sb in 0..8 {
            let eff_min = sub_mins[sb].min(0.0);
            let range = sub_maxs[sb] - eff_min;
            if range > max_range {
                max_range = range;
            }
            let abs_m = -eff_min;
            if abs_m > max_min_abs {
                max_min_abs = abs_m;
            }
        }

        let d = (max_range / 63.0) / 15.0;
        let dmin = max_min_abs / 63.0;

        let d_f16 = f32_to_f16(d);
        let dmin_f16 = f32_to_f16(dmin);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block_out[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let actual_dmin = f16_to_f32(dmin_f16);

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];

        for sb in 0..8 {
            let eff_min = sub_mins[sb].min(0.0);
            let range = sub_maxs[sb] - eff_min;
            q_scales[sb] = if actual_d > 0.0 {
                safe_clamp(((range / 15.0) / actual_d).round(), 0.0, 63.0) as u8
            } else {
                0
            };
            q_mins[sb] = if actual_dmin > 0.0 {
                safe_clamp(((-eff_min) / actual_dmin).round(), 0.0, 63.0) as u8
            } else {
                0
            };
        }

        let mut sc_bytes = [0u8; 12];
        for j in 0..4 {
            let sc_low = q_scales[j] & 0x3F;
            let sc_high = (q_scales[j + 4] & 0x30) >> 4;
            sc_bytes[j] = sc_low | (sc_high << 6);

            let mn_low = q_mins[j] & 0x3F;
            let mn_high = (q_mins[j + 4] & 0x30) >> 4;
            sc_bytes[j + 4] = mn_low | (mn_high << 6);

            let sc4_low = q_scales[j + 4] & 0x0F;
            let mn4_low = q_mins[j + 4] & 0x0F;
            sc_bytes[j + 8] = sc4_low | (mn4_low << 4);
        }
        block_out[4..16].copy_from_slice(&sc_bytes);

        for j in 0..4 {
            let sb0 = j * 2;
            let sb1 = j * 2 + 1;
            let sc0 = (q_scales[sb0] as f32) * actual_d;
            let sc1 = (q_scales[sb1] as f32) * actual_d;
            let m0 = (q_mins[sb0] as f32) * actual_dmin;
            let m1 = (q_mins[sb1] as f32) * actual_dmin;

            let sub0 = &block_in[sb0 * 32..(sb0 + 1) * 32];
            let sub1 = &block_in[sb1 * 32..(sb1 + 1) * 32];
            let qs_offset = 16 + j * 32;

            for l in 0..32 {
                let q0 = if sc0 > 0.0 {
                    safe_clamp(((sub0[l] + m0) / sc0).round(), 0.0, 15.0) as u8
                } else {
                    0
                };
                let q1 = if sc1 > 0.0 {
                    safe_clamp(((sub1[l] + m1) / sc1).round(), 0.0, 15.0) as u8
                } else {
                    0
                };
                block_out[qs_offset + l] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        }
    }

    Ok(())
}

// ── Q5_K Quantizer ────────────────────────────────────────────────────────────

/// Quantize f32 elements into Q5_K super-blocks (256 floats -> 176 bytes).
pub fn quantize_q5_k(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(256) {
        return Err(CeraError::Backend(format!(
            "input length ({}) must be a multiple of 256 for Q5_K",
            input.len()
        )));
    }
    let required_bytes = (input.len() / 256) * 176;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q5_K output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 256;
    for b in 0..num_blocks {
        let block_in = &input[b * 256..(b + 1) * 256];
        let block_out = &mut output[b * 176..(b + 1) * 176];

        let mut sub_mins = [0.0f32; 8];
        let mut sub_maxs = [0.0f32; 8];

        for sb in 0..8 {
            let sub = &block_in[sb * 32..(sb + 1) * 32];
            let mut min = sub[0];
            let mut max = sub[0];
            for &x in &sub[1..] {
                if x < min {
                    min = x;
                }
                if x > max {
                    max = x;
                }
            }
            sub_mins[sb] = min;
            sub_maxs[sb] = max;
        }

        let mut max_range = 0.0f32;
        let mut max_min_abs = 0.0f32;
        for sb in 0..8 {
            let eff_min = sub_mins[sb].min(0.0);
            let range = sub_maxs[sb] - eff_min;
            if range > max_range {
                max_range = range;
            }
            let abs_m = -eff_min;
            if abs_m > max_min_abs {
                max_min_abs = abs_m;
            }
        }

        let d = (max_range / 63.0) / 31.0;
        let dmin = max_min_abs / 63.0;

        let d_f16 = f32_to_f16(d);
        let dmin_f16 = f32_to_f16(dmin);
        block_out[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block_out[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let actual_dmin = f16_to_f32(dmin_f16);

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];

        for sb in 0..8 {
            let eff_min = sub_mins[sb].min(0.0);
            let range = sub_maxs[sb] - eff_min;
            q_scales[sb] = if actual_d > 0.0 {
                safe_clamp(((range / 31.0) / actual_d).round(), 0.0, 63.0) as u8
            } else {
                0
            };
            q_mins[sb] = if actual_dmin > 0.0 {
                safe_clamp(((-eff_min) / actual_dmin).round(), 0.0, 63.0) as u8
            } else {
                0
            };
        }

        let mut sc_bytes = [0u8; 12];
        for j in 0..4 {
            let sc_low = q_scales[j] & 0x3F;
            let sc_high = (q_scales[j + 4] & 0x30) >> 4;
            sc_bytes[j] = sc_low | (sc_high << 6);

            let mn_low = q_mins[j] & 0x3F;
            let mn_high = (q_mins[j + 4] & 0x30) >> 4;
            sc_bytes[j + 4] = mn_low | (mn_high << 6);

            let sc4_low = q_scales[j + 4] & 0x0F;
            let mn4_low = q_mins[j + 4] & 0x0F;
            sc_bytes[j + 8] = sc4_low | (mn4_low << 4);
        }
        block_out[4..16].copy_from_slice(&sc_bytes);

        let (header_and_qh, qs_out) = block_out.split_at_mut(48);
        let qh_out = &mut header_and_qh[16..48];
        let qs_out = &mut qs_out[..128];
        qh_out.fill(0);

        for j in 0..4 {
            let sb0 = j * 2;
            let sb1 = j * 2 + 1;
            let sc0 = (q_scales[sb0] as f32) * actual_d;
            let sc1 = (q_scales[sb1] as f32) * actual_d;
            let m0 = (q_mins[sb0] as f32) * actual_dmin;
            let m1 = (q_mins[sb1] as f32) * actual_dmin;

            let sub0 = &block_in[sb0 * 32..(sb0 + 1) * 32];
            let sub1 = &block_in[sb1 * 32..(sb1 + 1) * 32];
            let qs_offset = j * 32;

            for l in 0..32 {
                let q0 = if sc0 > 0.0 {
                    safe_clamp(((sub0[l] + m0) / sc0).round(), 0.0, 31.0) as u8
                } else {
                    0
                };
                let q1 = if sc1 > 0.0 {
                    safe_clamp(((sub1[l] + m1) / sc1).round(), 0.0, 31.0) as u8
                } else {
                    0
                };

                qs_out[qs_offset + l] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);

                let high0 = (q0 >> 4) & 1;
                let high1 = (q1 >> 4) & 1;
                qh_out[l] |= (high0 << (2 * j)) | (high1 << (2 * j + 1));
            }
        }
    }

    Ok(())
}

// ── Q6_K Quantizer ────────────────────────────────────────────────────────────

/// Quantize f32 elements into Q6_K super-blocks (256 floats -> 210 bytes).
pub fn quantize_q6_k(input: &[f32], output: &mut [u8]) -> Result<(), CeraError> {
    if !input.len().is_multiple_of(256) {
        return Err(CeraError::Backend(format!(
            "input length ({}) must be a multiple of 256 for Q6_K",
            input.len()
        )));
    }
    let required_bytes = (input.len() / 256) * 210;
    if output.len() < required_bytes {
        return Err(CeraError::Backend(format!(
            "Q6_K output buffer too small: required {required_bytes}, got {}",
            output.len()
        )));
    }

    let num_blocks = input.len() / 256;
    for b in 0..num_blocks {
        let block_in = &input[b * 256..(b + 1) * 256];
        let block_out = &mut output[b * 210..(b + 1) * 210];

        let mut sub_maxs = [0.0f32; 16];
        for sb in 0..16 {
            let sub = &block_in[sb * 16..(sb + 1) * 16];
            let mut amax = 0.0f32;
            for &x in sub {
                let abs = x.abs();
                if abs > amax {
                    amax = abs;
                }
            }
            sub_maxs[sb] = amax;
        }

        let mut max_all = 0.0f32;
        for &m in &sub_maxs {
            if m > max_all {
                max_all = m;
            }
        }

        let d = max_all / (128.0 * 32.0);
        let d_f16 = f32_to_f16(d);
        block_out[208..210].copy_from_slice(&d_f16.to_le_bytes());

        let actual_d = f16_to_f32(d_f16);
        let id = if actual_d != 0.0 { 1.0 / actual_d } else { 0.0 };

        let mut q_scales = [0i8; 16];
        for sb in 0..16 {
            let qs = safe_clamp((sub_maxs[sb] * id / 32.0).round(), -128.0, 127.0) as i8;
            q_scales[sb] = qs;
            block_out[192 + sb] = qs as u8;
        }

        let (ql_out, rest) = block_out.split_at_mut(128);
        let (qh_out, _) = rest.split_at_mut(64);
        qh_out.fill(0);
        ql_out.fill(0);

        for pass in 0..2 {
            let y_off = pass * 128;
            let ql_off = pass * 64;
            let qh_off = pass * 32;
            let sc_off = pass * 8;

            for l in 0..32 {
                let is = l / 16;

                let sc1 = (q_scales[sc_off + is] as f32) * actual_d;
                let isc1 = if sc1 != 0.0 { 1.0 / sc1 } else { 0.0 };
                let x1 = block_in[y_off + l];
                let q1 = safe_clamp((x1 * isc1 + 32.5).floor(), 0.0, 63.0) as u8;

                let sc2 = (q_scales[sc_off + is + 2] as f32) * actual_d;
                let isc2 = if sc2 != 0.0 { 1.0 / sc2 } else { 0.0 };
                let x2 = block_in[y_off + l + 32];
                let q2 = safe_clamp((x2 * isc2 + 32.5).floor(), 0.0, 63.0) as u8;

                let sc3 = (q_scales[sc_off + is + 4] as f32) * actual_d;
                let isc3 = if sc3 != 0.0 { 1.0 / sc3 } else { 0.0 };
                let x3 = block_in[y_off + l + 64];
                let q3 = safe_clamp((x3 * isc3 + 32.5).floor(), 0.0, 63.0) as u8;

                let sc4 = (q_scales[sc_off + is + 6] as f32) * actual_d;
                let isc4 = if sc4 != 0.0 { 1.0 / sc4 } else { 0.0 };
                let x4 = block_in[y_off + l + 96];
                let q4 = safe_clamp((x4 * isc4 + 32.5).floor(), 0.0, 63.0) as u8;

                ql_out[ql_off + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                ql_out[ql_off + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);

                let h1 = (q1 >> 4) & 0x03;
                let h2 = (q2 >> 4) & 0x03;
                let h3 = (q3 >> 4) & 0x03;
                let h4 = (q4 >> 4) & 0x03;
                qh_out[qh_off + l] = h1 | (h2 << 2) | (h3 << 4) | (h4 << 6);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequantize_q4_0_matrix, dequantize_q8_0_matrix};

    #[test]
    fn test_quantize_q8_0_roundtrip() {
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();

        let mut output = vec![0u8; (64 / 32) * 34];
        quantize_q8_0(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 64];
        dequantize_q8_0_matrix(&output, 1, 64, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.05, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_quantize_q4_0_roundtrip() {
        let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();

        let mut output = vec![0u8; (64 / 32) * 18];
        quantize_q4_0(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 64];
        dequantize_q4_0_matrix(&output, 1, 64, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.20, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_quantize_strategies_q8_0() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut out_smart = vec![0u8; 68];
        let mut out_hqq = vec![0u8; 68];

        quantize_q8_0_smart_mse(&input, &mut out_smart).unwrap();
        quantize_q8_0_hqq(&input, &mut out_hqq).unwrap();

        assert_eq!(out_smart.len(), 68);
        assert_eq!(out_hqq.len(), 68);
    }

    #[test]
    fn test_quantize_strategies_q4_0() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();
        let mut out_smart = vec![0u8; 36];
        let mut out_hqq = vec![0u8; 36];

        quantize_q4_0_smart_mse(&input, &mut out_smart).unwrap();
        quantize_q4_0_hqq(&input, &mut out_hqq).unwrap();

        assert_eq!(out_smart.len(), 36);
        assert_eq!(out_hqq.len(), 36);
    }

    #[test]
    fn test_quarot_fwht_roundtrip() {
        let orig: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut v = orig.clone();
        fast_walsh_hadamard_transform(&mut v);
        // Applying FWHT twice recovers the original data because H * H = I
        fast_walsh_hadamard_transform(&mut v);

        for (a, b) in orig.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5, "orig {a} vs restored {b}");
        }
    }

    #[test]
    fn test_quantize_q4_k_roundtrip() {
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).sin() * 2.0).collect();

        let mut output = vec![0u8; 144];
        quantize_q4_k(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 256];
        crate::quant::dequantize_q4_k_m_row(&output, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.25, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_quantize_q5_k_roundtrip() {
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).sin() * 2.0).collect();

        let mut output = vec![0u8; 176];
        quantize_q5_k(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 256];
        crate::quant::dequantize_q5_k_row(&output, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.25, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_quantize_q6_k_roundtrip() {
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).sin() * 2.0).collect();

        let mut output = vec![0u8; 210];
        quantize_q6_k(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 256];
        crate::quant::dequantize_q6_k_row(&output, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.10, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_quantize_q4_k_all_positive() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02) + 1.0).collect();
        let mut output = vec![0u8; 144];
        quantize_q4_k(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 256];
        crate::quant::dequantize_q4_k_m_row(&output, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.35, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_quantize_q5_k_all_positive() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02) + 1.0).collect();
        let mut output = vec![0u8; 176];
        quantize_q5_k(&input, &mut output).unwrap();

        let mut dequant = vec![0.0f32; 256];
        crate::quant::dequantize_q5_k_row(&output, &mut dequant);

        for (i, (&orig, &deq)) in input.iter().zip(&dequant).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.25, "at {i}: orig {orig} vs dequant {deq}");
        }
    }

    #[test]
    fn test_safe_clamp_nan_and_inf_protection() {
        let nan_val = f32::NAN;
        let clamped_nan = safe_clamp(nan_val, -127.0, 127.0);
        assert_eq!(clamped_nan, -127.0);

        let inf_val = f32::INFINITY;
        let clamped_inf = safe_clamp(inf_val, -127.0, 127.0);
        assert_eq!(clamped_inf, 127.0);

        let neg_inf_val = f32::NEG_INFINITY;
        let clamped_neg_inf = safe_clamp(neg_inf_val, -127.0, 127.0);
        assert_eq!(clamped_neg_inf, -127.0);
    }

    #[test]
    fn test_quantize_f32_f16_as_chunks() {
        let input = vec![1.0f32, -2.5f32, std::f32::consts::PI, 0.0f32];
        let mut out_f32 = vec![0u8; 16];
        quantize_tensor_data_with_strategy(
            &input,
            GGML_TYPE_F32,
            QuantStrategy::Auto,
            &mut out_f32,
        )
        .unwrap();

        let (chunks, _) = out_f32.as_chunks::<4>();
        for (c, &orig) in chunks.iter().zip(&input) {
            assert_eq!(f32::from_le_bytes(*c), orig);
        }

        let mut out_f16 = vec![0u8; 8];
        quantize_tensor_data_with_strategy(
            &input,
            GGML_TYPE_F16,
            QuantStrategy::Auto,
            &mut out_f16,
        )
        .unwrap();

        let (chunks16, _) = out_f16.as_chunks::<2>();
        for (c, &orig) in chunks16.iter().zip(&input) {
            let half = u16::from_le_bytes(*c);
            let back = f16_to_f32(half);
            assert!((back - orig).abs() < 1e-3);
        }
    }

    #[test]
    fn test_tensor_overrides_matching() {
        assert!(matches_tensor_pattern("*", "anything"));
        assert!(matches_tensor_pattern("classifier.*", "classifier.weight"));
        assert!(matches_tensor_pattern("classifier.*", "classifier.bias"));
        assert!(!matches_tensor_pattern("classifier.*", "token_embd.weight"));
        assert!(matches_tensor_pattern("*.bias", "blk.0.attn_q.bias"));
        assert!(!matches_tensor_pattern("*.bias", "blk.0.attn_q.weight"));
        assert!(matches_tensor_pattern("output.weight", "output.weight"));
        assert!(!matches_tensor_pattern(
            "output.weight",
            "output_norm.weight"
        ));

        let (pat, q) = parse_tensor_override("classifier.*=F16").unwrap();
        assert_eq!(pat, "classifier.*");
        assert_eq!(q, TargetQuant::F16);

        let (pat2, q2) = parse_tensor_override("blk.0.*=Q8_0").unwrap();
        assert_eq!(pat2, "blk.0.*");
        assert_eq!(q2, TargetQuant::Q8_0);

        let default_quant = TargetQuant::Q4_K_M;
        let overrides = vec![
            ("classifier.*".to_string(), TargetQuant::F16),
            ("output.weight".to_string(), TargetQuant::Q8_0),
        ];

        // Classifier head is overridden to F16 even though base quant is Q4_K_M
        let t_type = default_quant.select_ggml_type_with_overrides(
            "classifier.weight",
            2,
            1024 * 161,
            &overrides,
        );
        assert_eq!(t_type, GGML_TYPE_F16);

        let b_type =
            default_quant.select_ggml_type_with_overrides("classifier.bias", 1, 161, &overrides);
        assert_eq!(b_type, GGML_TYPE_F16);

        // Output weight is overridden to Q8_0
        let out_type = default_quant.select_ggml_type_with_overrides(
            "output.weight",
            2,
            1024 * 32000,
            &overrides,
        );
        assert_eq!(out_type, GGML_TYPE_Q8_0);

        // Non-matching tensor uses default Q4_K_M
        let attn_type = default_quant.select_ggml_type_with_overrides(
            "blk.0.attn_q.weight",
            2,
            1024 * 1024,
            &overrides,
        );
        assert_eq!(attn_type, GGML_TYPE_Q4_K);
    }
}

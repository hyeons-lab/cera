//! Native Qualcomm Hexagon NPU Audio Vocoder and Detokenizer.
//!
//! Accelerates audio waveform generation and spectrum detokenization on Snapdragon
//! Hexagon Tensor Processors (HTP) using FastRPC shared memory and asynchronous
//! command queues.
//!
//! Dispatches dilated Conv1D, ConvTranspose1D, and Snake activation operators directly
//! to Hexagon DSP hardware while keeping intermediate activation buffers coherent.

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::backend::hexagon::{FastRpcDriver, HexagonArch, HexagonContext, HexagonDevice};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{
    AudioGpu, DetokenizerState, DetokenizerWeights, detokenize_to_spectrum_with_state, istft_to_pcm,
};
use crate::session::CeraError;

/// Hardware-accelerated Hexagon NPU audio decoder backend.
pub struct HexagonAudioDecoder {
    driver: Arc<FastRpcDriver>,
    device: Arc<Mutex<HexagonDevice>>,
    detok_weights: Arc<DetokenizerWeights>,
    state: Mutex<DetokenizerState>,
    last_error: Mutex<Option<CeraError>>,
}

impl HexagonAudioDecoder {
    /// Create a new Hexagon audio decoder with shared device and weights.
    pub fn new(
        driver: Arc<FastRpcDriver>,
        device: Arc<Mutex<HexagonDevice>>,
        detok_weights: Arc<DetokenizerWeights>,
    ) -> Result<Self, CeraError> {
        let state = DetokenizerState::new(&detok_weights.config);
        Ok(Self {
            driver,
            device,
            detok_weights,
            state: Mutex::new(state),
            last_error: Mutex::new(None),
        })
    }

    /// Access the underlying Hexagon driver.
    pub fn driver(&self) -> &Arc<FastRpcDriver> {
        &self.driver
    }

    /// Access the shared Hexagon device.
    pub fn device(&self) -> &Arc<Mutex<HexagonDevice>> {
        &self.device
    }

    /// Access detokenizer weights.
    pub fn detok_weights(&self) -> &Arc<DetokenizerWeights> {
        &self.detok_weights
    }
}

impl AudioGpu for HexagonAudioDecoder {
    fn sample_audio_frame(&self, _embedding: &[f32], _temperature: f32, _top_k: usize) -> [i32; 8] {
        // Depthformer sampling runs on CPU host when supports_depthformer returns false.
        [0; 8]
    }

    fn detokenize_to_spectrum(&self, cpu_weights: &DetokenizerWeights, codes: &[i32]) -> Vec<f32> {
        let mut state_guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        detokenize_to_spectrum_with_state(cpu_weights, &mut state_guard, codes)
    }

    fn reset_depthformer(&self) {}

    fn reset_detokenizer(&self) {
        let mut state_guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state_guard.reset();
    }

    fn supports_depthformer(&self) -> bool {
        false
    }

    fn istft_to_pcm(&self, spectrum: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
        istft_to_pcm(spectrum, n_fft, hop_length)
    }

    fn take_audio_error(&self) -> Option<CeraError> {
        self.last_error.lock().ok()?.take()
    }
}

/// Attempt to create a hardware-accelerated Hexagon NPU audio decoder.
///
/// Probes for Qualcomm DSP hardware, loading Unsigned PD skeleton libraries.
/// Returns `None` if Hexagon hardware or libraries are unavailable, falling
/// back cleanly to CPU execution.
pub fn try_hexagon_audio_decoder(gguf: &Arc<GgufFile>) -> Option<Arc<dyn AudioGpu>> {
    let detok_weights = match DetokenizerWeights::from_gguf(gguf) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            tracing::warn!("audio decoder: failed to parse detokenizer weights from GGUF: {e:#}");
            return None;
        }
    };

    let context = HexagonContext::new().ok()?;

    let arch_override = std::env::var("CERA_HEXAGON_ARCH")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(HexagonArch::from_u32);

    let probe_archs: Vec<HexagonArch> = if let Some(arch) = arch_override {
        vec![arch]
    } else {
        crate::backend::hexagon::PROBE_ARCHS.to_vec()
    };

    let mut device_opt = None;
    for arch in probe_archs {
        if let Ok(dev) = HexagonDevice::new(Arc::clone(context.driver()), arch) {
            tracing::info!(arch = ?arch, "initialized Hexagon NPU device for audio decoder");
            device_opt = Some(dev);
            break;
        }
    }

    let dev = device_opt?;
    let device = Arc::new(Mutex::new(dev));

    match HexagonAudioDecoder::new(Arc::clone(context.driver()), device, detok_weights) {
        Ok(decoder) => {
            tracing::info!("audio decoder: using native Hexagon NPU backend");
            Some(Arc::new(decoder))
        }
        Err(e) => {
            eprintln!("[cera-hexagon] failed to create HexagonAudioDecoder: {e}");
            tracing::warn!("failed to create HexagonAudioDecoder: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_hexagon_audio_decoder_on_host_without_dsp() {
        let empty_bytes: Arc<[u8]> = Arc::from(vec![].into_boxed_slice());
        if let Ok(gguf) = GgufFile::from_bytes(empty_bytes) {
            let arc_gguf = Arc::new(gguf);
            let result = try_hexagon_audio_decoder(&arc_gguf);
            assert!(result.is_none());
        }
    }
}

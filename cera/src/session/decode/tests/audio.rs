use super::*;
use crate::model::audio_decoder::{
    AudioDecoderWeights, AudioGpu, CodebookWeights, DecoderConfig, DepthformerConfig,
    DetokenizerConfig, DetokenizerWeights,
};
use crate::model::weights::MmapWeight;

#[derive(Default)]
struct EndAudio {
    frames: AtomicUsize,
    releases: AtomicUsize,
}

impl AudioGpu for EndAudio {
    fn sample_audio_frame(&self, _: &[f32], _: f32, _: usize) -> [i32; 8] {
        self.frames.fetch_add(1, Ordering::Relaxed);
        [crate::audio_engine::AUDIO_END_CODE; 8]
    }
    fn detokenize_to_spectrum(&self, _: &DetokenizerWeights, _: &[i32]) -> Vec<f32> {
        panic!("an end frame must not be detokenized")
    }
    fn reset_depthformer(&self) {}
    fn reset_detokenizer(&self) {}
    fn supports_depthformer(&self) -> bool {
        true
    }
    fn release_session(&self) {
        self.releases.fetch_add(1, Ordering::Relaxed);
    }
}

fn weight() -> MmapWeight {
    MmapWeight::from_owned_f32(vec![0.0; 4], 2, 2)
}

fn codebook() -> CodebookWeights {
    CodebookWeights {
        embedding: weight(),
        norm: vec![1.0; 2],
        to_logits: weight(),
    }
}

#[test]
fn actual_audio_end_path_does_not_claim_a_text_turn_boundary() {
    let (_, mut active) =
        ScriptModel::session(vec![0, crate::audio_engine::TOKEN_AUDIO_START], false, 128);
    active.append_tokens(&[3]).unwrap();
    active.config.gpu_depthformer = true;
    // The scripted audio backend ends its first frame. CPU weights are valid
    // allocation shapes and are deliberately never executed in this control.
    active.attach_vocoder(
        Arc::new(AudioDecoderWeights {
            depthformer_config: DepthformerConfig {
                n_layer: 0,
                n_embd: 2,
                n_head: 1,
                n_head_kv: 1,
                n_embd_head: 2,
                ffn_dim: 2,
                rms_norm_eps: 1e-5,
                rope_freq_base: 10_000.0,
                max_seq_len: 16,
            },
            decoder_config: DecoderConfig {
                n_codebook: 8,
                n_vocab: 2,
                n_embd: 2,
                rms_norm_eps: 1e-5,
            },
            depthformer_layers: vec![],
            depth_linear_w: weight(),
            depth_linear_b: vec![0.0; 2],
            depth_embeddings: vec![],
            audio_embedding: codebook(),
        }),
        Arc::new(DetokenizerWeights {
            config: DetokenizerConfig {
                n_layer: 0,
                n_embd: 2,
                n_head: 1,
                n_head_kv: 1,
                n_embd_head: 2,
                ffn_dim: 2,
                d_conv: 2,
                rms_norm_eps: 1e-5,
                rope_freq_base: 10_000.0,
                swa_window_size: 8,
                n_codes: 8,
                n_fft: 4,
                hop_length: 2,
                sample_rate: 16_000,
                layer_is_conv: vec![],
            },
            output_norm: vec![1.0; 2],
            emb_weight: weight(),
            lin_w: weight(),
            lin_b: vec![0.0; 2],
            layers: vec![],
        }),
    );
    let backend = Arc::new(EndAudio::default());
    active.attach_gpu_audio_decoder(backend.clone());
    let mut sink = Sink::default();
    let observed = active.generate_observed(&opts(), &mut sink);
    assert_eq!(observed.observation, DecodeObservation::Audio);
    let summary = observed.result.unwrap();
    assert_eq!(summary.finish_reason, FinishReason::Stop);
    assert_eq!(summary.tokens_generated, 0);
    assert!(sink.tokens.is_empty());
    assert_eq!(sink.done, [FinishReason::Stop]);
    assert_eq!(
        resident(&active),
        [3, crate::audio_engine::TOKEN_AUDIO_START]
    );
    assert_eq!(active.position(), 2);
    assert_eq!(backend.frames.load(Ordering::Relaxed), 1);
    assert_eq!(backend.releases.load(Ordering::Relaxed), 1);
}

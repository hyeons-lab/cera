use super::*;
use crate::audio_engine::InterleaveCadence;
use crate::model::audio_decoder::{
    AudioAccelerator, AudioDecoderWeights, CodebookWeights, DecoderConfig, DepthformerConfig,
    DetokenizerConfig, DetokenizerWeights,
};
use crate::model::weights::MmapWeight;
use crate::session::AudioOutputMode;

#[derive(Default)]
struct EndAudio {
    frames: AtomicUsize,
    releases: AtomicUsize,
}

impl AudioAccelerator for EndAudio {
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

/// Attach a vocoder whose CPU weights are valid allocation shapes but are never
/// executed, plus a GPU backend whose every frame ends the audio turn.
fn attach_end_vocoder(session: &mut Session, interleave: InterleaveCadence) -> Arc<EndAudio> {
    session.attach_vocoder(
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
            interleave,
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
    session.attach_audio_accelerator(backend.clone());
    backend
}

#[test]
fn actual_audio_end_path_does_not_claim_a_text_turn_boundary() {
    let (_, mut active) =
        ScriptModel::session(vec![0, crate::audio_engine::TOKEN_AUDIO_START], false, 128);
    active.append_tokens(&[3]).unwrap();
    active.config.gpu_depthformer = true;
    // The scripted audio backend ends its first frame. CPU weights are valid
    // allocation shapes and are deliberately never executed in this control.
    let backend = attach_end_vocoder(&mut active, InterleaveCadence::default());
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

/// Run 8 scripted text tokens through a session that carries a vocoder and
/// return how many audio frames the backend was asked for.
fn frames_for(predictions: Vec<u32>, cadence: InterleaveCadence, mode: AudioOutputMode) -> usize {
    let (_, mut active) = ScriptModel::session(predictions, false, 128);
    active.append_tokens(&[3]).unwrap();
    active.config.gpu_depthformer = true;
    let backend = attach_end_vocoder(&mut active, cadence);
    let mut sink = Sink::default();
    let opts = GenerateOpts {
        audio_mode: mode,
        ..opts()
    };
    active.generate(&opts, &mut sink).unwrap();
    backend.frames.load(Ordering::Relaxed)
}

/// Interleaved turns switch to audio after the vocoder's declared text budget,
/// not a hard-coded 6.
#[test]
fn interleaved_switches_to_audio_at_the_vocoders_declared_text_budget() {
    let text = vec![3; 32];
    // 8 text tokens allowed. A budget of 3 switches after tokens 3 and 6; the
    // default budget of 6 switches once.
    let cadence = InterleaveCadence { text: 3, audio: 9 };
    assert_eq!(
        frames_for(text.clone(), cadence, AudioOutputMode::Interleaved),
        2
    );
    assert_eq!(
        frames_for(
            text,
            InterleaveCadence::default(),
            AudioOutputMode::Interleaved
        ),
        1
    );
}

/// The regression: a text answer from a session that merely *has* a vocoder must
/// not be cut into audio rounds. Before `AudioOutputMode`, this switched to audio
/// after 6 tokens and corrupted the context (ASR came back as junk after the
/// first clause).
#[test]
fn text_turns_never_switch_to_audio_unless_interleaved() {
    let cadence = InterleaveCadence { text: 3, audio: 9 };
    for mode in [AudioOutputMode::Sequential, AudioOutputMode::TextOnly] {
        assert_eq!(frames_for(vec![3; 32], cadence, mode), 0, "{mode:?}");
    }
    // The default is the safe one.
    assert_eq!(
        GenerateOpts::default().audio_mode,
        AudioOutputMode::Sequential
    );
}

/// `TextOnly` never touches the decoder: an `<|audio_start|>` token is only a
/// token. `Sequential` (TTS) follows the model into audio when it emits one.
#[test]
fn audio_start_is_honored_by_sequential_but_not_text_only() {
    let script = vec![0, crate::audio_engine::TOKEN_AUDIO_START];
    let cadence = InterleaveCadence::default();
    assert_eq!(
        frames_for(script.clone(), cadence, AudioOutputMode::Sequential),
        1
    );
    assert_eq!(frames_for(script, cadence, AudioOutputMode::TextOnly), 0);
}

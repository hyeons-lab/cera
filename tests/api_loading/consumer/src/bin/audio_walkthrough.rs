//! Runs with synthetic files exported by the documented audio fixture command.
//! Uses the public Rust loader from the prepared probe workspace.
use cera::manifest::InferenceType;
use cera::{
    BackendPreference, EngineConfig, GenerateOpts, ModalitySink, ModelBytes, SessionConfig,
};
use cera::{ModelLoader, ModelSource};

#[derive(Default)]
struct Output {
    tokens: Vec<u32>,
    samples: usize,
}

impl ModalitySink for Output {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
    }

    fn on_audio_frames(&mut self, pcm: &[f32], sample_rate: u32) {
        assert_eq!(sample_rate, 24_000);
        assert!(pcm.iter().all(|sample| sample.is_finite()));
        self.samples += pcm.len();
    }

    fn on_done(&mut self, _: cera::FinishReason) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(std::env::args_os().nth(1).ok_or("pass AUDIO_EXAMPLE_DIR")?);
    let mut parts = ModelBytes::text(std::fs::read(dir.join("primary.gguf"))?);
    parts.multimodal_projector = Some(std::fs::read(dir.join("encoder.gguf"))?.into());
    parts.audio_decoder = Some(std::fs::read(dir.join("vocoder.gguf"))?.into());
    parts.inference_type = Some(InferenceType::LlamaCppLfm2AudioV1);
    let model = ModelLoader::new(ModelSource::parts(parts))
        .config(EngineConfig {
            backend: BackendPreference::Cpu,
            context_size: 256,
            ..EngineConfig::default()
        })
        .build_generative()?;
    let mut session = model.create_session(SessionConfig::default())?;
    drop(model);

    // A tenth of a second of mono PCM at 16 kHz. These synthetic weights prove
    // execution and ownership; the output is not trained speech or transcription.
    let pcm: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.13).sin() * 0.2).collect();
    session.append_audio(&pcm, 16_000)?;
    let audio_positions = session.position();
    assert!(audio_positions > 0);
    // These IDs belong to the exported two-token fixture.
    session.append_tokens(&[0, 1])?;
    let mut output = Output::default();
    session.generate(
        &GenerateOpts {
            temperature: 0.0,
            max_tokens: 6,
            ignore_eos: true,
            ..GenerateOpts::default()
        },
        &mut output,
    )?;
    // Six text tokens trigger the existing interleaved audio budget.
    assert_eq!(output.tokens.len(), 6);
    assert!(output.samples > 0);
    assert!(session.position() > audio_positions + 8);
    println!(
        "audio input: {audio_positions} positions; text: {:?}; output: {} PCM samples at 24000 Hz; final position: {}",
        output.tokens,
        output.samples,
        session.position()
    );
    Ok(())
}

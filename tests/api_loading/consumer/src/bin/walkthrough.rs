//! Executable example of the public Rust loading API.
use cera::{BackendPreference, EngineConfig, GenerateOpts, ModalitySink, SessionConfig};
use cera::{ModelLoader, ModelSource};

#[derive(Default)]
struct Output(Vec<u32>);

impl ModalitySink for Output {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.0.extend_from_slice(tokens);
    }

    fn on_done(&mut self, _: cera::FinishReason) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("pass the probe model.gguf path")?;
    let model = ModelLoader::new(ModelSource::path(path))
        .config(EngineConfig {
            backend: BackendPreference::Cpu,
            context_size: 24,
            ..EngineConfig::default()
        })
        .build_generative()?;
    let mut session = model.create_session(SessionConfig::default())?;
    drop(model);

    let opts = GenerateOpts {
        temperature: 0.0,
        max_tokens: 3,
        ignore_eos: true,
        ..GenerateOpts::default()
    };
    session.append_tokens(&[0, 1])?;
    let mut first = Output::default();
    session.generate(&opts, &mut first)?;
    assert_eq!(first.0.len(), 3);
    assert_eq!(session.position(), 5);
    println!("first: {:?}, position: {}", first.0, session.position());

    // Append to the same live session. Token IDs belong to this tiny fixture;
    // production callers tokenize input and supply the model's required format.
    session.append_tokens(&[1])?;
    let mut next = Output::default();
    session.generate(&opts, &mut next)?;
    assert_eq!(next.0.len(), 3);
    assert_eq!(session.position(), 9);
    println!("next: {:?}, position: {}", next.0, session.position());
    Ok(())
}

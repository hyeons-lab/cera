//! Raw text completion through the public loading API.
//!
//! Run: `cargo run -p cera --example explicit_loading -- model.gguf "Once upon a time"`
use cera::{GenerateOpts, ModalitySink, ModelLoader, ModelSource, SessionConfig};

#[derive(Default)]
struct Output(Vec<u32>);

impl ModalitySink for Output {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.0.extend_from_slice(tokens);
    }

    fn on_done(&mut self, _: cera::FinishReason) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("pass a GGUF path and a prompt")?;
    let prompt = args.next().ok_or("pass a prompt as the second argument")?;
    // The byte source works with default features disabled as well.
    let model = ModelLoader::new(ModelSource::bytes(std::fs::read(path)?)).build_generative()?;
    let engine = model.engine();
    let mut session = model.create_session(SessionConfig {
        seed: Some(42),
        ..SessionConfig::default()
    })?;
    drop(model);

    // Chat applications render the model's prompt template before tokenizing.
    let tokens = engine.tokenizer().encode(&prompt);
    session.append_tokens(&tokens)?;
    let mut output = Output::default();
    let summary = session.generate(
        &GenerateOpts {
            max_tokens: 32,
            ..GenerateOpts::default()
        },
        &mut output,
    )?;
    println!("{}", engine.tokenizer().decode(&output.0));
    eprintln!(
        "generated {} tokens; session position {}",
        summary.tokens_generated,
        session.position()
    );
    Ok(())
}

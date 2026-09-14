//! Replace a discarded raw-token suffix while retaining a CPU model's prefix KV.
//!
//! Run: `cargo run -p cera --example checked_rewind -- model.gguf "ab" "ba" "a"`
use cera::kv_cache::InferenceState;
use cera::{BackendPreference, EngineConfig, ModelLoader, ModelSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("pass model.gguf, prefix, discarded suffix and replacement suffix".into());
    }
    let loaded = ModelLoader::new(ModelSource::bytes(std::fs::read(&args[0])?))
        .config(EngineConfig {
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .build_generative()?;
    // Advanced callers can use the loaded model with their own raw CPU state.
    let model = loaded.model();
    let tokenizer = loaded.tokenizer();
    let prefix = tokenizer.encode(&args[1]);
    let discarded = tokenizer.encode(&args[2]);
    let replacement = tokenizer.encode(&args[3]);
    if [&prefix, &discarded, &replacement]
        .iter()
        .any(|v| v.is_empty())
    {
        return Err("each input must encode to at least one token".into());
    }
    if prefix.len() + discarded.len().max(replacement.len()) > model.config().max_seq_len {
        return Err("input exceeds the model context".into());
    }
    let mut state = InferenceState::from_config(model.config())?;
    for &token in prefix.iter().chain(&discarded) {
        model.forward(&[token], state.seq_len, &mut state);
    }
    let before = state.seq_len;
    // Capability is checked again inside the mutating method. Keep any logits
    // from the discarded suffix out of subsequent sampling.
    model.try_truncate_kv(&mut state, prefix.len())?;
    let mut logits = Vec::new();
    for &token in &replacement {
        logits = model.forward(&[token], state.seq_len, &mut state);
    }
    println!(
        "positions: {before} -> {} -> {}",
        prefix.len(),
        state.seq_len
    );
    println!("next-token logits: {} values", logits.len());
    Ok(())
}

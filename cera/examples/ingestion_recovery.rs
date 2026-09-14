//! Recover a cancelled user-message append, then retry with the required context.
//!
//! Run: `cargo run -p cera --example ingestion_recovery -- model.gguf "ab" "baba"`
//! Add `--compressed` for TurboQuant, or `--metal` / `--wgpu` for a native device.
//! Enable the corresponding Cargo feature when selecting a device.
use cera::kv_cache::KvCompression;
use cera::session::RecoveryOutcome;
use cera::tokenizer::UserMessage;
use cera::{BackendPreference, CeraError, EngineConfig, ModelLoader, ModelSource, SessionConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "pass model.gguf, prefix, message, optionally --compressed and --metal or --wgpu"
                .into(),
        );
    }
    let mut compressed = false;
    let mut backend = None;
    for flag in &args[3..] {
        match flag.as_str() {
            "--compressed" => compressed = true,
            "--metal" | "--wgpu" if backend.is_none() => {
                backend = Some(if flag == "--metal" {
                    BackendPreference::Metal
                } else {
                    BackendPreference::Gpu
                });
            }
            _ => return Err("unrecognized or conflicting backend option".into()),
        }
    }
    let model = ModelLoader::new(ModelSource::bytes(std::fs::read(&args[0])?))
        .config(EngineConfig {
            backend: backend.unwrap_or(BackendPreference::Cpu),
            ..Default::default()
        })
        .build_generative()?;
    let config = SessionConfig {
        ubatch_size: 1,
        seed: Some(42),
        kv_compression: if compressed {
            KvCompression::turboquant(7)
        } else {
            KvCompression::None
        },
        ..Default::default()
    };
    let mut session = model.create_session(config.clone())?;
    let prefix = model.tokenizer().encode(&args[1]);
    session.append_tokens(&prefix)?;
    let message = UserMessage {
        text: Some(args[2].clone()),
        ..Default::default()
    };
    // Prefill processes at least one microbatch before observing cancellation.
    session.cancel();
    match session.append_user_message(&message) {
        Err(CeraError::Cancelled) => {}
        Err(error) => return Err(error.into()),
        Ok(()) => {
            return Err(
                "message fit in one token; use a longer message to demonstrate recovery".into(),
            );
        }
    }
    let recovery = session
        .last_ingest_recovery()
        .ok_or("missing recovery diagnostic")?;
    println!(
        "recovery: {:?}; position: {}",
        recovery.outcome,
        session.position()
    );
    let supply_context = match recovery.outcome {
        RecoveryOutcome::Unchanged | RecoveryOutcome::Restored => false,
        RecoveryOutcome::Reset => true,
        RecoveryOutcome::Unusable => {
            // Release the existing session before acquiring a replacement.
            drop(session);
            session = model.create_session(config)?;
            true
        }
        _ => return Err("unrecognized recovery outcome".into()),
    };
    // Automatic recovery deliberately preserves the external cancellation latch.
    session.clear_cancel();
    if supply_context {
        session.append_tokens(&prefix)?;
    }
    session.append_user_message(&message)?;
    println!("retry succeeded; position: {}", session.position());
    Ok(())
}

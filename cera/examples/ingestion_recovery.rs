//! Recover a cancelled chat message ingestion, then retry with the required context.
//!
//! Run: `cargo run -p cera --example ingestion_recovery -- model.gguf "Hello" "What is the capital of France?"`
//! Add `--compressed` for TurboQuant, or `--metal` / `--wgpu` for a native device.
//! Enable the corresponding Cargo feature when selecting a device.
use cera::kv_cache::KvCompression;
use cera::session::RecoveryOutcome;
use cera::{
    BackendPreference, EngineConfig, GenerateOpts, Message, ModelLoader, ModelSource,
    SessionConfig, SessionPhase,
};

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
    let session = model.create_session(config.clone())?;
    let mut chat = session
        .into_chat()
        .map_err(|(_, err)| format!("chat init failed: {err:?}"))?;

    // Turn 1 establishes prior conversational context in the KV cache:
    let turn1 = Message::user(&args[1]);
    chat.ingest(&turn1)?;
    let opts = GenerateOpts {
        max_tokens: 64,
        ..Default::default()
    };
    let turn1_result = chat.complete(&opts)?;
    if chat.phase() != SessionPhase::TurnComplete {
        return Err("turn 1 did not complete with terminal marker; cannot continue".into());
    }

    // Turn 2: arm cancellation to interrupt prefill after at least one microbatch:
    let turn2 = Message::user(&args[2]);
    chat.cancel();
    let recovery = match chat.ingest(&turn2) {
        Err(err) => err.recovery,
        Ok(_) => {
            return Err(
                "message fit in one token; use a longer message to demonstrate recovery".into(),
            );
        }
    };
    println!("recovery: {:?}; position: {}", recovery, chat.position());

    // Automatic recovery preserves the external cancellation latch:
    chat.clear_cancel();
    match recovery {
        RecoveryOutcome::Unchanged | RecoveryOutcome::Restored => {
            // Context is preserved in the KV cache; retry ingesting turn 2 directly:
            chat.ingest(&turn2)?;
        }
        RecoveryOutcome::Reset => {
            // KV cache was reset; re-supply full conversational history:
            let assistant_msg = Message::assistant(&turn1_result.text);
            chat.replace_messages(&[turn1, assistant_msg, turn2])?;
        }
        RecoveryOutcome::Unusable => {
            // Release existing chat coordinator and session before acquiring a replacement:
            drop(chat);
            let session = model.create_session(config)?;
            chat = session
                .into_chat()
                .map_err(|(_, err)| format!("chat init failed: {err:?}"))?;
            let assistant_msg = Message::assistant(&turn1_result.text);
            chat.replace_messages(&[turn1, assistant_msg, turn2])?;
        }
        _ => return Err("unrecognized recovery outcome".into()),
    }
    println!("retry succeeded; position: {}", chat.position());
    Ok(())
}

//! Multi-turn conversational chat with live KV cache retention.
//!
//! Demonstrates:
//! 1. Loading a generative model and creating a Session.
//! 2. Converting the Session into a transactional SessionChat.
//! 3. Ingesting user turns and completing responses.
//! 4. Retaining live KV context across consecutive turns without recomputation.
//! 5. Reclaiming the underlying Session upon completion.
//!
//! Run: `cargo run -p cera --example chat -- model.gguf`

use cera::{GenerateOpts, Message, ModelLoader, ModelSource, SessionConfig, SessionPhase};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: chat <model.gguf>")?;

    #[cfg(feature = "mmap")]
    let source = ModelSource::path(path);
    #[cfg(not(feature = "mmap"))]
    let source = ModelSource::bytes(std::fs::read(path)?);
    let model = ModelLoader::new(source).build_generative()?;
    let session = model.create_session(SessionConfig {
        seed: Some(42),
        ..SessionConfig::default()
    })?;
    drop(model);

    // Convert Session into transactional SessionChat.
    // If the model does not support warm chat (for example, sliding context or
    // incompatible chat template), into_chat returns the unmodified Session.
    let mut chat = match session.into_chat() {
        Ok(chat) => chat,
        Err((_, err)) => {
            return Err(format!("model does not support warm chat: {err:?}").into());
        }
    };

    println!("Initial chat phase: {:?}", chat.phase());
    println!("Initial position: {}", chat.position());

    let opts = GenerateOpts {
        max_tokens: 64,
        temperature: 0.7,
        ..GenerateOpts::default()
    };

    // --- Turn 1: Initialization with System and User messages ---
    println!("\n--- Turn 1 ---");
    let turn1_messages = vec![
        Message::system("You are a helpful and concise systems engineering assistant."),
        Message::user("What is a KV cache in LLM inference? Answer in one sentence."),
    ];

    let ingest1 = chat.ingest_messages(&turn1_messages)?;
    println!(
        "Ingested {} tokens (position: {} -> {})",
        ingest1.input_tokens, ingest1.position_before, ingest1.position_after
    );
    println!("Phase after ingest: {:?}", chat.phase());

    let turn1_result = chat.complete(&opts)?;
    println!("Assistant: {}", turn1_result.text.trim());
    println!(
        "Generated {} tokens (final position: {})",
        turn1_result.summary.tokens_generated,
        chat.position()
    );
    println!("Phase after completion: {:?}", chat.phase());
    if chat.phase() != SessionPhase::TurnComplete {
        println!(
            "Turn stopped before its terminal marker; reset or replace messages before a new user turn."
        );
        let reclaimed_session = chat.into_session();
        println!(
            "Reclaimed raw session at position {}",
            reclaimed_session.position()
        );
        return Ok(());
    }

    // --- Turn 2: Warm Continuation ---
    // The previous context remains in the KV cache. We only ingest the new user turn.
    println!("\n--- Turn 2 (Warm Continuation) ---");
    let turn2_message = Message::user("When should it be discarded?");

    let ingest2 = chat.ingest(&turn2_message)?;
    println!(
        "Ingested {} new tokens (position: {} -> {})",
        ingest2.input_tokens, ingest2.position_before, ingest2.position_after
    );

    let turn2_result = chat.complete(&opts)?;
    println!("Assistant: {}", turn2_result.text.trim());
    println!(
        "Generated {} tokens (final position: {})",
        turn2_result.summary.tokens_generated,
        chat.position()
    );
    println!("Phase after completion: {:?}", chat.phase());
    // --- Reclaim raw Session ---
    let reclaimed_session = chat.into_session();
    println!(
        "\nReclaimed raw session at position {}",
        reclaimed_session.position()
    );

    Ok(())
}

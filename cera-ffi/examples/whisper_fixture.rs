//! Write self-contained inputs for the Swift/Kotlin Whisper examples.
#[path = "../tests/common/whisper_fixture.rs"]
mod fixture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("output directory required")?,
    );
    std::fs::create_dir_all(&output)?;
    for (name, output_b, multilingual) in [("a", false, true), ("b", true, false)] {
        let bytes = fixture::model(output_b, multilingual);
        let (model, tokenizer) = cera::WhisperModel::from_bytes(bytes.clone())?;
        let text = model.transcribe(
            &tokenizer,
            &fixture::pcm(),
            &cera::WhisperTranscribeOpts {
                language: Some("en".into()),
                max_tokens: 3,
                ..Default::default()
            },
        )?;
        assert_eq!(text, if output_b { "bbb" } else { "aaa" });
        std::fs::write(output.join(format!("{name}.gguf")), bytes)?;
    }
    let bytes: Vec<_> = fixture::pcm()
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    std::fs::write(output.join("audio.f32"), bytes)?;
    println!("Whisper fixtures: a=aaa, b=bbb, 1600 mono samples at 16kHz");
    Ok(())
}

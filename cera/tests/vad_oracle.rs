#![cfg(all(feature = "mmap", not(target_arch = "wasm32")))]

use anyhow::Result;
use cera::vad::{SileroVad, VadConfig, VadSampleRate};

fn find_vad_model() -> Option<std::path::PathBuf> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("models/silero_vad.gguf"),
        manifest_dir.join("../models/silero_vad.gguf"),
        std::path::PathBuf::from("models/silero_vad.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn find_audio_sample() -> Option<std::path::PathBuf> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("models/en.wav"),
        manifest_dir.join("../models/en.wav"),
        std::path::PathBuf::from("models/en.wav"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn read_wav_pcm16_mono(path: &std::path::Path) -> (Vec<f32>, u32) {
    let buf = std::fs::read(path).expect("read fixture WAV");
    assert!(buf.len() >= 12, "WAV too short for RIFF/WAVE header");
    assert_eq!(&buf[0..4], b"RIFF", "missing RIFF header");
    assert_eq!(&buf[8..12], b"WAVE", "missing WAVE header");

    let read_u16 = |o: usize| -> u16 {
        let end = o.checked_add(2).expect("WAV offset overflow");
        u16::from_le_bytes(buf[o..end].try_into().unwrap())
    };
    let read_u32 = |o: usize| -> u32 {
        let end = o.checked_add(4).expect("WAV offset overflow");
        u32::from_le_bytes(buf[o..end].try_into().unwrap())
    };

    let mut offset = 12;
    let mut fmt_parsed = false;
    let mut sample_rate = 0;
    let mut pcm_bytes = None;

    while offset + 8 <= buf.len() {
        let chunk_id = &buf[offset..offset + 4];
        let chunk_size = read_u32(offset + 4) as usize;
        let data_start = offset + 8;
        let data_end = data_start + chunk_size;

        if chunk_id == b"fmt " {
            let audio_format = read_u16(data_start);
            let num_channels = read_u16(data_start + 2);
            sample_rate = read_u32(data_start + 4);
            let bits_per_sample = read_u16(data_start + 14);

            assert_eq!(audio_format, 1, "only PCM WAV supported");
            assert_eq!(num_channels, 1, "only mono WAV supported");
            assert_eq!(bits_per_sample, 16, "only 16-bit PCM supported");
            fmt_parsed = true;
        } else if chunk_id == b"data" {
            pcm_bytes = Some(&buf[data_start..data_end.min(buf.len())]);
        }

        offset = data_end + (chunk_size % 2);
    }

    assert!(fmt_parsed, "missing fmt chunk");
    let bytes = pcm_bytes.expect("missing data chunk");

    let mut samples = Vec::with_capacity(bytes.len() / 2);
    let (chunks, _) = bytes.as_chunks::<2>();
    for chunk in chunks {
        let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample_i16 as f32 / 32768.0);
    }

    (samples, sample_rate)
}

#[test]
fn test_silero_vad_silence_detection() -> Result<()> {
    let model_path = match find_vad_model() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/silero_vad.gguf not found");
            return Ok(());
        }
    };
    let mut vad = SileroVad::from_file(&model_path)?;
    vad.reset();

    // Constant silence chunks (16kHz)
    let silence_16k = [0.0f32; 512];
    for _ in 0..5 {
        let prob = vad.process_chunk(&silence_16k, VadSampleRate::Rate16kHz)?;
        assert!(
            prob < 0.05,
            "Silence chunk should have low speech probability, got {prob}"
        );
    }

    // Constant silence chunks (8kHz)
    vad.reset();
    let silence_8k = [0.0f32; 256];
    for _ in 0..5 {
        let prob = vad.process_chunk(&silence_8k, VadSampleRate::Rate8kHz)?;
        assert!(
            prob < 0.05,
            "8kHz silence chunk should have low speech probability, got {prob}"
        );
    }

    Ok(())
}

#[test]
fn test_silero_vad_16k_speech_streaming_and_timestamps() -> Result<()> {
    let model_path = match find_vad_model() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/silero_vad.gguf not found");
            return Ok(());
        }
    };
    let audio_path = match find_audio_sample() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/en.wav not found");
            return Ok(());
        }
    };

    let mut vad = SileroVad::from_file(&model_path)?;
    let (audio, sr) = read_wav_pcm16_mono(&audio_path);
    assert_eq!(sr, 16000);

    // Test first 5 chunks against known reference ONNX values:
    // [0.208342, 0.817943, 0.891196, 0.996361, 0.999178]
    let expected_probs = [0.208342, 0.817943, 0.891196, 0.996361, 0.999178];
    vad.reset();

    for (step, &expected) in expected_probs.iter().enumerate() {
        let chunk = &audio[step * 512..(step + 1) * 512];
        let prob = vad.process_chunk(chunk, VadSampleRate::Rate16kHz)?;
        let diff = (prob - expected).abs();
        assert!(
            diff < 1e-4,
            "Step {step}: expected {expected:.6}, got {prob:.6} (diff {diff:.2e})"
        );
    }

    // Test batch timestamp extraction
    let config = VadConfig::default();
    let timestamps = vad.get_speech_timestamps(&audio, VadSampleRate::Rate16kHz, &config)?;
    println!("Extracted {} speech segments from en.wav", timestamps.len());
    assert!(
        !timestamps.is_empty(),
        "Expected multiple speech segments from en.wav"
    );

    let first = &timestamps[0];
    println!(
        "First speech segment: start={:.1}ms, end={:.1}ms",
        first.start_ms, first.end_ms
    );
    assert!(first.end_ms > first.start_ms);
    assert!(first.end_ms > 1000.0);

    Ok(())
}

#[test]
fn test_silero_vad_8k_speech_streaming_and_timestamps() -> Result<()> {
    let model_path = match find_vad_model() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/silero_vad.gguf not found");
            return Ok(());
        }
    };
    let audio_path = match find_audio_sample() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/en.wav not found");
            return Ok(());
        }
    };

    let mut vad = SileroVad::from_file(&model_path)?;
    let (audio_16k, _) = read_wav_pcm16_mono(&audio_path);

    // Decimate by 2: 16kHz -> 8kHz
    let audio_8k: Vec<f32> = audio_16k.into_iter().step_by(2).collect();

    vad.reset();
    let mut probs = Vec::new();
    let (chunks, _) = audio_8k.as_chunks::<256>();
    for chunk in chunks.iter().take(20) {
        let prob = vad.process_chunk(chunk, VadSampleRate::Rate8kHz)?;
        probs.push(prob);
    }
    println!("8kHz first 20 probs in Rust: {:?}", probs);
    let speech_chunks = probs.iter().filter(|&&p| p > 0.5).count();
    println!("8kHz speech chunks in first 20: {}", speech_chunks);

    assert!(
        speech_chunks > 10,
        "8kHz stream should detect multiple speech chunks, got {speech_chunks}"
    );

    let config = VadConfig::default();
    let timestamps = vad.get_speech_timestamps(&audio_8k, VadSampleRate::Rate8kHz, &config)?;
    println!("8kHz Extracted {} speech segments", timestamps.len());
    assert!(
        !timestamps.is_empty(),
        "Expected speech segments in 8kHz downsampled audio"
    );

    Ok(())
}

#[test]
fn test_vad_iterator_streaming_events() -> Result<()> {
    let model_path = match find_vad_model() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/silero_vad.gguf not found");
            return Ok(());
        }
    };
    let audio_path = match find_audio_sample() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: models/en.wav not found");
            return Ok(());
        }
    };

    let mut vad = SileroVad::from_file(&model_path)?;
    let (audio_16k, _) = read_wav_pcm16_mono(&audio_path);

    let config = VadConfig::default();
    let mut iterator = cera::vad::VadIterator::new(VadSampleRate::Rate16kHz, config);

    let mut events = Vec::new();
    let (chunks, _) = audio_16k.as_chunks::<512>();
    for chunk in chunks {
        if let Some(event) = iterator.process_chunk(&mut vad, chunk)? {
            events.push(event);
        }
    }

    assert!(
        !events.is_empty(),
        "Streaming VadIterator should emit speech start/end events"
    );
    let starts = events
        .iter()
        .filter(|e| matches!(e, cera::vad::VadEvent::SpeechStart { .. }))
        .count();
    let ends = events
        .iter()
        .filter(|e| matches!(e, cera::vad::VadEvent::SpeechEnd { .. }))
        .count();

    println!(
        "Streaming VadIterator emitted {} total events: {} starts, {} ends",
        events.len(),
        starts,
        ends
    );
    assert!(starts > 0, "Expected at least one speech start event");
    assert!(ends > 0, "Expected at least one speech end event");

    Ok(())
}

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

#[test]
fn test_vad_iterator_streaming_20ms_frames() -> Result<()> {
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

    // 20 ms frames at 16 kHz = 320 samples per frame
    let config = VadConfig {
        frame_stride: Some(320),
        ..VadConfig::default()
    };
    let mut iterator = cera::vad::VadIterator::new(VadSampleRate::Rate16kHz, config);
    assert_eq!(iterator.frame_stride(), 320);
    assert_eq!(iterator.sample_rate(), VadSampleRate::Rate16kHz);

    let mut events = Vec::new();
    let (chunks, rem) = audio_16k.as_chunks::<320>();
    for chunk in chunks {
        if let Some(event) = iterator.process_chunk(&mut vad, chunk)? {
            events.push(event);
        }
    }
    if !rem.is_empty()
        && let Some(event) = iterator.process_chunk(&mut vad, rem)?
    {
        events.push(event);
    }
    if let Some(event) = iterator.flush() {
        events.push(event);
    }

    assert!(
        !events.is_empty(),
        "20ms streaming VadIterator should emit speech start/end events"
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
        "20ms streaming VadIterator emitted {} total events: {} starts, {} ends",
        events.len(),
        starts,
        ends
    );
    assert!(starts > 0, "Expected at least one speech start event");
    assert!(ends > 0, "Expected at least one speech end event");

    Ok(())
}

#[test]
fn test_silero_vad_stride_validation_and_20ms_timestamps() -> Result<()> {
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

    // Test stride boundary validation
    let dummy_chunk = [0.0f32; 512];
    assert!(
        vad.process_chunk_with_stride(&dummy_chunk, VadSampleRate::Rate16kHz, 0)
            .is_err(),
        "stride 0 should be rejected"
    );
    assert!(
        vad.process_chunk_with_stride(&dummy_chunk, VadSampleRate::Rate16kHz, 513)
            .is_err(),
        "stride > window_size should be rejected"
    );
    assert!(
        vad.process_chunk_with_stride(&dummy_chunk, VadSampleRate::Rate16kHz, 320)
            .is_ok(),
        "valid stride 320 should succeed"
    );

    let zero_config = VadConfig {
        frame_stride: Some(0),
        ..VadConfig::default()
    };
    assert!(
        vad.get_speech_timestamps(&audio_16k, VadSampleRate::Rate16kHz, &zero_config)
            .is_err(),
        "frame_stride 0 in VadConfig should be rejected by get_speech_timestamps"
    );
    let excessive_config = VadConfig {
        frame_stride: Some(1024),
        ..VadConfig::default()
    };
    assert!(
        vad.get_speech_timestamps(&audio_16k, VadSampleRate::Rate16kHz, &excessive_config)
            .is_err(),
        "frame_stride > window_size should be rejected by get_speech_timestamps"
    );

    // Compare batch timestamps with 20ms stride vs default 32ms stride
    let default_config = VadConfig::default();
    let default_timestamps =
        vad.get_speech_timestamps(&audio_16k, VadSampleRate::Rate16kHz, &default_config)?;

    let stride_20ms_config = VadConfig {
        frame_stride: Some(320),
        ..VadConfig::default()
    };
    let stride_timestamps =
        vad.get_speech_timestamps(&audio_16k, VadSampleRate::Rate16kHz, &stride_20ms_config)?;

    println!(
        "Timestamps count: default={}, 20ms stride={}",
        default_timestamps.len(),
        stride_timestamps.len()
    );

    assert!(
        !stride_timestamps.is_empty(),
        "20ms stride should detect speech segments"
    );

    let total_speech_default: f32 = default_timestamps
        .iter()
        .map(|t| t.end_ms - t.start_ms)
        .sum();
    let total_speech_20ms: f32 = stride_timestamps
        .iter()
        .map(|t| t.end_ms - t.start_ms)
        .sum();
    println!(
        "Total speech duration: default={:.1}ms, 20ms stride={:.1}ms",
        total_speech_default, total_speech_20ms
    );
    let dur_ratio = total_speech_20ms / total_speech_default;
    assert!(
        (0.85..=1.15).contains(&dur_ratio),
        "Total speech duration ratio {dur_ratio:.2} outside expected range [0.85, 1.15]"
    );

    for s in &stride_timestamps {
        assert!(s.end_ms > s.start_ms);
        assert!(
            s.end_ms - s.start_ms >= 60.0,
            "Segment duration {:.1}ms below min speech duration",
            s.end_ms - s.start_ms
        );
    }

    Ok(())
}

#[test]
fn test_vad_iterator_multi_window_and_pop_event() -> Result<()> {
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

    let config = VadConfig {
        frame_stride: Some(320),
        ..VadConfig::default()
    };
    let mut iterator = cera::vad::VadIterator::new(VadSampleRate::Rate16kHz, config);

    // Feed a large chunk (16,000 samples = 1 second) that contains speech
    let chunk_1s = &audio_16k[..audio_16k.len().min(16000)];
    let first_event = iterator.process_chunk(&mut vad, chunk_1s)?;

    let mut all_events = Vec::new();
    if let Some(ev) = first_event {
        all_events.push(ev);
    }
    while let Some(ev) = iterator.pop_event() {
        all_events.push(ev);
    }

    println!("Multi-window chunk yielded {} events", all_events.len());
    assert!(
        !all_events.is_empty(),
        "Ingesting 1 second chunk should produce speech events"
    );

    Ok(())
}

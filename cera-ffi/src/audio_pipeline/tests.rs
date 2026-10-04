//! Unit tests for FfiAudioPipeline.

use super::*;

#[test]
fn test_ffi_audio_pipeline_defaults() {
    let pipeline = FfiAudioPipeline::from_bytes(None, None, None, None)
        .expect("pipeline creation succeeds with defaults");

    assert_eq!(
        pipeline.state().expect("state query"),
        FfiAudioPipelineState::ListeningForSpeech
    );
    assert!(!pipeline.is_speech_active().expect("is_speech_active"));
    assert!(
        !pipeline
            .is_listening_for_hotword()
            .expect("is_listening_for_hotword")
    );
    assert_eq!(pipeline.current_sample().expect("current_sample"), 0);
    assert!(
        pipeline
            .last_utterance()
            .expect("last_utterance")
            .is_empty()
    );
}

#[test]
fn test_ffi_audio_pipeline_process_and_flush() {
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };

    let pipeline = FfiAudioPipeline::from_bytes(None, None, None, Some(config))
        .expect("pipeline creation succeeds");

    let chunk = vec![0.1f32; 1600];
    let events = pipeline
        .process_chunk(chunk)
        .expect("process_chunk succeeds");
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0],
        FfiAudioPipelineEvent::SpeechStart { .. }
    ));
    assert!(pipeline.is_speech_active().expect("is_speech_active"));

    let flush_events = pipeline.flush().expect("flush succeeds");
    assert_eq!(flush_events.len(), 1);
    assert!(matches!(
        flush_events[0],
        FfiAudioPipelineEvent::SpeechEnd { .. }
    ));

    let utterance = pipeline.take_last_utterance().expect("take_last_utterance");
    assert_eq!(utterance.len(), 1600);
}

#[test]
fn test_ffi_audio_pipeline_reset_and_cancel() {
    let pipeline =
        FfiAudioPipeline::from_bytes(None, None, None, None).expect("pipeline creation succeeds");

    let chunk = vec![0.05f32; 800];
    pipeline.process_chunk(chunk).expect("process_chunk");
    assert!(pipeline.is_speech_active().expect("is_speech_active"));

    pipeline.cancel().expect("cancel succeeds");
    pipeline.clear_cancel().expect("clear_cancel succeeds");

    pipeline.reset().expect("reset succeeds");
    assert_eq!(
        pipeline.state().expect("state query"),
        FfiAudioPipelineState::ListeningForSpeech
    );
    assert_eq!(pipeline.current_sample().expect("current_sample"), 0);
}

#[test]
fn test_ffi_audio_pipeline_wait_free_cancel() {
    let pipeline =
        FfiAudioPipeline::from_bytes(None, None, None, None).expect("pipeline creation succeeds");

    // Hold inner lock simulating active processing on another thread
    let _guard = pipeline.inner.lock().expect("lock held");

    // cancel() and clear_cancel() must succeed without deadlocking on inner lock
    pipeline.cancel().expect("cancel succeeds wait-free");
    assert!(pipeline.cancel.load(std::sync::atomic::Ordering::Relaxed));

    pipeline
        .clear_cancel()
        .expect("clear_cancel succeeds wait-free");
    assert!(!pipeline.cancel.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn test_ffi_audio_pipeline_has_no_diarizer_by_default() {
    let pipeline = FfiAudioPipeline::from_bytes(None, None, None, None).expect("pipeline");
    assert!(!pipeline.has_diarizer().expect("has_diarizer"));
    assert!(!pipeline.diarizer_on_npu());
    assert!(
        !pipeline
            .add_utterance("hello".into(), 0.0, 100.0)
            .expect("add_utterance")
    );
}

/// The NPU flag follows the live diarizer, not just staging: a bare pipeline with the flag
/// forced on still reports false. Needs no models.
#[test]
fn test_diarizer_on_npu_is_false_without_a_running_diarizer() {
    let pipeline = cera::audio_pipeline::AudioPipeline::builder()
        .build()
        .expect("pipeline");
    let cancel = pipeline.cancel_handle();
    let ffi = FfiAudioPipeline {
        inner: Mutex::new(pipeline),
        cancel,
        diarizer_on_npu: true,
    };
    assert!(!ffi.diarizer_on_npu());
}

/// With a diarizer attached the flag reads true while it runs and false once it stops (here: a
/// sample the front end refuses). Skipped when the model is absent.
#[test]
fn test_diarizer_on_npu_follows_a_failing_diarizer() {
    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf");
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display()
        );
        eprintln!("{} not found, skipping", path.display());
        return;
    }
    let model = cera::model::sortformer::SortformerModel::from_file(&path).unwrap();
    let params = model.default_streaming().clone();
    let pipeline = cera::audio_pipeline::AudioPipeline::builder()
        .with_diarizer(model, params)
        .build()
        .unwrap();
    let cancel = pipeline.cancel_handle();
    let ffi = FfiAudioPipeline {
        inner: Mutex::new(pipeline),
        cancel,
        diarizer_on_npu: true,
    };
    assert!(ffi.diarizer_on_npu());
    let mut bad = vec![0.0f32; 1600];
    bad[10] = 2e9; // finite, so the pipeline passes it on; the mel front end refuses it
    ffi.process_chunk(bad).expect("the pipeline itself is fine");
    assert!(!ffi.has_diarizer().unwrap());
    assert!(!ffi.diarizer_on_npu());
}

#[test]
fn test_ffi_utterance_labeled_event_conversion() {
    let core = cera::audio_pipeline::AudioPipelineEvent::UtteranceLabeled {
        text: "hello".into(),
        start_ms: 480.0,
        end_ms: 4160.0,
        speaker: Some(2),
        confidence: Some(0.75),
        overlapping: Some(1),
    };
    assert_eq!(
        FfiAudioPipelineEvent::from(core),
        FfiAudioPipelineEvent::UtteranceLabeled {
            text: "hello".into(),
            start_ms: 480.0,
            end_ms: 4160.0,
            speaker: Some(2),
            confidence: Some(0.75),
            overlapping: Some(1),
        }
    );
}

/// With a Sortformer model (skipped when it is absent) the pipeline labels a registered
/// utterance; the NPU is only used when asked for and available, so on a host it is `false`.
#[test]
fn test_ffi_audio_pipeline_with_a_diarizer() {
    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf");
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display()
        );
        eprintln!("{} not found, skipping", path.display());
        return;
    }
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    let pipeline = FfiAudioPipeline::from_files_with_diarizer(
        None,
        None,
        None,
        path.to_string_lossy().into_owned(),
        false,
        Some(config),
    )
    .expect("pipeline with a diarizer");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(!pipeline.diarizer_on_npu());
    assert!(pipeline.add_utterance("a".into(), 0.0, 300.0).unwrap());
    // A missing model is an error that names the file, not a pipeline without a diarizer.
    let Err(err) = FfiAudioPipeline::from_files_with_diarizer(
        None,
        None,
        None,
        "/nonexistent/sortformer.gguf".into(),
        false,
        None,
    ) else {
        panic!("a missing diarizer model was accepted");
    };
    assert!(
        format!("{err:?}").contains("/nonexistent/sortformer.gguf"),
        "{err:?}"
    );
}

/// The bytes constructor attaches the same diarizer as the file constructor (skipped when
/// the model is absent); on a host the NPU flag stays `false`.
#[test]
fn test_ffi_audio_pipeline_with_a_diarizer_from_bytes() {
    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf");
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display()
        );
        eprintln!("{} not found, skipping", path.display());
        return;
    }
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    let bytes = std::fs::read(&path).unwrap();
    let pipeline =
        FfiAudioPipeline::from_bytes_with_diarizer(None, None, None, bytes, false, Some(config))
            .expect("pipeline with a diarizer from bytes");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(!pipeline.diarizer_on_npu());
    assert!(pipeline.add_utterance("a".into(), 0.0, 300.0).unwrap());
    // Garbage bytes are an error, not a pipeline without a diarizer.
    let Err(err) =
        FfiAudioPipeline::from_bytes_with_diarizer(None, None, None, vec![0u8; 64], false, None)
    else {
        panic!("garbage diarizer bytes were accepted");
    };
    assert!(format!("{err:?}").contains("from bytes"), "{err:?}");
}

/// Mono 16 kHz s16 WAV bytes to f32 samples. Mirrors `cera/tests/common/mod.rs::read_wav_f32`
/// (this crate cannot import the `cera` test helpers): chunks are walked with their declared
/// lengths, `fmt` is validated before `data` is trusted, and the `data` slice is bounded by
/// its declared length, so a `data` byte run inside an earlier chunk or a corrupt length
/// cannot mislead the parse.
fn wav_bytes_to_f32(clip: &[u8]) -> Vec<f32> {
    assert!(
        clip.len() >= 12 && &clip[0..4] == b"RIFF" && &clip[8..12] == b"WAVE",
        "{} bytes with no RIFF/WAVE header",
        clip.len()
    );
    let mut pos = 12;
    let mut fmt_ok = false;
    while pos + 8 <= clip.len() {
        let len = u32::from_le_bytes(clip[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = pos + 8 + len;
        assert!(
            end <= clip.len(),
            "chunk at {pos} runs past the file ({} bytes)",
            clip.len()
        );
        let body = &clip[pos + 8..end];
        if &clip[pos..pos + 4] == b"fmt " {
            assert!(
                body.len() >= 16,
                "fmt chunk holds {} bytes, need 16",
                body.len()
            );
            let (tag, ch, rate, bits) = (
                u16::from_le_bytes(body[0..2].try_into().unwrap()),
                u16::from_le_bytes(body[2..4].try_into().unwrap()),
                u32::from_le_bytes(body[4..8].try_into().unwrap()),
                u16::from_le_bytes(body[14..16].try_into().unwrap()),
            );
            assert_eq!((tag, ch, rate, bits), (1, 1, 16_000, 16), "WAV format");
            fmt_ok = true;
        } else if &clip[pos..pos + 4] == b"data" {
            assert!(fmt_ok, "data before fmt");
            return body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect();
        }
        pos = end + (len & 1);
    }
    panic!("no data chunk");
}

fn wav_bytes(extra_chunks: &[(&[u8; 4], &[u8])], data: &[u8]) -> Vec<u8> {
    let mut wav = b"RIFF....WAVE".to_vec();
    let mut body = vec![1, 0, 1, 0, 0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0, 2, 0, 16, 0];
    let mut chunks = extra_chunks.to_vec();
    chunks.push((b"data", data));
    for (id, chunk) in chunks {
        // `fmt` first: the parser trusts `data` only after validating the format.
        if id == b"data" {
            wav.extend_from_slice(b"fmt ");
            wav.extend_from_slice(&(body.len() as u32).to_le_bytes());
            wav.append(&mut body);
        }
        wav.extend_from_slice(id);
        wav.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        wav.extend_from_slice(chunk);
        if chunk.len() % 2 == 1 {
            wav.push(0);
        }
    }
    let len = (wav.len() - 8) as u32;
    wav[4..8].copy_from_slice(&len.to_le_bytes());
    wav
}

/// A `data` byte run inside an earlier chunk does not fool the parser: chunks are walked by
/// their declared lengths, not by scanning for the tag.
#[test]
fn wav_bytes_to_f32_ignores_a_data_run_inside_an_earlier_chunk() {
    let pcm = wav_bytes_to_f32(&wav_bytes(&[(b"JUNK", b"xxdatayy")], &[0, 0, 255, 127]));
    assert_eq!(pcm, vec![0.0, 32767.0 / 32768.0]);
}

/// A corrupt chunk length is refused, and only the declared `data` bytes are decoded.
#[test]
fn wav_bytes_to_f32_bounds_every_slice() {
    let good = wav_bytes(&[], &[1, 0, 2, 0]);
    assert_eq!(wav_bytes_to_f32(&good), vec![1.0 / 32768.0, 2.0 / 32768.0]);
    // Overlong `data` length: the chunk runs past the file.
    let mut bad = good.clone();
    let at = bad.len() - 4 - 8;
    bad[at + 4..at + 8].copy_from_slice(&1_000_000u32.to_le_bytes());
    let err = std::panic::catch_unwind(|| wav_bytes_to_f32(&bad)).unwrap_err();
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(msg.contains("runs past the file"), "{msg}");
}

/// Device test, run with `--ignored` on a Qualcomm phone: `from_files_with_diarizer` with
/// `prefer_npu` puts the diarizer on the NPU, and a registered utterance gets the speaker the
/// CPU gives it. Needs `SORTFORMER_GGUF` (a model converted with `--tail-outtype q8_0`) and
/// `SORTFORMER_CLIP` (the committed `clip.wav`) in the environment.
#[cfg(feature = "hexagon")]
#[test]
#[ignore = "needs a Hexagon NPU, a Q8_0-tail Sortformer GGUF (SORTFORMER_GGUF) and clip.wav (SORTFORMER_CLIP)"]
fn test_ffi_audio_pipeline_diarizer_on_the_npu() {
    let gguf = std::env::var("SORTFORMER_GGUF").expect("SORTFORMER_GGUF");
    let clip = std::fs::read(std::env::var("SORTFORMER_CLIP").expect("SORTFORMER_CLIP"))
        .expect("read the clip");
    let pcm: Vec<f32> = wav_bytes_to_f32(&clip);
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    let pipeline =
        FfiAudioPipeline::from_files_with_diarizer(None, None, None, gguf, true, Some(config))
            .expect("pipeline with an NPU diarizer");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(
        pipeline.diarizer_on_npu(),
        "the diarizer did not reach the NPU"
    );
    for (text, start, end) in [
        ("first", 480.0, 4160.0),
        ("second", 4960.0, 8240.0),
        ("third", 8480.0, 10640.0),
    ] {
        assert!(pipeline.add_utterance(text.into(), start, end).unwrap());
    }
    let mut events = Vec::new();
    for piece in pcm.chunks(1600) {
        events.extend(pipeline.process_chunk(piece.to_vec()).unwrap());
    }
    events.extend(pipeline.flush().unwrap());
    let labeled: Vec<(String, Option<u32>)> = events
        .into_iter()
        .filter_map(|e| match e {
            FfiAudioPipelineEvent::UtteranceLabeled { text, speaker, .. } => Some((text, speaker)),
            _ => None,
        })
        .collect();
    assert_eq!(
        labeled,
        vec![
            ("first".to_string(), Some(0)),
            ("second".to_string(), Some(1)),
            ("third".to_string(), Some(2)),
        ]
    );
}

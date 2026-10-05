//! Unit tests for FfiAudioPipeline.

use super::*;

/// Collects the `message` field of every `WARN` and `ERROR` event. Textual port of
/// `cera/tests/common/mod.rs::WarnCapture` (this crate cannot import those helpers): the
/// shared name promises identical semantics, so the predicate is copied, not rewritten.
/// `tracing::Level`'s ordering is by verbosity and reads backwards: `ERROR` is the
/// *smallest* level, so `> WARN` keeps exactly WARN and ERROR.
#[derive(Clone, Default)]
struct WarnCapture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WarnCapture {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() > tracing::Level::WARN {
            return;
        }
        struct Visit(Option<String>);
        impl tracing::field::Visit for Visit {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = Some(format!("{value:?}"));
                }
            }
        }
        let mut v = Visit(None);
        event.record(&mut v);
        // Events with no `message` field still matter: an ERROR-level `sortformer: ` fault
        // with no message must still trip the zero-fallback pin below.
        let msg =
            v.0.unwrap_or_else(|| format!("<no message> target={}", event.metadata().target()));
        match self.0.lock() {
            Ok(mut g) => g.push(msg),
            Err(p) => p.into_inner().push(msg),
        }
    }
}

impl WarnCapture {
    fn messages(&self) -> Vec<String> {
        match self.0.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

/// The capture port keeps WARN and ERROR events and drops everything quieter, like the
/// canonical capture: an ERROR-level fault must trip the zero-fallback pin, not slip past.
#[test]
fn warn_capture_keeps_warn_and_error_only() {
    use tracing_subscriber::layer::SubscriberExt;
    let warns = WarnCapture::default();
    let sub = tracing_subscriber::registry().with(warns.clone());
    tracing::subscriber::with_default(sub, || {
        tracing::error!("kept-error");
        tracing::warn!("kept-warn");
        tracing::info!("dropped-info");
    });
    let got = warns.messages();
    assert_eq!(got.len(), 2, "expected exactly ERROR and WARN, got {got:?}");
    assert!(
        got.iter().any(|m| m.contains("kept-error")),
        "ERROR was dropped, so the level comparison is wrong: {got:?}"
    );
    assert!(
        got.iter().any(|m| m.contains("kept-warn")),
        "WARN was dropped, so the level comparison is wrong: {got:?}"
    );
}

/// Model path under `~/.leap/models`, or `None` (skip the test) when it is absent. One
/// home for this crate's unit tests; the cera unit tests, `model/sortformer_hexagon.rs`,
/// `model/nemotron3_diarization_hexagon.rs`, and the integration suites resolve the same
/// files on their own, so keep them in sync on a rename.
fn model_or_skip(rel: &str) -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rel);
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display()
        );
        eprintln!("{} not found, skipping", path.display());
        return None;
    }
    Some(path)
}

fn nemotron3_model_or_skip() -> Option<std::path::PathBuf> {
    model_or_skip(".leap/models/nemotron3-diarization/nemotron3-diarization-q8_0-npu.gguf")
}

fn sortformer_model_or_skip() -> Option<std::path::PathBuf> {
    model_or_skip(".leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf")
}

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
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
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
fn test_vad_on_cpu_flag_reaches_the_core_config() {
    let ffi = FfiAudioPipelineConfig {
        vad_on_cpu: true,
        ..audio_pipeline_default_config()
    };
    let core = cera::audio_pipeline::AudioPipelineConfig::from(ffi);
    assert!(core.vad_on_cpu);
}

/// A bare pipeline reports the VAD off the NPU. Needs no models.
#[test]
fn test_vad_on_npu_is_false_without_a_vad() {
    let pipeline = cera::audio_pipeline::AudioPipeline::builder()
        .build()
        .expect("pipeline");
    let cancel = pipeline.cancel_handle();
    let ffi = FfiAudioPipeline {
        inner: Mutex::new(pipeline),
        cancel,
        diarizer_on_npu: false,
    };
    assert!(!ffi.vad_on_npu());
}

#[test]
fn test_ffi_utterance_labeled_event_conversion() {
    // Both axes crossed: the stalled-diarizer event is `None` + `dropped`, so a leg with
    // a labeled speaker alone would pass a mapping that gates the flag on speaker presence.
    for speaker in [Some(2), None] {
        for dropped in [true, false] {
            let core = cera::audio_pipeline::AudioPipelineEvent::UtteranceLabeled {
                text: "hello".into(),
                start_ms: 480.0,
                end_ms: 4160.0,
                speaker,
                confidence: Some(0.75),
                overlapping: Some(1),
                dropped,
            };
            assert_eq!(
                FfiAudioPipelineEvent::from(core),
                FfiAudioPipelineEvent::UtteranceLabeled {
                    text: "hello".into(),
                    start_ms: 480.0,
                    end_ms: 4160.0,
                    speaker,
                    confidence: Some(0.75),
                    overlapping: Some(1),
                    dropped,
                }
            );
        }
    }
}

/// With a Sortformer model (skipped when it is absent) the pipeline labels a registered
/// utterance; the NPU is only used when asked for and available, so on a host it is `false`.
#[test]
fn test_ffi_audio_pipeline_with_a_diarizer() {
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
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
        Some(config.clone()),
    )
    .expect("pipeline with a diarizer");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(!pipeline.diarizer_on_npu());
    assert!(pipeline.add_utterance("a".into(), 0.0, 300.0).unwrap());
    // `prefer_npu` degrades gracefully when staging fails: a CPU diarizer, not an error and
    // not a stuck on-NPU flag. (On Android staging may legitimately succeed, so the flag
    // assert only holds off-device; the graceful-build asserts hold everywhere.)
    let cpu = FfiAudioPipeline::from_files_with_diarizer(
        None,
        None,
        None,
        path.to_string_lossy().into_owned(),
        true,
        Some(config),
    )
    .expect("prefer_npu degrades gracefully when staging fails");
    assert!(cpu.has_diarizer().unwrap());
    if !cfg!(target_os = "android") {
        assert!(
            !cpu.diarizer_on_npu(),
            "staging failed on host yet the flag claims the NPU"
        );
    }
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
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    let bytes = std::fs::read(&path).unwrap();
    let pipeline = FfiAudioPipeline::from_bytes_with_diarizer(
        None,
        None,
        None,
        bytes,
        false,
        Some(config.clone()),
    )
    .expect("pipeline with a diarizer from bytes");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(!pipeline.diarizer_on_npu());
    assert!(pipeline.add_utterance("a".into(), 0.0, 300.0).unwrap());
    // `prefer_npu` degrades gracefully when staging fails: a CPU diarizer, not an error and
    // not a stuck on-NPU flag (the flag assert only holds off-Android; see the file test).
    let cpu = FfiAudioPipeline::from_bytes_with_diarizer(
        None,
        None,
        None,
        std::fs::read(&path).unwrap(),
        true,
        Some(config),
    )
    .expect("prefer_npu degrades gracefully when staging fails");
    assert!(cpu.has_diarizer().unwrap());
    if !cfg!(target_os = "android") {
        assert!(
            !cpu.diarizer_on_npu(),
            "staging failed on host yet the flag claims the NPU"
        );
    }
    // Garbage bytes are an error, not a pipeline without a diarizer.
    let Err(err) =
        FfiAudioPipeline::from_bytes_with_diarizer(None, None, None, vec![0u8; 64], false, None)
    else {
        panic!("garbage diarizer bytes were accepted");
    };
    assert!(format!("{err:?}").contains("from bytes"), "{err:?}");
}

/// The Nemotron-3 file constructor attaches a running diarizer (skipped when the model
/// is absent).
#[test]
fn test_ffi_audio_pipeline_with_a_nemotron3_diarizer() {
    let Some(path) = nemotron3_model_or_skip() else {
        return;
    };
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    let pipeline = FfiAudioPipeline::from_files_with_diarizer_nemotron3(
        None,
        None,
        None,
        path.to_string_lossy().into_owned(),
        false,
        Some(config.clone()),
    )
    .expect("pipeline with a Nemotron-3 diarizer");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(!pipeline.diarizer_on_npu());
    assert!(pipeline.add_utterance("a".into(), 0.0, 300.0).unwrap());
    // `prefer_npu` degrades gracefully when staging fails: a CPU diarizer, not an error and
    // not a stuck on-NPU flag. (On Android staging may legitimately succeed, so the flag
    // assert only holds off-device; the graceful-build asserts hold everywhere.)
    let cpu = FfiAudioPipeline::from_files_with_diarizer_nemotron3(
        None,
        None,
        None,
        path.to_string_lossy().into_owned(),
        true,
        Some(config),
    )
    .expect("prefer_npu degrades gracefully when staging fails");
    assert!(cpu.has_diarizer().unwrap());
    if !cfg!(target_os = "android") {
        assert!(
            !cpu.diarizer_on_npu(),
            "staging failed on host yet the flag claims the NPU"
        );
    }
    // A missing model is an error that names the file, not a pipeline without a diarizer.
    let Err(err) = FfiAudioPipeline::from_files_with_diarizer_nemotron3(
        None,
        None,
        None,
        "/nonexistent/nemotron3.gguf".into(),
        false,
        None,
    ) else {
        panic!("a missing diarizer model was accepted");
    };
    assert!(
        format!("{err:?}").contains("/nonexistent/nemotron3.gguf"),
        "{err:?}"
    );
}

/// The Nemotron-3 bytes constructor attaches the same diarizer as the file constructor
/// (skipped when the model is absent); the NPU flag stays `false`.
#[test]
fn test_ffi_audio_pipeline_with_a_nemotron3_diarizer_from_bytes() {
    let Some(path) = nemotron3_model_or_skip() else {
        return;
    };
    let config = FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    let bytes = std::fs::read(&path).unwrap();
    let pipeline = FfiAudioPipeline::from_bytes_with_diarizer_nemotron3(
        None,
        None,
        None,
        bytes,
        false,
        Some(config.clone()),
    )
    .expect("pipeline with a Nemotron-3 diarizer from bytes");
    assert!(pipeline.has_diarizer().unwrap());
    assert!(!pipeline.diarizer_on_npu());
    assert!(pipeline.add_utterance("a".into(), 0.0, 300.0).unwrap());
    // `prefer_npu` degrades gracefully when staging fails: a CPU diarizer, not an error and
    // not a stuck on-NPU flag (the flag assert only holds off-Android; see the file test).
    let cpu = FfiAudioPipeline::from_bytes_with_diarizer_nemotron3(
        None,
        None,
        None,
        std::fs::read(&path).unwrap(),
        true,
        Some(config),
    )
    .expect("prefer_npu degrades gracefully when staging fails");
    assert!(cpu.has_diarizer().unwrap());
    if !cfg!(target_os = "android") {
        assert!(
            !cpu.diarizer_on_npu(),
            "staging failed on host yet the flag claims the NPU"
        );
    }
    // Garbage bytes are an error, not a pipeline without a diarizer.
    let Err(err) = FfiAudioPipeline::from_bytes_with_diarizer_nemotron3(
        None,
        None,
        None,
        vec![0u8; 64],
        false,
        None,
    ) else {
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
    // Zero-fallback pin: per-call CPU fallback reproduces these labels bit-identically
    // while leaving the flag set, so without this an NPU that fails every step passes as
    // "NPU verified". Scoped to this thread; the `sortformer: ` prefix is the model's own
    // fault tag. This catches faults, not silent declines: a declined step returns Ok(None)
    // with no warning by design, so a step the NPU declines still passes here. Width
    // declines are pinned instead by the parity suite's staging-bound asserts (`max_t` for
    // predict steps, `max_n` for stem batches against `window_frames()`); other decline
    // reasons remain a known residual until the staged tail exposes per-stage call counts.
    use tracing_subscriber::layer::SubscriberExt;
    let warns = WarnCapture::default();
    let sub = tracing_subscriber::registry().with(warns.clone());
    let mut events = Vec::new();
    tracing::subscriber::with_default(sub, || {
        for piece in pcm.chunks(1600) {
            events.extend(pipeline.process_chunk(piece.to_vec()).unwrap());
        }
        events.extend(pipeline.flush().unwrap());
    });
    assert!(
        warns.messages().iter().all(|m| !m.contains("sortformer: ")),
        "NPU steps fell back to the CPU: {:?}",
        warns.messages()
    );
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

#[test]
fn pcm16_converts_little_endian_samples_to_unit_floats() {
    let bytes = [
        0x00, 0x80, // i16::MIN
        0x00, 0x00, // 0
        0xFF, 0x7F, // i16::MAX
        0x00, 0x40, // 16384
    ];
    let floats = pcm16_le_to_f32(&bytes).expect("even length");
    assert_eq!(floats, vec![-1.0, 0.0, 32767.0 / 32768.0, 0.5]);
}

#[test]
fn pcm16_with_an_odd_length_is_refused_not_truncated() {
    let err = pcm16_le_to_f32(&[0u8; 3]).expect_err("a torn read");
    assert!(
        matches!(&err, FfiError::Backend { detail } if detail.contains("odd length")),
        "got {err:?}"
    );
}

/// The PCM16 entry point must drive the pipeline exactly as `process_chunk` does with the same
/// samples: it is only a cheaper way across the FFI.
#[test]
fn process_chunk_pcm16_matches_process_chunk() {
    let config = || FfiAudioPipelineConfig {
        auto_transcribe: false,
        ..audio_pipeline_default_config()
    };
    // 0x0CCC / 32768 is the f32 the byte path produces for 3276.
    let samples: Vec<i16> = vec![3276; 1600];
    let floats: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let by_float = FfiAudioPipeline::from_bytes(None, None, None, Some(config())).unwrap();
    let by_bytes = FfiAudioPipeline::from_bytes(None, None, None, Some(config())).unwrap();
    let a = by_float.process_chunk(floats).unwrap();
    let b = by_bytes.process_chunk_pcm16(bytes).unwrap();
    assert_eq!(a, b);
    assert!(
        !a.is_empty(),
        "the chunk should start speech, or the comparison is vacuous"
    );
    assert_eq!(
        by_float.flush().unwrap(),
        by_bytes.flush().unwrap(),
        "flush must agree too"
    );
    assert_eq!(
        by_float.take_last_utterance().unwrap(),
        by_bytes.take_last_utterance().unwrap()
    );
}

#[test]
fn process_chunk_pcm16_rejects_a_torn_chunk_without_advancing_the_stream() {
    let pipeline = FfiAudioPipeline::from_bytes(None, None, None, None).unwrap();
    assert!(pipeline.process_chunk_pcm16(vec![0u8; 3201]).is_err());
    assert_eq!(pipeline.current_sample().unwrap(), 0);
}

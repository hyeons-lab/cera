//! Unit tests for the unified AudioPipeline facade.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

/// Sortformer model path, or `None` (skip the test) when it is absent. One home per
/// crate's unit tests; `model/sortformer_hexagon.rs` and the integration suites resolve
/// the same file on their own, so keep them in sync on a rename.
///
/// `mmap`-gated like every caller: in no-`mmap` builds the helper would be dead code and
/// fail the `-D warnings` clippy legs.
#[cfg(feature = "mmap")]
fn sortformer_model_or_skip() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".leap/models/sortformer/sortformer-4spk-v2.1-q8_0.gguf");
    // Same `== Ok("1")` require idiom as the FFI twin and the other inline skip sites
    // (`require_model_or_skip` fails loud on any non-empty value instead).
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

/// Session-local conversion keeps its clamp for finite times (pre-session starts at the
/// first frame) but passes non-finite times through: clamping NaN or -inf to 0.0 would
/// smuggle an unplaceable utterance past the labeler's non-finite guard.
#[test]
fn session_local_ms_passes_non_finite_times_through() {
    assert_eq!(session_local_ms(1500.0, 1000.0), 500.0);
    assert_eq!(session_local_ms(500.0, 1000.0), 0.0);
    assert!(session_local_ms(f32::NAN, 1000.0).is_nan());
    assert_eq!(
        session_local_ms(f32::NEG_INFINITY, 1000.0),
        f64::NEG_INFINITY
    );
    assert_eq!(session_local_ms(f32::INFINITY, 1000.0), f64::INFINITY);
}

#[test]
fn test_audio_pipeline_builder_defaults() {
    let pipeline = AudioPipelineBuilder::new()
        .build()
        .expect("builder must succeed with defaults");

    assert_eq!(pipeline.state(), AudioPipelineState::ListeningForSpeech);
    assert!(!pipeline.is_speech_active());
    assert!(!pipeline.is_listening_for_hotword());
    assert_eq!(pipeline.current_sample(), 0);
    assert!(!pipeline.has_hotword());
    assert!(!pipeline.has_vad());
    assert!(!pipeline.has_whisper());
    assert!(pipeline.last_utterance().is_empty());
}

#[test]
fn test_audio_pipeline_require_hotword_without_detector() {
    let pipeline = AudioPipelineBuilder::new()
        .with_require_hotword(true)
        .build()
        .expect("builder succeeds");

    // Without a hotword detector attached, pipeline falls back to listening for speech
    assert_eq!(pipeline.state(), AudioPipelineState::ListeningForSpeech);
}

#[test]
fn test_audio_pipeline_continuous_speech_accumulation() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    // Feed silence chunk (1600 samples = 100 ms at 16 kHz)
    let chunk = vec![0.0f32; 1600];
    let events = pipeline.process_chunk(&chunk).expect("process chunk");

    // Without VAD, first chunk triggers immediate speech start
    assert_eq!(events.len(), 1);
    match &events[0] {
        AudioPipelineEvent::SpeechStart { sample, ms } => {
            assert_eq!(*sample, 0);
            assert_eq!(*ms, 0.0);
        }
        other => panic!("expected SpeechStart, got {other:?}"),
    }

    assert!(pipeline.is_speech_active());
    assert_eq!(pipeline.state(), AudioPipelineState::SpeechActive);
    assert_eq!(pipeline.current_sample(), 1600);

    // Feed second chunk
    let events2 = pipeline.process_chunk(&chunk).expect("process chunk 2");
    assert!(events2.is_empty());
    assert_eq!(pipeline.current_sample(), 3200);

    // Flush to conclude utterance
    let flush_events = pipeline.flush().expect("flush succeeds");
    assert_eq!(flush_events.len(), 1);
    match &flush_events[0] {
        AudioPipelineEvent::SpeechEnd {
            start_sample,
            end_sample,
            ..
        } => {
            assert_eq!(*start_sample, 0);
            assert_eq!(*end_sample, 3200);
        }
        other => panic!("expected SpeechEnd, got {other:?}"),
    }

    assert_eq!(pipeline.last_utterance().len(), 3200);
    assert_eq!(pipeline.state(), AudioPipelineState::ListeningForSpeech);
}

#[test]
fn test_audio_pipeline_reset_and_cancel_lifecycle() {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut pipeline = AudioPipelineBuilder::new()
        .with_cancel(cancel.clone())
        .build()
        .expect("builder succeeds");

    let chunk = vec![0.1f32; 800];
    pipeline.process_chunk(&chunk).expect("process chunk");
    assert!(pipeline.is_speech_active());
    assert_eq!(pipeline.current_sample(), 800);

    pipeline.cancel();
    assert!(cancel.load(Ordering::Relaxed));

    pipeline.clear_cancel();
    assert!(!cancel.load(Ordering::Relaxed));

    pipeline.reset();
    assert_eq!(pipeline.state(), AudioPipelineState::ListeningForSpeech);
    assert_eq!(pipeline.current_sample(), 0);
    assert!(pipeline.last_utterance().is_empty());
    assert!(!cancel.load(Ordering::Relaxed));
}

#[test]
fn test_audio_pipeline_max_utterance_boundary() {
    let config = AudioPipelineConfig {
        max_utterance_ms: 1000,
        auto_transcribe: false,
        ..Default::default()
    };

    let mut pipeline = AudioPipelineBuilder::new()
        .with_config(config)
        .build()
        .expect("builder succeeds");

    // Feed 16,000 samples in one chunk
    let big_chunk = vec![0.05f32; 16_000];
    let events = pipeline.process_chunk(&big_chunk).expect("process chunk");

    // Should contain SpeechStart and then forced SpeechEnd due to max utterance cap
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], AudioPipelineEvent::SpeechStart { .. }));
    assert!(matches!(events[1], AudioPipelineEvent::SpeechEnd { .. }));

    assert_eq!(pipeline.last_utterance().len(), 16_000);
    assert_eq!(pipeline.state(), AudioPipelineState::ListeningForSpeech);
}

#[test]
fn test_audio_pipeline_events_preserved_in_pending_queue() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    let chunk = vec![0.1f32; 1600];
    let events = pipeline.process_chunk(&chunk).expect("process chunk");
    assert_eq!(events.len(), 1);

    // pop_event must return the identical SpeechStart event
    let popped = pipeline.pop_event().expect("event in queue");
    assert!(matches!(popped, AudioPipelineEvent::SpeechStart { .. }));
    assert!(pipeline.pop_event().is_none());

    // Flush generates SpeechEnd
    let flush_events = pipeline.flush().expect("flush succeeds");
    assert_eq!(flush_events.len(), 1);

    let popped_end = pipeline.pop_event().expect("end event in queue");
    assert!(matches!(popped_end, AudioPipelineEvent::SpeechEnd { .. }));
    assert!(pipeline.pop_event().is_none());
}

#[test]
fn test_audio_pipeline_pending_events_bounded_cap() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    let chunk = vec![0.1f32; 1600];
    // Each iteration produces SpeechStart on chunk and SpeechEnd on flush (2 events)
    for _ in 0..100 {
        let _ = pipeline.process_chunk(&chunk).expect("process chunk");
        let _ = pipeline.flush().expect("flush succeeds");
    }

    let mut count = 0;
    while pipeline.pop_event().is_some() {
        count += 1;
    }
    // High water mark must cap at MAX_PENDING_EVENTS (128)
    assert_eq!(count, 128);
}

/// Deterministic pseudo-speech frame for tests that drive the real VAD model.
///
/// A real Silero VAD classifies a constant-DC signal as non-speech
/// (probability ~0.02), so flat-amplitude frames can never trigger
/// `SpeechStart`. This generator instead emits a syllabic voiced / noise-burst
/// / gap rhythm (LCG-driven, fully deterministic) that the model sustains as
/// speech across dozens of frames. Frame `i` always yields the same samples,
/// so multi-chunk tests stay aligned by passing the chunk index.
///
/// Gated like its callers: every test driving the real VAD model is
/// `not(wasm32)` (it loads a GGUF file), so without this the helper is
/// dead code on wasm32 and trips `-D warnings`.
#[cfg(not(target_arch = "wasm32"))]
fn speech_like_frame(frame: usize) -> Vec<f32> {
    const TAU: f32 = 2.0 * std::f32::consts::PI;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (u32::MAX as f32)
        }
    }

    // Walk the deterministic segment map up to `frame`: voiced (3-9 frames) /
    // noise burst (3-9 frames) / gap (1-2 frames), rotating.
    let mut rng = Rng(99);
    let mut start = 0;
    let mut kind = 0;
    let (mut seg_kind, mut seg_start) = (0, 0);
    loop {
        if start > frame {
            break;
        }
        seg_kind = kind;
        seg_start = start;
        let len = if kind == 2 {
            1 + (rng.next() * 2.0) as usize
        } else {
            3 + (rng.next() * 7.0) as usize
        };
        start += len;
        kind = (kind + 1) % 3;
    }
    if seg_kind == 2 {
        return vec![0.0f32; 512];
    }

    let mut rng = Rng(1000 + seg_start as u64 * 131);
    let f0 = 110.0 + 160.0 * rng.next();
    let amp = 0.18 + 0.12 * rng.next();
    let offset = frame * 512;
    if seg_kind == 1 {
        return (0..512)
            .map(|n| {
                let env = 1.0 - n as f32 / 512.0;
                (Rng(n as u64 * 17 + offset as u64).next() - 0.5) * 0.5 * env
            })
            .collect();
    }
    (0..512)
        .map(|n| {
            let t = (offset + n) as f32 / 16000.0;
            let s = (TAU * f0 * t).sin()
                + 0.5 * (TAU * 2.02 * f0 * t).sin()
                + 0.25 * (TAU * 2.97 * f0 * t).sin();
            amp * s
        })
        .collect()
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_audio_pipeline_max_utterance_preserves_vad_state_on_continuation() {
    let candidates = [
        std::path::PathBuf::from("../../models/silero_vad.gguf"),
        std::path::PathBuf::from("models/silero_vad.gguf"),
        std::path::PathBuf::from("../models/silero_vad.gguf"),
    ];
    let Some(vad_path) = candidates.into_iter().find(|p| p.exists()) else {
        eprintln!(
            "Skipping test_audio_pipeline_max_utterance_preserves_vad_state_on_continuation: models/silero_vad.gguf not found"
        );
        return;
    };

    let config = AudioPipelineConfig {
        max_utterance_ms: 1000,
        auto_transcribe: false,
        ..Default::default()
    };

    let mut pipeline = AudioPipelineBuilder::new()
        .with_config(config)
        .with_vad_from_file(vad_path)
        .expect("load vad")
        .build()
        .expect("builder succeeds");

    // Feed speech-like audio to warm up VAD hidden states
    // 512 samples per frame (standard 16kHz window). Frames are indexed so the
    // syllabic rhythm stays aligned across chunks.
    let mut triggered = false;
    for i in 0..40 {
        let events = pipeline
            .process_chunk(&speech_like_frame(i))
            .expect("process chunk");
        for ev in &events {
            if matches!(ev, AudioPipelineEvent::SpeechStart { .. }) {
                triggered = true;
            }
        }
    }
    assert!(triggered, "VAD should have triggered SpeechStart");
    assert!(pipeline.is_speech_active());

    let (h_before, c_before) = pipeline.vad().unwrap().hidden_states();
    assert!(
        h_before.iter().any(|&v| v != 0.0),
        "VAD hidden state h should be non-zero during active speech"
    );
    assert!(
        c_before.iter().any(|&v| v != 0.0),
        "VAD cell state c should be non-zero during active speech"
    );

    // Continue feeding speech until max utterance (16,000 samples = 1s) is exceeded
    let mut hit_speech_end = false;
    let mut hit_new_speech_start = false;
    for i in 40..80 {
        let events = pipeline
            .process_chunk(&speech_like_frame(i))
            .expect("process chunk");
        for ev in &events {
            match ev {
                AudioPipelineEvent::SpeechEnd { .. } => hit_speech_end = true,
                AudioPipelineEvent::SpeechStart { .. } => hit_new_speech_start = true,
                _ => {}
            }
        }
        if hit_speech_end {
            break;
        }
    }

    assert!(
        hit_speech_end,
        "Duration cutoff should have emitted SpeechEnd"
    );
    assert!(
        hit_new_speech_start,
        "Continuation should have emitted new SpeechStart"
    );
    assert!(
        pipeline.is_speech_active(),
        "Pipeline should remain in SpeechActive"
    );
    assert_eq!(pipeline.state(), AudioPipelineState::SpeechActive);

    // Verify VAD hidden states were preserved and not reset to 0
    let (h_after, c_after) = pipeline.vad().unwrap().hidden_states();
    assert!(
        h_after.iter().any(|&v| v != 0.0),
        "VAD hidden state h must not be wiped to zero on max duration cutoff"
    );
    assert!(
        c_after.iter().any(|&v| v != 0.0),
        "VAD cell state c must not be wiped to zero on max duration cutoff"
    );
}

#[test]
fn test_audio_pipeline_sanitizes_nan_audio() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    let nan_chunk = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 0.5];
    let events = pipeline.process_chunk(&nan_chunk).expect("process chunk");

    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], AudioPipelineEvent::SpeechStart { .. }));
    let last = pipeline.flush().expect("flush succeeds");
    assert_eq!(last.len(), 1);
    assert_eq!(pipeline.last_utterance().len(), 4);
    assert_eq!(pipeline.last_utterance()[0], 0.0);
    assert_eq!(pipeline.last_utterance()[1], 0.0);
    assert_eq!(pipeline.last_utterance()[2], 0.0);
    assert_eq!(pipeline.last_utterance()[3], 0.5);
}

#[test]
fn test_audio_pipeline_pop_event_and_take_utterance() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    let chunk = vec![0.2f32; 1600];
    let events = pipeline.process_chunk(&chunk).expect("process chunk");
    assert_eq!(events.len(), 1);

    // pop_event drains the queued event
    let popped = pipeline.pop_event();
    assert!(matches!(
        popped,
        Some(AudioPipelineEvent::SpeechStart { .. })
    ));

    let end_events = pipeline.flush().expect("flush succeeds");
    assert_eq!(end_events.len(), 1);

    let utterance = pipeline.take_last_utterance();
    assert_eq!(utterance.len(), 1600);
    assert!(pipeline.last_utterance().is_empty());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_audio_pipeline_listening_for_speech_handles_speech_end_in_same_chunk() {
    let candidates = [
        std::path::PathBuf::from("../../models/silero_vad.gguf"),
        std::path::PathBuf::from("models/silero_vad.gguf"),
        std::path::PathBuf::from("../models/silero_vad.gguf"),
    ];
    let Some(vad_path) = candidates.into_iter().find(|p| p.exists()) else {
        eprintln!(
            "Skipping test_audio_pipeline_listening_for_speech_handles_speech_end_in_same_chunk: models/silero_vad.gguf not found"
        );
        return;
    };

    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .with_vad_from_file(vad_path)
        .expect("load vad")
        .build()
        .expect("builder succeeds");

    // First warm up with silence so VAD baseline is stable
    let silence = vec![0.0f32; 512];
    for _ in 0..5 {
        pipeline.process_chunk(&silence).expect("process chunk");
    }
    assert_eq!(pipeline.state(), AudioPipelineState::ListeningForSpeech);

    // Create a multi-frame buffer: 30 speech-like frames followed by 40
    // silence frames. Flat-DC frames never trigger a real Silero VAD, so the
    // speech section uses the same syllabic stimulus as the other VAD tests.
    let mut multi_frame_chunk = Vec::with_capacity(70 * 512);
    for i in 0..30 {
        multi_frame_chunk.extend_from_slice(&speech_like_frame(i));
    }
    for _ in 0..40 {
        multi_frame_chunk.extend_from_slice(&[0.0f32; 512]);
    }

    let events = pipeline
        .process_chunk(&multi_frame_chunk)
        .expect("process multi-frame chunk");

    let has_speech_start = events
        .iter()
        .any(|ev| matches!(ev, AudioPipelineEvent::SpeechStart { .. }));
    let has_speech_end = events
        .iter()
        .any(|ev| matches!(ev, AudioPipelineEvent::SpeechEnd { .. }));

    assert!(
        has_speech_start,
        "Chunk should emit SpeechStart when speech frames trigger VAD"
    );
    assert!(
        has_speech_end,
        "Chunk should emit SpeechEnd when trailing silence ends speech"
    );
    assert_eq!(
        pipeline.state(),
        AudioPipelineState::ListeningForSpeech,
        "Pipeline must return to ListeningForSpeech after utterance completes"
    );
}

#[test]
fn test_audio_pipeline_utterance_buffer_allocation_reused() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    let chunk = vec![0.1f32; 1600];
    pipeline.process_chunk(&chunk).expect("process chunk 1");
    pipeline.flush().expect("flush 1");
    let ptr1 = pipeline.last_utterance().as_ptr();
    assert_eq!(pipeline.last_utterance().len(), 1600);

    // Turn 2 uses the swapped buffer
    pipeline.process_chunk(&chunk).expect("process chunk 2");
    pipeline.flush().expect("flush 2");
    let ptr2 = pipeline.last_utterance().as_ptr();
    assert_eq!(pipeline.last_utterance().len(), 1600);

    // Turn 3 returns to the original buffer, proving ping-pong without reallocation
    pipeline.process_chunk(&chunk).expect("process chunk 3");
    pipeline.flush().expect("flush 3");
    let ptr3 = pipeline.last_utterance().as_ptr();
    assert_eq!(pipeline.last_utterance().len(), 1600);
    assert_eq!(ptr3, ptr1, "buffers must ping-pong without reallocation");
    assert_ne!(ptr1, ptr2, "buffers must swap alternating instances");
}

#[test]
fn test_audio_pipeline_take_last_utterance_reserves_capacity_on_next_turn() {
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .build()
        .expect("builder succeeds");

    let chunk = vec![0.1f32; 1600];
    pipeline.process_chunk(&chunk).expect("process chunk 1");
    pipeline.flush().expect("flush 1");

    // Taking the utterance empties last_utterance and leaves it with 0 capacity
    let taken = pipeline.take_last_utterance();
    assert_eq!(taken.len(), 1600);
    assert!(pipeline.last_utterance().is_empty());

    // Subsequent turn must cleanly restore capacity without panic
    pipeline.process_chunk(&chunk).expect("process chunk 2");
    pipeline.flush().expect("flush 2");
    assert_eq!(pipeline.last_utterance().len(), 1600);
}

#[test]
fn test_audio_pipeline_flush_clamps_speech_end_to_utterance_start_sample() {
    let config = AudioPipelineConfig {
        auto_transcribe: false,
        max_utterance_ms: 100, // 1600 samples
        ..Default::default()
    };
    let mut pipeline = AudioPipelineBuilder::new()
        .with_config(config)
        .build()
        .expect("builder succeeds");

    // Push 3200 samples (2 chunks of 1600) to exceed max utterance duration
    let chunk = vec![0.1f32; 1600];
    pipeline.process_chunk(&chunk).expect("chunk 1");
    let cutoff_start = pipeline.current_utterance_start_sample;

    // Flush active speech
    let events = pipeline.flush().expect("flush succeeds");
    for event in &events {
        if let AudioPipelineEvent::SpeechEnd {
            start_sample,
            end_sample,
            start_ms,
            end_ms,
        } = event
        {
            assert!(
                start_sample <= end_sample,
                "start_sample ({start_sample}) must not exceed end_sample ({end_sample})"
            );
            assert!(
                start_ms <= end_ms,
                "start_ms ({start_ms}) must not exceed end_ms ({end_ms})"
            );
            assert!(
                start_sample >= &cutoff_start,
                "start_sample ({start_sample}) must be clamped to cutoff boundary ({cutoff_start})"
            );
        }
    }
}

/// The pipeline tries the NPU for its VAD by default (in a build that has one) and
/// `with_vad_on_cpu(true)` opts out; the `build` branch behind the flag is truth-tabled so the
/// wiring, not just the flag, is pinned.
#[test]
fn test_audio_pipeline_builder_vad_on_cpu_opt_out() {
    assert!(!AudioPipelineBuilder::new().vad_on_cpu);
    assert!(AudioPipelineBuilder::new().with_vad_on_cpu(true).vad_on_cpu);
    assert!(
        !AudioPipelineBuilder::new()
            .with_vad_on_cpu(false)
            .vad_on_cpu
    );
}

#[test]
fn vad_backend_action_truth_table() {
    use super::{VadBackendAction::*, vad_backend_action};
    for (on_cpu, has_vad, accelerated, want) in [
        (false, false, false, Keep),
        (false, false, true, Keep),
        (false, true, false, TryHexagon),
        (false, true, true, Keep),
        (true, false, false, Keep),
        (true, false, true, Keep),
        (true, true, false, ForceCpu),
        (true, true, true, ForceCpu),
    ] {
        assert_eq!(
            vad_backend_action(on_cpu, has_vad, accelerated),
            want,
            "on_cpu={on_cpu} has_vad={has_vad} accelerated={accelerated}"
        );
    }
}

/// A VAD over zero weights built from synthetic GGUF bytes: no model file, so the `build`
/// backend wiring is pinned on every host.
fn synthetic_vad() -> crate::vad::SileroVad {
    let tensors = [
        ("stft.16k.basis", 258 * 256),
        ("encoder.16k.0.weight", 128 * 129 * 3),
        ("encoder.16k.0.bias", 128),
        ("encoder.16k.1.weight", 64 * 128 * 3),
        ("encoder.16k.1.bias", 64),
        ("encoder.16k.2.weight", 64 * 64 * 3),
        ("encoder.16k.2.bias", 64),
        ("encoder.16k.3.weight", 128 * 64 * 3),
        ("encoder.16k.3.bias", 128),
        ("decoder.16k.rnn.weight_ih", 512 * 128),
        ("decoder.16k.rnn.weight_hh", 512 * 128),
        ("decoder.16k.rnn.bias_ih", 512),
        ("decoder.16k.rnn.bias_hh", 512),
        ("decoder.16k.head.weight", 128),
        ("decoder.16k.head.bias", 1),
        ("stft.8k.basis", 130 * 128),
        ("encoder.8k.0.weight", 128 * 65 * 3),
        ("encoder.8k.0.bias", 128),
        ("encoder.8k.1.weight", 64 * 128 * 3),
        ("encoder.8k.1.bias", 64),
        ("encoder.8k.2.weight", 64 * 64 * 3),
        ("encoder.8k.2.bias", 64),
        ("encoder.8k.3.weight", 128 * 64 * 3),
        ("encoder.8k.3.bias", 128),
        ("decoder.8k.rnn.weight_ih", 512 * 128),
        ("decoder.8k.rnn.weight_hh", 512 * 128),
        ("decoder.8k.rnn.bias_ih", 512),
        ("decoder.8k.rnn.bias_hh", 512),
        ("decoder.8k.head.weight", 128),
        ("decoder.8k.head.bias", 1),
    ];
    let mut gguf = crate::gguf::GgufBuilder::new();
    for (name, numel) in tensors {
        gguf = gguf.tensor_f32(name, &[numel], &vec![0.0f32; numel]);
    }
    crate::vad::SileroVad::from_bytes(gguf.build_bytes()).unwrap()
}

struct FakeAccel;

impl crate::vad::VadAccelerator for FakeAccel {
    fn window_16k(
        &self,
        _: &[f32; 640],
        _: &[f32; 128],
        _: &[f32; 128],
    ) -> anyhow::Result<Option<crate::vad::VadStep>> {
        Ok(Some(crate::vad::VadStep {
            prob: 0.5,
            h: [0.0; 128],
            c: [0.0; 128],
        }))
    }
}

#[test]
fn build_with_vad_on_cpu_drops_a_pre_attached_accelerator() {
    let mut vad = synthetic_vad();
    vad.set_accelerator(Arc::new(FakeAccel));
    let pipeline = AudioPipelineBuilder::new()
        .with_vad(vad)
        .with_vad_on_cpu(true)
        .build()
        .unwrap();
    assert!(!pipeline.vad().unwrap().is_accelerated());
}

#[test]
fn build_with_config_vad_on_cpu_drops_a_pre_attached_accelerator() {
    let mut vad = synthetic_vad();
    vad.set_accelerator(Arc::new(FakeAccel));
    let config = AudioPipelineConfig {
        vad_on_cpu: true,
        ..Default::default()
    };
    let pipeline = AudioPipelineBuilder::new()
        .with_vad(vad)
        .with_config(config)
        .build()
        .unwrap();
    assert!(!pipeline.vad().unwrap().is_accelerated());
}

#[test]
fn build_without_opt_out_keeps_a_pre_attached_accelerator() {
    // The Keep arm never touches the DSP, so this runs on every host in every build.
    let mut vad = synthetic_vad();
    vad.set_accelerator(Arc::new(FakeAccel));
    let pipeline = AudioPipelineBuilder::new().with_vad(vad).build().unwrap();
    assert!(pipeline.vad().unwrap().is_accelerated());
}

/// Utterances still waiting on the diarizer are forgotten with the session on `reset`: nothing
/// outside can see them (the new session never returns them), so they would only pile up.
#[cfg(feature = "mmap")]
#[test]
fn reset_drops_the_utterances_waiting_on_the_old_diarizer_session() {
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
    let model = crate::model::sortformer::SortformerModel::from_file(&path).unwrap();
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .with_diarizer(model.clone(), model.default_streaming().clone())
        .build()
        .unwrap();
    assert!(pipeline.add_utterance("a".into(), 0.0, 100.0));
    assert!(pipeline.add_utterance("b".into(), 200.0, 300.0));
    assert_eq!(pipeline.diarizer.as_ref().unwrap().pending.len(), 2);
    pipeline.reset();
    assert!(pipeline.has_diarizer());
    assert!(pipeline.diarizer.as_ref().unwrap().pending.is_empty());
    assert_eq!(pipeline.diarizer.as_ref().unwrap().origin_ms, 0.0);
}

/// Past `max_pending` waiting utterances `register` refuses instead of returning true for
/// an utterance the labeler can never park (it would pin its text with no event ever).
#[cfg(feature = "mmap")]
#[test]
fn register_refuses_utterances_past_max_pending() {
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
    let model = crate::model::sortformer::SortformerModel::from_file(&path).unwrap();
    let cfg = crate::speaker_labeler::SpeakerLabelerConfig {
        max_pending: 2,
        ..Default::default()
    };
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .with_diarizer(model.clone(), model.default_streaming().clone())
        .with_speaker_labeler_config(cfg)
        .build()
        .unwrap();
    assert!(pipeline.add_utterance("a".into(), 0.0, 100.0));
    assert!(pipeline.add_utterance("b".into(), 200.0, 300.0));
    assert!(
        !pipeline.add_utterance("c".into(), 400.0, 500.0),
        "the third utterance past max_pending 2 must be refused"
    );
    assert_eq!(pipeline.diarizer.as_ref().unwrap().pending.len(), 2);
}

/// An utterance with non-finite times is granted, then flushed as `dropped`: the labeler
/// cannot place it on the clock, so it must read as a give-up, not as silence.
#[cfg(feature = "mmap")]
#[test]
fn non_finite_utterance_times_flush_as_dropped() {
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
    let model = crate::model::sortformer::SortformerModel::from_file(&path).unwrap();
    let mut pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .with_diarizer(model.clone(), model.default_streaming().clone())
        .build()
        .unwrap();
    assert!(pipeline.add_utterance("when".into(), f32::NAN, f32::NAN));
    let events = pipeline.flush().unwrap();
    assert_eq!(events.len(), 1, "{events:?}");
    match &events[0] {
        AudioPipelineEvent::UtteranceLabeled {
            speaker, dropped, ..
        } => {
            assert_eq!(*speaker, None);
            assert!(*dropped, "{events:?}");
        }
        other => panic!("wrong event: {other:?}"),
    }
}

/// The bytes loader attaches the same checkpoint as the file loader.
#[cfg(feature = "mmap")]
#[test]
fn with_diarizer_from_bytes_matches_from_file() {
    let Some(path) = sortformer_model_or_skip() else {
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    let pipeline = AudioPipelineBuilder::new()
        .with_auto_transcribe(false)
        .with_diarizer_from_bytes(bytes)
        .expect("bytes loader attaches the diarizer")
        .build()
        .unwrap();
    assert!(pipeline.has_diarizer());
    let from_file = crate::model::sortformer::SortformerModel::from_file(&path).unwrap();
    assert_eq!(
        pipeline.diarizer.as_ref().unwrap().params,
        *from_file.default_streaming(),
        "the bytes loader uses the checkpoint's own streaming parameters"
    );
}

/// Session-local time without a model: spans inside the session pass through, spans that
/// began before it clamp at the session's first frame.
#[test]
fn session_local_ms_clamps_straddling_spans_at_zero() {
    assert_eq!(session_local_ms(1500.0, 1000.0), 500.0);
    assert_eq!(session_local_ms(1000.0, 1000.0), 0.0);
    assert_eq!(session_local_ms(500.0, 1000.0), 0.0);
    assert_eq!(session_local_ms(0.0, 0.0), 0.0);
}

fn pending_utterance() -> (u64, PendingUtterance) {
    (
        7,
        PendingUtterance {
            text: "hello".into(),
            start_ms: 100.0,
            end_ms: 900.0,
        },
    )
}

fn labeled(id: u64, label: Option<crate::speaker_labeler::SpeakerLabel>) -> LabeledUtterance {
    LabeledUtterance {
        id,
        start_ms: 100.0,
        end_ms: 900.0,
        label,
        dropped: false,
    }
}

fn speaker_label() -> crate::speaker_labeler::SpeakerLabel {
    crate::speaker_labeler::SpeakerLabel {
        speaker: 2,
        confidence: 0.75,
        overlapping: Some(1),
        active: [0.0; crate::speaker_labeler::SPEAKERS],
        activity: [0.0; crate::speaker_labeler::SPEAKERS],
    }
}

/// Label mapping without a model: the event carries the utterance's own text and span with
/// the diarizer's speaker, and taking it consumes the pending entry.
#[test]
fn take_labeled_event_maps_speaker_and_consumes_pending() {
    let (id, p) = pending_utterance();
    let mut pending = std::collections::HashMap::from([(id, p)]);
    let ev = take_labeled_event(&mut pending, labeled(id, Some(speaker_label())))
        .expect("registered id labels");
    assert!(pending.is_empty(), "labeling consumes the pending entry");
    match ev {
        AudioPipelineEvent::UtteranceLabeled {
            text,
            start_ms,
            end_ms,
            speaker,
            confidence,
            overlapping,
            dropped,
        } => {
            assert_eq!(text, "hello");
            assert_eq!((start_ms, end_ms), (100.0, 900.0));
            assert_eq!(speaker, Some(2));
            assert_eq!(confidence, Some(0.75));
            assert_eq!(overlapping, Some(1));
            assert!(!dropped);
        }
        other => panic!("wrong event: {other:?}"),
    }
}

/// A dropped utterance keeps its dropped flag through the event: silence (`None` speaker,
/// not dropped) and a stalled diarizer (`None` speaker, dropped) stay distinguishable.
#[test]
fn take_labeled_event_carries_the_dropped_flag() {
    let (id, p) = pending_utterance();
    let mut pending = std::collections::HashMap::from([(id, p)]);
    let mut u = labeled(id, None);
    u.dropped = true;
    let ev = take_labeled_event(&mut pending, u).expect("registered id labels");
    match ev {
        AudioPipelineEvent::UtteranceLabeled {
            speaker, dropped, ..
        } => {
            assert_eq!(speaker, None);
            assert!(dropped);
        }
        other => panic!("wrong event: {other:?}"),
    }
}

/// No speaker found: the text and span still come through, the speaker fields stay empty.
#[test]
fn take_labeled_event_without_a_label_keeps_text_and_span() {
    let (id, p) = pending_utterance();
    let mut pending = std::collections::HashMap::from([(id, p)]);
    let ev = take_labeled_event(&mut pending, labeled(id, None)).expect("registered id labels");
    match ev {
        AudioPipelineEvent::UtteranceLabeled {
            speaker,
            confidence,
            overlapping,
            text,
            dropped,
            ..
        } => {
            assert_eq!(text, "hello");
            assert_eq!(speaker, None);
            assert_eq!(confidence, None);
            assert_eq!(overlapping, None);
            assert!(!dropped, "covered silence is not a give-up");
        }
        other => panic!("wrong event: {other:?}"),
    }
}

/// Unknown ids (never registered, or taken already) yield nothing and touch nothing.
#[test]
fn take_labeled_event_ignores_unknown_ids() {
    let (id, p) = pending_utterance();
    let mut pending = std::collections::HashMap::from([(id, p)]);
    assert!(take_labeled_event(&mut pending, labeled(999, Some(speaker_label()))).is_none());
    assert_eq!(pending.len(), 1, "an unknown id takes nothing");
    assert!(take_labeled_event(&mut pending, labeled(id, Some(speaker_label()))).is_some());
    assert!(take_labeled_event(&mut pending, labeled(id, Some(speaker_label()))).is_none());
}

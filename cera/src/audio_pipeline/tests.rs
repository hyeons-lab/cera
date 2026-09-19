//! Unit tests for the unified AudioPipeline facade.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

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

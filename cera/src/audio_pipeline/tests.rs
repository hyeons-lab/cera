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

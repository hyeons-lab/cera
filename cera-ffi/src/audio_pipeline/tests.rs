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

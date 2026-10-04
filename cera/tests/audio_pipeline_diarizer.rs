//! The speaker diarizer inside `AudioPipeline`: `with_diarizer`, `add_utterance` and the
//! `UtteranceLabeled` event, on the committed three-speaker clip.
//!
//! The Sortformer model lives in `~/.leap/models/sortformer/` and the test skips with a message
//! when it is absent (`CERA_REQUIRE_MODEL=1` turns a skip into a failure). Run it with
//! `--release`: the CPU model is slow in a debug build.

#![cfg(feature = "mmap")]

mod common;

use std::path::PathBuf;

use cera::audio_pipeline::{AudioPipeline, AudioPipelineEvent};
use cera::model::sortformer::SortformerModel;

const MODEL: &str = "sortformer-4spk-v2.1-q8_0.gguf";

fn model() -> Option<SortformerModel> {
    let path = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".leap/models/sortformer")
        .join(MODEL);
    if !path.exists() {
        assert!(
            std::env::var("CERA_REQUIRE_MODEL").as_deref() != Ok("1"),
            "CERA_REQUIRE_MODEL=1 but {} is absent",
            path.display()
        );
        eprintln!("{} not found, skipping", path.display());
        return None;
    }
    Some(SortformerModel::from_file(&path).unwrap())
}

/// The committed clip: 15.4 s, three speakers taking turns.
fn clip() -> Vec<f32> {
    common::read_wav_f32(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sortformer/clip.wav"),
    )
}

/// A `data` chunk before the validated `fmt` chunk is refused, even when the `fmt`
/// chunk itself is well-formed: decoding stereo 48 kHz bytes as mono 16 kHz would
/// still print plausible numbers. This pins the shared `fmt_ok` predicate that the
/// `hexagon_sortformer_probe` twin mirrors textually (a device-only example whose
/// own reader cannot run under host CI).
#[test]
#[should_panic(expected = "data before fmt")]
fn wav_reader_refuses_data_before_fmt() {
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&0u32.to_le_bytes()); // Length is unchecked; chunks bound it.
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&2u32.to_le_bytes());
    wav.extend_from_slice(&[0, 0]);
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes()); // Byte rate.
    wav.extend_from_slice(&2u16.to_le_bytes()); // Block align.
    wav.extend_from_slice(&16u16.to_le_bytes());
    // Fixed name: every run writes identical bytes, so concurrent runs are benign and
    // no per-run litter accumulates (`should_panic` unwinds before cleanup could run).
    let path = std::env::temp_dir().join("cera-wav-data-before-fmt.wav");
    std::fs::write(&path, wav).unwrap();
    common::read_wav_f32(&path);
}

/// Without a VAD the pipeline treats the stream as one utterance and transcribes nothing, so
/// the test registers the utterances itself, as a caller with its own recognizer would.
fn pipeline(m: &SortformerModel) -> AudioPipeline {
    AudioPipeline::builder()
        .with_auto_transcribe(false)
        .with_diarizer(m.clone(), m.default_streaming().clone())
        .build()
        .unwrap()
}

fn labeled(events: &[AudioPipelineEvent]) -> Vec<(String, f32, f32, Option<u32>)> {
    events
        .iter()
        .filter_map(|e| match e {
            AudioPipelineEvent::UtteranceLabeled {
                text,
                start_ms,
                end_ms,
                speaker,
                ..
            } => Some((text.clone(), *start_ms, *end_ms, *speaker)),
            _ => None,
        })
        .collect()
}

/// Push the clip through in 100 ms pieces, then flush; every event, in order.
fn run(p: &mut AudioPipeline, pcm: &[f32]) -> Vec<AudioPipelineEvent> {
    let mut events = Vec::new();
    for piece in pcm.chunks(1600) {
        events.extend(p.process_chunk(piece).unwrap());
    }
    events.extend(p.flush().unwrap());
    events
}

#[test]
fn utterances_are_labeled_with_the_speaker_who_held_the_floor() {
    let Some(m) = model() else { return };
    let pcm = clip();
    let mut p = pipeline(&m);
    assert!(p.has_diarizer());

    // The diarizer's own segmentation of this clip (`cera diarize`): speakers 0, 1 and 2 in turn.
    for (text, start, end) in [
        ("first", 480.0, 4160.0),
        ("second", 4960.0, 8240.0),
        ("third", 8480.0, 10640.0),
    ] {
        assert!(p.add_utterance(text.into(), start, end));
    }
    let events = run(&mut p, &pcm);
    assert_eq!(
        labeled(&events),
        vec![
            ("first".into(), 480.0, 4160.0, Some(0)),
            ("second".into(), 4960.0, 8240.0, Some(1)),
            ("third".into(), 8480.0, 10640.0, Some(2)),
        ]
    );
    // The same events are queued for `pop_event` callers.
    let mut queued = Vec::new();
    while let Some(e) = p.pop_event() {
        queued.push(e);
    }
    assert_eq!(labeled(&queued).len(), 3);
}

/// A flush ends the diarizer's session and starts a new one at the pipeline's current sample:
/// utterances in the second pass are timed on the pipeline's clock, not the session's.
#[test]
fn a_flush_restarts_the_session_on_the_pipeline_clock() {
    let Some(m) = model() else { return };
    let pcm = clip();
    let mut p = pipeline(&m);
    p.add_utterance("one".into(), 480.0, 4160.0);
    let first = run(&mut p, &pcm);
    assert_eq!(labeled(&first).len(), 1);

    let t0 = p.current_sample() as f32 / 16.0;
    assert!(t0 > 15_000.0);
    assert!(p.add_utterance("two".into(), t0 + 4960.0, t0 + 8240.0));
    let second = run(&mut p, &pcm);
    let got = labeled(&second);
    assert_eq!(got.len(), 1, "{got:?}");
    // The event echoes the pipeline-clock times it was registered with, and finds the speaker
    // who talks 4.96 to 8.24 s into the clip.
    assert_eq!((got[0].1, got[0].2), (t0 + 4960.0, t0 + 8240.0));
    assert!(got[0].3.is_some(), "no speaker over a span of speech");
}

/// An utterance over silence (before anyone speaks) comes back with no speaker, not dropped.
#[test]
fn an_utterance_over_silence_has_no_speaker() {
    let Some(m) = model() else { return };
    let mut p = pipeline(&m);
    p.add_utterance("hush".into(), 0.0, 300.0);
    let events = run(&mut p, &clip());
    assert_eq!(
        labeled(&events),
        vec![("hush".into(), 0.0, 300.0, None)],
        "{events:?}"
    );
}

/// A diarizer that fails (here: a sample the front end refuses) stops with a warning; the
/// pipeline carries on and reports that it has no diarizer.
#[test]
fn a_failing_diarizer_does_not_stop_the_pipeline() {
    let Some(m) = model() else { return };
    let mut p = pipeline(&m);
    assert!(p.has_diarizer());
    let mut bad = vec![0.0f32; 1600];
    bad[10] = 2e9; // finite, so the pipeline passes it on; the mel front end refuses it
    p.process_chunk(&bad).expect("the pipeline itself is fine");
    assert!(!p.has_diarizer());
    assert!(!p.add_utterance("late".into(), 0.0, 100.0));
    let more = p.process_chunk(&vec![0.0f32; 1600]).unwrap();
    assert!(labeled(&more).is_empty());
    assert_eq!(p.current_sample(), 3200);
}

/// Child leg of [`a_failing_diarizer_warns_on_stderr_without_a_subscriber`]: fails the
/// diarizer once and returns. Harmless under the normal suite; the parent runs it in a
/// subprocess.
#[test]
fn failing_diarizer_stderr_child_fails_once() {
    let Some(m) = model() else { return };
    let mut p = pipeline(&m);
    let mut bad = vec![0.0f32; 1600];
    bad[10] = 2e9; // finite, so the pipeline passes it on; the mel front end refuses it
    p.process_chunk(&bad).expect("the pipeline itself is fine");
    assert!(!p.has_diarizer());
}

/// The diarizer-failure line reaches stderr with no subscriber installed: `fail()` clears
/// the session and pending utterances, and the `eprintln!` is the only on-device record
/// of why labels stopped, so deleting it must fail this test, not pass silently. A
/// subprocess with `--nocapture` carries the proof: the harness captures `eprintln!`
/// in-process, so only a subprocess's real fd 2 shows the emission.
#[test]
fn a_failing_diarizer_warns_on_stderr_without_a_subscriber() {
    if model().is_none() {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .arg("--exact")
        .arg("failing_diarizer_stderr_child_fails_once")
        .arg("--nocapture")
        .output()
        .unwrap();
    assert!(out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.lines().any(
            |l| l.starts_with("cera-audio-pipeline: speaker diarizer audio failed")
                && l.contains("continuing without speaker labels")
        ),
        "stderr must carry the diarizer failure: {err:?}"
    );
}

/// `reset` starts a new stream: pending utterances are forgotten and the clock restarts at zero.
#[test]
fn reset_forgets_pending_utterances_and_restarts_the_clock() {
    let Some(m) = model() else { return };
    let pcm = clip();
    let mut p = pipeline(&m);
    p.add_utterance("stale".into(), 480.0, 4160.0);
    p.process_chunk(&pcm[..16_000]).unwrap();
    p.reset();
    assert!(p.has_diarizer());
    p.add_utterance("fresh".into(), 4960.0, 8240.0);
    let events = run(&mut p, &pcm);
    assert_eq!(
        labeled(&events),
        vec![("fresh".into(), 4960.0, 8240.0, Some(1))]
    );
}

#[test]
fn without_a_diarizer_nothing_is_labeled() {
    let mut p = AudioPipeline::builder()
        .with_auto_transcribe(false)
        .build()
        .unwrap();
    assert!(!p.has_diarizer());
    assert!(!p.add_utterance("x".into(), 0.0, 100.0));
    let events = p.process_chunk(&vec![0.0f32; 1600]).unwrap();
    assert!(labeled(&events).is_empty());
}

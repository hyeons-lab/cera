//! The speaker labeler against NeMo's actual predictions on the committed clip.
//!
//! `golden.json` (committed, no model needed) holds NeMo's per-frame speaker activities for
//! `clip.wav`, a 15 s clip built by `scripts/sortformer/make_clip.py`:
//!
//! ```text
//!  0.0  0.5 silence
//!  0.5  4.0 voice A     -> slot 0
//!  4.5  8.0 voice B     -> slot 1
//!  8.5 10.5 voice C     -> slot 2
//! 11.0 13.0 A over B    (overlapped; NeMo gives it to one or the other, frame by frame)
//! 13.5 15.4 voice C     -> slot 2
//! ```
//!
//! This is the full path a transcriber would take: utterances are registered when their text is
//! ready, the diarizer's frames arrive later in chunk-sized pieces, and each utterance is
//! released the moment its span is covered.

use cera::speaker_labeler::{FRAME_MS, SPEAKERS, SpeakerLabeler, SpeakerLabelerConfig};

fn golden_frames() -> Vec<f32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sortformer/golden.json");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let frames: Vec<f32> = json["preds_offline"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    assert_eq!(
        frames.len(),
        json["n_frames"].as_u64().unwrap() as usize * SPEAKERS
    );
    frames
}

/// (span in ms, expected slot). Spans sit inside the voices, away from their edges.
const SPANS: [((f64, f64), usize); 4] = [
    ((700.0, 3_800.0), 0),
    ((4_700.0, 7_800.0), 1),
    ((8_700.0, 10_300.0), 2),
    ((13_700.0, 15_200.0), 2),
];

#[test]
fn labels_each_voice_in_the_clip() {
    let mut l = SpeakerLabeler::new(SpeakerLabelerConfig::default());
    l.push_frames(&golden_frames());
    for ((a, b), slot) in SPANS {
        let label = l
            .label(a, b)
            .unwrap_or_else(|| panic!("{a}..{b} ms unlabeled"));
        assert_eq!(label.speaker, slot, "{a}..{b} ms: {label:?}");
        assert!(label.confidence > 0.9, "{a}..{b} ms: {label:?}");
    }
    // The opening silence is not speech, even though the model hovers near 0.35 for slot 0
    // before voice A (below the 0.5 decision, above a mean-probability floor), and neither is the
    // instant between voices B and C when B has faded and C has not yet crossed 0.5.
    assert_eq!(l.label(0.0, 400.0), None);
    assert_eq!(l.label(8_400.0, 8_440.0), None);
}

#[test]
fn utterances_are_released_as_the_diarizer_catches_up() {
    let frames = golden_frames();
    let n = frames.len() / SPEAKERS;
    let mut l = SpeakerLabeler::new(SpeakerLabelerConfig::default());
    // Whisper is done with every utterance up front; the diarizer delivers 12-frame chunks.
    for (id, ((a, b), _)) in SPANS.iter().enumerate() {
        l.add_utterance(id as u64, *a, *b);
    }
    let mut released = Vec::new();
    for (i, chunk) in frames.chunks(12 * SPEAKERS).enumerate() {
        l.push_frames(chunk);
        for u in l.poll() {
            // Never before the diarizer covers the utterance's end.
            assert!(
                u.end_ms <= l.covered_ms(),
                "chunk {i}: released early: {u:?}"
            );
            released.push((i, u));
        }
    }
    assert_eq!(l.frames_received(), n);
    released.extend(l.flush().into_iter().map(|u| (usize::MAX, u)));
    assert_eq!(released.len(), SPANS.len());
    for (idx, (_, u)) in released.iter().enumerate() {
        assert_eq!(u.id as usize, idx, "released in registration order here");
        let label = u.label.as_ref().expect("labeled");
        assert_eq!(label.speaker, SPANS[idx].1, "{u:?}");
    }
    // The first utterance ends at 3.8 s = frame 47.5, so it is out with the 4th chunk (48 frames).
    assert_eq!(released[0].0, 3);
    assert!(l.covered_ms() >= n as f64 * FRAME_MS - 1.0);
}

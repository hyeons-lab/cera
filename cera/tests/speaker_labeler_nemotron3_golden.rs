//! The speaker labeler at 8 slots against NeMo's actual Nemotron-3 predictions on the
//! committed clip.
//!
//! `tests/fixtures/nemotron3/golden.json` (committed, no model needed) holds NeMo's per-frame
//! speaker activities for `clip.wav` (see `scripts/nemotron3_diarization/gen_golden.py`):
//!
//! ```text
//!  0.0  0.5 silence
//!  0.5  4.0 voice A     -> slot 0
//!  4.9  8.0 voice B     -> slot 1
//!  8.5 10.5 voice C     -> slot 2
//! 11.0 13.0 A over B    (overlapped speech)
//! 13.5 15.4 voice C     -> slot 2
//! ```
//!
//! Mirrors `speaker_labeler_golden.rs` (4 slots, 80 ms) at the Nemotron-3 shape (8 slots,
//! 10 ms frames): the `presets.default.total_preds` frames feed a `SpeakerLabeler<8>`.

use cera::model::nemotron3_diarization::FRAME_MS as NEMOTRON3_FRAME_MS;
use cera::speaker_labeler::{SpeakerLabeler, SpeakerLabelerConfig};

const SLOTS: usize = 8;

fn golden_frames() -> Vec<f32> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/nemotron3/golden.json");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let frames: Vec<f32> = json["presets"]["default"]["total_preds"]
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
        json["n_mel_frames"].as_u64().unwrap() as usize * SLOTS
    );
    frames
}

fn labeler() -> SpeakerLabeler<SLOTS> {
    SpeakerLabeler::<SLOTS>::new(SpeakerLabelerConfig {
        frame_ms: NEMOTRON3_FRAME_MS,
        ..SpeakerLabelerConfig::default()
    })
}

/// (span in ms, expected slot). Spans sit inside the voices, away from their edges.
const SPANS: [((f64, f64), usize); 4] = [
    ((700.0, 3_800.0), 0),
    ((5_100.0, 7_800.0), 1),
    ((8_700.0, 10_300.0), 2),
    ((13_700.0, 15_200.0), 2),
];

#[test]
fn labels_each_voice_in_the_clip() {
    let mut l = labeler();
    l.push_frames(&golden_frames());
    for ((a, b), slot) in SPANS {
        let label = l
            .label(a, b)
            .unwrap_or_else(|| panic!("{a}..{b} ms unlabeled"));
        assert_eq!(label.speaker, slot, "{a}..{b} ms: {label:?}");
        assert!(label.confidence > 0.9, "{a}..{b} ms: {label:?}");
    }
    // The opening silence is not speech, and neither is the instant between voices B and C
    // when B has faded and C has not yet crossed 0.5.
    assert_eq!(l.label(0.0, 400.0), None);
    assert_eq!(l.label(8_100.0, 8_450.0), None);
}

#[test]
fn utterances_are_released_as_the_diarizer_catches_up() {
    let frames = golden_frames();
    let n = frames.len() / SLOTS;
    let mut l = labeler();
    // Whisper is done with every utterance up front; the diarizer delivers 12-frame chunks.
    for (id, ((a, b), _)) in SPANS.iter().enumerate() {
        l.add_utterance(id as u64, *a, *b);
    }
    let mut released = Vec::new();
    for (i, chunk) in frames.chunks(12 * SLOTS).enumerate() {
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
    // The first utterance ends at 3.8 s = frame 380, so it is out with the 32nd chunk.
    assert_eq!(released[0].0, 31);
    assert!(l.covered_ms() >= n as f64 * NEMOTRON3_FRAME_MS - 1.0);
}

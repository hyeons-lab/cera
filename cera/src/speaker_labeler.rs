//! Attach speaker labels to transcribed utterances, from per-frame speaker activity.
//!
//! The diarizer ([`crate::model::sortformer`]) and the transcriber
//! ([`crate::AudioPipeline`]) run side by side on the same audio and finish at different
//! times: Whisper has the text as soon as an utterance ends, while Sortformer's prediction for
//! the same stretch arrives a chunk plus its lookahead later (seconds, depending on the
//! preset). A [`SpeakerLabeler`] bridges that gap. Register each utterance's span when its text
//! is ready, feed it the diarizer's frames as they become final, and poll for utterances whose
//! whole span the diarizer has now covered. The text can be shown at once; the speaker follows.
//!
//! Both sides must count time from the same audio origin: frame `i` covers
//! `[i * 80, (i + 1) * 80)` ms of the audio pushed to the diarizer, and an utterance's
//! `start_ms`/`end_ms` are on that same clock (the pipeline's sample counter, if the same PCM is
//! fed to both from the start).
//!
//! **Speaker ids** are Sortformer's output slots, `0..4`. They are arrival ordered and stable for
//! a session (the speaker cache keeps a speaker in its slot), but they are not names, and they
//! restart when the diarizer does.
//!
//! **Rule.** A speaker counts as active in a frame when their probability reaches
//! `active_threshold` (0.5, the usual diarization decision). Over an utterance's span, a
//! speaker's *active fraction* is the share of the span they were active, weighting the frames at
//! the edges by how much of them the span covers. The speaker with the highest active fraction
//! (mean probability breaks ties) is the label, and is reported only if that fraction reaches
//! `min_active`, so a stretch the diarizer calls silence stays unlabeled even when it hovers
//! below the threshold (the model sits near 0.3 in the first moments of a stream, before it has
//! heard anyone). `confidence` is that speaker's share of all active time; a runner-up whose active
//! fraction reaches `overlap_threshold` of the winner's is reported as an overlap.

use std::collections::VecDeque;

/// Milliseconds per diarizer frame (the FastConformer's 8x subsampling of 10 ms mel frames).
pub const FRAME_MS: f32 = 80.0;

/// Speaker slots the diarizer predicts.
pub const SPEAKERS: usize = 4;

/// How the label is decided and how long frames are kept.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerLabelerConfig {
    /// A speaker is active in a frame when their probability reaches this.
    pub active_threshold: f32,
    /// The winner must be active for at least this fraction of the span, or it is unlabeled.
    pub min_active: f32,
    /// A runner-up counts as overlapping speech when its active fraction is at least this
    /// fraction of the winner's.
    pub overlap_threshold: f32,
    /// Frames older than this (relative to the newest) are dropped. An utterance older than the
    /// retained window when it is registered is returned unlabeled.
    pub history_ms: f32,
    /// Utterances waiting for the diarizer beyond this many are released unlabeled, oldest first,
    /// so a stalled diarizer cannot grow the queue without bound.
    pub max_pending: usize,
}

impl Default for SpeakerLabelerConfig {
    fn default() -> Self {
        Self {
            active_threshold: 0.5,
            min_active: 0.2,
            overlap_threshold: 0.5,
            history_ms: 10.0 * 60.0 * 1000.0,
            max_pending: 1024,
        }
    }
}

/// Who spoke during a span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerLabel {
    /// The most active speaker's slot (`0..4`).
    pub speaker: usize,
    /// That speaker's share of all speakers' active time over the span, in `(0, 1]`.
    pub confidence: f32,
    /// A second speaker who was also clearly active over the span, if any.
    pub overlapping: Option<usize>,
    /// Fraction of the span each slot was active (probability at or above `active_threshold`).
    pub active: [f32; SPEAKERS],
    /// Mean probability of each slot over the span.
    pub activity: [f32; SPEAKERS],
}

/// An utterance and the speaker the diarizer assigned it.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledUtterance {
    /// The id the caller registered the utterance with.
    pub id: u64,
    /// Utterance start, ms from the audio origin.
    pub start_ms: f32,
    /// Utterance end, ms from the audio origin.
    pub end_ms: f32,
    /// `None` when the diarizer found no speaker active over the span (or had no frames for it).
    pub label: Option<SpeakerLabel>,
}

struct Pending {
    id: u64,
    start_ms: f32,
    end_ms: f32,
}

/// Matches utterances with the diarizer's per-frame speaker activity. See the module docs.
pub struct SpeakerLabeler {
    cfg: SpeakerLabelerConfig,
    /// Frames from `first_frame` on.
    frames: VecDeque<[f32; SPEAKERS]>,
    /// Absolute index of `frames[0]`.
    first_frame: usize,
    pending: VecDeque<Pending>,
}

impl SpeakerLabeler {
    /// A labeler with `cfg`.
    pub fn new(cfg: SpeakerLabelerConfig) -> Self {
        Self {
            cfg,
            frames: VecDeque::new(),
            first_frame: 0,
            pending: VecDeque::new(),
        }
    }

    /// Frames received so far (80 ms each), including any already dropped from the history.
    pub fn frames_received(&self) -> usize {
        self.first_frame + self.frames.len()
    }

    /// Audio the diarizer has covered, in ms.
    pub fn covered_ms(&self) -> f32 {
        self.frames_received() as f32 * FRAME_MS
    }

    /// Utterances registered and not yet released.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Append the diarizer's next frames, `[k x 4]` row-major, in order (the output of
    /// `SortformerLive::push_audio` / `finish`).
    ///
    /// # Panics
    /// If `preds.len()` is not a multiple of 4.
    pub fn push_frames(&mut self, preds: &[f32]) {
        assert_eq!(
            preds.len() % SPEAKERS,
            0,
            "frames must be [k x {SPEAKERS}] speaker activities"
        );
        self.frames
            .extend(preds.as_chunks::<SPEAKERS>().0.iter().copied());
        // Keep the history window; pending utterances keep their frames alive until released.
        let keep = (self.cfg.history_ms / FRAME_MS).ceil().max(1.0) as usize;
        let oldest_needed = self
            .pending
            .iter()
            .map(|p| (p.start_ms / FRAME_MS).floor().max(0.0) as usize)
            .min()
            .unwrap_or(usize::MAX);
        while self.frames.len() > keep && self.first_frame < oldest_needed {
            self.frames.pop_front();
            self.first_frame += 1;
        }
    }

    /// Register an utterance whose text is ready. It is released by [`Self::poll`] once the
    /// diarizer's frames cover `end_ms`.
    pub fn add_utterance(&mut self, id: u64, start_ms: f32, end_ms: f32) {
        self.pending.push_back(Pending {
            id,
            start_ms: start_ms.max(0.0),
            end_ms: end_ms.max(start_ms.max(0.0)),
        });
    }

    /// Utterances whose whole span the diarizer has covered, in registration order, labeled.
    /// Also releases (unlabeled) the oldest utterances beyond `max_pending`.
    pub fn poll(&mut self) -> Vec<LabeledUtterance> {
        let mut out = Vec::new();
        while self.pending.len() > self.cfg.max_pending {
            let p = self.pending.pop_front().expect("len checked");
            out.push(self.release(p, false));
        }
        // Utterances finish in order, but a long one can end after a shorter later one; release
        // every covered one, not only the head.
        let covered = self.covered_ms();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].end_ms <= covered {
                let p = self.pending.remove(i).expect("index checked");
                out.push(self.release(p, true));
            } else {
                i += 1;
            }
        }
        out
    }

    /// End of stream: label every remaining utterance with the frames that exist (the diarizer
    /// has been flushed), whether or not they reach the utterance's end.
    pub fn flush(&mut self) -> Vec<LabeledUtterance> {
        let mut out = self.poll();
        while let Some(p) = self.pending.pop_front() {
            out.push(self.release(p, true));
        }
        out
    }

    fn release(&self, p: Pending, label: bool) -> LabeledUtterance {
        LabeledUtterance {
            id: p.id,
            start_ms: p.start_ms,
            end_ms: p.end_ms,
            label: if label {
                self.label(p.start_ms, p.end_ms)
            } else {
                None
            },
        }
    }

    /// Who spoke over `[start_ms, end_ms)`, from the frames received so far. `None` if no frame
    /// overlaps the span (not yet received, or older than the history) or nobody reaches
    /// `min_activity`.
    pub fn label(&self, start_ms: f32, end_ms: f32) -> Option<SpeakerLabel> {
        let (start_ms, end_ms) = (start_ms.max(0.0), end_ms.max(start_ms.max(0.0)));
        // A zero-length span is a point: give it the frame it falls in.
        let end_ms = end_ms.max(start_ms + 1e-3);

        let first = (start_ms / FRAME_MS).floor() as usize;
        let last = ((end_ms / FRAME_MS).ceil() as usize).max(first + 1); // exclusive
        let mut sum = [0.0f64; SPEAKERS];
        let mut active_w = [0.0f64; SPEAKERS];
        let mut weight = 0.0f64;
        for f in first.max(self.first_frame)..last.min(self.frames_received()) {
            let (f_lo, f_hi) = (f as f32 * FRAME_MS, (f + 1) as f32 * FRAME_MS);
            let w = (end_ms.min(f_hi) - start_ms.max(f_lo)).max(0.0) as f64;
            if w <= 0.0 {
                continue;
            }
            let row = &self.frames[f - self.first_frame];
            for k in 0..SPEAKERS {
                sum[k] += w * row[k] as f64;
                if row[k] >= self.cfg.active_threshold {
                    active_w[k] += w;
                }
            }
            weight += w;
        }
        if weight <= 0.0 {
            return None;
        }

        let mut activity = [0.0f32; SPEAKERS];
        let mut active = [0.0f32; SPEAKERS];
        for k in 0..SPEAKERS {
            activity[k] = (sum[k] / weight) as f32;
            active[k] = (active_w[k] / weight) as f32;
        }
        let mut order: Vec<usize> = (0..SPEAKERS).collect();
        order.sort_by(|&a, &b| {
            active[b]
                .total_cmp(&active[a])
                .then(activity[b].total_cmp(&activity[a]))
                .then(a.cmp(&b))
        });
        let (best, second) = (order[0], order[1]);
        if active[best] < self.cfg.min_active {
            return None;
        }
        let total: f32 = active.iter().sum();
        let overlapping = (active[second] >= self.cfg.overlap_threshold * active[best]
            && active[second] >= self.cfg.min_active)
            .then_some(second);
        Some(SpeakerLabel {
            speaker: best,
            confidence: active[best] / total,
            overlapping,
            active,
            activity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` frames of one speaker talking (probability 0.95, the rest 0.02).
    fn solo(speaker: usize, n: usize) -> Vec<f32> {
        (0..n)
            .flat_map(|_| {
                let mut f = [0.02f32; SPEAKERS];
                f[speaker] = 0.95;
                f
            })
            .collect()
    }

    fn silence(n: usize) -> Vec<f32> {
        vec![0.01; n * SPEAKERS]
    }

    fn labeler() -> SpeakerLabeler {
        SpeakerLabeler::new(SpeakerLabelerConfig::default())
    }

    #[test]
    fn labels_the_speaker_active_over_the_span() {
        let mut l = labeler();
        l.push_frames(&[solo(0, 10), silence(5), solo(2, 10)].concat());
        let a = l.label(0.0, 800.0).unwrap();
        assert_eq!(a.speaker, 0);
        assert!(a.confidence > 0.99 && a.overlapping.is_none());
        assert!((a.active[0] - 1.0).abs() < 1e-6, "{:?}", a.active);
        assert_eq!(l.label(1200.0, 1999.0).unwrap().speaker, 2);
        // Silence stays unlabeled.
        assert_eq!(l.label(800.0, 1200.0), None);
    }

    #[test]
    fn a_speaker_hovering_below_the_threshold_is_not_a_label() {
        // The model sits near 0.35 for slot 0 before it has heard anyone. Mean probability is
        // above any sensible floor, but nobody is active.
        let mut l = labeler();
        l.push_frames(
            &(0..10)
                .flat_map(|_| [0.35, 0.0, 0.0, 0.0])
                .collect::<Vec<f32>>(),
        );
        assert_eq!(l.label(0.0, 800.0), None);
        // Once it crosses 0.5 for a fifth of the span it is.
        let mut l = labeler();
        let mixed: Vec<f32> = (0..10)
            .flat_map(|f| [if f < 2 { 0.9 } else { 0.35 }, 0.0, 0.0, 0.0])
            .collect();
        l.push_frames(&mixed);
        let label = l.label(0.0, 800.0).unwrap();
        assert!((label.active[0] - 0.2).abs() < 1e-6, "{:?}", label.active);
        assert!(label.activity[0] < 0.5);
    }

    #[test]
    fn a_span_straddling_a_change_goes_to_whoever_talks_longer() {
        let mut l = labeler();
        l.push_frames(&[solo(1, 10), solo(3, 10)].concat());
        // 800 ms is the boundary: 600 ms of slot 1, 300 ms of slot 3.
        let label = l.label(200.0, 1100.0).unwrap();
        assert_eq!(label.speaker, 1);
        // Edge frames count by how much of them the span covers: a span ending 10 ms into a
        // frame barely counts that frame.
        let edge = l.label(0.0, 810.0).unwrap();
        assert_eq!(edge.speaker, 1);
        assert!(edge.active[3] < 0.02, "{:?}", edge.active);
    }

    #[test]
    fn reports_overlapping_speech() {
        let mut l = labeler();
        let both: Vec<f32> = (0..10).flat_map(|_| [0.9, 0.8, 0.02, 0.02]).collect();
        l.push_frames(&both);
        let label = l.label(0.0, 800.0).unwrap();
        assert_eq!((label.speaker, label.overlapping), (0, Some(1)));
        // Two speakers active the whole time share the span: confidence is the winner's half.
        assert!((label.confidence - 0.5).abs() < 1e-6, "{label:?}");
        // A faint second speaker is not an overlap.
        let mut l = labeler();
        let faint: Vec<f32> = (0..10).flat_map(|_| [0.9, 0.3, 0.02, 0.02]).collect();
        l.push_frames(&faint);
        assert_eq!(l.label(0.0, 800.0).unwrap().overlapping, None);
    }

    #[test]
    fn an_overlap_must_itself_clear_the_min_active_floor() {
        // Slot 0 is active for 30% of the span, slot 1 for 15%: half the winner's, so it meets
        // the relative threshold, but 15% is below the 20% floor, so it is not reported.
        let mut l = labeler();
        let frames: Vec<f32> = (0..20)
            .flat_map(|f| {
                [
                    if f < 6 { 0.9 } else { 0.0 },
                    if (6..9).contains(&f) { 0.9 } else { 0.0 },
                    0.0,
                    0.0,
                ]
            })
            .collect();
        l.push_frames(&frames);
        let label = l.label(0.0, 1600.0).unwrap();
        assert_eq!(label.speaker, 0);
        assert!((label.active[0] - 0.3).abs() < 1e-6 && (label.active[1] - 0.15).abs() < 1e-6);
        assert_eq!(label.overlapping, None);
    }

    #[test]
    fn an_utterance_waits_until_its_whole_span_is_covered() {
        let mut l = labeler();
        l.add_utterance(7, 400.0, 1600.0);
        l.push_frames(&solo(1, 19)); // covers 1520 ms
        assert!(l.poll().is_empty(), "not covered yet");
        assert_eq!(l.pending(), 1);
        l.push_frames(&solo(1, 1)); // 1600 ms
        let done = l.poll();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].id, 7);
        assert_eq!(done[0].label.as_ref().unwrap().speaker, 1);
        assert_eq!(l.pending(), 0);
        assert!(l.poll().is_empty());
    }

    #[test]
    fn a_covered_utterance_is_not_held_up_by_an_earlier_uncovered_one() {
        let mut l = labeler();
        l.add_utterance(1, 0.0, 5000.0); // long, still waiting
        l.add_utterance(2, 400.0, 800.0);
        l.push_frames(&solo(0, 12)); // 960 ms
        let done = l.poll();
        assert_eq!(done.iter().map(|u| u.id).collect::<Vec<_>>(), vec![2]);
        assert_eq!(l.pending(), 1);
    }

    #[test]
    fn flush_labels_what_exists_and_releases_the_rest_unlabeled() {
        let mut l = labeler();
        l.add_utterance(1, 0.0, 700.0);
        l.add_utterance(2, 10_000.0, 11_000.0); // far beyond anything diarized
        l.push_frames(&solo(2, 10));
        let done = l.flush();
        assert_eq!(done.len(), 2);
        assert_eq!(done[0].label.as_ref().unwrap().speaker, 2);
        assert_eq!(done[1].label, None);
        assert_eq!(l.pending(), 0);
    }

    #[test]
    fn history_is_bounded_but_pending_utterances_keep_their_frames() {
        let cfg = SpeakerLabelerConfig {
            history_ms: 800.0, // 10 frames
            ..Default::default()
        };
        let mut l = SpeakerLabeler::new(cfg);
        l.add_utterance(1, 0.0, 400.0);
        l.push_frames(&solo(3, 100));
        // The pending utterance pins frame 0, so nothing was dropped yet.
        assert_eq!(l.frames.len(), 100);
        let done = l.poll();
        assert_eq!(done[0].label.as_ref().unwrap().speaker, 3);
        // With nothing pending, the next push trims to the window.
        l.push_frames(&solo(3, 1));
        assert_eq!(l.frames.len(), 10);
        assert_eq!(l.frames_received(), 101);
        // A span older than the window now has no frames.
        assert_eq!(l.label(0.0, 400.0), None);
        assert_eq!(l.label(7_500.0, 8_000.0).unwrap().speaker, 3);
    }

    #[test]
    fn a_stalled_diarizer_cannot_grow_the_queue() {
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            max_pending: 2,
            ..Default::default()
        });
        for id in 0..5 {
            l.add_utterance(id, id as f32 * 1000.0, id as f32 * 1000.0 + 500.0);
        }
        let done = l.poll();
        assert_eq!(done.iter().map(|u| u.id).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(done.iter().all(|u| u.label.is_none()));
        assert_eq!(l.pending(), 2);
    }

    #[test]
    fn degenerate_spans() {
        let mut l = labeler();
        l.push_frames(&solo(0, 5));
        // A zero-length span takes the frame it falls in.
        assert_eq!(l.label(160.0, 160.0).unwrap().speaker, 0);
        // Reversed or negative spans do not panic.
        assert_eq!(l.label(300.0, 100.0).unwrap().speaker, 0);
        assert_eq!(l.label(-50.0, 100.0).unwrap().speaker, 0);
        // Not received yet.
        assert_eq!(l.label(5_000.0, 6_000.0), None);
    }

    #[test]
    #[should_panic(expected = "k x 4")]
    fn rejects_ragged_frames() {
        labeler().push_frames(&[0.0; 5]);
    }
}

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
pub const FRAME_MS: f64 = 80.0;

/// Speaker slots the diarizer predicts.
pub const SPEAKERS: usize = crate::model::sortformer::MAX_SPEAKERS;

/// One stretch of one speaker slot's activity, from [`speaker_segments`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerSegment {
    /// The speaker slot (`0..4`).
    pub speaker: usize,
    /// Start, ms from the audio origin.
    pub start_ms: f64,
    /// End (exclusive), ms from the audio origin.
    pub end_ms: f64,
}

/// Turn per-frame speaker activities (`[k x 4]`, 80 ms frames, as the diarizer returns them)
/// into contiguous per-speaker segments, ordered by start time.
///
/// A slot is active in a frame when its probability reaches `threshold`. Runs separated by at
/// most `max_gap_frames` inactive frames are joined, and joined runs shorter than `min_frames`
/// are dropped. Speakers are independent, so segments of different speakers can overlap (that
/// is how overlapped speech shows up).
///
/// # Panics
/// If `frames.len()` is not a multiple of 4.
pub fn speaker_segments(
    frames: &[f32],
    threshold: f32,
    max_gap_frames: usize,
    min_frames: usize,
) -> Vec<SpeakerSegment> {
    assert_eq!(
        frames.len() % SPEAKERS,
        0,
        "frames must be [k x {SPEAKERS}] speaker activities"
    );
    let rows = frames.as_chunks::<SPEAKERS>().0;
    let mut out = Vec::new();
    for speaker in 0..SPEAKERS {
        // (first active frame, one past the last active frame) of the run being built.
        let mut run: Option<(usize, usize)> = None;
        let flush = |run: Option<(usize, usize)>, out: &mut Vec<SpeakerSegment>| {
            if let Some((a, b)) = run
                && b - a >= min_frames.max(1)
            {
                out.push(SpeakerSegment {
                    speaker,
                    start_ms: a as f64 * FRAME_MS,
                    end_ms: b as f64 * FRAME_MS,
                });
            }
        };
        for (f, row) in rows.iter().enumerate() {
            if row[speaker] >= threshold {
                run = match run {
                    Some((a, b)) if f - b <= max_gap_frames => Some((a, f + 1)),
                    prev => {
                        flush(prev, &mut out);
                        Some((f, f + 1))
                    }
                };
            }
        }
        flush(run, &mut out);
    }
    out.sort_by(|a, b| {
        a.start_ms
            .total_cmp(&b.start_ms)
            .then(a.speaker.cmp(&b.speaker))
    });
    out
}

/// How the label is decided and how long frames are kept. Thresholds are probabilities in
/// `[0, 1]`; `history_ms` should be at least a few diarizer chunks (a NaN, zero or negative value
/// keeps a one-frame history); `max_pending` of 0 releases every utterance at once, unlabeled.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerLabelerConfig {
    /// A speaker is active in a frame when their probability reaches this.
    pub active_threshold: f32,
    /// The winner must be active for at least this fraction of the span, or it is unlabeled.
    pub min_active: f32,
    /// A runner-up counts as overlapping speech when its active fraction is at least this
    /// fraction of the winner's.
    pub overlap_threshold: f32,
    /// Frames older than this (relative to the newest) are dropped, and an utterance whose end
    /// the diarizer has still not reached once its start is this old is released unlabeled (it
    /// would otherwise pin every frame since its start). An utterance older than the retained
    /// window when it is registered is returned unlabeled.
    pub history_ms: f64,
    /// Utterances waiting for the diarizer beyond this many are released unlabeled (with
    /// [`LabeledUtterance::dropped`] set), oldest first, so a stalled diarizer cannot grow the
    /// queue without bound.
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
    pub start_ms: f64,
    /// Utterance end, ms from the audio origin.
    pub end_ms: f64,
    /// `None` when the diarizer found no speaker active over the span (or had no frames for it).
    pub label: Option<SpeakerLabel>,
    /// The labeler gave up on this utterance without asking the diarizer: the queue overflowed
    /// (`max_pending`), the diarizer never reached its end within `history_ms`, or its times were
    /// not finite. A stalled diarizer therefore shows up as `dropped`, not as silence.
    pub dropped: bool,
}

struct Pending {
    id: u64,
    start_ms: f64,
    end_ms: f64,
}

/// Matches utterances with the diarizer's per-frame speaker activity. See the module docs.
pub struct SpeakerLabeler {
    cfg: SpeakerLabelerConfig,
    /// Frames from `first_frame` on.
    frames: VecDeque<[f32; SPEAKERS]>,
    /// Absolute index of `frames[0]`.
    first_frame: usize,
    pending: VecDeque<Pending>,
    /// Utterances given up on (see [`LabeledUtterance::dropped`]), handed out by the next poll.
    /// At most `max_pending` are parked; older ones are counted in `lost`.
    dropped: VecDeque<LabeledUtterance>,
    lost: u64,
}

impl SpeakerLabeler {
    /// A labeler with `cfg`.
    pub fn new(cfg: SpeakerLabelerConfig) -> Self {
        Self {
            cfg,
            frames: VecDeque::new(),
            first_frame: 0,
            pending: VecDeque::new(),
            dropped: VecDeque::new(),
            lost: 0,
        }
    }

    /// Frames received so far (80 ms each), including any already dropped from the history.
    pub fn frames_received(&self) -> usize {
        self.first_frame + self.frames.len()
    }

    /// Audio the diarizer has covered, in ms.
    pub fn covered_ms(&self) -> f64 {
        self.frames_received() as f64 * FRAME_MS
    }

    /// Utterances registered and not yet released.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Dropped utterances discarded unreported because the caller did not [`Self::poll`] often
    /// enough to take them (more than `max_pending` accumulated between polls). Zero if the
    /// caller polls after registering utterances.
    pub fn lost(&self) -> u64 {
        self.lost
    }

    fn park(&mut self, u: LabeledUtterance) {
        self.dropped.push_back(u);
        while self.dropped.len() > self.cfg.max_pending.max(1) {
            self.dropped.pop_front();
            self.lost += 1;
        }
    }

    /// Append the diarizer's next frames, `[k x 4]` row-major, in order (the output of
    /// `SortformerLive::push_audio` / `finish`).
    ///
    /// Poll between pushes: an utterance the diarizer has covered but nobody has polled still
    /// holds every frame since its start, so a caller that never polls keeps the whole history.
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
        // An utterance the diarizer still has not reached after a whole history window is stuck
        // (an end past the audio, a wrong clock): give up on it rather than let it pin frames.
        let covered = self.covered_ms();
        let horizon = covered - self.cfg.history_ms;
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].end_ms > covered && self.pending[i].start_ms < horizon {
                let Some(p) = self.pending.remove(i) else {
                    // Unreachable: `i < len` by the loop condition. Skip, do not panic.
                    i += 1;
                    continue;
                };
                let u = self.release(p, false);
                self.park(u);
            } else {
                i += 1;
            }
        }
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
    ///
    /// An utterance with a NaN or infinite time cannot be placed on the clock; it is released
    /// by the next poll unlabeled and `dropped`. The queue never holds more than `max_pending`:
    /// the oldest beyond that are released the same way. Call [`Self::poll`] after registering:
    /// released-but-unpolled utterances are themselves capped at `max_pending` (see
    /// [`Self::lost`]).
    pub fn add_utterance(&mut self, id: u64, start_ms: f64, end_ms: f64) {
        if !(start_ms.is_finite() && end_ms.is_finite()) {
            // Reported with the times the caller gave, not clamped ones.
            let u = self.release(
                Pending {
                    id,
                    start_ms,
                    end_ms,
                },
                false,
            );
            self.park(u);
            return;
        }
        self.pending.push_back(Pending {
            id,
            start_ms: start_ms.max(0.0),
            end_ms: end_ms.max(start_ms.max(0.0)),
        });
        while self.pending.len() > self.cfg.max_pending {
            let Some(p) = self.pending.pop_front() else {
                break;
            };
            let u = self.release(p, false);
            self.park(u);
        }
    }

    /// Utterances whose whole span the diarizer has covered, in registration order, labeled.
    /// Utterances the labeler gave up on come first, unlabeled and `dropped`.
    pub fn poll(&mut self) -> Vec<LabeledUtterance> {
        let mut out: Vec<LabeledUtterance> = std::mem::take(&mut self.dropped).into();
        // Utterances finish in order, but a long one can end after a shorter later one; release
        // every covered one, not only the head.
        let covered = self.covered_ms();
        let mut i = 0;
        while i < self.pending.len() {
            let p = &self.pending[i];
            // A zero-length utterance is a point that needs its own frame, which has not
            // arrived while `start == covered`.
            if p.end_ms <= covered && (p.end_ms > p.start_ms || p.start_ms < covered) {
                let Some(p) = self.pending.remove(i) else {
                    // Unreachable: `i < len` by the loop condition. Skip, do not panic.
                    i += 1;
                    continue;
                };
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
            dropped: !label,
        }
    }

    /// Who spoke over `[start_ms, end_ms)`, from the frames received so far. `None` if no frame
    /// overlaps the span (not yet received, or older than the history) or nobody reaches
    /// `min_active`.
    pub fn label(&self, start_ms: f64, end_ms: f64) -> Option<SpeakerLabel> {
        if start_ms.is_nan() || end_ms.is_nan() {
            return None;
        }
        let (start_ms, end_ms) = (start_ms.max(0.0), end_ms.max(start_ms.max(0.0)));
        // A zero-length span is a point: it takes the whole frame it falls in.
        let point = end_ms <= start_ms;

        let first = (start_ms / FRAME_MS).floor() as usize;
        let last = if point {
            first.saturating_add(1)
        } else {
            ((end_ms / FRAME_MS).ceil() as usize).max(first.saturating_add(1))
        }; // exclusive
        let mut sum = [0.0f64; SPEAKERS];
        let mut active_w = [0.0f64; SPEAKERS];
        let mut weight = 0.0f64;
        for f in first.max(self.first_frame)..last.min(self.frames_received()) {
            let (f_lo, f_hi) = (f as f64 * FRAME_MS, (f + 1) as f64 * FRAME_MS);
            let w = if point {
                FRAME_MS
            } else {
                (end_ms.min(f_hi) - start_ms.max(f_lo)).max(0.0)
            };
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
        let total: f32 = active.iter().sum();
        // `total` is 0 only if nobody is active; that is silence even with a `min_active` of 0.
        if active[best] < self.cfg.min_active || total <= 0.0 {
            return None;
        }
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
    fn equally_active_speakers_are_ordered_by_mean_probability() {
        // Slots 0 and 1 are both active for the whole span; slot 1 is the more confident, so it
        // wins the tie (the lower index would otherwise).
        let mut l = labeler();
        let frames: Vec<f32> = (0..10).flat_map(|_| [0.6, 0.9, 0.02, 0.02]).collect();
        l.push_frames(&frames);
        let label = l.label(0.0, 800.0).unwrap();
        assert_eq!((label.speaker, label.overlapping), (1, Some(0)));
    }

    #[test]
    fn a_zero_max_pending_still_hands_out_the_newest_dropped_utterance() {
        // `max_pending: 0` releases every utterance at once; the parked list keeps one, so the
        // newest comes out of the next poll and the others are counted as lost.
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            max_pending: 0,
            ..Default::default()
        });
        for id in 0..3 {
            l.add_utterance(id, 0.0, 400.0);
        }
        assert_eq!(l.pending(), 0);
        let done = l.poll();
        assert_eq!(done.iter().map(|u| u.id).collect::<Vec<_>>(), vec![2]);
        assert_eq!(l.lost(), 2);
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
            l.add_utterance(id, id as f64 * 1000.0, id as f64 * 1000.0 + 500.0);
        }
        // Three were released, but only `max_pending` (2) are parked for the poll; the oldest was
        // counted as lost.
        let done = l.poll();
        assert_eq!(done.iter().map(|u| u.id).collect::<Vec<_>>(), vec![1, 2]);
        assert!(done.iter().all(|u| u.label.is_none() && u.dropped));
        assert_eq!(l.lost(), 1);
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

    #[test]
    fn a_point_span_late_in_a_long_stream_still_takes_its_frame() {
        // f32 milliseconds lose a 1 ms nudge past about 33 s, which used to give a zero-length
        // span no weight at all.
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            history_ms: 1e12,
            ..Default::default()
        });
        l.push_frames(&solo(0, 46_000)); // about 61 minutes
        for at in [40_000.0, 100_000.0, 3_600_000.0] {
            assert_eq!(l.label(at, at).map(|x| x.speaker), Some(0), "at {at} ms");
        }
    }

    #[test]
    fn absurd_times_neither_panic_nor_label() {
        let mut l = labeler();
        l.push_frames(&solo(0, 10));
        for (a, b) in [
            (f64::INFINITY, f64::INFINITY),
            (1e30, 1e30),
            (f64::NAN, f64::NAN),
            (f64::MAX, f64::MAX),
        ] {
            assert_eq!(l.label(a, b), None, "{a}..{b}");
        }
        assert_eq!(l.label(f64::NAN, 400.0), None);
        assert_eq!(l.label(0.0, f64::NAN), None);
        // An infinite end reads as "to the last frame".
        assert_eq!(l.label(0.0, f64::INFINITY).unwrap().speaker, 0);
    }

    #[test]
    fn nobody_active_is_unlabeled_even_with_no_minimum() {
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            min_active: 0.0,
            ..Default::default()
        });
        l.push_frames(&[0.01; 40]);
        assert_eq!(l.label(0.0, 400.0), None);
    }

    #[test]
    fn utterances_with_unusable_times_are_released_dropped() {
        let mut l = labeler();
        l.add_utterance(1, f64::NAN, 100.0);
        l.add_utterance(2, 0.0, f64::INFINITY);
        l.add_utterance(3, 0.0, 400.0);
        assert_eq!(l.pending(), 1);
        l.push_frames(&solo(1, 10));
        let done = l.poll();
        assert_eq!(done.iter().map(|u| u.id).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            done.iter().map(|u| u.dropped).collect::<Vec<_>>(),
            vec![true, true, false]
        );
        assert!(done[2].label.is_some());
    }

    #[test]
    fn the_queue_is_bounded_at_registration_not_only_at_poll() {
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            max_pending: 3,
            ..Default::default()
        });
        for id in 0..1000 {
            l.add_utterance(id, 0.0, 1e9);
        }
        assert_eq!(l.pending(), 3, "no poll in between");
        // The released ones are parked for the next poll, but only `max_pending` of them: the
        // rest are counted, not kept, so an unpolled labeler stays bounded too.
        let done = l.poll();
        assert_eq!(done.len(), 3);
        assert!(done.iter().all(|u| u.dropped && u.label.is_none()));
        assert_eq!(
            done.iter().map(|u| u.id).collect::<Vec<_>>(),
            vec![994, 995, 996]
        );
        assert_eq!(l.lost(), 994);
    }

    #[test]
    fn flush_labels_a_partly_covered_utterance_from_the_frames_that_exist() {
        let mut l = labeler();
        l.add_utterance(1, 0.0, 700.0);
        l.push_frames(&solo(2, 5)); // 400 ms of the 700
        assert!(l.poll().is_empty());
        let done = l.flush();
        assert_eq!(done.len(), 1);
        assert!(!done[0].dropped);
        assert_eq!(done[0].label.as_ref().map(|x| x.speaker), Some(2));
    }

    #[test]
    fn a_point_utterance_waits_for_its_own_frame() {
        let mut l = labeler();
        l.push_frames(&solo(1, 10)); // covered to 800 ms
        l.add_utterance(1, 800.0, 800.0); // frame 10 has not arrived
        assert!(l.poll().is_empty(), "released before its frame exists");
        l.push_frames(&solo(1, 1));
        let done = l.poll();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].label.as_ref().map(|x| x.speaker), Some(1));
    }

    #[test]
    fn a_dropped_record_carries_the_times_the_caller_gave() {
        let mut l = labeler();
        l.add_utterance(1, f64::NAN, 5000.0);
        let done = l.poll();
        assert!(done[0].dropped && done[0].start_ms.is_nan() && done[0].end_ms == 5000.0);
    }

    #[test]
    fn an_utterance_the_diarizer_never_reaches_cannot_pin_the_history() {
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            history_ms: 10_000.0, // 125 frames
            ..Default::default()
        });
        l.add_utterance(0, 0.0, 1e9); // ends far past any audio
        let mut dropped = Vec::new();
        for _ in 0..2000 {
            l.push_frames(&solo(0, 1));
            dropped.extend(l.poll());
        }
        assert_eq!(dropped.len(), 1);
        assert!(dropped[0].dropped && dropped[0].id == 0);
        assert_eq!(l.pending(), 0);
        assert!(l.frames.len() <= 126, "{} frames held", l.frames.len());
        // A later utterance is unaffected.
        l.add_utterance(1, 150_000.0, 150_400.0);
        l.push_frames(&solo(0, 1));
        let done = l.poll();
        assert_eq!(done.len(), 1);
        assert!(!done[0].dropped && done[0].label.is_some());
    }

    #[test]
    fn normal_utterances_are_not_dropped_even_when_silent() {
        let mut l = labeler();
        l.add_utterance(1, 0.0, 400.0);
        l.push_frames(&silence(10));
        let done = l.poll();
        assert_eq!(done.len(), 1);
        assert!(done[0].label.is_none() && !done[0].dropped);
    }

    #[test]
    fn registration_clamps_a_negative_start_and_a_reversed_span() {
        let mut l = labeler();
        l.add_utterance(1, -50.0, 400.0); // a negative start reads as 0
        l.add_utterance(2, 5000.0, 1000.0); // reversed: reported as a point at its start
        l.push_frames(&solo(0, 65)); // covers 5200 ms
        let done = l.poll();
        let spans: Vec<_> = done.iter().map(|u| (u.id, u.start_ms, u.end_ms)).collect();
        assert_eq!(spans, vec![(1, 0.0, 400.0), (2, 5000.0, 5000.0)]);
    }

    #[test]
    fn a_non_positive_or_nan_history_keeps_one_frame() {
        for history_ms in [0.0, -5.0, f64::NAN] {
            let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
                history_ms,
                ..Default::default()
            });
            l.push_frames(&solo(0, 10));
            assert_eq!(l.frames.len(), 1, "history_ms {history_ms}");
        }
    }

    #[test]
    fn frame_ms_is_the_models_frame_hop() {
        use crate::model::audio_encoder::{HOP_LEN, SAMPLE_RATE};
        assert_eq!(FRAME_MS, (HOP_LEN * 8) as f64 * 1000.0 / SAMPLE_RATE as f64);
    }

    #[test]
    fn the_documented_thresholds_are_inclusive() {
        // "Reaches the threshold": a probability of exactly 0.5 is active.
        let mut l = labeler();
        l.push_frames(&[0.5, 0.0, 0.0, 0.0].repeat(10));
        assert_eq!(l.label(0.0, 800.0).unwrap().active[0], 1.0);
        // "At least": a runner-up at exactly half the winner's fraction (and exactly the floor)
        // overlaps. Slot 0 active in 4 of 10 frames, slot 1 in 2 of 10.
        let mut l = labeler();
        let frames: Vec<f32> = (0..10)
            .flat_map(|f| {
                [
                    if f < 4 { 0.9 } else { 0.0 },
                    if f < 2 { 0.9 } else { 0.0 },
                    0.0,
                    0.0,
                ]
            })
            .collect();
        l.push_frames(&frames);
        let label = l.label(0.0, 800.0).unwrap();
        assert_eq!(
            (label.speaker, label.overlapping),
            (0, Some(1)),
            "{label:?}"
        );
    }

    #[test]
    fn a_stuck_utterance_expires_only_once_strictly_older_than_the_history() {
        let mut l = SpeakerLabeler::new(SpeakerLabelerConfig {
            history_ms: 800.0, // 10 frames
            ..Default::default()
        });
        l.add_utterance(1, 0.0, 1e9);
        l.push_frames(&solo(0, 10)); // covered 800 ms: start 0 is exactly one history old
        assert!(l.poll().is_empty());
        assert_eq!(l.pending(), 1);
        l.push_frames(&solo(0, 1));
        let done = l.poll();
        assert_eq!(done.len(), 1);
        assert!(done[0].dropped);
    }

    fn seg(speaker: usize, a: f64, b: f64) -> SpeakerSegment {
        SpeakerSegment {
            speaker,
            start_ms: a,
            end_ms: b,
        }
    }

    #[test]
    fn segments_follow_each_speakers_activity() {
        let frames = [solo(0, 5), silence(2), solo(2, 3)].concat();
        assert_eq!(
            speaker_segments(&frames, 0.5, 0, 1),
            vec![seg(0, 0.0, 400.0), seg(2, 560.0, 800.0)]
        );
        assert!(speaker_segments(&[], 0.5, 0, 1).is_empty());
        assert!(speaker_segments(&silence(10), 0.5, 0, 1).is_empty());
    }

    #[test]
    fn short_gaps_are_joined_and_short_runs_dropped() {
        // Speaker 0 talks for 3 frames, pauses 2, talks 2 more; speaker 1 blips for 1 frame.
        let mut frames = Vec::new();
        for f in 0..8 {
            let active0 = !(3..5).contains(&f);
            let active1 = f == 6;
            frames.extend([
                if active0 { 0.9 } else { 0.0 },
                if active1 { 0.9 } else { 0.0 },
                0.0,
                0.0,
            ]);
        }
        // A 2-frame gap is not joined at max_gap 1, and is at max_gap 2.
        assert_eq!(
            speaker_segments(&frames, 0.5, 1, 2),
            vec![seg(0, 0.0, 240.0), seg(0, 400.0, 640.0)]
        );
        assert_eq!(
            speaker_segments(&frames, 0.5, 2, 2),
            vec![seg(0, 0.0, 640.0)]
        );
        // With a minimum of 1 frame the blip survives; overlapping speakers both appear.
        let all = speaker_segments(&frames, 0.5, 1, 1);
        assert!(all.contains(&seg(1, 480.0, 560.0)), "{all:?}");
    }

    #[test]
    fn a_probability_at_the_threshold_is_active() {
        let frames = [0.5, 0.0, 0.0, 0.0].repeat(2);
        assert_eq!(
            speaker_segments(&frames, 0.5, 0, 1),
            vec![seg(0, 0.0, 160.0)]
        );
    }
}

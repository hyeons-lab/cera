//! Live diarization paired with the speaker labeler: one object a background service drives.
//!
//! A service already has an utterance source (the VAD plus Whisper
//! [`AudioPipeline`](crate::AudioPipeline), whose `UtteranceTranscribed` events carry
//! `start_ms`/`end_ms`). Feed the same 16 kHz PCM to a [`LiveDiarizer`], register each utterance
//! as its text arrives, and poll: an utterance comes back, labeled with a speaker slot, once the
//! diarizer's predictions cover it (a chunk plus its lookahead later; see
//! [`SortformerLive::latency_frames`]). Both must count time from the same audio origin, so feed
//! the diarizer exactly the samples the pipeline sees, from the start.
//!
//! This sits beside the pipeline instead of inside it so the pipeline's event enum, and the
//! bindings generated from it, stay as they are.

use anyhow::Result;

use crate::model::sortformer::{SortformerLive, SortformerModel, StreamingParams};
use crate::speaker_labeler::{LabeledUtterance, SpeakerLabeler, SpeakerLabelerConfig};

/// A [`SortformerLive`] whose frames feed a [`SpeakerLabeler`].
pub struct LiveDiarizer {
    live: SortformerLive,
    labeler: SpeakerLabeler,
}

impl LiveDiarizer {
    /// Start a session with `params` (see [`SortformerModel::default_streaming`]) and a labeler
    /// configured by `cfg`.
    pub fn new(
        model: &SortformerModel,
        params: StreamingParams,
        cfg: SpeakerLabelerConfig,
    ) -> Result<Self> {
        Ok(Self {
            live: model.new_live(params)?,
            labeler: SpeakerLabeler::new(cfg),
        })
    }

    /// Feed mono 16 kHz PCM of any length (the same audio the transcriber hears). Returns the
    /// speaker activities that became final, `[k x 4]` at 80 ms per frame; they are already in
    /// the labeler. Errors as [`SortformerLive::push_audio`] does (nothing is consumed).
    pub fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let frames = self.live.push_audio(pcm)?;
        self.labeler.push_frames(&frames);
        Ok(frames)
    }

    /// Register an utterance whose text is ready, by its span on the audio clock.
    pub fn add_utterance(&mut self, id: u64, start_ms: f64, end_ms: f64) {
        self.labeler.add_utterance(id, start_ms, end_ms);
    }

    /// Utterances the diarizer has now covered, labeled (see [`SpeakerLabeler::poll`]).
    pub fn poll(&mut self) -> Vec<LabeledUtterance> {
        self.labeler.poll()
    }

    /// End of audio: flush the diarizer, then label every utterance still waiting with the
    /// frames that exist (see [`SpeakerLabeler::flush`]).
    pub fn finish(&mut self) -> Result<Vec<LabeledUtterance>> {
        let frames = self.live.finish()?;
        self.labeler.push_frames(&frames);
        Ok(self.labeler.flush())
    }

    /// Worst-case delay in 80 ms frames before a frame's activity is final.
    pub fn latency_frames(&self) -> usize {
        self.live.latency_frames()
    }

    /// The labeler, for [`SpeakerLabeler::label`] queries and its counters.
    pub fn labeler(&self) -> &SpeakerLabeler {
        &self.labeler
    }

    /// The underlying live diarizer.
    pub fn live(&self) -> &SortformerLive {
        &self.live
    }
}

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
//!
//! [`LiveDiarizer`] is the 4-speaker Sortformer path (80 ms frames);
//! [`Nemotron3LiveDiarizer`] is the Nemotron-3-Diarization path (8 speakers, 10 ms frames).

use anyhow::Result;

use crate::model::nemotron3_diarization::{
    FRAME_MS as NEMOTRON3_FRAME_MS, Nemotron3Live, Nemotron3Model,
    StreamingParams as Nemotron3StreamingParams,
};
use crate::model::sortformer::{SortformerLive, SortformerModel, StreamingParams};
use crate::speaker_labeler::{FRAME_MS, LabeledUtterance, SpeakerLabeler, SpeakerLabelerConfig};

/// A [`SortformerLive`] whose frames feed a [`SpeakerLabeler`].
pub struct LiveDiarizer {
    live: SortformerLive,
    labeler: SpeakerLabeler<4>,
}

impl LiveDiarizer {
    /// Start a session with `params` (see [`SortformerModel::default_streaming`]) and a labeler
    /// configured by `cfg`. The labeler runs at the model's 80 ms frame rate whatever `cfg`
    /// says: the frame rate is the model's, not the caller's.
    pub fn new(
        model: &SortformerModel,
        params: StreamingParams,
        cfg: SpeakerLabelerConfig,
    ) -> Result<Self> {
        let cfg = SpeakerLabelerConfig {
            frame_ms: FRAME_MS,
            ..cfg
        };
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
    pub fn poll(&mut self) -> Vec<LabeledUtterance<4>> {
        self.labeler.poll()
    }

    /// End of audio: flush the diarizer, then label every utterance still waiting with the
    /// frames that exist (see [`SpeakerLabeler::flush`]).
    pub fn finish(&mut self) -> Result<Vec<LabeledUtterance<4>>> {
        let (_, utterances) = self.finish_with_frames()?;
        Ok(utterances)
    }

    /// End of audio: flush the diarizer and return both the final frames and the final labeled utterances.
    pub fn finish_with_frames(&mut self) -> Result<(Vec<f32>, Vec<LabeledUtterance<4>>)> {
        let frames = self.live.finish()?;
        self.labeler.push_frames(&frames);
        let utterances = self.labeler.flush();
        Ok((frames, utterances))
    }

    /// Worst-case delay in 80 ms frames before a frame's activity is final.
    pub fn latency_frames(&self) -> usize {
        self.live.latency_frames()
    }

    /// The labeler, for [`SpeakerLabeler::label`] queries and its counters.
    pub fn labeler(&self) -> &SpeakerLabeler<4> {
        &self.labeler
    }

    /// The underlying live diarizer.
    pub fn live(&self) -> &SortformerLive {
        &self.live
    }
}

/// A [`Nemotron3Live`] whose frames feed a [`SpeakerLabeler`]: the 8-speaker,
/// 10 ms-frame Nemotron-3-Diarization path beside [`LiveDiarizer`].
pub struct Nemotron3LiveDiarizer {
    live: Nemotron3Live,
    labeler: SpeakerLabeler<8>,
}

impl Nemotron3LiveDiarizer {
    /// Start a session with `params` (see [`Nemotron3Model::default_streaming`]) and a
    /// labeler configured by `cfg`. The labeler runs at the model's 10 ms frame rate whatever
    /// `cfg` says: the frame rate is the model's, not the caller's.
    pub fn new(
        model: &Nemotron3Model,
        params: Nemotron3StreamingParams,
        cfg: SpeakerLabelerConfig,
    ) -> Result<Self> {
        let cfg = SpeakerLabelerConfig {
            frame_ms: NEMOTRON3_FRAME_MS,
            ..cfg
        };
        Ok(Self {
            live: model.new_live(params)?,
            labeler: SpeakerLabeler::new(cfg),
        })
    }

    /// Feed mono 16 kHz PCM of any length (the same audio the transcriber hears). Returns the
    /// speaker activities that became final, `[k x 8]` at 10 ms per frame; they are already in
    /// the labeler. Errors as [`Nemotron3Live::push_audio`] does (nothing is consumed).
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
    pub fn poll(&mut self) -> Vec<LabeledUtterance<8>> {
        self.labeler.poll()
    }

    /// End of audio: flush the diarizer, then label every utterance still waiting with the
    /// frames that exist (see [`SpeakerLabeler::flush`]).
    pub fn finish(&mut self) -> Result<Vec<LabeledUtterance<8>>> {
        let (_, utterances) = self.finish_with_frames()?;
        Ok(utterances)
    }

    /// End of audio: flush the diarizer and return both the final frames and the final labeled utterances.
    pub fn finish_with_frames(&mut self) -> Result<(Vec<f32>, Vec<LabeledUtterance<8>>)> {
        let frames = self.live.finish()?;
        self.labeler.push_frames(&frames);
        let utterances = self.labeler.flush();
        Ok((frames, utterances))
    }

    /// Worst-case delay in 80 ms encoder frames before a frame's activity is final.
    pub fn latency_frames(&self) -> usize {
        self.live.latency_frames()
    }

    /// The labeler, for [`SpeakerLabeler::label`] queries and its counters.
    pub fn labeler(&self) -> &SpeakerLabeler<8> {
        &self.labeler
    }

    /// The underlying live diarizer.
    pub fn live(&self) -> &Nemotron3Live {
        &self.live
    }
}

/// A live diarization session the audio pipeline can drive: [`LiveDiarizer`] (Sortformer, 4
/// slots) or [`Nemotron3LiveDiarizer`] (Nemotron-3, 8 slots). The pipeline's bookkeeping is
/// written once against this trait; only the model half (open, poll, finish) differs per
/// session type.
pub trait LiveDiarizerSession: Send {
    /// The model a session runs on.
    type Model: Send;
    /// The streaming parameters a session runs with.
    type Params: Clone + std::fmt::Debug + Send;
    /// One labeled utterance.
    type Labeled;
    /// Open a session, as [`LiveDiarizer::new`] does.
    fn open(model: &Self::Model, params: Self::Params, cfg: SpeakerLabelerConfig) -> Result<Self>
    where
        Self: Sized;
    /// Feed mono 16 kHz PCM; the frames that became final (already in the labeler).
    fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>>;
    /// Register an utterance by its span on the audio clock.
    fn add_utterance(&mut self, id: u64, start_ms: f64, end_ms: f64);
    /// Utterances covered so far, labeled.
    fn poll(&mut self) -> Vec<Self::Labeled>;
    /// End of audio: label everything waiting.
    fn finish(&mut self) -> Result<Vec<Self::Labeled>>;
}

impl LiveDiarizerSession for LiveDiarizer {
    type Model = SortformerModel;
    type Params = StreamingParams;
    type Labeled = LabeledUtterance<4>;

    fn open(model: &Self::Model, params: Self::Params, cfg: SpeakerLabelerConfig) -> Result<Self> {
        LiveDiarizer::new(model, params, cfg)
    }

    fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        LiveDiarizer::push_audio(self, pcm)
    }

    fn add_utterance(&mut self, id: u64, start_ms: f64, end_ms: f64) {
        LiveDiarizer::add_utterance(self, id, start_ms, end_ms);
    }

    fn poll(&mut self) -> Vec<Self::Labeled> {
        LiveDiarizer::poll(self)
    }

    fn finish(&mut self) -> Result<Vec<Self::Labeled>> {
        LiveDiarizer::finish(self)
    }
}

impl LiveDiarizerSession for Nemotron3LiveDiarizer {
    type Model = Nemotron3Model;
    type Params = Nemotron3StreamingParams;
    type Labeled = LabeledUtterance<8>;

    fn open(model: &Self::Model, params: Self::Params, cfg: SpeakerLabelerConfig) -> Result<Self> {
        Nemotron3LiveDiarizer::new(model, params, cfg)
    }

    fn push_audio(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        Nemotron3LiveDiarizer::push_audio(self, pcm)
    }

    fn add_utterance(&mut self, id: u64, start_ms: f64, end_ms: f64) {
        Nemotron3LiveDiarizer::add_utterance(self, id, start_ms, end_ms);
    }

    fn poll(&mut self) -> Vec<Self::Labeled> {
        Nemotron3LiveDiarizer::poll(self)
    }

    fn finish(&mut self) -> Result<Vec<Self::Labeled>> {
        Nemotron3LiveDiarizer::finish(self)
    }
}

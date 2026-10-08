//! Decisions with the open `d1-omni` models.
//!
//! A d1 model answers named questions about a state with zero output tokens: every answer is
//! read from the model's distribution over the options. The state is text, optionally with
//! images or a speech clip; each question is rendered as its own sequence ([`prompt`]), the
//! bidirectional LFM2 trunk turns it into one hidden state per token, and the head ([`head`])
//! scores the state at the option markers. Images ([`vision`]) and speech ([`audio`]) become a
//! prefix of embeddings in front of the text.
//!
//! The trunk is an ordinary non-causal `lfm2` model, so it runs wherever the engine does: this
//! module only adds what sits on top of it, reading the `d1.*` tensors and settings of a GGUF
//! made by [`crate::convert::d1`].
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use cera::d1::{D1Model, D1Request};
//! # let (gguf, session): (std::sync::Arc<cera::gguf::GgufFile>, cera::Session) = todo!();
//! let tokenizer = cera::tokenizer::BpeTokenizer::from_gguf(&gguf)?;
//! let model = D1Model::from_gguf(&gguf, &tokenizer)?;
//! let request = D1Request::parse(
//!     r#"{"state": "I was charged twice.",
//!         "questions": {"refund": {"type": "noul", "instructions": "Is this a refund request?"}}}"#,
//! )?;
//! # let mut session = session;
//! let response = model.answer(&mut session, &tokenizer, &request)?;
//! println!("{}", response.to_json().dumps());
//! # Ok(()) }
//! ```

pub mod audio;
pub mod head;
pub mod json;
pub mod prompt;
pub mod vision;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use crate::engine::BackendPreference;
use crate::gguf::GgufFile;
use crate::session::Session;
use crate::tokenizer::BpeTokenizer;

pub use json::Json;
pub use prompt::{Question, QuestionType};

/// A request: a state and the named questions to answer about it, in order.
#[derive(Debug, Clone)]
pub struct D1Request {
    /// The state; `Json::Null` when the request has none (the media are the whole state).
    pub state: Json,
    pub questions: Vec<(String, Question)>,
    /// Encoded images (PNG or JPEG), in the order they are laid out ahead of the text.
    pub images: Vec<Vec<u8>>,
    /// An encoded speech clip (WAV). A request carries images or speech, not both.
    pub audio: Option<Vec<u8>>,
}

impl D1Request {
    /// Parse `{"state": <any JSON>, "questions": {name: {"type", "instructions", "criteria"}},
    /// "images": [<base64 data URL>, ...], "audio": <base64 data URL of a WAV>}`. A missing or
    /// null `state` is no state: the empty string, or `{}` after speech, as the model was
    /// trained.
    ///
    /// # Errors
    ///
    /// Fails on malformed JSON, no questions, or a question the model cannot answer.
    pub fn parse(text: &str) -> Result<Self> {
        let body = Json::parse(text)?;
        let state = body.get("state").cloned().unwrap_or(Json::Null);
        let Some(Json::Object(entries)) = body.get("questions") else {
            anyhow::bail!("\"questions\" must be an object of named questions");
        };
        ensure!(!entries.is_empty(), "\"questions\" must not be empty");
        let questions = entries
            .iter()
            .map(|(name, q)| {
                Question::from_json(q)
                    .map(|q| (name.clone(), q))
                    .with_context(|| format!("question `{name}`"))
            })
            .collect::<Result<_>>()?;
        let images = match body.get("images") {
            None | Some(Json::Null) => Vec::new(),
            Some(Json::Array(urls)) => urls
                .iter()
                .enumerate()
                .map(|(i, url)| {
                    let url = url
                        .as_str()
                        .with_context(|| format!("images[{i}] must be a base64 data URL"))?;
                    vision::decode_data_url(url).with_context(|| format!("images[{i}]"))
                })
                .collect::<Result<_>>()?,
            Some(_) => anyhow::bail!("\"images\" must be an array of base64 data URLs"),
        };
        let audio = match body.get("audio") {
            None | Some(Json::Null) => None,
            Some(url) => {
                let url = url
                    .as_str()
                    .context("\"audio\" must be a base64 data URL of a WAV file")?;
                Some(vision::decode_data_url(url).context("audio")?)
            }
        };
        ensure!(
            images.is_empty() || audio.is_none(),
            "a request carries images or audio, not both"
        );
        Ok(Self {
            state,
            questions,
            images,
            audio,
        })
    }

    /// Add an encoded image after the ones already there.
    pub fn with_image(mut self, bytes: Vec<u8>) -> Self {
        self.images.push(bytes);
        self
    }

    /// Set the encoded speech clip (WAV).
    pub fn with_audio(mut self, bytes: Vec<u8>) -> Self {
        self.audio = Some(bytes);
        self
    }
}

/// The answers to a request.
#[derive(Debug, Clone)]
pub struct D1Response {
    /// One answer object per question, in request order.
    pub answers: Vec<(String, Json)>,
    /// Every position the trunk read, summed over the questions.
    pub input_tokens: usize,
}

impl D1Response {
    /// `{"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}`.
    pub fn to_json(&self) -> Json {
        Json::Object(vec![
            ("answers".into(), Json::Object(self.answers.clone())),
            (
                "usage".into(),
                Json::Object(vec![
                    (
                        "input_tokens".into(),
                        Json::Int(self.input_tokens.to_string()),
                    ),
                    ("output_tokens".into(), Json::Int("0".into())),
                ]),
            ),
        ])
    }
}

/// The d1 half of a loaded model: the head, the calibration and the prompt's token ids.
pub struct D1Model {
    head: head::Head,
    vision: Option<vision::VisionTower>,
    temperatures: HashMap<String, f32>,
    ids: prompt::PromptTokens,
    max_length: usize,
    /// Room an image request leaves its text.
    image_text_length: usize,
    audio: Option<audio::AudioTower>,
    /// Room a speech request leaves its text.
    audio_text_length: usize,
}

impl D1Model {
    /// Whether `gguf` is a d1 model.
    pub fn is_d1(gguf: &GgufFile) -> bool {
        head::Head::is_present(gguf)
    }

    /// Read the head, the calibration and the special-token ids.
    ///
    /// # Errors
    ///
    /// Fails when `gguf` is not a d1 model or lacks a tensor or setting of one.
    pub fn from_gguf(gguf: &Arc<GgufFile>, tokenizer: &BpeTokenizer) -> Result<Self> {
        ensure!(Self::is_d1(gguf), "not a d1 model (no d1.head.block_count)");
        let n_embd = gguf
            .get_u32("lfm2.embedding_length")
            .context("missing lfm2.embedding_length")? as usize;
        let max_length = gguf
            .get_u32("d1.context_length")
            .context("missing d1.context_length")? as usize;
        ensure!(max_length >= 96, "d1 context is too small ({max_length})");
        let keys = gguf
            .get_string_array("d1.temperature.keys")
            .unwrap_or_default();
        let values = gguf
            .get_f32_array("d1.temperature.values")
            .unwrap_or_default();
        ensure!(
            keys.len() == values.len(),
            "d1.temperature.keys and d1.temperature.values differ in length"
        );
        let mut temperatures = HashMap::new();
        for (key, value) in keys.into_iter().zip(values) {
            ensure!(value > 0.0, "invalid d1 temperature {key} = {value}");
            temperatures.insert(key.to_string(), value);
        }
        let vision = vision::VisionTower::is_present(gguf)
            .then(|| vision::VisionTower::from_gguf(gguf, n_embd))
            .transpose()?;
        let image_text_length = gguf
            .get_u32("d1.image_text_length")
            .map_or(max_length, |n| n as usize);
        let audio = audio::AudioTower::is_present(gguf)
            .then(|| audio::AudioTower::from_gguf(gguf, n_embd))
            .transpose()?;
        let audio_text_length = gguf
            .get_u32("d1.audio_text_length")
            .map_or(max_length, |n| n as usize);
        Ok(Self {
            head: head::Head::from_gguf(gguf, n_embd)?,
            vision,
            temperatures,
            ids: prompt::PromptTokens::from_tokenizer(tokenizer)?,
            max_length,
            image_text_length,
            audio,
            audio_text_length,
        })
    }

    /// Run the vision and speech towers on the GPU `backend` names, where there is one. The
    /// trunk follows the session's own backend; this is for the encoders in front of it, which
    /// are most of the time of a request that carries media. Does nothing for the CPU, for a
    /// device that will not open, or for a tower the model lacks.
    pub fn accelerate(&mut self, gguf: &Arc<GgufFile>, backend: BackendPreference) {
        if let Some(vision) = &mut self.vision {
            vision.accelerate(gguf, backend);
        }
        if let Some(audio) = &mut self.audio {
            audio.accelerate(backend);
        }
    }

    /// The longest prompt, in tokens.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// The calibration temperature of a text question: its `type:bucket` entry, else its type's,
    /// else 1.
    fn temperature(&self, q: &Question) -> f32 {
        self.temperatures
            .get(&q.temperature_key())
            .or_else(|| self.temperatures.get(q.kind.name()))
            .copied()
            .unwrap_or(1.0)
    }

    /// Every question's probabilities over its options, in option order (`[yes, no]` for a
    /// `noul`), and the number of positions the trunk read.
    ///
    /// # Errors
    ///
    /// Fails when a question's options do not fit the context, or the trunk fails.
    pub fn probabilities(
        &self,
        session: &mut Session,
        tokenizer: &BpeTokenizer,
        request: &D1Request,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let hidden_size = session.hidden_size();
        ensure!(
            request.images.is_empty() || request.audio.is_none(),
            "a request carries images or audio, not both"
        );
        // the media are encoded once for all the questions
        let (media, prefix, text_cap) = if !request.images.is_empty() {
            let tower = self
                .vision
                .as_ref()
                .context("this model has no vision tower (reconvert it with images)")?;
            let rgb = request
                .images
                .iter()
                .enumerate()
                .map(|(i, bytes)| {
                    vision::decode_image(bytes).with_context(|| format!("images[{i}]"))
                })
                .collect::<Result<Vec<_>>>()?;
            (
                prompt::Media::Image,
                tower.encode(&rgb)?,
                self.image_text_length,
            )
        } else if let Some(bytes) = &request.audio {
            let tower = self
                .audio
                .as_ref()
                .context("this model has no speech tower (reconvert it with speech)")?;
            let (samples, rate) = audio::decode_wav(bytes).context("audio")?;
            let pcm = audio::prepare_clip(&samples, rate);
            (
                prompt::Media::Audio,
                tower.encode(&pcm)?,
                self.audio_text_length,
            )
        } else {
            (prompt::Media::None, Vec::new(), self.max_length)
        };
        let has_media = media != prompt::Media::None;
        let prefix_rows = prefix.len() / hidden_size;
        // with media the text gets what the media leave, and no more than the model was trained on
        let max_len = if has_media {
            let room = self.max_length.saturating_sub(prefix_rows);
            ensure!(
                room >= 64,
                "the media take {prefix_rows} of the {} positions; send less",
                self.max_length
            );
            room.min(text_cap)
        } else {
            self.max_length
        };
        // no state: nothing for text and images, and the empty object after speech
        let state = match (&request.state, media) {
            (Json::Null, prompt::Media::Audio) => Json::Object(Vec::new()),
            (Json::Null, _) => Json::Str(String::new()),
            (state, _) => state.clone(),
        };
        let mut all = Vec::with_capacity(request.questions.len());
        let mut read = 0;
        for (name, q) in &request.questions {
            let (ids, markers) = prompt::encode(tokenizer, &self.ids, &state, q, max_len, media)
                .with_context(|| format!("question `{name}`"))?;
            let hidden = if has_media {
                session.hidden_states_with_prefix(&prefix, &ids)?
            } else {
                session.hidden_states_for_tokens(&ids)?
            };
            let scores = self
                .head
                .scores(&hidden, ids.len(), q.kind as usize, &markers)
                .with_context(|| format!("question `{name}` (hidden size {hidden_size})"))?;
            // the calibration was fitted on text; media answers are the raw softmax
            let temperature = if has_media { 1.0 } else { self.temperature(q) };
            all.push(prompt::probabilities(q, &scores, temperature));
            read += prefix_rows + ids.len();
        }
        Ok((all, read))
    }

    /// Answer every question of `request`.
    ///
    /// # Errors
    ///
    /// As [`Self::probabilities`].
    pub fn answer(
        &self,
        session: &mut Session,
        tokenizer: &BpeTokenizer,
        request: &D1Request,
    ) -> Result<D1Response> {
        let (all, input_tokens) = self.probabilities(session, tokenizer, request)?;
        let answers = request
            .questions
            .iter()
            .zip(&all)
            .map(|((name, q), p)| (name.clone(), prompt::answer(q, p)))
            .collect();
        Ok(D1Response {
            answers,
            input_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_keeps_its_question_order_and_defaults_the_state() {
        let r = D1Request::parse(
            r#"{"questions": {"z": {"type": "noul", "instructions": "a"},
                              "a": {"type": "noul", "instructions": "b"}}}"#,
        )
        .unwrap();
        assert_eq!(r.state, Json::Null);
        let names: Vec<_> = r.questions.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["z", "a"]);
    }

    #[test]
    fn a_request_without_questions_is_refused() {
        for bad in [
            r#"{"state": "x"}"#,
            r#"{"state": "x", "questions": {}}"#,
            r#"{"state": "x", "questions": [1]}"#,
            r#"{"state": "x", "questions": {"q": {"type": "choice", "instructions": "i"}}}"#,
        ] {
            assert!(D1Request::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_response_is_the_reference_shape() {
        let r = D1Response {
            answers: vec![(
                "q".into(),
                Json::Object(vec![("type".into(), Json::Str("noul".into()))]),
            )],
            input_tokens: 42,
        };
        assert_eq!(
            r.to_json().dumps(),
            r#"{"answers": {"q": {"type": "noul"}}, "usage": {"input_tokens": 42, "output_tokens": 0}}"#
        );
    }
}

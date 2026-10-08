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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use crate::engine::BackendPreference;
use crate::gguf::GgufFile;
use crate::session::Session;
use crate::tokenizer::BpeTokenizer;

pub use json::Json;
pub use prompt::{Question, QuestionType};

/// Read a u32 setting as a `usize`, failing closed when it is missing.
pub(crate) fn setting_u32(gguf: &GgufFile, key: &str) -> Result<usize> {
    gguf.get_u32(key)
        .map(|v| v as usize)
        .with_context(|| format!("missing {key}"))
}

/// Multiply file-declared counts without wrapping: on 32-bit targets the plain
/// product reads back small and admits a corrupt load that panics later in
/// inference instead of refusing here.
pub(crate) fn checked_len(name: &str, factors: &[usize]) -> Result<usize> {
    let mut acc = 1usize;
    for &f in factors {
        acc = acc.checked_mul(f).with_context(|| {
            format!("`{name}` counts {factors:?}, more values than fit in usize")
        })?;
    }
    Ok(acc)
}

/// Read a `d1.*` tensor as f32, failing closed when it is missing or holds the wrong
/// number of values. `owner` names the loader for the missing-tensor message.
pub(crate) fn load_f32(
    gguf: &GgufFile,
    owner: &str,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>> {
    let values = gguf
        .get_tensor(name)
        .with_context(|| format!("the {owner} needs the tensor `{name}`"))?
        .to_f32_vec();
    ensure!(
        values.len() == elements,
        "`{name}` has {} values, expected {elements}",
        values.len()
    );
    Ok(values)
}

/// The exact (erf) GELU, `torch.nn.GELU()`'s default.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// The exact GELU over a slice, shared by the head and the vision projector rather than
/// [`crate::backend::cpu::gelu_erf_inplace`], which is the coarser Abramowitz & Stegun
/// 7.1.26 form `libm_erf` exists to beat.
pub(crate) fn gelu_exact_inplace(h: &mut [f32]) {
    for v in h.iter_mut() {
        *v = gelu(*v);
    }
}

/// `erf` to f32 precision (Abramowitz & Stegun 7.1.26 is too coarse for parity; this is the
/// W. J. Cody rational approximation as used by the C library, evaluated in f64).
fn libm_erf(x: f32) -> f32 {
    erf64(f64::from(x)) as f32
}

fn erf64(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1)) converges for all x but
    // loses precision for large |x|; use the continued fraction of erfc there.
    let ax = x.abs();
    let value = if ax == 0.0 {
        0.0
    } else if ax < 2.5 {
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        let mut n = 0.0f64;
        loop {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else if ax > 6.0 {
        1.0
    } else {
        // erfc(x) = exp(-x^2) / (x sqrt(pi)) * 1 / (1 + 1/(2x^2) / (1 + 2/(2x^2) / (1 + ...)))
        let x2 = ax * ax;
        let mut frac = 0.0f64;
        for k in (1..=60).rev() {
            frac = (k as f64 / 2.0) / (ax + frac);
        }
        let erfc = (-x2).exp() / ((ax + frac) * std::f64::consts::PI.sqrt());
        1.0 - erfc
    };
    if x < 0.0 { -value } else { value }
}

/// Softmax in place with exact `.exp()`, matching the PyTorch reference the d1 modules
/// parity-track. Deliberately not [`crate::backend::cpu::softmax_inplace`], which uses
/// ggml's polynomial exp approximation. Callers pass a non-empty slice; an empty one
/// is a no-op (both loops no-op, so no division ever runs).
pub(crate) fn softmax_exact(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
}

/// A request: a state and the named questions to answer about it, in order.
#[derive(Debug, Clone)]
pub struct D1Request {
    /// The state; `Json::Null` when the request has none (the media are the whole state).
    pub state: Json,
    pub questions: Vec<(String, Question)>,
    /// Encoded images (PNG or JPEG), in the order they are laid out ahead of the text.
    pub images: Vec<Vec<u8>>,
    /// An encoded speech clip (WAV). A request carries images or audio, not both.
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
    /// Fails on malformed JSON, no questions, a question the model cannot answer,
    /// or a request that carries images and audio together.
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
        // the combo refusal runs on key presence, before either payload decodes:
        // a both-media body reports the combo error, never a base64 error, and
        // no refused byte is decoded (presence matches the decoded check exactly:
        // each array entry yields one image, each non-null audio yields the clip)
        let images_present =
            matches!(body.get("images"), Some(Json::Array(urls)) if !urls.is_empty());
        let audio_present = !matches!(body.get("audio"), None | Some(Json::Null));
        ensure!(
            Self::media_combo_ok(images_present, audio_present),
            "{}",
            Self::MEDIA_COMBO_ERR
        );
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
            Self::media_combo_ok(!images.is_empty(), audio.is_some()),
            "{}",
            Self::MEDIA_COMBO_ERR
        );
        Ok(Self {
            state,
            questions,
            images,
            audio,
        })
    }

    /// Add an encoded image after the ones already there. A request carries images or
    /// audio, not both: combining this with [`Self::with_audio`] is refused by
    /// [`D1Model::probabilities`].
    pub fn with_image(mut self, bytes: Vec<u8>) -> Self {
        self.images.push(bytes);
        self
    }

    /// Set the encoded speech clip (WAV). A request carries images or audio, not both:
    /// combining this with [`Self::with_image`] is refused by [`D1Model::probabilities`].
    pub fn with_audio(mut self, bytes: Vec<u8>) -> Self {
        self.audio = Some(bytes);
        self
    }

    /// Whether media sources combine: images or audio, not both. The single
    /// spelling of the rule: [`Self::parse`] and [`D1Model::probabilities`]
    /// call it on one request, the CLI `Decide` arm on flags-or-body presence.
    pub fn media_combo_ok(has_images: bool, has_audio: bool) -> bool {
        !has_images || !has_audio
    }

    /// The refusal when sources do not combine, spelled once for [`Self::parse`],
    /// [`D1Model::probabilities`], and the CLI `Decide` arm.
    pub const MEDIA_COMBO_ERR: &str = "a request carries images or audio, not both";
}

/// Why a backend cannot serve d1 questions, spelled once for the
/// [`D1Model::probabilities`] gates and the CLI `Decide` fail-early.
pub const HIDDEN_STATES_REASON: &str = "cannot read hidden states";
/// Why a causal-only backend cannot serve d1 questions, spelled once for the
/// [`D1Model::probabilities`] gate and the CLI `Decide` fail-early.
pub const BIDIRECTIONAL_STATES_REASON: &str = "reads causal hidden states only";
/// Why a backend without the media path cannot serve media questions, spelled
/// once for the [`D1Model::probabilities`] gate and the CLI `Decide` fail-early.
pub const MEDIA_PREFIX_REASON: &str = "cannot read a media prefix";

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
    /// Trunk width (`lfm2.embedding_length`); the answering session must match it.
    n_embd: usize,
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

/// Text positions that must survive the media prefix; below this the request is refused.
const MIN_TEXT_ROOM: usize = 64;

/// Refuse a media prefix that leaves no room for text.
fn ensure_media_fits(max_length: usize, rows: usize) -> Result<()> {
    ensure!(
        max_length.saturating_sub(rows) >= MIN_TEXT_ROOM,
        "the media take {rows} of the {max_length} positions; send less"
    );
    Ok(())
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
    /// Fails when `gguf` is not a d1 model, the tokenizer lacks a d1 delimiter, its
    /// trunk is causal (`lfm2.is_causal`), its embedding width is 0, its context is
    /// below the prompt minimum, its temperature table is malformed, its media text
    /// room is 0, or it lacks a tensor or setting of one.
    pub fn from_gguf(gguf: &Arc<GgufFile>, tokenizer: &BpeTokenizer) -> Result<Self> {
        ensure!(Self::is_d1(gguf), "not a d1 model (no d1.head.block_count)");
        // the cheapest refusal first: seven hash lookups, before any weight load
        let ids = prompt::PromptTokens::from_tokenizer(tokenizer)?;
        ensure!(
            !crate::model::lfm2::gguf_is_causal(gguf, "lfm2"),
            "a d1 decision model needs a bidirectional trunk (lfm2.is_causal)"
        );
        let n_embd = setting_u32(gguf, "lfm2.embedding_length")?;
        ensure!(n_embd >= 1, "lfm2.embedding_length is 0");
        let max_length = setting_u32(gguf, "d1.context_length")?;
        ensure!(
            max_length >= prompt::MIN_OPTION_BUDGET,
            "d1 context is too small ({max_length}, need {})",
            prompt::MIN_OPTION_BUDGET
        );
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
            ensure!(
                value.is_finite() && value > 0.0,
                "invalid d1 temperature {key} = {value}"
            );
            temperatures.insert(key.to_string(), value);
        }
        // both rooms before either tower: they need only `max_length`, and a
        // scalar refusal must not pay a tower load first
        let image_text_length = gguf
            .get_u32("d1.image_text_length")
            .map_or(max_length, |n| n as usize);
        // a zero room would load, then refuse every media question with a
        // prompt-blaming message; refuse the file instead
        ensure!(image_text_length >= 1, "d1.image_text_length is 0");
        let audio_text_length = gguf
            .get_u32("d1.audio_text_length")
            .map_or(max_length, |n| n as usize);
        ensure!(audio_text_length >= 1, "d1.audio_text_length is 0");
        // the head before either tower: its scalar guards are hash lookups, and
        // a broken head must not pay a tower load first (same rule as the rooms)
        let head = head::Head::from_gguf(gguf, n_embd)?;
        let vision = vision::VisionTower::is_present(gguf)
            .then(|| vision::VisionTower::from_gguf(gguf, n_embd))
            .transpose()?;
        let audio = audio::AudioTower::is_present(gguf)
            .then(|| audio::AudioTower::from_gguf(gguf, n_embd))
            .transpose()?;
        Ok(Self {
            head,
            n_embd,
            vision,
            temperatures,
            ids,
            max_length,
            image_text_length,
            audio,
            audio_text_length,
        })
    }

    /// Run the decision head and the vision and speech towers on the GPU `backend` names, where
    /// there is one. The trunk follows the session's own backend; this is for the encoders in
    /// front of it, which are most of the time of a request that carries media, and the head
    /// behind it, which is most of the time of a long text prompt. Does nothing for the CPU, for a
    /// device that will not open, or for a tower the model lacks.
    pub fn accelerate(&mut self, gguf: &Arc<GgufFile>, backend: BackendPreference) {
        self.head.accelerate(gguf, backend);
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
    /// Cost: one trunk forward per question, so N questions cost N full-model reads.
    /// The media prefix itself is encoded once for all the questions.
    ///
    /// # Errors
    ///
    /// Fails on an empty or duplicate-named question list, a request carrying images
    /// and audio together, a session that is not this model's bidirectional trunk, a
    /// backend without the needed hidden-states path, an adapter attached to the session
    /// on a media question, a media question the model has no tower for, media bytes
    /// that do not decode, media that leave no room for text, a question's options that
    /// do not fit the context, or a trunk failure.
    pub fn probabilities(
        &self,
        session: &mut Session,
        tokenizer: &BpeTokenizer,
        request: &D1Request,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let hidden_size = session.hidden_size();
        ensure!(
            !request.questions.is_empty(),
            "\"questions\" must not be empty"
        );
        let mut seen = HashSet::with_capacity(request.questions.len());
        for (name, _) in &request.questions {
            ensure!(seen.insert(name), "duplicate question `{name}`");
        }
        ensure!(
            hidden_size == self.n_embd,
            "the session is {hidden_size} wide, the d1 model {}",
            self.n_embd
        );
        ensure!(
            !session.model().config().is_causal,
            "a d1 decision model needs a bidirectional trunk (this session is causal)"
        );
        ensure!(
            session.model().supports_hidden_states(),
            "this backend {HIDDEN_STATES_REASON}"
        );
        // the trunk config is bidirectional (checked above), but a causal-only
        // extractor would still blind every row to the future: refuse, not mis-score
        ensure!(
            session.model().supports_bidirectional_hidden_states(),
            "this backend {BIDIRECTIONAL_STATES_REASON} (d1 questions need cpu)"
        );
        ensure!(
            D1Request::media_combo_ok(!request.images.is_empty(), request.audio.is_some()),
            "{}",
            D1Request::MEDIA_COMBO_ERR
        );
        let wants_media = !request.images.is_empty() || request.audio.is_some();
        if wants_media {
            ensure!(
                session.model().supports_media_prefix(),
                "this backend {MEDIA_PREFIX_REASON} (media questions need cpu)"
            );
            // the per-question media path refuses an attached adapter, so probe it
            // here: a full tower forward is too late to discover it
            ensure!(
                !session.has_lora_adapters(),
                "detach the adapter for media questions (media-prefix hidden states have no LoRA apply hooks)"
            );
        }
        // the trunk refuses past max_seq_len anyway (ContextOverflow), so the
        // media budget uses the readable context: a hostile d1.context_length
        // cannot switch the budget off
        let budget = self.max_length.min(session.model().config().max_seq_len);
        // the media are encoded once for all the questions, but only after the budget
        // says they fit: a full tower forward is too late to discover they do not
        let (media, prefix, text_cap) = if !request.images.is_empty() {
            let tower = self
                .vision
                .as_ref()
                .context("this model has no vision tower (reconvert it with images)")?;
            // budget from the header dims, then decode, encode, and drop one image at
            // a time: rows add up, so a refusing prefix implies a refusing total, the
            // tail is never decoded, and only one image's pixels are ever resident
            let mut prefix = Vec::new();
            let mut predicted: usize = 0;
            for (i, bytes) in request.images.iter().enumerate() {
                let (width, height) =
                    vision::image_dims(bytes).with_context(|| format!("images[{i}]"))?;
                let rows = vision::VisionTower::prefix_rows(&[(width, height)])
                    .with_context(|| format!("images[{i}]"))?;
                predicted = predicted.saturating_add(rows);
                ensure_media_fits(budget, predicted)?;
                // the tower returns exactly rows * width floats per the agreement
                // below, so reserve this image's share up front
                prefix.reserve(rows.saturating_mul(hidden_size));
                let r = vision::decode_image(bytes).with_context(|| format!("images[{i}]"))?;
                // the decoder reports the header dims the budget counted above
                ensure!(
                    (r.width, r.height) == (width, height),
                    "decoded image[{i}] dimensions ({}, {}) do not match header ({width}, {height})",
                    r.width,
                    r.height
                );
                prefix.extend(tower.encode(std::slice::from_ref(&r))?);
            }
            ensure!(
                prefix.len() / hidden_size == predicted,
                "the vision tower returned {} rows for {} images, expected {predicted}",
                prefix.len() / hidden_size,
                request.images.len()
            );
            (prompt::Media::Image, prefix, self.image_text_length)
        } else if let Some(bytes) = &request.audio {
            let tower = self
                .audio
                .as_ref()
                .context("this model has no speech tower (reconvert it with speech)")?;
            let (samples, rate) =
                audio::decode_wav_capped(bytes, audio::MAX_SECONDS).context("audio")?;
            let pcm = audio::prepare_clip(&samples, rate)?;
            let predicted = audio::prefix_rows(pcm.len());
            ensure_media_fits(budget, predicted)?;
            let prefix = tower.encode(&pcm)?;
            // debug-only: `encode` above already refuses a row-count disagreement, so
            // this pins the plumbing; the vision path needs its `ensure!` because its
            // tower does no self-check.
            debug_assert_eq!(prefix.len() / hidden_size, predicted);
            (prompt::Media::Audio, prefix, self.audio_text_length)
        } else {
            (prompt::Media::None, Vec::new(), self.max_length)
        };
        let has_media = media != prompt::Media::None;
        let prefix_rows = prefix.len() / hidden_size;
        // with media the text gets what the media leave, and no more than the model was trained on
        let max_len = if has_media {
            ensure_media_fits(budget, prefix_rows)?;
            budget.saturating_sub(prefix_rows).min(text_cap)
        } else {
            budget
        };
        // no state: nothing for text and images, and the empty object after speech
        let state = match (&request.state, media) {
            (Json::Null, prompt::Media::Audio) => Json::Object(Vec::new()),
            (Json::Null, _) => Json::Str(String::new()),
            (state, _) => state.clone(),
        };
        let mut all = Vec::with_capacity(request.questions.len());
        let mut read: usize = 0;
        // the state tokenizes once; each question only takes the room its options leave
        let state_ids = prompt::encode_state(tokenizer, &self.ids, &state);
        for (name, q) in &request.questions {
            let (ids, markers) =
                prompt::encode_with_state_ids(tokenizer, &self.ids, &state_ids, q, max_len, media)
                    .with_context(|| format!("question `{name}`"))?;
            let hidden = if has_media {
                session
                    .hidden_states_with_prefix(&prefix, &ids)
                    .with_context(|| format!("question `{name}`"))?
            } else {
                session
                    .hidden_states_for_tokens(&ids)
                    .with_context(|| format!("question `{name}`"))?
            };
            let scores = self
                .head
                .scores(&hidden, ids.len(), q.kind.index(), &markers)
                .with_context(|| format!("question `{name}` (hidden size {hidden_size})"))?;
            // the calibration was fitted on text; media answers are the raw softmax
            let temperature = if has_media { 1.0 } else { self.temperature(q) };
            all.push(
                prompt::probabilities(q, &scores, temperature)
                    .with_context(|| format!("question `{name}`"))?,
            );
            read = read.saturating_add(prefix_rows).saturating_add(ids.len());
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
            .map(|((name, q), p)| {
                prompt::answer(q, p)
                    .map(|a| (name.clone(), a))
                    .with_context(|| format!("question `{name}`"))
            })
            .collect::<Result<_>>()?;
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
    fn checked_len_refuses_a_wrapping_product() {
        assert_eq!(checked_len("t", &[3, 4]).unwrap(), 12);
        assert_eq!(checked_len("t", &[7]).unwrap(), 7);
        let err = checked_len("t", &[usize::MAX, 2]).expect_err("expected a refusal");
        assert!(
            err.to_string().contains("more values than fit in usize"),
            "unexpected message: {err}"
        );
        // the message names the overflowing factors, not just the tensor
        assert!(
            err.to_string().contains(&format!("{:?}", [usize::MAX, 2])),
            "unexpected message: {err}"
        );
        // the reviewer's 32-bit case, scaled to this host: the wrapped count
        // reads back small on narrow targets and must refuse, not load short
        let half = usize::MAX / 65536 + 1;
        let err = checked_len("t", &[half, 65536]).expect_err("expected a refusal");
        assert!(
            err.to_string().contains("more values than fit in usize"),
            "unexpected message: {err}"
        );
    }

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
    fn a_request_carrying_images_and_audio_is_refused_at_parse() {
        // both media keys in the body: the parse guard refuses without building
        let err = D1Request::parse(
            r#"{"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}},
                "images": ["data:image/png;base64,AA=="], "audio": "data:audio/wav;base64,AA=="}"#,
        )
        .expect_err("expected the guard to refuse");
        assert!(err.to_string().contains("not both"), "{err}");
        // ... and the combo refusal precedes either decode: bad base64 in a
        // both-media body still reports the combo error, never the decode error
        let err = D1Request::parse(
            r#"{"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}},
                "images": ["!!!not-base64!!!"], "audio": "data:audio/wav;base64,AA=="}"#,
        )
        .expect_err("expected the guard to refuse");
        assert!(err.to_string().contains("not both"), "{err}");
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

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (-1.0, -0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (5.0, 0.999_999_999_998_462_5),
        ] {
            assert!((erf64(x) - want).abs() < 1e-13, "erf({x}) = {}", erf64(x));
        }
    }

    #[test]
    fn gelu_is_the_exact_form() {
        assert!(gelu(0.0).abs() < 1e-9);
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
        assert!((gelu(10.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_exact_normalizes() {
        let mut scores = vec![0.0f32, 3f32.ln()];
        softmax_exact(&mut scores);
        assert!((scores[0] - 0.25).abs() < 1e-6);
        assert!((scores[1] - 0.75).abs() < 1e-6);
        // the documented empty no-op: no division ever runs
        let mut empty: Vec<f32> = Vec::new();
        softmax_exact(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn softmax_exact_is_not_the_approximate_softmax() {
        // `x` puts the exact and ggml-polynomial paths 5 ULP apart, past any
        // cross-platform libm variance: a body swapped to `cpu::softmax_inplace`
        // lands bit-equal and fails here
        let x = f32::from_bits(0xBFF9_D22E);
        let mut exact = [x, 0.0];
        softmax_exact(&mut exact);
        let mut approx = [x, 0.0];
        crate::backend::cpu::softmax_inplace(&mut approx);
        assert_ne!(exact.map(f32::to_bits), approx.map(f32::to_bits));
    }

    #[test]
    fn the_media_budget_leaves_text_room_or_refuses() {
        assert_eq!(MIN_TEXT_ROOM, 64); // the policy value; the legs below pin the mechanics
        assert!(ensure_media_fits(2048, 0).is_ok());
        assert!(ensure_media_fits(2048, 2048 - MIN_TEXT_ROOM).is_ok());
        for rows in [2048 - MIN_TEXT_ROOM + 1, 2048, 2048 * 2] {
            let err = ensure_media_fits(2048, rows).unwrap_err();
            assert!(
                err.to_string().contains("send less"),
                "unexpected message: {err}"
            );
        }
    }

    /// A trunk with no weights: deterministic position-dependent rows, so the pipeline
    /// test pins the plumbing (prompt to trunk to head to softmax, token accounting)
    /// rather than any learned values. Marker sensitivity itself is pinned at the head
    /// layer (`scores_follow_the_marker_rows_not_the_first_rows`): the zero head here
    /// scores every row identically by construction.
    struct TrunkForTest {
        config: crate::model::ModelConfig,
        can_read_hidden: bool,
        can_read_bidirectional: bool,
        can_prefix_media: bool,
        // atomic: `Model` is `Send + Sync`, so a `RefCell` would not compile here
        seen_prefix_len: std::sync::atomic::AtomicUsize,
        seen_ids: std::sync::Mutex<Vec<Vec<u32>>>,
    }

    impl crate::model::Model for TrunkForTest {
        fn forward(
            &self,
            _tokens: &[u32],
            _pos: usize,
            _state: &mut crate::kv_cache::InferenceState,
        ) -> Vec<f32> {
            vec![0.0; self.config.vocab_size]
        }

        fn config(&self) -> &crate::model::ModelConfig {
            &self.config
        }

        fn supports_hidden_states(&self) -> bool {
            self.can_read_hidden
        }

        fn supports_bidirectional_hidden_states(&self) -> bool {
            self.can_read_bidirectional
        }

        fn supports_media_prefix(&self) -> bool {
            self.can_prefix_media
        }

        fn hidden_states(
            &self,
            tokens: &[u32],
            _state: &mut crate::kv_cache::InferenceState,
        ) -> Vec<f32> {
            // record the ids themselves, not just their count: the rows below are
            // position-derived, so a same-length id corruption would otherwise be
            // output-identical and no leg could catch it
            self.seen_ids.lock().unwrap().push(tokens.to_vec());
            let d = self.config.hidden_size;
            let mut out = vec![0.0; tokens.len() * d];
            // a position-varying slope, not just a shift: rows differing by a
            // uniform shift norm identically, so every option would score alike
            // and no question-type row could ever show through the softmax
            for (t, row) in out.chunks_exact_mut(d).enumerate() {
                let slope = (t % 5) as f32 + 1.0;
                for (c, v) in row.iter_mut().enumerate() {
                    *v = (t % 7) as f32 + c as f32 * 0.01 * slope;
                }
            }
            out
        }

        // the media success legs answer through this; the rows stay
        // position-derived, but the received prefix length is recorded so the legs
        // pin forwarding, not just the locally computed token count
        fn hidden_states_with_prefix(
            &self,
            prefix: &[f32],
            tokens: &[u32],
            state: &mut crate::kv_cache::InferenceState,
        ) -> Vec<f32> {
            self.seen_prefix_len
                .store(prefix.len(), std::sync::atomic::Ordering::Relaxed);
            self.hidden_states(tokens, state)
        }
    }

    fn trunk_config() -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            architecture: "d1-test".into(),
            n_layers: 0,
            hidden_size: 64,
            intermediate_size: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: 0,
            vocab_size: 256,
            max_seq_len: 4096,
            rope_theta: 0.0,
            rms_norm_eps: 0.0,
            block_types: Vec::new(),
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer: Vec::new(),
            scalars: crate::model::ScalarMultipliers::default(),
            moe: None,
            is_causal: false,
            class_labels: Vec::new(),
        }
    }

    fn test_session(
        config: crate::model::ModelConfig,
        can_read_hidden: bool,
        can_read_bidirectional: bool,
        can_prefix_media: bool,
    ) -> Session {
        test_session_with_trunk(
            config,
            can_read_hidden,
            can_read_bidirectional,
            can_prefix_media,
        )
        .0
    }

    // the media success legs keep the trunk to assert what prefix it received
    fn test_session_with_trunk(
        config: crate::model::ModelConfig,
        can_read_hidden: bool,
        can_read_bidirectional: bool,
        can_prefix_media: bool,
    ) -> (Session, std::sync::Arc<TrunkForTest>) {
        let trunk = std::sync::Arc::new(TrunkForTest {
            config,
            can_read_hidden,
            can_read_bidirectional,
            can_prefix_media,
            seen_prefix_len: std::sync::atomic::AtomicUsize::new(0),
            seen_ids: std::sync::Mutex::new(Vec::new()),
        });
        let session = Session::new(
            trunk.clone(),
            std::sync::Arc::new(BpeTokenizer::empty_for_test()),
            crate::session::ModalityCapabilities::text_only(),
            crate::session::SessionConfig::default(),
        )
        .unwrap();
        (session, trunk)
    }

    // the two repeated capability shapes, named so legs read as the session
    // they need instead of a bool triplet (the capability matrix below keeps
    // the explicit triple, where the per-row difference is the point)
    fn text_session(config: crate::model::ModelConfig) -> Session {
        test_session(config, true, true, false)
    }

    fn media_session(config: crate::model::ModelConfig) -> Session {
        test_session(config, true, true, true)
    }

    fn text_session_with_trunk(
        config: crate::model::ModelConfig,
    ) -> (Session, std::sync::Arc<TrunkForTest>) {
        test_session_with_trunk(config, true, true, false)
    }

    fn media_session_with_trunk(
        config: crate::model::ModelConfig,
    ) -> (Session, std::sync::Arc<TrunkForTest>) {
        test_session_with_trunk(config, true, true, true)
    }

    fn test_model() -> D1Model {
        let file = head::head_builder_default().build();
        D1Model {
            head: head::Head::from_gguf(&file, 64).unwrap(),
            n_embd: 64,
            vision: None,
            temperatures: HashMap::new(),
            ids: prompt::PromptTokens {
                bos: 1,
                marker: 2,
                state: 3,
                question: 4,
                option: 5,
                option_end: 6,
                decide: 7,
            },
            max_length: 2048,
            image_text_length: 896,
            audio: None,
            audio_text_length: 1536,
        }
    }

    /// A d1 GGUF with the standard test geometry plus trunk keys and one
    /// temperature; legs chain their own rooms (builder keys are last-wins).
    fn d1_file(context: u32) -> crate::gguf::GgufBuilder {
        head::head_builder_default()
            .kv_bool("lfm2.is_causal", false)
            .kv_u32("lfm2.embedding_length", 64)
            .kv_u32("d1.context_length", context)
            .kv_string_array("d1.temperature.keys", vec!["choice".to_string()])
            .kv_f32_array("d1.temperature.values", vec![1.5])
    }

    fn test_request() -> D1Request {
        D1Request::parse(
            r#"{"state": "the door is open",
                "questions": {"q": {"type": "noul", "instructions": "is it open"}}}"#,
        )
        .unwrap()
    }

    /// A request over a null state; the legs below differ by one field each.
    fn request_with(
        questions: Vec<(String, Question)>,
        images: Vec<Vec<u8>>,
        audio: Option<Vec<u8>>,
    ) -> D1Request {
        D1Request {
            state: Json::Null,
            questions,
            images,
            audio,
        }
    }

    /// A single-question request over the shared test question; the media legs
    /// below differ by their media only.
    fn request_with_media(images: Vec<Vec<u8>>, audio: Option<Vec<u8>>) -> D1Request {
        let request = test_request();
        let q = &request.questions[0].1;
        request_with([("q".to_string(), q.clone())].into(), images, audio)
    }

    #[test]
    fn a_synthetic_trunk_answers_with_normalized_probabilities() {
        let model = test_model();
        let (mut session, trunk) = text_session_with_trunk(trunk_config());
        let tok = BpeTokenizer::empty_for_test();
        let request = test_request();
        // the head is all zeros, so both options score the same and the pair is exactly even
        let (all, read) = model.probabilities(&mut session, &tok, &request).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], [0.5, 0.5]);
        assert!((all[0].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // the token count is the encoded prompt: nothing read twice, nothing skipped
        let (ids, _) = prompt::encode(
            &tok,
            &model.ids,
            &request.state,
            &request.questions[0].1,
            model.max_length,
            prompt::Media::None,
        )
        .unwrap();
        assert_eq!(read, ids.len());
        // the trunk received those exact ids, not just that many
        assert_eq!(*trunk.seen_ids.lock().unwrap(), vec![ids.clone()]);
        let response = model.answer(&mut session, &tok, &request).unwrap();
        assert_eq!(response.input_tokens, read);
        assert_eq!(
            response.answers[0].1,
            Json::Object(vec![
                ("type".into(), Json::Str("noul".into())),
                ("noul".into(), Json::Float(0.5)),
            ])
        );
    }

    #[test]
    fn the_pipeline_scores_each_type_from_its_own_row() {
        // a choice and a noul rendering identically: same instructions, same state,
        // and criteria spelling the same two options, so only `q.kind.index()`
        // can separate the answers
        let criteria = vec![
            ("false".to_string(), Json::Str("no".into())),
            ("true".to_string(), Json::Str("yes".into())),
        ];
        let request = request_with(
            vec![
                (
                    "c".to_string(),
                    prompt::Question {
                        kind: prompt::QuestionType::Choice,
                        instructions: "holds?".to_string(),
                        criteria: criteria.clone(),
                    },
                ),
                (
                    "n".to_string(),
                    prompt::Question {
                        kind: prompt::QuestionType::Noul,
                        instructions: "holds?".to_string(),
                        criteria,
                    },
                ),
            ],
            Vec::new(),
            None,
        );
        let tok = BpeTokenizer::empty_for_test();
        let mut model = test_model();
        // premise: identical rendering, so the row is the only separator
        let (ids_c, markers_c) = prompt::encode(
            &tok,
            &model.ids,
            &request.state,
            &request.questions[0].1,
            model.max_length,
            prompt::Media::None,
        )
        .unwrap();
        let (ids_n, markers_n) = prompt::encode(
            &tok,
            &model.ids,
            &request.state,
            &request.questions[1].1,
            model.max_length,
            prompt::Media::None,
        )
        .unwrap();
        assert_eq!(ids_c, ids_n);
        assert_eq!(markers_c, markers_n);
        let mut session = text_session(trunk_config());
        // premise: the marker rows differ in shape, so the row shift lands
        // differently per option instead of norming away into an even split
        let hidden = session.hidden_states_for_tokens(&ids_c).unwrap();
        let d = model.n_embd;
        assert_ne!(
            &hidden[markers_c[0] * d..(markers_c[0] + 1) * d],
            &hidden[markers_c[1] * d..(markers_c[1] + 1) * d]
        );
        // the typed head scores each row differently; a hardcoded row answers both alike
        let file = head::head_builder_typed().build();
        model.head = head::Head::from_gguf(&file, 64).unwrap();
        let (all, _) = model.probabilities(&mut session, &tok, &request).unwrap();
        assert_eq!(all.len(), 2);
        // before the noul reversal: `probabilities` reverses noul outputs, so a
        // naive cross-type compare would differ under a hardcoded row too
        let unreversed = vec![all[1][1], all[1][0]];
        assert_ne!(
            all[0], unreversed,
            "identical vectors: the type row is not wired through"
        );
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn a_fitting_image_answers_with_normalized_probabilities() {
        let mut model = test_model();
        model.vision = Some(
            vision::VisionTower::from_gguf(&vision::tower_builder_default().build(), 64).unwrap(),
        );
        let (mut session, trunk) = media_session_with_trunk(trunk_config());
        let tok = BpeTokenizer::empty_for_test();
        let request = test_request();
        let q = &request.questions[0].1;
        let with_image = request_with_media(vec![vision::png_bytes_for_test(64, 64)], None);
        // the media success leg: the budget and agreement guards pass, and the
        // zero tower plus zero head score evenly through the media path
        let (all, read) = model
            .probabilities(&mut session, &tok, &with_image)
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], [0.5, 0.5]);
        // the token count is the prefix rows plus the encoded prompt
        let predicted = vision::VisionTower::prefix_rows(&[(64, 64)]).unwrap();
        let max_len = model
            .max_length
            .saturating_sub(predicted)
            .min(model.image_text_length);
        // the pipeline maps the null state to the empty string ahead of images:
        // recompute that mapping, not the null itself (which renders "null")
        let (ids, _) = prompt::encode(
            &tok,
            &model.ids,
            &Json::Str(String::new()),
            q,
            max_len,
            prompt::Media::Image,
        )
        .unwrap();
        assert_eq!(read, predicted + ids.len());
        // the trunk received the full media prefix, not an empty or cut one
        assert_eq!(
            trunk
                .seen_prefix_len
                .load(std::sync::atomic::Ordering::Relaxed),
            predicted * model.n_embd
        );
        // ... and the encoded prompt ids, not just their count
        assert_eq!(*trunk.seen_ids.lock().unwrap(), vec![ids.clone()]);
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn a_tower_returning_narrow_rows_fires_the_agreement_guard() {
        // a 32-wide tower on the 64-wide model: the predictor counts rows for the
        // image, the tower returns half as many 64-wide rows, and the pipeline
        // ensure (not the tower, which does no self-check) refuses. The narrow
        // width is tensor bytes, so this leg keeps the positional builder.
        let mut model = test_model();
        model.vision = Some(
            vision::VisionTower::from_gguf(&vision::tower_builder(1, 32, 16, 16, 32).build(), 32)
                .unwrap(),
        );
        let mut session = media_session(trunk_config());
        let tok = BpeTokenizer::empty_for_test();
        let with_image = request_with_media(vec![vision::png_bytes_for_test(64, 64)], None);
        let err = model
            .probabilities(&mut session, &tok, &with_image)
            .expect_err("expected the agreement guard to refuse");
        assert!(
            err.to_string().contains("the vision tower returned"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn a_fitting_clip_answers_with_normalized_probabilities() {
        let mut model = test_model();
        model.audio = Some(audio::AudioTower::silent_tower_for_test(64));
        let (mut session, trunk) = media_session_with_trunk(trunk_config());
        let tok = BpeTokenizer::empty_for_test();
        // half a second of 16-bit silence: prepare keeps it whole, no resampling
        let samples = vec![0u8; 8_000 * 2];
        let wav = audio::wav_pcm16_mono_16k_for_test(&samples);
        let request = test_request();
        let q = &request.questions[0].1;
        let with_audio = request_with_media(Vec::new(), Some(wav));
        // the audio success leg: the capped decode, the agreement guards, and the
        // zero tower plus zero head score evenly through the speech path
        let (all, read) = model
            .probabilities(&mut session, &tok, &with_audio)
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], [0.5, 0.5]);
        // the token count is the prefix rows plus the encoded prompt
        let predicted = audio::prefix_rows(8_000);
        let max_len = model
            .max_length
            .saturating_sub(predicted)
            .min(model.audio_text_length);
        // the pipeline maps the null state to the empty object after speech:
        // recompute that mapping, not the null itself (which renders "null")
        let (ids, _) = prompt::encode(
            &tok,
            &model.ids,
            &Json::Object(Vec::new()),
            q,
            max_len,
            prompt::Media::Audio,
        )
        .unwrap();
        assert_eq!(read, predicted + ids.len());
        // the trunk received the full media prefix, not an empty or cut one
        assert_eq!(
            trunk
                .seen_prefix_len
                .load(std::sync::atomic::Ordering::Relaxed),
            predicted * model.n_embd
        );
        // ... and the encoded prompt ids, not just their count
        assert_eq!(*trunk.seen_ids.lock().unwrap(), vec![ids.clone()]);
    }

    #[test]
    fn the_pipeline_refuses_a_session_it_cannot_answer_with() {
        let model = test_model();
        let tok = BpeTokenizer::empty_for_test();
        let request = test_request();
        let mut wrong_width = trunk_config();
        wrong_width.hidden_size = 32;
        let mut causal = trunk_config();
        causal.is_causal = true;
        for (session, want) in [
            (
                test_session(wrong_width, true, true, false),
                "the session is 32 wide",
            ),
            (
                test_session(causal, true, true, false),
                "needs a bidirectional trunk",
            ),
            (
                test_session(trunk_config(), false, true, false),
                "cannot read hidden states",
            ),
            // a causal-only extractor behind a bidirectional config: the rows
            // would blind every option to the future, so refuse, not mis-score
            (
                test_session(trunk_config(), true, false, false),
                "causal hidden states only",
            ),
        ] {
            let mut session = session;
            let err = model
                .probabilities(&mut session, &tok, &request)
                .expect_err("expected the guard to refuse");
            assert!(err.to_string().contains(want), "unexpected message: {err}");
        }
        // empty and duplicate question lists never reach the trunk
        let empty = request_with(Vec::new(), Vec::new(), None);
        let mut session = text_session(trunk_config());
        let err = model
            .probabilities(&mut session, &tok, &empty)
            .expect_err("expected the guard to refuse");
        assert!(err.to_string().contains("must not be empty"), "{err}");
        let q = &request.questions[0].1;
        let dup = request_with(
            [("q".to_string(), q.clone()), ("q".to_string(), q.clone())].into(),
            Vec::new(),
            None,
        );
        let err = model
            .probabilities(&mut session, &tok, &dup)
            .expect_err("expected the guard to refuse");
        assert!(err.to_string().contains("duplicate question"), "{err}");
        // images and audio together are refused before either tower loads
        let both = request_with_media(vec![vec![0]], Some(vec![0]));
        let err = model
            .probabilities(&mut session, &tok, &both)
            .expect_err("expected the guard to refuse");
        assert!(err.to_string().contains("not both"), "{err}");
    }

    #[test]
    fn media_questions_refuse_a_backend_or_a_model_without_towers() {
        let model = test_model();
        let tok = BpeTokenizer::empty_for_test();
        let images = request_with_media(vec![vec![0]], None);
        // the capability probe fires before any byte is decoded
        let mut session = text_session(trunk_config());
        let err = model
            .probabilities(&mut session, &tok, &images)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("cannot read a media prefix"),
            "unexpected message: {err}"
        );
        // the speech side of the same probe: an incapable backend refuses before
        // any WAV byte decodes (garbage bytes prove the ordering)
        let audio = request_with_media(Vec::new(), Some(vec![0]));
        let mut session = text_session(trunk_config());
        let err = model
            .probabilities(&mut session, &tok, &audio)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("cannot read a media prefix"),
            "unexpected message: {err}"
        );
        // ... and past it, the missing tower names the reconversion, not a tensor
        let mut session = media_session(trunk_config());
        let err = model
            .probabilities(&mut session, &tok, &images)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no vision tower"),
            "unexpected message: {err}"
        );
        // the speech side mirrors it, before any WAV byte decodes
        let mut session = media_session(trunk_config());
        let err = model
            .probabilities(&mut session, &tok, &audio)
            .expect_err("expected the guard to refuse");
        assert!(
            err.to_string().contains("no speech tower"),
            "unexpected message: {err}"
        );
    }

    // an adapter-attached session for the fail-fast legs: the probe only needs
    // an adapter present, so install one without validation
    fn session_with_adapter() -> Session {
        let (mut session, _trunk) = media_session_with_trunk(trunk_config());
        session.install_lora_for_test(crate::lora::LoraAdapterWeights::new_classifier_for_testing(
            vec![0.0; 2],
            None,
            vec!["class".into()],
        ));
        session
    }

    #[test]
    fn media_questions_refuse_an_attached_adapter_before_any_decode() {
        let mut model = test_model();
        model.audio = Some(audio::AudioTower::silent_tower_for_test(64));
        let mut session = session_with_adapter();
        let tok = BpeTokenizer::empty_for_test();
        let request = test_request();
        // garbage bytes: only the adapter refusal proves they were never decoded
        let with_garbage = request_with_media(Vec::new(), Some(vec![0]));
        let err = model
            .probabilities(&mut session, &tok, &with_garbage)
            .expect_err("expected the adapter probe to refuse");
        assert!(
            err.to_string().contains("detach the adapter"),
            "unexpected message: {err}"
        );
        // ... while a text question flows past the probe: the hookless test trunk
        // still refuses it downstream, but never with the probe's message (the
        // refusal sits in the cause chain under the per-question context)
        let err = model
            .probabilities(&mut session, &tok, &request)
            .unwrap_err();
        let chain = format!("{err:?}");
        assert!(
            chain.contains("no backend runs it"),
            "unexpected chain: {chain}"
        );
        assert!(
            !chain.contains("detach the adapter"),
            "the media probe fired on a text question: {chain}"
        );
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn image_questions_refuse_an_attached_adapter_before_any_decode() {
        let mut model = test_model();
        model.vision = Some(
            vision::VisionTower::from_gguf(&vision::tower_builder_default().build(), 64).unwrap(),
        );
        let mut session = session_with_adapter();
        let tok = BpeTokenizer::empty_for_test();
        // garbage bytes: only the adapter refusal proves they were never decoded
        let with_garbage = request_with_media(vec![vec![0]], None);
        let err = model
            .probabilities(&mut session, &tok, &with_garbage)
            .expect_err("expected the adapter probe to refuse");
        assert!(
            err.to_string().contains("detach the adapter"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn from_gguf_loads_a_zero_block_head() {
        let file = Arc::new(d1_file(2048).build());
        let model = D1Model::from_gguf(&file, &BpeTokenizer::d1_for_test()).unwrap();
        assert_eq!(model.n_embd, 64);
        assert_eq!(model.max_length, 2048);
        assert_eq!(model.image_text_length, 2048);
        assert_eq!(model.audio_text_length, 2048);
        assert!(model.vision.is_none());
        assert!(model.audio.is_none());
        assert_eq!(model.temperatures.get("choice"), Some(&1.5));
        // all seven delimiter ids: a state/question or option/option_end swap
        // passes a partial pin, and pipeline tests build PromptTokens literally
        assert_eq!(
            (
                model.ids.bos,
                model.ids.marker,
                model.ids.state,
                model.ids.question,
                model.ids.option,
                model.ids.option_end,
                model.ids.decide,
            ),
            (1, 2, 3, 4, 5, 6, 7)
        );
        // ... and exactly the context floor loads
        let file = Arc::new(d1_file(96).build());
        let model = D1Model::from_gguf(&file, &BpeTokenizer::d1_for_test()).unwrap();
        assert_eq!(model.max_length, 96);
    }

    #[test]
    fn from_gguf_refuses_files_its_guards_cannot_use() {
        let tok = BpeTokenizer::d1_for_test();
        // a match, not unwrap_err: the model holds towers without Debug
        let check = |b: crate::gguf::GgufBuilder, want: &str| {
            let msg = match D1Model::from_gguf(&Arc::new(b.build()), &tok) {
                Ok(_) => panic!("expected `{want}`"),
                Err(e) => e.to_string(),
            };
            assert!(msg.contains(want), "unexpected message: {msg}");
        };
        let base = || crate::gguf::GgufBuilder::new().kv_u32("d1.head.block_count", 0);
        check(crate::gguf::GgufBuilder::new(), "not a d1 model");
        // the tokenizer fires before any weight load: no head tensors here
        let msg =
            match D1Model::from_gguf(&Arc::new(base().build()), &BpeTokenizer::empty_for_test()) {
                Ok(_) => panic!("expected the tokenizer refusal"),
                Err(e) => e.to_string(),
            };
        assert!(
            msg.contains("has no BOS token"),
            "unexpected message: {msg}"
        );
        // missing keys read causal ...
        check(base(), "bidirectional trunk");
        // ... and so does an explicit causal trunk
        check(
            base().kv_bool("lfm2.is_causal", true),
            "bidirectional trunk",
        );
        let bidi = || base().kv_bool("lfm2.is_causal", false);
        check(bidi(), "missing lfm2.embedding_length");
        check(bidi().kv_u32("lfm2.embedding_length", 0), "is 0");
        let wide = || bidi().kv_u32("lfm2.embedding_length", 64);
        check(wide(), "missing d1.context_length");
        check(wide().kv_u32("d1.context_length", 64), "too small");
        // one below the floor refuses too: the gate is `<`, not `<=`
        check(wide().kv_u32("d1.context_length", 95), "too small");
        // the refusal names the floor, so a refused file says what to fix
        check(
            wide().kv_u32("d1.context_length", 64),
            &format!("need {}", prompt::MIN_OPTION_BUDGET),
        );
        let roomy = || wide().kv_u32("d1.context_length", 2048);
        check(
            roomy()
                .kv_string_array("d1.temperature.keys", vec!["a".into(), "b".into()])
                .kv_f32_array("d1.temperature.values", vec![1.0]),
            "differ in length",
        );
        // one leg per falsified term: 0.0 and -1.0 falsify `> 0.0` only, inf
        // falsifies `is_finite` only, and NaN falsifies both
        for bad in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            check(
                roomy()
                    .kv_string_array("d1.temperature.keys", vec!["a".into()])
                    .kv_f32_array("d1.temperature.values", vec![bad]),
                "invalid d1 temperature",
            );
        }
        // a classifier file without causal keys reads bidirectional, like the trunk loader
        check(
            base().tensor_f32("classifier.weight", &[1], &[0.0]),
            "missing lfm2.embedding_length",
        );
        // a zero text room would load, then blame the prompt for every refusal
        check(
            roomy().kv_u32("d1.image_text_length", 0),
            "d1.image_text_length is 0",
        );
        check(
            roomy().kv_u32("d1.audio_text_length", 0),
            "d1.audio_text_length is 0",
        );
        // the scalar refusal wins over a tower load: a present-but-broken tower
        // behind a zero room still reports the room, not the tower error
        check(
            roomy()
                .kv_u32("d1.image_text_length", 0)
                .kv_u32("d1.vision.block_count", 0),
            "d1.image_text_length is 0",
        );
        check(
            roomy()
                .kv_u32("d1.audio_text_length", 0)
                .kv_u32("clip.audio.block_count", 1)
                .kv_u32("d1.audio.residual_width", 1),
            "d1.audio_text_length is 0",
        );
        // a broken head beats a broken tower: the head loads first, so its
        // scalar refusal surfaces before any tower tensor loads
        check(
            head::head_builder_default()
                .kv_u32("d1.head.attention.head_count", 2)
                .kv_u32("d1.vision.block_count", 0)
                .kv_bool("lfm2.is_causal", false)
                .kv_u32("lfm2.embedding_length", 64)
                .kv_u32("d1.context_length", 2048),
            "not 64 wide",
        );
    }

    #[test]
    fn from_gguf_loads_explicit_text_rooms() {
        // bespoke, not d1_file: the absent temperature table is the point here
        let file = Arc::new(
            head::head_builder_default()
                .kv_bool("lfm2.is_causal", false)
                .kv_u32("lfm2.embedding_length", 64)
                .kv_u32("d1.context_length", 2048)
                .kv_u32("d1.image_text_length", 896)
                .kv_u32("d1.audio_text_length", 1536)
                .build(),
        );
        let model = D1Model::from_gguf(&file, &BpeTokenizer::d1_for_test()).unwrap();
        assert_eq!(model.image_text_length, 896);
        assert_eq!(model.audio_text_length, 1536);
    }

    /// A vision model budgeted at 100 positions with a session, a tokenizer, and a
    /// 1024 x 1024 PNG: the shared setup of the over-budget image legs.
    #[cfg(feature = "vl-preprocess")]
    fn budget_model_with_1024_png() -> (D1Model, Session, BpeTokenizer, Vec<u8>) {
        let mut model = test_model();
        model.vision = Some(
            vision::VisionTower::from_gguf(&vision::tower_builder_default().build(), 64).unwrap(),
        );
        model.max_length = 100;
        let session = media_session(trunk_config());
        (
            model,
            session,
            BpeTokenizer::empty_for_test(),
            vision::png_bytes_for_test(1024, 1024),
        )
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn an_over_budget_image_refuses_before_the_tail_decodes() {
        let (model, mut session, tok, png) = budget_model_with_1024_png();
        let over = request_with_media(vec![png, vec![0, 1, 2, 3]], None);
        let err = model
            .probabilities(&mut session, &tok, &over)
            .expect_err("expected the budget to refuse");
        // the tail is garbage: only the budget refusal proves it was never decoded
        assert!(
            err.to_string().contains("send less"),
            "unexpected message: {err}"
        );
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn a_hostile_context_length_cannot_switch_the_media_budget_off() {
        // the file claims a gigantic context, but the trunk reads 4096: the
        // budget binds to the readable context, not the file's claim
        let (mut model, mut session, tok, png) = budget_model_with_1024_png();
        model.max_length = 1_000_000;
        let many = request_with_media(vec![png; 20], None);
        let err = model
            .probabilities(&mut session, &tok, &many)
            .expect_err("expected the budget to refuse");
        assert!(
            err.to_string().contains("of the 4096 positions"),
            "unexpected message: {err}"
        );
    }

    #[test]
    #[cfg(feature = "vl-preprocess")]
    fn an_over_budget_image_with_corrupt_pixels_refuses_on_the_budget() {
        let (model, mut session, tok, mut png) = budget_model_with_1024_png();
        // break the pixels past the header: the dims still read, the decode fails
        let at = png.len() / 2;
        png[at] ^= 0xff;
        assert_eq!(vision::image_dims(&png).unwrap(), (1024, 1024));
        assert!(vision::decode_image(&png).is_err());
        let over = request_with_media(vec![png], None);
        // the budget counts the header dims, so it refuses before the broken decode runs
        let err = model
            .probabilities(&mut session, &tok, &over)
            .expect_err("expected the budget to refuse");
        assert!(
            err.to_string().contains("send less"),
            "unexpected message: {err}"
        );
    }
}

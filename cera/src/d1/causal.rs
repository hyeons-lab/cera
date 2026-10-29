//! The public causal d1 model (`lfm2-d1`, `d1-3B`).
//!
//! An LFM2-VL style decoder reads the question as a prompt (the `systemone` chat template the
//! GGUF stores) and answers in the next token: each option has a short label, and the model's
//! logits at the end of the prompt, taken at the label tokens, are the option scores. There is
//! no head and no extra tensor: this is the ordinary causal trunk, plus the vision tower of its
//! mmproj for images, so nothing here touches weights.
//!
//! One forward pass per question reads the whole prompt. The questions of a request share the
//! images and the state at the front of their prompts, so that prefix is prefilled once and
//! each question continues from a checkpoint of it.
//!
//! The labels follow the model repository's `prompt.py` as llama.cpp's `/v1/systemone` has them:
//!
//! - `choice`: the option keys when all are single letters, else `A`, `B`, ... (`00`, `01`, ...
//!   past 26 options), each skipping to the next free single-token code when its own is not one
//!   token or is taken; an option is scored by the larger of its code's logit with and without a
//!   leading space.
//! - `noul`: `yes` / `Yes` / `YES` against `no` / `No` / `NO`, the larger logit of each group.
//! - `score`: the digit of the level.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use minijinja::Environment;

use super::json::Json;
use super::prompt::{Question, QuestionType};
use super::{Cancelled, D1Request, D1Response, public};
use crate::session::{EncodedImage, Session};
use crate::tokenizer::BpeTokenizer;

/// The text in the rendered prompt that stands for one image.
const MEDIA_MARKER: &str = "<__media__>";

/// Most images a request may carry (llama.cpp's limit, so the two engines refuse alike).
const MAX_IMAGES: usize = 8;

/// The context window a causal d1 session is opened with, in tokens. A prompt is the images
/// (about 1800 tokens for a 1280 x 720 photo) plus the state and the question, and the key-value
/// cache grows with this value, so it is a fixed middle: larger requests are refused with the
/// number of tokens they needed.
pub const CONTEXT_TOKENS: usize = 8192;

/// The causal d1 model: the prompt template and the tokens it is built from.
pub struct CausalD1 {
    env: Environment<'static>,
    image_start: u32,
    image_end: u32,
    temperatures: std::collections::HashMap<String, f32>,
    option_code_scheme: u32,
    is_omni: bool,
}

/// One step of a prompt: a text token, the rows of an image, or a speech clip.
#[derive(Debug, Clone, PartialEq)]
enum Item {
    Token(u32),
    Image(usize),
    Audio,
}

impl CausalD1 {
    /// Read the template and the media delimiters.
    ///
    /// # Errors
    ///
    /// Fails when `gguf` is not a public prompt-based d1 file, has no `systemone` template, or
    /// the tokenizer lacks the media delimiters.
    pub fn from_gguf(gguf: &crate::gguf::GgufFile, tokenizer: &BpeTokenizer) -> Result<Self> {
        let decision_type = public::decision_type(gguf);
        ensure!(
            public::is_prompt_d1(gguf),
            "not a public d1 model (lfm2.decision.type is {:?}, lfm3.decision.type is {:?})",
            gguf.get_str("lfm2.decision.type"),
            gguf.get_str("lfm3.decision.type")
        );
        if decision_type == Some(public::CAUSAL_TYPE) {
            ensure!(
                crate::model::lfm2::gguf_is_causal(gguf, "lfm2"),
                "a d1 model reads its answer from the next token and needs a causal trunk"
            );
        }
        let template = gguf
            .get_str("tokenizer.chat_template.systemone")
            .context("the model has no `systemone` template (tokenizer.chat_template.systemone)")?;
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("systemone", template.to_string())
            .context("the `systemone` template does not parse")?;
        let special = |name: &str| {
            tokenizer
                .special_token_id(name)
                .with_context(|| format!("the tokenizer has no `{name}` token"))
        };
        let (start_name, end_name) = if public::is_lfm3_d1(gguf)
            && (tokenizer.special_token_id("<|vision_start|>").is_some()
                || tokenizer.special_token_id("<|image_start|>").is_none())
        {
            ("<|vision_start|>", "<|vision_end|>")
        } else {
            ("<|image_start|>", "<|image_end|>")
        };
        Ok(Self {
            env,
            image_start: special(start_name)?,
            image_end: special(end_name)?,
            temperatures: public::temperatures(gguf)?,
            option_code_scheme: public::option_code_scheme(gguf),
            is_omni: decision_type == Some(public::LFM3_OMNI_TYPE),
        })
    }

    /// The calibration temperature of a question: its `type.bucket` entry, else its type's,
    /// else 1. The public d1-3B file has none.
    fn temperature(&self, q: &Question) -> f32 {
        self.temperatures
            .get(&q.temperature_key())
            .or_else(|| self.temperatures.get(q.kind.name()))
            .copied()
            .unwrap_or(1.0)
    }

    /// Whether this model is an `lfm3-d1-omni` variant (supports both image and speech inputs).
    pub fn is_omni(&self) -> bool {
        self.is_omni
    }

    /// Answer every question of `request`, stopping with [`Cancelled`] when `cancel` is set
    /// (read before each question).
    ///
    /// # Errors
    ///
    /// Fails on a request with audio when the model has no speech input or with more than eight
    /// images, on a question whose options cannot be labelled with single tokens, on media without
    /// the corresponding tower in `session`, on a prompt over the session's context, or on a trunk
    /// failure.
    pub fn answer(
        &self,
        session: &mut Session,
        tokenizer: &BpeTokenizer,
        request: &D1Request,
        cancel: &AtomicBool,
    ) -> Result<D1Response> {
        if self.is_omni {
            ensure!(
                request.images.is_empty() || request.audio.is_none(),
                "{}",
                D1Request::MEDIA_COMBO_ERR
            );
        } else {
            ensure!(
                request.audio.is_none(),
                "this model has no speech input (d1-3B reads text and images)"
            );
        }
        ensure!(
            request.images.len() <= MAX_IMAGES,
            "too many images, the maximum is {MAX_IMAGES}"
        );
        let mut seen = std::collections::HashSet::with_capacity(request.questions.len());
        for (name, _) in &request.questions {
            ensure!(seen.insert(name), "duplicate question `{name}`");
        }
        // each speech clip goes through the audio tower once, however many questions follow it
        let audio = request
            .audio
            .as_deref()
            .map(|wav| -> Result<(Vec<f32>, usize)> {
                let (samples, sample_rate) =
                    super::audio::decode_wav_capped(wav, super::audio::MAX_SECONDS)
                        .context("audio")?;
                session
                    .encode_audio_rows(&samples, sample_rate)
                    .context("audio")
            })
            .transpose()?;
        // each image goes through the vision tower once, however many questions follow it
        let images = request
            .images
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                session
                    .encode_image(bytes)
                    .with_context(|| format!("images[{i}]"))
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared = request
            .questions
            .iter()
            .map(|(name, q)| {
                self.prepare(
                    tokenizer,
                    &request.state,
                    q,
                    request.images.len(),
                    audio.is_some(),
                )
                .with_context(|| format!("question `{name}`"))
            })
            .collect::<Result<Vec<_>>>()?;

        // the images and the state come first, so the questions share a prefix; one tail token
        // at least stays per question, it is the position the answer is read at. A non-causal
        // trunk (`lfm3-d1`, `lfm3-d1-omni`) reads the whole prompt both ways on its global
        // attention layers, so its questions cannot share a causal prefix checkpoint.
        let shared = if session.model().config().is_causal && prepared.len() > 1 {
            let first = &prepared[0].items;
            let common = prepared
                .iter()
                .skip(1)
                .map(|p| {
                    first
                        .iter()
                        .zip(&p.items)
                        .take_while(|(a, b)| a == b)
                        .count()
                })
                .min()
                .unwrap_or(0);
            let shortest = prepared.iter().map(|p| p.items.len()).min().unwrap_or(0);
            common.min(shortest.saturating_sub(1))
        } else {
            0
        };
        // A session whose key-value state lives on the device (Metal, wgpu, the NPU) cannot be
        // checkpointed: its questions each start over, which the device's prefill makes cheap.
        let prefix_checkpoint = if shared > 0 && prepared.len() > 1 && session.checkpoint().is_ok()
        {
            ensure!(!cancel.load(Ordering::Relaxed), Cancelled);
            feed(
                session,
                &prepared[0].items[..shared],
                &images,
                audio.as_ref(),
            )?;
            Some(
                session
                    .checkpoint()
                    .context("checkpointing the shared prefix")?,
            )
        } else {
            None
        };
        let shared = if prefix_checkpoint.is_some() {
            shared
        } else {
            0
        };

        let mut answers = Vec::with_capacity(prepared.len());
        let mut input_tokens = 0usize;
        for ((name, q), prompt) in request.questions.iter().zip(&prepared) {
            ensure!(!cancel.load(Ordering::Relaxed), Cancelled);
            let from = match &prefix_checkpoint {
                Some(checkpoint) => {
                    session
                        .restore(checkpoint)
                        .context("restoring the shared prefix")?;
                    shared
                }
                None => {
                    session.reset()?;
                    0
                }
            };
            feed(session, &prompt.items[from..], &images, audio.as_ref())
                .with_context(|| format!("question `{name}`"))?;
            input_tokens = input_tokens.saturating_add(session.position() as usize);
            let logits = session
                .last_logits()
                .context("the model returned no logits for the prompt")?;
            let scores = prompt
                .groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|&t| logits.get(t as usize).copied())
                        .collect::<Option<Vec<f32>>>()
                        .map(|g| g.into_iter().fold(f32::NEG_INFINITY, f32::max))
                        .with_context(|| {
                            format!(
                                "a label token is outside the vocabulary of {} logits",
                                logits.len()
                            )
                        })
                })
                .collect::<Result<Vec<f32>>>()?;
            let probabilities = super::prompt::probabilities(q, &scores, self.temperature(q))
                .with_context(|| format!("question `{name}`"))?;
            answers.push((
                name.clone(),
                super::prompt::answer(q, &probabilities)
                    .with_context(|| format!("question `{name}`"))?,
            ));
        }
        Ok(D1Response {
            answers,
            input_tokens,
        })
    }

    /// The `systemone` template rendered for one question; an image is [`MEDIA_MARKER`].
    fn render(
        &self,
        q: &Question,
        state: &Json,
        options: &[serde_json::Value],
        n_images: usize,
    ) -> Result<String> {
        // Caller text becomes prompt tokens, so a `<|name|>` in it must not reach the tokenizer as a
        // special token (the omni route escapes the same way, `prompt::escape`).
        let context = serde_json::json!({
            "type": q.kind.name(),
            "instructions": escape_json_strings(&serde_json::json!(q.instructions)),
            "state": escape_json_strings(&match state {
                Json::Null => serde_json::Value::Null,
                Json::Str(s) => serde_json::Value::String(s.clone()),
                other => serde_json::Value::String(other.dumps_indent(2)),
            }),
            "options": options.iter().map(escape_json_strings).collect::<Vec<_>>(),
            "images": vec![MEDIA_MARKER; n_images],
        });
        self.env
            .get_template("systemone")?
            .render(minijinja::value::Serde(context))
            .context("rendering the `systemone` template")
    }

    /// The prompt of one question and the label tokens its options are scored at.
    fn prepare(
        &self,
        tokenizer: &BpeTokenizer,
        state: &Json,
        q: &Question,
        n_images: usize,
        has_audio: bool,
    ) -> Result<Prepared> {
        let (options, groups) = options_and_groups_with_scheme(
            &|text| tokenizer.encode(text),
            q,
            self.option_code_scheme,
        )?;
        let n_media = if has_audio { 1 } else { n_images };
        let rendered = self.render(q, state, &options, n_media)?;
        let mut items = Vec::new();
        let pieces: Vec<&str> = rendered.split(MEDIA_MARKER).collect();
        ensure!(
            pieces.len() == n_media + 1,
            "the template placed {} images for {n_media}",
            pieces.len() - 1
        );
        for (i, piece) in pieces.iter().enumerate() {
            if i > 0 {
                items.push(Item::Token(self.image_start));
                if has_audio {
                    items.push(Item::Audio);
                } else {
                    items.push(Item::Image(i - 1));
                }
                items.push(Item::Token(self.image_end));
            }
            items.extend(tokenizer.encode(piece).into_iter().map(Item::Token));
        }
        ensure!(
            matches!(items.last(), Some(Item::Token(_))),
            "the prompt is empty"
        );
        Ok(Prepared { items, groups })
    }
}

struct Prepared {
    items: Vec<Item>,
    /// Per option, in the order the answer reads them, the tokens whose logits score it.
    groups: Vec<Vec<u32>>,
}

/// Append `items` to the session: when the segment contains an image or speech clip and the
/// backend can hand out token embedding rows, splice all text and media rows into one
/// `append_embeddings` call (required for non-causal trunks so bidirectional attention sees
/// the full prompt in one pass, and faster on causal trunks by avoiding per-segment launches).
fn feed(
    session: &mut Session,
    items: &[Item],
    images: &[EncodedImage],
    audio: Option<&(Vec<f32>, usize)>,
) -> Result<()> {
    let has_media = items
        .iter()
        .any(|item| matches!(item, Item::Image(_) | Item::Audio));
    if has_media && session.model().supports_embedding_input() {
        let hs = session.model().config().hidden_size;
        let mut fused_rows = Vec::new();
        let mut fused_tokens = 0usize;
        let mut run = Vec::new();
        let mut can_fuse = true;
        for item in items {
            match item {
                Item::Token(t) => run.push(*t),
                Item::Image(i) => {
                    if !run.is_empty() {
                        if let Some(rows) = session.model().embed_image_marker_rows(&run) {
                            fused_rows.extend_from_slice(&rows);
                            fused_tokens += run.len();
                            run.clear();
                        } else {
                            can_fuse = false;
                            break;
                        }
                    }
                    let image = images.get(*i).context("the prompt names a missing image")?;
                    fused_rows.extend_from_slice(image.rows());
                    fused_tokens += image.n_tokens();
                }
                Item::Audio => {
                    if !run.is_empty() {
                        if let Some(rows) = session.model().embed_image_marker_rows(&run) {
                            fused_rows.extend_from_slice(&rows);
                            fused_tokens += run.len();
                            run.clear();
                        } else {
                            can_fuse = false;
                            break;
                        }
                    }
                    let (rows, n_tokens) =
                        audio.context("the prompt names a missing speech clip")?;
                    fused_rows.extend_from_slice(rows);
                    fused_tokens += *n_tokens;
                }
            }
        }
        if can_fuse && !run.is_empty() {
            if let Some(rows) = session.model().embed_image_marker_rows(&run) {
                fused_rows.extend_from_slice(&rows);
                fused_tokens += run.len();
            } else {
                can_fuse = false;
            }
        }
        if can_fuse && fused_tokens > 0 && fused_rows.len() == fused_tokens * hs {
            session.append_embeddings(&fused_rows, fused_tokens)?;
            return Ok(());
        }
    }

    let mut run: Vec<u32> = Vec::new();
    for item in items {
        match item {
            Item::Token(t) => run.push(*t),
            Item::Image(i) => {
                if !run.is_empty() {
                    session.append_tokens(&std::mem::take(&mut run))?;
                }
                let image = images.get(*i).context("the prompt names a missing image")?;
                session
                    .append_encoded_image(image)
                    .with_context(|| format!("images[{i}]"))?;
            }
            Item::Audio => {
                if !run.is_empty() {
                    session.append_tokens(&std::mem::take(&mut run))?;
                }
                let (rows, n_tokens) = audio.context("the prompt names a missing speech clip")?;
                session
                    .append_embeddings(rows, *n_tokens)
                    .context("audio")?;
            }
        }
    }
    if !run.is_empty() {
        session.append_tokens(&run)?;
    }
    Ok(())
}

/// The template's `options` for a question (with the labels it shows) and, per option in the
/// order the answer reads them, the label tokens that score it.
#[cfg(test)]
fn options_and_groups(
    encode: &dyn Fn(&str) -> Vec<u32>,
    q: &Question,
) -> Result<(Vec<serde_json::Value>, Vec<Vec<u32>>)> {
    options_and_groups_with_scheme(encode, q, 1)
}

/// `value` with every string run through [`super::prompt::escape`], recursively.
fn escape_json_strings(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        // the template's own media marker splits the rendered prompt into text and media items,
        // so caller text must not be able to spell it either
        Value::String(s) => {
            Value::String(super::prompt::escape(s).replace(MEDIA_MARKER, "<¦_media_¦>"))
        }
        Value::Array(items) => Value::Array(items.iter().map(escape_json_strings).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), escape_json_strings(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn options_and_groups_with_scheme(
    encode: &dyn Fn(&str) -> Vec<u32>,
    q: &Question,
    scheme: u32,
) -> Result<(Vec<serde_json::Value>, Vec<Vec<u32>>)> {
    let value = |v: &Json| match v {
        Json::Null => serde_json::Value::Null,
        Json::Str(s) => serde_json::Value::String(s.clone()),
        other => serde_json::Value::String(other.dumps()),
    };
    let single = |forms: &[&str]| -> Vec<u32> {
        forms.iter().fold(Vec::new(), |mut out, form| {
            if let [token] = encode(form)[..]
                && !out.contains(&token)
            {
                out.push(token);
            }
            out
        })
    };
    match q.kind {
        QuestionType::Noul => {
            // the request parser accepts `yes`/`no` for `true`/`false`, as the omni prompt does
            let side = |names: [&str; 2]| {
                names
                    .iter()
                    .find_map(|n| q.criteria.iter().find(|(k, _)| k == n))
                    .map_or(serde_json::Value::Null, |(_, v)| value(v))
            };
            // the template shows `true` first; the answer reads `false` then `true`, which
            // `prompt::probabilities` turns back into [true, false]
            let options = vec![
                serde_json::json!({"key": "true", "description": side(["true", "yes"]), "label": "true"}),
                serde_json::json!({"key": "false", "description": side(["false", "no"]), "label": "false"}),
            ];
            let no = single(&["no", "No", "NO"]);
            let yes = single(&["yes", "Yes", "YES"]);
            ensure!(
                !no.is_empty() && !yes.is_empty(),
                "the yes and no labels are not single tokens in this vocabulary"
            );
            Ok((options, vec![no, yes]))
        }
        QuestionType::Score => {
            let mut options = Vec::new();
            let mut groups = Vec::new();
            for (i, (_, level)) in q.criteria.iter().enumerate() {
                let key = i.to_string();
                let group = single(&[&key]);
                ensure!(
                    !group.is_empty(),
                    "the label `{key}` is not a single token in this vocabulary"
                );
                options.push(
                    serde_json::json!({"key": key, "description": value(level), "label": key}),
                );
                groups.push(group);
            }
            Ok((options, groups))
        }
        QuestionType::Choice => {
            let keys: Vec<&str> = q.criteria.iter().map(|(k, _)| k.as_str()).collect();
            let labelled = choice_labels_with_scheme(encode, &keys, scheme)?;
            let options = q
                .criteria
                .iter()
                .zip(&labelled)
                .map(|((key, description), (label, _))| {
                    serde_json::json!({"key": key, "description": value(description), "label": label})
                })
                .collect();
            Ok((
                options,
                labelled.into_iter().map(|(_, group)| group).collect(),
            ))
        }
    }
}

/// The code of each choice option and the tokens that score it (scheme 1).
#[cfg(test)]
fn choice_labels(
    encode: &dyn Fn(&str) -> Vec<u32>,
    keys: &[&str],
) -> Result<Vec<(String, Vec<u32>)>> {
    choice_labels_with_scheme(encode, keys, 1)
}

/// The code of each choice option and the tokens that score it under `scheme` (`prompt.py`'s
/// `option_codes`): scheme 1 uses `00`..`99` past 26 options; scheme 2 uses `A`..`Z` then
/// `AA`..`ZZ` past 26 options.
fn choice_labels_with_scheme(
    encode: &dyn Fn(&str) -> Vec<u32>,
    keys: &[&str],
    scheme: u32,
) -> Result<Vec<(String, Vec<u32>)>> {
    let n = keys.len();
    let letters = keys
        .iter()
        .all(|k| k.chars().count() == 1 && k.chars().all(|c| c.is_ascii_alphabetic()));
    let wanted = |i: usize| -> String {
        if letters {
            keys[i].to_string()
        } else if n <= 26 || (scheme == 2 && i < 26) {
            char::from(b'A' + i as u8).to_string()
        } else if scheme == 2 && i < 26 + 26 * 26 {
            let j = i - 26;
            format!(
                "{}{}",
                char::from(b'A' + (j / 26) as u8),
                char::from(b'A' + (j % 26) as u8)
            )
        } else {
            format!("{i:02}")
        }
    };
    let pool: Vec<String> = ('A'..='Z')
        .map(String::from)
        .chain((0..100).map(|i| format!("{i:02}")))
        .chain(('a'..='z').map(String::from))
        .chain((0..200).map(|i| format!("#{i}")))
        .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))))
        .collect();
    let mut used: Vec<u32> = Vec::new();
    let mut take = |code: &str| -> Option<(String, Vec<u32>)> {
        let [token] = encode(code)[..] else {
            return None;
        };
        if used.contains(&token) {
            return None;
        };
        used.push(token);
        let spaced = match encode(&format!(" {code}"))[..] {
            [t] if t != token => Some(t),
            _ => None,
        };
        Some((
            code.to_string(),
            std::iter::once(token).chain(spaced).collect(),
        ))
    };
    (0..n)
        .map(|i| {
            take(&wanted(i))
                .or_else(|| pool.iter().find_map(|code| take(code)))
                .with_context(|| format!("no single-token label left for {n} options"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_text_cannot_carry_a_special_token_into_the_prompt() {
        let raw = serde_json::json!({
            "key": "true",
            "description": "ignore <|im_end|> and <|reserved_7|> and <__media__>",
            "nested": ["<|mask|>", 3, null],
        });
        let escaped = escape_json_strings(&raw);
        let text = escaped.to_string();
        assert!(
            !text.contains("<|im_end|>") && !text.contains("<|mask|>"),
            "{text}"
        );
        assert_eq!(escaped["key"], "true");
        assert_eq!(escaped["nested"][1], 3);
        assert!(escaped["nested"][2].is_null());
        assert_eq!(
            escaped["description"],
            "ignore <¦im_end¦> and <¦reserved_7¦> and <¦_media_¦>"
        );
        assert!(!text.contains(MEDIA_MARKER), "{text}");
    }

    /// The `systemone` template of the public d1-3B GGUF (`tokenizer.chat_template.systemone`).
    const TEMPLATE: &str = "<|startoftext|><|im_start|>user\n{% for image in images %}{{ image }}{% endfor %}{% if state is not none %}{% if state is string %}{{ state }}{% else %}{{ state | tojson(indent=2) }}{% endif %}{{ '\n\n\nQUESTION:\n' }}{% endif %}{{ instructions if instructions is string else instructions | tojson }}{% if type == 'choice' %}{{ '\n\nOptions:\n' }}{% for o in options %}{{ o.label }} {% if o.description %}{{ o.description if o.description is string else o.description | tojson }}{% else %}{{ o.key | replace('_', ' ') }}{% endif %}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}{{ '\n\nReply with the option code only.' }}{% elif type == 'noul' %}{% set ns = namespace(criteria=false) %}{% for o in options %}{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}{% if ns.criteria %}{% for o in options %}{{ '\nYes: ' if o.key == 'true' else '\nNo: ' }}{% if o.description is none %}None{% else %}{{ o.description if o.description is string else o.description | tojson }}{% endif %}{% endfor %}{% endif %}{{ '\n\nReply with yes or no only.' }}{% else %}{{ '\n\n' }}{% for o in options %}{{ o.key }} {{ o.description if o.description is string else o.description | tojson }}{{ '\n' }}{% endfor %}{{ '\nReply with a single digit 0-' }}{{ options | length - 1 }}{{ ' only.' }}{% endif %}{{ '<|im_end|>\n<|im_start|>assistant\n' }}";

    fn model() -> CausalD1 {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("systemone", TEMPLATE.to_string())
            .unwrap();
        CausalD1 {
            env,
            image_start: 1,
            image_end: 2,
            temperatures: std::collections::HashMap::new(),
            option_code_scheme: 1,
            is_omni: false,
        }
    }

    fn question(text: &str) -> Question {
        Question::from_json(&Json::parse(text).unwrap()).unwrap()
    }

    /// A toy vocabulary: every ASCII letter, digit and the words `yes` / `no` in three cases
    /// are one token, a space before a letter is another, nothing else is.
    fn toy_encode(text: &str) -> Vec<u32> {
        let table = ["yes", "Yes", "YES", "no", "No", "NO"];
        if let Some(i) = table.iter().position(|w| *w == text) {
            return vec![1000 + i as u32];
        }
        let mut chars = text.chars();
        match (chars.next(), chars.next(), chars.next()) {
            (Some(c), None, _) if c.is_ascii_alphanumeric() => vec![c as u32],
            (Some(' '), Some(c), None) if c.is_ascii_alphabetic() => vec![500 + c as u32],
            _ => text.chars().map(|c| 2000 + c as u32).collect(),
        }
    }

    fn render(text: &str, state: &Json, n_images: usize) -> String {
        let q = question(text);
        let (options, _) = options_and_groups(&toy_encode, &q).unwrap();
        model().render(&q, state, &options, n_images).unwrap()
    }

    #[test]
    fn a_choice_prompt_lists_lettered_options_and_asks_for_the_code() {
        let prompt = render(
            r#"{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges","tech_support":null}}"#,
            &Json::Str("I was charged twice.".into()),
            0,
        );
        assert_eq!(
            prompt,
            "<|startoftext|><|im_start|>user\nI was charged twice.\n\n\nQUESTION:\nWhich team?\n\nOptions:\nA Charges\nB tech support\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn a_noul_prompt_shows_yes_before_no_and_only_with_criteria() {
        let plain = render(
            r#"{"type":"noul","instructions":"Refund?"}"#,
            &Json::Str("s".into()),
            0,
        );
        assert!(
            plain.ends_with(
                "Refund?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
            ),
            "{plain}"
        );
        let defined = render(
            r#"{"type":"noul","instructions":"Refund?","criteria":{"false":"nothing asked","true":{"asked":1}}}"#,
            &Json::Str("s".into()),
            0,
        );
        assert!(
            defined.contains(
                "Refund?\nYes: {\"asked\": 1}\nNo: nothing asked\n\nReply with yes or no only."
            ),
            "{defined}"
        );
        let half = render(
            r#"{"type":"noul","instructions":"Refund?","criteria":{"true":"asked"}}"#,
            &Json::Str("s".into()),
            0,
        );
        assert!(half.contains("\nYes: asked\nNo: None\n"), "{half}");
    }

    #[test]
    fn a_noul_prompt_reads_yes_and_no_as_true_and_false() {
        // the request parser and the omni prompt accept both spellings
        let yes_no = render(
            r#"{"type":"noul","instructions":"Refund?","criteria":{"yes":"asked","no":"nothing"}}"#,
            &Json::Str("s".into()),
            0,
        );
        assert!(
            yes_no.contains("Refund?\nYes: asked\nNo: nothing\n\nReply with yes or no only."),
            "{yes_no}"
        );
    }

    #[test]
    fn a_score_prompt_numbers_the_levels_and_names_the_last_digit() {
        let prompt = render(
            r#"{"type":"score","instructions":"How urgent?","criteria":["low","mid",null]}"#,
            &Json::Str("s".into()),
            0,
        );
        assert!(
            prompt.contains(
                "How urgent?\n\n0 low\n1 mid\n2 null\n\nReply with a single digit 0-2 only."
            ),
            "{prompt}"
        );
    }

    #[test]
    fn a_structured_state_is_indented_and_no_state_drops_the_question_header() {
        let prompt = render(
            r#"{"type":"noul","instructions":"?"}"#,
            &Json::parse(r#"{"a": [1, {"b": null}]}"#).unwrap(),
            0,
        );
        assert!(
            prompt.contains("user\n{\n  \"a\": [\n    1,\n    {\n      \"b\": null\n    }\n  ]\n}\n\n\nQUESTION:\n?"),
            "{prompt}"
        );
        let none = render(r#"{"type":"noul","instructions":"?"}"#, &Json::Null, 0);
        assert!(none.contains("user\n?\n\nReply"), "{none}");
    }

    #[test]
    fn images_come_first_one_marker_each() {
        let prompt = render(r#"{"type":"noul","instructions":"?"}"#, &Json::Null, 2);
        assert!(
            prompt.starts_with(&format!(
                "<|startoftext|><|im_start|>user\n{MEDIA_MARKER}{MEDIA_MARKER}?"
            )),
            "{prompt}"
        );
    }

    #[test]
    fn choice_options_are_lettered_by_position_or_by_their_own_single_letters() {
        let labelled = |keys: &[&str]| -> Vec<String> {
            choice_labels(&toy_encode, keys)
                .unwrap()
                .into_iter()
                .map(|(label, _)| label)
                .collect()
        };
        assert_eq!(labelled(&["billing", "tech"]), ["A", "B"]);
        assert_eq!(labelled(&["x", "y", "z"]), ["x", "y", "z"]);
        // a single letter among longer keys is not enough
        assert_eq!(labelled(&["a", "bb"]), ["A", "B"]);
        // 27 options want `00`..`26`, which this vocabulary has no single token for, so the
        // pool supplies `A`..`Z` and then `a`
        let keys: Vec<String> = (0..27).map(|i| format!("key{i}")).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let many = labelled(&keys);
        assert_eq!(many.len(), 27);
        assert_eq!(many[25], "Z");
        assert_eq!(many[26], "a");
    }

    #[test]
    fn a_choice_label_scores_with_and_without_the_leading_space() {
        let labelled = choice_labels(&toy_encode, &["one", "two"]).unwrap();
        assert_eq!(labelled[0].1, vec!['A' as u32, 500 + 'A' as u32]);
        assert_eq!(labelled[1].1, vec!['B' as u32, 500 + 'B' as u32]);
    }

    #[test]
    fn a_taken_or_multi_token_code_falls_to_the_next_free_one() {
        // `A` is two tokens in this vocabulary, so the first option takes `B`, the second `C`
        let encode = |text: &str| {
            if text == "A" {
                vec![1, 2]
            } else {
                toy_encode(text)
            }
        };
        let labelled = choice_labels(&encode, &["one", "two"]).unwrap();
        let labels: Vec<_> = labelled.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(labels, ["B", "C"]);
        // nothing is a single token: no label is left
        let none = |_: &str| vec![1, 2];
        let err = choice_labels(&none, &["one", "two"]).expect_err("expected a refusal");
        assert!(err.to_string().contains("no single-token label"), "{err}");
    }

    #[test]
    fn noul_and_score_read_yes_no_and_digit_tokens() {
        let (_, groups) = options_and_groups(
            &toy_encode,
            &question(r#"{"type":"noul","instructions":"?"}"#),
        )
        .unwrap();
        // read as [false, true]
        assert_eq!(groups, vec![vec![1003, 1004, 1005], vec![1000, 1001, 1002]]);
        let (options, groups) = options_and_groups(
            &toy_encode,
            &question(r#"{"type":"score","instructions":"?","criteria":["a","b"]}"#),
        )
        .unwrap();
        assert_eq!(groups, vec![vec!['0' as u32], vec!['1' as u32]]);
        assert_eq!(options[1]["key"], "1");
    }

    #[test]
    fn scheme_two_uses_two_letter_codes_past_twenty_six_options() {
        let encode_lfm3 = |text: &str| {
            let mut chars = text.chars();
            match (chars.next(), chars.next(), chars.next()) {
                (Some(a), Some(b), None) if a.is_ascii_uppercase() && b.is_ascii_uppercase() => {
                    vec![3000 + (a as u32 - 'A' as u32) * 26 + (b as u32 - 'A' as u32)]
                }
                _ => toy_encode(text),
            }
        };
        let keys: Vec<String> = (0..28).map(|i| format!("key{i}")).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let labelled = choice_labels_with_scheme(&encode_lfm3, &keys, 2).unwrap();
        let labels: Vec<_> = labelled.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(labels[0], "A");
        assert_eq!(labels[25], "Z");
        assert_eq!(labels[26], "AA");
        assert_eq!(labels[27], "AB");
    }
}

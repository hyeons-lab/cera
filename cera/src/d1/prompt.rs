//! Questions in, token sequences out, answers back.
//!
//! A question is `type` (`noul`, `choice` or `score`), `instructions` and `criteria`. Each one is
//! rendered against the state as its own sequence:
//!
//! ```text
//! <bos> <state> state  <q> instructions  (<opt> <mask> option_i </opt>)*  <decide>
//! ```
//!
//! The model scores the trunk's hidden state at every `<mask>` and softmaxes over the question's
//! options. The delimiters are the tokenizer's reserved block, and a `<|name|>` in caller text is
//! rewritten so a state can never forge one.

use anyhow::{Context, Result, bail, ensure};

use super::json::Json;
use crate::tokenizer::BpeTokenizer;

/// What a question asks for. The row each variant scores lives in
/// [`QuestionType::index`], the only place the mapping is spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// One of the named options.
    Choice,
    /// An ordered level, two to ten of them.
    Score,
    /// Yes or no.
    Noul,
}

impl QuestionType {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }

    /// The name a request uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// The row in the head's question-type table, the contract
    /// [`crate::d1::head::Head::scores`] reads.
    pub const fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }
}

/// The media a request carries ahead of its text. A question is worded for the kind it was
/// trained with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Media {
    /// Text only.
    None,
    /// One or more images.
    Image,
    /// A speech clip.
    Audio,
}

/// One question, validated.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub kind: QuestionType,
    pub instructions: String,
    /// `choice`: `{name: description}`; `score`: the levels, lowest first (their keys are
    /// unused); `noul`: the optional `{"true": .., "false": ..}` (or `yes`/`no`) definitions.
    pub criteria: Vec<(String, Json)>,
}

impl Question {
    /// Read a question from its JSON object.
    ///
    /// # Errors
    ///
    /// Refuses what the model was not trained on: an unknown type, fewer than two choices, a
    /// score outside two to ten levels, or `noul` criteria that are not an object.
    pub fn from_json(q: &Json) -> Result<Self> {
        let (Some(kind), Some(instructions)) = (q.get("type"), q.get("instructions")) else {
            bail!(
                "a question is an object with `type`, `instructions` and, for choice and score, `criteria`"
            );
        };
        let kind_name = kind.as_str().unwrap_or("");
        let Some(kind) = QuestionType::from_name(kind_name) else {
            bail!("question type must be one of choice, noul, score, got {kind_name:?}");
        };
        let instructions = render_value(instructions);
        let criteria = q.get("criteria").filter(|c| !matches!(c, Json::Null));
        let criteria = match (kind, criteria) {
            (QuestionType::Choice, Some(Json::Object(entries))) if entries.len() >= 2 => {
                entries.clone()
            }
            (QuestionType::Choice, _) => {
                bail!("a choice needs criteria {{name: description}} with at least two options")
            }
            (QuestionType::Score, Some(Json::Array(levels)))
                if (2..=10).contains(&levels.len()) =>
            {
                levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| (i.to_string(), level.clone()))
                    .collect()
            }
            (QuestionType::Score, _) => {
                bail!("a score needs criteria: a list of 2 to 10 level descriptions, lowest first")
            }
            (QuestionType::Noul, None) => Vec::new(),
            (QuestionType::Noul, Some(Json::Object(entries))) => entries.clone(),
            (QuestionType::Noul, Some(_)) => {
                bail!(
                    "noul criteria are optional: {{\"true\": \"...\", \"false\": \"...\"}} (or \"yes\", \"no\")"
                )
            }
        };
        Ok(Self {
            kind,
            instructions,
            criteria,
        })
    }

    /// How many options the model scores.
    pub fn options(&self) -> usize {
        match self.kind {
            QuestionType::Noul => 2,
            _ => self.criteria.len(),
        }
    }

    /// The calibration bucket of this question: `choice:3-5`, `noul:2`, `score:11+` and so on.
    pub fn temperature_key(&self) -> String {
        let bucket = match self.options() {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        format!("{}:{bucket}", self.kind.name())
    }
}

fn is_blank(value: &Json) -> bool {
    matches!(value, Json::Null) || matches!(value, Json::Str(s) if s.is_empty())
}

/// The option texts in the model's order. A `noul` is read as `[false, true]`.
///
/// After an image a `noul` without definitions of its own is worded `false: no`, `true: yes`.
/// After speech every `noul` is, and a `choice` is written as `option_000: <description>` (its
/// name when it has no description): the questions were trained that way.
pub fn render_options(q: &Question, media: Media) -> Vec<String> {
    match q.kind {
        QuestionType::Choice if media == Media::Audio => q
            .criteria
            .iter()
            .enumerate()
            .map(|(i, (key, description))| {
                let text = if is_blank(description) {
                    key.clone()
                } else {
                    render_value(description)
                };
                format!("option_{i:03}: {text}")
            })
            .collect(),
        QuestionType::Choice => q
            .criteria
            .iter()
            .map(|(key, description)| {
                if is_blank(description) {
                    key.clone()
                } else {
                    format!("{key}: {}", render_value(description))
                }
            })
            .collect(),
        QuestionType::Score => q
            .criteria
            .iter()
            .enumerate()
            .map(|(i, (_, level))| format!("level {i}: {}", render_value(level)))
            .collect(),
        QuestionType::Noul
            if media == Media::Audio || (media == Media::Image && q.criteria.is_empty()) =>
        {
            vec!["false: no".to_string(), "true: yes".to_string()]
        }
        QuestionType::Noul => {
            let side = |names: [&str; 2]| {
                names
                    .iter()
                    .find_map(|n| q.criteria.iter().find(|(k, _)| k == n))
                    .map(|(_, v)| v)
            };
            let describe = |value: Option<&Json>, fallback: &str| match value {
                Some(v) if !is_blank(v) => render_value(v),
                _ => fallback.to_string(),
            };
            vec![
                format!(
                    "false: {}",
                    describe(side(["false", "no"]), "no, the statement does not hold")
                ),
                format!(
                    "true: {}",
                    describe(side(["true", "yes"]), "yes, the statement holds")
                ),
            ]
        }
    }
}

/// `<|name|>` becomes `<¦name¦>`, so caller text cannot emit a delimiter or marker token.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let name_len = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if name_len > 0 && after[name_len..].starts_with("|>") {
            out.push_str("<¦");
            out.push_str(&after[..name_len]);
            out.push_str("¦>");
            rest = &after[name_len + 2..];
        } else {
            out.push_str("<|");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// A value as the prompt writes it: a string as it is, anything else as `json.dumps`.
pub fn render_value(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => other.dumps(),
    }
}

/// The token ids the layout is built from.
#[derive(Debug, Clone, Copy)]
pub struct PromptTokens {
    pub bos: u32,
    pub marker: u32,
    pub state: u32,
    pub question: u32,
    pub option: u32,
    pub option_end: u32,
    pub decide: u32,
}

impl PromptTokens {
    /// Look the delimiters up in the model's tokenizer.
    ///
    /// # Errors
    ///
    /// Fails when the tokenizer lacks one: it is not a d1 tokenizer.
    pub fn from_tokenizer(tok: &BpeTokenizer) -> Result<Self> {
        let special = |name: &str| {
            tok.special_token_id(name)
                .ok_or_else(|| anyhow::anyhow!("the tokenizer has no `{name}` token"))
        };
        Ok(Self {
            bos: tok
                .bos_token()
                .ok_or_else(|| anyhow::anyhow!("the tokenizer has no BOS token"))?,
            marker: special("<|mask|>")?,
            state: special("<|reserved_7|>")?,
            question: special("<|reserved_8|>")?,
            option: special("<|reserved_9|>")?,
            option_end: special("<|reserved_10|>")?,
            decide: special("<|reserved_11|>")?,
        })
    }
}

/// Tokens a question gets per option, before the shared allowance.
const PER_OPTION: usize = 24;

/// Smallest option-block allowance, in tokens, and therefore the smallest context a d1
/// model can answer from (gated in [`super::D1Model::from_gguf`] and
/// [`crate::convert::d1::apply_metadata`]).
pub(crate) const MIN_OPTION_BUDGET: usize = 96;

/// The state's token stream, uncapped: the state delimiter plus every state token. Encode
/// once per request with [`encode_with_state_ids`]; each question then takes the room its
/// options leave, instead of re-running BPE over the whole state per question.
pub(crate) fn encode_state(tok: &BpeTokenizer, ids: &PromptTokens, state: &Json) -> Vec<u32> {
    let mut out = vec![ids.state];
    out.extend(tok.encode(&escape(&render_value(state))));
    out
}

/// The token ids of one question over one state, and the position of each option's marker.
///
/// The option block gets `max(MIN_OPTION_BUDGET, min(24 k + 32, max_len / 2))` tokens,
/// shared out evenly; the
/// state is cut on the right to the room that is left.
///
/// # Errors
///
/// Fails when the question has fewer than two options, or the options do not fit in
/// `max_len`.
pub fn encode(
    tok: &BpeTokenizer,
    ids: &PromptTokens,
    state: &Json,
    q: &Question,
    max_len: usize,
    media: Media,
) -> Result<(Vec<u32>, Vec<usize>)> {
    encode_with_state_ids(tok, ids, &encode_state(tok, ids, state), q, max_len, media)
}

/// Like [`encode`], over an already-encoded state stream (see [`encode_state`]).
///
/// `state_ids[0]` must be the state delimiter; anything else fails.
///
/// # Errors
///
/// Fails on an empty state stream, a state stream not starting with the state delimiter,
/// fewer than two options, or options that do not fit in `max_len`.
pub(crate) fn encode_with_state_ids(
    tok: &BpeTokenizer,
    ids: &PromptTokens,
    state_ids: &[u32],
    q: &Question,
    max_len: usize,
    media: Media,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let enc = |s: &str| tok.encode(&escape(s));
    let options = render_options(q, media);
    let k = options.len();
    ensure!(k >= 2, "a question needs at least two options (has {k})");
    let budget = MIN_OPTION_BUDGET.max((k * PER_OPTION + 32).min(max_len / 2));
    let per = 2.max(budget.saturating_sub(3 * k) / k);

    let mut question = vec![ids.question];
    question.extend(enc(&q.instructions));
    question.truncate(16.max(budget));
    let mut markers = Vec::with_capacity(k);
    for text in &options {
        markers.push(question.len() + 1);
        question.push(ids.option);
        question.push(ids.marker);
        question.extend(enc(&format!(" {text}")).into_iter().take(per));
        question.push(ids.option_end);
    }
    question.push(ids.decide);

    let room = max_len.saturating_sub(question.len() + 2);
    let (head, tail) = state_ids
        .split_first()
        .context("the encoded state is empty")?;
    ensure!(
        *head == ids.state,
        "the encoded state must start with the state delimiter"
    );
    let mut kept = Vec::with_capacity(1 + room.min(tail.len()));
    kept.push(*head);
    kept.extend(tail.iter().take(room).copied());

    let mut out = Vec::with_capacity(1 + kept.len() + question.len());
    out.push(ids.bos);
    out.extend_from_slice(&kept);
    out.extend_from_slice(&question);
    // the state already took its cut above, so the question must fit whole: truncating
    // here would cut decide and the tail options while every marker still resolves
    ensure!(
        out.len() <= max_len,
        "the options do not fit in the context (need {}, have {max_len})",
        out.len()
    );
    // markers index into `out`, so they resolve exactly when the length check
    // above passes; a second guard here could never fire (and the small-overrun
    // test below is what pins the refusal, not this mapping).
    let markers: Vec<usize> = markers.iter().map(|m| m + 1 + kept.len()).collect();
    Ok((out, markers))
}

/// The probabilities a question's scores mean: softmax over the options after the calibration
/// `temperature`, in option order (`[yes, no]` for a `noul`: the model reads it as
/// `[false, true]`).
///
/// # Errors
///
/// Fails when the question has fewer than two options, `scores` does not hold exactly one
/// score per option, the temperature is not positive and finite (dividing by zero
/// yields NaN probabilities, not an error, without this), or the scores divide to
/// probabilities that are not finite (a denormal temperature overflows ordinary
/// scores to infinity, and `inf - inf` in the softmax is NaN).
pub fn probabilities(q: &Question, scores: &[f32], temperature: f32) -> Result<Vec<f32>> {
    ensure!(
        q.options() >= 2,
        "a question needs at least two options (has {})",
        q.options()
    );
    ensure!(
        scores.len() == q.options(),
        "probabilities needs one score per option ({} != {})",
        scores.len(),
        q.options()
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be positive and finite, got {temperature}"
    );
    // names the culprit when the weights themselves are bad; the output check
    // below is what closes the overflow trigger
    ensure!(
        scores.iter().all(|s| s.is_finite()),
        "scores must be finite"
    );
    let mut p: Vec<f32> = scores.iter().map(|s| s / temperature).collect();
    super::softmax_exact(&mut p);
    ensure!(
        p.iter().all(|v| v.is_finite()),
        "probabilities are not finite"
    );
    if q.kind == QuestionType::Noul {
        p.reverse();
    }
    Ok(p)
}

/// TypeSafe's confidence of a `choice`: how far the top probability is above uniform, scaled so
/// uniform is 0 and certainty is 1.
fn confidence_choice(p: &[f32]) -> f64 {
    let n = p.len() as f64;
    let top = p.iter().copied().map(f64::from).fold(0.0, f64::max);
    ((top - 1.0 / n) / (1.0 - 1.0 / n)).max(0.0)
}

/// TypeSafe's confidence of a `score`: 1 minus the mean distance to the most likely level,
/// relative to that of a uniform distribution around the middle level.
fn confidence_score(p: &[f32]) -> f64 {
    let n = p.len() as f64;
    let mode = p
        .iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > p[best] { i } else { best });
    let dist: f64 = p
        .iter()
        .enumerate()
        .map(|(i, v)| f64::from(*v) * (i as f64 - mode as f64).abs())
        .sum();
    let uniform: f64 = (0..p.len())
        .map(|i| (i as f64 - (n - 1.0) / 2.0).abs() / n)
        .sum();
    (1.0 - dist / uniform).max(0.0)
}

/// A question's answer as the response object: a `noul`'s P(yes); a `choice`'s pick, its
/// confidence (TypeSafe's formulas) and its probabilities; a `score`'s expected level, its confidence, its
/// probabilities and the legend of the levels.
///
/// # Errors
///
/// Fails when the question has fewer than two options, or `p` does not hold exactly one
/// probability per option (the `zip` below would silently drop the surplus side).
pub fn answer(q: &Question, p: &[f32]) -> Result<Json> {
    ensure!(
        q.options() >= 2,
        "a question needs at least two options (has {})",
        q.options()
    );
    ensure!(
        p.len() == q.options(),
        "answer needs one probability per option ({} != {})",
        p.len(),
        q.options()
    );
    let kind = |name: &str| ("type".to_string(), Json::Str(name.to_string()));
    let f = |v: f32| Json::Float(f64::from(v));
    if q.kind == QuestionType::Noul {
        return Ok(Json::Object(vec![kind("noul"), ("noul".into(), f(p[0]))]));
    }
    // the first maximum
    let best = p
        .iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > p[best] { i } else { best });
    let probabilities =
        |names: Vec<String>| Json::Object(names.into_iter().zip(p.iter().map(|v| f(*v))).collect());
    if q.kind == QuestionType::Choice {
        let names: Vec<String> = q.criteria.iter().map(|(k, _)| k.clone()).collect();
        return Ok(Json::Object(vec![
            kind("choice"),
            ("choice".into(), Json::Str(names[best].clone())),
            ("confidence".into(), Json::Float(confidence_choice(p))),
            ("probabilities".into(), probabilities(names)),
        ]));
    }
    let levels: Vec<String> = (0..p.len()).map(|i| i.to_string()).collect();
    let expected: f32 = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
    let legend = Json::Object(
        q.criteria
            .iter()
            .enumerate()
            .map(|(i, (_, level))| (i.to_string(), level.clone()))
            .collect(),
    );
    Ok(Json::Object(vec![
        kind("score"),
        ("score".into(), f(expected)),
        ("confidence".into(), Json::Float(confidence_score(p))),
        ("probabilities".into(), probabilities(levels)),
        ("legend".into(), legend),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(text: &str) -> Question {
        Question::from_json(&Json::parse(text).unwrap()).unwrap()
    }

    #[test]
    fn choice_options_read_their_description_or_just_their_name() {
        let q = q(
            r#"{"type":"choice","instructions":"?","criteria":{"a":"Alpha","b":"","c":null,"d":{"x":1}}}"#,
        );
        assert_eq!(
            render_options(&q, Media::None),
            ["a: Alpha", "b", "c", r#"d: {"x": 1}"#]
        );
    }

    #[test]
    fn score_options_are_numbered_levels() {
        let q = q(r#"{"type":"score","instructions":"?","criteria":["low","high"]}"#);
        assert_eq!(
            render_options(&q, Media::None),
            ["level 0: low", "level 1: high"]
        );
    }

    #[test]
    fn noul_options_take_the_given_definitions_or_the_defaults() {
        let plain = q(r#"{"type":"noul","instructions":"?"}"#);
        assert_eq!(
            render_options(&plain, Media::None),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let defined = q(r#"{"type":"noul","instructions":"?","criteria":{"yes":"Y","false":""}}"#);
        assert_eq!(
            render_options(&defined, Media::None),
            ["false: no, the statement does not hold", "true: Y"]
        );
    }

    #[test]
    fn an_image_noul_is_worded_yes_and_no_unless_it_defines_its_own() {
        let plain = q(r#"{"type":"noul","instructions":"?"}"#);
        assert_eq!(
            render_options(&plain, Media::Image),
            ["false: no", "true: yes"]
        );
        let defined = q(r#"{"type":"noul","instructions":"?","criteria":{"true":"Y"}}"#);
        assert_eq!(
            render_options(&defined, Media::Image),
            ["false: no, the statement does not hold", "true: Y"]
        );
        // other types are worded the same with or without an image
        let choice = q(r#"{"type":"choice","instructions":"?","criteria":{"a":"A","b":"B"}}"#);
        assert_eq!(
            render_options(&choice, Media::Image),
            render_options(&choice, Media::None)
        );
    }

    #[test]
    fn speech_questions_are_worded_the_way_they_were_trained() {
        let choice = q(
            r#"{"type":"choice","instructions":"?","criteria":{"a":"Alpha","b":"","c":null,"d":{"x":1}}}"#,
        );
        assert_eq!(
            render_options(&choice, Media::Audio),
            [
                "option_000: Alpha",
                "option_001: b",
                "option_002: c",
                r#"option_003: {"x": 1}"#
            ]
        );
        // a noul ignores its own definitions after speech
        let defined =
            q(r#"{"type":"noul","instructions":"?","criteria":{"true":"Y","false":"N"}}"#);
        assert_eq!(
            render_options(&defined, Media::Audio),
            ["false: no", "true: yes"]
        );
        // a score is unchanged
        let score = q(r#"{"type":"score","instructions":"?","criteria":["low","high"]}"#);
        assert_eq!(
            render_options(&score, Media::Audio),
            render_options(&score, Media::None)
        );
    }

    #[test]
    fn questions_the_model_was_not_trained_on_are_refused() {
        for bad in [
            r#"{"type":"choice","instructions":"?","criteria":{"only":"one"}}"#,
            r#"{"type":"choice","instructions":"?"}"#,
            r#"{"type":"score","instructions":"?","criteria":["one"]}"#,
            r#"{"type":"noul","instructions":"?","criteria":["x"]}"#,
            r#"{"type":"rank","instructions":"?"}"#,
            r#"{"instructions":"?"}"#,
        ] {
            assert!(
                Question::from_json(&Json::parse(bad).unwrap()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn escape_defuses_only_complete_special_tokens() {
        assert_eq!(escape("a <|mask|> b"), "a <¦mask¦> b");
        assert_eq!(
            escape("<|im_end|><|reserved_7|>"),
            "<¦im_end¦><¦reserved_7¦>"
        );
        assert_eq!(escape("<| no |> <|open"), "<| no |> <|open");
        assert_eq!(escape("<|"), "<|");
    }

    #[test]
    fn temperature_buckets_follow_the_option_count() {
        assert_eq!(
            q(r#"{"type":"noul","instructions":"?"}"#).temperature_key(),
            "noul:2"
        );
        let many = format!(
            r#"{{"type":"choice","instructions":"?","criteria":{{{}}}}}"#,
            (0..12)
                .map(|i| format!("\"o{i}\":\"d\""))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert_eq!(q(&many).temperature_key(), "choice:11+");
        let score = q(r#"{"type":"score","instructions":"?","criteria":["a","b","c","d"]}"#);
        assert_eq!(score.temperature_key(), "score:3-5");
    }

    #[test]
    fn a_noul_reads_the_model_as_false_then_true() {
        let noul = q(r#"{"type":"noul","instructions":"?"}"#);
        // scores [false, true] = [0, ln 3]: P(true) = 0.75
        let p = probabilities(&noul, &[0.0, 3f32.ln()], 1.0).unwrap();
        assert!((p[0] - 0.75).abs() < 1e-6 && (p[1] - 0.25).abs() < 1e-6);
        let answer = answer(&noul, &p).unwrap();
        assert!(
            (answer
                .get("noul")
                .map(|j| match j {
                    Json::Float(f) => *f,
                    _ => f64::NAN,
                })
                .unwrap()
                - 0.75)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn a_score_answers_with_its_expected_level() {
        let score = q(r#"{"type":"score","instructions":"?","criteria":["a","b","c"]}"#);
        let a = answer(&score, &[0.25, 0.5, 0.25]).unwrap();
        match a.get("score") {
            Some(Json::Float(f)) => assert!((f - 1.0).abs() < 1e-6),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            a.get("legend").unwrap().dumps(),
            r#"{"0": "a", "1": "b", "2": "c"}"#
        );
    }

    /// Delimiter ids far outside the toy vocab, so a marker hit cannot be a coincidence.
    fn ids() -> PromptTokens {
        PromptTokens {
            bos: 100,
            marker: 101,
            state: 102,
            question: 103,
            option: 104,
            option_end: 105,
            decide: 106,
        }
    }

    /// A merge-less tokenizer over the only bytes the fixtures below use.
    fn tiny_tok() -> BpeTokenizer {
        BpeTokenizer::from_vocab(vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b" ".to_vec(),
            b"?".to_vec(),
        ])
    }

    fn two_choice() -> Question {
        q(r#"{"type":"choice","instructions":"?","criteria":{"a":"","b":""}}"#)
    }

    #[test]
    fn markers_land_on_the_mask_ids_in_option_order() {
        let (out, markers) = encode(
            &tiny_tok(),
            &ids(),
            &Json::Str("a".into()),
            &two_choice(),
            256,
            Media::None,
        )
        .unwrap();
        assert_eq!(markers.len(), 2);
        assert!(markers[0] < markers[1]);
        for m in &markers {
            assert_eq!(out[*m], 101);
        }
        assert!(out.len() < 256);
    }

    #[test]
    fn a_long_state_is_cut_on_the_right_and_the_question_survives() {
        let tok = tiny_tok();
        let ids = ids();
        let q = two_choice();
        // head and tail differ, so keeping the wrong end fails
        let state = Json::Str(format!("{} {}", "a ".repeat(300), "b ".repeat(300)));
        let (full, _) = encode(&tok, &ids, &state, &q, 1_000_000, Media::None).unwrap();
        let (out, markers) = encode(&tok, &ids, &state, &q, 64, Media::None).unwrap();
        assert_eq!(out.len(), 64);
        assert!(out.len() < full.len());
        // the tail cut nothing: decide is still there and every marker resolves
        assert!(out.contains(&ids.decide));
        for m in &markers {
            assert_eq!(out[*m], ids.marker);
            assert!(*m < 64);
        }
        // ... and the cut kept the head: the kept state is the full stream's prefix
        let q_at = out.iter().position(|&t| t == ids.question).unwrap();
        assert_eq!(&out[1..q_at], &full[1..q_at]);
    }

    #[test]
    fn a_state_stream_starting_elsewhere_is_refused() {
        let ids = ids();
        let bad = [ids.bos, ids.marker];
        let err = encode_with_state_ids(&tiny_tok(), &ids, &bad, &two_choice(), 256, Media::None)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("must start with the state delimiter"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn an_empty_state_stream_is_refused_not_indexed() {
        let err = encode_with_state_ids(&tiny_tok(), &ids(), &[], &two_choice(), 256, Media::None)
            .unwrap_err();
        assert!(
            err.to_string().contains("the encoded state is empty"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn a_tiny_context_refuses_instead_of_truncating_the_options() {
        let err = encode(
            &tiny_tok(),
            &ids(),
            &Json::Str("a".into()),
            &two_choice(),
            8,
            Media::None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("do not fit"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn a_question_barely_over_budget_refuses_instead_of_dropping_decide() {
        let tok = tiny_tok();
        let ids = ids();
        // 86 instruction tokens plus two 5-token options is a 98-token question
        // against max_len 96: the 4-token overrun is shorter than the last option
        // block, so every marker still resolves while decide is cut
        let mut q = two_choice();
        q.instructions = "a ".repeat(43);
        let err = encode(&tok, &ids, &Json::Str("a".into()), &q, 96, Media::None).unwrap_err();
        assert!(
            err.to_string().contains("do not fit"),
            "unexpected message: {err}"
        );
        // the refusal reports both sides, so the caller sees the overshoot
        // (the 98-token question plus the bos and state framing)
        assert!(
            err.to_string().contains("need 100") && err.to_string().contains("have 96"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn a_manually_built_question_still_needs_two_options() {
        let tok = tiny_tok();
        let ids = ids();
        let state = Json::Str("a".into());
        for (criteria, want) in [
            (vec![], "(has 0)"),
            (vec![("a".to_string(), Json::Str("b".into()))], "(has 1)"),
        ] {
            let q = Question {
                kind: QuestionType::Choice,
                instructions: String::new(),
                criteria,
            };
            let err = encode(&tok, &ids, &state, &q, 256, Media::None).unwrap_err();
            assert!(
                err.to_string().contains("at least two options"),
                "unexpected message: {err}"
            );
            assert!(err.to_string().contains(want), "unexpected message: {err}");
        }
    }

    #[test]
    fn question_type_rows_follow_the_head_table() {
        assert_eq!(
            (
                QuestionType::Choice.index(),
                QuestionType::Score.index(),
                QuestionType::Noul.index()
            ),
            (0, 1, 2)
        );
    }

    fn close_enough(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn temperature_scales_every_question_type_before_the_softmax() {
        let noul = q(r#"{"type":"noul","instructions":"?"}"#);
        let choice = two_choice();
        let score = q(r#"{"type":"score","instructions":"?","criteria":["a","b"]}"#);
        // scores [0, ln 2]: T=1 -> [1/3, 2/3]; T=2 halves the logits first. The noul
        // reports [yes, no], so its expectations are the reversal of the other two.
        for (q, t1, t2) in [
            (&noul, [2.0 / 3.0, 1.0 / 3.0], [0.58578644, 0.41421356]),
            (&choice, [1.0 / 3.0, 2.0 / 3.0], [0.41421356, 0.58578644]),
            (&score, [1.0 / 3.0, 2.0 / 3.0], [0.41421356, 0.58578644]),
        ] {
            close_enough(&probabilities(q, &[0.0, 2f32.ln()], 1.0).unwrap(), &t1);
            close_enough(&probabilities(q, &[0.0, 2f32.ln()], 2.0).unwrap(), &t2);
        }
        // the [false, true] model order is reported [yes, no]: [ln 3, 0] reads [0.25, 0.75]
        close_enough(
            &probabilities(&noul, &[3f32.ln(), 0.0], 1.0).unwrap(),
            &[0.25, 0.75],
        );
    }

    #[test]
    fn a_non_positive_temperature_is_refused_not_nan() {
        let noul = q(r#"{"type":"noul","instructions":"?"}"#);
        for t in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = probabilities(&noul, &[0.0, 1.0], t).unwrap_err();
            assert!(
                err.to_string().contains("temperature"),
                "unexpected message: {err}"
            );
        }
    }

    #[test]
    fn encode_and_encode_with_state_ids_agree() {
        let tok = tiny_tok();
        let ids = ids();
        let q = two_choice();
        for (state, max) in [
            (Json::Str("a".into()), 256),
            (Json::Str("a ".repeat(300)), 64),
        ] {
            let a = encode(&tok, &ids, &state, &q, max, Media::None).unwrap();
            let b = encode_with_state_ids(
                &tok,
                &ids,
                &encode_state(&tok, &ids, &state),
                &q,
                max,
                Media::None,
            )
            .unwrap();
            assert_eq!(a, b);
        }
    }

    fn one_option() -> Question {
        Question {
            kind: QuestionType::Choice,
            instructions: String::new(),
            criteria: vec![("a".to_string(), Json::Str("b".into()))],
        }
    }

    #[test]
    fn probabilities_needs_one_score_per_option() {
        let choice = two_choice();
        for scores in [vec![0.0], vec![0.0, 1.0, 2.0]] {
            let err = probabilities(&choice, &scores, 1.0).unwrap_err();
            assert!(
                err.to_string().contains("one score per option"),
                "unexpected message: {err}"
            );
        }
        // the count matches but the question is short an option: only k >= 2 fires
        let err = probabilities(&one_option(), &[0.5], 1.0).unwrap_err();
        assert!(
            err.to_string().contains("at least two options"),
            "unexpected message: {err}"
        );
        assert!(
            err.to_string().contains("(has 1)"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn probabilities_refuse_scores_that_divide_out_of_finite_range() {
        let choice = two_choice();
        // non-finite scores name themselves, before any division
        let err = probabilities(&choice, &[0.5, f32::NAN], 1.0).unwrap_err();
        assert!(
            err.to_string().contains("scores must be finite"),
            "unexpected message: {err}"
        );
        // ... while a denormal temperature passes the positive check, then
        // overflows ordinary scores to infinity (inf - inf is NaN)
        let err = probabilities(&choice, &[0.5, -0.5], 1e-45).unwrap_err();
        assert!(
            err.to_string().contains("probabilities are not finite"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn an_answer_needs_one_probability_per_option() {
        let choice = two_choice();
        let err = answer(&choice, &[]).unwrap_err();
        assert!(
            err.to_string().contains("one probability per option"),
            "unexpected message: {err}"
        );
        let err = answer(&choice, &[0.5]).unwrap_err();
        assert!(
            err.to_string().contains("one probability per option"),
            "unexpected message: {err}"
        );
        // ... and a surplus, which the zip below would silently drop
        let err = answer(&choice, &[0.25, 0.25, 0.5]).unwrap_err();
        assert!(
            err.to_string().contains("one probability per option"),
            "unexpected message: {err}"
        );
        // the count matches but the question is short an option: only k >= 2 fires
        let err = answer(&one_option(), &[0.5]).unwrap_err();
        assert!(
            err.to_string().contains("at least two options"),
            "unexpected message: {err}"
        );
        assert!(
            err.to_string().contains("(has 1)"),
            "unexpected message: {err}"
        );
    }

    /// TypeSafe's formulas: uniform is 0, certainty is 1, and a score is scored by how
    /// concentrated it is around its mode.
    #[test]
    fn confidences_follow_the_typesafe_formulas() {
        let choice =
            q(r#"{"type":"choice","instructions":"?","criteria":{"a":null,"b":null,"c":null}}"#);
        let answer_of = |q: &Question, p: &[f32]| answer(q, p).unwrap();
        let confidence = |a: &Json| match a.get("confidence") {
            Some(Json::Float(v)) => *v,
            other => panic!("no float confidence: {other:?}"),
        };
        let uniform = answer_of(&choice, &[1.0 / 3.0; 3]);
        assert!(confidence(&uniform).abs() < 1e-6);
        let sure = answer_of(&choice, &[1.0, 0.0, 0.0]);
        assert!((confidence(&sure) - 1.0).abs() < 1e-9);
        // (0.8 - 1/3) / (1 - 1/3)
        let likely = answer_of(&choice, &[0.8, 0.1, 0.1]);
        assert!((confidence(&likely) - 0.7).abs() < 1e-6);
        let score = q(r#"{"type":"score","instructions":"?","criteria":["a","b","c"]}"#);
        let sure = answer_of(&score, &[0.0, 1.0, 0.0]);
        assert!((confidence(&sure) - 1.0).abs() < 1e-9);
        // the mode is level 0 (first maximum), the mean distance to it 0.5 * 2, and a uniform
        // distribution around the middle level has (1 + 0 + 1) / 3
        let spread = answer_of(&score, &[0.5, 0.0, 0.5]);
        assert_eq!(confidence(&spread), 0.0);
        let mild = answer_of(&score, &[0.7, 0.2, 0.1]);
        let want = 1.0 - (0.2 * 1.0 + 0.1 * 2.0) / (2.0 / 3.0);
        assert!((confidence(&mild) - want).abs() < 1e-6);
    }

    /// A score's legend returns the level descriptions as given, structured ones included.
    #[test]
    fn a_score_legend_keeps_structured_levels() {
        let q = q(r#"{"type":"score","instructions":"?","criteria":["low",{"x":1}]}"#);
        let a = answer(&q, &[0.5, 0.5]).unwrap();
        assert_eq!(
            a.get("legend").unwrap().dumps(),
            r#"{"0": "low", "1": {"x": 1}}"#
        );
    }
}

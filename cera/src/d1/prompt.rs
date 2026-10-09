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

use anyhow::{Result, bail, ensure};

use super::json::Json;
use crate::tokenizer::BpeTokenizer;

/// What a question asks for, in the order of the head's question-type table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// One of the named options.
    Choice = 0,
    /// An ordered level, two to ten of them.
    Score = 1,
    /// Yes or no.
    Noul = 2,
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
        let instructions = match instructions {
            Json::Str(s) => s.clone(),
            other => other.dumps(),
        };
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

/// A description as the prompt writes it: a string as it is, anything else as JSON.
fn criterion(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => other.dumps(),
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
                    criterion(description)
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
                    format!("{key}: {}", criterion(description))
                }
            })
            .collect(),
        QuestionType::Score => q
            .criteria
            .iter()
            .enumerate()
            .map(|(i, (_, level))| format!("level {i}: {}", criterion(level)))
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
                Some(v) if !is_blank(v) => criterion(v),
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

/// A state as the prompt writes it: a string as it is, anything else as `json.dumps`.
pub fn serialize_state(state: &Json) -> String {
    match state {
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

/// The token ids of one question over one state, and the position of each option's marker.
///
/// The option block gets `max(96, min(24 k + 32, max_len / 2))` tokens, shared out evenly; the
/// state is cut on the right to the room that is left.
///
/// # Errors
///
/// Fails when the options do not fit in `max_len`.
pub fn encode(
    tok: &BpeTokenizer,
    ids: &PromptTokens,
    state: &Json,
    q: &Question,
    max_len: usize,
    media: Media,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let enc = |s: &str| tok.encode(&escape(s));
    let options = render_options(q, media);
    let k = options.len();
    let budget = 96.max((k * PER_OPTION + 32).min(max_len / 2));
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
    let mut state_ids = vec![ids.state];
    state_ids.extend(enc(&serialize_state(state)).into_iter().take(room));

    let mut out = Vec::with_capacity(1 + state_ids.len() + question.len());
    out.push(ids.bos);
    out.extend_from_slice(&state_ids);
    out.extend_from_slice(&question);
    out.truncate(max_len);
    let markers: Vec<usize> = markers.iter().map(|m| m + 1 + state_ids.len()).collect();
    ensure!(
        markers.last().is_some_and(|m| *m < max_len),
        "the options do not fit in the context"
    );
    Ok((out, markers))
}

/// The probabilities a question's scores mean: softmax over the options after the calibration
/// `temperature`, in option order (`[yes, no]` for a `noul`: the model reads it as
/// `[false, true]`).
pub fn probabilities(q: &Question, scores: &[f32], temperature: f32) -> Vec<f32> {
    let z: Vec<f32> = scores
        .iter()
        .take(q.options())
        .map(|s| s / temperature)
        .collect();
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    if q.kind == QuestionType::Noul {
        p.reverse();
    }
    p
}

/// A question's answer as the response object: a `noul`'s P(yes); a `choice`'s pick, its
/// confidence and its probabilities; a `score`'s expected level, its confidence, its
/// probabilities and the legend of the levels.
pub fn answer(q: &Question, p: &[f32]) -> Json {
    let kind = |name: &str| ("type".to_string(), Json::Str(name.to_string()));
    let f = |v: f32| Json::Float(f64::from(v));
    if q.kind == QuestionType::Noul {
        return Json::Object(vec![kind("noul"), ("noul".into(), f(p[0]))]);
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
        return Json::Object(vec![
            kind("choice"),
            ("choice".into(), Json::Str(names[best].clone())),
            ("confidence".into(), f(p[best])),
            ("probabilities".into(), probabilities(names)),
        ]);
    }
    let levels: Vec<String> = (0..p.len()).map(|i| i.to_string()).collect();
    let expected: f32 = p.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
    let legend = Json::Object(
        q.criteria
            .iter()
            .enumerate()
            .map(|(i, (_, level))| (i.to_string(), Json::Str(criterion(level))))
            .collect(),
    );
    Json::Object(vec![
        kind("score"),
        ("score".into(), f(expected)),
        ("confidence".into(), f(p[best])),
        ("probabilities".into(), probabilities(levels)),
        ("legend".into(), legend),
    ])
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
        let p = probabilities(&noul, &[0.0, 3f32.ln()], 1.0);
        assert!((p[0] - 0.75).abs() < 1e-6 && (p[1] - 0.25).abs() < 1e-6);
        let answer = answer(&noul, &p);
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
        let a = answer(&score, &[0.25, 0.5, 0.25]);
        match a.get("score") {
            Some(Json::Float(f)) => assert!((f - 1.0).abs() < 1e-6),
            other => panic!("{other:?}"),
        }
        assert_eq!(
            a.get("legend").unwrap().dumps(),
            r#"{"0": "a", "1": "b", "2": "c"}"#
        );
    }
}

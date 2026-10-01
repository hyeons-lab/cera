//! What an audio model needs to be told to speak.
//!
//! The LFM2-Audio family selects text-to-speech and interleaved output by
//! *system prompt*, and the right prompt (and which voices it accepts) differs
//! per release: the English model takes `Perform TTS.` plus one of four voice
//! phrases, the Japanese one takes `Perform TTS in japanese.` and no voice at
//! all. A phrase the model was not trained on makes it answer in text and
//! produce no audio, so a client that hardcodes one model's strings breaks on
//! the next.
//!
//! A GGUF carries none of this (`general.name` is a checkpoint id and the chat
//! template has no voice slot), so it is explicit data, resolved in this order:
//!
//! 1. an `audio_profile` object in the bundle manifest, so a bundle can
//!    describe itself;
//! 2. the built-in registry in `audio_profiles.json`, matched against the
//!    manifest's model and vocoder file names, for released bundles whose
//!    manifests cannot be edited;
//! 3. [`AudioProfile::generic`]: plain `Perform TTS.` and no voices, which every
//!    LFM2-Audio model accepts.
//!
//! The profile is fully resolved here: every voice carries the complete system
//! prompt to send, so no client assembles prompt strings of its own.

use std::sync::OnceLock;

use crate::manifest::Manifest;

/// System prompt for interleaved text-and-audio replies.
pub const INTERLEAVED_SYSTEM_PROMPT: &str = "Respond with interleaved text and audio.";

/// Placeholder in [`AudioProfile::sample_texts`] for the model's display name.
pub const MODEL_PLACEHOLDER: &str = "{model}";

/// One speaker a model understands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsVoice {
    /// Human-readable name for a picker.
    pub label: String,
    /// The phrase that selects the voice (`Use the US female voice.`). Also the
    /// stable identifier a client saves as the user's choice.
    pub prompt: String,
    /// The complete system prompt for text-to-speech in this voice.
    pub tts_system_prompt: String,
    /// The complete system prompt for interleaved replies in this voice.
    pub interleaved_system_prompt: String,
}

/// The prompts, voices and sample text of one audio model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioProfile {
    /// The system prompt for text-to-speech when the user picked no voice: the
    /// first voice's, or the bare prompt for a model without voices.
    pub tts_system_prompt: String,
    /// The same for interleaved replies.
    pub interleaved_system_prompt: String,
    /// Voices the model understands. Empty means it has none: a persona it was
    /// not trained on silences it, so none is ever appended.
    pub voices: Vec<TtsVoice>,
    /// Sample sentences for a demo, with [`MODEL_PLACEHOLDER`] where the model's
    /// name goes. Empty means the client's own generic samples.
    pub sample_texts: Vec<String>,
}

impl AudioProfile {
    /// Build a profile from its base prompts and `(label, prompt)` voices.
    pub fn new(
        tts_prompt: &str,
        interleaved_prompt: &str,
        voices: &[(&str, &str)],
        sample_texts: Vec<String>,
    ) -> Self {
        let voices: Vec<TtsVoice> = voices
            .iter()
            .map(|(label, prompt)| TtsVoice {
                label: (*label).to_string(),
                prompt: (*prompt).to_string(),
                tts_system_prompt: format!("{tts_prompt} {prompt}"),
                interleaved_system_prompt: format!("{interleaved_prompt} {prompt}"),
            })
            .collect();
        let default_of = |base: &str, pick: fn(&TtsVoice) -> &str| match voices.first() {
            Some(v) => pick(v).to_string(),
            None => base.to_string(),
        };
        Self {
            tts_system_prompt: default_of(tts_prompt, |v| &v.tts_system_prompt),
            interleaved_system_prompt: default_of(interleaved_prompt, |v| {
                &v.interleaved_system_prompt
            }),
            voices,
            sample_texts,
        }
    }

    /// Plain `Perform TTS.` and no voices: what any model nobody described gets.
    pub fn generic() -> Self {
        Self::new("Perform TTS.", INTERLEAVED_SYSTEM_PROMPT, &[], Vec::new())
    }

    /// The voice a saved choice resolves to: the one whose [`TtsVoice::prompt`]
    /// is `saved`, else the model's first, else none. A voice saved under another
    /// model is therefore never sent to this one.
    pub fn voice_for(&self, saved: Option<&str>) -> Option<&TtsVoice> {
        let saved = saved.map(str::trim).filter(|s| !s.is_empty());
        saved
            .and_then(|s| self.voices.iter().find(|v| v.prompt == s))
            .or_else(|| self.voices.first())
    }

    /// The text-to-speech system prompt for a saved voice choice.
    pub fn tts_system_prompt_for(&self, saved: Option<&str>) -> &str {
        match self.voice_for(saved) {
            Some(v) => &v.tts_system_prompt,
            None => &self.tts_system_prompt,
        }
    }

    /// The interleaved-reply system prompt for a saved voice choice.
    pub fn interleaved_system_prompt_for(&self, saved: Option<&str>) -> &str {
        match self.voice_for(saved) {
            Some(v) => &v.interleaved_system_prompt,
            None => &self.interleaved_system_prompt,
        }
    }

    /// The profile for a model known by `identity` (its file or bundle name),
    /// from the built-in registry; [`AudioProfile::generic`] when none matches.
    pub fn for_model(identity: &str) -> Self {
        let tokens = tokens_of(identity);
        registry()
            .iter()
            .find(|e| e.matches(&tokens))
            .map(|e| e.profile.clone())
            .unwrap_or_else(Self::generic)
    }

    /// The profile for a loaded bundle: its own `audio_profile` if the manifest
    /// carries one, else the registry entry for its model and vocoder files.
    pub fn for_manifest(manifest: &Manifest) -> Self {
        if let Some(profile) = manifest.raw.get("audio_profile").and_then(parse_profile) {
            return profile;
        }
        let mut identity = manifest.files.model.clone();
        if let Some(decoder) = &manifest.files.audio_decoder {
            identity.push(' ');
            identity.push_str(decoder);
        }
        Self::for_model(&identity)
    }
}

/// Lowercased alphabetic runs of `identity`: `LFM2.5-Audio-1.5B-JP-Q8_0.gguf`
/// becomes `lfm audio b jp q gguf`. Matching on whole runs keeps `JPEG` and
/// `RJP3` from reading as the Japanese release.
fn tokens_of(identity: &str) -> Vec<String> {
    identity
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|t| !t.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

struct RegistryEntry {
    any_token: Vec<String>,
    all_tokens: Vec<String>,
    profile: AudioProfile,
}

impl RegistryEntry {
    fn matches(&self, tokens: &[String]) -> bool {
        let has = |t: &String| tokens.contains(t);
        (self.any_token.is_empty() || self.any_token.iter().any(has))
            && self.all_tokens.iter().all(has)
    }
}

/// First match wins, so the specific entries come before the broad ones.
fn registry() -> &'static [RegistryEntry] {
    static REGISTRY: OnceLock<Vec<RegistryEntry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let raw: serde_json::Value = serde_json::from_str(include_str!("audio_profiles.json"))
            .expect("audio_profiles.json is valid JSON (checked by a unit test)");
        raw.as_array()
            .map(|entries| entries.iter().filter_map(parse_entry).collect())
            .unwrap_or_default()
    })
}

fn string_list(v: Option<&serde_json::Value>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_ascii_lowercase))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_entry(v: &serde_json::Value) -> Option<RegistryEntry> {
    let any_token = string_list(v.get("any_token"));
    let all_tokens = string_list(v.get("all_tokens"));
    // An entry with no condition would match every model.
    if any_token.is_empty() && all_tokens.is_empty() {
        return None;
    }
    Some(RegistryEntry {
        any_token,
        all_tokens,
        profile: parse_profile(v)?,
    })
}

/// Parse `{ tts_system_prompt, interleaved_system_prompt?, voices?, sample_texts? }`.
/// `None` when the required prompt is missing, so a malformed manifest block
/// falls back to the registry instead of silencing the model.
fn parse_profile(v: &serde_json::Value) -> Option<AudioProfile> {
    let tts = v.get("tts_system_prompt")?.as_str()?.trim();
    if tts.is_empty() {
        return None;
    }
    let interleaved = v
        .get("interleaved_system_prompt")
        .and_then(|s| s.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(INTERLEAVED_SYSTEM_PROMPT);
    let voices: Vec<(String, String)> = v
        .get("voices")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|voice| {
                    let prompt = voice.get("prompt")?.as_str()?.trim();
                    let label = voice
                        .get("label")
                        .and_then(|l| l.as_str())
                        .unwrap_or(prompt);
                    (!prompt.is_empty()).then(|| (label.to_string(), prompt.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    let voice_refs: Vec<(&str, &str)> = voices
        .iter()
        .map(|(l, p)| (l.as_str(), p.as_str()))
        .collect();
    let samples = v
        .get("sample_texts")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Some(AudioProfile::new(tts, interleaved, &voice_refs, samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    fn manifest(model: &str, decoder: Option<&str>, extra: &str) -> Manifest {
        let decoder = decoder
            .map(|d| format!(r#","audio_decoder":"{d}""#))
            .unwrap_or_default();
        let json = format!(
            r#"{{"schema_version":"1.1.0","inference_type":"llama.cpp/lfm2-audio-v1",
               "load_time_parameters":{{"model":"{model}"{decoder}}}{extra}}}"#
        );
        Manifest::from_bytes(json.as_bytes()).expect("manifest")
    }

    #[test]
    fn the_registry_file_is_valid_and_every_entry_loads() {
        let raw: serde_json::Value = serde_json::from_str(include_str!("audio_profiles.json"))
            .expect("audio_profiles.json parses");
        let entries = raw.as_array().expect("an array of entries");
        assert_eq!(registry().len(), entries.len(), "an entry failed to load");
    }

    #[test]
    fn the_japanese_release_gets_its_model_card_prompt_and_no_voices() {
        for id in [
            "LFM2.5-Audio-1.5B-JP-GGUF",
            "LFM2.5-Audio-1.5B-JP-Q8_0.gguf vocoder-LFM2.5-Audio-1.5B-JP-Q8_0.gguf",
            "some-japanese-voice-model",
        ] {
            let p = AudioProfile::for_model(id);
            assert_eq!(p.tts_system_prompt, "Perform TTS in japanese.", "{id}");
            assert!(p.voices.is_empty(), "{id}");
            assert_eq!(p.sample_texts.len(), 4, "{id}");
            assert!(p.sample_texts[0].contains(MODEL_PLACEHOLDER), "{id}");
        }
    }

    #[test]
    fn the_english_release_keeps_its_four_voices() {
        let p = AudioProfile::for_model("LFM2.5-Audio-1.5B-Q4_0.gguf");
        let prompts: Vec<_> = p.voices.iter().map(|v| v.prompt.as_str()).collect();
        assert_eq!(
            prompts,
            [
                "Use the US female voice.",
                "Use the US male voice.",
                "Use the UK female voice.",
                "Use the UK male voice."
            ]
        );
        assert_eq!(p.tts_system_prompt, "Perform TTS. Use the US female voice.");
        assert_eq!(
            p.interleaved_system_prompt,
            "Respond with interleaved text and audio. Use the US female voice."
        );
    }

    #[test]
    fn jp_inside_a_longer_word_is_not_the_japanese_release() {
        for id in ["Vision-JPEG-1B", "RJP3-Audio", "jpx"] {
            assert_eq!(AudioProfile::for_model(id), AudioProfile::generic(), "{id}");
        }
    }

    #[test]
    fn an_unknown_model_gets_plain_tts_and_no_voices() {
        for id in ["", "SomeNewAudioModel-2B", "Qwen3-0.6B-Q8_0.gguf"] {
            let p = AudioProfile::for_model(id);
            assert_eq!(p.tts_system_prompt, "Perform TTS.", "{id}");
            assert_eq!(
                p.interleaved_system_prompt, INTERLEAVED_SYSTEM_PROMPT,
                "{id}"
            );
            assert!(p.voices.is_empty() && p.sample_texts.is_empty(), "{id}");
        }
    }

    #[test]
    fn a_voice_the_model_does_not_list_is_never_sent_to_it() {
        // The saved default of a client built for the English model.
        let saved = Some("Use the US female voice.");
        let jp = AudioProfile::for_model("LFM2.5-Audio-1.5B-JP-GGUF");
        assert_eq!(jp.tts_system_prompt_for(saved), "Perform TTS in japanese.");
        let unknown = AudioProfile::for_model("SomeNewAudioModel-2B");
        assert_eq!(unknown.tts_system_prompt_for(saved), "Perform TTS.");
    }

    #[test]
    fn a_listed_voice_is_used_and_an_unlisted_one_falls_back_to_the_default() {
        let en = AudioProfile::for_model("LFM2.5-Audio-1.5B-GGUF");
        assert_eq!(
            en.tts_system_prompt_for(Some("Use the UK male voice.")),
            "Perform TTS. Use the UK male voice."
        );
        assert_eq!(
            en.interleaved_system_prompt_for(Some(" Use the UK male voice. ")),
            "Respond with interleaved text and audio. Use the UK male voice."
        );
        for unlisted in [Some("Use the Martian voice."), Some("  "), Some(""), None] {
            assert_eq!(
                en.tts_system_prompt_for(unlisted),
                "Perform TTS. Use the US female voice.",
                "{unlisted:?}"
            );
        }
    }

    #[test]
    fn a_manifest_describes_its_own_bundle() {
        let m = manifest(
            "SomeNewAudioModel-2B-Q4_0.gguf",
            None,
            r#","audio_profile":{
                "tts_system_prompt":"Speak.",
                "voices":[{"label":"Ana","prompt":"Voice: ana."},{"prompt":"Voice: bo."}],
                "sample_texts":["Hi {model}."]}"#,
        );
        let p = AudioProfile::for_manifest(&m);
        assert_eq!(p.tts_system_prompt, "Speak. Voice: ana.");
        assert_eq!(
            p.voices[1].label, "Voice: bo.",
            "label defaults to the prompt"
        );
        assert_eq!(p.sample_texts, ["Hi {model}."]);
        assert_eq!(
            p.interleaved_system_prompt_for(Some("Voice: bo.")),
            "Respond with interleaved text and audio. Voice: bo."
        );
    }

    #[test]
    fn a_manifest_profile_beats_the_registry_but_a_malformed_one_does_not() {
        let jp_file = "LFM2.5-Audio-1.5B-JP-Q8_0.gguf";
        let own = manifest(
            jp_file,
            None,
            r#","audio_profile":{"tts_system_prompt":"Mine."}"#,
        );
        assert_eq!(AudioProfile::for_manifest(&own).tts_system_prompt, "Mine.");
        for bad in [
            r#","audio_profile":{}"#,
            r#","audio_profile":{"tts_system_prompt":"  "}"#,
            r#","audio_profile":"nope""#,
        ] {
            let m = manifest(jp_file, None, bad);
            assert_eq!(
                AudioProfile::for_manifest(&m).tts_system_prompt,
                "Perform TTS in japanese.",
                "{bad}"
            );
        }
    }

    #[test]
    fn malformed_optional_fields_are_dropped_not_fatal() {
        let m = manifest(
            "SomeNewAudioModel-2B-Q4_0.gguf",
            None,
            r#","audio_profile":{
                "tts_system_prompt":"Speak.",
                "interleaved_system_prompt":42,
                "voices":[{"prompt":7},{"label":"No prompt"},"x",{"prompt":"Voice: ana."}],
                "sample_texts":[42,"Hi.",null]}"#,
        );
        let p = AudioProfile::for_manifest(&m);
        assert_eq!(p.tts_system_prompt, "Speak. Voice: ana.");
        assert_eq!(p.voices.len(), 1, "only the voice with a text prompt stays");
        assert_eq!(p.sample_texts, ["Hi."]);
        assert_eq!(
            p.interleaved_system_prompt_for(None),
            "Respond with interleaved text and audio. Voice: ana.",
            "a non-string interleaved prompt falls back to the default"
        );
    }

    #[test]
    fn a_manifest_without_a_profile_resolves_from_its_file_names() {
        let en = manifest(
            "LFM2.5-Audio-1.5B-Q4_0.gguf",
            Some("vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"),
            "",
        );
        assert_eq!(AudioProfile::for_manifest(&en).voices.len(), 4);
        // The vocoder name alone can identify the release.
        let jp = manifest(
            "model.gguf",
            Some("vocoder-LFM2.5-Audio-1.5B-JP-Q8_0.gguf"),
            "",
        );
        assert_eq!(
            AudioProfile::for_manifest(&jp).tts_system_prompt,
            "Perform TTS in japanese."
        );
    }
}

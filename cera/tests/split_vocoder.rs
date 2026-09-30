//! Split-layout vocoder merge against the real llama.cpp release files.
//!
//! Fixtures live under `~/.leap/models/split-vocoder-fixtures/{en,jp}` (see the
//! `vocoder-*` / `tokenizer-*` files of `LiquidAI/LFM2.5-Audio-1.5B[-JP]-GGUF`).
//! A missing fixture skips with a message, which is a *pass*: run with
//! `CERA_REQUIRE_SPLIT_VOCODER_FIXTURES=1` to make absence a failure.

use std::path::PathBuf;
use std::sync::Arc;

use cera::audio_engine::InterleaveCadence;
use cera::gguf::GgufFile;
use cera::model::audio_decoder::{AudioDecoderWeights, DetokenizerWeights};
use cera::model::split_vocoder::{is_split_vocoder, merge_split_vocoder};

fn fixture(lang: &str, file: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".leap/models/split-vocoder-fixtures")
        .join(lang)
        .join(file);
    if p.exists() {
        return Some(p);
    }
    assert!(
        std::env::var_os("CERA_REQUIRE_SPLIT_VOCODER_FIXTURES").is_none(),
        "missing fixture {}",
        p.display()
    );
    eprintln!("skipping: {} not found", p.display());
    None
}

fn merged(lang: &str, stem: &str) -> Option<(Arc<GgufFile>, Arc<GgufFile>, Arc<GgufFile>)> {
    let voc = GgufFile::open_arc(&fixture(lang, &format!("vocoder-{stem}.gguf"))?).unwrap();
    let tok = GgufFile::open_arc(&fixture(lang, &format!("tokenizer-{stem}.gguf"))?).unwrap();
    assert!(is_split_vocoder(&voc), "llama.cpp vocoder should be split");
    let bytes = merge_split_vocoder(&voc, &tok).expect("merge");
    let m = Arc::new(GgufFile::from_bytes(bytes.into()).unwrap());
    Some((voc, tok, m))
}

#[test]
fn merged_jp_vocoder_loads_as_detokenizer_and_depthformer() {
    let Some((_, _, m)) = merged("jp", "LFM2.5-Audio-1.5B-JP-Q4_0") else {
        return;
    };
    assert!(!is_split_vocoder(&m));
    let detok = DetokenizerWeights::from_gguf(&m).expect("detokenizer");
    assert_eq!(detok.config.n_embd, 512);
    assert_eq!(detok.layers.len(), 8);
    let dec = AudioDecoderWeights::from_gguf(&m).expect("depthformer");
    // The JP vocoder declares its own interleave cadence, and the merge keeps it.
    assert_eq!(dec.interleave, InterleaveCadence { text: 6, audio: 9 });
}

/// The EN llama.cpp files and the LEAP-merged EN vocoder describe the same
/// model, so every renamed sidecar tensor must match the merged file's `lfm.*`
/// tensor of the same name. F32 tensors (norms, conv kernels) must be
/// byte-identical. The Q4_0 projections were quantized separately in the LEAP
/// file, so they agree to within quantization noise (relative RMS well under
/// 0.2) but not bytewise. A mis-mapped tensor (say `w1` and `w3` swapped) is
/// an unrelated matrix of similar scale and lands near 1.4, so the bound still
/// pins the rename table.
#[test]
fn merged_en_matches_leap_merged_vocoder() {
    let Some((_, _, m)) = merged("en", "LFM2.5-Audio-1.5B-Q4_0") else {
        return;
    };
    let leap = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !leap.exists() {
        assert!(
            std::env::var_os("CERA_REQUIRE_SPLIT_VOCODER_FIXTURES").is_none(),
            "missing fixture {}",
            leap.display()
        );
        eprintln!("skipping: {} not found", leap.display());
        return;
    }
    let leap = GgufFile::open_arc(&leap).unwrap();

    let (mut exact, mut quantized) = (0, 0);
    for (name, a) in &m.tensors {
        if !name.starts_with("lfm.") && !name.starts_with("lin.") {
            continue;
        }
        let b = leap
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` missing from LEAP vocoder"));
        assert_eq!(a.shape, b.shape, "{name} shape");
        assert_eq!(a.ggml_type_id, b.ggml_type_id, "{name} type");
        if m.tensor_data(name).unwrap() == leap.tensor_data(name).unwrap() {
            exact += 1;
            continue;
        }
        assert_ne!(
            a.ggml_type_id, 0,
            "{name}: F32 tensor must be byte-identical"
        );
        let (x, y) = (
            m.get_tensor(name).unwrap().to_f32_vec(),
            leap.get_tensor(name).unwrap().to_f32_vec(),
        );
        let err: f64 = x
            .iter()
            .zip(&y)
            .map(|(p, q)| ((p - q) as f64).powi(2))
            .sum();
        let mag: f64 = y.iter().map(|q| (*q as f64).powi(2)).sum();
        let rel = (err / mag).sqrt();
        assert!(
            rel < 0.2,
            "{name}: relative RMS {rel:.3} looks like a different tensor"
        );
        quantized += 1;
    }
    // Every sidecar tensor except `token_embd.weight` (77 - 1).
    assert_eq!(exact + quantized, 76, "exact={exact} quantized={quantized}");
    eprintln!("EN merged vs LEAP: {exact} identical, {quantized} within quantization noise");

    DetokenizerWeights::from_gguf(&m).expect("detokenizer");
    let dec = AudioDecoderWeights::from_gguf(&m).expect("depthformer");
    // The EN vocoder declares no cadence, so it keeps the 6/12 defaults.
    assert_eq!(dec.interleave, InterleaveCadence::default());
}

/// The path the apps take: a LeapBundles-style manifest whose `audio_tokenizer`
/// is empty and whose vocoder is the split half. Resolution must attach the
/// sibling `tokenizer-*.gguf` so the audio loaders can merge it.
#[test]
fn engine_attaches_the_sibling_tokenizer_for_an_empty_manifest_slot() {
    let Some(manifest) = fixture("jp8", "LFM2.5-Audio-1.5B-JP-Q8_0.json") else {
        return;
    };
    let engine =
        cera::engine::CeraEngine::from_path(&manifest, cera::engine::EngineConfig::default())
            .expect("engine loads");
    let tok = engine
        .manifest()
        .files
        .audio_tokenizer
        .as_deref()
        .expect("sibling tokenizer attached");
    assert!(
        tok.ends_with("tokenizer-LFM2.5-Audio-1.5B-JP-Q8_0.gguf"),
        "unexpected sidecar: {tok}"
    );
}

fn read_wav16(path: &std::path::Path) -> (Vec<f32>, u32) {
    let b = std::fs::read(path).unwrap();
    let sr = u32::from_le_bytes(b[24..28].try_into().unwrap());
    let d = b.windows(4).position(|w| w == b"data").unwrap();
    let n = u32::from_le_bytes(b[d + 4..d + 8].try_into().unwrap()) as usize;
    let pcm = b[d + 8..d + 8 + n]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
        .collect();
    (pcm, sr)
}

/// Regression for the forced-interleave bug. The bundle carries a vocoder, and
/// `Session::generate` used to cut every answer into text/audio rounds after 6
/// tokens, so ASR was right for six tokens and junk (and different on every run)
/// after. Real Japanese speech (macOS `Kyoko`), the model's own prompt, greedy.
#[test]
fn jp_asr_with_the_vocoder_attached_is_correct_and_deterministic() {
    use cera::engine::CeraEngine;
    use cera::session::{AudioOutputMode, FinishReason, GenerateOpts, ModalitySink};
    use cera::tokenizer::{ChatMessage, apply_chat_template};

    struct Sink(Vec<u32>);
    impl ModalitySink for Sink {
        fn on_text_tokens(&mut self, t: &[u32]) {
            self.0.extend_from_slice(t);
        }
        fn on_done(&mut self, _: FinishReason) {}
    }

    let Some(manifest) = fixture("jp8", "LFM2.5-Audio-1.5B-JP-Q8_0.json") else {
        return;
    };
    let dir = manifest.parent().unwrap().to_path_buf();
    let engine = CeraEngine::from_path(&manifest, cera::engine::EngineConfig::default()).unwrap();
    let tok = engine.tokenizer();
    let (marker_id, marker) = CeraEngine::AUDIO_MARKER_CANDIDATES
        .into_iter()
        .find_map(|n| tok.special_token_id(n).map(|id| (id, n)))
        .unwrap();
    let msgs = [
        ChatMessage {
            role: "system".into(),
            content: "Perform ASR in japanese.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: marker.into(),
        },
    ];
    let toks = tok.encode(&apply_chat_template(tok, &msgs, true).unwrap());
    let split = toks.iter().position(|&t| t == marker_id).unwrap();

    let cases = [
        ("kyoko1.wav", "こんにちは。ご元気ですか。"),
        ("kyoko2.wav", "今日はいい天気ですね。"),
        (
            "kyoko3.wav",
            "東京は日本の首都で、たくさんの人が住んでいます。",
        ),
    ];
    // The default mode is what an unaware caller gets; TextOnly is what
    // `transcribe` pins. Both must be right, and repeatable.
    for mode in [AudioOutputMode::Sequential, AudioOutputMode::TextOnly] {
        for (file, want) in cases {
            let (pcm, sr) = read_wav16(&dir.join(file));
            for run in 0..2 {
                let mut s = engine.new_session(cera::SessionConfig::default()).unwrap();
                s.append_tokens(&toks[..split]).unwrap();
                s.append_audio(&pcm, sr).unwrap();
                s.append_tokens(&toks[split + 1..]).unwrap();
                let mut sink = Sink(vec![]);
                let opts = GenerateOpts {
                    temperature: 0.0,
                    audio_mode: mode,
                    ..Default::default()
                };
                s.generate(&opts, &mut sink).unwrap();
                assert_eq!(
                    tok.decode(&sink.0).trim(),
                    want,
                    "{mode:?} {file} run {run}"
                );
            }
        }
    }
}

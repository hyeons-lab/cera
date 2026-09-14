use super::*;
use crate::model::audio_decoder::{AudioDecoderWeights, DetokenizerWeights};
use crate::model::audio_encoder::{AudioEncoderWeights, encode_audio_pcm};

#[derive(Default)]
struct Sink {
    text: Vec<u32>,
    pcm: Vec<f32>,
    done: Vec<crate::FinishReason>,
}

impl ModalitySink for Sink {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.text.extend_from_slice(tokens);
    }
    fn on_audio_frames(&mut self, pcm: &[f32], sample_rate: u32) {
        assert_eq!(sample_rate, 24_000);
        self.pcm.extend_from_slice(pcm);
    }
    fn on_done(&mut self, reason: crate::FinishReason) {
        self.done.push(reason);
    }
}

fn render(session: &mut Session) -> Sink {
    session.append_tokens(&[0, 1]).unwrap();
    let mut sink = Sink::default();
    let summary = session
        .generate(
            &GenerateOpts {
                temperature: 0.0,
                max_tokens: 6,
                ignore_eos: true,
                ..GenerateOpts::default()
            },
            &mut sink,
        )
        .unwrap();
    assert_eq!(summary.tokens_generated, 6);
    assert_eq!(sink.text.len(), 6);
    assert_eq!(sink.done, [crate::FinishReason::MaxTokens]);
    assert_eq!(sink.pcm.len(), 22_560);
    assert!(sink.pcm.iter().all(|v| v.is_finite()));
    sink
}

fn gguf(bytes: Arc<[u8]>) -> Arc<GgufFile> {
    Arc::new(GgufFile::from_bytes(bytes).unwrap())
}

fn control(full: bool) -> Session {
    let mut session = CeraEngine::from_bytes(
        fixture::vision_primary(),
        LoadConfig {
            context_size: 256,
            ..cpu_config()
        },
    )
    .unwrap()
    .new_session(SessionConfig::default())
    .unwrap();
    session.attach_vocoder(
        Arc::new(
            AudioDecoderWeights::from_gguf(&gguf(audio_fixture::vocoder(7, 32, true, true)))
                .unwrap(),
        ),
        Arc::new(
            DetokenizerWeights::from_gguf(&gguf(if full {
                audio_fixture::vocoder(7, 32, true, true)
            } else {
                audio_fixture::vocoder(29, 32, false, true)
            }))
            .unwrap(),
        ),
    );
    session
}

fn files(full: bool) -> Vec<(&'static str, Arc<[u8]>)> {
    vec![
        ("model-Q4_K_M.gguf", fixture::vision_primary()),
        ("mmproj-Q4_K_M.gguf", audio_fixture::encoder(7, 32)),
        (
            "audio_decoder-Q4_K_M.gguf",
            audio_fixture::vocoder(7, 32, true, full),
        ),
        ("audio_decoder-Q8_0.gguf", header("unused-decoder")),
        (
            "tokenizer-Q4_K_M.gguf",
            audio_fixture::vocoder(29, 32, false, true),
        ),
        (
            "vocoder-Q4_K_M.gguf",
            audio_fixture::vocoder(29, 32, true, true),
        ),
    ]
}

#[test]
fn hf_audio_executes_selected_encoder_decoder_and_detokenizer_after_release() {
    isolated_with_routes(
        "companions::audio::hf_audio_executes_selected_encoder_decoder_and_detokenizer_after_release",
        || {
            let mut all = routes("audio-full", "automatic-speech-recognition", &files(true));
            all.extend(routes(
                "audio-split",
                "automatic-speech-recognition",
                &files(false),
            ));
            all
        },
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = cfg(&ctx.root, &progress);
            let repo = cfg.bundle_repo.as_ref().unwrap();
            let pcm: Vec<_> = (0..1600).map(|i| (i as f32 * 0.13).sin() * 0.2).collect();
            let encoder =
                AudioEncoderWeights::from_gguf(&gguf(audio_fixture::encoder(7, 32))).unwrap();
            let (embeddings, frames) = encode_audio_pcm(&pcm, &encoder);
            assert_eq!(frames, 2);
            distinct(
                &embeddings,
                &encode_audio_pcm(&vec![0.0; pcm.len()], &encoder).0,
            );
            for (repo_name, full) in [("audio-full", true), ("audio-split", false)] {
                let mut reference = control(full);
                reference.append_embeddings(&embeddings, frames).unwrap();
                let input_logits = reference.last_logits().unwrap().to_vec();
                let expected = render(&mut reference);
                assert!(reference.last_logits().is_none());
                reference.append_tokens(&[1]).unwrap();
                let mut other = control(!full);
                other.append_embeddings(&embeddings, frames).unwrap();
                distinct(&expected.pcm, &render(&mut other).pcm);
                for engine in hf_engines(&format!("fixture/{repo_name}:Q4_K_M"), &cfg) {
                    assert!(engine.capabilities().audio_in && engine.capabilities().audio_out);
                    let resolved = &engine.manifest().files;
                    for (actual, file) in [
                        (&resolved.multimodal_projector, "mmproj-Q4_K_M.gguf"),
                        (&resolved.audio_decoder, "audio_decoder-Q4_K_M.gguf"),
                        (&resolved.audio_tokenizer, "tokenizer-Q4_K_M.gguf"),
                    ] {
                        let url = format!(
                            "{}/fixture/{repo_name}/resolve/{MAIN_COMMIT}/{file}",
                            ctx.url
                        );
                        assert_eq!(
                            Path::new(actual.as_ref().unwrap()),
                            repo.fixture_path(&url).unwrap()
                        );
                    }
                    let mut session = engine.new_session(SessionConfig::default()).unwrap();
                    drop(engine);
                    session.append_audio(&pcm, 16_000).unwrap();
                    assert_eq!(session.position(), 2);
                    close(session.last_logits().unwrap(), &input_logits);
                    let actual = render(&mut session);
                    assert_eq!(actual.text, expected.text);
                    close(&actual.pcm, &expected.pcm);
                    assert_eq!(session.position(), 22);
                    assert!(session.last_logits().is_none());
                    session.append_tokens(&[1]).unwrap();
                    assert_eq!(session.position(), reference.position());
                    close(
                        session.last_logits().unwrap(),
                        reference.last_logits().unwrap(),
                    );
                }
                for (file, bytes) in files(full) {
                    if file == "audio_decoder-Q8_0.gguf" {
                        continue;
                    }
                    let url = format!(
                        "{}/fixture/{repo_name}/resolve/{MAIN_COMMIT}/{file}",
                        ctx.url
                    );
                    assert_cached(repo, &progress, &url, &bytes);
                }
            }
        },
        |requests| {
            for repo in ["audio-full", "audio-split"] {
                for (file, expected) in [
                    ("model-Q4_K_M.gguf", 1),
                    ("mmproj-Q4_K_M.gguf", 1),
                    ("audio_decoder-Q4_K_M.gguf", 1),
                    ("audio_decoder-Q8_0.gguf", 0),
                    ("tokenizer-Q4_K_M.gguf", 1),
                    ("vocoder-Q4_K_M.gguf", 1),
                ] {
                    assert_eq!(
                        count(
                            requests,
                            "GET",
                            &format!("/fixture/{repo}/resolve/{MAIN_COMMIT}/{file}")
                        ),
                        expected
                    );
                }
            }
        },
    );
}

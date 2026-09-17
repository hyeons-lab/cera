use super::*;

fn hidden() -> Vec<f32> {
    (0..64).map(|i| (i as f32 * 0.21).cos() * 0.2).collect()
}

fn spectrum(dec: &AudioDecoderWeights, detok: &DetokenizerWeights) -> Vec<f32> {
    detokenize_to_spectrum(
        detok,
        dec,
        &mut DetokenizerState::new(&detok.config),
        &[0; 8],
    )
}

fn check(
    engine: CeraEngine,
    expected_decoder: Option<&Arc<[u8]>>,
    expected_detok: Option<&Arc<[u8]>>,
) {
    assert_eq!(engine.audio_decoder.is_some(), expected_decoder.is_some());
    assert_eq!(engine.detok_weights.is_some(), expected_detok.is_some());
    assert!(!engine.has_gpu_audio_decoder());
    if let Some(bytes) = expected_decoder {
        let expected = decoder(bytes.clone());
        let actual = engine.audio_decoder.as_ref().unwrap();
        let actual_hidden = depthformer_forward(
            actual,
            &mut DepthformerState::new(&actual.depthformer_config),
            &hidden(),
        );
        let expected_hidden = depthformer_forward(
            &expected,
            &mut DepthformerState::new(&expected.depthformer_config),
            &hidden(),
        );
        close(&actual_hidden, &expected_hidden);
        assert_eq!(
            sample_audio_frame(
                actual,
                &mut DepthformerState::new(&actual.depthformer_config),
                &[0.2; 32],
                0.0,
                1
            ),
            [0; 8]
        );
        close(
            &embed_audio_token(actual, &[0; 8]),
            &embed_audio_token(&expected, &[0; 8]),
        );
        if let Some(bytes) = expected_detok {
            let expected = detok(bytes.clone());
            let actual_spectrum = spectrum(actual, engine.detok_weights.as_ref().unwrap());
            let expected_spectrum = spectrum(actual, &expected);
            assert_eq!(actual_spectrum.len(), 6 * 641 * 2);
            close(&actual_spectrum, &expected_spectrum);
            close(
                &istft_to_pcm(&actual_spectrum, 1280, 320),
                &istft_to_pcm(&expected_spectrum, 1280, 320),
            );
        }
    }
    let mut session = engine.new_session(SessionConfig::default()).unwrap();
    drop(engine);
    // Three tokens do not trigger the six-token interleaved audio budget.
    assert_eq!(generate(&mut session), generate(&mut control()));
}

#[test]
fn distinct_decoder_and_detokenizer_fixtures_execute_real_computation() {
    let a = decoder(fixture::vocoder(7, 32, true, true));
    let b = decoder(fixture::vocoder(29, 32, true, true));
    distinct(
        &depthformer_forward(
            &a,
            &mut DepthformerState::new(&a.depthformer_config),
            &hidden(),
        ),
        &depthformer_forward(
            &b,
            &mut DepthformerState::new(&b.depthformer_config),
            &hidden(),
        ),
    );
    distinct(
        &embed_audio_token(&a, &[0; 8]),
        &embed_audio_token(&b, &[0; 8]),
    );
    let da = detok(fixture::vocoder(7, 32, true, true));
    let db = detok(fixture::vocoder(29, 32, true, true));
    distinct(&spectrum(&a, &da), &spectrum(&a, &db));
    distinct(
        &istft_to_pcm(&spectrum(&a, &da), 1280, 320),
        &istft_to_pcm(&spectrum(&a, &db), 1280, 320),
    );
}

#[test]
fn decoder_and_detokenizer_precedence_preserves_source_specific_fallbacks() {
    let full = fixture::vocoder(7, 32, true, true);
    let alternative = fixture::vocoder(29, 32, true, true);
    let dec_only = fixture::vocoder(7, 32, true, false);
    let detok_only = fixture::vocoder(29, 32, false, true);
    let mismatch = fixture::vocoder(7, 64, true, true);
    let bad: Arc<[u8]> = Arc::from(&b"bad GGUF"[..]);
    let header = header("audio-fixture");
    // vocoder, tokenizer, projector, memory decoder/detok, file decoder/detok.
    let cases = [
        (None, None, None, None, None, None, None),
        (
            Some(&full),
            Some(&alternative),
            Some(&alternative),
            Some(&full),
            Some(&full),
            Some(&full),
            Some(&full),
        ),
        (
            Some(&dec_only),
            Some(&detok_only),
            Some(&full),
            Some(&dec_only),
            Some(&detok_only),
            Some(&dec_only),
            Some(&detok_only),
        ),
        (
            Some(&dec_only),
            Some(&header),
            Some(&full),
            Some(&dec_only),
            Some(&full),
            Some(&dec_only),
            None,
        ),
        (
            Some(&dec_only),
            Some(&bad),
            Some(&full),
            Some(&dec_only),
            Some(&full),
            Some(&dec_only),
            None,
        ),
        (
            None,
            Some(&alternative),
            Some(&full),
            Some(&full),
            Some(&alternative),
            None,
            None,
        ),
        (
            Some(&bad),
            Some(&header),
            Some(&full),
            Some(&full),
            Some(&full),
            None,
            None,
        ),
        // A parsed but structurally invalid vocoder blocks decoder fallback.
        (
            Some(&header),
            Some(&alternative),
            Some(&full),
            None,
            None,
            None,
            None,
        ),
        (
            Some(&detok_only),
            Some(&alternative),
            Some(&full),
            None,
            None,
            None,
            None,
        ),
        (
            Some(&mismatch),
            Some(&alternative),
            Some(&full),
            None,
            None,
            None,
            None,
        ),
        (
            Some(&dec_only),
            None,
            None,
            Some(&dec_only),
            None,
            Some(&dec_only),
            None,
        ),
        (
            None,
            None,
            Some(&full),
            Some(&full),
            Some(&full),
            None,
            None,
        ),
    ];
    for (voc, tok, projector, mem_dec, mem_detok, file_dec, file_detok) in cases {
        let mut source = parts();
        source.audio_decoder = voc.cloned();
        source.audio_tokenizer = tok.cloned();
        source.multimodal_projector = projector.cloned();
        for engine in parts_engines(source.clone(), config()) {
            check(engine, mem_dec, mem_detok);
        }
        #[cfg(feature = "mmap")]
        {
            let (engines, _directory) = paths(&source, false);
            for engine in engines {
                check(engine, file_dec, file_detok);
            }
        }
        #[cfg(not(feature = "mmap"))]
        let _ = (file_dec, file_detok);
    }
    // Byte vocoders attach even under explicit text, whereas filesystem audio
    // loading is gated by the audio inference declaration.
    let mut source = parts();
    source.inference_type = Some(InferenceType::LlamaCppTextToText);
    source.audio_decoder = Some(full.clone());
    for engine in parts_engines(source.clone(), config()) {
        assert!(!engine.capabilities().audio_out);
        check(engine, Some(&full), Some(&full));
    }
    #[cfg(feature = "mmap")]
    {
        let (engines, _directory) = paths(&source, false);
        for engine in engines {
            check(engine, None, None);
        }
        source.inference_type = Some(InferenceType::LlamaCppLfm2AudioV1);
        let (engines, _directory) = paths(&source, true);
        for engine in engines {
            assert!(engine.capabilities().audio_out);
            check(engine, None, None);
        }
    }
}

#[derive(Default)]
struct AudioSink {
    text: Vec<u32>,
    pcm: Vec<f32>,
    done: Vec<crate::FinishReason>,
}

impl crate::ModalitySink for AudioSink {
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

fn render(session: &mut crate::Session, prefix: &[u32]) -> AudioSink {
    session.append_tokens(prefix).unwrap();
    let mut sink = AudioSink::default();
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
    assert!(!sink.pcm.is_empty());
    assert!(sink.pcm.iter().all(|v| v.is_finite()));
    sink
}

fn output_control(bytes: &Arc<[u8]>) -> crate::Session {
    let mut session = control();
    session.attach_vocoder(
        Arc::new(decoder(bytes.clone())),
        Arc::new(detok(bytes.clone())),
    );
    session
}

#[test]
fn retained_sessions_execute_attached_vocoders_and_reset_independently() {
    let full = fixture::vocoder(7, 32, true, true);
    let mut ca = output_control(&full);
    let mut cb = output_control(&full);
    let expected_a = render(&mut ca, &[0, 1]);
    let expected_b = render(&mut cb, &[1, 0, 0]);
    let other = render(
        &mut output_control(&fixture::vocoder(29, 32, true, true)),
        &[0, 1],
    );
    distinct(&expected_a.pcm, &other.pcm);
    let mut source = parts();
    source.audio_decoder = Some(full);
    let (engines, _directory) = all_engines(source);
    for engine in engines {
        let mut a = engine.new_session(SessionConfig::default()).unwrap();
        let mut b = engine.new_session(SessionConfig::default()).unwrap();
        drop(engine);
        let actual_a = render(&mut a, &[0, 1]);
        assert_eq!(actual_a.text, expected_a.text);
        close(&actual_a.pcm, &expected_a.pcm);
        assert_eq!(a.position(), ca.position());
        a.reset().unwrap();
        let actual_b = render(&mut b, &[1, 0, 0]);
        assert_eq!(actual_b.text, expected_b.text);
        close(&actual_b.pcm, &expected_b.pcm);
        assert_eq!(b.position(), cb.position());
        close(&render(&mut a, &[0, 1]).pcm, &expected_a.pcm);
    }
}

#[test]
fn detokenizer_state_depends_on_prior_frames_and_resets_independently() {
    let weights = detok(fixture::vocoder(7, 32, true, true));
    let dec = decoder(fixture::vocoder(7, 32, true, true));
    let mut a = DetokenizerState::new(&weights.config);
    let mut b = DetokenizerState::new(&weights.config);
    let mut ca = DetokenizerState::new(&weights.config);
    let mut cb = DetokenizerState::new(&weights.config);
    for (state, codes) in [
        (&mut a, [0; 8]),
        (&mut ca, [0; 8]),
        (&mut b, [1; 8]),
        (&mut cb, [1; 8]),
    ] {
        let first = detokenize_to_spectrum(&weights, &dec, state, &codes);
        assert!(first.iter().all(|v| v.is_finite()));
    }
    // Equal history lengths and identical current codes isolate retained state.
    let a_next = detokenize_to_spectrum(&weights, &dec, &mut a, &[0; 8]);
    let b_next = detokenize_to_spectrum(&weights, &dec, &mut b, &[0; 8]);
    distinct(&a_next, &b_next);
    close(
        &a_next,
        &detokenize_to_spectrum(&weights, &dec, &mut ca, &[0; 8]),
    );
    close(
        &b_next,
        &detokenize_to_spectrum(&weights, &dec, &mut cb, &[0; 8]),
    );
    a.reset();
    close(
        &detokenize_to_spectrum(&weights, &dec, &mut a, &[0; 8]),
        &spectrum(&dec, &weights),
    );
    close(
        &detokenize_to_spectrum(&weights, &dec, &mut b, &[1; 8]),
        &detokenize_to_spectrum(&weights, &dec, &mut cb, &[1; 8]),
    );
}

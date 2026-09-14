use super::*;
use crate::kv_cache::{KvCacheConfig, KvCompression};
use crate::session::FinishReason;

#[cfg(feature = "disk-cache")]
mod anonymous;
#[cfg(all(feature = "disk-cache", feature = "mmap"))]
mod cache;
mod checked_rewind;
mod fixture;
#[cfg(all(
    not(target_arch = "wasm32"),
    any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    )
))]
mod gpu_recovery;
#[cfg(all(
    not(target_arch = "wasm32"),
    any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    )
))]
mod gpu_sessions;
#[cfg(all(
    not(target_arch = "wasm32"),
    any(
        feature = "gpu",
        all(feature = "metal", any(target_os = "macos", target_os = "ios"))
    )
))]
mod gpu_transcription;
#[cfg(all(feature = "disk-cache", feature = "mmap"))]
mod named;
mod recovery;

fn load(bytes: Arc<[u8]>) -> GenerativeModel {
    ModelLoader::new(ModelSource::bytes(bytes))
        .config(cpu_config())
        .build_generative()
        .unwrap()
}

fn session(model: &GenerativeModel, compression: KvCompression) -> Session {
    model
        .create_session(SessionConfig {
            kv_compression: compression,
            seed: Some(42),
            ..SessionConfig::default()
        })
        .unwrap()
}

#[track_caller]
fn close(actual: &[f32], expected: &[f32]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} != {b}");
    }
}

#[track_caller]
fn distinct(actual: &[f32], control: &[f32]) {
    assert_eq!(actual.len(), control.len());
    assert!(actual.iter().chain(control).all(|v| v.is_finite()));
    assert!(
        actual
            .iter()
            .zip(control)
            .any(|(a, b)| (a - b).abs() > 1e-3)
    );
}

#[track_caller]
fn same_live(actual: &Session, control: &Session) {
    assert_eq!(actual.position(), control.position());
    close(
        actual.last_logits().unwrap(),
        control.last_logits().unwrap(),
    );
}

#[derive(Default)]
struct Output {
    tokens: Vec<u32>,
    done: Vec<FinishReason>,
}

impl ModalitySink for Output {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
    }

    fn on_done(&mut self, reason: FinishReason) {
        self.done.push(reason);
    }
}

fn generate(session: &mut Session) -> Output {
    let mut output = Output::default();
    let summary = session
        .generate(
            &GenerateOpts {
                temperature: 0.0,
                max_tokens: 3,
                ignore_eos: true,
                ..GenerateOpts::default()
            },
            &mut output,
        )
        .unwrap();
    assert_eq!(summary.tokens_generated as usize, output.tokens.len());
    assert_eq!(output.done.len(), 1);
    // Greedy generation can intentionally discard last_logits. A subsequent
    // raw append observes the resulting KV through fresh numerical output.
    if summary.tokens_generated > 0 {
        session.append_tokens(&[1]).unwrap();
    }
    output
}

#[test]
fn retained_operations_keep_borrows_shared_handles_defaults_and_errors() {
    let mut parts = ModelBytes::text(fixture::tiny_dense());
    parts.chat_template = Some("retained override".into());
    parts.generation_defaults = Some(GenerationDefaults::Text {
        temperature: Some(0.25),
        top_p: Some(0.75),
        top_k: Some(7),
        min_p: Some(0.1),
        repetition_penalty: Some(1.2),
    });
    let legacy = CeraEngine::from_parts(parts.clone(), cpu_config()).unwrap();
    let model = ModelLoader::new(ModelSource::parts(parts))
        .config(cpu_config())
        .build_generative()
        .unwrap();
    assert!(std::ptr::eq(model.metadata(), model.engine.metadata()));
    assert!(std::ptr::eq(model.manifest(), model.engine.manifest()));
    assert!(std::ptr::eq(model.config(), model.engine.config()));
    assert!(std::ptr::eq(model.model(), model.engine.model()));
    assert!(std::ptr::eq(model.tokenizer(), model.engine.tokenizer()));
    assert_eq!(
        model.manifest().chat_template.as_deref(),
        Some("retained override")
    );
    let opts = model.default_generate_opts();
    let expected = legacy.default_generate_opts();
    assert_eq!(
        (
            opts.temperature,
            opts.top_p,
            opts.top_k,
            opts.min_p,
            opts.repetition_penalty
        ),
        (
            expected.temperature,
            expected.top_p,
            expected.top_k,
            expected.min_p,
            expected.repetition_penalty
        )
    );
    let caps = |c: crate::ModalityCapabilities| {
        (c.text_in, c.text_out, c.image_in, c.audio_in, c.audio_out)
    };
    assert_eq!(caps(model.capabilities()), caps(legacy.capabilities()));
    assert_eq!(
        model.transcribe(&[], 16_000).unwrap_err().to_string(),
        legacy.transcribe(&[], 16_000).unwrap_err().to_string()
    );
    let weights = model.model_arc();
    let tokenizer = model.tokenizer_arc();
    let clone = model.clone();
    assert!(Arc::ptr_eq(&weights, &clone.model_arc()));
    assert!(Arc::ptr_eq(&tokenizer, &clone.tokenizer_arc()));
    let caps = model.capabilities();
    drop(model);
    drop(clone);
    let mut raw = Session::new(weights, tokenizer, caps, SessionConfig::default()).unwrap();
    raw.append_tokens(&[0, 1]).unwrap();
    assert_eq!(generate(&mut raw).tokens.len(), 3);
}

#[test]
fn cpu_interleaving_reset_cancel_and_extraction_preserve_other_live_sessions() {
    for bytes in [fixture::tiny_dense(), fixture::tiny_hybrid()] {
        for compression in [KvCompression::None, KvCompression::F16] {
            let shared = load(bytes.clone());
            let mut a = session(&shared, compression.clone());
            let mut b = session(&shared.clone(), compression.clone());
            let mut control_a = session(&load(bytes.clone()), compression.clone());
            let mut control_b = session(&load(bytes.clone()), compression.clone());
            assert!(!Arc::ptr_eq(&a.cancel_handle(), &b.cancel_handle()));
            assert!(!Arc::ptr_eq(&a.position_handle(), &b.position_handle()));
            for s in [&mut a, &mut control_a] {
                s.append_tokens(&[0, 1]).unwrap();
            }
            for s in [&mut b, &mut control_b] {
                s.append_tokens(&[1, 0, 0]).unwrap();
            }
            distinct(a.last_logits().unwrap(), b.last_logits().unwrap());
            same_live(&a, &control_a);
            same_live(&b, &control_b);

            let query = [1, 0, 1];
            let hidden = a.hidden_states_for_tokens(&query).unwrap();
            // Keep continuation controls untouched by extraction: applying the
            // same destructive operation to both sides could conceal lost KV.
            let mut extraction_control = session(&load(bytes.clone()), compression.clone());
            close(
                &hidden,
                &extraction_control.hidden_states_for_tokens(&query).unwrap(),
            );
            assert_eq!(hidden.len(), query.len() * a.hidden_size());
            assert!(matches!(
                a.hidden_states_for_tokens(&[2]),
                Err(CeraError::InvalidToken { .. })
            ));
            same_live(&a, &control_a);
            same_live(&b, &control_b);

            a.cancel();
            let cancelled = generate(&mut a);
            assert!(cancelled.tokens.is_empty());
            assert_eq!(cancelled.done, vec![FinishReason::Cancelled]);
            same_live(&a, &control_a);
            same_live(&b, &control_b);
            for s in [&mut b, &mut control_b] {
                s.reset().unwrap();
                s.append_tokens(&[1, 1, 0]).unwrap();
            }
            // Prefix-cache controls must leave the live session state alone.
            shared.configure_cache(KvCacheConfig::default());
            shared.clear_warm_cache();
            shared.clear_cache();
            drop(shared);
            for s in [&mut a, &mut control_a] {
                s.append_tokens(&[0]).unwrap();
            }
            same_live(&a, &control_a);
            // The same final token without its history must produce different
            // logits, otherwise a lost KV history could pass this fixture.
            let mut fresh = session(&load(bytes.clone()), compression.clone());
            fresh.append_tokens(&[0]).unwrap();
            distinct(a.last_logits().unwrap(), fresh.last_logits().unwrap());
            same_live(&b, &control_b);
            let output_a = generate(&mut a);
            let output_b = generate(&mut b);
            assert_eq!(output_a.tokens, generate(&mut control_a).tokens);
            assert_eq!(output_b.tokens, generate(&mut control_b).tokens);
            assert_eq!(output_a.tokens.len(), 3);
            assert_eq!(output_b.tokens.len(), 3);
            same_live(&a, &control_a);
            same_live(&b, &control_b);
        }
    }
}

#[test]
fn cpu_parallel_sessions_match_separately_loaded_controls() {
    for bytes in [fixture::tiny_dense(), fixture::tiny_hybrid()] {
        let shared = load(bytes.clone());
        let a = session(&shared, KvCompression::None);
        let b = session(&shared, KvCompression::None);
        drop(shared);
        let run = |mut session: Session, prompt: &[u32]| {
            session.append_tokens(prompt).unwrap();
            let before = session.last_logits().unwrap().to_vec();
            let output = generate(&mut session);
            (
                before,
                output.tokens,
                session.last_logits().unwrap().to_vec(),
            )
        };
        let barrier = std::sync::Barrier::new(2);
        let (a, b) = std::thread::scope(|scope| {
            let barrier = &barrier;
            let a = scope.spawn(move || {
                barrier.wait();
                run(a, &[0, 1])
            });
            let b = scope.spawn(move || {
                barrier.wait();
                run(b, &[1, 0, 1])
            });
            (a.join().unwrap(), b.join().unwrap())
        });
        for (actual, prompt) in [(a, &[0, 1][..]), (b, &[1, 0, 1][..])] {
            let expected = run(session(&load(bytes.clone()), KvCompression::None), prompt);
            close(&actual.0, &expected.0);
            assert_eq!(actual.1, expected.1);
            assert_eq!(actual.1.len(), 3);
            close(&actual.2, &expected.2);
        }
    }
}

#[test]
fn cpu_adapter_and_embedding_scratch_are_session_local() {
    for bytes in [fixture::tiny_dense(), fixture::tiny_hybrid()] {
        let shared = load(bytes.clone());
        let mut adapted = session(&shared, KvCompression::None);
        let mut base = session(&shared, KvCompression::None);
        let mut expected_adapted = session(&load(bytes.clone()), KvCompression::None);
        let mut expected_base = session(&load(bytes.clone()), KvCompression::None);
        let adapter = fixture::adapter(32);
        for s in [&mut adapted, &mut expected_adapted] {
            s.attach_lora_adapters(adapter.clone()).unwrap();
            s.append_tokens(&[0, 1]).unwrap();
        }
        for s in [&mut base, &mut expected_base] {
            s.append_tokens(&[0, 1]).unwrap();
        }
        distinct(adapted.last_logits().unwrap(), base.last_logits().unwrap());
        let query = [1, 0, 1];
        let adapted_hidden = adapted.hidden_states_for_tokens(&query).unwrap();
        let base_hidden = base.hidden_states_for_tokens(&query).unwrap();
        distinct(&adapted_hidden, &base_hidden);
        let mut hidden_adapted_control = session(&load(bytes.clone()), KvCompression::None);
        hidden_adapted_control
            .attach_lora_adapters(adapter)
            .unwrap();
        let mut hidden_base_control = session(&load(bytes), KvCompression::None);
        close(
            &adapted_hidden,
            &hidden_adapted_control
                .hidden_states_for_tokens(&query)
                .unwrap(),
        );
        close(
            &base_hidden,
            &hidden_base_control
                .hidden_states_for_tokens(&query)
                .unwrap(),
        );
        assert!(matches!(
            adapted.attach_lora_adapters(fixture::adapter(16)),
            Err(CeraError::LoraDimMismatch(_))
        ));
        assert!(adapted.has_lora_adapters());
        assert!(!base.has_lora_adapters());
        same_live(&adapted, &expected_adapted);
        same_live(&base, &expected_base);
        assert_eq!(
            generate(&mut adapted).tokens,
            generate(&mut expected_adapted).tokens
        );
        assert_eq!(
            generate(&mut base).tokens,
            generate(&mut expected_base).tokens
        );
        same_live(&adapted, &expected_adapted);
        same_live(&base, &expected_base);
        adapted.reset().unwrap();
        assert!(adapted.has_lora_adapters());
        close(
            &adapted.hidden_states_for_tokens(&query).unwrap(),
            &adapted_hidden,
        );
        adapted.remove_lora_adapters();
        assert!(!adapted.has_lora_adapters());
        close(
            &adapted.hidden_states_for_tokens(&query).unwrap(),
            &base_hidden,
        );
        same_live(&base, &expected_base);
    }
}

#[test]
fn cpu_kv_format_sharing_preserves_backend_specific_rules() {
    let bytes = fixture::tiny_hybrid();
    for (first, second) in [
        (KvCompression::None, KvCompression::F16),
        (KvCompression::F16, KvCompression::None),
        (KvCompression::turboquant(7), KvCompression::turboquant(8)),
    ] {
        let shared = load(bytes.clone());
        let mut live = session(&shared, first.clone());
        let mut control = session(&load(bytes.clone()), first.clone());
        for s in [&mut live, &mut control] {
            s.append_tokens(&[0, 1]).unwrap();
        }
        for _ in 0..2 {
            let result = shared.clone().create_session(SessionConfig {
                kv_compression: second.clone(),
                ..SessionConfig::default()
            });
            assert!(matches!(
                result,
                Err(CeraError::KvCompressionConflict { .. })
            ));
            same_live(&live, &control);
            shared.clear_warm_cache();
            shared.clear_cache();
            shared.configure_cache(KvCacheConfig::default());
        }
        drop(session(&shared, first.clone()));
        assert_eq!(generate(&mut live).tokens, generate(&mut control).tokens);
        same_live(&live, &control);
        drop(live);
        assert!(matches!(
            shared.create_session(SessionConfig {
                kv_compression: second,
                ..SessionConfig::default()
            }),
            Err(CeraError::KvCompressionConflict { .. })
        ));
    }
    // Dense CPU Llama uses the trait's no-op compression configuration and
    // allocates the requested f32/f16 buffers on each Session's state instead.
    let shared = load(fixture::tiny_dense());
    let mut f32_session = session(&shared, KvCompression::None);
    let mut f16_session = session(&shared, KvCompression::F16);
    let mut control = session(&load(fixture::tiny_dense()), KvCompression::None);
    for s in [&mut f32_session, &mut control] {
        s.append_tokens(&[0, 1]).unwrap();
    }
    f16_session.append_tokens(&[1, 0, 1]).unwrap();
    for s in [&mut f32_session, &mut control] {
        s.append_tokens(&[0]).unwrap();
    }
    same_live(&f32_session, &control);
}

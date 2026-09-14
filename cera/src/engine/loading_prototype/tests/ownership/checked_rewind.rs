use super::*;
use crate::kv_cache::InferenceState;

#[test]
fn checked_rewind_real_cpu_models_resume_with_reference_logits() {
    for bytes in [
        fixture::tiny_dense(),
        fixture::tiny_hybrid(),
        fixture::tiny_attention_lfm2(true),
    ] {
        let loaded = load(bytes);
        let model = loaded.model();
        for compression in [KvCompression::None, KvCompression::F16] {
            let mut actual =
                InferenceState::from_config_with_compression(model.config(), &compression).unwrap();
            let mut reference =
                InferenceState::from_config_with_compression(model.config(), &compression).unwrap();
            for (position, token) in [0, 1, 1, 0].into_iter().enumerate() {
                model.forward(&[token], position, &mut actual);
            }
            model.check_kv_rewind(&actual, 2).unwrap();
            model.try_truncate_kv(&mut actual, 2).unwrap();
            assert_eq!(actual.seq_len, 2);
            for (position, token) in [0, 1].into_iter().enumerate() {
                model.forward(&[token], position, &mut reference);
            }
            let actual_logits = model.forward(&[0], 2, &mut actual);
            let reference_logits = model.forward(&[0], 2, &mut reference);
            close(&actual_logits, &reference_logits);
            assert_eq!(actual.seq_len, reference.seq_len);
        }
    }
}

#[test]
fn checked_rewind_rejects_bidirectional_models_and_classifier_adapters() {
    use crate::kv_cache::KvRewindError;
    use crate::lora::LoraAdapterWeights;

    for classifier in [false, true] {
        let loaded = load(fixture::tiny_attention_lfm2(classifier));
        let model = loaded.model();
        let mut state = InferenceState::from_config(model.config()).unwrap();
        if classifier {
            state.lora = Some(LoraAdapterWeights::new_classifier_for_testing(
                vec![0.5; 32 * 2],
                None,
                vec!["first".into(), "second".into()],
            ));
        }
        // An asymmetric suffix changes the bidirectional token distribution.
        model.forward_prefill(&[0, 1, 1, 1], 0, &mut state);
        let before = format!("{:?}", state.snapshot());
        assert_eq!(
            model.check_kv_rewind(&state, 2),
            Err(KvRewindError::NonCausal)
        );
        assert_eq!(
            model.try_truncate_kv(&mut state, 2),
            Err(KvRewindError::NonCausal)
        );
        assert_eq!(format!("{:?}", state.snapshot()), before);
        assert_eq!(state.seq_len, 4);
        let mut prefix = InferenceState::from_config(model.config()).unwrap();
        prefix.lora = state.lora.clone();
        model.forward_prefill(&[0, 1], 0, &mut prefix);
        // Preserve the defect evidence: unchecked slicing keeps prefix rows
        // that depended on the discarded suffix in bidirectional attention.
        state.truncate_to(2);
        if cfg!(has_blas) || crate::backend::cpu::int8_gemm_available() {
            assert_ne!(
                format!("{:?}", state.snapshot()),
                format!("{:?}", prefix.snapshot()),
                "unchecked bidirectional rewind with classifier={classifier}"
            );
        } else {
            eprintln!(
                "bidirectional suffix-dependence control needs batched GEMM; mode rejection still checked"
            );
        }
    }
}

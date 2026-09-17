use super::*;
use crate::kv_cache::ConvHistory;
use crate::model::{Model, ModelConfig};
use crate::turboquant::{CompressedKeyCache, CompressedValueCache};

fn state(f16: bool, positions: usize) -> InferenceState {
    let mut state = InferenceState::new(2);
    state.kv_f16 = f16;
    state.seq_len = positions;
    for (index, layer) in state.layers.iter_mut().enumerate() {
        let LayerState::Attention {
            key_cache,
            value_cache,
            key_cache_f16,
            value_cache_f16,
            ..
        } = layer
        else {
            unreachable!()
        };
        for n in 0..positions * 2 {
            let value = (index * 1000 + n) as f32;
            if f16 {
                key_cache_f16.push(half::f16::from_f32(value).to_bits());
                value_cache_f16.push(half::f16::from_f32(-value).to_bits());
            } else {
                key_cache.push(value);
                value_cache.push(-value);
            }
        }
    }
    state
}

fn conv(positions: std::ops::RangeInclusive<usize>) -> LayerState {
    let mut history = ConvHistory::new(3);
    let mut buffer = vec![0.0; 3];
    for position in positions {
        buffer.fill(position as f32);
        history.push(position, &buffer);
    }
    LayerState::Conv { buffer, history }
}

// Include the ring, inactive precision and one-sided compression: snapshot()
// alone cannot reveal changes to all of these on a rejected rewind.
fn image(state: &InferenceState) -> String {
    let mut out = format!("{}:{}", state.seq_len, state.kv_f16);
    for layer in &state.layers {
        out.push_str(&match layer {
            LayerState::Attention {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                compressed_keys,
                compressed_values,
            } => format!(
                "{key_cache:?}{value_cache:?}{key_cache_f16:?}{value_cache_f16:?}{:?}{:?}",
                compressed_keys
                    .as_ref()
                    .map(crate::turboquant::encode_compressed_keys),
                compressed_values
                    .as_ref()
                    .map(crate::turboquant::encode_compressed_values)
            ),
            LayerState::Conv { buffer, history } => format!("{buffer:?}{history:?}"),
            LayerState::Mamba2 {
                conv_state,
                ssm_state,
            }
            | LayerState::DeltaNet {
                conv_state,
                ssm_state,
            } => format!("{conv_state:?}{ssm_state:?}"),
            LayerState::ParallelAttentionMamba2 {
                key_cache,
                value_cache,
                key_cache_f16,
                value_cache_f16,
                compressed_keys,
                compressed_values,
                conv_state,
                ssm_state,
            } => format!(
                "{key_cache:?}{value_cache:?}{key_cache_f16:?}{value_cache_f16:?}{:?}{:?}{conv_state:?}{ssm_state:?}",
                compressed_keys
                    .as_ref()
                    .map(crate::turboquant::encode_compressed_keys),
                compressed_values
                    .as_ref()
                    .map(crate::turboquant::encode_compressed_values)
            ),
        });
    }
    out
}

fn rejected(state: &mut InferenceState, target: usize, expected: KvRewindError) {
    let before = image(state);
    assert_eq!(state.check_truncate_to(target), Err(expected.clone()));
    assert_eq!(image(state), before);
    assert_eq!(state.try_truncate_to(target), Err(expected));
    assert_eq!(image(state), before);
}

#[test]
fn checked_rewind_preserves_exact_f32_and_f16_prefixes() {
    for f16 in [false, true] {
        let mut actual = state(f16, 8);
        let before = image(&actual);
        actual.check_truncate_to(3).unwrap();
        assert_eq!(image(&actual), before);
        actual.try_truncate_to(3).unwrap();
        assert_eq!(image(&actual), image(&state(f16, 3)));
        actual.try_truncate_to(0).unwrap();
        assert_eq!(image(&actual), image(&state(f16, 0)));
        actual.try_truncate_to(0).unwrap();
    }
}

#[test]
fn checked_rewind_noop_and_bounds_do_not_mutate() {
    let mut actual = state(false, 8);
    actual.layers.push(conv(1..=8));
    let before = image(&actual);
    actual.try_truncate_to(8).unwrap();
    assert_eq!(image(&actual), before);
    rejected(
        &mut actual,
        9,
        KvRewindError::OutOfBounds {
            requested: 9,
            current: 8,
        },
    );
}

#[test]
fn checked_rewind_rejects_all_compression_modes_before_any_layer_changes() {
    for (keys, values) in [(true, false), (false, true), (true, true)] {
        let mut actual = state(false, 8);
        let LayerState::Attention {
            compressed_keys,
            compressed_values,
            ..
        } = &mut actual.layers[1]
        else {
            unreachable!()
        };
        *compressed_keys = keys.then(|| CompressedKeyCache::new(2, 8, 4));
        *compressed_values = values.then(|| CompressedValueCache::new(2, 8, 4));
        for target in [0, 3, 8] {
            rejected(&mut actual, target, KvRewindError::Compressed);
        }
    }
}

#[test]
fn checked_rewind_rejects_malformed_later_attention_before_mutation() {
    for f16 in [false, true] {
        for (key_len, value_len) in [(15, 16), (15, 15), (0, 0)] {
            let mut actual = state(f16, 8);
            let LayerState::Attention {
                key_cache,
                key_cache_f16,
                value_cache,
                value_cache_f16,
                ..
            } = &mut actual.layers[1]
            else {
                unreachable!()
            };
            if f16 {
                key_cache_f16.truncate(key_len);
                value_cache_f16.truncate(value_len);
            } else {
                key_cache.truncate(key_len);
                value_cache.truncate(value_len);
            }
            rejected(
                &mut actual,
                3,
                KvRewindError::InvalidCacheLayout {
                    layer: 1,
                    detail: "key/value rows do not match the current position",
                },
            );
        }
    }
    let mut actual = state(false, 8);
    let LayerState::Attention { key_cache_f16, .. } = &mut actual.layers[1] else {
        unreachable!()
    };
    key_cache_f16.push(1);
    rejected(
        &mut actual,
        3,
        KvRewindError::InvalidCacheLayout {
            layer: 1,
            detail: "inactive precision contains live rows",
        },
    );
}

#[test]
fn checked_rewind_checks_conv_width_before_mutating_earlier_layers() {
    let mut actual = state(false, 8);
    let mut layer = conv(1..=8);
    let LayerState::Conv { buffer, .. } = &mut layer else {
        unreachable!()
    };
    buffer.pop();
    actual.layers.push(layer);
    rejected(
        &mut actual,
        3,
        KvRewindError::InvalidCacheLayout {
            layer: 2,
            detail: "convolution buffer and history widths differ",
        },
    );
}

#[test]
fn checked_rewind_restores_recent_conv_even_at_deep_positions() {
    let mut actual = state(false, 120);
    actual.layers.push(conv(1..=120));
    actual.try_truncate_to(100).unwrap();
    let mut expected = state(false, 100);
    // Rewind retains only the ring entries still present before truncation.
    expected.layers.push(conv(57..=100));
    assert_eq!(actual.seq_len, 100);
    let LayerState::Conv { buffer, history } = actual.layers.last().unwrap() else {
        unreachable!()
    };
    assert_eq!(buffer, &[100.0; 3]);
    assert!(history.has_pos(100));
    assert!(!history.has_pos(101));
    assert_eq!(
        format!("{:?}", actual.snapshot()),
        format!("{:?}", expected.snapshot())
    );
    actual.try_truncate_to(0).unwrap();
    let LayerState::Conv { buffer, history } = actual.layers.last().unwrap() else {
        unreachable!()
    };
    assert_eq!(buffer, &[0.0; 3]);
    assert!(!history.has_pos(100));
}

#[test]
fn checked_rewind_rechecks_expired_checkpoint_without_silent_reset() {
    let mut actual = state(false, 120);
    actual.layers.push(conv(1..=120));
    actual.check_truncate_to(100).unwrap();
    // Later execution evicts the checkpoint that passed the earlier check.
    let LayerState::Conv { buffer, history } = actual.layers.last_mut().unwrap() else {
        unreachable!()
    };
    for position in 121..=185 {
        buffer.fill(position as f32);
        history.push(position, buffer);
    }
    for (index, layer) in actual.layers[..2].iter_mut().enumerate() {
        let LayerState::Attention {
            key_cache,
            value_cache,
            ..
        } = layer
        else {
            unreachable!()
        };
        for n in 120 * 2..185 * 2 {
            let value = (index * 1000 + n) as f32;
            key_cache.push(value);
            value_cache.push(-value);
        }
    }
    actual.seq_len = 185;
    rejected(
        &mut actual,
        100,
        KvRewindError::MissingConvolutionCheckpoint {
            layer: 2,
            position: 100,
        },
    );
}

struct UncheckedModel;
impl Model for UncheckedModel {
    fn config(&self) -> &ModelConfig {
        panic!("not needed for refusal")
    }
    fn forward(&self, _: &[u32], _: usize, _: &mut InferenceState) -> Vec<f32> {
        panic!("not called")
    }
    fn truncate_kv(&self, _: &mut InferenceState, _: usize) {
        panic!("legacy path must not imply checked support")
    }
}

#[test]
fn checked_rewind_model_default_never_calls_legacy_truncation() {
    let mut actual = state(false, 8);
    let before = image(&actual);
    assert_eq!(
        UncheckedModel.check_kv_rewind(&actual, 3),
        Err(KvRewindError::BackendUnsupported)
    );
    assert_eq!(
        UncheckedModel.try_truncate_kv(&mut actual, 3),
        Err(KvRewindError::BackendUnsupported)
    );
    assert_eq!(image(&actual), before);
}

#[test]
fn non_causal_and_classifier_rewinds_are_rejected() {
    struct CausalGateModel {
        is_causal: bool,
    }
    impl Model for CausalGateModel {
        fn config(&self) -> &ModelConfig {
            panic!("not needed")
        }
        fn forward(&self, _: &[u32], _: usize, _: &mut InferenceState) -> Vec<f32> {
            panic!("not called")
        }
        fn truncate_kv(&self, _: &mut InferenceState, _: usize) {
            panic!("not called")
        }
        fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
            if !self.is_causal || state.lora.as_ref().is_some_and(|l| l.is_classifier()) {
                return Err(KvRewindError::NonCausal);
            }
            state.check_truncate_to(len)
        }
        fn try_truncate_kv(
            &self,
            state: &mut InferenceState,
            len: usize,
        ) -> Result<(), KvRewindError> {
            self.check_kv_rewind(state, len)?;
            state.try_truncate_to(len)
        }
    }

    let mut actual = state(false, 8);
    let non_causal = CausalGateModel { is_causal: false };
    assert_eq!(
        non_causal.check_kv_rewind(&actual, 3),
        Err(KvRewindError::NonCausal)
    );
    assert_eq!(
        non_causal.try_truncate_kv(&mut actual, 3),
        Err(KvRewindError::NonCausal)
    );

    let causal = CausalGateModel { is_causal: true };
    actual.lora = Some(crate::lora::LoraAdapterWeights::new_classifier_for_testing(
        vec![0.5; 32 * 2],
        None,
        vec!["first".into(), "second".into()],
    ));
    assert_eq!(
        causal.check_kv_rewind(&actual, 3),
        Err(KvRewindError::NonCausal)
    );
    assert_eq!(
        causal.try_truncate_kv(&mut actual, 3),
        Err(KvRewindError::NonCausal)
    );
}

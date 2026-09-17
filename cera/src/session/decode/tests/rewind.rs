use super::*;
use crate::kv_cache::KvRewindError;

#[derive(Clone, Copy)]
enum RewindMode {
    CpuHistory,
    CounterOnly,
    FirstCheckUnsupported,
}

struct LegacyRewindModel {
    core: Arc<ScriptModel>,
    config: ModelConfig,
    mode: RewindMode,
    checks: Mutex<Vec<(usize, usize, bool)>>,
}

impl LegacyRewindModel {
    fn session(mode: RewindMode) -> (Arc<Self>, Session) {
        let (core, original) = ScriptModel::session(vec![0, 0, 2], true, 128);
        let mut config = core.config.clone();
        config.n_layers = 2;
        config.block_types.push(BlockType::GatedConv);
        config.kv_heads_per_layer.push(1);
        config.conv_kernel_size = Some(2);
        let model = Arc::new(Self {
            core,
            config,
            mode,
            checks: Mutex::new(Vec::new()),
        });
        let session = Session::new(
            model.clone(),
            original.tokenizer_arc(),
            ModalityCapabilities::text_only(),
            original.config.clone(),
        )
        .unwrap();
        (model, session)
    }
}

impl Model for LegacyRewindModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn supports_all_logits(&self) -> bool {
        true
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let mut logits = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            logits = self.core.forward(&[*token], pos + index, state);
            let LayerState::Conv { buffer, history } = &mut state.layers[1] else {
                unreachable!()
            };
            buffer.fill(state.seq_len as f32);
            history.push(state.seq_len, buffer);
        }
        logits
    }
    fn forward_prefill_logits_all(
        &self,
        tokens: &[u32],
        pos: usize,
        state: &mut InferenceState,
    ) -> Vec<f32> {
        self.core.verifications.fetch_add(1, Ordering::Relaxed);
        tokens
            .iter()
            .enumerate()
            .flat_map(|(index, token)| self.forward(&[*token], pos + index, state))
            .collect()
    }
    fn check_kv_rewind(&self, state: &InferenceState, len: usize) -> Result<(), KvRewindError> {
        let mut checks = self.checks.lock().unwrap();
        let result = match self.mode {
            RewindMode::CounterOnly => Err(KvRewindError::BackendUnsupported),
            RewindMode::FirstCheckUnsupported if checks.is_empty() => {
                Err(KvRewindError::BackendUnsupported)
            }
            _ => state.check_truncate_to(len),
        };
        checks.push((state.seq_len, len, result.is_ok()));
        result
    }
    fn truncate_kv(&self, state: &mut InferenceState, len: usize) {
        self.core.truncations.lock().unwrap().push(len);
        match self.mode {
            // Models the legacy device override's counter-only behavior. The
            // test deliberately leaves physical convolution/attention at the tail.
            RewindMode::CounterOnly => state.seq_len = len,
            _ => state.truncate_to(len),
        }
    }
}

#[test]
fn expired_convolution_at_either_rewind_cannot_certify_a_terminal() {
    for stop_in_accepted in [false, true] {
        let (model, mut active) = LegacyRewindModel::session(RewindMode::CpuHistory);
        active.append_tokens(&[3]).unwrap();
        // More than the real 64-entry ring. In one case every draft matches and
        // EOS is removed by Session; in the other, verify_draft rejects the tail
        // and the next guaranteed token is EOS. Both legacy paths return Stop.
        let mut draft = vec![0; 70];
        draft[0] = if stop_in_accepted { 2 } else { 1 };
        active.attach_drafter(&FixedDraft(draft));
        let mut config = opts();
        config.spec = Some(SpecDecode { ngram: 2, k: 70 });
        let mut sink = Sink::default();
        let observed = active.generate_observed(&config, &mut sink);
        assert_eq!(observed.observation, DecodeObservation::Unproven);
        assert_eq!(observed.result.unwrap().finish_reason, FinishReason::Stop);
        assert_eq!(sink.tokens, [0]);
        assert_eq!(sink.done, [FinishReason::Stop]);
        assert_eq!(active.position(), 2);
        assert_eq!(
            active.state.seq_len, 0,
            "legacy expired rewind clears state"
        );
        assert!(resident(&active).is_empty());
        let checks = model.checks.lock().unwrap();
        assert!(checks.contains(&(72, 2, false)));
        assert_eq!(checks.len(), if stop_in_accepted { 2 } else { 1 });
    }
}

#[test]
fn counter_only_backend_rewind_is_unproven_despite_matching_position() {
    let (model, mut active) = LegacyRewindModel::session(RewindMode::CounterOnly);
    active.append_tokens(&[3]).unwrap();
    active.attach_drafter(&FixedDraft(vec![2, 0, 0]));
    let mut sink = Sink::default();
    let observed = active.generate_observed(&opts(), &mut sink);
    assert_eq!(observed.observation, DecodeObservation::Unproven);
    assert_eq!(observed.result.unwrap().finish_reason, FinishReason::Stop);
    assert_eq!(active.state.seq_len, active.position() as usize);
    assert_eq!(active.position(), 2);
    assert_eq!(
        resident(&active).len(),
        5,
        "physical cache was not restored"
    );
    let LayerState::Conv { buffer, .. } = &active.state.layers[1] else {
        unreachable!()
    };
    assert!(buffer.iter().all(|v| *v == 5.0));
    assert!(model.checks.lock().unwrap().iter().all(|c| !c.2));
}

#[test]
fn a_later_successful_rewind_check_cannot_erase_an_unproven_round() {
    let (model, mut active) = LegacyRewindModel::session(RewindMode::FirstCheckUnsupported);
    active.append_tokens(&[3]).unwrap();
    active.attach_drafter(&FixedDraft(vec![2, 0, 0]));
    let observed = active.generate_observed(&opts(), &mut Sink::default());
    assert_eq!(observed.observation, DecodeObservation::Unproven);
    assert_eq!(observed.result.unwrap().finish_reason, FinishReason::Stop);
    assert_eq!(active.state.seq_len, 2);
    assert_eq!(resident(&active), [3, 0]);
    assert_eq!(*model.checks.lock().unwrap(), [(5, 5, false), (5, 2, true)]);
}

#[test]
fn standalone_legacy_verification_does_not_invoke_new_checks() {
    let (model, mut active) = LegacyRewindModel::session(RewindMode::CpuHistory);
    active.append_tokens(&[3]).unwrap();
    let result = crate::spec::verify_draft(model.as_ref(), &mut active.state, 0, &[2, 0], 4);
    assert_eq!(result.accepted, [2, 0]);
    assert!(model.checks.lock().unwrap().is_empty());
    assert_eq!(active.state.seq_len, 4);
}

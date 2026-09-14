//! Actual Session adapter for the private chat contract. Publication is gated by
//! numerical/backend/foreign-consumer validation; no public test hooks are added.

// The shared contract files import the crate as `super::core_api` so the same
// sources compile here and in the `chat_contract` integration test.
use crate as core_api;
use std::sync::{Arc, Weak};

use super::{
    CeraError, DecodeObservation, GenerateOpts, IngestRecovery, ModalitySink, RecoveryOutcome,
    Session,
};
use crate::model::Model;
use crate::tokenizer::BpeTokenizer;

#[path = "../../tests/api_chat/contract.rs"]
mod contract;
// The scripted contract tests run in this binary as well as in `chat_contract`.
// The runner's core mode relies on their pinned public-tokenizer case executing
// against the exact lib-test artifact, so the duplication is deliberate.
#[path = "../../tests/api_chat/tests.rs"]
mod contract_tests;
#[path = "../../tests/api_chat/fixtures.rs"]
mod fixtures;

use contract::{
    Chat, DecodeReport, DecodeState, Execution, IngestCause, IngestError, Profile, SessionPhase,
    ValidationError,
};

struct CoreExecution {
    session: Session,
    tokenizer: Arc<BpeTokenizer>,
    model: Weak<dyn Model>,
}

fn core_chat(session: Session) -> Result<Chat<CoreExecution>, ValidationError> {
    let tokenizer = session.tokenizer_arc();
    let profile = Profile::discover(tokenizer.clone())?;
    let keep = session.config.n_keep;
    let execution = CoreExecution {
        tokenizer,
        model: Arc::downgrade(&session.model),
        session,
    };
    Chat::new(execution, profile, keep)
}

impl CoreExecution {
    // Holding the `Weak` pins the model allocation without retaining a swapped
    // model's weights, so an address comparison cannot alias a reused block.
    fn validate_identity(&self) -> Result<(), ValidationError> {
        if !Arc::ptr_eq(&self.tokenizer, &self.session.tokenizer)
            || !std::ptr::addr_eq(self.model.as_ptr(), Arc::as_ptr(&self.session.model))
        {
            return Err(ValidationError::UnsupportedProfile);
        }
        let config = self.session.model.config();
        if config.vocab_size < self.tokenizer.vocab_size()
            || !config.is_causal
            || self.session.model.is_classifier()
            || self
                .session
                .lora
                .as_ref()
                .is_some_and(|adapter| adapter.is_classifier())
            || !self.session.capabilities.text_in
            || !self.session.capabilities.text_out
        {
            return Err(ValidationError::UnsupportedProfile);
        }
        // `Chat::new` checked this too, but a raw swap can install a Session
        // that shares the same model and tokenizer with a sliding context.
        if self.session.config.n_keep != 0 {
            return Err(ValidationError::SlidingContext);
        }
        Ok(())
    }
}

// `generate_observed` leaves the failure guard to its caller: an unwind inside
// forward or a sink callback produces no observation, and an Unproven one or an
// error after mutation cannot certify the cache either. In both cases this
// guard is what disables raw execution; Chat's phase guard only covers the
// chat cursor.
struct DecodeGuard<'a> {
    session: &'a mut Session,
    finished: bool,
}

impl Drop for DecodeGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.session.usable = false;
            self.session.last_logits = None;
        }
    }
}

impl Execution for CoreExecution {
    fn position(&self) -> usize {
        self.session.current_pos
    }
    fn capacity(&self) -> usize {
        self.session.max_seq_len
    }
    fn audio_output(&self) -> bool {
        self.session.capabilities.audio_out || self.session.audio_decoder.is_some()
    }
    fn validate_prepare(&self) -> Result<(), ValidationError> {
        self.validate_identity()?;
        if !self.session.is_usable() {
            return Err(ValidationError::Phase(SessionPhase::Unusable));
        }
        Ok(())
    }
    fn validate_decode(&self, _: &GenerateOpts) -> Result<(), ValidationError> {
        self.validate_prepare()
    }
    // `Chat::prepare` validated identity and usability immediately before this
    // call, and a successful replacement reset leaves the Session usable, so
    // neither check is repeated here.
    fn append(&mut self, tokens: &[u32]) -> Result<(), IngestError> {
        // A diagnostic left by an earlier raw `append_user_message` must not be
        // mistaken for this operation's, so only this call can populate the slot.
        self.session.last_ingest_recovery = None;
        self.session
            .with_ingest_recovery(|session| session.append_tokens(tokens))
            .map_err(|cause| {
                // Recovery records its diagnostic on the Session; take it whole
                // instead of stringifying a non-Clone primary/reset error to
                // copy it between layers. The legacy getter stays for
                // `append_user_message`. Recovery always records a diagnostic
                // after a mutation attempt; the fallback is the fail-closed
                // default for a usability refusal, which records none.
                let IngestRecovery {
                    outcome,
                    rewind_error,
                    reset_error,
                } = self
                    .session
                    .last_ingest_recovery
                    .take()
                    .unwrap_or(IngestRecovery {
                        outcome: RecoveryOutcome::Unusable,
                        rewind_error: None,
                        reset_error: None,
                    });
                IngestError {
                    cause: IngestCause::Execution(cause),
                    recovery: outcome,
                    rewind_error: rewind_error.map(Box::new),
                    recovery_error: reset_error,
                }
            })
    }
    // Reset does not depend on the chat profile, so identity drift is not
    // checked here: an explicit reset must stay available after `raw()` use,
    // and the next `prepare` reports the mismatch before touching execution.
    fn reset(&mut self, explicit: bool) -> Result<(), CeraError> {
        // Always require the checked complete reset, including replacement from
        // a previously usable state. Only an explicit caller reset clears cancel.
        // As at every other unusable transition, stale prefill logits must not
        // outlive a cache the backend could not certify, on error or unwind.
        self.session.usable = false;
        self.session.last_logits = None;
        self.session.reset_execution_checked()?;
        self.session.usable = true;
        self.session.last_ingest_recovery = None;
        if explicit {
            self.session.clear_cancel();
        }
        Ok(())
    }
    fn decode(&mut self, opts: &GenerateOpts, sink: &mut dyn ModalitySink) -> DecodeReport {
        let mut guard = DecodeGuard {
            session: &mut self.session,
            finished: false,
        };
        let observed = guard.session.generate_observed(opts, sink);
        // NoProgress is a proven non-mutation whatever the Result, so raw
        // execution stays enabled; Chat decides whether it certifies a prompt.
        // A successful audio-path exit is a proven outcome without a text
        // boundary; only unproven observations and errors disable execution.
        let state = match observed.observation {
            DecodeObservation::NoProgress
                if observed
                    .result
                    .as_ref()
                    .is_ok_and(|r| r.tokens_generated != 0) =>
            {
                DecodeState::Unusable
            }
            DecodeObservation::NoProgress => DecodeState::NoProgress,
            DecodeObservation::TokenStop { token } if observed.result.is_ok() => {
                DecodeState::Terminal {
                    token,
                    committed: false,
                }
            }
            DecodeObservation::Interrupted | DecodeObservation::Audio
                if observed.result.is_ok() =>
            {
                DecodeState::Interrupted
            }
            _ => DecodeState::Unusable,
        };
        guard.finished = !matches!(state, DecodeState::Unusable);
        DecodeReport {
            result: observed.result,
            state,
        }
    }
}

mod tests;

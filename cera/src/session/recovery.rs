//! Bounded recovery for a complete user-message append.

use super::*;
use crate::kv_cache::KvRewindError;

/// Execution state after a failed user-message append.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RecoveryOutcome {
    /// No execution mutation was attempted.
    Unchanged,
    /// The complete pre-call execution state was restored.
    Restored,
    /// Execution was reset; the caller must supply context again.
    Reset,
    /// State validity is unknown; a checked reset or recreation is required.
    Unusable,
}

/// Diagnostic for the last failed [`Session::append_user_message`].
///
/// The method still returns its original [`CeraError`]. This diagnostic retains
/// recovery failures separately. Automatic recovery never clears cancellation;
/// a caller retrying prefill must explicitly clear a pending cancellation.
#[derive(Debug)]
#[non_exhaustive]
pub struct IngestRecovery {
    pub outcome: RecoveryOutcome,
    /// Why checked rewind was unavailable, if a backend check rejected it.
    /// A destructive context shift skips rewind without a backend error.
    pub rewind_error: Option<KvRewindError>,
    /// The secondary error, if a complete execution reset failed.
    pub reset_error: Option<CeraError>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum Mutation {
    None,
    Tail,
    Destructive,
}

struct Checkpoint {
    position: usize,
    history_len: usize,
    logits: Option<Vec<f32>>,
    prefill_tokens: u32,
    prefill_elapsed: Duration,
    rewind_error: Option<KvRewindError>,
}

// Retain a mutable borrow until recovery finishes. An unwind during forward,
// rewind, reset or drafter cleanup cannot leave the session marked usable.
struct Ingestion<'a> {
    session: &'a mut Session,
    checkpoint: Checkpoint,
    finished: bool,
}

impl Drop for Ingestion<'_> {
    fn drop(&mut self) {
        if !self.finished && self.session.ingest_mutation != Some(Mutation::None) {
            self.session.usable = false;
            self.session.last_logits = None;
            self.session.last_ingest_recovery = Some(IngestRecovery {
                outcome: RecoveryOutcome::Unusable,
                rewind_error: self.checkpoint.rewind_error.take(),
                reset_error: None,
            });
        }
        self.session.ingest_mutation = None;
    }
}

impl Ingestion<'_> {
    fn recover(&mut self) {
        let session = &mut self.session;
        let checkpoint = &mut self.checkpoint;
        let mut outcome = RecoveryOutcome::Unchanged;
        let mut rewind_error = None;
        let mut reset_error = None;
        if session.ingest_mutation != Some(Mutation::None) {
            session.usable = false;
            rewind_error = checkpoint.rewind_error.take();
            let restored =
                if session.ingest_mutation == Some(Mutation::Tail) && rewind_error.is_none() {
                    match session
                        .model
                        .try_truncate_kv(&mut session.state, checkpoint.position)
                    {
                        Ok(()) => true,
                        Err(error) => {
                            rewind_error = Some(error);
                            false
                        }
                    }
                } else {
                    false
                };
            if restored {
                session.current_pos = checkpoint.position;
                session
                    .position_atomic
                    .store(checkpoint.position as u32, Ordering::Relaxed);
                session.token_history.truncate(checkpoint.history_len);
                session.last_logits = checkpoint.logits.take();
                session.prefill_tokens = checkpoint.prefill_tokens;
                session.prefill_elapsed = checkpoint.prefill_elapsed;
                // Ingestion does not advance the sampler or drafter. A context
                // shift resets the drafter and is excluded from this branch.
                session.usable = true;
                outcome = RecoveryOutcome::Restored;
            } else {
                match session.reset_execution_checked() {
                    Ok(()) => {
                        session.usable = true;
                        outcome = RecoveryOutcome::Reset;
                    }
                    Err(error) => {
                        session.last_logits = None;
                        reset_error = Some(error);
                        outcome = RecoveryOutcome::Unusable;
                    }
                }
            }
        }
        session.last_ingest_recovery = Some(IngestRecovery {
            outcome,
            rewind_error,
            reset_error,
        });
    }
}

impl Session {
    /// Whether inference is allowed. `false` requires a successful checked
    /// [`Self::reset`] or recreation; clearing cancellation does not repair KV.
    pub fn is_usable(&self) -> bool {
        self.usable
    }

    /// Recovery diagnostic from the last failed user-message append. Cleared
    /// by a successful user-message append or explicit reset. Raw append calls
    /// retain their existing partial-prefill behavior and do not set this report.
    pub fn last_ingest_recovery(&self) -> Option<&IngestRecovery> {
        self.last_ingest_recovery.as_ref()
    }

    pub(super) fn ensure_usable(&self) -> Result<(), CeraError> {
        if self.usable {
            Ok(())
        } else {
            Err(CeraError::Backend(
                "session is unusable after failed recovery; reset successfully or recreate it"
                    .into(),
            ))
        }
    }

    pub(super) fn note_ingest_mutation(&mut self, destructive: bool) {
        if let Some(mutation) = &mut self.ingest_mutation {
            if destructive {
                *mutation = Mutation::Destructive;
            } else if *mutation == Mutation::None {
                *mutation = Mutation::Tail;
            }
        }
    }

    pub(super) fn with_ingest_recovery(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<(), CeraError>,
    ) -> Result<(), CeraError> {
        self.ensure_usable()?;
        debug_assert!(self.ingest_mutation.is_none());
        let checkpoint = Checkpoint {
            position: self.current_pos,
            history_len: self.token_history.len(),
            logits: self.last_logits.clone(),
            prefill_tokens: self.prefill_tokens,
            prefill_elapsed: self.prefill_elapsed,
            rewind_error: self
                .model
                .check_kv_rewind(&self.state, self.current_pos)
                .err(),
        };
        self.ingest_mutation = Some(Mutation::None);
        let mut ingestion = Ingestion {
            session: self,
            checkpoint,
            finished: false,
        };
        let result = operation(ingestion.session);
        if result.is_err() {
            ingestion.recover();
        } else {
            ingestion.session.last_ingest_recovery = None;
        }
        ingestion.finished = true;
        result
    }

    // No cancellation load/store: a new external request during recovery must
    // survive just like one that was already pending before recovery started.
    pub(super) fn reset_execution_checked(&mut self) -> Result<(), CeraError> {
        self.model.try_reset_kv(
            &mut self.state,
            &self.config.kv_compression,
            self.max_seq_len,
        )?;
        self.clear_execution_metadata();
        Ok(())
    }

    pub(super) fn clear_execution_metadata(&mut self) {
        self.current_pos = 0;
        self.token_history.clear();
        self.position_atomic.store(0, Ordering::Relaxed);
        self.last_logits = None;
        self.prefill_tokens = 0;
        self.prefill_elapsed = Duration::ZERO;
        if let Some(drafter) = &mut self.drafter {
            drafter.reset();
        }
        self.sampler = Sampler::new(SamplerConfig {
            seed: self.config.seed,
            ..SamplerConfig::default()
        });
    }
}

#[cfg(test)]
mod tests;

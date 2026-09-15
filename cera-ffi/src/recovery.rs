//! Owned observations of whole-message ingestion recovery.

use crate::{FfiError, Session};

/// Execution state after a failed whole-message append.
#[derive(Debug, PartialEq, Eq, uniffi::Enum)]
pub enum RecoveryOutcome {
    Unchanged,
    Restored,
    Reset,
    Unusable,
    /// A newer core outcome; conservatively recreate the session.
    Unknown,
}

/// Why checked tail rewind was unavailable. Numeric positions are token counts.
#[derive(Debug, PartialEq, Eq, uniffi::Enum)]
pub enum KvRewindFailure {
    OutOfBounds { requested: u64, current: u64 },
    Compressed,
    NonCausal,
    MissingConvolutionCheckpoint { layer: u64, position: u64 },
    InvalidCacheLayout { layer: u64, detail: String },
    BackendUnsupported,
    Unknown { detail: String },
}

/// Recovery diagnostic retained after a failed `send_message` ingestion.
/// The call's original error is still returned separately. Generation failures
/// after successful ingestion do not create this report.
#[derive(Debug, uniffi::Record)]
pub struct IngestRecovery {
    pub outcome: RecoveryOutcome,
    pub rewind_error: Option<KvRewindFailure>,
    pub reset_error: Option<FfiError>,
}

/// Coherent snapshot of a session at the instant the lock was acquired.
/// Another thread may change the session after this method returns.
#[derive(Debug, uniffi::Record)]
pub struct SessionRecoveryStatus {
    /// False requires a successful checked reset or recreation.
    pub usable: bool,
    /// Meaningful as reusable context only when `usable` is true.
    pub position: u32,
    /// Cleared by successful whole-message ingestion or explicit reset.
    /// Raw append calls and cancellation controls leave it unchanged.
    pub last_ingest_recovery: Option<IngestRecovery>,
}

impl From<&cera::kv_cache::KvRewindError> for KvRewindFailure {
    fn from(error: &cera::kv_cache::KvRewindError) -> Self {
        use cera::kv_cache::KvRewindError as E;
        match error {
            E::OutOfBounds { requested, current } => Self::OutOfBounds {
                requested: *requested as u64,
                current: *current as u64,
            },
            E::Compressed => Self::Compressed,
            E::NonCausal => Self::NonCausal,
            E::MissingConvolutionCheckpoint { layer, position } => {
                Self::MissingConvolutionCheckpoint {
                    layer: *layer as u64,
                    position: *position as u64,
                }
            }
            E::InvalidCacheLayout { layer, detail } => Self::InvalidCacheLayout {
                layer: *layer as u64,
                detail: (*detail).into(),
            },
            E::BackendUnsupported => Self::BackendUnsupported,
            other => Self::Unknown {
                detail: other.to_string(),
            },
        }
    }
}

impl From<&cera::session::IngestRecovery> for IngestRecovery {
    fn from(report: &cera::session::IngestRecovery) -> Self {
        Self {
            outcome: match report.outcome {
                cera::session::RecoveryOutcome::Unchanged => RecoveryOutcome::Unchanged,
                cera::session::RecoveryOutcome::Restored => RecoveryOutcome::Restored,
                cera::session::RecoveryOutcome::Reset => RecoveryOutcome::Reset,
                cera::session::RecoveryOutcome::Unusable => RecoveryOutcome::Unusable,
                _ => RecoveryOutcome::Unknown,
            },
            rewind_error: report.rewind_error.as_ref().map(Into::into),
            reset_error: report.reset_error.as_ref().map(Into::into),
        }
    }
}

#[uniffi::export]
impl Session {
    /// Observe recovery after a failed whole-message call without changing KV,
    /// cancellation or the retained report. Returns `Busy` if any call holds
    /// the session lock, including a streaming callback's enclosing operation.
    /// A poisoned lock returns `Backend`; recreate that session.
    ///
    /// `Reset` requires replaying prior context. `Restored` and `Unchanged`
    /// retain it when `usable` is true. Clear cancellation explicitly before
    /// retrying a cancelled append. A missing report gives no recovery guarantee
    /// for raw append operations, which retain their partial-prefill behavior.
    pub fn recovery_status(&self) -> Result<SessionRecoveryStatus, FfiError> {
        let guard = self.inner_mutex().try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => FfiError::Busy,
            std::sync::TryLockError::Poisoned(_) => FfiError::Backend {
                detail: "session mutex poisoned; recreate the session".into(),
            },
        })?;
        let session = guard.as_ref().ok_or_else(|| FfiError::Backend {
            detail: "session has been moved into a ChatSession".into(),
        })?;
        Ok(SessionRecoveryStatus {
            usable: session.is_usable(),
            position: session.position(),
            last_ingest_recovery: session.last_ingest_recovery().map(Into::into),
        })
    }
}

#[cfg(test)]
mod tests;

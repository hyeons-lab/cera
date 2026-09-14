use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::CeraError;

/// Exclusive session ownership for a model that stores live inference state.
///
/// Keep one gate per independently owned KV/conv context and return its lease
/// from [`super::Model::acquire_session`]. Models whose state is entirely owned
/// by the caller's `InferenceState` can keep the trait's default implementation.
#[derive(Debug, Default)]
pub struct ModelSessionGate {
    active: Arc<AtomicBool>,
}

impl ModelSessionGate {
    /// Reserve this context without waiting, or return [`CeraError::Busy`].
    /// Dropping the returned lease makes it available again.
    pub fn try_acquire(&self) -> Result<ModelSessionLease, CeraError> {
        self.active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| CeraError::Busy)?;
        Ok(ModelSessionLease {
            active: Arc::clone(&self.active),
        })
    }
}

/// A non-cloneable reservation of a model's live inference context.
///
/// [`crate::Session`] retains this until it is destroyed, including across
/// reset and cancellation. The gate's storage remains alive independently of
/// the model so destruction never accesses an already-dropped model.
#[derive(Debug)]
#[must_use = "dropping the lease releases the model's session reservation"]
pub struct ModelSessionLease {
    active: Arc<AtomicBool>,
}

impl Drop for ModelSessionLease {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_acquire_is_busy_until_lease_drops() {
        let gate = ModelSessionGate::default();
        let lease = gate.try_acquire().expect("first acquire succeeds");
        assert!(matches!(gate.try_acquire(), Err(CeraError::Busy)));
        drop(lease);
        assert!(
            gate.try_acquire().is_ok(),
            "dropping the lease releases the gate"
        );
    }

    #[test]
    fn concurrent_acquire_has_exactly_one_winner() {
        use std::sync::Barrier;
        use std::sync::atomic::AtomicUsize;

        // The gate's purpose is cross-thread mutual exclusion, which the
        // sequential test above cannot verify: a racy reimplementation (a
        // `load` plus `store` instead of `compare_exchange`) would pass it
        // yet admit two live sessions on one GPU-state model. One barrier
        // round catches an adjacent load+store race only ~1% of runs (the
        // race window is the sub-microsecond gap between one load and
        // store), so repeat: 500 rounds push the kill rate past 99%. The
        // exit barrier is load-bearing: winners hold their lease past it,
        // so every attempt in a round lands while the gate is held and a
        // second winner is a real double-acquire, never a post-release
        // late arrival (which would flake on correct code).
        let enter = Barrier::new(8);
        let exit = Barrier::new(8);
        for _ in 0..500 {
            let gate = ModelSessionGate::default();
            let wins = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(|| {
                        enter.wait();
                        let held = gate.try_acquire();
                        if held.is_ok() {
                            wins.fetch_add(1, Ordering::Relaxed);
                        }
                        exit.wait();
                    });
                }
            });
            assert_eq!(wins.load(Ordering::Relaxed), 1);
        }
    }
}

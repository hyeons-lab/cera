//! Fallible tail rewind for CPU-owned execution state.

use super::{InferenceState, LayerState};

/// A rejected KV rewind. Rejection leaves the supplied state unchanged.
///
/// This describes KV capability, not complete Session recovery. Logits, sampler,
/// drafter and other session bookkeeping need their own recovery contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum KvRewindError {
    #[error("rewind target {requested} exceeds current position {current}")]
    OutOfBounds { requested: usize, current: usize },
    #[error("tail rewind is unsupported for compressed KV caches")]
    Compressed,
    #[error("checked rewind requires a prefix produced by causal attention")]
    NonCausal,
    #[error("convolution layer {layer} has no checkpoint at position {position}")]
    MissingConvolutionCheckpoint { layer: usize, position: usize },
    #[error("invalid cache layout in layer {layer}: {detail}")]
    InvalidCacheLayout { layer: usize, detail: &'static str },
    #[error("this backend has not implemented checked KV rewind")]
    BackendUnsupported,
}

impl InferenceState {
    /// Check whether CPU-owned KV can be rewound to `len` without mutation.
    ///
    /// Checks every layer before any cache changes. Compression, missing
    /// convolution snapshots and malformed row layouts return errors. A check
    /// is only valid at this instant: prefill can evict a ring-buffer checkpoint.
    /// [`Self::try_truncate_to`] repeats it immediately before mutation.
    ///
    /// Use [`crate::model::Model::check_kv_rewind`] for model execution: CPU
    /// state alone cannot establish the validity of device-owned state. This
    /// also cannot detect a prior destructive context shift or replacement;
    /// the caller must ensure the retained prefix is still the intended one.
    pub fn check_truncate_to(&self, len: usize) -> Result<(), KvRewindError> {
        if len > self.seq_len {
            return Err(KvRewindError::OutOfBounds {
                requested: len,
                current: self.seq_len,
            });
        }
        if self.is_compressed() {
            return Err(KvRewindError::Compressed);
        }
        if len == self.seq_len {
            return Ok(());
        }
        // Here len < seq_len, so divisions below always have a nonzero divisor.
        for (layer, state) in self.layers.iter().enumerate() {
            let invalid = |detail| KvRewindError::InvalidCacheLayout { layer, detail };
            match state {
                LayerState::Attention {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    let (keys, values, inactive_keys, inactive_values) = if self.kv_f16 {
                        (
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                            key_cache.len(),
                            value_cache.len(),
                        )
                    } else {
                        (
                            key_cache.len(),
                            value_cache.len(),
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                        )
                    };
                    if inactive_keys != 0 || inactive_values != 0 {
                        return Err(invalid("inactive precision contains live rows"));
                    }
                    if keys == 0 || keys != values || !keys.is_multiple_of(self.seq_len) {
                        return Err(invalid("key/value rows do not match the current position"));
                    }
                }
                LayerState::Conv { buffer, history } => {
                    if buffer.len() != history.buf_len {
                        return Err(invalid("convolution buffer and history widths differ"));
                    }
                    if !history.has_pos(len) {
                        return Err(KvRewindError::MissingConvolutionCheckpoint {
                            layer,
                            position: len,
                        });
                    }
                }
                LayerState::Mamba2 { .. } | LayerState::DeltaNet { .. } => {
                    if len > 0 {
                        return Err(invalid(
                            "recurrent layer cannot be rewound to non-zero position",
                        ));
                    }
                }
                LayerState::ParallelAttentionMamba2 {
                    key_cache,
                    value_cache,
                    key_cache_f16,
                    value_cache_f16,
                    ..
                } => {
                    let (keys, values, inactive_keys, inactive_values) = if self.kv_f16 {
                        (
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                            key_cache.len(),
                            value_cache.len(),
                        )
                    } else {
                        (
                            key_cache.len(),
                            value_cache.len(),
                            key_cache_f16.len(),
                            value_cache_f16.len(),
                        )
                    };
                    if inactive_keys != 0 || inactive_values != 0 {
                        return Err(invalid("inactive precision contains live rows"));
                    }
                    if keys == 0 || keys != values || !keys.is_multiple_of(self.seq_len) {
                        return Err(invalid("key/value rows do not match the current position"));
                    }
                    if len > 0 {
                        return Err(invalid(
                            "recurrent layer cannot be rewound to non-zero position",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Rewind CPU-owned KV after validating every layer; errors never partially
    /// truncate a cache or silently reset a missing convolution checkpoint.
    ///
    /// Reuses [`Self::truncate_to`] after checking all of its failure conditions
    /// under the same exclusive borrow. No KV snapshot, token-history copy or
    /// device readback is taken. Model users should instead call
    /// [`crate::model::Model::try_truncate_kv`], which accounts for backend state.
    /// Legacy `truncate_to` retains its existing behavior.
    pub fn try_truncate_to(&mut self, len: usize) -> Result<(), KvRewindError> {
        self.check_truncate_to(len)?;
        self.truncate_to(len);
        Ok(())
    }
}

#[cfg(test)]
mod tests;

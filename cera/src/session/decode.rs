//! Facts recorded by the decode loop for the future chat transaction boundary.
//! These are internal observations, not new legacy finish reasons or a claim
//! that a profile supports every observed generation path.

use super::{CeraError, GenerateSummary};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeObservation {
    /// No logits, RNG, token history or KV mutation. Existing prefill telemetry
    /// consumption and cancellation-guard cleanup still take place.
    NoProgress,
    /// This occurrence of the sampled stop token is not resident in KV, either
    /// because it stopped before forward or because speculative rewind removed it.
    TokenStop { token: u32 },
    /// Generation returned through the audio path. Its Stop is not a text turn end.
    Audio,
    /// A successful decode ended without a recognized token stop. Zero emitted
    /// tokens alone does not turn grammar/setup/other work into NoProgress.
    Interrupted,
    /// The operation did not establish state validity. A chat caller must fail
    /// closed; this observation does not change legacy Session error handling.
    Unproven,
}

impl DecodeObservation {
    pub(super) fn trace(self) {
        match self {
            Self::NoProgress => tracing::trace!(target: "cera::decode", outcome = "no-progress"),
            Self::TokenStop { token } => {
                tracing::trace!(target: "cera::decode", outcome = "token-stop", token, committed = false);
            }
            Self::Audio => tracing::trace!(target: "cera::decode", outcome = "audio"),
            Self::Interrupted => tracing::trace!(target: "cera::decode", outcome = "interrupted"),
            Self::Unproven => tracing::trace!(target: "cera::decode", outcome = "unproven"),
        }
    }
}

pub(super) struct ObservedGeneration {
    pub(super) result: Result<GenerateSummary, CeraError>,
    pub(super) observation: DecodeObservation,
}

#[cfg(test)]
mod tests;

use std::collections::HashSet;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::backend::cpu;

/// NaN-safe argmax over a logits slice: the greedy-decoding token pick.
///
/// NaN values compare as `-inf` (never selected); ties break to the lowest
/// index (matching the GPU argmax kernels). Public so external harnesses
/// (e.g. benchmark runners driving [`crate::model::Model`] directly) pick
/// tokens identically to cera's own greedy decode path.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a = if a.is_nan() { f32::NEG_INFINITY } else { **a };
            let b = if b.is_nan() { f32::NEG_INFINITY } else { **b };
            a.total_cmp(&b)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Min-p (relative) nucleus cutoff: drop tokens whose probability is below
    /// `min_p * p_max`. `0.0` disables it. Applied after top-k/top-p. Many
    /// LeapBundles text models recommend min-p over top-p.
    pub min_p: f32,
    /// Repetition penalty over tokens already generated this call (CTRL-style).
    /// `1.0` disables it; `>1.0` discourages repeats, `<1.0` encourages them.
    /// Presence-based: each token that has appeared is penalized exactly once,
    /// no matter how many times it recurred (not compounded per occurrence),
    /// before temperature. Non-positive / non-finite values disable it. Only
    /// the stochastic path applies it — greedy/argmax (`temperature <= 0` or
    /// `top_k == 1`) is unaffected by design.
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

/// Token sampler with temperature, top-k, top-p, min-p, and repetition-penalty
/// filtering. Tracks the tokens it has emitted this generation so the
/// repetition penalty can reference them; call [`Sampler::reset_history`] at the
/// start of each logical generation.
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
    /// Distinct tokens emitted since the last [`reset_history`]. Used by the
    /// repetition penalty. A set (not a window) — presence-based, like
    /// llama.cpp, so a token is penalized once regardless of how often it
    /// recurs.
    ///
    /// [`reset_history`]: Sampler::reset_history
    history: HashSet<u32>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self {
            config,
            rng,
            history: HashSet::new(),
        }
    }

    /// Replace the sampler's config mid-session. Preserves the existing RNG
    /// so intra-session determinism survives per-call opts changes (e.g.,
    /// adjusting temperature between turns of a chat).
    pub fn set_config(&mut self, config: SamplerConfig) {
        self.config = config;
    }

    /// Clear the repetition-penalty history. Call at the start of each logical
    /// generation so penalties don't leak across independent `generate()` calls.
    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    /// Sample a token ID from logits. Panics if logits is empty.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        assert!(!logits.is_empty(), "cannot sample from empty logits");

        // Greedy: argmax (NaN-safe). Triggered by temperature<=0 OR top_k=1
        // (single candidate makes temp/top_p/penalties irrelevant). Greedy
        // skips history bookkeeping too — it's deterministic by contract.
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            return argmax(logits);
        }

        // Repetition penalty over already-emitted tokens, before temperature
        // (matches llama.cpp ordering).
        if self.penalty_active() && !self.history.is_empty() {
            self.apply_repetition_penalty(logits);
        }

        // Temperature scaling
        let inv_temp = 1.0 / self.config.temperature;
        for l in logits.iter_mut() {
            *l *= inv_temp;
        }

        // Top-K filtering
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            self.apply_top_k(logits);
        }

        // Top-P (nucleus) filtering
        if self.config.top_p < 1.0 {
            self.apply_top_p(logits);
        }

        // Min-P (relative) filtering — trims the long tail left by top-p.
        if self.config.min_p > 0.0 {
            self.apply_min_p(logits);
        }

        // Softmax + weighted random selection
        cpu::softmax_inplace(logits);
        let token = self.weighted_sample(logits);
        // Only record history when the penalty is active — keeps the common
        // (penalty-disabled) path off the per-token HashSet insert.
        if self.penalty_active() {
            self.history.insert(token);
        }
        token
    }

    /// Whether the repetition penalty is in effect: finite, positive, and not
    /// the disabling `1.0`. Non-finite (`NaN` / `±inf`) or non-positive values
    /// would corrupt logits — divide-by-zero, sign flip, or `0.0 * inf = NaN` —
    /// so they're treated as disabled. When inactive, history is neither
    /// recorded nor read.
    fn penalty_active(&self) -> bool {
        let penalty = self.config.repetition_penalty;
        penalty.is_finite() && penalty > 0.0 && penalty != 1.0
    }

    /// CTRL-style repetition penalty: divide a positive logit by the penalty,
    /// multiply a non-positive one. Applied once per distinct prior token.
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        let penalty = self.config.repetition_penalty;
        for &token in &self.history {
            if let Some(logit) = logits.get_mut(token as usize) {
                if *logit > 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
    }

    /// Min-p filtering: drop tokens whose probability is below `min_p * p_max`.
    /// Since softmax is monotonic, this is a logit-space cutoff at
    /// `max_logit + ln(min_p)` — no softmax needed. `min_p` outside `(0, 1]` is
    /// ignored: `0` disables, `1.0` keeps only the max-logit token(s), and `>1`
    /// would erase every candidate. Operates on the current (post-temperature,
    /// post-top-k/p) logits.
    fn apply_min_p(&self, logits: &mut [f32]) {
        let min_p = self.config.min_p;
        // Only (0, 1] is meaningful: 0 disables, >1 would erase every candidate.
        if !(min_p > 0.0 && min_p <= 1.0) {
            return;
        }
        let max_logit = logits
            .iter()
            .copied()
            .filter(|l| l.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return;
        }
        let threshold = max_logit + min_p.ln();
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn apply_top_k(&self, logits: &mut [f32]) {
        let k = self.config.top_k;
        let mut sorted: Vec<f32> = logits.to_vec();
        let (_, &mut threshold, _) = sorted.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn apply_top_p(&self, logits: &mut [f32]) {
        let mut indices: Vec<usize> = (0..logits.len())
            .filter(|&i| logits[i].is_finite())
            .collect();
        if indices.is_empty() {
            return;
        }
        indices.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_val = logits[indices[0]];
        let mut probs: Vec<f32> = indices
            .iter()
            .map(|&i| (logits[i] - max_val).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        let mut cutoff_idx = probs.len();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        for &idx in &indices[cutoff_idx..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn weighted_sample(&mut self, probs: &[f32]) -> u32 {
        if probs.is_empty() {
            return 0;
        }
        let r: f32 = self.rng.r#gen();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(min_p: f32, repetition_penalty: f32) -> Sampler {
        Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p,
            repetition_penalty,
            seed: Some(42),
        })
    }

    #[test]
    fn min_p_drops_low_probability_tail() {
        // threshold = max_logit + ln(0.5) = 3.0 - 0.693 = 2.307
        let s = sampler(0.5, 1.0);
        let mut logits = vec![3.0f32, 2.9, 0.0, -5.0];
        s.apply_min_p(&mut logits);
        assert!(logits[0].is_finite(), "max token always survives");
        assert!(logits[1].is_finite(), "2.9 >= 2.307 survives");
        assert_eq!(logits[2], f32::NEG_INFINITY, "0.0 < 2.307 dropped");
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn min_p_keeps_only_near_max_when_threshold_high() {
        let s = sampler(0.99, 1.0);
        let mut logits = vec![5.0f32, 4.0, 3.0];
        s.apply_min_p(&mut logits);
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY, "exp(4-5)=0.37 < 0.99");
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn min_p_out_of_range_is_noop() {
        for bad in [0.0f32, 1.5, -0.2] {
            let s = sampler(bad, 1.0);
            let mut logits = vec![1.0f32, 0.0, -1.0];
            let before = logits.clone();
            s.apply_min_p(&mut logits);
            assert_eq!(logits, before, "min_p={bad} should be a no-op");
        }
    }

    #[test]
    fn repetition_penalty_lowers_repeated_token_logits() {
        let mut s = sampler(0.0, 2.0);
        s.history.insert(0);
        s.history.insert(2);
        let mut logits = vec![4.0f32, 1.0, -2.0];
        s.apply_repetition_penalty(&mut logits);
        assert_eq!(logits[0], 2.0, "positive logit divided by penalty");
        assert_eq!(logits[1], 1.0, "token not in history is untouched");
        assert_eq!(logits[2], -4.0, "non-positive logit multiplied by penalty");
    }

    #[test]
    fn reset_history_clears_penalty_state() {
        let mut s = sampler(0.0, 2.0);
        s.history.insert(1);
        s.reset_history();
        let mut logits = vec![0.5f32, 3.0];
        s.apply_repetition_penalty(&mut logits);
        assert_eq!(logits, vec![0.5, 3.0], "empty history → no penalty");
    }

    #[test]
    fn sample_records_history_in_stochastic_path() {
        let mut s = sampler(0.0, 1.1);
        let mut logits = vec![0.1f32, 5.0, 0.1];
        let token = s.sample(&mut logits);
        assert!(s.history.contains(&token), "sampled token enters history");
    }

    #[test]
    fn non_positive_repetition_penalty_does_not_corrupt() {
        // penalty 0.0 would divide a positive history logit by zero → +inf and
        // hijack the choice. The `> 0.0` guard skips it, so the true argmax wins.
        let mut s = sampler(0.0, 0.0);
        s.history.insert(1); // positive, non-max logit
        let mut logits = vec![100.0f32, 50.0, 0.0];
        let token = s.sample(&mut logits);
        assert_eq!(token, 0, "non-positive penalty must be ignored, not /0");
    }

    #[test]
    fn non_finite_repetition_penalty_is_disabled() {
        // +inf penalty against a 0.0 history logit would be `0 * inf = NaN`; the
        // `is_finite()` guard disables it so the true argmax (token 0) wins.
        let mut s = sampler(0.0, f32::INFINITY);
        s.history.insert(1); // logit 0.0 below
        let mut logits = vec![10.0f32, 0.0, 5.0];
        let token = s.sample(&mut logits);
        assert_eq!(token, 0, "non-finite penalty must be treated as disabled");
    }

    #[test]
    fn greedy_path_skips_history() {
        // temperature 0 → argmax, no history bookkeeping.
        let mut s = sampler(0.0, 1.1);
        s.set_config(SamplerConfig {
            temperature: 0.0,
            ..s.config.clone()
        });
        let mut logits = vec![0.1f32, 5.0, 0.1];
        let token = s.sample(&mut logits);
        assert_eq!(token, 1, "argmax picks the largest logit");
        assert!(s.history.is_empty(), "greedy path records no history");
    }
}

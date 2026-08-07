//! Decode worker-count selection.
//!
//! Decode runs on the detected performance cores, narrowed to what the loaded
//! model can actually keep busy. On heterogeneous big.LITTLE mobile the full
//! big-core set is the measured optimum — decode scales cleanly across every
//! big core (e.g. Tensor G5: 44.7 → 76.1 tok/s at 1 → 6 threads on LFM2-350M
//! Q4_0), and the [`RowPool`](super::threadpool::RowPool) already pins decode
//! workers to exactly those cores.
//!
//! A per-device throughput sweep was prototyped (a DRAM-bandwidth stream, then a
//! real Q4_0-GEMV probe) but abandoned: a *synthetic* probe can't reproduce the
//! real decode graph's multi-core scaling. Decode is a stream of small GEMVs
//! interleaved with norms/attention/softmax — how many depends on the model (25
//! to 257 per token across the set measured below) — a compute-heavier mix that
//! scales to all P-cores, whereas any single-matrix probe is pure weight-streaming
//! that saturates the memory bus at ~3–4 cores and so under-provisions. What
//! replaced it is not a probe but a *model property*: see [`DecodeShape`].
//! `CERA_DECODE_THREADS=<n>` still overrides everything with a fixed count.

use crate::backend::cpu_features::{CoreTopology, env_disabled, env_usize, physical_core_count};
use std::sync::OnceLock;

/// Fallback width cap, used whenever shape-based sizing does not apply: no
/// model loaded, `CERA_DECODE_SIZING=off`, or one of the two cases
/// [`width_for_host`] declines — a heterogeneous topology, or a host whose
/// physical core count cannot be detected.
///
/// It is also the ceiling on the *narrow* arm inside the sizing path, so
/// retuning it moves that arm on every sized host too, not just the fallback.
///
/// On a heterogeneous topology `perf_core_count` is already the
/// big-core count, so this only binds on the homogeneous fallback where
/// `perf_core_count` is *all logical CPUs*: a many-core host must not spin-wait
/// a barrier across every core for memory-bound decode. 12 covers every
/// single-die Apple Silicon P-core count (M4 Max = 12).
///
/// **There is no globally correct constant — that is why this is now only the
/// fallback.** Measured on a Ryzen AI MAX+ 395 (16 physical / 32 logical),
/// 12 model/quant combos, medians of 3 interleaved rounds, decode tok/s change
/// going from 12 to 20 workers (shallow / 1024-deep):
///
/// | model                | weights | bytes/dispatch |    12 → 20 |
/// |----------------------|--------:|---------------:|-----------:|
/// | TinyStories-20M Q8_0 |   21 MB |        0.84 MB | −47% / −17%|
/// | SmolLM-135M Q4_0     |   92 MB |        0.51 MB | −11% / −16%|
/// | SmolLM-135M Q8_0     |  145 MB |        0.80 MB | −13% / −14%|
/// | LFM2.5-230M Q4_K_M   |  153 MB |        1.72 MB |  −9% /  +6%|
/// | LFM2.5-350M Q4_0     |  219 MB |        2.21 MB |  +3% / +10%|
/// | qwen2-0.5B Q4_0      |  353 MB |        2.43 MB | −11% /  +6%|
/// | LFM2.5-350M Q8_0     |  379 MB |        3.83 MB | +29% / +24%|
/// | SmolLM-360M Q8_0     |  386 MB |        1.50 MB |  −7% /  +6%|
/// | qwen2-0.5B Q8_0      |  531 MB |        3.66 MB | +10% / +15%|
/// | Qwen3-0.6B Q8_0      |  639 MB |        2.84 MB |  +6% /  +8%|
/// | Llama-3.2-1B Q4_K_M  |  808 MB |        6.26 MB | +17% / +15%|
/// | Llama-3.2-1B Q8_0    | 1321 MB |       10.24 MB | +38% / +32%|
///
/// Note the two 379/386 MB rows: near-identical weight bytes, opposite answers.
/// Model *size* does not predict the optimum; bytes per pool dispatch does —
/// which is what [`DecodeShape`] computes.
const DECODE_MAX_AUTO: usize = 12;

/// Bytes-per-dispatch at or above which decode wants the wide arm, in decimal KB.
///
/// Below this a token's work is split across so many pool barriers that extra
/// workers cost more than they add; above it each dispatch carries enough bytes
/// to amortize the barrier and decode becomes bandwidth-bound.
///
/// 2.5 MB is where a parameter sweep scored best, **not** the midpoint of the
/// sign change — in the table on [`DECODE_MAX_AUTO`] the shallow column already
/// turns positive at 2.21 MB, so a midpoint rule would sit nearer 1.9 MB. The
/// sweep preferred 2.5 because the models just below it lose very little on the
/// narrow arm while the ones just above gain a lot, which a sign-change
/// midpoint does not capture.
///
/// Leave-one-out validation over the 12 combos above — parameters tuned on 11,
/// scored on the held-out 12th — put this rule at **98.1% of each model's
/// measured peak (worst case 90.4%)** against **90.1% (worst 73.8%)** for the
/// flat `DECODE_MAX_AUTO`. 11 of the 12 folds independently chose these same
/// constants, and the surrounding grid is a plateau rather than a spike, so the
/// exact value is not delicate.
///
/// **Calibrated on one x86 host.** Decimal KB, matching how the table above
/// quotes bytes-per-dispatch. Override with `CERA_DECODE_BPD_KB`.
const BPD_THRESHOLD_KB_DEFAULT: usize = 2500;

/// What the loaded model asks of the decode pool.
///
/// The predictive quantity is `weight_bytes / dispatches_per_token`, *not*
/// model size. Two models of the same size disagree sharply when they spread
/// their bytes across different numbers of pool barriers: LFM2.5-350M Q8_0
/// (379 MB) issues 99 dispatches per token and gains +29% from a wide pool,
/// while SmolLM-360M Q8_0 (386 MB) issues 257 and loses 7%.
#[derive(Debug, Clone, Copy)]
pub struct DecodeShape {
    /// The file's whole tensor payload. A close proxy for bytes streamed per
    /// token, not an exact one: the embedding matrix is summed in here but is a
    /// row lookup at decode, so an untied-embedding model is overcounted by it.
    /// Deliberately left as-is — the "weights" column of the calibration table
    /// on [`DECODE_MAX_AUTO`] is whole-file too, so the threshold is calibrated
    /// against exactly this quantity.
    pub weight_bytes: u64,
    /// Row-pool dispatches per decoded token.
    pub dispatches_per_token: usize,
}

impl DecodeShape {
    /// Average bytes moved per pool dispatch.
    pub fn bytes_per_dispatch(&self) -> u64 {
        self.weight_bytes / self.dispatches_per_token.max(1) as u64
    }

    /// Derive the shape from a GGUF's tensor table.
    ///
    /// A weight only *dispatches* if its output-row count clears
    /// [`gemv_par_threshold`](super::cpu::gemv_par_threshold) — narrower
    /// projections run serially on the caller and cost no barrier. That is why
    /// dispatches-per-layer is 6 on some models and 8 on others, and why the
    /// count cannot be read off the layer count alone.
    ///
    /// Validated against dispatch counts measured by instrumenting the pool:
    /// **exact** on every dense model — TinyStories-20M (25), SmolLM-135M
    /// (181), SmolLM-360M (257), qwen2-0.5B (145), Qwen3-0.6B (225),
    /// Llama-3.2-1B (129) — measured at a 32-token prompt.
    ///
    /// The attention term is itself gated at runtime (`decode_attention` only
    /// fans out above `decode_attn_par_min_work`), so at very shallow contexts
    /// those dispatches do not happen and the computed count reads high for
    /// dense models too. It converges as the context grows, which is where
    /// decode width matters, so the shape is computed for the steady state
    /// rather than the first few tokens.
    ///
    /// LFM2 reads ~20% high (105 vs 89 measured; 119 vs 99) because the
    /// attention term below adds one dispatch per *block*, while an LFM2 hybrid
    /// stack has attention in only some blocks. That errs toward a smaller
    /// bytes-per-dispatch and so toward the narrow arm — the cheap direction to
    /// be wrong in. It does move one model: LFM2.5-350M **Q4_0** computes to
    /// 1.84 MB/dispatch rather than its measured 2.21, pushing it further onto
    /// the narrow arm it was already assigned (see the note on
    /// `rule_matches_measured_direction` — it is the one combo whose measured
    /// 12 → 20 delta is mildly positive while the rule sizes it narrow). The
    /// Q8_0 build is unaffected: 3.19 MB/dispatch computed against 3.83
    /// measured, wide either way. Refining this means reading per-layer block
    /// types, which is arch-specific — deliberately not done here.
    pub fn from_gguf(gguf: &crate::gguf::GgufFile) -> Option<Self> {
        // An arch-less GGUF contributes no attention term rather than looking
        // up a nonsense ".block_count" key.
        let block_count = gguf
            .architecture()
            .and_then(|arch| gguf.get_u32(&format!("{arch}.block_count")))
            .unwrap_or(0) as usize;
        Self::from_tensors(
            gguf.tensors.iter().map(|(name, info)| {
                // GGUF shape is [ne0, ne1] = [input, output]; rows = ne1.
                (
                    name.as_str(),
                    info.shape.get(1).copied().unwrap_or(0),
                    info.size_bytes as u64,
                )
            }),
            block_count,
            super::cpu::gemv_par_threshold(),
        )
    }

    /// The counting itself, over `(tensor name, output rows, bytes)`. Split out
    /// from [`Self::from_gguf`] so it is testable without building a synthetic
    /// GGUF container.
    fn from_tensors<'a>(
        tensors: impl Iterator<Item = (&'a str, usize, u64)>,
        block_count: usize,
        par_threshold: usize,
    ) -> Option<Self> {
        let mut weight_bytes: u64 = 0;
        let mut dispatches = 0usize;
        let mut has_output_head = false;

        for (name, rows, bytes) in tensors {
            weight_bytes = weight_bytes.saturating_add(bytes);
            // The embedding is a row lookup per token, not a GEMV. Matched
            // exactly: LFM2 also ships a `token_embd_norm.weight`, which a
            // substring test would swallow.
            if name == "token_embd.weight" {
                continue;
            }
            if name == "output.weight" {
                has_output_head = true;
            }
            if rows >= par_threshold {
                dispatches += 1;
            }
        }
        // Tied embeddings: no `output.weight`, but the vocab GEMV still runs
        // (over the embedding matrix), and it is the single widest dispatch of
        // the token.
        if !has_output_head {
            dispatches += 1;
        }
        // One decode-attention fan-out per block (see
        // `model::transformer::decode_attention`).
        dispatches = dispatches.saturating_add(block_count);

        (weight_bytes > 0 && dispatches > 0).then_some(Self {
            weight_bytes,
            dispatches_per_token: dispatches,
        })
    }
}

static DECODE_SHAPE: OnceLock<DecodeShape> = OnceLock::new();

/// Register the loaded model's decode shape, so the decode pool can size itself
/// to it. First writer wins: the pool is a process-wide singleton built once on
/// first dispatch, so a second model loaded into the same process inherits the
/// first one's width (as it already inherited its thread count before this).
pub fn set_decode_shape(shape: DecodeShape) {
    let _ = DECODE_SHAPE.set(shape);
}

/// The registered decode shape, if a model has been loaded.
fn decode_shape() -> Option<DecodeShape> {
    DECODE_SHAPE.get().copied()
}

/// Whether shape-based sizing is active. `CERA_DECODE_SIZING=off` (also `0` /
/// `false`) restores the flat [`DECODE_MAX_AUTO`] cap, for A/B testing on a
/// host this has not been calibrated against.
fn sizing_enabled() -> bool {
    !env_disabled("CERA_DECODE_SIZING")
}

/// Resolve the decode worker count for this device.
///
/// Precedence: `CERA_DECODE_THREADS=<n>` (a fixed count, clamped to the detected
/// performance cores) > shape-based sizing, when a [`DecodeShape`] is
/// registered, sizing is on, and [`width_for_host`] does not decline this host
/// > the flat [`DECODE_MAX_AUTO`] cap.
pub fn decode_thread_count(topo: &CoreTopology) -> usize {
    let max_t = topo.perf_core_count.max(1);

    if let Ok(v) = std::env::var("CERA_DECODE_THREADS") {
        let v = v.trim();
        if !v.eq_ignore_ascii_case("auto")
            && let Ok(n) = v.parse::<usize>()
            && n >= 1
        {
            if n > max_t {
                tracing::warn!(
                    "cera: CERA_DECODE_THREADS={n} exceeds the {max_t} detected \
                     performance cores; clamping to {max_t} (set CERA_THREADS to \
                     raise the detected count, though that is itself clamped to \
                     the pinnable cores)"
                );
            }
            return n.min(max_t);
        }
    }

    if sizing_enabled()
        && let Some(shape) = decode_shape()
        && let Some(n) = width_for_host(topo, shape, env_overrides(), physical_core_count())
    {
        tracing::debug!(
            "cera: decode width {n} of {max_t} (weights {} MB, {} dispatches/token, \
             {} KB/dispatch)",
            shape.weight_bytes / 1_000_000,
            shape.dispatches_per_token,
            shape.bytes_per_dispatch() / 1000,
        );
        return n;
    }

    max_t.min(DECODE_MAX_AUTO)
}

/// Resolve the prefill worker count for this device: the performance cores,
/// same as decode, overridable with `CERA_PREFILL_THREADS=<n>` up to every
/// pinnable core.
///
/// **The efficiency cores are deliberately left out, and this was measured, not
/// assumed.** The tempting argument for including them is that prefill is one
/// large compute-bound GEMM handed out as work-stealing chunks
/// (`STEAL_CHUNKS_PER_WORKER` per worker), so a slow core should just claim
/// fewer chunks and still contribute, unlike decode's per-token full barrier.
/// That argument is wrong. The likeliest reason is that work-stealing balances
/// *throughput* but not the *tail*: the dispatch ends when the last chunk ends,
/// and an efficiency core that claims a chunk late holds every other worker at
/// the barrier for its whole duration. The competing explanation, that a pool
/// spanning every core starves the system and pays for it in spin contention,
/// is not supported: `CERA_SPIN=0` does not recover the loss, giving 70 and 78
/// over two rounds at the wide width against 147 and 165 for the narrow one.
/// The spin knob is noisy enough on this device that this rules the hypothesis
/// out rather than confirming the tail one.
///
/// A third explanation, that the wide pool loses its `dispatch_lock` `try_lock`
/// and silently runs whole GEMMs serially, is also ruled out: the counters in
/// `threadpool::stats` report **zero** serial fallbacks out of 863 fan-out
/// dispatches per run, on fast and slow runs alike.
///
/// Measured on a Pixel 10 Pro Fold (Tensor G5: 2x A520 at `cpu_capacity` 207,
/// 5x A725 at 824, 1x X4 at 1024), LFM2.5-350M-Q4_K_M, 512-token prompt, five
/// interleaved rounds, p50 prefill tok/s:
///
/// | prefill workers | cores spanned | prefill tok/s |
/// |----------------:|---------------|--------------:|
/// | 6               | perf only     | 141           |
/// | 8               | perf + 2x A520| 84.5          |
///
/// A 1.67x loss to add two cores worth a nominal 8% of the machine's capacity.
/// `cpu_capacity` rates the A520 4x slower than an A725, but on a quantized
/// NEON dotprod kernel it is worse than that, and one tail chunk at 6-8x
/// against ~32 chunks accounts for most of the gap before shared-cache
/// contention is counted at all.
///
/// An earlier Tensor G4 measurement appeared to show the opposite (prefill 123
/// → 138-146 going from 4 to 8 workers). It does not transfer: `pin_cores` held
/// only the 4 performance cores there, so the surplus workers ran *unpinned*
/// and the kernel was free to place and migrate them. Pinning a worker to a
/// specific A520 is a different configuration, and the one measured above.
pub fn prefill_thread_count(topo: &CoreTopology) -> usize {
    let default = topo.perf_core_count.max(1);
    let Some(n) = env_usize("CERA_PREFILL_THREADS") else {
        return default;
    };
    // The knob may reach past the performance cores, up to whatever can still
    // be pinned: widening is a loss on the parts measured here, not on every
    // part, and this is how the next one gets swept without a rebuild.
    //
    // Hosts where nothing will be pinned are *not* clamped, matching
    // `cpu_features::apply_thread_override`. The cliff this guards against is
    // pinned workers spin-waiting on the cores unpinned ones need; with nothing
    // pinned there is no such split, and clamping to the P-core count would
    // make the knob unusable on exactly the hosts (macOS, homogeneous desktop)
    // where a sweep is most convenient to run. `CERA_PIN=0` is that same state
    // asked for explicitly, so it is honoured here too rather than only the
    // platform-has-no-affinity case.
    if topo.pin_cores.is_empty() || !super::threadpool::pinning_enabled() {
        return n;
    }
    // Just `pin_cores.len()`: `perf_core_count` counts a subset of `pin_cores`
    // on the sysfs path, and `apply_thread_override` truncates the two into
    // lockstep, so it can never exceed the pinnable count.
    let ceiling = topo.pin_cores.len();
    if n > ceiling {
        tracing::warn!(
            "cera: CERA_PREFILL_THREADS={n} exceeds the {ceiling} pinnable cores; \
             clamping to {ceiling}"
        );
    }
    n.min(ceiling)
}

/// Ceiling on either arm, for the same reason [`DECODE_MAX_AUTO`] has one:
/// decode is memory-bound and runs on a *full* barrier (every worker
/// decrements), so past some width more workers only add barrier traffic. The
/// arms scale with the physical core count, and nothing in the calibration says
/// what a 96-core host wants — it topped out at 16 physical / 20 workers. 24
/// leaves headroom above the measured peak without letting `phys + phys/4` run
/// away to 120 workers on a big server.
///
/// Unmeasured above 20; it is a bound on extrapolation, not a tuned value. It
/// does raise the ceiling for the dual-die Apple parts the old flat cap covered
/// by accident: an M3 Ultra (24 P-cores, no SMT) can now reach 24 decode
/// workers where 12 was the limit, and decode scaling across that die
/// interconnect remains unproven — `CERA_DECODE_WIDE` or
/// `CERA_DECODE_SIZING=off` is the lever if it disappoints.
const DECODE_WIDTH_MAX: usize = 24;

/// Resolve the rule's inputs for this host, or decline to size it.
///
/// Pinning a width (`CERA_DECODE_NARROW` / `CERA_DECODE_WIDE`) overrides the
/// decline checks below: those knobs exist so an uncalibrated machine can be
/// swept, and the hosts that decline are exactly the ones most worth sweeping.
/// A heterogeneous host needs only *one* arm pinned — it still knows its core
/// count, so the other arm derives normally. A host whose physical count is
/// unknown needs **both**, since deriving even one arm there would derive it
/// from the logical count.
///
/// `CERA_DECODE_BPD_KB` deliberately does **not** have that power — it moves the
/// threshold, it does not pin a width. Letting a threshold tweak switch sizing on
/// where physical detection failed would derive both arms from the *logical*
/// count and land on the full-width configuration this module measures as
/// catastrophic; and on a heterogeneous host it would halve a big-core width
/// that was measured optimal without the user having asked for any width at all.
///
/// Absent a width override, `None` — fall back to [`DECODE_MAX_AUTO`] — in two
/// cases where the rule has no business firing:
///
/// 1. **A heterogeneous topology** (`pin_cores` populated). There
///    `perf_core_count` is *already* the big-core set, and decode measured best
///    across all of it — Tensor G5, LFM2-350M Q4_0, 44.7 → 76.1 tok/s from
///    1 → 6 threads. That model computes to ~1.8 MB/dispatch and would land on
///    the narrow arm, i.e. the rule would halve a width that was measured
///    optimal. The calibration behind the threshold is from one homogeneous x86
///    host; it does not get to overrule a direct measurement on the device in
///    question. Note `detect_topology_sysfs` also populates `pin_cores` for x86
///    *hybrid* parts (Alder Lake and later) through its frequency path, so
///    those decline too — same argument, less direct evidence.
/// 2. **Unknown physical core count** (Windows, BSD, Intel macOS). Deriving the
///    arms from the *logical* count instead would make `wide` the full logical
///    width — the configuration measured at 29.7 → 17.9 tok/s. Better to keep
///    today's flat cap than to guess wrong in the expensive direction.
///
/// Otherwise both arms derive from the **physical** core count so the rule ports
/// off its calibration host:
///
/// - `wide` = physical + physical/4, capped by [`DECODE_WIDTH_MAX`] and clamped
///   to the detected cores. On an SMT host that spends a little of the SMT
///   budget (16 → 20 on Zen 5, the measured peak); with no SMT to spend the
///   clamp pins it to the core count (Apple Silicon M4 Max: 12 → 12).
/// - `narrow` = physical/2, the width small barrier-bound models want, capped by
///   [`DECODE_MAX_AUTO`] — a barrier-bound model must never get *more* workers
///   than the flat default used to give it — and never above `wide`.
///
/// **Apple Silicon is sized, not declined.** M-series parts are heterogeneous in
/// hardware, but `macos_perf_core_count` reports only `perflevel0` and leaves
/// `pin_cores` empty, so neither decline case fires: an M4 Max gets `wide` = 12
/// (its full P-core count, the clamp binding since there is no SMT) and
/// `narrow` = 6. **That narrow arm is unmeasured on non-SMT hosts** — halving
/// the P-cores for small models is the hypothesis this wants tested, not an
/// established result. `CERA_DECODE_SIZING=off` backs the whole thing out.
fn width_for_host(
    topo: &CoreTopology,
    shape: DecodeShape,
    ov: Overrides,
    physical: Option<usize>,
) -> Option<usize> {
    let max_t = topo.perf_core_count.max(1);
    // A heterogeneous host still *knows* its core count, so pinning either arm
    // is enough to say "size me anyway" — the other arm derives from a real
    // physical count as usual.
    if ov.narrow.is_none() && ov.wide.is_none() && !topo.pin_cores.is_empty() {
        return None;
    }
    let phys = match physical {
        Some(p) => p.clamp(1, max_t),
        // Physical count unknown. Only proceed when *both* arms are explicit:
        // deriving even one of them here would derive it from the logical
        // count, which is the configuration this module measured as
        // catastrophic (Llama-1B Q8_0, 32 workers: 29.7 → 17.9 tok/s).
        None if ov.narrow.is_some() && ov.wide.is_some() => max_t,
        None => return None,
    };

    let wide = ov
        .wide
        .unwrap_or_else(|| phys.saturating_add(phys / 4).min(DECODE_WIDTH_MAX));
    // `min(wide)`: a user who pins only one arm can otherwise invert the rule —
    // pinning `wide` below the derived `narrow` would hand barrier-bound models
    // *more* workers than bandwidth-bound ones.
    // Capped at `DECODE_MAX_AUTO`, not `DECODE_WIDTH_MAX`: the barrier-bound
    // arm must never hand a small model *more* workers than the flat default
    // used to, which is the direction measured as costly (TinyStories-20M:
    // −47% going 12 → 20). Without it a 64-physical-core host would derive
    // narrow = 24 and, at ≥48 cores, narrow == wide, quietly turning the rule
    // into a no-op that also doubled the small-model width.
    //
    // `.min(wide)` keeps the arms ordered whenever either is pinned — including
    // both pinned inconsistently (`NARROW=8 WIDE=4` yields 4).
    let narrow = ov
        .narrow
        .unwrap_or_else(|| (phys / 2).clamp(1, DECODE_MAX_AUTO))
        .min(wide);
    let threshold_kb = ov.threshold_kb.unwrap_or(BPD_THRESHOLD_KB_DEFAULT);
    Some(width_for_shape(max_t, narrow, wide, threshold_kb, shape))
}

/// The three tuning knobs, resolved from the environment. Passed in rather than
/// read inside [`width_for_host`] so that function is pure and its tests do not
/// depend on the ambient environment of whoever runs them — which, for knobs
/// whose entire purpose is sweeping a machine, is a real hazard.
#[derive(Debug, Clone, Copy, Default)]
struct Overrides {
    narrow: Option<usize>,
    wide: Option<usize>,
    threshold_kb: Option<usize>,
}

fn env_overrides() -> Overrides {
    Overrides {
        narrow: env_usize("CERA_DECODE_NARROW"),
        wide: env_usize("CERA_DECODE_WIDE"),
        threshold_kb: env_usize("CERA_DECODE_BPD_KB"),
    }
}

/// The rule itself: narrow below the bytes-per-dispatch threshold, wide at or
/// above it, clamped to the pool's ceiling.
///
/// `threshold_kb` is decimal KB, matching the bytes-per-dispatch column of the
/// table on [`DECODE_MAX_AUTO`] (which is decimal MB, as GGUF sizes are quoted
/// everywhere in this repo).
///
/// Pure — every input is a parameter, so tests exercise the rule without
/// touching process environment or depending on the runner's topology. (The env
/// lookups above are read per call rather than cached in a `OnceLock` like the
/// sibling knobs; this runs exactly once, inside `RowPool::decode()`'s own
/// `OnceLock`, so caching would buy nothing.)
fn width_for_shape(
    max_t: usize,
    narrow: usize,
    wide: usize,
    threshold_kb: usize,
    shape: DecodeShape,
) -> usize {
    let want = if shape.bytes_per_dispatch() / 1000 < threshold_kb as u64 {
        narrow
    } else {
        wide
    };
    want.clamp(1, max_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Zen 5 host the threshold was calibrated on: 16 physical, 32 logical.
    const PHYS: usize = 16;
    const MAX_T: usize = 32;
    const NARROW: usize = PHYS / 2; // 8
    const WIDE: usize = PHYS + PHYS / 4; // 20

    fn shape(mb: u64, dispatches: usize) -> DecodeShape {
        DecodeShape {
            weight_bytes: mb * 1_000_000,
            dispatches_per_token: dispatches,
        }
    }

    /// Apply the rule with the calibration host's arms pinned, so assertions
    /// test the *rule* and not the runner's topology or ambient environment.
    fn width(mb: u64, dispatches: usize) -> usize {
        width_for_shape(
            MAX_T,
            NARROW,
            WIDE,
            BPD_THRESHOLD_KB_DEFAULT,
            shape(mb, dispatches),
        )
    }

    #[test]
    fn bytes_per_dispatch_is_bytes_over_dispatches() {
        assert_eq!(shape(100, 100).bytes_per_dispatch(), 1_000_000);
        // Never divides by zero.
        assert_eq!(shape(100, 0).bytes_per_dispatch(), 100_000_000);
    }

    /// The pair that motivated the whole rule: near-identical weight bytes,
    /// opposite answers, discriminated only by dispatches per token.
    #[test]
    fn rule_separates_the_controlled_pair() {
        let lfm2_379 = width(379, 99); // 3.83 MB/dispatch — measured +29%
        let dense_386 = width(386, 257); // 1.50 MB/dispatch — measured −7%
        assert_eq!(lfm2_379, WIDE);
        assert_eq!(dense_386, NARROW);
    }

    /// Every measured combo lands on the side its benchmark said it wanted.
    ///
    /// One documented exception, marked `false` below: **LFM2.5-350M Q4_0**
    /// (219 MB over its 99 *measured* dispatches = 2.21 MB/dispatch, and 1.84
    /// as `from_gguf` computes it) sits just under the 2.5 MB threshold either
    /// way, so the rule sizes it narrow — while its measured 12 → 20
    /// delta is mildly positive (+3% shallow, +10% deep). The rule knowingly
    /// gives that up: a threshold has to fall somewhere, and this model sits
    /// closest to it of the twelve. The cost is already counted in the
    /// leave-one-out score on [`BPD_THRESHOLD_KB_DEFAULT`] (it scores ~91% of
    /// its own peak). Moving the threshold down to catch it costs more
    /// elsewhere — 2.0 MB scored worse overall in the sweep.
    #[test]
    fn rule_matches_measured_direction() {
        // (weights MB, dispatches/token, rule should pick wide) from the table
        // on `DECODE_MAX_AUTO`.
        let cases = [
            (21, 25, false),
            (92, 181, false),
            (145, 181, false),
            (153, 89, false),
            (219, 99, false), // see doc comment: the one deliberate miss
            (353, 145, false),
            (379, 99, true),
            (386, 257, false),
            (531, 145, true),
            (639, 225, true),
            (808, 129, true),
            (1321, 129, true),
        ];
        for (mb, disp, wants_wide) in cases {
            let want = if wants_wide { WIDE } else { NARROW };
            assert_eq!(width(mb, disp), want, "{mb} MB / {disp} dispatches");
        }
    }

    #[test]
    fn width_never_exceeds_the_pool_ceiling() {
        for max_t in [1, 2, 4, 6, 12, 16, 32] {
            for s in [shape(4096, 64), shape(1, 4096)] {
                let n = width_for_shape(max_t, NARROW, WIDE, BPD_THRESHOLD_KB_DEFAULT, s);
                assert!(n >= 1 && n <= max_t, "max_t={max_t} gave {n}");
            }
        }
    }

    /// A many-core host must not get an arm that scales with its core count:
    /// decode runs on a full barrier and the calibration topped out at 16
    /// physical cores. Without the cap a 96-core server would take 48 workers
    /// for a tiny model and 120 for a large one.
    #[test]
    fn arms_are_capped_on_many_core_hosts() {
        let epyc = CoreTopology {
            perf_core_count: 192,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        for phys in [16usize, 32, 64, 96] {
            let narrow = width_for_host(&epyc, shape(21, 25), Overrides::default(), Some(phys))
                .expect("homogeneous host with known physical count sizes");
            let wide = width_for_host(&epyc, shape(1321, 129), Overrides::default(), Some(phys))
                .expect("homogeneous host with known physical count sizes");
            assert!(
                wide <= DECODE_WIDTH_MAX,
                "phys={phys}: wide {wide} over cap"
            );
            // The barrier-bound arm must never exceed the old flat default —
            // small models measured *worse* with more workers.
            assert!(
                narrow <= DECODE_MAX_AUTO,
                "phys={phys}: narrow {narrow} exceeds the flat default"
            );
            // And the arms must stay distinct, or the rule is a no-op.
            assert!(narrow < wide, "phys={phys}: arms collapsed at {narrow}");
        }
        // The calibration host is below both caps, so they do not perturb it.
        let zen5 = CoreTopology {
            perf_core_count: 32,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        assert_eq!(
            width_for_host(&zen5, shape(1321, 129), Overrides::default(), Some(16)),
            Some(20)
        );
        assert_eq!(
            width_for_host(&zen5, shape(21, 25), Overrides::default(), Some(16)),
            Some(8)
        );
    }

    /// A pinned width is what makes it safe to size a host the rule would
    /// otherwise decline; moving the *threshold* is not, and must not unlock it.
    /// The unknown-physical case needs both arms, since deriving either one
    /// there would derive it from the logical count.
    #[test]
    fn only_a_pinned_width_overrides_a_decline() {
        let big_little = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![7, 6, 5, 4, 3, 2],
            fast_cores: 6,
        };
        let unknown_phys = CoreTopology {
            perf_core_count: 32,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        let s = shape(219, 99);
        let threshold_only = Overrides {
            threshold_kb: Some(1000),
            ..Default::default()
        };
        let one_width = Overrides {
            wide: Some(6),
            ..Default::default()
        };
        let both_widths = Overrides {
            narrow: Some(3),
            wide: Some(6),
            ..Default::default()
        };

        // Threshold alone unlocks neither.
        assert_eq!(
            width_for_host(&big_little, s, threshold_only, Some(8)),
            None
        );
        assert_eq!(width_for_host(&unknown_phys, s, threshold_only, None), None);
        // One arm is enough where the core count is known...
        assert!(width_for_host(&big_little, s, one_width, Some(8)).is_some());
        // ...but not where it is unknown, which would derive the other arm
        // from the logical count.
        assert_eq!(width_for_host(&unknown_phys, s, one_width, None), None);
        assert!(width_for_host(&unknown_phys, s, both_widths, None).is_some());
    }

    /// Pinned arms must actually be *used*, not merely unlock sizing — a
    /// regression that dropped them would otherwise leave every other test
    /// green while the documented knobs silently did nothing.
    #[test]
    fn pinned_arms_are_the_widths_used() {
        let host = CoreTopology {
            perf_core_count: 32,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        let ov = Overrides {
            narrow: Some(3),
            wide: Some(7),
            ..Default::default()
        };
        // Below the threshold -> the pinned narrow arm; above -> pinned wide.
        assert_eq!(width_for_host(&host, shape(21, 25), ov, Some(16)), Some(3));
        assert_eq!(
            width_for_host(&host, shape(1321, 129), ov, Some(16)),
            Some(7)
        );
        // And the threshold knob moves the boundary.
        let low_threshold = Overrides {
            threshold_kb: Some(100),
            ..ov
        };
        assert_eq!(
            width_for_host(&host, shape(21, 25), low_threshold, Some(16)),
            Some(7),
            "a 840 KB/dispatch model should take the wide arm under a 100 KB threshold"
        );
    }

    /// Pinning only `wide` must not leave a derived `narrow` above it, which
    /// would hand barrier-bound models more workers than bandwidth-bound ones.
    #[test]
    fn narrow_never_exceeds_wide() {
        let host = CoreTopology {
            perf_core_count: 32,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        let low_wide = Overrides {
            wide: Some(4),
            ..Default::default()
        };
        // phys=16 would derive narrow=8; it must be pulled down to wide.
        assert_eq!(
            width_for_host(&host, shape(21, 25), low_wide, Some(16)),
            Some(4)
        );
        assert_eq!(
            width_for_host(&host, shape(1321, 129), low_wide, Some(16)),
            Some(4)
        );
    }

    /// The dispatch count is the part that reads the model file, so pin its
    /// three rules: sub-threshold projections do not dispatch, `token_embd` is
    /// a lookup rather than a GEMV, and a tied-embedding model still pays for
    /// the vocab GEMV it runs over that same matrix.
    #[test]
    fn from_tensors_counts_dispatches() {
        const T: usize = 256; // gemv_par_threshold
        // Untied: an explicit output head, two wide projections, one narrow.
        let untied = DecodeShape::from_tensors(
            [
                ("token_embd.weight", 32_000, 1_000),
                ("blk.0.attn_q.weight", 512, 100),
                ("blk.0.attn_k.weight", 128, 100), // below threshold -> serial
                ("blk.0.ffn_up.weight", 1024, 100),
                ("output.weight", 32_000, 100),
            ]
            .into_iter(),
            1, // one block -> one attention dispatch
            T,
        )
        .expect("shape");
        // q + ffn_up + output = 3 GEMVs, + 1 attention, and no tied bonus.
        assert_eq!(untied.dispatches_per_token, 4);
        assert_eq!(untied.weight_bytes, 1_400);

        // Tied: same but no `output.weight`; the vocab GEMV still runs.
        let tied = DecodeShape::from_tensors(
            [
                ("token_embd.weight", 32_000, 1_000),
                ("blk.0.attn_q.weight", 512, 100),
                ("blk.0.attn_k.weight", 128, 100),
                ("blk.0.ffn_up.weight", 1024, 100),
            ]
            .into_iter(),
            1,
            T,
        )
        .expect("shape");
        // q + ffn_up = 2, + 1 tied vocab GEMV, + 1 attention.
        assert_eq!(tied.dispatches_per_token, 4);

        // A file with no tensors has no shape to offer.
        assert!(DecodeShape::from_tensors([].into_iter(), 0, T).is_none());
    }

    /// A heterogeneous topology declines sizing: `perf_core_count` is already
    /// the big-core set there and decode measured best across all of it, so the
    /// x86-calibrated threshold must not halve it.
    #[test]
    fn heterogeneous_topology_declines_sizing() {
        let big_little = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![7, 6, 5, 4, 3, 2],
            fast_cores: 6,
        };
        // LFM2-350M Q4_0 — the exact model measured scaling to all 6 big cores.
        assert_eq!(
            width_for_host(&big_little, shape(219, 99), Overrides::default(), Some(8)),
            None
        );
    }

    /// A host whose physical core count cannot be detected declines rather than
    /// deriving both arms from the logical count.
    #[test]
    fn unknown_physical_count_declines_sizing() {
        let windows_box = CoreTopology {
            perf_core_count: 32,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        assert_eq!(
            width_for_host(&windows_box, shape(1321, 129), Overrides::default(), None),
            None
        );
    }

    /// `CERA_PREFILL_THREADS` is read from the environment, so only the
    /// no-override default and the pure clamp are testable here without racing
    /// other tests over process env. The clamp's shape is the interesting part:
    /// it must NOT bind on hosts with nothing to pin, matching
    /// `cpu_features::apply_thread_override`, or the knob is unusable on macOS
    /// and homogeneous desktops.
    #[test]
    fn prefill_width_defaults_to_perf_cores() {
        let big_little = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![7, 6, 5, 4, 3, 2, 1, 0],
            fast_cores: 6,
        };
        assert_eq!(prefill_thread_count(&big_little), 6);

        // A host with no affinity still gets its perf-core count as the width.
        let unpinned = CoreTopology {
            perf_core_count: 10,
            pin_cores: Vec::new(),
            fast_cores: 0,
        };
        assert_eq!(prefill_thread_count(&unpinned), 10);
    }
}

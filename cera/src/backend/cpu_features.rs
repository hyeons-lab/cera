//! Runtime CPU capability detection + tier selection.
//!
//! Single source of truth for "what SIMD can this host run", replacing the
//! scattered `is_x86_feature_detected!` calls in the dispatchers ([`super::simd`]).
//! Detected once at first use and cached in a [`OnceLock`].
//!
//! ## Why this exists (vs. llama.cpp)
//!
//! llama.cpp compiles its whole CPU backend multiple times (sandybridge /
//! haswell / skylake-avx512 / ...), ships each as a separate shared library,
//! and at startup runs a *score* function to `dlopen` the best-matching build.
//! Rust doesn't need any of that: every `#[target_feature]` kernel coexists in
//! one binary, so "load the best variant" collapses to "resolve the [`CpuTier`]
//! once, then branch per call". This module is that resolver.
//!
//! ## Implemented vs. detected
//!
//! [`CpuFeatures::tier`] reports the best tier cera actually has *kernels* for,
//! so a dispatcher can never route to a kernel that doesn't exist. On x86 that
//! is [`CpuTier::Avx512Vnni`] (native `dpbusd` int8 GEMM/GEMV; needs the
//! default-on `avx512` crate feature) down through [`CpuTier::Avx512`] to
//! [`CpuTier::Avx2`], which runs the same int8 kernels with `dpbusd` emulated
//! and is therefore the floor for the whole x86 int8 path; on aarch64 it is
//! [`CpuTier::NeonI8mm`] (Q8_0/Q4_0/Q4_K/Q6_K GEMM) down to
//! [`CpuTier::NeonDotprod`].
//!
//! The raw feature bools (e.g. [`CpuFeatures::avx512vnni`]) are detected and
//! exposed regardless, for diagnostics and so future kernels can light up
//! without re-plumbing.

use std::sync::OnceLock;

/// Ordered CPU capability tier. Higher is more capable.
///
/// `Ord` is derived from declaration order, so within a single architecture the
/// comparison is meaningful (`Scalar < Avx2 < Avx512`, `Scalar < Neon <
/// NeonDotprod < NeonI8mm`). Cross-architecture comparisons are nonsensical but
/// harmless — only one architecture's variants are ever produced at runtime.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CpuTier {
    /// Portable scalar reference path. Always available.
    Scalar,
    /// x86_64 AVX2 + FMA. Also the threshold for the x86 int8 GEMM/GEMV path:
    /// `dpbusd` is emulated with `maddubs`+`madd` here, so every tier from this
    /// one up runs the same int8 kernels for Q4_0/Q8_0/Q4_K/Q6_K.
    Avx2,
    /// x86_64 AVX-512 (needs only `avx512f`). Produced when the default-on
    /// `avx512` crate feature is enabled; disable it to cap the x86 tier at
    /// AVX2.
    ///
    /// Its own contribution is now narrow: the 512-bit f32 `vec_dot` for
    /// Q8_0/Q4_0, plus the AVX-512 activation quantizer (which needs
    /// `avx512vl` on top). The production GEMM and GEMV at this tier are the
    /// [`CpuTier::Avx2`] int8 kernels — see `cpu::avx512_quantizer_available`.
    Avx512,
    /// x86_64 AVX-512 + VNNI (`avx512vnni` + `avx512vl`) — the x86 analogue of
    /// [`CpuTier::NeonI8mm`]. Runs the same int8 kernels as [`CpuTier::Avx2`]
    /// with a native `dpbusd` instead of the emulation, wider tiles, and a
    /// 512-bit activation quantizer. This tier is a speed difference, not a
    /// capability one — the dtype coverage is identical.
    Avx512Vnni,
    /// aarch64 baseline NEON.
    Neon,
    /// aarch64 NEON + dotprod (FEAT_DotProd, `vdotq_s32`).
    NeonDotprod,
    /// aarch64 NEON + i8mm (FEAT_I8MM, `vmmlaq_s32`) — Q8_0 and Q4_0 GEMM; other
    /// ops use the dotprod path (i8mm implies dotprod).
    NeonI8mm,
}

impl CpuTier {
    /// Lowercase label used by [`CpuFeatures::report`] and parsed by the
    /// `CERA_CPU_TIER` override.
    pub fn label(self) -> &'static str {
        match self {
            CpuTier::Scalar => "scalar",
            CpuTier::Avx2 => "avx2",
            CpuTier::Avx512 => "avx512",
            CpuTier::Avx512Vnni => "avx512+vnni",
            CpuTier::Neon => "neon",
            CpuTier::NeonDotprod => "neon+dotprod",
            CpuTier::NeonI8mm => "neon+i8mm",
        }
    }

    /// Parse a `CERA_CPU_TIER` label. Accepts a few spellings; returns `None`
    /// for anything unrecognized (the override is then ignored).
    ///
    /// Labels are arch-gated: only tiers valid for the current `target_arch`
    /// (plus `Scalar`) parse. Otherwise a cross-arch label like `avx2` on
    /// aarch64 would parse to `Avx2`, which — because `Avx2 < Neon*` in the
    /// ordering — `with_tier_override` would accept as a "downgrade", leaving
    /// the host with a tier it can't run. Returning `None` makes such an
    /// override a no-op instead.
    fn parse(s: &str) -> Option<CpuTier> {
        match s.trim().to_ascii_lowercase().as_str() {
            "scalar" | "none" | "off" => Some(CpuTier::Scalar),
            #[cfg(target_arch = "x86_64")]
            "avx2" => Some(CpuTier::Avx2),
            #[cfg(target_arch = "x86_64")]
            "avx512" => Some(CpuTier::Avx512),
            #[cfg(target_arch = "x86_64")]
            "vnni" | "avx512+vnni" | "avx512vnni" => Some(CpuTier::Avx512Vnni),
            #[cfg(target_arch = "aarch64")]
            "neon" => Some(CpuTier::Neon),
            #[cfg(target_arch = "aarch64")]
            "dotprod" | "neon+dotprod" | "neon,dotprod" => Some(CpuTier::NeonDotprod),
            #[cfg(target_arch = "aarch64")]
            "i8mm" | "neon+i8mm" | "neon,i8mm" => Some(CpuTier::NeonI8mm),
            _ => None,
        }
    }
}

/// Resolved CPU capabilities for this host.
///
/// `tier` is the selection the dispatchers act on (capped at implemented
/// kernels); the individual bools are the raw detection results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// Best tier cera has kernels for on this host (see module docs).
    pub tier: CpuTier,
    // ── x86_64 ──
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    // ── aarch64 ──
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    /// FEAT_FP16. Needed only to *declare* what `simd::neon::f16_bits_to_f32`
    /// uses: `core::arch`'s `vcvt_f32_f16` is gated `neon,fp16` because its
    /// operand type is `float16x4_t`, even though the `FCVTL` it lowers to is
    /// baseline ARMv8.0-A and cannot trap on any AArch64 core. Detected so the
    /// declaration is honest rather than relying on FEAT_I8MM (v8.6) implying
    /// FEAT_FP16 (v8.2), the same reason the i8mm parity test gates on dotprod
    /// instead of leaning on i8mm implying it.
    pub fp16: bool,
}

impl CpuFeatures {
    const NONE: CpuFeatures = CpuFeatures {
        tier: CpuTier::Scalar,
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vl: false,
        avx512vnni: false,
        neon: false,
        dotprod: false,
        i8mm: false,
        fp16: false,
    };

    /// The active SIMD feature flags in a stable, arch-independent order.
    /// Shared by [`CpuFeatures::report`] and [`CpuFeatures::descriptor`] so the
    /// two never drift.
    fn active_flags(&self) -> Vec<&'static str> {
        let mut flags: Vec<&str> = Vec::new();
        for (on, name) in [
            (self.avx2, "avx2"),
            (self.fma, "fma"),
            (self.avx512f, "avx512f"),
            (self.avx512bw, "avx512bw"),
            (self.avx512vl, "avx512vl"),
            (self.avx512vnni, "avx512vnni"),
            (self.neon, "neon"),
            (self.dotprod, "dotprod"),
            (self.i8mm, "i8mm"),
            (self.fp16, "fp16"),
        ] {
            if on {
                flags.push(name);
            }
        }
        flags
    }

    /// Human-readable one-line summary for CLI `inspect` / bug reports, e.g.
    /// `cpu: tier=avx2 [avx2 fma]` or `cpu: tier=neon+dotprod [neon dotprod fp16]`.
    ///
    /// Shares `active_flags` with [`Self::descriptor`], so it moves whenever that
    /// list gains an entry.
    pub fn report(&self) -> String {
        format!(
            "cpu: tier={} [{}]",
            self.tier.label(),
            self.active_flags().join(" ")
        )
    }

    /// Compact CPU-variant descriptor for telemetry: the active SIMD features
    /// joined by commas, e.g. `"neon,dotprod,i8mm,fp16"` or `"avx2,fma"`,
    /// falling back to the tier label (e.g. `"scalar"`) when no accelerated
    /// features are present. Deterministic on a given host, so it can key a
    /// benchmark submission's CPU-variant field (the analog of llama.cpp's ggml
    /// CPU-backend descriptor).
    ///
    /// Stable for a given set of detected features, **not** across cera
    /// versions: adding a flag to `active_flags` changes the string for every
    /// host that reports it. That happened once already, when `fp16` was added,
    /// which moved it on every aarch64 host from v8.2 on (so, in practice, all
    /// of them except pre-FEAT_FP16 parts like Cortex-A53/A72). Treat the
    /// descriptor as a grouping key within a release rather than a join key
    /// across them.
    pub fn descriptor(&self) -> String {
        let flags = self.active_flags();
        if flags.is_empty() {
            self.tier.label().to_string()
        } else {
            flags.join(",")
        }
    }

    /// Verify the host can safely run cera's compiled kernels.
    ///
    /// Every aarch64 GEMV/GEMM entry point in `super::simd::neon` now runtime-
    /// dispatches between its `dotprod` kernel and a plain-NEON fallback, so
    /// `dotprod` is an accelerator rather than a hard requirement and NEON
    /// (mandatory on aarch64) is always sufficient. x86_64 always has a scalar
    /// fallback. This is therefore a no-op today, kept as the hook for any
    /// future hard ISA requirement.
    pub fn ensure_supported(&self) -> Result<(), String> {
        let _ = self;
        Ok(())
    }
}

/// Raw, uncached detection. Prefer [`cpu_features`] — this is exposed only for
/// tests that need a fresh probe.
pub fn detect() -> CpuFeatures {
    // Only the x86_64 / aarch64 blocks below mutate `f`; on other targets
    // (e.g. wasm32) it's built once and returned as-is.
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_mut)
    )]
    let mut f = CpuFeatures::NONE;

    #[cfg(target_arch = "x86_64")]
    {
        f.avx2 = is_x86_feature_detected!("avx2");
        f.fma = is_x86_feature_detected!("fma");
        f.avx512f = is_x86_feature_detected!("avx512f");
        f.avx512bw = is_x86_feature_detected!("avx512bw");
        f.avx512vl = is_x86_feature_detected!("avx512vl");
        f.avx512vnni = is_x86_feature_detected!("avx512vnni");
        // The Q8_0/Q4_0 AVX-512 kernels need only `avx512f` (the 512-bit FMA is
        // part of AVX512F, not the legacy `fma` feature). But at the Avx512 tier
        // Q4_K_M still routes to the AVX2 kernel, which needs `avx2`+`fma`, so
        // require those too: no shipping AVX-512F CPU lacks them, but it keeps
        // the tier honest about every kernel it can dispatch to (e.g. a
        // hypothetical F-without-AVX2 part would fall to Avx2/Scalar, not SIGILL).
        // The kernels use `_mm512_*` intrinsics, so they live behind the
        // default-on `avx512` feature; with it off the tier caps at Avx2.
        //
        // The VNNI tier additionally needs `avx512vl` — the int8 kernels operate
        // on 256-bit vectors (one Q4_0/Q8_0 block is exactly 32 bytes), and
        // `_mm256_dpbusd_epi32` is an AVX512VL-encoded form. Every shipping
        // VNNI part has VL, but requiring it keeps the tier honest rather than
        // trusting that (a VNNI-without-VL part would SIGILL, not fall back).
        f.tier = if f.avx512f
            && f.avx512vl
            && f.avx512vnni
            && f.avx2
            && f.fma
            && cfg!(feature = "avx512")
        {
            CpuTier::Avx512Vnni
        } else if f.avx512f && f.avx2 && f.fma && cfg!(feature = "avx512") {
            CpuTier::Avx512
        } else if f.avx2 && f.fma {
            CpuTier::Avx2
        } else {
            CpuTier::Scalar
        };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64, but probe anyway for honest reporting.
        f.neon = std::arch::is_aarch64_feature_detected!("neon");
        f.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        f.i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        f.fp16 = std::arch::is_aarch64_feature_detected!("fp16");
        // NeonI8mm lights up the Q8_0, Q4_0, Q4_K and Q6_K GEMM kernels;
        // everything else uses the dotprod path (i8mm implies dotprod). Gated
        // behind real i8mm detection so non-i8mm hosts never reach it; the
        // kernels are validated on CI by the `simd-i8mm` job (ubuntu-24.04-arm,
        // Neoverse N2).
        //
        // `fp16` joins the condition because those four kernels widen their
        // block scales with `f16_bits_to_f32`, whose `vcvt_f32_f16` is declared
        // `neon,fp16` by `core::arch`. FEAT_I8MM (v8.6) implies FEAT_FP16
        // (v8.2) on any conformant core, so this never costs a real host the
        // i8mm path; it is here so the tier states the features its kernels
        // declare instead of resting on that implication.
        f.tier = if f.neon && f.dotprod && f.i8mm && f.fp16 {
            CpuTier::NeonI8mm
        } else if f.neon && f.dotprod {
            CpuTier::NeonDotprod
        } else if f.neon {
            CpuTier::Neon
        } else {
            CpuTier::Scalar
        };
    }

    apply_env_override(f)
}

/// Apply the `CERA_CPU_TIER` override. It may only **downgrade** the detected
/// tier — forcing a tier the hardware can't run would reintroduce the UB this
/// module exists to prevent. An unparseable or higher tier is ignored.
fn apply_env_override(f: CpuFeatures) -> CpuFeatures {
    match std::env::var("CERA_CPU_TIER") {
        Ok(val) => with_tier_override(f, CpuTier::parse(&val)),
        Err(_) => f,
    }
}

/// Pure core of [`apply_env_override`], split out so the downgrade-only policy
/// is testable without touching process-global env (which races parallel tests).
fn with_tier_override(mut f: CpuFeatures, forced: Option<CpuTier>) -> CpuFeatures {
    if let Some(t) = forced
        && t < f.tier
    {
        f.tier = t;
    }
    f
}

/// Resolved CPU capabilities for this host, detected once and cached.
///
/// This is the hot-path entry point: dispatchers read `cpu_features().tier`.
/// The detection (and any `CERA_CPU_TIER` env read) happens exactly once.
pub fn cpu_features() -> &'static CpuFeatures {
    static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
    FEATURES.get_or_init(detect)
}

/// Convenience: the resolved [`CpuTier`] for this host.
pub fn cpu_tier() -> CpuTier {
    cpu_features().tier
}

// ── CPU core topology (thread-pool sizing + affinity) ───────────────────────

/// Performance-core topology for sizing the compute thread pool and pinning
/// its workers.
///
/// `perf_core_count` is how many compute threads to run; `pin_cores` are the OS
/// core indices to pin those workers to via `sched_setaffinity` (Linux/Android
/// only; empty elsewhere, where the OS scheduler or Darwin QoS handles
/// placement and affinity masks are inert).
///
/// Both pools default to `perf_core_count`. Widening prefill over the
/// efficiency cores was tried and is a **loss**: see
/// `calibrate::prefill_thread_count`, which owns that policy and the knob that
/// overrides it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreTopology {
    /// Number of compute worker threads to run (always ≥ 1). Also the ceiling
    /// `CERA_DECODE_THREADS` is clamped to.
    pub perf_core_count: usize,
    /// OS core indices to pin workers to, fastest-first, covering **every**
    /// usable core rather than only the performance ones. Empty when the
    /// platform has no usable affinity. A pool wider than this list would leave
    /// its surplus workers unpinned, which is the 35x cliff described on
    /// `apply_thread_override`; both that function and
    /// `calibrate::prefill_thread_count` clamp to this length precisely so that
    /// state is unreachable while pinning is on.
    ///
    /// Listing the efficiency cores here does not put them to work: no pool is
    /// that wide by default. It is what makes a *deliberately* widened pool
    /// (`CERA_THREADS`, `CERA_PREFILL_THREADS`) merely slower instead of
    /// catastrophic, by giving every worker a distinct core. Measured on a
    /// Tensor G4 (4x A520 + 3x A720 + 1x X4), a pool widened past the cores it
    /// could pin cost decode 63.4 → 1.8 tok/s: the pinned workers spin
    /// (`SPIN_BEFORE_PARK`) on exactly the cores the unpinned ones need in
    /// order to reach the per-token barrier.
    pub pin_cores: Vec<usize>,
    /// How many leading entries of `pin_cores` are *performance* cores, as
    /// detected. Distinct from `perf_core_count`, which is a pool width and so
    /// moves with `CERA_THREADS`; this is a fact about the silicon and does
    /// not. `pin_cores` is fastest-first, so `pin_cores[..fast_cores]` is
    /// exactly the fast set.
    ///
    /// Kept separate because the two diverge exactly where it matters:
    /// `CERA_THREADS=8` on a 6P+2E part raises the width to 8, and a consumer
    /// that derived "the perf cores" from the width would then hand out a mask
    /// covering both efficiency cores. `super::threadpool::perf_pinned_cores`
    /// is that consumer.
    pub fast_cores: usize,
    /// Relative speed of each core in `pin_cores`, same order and same length,
    /// normalized so the fastest core is [`WEIGHT_FULL`]. Empty when the host
    /// gives no usable spread (homogeneous, no affinity, or undetectable).
    ///
    /// This is the *magnitude* of the signal `pin_cores` keeps only the
    /// ordering of, and it comes from whichever source ranked the cores:
    /// `cpu_capacity` (the kernel's own EAS throughput scale, and the good
    /// case) or, on a host without it such as an x86 hybrid part,
    /// `cpufreq/cpuinfo_max_freq` ratios, which are a weaker proxy because
    /// clock alone says nothing about IPC. Consumers are expected to tolerate
    /// an approximate weight; `threadpool::worker_chunk_rows` documents why
    /// that is safe. Keeping the numbers costs nothing, and it is what lets
    /// `RowPool` size a worker's chunk to the core it sits on (see
    /// `super::threadpool::pinned_core_weights`). Derived from the same
    /// sorted, sibling-dropped list as `pin_cores` so the two indices cannot
    /// drift: `core_weights[i]` is the weight of the core at `pin_cores[i]`.
    ///
    /// Empty is a meaningful answer, not a missing one: a host with no spread
    /// wants uniform chunks, which is exactly what a consumer that finds no
    /// weight for a worker falls back to.
    pub core_weights: Vec<u32>,
}

impl CoreTopology {
    /// One-line topology summary, for benchmark provenance and bug reports.
    ///
    /// The part that matters when reading a benchmark is `placement`. Two
    /// scheduling behaviours only do anything where cores differ in speed:
    /// capacity-aware chunk sizing scales a worker's chunk by its core's
    /// weight, and the prefill pool drops pinning once it is wider than
    /// `fast_cores`. Both are inert unless `detect_topology_sysfs` returned a
    /// topology, which it does *only* for heterogeneous parts.
    ///
    /// Without this line a flat benchmark trend on a hosted runner reads like
    /// "that work achieved nothing", when the honest reading is "this host
    /// cannot express it". Printing the placement makes that distinction
    /// visible rather than something a reader has to know to look for.
    ///
    /// Deliberately reports the observable *state* and not a guess at its
    /// cause. An empty `pin_cores` has several: `detect_topology_sysfs`
    /// declines homogeneous parts outright, macOS exposes no affinity API at
    /// all, and sysfs can be unreadable. Naming any one of them would be a
    /// false statement on the other hosts, and the distinction that matters
    /// here is the same in every case.
    pub fn report(&self) -> String {
        let placement = if self.pin_cores.is_empty() {
            // The all-cores fallback: no per-core placement, so weight-scaled
            // chunks and pin-widening cannot do anything. `fast_cores` is 0
            // here, which is what makes `prefill_should_pin` short-circuit.
            "flat (no per-core placement, so capacity-aware sizing and \
             pin-widening are both inert)"
                .to_string()
        } else {
            match self.core_weights.iter().min() {
                // Weights are normalized so the fastest core is `WEIGHT_FULL`;
                // the slowest one therefore carries the whole ratio.
                Some(&min) if min < WEIGHT_FULL => format!(
                    "tiered ({:.2}x fastest:slowest)",
                    f64::from(WEIGHT_FULL) / f64::from(min)
                ),
                // Reachable only if a future detector starts returning uniform
                // weights; today `detect_topology_sysfs` rejects that case.
                _ => "uniform (capacity-aware sizing is inert)".to_string(),
            }
        };
        format!(
            "cores: pinnable={} fast={} perf_pool={} placement={}",
            self.pin_cores.len(),
            self.fast_cores,
            self.perf_core_count,
            placement,
        )
    }
}

/// Normalized weight of the fastest core in [`CoreTopology::core_weights`].
/// A power of two so scaling by it is a multiply and a shift, and large enough
/// that a ~5x spread (Tensor G5's 1024 against 207) keeps usable resolution.
pub const WEIGHT_FULL: u32 = 256;

/// Highest plausible CPU index to probe in sysfs. A hard bound so a malformed
/// `/sys` can't loop unboundedly; real parts are far below this.
#[cfg(any(target_os = "linux", target_os = "android"))]
const MAX_CPUS: usize = 512;

/// `cpu_capacity` (kernel EAS scale, 1024 = fastest core on the SoC) at/above
/// which a core counts as a performance core. Prime + performance clusters on
/// current Android big.LITTLE parts sit at/above `CAP_MID`; efficiency cores
/// sit well below (e.g. Tensor G5: E=207, P=824, prime=1024).
#[cfg(any(target_os = "linux", target_os = "android"))]
const CAP_MID: u32 = 400;

/// Resolved core topology for this host, detected once and cached.
pub fn core_topology() -> &'static CoreTopology {
    static TOPOLOGY: OnceLock<CoreTopology> = OnceLock::new();
    TOPOLOGY.get_or_init(detect_topology)
}

/// Parse a `usize ≥ 1` from an environment variable; `None` when unset,
/// unparsable, or zero. Shared by the `CERA_*` tuning knobs.
pub(crate) fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
}

/// Whether an environment variable explicitly switches a feature **off**
/// (`0` / `false` / `off`, case-insensitive). Unset ⇒ `false` (leave it on).
/// Shared by the `CERA_*` kill switches so they all spell "off" the same way.
///
/// Ungated, unlike most of the pool plumbing: [`pinning_disabled`] calls it and
/// is itself reachable from `detect_topology`, which every target builds.
pub(crate) fn env_disabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim();
            ["0", "false", "off"]
                .iter()
                .any(|d| v.eq_ignore_ascii_case(d))
        })
        .unwrap_or(false)
}

/// Whether `CERA_PIN` switches worker pinning off, resolved once.
///
/// Lives here rather than next to `threadpool::pinning_enabled` (which is now
/// its inverse) because the *sizing* policy needs the same answer, and both
/// `threadpool` and `calibrate` are absent on targets that still size a pool.
///
/// It matters to sizing because the clamps in [`apply_thread_override`] and
/// `calibrate::prefill_thread_count` exist to prevent one specific failure:
/// surplus *unpinned* workers contending for cores that *pinned* workers are
/// busy spin-waiting on. With `CERA_PIN=0` nothing is pinned, so that failure
/// cannot occur and the clamp would only stop a deliberate oversubscription
/// sweep from running.
pub(crate) fn pinning_disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| env_disabled("CERA_PIN"))
}

/// Number of performance cores to run compute threads on (convenience over
/// [`core_topology`]). Always ≥ 1.
pub fn performance_core_count() -> usize {
    core_topology().perf_core_count
}

/// Physical (non-SMT) core count, cached. `None` when the platform gives us no
/// way to tell — Windows, BSD, Intel macOS.
///
/// Distinct from [`performance_core_count`], which on a homogeneous host is
/// *all logical* CPUs — double the physical count on an SMT part. Decode sizing
/// has to tell those apart: measured on Zen 5 (16 physical / 32 logical),
/// decode peaked at 20 workers and collapsed at 32 (Llama-1B Q8_0: 29.7 → 17.9
/// tok/s), i.e. *some* SMT helps and full logical width badly does not.
///
/// **Returns `Option` on purpose.** Substituting the logical count when
/// detection fails is not conservative — it is the opposite: a caller deriving
/// a width from `physical` would then derive it from *logical* and land on
/// exactly the full-width configuration measured above as catastrophic.
/// Callers must decide explicitly what to do when the answer is unknown.
pub fn physical_core_count() -> Option<usize> {
    static COUNT: OnceLock<Option<usize>> = OnceLock::new();
    *COUNT.get_or_init(|| {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let Some(n) = linux_physical_cores() {
            return Some(n);
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(n) = macos_sysctl_usize(c"hw.perflevel0.physicalcpu") {
            return Some(n);
        }
        None
    })
}

/// Count distinct `thread_siblings_list` sets — SMT siblings share one, so the
/// number of sets is the physical core count.
///
/// Keyed on the sibling set, **not** `core_id`: Linux restarts `core_id` per
/// *cluster* on multi-cluster ARM device trees (gs101 maps [0,1,2,3, 0,1, 0,1]),
/// so keying on it would collapse whole big/prime clusters into one core. This
/// is the same key, and the same reasoning, as the SMT-sibling drop in
/// [`detect_topology_sysfs`]. Reuses [`read_per_cpu_trimmed`], which ends the
/// scan at a missing `cpuN` *directory* rather than at the first unreadable
/// file, so one offline core cannot truncate the count.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_physical_cores() -> Option<usize> {
    let sets = read_per_cpu_trimmed("topology/thread_siblings_list");
    if sets.is_empty() {
        return None;
    }
    let distinct: std::collections::HashSet<&str> = sets.iter().map(|(_, s)| s.as_str()).collect();
    Some(distinct.len())
}

/// Uncached topology detection. Prefer [`core_topology`]; exposed for tests.
///
/// Precedence: a valid `CERA_THREADS` override sets the thread count (see
/// `apply_thread_override` for the clamp it is subject to); otherwise the
/// platform detector picks the perf-core count; otherwise all logical cores.
pub fn detect_topology() -> CoreTopology {
    let forced = env_usize("CERA_THREADS");
    let pinning_on = !pinning_disabled();

    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Some(topo) = detect_topology_sysfs() {
        return apply_thread_override(topo, forced, pinning_on);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if let Some(count) = macos_perf_core_count() {
        // Apple Silicon is heterogeneous in hardware, but `perflevel0` reports
        // only the P-cores and there is no usable affinity to pin an E-core
        // worker with, so the pools stay on the P-core count.
        return apply_thread_override(
            CoreTopology {
                perf_core_count: count,
                pin_cores: Vec::new(),
                // No pinnable cores, so there is no prefix to hand out.
                fast_cores: 0,
                // Apple Silicon is heterogeneous, but with no affinity a worker
                // is not on a known core, so a per-core weight would describe
                // nothing.
                core_weights: Vec::new(),
            },
            forced,
            pinning_on,
        );
    }

    // Fallback: all logical cores, unpinned (the override still applies).
    let n = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    apply_thread_override(
        CoreTopology {
            perf_core_count: n,
            pin_cores: Vec::new(),
            fast_cores: 0,
            // Unpinned and, on everything that reaches this fallback,
            // homogeneous: uniform chunks are correct here.
            core_weights: Vec::new(),
        },
        forced,
        pinning_on,
    )
}

/// Apply a `CERA_THREADS` override to a detected topology: set the thread count
/// and pin at most that many of the detected cores (fastest-first). Both pools
/// derive from it, and it also lowers the ceiling `CERA_PREFILL_THREADS` is
/// clamped to, so it stays the single global cap. Pure, so the override policy
/// is testable without touching process env.
///
/// **Clamped to the number of pinnable cores**, where the host has any *and*
/// pinning is on. That is not paternalism about power draw; asking for more
/// workers than there are cores to pin does not oversubscribe gracefully, it
/// falls off a cliff. The surplus workers run unpinned, and an unpinned worker
/// must contend for a core that a pinned one is busy spin-waiting on
/// (`SPIN_BEFORE_PARK`) rather than yielding it. Measured on a Tensor G4,
/// `CERA_THREADS=8` against 4 pinnable cores took decode from 63.4 to 1.8
/// tok/s, a 35x loss, silently. Below the core count the override still works
/// in both directions, which is what it is for.
///
/// `pinning_on` is [`pinning_disabled`]'s inverse, passed in rather than read
/// so this stays pure and the policy is testable without touching process env.
/// When it is false the clamp is skipped: the cliff above is *caused by* pinned
/// workers spinning, so with `CERA_PIN=0` there is nothing to fall off, and
/// clamping would only block the deliberate oversubscription sweep that
/// `CERA_PIN=0` is the natural way to set up.
fn apply_thread_override(
    mut topo: CoreTopology,
    forced: Option<usize>,
    pinning_on: bool,
) -> CoreTopology {
    let Some(n) = forced else { return topo };
    let n = if topo.pin_cores.is_empty() || !pinning_on {
        // Nothing will be pinned, either because the platform has no affinity or
        // because `CERA_PIN` turned it off, so there is no pinned/unpinned split
        // to fall off; leave the count to the caller.
        n
    } else {
        let cap = topo.pin_cores.len();
        if n > cap {
            tracing::warn!(
                "cera: CERA_THREADS={n} exceeds the {cap} pinnable cores on this host; \
                 clamping to {cap} (the surplus workers would run unpinned and contend \
                 with spinning ones)"
            );
        }
        n.min(cap)
    };
    topo.perf_core_count = n;
    topo.pin_cores.truncate(n);
    // Truncated in lockstep: `core_weights[i]` describes `pin_cores[i]`, and a
    // weight list left longer than the core list would hand a worker the weight
    // of a core no longer in the pool.
    topo.core_weights.truncate(topo.pin_cores.len());
    // `fast_cores` is a property of the silicon, so the override does not move
    // it; it is only re-clamped so it stays a valid prefix of the truncated
    // list. Without this, `CERA_THREADS=8` on a 6P+2E part would leave anything
    // deriving "the perf cores" from the width masking onto the E-cores.
    topo.fast_cores = topo.fast_cores.min(topo.pin_cores.len());
    topo
}

/// Rescale a fastest-first `(cpu_index, weight)` list to
/// [`CoreTopology::core_weights`]: relative to the fastest entry, in units of
/// [`WEIGHT_FULL`].
///
/// Two separate guards keep the result usable, and they cover different cases:
/// `div_ceil` stops a nonzero-but-tiny capacity from rounding down to zero on
/// an extreme spread, and the `clamp` lower bound covers the one case rounding
/// cannot, a core that reports a capacity of `0`. A zero weight would scale a
/// consumer's chunk to nothing, forcing every consumer to invent its own floor
/// just to make progress. Rounding up biases slow cores large by at most one
/// part in `WEIGHT_FULL`, which is far below the accuracy the consumer needs.
///
/// Not gated to the platforms that call it: it is pure arithmetic, and keeping
/// it buildable everywhere is what lets its rounding be tested on the host
/// rather than only on a device with a heterogeneous `/sys`.
// `allow`, not `expect`: on a non-Linux host this is dead in a normal build but
// live in a test build, so an `expect` would fire as unfulfilled in one of them.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "android")),
    allow(
        dead_code,
        reason = "only the sysfs detector calls it; tested on all hosts"
    )
)]
fn normalize_core_weights(cores: &[(usize, u32)]) -> Vec<u32> {
    // The actual max, not `cores[0]`: callers do pass a fastest-first list, but
    // normalizing against the true maximum means an unsorted one gets wrong-free
    // ratios rather than everything above entry 0 silently clamped to full. This
    // runs once per process, so the extra pass is free. `max(1)` guards an
    // all-zero list rather than dividing by zero.
    let max_weight = u64::from(cores.iter().map(|&(_, w)| w).max().unwrap_or(0)).max(1);
    cores
        .iter()
        .map(|&(_, w)| {
            let scaled = u64::from(w) * u64::from(WEIGHT_FULL);
            // The upper bound cannot bind (dividing by the true maximum keeps
            // every ratio at or below `WEIGHT_FULL`); it is kept so the return
            // type's contract is enforced at the one place that produces it
            // rather than assumed by every consumer. The lower bound does real
            // work, for a `w` of 0. Both applied in `u64`, before narrowing.
            scaled.div_ceil(max_weight).clamp(1, u64::from(WEIGHT_FULL)) as u32
        })
        .collect()
}

/// Detect performance cores from Linux/Android sysfs. Prefers `cpu_capacity`
/// (kernel EAS), falls back to `cpufreq/cpuinfo_max_freq`. Returns `None` for
/// **homogeneous** topologies as well as unreadable ones (→ caller uses the
/// all-cores, unpinned fallback): the cap/pinning policy exists for
/// heterogeneous big.LITTLE parts, and applying it to a homogeneous many-core
/// desktop/server would shrink its pool for no reason.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect_topology_sysfs() -> Option<CoreTopology> {
    // (cpu_index, weight) with higher weight = faster; ranked fastest-first.
    let caps = read_per_cpu_u32("cpu_capacity");
    let mut cores: Vec<(usize, u32)>;

    // Weight at/above which a core counts as a performance core. Both branches
    // below keep the *whole* core list and only derive this threshold: the
    // efficiency cores are excluded from `perf_core_count` (the decode width),
    // not from `pin_cores`, which the wider prefill pool needs in order to pin
    // each of its workers to a distinct core.
    let perf_threshold: u32;

    if !caps.is_empty() {
        // Homogeneous capacities (desktop/server arch_topology) → fallback.
        if caps.iter().all(|&(_, c)| c == caps[0].1) {
            return None;
        }
        perf_threshold = CAP_MID;
        cores = caps;
    } else {
        // No cpu_capacity: rank by max frequency instead.
        let freqs = read_per_cpu_u32("cpufreq/cpuinfo_max_freq");
        let max = freqs.iter().map(|&(_, f)| f).max()?;
        // Cores within 15% of the fastest are the top (big/prime) cluster.
        // If *every* core clears the cutoff the machine is homogeneous → fallback.
        let cutoff = (max / 100) * 85;
        if freqs.iter().all(|&(_, f)| f >= cutoff) {
            return None;
        }
        perf_threshold = cutoff;
        cores = freqs;
    }

    if cores.is_empty() {
        return None;
    }
    // Drop SMT siblings: on x86 hybrid parts (which reach this via the
    // frequency path — no `cpu_capacity` on x86) both hyperthreads of each
    // P-core clear the cutoff, and pinning two workers to one physical core
    // halves its throughput. Two CPUs are siblings iff they report the same
    // `thread_siblings_list` — keyed on that set, NOT on `core_id`, which
    // Linux restarts per *cluster* on multi-cluster ARM device trees (e.g.
    // gs101's map is [0,1,2,3, 0,1, 0,1]) and would wrongly discard whole
    // big/prime clusters as "siblings". ARM cores list only themselves, so
    // this is a no-op there.
    let sibling_sets: std::collections::HashMap<usize, String> =
        read_per_cpu_trimmed("topology/thread_siblings_list")
            .into_iter()
            .collect();
    let mut seen_sets = std::collections::HashSet::new();
    cores.retain(|&(cpu, _)| match sibling_sets.get(&cpu) {
        Some(set) => seen_sets.insert(set.clone()),
        // Unknown siblings → treat as its own physical core.
        None => true,
    });

    // Fastest-first (higher weight first; break ties by lower index for
    // determinism).
    cores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    // Counted *after* the sibling drop, so a hyperthreaded x86 hybrid part
    // counts physical P-cores rather than logical ones. Counting before would
    // size decode to both siblings of each P-core and so pin half its workers
    // onto E-cores.
    //
    // Deliberately uncapped. This used to be clamped to a `MAX_AUTO_THREADS` of
    // 6, which was never measured; it was a guess at where decode plateaus,
    // and it binds on shipping parts: a Snapdragon 8 Elite (2 prime + 6
    // performance) would leave two big cores idle. The real ceilings on decode
    // live in `calibrate` (`DECODE_MAX_AUTO`, `DECODE_WIDTH_MAX`), which are
    // calibrated; a heterogeneous SoC has few enough cores that a second,
    // arbitrary cap here only ever subtracts.
    let perf_core_count = cores
        .iter()
        .filter(|&&(_, w)| w >= perf_threshold)
        .count()
        .max(1);
    let pin_cores: Vec<usize> = cores.iter().map(|&(i, _)| i).collect();
    let fast_cores = perf_core_count;
    // Same list, same order, after the same sibling drop and the same sort, so
    // `core_weights[i]` describes `pin_cores[i]` by construction rather than by
    // a convention two call sites have to keep agreeing on.
    let core_weights = normalize_core_weights(&cores);
    Some(CoreTopology {
        perf_core_count,
        pin_cores,
        fast_cores,
        core_weights,
    })
}

/// Read `/sys/devices/system/cpu/cpuN/<file>` (trimmed) for every present
/// CPU. The scan ends at the first missing `cpuN` *directory*; an unreadable
/// file on a present CPU is skipped, not treated as end-of-list — an offline
/// core (hotplug, `nosmt`) loses its `cpufreq` dir while later cores are
/// still very much present, and breaking there would truncate the topology
/// to whatever enumerated before the hole.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_per_cpu_trimmed(file: &str) -> Vec<(usize, String)> {
    let mut values = Vec::new();
    for cpu in 0..MAX_CPUS {
        let dir = format!("/sys/devices/system/cpu/cpu{cpu}");
        if !std::path::Path::new(&dir).is_dir() {
            break;
        }
        if let Ok(s) = std::fs::read_to_string(format!("{dir}/{file}")) {
            values.push((cpu, s.trim().to_string()));
        }
    }
    values
}

/// [`read_per_cpu_trimmed`], parsed as `u32` (unparsable entries skipped).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_per_cpu_u32(file: &str) -> Vec<(usize, u32)> {
    read_per_cpu_trimmed(file)
        .into_iter()
        .filter_map(|(cpu, s)| s.parse().ok().map(|v| (cpu, v)))
        .collect()
}

/// Performance-core count on Apple Silicon via `hw.perflevel0.logicalcpu`
/// (no subprocess). `None` if the sysctl is unavailable — or under Miri,
/// which cannot interpret the foreign call; the topology sits on the GEMV
/// hot path, and returning `None` keeps the full test suite Miri-runnable
/// via the `available_parallelism` fallback.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn macos_perf_core_count() -> Option<usize> {
    macos_sysctl_usize(c"hw.perflevel0.logicalcpu")
}

/// Read a positive `i32` sysctl by name. `None` if the sysctl is unavailable —
/// or under Miri, which cannot interpret the foreign call; the topology sits on
/// the GEMV hot path, and returning `None` keeps the full test suite
/// Miri-runnable via the `available_parallelism` fallback.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn macos_sysctl_usize(name: &std::ffi::CStr) -> Option<usize> {
    if cfg!(miri) {
        return None;
    }
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let ret = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    (ret == 0 && value > 0).then_some(value as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_has_at_least_one_thread() {
        let topo = detect_topology();
        assert!(topo.perf_core_count >= 1);
        // Every worker has a distinct core to pin to, or the host has no
        // affinity at all. An in-between state is the oversubscribed
        // configuration `apply_thread_override` exists to prevent.
        assert!(topo.pin_cores.is_empty() || topo.pin_cores.len() >= topo.perf_core_count);
        // Cached accessor agrees with a fresh detect (modulo env, which both read).
        assert_eq!(core_topology().perf_core_count, performance_core_count());
    }

    #[test]
    fn thread_override_sets_count_and_caps_pins() {
        let base = CoreTopology {
            perf_core_count: 3,
            pin_cores: vec![7, 6, 5],
            fast_cores: 3,
            core_weights: vec![WEIGHT_FULL, WEIGHT_FULL, WEIGHT_FULL],
        };
        // Fewer threads than detected cores → pin the fastest N.
        let two = apply_thread_override(base.clone(), Some(2), true);
        assert_eq!(two.perf_core_count, 2);
        assert_eq!(two.pin_cores, vec![7, 6]);
        // No override → unchanged.
        assert_eq!(apply_thread_override(base.clone(), None, true), base);
    }

    /// `CERA_THREADS` moves the pool *width*; it must not move `fast_cores`,
    /// which is what `threadpool::perf_pinned_cores` slices the rayon mask from.
    /// Letting the two move together put the mask on the efficiency cores.
    #[test]
    fn thread_override_leaves_fast_cores_alone() {
        // 6 performance cores followed by 2 efficiency cores, fastest-first.
        let big_little = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![7, 6, 5, 4, 3, 2, 1, 0],
            fast_cores: 6,
            core_weights: vec![256, 206, 206, 206, 206, 206, 52, 52],
        };
        // Widening past the perf cores must not widen the fast prefix.
        let wide = apply_thread_override(big_little.clone(), Some(8), true);
        assert_eq!(wide.perf_core_count, 8);
        assert_eq!(wide.fast_cores, 6, "widening leaked onto the E-cores");
        assert_eq!(&wide.pin_cores[..wide.fast_cores], &[7, 6, 5, 4, 3, 2]);
        // Narrowing below the perf cores truncates `pin_cores`, so the prefix
        // has to shrink with it or it would index out of bounds.
        let narrow = apply_thread_override(big_little, Some(3), true);
        assert_eq!(narrow.fast_cores, 3);
        assert!(narrow.fast_cores <= narrow.pin_cores.len());
    }

    /// Asking for more workers than there are pinnable cores is the
    /// configuration measured at 63.4 → 1.8 tok/s on a Tensor G4, so it is
    /// clamped rather than honored. Without pins there is no cliff to avoid and
    /// the count passes through.
    #[test]
    fn thread_override_clamps_to_pinnable_cores() {
        let pinned = CoreTopology {
            perf_core_count: 3,
            pin_cores: vec![7, 6, 5],
            fast_cores: 3,
            core_weights: vec![WEIGHT_FULL, WEIGHT_FULL, WEIGHT_FULL],
        };
        let five = apply_thread_override(pinned, Some(5), true);
        assert_eq!(five.perf_core_count, 3);
        assert_eq!(five.pin_cores, vec![7, 6, 5]);

        // No affinity (macOS/desktop fallback) → nothing to clamp against.
        let unpinned = CoreTopology {
            perf_core_count: 8,
            pin_cores: Vec::new(),
            fast_cores: 0,
            core_weights: Vec::new(),
        };
        let wide = apply_thread_override(unpinned, Some(16), true);
        assert_eq!(wide.perf_core_count, 16);
    }

    /// A Tensor G5's `cpu_capacity` values rescaled to `WEIGHT_FULL`. The
    /// numbers matter to the thing that consumes them: a worker on the A520
    /// takes ~1/5 the rows per steal, which is what stops it from holding the
    /// fork-join barrier for ~5x a chunk.
    #[test]
    fn core_weights_rescale_relative_to_the_fastest_core() {
        let g5 = [(7, 1024u32), (6, 824), (5, 824), (1, 207), (0, 207)];
        assert_eq!(
            normalize_core_weights(&g5),
            vec![256, 206, 206, 52, 52],
            "weights are not relative to the prime core"
        );
    }

    /// No spread ⇒ every worker keeps the full chunk, which is the property that
    /// makes this inert on homogeneous hosts.
    #[test]
    fn core_weights_are_full_when_cores_are_equal() {
        let homogeneous = [(0, 1024u32), (1, 1024), (2, 1024), (3, 1024)];
        assert!(
            normalize_core_weights(&homogeneous)
                .iter()
                .all(|&w| w == WEIGHT_FULL)
        );
    }

    /// A weight of zero would scale a consumer's chunk to nothing, so the
    /// rounding has to keep even an absurdly slow core above zero.
    #[test]
    fn core_weights_never_normalize_to_zero() {
        // A spread far wider than any real part, plus the degenerate all-zero
        // list a malformed `/sys` could produce.
        let extreme = [(0, u32::MAX), (1, 1)];
        assert_eq!(normalize_core_weights(&extreme), vec![256, 1]);
        assert_eq!(normalize_core_weights(&[(0, 0), (1, 0)]), vec![1, 1]);
        assert!(normalize_core_weights(&[]).is_empty());
    }

    /// `core_weights[i]` describes `pin_cores[i]`, so a `CERA_THREADS` override
    /// that truncates one must truncate the other. Leaving the weights long
    /// would not fail loudly; it would silently hand a worker the weight of a
    /// core that is no longer in the pool.
    #[test]
    fn thread_override_keeps_weights_aligned_with_pins() {
        let big_little = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![7, 6, 5, 4, 3, 2, 1, 0],
            fast_cores: 6,
            core_weights: vec![256, 206, 206, 206, 206, 206, 52, 52],
        };
        let narrow = apply_thread_override(big_little.clone(), Some(3), true);
        assert_eq!(narrow.pin_cores.len(), narrow.core_weights.len());
        assert_eq!(narrow.core_weights, vec![256, 206, 206]);
        // Widening past the detected cores clamps the width, so the lists still
        // match rather than one growing to a length the other cannot cover.
        let wide = apply_thread_override(big_little, Some(16), true);
        assert_eq!(wide.pin_cores.len(), wide.core_weights.len());
    }

    /// With `CERA_PIN=0` nothing is pinned, so the clamp's justification (surplus
    /// unpinned workers contending with pinned spinning ones) does not apply and
    /// the override must pass through. Before this, `CERA_PIN=0 CERA_THREADS=16`
    /// on an 8-core part was still silently reduced to 8, blocking the very
    /// oversubscription sweep that turning pinning off is the way to set up.
    #[test]
    fn thread_override_is_not_clamped_when_pinning_is_off() {
        let big_little = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![7, 6, 5, 4, 3, 2, 1, 0],
            fast_cores: 6,
            core_weights: vec![256, 206, 206, 206, 206, 206, 52, 52],
        };
        let pinned = apply_thread_override(big_little.clone(), Some(16), true);
        assert_eq!(pinned.perf_core_count, 8, "clamped to the pinnable cores");

        let unpinned = apply_thread_override(big_little, Some(16), false);
        assert_eq!(
            unpinned.perf_core_count, 16,
            "CERA_PIN=0 still clamped, so an oversubscription sweep is impossible"
        );
        // Nothing is pinned, so neither the core list nor the weights parallel to
        // it are truncated to the width: there is no per-worker core to hand out,
        // and so no weight that describes one.
        assert_eq!(unpinned.pin_cores.len(), 8);
        assert_eq!(unpinned.core_weights.len(), 8);
    }

    #[test]
    fn tier_ordering_is_monotonic_per_arch() {
        assert!(CpuTier::Scalar < CpuTier::Avx2);
        assert!(CpuTier::Avx2 < CpuTier::Avx512);
        assert!(CpuTier::Scalar < CpuTier::Neon);
        assert!(CpuTier::Neon < CpuTier::NeonDotprod);
        assert!(CpuTier::NeonDotprod < CpuTier::NeonI8mm);
    }

    #[test]
    fn descriptor_is_compact_sorted_and_never_empty() {
        // Scalar host with no accelerated features → the tier label, never "".
        assert_eq!(CpuFeatures::NONE.descriptor(), "scalar");

        // aarch64-shape flags join comma-separated in the stable order.
        //
        // This fixture is deliberately one detection cannot produce: `detect`
        // will not report `NeonI8mm` without `fp16` (FEAT_I8MM is v8.6, FEAT_FP16
        // v8.2), so it pins the ordering rather than a real host. The realistic
        // shape is below, and it is the one that matters: `fp16` joined
        // `active_flags` after this descriptor was already documented as stable,
        // which moved the string on every aarch64 host from v8.2 on. Anything keying
        // benchmark history on it sees a discontinuity there, not a regression.
        let neon = CpuFeatures {
            tier: CpuTier::NeonI8mm,
            neon: true,
            dotprod: true,
            i8mm: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(neon.descriptor(), "neon,dotprod,i8mm");

        // What an i8mm-class host actually reports, and what an M1 reports.
        let real = CpuFeatures { fp16: true, ..neon };
        assert_eq!(real.descriptor(), "neon,dotprod,i8mm,fp16");
        let m1 = CpuFeatures {
            tier: CpuTier::NeonDotprod,
            neon: true,
            dotprod: true,
            fp16: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(m1.descriptor(), "neon,dotprod,fp16");

        // x86-shape flags likewise; report() shares the same active-flag set.
        let x86 = CpuFeatures {
            tier: CpuTier::Avx2,
            avx2: true,
            fma: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(x86.descriptor(), "avx2,fma");
        assert!(x86.report().contains("[avx2 fma]"));
    }

    #[test]
    fn detect_is_stable_and_cached() {
        // Cached accessor returns the same value as a fresh probe (modulo the
        // env override, which both apply).
        assert_eq!(*cpu_features(), detect());
        assert_eq!(cpu_features().tier, cpu_tier());
    }

    #[test]
    fn detected_tier_matches_arch() {
        let t = detect().tier;
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(
            t,
            CpuTier::Scalar | CpuTier::Avx2 | CpuTier::Avx512 | CpuTier::Avx512Vnni
        ));
        #[cfg(target_arch = "aarch64")]
        assert!(matches!(
            t,
            CpuTier::Scalar | CpuTier::Neon | CpuTier::NeonDotprod | CpuTier::NeonI8mm
        ));
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(t, CpuTier::Scalar);
    }

    #[test]
    fn env_override_only_downgrades() {
        let at = |t: CpuTier| CpuFeatures {
            tier: t,
            ..CpuFeatures::NONE
        };
        // Lower tier → applied.
        assert_eq!(
            with_tier_override(at(CpuTier::Avx2), Some(CpuTier::Scalar)).tier,
            CpuTier::Scalar
        );
        // Higher tier → ignored (never upgrade onto unsupported hardware).
        assert_eq!(
            with_tier_override(at(CpuTier::Avx2), Some(CpuTier::Avx512)).tier,
            CpuTier::Avx2
        );
        // Equal tier → no-op.
        assert_eq!(
            with_tier_override(at(CpuTier::NeonDotprod), Some(CpuTier::NeonDotprod)).tier,
            CpuTier::NeonDotprod
        );
        // Unparseable (None) → no-op.
        assert_eq!(
            with_tier_override(at(CpuTier::Avx2), None).tier,
            CpuTier::Avx2
        );
    }

    #[test]
    fn tier_label_roundtrips_through_parse() {
        // `parse` is arch-gated, so only the current arch's tiers round-trip.
        // Built per-arch rather than extended in place: on a target that is
        // neither, both `extend` calls vanish and the `mut` becomes an
        // `unused_mut` error under `-D warnings`.
        #[cfg(target_arch = "x86_64")]
        let tiers = vec![
            CpuTier::Scalar,
            CpuTier::Avx2,
            CpuTier::Avx512,
            CpuTier::Avx512Vnni,
        ];
        #[cfg(target_arch = "aarch64")]
        let tiers = vec![
            CpuTier::Scalar,
            CpuTier::Neon,
            CpuTier::NeonDotprod,
            CpuTier::NeonI8mm,
        ];
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let tiers = vec![CpuTier::Scalar];
        for t in tiers {
            assert_eq!(CpuTier::parse(t.label()), Some(t), "label {:?}", t.label());
        }
    }

    #[test]
    fn cross_arch_override_label_is_rejected() {
        // The label for a tier from the *other* arch must not parse — otherwise
        // it could be applied as a bogus "downgrade" (e.g. `avx2` on aarch64).
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(CpuTier::parse("avx2"), None);
            assert_eq!(CpuTier::parse("avx512"), None);
        }
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(CpuTier::parse("neon"), None);
            assert_eq!(CpuTier::parse("i8mm"), None);
        }
    }

    /// The topology line is benchmark provenance, so the three cases must stay
    /// distinguishable. In particular an empty `core_weights` must not be
    /// reported as "homogeneous" when the real cause is that the platform
    /// exposes no affinity: macOS hits that path on heterogeneous silicon, and
    /// claiming the hardware is uniform there would be a false statement.
    #[test]
    fn topology_report_distinguishes_flat_from_tiered() {
        let unknown = CoreTopology {
            perf_core_count: 8,
            pin_cores: Vec::new(),
            fast_cores: 0,
            core_weights: Vec::new(),
        };
        let r = unknown.report();
        assert!(r.contains("placement=flat"), "{r}");

        let homogeneous = CoreTopology {
            perf_core_count: 4,
            pin_cores: vec![0, 1, 2, 3],
            fast_cores: 4,
            core_weights: vec![WEIGHT_FULL; 4],
        };
        let r = homogeneous.report();
        assert!(r.contains("placement=uniform"), "{r}");

        // Tensor G5's real spread: 1024 against 207, which normalizes to 256
        // against 52 and reports as ~4.9x.
        let heterogeneous = CoreTopology {
            perf_core_count: 6,
            pin_cores: vec![0, 1, 2, 3, 4, 5, 6, 7],
            fast_cores: 6,
            core_weights: vec![WEIGHT_FULL, WEIGHT_FULL, 206, 206, 206, 206, 52, 52],
        };
        let r = heterogeneous.report();
        assert!(r.contains("placement=tiered"), "{r}");
        assert!(r.contains("4.92x"), "{r}");
        assert!(r.contains("pinnable=8"), "{r}");
        assert!(r.contains("fast=6"), "{r}");
    }

    #[test]
    fn report_includes_tier_label() {
        let r = cpu_features().report();
        assert!(r.contains("tier="));
        assert!(r.contains(cpu_tier().label()));
    }
}

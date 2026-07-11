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
//! is [`CpuTier::Avx512`] (Q8_0/Q4_0 `vec_dot`; needs the default-on `avx512`
//! crate feature, else [`CpuTier::Avx2`]); on aarch64 it is [`CpuTier::NeonI8mm`]
//! (Q8_0 GEMM) down to [`CpuTier::NeonDotprod`]. The raw feature bools (e.g.
//! [`CpuFeatures::avx512vnni`]) are detected and exposed regardless, for
//! diagnostics and so future kernels can light up without re-plumbing.

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
    /// x86_64 AVX2 + FMA.
    Avx2,
    /// x86_64 AVX-512 — 512-bit f32 `vec_dot` for Q8_0/Q4_0 (needs only
    /// `avx512f`). Produced when the default-on `avx512` crate feature is
    /// enabled; disable it for a Rust 1.85-compatible x86 build.
    Avx512,
    /// aarch64 baseline NEON.
    Neon,
    /// aarch64 NEON + dotprod (FEAT_DotProd, `vdotq_s32`).
    NeonDotprod,
    /// aarch64 NEON + i8mm (FEAT_I8MM, `vmmlaq_s32`) — Q8_0 GEMM only; other ops
    /// use the dotprod path (i8mm implies dotprod).
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
    pub avx512vnni: bool,
    // ── aarch64 ──
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
}

impl CpuFeatures {
    const NONE: CpuFeatures = CpuFeatures {
        tier: CpuTier::Scalar,
        avx2: false,
        fma: false,
        avx512f: false,
        avx512bw: false,
        avx512vnni: false,
        neon: false,
        dotprod: false,
        i8mm: false,
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
            (self.avx512vnni, "avx512vnni"),
            (self.neon, "neon"),
            (self.dotprod, "dotprod"),
            (self.i8mm, "i8mm"),
        ] {
            if on {
                flags.push(name);
            }
        }
        flags
    }

    /// Human-readable one-line summary for CLI `inspect` / bug reports, e.g.
    /// `cpu: tier=avx2 [avx2 fma]` or `cpu: tier=neon+dotprod [neon dotprod]`.
    pub fn report(&self) -> String {
        format!(
            "cpu: tier={} [{}]",
            self.tier.label(),
            self.active_flags().join(" ")
        )
    }

    /// Compact, stable CPU-variant descriptor for telemetry — the active SIMD
    /// features joined by commas, e.g. `"neon,dotprod,i8mm"` or `"avx2,fma"`,
    /// falling back to the tier label (e.g. `"scalar"`) when no accelerated
    /// features are present. Deterministic on a given host, so it can key a
    /// benchmark submission's CPU-variant field (the analog of llama.cpp's ggml
    /// CPU-backend descriptor).
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
        f.avx512vnni = is_x86_feature_detected!("avx512vnni");
        // The Q8_0/Q4_0 AVX-512 kernels need only `avx512f` (the 512-bit FMA is
        // part of AVX512F, not the legacy `fma` feature). But at the Avx512 tier
        // Q4_K_M still routes to the AVX2 kernel, which needs `avx2`+`fma`, so
        // require those too: no shipping AVX-512F CPU lacks them, but it keeps
        // the tier honest about every kernel it can dispatch to (e.g. a
        // hypothetical F-without-AVX2 part would fall to Avx2/Scalar, not SIGILL).
        // The kernels use Rust-1.89 `_mm512_*` intrinsics, past the crate's 1.85
        // MSRV, so they live behind the default-on `avx512` feature; with it off
        // the tier caps at Avx2 and the x86 build stays 1.85-compatible. VNNI is
        // detected for diagnostics only.
        f.tier = if f.avx512f && f.avx2 && f.fma && cfg!(feature = "avx512") {
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
        // NeonI8mm currently lights up only the Q8_0 GEMM kernel; everything
        // else uses the dotprod path (i8mm implies dotprod). Gated behind real
        // i8mm detection so non-i8mm hosts never reach it; the kernel is
        // validated on CI by the `simd-i8mm` job (ubuntu-24.04-arm, Neoverse N2).
        f.tier = if f.neon && f.dotprod && f.i8mm {
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
    if let Some(t) = forced {
        if t < f.tier {
            f.tier = t;
        }
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
/// only — empty elsewhere, where the OS scheduler or Darwin QoS handles
/// placement and affinity masks are inert).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreTopology {
    /// Number of compute worker threads to run (always ≥ 1).
    pub perf_core_count: usize,
    /// OS core indices to pin workers to, fastest-first. Empty when the
    /// platform has no usable affinity; when shorter than `perf_core_count`,
    /// the surplus workers run unpinned.
    pub pin_cores: Vec<usize>,
}

/// Upper bound on the auto-detected big-core count (and thus decode/prefill
/// pool width). Decode scales across the performance cores but plateaus around
/// the big-core count, beyond which more threads only add barrier/scheduling
/// overhead and power draw; 6 covers current big.LITTLE mobile (Tensor G5,
/// Snapdragon 8-series) with a sensible power/thermal margin. On a SoC with
/// more than 6 performance cores (e.g. an 8-prime part) this leaves a couple
/// idle — deliberately conservative; `CERA_THREADS` overrides in both
/// directions for tuning. The homogeneous/unpinned decode fallback has its own,
/// separate ceiling (`calibrate::DECODE_MAX_AUTO`); keep the two in mind
/// together when retuning either.
#[cfg(any(target_os = "linux", target_os = "android"))]
const MAX_AUTO_THREADS: usize = 6;

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

/// Number of performance cores to run compute threads on (convenience over
/// [`core_topology`]). Always ≥ 1.
pub fn performance_core_count() -> usize {
    core_topology().perf_core_count
}

/// Uncached topology detection. Prefer [`core_topology`]; exposed for tests.
///
/// Precedence: a valid `CERA_THREADS` override sets the thread count (workers
/// still pin to the detected perf cores where available); otherwise the
/// platform detector picks the perf-core count; otherwise all logical cores.
pub fn detect_topology() -> CoreTopology {
    let forced = env_usize("CERA_THREADS");

    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Some(topo) = detect_topology_sysfs() {
        return apply_thread_override(topo, forced);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if let Some(count) = macos_perf_core_count() {
        return apply_thread_override(
            CoreTopology {
                perf_core_count: count,
                pin_cores: Vec::new(),
            },
            forced,
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
        },
        forced,
    )
}

/// Apply a `CERA_THREADS` override to a detected topology: set the thread count
/// and pin at most that many of the detected cores (fastest-first). Pure, so
/// the override policy is testable without touching process env.
fn apply_thread_override(mut topo: CoreTopology, forced: Option<usize>) -> CoreTopology {
    if let Some(n) = forced {
        topo.perf_core_count = n;
        // Never pin more cores than were detected; surplus workers run unpinned.
        topo.pin_cores.truncate(n);
    }
    topo
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

    if !caps.is_empty() {
        // Homogeneous capacities (desktop/server arch_topology) → fallback.
        if caps.iter().all(|&(_, c)| c == caps[0].1) {
            return None;
        }
        cores = caps.into_iter().filter(|&(_, c)| c >= CAP_MID).collect();
    } else {
        // No cpu_capacity — rank the top frequency cluster instead.
        let freqs = read_per_cpu_u32("cpufreq/cpuinfo_max_freq");
        let max = freqs.iter().map(|&(_, f)| f).max()?;
        // Keep cores within 15% of the fastest — the top (big/prime) cluster.
        // If *every* core clears the cutoff the machine is homogeneous → fallback.
        let cutoff = (max / 100) * 85;
        if freqs.iter().all(|&(_, f)| f >= cutoff) {
            return None;
        }
        cores = freqs.into_iter().filter(|&(_, f)| f >= cutoff).collect();
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
    // determinism), then cap.
    cores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    cores.truncate(MAX_AUTO_THREADS);
    let pin_cores: Vec<usize> = cores.iter().map(|&(i, _)| i).collect();
    Some(CoreTopology {
        perf_core_count: pin_cores.len(),
        pin_cores,
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
    let name = c"hw.perflevel0.logicalcpu";
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
    if ret == 0 && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_has_at_least_one_thread() {
        let topo = detect_topology();
        assert!(topo.perf_core_count >= 1);
        // Cached accessor agrees with a fresh detect (modulo env, which both read).
        assert_eq!(core_topology().perf_core_count, performance_core_count());
    }

    #[test]
    fn thread_override_sets_count_and_caps_pins() {
        let base = CoreTopology {
            perf_core_count: 3,
            pin_cores: vec![7, 6, 5],
        };
        // Fewer threads than detected cores → pin the fastest N.
        let two = apply_thread_override(base.clone(), Some(2));
        assert_eq!(two.perf_core_count, 2);
        assert_eq!(two.pin_cores, vec![7, 6]);
        // More threads than detected cores → keep all pins, surplus unpinned.
        let five = apply_thread_override(base.clone(), Some(5));
        assert_eq!(five.perf_core_count, 5);
        assert_eq!(five.pin_cores, vec![7, 6, 5]);
        // No override → unchanged.
        assert_eq!(apply_thread_override(base.clone(), None), base);
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
        let neon = CpuFeatures {
            tier: CpuTier::NeonI8mm,
            neon: true,
            dotprod: true,
            i8mm: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(neon.descriptor(), "neon,dotprod,i8mm");

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
            CpuTier::Scalar | CpuTier::Avx2 | CpuTier::Avx512
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
        let mut tiers = vec![CpuTier::Scalar];
        #[cfg(target_arch = "x86_64")]
        tiers.extend([CpuTier::Avx2, CpuTier::Avx512]);
        #[cfg(target_arch = "aarch64")]
        tiers.extend([CpuTier::Neon, CpuTier::NeonDotprod, CpuTier::NeonI8mm]);
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

    #[test]
    fn report_includes_tier_label() {
        let r = cpu_features().report();
        assert!(r.contains("tier="));
        assert!(r.contains(cpu_tier().label()));
    }
}

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
//! [`CpuFeatures::tier`] reports the best tier cera actually has *kernels* for
//! (today: [`CpuTier::Avx2`] on x86, [`CpuTier::NeonDotprod`] on aarch64), so a
//! dispatcher can never route to a kernel that doesn't exist. The raw feature
//! bools (e.g. [`CpuFeatures::avx512f`]) are detected and exposed regardless,
//! for diagnostics and so future kernels can light up without re-plumbing.

use std::sync::OnceLock;

/// Ordered CPU capability tier. Higher is more capable.
///
/// `Ord` is derived from declaration order, so within a single architecture the
/// comparison is meaningful (`Scalar < Avx2 < Avx512`, `Scalar < Neon <
/// NeonDotprod < NeonI8mm`). Cross-architecture comparisons are nonsensical but
/// harmless — only one architecture's variants are ever produced at runtime.
///
/// [`Avx512`](CpuTier::Avx512) and [`NeonI8mm`](CpuTier::NeonI8mm) are reserved
/// for when those kernels land; [`detect`] does not produce them yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CpuTier {
    /// Portable scalar reference path. Always available.
    Scalar,
    /// x86_64 AVX2 + FMA.
    Avx2,
    /// x86_64 AVX-512 (F + BW, optionally VNNI). Reserved — no kernels yet.
    Avx512,
    /// aarch64 baseline NEON.
    Neon,
    /// aarch64 NEON + dotprod (FEAT_DotProd, `vdotq_s32`).
    NeonDotprod,
    /// aarch64 NEON + i8mm (FEAT_I8MM, `vmmlaq_s32`). Reserved — no kernels yet.
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

    /// Human-readable one-line summary for CLI `inspect` / bug reports, e.g.
    /// `cpu: tier=avx2 [avx2 fma]` or `cpu: tier=neon+dotprod [neon dotprod]`.
    pub fn report(&self) -> String {
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
        format!("cpu: tier={} [{}]", self.tier.label(), flags.join(" "))
    }

    /// Verify the host can safely run cera's compiled kernels.
    ///
    /// Every aarch64 GEMV/GEMM entry point in [`super::simd::neon`] now runtime-
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
    let mut f = CpuFeatures::NONE;

    #[cfg(target_arch = "x86_64")]
    {
        f.avx2 = is_x86_feature_detected!("avx2");
        f.fma = is_x86_feature_detected!("fma");
        f.avx512f = is_x86_feature_detected!("avx512f");
        f.avx512bw = is_x86_feature_detected!("avx512bw");
        f.avx512vnni = is_x86_feature_detected!("avx512vnni");
        // Avx512 covers the 512-bit f32 Q8_0/Q4_0 `vec_dot` kernels (avx512f is
        // sufficient for those; it implies AVX2+FMA, which the Q4_K_M fallback
        // to the AVX2 kernel needs). VNNI is detected for diagnostics but not
        // yet exploited — a true int8 path on x86 is a separate change.
        f.tier = if f.avx512f {
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
        // else uses the dotprod path (i8mm implies dotprod). It is UNVERIFIED on
        // available hardware (see `simd::neon::gemm_q8_0_q8_0_neon_i8mm`) — gated
        // behind real i8mm detection so non-i8mm hosts never reach it.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering_is_monotonic_per_arch() {
        assert!(CpuTier::Scalar < CpuTier::Avx2);
        assert!(CpuTier::Avx2 < CpuTier::Avx512);
        assert!(CpuTier::Scalar < CpuTier::Neon);
        assert!(CpuTier::Neon < CpuTier::NeonDotprod);
        assert!(CpuTier::NeonDotprod < CpuTier::NeonI8mm);
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

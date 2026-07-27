//! Best-effort host physical-memory queries for proactive load pre-flight.
//!
//! cera's weight and KV-cache allocations are infallible (`Vec::with_capacity`
//! / owned-buffer reads), so loading a model larger than available RAM aborts
//! the process rather than returning an error. A benchmark harness that runs
//! many models on constrained devices (e.g. Pipette on Android) uses these
//! helpers to estimate the fit and **skip** an over-large model *before*
//! loading, instead of crashing.
//!
//! Best-effort by design: [`available_memory_bytes`] returns `None` on
//! platforms where it can't query (currently everything except Linux/Android),
//! and the caller decides whether to proceed. Converting the abort itself into
//! a recoverable `Err` at the allocation site is deliberately out of scope —
//! that needs fallible allocation (`try_reserve`) threaded through the load path.

/// Currently-available physical memory in bytes, or `None` when it can't be
/// determined on this platform.
///
/// Linux/Android parse `MemAvailable` from `/proc/meminfo` — the kernel's own
/// estimate of what can be allocated without swapping, which is more honest for
/// a fit check than `MemFree`. Other platforms (macOS, iOS, wasm) return `None`
/// today; consumers there rely on their own memory gate.
pub fn available_memory_bytes() -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_mem_available_kib(&meminfo).map(|kib| kib.saturating_mul(1024))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

/// Parse the `MemAvailable:` value (in kiB) out of `/proc/meminfo` contents.
/// Split out so the parse is unit-testable without a real `/proc`.
#[cfg(any(target_os = "linux", target_os = "android", test))]
fn parse_mem_available_kib(meminfo: &str) -> Option<u64> {
    // Line shape: `MemAvailable:   8388608 kB`.
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
}

/// The fit predicate itself, against a caller-supplied `available` figure.
///
/// Split out from [`fits_in_available_memory`] so the boundary behaviour can be
/// tested against fixed numbers. Testing it through the public function means
/// sampling live memory twice — once to pick the input, once inside the call —
/// and asserting on a value that moves in between, which is a race rather than
/// a test (see `fits_is_exact_at_the_boundary`).
fn fits_within(required_bytes: u64, headroom_bytes: u64, available: u64) -> bool {
    required_bytes.saturating_add(headroom_bytes) <= available
}

/// Whether `required_bytes` fits in currently-available physical memory leaving
/// at least `headroom_bytes` free. `None` when available memory can't be queried
/// (see [`available_memory_bytes`]) — the caller then decides whether to proceed.
///
/// `headroom_bytes` is the caller's safety margin for fragmentation, other live
/// allocations, and the OS low-memory killer; there is no baked-in policy here.
///
/// The answer is a snapshot: it reflects one `/proc/meminfo` read and is stale
/// the moment it returns. Callers use it to skip an obviously-too-large model,
/// not as a guarantee that the load will succeed.
pub fn fits_in_available_memory(required_bytes: u64, headroom_bytes: u64) -> Option<bool> {
    let available = available_memory_bytes()?;
    Some(fits_within(required_bytes, headroom_bytes, available))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mem_available_line() {
        let sample = "MemTotal:       16000000 kB\n\
                      MemFree:         2000000 kB\n\
                      MemAvailable:    8388608 kB\n\
                      Buffers:          100000 kB\n";
        assert_eq!(parse_mem_available_kib(sample), Some(8_388_608));
    }

    #[test]
    fn missing_mem_available_is_none() {
        assert_eq!(parse_mem_available_kib("MemTotal: 16000000 kB\n"), None);
        assert_eq!(parse_mem_available_kib(""), None);
    }

    /// The fit boundary, against fixed numbers rather than a live sample.
    ///
    /// This replaces an assertion that read `available_memory_bytes()` and then
    /// called `fits_in_available_memory(avail + 1, 0)` expecting `Some(false)`.
    /// That is a TOCTOU race: the public function samples `/proc/meminfo`
    /// *again*, and if free memory rose in between — a page cache reclaim, a
    /// sibling test's allocation being dropped — then `avail + 1` fits after all
    /// and the assertion inverts. It failed exactly that way on a busy CI runner.
    #[test]
    fn fits_is_exact_at_the_boundary() {
        assert!(fits_within(0, 0, 0), "zero always fits, even in nothing");
        assert!(fits_within(100, 0, 100), "exactly available must fit");
        assert!(!fits_within(101, 0, 100), "one byte over must not");
    }

    #[test]
    fn headroom_is_deducted_from_the_budget() {
        assert!(fits_within(60, 40, 100), "required + headroom == available");
        assert!(
            !fits_within(61, 40, 100),
            "one byte over once headroom is paid"
        );
        assert!(
            !fits_within(0, 101, 100),
            "headroom alone can exceed available"
        );
    }

    /// `required + headroom` must not wrap. A plain `+` would overflow (panic in
    /// debug, wrap in release) and a wrapped sum is a *small* number, which
    /// would report a colossal request as fitting comfortably.
    ///
    /// Note the one case `saturating_add` cannot distinguish: when the sum
    /// saturates to `u64::MAX` and `available` is also `u64::MAX`, it compares
    /// equal and reports a fit. Left alone deliberately — `available` comes from
    /// `MemAvailable`, so reaching `u64::MAX` would mean 16 exabytes of free
    /// RAM. Asserting the "right" answer there would mean carrying a
    /// `checked_add` branch for a state that cannot occur.
    #[test]
    fn oversized_request_saturates_rather_than_wrapping() {
        assert!(
            !fits_within(u64::MAX, 1, 100),
            "saturation must not wrap a colossal request into a fit"
        );
        assert!(!fits_within(u64::MAX - 1, 10, 1000));
        assert!(!fits_within(1, u64::MAX, 1000), "headroom overflows too");
    }

    /// The platform wiring: that `fits_in_available_memory` agrees with
    /// `available_memory_bytes` about whether this platform can answer at all.
    ///
    /// Deliberately asserts only on the `Some`/`None` shape and on the one
    /// comparison that cannot move — `0` fits in any figure, however the live
    /// sample drifts. The value-level behaviour is covered above.
    ///
    /// In particular it does **not** assert `avail > 0`. `MemAvailable` is the
    /// kernel's estimate and is clamped at zero, so `Some(0)` is a legitimate
    /// reading under real memory pressure or a tight cgroup limit — asserting
    /// otherwise would reintroduce exactly the kind of environment-dependent
    /// failure this test was rewritten to remove. A zero reading still means the
    /// platform *can* answer, which is all this test is about.
    #[test]
    fn public_fits_matches_platform_support() {
        match available_memory_bytes() {
            Some(_) => {
                // Zero fits in any figure, including zero.
                assert_eq!(fits_in_available_memory(0, 0), Some(true));
            }
            None => {
                // Platform without support (e.g. macOS): fits is likewise None.
                assert_eq!(fits_in_available_memory(0, 0), None);
            }
        }
    }
}

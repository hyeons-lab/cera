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

/// Whether `required_bytes` fits in currently-available physical memory leaving
/// at least `headroom_bytes` free. `None` when available memory can't be queried
/// (see [`available_memory_bytes`]) — the caller then decides whether to proceed.
///
/// `headroom_bytes` is the caller's safety margin for fragmentation, other live
/// allocations, and the OS low-memory killer; there is no baked-in policy here.
pub fn fits_in_available_memory(required_bytes: u64, headroom_bytes: u64) -> Option<bool> {
    let available = available_memory_bytes()?;
    Some(required_bytes.saturating_add(headroom_bytes) <= available)
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

    #[test]
    fn fits_contract_holds_when_available_known() {
        match available_memory_bytes() {
            Some(avail) => {
                assert!(avail > 0);
                // Zero always fits; more-than-available never does.
                assert_eq!(fits_in_available_memory(0, 0), Some(true));
                assert_eq!(
                    fits_in_available_memory(avail.saturating_add(1), 0),
                    Some(false)
                );
            }
            None => {
                // Platform without support (e.g. macOS): fits is likewise None.
                assert_eq!(fits_in_available_memory(0, 0), None);
            }
        }
    }
}

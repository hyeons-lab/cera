//! Decode worker-count selection.
//!
//! Decode runs on all detected performance cores (capped). On heterogeneous
//! big.LITTLE mobile this is the measured optimum — decode scales cleanly
//! across every big core (e.g. Tensor G5: 44.7 → 76.1 tok/s at 1 → 6 threads on
//! LFM2-350M Q4_0), and the [`RowPool`](super::threadpool::RowPool) already pins
//! decode workers to exactly those cores.
//!
//! A per-device throughput sweep was prototyped (a DRAM-bandwidth stream, then a
//! real Q4_0-GEMV probe) but abandoned: a *synthetic* probe can't reproduce the
//! real decode graph's multi-core scaling. Decode is ~113 small GEMVs per token
//! interleaved with norms/attention/softmax — a compute-heavier mix that scales
//! to all P-cores — whereas any single-matrix probe is pure weight-streaming
//! that saturates the memory bus at ~3–4 cores and so under-provisions. Faithful
//! calibration needs to measure the actual decode loop (replay or live tok/s),
//! which is deferred to the adaptive-backend work. `CERA_DECODE_THREADS=<n>`
//! overrides with a fixed count.

use crate::backend::cpu_features::CoreTopology;

/// Upper bound on the auto-selected decode width. On a heterogeneous topology
/// `perf_core_count` is already the (capped) big-core count, so this only binds
/// on the homogeneous fallback where `perf_core_count` is *all* logical CPUs: a
/// many-core host must not spin-wait a barrier across every core for
/// memory-bound decode. 12 covers every single-die Apple Silicon P-core count
/// (M4 Max = 12; decode measurably scales across all P-cores there); dual-die
/// Ultra parts have more, but decode scaling across the die interconnect is
/// unproven — `CERA_DECODE_THREADS` overrides for anyone tuning such a machine.
const DECODE_MAX_AUTO: usize = 12;

/// Resolve the decode worker count for this device: all detected performance
/// cores, capped by [`DECODE_MAX_AUTO`]. `CERA_DECODE_THREADS=<n>` pins a fixed
/// count (clamped to the detected performance-core count); `=auto` (or any
/// unparseable value) falls back to the default.
pub fn decode_thread_count(topo: &CoreTopology) -> usize {
    let max_t = topo.perf_core_count.max(1);

    if let Ok(v) = std::env::var("CERA_DECODE_THREADS") {
        let v = v.trim();
        if !v.eq_ignore_ascii_case("auto") {
            if let Ok(n) = v.parse::<usize>() {
                if n >= 1 {
                    if n > max_t {
                        tracing::warn!(
                            "cera: CERA_DECODE_THREADS={n} exceeds the {max_t} detected \
                             performance cores; clamping to {max_t} (set CERA_THREADS to \
                             raise the detected count)"
                        );
                    }
                    return n.min(max_t);
                }
            }
        }
    }

    max_t.min(DECODE_MAX_AUTO)
}

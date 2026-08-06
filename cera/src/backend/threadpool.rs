//! Persistent, affinity-pinned, spin-wait thread pool for the row-parallel
//! decode hot path.
//!
//! ## Why this exists (vs. rayon)
//!
//! Single-token decode issues tens to hundreds of GEMVs (one per projection per
//! layer that clears the parallel threshold, plus the vocab output) — 25 to 257
//! across the models measured in `super::calibrate`, which is why the decode
//! pool's width is derived from the loaded model rather than a constant. Above
//! [`super::cpu::gemv_par_threshold`] each parallelizes through
//! [`super::cpu::par_rows`] / [`super::cpu::par_rows_n`]. With rayon that
//! is a `par_chunks_mut().for_each()` — a **fork-join with a park/unpark barrier
//! per GEMV**. On Android big.LITTLE the per-dispatch cost (futex wake + core
//! migration + scheduler scatter) dwarfs the tiny per-GEMV compute, so more
//! threads run *slower* (measured Tensor G5, LFM2-350M Q4_0: 1 thread 49 tok/s,
//! 4 threads 8). llama.cpp/ggml avoid this with a persistent pool whose workers
//! stay hot on a spin-wait barrier, so dispatching a GEMV costs an atomic store,
//! not a thread wake. This module is that pool, localized to the two row-parallel
//! entry points.
//!
//! ## Protocol
//!
//! `N` worker threads, where the **calling thread is worker 0** and runs inline;
//! the pool spawns `N-1` background workers. `N` and the cores to pin to come
//! from [`super::cpu_features::core_topology`].
//!
//! Per dispatch, the caller writes a `Job` (output base pointer, chunk size,
//! type-erased closure + monomorphized trampoline) and bumps `state` (Release) —
//! an atomic whose high bits are the **epoch** (workers wake when it changes) —
//! then joins the steal loop as worker 0. Background workers observe the new
//! epoch (Acquire) and, alongside the caller, **claim contiguous row chunks**
//! from a shared `next_chunk` atomic until the row space is exhausted — dynamic
//! work-stealing, so faster cores grab more chunks and every worker reaches the
//! barrier together (heterogeneous big.LITTLE load balancing). Each participating
//! worker then `fetch_sub`s a `pending` counter; the caller spins on `pending ==
//! 0` (Acquire) before returning. That Release/Acquire pair both (a) publishes
//! the `Job` to workers and (b) establishes happens-before for every worker's
//! writes to the output, so the disjoint `&mut` handoff is sound and the caller
//! may read the output once it returns. (Each chunk index is claimed by exactly
//! one worker, so chunks are disjoint row ranges.)
//!
//! **Active-sized barrier (prefill pool).** Only the `active` workers a dispatch
//! needs (worker 0 = caller included) run and decrement `pending`; the rest read
//! the same packed `state`, see `id >= active`, and skip without touching `job`
//! or `pending`. So the barrier's shared-line `fetch_sub` storm — which bounces
//! across the CCD fabric on a multi-die host — costs O(active), not O(pool): a
//! small GEMM no longer drags cores it can't fill through the barrier, and idle
//! workers are left to park (spinning them back up re-reads the cross-die
//! `state` and measured the win back away). Packing `active` *with* the epoch is
//! what makes this race-free: an idle worker can't read an `active` from a newer
//! epoch than the one it observed, so it never runs a `job` the dispatcher has
//! already retired. `active` is chosen from the row count and, for prefill GEMMs,
//! an arithmetic-work cap (see `GEMM_WORK_PER_WORKER`).
//!
//! The **decode pool** instead uses a *full* barrier (`Shared::active_barrier ==
//! false`): every worker decrements and is kept unparked. Decode alternates many
//! tiny GEMVs (few active) with the huge vocab GEMV (all active), and keeping the
//! whole pool hot beats parking/re-waking it — the active-sized barrier measured
//! −15% there. Decode is memory-bound and narrow, so it never pays the multi-die
//! barrier tax the prefill GEMMs do.
//!
//! Between GEMVs (µs apart) workers spin. Between tokens (ms idle) a bounded spin
//! falls back to [`std::thread::park`]; the caller `unpark`s on the next
//! dispatch. So the hot path never pays a wake, and idle workers don't burn power.
//!
//! ## Determinism
//!
//! Each output row is computed by exactly one worker, at the same absolute row
//! index it would have serially — no float reassociation — so greedy output is
//! bit-for-bit identical to the serial and rayon paths.
//!
//! ## Concurrent and nested dispatch
//!
//! One dispatch owns the pool at a time: [`RowPool::dispatch_rows`] takes an
//! internal dispatch lock. A second thread (or a closure re-entering the same
//! pool) that finds the lock held simply runs its rows serially on its own
//! thread — always correct, never deadlocks, and the contended case is the
//! rare one (cera's decode/prefill loops are single-threaded per session).
//!
//! ## Panics
//!
//! A panic in the row closure is contained: workers catch it, the dispatcher
//! drains the pool (so no pointer outlives the dispatch), and the panic is
//! re-raised on the calling thread — same contract as rayon. The pool stays
//! usable afterwards.
//!
//! ## Affinity side effects
//!
//! On Linux/Android the spawned workers *and the calling thread* are pinned
//! to distinct detected cores, fastest-first, so a default-width pool lands
//! entirely on the performance ones (the caller to the fastest one, because
//! unpinned a big.LITTLE scheduler can strand it on an efficiency core where
//! it stalls every barrier). The caller pin is held by one thread at a time,
//! process-wide, for as long as that thread lives: concurrent additional
//! dispatchers (a second session) keep floating rather than piling onto the
//! same core, and when the holding thread exits the claim frees for the next
//! dispatcher. `CERA_PIN=0` disables all affinity pinning for hosts that
//! manage placement themselves.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

/// Bounded spin iterations before a waiting worker parks. Sized to comfortably
/// cover the between-GEMV gap (the caller's serial work between matmuls, ~µs)
/// while still parking during the longer between-token gap (~ms), so idle
/// workers don't spin the battery flat. Heuristic; the on-device power/throughput
/// benchmark is the real validation.
#[cfg(not(miri))]
const SPIN_BEFORE_PARK: u32 = 100_000;
/// Miri executes far too slowly to spin 100k times on idle workers; a tiny
/// bound keeps the interpreter's data-race checking tractable while still
/// exercising both the spin and park branches.
#[cfg(miri)]
const SPIN_BEFORE_PARK: u32 = 4;

/// Spin iterations the dispatcher burns waiting for the barrier before it
/// starts yielding its timeslice. Workers finish within µs when running; the
/// yield fallback keeps a preempted/descheduled worker (backgrounded app,
/// restricted cgroup) from turning the caller's wait into an unbounded
/// 100%-CPU spin that starves the very straggler it waits on.
#[cfg(not(miri))]
const DRAIN_SPIN_BEFORE_YIELD: u32 = 10_000;
#[cfg(miri)]
const DRAIN_SPIN_BEFORE_YIELD: u32 = 4;

/// Dispatches between caller-pin affinity syscalls: the pacing both for
/// re-attempting a REFUSED pin (restricted cpuset) and for RE-ASSERTING a
/// held one (Android cpuset cgroup migrations overwrite per-thread masks on
/// background ↔ foreground transitions, silently revoking pins). ~10 decoded
/// tokens at ~100 GEMVs/token — quick to recover after foregrounding, sparse
/// enough that the syscall never shows up in a profile.
const PIN_RETRY_BACKOFF: u32 = 1024;

/// A single row-parallel job. All pointers are valid only for the duration of
/// one dispatch — the dispatcher blocks until every participating worker
/// finishes — so nothing here outlives the borrowed slice or closure.
///
/// `Copy` so a worker snapshots it out of the shared cell instead of holding a
/// borrow across execution.
#[derive(Clone, Copy)]
struct Job {
    /// Base pointer to the output slice `y`.
    y_ptr: *mut f32,
    /// Elements per row (1 for `par_rows`, `n` for `par_rows_n`).
    n: usize,
    /// Rows per steal unit. Workers claim chunks of this many contiguous rows
    /// from the shared `next_chunk` counter — dynamic work-stealing, so fast
    /// cores grab more chunks and all workers reach the barrier together
    /// (heterogeneous big.LITTLE load balancing).
    chunk_rows: usize,
    /// Total rows = `y.len() / n`.
    total_rows: usize,
    /// Workers participating this dispatch (`≤` pool size), worker 0 included.
    /// Read by the *full*-barrier (decode) worker path to decide whether it runs,
    /// keeping that path byte-identical to the pre-active-barrier pool. The
    /// active-barrier (prefill) path instead reads `active` from the packed
    /// `state` (so an idle worker can skip without touching `job`).
    active: usize,
    /// Type-erased `&F` — the per-row closure, borrowed for the dispatch.
    closure: *const (),
    /// Monomorphized trampoline: runs `closure` over rows `[start, end)`.
    run: unsafe fn(closure: *const (), y_ptr: *mut f32, n: usize, start: usize, end: usize),
}

/// Runs the erased closure over a worker's contiguous row range.
///
/// # Safety
/// - `closure` must point to a live `&F` for the whole call.
/// - `[start, end)` must be within `[0, total_rows)` and **disjoint** from every
///   other worker's range this dispatch — each row is written exactly once, so
///   the reconstructed `&mut [f32]` never aliases another thread's.
unsafe fn trampoline<F: Fn(usize, &mut [f32]) + Sync>(
    closure: *const (),
    y_ptr: *mut f32,
    n: usize,
    start: usize,
    end: usize,
) {
    let f = unsafe { &*(closure as *const F) };
    for row in start..end {
        // SAFETY: `row < total_rows` ⇒ `row * n + n <= y.len()`; disjoint ranges
        // guarantee no other thread holds an overlapping `&mut`.
        let slice = unsafe { std::slice::from_raw_parts_mut(y_ptr.add(row * n), n) };
        f(row, slice);
    }
}

/// Shared state between the dispatcher and the background workers.
struct Shared {
    /// Active-sized barrier (prefill pool) vs full barrier (decode pool). When
    /// `true`, only the `active` workers a dispatch needs decrement `pending`
    /// (`pending == active - 1`) and idle workers skip untouched — the O(active)
    /// barrier that unblocks small prefill GEMMs. When `false`, *every* worker
    /// decrements (`pending == num_threads - 1`), keeping the whole pool engaged
    /// each dispatch: decode alternates tiny GEMVs with the huge vocab GEMV and
    /// wants all workers hot (an active-sized barrier there measured −15% decode,
    /// as parking then re-waking workers costs more than it saves). Immutable
    /// after `build`.
    active_barrier: bool,
    /// Packed `(epoch, active)` — see [`pack_state`]. Bumped once per dispatch;
    /// workers wake when the epoch changes and read `active` from the same load.
    state: AtomicU64,
    /// Next unclaimed chunk index. Active workers (and the caller) claim chunks
    /// via `fetch_add`; reset to 0 before each dispatch's `state` bump.
    next_chunk: AtomicUsize,
    /// Background workers the dispatcher still waits on, reaching 0 when the
    /// dispatch has drained. Under the active-sized barrier this is `active - 1`
    /// (only participating workers decrement); under the full barrier it is
    /// `num_threads - 1` (every background worker decrements).
    pending: AtomicUsize,
    /// Set on drop to release the workers.
    shutdown: AtomicBool,
    /// Set by a worker whose closure panicked; the dispatcher re-raises the
    /// panic on the calling thread after the barrier drains.
    panicked: AtomicBool,
    /// The panicking worker's original payload, resumed on the calling thread
    /// so the panic message/type survive the thread hop (rayon's contract).
    /// Only touched on the panic path — never on a normal dispatch.
    panic_payload: Mutex<Option<Box<dyn std::any::Any + Send>>>,
    /// The current job. Written before the `epoch` bump, read after observing it.
    job: UnsafeCell<Option<Job>>,
}

/// Target chunks per active worker — enough finer-than-worker granularity that
/// a faster core can steal extra chunks to cover for a slower one (the X4 vs
/// A725 ~1.24× speed gap needs a handful of chunks each to balance).
const STEAL_CHUNKS_PER_WORKER: usize = 4;
/// Floor on chunk size: below this, per-chunk atomic + kernel-setup overhead
/// starts to matter and contiguous streaming gets choppy.
const MIN_CHUNK_ROWS: usize = 16;

/// Default MAC count assigned to one prefill-GEMM worker before another is
/// added (the [`RowPool::dispatch_rows_work`] cap). A small GEMM — a tiny model
/// and/or a short prompt — has little total arithmetic to share; spreading it
/// across every core makes them contend on the dispatch barrier and the memory
/// bus for no compute payoff. Because the barrier's `fetch_sub` now runs only
/// across the `active` workers (the packed-`active` protocol, see
/// [`pack_state`]), capping `active` at `total_macs / GEMM_WORK_PER_WORKER`
/// keeps that shared-line storm off the cores the GEMM can't fill — decisive on
/// a multi-CCD host, where an unneeded worker's `fetch_sub` bounces the barrier
/// line across the inter-die fabric.
///
/// **There is no globally correct value; do not "fix" this without data.**
/// Measured on a Ryzen AI MAX+ 395 (16 Zen 5 cores = 2×8-core CCDs, VNNI),
/// prefill tok/s of this build vs origin, interleaved 2-binary A/B:
///
/// | model               | pp64  | pp512 |
/// |---------------------|------:|------:|
/// | SmolLM-135M Q4_0     |  +93% |  +38% |
/// | LFM2.5-230M Q4_K_M   |  +24% |  +15% |
/// | Llama-3.2-1B Q8_0    |  +15% |   +4% |
///
/// The win is largest where the arithmetic is smallest (tiny model / short
/// prompt); the big model, whose GEMMs stay above the cap, still gains (its
/// narrow k/v projections and short-prompt GEMMs cap down). A larger quantum
/// (≥ ~96M here) over-caps and *regresses* short prompts; this value is the
/// conservative end of the safe range, so it caps only genuinely small GEMMs and
/// stays correct-or-harmless on the single-CCD / high-bandwidth hosts it was not
/// measured on.
/// `CERA_GEMM_WORK_PER_WORKER` overrides it for per-device tuning, matching the
/// `CERA_MIN_ROWS` / `CERA_PREQUANT_MIN_COLS` knob style.
const GEMM_WORK_PER_WORKER_DEFAULT: usize = 48_000_000;

/// Resolved [`GEMM_WORK_PER_WORKER_DEFAULT`], read once. Uses the same
/// [`env_usize`](super::cpu_features::env_usize) parser as `CERA_MIN_ROWS` /
/// `CERA_PREQUANT_MIN_COLS` (trimmed, `>= 1`), so a `0`, whitespace-padded, or
/// unparseable override falls back to the default — the knob can only retune the
/// threshold, never zero it (a `0` quantum would divide-by-zero the cap).
fn gemm_work_per_worker() -> usize {
    static Q: OnceLock<usize> = OnceLock::new();
    *Q.get_or_init(|| {
        super::cpu_features::env_usize("CERA_GEMM_WORK_PER_WORKER")
            .unwrap_or(GEMM_WORK_PER_WORKER_DEFAULT)
    })
}

/// The pool's dispatch state, packed into one atomic so a worker learns both
/// facts from a single load: the **epoch** (high bits, bumped once per dispatch
/// — workers wake when it changes) and the **active** worker count (low
/// [`ACTIVE_BITS`]; how many workers, worker 0 = caller included, run this
/// dispatch). Packing them is what lets an *idle* worker (`id >= active`) skip a
/// dispatch without reading the overwritable `job` or touching `pending`: it can
/// never read an `active` that belongs to a different epoch than the one it just
/// observed, so there is no torn `(epoch, active)` pair and no double-run. The
/// barrier then waits on only `active - 1` background workers, so a small GEMM
/// never wakes the far CCD — whose cross-fabric `pending`/steal traffic is the
/// tax that made a 16-wide prefill dispatch slower than an 8-wide one.
const ACTIVE_BITS: u64 = 16;
const ACTIVE_MASK: u64 = (1 << ACTIVE_BITS) - 1;

#[inline]
fn pack_state(epoch: u64, active: usize) -> u64 {
    debug_assert!(
        active as u64 <= ACTIVE_MASK,
        "active {active} exceeds ACTIVE_MASK ({ACTIVE_MASK})"
    );
    (epoch << ACTIVE_BITS) | (active as u64 & ACTIVE_MASK)
}
#[inline]
fn state_epoch(state: u64) -> u64 {
    state >> ACTIVE_BITS
}
#[inline]
fn state_active(state: u64) -> usize {
    (state & ACTIVE_MASK) as usize
}

/// Claim and run chunks from `shared.next_chunk` until the row space is
/// exhausted. Shared by the caller (worker 0) and every active background
/// worker — each `fetch_add` hands out a unique, disjoint contiguous row range.
#[inline]
fn steal_and_run(shared: &Shared, job: &Job) {
    loop {
        // Relaxed: uniqueness/atomicity of the claim is all we need here; the
        // visibility of each worker's output writes to the caller is provided
        // by the `pending` Release/Acquire barrier at the end of the dispatch.
        let chunk = shared.next_chunk.fetch_add(1, Ordering::Relaxed);
        let start = chunk * job.chunk_rows;
        if start >= job.total_rows {
            break;
        }
        let end = (start + job.chunk_rows).min(job.total_rows);
        // SAFETY: each chunk index is claimed by exactly one worker, so the row
        // range `[start, end)` is disjoint from every other worker's — no two
        // reconstructed `&mut` slices overlap (see `trampoline`).
        unsafe { (job.run)(job.closure, job.y_ptr, job.n, start, end) };
    }
}

// SAFETY: `job`'s raw pointers are only dereferenced by a worker after it
// observes an `epoch` change (Acquire) that the dispatcher published (Release)
// *after* writing a fresh `Job`; the dispatcher then blocks until `pending == 0`,
// so the borrowed slice/closure outlive every access. Access to the `UnsafeCell`
// is disciplined entirely by the `epoch`/`pending` atomics, so `Shared` is safe
// to both share (`Sync`) and move into the worker threads (`Send`) via `Arc`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// Persistent spin-wait worker pool. See the module docs.
pub struct RowPool {
    shared: Arc<Shared>,
    /// Join handles — `handle.thread()` for `unpark` on dispatch (index `i` ⇒
    /// worker id `i + 1`), drained/joined on drop.
    workers: Vec<JoinHandle<()>>,
    /// Serializes dispatches: exactly one thread drives the pool at a time. A
    /// contender (second thread, or a closure re-entering the pool) runs its
    /// rows serially instead of blocking — see [`RowPool::dispatch_rows`].
    dispatch_lock: Mutex<()>,
    /// Core to pin the calling thread (worker 0) to on its first dispatch, i.e.
    /// the fastest detected core. `None` ⇒ no caller pinning (macOS/desktop).
    caller_pin: Option<usize>,
    /// Total workers including the caller (worker 0); always `≥ 1`.
    num_threads: usize,
}

/// Counters for the dispatch fan-out path.
///
/// Exists for one question: when a dispatch wants `active > 1` workers but
/// `dispatch_lock.try_lock()` fails, it silently runs the whole operation
/// serially on the caller. That is an all-or-nothing slowdown on that
/// operation, and it does not show up anywhere except as a throughput number
/// that is mysteriously half what it should be. Counting it is the only way to
/// tell "this dispatch fanned out and the cores were slow" from "this dispatch
/// never fanned out at all".
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static FANOUT_DISPATCHES: AtomicU64 = AtomicU64::new(0);
    pub(super) static SERIAL_FALLBACKS: AtomicU64 = AtomicU64::new(0);
    pub(super) static FANOUT_MACS: AtomicU64 = AtomicU64::new(0);
    pub(super) static SERIAL_MACS: AtomicU64 = AtomicU64::new(0);

    /// Dispatch-path counters since process start. Monotonic; use
    /// [`PoolStats::since`] to get the delta over an interval.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct PoolStats {
        /// Dispatches that wanted more than one worker.
        pub fanout_dispatches: u64,
        /// Of those, how many ran serially because the pool was already busy.
        pub serial_fallbacks: u64,
        /// Work over all `fanout_dispatches`, in output elements times the
        /// caller-supplied `depth`.
        ///
        /// **Only the work-capped GEMM dispatches supply a depth**
        /// ([`super::super::cpu::par_rows_n_work`]); every other caller passes
        /// zero and so contributes plain output elements. The unit is therefore
        /// mixed, and a serialized decode GEMV is understated against a
        /// serialized prefill GEMM by roughly its contraction length. Treat this
        /// as a rough weighting *within* one dispatch kind, and treat
        /// `serial_fallbacks` / `fanout_dispatches` as the reliable signal:
        /// those are exact, and detecting the silent fallback at all is what
        /// these counters are for.
        pub fanout_macs: u64,
        /// The same measure over just the `serial_fallbacks`, so one serialized
        /// large GEMM outweighs many serialized small ones. Same mixed-unit
        /// caveat as `fanout_macs`.
        pub serial_macs: u64,
    }

    impl PoolStats {
        /// Fraction of fan-out work that ran serially, 0.0 to 1.0. Carries the
        /// mixed-unit caveat documented on [`PoolStats::fanout_macs`]; the
        /// dispatch counts are the trustworthy number.
        pub fn serial_mac_fraction(&self) -> f64 {
            if self.fanout_macs == 0 {
                return 0.0;
            }
            self.serial_macs as f64 / self.fanout_macs as f64
        }

        /// Counters accumulated between two snapshots.
        pub fn since(&self, earlier: &PoolStats) -> PoolStats {
            PoolStats {
                fanout_dispatches: self
                    .fanout_dispatches
                    .saturating_sub(earlier.fanout_dispatches),
                serial_fallbacks: self
                    .serial_fallbacks
                    .saturating_sub(earlier.serial_fallbacks),
                fanout_macs: self.fanout_macs.saturating_sub(earlier.fanout_macs),
                serial_macs: self.serial_macs.saturating_sub(earlier.serial_macs),
            }
        }
    }

    /// Read the counters.
    pub fn snapshot() -> PoolStats {
        PoolStats {
            fanout_dispatches: FANOUT_DISPATCHES.load(Ordering::Relaxed),
            serial_fallbacks: SERIAL_FALLBACKS.load(Ordering::Relaxed),
            fanout_macs: FANOUT_MACS.load(Ordering::Relaxed),
            serial_macs: SERIAL_MACS.load(Ordering::Relaxed),
        }
    }
}

/// Whether affinity pinning is enabled (`CERA_PIN=0`/`false`/`off`,
/// case-insensitive, disables it — for host apps that manage thread placement
/// and don't want cera's permanent caller pin). Resolved once.
pub(crate) fn pinning_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !super::cpu_features::env_disabled("CERA_PIN"))
}

impl RowPool {
    /// Full-width pool for compute-bound batched work — prefill GEMM
    /// ([`super::cpu::par_rows_n`]). Width comes from
    /// `super::calibrate::prefill_thread_count`: the performance cores, since
    /// batched matmul is compute-bound and scales with threads, but *not* the
    /// efficiency cores, which measured as a large net loss. Overridable with
    /// `CERA_PREFILL_THREADS=<n>` up to every pinnable core. Lazily built so
    /// `CeraEngine` consumers get it without calling
    /// [`super::cpu::configure_thread_pool`].
    pub fn prefill() -> &'static RowPool {
        static POOL: OnceLock<RowPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let topo = super::cpu_features::core_topology();
            let n = super::calibrate::prefill_thread_count(topo);
            // Active-sized barrier: prefill GEMMs vary widely in size, and a
            // small one is better run on the few cores its work can fill than
            // dragged across the whole pool's barrier (see `Shared::active_barrier`).
            RowPool::build(n, pinned_cores(), true)
        })
    }

    /// Narrow pool for per-token work — decode GEMV
    /// ([`super::cpu::par_rows`]). Width comes from
    /// `super::calibrate::decode_thread_count`, which sizes it from the loaded
    /// model's `DecodeShape` (bytes per pool dispatch) on a homogeneous host,
    /// and on heterogeneous big.LITTLE keeps the full big-core set — decode's
    /// measured optimum there. Overridable with `CERA_DECODE_THREADS=<n>`.
    ///
    /// **Whoever touches this `OnceLock` first freezes the width for the
    /// process.** That is why `super::cpu::configure_thread_pool` deliberately
    /// does not warm it: it runs before any model is loaded, so warming it
    /// there would pin the pool to the model-less fallback and silently disable
    /// shape-based sizing. Keep it lazy.
    pub fn decode() -> &'static RowPool {
        static POOL: OnceLock<RowPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let topo = super::cpu_features::core_topology();
            let n = super::calibrate::decode_thread_count(topo);
            // Full barrier: decode wants every worker hot across its tiny-GEMV /
            // huge-vocab-GEMV mix (see `Shared::active_barrier`).
            RowPool::build(n, pinned_cores(), false)
        })
    }

    /// Build a pool with `num_threads` total workers, pinning worker `i` to
    /// `pin_cores[i]` when present (surplus workers run unpinned). `pin_cores`
    /// empty ⇒ no pinning (macOS/desktop). Spawn failures degrade the thread
    /// count rather than panicking. The pool pins *and claims the process-wide
    /// caller pin for* whatever thread first dispatches.
    ///
    /// `pin_cores` is taken as already policy-filtered: callers outside the
    /// tests pass [`pinned_cores`], which is where `CERA_PIN` is honoured.
    fn build(num_threads: usize, pin_cores: &[usize], active_barrier: bool) -> RowPool {
        // `active` (≤ num_threads) is packed into the low `ACTIVE_BITS` of the
        // dispatch state; no real host has this many cores, but keep the pool
        // within the field rather than silently corrupt the epoch above it.
        let num_threads = num_threads.max(1).min(ACTIVE_MASK as usize);
        // Spin iterations before an idle worker parks. `CERA_SPIN` overrides the
        // default for tuning the spin-vs-park trade-off on a given device.
        let spin_limit = std::env::var("CERA_SPIN")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(SPIN_BEFORE_PARK);
        let shared = Arc::new(Shared {
            active_barrier,
            state: AtomicU64::new(0),
            next_chunk: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
            job: UnsafeCell::new(None),
        });

        let mut workers = Vec::new();
        // Worker 0 is the caller; spawn the rest.
        for id in 1..num_threads {
            let shared = Arc::clone(&shared);
            let pin = pin_cores.get(id).copied();
            match thread::Builder::new()
                .name(format!("cera-rowpool-{id}"))
                .spawn(move || worker_loop(shared, id, pin, spin_limit))
            {
                Ok(handle) => workers.push(handle),
                // Couldn't spawn — cap the pool at what we have.
                Err(_) => break,
            }
        }

        // Actual size may be less than requested if spawns failed.
        let num_threads = 1 + workers.len();
        // Worker 0 (the caller) pins to the fastest detected core. Without this
        // the caller floats — on Android big.LITTLE it can land on an efficiency
        // core and, as the barrier's straggler, stall every GEMV (measured: 4
        // threads no faster than 1 until the caller is confined to a perf core).
        let caller_pin = pin_cores.first().copied();
        RowPool {
            shared,
            workers,
            dispatch_lock: Mutex::new(()),
            caller_pin,
            num_threads,
        }
    }

    /// Total worker count (caller + spawned background workers).
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Pin the calling thread (worker 0) to the pool's fastest core — held by
    /// at most one caller thread at a time, process-wide. Without the claim,
    /// every host thread that ever dispatches would be permanently pinned to
    /// the *same* core (both pools share `pin_cores[0]`), so two concurrent
    /// sessions would timeshare one core for all their serial work; a
    /// concurrent second caller just stays floating instead. The claim is
    /// RELEASED when the holding thread exits (thread-local guard `Drop`) and
    /// unclaimed callers retry on later dispatches — a host running inference
    /// from a recycled thread pool (e.g. tokio `spawn_blocking`) would
    /// otherwise lose the caller pin forever the first time a claiming thread
    /// got reaped. A REFUSED pin (restricted cpuset, offline core) backs off
    /// for [`PIN_RETRY_BACKOFF`] dispatches before retrying, and a HELD pin
    /// is re-asserted on the same cadence — cpuset cgroup migrations
    /// (Android background ↔ foreground) overwrite per-thread masks, silently
    /// revoking pins. Steady-state cost: one thread-local read per dispatch
    /// (plus one relaxed load while another thread holds the claim). No-op
    /// when the platform has no affinity (`caller_pin == None`).
    fn pin_caller_once(&self) {
        static CALLER_PIN_CLAIMED: AtomicBool = AtomicBool::new(false);
        /// Releases the claim when the holding thread exits.
        struct ClaimGuard;
        impl Drop for ClaimGuard {
            fn drop(&mut self) {
                CALLER_PIN_CLAIMED.store(false, Ordering::Release);
            }
        }
        struct CallerClaim {
            guard: Option<ClaimGuard>,
            /// Dispatches to skip before re-attempting a refused pin.
            retry_cooldown: u32,
        }
        thread_local! {
            static CLAIM: std::cell::RefCell<CallerClaim> = const {
                std::cell::RefCell::new(CallerClaim {
                    guard: None,
                    retry_cooldown: 0,
                })
            };
        }
        let Some(core) = self.caller_pin else {
            return;
        };
        CLAIM.with(|c| {
            let mut claim = c.borrow_mut();
            if claim.retry_cooldown > 0 {
                claim.retry_cooldown -= 1;
                return;
            }
            if claim.guard.is_some() {
                // Periodically RE-ASSERT the held pin: Android cpuset cgroup
                // migrations (background ↔ foreground) overwrite per-thread
                // affinity masks wholesale, silently revoking it. One ~µs
                // syscall per PIN_RETRY_BACKOFF dispatches (~10 tokens).
                let _ = pin_current_thread_to_core(core);
                claim.retry_cooldown = PIN_RETRY_BACKOFF;
                return;
            }
            if !CALLER_PIN_CLAIMED.load(Ordering::Relaxed)
                && CALLER_PIN_CLAIMED
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                if pin_current_thread_to_core(core) {
                    claim.guard = Some(ClaimGuard);
                    claim.retry_cooldown = PIN_RETRY_BACKOFF;
                } else {
                    // Pin refused: release so another (or this) thread can
                    // claim later, and back off before the next attempt.
                    CALLER_PIN_CLAIMED.store(false, Ordering::Release);
                    claim.retry_cooldown = PIN_RETRY_BACKOFF;
                }
            }
        });
    }

    /// Run `f` over each of the `y.len() / n` rows of `y`, in parallel across the
    /// pool. `f` receives `(row_index, &mut row_slice_of_len_n)`. Rows are handed
    /// out in contiguous chunks via dynamic work-stealing, so faster cores cover
    /// more of the range and every worker reaches the barrier together.
    /// `min_rows` gates how many workers participate (small ops stay serial).
    /// A trailing partial row (`y.len() % n != 0`) is run on the caller after
    /// the full rows, matching the serial `chunks_mut(n)` semantics.
    ///
    /// `n == 1` gives the element-wise `par_rows` shape; `n > 1` the
    /// `par_rows_n` (row-of-`n`) shape.
    ///
    /// Safe under concurrent callers: one dispatch owns the pool at a time and
    /// a contender (or a closure re-entering the same pool) runs serially. A
    /// panicking closure is drained and re-raised on the calling thread.
    pub fn dispatch_rows<F>(&self, y: &mut [f32], n: usize, min_rows: usize, f: F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        self.dispatch_rows_chunked(y, n, min_rows, MIN_CHUNK_ROWS, f);
    }

    /// Like [`RowPool::dispatch_rows`], but with an explicit steal-chunk floor.
    ///
    /// `dispatch_rows` floors the steal chunk at `MIN_CHUNK_ROWS` (tuned for many
    /// *cheap* rows, e.g. a GEMV's output rows, where sub-`MIN_CHUNK_ROWS` chunks
    /// would spend more on steal bookkeeping than on the row). That floor
    /// *under-parallelizes* the opposite shape — few rows, each expensive: flash
    /// attention hands one whole *head* per row, so 32 heads at a 16-row floor
    /// collapse to 2 steal chunks = 2 busy workers, 14 idle. Such callers pass
    /// `min_chunk_rows = 1` so every heavy row is its own steal unit and all
    /// `active` workers participate.
    pub fn dispatch_rows_chunked<F>(
        &self,
        y: &mut [f32],
        n: usize,
        min_rows: usize,
        min_chunk_rows: usize,
        f: F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        self.dispatch_inner(y, n, min_rows, min_chunk_rows, 0, f);
    }

    /// Like [`RowPool::dispatch_rows`], but caps the active worker count by the
    /// dispatch's total arithmetic so a small GEMM doesn't fork wider than its
    /// work can fill. `depth` is the contraction length `k`; total work is
    /// `y.len() * depth` MACs, capped at `total_macs / GEMM_WORK_PER_WORKER`
    /// workers. Under the active-sized barrier only those workers take part, so
    /// trimming the count for a narrow GEMM directly sheds the barrier +
    /// memory-bus contention its work can't amortize — on a tiny model (or a
    /// short prompt) a large prefill win. Uses the default steal-chunk floor.
    ///
    /// `depth == 0` disables the work cap (the plain row-count gate), so callers
    /// with no depth notion keep the previous behaviour.
    pub fn dispatch_rows_work<F>(
        &self,
        y: &mut [f32],
        n: usize,
        min_rows: usize,
        depth: usize,
        f: F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        self.dispatch_inner(y, n, min_rows, MIN_CHUNK_ROWS, depth, f);
    }

    /// Shared body of the `dispatch_rows*` family: split off any trailing
    /// partial row (run on the caller, matching serial `chunks_mut(n)`
    /// semantics), then run the exact rows in parallel. `depth` feeds the
    /// work-based active cap (`0` = no cap); `min_chunk_rows` the steal floor.
    fn dispatch_inner<F>(
        &self,
        y: &mut [f32],
        n: usize,
        min_rows: usize,
        min_chunk_rows: usize,
        depth: usize,
        f: F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        debug_assert!(n >= 1, "dispatch_inner: n must be ≥ 1");
        if n == 0 || y.is_empty() {
            return;
        }
        self.pin_caller_once();
        let total_rows = y.len() / n;
        // Split off any trailing partial row now; it runs on the caller after
        // the full rows (the parallel body only handles exact rows).
        let (body, tail) = y.split_at_mut(total_rows * n);
        self.dispatch_body(body, n, total_rows, min_rows, min_chunk_rows, depth, &f);
        if !tail.is_empty() {
            f(total_rows, tail);
        }
    }

    /// The parallel body of [`RowPool::dispatch_rows`]: exactly `total_rows`
    /// full rows of `n` elements (`y.len() == total_rows * n`).
    #[allow(clippy::too_many_arguments)] // internal fan-out knobs, all load-bearing
    fn dispatch_body<F>(
        &self,
        y: &mut [f32],
        n: usize,
        total_rows: usize,
        min_rows: usize,
        min_chunk_rows: usize,
        depth: usize,
        f: &F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        if total_rows == 0 {
            return;
        }
        let min_rows = min_rows.max(1);
        // `active` = how many workers participate, gated by `min_rows` so small
        // ops don't wake the whole pool. Within `active`, work is stolen (below).
        let rows_per_worker = total_rows.div_ceil(self.num_threads).max(min_rows);
        let mut active = total_rows.div_ceil(rows_per_worker).min(self.num_threads);

        // Work cap: a GEMM with little total arithmetic can't keep the whole
        // pool busy, and only the `active` workers chosen here take part in the
        // barrier (`pending == active - 1`), so trimming `active` for a narrow
        // GEMM directly avoids waking — and cross-fabric-taxing — cores it can't
        // fill. Cap `active` so each participating worker gets ≥ one work
        // quantum. `total_rows * n` is the output element count; `* depth` (the
        // contraction length `k`) makes it the MAC count. `depth == 0` disables
        // the cap (element-wise / no-depth callers), preserving their prior
        // width.
        if depth != 0 {
            let total_macs = total_rows.saturating_mul(n).saturating_mul(depth);
            let work_cap = (total_macs / gemm_work_per_worker()).clamp(1, self.num_threads);
            active = active.min(work_cap);
        }

        // One dispatch owns the pool at a time. If another thread is mid-
        // dispatch (or a closure re-entered the pool), fall back to the serial
        // path below rather than blocking — always sound, never deadlocks. A
        // poisoned lock (a dispatcher panicked) is safe to take: the drain
        // guard below leaves the pool state consistent even on unwind.
        let wanted_fanout = active > 1;
        let guard = if wanted_fanout {
            match self.dispatch_lock.try_lock() {
                Ok(g) => Some(g),
                Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
                Err(std::sync::TryLockError::WouldBlock) => None,
            }
        } else {
            None
        };

        // Count how often that fallback fires, and how much work it swallows.
        // A dispatch that wanted `active` workers and got one is a silent ~Nx on
        // that operation, invisible in any throughput number, so it has to be
        // counted rather than reasoned about. Relaxed adds on a path that is
        // about to do at least `total_rows * n` MACs are free.
        //
        // `depth.max(1)` means a caller that supplied no depth contributes
        // output elements rather than MACs; see `stats::PoolStats::fanout_macs`
        // for why that is tolerated and what to read instead.
        if wanted_fanout {
            let macs = (total_rows as u64)
                .saturating_mul(n as u64)
                .saturating_mul(depth.max(1) as u64);
            stats::FANOUT_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            stats::FANOUT_MACS.fetch_add(macs, Ordering::Relaxed);
            if guard.is_none() {
                stats::SERIAL_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                stats::SERIAL_MACS.fetch_add(macs, Ordering::Relaxed);
            }
        }

        // Single active worker (small op, a 1-thread pool, or a contended
        // dispatch): run serially on the caller with safe slicing — no pointer
        // handoff, no wake.
        let Some(_guard) = guard else {
            for row in 0..total_rows {
                f(row, &mut y[row * n..row * n + n]);
            }
            return;
        };

        // Chunk finer than one-range-per-worker so a fast core can steal extra
        // chunks to balance out a slow one. ~`STEAL_CHUNKS_PER_WORKER` chunks per
        // active worker, floored at the caller's `min_chunk_rows`.
        let chunk_rows = total_rows
            .div_ceil(active * STEAL_CHUNKS_PER_WORKER)
            .max(min_chunk_rows.max(1));

        // Publish the job, then join the steal loop as worker 0.
        //
        // Take the output pointer exactly once: this single `as_mut_ptr` tag is
        // shared (as a raw, aliasable pointer) by every worker *and* the caller's
        // steal loop below. Reborrowing `y` again here — a second `as_mut_ptr`, or
        // any `y[..]` access — would invalidate the tag the worker threads still
        // hold (a use-after-invalidate the Miri Stacked-Borrows check catches).
        // So `y` must not be touched again until this dispatch drains.
        let y_ptr = y.as_mut_ptr();
        let closure_ptr = (f as *const F).cast::<()>();
        let job = Job {
            y_ptr,
            n,
            chunk_rows,
            total_rows,
            active,
            closure: closure_ptr,
            run: trampoline::<F>,
        };
        // Consume any stale panic state: if the previous dispatch's caller
        // itself panicked, its worker-panic flag/payload were never consumed
        // (a panic did propagate — the worker one was redundant). Plain load
        // on the hot path; the branch is only taken after such a panic.
        if self.shared.panicked.load(Ordering::Relaxed) {
            drop(self.take_panic_payload());
        }
        // SAFETY: no worker reads `job` until it observes the `state` epoch bump
        // below, which is Released after this write; the previous dispatch
        // already drained (`pending == 0`) before releasing the dispatch lock.
        unsafe {
            *self.shared.job.get() = Some(job);
        }
        // Reset the chunk cursor for this dispatch (ordered before workers can
        // claim by the `state` Release/Acquire below).
        self.shared.next_chunk.store(0, Ordering::Relaxed);
        // Publish the dispatch. Two barrier shapes (see `Shared::active_barrier`):
        //
        // - **Active-sized (prefill):** only the `active` workers (worker 0 =
        //   caller included) take part, so `pending == active - 1` and only they
        //   are unparked. An idle worker (`id >= active`) reads the packed `state`,
        //   sees it's idle, and skips — never touching `job` or `pending`, so its
        //   `fetch_sub` never bounces the barrier line across the CCD fabric. This
        //   is race-free because `active` is packed *with* the epoch (see
        //   [`pack_state`]): an idle worker can't read an `active` from a newer
        //   epoch than the one it observed, so it never acts on a retired `job`.
        //   The dispatcher drains fully before the next epoch bump, so each active
        //   worker observes every epoch it participates in exactly once.
        //
        // - **Full (decode):** byte-identical to the pre-active-barrier pool —
        //   `pending == num_threads - 1`, a plain `state.fetch_add(1)` that just
        //   changes the word so workers wake (this pool never unpacks `state`, so
        //   which bits move is irrelevant — it's only a monotonic change-detector),
        //   all workers unparked, and each worker decides via `job.active`. Decode
        //   is narrow and memory-bound and wants the whole pool kept hot across
        //   its tiny-GEMV / huge-vocab-GEMV mix, so it pays neither the
        //   active-sizing nor the packed-`state` plumbing.
        if self.shared.active_barrier {
            self.shared.pending.store(active - 1, Ordering::Release);
            let next_epoch = state_epoch(self.shared.state.load(Ordering::Relaxed)) + 1;
            self.shared
                .state
                .store(pack_state(next_epoch, active), Ordering::Release);
            for h in self.workers.iter().take(active - 1) {
                h.thread().unpark();
            }
        } else {
            self.shared
                .pending
                .store(self.num_threads - 1, Ordering::Release);
            self.shared.state.fetch_add(1, Ordering::Release);
            for h in &self.workers {
                h.thread().unpark();
            }
        }

        // From here until the barrier drains, workers hold raw pointers into
        // `y` and `f` — so the drain must happen even if the caller's own
        // closure panics below. The guard's Drop blocks until `pending == 0`
        // on both the normal and unwind paths.
        {
            let _drain = DrainGuard {
                shared: &self.shared,
            };
            // Caller (worker 0) steals chunks alongside the background workers,
            // reusing `y_ptr` (same tag). Being the fastest core (pinned to the
            // prime core), it naturally claims the most.
            steal_and_run(&self.shared, &job);
        }

        // A worker's closure panicked (caught in `worker_loop` so the pool
        // survives): resume its original payload on the calling thread, like
        // rayon, so the panic message/type survive the thread hop. Plain load
        // on the hot path (visibility rides the `pending` Release/Acquire
        // barrier the drain just crossed; stale state is consumed pre-publish).
        if self.shared.panicked.load(Ordering::Relaxed) {
            match self.take_panic_payload() {
                Some(payload) => std::panic::resume_unwind(payload),
                None => panic!("cera RowPool: a row closure panicked on a worker thread"),
            }
        }
    }

    /// Consume the worker-panic state: clears the flag and takes the stored
    /// payload as ONE primitive, so no call site can clear the flag while
    /// leaving a stale payload behind (which a later panic would then
    /// mis-report). Tolerates a poisoned slot (the mutex is only locked
    /// around a store/take, but a panicking payload `Drop` elsewhere could in
    /// principle poison it).
    fn take_panic_payload(&self) -> Option<Box<dyn std::any::Any + Send>> {
        self.shared.panicked.store(false, Ordering::Relaxed);
        self.shared
            .panic_payload
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
    }
}

/// Blocks until the current dispatch's barrier drains (`pending == 0`), on
/// both the normal path and the unwind path — workers hold raw pointers into
/// the dispatcher's frame until then. The Acquire load synchronizes with each
/// worker's Release `fetch_sub`, making their output writes visible. Spins
/// briefly, then yields, so a preempted worker isn't starved by the wait.
struct DrainGuard<'a> {
    shared: &'a Shared,
}

impl Drop for DrainGuard<'_> {
    fn drop(&mut self) {
        let mut spins = 0u32;
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            // saturating: an overflow panic inside Drop on the unwind path
            // would abort the process; a wedged barrier should stay a
            // diagnosable spin/yield loop instead.
            spins = spins.saturating_add(1);
            if spins < DRAIN_SPIN_BEFORE_YIELD {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
    }
}

impl Drop for RowPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Bump the epoch + unpark so spinning/parked workers observe shutdown.
        // `fetch_add(1 << ACTIVE_BITS)` steps the epoch field (leaving the stale
        // `active` bits — they don't matter): `shutdown` is Released *before*
        // this store, so any worker that Acquire-observes the new epoch also
        // sees `shutdown == true` and returns before reading `job`.
        self.shared
            .state
            .fetch_add(1 << ACTIVE_BITS, Ordering::Release);
        for h in &self.workers {
            h.thread().unpark();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Background worker: wait for each new epoch, run this worker's row range,
/// signal completion. Exits on shutdown.
fn worker_loop(shared: Arc<Shared>, worker_id: usize, pin_core: Option<usize>, spin_limit: u32) {
    if let Some(core) = pin_core {
        let _ = pin_current_thread_to_core(core);
    }
    // Run this worker's chunks of `job`, catching a closure panic so the worker
    // still reaches its `pending` decrement (a dead worker would wedge the pool);
    // the dispatcher re-raises the panic on the calling thread after the drain.
    let run_job = |shared: &Shared, job: &Job| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            steal_and_run(shared, job);
        }));
        if let Err(payload) = result {
            let mut slot = shared
                .panic_payload
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(payload);
                drop(slot);
            } else {
                // Another worker already stored its payload. Leak this one rather
                // than dropping it here: a payload whose `Drop` panics would unwind
                // past the `pending` decrement below and wedge the pool.
                drop(slot);
                std::mem::forget(payload);
            }
            shared.panicked.store(true, Ordering::Release);
        }
    };

    let mut last_state = 0u64;
    loop {
        // Wait for a new dispatch (or shutdown). Every dispatch changes the
        // `state` word — the active-sized barrier bumps the epoch in the high bits
        // (`pack_state`), the full barrier `fetch_add(1)`s the low bits as a bare
        // counter — so a raw-value change is the wake signal in either mode.
        let mut spins = 0u32;
        let mut parked = false;
        let state;
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let st = shared.state.load(Ordering::Acquire);
            if st != last_state {
                last_state = st;
                state = st;
                break;
            }
            // Saturating so a long idle wait (huge `CERA_SPIN`, or repeated
            // spurious park wakeups) can't overflow-panic this worker in debug
            // builds — a panic here would wedge the pool (`pending` never hits
            // 0). Matches the `DrainGuard` counter.
            spins = spins.saturating_add(1);
            if spins < spin_limit {
                std::hint::spin_loop();
            } else {
                // park() returns immediately if an unpark token is pending, so
                // there's no lost-wakeup between the epoch check and the park.
                thread::park();
                parked = true;
            }
        }
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        // Re-assert the pin after an inter-token park: Android cpuset cgroup
        // migrations (background ↔ foreground) overwrite per-thread affinity
        // masks, silently unpinning parked workers. One ~µs syscall per
        // worker per token at most — parks only happen in the ms-scale
        // inter-token gaps, never between the µs-apart GEMVs of one token.
        if parked && let Some(core) = pin_core {
            let _ = pin_current_thread_to_core(core);
        }
        // SAFETY (both arms): the Acquire load of `state` above synchronizes with
        // the dispatcher's Release bump, so the fresh `Job` (written before that
        // bump) is visible and its pointers are live until this worker decrements
        // `pending` below.
        if shared.active_barrier {
            // Active-sized barrier: `active` rides the same atomic as the epoch
            // (see [`pack_state`]), so it can't name a newer dispatch whose `job`
            // the dispatcher has retired. An idle worker (`id >= active`) skips
            // *entirely* — no `job` read, no `pending` touch — so its `fetch_sub`
            // never bounces the barrier line across the CCD fabric.
            if worker_id >= state_active(state) {
                continue;
            }
            if let Some(job) = unsafe { *shared.job.get() } {
                run_job(&shared, &job);
            }
            shared.pending.fetch_sub(1, Ordering::Release);
        } else {
            // Full barrier: byte-identical to the pre-active-barrier pool. Every
            // worker reads `job`, runs iff `id < job.active`, and decrements — so
            // the dispatcher's `pending == num_threads - 1` wait covers every
            // worker's `job` access this epoch. A `None` job (only on the
            // shutdown/spurious path) skips the decrement, as before.
            let job = match unsafe { *shared.job.get() } {
                Some(job) => job,
                None => continue,
            };
            if worker_id < job.active {
                run_job(&shared, &job);
            }
            shared.pending.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Pin the calling thread to `core` via `sched_setaffinity`. Best-effort:
/// returns whether the pin took — it fails when the core is offline or the
/// process cpuset excludes it (e.g. an Android background cgroup restricted
/// to little cores), in which case the thread stays schedulable as before.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn pin_current_thread_to_core(core: usize) -> bool {
    set_current_thread_affinity(&[core])
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) fn pin_current_thread_to_core(_core: usize) -> bool {
    false
}

/// Every core this crate is willing to pin a worker to, fastest-first, or
/// nothing at all when `CERA_PIN` is off. The single place that decision is
/// made; `RowPool::build` takes its core list already filtered through here.
///
/// This is the *full* usable set, efficiency cores included, because a widened
/// pool needs a distinct core for every worker (see `CoreTopology::pin_cores`
/// for the 35x measurement behind that). A caller that wants only the fast
/// cores wants [`perf_pinned_cores`] instead.
pub(crate) fn pinned_cores() -> &'static [usize] {
    if pinning_enabled() {
        &super::cpu_features::core_topology().pin_cores
    } else {
        &[]
    }
}

/// The performance-core prefix of [`pinned_cores`], for callers that want a set
/// mask rather than one core per worker: today
/// [`super::cpu::ensure_rayon_global_pool`], which confines rayon's workers to
/// the fast cores without handing each a private one.
///
/// `pin_cores` is fastest-first and `fast_cores` counts how many of its leading
/// entries are performance cores, so the prefix is exactly that set. It reads
/// `fast_cores` and **not** `perf_core_count`: the latter is a pool width that
/// `CERA_THREADS` moves, so on a 6P+2E part `CERA_THREADS=8` would widen this
/// mask onto both efficiency cores, which is the straggler-on-every-barrier
/// problem the width policy exists to avoid.
pub(crate) fn perf_pinned_cores() -> &'static [usize] {
    let cores = pinned_cores();
    let fast = super::cpu_features::core_topology().fast_cores;
    &cores[..fast.min(cores.len())]
}

/// Set the calling thread's affinity to exactly `cores`, via
/// `sched_setaffinity`. Same best-effort contract as
/// [`pin_current_thread_to_core`], of which this is the general case.
///
/// Unlike that function this can *widen* a mask, which is the point: a thread
/// inherits the mask of whoever spawned it, so a pool spawned from a thread
/// [`RowPool::pin_caller_once`] already confined to one core would otherwise
/// run every worker on that core. An empty `cores` is a no-op returning
/// `false`: `sched_setaffinity` rejects an empty mask, and callers that got
/// an empty core list have nothing to assert anyway.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn set_current_thread_affinity(cores: &[usize]) -> bool {
    if cores.is_empty() {
        return false;
    }
    // Indices at or beyond the set's capacity are out of bounds for
    // `cpu_set_t` and make `CPU_SET` panic, so they are dropped instead. The
    // bound is spelled from `size_of::<cpu_set_t>()` rather than `CPU_SETSIZE`
    // because the latter is `c_int` on glibc and `size_t` on Android, so no
    // one cast of it is redundant-free on both.
    let capacity = std::mem::size_of::<libc::cpu_set_t>() * 8;
    // SAFETY: `set` is zero-initialized then populated via the libc CPU_SET
    // macro, only at in-bounds indices; `sched_setaffinity(0, ...)` targets
    // the current thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let mut any = false;
        for &core in cores {
            if core < capacity {
                libc::CPU_SET(core, &mut set);
                any = true;
            }
        }
        any && libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) fn set_current_thread_affinity(_cores: &[usize]) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_matches_serial_across_shapes_and_threads() {
        // Fill each row with a function of its absolute index; compare pool
        // output to the serial reference for several sizes, `n`, and pool sizes,
        // under both barrier modes (active-sized = prefill, full = decode).
        for &active_barrier in &[true, false] {
            for &num_threads in &[1usize, 2, 4, 7] {
                let pool = RowPool::build(num_threads, &[], active_barrier);
                for &n in &[1usize, 3, 8] {
                    for &total_rows in &[0usize, 1, 5, 256, 1000] {
                        let len = total_rows * n;
                        let mut got = vec![0.0f32; len];
                        let mut want = vec![0.0f32; len];
                        let fill = |row: usize, slice: &mut [f32]| {
                            for (k, v) in slice.iter_mut().enumerate() {
                                *v = (row * 100 + k) as f32;
                            }
                        };
                        pool.dispatch_rows(&mut got, n, 64, fill);
                        for row in 0..total_rows {
                            fill(row, &mut want[row * n..row * n + n]);
                        }
                        assert_eq!(
                            got, want,
                            "mismatch: barrier={active_barrier} threads={num_threads} n={n} rows={total_rows}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_row_written_exactly_once() {
        // A concurrency-stress check for the disjoint partition: each row adds 1
        // to a counter; after dispatch every counter must be exactly 1 (no lost
        // or double writes).
        let pool = RowPool::build(4, &[], true);
        let total_rows = 10_000usize;
        let mut counts = vec![0.0f32; total_rows];
        pool.dispatch_rows(&mut counts, 1, 1, |_row, slice| {
            slice[0] += 1.0;
        });
        assert!(counts.iter().all(|&c| c == 1.0));
    }

    #[test]
    fn repeated_dispatches_reuse_workers() {
        // Same pool, many dispatches — exercises the epoch/park/unpark cycle and
        // confirms the workers stay correct across rounds.
        let pool = RowPool::build(4, &[], true);
        let mut y = vec![0.0f32; 2048];
        for iter in 0..50 {
            pool.dispatch_rows(&mut y, 1, 1, |row, slice| {
                slice[0] = (row + iter) as f32;
            });
            for (row, &v) in y.iter().enumerate() {
                assert_eq!(v, (row + iter) as f32);
            }
        }
    }

    #[test]
    fn varying_active_across_dispatches_is_correct() {
        // Stress the idle→active transition that the packed-(epoch, active)
        // barrier must get right: on a wide pool, alternate a 1-row dispatch
        // (only worker 0 active, every background worker idle) with a full-width
        // one, many times. A worker idle one epoch must participate correctly the
        // next — and an idle worker must never touch `job`/`pending` for an epoch
        // it skipped (a double-run or missed decrement would corrupt the next
        // dispatch). Each dispatch fully overwrites `y`, so any stale/double run
        // or lost row is caught. Run under Miri, this exercises the data-race and
        // Stacked-Borrows discipline of the `active`-gated `job` read. Both
        // barrier modes: the active-sized barrier is where idle workers skip the
        // decrement, but the full barrier's idle-worker decrement must stay sound
        // under the same swing too.
        for &active_barrier in &[true, false] {
            for &num_threads in &[2usize, 4, 8] {
                let pool = RowPool::build(num_threads, &[], active_barrier);
                for iter in 0..60usize {
                    // Swing `active` between 1 (narrow) and the full width.
                    let total_rows = if iter % 2 == 0 { 1 } else { 2048 };
                    let mut y = vec![0.0f32; total_rows];
                    pool.dispatch_rows(&mut y, 1, 1, |row, slice| slice[0] = (row + iter) as f32);
                    for (row, &v) in y.iter().enumerate() {
                        assert_eq!(
                            v,
                            (row + iter) as f32,
                            "barrier={active_barrier} threads={num_threads} iter={iter} row={row}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dispatch_rows_work_matches_serial() {
        // The work-cap only changes *how many* (and thus which) workers run a
        // row, never the per-row result. Across depths (⇒ different `active`
        // caps) the output must still equal the serial reference, with no row
        // lost when the cap idles most of the pool.
        let pool = RowPool::build(8, &[], true);
        for &depth in &[0usize, 1, 4096, 1_000_000] {
            for &total_rows in &[0usize, 1, 3, 64, 500] {
                let n = 4;
                let mut got = vec![0.0f32; total_rows * n];
                let mut want = vec![0.0f32; total_rows * n];
                let fill = |row: usize, slice: &mut [f32]| {
                    for (k, v) in slice.iter_mut().enumerate() {
                        *v = (row * 7 + k) as f32;
                    }
                };
                pool.dispatch_rows_work(&mut got, n, 1, depth, fill);
                for row in 0..total_rows {
                    fill(row, &mut want[row * n..row * n + n]);
                }
                assert_eq!(got, want, "depth={depth} rows={total_rows}");
            }
        }
    }

    #[test]
    fn single_thread_pool_runs_serially() {
        let pool = RowPool::build(1, &[], true);
        assert_eq!(pool.num_threads(), 1);
        let mut y = vec![0.0f32; 100];
        pool.dispatch_rows(&mut y, 1, 1, |row, slice| slice[0] = row as f32);
        assert!(y.iter().enumerate().all(|(i, &v)| v == i as f32));
    }

    #[test]
    fn trailing_partial_row_matches_serial_chunks() {
        // y.len() % n != 0: the tail must be visited with the short slice,
        // exactly like the serial `chunks_mut(n)` fallback.
        let pool = RowPool::build(4, &[], true);
        let n = 8usize;
        let len = 8 * 300 + 5; // 300 full rows + a 5-element tail
        let mut got = vec![0.0f32; len];
        let fill = |row: usize, slice: &mut [f32]| {
            for (k, v) in slice.iter_mut().enumerate() {
                *v = (row * 1000 + k) as f32 + 1.0;
            }
        };
        pool.dispatch_rows(&mut got, n, 1, fill);
        let mut want = vec![0.0f32; len];
        for (j, row) in want.chunks_mut(n).enumerate() {
            fill(j, row);
        }
        assert_eq!(got, want);
    }

    #[test]
    fn concurrent_dispatchers_are_safe() {
        // Two threads dispatching on the same pool simultaneously: the loser of
        // the dispatch lock runs serially — both outputs must still be exact.
        let pool = RowPool::build(4, &[], true);
        for _ in 0..20 {
            let mut a = vec![0.0f32; 4096];
            let mut b = vec![0.0f32; 4096];
            thread::scope(|s| {
                let pool = &pool;
                s.spawn(|| pool.dispatch_rows(&mut a, 1, 1, |row, s| s[0] = row as f32 + 1.0));
                pool.dispatch_rows(&mut b, 1, 1, |row, s| s[0] = row as f32 + 2.0);
            });
            assert!(a.iter().enumerate().all(|(i, &v)| v == i as f32 + 1.0));
            assert!(b.iter().enumerate().all(|(i, &v)| v == i as f32 + 2.0));
        }
    }

    #[test]
    fn worker_panic_propagates_and_pool_survives() {
        let pool = RowPool::build(4, &[], true);
        let mut y = vec![0.0f32; 4096];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.dispatch_rows(&mut y, 1, 1, |row, slice| {
                if row == 2048 {
                    panic!("boom");
                }
                slice[0] = row as f32;
            });
        }));
        // The panic must propagate with its ORIGINAL payload (rayon's
        // contract) — whether row 2048 landed on the caller or a worker.
        let payload = result.expect_err("closure panic must propagate to the caller");
        let msg: &str = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .expect("payload must be the original panic message");
        assert_eq!(msg, "boom");
        // The pool must remain fully usable after a panicked dispatch.
        let mut z = vec![0.0f32; 4096];
        pool.dispatch_rows(&mut z, 1, 1, |row, slice| slice[0] = row as f32);
        assert!(z.iter().enumerate().all(|(i, &v)| v == i as f32));
    }
}

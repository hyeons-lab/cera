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
//! from a shared `next_row` cursor until the row space is exhausted: dynamic
//! work-stealing, so faster cores grab more chunks and every worker reaches the
//! barrier together (heterogeneous big.LITTLE load balancing). On a part whose
//! cores differ in speed, a worker also sizes its chunk to the core it is pinned
//! to, which shrinks (but does not remove) the time a chunk claimed late by a
//! slow core holds everyone at the barrier (see `worker_chunk_rows`, which
//! documents why the residual matters). Each participating worker then
//! `fetch_sub`s a `pending` counter; the caller spins on `pending ==
//! 0` (Acquire) before returning. That Release/Acquire pair both (a) publishes
//! the `Job` to workers and (b) establishes happens-before for every worker's
//! writes to the output, so the disjoint `&mut` handoff is sound and the caller
//! may read the output once it returns. (Each claim advances the cursor by
//! exactly the rows it reserves, so chunks are disjoint row ranges even when
//! workers claim different amounts.)
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
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::sync::atomic::AtomicI64;
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
    /// Rows per steal unit **for a worker on the fastest core**. Workers claim
    /// contiguous ranges from the shared `next_row` cursor: dynamic
    /// work-stealing, so fast cores grab more ranges and all workers reach the
    /// barrier together (heterogeneous big.LITTLE load balancing). A worker on a
    /// slower core scales this down by its weight; see
    /// [`Shared::worker_weights`] and [`worker_chunk_rows`].
    chunk_rows: usize,
    /// Floor for the per-worker scale-down, as resolved by the dispatcher from
    /// the caller's `min_chunk_rows`. Carried on the job because the scaling
    /// happens per worker, after `chunk_rows` has already been decided.
    min_chunk_rows: usize,
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
    /// Next unclaimed **row**. Active workers (and the caller) claim a range by
    /// `fetch_add`-ing the number of rows they want; reset to 0 before each
    /// dispatch's `state` bump.
    ///
    /// A row cursor rather than a chunk index because chunk size is per-worker
    /// (see `worker_weights`): `chunk_index * chunk_rows` only reconstructs a
    /// range when every worker uses the same size. Advancing the cursor by
    /// exactly the rows being reserved keeps claims disjoint whatever sizes the
    /// workers pick.
    next_row: AtomicUsize,
    /// Per-worker chunk scale, indexed by worker id, in units of
    /// [`super::cpu_features::WEIGHT_FULL`] (so a full-weight worker gets the
    /// dispatcher's whole chunk). Empty when every worker should use the full
    /// chunk, which is every homogeneous host and every host without affinity.
    /// Immutable after `build`.
    worker_weights: Vec<u32>,
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
    /// Number of background workers currently in `thread::park()`. Avoids
    /// issuing `unpark()` syscalls on every GEMV dispatch when workers are
    /// actively spin-waiting in user space.
    parked_count: AtomicUsize,
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
///
/// This sizes the chunk of a worker on the *fastest* core; a worker on a slower
/// one scales down from it (see [`Shared::worker_weights`]). Raising it to cover
/// a wider spread instead was measured and rejected: it works on the device with
/// the spread (350M-Q4_K_M prefill p50 201 → 213 at the shipping width) but it
/// moves every decode GEMV and every homogeneous host too, paying the finer
/// granularity's atomic and setup cost where there is no straggler to fix.
const STEAL_CHUNKS_PER_WORKER: usize = 4;
/// Floor on chunk size: below this, per-chunk atomic + kernel-setup overhead
/// starts to matter and contiguous streaming gets choppy. Applies to the
/// per-worker scaled size too, so weighting can shrink a slow core's chunk but
/// not past the point where the claim costs more than the work.
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

/// Whether the prefill pool should pin its workers 1:1 at width `n`.
///
/// False exactly when some worker would land on a core outside the fast set:
/// `pin_cores` is fastest-first and worker `i` takes `pin_cores[i]`, so
/// `n > fast_cores` means at least one worker is nailed to an efficiency core
/// it can never be moved off. `fast_cores == 0` (macOS/iOS, homogeneous
/// desktop) has no split to act on, and those hosts have an empty `pin_cores`
/// anyway, so the policy is inert there.
///
/// A named function rather than an inline expression so the policy test guards
/// *this* predicate instead of a copy of it.
#[inline]
fn prefill_should_pin(fast_cores: usize, n: usize) -> bool {
    fast_cores == 0 || n <= fast_cores
}

/// How many rows worker `worker_id` claims per steal, scaling the job's chunk
/// size by the relative speed of the core that worker is *meant* to be on.
///
/// "Meant to be": pinning is best-effort everywhere (a restricted cpuset can
/// refuse it, and worker 0 only pins if it wins the process-wide caller-pin
/// claim), so a weight is the intended placement rather than an observed one.
/// Nothing here depends on it being exact, for the reason given below.
///
/// The point is the fork-join barrier: every worker's chunk should cost roughly
/// the same *wall-clock time*, so a chunk claimed late by a slow core cannot
/// hold everyone else at the barrier for a multiple of what it cost to claim.
///
/// **Two different spreads, two different sizes of effect.** The default pools
/// are sized by `perf_core_count`, which counts only cores at or above the
/// detector's performance threshold (`cpu_features::CAP_MID` on the
/// `cpu_capacity` path; 85% of the top clock on the `cpuinfo_max_freq`
/// fallback that x86 hybrid parts take). So on a Tensor G5 they hold the prime
/// core (1024) and five A725s (824/825), and the widest ratio a default
/// dispatch sees is ~1.24x. The 207-capacity A520s only become workers when `CERA_THREADS` /
/// `CERA_PREFILL_THREADS` widens the pool past the performance cores, and that
/// is where an unweighted chunk hurts most: an A520 holds a full-size chunk for
/// ~5x what it costs the prime core. Measured on that part, 350M-Q4_K_M at 512
/// prompt tokens, interleaved and thermally gated:
///
/// | width | prefill tok/s | decode tok/s |
/// |-------|--------------:|-------------:|
/// | default (6), unweighted | 206.5 | 93.7 |
/// | default (6), weighted   | 212.5 | 94.7 |
/// | `CERA_THREADS=8`, unweighted | 116.5 | 68.8 |
/// | `CERA_THREADS=8`, weighted   | 142.0 | 82.2 |
///
/// So this is a small win at the shipping width and a large one on a widened
/// pool. It does not make widening a good idea: 142 is still well below 212.
///
/// **The widened row is now historical.** `RowPool::prefill` stops pinning once
/// its width exceeds `fast_cores`, and unpinned workers carry no weights (there
/// is no known core to weight them by), so weighting no longer applies to that
/// configuration at all. What ships today is the default-width row, plus decode,
/// which still pins at every width. The widened numbers are kept because they
/// are what motivated the mechanism, not because they describe current
/// behaviour.
///
/// Weights only ever scale *down* from `job.chunk_rows`, so the fastest core's
/// granularity is unchanged and the chunk count only ever rises: this is a
/// refinement of the uniform schedule, not a different one. With no weights
/// (homogeneous host, or no affinity to place a worker with) it returns
/// `job.chunk_rows` unchanged and the whole mechanism is inert.
///
/// **The weights under-correct, and that is the honest reason this is safe.**
/// Both sources over-rate a slow core's real throughput on these kernels:
/// `calibrate::prefill_thread_count` records that `cpu_capacity` rates an A520
/// 4x slower than an A725 while a quantized NEON dotprod kernel makes it worse
/// than that, and `cpuinfo_max_freq` ratios ignore IPC entirely (an x86 E-core
/// at 0.75 the clock is well under 0.75 the throughput on a wide FMA kernel).
/// So the safety does not come from a margin in the weights. It comes from the
/// direction of the change: any weight below full is strictly closer to
/// equal-time chunks than the uniform chunk it replaces, so the straggler can
/// only shrink. The residual under-correction is visible in the table above,
/// as the gap between the weighted wide pool and the narrow one.
///
/// Two consequences worth keeping straight before tuning this. Weighting does
/// not *bound* the tail, so it is not a licence to widen the default pool. And
/// an undersized chunk is cheap (one extra `fetch_add` and the worker steals
/// again), so if these weights are ever replaced by measured ones, biasing
/// them low is the safe direction to be wrong in.
#[inline]
fn worker_chunk_rows(shared: &Shared, job: &Job, worker_id: usize) -> usize {
    match shared.worker_weights.get(worker_id) {
        Some(&w) if w < super::cpu_features::WEIGHT_FULL => {
            // Widened for the multiply so the scale cannot overflow a 32-bit
            // `usize` on a large row space; the quotient is ≤ `chunk_rows` and
            // so always fits back.
            let scaled = (job.chunk_rows as u64 * u64::from(w)
                / u64::from(super::cpu_features::WEIGHT_FULL)) as usize;
            // `max(1)` is the progress guarantee, not a tuning choice: a
            // zero-row claim would never advance the cursor and would spin here
            // forever. `min_chunk_rows` is already ≥ 1 at the dispatcher, but
            // this does not depend on that holding.
            scaled.max(job.min_chunk_rows).max(1)
        }
        // No weight for this worker (unpinned surplus worker, or a host with no
        // spread) ⇒ full chunk, exactly as before weighting existed.
        _ => job.chunk_rows,
    }
}

/// Claim and run row ranges from `shared.next_row` until the row space is
/// exhausted. Shared by the caller (worker 0) and every active background
/// worker — each `fetch_add` hands out a unique, disjoint contiguous row range.
#[inline]
fn steal_and_run(shared: &Shared, job: &Job, worker_id: usize) {
    // Resolved once per dispatch rather than per claim: it depends only on the
    // worker and the job, and the steal loop is the hot path.
    let chunk_rows = worker_chunk_rows(shared, job, worker_id);
    debug_assert!(chunk_rows >= 1, "a zero-row claim would never terminate");
    loop {
        // Relaxed: uniqueness/atomicity of the claim is all we need here; the
        // visibility of each worker's output writes to the caller is provided
        // by the `pending` Release/Acquire barrier at the end of the dispatch.
        //
        // Each claim advances the cursor by exactly the rows it reserves, so
        // ranges stay disjoint even though workers use different sizes. The
        // cursor is allowed to run past `total_rows`: a worker that lands there
        // stops, so it overshoots by at most one claim per worker, far below
        // any `usize` wrap.
        let start = shared.next_row.fetch_add(chunk_rows, Ordering::Relaxed);
        if start >= job.total_rows {
            break;
        }
        let end = (start + chunk_rows).min(job.total_rows);
        // SAFETY: each row range is claimed by exactly one worker, so
        // `[start, end)` is disjoint from every other worker's, so no two
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
    /// Whether this pool pinned its workers 1:1 to distinct cores.
    ///
    /// Recorded rather than re-derived: the policy is decided from the
    /// *requested* width, while [`RowPool::num_threads`] is the width actually
    /// achieved after any spawn failures. Those can differ, so deriving the
    /// placement from `num_threads` would report the wrong answer for a pool
    /// that failed to spawn down below `fast_cores`.
    ///
    /// Test-only: nothing in the running engine needs to know, and carrying it
    /// unconditionally would be a field the lib build never reads.
    #[cfg(test)]
    workers_pinned: bool,
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
        /// those are exact counts of exact events, and detecting the silent
        /// fallback at all is what these counters are for. ("Exact" is about
        /// the counting, not the read: [`snapshot`] takes four separate relaxed
        /// loads, so a concurrent dispatch can leave one of them a single
        /// increment behind. See its docs.)
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
            // Clamped so the documented 0.0..=1.0 range holds for a `PoolStats`
            // assembled by hand or differenced across a counter reset, not only
            // for one that came from `snapshot`.
            (self.serial_macs as f64 / self.fanout_macs as f64).clamp(0.0, 1.0)
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
    ///
    /// Four independent relaxed loads, so this is not an atomic snapshot: a
    /// dispatch running concurrently can land between them. The **order** of
    /// the loads is therefore load-bearing. A dispatch increments its `FANOUT_*`
    /// counter before the matching `SERIAL_*` one, so reading the `SERIAL_*`
    /// side *first* keeps the pair consistent in the common case: whatever
    /// `FANOUT_*` is read afterwards is at least as advanced as the `SERIAL_*`
    /// value already in hand. Reading them the other way round can report
    /// `serial_fallbacks=1, fanout_dispatches=0`, i.e. a fallback rate above
    /// 100%, which is exactly backwards for a diagnostic whose whole job is to
    /// say how much work fell back.
    ///
    /// Ordering alone is not a proof under a weak memory model, so the
    /// invariant is also enforced rather than merely argued: the `serial_*`
    /// fields are clamped to their `fanout_*` counterparts. The clamp can only
    /// ever *understate* a fallback by one in-flight dispatch, and the counters
    /// are read once per bench run against a hot path that must stay relaxed,
    /// which is the trade being made here.
    pub fn snapshot() -> PoolStats {
        let serial_fallbacks = SERIAL_FALLBACKS.load(Ordering::Relaxed);
        let serial_macs = SERIAL_MACS.load(Ordering::Relaxed);
        let fanout_dispatches = FANOUT_DISPATCHES.load(Ordering::Relaxed);
        let fanout_macs = FANOUT_MACS.load(Ordering::Relaxed);
        PoolStats {
            fanout_dispatches,
            fanout_macs,
            serial_fallbacks: serial_fallbacks.min(fanout_dispatches),
            serial_macs: serial_macs.min(fanout_macs),
        }
    }
}

/// Whether affinity pinning is enabled (`CERA_PIN=0`/`false`/`off`,
/// case-insensitive, disables it — for host apps that manage thread placement
/// and don't want cera's permanent caller pin). Resolved once.
pub(crate) fn pinning_enabled() -> bool {
    // Inverse of the shared reader. The sizing policy in `cpu_features` and
    // `calibrate` needs the same answer, so the switch is resolved in one place
    // rather than parsed once per consumer.
    !super::cpu_features::pinning_disabled()
}

/// A global pool plus the usable CPUs it was sized for. Pools are rebuilt, not
/// mutated, when usable CPUs change: dispatches in flight hold their own
/// `Arc` and finish on the old width, while new dispatches take the new one.
/// `None` until first use so the decode pool stays lazy (its width needs the
/// loaded model's shape; see [`RowPool::decode`]).
struct PoolSlot {
    pool: Arc<RowPool>,
    usable: super::cpu_features::UsableCpus,
}

static PREFILL_SLOT: Mutex<Option<PoolSlot>> = Mutex::new(None);
static DECODE_SLOT: Mutex<Option<PoolSlot>> = Mutex::new(None);

impl RowPool {
    /// Full-width pool for compute-bound batched work: prefill GEMM
    /// ([`super::cpu::par_rows_n`]). Width comes from
    /// `super::calibrate::prefill_thread_count`: the performance cores, since
    /// batched matmul is compute-bound and scales with threads, but *not* the
    /// efficiency cores, which measured as a large net loss. Overridable with
    /// `CERA_PREFILL_THREADS=<n>` up to every pinnable core. Lazily built so
    /// `CeraEngine` consumers get it without calling
    /// [`super::cpu::configure_thread_pool`], and rebuilt by
    /// `resize_pools_for_cpuset` when the cpuset moves. The returned `Arc`
    /// keeps a superseded pool alive until in-flight dispatches finish, so a
    /// resize never strands a dispatcher.
    pub fn prefill() -> Arc<RowPool> {
        get_or_build_pool(&PREFILL_SLOT, build_prefill)
    }

    /// Narrow pool for per-token work: decode GEMV
    /// ([`super::cpu::par_rows`]). Width comes from
    /// `super::calibrate::decode_thread_count`, which sizes it from the loaded
    /// model's `DecodeShape` (bytes per pool dispatch) on a homogeneous host,
    /// and on heterogeneous big.LITTLE keeps the full big-core set, decode's
    /// measured optimum there. Overridable with `CERA_DECODE_THREADS=<n>`.
    ///
    /// **Whoever touches this pool first sizes it.** That is why
    /// `super::cpu::configure_thread_pool` deliberately does not warm it: it
    /// runs before any model is loaded, so warming it there would size the
    /// pool to the model-less fallback and silently disable shape-based
    /// sizing (a later cpuset resize would only heal that if the allowance
    /// also changed). Keep it lazy.
    pub fn decode() -> Arc<RowPool> {
        get_or_build_pool(&DECODE_SLOT, build_decode)
    }

    /// Build a pool with `num_threads` total workers, pinning worker `i` to
    /// `pin_cores[i]` when present (surplus workers run unpinned). `pin_cores`
    /// empty ⇒ no 1:1 pinning; the workers are then confined to `spread_mask`
    /// instead, if that is non-empty, so they are *placed* rather than left to
    /// inherit the caller's mask. Spawn failures degrade the thread count rather
    /// than panicking.
    ///
    /// **The caller pin follows `pin_cores`, so it is not unconditional.**
    /// `caller_pin` is `pin_cores.first()`, so with an empty `pin_cores` the
    /// pool claims no process-wide caller pin and [`RowPool::pin_caller_once`]
    /// is a no-op for it. Two callers reach that state deliberately: a host with
    /// no usable affinity, and a prefill pool wider than the fast cores (see
    /// [`RowPool::prefill`]). A pool that *does* pin still claims the caller pin
    /// for whatever thread first dispatches through it.
    ///
    /// `core_weights` is the relative speed of each entry of `pin_cores`, in
    /// units of [`super::cpu_features::WEIGHT_FULL`]; worker `i` sizes its steal
    /// chunk by `core_weights[i]` (see [`Shared::worker_weights`]). Empty, or
    /// shorter than the pool, leaves the uncovered workers on the full chunk.
    ///
    /// `pin_cores` is taken as already policy-filtered: callers pass
    /// [`pin_cores_in`]/[`pin_weights_in`] over their sized topo, which is
    /// where `CERA_PIN` is honoured; [`pinned_cores`] is the cached-topology
    /// shorthand.
    fn build(
        num_threads: usize,
        pin_cores: &[usize],
        core_weights: &[u32],
        spread_mask: &'static [usize],
        active_barrier: bool,
    ) -> RowPool {
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
        // Only carry weights when at least one worker would actually scale down.
        // An all-`WEIGHT_FULL` vector (a host with no spread) would make
        // `worker_chunk_rows` do lookup work per dispatch to conclude nothing,
        // so drop it and let the empty-vector path answer for the whole pool.
        //
        // Truncated to the requested width: a weight beyond it belongs to a
        // core no worker sits on. It is the *requested* width because the pool
        // has not been spawned yet; if a spawn later fails the pool shrinks and
        // this keeps a few weights no worker id can reach, which is harmless
        // (nothing indexes them) but means the vector is an upper bound rather
        // than an exact match.
        let worker_weights: Vec<u32> = core_weights.iter().take(num_threads).copied().collect();
        let worker_weights = if worker_weights
            .iter()
            .any(|&w| w < super::cpu_features::WEIGHT_FULL)
        {
            worker_weights
        } else {
            Vec::new()
        };
        let shared = Arc::new(Shared {
            active_barrier,
            state: AtomicU64::new(0),
            next_row: AtomicUsize::new(0),
            worker_weights,
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            parked_count: AtomicUsize::new(0),
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
                .spawn(move || worker_loop(shared, id, pin, spread_mask, spin_limit))
            {
                Ok(handle) => workers.push(handle),
                // Couldn't spawn: cap the pool at what we have, loudly. The
                // slot records the requested allowance either way, so a
                // silent shortfall would read as tracked-but-narrow forever.
                Err(e) => {
                    tracing::warn!(
                        "cera: worker spawn failed ({e}); capping pool at {} of {num_threads} workers",
                        1 + workers.len(),
                    );
                    break;
                }
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
            #[cfg(test)]
            workers_pinned: !pin_cores.is_empty(),
        }
    }

    /// Total worker count (caller + spawned background workers).
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

/// Fetch the pool in `slot`, building it on first use: size from a fresh
/// probe, never from a cached width (the cpuset may have moved between
/// topology detection and now). Built under the slot lock (like the old
/// `OnceLock`) so concurrent first touches serialize instead of building
/// twice. Shared by both accessors so the allowance bookkeeping cannot drift
/// between them.
fn get_or_build_pool(
    slot: &Mutex<Option<PoolSlot>>,
    build: fn(&super::cpu_features::CoreTopology) -> RowPool,
) -> Arc<RowPool> {
    let mut guard = slot.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(built) = guard.as_ref() {
        return Arc::clone(&built.pool);
    }
    let usable = super::cpu_features::cpu_usable_cpus();
    let topo = sized_topo_for_pools(&usable);
    let pool = Arc::new(build(&topo));
    let out = Arc::clone(&pool);
    *guard = Some(PoolSlot { pool, usable });
    out
}

/// Build the prefill pool for `topo`: width from
/// `super::calibrate::prefill_thread_count`, pins from `topo`, spread mask
/// from the cached usable set. The spread stays cache-sourced because it
/// needs `&'static` for the worker threads. Membership drift is benign: the
/// kernel constrains the mask to the live cpuset, and a fully disjoint mask
/// fails the call, whose result is ignored (the thread keeps its inherited
/// mask).
fn build_prefill(topo: &super::cpu_features::CoreTopology) -> RowPool {
    let n = super::calibrate::prefill_thread_count(topo);
    // Active-sized barrier: prefill GEMMs vary widely in size, and a
    // small one is better run on the few cores its work can fill than
    // dragged across the whole pool's barrier (see `Shared::active_barrier`).
    //
    // **Pinning is dropped once the pool outgrows the fast cores.**
    // 1:1 pinning nails a worker to a specific efficiency core and
    // leaves it there; the scheduler cannot move it even when it is the
    // straggler every other worker is waiting on at the barrier. Below
    // that width every worker lands on a fast core and pinning is a
    // clear win, so the policy flips rather than being abandoned.
    //
    // Measured on a Pixel 10 Pro Fold (Tensor G5, 6 fast + 2 A520),
    // 350M-Q4_K_M at 512 prompt tokens, 5 interleaved rounds each,
    // thermally and load gated, prefill tok/s:
    //
    // | width | pinned | unpinned |
    // |-------|-------:|---------:|
    // | default (6, fits the fast cores) | **211** | 189 |
    // | `CERA_THREADS=8` (reaches the A520s) | 148 | **203** |
    //
    // Non-overlapping in both directions. That is a 1.43x cost to
    // widening, cut to 1.04x, against llama.cpp's own 1.06x.
    //
    // Decode keeps pinning at every width; see [`build_decode`].
    let widened = !prefill_should_pin(topo.fast_cores, n);
    let (cores, weights, spread) = if widened {
        // Weights go with the pins: `core_weights[i]` describes the core
        // worker `i` was placed on, and once nothing is placed it
        // describes nothing (the same reasoning as `pin_weights_in`
        // under `CERA_PIN=0`).
        // Empty pins, but a non-empty *mask*: the workers must be
        // confined to the usable core set explicitly rather than
        // inheriting the caller's single-core mask. See `worker_loop`.
        (&[][..], &[][..], pinned_cores())
    } else {
        (pin_cores_in(topo), pin_weights_in(topo), &[][..])
    };
    RowPool::build(n, cores, weights, spread, true)
}

/// Build the decode pool for `topo`: width from
/// `super::calibrate::decode_thread_count`, pins from `topo`. Unlike
/// prefill, decode pins at every width (measurement below); full barrier,
/// every worker hot across the tiny-GEMV / huge-vocab-GEMV mix.
fn build_decode(topo: &super::cpu_features::CoreTopology) -> RowPool {
    let n = super::calibrate::decode_thread_count(topo);
    // Full barrier: decode wants every worker hot across its tiny-GEMV /
    // huge-vocab-GEMV mix (see `Shared::active_barrier`).
    //
    // **Unlike prefill, decode pins at every width.** The two want
    // opposite policies and it is worth being explicit about why: decode
    // is a per-token full barrier over tiny dispatches, where keeping
    // each worker resident on its own core matters more than letting the
    // scheduler move it. Same device and model as the table on
    // [`build_prefill`], decode tok/s: 95.2 pinned against 83.1
    // unpinned at the default width, and 81.0 against 67.8 at
    // `CERA_THREADS=8`. Pinned wins at both, so there is no width at
    // which relaxing it would pay.
    RowPool::build(n, pin_cores_in(topo), pin_weights_in(topo), &[], false)
}

/// Topology sized for pool construction: the cached RAW detection with policy
/// (usable-set shaping, env overrides) applied fresh. Pool builds (initial
/// and resize) always go through here rather than trusting a cached width, so
/// a cpuset move between topology detection and first use cannot stick the
/// pools, and widening restores width (re-clamping an already-clamped topo
/// could only narrow). The raw input is load-bearing and pinned only by
/// review: re-applying policy to an already-shaped cache is a fixed point at
/// a fixed allowance, so no test distinguishes this call site reading the raw
/// static from the policy static without a startup-narrowed cpuset, which
/// tests cannot arrange deterministically. Pins, weights, and the fast-core
/// set start from the raw detection, which is host-wide (sysfs is not
/// re-read, so CPU hotplug still needs a restart), and are intersected with
/// the live usable set on every build, so a cpuset move re-seats pins as
/// well as width.
fn sized_topo_for_pools(
    usable: &super::cpu_features::UsableCpus,
) -> super::cpu_features::CoreTopology {
    let topo = super::cpu_features::core_topology_raw().clone();
    super::cpu_features::apply_pool_policy(topo, usable)
}

/// `try_lock` a pool's dispatch lock: `WouldBlock` means a dispatch is in
/// flight (the caller falls back to serial, or defers the rebuild); poisoned
/// proceeds (the last holder unwound, so nothing is in flight).
fn idle_guard(pool: &RowPool) -> Option<std::sync::MutexGuard<'_, ()>> {
    match pool.dispatch_lock.try_lock() {
        Ok(g) => Some(g),
        Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// What one pool slot needs at a resize boundary: already tracking (or
/// unbuilt, which first use sizes fresh), holding a pool that must be
/// rebuilt, or mid-first-use-build. The third state exists so a deferred
/// slot is logged as deferred, never misread as tracking.
enum PoolNeed {
    Tracking,
    Rebuild(Arc<RowPool>),
    Deferred,
}

impl PoolNeed {
    fn rebuild_pool(self) -> Option<Arc<RowPool>> {
        match self {
            PoolNeed::Rebuild(pool) => Some(pool),
            PoolNeed::Tracking | PoolNeed::Deferred => None,
        }
    }

    fn is_deferred(&self) -> bool {
        matches!(self, PoolNeed::Deferred)
    }
}

/// Slot snapshot for the resize gate. `try_lock`, poison-tolerant: a slot
/// mid-first-use-build reports [`PoolNeed::Deferred`] rather than stalling
/// this boundary, keeping the resize path non-blocking end to end.
fn pool_resize_need(
    slot: &Mutex<Option<PoolSlot>>,
    name: &str,
    usable: &super::cpu_features::UsableCpus,
) -> PoolNeed {
    let allowed = usable.count();
    let guard = match slot.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            tracing::debug!(
                "cera: {name} pool slot mid-build; resize to {allowed} deferred to next boundary"
            );
            return PoolNeed::Deferred;
        }
    };
    match guard.as_ref() {
        Some(s) if &s.usable != usable => PoolNeed::Rebuild(Arc::clone(&s.pool)),
        _ => PoolNeed::Tracking,
    }
}

/// Rebuild the pools that are not tracking the usable answer, skipping
/// unbuilt ones (first use sizes them fresh, which keeps decode lazy until
/// its model's shape is registered). The set half threads through to the
/// rebuild so width and pins shape from the one sample the caller probed.
/// Gated on idleness BEFORE anything is built, so a
/// refused gate costs no spawn. Each pool gates independently: a mid-dispatch
/// pool defers to the next boundary while an idle one still swaps, so a
/// persistently busy pool cannot pin the other at a departed width. Never
/// blocks a dispatcher. A poisoned dispatch lock proceeds like the dispatch
/// path does (the last dispatcher unwound, so nothing is in flight, and the
/// swap heals the poison with a fresh lock). Returns true when it swapped at
/// least one pool.
///
/// Locking: the need snapshot is `try_lock` per slot (an accessor holds a
/// slot lock only for an `Arc` clone, except during first-use builds, which
/// report [`PoolNeed::Deferred`], logged distinctly from the tracking
/// no-op); everything after that is `try_lock` in fixed prefill-then-decode
/// order, so there is no lock-order cycle (accessors never hold a slot guard
/// while taking a dispatch lock) and concurrent rebuilders serialize on the
/// dispatch guards. A cpuset move between the probe and the swap just leaves
/// the stored allowance stale, and the next boundary re-resizes. Every early
/// `false` logs at debug with its leg and target allowance so a stuck width
/// stays diagnosable.
fn rebuild_pools_for_allowance(usable: &super::cpu_features::UsableCpus) -> bool {
    let allowed = usable.count();
    let prefill_need = pool_resize_need(&PREFILL_SLOT, "prefill", usable);
    let decode_need = pool_resize_need(&DECODE_SLOT, "decode", usable);
    let deferred = prefill_need.is_deferred() || decode_need.is_deferred();
    let old_prefill = prefill_need.rebuild_pool();
    let old_decode = decode_need.rebuild_pool();
    if old_prefill.is_none() && old_decode.is_none() {
        // A deferred slot already logged at the snapshot site; only the
        // all-tracking case is a no-op.
        if !deferred {
            tracing::debug!(
                "cera: pools already track usable CPUs {usable:?} (allowance {allowed}); resize no-op"
            );
        }
        return false;
    }
    // Gate each pool on idleness before building: a refused gate must not
    // cost a spawn. Fixed order, never blocking; guards held through the
    // swap. A busy pool defers to the next boundary while an idle one still
    // swaps.
    let _prefill_guard = old_prefill.as_ref().and_then(|p| idle_guard(p));
    let _decode_guard = old_decode.as_ref().and_then(|p| idle_guard(p));
    let rebuild_prefill = old_prefill.is_some() && _prefill_guard.is_some();
    let rebuild_decode = old_decode.is_some() && _decode_guard.is_some();
    if old_prefill.is_some() && !rebuild_prefill {
        tracing::debug!(
            "cera: prefill pool mid-dispatch; resize to {usable:?} (allowance {allowed}) deferred to next boundary"
        );
    }
    if old_decode.is_some() && !rebuild_decode {
        tracing::debug!(
            "cera: decode pool mid-dispatch; resize to {usable:?} (allowance {allowed}) deferred to next boundary"
        );
    }
    if !rebuild_prefill && !rebuild_decode {
        return false;
    }
    // Build replacements while gated: concurrent dispatches from other
    // sessions fall back to serial for one build (~ms, only on real
    // transitions) instead of piling onto the old width.
    let topo = sized_topo_for_pools(usable);
    let new_prefill = rebuild_prefill.then(|| build_prefill(&topo));
    let new_decode = rebuild_decode.then(|| build_decode(&topo));
    // Swap each rebuilt pool under its slot try-lock, fixed order, with no
    // early return between: a contended slot defers only its own pool, so a
    // first-use build on one slot never strands a ready swap on the other
    // (see `pool_swap_rebuilt_pools_attempts_both_despite_contention`).
    swap_rebuilt_pools(
        (&PREFILL_SLOT, new_prefill),
        (&DECODE_SLOT, new_decode),
        usable,
    )
}

/// Swap each rebuilt pool into its slot, attempting every pool even when an
/// earlier slot contends: contention defers only its own pool. Slots are
/// parameters (rather than the globals) so tests can drive one contended
/// and one free slot deterministically.
fn swap_rebuilt_pools(
    (prefill_slot, prefill_new): (&Mutex<Option<PoolSlot>>, Option<RowPool>),
    (decode_slot, decode_new): (&Mutex<Option<PoolSlot>>, Option<RowPool>),
    usable: &super::cpu_features::UsableCpus,
) -> bool {
    let mut rebuilt = false;
    if let Some(new) = prefill_new {
        rebuilt |= swap_rebuilt_pool(prefill_slot, "prefill", new, usable);
    }
    if let Some(new) = decode_new {
        rebuilt |= swap_rebuilt_pool(decode_slot, "decode", new, usable);
    }
    rebuilt
}

/// Swap one rebuilt pool into its slot. Returns whether it swapped; a
/// contended slot logs and defers this pool to the next boundary (the caller
/// still attempts the other pool). Poisoned proceeds (the last accessor
/// unwound, so nothing is mid-build).
fn swap_rebuilt_pool(
    slot: &Mutex<Option<PoolSlot>>,
    name: &str,
    new: RowPool,
    usable: &super::cpu_features::UsableCpus,
) -> bool {
    let allowed = usable.count();
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            tracing::debug!(
                "cera: {name} pool slot contended; resize to {allowed} deferred to next boundary"
            );
            return false;
        }
    };
    let Some(entry) = guard.as_mut() else {
        // Unreachable in practice (slots transition None->Some and
        // Some->Some only, and the snapshot saw Some): fail closed with an
        // honest message rather than misreporting a deferral.
        tracing::debug!("cera: {name} pool slot unbuilt at swap; first use will size fresh");
        return false;
    };
    tracing::info!(
        "cera: cpuset topology is now {usable:?} (allowance {allowed}); rebuilding {name} pool {} -> {} workers",
        entry.pool.num_threads(),
        new.num_threads(),
    );
    *entry = PoolSlot {
        pool: Arc::new(new),
        usable: usable.clone(),
    };
    true
}

/// Rebuild both pools when the usable CPU set or allowance changed since they were
/// sized (cpuset/cgroup migration, e.g. Android background to foreground, or core
/// reallocations at constant width), so workers are re-seated onto valid cores and
/// performance is never stuck at a departed cpuset's shape. Cheap no-op when
/// unchanged: one three-tier probe (cgroup filesystem reads, falling back to
/// affinity syscalls and online CPUs only when earlier tiers miss) plus set
/// identity comparisons against each slot. On change it rebuilds only idle pools;
/// a mid-dispatch pool defers to the next boundary instead of blocking it. Returns
/// true when it swapped at least one pool.
///
/// Called at generation boundaries (append/generate entry, each decode
/// token, each prefill chunk): between dispatches, never within one.
/// In-flight dispatches hold their own `Arc` and finish on the old width.
///
/// Scope is the RowPools' sizing (width plus pins). The rayon global pool
/// (vision preprocessing) is build-once and has no public resize API; the
/// raw detection stays host-wide and frozen, but each rebuild re-intersects
/// its pins with the live usable set, so a cpuset move re-seats pins as well
/// as width instead of stranding workers on departed cores.
pub(crate) fn resize_pools_for_cpuset() -> bool {
    rebuild_pools_for_allowance(&super::cpu_features::cpu_usable_cpus())
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn set_macos_thread_qos_interactive() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

/// Caller-pin claim, process-wide: at most one thread holds it (see
/// `pin_caller_once`). `OWNER` is the claiming thread's tid (`-1` = none)
/// and `CORE` the core it was pinned to. The allowance probe reads these to
/// tell our own pinning apart from an external restriction: a leader whose
/// live mask is exactly our pin carries no information (see
/// `thread_self_pinned`). Invariant: `OWNER != -1` exactly while `CLAIMED`.
/// Release order is owner/core first, `CLAIMED` last; readers treat a
/// half-published claim (`CLAIMED` set, `OWNER` still `-1`) as unclaimed,
/// which is correct — the pin has not happened yet at that instant.
/// `CLAIMED` lives on every platform (the claim protocol runs everywhere);
/// owner/core only where pins are real.
static CALLER_PIN_CLAIMED: AtomicBool = AtomicBool::new(false);
#[cfg(any(target_os = "linux", target_os = "android"))]
static CALLER_PIN_OWNER: AtomicI64 = AtomicI64::new(-1);
#[cfg(any(target_os = "linux", target_os = "android"))]
static CALLER_PIN_CORE: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Whether thread `tid` currently holds our caller-pin: it claimed the pin
/// and its live mask is still exactly that pin. A cpuset migration rewrites
/// masks wholesale, which shows up here as "not ours" and lets the allowance
/// probe trust the live mask again. Linux/Android only: the pin is a stub
/// elsewhere, so there is nothing to detect.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn thread_self_pinned(tid: libc::pid_t) -> bool {
    if CALLER_PIN_OWNER.load(Ordering::Acquire) != tid as i64 {
        return false;
    }
    let core = CALLER_PIN_CORE.load(Ordering::Acquire);
    if core == usize::MAX {
        return false;
    }
    // SAFETY: `cpu_set_t` is a plain bitmask; all-zero is a valid set.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: the size/pointer pair describes `set` exactly. Any failure
    // (notably ESRCH for an exited thread) means "not ours".
    let ok =
        unsafe { libc::sched_getaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if ok != 0 {
        return false;
    }
    // Bound spelled from the set size rather than `CPU_SETSIZE`: the
    // constant is `size_t` on Android, where `as usize` would trip
    // `unnecessary_cast` under the repo's `-D warnings` gate.
    let capacity = std::mem::size_of::<libc::cpu_set_t>() * 8;
    let mut count = 0;
    let mut found = false;
    for cpu in 0..capacity {
        // SAFETY: `cpu` is bounded by the set's own bit capacity.
        if unsafe { libc::CPU_ISSET(cpu, &set) } {
            count += 1;
            found = cpu == core;
        }
    }
    // True only for exactly our singleton: a migrated (rewritten) mask or a
    // widened one belongs to the outside world again.
    count == 1 && found
}

/// Test-only injector for the caller-pin claim statics: pretend thread
/// `owner` holds a pin to `core`. The live-mask half of detection still
/// reads the real mask, so tests pin real threads and only fake the claim.
/// Restores the unclaimed state on drop. Going through a real dispatch
/// instead would be racy: the claim is process-global, and a concurrent
/// pool test could hold it first.
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
pub(crate) fn debug_set_caller_pin_for_test(owner: i64, core: usize) -> impl Drop {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            CALLER_PIN_OWNER.store(-1, Ordering::Release);
            CALLER_PIN_CORE.store(usize::MAX, Ordering::Release);
        }
    }
    CALLER_PIN_OWNER.store(owner, Ordering::Release);
    CALLER_PIN_CORE.store(core, Ordering::Release);
    Restore
}

impl RowPool {
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
        /// Releases the claim when the holding thread exits.
        struct ClaimGuard;
        impl Drop for ClaimGuard {
            fn drop(&mut self) {
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    CALLER_PIN_OWNER.store(-1, Ordering::Release);
                    CALLER_PIN_CORE.store(usize::MAX, Ordering::Release);
                }
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
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            static QOS_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        QOS_SET.with(|q| {
            if !q.get() {
                set_macos_thread_qos_interactive();
                q.set(true);
            }
        });
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
                    // Record the claim for the allowance probe (see
                    // `thread_self_pinned`): the pinning thread's tid and
                    // core, so a leader mask that is exactly our own pin
                    // is never mistaken for an external restriction.
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    {
                        // SAFETY: `gettid` always succeeds.
                        let tid = unsafe { libc::gettid() };
                        CALLER_PIN_OWNER.store(tid as i64, Ordering::Release);
                        CALLER_PIN_CORE.store(core, Ordering::Release);
                    }
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
            idle_guard(self)
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
        // active worker, floored at the caller's `min_chunk_rows`. This is the
        // size for a worker on the fastest core; slower cores scale down from it
        // in `worker_chunk_rows`, subject to the same floor.
        let min_chunk_rows = min_chunk_rows.max(1);
        let chunk_rows = total_rows
            .div_ceil(active * STEAL_CHUNKS_PER_WORKER)
            .max(min_chunk_rows);

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
            min_chunk_rows,
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
        self.shared.next_row.store(0, Ordering::Relaxed);
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
                .store(pack_state(next_epoch, active), Ordering::SeqCst);
            if self.shared.parked_count.load(Ordering::SeqCst) > 0 {
                for h in self.workers.iter().take(active - 1) {
                    h.thread().unpark();
                }
            }
        } else {
            self.shared
                .pending
                .store(self.num_threads - 1, Ordering::Release);
            self.shared.state.fetch_add(1, Ordering::SeqCst);
            if self.shared.parked_count.load(Ordering::SeqCst) > 0 {
                for h in &self.workers {
                    h.thread().unpark();
                }
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
            steal_and_run(&self.shared, &job, 0);
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

/// RAII guard ensuring `parked_count` is decremented on scope exit,
/// preventing counter skew or thread desynchronization if a worker unwinds or panics.
struct ParkedGuard<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> ParkedGuard<'a> {
    #[inline]
    fn enter(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self { counter }
    }
}

impl Drop for ParkedGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
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
fn worker_loop(
    shared: Arc<Shared>,
    worker_id: usize,
    pin_core: Option<usize>,
    spread_mask: &'static [usize],
    spin_limit: u32,
) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        thread_local! {
            static WORKER_QOS_SET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        WORKER_QOS_SET.with(|q| {
            if !q.get() {
                set_macos_thread_qos_interactive();
                q.set(true);
            }
        });
    }

    if let Some(core) = pin_core {
        let _ = pin_current_thread_to_core(core);
    } else if !spread_mask.is_empty() {
        // Not pinned 1:1, but affinity must still be *asserted*, not inherited.
        // A spawned thread inherits its creator's mask, and `pin_caller_once`
        // confines that creator to a single core process-wide, so leaving this
        // out would put every worker of an unpinned pool on one core: the
        // failure `set_current_thread_affinity` and `ensure_rayon_global_pool`
        // both exist to prevent. Widening to the full set restores the
        // scheduler's freedom to move a worker off a stalling core, which is
        // the entire point of not pinning here.
        let _ = set_current_thread_affinity(spread_mask);
    }
    // Run this worker's chunks of `job`, catching a closure panic so the worker
    // still reaches its `pending` decrement (a dead worker would wedge the pool);
    // the dispatcher re-raises the panic on the calling thread after the drain.
    let run_job = |shared: &Shared, job: &Job| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            steal_and_run(shared, job, worker_id);
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
                let _guard = ParkedGuard::enter(&shared.parked_count);
                // Check state one last time after declaring intent to park
                if shared.state.load(Ordering::SeqCst) != last_state
                    || shared.shutdown.load(Ordering::SeqCst)
                {
                    continue;
                }
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
        if parked {
            if let Some(core) = pin_core {
                let _ = pin_current_thread_to_core(core);
            } else if !spread_mask.is_empty() {
                let _ = set_current_thread_affinity(spread_mask);
            }
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
/// nothing at all when `CERA_PIN` is off. Cached-topology shorthand over
/// [`pin_cores_in`]; `RowPool::build` takes its core list already filtered
/// through here.
///
/// This is the *full* usable set, efficiency cores included, because a widened
/// pool needs a distinct core for every worker (see `CoreTopology::pin_cores`
/// for the 35x measurement behind that). A caller that wants only the fast
/// cores wants [`perf_pinned_cores`] instead.
pub(crate) fn pinned_cores() -> &'static [usize] {
    pin_cores_in(super::cpu_features::core_topology())
}

/// [`pinned_cores`] for an explicit topology: pool (re)builds pass their
/// freshly-sized topo so pins track it. The single place the `CERA_PIN`
/// decision is made.
fn pin_cores_in(topo: &super::cpu_features::CoreTopology) -> &[usize] {
    if pinning_enabled() {
        &topo.pin_cores
    } else {
        &[]
    }
}

/// Relative speed of each core in [`pinned_cores`], same order and length, in
/// units of [`super::cpu_features::WEIGHT_FULL`]. Empty when the host has no
/// usable spread, and empty when `CERA_PIN` is off.
///
/// The `CERA_PIN` case is the load-bearing one: with pinning off a worker is
/// wherever the OS put it, so `core_weights[i]` no longer describes worker `i`
/// and sizing its chunk from that would be a guess dressed as a measurement.
/// Returning nothing puts every worker back on the uniform chunk, which is the
/// right behaviour when placement is unknown.
///
/// Test-only: pool builds take weights from their sized topo (see
/// [`pin_weights_in`]).
#[cfg(test)]
pub(crate) fn pinned_core_weights() -> &'static [u32] {
    pin_weights_in(super::cpu_features::core_topology())
}

/// [`pinned_core_weights`] for an explicit topology; see [`pin_cores_in`].
fn pin_weights_in(topo: &super::cpu_features::CoreTopology) -> &[u32] {
    if pinning_enabled() {
        &topo.core_weights
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
    use crate::backend::cpu_features::UsableCpus;

    /// Serializes the resize tests: they all mutate the global pools, so
    /// interleaved rebuilds would race each other's width assertions. Other
    /// tests only dispatch (correct at any width) and run concurrently.
    static RESIZE_TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Drive both pools to tracking `allowed`, retrying past concurrent-test
    /// dispatches that legitimately refuse an attempt. Callers ensure the
    /// pools are built first (unbuilt slots are skipped, never staged).
    /// Returns whether both pools landed there (false only under sustained
    /// contention); the panicking wrapper below is for test bodies, this one
    /// for `Drop`, which must not panic.
    ///
    /// Deadline-bounded, not count-bounded: a refused attempt costs
    /// microseconds (the gate runs before any build), so a fixed iteration
    /// count could all land inside one concurrent multi-millisecond dispatch.
    fn try_stage_usable(usable: &UsableCpus) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            rebuild_pools_for_allowance(usable);
            let tracking = [&PREFILL_SLOT, &DECODE_SLOT].iter().all(|slot| {
                slot.lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .as_ref()
                    .is_some_and(|s| &s.usable == usable)
            });
            if tracking {
                return true;
            }
            if std::time::Instant::now() > deadline {
                return false;
            }
            std::thread::yield_now();
        }
    }

    fn stage_usable(usable: &UsableCpus) {
        assert!(
            try_stage_usable(usable),
            "staging usable {usable:?} never landed"
        );
    }

    fn stage_allowance(allowed: usize) {
        stage_usable(&UsableCpus::CountOnly(allowed));
    }

    /// Staged/target ends for an explicit-allowance excursion. The target
    /// stays distinct from ambient, flipping the ends on 1-CPU hosts; the
    /// staged end may equal ambient there (or when ambient is unknown),
    /// where staging is a harmless no-op. A concurrent ambient resize
    /// pre-landing the target would make `true` unattainable (the staged
    /// pool already tracks, the held pool refuses), spinning the test to
    /// its deadline; with the target distinct, ambient resizes only pull
    /// away, which retries absorb.
    fn flip_ends(ambient: usize) -> (usize, usize) {
        if ambient == 1 {
            (1, usize::MAX)
        } else {
            (usize::MAX, 1)
        }
    }

    /// `flip_ends` keeps the target off-ambient on every host shape. Direct
    /// mapping pin: no host in CI exercises the 1-CPU leg, so deleting the
    /// branch would regress those runners (target pre-landed, `true`
    /// unattainable) with a green suite.
    #[test]
    fn flip_ends_pins_off_ambient_mapping() {
        assert_eq!(flip_ends(1), (1, usize::MAX));
        assert_eq!(flip_ends(8), (usize::MAX, 1));
        assert_eq!(flip_ends(usize::MAX), (usize::MAX, 1));
        for ambient in [1usize, 2, 8, usize::MAX] {
            let (staged, target) = flip_ends(ambient);
            assert_ne!(target, ambient, "target must stay off-ambient");
            assert_ne!(staged, target, "ends must differ");
        }
    }

    /// Restores the ambient allowance on exit, including on panic, so later
    /// tests inherit normally-sized pools. Best-effort retry: a single shot
    /// can lose to a concurrent test dispatch; correctness never depends on
    /// width.
    struct RestoreAmbient(UsableCpus);
    impl Drop for RestoreAmbient {
        fn drop(&mut self) {
            try_stage_usable(&self.0);
        }
    }

    /// Expected `(prefill, decode)` widths for an allowance, through the same
    /// sizing path the rebuild uses (frozen detection, fresh policy, current
    /// env).
    fn expect_widths(allowed: usize) -> (usize, usize) {
        let topo = sized_topo_for_pools(&UsableCpus::CountOnly(allowed));
        (
            crate::backend::calibrate::prefill_thread_count(&topo),
            crate::backend::calibrate::decode_thread_count(&topo),
        )
    }

    fn live_widths() -> (usize, usize) {
        (
            RowPool::prefill().num_threads(),
            RowPool::decode().num_threads(),
        )
    }

    /// Stage `allowed` and verify the exact widths, folding the pins into a
    /// deadline loop: session/engine tests drive ambient resizes concurrently
    /// in this binary (outside the serial lock), so a bare assert after
    /// staging can catch a slipped-in ambient width and flake.
    fn stage_and_verify(allowed: usize) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            stage_allowance(allowed);
            if live_widths() == expect_widths(allowed) {
                return;
            }
            assert!(
                std::time::Instant::now() <= deadline,
                "widths never settled at allowance {allowed}"
            );
            std::thread::yield_now();
        }
    }

    /// Resize tracks an explicit allowance: staging one end and landing the
    /// other reaches the calibrate-predicted widths for each, and the
    /// landing rebuild reports true. Uses the explicit-allowance seam so the
    /// test is deterministic (no cpuset mutation); restores the ambient
    /// allowance on exit, including on panic, so later tests inherit
    /// normally-sized pools.
    ///
    /// Degenerate-host caveat: on a 1-CPU runner the narrow leg equals the
    /// wide leg, so only the exact-width assertions (not the change itself)
    /// carry weight there.
    #[test]
    fn pool_resize_tracks_explicit_allowance() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let ambient_count = ambient.count();
        if ambient_count == usize::MAX {
            eprintln!("skip: process allowance unknown, cannot stage a change");
            return;
        }
        let _restore = RestoreAmbient(ambient);
        let _ = RowPool::prefill();
        let _ = RowPool::decode();
        // Drive a change in both directions (ends flip on 1-CPU hosts;
        // see `flip_ends`).
        let (staged, target) = flip_ends(ambient_count);
        stage_and_verify(staged);
        // The landing rebuild reports true (retried past concurrent
        // dispatches, like the staging above; deadline-bounded for the same
        // reason).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut landed = false;
        loop {
            if rebuild_pools_for_allowance(&UsableCpus::CountOnly(target)) {
                landed = true;
                break;
            }
            if std::time::Instant::now() > deadline {
                break;
            }
            std::thread::yield_now();
        }
        assert!(landed);
        // Per-pool gating means the landing rebuild may have swapped only
        // the idle pool; converge the other before asserting widths.
        stage_and_verify(target);
        stage_and_verify(ambient_count);
    }

    /// A mid-dispatch pool defers resize while the idle pool still swaps:
    /// with the decode dispatch lock held, rebuild swaps prefill, reports
    /// true, and leaves the decode width untouched. Holds the lock directly
    /// (same module); `std` mutexes are non-reentrant so the rebuild's
    /// `try_lock` deterministically fails on this thread. The lock is
    /// acquire-then-verified (`Arc::ptr_eq` against the live slot): a
    /// concurrent ambient resize slipping between fetch and lock would
    /// otherwise hand this test a stale pool's guard while its own rebuild
    /// swaps the live pool. Width pins fold into the landing loop (a
    /// concurrent resize can slip between a true report and the width read);
    /// deadline-bounded like the staging above. Restores ambient on exit.
    #[test]
    fn pool_resize_skips_when_busy() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        // `_restore` drops at fn scope end, after any pin-loop lock
        // release, then re-stages ambient.
        let _restore = RestoreAmbient(ambient.clone());
        let _ = RowPool::decode();
        let _ = RowPool::prefill();
        // Stage a tracked allowance that differs from the rebuild target so
        // the busy gate (not the no-op leg) is what refuses decode below.
        let (staged, target) = flip_ends(ambient.count());
        stage_allowance(staged);
        let expect_pre = expect_widths(target).0;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        // Pin the CURRENT decode pool for the whole attempt: fetch, lock,
        // verify the slot still holds it, then drive the landing loop with
        // both alive (the guard borrows the pool, so the loop lives in this
        // scope). A concurrent ambient resize between fetch and lock hands
        // us a stale guard, in which case retry.
        let mut saw_true = false;
        let landed = 'pin: loop {
            let decode = RowPool::decode();
            let _held = decode
                .dispatch_lock
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if !Arc::ptr_eq(&decode, &RowPool::decode()) {
                drop(_held);
                assert!(
                    std::time::Instant::now() <= deadline,
                    "decode pool never stable enough to pin"
                );
                std::thread::yield_now();
                continue 'pin;
            }
            let w_dec = decode.num_threads();
            loop {
                if rebuild_pools_for_allowance(&UsableCpus::CountOnly(target)) {
                    saw_true = true;
                }
                // Break only on an observed swap plus the full post-state:
                // prefill swapped, decode untouched. The widths alone
                // cannot gate the break: under a width-dominating override
                // (`CERA_THREADS`) they match before any swap is reported.
                if saw_true
                    && RowPool::prefill().num_threads() == expect_pre
                    && RowPool::decode().num_threads() == w_dec
                {
                    break 'pin true;
                }
                if std::time::Instant::now() > deadline {
                    break 'pin false;
                }
                std::thread::yield_now();
            }
        };
        assert!(saw_true, "rebuild never reported a swap");
        assert!(landed, "prefill never swapped while decode was held");
    }

    /// Need-stage deferral is per-pool: with the prefill slot lock held
    /// (mid-build), a rebuild needing both pools still swaps decode and
    /// reports true. Holds the slot directly (same module). Prefill cannot
    /// have swapped (every swap path takes the held slot lock), so only the
    /// decode width is asserted; concurrent ambient resizes cannot touch
    /// prefill either, for the same reason. (This pins the need snapshot,
    /// not the swap: the held slot reports `Deferred` before any swap runs.
    /// The swap helper's own contract is pinned below.)
    #[test]
    fn pool_resize_deferred_slot_does_not_block_other_pool() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let _restore = RestoreAmbient(ambient.clone());
        let _ = RowPool::decode();
        let _ = RowPool::prefill();
        let (staged, target) = flip_ends(ambient.count());
        stage_allowance(staged);
        let expect_dec = expect_widths(target).1;
        let _slot_held = PREFILL_SLOT.lock().unwrap_or_else(|p| p.into_inner());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut saw_true = false;
        let mut landed = false;
        loop {
            if rebuild_pools_for_allowance(&UsableCpus::CountOnly(target)) {
                saw_true = true;
            }
            // Gate the break on an observed swap, not the width alone:
            // under a width-dominating override (`CERA_THREADS`) the width
            // matches before any swap is reported.
            if saw_true && RowPool::decode().num_threads() == expect_dec {
                landed = true;
                break;
            }
            if std::time::Instant::now() > deadline {
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            saw_true,
            "rebuild never reported a swap with prefill slot held"
        );
        assert!(landed, "decode never swapped with prefill slot held");
    }

    /// The swap helper refuses a contended slot and swaps a free one. Direct
    /// pin of the per-pool swap contract; the no-early-return wiring across
    /// pools is pinned by
    /// `pool_swap_rebuilt_pools_attempts_both_despite_contention`. The
    /// decode leg loops: a concurrent accessor's brief slot hold can refuse
    /// one attempt, and a concurrent ambient resize can slip between the
    /// swap and the width read.
    #[test]
    fn pool_swap_helper_defers_only_contended_slot() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let _restore = RestoreAmbient(ambient);
        let _ = RowPool::decode();
        let _ = RowPool::prefill();
        stage_allowance(usize::MAX);
        let expect_dec = expect_widths(1).1;
        let _slot_held = PREFILL_SLOT.lock().unwrap_or_else(|p| p.into_inner());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let topo = sized_topo_for_pools(&UsableCpus::CountOnly(1));
            // Prefill deterministically refuses: we hold its slot.
            assert!(!swap_rebuilt_pool(
                &PREFILL_SLOT,
                "prefill",
                build_prefill(&topo),
                &UsableCpus::CountOnly(1),
            ));
            if swap_rebuilt_pool(
                &DECODE_SLOT,
                "decode",
                build_decode(&topo),
                &UsableCpus::CountOnly(1),
            ) && RowPool::decode().num_threads() == expect_dec
            {
                break;
            }
            assert!(
                std::time::Instant::now() <= deadline,
                "decode swap never landed with prefill slot held"
            );
            std::thread::yield_now();
        }
    }

    /// The multi-pool swap attempts every pool even when an earlier slot
    /// contends: with the prefill slot held, the decode swap still lands
    /// and the call reports true, while prefill is untouched. Local slots
    /// keep this deterministic with no serial lock: the globals path cannot
    /// stage this state, since a held slot reports `Deferred` before any
    /// swap runs.
    #[test]
    fn pool_swap_rebuilt_pools_attempts_both_despite_contention() {
        fn lock(slot: &Mutex<Option<PoolSlot>>) -> std::sync::MutexGuard<'_, Option<PoolSlot>> {
            slot.lock().unwrap_or_else(|p| p.into_inner())
        }
        let fresh = sized_topo_for_pools(&UsableCpus::CountOnly(usize::MAX));
        let prefill_slot = Mutex::new(Some(PoolSlot {
            pool: Arc::new(build_prefill(&fresh)),
            usable: UsableCpus::CountOnly(usize::MAX),
        }));
        let decode_slot = Mutex::new(Some(PoolSlot {
            pool: Arc::new(build_decode(&fresh)),
            usable: UsableCpus::CountOnly(usize::MAX),
        }));
        let _held = lock(&prefill_slot);
        let topo = sized_topo_for_pools(&UsableCpus::CountOnly(1));
        let rebuilt = swap_rebuilt_pools(
            (&prefill_slot, Some(build_prefill(&topo))),
            (&decode_slot, Some(build_decode(&topo))),
            &UsableCpus::CountOnly(1),
        );
        assert!(rebuilt, "contended prefill vetoed the decode swap");
        drop(_held);
        assert_eq!(
            lock(&prefill_slot)
                .as_ref()
                .expect("prefill slot built")
                .usable,
            UsableCpus::CountOnly(usize::MAX),
            "contended pool must be untouched"
        );
        let decode = lock(&decode_slot);
        let entry = decode.as_ref().expect("decode slot built");
        assert_eq!(entry.usable, UsableCpus::CountOnly(1));
        assert_eq!(entry.pool.num_threads(), expect_widths(1).1);
    }

    /// The per-chunk resize hook in `Model::forward_prefill_chunked`
    /// converges the pools: staging off-ambient, then driving chunks until
    /// both slots track the ambient allowance. A stub `Model` (only
    /// `forward` + `config` are required)
    /// keeps this free of weights. Pins the tracked allowance, not widths
    /// (exact under any env override). Kill note: with the hook deleted
    /// this fails deterministically single-threaded; under parallel load a
    /// concurrent ambient resize can also converge the slots, so
    /// demonstrate the kill with `--test-threads=1`.
    #[test]
    fn chunked_prefill_per_chunk_hook_converges_pools() {
        use crate::kv_cache::InferenceState;
        use crate::model::{Model, ModelConfig, ScalarMultipliers};
        use std::sync::atomic::AtomicBool;

        struct StubModel {
            config: ModelConfig,
        }
        impl Model for StubModel {
            fn forward(
                &self,
                _tokens: &[u32],
                _pos: usize,
                _state: &mut InferenceState,
            ) -> Vec<f32> {
                Vec::new()
            }
            fn config(&self) -> &ModelConfig {
                &self.config
            }
        }

        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let ambient_count = ambient.count();
        let _restore = RestoreAmbient(ambient.clone());
        let _ = RowPool::prefill();
        let _ = RowPool::decode();
        // Stage the off-ambient end so the hook has work to do.
        let (_, off_ambient) = flip_ends(ambient_count);
        stage_allowance(off_ambient);
        let config = ModelConfig {
            architecture: "stub".into(),
            n_layers: 1,
            hidden_size: 8,
            intermediate_size: 8,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            vocab_size: 16,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            block_types: Vec::new(),
            conv_kernel_size: None,
            ssm: None,
            kv_heads_per_layer: vec![2],
            scalars: ScalarMultipliers::default(),
            moe: None,
            is_causal: true,
            class_labels: Vec::new(),
        };
        let model = StubModel { config };
        let mut state =
            InferenceState::for_prefill(&model.config, 4).expect("scratch state builds");
        let cancel = AtomicBool::new(false);
        let tracks_ambient = || {
            [&PREFILL_SLOT, &DECODE_SLOT].iter().all(|slot| {
                slot.lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .as_ref()
                    .expect("pool built")
                    .usable
                    == ambient
            })
        };
        // Drive chunks until converged: a concurrent dispatch can defer one
        // pool on any single pass, and per-pool gating converges each pool
        // independently across passes. Deadline-bounded like the staging.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let (consumed, _) =
                model.forward_prefill_chunked(&[1, 2, 3, 4], 0, &mut state, 2, &cancel);
            assert_eq!(consumed, 4, "both chunks must run through the hook");
            if tracks_ambient() {
                break;
            }
            assert!(
                std::time::Instant::now() <= deadline,
                "chunked prefill did not converge pools to ambient"
            );
            std::thread::yield_now();
        }
    }

    /// A poisoned dispatch lock (a dispatcher panicked) must not wedge
    /// resize: the gate proceeds through poison, the swap heals the slot
    /// with a fresh lock, and the new pool dispatches. Staging ambient
    /// first makes concurrent ambient resizes no-ops, so nothing swaps
    /// between the poison and the rebuild gating it: that is what makes
    /// the heal deterministic rather than racy.
    #[test]
    fn pool_resize_heals_poisoned_dispatch_lock() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let _restore = RestoreAmbient(ambient.clone());
        let _ = RowPool::prefill();
        let _ = RowPool::decode();
        stage_usable(&ambient);
        let pool = RowPool::prefill();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = pool.dispatch_lock.lock().unwrap_or_else(|p| p.into_inner());
            panic!("intentional poison probe");
        }));
        assert!(outcome.is_err(), "poison probe did not panic");
        assert!(pool.dispatch_lock.is_poisoned());
        // Poison is not business: the gate proceeds through it. Retried: a
        // concurrent dispatch holds the lock transiently, and `try_lock`
        // reports `WouldBlock` on a held lock even when poisoned.
        let gate_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if idle_guard(&pool).is_some() {
                break;
            }
            assert!(
                std::time::Instant::now() <= gate_deadline,
                "prefill dispatch lock never idle"
            );
            std::thread::yield_now();
        }
        // ...and the rebuild heals the slot with a fresh lock. The target
        // flips on 1-CPU hosts (see `flip_ends`) so the heal is a real swap
        // there too.
        let (_, target) = flip_ends(ambient.count());
        stage_and_verify(target);
        assert!(!RowPool::prefill().dispatch_lock.is_poisoned());
        // The healed pool dispatches correctly.
        let mut y = vec![0.0f32; 64];
        RowPool::prefill().dispatch_rows(&mut y, 1, 1, |row, slice| {
            slice[0] = row as f32;
        });
        for (row, &v) in y.iter().enumerate() {
            assert_eq!(v, row as f32);
        }
    }

    /// With pools tracking the ambient allowance, the probed entry is a
    /// no-op: no rebuild, same widths.
    #[test]
    fn pool_resize_noop_when_allowance_unchanged() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let _restore = RestoreAmbient(ambient.clone());
        let _ = RowPool::prefill();
        let _ = RowPool::decode();
        stage_usable(&ambient);
        let before = (
            RowPool::prefill().num_threads(),
            RowPool::decode().num_threads(),
        );
        assert!(!resize_pools_for_cpuset());
        let after = (
            RowPool::prefill().num_threads(),
            RowPool::decode().num_threads(),
        );
        assert_eq!(before, after);
    }

    /// A cpuset migration that moves to a different set of cores with the same
    /// count must trigger a pool rebuild so workers re-seat onto the new cores.
    #[test]
    fn pool_resize_rebuilds_on_cpuset_migration_at_constant_width() {
        let _serial = RESIZE_TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let ambient = crate::backend::cpu_features::cpu_usable_cpus();
        let _restore = RestoreAmbient(ambient);
        let _ = RowPool::prefill();
        let _ = RowPool::decode();
        let set1 = UsableCpus::Known(vec![0, 1, 2, 3]);
        let set2 = UsableCpus::Known(vec![4, 5, 6, 7]);
        stage_usable(&set1);
        stage_usable(&set2);
        assert!(!rebuild_pools_for_allowance(&set2));
    }

    #[test]
    fn dispatch_matches_serial_across_shapes_and_threads() {
        // Fill each row with a function of its absolute index; compare pool
        // output to the serial reference for several sizes, `n`, and pool sizes,
        // under both barrier modes (active-sized = prefill, full = decode).
        for &active_barrier in &[true, false] {
            for &num_threads in &[1usize, 2, 4, 7] {
                let pool = RowPool::build(num_threads, &[], &[], &[], active_barrier);
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
        let pool = RowPool::build(4, &[], &[], &[], true);
        let total_rows = 10_000usize;
        let mut counts = vec![0.0f32; total_rows];
        pool.dispatch_rows(&mut counts, 1, 1, |_row, slice| {
            slice[0] += 1.0;
        });
        assert!(counts.iter().all(|&c| c == 1.0));
    }

    /// The disjoint-partition guarantee has to survive workers claiming
    /// *different* amounts. It used to hold trivially (a claim was a chunk
    /// index times one shared size) and now rests on each claim advancing the
    /// row cursor by exactly what it reserves. Run under Miri this is also the
    /// data-race check for the mixed-size handoff.
    #[test]
    fn weighted_workers_still_partition_rows_exactly_once() {
        // A Tensor-G5-shaped spread (prime, two mid, two little) plus a couple
        // of exact-power-of-two ratios, so the scaling is exercised where it
        // divides evenly and where it does not.
        for weights in [
            vec![256u32, 206, 206, 52, 52],
            vec![256, 128, 64, 32],
            vec![256, 1, 256, 1],
        ] {
            for &active_barrier in &[true, false] {
                let pool = RowPool::build(weights.len(), &[], &weights, &[], active_barrier);
                // Sizes chosen so the weighted chunks actually differ: below
                // roughly `MIN_CHUNK_ROWS * active * STEAL_CHUNKS_PER_WORKER`
                // every weight collapses onto the floor and the test would
                // silently re-run the uniform-partition case. The small sizes
                // are kept deliberately, to cover that collapse too.
                for &total_rows in &[1usize, 7, 256, 4_096, 10_000, 65_537] {
                    let mut counts = vec![0.0f32; total_rows];
                    pool.dispatch_rows(&mut counts, 1, 1, |_row, slice| {
                        slice[0] += 1.0;
                    });
                    assert!(
                        counts.iter().all(|&c| c == 1.0),
                        "rows lost or written twice: weights={weights:?} \
                         barrier={active_barrier} rows={total_rows}"
                    );
                }
            }
        }
    }

    /// The proportionality itself: a worker's chunk is its weight's share of
    /// the dispatcher's chunk. This is the whole behavior of the change, and
    /// nothing else pins it. The partition test cannot: disjointness holds for
    /// *any* chunk size, so a scale that is silently half or double what it
    /// should be still passes it while shipping a different schedule.
    #[test]
    fn chunk_scales_in_proportion_to_weight() {
        let full = super::super::cpu_features::WEIGHT_FULL;
        // A Tensor G5's three tiers: prime 1024, A725 824, A520 207, rescaled.
        let shared = Shared {
            active_barrier: true,
            state: AtomicU64::new(0),
            next_row: AtomicUsize::new(0),
            worker_weights: vec![full, 206, 52],
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            parked_count: AtomicUsize::new(0),
            panic_payload: Mutex::new(None),
            job: UnsafeCell::new(None),
        };
        let job = Job {
            y_ptr: std::ptr::null_mut(),
            n: 1,
            // Not a multiple of `WEIGHT_FULL`, so both scaled results truncate
            // rather than dividing evenly.
            chunk_rows: 500,
            // Below every expected result, so the floor cannot mask the scale.
            min_chunk_rows: 1,
            total_rows: 1 << 20,
            active: 3,
            closure: std::ptr::null(),
            run: trampoline::<fn(usize, &mut [f32])>,
        };
        assert_eq!(
            worker_chunk_rows(&shared, &job, 0),
            500,
            "prime core scaled"
        );
        // 500 * 206 / 256 = 402.34, truncated.
        assert_eq!(worker_chunk_rows(&shared, &job, 1), 402, "A725 scale wrong");
        // 500 * 52 / 256 = 101.56, truncated: the ~1/5 that keeps an A520 from
        // holding the barrier with a full-size chunk.
        assert_eq!(worker_chunk_rows(&shared, &job, 2), 101, "A520 scale wrong");
    }

    /// Weighting may only ever scale a chunk *down* from the dispatcher's size,
    /// never to zero: a zero-row claim leaves the cursor where it was and spins
    /// the steal loop forever. The floor is checked directly here because the
    /// dispatch test above would hang rather than fail if it broke.
    #[test]
    fn weight_scaling_floors_at_one_row() {
        let shared = Shared {
            active_barrier: true,
            state: AtomicU64::new(0),
            next_row: AtomicUsize::new(0),
            worker_weights: vec![super::super::cpu_features::WEIGHT_FULL, 1],
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            parked_count: AtomicUsize::new(0),
            panic_payload: Mutex::new(None),
            job: UnsafeCell::new(None),
        };
        let job = Job {
            y_ptr: std::ptr::null_mut(),
            n: 1,
            chunk_rows: 4,
            // Deliberately 0, which the dispatcher never produces: with any
            // floor of its own the `.max(job.min_chunk_rows)` would return a
            // non-zero size on its own and the trailing `.max(1)` would be
            // untestable. The guard exists precisely so `worker_chunk_rows`
            // does not depend on the dispatcher having floored anything.
            min_chunk_rows: 0,
            total_rows: 1024,
            active: 2,
            closure: std::ptr::null(),
            run: trampoline::<fn(usize, &mut [f32])>,
        };
        // Full weight ⇒ the dispatcher's size, untouched.
        assert_eq!(worker_chunk_rows(&shared, &job, 0), 4);
        // 4 * 1 / 256 truncates to 0; the floor has to lift it.
        assert_eq!(worker_chunk_rows(&shared, &job, 1), 1);
        // A worker with no weight at all (unpinned surplus) gets the full size.
        assert_eq!(worker_chunk_rows(&shared, &job, 7), 4);

        // The dispatcher's own floor is the other half of the guarantee
        // `MIN_CHUNK_ROWS` documents: weighting may shrink a slow core's chunk,
        // but not below the size where the claim costs more than the work.
        // Scaling alone would give 256 * 1 / 256 = 1 row here.
        let floored = Job {
            chunk_rows: 256,
            min_chunk_rows: 128,
            ..job
        };
        assert_eq!(worker_chunk_rows(&shared, &floored, 0), 256);
        assert_eq!(
            worker_chunk_rows(&shared, &floored, 1),
            128,
            "weighting scaled past the dispatcher's floor"
        );
    }

    /// An all-full weight list describes a host with no spread, so the pool
    /// drops it and every worker takes the uniform path. This is what keeps the
    /// homogeneous case free of per-dispatch lookup work rather than merely
    /// arriving at the same answer.
    #[test]
    fn uniform_weights_are_dropped_at_build() {
        let full = super::super::cpu_features::WEIGHT_FULL;
        let uniform = RowPool::build(4, &[], &[full, full, full, full], &[], true);
        assert!(uniform.shared.worker_weights.is_empty());
        let spread = RowPool::build(4, &[], &[full, full, full, 52], &[], true);
        assert_eq!(spread.shared.worker_weights.len(), 4);
        // Weights past the pool width belong to cores no worker sits on.
        let narrow = RowPool::build(2, &[], &[full, 52, 52, 52], &[], true);
        assert_eq!(narrow.shared.worker_weights, vec![full, 52]);
    }

    /// The junction between the two halves of this feature: `cpu_features`
    /// detects weights and `worker_chunk_rows` consumes them, both tested, but
    /// nothing asserted that the *shipping* pools actually carry them from one
    /// to the other. Wiring `prefill()` to `&[]` would leave every other test
    /// green while silently restoring unweighted chunks.
    ///
    /// Asserts the invariant rather than a specific weight vector, because what
    /// is true depends on the host: with weights the pool must hold exactly one
    /// per worker, and without them (homogeneous host, no affinity, or
    /// `CERA_PIN=0`) it must hold none.
    ///
    /// **Host-dependent by nature, and only a real guard where there is a
    /// spread.** On a homogeneous host or one without affinity there are no
    /// weights to lose, so severing the wiring is not observable and this test
    /// passes either way (verified: mutating `prefill()` to pass `&[]` does not
    /// fail it on an Apple Silicon host). That is the correct instrument rather
    /// than a weak one, since the mutation is only a defect where the weights
    /// exist, but it does mean CI on a homogeneous runner does not cover this
    /// and a big.LITTLE device is what actually exercises it.
    #[test]
    fn shipping_pools_carry_the_detected_weights() {
        let expected = pinned_core_weights();
        // The prefill pool deliberately carries no weights when it is wider
        // than the fast cores: it is unpinned there, so no weight describes any
        // worker. Checking it in that state would assert the opposite of the
        // intended behaviour, so it is skipped.
        //
        // Read from the pool's recorded placement, not re-derived from its
        // width: the policy is decided on the *requested* width while
        // `num_threads()` reports the width actually achieved, and a spawn
        // failure can move one without the other.
        let pools: Vec<Arc<RowPool>> = [RowPool::prefill(), RowPool::decode()]
            .into_iter()
            .filter(|p| p.workers_pinned || pinned_core_weights().is_empty())
            .collect();
        for pool in pools {
            let got = &pool.shared.worker_weights;
            if got.is_empty() {
                // Dropped either because the host has no spread at all, or
                // because every worker in this pool's width is full-weight.
                assert!(
                    expected.is_empty()
                        || expected
                            .iter()
                            .take(pool.num_threads())
                            .all(|&w| w == super::super::cpu_features::WEIGHT_FULL),
                    "pool dropped weights that would have scaled a worker down: {expected:?}"
                );
            } else {
                assert_eq!(
                    got.len(),
                    expected.len().min(pool.num_threads()),
                    "pool weights are not one per worker"
                );
                assert_eq!(
                    got[..],
                    expected[..got.len()],
                    "pool weights do not match the detected topology"
                );
            }
        }
    }

    /// The prefill pool drops pinning exactly when its width outgrows the fast
    /// cores, and not before. Both directions matter: pinning is a 12% win at
    /// the default width and a 27% loss once the pool reaches the E-cores, so a
    /// policy that got the threshold wrong would regress one of the two.
    ///
    /// Asserts the predicate rather than the built pool, because `prefill()`
    /// width follows the live cpuset allowance and cannot be set directly
    /// within a test run.
    #[test]
    fn prefill_drops_pinning_only_once_wider_than_the_fast_cores() {
        // Drives the production predicate, so a change to it fails this test.
        let widened = |fast: usize, n: usize| !prefill_should_pin(fast, n);

        // Tensor G5: 6 fast + 2 A520.
        assert!(!widened(6, 6), "default width must keep pinning");
        assert!(
            !widened(6, 4),
            "narrower than the fast set must keep pinning"
        );
        assert!(
            widened(6, 7),
            "one worker past the fast set already reaches an E-core"
        );
        assert!(widened(6, 8), "CERA_THREADS=8 must drop pinning");

        // A host with no detected fast set (macOS, homogeneous desktop) has no
        // E-cores to fall onto, so the policy must not fire: there `pin_cores`
        // is empty anyway and dropping it would be a no-op dressed as a
        // decision.
        assert!(
            !widened(0, 32),
            "policy fired on a host with no fast-core split"
        );
    }

    #[test]
    fn repeated_dispatches_reuse_workers() {
        // Same pool, many dispatches — exercises the epoch/park/unpark cycle and
        // confirms the workers stay correct across rounds.
        let pool = RowPool::build(4, &[], &[], &[], true);
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
                let pool = RowPool::build(num_threads, &[], &[], &[], active_barrier);
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
        let pool = RowPool::build(8, &[], &[], &[], true);
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
        let pool = RowPool::build(1, &[], &[], &[], true);
        assert_eq!(pool.num_threads(), 1);
        let mut y = vec![0.0f32; 100];
        pool.dispatch_rows(&mut y, 1, 1, |row, slice| slice[0] = row as f32);
        assert!(y.iter().enumerate().all(|(i, &v)| v == i as f32));
    }

    #[test]
    fn trailing_partial_row_matches_serial_chunks() {
        // y.len() % n != 0: the tail must be visited with the short slice,
        // exactly like the serial `chunks_mut(n)` fallback.
        let pool = RowPool::build(4, &[], &[], &[], true);
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
        let pool = RowPool::build(4, &[], &[], &[], true);
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

    /// The fan-out counters exist to make a silent serial fallback visible, so a
    /// counter that never increments, or a snapshot that reports an impossible
    /// pair, fails open: the readout says the pool is healthy and the operator
    /// believes it.
    ///
    /// Drives real contended dispatches rather than poking the atomics, so it
    /// covers the increments in `dispatch_body` and the read in `snapshot`
    /// together. The counters are process-global and other tests dispatch
    /// concurrently, hence deltas via `since` and inequalities rather than
    /// exact totals.
    #[test]
    fn fanout_counters_track_dispatches_and_stay_coherent() {
        let before = stats::snapshot();
        let pool = RowPool::build(4, &[], &[], &[], true);
        // Two threads on one pool: whoever loses `try_lock` is a serial
        // fallback, which is the event these counters exist to catch.
        for _ in 0..20 {
            let mut a = vec![0.0f32; 4096];
            let mut b = vec![0.0f32; 4096];
            thread::scope(|s| {
                let pool = &pool;
                s.spawn(|| pool.dispatch_rows(&mut a, 1, 1, |row, s| s[0] = row as f32 + 1.0));
                pool.dispatch_rows(&mut b, 1, 1, |row, s| s[0] = row as f32 + 2.0);
            });
        }
        let delta = stats::snapshot().since(&before);
        assert!(
            delta.fanout_dispatches >= 40,
            "40 multi-worker dispatches counted as {}",
            delta.fanout_dispatches
        );
        // The invariant `snapshot` enforces: a subset can never exceed its set.
        // Read from a live snapshot, not a hand-built one, so a regression in
        // the load order or the clamp shows up here.
        let live = stats::snapshot();
        assert!(
            live.serial_fallbacks <= live.fanout_dispatches,
            "serial {} > fanout {}",
            live.serial_fallbacks,
            live.fanout_dispatches
        );
        assert!(live.serial_macs <= live.fanout_macs);
        let f = live.serial_mac_fraction();
        assert!((0.0..=1.0).contains(&f), "fraction {f} outside 0.0..=1.0");
    }

    #[test]
    fn worker_panic_propagates_and_pool_survives() {
        let pool = RowPool::build(4, &[], &[], &[], true);
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

//! Persistent, affinity-pinned, spin-wait thread pool for the row-parallel
//! decode hot path.
//!
//! ## Why this exists (vs. rayon)
//!
//! Single-token decode issues ~50–100 GEMVs (one per projection per layer, plus
//! the vocab output). Above [`super::cpu::gemv_par_threshold`] each parallelizes
//! through [`super::cpu::par_rows`] / [`super::cpu::par_rows_n`]. With rayon that
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
//! type-erased closure + monomorphized trampoline), bumps `epoch` (Release),
//! then joins the steal loop as worker 0. Background workers observe the new
//! `epoch` (Acquire) and, alongside the caller, **claim contiguous row chunks**
//! from a shared `next_chunk` atomic until the row space is exhausted — dynamic
//! work-stealing, so faster cores grab more chunks and every worker reaches the
//! barrier together (heterogeneous big.LITTLE load balancing). Each worker then
//! `fetch_sub`s a `pending` counter; the caller spins on `pending == 0`
//! (Acquire) before returning. That Release/Acquire pair both (a) publishes the
//! `Job` to workers and (b) establishes happens-before for every worker's writes
//! to the output, so the disjoint `&mut` handoff is sound and the caller may read
//! the output once it returns. (Each chunk index is claimed by exactly one
//! worker, so chunks are disjoint row ranges.)
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
//! to the detected performance cores (the caller to the fastest one —
//! unpinned, a big.LITTLE scheduler can strand it on an efficiency core where
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
    /// Bumped once per dispatch; workers wake when it changes.
    epoch: AtomicU64,
    /// Next unclaimed chunk index. Active workers (and the caller) claim chunks
    /// via `fetch_add`; reset to 0 before each dispatch's `epoch` bump.
    next_chunk: AtomicUsize,
    /// Active background workers still running the current job.
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

/// Whether affinity pinning is enabled (`CERA_PIN=0`/`false`/`off`,
/// case-insensitive, disables it — for host apps that manage thread placement
/// and don't want cera's permanent caller pin). Resolved once. `pub(crate)` so
/// calibration can skip the sweep when its pools would run unpinned.
pub(crate) fn pinning_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("CERA_PIN")
            .map(|v| {
                let v = v.trim();
                ["0", "false", "off"]
                    .iter()
                    .any(|d| v.eq_ignore_ascii_case(d))
            })
            .unwrap_or(false)
    })
}

impl RowPool {
    /// Full-width pool for compute-bound batched work — prefill GEMM
    /// ([`super::cpu::par_rows_n`]). Sized to all performance cores; batched
    /// matmul is compute-bound and scales with threads. Lazily built so
    /// `CeraEngine` consumers get it without calling
    /// [`super::cpu::configure_thread_pool`].
    pub fn prefill() -> &'static RowPool {
        static POOL: OnceLock<RowPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let topo = super::cpu_features::core_topology();
            RowPool::build(topo.perf_core_count, &topo.pin_cores)
        })
    }

    /// Narrow pool for memory-bound per-token work — decode GEMV
    /// ([`super::cpu::par_rows`]). Single-token decode is memory-bandwidth
    /// bound, so its thread sweet spot is where the SoC memory bus saturates —
    /// a per-device property no topology query exposes. The width therefore
    /// comes from `super::calibrate::decode_thread_count`: a measured,
    /// per-device-cached bandwidth calibration where workers can be pinned
    /// (a capped perf-core count elsewhere), overridable with
    /// `CERA_DECODE_THREADS=<n>` (`=auto` forces recalibration).
    pub fn decode() -> &'static RowPool {
        static POOL: OnceLock<RowPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let topo = super::cpu_features::core_topology();
            let n = super::calibrate::decode_thread_count(topo);
            RowPool::build(n, &topo.pin_cores)
        })
    }

    /// Build a pool with `num_threads` total workers, pinning worker `i` to
    /// `pin_cores[i]` when present (surplus workers run unpinned). `pin_cores`
    /// empty ⇒ no pinning (macOS/desktop). Spawn failures degrade the thread
    /// count rather than panicking. The pool pins *and claims the process-wide
    /// caller pin for* whatever thread first dispatches.
    fn build(num_threads: usize, pin_cores: &[usize]) -> RowPool {
        let num_threads = num_threads.max(1);
        // Spin iterations before an idle worker parks. `CERA_SPIN` overrides the
        // default for tuning the spin-vs-park trade-off on a given device.
        let spin_limit = std::env::var("CERA_SPIN")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(SPIN_BEFORE_PARK);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            next_chunk: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
            job: UnsafeCell::new(None),
        });

        let pin_cores: &[usize] = if pinning_enabled() { pin_cores } else { &[] };
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
        debug_assert!(n >= 1, "dispatch_rows: n must be ≥ 1");
        if n == 0 || y.is_empty() {
            return;
        }
        self.pin_caller_once();
        let total_rows = y.len() / n;
        // Split off any trailing partial row now; it runs on the caller after
        // the full rows (the parallel body only handles exact rows).
        let (body, tail) = y.split_at_mut(total_rows * n);
        self.dispatch_body(body, n, total_rows, min_rows, &f);
        if !tail.is_empty() {
            f(total_rows, tail);
        }
    }

    /// The parallel body of [`RowPool::dispatch_rows`]: exactly `total_rows`
    /// full rows of `n` elements (`y.len() == total_rows * n`).
    fn dispatch_body<F>(&self, y: &mut [f32], n: usize, total_rows: usize, min_rows: usize, f: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        if total_rows == 0 {
            return;
        }
        let min_rows = min_rows.max(1);
        // `active` = how many workers participate, gated by `min_rows` so small
        // ops don't wake the whole pool. Within `active`, work is stolen (below).
        let rows_per_worker = total_rows.div_ceil(self.num_threads).max(min_rows);
        let active = total_rows.div_ceil(rows_per_worker).min(self.num_threads);

        // One dispatch owns the pool at a time. If another thread is mid-
        // dispatch (or a closure re-entered the pool), fall back to the serial
        // path below rather than blocking — always sound, never deadlocks. A
        // poisoned lock (a dispatcher panicked) is safe to take: the drain
        // guard below leaves the pool state consistent even on unwind.
        let guard = if active > 1 {
            match self.dispatch_lock.try_lock() {
                Ok(g) => Some(g),
                Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
                Err(std::sync::TryLockError::WouldBlock) => None,
            }
        } else {
            None
        };

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
        // active worker, floored at `MIN_CHUNK_ROWS`.
        let chunk_rows = total_rows
            .div_ceil(active * STEAL_CHUNKS_PER_WORKER)
            .max(MIN_CHUNK_ROWS);

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
        // SAFETY: no worker reads `job` until it observes the `epoch` bump below,
        // which is Released after this write; the previous dispatch already
        // drained (`pending == 0`) before releasing the dispatch lock.
        unsafe {
            *self.shared.job.get() = Some(job);
        }
        // Reset the chunk cursor for this dispatch (ordered before workers can
        // claim by the `epoch` Release/Acquire below).
        self.shared.next_chunk.store(0, Ordering::Relaxed);
        // Every spawned worker decrements `pending` this epoch — including the
        // ones that do no work (`id >= active`) — so the dispatcher waits for
        // *all* of them to finish reading `job` before it can overwrite it on
        // the next dispatch. Without this full barrier an idle worker's `job`
        // read races the next write (a data race Miri catches). Because the
        // dispatcher blocks until fully drained before bumping `epoch` again,
        // each worker observes every epoch exactly once (no skipped epochs).
        self.shared
            .pending
            .store(self.num_threads - 1, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        // Wake any parked workers. Unparking a non-participating (idle) worker is
        // harmless — it observes the epoch, sees `id >= active`, and re-waits.
        for h in &self.workers {
            h.thread().unpark();
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
        // Bump epoch + unpark so spinning/parked workers observe shutdown.
        self.shared.epoch.fetch_add(1, Ordering::Release);
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
    let mut last_epoch = 0u64;
    loop {
        // Wait for a new epoch (or shutdown).
        let mut spins = 0u32;
        let mut parked = false;
        loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let e = shared.epoch.load(Ordering::Acquire);
            if e != last_epoch {
                last_epoch = e;
                break;
            }
            spins += 1;
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
        if parked {
            if let Some(core) = pin_core {
                let _ = pin_current_thread_to_core(core);
            }
        }
        // SAFETY: the Acquire load of `epoch` above synchronizes with the
        // dispatcher's Release bump, so this fresh `Job` (written before that
        // bump) is visible and its pointers are live until we decrement below.
        let job = match unsafe { *shared.job.get() } {
            Some(job) => job,
            None => continue,
        };
        if worker_id < job.active {
            // Steal contiguous chunks until the row space is exhausted (fast
            // cores claim more, balancing heterogeneous cores). A panicking
            // closure is caught so this worker still reaches the `pending`
            // decrement below — otherwise the dispatcher (and every future
            // dispatch) would wait forever on a dead worker. The dispatcher
            // re-raises the panic on the calling thread after the drain.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                steal_and_run(&shared, &job);
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
                    // Another worker already stored its payload. Leak this one
                    // rather than dropping it here: a payload whose `Drop`
                    // panics would unwind past the `pending` decrement below
                    // and wedge the pool.
                    drop(slot);
                    std::mem::forget(payload);
                }
                shared.panicked.store(true, Ordering::Release);
            }
        }
        // Signal completion *after* the `job` read + any work — including idle
        // workers (`id >= job.active`) — so the dispatcher's `pending == 0` wait
        // covers every worker's access to `job` this epoch (see `dispatch_rows`).
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

/// Pin the calling thread to `core` via `sched_setaffinity`. Best-effort:
/// returns whether the pin took — it fails when the core is offline or the
/// process cpuset excludes it (e.g. an Android background cgroup restricted
/// to little cores), in which case the thread stays schedulable as before.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn pin_current_thread_to_core(core: usize) -> bool {
    // SAFETY: `set` is zero-initialized then populated via the libc CPU_SET
    // macro; `sched_setaffinity(0, ...)` targets the current thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) fn pin_current_thread_to_core(_core: usize) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_matches_serial_across_shapes_and_threads() {
        // Fill each row with a function of its absolute index; compare pool
        // output to the serial reference for several sizes, `n`, and pool sizes.
        for &num_threads in &[1usize, 2, 4, 7] {
            let pool = RowPool::build(num_threads, &[]);
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
                        "mismatch: threads={num_threads} n={n} rows={total_rows}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_row_written_exactly_once() {
        // A concurrency-stress check for the disjoint partition: each row adds 1
        // to a counter; after dispatch every counter must be exactly 1 (no lost
        // or double writes).
        let pool = RowPool::build(4, &[]);
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
        let pool = RowPool::build(4, &[]);
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
    fn single_thread_pool_runs_serially() {
        let pool = RowPool::build(1, &[]);
        assert_eq!(pool.num_threads(), 1);
        let mut y = vec![0.0f32; 100];
        pool.dispatch_rows(&mut y, 1, 1, |row, slice| slice[0] = row as f32);
        assert!(y.iter().enumerate().all(|(i, &v)| v == i as f32));
    }

    #[test]
    fn trailing_partial_row_matches_serial_chunks() {
        // y.len() % n != 0: the tail must be visited with the short slice,
        // exactly like the serial `chunks_mut(n)` fallback.
        let pool = RowPool::build(4, &[]);
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
        let pool = RowPool::build(4, &[]);
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
        let pool = RowPool::build(4, &[]);
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

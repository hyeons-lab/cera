//! Thin BLAS wrapper for prefill GEMM via Apple's Accelerate framework.
//!
//! On macOS/iOS this dispatches through Apple's Accelerate framework, which
//! routes SGEMM to the AMX (Apple Matrix eXtension) coprocessor unit - delivering
//! ~1.5-2 TFLOPs f32.
//!
//! Non-Apple platforms (Linux, Android, Windows, WASM) exclusively use Cera's
//! native handcrafted SIMD kernels (AVX2, AVX-512, VNNI, NEON, i8mm, Wasm SIMD128).

// Pull in the provider so its #[link] attribute takes effect at link time.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(unused_imports)]
use accelerate_src as _;

use cblas_sys::{CBLAS_ORDER, CBLAS_TRANSPOSE, cblas_sgemm};

/// Compute `C[m, n] = A[m, k] * B[k, n]` in row-major layout, no transpose on either input.
///
/// `ld_a = k, ld_b = n, ld_c = n`. Alpha=1, beta=0 (output is overwritten, not accumulated).
///
/// # Panics
/// - if `a.len() < m * k`
/// - if `b.len() < k * n`
/// - if `c.len() < m * n`
///
/// # Aliasing
/// `a`, `b`, and `c` must reference non-overlapping memory regions. The CBLAS
/// contract requires distinct input and output buffers — passing the same
/// allocation for two slots is undefined behavior. Rust's `&mut` rules already
/// prevent `c` from aliasing `a` or `b` at the call site (you can't have a
/// shared borrow alive alongside an exclusive borrow), but `a` and `b` could
/// in principle be the same shared slice. We never do that and BLAS would
/// happily compute a meaningless result if we did.
pub fn sgemm_rowmajor_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(
        a.len() >= m * k,
        "sgemm_rowmajor_nn: A buffer too small: {} < {} * {}",
        a.len(),
        m,
        k
    );
    assert!(
        b.len() >= k * n,
        "sgemm_rowmajor_nn: B buffer too small: {} < {} * {}",
        b.len(),
        k,
        n
    );
    assert!(
        c.len() >= m * n,
        "sgemm_rowmajor_nn: C buffer too small: {} < {} * {}",
        c.len(),
        m,
        n
    );

    // CBLAS integer widths are c_int on Linux, c_int on macOS too — just i32
    // on both supported hosts. cast_int is guarded because rustc complains
    // about potential truncation on 16-bit targets which we don't care about.
    let Ok(m_i) = i32::try_from(m) else {
        return;
    };
    let Ok(n_i) = i32::try_from(n) else {
        return;
    };
    let Ok(k_i) = i32::try_from(k) else {
        return;
    };

    // SAFETY:
    // - lengths verified above (a ≥ m*k, b ≥ k*n, c ≥ m*n).
    // - row-major leading dims match: lda=k, ldb=n, ldc=n with no transpose.
    // - non-aliasing: see the function-level Aliasing note. `&mut c` cannot
    //   alias `&a` or `&b` at the call site due to Rust borrow rules.
    // - cblas_sgemm reads a/b and writes c synchronously and does not retain
    //   the pointers after returning.
    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m_i,
            n_i,
            k_i,
            1.0, // alpha
            a.as_ptr(),
            k_i, // lda
            b.as_ptr(),
            n_i, // ldb
            0.0, // beta
            c.as_mut_ptr(),
            n_i, // ldc
        );
    }
}

/// Compute `C[n, m] = B[n, k] * A^T[k, m]` in row-major layout where B has shape [n, k] (ld_b = k)
/// and A has shape [m, k] (ld_a = k). Both inputs are 100% contiguous along the inner contraction dimension k.
pub fn sgemm_rowmajor_nt(n: usize, m: usize, k: usize, b: &[f32], a: &[f32], c: &mut [f32]) {
    assert!(
        b.len() >= n * k,
        "sgemm_rowmajor_nt: B buffer too small: {} < {} * {}",
        b.len(),
        n,
        k
    );
    assert!(
        a.len() >= m * k,
        "sgemm_rowmajor_nt: A buffer too small: {} < {} * {}",
        a.len(),
        m,
        k
    );
    assert!(
        c.len() >= n * m,
        "sgemm_rowmajor_nt: C buffer too small: {} < {} * {}",
        c.len(),
        n,
        m
    );

    let n_i = i32::try_from(n).expect("n overflow");
    let m_i = i32::try_from(m).expect("m overflow");
    let k_i = i32::try_from(k).expect("k overflow");

    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            n_i,
            m_i,
            k_i,
            1.0,
            b.as_ptr(),
            k_i,
            a.as_ptr(),
            k_i,
            0.0,
            c.as_mut_ptr(),
            m_i,
        );
    }
}

/// Compute `C[n, m] = B[n, k] * A[k, m]` in row-major layout with custom leading dimensions `ldb`, `lda`, `ldc`.
///
/// # Safety
/// Caller must ensure pointers are valid for the requested strides and extents.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_rowmajor_nn_ld(
    n: usize,
    m: usize,
    k: usize,
    b: *const f32,
    ldb: usize,
    a: *const f32,
    lda: usize,
    c: *mut f32,
    ldc: usize,
) {
    let n_i = i32::try_from(n).expect("n overflow");
    let m_i = i32::try_from(m).expect("m overflow");
    let k_i = i32::try_from(k).expect("k overflow");
    let ldb_i = i32::try_from(ldb).expect("ldb overflow");
    let lda_i = i32::try_from(lda).expect("lda overflow");
    let ldc_i = i32::try_from(ldc).expect("ldc overflow");

    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            n_i,
            m_i,
            k_i,
            1.0,
            b,
            ldb_i,
            a,
            lda_i,
            0.0,
            c,
            ldc_i,
        );
    }
}

/// Compute `C[n, m] = B[n, k] * A[k, m]` in row-major layout partitioned across all CPU threads/AMX units.
pub fn sgemm_rowmajor_nn_parallel(
    n: usize,
    m: usize,
    k: usize,
    b: &[f32],
    a: &[f32],
    c: &mut [f32],
) {
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        sgemm_rowmajor_nn(n, m, k, b, a, c);
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let num_threads = crate::backend::cpu::configure_thread_pool();
        if num_threads <= 1 || m < 512 {
            sgemm_rowmajor_nn(n, m, k, b, a, c);
            return;
        }

        let b_ptr = b.as_ptr() as usize;
        let a_ptr = a.as_ptr() as usize;
        let c_ptr = c.as_mut_ptr() as usize;

        crate::backend::cpu::par_range(m, 1, move |m_start, m_t| unsafe {
            let b = b_ptr as *const f32;
            let a_t = (a_ptr as *const f32).add(m_start);
            let c_t = (c_ptr as *mut f32).add(m_start);

            let n_i = i32::try_from(n).expect("n overflow");
            let mt_i = i32::try_from(m_t).expect("m_t overflow");
            let k_i = i32::try_from(k).expect("k overflow");
            let lda_i = i32::try_from(m).expect("lda overflow");
            let ldb_i = i32::try_from(k).expect("ldb overflow");
            let ldc_i = i32::try_from(m).expect("ldc overflow");

            cblas_sgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                n_i,
                mt_i,
                k_i,
                1.0,
                b,
                ldb_i,
                a_t,
                lda_i,
                0.0,
                c_t,
                ldc_i,
            );
        });
    }
}

/// Compute `C[n, m] = B[n, k] * A[k, m]` with custom leading strides partitioned across all CPU threads/AMX units.
///
/// # Safety
/// Caller must ensure pointers are valid for the requested strides and extents.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_rowmajor_nn_ld_parallel(
    n: usize,
    m: usize,
    k: usize,
    b: *const f32,
    ldb: usize,
    a: *const f32,
    lda: usize,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        unsafe {
            sgemm_rowmajor_nn_ld(n, m, k, b, ldb, a, lda, c, ldc);
        }
        return;
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let num_threads = crate::backend::cpu::configure_thread_pool();
        if num_threads <= 1 || m < 512 {
            unsafe {
                sgemm_rowmajor_nn_ld(n, m, k, b, ldb, a, lda, c, ldc);
            }
            return;
        }

        let b_ptr = b as usize;
        let a_ptr = a as usize;
        let c_ptr = c as usize;

        crate::backend::cpu::par_range(m, 1, move |m_start, m_t| unsafe {
            let b_p = b_ptr as *const f32;
            let a_t = (a_ptr as *const f32).add(m_start);
            let c_t = (c_ptr as *mut f32).add(m_start);

            let n_i = i32::try_from(n).expect("n overflow");
            let mt_i = i32::try_from(m_t).expect("m_t overflow");
            let k_i = i32::try_from(k).expect("k overflow");
            let ldb_i = i32::try_from(ldb).expect("ldb overflow");
            let lda_i = i32::try_from(lda).expect("lda overflow");
            let ldc_i = i32::try_from(ldc).expect("ldc overflow");

            cblas_sgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                n_i,
                mt_i,
                k_i,
                1.0,
                b_p,
                ldb_i,
                a_t,
                lda_i,
                0.0,
                c_t,
                ldc_i,
            );
        });
    }
}

/// Compute `C[n, m] = B[n, k] * A^T[k, m]` in row-major layout with custom leading dimensions `ldb`, `lda`, `ldc`.
///
/// # Safety
/// Caller must ensure pointers are valid for the requested strides and extents.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_rowmajor_nt_ld(
    n: usize,
    m: usize,
    k: usize,
    b: *const f32,
    ldb: usize,
    a: *const f32,
    lda: usize,
    c: *mut f32,
    ldc: usize,
) {
    let n_i = i32::try_from(n).expect("n overflow");
    let m_i = i32::try_from(m).expect("m overflow");
    let k_i = i32::try_from(k).expect("k overflow");
    let ldb_i = i32::try_from(ldb).expect("ldb overflow");
    let lda_i = i32::try_from(lda).expect("lda overflow");
    let ldc_i = i32::try_from(ldc).expect("ldc overflow");

    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            n_i,
            m_i,
            k_i,
            1.0,
            b,
            ldb_i,
            a,
            lda_i,
            0.0,
            c,
            ldc_i,
        );
    }
}

/// Task descriptor for background AMX worker
#[derive(Default, Copy, Clone)]
struct AmxTask {
    n: i32,
    m: i32,
    k: i32,
    b_ptr: usize,
    ldb: i32,
    a_ptr: usize,
    lda: i32,
    c_ptr: usize,
    ldc: i32,
}

const STATE_IDLE: u32 = 0;
const STATE_RUNNING: u32 = 1;
const STATE_DONE: u32 = 2;
const STATE_LOCKED: u32 = 3;

use std::cell::UnsafeCell;

struct TaskSlot(UnsafeCell<AmxTask>);
unsafe impl Sync for TaskSlot {}

static WORKER_STATE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(STATE_IDLE);
static WORKER_TASK: TaskSlot = TaskSlot(UnsafeCell::new(AmxTask {
    n: 0,
    m: 0,
    k: 0,
    b_ptr: 0,
    ldb: 0,
    a_ptr: 0,
    lda: 0,
    c_ptr: 0,
    ldc: 0,
}));
static WORKER_INIT: std::sync::Once = std::sync::Once::new();

#[inline(always)]
fn set_thread_affinity(_cluster_tag: i32) {
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn mach_thread_self() -> u32;
            fn mach_port_deallocate(task: u32, name: u32) -> i32;
            fn thread_policy_set(
                thread: u32,
                flavor: u32,
                policy_info: *const i32,
                count: u32,
            ) -> i32;
        }
        const THREAD_AFFINITY_POLICY: u32 = 4;
        let policy: [i32; 1] = [_cluster_tag];
        let thread = mach_thread_self();
        thread_policy_set(thread, THREAD_AFFINITY_POLICY, policy.as_ptr(), 1);
        mach_port_deallocate(mach_task_self(), thread);
    }
}

/// RAII Guard that temporarily binds the current thread to a cluster,
/// and restores unconstrained affinity (tag 0) on drop to prevent thread pool pollution.
struct AffinityGuard;

impl AffinityGuard {
    fn set(cluster_tag: i32) -> Self {
        set_thread_affinity(cluster_tag);
        Self
    }
}

impl Drop for AffinityGuard {
    fn drop(&mut self) {
        set_thread_affinity(0);
    }
}

/// RAII Guard that resets `WORKER_STATE` to `STATE_IDLE` on drop to avoid poisoning workers on panic.
struct AmxWorkerGuard;

impl Drop for AmxWorkerGuard {
    fn drop(&mut self) {
        // We MUST block until the background thread is done writing to the caller's stack/heap buffers.
        while WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) == STATE_RUNNING {
            for _ in 0..64 {
                core::hint::spin_loop();
            }
            if WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) == STATE_RUNNING {
                std::thread::yield_now();
            }
        }
        WORKER_STATE.store(STATE_IDLE, std::sync::atomic::Ordering::Release);
    }
}

static WORKER_THREAD: std::sync::OnceLock<std::thread::Thread> = std::sync::OnceLock::new();
static WORKER_AVAILABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn init_dual_amx_pool() -> bool {
    WORKER_INIT.call_once(|| {
        if let Ok(handle) = std::thread::Builder::new()
            .name("cera-amx-worker-1".into())
            .spawn(|| {
                set_thread_affinity(2); // Cluster 1
                let _ = WORKER_THREAD.set(std::thread::current());
                WORKER_AVAILABLE.store(true, std::sync::atomic::Ordering::Release);
                loop {
                    // Spin-wait for work, parking after brief spin to prevent 100% CPU core pinning
                    let mut spins = 0u32;
                    while WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) != STATE_RUNNING {
                        core::hint::spin_loop();
                        spins += 1;
                        if spins > 5_000 {
                            std::thread::park_timeout(std::time::Duration::from_millis(5));
                            spins = 0;
                        }
                    }

                    // Execute task on AMX 1, catching unwinds so STATE_DONE is guaranteed to be set
                    let task = unsafe { *WORKER_TASK.0.get() };
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                        cblas_sgemm(
                            CBLAS_ORDER::CblasRowMajor,
                            CBLAS_TRANSPOSE::CblasNoTrans,
                            CBLAS_TRANSPOSE::CblasTrans,
                            task.n,
                            task.m,
                            task.k,
                            1.0,
                            task.b_ptr as *const f32,
                            task.ldb,
                            task.a_ptr as *const f32,
                            task.lda,
                            0.0,
                            task.c_ptr as *mut f32,
                            task.ldc,
                        );
                    }));

                    // Mark done
                    WORKER_STATE.store(STATE_DONE, std::sync::atomic::Ordering::Release);
                }
            })
        {
            let _ = handle;
        }
    });
    WORKER_AVAILABLE.load(std::sync::atomic::Ordering::Acquire)
}

/// Concurrently compute two independent GEMMs across Cluster 0 (AMX 0) and Cluster 1 (AMX 1):
/// - Matrix 1: `C1[n1, m1] = B1[n1, k1] * A1^T[m1, k1]` (computed on AMX 0 by caller)
/// - Matrix 2: `C2[n2, m2] = B2[n2, k2] * A2^T[m2, k2]` (computed on AMX 1 by background worker)
#[allow(clippy::too_many_arguments)]
pub fn sgemm_dual_parallel(
    n1: usize,
    m1: usize,
    k1: usize,
    b1: &[f32],
    a1: &[f32],
    c1: &mut [f32],
    n2: usize,
    m2: usize,
    k2: usize,
    b2: &[f32],
    a2: &[f32],
    c2: &mut [f32],
) {
    assert!(b1.len() >= n1 * k1, "sgemm_dual_parallel: b1 underflow");
    assert!(a1.len() >= m1 * k1, "sgemm_dual_parallel: a1 underflow");
    assert!(c1.len() >= n1 * m1, "sgemm_dual_parallel: c1 underflow");
    assert!(b2.len() >= n2 * k2, "sgemm_dual_parallel: b2 underflow");
    assert!(a2.len() >= m2 * k2, "sgemm_dual_parallel: a2 underflow");
    assert!(c2.len() >= n2 * m2, "sgemm_dual_parallel: c2 underflow");
    assert!(
        n1 <= i32::MAX as usize && m1 <= i32::MAX as usize && k1 <= i32::MAX as usize,
        "sgemm_dual_parallel: task 1 dimensions exceed i32::MAX"
    );
    assert!(
        n2 <= i32::MAX as usize && m2 <= i32::MAX as usize && k2 <= i32::MAX as usize,
        "sgemm_dual_parallel: task 2 dimensions exceed i32::MAX"
    );

    if !init_dual_amx_pool() {
        sgemm_rowmajor_nt(n1, m1, k1, b1, a1, c1);
        sgemm_rowmajor_nt(n2, m2, k2, b2, a2, c2);
        return;
    }

    // If worker is busy or contended, fall back to sequential CBLAS to avoid clobbering state
    if WORKER_STATE
        .compare_exchange(
            STATE_IDLE,
            STATE_LOCKED,
            std::sync::atomic::Ordering::Acquire,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        sgemm_rowmajor_nt(n1, m1, k1, b1, a1, c1);
        sgemm_rowmajor_nt(n2, m2, k2, b2, a2, c2);
        return;
    }

    let _amx_guard = AmxWorkerGuard;
    let _affinity_guard = AffinityGuard::set(1); // Ensure caller is on Cluster 0 during compute

    // Set up task for Worker 1 (AMX 1)
    unsafe {
        *WORKER_TASK.0.get() = AmxTask {
            n: n2 as i32,
            m: m2 as i32,
            k: k2 as i32,
            b_ptr: b2.as_ptr() as usize,
            ldb: k2 as i32,
            a_ptr: a2.as_ptr() as usize,
            lda: k2 as i32,
            c_ptr: c2.as_mut_ptr() as usize,
            ldc: m2 as i32,
        };
    }
    WORKER_STATE.store(STATE_RUNNING, std::sync::atomic::Ordering::Release);
    if let Some(t) = WORKER_THREAD.get() {
        t.unpark();
    }

    // Concurrently compute Task 1 on AMX 0 (Caller thread)
    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            n1 as i32,
            m1 as i32,
            k1 as i32,
            1.0,
            b1.as_ptr(),
            k1 as i32,
            a1.as_ptr(),
            k1 as i32,
            0.0,
            c1.as_mut_ptr(),
            m1 as i32,
        );
    }

    // Wait for Worker 1 to finish with yield backoff
    while WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) != STATE_DONE {
        for _ in 0..64 {
            core::hint::spin_loop();
        }
        if WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) != STATE_DONE {
            std::thread::yield_now();
        }
    }
}

/// Compute `C[n, m] = B[n, k] * A^T[k, m]` partitioned equally across AMX 0 (top half) and AMX 1 (bottom half).
pub fn sgemm_split2_parallel(n: usize, m: usize, k: usize, b: &[f32], a: &[f32], c: &mut [f32]) {
    assert!(b.len() >= n * k, "sgemm_split2_parallel: b underflow");
    assert!(a.len() >= m * k, "sgemm_split2_parallel: a underflow");
    assert!(c.len() >= n * m, "sgemm_split2_parallel: c underflow");
    assert!(
        n <= i32::MAX as usize && m <= i32::MAX as usize && k <= i32::MAX as usize,
        "sgemm_split2_parallel: dimensions exceed i32::MAX"
    );

    if m < 512 {
        sgemm_rowmajor_nt(n, m, k, b, a, c);
        return;
    }

    if !init_dual_amx_pool() {
        sgemm_rowmajor_nt(n, m, k, b, a, c);
        return;
    }

    // If worker is busy or contended, fall back to single-threaded CBLAS
    if WORKER_STATE
        .compare_exchange(
            STATE_IDLE,
            STATE_LOCKED,
            std::sync::atomic::Ordering::Acquire,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        sgemm_rowmajor_nt(n, m, k, b, a, c);
        return;
    }

    let _amx_guard = AmxWorkerGuard;
    let _affinity_guard = AffinityGuard::set(1);

    let m_top = m / 2;
    let m_bot = m - m_top;

    let b_ptr = b.as_ptr() as usize;
    let a_top_ptr = a.as_ptr() as usize;
    let a_bot_ptr = unsafe { a.as_ptr().add(m_top * k) } as usize;
    let c_top_ptr = c.as_mut_ptr() as usize;
    let c_bot_ptr = unsafe { c.as_mut_ptr().add(m_top) } as usize;

    // Set up bottom half for Worker 1 (AMX 1)
    unsafe {
        *WORKER_TASK.0.get() = AmxTask {
            n: n as i32,
            m: m_bot as i32,
            k: k as i32,
            b_ptr,
            ldb: k as i32,
            a_ptr: a_bot_ptr,
            lda: k as i32,
            c_ptr: c_bot_ptr,
            ldc: m as i32,
        };
    }
    WORKER_STATE.store(STATE_RUNNING, std::sync::atomic::Ordering::Release);
    if let Some(t) = WORKER_THREAD.get() {
        t.unpark();
    }

    // Compute top half on AMX 0 (Caller thread)
    unsafe {
        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            n as i32,
            m_top as i32,
            k as i32,
            1.0,
            b_ptr as *const f32,
            k as i32,
            a_top_ptr as *const f32,
            k as i32,
            0.0,
            c_top_ptr as *mut f32,
            m as i32,
        );
    }

    // Wait for Worker 1 with yield backoff
    while WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) != STATE_DONE {
        for _ in 0..64 {
            core::hint::spin_loop();
        }
        if WORKER_STATE.load(std::sync::atomic::Ordering::Acquire) != STATE_DONE {
            std::thread::yield_now();
        }
    }
}

/// Compute `C[n, m] = B[n, k] * A^T[k, m]` partitioned across parallel worker threads.
pub fn sgemm_rowmajor_nt_parallel(
    n: usize,
    m: usize,
    k: usize,
    b: &[f32],
    a: &[f32],
    c: &mut [f32],
) {
    assert!(
        b.len() >= n * k,
        "sgemm_rowmajor_nt_parallel: B buffer too small"
    );
    assert!(
        a.len() >= m * k,
        "sgemm_rowmajor_nt_parallel: A buffer too small"
    );
    assert!(
        c.len() >= n * m,
        "sgemm_rowmajor_nt_parallel: C buffer too small"
    );

    let num_threads = crate::backend::cpu::configure_thread_pool();
    if num_threads <= 1 || m < 512 {
        sgemm_rowmajor_nt(n, m, k, b, a, c);
        return;
    }

    let b_ptr = b.as_ptr() as usize;
    let a_ptr = a.as_ptr() as usize;
    let c_ptr = c.as_mut_ptr() as usize;

    crate::backend::cpu::par_range(m, 64, move |m_start, m_t| unsafe {
        let b = b_ptr as *const f32;
        let a_t = (a_ptr as *const f32).add(m_start * k);
        let c_t = (c_ptr as *mut f32).add(m_start);

        let (Ok(n_i), Ok(mt_i), Ok(k_i), Ok(lda_i), Ok(ldb_i), Ok(ldc_i)) = (
            i32::try_from(n),
            i32::try_from(m_t),
            i32::try_from(k),
            i32::try_from(k),
            i32::try_from(k),
            i32::try_from(m),
        ) else {
            return;
        };

        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            n_i,
            mt_i,
            k_i,
            1.0,
            b,
            ldb_i,
            a_t,
            lda_i,
            0.0,
            c_t,
            ldc_i,
        );
    });
}

/// Compute `C[n, m] = B[n, k] * A^T[k, m]` partitioned across parallel worker threads with custom strides.
///
/// # Safety
/// Caller must ensure pointer validities, non-aliasing, and buffer bounds for `b`, `a`, and `c`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_rowmajor_nt_ld_parallel(
    n: usize,
    m: usize,
    k: usize,
    b: *const f32,
    ldb: usize,
    a: *const f32,
    lda: usize,
    c: *mut f32,
    ldc: usize,
) {
    let num_threads = crate::backend::cpu::configure_thread_pool();
    if num_threads <= 1 || m < 512 {
        unsafe {
            sgemm_rowmajor_nt_ld(n, m, k, b, ldb, a, lda, c, ldc);
        }
        return;
    }

    let b_ptr = b as usize;
    let a_ptr = a as usize;
    let c_ptr = c as usize;

    crate::backend::cpu::par_range(m, 64, move |m_start, m_t| unsafe {
        let b = b_ptr as *const f32;
        let a_t = (a_ptr as *const f32).add(m_start * lda);
        let c_t = (c_ptr as *mut f32).add(m_start);

        let (Ok(n_i), Ok(mt_i), Ok(k_i), Ok(lda_i), Ok(ldb_i), Ok(ldc_i)) = (
            i32::try_from(n),
            i32::try_from(m_t),
            i32::try_from(k),
            i32::try_from(lda),
            i32::try_from(ldb),
            i32::try_from(ldc),
        ) else {
            return;
        };

        cblas_sgemm(
            CBLAS_ORDER::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            n_i,
            mt_i,
            k_i,
            1.0,
            b,
            ldb_i,
            a_t,
            lda_i,
            0.0,
            c_t,
            ldc_i,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_identity() {
        // C = I * B should equal B
        let m = 4;
        let k = 4;
        let n = 3;
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            a[i * k + i] = 1.0;
        }
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let mut c = vec![0.0f32; m * n];
        sgemm_rowmajor_nn(m, n, k, &a, &b, &mut c);
        for i in 0..m * n {
            assert_eq!(c[i], b[i], "identity GEMM failed at {i}");
        }
    }

    #[test]
    fn test_sgemm_simple_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A*B = [[19,22],[43,50]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm_rowmajor_nn(2, 2, 2, &a, &b, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_concurrent_sgemm_dual_parallel() {
        let mut handles = Vec::new();
        for _ in 0..16 {
            let handle = std::thread::spawn(|| {
                let m = 32;
                let n = 16;
                let k = 32;
                let a1 = vec![1.0f32; m * k];
                let a2 = vec![2.0f32; m * k];
                let b = vec![1.0f32; n * k];
                let mut c1 = vec![0.0f32; m * n];
                let mut c2 = vec![0.0f32; m * n];
                sgemm_dual_parallel(m, n, k, &a1, &a2, &b, &mut c1, &mut c2);
                for &val in &c1 {
                    assert!((val - k as f32).abs() < 1e-4);
                }
                for &val in &c2 {
                    assert!((val - (2.0 * k as f32)).abs() < 1e-4);
                }
            });
            handles.push(handle);
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Microbenchmark: ffn_up shape (m=6912, n=2002, k=2048) — compare the
    /// NEON integer kernel (quantize input + q4_0×q8_0 GEMM) against the
    /// dequant + cblas_sgemm path the smoke test is wiring up. Ignored by
    /// default; run with:
    /// `cargo test -p cera --release --lib backend::blas::tests::microbench_ffn_up_gemm -- --ignored --nocapture`
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn microbench_ffn_up_gemm() {
        use crate::backend::simd::neon;
        use crate::quant::{BlockQ4_0, dequantize_q4_0_matrix};
        use std::time::Instant;

        fn gflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
            (2.0 * m as f64 * n as f64 * k as f64) / (seconds * 1e9)
        }

        let m = 6912; // is
        let k = 2048; // hs
        let n = 2002; // prompt length
        let iters = 4;

        // Random Q4_0 weight buffer — contents aren't statistically realistic
        // but the kernel work is identical regardless of byte content.
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * size_of::<BlockQ4_0>();
        let mut weight = vec![0u8; m * row_bytes];
        let mut s: u64 = 0xdead_beef;
        for byte in weight.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (s >> 33) as u8;
        }

        let input: Vec<f32> = (0..k * n)
            .map(|i| ((i * 31) % 127) as f32 * 0.01 - 0.5)
            .collect();

        // ── Path A: BLAS (dequant + cblas_sgemm) ──────────────────────
        let mut dequant = vec![0.0f32; m * k];
        let mut out_blas = vec![0.0f32; m * n];

        // Warmup
        dequantize_q4_0_matrix(&weight, m, k, &mut dequant);
        sgemm_rowmajor_nn(m, n, k, &dequant, &input, &mut out_blas);

        let t0 = Instant::now();
        for _ in 0..iters {
            dequantize_q4_0_matrix(&weight, m, k, &mut dequant);
        }
        let dequant_per = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            sgemm_rowmajor_nn(m, n, k, &dequant, &input, &mut out_blas);
        }
        let sgemm_per = t0.elapsed().as_secs_f64() / iters as f64;
        let blas_total_per = dequant_per + sgemm_per;

        // ── Path B: NEON integer GEMM ─────────────────────────────────
        let nb_k = k / 32;
        let mut b_scales = vec![0.0f32; n * nb_k];
        let mut b_quants = vec![0i8; n * k];
        let mut col = vec![0.0f32; k];

        let t0 = Instant::now();
        for _ in 0..iters {
            for j in 0..n {
                for i in 0..k {
                    col[i] = input[i * n + j];
                }
                unsafe {
                    neon::quantize_f32_to_q8_0_neon(
                        &col,
                        &mut b_scales[j * nb_k..(j + 1) * nb_k],
                        &mut b_quants[j * k..(j + 1) * k],
                    );
                }
            }
        }
        let quantize_per = t0.elapsed().as_secs_f64() / iters as f64;

        let mut out_neon = vec![0.0f32; m * n];
        // Warmup
        unsafe {
            neon::gemm_q4_0_q8_0_neon(&weight, &b_scales, &b_quants, &mut out_neon, m, n, k);
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                neon::gemm_q4_0_q8_0_neon(&weight, &b_scales, &b_quants, &mut out_neon, m, n, k);
            }
        }
        let neon_gemm_per = t0.elapsed().as_secs_f64() / iters as f64;
        let neon_total_per = quantize_per + neon_gemm_per;

        eprintln!("\n=== ffn_up GEMM microbench ({m} × {n} × {k}) ===");
        eprintln!("BLAS (dequant + sgemm):");
        eprintln!("  dequant:  {:>7.1} ms", dequant_per * 1000.0);
        eprintln!(
            "  sgemm:    {:>7.1} ms   ({:.1} GFLOPs/s)",
            sgemm_per * 1000.0,
            gflops(m, n, k, sgemm_per)
        );
        eprintln!(
            "  total:    {:>7.1} ms   ({:.1} GFLOPs/s effective)",
            blas_total_per * 1000.0,
            gflops(m, n, k, blas_total_per)
        );
        eprintln!("NEON (quantize + q4_0×q8_0 gemm):");
        eprintln!("  quantize: {:>7.1} ms", quantize_per * 1000.0);
        eprintln!(
            "  gemm:     {:>7.1} ms   ({:.1} GFLOPs/s)",
            neon_gemm_per * 1000.0,
            gflops(m, n, k, neon_gemm_per)
        );
        eprintln!(
            "  total:    {:>7.1} ms   ({:.1} GFLOPs/s effective)",
            neon_total_per * 1000.0,
            gflops(m, n, k, neon_total_per)
        );
        eprintln!(
            "\nNEON / BLAS total: {:.2}×   (>1 means NEON wins)",
            neon_total_per / blas_total_per
        );
        eprintln!(
            "NEON gemm / BLAS sgemm only: {:.2}×   (isolates kernel, excludes dequant/quantize)",
            neon_gemm_per / sgemm_per
        );
    }
}

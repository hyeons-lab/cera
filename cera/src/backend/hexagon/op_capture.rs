//! Test-only recorder of flushed batches (thread-local: each test runs on its
//! own thread). `HexagonQueueSession::flush` calls [`record`], so host tests
//! can pin the exact op sequence a forward pass emits without a device,
//! including the queue's own capped and step-mode flushes.

use crate::backend::hexagon::{HexagonDevice, HexagonQueueSession, HtpOpCode};
use std::cell::{Cell, RefCell};

thread_local! {
    static BATCHES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static PANIC_ON_RECORD: Cell<bool> = const { Cell::new(false) };
}

/// Make the next [`record`] on this thread panic: an unwind from inside a
/// forward's flush, for tests of poisoned-lock recovery (the fake driver's
/// `extern "C"` entry points cannot unwind).
pub(crate) fn panic_on_next_record() {
    PANIC_ON_RECORD.with(|p| p.set(true));
}

/// Serialize one pending batch: per op, the opcode, params, kernel params
/// and every operand tensor (buffer index, offset, size, flags, dtype,
/// shape, strides). Buffer addresses and fds are left out, so the text is
/// stable across runs.
pub(crate) fn record(session: &HexagonQueueSession) {
    if PANIC_ON_RECORD.with(|p| p.replace(false)) {
        panic!("op_capture: injected panic");
    }
    let (bufs, _map, tens, ops) = session.export_batch();
    let mut out = String::new();
    for op in &ops {
        let name = HtpOpCode::from_u32(op.opcode).map_or("?", |o| o.name());
        out.push_str(&format!(
            "{name} params={:?} kparams={:?}",
            op.params, op.kernel_params
        ));
        let operand = |i: u16| {
            let t = &tens[i as usize];
            format!(
                " [b{} off={} size={} fl={} ty={} ne={:?} nb={:?} bsize={}]",
                t.bi, t.data, t.size, t.flags, t.dtype, t.ne, t.nb, bufs[t.bi as usize].size
            )
        };
        out.push_str(" src:");
        for &i in op.src.iter().take_while(|&&i| i != 0xffff) {
            out.push_str(&operand(i));
        }
        out.push_str(" dst:");
        for &i in op.dst.iter().take_while(|&&i| i != 0xffff) {
            out.push_str(&operand(i));
        }
        out.push('\n');
    }
    BATCHES.with(|b| b.borrow_mut().push(out));
}

/// Pin a device's queue to the environment-free defaults: no ops-per-flush
/// cap and no step mode (`CERA_HEXAGON_STEP` is read into the session at
/// creation), so goldens are identical whatever the process exports.
pub(crate) fn hermetic_session(device: &mut HexagonDevice) {
    let session = device.queue_session_mut();
    session.set_max_ops_per_flush(None);
    session.set_step_mode(false);
}

/// Reset the fake driver and this thread's recorder, then open a V79 device
/// on it with a [`hermetic_session`]. The five-line scaffold every host test
/// that drives a fake device starts with.
pub(crate) fn fresh_device() -> (
    std::sync::Arc<crate::backend::hexagon::FastRpcDriver>,
    HexagonDevice,
) {
    use crate::backend::hexagon::{HexagonArch, sys::fake};
    fake::reset();
    take();
    PANIC_ON_RECORD.with(|p| p.set(false));
    let driver = fake::driver();
    let mut device = HexagonDevice::new(std::sync::Arc::clone(&driver), HexagonArch::V79).unwrap();
    hermetic_session(&mut device);
    (driver, device)
}

/// Per-opcode count over recorded batches, for readable golden failures.
pub(crate) fn histogram(batches: &[String]) -> std::collections::BTreeMap<String, usize> {
    let mut h = std::collections::BTreeMap::new();
    for l in batches.iter().flat_map(|b| b.lines()) {
        *h.entry(l.split(' ').next().unwrap().to_string())
            .or_insert(0) += 1;
    }
    h
}

/// `CERA_UPDATE_GOLDEN` parse: only `1` / `true` (any case) turn it on.
fn golden_flag(v: Option<&str>) -> bool {
    v.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// With `CERA_UPDATE_GOLDEN=1`, write the recorded op text of `label` to
/// `target/golden/<label>.txt` (one blank-line-separated block per flush)
/// so a changed golden can be diffed against the previous dump.
pub(crate) fn dump_if_requested(label: &str, batches: &[String]) {
    // Only `1`/`true` turn it on, so `CERA_UPDATE_GOLDEN=0` does not dump.
    if !golden_flag(std::env::var("CERA_UPDATE_GOLDEN").ok().as_deref()) {
        return;
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target"));
    let dir = target.join("golden");
    std::fs::create_dir_all(&dir).expect("create golden dump dir");
    let path = dir.join(format!("{label}.txt"));
    std::fs::write(&path, batches.join("\n")).expect("write golden dump");
    eprintln!("golden dump: {}", path.display());
}

/// Assert `got == expected`; on mismatch print the per-opcode histogram and
/// how to dump the full op text. The digest stays the tight pin, this only
/// makes a failure readable.
pub(crate) fn assert_golden<T: PartialEq + std::fmt::Debug>(
    label: &str,
    got: T,
    expected: T,
    batches: &[String],
) {
    dump_if_requested(label, batches);
    assert!(
        got == expected,
        "golden `{label}` changed\n  got      {got:?}\n  expected {expected:?}\n  {} flushes, ops per kind: {:?}\n  rerun with CERA_UPDATE_GOLDEN=1 to dump the op text to target/golden/{label}.txt and diff it",
        batches.len(),
        histogram(batches),
    );
}

/// Take everything recorded on this thread, one string per flush.
pub(crate) fn take() -> Vec<String> {
    BATCHES.with(|b| std::mem::take(&mut *b.borrow_mut()))
}

/// `CERA_UPDATE_GOLDEN` only turns on for `1` / `true` (any case).
#[test]
fn update_golden_only_on_for_one_or_true() {
    assert!(golden_flag(Some("1")));
    assert!(golden_flag(Some("true")));
    assert!(golden_flag(Some("TRUE")));
    for off in [None, Some("0"), Some(""), Some("false"), Some("yes")] {
        assert!(!golden_flag(off), "{off:?}");
    }
}

//! Host tests of the bidirectional encoder pass (`encoder.rs`).
//!
//! No DSP is involved. Three layers of evidence:
//!
//! * pure functions (row-count plan, mask, scratch layout, state-slab slots);
//! * the conv window emission, executed by a small interpreter of the four ops
//!   it uses (`Mul`, `Cpy`, `Concat`, `SsmConv`, per the semantics documented on
//!   `dispatch::ssm_conv` / `concat_time_inner`) against the CPU encoder's conv
//!   arithmetic on random data. The interpreter is first checked against the
//!   *causal* prefill's window (the ops `emit_conv_prefill` emits), whose DSP
//!   behaviour is already device-verified, so a wrong reading of an op would
//!   fail there before it could vouch for the new code;
//! * whole passes through a queue session on the fake FastRPC driver, pinning
//!   the op stream and checking the host plumbing (padding, positions, mask,
//!   prefix capture and restore, refusals). Numerics are not checked there
//!   (all weights are zero); on a device, run `cera/tests/d1_public.rs` with `CERA_D1_BACKEND=npu`
//!   (ignored by default) for end-to-end parity with llama.cpp.

use super::encoder::*;
use super::forward_golden_tests as fx;
use super::*;
use crate::backend::hexagon::op_capture;
use crate::backend::hexagon::{HtpBufDesc, HtpOpDesc};
use crate::model::{EncoderInput, EncoderPrefix};

const HS: usize = fx::HS;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn rand_vec(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n).map(|_| lcg(seed)).collect()
}

// ---------------------------------------------------------------------------
// Pure pieces
// ---------------------------------------------------------------------------

#[test]
fn shape_pads_to_eight_rows_and_keeps_real_rows_apart() {
    for (n, m) in [(1, 8), (3, 8), (7, 8), (8, 8), (9, 9), (512, 512)] {
        let s = EncoderShape::plan(0, n, 4096).unwrap();
        assert_eq!((s.n_real, s.m, s.start_pos), (n, m, 0));
        // The flash attention spans every row's slot (never fewer slots than
        // queries, like a causal chunk); queries may read only the real ones.
        assert_eq!((s.kv_len(), s.visible()), (m, n));
    }
    // Rows after a prefix: positions continue, attention covers the prefix too.
    let s = EncoderShape::plan(40, 3, 4096).unwrap();
    assert_eq!((s.start_pos, s.n_real, s.m), (40, 3, 8));
    assert_eq!((s.kv_len(), s.visible()), (48, 43));
}

#[test]
fn shape_refuses_what_does_not_fit_with_typed_errors() {
    assert!(matches!(
        EncoderShape::plan(0, 0, 64),
        Err(CeraError::EmptyInput)
    ));
    // The single-chunk shape is for up to one chunk; longer passes are planned
    // layer-outer (`plan_pass`, tested below).
    let err = EncoderShape::plan(0, 513, 4096).unwrap_err().to_string();
    assert!(
        err.contains("single-chunk encoder pass takes at most 512"),
        "{err}"
    );
    // A 512-row prefix then a short text is a plain single-chunk pass.
    EncoderShape::plan(512, 100, 4096).unwrap();
    // The padding rows write K/V too, so they need slots: 60 + 8 > 64.
    assert!(matches!(
        EncoderShape::plan(60, 5, 64),
        Err(CeraError::ContextOverflow {
            max_seq_len: 64,
            by: 4
        })
    ));
    EncoderShape::plan(56, 5, 64).unwrap();
}

#[test]
fn bidirectional_mask_has_no_causal_triangle() {
    // (start_pos, real rows, m): slots 0..start+real are visible to every query.
    for (start_pos, n, m) in [(0, 5, 8), (6, 3, 8), (0, 16, 16), (40, 8, 8), (0, 8, 8)] {
        let (kv_len, visible) = (start_pos + m, start_pos + n);
        let mut mask = vec![0xABCDu16; kv_len * m + 3];
        fill_bidirectional_mask(&mut mask, m, kv_len, visible);
        for (q, row) in mask[..kv_len * m].chunks(kv_len).enumerate() {
            assert!(
                row[..visible].iter().all(|&v| v == 0),
                "query {q} sees the real slots"
            );
            assert!(
                row[visible..].iter().all(|&v| v == MASK_NEG_INF),
                "query {q} must not see padding slots"
            );
        }
        // Only the [kv_len, m] block is written.
        assert!(mask[kv_len * m..].iter().all(|&v| v == 0xABCD));
        // The causal fill on the same block hides later slots from earlier
        // queries; the bidirectional one does not (the first query sees the
        // last real slot).
        let mut causal = vec![0u16; kv_len * m];
        fill_prefill_mask(&mut causal, start_pos, m, kv_len, None);
        if n > 1 {
            assert_eq!(causal[visible - 1], MASK_NEG_INF);
            assert_eq!(mask[visible - 1], 0);
        }
    }
}

#[test]
fn conv_window_fits_the_scratch_for_every_row_count() {
    for (hs, q_dim, kv_dim, inter) in [
        (64, 64, 32, 128),
        (1024, 1024, 512, 4096),
        (2048, 2048, 512, 8192),
    ] {
        let so = ScratchOffsets::new(hs, q_dim, kv_dim, inter, 1000, 4096, None, None);
        let avail = so.conv_ssm_y - so.conv_x;
        for m in ENCODER_MIN_ROWS..=PREFILL_MAX_ROWS {
            let w = ConvWindow::plan(so.conv_x, avail, hs, m).unwrap();
            let row = hs * 4;
            // Disjoint, aligned, inside conv_x.
            assert_eq!(w.bx, so.conv_x);
            assert!(w.bx + (m + 1) * row <= w.window, "hs {hs} m {m}");
            assert_eq!(w.window % 256, 0);
            assert!(w.window + (m + 2) * row <= so.conv_ssm_y, "hs {hs} m {m}");
        }
    }
    // A scratch that is too small is an error, not an overlap.
    assert!(ConvWindow::plan(0, 1000, 64, 8).is_err());
}

#[test]
fn state_slab_slots_interleave_per_channel() {
    let hs = 5;
    let mut slab = vec![0u8; hs * 8];
    let a: Vec<f32> = (0..hs).map(|c| c as f32 + 0.5).collect();
    let b: Vec<f32> = (0..hs).map(|c| -(c as f32) - 1.0).collect();
    slab_write_slot(&mut slab, 0, &a);
    slab_write_slot(&mut slab, 1, &b);
    // Channel c: slot 0 at c * 8, slot 1 at c * 8 + 4 (the layout of the DSP's
    // `[C, 2]` state, as the causal path writes it).
    for c in 0..hs {
        let at = |o: usize| f32::from_ne_bytes(slab[o..o + 4].try_into().unwrap());
        assert_eq!(at(c * 8), a[c]);
        assert_eq!(at(c * 8 + 4), b[c]);
    }
    let mut got = vec![0.0; hs];
    slab_read_slot(&slab, 0, &mut got);
    assert_eq!(got, a);
    slab_read_slot(&slab, 1, &mut got);
    assert_eq!(got, b);
}

#[test]
fn balanced_chunks_cover_every_row_and_are_never_tiny() {
    for (n, c) in [
        (17, 16),
        (33, 16),
        (40, 16),
        (513, 512),
        (1025, 512),
        (16384, 512),
    ] {
        let chunks = balanced_chunks(n, c);
        assert_eq!(chunks.len(), n.div_ceil(c), "{n} rows in chunks of {c}");
        // contiguous, covering exactly n rows
        let mut next = 0;
        for &(first, rows) in &chunks {
            assert_eq!(first, next);
            next += rows;
        }
        assert_eq!(next, n);
        // sizes differ by at most one, and none is under the DSP's 8-row minimum
        let (lo, hi) = (
            chunks.iter().map(|c| c.1).min().unwrap(),
            chunks.iter().map(|c| c.1).max().unwrap(),
        );
        assert!(hi - lo <= 1, "{chunks:?}");
        assert!(lo >= ENCODER_MIN_ROWS && hi <= c, "{chunks:?}");
    }
    // a pass that fits one chunk stays one chunk
    assert_eq!(balanced_chunks(16, 16), vec![(0, 16)]);
}

#[test]
fn a_pass_plans_single_up_to_one_chunk_and_layer_outer_beyond() {
    let single = plan_pass(0, 16, 4096, 16).unwrap();
    assert!(matches!(single, PassPlan::Single(_)), "{single:?}");
    match plan_pass(10, 40, 4096, 16).unwrap() {
        PassPlan::Long(long) => {
            assert_eq!((long.start_pos, long.n), (10, 40));
            assert_eq!(long.chunks, balanced_chunks(40, 16));
        }
        other => panic!("{other:?}"),
    }
    // refusals are typed: nothing, the cache, the pass limit, a bad chunk size
    assert!(matches!(
        plan_pass(0, 0, 4096, 16),
        Err(CeraError::EmptyInput)
    ));
    assert!(matches!(
        plan_pass(4090, 40, 4096, 16),
        Err(CeraError::ContextOverflow { by: 34, .. })
    ));
    assert!(matches!(
        plan_pass(0, ENCODER_MAX_PASS_ROWS + 1, 1 << 20, 512),
        Err(CeraError::ContextOverflow { by: 1, .. })
    ));
    for bad in [0, 2 * ENCODER_MIN_ROWS - 1, PREFILL_MAX_ROWS + 1] {
        let err = plan_pass(0, 40, 4096, bad).unwrap_err().to_string();
        assert!(err.contains("outside"), "{err}");
    }
}

// ---------------------------------------------------------------------------
// A small interpreter of the ops the conv window uses
// ---------------------------------------------------------------------------

struct TView {
    base: *mut u8,
    ne: [usize; 3],
    nb: [usize; 3],
}

impl TView {
    fn new(bufs: &[HtpBufDesc], t: &crate::backend::hexagon::HtpTensor) -> Self {
        assert_eq!(t.dtype, HtpDataType::F32 as u32, "interpreter is f32 only");
        TView {
            base: (bufs[t.bi as usize].base as usize + t.data as usize) as *mut u8,
            ne: [t.ne[0] as usize, t.ne[1] as usize, t.ne[2] as usize],
            nb: [t.nb[0] as usize, t.nb[1] as usize, t.nb[2] as usize],
        }
    }

    fn at(&self, i: [usize; 3]) -> *mut f32 {
        let off = i[0] * self.nb[0] + i[1] * self.nb[1] + i[2] * self.nb[2];
        unsafe { self.base.add(off) as *mut f32 }
    }

    fn get(&self, i: [usize; 3]) -> f32 {
        unsafe { *self.at(i) }
    }

    fn set(&self, i: [usize; 3], v: f32) {
        unsafe { *self.at(i) = v }
    }
}

/// Execute the `Mul`, `Cpy`, `Concat` and `SsmConv` ops of an exported batch on
/// the host memory behind its buffers (the fake driver's buffers are plain
/// allocations). Anything else is a test bug.
fn interpret(bufs: &[HtpBufDesc], tens: &[crate::backend::hexagon::HtpTensor], ops: &[HtpOpDesc]) {
    for op in ops {
        let src = |i: usize| TView::new(bufs, &tens[op.src[i] as usize]);
        let dst = TView::new(bufs, &tens[op.dst[0] as usize]);
        match HtpOpCode::from_u32(op.opcode).expect("opcode") {
            HtpOpCode::Mul => {
                let (a, b) = (src(0), src(1));
                assert_eq!(a.ne, b.ne, "no broadcasting in the conv window");
                for i2 in 0..a.ne[2] {
                    for i1 in 0..a.ne[1] {
                        for i0 in 0..a.ne[0] {
                            let i = [i0, i1, i2];
                            dst.set(i, a.get(i) * b.get(i));
                        }
                    }
                }
            }
            HtpOpCode::Cpy => {
                let s = src(0);
                assert_eq!(s.ne, dst.ne);
                // memcpy semantics: read everything first would hide an
                // overlap bug the DSP has, so copy element by element in order.
                for i2 in 0..s.ne[2] {
                    for i1 in 0..s.ne[1] {
                        for i0 in 0..s.ne[0] {
                            let i = [i0, i1, i2];
                            dst.set(i, s.get(i));
                        }
                    }
                }
            }
            HtpOpCode::Concat => {
                assert_eq!(op.params[0], 0, "concat along dim 0 only");
                let (a, b) = (src(0), src(1));
                assert_eq!(dst.ne[0], a.ne[0] + b.ne[0]);
                assert_eq!(a.ne[1], b.ne[1]);
                for c in 0..dst.ne[1] {
                    for t in 0..a.ne[0] {
                        dst.set([t, c, 0], a.get([t, c, 0]));
                    }
                    for t in 0..b.ne[0] {
                        dst.set([a.ne[0] + t, c, 0], b.get([t, c, 0]));
                    }
                }
            }
            HtpOpCode::SsmConv => {
                // y[c, t] = sum_j x[t + j, c] * w[j, c]
                let (x, w) = (src(0), src(1));
                let (d_conv, d_inner) = (w.ne[0], w.ne[1]);
                let n_t = dst.ne[1];
                assert_eq!(dst.ne[0], d_inner);
                assert_eq!(x.ne[0], d_conv - 1 + n_t);
                for t in 0..n_t {
                    for c in 0..d_inner {
                        let mut acc = 0.0f32;
                        for j in 0..d_conv {
                            acc += x.get([t + j, c, 0]) * w.get([j, c, 0]);
                        }
                        dst.set([c, t, 0], acc);
                    }
                }
            }
            other => panic!("interpreter has no {other:?}"),
        }
    }
}

fn encoder_spec(n_layers: usize) -> fx::Spec {
    let mut s = fx::spec(
        |p| {
            (0..n_layers)
                .map(|i| {
                    if i % 2 == 0 {
                        p.conv()
                    } else {
                        p.attention(fx::Extras::default())
                    }
                })
                .collect()
        },
        false,
    );
    s.config.is_causal = false;
    s
}

/// A non-causal conv/attention model on the fake device, one fd per buffer so
/// the interpreter can tell them apart.
fn build_encoder(n_layers: usize) -> HexagonLfmModel {
    fx::DISTINCT_FDS.with(|d| d.set(true));
    fx::build(encoder_spec(n_layers))
}

fn conv_layer(model: &HexagonLfmModel, i: usize) -> HexagonConvLayer {
    match &model.layers[i] {
        HexagonLayer::Conv(c) => *c,
        _ => panic!("layer {i} is not a conv layer"),
    }
}

fn write_f32(buf: &RpcmemBuffer, offset: usize, values: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            buf.as_mut_ptr().add(offset),
            values.len() * 4,
        );
    }
}

fn read_f32(buf: &RpcmemBuffer, offset: usize, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.as_ptr().add(offset) as *const f32, n) }.to_vec()
}

fn slab(model: &HexagonLfmModel, conv: &HexagonConvLayer) -> &'static mut [u8] {
    // The model owns the buffer for the whole test; tests are single threaded.
    unsafe {
        std::slice::from_raw_parts_mut(
            model.kv_state_buf.as_mut_ptr().add(conv.state_offset),
            HS * 8,
        )
    }
}

/// CPU encoder arithmetic (`LfmModel::prefill_layers_loop`, non-causal branch)
/// for the pre-gate conv output: `bx = b * x`, then
/// `y = prev * w0 + curr * w1 + next * w2` with a zero neighbour past either end
/// (`left` replacing the zero before the first row).
fn cpu_symmetric_conv(conv_in: &[f32], taps: &[f32], n: usize, left: Option<&[f32]>) -> Vec<f32> {
    let bx: Vec<Vec<f32>> = (0..n)
        .map(|j| {
            let row = &conv_in[j * 3 * HS..(j + 1) * 3 * HS];
            (0..HS).map(|c| row[c] * row[2 * HS + c]).collect()
        })
        .collect();
    let zero = vec![0.0f32; HS];
    let mut y = vec![0.0f32; n * HS];
    for j in 0..n {
        let prev = if j > 0 {
            &bx[j - 1][..]
        } else {
            left.unwrap_or(&zero)
        };
        let next = if j + 1 < n { &bx[j + 1][..] } else { &zero };
        for c in 0..HS {
            let (w0, w1, w2) = (taps[c * 3], taps[c * 3 + 1], taps[c * 3 + 2]);
            y[j * HS + c] = prev[c] * w0 + bx[j][c] * w1 + next[c] * w2;
        }
    }
    y
}

struct ConvRun {
    conv_in: Vec<f32>,
    /// Pre-gate conv output rows `0..n_real`.
    y: Vec<f32>,
    /// Slot 0 of the state slab after the pass.
    slot0: Vec<f32>,
    taps: Vec<f32>,
}

/// Emit the encoder conv window for `n_real` rows after `start_pos` on random
/// data and interpret it. `left` is the slot-0 left neighbour the host would
/// have written (`None` = zeros).
fn run_conv_window(
    model: &HexagonLfmModel,
    seed: u64,
    start_pos: usize,
    n_real: usize,
    left: Option<&[f32]>,
    save_state: bool,
    taps: Option<Vec<f32>>,
) -> ConvRun {
    let mut seed = seed;
    let so = model.scratch_offsets;
    let scratch = &model.scratch_buf;
    let conv = conv_layer(model, 0);
    let shape = EncoderShape::plan(start_pos, n_real, fx::MAX_SEQ).unwrap();
    let m = shape.m;
    let win = ConvWindow::plan(so.conv_x, so.conv_ssm_y - so.conv_x, HS, m).unwrap();

    let conv_in = rand_vec(&mut seed, m * 3 * HS);
    let taps = taps.unwrap_or_else(|| rand_vec(&mut seed, HS * 3));
    write_f32(scratch, so.conv_in, &conv_in);
    write_f32(&model.weights_buf, conv.conv_ssm_offset, &taps);
    // Sentinels everywhere the emission must overwrite (or never read).
    let sentinel = vec![7.0f32; (so.conv_ssm_y - so.conv_x) / 4];
    write_f32(scratch, so.conv_x, &sentinel);
    write_f32(scratch, so.conv_ssm_y, &vec![5.0f32; m * HS]);
    // Host zeroing of the bx rows past the real ones, through the right
    // neighbour row, exactly as `encoder_pass` does.
    write_f32(
        scratch,
        win.bx + n_real * HS * 4,
        &vec![0.0f32; (m - n_real + 1) * HS],
    );
    let zero = vec![0.0f32; HS];
    slab_write_slot(slab(model, &conv), 0, left.unwrap_or(&zero));
    slab_write_slot(slab(model, &conv), 1, &vec![9.0f32; HS]);

    let mut device = model.lock_device();
    let session = device.queue_session_mut();
    session.drop_pending_batch();
    model
        .emit_conv_window(session, &conv, &shape, &win, save_state)
        .unwrap();
    let (bufs, _, tens, ops) = session.export_batch();
    session.drop_pending_batch();
    drop(device);
    interpret(&bufs, &tens, &ops);

    let y = read_f32(scratch, so.conv_ssm_y, n_real * HS);
    let mut slot0 = vec![0.0f32; HS];
    slab_read_slot(slab(model, &conv), 0, &mut slot0);
    ConvRun {
        conv_in,
        y,
        slot0,
        taps,
    }
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 * (1.0 + w.abs()),
            "{what}: element {i}: {g} vs {w}"
        );
    }
}

/// The interpreter against the causal prefill's window: the same two helper
/// calls `emit_conv_prefill` makes (state prepend concat, then SsmConv), compared
/// with the CPU causal conv (`b0 * w0 + b1 * w1 + bx * w2`). Device-verified ops,
/// so this pins the interpreter's reading of them.
#[test]
fn interpreter_reproduces_the_causal_conv_window() {
    let model = build_encoder(2);
    let so = model.scratch_offsets;
    let scratch = &model.scratch_buf;
    let conv = conv_layer(&model, 0);
    let mut seed = 11;
    for m in [1usize, 2, 5, 16] {
        let bx = rand_vec(&mut seed, m * HS);
        let taps = rand_vec(&mut seed, HS * 3);
        let (s0, s1) = (rand_vec(&mut seed, HS), rand_vec(&mut seed, HS));
        write_f32(scratch, so.conv_bx, &bx);
        write_f32(&model.weights_buf, conv.conv_ssm_offset, &taps);
        slab_write_slot(slab(&model, &conv), 0, &s0);
        slab_write_slot(slab(&model, &conv), 1, &s1);
        let mut device = model.lock_device();
        let session = device.queue_session_mut();
        session.drop_pending_batch();
        HexagonLfmModel::dispatch_concat_2d(
            session,
            &model.kv_state_buf,
            conv.state_offset,
            2,
            scratch,
            so.conv_bx,
            m,
            HS * 4,
            4,
            scratch,
            so.conv_x,
            HS,
        )
        .unwrap();
        model
            .dispatch_ssm_conv(
                session,
                &model.weights_buf,
                conv.conv_ssm_offset,
                scratch,
                so.conv_x,
                scratch,
                so.conv_ssm_y,
                3,
                HS,
                m,
            )
            .unwrap();
        let (bufs, _, tens, ops) = session.export_batch();
        session.drop_pending_batch();
        drop(device);
        interpret(&bufs, &tens, &ops);
        let got = read_f32(scratch, so.conv_ssm_y, m * HS);
        let mut want = vec![0.0f32; m * HS];
        for t in 0..m {
            for c in 0..HS {
                let at = |k: isize| -> f32 {
                    // Rows before the block come from the two-slot state.
                    match t as isize + k {
                        i if i >= 0 => bx[i as usize * HS + c],
                        -1 => s1[c],
                        _ => s0[c],
                    }
                };
                want[t * HS + c] =
                    at(-2) * taps[c * 3] + at(-1) * taps[c * 3 + 1] + at(0) * taps[c * 3 + 2];
            }
        }
        assert_close(&got, &want, &format!("causal m={m}"));
    }
}

/// The encoder window against the CPU's symmetric conv: prefix-less (zero left
/// neighbour), with a left neighbour, with and without padding rows, and at a
/// start position past zero.
#[test]
fn symmetric_conv_window_matches_the_cpu_encoder() {
    let model = build_encoder(2);
    let mut seed_left = 99;
    for (n_real, start_pos, with_left) in [
        (1, 0, false),
        (2, 0, false),
        (5, 0, false),
        (7, 0, true),
        (8, 0, false),
        (8, 12, true),
        (13, 0, true),
        (20, 30, true),
        (32, 0, false),
    ] {
        let left = with_left.then(|| rand_vec(&mut seed_left, HS));
        let run = run_conv_window(
            &model,
            1000 + n_real as u64,
            start_pos,
            n_real,
            left.as_deref(),
            false,
            None,
        );
        let want = cpu_symmetric_conv(&run.conv_in, &run.taps, n_real, left.as_deref());
        assert_close(
            &run.y,
            &want,
            &format!("n={n_real} start={start_pos} left={with_left}"),
        );
    }
}

/// A prefix pass leaves its last real `bx` in slot 0, and a text pass that
/// starts from it sees that row as its left neighbour; a pass without
/// `save_state` leaves slot 0 alone.
#[test]
fn prefix_state_flows_into_the_next_pass() {
    let model = build_encoder(2);
    // Prefix of 5 real rows (padded to 8), saving state.
    let prefix = run_conv_window(&model, 31, 0, 5, None, true, None);
    let bx = |j: usize| -> Vec<f32> {
        (0..HS)
            .map(|c| prefix.conv_in[j * 3 * HS + c] * prefix.conv_in[j * 3 * HS + 2 * HS + c])
            .collect()
    };
    assert_close(&prefix.slot0, &bx(4), "last real bx of the prefix");
    // Text pass of 9 rows from position 5 with that left neighbour, same taps.
    let text = run_conv_window(
        &model,
        32,
        5,
        9,
        Some(&prefix.slot0),
        false,
        Some(prefix.taps.clone()),
    );
    let want = cpu_symmetric_conv(&text.conv_in, &text.taps, 9, Some(&prefix.slot0));
    assert_close(&text.y, &want, "text after prefix");
    // Without save_state slot 0 still holds what the host put there.
    assert_close(
        &text.slot0,
        &prefix.slot0,
        "slot 0 untouched by a text pass",
    );
    // Joined equivalence: encoding prefix ++ text as one block (the CPU
    // reference for a joined sequence) differs from the two-pass result only
    // by the prefix's right neighbour, which the prefix pass (zero) and the
    // joined pass (the first text row) disagree on: pin that this is the
    // documented prefix semantics and not a leak of text into the prefix.
    let mut joined_in = prefix.conv_in[..5 * 3 * HS].to_vec();
    joined_in.extend_from_slice(&text.conv_in[..9 * 3 * HS]);
    let joined = cpu_symmetric_conv(&joined_in, &prefix.taps, 14, None);
    assert_close(
        &joined[5 * HS..],
        &want,
        "text rows are the same joined or not",
    );
    let prefix_alone = cpu_symmetric_conv(&prefix.conv_in, &prefix.taps, 5, None);
    assert_close(&prefix_alone, &prefix.y, "prefix pass on its own");
    assert!(
        joined[4 * HS..5 * HS]
            .iter()
            .zip(&prefix.y[4 * HS..5 * HS])
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "the joined sequence's prefix row 4 would see text; the prefix semantics do not"
    );
}

// ---------------------------------------------------------------------------
// Whole passes on the fake driver
// ---------------------------------------------------------------------------

fn text_state(model: &HexagonLfmModel) -> InferenceState {
    fx::fresh_state(model)
}

fn histogram_of(batches: &[String]) -> std::collections::BTreeMap<String, usize> {
    op_capture::histogram(batches)
}

#[test]
fn only_a_non_causal_dense_trunk_reports_the_encoder_pass() {
    let encoder = build_encoder(4);
    assert!(encoder.supports_bidirectional_encode());
    // The hidden-state entry points of a d1 trunk are these passes.
    assert!(encoder.supports_hidden_states());
    assert!(encoder.supports_bidirectional_hidden_states());
    assert!(encoder.supports_media_prefix());

    // The same layers with a causal config are refused.
    fx::DISTINCT_FDS.with(|d| d.set(false));
    let mut causal = encoder_spec(4);
    causal.config.is_causal = true;
    let causal = fx::build(causal);
    assert!(!causal.supports_bidirectional_encode());
    assert!(!causal.supports_hidden_states());
    assert!(!causal.supports_media_prefix());
    let mut state = text_state(&causal);
    let err = causal
        .encode_rows(EncoderInput::Tokens(&[1, 2, 3]), None, &mut state)
        .unwrap_err()
        .to_string();
    assert!(err.contains("the model is causal"), "{err}");

    // A layer with a Q gate (Qwen 3.5 style) is outside the encoder pass.
    let mut gated = fx::spec(
        |p| {
            vec![
                p.conv(),
                p.attention(fx::Extras {
                    q_gate: true,
                    ..Default::default()
                }),
            ]
        },
        false,
    );
    gated.config.is_causal = false;
    assert!(!fx::build(gated).supports_bidirectional_encode());
}

/// Op-stream facts of a pass: no LM head, the right op counts, and operand
/// shapes that encode the semantics (padding, positions, the zero mask).
#[test]
fn a_pass_emits_the_bidirectional_ops_and_no_lm_head() {
    let model = build_encoder(4); // conv, attn, conv, attn
    let so = model.scratch_offsets;
    let mut state = text_state(&model);
    op_capture::take();
    let rows = model
        .encode_rows(EncoderInput::Tokens(&[3, 1, 4, 1, 5]), None, &mut state)
        .unwrap();
    assert_eq!(rows.len(), 5 * HS);
    let batches = op_capture::take();
    let h = histogram_of(&batches);
    assert_eq!(h.get("Concat"), Some(&2), "{h:?}");
    assert_eq!(h.get("SsmConv"), Some(&2), "{h:?}");
    assert_eq!(h.get("FlashAttnExt"), Some(&2), "{h:?}");
    assert_eq!(h.get("Argmax"), None, "{h:?}");
    let text = batches.join("");
    // No LM head: the causal prefill of the same layers and rows has exactly one
    // more MulMat (the head), and nothing else differs in the matmul count.
    fx::DISTINCT_FDS.with(|d| d.set(false));
    let causal = {
        let mut spec = encoder_spec(4);
        spec.config.is_causal = true;
        fx::build(spec)
    };
    let mut cstate = text_state(&causal);
    causal
        .try_forward_prefill_chunk(&[3, 1, 4, 1, 5, 0, 0, 0], 0, &mut cstate)
        .unwrap();
    let ch = histogram_of(&op_capture::take());
    assert_eq!(
        h.get("MulMat").copied().unwrap_or(0) + 1,
        ch.get("MulMat").copied().unwrap_or(0),
        "encoder {h:?} vs causal {ch:?}"
    );
    // 5 real rows run as 8; the flash attention spans 8 slots, mask [8, 8].
    let fa: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("FlashAttnExt"))
        .collect();
    assert_eq!(fa.len(), 2);
    for line in &fa {
        assert!(line.contains("ne=[8, 8, 1, 1]"), "mask shape: {line}");
        assert!(
            line.contains(&format!("ne=[{}, 8, {}, 1]", fx::HEAD_DIM, fx::KV_HEADS)),
            "K over 8 slots: {line}"
        );
    }
    // Final norm over 8 rows, then nothing else after it.
    assert!(text.contains(&format!("ne=[{HS}, 8, 1, 1]")));
    // Host staging: positions, zero mask, zero padding rows, zero bx tail.
    let scratch = &model.scratch_buf;
    let pos: Vec<i32> = (0..8)
        .map(|i| unsafe { *(scratch.as_ptr().add(so.pos) as *const i32).add(i) })
        .collect();
    assert_eq!(pos, (0..8).collect::<Vec<i32>>());
    let mask =
        unsafe { std::slice::from_raw_parts(scratch.as_ptr().add(so.mask) as *const u16, 8 * 8) };
    for row in mask.chunks(8) {
        assert!(row[..5].iter().all(|&v| v == 0), "real slots are visible");
        assert!(
            row[5..].iter().all(|&v| v == MASK_NEG_INF),
            "padding slots are not"
        );
    }
    let act = read_f32(scratch, so.activation + 5 * HS * 4, 3 * HS);
    assert!(act.iter().all(|&v| v == 0.0), "padding rows are zero");
    let win = ConvWindow::plan(so.conv_x, so.conv_ssm_y - so.conv_x, HS, 8).unwrap();
    let tail = read_f32(scratch, win.bx + 5 * HS * 4, 4 * HS);
    assert!(tail.iter().all(|&v| v == 0.0), "bx rows 5..=8 are zero");
}

/// The encoder's attention layers are the causal prefill's, op for op: with no
/// padding the flash-attention ops of an 8-row encoder pass and an 8-row causal
/// prefill chunk are byte-identical (the semantics live in the host-built mask).
#[test]
fn attention_emission_is_the_causal_prefills() {
    let fa_lines = |causal: bool| -> Vec<String> {
        fx::DISTINCT_FDS.with(|d| d.set(false));
        let mut spec = encoder_spec(4);
        spec.config.is_causal = causal;
        let model = fx::build(spec);
        let mut state = text_state(&model);
        op_capture::take();
        let tokens: Vec<u32> = (0..8).collect();
        if causal {
            model
                .try_forward_prefill_chunk(&tokens, 0, &mut state)
                .unwrap();
        } else {
            model
                .encode_rows(EncoderInput::Tokens(&tokens), None, &mut state)
                .unwrap();
        }
        op_capture::take()
            .join("")
            .lines()
            .filter(|l| l.starts_with("FlashAttnExt"))
            .map(str::to_string)
            .collect()
    };
    let (enc, cau) = (fa_lines(false), fa_lines(true));
    assert_eq!(enc.len(), 2);
    assert_eq!(enc, cau);
}

/// Prefix then text: K/V and conv state captured to the host, restored before
/// the text pass even after something else overwrote the device, and the text
/// pass does not disturb what the next one restores.
#[test]
fn a_prefix_serves_several_text_passes() {
    let model = build_encoder(4);
    let so = model.scratch_offsets;
    let hs = HS;
    let kv_dim = fx::KV_HEADS * fx::HEAD_DIM;
    let row = kv_row_stride(model.kv_dtype, kv_dim);
    let (k_off, v_off) = match &model.layers[1] {
        HexagonLayer::Attention(a) => (a.k_offset, a.v_offset),
        _ => unreachable!(),
    };
    let kvb = &model.kv_state_buf;
    let n_prefix = 6usize;

    // Stand-in for the DSP's K/V rows of the prefix pass (the fake computes
    // nothing): a recognisable pattern in rows 0..6 of layer 1's slabs.
    let pattern: Vec<u8> = (0..n_prefix * row).map(|i| (i % 251) as u8).collect();
    unsafe {
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), kvb.as_mut_ptr().add(k_off), pattern.len());
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), kvb.as_mut_ptr().add(v_off), pattern.len());
    }
    let mut state = text_state(&model);
    let mut seed = 5;
    let prefix_rows = rand_vec(&mut seed, n_prefix * hs);
    let prefix = model
        .encode_prefix(EncoderInput::Embeddings(&prefix_rows), &mut state)
        .unwrap();
    assert_eq!(prefix.n_rows(), n_prefix);
    let inner = prefix.downcast_ref::<HexagonEncoderPrefix>().unwrap();
    assert_eq!(inner.kv.len(), 2, "two attention layers");
    assert_eq!(inner.kv[0].0, pattern, "K rows captured");
    assert_eq!(inner.kv[0].1, pattern, "V rows captured");
    assert_eq!(inner.conv_left.len(), 2, "two conv layers");

    // Something else (another pass without a prefix) scribbles over the device. A real
    // pass would also have forgotten that the prefix is resident; the host-side scribble
    // here bypasses it, so say so.
    unsafe {
        std::ptr::write_bytes(kvb.as_mut_ptr().add(k_off), 0xEE, 16 * row);
        std::ptr::write_bytes(kvb.as_mut_ptr().add(v_off), 0xEE, 16 * row);
    }
    *model.encoder_resident.lock().unwrap() = None;
    op_capture::take();
    for tokens in [
        &[1u32, 2, 3][..],
        &[9, 8, 7, 6, 5, 4, 3, 2, 1, 0][..],
        &[1, 2, 3][..],
    ] {
        let rows = model
            .encode_rows(EncoderInput::Tokens(tokens), Some(&prefix), &mut state)
            .unwrap();
        assert_eq!(rows.len(), tokens.len() * hs);
        // The prefix rows are back on the device, the text rows follow them.
        let k = unsafe { std::slice::from_raw_parts(kvb.as_ptr().add(k_off), n_prefix * row) };
        assert_eq!(k, &pattern[..], "prefix K restored");
        let after = unsafe { *kvb.as_ptr().add(k_off + n_prefix * row) };
        assert_eq!(after, 0xEE, "rows past the prefix are not rewritten");
        // Positions continue at the prefix length; attention covers prefix+text.
        let m = tokens.len().max(ENCODER_MIN_ROWS);
        let pos: Vec<i32> = (0..m)
            .map(|i| unsafe { *(model.scratch_buf.as_ptr().add(so.pos) as *const i32).add(i) })
            .collect();
        assert_eq!(
            pos,
            (n_prefix as i32..(n_prefix + m) as i32).collect::<Vec<_>>()
        );
    }
    let text = op_capture::take().join("");
    let kv_len = n_prefix + 8;
    assert!(
        text.contains(&format!("ne=[{kv_len}, 8, 1, 1]")),
        "mask [prefix + padded rows, rows]"
    );

    // A pass without a prefix zeroes the left neighbour of every conv layer.
    model
        .encode_rows(EncoderInput::Tokens(&[1, 2]), None, &mut state)
        .unwrap();
    for i in [0usize, 2] {
        let conv = conv_layer(&model, i);
        let mut slot0 = vec![1.0f32; hs];
        slab_read_slot(slab(&model, &conv), 0, &mut slot0);
        assert!(slot0.iter().all(|&v| v == 0.0));
    }
}

/// A prefix's conv left neighbour reaches slot 0 of every conv slab before the
/// pass (built by hand: the fake DSP computes no `bx`).
#[test]
fn the_conv_left_neighbour_is_restored_into_every_slab() {
    let model = build_encoder(4);
    let mut state = text_state(&model);
    let kv_dim = fx::KV_HEADS * fx::HEAD_DIM;
    let row = kv_row_stride(model.kv_dtype, kv_dim);
    let n = 4usize;
    let mut seed = 77;
    let left0 = rand_vec(&mut seed, HS);
    let left1 = rand_vec(&mut seed, HS);
    let prefix = EncoderPrefix::new(
        n,
        HexagonEncoderPrefix {
            model_id: model.encoder_id,
            // 0 is never issued by a capture, so it can never match the resident prefix
            id: 0,
            n_rows: n,
            kv: vec![(vec![1; n * row], vec![2; n * row]); 2],
            conv_left: vec![left0.clone(), left1.clone()],
        },
    );
    model
        .encode_rows(EncoderInput::Tokens(&[1, 2, 3]), Some(&prefix), &mut state)
        .unwrap();
    for (i, want) in [(0usize, &left0), (2, &left1)] {
        let mut got = vec![0.0f32; HS];
        slab_read_slot(slab(&model, &conv_layer(&model, i)), 0, &mut got);
        assert_eq!(&got, want, "conv layer {i}");
    }
}

#[test]
fn a_failed_hidden_states_pass_records_its_fault_for_the_session() {
    use crate::model::Model as _;
    let model = build_encoder(4);
    let mut state = text_state(&model);
    assert!(model.take_decode_error().is_none());
    // The trait returns bare vectors, so a refused pass comes back as zeros: the fault has to
    // be recorded or `Session` would hand the zeros to the d1 head as a real answer.
    let rows = model.hidden_states(&[1, fx::VOCAB as u32], &mut state);
    assert_eq!(rows, vec![0.0; 2 * HS]);
    assert!(model.take_decode_error().is_some(), "an invalid token");
    // an empty prefix is refused by the pass too
    let rows = model.hidden_states_with_prefix(&[], &[1, 2], &mut state);
    assert_eq!(rows, vec![0.0; 2 * HS]);
    assert!(model.take_decode_error().is_some(), "an empty prefix");
}

#[test]
fn a_pass_refuses_what_it_cannot_run() {
    let model = build_encoder(4);
    let mut state = text_state(&model);
    // Token outside the vocabulary.
    assert!(matches!(
        model.encode_rows(
            EncoderInput::Tokens(&[1, fx::VOCAB as u32]),
            None,
            &mut state
        ),
        Err(CeraError::InvalidToken { .. })
    ));
    // Empty input.
    assert!(matches!(
        model.encode_rows(EncoderInput::Tokens(&[]), None, &mut state),
        Err(CeraError::EmptyInput)
    ));
    // Ragged embedding rows.
    assert!(matches!(
        model.encode_rows(EncoderInput::Embeddings(&[0.0; HS + 1]), None, &mut state),
        Err(CeraError::Backend(_))
    ));
    // More rows than the KV cache has slots (the fixture holds 64).
    let long = vec![0u32; fx::MAX_SEQ + 1];
    assert!(matches!(
        model.encode_rows(EncoderInput::Tokens(&long), None, &mut state),
        Err(CeraError::ContextOverflow { .. })
    ));
    // A prefix made by another instance, or by another backend.
    let other = build_encoder(4);
    let prefix = other
        .encode_prefix(EncoderInput::Tokens(&[1, 2, 3]), &mut state)
        .unwrap();
    let err = model
        .encode_rows(EncoderInput::Tokens(&[1]), Some(&prefix), &mut state)
        .unwrap_err()
        .to_string();
    assert!(err.contains("another model instance"), "{err}");
    let foreign = EncoderPrefix::new(3, 42u32);
    let err = model
        .encode_rows(EncoderInput::Tokens(&[1]), Some(&foreign), &mut state)
        .unwrap_err()
        .to_string();
    assert!(err.contains("different backend"), "{err}");
}

/// The pinned op stream of a two-pass encode (a prefix, then a text after it)
/// on the tiny 4-layer model. A change to any op, operand or kernel parameter of
/// the encoder pass fails here; regenerate with `CERA_UPDATE_GOLDEN=1` after
/// reviewing the dump.
#[test]
fn encoder_op_sequence_is_pinned() {
    fx::DISTINCT_FDS.with(|d| d.set(false));
    let model = fx::build(encoder_spec(4));
    let mut state = text_state(&model);
    let mut seed = 3;
    let prefix_rows = rand_vec(&mut seed, 6 * HS);
    let prefix = model
        .encode_prefix(EncoderInput::Embeddings(&prefix_rows), &mut state)
        .unwrap();
    let text = model
        .encode_rows(
            EncoderInput::Tokens(&[3, 1, 4, 1, 5]),
            Some(&prefix),
            &mut state,
        )
        .unwrap();
    assert_eq!(text.len(), 5 * HS);
    let batches = op_capture::take();
    let joined = batches.join("--flush--\n");
    let fnv = joined.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    });
    let key = (joined.lines().filter(|l| !l.starts_with("--")).count(), fnv);
    op_capture::assert_golden("hexagon_encoder_two_pass", key, GOLDEN, &batches);
}

const GOLDEN: (usize, u64) = (115, 15225821633316841891);

/// A real LFM2 GGUF (full-size scratch: hidden 1024, 16 layers) through both
/// encoder passes on the fake device, with the config flipped to non-causal:
/// the real tensor names and shapes, a 300-row prefix, and a 512-row text, the
/// row limit. Plumbing only (the fake DSP computes nothing). Skips without the
/// fixture; fails under `CERA_REQUIRE_MODEL`.
#[cfg(feature = "mmap")]
#[test]
fn real_lfm2_gguf_runs_both_passes_at_full_size_on_the_fake_device() {
    let home = std::env::var("HOME").expect("HOME unset");
    let path =
        std::path::PathBuf::from(home).join(".leap/models/LFM2-350M-Q4_0/LFM2-350M-Q4_0.gguf");
    if !crate::model::transformer::require_model_or_skip(&path) {
        return;
    }
    let (driver, device) = op_capture::fresh_device();
    let backend = Backend::with_device(driver, device, fx::golden_knobs());
    let mut model =
        HexagonLfmModel::from_gguf_on(backend, GgufFile::open(&path).unwrap(), 1024).unwrap();
    // The GGUF is a causal LFM2; the pass is selected by the config flag alone.
    model.config.is_causal = false;
    assert!(model.supports_bidirectional_encode());
    let hs = model.config.hidden_size;
    let n_conv = model
        .layers
        .iter()
        .filter(|l| matches!(l, HexagonLayer::Conv(_)))
        .count();
    let n_attn = model.layers.len() - n_conv;
    let mut state = fx::fresh_state(&model);
    let mut seed = 21;

    op_capture::take();
    let emb = rand_vec(&mut seed, 300 * hs);
    let prefix = model
        .encode_prefix(EncoderInput::Embeddings(&emb), &mut state)
        .unwrap();
    assert_eq!(prefix.n_rows(), 300);
    let h = histogram_of(&op_capture::take());
    assert_eq!(h.get("Concat"), Some(&n_conv), "{h:?}");
    assert_eq!(h.get("FlashAttnExt"), Some(&n_attn), "{h:?}");

    for n in [512usize, 5] {
        let tokens: Vec<u32> = (0..n as u32).map(|i| 100 + i % 50).collect();
        let rows = model
            .encode_rows(EncoderInput::Tokens(&tokens), Some(&prefix), &mut state)
            .unwrap();
        assert_eq!(rows.len(), n * hs);
        let batches = op_capture::take();
        let text = batches.join("");
        let m = n.max(ENCODER_MIN_ROWS);
        // Window `[m + 2, hidden]`, mask `[prefix + text, m]`, K over prefix + text.
        assert!(
            text.contains(&format!("ne=[{}, {hs}, 1, 1]", m + 2)),
            "window n={n}"
        );
        let kv_len = 300 + m;
        let fa = text
            .lines()
            .find(|l| l.starts_with("FlashAttnExt"))
            .expect("a flash attention op");
        assert!(
            fa.contains(&format!("ne=[{kv_len}, {m}, 1, 1]")),
            "mask n={n}: {fa}"
        );
    }
    // No prefix: a full chunk is one pass, one row more is a layer-outer pass of
    // two chunks (no row limit any more), and both run.
    for n in [PREFILL_MAX_ROWS, PREFILL_MAX_ROWS + 1] {
        let tokens: Vec<u32> = (0..n as u32).map(|i| 100 + i % 50).collect();
        op_capture::take();
        let rows = model
            .encode_rows(EncoderInput::Tokens(&tokens), None, &mut state)
            .unwrap();
        assert_eq!(rows.len(), n * hs);
        let h = histogram_of(&op_capture::take());
        // Every layer's sweeps ran: one window per conv layer and chunk, one
        // flash attention per attention layer and chunk.
        let chunks = if n > PREFILL_MAX_ROWS { 2 } else { 1 };
        assert_eq!(h.get("Concat"), Some(&(n_conv * chunks)), "n={n}: {h:?}");
        assert_eq!(
            h.get("FlashAttnExt"),
            Some(&(n_attn * chunks)),
            "n={n}: {h:?}"
        );
    }
}

/// A decision model is loaded without its LM head: the encoder pass does not need it, and
/// the causal paths that would read it refuse with a typed error instead of dispatching a
/// weight that was never uploaded.
#[test]
fn an_unplanned_lm_head_serves_the_encoder_and_refuses_logits() {
    let mut model = build_encoder(4);
    model.lm_head = HexagonWeight::unplanned(HS);
    assert!(!model.lm_head.is_planned());
    let mut state = text_state(&model);
    let rows = model
        .encode_rows(EncoderInput::Tokens(&[3, 1, 4, 1, 5]), None, &mut state)
        .unwrap();
    assert_eq!(rows.len(), 5 * HS);
    let e = model
        .try_forward_prefill_chunk(&[3, 1, 4, 1, 5, 0, 0, 0], 0, &mut state)
        .unwrap_err();
    assert!(e.to_string().contains("bidirectional encoding only"), "{e}");
}

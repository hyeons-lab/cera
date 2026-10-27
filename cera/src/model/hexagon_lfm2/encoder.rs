//! Bidirectional (non-causal) encoder pass of an LFM2 trunk on the Hexagon NPU
//! (the "decision model": `lfm2.attention.causal = false`).
//!
//! One pass runs every layer over a whole row block with no LM head and returns
//! the final-normed rows. Semantics, row for row the CPU reference
//! (`LfmModel::encode_pass`):
//!
//! * **Attention** layers: every row attends every row (an additive mask that
//!   is zero over the real slots, no causal triangle) plus the prefix K/V when
//!   there is one. RoPE positions continue at the prefix length. The layer is
//!   emitted by the causal prefill's `emit_attention_block`, unchanged: only the
//!   host-built mask differs.
//! * **Gated short-conv** layers: the symmetric centred 3-tap kernel,
//!   `out = c * (prev * w0 + curr * w1 + next * w2)` with `bx = b * x`. The
//!   left neighbour of the first row is the prefix's last `bx` (zero without a
//!   prefix); the right neighbour of the last row is zero.
//! * A **prefix** (media embeddings) is encoded on its own and leaves K/V per
//!   attention layer and the last `bx` row per conv layer. Those are copied to
//!   host memory ([`HexagonEncoderPrefix`]) and written back before every later
//!   pass, so one prefix serves any number of text passes, and a pass without a
//!   prefix (which overwrites the device KV from position 0) cannot corrupt it.
//!
//! # How the symmetric conv reuses the causal SsmConv op
//!
//! `SsmConv` computes `y[t] = w0 * x[t] + w1 * x[t + 1] + w2 * x[t + 2]` over a
//! channel-major window `x` of `m + 2` positions. The causal prefill builds
//! `x = [s0, s1, bx_0 .. bx_{m-1}]` (the two state rows, then the new rows), so
//! `y[t]` sees `bx[t-2], bx[t-1], bx[t]`. The centred kernel needs
//! `bx[t-1], bx[t], bx[t+1]`, i.e. the same window shifted one row:
//!
//! ```text
//! x = [left, bx_0, bx_1, .. bx_{m-1}, 0]      (m + 2 positions)
//!      s0    s1    <---- s1 side: m rows ---->
//! ```
//!
//! so `s0 = [left, bx_0]` is a two-slot state slab (exactly the shape the
//! causal path feeds the DSP's transposed-concat worker) and the `s1` side is
//! the bx buffer read from its second row, whose last row is the zero right
//! neighbour. Nothing but the window changes; the op, the tap order and the
//! `[3, C]` GGUF tap layout are the causal path's.
//!
//! # Padding to the 8-row minimum
//!
//! The DSP matmuls of 1 to 7 rows take an HVX kernel that is measurably less
//! precise than the HMX one (cosine 0.92 to 0.99 against the CPU), so a pass
//! always runs `m = max(n, 8)` rows. The `m - n` padding rows are zero
//! embeddings that sit after the real rows and influence nothing:
//!
//! * attention runs over `start_pos + m` KV slots, the shape regime of a causal
//!   prefill chunk (never fewer slots than queries), and the padding slots'
//!   mask columns are `-inf` for every query, so no row reads padding K/V
//!   (it is still written, into the slots past the real rows);
//! * the conv `bx` of a padding row is never computed (the multiply covers the
//!   `n` real rows) and its buffer rows are zero from the host, so the last
//!   real row's right neighbour is exactly zero at every layer;
//! * every other op is row-independent.
//!
//! # Long passes: the layer-outer driver
//!
//! One chunk is at most [`ENCODER_CHUNK_ROWS`] (512) rows, the activation
//! scratch's capacity. A pass of up to that many rows (the common text case) is
//! the single-chunk emission above, unchanged. Longer ones run **layer-outer**:
//! a bidirectional pass cannot be chunked the way causal prefill is (chunk-outer,
//! layer-inner), because a later chunk's rows change an earlier chunk's attention
//! and conv outputs. So every layer is swept over all chunks before the next
//! layer starts, with the residual stream of all `N` rows held in a *long
//! buffer* (a second rpcmem buffer, allocated on the first long pass, see
//! [`LongLayout`]):
//!
//! * the rows are cut into `ceil(N / 512)` balanced chunks (sizes differ by at
//!   most one, never under 8 rows, so a long pass has no padding);
//! * per chunk and layer, the residual rows are copied from the long buffer into
//!   the scratch's `cur_act`, the unchanged per-block emitters run on them, and
//!   the layer output is copied back (the long buffer is updated in place:
//!   a chunk only ever reads and writes its own residual rows within a layer);
//! * **attention** layers take two sweeps. Sweep 1 ([`AttnStage::Front`]): norm,
//!   QKV, Q/K norm, RoPE and the K/V write of every chunk, with the roped Q kept
//!   in the long buffer. Sweep 2 ([`AttnStage::Back`]): flash attention of each
//!   chunk's Q over all `start_pos + N` slots with an all-zero mask (there is no
//!   padding), output projection, residual and FFN;
//! * **conv** layers take two sweeps. Sweep 1: norm, `in_proj` (written straight
//!   into the long buffer's `[N, 3 * hidden]` store) and `bx = b * x` for every
//!   chunk. Sweep 2: each chunk's window is the layer's two-slot state slab
//!   `[left, bx_0]` followed by `bx` rows `1..s` and the row after the chunk, so
//!   its neighbours are the adjacent chunks' `bx` rows (slot 0 holds the prefix's
//!   last `bx`, or zero, for the first chunk; the row past the last chunk is a
//!   zero row), then the conv, the gate with the stored `c` third, `out_proj`,
//!   residual and FFN;
//! * the final norm runs per chunk in place and the host reads all `N` rows.
//!
//! Positions of every chunk are staged on the host once, in the long buffer, so
//! no host write ever races a queued op. The prefix semantics are the
//! single-chunk pass's: a prefix pass leaves K/V of all its rows and the last
//! `bx` per conv layer; a text pass restores them, writes only slots at or after
//! the prefix, and attends over the prefix whatever its length. At most
//! [`ENCODER_MAX_PASS_ROWS`] rows are taken per pass (and the KV cache must hold
//! `start_pos + N` slots); both limits are typed [`CeraError::ContextOverflow`]s.
//!

use super::*;
use crate::model::{EncoderInput, EncoderPrefix};
use std::sync::atomic::AtomicU64;

/// Rows a pass is padded up to (see the module docs).
pub(super) const ENCODER_MIN_ROWS: usize = 8;

/// Rows of one chunk of a layer-outer pass: the activation scratch's capacity.
pub(super) const ENCODER_CHUNK_ROWS: usize = PREFILL_MAX_ROWS;

/// Most rows one pass takes. The long buffer costs `N * (4 * hidden + 16 *
/// hidden)` bytes (335 MiB at 16384 rows of a hidden-1024 model), and 16384 is
/// the decision model's context length, so nothing a model accepts is refused
/// for this bound; a model with a longer window gets a typed
/// [`CeraError::ContextOverflow`] naming it.
pub(super) const ENCODER_MAX_PASS_ROWS: usize = 16384;

/// A long buffer larger than this is released at the end of the pass that
/// needed it instead of being kept for the model's lifetime: a few hundred
/// megabytes of pinned shared memory idling between requests is the wrong trade
/// on a phone, and the allocation is negligible next to a pass over that many
/// rows.
const LONG_BUFFER_KEEP_BYTES: usize = 64 << 20;

static NEXT_ENCODER_MODEL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ENCODER_PREFIX_ID: AtomicU64 = AtomicU64::new(1);

/// An opt-out knob (`0` or `false` turns it off), read once per process.
fn knob_on(cell: &'static std::sync::OnceLock<bool>, name: &str) -> bool {
    *cell.get_or_init(|| {
        !std::env::var(name).is_ok_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
    })
}

/// `CERA_HEXAGON_PIPELINE` (default on): the capped flushes of an encoder pass (and a head
/// request) do not wait for the DSP between batches (see
/// [`HexagonQueueSession::set_pipelined`]).
pub(super) fn pipeline_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    knob_on(&ON, "CERA_HEXAGON_PIPELINE")
}

/// `CERA_HEXAGON_RESIDENT_PREFIX` (default on): skip restoring a prefix the device holds.
fn resident_prefix_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    knob_on(&ON, "CERA_HEXAGON_RESIDENT_PREFIX")
}

/// Rows from which a pass naps between polls for its responses instead of spinning
/// (`PassWait::Auto`). On the S25 Ultra a host that sleeps through a short pass makes it
/// markedly slower (a 200-row question 38 ms spinning, 61 to 66 ms napping or blocking;
/// the DSP side seems to clock down when the host idles), while a pass that keeps the DSP
/// busy for hundreds of milliseconds loses about 8% (the 2672-row request 468 ms to 506 ms)
/// and gives back 80% of the core it would have burned.
const NAP_MIN_ROWS: usize = 1024;

/// How an encoder pass waits for its batches' responses (`CERA_HEXAGON_ENCODER_WAIT`, read
/// once): `auto` (the default) spins for a pass under [`NAP_MIN_ROWS`] rows and naps for
/// one of that many or more, `spin` busy-polls, `block` sleeps in the kernel on every read,
/// `nap` polls and sleeps between polls. `CERA_HEXAGON_NAP_US` is the nap (default 250).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PassWait {
    Auto(u64),
    Spin,
    Block,
    Nap(u64),
}

impl PassWait {
    pub(super) fn get() -> Self {
        static MODE: std::sync::OnceLock<PassWait> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            let nap = std::env::var("CERA_HEXAGON_NAP_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(250);
            match std::env::var("CERA_HEXAGON_ENCODER_WAIT").as_deref() {
                Ok("spin") => PassWait::Spin,
                Ok("block") => PassWait::Block,
                Ok("nap") => PassWait::Nap(nap),
                _ => PassWait::Auto(nap),
            }
        })
    }

    /// The wait for a pass of `rows` rows.
    pub(super) fn for_rows(rows: usize) -> Self {
        match Self::get() {
            PassWait::Auto(nap) if rows >= NAP_MIN_ROWS || crate::backend::hexagon::low_cpu() => {
                PassWait::Nap(nap)
            }
            PassWait::Auto(_) => PassWait::Spin,
            other => other,
        }
    }

    /// Set the session's wait for the start of a pass, returning what to restore.
    pub(super) fn begin(self, session: &mut HexagonQueueSession) -> (bool, u64) {
        let nap = if let PassWait::Nap(us) = self { us } else { 0 };
        (
            session.set_blocking_wait(matches!(self, PassWait::Block | PassWait::Nap(_))),
            session.set_nap_wait(nap),
        )
    }

    pub(super) fn restore(session: &mut HexagonQueueSession, (blocking, nap): (bool, u64)) {
        session.set_blocking_wait(blocking);
        session.set_nap_wait(nap);
    }
}

/// Trace line for an rpcmem allocation made on a request's path (under
/// `CERA_HEXAGON_TRACE_PASS`).
pub(super) fn trace_alloc(what: &str, bytes: usize, started: std::time::Instant) {
    if std::env::var_os("CERA_HEXAGON_TRACE_PASS").is_some() {
        eprintln!(
            "cera-hexagon: alloc {what}: {} MiB in {} us",
            bytes >> 20,
            started.elapsed().as_micros()
        );
    }
}

/// `CERA_HEXAGON_TRACE_PASS` (opt-in, read once): one stderr line per encoder pass with
/// where its wall time went, for finding what a request spends on the host.
pub(super) struct PassTrace {
    start: std::time::Instant,
    staged: Option<std::time::Instant>,
    ran: Option<std::time::Instant>,
    wait0: u64,
    seq0: u64,
}

impl PassTrace {
    pub(super) fn start(session: &HexagonQueueSession) -> Option<Self> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        ON.get_or_init(|| std::env::var_os("CERA_HEXAGON_TRACE_PASS").is_some())
            .then(|| Self {
                start: std::time::Instant::now(),
                staged: None,
                ran: None,
                wait0: session.wait_ns(),
                seq0: session.dispatch_attempts(),
            })
    }

    pub(super) fn staged(this: &mut Option<Self>) {
        if let Some(t) = this {
            t.staged = Some(std::time::Instant::now());
        }
    }

    pub(super) fn ran(this: &mut Option<Self>) {
        if let Some(t) = this {
            t.ran = Some(std::time::Instant::now());
        }
    }

    pub(super) fn finish(
        this: Option<Self>,
        session: &HexagonQueueSession,
        what: std::fmt::Arguments<'_>,
    ) {
        let Some(t) = this else { return };
        let end = std::time::Instant::now();
        let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros();
        let staged = t.staged.unwrap_or(t.start);
        let ran = t.ran.unwrap_or(end);
        eprintln!(
            "cera-hexagon: pass {what} stage_us={} run_us={} (wait_us={} flushes={}) read_us={} total_us={}",
            us(t.start, staged),
            us(staged, ran),
            (session.wait_ns() - t.wait0) / 1000,
            session.dispatch_attempts() - t.seq0,
            us(ran, end),
            us(t.start, end),
        );
    }
}

/// A process-unique id for a model instance, so a prefix cannot be replayed
/// into a model whose weights it was not computed with.
pub(super) fn next_encoder_model_id() -> u64 {
    NEXT_ENCODER_MODEL_ID.fetch_add(1, Ordering::Relaxed)
}

/// Geometry of one single-chunk encoder pass over a row block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EncoderShape {
    /// Absolute position of the first row (the prefix length, or 0).
    pub start_pos: usize,
    /// Real rows.
    pub n_real: usize,
    /// Rows the DSP runs: `n_real` padded up to [`ENCODER_MIN_ROWS`].
    pub m: usize,
}

impl EncoderShape {
    /// Shape for `n_real` rows after `start_pos` rows, or the typed error for a
    /// block that does not fit.
    pub(super) fn plan(
        start_pos: usize,
        n_real: usize,
        max_seq_len: usize,
    ) -> Result<Self, CeraError> {
        if n_real == 0 {
            return Err(CeraError::EmptyInput);
        }
        if n_real > PREFILL_MAX_ROWS {
            // Callers route such a block to the layer-outer driver.
            return Err(CeraError::Backend(format!(
                "a single-chunk encoder pass takes at most {PREFILL_MAX_ROWS} rows, not {n_real}"
            )));
        }
        let m = n_real.max(ENCODER_MIN_ROWS);
        // The padding rows also write their K/V, so the cache needs their slots.
        let slots = start_pos + m;
        if slots > max_seq_len {
            return Err(CeraError::ContextOverflow {
                max_seq_len: max_seq_len as u32,
                by: (slots - max_seq_len) as u32,
            });
        }
        Ok(Self {
            start_pos,
            n_real,
            m,
        })
    }

    /// KV slots the flash attention spans: the prefix and all `m` rows.
    pub(super) fn kv_len(&self) -> usize {
        self.start_pos + self.m
    }

    /// Leading slots a query may read: the prefix and the real rows. The mask
    /// hides the padding slots after them.
    pub(super) fn visible(&self) -> usize {
        self.start_pos + self.n_real
    }
}

/// How a pass runs: as one padded chunk, or layer-outer over several.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum PassPlan {
    Single(EncoderShape),
    Long(LongPlan),
}

/// The chunks of a layer-outer pass.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct LongPlan {
    /// Absolute position of the first row (the prefix length, or 0).
    pub start_pos: usize,
    /// Rows of the pass.
    pub n: usize,
    /// `(first row, rows)` per chunk, in order.
    pub chunks: Vec<(usize, usize)>,
}

/// Cut `n` rows into `ceil(n / chunk_rows)` chunks whose sizes differ by at most
/// one. Balanced rather than greedy so that no chunk is ever tiny (the DSP's
/// matmuls under 8 rows take a less precise kernel): every chunk of an `n >
/// chunk_rows` split holds more than half a chunk.
pub(super) fn balanced_chunks(n: usize, chunk_rows: usize) -> Vec<(usize, usize)> {
    let k = n.div_ceil(chunk_rows).max(1);
    let (base, extra) = (n / k, n % k);
    let mut first = 0;
    (0..k)
        .map(|i| {
            let rows = base + usize::from(i < extra);
            let chunk = (first, rows);
            first += rows;
            chunk
        })
        .collect()
}

/// Plan a pass of `n_real` rows after `start_pos` rows, with chunks of at most
/// `chunk_rows` (production passes [`ENCODER_CHUNK_ROWS`]; the planner tests use a small one so
/// that tiny inputs take the layer-outer path).
pub(super) fn plan_pass(
    start_pos: usize,
    n_real: usize,
    max_seq_len: usize,
    chunk_rows: usize,
) -> Result<PassPlan, CeraError> {
    if n_real == 0 {
        return Err(CeraError::EmptyInput);
    }
    if !(2 * ENCODER_MIN_ROWS..=PREFILL_MAX_ROWS).contains(&chunk_rows) {
        return Err(CeraError::Backend(format!(
            "encoder chunk of {chunk_rows} rows is outside {}..={PREFILL_MAX_ROWS}",
            2 * ENCODER_MIN_ROWS
        )));
    }
    if n_real <= chunk_rows {
        return EncoderShape::plan(start_pos, n_real, max_seq_len).map(PassPlan::Single);
    }
    if n_real > ENCODER_MAX_PASS_ROWS {
        return Err(CeraError::ContextOverflow {
            max_seq_len: ENCODER_MAX_PASS_ROWS as u32,
            by: (n_real - ENCODER_MAX_PASS_ROWS) as u32,
        });
    }
    let slots = start_pos + n_real;
    if slots > max_seq_len {
        return Err(CeraError::ContextOverflow {
            max_seq_len: max_seq_len as u32,
            by: (slots - max_seq_len) as u32,
        });
    }
    Ok(PassPlan::Long(LongPlan {
        start_pos,
        n: n_real,
        chunks: balanced_chunks(n_real, chunk_rows),
    }))
}

/// Fill the `[kv_len, m]` bidirectional attention mask (one `kv_len` row per
/// query): every query attends every slot below `visible`, so those mask entries
/// are zero (no causal triangle, no sliding window), and the `kv_len - visible`
/// padding slots after them are `-inf` for every query. Kept as a function so the
/// one place that defines the semantics is tested against the CPU reference.
pub(super) fn fill_bidirectional_mask(mask: &mut [u16], m: usize, kv_len: usize, visible: usize) {
    debug_assert!(mask.len() >= kv_len * m && visible <= kv_len);
    for row in mask[..kv_len * m].chunks_mut(kv_len) {
        row[..visible].fill(0x0000);
        row[visible..].fill(MASK_NEG_INF);
    }
}

/// Where the symmetric conv's two buffers live inside the `conv_x` scratch
/// (which is `(512 + 2) * 3 * hidden` floats, so both fit with room to spare).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ConvWindow {
    /// Row-major `[m + 1, hidden]` `bx` rows: `bx_0 ..` for the real rows,
    /// then zero rows through index `m` (the right neighbour).
    pub bx: usize,
    /// Channel-major `[m + 2, hidden]` window `SsmConv` reads.
    pub window: usize,
}

impl ConvWindow {
    /// Lay both buffers out in `conv_x`, `avail` bytes from `conv_x`.
    pub(super) fn plan(
        conv_x: usize,
        avail: usize,
        hidden: usize,
        m: usize,
    ) -> Result<Self, CeraError> {
        let row = hidden * 4;
        let window = conv_x + ((m + 1) * row).next_multiple_of(256);
        let end = window + (m + 2) * row;
        if end > conv_x + avail {
            return Err(CeraError::Backend(format!(
                "encoder conv window needs {} bytes of conv scratch, {avail} available",
                end - conv_x
            )));
        }
        Ok(Self { bx: conv_x, window })
    }
}

/// Write `values` into slot `slot` (0 or 1) of a channel-interleaved `[C, 2]`
/// conv state slab: slot `t` of channel `c` sits at `c * 8 + t * 4`.
pub(super) fn slab_write_slot(slab: &mut [u8], slot: usize, values: &[f32]) {
    debug_assert!(slot < 2 && slab.len() >= values.len() * 8);
    for (c, v) in values.iter().enumerate() {
        let at = c * 8 + slot * 4;
        slab[at..at + 4].copy_from_slice(&v.to_ne_bytes());
    }
}

/// Read slot `slot` of a `[C, 2]` conv state slab into `out`.
pub(super) fn slab_read_slot(slab: &[u8], slot: usize, out: &mut [f32]) {
    debug_assert!(slot < 2 && slab.len() >= out.len() * 8);
    for (c, v) in out.iter_mut().enumerate() {
        let at = c * 8 + slot * 4;
        *v = f32::from_ne_bytes([slab[at], slab[at + 1], slab[at + 2], slab[at + 3]]);
    }
}

// ---------------------------------------------------------------------------
// The long buffer of the layer-outer driver
// ---------------------------------------------------------------------------

/// Offsets inside the long buffer of a layer-outer pass over `n` rows.
///
/// ```text
/// res    [n, hidden] f32      residual stream, updated in place layer by layer
/// pos    chunk slots of i32   absolute positions of each chunk (256-aligned slots)
/// ucin   [n, 3 hidden] f32    conv layers: the in_proj output (b | c | x per row)
/// uq     = ucin               attention layers: roped Q rows [n, q_dim]
/// ubx    [n + 1, hidden] f32  conv layers: bx rows; row n is the zero row
/// ```
///
/// `uq` and `ucin` are the same region (an attention layer and a conv layer
/// never need theirs at once) and `ubx` sits past the larger of the two, so the
/// zero row at `ubx[n]` survives the attention layers.
///
/// Cost per row, hidden `H`, `q_dim` `Q`: `4 H` (res) `+ 4 * max(3 H, Q)` (the
/// shared store) `+ 4 H` (bx) bytes with conv layers, `4 H + 4 Q` without. For
/// the decision model (`H = Q = 1024`): 20 KiB per row, so 55 MiB at 2816 rows,
/// 107 MiB at 5500 and 320 MiB at 16384, plus the position slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LongLayout {
    pub res: usize,
    pub pos: usize,
    /// Bytes between the position slots of consecutive chunks.
    pub pos_stride: usize,
    pub ucin: usize,
    pub uq: usize,
    pub ubx: usize,
    pub total: usize,
}

impl LongLayout {
    pub(super) fn plan(
        n: usize,
        n_chunks: usize,
        chunk_rows: usize,
        hidden: usize,
        q_dim: usize,
        has_conv: bool,
    ) -> Result<Self, CeraError> {
        let align = |x: usize| x.next_multiple_of(4096);
        let overflow = || CeraError::Backend(format!("long buffer for {n} rows overflows usize"));
        let row = hidden.checked_mul(4).ok_or_else(overflow)?;
        let res = 0;
        let pos = align(n.checked_mul(row).ok_or_else(overflow)?);
        let pos_stride = (chunk_rows * 4).next_multiple_of(256);
        let ucin = align(pos + n_chunks * pos_stride);
        let cin_bytes = if has_conv { n * 3 * row } else { 0 };
        let store = cin_bytes.max(n.checked_mul(q_dim * 4).ok_or_else(overflow)?);
        let ubx = align(ucin + store);
        let total = align(ubx + if has_conv { (n + 1) * row } else { 0 });
        Ok(Self {
            res,
            pos,
            pos_stride,
            ucin,
            uq: ucin,
            ubx,
            total,
        })
    }
}

/// The long buffer and its layout.
pub(super) struct LongBuf<'a> {
    pub buf: &'a RpcmemBuffer,
    pub lay: LongLayout,
}

/// One chunk of a layer-outer pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LongChunk {
    pub idx: usize,
    /// First row of the chunk within the pass.
    pub first: usize,
    pub rows: usize,
    pub last: bool,
}

impl LongPlan {
    pub(super) fn long_chunks(&self) -> Vec<LongChunk> {
        let k = self.chunks.len();
        self.chunks
            .iter()
            .enumerate()
            .map(|(idx, &(first, rows))| LongChunk {
                idx,
                first,
                rows,
                last: idx + 1 == k,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Attention staging
// ---------------------------------------------------------------------------

/// Which part of an attention layer an emission covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttnStage {
    /// The whole layer (decode, causal prefill, the single-chunk encoder pass).
    Whole,
    /// Block norm through the K/V write.
    Front,
    /// Flash attention through the FFN.
    Back,
}

/// Operands of an attention emission that default to the scratch buffer's own
/// regions. The layer-outer driver redirects them; the defaults emit exactly
/// what the unsplit block always has.
#[derive(Clone, Copy, Default)]
pub(super) struct AttnRoute<'a> {
    /// Position vector of RoPE and the K/V write (`None` = `so.pos` in scratch).
    pub pos: Option<(&'a RpcmemBuffer, usize)>,
    /// Roped Q rows the flash attention reads (`None` = `so.q` in scratch).
    pub q: Option<(&'a RpcmemBuffer, usize)>,
    /// KV slots the flash attention spans (`None` = this block's `first + rows`).
    pub kv_len: Option<usize>,
}

/// What an encoder prefix leaves behind, in host memory: the K/V rows of every
/// attention layer and the last `bx` row of every conv layer. Opaque to
/// callers (it travels inside [`EncoderPrefix`]).
pub(super) struct HexagonEncoderPrefix {
    /// The model instance that computed it.
    pub(super) model_id: u64,
    /// Process-unique id of this capture, so the model can tell that the device still
    /// holds exactly this prefix (see [`HexagonLfmModel::restore_encoder_state`]).
    pub(super) id: u64,
    pub(super) n_rows: usize,
    /// `(K, V)` bytes of rows `0..n_rows` per attention layer, layer order.
    pub(super) kv: Vec<(Vec<u8>, Vec<u8>)>,
    /// The last `bx` row (`hidden` floats) per conv layer, layer order.
    pub(super) conv_left: Vec<Vec<f32>>,
}

/// What a pass returns besides the rows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Capture {
    /// Return the final-normed rows.
    Rows,
    /// Keep a prefix; no final norm, no rows.
    Prefix,
}

pub(super) struct EncoderOutput {
    pub rows: Vec<f32>,
    pub prefix: Option<HexagonEncoderPrefix>,
}

impl HexagonLfmModel {
    /// Why this model cannot run the bidirectional pass, or `None` when it can.
    /// The pass is the decision model's trunk: a non-causal LFM2 with dense FFNs
    /// and none of the dense-path extras the shared emitters also serve.
    pub(super) fn encoder_blocker(&self) -> Option<&'static str> {
        if self.config.is_causal {
            return Some("the model is causal");
        }
        if self.config.conv_kernel_size.unwrap_or(3) != 3 {
            return Some("the symmetric conv kernel has 3 taps");
        }
        if self.has_deltanet || self.has_moe {
            return Some("DeltaNet and routed-expert layers have no bidirectional pass");
        }
        let d = &self.dense;
        if d.post_norm
            || d.loop_norm_interval.is_some()
            || d.attn_temp.is_some()
            || d.swa_window.is_some()
            || d.mask_swa.is_some()
        {
            return Some("the model uses dense-path semantics the encoder pass does not cover");
        }
        let plain_layers = self.layers.iter().all(|l| match l {
            HexagonLayer::Attention(a) => {
                !a.has_q_gate && !a.swa && matches!(a.ffn, HexagonFfn::Dense(_))
            }
            HexagonLayer::Conv(c) => matches!(c.ffn, HexagonFfn::Dense(_)),
            HexagonLayer::DeltaNet(_) => false,
        });
        if !plain_layers {
            return Some("a layer needs a gate, a window or a routed FFN");
        }
        None
    }

    /// One bidirectional pass: the rows of `input` after `prefix` (or from
    /// position 0), returning the final-normed rows or a new prefix.
    pub(super) fn encoder_pass(
        &self,
        input: EncoderInput<'_>,
        prefix: Option<&HexagonEncoderPrefix>,
        capture: Capture,
    ) -> Result<EncoderOutput, CeraError> {
        self.encoder_pass_chunked(input, prefix, capture, ENCODER_CHUNK_ROWS)
    }

    /// [`Self::encoder_pass`] with a chunk size: a pass of more rows than
    /// `chunk_rows` runs layer-outer. The planner tests use a small one.
    pub(super) fn encoder_pass_chunked(
        &self,
        input: EncoderInput<'_>,
        prefix: Option<&HexagonEncoderPrefix>,
        capture: Capture,
        chunk_rows: usize,
    ) -> Result<EncoderOutput, CeraError> {
        if let Some(reason) = self.encoder_blocker() {
            return Err(CeraError::Backend(format!(
                "this model has no NPU bidirectional encoder pass: {reason}"
            )));
        }
        let hs = self.config.hidden_size;
        let n_real = input.rows(hs);
        match input {
            EncoderInput::Tokens(tokens) => {
                let vocab = self.config.vocab_size;
                if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= vocab) {
                    return Err(CeraError::InvalidToken {
                        id: bad,
                        vocab_size: vocab as u32,
                    });
                }
            }
            EncoderInput::Embeddings(rows) => {
                if hs == 0 || !rows.len().is_multiple_of(hs) {
                    return Err(CeraError::Backend(format!(
                        "embedding rows of {} floats do not divide into {hs}-wide rows",
                        rows.len()
                    )));
                }
            }
        }
        let start_pos = prefix.map_or(0, |p| p.n_rows);
        let plan = plan_pass(start_pos, n_real, self.config.max_seq_len, chunk_rows)?;
        if let Some(prefix) = prefix {
            self.check_encoder_prefix(prefix)?;
        }

        let mut device = self.lock_device();
        self.ensure_state_intact(&device)?;
        match plan {
            PassPlan::Single(shape) => {
                self.encoder_pass_single(&mut device, input, prefix, capture, &shape)
            }
            PassPlan::Long(plan) => {
                self.encoder_pass_long(&mut device, input, prefix, capture, &plan)
            }
        }
    }

    /// Queue-session settings of a pass whose smallest chunk is `min_rows`: the
    /// same flush policy as the causal prefill chunk (see `PREFILL_MAX_ROWS` and
    /// `MAX_TENSORS_PER_FLUSH` for why). The caller resets it on every path.
    fn set_encoder_flush_policy(&self, session: &mut HexagonQueueSession, min_rows: usize) {
        session.set_max_ops_per_flush(if min_rows < SMALL_M_FLUSH_CAP_ROWS {
            Some(MAX_OPS_PER_FLUSH)
        } else {
            None
        });
        session.set_max_tensors_per_flush(self.encoder_batch_tensor_cap);
    }

    /// The single-chunk pass: every row in one padded block (the pass up to
    /// `ENCODER_CHUNK_ROWS` rows, and the one a short text after a long prefix
    /// takes).
    fn encoder_pass_single(
        &self,
        device: &mut HexagonDevice,
        input: EncoderInput<'_>,
        prefix: Option<&HexagonEncoderPrefix>,
        capture: Capture,
        shape: &EncoderShape,
    ) -> Result<EncoderOutput, CeraError> {
        let hs = self.config.hidden_size;
        let mut trace = PassTrace::start(device.queue_session_mut());
        let EncoderShape {
            start_pos,
            n_real,
            m,
        } = *shape;
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let has_conv = self
            .layers
            .iter()
            .any(|l| matches!(l, HexagonLayer::Conv(_)));
        let window = if has_conv {
            Some(ConvWindow::plan(
                so.conv_x,
                so.conv_ssm_y - so.conv_x,
                hs,
                m,
            )?)
        } else {
            None
        };

        // Host staging, all before the first op reaches the DSP.
        unsafe {
            let act = scratch.as_mut_ptr().add(so.activation);
            match input {
                EncoderInput::Tokens(tokens) => {
                    for (i, &t) in tokens.iter().enumerate() {
                        let row =
                            std::slice::from_raw_parts_mut(act.add(i * hs * 4) as *mut f32, hs);
                        self.token_embd.row_into(t as usize, row);
                    }
                }
                EncoderInput::Embeddings(rows) => {
                    std::ptr::copy_nonoverlapping(rows.as_ptr() as *const u8, act, n_real * hs * 4);
                }
            }
            // Padding rows are zero embeddings, never stale scratch.
            std::ptr::write_bytes(act.add(n_real * hs * 4), 0, (m - n_real) * hs * 4);
            let pos_slice =
                std::slice::from_raw_parts_mut(scratch.as_mut_ptr().add(so.pos) as *mut i32, m);
            for (i, slot) in pos_slice.iter_mut().enumerate() {
                *slot = (start_pos + i) as i32;
            }
            let mask = std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.mask) as *mut u16,
                shape.kv_len() * m,
            );
            fill_bidirectional_mask(mask, m, shape.kv_len(), shape.visible());
            if let Some(win) = window {
                // bx rows past the real ones, through the right-neighbour row.
                let zero_from = win.bx + n_real * hs * 4;
                std::ptr::write_bytes(
                    scratch.as_mut_ptr().add(zero_from),
                    0,
                    (m - n_real + 1) * hs * 4,
                );
            }
        }
        scratch.flush_cpu_cache(so.activation, m * hs * 4);
        scratch.flush_cpu_cache(so.pos, m * 4);
        scratch.flush_cpu_cache(so.mask, shape.kv_len() * m * 2);
        if let Some(win) = window {
            scratch.flush_cpu_cache(win.bx + n_real * hs * 4, (m - n_real + 1) * hs * 4);
        }
        let resident =
            self.restore_encoder_state(prefix, device.queue_session_mut().dispatch_attempts());
        PassTrace::staged(&mut trace);

        let eps = self.config.rms_norm_eps;
        let session = device.queue_session_mut();
        session.drop_pending_batch();
        let dispatches_before = session.dispatch_attempts();
        self.set_encoder_flush_policy(session, m);
        let wait = PassWait::for_rows(m);
        let polled = wait.begin(session);
        let piped = session.set_pipelined(pipeline_enabled());

        let save_state = capture == Capture::Prefix;
        let final_normed = if self.layers.len().is_multiple_of(2) {
            so.normed
        } else {
            so.normed_b
        };
        let run_res = (|| -> Result<(), CeraError> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (cur_act, next_act, cur_normed) = self.layer_buffers(layer_idx);
                match layer {
                    // The causal prefill's attention emission as is: the bidirectional
                    // semantics are the host-built mask in `so.mask`.
                    HexagonLayer::Attention(attn) => self.emit_attention_stage(
                        session,
                        attn,
                        layer_idx,
                        cur_act,
                        next_act,
                        cur_normed,
                        AttnPass::Prefill { start_pos, m },
                        AttnStage::Whole,
                        &AttnRoute::default(),
                    )?,
                    HexagonLayer::Conv(conv) => {
                        let Some(win) = window else {
                            return Err(CeraError::Backend("encoder conv window missing".into()));
                        };
                        self.emit_conv_encode(
                            session, conv, layer_idx, cur_act, next_act, cur_normed, shape, &win,
                            save_state,
                        )?;
                    }
                    HexagonLayer::DeltaNet(_) => {
                        return Err(CeraError::Backend(
                            "DeltaNet layers have no bidirectional pass".into(),
                        ));
                    }
                }
                if self.debug_barriers
                    && let Err(e) = session.flush()
                {
                    return Err(CeraError::Backend(format!(
                        "encoder layer {layer_idx} flush failed: {e}"
                    )));
                }
            }
            let final_act = if self.layers.len().is_multiple_of(2) {
                so.activation
            } else {
                so.activation_b
            };
            if capture == Capture::Rows {
                Self::dispatch_rms_norm_mul(
                    session,
                    scratch,
                    final_act,
                    &self.weights_buf,
                    self.output_norm_offset,
                    scratch,
                    final_normed,
                    eps,
                    hs,
                    m,
                )?;
            }
            session.flush().map_err(|e| {
                CeraError::Backend(format!("Hexagon NPU encoder execution failed: {e}"))
            })?;
            Ok(())
        })();
        session.set_max_ops_per_flush(None);
        session.set_max_tensors_per_flush(None);
        PassWait::restore(session, polled);
        session.set_pipelined(piped);
        // Whatever a failed pass left in flight still answers: read it, so the next pass
        // starts with an idle queue.
        let run_res = run_res.and_then(|()| session.drain());
        if run_res.is_err() {
            let _ = session.drain();
        }
        PassTrace::ran(&mut trace);
        if let Err(e) = run_res {
            session.drop_pending_batch();
            self.mark_state_torn_if_dispatched(session, dispatches_before);
            return Err(e);
        }

        if capture == Capture::Rows {
            self.note_resident(prefix.map(|p| p.id), session.dispatch_attempts());
        }
        match capture {
            Capture::Rows => {
                scratch.invalidate_cpu_cache(final_normed, n_real * hs * 4);
                let rows = unsafe {
                    std::slice::from_raw_parts(
                        scratch.as_ptr().add(final_normed) as *const f32,
                        n_real * hs,
                    )
                }
                .to_vec();
                PassTrace::finish(
                    trace,
                    session,
                    format_args!(
                        "single rows={n_real} m={m} prefix={start_pos} resident={}",
                        u8::from(resident)
                    ),
                );
                Ok(EncoderOutput { rows, prefix: None })
            }
            Capture::Prefix => {
                let prefix = self.capture_encoder_prefix(n_real);
                self.note_resident(Some(prefix.id), session.dispatch_attempts());
                PassTrace::finish(
                    trace,
                    session,
                    format_args!("single-prefix rows={n_real} m={m}"),
                );
                Ok(EncoderOutput {
                    rows: Vec::new(),
                    prefix: Some(prefix),
                })
            }
        }
    }

    /// Copy `rows` rows of `dim` floats between two buffers as a contiguous 2D
    /// copy (one DSP copy op over `[dim, rows]`, never a flat `dim * rows` vector:
    /// the DSP's elementwise workers keep a whole row in VTCM, so a flat
    /// two-megabyte row is the shape that fails there).
    #[allow(clippy::too_many_arguments)]
    fn copy_rows(
        session: &mut HexagonQueueSession,
        src: &RpcmemBuffer,
        src_offset: usize,
        dst: &RpcmemBuffer,
        dst_offset: usize,
        dim: usize,
        rows: usize,
    ) -> Result<(), CeraError> {
        Self::dispatch_cpy_2d(
            session,
            src,
            src_offset,
            dim,
            rows,
            4,
            dim * 4,
            dst,
            dst_offset,
            4,
            dim * 4,
        )
    }

    /// The layer-outer pass over more rows than one chunk (see the module docs).
    fn encoder_pass_long(
        &self,
        device: &mut HexagonDevice,
        input: EncoderInput<'_>,
        prefix: Option<&HexagonEncoderPrefix>,
        capture: Capture,
        plan: &LongPlan,
    ) -> Result<EncoderOutput, CeraError> {
        let hs = self.config.hidden_size;
        let row = hs * 4;
        let LongPlan { start_pos, n, .. } = *plan;
        let chunks = plan.long_chunks();
        let chunk_max = chunks.iter().map(|c| c.rows).max().unwrap_or(0);
        let chunk_min = chunks.iter().map(|c| c.rows).min().unwrap_or(0);
        let so = self.scratch_offsets;
        let scratch = &self.scratch_buf;
        let has_conv = self
            .layers
            .iter()
            .any(|l| matches!(l, HexagonLayer::Conv(_)));
        let q_dim_max = self
            .layers
            .iter()
            .filter_map(|l| match l {
                HexagonLayer::Attention(a) => Some(a.q_dim),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let lay = LongLayout::plan(n, chunks.len(), chunk_max, hs, q_dim_max, has_conv)?;
        let long_guard = self.ensure_long_buffer(device.queue_session_mut(), lay.total, n)?;
        let Some(long_buf) = long_guard.as_ref() else {
            return Err(CeraError::Backend("encoder long buffer missing".into()));
        };
        let lb = LongBuf { buf: long_buf, lay };
        let kv_len = start_pos + n;
        let mut trace = PassTrace::start(device.queue_session_mut());
        // Host staging, all before the first op reaches the DSP.
        unsafe {
            let res = long_buf.as_mut_ptr().add(lay.res);
            match input {
                EncoderInput::Tokens(tokens) => {
                    for (i, &t) in tokens.iter().enumerate() {
                        let dst = std::slice::from_raw_parts_mut(res.add(i * row) as *mut f32, hs);
                        self.token_embd.row_into(t as usize, dst);
                    }
                }
                EncoderInput::Embeddings(rows) => {
                    std::ptr::copy_nonoverlapping(rows.as_ptr() as *const u8, res, n * row);
                }
            }
            for c in &chunks {
                let slot = std::slice::from_raw_parts_mut(
                    long_buf.as_mut_ptr().add(lay.pos + c.idx * lay.pos_stride) as *mut i32,
                    c.rows,
                );
                for (i, p) in slot.iter_mut().enumerate() {
                    *p = (start_pos + c.first + i) as i32;
                }
            }
            if has_conv {
                // The row past the last bx row: the last chunk's right neighbour.
                std::ptr::write_bytes(long_buf.as_mut_ptr().add(lay.ubx + n * row), 0, row);
            }
            // No padding rows, so every query sees every slot.
            let mask = std::slice::from_raw_parts_mut(
                scratch.as_mut_ptr().add(so.mask) as *mut u16,
                kv_len * chunk_max,
            );
            fill_bidirectional_mask(mask, chunk_max, kv_len, kv_len);
        }
        long_buf.flush_cpu_cache(lay.res, n * row);
        long_buf.flush_cpu_cache(lay.pos, chunks.len() * lay.pos_stride);
        if has_conv {
            long_buf.flush_cpu_cache(lay.ubx + n * row, row);
        }
        scratch.flush_cpu_cache(so.mask, kv_len * chunk_max * 2);
        let resident =
            self.restore_encoder_state(prefix, device.queue_session_mut().dispatch_attempts());
        PassTrace::staged(&mut trace);

        let eps = self.config.rms_norm_eps;
        let session = device.queue_session_mut();
        session.drop_pending_batch();
        let dispatches_before = session.dispatch_attempts();
        self.set_encoder_flush_policy(session, chunk_min);
        let wait = PassWait::for_rows(n);
        let polled = wait.begin(session);
        let piped = session.set_pipelined(pipeline_enabled());

        let save_state = capture == Capture::Prefix;
        let run_res = (|| -> Result<(), CeraError> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (cur_act, next_act, cur_normed) = self.layer_buffers(layer_idx);
                let res_at = |c: &LongChunk| lay.res + c.first * row;
                // The chunk's residual rows into the scratch activation.
                let load = |session: &mut HexagonQueueSession, c: &LongChunk| {
                    Self::copy_rows(session, long_buf, res_at(c), scratch, cur_act, hs, c.rows)
                };
                match layer {
                    HexagonLayer::Attention(attn) => {
                        let q_row = attn.q_dim * 4;
                        // Sweep 1: norm .. K/V write for every chunk, Q kept.
                        for c in &chunks {
                            load(session, c)?;
                            let route = AttnRoute {
                                pos: Some((long_buf, lay.pos + c.idx * lay.pos_stride)),
                                ..AttnRoute::default()
                            };
                            self.emit_attention_stage(
                                session,
                                attn,
                                layer_idx,
                                cur_act,
                                next_act,
                                cur_normed,
                                AttnPass::Prefill {
                                    start_pos: start_pos + c.first,
                                    m: c.rows,
                                },
                                AttnStage::Front,
                                &route,
                            )?;
                            Self::copy_rows(
                                session,
                                scratch,
                                so.q,
                                long_buf,
                                lay.uq + c.first * q_row,
                                attn.q_dim,
                                c.rows,
                            )?;
                        }
                        // Sweep 2: attention over every slot .. FFN.
                        for c in &chunks {
                            load(session, c)?;
                            let route = AttnRoute {
                                q: Some((long_buf, lay.uq + c.first * q_row)),
                                kv_len: Some(kv_len),
                                ..AttnRoute::default()
                            };
                            self.emit_attention_stage(
                                session,
                                attn,
                                layer_idx,
                                cur_act,
                                next_act,
                                cur_normed,
                                AttnPass::Prefill {
                                    start_pos: start_pos + c.first,
                                    m: c.rows,
                                },
                                AttnStage::Back,
                                &route,
                            )?;
                            Self::copy_rows(
                                session,
                                scratch,
                                next_act,
                                long_buf,
                                res_at(c),
                                hs,
                                c.rows,
                            )?;
                        }
                    }
                    HexagonLayer::Conv(conv) => {
                        for c in &chunks {
                            load(session, c)?;
                            self.emit_conv_long_front(session, conv, cur_act, cur_normed, c, &lb)?;
                        }
                        for c in &chunks {
                            load(session, c)?;
                            self.emit_conv_long_back(
                                session, conv, layer_idx, cur_act, next_act, cur_normed, c, &lb,
                                save_state,
                            )?;
                            Self::copy_rows(
                                session,
                                scratch,
                                next_act,
                                long_buf,
                                res_at(c),
                                hs,
                                c.rows,
                            )?;
                        }
                    }
                    HexagonLayer::DeltaNet(_) => {
                        return Err(CeraError::Backend(
                            "DeltaNet layers have no bidirectional pass".into(),
                        ));
                    }
                }
                if self.debug_barriers
                    && let Err(e) = session.flush()
                {
                    return Err(CeraError::Backend(format!(
                        "encoder layer {layer_idx} flush failed: {e}"
                    )));
                }
            }
            if capture == Capture::Rows {
                for c in &chunks {
                    let at = lay.res + c.first * row;
                    Self::dispatch_rms_norm_mul(
                        session,
                        long_buf,
                        at,
                        &self.weights_buf,
                        self.output_norm_offset,
                        long_buf,
                        at,
                        eps,
                        hs,
                        c.rows,
                    )?;
                }
            }
            session.flush().map_err(|e| {
                CeraError::Backend(format!("Hexagon NPU encoder execution failed: {e}"))
            })?;
            Ok(())
        })();
        session.set_max_ops_per_flush(None);
        session.set_max_tensors_per_flush(None);
        PassWait::restore(session, polled);
        session.set_pipelined(piped);
        // Whatever a failed pass left in flight still answers: read it, so the next pass
        // starts with an idle queue.
        let run_res = run_res.and_then(|()| session.drain());
        if run_res.is_err() {
            let _ = session.drain();
        }
        PassTrace::ran(&mut trace);
        if let Err(e) = run_res {
            session.drop_pending_batch();
            self.mark_state_torn_if_dispatched(session, dispatches_before);
            drop(long_guard);
            self.trim_long_buffer(session);
            return Err(e);
        }

        if capture == Capture::Rows {
            // A layer-outer pass overwrites slot 0 of every conv slab with the previous
            // chunk's last `bx` for every chunk after the first, so the prefix's left
            // neighbour is gone: nothing is resident.
            self.note_resident(None, session.dispatch_attempts());
        }
        let out = match capture {
            Capture::Rows => {
                long_buf.invalidate_cpu_cache(lay.res, n * row);
                let rows = unsafe {
                    std::slice::from_raw_parts(long_buf.as_ptr().add(lay.res) as *const f32, n * hs)
                }
                .to_vec();
                EncoderOutput { rows, prefix: None }
            }
            Capture::Prefix => {
                let prefix = self.capture_encoder_prefix(n);
                self.note_resident(Some(prefix.id), session.dispatch_attempts());
                EncoderOutput {
                    rows: Vec::new(),
                    prefix: Some(prefix),
                }
            }
        };
        drop(long_guard);
        self.trim_long_buffer(session);
        PassTrace::finish(
            trace,
            session,
            format_args!(
                "long rows={n} prefix={start_pos} resident={}",
                u8::from(resident)
            ),
        );
        Ok(out)
    }

    /// The long buffer, at least `bytes` large (allocated, or grown, on demand;
    /// a larger one replaces the old after the DSP lets go of it). `rows` is for
    /// the error only.
    fn ensure_long_buffer(
        &self,
        session: &mut HexagonQueueSession,
        bytes: usize,
        rows: usize,
    ) -> Result<MutexGuard<'_, Option<RpcmemBuffer>>, CeraError> {
        let mut guard = self.encoder_long.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().is_none_or(|b| b.size() < bytes) {
            if let Some(old) = guard.take() {
                session.release_dsp_reference(&old);
            }
            let driver = self.scratch_buf.driver();
            // A quarter of headroom so passes of nearby lengths share a buffer;
            // the exact size when the headroom does not fit.
            let roomy = (bytes + bytes / 4).next_multiple_of(1 << 20);
            let started = std::time::Instant::now();
            let buf = RpcmemBuffer::alloc(Arc::clone(driver), roomy, true)
                .or_else(|_| RpcmemBuffer::alloc(Arc::clone(driver), bytes, true))
                .map_err(|e| {
                    CeraError::Backend(format!(
                        "an NPU encoder pass over {rows} rows needs {} MiB of shared memory: {e}",
                        bytes >> 20
                    ))
                })?;
            trace_alloc("encoder long buffer", buf.size(), started);
            *guard = Some(buf);
        }
        Ok(guard)
    }

    /// Release a long buffer too large to keep (see [`LONG_BUFFER_KEEP_BYTES`]).
    fn trim_long_buffer(&self, session: &mut HexagonQueueSession) {
        // A batch that never answered (a hung DSP) may still be writing the buffer, and
        // dropping it unmaps memory the DSP owns: keep it until the queue has quiesced.
        if session.outstanding_batches() > 0 {
            return;
        }
        let mut guard = self.encoder_long.lock().unwrap_or_else(|e| e.into_inner());
        if guard
            .as_ref()
            .is_some_and(|b| b.size() > LONG_BUFFER_KEEP_BYTES)
            && let Some(old) = guard.take()
        {
            session.release_dsp_reference(&old);
        }
    }

    /// Per-layer `(kv_dim, k_offset, v_offset)` of the attention layers, and the
    /// state offsets of the conv layers, in layer order.
    pub(super) fn encoder_state_layers(&self) -> (Vec<(usize, usize, usize)>, Vec<usize>) {
        let mut attn = Vec::new();
        let mut conv = Vec::new();
        for layer in &self.layers {
            match layer {
                HexagonLayer::Attention(a) => attn.push((a.kv_dim, a.k_offset, a.v_offset)),
                HexagonLayer::Conv(c) => conv.push(c.state_offset),
                HexagonLayer::DeltaNet(_) => {}
            }
        }
        (attn, conv)
    }

    /// Refuse a prefix this model did not make (or one whose layout differs).
    fn check_encoder_prefix(&self, prefix: &HexagonEncoderPrefix) -> Result<(), CeraError> {
        let bad = |why: &str| {
            Err(CeraError::Backend(format!(
                "the encoder prefix does not belong to this model: {why}"
            )))
        };
        if prefix.model_id != self.encoder_id {
            return bad("it was made by another model instance");
        }
        let (attn, conv) = self.encoder_state_layers();
        if prefix.kv.len() != attn.len() || prefix.conv_left.len() != conv.len() {
            return bad("layer count differs");
        }
        let hs = self.config.hidden_size;
        for (&(kv_dim, _, _), (k, v)) in attn.iter().zip(&prefix.kv) {
            let want = prefix.n_rows * kv_row_stride(self.kv_dtype, kv_dim);
            if k.len() != want || v.len() != want {
                return bad("KV size differs");
            }
        }
        if prefix.conv_left.iter().any(|row| row.len() != hs) {
            return bad("conv state width differs");
        }
        Ok(())
    }

    /// Put the device in the state a pass starting after `prefix` expects: the
    /// prefix K/V rows in the cache and its last `bx` in slot 0 of every conv
    /// slab. Without a prefix only the conv slabs are zeroed (slot 0 is the left
    /// neighbour of the first row; slot 1 is rewritten by the pass itself).
    ///
    /// A question's pass after the previous question's against the same prefix finds all of
    /// that already in place: a pass writes only the cache slots from the prefix's end on
    /// and slot 1 of each slab, never the prefix's rows or slot 0. So when the last pass
    /// left the device holding this very prefix (`encoder_resident`) and no other batch
    /// has reached the queue since (`seq` is the queue's dispatch count), the host-to-device
    /// copy of every layer's K/V (12 KB a row of the prefix) is skipped. Returns whether it
    /// was.
    fn restore_encoder_state(&self, prefix: Option<&HexagonEncoderPrefix>, seq: u64) -> bool {
        let hs = self.config.hidden_size;
        let kvb = &self.kv_state_buf;
        // Cleared whatever happens: a pass that fails half way leaves nothing certain.
        let resident = self
            .encoder_resident
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(prefix) = prefix
            && resident == Some((prefix.id, seq))
            && resident_prefix_enabled()
        {
            return true;
        }
        let (attn, conv) = self.encoder_state_layers();
        if let Some(prefix) = prefix {
            for (&(kv_dim, k_off, v_off), (k, v)) in attn.iter().zip(&prefix.kv) {
                debug_assert_eq!(
                    k.len(),
                    prefix.n_rows * kv_row_stride(self.kv_dtype, kv_dim)
                );
                for (off, bytes) in [(k_off, k), (v_off, v)] {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            kvb.as_mut_ptr().add(off),
                            bytes.len(),
                        );
                    }
                    kvb.flush_cpu_cache(off, bytes.len());
                }
            }
        }
        let zero = vec![0.0f32; hs];
        for (i, &state_offset) in conv.iter().enumerate() {
            let left = prefix.map_or(&zero[..], |p| &p.conv_left[i][..]);
            let slab = unsafe {
                std::slice::from_raw_parts_mut(kvb.as_mut_ptr().add(state_offset), hs * 8)
            };
            slab_write_slot(slab, 0, left);
            kvb.flush_cpu_cache(state_offset, hs * 8);
        }
        false
    }

    /// Record what the device holds after a pass that ended with the queue at `seq`
    /// dispatches: `prefix` (the id of the prefix whose K/V rows and last `bx` sit there), or
    /// nothing when the pass overwrote them (a pass without a prefix writes from position 0).
    fn note_resident(&self, prefix: Option<u64>, seq: u64) {
        *self
            .encoder_resident
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = prefix.map(|id| (id, seq));
    }

    /// Copy what a just-finished prefix pass left on the device into host memory.
    fn capture_encoder_prefix(&self, n_rows: usize) -> HexagonEncoderPrefix {
        let hs = self.config.hidden_size;
        let kvb = &self.kv_state_buf;
        let (attn, conv) = self.encoder_state_layers();
        let read = |off: usize, len: usize| {
            kvb.invalidate_cpu_cache(off, len);
            unsafe { std::slice::from_raw_parts(kvb.as_ptr().add(off), len) }.to_vec()
        };
        let kv = attn
            .iter()
            .map(|&(kv_dim, k_off, v_off)| {
                let len = n_rows * kv_row_stride(self.kv_dtype, kv_dim);
                (read(k_off, len), read(v_off, len))
            })
            .collect();
        let conv_left = conv
            .iter()
            .map(|&state_offset| {
                kvb.invalidate_cpu_cache(state_offset, hs * 8);
                let slab =
                    unsafe { std::slice::from_raw_parts(kvb.as_ptr().add(state_offset), hs * 8) };
                let mut row = vec![0.0f32; hs];
                slab_read_slot(slab, 0, &mut row);
                row
            })
            .collect();
        HexagonEncoderPrefix {
            model_id: self.encoder_id,
            id: NEXT_ENCODER_PREFIX_ID.fetch_add(1, Ordering::Relaxed),
            n_rows,
            kv,
            conv_left,
        }
    }

    /// Whether this model runs the bidirectional encoder pass (`encode_prefix` / `encode_rows`).
    pub(super) fn supports_bidirectional_encode(&self) -> bool {
        self.encoder_blocker().is_none()
    }

    /// Run a bidirectional pass over `input` from position 0 and keep what a later
    /// [`Self::encode_rows`] needs to continue after it (K/V rows, conv state). The rows are not
    /// returned: a prefix exists to be attended to.
    pub(super) fn encode_prefix(
        &self,
        input: EncoderInput<'_>,
        _state: &mut crate::kv_cache::InferenceState,
    ) -> Result<EncoderPrefix, CeraError> {
        let out = self.encoder_pass(input, None, Capture::Prefix)?;
        let prefix = out
            .prefix
            .ok_or_else(|| CeraError::Backend("encoder pass returned no prefix".into()))?;
        Ok(EncoderPrefix::new(prefix.n_rows, prefix))
    }

    /// Run a bidirectional pass over `input` and return the per-row hidden states after the
    /// final norm, row-major `[rows * hidden_size]`. With `prefix`, the rows sit after it:
    /// positions continue from the prefix, the first row's conv left neighbour is the prefix's
    /// last, and every row attends to the prefix and to all the other rows (never the other way:
    /// the prefix is unchanged). Can be called many times against one prefix.
    pub(super) fn encode_rows(
        &self,
        input: EncoderInput<'_>,
        prefix: Option<&EncoderPrefix>,
        _state: &mut crate::kv_cache::InferenceState,
    ) -> Result<Vec<f32>, CeraError> {
        let prefix = match prefix {
            Some(p) => Some(p.downcast_ref::<HexagonEncoderPrefix>().ok_or_else(|| {
                CeraError::Backend("the encoder prefix was made by a different backend".into())
            })?),
            None => None,
        };
        Ok(self.encoder_pass(input, prefix, Capture::Rows)?.rows)
    }
}

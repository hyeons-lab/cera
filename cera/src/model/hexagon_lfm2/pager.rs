//! Paged routed-expert weights.
//!
//! The CDSP maps about 3.9 GiB in total, so a routed model whose experts
//! alone exceed that (LFM2.5-8B-A1B: 4.4 GiB of experts) cannot keep them all
//! mapped. Each routed layer's expert stacks live in their own `rpcmem`
//! buffer. All of them stay allocated, so paging never copies weights; only
//! the DSP mapping moves.
//!
//! Measured on an S25 Ultra (`examples/hexagon_map_probe.rs`): unmapping
//! returns the address space, and mapping costs about 7 ms per GiB.
//!
//! The router runs on the DSP, so the host cannot know which experts a layer
//! will pick. A layer's whole stack therefore has to be mapped for the batch
//! that runs it. [`ExpertPager`] keeps as many layers mapped for good
//! ("pinned") as the budget allows and rotates the rest through a window.
//! When the window is full the pending batch is flushed (the DSP must be done
//! with a buffer before it is unmapped) and the whole window is released at
//! once, so a pass over the model costs one flush per window rather than one
//! per layer.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::hexagon::{FastRpcDriver, RpcmemBuffer};
use crate::session::CeraError;

const MIB: usize = 1 << 20;

/// Default cap on bytes mapped into the CDSP, below the 3.9 GiB hard ceiling:
/// several mid-sized buffers fragment the address space before the total is
/// reached (four 1 GiB buffers did not fit on the S25 Ultra).
/// `CERA_HEXAGON_MAP_BUDGET_MIB` overrides it.
const DEFAULT_MAP_BUDGET: usize = 3584 * MIB;

/// How the loader decides to page and how much it may keep mapped.
#[derive(Clone, Copy, Debug)]
pub(super) struct Paging {
    /// Page even when the model would fit (tests, A/B runs).
    pub(super) force: bool,
    /// Cap on bytes mapped into the CDSP.
    pub(super) budget: usize,
}

impl Paging {
    /// `CERA_HEXAGON_MAP_BUDGET_MIB` (default 3584, never past the driver's
    /// hard ceiling) and `CERA_HEXAGON_PAGE_EXPERTS=1`.
    pub(super) fn from_env() -> Self {
        let budget = std::env::var("CERA_HEXAGON_MAP_BUDGET_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|mib| mib.saturating_mul(MIB))
            .unwrap_or(DEFAULT_MAP_BUDGET)
            .min(crate::backend::hexagon::sys::DSP_MAP_CEILING);
        Self {
            force: std::env::var("CERA_HEXAGON_PAGE_EXPERTS").is_ok_and(|v| v == "1"),
            budget,
        }
    }

    /// Whether to page routed experts: when everything the model needs would
    /// not fit under the budget (`resident_mapped` bytes are already mapped,
    /// plus the weights and a margin for KV and conv state), or when forced.
    pub(super) fn should_page(&self, gguf_bytes: u64, resident_mapped: usize) -> bool {
        const KV_MARGIN: usize = 256 * MIB;
        self.force
            || (gguf_bytes as usize)
                .saturating_add(resident_mapped)
                .saturating_add(KV_MARGIN)
                > self.budget
    }
}

/// Counters over a pager's lifetime.
#[cfg(test)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct PagerStats {
    /// Layers mapped on demand (misses).
    pub(super) page_ins: u64,
    /// Window rotations: each one flushed the pending batch and unmapped the
    /// window.
    pub(super) rotations: u64,
}

pub(super) struct ExpertPager {
    bufs: Vec<RpcmemBuffer>,
    driver: Arc<FastRpcDriver>,
    budget: usize,
    /// Mapped on demand and not pinned, oldest first.
    window: Mutex<VecDeque<usize>>,
    page_ins: AtomicU64,
    rotations: AtomicU64,
}

impl ExpertPager {
    /// Take ownership of the per-layer expert buffers and pin as many as fit.
    ///
    /// `driver.mapped_bytes()` must already include everything else the model
    /// maps (weights, scratch, KV). Layers are pinned in order; if they do not
    /// all fit, room for a two-layer window is kept back for the rest.
    pub(super) fn new(
        driver: Arc<FastRpcDriver>,
        bufs: Vec<RpcmemBuffer>,
        budget: usize,
    ) -> Result<Self, CeraError> {
        let mapped = driver.mapped_bytes();
        let total: usize = bufs.iter().map(RpcmemBuffer::size).sum();
        let largest = bufs.iter().map(RpcmemBuffer::size).max().unwrap_or(0);
        let everything_fits = mapped.saturating_add(total) <= budget;
        // A window of two layers: the one being run and the next one.
        let reserve = if everything_fits {
            0
        } else {
            largest.saturating_mul(2)
        };
        if mapped.saturating_add(reserve) > budget {
            return Err(CeraError::Backend(format!(
                "paging routed experts needs room for two layers ({} MiB) beside the {} MiB \
                 already mapped, but the DSP mapping budget is {} MiB",
                reserve >> 20,
                mapped >> 20,
                budget >> 20
            )));
        }
        let mut free = budget - mapped - reserve;
        for buf in &bufs {
            if buf.size() > free {
                break;
            }
            buf.map_to_dsp()?;
            free -= buf.size();
        }
        Ok(Self {
            bufs,
            driver,
            budget,
            window: Mutex::new(VecDeque::new()),
            page_ins: AtomicU64::new(0),
            rotations: AtomicU64::new(0),
        })
    }

    /// The buffer holding layer group `group`'s expert stacks.
    pub(super) fn buf(&self, group: usize) -> &RpcmemBuffer {
        &self.bufs[group]
    }

    /// Bytes mapped into the DSP through this pager's driver (everything the
    /// model maps, not just experts).
    #[cfg(test)]
    pub(super) fn mapped_bytes(&self) -> usize {
        self.driver.mapped_bytes()
    }

    /// Layers kept mapped for good.
    pub(super) fn pinned(&self) -> usize {
        // Pinned layers are the leading mapped ones and are never windowed.
        let windowed = self.window.lock().unwrap_or_else(|p| p.into_inner()).len();
        self.bufs
            .iter()
            .filter(|b| b.is_mapped())
            .count()
            .saturating_sub(windowed)
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> PagerStats {
        PagerStats {
            page_ins: self.page_ins.load(Ordering::SeqCst),
            rotations: self.rotations.load(Ordering::SeqCst),
        }
    }

    /// Make layer group `group` mapped before ops that read it are queued.
    ///
    /// When the window is full, `finish_pending` must complete every batch
    /// that may still read the buffers about to be unmapped (flush the queue
    /// and wait for the DSP). It is not called when the layer is already
    /// mapped or there is room.
    pub(super) fn page_in(
        &self,
        group: usize,
        finish_pending: impl FnOnce() -> Result<(), CeraError>,
    ) -> Result<(), CeraError> {
        let buf = &self.bufs[group];
        if buf.is_mapped() {
            return Ok(());
        }
        let mut window = self.window.lock().unwrap_or_else(|p| p.into_inner());
        if self.driver.mapped_bytes().saturating_add(buf.size()) > self.budget {
            finish_pending()?;
            while let Some(old) = window.pop_front() {
                self.bufs[old].unmap_from_dsp()?;
            }
            self.rotations.fetch_add(1, Ordering::SeqCst);
        }
        buf.map_to_dsp().map_err(|e| {
            CeraError::Backend(format!(
                "paging routed experts: mapping layer group {group} ({} MiB) failed with {} MiB \
                 mapped: {e}",
                buf.size() >> 20,
                self.driver.mapped_bytes() >> 20
            ))
        })?;
        window.push_back(group);
        self.page_ins.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::hexagon::sys::fake;

    fn pager(sizes_mib: &[usize], budget_mib: usize) -> (Arc<FastRpcDriver>, ExpertPager) {
        fake::reset();
        let driver = fake::driver();
        let bufs = sizes_mib
            .iter()
            .map(|&m| RpcmemBuffer::alloc(driver.clone(), m * MIB, false).unwrap())
            .collect();
        let pager = ExpertPager::new(driver.clone(), bufs, budget_mib * MIB).unwrap();
        (driver, pager)
    }

    fn mapped(p: &ExpertPager) -> Vec<bool> {
        (0..p.bufs.len()).map(|i| p.bufs[i].is_mapped()).collect()
    }

    #[test]
    fn everything_that_fits_is_pinned_and_never_flushes() {
        let (_d, p) = pager(&[100, 100, 100], 400);
        assert_eq!(mapped(&p), [true, true, true]);
        assert_eq!(p.pinned(), 3);
        for g in 0..3 {
            p.page_in(g, || panic!("a resident layer must not flush"))
                .unwrap();
        }
        assert_eq!(p.stats(), PagerStats::default());
    }

    #[test]
    fn leading_layers_are_pinned_and_a_two_layer_window_is_kept_back() {
        // 6 x 100 MiB under a 450 MiB budget: reserve 200, pin the first two.
        let (d, p) = pager(&[100; 6], 450);
        assert_eq!(mapped(&p), [true, true, false, false, false, false]);
        assert_eq!(p.pinned(), 2);
        assert_eq!(d.mapped_bytes(), 200 * MIB);
    }

    #[test]
    fn a_full_window_flushes_once_and_releases_all_of_it() {
        let (d, p) = pager(&[100; 6], 450);
        let flushes = std::cell::Cell::new(0);
        let flush = || {
            flushes.set(flushes.get() + 1);
            Ok(())
        };
        // Pinned layers cost nothing; the paged ones fill the 250 MiB of room
        // two at a time.
        for g in 0..6 {
            p.page_in(g, flush).unwrap();
        }
        // Layers 2,3 fit; 4 forces a rotation (flush #1) and holds 4,5.
        assert_eq!(flushes.get(), 1);
        assert_eq!(mapped(&p), [true, true, false, false, true, true]);
        assert!(d.mapped_bytes() <= 450 * MIB);
        assert_eq!(
            p.stats(),
            PagerStats {
                page_ins: 4,
                rotations: 1
            }
        );
        // The next pass wraps: layers 2 and 3 are out, so mapping 2 rotates.
        for g in 0..6 {
            p.page_in(g, flush).unwrap();
        }
        assert_eq!(flushes.get(), 3, "one flush per window, not per layer");
        assert_eq!(p.pinned(), 2);
    }

    #[test]
    fn a_failed_flush_leaves_the_window_mapped() {
        let (_d, p) = pager(&[100; 6], 450);
        for g in 2..4 {
            p.page_in(g, || Ok(())).unwrap();
        }
        let err = p
            .page_in(4, || Err(CeraError::Backend("dsp fault".into())))
            .expect_err("the flush failure must surface");
        assert!(err.to_string().contains("dsp fault"));
        // Nothing was unmapped under a batch the DSP may still be running.
        assert!(p.bufs[2].is_mapped() && p.bufs[3].is_mapped());
        assert!(!p.bufs[4].is_mapped());
    }

    #[test]
    fn no_room_for_a_window_is_a_named_error() {
        fake::reset();
        let driver = fake::driver();
        let bufs = (0..4)
            .map(|_| RpcmemBuffer::alloc(driver.clone(), 300 * MIB, false).unwrap())
            .collect();
        let err = ExpertPager::new(driver, bufs, 500 * MIB)
            .err()
            .expect("two 300 MiB layers cannot fit in 500 MiB");
        assert!(err.to_string().contains("room for two layers"), "{err}");
    }

    #[test]
    fn paging_is_chosen_only_when_the_model_does_not_fit() {
        let gib = 1u64 << 30;
        let p = Paging {
            force: false,
            budget: 3584 * MIB,
        };
        assert!(!p.should_page(gib, 200 * MIB));
        assert!(p.should_page(5 * gib, 200 * MIB));
        // The resident mappings count against the budget too.
        assert!(p.should_page(3 * gib, 300 * MIB));
        // Forcing pages a model that fits.
        assert!(Paging { force: true, ..p }.should_page(1, 0));
    }
}

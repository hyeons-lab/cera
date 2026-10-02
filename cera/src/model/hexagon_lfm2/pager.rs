//! Paged routed-expert weights.
//!
//! The CDSP maps about 3.9 GiB in total, so a routed model whose experts
//! alone exceed that (LFM2.5-8B-A1B: 4.4 GiB of experts) cannot keep them all
//! mapped. Each routed layer's expert stacks live in their own `rpcmem`
//! buffer. All of them stay allocated, so paging never copies weights; only
//! the DSP mapping moves.
//!
//! Measured on an S25 Ultra (`examples/hexagon_map_probe.rs`): unmapping
//! returns the address space, and mapping costs about 11 ms per GiB.
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

use super::weights::backend_msg;
use crate::backend::hexagon::{FastRpcDriver, RpcmemBuffer, hexagon_warn};
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
    /// Whether a model that does not fit may be paged at all. `--device auto`
    /// clears it (see [`Self::for_auto`]): paging runs slower than the CPU, so
    /// it must be asked for.
    pub(super) allow_paged: bool,
}

impl Default for Paging {
    fn default() -> Self {
        Self {
            force: false,
            budget: DEFAULT_MAP_BUDGET,
            allow_paged: true,
        }
    }
}

impl Paging {
    /// The policy the process environment asks for; see [`Self::from_lookup`].
    pub(super) fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// `CERA_HEXAGON_MAP_BUDGET_KIB` or `_MIB` (KiB wins; default 3584 MiB,
    /// never past the driver's hard ceiling) and `CERA_HEXAGON_PAGE_EXPERTS`
    /// (`1` or `true`, like the other opt-in knobs). An unparsable budget is
    /// warned about and ignored.
    pub(super) fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let budget_of = |key: &str, unit: usize| {
            let raw = get(key)?;
            match raw.trim().parse::<usize>() {
                Ok(n) => Some(n.saturating_mul(unit)),
                Err(_) => {
                    hexagon_warn!("ignoring {key}={raw:?}: not a whole number");
                    None
                }
            }
        };
        // `_KIB` is for tiny models, where a MiB is more than the whole window.
        let budget = budget_of("CERA_HEXAGON_MAP_BUDGET_KIB", 1024)
            .or_else(|| budget_of("CERA_HEXAGON_MAP_BUDGET_MIB", MIB))
            .unwrap_or(DEFAULT_MAP_BUDGET)
            .min(crate::backend::hexagon::sys::DSP_MAP_CEILING);
        let force = get("CERA_HEXAGON_PAGE_EXPERTS").is_some_and(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true")
        });
        Self {
            force,
            budget,
            allow_paged: true,
        }
    }

    /// The policy for `--device auto`: a model that fits is loaded as usual,
    /// one that would need paging is refused (and falls back to the CPU)
    /// unless paging was asked for with `CERA_HEXAGON_PAGE_EXPERTS`.
    pub(super) fn for_auto(self) -> Self {
        Self {
            allow_paged: self.force,
            ..self
        }
    }

    /// Whether to page routed experts: when everything the model needs would
    /// not fit under the budget (`resident_mapped` bytes are already mapped,
    /// plus the weights and `kv_bytes` of KV and conv state, plus a small
    /// slack for what the repacked weights add over the file's bytes), or when
    /// forced.
    pub(super) fn should_page(
        &self,
        gguf_bytes: u64,
        resident_mapped: usize,
        kv_bytes: usize,
    ) -> bool {
        const SLACK: usize = 64 * MIB;
        self.force
            || usize::try_from(gguf_bytes)
                .unwrap_or(usize::MAX)
                .saturating_add(resident_mapped)
                .saturating_add(kv_bytes)
                .saturating_add(SLACK)
                > self.budget
    }
}

/// What the pager needs from the queue it pages for.
pub(super) trait PagerHost {
    /// Complete every batch that may still read a buffer about to be unmapped
    /// (flush the queue and wait for the DSP).
    fn finish_pending(&mut self) -> Result<(), CeraError>;
    /// Tell the DSP to drop its hold on `buf`; the host's unmap is refused
    /// while the DSP keeps one after a batch has read the buffer.
    fn release(&mut self, buf: &RpcmemBuffer);
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
    /// Leading layers mapped for good in [`Self::new`].
    pinned: usize,
    /// Mapped on demand and not pinned, oldest first. Every mutation leaves it
    /// consistent with the buffers' mapped state (a layer leaves it only after
    /// its unmap succeeded), so a poisoned lock is safe to take over.
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
        let mut pinned = 0;
        for (i, buf) in bufs.iter().enumerate() {
            if buf.size() > free {
                break;
            }
            // The DSP can refuse a mapping the budget allows (fragmentation);
            // pin fewer layers rather than fail the load. The reserved window
            // is still free, and `page_in` reports a refusal there.
            if let Err(e) = buf.map_to_dsp() {
                hexagon_warn!("pinned {i} expert layers; the DSP refused more: {e}");
                break;
            }
            free -= buf.size();
            pinned += 1;
        }
        Ok(Self {
            bufs,
            driver,
            budget,
            pinned,
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
        self.pinned
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
    /// A DSP refusal of the map itself (address-space fragmentation) is
    /// reported, not retried: after a rotation the window is already empty, so
    /// there is nothing left to free.
    ///
    /// When the window is full, `host` must finish every pending batch before
    /// the window's buffers are released and unmapped. Neither happens when
    /// the layer is already mapped or there is room.
    pub(super) fn page_in(&self, group: usize, host: &mut impl PagerHost) -> Result<(), CeraError> {
        let buf = &self.bufs[group];
        // Held for the whole call, so the check-then-act on the mapped flag is
        // atomic against any other caller (the device lock already keeps the
        // model to one forward at a time).
        let mut window = self.window.lock().unwrap_or_else(|p| p.into_inner());
        if buf.is_mapped() {
            return Ok(());
        }
        if self.driver.mapped_bytes().saturating_add(buf.size()) > self.budget {
            host.finish_pending()?;
            // A layer leaves the window only once its unmap succeeded, so a
            // refused unmap keeps it tracked and retried at the next rotation.
            while let Some(&old) = window.front() {
                host.release(&self.bufs[old]);
                self.bufs[old].unmap_from_dsp().map_err(|e| {
                    CeraError::Backend(format!(
                        "paging routed experts: unmapping layer group {old} failed: {}",
                        backend_msg(e)
                    ))
                })?;
                window.pop_front();
            }
            let rotation = self.rotations.fetch_add(1, Ordering::SeqCst) + 1;
            tracing::debug!(
                "cera::hexagon: expert window rotation {rotation} for layer group {group} \
                 ({} page-ins so far)",
                self.page_ins.load(Ordering::SeqCst)
            );
        }
        buf.map_to_dsp().map_err(|e| {
            CeraError::Backend(format!(
                "paging routed experts: mapping layer group {group} ({} MiB) failed with {} MiB \
                 mapped: {}",
                buf.size() >> 20,
                self.driver.mapped_bytes() >> 20,
                backend_msg(e)
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

    /// Counts what the pager asks of its queue; `fail` makes the flush fail.
    #[derive(Default)]
    struct Host {
        flushes: usize,
        released: Vec<usize>,
        fail: bool,
    }

    impl PagerHost for Host {
        fn finish_pending(&mut self) -> Result<(), CeraError> {
            if self.fail {
                return Err(CeraError::Backend("dsp fault".into()));
            }
            self.flushes += 1;
            Ok(())
        }

        fn release(&mut self, buf: &RpcmemBuffer) {
            self.released.push(buf.size());
        }
    }

    fn mapped(p: &ExpertPager) -> Vec<bool> {
        (0..p.bufs.len()).map(|i| p.bufs[i].is_mapped()).collect()
    }

    #[test]
    fn everything_that_fits_is_pinned_and_never_flushes() {
        let (_d, p) = pager(&[100, 100, 100], 400);
        assert_eq!(mapped(&p), [true, true, true]);
        assert_eq!(p.pinned(), 3);
        let mut host = Host::default();
        for g in 0..3 {
            p.page_in(g, &mut host).unwrap();
        }
        assert_eq!((host.flushes, host.released.len()), (0, 0));
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
        let mut host = Host::default();
        // Pinned layers cost nothing; the paged ones fill the 250 MiB of room
        // two at a time.
        for g in 0..6 {
            p.page_in(g, &mut host).unwrap();
        }
        // Layers 2,3 fit; 4 forces a rotation (flush #1) and holds 4,5.
        assert_eq!(host.flushes, 1);
        // The DSP is told to let go of both window buffers before the unmap.
        assert_eq!(host.released, [100 * MIB, 100 * MIB]);
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
            p.page_in(g, &mut host).unwrap();
        }
        assert_eq!(host.flushes, 3, "one flush per window, not per layer");
        assert_eq!(
            p.stats(),
            PagerStats {
                page_ins: 8,
                rotations: 3
            }
        );
        assert_eq!(p.pinned(), 2);
    }

    /// A host that releases through the fake skel, as the real one does.
    struct ReleasingHost;

    impl PagerHost for ReleasingHost {
        fn finish_pending(&mut self) -> Result<(), CeraError> {
            Ok(())
        }

        fn release(&mut self, buf: &RpcmemBuffer) {
            fake::with(|s| s.events.push(fake::Event::Release(buf.fd())));
        }
    }

    #[test]
    fn every_window_buffer_is_released_before_it_is_unmapped() {
        use fake::Event::{Map, Release, Unmap};
        fake::reset();
        fake::with(|s| s.distinct_fds = true);
        let driver = fake::driver();
        let bufs = (0..6)
            .map(|_| RpcmemBuffer::alloc(driver.clone(), 100 * MIB, false).unwrap())
            .collect();
        let p = ExpertPager::new(driver, bufs, 450 * MIB).unwrap();
        for g in 2..6 {
            p.page_in(g, &mut ReleasingHost).unwrap();
        }
        let (fd2, fd3) = (p.bufs[2].fd(), p.bufs[3].fd());
        let ev = fake::events();
        let at = |e: fake::Event| {
            ev.iter()
                .position(|x| *x == e)
                .unwrap_or_else(|| panic!("{e:?} in {ev:?}"))
        };
        assert!(at(Map(fd2)) < at(Release(fd2)) && at(Release(fd2)) < at(Unmap(fd2)));
        assert!(at(Map(fd3)) < at(Release(fd3)) && at(Release(fd3)) < at(Unmap(fd3)));
    }

    #[test]
    fn dropping_the_pager_unmaps_everything_it_mapped() {
        let (d, p) = pager(&[100; 6], 450);
        let mut host = Host::default();
        for g in 2..6 {
            p.page_in(g, &mut host).unwrap();
        }
        assert!(d.mapped_bytes() > 0);
        drop(p);
        assert_eq!(d.mapped_bytes(), 0);
    }

    #[test]
    fn a_failed_flush_leaves_the_window_mapped() {
        let (_d, p) = pager(&[100; 6], 450);
        let mut host = Host::default();
        for g in 2..4 {
            p.page_in(g, &mut host).unwrap();
        }
        host.fail = true;
        let err = p
            .page_in(4, &mut host)
            .expect_err("the flush failure must surface");
        assert!(err.to_string().contains("dsp fault"));
        // Nothing was released or unmapped under a batch the DSP may still run.
        assert!(host.released.is_empty());
        assert!(p.bufs[2].is_mapped() && p.bufs[3].is_mapped());
        assert!(!p.bufs[4].is_mapped());
    }

    #[test]
    fn a_refused_unmap_keeps_the_layer_tracked_and_is_retried() {
        let (d, p) = pager(&[100; 6], 450);
        let mut host = Host::default();
        for g in 2..4 {
            p.page_in(g, &mut host).unwrap();
        }
        fake::with(|s| s.fail_munmap = true);
        let err = p
            .page_in(4, &mut host)
            .expect_err("the refused unmap must surface");
        assert!(err.to_string().contains("unmapping layer group 2"), "{err}");
        // Still mapped, so still counted and still in the window.
        assert!(p.bufs[2].is_mapped() && p.bufs[3].is_mapped());
        assert_eq!(d.mapped_bytes(), 400 * MIB);
        fake::with(|s| s.fail_munmap = false);
        p.page_in(4, &mut host).unwrap();
        assert_eq!(mapped(&p), [true, true, false, false, true, false]);
        assert_eq!(d.mapped_bytes(), 300 * MIB);
    }

    #[test]
    fn a_dsp_refusal_while_pinning_pins_fewer_layers() {
        fake::reset();
        fake::with(|s| s.fail_mmap = true);
        let driver = fake::driver();
        let bufs = (0..3)
            .map(|_| RpcmemBuffer::alloc(driver.clone(), 100 * MIB, false).unwrap())
            .collect();
        let p = ExpertPager::new(driver, bufs, 400 * MIB).expect("a refusal is not fatal");
        assert_eq!(p.pinned(), 0);
        // The same refusal at page-in time is reported with its layer.
        let err = p.page_in(0, &mut Host::default()).expect_err("refused map");
        let msg = err.to_string();
        assert!(msg.contains("mapping layer group 0"), "{msg}");
        assert_eq!(msg.matches("backend:").count(), 1, "one prefix only: {msg}");
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
        let p = Paging::default();
        assert_eq!(p.budget, 3584 * MIB);
        assert!(!p.should_page(gib, 200 * MIB, 100 * MIB));
        assert!(p.should_page(5 * gib, 200 * MIB, 100 * MIB));
        // The resident mappings count against the budget too.
        assert!(p.should_page(3 * gib, 400 * MIB, 100 * MIB));
        // So does the KV cache: a long context can tip a model over.
        assert!(!p.should_page(3 * gib, 300 * MIB, 100 * MIB));
        assert!(p.should_page(3 * gib, 300 * MIB, 400 * MIB));
        // Forcing pages a model that fits.
        assert!(Paging { force: true, ..p }.should_page(1, 0, 0));
    }

    #[test]
    fn auto_pages_only_when_paging_was_asked_for() {
        assert!(Paging::default().allow_paged, "an explicit NPU load pages");
        assert!(!Paging::default().for_auto().allow_paged);
        let forced = Paging {
            force: true,
            ..Paging::default()
        };
        assert!(forced.for_auto().allow_paged);
    }

    #[test]
    fn env_knobs_parse_like_the_other_hexagon_knobs() {
        let from = |pairs: &'static [(&'static str, &'static str)]| {
            Paging::from_lookup(move |k| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == k)
                    .map(|(_, v)| v.to_string())
            })
        };
        let d = Paging::default();
        let p = from(&[]);
        assert_eq!((p.force, p.budget), (d.force, d.budget));
        // An explicit NPU load (which reads the environment) may always page.
        assert!(p.allow_paged);
        assert!(from(&[("CERA_HEXAGON_PAGE_EXPERTS", "1")]).allow_paged);
        assert!(from(&[("CERA_HEXAGON_PAGE_EXPERTS", " TRUE ")]).force);
        assert!(from(&[("CERA_HEXAGON_PAGE_EXPERTS", "1")]).force);
        assert!(!from(&[("CERA_HEXAGON_PAGE_EXPERTS", "0")]).force);
        assert_eq!(
            from(&[("CERA_HEXAGON_MAP_BUDGET_MIB", "512")]).budget,
            512 * MIB
        );
        // KiB wins over MiB.
        let both = from(&[
            ("CERA_HEXAGON_MAP_BUDGET_KIB", "2048"),
            ("CERA_HEXAGON_MAP_BUDGET_MIB", "512"),
        ]);
        assert_eq!(both.budget, 2048 * 1024);
        // Never past the driver's ceiling; an unparsable value is ignored.
        assert_eq!(
            from(&[("CERA_HEXAGON_MAP_BUDGET_MIB", "999999")]).budget,
            crate::backend::hexagon::sys::DSP_MAP_CEILING
        );
        assert_eq!(
            from(&[("CERA_HEXAGON_MAP_BUDGET_MIB", "lots")]).budget,
            d.budget
        );
    }
}

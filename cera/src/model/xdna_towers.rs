//! AMD XDNA2 NPU vision and audio encoder towers via the dense `gemm_bf16` IRON kernel.
//!
//! All dense `linear` projections (ViT QKV/O/FFN + pixel-shuffle projector, d1 head
//! transformer stack + projector, Conformer QKV/P/O + macaron FFNs + conv pointwise +
//! adapter MLP) execute on the 32-core AIE2P `gemm_bf16` array (`M_TILE=128`,
//! `N_TILE=256`, `K=k_chunks*64`) batched with `XdnaRunList`. Non-linearities, layer
//! norms, conv2d/1d stem + depthwise passes, attention softmax/PV, and STFT/mel
//! reductions run on the host using the crate's SIMD/CPU primitives (`dot_f32`,
//! `layer_norm_inplace`, `gelu_inplace`, `gelu_erf_inplace`, `relu_inplace`,
//! `silu_inplace`, `softmax_inplace`, `add_inplace`, `axpy_inplace`, `glu_split`).
//!
//! Because `VitGpuOps` and `AudioEncoderGpuOps` methods return buffers directly rather
//! than `Result`, NPU allocation, sync, or execution errors are recorded into the
//! runner's fault slot (`last_err`) and surfaced as `Err(...)` at the surrounding
//! fallible boundary (`build`, `encode_image`, `encode_pcm`, `encode_mel`, `run`,
//! `run_projector`), matching the Metal/wgpu fault discipline.

use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::Result;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::backend::cpu::{
    add_inplace, axpy_inplace, conv1d, dot_f32, gelu_erf_inplace, gelu_inplace, glu_split,
    layer_norm_inplace, par_rows_n, relu_inplace, rmsnorm, rmsnorm_into, sigmoid_inplace,
    silu_inplace, silu_mul_inplace, softmax_inplace,
};
use crate::backend::xdna::device::{XdnaBo, XdnaDevice, XdnaKernel, XdnaRun};
use crate::backend::xdna::kernels::{
    GEMM_BF16_BUCKETS, GEMM_BF16_K_CHUNK, GEMM_BF16_M_TILE, GEMM_BF16_N_TILE, GEMM_BF16_XCLBIN,
};
use crate::backend::xdna::repack::{
    f32_to_bf16, gemm_bf16_bucket_chunks, pack_gemm_bf16_f32_slice, pack_gemm_bf16_weight_slice,
};
use crate::model::audio_encoder::{
    AudioEncoderWeights, ConvLayerWeights, HOP_LEN, LOG_MEL_EPS, N_FFT, NORM_VAR_EPS, PREEMPH,
    apply_rope_neox_head, conv_stem_lfm3a_features,
};
use crate::model::audio_encoder_gpu::{
    AudioEncoderGpuOps, AudioGpuEncode, Conv2dSpec, GpuAudioWeights, encode_audio_mel_gpu,
    encode_audio_pcm_gpu,
};
use crate::model::audio_preprocessor::N_FFT_BINS;
use crate::model::vision_encoder::{
    VisionEncoderWeights, apply_2d_vision_mrope_qk, depthwise_conv2d_same_into, vit_attention_gqa,
};
use crate::model::vision_encoder_gpu::{
    GpuStack, GpuVitWeights, VisionGpuEncode, VitGpuOps, VitStack, VitStackSpec, encode_image_gpu,
    warn_stack_unavailable,
};
use crate::model::weights::MmapWeight;
use crate::tensor::DType;

/// Runlist wait timeout for tower GEMM dispatches (`ms`).
const GEMM_WAIT_TIMEOUT_MS: u32 = 30_000;
/// Maximum `K` per single `gemm_bf16` invocation (`128 * 64 = 8192`).
const MAX_K_SLICE: usize = 128 * GEMM_BF16_K_CHUNK;
/// Byte size of one `C` output tile (`128 x 256` f32).
const C_TILE_BYTES: usize = GEMM_BF16_M_TILE * GEMM_BF16_N_TILE * std::mem::size_of::<f32>();
/// Byte size of the `gemm_bf16` runtime header packet BO (`1024 x u32` = `4096 B`).
const P_BO_BYTES: usize = 4096;

/// Host buffer handle backing [`VitGpuOps::Buf`] and [`AudioEncoderGpuOps::Buf`].
pub struct XdnaBuf(Arc<Mutex<Vec<f32>>>);

impl XdnaBuf {
    fn from_vec(v: Vec<f32>) -> Self {
        Self(Arc::new(Mutex::new(v)))
    }

    fn zeroed(len: usize) -> Self {
        Self::from_vec(vec![0.0; len])
    }

    fn read_clone(&self, len: usize) -> Vec<f32> {
        let g = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let copy_len = len.min(g.len());
        g[..copy_len].to_vec()
    }

    fn with_mut_and_ref<R>(
        dst: &Self,
        src: &Self,
        f: impl FnOnce(&mut [f32], Option<&[f32]>) -> R,
    ) -> R {
        if Arc::ptr_eq(&dst.0, &src.0) {
            let mut d = dst.0.lock().unwrap_or_else(|e| e.into_inner());
            return f(&mut d, None);
        }
        let dst_addr = Arc::as_ptr(&dst.0) as usize;
        let src_addr = Arc::as_ptr(&src.0) as usize;
        if dst_addr < src_addr {
            let mut d = dst.0.lock().unwrap_or_else(|e| e.into_inner());
            let s = src.0.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut d, Some(&s))
        } else {
            let s = src.0.lock().unwrap_or_else(|e| e.into_inner());
            let mut d = dst.0.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut d, Some(&s))
        }
    }

    fn with_read_bufs<const N: usize, R>(bufs: [&Self; N], f: impl FnOnce([&[f32]; N]) -> R) -> R {
        let mut order: [(usize, usize); N] =
            std::array::from_fn(|i| (Arc::as_ptr(&bufs[i].0) as usize, i));
        order.sort_unstable_by_key(|&(addr, _)| addr);
        let mut guards = Vec::with_capacity(N);
        let mut slots = [0usize; N];
        for (pos, &(addr, orig_idx)) in order.iter().enumerate() {
            if pos > 0 && addr == order[pos - 1].0 {
                slots[orig_idx] = guards.len() - 1;
            } else {
                guards.push(bufs[orig_idx].0.lock().unwrap_or_else(|e| e.into_inner()));
                slots[orig_idx] = guards.len() - 1;
            }
        }
        let slices: [&[f32]; N] = std::array::from_fn(|i| guards[slots[i]].as_slice());
        f(slices)
    }
}

/// Wrapper for static weight BOs synced once during upload and read under lock.
struct SendSyncBo(XdnaBo);

// SAFETY: `XdnaBo` wraps an XRT buffer object handle. A static weight BO is written
// and synced to the device during upload, then only referenced read-only inside
// `XdnaGemmRunner`'s mutex while launching `gemm_bf16` runs.
unsafe impl Send for SendSyncBo {}
// SAFETY: Shared read-only reference access (`&XdnaGemmWeight`) never mutates the BO.
unsafe impl Sync for SendSyncBo {}

/// One K-slice `[k_start .. k_start + k_len]` of a linear weight matrix `[out_dim, in_dim]`.
struct WeightKSlice {
    k_start: usize,
    k_len: usize,
    k_chunks: usize,
    tiles: Vec<SendSyncBo>,
}

/// Linear weight matrix `[out_dim, in_dim]` pre-packed into `gemm_bf16` BO tiles.
pub struct XdnaGemmWeight {
    out_dim: usize,
    in_dim: usize,
    k_slices: Vec<WeightKSlice>,
}

struct BucketSlot {
    kernel: XdnaKernel,
    p_bo: XdnaBo,
    a_pool: Vec<XdnaBo>,
    c_pool: Vec<XdnaBo>,
}

struct XdnaGemmInner {
    /// One slot per entry in [`GEMM_BF16_BUCKETS`], populated lazily after probing bucket 0.
    buckets: Vec<Option<BucketSlot>>,
    device: XdnaDevice,
}

// SAFETY: All fields of `XdnaGemmInner` (including XRT device/kernel/BO pointers)
// are serialized behind `XdnaGemmRunner::inner: Mutex<XdnaGemmInner>`, and BOs/kernels
// precede `device` in struct declaration order so Rust drops children before `device`.
unsafe impl Send for XdnaGemmInner {}
// SAFETY: Access only occurs while holding `XdnaGemmRunner::inner`.
unsafe impl Sync for XdnaGemmInner {}

/// Shared `gemm_bf16` XDNA2 NPU executor used by the vision and audio towers.
#[derive(Clone)]
pub struct XdnaGemmRunner {
    inner: Arc<Mutex<XdnaGemmInner>>,
    last_err: Arc<Mutex<Option<anyhow::Error>>>,
}

impl XdnaGemmRunner {
    fn load_bucket(device: &XdnaDevice, idx: usize) -> Result<BucketSlot> {
        let (k_chunks, inst_bytes) = GEMM_BF16_BUCKETS[idx];
        let words = bytes_to_u32_words(inst_bytes);
        let kernel = device
            .load_kernel(GEMM_BF16_XCLBIN, &words)
            .map_err(|e| anyhow::anyhow!("load gemm_bf16 kernel (k_chunks={k_chunks}): {e}"))?;
        let mut p_bo = kernel
            .alloc_bo(P_BO_BYTES, 6)
            .map_err(|e| anyhow::anyhow!("alloc gemm_bf16 param BO (k_chunks={k_chunks}): {e}"))?;
        let p_bytes = p_bo.as_mut_slice();
        p_bytes.fill(0);
        p_bytes[..4].copy_from_slice(&(k_chunks as u32).to_le_bytes());
        p_bo.sync_to_device()
            .map_err(|e| anyhow::anyhow!("sync gemm_bf16 param BO (k_chunks={k_chunks}): {e}"))?;
        Ok(BucketSlot {
            kernel,
            p_bo,
            a_pool: Vec::new(),
            c_pool: Vec::new(),
        })
    }

    /// Open XRT device 0 and validate `gemm_bf16.xclbin` by loading its smallest bucket.
    pub fn open() -> Result<Self> {
        let device = XdnaDevice::open(0).map_err(|e| anyhow::anyhow!("open XDNA device: {e}"))?;
        let mut buckets: Vec<Option<BucketSlot>> =
            (0..GEMM_BF16_BUCKETS.len()).map(|_| None).collect();
        buckets[0] = Some(Self::load_bucket(&device, 0)?);
        Ok(Self {
            inner: Arc::new(Mutex::new(XdnaGemmInner { buckets, device })),
            last_err: Arc::new(Mutex::new(None)),
        })
    }

    /// Open or reuse a process-wide shared `XdnaDevice` + `gemm_bf16` kernel context so
    /// multiple encoders (e.g. LFM3-Omni vision + audio) share one XRT hardware context
    /// reservation, while each caller owns its own error slot.
    pub fn open_shared() -> Result<Self> {
        #[cfg(test)]
        if crate::backend::xdna::sys::fake::active_shim().is_some() {
            thread_local! {
                static TEST_SHARED: Mutex<Weak<Mutex<XdnaGemmInner>>> =
                    const { Mutex::new(Weak::new()) };
            }
            return TEST_SHARED.with(|slot| {
                let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(inner) = guard.upgrade() {
                    return Ok(Self {
                        inner,
                        last_err: Arc::new(Mutex::new(None)),
                    });
                }
                let fresh = Self::open()?;
                *guard = Arc::downgrade(&fresh.inner);
                Ok(fresh)
            });
        }
        static SHARED: OnceLock<Mutex<Weak<Mutex<XdnaGemmInner>>>> = OnceLock::new();
        let slot = SHARED.get_or_init(|| Mutex::new(Weak::new()));
        let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(inner) = guard.upgrade() {
            return Ok(Self {
                inner,
                last_err: Arc::new(Mutex::new(None)),
            });
        }
        let fresh = Self::open()?;
        *guard = Arc::downgrade(&fresh.inner);
        Ok(fresh)
    }

    fn record_err(&self, err: anyhow::Error) {
        let mut slot = self.last_err.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    /// Take and clear the first recorded NPU fault, if any.
    pub fn take_err(&self) -> Option<anyhow::Error> {
        self.last_err
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    /// Ensure the bucket slot for `k_chunks` is loaded and return its mutable reference.
    fn bucket_mut(inner: &mut XdnaGemmInner, k_chunks: usize) -> Result<&mut BucketSlot> {
        let idx = GEMM_BF16_BUCKETS
            .iter()
            .position(|&(c, _)| c == k_chunks)
            .ok_or_else(|| anyhow::anyhow!("unsupported gemm_bf16 k_chunks={k_chunks}"))?;
        if inner.buckets[idx].is_none() {
            inner.buckets[idx] = Some(Self::load_bucket(&inner.device, idx)?);
        }
        Ok(inner.buckets[idx]
            .as_mut()
            .expect("bucket populated immediately above"))
    }

    fn upload_weight_with(
        &self,
        out_dim: usize,
        in_dim: usize,
        fill_tile: impl Fn(&mut [u8], usize, usize, usize, usize),
    ) -> Result<XdnaGemmWeight> {
        if out_dim == 0 || in_dim == 0 {
            return Ok(XdnaGemmWeight {
                out_dim,
                in_dim,
                k_slices: Vec::new(),
            });
        }
        let n_tiles = out_dim.div_ceil(GEMM_BF16_N_TILE);
        let mut k_slices = Vec::with_capacity(in_dim.div_ceil(MAX_K_SLICE));
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut k_start = 0;
        while k_start < in_dim {
            let k_len = (in_dim - k_start).min(MAX_K_SLICE);
            let k_chunks = gemm_bf16_bucket_chunks(k_len);
            let b_bytes =
                GEMM_BF16_N_TILE * k_chunks * GEMM_BF16_K_CHUNK * std::mem::size_of::<u16>();
            let bucket = Self::bucket_mut(&mut inner, k_chunks)?;
            let mut tiles = Vec::with_capacity(n_tiles);
            for nt in 0..n_tiles {
                let row_start = nt * GEMM_BF16_N_TILE;
                let mut bo = bucket
                    .kernel
                    .alloc_bo(b_bytes, 4)
                    .map_err(|e| anyhow::anyhow!("alloc weight BO: {e}"))?;
                fill_tile(bo.as_mut_slice(), row_start, k_start, k_len, k_chunks);
                bo.sync_to_device()
                    .map_err(|e| anyhow::anyhow!("sync weight BO: {e}"))?;
                tiles.push(SendSyncBo(bo));
            }
            k_slices.push(WeightKSlice {
                k_start,
                k_len,
                k_chunks,
                tiles,
            });
            k_start += k_len;
        }
        Ok(XdnaGemmWeight {
            out_dim,
            in_dim,
            k_slices,
        })
    }

    /// Pack and upload an [`MmapWeight`] `[out_dim, in_dim]` into `gemm_bf16` BO tiles.
    pub fn upload_mmap_weight(&self, w: &MmapWeight) -> Result<XdnaGemmWeight> {
        anyhow::ensure!(
            matches!(
                w.dtype,
                DType::F32
                    | DType::F16
                    | DType::BF16
                    | DType::Q4_0
                    | DType::Q4_1
                    | DType::Q8_0
                    | DType::Q4KM
                    | DType::Q5KM
                    | DType::Q6K
            ),
            "gemm_bf16 unsupported weight dtype {:?}",
            w.dtype
        );
        let block_size = w.dtype.block_size();
        anyhow::ensure!(
            w.cols.is_multiple_of(block_size),
            "gemm_bf16 weight cols ({}) must be a multiple of {:?} block_size ({block_size})",
            w.cols,
            w.dtype
        );
        let row_bytes = (w.cols / block_size)
            .checked_mul(w.dtype.block_bytes())
            .ok_or_else(|| anyhow::anyhow!("gemm_bf16 row byte size overflow"))?;
        let expect_bytes = w
            .rows
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("gemm_bf16 weight byte size overflow"))?;
        anyhow::ensure!(
            w.data().len() >= expect_bytes,
            "gemm_bf16 weight buffer too small: {} < {expect_bytes}",
            w.data().len()
        );
        self.upload_weight_with(
            w.rows,
            w.cols,
            |dst, row_start, k_start, k_len, k_chunks| {
                pack_gemm_bf16_weight_slice(dst, w, row_start, k_start, k_len, k_chunks);
            },
        )
    }

    /// Pack and upload a dense `f32` matrix `[out_dim, in_dim]` into `gemm_bf16` BO tiles.
    pub fn upload_f32_weight(
        &self,
        data: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<XdnaGemmWeight> {
        let expected_len = out_dim.checked_mul(in_dim).ok_or_else(|| {
            anyhow::anyhow!("gemm_bf16 weight size overflow for {out_dim}x{in_dim}")
        })?;
        anyhow::ensure!(
            data.len() == expected_len,
            "gemm_bf16 weight length {} != {out_dim} * {in_dim}",
            data.len()
        );
        self.upload_weight_with(
            out_dim,
            in_dim,
            |dst, row_start, k_start, k_len, k_chunks| {
                pack_gemm_bf16_f32_slice(
                    dst, data, out_dim, in_dim, row_start, k_start, k_len, k_chunks,
                );
            },
        )
    }

    /// Execute `Y[rows, out_dim] = X[rows, in_dim] @ W[out_dim, in_dim]^T` on XDNA2.
    pub fn matmul_f32(
        &self,
        x: &[f32],
        w: &XdnaGemmWeight,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            out_dim == w.out_dim && in_dim == w.in_dim,
            "gemm_bf16 weight shape [{}, {}] does not match linear [{out_dim}, {in_dim}]",
            w.out_dim,
            w.in_dim
        );
        let min_x = rows
            .checked_mul(in_dim)
            .ok_or_else(|| anyhow::anyhow!("gemm_bf16 input size overflow for {rows}x{in_dim}"))?;
        let out_len = rows.checked_mul(out_dim).ok_or_else(|| {
            anyhow::anyhow!("gemm_bf16 output size overflow for {rows}x{out_dim}")
        })?;
        anyhow::ensure!(
            x.len() >= min_x,
            "gemm_bf16 input length {} < {rows} * {in_dim}",
            x.len()
        );
        let mut out = vec![0.0f32; out_len];
        if rows == 0 || out_dim == 0 || in_dim == 0 || w.k_slices.is_empty() {
            return Ok(out);
        }

        let m_tiles = rows.div_ceil(GEMM_BF16_M_TILE);
        let n_tiles = out_dim.div_ceil(GEMM_BF16_N_TILE);
        let total_tiles = m_tiles * n_tiles;

        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for (slice_idx, slice) in w.k_slices.iter().enumerate() {
            let bucket = Self::bucket_mut(&mut inner, slice.k_chunks)?;
            let k_pad = slice.k_chunks * GEMM_BF16_K_CHUNK;
            let a_tile_bytes = GEMM_BF16_M_TILE * k_pad * std::mem::size_of::<u16>();

            while bucket.a_pool.len() < m_tiles {
                let bo = bucket
                    .kernel
                    .alloc_bo(a_tile_bytes, 3)
                    .map_err(|e| anyhow::anyhow!("alloc activation BO: {e}"))?;
                bucket.a_pool.push(bo);
            }
            while bucket.c_pool.len() < total_tiles {
                let bo = bucket
                    .kernel
                    .alloc_bo(C_TILE_BYTES, 5)
                    .map_err(|e| anyhow::anyhow!("alloc output BO: {e}"))?;
                bucket.c_pool.push(bo);
            }

            // Pack and sync each M_TILE=128 strip of A once, reused across N_TILEs.
            for mt in 0..m_tiles {
                let r_start = mt * GEMM_BF16_M_TILE;
                let r_count = (rows - r_start).min(GEMM_BF16_M_TILE);
                pack_activation_tile(
                    bucket.a_pool[mt].as_mut_slice(),
                    x,
                    in_dim,
                    r_start,
                    r_count,
                    slice.k_start,
                    slice.k_len,
                    k_pad,
                );
                bucket.a_pool[mt]
                    .sync_to_device()
                    .map_err(|e| anyhow::anyhow!("sync activation BO: {e}"))?;
            }

            // Submit all (m_tile, n_tile) tiles in one ordered runlist batch.
            // Declare `runs` before `runlist` so error unwind drops `runlist` before `runs`.
            {
                let mut runs: Vec<XdnaRun> = Vec::with_capacity(total_tiles);
                let mut runlist = bucket
                    .kernel
                    .create_runlist()
                    .map_err(|e| anyhow::anyhow!("create gemm_bf16 runlist: {e}"))?;
                for mt in 0..m_tiles {
                    for nt in 0..n_tiles {
                        let c_idx = mt * n_tiles + nt;
                        let run = bucket
                            .kernel
                            .create_run(&[
                                &bucket.a_pool[mt],
                                &slice.tiles[nt].0,
                                &bucket.c_pool[c_idx],
                                &bucket.p_bo,
                            ])
                            .map_err(|e| anyhow::anyhow!("create gemm_bf16 run: {e}"))?;
                        runlist
                            .add(&run)
                            .map_err(|e| anyhow::anyhow!("add gemm_bf16 run: {e}"))?;
                        runs.push(run);
                    }
                }
                runlist
                    .execute()
                    .map_err(|e| anyhow::anyhow!("execute gemm_bf16 runlist: {e}"))?;
                runlist
                    .wait(GEMM_WAIT_TIMEOUT_MS)
                    .map_err(|e| anyhow::anyhow!("wait gemm_bf16 runlist: {e}"))?;
                // Drop `runlist` before `runs`, and `runs` before reading/syncing `c_pool`.
                drop(runlist);
                drop(runs);
            }

            // Read back output tiles and unpack valid [r_count, n_count] regions.
            let accumulate = slice_idx > 0;
            for mt in 0..m_tiles {
                let r_start = mt * GEMM_BF16_M_TILE;
                let r_count = (rows - r_start).min(GEMM_BF16_M_TILE);
                for nt in 0..n_tiles {
                    let c_idx = mt * n_tiles + nt;
                    let bo_c = &mut bucket.c_pool[c_idx];
                    bo_c.sync_from_device()
                        .map_err(|e| anyhow::anyhow!("sync output BO: {e}"))?;
                    let c_start = nt * GEMM_BF16_N_TILE;
                    let c_count = (out_dim - c_start).min(GEMM_BF16_N_TILE);
                    unpack_output_tile(
                        bo_c.as_slice(),
                        &mut out,
                        out_dim,
                        r_start,
                        r_count,
                        c_start,
                        c_count,
                        accumulate,
                    );
                }
            }
        }

        Ok(out)
    }
}

fn bytes_to_u32_words(bytes: &[u8]) -> Vec<u32> {
    debug_assert!(bytes.len().is_multiple_of(4));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect()
}

/// Pack `[r_count, k_len]` f32 activation strip into row-major `[128, k_pad]` bf16 bytes.
#[allow(clippy::too_many_arguments)]
fn pack_activation_tile(
    dst_bytes: &mut [u8],
    x: &[f32],
    stride_k: usize,
    r_start: usize,
    r_count: usize,
    k_start: usize,
    k_len: usize,
    k_pad: usize,
) {
    if let Ok(dst_words) = bytemuck::try_cast_slice_mut::<u8, u16>(dst_bytes) {
        let total_rows = GEMM_BF16_M_TILE;
        for r in 0..r_count {
            let src_row =
                &x[(r_start + r) * stride_k + k_start..(r_start + r) * stride_k + k_start + k_len];
            let dst_row = &mut dst_words[r * k_pad..(r + 1) * k_pad];
            for (d, &v) in dst_row[..k_len].iter_mut().zip(src_row.iter()) {
                *d = f32_to_bf16(v);
            }
            if k_len < k_pad {
                dst_row[k_len..].fill(0);
            }
        }
        if r_count < total_rows {
            let tail_start = r_count * k_pad;
            let tail_end = dst_words.len().min(total_rows * k_pad);
            if tail_start < tail_end {
                dst_words[tail_start..tail_end].fill(0);
            }
        }
        return;
    }

    dst_bytes.fill(0);
    for r in 0..r_count {
        let src_row =
            &x[(r_start + r) * stride_k + k_start..(r_start + r) * stride_k + k_start + k_len];
        let dst_row_off = r * k_pad * 2;
        for (k, &v) in src_row.iter().enumerate() {
            let le = f32_to_bf16(v).to_le_bytes();
            dst_bytes[dst_row_off + k * 2] = le[0];
            dst_bytes[dst_row_off + k * 2 + 1] = le[1];
        }
    }
}

/// Copy (or accumulate for split-K) `[r_count, c_count]` from `[128, 256]` f32 BO bytes.
#[allow(clippy::too_many_arguments)]
fn unpack_output_tile(
    src_bytes: &[u8],
    out: &mut [f32],
    out_dim: usize,
    r_start: usize,
    r_count: usize,
    c_start: usize,
    c_count: usize,
    accumulate: bool,
) {
    if let Ok(src_f32) = bytemuck::try_cast_slice::<u8, f32>(src_bytes) {
        for r in 0..r_count {
            let src_off = r * GEMM_BF16_N_TILE;
            let src_row = &src_f32[src_off..src_off + c_count];
            let dst_row = &mut out
                [(r_start + r) * out_dim + c_start..(r_start + r) * out_dim + c_start + c_count];
            if accumulate {
                for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                    *d += s;
                }
            } else {
                dst_row.copy_from_slice(src_row);
            }
        }
        return;
    }

    for r in 0..r_count {
        let src_row_off = r * GEMM_BF16_N_TILE * 4;
        let dst_row = &mut out
            [(r_start + r) * out_dim + c_start..(r_start + r) * out_dim + c_start + c_count];
        for (c, dst_val) in dst_row.iter_mut().enumerate() {
            let off = src_row_off + c * 4;
            let val = f32::from_le_bytes([
                src_bytes[off],
                src_bytes[off + 1],
                src_bytes[off + 2],
                src_bytes[off + 3],
            ]);
            if accumulate {
                *dst_val += val;
            } else {
                *dst_val = val;
            }
        }
    }
}

/// Hybrid parallel helper: uses `rayon` when compiled with `parallel`, sequential otherwise.
fn maybe_par_for(count: usize, f: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..count).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "parallel"))]
    {
        for idx in 0..count {
            f(idx);
        }
    }
}

/// XDNA implementation of [`VitGpuOps`].
pub struct XdnaVitOps {
    runner: XdnaGemmRunner,
    attn_limit: usize,
}

impl XdnaVitOps {
    /// Create vision ops backed by an open [`XdnaGemmRunner`].
    pub fn new(runner: XdnaGemmRunner) -> Self {
        Self {
            runner,
            attn_limit: usize::MAX,
        }
    }

    /// Override the attention sequence length cap advertised through [`VitGpuOps`].
    pub fn with_attention_limit(mut self, limit: usize) -> Self {
        self.attn_limit = limit;
        self
    }

    /// Drain any fault captured during non-fallible [`VitGpuOps`] calls.
    pub fn take_err(&self) -> Option<anyhow::Error> {
        self.runner.take_err()
    }
}

impl VitGpuOps for XdnaVitOps {
    type Buf = XdnaBuf;
    type Weight = XdnaGemmWeight;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        XdnaBuf::from_vec(data.to_vec())
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        buf.read_clone(len)
    }

    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight {
        match self.runner.upload_mmap_weight(w) {
            Ok(packed) => packed,
            Err(e) => {
                let out_dim = w.rows;
                let in_dim = w.cols;
                self.runner.record_err(e);
                XdnaGemmWeight {
                    out_dim,
                    in_dim,
                    k_slices: Vec::new(),
                }
            }
        }
    }

    fn upload_weight_f32(&self, data: &[f32], out_dim: usize, in_dim: usize) -> Self::Weight {
        match self.runner.upload_f32_weight(data, out_dim, in_dim) {
            Ok(packed) => packed,
            Err(e) => {
                self.runner.record_err(e);
                XdnaGemmWeight {
                    out_dim,
                    in_dim,
                    k_slices: Vec::new(),
                }
            }
        }
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        let x_data = x.0.lock().unwrap_or_else(|e| e.into_inner());
        match self.runner.matmul_f32(&x_data, w, rows, out_dim, in_dim) {
            Ok(out) => XdnaBuf::from_vec(out),
            Err(e) => {
                self.runner.record_err(e);
                XdnaBuf::zeroed(rows * out_dim)
            }
        }
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        if rows == 0 || dim == 0 {
            return;
        }
        XdnaBuf::with_mut_and_ref(x, bias, |d, b_opt| {
            if let Some(b) = b_opt {
                for row in d[..rows * dim].chunks_exact_mut(dim) {
                    add_inplace(row, &b[..dim]);
                }
            } else {
                let b = d[..dim].to_vec();
                for row in d[..rows * dim].chunks_exact_mut(dim) {
                    add_inplace(row, &b);
                }
            }
        });
    }

    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        if rows == 0 || dim == 0 {
            return XdnaBuf::zeroed(0);
        }
        XdnaBuf::with_read_bufs([src, weight, bias], |[s, w, b]| {
            let mut out = s[..rows * dim].to_vec();
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                out.par_chunks_exact_mut(dim).for_each(|row| {
                    layer_norm_inplace(row, &w[..dim], &b[..dim], eps);
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for row in out.chunks_exact_mut(dim) {
                    layer_norm_inplace(row, &w[..dim], &b[..dim], eps);
                }
            }
            XdnaBuf::from_vec(out)
        })
    }

    fn gelu(&self, x: &Self::Buf, len: usize) {
        let mut d = x.0.lock().unwrap_or_else(|e| e.into_inner());
        gelu_inplace(&mut d[..len]);
    }

    fn relu(&self, x: &Self::Buf, len: usize) {
        let mut d = x.0.lock().unwrap_or_else(|e| e.into_inner());
        relu_inplace(&mut d[..len]);
    }

    fn supports_relu(&self) -> bool {
        true
    }

    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf {
        let dim = n_head * head_dim;
        if tokens == 0 || dim == 0 {
            return XdnaBuf::zeroed(tokens * dim);
        }
        let q_vec = q.read_clone(tokens * dim);
        let k_vec = k.read_clone(tokens * dim);
        let v_vec = v.read_clone(tokens * dim);

        // Rearrange to `[n_head, tokens, head_dim]` so `dot_f32` and the value
        // accumulation sweep contiguous `[head_dim]` slices per key token.
        let head_stride = tokens * head_dim;
        let mut k_hm = vec![0.0f32; n_head * head_stride];
        let mut v_hm = vec![0.0f32; n_head * head_stride];
        for t in 0..tokens {
            for h in 0..n_head {
                let src = t * dim + h * head_dim;
                let dst = h * head_stride + t * head_dim;
                k_hm[dst..dst + head_dim].copy_from_slice(&k_vec[src..src + head_dim]);
                v_hm[dst..dst + head_dim].copy_from_slice(&v_vec[src..src + head_dim]);
            }
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out_hm = vec![0.0f32; n_head * head_stride];
        let out_hm_ptr = out_hm.as_mut_ptr() as usize;

        maybe_par_for(n_head, |h| {
            let k_head = &k_hm[h * head_stride..(h + 1) * head_stride];
            let v_head = &v_hm[h * head_stride..(h + 1) * head_stride];
            let mut scores = vec![0.0f32; tokens];
            // SAFETY: distinct heads write disjoint `[h * head_stride .. (h+1) * head_stride]` slices.
            let out_head = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_hm_ptr as *mut f32).add(h * head_stride),
                    head_stride,
                )
            };
            for qt in 0..tokens {
                let q_row = &q_vec[qt * dim + h * head_dim..qt * dim + (h + 1) * head_dim];
                for kt in 0..tokens {
                    let k_row = &k_head[kt * head_dim..(kt + 1) * head_dim];
                    scores[kt] = dot_f32(q_row, k_row) * scale;
                }
                softmax_inplace(&mut scores);
                let out_row = &mut out_head[qt * head_dim..(qt + 1) * head_dim];
                out_row.fill(0.0);
                for kt in 0..tokens {
                    axpy_inplace(
                        out_row,
                        &v_head[kt * head_dim..(kt + 1) * head_dim],
                        scores[kt],
                    );
                }
            }
        });

        let mut out = vec![0.0f32; tokens * dim];
        for t in 0..tokens {
            for h in 0..n_head {
                let src = h * head_stride + t * head_dim;
                let dst = t * dim + h * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&out_hm[src..src + head_dim]);
            }
        }
        XdnaBuf::from_vec(out)
    }

    fn attention_token_limit(&self, _head_dim: usize) -> usize {
        self.attn_limit
    }

    fn max_rows(&self) -> usize {
        usize::MAX
    }

    fn max_buffer_bytes(&self) -> u64 {
        u64::MAX
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        if len == 0 {
            return;
        }
        XdnaBuf::with_mut_and_ref(dst, src, |d, s_opt| {
            if let Some(s) = s_opt {
                add_inplace(&mut d[..len], &s[..len]);
            } else {
                for v in &mut d[..len] {
                    *v += *v;
                }
            }
        });
    }

    fn supports_lfm3(&self) -> bool {
        true
    }

    fn rmsnorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        if rows == 0 || dim == 0 {
            return XdnaBuf::zeroed(0);
        }
        XdnaBuf::with_read_bufs([src, weight], |[x, w]| {
            let mut out = vec![0.0f32; rows * dim];
            let x_rows = &x[..rows * dim];
            let w_row = &w[..dim];
            par_rows_n(&mut out, dim, 8, |(r, dst_row)| {
                rmsnorm_into(&x_rows[r * dim..(r + 1) * dim], dst_row, w_row, eps);
            });
            XdnaBuf::from_vec(out)
        })
    }

    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, len: usize) {
        if len == 0 {
            return;
        }
        XdnaBuf::with_mut_and_ref(gate, up, |g, u_opt| {
            if let Some(u) = u_opt {
                silu_mul_inplace(&mut g[..len], &u[..len]);
            } else {
                let u = g[..len].to_vec();
                silu_mul_inplace(&mut g[..len], &u);
            }
        });
    }

    fn lfm3_conv_spatial(
        &self,
        proj: &Self::Buf,
        proj_rows: usize,
        depthwise_w: &[f32],
        depthwise_b: Option<&[f32]>,
        grid_w: usize,
        grid_h: usize,
        d: usize,
        kernel_size: usize,
    ) -> Self::Buf {
        let n_tokens = grid_w * grid_h;
        if n_tokens == 0 || d == 0 {
            return XdnaBuf::zeroed(0);
        }
        let proj_guard = proj.0.lock().unwrap_or_else(|e| e.into_inner());
        let proj_slice = &proj_guard[..n_tokens * proj_rows];

        let mut gated = vec![0.0f32; n_tokens * d];
        if proj_rows == 3 * d {
            par_rows_n(&mut gated, d, 8, |(t, dst)| {
                let row = &proj_slice[t * 3 * d..(t + 1) * 3 * d];
                let first = &row[0..d];
                let third = &row[2 * d..3 * d];
                for c in 0..d {
                    dst[c] = first[c] * third[c];
                }
            });
        } else {
            par_rows_n(&mut gated, d, 8, |(t, dst)| {
                let row = &proj_slice[t * 2 * d..(t + 1) * 2 * d];
                let b_part = &row[0..d];
                let c_part = &row[d..2 * d];
                for c in 0..d {
                    dst[c] = b_part[c] * c_part[c];
                }
            });
        }

        let mut out = vec![0.0f32; n_tokens * d];
        depthwise_conv2d_same_into(
            &gated,
            depthwise_w,
            depthwise_b,
            grid_w,
            grid_h,
            d,
            kernel_size,
            &mut out,
        );
        if proj_rows == 3 * d {
            par_rows_n(&mut out, d, 8, |(t, dst)| {
                let middle = &proj_slice[t * 3 * d + d..t * 3 * d + 2 * d];
                for c in 0..d {
                    dst[c] *= middle[c];
                }
            });
        }
        XdnaBuf::from_vec(out)
    }

    fn lfm3_gqa_mrope_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        grid_w: usize,
        grid_h: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Self::Buf {
        let tokens = grid_w * grid_h;
        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        if tokens == 0 || q_dim == 0 || kv_dim == 0 {
            return XdnaBuf::zeroed(tokens * q_dim);
        }
        let mut q_vec = q.read_clone(tokens * q_dim);
        let mut k_vec = k.read_clone(tokens * kv_dim);
        let v_vec = v.read_clone(tokens * kv_dim);

        apply_2d_vision_mrope_qk(
            &mut q_vec, &mut k_vec, grid_w, grid_h, n_head, n_head_kv, head_dim, rope_theta,
        );

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let chunk_q = q_dim.saturating_sub(1).clamp(1, 32);
        let padded = tokens.next_multiple_of(chunk_q);
        let mut heads = vec![0.0f32; n_head * padded * head_dim];
        let mut out = vec![0.0f32; tokens * q_dim];
        vit_attention_gqa(
            &q_vec, &k_vec, &v_vec, &mut out, &mut heads, tokens, n_head, n_head_kv, head_dim,
            scale,
        );
        XdnaBuf::from_vec(out)
    }
}

/// XDNA implementation of [`VitStack::run`], forwarding through [`GpuStack<XdnaVitOps>`].
impl VitStack for GpuStack<XdnaVitOps> {
    fn run(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        let _ = self.ops.take_err();
        let out = GpuStack::run(self, x, tokens)?;
        if let Some(e) = self.ops.take_err() {
            return Err(e);
        }
        Ok(out)
    }
}

struct XdnaProjector {
    in_dim: usize,
    inter_dim: usize,
    out_dim: usize,
    mm1_w: XdnaGemmWeight,
    mm1_b: XdnaBuf,
    mm2_w: XdnaGemmWeight,
    mm2_b: XdnaBuf,
}

/// [`VitStack`] on AMD XDNA2: block stack + optional two-layer exact-GELU projector.
pub struct XdnaVitStack {
    stack: GpuStack<XdnaVitOps>,
    projector: Option<XdnaProjector>,
    run_lock: Mutex<()>,
}

impl XdnaVitStack {
    /// Build an XDNA transformer stack (and optional projector) from `spec`.
    pub fn build(runner: XdnaGemmRunner, spec: &VitStackSpec, max_rows: usize) -> Result<Self> {
        let ops = XdnaVitOps::new(runner).with_attention_limit(max_rows);
        let _ = ops.take_err();
        let stack = GpuStack::build(ops, spec)?;
        if let Some(e) = stack.ops.take_err() {
            return Err(e);
        }

        let projector = if let Some(proj) = &spec.projector {
            anyhow::ensure!(
                proj.mm1_b.len() == proj.mm1.rows
                    && proj.mm2.cols == proj.mm1.rows
                    && proj.mm2_b.len() == proj.mm2.rows,
                "vit stack projector dimensions disagree (mm1 [{}x{}] / bias {}, mm2 [{}x{}] / bias {})",
                proj.mm1.rows,
                proj.mm1.cols,
                proj.mm1_b.len(),
                proj.mm2.rows,
                proj.mm2.cols,
                proj.mm2_b.len(),
            );
            let mm1_w = stack.ops.runner.upload_mmap_weight(&proj.mm1)?;
            let mm1_b = stack.ops.upload(&proj.mm1_b);
            let mm2_w = stack.ops.runner.upload_mmap_weight(&proj.mm2)?;
            let mm2_b = stack.ops.upload(&proj.mm2_b);
            Some(XdnaProjector {
                in_dim: proj.mm1.cols,
                inter_dim: proj.mm1.rows,
                out_dim: proj.mm2.rows,
                mm1_w,
                mm1_b,
                mm2_w,
                mm2_b,
            })
        } else {
            None
        };

        Ok(Self {
            stack,
            projector,
            run_lock: Mutex::new(()),
        })
    }
}

impl VitStack for XdnaVitStack {
    fn run(&self, x: &[f32], tokens: usize) -> Result<Vec<f32>> {
        let _serial = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        VitStack::run(&self.stack, x, tokens)
    }

    fn run_projector(&self, pooled: &[f32]) -> Result<Option<Vec<f32>>> {
        let Some(proj) = &self.projector else {
            return Ok(None);
        };
        anyhow::ensure!(
            proj.in_dim > 0 && pooled.len().is_multiple_of(proj.in_dim),
            "vit projector input length {} is not a multiple of width {}",
            pooled.len(),
            proj.in_dim
        );
        let rows = pooled.len() / proj.in_dim;
        if rows == 0 {
            return Ok(Some(Vec::new()));
        }
        let _serial = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.stack.ops.take_err();
        let x = self.stack.ops.upload(pooled);
        let mid = self
            .stack
            .ops
            .linear(&x, &proj.mm1_w, rows, proj.inter_dim, proj.in_dim);
        self.stack
            .ops
            .bias_add(&mid, &proj.mm1_b, rows, proj.inter_dim);
        {
            let mut d = mid.0.lock().unwrap_or_else(|e| e.into_inner());
            gelu_erf_inplace(&mut d[..rows * proj.inter_dim]);
        }
        let out = self
            .stack
            .ops
            .linear(&mid, &proj.mm2_w, rows, proj.out_dim, proj.inter_dim);
        self.stack
            .ops
            .bias_add(&out, &proj.mm2_b, rows, proj.out_dim);
        let res = self.stack.ops.download(&out, rows * proj.out_dim);
        if let Some(e) = self.stack.ops.take_err() {
            return Err(e);
        }
        Ok(Some(res))
    }
}

struct XdnaVisionEncoder {
    ops: XdnaVitOps,
    weights: GpuVitWeights<XdnaVitOps>,
    encode_lock: Mutex<()>,
}

impl VisionGpuEncode for XdnaVisionEncoder {
    fn backend_label(&self) -> &'static str {
        "xdna"
    }

    fn max_tokens(&self) -> usize {
        self.ops.attn_limit
    }

    fn encode_image(&self, pixels: &[f32], grid_w: usize, grid_h: usize) -> Result<Vec<f32>> {
        let _serial = self.encode_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.ops.take_err();
        let out = encode_image_gpu(&self.ops, &self.weights, pixels, grid_w, grid_h)?;
        if let Some(e) = self.ops.take_err() {
            return Err(e);
        }
        Ok(out)
    }

    fn encode_image_async<'a>(
        &'a self,
        pixels: &'a [f32],
        grid_w: usize,
        grid_h: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move { self.encode_image(pixels, grid_w, grid_h) })
    }
}

/// Build an XDNA [`VitStack`] for `spec`, returning `None` if XRT/NPU is unavailable.
pub(crate) fn try_xdna_vit_stack(
    spec: &VitStackSpec,
    max_rows: usize,
) -> Option<Arc<dyn VitStack>> {
    let runner = match XdnaGemmRunner::open() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("vit stack: XDNA backend unavailable ({e:#}); falling back");
            return None;
        }
    };
    match XdnaVitStack::build(runner, spec, max_rows) {
        Ok(stack) => {
            tracing::info!("vit stack: using AMD XDNA2 NPU backend");
            Some(Arc::new(stack))
        }
        Err(e) => {
            warn_stack_unavailable("XDNA stack spec", e);
            None
        }
    }
}

/// Build an XDNA [`VisionGpuEncode`] for `weights`, returning `None` if unavailable.
pub(crate) fn try_xdna_vision_encoder(
    weights: &VisionEncoderWeights,
) -> Option<Arc<dyn VisionGpuEncode>> {
    let runner = match XdnaGemmRunner::open_shared() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("vision encoder: XDNA backend unavailable ({e:#}); falling back");
            return None;
        }
    };
    let ops = XdnaVitOps::new(runner);
    let _ = ops.take_err();
    let gpu_w = match GpuVitWeights::try_build(&ops, weights) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("vision encoder: XDNA backend unavailable for this model: {e:#}");
            eprintln!("vision encoder: XDNA backend unavailable for this model: {e:#}");
            return None;
        }
    };
    if let Some(e) = ops.take_err() {
        tracing::warn!("vision encoder: XDNA weight upload failed, falling back: {e:#}");
        eprintln!("vision encoder: XDNA weight upload failed, falling back: {e:#}");
        return None;
    }
    tracing::info!("vision encoder: using AMD XDNA2 NPU backend");
    Some(Arc::new(XdnaVisionEncoder {
        ops,
        weights: gpu_w,
        encode_lock: Mutex::new(()),
    }))
}

/// XDNA implementation of [`AudioEncoderGpuOps`].
pub struct XdnaAudioOps {
    vit: XdnaVitOps,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl XdnaAudioOps {
    /// Create audio encoder ops backed by an open [`XdnaGemmRunner`].
    pub fn new(runner: XdnaGemmRunner) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self {
            vit: XdnaVitOps::new(runner),
            fft,
        }
    }

    fn take_err(&self) -> Option<anyhow::Error> {
        self.vit.take_err()
    }
}

impl AudioEncoderGpuOps for XdnaAudioOps {
    type Buf = XdnaBuf;
    type Weight = XdnaGemmWeight;

    fn upload(&self, data: &[f32]) -> Self::Buf {
        self.vit.upload(data)
    }

    fn download(&self, buf: &Self::Buf, len: usize) -> Vec<f32> {
        self.vit.download(buf, len)
    }

    fn upload_weight(&self, w: &MmapWeight) -> Self::Weight {
        self.vit.upload_weight(w)
    }

    fn linear(
        &self,
        x: &Self::Buf,
        w: &Self::Weight,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Self::Buf {
        self.vit.linear(x, w, rows, out_dim, in_dim)
    }

    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, rows: usize, dim: usize) {
        self.vit.bias_add(x, bias, rows, dim);
    }

    fn layernorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        self.vit.layernorm(src, weight, bias, eps, rows, dim)
    }

    fn relu(&self, x: &Self::Buf, len: usize) {
        self.vit.relu(x, len);
    }

    fn silu(&self, x: &Self::Buf, len: usize) {
        let mut d = x.0.lock().unwrap_or_else(|e| e.into_inner());
        silu_inplace(&mut d[..len]);
    }

    fn gelu_erf(&self, x: &Self::Buf, len: usize) {
        let mut d = x.0.lock().unwrap_or_else(|e| e.into_inner());
        gelu_erf_inplace(&mut d[..len]);
    }

    fn add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize) {
        self.vit.add(dst, src, len);
    }

    fn scaled_add(&self, dst: &Self::Buf, src: &Self::Buf, len: usize, scale: f32) {
        if len == 0 {
            return;
        }
        XdnaBuf::with_mut_and_ref(dst, src, |d, s_opt| {
            if let Some(s) = s_opt {
                axpy_inplace(&mut d[..len], &s[..len], scale);
            } else {
                let s = d[..len].to_vec();
                axpy_inplace(&mut d[..len], &s, scale);
            }
        });
    }

    fn conv2d(
        &self,
        input: &Self::Buf,
        weight: &Self::Buf,
        bias: &Self::Buf,
        spec: &Conv2dSpec,
    ) -> Self::Buf {
        let total = spec.out_len();
        if total == 0 {
            return XdnaBuf::zeroed(0);
        }
        XdnaBuf::with_read_bufs([input, weight, bias], |[in_data, w_data, b_data]| {
            let in_per_group = spec.in_ch / spec.groups;
            let out_per_group = spec.out_ch / spec.groups;
            let plane_out = spec.h_out * spec.w_out;
            let plane_in = spec.h_in * spec.w_in;
            let k_plane = spec.kh * spec.kw;

            let mut out = vec![0.0f32; total];
            let out_ptr = out.as_mut_ptr() as usize;

            maybe_par_for(spec.out_ch, |oc| {
                let g = oc / out_per_group;
                let bias_val = b_data[oc];
                // SAFETY: distinct `oc` channels own disjoint `[oc * plane_out .. (oc+1) * plane_out]` output planes.
                let out_plane = unsafe {
                    std::slice::from_raw_parts_mut(
                        (out_ptr as *mut f32).add(oc * plane_out),
                        plane_out,
                    )
                };
                for oh in 0..spec.h_out {
                    let ih_base = (oh * spec.stride_h) as isize - spec.pad_h as isize;
                    for ow in 0..spec.w_out {
                        let iw_base = (ow * spec.stride_w) as isize - spec.pad_w as isize;
                        let mut acc = bias_val;
                        for ic_local in 0..in_per_group {
                            let ic = g * in_per_group + ic_local;
                            let w_row = (oc * in_per_group + ic_local) * k_plane;
                            let in_plane = ic * plane_in;
                            for ki in 0..spec.kh {
                                let ih = ih_base + ki as isize;
                                if ih < 0 || ih >= spec.h_in as isize {
                                    continue;
                                }
                                let in_row = in_plane + ih as usize * spec.w_in;
                                for kj in 0..spec.kw {
                                    let iw = iw_base + kj as isize;
                                    if iw < 0 || iw >= spec.w_in as isize {
                                        continue;
                                    }
                                    acc += w_data[w_row + ki * spec.kw + kj]
                                        * in_data[in_row + iw as usize];
                                }
                            }
                        }
                        out_plane[oh * spec.w_out + ow] = acc;
                    }
                }
            });

            XdnaBuf::from_vec(out)
        })
    }

    fn transpose_blocked(&self, src: &Self::Buf, a: usize, b: usize, k: usize) -> Self::Buf {
        let s = src.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut dst = vec![0.0f32; a * b * k];
        for ia in 0..a {
            for ib in 0..b {
                let src_off = (ia * b + ib) * k;
                let dst_off = (ib * a + ia) * k;
                dst[dst_off..dst_off + k].copy_from_slice(&s[src_off..src_off + k]);
            }
        }
        XdnaBuf::from_vec(dst)
    }

    fn glu_split(&self, src: &Self::Buf, rows: usize, n: usize) -> Self::Buf {
        let s = src.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut dst = vec![0.0f32; rows * n];
        for r in 0..rows {
            glu_split(&s[r * 2 * n..(r + 1) * 2 * n], &mut dst[r * n..(r + 1) * n]);
        }
        XdnaBuf::from_vec(dst)
    }

    fn chan_affine_silu(
        &self,
        x: &Self::Buf,
        w: &Self::Buf,
        b: &Self::Buf,
        channels: usize,
        t: usize,
    ) {
        if channels == 0 || t == 0 {
            return;
        }
        let w_data = w.read_clone(channels);
        let b_data = b.read_clone(channels);
        let mut x_data = x.0.lock().unwrap_or_else(|e| e.into_inner());
        for c in 0..channels {
            let wc = w_data[c];
            let bc = b_data[c];
            let row = &mut x_data[c * t..(c + 1) * t];
            for v in row.iter_mut() {
                *v = *v * wc + bc;
            }
            silu_inplace(row);
        }
    }

    fn xl_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        p: &Self::Buf,
        bias_u: &Self::Buf,
        bias_v: &Self::Buf,
        tokens: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Self::Buf {
        let dim = n_head * head_dim;
        if tokens == 0 || dim == 0 {
            return XdnaBuf::zeroed(tokens * dim);
        }
        let q_vec = q.read_clone(tokens * dim);
        let k_vec = k.read_clone(tokens * dim);
        let v_vec = v.read_clone(tokens * dim);
        let p_rows = 2 * tokens - 1;
        let p_vec = p.read_clone(p_rows * dim);
        let u_vec = bias_u.read_clone(dim);
        let vb_vec = bias_v.read_clone(dim);

        let kv_stride = tokens * head_dim;
        let p_stride = p_rows * head_dim;
        let mut k_hm = vec![0.0f32; n_head * kv_stride];
        let mut v_hm = vec![0.0f32; n_head * kv_stride];
        let mut p_hm = vec![0.0f32; n_head * p_stride];
        for t in 0..tokens {
            for h in 0..n_head {
                let src = t * dim + h * head_dim;
                let dst = h * kv_stride + t * head_dim;
                k_hm[dst..dst + head_dim].copy_from_slice(&k_vec[src..src + head_dim]);
                v_hm[dst..dst + head_dim].copy_from_slice(&v_vec[src..src + head_dim]);
            }
        }
        for pr in 0..p_rows {
            for h in 0..n_head {
                let src = pr * dim + h * head_dim;
                let dst = h * p_stride + pr * head_dim;
                p_hm[dst..dst + head_dim].copy_from_slice(&p_vec[src..src + head_dim]);
            }
        }

        let scale = 1.0f64 / (head_dim as f64).sqrt();
        let t_minus_1 = tokens - 1;
        let mut out_hm = vec![0.0f32; n_head * kv_stride];
        let out_hm_ptr = out_hm.as_mut_ptr() as usize;

        maybe_par_for(n_head, |h| {
            let k_head = &k_hm[h * kv_stride..(h + 1) * kv_stride];
            let v_head = &v_hm[h * kv_stride..(h + 1) * kv_stride];
            let p_head = &p_hm[h * p_stride..(h + 1) * p_stride];
            let u_h = &u_vec[h * head_dim..(h + 1) * head_dim];
            let v_h = &vb_vec[h * head_dim..(h + 1) * head_dim];
            let mut q_u = vec![0.0f32; head_dim];
            let mut q_v = vec![0.0f32; head_dim];
            let mut scores = vec![0.0f32; tokens];
            // SAFETY: distinct heads write disjoint `[h * kv_stride .. (h+1) * kv_stride]` slices.
            let out_head = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_hm_ptr as *mut f32).add(h * kv_stride),
                    kv_stride,
                )
            };
            for qt in 0..tokens {
                let q_row = &q_vec[qt * dim + h * head_dim..qt * dim + (h + 1) * head_dim];
                for d in 0..head_dim {
                    q_u[d] = q_row[d] + u_h[d];
                    q_v[d] = q_row[d] + v_h[d];
                }
                for kt in 0..tokens {
                    let pos_idx = t_minus_1 + kt - qt;
                    let k_row = &k_head[kt * head_dim..(kt + 1) * head_dim];
                    let p_row = &p_head[pos_idx * head_dim..(pos_idx + 1) * head_dim];
                    let ac = dot_f32(&q_u, k_row) as f64;
                    let bd = dot_f32(&q_v, p_row) as f64;
                    scores[kt] = ((ac + bd) * scale) as f32;
                }
                softmax_inplace(&mut scores);
                let out_row = &mut out_head[qt * head_dim..(qt + 1) * head_dim];
                out_row.fill(0.0);
                for kt in 0..tokens {
                    axpy_inplace(
                        out_row,
                        &v_head[kt * head_dim..(kt + 1) * head_dim],
                        scores[kt],
                    );
                }
            }
        });

        let mut out = vec![0.0f32; tokens * dim];
        for t in 0..tokens {
            for h in 0..n_head {
                let src = h * kv_stride + t * head_dim;
                let dst = t * dim + h * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&out_hm[src..src + head_dim]);
            }
        }
        XdnaBuf::from_vec(out)
    }

    fn stft_frames(
        &self,
        pcm: &Self::Buf,
        hann: &Self::Buf,
        n_samples: usize,
        n_frames: usize,
    ) -> Self::Buf {
        if n_frames == 0 {
            return XdnaBuf::zeroed(0);
        }
        XdnaBuf::with_read_bufs([pcm, hann], |[p, h]| {
            let pad = N_FFT / 2;
            let mut frames = vec![0.0f32; n_frames * N_FFT];
            for ti in 0..n_frames {
                let base = ti * HOP_LEN;
                let dst = &mut frames[ti * N_FFT..(ti + 1) * N_FFT];
                for k in 0..N_FFT {
                    let idx = base + k;
                    if idx < pad || idx >= pad + n_samples {
                        dst[k] = 0.0;
                    } else {
                        let s = idx - pad;
                        let cur = p[s];
                        let sample = if s == 0 {
                            cur
                        } else {
                            cur - PREEMPH * p[s - 1]
                        };
                        dst[k] = sample * h[k];
                    }
                }
            }
            XdnaBuf::from_vec(frames)
        })
    }

    fn power_spec(&self, frames: &Self::Buf, _twiddle: &Self::Buf, n_frames: usize) -> Self::Buf {
        let fr = frames.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut fft_buf = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = vec![0.0f32; n_frames * N_FFT_BINS];
        for ti in 0..n_frames {
            let src = &fr[ti * N_FFT..(ti + 1) * N_FFT];
            for (fb, &s) in fft_buf.iter_mut().zip(src.iter()) {
                *fb = Complex32::new(s, 0.0);
            }
            self.fft.process(&mut fft_buf);
            let dst = &mut power[ti * N_FFT_BINS..(ti + 1) * N_FFT_BINS];
            for (p, c) in dst.iter_mut().zip(fft_buf.iter().take(N_FFT_BINS)) {
                *p = ((c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64)) as f32;
            }
        }
        XdnaBuf::from_vec(power)
    }

    fn mel_project(
        &self,
        power: &Self::Buf,
        filters: &Self::Buf,
        n_mel: usize,
        n_frames: usize,
    ) -> Self::Buf {
        if n_mel == 0 || n_frames == 0 {
            return XdnaBuf::zeroed(0);
        }
        XdnaBuf::with_read_bufs([power, filters], |[p, f]| {
            let mut mel = vec![0.0f32; n_mel * n_frames];
            for mi in 0..n_mel {
                let f_row = &f[mi * N_FFT_BINS..(mi + 1) * N_FFT_BINS];
                for ti in 0..n_frames {
                    let p_row = &p[ti * N_FFT_BINS..(ti + 1) * N_FFT_BINS];
                    let sum: f64 = p_row
                        .iter()
                        .zip(f_row.iter())
                        .map(|(&pv, &fv)| (pv as f64) * (fv as f64))
                        .sum();
                    mel[mi * n_frames + ti] = (sum + LOG_MEL_EPS as f64).ln() as f32;
                }
            }
            XdnaBuf::from_vec(mel)
        })
    }

    fn mel_norm(
        &self,
        mel: &Self::Buf,
        n_mel: usize,
        n_frames: usize,
        effective_n_len: usize,
    ) -> Self::Buf {
        let mut norm = mel.read_clone(n_mel * n_frames);
        let eff = effective_n_len.min(n_frames);
        for mi in 0..n_mel {
            let row = &mut norm[mi * n_frames..(mi + 1) * n_frames];
            if eff > 1 {
                let mean = row[..eff].iter().map(|&v| v as f64).sum::<f64>() / eff as f64;
                let var = row[..eff]
                    .iter()
                    .map(|&v| {
                        let d = v as f64 - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / (eff - 1) as f64;
                let inv_std = 1.0 / (var.sqrt() + NORM_VAR_EPS);
                for v in row[..eff].iter_mut() {
                    *v = ((*v as f64 - mean) * inv_std) as f32;
                }
                row[eff..].fill(0.0);
            } else {
                row.fill(0.0);
            }
        }
        let mut time_major = vec![0.0f32; n_frames * n_mel];
        for mi in 0..n_mel {
            for ti in 0..n_frames {
                time_major[ti * n_mel + mi] = norm[mi * n_frames + ti];
            }
        }
        XdnaBuf::from_vec(time_major)
    }

    fn supports_lfm3a(&self) -> bool {
        true
    }

    fn max_audio_tokens(&self) -> usize {
        usize::MAX
    }

    fn rmsnorm(
        &self,
        src: &Self::Buf,
        weight: &Self::Buf,
        eps: f32,
        rows: usize,
        dim: usize,
    ) -> Self::Buf {
        self.vit.rmsnorm(src, weight, eps, rows, dim)
    }

    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, len: usize) {
        self.vit.silu_mul(gate, up, len);
    }

    fn lfm3a_conv_stem(
        &self,
        mel: &Self::Buf,
        n_frames: usize,
        n_mel: usize,
        layers: &[ConvLayerWeights],
    ) -> (Self::Buf, usize) {
        let mel_guard = mel.0.lock().unwrap_or_else(|e| e.into_inner());
        let (flat_seq, t_out, _plane) =
            conv_stem_lfm3a_features(&mel_guard[..n_frames * n_mel], n_frames, layers, n_mel);
        (XdnaBuf::from_vec(flat_seq), t_out)
    }

    fn lfm3a_shortconv_dw(
        &self,
        bcx: &Self::Buf,
        conv_dw_w: &[f32],
        conv_dw_b: &[f32],
        kernel_size: usize,
        t: usize,
        n_embd: usize,
    ) -> Self::Buf {
        if t == 0 || n_embd == 0 {
            return XdnaBuf::from_vec(Vec::new());
        }
        let bcx_guard = bcx.0.lock().unwrap_or_else(|e| e.into_inner());
        let bcx_slice = &bcx_guard[..t * 3 * n_embd];
        let pad_left = kernel_size - 1;
        let t_padded = t + pad_left;

        let mut padded_bx = vec![0.0f32; n_embd * t_padded];
        let mut c_gate = vec![0.0f32; t * n_embd];
        for ti in 0..t {
            let row = &bcx_slice[ti * 3 * n_embd..(ti + 1) * 3 * n_embd];
            let (b_part, rest) = row.split_at(n_embd);
            let (c_part, x_part) = rest.split_at(n_embd);
            c_gate[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(c_part);
            for ch in 0..n_embd {
                padded_bx[ch * t_padded + pad_left + ti] = b_part[ch] * x_part[ch];
            }
        }

        let mut conv_out = vec![0.0f32; n_embd * t];
        let dw_bias = (!conv_dw_b.is_empty()).then_some(conv_dw_b);
        let t_out = conv1d(
            &padded_bx,
            conv_dw_w,
            dw_bias,
            &mut conv_out,
            n_embd,
            n_embd,
            t_padded,
            kernel_size,
            1,
            0,
            n_embd,
        );
        debug_assert_eq!(t_out, t);

        let mut out = vec![0.0f32; t * n_embd];
        par_rows_n(&mut out, n_embd, 8, |(ti, y_in)| {
            let c_row = &c_gate[ti * n_embd..(ti + 1) * n_embd];
            for ch in 0..n_embd {
                y_in[ch] = conv_out[ch * t + ti] * c_row[ch];
            }
        });
        XdnaBuf::from_vec(out)
    }

    fn lfm3a_swa_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        gate: Option<&Self::Buf>,
        gate_rows: usize,
        q_norm_w: &[f32],
        k_norm_w: &[f32],
        t: usize,
        n_head: usize,
        n_head_kv: usize,
        d_head: usize,
        eps: f32,
        rope_theta: f32,
        sliding_window: usize,
    ) -> Self::Buf {
        let n_embd = n_head * d_head;
        let n_embd_kv = n_head_kv * d_head;
        if t == 0 || n_embd == 0 || n_embd_kv == 0 {
            return XdnaBuf::zeroed(t * n_embd);
        }

        let mut q_proj = q.read_clone(t * n_embd);
        let mut k_proj = k.read_clone(t * n_embd_kv);
        let v_proj = v.read_clone(t * n_embd_kv);

        par_rows_n(&mut q_proj, n_embd, 8, |(ti, q_row)| {
            for h in 0..n_head {
                let q_head = &mut q_row[h * d_head..(h + 1) * d_head];
                rmsnorm(q_head, q_norm_w, eps);
                apply_rope_neox_head(q_head, ti, rope_theta);
            }
        });
        par_rows_n(&mut k_proj, n_embd_kv, 8, |(ti, k_row)| {
            for h_kv in 0..n_head_kv {
                let k_head = &mut k_row[h_kv * d_head..(h_kv + 1) * d_head];
                rmsnorm(k_head, k_norm_w, eps);
                apply_rope_neox_head(k_head, ti, rope_theta);
            }
        });

        let kv_head_stride = t * d_head;
        let mut k_hm = vec![0.0f32; n_head_kv * kv_head_stride];
        let mut v_hm = vec![0.0f32; n_head_kv * kv_head_stride];
        for ti in 0..t {
            for h_kv in 0..n_head_kv {
                let src = ti * n_embd_kv + h_kv * d_head;
                let dst = h_kv * kv_head_stride + ti * d_head;
                k_hm[dst..dst + d_head].copy_from_slice(&k_proj[src..src + d_head]);
                v_hm[dst..dst + d_head].copy_from_slice(&v_proj[src..src + d_head]);
            }
        }

        let scale = 1.0f32 / (d_head as f32).sqrt();
        let n_rep = (n_head / n_head_kv).max(1);
        let out_head_stride = t * d_head;
        let mut out_hm = vec![0.0f32; n_head * out_head_stride];
        let out_hm_ptr = out_hm.as_mut_ptr() as usize;

        maybe_par_for(n_head, |h| {
            let h_kv = h / n_rep;
            let k_head = &k_hm[h_kv * kv_head_stride..(h_kv + 1) * kv_head_stride];
            let v_head = &v_hm[h_kv * kv_head_stride..(h_kv + 1) * kv_head_stride];
            let mut scores = vec![0.0f32; t];
            // SAFETY: distinct heads write disjoint `[h * out_head_stride .. (h+1) * out_head_stride]` slices.
            let out_head = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_hm_ptr as *mut f32).add(h * out_head_stride),
                    out_head_stride,
                )
            };
            for qt in 0..t {
                let k_start = if sliding_window > 0 {
                    qt.saturating_sub(sliding_window - 1)
                } else {
                    0
                };
                let k_end = qt + 1;
                let q_row = &q_proj[qt * n_embd + h * d_head..qt * n_embd + (h + 1) * d_head];
                let score_slice = &mut scores[..k_end - k_start];
                for (idx, kt) in (k_start..k_end).enumerate() {
                    let k_row = &k_head[kt * d_head..(kt + 1) * d_head];
                    score_slice[idx] = dot_f32(q_row, k_row) * scale;
                }
                softmax_inplace(score_slice);
                let out_row = &mut out_head[qt * d_head..(qt + 1) * d_head];
                out_row.fill(0.0);
                for (idx, kt) in (k_start..k_end).enumerate() {
                    axpy_inplace(
                        out_row,
                        &v_head[kt * d_head..(kt + 1) * d_head],
                        score_slice[idx],
                    );
                }
            }
        });

        let mut attn_v = vec![0.0f32; t * n_embd];
        par_rows_n(&mut attn_v, n_embd, 8, |(ti, dst_row)| {
            for h in 0..n_head {
                let src = h * out_head_stride + ti * d_head;
                dst_row[h * d_head..(h + 1) * d_head].copy_from_slice(&out_hm[src..src + d_head]);
            }
        });

        if let Some(gate_buf) = gate {
            let mut gate_proj = gate_buf.read_clone(t * gate_rows);
            if gate_rows == n_head {
                par_rows_n(&mut attn_v, n_embd, 8, |(ti, row)| {
                    let g_row = &gate_proj[ti * n_head..(ti + 1) * n_head];
                    for h in 0..n_head {
                        let sig = 1.0f32 / (1.0f32 + (-g_row[h]).exp());
                        for v in &mut row[h * d_head..(h + 1) * d_head] {
                            *v *= sig;
                        }
                    }
                });
            } else if gate_rows == n_embd {
                sigmoid_inplace(&mut gate_proj);
                par_rows_n(&mut attn_v, n_embd, 8, |(ti, row)| {
                    let g_row = &gate_proj[ti * n_embd..(ti + 1) * n_embd];
                    for (dst, &sig) in row.iter_mut().zip(g_row.iter()) {
                        *dst *= sig;
                    }
                });
            }
        }
        XdnaBuf::from_vec(attn_v)
    }
}

struct XdnaAudioEncoder {
    ops: XdnaAudioOps,
    weights: GpuAudioWeights<XdnaAudioOps>,
    encode_lock: Mutex<()>,
}

impl AudioGpuEncode for XdnaAudioEncoder {
    fn encode_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let _serial = self.encode_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.ops.take_err();
        let out = encode_audio_pcm_gpu(&self.ops, &self.weights, pcm)?;
        if let Some(e) = self.ops.take_err() {
            return Err(e);
        }
        Ok(out)
    }

    fn encode_mel(&self, mel: &[f32], n_frames: usize) -> Result<(Vec<f32>, usize)> {
        let _serial = self.encode_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.ops.take_err();
        let out = encode_audio_mel_gpu(&self.ops, &self.weights, mel, n_frames)?;
        if let Some(e) = self.ops.take_err() {
            return Err(e);
        }
        Ok(out)
    }
}

/// Build an XDNA [`AudioGpuEncode`] for `weights`, returning `None` if unavailable.
pub(crate) fn try_xdna_audio_encoder(
    weights: &Arc<AudioEncoderWeights>,
) -> Option<Arc<dyn AudioGpuEncode>> {
    let runner = match XdnaGemmRunner::open_shared() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("audio encoder: XDNA backend unavailable ({e:#}); falling back");
            return None;
        }
    };
    let ops = XdnaAudioOps::new(runner);
    let _ = ops.take_err();
    let gpu_w = match GpuAudioWeights::build(&ops, weights) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("audio encoder: XDNA backend unavailable for this model: {e:#}");
            eprintln!("audio encoder: XDNA backend unavailable for this model: {e:#}");
            return None;
        }
    };
    if let Some(e) = ops.take_err() {
        tracing::warn!("audio encoder: XDNA weight upload failed, falling back: {e:#}");
        eprintln!("audio encoder: XDNA weight upload failed, falling back: {e:#}");
        return None;
    }
    tracing::info!("audio encoder: using AMD XDNA2 NPU backend");
    Some(Arc::new(XdnaAudioEncoder {
        ops,
        weights: gpu_w,
        encode_lock: Mutex::new(()),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::xdna::sys::fake::{self, Config as FakeConfig};
    use crate::model::audio_encoder::{
        AudioEncoderConfig, AudioMlpAdapterWeights, ConformerLayerWeights, ConvLayerWeights,
        ConvStemWeights, Lfm3aLayerExtras, POS_EMB_DIM, audio_encoder_forward,
        audio_encoder_forward_lfm3a, conformer_self_attention_forward, encode_audio_pcm,
    };
    use crate::model::audio_preprocessor::log_mel_spectrogram;
    use crate::model::vision_encoder::{
        Lfm3AttnBlockWeights, Lfm3BlockOp, Lfm3ConvBlockWeights, Lfm3FfnWeights, Lfm3NormWeights,
        Lfm3VisionBlockWeights, PatchEmbedWeights, ProjectorWeights, VisionEncoderConfig,
        VisionProjectorType, VitBlockWeights,
    };
    use crate::model::vision_encoder_gpu::{VitStackActivation, VitStackBlock, VitStackProjector};
    use crate::tensor::DType;

    fn det_vec(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i * 131 + seed * 17) % 97) as f32 / 97.0 - 0.5;
                x * scale
            })
            .collect()
    }

    fn f32_weight(rows: usize, cols: usize, seed: usize, scale: f32) -> MmapWeight {
        MmapWeight::from_owned_f32(det_vec(rows * cols, seed, scale), rows, cols)
    }

    fn q8_0_weight(rows: usize, cols: usize, seed: usize, scale: f32) -> MmapWeight {
        assert!(cols.is_multiple_of(32));
        let dense = det_vec(rows * cols, seed, scale);
        let blocks_per_row = cols / 32;
        let mut bytes = Vec::with_capacity(rows * blocks_per_row * 34);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let block = &dense[r * cols + b * 32..r * cols + (b + 1) * 32];
                let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                let d_f16 = half::f16::from_f32(d);
                bytes.extend_from_slice(&d_f16.to_le_bytes());
                let d_real = d_f16.to_f32();
                for &v in block {
                    let q = (v / d_real).round().clamp(-127.0, 127.0) as i8;
                    bytes.push(q as u8);
                }
            }
        }
        MmapWeight::from_owned_bytes(bytes, DType::Q8_0, rows, cols)
    }

    fn synth_vit_weights(
        q8: bool,
        n_layer: usize,
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
    ) -> VisionEncoderWeights {
        let patch_size = 8;
        let in_dim = 3 * patch_size * patch_size;
        let proj_dim = 64;
        let scale_factor = 2;
        let pooled_dim = n_embd * scale_factor * scale_factor;
        let cfg = VisionEncoderConfig {
            projector_type: VisionProjectorType::Lfm2,
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv: n_head,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            image_size: 32,
            patch_size,
            n_trained_patches: 16,
            projection_dim: proj_dim,
            scale_factor,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 256,
            image_max_pixels: 4096,
        };
        let w_scale = 0.15 * (128.0f32 / n_embd as f32).sqrt();
        let lin = |r: usize, c: usize, seed: usize| {
            let s = if c > n_embd {
                w_scale * (n_embd as f32 / c as f32).sqrt()
            } else {
                w_scale
            };
            if q8 {
                q8_0_weight(r, c, seed, s)
            } else {
                f32_weight(r, c, seed, s)
            }
        };
        let blocks = (0..n_layer)
            .map(|l| VitBlockWeights {
                ln1_w: det_vec(n_embd, l * 10 + 1, 0.5)
                    .into_iter()
                    .map(|v| 1.0 + v * 0.2)
                    .collect(),
                ln1_b: det_vec(n_embd, l * 10 + 2, 0.05),
                q_w: lin(n_embd, n_embd, l * 10 + 3),
                q_b: det_vec(n_embd, l * 10 + 4, 0.05),
                k_w: lin(n_embd, n_embd, l * 10 + 5),
                k_b: det_vec(n_embd, l * 10 + 6, 0.05),
                v_w: lin(n_embd, n_embd, l * 10 + 7),
                v_b: det_vec(n_embd, l * 10 + 8, 0.05),
                o_w: lin(n_embd, n_embd, l * 10 + 9),
                o_b: det_vec(n_embd, l * 10 + 10, 0.05),
                ln2_w: det_vec(n_embd, l * 10 + 11, 0.5)
                    .into_iter()
                    .map(|v| 1.0 + v * 0.2)
                    .collect(),
                ln2_b: det_vec(n_embd, l * 10 + 12, 0.05),
                ffn_up_w: lin(n_ff, n_embd, l * 10 + 13),
                ffn_up_b: det_vec(n_ff, l * 10 + 14, 0.05),
                ffn_down_w: lin(n_embd, n_ff, l * 10 + 15),
                ffn_down_b: det_vec(n_embd, l * 10 + 16, 0.05),
            })
            .collect();

        VisionEncoderWeights {
            config: cfg,
            patch_embed: PatchEmbedWeights {
                conv_w: det_vec(in_dim * n_embd, 100, 0.1),
                conv_b: det_vec(n_embd, 101, 0.05),
            },
            patch_embed_1: None,
            embedding_norm: None,
            position_embed: det_vec(16 * n_embd, 102, 0.05),
            blocks,
            lfm3_blocks: Vec::new(),
            post_ln_w: det_vec(n_embd, 103, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect(),
            post_ln_b: det_vec(n_embd, 104, 0.05),
            mm_input_norm: None,
            projector: ProjectorWeights {
                mm1_w: lin(proj_dim * 2, pooled_dim, 105),
                mm1_b: det_vec(proj_dim * 2, 106, 0.05),
                mm2_w: lin(proj_dim, proj_dim * 2, 107),
                mm2_b: det_vec(proj_dim, 108, 0.05),
            },
        }
    }

    fn synth_vit_stack_spec(activation: VitStackActivation, with_proj: bool) -> VitStackSpec {
        let (width, heads, ffn) = (64, 4, 128);
        VitStackSpec {
            width,
            heads,
            ffn,
            eps: 1e-6,
            activation,
            blocks: vec![VitStackBlock {
                ln1_w: vec![1.0; width],
                ln1_b: det_vec(width, 1, 0.02),
                q: f32_weight(width, width, 2, 0.1),
                q_b: det_vec(width, 3, 0.02),
                k: f32_weight(width, width, 4, 0.1),
                k_b: det_vec(width, 5, 0.02),
                v: f32_weight(width, width, 6, 0.1),
                v_b: det_vec(width, 7, 0.02),
                o: f32_weight(width, width, 8, 0.1),
                o_b: det_vec(width, 9, 0.02),
                ln2_w: vec![1.0; width],
                ln2_b: det_vec(width, 10, 0.02),
                up: f32_weight(ffn, width, 11, 0.1),
                up_b: det_vec(ffn, 12, 0.02),
                down: f32_weight(width, ffn, 13, 0.1),
                down_b: det_vec(width, 14, 0.02),
            }],
            post: Some((vec![1.0; width], det_vec(width, 15, 0.02))),
            projector: with_proj.then(|| VitStackProjector {
                mm1: f32_weight(96, width, 16, 0.1),
                mm1_b: det_vec(96, 17, 0.02),
                mm2: f32_weight(48, 96, 18, 0.1),
                mm2_b: det_vec(48, 19, 0.02),
            }),
        }
    }

    fn synth_audio_weights(
        n_layer: usize,
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
    ) -> AudioEncoderWeights {
        let n_mel_bins = 64;
        let stem_ch = 32;
        let llm_hidden_size = 48;
        let f_out = n_mel_bins / 8; // 3 stride-2 layers: 64 -> 32 -> 16 -> 8
        let cfg = AudioEncoderConfig {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            eps: 1e-5,
            n_mel_bins,
            llm_hidden_size,
        };
        let stem = ConvStemWeights {
            layers: vec![
                ConvLayerWeights {
                    name: "conv1d.0".into(),
                    weight: det_vec(stem_ch * 3 * 3, 1, 0.05),
                    bias: det_vec(stem_ch, 2, 0.02),
                    shape: vec![3, 3, 1, stem_ch],
                },
                ConvLayerWeights {
                    name: "conv1d.2".into(),
                    weight: det_vec(stem_ch * 3 * 3, 3, 0.05),
                    bias: det_vec(stem_ch, 4, 0.02),
                    shape: vec![3, 3, 1, stem_ch],
                },
                ConvLayerWeights {
                    name: "conv1d.3".into(),
                    weight: det_vec(stem_ch * stem_ch, 5, 0.05),
                    bias: det_vec(stem_ch, 6, 0.02),
                    shape: vec![1, 1, stem_ch, stem_ch],
                },
                ConvLayerWeights {
                    name: "conv1d.5".into(),
                    weight: det_vec(stem_ch * 3 * 3, 7, 0.05),
                    bias: det_vec(stem_ch, 8, 0.02),
                    shape: vec![3, 3, 1, stem_ch],
                },
                ConvLayerWeights {
                    name: "conv1d.6".into(),
                    weight: det_vec(stem_ch * stem_ch, 9, 0.05),
                    bias: det_vec(stem_ch, 10, 0.02),
                    shape: vec![1, 1, stem_ch, stem_ch],
                },
            ],
            pre_encode_out_w: f32_weight(n_embd, stem_ch * f_out, 11, 0.05),
            pre_encode_out_b: det_vec(n_embd, 12, 0.02),
        };
        let dw_k = 7;
        let layers = (0..n_layer)
            .map(|l| ConformerLayerWeights {
                ffn_norm_w: vec![1.0; n_embd],
                ffn_norm_b: det_vec(n_embd, l * 30 + 1, 0.02),
                ffn_up_w: f32_weight(n_ff, n_embd, l * 30 + 2, 0.08),
                ffn_up_b: det_vec(n_ff, l * 30 + 3, 0.02),
                ffn_down_w: f32_weight(n_embd, n_ff, l * 30 + 4, 0.08),
                ffn_down_b: det_vec(n_embd, l * 30 + 5, 0.02),
                ln1_w: vec![1.0; n_embd],
                ln1_b: det_vec(n_embd, l * 30 + 6, 0.02),
                attn_q_w: f32_weight(n_embd, n_embd, l * 30 + 7, 0.08),
                attn_q_b: det_vec(n_embd, l * 30 + 8, 0.02),
                attn_k_w: f32_weight(n_embd, n_embd, l * 30 + 9, 0.08),
                attn_k_b: det_vec(n_embd, l * 30 + 10, 0.02),
                attn_v_w: f32_weight(n_embd, n_embd, l * 30 + 11, 0.08),
                attn_v_b: det_vec(n_embd, l * 30 + 12, 0.02),
                attn_o_w: f32_weight(n_embd, n_embd, l * 30 + 13, 0.08),
                attn_o_b: det_vec(n_embd, l * 30 + 14, 0.02),
                pos_bias_u: det_vec(n_embd, l * 30 + 15, 0.05),
                pos_bias_v: det_vec(n_embd, l * 30 + 16, 0.05),
                linear_pos_w: f32_weight(n_embd, POS_EMB_DIM, l * 30 + 17, 0.08),
                norm_conv_w: vec![1.0; n_embd],
                norm_conv_b: det_vec(n_embd, l * 30 + 18, 0.02),
                conv_pw1_w: f32_weight(2 * n_embd, n_embd, l * 30 + 19, 0.08),
                conv_pw1_b: det_vec(2 * n_embd, l * 30 + 20, 0.02),
                conv_dw_w: det_vec(n_embd * dw_k, l * 30 + 21, 0.08),
                conv_dw_b: det_vec(n_embd, l * 30 + 22, 0.02),
                conv_dw_shape: vec![dw_k, 1, n_embd],
                conv_norm_w: vec![1.0; n_embd],
                conv_norm_b: det_vec(n_embd, l * 30 + 23, 0.02),
                conv_pw2_w: f32_weight(n_embd, n_embd, l * 30 + 24, 0.08),
                conv_pw2_b: det_vec(n_embd, l * 30 + 25, 0.02),
                ffn_norm_1_w: vec![1.0; n_embd],
                ffn_norm_1_b: det_vec(n_embd, l * 30 + 26, 0.02),
                ffn_up_1_w: f32_weight(n_ff, n_embd, l * 30 + 27, 0.08),
                ffn_up_1_b: det_vec(n_ff, l * 30 + 28, 0.02),
                ffn_down_1_w: f32_weight(n_embd, n_ff, l * 30 + 29, 0.08),
                ffn_down_1_b: det_vec(n_embd, l * 30 + 30, 0.02),
                ln2_w: vec![1.0; n_embd],
                ln2_b: det_vec(n_embd, l * 30 + 31, 0.02),
                lfm3a: None,
            })
            .collect();

        AudioEncoderWeights {
            config: cfg,
            conv_stem: stem,
            layers,
            mlp_adapter: AudioMlpAdapterWeights {
                norm_w: vec![1.0; n_embd],
                norm_b: det_vec(n_embd, 500, 0.02),
                up_w: f32_weight(n_ff, n_embd, 501, 0.08),
                up_b: det_vec(n_ff, 502, 0.02),
                down_w: f32_weight(llm_hidden_size, n_ff, 503, 0.08),
                down_b: det_vec(llm_hidden_size, 504, 0.02),
            },
        }
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let x = x as f64;
            let y = y as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn cpu_matmul_nt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += x[r * k + i] as f64 * w[c * k + i] as f64;
                }
                out[r * n + c] = acc as f32;
            }
        }
        out
    }

    #[test]
    fn fake_shim_vision_encoder_lifecycle_zero_bo_leaks() {
        fake::with_shim(FakeConfig::default(), || {
            let w = synth_vit_weights(true, 1, 64, 128, 4);
            let enc = try_xdna_vision_encoder(&w).expect("xdna vision encoder on fake shim");
            assert_eq!(enc.backend_label(), "xdna");
            let pixels = det_vec(3 * 32 * 32, 99, 1.0);
            let out = enc.encode_image(&pixels, 4, 4).expect("encode_image");
            assert_eq!(out.len(), 4 * 64);
            drop(enc);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_vit_stack_gelu_relu_and_projector_zero_leaks() {
        fake::with_shim(FakeConfig::default(), || {
            let runner = XdnaGemmRunner::open().expect("open");
            let spec_gelu = synth_vit_stack_spec(VitStackActivation::GeluTanh, true);
            let stack_gelu = XdnaVitStack::build(runner.clone(), &spec_gelu, 128).expect("gelu");
            let x = det_vec(6 * 64, 3, 0.5);
            assert_eq!(stack_gelu.run(&x, 6).expect("run").len(), 6 * 64);
            assert_eq!(
                stack_gelu
                    .run_projector(&x)
                    .expect("proj")
                    .expect("some")
                    .len(),
                6 * 48
            );

            let spec_relu = synth_vit_stack_spec(VitStackActivation::Relu, false);
            let stack_relu = XdnaVitStack::build(runner, &spec_relu, 128).expect("relu");
            assert_eq!(stack_relu.run(&x, 6).expect("run").len(), 6 * 64);
            assert!(stack_relu.run_projector(&x).expect("no proj").is_none());

            drop(stack_gelu);
            drop(stack_relu);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_audio_encoder_lifecycle_zero_leaks() {
        fake::with_shim(FakeConfig::default(), || {
            let w = Arc::new(synth_audio_weights(1, 64, 128, 4));
            let enc = try_xdna_audio_encoder(&w).expect("xdna audio encoder on fake shim");
            let pcm: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
            let (out_pcm, t_pcm) = enc.encode_pcm(&pcm).expect("encode_pcm");
            assert!(t_pcm > 0);
            assert_eq!(out_pcm.len(), t_pcm * 48);

            let mel = det_vec(25 * 64, 77, 0.5);
            let (out_mel, t_mel) = enc.encode_mel(&mel, 25).expect("encode_mel");
            assert!(t_mel > 0);
            assert_eq!(out_mel.len(), t_mel * 48);

            drop(enc);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn fake_shim_gemm_runlist_fault_surfaces_and_clears() {
        let cfg = FakeConfig {
            fail_runlist_wait: true,
            ..Default::default()
        };
        fake::with_shim(cfg, || {
            let w = synth_vit_weights(false, 1, 64, 128, 4);
            let enc = try_xdna_vision_encoder(&w).expect("weights load before any runlist wait");
            let pixels = det_vec(3 * 32 * 32, 12, 1.0);
            let err = enc
                .encode_image(&pixels, 4, 4)
                .expect_err("runlist wait failure must surface as Err, not zeros");
            assert!(err.to_string().contains("wait gemm_bf16 runlist"));
            drop(enc);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn xdna_audio_host_ops_match_cpu_reference() {
        fake::with_shim(FakeConfig::default(), || {
            let runner = XdnaGemmRunner::open().expect("open");
            let ops = XdnaAudioOps::new(runner);
            let fe = crate::model::audio_encoder_gpu::GpuMelFrontend::build(&ops, 64).expect("fe");
            let pcm: Vec<f32> = (0..6400)
                .map(|i| (i as f32 * 0.07).sin() * 0.4 + (i as f32 * 0.19).cos() * 0.2)
                .collect();
            let (cpu_mel, cpu_frames) = log_mel_spectrogram(&pcm, 64);
            let (buf, gpu_frames) =
                crate::model::audio_encoder_gpu::log_mel_spectrogram_gpu(&ops, &fe, &pcm)
                    .expect("gpu mel")
                    .expect("non-empty");
            assert_eq!(gpu_frames, cpu_frames);
            let got_mel = ops.download(&buf, cpu_frames * 64);
            let cos = cosine_sim(&got_mel, &cpu_mel);
            assert!(cos > 0.9999, "mel cosine {cos} < 0.9999");

            // Check XL attention against CPU `conformer_self_attention_forward`.
            let (t, n_head, head_dim) = (9usize, 4usize, 16usize);
            let dim = n_head * head_dim;
            let q = det_vec(t * dim, 1, 0.3);
            let k = det_vec(t * dim, 2, 0.3);
            let v = det_vec(t * dim, 3, 0.3);
            let p = det_vec((2 * t - 1) * dim, 4, 0.3);
            let bu = det_vec(dim, 5, 0.1);
            let bv = det_vec(dim, 6, 0.1);
            let out_buf = ops.xl_attention(
                &ops.upload(&q),
                &ops.upload(&k),
                &ops.upload(&v),
                &ops.upload(&p),
                &ops.upload(&bu),
                &ops.upload(&bv),
                t,
                n_head,
                head_dim,
            );
            let got_attn = ops.download(&out_buf, t * dim);
            // Reference via identity Q/K/V/O/P linears and ln1=identity: use XdnaAudioOps direct math check via conformer helper isn't needed if identity,
            // wait: verify against scalar formula directly.
            let mut ref_attn = vec![0.0f32; t * dim];
            let scale = 1.0f64 / (head_dim as f64).sqrt();
            for h in 0..n_head {
                for qt in 0..t {
                    let mut sc = vec![0.0f32; t];
                    for kt in 0..t {
                        let pidx = (t - 1) + kt - qt;
                        let mut ac = 0.0f64;
                        let mut bd = 0.0f64;
                        for d in 0..head_dim {
                            let qv = q[qt * dim + h * head_dim + d] as f64;
                            let uv = bu[h * head_dim + d] as f64;
                            let vv = bv[h * head_dim + d] as f64;
                            ac += (qv + uv) * k[kt * dim + h * head_dim + d] as f64;
                            bd += (qv + vv) * p[pidx * dim + h * head_dim + d] as f64;
                        }
                        sc[kt] = ((ac + bd) * scale) as f32;
                    }
                    softmax_inplace(&mut sc);
                    for d in 0..head_dim {
                        let mut sum = 0.0f64;
                        for kt in 0..t {
                            sum += sc[kt] as f64 * v[kt * dim + h * head_dim + d] as f64;
                        }
                        ref_attn[qt * dim + h * head_dim + d] = sum as f32;
                    }
                }
            }
            let max_diff = got_attn
                .iter()
                .zip(ref_attn.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-5, "xl_attention max_diff={max_diff}");
            let _ = (conformer_self_attention_forward,);
        });
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_gemm_bf16_parity_and_throughput() {
        let runner = XdnaGemmRunner::open().expect("open physical XDNA device");
        for &(m, n, k, label) in &[
            (17, 32, 64, "tiny_unaligned"),
            (32, 32, 64, "core_k1"),
            (32, 32, 128, "core_k2"),
            (32, 32, 256, "core_k4"),
            (128, 256, 64, "tile_k1"),
            (128, 256, 128, "tile_k2"),
            (128, 256, 256, "tile_k4"),
            (128, 256, 768, "tile_768"),
            (384, 768, 768, "vit_qkv_768"),
            (729, 3072, 768, "vit_up_729"),
            (729, 768, 3072, "vit_down_729"),
            (200, 1152, 4304, "siglip_4304"),
            (16, 128, 8256, "split_k_8256"),
        ] {
            let x = det_vec(m * k, 13, 0.5);
            let w_f32 = det_vec(n * k, 29, 0.2);
            let w_npu = runner
                .upload_f32_weight(&w_f32, n, k)
                .expect("upload weight");
            let t0 = std::time::Instant::now();
            let got = runner.matmul_f32(&x, &w_npu, m, n, k).expect("matmul_f32");
            let dt = t0.elapsed();
            let want = cpu_matmul_nt(&x, &w_f32, m, n, k);
            let x_bf: Vec<f32> = x
                .iter()
                .map(|&v| half::bf16::from_bits(f32_to_bf16(v)).to_f32())
                .collect();
            let w_bf: Vec<f32> = w_f32
                .iter()
                .map(|&v| half::bf16::from_bits(f32_to_bf16(v)).to_f32())
                .collect();
            let want_bf = cpu_matmul_nt(&x_bf, &w_bf, m, n, k);
            let cos = cosine_sim(&got, &want);
            let cos_bf = cosine_sim(&got, &want_bf);
            let max_diff_bf = got
                .iter()
                .zip(want_bf.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mut worst_rc = (0, 0, 0.0f32, 0.0f32);
            for r in 0..m {
                for c in 0..n {
                    let g = got[r * n + c];
                    let wb = want_bf[r * n + c];
                    let d = (g - wb).abs();
                    if d >= max_diff_bf {
                        worst_rc = (r, c, g, wb);
                    }
                }
            }
            let gflops = (2.0 * m as f64 * n as f64 * k as f64) / (dt.as_secs_f64() * 1e9);
            eprintln!(
                "gemm_bf16 {label} ({m}x{n}x{k}): cos={cos:.6} cos_bf16={cos_bf:.6} max_diff_bf={max_diff_bf:.4e} worst=({},{}: got={:.4} want_bf={:.4}) time={:.2}ms ({gflops:.1} GFLOP/s)",
                worst_rc.0,
                worst_rc.1,
                worst_rc.2,
                worst_rc.3,
                dt.as_secs_f64() * 1e3
            );
            assert!(cos > 0.999, "{label} cos={cos}");
        }
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_vision_encoder_parity_and_throughput() {
        for (q8, tag) in [(false, "F32"), (true, "Q8_0")] {
            let w = synth_vit_weights(q8, 2, 128, 256, 4);
            let enc = try_xdna_vision_encoder(&w).expect("XDNA vision encoder");
            let pixels = det_vec(3 * 32 * 32, 42, 0.8);
            let want = w.encode_image(&pixels, 4, 4).expect("CPU encode_image");
            let got = enc.encode_image(&pixels, 4, 4).expect("NPU encode_image");
            let cos = cosine_sim(&got, &want);
            eprintln!("ViT parity ({tag}): len={} cos={cos:.6}", got.len());
            assert!(cos > 0.999, "ViT {tag} cos={cos}");
        }

        // Realistic SigLIP-style throughput check: 28x28=784 patches, width=768, ffn=3072.
        let bench_w = synth_vit_weights(false, 4, 768, 3072, 12);
        let enc = try_xdna_vision_encoder(&bench_w).expect("XDNA vision encoder bench");
        let pixels = det_vec(3 * (28 * 8) * (28 * 8), 9, 0.5);
        let _warm = enc.encode_image(&pixels, 28, 28).expect("warmup");
        let t0 = std::time::Instant::now();
        let out = enc.encode_image(&pixels, 28, 28).expect("timed encode");
        let dt_npu = t0.elapsed();
        let t1 = std::time::Instant::now();
        let out_cpu = bench_w.encode_image(&pixels, 28, 28).expect("CPU encode");
        let dt_cpu = t1.elapsed();
        let cos = cosine_sim(&out, &out_cpu);
        eprintln!(
            "ViT bench (4L, 784 patches, d=768, ff=3072): npu={:.1}ms cpu={:.1}ms cos={cos:.6}",
            dt_npu.as_secs_f64() * 1e3,
            dt_cpu.as_secs_f64() * 1e3,
        );
        assert!(cos > 0.999);
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_vit_stack_gelu_relu_and_projector_parity() {
        for act in [VitStackActivation::GeluTanh, VitStackActivation::Relu] {
            let spec = synth_vit_stack_spec(act, act == VitStackActivation::GeluTanh);
            let stack = try_xdna_vit_stack(&spec, 256).expect("XDNA vit stack");
            let tokens = 19;
            let x = det_vec(tokens * spec.width, 7, 0.5);
            let got = stack.run(&x, tokens).expect("NPU stack run");

            // Compute exact host reference for the single block + optional post LN.
            let b = &spec.blocks[0];
            let mut cur = x.clone();
            let mut n1 = cur.clone();
            for row in n1.chunks_exact_mut(spec.width) {
                layer_norm_inplace(row, &b.ln1_w, &b.ln1_b, spec.eps);
            }
            let lin_bias = |inp: &[f32], mw: &MmapWeight, bias: &[f32]| -> Vec<f32> {
                let mut o = vec![0.0f32; tokens * mw.rows];
                mw.batched_matmul(inp, &mut o, tokens);
                for row in o.chunks_exact_mut(mw.rows) {
                    add_inplace(row, bias);
                }
                o
            };
            let q = lin_bias(&n1, &b.q, &b.q_b);
            let k = lin_bias(&n1, &b.k, &b.k_b);
            let v = lin_bias(&n1, &b.v, &b.v_b);
            let head_dim = spec.width / spec.heads;
            let runner = XdnaGemmRunner::open().expect("open");
            let ops = XdnaVitOps::new(runner);
            let attn = ops.download(
                &ops.attention(
                    &ops.upload(&q),
                    &ops.upload(&k),
                    &ops.upload(&v),
                    tokens,
                    spec.heads,
                    head_dim,
                ),
                tokens * spec.width,
            );
            let proj = lin_bias(&attn, &b.o, &b.o_b);
            add_inplace(&mut cur, &proj);
            let mut n2 = cur.clone();
            for row in n2.chunks_exact_mut(spec.width) {
                layer_norm_inplace(row, &b.ln2_w, &b.ln2_b, spec.eps);
            }
            let mut mid = lin_bias(&n2, &b.up, &b.up_b);
            match act {
                VitStackActivation::GeluTanh => gelu_inplace(&mut mid),
                VitStackActivation::Relu => relu_inplace(&mut mid),
            }
            let down = lin_bias(&mid, &b.down, &b.down_b);
            add_inplace(&mut cur, &down);
            if let Some((pw, pb)) = &spec.post {
                for row in cur.chunks_exact_mut(spec.width) {
                    layer_norm_inplace(row, pw, pb, spec.eps);
                }
            }
            let cos = cosine_sim(&got, &cur);
            eprintln!("VitStack ({act:?}): cos={cos:.6}");
            assert!(cos > 0.999, "VitStack {act:?} cos={cos}");

            if let Some(p) = &spec.projector {
                let got_p = stack.run_projector(&got).expect("proj").expect("some");
                let mut m1 = lin_bias(&got, &p.mm1, &p.mm1_b);
                gelu_erf_inplace(&mut m1);
                let want_p = lin_bias(&m1, &p.mm2, &p.mm2_b);
                let pcos = cosine_sim(&got_p, &want_p);
                eprintln!("VitStack projector (exact GELU): cos={pcos:.6}");
                assert!(pcos > 0.999, "projector cos={pcos}");
            }
        }
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_audio_encoder_parity_and_throughput() {
        let w = Arc::new(synth_audio_weights(2, 128, 256, 4));
        let enc = try_xdna_audio_encoder(&w).expect("XDNA audio encoder");

        // 1. Direct mel entry point parity.
        let mel = det_vec(81 * 64, 19, 0.5);
        let (want_mel, want_tm) = audio_encoder_forward(&mel, 81, &w);
        let (got_mel, got_tm) = enc.encode_mel(&mel, 81).expect("encode_mel");
        assert_eq!(got_tm, want_tm);
        let cos_mel = cosine_sim(&got_mel, &want_mel);
        eprintln!("Audio encode_mel parity: t={got_tm} cos={cos_mel:.6}");
        assert!(cos_mel > 0.999, "audio mel cos={cos_mel}");

        // 2. End-to-end PCM entry point parity.
        let pcm: Vec<f32> = (0..24_000)
            .map(|i| (i as f32 * 0.04).sin() * 0.4 + (i as f32 * 0.13).cos() * 0.2)
            .collect();
        let (want_pcm, want_tp) = encode_audio_pcm(&pcm, &w);
        let (got_pcm, got_tp) = enc.encode_pcm(&pcm).expect("encode_pcm");
        assert_eq!(got_tp, want_tp);
        let cos_pcm = cosine_sim(&got_pcm, &want_pcm);
        eprintln!("Audio encode_pcm parity: t={got_tp} cos={cos_pcm:.6}");
        assert!(cos_pcm > 0.999, "audio pcm cos={cos_pcm}");

        // 3. Realistic Conformer throughput check (4 layers, n_embd=512, n_ff=2048, ~2.5s PCM -> ~250 mel frames).
        let bench_w = Arc::new(synth_audio_weights(4, 512, 2048, 8));
        let bench_enc = try_xdna_audio_encoder(&bench_w).expect("bench audio encoder");
        let long_pcm: Vec<f32> = (0..40_000).map(|i| (i as f32 * 0.03).sin() * 0.3).collect();
        let _ = bench_enc.encode_pcm(&long_pcm).expect("warmup");
        let t0 = std::time::Instant::now();
        let (o_npu, tn) = bench_enc.encode_pcm(&long_pcm).expect("npu");
        let dt_npu = t0.elapsed();
        let t1 = std::time::Instant::now();
        let (o_cpu, tc) = encode_audio_pcm(&long_pcm, &bench_w);
        let dt_cpu = t1.elapsed();
        assert_eq!(tn, tc);
        let cos = cosine_sim(&o_npu, &o_cpu);
        eprintln!(
            "Audio bench (4L, d=512, ff=2048, 2.5s PCM -> t={tn}): npu={:.1}ms cpu={:.1}ms cos={cos:.6}",
            dt_npu.as_secs_f64() * 1e3,
            dt_cpu.as_secs_f64() * 1e3,
        );
        assert!(cos > 0.999);
    }

    fn synth_lfm3_vision_weights(
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
        n_head_kv: usize,
        n_conv: usize,
        n_attn: usize,
    ) -> VisionEncoderWeights {
        let patch_size = 16;
        let in_dim = 3 * patch_size * patch_size;
        let proj_dim = 64;
        let scale_factor = 2;
        let pooled_dim = n_embd * scale_factor * scale_factor;
        let head_dim = n_embd / n_head;
        let kv_dim = n_head_kv * head_dim;
        let cfg = VisionEncoderConfig {
            projector_type: VisionProjectorType::Lfm3Vl,
            n_layer: n_conv + n_attn,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            rope_theta: 1_000_000.0,
            eps: 1e-5,
            image_size: 128,
            patch_size,
            n_trained_patches: 64,
            projection_dim: proj_dim,
            scale_factor,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: 64,
            image_max_pixels: 1024 * 1024,
        };
        let norm = |len: usize, seed: usize| Lfm3NormWeights {
            weight: det_vec(len, seed, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect(),
            bias: None,
        };
        let mut lfm3_blocks = Vec::new();
        for l in 0..(n_conv + n_attn) {
            let s = (l + 1) * 31;
            let op = if l < n_conv {
                let mult = if l % 2 == 0 { 3 } else { 2 };
                Lfm3BlockOp::Conv(Lfm3ConvBlockWeights {
                    in_proj_w: q8_0_weight(mult * n_embd, n_embd, s + 1, 0.25),
                    in_proj_b: None,
                    depthwise_w: det_vec(n_embd * 3 * 3, s + 2, 0.3),
                    depthwise_b: None,
                    kernel_size: 3,
                    out_proj_w: q8_0_weight(n_embd, n_embd, s + 3, 0.25),
                    out_proj_b: None,
                })
            } else {
                Lfm3BlockOp::Attn(Lfm3AttnBlockWeights {
                    q_w: q8_0_weight(n_embd, n_embd, s + 1, 0.25),
                    q_b: None,
                    k_w: q8_0_weight(kv_dim, n_embd, s + 2, 0.25),
                    k_b: None,
                    v_w: q8_0_weight(kv_dim, n_embd, s + 3, 0.25),
                    v_b: None,
                    q_norm: norm(head_dim, s + 4),
                    k_norm: norm(head_dim, s + 5),
                    o_w: q8_0_weight(n_embd, n_embd, s + 6, 0.25),
                    o_b: None,
                    n_head,
                    n_head_kv,
                    head_dim,
                })
            };
            let swiglu = l % 2 == 0;
            lfm3_blocks.push(Lfm3VisionBlockWeights {
                ln1: norm(n_embd, s + 7),
                op,
                ln2: norm(n_embd, s + 8),
                ffn: Lfm3FfnWeights {
                    gate_w: swiglu.then(|| q8_0_weight(n_ff, n_embd, s + 9, 0.25)),
                    gate_b: None,
                    up_w: q8_0_weight(n_ff, n_embd, s + 10, 0.25),
                    up_b: None,
                    down_w: q8_0_weight(n_embd, n_ff, s + 11, 0.25),
                    down_b: None,
                },
            });
        }

        VisionEncoderWeights {
            config: cfg,
            patch_embed: PatchEmbedWeights {
                conv_w: det_vec(in_dim * n_embd, 100, 0.15),
                conv_b: vec![0.0; n_embd],
            },
            patch_embed_1: Some(PatchEmbedWeights {
                conv_w: det_vec(in_dim * n_embd, 101, 0.15),
                conv_b: vec![0.0; n_embd],
            }),
            embedding_norm: Some(norm(n_embd, 102)),
            position_embed: Vec::new(),
            blocks: Vec::new(),
            lfm3_blocks,
            post_ln_w: Vec::new(),
            post_ln_b: Vec::new(),
            mm_input_norm: Some(Lfm3NormWeights {
                weight: det_vec(n_embd, 103, 0.2)
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect(),
                bias: Some(det_vec(n_embd, 104, 0.05)),
            }),
            projector: ProjectorWeights {
                mm1_w: f32_weight(pooled_dim, pooled_dim, 105, 0.2),
                mm1_b: det_vec(pooled_dim, 106, 0.05),
                mm2_w: f32_weight(proj_dim, pooled_dim, 107, 0.2),
                mm2_b: det_vec(proj_dim, 108, 0.05),
            },
        }
    }

    fn synth_lfm3a_audio_weights(
        n_embd: usize,
        n_ff: usize,
        n_head: usize,
        n_head_kv: usize,
        n_conv: usize,
        n_attn: usize,
        stem_ch: usize,
    ) -> AudioEncoderWeights {
        let n_mel_bins = 128;
        let llm_hidden_size = 64;
        let d_head = n_embd / n_head;
        let n_embd_kv = n_head_kv * d_head;
        let with_post_norms = n_embd <= 128;
        // Scale linear weights by 1/sqrt(n_embd) (Xavier/Kaiming scale, avoiding artificial
        // activation explosion across 17 residual blocks); mix odd-modulus sequences so rows
        // are not collinear at n_embd=768 (`97`-periodic resonance in det_vec).
        let w_scale = 2.0 / (n_embd as f32).sqrt();
        let mat_vec = |rows: usize, cols: usize, seed: usize, scale: f32| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| {
                    let v1 =
                        ((i.wrapping_mul(131).wrapping_add(seed * 977)) % 997) as f32 / 997.0 - 0.5;
                    let v2 =
                        ((i.wrapping_mul(257).wrapping_add(seed * 313)) % 991) as f32 / 991.0 - 0.5;
                    (v1 + 0.5 * v2) * scale
                })
                .collect()
        };
        let q8_mat = |rows: usize, cols: usize, seed: usize| -> MmapWeight {
            assert!(cols.is_multiple_of(32));
            let dense = mat_vec(rows, cols, seed, w_scale);
            let blocks_per_row = cols / 32;
            let mut bytes = Vec::with_capacity(rows * blocks_per_row * 34);
            for r in 0..rows {
                for b in 0..blocks_per_row {
                    let block = &dense[r * cols + b * 32..r * cols + (b + 1) * 32];
                    let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                    let d_f16 = half::f16::from_f32(d);
                    bytes.extend_from_slice(&d_f16.to_le_bytes());
                    let inv = 1.0 / d_f16.to_f32();
                    for &v in block {
                        bytes.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
                    }
                }
            }
            MmapWeight::from_owned_bytes(bytes, DType::Q8_0, rows, cols)
        };
        // 5-layer LFM3A depthwise-separable stem downsampling time/freq by 8x (`128 -> 16`).
        let stem_specs: [(&str, [usize; 4]); 5] = [
            ("a.conv1d.0", [3, 3, 1, stem_ch]),
            ("a.conv1d.2", [3, 3, 1, stem_ch]),
            ("a.conv1d.3", [1, 1, stem_ch, stem_ch]),
            ("a.conv1d.5", [3, 3, 1, stem_ch]),
            ("a.conv1d.6", [1, 1, stem_ch, stem_ch]),
        ];
        let stem_layers = stem_specs
            .iter()
            .enumerate()
            .map(|(i, &(name, shape))| {
                let n: usize = shape.iter().product();
                ConvLayerWeights {
                    name: name.to_string(),
                    weight: det_vec(n, i + 10, 0.15),
                    bias: det_vec(shape[3], i + 20, 0.03),
                    shape: shape.to_vec(),
                }
            })
            .collect();
        let pre_in = stem_ch * 16;
        let dummy = || MmapWeight::from_owned_f32(Vec::new(), 0, 0);
        let mut layers = Vec::new();
        for l in 0..(n_conv + n_attn) {
            let s = (l + 1) * 43;
            let is_conv = l < n_conv;
            let post_ln_w = if l == 0 {
                det_vec(n_embd, 800, 0.2)
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect()
            } else {
                Vec::new()
            };
            let opt_pn = |seed: usize| -> Vec<f32> {
                if with_post_norms {
                    det_vec(n_embd, seed, 0.2)
                        .into_iter()
                        .map(|v| 1.0 + v)
                        .collect()
                } else {
                    Vec::new()
                }
            };
            if is_conv {
                layers.push(ConformerLayerWeights {
                    ffn_norm_w: Vec::new(),
                    ffn_norm_b: Vec::new(),
                    ffn_up_w: q8_mat(n_ff, n_embd, s + 1),
                    ffn_up_b: Vec::new(),
                    ffn_down_w: q8_mat(n_embd, n_ff, s + 2),
                    ffn_down_b: Vec::new(),
                    ln1_w: det_vec(n_embd, s + 3, 0.2)
                        .into_iter()
                        .map(|v| 1.0 + v)
                        .collect(),
                    ln1_b: Vec::new(),
                    attn_q_w: dummy(),
                    attn_q_b: Vec::new(),
                    attn_k_w: dummy(),
                    attn_k_b: Vec::new(),
                    attn_v_w: dummy(),
                    attn_v_b: Vec::new(),
                    attn_o_w: dummy(),
                    attn_o_b: Vec::new(),
                    pos_bias_u: Vec::new(),
                    pos_bias_v: Vec::new(),
                    linear_pos_w: dummy(),
                    norm_conv_w: Vec::new(),
                    norm_conv_b: Vec::new(),
                    conv_pw1_w: q8_mat(3 * n_embd, n_embd, s + 4),
                    conv_pw1_b: Vec::new(),
                    conv_dw_w: det_vec(3 * n_embd, s + 5, 0.2),
                    conv_dw_b: Vec::new(),
                    conv_dw_shape: vec![3, n_embd],
                    conv_norm_w: Vec::new(),
                    conv_norm_b: Vec::new(),
                    conv_pw2_w: q8_mat(n_embd, n_embd, s + 6),
                    conv_pw2_b: Vec::new(),
                    ffn_norm_1_w: Vec::new(),
                    ffn_norm_1_b: Vec::new(),
                    ffn_up_1_w: q8_mat(n_ff, n_embd, s + 7),
                    ffn_up_1_b: Vec::new(),
                    ffn_down_1_w: dummy(),
                    ffn_down_1_b: Vec::new(),
                    ln2_w: det_vec(n_embd, s + 8, 0.2)
                        .into_iter()
                        .map(|v| 1.0 + v)
                        .collect(),
                    ln2_b: Vec::new(),
                    lfm3a: Some(Lfm3aLayerExtras {
                        q_norm_w: Vec::new(),
                        k_norm_w: Vec::new(),
                        attn_gate_w: None,
                        attn_post_norm_w: opt_pn(s + 9),
                        ffn_post_norm_w: opt_pn(s + 10),
                        post_ln_w,
                        rope_theta: 10_000.0,
                        n_head_kv,
                        sliding_window: 32,
                    }),
                });
            } else {
                let gate_rows = if l % 2 == 0 { n_head } else { n_embd };
                layers.push(ConformerLayerWeights {
                    ffn_norm_w: Vec::new(),
                    ffn_norm_b: Vec::new(),
                    ffn_up_w: q8_mat(n_ff, n_embd, s + 1),
                    ffn_up_b: Vec::new(),
                    ffn_down_w: q8_mat(n_embd, n_ff, s + 2),
                    ffn_down_b: Vec::new(),
                    ln1_w: det_vec(n_embd, s + 3, 0.2)
                        .into_iter()
                        .map(|v| 1.0 + v)
                        .collect(),
                    ln1_b: Vec::new(),
                    attn_q_w: q8_mat(n_embd, n_embd, s + 4),
                    attn_q_b: Vec::new(),
                    attn_k_w: q8_mat(n_embd_kv, n_embd, s + 5),
                    attn_k_b: Vec::new(),
                    attn_v_w: q8_mat(n_embd_kv, n_embd, s + 6),
                    attn_v_b: Vec::new(),
                    attn_o_w: q8_mat(n_embd, n_embd, s + 7),
                    attn_o_b: Vec::new(),
                    pos_bias_u: Vec::new(),
                    pos_bias_v: Vec::new(),
                    linear_pos_w: dummy(),
                    norm_conv_w: Vec::new(),
                    norm_conv_b: Vec::new(),
                    conv_pw1_w: dummy(),
                    conv_pw1_b: Vec::new(),
                    conv_dw_w: Vec::new(),
                    conv_dw_b: Vec::new(),
                    conv_dw_shape: Vec::new(),
                    conv_norm_w: Vec::new(),
                    conv_norm_b: Vec::new(),
                    conv_pw2_w: dummy(),
                    conv_pw2_b: Vec::new(),
                    ffn_norm_1_w: Vec::new(),
                    ffn_norm_1_b: Vec::new(),
                    ffn_up_1_w: q8_mat(n_ff, n_embd, s + 8),
                    ffn_up_1_b: Vec::new(),
                    ffn_down_1_w: dummy(),
                    ffn_down_1_b: Vec::new(),
                    ln2_w: det_vec(n_embd, s + 9, 0.2)
                        .into_iter()
                        .map(|v| 1.0 + v)
                        .collect(),
                    ln2_b: Vec::new(),
                    lfm3a: Some(Lfm3aLayerExtras {
                        q_norm_w: det_vec(d_head, s + 10, 0.2)
                            .into_iter()
                            .map(|v| 1.0 + v)
                            .collect(),
                        k_norm_w: det_vec(d_head, s + 11, 0.2)
                            .into_iter()
                            .map(|v| 1.0 + v)
                            .collect(),
                        attn_gate_w: Some(q8_mat(gate_rows, n_embd, s + 12)),
                        attn_post_norm_w: opt_pn(s + 13),
                        ffn_post_norm_w: Vec::new(),
                        post_ln_w,
                        rope_theta: 10_000.0,
                        n_head_kv,
                        sliding_window: 32,
                    }),
                });
            }
        }

        AudioEncoderWeights {
            config: AudioEncoderConfig {
                n_layer: n_conv + n_attn,
                n_embd,
                n_ff,
                n_head,
                eps: 1e-5,
                n_mel_bins,
                llm_hidden_size,
            },
            conv_stem: ConvStemWeights {
                layers: stem_layers,
                pre_encode_out_w: f32_weight(n_embd, pre_in, 900, 0.1),
                pre_encode_out_b: det_vec(n_embd, 901, 0.05),
            },
            layers,
            mlp_adapter: AudioMlpAdapterWeights {
                norm_w: det_vec(n_embd, 910, 0.2)
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect(),
                norm_b: det_vec(n_embd, 911, 0.05),
                up_w: f32_weight(llm_hidden_size * 2, n_embd, 912, 0.2),
                up_b: det_vec(llm_hidden_size * 2, 913, 0.05),
                down_w: f32_weight(llm_hidden_size, llm_hidden_size * 2, 914, 0.2),
                down_b: det_vec(llm_hidden_size, 915, 0.05),
            },
        }
    }

    #[test]
    fn fake_shim_lfm3_vision_and_audio_share_gemm_runner_zero_leaks() {
        fake::with_shim(FakeConfig::default(), || {
            let vw = synth_lfm3_vision_weights(64, 128, 4, 2, 2, 1);
            let aw = Arc::new(synth_lfm3a_audio_weights(64, 128, 4, 2, 1, 2, 16));
            let venc = crate::model::vision_encoder_gpu::build_gpu_vision_encoder(
                &vw,
                crate::engine::BackendPreference::Npu,
            )
            .expect("LFM3 vision routed to XDNA");
            let aenc = crate::model::audio_encoder_gpu::build_gpu_audio_encoder(
                &aw,
                crate::engine::BackendPreference::Npu,
            )
            .expect("LFM3A audio routed to XDNA");

            assert_eq!(venc.backend_label(), "xdna");
            assert_eq!(venc.max_tokens(), usize::MAX);

            let pixels = det_vec(3 * 64 * 64, 12, 0.5);
            let vout = venc.encode_image(&pixels, 4, 4).expect("lfm3 encode_image");
            assert_eq!(vout.len(), 4 * 64);

            let pcm: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.03).sin() * 0.3).collect();
            let (aout, t_out) = aenc.encode_pcm(&pcm).expect("lfm3a encode_pcm");
            assert!(t_out > 0);
            assert_eq!(aout.len(), t_out * 64);

            drop(venc);
            drop(aenc);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_lfm3_vision_encoder_parity_and_throughput() {
        // 1. Parity across Conv (3d & 2d), 2D M-RoPE GQA Attn (n_head=8, kv=4), SwiGLU + GELU,
        //    patch_embed_1, embedding_norm, PreMerge mm.input_norm.
        let vw = synth_lfm3_vision_weights(128, 256, 8, 4, 2, 2);
        let enc = try_xdna_vision_encoder(&vw).expect("XDNA LFM3 vision encoder");
        let pixels = det_vec(3 * (8 * 16) * (8 * 16), 44, 0.5);
        let want = vw.encode_image(&pixels, 8, 8).expect("CPU lfm3 encode");
        let got = enc.encode_image(&pixels, 8, 8).expect("NPU lfm3 encode");
        let cos = cosine_sim(&got, &want);
        eprintln!(
            "LFM3 ViT parity (64 patches, d=128): len={} cos={cos:.6}",
            got.len()
        );
        assert!(cos > 0.999, "LFM3 ViT cos={cos}");

        // 2. Realistic LFM3-VL-2B / LFM3-Omni-2B geometry: 14 layers (8 conv + 6 attn),
        //    d=1024, ff=2560, 16Q/8KV, 512x512 (32x32 = 1024 patches).
        let bench_w = synth_lfm3_vision_weights(1024, 2560, 16, 8, 8, 6);
        let bench_enc = try_xdna_vision_encoder(&bench_w).expect("XDNA LFM3-VL-2B encoder");
        let bench_px = det_vec(3 * 512 * 512, 7, 0.4);
        let _ = bench_enc.encode_image(&bench_px, 32, 32).expect("warmup");
        let t0 = std::time::Instant::now();
        let o_npu = bench_enc.encode_image(&bench_px, 32, 32).expect("npu");
        let dt_npu = t0.elapsed();
        let t1 = std::time::Instant::now();
        let o_cpu = bench_w.encode_image(&bench_px, 32, 32).expect("cpu");
        let dt_cpu = t1.elapsed();
        let cos_bench = cosine_sim(&o_npu, &o_cpu);
        eprintln!(
            "LFM3-VL-2B ViT bench (14L, 1024 patches, d=1024, ff=2560): npu={:.1}ms cpu={:.1}ms cos={cos_bench:.6}",
            dt_npu.as_secs_f64() * 1e3,
            dt_cpu.as_secs_f64() * 1e3,
        );
        assert!(cos_bench > 0.999, "LFM3-VL-2B cos={cos_bench}");
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_lfm3a_audio_encoder_parity_and_throughput() {
        // Quantize every weight to bf16 on a cloned copy so we can isolate pure bf16 numeric
        // sensitivity across deep random layers vs any actual kernel/op discrepancy.
        let to_bf16_weight = |w: &MmapWeight| -> MmapWeight {
            if w.rows == 0 || w.cols == 0 {
                return MmapWeight::from_owned_f32(Vec::new(), 0, 0);
            }
            let mut dense = vec![0.0f32; w.rows * w.cols];
            for r in 0..w.rows {
                w.dequantize_row(r, &mut dense[r * w.cols..(r + 1) * w.cols]);
            }
            for v in &mut dense {
                *v = half::bf16::from_bits(f32_to_bf16(*v)).to_f32();
            }
            MmapWeight::from_owned_f32(dense, w.rows, w.cols)
        };
        let to_bf16_audio = |w: &AudioEncoderWeights| -> AudioEncoderWeights {
            let layers = w
                .layers
                .iter()
                .map(|l| ConformerLayerWeights {
                    ffn_norm_w: l.ffn_norm_w.clone(),
                    ffn_norm_b: l.ffn_norm_b.clone(),
                    ffn_up_w: to_bf16_weight(&l.ffn_up_w),
                    ffn_up_b: l.ffn_up_b.clone(),
                    ffn_down_w: to_bf16_weight(&l.ffn_down_w),
                    ffn_down_b: l.ffn_down_b.clone(),
                    ln1_w: l.ln1_w.clone(),
                    ln1_b: l.ln1_b.clone(),
                    attn_q_w: to_bf16_weight(&l.attn_q_w),
                    attn_q_b: l.attn_q_b.clone(),
                    attn_k_w: to_bf16_weight(&l.attn_k_w),
                    attn_k_b: l.attn_k_b.clone(),
                    attn_v_w: to_bf16_weight(&l.attn_v_w),
                    attn_v_b: l.attn_v_b.clone(),
                    attn_o_w: to_bf16_weight(&l.attn_o_w),
                    attn_o_b: l.attn_o_b.clone(),
                    pos_bias_u: l.pos_bias_u.clone(),
                    pos_bias_v: l.pos_bias_v.clone(),
                    linear_pos_w: to_bf16_weight(&l.linear_pos_w),
                    norm_conv_w: l.norm_conv_w.clone(),
                    norm_conv_b: l.norm_conv_b.clone(),
                    conv_pw1_w: to_bf16_weight(&l.conv_pw1_w),
                    conv_pw1_b: l.conv_pw1_b.clone(),
                    conv_dw_w: l.conv_dw_w.clone(),
                    conv_dw_b: l.conv_dw_b.clone(),
                    conv_dw_shape: l.conv_dw_shape.clone(),
                    conv_norm_w: l.conv_norm_w.clone(),
                    conv_norm_b: l.conv_norm_b.clone(),
                    conv_pw2_w: to_bf16_weight(&l.conv_pw2_w),
                    conv_pw2_b: l.conv_pw2_b.clone(),
                    ffn_norm_1_w: l.ffn_norm_1_w.clone(),
                    ffn_norm_1_b: l.ffn_norm_1_b.clone(),
                    ffn_up_1_w: to_bf16_weight(&l.ffn_up_1_w),
                    ffn_up_1_b: l.ffn_up_1_b.clone(),
                    ffn_down_1_w: to_bf16_weight(&l.ffn_down_1_w),
                    ffn_down_1_b: l.ffn_down_1_b.clone(),
                    ln2_w: l.ln2_w.clone(),
                    ln2_b: l.ln2_b.clone(),
                    lfm3a: l.lfm3a.as_ref().map(|e| Lfm3aLayerExtras {
                        q_norm_w: e.q_norm_w.clone(),
                        k_norm_w: e.k_norm_w.clone(),
                        attn_gate_w: e.attn_gate_w.as_ref().map(&to_bf16_weight),
                        attn_post_norm_w: e.attn_post_norm_w.clone(),
                        ffn_post_norm_w: e.ffn_post_norm_w.clone(),
                        post_ln_w: e.post_ln_w.clone(),
                        rope_theta: e.rope_theta,
                        n_head_kv: e.n_head_kv,
                        sliding_window: e.sliding_window,
                    }),
                })
                .collect();
            AudioEncoderWeights {
                config: w.config.clone(),
                conv_stem: ConvStemWeights {
                    layers: w.conv_stem.layers.clone(),
                    pre_encode_out_w: to_bf16_weight(&w.conv_stem.pre_encode_out_w),
                    pre_encode_out_b: w.conv_stem.pre_encode_out_b.clone(),
                },
                layers,
                mlp_adapter: AudioMlpAdapterWeights {
                    norm_w: w.mlp_adapter.norm_w.clone(),
                    norm_b: w.mlp_adapter.norm_b.clone(),
                    up_w: to_bf16_weight(&w.mlp_adapter.up_w),
                    up_b: w.mlp_adapter.up_b.clone(),
                    down_w: to_bf16_weight(&w.mlp_adapter.down_w),
                    down_b: w.mlp_adapter.down_b.clone(),
                },
            }
        };

        // 1. Parity across 5-layer SiLU stem, ShortConv, SWA GQA + NeoX RoPE + attn_gate, SwiGLU, mlp_adapter.
        let aw = Arc::new(synth_lfm3a_audio_weights(128, 256, 4, 2, 2, 2, 32));
        let aw_bf16 = Arc::new(to_bf16_audio(&aw));
        let enc = try_xdna_audio_encoder(&aw).expect("XDNA LFM3A encoder");

        let mel = det_vec(160 * 128, 31, 0.5);
        let (want_mel, want_tm) = audio_encoder_forward_lfm3a(&mel, 160, &aw);
        let (want_mel_bf, _) = audio_encoder_forward_lfm3a(&mel, 160, &aw_bf16);
        let (got_mel, got_tm) = enc.encode_mel(&mel, 160).expect("lfm3a encode_mel");
        assert_eq!(got_tm, want_tm);
        let cos_mel = cosine_sim(&got_mel, &want_mel);
        let cos_mel_bf = cosine_sim(&got_mel, &want_mel_bf);
        eprintln!(
            "LFM3A encode_mel parity: t={got_tm} cos={cos_mel:.6} cos_vs_bf16_ref={cos_mel_bf:.6}"
        );
        assert!(cos_mel_bf > 0.999, "LFM3A mel bf16_ref cos={cos_mel_bf}");
        assert!(cos_mel > 0.998, "LFM3A mel cos={cos_mel}");

        let pcm: Vec<f32> = (0..24_000)
            .map(|i| (i as f32 * 0.04).sin() * 0.4 + (i as f32 * 0.13).cos() * 0.2)
            .collect();
        let (want_pcm, want_tp) = encode_audio_pcm(&pcm, &aw);
        let (want_pcm_bf, _) = encode_audio_pcm(&pcm, &aw_bf16);
        let (got_pcm, got_tp) = enc.encode_pcm(&pcm).expect("lfm3a encode_pcm");
        assert_eq!(got_tp, want_tp);
        let cos_pcm = cosine_sim(&got_pcm, &want_pcm);
        let cos_pcm_bf = cosine_sim(&got_pcm, &want_pcm_bf);
        eprintln!(
            "LFM3A encode_pcm parity: t={got_tp} cos={cos_pcm:.6} cos_vs_bf16_ref={cos_pcm_bf:.6}"
        );
        assert!(cos_pcm_bf > 0.999, "LFM3A pcm bf16_ref cos={cos_pcm_bf}");
        assert!(cos_pcm > 0.998, "LFM3A pcm cos={cos_pcm}");

        // 2. Realistic LFM3-Audio-2B / LFM3-Omni-2B geometry: 17 layers (12 ShortConv + 5 SWA Attn),
        //    d=768, ff=1792, 8Q/4KV (head_dim=96), 5-layer 256-ch stem, ~2.5s PCM.
        let bench_w = Arc::new(synth_lfm3a_audio_weights(768, 1792, 8, 4, 12, 5, 256));
        let bench_bf16 = Arc::new(to_bf16_audio(&bench_w));
        let bench_enc = try_xdna_audio_encoder(&bench_w).expect("XDNA LFM3-Audio-2B encoder");
        let long_pcm: Vec<f32> = (0..40_000).map(|i| (i as f32 * 0.03).sin() * 0.3).collect();
        let _ = bench_enc.encode_pcm(&long_pcm).expect("warmup");
        let t0 = std::time::Instant::now();
        let (o_npu, tn) = bench_enc.encode_pcm(&long_pcm).expect("npu");
        let dt_npu = t0.elapsed();
        let t1 = std::time::Instant::now();
        let (o_cpu, tc) = encode_audio_pcm(&long_pcm, &bench_w);
        let dt_cpu = t1.elapsed();
        let (o_bf, _) = encode_audio_pcm(&long_pcm, &bench_bf16);
        assert_eq!(tn, tc);
        let cos_bench = cosine_sim(&o_npu, &o_cpu);
        let cos_bench_bf = cosine_sim(&o_npu, &o_bf);
        eprintln!(
            "LFM3-Audio-2B bench (17L, d=768, ff=1792, 2.5s PCM -> t={tn}): npu={:.1}ms cpu={:.1}ms cos={cos_bench:.6} cos_vs_bf16_ref={cos_bench_bf:.6}",
            dt_npu.as_secs_f64() * 1e3,
            dt_cpu.as_secs_f64() * 1e3,
        );
        assert!(cos_bench_bf > 0.998 || cos_bench > 0.999);
    }

    #[test]
    #[ignore = "requires physical AMD XDNA2 NPU2 hardware"]
    fn hw_lfm3_omni_concurrent_npu_and_lfm3a_gen_eval() {
        // Verify LFM3-Omni concurrent NPU allocation (LLM device + shared Vision/Audio XdnaGemmRunner)
        // AND measure lfm3a_gen M=6 detokenizer & M=13 depthformer GEMM cost on gemm_bf16 vs CPU.
        let extra_dev = XdnaDevice::open(0).expect("concurrent XdnaDevice (LLM trunk slot)");
        let vw = synth_lfm3_vision_weights(128, 256, 8, 4, 1, 1);
        let aw = Arc::new(synth_lfm3a_audio_weights(128, 256, 4, 2, 1, 1, 32));
        let venc = try_xdna_vision_encoder(&vw).expect("Omni XDNA vision encoder");
        let aenc = try_xdna_audio_encoder(&aw).expect("Omni XDNA audio encoder");

        let px = det_vec(3 * 64 * 64, 5, 0.4);
        let v_out = venc.encode_image(&px, 4, 4).expect("Omni vision encode");
        assert!(!v_out.is_empty());

        let pcm: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.04).sin() * 0.2).collect();
        let (a_out, a_t) = aenc.encode_pcm(&pcm).expect("Omni audio encode");
        assert!(a_t > 0 && !a_out.is_empty());

        drop(venc);
        drop(aenc);
        drop(extra_dev);

        // Microbenchmark lfm3a_gen single-frame GEMM shapes on gemm_bf16 (M_TILE=128) vs CPU.
        let runner = XdnaGemmRunner::open().expect("open");
        for (m, n, k, label) in [
            (6, 2304, 512, "detok_ffn_M6"),
            (13, 2816, 1024, "depthformer_M13"),
        ] {
            let x = det_vec(m * k, 11, 0.3);
            let w_f32 = det_vec(n * k, 22, 0.2);
            let w_npu = runner.upload_f32_weight(&w_f32, n, k).expect("upload");
            let w_mmap = MmapWeight::from_owned_f32(w_f32, n, k);
            let _ = runner.matmul_f32(&x, &w_npu, m, n, k).expect("warm");
            let t0 = std::time::Instant::now();
            for _ in 0..8 {
                let _ = runner.matmul_f32(&x, &w_npu, m, n, k).expect("npu");
            }
            let dt_npu = t0.elapsed() / 8;
            let mut y = vec![0.0f32; m * n];
            let t1 = std::time::Instant::now();
            for _ in 0..8 {
                w_mmap.batched_matmul(&x, &mut y, m);
            }
            let dt_cpu = t1.elapsed() / 8;
            eprintln!(
                "lfm3a_gen {label} ({m}x{k} -> {m}x{n}): gemm_bf16={:.2}ms cpu={:.2}ms",
                dt_npu.as_secs_f64() * 1e3,
                dt_cpu.as_secs_f64() * 1e3,
            );
        }
    }

    #[test]
    fn xdna_vit_and_audio_lfm3_host_ops_match_cpu_reference() {
        fake::with_shim(FakeConfig::default(), || {
            let runner = XdnaGemmRunner::open().expect("open");
            let vops = XdnaVitOps::new(runner.clone());
            let aops = XdnaAudioOps::new(runner);

            // 1. Vit rmsnorm & silu_mul vs CPU primitives.
            let (rows, dim) = (5usize, 16usize);
            let src = det_vec(rows * dim, 10, 0.8);
            let rw = det_vec(dim, 11, 0.4)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>();
            let got_rms = vops.download(
                &vops.rmsnorm(&vops.upload(&src), &vops.upload(&rw), 1e-5, rows, dim),
                rows * dim,
            );
            let mut want_rms = src.clone();
            for row in want_rms.chunks_exact_mut(dim) {
                crate::backend::cpu::rmsnorm(row, &rw, 1e-5);
            }
            let rms_diff = got_rms
                .iter()
                .zip(want_rms.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(rms_diff < 1e-5, "rmsnorm diff {rms_diff}");

            let gate = det_vec(rows * dim, 12, 0.7);
            let up = det_vec(rows * dim, 13, 0.7);
            let g_buf = vops.upload(&gate);
            let u_buf = vops.upload(&up);
            vops.silu_mul(&g_buf, &u_buf, rows * dim);
            let got_silu_mul = vops.download(&g_buf, rows * dim);
            let mut want_silu_mul = gate.clone();
            silu_mul_inplace(&mut want_silu_mul, &up);
            let sm_diff = got_silu_mul
                .iter()
                .zip(want_silu_mul.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(sm_diff < 1e-5, "silu_mul diff {sm_diff}");

            // 2. Vit lfm3_conv_spatial (3*d double-gated with bias and 2*d single-gated without bias).
            let (grid_w, grid_h, d, kernel_size) = (4usize, 3usize, 8usize, 3usize);
            let n_tokens = grid_w * grid_h;
            let dw = det_vec(d * kernel_size * kernel_size, 14, 0.3);
            let db = det_vec(d, 15, 0.1);
            for (proj_mult, bias_opt) in [(3usize, Some(db.as_slice())), (2usize, None)] {
                let proj_rows = proj_mult * d;
                let proj = det_vec(n_tokens * proj_rows, 16 + proj_mult, 0.5);
                let got_conv = vops.download(
                    &vops.lfm3_conv_spatial(
                        &vops.upload(&proj),
                        proj_rows,
                        &dw,
                        bias_opt,
                        grid_w,
                        grid_h,
                        d,
                        kernel_size,
                    ),
                    n_tokens * d,
                );
                let mut gated = vec![0.0f32; n_tokens * d];
                for t in 0..n_tokens {
                    let row = &proj[t * proj_rows..(t + 1) * proj_rows];
                    for c in 0..d {
                        let rhs = if proj_mult == 3 {
                            row[2 * d + c]
                        } else {
                            row[d + c]
                        };
                        gated[t * d + c] = row[c] * rhs;
                    }
                }
                let mut want_conv = vec![0.0f32; n_tokens * d];
                depthwise_conv2d_same_into(
                    &gated,
                    &dw,
                    bias_opt,
                    grid_w,
                    grid_h,
                    d,
                    kernel_size,
                    &mut want_conv,
                );
                if proj_mult == 3 {
                    for t in 0..n_tokens {
                        let middle = &proj[t * 3 * d + d..t * 3 * d + 2 * d];
                        for c in 0..d {
                            want_conv[t * d + c] *= middle[c];
                        }
                    }
                }
                let cdiff = got_conv
                    .iter()
                    .zip(want_conv.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    cdiff < 1e-5,
                    "lfm3_conv_spatial mult={proj_mult} diff {cdiff}"
                );
            }

            // 3. Vit lfm3_gqa_mrope_attention vs CPU apply_2d_vision_mrope_qk + vit_attention_gqa.
            let (gw, gh, n_head, n_head_kv, hd) = (5usize, 3usize, 8usize, 2usize, 8usize);
            let toks = gw * gh;
            let q_dim = n_head * hd;
            let kv_dim = n_head_kv * hd;
            let q = det_vec(toks * q_dim, 20, 0.35);
            let k = det_vec(toks * kv_dim, 21, 0.35);
            let v = det_vec(toks * kv_dim, 22, 0.35);
            let got_gqa = vops.download(
                &vops.lfm3_gqa_mrope_attention(
                    &vops.upload(&q),
                    &vops.upload(&k),
                    &vops.upload(&v),
                    gw,
                    gh,
                    n_head,
                    n_head_kv,
                    hd,
                    1_000_000.0,
                ),
                toks * q_dim,
            );
            let mut q_ref = q;
            let mut k_ref = k;
            apply_2d_vision_mrope_qk(
                &mut q_ref,
                &mut k_ref,
                gw,
                gh,
                n_head,
                n_head_kv,
                hd,
                1_000_000.0,
            );
            let chunk_q = q_dim.saturating_sub(1).clamp(1, 32);
            let mut heads_ref = vec![0.0f32; n_head * toks.next_multiple_of(chunk_q) * hd];
            let mut want_gqa = vec![0.0f32; toks * q_dim];
            vit_attention_gqa(
                &q_ref,
                &k_ref,
                &v,
                &mut want_gqa,
                &mut heads_ref,
                toks,
                n_head,
                n_head_kv,
                hd,
                1.0f32 / (hd as f32).sqrt(),
            );
            let gqa_diff = got_gqa
                .iter()
                .zip(want_gqa.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(gqa_diff < 1e-5, "lfm3_gqa_mrope_attention diff {gqa_diff}");

            // 4. Audio lfm3a_conv_stem vs conv_stem_lfm3a_features.
            let aw = synth_lfm3a_audio_weights(64, 128, 4, 2, 1, 1, 16);
            let n_frames = 33usize;
            let n_mel = aw.config.n_mel_bins;
            let mel = det_vec(n_frames * n_mel, 30, 0.5);
            let (stem_buf, got_t_out) =
                aops.lfm3a_conv_stem(&aops.upload(&mel), n_frames, n_mel, &aw.conv_stem.layers);
            let (want_stem, want_t_out, plane) =
                conv_stem_lfm3a_features(&mel, n_frames, &aw.conv_stem.layers, n_mel);
            assert_eq!(got_t_out, want_t_out);
            let got_stem = aops.download(&stem_buf, got_t_out * plane);
            assert_eq!(got_stem, want_stem);

            // 5. Audio lfm3a_shortconv_dw vs scalar causal conv1d reference.
            let (t_sc, d_sc, k_sc) = (7usize, 12usize, 3usize);
            let bcx = det_vec(t_sc * 3 * d_sc, 31, 0.4);
            let dw_w = det_vec(k_sc * d_sc, 32, 0.3);
            let dw_b = det_vec(d_sc, 33, 0.1);
            for use_bias in [true, false] {
                let b_slice: &[f32] = if use_bias { &dw_b } else { &[] };
                let got_sc = aops.download(
                    &aops.lfm3a_shortconv_dw(&aops.upload(&bcx), &dw_w, b_slice, k_sc, t_sc, d_sc),
                    t_sc * d_sc,
                );
                let mut want_sc = vec![0.0f32; t_sc * d_sc];
                for ti in 0..t_sc {
                    for ch in 0..d_sc {
                        let mut acc = if use_bias { dw_b[ch] } else { 0.0 };
                        for tap in 0..k_sc {
                            if ti + tap >= k_sc - 1 {
                                let src_t = ti + tap - (k_sc - 1);
                                let row = &bcx[src_t * 3 * d_sc..(src_t + 1) * 3 * d_sc];
                                let bx = row[ch] * row[2 * d_sc + ch];
                                acc += bx * dw_w[ch * k_sc + tap];
                            }
                        }
                        let c_val = bcx[ti * 3 * d_sc + d_sc + ch];
                        want_sc[ti * d_sc + ch] = acc * c_val;
                    }
                }
                let sc_diff = got_sc
                    .iter()
                    .zip(want_sc.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(sc_diff < 1e-5, "lfm3a_shortconv_dw diff {sc_diff}");
            }

            // 6. Audio lfm3a_swa_attention (None, head-gate, dim-gate) vs CPU math.
            let (t_a, nh, nkv, dh, sw) = (9usize, 4usize, 2usize, 8usize, 4usize);
            let da = nh * dh;
            let dakv = nkv * dh;
            let qa = det_vec(t_a * da, 40, 0.3);
            let ka = det_vec(t_a * dakv, 41, 0.3);
            let va = det_vec(t_a * dakv, 42, 0.3);
            let qnw = det_vec(dh, 43, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>();
            let knw = det_vec(dh, 44, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>();
            for gate_rows in [0usize, nh, da] {
                let gate_vec =
                    (gate_rows > 0).then(|| det_vec(t_a * gate_rows, 45 + gate_rows, 0.4));
                let gate_buf = gate_vec.as_deref().map(|g| aops.upload(g));
                let got_swa = aops.download(
                    &aops.lfm3a_swa_attention(
                        &aops.upload(&qa),
                        &aops.upload(&ka),
                        &aops.upload(&va),
                        gate_buf.as_ref(),
                        gate_rows,
                        &qnw,
                        &knw,
                        t_a,
                        nh,
                        nkv,
                        dh,
                        1e-5,
                        10_000.0,
                        sw,
                    ),
                    t_a * da,
                );
                let mut q_ref = qa.clone();
                let mut k_ref = ka.clone();
                for ti in 0..t_a {
                    for h in 0..nh {
                        let qh = &mut q_ref[ti * da + h * dh..ti * da + (h + 1) * dh];
                        crate::backend::cpu::rmsnorm(qh, &qnw, 1e-5);
                        apply_rope_neox_head(qh, ti, 10_000.0);
                    }
                    for hk in 0..nkv {
                        let kh = &mut k_ref[ti * dakv + hk * dh..ti * dakv + (hk + 1) * dh];
                        crate::backend::cpu::rmsnorm(kh, &knw, 1e-5);
                        apply_rope_neox_head(kh, ti, 10_000.0);
                    }
                }
                let mut want_swa = vec![0.0f32; t_a * da];
                let n_rep = nh / nkv;
                let scale = 1.0f32 / (dh as f32).sqrt();
                for h in 0..nh {
                    let hk = h / n_rep;
                    for qt in 0..t_a {
                        let k0 = qt.saturating_sub(sw - 1);
                        let mut sc = vec![0.0f32; qt + 1 - k0];
                        for (idx, kt) in (k0..=qt).enumerate() {
                            sc[idx] = dot_f32(
                                &q_ref[qt * da + h * dh..qt * da + (h + 1) * dh],
                                &k_ref[kt * dakv + hk * dh..kt * dakv + (hk + 1) * dh],
                            ) * scale;
                        }
                        softmax_inplace(&mut sc);
                        let out_h = &mut want_swa[qt * da + h * dh..qt * da + (h + 1) * dh];
                        for (idx, kt) in (k0..=qt).enumerate() {
                            axpy_inplace(
                                out_h,
                                &va[kt * dakv + hk * dh..kt * dakv + (hk + 1) * dh],
                                sc[idx],
                            );
                        }
                    }
                }
                if let Some(gv) = &gate_vec {
                    if gate_rows == nh {
                        for ti in 0..t_a {
                            for h in 0..nh {
                                let sig = 1.0f32 / (1.0f32 + (-gv[ti * nh + h]).exp());
                                for v in &mut want_swa[ti * da + h * dh..ti * da + (h + 1) * dh] {
                                    *v *= sig;
                                }
                            }
                        }
                    } else {
                        for (dst, &g) in want_swa.iter_mut().zip(gv.iter()) {
                            *dst *= 1.0f32 / (1.0f32 + (-g).exp());
                        }
                    }
                }
                let swa_diff = got_swa
                    .iter()
                    .zip(want_swa.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    swa_diff < 1e-5,
                    "lfm3a_swa_attention gate_rows={gate_rows} diff {swa_diff}"
                );
            }
        });
    }

    #[test]
    fn xdna_multi_buffer_ops_handle_aliased_buffers_without_deadlock() {
        fake::with_shim(FakeConfig::default(), || {
            let runner = XdnaGemmRunner::open().expect("open");
            let vops = XdnaVitOps::new(runner.clone());
            let aops = XdnaAudioOps::new(runner);

            // add(a, a) doubles in place
            let a = vops.upload(&[1.0, -2.0, 3.5, 4.0]);
            vops.add(&a, &a, 4);
            assert_eq!(vops.download(&a, 4), vec![2.0, -4.0, 7.0, 8.0]);

            // bias_add(b, b, 2, 2) snapshots row 0 and adds to both rows
            let b = vops.upload(&[1.0, 2.0, 10.0, 20.0]);
            vops.bias_add(&b, &b, 2, 2);
            assert_eq!(vops.download(&b, 4), vec![2.0, 4.0, 11.0, 22.0]);

            // silu_mul(g, g, 4) computes silu(x) * x
            let orig = [0.5f32, -1.0, 2.0, 0.0];
            let g = vops.upload(&orig);
            vops.silu_mul(&g, &g, 4);
            let mut want_g = orig;
            silu_mul_inplace(&mut want_g, &orig);
            assert_eq!(vops.download(&g, 4), want_g);

            // layernorm(c, c, c) & rmsnorm(c, c)
            let c = vops.upload(&[1.0, 2.0, 3.0, 4.0]);
            let ln = vops.layernorm(&c, &c, &c, 1e-5, 1, 4);
            assert_eq!(vops.download(&ln, 4).len(), 4);
            let rn = vops.rmsnorm(&c, &c, 1e-5, 1, 4);
            assert_eq!(vops.download(&rn, 4).len(), 4);

            // scaled_add(s, s, 4, 0.5) scales by 1.5
            let s = aops.upload(&[2.0, -4.0, 6.0, 8.0]);
            aops.scaled_add(&s, &s, 4, 0.5);
            assert_eq!(aops.download(&s, 4), vec![3.0, -6.0, 9.0, 12.0]);

            // chan_affine_silu(ca, ca, ca, 2, 2)
            let ca = aops.upload(&[1.0, 2.0, 3.0, 4.0]);
            aops.chan_affine_silu(&ca, &ca, &ca, 2, 2);
            assert_eq!(aops.download(&ca, 4).len(), 4);

            // conv2d(cv, cv, cv)
            let cv = aops.upload(&[1.0; 16]);
            let spec = Conv2dSpec::padded(1, 1, 2, 2, (1, 1), (1, 1), (0, 0), (0, 0), 1).unwrap();
            let cv_out = aops.conv2d(&cv, &cv, &cv, &spec);
            assert_eq!(aops.download(&cv_out, 4), vec![2.0; 4]);

            // stft_frames & mel_project with aliased inputs
            let pcm = aops.upload(&vec![0.1f32; N_FFT]);
            let fr = aops.stft_frames(&pcm, &pcm, N_FFT, 1);
            assert_eq!(aops.download(&fr, N_FFT).len(), N_FFT);
            let pwr = aops.upload(&vec![0.5f32; N_FFT_BINS]);
            let mp = aops.mel_project(&pwr, &pwr, 1, 1);
            assert_eq!(aops.download(&mp, 1).len(), 1);
        });
    }

    #[test]
    fn fake_shim_split_k_matmul_f32_and_validation_errors() {
        fake::with_shim(FakeConfig::default(), || {
            let runner = XdnaGemmRunner::open().expect("open");
            // Split-K across 2 slices (`in_dim = 8256 > MAX_K_SLICE = 8192`).
            let (rows, out_dim, in_dim) = (4usize, 32usize, 8256usize);
            let x = det_vec(rows * in_dim, 51, 0.2);
            let w_f32 = det_vec(out_dim * in_dim, 52, 0.1);
            let w = runner
                .upload_f32_weight(&w_f32, out_dim, in_dim)
                .expect("split-K upload");
            assert_eq!(w.k_slices.len(), 2);
            assert_eq!(
                (
                    w.k_slices[0].k_start,
                    w.k_slices[0].k_len,
                    w.k_slices[0].k_chunks
                ),
                (0, 8192, 128)
            );
            assert_eq!(
                (
                    w.k_slices[1].k_start,
                    w.k_slices[1].k_len,
                    w.k_slices[1].k_chunks
                ),
                (8192, 64, 1)
            );
            let y = runner
                .matmul_f32(&x, &w, rows, out_dim, in_dim)
                .expect("split-K matmul");
            assert_eq!(y.len(), rows * out_dim);

            // `upload_mmap_weight` validation errors instead of panics.
            let bad_u8 = MmapWeight::from_owned_bytes(vec![0u8; 64], DType::U8, 2, 32);
            assert!(runner.upload_mmap_weight(&bad_u8).is_err());
            let bad_i32 = MmapWeight::from_owned_bytes(vec![0u8; 64], DType::I32, 1, 16);
            assert!(runner.upload_mmap_weight(&bad_i32).is_err());
            let bad_q8_cols = MmapWeight::from_owned_bytes(vec![0u8; 68], DType::Q8_0, 1, 48);
            assert!(runner.upload_mmap_weight(&bad_q8_cols).is_err());
            let bad_short = MmapWeight::from_owned_bytes(vec![0u8; 16], DType::F32, 4, 4);
            assert!(runner.upload_mmap_weight(&bad_short).is_err());

            // `upload_f32_weight` checked_mul / length errors.
            assert!(runner.upload_f32_weight(&[], usize::MAX, 2).is_err());
            assert!(runner.upload_f32_weight(&[0.0; 3], 2, 2).is_err());

            drop(w);
            drop(runner);
            assert_eq!(fake::live_bo_count(), 0);
        });

        // Error unwind inside runlist execution drops `runlist` before `runs` and leaks zero BOs.
        let fail_cfg = FakeConfig {
            fail_runlist_wait: true,
            ..Default::default()
        };
        fake::with_shim(fail_cfg, || {
            let runner = XdnaGemmRunner::open().expect("open");
            let w = runner
                .upload_f32_weight(&vec![0.1f32; 32 * 64], 32, 64)
                .expect("upload");
            let err = runner
                .matmul_f32(&vec![0.1f32; 4 * 64], &w, 4, 32, 64)
                .expect_err("runlist wait error must surface");
            assert!(err.to_string().contains("wait gemm_bf16 runlist"));
            drop(w);
            drop(runner);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }

    #[test]
    fn lfm3_and_lfm3a_malformed_weights_and_auto_routing() {
        use crate::engine::BackendPreference as BP;
        use crate::model::audio_encoder_gpu::{GpuAudioWeights, build_gpu_audio_encoder};
        use crate::model::vision_encoder_gpu::{GpuVitWeights, build_gpu_vision_encoder};

        fake::with_shim(FakeConfig::default(), || {
            let runner = XdnaGemmRunner::open().expect("open");
            let vops = XdnaVitOps::new(runner.clone());
            let aops = XdnaAudioOps::new(runner);

            // Malformed LFM3 vision weights fail cleanly instead of panicking.
            let mut bad_in_proj = synth_lfm3_vision_weights(64, 128, 4, 2, 1, 1);
            if let Lfm3BlockOp::Conv(ref mut c) = bad_in_proj.lfm3_blocks[0].op {
                c.in_proj_w = f32_weight(64, 64, 1, 0.1);
            }
            assert!(GpuVitWeights::try_build(&vops, &bad_in_proj).is_err());
            assert!(try_xdna_vision_encoder(&bad_in_proj).is_none());

            let mut bad_heads = synth_lfm3_vision_weights(64, 128, 4, 2, 1, 1);
            bad_heads.config.n_head = 0;
            assert!(GpuVitWeights::try_build(&vops, &bad_heads).is_err());

            let mut bad_mm_norm = synth_lfm3_vision_weights(64, 128, 4, 2, 1, 1);
            bad_mm_norm.mm_input_norm = Some(Lfm3NormWeights {
                weight: vec![1.0; 12],
                bias: None,
            });
            assert!(GpuVitWeights::try_build(&vops, &bad_mm_norm).is_err());

            // Malformed LFM3A audio weights fail cleanly instead of panicking.
            let mut bad_kv = synth_lfm3a_audio_weights(64, 128, 4, 2, 1, 1, 16);
            bad_kv.layers[0].lfm3a.as_mut().unwrap().n_head_kv = 0;
            assert!(GpuAudioWeights::build(&aops, &bad_kv).is_err());
            let bad_kv = Arc::new(bad_kv);
            assert!(try_xdna_audio_encoder(&bad_kv).is_none());

            let mut bad_post_ln = synth_lfm3a_audio_weights(64, 128, 4, 2, 1, 1, 16);
            bad_post_ln.layers[0].lfm3a.as_mut().unwrap().post_ln_w = vec![1.0; 7];
            assert!(GpuAudioWeights::build(&aops, &bad_post_ln).is_err());

            let mut bad_attn_b = synth_lfm3a_audio_weights(64, 128, 4, 2, 1, 1, 16);
            bad_attn_b.layers[1].attn_q_b = vec![0.0; 3];
            assert!(GpuAudioWeights::build(&aops, &bad_attn_b).is_err());

            drop(vops);
            drop(aops);

            // Auto routing on LFM3 vision uses `lfm3_blocks.len()`, not empty `blocks`.
            // 5 layers, n_embd=64 <= 768 -> worthwhile under Auto!
            let vw_deep_narrow = synth_lfm3_vision_weights(64, 128, 4, 2, 3, 2);
            let enc_auto = build_gpu_vision_encoder(&vw_deep_narrow, BP::Auto)
                .expect("5-layer narrow LFM3 vision encoder must route to XDNA under Auto");
            assert_eq!(enc_auto.backend_label(), "xdna");

            // 3 layers, n_embd=64 <= 768 -> skipped under Auto, admitted under Npu.
            let vw_short_narrow = synth_lfm3_vision_weights(64, 128, 4, 2, 2, 1);
            assert!(build_gpu_vision_encoder(&vw_short_narrow, BP::Auto).is_none());
            let enc_npu = build_gpu_vision_encoder(&vw_short_narrow, BP::Npu)
                .expect("short narrow LFM3 vision encoder must route to XDNA under Npu");
            assert_eq!(enc_npu.backend_label(), "xdna");

            // 2 layers, n_embd=1024 > 768 -> worthwhile under Auto.
            let vw_short_wide = synth_lfm3_vision_weights(1024, 1024, 8, 4, 1, 1);
            let enc_wide = build_gpu_vision_encoder(&vw_short_wide, BP::Auto)
                .expect("wide LFM3 vision encoder must route to XDNA under Auto");
            assert_eq!(enc_wide.backend_label(), "xdna");

            // Auto routing on LFM3A audio: 5 layers, n_embd=64 <= 512 -> routes to XDNA under Auto.
            let aw_deep = Arc::new(synth_lfm3a_audio_weights(64, 128, 4, 2, 3, 2, 16));
            let aenc_auto = build_gpu_audio_encoder(&aw_deep, BP::Auto)
                .expect("5-layer narrow LFM3A audio encoder must route to XDNA under Auto");
            let pcm: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.05).sin() * 0.2).collect();
            let (aout, t_out) = aenc_auto.encode_pcm(&pcm).expect("auto encode_pcm");
            assert!(t_out > 0 && aout.len() == t_out * 64);

            // 3 layers, n_embd=64 <= 512 -> skipped under Auto, admitted under Npu.
            let aw_short = Arc::new(synth_lfm3a_audio_weights(64, 128, 4, 2, 1, 2, 16));
            assert!(build_gpu_audio_encoder(&aw_short, BP::Auto).is_none());
            assert!(build_gpu_audio_encoder(&aw_short, BP::Npu).is_some());

            drop(enc_auto);
            drop(enc_npu);
            drop(enc_wide);
            drop(aenc_auto);
            assert_eq!(fake::live_bo_count(), 0);
        });
    }
}

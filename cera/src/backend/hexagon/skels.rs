//! Embedded Hexagon DSP skels + device probe.
//!
//! The `skels/` directory vendors the four `libggml-htp-vXX.so` DSP
//! libraries (see `skels/SOURCE.md` for provenance). They are embedded
//! in the binary so apps ship one artifact: at runtime the engine writes
//! the matching skel next to itself (or into an app-provided directory)
//! where FastRPC's loader can find it.

use std::sync::Arc;

use super::device::{HexagonArch, HexagonDevice};
use super::sys::FastRpcDriver;
use crate::session::CeraError;

/// Embedded skel bytes per architecture (see `skels/SOURCE.md`).
pub fn embedded_skel(arch: HexagonArch) -> &'static [u8] {
    match arch {
        HexagonArch::V73 => include_bytes!("skels/libggml-htp-v73.so"),
        HexagonArch::V75 => include_bytes!("skels/libggml-htp-v75.so"),
        HexagonArch::V79 => include_bytes!("skels/libggml-htp-v79.so"),
        HexagonArch::V81 => include_bytes!("skels/libggml-htp-v81.so"),
    }
}

/// All architectures we ship skels for, probe order (most common first).
pub const PROBE_ARCHS: [HexagonArch; 4] = [
    HexagonArch::V79,
    HexagonArch::V75,
    HexagonArch::V73,
    HexagonArch::V81,
];

/// Write the embedded skels into `dir` (created if missing) and point
/// FastRPC's loader at it via `ADSP_LIBRARY_PATH`. Prepends to an
/// existing value instead of replacing it. Returns the number of skels
/// written. Files are only rewritten when the size differs, so repeated
/// calls are cheap.
pub fn install_skels(dir: &std::path::Path) -> Result<usize, CeraError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CeraError::Backend(format!("skel dir {}: {e}", dir.display())))?;
    let mut count = 0;
    for arch in PROBE_ARCHS {
        let bytes = embedded_skel(arch);
        let path = dir.join(arch.skel_filename());
        let rewrite = std::fs::metadata(&path)
            .map(|m| m.len() != bytes.len() as u64)
            .unwrap_or(true);
        if rewrite {
            std::fs::write(&path, bytes)
                .map_err(|e| CeraError::Backend(format!("write {}: {e}", path.display())))?;
        }
        count += 1;
    }
    let dir_str = dir
        .to_str()
        .ok_or_else(|| CeraError::Backend("skel dir is not UTF-8".into()))?;
    let merged = match std::env::var("ADSP_LIBRARY_PATH") {
        Ok(cur) if !cur.is_empty() => format!("{dir_str}:{cur}"),
        _ => dir_str.to_string(),
    };
    // SAFETY: single-threaded setup call; the process has no DSP threads yet.
    unsafe { std::env::set_var("ADSP_LIBRARY_PATH", merged) };
    Ok(count)
}

/// Successful NPU probe result: the working arch plus DSP capabilities.
#[derive(Debug, Clone)]
pub struct HexagonProbe {
    pub arch: HexagonArch,
    pub n_threads: u32,
    pub n_hvx: u32,
    pub n_hmx: u32,
    pub vtcm_bytes: u64,
}

/// Open the FastRPC driver, try each bundled skel in [`PROBE_ARCHS`]
/// order, and return the first working device's capabilities. The probe
/// device is dropped before returning; use [`HexagonDevice`] for real
/// sessions. On failure the error aggregates every arch's message.
pub fn probe() -> Result<HexagonProbe, CeraError> {
    let driver = FastRpcDriver::load()?;
    let mut errors = Vec::new();
    for arch in PROBE_ARCHS {
        match HexagonDevice::new(Arc::clone(&driver), arch) {
            Ok(dev) => {
                let hw = dev.hw_info();
                return Ok(HexagonProbe {
                    arch: dev.arch(),
                    n_threads: hw.n_threads,
                    n_hvx: hw.n_hvx,
                    n_hmx: hw.n_hmx,
                    vtcm_bytes: hw.vtcm_size,
                });
            }
            Err(e) => errors.push(format!("{arch:?}: {e}")),
        }
    }
    Err(CeraError::Backend(format!(
        "no working Hexagon NPU ({}); {}",
        PROBE_ARCHS
            .iter()
            .map(|a| a.skel_filename())
            .collect::<Vec<_>>()
            .join(", "),
        errors.join("; ")
    )))
}

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

/// Serializes `install_skels`: two concurrent installs must neither
/// interleave a `var`/`set_var` pair (lost prepend) nor share a `.so.tmp`
/// path mid-write.
static SKEL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Write the embedded skels into `dir` (created if missing) and point
/// FastRPC's loader at it via `ADSP_LIBRARY_PATH`. Prepends to an
/// existing value instead of replacing it (no-op when `dir` is already
/// listed). Returns the number of skels written; files are only rewritten
/// when their bytes differ, so repeated calls are cheap and idempotent.
///
/// Call once during single-threaded startup, before spawning threads that
/// read the environment: `set_var` is process-global and cannot be made
/// sound against foreign threads (JVM, loader) racing a `getenv` from
/// inside this process.
pub fn install_skels(dir: &std::path::Path) -> Result<usize, CeraError> {
    // Held for the whole install: serializes the file writes (shared
    // `.so.tmp` names) as well as the `ADSP_LIBRARY_PATH` update below.
    let _env_guard = SKEL_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::create_dir_all(dir)
        .map_err(|e| CeraError::Backend(format!("skel dir {}: {e}", dir.display())))?;
    let mut count = 0;
    for arch in PROBE_ARCHS {
        let bytes = embedded_skel(arch);
        let path = dir.join(arch.skel_filename());
        // Compare contents, not just size: a corrupt same-length skel must
        // still be repaired, or the DSP loads wrong code with no signal.
        // (Length stat first: no point reading a file already known stale.)
        let rewrite = match std::fs::metadata(&path) {
            Ok(m) if m.len() != bytes.len() as u64 => true,
            Ok(_) => match std::fs::read(&path) {
                Ok(existing) => existing.as_slice() != bytes,
                Err(_) => true,
            },
            Err(_) => true,
        };
        if rewrite {
            // Publish atomically: FastRPC reads the loader-visible path, so
            // a crash mid-write must never leave a truncated `.so` behind.
            // Unique per process: same-process installs are mutex-serialized,
            // but two app processes share filesDir and would otherwise share
            // one staging path (truncate race publishes a truncated `.so`).
            // Disjoint tmps make last-rename-wins correct (identical bytes).
            // A killed install may leave its `*.so.tmp.<pid>` behind; it is
            // never loader-visible and safe to delete.
            let tmp = path.with_extension(format!("so.tmp.{}", std::process::id()));
            std::fs::write(&tmp, bytes)
                .map_err(|e| CeraError::Backend(format!("write {}: {e}", tmp.display())))?;
            std::fs::rename(&tmp, &path)
                .map_err(|e| CeraError::Backend(format!("rename {}: {e}", path.display())))?;
            count += 1;
        }
    }
    let dir_str = dir
        .to_str()
        .ok_or_else(|| CeraError::Backend("skel dir is not UTF-8".into()))?;
    let merged = match std::env::var("ADSP_LIBRARY_PATH") {
        Ok(cur) if !cur.is_empty() => {
            if cur.split(':').any(|p| p == dir_str) {
                cur
            } else {
                format!("{dir_str}:{cur}")
            }
        }
        _ => dir_str.to_string(),
    };
    // SAFETY: the mutex serializes every Rust-side `ADSP_LIBRARY_PATH`
    // read-modify-write; the caller upholds the documented startup-only
    // requirement, which is what keeps foreign-thread `getenv` out of the
    // race (no in-process primitive can enforce that half).
    unsafe { std::env::set_var("ADSP_LIBRARY_PATH", merged) };
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the `install_skels` contract the FFI docs promise: count means
    /// files written (0 when fresh), same-length corruption is repaired,
    /// the loader path is not duplicated, and re-running is a no-op.
    #[test]
    fn install_is_idempotent_and_repairs_corruption() {
        // NOTE: process-global env assertion; safe only because no other
        // test touches `ADSP_LIBRARY_PATH` (a second one would race this).
        // Save/restore so the suite leaves no trace either way.
        let prev = std::env::var_os("ADSP_LIBRARY_PATH");
        let dir = std::env::temp_dir().join(format!("skel-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // First install writes every skel...
        assert_eq!(install_skels(&dir).unwrap(), PROBE_ARCHS.len());
        // ...second is a no-op returning 0, env path not duplicated.
        assert_eq!(install_skels(&dir).unwrap(), 0);
        let v = std::env::var("ADSP_LIBRARY_PATH").unwrap();
        assert_eq!(
            v.split(':').filter(|p| *p == dir.to_str().unwrap()).count(),
            1
        );
        // Same-length corruption is still repaired.
        let p = dir.join(PROBE_ARCHS[0].skel_filename());
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(install_skels(&dir).unwrap(), 1);
        assert_eq!(std::fs::read(&p).unwrap(), embedded_skel(PROBE_ARCHS[0]));

        let _ = std::fs::remove_dir_all(&dir);
        match prev {
            Some(v) => unsafe { std::env::set_var("ADSP_LIBRARY_PATH", v) },
            None => unsafe { std::env::remove_var("ADSP_LIBRARY_PATH") },
        }
    }
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

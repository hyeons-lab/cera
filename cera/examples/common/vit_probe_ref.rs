//! The f32 reference embedding shared by the vision-tower probes (`vit_cpu_probe`, `vit_gpu_probe`).
//!
//! The int8 switch (`CERA_VIT_INT8`) is read from the environment on every encode, and changing the
//! environment of a process that already runs worker threads is not sound. So the reference comes from
//! a child process: the probe re-runs its own binary with `CERA_VIT_INT8=0` set from the start, and the
//! child writes the embedding it computed to a file the parent reads back.

use std::ffi::OsString;
use std::io::Write;
use std::path::Path;

/// Set (by the parent, for the child only) to the file the child writes its embedding to.
const REF_OUT_ENV: &str = "CERA_VIT_PROBE_INTERNAL_REF_OUT";

/// The file this process is asked to write the reference embedding to, when it is the reference child.
pub fn reference_out() -> Option<OsString> {
    std::env::var_os(REF_OUT_ENV)
}

/// The reference child's one job: write `embedding` raw (little-endian f32) to `path`. `create_new`, so a
/// pre-planted file or link at the (unique) path is never followed.
pub fn write_reference(path: &Path, embedding: &[f32]) {
    let raw: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("create the reference embedding file");
    f.write_all(&raw).expect("write the reference embedding");
}

/// The f32 reference embedding for `mmproj` and `image`, produced by the child run described above.
pub fn f32_reference(mmproj: &str, image: &str) -> Vec<f32> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let out =
        std::env::temp_dir().join(format!("vit_probe_ref_{}_{nonce}.f32", std::process::id()));
    let status = std::process::Command::new(std::env::current_exe().expect("current exe"))
        .args([mmproj, image])
        .env("CERA_VIT_INT8", "0")
        .env(REF_OUT_ENV, &out)
        .status()
        .expect("spawn the f32 reference run");
    let bytes = std::fs::read(&out);
    let _ = std::fs::remove_file(&out);
    assert!(status.success(), "the f32 reference run failed");
    let bytes = bytes.expect("read the reference embedding");
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "the reference file is truncated");
    chunks.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

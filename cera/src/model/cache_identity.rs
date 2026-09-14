//! Versioned backing-byte identity for built-in persistent prefix caches.

#[cfg(feature = "disk-cache")]
use crate::gguf::GgufFile;

/// Keep the caller's namespace, binding it to all loaded bytes when disk caching
/// is available. Empty identifiers retain the anonymous warm-only policy.
#[cfg(feature = "disk-cache")]
pub(super) fn for_loaded_weights(gguf: &GgufFile, name: &str) -> String {
    for_sources(&[gguf], name)
}

#[cfg(feature = "disk-cache")]
fn for_sources(sources: &[&GgufFile], name: &str) -> String {
    if !name.is_empty() {
        use sha2::{Digest, Sha256};
        #[cfg(test)]
        if let Some(pause) = HASH_PAUSE.with_borrow_mut(Option::take) {
            pause();
        }
        let mut hash = Sha256::new();
        hash.update(b"Cera cold cache weights v1\0");
        hash.update((sources.len() as u64).to_le_bytes());
        for source in sources {
            let bytes = source.mmap_data();
            #[cfg(test)]
            HASHED_BYTES.with(|count| count.set(count.get() + bytes.len()));
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        return format!("{name}:gguf-sha256-v1:{:x}", hash.finalize());
    }
    name.to_owned()
}

#[cfg(any(
    feature = "gpu",
    all(feature = "metal", any(target_os = "macos", target_os = "ios"))
))]
pub(crate) fn for_gpu_source(
    source: &dyn super::gpu_weight_source::GpuWeightSource,
    name: &str,
) -> String {
    #[cfg(feature = "disk-cache")]
    if !name.is_empty()
        && let Some(sources) = source.cache_identity_sources()
    {
        return for_sources(&sources, name);
    }
    let _ = source;
    name.to_owned()
}

#[cfg(all(test, feature = "disk-cache"))]
thread_local! {
    pub(crate) static HASHED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static HASH_PAUSE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, feature = "disk-cache"))]
mod tests {
    use super::*;
    use crate::convert::writer::{GGML_TYPE_F32, GgufWriter};

    #[test]
    #[ignore = "manual loaded-byte hashing cost measurement; run in release mode"]
    fn measure_loaded_byte_identity_cost() {
        let payload_bytes = 64 * 1024 * 1024;
        let mut writer = GgufWriter::new();
        writer.add_string("general.architecture", "identity-cost-fixture");
        writer.add_tensor(
            "weights",
            vec![payload_bytes as u64 / 4],
            GGML_TYPE_F32,
            payload_bytes,
        );
        let mut bytes = Vec::new();
        writer.write_header_and_tensor_info(&mut bytes).unwrap();
        writer
            .write_tensor_data(&mut bytes, &vec![0x3f; payload_bytes])
            .unwrap();
        let gguf = GgufFile::from_bytes(bytes.into()).unwrap();
        let mut elapsed = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            std::hint::black_box(for_loaded_weights(&gguf, "cost-fixture"));
            elapsed.push(start.elapsed().as_micros());
        }
        elapsed.sort_unstable();
        eprintln!(
            "identity_cost bytes={} samples=5 median_us={} min_us={} max_us={}",
            gguf.mmap_data().len(),
            elapsed[2],
            elapsed[0],
            elapsed[4]
        );
    }
}

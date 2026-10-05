//! Probe-grade WAV input: a strict mono 16 kHz s16 decode plus fixed-window framing.
//!
//! This is deliberately stricter than the CLI's general reader (`read_wav_pcm16_mono`, which
//! down-mixes and accepts any rate): benchmark probes must refuse anything but exactly what they
//! measure, or a mis-decoded clip corrupts the verdict it reports. Every refusal below is
//! unit-tested; keep it that way.

use anyhow::{Context, bail, ensure};

/// A mono 16 kHz s16 WAV as f32 samples in `[-1, 1]`.
///
/// Walks RIFF chunks structurally (never a `b"data"` substring search, which misfires on earlier
/// chunks) and validates `fmt` before decoding. Anything else (another rate, channel count, bit
/// depth, truncated chunks, a data chunk before any `fmt` chunk, an odd-length data body) is a
/// typed error, never a silent mis-decode.
pub fn read_wav_mono_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
    ensure!(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{path}: not a WAV file"
    );
    let mut pos = 12;
    let mut fmt_seen = false;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = pos
            .checked_add(8)
            .and_then(|s| s.checked_add(len))
            .filter(|&e| e <= bytes.len());
        let Some(end) = end else {
            bail!("{path}: truncated {} chunk", String::from_utf8_lossy(id));
        };
        let body = &bytes[pos + 8..end];
        if id == b"fmt " {
            ensure!(body.len() >= 16, "{path}: truncated fmt chunk");
            let tag = u16::from_le_bytes([body[0], body[1]]);
            let ch = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            ensure!(
                (tag, ch, rate, bits) == (1, 1, 16_000, 16),
                "{path}: need mono 16 kHz s16, found tag={tag} ch={ch} rate={rate} bits={bits}"
            );
            fmt_seen = true;
        } else if id == b"data" {
            ensure!(fmt_seen, "{path}: data chunk before any fmt chunk");
            ensure!(body.len() % 2 == 0, "{path}: odd-length data chunk");
            return Ok(body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect());
        }
        pos = end + (len & 1);
    }
    bail!("{path}: no data chunk")
}

/// The complete `size`-sample windows of `pcm`, dropping a trailing partial window.
///
/// # Panics
///
/// Panics if `size` is zero.
pub fn full_windows(pcm: &[f32], size: usize) -> Vec<&[f32]> {
    pcm.chunks_exact(size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(
        fmt: Option<(u16, u16, u32, u16)>,
        data: Option<&[u8]>,
        prefix_chunks: &[(&[u8; 4], &[u8])],
    ) -> Vec<u8> {
        fn chunk(out: &mut Vec<u8>, id: &[u8; 4], body: &[u8]) {
            out.extend_from_slice(id);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
            if body.len() % 2 == 1 {
                out.push(0);
            }
        }
        let mut body = b"WAVE".to_vec();
        for (id, bytes) in prefix_chunks {
            chunk(&mut body, id, bytes);
        }
        if let Some((tag, ch, rate, bits)) = fmt {
            let mut fmt_body = Vec::new();
            fmt_body.extend_from_slice(&tag.to_le_bytes());
            fmt_body.extend_from_slice(&ch.to_le_bytes());
            fmt_body.extend_from_slice(&rate.to_le_bytes());
            fmt_body.extend_from_slice(&(rate * u32::from(ch) * u32::from(bits) / 8).to_le_bytes());
            fmt_body.extend_from_slice(&(ch * bits / 8).to_le_bytes());
            fmt_body.extend_from_slice(&bits.to_le_bytes());
            chunk(&mut body, b"fmt ", &fmt_body);
        }
        if let Some(data) = data {
            chunk(&mut body, b"data", data);
        }
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn write_temp(bytes: &[u8]) -> (tempfile::NamedTempFile, String) {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        let path = f.path().to_str().unwrap().to_string();
        (f, path)
    }

    const MONO_16K: (u16, u16, u32, u16) = (1, 1, 16_000, 16);

    #[test]
    fn valid_mono_16k_decodes_exactly() {
        let pcm: Vec<u8> = [0i16, 32767, -32768, 16384]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let bytes = wav_bytes(Some(MONO_16K), Some(&pcm), &[]);
        let (_f, path) = write_temp(&bytes);
        assert_eq!(
            read_wav_mono_16k(&path).unwrap(),
            vec![0.0, 32767.0 / 32768.0, -1.0, 0.5]
        );
    }

    #[test]
    fn decoy_data_magic_in_an_earlier_chunk_is_skipped() {
        let pcm: Vec<u8> = [1234i16].iter().flat_map(|s| s.to_le_bytes()).collect();
        let bytes = wav_bytes(
            Some(MONO_16K),
            Some(&pcm),
            // A naive `b"data"` substring search points inside this JUNK body.
            &[(b"JUNK", b"datajunk".as_slice())],
        );
        let (_f, path) = write_temp(&bytes);
        assert_eq!(read_wav_mono_16k(&path).unwrap(), vec![1234.0 / 32768.0]);
    }

    #[test]
    fn non_wav_and_empty_inputs_refused() {
        for bytes in [vec![], vec![0u8; 2056], b"data".to_vec()] {
            let (_f, path) = write_temp(&bytes);
            let err = read_wav_mono_16k(&path).unwrap_err().to_string();
            assert!(err.contains("not a WAV file"), "{err}");
        }
    }

    #[test]
    fn truncated_chunk_refused() {
        let mut bytes = wav_bytes(Some(MONO_16K), None, &[]);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&1000u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 10]);
        let (_f, path) = write_temp(&bytes);
        let err = read_wav_mono_16k(&path).unwrap_err().to_string();
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn fmt_mismatches_refused() {
        for (fmt, field) in [
            ((1, 2, 16_000, 16), "ch=2"),
            ((1, 1, 8_000, 16), "rate=8000"),
            ((1, 1, 16_000, 8), "bits=8"),
            ((3, 1, 16_000, 16), "tag=3"),
        ] {
            let bytes = wav_bytes(Some(fmt), Some(&[0u8; 4]), &[]);
            let (_f, path) = write_temp(&bytes);
            let err = read_wav_mono_16k(&path).unwrap_err().to_string();
            assert!(err.contains("need mono 16 kHz s16"), "{err}");
            assert!(err.contains(field), "{err}");
        }
    }

    #[test]
    fn fmt_without_data_refused() {
        let bytes = wav_bytes(Some(MONO_16K), None, &[]);
        let (_f, path) = write_temp(&bytes);
        let err = read_wav_mono_16k(&path).unwrap_err().to_string();
        assert!(err.contains("no data chunk"), "{err}");
    }

    #[test]
    fn data_before_fmt_refused() {
        let mut body = b"WAVE".to_vec();
        body.extend_from_slice(b"data");
        body.extend_from_slice(&4u32.to_le_bytes());
        body.extend_from_slice(&[0u8; 4]);
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        let (_f, path) = write_temp(&out);
        let err = read_wav_mono_16k(&path).unwrap_err().to_string();
        assert!(err.contains("data chunk before any fmt chunk"), "{err}");
    }

    #[test]
    fn odd_length_data_refused() {
        let bytes = wav_bytes(Some(MONO_16K), Some(&[0u8; 3]), &[]);
        let (_f, path) = write_temp(&bytes);
        let err = read_wav_mono_16k(&path).unwrap_err().to_string();
        assert!(err.contains("odd-length data chunk"), "{err}");
    }

    #[test]
    fn full_windows_drops_a_trailing_partial_window() {
        let pcm = vec![0.0f32; 1025];
        assert_eq!(full_windows(&pcm, 512).len(), 2);
        assert_eq!(full_windows(&pcm[..513], 512).len(), 1);
        assert!(full_windows(&pcm[..511], 512).is_empty());
    }
}

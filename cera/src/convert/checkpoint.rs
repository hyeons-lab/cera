//! Incremental output hashing and bounded verification of a checkpoint prefix.

use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};

/// Hash only bytes accepted by the wrapped writer, including partial writes.
pub(super) struct HashingWriter<W> {
    pub(super) inner: W,
    pub(super) hasher: Sha256,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.hasher.update(&buf[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Read exactly the durable prefix, excluding any uncheckpointed tail. The
/// returned state can continue hashing new tensors without rereading old ones.
pub(super) fn hash_prefix(reader: &mut impl Read, mut bytes: u64) -> io::Result<Sha256> {
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    while bytes != 0 {
        let len = bytes.min(buf.len() as u64) as usize;
        reader.read_exact(&mut buf[..len])?;
        hasher.update(&buf[..len]);
        bytes -= len as u64;
    }
    Ok(hasher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_writes_and_resume_hash_the_exact_stream() {
        struct ShortWriter(Vec<u8>);
        impl Write for ShortWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                let n = buf.len().min(3);
                self.0.extend_from_slice(&buf[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let prefix = vec![0xa5; 64 * 1024 + 19];
        let mut input = prefix.clone();
        input.extend_from_slice(b"ignored tail");
        let mut reader = input.as_slice();
        let hasher = hash_prefix(&mut reader, prefix.len() as u64).unwrap();
        assert_eq!(reader, b"ignored tail");
        let mut writer = HashingWriter {
            inner: ShortWriter(prefix),
            hasher,
        };
        writer.write_all(b"new tensor and padding").unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.hasher.finalize(), Sha256::digest(&writer.inner.0));
        assert_eq!(
            hash_prefix(&mut b"short".as_slice(), 6)
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}

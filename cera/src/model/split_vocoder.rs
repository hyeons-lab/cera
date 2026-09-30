//! Split-layout LFM2-Audio vocoders.
//!
//! The llama.cpp release of an LFM2-Audio bundle (`LiquidAI/LFM2.5-Audio-*-GGUF`,
//! and the LeapBundles that point at it) splits the audio decoder across two
//! files:
//!
//! - `vocoder-*.gguf`: the depthformer, the audio/code embeddings, and the ISTFT
//!   window (`emb.emb.weight`, `depthformer.*`, `depth_*`, `audio_embedding.*`).
//! - `tokenizer-*.gguf`: the 8-layer detokenizer backbone in llama.cpp's
//!   `blk.N.*` naming, plus its `dense_2` head and norm.
//!
//! The merged layout (`lfm.layers.N.*`, `lin.*`) that every backend loader reads
//! keeps both in one vocoder file. Rather than teach the CPU, Metal, wgpu and
//! wasm loaders about two sources, [`merge_split_vocoder`] renames the sidecar's
//! tensors into that layout and produces one ordinary GGUF.
//!
//! The sidecar's `token_embd.weight` (65,536 rows) is deliberately not carried
//! over: the detokenizer reads its code embedding from the vocoder's
//! `emb.emb.weight` (8 codebooks x 2048 codes). Loading the sidecar as a plain
//! fallback would pair the wrong embedding with these layers, which is why
//! `DetokenizerWeights::from_gguf` rejects it.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::convert::writer::{GgufWriter, MetadataValue};
use crate::gguf::{GgufFile, GgufValue};

/// A tensor only the merged layout has, so its absence from a vocoder marks it
/// as one that still needs the sidecar.
const MERGED_PROBE: &str = "lfm.layers.0.conv.in_proj.weight";
/// The same tensor in the sidecar's llama.cpp naming.
const SIDECAR_PROBE: &str = "blk.0.shortconv.in_proj.weight";

/// Layer-relative tensor names: (sidecar suffix, merged suffix).
const LAYER_SUFFIXES: &[(&str, &str)] = &[
    ("attn_norm.weight", "operator_norm.weight"),
    ("ffn_norm.weight", "ffn_norm.weight"),
    ("ffn_gate.weight", "feed_forward.w1.weight"),
    ("ffn_down.weight", "feed_forward.w2.weight"),
    ("ffn_up.weight", "feed_forward.w3.weight"),
    ("shortconv.in_proj.weight", "conv.in_proj.weight"),
    ("shortconv.out_proj.weight", "conv.out_proj.weight"),
    ("shortconv.conv.weight", "conv.conv.weight"),
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_output.weight", "self_attn.out_proj.weight"),
    ("attn_q_norm.weight", "self_attn.q_layernorm.weight"),
    ("attn_k_norm.weight", "self_attn.k_layernorm.weight"),
];

/// Non-layer tensors: (sidecar name, merged name).
const GLOBAL_TENSORS: &[(&str, &str)] = &[
    ("token_embd_norm.weight", "lfm.embedding_norm.weight"),
    ("dense_2.weight", "lin.weight"),
    ("dense_2.bias", "lin.bias"),
];

/// True when `vocoder` carries no detokenizer backbone of its own, i.e. it is
/// the llama.cpp half of a split layout (or not a vocoder at all).
pub fn is_split_vocoder(vocoder: &GgufFile) -> bool {
    !vocoder.tensors.contains_key(MERGED_PROBE) && !vocoder.tensors.contains_key(SIDECAR_PROBE)
}

/// True when `sidecar` is a llama.cpp audio-tokenizer GGUF carrying the
/// detokenizer backbone.
pub fn is_detok_sidecar(sidecar: &GgufFile) -> bool {
    sidecar.tensors.contains_key(SIDECAR_PROBE)
}

/// Map a sidecar tensor name onto the merged layout. `None` for tensors the
/// merged vocoder does not carry (notably `token_embd.weight`).
pub fn merged_name(sidecar_name: &str) -> Option<String> {
    if let Some((_, merged)) = GLOBAL_TENSORS.iter().find(|(s, _)| *s == sidecar_name) {
        return Some((*merged).to_string());
    }
    let rest = sidecar_name.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<usize>().ok()?;
    let (_, merged) = LAYER_SUFFIXES.iter().find(|(s, _)| *s == suffix)?;
    Some(format!("lfm.layers.{idx}.{merged}"))
}

fn metadata_value(v: &GgufValue) -> Option<MetadataValue> {
    Some(match v {
        GgufValue::U8(x) => MetadataValue::Uint8(*x),
        GgufValue::I8(x) => MetadataValue::Int8(*x),
        GgufValue::U16(x) => MetadataValue::Uint16(*x),
        GgufValue::I16(x) => MetadataValue::Int16(*x),
        GgufValue::U32(x) => MetadataValue::Uint32(*x),
        GgufValue::I32(x) => MetadataValue::Int32(*x),
        GgufValue::U64(x) => MetadataValue::Uint64(*x),
        GgufValue::I64(x) => MetadataValue::Int64(*x),
        GgufValue::F32(x) => MetadataValue::Float32(*x),
        GgufValue::F64(x) => MetadataValue::Float64(*x),
        GgufValue::Bool(x) => MetadataValue::Bool(*x),
        GgufValue::String(x) => MetadataValue::String(x.clone()),
        // Vocoder metadata is scalar; an array here is not one the audio
        // loaders read, so drop it rather than guess an element type.
        GgufValue::Array(_) => return None,
    })
}

/// Serialize `vocoder` plus the sidecar's detokenizer tensors, renamed into the
/// merged layout, as one GGUF.
pub fn merge_split_vocoder(vocoder: &GgufFile, sidecar: &GgufFile) -> Result<Vec<u8>> {
    ensure!(
        is_split_vocoder(vocoder),
        "vocoder already carries a detokenizer backbone; nothing to merge"
    );
    ensure!(
        is_detok_sidecar(sidecar),
        "audio tokenizer GGUF has no `{SIDECAR_PROBE}`; it is not a detokenizer sidecar"
    );

    let mut writer = GgufWriter::new();
    let mut meta_keys: Vec<_> = vocoder.metadata.keys().collect();
    meta_keys.sort();
    for k in meta_keys {
        // The writer emits its own alignment; a copied key would contradict it.
        if k == "general.alignment" {
            continue;
        }
        if let Some(v) = metadata_value(&vocoder.metadata[k]) {
            writer.add_metadata(k.clone(), v);
        }
    }

    // Deterministic order, so the same inputs always give the same bytes.
    let mut entries: Vec<(String, &GgufFile, &str)> = Vec::new();
    let mut voc_names: Vec<_> = vocoder.tensors.keys().collect();
    voc_names.sort();
    for n in voc_names {
        entries.push((n.clone(), vocoder, n.as_str()));
    }
    let mut side_names: Vec<_> = sidecar.tensors.keys().collect();
    side_names.sort();
    for n in side_names {
        let Some(merged) = merged_name(n) else {
            continue;
        };
        if vocoder.tensors.contains_key(&merged) {
            bail!("vocoder and sidecar both define `{merged}`");
        }
        entries.push((merged, sidecar, n.as_str()));
    }

    let mut payloads: Vec<&[u8]> = Vec::with_capacity(entries.len());
    for (name, src, src_name) in &entries {
        let info = &src.tensors[*src_name];
        let data = src
            .tensor_data(src_name)
            .with_context(|| format!("reading `{src_name}` for merged tensor `{name}`"))?;
        writer.add_tensor(
            name.clone(),
            info.shape.iter().map(|&d| d as u64).collect(),
            info.ggml_type_id,
            data.len(),
        );
        payloads.push(data);
    }

    let mut out = Vec::new();
    writer.write_header_and_tensor_info(&mut out)?;
    for data in payloads {
        writer.write_tensor_data(&mut out, data)?;
    }
    Ok(out)
}

/// The vocoder to hand to the audio loaders: `vocoder` itself when it is already
/// merged, otherwise the merge of `vocoder` and `sidecar`.
///
/// A split vocoder with no usable sidecar is returned unchanged with a warning,
/// so the loaders report their own missing-tensor error.
pub fn resolve_vocoder(vocoder: Arc<GgufFile>, sidecar: Option<&Arc<GgufFile>>) -> Arc<GgufFile> {
    if !is_split_vocoder(&vocoder) {
        return vocoder;
    }
    let Some(sidecar) = sidecar.filter(|s| is_detok_sidecar(s)) else {
        tracing::warn!(
            target: "cera::engine",
            "audio vocoder has no detokenizer backbone and no `tokenizer-*.gguf` sidecar \
             carries one; audio output will be unavailable"
        );
        return vocoder;
    };
    match merge_split_vocoder(&vocoder, sidecar).and_then(|b| GgufFile::from_bytes(b.into())) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                error = %format!("{e:#}"),
                "failed to merge split vocoder and tokenizer sidecar"
            );
            vocoder
        }
    }
}

/// The `tokenizer-*` file that pairs with a `vocoder-*` reference: same
/// directory or URL prefix, same quantization suffix. `None` when the reference
/// is not named like a llama.cpp vocoder.
pub fn sibling_tokenizer_ref(vocoder_ref: &str) -> Option<String> {
    let (head, file) = match vocoder_ref.rsplit_once('/') {
        Some((head, file)) => (Some(head), file),
        None => (None, vocoder_ref),
    };
    let rest = file.strip_prefix("vocoder-")?;
    Some(match head {
        Some(head) => format!("{head}/tokenizer-{rest}"),
        None => format!("tokenizer-{rest}"),
    })
}

/// Open a vocoder GGUF for a direct (non-manifest) load, folding in the sibling
/// `tokenizer-*.gguf` beside it when the vocoder is the split half of a
/// llama.cpp bundle. A merged vocoder opens exactly as [`GgufFile::open_arc`].
#[cfg(feature = "mmap")]
pub fn open_vocoder(path: &std::path::Path) -> Result<Arc<GgufFile>> {
    let vocoder = GgufFile::open_arc(path)?;
    if !is_split_vocoder(&vocoder) {
        return Ok(vocoder);
    }
    let sidecar_path = sibling_tokenizer_ref(&path.to_string_lossy())
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists());
    let sidecar = sidecar_path
        .as_deref()
        .and_then(|p| GgufFile::open_arc(p).ok());
    Ok(resolve_vocoder_cached(
        path,
        vocoder,
        sidecar_path.as_deref(),
        sidecar.as_ref(),
    ))
}

/// Path-based [`resolve_vocoder`] that keeps the merged GGUF on disk, so the
/// ~135 MB result is mmapped like any other vocoder instead of held on the heap
/// (and merged once, not on every load).
///
/// The merged file lives in a hidden `.cera-merged/` directory beside the
/// vocoder, and is reused while it is newer than both sources. Any I/O failure
/// (read-only bundle directory, full disk) falls back to the in-memory merge.
#[cfg(feature = "mmap")]
pub fn resolve_vocoder_cached(
    vocoder_path: &std::path::Path,
    vocoder: Arc<GgufFile>,
    sidecar_path: Option<&std::path::Path>,
    sidecar: Option<&Arc<GgufFile>>,
) -> Arc<GgufFile> {
    use std::time::SystemTime;

    if !is_split_vocoder(&vocoder) {
        return vocoder;
    }
    let (Some(sidecar_path), Some(sidecar)) = (sidecar_path, sidecar) else {
        return resolve_vocoder(vocoder, None);
    };
    if !is_detok_sidecar(sidecar) {
        return resolve_vocoder(vocoder, Some(sidecar));
    }

    let Some(name) = vocoder_path.file_name() else {
        return resolve_vocoder(vocoder, Some(sidecar));
    };
    let dir = vocoder_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(".cera-merged");
    let merged_path = dir.join(name);

    let mtime = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let newest_source = [mtime(vocoder_path), mtime(sidecar_path)]
        .into_iter()
        .flatten()
        .max();
    let fresh = |m: SystemTime| newest_source.is_none_or(|src| m >= src);

    if mtime(&merged_path).is_some_and(fresh)
        && let Ok(g) = GgufFile::open_arc(&merged_path)
        && !is_split_vocoder(&g)
    {
        return g;
    }

    let write = || -> Result<Arc<GgufFile>> {
        let bytes = merge_split_vocoder(&vocoder, sidecar)?;
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(format!(
            ".{}.{}.tmp",
            name.to_string_lossy(),
            std::process::id()
        ));
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &merged_path)?;
        GgufFile::open_arc(&merged_path)
    };
    match write() {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                target: "cera::engine",
                path = %merged_path.display(),
                error = %format!("{e:#}"),
                "could not cache merged vocoder on disk; merging in memory"
            );
            resolve_vocoder(vocoder, Some(sidecar))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::writer::{GGML_TYPE_F32, GgufWriter};

    fn gguf(tensors: &[(&str, Vec<u64>, Vec<f32>)], meta: &[(&str, u32)]) -> GgufFile {
        let mut w = GgufWriter::new();
        for (k, v) in meta {
            w.add_u32(*k, *v);
        }
        for (name, dims, data) in tensors {
            w.add_tensor(*name, dims.clone(), GGML_TYPE_F32, data.len() * 4);
        }
        let mut out = Vec::new();
        w.write_header_and_tensor_info(&mut out).unwrap();
        for (_, _, data) in tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            w.write_tensor_data(&mut out, &bytes).unwrap();
        }
        GgufFile::from_bytes(out.into()).unwrap()
    }

    fn f(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    #[test]
    fn renames_every_sidecar_tensor_the_loaders_read() {
        let cases = [
            (
                "blk.3.attn_norm.weight",
                "lfm.layers.3.operator_norm.weight",
            ),
            (
                "blk.0.ffn_gate.weight",
                "lfm.layers.0.feed_forward.w1.weight",
            ),
            (
                "blk.0.ffn_down.weight",
                "lfm.layers.0.feed_forward.w2.weight",
            ),
            ("blk.0.ffn_up.weight", "lfm.layers.0.feed_forward.w3.weight"),
            (
                "blk.1.shortconv.conv.weight",
                "lfm.layers.1.conv.conv.weight",
            ),
            (
                "blk.2.attn_output.weight",
                "lfm.layers.2.self_attn.out_proj.weight",
            ),
            (
                "blk.2.attn_q_norm.weight",
                "lfm.layers.2.self_attn.q_layernorm.weight",
            ),
            ("token_embd_norm.weight", "lfm.embedding_norm.weight"),
            ("dense_2.weight", "lin.weight"),
            ("dense_2.bias", "lin.bias"),
        ];
        for (from, to) in cases {
            assert_eq!(merged_name(from).as_deref(), Some(to), "{from}");
        }
        assert_eq!(merged_name("token_embd.weight"), None);
        assert_eq!(merged_name("blk.x.attn_norm.weight"), None);
        assert_eq!(merged_name("blk.0.unknown.weight"), None);
    }

    #[test]
    fn sibling_tokenizer_follows_the_vocoder_reference() {
        assert_eq!(
            sibling_tokenizer_ref(
                "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF/resolve/main/vocoder-LFM2.5-Audio-1.5B-JP-Q4_0.gguf"
            )
            .as_deref(),
            Some(
                "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF/resolve/main/tokenizer-LFM2.5-Audio-1.5B-JP-Q4_0.gguf"
            )
        );
        assert_eq!(
            sibling_tokenizer_ref("/models/a/vocoder-X-Q8_0.gguf").as_deref(),
            Some("/models/a/tokenizer-X-Q8_0.gguf")
        );
        assert_eq!(
            sibling_tokenizer_ref("vocoder-X.gguf").as_deref(),
            Some("tokenizer-X.gguf")
        );
        // Only llama.cpp-style names pair up; anything else has no sibling.
        assert_eq!(sibling_tokenizer_ref("/m/audio_decoder-Q4_0.gguf"), None);
    }

    #[test]
    fn merges_sidecar_into_vocoder_and_drops_token_embd() {
        let voc = gguf(
            &[
                ("emb.emb.weight", vec![4, 2], f(8, 0.0)),
                ("depthformer.layers.0.ffn_norm.weight", vec![2], f(2, 50.0)),
            ],
            &[("depthformer_n_layer", 6)],
        );
        let side = gguf(
            &[
                ("blk.0.shortconv.in_proj.weight", vec![4, 3], f(12, 100.0)),
                ("blk.2.attn_q.weight", vec![4, 2], f(8, 200.0)),
                ("dense_2.bias", vec![3], f(3, 300.0)),
                ("token_embd.weight", vec![4, 5], f(20, 400.0)),
            ],
            &[],
        );
        assert!(is_split_vocoder(&voc));
        assert!(is_detok_sidecar(&side));

        let merged =
            GgufFile::from_bytes(merge_split_vocoder(&voc, &side).unwrap().into()).unwrap();
        assert!(!is_split_vocoder(&merged));
        assert_eq!(merged.get_u32("depthformer_n_layer"), Some(6));
        assert!(!merged.tensors.contains_key("token_embd.weight"));
        for (name, want) in [
            ("emb.emb.weight", f(8, 0.0)),
            ("depthformer.layers.0.ffn_norm.weight", f(2, 50.0)),
            ("lfm.layers.0.conv.in_proj.weight", f(12, 100.0)),
            ("lfm.layers.2.self_attn.q_proj.weight", f(8, 200.0)),
            ("lin.bias", f(3, 300.0)),
        ] {
            assert_eq!(
                merged.get_tensor(name).unwrap().to_f32_vec(),
                want,
                "{name}"
            );
        }
        assert_eq!(
            merged
                .tensor_meta("lfm.layers.0.conv.in_proj.weight")
                .unwrap()
                .2,
            4,
            "input dim survives the rename"
        );
    }

    #[test]
    fn merge_is_deterministic() {
        let voc = gguf(
            &[("emb.emb.weight", vec![4, 2], f(8, 0.0))],
            &[("a", 1), ("b", 2)],
        );
        let side = gguf(
            &[("blk.0.shortconv.in_proj.weight", vec![4, 3], f(12, 1.0))],
            &[],
        );
        assert_eq!(
            merge_split_vocoder(&voc, &side).unwrap(),
            merge_split_vocoder(&voc, &side).unwrap()
        );
    }

    #[test]
    fn rejects_non_split_vocoder_and_non_sidecar() {
        let merged = gguf(&[(MERGED_PROBE, vec![4, 3], f(12, 0.0))], &[]);
        let side = gguf(&[(SIDECAR_PROBE, vec![4, 3], f(12, 0.0))], &[]);
        assert!(merge_split_vocoder(&merged, &side).is_err());

        let voc = gguf(&[("emb.emb.weight", vec![4, 2], f(8, 0.0))], &[]);
        let not_side = gguf(&[("something", vec![2], f(2, 0.0))], &[]);
        assert!(merge_split_vocoder(&voc, &not_side).is_err());
    }

    #[test]
    fn rejects_a_name_collision() {
        let voc = gguf(&[("lin.bias", vec![3], f(3, 0.0))], &[]);
        let side = gguf(
            &[
                (SIDECAR_PROBE, vec![4, 3], f(12, 0.0)),
                ("dense_2.bias", vec![3], f(3, 1.0)),
            ],
            &[],
        );
        assert!(merge_split_vocoder(&voc, &side).is_err());
    }

    #[test]
    fn resolve_passes_through_when_merged_or_sidecar_missing() {
        let merged = Arc::new(gguf(&[(MERGED_PROBE, vec![4, 3], f(12, 0.0))], &[]));
        assert!(Arc::ptr_eq(&resolve_vocoder(merged.clone(), None), &merged));

        let voc = Arc::new(gguf(&[("emb.emb.weight", vec![4, 2], f(8, 0.0))], &[]));
        assert!(Arc::ptr_eq(&resolve_vocoder(voc.clone(), None), &voc));
    }
}

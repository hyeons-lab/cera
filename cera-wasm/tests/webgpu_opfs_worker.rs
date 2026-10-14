//! Headless-Chrome Dedicated Worker test for OPFS -> WebGPU streaming
//! (`OpfsSyncHandle`, `OpfsGpuWeightSource`, and `GpuLfmModel::from_weight_source_with_ctx`).
//!
//! `FileSystemSyncAccessHandle` (`createSyncAccessHandle`) is only exposed by browsers
//! inside Dedicated Web Workers (such as `cera_worker.js` in the web demo). Running this
//! test with `run_in_dedicated_worker` exercises the exact worker OPFS -> header-only GGUF ->
//! `OpfsGpuWeightSource` -> WebGPU upload and forward path that previously failed with
//! `tensor token_embd.weight data extends beyond GGUF buffer`.

#![cfg(all(target_arch = "wasm32", feature = "wgpu"))]

use std::sync::Arc;

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemReadWriteOptions, FileSystemSyncAccessHandle, StorageManager,
};

use cera::gguf::GgufFile;
use cera::kv_cache::InferenceState;
use cera::model::gpu_lfm2::GpuLfmModel;
use cera::model::lfm2::LfmModel;
use cera::model::llama::LlamaModel;
use cera::model::{GpuWeightSource, Model};
use cera_wasm::opfs_source::{OpfsGpuWeightSource, OpfsSyncHandle};

wasm_bindgen_test_configure!(run_in_dedicated_worker);

/// Build a minimal valid GGUF v3 binary for either `"lfm2"` or `"llama"`.
/// Uses a large padding gap before tensor data so `GgufFile::from_header_bytes`
/// holding only the header (`..data_offset`) does NOT contain the tensor payload,
/// reproducing the 4 MiB header-only `GgufFile` in `read_gguf_header_from_sync_handle`.
fn build_minimal_gguf_bytes(arch: &str) -> Vec<u8> {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    fn put_kv_str(out: &mut Vec<u8>, k: &str, v: &str) {
        put_str(out, k);
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(out, v);
    }
    fn put_kv_u32(out: &mut Vec<u8>, k: &str, v: u32) {
        put_str(out, k);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_kv_f32(out: &mut Vec<u8>, k: &str, v: f32) {
        put_str(out, k);
        out.extend_from_slice(&6u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_kv_i32_array(out: &mut Vec<u8>, k: &str, vals: &[i32]) {
        put_str(out, k);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&5u32.to_le_bytes());
        out.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    const HS: usize = 128;
    const INTER: usize = 256;
    const VOCAB: usize = 32;

    let mut tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = Vec::new();
    let f32_vals = |n: usize, scale: f32, bias: f32| -> Vec<f32> {
        (0..n)
            .map(|i| bias + ((i * 7 + 3) % 29) as f32 * scale)
            .collect()
    };

    tensors.push((
        "token_embd.weight",
        vec![HS, VOCAB],
        f32_vals(HS * VOCAB, 0.01, -0.1),
    ));
    if arch == "lfm2" {
        tensors.push(("token_embd_norm.weight", vec![HS], vec![1.0; HS]));
        tensors.push(("blk.0.attn_q_norm.weight", vec![HS / 2], vec![1.0; HS / 2]));
        tensors.push(("blk.0.attn_k_norm.weight", vec![HS / 2], vec![1.0; HS / 2]));
    } else {
        tensors.push(("output_norm.weight", vec![HS], vec![1.0; HS]));
    }
    tensors.push(("blk.0.attn_norm.weight", vec![HS], vec![1.0; HS]));
    tensors.push(("blk.0.ffn_norm.weight", vec![HS], vec![1.0; HS]));
    tensors.push((
        "blk.0.attn_q.weight",
        vec![HS, HS],
        f32_vals(HS * HS, 0.01, -0.1),
    ));
    tensors.push((
        "blk.0.attn_k.weight",
        vec![HS, HS],
        f32_vals(HS * HS, 0.01, -0.1),
    ));
    tensors.push((
        "blk.0.attn_v.weight",
        vec![HS, HS],
        f32_vals(HS * HS, 0.01, -0.1),
    ));
    tensors.push((
        "blk.0.attn_output.weight",
        vec![HS, HS],
        f32_vals(HS * HS, 0.01, -0.1),
    ));
    tensors.push((
        "blk.0.ffn_gate.weight",
        vec![HS, INTER],
        f32_vals(HS * INTER, 0.01, -0.1),
    ));
    tensors.push((
        "blk.0.ffn_up.weight",
        vec![HS, INTER],
        f32_vals(HS * INTER, 0.01, -0.1),
    ));
    tensors.push((
        "blk.0.ffn_down.weight",
        vec![INTER, HS],
        f32_vals(INTER * HS, 0.01, -0.1),
    ));

    let mut kv_bytes = Vec::new();
    let mut n_kv = 0u64;
    put_kv_str(&mut kv_bytes, "general.architecture", arch);
    n_kv += 1;
    put_kv_u32(&mut kv_bytes, &format!("{arch}.block_count"), 1);
    n_kv += 1;
    put_kv_u32(
        &mut kv_bytes,
        &format!("{arch}.embedding_length"),
        HS as u32,
    );
    n_kv += 1;
    put_kv_u32(
        &mut kv_bytes,
        &format!("{arch}.feed_forward_length"),
        INTER as u32,
    );
    n_kv += 1;
    put_kv_u32(&mut kv_bytes, &format!("{arch}.attention.head_count"), 2);
    n_kv += 1;
    if arch == "lfm2" {
        put_kv_i32_array(&mut kv_bytes, "lfm2.attention.head_count_kv", &[2]);
        n_kv += 1;
        put_kv_u32(&mut kv_bytes, "lfm2.shortconv.l_cache", 3);
        n_kv += 1;
    } else {
        put_kv_u32(&mut kv_bytes, &format!("{arch}.attention.head_count_kv"), 2);
        n_kv += 1;
    }
    put_kv_f32(
        &mut kv_bytes,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );
    n_kv += 1;
    put_kv_f32(&mut kv_bytes, &format!("{arch}.rope.freq_base"), 10_000.0);
    n_kv += 1;
    put_kv_u32(&mut kv_bytes, &format!("{arch}.context_length"), 256);
    n_kv += 1;
    put_kv_u32(&mut kv_bytes, &format!("{arch}.vocab_size"), VOCAB as u32);
    n_kv += 1;

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&n_kv.to_le_bytes());
    out.extend_from_slice(&kv_bytes);

    let mut rel_offset = 0usize;
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
    for (name, dims, data) in &tensors {
        put_str(&mut out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // F32
        out.extend_from_slice(&(rel_offset as u64).to_le_bytes());
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        rel_offset = (rel_offset + bytes.len()).next_multiple_of(32);
        payloads.push(bytes);
    }

    let data_offset = out.len().next_multiple_of(32);
    out.resize(data_offset, 0);
    for bytes in payloads {
        out.extend_from_slice(&bytes);
        let padded = out.len().next_multiple_of(32);
        out.resize(padded, 0);
    }
    out
}

async fn write_and_open_opfs_sync(filename: &str, bytes: &[u8]) -> FileSystemSyncAccessHandle {
    let global = js_sys::global();
    let navigator =
        js_sys::Reflect::get(&global, &JsValue::from_str("navigator")).expect("global.navigator");
    let storage: StorageManager = js_sys::Reflect::get(&navigator, &JsValue::from_str("storage"))
        .expect("navigator.storage")
        .unchecked_into();
    let root: FileSystemDirectoryHandle = JsFuture::from(storage.get_directory())
        .await
        .expect("navigator.storage.getDirectory")
        .unchecked_into();
    let opts = FileSystemGetFileOptions::new();
    opts.set_create(true);
    let file_handle: FileSystemFileHandle =
        JsFuture::from(root.get_file_handle_with_options(filename, &opts))
            .await
            .expect("get_file_handle")
            .unchecked_into();
    let sync: FileSystemSyncAccessHandle = JsFuture::from(file_handle.create_sync_access_handle())
        .await
        .expect("create_sync_access_handle")
        .unchecked_into();
    sync.truncate_with_u32(0).expect("truncate");
    let rw = FileSystemReadWriteOptions::new();
    rw.set_at(0.0);
    let written = sync
        .write_with_u8_array_and_options(bytes, &rw)
        .expect("write");
    assert_eq!(written as usize, bytes.len());
    sync.flush().expect("flush");
    sync
}

fn assert_weight_source_parity(
    opfs: &OpfsGpuWeightSource,
    full: &dyn GpuWeightSource,
    header_only_gguf: &Arc<GgufFile>,
    full_gguf: &Arc<GgufFile>,
) {
    assert_eq!(opfs.config().hidden_size, full.config().hidden_size);
    assert_eq!(opfs.config().vocab_size, full.config().vocab_size);
    assert_eq!(opfs.output_norm_weight(), full.output_norm_weight());
    assert_eq!(opfs.attn_norm_weight(0), full.attn_norm_weight(0));
    assert_eq!(opfs.ffn_norm_weight(0), full.ffn_norm_weight(0));
    assert_eq!(opfs.attn_q_norm_weight(0), full.attn_q_norm_weight(0));
    assert_eq!(opfs.attn_k_norm_weight(0), full.attn_k_norm_weight(0));

    let emb_opfs = opfs
        .embedding_mmap_weight(header_only_gguf)
        .expect("opfs embedding_mmap_weight");
    let emb_full = full
        .embedding_mmap_weight(full_gguf)
        .expect("full embedding_mmap_weight");
    assert_eq!(emb_opfs.rows, emb_full.rows);
    assert_eq!(emb_opfs.cols, emb_full.cols);
    assert_eq!(emb_opfs.data(), emb_full.data());

    let pairs = [
        (
            opfs.attn_q_ref(0).unwrap(),
            full.attn_q_ref(0).unwrap(),
            "attn_q",
        ),
        (
            opfs.attn_k_ref(0).unwrap(),
            full.attn_k_ref(0).unwrap(),
            "attn_k",
        ),
        (
            opfs.attn_v_ref(0).unwrap(),
            full.attn_v_ref(0).unwrap(),
            "attn_v",
        ),
        (
            opfs.attn_output_ref(0).unwrap(),
            full.attn_output_ref(0).unwrap(),
            "attn_output",
        ),
        (
            opfs.ffn_gate_ref(0).unwrap(),
            full.ffn_gate_ref(0).unwrap(),
            "ffn_gate",
        ),
        (
            opfs.ffn_up_ref(0).unwrap(),
            full.ffn_up_ref(0).unwrap(),
            "ffn_up",
        ),
        (
            opfs.ffn_down_ref(0).unwrap(),
            full.ffn_down_ref(0).unwrap(),
            "ffn_down",
        ),
    ];
    for (o_ref, f_ref, label) in pairs {
        assert_eq!(
            opfs.weight_bytes(o_ref).unwrap().as_ref(),
            full.weight_bytes(f_ref).unwrap().as_ref(),
            "weight_bytes mismatch on {label}"
        );
        assert_eq!(
            opfs.dequantize_weight(o_ref).unwrap(),
            full.dequantize_weight(f_ref).unwrap(),
            "dequantize_weight mismatch on {label}"
        );
    }
}

#[wasm_bindgen_test]
async fn opfs_sync_handle_releases_lock_on_drop_and_streams_lfm2_and_llama_to_webgpu() {
    let ctx = cera::backend::wgpu::GpuContext::new_async()
        .await
        .expect("GpuContext::new_async");

    for arch in ["lfm2", "llama"] {
        let filename = format!("test-webgpu-opfs-{arch}.gguf");
        let full_bytes = build_minimal_gguf_bytes(arch);
        let full_gguf = GgufFile::from_bytes(Arc::from(full_bytes.clone().into_boxed_slice()))
            .expect("parse full gguf");
        let full_gguf_arc = Arc::new(full_gguf.clone());

        // 1. Verify RAII OpfsSyncHandle releases the exclusive lock on drop (simulating early error)
        {
            let raw_sync = write_and_open_opfs_sync(&filename, &full_bytes).await;
            let _guard = OpfsSyncHandle::new(raw_sync);
            // `_guard` drops here without explicit `.close()`
        }

        // Re-opening `createSyncAccessHandle()` on the same file must succeed immediately;
        // without the RAII Drop guard it fails with InvalidStateError / NoModificationAllowedError.
        let raw_sync = write_and_open_opfs_sync(&filename, &full_bytes).await;
        let sync = OpfsSyncHandle::new(raw_sync);

        // 2. Build a header-only GgufFile containing ONLY the metadata header (no tensor bytes!),
        // so `MmapWeight::from_gguf(&header_only_gguf, "token_embd.weight")` would fail.
        let header_len = full_gguf.data_offset();
        let header_bytes: Arc<[u8]> = Arc::from(&full_bytes[..header_len]);
        let header_only_gguf =
            Arc::new(GgufFile::from_header_bytes(header_bytes, full_bytes.len() as u64).unwrap());

        let opfs_source = OpfsGpuWeightSource::new(
            sync,
            Arc::clone(&header_only_gguf),
            128,
            format!("opfs-{arch}"),
        )
        .expect("OpfsGpuWeightSource::new must succeed on header-only GGUF");

        if arch == "lfm2" {
            let cpu_full = LfmModel::from_gguf(full_gguf, 128).expect("LfmModel::from_gguf");
            assert_weight_source_parity(&opfs_source, &cpu_full, &header_only_gguf, &full_gguf_arc);

            // Compile `GpuPipelines` once for `"lfm2"` (avoiding 5x redundant LLVM JIT
            // shader compiles on software Vulkan `lavapipe` in CI) and verify that the
            // WebGPU forward pass matches the full in-memory reference model.
            let gpu_from_opfs = GpuLfmModel::from_weight_source_with_ctx(
                &opfs_source,
                128,
                format!("opfs-{arch}"),
                ctx.clone(),
            )
            .expect(
                "GpuLfmModel::from_weight_source_with_ctx must succeed from OpfsGpuWeightSource",
            );
            opfs_source.close();

            let mut state_opfs = InferenceState::from_config(gpu_from_opfs.config()).unwrap();
            let mut state_cpu = InferenceState::from_config(Model::config(&cpu_full)).unwrap();

            gpu_from_opfs.forward_prefill_step(1, 0, &mut state_opfs);
            cpu_full.forward_prefill(&[1], 0, &mut state_cpu);

            let logits_opfs = gpu_from_opfs
                .forward_logits_async(2, 1, &mut state_opfs)
                .await
                .expect("forward_logits_async opfs");
            let logits_cpu = cpu_full.forward(&[2], 1, &mut state_cpu);

            assert!(
                logits_opfs.iter().all(|v| v.is_finite()) && logits_opfs.iter().any(|&v| v != 0.0),
                "WebGPU forward pass must produce non-zero finite logits for {arch} (catches WGSL pipeline validation errors)"
            );
            assert_eq!(logits_opfs.len(), logits_cpu.len());
            for (i, (g, c)) in logits_opfs.iter().zip(logits_cpu.iter()).enumerate() {
                assert!(
                    (g - c).abs() < 1e-2,
                    "OPFS-streamed WebGPU logit[{i}] ({g}) diverged from full in-memory reference ({c})"
                );
            }
        } else {
            let cpu_full = LlamaModel::from_gguf(full_gguf, 128).expect("LlamaModel::from_gguf");
            assert_weight_source_parity(&opfs_source, &cpu_full, &header_only_gguf, &full_gguf_arc);
            opfs_source.close();
        }
    }

    // 3. Verify `read_gguf_header_from_sync_handle` escalates past the initial 4 MiB probe
    // when a GGUF header exceeds 4 MiB (e.g. LFM2.5-2.6B at 7.84 MiB).
    let mut big_header_gguf = Vec::new();
    big_header_gguf.extend_from_slice(b"GGUF");
    big_header_gguf.extend_from_slice(&3u32.to_le_bytes());
    big_header_gguf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    big_header_gguf.extend_from_slice(&6u64.to_le_bytes()); // 6 KV entries (~4.8 MB total)
    let filler = "x".repeat(800_000);
    for i in 0..6 {
        let key = format!("test.padding.{i}");
        big_header_gguf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        big_header_gguf.extend_from_slice(key.as_bytes());
        big_header_gguf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
        big_header_gguf.extend_from_slice(&(filler.len() as u64).to_le_bytes());
        big_header_gguf.extend_from_slice(filler.as_bytes());
    }
    assert!(big_header_gguf.len() > 4 * 1024 * 1024);
    let raw_sync = write_and_open_opfs_sync("test-big-header.gguf", &big_header_gguf).await;
    let sync = OpfsSyncHandle::new(raw_sync);
    let parsed = cera_wasm::bundle::read_gguf_header_from_sync_handle(sync.as_raw())
        .expect("read_gguf_header_from_sync_handle must parse >4 MiB GGUF headers");
    assert_eq!(
        parsed.get_str("test.padding.0").map(str::len),
        Some(800_000)
    );
    sync.close();

    // 4. Verify `OpfsGpuWeightSource` fails closed when the OPFS file is truncated mid-load
    // rather than silently uploading empty/zeroed weights.
    let full_bytes = build_minimal_gguf_bytes("lfm2");
    let full_gguf = GgufFile::from_bytes(Arc::from(full_bytes.clone().into_boxed_slice())).unwrap();
    let header_len = full_gguf.data_offset();
    let header_bytes: Arc<[u8]> = Arc::from(&full_bytes[..header_len]);
    let header_only_gguf =
        Arc::new(GgufFile::from_header_bytes(header_bytes, full_bytes.len() as u64).unwrap());
    let raw_sync = write_and_open_opfs_sync("test-truncated-opfs.gguf", &full_bytes).await;
    let raw_sync_for_trunc = raw_sync.clone();
    let sync = OpfsSyncHandle::new(raw_sync);
    let opfs_source = OpfsGpuWeightSource::new(
        sync,
        Arc::clone(&header_only_gguf),
        128,
        "opfs-truncated".to_string(),
    )
    .expect("OpfsGpuWeightSource::new succeeds before truncation");
    // Truncate the OPFS file down to the header so 2D weight reads come back short.
    raw_sync_for_trunc
        .truncate_with_u32(header_len as u32)
        .expect("truncate to header");
    let err = match GpuLfmModel::from_weight_source_with_ctx(
        &opfs_source,
        128,
        "opfs-truncated".to_string(),
        ctx,
    ) {
        Ok(_) => panic!("truncated OPFS file must fail GpuLfmModel::from_weight_source_with_ctx"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("short read from OPFS"),
        "expected short read from OPFS error, got: {err}"
    );
    opfs_source.close();
}

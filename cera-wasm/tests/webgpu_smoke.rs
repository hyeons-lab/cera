//! Headless-Chrome smoke test for the async WebGPU host layer.
//!
//! Proves the wasm/WebGPU async primitives — `GpuContext::new_async` (async
//! adapter/device request) and `download_f32_async` (async `map_async`
//! readback) — actually initialize and round-trip data on real browser
//! WebGPU, not just compile for wasm32. This is the CI-able gate for the
//! browser GPU path; the full LFM2 generation is exercised manually via the
//! `examples/webgpu` page on a real model (a real GGUF is too large to embed
//! in an automated wasm test). See devlog 000169.
//!
//! Run: `wasm-pack test --headless --chrome --features wgpu` (the
//! `cera-wasm/webdriver.json` capabilities enable WebGPU in headless Chrome).
//! Requires a WebGPU-capable Chrome + matching chromedriver.

#![cfg(all(target_arch = "wasm32", feature = "wgpu"))]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn webgpu_async_init_and_readback() {
    // Async adapter + device request. Succeeding here is itself the proof
    // that the async init path works on real browser WebGPU. (The adapter
    // *name* is intentionally not asserted — headless Chrome's WebGPU/Dawn
    // adapter often reports an empty `name`, which is not an error.)
    let ctx = cera::backend::wgpu::GpuContext::new_async()
        .await
        .expect("WebGPU device init (new_async) failed — is WebGPU enabled?");

    // Upload → async readback round-trip on real browser WebGPU. Odd length
    // exercises the same path the engine uses for logits readback.
    let data: Vec<f32> = (0..257).map(|i| i as f32 * 0.5).collect();
    let buf = ctx.upload_f32(&data, "smoke");
    let out = ctx.download_f32_async(&buf, data.len()).await;
    assert_eq!(data, out, "async readback mismatch");
}

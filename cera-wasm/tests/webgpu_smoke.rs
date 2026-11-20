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
//! Requires a WebGPU-capable Chrome + matching chromedriver. That file's
//! flags are macOS-specific; CI points the runner at `webdriver-linux.json`
//! through `WASM_BINDGEN_TEST_WEBDRIVER_JSON`, since a software adapter needs
//! different flags than a Metal one.

#![cfg(all(target_arch = "wasm32", feature = "wgpu"))]

use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Yield one browser microtask turn followed by a short `setTimeout` macrotask turn so
/// WebGPU implementations (Chrome/Dawn) can settle `GPUDevice.lost` and invoke wgpu's
/// `set_device_lost_callback`. Uses `js_sys::global()` so it works without extra `web-sys`
/// features in both Window and Worker contexts.
async fn yield_browser_tick() {
    let _ = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(
        &wasm_bindgen::JsValue::UNDEFINED,
    ))
    .await;
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        if let Ok(set_timeout) =
            js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout"))
            && let Some(func) = set_timeout.dyn_ref::<js_sys::Function>()
        {
            let resolve_val: wasm_bindgen::JsValue = resolve.clone().into();
            let delay = wasm_bindgen::JsValue::from_f64(10.0);
            if func.call2(&global, &resolve_val, &delay).is_ok() {
                return;
            }
        }
        let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

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
    let out = ctx
        .download_f32_async(&buf, data.len())
        .await
        .expect("async readback failed");
    assert_eq!(data, out, "async readback mismatch");
}

#[wasm_bindgen_test]
async fn checked_pipeline_creation_is_unchecked_on_wasm32() {
    let ctx = cera::backend::wgpu::GpuContext::new_async()
        .await
        .expect("WebGPU device init (new_async) failed: is WebGPU enabled?");

    let trivial = "@compute @workgroup_size(1) fn main() {}";
    assert!(
        ctx.try_create_pipeline_checked(trivial, "main", "trivial_probe")
            .is_ok(),
        "valid shader must succeed on wasm32"
    );
    // On wasm32, popping error scopes needs a blocking executor that browser
    // WebGPU lacks, so `check_pipelines` skips error scopes there and returns `Ok`
    // while still checking pre-existing device loss recorded via `GPUDevice.lost`.
    assert!(
        ctx.try_create_pipeline_checked(trivial, "main_missing", "bad_entry_probe")
            .is_ok(),
        "unchecked wasm32 pipeline build must return Ok on invalid entry point"
    );
    assert!(
        ctx.try_create_pipeline_checked(
            "@compute @workgroup_size(1) fn main() { let x: u32 = ; }",
            "main",
            "bad_module_probe",
        )
        .is_ok(),
        "unchecked wasm32 pipeline build must return Ok on invalid WGSL"
    );

    // Destroying the browser GPUDevice must fire `set_device_lost_callback`,
    // set `ctx.is_device_lost()`, and reject subsequent checked pipeline builds.
    ctx.device.destroy();
    for _ in 0..200 {
        if ctx.is_device_lost() {
            break;
        }
        yield_browser_tick().await;
    }
    assert!(
        ctx.is_device_lost(),
        "device_lost callback must set ctx.is_device_lost() after device.destroy()"
    );
    let err = ctx
        .try_create_pipeline_checked(trivial, "main", "post_destroy_probe")
        .expect_err("checked pipeline creation must fail once GPUDevice is destroyed");
    let msg = format!("{err:#}");
    assert!(msg.contains("GPU device lost"), "{msg}");
    assert!(msg.contains("post_destroy_probe"), "{msg}");
}

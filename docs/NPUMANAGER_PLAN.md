# NPU Manager + Android App Integration Plan

Status: plan (no code yet). Target: Android 17+ devices where
`com.android.npumanager` is present; Android ≤16 keeps the direct-driver
path (current CLI behavior).

## Background: what NPU Manager is

Android 17's NPU Manager (`com.android.npumanager` APEX, `Context.NPU_SERVICE`)
is an arbitration + buffer service, **not** an inference runtime:

- Admission control: `canLoadModel()` answers `CAN_LOAD_NOW` /
  `WAIT_FOR_UNLOAD` / `NOT_PRIORITIZED` over an async callback; the app
  must honestly report `notifyModelLoaded/Unloaded`. Policies:
  StatusQuo / Budget / TurnTaking.
- Protected buffers: the Rust NDK (`ANpuBuffer` in
  `libcom.android.npumanager.so`, `__INTRODUCED_IN(37)`) allocates
  `MODEL_EXECUTABLE` / `MODEL_WEIGHTS` / `CACHE` / `AUXILIARY` buffers
  backed by `/dev/wrapfd` dma-bufs, with async alloc, `map`/`unmap`,
  `loadAsync`, `setPriority`, and **preemption callbacks** (a preempted
  buffer's next `map` fails with `ENOENT`).
- Vendor HAL `android.hardware.npu` v1 is priority/observation only
  (`IScheduling` + `ISchedulingCallback`); compute still goes through
  the vendor driver (for us: FastRPC/HTP).

Implication for cera: our DSP submission path does not change. NPU Manager
integration is about (1) asking before loading, (2) allocating weights/KV
from the manager's heap instead of raw rpcmem, and (3) surviving eviction.

## Engine-level support (cera + cera-ffi)

### 1. Buffer backend abstraction (required)

`RpcmemBuffer` is currently the only device-memory type. Introduce a
`DeviceBuffer` trait (map/unmap as f32 bytes, DSP-visible address/fd,
cache clean/invalidate) with two implementations:

- `RpcmemBuffer` (existing): Android ≤16 and non-managed fallback.
- `NpuBuffer`: wraps `ANpuBuffer` via the NDK C ABI (lazy `dlopen` of
  `libcom.android.npumanager.so`, same pattern as the NDK shim —
  keeps the `cdylib` loadable where the APEX is absent).

Buffer-type mapping: weights slab → `MODEL_WEIGHTS`, KV + conv state →
`CACHE`, scratch → `AUXILIARY` (scratch is the best preemption canary:
it is rewritten every forward, so losing it is harmless if the engine
re-allocates per session).

### 2. Zero-copy requirement (the CLI-parity crux)

CLI perf must be reproducible in-app. Risks, in order:

1. **DSP import of wrapfd dma-bufs.** Our HTP path submits rpcmem
   pointers. If the DSP/FastRPC stack can import the manager's dma-buf
   fds into the same SMMU mapping (no copy), parity is achievable.
   If it forces a copy into rpcmem, every weight/KV upload pays
   memcpy + double memory. SPIKE FIRST: allocate one `ANpuBuffer`,
   import into a FastRPC session, submit a 1-op batch, compare
   against the rpcmem path. No-go criterion: >2% steady-state gap.
2. **ADPF + oppoll preserved.** Our decode tuning (ADPF session hints,
   non-blocking completion polling) lives below the buffer layer and
   is unaffected — but must be re-verified in-app (app power profile
   differs from `adb shell`; see 5).
3. **No extra Java copies.** Tokenizer + sampling stay in Rust via
   the existing UniFFI/JNI surface; the app passes prompts in and
   streams tokens out. No per-token JNI array copies in the hot loop.

### 3. Admission + lifecycle (required for good citizenship)

- Before `HexagonLfm2Model::load_weights`: `canLoadModel()` with the
  GGUF size bucket (`LESS_THAN_1GB` for 350M; 8B-A1B is
  `BETWEEN_1GB_AND_2GB`). `WAIT_FOR_UNLOAD` → surface "waiting for
  NPU" to the app (do not spin); `NOT_PRIORITIZED` → fall back to
  CPU/GPU engine automatically (the engine already multi-backends).
- After weights resident: `notifyModelLoaded()`; on engine drop /
  `onRequestUnloadModel`: release DSP session + buffers, then
  `notifyModelUnloaded()`.
- Preemption (`onNotifyPreempted` → buffer `Gone`): the engine must
  treat mapped-pointer access after preemption as fatal for that
  session and rebuild (re-alloc + reload weights + cold KV). Design
  the session so rebuild reuses the normal load path (~seconds, rare).

### 4. Packaging (cera-ffi, Android)

- `cera-ffi` already builds `cdylib`/`staticlib`. Ship an AAR
  (`libcera_ffi.so` per ABI + UniFFI-generated Kotlin) via
  `uniffi-bindgen --language kotlin`. JNI stays inside the UniFFI
  runtime (no hand-written JNI).
- Feature-gate: `cera/hexagon` on, `npumanager` as a separate cargo
  feature (default off until the §2 spike passes; default on after).
- The NDK API level for `ANpuBuffer` is 37: guard all manager calls
  behind an API-level + `dlopen` check so one AAR runs on 16 (direct)
  and 17+ (managed).

### 5. App-side integration (sample app)

- Foreground service + `HIGH_PERFORMANCE` power hint while generating
  (matches the CLI's ADPF posture; background apps get demoted CPUs
  and lose NPU priority under Budget/TurnTaking).
- Model store: download GGUF to app storage once; pass the fd to
  `setFileSegmentToLoad`/`loadAsync` so weights stream manager-side.
- UX states: `LOADING` (admission wait), `READY`, `PREEMPTED`
  (reloading), `FALLBACK_CPU` (not prioritized). Never block the UI
  thread on `canLoadModel`.

## Validation gates (each must pass before calling it done)

1. §2 spike: 512-prefill + decode bench, managed vs direct buffers,
   same device: within noise (≤2%).
2. Admission matrix on Android 17 (Cuttlefish ok): Budget policy with
   two loaders (second gets WAIT/NOT_PRIORITIZED, first keeps running);
   TurnTaking unload round-trip; preemption reload recovers.
3. Missing-manager fallback: API 36 device runs the direct path with
   zero behavior change (existing determinism suite passes).
4. Thermal/perf: in-app foreground decode within 5% of CLI on the same
   device/cooldown (controls for the app power profile).

## Open questions

- Does this device fleet's FastRPC accept wrapfd dma-bufs, or only
  Ion/rpcmem heaps? (§2 spike answers.)
- Which policy do Pixel/Samsung 17 builds ship as default?
- `canAttributeOtherUid`: needed if inference runs in a `:npu`
  isolated process (recommended for 1GB+ weight slabs).

## Sources

- AOSP `packages/modules/NpuManager` (service, framework, NDK, HAL
  `android.hardware.npu` v1), via the aosp-internal-book ch. 53 summary.
- HAL note: vendors keep executing work through their own SDK; the
  HAL/manager only arbitrate priority and buffers.

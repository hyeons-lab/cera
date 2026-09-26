# Android NPU Packaging

How the Hexagon backend ships in published artifacts, which devices it
runs on, and what blocks the rest.

## What ships where

| artifact | NPU content | notes |
|---|---|---|
| `cera` (crates.io) | `hexagon` feature (off by default) + 4 embedded DSP skels (~3.2 MB) | `cargo package` includes `src/backend/hexagon/skels/` (no `package.include` filter) |
| `cera-ffi-android` AAR | hexagon on arm64-v8a + x86_64; lean on 32-bit; skels embedded in libcera_ffi.so | FFI surface identical everywhere; `hexagon_probe()` reports unavailable where off |
| AAR `assets/NOTICE` | MIT attribution for the embedded skels | required: MIT covers binaries too |
| `cera-ffi-jvm` / xcframework / npm / crates | unchanged (no hexagon) | host platforms have no DSP |

Recipes: `just android-libs` (split 64/32 build), `just bindings`
(Kotlin/Swift/Python regen: the AAR consumes
`cera-ffi/bindings/kotlin` directly, no copy step), publish pipeline
(`publish.yml`) calls `just android-libs`.

## App integration (Kotlin)

```kotlin
import com.hyeonslab.cera.android.HexagonNpu
import uniffi.cera_ffi.*

// Once at startup, on the main thread, before loading a model:
HexagonNpu.setup(context) // extracts skels to noBackupFilesDir, sets ADSP_LIBRARY_PATH

// Gate NPU use on a live probe (fast: one driver open + hwinfo):
val backend = try {
    val p = hexagonProbe() // { arch, threads, hvxUnits, hmxUnits, vtcmBytes }
    Log.i("npu", "Hexagon ${p.arch} hmx=${p.hmxUnits}")
    BackendPreference.HEXAGON
} catch (e: Exception) {
    BackendPreference.CPU // or GPU: no NPU in this process
}
```

Two packaging requirements make the stock tier work (all verified on
a retail S25 Ultra, SELinux enforcing, via a normally installed APK):

1. **Manifest**: `<uses-native-library android:name="libcdsprpc.so"
   android:required="false"/>` inside `<application>`: grants the app
   linker namespace access to the vendor FastRPC client (on the vendor
   public list). The AAR manifest carries this and it merges into
   consumers automatically; `required=false` so devices without the lib
   still install (the probe then reports unavailable).
2. **Embedded skels & `ADSP_LIBRARY_PATH`**: `libcera_ffi.so` embeds the
   four DSP skels in `.rodata` via `include_bytes!`. At startup,
   `HexagonNpu.setup(context)` writes them to `context.noBackupFilesDir/cera_skels`
   and points FastRPC's loader at that directory plus vendor fallback paths.
   Because skels are not packaged as host `.so` files in `jniLibs/`, all
   libraries in the AAR remain 16KB-page-aligned (`0x4000`) and apps do
   not require `android:extractNativeLibs="true"`.

`hexagonInstallSkels` (write embedded skels into a caller-staged dir)
underlies `HexagonNpu.setup` and remains available for JVM/desktop/shell
flows. The `probe-app` module is a runnable reference and the on-device
gate: `./gradlew :probe-app:installDebug`, launch from the launcher,
`adb logcat -s CeraProbe`. It also reports the access route (`direct` vs
`hal-fallback`, see below), since DSP policy varies per OEM/SoC/firmware;
that route string is the datum to record when validating a new device.

## Device support matrix

Skels ship for v73/v75/v79/v81 (llama.cpp upstream scope; the probe
tries all four and uses the first that opens).

| SoC (examples) | arch | status |
|---|---|---|
| 8 Elite, X Elite Gen 2-class | v79/v81 | ✅ validated on-device (S25 Ultra) |
| 8 Gen 3 / 8s Gen 3 | v75 | ✅ skel ships, same code path (not yet run) |
| 8 Gen 2 / 8+ Gen 1 / 7+ Gen 2 / X Elite | v73 | ✅ skel ships, same code path (not yet run) |
| 8 Gen 1 / 7 Gen 1 | v69 | ❌ no skel (outside llama upstream scope) |
| 888 / 888+ / 870 / 778G+ | v68 | ❌ no skel (outside llama upstream scope) |
| 865 and older (v66-) | — | ❌ no HTP AI-stack path in this codebase |

"Support all Snapdragons": **v73+ (2022 flagships onward) yes** with
what ships today; **v68/v69 needs skel builds** (`-DDSP_VERSION=v68`
from the same pinned sources) **plus** a device to validate on — the
host side already handles missing HMX (`n_hmx == 0` → HVX kernels), so
the risk is bounded to worker/arch quirks; **v66- has no path** (no
skel exists) and stays on CPU/GPU fallback. Non-Snapdragon devices:
`dlopen` fails cleanly → CPU fallback, no crash (same for 32-bit ABIs).

## Deployment tiers (OS policy, measured)

| tier | NPU works? | evidence |
|---|---|---|
| adb shell / dev tools / benchmarks | ✅ yes | all NPU validation runs as `shell` |
| system / priv-app / OEM preload / rooted / eng builds | ✅ expected | same UID class as shell-or-better; probe app reports the exact failure if not |
| stock third-party APK (Play install) | ✅ yes, with the packaging above | installed-APK probe on S25U (SM8750, Android 16): unsigned PD up, inference running |

How the stock tier works: the app cannot open `/dev/fastrpc-cdsp`
itself, but `libcdsprpc.so` falls back to Qualcomm's DSP service,
which opens the node and passes the fd back over binder; the session
then creates an unsigned user PD as usual. No cera code is involved in
the fallback — it is Qualcomm's own `open_device_node` logic — but it
rests on three device grants, all present on the S25 Ultra:

- `allow appdomain vendor_qdsp_device (chr_file (ioctl read))`: apps
  may ioctl an already-open DSP fd (not `open` it).
- `vendor_hal_dspmanager_client` includes `untrusted_app` (and
  `untrusted_app_25..32`, `isolated_compute_app`, ...): apps may find
  and call `vendor.qti.hardware.dsp.IDspService/default`. Note the
  grant lives in system_ext/product policy, not vendor — grep every
  partition's `.cil` before judging a new device.
- `libcdsprpc.so` is on the vendor public list
  (`/vendor/etc/public.libraries.txt`), so the manifest entry above
  makes it loadable.

Logcat markers of the working path: `open thru HAL` followed by
`Created user PD ... Unsigned:Y`, with no `avc: denied` lines.

Earlier "blocked" verdicts were false negatives from three
measurement errors (recorded here so they are not repeated):

1. Judging app access from `run-as`: `runas_app` is NOT a DSP-HAL
   client while `untrusted_app` IS, so a `run-as` denial proves
   nothing about Play apps. Only a real installed APK counts — never
   `run-as`, never `service check` from a shell domain.
2. Missing `<uses-native-library>`: without it the soname `dlopen`
   fails; with it, it resolves. (Absolute `/vendor/lib64` paths stay
   blocked by the linker namespace either way; the driver tries the
   soname first and treats absolute-path misses as non-fatal.)
3. `lshal` shows no DSP service — but `IDspService` is AIDL and `lshal`
   lists HIDL only; `service list` shows it.

Caveats that still hold: raw FastRPC device opens remain blocked for
apps (only the HAL route works); DSP policy varies per OEM/SoC/firmware
(the probe-app route string is how a new device is checked); NPU
Manager (Android 17+) arbitrates buffers/admission but grants no access
and is not needed for it.

## Verification gates (run before calling an NPU release done)

1. `just android-libs` green; arm64/x86_64 `.so` contain
   `libggml-htp-v*` strings, 32-bit don't; all four pass
   `assert-ffibuffer.sh`; all four pass `assert-16k-pages.py`.
2. AAR unzips with `jni/<4 abis>/libcera_ffi.so` (16KB page-aligned) +
   `assets/NOTICE` (byte-identical MIT block to `skels/LICENSE`);
   AAR manifest carries `<uses-native-library android:name="libcdsprpc.so">`.
3. `probe-app`, installed normally and launched from the launcher
   (never via `run-as`), reports `OK route=hal-fallback arch=V..`
   on a granting device; where the grant is absent it must fail
   cleanly (no crash). Record the route string per device.
4. Existing on-device determinism matrix still green (logits m=1..8
   x5, greedy md5 2x2x6, 256-token pair): any skel/host change can
   silently re-phase the DSP race.
5. Skel/host skew guard: after any skel rebuild, update `SOURCE.md`
   md5s + provenance and re-run gate 4.

## Open items

- v68/v69 skel builds + a loaner 888/8-Gen-1 device for validation.
- One-time confirm of the Hexagon SDK EULA redistribution clause
  (flagged in `skels/SOURCE.md`).
- v75/v73 on-device smoke (same code path as v79; want one boot each).
- OEM/SoC validation matrix: stock-APK probe route (`direct` vs
  `hal-fallback` vs clean failure) on one device per major OEM skin,
  since the DSP-service grant lives in per-OEM system_ext/product
  policy — the S25U result must not be assumed universal.

## Head-to-head

Shell-tier numbers vs llama.cpp (CPU/NPU/GPU, LFM2-VL-450M-Q4_0, S25U):
`benchmarks/BASELINE.md`, "Galaxy S25 Ultra" section. NPU prefill is the
one cell cera leads (8685 vs 8103); llama leads NPU decode, all of GPU,
and CPU decode ~2x at matched cores.

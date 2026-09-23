# Android NPU Packaging

How the Hexagon backend ships in published artifacts, which devices it
runs on, and what blocks the rest.

## What ships where

| artifact | NPU content | notes |
|---|---|---|
| `cera` (crates.io) | `hexagon` feature (off by default) + 4 embedded DSP skels (~3.2 MB) | `cargo package` includes `src/backend/hexagon/skels/` (no `package.include` filter) |
| `cera-ffi-android` AAR | hexagon on arm64-v8a + x86_64; lean on 32-bit | FFI surface identical everywhere; `hexagon_probe()` reports unavailable where off |
| AAR `assets/NOTICE` | MIT attribution for the embedded skels | required: MIT covers binaries too |
| `cera-ffi-jvm` / xcframework / npm / crates | unchanged (no hexagon) | host platforms have no DSP |

Recipes: `just android-libs` (split 64/32 build), `just bindings`
(Kotlin/Swift/Python regen — the AAR consumes
`cera-ffi/bindings/kotlin` directly, no copy step), publish pipeline
(`publish.yml`) calls `just android-libs`.

## App integration (Kotlin)

```kotlin
import uniffi.cera_ffi.*
import java.io.File

// Once at startup, before loading a model:
hexagonInstallSkels(File(filesDir, "hexagon-skels").absolutePath)

// Gate NPU use on a live probe (fast: one driver open + hwinfo):
val backend = try {
    val p = hexagonProbe() // { arch, threads, hvxUnits, hmxUnits, vtcmBytes }
    Log.i("npu", "Hexagon ${p.arch} hmx=${p.hmxUnits}")
    BackendPreference.HEXAGON
} catch (e: Exception) {
    BackendPreference.CPU // or GPU: no NPU in this process
}
```

`hexagonInstallSkels` writes the four embedded skels (only when the size
differs) and prepends the dir to `ADSP_LIBRARY_PATH`, which FastRPC's
loader honors (verified: skel in a non-CWD dir + env var → inference
works). The `probe-app` module is a runnable reference + the on-device
gate: `./gradlew :probe-app:installDebug`, launch, `adb logcat -s
CeraProbe`.

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
| stock third-party APK (Play install) | ❌ **blocked by Android** | measured, see below |

Why stock APKs are blocked (all verified on a retail S25 Ultra,
SELinux enforcing):

1. `dlopen("libcdsprpc.so")` fails: linker namespaces hide
   `/vendor/lib64` from apps (soname **and** absolute path both fail;
   the probe app reports `FastRPC driver not found`).
2. `/dev/fastrpc-cdsp` can't even be opened `O_RDONLY` as an app UID
   (`run-as` test → `Permission denied`), so a bundled FastRPC client
   over the kernel node is dead too. (The shell path works over an
   `O_RDONLY` fd + ioctl — DAC other-read suffices there.)
3. No DSP/NPU HAL service exists to proxy through (`lshal` empty).

This is uniform AOSP behavior (it is also why QNN-from-app needs OEM
cooperation on some devices), not a Samsung quirk. Consequences:

- The sanctioned app path to phone NPUs is NNAPI/QNN graph APIs, not
  FastRPC — a different backend (months, not days).
- Niches that keep our FastRPC path viable in apps: OEM/system
  integration, MDM-managed fleets, and a Shizuku-style shell-UID proxy
  (shell *can* open the node `O_RDONLY`; unverified end-to-end).
- NPU Manager (Android 17+) arbitrates buffers/admission but does not
  grant compute access — it does not unlock this tier either.

## Verification gates (run before calling an NPU release done)

1. `just android-libs` green; arm64/x86_64 `.so` contain
   `libggml-htp-v*` strings, 32-bit don't; all four pass
   `assert-ffibuffer.sh`.
2. AAR unzips with `jni/<4 abis>/libcera_ffi.so` + `assets/NOTICE`
   (byte-identical MIT block to `skels/LICENSE`).
3. `probe-app` on a privileged tier reports `OK arch=V..` (documents
   the tier it ran in); on a stock APK it must report the clean
   `FastRPC driver not found` failure (no crash).
4. Existing on-device determinism matrix still green (logits m=1..8
   x5, greedy md5 2x2x6, 256-token pair) — any skel/host change can
   silently re-phase the DSP race.
5. Skel/host skew guard: after any skel rebuild, update `SOURCE.md`
   md5s + provenance and re-run gate 4.

## Open items

- v68/v69 skel builds + a loaner 888/8-Gen-1 device for validation.
- One-time confirm of the Hexagon SDK EULA redistribution clause
  (flagged in `skels/SOURCE.md`).
- v75/v73 on-device smoke (same code path as v79; want one boot each).
- If a stock-APK story is ever required: LiteRT + Qualcomm AI Engine
  Direct delegate, via a GGUF→TFLite converter (reviewed 2026-09: LFM2's
  shortconv blocks are GEMV-dominated and delegatable; the converter +
  requant validation is the project, not the delegate wiring). Raw NNAPI
  is a poor fit for an LLM engine; raw QNN duplicates what the delegate
  already ships on Maven.

## Head-to-head

Shell-tier numbers vs llama.cpp (CPU/NPU/GPU, LFM2-VL-450M-Q4_0, S25U):
`benchmarks/BASELINE.md`, "Galaxy S25 Ultra" section. NPU prefill is the
one cell cera leads (8685 vs 8103); llama leads NPU decode, all of GPU,
and CPU decode ~2x at matched cores.

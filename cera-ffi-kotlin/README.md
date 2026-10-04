# cera-ffi-kotlin

Maven publishing for the cera inference engine's **UniFFI/JNA Kotlin bindings**,
as two artifacts under the `com.hyeons-lab` group:

| Artifact | Consumer | Native libs |
|----------|----------|-------------|
| `com.hyeons-lab:cera-ffi-jvm`     | Desktop JVM | bundled in `resources/<jna-prefix>/` (macOS arm64, Linux x64, Windows x64) |
| `com.hyeons-lab:cera-ffi-android` | Android     | `jniLibs/<abi>/` (arm64-v8a, armeabi-v7a, x86_64, x86) |

Both compile the **same vendored binding**: `cera-ffi/bindings/kotlin/cera_ffi.kt`,
wired in via `srcDir` (no copy) so it never drifts from `just bindings`. JNA loads
the `cera_ffi` native library from the classpath at runtime.

Naming follows the decoupled convention (mirrors the prism repo): the Maven
**groupId** is the hyphenated, Central-verified `com.hyeons-lab`, while the Android
**namespace / package** identifiers stay un-hyphenated (`com.hyeonslab.cera.*`),
since Java/Kotlin package syntax forbids dashes.

## Build & publish

Native libraries are **not** committed; they're built by cargo / cargo-ndk and
staged before packaging.

```bash
# Local desktop-JVM smoke test (host platform only):
just jvm-libs-host                                    # build + stage the host .dylib/.so
cd cera-ffi-kotlin
JAVA_HOME=<jdk21> ./gradlew :cera-ffi-jvm:publishToMavenLocal

# Android jniLibs (needs cargo-ndk + NDK):
just android-libs
```

CI (the `jvm` leg of `.github/workflows/publish.yml`, manual `workflow_dispatch`) cross-builds
the native libs per runner (macOS/Linux/Windows + Android NDK), then publishes
`cera-ffi` to Maven Central via the vanniktech plugin. The version (`VERSION_NAME`
in `gradle.properties`) tracks the Cargo workspace version, so the Kotlin/Android
artifacts release under the **same** version as the crates.io and npm artifacts,
e.g. `0.4.0` everywhere.

- A release version (no `-SNAPSHOT`) is a **real** Maven Central release and is
  **GPG-signed** (`signAllPublications()`), so it needs both the Central Portal
  token secrets `MAVEN_CENTRAL_USERNAME` / `MAVEN_CENTRAL_PASSWORD` **and** the
  signing secrets `MAVEN_SIGNING_KEY` (ASCII-armored private key) /
  `MAVEN_SIGNING_PASSWORD` (its passphrase). The real run uses
  `publishAndReleaseToMavenCentral` (uploads, signs, and auto-releases the
  deployment).
- A `-SNAPSHOT` version instead routes to the snapshot repo and skips signing.
- Run with `dry_run = true` first; it publishes a `-SNAPSHOT` to your local
  Maven repo (`publishToMavenLocal`), needing no token or signing key.

Versions (kotlin, AGP, vanniktech, compile/minSdk) are pinned in
`gradle/libs.versions.toml`; publishing coordinates + POM in `gradle.properties`.

## Android Model Downloading & Foreground Service

`cera-ffi-android` includes idiomatic Android utilities under `com.hyeonslab.cera.android.download`:

- **`AndroidBundleRepo`**: Companion helpers configured for Android persistent storage (`filesDir/cera-bundles`, never OS-purgeable `cacheDir`).
- **`AndroidModelDownloader`**: Coroutine and Kotlin `Flow<DownloadState>` downloader that favors `CeraDownloadService` by default.
- **`CeraDownloadService`**: Foreground service managing long-running model downloads with live progress notifications and process-death resilience.
- **Resumable Downloads**: Interrupted downloads automatically resume via HTTP range requests from existing bytes in `<dest>.partial`.

```kotlin
import com.hyeonslab.cera.android.download.*
import uniffi.cera_ffi.*

// 1. One-line coroutine download + engine initialization:
// By default (useService = true), downloading runs in CeraDownloadService with live notifications.
val downloader = AndroidModelDownloader(context)
val engine = downloader.downloadAndLoad(
    bundleId = "LFM2-1.2B-GGUF",
    quant = "Q4_0"
) { bytesDownloaded, totalBytes, percent ->
    println("Progress: $percent%")
}

// 2. Cold Flow streaming state:
// Emits Connecting, Progress, Success, and Error states from the foreground service.
// Set useService = false to download directly in-coroutine without system notifications.
downloader.download("LFM2-1.2B-GGUF", "Q4_0").collect { state ->
    when (state) {
        is DownloadState.Connecting -> showSpinner()
        is DownloadState.Progress -> updateProgressBar(state.percent ?: 0)
        is DownloadState.Success -> onReady()
        is DownloadState.Error -> showError(state.message)
        else -> Unit
    }
}

// 3. Or trigger via AndroidBundleRepo convenience functions:
AndroidBundleRepo.download(context, "LFM2-1.2B-GGUF", "Q4_0").collect { ... }
```

## Always-on transcription service (probe-app)

`probe-app` contains a reference foreground service, `AudioPipelineService`, that feeds the
microphone to a cera `AudioPipeline` (VAD, Whisper, Sortformer speaker diarizer, optional wake
word). In a build with the `hexagon` feature every stage runs on the NPU, which Android does not
demote when the app is in the background (it does demote background CPU work).

- A microphone foreground service (`foregroundServiceType="microphone"`). Android 14+ only starts
  one from a visible activity with `RECORD_AUDIO` already granted, so `AudioServiceActivity` starts
  it; the service does not restart itself after the system kills it (a restart from the background
  would be refused), so reopen the app.
- Models are read from `audio-models/` under the app's `filesDir` (or the external files dir):
  `vad.gguf` (required), `whisper.gguf`, `diarizer.gguf`, `hotword.gguf`. Whisper must be Q8_0 or
  Q4_0 and the diarizer a `--tail-outtype q8_0` GGUF to run on the NPU. The startup log line names
  each stage and its file size (`pipeline ready: vad + whisper (82 MB) + diarizer (127 MB); ...`),
  which is how to tell which Whisper is loaded.
- **Use Whisper `base` at Q8_0 (recommended).** Make the file with
  `cera transcribe --model base --quant q8_0 --download-model` (it lands in
  `~/.cache/cera/huggingface.co/openai/whisper-base/quantized/Q8_0/model.gguf`) and push it as
  `whisper.gguf`.

  | Whisper (Q8_0) | Size | Time for 61 s of dense speech | CPU per audio second | Notes |
  |----------------|------|-------------------------------|----------------------|-------|
  | tiny | 44 MB | 17 s | 0.0093 | mishears words and invents fragments on noise |
  | **base** | 82 MB | 27 s | 0.0099 | clean on live speech; the default to use |
  | small | 265 MB | 73 s | 0.0115 | most accurate and consistent, but **slower than real time** on continuous speech |

  Measured on a Galaxy S25 Ultra by replaying a 61 s clip (the same 15 s passage four times, in
  English and Japanese) through the service at 4x. `small` got the Japanese phrase right where
  `tiny` and `base` did not and gave identical English transcripts on all four repeats, but at
  about 3.5 s per utterance it cannot keep up with continuous talking, and the capture buffer is
  only 2 s, so a live microphone would drop audio while it transcribes. It suits sparse speech.
  `medium` and `large` are larger still and were not tried.
- Transcripts, with speaker labels, are appended to `filesDir/transcript.jsonl` (one `transcript`
  record when Whisper finishes an utterance, then an `utterance` record with its speaker once the
  diarizer has covered it, about 15 s later). Whisper's bracketed non-speech tags such as
  `[BLANK_AUDIO]` are dropped.
- It logs CPU seconds per audio second every minute, with the input peak and whether Android is
  silencing the recorder: `adb logcat -s CeraAudio`.
- Intent extras (on `AudioServiceActivity`, forwarded to the service):

  | Extra | Default | Meaning |
  |-------|---------|---------|
  | `autostart` (activity only) | false | start the service once the permissions are granted |
  | `wake_lock` | true | hold a partial wake lock while running |
  | `require_hotword` | false | wait for the wake word before transcribing |
  | `chunk_ms` | 500 | audio per pipeline call, clamped to 100..2000 |
  | `wav` | none | replay a 16 kHz mono 16-bit WAV instead of the microphone, then stop |
  | `wav_speed` | 1.0 | replay rate for `wav` (0 = as fast as possible) |

Two things matter for the CPU numbers the service is built to minimise:

- **Measure with the `field` build** (`./gradlew :probe-app:assembleField`), not `debug`. A
  debuggable app runs its managed code in a deoptimizable interpreter, which made the service look
  about 1.7x more expensive. The `field` build is non-debuggable but `profileable`, so
  `simpleperf record --app` still works; push models to the external files dir
  (`/sdcard/Android/data/com.hyeonslab.cera.probe/files/audio-models/`) because `run-as` needs a
  debuggable app.
- **Feed it PCM16 bytes and big chunks.** `processChunkPcm16` takes the bytes `AudioRecord`
  delivers; `processChunk` lowers a `List<Float>` element by element (79% of the service's CPU).
  And every FFI call pays a fixed JNA cost for its call-status and buffer structures, so 500 ms
  chunks cost a fifth of what 100 ms chunks do.

Measured on a Galaxy S25 Ultra (VAD, Whisper tiny Q8_0 and the Sortformer diarizer, all on the
NPU), CPU seconds per second of audio, whole process:

| Configuration | CPU-s per audio-s |
|---------------|-------------------|
| debug build, `processChunk` floats, 100 ms chunks | 0.052 |
| debug build, `processChunkPcm16`, 100 ms | 0.035 |
| `field` build, 100 ms | 0.020 |
| `field` build, 500 ms (default), quiet room | 0.0095 |
| same, backgrounded with the screen off | 0.0093 to 0.0107 |
| same, replaying 61 s of speech (Whisper and diarizer active) | 0.0112 |

That is about 34 to 40 CPU-seconds per hour of audio, about 1% of one core.

The pure parts (model discovery, PCM conversion, the capture loop, event formatting, CPU metering)
have JVM unit tests: `./gradlew :probe-app:testDebugUnitTest`.

## Hexagon NPU (Android)

`cera-ffi-android` can run inference on Qualcomm Hexagon NPUs from a
normally installed app (no root/setup). `libcera_ffi.so` embeds the
prebuilt DSP skeletons directly; at startup, `HexagonNpu.setup(context)`
extracts them into `context.noBackupFilesDir/cera_skels` and sets
`ADSP_LIBRARY_PATH`. Because skeletons are extracted at runtime rather
than packaged as host shared libraries in `jniLibs/`, all libraries
in the published AAR remain 16KB-page-aligned (`0x4000`) and apps do
not require `android:extractNativeLibs="true"`. The AAR manifest merges
`<uses-native-library android:name="libcdsprpc.so" android:required="false"/>`
into consumers automatically.

```kotlin
import com.hyeonslab.cera.android.HexagonNpu
import uniffi.cera_ffi.*

// Once at startup, on the main thread:
HexagonNpu.setup(context) // verifies skels, sets ADSP_LIBRARY_PATH

// Gate NPU use on a live probe:
val backend = try {
    val p = hexagonProbe()
    Log.i("npu", "Hexagon ${p.arch} hmx=${p.hmxUnits}")
    BackendPreference.HEXAGON
} catch (e: Exception) {
    BackendPreference.CPU // no NPU in this process
}
```

`HexagonNpu.setup` fails fast when the skels are missing (packaging
bug on arm64-v8a) or the ABI ships none (x86_64 Android has no Hexagon
DSP; treat the NPU as unavailable). DSP policy varies per
OEM/SoC/firmware; `HexagonNpu.hasDirectNodeAccess()` reports which
access route the device uses. Details, the support matrix, and the
release gates live in `docs/ANDROID_NPU_PACKAGING.md`; the `probe-app`
module is the runnable on-device reference.

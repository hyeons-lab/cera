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

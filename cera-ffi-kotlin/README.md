# cera-ffi-kotlin

Maven publishing for the cera inference engine's **UniFFI/JNA Kotlin bindings**,
as two artifacts under the `com.hyeons-lab` group:

| Artifact | Consumer | Native libs |
|----------|----------|-------------|
| `com.hyeons-lab:cera-ffi-jvm`     | Desktop JVM | bundled in `resources/<jna-prefix>/` (macOS arm64, Linux x64, Windows x64) |
| `com.hyeons-lab:cera-ffi-android` | Android     | `jniLibs/<abi>/` (arm64-v8a, armeabi-v7a, x86_64, x86) |

Both compile the **same vendored binding** — `cera-ffi/bindings/kotlin/cera_ffi.kt`,
wired in via `srcDir` (no copy) so it never drifts from `just bindings`. JNA loads
the `cera_ffi` native library from the classpath at runtime.

Naming follows the decoupled convention (mirrors the prism repo): the Maven
**groupId** is the hyphenated, Central-verified `com.hyeons-lab`, while the Android
**namespace / package** identifiers stay un-hyphenated (`com.hyeonslab.cera.*`),
since Java/Kotlin package syntax forbids dashes.

## Build & publish

Native libraries are **not** committed — they're built by cargo / cargo-ndk and
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
`0.1.0-SNAPSHOT` to the Maven Central **snapshot** repo via the vanniktech plugin.

- SNAPSHOT publishing needs **no GPG signing** — only the Central Portal token
  secrets `MAVEN_CENTRAL_USERNAME` / `MAVEN_CENTRAL_PASSWORD`.
- Run with `dry_run = true` first (`publishToMavenLocal`, no upload).

Versions (kotlin, AGP, vanniktech, compile/minSdk) are pinned in
`gradle/libs.versions.toml`; publishing coordinates + POM in `gradle.properties`.

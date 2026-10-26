plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.maven.publish)
}

kotlin {
    jvmToolchain(21)
}

sourceSets {
    main {
        // Compile the vendored UniFFI Kotlin binding directly from the cera-ffi
        // crate's output dir — no copy, so it can't drift from `just bindings`.
        kotlin.srcDir("../../cera-ffi/bindings/kotlin")
        // Hand-written helpers shared with the Android module (decisions).
        kotlin.srcDir("../shared/src/main/kotlin")
        // Native libraries are staged into the default `src/main/resources` by
        // `just jvm-libs` (gitignored): darwin-aarch64/libcera_ffi.dylib,
        // linux-x86-64/libcera_ffi.so, win32-x86-64/cera_ffi.dll. JNA discovers
        // them on the classpath via its platform resource prefix.
    }
}

dependencies {
    // `api` (not `implementation`): the generated binding's public signatures
    // expose JNA + coroutines types, so consumers need them transitively.
    api(libs.jna)
    api(libs.kotlinx.coroutines.core)
    testImplementation(libs.junit)
}

tasks.test {
    // Opt-in tests that load the real native library read it from here (the
    // dir holding libcera_ffi.dylib / .so); unset, they skip.
    System.getenv("CERA_FFI_LIB_DIR")?.let { systemProperty("jna.library.path", it) }
    // The forked JVM inherits the environment; declare it as an input so a
    // cached skip is never replayed for a run that has the model.
    inputs.property("ceraFfiLibDir", System.getenv("CERA_FFI_LIB_DIR") ?: "")
    inputs.property("ceraD1Gguf", System.getenv("CERA_D1_OMNI_GGUF") ?: "")
    inputs.property("ceraD1Mmproj", System.getenv("CERA_D1_OMNI_MMPROJ") ?: "")
}

mavenPublishing {
    publishToMavenCentral()
    // SNAPSHOTs to the Central Portal snapshot repo don't require GPG signing;
    // turn it on only for real releases (when the version drops -SNAPSHOT).
    if (!version.toString().endsWith("SNAPSHOT")) {
        signAllPublications()
    }
    pom {
        name.set("cera-ffi (JVM)")
        description.set(
            "UniFFI/JNA Kotlin bindings for the cera inference engine — desktop JVM, " +
                "with native libraries bundled for macOS/Linux/Windows."
        )
        // Dual-licensed Apache-2.0 OR MIT, matching the Cargo workspace.
        licenses {
            license {
                name.set("The Apache License, Version 2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                distribution.set("repo")
            }
            license {
                name.set("MIT License")
                url.set("https://opensource.org/licenses/MIT")
                distribution.set("repo")
            }
        }
    }
}

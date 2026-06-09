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
    }
}

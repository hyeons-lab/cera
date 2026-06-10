plugins {
    // AGP 9+ has built-in Kotlin support — the `org.jetbrains.kotlin.android`
    // plugin is no longer applied here (see kotl.in/gradle/agp-built-in-kotlin).
    alias(libs.plugins.android.library)
    alias(libs.plugins.maven.publish)
}

android {
    // Package identifier — un-hyphenated (Java/Kotlin syntax forbids `-`);
    // distinct from the Maven groupId `com.hyeons-lab`.
    namespace = "com.hyeonslab.cera.ffi"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
    }

    sourceSets {
        named("main") {
            // Same vendored binding the JVM module uses (no copy). Native libs
            // are staged into the default `src/main/jniLibs/<abi>/` by
            // `just android-libs` (gitignored) — no custom srcDir needed.
            kotlin.srcDir("../../cera-ffi/bindings/kotlin")
        }
    }
    // Release-variant publishing (+ sources/javadoc jars) is configured by the
    // vanniktech maven-publish plugin; don't call `publishing.singleVariant`
    // here or AGP rejects the duplicate registration.
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    // Android needs JNA's `@aar` artifact (bundles the per-ABI JNI dispatch libs).
    api("net.java.dev.jna:jna:${libs.versions.jna.get()}@aar")
    api(libs.kotlinx.coroutines.core)
}

mavenPublishing {
    publishToMavenCentral()
    if (!version.toString().endsWith("SNAPSHOT")) {
        signAllPublications()
    }
    pom {
        name.set("cera-ffi (Android)")
        description.set(
            "UniFFI/JNA Kotlin bindings for the cera inference engine — Android AAR with " +
                "jniLibs for arm64-v8a, armeabi-v7a, x86_64, and x86."
        )
    }
}

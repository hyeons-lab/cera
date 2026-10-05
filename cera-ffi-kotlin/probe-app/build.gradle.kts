plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.hyeonslab.cera.probe"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "com.hyeonslab.cera.probe"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.compileSdk.get().toInt()
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
        // The build to measure background CPU with. A debuggable app runs its managed code in a
        // deoptimizable interpreter (ART drops ahead-of-time code for it), which made the audio
        // service look about 1.7x more expensive (0.035 vs 0.020 CPU-s per audio-s at 100 ms
        // chunks; see README). Signed with the debug key so it installs over the debug build;
        // push models to the external files dir, since `run-as` needs a debuggable app.
        create("field") {
            initWith(getByName("debug"))
            isDebuggable = false
            signingConfig = signingConfigs.getByName("debug")
            // cera-ffi-android only has debug and release variants.
            matchingFallbacks += listOf("release", "debug")
        }
    }


    lint {
        // Fail on errors like cera-ffi-android does: CI runs lintField, and an Error-level
        // NewApi here once shipped green because unit tests exercise only host APIs.
        abortOnError = true
    }
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":cera-ffi-android"))
    implementation(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
}

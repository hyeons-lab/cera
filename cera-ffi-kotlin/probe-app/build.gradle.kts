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
    }

    // Same flag as the manifest's extractNativeLibs (kept in both places:
    // the DSL is authoritative for the app build, the manifest documents
    // and merges). The DSP skels must be extracted files at install time.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    lint {
        abortOnError = false
    }
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":cera-ffi-android"))
    implementation(libs.kotlinx.coroutines.core)
}

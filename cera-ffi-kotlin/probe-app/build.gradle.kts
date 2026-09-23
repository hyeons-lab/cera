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

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    // Resolves the JDK 21 toolchain on machines that don't have it installed.
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "cera-ffi-kotlin"

include(":cera-ffi-jvm")
include(":cera-ffi-android")

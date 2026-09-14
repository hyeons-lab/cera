plugins {
    kotlin("multiplatform") version "2.3.20"
    id("co.touchlab.skie") version "0.10.11"
}

val compatibilityProfile = providers.gradleProperty("compatibilityProfile").getOrElse("stable")
require(compatibilityProfile in setOf("stable", "extended"))
layout.buildDirectory.set(layout.projectDirectory.dir("build/$compatibilityProfile"))

// Isolated export experiment: no publishing plugin or inference engine dependency.
kotlin {
    jvm()
    macosArm64 {
        binaries.framework {
            baseName = "LeapSDK"
            isStatic = true
            binaryOption("bundleId", "ai.cera.probes.leap")
        }
    }
    sourceSets.commonMain.dependencies {
        implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    }
    sourceSets.commonMain {
        kotlin.srcDir("src/profiles/$compatibilityProfile")
    }
}

skie {
    analytics { enabled.set(false) }
}

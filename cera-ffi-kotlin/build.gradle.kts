// Root build script for the cera-ffi-kotlin multi-module build.
//
// The module plugins are declared here once with `apply false` so their classes
// load in a single (root) classloader scope, then each module applies them.
// This matters for the vanniktech maven-publish plugin in particular: it
// registers a shared `MavenCentralBuildService`. If the plugin is only applied
// inside each module, both the `:cera-ffi-jvm` and `:cera-ffi-android` scopes
// load their own copy of the plugin classes, and Gradle then refuses to share
// the build service across them — failing the publish with:
//   "Cannot set the value of task ':cera-ffi-jvm:prepareMavenCentralPublishing'
//    property 'buildService' ... loaded with [jvm scope] using a provider ...
//    loaded with [android scope]."
// Declaring the plugins at the root collapses them to one classloader scope and
// resolves the conflict.
plugins {
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.maven.publish) apply false
}

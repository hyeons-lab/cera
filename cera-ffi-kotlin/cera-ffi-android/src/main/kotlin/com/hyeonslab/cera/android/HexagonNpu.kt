package com.hyeonslab.cera.android

import android.content.Context
import android.os.Build
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import java.io.File

/**
 * One-time Hexagon NPU setup for Android apps. Call [setup] once at
 * startup, on the main thread, before [uniffi.cera_ffi.hexagonProbe] or
 * loading a model with the Hexagon backend.
 *
 * What it does: verifies the DSP skels the AAR ships in `jniLibs` were
 * extracted to `nativeLibraryDir` (real files the FastRPC loader can
 * open by path), then points the loader there via `ADSP_LIBRARY_PATH`.
 * The app never opens `/dev/fastrpc-*` itself: `libcdsprpc.so` routes
 * around the denied open through Qualcomm's DSP service, which hands
 * back an already-open fd.
 *
 * Requires the AAR manifest entries (merged automatically into
 * consumers): `extractNativeLibs="true"` and
 * `<uses-native-library android:name="libcdsprpc.so">`.
 */
object HexagonNpu {
    /**
     * DSP skel filenames, one per supported Hexagon architecture. Must
     * match `HexagonArch::skel_filename` in the Rust backend and the
     * files `just android-libs` stages into `jniLibs/arm64-v8a`.
     */
    val skelFiles: List<String> = listOf(
        "libggml-htp-v73.so",
        "libggml-htp-v75.so",
        "libggml-htp-v79.so",
        "libggml-htp-v81.so",
    )

    /**
     * Vendor fallback paths, searched after the app's own skel dir (the
     * device ships no QNN/HTP skels in `/vendor`, but the loader
     * resolves its own support files through these).
     */
    private const val VENDOR_PATHS = "/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/vendor/dsp"

    @Volatile
    private var installed = false

    /**
     * Point the FastRPC loader at this app's extracted skels. Idempotent
     * (repeats are no-ops); the process environment is set exactly once
     * because `setenv` is not thread-safe.
     *
     * @throws IllegalStateException when the skels are missing from
     *   `nativeLibraryDir`: either the APK was built without extracted
     *   native libs, or this ABI ships no skels (arm64-v8a only — x86_64
     *   Android has no Hexagon DSP). The message names which case it is.
     * @throws RuntimeException when the environment update itself fails.
     */
    @Synchronized
    fun setup(context: Context) {
        setup(context.applicationInfo.nativeLibraryDir)
    }

    /** Same as [setup], for callers that already hold the library dir. */
    @Synchronized
    fun setup(nativeLibraryDir: String) {
        if (installed) {
            return
        }
        val missing = skelFiles.filter { !File(nativeLibraryDir, it).exists() }
        if (missing.isNotEmpty()) {
            val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
            throw IllegalStateException(
                "Hexagon DSP skels missing from $nativeLibraryDir " +
                    "(${missing.joinToString()}): " +
                    if (abi == "arm64-v8a") {
                        "the APK was built without extracted native libs " +
                            "(needs extractNativeLibs, see the AAR manifest)"
                    } else {
                        "skels ship on arm64-v8a only (this device is $abi, " +
                            "which has no Hexagon DSP); treat the NPU as unavailable"
                    },
            )
        }
        try {
            Os.setenv("ADSP_LIBRARY_PATH", "$nativeLibraryDir;$VENDOR_PATHS", true)
        } catch (e: ErrnoException) {
            throw RuntimeException("failed to set ADSP_LIBRARY_PATH", e)
        }
        installed = true
    }

    /**
     * Whether this process can open the FastRPC device node directly.
     * Stock apps cannot (SELinux denies the open) and reach the DSP
     * through the DSP-service fallback instead; shells, rooted and eng
     * builds can. Diagnostic only: [setup] + probe is the same call
     * sequence either way, but the route varies per OEM/SoC/firmware, so
     * reporters should log it alongside probe results.
     */
    fun hasDirectNodeAccess(): Boolean {
        return try {
            Os.close(Os.open("/dev/fastrpc-cdsp", OsConstants.O_RDONLY, 0))
            true
        } catch (e: ErrnoException) {
            false
        }
    }
}

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
 * What it does: extracts the embedded DSP skels from `libcera_ffi.so`
 * into `context.noBackupFilesDir/cera_skels` (real files the FastRPC loader
 * can open by path), then points the loader there via `ADSP_LIBRARY_PATH`.
 * The app never opens `/dev/fastrpc-*` itself: `libcdsprpc.so` routes
 * around the denied open through Qualcomm's DSP service, which hands
 * back an already-open fd.
 *
 * Requires the AAR manifest entry (merged automatically into consumers):
 * `<uses-native-library android:name="libcdsprpc.so">`.
 */
object HexagonNpu {
    /**
     * DSP skel filenames, one per supported Hexagon architecture. Must
     * match `HexagonArch::skel_filename` in the Rust backend and the
     * files embedded in `libcera_ffi.so`.
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
     * Extract embedded DSP skels into `context.noBackupFilesDir/cera_skels`
     * and point the FastRPC loader there. Idempotent (repeats are no-ops);
     * the process environment is set exactly once because `setenv` is not
     * thread-safe.
     *
     * @param context Application or activity context.
     * @return Absolute path to the directory where skels were installed.
     * @throws IllegalArgumentException when the target dir contains `;`.
     * @throws IllegalStateException when skels fail to extract or this ABI
     *   has no Hexagon DSP support (arm64-v8a only).
     * @throws RuntimeException when the environment update itself fails.
     */
    @Synchronized
    fun setup(context: Context): String {
        val skelDir = File(context.noBackupFilesDir, "cera_skels")
        val dir = skelDir.absolutePath
        setup(dir)
        return dir
    }

    /**
     * Point the FastRPC loader at the specified skel directory, extracting
     * embedded skels into it if they are not already present.
     *
     * @param dir Absolute path to a writable directory for skel files.
     */
    @Synchronized
    fun setup(dir: String) {
        if (installed) {
            return
        }
        // A ';' in the dir would silently become two loader search entries,
        // breaking skel resolution with no error naming the cause (mirrors
        // the Rust-side rejection in `install_skels`).
        require(!dir.contains(';')) {
            "skel dir contains ';', which splits into two loader entries: $dir"
        }
        val targetDir = File(dir)
        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }
        try {
            uniffi.cera_ffi.hexagonInstallSkels(dir)
        } catch (e: Exception) {
            val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
            val missing = skelFiles.filter { !File(dir, it).exists() }
            if (missing.isNotEmpty()) {
                throw IllegalStateException(
                    "Hexagon DSP skels missing from $dir (${missing.joinToString()}): " +
                        if (abi == "arm64-v8a") {
                            "failed to extract embedded skels: ${e.message}"
                        } else {
                            "skels ship on arm64-v8a only (this device is $abi, " +
                                "which has no Hexagon DSP); treat the NPU as unavailable"
                        },
                    e,
                )
            }
        }
        val missing = skelFiles.filter { !File(dir, it).exists() }
        if (missing.isNotEmpty()) {
            val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
            throw IllegalStateException(
                "Hexagon DSP skels missing from $dir (${missing.joinToString()}): " +
                    if (abi == "arm64-v8a") {
                        "skel extraction failed to write DSP binaries"
                    } else {
                        "skels ship on arm64-v8a only (this device is $abi, " +
                            "which has no Hexagon DSP); treat the NPU as unavailable"
                    },
            )
        }
        // ';' is the separator the FastRPC loader parses (same form the
        // Rust `install_skels` writes); entries are deduplicated so a
        // second staging flow composing in either order stays valid.
        // A failed read aborts setup: it is not proof of absence, and the
        // `setenv(..., true)` below would clobber someone else's loader
        // paths. Matches the fail-loud `setenv` arm and the Rust side.
        val cur = try {
            Os.getenv("ADSP_LIBRARY_PATH")
        } catch (e: ErrnoException) {
            throw RuntimeException("failed to read ADSP_LIBRARY_PATH", e)
        }
        val merged = mergeAdspPaths("$dir;$VENDOR_PATHS", cur)
        try {
            Os.setenv("ADSP_LIBRARY_PATH", merged, true)
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

/**
 * Merge a staged path list with the current `ADSP_LIBRARY_PATH` value:
 * `;`-joined, empty entries dropped, first occurrence wins. Pure (no
 * Android APIs) so plain JVM unit tests can pin the separator and the
 * dedup order without a device (compiling still needs the Android SDK).
 * Mirrors Rust `merge_adsp_paths` case for case; keep the two in sync.
 */
internal fun mergeAdspPaths(staged: String, current: String?): String {
    return (staged.split(';') + current?.split(';').orEmpty())
        .filter { it.isNotEmpty() }
        .distinct()
        .joinToString(";")
}

package com.hyeonslab.cera.android.download

/**
 * Lifecycle states for background model downloads on Android.
 */
sealed class DownloadState {
    /** Download is idle or not started. */
    object Idle : DownloadState()

    /** Establishing HTTP connection to manifest or weights. */
    data class Connecting(val bundleId: String, val url: String) : DownloadState()

    /** Active download progress. */
    data class Progress(
        val bundleId: String,
        val url: String,
        val bytesDownloaded: ULong,
        val totalBytes: ULong?,
        val percent: Int?
    ) : DownloadState()

    /** Download completed successfully and verified. */
    data class Success(
        val bundleId: String,
        val quant: String,
        val storeDir: String
    ) : DownloadState()

    /** Download failed or was cancelled. */
    data class Error(
        val bundleId: String,
        val message: String,
        val cause: Throwable? = null
    ) : DownloadState()
}

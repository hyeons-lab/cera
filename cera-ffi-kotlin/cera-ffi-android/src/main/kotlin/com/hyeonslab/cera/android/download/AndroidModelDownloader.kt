package com.hyeonslab.cera.android.download

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.BundleRepo
import uniffi.cera_ffi.CeraEngine
import uniffi.cera_ffi.DownloadProgressSink
import uniffi.cera_ffi.EngineConfig

/**
 * High-level downloader with Kotlin coroutines and Flow support.
 */
class AndroidModelDownloader(
    private val context: Context,
    private val storeDir: String = AndroidBundleRepo.defaultStoreDir(context)
) {

    /**
     * Download a model as a cold Kotlin Flow of [DownloadState].
     *
     * On Android, this defaults to running under [CeraDownloadService] (`useService = true`)
     * to protect downloads from process death and display ongoing system notifications.
     * Set `useService = false` to run directly in the calling coroutine without foreground service.
     */
    fun download(
        bundleId: String,
        quant: String = "Q4_0",
        useService: Boolean = true
    ): Flow<DownloadState> = callbackFlow {
        if (!useService) {
            trySend(DownloadState.Connecting(bundleId, ""))

            val sink = object : DownloadProgressSink {
                override fun onProgress(url: String, bytesDownloaded: ULong, totalBytes: ULong?) {
                    val percent = totalBytes?.let {
                        if (it > 0u) ((bytesDownloaded * 100u) / it).toInt() else null
                    }
                    trySend(
                        DownloadState.Progress(
                            bundleId = bundleId,
                            url = url,
                            bytesDownloaded = bytesDownloaded,
                            totalBytes = totalBytes,
                            percent = percent
                        )
                    )
                }
            }

            try {
                val repo = BundleRepo.withProgress(storeDir = storeDir, progress = sink)
                val config = EngineConfig(
                    contextSize = 0u,
                    backend = BackendPreference.AUTO,
                    bundleRepo = repo
                )
                withContext(Dispatchers.IO) {
                    CeraEngine.fromBundleId(bundleId, quant, config).use { }
                }
                send(DownloadState.Success(bundleId, quant, storeDir))
                close()
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) {
                    close(t)
                    throw t
                }
                send(DownloadState.Error(bundleId, t.message ?: "Download failed", t))
                close()
            }

            awaitClose { }
            return@callbackFlow
        }

        // Fast path: if already cached locally, emit Success immediately.
        if (AndroidBundleRepo.isCached(context, bundleId, quant, storeDir)) {
            send(DownloadState.Success(bundleId, quant, storeDir))
            close()
            return@callbackFlow
        }

        // Start managed foreground service
        CeraDownloadService.start(context, bundleId, quant, storeDir)

        val job = launch {
            CeraDownloadService.downloadState.collect { state ->
                when (state) {
                    is DownloadState.Connecting -> {
                        if (state.bundleId == bundleId) send(state)
                    }
                    is DownloadState.Progress -> {
                        if (state.bundleId == bundleId) send(state)
                    }
                    is DownloadState.Success -> {
                        if (state.bundleId == bundleId) {
                            send(state)
                            close()
                        }
                    }
                    is DownloadState.Error -> {
                        if (state.bundleId == bundleId) {
                            send(state)
                            close(state.cause)
                        }
                    }
                    is DownloadState.Idle -> { }
                }
            }
        }

        awaitClose {
            job.cancel()
        }
    }

    /**
     * Download and initialize a [CeraEngine] with a progress callback.
     *
     * By default (`useService = true`), model weights are downloaded via [CeraDownloadService]
     * with an ongoing notification. Once cached, the engine is initialized and returned.
     */
    suspend fun downloadAndLoad(
        bundleId: String,
        quant: String = "Q4_0",
        backend: BackendPreference = BackendPreference.AUTO,
        contextSize: ULong = 0u,
        useService: Boolean = true,
        onProgress: ((bytesDownloaded: ULong, totalBytes: ULong?, percent: Int?) -> Unit)? = null
    ): CeraEngine = withContext(Dispatchers.IO) {
        if (useService && !AndroidBundleRepo.isCached(context, bundleId, quant, storeDir)) {
            download(bundleId, quant, useService = true).collect { state ->
                if (state is DownloadState.Progress) {
                    onProgress?.invoke(state.bytesDownloaded, state.totalBytes, state.percent)
                } else if (state is DownloadState.Error) {
                    throw RuntimeException(state.message, state.cause)
                }
            }
        }

        val sink = if (!useService && onProgress != null) {
            object : DownloadProgressSink {
                override fun onProgress(url: String, bytesDownloaded: ULong, totalBytes: ULong?) {
                    val percent = totalBytes?.let {
                        if (it > 0u) ((bytesDownloaded * 100u) / it).toInt() else null
                    }
                    onProgress(bytesDownloaded, totalBytes, percent)
                }
            }
        } else {
            null
        }

        val config = AndroidBundleRepo.createConfig(
            context = context,
            backend = backend,
            contextSize = contextSize,
            progress = sink,
            storeDir = storeDir
        )

        CeraEngine.fromBundleIdAsync(bundleId, quant, config)
    }

    /**
     * Trigger a managed background download using [CeraDownloadService].
     */
    fun startServiceDownload(bundleId: String, quant: String = "Q4_0") {
        CeraDownloadService.start(context, bundleId, quant, storeDir)
    }
}

package com.hyeonslab.cera.android.download

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.BundleRepo
import uniffi.cera_ffi.CeraEngine
import uniffi.cera_ffi.DownloadProgressSink
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.LeapBundleEntry
import uniffi.cera_ffi.listLeapBundles
import uniffi.cera_ffi.listLeapBundlesAsync

/**
 * Idiomatic Android wrapper around cera's [BundleRepo] and model downloading.
 */
object AndroidBundleRepo {

    /**
     * The recommended persistent store directory for Android (`filesDir/cera-bundles`).
     *
     * Never uses `cacheDir` which the OS can wipe under memory pressure.
     */
    fun defaultStoreDir(context: Context): String {
        return context.filesDir.absolutePath + "/cera-bundles"
    }

    /**
     * Create a [BundleRepo] targeting the app's persistent storage.
     */
    fun create(
        context: Context,
        progress: DownloadProgressSink? = null,
        storeDir: String = defaultStoreDir(context)
    ): BundleRepo {
        return if (progress != null) {
            BundleRepo.withProgress(storeDir = storeDir, progress = progress)
        } else {
            BundleRepo(storeDir = storeDir)
        }
    }

    /**
     * Create an [EngineConfig] configured with the Android bundle repository.
     */
    fun createConfig(
        context: Context,
        backend: BackendPreference = BackendPreference.AUTO,
        contextSize: ULong = 0u,
        progress: DownloadProgressSink? = null,
        storeDir: String = defaultStoreDir(context)
    ): EngineConfig {
        val repo = create(context, progress, storeDir)
        return EngineConfig(
            contextSize = contextSize,
            backend = backend,
            bundleRepo = repo
        )
    }

    /**
     * Check if a bundle quant is already downloaded and cached locally.
     *
     * Performs a local filesystem check without network requests or engine allocation.
     */
    suspend fun isCached(
        context: Context,
        bundleId: String,
        quant: String = "Q4_0",
        storeDir: String = defaultStoreDir(context)
    ): Boolean = withContext(Dispatchers.IO) {
        val root = java.io.File(storeDir)
        if (!root.exists() || !root.isDirectory) return@withContext false
        val cleanQuant = quant.lowercase()
        val cleanId = bundleId.lowercase().substringAfterLast('/')
        root.walkTopDown().any { file ->
            file.isFile && file.length() > 0L &&
                !file.name.endsWith(".partial") &&
                !file.name.endsWith(".sha256") &&
                file.absolutePath.lowercase().contains(cleanId) &&
                file.name.lowercase().contains(cleanQuant)
        }
    }

    /**
     * Compute total storage bytes used by downloaded models on Android.
     */
    suspend fun cacheSize(
        context: Context,
        storeDir: String = defaultStoreDir(context)
    ): ULong = withContext(Dispatchers.IO) {
        val repo = BundleRepo(storeDir = storeDir)
        repo.cacheSize()
    }

    /**
     * Clear all cached models from persistent storage on Android.
     */
    suspend fun clearCache(
        context: Context,
        storeDir: String = defaultStoreDir(context)
    ) = withContext(Dispatchers.IO) {
        val repo = BundleRepo(storeDir = storeDir)
        repo.clearCache()
    }

    /**
     * List available models from the LeapBundles catalog.
     */
    suspend fun listBundles(): List<LeapBundleEntry> = listLeapBundlesAsync()

    /**
     * Download a model bundle as a Flow of [DownloadState], favoring the Android foreground service by default.
     */
    fun download(
        context: Context,
        bundleId: String,
        quant: String = "Q4_0",
        useService: Boolean = true,
        storeDir: String = defaultStoreDir(context)
    ): kotlinx.coroutines.flow.Flow<DownloadState> =
        AndroidModelDownloader(context, storeDir).download(bundleId, quant, useService)

    /**
     * Download and load an engine, favoring the Android foreground service for downloading by default.
     */
    suspend fun downloadAndLoad(
        context: Context,
        bundleId: String,
        quant: String = "Q4_0",
        backend: BackendPreference = BackendPreference.AUTO,
        contextSize: ULong = 0u,
        useService: Boolean = true,
        storeDir: String = defaultStoreDir(context),
        onProgress: ((bytesDownloaded: ULong, totalBytes: ULong?, percent: Int?) -> Unit)? = null
    ): CeraEngine = AndroidModelDownloader(context, storeDir).downloadAndLoad(
        bundleId = bundleId,
        quant = quant,
        backend = backend,
        contextSize = contextSize,
        useService = useService,
        onProgress = onProgress
    )
}

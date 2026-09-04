package com.hyeonslab.cera.android.download

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.BundleRepo
import uniffi.cera_ffi.CeraEngine
import uniffi.cera_ffi.DownloadProgressSink
import uniffi.cera_ffi.EngineConfig

/**
 * Foreground Service for running model downloads safely in the background on Android.
 */
class CeraDownloadService : Service() {

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var downloadJob: Job? = null

    val downloadState: StateFlow<DownloadState> get() = Companion.downloadState

    inner class LocalBinder : Binder() {
        fun getService(): CeraDownloadService = this@CeraDownloadService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val bundleId = intent?.getStringExtra(EXTRA_BUNDLE_ID)
        val quant = intent?.getStringExtra(EXTRA_QUANT) ?: "Q4_0"
        val storeDir = intent?.getStringExtra(EXTRA_STORE_DIR)
            ?: AndroidBundleRepo.defaultStoreDir(applicationContext)

        if (bundleId != null) {
            startModelDownload(bundleId, quant, storeDir, startId)
        } else {
            stopSelf(startId)
        }
        return START_NOT_STICKY
    }

    private fun startModelDownload(bundleId: String, quant: String, storeDir: String, startId: Int) {
        downloadJob?.cancel()
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel(notificationManager)

        val initialNotification = buildNotification(
            title = "Downloading $bundleId ($quant)",
            content = "Connecting...",
            progress = 0,
            indeterminate = true
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                notificationConfig.notificationId,
                initialNotification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(notificationConfig.notificationId, initialNotification)
        }

        downloadJob = serviceScope.launch {
            try {
                _downloadState.value = DownloadState.Connecting(bundleId, "")

                val sink = object : DownloadProgressSink {
                    private var lastPercent: Int? = null
                    private var lastBytes: ULong = 0u

                    override fun onProgress(url: String, bytesDownloaded: ULong, totalBytes: ULong?) {
                        val percent = totalBytes?.let {
                            if (it > 0u) ((bytesDownloaded * 100u) / it).toInt() else null
                        }
                        _downloadState.value = DownloadState.Progress(
                            bundleId = bundleId,
                            url = url,
                            bytesDownloaded = bytesDownloaded,
                            totalBytes = totalBytes,
                            percent = percent
                        )

                        // Rate-limit notification updates on integer percentage changes or ~1MB on indeterminate
                        val shouldNotify = if (percent != null) {
                            percent != lastPercent
                        } else {
                            bytesDownloaded.toLong() - lastBytes.toLong() >= 1_000_000L || lastPercent != -1
                        }

                        if (shouldNotify) {
                            lastPercent = percent ?: -1
                            lastBytes = bytesDownloaded
                            val fileName = url.substringAfterLast('/')
                            val title = notificationConfig.title ?: "Downloading $bundleId"
                            val notification = buildNotification(
                                title = title,
                                content = if (percent != null) "$fileName ($percent%)" else fileName,
                                progress = percent ?: 0,
                                indeterminate = percent == null
                            )
                            notificationManager.notify(notificationConfig.notificationId, notification)
                        }
                    }
                }

                val repo = BundleRepo.withProgress(storeDir = storeDir, progress = sink)
                val config = EngineConfig(
                    contextSize = 0u,
                    backend = BackendPreference.AUTO,
                    bundleRepo = repo
                )

                // Blocking call offloaded to IO dispatcher
                CeraEngine.fromBundleId(bundleId, quant, config).use { }

                _downloadState.value = DownloadState.Success(bundleId, quant, storeDir)
            } catch (t: Throwable) {
                if (t is CancellationException) {
                    throw t
                }
                _downloadState.value = DownloadState.Error(bundleId, t.message ?: "Download failed", t)
            } finally {
                val thisJob = coroutineContext[Job]
                if (thisJob?.isCancelled != true && downloadJob === thisJob) {
                    if (notificationConfig.autoCancelOnComplete) {
                        stopForeground(STOP_FOREGROUND_REMOVE)
                    } else {
                        stopForeground(STOP_FOREGROUND_DETACH)
                    }
                    stopSelf(startId)
                }
            }
        }
    }

    private fun createNotificationChannel(manager: NotificationManager) {
        val channel = NotificationChannel(
            notificationConfig.channelId,
            notificationConfig.channelName,
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = notificationConfig.channelDescription
        }
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(
        title: String,
        content: String,
        progress: Int,
        indeterminate: Boolean
    ): Notification {
        return Notification.Builder(this, notificationConfig.channelId)
            .setContentTitle(title)
            .setContentText(content)
            .setSmallIcon(notificationConfig.smallIconResId)
            .setOngoing(true)
            .setProgress(100, progress, indeterminate)
            .build()
    }

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    companion object {
        var notificationConfig = ModelDownloadNotificationConfig()

        private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
        val downloadState: StateFlow<DownloadState> = _downloadState.asStateFlow()

        const val EXTRA_BUNDLE_ID = "com.hyeonslab.cera.EXTRA_BUNDLE_ID"
        const val EXTRA_QUANT = "com.hyeonslab.cera.EXTRA_QUANT"
        const val EXTRA_STORE_DIR = "com.hyeonslab.cera.EXTRA_STORE_DIR"

        fun start(context: Context, bundleId: String, quant: String = "Q4_0", storeDir: String? = null) {
            val intent = Intent(context, CeraDownloadService::class.java).apply {
                putExtra(EXTRA_BUNDLE_ID, bundleId)
                putExtra(EXTRA_QUANT, quant)
                if (storeDir != null) {
                    putExtra(EXTRA_STORE_DIR, storeDir)
                }
            }
            context.startForegroundService(intent)
        }
    }
}

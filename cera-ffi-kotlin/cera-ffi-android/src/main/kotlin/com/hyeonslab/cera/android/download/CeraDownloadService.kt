package com.hyeonslab.cera.android.download

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.os.Binder
import android.os.Build
import android.os.IBinder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.cera_ffi.BundleRepo
import uniffi.cera_ffi.DownloadProgressSink

/**
 * Foreground Service for running model downloads safely in the background on Android.
 */
class CeraDownloadService : Service() {

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var downloadJob: Job? = null

    val downloadState: SharedFlow<DownloadState> get() = Companion.downloadState

    inner class LocalBinder : Binder() {
        fun getService(): CeraDownloadService = this@CeraDownloadService
    }

    private var activeBundleId: String? = null
    private var activeQuant: String? = null
    private var latestStartId: Int = 0

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        latestStartId = startId
        if (intent?.action == ACTION_CANCEL) {
            cancelActiveDownload()
            return START_NOT_STICKY
        }
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

    private fun cancelActiveDownload() {
        val bundleId = activeBundleId
        downloadJob?.cancel()
        downloadJob = null
        activeBundleId = null
        activeQuant = null
        if (bundleId != null) {
            _downloadState.tryEmit(
                DownloadState.Error(
                    bundleId = bundleId,
                    message = "Download cancelled by user",
                    cause = CancellationException("Download cancelled by user")
                )
            )
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf(latestStartId)
    }

    private fun startModelDownload(bundleId: String, quant: String, storeDir: String, startId: Int) {
        latestStartId = startId
        if (downloadJob?.isActive == true && activeBundleId == bundleId && activeQuant == quant) {
            return
        }
        downloadJob?.cancel()
        activeBundleId = bundleId
        activeQuant = quant

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel(notificationManager)

        val initialNotification = buildNotification(
            title = "Downloading $bundleId ($quant)",
            content = "Connecting...",
            progress = 0,
            indeterminate = true
        )

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    notificationConfig.notificationId,
                    initialNotification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                )
            } else {
                startForeground(notificationConfig.notificationId, initialNotification)
            }
        } catch (e: Exception) {
            _downloadState.tryEmit(
                DownloadState.Error(
                    bundleId = bundleId,
                    message = "Failed to start foreground service: ${e.message}",
                    cause = e
                )
            )
            stopSelf(startId)
            return
        }

        downloadJob = serviceScope.launch {
            try {
                _downloadState.emit(DownloadState.Connecting(bundleId, ""))

                val sink = object : DownloadProgressSink {
                    private var lastUrl: String? = null
                    private var lastPercent: Int? = null
                    private var lastBytes: ULong = 0u

                    override fun onProgress(url: String, bytesDownloaded: ULong, totalBytes: ULong?) {
                        if (url != lastUrl) {
                            lastUrl = url
                            lastPercent = null
                            lastBytes = 0u
                        }
                        val percent = totalBytes?.let {
                            if (it > 0u) ((bytesDownloaded * 100u) / it).toInt() else null
                        }
                        _downloadState.tryEmit(
                            DownloadState.Progress(
                                bundleId = bundleId,
                                url = url,
                                bytesDownloaded = bytesDownloaded,
                                totalBytes = totalBytes,
                                percent = percent
                            )
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
                // Download bundle files directly without allocating an engine
                repo.downloadBundle(bundleId, quant)

                _downloadState.emit(DownloadState.Success(bundleId, quant, storeDir))
            } catch (t: Throwable) {
                withContext(NonCancellable) {
                    val message = if (t is CancellationException) "Download cancelled" else (t.message ?: "Download failed")
                    _downloadState.emit(DownloadState.Error(bundleId, message, t))
                }
                if (t is CancellationException) {
                    throw t
                }
            } finally {
                val thisJob = coroutineContext[Job]
                if (thisJob?.isCancelled != true && downloadJob === thisJob) {
                    if (notificationConfig.autoCancelOnComplete) {
                        stopForeground(STOP_FOREGROUND_REMOVE)
                    } else {
                        stopForeground(STOP_FOREGROUND_DETACH)
                    }
                    stopSelf(latestStartId)
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
        val cancelIntent = Intent(this, CeraDownloadService::class.java).apply {
            action = ACTION_CANCEL
        }
        val pendingCancel = PendingIntent.getService(
            this,
            0,
            cancelIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, notificationConfig.channelId)
            .setContentTitle(title)
            .setContentText(content)
            .setSmallIcon(notificationConfig.smallIconResId)
            .setOngoing(true)
            .setProgress(100, progress, indeterminate)
            .addAction(
                Notification.Action.Builder(
                    Icon.createWithResource(this, android.R.drawable.ic_menu_close_clear_cancel),
                    "Cancel",
                    pendingCancel
                ).build()
            )
            .build()
    }

    override fun onDestroy() {
        serviceScope.cancel()
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
        notificationManager?.cancel(notificationConfig.notificationId)
        super.onDestroy()
    }

    companion object {
        var notificationConfig = ModelDownloadNotificationConfig()

        private val _downloadState = MutableSharedFlow<DownloadState>(
            replay = 0,
            extraBufferCapacity = 64,
            onBufferOverflow = kotlinx.coroutines.channels.BufferOverflow.DROP_OLDEST
        )
        val downloadState: SharedFlow<DownloadState> = _downloadState.asSharedFlow()

        const val ACTION_CANCEL = "com.hyeonslab.cera.ACTION_CANCEL"
        const val EXTRA_BUNDLE_ID = "com.hyeonslab.cera.EXTRA_BUNDLE_ID"
        const val EXTRA_QUANT = "com.hyeonslab.cera.EXTRA_QUANT"
        const val EXTRA_STORE_DIR = "com.hyeonslab.cera.EXTRA_STORE_DIR"

        fun cancel(context: Context) {
            val intent = Intent(context, CeraDownloadService::class.java).apply {
                action = ACTION_CANCEL
            }
            context.startService(intent)
        }

        fun start(
            context: Context,
            bundleId: String,
            quant: String = "Q4_0",
            storeDir: String? = null
        ): Boolean {
            return try {
                val intent = Intent(context, CeraDownloadService::class.java).apply {
                    putExtra(EXTRA_BUNDLE_ID, bundleId)
                    putExtra(EXTRA_QUANT, quant)
                    if (storeDir != null) {
                        putExtra(EXTRA_STORE_DIR, storeDir)
                    }
                }
                context.startForegroundService(intent)
                true
            } catch (e: Exception) {
                false
            }
        }
    }
}

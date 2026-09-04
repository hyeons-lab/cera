package com.hyeonslab.cera.android.download

/**
 * Configuration for Android foreground service notifications.
 */
data class ModelDownloadNotificationConfig(
    val channelId: String = DEFAULT_CHANNEL_ID,
    val channelName: String = DEFAULT_CHANNEL_NAME,
    val channelDescription: String? = DEFAULT_CHANNEL_DESC,
    val notificationId: Int = DEFAULT_NOTIFICATION_ID,
    val smallIconResId: Int = android.R.drawable.stat_sys_download,
    val title: String? = null,
    val autoCancelOnComplete: Boolean = true
) {
    companion object {
        const val DEFAULT_CHANNEL_ID = "cera_model_downloads"
        const val DEFAULT_CHANNEL_NAME = "Model Downloads"
        const val DEFAULT_CHANNEL_DESC = "Progress notifications for AI model downloads"
        const val DEFAULT_NOTIFICATION_ID = 4280
    }
}

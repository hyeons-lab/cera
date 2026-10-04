package com.hyeonslab.cera.probe

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.Process
import android.util.Log
import com.hyeonslab.cera.android.HexagonNpu
import uniffi.cera_ffi.FfiAudioPipeline
import uniffi.cera_ffi.audioPipelineDefaultConfig
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Always-on transcription with speaker labels: a foreground service of type `microphone` that
 * feeds the microphone to a cera `AudioPipeline` (VAD, Whisper, Sortformer diarizer, optional wake
 * word), all on the Hexagon NPU where the device has one. Android demotes background CPU work and
 * does not demote the NPU, so the CPU time the service itself spends is the number to watch: it
 * logs CPU seconds per audio second to logcat (`CeraAudio`) every minute.
 *
 * Start it with [start] from a visible activity: Android 14 and later refuse to start a microphone
 * service from the background, and it also needs `RECORD_AUDIO` granted first. It does not restart
 * itself after the system kills it (`START_NOT_STICKY`), because a restart from the background
 * would be refused anyway; reopen the app.
 *
 * Models are read from `audio-models/` (see [AudioModels]). Transcripts are appended to
 * `filesDir/transcript.jsonl`.
 */
class AudioPipelineService : Service() {
    private val stopRequested = AtomicBoolean(false)
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        // HexagonNpu.setup writes the process environment, and setenv is not thread-safe: do it
        // here, once, on the main thread, before the worker thread exists.
        npuSetup = runCatching { HexagonNpu.setup(this) }
            .onFailure { Log.w(TAG, "HexagonNpu.setup failed; the pipeline will run on the CPU", it) }
            .map { "skels in $it" }
            .getOrElse { "NPU setup failed: ${it.message}" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Transcription", NotificationManager.IMPORTANCE_LOW),
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopRequested.set(true)
            return stopAndRemove()
        }
        try {
            enterForeground()
        } catch (e: Exception) {
            // Android 14+ throws when a microphone service is started from the background or
            // without RECORD_AUDIO. Say so instead of crashing.
            Log.e(TAG, "cannot start in the foreground", e)
            AudioServiceState.status(
                AudioServiceState.Status.FAILED,
                "cannot start the microphone service: ${e.message}. Open the app and start it again.",
            )
            stopSelf()
            return START_NOT_STICKY
        }
        if (worker == null) {
            stopRequested.set(false)
            AudioServiceState.status(AudioServiceState.Status.STARTING, "loading models")
            acquireWakeLock(intent?.getBooleanExtra(EXTRA_WAKE_LOCK, true) ?: true)
            val requireHotword = intent?.getBooleanExtra(EXTRA_REQUIRE_HOTWORD, false) ?: false
            val chunkSamples = intent?.getIntExtra(EXTRA_CHUNK_MS, 0)
                ?.takeIf { it > 0 }
                ?.let(::chunkSamplesForMs)
                ?: CHUNK_SAMPLES
            val wavPath = intent?.getStringExtra(EXTRA_WAV)
            val wavSpeed = intent?.getDoubleExtra(EXTRA_WAV_SPEED, 1.0) ?: 1.0
            worker = Thread(
                { work(requireHotword, chunkSamples, wavPath, wavSpeed) },
                "cera-audio",
            ).also { it.start() }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopRequested.set(true)
        worker?.join(STOP_JOIN_MS)
        releaseWakeLock()
        if (AudioServiceState.state.value.status != AudioServiceState.Status.FAILED) {
            AudioServiceState.status(AudioServiceState.Status.STOPPED, "")
        }
        super.onDestroy()
    }

    private fun stopAndRemove(): Int {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        return START_NOT_STICKY
    }

    private fun work(requireHotword: Boolean, chunkSamples: Int, wavPath: String?, wavSpeed: Double) {
        var pipeline: FfiAudioPipeline? = null
        var source: AudioSource? = null
        try {
            val models = AudioModels.find(modelDirs())
                ?: error("no ${AudioModels.VAD} in ${modelDirs().joinToString { it.path }}")
            pipeline = buildPipeline(models, requireHotword)
            val onNpu = if (models.diarizer != null) pipeline.diarizerOnNpu() else null
            val detail = "${models.summary}; $npuSetup" +
                (onNpu?.let { "; diarizer on ${if (it) "NPU" else "CPU"}" } ?: "")
            Log.i(TAG, "pipeline ready: $detail; chunk ${chunkSamples * 1000 / SAMPLE_RATE} ms")
            source = if (wavPath != null) {
                Log.i(TAG, "replaying $wavPath at ${wavSpeed}x instead of the microphone")
                WavSource.open(File(wavPath), wavSpeed)
            } else {
                MicSource.open()
            }
            AudioServiceState.status(AudioServiceState.Status.RUNNING, detail)
            val meter = CpuMeter { Process.getElapsedCpuTime() }
            val log = File(filesDir, TRANSCRIPT_FILE)
            val runner = PipelineRunner(
                source = source,
                pipeline = FfiPipeline(pipeline, chunkSamples),
                onEvent = { event ->
                    eventLine(event)?.let {
                        Log.i(TAG, it)
                        AudioServiceState.line(it)
                    }
                    eventJson(event)?.let { log.appendText(it + "\n") }
                },
                chunkSamples = chunkSamples,
                onProgress = { audio, peak ->
                    val lap = meter.lap(audio)
                    Log.i(
                        TAG,
                        "stats audio=${"%.0f".format(audio)}s " +
                            "interval_cpu=${lap.cpuMs}ms/${"%.0f".format(lap.audioSeconds)}s " +
                            "cpu_per_audio_s=${"%.4f".format(lap.cpuSecondsPerAudioSecond)} " +
                            "total_cpu=${meter.totalCpuMs()}ms " +
                            "peak=$peak silenced=${micSilenced()}",
                    )
                },
            )
            runner.run { stopRequested.get() }
            Log.i(TAG, "stopped after ${runner.samples / SAMPLE_RATE}s of audio")
        } catch (e: Throwable) {
            Log.e(TAG, "pipeline failed", e)
            AudioServiceState.status(
                AudioServiceState.Status.FAILED,
                "${e.javaClass.simpleName}: ${e.message}",
            )
        } finally {
            runCatching { source?.close() }
            runCatching { pipeline?.close() }
            stopSelf()
        }
    }

    /**
     * Whether Android is feeding this app's recorder silence. A backgrounded app's microphone can
     * be silenced by the system (the recorder keeps running and reads zeros), so the service logs
     * it next to the input peak.
     */
    private fun micSilenced(): Boolean =
        getSystemService(AudioManager::class.java).activeRecordingConfigurations
            .any { it.isClientSilenced }

    private fun buildPipeline(models: AudioModels, requireHotword: Boolean): FfiAudioPipeline {
        val config = audioPipelineDefaultConfig().copy(
            requireHotword = requireHotword && models.hotword != null,
            autoTranscribe = models.whisper != null,
        )
        val vad = models.vad.absolutePath
        val hotword = models.hotword?.absolutePath
        val whisper = models.whisper?.absolutePath
        val diarizer = models.diarizer?.absolutePath
        return if (diarizer != null) {
            FfiAudioPipeline.fromFilesWithDiarizer(vad, hotword, whisper, diarizer, true, config)
        } else {
            FfiAudioPipeline.fromFiles(vad, hotword, whisper, config)
        }
    }

    private fun modelDirs(): List<File> = listOfNotNull(
        File(filesDir, AudioModels.DIR_NAME),
        getExternalFilesDir(null)?.let { File(it, AudioModels.DIR_NAME) },
    )

    private fun enterForeground() {
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, AudioServiceActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, AudioPipelineService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Transcribing")
            .setContentText("On-device, on the NPU")
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun acquireWakeLock(enabled: Boolean) {
        if (!enabled || wakeLock != null) return
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cera:audio")
            .apply { acquire() }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    companion object {
        private const val TAG = "CeraAudio"
        private const val CHANNEL_ID = "cera-transcription"
        private const val NOTIFICATION_ID = 1
        private const val STOP_JOIN_MS = 5_000L
        const val TRANSCRIPT_FILE = "transcript.jsonl"
        const val ACTION_STOP = "com.hyeonslab.cera.probe.STOP"

        /** Boolean extra: hold a partial wake lock while running. Default true. */
        const val EXTRA_WAKE_LOCK = "wake_lock"

        /** Boolean extra: wait for the wake word before transcribing. Default false. */
        const val EXTRA_REQUIRE_HOTWORD = "require_hotword"

        /** Int extra: milliseconds of audio per pipeline call, clamped to 100..2000. Default 500. */
        const val EXTRA_CHUNK_MS = "chunk_ms"

        /**
         * String extra: path of a 16 kHz mono 16-bit WAV to replay instead of the microphone, for
         * exercising the speech path on a device without making noise. The service stops when the
         * file ends.
         */
        const val EXTRA_WAV = "wav"

        /** Double extra: replay rate for [EXTRA_WAV], 1.0 for real time (default), 0 for as fast as possible. */
        const val EXTRA_WAV_SPEED = "wav_speed"

        @Volatile
        private var npuSetup: String = ""

        /** Start the service. Call from a visible activity with RECORD_AUDIO granted. */
        fun start(
            context: Context,
            wakeLock: Boolean = true,
            requireHotword: Boolean = false,
            chunkMs: Int = 0,
            wavPath: String? = null,
            wavSpeed: Double = 1.0,
        ) {
            context.startForegroundService(
                Intent(context, AudioPipelineService::class.java)
                    .putExtra(EXTRA_WAKE_LOCK, wakeLock)
                    .putExtra(EXTRA_REQUIRE_HOTWORD, requireHotword)
                    .putExtra(EXTRA_CHUNK_MS, chunkMs)
                    .putExtra(EXTRA_WAV, wavPath)
                    .putExtra(EXTRA_WAV_SPEED, wavSpeed),
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, AudioPipelineService::class.java).setAction(ACTION_STOP),
            )
        }
    }
}

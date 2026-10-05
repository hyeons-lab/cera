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
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Always-on transcription with speaker labels: a foreground service of type `microphone` that
 * feeds the microphone to a cera `AudioPipeline` (VAD, Whisper, a speaker diarizer, optional
 * wake word), all on the Hexagon NPU where the device has one. Android demotes background CPU work and
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

    /**
     * True once the worker thread has finished its cleanup and asked to stop: a start that
     * arrives between that moment and onDestroy may replace it instead of being swallowed.
     */
    private val workerDone = AtomicBoolean(true)

    private val startLock = Any()

    /**
     * The startId that launched the current worker, or -1 when no worker owns a stop. Guarded
     * by [startLock]: an ignored start consumes a startId without launching anything, so the
     * worker must not stop with stopSelfResult (the ignored id would veto it). Only the start
     * that launched the current worker owns the stop.
     */
    private var workerStartId = -1

    // A stop the worker requested on its way out, and whether a later start was swallowed by
    // it (ignored while winding down, or launched into the dying instance): onDestroy names
    // the loss so a tap that did nothing reads as a retry hint, not a dead button. Guarded
    // by startLock.
    private var terminalStopStartId = -1
    private var startSwallowed = false

    // Detail of a worker failure that no stop owns, so a replacement start landing between
    // the failure and onDestroy cannot bury it under a blank STOPPED. Guarded by startLock.
    private var terminalFailure: String? = null

    /** The source the worker is reading, so onDestroy can unblock a read stuck in the driver. */
    @Volatile
    private var currentSource: AudioSource? = null

    /** The pipeline the worker is running, so onDestroy can hurry it with cancel(). */
    @Volatile
    private var currentPipeline: FfiAudioPipeline? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        running = true
        // HexagonNpu.setup writes the process environment, and setenv is not thread-safe: do it
        // here, once, on the main thread, before the worker thread exists.
        npuSetup = runCatching { HexagonNpu.setup(this) }
            .onFailure { Log.w(TAG, "HexagonNpu.setup failed; the pipeline will run on the CPU", it) }
            .map { "skels in $it" }
            .getOrElse { "NPU setup failed: ${it.message ?: it.javaClass.simpleName}" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Transcription", NotificationManager.IMPORTANCE_LOW),
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopRequested.set(true)
            // A deliberate stop owns the outcome: clear any swallowed-start record and any
            // recorded failure so the worker's own finally cannot make onDestroy report a loss.
            synchronized(startLock) {
                terminalStopStartId = -1
                startSwallowed = false
                terminalFailure = null
            }
            return stopAndRemove()
        }
        // A START was delivered: consume the latch stop() may have read while onCreate was
        // still pending, so a later idle stop() does not create the service just to stop it.
        startRequested = false
        try {
            enterForeground()
        } catch (e: Exception) {
            // Android 14+ throws when a microphone service is started from the background or
            // without RECORD_AUDIO. Say so instead of crashing.
            Log.e(TAG, "cannot start in the foreground", e)
            val detail = "cannot start the microphone service: " +
                "${e.message ?: e.javaClass.simpleName}. Open the app and start it again."
            // Recorded under the lock like a worker failure: a replacement start landing
            // before onDestroy must not bury it under a blank STOPPED.
            synchronized(startLock) { terminalFailure = detail }
            AudioServiceState.status(AudioServiceState.Status.FAILED, detail)
            stopSelf()
            return START_NOT_STICKY
        }
        if (worker == null || workerDone.get()) {
            workerDone.set(false)
            stopRequested.set(false)
            AudioServiceState.status(AudioServiceState.Status.STARTING, "loading models")
            syncWakeLock(intent?.getBooleanExtra(EXTRA_WAKE_LOCK, true) ?: true)
            val requireHotword = intent?.getBooleanExtra(EXTRA_REQUIRE_HOTWORD, false) ?: false
            // Absent means the default; every present value goes through the documented clamp.
            val chunkSamples = chunkSamplesForMs(intent?.getIntExtra(EXTRA_CHUNK_MS, 500) ?: 500)
            val wavPath = intent?.getStringExtra(EXTRA_WAV)
            val wavSpeed = intent?.getDoubleExtra(EXTRA_WAV_SPEED, 1.0) ?: 1.0
            // A launch publishes STARTING over the FAILED display, so the record follows
            // the display: forget the earlier run's failure, or a replacement destroyed
            // in its load window would report the old failure instead of the retry
            // hint. terminalStopStartId stays: its staleness feeds that hint.
            synchronized(startLock) { workerStartId = startId; startSwallowed = false; terminalFailure = null }
            worker = Thread(
                { work(startId, requireHotword, chunkSamples, wavPath, wavSpeed) },
                "cera-audio",
            ).also { it.start() }
        } else {
            synchronized(startLock) { startSwallowed = true }
            Log.i(TAG, "start ignored: service already running (new extras are not re-applied)")
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        running = false
        stopRequested.set(true)
        // Unblock a worker stuck in a microphone read: closing the AudioRecord fails the read,
        // which the runner treats as a clean stop. An in-flight FFI call is hurried with
        // cancel(), which is a wait-free latch the transcription loop polls.
        runCatching { currentSource?.close() }.onFailure { Log.w(TAG, "audio source close failed (onDestroy)", it) }
        try {
            worker?.join(GRACE_JOIN_MS)
            if (worker?.isAlive == true) {
                Log.w(TAG, "worker still finishing; cancelling in-flight transcription")
                runCatching { currentPipeline?.cancel() }.onFailure { Log.w(TAG, "cancel failed", it) }
                worker?.join(STOP_JOIN_MS - GRACE_JOIN_MS)
            }
            if (worker?.isAlive == true) {
                Log.e(TAG, "worker still alive after ${STOP_JOIN_MS}ms; it keeps the pipeline past onDestroy (the mic was already released)")
            }
        } catch (e: InterruptedException) {
            // The main thread is framework-owned: do not re-interrupt it (the flag would
            // stick past onDestroy and fail the next blocking call). Hurry the worker with
            // cancel() instead, as the grace-expiry path does.
            Log.w(TAG, "interrupted while waiting for the worker; cancelling in-flight transcription", e)
            runCatching { currentPipeline?.cancel() }.onFailure { Log.w(TAG, "cancel failed", it) }
            if (worker?.isAlive == true) {
                Log.e(TAG, "worker still alive after interrupt; it keeps the pipeline past onDestroy (the mic was already released)")
            }
        } finally {
            releaseWakeLock()
            val current = AudioServiceState.state.value.status
            val outcome = synchronized(startLock) {
                destroyOutcome(current, terminalFailure, terminalStopStartId, startSwallowed, workerStartId)
            }
            outcome?.let { (status, detail) -> AudioServiceState.status(status, detail) }
            super.onDestroy()
        }
    }

    private fun stopAndRemove(): Int {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        return START_NOT_STICKY
    }

    private fun work(
        startId: Int,
        requireHotword: Boolean,
        chunkSamples: Int,
        wavPath: String?,
        wavSpeed: Double,
    ) {
        var pipeline: FfiAudioPipeline? = null
        var source: AudioSource? = null
        try {
            val dirs = modelDirs()
            val models = AudioModels.find(dirs)
                ?: error("no ${AudioModels.VAD} in ${dirs.joinToString { it.path }}")
            val hotwordIgnored = requireHotword && models.hotword == null
            if (hotwordIgnored) {
                Log.w(TAG, "require_hotword was requested but no hotword.gguf was found; transcribing without wake-word gating")
            }
            if (wavPath != null) {
                // Fail fast on intent input before the heavy model load below.
                WavSource.requireValidSpeed(wavSpeed)
                require(File(wavPath).isFile) { "wav file not found: $wavPath" }
            }
            pipeline = buildPipeline(models, hotwordGating = requireHotword && !hotwordIgnored)
            val livePipeline = pipeline
            currentPipeline = livePipeline
            val onNpu = if (models.diarizer != null) livePipeline.diarizerOnNpu() else null
            val detail = "${models.summary}; $npuSetup" +
                (onNpu?.let { "; diarizer on ${if (it) "NPU" else "CPU"}" } ?: "") +
                (if (hotwordIgnored) "; require_hotword ignored (no hotword.gguf)" else "")
            Log.i(TAG, "pipeline ready: $detail; chunk ${chunkSamples * 1000 / SAMPLE_RATE} ms")
            if (stopRequested.get()) return // finally still closes the pipeline
            source = if (wavPath != null) {
                Log.i(TAG, "replaying $wavPath at ${wavSpeed}x instead of the microphone")
                WavSource.open(File(wavPath), wavSpeed)
            } else {
                MicSource.open()
            }
            currentSource = source
            if (stopRequested.get()) return // finally still closes source and pipeline
            AudioServiceState.status(AudioServiceState.Status.RUNNING, detail)
            // A run that reaches RUNNING survived, so it owns the outcome from here: forget
            // any failure an earlier attempt recorded.
            synchronized(startLock) { terminalFailure = null }
            val meter = CpuMeter { Process.getElapsedCpuTime() }
            val log = File(filesDir, TRANSCRIPT_FILE)
            var diarizerLostLogged = false
            var rotationWarned = false
            // Rotation warns once total (not once at startup plus once at the first
            // record): the startup check and the per-record path share the latch, the
            // message, and the gating below.
            fun noteRotation(rotated: Boolean) {
                if (!rotated && !rotationWarned) {
                    rotationWarned = true
                    Log.w(TAG, ROTATION_FAILED_MESSAGE)
                }
            }
            fun checkRotation() {
                noteRotation(rotateTranscriptIfNeeded(log))
            }
            checkRotation()
            val runner = PipelineRunner(
                source = source,
                pipeline = FfiPipeline(livePipeline, chunkSamples),
                onEvent = { event ->
                    eventLine(event)?.let {
                        Log.i(TAG, it)
                        AudioServiceState.line(it)
                    }
                    eventJson(event)?.let { record ->
                        val (rotated, fault) = storeTranscriptRecord(log, record)
                        noteRotation(rotated)
                        fault?.let {
                            Log.w(TAG, "transcript append failed; record kept in logcat only: $record", it)
                        }
                    }
                },
                chunkSamples = chunkSamples,
                onProgress = { audio, peak ->
                    // Telemetry must not end capture: the FFI and binder queries below can
                    // throw, and the runner does not guard this callback.
                    runCatching {
                        if (models.diarizer != null && !diarizerLostLogged && !livePipeline.hasDiarizer()) {
                            diarizerLostLogged = true
                            Log.w(TAG, "diarizer stopped; continuing without speaker labels")
                            AudioServiceState.status(
                                AudioServiceState.Status.RUNNING,
                                "$detail; diarizer stopped",
                            )
                        }
                        val lap = meter.lap(audio)
                        Log.i(TAG, statsLine(audio, lap, meter.totalCpuMs(), peak, micSilenced()))
                    }.onFailure { Log.w(TAG, "stats report failed; continuing capture", it) }
                },
            )
            runner.run { stopRequested.get() }
            Log.i(TAG, "stopped after ${runner.samples / SAMPLE_RATE}s of audio")
        } catch (e: Throwable) {
            if (stopRequested.get()) {
                Log.i(TAG, "pipeline ended with a pending failure after stop; treating as a clean stop", e)
                AudioServiceState.status(failureStatus(stopWasRequested = true), "")
            } else {
                Log.e(TAG, "pipeline failed", e)
                val detail = "${e.javaClass.simpleName}: ${e.message ?: "no detail"}"
                synchronized(startLock) { terminalFailure = detail }
                AudioServiceState.status(failureStatus(stopWasRequested = false), detail)
            }
        } finally {
            currentSource = null
            currentPipeline = null
            runCatching { source?.close() }.onFailure { Log.w(TAG, "audio source close failed (worker)", it) }
            runCatching { pipeline?.close() }.onFailure { Log.w(TAG, "pipeline close failed (worker)", it) }
            workerDone.set(true)
            // Only the start that launched the current worker owns the stop: an ignored
            // start consumed a startId without launching anything, so stopSelfResult(startId)
            // would let it veto this stop and linger foreground. Decided under one lock so
            // a replacement launch cannot slip between the check and the stop.
            synchronized(startLock) {
                if (workerStartId == startId) {
                    workerStartId = -1
                    terminalStopStartId = startId
                    stopSelf()
                }
            }
        }
    }

    /**
     * Whether Android is feeding this app's recorder silence. A backgrounded app's microphone can
     * be silenced by the system (the recorder keeps running and reads zeros), so the service logs
     * it next to the input peak.
     */
    private fun micSilenced(): Boolean {
        // isClientSilenced() is API 29+; below that there is no silence signal to read.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
        return getSystemService(AudioManager::class.java).activeRecordingConfigurations
            .any { it.isClientSilenced }
    }

    private fun buildPipeline(models: AudioModels, hotwordGating: Boolean): FfiAudioPipeline {
        val config = audioPipelineDefaultConfig().copy(
            requireHotword = hotwordGating,
            autoTranscribe = models.whisper != null,
        )
        val vad = models.vad.absolutePath
        val hotword = models.hotword?.absolutePath
        val whisper = models.whisper?.absolutePath
        val diarizer = models.diarizer?.absolutePath
        return if (diarizer != null) {
            when (models.diarizerKind) {
                DiarizerKind.NEMOTRON3 ->
                    FfiAudioPipeline.fromFilesWithDiarizerNemotron3(vad, hotword, whisper, diarizer, preferNpu = true, config)
                DiarizerKind.SORTFORMER ->
                    FfiAudioPipeline.fromFilesWithDiarizer(vad, hotword, whisper, diarizer, preferNpu = true, config)
            }
        } else {
            FfiAudioPipeline.fromFiles(vad, hotword, whisper, config)
        }
    }

    private fun modelDirs(): List<File> = AudioModels.dirs(filesDir, getExternalFilesDir(null))

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
            .setContentText("On-device")
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

    /**
     * Holds the wake lock exactly when [enabled]: a restart with `wake_lock=false` releases
     * the lock the previous run held instead of silently keeping it.
     */
    private fun syncWakeLock(enabled: Boolean) {
        if (enabled) {
            if (wakeLock == null) {
                wakeLock = getSystemService(PowerManager::class.java)
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "cera:audio")
                    .apply { acquire() }
            }
        } else {
            releaseWakeLock()
        }
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

        /**
         * Grace join before onDestroy hurries the worker with cancel(): long enough for a
         * trailing flush, after which the remainder of [STOP_JOIN_MS] covers the cancelled
         * wind-down.
         */
        private const val GRACE_JOIN_MS = 3_000L

        /**
         * The transcript is rotated aside past this size: an always-on service would otherwise
         * append to one file for the life of the install.
         */
        private const val MAX_TRANSCRIPT_BYTES = 10_000_000L
        const val TRANSCRIPT_FILE = "transcript.jsonl"
        const val ACTION_STOP = "com.hyeonslab.cera.probe.STOP"

        /**
         * Rotates [log] aside to `<name>.1` once it passes the cap. Returns false when a needed
         * rotation failed (the caller warns and keeps appending); true when the log is under the
         * cap or rotated cleanly. Pure file work, so a JVM test pins it.
         */
        internal fun rotateTranscriptIfNeeded(log: File): Boolean {
            if (!log.exists() || log.length() <= MAX_TRANSCRIPT_BYTES) return true
            return log.renameTo(File(log.parentFile, "${log.name}.1"))
        }

        /**
         * Appends one record to [log], returning the failure instead of throwing: a full disk
         * or transient storage fault must not end capture (the record survives in logcat).
         * Pure file work, so a JVM test pins it.
         */
        internal fun appendTranscriptRecord(log: File, record: String): Throwable? =
            runCatching { log.appendText(record + "\n") }.exceptionOrNull()

        private const val ROTATION_FAILED_MESSAGE = "transcript rotation failed; appending past the cap"

        /**
         * Rotation plus append for one transcript record: whether rotation succeeded, and the
         * append fault if any, instead of throwing. Pure file work, so a JVM test pins the
         * production sequence through the runner; the caller owns the warn-once latch and the
         * log lines.
         */
        internal fun storeTranscriptRecord(log: File, record: String): Pair<Boolean, Throwable?> {
            val rotated = rotateTranscriptIfNeeded(log)
            return rotated to appendTranscriptRecord(log, record)
        }

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

        /** Double extra: replay rate for [EXTRA_WAV], 1.0 for real time (default), 0 for as fast as possible. Refused unless 0 or at least 0.1. */
        const val EXTRA_WAV_SPEED = "wav_speed"

        @Volatile
        private var npuSetup: String = ""

        // Same-process guard so stop() while idle does not create the service just to stop it
        // (which would pay NPU setup in onCreate). Sound because the service and its callers
        // share a process, like AudioServiceState.
        @Volatile
        private var running = false

        // Armed by start() so a stop() issued before onCreate is delivered still stops the
        // service; consumed by the START command. Preserves the idle-stop guard in stop().
        @Volatile
        private var startRequested = false

        /**
         * Start the service. Call from a visible activity with RECORD_AUDIO granted.
         *
         * @param chunkMs milliseconds of audio per pipeline call, null for the 500 ms default;
         * clamped to 100..2000.
         * @param wavSpeed replay rate for [wavPath], 1.0 for real time; 0 reads as fast as
         * possible. Refused unless 0 or at least 0.1.
         */
        fun start(
            context: Context,
            wakeLock: Boolean = true,
            requireHotword: Boolean = false,
            chunkMs: Int? = null,
            wavPath: String? = null,
            wavSpeed: Double = 1.0,
        ) {
            startRequested = true
            try {
                context.startForegroundService(
                    Intent(context, AudioPipelineService::class.java)
                        .putExtra(EXTRA_WAKE_LOCK, wakeLock)
                        .putExtra(EXTRA_REQUIRE_HOTWORD, requireHotword)
                        .apply { chunkMs?.let { putExtra(EXTRA_CHUNK_MS, it) } }
                        .putExtra(EXTRA_WAV, wavPath)
                        .putExtra(EXTRA_WAV_SPEED, wavSpeed),
                )
            } catch (e: RuntimeException) {
                // Nothing was delivered, so disarm: a later idle stop() must not create the
                // service just to stop it.
                startRequested = false
                throw e
            }
        }

        fun stop(context: Context) {
            if (!running && !startRequested) return
            context.startService(
                Intent(context, AudioPipelineService::class.java).setAction(ACTION_STOP),
            )
        }
    }
}

/** Which status a worker failure publishes: a stop already requested owns the outcome. */
internal fun failureStatus(stopWasRequested: Boolean): AudioServiceState.Status =
    if (stopWasRequested) AudioServiceState.Status.STOPPED else AudioServiceState.Status.FAILED

/**
 * Terminal status for onDestroy, or null to keep the published failure. A recorded failure
 * beats the swallowed-start computation: a replacement start landing between a worker failure
 * and onDestroy would otherwise bury the FAILED detail under a blank STOPPED. Pure so a JVM
 * test pins it.
 */
internal fun destroyOutcome(
    current: AudioServiceState.Status,
    failure: String?,
    terminalStop: Int,
    swallowedFlag: Boolean,
    owner: Int,
): Pair<AudioServiceState.Status, String>? {
    if (current == AudioServiceState.Status.FAILED) return null
    if (failure != null) return AudioServiceState.Status.FAILED to failure
    val swallowed = terminalStop != -1 &&
        (swallowedFlag || (owner != -1 && owner != terminalStop))
    return if (swallowed) {
        AudioServiceState.Status.STOPPED to "stopped while starting; tap Start again"
    } else {
        AudioServiceState.Status.STOPPED to ""
    }
}

/**
 * The per-minute stats line for logcat. Pure formatting, so a JVM test pins the fixed locale
 * under both a comma-decimal and a digit-shaping device locale: the %.4f site needs the
 * former, the two %.0f sites (no decimals to pin) need the latter.
 */
internal fun statsLine(
    audioSeconds: Double,
    lap: CpuMeter.Interval,
    totalCpuMs: Long,
    peak: Int,
    silenced: Boolean,
): String =
    "stats audio=${String.format(Locale.US, "%.0f", audioSeconds)}s " +
        "interval_cpu=${lap.cpuMs}ms/${String.format(Locale.US, "%.0f", lap.audioSeconds)}s " +
        "cpu_per_audio_s=${String.format(Locale.US, "%.4f", lap.cpuSecondsPerAudioSecond)} " +
        "total_cpu=${totalCpuMs}ms " +
        "peak=$peak silenced=$silenced"

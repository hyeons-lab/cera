package com.hyeonslab.cera.probe

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Starts and stops [AudioPipelineService] and shows what it hears. The service has to be started
 * from here (a visible activity) because Android 14 and later refuse to start a microphone
 * foreground service from the background.
 *
 * Launch with `--ez autostart true` to start the service as soon as the permissions are granted,
 * which is how the device test drives it without touching the screen.
 */
class AudioServiceActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var scroll: ScrollView
    private var ui: Job? = null
    private var scope: CoroutineScope? = null
    private var startAfterPermission = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { textSize = 14f }
        transcript = TextView(this).apply { textSize = 16f }
        scroll = ScrollView(this).apply { addView(transcript) }
        val start = Button(this).apply {
            text = "Start"
            setOnClickListener { startService() }
        }
        val stop = Button(this).apply {
            text = "Stop"
            setOnClickListener { AudioPipelineService.stop(this@AudioServiceActivity) }
        }
        val buttons = LinearLayout(this).apply {
            gravity = Gravity.CENTER
            addView(start, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(stop, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        setContentView(
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(32, 64, 32, 32)
                addView(status)
                addView(buttons)
                addView(scroll, LinearLayout.LayoutParams(MATCH, 0, 1f))
            },
        )
        if (intent?.getBooleanExtra(EXTRA_AUTOSTART, false) == true) startService()
    }

    override fun onStart() {
        super.onStart()
        // No Dispatchers.Main: the AAR brings coroutines-core only, so collect on a background
        // dispatcher and post each snapshot to the UI thread.
        val s = CoroutineScope(Dispatchers.Default)
        scope = s
        ui = s.launch {
            AudioServiceState.state.collect { snap ->
                val head = modelStatus() + "\n" + snap.status + " " + snap.detail
                val body = snap.lines.joinToString("\n")
                runOnUiThread {
                    status.text = head
                    transcript.text = body
                    scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
                }
            }
        }
    }

    override fun onStop() {
        ui?.cancel()
        scope?.cancel()
        super.onStop()
    }

    private fun startService() {
        val needed = buildList {
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (needed.isEmpty()) {
            launchService()
        } else {
            startAfterPermission = true
            requestPermissions(needed.toTypedArray(), REQUEST_PERMISSIONS)
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onRequestPermissionsResult(code: Int, permissions: Array<String>, results: IntArray) {
        @Suppress("DEPRECATION")
        super.onRequestPermissionsResult(code, permissions, results)
        // The notification permission is optional: the service runs without it.
        val micGranted = checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (code == REQUEST_PERMISSIONS && startAfterPermission && micGranted) {
            startAfterPermission = false
            launchService()
        }
    }

    /** Starts the service, passing along the launch extras (see [AudioPipelineService]). */
    private fun launchService() {
        val extras = intent
        AudioPipelineService.start(
            this,
            wakeLock = extras?.getBooleanExtra(AudioPipelineService.EXTRA_WAKE_LOCK, true) ?: true,
            requireHotword =
                extras?.getBooleanExtra(AudioPipelineService.EXTRA_REQUIRE_HOTWORD, false) ?: false,
            chunkMs = extras?.getIntExtra(AudioPipelineService.EXTRA_CHUNK_MS, 0) ?: 0,
            wavPath = extras?.getStringExtra(AudioPipelineService.EXTRA_WAV),
            wavSpeed = extras?.getDoubleExtra(AudioPipelineService.EXTRA_WAV_SPEED, 1.0) ?: 1.0,
        )
    }

    private fun modelStatus(): String {
        val dirs = AudioModels.dirs(filesDir, getExternalFilesDir(null))
        return AudioModels.find(dirs)?.let { "models: ${it.stages.joinToString(" + ")}" }
            ?: "no ${AudioModels.VAD} in ${dirs.joinToString { it.path }}"
    }

    companion object {
        const val EXTRA_AUTOSTART = "autostart"
        private const val REQUEST_PERMISSIONS = 1
        private const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
    }
}

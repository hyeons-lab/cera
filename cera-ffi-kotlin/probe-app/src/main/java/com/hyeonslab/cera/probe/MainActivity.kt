package com.hyeonslab.cera.probe

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import uniffi.cera_ffi.hexagonInstallSkels
import uniffi.cera_ffi.hexagonProbe
import java.io.File

/**
 * On-device NPU gate: installs the bundled DSP skels into the app's
 * private files dir, then probes the Hexagon NPU — all as a normal app
 * UID (no adb shell privileges). Result goes to logcat (`CeraProbe`)
 * and on screen.
 *
 * Run: `./gradlew :probe-app:installDebug` with a Snapdragon device
 * attached, launch the app, `adb logcat -s CeraProbe`.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = TextView(this)
        view.text = "probing…"
        view.textSize = 16f
        view.setPadding(32, 64, 32, 32)
        setContentView(view)
        CoroutineScope(Dispatchers.IO).launch {
            val result = try {
                val skelDir = File(filesDir, "hexagon-skels")
                val n = hexagonInstallSkels(skelDir.absolutePath)
                val p = hexagonProbe()
                "OK skels=$n arch=${p.arch} threads=${p.threads} " +
                    "hvx=${p.hvxUnits} hmx=${p.hmxUnits} vtcm=${p.vtcmBytes}"
            } catch (e: Exception) {
                "FAIL ${e.javaClass.simpleName}: ${e.message}"
            }
            Log.i("CeraProbe", result)
            runOnUiThread { view.text = result }
        }
    }
}

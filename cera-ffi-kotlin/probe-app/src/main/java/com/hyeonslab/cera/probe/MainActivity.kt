package com.hyeonslab.cera.probe

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import com.hyeonslab.cera.android.HexagonNpu
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.CeraEngine
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.SessionConfig
import uniffi.cera_ffi.hexagonProbe
import java.io.File

/**
 * On-device NPU gate: runs the AAR's [HexagonNpu.setup] (extracts the
 * DSP skels, points the FastRPC loader at them), then probes the
 * Hexagon NPU, all as a normal app UID (no adb shell privileges).
 * Reports the access route too, since DSP policy varies per
 * OEM/SoC/firmware: `direct` means this process can open the FastRPC
 * node itself (shell/rooted/eng), `hal-fallback` means the probe
 * succeeded without direct access (stock app via the DSP service).
 * Result goes to logcat (`CeraProbe`) and on screen.
 *
 * When `filesDir/model.gguf` exists (push it with the debuggable
 * build, e.g. `adb push` to `/data/local/tmp` + `run-as ... cp`),
 * the probe is followed by an in-app generate benchmark on the
 * Hexagon backend (512-token prompt, 128 decoded, greedy, 1 warmup +
 * 3 measured), mirroring the shell `bench_android.sh` NPU cells.
 *
 * Run: `./gradlew :probe-app:installDebug` with a Snapdragon device
 * attached, launch the app from the launcher (NOT via `run-as`: that
 * domain lacks the DSP-service grant and cannot judge app access),
 * `adb logcat -s CeraProbe`. On success grep for `open thru HAL` and
 * `Created user PD ... Unsigned:Y`.
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
            // Route first: setup itself may throw, and the route is the
            // interesting datum on every tier, success or failure.
            val route = if (HexagonNpu.hasDirectNodeAccess()) "direct" else "hal-fallback"
            val result = try {
                val skelDir = HexagonNpu.setup(this@MainActivity)
                val p = hexagonProbe()
                val probe =
                    "OK route=$route skelDir=$skelDir " +
                        "arch=${p.arch} threads=${p.threads} " +
                        "hvx=${p.hvxUnits} hmx=${p.hmxUnits} vtcm=${p.vtcmBytes}"
                Log.i("CeraProbe", probe)
                runOnUiThread { view.text = "$probe\nloading model…" }
                val model = File(filesDir, "model.gguf")
                if (!model.exists()) {
                    "$probe | no model.gguf, generate skipped"
                } else {
                    "$probe | ${runGenerateBenchmark(model) { runOnUiThread { view.text = "$probe\n$it" } }}"
                }
            } catch (e: Exception) {
                "FAIL route=$route ${e.javaClass.simpleName}: ${e.message}"
            }
            Log.i("CeraProbe", result)
            runOnUiThread { view.text = result }
        }
    }

    /**
     * In-app Hexagon generate benchmark. Loads [model] with the Hexagon
     * backend, builds an exact 512-token prompt by tiling one encoded
     * sentence, then runs greedy decode to 128 tokens (EOS ignored):
     * one warmup (discarded) plus three measured runs on a reset
     * session each. Returns a one-line summary; per-run numbers go to
     * logcat. [status] receives progress lines for the screen.
     */
    private fun runGenerateBenchmark(model: File, status: (String) -> Unit): String {
        status("loading ${model.name} (${model.length() / 1024 / 1024} MiB)…")
        val engine = CeraEngine.fromPath(
            model.absolutePath,
            EngineConfig(backend = BackendPreference.HEXAGON, contextSize = 2048uL),
        )
        val session = engine.newSession(SessionConfig(seed = 42uL))
        val sent = engine.encodeText("The Hexagon DSP accelerates matrix math for on-device inference. ")
        require(sent.isNotEmpty()) { "prompt sentence encoded to zero tokens" }
        val prompt = List(512) { sent[it % sent.size] }
        val opts = GenerateOpts(maxTokens = 128u, temperature = 0.0f, ignoreEos = true)
        val prefill = mutableListOf<Double>()
        val decode = mutableListOf<Double>()
        repeat(4) { i ->
            val tag = if (i == 0) "warmup" else "run$i"
            status("generate $tag…")
            session.reset()
            session.appendTokens(prompt)
            val out = session.generate(opts)
            val s = out.summary
            val line =
                "GEN $tag prefill=${"%.1f".format(s.promptEvalTokPerSec)} " +
                    "(${s.promptEvalTokens} tok) " +
                    "decode=${"%.1f".format(s.decodeTokPerSec)} " +
                    "(${s.tokensGenerated} tok) finish=${s.finishReason}"
            Log.i("CeraProbe", line)
            if (i > 0) {
                prefill += s.promptEvalTokPerSec
                decode += s.decodeTokPerSec
            }
        }
        // Median of three == middle after sort; no statistics library needed.
        prefill.sort()
        decode.sort()
        return "gen prefill=${"%.1f".format(prefill[1])} decode=${"%.1f".format(decode[1])} " +
            "(median of 3, prompt=512 decode=128 greedy)"
    }
}

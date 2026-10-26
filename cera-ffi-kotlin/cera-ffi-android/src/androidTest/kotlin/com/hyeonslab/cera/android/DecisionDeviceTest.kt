package com.hyeonslab.cera.android

import android.os.Bundle
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.hyeonslab.cera.decision.DecisionClient
import java.io.File
import kotlin.math.abs
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import uniffi.cera_ffi.BackendPreference

/**
 * The public d1 decision models through the AAR on a device, on the backend named by the
 * instrumentation argument `d1Backend` (`auto` by default; `cpu`, `gpu`, `npu`).
 *
 * Stage the models, the requests and the `llama-server` answers under `/data/local/tmp/leap-d1/`
 * (`d1-omni-600M-Q8_0.gguf` and `mmproj-d1-omni-600M-Q8_0.gguf`, `d1-3B-Q8_0.gguf` and
 * `mmproj-d1-3B-Q8_0.gguf`, `req-{text,image,audio}.json`, `resp-{omni,3b}-{text,image,audio}.json`);
 * a leg without its files is skipped. Run with
 * `./gradlew :cera-ffi-android:connectedDebugAndroidTest
 * -Pandroid.testInstrumentationRunnerArguments.d1Backend=npu`.
 *
 * Timings and the backend report are logged under [TAG]. Choices must equal the golden ones;
 * probabilities may differ by [TOLERANCE] (Q8 activation noise).
 */
@RunWith(AndroidJUnit4::class)
class DecisionDeviceTest {
    private var client: DecisionClient? = null

    @After
    fun cleanup() {
        client?.close()
        client = null
    }

    private val arguments: Bundle get() = InstrumentationRegistry.getArguments()

    private val backend: BackendPreference
        get() =
            when (arguments.getString("d1Backend", "auto").lowercase()) {
                "cpu" -> BackendPreference.CPU
                "gpu" -> BackendPreference.GPU
                "npu" -> BackendPreference.NPU
                "hexagon" -> BackendPreference.HEXAGON
                "metal" -> BackendPreference.METAL
                else -> BackendPreference.AUTO
            }

    @Test(timeout = 1_200_000)
    fun omniAnswersTextAudioAndImage() {
        val d1 = open("d1-omni-600M-Q8_0.gguf", "mmproj-d1-omni-600M-Q8_0.gguf")
        assertEquals("lfm2-d1-omni", d1.decisionType)
        assertTrue(d1.supportsImages)
        assertTrue(d1.supportsAudio)
        for (case in listOf("text", "audio", "image")) check(d1, case, "resp-omni-$case.json")
    }

    @Test(timeout = 1_200_000)
    fun threeBAnswersTextAndImage() {
        val d1 = open("d1-3B-Q8_0.gguf", "mmproj-d1-3B-Q8_0.gguf")
        assertEquals("lfm2-d1", d1.decisionType)
        assertTrue(d1.supportsImages)
        for (case in listOf("text", "image")) check(d1, case, "resp-3b-$case.json")
    }

    private fun open(model: String, mmproj: String): DecisionClient {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val staged = listOf(model, mmproj).map { File(STAGING_DIR, it) }
        assumeTrue("Not staged under $STAGING_DIR: $model", staged.all { it.exists() })
        // The NPU needs its skels on the loader path before the first model, on the main thread.
        InstrumentationRegistry.getInstrumentation().runOnMainSync { HexagonNpu.setup(context) }
        // The app reads /data/local/tmp but loads models from its own cache directory.
        val dir = File(context.cacheDir, "d1").apply { mkdirs() }
        val (modelFile, mmprojFile) =
            staged.map { src ->
                File(dir, src.name).also { dst ->
                    if (!dst.exists() || dst.length() != src.length()) {
                        src.inputStream().use { input -> dst.outputStream().use { input.copyTo(it) } }
                    }
                }
            }
        val start = System.nanoTime()
        val opened = runBlocking {
            DecisionClient.open(modelFile.absolutePath, mmprojFile.absolutePath, backend)
        }
        Log.i(TAG, "${model} on $backend: load ${ms(start)} ms; ${opened.backendReport}")
        client = opened
        return opened
    }

    private fun ms(start: Long) = (System.nanoTime() - start) / 1_000_000

    private fun check(d1: DecisionClient, case: String, golden: String) {
        val request = File(STAGING_DIR, "req-$case.json")
        val want = File(STAGING_DIR, golden)
        assumeTrue("Not staged: ${request.name} ${want.name}", request.exists() && want.exists())
        val start = System.nanoTime()
        val got = JSONObject(runBlocking { d1.decide(request.readText()) })
        Log.i(TAG, "${d1.decisionType} $case on ${d1.backendReport}: ${ms(start)} ms")
        compare(case, got.getJSONObject("answers"), JSONObject(want.readText()).getJSONObject("answers"))
    }

    /** Same shape, same choices, numbers within [TOLERANCE]. */
    private fun compare(path: String, got: Any, want: Any) {
        when (want) {
            is JSONObject -> {
                val gotObject = got as JSONObject
                assertEquals("$path keys", want.keys().asSequence().toSet(), gotObject.keys().asSequence().toSet())
                want.keys().forEach { compare("$path.$it", gotObject.get(it), want.get(it)) }
            }
            is JSONArray -> Unit
            is String -> assertEquals(path, want, got)
            is Number -> {
                val diff = abs((got as Number).toDouble() - want.toDouble())
                // a confidence is derived from the probabilities, so it moves about twice as much
                val tolerance = if (path.endsWith("confidence")) 2 * TOLERANCE else TOLERANCE
                assertTrue("$path differs by $diff (want $want)", diff <= tolerance)
            }
            else -> Unit
        }
    }

    private companion object {
        const val TAG = "D1Device"
        const val STAGING_DIR = "/data/local/tmp/leap-d1"
        const val TOLERANCE = 0.06
    }
}

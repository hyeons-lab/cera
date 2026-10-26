package com.hyeonslab.cera.decision

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.FfiException

/**
 * Round trip through the real native library and a real public d1-omni model. Opt in with
 * `CERA_FFI_LIB_DIR` (the dir holding `libcera_ffi`, e.g. `target/release`) and
 * `CERA_D1_OMNI_GGUF` (plus `CERA_D1_OMNI_MMPROJ` for the media test); without them the tests
 * skip, so CI (which has no model) stays model-free.
 */
class DecisionClientModelTest {
    private val gguf: String? = env("CERA_D1_OMNI_GGUF")
    private val mmproj: String? = env("CERA_D1_OMNI_MMPROJ")
    private val libDir: String? = env("CERA_FFI_LIB_DIR")

    private fun withClient(
        mmprojPath: String? = null,
        block: suspend (DecisionClient) -> Unit,
    ) {
        assumeTrue("set CERA_FFI_LIB_DIR and CERA_D1_OMNI_GGUF to run", gguf != null && libDir != null)
        runBlocking { DecisionClient.open(gguf!!, mmprojPath).use { block(it) } }
    }

    @Test
    fun `a text request is answered with the llama server response shape`() =
        withClient { client ->
            assertEquals("lfm2-d1-omni", client.decisionType)
            assertEquals("cpu", client.backendName)
            assertFalse(client.supportsImages)
            val response = client.decide(TEXT_REQUEST)
            assertTrue(response, response.contains("\"billing\""))
            assertTrue(response, response.contains("\"output_tokens\": 0"))
            assertTrue(response, response.contains("\"refund\": {\"type\": \"noul\""))
        }

    @Test
    fun `a bad request is an FfiException that says what is wrong`() =
        withClient { client ->
            try {
                client.decide("{\"state\": \"x\"}")
                fail("expected FfiException")
            } catch (e: FfiException) {
                assertTrue(e.message.orEmpty(), e.message.orEmpty().contains("questions"))
            }
            // The client still answers after a refusal.
            assertTrue(client.decide(TEXT_REQUEST).contains("answers"))
        }

    @Test
    fun `the mmproj turns on the image and speech towers`() {
        assumeTrue("set CERA_D1_OMNI_MMPROJ to run", mmproj != null)
        withClient(mmproj) { client ->
            assertTrue(client.supportsImages)
            assertTrue(client.supportsAudio)
        }
    }

    @Test
    fun `an accelerator is refused before any file is read`() {
        assumeTrue("set CERA_FFI_LIB_DIR to run", libDir != null)
        try {
            runBlocking { DecisionClient.open("/no/such/d1.gguf", backend = BackendPreference.GPU) }
            fail("expected FfiException")
        } catch (e: FfiException) {
            assertTrue(e.message.orEmpty(), e.message.orEmpty().contains("run on the CPU"))
        }
    }

    private companion object {
        fun env(name: String): String? = System.getenv(name)?.takeIf { it.isNotEmpty() }

        const val TEXT_REQUEST =
            """{"state": "I was charged twice this month, please refund one of them.",
               "questions": {
                 "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                 "team": {"type": "choice", "instructions": "Which team should handle this?",
                          "criteria": {"billing": "Charges, refunds, invoices",
                                       "technical": "App or site faults"}}}}"""
    }
}

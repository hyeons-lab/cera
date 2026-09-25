package com.hyeonslab.cera.android

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Pins the `ADSP_LIBRARY_PATH` merge `HexagonNpu.setup` performs: `;`
 * separator (what the FastRPC loader parses — `:` would be one bogus
 * entry), staged entries first, pre-existing value preserved, first
 * occurrence wins on duplicates. Plain JVM tests: the merge is pure,
 * but compiling this source set still needs the Android SDK.
 */
class MergeAdspPathsTest {
    @Test
    fun `unset current value yields staged list unchanged`() {
        assertEquals(
            "/data/app/lib;/odm/lib/rfsa/adsp",
            mergeAdspPaths("/data/app/lib;/odm/lib/rfsa/adsp", null),
        )
    }

    @Test
    fun `current value is appended after staged entries`() {
        assertEquals(
            "/data/app/lib;/vendor/dsp;/staged/by/rust",
            mergeAdspPaths("/data/app/lib;/vendor/dsp", "/staged/by/rust"),
        )
    }

    @Test
    fun `duplicate entries keep the staged position`() {
        assertEquals(
            "/data/app/lib;/vendor/dsp",
            mergeAdspPaths("/data/app/lib;/vendor/dsp", "/vendor/dsp;/data/app/lib"),
        )
    }

    @Test
    fun `empty segments are dropped`() {
        assertEquals(
            "/data/app/lib",
            mergeAdspPaths("/data/app/lib;", ";/data/app/lib;"),
        )
    }

    @Test
    fun `colon is not a separator`() {
        // A `:`-joined value is one opaque entry to the loader; the merge
        // must pass it through untouched, never split it.
        assertEquals(
            "/data/app/lib;/a:/b",
            mergeAdspPaths("/data/app/lib", "/a:/b"),
        )
    }

    @Test
    fun `empty-string current value yields staged list unchanged`() {
        // Mirrors the Rust truth table's empty-string arm: `""` contributes
        // no entries, exactly like `null` (both must hold — a `filter` that
        // admits empties would join a stray `;` here, or an empty entry the
        // loader reads as cwd).
        assertEquals(
            "/data/app/lib",
            mergeAdspPaths("/data/app/lib", ""),
        )
    }

    @Test
    fun `merging the merged value is idempotent`() {
        // Mirrors the Rust truth table's idempotency arm.
        val once = mergeAdspPaths("/d", "/v;/o")
        assertEquals("/d;/v;/o", once)
        assertEquals(once, mergeAdspPaths("/d", once))
    }
}

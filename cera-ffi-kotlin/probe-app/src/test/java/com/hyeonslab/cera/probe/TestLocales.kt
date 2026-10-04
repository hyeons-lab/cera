package com.hyeonslab.cera.probe

import java.util.Locale

/** Runs [block] with the default locale swapped, restoring it even on failure. */
internal fun withLocale(locale: Locale, block: () -> Unit) {
    val previous = Locale.getDefault()
    Locale.setDefault(locale)
    try {
        block()
    } finally {
        Locale.setDefault(previous)
    }
}

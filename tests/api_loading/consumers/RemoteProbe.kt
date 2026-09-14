package loadingprobe

import uniffi.loading_native.LoadConfig
import uniffi.loading_native.Source
import java.io.File
import uniffi.loading_native.ProbeBundleRepo as BundleRepo
import uniffi.loading_native.ProbeDownloadProgressSink as DownloadProgressSink
import uniffi.loading_native.ProbeLoadException as LoadException
import uniffi.loading_native.ProbeModelLoader as ModelLoader
import uniffi.loading_native.ProbeSession as Session

private data class ProgressEvent(
    val url: String,
    val bytes: ULong,
    val total: ULong?,
)

private class RemoteTrace {
    private val events = mutableListOf<ProgressEvent>()

    @Synchronized
    fun record(
        url: String,
        bytes: ULong,
        total: ULong?,
    ) {
        events += ProgressEvent(url, bytes, total)
    }

    @Synchronized
    fun snapshot() = events.toList()
}

private class RemoteProgress(
    private val trace: RemoteTrace,
) : DownloadProgressSink {
    override fun onProgress(
        url: String,
        bytesDownloaded: ULong,
        totalBytes: ULong?,
    ) {
        trace.record(url, bytesDownloaded, totalBytes)
    }
}

private fun remoteSource(
    profile: String,
    root: File,
): Source =
    when (profile) {
        "hf" -> Source.HuggingFace("fixture/text:Q4_K_M@release", "Q8_0", "hqq")
        "bundle" -> Source.BundleId("LiquidAI/fixture-model", "Q8_0")
        "manifest" -> Source.Path(File(root, "manifest/inputs/model.json").absolutePath)
        else -> Source.Path(File(root, "directory/inputs").absolutePath)
    }

private fun productionRemoteSource(
    profile: String,
    root: File,
): uniffi.cera_ffi.ModelSource =
    when (profile) {
        "hf" -> uniffi.cera_ffi.ModelSource.HuggingFace("fixture/text:Q4_K_M@release", "Q8_0", "hqq")
        "bundle" -> uniffi.cera_ffi.ModelSource.BundleId("LiquidAI/fixture-model", "Q8_0")
        "manifest" -> uniffi.cera_ffi.ModelSource.Path(File(root, "manifest/inputs/model.json").absolutePath)
        else -> uniffi.cera_ffi.ModelSource.Path(File(root, "directory/inputs").absolutePath)
    }

private fun remoteLoaders(
    profile: String,
    root: File,
    trace: RemoteTrace,
): List<ModelLoader> =
    BundleRepo.withProgress(File(root, "$profile/store").absolutePath, RemoteProgress(trace)).use { repo ->
        val options = LoadConfig(contextSize = 24uL, backend = "cpu", bundleRepo = repo)
        List(2) { ModelLoader(remoteSource(profile, root), options) }
    }

fun runProductionRemote(root: File): List<String> {
    for (profile in listOf("hf", "bundle", "manifest", "directory")) {
        for (typed in listOf(false, true)) {
            val loader =
                uniffi.cera_ffi.BundleRepo(File(root, "$profile/store").absolutePath).use { repo ->
                    uniffi.cera_ffi.ModelLoader(
                        productionRemoteSource(profile, root),
                        uniffi.cera_ffi.EngineConfig(
                            contextSize = 24uL,
                            backend = uniffi.cera_ffi.BackendPreference.CPU,
                            bundleRepo = repo,
                        ),
                    )
                }
            val handle = if (typed) null else loader.build()
            val model = if (typed) loader.buildGenerative() else checkNotNull(handle!!.asGenerative())
            productionConsumed(loader)
            model.engine().use { check(it.contextSize() == 24uL) }
            val session = model.createSession(uniffi.cera_ffi.SessionConfig(seed = 0uL))
            loader.close()
            handle?.close()
            model.close()
            session.use {
                session.appendTokens(listOf(0u, 1u))
                val output = session.generate(uniffi.cera_ffi.GenerateOpts(maxTokens = 1u, temperature = 0f, ignoreEos = true))
                check(output.tokens.size == 1 && session.position() == 3u)
            }
        }
    }
    return listOf("production-remote-sources")
}

fun runRemote(root: File): List<String> {
    val endpoint = File(root, "endpoint.txt").readText()
    val commit = "2".repeat(40)
    val cacheHost = endpoint.removePrefix("http://").replace(":", "_")
    for (profile in listOf("hf", "bundle", "manifest", "directory")) {
        val trace = RemoteTrace()
        val loaders = remoteLoaders(profile, root, trace)
        var retained: BundleRepo? = null
        val sessions = mutableListOf<Session>()
        var coldEvents = 0
        for ((index, loader) in loaders.withIndex()) {
            val handle = if (index == 0) loader.build() else null
            val model = if (index == 0) checkNotNull(handle!!.asGenerative()) else loader.buildGenerative()
            consumed(loader)
            checkInfo(model)
            retained?.close()
            retained = checkNotNull(model.repositoryForProbe())
            val store = File(root, "$profile/store").absolutePath
            check(retained.storeDir() == store)
            val suffix =
                when (profile) {
                    "hf" -> "$cacheHost/fixture/text/resolve/$commit/model-Q8_0.gguf"
                    "bundle" -> "huggingface.co/LiquidAI/LeapBundles/resolve/main/fixture-model/relative.gguf"
                    else -> "$cacheHost/assets/$profile.gguf"
                }
            check(model.files().model == "$store/$suffix")
            sessions += model.createSession()
            loader.close()
            handle?.close()
            model.close()
            val events = trace.snapshot()
            if (index == 0) {
                coldEvents = events.size
                if (profile == "bundle") {
                    check(events.isEmpty())
                } else {
                    val url =
                        if (profile ==
                            "hf"
                        ) {
                            "$endpoint/fixture/text/resolve/$commit/model-Q8_0.gguf"
                        } else {
                            "$endpoint/assets/$profile.gguf"
                        }
                    check(events.isNotEmpty() && events.all { it.url == url && it.total == 614400uL })
                    check(events.last().bytes == 614400uL)
                    check(events.any { it.bytes > 0uL && it.bytes < 614400uL })
                    check(events.zipWithNext().all { (a, b) -> a.bytes <= b.bytes })
                }
            } else {
                check(events.size == coldEvents) { "cache hit emitted progress" }
            }
        }
        val url = "$endpoint/after/$profile.bin"
        checkNotNull(retained).use { repo ->
            val path = repo.resolveForProbe(url)
            check(File(path).readBytes().contentEquals("retained-callback".repeat(32).toByteArray()))
            val events = trace.snapshot()
            val added = events.drop(coldEvents)
            check(added.isNotEmpty() && added.all { it.url == url && it.total == 544uL })
            check(added.last().bytes == 544uL)
            check(repo.resolveForProbe(url) == path)
            check(trace.snapshot().size == events.size)
        }
        for (session in sessions) check(runSession(session) == "{\"tokens\":[0,1,0],\"position\":5}")
    }
    for ((source, expected) in listOf(
        Source.HuggingFace("fixture/text", null, null) to "hf",
        Source.BundleId("fixture-model", "Q8_0") to "bundle",
        remoteSource("manifest", root) to "path",
    )) {
        for (typed in listOf(false, true)) {
            ModelLoader(source, LoadConfig(contextSize = 24uL, backend = "cpu")).use { loader ->
                try {
                    if (typed) loader.buildGenerative() else loader.build()
                    error("missing repository succeeded")
                } catch (error: LoadException.Source) {
                    check(error.sourceKind == expected && error.detail.isNotEmpty())
                }
                consumed(loader)
            }
        }
    }
    BundleRepo(File(root, "errors").absolutePath).use { repo ->
        for (name in listOf("source", "kind", "assembly", "bundle-invalid")) {
            for (typed in listOf(false, true)) {
                val source =
                    if (name ==
                        "bundle-invalid"
                    ) {
                        Source.BundleId("fixture-model", "bad/quant")
                    } else {
                        Source.HuggingFace("$endpoint/fixture/$name/resolve/$commit/model.gguf", null, null)
                    }
                val options = LoadConfig(contextSize = 24uL, backend = if (name == "kind") "metal" else "cpu", bundleRepo = repo)
                ModelLoader(source, options).use { loader ->
                    try {
                        if (typed) loader.buildGenerative() else loader.build()
                        error("remote failure succeeded")
                    } catch (error: LoadException.Source) {
                        check(name == "source" || name == "bundle-invalid")
                        check(error.sourceKind == (if (name == "source") "hf" else "bundle") && error.detail.isNotEmpty())
                    } catch (error: LoadException.KindMismatch) {
                        check(name == "kind" && error.expected == "Generative" && error.actual == "Hotword" && error.architecture == "kws")
                    } catch (error: LoadException.Assembly) {
                        check(name == "assembly" && error.backend == "Cpu" && error.detail.isNotEmpty())
                    }
                    consumed(loader)
                }
            }
        }
    }
    return listOf(
        "native-remote-hf",
        "native-remote-bundle",
        "native-remote-manifest",
        "native-remote-directory",
        "native-remote-no-repo",
        "native-remote-hf-source",
        "native-remote-hf-kind",
        "native-remote-hf-assembly",
        "native-remote-bundle-invalid",
    )
}

import uniffi.cera_ffi.BackendPreference
import uniffi.cera_ffi.EngineConfig
import uniffi.cera_ffi.GenerateOpts
import uniffi.cera_ffi.ModelLoader
import uniffi.cera_ffi.ModelSource
import uniffi.cera_ffi.SessionConfig

// Run with a generative GGUF path and a raw completion prompt.
fun main(args: Array<String>) {
    require(args.size == 2) { "usage: ExplicitLoadingKt model.gguf prompt" }
    ModelLoader(ModelSource.Path(args[0]), EngineConfig(backend = BackendPreference.CPU)).use { loader ->
        loader.buildGenerative().use { model ->
            model.engine().use { engine ->
                model.createSession(SessionConfig(seed = 42uL)).use { session ->
                    session.appendTokens(engine.encodeText(args[1]))
                    val output = session.generate(GenerateOpts(maxTokens = 32u, temperature = 0.7f))
                    println(engine.decodeTokens(output.tokens))
                }
            }
        }
    }
}

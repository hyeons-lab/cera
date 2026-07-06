import Foundation
import Cera

/// Shared, observable owner of the loaded `CeraEngine` and its `Session`.
///
/// Everything that touches published UI state runs on the main actor. The
/// heavy work (network download + GGUF parse in `fromBundleIdAsync`) happens
/// off the main thread inside the FFI's own blocking worker — we just `await`
/// it, so the UI stays responsive without us managing threads by hand.
@MainActor
final class EngineStore: ObservableObject {
    enum LoadState: Equatable {
        case idle
        case downloading(fraction: Double, detail: String)
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var state: LoadState = .idle
    /// Human-readable summary of the loaded model, shown on the Load screen.
    @Published private(set) var modelSummary: String?

    /// The live engine + session. `nil` until a model finishes loading.
    private(set) var engine: CeraEngine?
    private(set) var session: Session?

    /// The bundle download cache, created once and reused across loads so the
    /// HTTP client pool and on-disk cache are shared (per the FFI guidance).
    private lazy var bundleRepo: BundleRepo = {
        BundleRepo.withProgress(storeDir: Self.cacheDirectory().path,
                                progress: progressSink)
    }()

    private lazy var progressSink = DownloadProgress { [weak self] fraction, detail in
        // Callbacks arrive on the FFI download worker; hop to the main actor.
        Task { @MainActor in
            guard let self, case .downloading = self.state else { return }
            self.state = .downloading(fraction: fraction, detail: detail)
        }
    }

    var isReady: Bool { session != nil }

    /// We ask the engine for `.auto`, which probes Metal → CPU at load time.
    /// There is no FFI to read back the backend that actually won, so the UI
    /// only ever advertises what was *requested*.
    let requestedBackendDescription = "Auto (Metal → CPU)"

    // MARK: - Loading

    /// Download `LFM2.5-1.2B-Instruct` / `Q4_0` from the LeapBundles registry
    /// (cached after the first run) and build the engine.
    func loadFromRegistry() async {
        await load {
            let config = self.makeConfig(withRepo: true)
            self.state = .downloading(fraction: 0, detail: "Starting download…")
            let engine = try await CeraEngine.fromBundleIdAsync(
                bundleId: "LFM2.5-1.2B-Instruct-GGUF",
                quant: "Q4_0",
                config: config
            )
            return engine
        }
    }

    /// Load a `.gguf` already present on disk (file-importer pick, or the
    /// `CERA_MODEL_PATH` sideload hook). No network, no bundle repo needed.
    func loadFromPath(_ path: String) async {
        await load {
            self.state = .loading
            // Build the (Sendable) config on the main actor, then hand only
            // plain values to the detached task — `fromPath` is blocking
            // (mmap + parse), so keep the main actor free.
            let config = self.makeConfig(withRepo: false)
            return try await Task.detached(priority: .userInitiated) {
                try CeraEngine.fromPath(path: path, config: config)
            }.value
        }
    }

    /// Auto-load hook for simulator/CI runs. Set the `CERA_MODEL_PATH`
    /// environment variable in the launch scheme to a local `.gguf`.
    func autoLoadFromEnvironmentIfConfigured() async {
        guard session == nil,
              case .idle = state,
              let path = ProcessInfo.processInfo.environment["CERA_MODEL_PATH"],
              FileManager.default.fileExists(atPath: path)
        else { return }
        await loadFromPath(path)
    }

    /// Headless smoke hook for `xcrun simctl launch --console`: if
    /// `CERA_SMOKE_PROMPT` is set and a model is loaded, run one streamed
    /// generation and print the reply + timings to the console. This exercises
    /// the same chat-template → prefill → streamed-decode path the Chat tab
    /// uses (i.e. the Metal GPU path on device), so CI can validate inference
    /// without driving the UI. No-op when the variable is unset.
    func runSmokeGenerationIfConfigured() async {
        guard let engine, let session,
              let prompt = ProcessInfo.processInfo.environment["CERA_SMOKE_PROMPT"],
              !prompt.isEmpty
        else { return }
        do {
            let rendered = try engine.applyChatTemplate(
                messages: [ChatMessage(role: "user", content: prompt)],
                addGenerationPrompt: true)
            try session.reset()
            try session.appendText(text: rendered)

            let collected = TextBox()
            let sink = StreamingTextSink(engine: engine, session: session) { collected.value = $0 }
            let opts = GenerateOpts(
                maxTokens: 64, temperature: 0.7, topP: 0.95, topK: 40, minP: 0,
                repetitionPenalty: 1.1, stopTokens: engine.eosToken().map { [$0] } ?? [],
                grammar: nil, flushEveryTokens: 0, flushEveryMs: 0)
            let summary = try await session.generateStreamingAsync(opts: opts, sink: sink)
            print("[CeraChat smoke] prompt: \(prompt)")
            print("[CeraChat smoke] reply: \(collected.value)")
            print("[CeraChat smoke] \(summary.tokensGenerated) tokens · decode \(summary.decodeMs)ms · finish \(summary.finishReason)")
        } catch {
            print("[CeraChat smoke] failed: \(Self.message(for: error))")
        }
    }

    /// Shared load pipeline: run `build`, then open a session and cache the
    /// model summary. Any thrown error lands in `.failed`.
    private func load(_ build: @escaping () async throws -> CeraEngine) async {
        do {
            let engine = try await build()
            self.state = .loading
            let session = engine.newSession(config: Self.defaultSessionConfig())
            self.engine = engine
            self.session = session
            self.modelSummary = Self.describe(engine.metadata())
            self.state = .ready
        } catch {
            self.state = .failed(Self.message(for: error))
        }
    }

    private func makeConfig(withRepo: Bool) -> EngineConfig {
        EngineConfig(
            contextSize: 4096,
            backend: .auto, // prefer Metal on device, fall back to CPU
            bundleRepo: withRepo ? bundleRepo : nil
        )
    }

    // MARK: - Helpers

    private static func defaultSessionConfig() -> SessionConfig {
        SessionConfig(
            maxSeqLen: nil, // use the model's own max_seq_len
            kvCompression: .none, // GPU backends use the f32 KV path anyway
            nKeep: 0,
            seed: nil,
            ubatchSize: 0
        )
    }

    private static func cacheDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("CeraBundles", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func describe(_ meta: ModelMetadata) -> String {
        """
        \(meta.architecture) · \(meta.quantization)
        vocab \(meta.vocabSize) · ctx \(meta.maxSeqLen)
        CPU tier: \(meta.cpuBackend)
        """
    }

    private static func message(for error: Error) -> String {
        if let ffi = error as? FfiError { return ffi.localizedDescription }
        return error.localizedDescription
    }
}

/// Tiny mutable holder used to ferry the smoke reply off the decode thread.
/// The sink (decode thread) writes; the awaiting caller reads once generation
/// has finished, so there's no concurrent access in practice.
private final class TextBox: @unchecked Sendable {
    var value = ""
}

/// `DownloadProgressSink` adapter that maps byte counts to a 0…1 fraction and a
/// short per-file detail string, forwarding both to a closure.
private final class DownloadProgress: DownloadProgressSink, @unchecked Sendable {
    private let onUpdate: (Double, String) -> Void

    init(onUpdate: @escaping (Double, String) -> Void) {
        self.onUpdate = onUpdate
    }

    func onProgress(url: String, bytesDownloaded: UInt64, totalBytes: UInt64?) {
        let name = URL(string: url)?.lastPathComponent ?? "bundle"
        let downloadedMB = Double(bytesDownloaded) / 1_048_576
        if let total = totalBytes, total > 0 {
            let fraction = min(1, Double(bytesDownloaded) / Double(total))
            let totalMB = Double(total) / 1_048_576
            onUpdate(fraction, String(format: "%@ · %.0f / %.0f MB", name, downloadedMB, totalMB))
        } else {
            // No Content-Length (chunked transfer): show bytes, indeterminate bar.
            onUpdate(0, String(format: "%@ · %.0f MB", name, downloadedMB))
        }
    }
}

import SwiftUI
import UniformTypeIdentifiers
import Cera

/// Second feature tab: mean-pooled hidden-state embeddings, plus an optional
/// LoRA adapter attach. Type text, run it through `hiddenStatesMeanPooled`, and
/// inspect the resulting vector (dimension, first values, L2 norm). Attaching a
/// LoRA changes every subsequent forward pass, so re-running after attach shows
/// the adapter's effect on the embedding.
struct EmbedView: View {
    @EnvironmentObject private var engineStore: EngineStore
    @StateObject private var vm = EmbedViewModel()
    @State private var text = "The quick brown fox jumps over the lazy dog."
    @State private var showingLoraImporter = false

    var body: some View {
        NavigationView {
            Group {
                if engineStore.isReady {
                    form
                } else {
                    ContentPlaceholder(
                        systemImage: "function",
                        title: "No model loaded",
                        message: "Load a model on the Load tab to compute embeddings."
                    )
                }
            }
            .navigationTitle("Embed / LoRA")
        }
        .navigationViewStyle(.stack)
    }

    private var form: some View {
        Form {
            Section("Input") {
                TextEditor(text: $text)
                    .frame(minHeight: 80)
                    .font(.body)
                Button {
                    Task { await vm.embed(text, using: engineStore) }
                } label: {
                    Label("Compute embedding", systemImage: "sparkle.magnifyingglass")
                }
                .disabled(text.isEmpty || vm.isBusy)
            }

            if let result = vm.result {
                Section("Embedding") {
                    row("Dimension", "\(result.dimension)")
                    row("L2 norm", String(format: "%.4f", result.l2Norm))
                    VStack(alignment: .leading, spacing: 4) {
                        Text("First \(result.preview.count) values")
                            .font(.caption).foregroundStyle(.secondary)
                        Text(result.previewText)
                            .font(.footnote.monospaced())
                    }
                }
            }

            Section("LoRA adapter") {
                if let name = vm.attachedLoraName {
                    Label("Attached: \(name)", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                    Button(role: .destructive) {
                        Task { await vm.detachLora(using: engineStore) }
                    } label: {
                        Label("Detach LoRA", systemImage: "xmark.circle")
                    }
                    .disabled(vm.isBusy)
                } else {
                    Button {
                        showingLoraImporter = true
                    } label: {
                        Label("Attach LoRA (.gguf / .safetensors)…", systemImage: "square.stack.3d.up")
                    }
                    .disabled(vm.isBusy)
                }
            }

            if let status = vm.status {
                Section { Text(status).font(.caption).foregroundStyle(.secondary) }
            }
        }
        .fileImporter(isPresented: $showingLoraImporter,
                      allowedContentTypes: loraTypes,
                      allowsMultipleSelection: false) { result in
            handleLoraImport(result)
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary).font(.footnote.monospaced())
        }
    }

    private func handleLoraImport(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result, let url = urls.first else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        let path = url.path
        Task {
            await vm.attachLora(path: path, using: engineStore)
            if scoped { url.stopAccessingSecurityScopedResource() }
        }
    }

    private var loraTypes: [UTType] {
        [UTType(filenameExtension: "gguf"),
         UTType(filenameExtension: "safetensors")].compactMap { $0 }
    }
}

// MARK: - View model

@MainActor
final class EmbedViewModel: ObservableObject {
    struct EmbeddingResult {
        let dimension: Int
        let l2Norm: Float
        let preview: [Float]

        var previewText: String {
            preview.map { String(format: "% .4f", $0) }.joined(separator: "  ")
        }
    }

    @Published private(set) var result: EmbeddingResult?
    @Published private(set) var attachedLoraName: String?
    @Published private(set) var status: String?
    @Published private(set) var isBusy = false

    func embed(_ text: String, using store: EngineStore) async {
        guard let engine = store.engine, let session = store.session else { return }
        isBusy = true
        status = "Computing hidden states…"
        defer { isBusy = false }
        do {
            let tokens = engine.encodeText(text: text)
            guard !tokens.isEmpty else {
                status = "Nothing to embed (empty tokenization)."
                return
            }
            // Off the main actor: this holds the session mutex and runs a full
            // forward pass over the prompt.
            let vector = try await Task.detached(priority: .userInitiated) {
                try session.hiddenStatesMeanPooled(tokens: tokens)
            }.value
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            result = EmbeddingResult(dimension: vector.count,
                                     l2Norm: norm,
                                     preview: Array(vector.prefix(8)))
            status = "Embedded \(tokens.count) tokens."
        } catch {
            status = "⚠️ \(Self.message(for: error))"
        }
    }

    func attachLora(path: String, using store: EngineStore) async {
        guard let session = store.session else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            let name = (path as NSString).lastPathComponent
            let adapters = try await Task.detached(priority: .userInitiated) { () -> LoraAdapters in
                if path.hasSuffix(".safetensors") {
                    return try LoraAdapters.fromSafetensors(path: path, alpha: nil)
                }
                return try LoraAdapters.fromGguf(path: path)
            }.value
            try session.attachLora(adapters: adapters)
            attachedLoraName = name
            status = "Attached LoRA with \(adapters.targetCount()) targets. Re-run to see its effect."
        } catch {
            status = "⚠️ LoRA attach failed: \(Self.message(for: error))"
        }
    }

    func detachLora(using store: EngineStore) async {
        guard let session = store.session else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            try session.removeLora()
            attachedLoraName = nil
            status = "Detached LoRA (back to base model)."
        } catch {
            status = "⚠️ Detach failed: \(Self.message(for: error))"
        }
    }

    private static func message(for error: Error) -> String {
        if let ffi = error as? FfiError { return ffi.localizedDescription }
        return error.localizedDescription
    }
}

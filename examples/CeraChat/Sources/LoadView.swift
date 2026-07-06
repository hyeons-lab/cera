import SwiftUI
import UniformTypeIdentifiers

/// First tab: choose how to get a model into the engine, then show status.
///
/// Two paths:
///  1. Download `LFM2.5-1.2B-Instruct / Q4_0` from the LeapBundles registry
///     (needs network the first time; cached afterwards).
///  2. Import a local `.gguf` via the system file picker (works offline).
struct LoadView: View {
    @EnvironmentObject private var engine: EngineStore
    @State private var showingImporter = false

    var body: some View {
        NavigationView {
            Form {
                Section("Model") {
                    Button {
                        Task { await engine.loadFromRegistry() }
                    } label: {
                        Label("Download LFM2.5-1.2B-Instruct (Q4_0)", systemImage: "icloud.and.arrow.down")
                    }
                    .disabled(isBusy)

                    Button {
                        showingImporter = true
                    } label: {
                        Label("Import local .gguf…", systemImage: "folder")
                    }
                    .disabled(isBusy)
                }

                Section("Status") {
                    statusRow
                    HStack {
                        Text("Backend")
                        Spacer()
                        Text(engine.requestedBackendDescription).foregroundStyle(.secondary)
                    }
                    if let summary = engine.modelSummary {
                        Text(summary)
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Text("Backend is requested as Auto — the engine probes Metal, "
                         + "then falls back to CPU. There is no API to read back which "
                         + "one actually ran, so this only reflects the request.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("CeraChat")
            .fileImporter(isPresented: $showingImporter,
                          allowedContentTypes: [ggufType],
                          allowsMultipleSelection: false) { result in
                handleImport(result)
            }
        }
        .navigationViewStyle(.stack)
    }

    private var isBusy: Bool {
        switch engine.state {
        case .downloading, .loading: return true
        default: return false
        }
    }

    @ViewBuilder
    private var statusRow: some View {
        switch engine.state {
        case .idle:
            Label("No model loaded", systemImage: "circle.dashed")
                .foregroundStyle(.secondary)
        case let .downloading(fraction, detail):
            VStack(alignment: .leading, spacing: 6) {
                Label("Downloading", systemImage: "arrow.down.circle")
                if fraction > 0 {
                    ProgressView(value: fraction)
                } else {
                    ProgressView() // indeterminate (no Content-Length yet)
                }
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        case .loading:
            Label("Loading model…", systemImage: "gearshape")
                .symbolEffectPulseIfAvailable()
        case .ready:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case let .failed(message):
            VStack(alignment: .leading, spacing: 4) {
                Label("Failed", systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.red)
                Text(message).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case let .success(urls):
            guard let url = urls.first else { return }
            // Files picked outside the app sandbox are security-scoped.
            let scoped = url.startAccessingSecurityScopedResource()
            let path = url.path
            Task {
                await engine.loadFromPath(path)
                if scoped { url.stopAccessingSecurityScopedResource() }
            }
        case let .failure(error):
            print("File import failed: \(error.localizedDescription)")
        }
    }

    /// `.gguf` has no registered UTType; fall back to a filename-extension type.
    private var ggufType: UTType {
        UTType(filenameExtension: "gguf") ?? .data
    }
}

private extension View {
    /// `symbolEffect(.pulse)` is iOS 17+. Degrade gracefully on iOS 15/16.
    @ViewBuilder
    func symbolEffectPulseIfAvailable() -> some View {
        if #available(iOS 17.0, *) {
            self.symbolEffect(.pulse)
        } else {
            self
        }
    }
}

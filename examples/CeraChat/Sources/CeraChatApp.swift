import SwiftUI

/// Entry point for the CeraChat example.
///
/// A single `EngineStore` (the loaded `CeraEngine` + its `Session`) is created
/// once and shared across every tab as an `@EnvironmentObject`, so the model is
/// loaded a single time and the chat / embedding screens reuse it.
@main
struct CeraChatApp: App {
    @StateObject private var engine = EngineStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(engine)
        }
    }
}

/// Top-level tab layout. The Chat and Embed tabs are inert until a model is
/// loaded on the Load tab — each screen renders a gentle "load a model first"
/// placeholder while `engine.session` is `nil`.
struct RootView: View {
    @EnvironmentObject private var engine: EngineStore

    var body: some View {
        TabView {
            LoadView()
                .tabItem { Label("Load", systemImage: "square.and.arrow.down") }

            ChatView()
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }

            EmbedView()
                .tabItem { Label("Embed / LoRA", systemImage: "function") }
        }
        // Convenience for on-simulator validation: if CERA_MODEL_PATH is set in
        // the launch environment, load that local .gguf automatically on first
        // appearance (no network, no manual tap). Harmless on device where the
        // variable is unset.
        .task {
            await engine.autoLoadFromEnvironmentIfConfigured()
            // Headless inference smoke check (no-op unless CERA_SMOKE_PROMPT set).
            await engine.runSmokeGenerationIfConfigured()
        }
    }
}

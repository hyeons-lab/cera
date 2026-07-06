import SwiftUI
import Cera

/// A single chat turn shown in the transcript.
struct ChatTurn: Identifiable, Equatable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var text: String
    /// True while the assistant bubble is still receiving streamed tokens.
    var isStreaming: Bool = false
}

/// Streaming multi-turn chat. On send we render the whole conversation through
/// the model's chat template, reset the session, prefill the rendered prompt,
/// then stream the reply token-by-token into the assistant bubble.
struct ChatView: View {
    @EnvironmentObject private var engineStore: EngineStore
    @StateObject private var vm = ChatViewModel()
    @State private var draft = ""

    var body: some View {
        NavigationView {
            Group {
                if engineStore.isReady {
                    chat
                } else {
                    ContentPlaceholder(
                        systemImage: "bubble.left.and.bubble.right",
                        title: "No model loaded",
                        message: "Load a model on the Load tab to start chatting."
                    )
                }
            }
            .navigationTitle("Chat")
        }
        .navigationViewStyle(.stack)
    }

    private var chat: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(vm.turns) { turn in
                            ChatBubble(turn: turn).id(turn.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: vm.turns) { _ in
                    if let last = vm.turns.last?.id {
                        withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                    }
                }
            }

            Divider()
            composer
        }
    }

    private var composer: some View {
        HStack(spacing: 8) {
            TextField("Message", text: $draft)
                .textFieldStyle(.roundedBorder)
                .disabled(vm.isGenerating)
                .onSubmit(send)

            if vm.isGenerating {
                Button(role: .destructive) {
                    vm.stop()
                } label: {
                    Image(systemName: "stop.circle.fill").font(.title2)
                }
            } else {
                Button {
                    send()
                } label: {
                    Image(systemName: "arrow.up.circle.fill").font(.title2)
                }
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        draft = ""
        vm.send(text, using: engineStore)
    }
}

/// One transcript bubble, aligned by role.
private struct ChatBubble: View {
    let turn: ChatTurn

    var body: some View {
        HStack {
            if turn.role == .user { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 4) {
                Text(turn.text.isEmpty && turn.isStreaming ? "…" : turn.text)
                    .textSelection(.enabled)
            }
            .padding(10)
            .background(turn.role == .user ? Color.accentColor.opacity(0.2) : Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            if turn.role == .assistant { Spacer(minLength: 40) }
        }
    }
}

/// Empty-state helper reused by the Chat and Embed tabs.
struct ContentPlaceholder: View {
    let systemImage: String
    let title: String
    let message: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text(title).font(.headline)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
}

// MARK: - View model

@MainActor
final class ChatViewModel: ObservableObject {
    @Published private(set) var turns: [ChatTurn] = []
    @Published private(set) var isGenerating = false

    private var task: Task<Void, Never>?
    private var currentSink: StreamingTextSink?

    func send(_ text: String, using store: EngineStore) {
        guard let engine = store.engine, let session = store.session else { return }

        turns.append(ChatTurn(role: .user, text: text))
        // Snapshot the full history for the template BEFORE adding the
        // placeholder assistant bubble.
        let history = turns.map { ChatMessage(role: $0.role == .user ? "user" : "assistant",
                                              content: $0.text) }
        let assistantIndex = turns.count
        turns.append(ChatTurn(role: .assistant, text: "", isStreaming: true))

        isGenerating = true
        task = Task {
            await self.generate(history: history, into: assistantIndex,
                                engine: engine, session: session)
        }
    }

    func stop() {
        // Atomic-backed on the Rust side; safe from the main actor. The
        // in-flight decode exits at its next between-token check.
        currentSink?.session?.cancel()
        task?.cancel()
    }

    private func generate(history: [ChatMessage], into index: Int,
                          engine: CeraEngine, session: Session) async {
        defer {
            isGenerating = false
            currentSink = nil
            if turns.indices.contains(index) { turns[index].isStreaming = false }
        }

        do {
            // Render the whole conversation, then feed it as one prefill. We
            // reset first so the KV cache doesn't double-count earlier turns —
            // simplest correct approach for an example (re-prefills each turn).
            let prompt = try engine.applyChatTemplate(messages: history,
                                                      addGenerationPrompt: true)
            try session.reset()
            try session.appendText(text: prompt)

            let sink = StreamingTextSink(engine: engine, session: session) { [weak self] display in
                Task { @MainActor in
                    guard let self, self.turns.indices.contains(index) else { return }
                    self.turns[index].text = display
                }
            }
            currentSink = sink

            let opts = GenerateOpts(
                maxTokens: 512,
                temperature: 0.7,
                topP: 0.95,
                topK: 40,
                minP: 0.0,
                repetitionPenalty: 1.1,
                stopTokens: engine.eosToken().map { [$0] } ?? [],
                grammar: nil,
                flushEveryTokens: 0,
                flushEveryMs: 0
            )

            _ = try await session.generateStreamingAsync(opts: opts, sink: sink)
        } catch is CancellationError {
            // User tapped stop; leave whatever streamed so far.
        } catch {
            if turns.indices.contains(index) {
                let partial = turns[index].text
                turns[index].text = partial.isEmpty
                    ? "⚠️ \(Self.message(for: error))"
                    : partial + "\n\n⚠️ \(Self.message(for: error))"
            }
        }
    }

    private static func message(for error: Error) -> String {
        if let ffi = error as? FfiError { return ffi.localizedDescription }
        return error.localizedDescription
    }
}

/// `ModalitySink` that turns streamed token IDs into display text on the fly.
///
/// The FFI calls these methods on the decode worker thread. We accumulate the
/// raw token IDs here (single-threaded within one generate call, so no locking
/// needed), decode the full run each time — `decodeTokens` reassembles
/// multi-byte UTF-8 / BPE merges correctly, which per-token decoding can split
/// — and forward the text to the main actor via `onDisplay`.
final class StreamingTextSink: ModalitySink, @unchecked Sendable {
    /// Exposed so the view model can `cancel()` an in-flight decode.
    let session: Session?
    private let engine: CeraEngine
    private let onDisplay: (String) -> Void
    private var tokens: [UInt32] = []

    init(engine: CeraEngine, session: Session, onDisplay: @escaping (String) -> Void) {
        self.engine = engine
        self.session = session
        self.onDisplay = onDisplay
    }

    func onTextTokens(tokens newTokens: [UInt32]) {
        tokens.append(contentsOf: newTokens)
        // Drop special/control tokens (e.g. <|im_end|>) from the visible text.
        let visible = tokens.filter { !engine.isSpecialToken(id: $0) }
        onDisplay(engine.decodeTokens(tokens: visible))
    }

    func onAudioFrames(pcm: [Float], sampleRate: UInt32) {
        // Text-only model: nothing to do.
    }

    func onDone(reason: FinishReason) {
        // The awaited `generateStreamingAsync` returns the summary, so the
        // view model finalizes there; nothing extra needed here.
    }
}

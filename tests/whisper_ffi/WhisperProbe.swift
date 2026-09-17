import Cera
import Foundation

// This helper is also the application-facing example in the Whisper guide.
func transcribeRecording(modelPath: String, pcm16kMono: [Float]) async throws -> String {
    let model = try await Task.detached {
        try FfiWhisperModel.fromFile(path: modelPath)
    }.value
    var opts = whisperDefaultTranscribeOpts()
    opts.language = "en"
    return try await model.transcribeAsync(pcm: pcm16kMono, opts: opts)
}

@main
struct WhisperProbe {
    static func main() async throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[1])
        let raw = try Data(contentsOf: directory.appendingPathComponent("audio.f32"))
        precondition(raw.count > 0 && raw.count % 4 == 0)
        let pcm: [Float] = stride(from: 0, to: raw.count, by: 4).map { offset in
            let bits = (0..<4).reduce(UInt32(0)) { $0 | UInt32(raw[offset + $1]) << (8 * $1) }
            return Float(bitPattern: bits)
        }
        var bytes = try Data(contentsOf: directory.appendingPathComponent("a.gguf"))
        let a = try FfiWhisperModel.fromBytes(bytes: bytes)
        bytes.resetBytes(in: 0..<bytes.count)
        let copy = directory.appendingPathComponent("swift-owned.gguf")
        try FileManager.default.copyItem(at: directory.appendingPathComponent("b.gguf"), to: copy)
        let b = try FfiWhisperModel.fromFile(path: copy.path)
        try FileManager.default.removeItem(at: copy)
        precondition(a.isMultilingual() && !b.isMultilingual())
        precondition(a.languages().count == 100 && Array(a.languages().prefix(2)) == ["en", "zh"])
        var opts = whisperDefaultTranscribeOpts()
        precondition(opts.language == nil && !opts.translate && !opts.timestamps)
        precondition(opts.maxTokens == 448 && opts.temperature == 0)
        opts.language = "en"
        opts.maxTokens = 3
        let sync = try a.transcribe(pcm: pcm, opts: opts)
        precondition(sync == "aaa")
        let empty = try a.transcribe(pcm: [], opts: nil)
        precondition(empty.isEmpty)
        var short = opts
        short.maxTokens = 2
        let fullOptions = opts
        let shortOptions = short
        async let first = a.transcribeAsync(pcm: pcm, opts: fullOptions)
        async let second = a.transcribeAsync(pcm: pcm, opts: shortOptions)
        async let third = b.transcribeAsync(pcm: pcm, opts: shortOptions)
        let results = try await [first, second, third]
        precondition(results == ["aaa", "aa", "bb"])
        // None/null option fields use core defaults; context caps this tiny model.
        opts.maxTokens = nil
        opts.temperature = nil
        let defaults = try a.transcribe(pcm: pcm, opts: opts)
        precondition(defaults == String(repeating: "a", count: 13))
        let recording = try await transcribeRecording(
            modelPath: directory.appendingPathComponent("a.gguf").path, pcm16kMono: pcm)
        precondition(recording == defaults)
        do {
            _ = try FfiWhisperModel.fromBytes(bytes: Data([1, 2]))
            preconditionFailure("Malformed GGUF accepted")
        } catch FfiError.Backend { }
        let missing = directory.appendingPathComponent("missing.gguf").path
        precondition(!FileManager.default.fileExists(atPath: missing))
        do {
            _ = try FfiWhisperModel.fromFile(path: missing)
            preconditionFailure("Missing GGUF accepted")
        } catch FfiError.Backend { }
        print(#"{"cases":["bytes-owned","file-owned","languages","defaults","sync","async-shared","async-distinct","empty","malformed","missing","recording-example"],"text":["aaa","aa","bb"]}"#)
    }
}

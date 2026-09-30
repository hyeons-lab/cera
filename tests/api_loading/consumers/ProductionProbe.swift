import Foundation
import loading_native

private final class ProductionSink: loading_native.ModalitySink, @unchecked Sendable {
  let session: loading_native.Session
  private let lock = NSLock()
  private var completed: [loading_native.FinishReason] = []
  private var content = ""
  var done: [loading_native.FinishReason] { lock.withLock { completed } }
  var text: String { lock.withLock { content } }

  init(_ session: loading_native.Session) { self.session = session }
  func onThoughtChunk(text: String) { preconditionFailure("unexpected thoughts") }
  func onAudioFrames(pcm: [Float], sampleRate: UInt32) {
    preconditionFailure("unexpected audio")
  }
  func onTextChunk(text: String) {
    lock.withLock { content += text }
    session.cancel()
  }
  func onDone(reason: loading_native.FinishReason) { lock.withLock { completed.append(reason) } }
}

func productionConsumed(_ loader: loading_native.ModelLoader) {
  for direct in [false, true] {
    do {
      if direct { _ = try loader.buildGenerative() } else { _ = try loader.build() }
      preconditionFailure("production loader reused")
    } catch loading_native.LoadError.Consumed {} catch {
      preconditionFailure("unexpected consumed error: \(error)")
    }
  }
}

func runProduction(_ root: URL, bytes: Data) throws -> [String] {
  for (index, backend) in [BackendPreference.auto, .cpu, .gpu, .metal, .hexagon].enumerated() {
    var encoded: [UInt8] = []
    FfiConverterTypeBackendPreference.write(backend, into: &encoded)
    precondition(encoded == [0, 0, 0, UInt8(index + 1)])
    var buffer = (data: Data(encoded), offset: 0)
    let decoded = try FfiConverterTypeBackendPreference.read(from: &buffer)
    precondition(decoded == backend && buffer.offset == 4)
  }
  for tag: Int32 in [0, 6, -1, Int32.max] {
    var wire = tag.bigEndian
    let data = withUnsafeBytes(of: &wire) { Data($0) }
    var buffer = (data: data, offset: 0)
    do {
      _ = try FfiConverterTypeBackendPreference.read(from: &buffer)
      preconditionFailure("malformed production backend enum accepted")
    } catch {
      precondition(error.localizedDescription == "Raw enum value doesn't match any cases")
      precondition(buffer.offset == 4)
    }
  }
  for backend in [BackendPreference.gpu, .metal] {
    for direct in [false, true] {
      let loader = loading_native.ModelLoader(
        source: .bytes(bytes: bytes), config: EngineConfig(backend: backend))
      do {
        if direct { _ = try loader.buildGenerative() } else { _ = try loader.build() }
        preconditionFailure("unavailable production backend accepted")
      } catch loading_native.LoadError.Assembly(let actual, let detail) {
        precondition(actual == (backend == .gpu ? "Gpu" : "Metal") && !detail.isEmpty)
      }
      productionConsumed(loader)
    }
  }
  let defaults = loading_native.EngineConfig(backend: .cpu)
  precondition(defaults.contextSize == 4096 && defaults.bundleRepo == nil)
  precondition(defaults.draftModel == nil && !defaults.gpuDepthformer)
  let sessionDefaults = loading_native.SessionConfig()
  precondition(sessionDefaults.maxSeqLen == nil && sessionDefaults.kvCompression == nil)
  precondition(sessionDefaults.nKeep == 0 && sessionDefaults.seed == nil)
  precondition(sessionDefaults.ubatchSize == 512 && !sessionDefaults.gpuDepthformer)
  let path = root.appendingPathComponent("model.gguf").path
  let sources: [loading_native.ModelSource] = [
    .bytes(bytes: bytes), .path(path: path),
    .files(
      files: loading_native.ModelFiles(
        model: path, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
        draftModel: nil, extras: [:], inferenceType: nil, chatTemplate: nil)),
    .parts(
      parts: loading_native.ModelParts(
        model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
        draftModel: nil, inferenceType: nil, chatTemplate: nil, generationDefaults: nil)),
  ]
  for source in sources {
    for direct in [false, true] {
      var loader: loading_native.ModelLoader? = loading_native.ModelLoader(
        source: source, config: defaults)
      var handle: loading_native.ModelHandle? = direct ? nil : try loader!.build()
      var model: loading_native.GenerativeModel? =
        direct ? try loader!.buildGenerative() : handle!.asGenerative()
      productionConsumed(loader!)
      var engine: loading_native.CeraEngine? = model!.engine()
      precondition(enginesShareForProbe(first: engine!, second: model!.engine()))
      precondition(engine!.contextSize() == 4096 && engine!.metadata().maxSeqLen == 64)
      let independent = try loading_native.CeraEngine.fromBytes(bytes: bytes, config: defaults)
      precondition(!enginesShareForProbe(first: engine!, second: independent))
      let config = loading_native.SessionConfig(
        maxSeqLen: 8, kvCompression: .f16, seed: UInt64.max, ubatchSize: 1,
        gpuDepthformer: true)
      let session: loading_native.Session = try model!.createSession(config: config)
      let observed = try sessionConfigForProbe(session: session)
      precondition(observed == config)
      let sibling = try engine!.newSession(config: loading_native.SessionConfig())
      weak let releasedLoader = loader
      weak let releasedHandle = handle
      weak let releasedModel = model
      loader = nil
      handle = nil
      model = nil
      precondition(releasedLoader == nil && releasedHandle == nil && releasedModel == nil)
      precondition(engine!.contextSize() == 4096)
      precondition(engine!.encodeText(text: "ab") == [0, 1])
      precondition(engine!.decodeTokens(tokens: [0, 1]) == "ab")
      try sibling.appendTokens(tokens: [1])
      try session.appendTokens(tokens: [0, 1])
      engine!.clearPrefixCache()
      precondition(session.position() == 2 && sibling.position() == 1)
      weak let releasedEngine = engine
      engine = nil
      precondition(releasedEngine == nil)
      let one = loading_native.GenerateOpts(maxTokens: 1, temperature: 0.7, ignoreEos: true)
      let first = try session.generate(opts: one)
      precondition(first.tokens.count == 1 && session.position() == 3)
      let two = loading_native.GenerateOpts(maxTokens: 2, temperature: 0.7, ignoreEos: true)
      let next = try session.generate(opts: two)
      precondition(next.tokens.count == 2 && session.position() == 5)
      precondition(sibling.position() == 1)
      try session.reset()
      precondition(session.position() == 0)
      do {
        try session.appendTokens(tokens: [])
        preconditionFailure("empty input accepted")
      } catch loading_native.FfiError.EmptyInput {}
      try session.appendTokens(tokens: [0, 1])
      let sink = ProductionSink(session)
      let streamed = try session.generateStreaming(
        opts: loading_native.GenerateOpts(
          maxTokens: 3, temperature: 0.7, ignoreEos: true, flushEveryTokens: 1), sink: sink)
      precondition(!sink.text.isEmpty && sink.done.count == 1)
      precondition(streamed.finishReason == .cancelled && streamed.tokensGenerated == 1)
      let position = session.position()
      session.clearCancel()
      _ = try session.generate(opts: one)
      precondition(session.position() == position + 1)
    }
  }
  let loader = loading_native.ModelLoader(source: .bytes(bytes: bytes), config: defaults)
  let model = try loader.buildGenerative()
  let modes: [loading_native.KvCompression?] = [
    nil, loading_native.KvCompression.none, .f16,
    .turboQuant(seed: 0, keys: true, values: true),
    .turboQuant(seed: UInt64.max, keys: true, values: false),
    .turboQuant(seed: 42, keys: false, values: true),
  ]
  for mode in modes {
    let config = loading_native.SessionConfig(
      maxSeqLen: 8, kvCompression: mode, nKeep: 1,
      seed: 0, ubatchSize: 0, gpuDepthformer: true)
    let session = try model.createSession(config: config)
    var expected = config
    expected.kvCompression = mode ?? loading_native.KvCompression.none
    let observed = try sessionConfigForProbe(session: session)
    precondition(observed == expected)
    try session.appendTokens(tokens: [0, 1])
    let generated = try session.generate(
      opts: loading_native.GenerateOpts(
        maxTokens: 1, temperature: 0, ignoreEos: true))
    precondition(generated.tokens.count == 1)
  }
  let capped = try model.createSession(config: loading_native.SessionConfig(maxSeqLen: 1))
  try capped.appendTokens(tokens: [0])
  do {
    try capped.appendTokens(tokens: [1])
    preconditionFailure("session cap ignored")
  } catch let loading_native.FfiError.ContextOverflow(maxSeqLen, by) {
    precondition(maxSeqLen == 1 && by == 1 && capped.position() == 1)
  }
  return try runProductionDefaults(bytes) + runProductionErrors(root, bytes) + [
    "production-backend-transport",
    "production-defaults", "production-sources", "production-session",
    "production-shared-engine", "production-kv-config", "production-stream-cancel",
  ]
}

func runProductionDefaults(_ bytes: Data) throws -> [String] {
  let empty = loading_native.SamplingDefaults(
    temperature: nil, topP: nil, topK: nil, minP: nil, repetitionPenalty: nil)
  let sampling = loading_native.SamplingDefaults(
    temperature: 0.37, topP: 0.71, topK: 7, minP: 0.13, repetitionPenalty: 1.23)
  var profiles: [(String, loading_native.GenerationDefaults?, loading_native.GenerationDefaults)] =
    [
      ("absent", nil, .text(sampling: empty)),
      ("text-empty", .text(sampling: empty), .text(sampling: empty)),
    ]
  for (name, value) in [
    (
      "audio",
      loading_native.GenerationDefaults.audio(
        sampling: sampling, numberOfDecodingThreads: 3, audioTemperature: 0.625, audioTopK: 11)
    ),
    (
      "audio",
      .audio(
        sampling: empty, numberOfDecodingThreads: 0, audioTemperature: 0, audioTopK: 0)
    ),
    (
      "audio",
      .audio(
        sampling: empty, numberOfDecodingThreads: UInt32.max, audioTemperature: 1,
        audioTopK: UInt32.max)
    ),
    (
      "audio-empty",
      .audio(
        sampling: empty, numberOfDecodingThreads: nil, audioTemperature: nil, audioTopK: nil)
    ),
  ] {
    profiles.append((name, value, value))
  }
  for (raw, canonical) in [
    (
      " { \"nested\" : [true, null, {\"x\":7}], \"label\": \"line\\ntext\" } ",
      "{\"label\":\"line\\ntext\",\"nested\":[true,null,{\"x\":7}]}"
    ),
    ("[1, 2, null]", "[1,2,null]"), ("42", "42"), ("true", "true"),
    ("\"text\"", "\"text\""), ("null", "null"),
  ] {
    profiles.append(("other", .other(rawJson: raw), .other(rawJson: canonical)))
  }
  for (_, defaults, expected) in profiles {
    for typed in [false, true] {
      let parts = loading_native.ModelParts(
        model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
        draftModel: nil, inferenceType: nil, chatTemplate: nil, generationDefaults: defaults)
      var loader: loading_native.ModelLoader? = loading_native.ModelLoader(
        source: .parts(parts: parts), config: EngineConfig(backend: .cpu))
      var handle: loading_native.ModelHandle?
      var model: loading_native.GenerativeModel?
      if typed {
        model = try loader!.buildGenerative()
      } else {
        handle = try loader!.build()
        model = handle!.asGenerative()
      }
      productionConsumed(loader!)
      let observed = productionDefaultsForProbe(engine: model!.engine())
      let session = try model!.createSession(config: SessionConfig(seed: 42))
      weak let releasedLoader = loader
      weak let releasedHandle = handle
      weak let releasedModel = model
      loader = nil
      handle = nil
      model = nil
      precondition(releasedLoader == nil && releasedHandle == nil && releasedModel == nil)
      precondition(observed == expected)
      try session.appendTokens(tokens: [0, 1])
      let result = try session.generate(
        opts: GenerateOpts(maxTokens: 3, temperature: 0, ignoreEos: true))
      precondition(result.tokens == [0, 1, 0] && session.position() == 5)
    }
  }
  for raw in ["", "{", "null trailing", "{\"x\":NaN}"] {
    for typed in [false, true] {
      let parts = loading_native.ModelParts(
        model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
        draftModel: nil, inferenceType: nil, chatTemplate: nil,
        generationDefaults: .other(rawJson: raw))
      let loader = loading_native.ModelLoader(
        source: .parts(parts: parts), config: EngineConfig(backend: .cpu))
      do {
        if typed { _ = try loader.buildGenerative() } else { _ = try loader.build() }
        fatalError("malformed defaults succeeded")
      } catch let loading_native.LoadError.InvalidConfig(field, value, reason, detail) {
        precondition(field == "generation_defaults.raw_json" && value == raw)
        precondition(reason == "invalid_json" && !detail.isEmpty)
      }
      productionConsumed(loader)
    }
  }
  return Set(profiles.map { "production-parts-defaults-\($0.0)" }).sorted()
    + ["production-parts-defaults-invalid-json"]
}

private func runProductionErrors(_ root: URL, _ bytes: Data) throws -> [String] {
  var cases: [String] = []
  let parts = loading_native.ModelParts(
    model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil, draftModel: nil,
    inferenceType: nil, chatTemplate: nil, generationDefaults: nil)
  for (arch, kind) in [
    ("bert", "Encoder"), ("modernbert", "Encoder"), ("whisper", "Whisper"), ("silero_vad", "Vad"),
    ("kws", "Hotword"),
  ] {
    for typed in [false, true] {
      let data = try Data(contentsOf: root.appendingPathComponent("\(arch).gguf"))
      let bad = loading_native.ModelLoader(
        source: .bytes(bytes: data), config: EngineConfig(backend: .metal))
      do {
        if typed { _ = try bad.buildGenerative() } else { _ = try bad.build() }
        fatalError("wrong kind succeeded")
      } catch let loading_native.LoadError.KindMismatch(expected, actual, architecture) {
        precondition(
          expected == "Generative" && actual == kind && architecture == arch,
          "\(arch) kind mismatch payload (typed=\(typed))")
      }
      productionConsumed(bad)
    }
    cases.append("production-kind-\(arch)")
  }
  for name in ["unknown", "malformed", "backend", "assembly", "inference"] {
    for typed in [false, true] {
      let data =
        name == "unknown"
        ? try Data(contentsOf: root.appendingPathComponent("unknown.gguf"))
        : name == "assembly"
          ? try Data(contentsOf: root.appendingPathComponent("llama.gguf"))
          : name == "malformed" ? Data([0, 1]) : bytes
      var unsupportedParts = parts
      unsupportedParts.inferenceType = "future/unsupported"
      let bad = loading_native.ModelLoader(
        source: name == "inference" ? .parts(parts: unsupportedParts) : .bytes(bytes: data),
        config: EngineConfig(backend: name == "backend" ? .metal : .cpu))
      do {
        if typed { _ = try bad.buildGenerative() } else { _ = try bad.build() }
        fatalError("invalid load succeeded")
      } catch let loading_native.LoadError.UnsupportedArchitecture(architecture) {
        precondition(name == "unknown" && architecture == "future_probe")
      } catch let loading_native.LoadError.Source(sourceKind, detail) {
        precondition(name == "malformed" && sourceKind == "bytes" && !detail.isEmpty)
      } catch let loading_native.LoadError.Assembly(backend, detail) {
        precondition(name == "backend" || name == "assembly")
        precondition(backend == (name == "backend" ? "Metal" : "Cpu") && !detail.isEmpty)
      } catch let loading_native.LoadError.UnsupportedInferenceType(inferenceType) {
        precondition(name == "inference" && inferenceType == "future/unsupported")
      }
      productionConsumed(bad)
    }
    cases.append("production-" + name)
  }
  return cases
}

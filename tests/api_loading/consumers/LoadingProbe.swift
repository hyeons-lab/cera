import Foundation
import loading_native

typealias ModelLoader = ProbeModelLoader
typealias ModelHandle = ProbeModelHandle
typealias GenerativeModel = ProbeGenerativeModel
typealias ModelFiles = ProbeModelFiles
typealias ModelParts = ProbeModelParts
typealias SamplingDefaults = ProbeSamplingDefaults
typealias GenerationDefaults = ProbeGenerationDefaults
typealias LoadError = ProbeLoadError
typealias Session = ProbeSession
typealias BundleRepo = ProbeBundleRepo
typealias DownloadProgressSink = ProbeDownloadProgressSink

func config(_ backend: String = "cpu", parts: Bool = false) -> LoadConfig {
  LoadConfig(
    contextSize: 24, backend: backend, draftModel: parts ? "missing-probe-draft.gguf" : nil,
    gpuDepthformer: parts)
}

func consumed(_ loader: ModelLoader) {
  do {
    _ = try loader.build()
    fatalError("reused loader succeeded")
  } catch LoadError.Consumed {} catch { fatalError("unexpected reuse error: \(error)") }
  do {
    _ = try loader.buildGenerative()
    fatalError("reused typed loader succeeded")
  } catch LoadError.Consumed {} catch { fatalError("unexpected typed reuse error: \(error)") }
}

func checkInfo(_ model: GenerativeModel, parts: Bool = false) {
  let info = model.info()
  precondition(info.requestedContext == 24 && info.capacity == 24)
  precondition(info.backend == "Cpu" && info.gpuDepthformer == parts)
  precondition(info.draftModel == (parts ? "missing-probe-draft.gguf" : nil))
  if parts {
    precondition(info.chatTemplate == "probe-template")
    precondition(abs(info.temperature - 0.37) < 0.00001 && abs(info.topP - 0.71) < 0.00001)
    precondition(info.topK == 7 && abs(info.minP - 0.13) < 0.00001)
    precondition(abs(info.repetitionPenalty - 1.23) < 0.00001)
  }
}

func runSession(_ session: Session) throws -> [String: Any] {
  try session.append(tokens: [0, 1])
  precondition(session.position() == 2)
  let tokens = try session.generate()
  precondition(tokens.count == 3 && tokens.allSatisfy { $0 < 2 })
  precondition(session.position() == 5)
  return ["tokens": tokens, "position": session.position()]
}

func runDefaults(_ bytes: Data) throws -> [String] {
  let empty = SamplingDefaults(
    temperature: nil, topP: nil, topK: nil, minP: nil, repetitionPenalty: nil)
  let sampling = SamplingDefaults(
    temperature: 0.37, topP: 0.71, topK: 7, minP: 0.13, repetitionPenalty: 1.23)
  var profiles: [(String, GenerationDefaults?, GenerationDefaults)] = [
    ("absent", nil, .text(sampling: empty)),
    ("text-empty", .text(sampling: empty), .text(sampling: empty)),
  ]
  for (name, value) in [
    (
      "audio",
      GenerationDefaults.audio(
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
      let parts = ModelParts(
        model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
        draftModel: nil, inferenceType: nil, chatTemplate: nil, generationDefaults: defaults)
      var loader: ModelLoader? = ModelLoader(source: .parts(parts: parts), config: config())
      var handle: ModelHandle?
      var model: GenerativeModel?
      if typed {
        model = try loader!.buildGenerative()
      } else {
        handle = try loader!.build()
        model = handle!.asGenerative()
      }
      consumed(loader!)
      let observed = model!.generationDefaultsForProbe()
      let session = try model!.createSession()
      weak let releasedLoader = loader
      weak let releasedHandle = handle
      weak let releasedModel = model
      loader = nil
      handle = nil
      model = nil
      precondition(releasedLoader == nil && releasedHandle == nil && releasedModel == nil)
      precondition(observed == expected)
      let result = try runSession(session)
      precondition(result["tokens"] as? [UInt32] == [0, 1, 0])
    }
  }
  for raw in ["", "{", "null trailing", "{\"x\":NaN}"] {
    for typed in [false, true] {
      let parts = ModelParts(
        model: bytes, multimodalProjector: nil, audioDecoder: nil, audioTokenizer: nil,
        draftModel: nil, inferenceType: nil, chatTemplate: nil,
        generationDefaults: .other(rawJson: raw))
      let loader = ModelLoader(source: .parts(parts: parts), config: config())
      do {
        if typed { _ = try loader.buildGenerative() } else { _ = try loader.build() }
        fatalError("malformed defaults succeeded")
      } catch let LoadError.InvalidConfig(field, value, reason, detail) {
        precondition(field == "generation_defaults.raw_json" && value == raw)
        precondition(reason == "invalid_json" && !detail.isEmpty)
      }
      consumed(loader)
    }
  }
  return Set(profiles.map { "parts-defaults-\($0.0)" }).sorted()
    + ["parts-defaults-invalid-json"]
}

func runConfigs(_ bytes: Data) throws -> [String] {
  let profiles: [(String, UInt64?, UInt64, UInt32)] = [
    ("default", nil, 4096, 64), ("zero", 0, 64, 64),
    ("cap", 64, 64, 64), ("cap", 65, 65, 64),
    ("wide", UInt64(UInt32.max) + 25, UInt64(UInt32.max) + 25, 64),
    ("wide", UInt64.max, 64, 64), ("small", 1, 1, 1),
  ]
  for (_, request, observed, capacity) in profiles {
    let options =
      request.map { LoadConfig(contextSize: $0, backend: "cpu") }
      ?? LoadConfig(backend: "cpu")
    precondition(options.contextSize == (request ?? 4096))
    precondition(options.draftModel == nil && !options.gpuDepthformer)
    for typed in [false, true] {
      var loader: ModelLoader? = ModelLoader(source: .bytes(bytes: bytes), config: options)
      var handle: ModelHandle?
      var model: GenerativeModel?
      if typed {
        let typedConfig = EngineConfig(
          contextSize: options.contextSize, backend: .cpu,
          bundleRepo: nil, draftModel: nil, gpuDepthformer: false)
        loader = modelLoaderWithEngineConfig(source: .bytes(bytes: bytes), config: typedConfig)
        model = try loader!.buildGenerative()
      } else {
        handle = try loader!.build()
        model = handle!.asGenerative()
      }
      consumed(loader!)
      let info = model!.info()
      precondition(info.requestedContext == observed && info.capacity == capacity)
      precondition(info.backend == "Cpu" && info.draftModel == nil && !info.gpuDepthformer)
      let session = try model!.createSession()
      weak let releasedLoader = loader
      weak let releasedHandle = handle
      weak let releasedModel = model
      loader = nil
      handle = nil
      model = nil
      precondition(releasedLoader == nil && releasedHandle == nil && releasedModel == nil)
      if capacity == 1 {
        try session.append(tokens: [0])
        do {
          try session.append(tokens: [1])
          fatalError("append exceeded the configured context")
        } catch let LoadError.Engine(detail) { precondition(!detail.isEmpty) }
        precondition(session.position() == 1)
      } else {
        let result = try runSession(session)
        precondition(result["tokens"] as? [UInt32] == [0, 1, 0])
      }
    }
  }
  return [
    "native-config-default", "native-config-zero", "native-config-cap", "native-config-wide",
    "native-config-small",
  ]
}

func runFiles(_ root: URL) throws -> [String] {
  func path(_ name: String) -> String { root.appendingPathComponent(name).path }
  func files(_ primary: String, inference: String?) -> ModelFiles {
    ModelFiles(
      model: path(primary), multimodalProjector: "missing-projector.gguf",
      audioDecoder: path("missing-decoder.gguf"), audioTokenizer: "missing-tokenizer.gguf",
      draftModel: "missing-draft.gguf",
      extras: ["future_file": "future.bin", "absolute_file": path("absolute.bin")],
      inferenceType: inference, chatTemplate: "file-template")
  }
  for typed in [false, true] {
    for inference: String? in [nil, "llama.cpp/text-to-text"] {
      var loader: ModelLoader? = ModelLoader(
        source: .files(files: files("model.gguf", inference: inference)), config: config())
      var handle: ModelHandle?
      var model: GenerativeModel?
      if typed {
        model = try loader!.buildGenerative()
      } else {
        handle = try loader!.build()
        precondition(handle!.kind() == "Generative")
        model = handle!.asGenerative()
      }
      consumed(loader!)
      checkInfo(model!)
      let resolved = model!.files()
      precondition(resolved.model == path("model.gguf"))
      precondition(resolved.multimodalProjector == path("missing-projector.gguf"))
      precondition(resolved.audioDecoder == path("missing-decoder.gguf"))
      precondition(resolved.audioTokenizer == path("missing-tokenizer.gguf"))
      precondition(resolved.draftModel == path("missing-draft.gguf"))
      precondition(
        resolved.extras == [
          "future_file": path("future.bin"), "absolute_file": path("absolute.bin"),
        ])
      precondition(resolved.inferenceType == "llama.cpp/text-to-text")
      precondition(
        resolved.chatTemplate == "file-template" && model!.info().chatTemplate == "file-template")
      let session = try model!.createSession()
      weak let releasedLoader = loader
      weak let releasedHandle = handle
      weak let releasedModel = model
      loader = nil
      handle = nil
      model = nil
      precondition(releasedLoader == nil && releasedHandle == nil && releasedModel == nil)
      let result = try runSession(session)
      precondition(result["tokens"] as? [UInt32] == [0, 1, 0])
    }
  }
  for name in ["kind", "missing", "inference"] {
    for typed in [false, true] {
      let source = files(
        name == "kind" ? "kws.gguf" : "missing-primary.gguf",
        inference: name == "inference" ? "future/unsupported" : nil)
      let loader = ModelLoader(source: .files(files: source), config: config("metal"))
      do {
        if typed { _ = try loader.buildGenerative() } else { _ = try loader.build() }
        fatalError("invalid file load succeeded")
      } catch let LoadError.KindMismatch(expected, actual, architecture) {
        precondition(
          name == "kind" && expected == "Generative" && actual == "Hotword" && architecture == "kws"
        )
      } catch let LoadError.Source(sourceKind, detail) {
        precondition(name == "missing" && sourceKind == "files" && !detail.isEmpty)
      } catch let LoadError.UnsupportedInferenceType(inferenceType) {
        precondition(name == "inference" && inferenceType == "future/unsupported")
      }
      consumed(loader)
    }
  }
  return ["native-files", "native-files-kind", "native-files-missing", "native-files-inference"]
}

@main
struct LoadingProbe {
  static func main() throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let bytes = try Data(contentsOf: root.appendingPathComponent("model.gguf"))
    var cases: [String] = []
    var input = Data(count: bytes.count)
    _ = input.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
    let bufferAddress = input.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    precondition(bufferAddress != bytes.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) })
    var loader: ModelLoader? = ModelLoader(source: .bytes(bytes: input), config: config())
    input.resetBytes(in: 0..<input.count)
    precondition(bufferAddress == input.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) })
    precondition(input.allSatisfy { $0 == 0 })
    var handle: ModelHandle? = try loader!.build()
    consumed(loader!)
    precondition(handle!.kind() == "Generative")
    var first = handle!.asGenerative()
    var second = handle!.asGenerative()
    precondition(first != nil && second != nil)
    checkInfo(second!)
    weak let releasedLoader = loader
    weak let releasedHandle = handle
    weak let releasedFirst = first
    loader = nil
    handle = nil
    first = nil
    precondition(releasedLoader == nil && releasedHandle == nil && releasedFirst == nil)
    let session = try second!.createSession()
    weak let releasedModel = second
    second = nil
    precondition(releasedModel == nil)
    let result = try runSession(session)
    cases.append("bytes-lifetime")

    let defaults = SamplingDefaults(
      temperature: 0.37, topP: 0.71, topK: 7, minP: 0.13, repetitionPenalty: 1.23)
    let parts = ModelParts(
      model: bytes, multimodalProjector: Data([1]), audioDecoder: Data([2]),
      audioTokenizer: Data([3]), draftModel: Data([4]), inferenceType: "llama.cpp/text-to-text",
      chatTemplate: "probe-template", generationDefaults: .text(sampling: defaults))
    let partsLoader = ModelLoader(source: .parts(parts: parts), config: config(parts: true))
    let partsModel = try partsLoader.buildGenerative()
    consumed(partsLoader)
    checkInfo(partsModel, parts: true)
    _ = try runSession(partsModel.createSession())
    cases.append("parts-defaults")
    cases += try runDefaults(bytes)

    let pathLoader = ModelLoader(
      source: .path(path: root.appendingPathComponent("model.gguf").path), config: config())
    let pathModel = try pathLoader.buildGenerative()
    consumed(pathLoader)
    checkInfo(pathModel)
    _ = try runSession(pathModel.createSession())
    cases.append("native-path")
    cases += try runFiles(root)
    cases += try runConfigs(bytes)
    cases += try runProduction(root, bytes: bytes)
    cases += try runRemote(URL(fileURLWithPath: CommandLine.arguments[2]))
    cases += try runProductionRemote(URL(fileURLWithPath: CommandLine.arguments[2]))

    for (arch, kind) in [
      ("bert", "Encoder"), ("modernbert", "Encoder"), ("whisper", "Whisper"), ("silero_vad", "Vad"),
      ("kws", "Hotword"),
    ] {
      for typed in [false, true] {
        let data = try Data(contentsOf: root.appendingPathComponent("\(arch).gguf"))
        let bad = ModelLoader(source: .bytes(bytes: data), config: config("metal"))
        do {
          if typed { _ = try bad.buildGenerative() } else { _ = try bad.build() }
          fatalError("wrong kind succeeded")
        } catch let LoadError.KindMismatch(expected, actual, architecture) {
          precondition(
            expected == "Generative" && actual == kind && architecture == arch,
            "\(arch) kind mismatch payload (typed=\(typed))")
        }
        consumed(bad)
      }
      cases.append("kind-\(arch)")
    }
    for name in ["unknown", "malformed", "backend", "invalid-backend", "assembly", "inference"] {
      for typed in [false, true] {
        let data =
          name == "unknown"
          ? try Data(contentsOf: root.appendingPathComponent("unknown.gguf"))
          : name == "assembly"
            ? try Data(contentsOf: root.appendingPathComponent("llama.gguf"))
            : name == "malformed" ? Data([0, 1]) : bytes
        var unsupportedParts = parts
        unsupportedParts.inferenceType = "future/unsupported"
        let bad = ModelLoader(
          source: name == "inference" ? .parts(parts: unsupportedParts) : .bytes(bytes: data),
          config: config(
            name == "backend" ? "metal" : name == "invalid-backend" ? "invalid-probe" : "cpu"))
        do {
          if typed { _ = try bad.buildGenerative() } else { _ = try bad.build() }
          fatalError("invalid load succeeded")
        } catch let LoadError.UnsupportedArchitecture(architecture) {
          precondition(name == "unknown" && architecture == "future_probe")
        } catch let LoadError.Source(sourceKind, detail) {
          precondition(name == "malformed" && sourceKind == "bytes" && !detail.isEmpty)
        } catch let LoadError.Assembly(backend, detail) {
          precondition(name == "backend" || name == "assembly")
          precondition(backend == (name == "backend" ? "Metal" : "Cpu") && !detail.isEmpty)
        } catch let LoadError.InvalidConfig(field, value, reason, detail) {
          precondition(name == "invalid-backend" && field == "backend")
          precondition(value == "invalid-probe" && reason == "unknown_backend" && !detail.isEmpty)
        } catch let LoadError.UnsupportedInferenceType(inferenceType) {
          precondition(name == "inference" && inferenceType == "future/unsupported")
        }
        consumed(bad)
      }
      cases.append(name)
    }
    let future = futureHandleForProbe()
    precondition(future.kind() == "future-probe" && future.asGenerative() == nil)
    cases.append("future-kind")
    let output = try JSONSerialization.data(
      withJSONObject: ["cases": cases.sorted(), "generation": result], options: [.sortedKeys])
    print(String(decoding: output, as: UTF8.self))
  }
}

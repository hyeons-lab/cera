import Foundation
import loading_native

private final class RemoteTrace: @unchecked Sendable {
  private let lock = NSLock()
  private var events: [(String, UInt64, UInt64?)] = []
  weak var sink: RemoteProgress?

  func record(_ url: String, _ bytes: UInt64, _ total: UInt64?) {
    lock.lock()
    defer { lock.unlock() }
    events.append((url, bytes, total))
  }

  func snapshot() -> [(String, UInt64, UInt64?)] {
    lock.lock()
    defer { lock.unlock() }
    return events
  }
}

private final class RemoteProgress: DownloadProgressSink, @unchecked Sendable {
  let trace: RemoteTrace
  init(_ trace: RemoteTrace) { self.trace = trace }
  func onProgress(url: String, bytesDownloaded: UInt64, totalBytes: UInt64?) {
    trace.record(url, bytesDownloaded, totalBytes)
  }
}

private func remoteSource(_ profile: String, _ root: URL) -> Source {
  switch profile {
  case "hf":
    return .huggingFace(spec: "fixture/text:Q4_K_M@release", quant: "Q8_0", strategy: "hqq")
  case "bundle": return .bundleId(id: "LiquidAI/fixture-model", quant: "Q8_0")
  case "manifest":
    return .path(path: root.appendingPathComponent("manifest/inputs/model.json").path)
  default: return .path(path: root.appendingPathComponent("directory/inputs").path)
  }
}

private func productionRemoteSource(_ profile: String, _ root: URL) -> loading_native.ModelSource {
  switch profile {
  case "hf":
    return .huggingFace(spec: "fixture/text:Q4_K_M@release", quant: "Q8_0", strategy: "hqq")
  case "bundle": return .bundleId(id: "LiquidAI/fixture-model", quant: "Q8_0")
  case "manifest":
    return .path(path: root.appendingPathComponent("manifest/inputs/model.json").path)
  default: return .path(path: root.appendingPathComponent("directory/inputs").path)
  }
}

func runProductionRemote(_ root: URL) throws -> [String] {
  for profile in ["hf", "bundle", "manifest", "directory"] {
    for typed in [false, true] {
      var loader: loading_native.ModelLoader? = {
        let repo = loading_native.BundleRepo(
          storeDir: root.appendingPathComponent("\(profile)/store").path)
        return loading_native.ModelLoader(
          source: productionRemoteSource(profile, root),
          config: EngineConfig(contextSize: 24, backend: .cpu, bundleRepo: repo))
      }()
      var handle = typed ? nil : try loader!.build()
      var model: loading_native.GenerativeModel? =
        typed ? try loader!.buildGenerative() : handle!.asGenerative()!
      productionConsumed(loader!)
      precondition(model!.engine().contextSize() == 24)
      let session = try model!.createSession(config: SessionConfig(seed: 0))
      loader = nil
      handle = nil
      model = nil
      try session.appendTokens(tokens: [0, 1])
      let output = try session.generate(
        opts: GenerateOpts(maxTokens: 1, temperature: 0, ignoreEos: true))
      precondition(output.tokens.count == 1 && session.position() == 3)
    }
  }
  return ["production-remote-sources"]
}

private func remoteLoaders(_ profile: String, _ root: URL, _ trace: RemoteTrace) -> [ModelLoader?] {
  let sink = RemoteProgress(trace)
  trace.sink = sink
  let repo = BundleRepo.withProgress(
    storeDir: root.appendingPathComponent("\(profile)/store").path, progress: sink)
  let options = LoadConfig(contextSize: 24, backend: "cpu", bundleRepo: repo)
  return (0..<2).map { _ in ModelLoader(source: remoteSource(profile, root), config: options) }
}

func runRemote(_ root: URL) throws -> [String] {
  let endpoint = try String(
    contentsOf: root.appendingPathComponent("endpoint.txt"), encoding: .utf8)
  let commit = String(repeating: "2", count: 40)
  let cacheHost = endpoint.replacingOccurrences(of: "http://", with: "").replacingOccurrences(
    of: ":", with: "_")
  for profile in ["hf", "bundle", "manifest", "directory"] {
    let trace = RemoteTrace()
    var loaders = remoteLoaders(profile, root, trace)
    precondition(trace.sink != nil)
    var retained: BundleRepo?
    var sessions: [Session] = []
    var coldEvents = 0
    for index in 0..<2 {
      var loader = loaders[index]
      loaders[index] = nil
      var handle: ModelHandle? = index == 0 ? try loader!.build() : nil
      var model = index == 0 ? handle!.asGenerative() : try loader!.buildGenerative()
      consumed(loader!)
      checkInfo(model!)
      retained = model!.repositoryForProbe()
      let store = root.appendingPathComponent("\(profile)/store").path
      precondition(retained!.storeDir() == store)
      let suffix =
        profile == "hf"
        ? "\(cacheHost)/fixture/text/resolve/\(commit)/model-Q8_0.gguf"
        : profile == "bundle"
          ? "huggingface.co/LiquidAI/LeapBundles/resolve/main/fixture-model/relative.gguf"
          : "\(cacheHost)/assets/\(profile).gguf"
      precondition(model!.files().model == "\(store)/\(suffix)")
      sessions.append(try model!.createSession())
      weak let releasedLoader = loader
      weak let releasedHandle = handle
      weak let releasedModel = model
      loader = nil
      handle = nil
      model = nil
      precondition(releasedLoader == nil && releasedHandle == nil && releasedModel == nil)
      let events = trace.snapshot()
      if index == 0 {
        coldEvents = events.count
        if profile == "bundle" {
          precondition(events.isEmpty)
        } else {
          let url =
            profile == "hf"
            ? "\(endpoint)/fixture/text/resolve/\(commit)/model-Q8_0.gguf"
            : "\(endpoint)/assets/\(profile).gguf"
          precondition(!events.isEmpty && events.allSatisfy { $0.0 == url && $0.2 == 614400 })
          precondition(events.last!.1 == 614400)
          precondition(events.contains { $0.1 > 0 && $0.1 < 614400 })
          precondition(zip(events, events.dropFirst()).allSatisfy { $0.0.1 <= $0.1.1 })
        }
      } else {
        precondition(events.count == coldEvents, "cache hit emitted progress")
      }
    }
    precondition(trace.sink != nil)
    let url = "\(endpoint)/after/\(profile).bin"
    let path = try retained!.resolveForProbe(url: url)
    let downloaded = try Data(contentsOf: URL(fileURLWithPath: path))
    precondition(downloaded == Data(String(repeating: "retained-callback", count: 32).utf8))
    let events = trace.snapshot()
    let added = Array(events.dropFirst(coldEvents))
    precondition(!added.isEmpty && added.allSatisfy { $0.0 == url && $0.2 == 544 })
    precondition(added.last!.1 == 544)
    let cached = try retained!.resolveForProbe(url: url)
    precondition(cached == path)
    precondition(trace.snapshot().count == events.count)
    retained = nil
    precondition(trace.sink == nil, "callback outlived all repository owners")
    for session in sessions {
      let result = try runSession(session)
      precondition(result["tokens"] as? [UInt32] == [0, 1, 0])
    }
  }
  for source in [
    Source.huggingFace(spec: "fixture/text", quant: nil, strategy: nil),
    .bundleId(id: "fixture-model", quant: "Q8_0"), remoteSource("manifest", root),
  ] {
    for typed in [false, true] {
      let loader = ModelLoader(source: source, config: config())
      do {
        if typed { _ = try loader.buildGenerative() } else { _ = try loader.build() }
        fatalError("missing repository succeeded")
      } catch let LoadError.Source(sourceKind, detail) {
        let expected: String
        switch source {
        case .huggingFace: expected = "hf"
        case .bundleId: expected = "bundle"
        default: expected = "path"
        }
        precondition(sourceKind == expected && !detail.isEmpty)
      }
      consumed(loader)
    }
  }
  let repo = BundleRepo(storeDir: root.appendingPathComponent("errors").path)
  for name in ["source", "kind", "assembly", "bundle-invalid"] {
    for typed in [false, true] {
      let source: Source =
        name == "bundle-invalid"
        ? .bundleId(id: "fixture-model", quant: "bad/quant")
        : .huggingFace(
          spec: "\(endpoint)/fixture/\(name)/resolve/\(commit)/model.gguf", quant: nil,
          strategy: nil)
      let options = LoadConfig(
        contextSize: 24, backend: name == "kind" ? "metal" : "cpu", bundleRepo: repo)
      let loader = ModelLoader(source: source, config: options)
      do {
        if typed { _ = try loader.buildGenerative() } else { _ = try loader.build() }
        fatalError("remote failure succeeded")
      } catch let LoadError.Source(sourceKind, detail) {
        precondition(name == "source" || name == "bundle-invalid")
        precondition(sourceKind == (name == "source" ? "hf" : "bundle") && !detail.isEmpty)
      } catch let LoadError.KindMismatch(expected, actual, architecture) {
        precondition(
          name == "kind" && expected == "Generative" && actual == "Hotword" && architecture == "kws"
        )
      } catch let LoadError.Assembly(backend, detail) {
        precondition(name == "assembly" && backend == "Cpu" && !detail.isEmpty)
      }
      consumed(loader)
    }
  }
  return [
    "native-remote-hf", "native-remote-bundle", "native-remote-manifest", "native-remote-directory",
    "native-remote-no-repo", "native-remote-hf-source", "native-remote-hf-kind",
    "native-remote-hf-assembly",
    "native-remote-bundle-invalid",
  ]
}

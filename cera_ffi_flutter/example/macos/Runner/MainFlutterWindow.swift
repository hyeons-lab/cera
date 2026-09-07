import AVFoundation
import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  private var audioPlayer: NativeAudioPlayer?

  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    let windowFrame = self.frame
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    RegisterGeneratedPlugins(registry: flutterViewController)
    audioPlayer = NativeAudioPlayer(messenger: flutterViewController.engine.binaryMessenger)

    super.awakeFromNib()
  }
}

class NativeAudioPlayer: NSObject, AVAudioPlayerDelegate {
  private let channel: FlutterMethodChannel
  private var currentPlayer: AVAudioPlayer?
  private var pendingResult: FlutterResult?

  // Streaming audio engine state
  private var audioEngine: AVAudioEngine?
  private var playerNode: AVAudioPlayerNode?
  private var audioFormat: AVAudioFormat?
  private var isStreaming = false
  private var hasStartedPlayback = false
  private var prebuffer: [AVAudioPCMBuffer] = []
  private let prebufferThreshold = 3

  init(messenger: FlutterBinaryMessenger) {
    self.channel = FlutterMethodChannel(name: "cera/audio_player", binaryMessenger: messenger)
    super.init()
    self.channel.setMethodCallHandler { [weak self] call, result in
      self?.handle(call, result: result)
    }
  }

  private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "play":
      guard let args = call.arguments as? [String: Any],
            let typedData = args["data"] as? FlutterStandardTypedData else {
        result(FlutterError(code: "INVALID_ARGUMENTS", message: "Expected data field with audio bytes", details: nil))
        return
      }
      play(data: typedData.data, result: result)

    case "startStream":
      let args = call.arguments as? [String: Any]
      let sampleRate: Double
      if let srNum = args?["sampleRate"] as? NSNumber {
        sampleRate = srNum.doubleValue
      } else {
        sampleRate = 24000.0
      }
      startStream(sampleRate: sampleRate, result: result)

    case "appendStreamChunk":
      guard let args = call.arguments as? [String: Any],
            let typedData = args["data"] as? FlutterStandardTypedData else {
        result(FlutterError(code: "INVALID_ARGUMENTS", message: "Expected data field with float32 samples", details: nil))
        return
      }
      appendStreamChunk(typedData: typedData, result: result)

    case "finishStream":
      finishStream(result: result)

    case "stopStream":
      stopStream()
      result(nil)

    case "stop":
      stop()
      result(nil)

    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func startStream(sampleRate: Double, result: @escaping FlutterResult) {
    stop()

    let engine = AVAudioEngine()
    let node = AVAudioPlayerNode()
    engine.attach(node)

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
      result(FlutterError(code: "FORMAT_ERROR", message: "Failed to initialize AVAudioFormat", details: nil))
      return
    }

    engine.connect(node, to: engine.mainMixerNode, format: format)

    do {
      engine.prepare()
      try engine.start()
      self.audioEngine = engine
      self.playerNode = node
      self.audioFormat = format
      self.isStreaming = true
      self.hasStartedPlayback = false
      self.prebuffer = []
      result(nil)
    } catch {
      result(FlutterError(code: "ENGINE_START_FAILED", message: error.localizedDescription, details: nil))
    }
  }

  private func appendStreamChunk(typedData: FlutterStandardTypedData, result: @escaping FlutterResult) {
    guard isStreaming, let node = playerNode, let format = audioFormat else {
      result(nil)
      return
    }

    let byteCount = typedData.data.count
    let sampleCount = byteCount / MemoryLayout<Float>.size
    guard sampleCount > 0 else {
      result(nil)
      return
    }

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleCount)) else {
      result(FlutterError(code: "BUFFER_ERROR", message: "Failed to allocate AVAudioPCMBuffer", details: nil))
      return
    }
    buffer.frameLength = AVAudioFrameCount(sampleCount)

    guard let channelData = buffer.floatChannelData?[0] else {
      result(FlutterError(code: "BUFFER_ERROR", message: "Missing floatChannelData", details: nil))
      return
    }

    typedData.data.withUnsafeBytes { rawBuffer in
      if let baseAddress = rawBuffer.baseAddress {
        memcpy(channelData, baseAddress, sampleCount * MemoryLayout<Float>.size)
      }
    }

    if !hasStartedPlayback && prebuffer.count < prebufferThreshold {
      prebuffer.append(buffer)
      if prebuffer.count >= prebufferThreshold {
        for buf in prebuffer {
          node.scheduleBuffer(buf, completionHandler: nil)
        }
        prebuffer.removeAll()
        node.play()
        hasStartedPlayback = true
      }
    } else {
      if !hasStartedPlayback {
        node.play()
        hasStartedPlayback = true
      }
      node.scheduleBuffer(buffer, completionHandler: nil)
    }

    result(nil)
  }

  private func finishStream(result: @escaping FlutterResult) {
    guard isStreaming, let node = playerNode else {
      result(nil)
      return
    }
    if !prebuffer.isEmpty {
      for buf in prebuffer {
        node.scheduleBuffer(buf, completionHandler: nil)
      }
      prebuffer.removeAll()
      if !hasStartedPlayback {
        node.play()
        hasStartedPlayback = true
      }
    }
    isStreaming = false
    result(nil)
  }

  private func stopStream() {
    if let node = playerNode {
      node.stop()
    }
    if let engine = audioEngine {
      engine.stop()
    }
    playerNode = nil
    audioEngine = nil
    audioFormat = nil
    isStreaming = false
    hasStartedPlayback = false
    prebuffer.removeAll()
  }

  private func play(data: Data, result: @escaping FlutterResult) {
    stop()
    do {
      let player = try AVAudioPlayer(data: data)
      player.delegate = self
      self.currentPlayer = player
      self.pendingResult = result

      if !player.play() {
        self.currentPlayer = nil
        self.pendingResult = nil
        result(FlutterError(code: "PLAY_FAILED", message: "AVAudioPlayer.play() returned false", details: nil))
      }
    } catch {
      self.currentPlayer = nil
      self.pendingResult = nil
      result(FlutterError(code: "DECODE_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func stop() {
    stopStream()
    if let player = currentPlayer {
      player.stop()
      currentPlayer = nil
    }
    if let result = pendingResult {
      pendingResult = nil
      result(false)
    }
  }

  func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
    if player == currentPlayer {
      currentPlayer = nil
      let result = pendingResult
      pendingResult = nil
      result?(flag)
    }
  }
}

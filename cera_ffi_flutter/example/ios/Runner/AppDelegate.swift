import AVFoundation
import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private var audioPlayer: NativeAudioPlayer?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    if let controller = window?.rootViewController as? FlutterViewController {
      audioPlayer = NativeAudioPlayer(messenger: controller.binaryMessenger)
    }
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
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
  private var streamGeneration = 0

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

  private func activateAudioSession() throws {
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playback, mode: .default)
    try session.setActive(true)
  }

  private func deactivateAudioSession() {
    do {
      try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    } catch {
      // Deactivation may throw if already deactivated or in background.
    }
  }

  private func startStream(sampleRate: Double, result: @escaping FlutterResult) {
    stop()
    streamGeneration += 1

    do {
      try activateAudioSession()
    } catch {
      result(FlutterError(code: "SESSION_ERROR", message: error.localizedDescription, details: nil))
      return
    }

    let engine = AVAudioEngine()
    let node = AVAudioPlayerNode()
    engine.attach(node)

    guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
      deactivateAudioSession()
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
      deactivateAudioSession()
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
      guard let basePtr = rawBuffer.baseAddress else { return }
      memcpy(channelData, basePtr, sampleCount * MemoryLayout<Float>.size)
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
    isStreaming = false
    let currentGen = streamGeneration

    let scheduleTeardown = { [weak self] in
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
        guard let self = self, self.streamGeneration == currentGen else { return }
        self.stopStream()
      }
    }

    if !prebuffer.isEmpty {
      let buffersToSchedule = prebuffer
      prebuffer.removeAll()
      let lastIndex = buffersToSchedule.count - 1
      for (i, buf) in buffersToSchedule.enumerated() {
        if i == lastIndex {
          node.scheduleBuffer(buf) {
            scheduleTeardown()
          }
        } else {
          node.scheduleBuffer(buf, completionHandler: nil)
        }
      }
      if !hasStartedPlayback {
        node.play()
        hasStartedPlayback = true
      }
    } else if hasStartedPlayback {
      if let format = audioFormat, let trailing = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 256) {
        trailing.frameLength = 256
        if let channelData = trailing.floatChannelData?[0] {
          memset(channelData, 0, 256 * MemoryLayout<Float>.size)
        }
        node.scheduleBuffer(trailing) {
          scheduleTeardown()
        }
      } else {
        scheduleTeardown()
      }
    } else {
      stopStream()
    }
    result(nil)
  }

  private func stopStream() {
    streamGeneration += 1
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
    deactivateAudioSession()
  }

  private func play(data: Data, result: @escaping FlutterResult) {
    stop()
    do {
      try activateAudioSession()
      let player = try AVAudioPlayer(data: data)
      player.delegate = self
      self.currentPlayer = player
      self.pendingResult = result

      if !player.play() {
        self.currentPlayer = nil
        self.pendingResult = nil
        deactivateAudioSession()
        result(FlutterError(code: "PLAY_FAILED", message: "AVAudioPlayer.play() returned false", details: nil))
      }
    } catch {
      self.currentPlayer = nil
      self.pendingResult = nil
      deactivateAudioSession()
      result(FlutterError(code: "DECODE_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  private func stop() {
    stopStream()
    if let player = currentPlayer {
      player.stop()
      currentPlayer = nil
      deactivateAudioSession()
    }
    if let result = pendingResult {
      pendingResult = nil
      result(false)
    }
  }

  func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
    if player == currentPlayer {
      currentPlayer = nil
      deactivateAudioSession()
      let result = pendingResult
      pendingResult = nil
      result?(flag)
    }
  }
}


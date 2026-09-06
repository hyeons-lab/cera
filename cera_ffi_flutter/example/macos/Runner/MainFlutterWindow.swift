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

    case "stop":
      stop()
      result(nil)

    default:
      result(FlutterMethodNotImplemented)
    }
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

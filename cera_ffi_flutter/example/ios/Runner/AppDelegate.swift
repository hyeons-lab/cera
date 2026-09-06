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
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playback, mode: .default)
      try session.setActive(true)
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

  private func deactivateAudioSession() {
    do {
      try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    } catch {
      // Deactivation may throw if already deactivated or in background.
    }
  }

  private func stop() {
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


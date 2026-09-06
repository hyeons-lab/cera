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

class NativeAudioPlayer: NSObject, NSSoundDelegate {
  private let channel: FlutterMethodChannel
  private var currentSound: NSSound?
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
    guard let sound = NSSound(data: data) else {
      result(FlutterError(code: "DECODE_ERROR", message: "Failed to parse audio data with NSSound", details: nil))
      return
    }
    sound.delegate = self
    self.currentSound = sound
    self.pendingResult = result

    if !sound.play() {
      self.currentSound = nil
      self.pendingResult = nil
      result(FlutterError(code: "PLAY_FAILED", message: "NSSound.play() returned false", details: nil))
    }
  }

  private func stop() {
    if let sound = currentSound {
      sound.stop()
      currentSound = nil
    }
    if let result = pendingResult {
      pendingResult = nil
      result(false)
    }
  }

  func sound(_ sound: NSSound, didFinishPlaying flag: Bool) {
    if sound == currentSound {
      currentSound = nil
      let result = pendingResult
      pendingResult = nil
      result?(flag)
    }
  }
}


import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;

import 'app_constants.dart';
import 'model_source.dart';

/// Telemetry stats for a single generation turn.
class TurnStats {
  const TurnStats({
    required this.tokens,
    required this.totalMs,
    required this.ttftMs,
    required this.tps,
    this.audioDurationSeconds,
    this.audioRtf,
  });

  final int tokens;
  final int totalMs;
  final int? ttftMs;
  final double tps;

  /// Total seconds of audio synthesized by the neural vocoder.
  final double? audioDurationSeconds;

  /// Real-Time Factor (RTF = audioDuration / generationWallTime).
  /// > 1.0 means faster than real-time speech generation.
  final double? audioRtf;
}

/// A single message turn in the conversation transcript.
class Turn {
  const Turn({
    required this.role,
    required this.text,
    this.modelName,
    this.imageBytes,
    this.imageName,
    this.audioDurationSeconds,
    this.audioSamples,
    this.stats,
    this.isGenerating = false,
    this.statusText,
  });

  final String role;
  final String text;
  final String? modelName;
  final Uint8List? imageBytes;
  final String? imageName;
  final double? audioDurationSeconds;
  final List<double>? audioSamples;
  final TurnStats? stats;
  final bool isGenerating;
  final String? statusText;

  Turn copyWith({
    String? role,
    String? text,
    String? modelName,
    Uint8List? imageBytes,
    String? imageName,
    double? audioDurationSeconds,
    List<double>? audioSamples,
    TurnStats? stats,
    bool? isGenerating,
    String? Function()? statusText,
  }) {
    return Turn(
      role: role ?? this.role,
      text: text ?? this.text,
      modelName: modelName ?? this.modelName,
      imageBytes: imageBytes ?? this.imageBytes,
      imageName: imageName ?? this.imageName,
      audioDurationSeconds: audioDurationSeconds ?? this.audioDurationSeconds,
      audioSamples: audioSamples ?? this.audioSamples,
      stats: stats ?? this.stats,
      isGenerating: isGenerating ?? this.isGenerating,
      statusText: statusText != null ? statusText() : this.statusText,
    );
  }
}

/// Record of a downloaded and locally cached model bundle.
class DownloadedModelRecord {
  const DownloadedModelRecord({
    required this.bundleName,
    required this.quant,
    required this.displayName,
  });

  final String bundleName;
  final String quant;
  final String displayName;

  String get id => '$bundleName:$quant';

  Map<String, dynamic> toJson() => {
    ModelRecordKeys.bundleName: bundleName,
    ModelRecordKeys.quant: quant,
    ModelRecordKeys.displayName: displayName,
  };

  factory DownloadedModelRecord.fromJson(Map<String, dynamic> json) {
    final bundleName = json[ModelRecordKeys.bundleName] as String;
    return DownloadedModelRecord(
      bundleName: bundleName,
      quant: json[ModelRecordKeys.quant] as String,
      displayName: json[ModelRecordKeys.displayName] as String? ?? bundleName,
    );
  }
}

/// Voice interaction and generation modes when an audio-capable model is loaded.
enum AudioChatMode {
  /// Speech to Text (ASR): Transcribes recorded or uploaded audio into text.
  speechToText,

  /// Voice Chat (Interleaved): Model generates conversational text and spoken audio.
  interleaved,

  /// Text to Speech (TTS): Model speaks and synthesizes the user text prompt directly using its neural vocoder.
  textToSpeech,

  /// Text Only: Generates standard text responses without vocoder speech synthesis.
  textOnly,
}

/// Settings state for engine inference and vision/audio preprocessing.
class ChatSettings {
  const ChatSettings({
    this.backend = CeraBackend.auto,
    this.turboQuant = false,
    this.maxImageLongSize = ChatDefaults.maxImageLongSize,
    this.audioChatMode = AudioChatMode.interleaved,
    this.chatVoice = '',
    this.ttsStudioVoice = '',
    this.temperature,
    this.topP,
    this.topK,
    this.maxTokens = ChatDefaults.maxTokens,
  });

  /// Which compute backend to run on.
  final CeraBackend backend;

  /// Whether TurboQuant KV-cache compression is enabled.
  /// Defaults to false (cera default).
  final bool turboQuant;

  /// Maximum long side resolution for image inputs to the vision encoder.
  /// null means use model's native resolution limit. Defaults to 256 for fast inference.
  final int? maxImageLongSize;

  /// Mode for audio generation when an audio-capable model is loaded.
  final AudioChatMode audioChatMode;

  /// Voice persona description for Voice Chat mode.
  final String chatVoice;

  /// Voice persona description for dedicated TTS Studio.
  final String ttsStudioVoice;

  /// Sampling temperature override.
  /// null uses the engine/bundle default (0.0 greedy on WebGPU text, 0.7 on audio).
  /// 0.0 forces deterministic greedy decoding; > 0.0 enables stochastic sampling.
  final double? temperature;

  /// Top-p (nucleus) sampling threshold override.
  final double? topP;

  /// Top-k sampling cutoff override.
  final int? topK;

  /// Maximum tokens to generate per turn. Defaults to 512.
  final int maxTokens;

  /// Short human-readable label for the active sampling mode.
  String get samplingLabel {
    final temp = temperature;
    if (temp == null) return AppStrings.samplingAuto;
    if (temp <= 0.0) return AppStrings.samplingGreedy;
    return 'T=${temp.toStringAsFixed(2)}';
  }

  /// Converts settings to engine open options.
  CeraOptions get ceraOptions => CeraOptions(
    backend: backend,
    kvCompression: turboQuant
        ? CeraKvCompression.turboQuant
        : CeraKvCompression.none,
    web: const CeraWebAssets(
      workerUrl: ModelFormat.workerUrl,
      moduleUrl: ModelFormat.moduleUrl,
    ),
  );

  ChatSettings copyWith({
    CeraBackend? backend,
    bool? turboQuant,
    int? maxImageLongSize,
    bool clearMaxImageLongSize = false,
    AudioChatMode? audioChatMode,
    String? chatVoice,
    String? ttsStudioVoice,
    double? temperature,
    bool clearTemperature = false,
    double? topP,
    bool clearTopP = false,
    int? topK,
    bool clearTopK = false,
    int? maxTokens,
  }) {
    return ChatSettings(
      backend: backend ?? this.backend,
      turboQuant: turboQuant ?? this.turboQuant,
      maxImageLongSize: clearMaxImageLongSize
          ? null
          : (maxImageLongSize ?? this.maxImageLongSize),
      audioChatMode: audioChatMode ?? this.audioChatMode,
      chatVoice: chatVoice ?? this.chatVoice,
      ttsStudioVoice: ttsStudioVoice ?? this.ttsStudioVoice,
      temperature: clearTemperature ? null : (temperature ?? this.temperature),
      topP: clearTopP ? null : (topP ?? this.topP),
      topK: clearTopK ? null : (topK ?? this.topK),
      maxTokens: maxTokens ?? this.maxTokens,
    );
  }
}

/// Main UI mode for the Cera demo application.
enum AppUIMode {
  /// Conversational multimodal chat.
  chat,

  /// Dedicated on-device neural Text-to-Speech (TTS) Studio.
  ttsStudio,
}

/// Immutable MVI State representing the entire Chat UI and engine state.
class ChatState {
  const ChatState({
    this.loadedModel,
    this.status = AppStrings.initialStatus,
    this.isLoading = false,
    this.isGenerating = false,
    this.downloadFraction,
    this.turns = const [],
    this.pendingImageBytes,
    this.pendingImageName,
    this.capabilities,
    this.audioProfile,
    this.backend,
    this.downloadedModels = const [],
    this.settings = const ChatSettings(),
    this.uiMode = AppUIMode.chat,
  });

  final LoadedModel? loadedModel;
  final String status;
  final bool isLoading;
  final bool isGenerating;
  final double? downloadFraction;
  final List<Turn> turns;
  final Uint8List? pendingImageBytes;
  final String? pendingImageName;
  final CeraCapabilities? capabilities;

  /// What the loaded model needs to be told to speak (system prompts, voices,
  /// sample text), resolved by cera. Null until a model is loaded.
  final CeraAudioProfile? audioProfile;
  final String? backend;
  final List<DownloadedModelRecord> downloadedModels;
  final ChatSettings settings;
  final AppUIMode uiMode;

  bool get hasModel => loadedModel != null;

  bool get isBusy => isLoading || isGenerating;
  bool get canAttachImage =>
      hasModel && (capabilities?.imageIn ?? false) && !isBusy;
  bool get canAttachAudio =>
      hasModel && (capabilities?.audioIn ?? false) && !isBusy;

  ChatState copyWith({
    LoadedModel? Function()? loadedModel,
    String? status,
    bool? isLoading,
    bool? isGenerating,
    double? Function()? downloadFraction,
    List<Turn>? turns,
    Uint8List? Function()? pendingImageBytes,
    String? Function()? pendingImageName,
    CeraCapabilities? Function()? capabilities,
    CeraAudioProfile? Function()? audioProfile,
    String? Function()? backend,
    List<DownloadedModelRecord>? downloadedModels,
    ChatSettings? settings,
    AppUIMode? uiMode,
  }) {
    return ChatState(
      loadedModel: loadedModel != null ? loadedModel() : this.loadedModel,
      status: status ?? this.status,
      isLoading: isLoading ?? this.isLoading,
      isGenerating: isGenerating ?? this.isGenerating,
      downloadFraction: downloadFraction != null
          ? downloadFraction()
          : this.downloadFraction,
      turns: turns ?? this.turns,
      pendingImageBytes: pendingImageBytes != null
          ? pendingImageBytes()
          : this.pendingImageBytes,
      pendingImageName: pendingImageName != null
          ? pendingImageName()
          : this.pendingImageName,
      capabilities: capabilities != null ? capabilities() : this.capabilities,
      audioProfile: audioProfile != null ? audioProfile() : this.audioProfile,
      backend: backend != null ? backend() : this.backend,
      downloadedModels: downloadedModels ?? this.downloadedModels,
      settings: settings ?? this.settings,
      uiMode: uiMode ?? this.uiMode,
    );
  }
}

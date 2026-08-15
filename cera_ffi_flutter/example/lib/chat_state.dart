import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'model_source.dart';

/// Telemetry stats for a single generation turn.
class TurnStats {
  const TurnStats({
    required this.tokens,
    required this.totalMs,
    required this.ttftMs,
    required this.tps,
  });

  final int tokens;
  final int totalMs;
  final int? ttftMs;
  final double tps;
}

/// A single message turn in the conversation transcript.
class Turn {
  Turn({
    required this.role,
    required this.text,
    this.modelName,
    this.imageBytes,
    this.imageName,
    this.audioDurationSeconds,
    this.stats,
    this.isGenerating = false,
    this.statusText,
  });

  final String role;
  String text;
  final String? modelName;
  final Uint8List? imageBytes;
  final String? imageName;
  final double? audioDurationSeconds;
  TurnStats? stats;
  bool isGenerating;
  String? statusText;
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
    'bundleName': bundleName,
    'quant': quant,
    'displayName': displayName,
  };

  factory DownloadedModelRecord.fromJson(Map<String, dynamic> json) =>
      DownloadedModelRecord(
        bundleName: json['bundleName'] as String,
        quant: json['quant'] as String,
        displayName:
            json['displayName'] as String? ?? (json['bundleName'] as String),
      );
}

/// Settings state for engine inference and vision/audio preprocessing.
class ChatSettings {
  const ChatSettings({
    this.backend = CeraBackend.auto,
    this.turboQuant = false,
    this.maxImageLongSize = 256,
  });

  /// Which compute backend to run on.
  final CeraBackend backend;

  /// Whether TurboQuant KV-cache compression is enabled.
  /// Defaults to false (cera default).
  final bool turboQuant;

  /// Maximum long side resolution for image inputs to the vision encoder.
  /// null means use model's native resolution limit. Defaults to 256 for fast inference.
  final int? maxImageLongSize;

  /// Converts settings to engine open options.
  CeraOptions get ceraOptions =>
      CeraOptions(backend: backend, turboQuant: turboQuant);

  ChatSettings copyWith({
    CeraBackend? backend,
    bool? turboQuant,
    int? Function()? maxImageLongSize,
  }) {
    return ChatSettings(
      backend: backend ?? this.backend,
      turboQuant: turboQuant ?? this.turboQuant,
      maxImageLongSize: maxImageLongSize != null
          ? maxImageLongSize()
          : this.maxImageLongSize,
    );
  }
}

/// Immutable MVI State representing the entire Chat UI and engine state.
class ChatState {
  const ChatState({
    this.loadedModel,
    this.status = 'Download a published model, or open a .gguf, to start.',
    this.isLoading = false,
    this.isGenerating = false,
    this.downloadFraction,
    this.turns = const [],
    this.pendingImageBytes,
    this.pendingImageName,
    this.capabilities,
    this.backend,
    this.downloadedModels = const [],
    this.settings = const ChatSettings(),
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
  final String? backend;
  final List<DownloadedModelRecord> downloadedModels;
  final ChatSettings settings;

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
    String? Function()? backend,
    List<DownloadedModelRecord>? downloadedModels,
    ChatSettings? settings,
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
      backend: backend != null ? backend() : this.backend,
      downloadedModels: downloadedModels ?? this.downloadedModels,
      settings: settings ?? this.settings,
    );
  }
}

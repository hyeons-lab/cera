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
    this.imageBytes,
    this.imageName,
    this.stats,
    this.isGenerating = false,
    this.statusText,
  });

  final String role;
  String text;
  final Uint8List? imageBytes;
  final String? imageName;
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

  bool get hasModel => loadedModel != null;
  bool get isBusy => isLoading || isGenerating;
  bool get canAttachImage =>
      hasModel && (capabilities?.imageIn ?? false) && !isBusy;
  bool get canSend =>
      hasModel && !isBusy && (!turns.any((t) => t.isGenerating));

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
    );
  }
}

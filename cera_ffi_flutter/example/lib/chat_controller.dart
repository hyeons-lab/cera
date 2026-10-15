import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;
import 'package:shared_preferences/shared_preferences.dart';

import 'app_constants.dart';
import 'chat_intent.dart';
import 'chat_state.dart';
import 'model_source.dart';
import 'services/audio_player_service.dart';
import 'services/bundle_size_probe.dart';

/// MVI Controller / Store managing state transitions, Cera engine lifecycle,
/// download progress, and token generation.
class ChatController extends ValueNotifier<ChatState> {
  ChatController({Future<String?> Function()? defaultStoreDir})
    : _defaultStoreDir = defaultStoreDir ?? (() async => null),
      super(const ChatState()) {
    debugPrint(
      '[cera:chat:version] ChatController ${AppStrings.appVersion} (build: $_kAppRevisionBadge)',
    );
    _loadDownloadedRecords();
  }

  static const _kAppRevisionBadge = 'rev28-hexagon-backend';

  /// Fixed output sampling rate for Cera vocoder neural audio models (24 kHz).
  static const int vocoderSampleRate = ChatDefaults.vocoderSampleRate;

  final Future<String?> Function() _defaultStoreDir;
  final AudioPlayerService _audioPlayer = AudioPlayerService();
  Cera? _ceraEngine;
  StreamSubscription<String>? _generationSub;
  Completer<void>? _generationCompleter;
  SessionPhase _sessionPhase = SessionPhase.idle;
  bool _disposed = false;

  /// Active session coordinator phase.
  SessionPhase get sessionPhase => _sessionPhase;

  /// Audio player service instance.
  AudioPlayerService get audioPlayer => _audioPlayer;

  /// Loads locally tracked downloaded model records from persistent storage.
  Future<void> _loadDownloadedRecords() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list =
          prefs.getStringList(PrefKeys.downloadedModels) ?? const <String>[];
      final records = <DownloadedModelRecord>[];
      for (final s in list) {
        try {
          final m = jsonDecode(s) as Map<String, dynamic>;
          records.add(DownloadedModelRecord.fromJson(m));
        } catch (_) {}
      }
      if (!_disposed) {
        value = value.copyWith(downloadedModels: records);
      }
    } catch (_) {}
  }

  /// Dispatch an intent to be processed by the controller.
  Future<void> dispatch(ChatIntent intent) async {
    if (_disposed) return;

    switch (intent) {
      case RestoreLastModelIntent():
        await _onRestoreLastModel();
      case LoadBundleIntent():
        await _onLoadBundle(intent);
      case LoadLocalModelIntent():
        await _onLoadLocalModel(intent);
      case UnloadModelIntent():
        await _onUnloadModel();
      case SendMessageIntent():
        await _onSendMessage(intent.prompt);
      case SendAudioPromptIntent():
        await _onSendAudioPrompt(intent);
      case StopGenerationIntent():
        await _onStopGeneration();
      case AttachImageIntent():
        _onAttachImage(intent.bytes, intent.name);
      case ClearAttachedImageIntent():
        _onClearAttachedImage();
      case ClearTranscriptIntent():
        await _onClearTranscript();
      case UpdateSettingsIntent():
        await _onUpdateSettings(intent);
      case SetUIModeIntent():
        var newSettings = value.settings;
        if (intent.mode == AppUIMode.ttsStudio) {
          if ((value.capabilities?.audioOut ?? false) &&
              newSettings.audioChatMode != AudioChatMode.textToSpeech) {
            newSettings = newSettings.copyWith(
              audioChatMode: AudioChatMode.textToSpeech,
            );
          }
        }
        value = value.copyWith(uiMode: intent.mode, settings: newSettings);
    }
  }

  /// Explicitly and cleanly unloads the currently active Cera model and frees
  /// all underlying memory and native resources.
  Future<void> _unloadCurrentModel() async {
    _sessionPhase = SessionPhase.idle;
    _loadSessionId++;
    _generationId++;
    _audioPlayer.stop();
    await _generationSub?.cancel();
    _generationSub = null;
    final completer = _generationCompleter;
    _generationCompleter = null;
    if (completer != null && !completer.isCompleted) {
      completer.complete();
    }
    final old = _ceraEngine;
    _ceraEngine = null;
    try {
      await old?.close();
    } catch (err) {
      debugPrint('cera: error closing previous engine: $err');
    }
  }

  Future<void> _onUnloadModel() async {
    await _unloadCurrentModel();
    if (_disposed) return;
    value = value.copyWith(
      loadedModel: () => null,
      capabilities: () => null,
      audioProfile: () => null,
      backend: () => null,
      status: AppStrings.modelUnloadedStatus,
      isLoading: false,
      isGenerating: false,
      downloadFraction: () => null,
    );
  }

  Future<void> _onRestoreLastModel() async {
    // Avoid race conditions if a load is already initiated or completed.
    if (value.isLoading || _ceraEngine != null || _disposed) return;

    try {
      final prefs = await SharedPreferences.getInstance();
      if (_disposed || value.isLoading || _ceraEngine != null) return;

      final backendStr = prefs.getString(PrefKeys.backend);
      final backend = switch (backendStr) {
        PrefValues.backendCpu => CeraBackend.cpu,
        PrefValues.backendGpu => CeraBackend.gpu,
        PrefValues.backendHexagon => CeraBackend.hexagon,
        _ => CeraBackend.auto,
      };
      final turboQuant = prefs.getBool(PrefKeys.turboQuant) ?? false;
      final maxImageDim = prefs.getInt(PrefKeys.maxImageDim);
      final audioModeStr = prefs.getString(PrefKeys.audioChatMode);
      final audioMode = switch (audioModeStr) {
        PrefValues.audioAsr ||
        PrefValues.audioSpeechToText => AudioChatMode.speechToText,
        PrefValues.audioTts ||
        PrefValues.audioTextToSpeech => AudioChatMode.textToSpeech,
        PrefValues.audioTextOnly => AudioChatMode.textOnly,
        _ => AudioChatMode.interleaved,
      };
      // Empty means "the loaded model's own default voice": which voices exist
      // is the model's business (see `CeraAudioProfile`), not the app's.
      final chatVoice =
          prefs.getString(PrefKeys.chatVoice) ??
          prefs.getString(PrefKeys.legacyTtsVoice) ??
          '';
      final ttsStudioVoice = prefs.getString(PrefKeys.ttsStudioVoice) ?? '';
      final savedTemp = prefs.getDouble(PrefKeys.temperature);
      final savedTopP = prefs.getDouble(PrefKeys.topP);
      final savedTopK = prefs.getInt(PrefKeys.topK);
      final savedMaxTokens =
          prefs.getInt(PrefKeys.maxTokens) ?? ChatDefaults.maxTokens;
      final restoredSettings = ChatSettings(
        backend: backend,
        turboQuant: turboQuant,
        maxImageLongSize: maxImageDim == 0
            ? null
            : (maxImageDim ?? ChatDefaults.maxImageLongSize),
        audioChatMode: audioMode,
        chatVoice: chatVoice,
        ttsStudioVoice: ttsStudioVoice,
        temperature: savedTemp,
        topP: savedTopP,
        topK: savedTopK,
        maxTokens: savedMaxTokens.clamp(
          ChatDefaults.minMaxTokens,
          ChatDefaults.maxMaxTokens,
        ),
      );
      value = value.copyWith(settings: restoredSettings);

      final bundleName = prefs.getString(PrefKeys.lastBundleName);
      final quant = prefs.getString(PrefKeys.lastBundleQuant);
      if (bundleName != null && quant != null) {
        final displayName = ModelFormat.stripGgufSuffix(bundleName);
        final label = '$displayName${AppStrings.metaSeparator}$quant';

        await _load(
          modelSource: BundleModelSource(
            name: label,
            bundleName: bundleName,
            quant: quant,
            getStoreDir: _defaultStoreDir,
          ),
          openFn: (onProgress) async => Cera.openBundle(
            bundleName,
            quant,
            storeDir: await _defaultStoreDir(),
            options: restoredSettings.ceraOptions,
            onProgress: onProgress,
          ),
          label: label,
          isBundle: true,
          bundleName: bundleName,
          quant: quant,
          displayName: displayName,
        );
      }
    } catch (err) {
      debugPrint('cera: could not restore last model: $err');
      try {
        final prefs = await SharedPreferences.getInstance();
        await prefs.remove(PrefKeys.lastBundleName);
        await prefs.remove(PrefKeys.lastBundleQuant);
      } catch (_) {}
    }
  }

  Future<void> _onUpdateSettings(UpdateSettingsIntent intent) async {
    final current = value.settings;
    final updated = current.copyWith(
      backend: intent.backend,
      turboQuant: intent.turboQuant,
      maxImageLongSize: intent.maxImageLongSize,
      clearMaxImageLongSize: intent.clearMaxImageLongSize,
      audioChatMode: intent.audioChatMode,
      chatVoice: intent.clearChatVoice ? '' : intent.chatVoice,
      ttsStudioVoice: intent.ttsStudioVoice,
      temperature: intent.temperature,
      clearTemperature: intent.clearTemperature,
      topP: intent.topP,
      clearTopP: intent.clearTopP,
      topK: intent.topK,
      clearTopK: intent.clearTopK,
      maxTokens: intent.maxTokens,
    );

    value = value.copyWith(settings: updated);

    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(PrefKeys.backend, updated.backend.name);
      await prefs.setBool(PrefKeys.turboQuant, updated.turboQuant);
      await prefs.setInt(PrefKeys.maxImageDim, updated.maxImageLongSize ?? 0);
      await prefs.setString(
        PrefKeys.audioChatMode,
        switch (updated.audioChatMode) {
          AudioChatMode.speechToText => PrefValues.audioAsr,
          AudioChatMode.textToSpeech => PrefValues.audioTts,
          AudioChatMode.textOnly => PrefValues.audioTextOnly,
          AudioChatMode.interleaved => PrefValues.audioInterleaved,
        },
      );
      await prefs.setString(PrefKeys.chatVoice, updated.chatVoice);
      await prefs.setString(PrefKeys.ttsStudioVoice, updated.ttsStudioVoice);
      final newTemperature = updated.temperature;
      if (newTemperature != null) {
        await prefs.setDouble(PrefKeys.temperature, newTemperature);
      } else {
        await prefs.remove(PrefKeys.temperature);
      }
      final newTopP = updated.topP;
      if (newTopP != null) {
        await prefs.setDouble(PrefKeys.topP, newTopP);
      } else {
        await prefs.remove(PrefKeys.topP);
      }
      final newTopK = updated.topK;
      if (newTopK != null) {
        await prefs.setInt(PrefKeys.topK, newTopK);
      } else {
        await prefs.remove(PrefKeys.topK);
      }
      await prefs.setInt(PrefKeys.maxTokens, updated.maxTokens);
    } catch (err) {
      debugPrint('cera: error persisting settings: $err');
    }

    // If backend or TurboQuant changed and a model is currently loaded and idle, reload it.
    final backendChanged =
        intent.backend != null && intent.backend != current.backend;
    final turboChanged =
        intent.turboQuant != null && intent.turboQuant != current.turboQuant;
    if ((backendChanged || turboChanged) &&
        value.loadedModel != null &&
        !value.isBusy) {
      final model = value.loadedModel;
      if (model is BundleModelSource) {
        await _onLoadBundle(
          LoadBundleIntent(
            bundleName: model.bundleName,
            quant: model.quant,
            displayName: model.displayName ?? model.bundleName,
            storeDir: model.getStoreDir ?? _defaultStoreDir,
          ),
        );
      } else if (model is ModelSource) {
        await _onLoadLocalModel(LoadLocalModelIntent(model));
      }
    }
  }

  Future<void> _onLoadBundle(LoadBundleIntent intent) async {
    debugPrint(
      '[cera:flutter] LoadBundleIntent: bundleName=${intent.bundleName}, quant=${intent.quant}, displayName=${intent.displayName}',
    );
    final label =
        '${intent.displayName}${AppStrings.metaSeparator}${intent.quant}';
    final bundleSource = BundleModelSource(
      name: label,
      bundleName: intent.bundleName,
      quant: intent.quant,
      displayName: intent.displayName,
      getStoreDir: intent.storeDir,
    );

    await _load(
      modelSource: bundleSource,
      openFn: (onProgress) async => Cera.openBundle(
        intent.bundleName,
        intent.quant,
        storeDir: await intent.storeDir(),
        options: value.settings.ceraOptions,
        onProgress: onProgress,
      ),
      label: label,
      isBundle: true,
      bundleName: intent.bundleName,
      quant: intent.quant,
      displayName: intent.displayName,
    );
  }

  Future<void> _onLoadLocalModel(LoadLocalModelIntent intent) async {
    await _load(
      modelSource: intent.source,
      openFn: (_) => intent.source.open(options: value.settings.ceraOptions),
      label: intent.source.name,
      isBundle: false,
    );
  }

  int _loadSessionId = 0;

  Future<void> _load({
    required LoadedModel modelSource,
    required Future<Cera> Function(void Function(CeraDownload) onProgress)
    openFn,
    required String label,
    bool isBundle = false,
    String? bundleName,
    String? quant,
    String? displayName,
  }) async {
    // 1. Unload any existing model first to free memory / weights.
    await _unloadCurrentModel();

    if (_disposed) return;
    final loadId = ++_loadSessionId;

    value = value.copyWith(
      isLoading: true,
      status: 'Loading $label...',
      downloadFraction: () => null,
      pendingImageBytes: () => null,
      pendingImageName: () => null,
    );

    Map<String, int> knownFileSizes = {};
    if (isBundle && bundleName != null && quant != null) {
      try {
        final cleanQuant = quant.split(' ').first;
        knownFileSizes = await probeBundleFileSizes(bundleName, cleanQuant);
      } catch (_) {}
    }

    final uniqueSizes = <String, int>{};
    for (final entry in knownFileSizes.entries) {
      final key = entry.key.split('/').last;
      if (!key.endsWith(ModelFormat.jsonExtension)) {
        uniqueSizes[key] = entry.value;
      }
    }
    final int expectedTotalBytes = uniqueSizes.values.fold<int>(
      0,
      (a, b) => a + b,
    );

    final Map<String, int> fileDownloaded = {};
    final Map<String, int> fileTotals = {};
    double maxFraction = 0.0;

    try {
      final cera = await openFn((progress) {
        if (_disposed || _loadSessionId != loadId) return;
        final fileName = progress.url.split('/').last;

        // Skip tiny manifest JSON from triggering 100% false spikes
        if (fileName.endsWith(ModelFormat.jsonExtension) &&
            expectedTotalBytes > 0) {
          return;
        }

        fileDownloaded[fileName] = progress.bytesDownloaded;
        final progressTotal = progress.totalBytes;
        final resolvedTotal = (progressTotal != null && progressTotal > 0)
            ? progressTotal
            : (knownFileSizes[fileName] ?? knownFileSizes[progress.url]);
        if (resolvedTotal != null) {
          fileTotals[fileName] = resolvedTotal;
        }

        int totalBytes = expectedTotalBytes;
        if (totalBytes <= 0) {
          totalBytes = fileTotals.values.fold<int>(0, (a, b) => a + b);
        } else {
          for (final entry in fileTotals.entries) {
            if (!uniqueSizes.containsKey(entry.key)) {
              totalBytes += entry.value;
            }
          }
        }

        final totalDownloaded = fileDownloaded.values.fold<int>(
          0,
          (sum, b) => sum + b,
        );

        double? fraction;
        if (totalBytes > 0) {
          final computed = (totalDownloaded / totalBytes).clamp(0.0, 1.0);
          if (computed >= maxFraction) {
            maxFraction = computed;
          }
          fraction = maxFraction;
        }

        final downloadedMb = (totalDownloaded / 1024 / 1024).toStringAsFixed(1);
        final pctStr = fraction != null
            ? '${(fraction * 100).toStringAsFixed(0)}%'
            : '$downloadedMb MB';

        final totalSuffix = totalBytes > 0
            ? ' / ${(totalBytes / 1024 / 1024).toStringAsFixed(1)} MB'
            : ' MB';
        final bytesInfo = '$downloadedMb$totalSuffix';
        final fileSuffix = fileName.isNotEmpty
            ? '${AppStrings.metaSeparator}$fileName'
            : '';

        value = value.copyWith(
          downloadFraction: () => fraction,
          status:
              'Downloading $label${AppStrings.metaSeparator}$pctStr ($bytesInfo)$fileSuffix',
        );
      });

      if (_disposed || _loadSessionId != loadId) {
        await cera.close();
        return;
      }

      _ceraEngine = cera;
      _sessionPhase = SessionPhase.idle;
      final visionTag = cera.capabilities.imageIn ? AppStrings.tagVision : '';
      final voiceTag = voiceTagFor(cera.capabilities);

      final effectiveSettings = alignAudioMode(
        value.settings,
        cera.capabilities,
      );
      final effectiveProfile = resolveAudioProfile(
        cera.audioProfile,
        modelIdentity: bundleName ?? displayName ?? modelSource.name,
        capabilities: cera.capabilities,
      );

      value = value.copyWith(
        loadedModel: () => modelSource,
        capabilities: () => cera.capabilities,
        audioProfile: () => effectiveProfile,
        backend: () => cera.backend,
        status:
            '${modelSource.name}${AppStrings.metaSeparator}${cera.backend}$visionTag$voiceTag',
        isLoading: false,
        downloadFraction: () => null,
        settings: effectiveSettings,
      );

      // Persist preferences
      try {
        final prefs = await SharedPreferences.getInstance();
        if (isBundle && bundleName != null && quant != null) {
          await prefs.setString(PrefKeys.lastBundleName, bundleName);
          await prefs.setString(PrefKeys.lastBundleQuant, quant);
          await _saveDownloadedRecord(
            bundleName,
            quant,
            displayName ?? bundleName,
          );
        } else {
          await prefs.remove(PrefKeys.lastBundleName);
          await prefs.remove(PrefKeys.lastBundleQuant);
        }
      } catch (_) {}
    } catch (err, stack) {
      debugPrint('cera: model failed to load: $err\n$stack');
      if (!_disposed && _loadSessionId == loadId) {
        value = value.copyWith(
          isLoading: false,
          loadedModel: () => null,
          backend: () => null,
          capabilities: () => null,
          audioProfile: () => null,
          downloadFraction: () => null,
          status: 'Failed to load: $err',
        );
      }
    }
  }

  Future<void> _saveDownloadedRecord(
    String bundleName,
    String quant,
    String displayName,
  ) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList(PrefKeys.downloadedModels) ?? <String>[];
      final exists = list.any((item) {
        try {
          final map = jsonDecode(item) as Map<String, dynamic>;
          return map[ModelRecordKeys.bundleName] == bundleName &&
              map[ModelRecordKeys.quant] == quant;
        } on FormatException {
          return false;
        }
      });
      if (!exists) {
        final record = jsonEncode(
          DownloadedModelRecord(
            bundleName: bundleName,
            quant: quant,
            displayName: displayName,
          ).toJson(),
        );
        list.add(record);
        await prefs.setStringList(PrefKeys.downloadedModels, list);
        await _loadDownloadedRecords();
      }
    } catch (_) {}
  }

  String _activeModelLabel() {
    final loaded = value.loadedModel;
    if (loaded is BundleModelSource) {
      return '${loaded.bundleName} (${loaded.quant})';
    }
    return loaded?.name ?? AppStrings.unknownModel;
  }

  Future<void> _onSendMessage(String prompt) async {
    final cera = _ceraEngine;
    final imageBytes = value.pendingImageBytes;
    final imageName = value.pendingImageName;

    if ((prompt.trim().isEmpty && imageBytes == null) ||
        cera == null ||
        value.isBusy ||
        _sessionPhase == SessionPhase.unusable ||
        _disposed) {
      return;
    }

    final generationId = ++_generationId;
    final userTurn = Turn(
      role: TurnRole.user,
      text: prompt.trim(),
      imageBytes: imageBytes,
      imageName: imageName,
    );

    final audioOut = value.capabilities?.audioOut ?? false;
    final audioMode = audioOut
        ? value.settings.audioChatMode
        : AudioChatMode.textOnly;
    final isTts = audioMode == AudioChatMode.textToSpeech;
    final isInterleaved = audioMode == AudioChatMode.interleaved;

    final String initialStatus;
    if (imageBytes != null) {
      initialStatus = AppStrings.statusAnalyzingImage;
    } else if (isTts) {
      initialStatus = AppStrings.statusSynthesizingSpeech;
    } else if (isInterleaved) {
      initialStatus = AppStrings.statusThinkingAndSpeaking;
    } else {
      initialStatus = AppStrings.statusThinking;
    }

    final rawPromptText = prompt.trim();
    final promptText = isTts
        ? normalizeSpeechText(rawPromptText)
        : rawPromptText;
    final framedPromptText = promptText.isNotEmpty
        ? promptText
        : (imageBytes != null ? AppStrings.defaultImagePrompt : '');

    final assistantTurn = Turn(
      role: TurnRole.assistant,
      text: isTts ? framedPromptText : '',
      modelName: value.loadedModel?.name,
      isGenerating: true,
      statusText: initialStatus,
    );

    value = value.copyWith(
      turns: [...value.turns, userTurn, assistantTurn],
      isGenerating: true,
      pendingImageBytes: () => null,
      pendingImageName: () => null,
    );

    final modelLabel = _activeModelLabel();

    debugPrint(
      '[cera:chat] Submitting user message for "$modelLabel": "$framedPromptText" '
      '(image: ${imageBytes != null ? "${imageBytes.length} bytes" : "none"}, audioMode: ${audioMode.name})',
    );

    final isInitial = _sessionPhase == SessionPhase.idle;
    String formattedPrompt;
    if (isInitial) {
      final systemPrompt = systemPromptFor(
        settings: value.settings,
        uiMode: value.uiMode,
        isAudioPrompt: false,
        profile: value.audioProfile ?? cera.audioProfile,
      );
      final messages = <CeraMessage>[
        if (systemPrompt != null) CeraMessage.system(systemPrompt),
        CeraMessage.user(framedPromptText),
      ];
      try {
        formattedPrompt = await cera.applyChatTemplate(messages);
      } catch (_) {
        formattedPrompt = framedPromptText;
      }
    } else {
      try {
        var delta = await cera.applyChatTemplate([
          CeraMessage.user(framedPromptText),
        ]);
        for (final prefix in ChatTemplateMarkers.bosPrefixes) {
          if (delta.startsWith(prefix)) {
            delta = delta.substring(prefix.length);
            break;
          }
        }
        if (delta.startsWith('\n')) {
          delta = delta.substring(1);
        }
        // Seal the uncommitted assistant turn with this template's delimiter.
        // Match the user header, not marker text that may occur in user content.
        if (delta.startsWith(ChatTemplateMarkers.imStartUser)) {
          formattedPrompt = '${ChatTemplateMarkers.imEndPrefix}$delta';
        } else if (delta.startsWith(ChatTemplateMarkers.llama3UserHeader)) {
          formattedPrompt = '${ChatTemplateMarkers.llama3EotPrefix}$delta';
        } else if (delta.startsWith(ChatTemplateMarkers.gemmaUserTurn)) {
          formattedPrompt = '${ChatTemplateMarkers.gemmaEndTurnPrefix}$delta';
        } else {
          throw StateError('Unsupported continuation template');
        }
      } catch (_) {
        if (_disposed || generationId != _generationId) return;
        _updateLastTurn(
          (t) => t.copyWith(
            isGenerating: false,
            statusText: () => null,
            text: AppStrings.unsupportedContinuationMessage,
          ),
        );
        value = value.copyWith(isGenerating: false);
        return;
      }
    }
    if (_disposed || generationId != _generationId) return;

    if (imageBytes != null) {
      try {
        _updateLastTurn(
          (t) => t.copyWith(statusText: () => AppStrings.statusEncodingImage),
        );
        final maxLong = value.settings.maxImageLongSize;
        debugPrint(
          '[cera:chat] Encoding image patches with maxLongSize ${maxLong ?? "native"}...',
        );
        await cera.appendImage(imageBytes, maxLongSize: maxLong);
        debugPrint(
          '[cera:chat] Image successfully encoded and seeded into KV cache',
        );
        if (_disposed || generationId != _generationId) return;
        if (!value.isGenerating) {
          _updateLastTurn(
            (t) => t.copyWith(isGenerating: false, statusText: () => null),
          );
          return;
        }
        _updateLastTurn(
          (t) =>
              t.copyWith(statusText: () => AppStrings.statusGeneratingResponse),
        );
      } catch (err) {
        if (_disposed || generationId != _generationId) return;
        _updateLastTurn(
          (t) => t.copyWith(
            isGenerating: false,
            statusText: () => null,
            text: 'Failed to process image: $err',
          ),
        );
        value = value.copyWith(isGenerating: false);
        return;
      }
    }

    if (_disposed || generationId != _generationId || !value.isGenerating) {
      return;
    }
    await _runGeneration(formattedPrompt);
  }

  Future<void> _onSendAudioPrompt(SendAudioPromptIntent intent) async {
    final cera = _ceraEngine;
    if (intent.pcmSamples.isEmpty ||
        cera == null ||
        value.isBusy ||
        _sessionPhase == SessionPhase.unusable ||
        _disposed) {
      return;
    }

    final durationSec = intent.sampleRate > 0
        ? intent.pcmSamples.length / intent.sampleRate
        : 0.0;
    final promptText = intent.prompt.trim();
    final isAsr = value.settings.audioChatMode == AudioChatMode.speechToText;

    final generationId = ++_generationId;
    final userTurn = Turn(
      role: TurnRole.user,
      text: promptText,
      audioDurationSeconds: durationSec,
      audioSamples: intent.pcmSamples,
    );

    final assistantTurn = Turn(
      role: TurnRole.assistant,
      text: '',
      modelName: value.loadedModel?.name,
      isGenerating: true,
      statusText: isAsr
          ? AppStrings.statusTranscribingSpeech
          : AppStrings.statusProcessingAudio,
    );

    final newTurns = List<Turn>.from(value.turns)
      ..addAll([userTurn, assistantTurn]);

    value = value.copyWith(turns: newTurns, isGenerating: true);

    if (isAsr) {
      final sw = Stopwatch()..start();
      try {
        debugPrint(
          '[cera:chat] Running Speech to Text (ASR) on ${intent.pcmSamples.length} samples at ${intent.sampleRate} Hz...',
        );
        final transcribed = await cera.transcribe(
          intent.pcmSamples,
          sampleRate: intent.sampleRate,
        );
        sw.stop();
        if (_disposed || generationId != _generationId) return;

        debugPrint('[cera:chat] ASR result: "$transcribed"');
        final trimmed = transcribed.trim();
        final recognizedText = trimmed.isNotEmpty
            ? trimmed
            : AppStrings.noSpeechDetected;
        _updateLastTurn(
          (t) => t.copyWith(
            text: recognizedText,
            isGenerating: false,
            statusText: () => null,
            stats: TurnStats(
              tokens: 0,
              ttftMs: null,
              totalMs: sw.elapsedMilliseconds,
              tps: 0.0,
              audioDurationSeconds: durationSec,
              audioRtf: sw.elapsedMilliseconds > 0 && durationSec > 0
                  ? durationSec / (sw.elapsedMilliseconds / 1000.0)
                  : null,
            ),
          ),
        );
        value = value.copyWith(isGenerating: false);
        return;
      } catch (err) {
        if (_disposed || generationId != _generationId) return;
        _updateLastTurn(
          (t) => t.copyWith(
            isGenerating: false,
            statusText: () => null,
            text: 'Transcription failed: $err',
          ),
        );
        value = value.copyWith(isGenerating: false);
        return;
      }
    }

    final systemPrompt = systemPromptFor(
      settings: value.settings,
      uiMode: value.uiMode,
      isAudioPrompt: true,
      profile: value.audioProfile ?? cera.audioProfile,
    );

    try {
      debugPrint(
        '[cera:chat] Encoding audio prompt (${intent.pcmSamples.length} samples at ${intent.sampleRate} Hz, text: "$promptText")...',
      );
      await cera.appendAudio(
        intent.pcmSamples,
        sampleRate: intent.sampleRate,
        prompt: promptText,
        systemPrompt: _sessionPhase == SessionPhase.idle ? systemPrompt : null,
      );
      debugPrint(
        '[cera:chat] Audio successfully encoded and seeded into KV cache',
      );
      if (_disposed || generationId != _generationId) return;
      if (!value.isGenerating) {
        _updateLastTurn(
          (t) => t.copyWith(isGenerating: false, statusText: () => null),
        );
        return;
      }
      _updateLastTurn(
        (t) =>
            t.copyWith(statusText: () => AppStrings.statusGeneratingResponse),
      );
    } catch (err) {
      if (_disposed || generationId != _generationId) return;
      _updateLastTurn(
        (t) => t.copyWith(
          isGenerating: false,
          statusText: () => null,
          text: 'Failed to process audio: $err',
        ),
      );
      value = value.copyWith(isGenerating: false);
      return;
    }

    if (_disposed || generationId != _generationId || !value.isGenerating) {
      return;
    }
    await _runGeneration('');
  }

  void _updateLastTurn(Turn Function(Turn current) updater) {
    if (_disposed || value.turns.isEmpty) return;
    final turns = List<Turn>.from(value.turns);
    final last = turns.removeLast();
    turns.add(updater(last));
    value = value.copyWith(turns: turns);
  }

  int _generationId = 0;

  Future<void> _runGeneration(String formattedPrompt) async {
    final cera = _ceraEngine;
    if (cera == null || _disposed) {
      if (!_disposed) {
        _updateLastTurn(
          (t) => t.copyWith(isGenerating: false, statusText: () => null),
        );
        value = value.copyWith(isGenerating: false);
      }
      return;
    }

    final generationId = ++_generationId;
    _sessionPhase = SessionPhase.promptReady;
    final stopwatch = Stopwatch()..start();
    int? firstTokenMs;
    int tokenCount = 0;
    final done = Completer<void>();
    _generationCompleter = done;
    var hasError = false;

    final modelLabel = _activeModelLabel();

    debugPrint(
      '[cera:chat] Prefilling prompt (${formattedPrompt.length} chars) for "$modelLabel" and starting generation...',
    );

    final generatedAudioSamples = <double>[];
    int? audioSampleRate;
    int? firstAudioMs;
    int? lastAudioMs;
    int audioChunkCount = 0;
    final mode = value.settings.audioChatMode;
    final isTts = mode == AudioChatMode.textToSpeech;
    final isAudioChat = mode == AudioChatMode.interleaved || isTts;
    final shouldStreamAudio =
        (value.capabilities?.audioOut ?? false) && isAudioChat;

    if (shouldStreamAudio) {
      _audioPlayer.startStream(sampleRate: vocoderSampleRate);
    }

    try {
      final settings = value.settings;
      final stream = cera.generate(
        formattedPrompt,
        maxTokens: settings.maxTokens,
        temperature: isTts
            ? ChatDefaults.greedyTemperature
            : settings.temperature,
        topP: isTts ? null : settings.topP,
        topK: isTts ? null : settings.topK,
        // Must match the system prompt for this mode. Interleaving on an ASR or
        // plain-text turn corrupts the answer.
        audioMode: switch (mode) {
          AudioChatMode.interleaved => CeraAudioMode.interleaved,
          AudioChatMode.textToSpeech => CeraAudioMode.sequential,
          _ => CeraAudioMode.textOnly,
        },
        onAudio: shouldStreamAudio
            ? (pcm, rate) {
                if (rate <= 0 || _disposed || _generationId != generationId) {
                  return;
                }
                final nowMs = stopwatch.elapsedMilliseconds;
                firstAudioMs ??= nowMs;
                lastAudioMs = nowMs;
                audioChunkCount++;
                debugPrint(
                  '[cera:chat] Received ${pcm.length} audio samples at $rate Hz',
                );
                audioSampleRate = rate;
                generatedAudioSamples.addAll(pcm);
                _audioPlayer.appendChunk(pcm);
                _updateLastTurn(
                  (t) => t.copyWith(
                    audioSamples: List.of(generatedAudioSamples),
                    audioDurationSeconds: generatedAudioSamples.length / rate,
                  ),
                );
              }
            : null,
      );

      final sub = stream.listen(
        (piece) {
          if (_disposed || _generationId != generationId) return;
          tokenCount++;
          firstTokenMs ??= stopwatch.elapsedMilliseconds;
          if (!isTts) {
            _updateLastTurn(
              (t) => t.copyWith(
                isGenerating: true,
                statusText: () => null,
                text: t.text + piece,
              ),
            );
          }
        },
        onError: (err) {
          hasError = true;
          _sessionPhase = SessionPhase.interrupted;
          if (!_disposed && _generationId == generationId) {
            _updateLastTurn(
              (t) => t.copyWith(
                isGenerating: false,
                statusText: () => null,
                text: t.text.isEmpty
                    ? 'Error: $err'
                    : '${t.text}\n\n[Error: $err]',
              ),
            );
          }
          if (!done.isCompleted) done.complete();
        },
        onDone: () {
          if (!done.isCompleted) done.complete();
        },
        cancelOnError: true,
      );

      _generationSub = sub;

      await done.future;
    } catch (err) {
      hasError = true;
      _sessionPhase = SessionPhase.interrupted;
      if (!_disposed && _generationId == generationId) {
        _updateLastTurn(
          (t) => t.copyWith(
            isGenerating: false,
            statusText: () => null,
            text: t.text.isEmpty ? 'Error: $err' : '${t.text}\n\n[Error: $err]',
          ),
        );
      }
    } finally {
      stopwatch.stop();
      _audioPlayer.finishStream();
      _generationSub = null;
      _generationCompleter = null;
    }

    if (!hasError && _generationId == generationId) {
      _sessionPhase = SessionPhase.turnComplete;
    }

    TurnStats? stats;
    if (!hasError &&
        (tokenCount > 0 || generatedAudioSamples.isNotEmpty) &&
        !_disposed &&
        _generationId == generationId &&
        value.isGenerating) {
      final lastText = value.turns.lastOrNull?.text ?? '';
      int totalTokens = tokenCount;
      if (lastText.isNotEmpty) {
        try {
          final encoded = await cera.encode(lastText, addSpecial: false);
          totalTokens = encoded.length;
        } catch (_) {}
      }

      final totalMs = stopwatch.elapsedMilliseconds;
      final ttft = firstTokenMs ?? firstAudioMs;
      final decodeMs = totalMs - (ttft ?? 0);
      final tps = totalTokens > 1 && decodeMs > 0
          ? ((totalTokens - 1) / (decodeMs / 1000.0))
          : (totalTokens == 1 && totalMs > 0
                ? (totalTokens / (totalMs / 1000.0))
                : 0.0);

      final rate = (audioSampleRate ?? vocoderSampleRate).toDouble();
      final audioDurationSec = generatedAudioSamples.isNotEmpty
          ? (generatedAudioSamples.length / rate)
          : null;

      double? audioRtf;
      if (audioDurationSec != null) {
        final double audioGenMs;
        if (firstAudioMs != null &&
            lastAudioMs != null &&
            audioChunkCount > 1) {
          final intervalMs = lastAudioMs! - firstAudioMs!;
          final perChunkMs = intervalMs / (audioChunkCount - 1);
          audioGenMs = perChunkMs * audioChunkCount;
        } else {
          audioGenMs = decodeMs.toDouble();
        }
        if (audioGenMs > 0) {
          audioRtf = audioDurationSec / (audioGenMs / 1000.0);
        }
      }

      stats = TurnStats(
        tokens: totalTokens,
        totalMs: totalMs,
        ttftMs: ttft,
        tps: tps,
        audioDurationSeconds: audioDurationSec,
        audioRtf: audioRtf,
      );

      final latencyLabel = totalTokens > 0 ? 'TTFT' : 'TTFA';
      debugPrint(
        '[cera:chat] Generation completed for "$modelLabel": $totalTokens tokens in ${totalMs}ms '
        '(${tps.toStringAsFixed(1)} tok/s, $latencyLabel: ${ttft ?? totalMs}ms'
        '${audioDurationSec != null ? ", ${audioDurationSec.toStringAsFixed(1)}s audio, RTF: ${audioRtf?.toStringAsFixed(2)}x" : ""})',
      );
    }

    if (!_disposed && _generationId == generationId) {
      _updateLastTurn(
        (t) => t.copyWith(
          isGenerating: false,
          statusText: () => null,
          stats: stats ?? t.stats,
        ),
      );

      value = value.copyWith(isGenerating: false);
    }
  }

  Future<void> _onStopGeneration() async {
    _generationId++;
    _audioPlayer.stop();
    if (value.isGenerating) {
      _sessionPhase = SessionPhase.interrupted;
    }
    try {
      await _ceraEngine?.cancel();
    } catch (_) {}
    await _generationSub?.cancel();
    _generationSub = null;
    final completer = _generationCompleter;
    _generationCompleter = null;
    if (completer != null && !completer.isCompleted) {
      completer.complete();
    }
    if (_disposed) return;
    final updatedTurns = value.turns.map((turn) {
      if (turn.isGenerating) {
        return turn.copyWith(isGenerating: false, statusText: () => null);
      }
      return turn;
    }).toList();
    value = value.copyWith(isGenerating: false, turns: updatedTurns);
  }

  void _onAttachImage(Uint8List bytes, String name) {
    value = value.copyWith(
      pendingImageBytes: () => bytes,
      pendingImageName: () => name,
    );
  }

  void _onClearAttachedImage() {
    value = value.copyWith(
      pendingImageBytes: () => null,
      pendingImageName: () => null,
    );
  }

  Future<void> _onClearTranscript() async {
    await _onStopGeneration();
    try {
      await _ceraEngine?.reset();
      _sessionPhase = SessionPhase.idle;
      if (!_disposed) {
        value = value.copyWith(turns: []);
      }
    } catch (err) {
      _sessionPhase = SessionPhase.unusable;
      debugPrint('[cera:chat] Engine reset failed: $err');
      if (!_disposed) {
        value = value.copyWith(status: 'Engine reset failed: $err');
      }
    }
  }

  @visibleForTesting
  static String voiceTagFor(CeraCapabilities caps) {
    if (caps.audioIn && caps.audioOut) {
      return AppStrings.tagVoice;
    } else if (caps.audioIn) {
      return AppStrings.tagAsr;
    } else if (caps.audioOut) {
      return AppStrings.tagAudio;
    }
    return '';
  }

  @visibleForTesting
  static ChatSettings alignAudioMode(
    ChatSettings settings,
    CeraCapabilities caps,
  ) {
    if (caps.audioIn && !caps.audioOut) {
      if (settings.audioChatMode != AudioChatMode.speechToText) {
        return settings.copyWith(audioChatMode: AudioChatMode.speechToText);
      }
    } else if (!caps.audioIn && caps.audioOut) {
      if (settings.audioChatMode == AudioChatMode.speechToText ||
          settings.audioChatMode == AudioChatMode.interleaved) {
        return settings.copyWith(audioChatMode: AudioChatMode.textToSpeech);
      }
    } else if (caps.audioIn && caps.audioOut) {
      // Respect explicit textOnly fallback; otherwise upgrade to interleaved.
      if (settings.audioChatMode != AudioChatMode.textOnly &&
          settings.audioChatMode != AudioChatMode.interleaved) {
        return settings.copyWith(audioChatMode: AudioChatMode.interleaved);
      }
    } else {
      if (settings.audioChatMode != AudioChatMode.textOnly) {
        return settings.copyWith(audioChatMode: AudioChatMode.textOnly);
      }
    }
    return settings;
  }

  /// The system prompt for the current audio mode, or `null` when it needs none.
  ///
  /// The prompt and the voice persona come from the loaded model's [profile]:
  /// a persona the model does not list is dropped, because an unknown one makes
  /// it answer in text and never speak.
  @visibleForTesting
  static String? systemPromptFor({
    required ChatSettings settings,
    required AppUIMode uiMode,
    required CeraAudioProfile profile,
    bool isAudioPrompt = false,
  }) {
    final rawPersona = uiMode == AppUIMode.ttsStudio
        ? settings.ttsStudioVoice
        : settings.chatVoice;
    final voicePersona = rawPersona.trim();
    switch (settings.audioChatMode) {
      case AudioChatMode.interleaved:
        return profile.interleavedSystemPromptFor(voicePersona);
      case AudioChatMode.textToSpeech when !isAudioPrompt:
        return profile.ttsSystemPromptFor(voicePersona);
      // A voice note with no system prompt falls back to the model's
      // interleaved one and would be answered with speech.
      case AudioChatMode.textOnly when isAudioPrompt:
        return ceraTextOnlySystemPrompt;
      default:
        return null;
    }
  }

  static const List<CeraTtsVoice> _kDefaultLfmAudioVoices = [
    CeraTtsVoice(
      label: AudioProfileDefaults.usFemaleLabel,
      prompt: AudioProfileDefaults.usFemalePrompt,
      ttsSystemPrompt: AudioProfileDefaults.usFemaleTtsSystemPrompt,
      interleavedSystemPrompt:
          AudioProfileDefaults.usFemaleInterleavedSystemPrompt,
    ),
    CeraTtsVoice(
      label: AudioProfileDefaults.usMaleLabel,
      prompt: AudioProfileDefaults.usMalePrompt,
      ttsSystemPrompt: AudioProfileDefaults.usMaleTtsSystemPrompt,
      interleavedSystemPrompt:
          AudioProfileDefaults.usMaleInterleavedSystemPrompt,
    ),
    CeraTtsVoice(
      label: AudioProfileDefaults.ukFemaleLabel,
      prompt: AudioProfileDefaults.ukFemalePrompt,
      ttsSystemPrompt: AudioProfileDefaults.ukFemaleTtsSystemPrompt,
      interleavedSystemPrompt:
          AudioProfileDefaults.ukFemaleInterleavedSystemPrompt,
    ),
    CeraTtsVoice(
      label: AudioProfileDefaults.ukMaleLabel,
      prompt: AudioProfileDefaults.ukMalePrompt,
      ttsSystemPrompt: AudioProfileDefaults.ukMaleTtsSystemPrompt,
      interleavedSystemPrompt:
          AudioProfileDefaults.ukMaleInterleavedSystemPrompt,
    ),
  ];

  /// Ensures English `LFM2*-Audio` models always expose their 4 built-in voice
  /// personas even when loaded from raw `.gguf` bytes (`<bytes>`) or via an
  /// older web worker `.wasm` binary that returns an empty voice list.
  @visibleForTesting
  static CeraAudioProfile resolveAudioProfile(
    CeraAudioProfile profile, {
    required String modelIdentity,
    required CeraCapabilities capabilities,
  }) {
    if (profile.hasVoices || !capabilities.audioOut) {
      return profile;
    }
    final tokens = modelIdentity
        .toLowerCase()
        .split(RegExp(r'[^a-z0-9]+'))
        .where((t) => t.isNotEmpty)
        .toList();
    final hasLfm = tokens.any(
      (t) => t.startsWith(AudioProfileDefaults.lfmToken),
    );
    final hasAudio = tokens.contains(AudioProfileDefaults.audioToken);
    final isJapanese =
        tokens.contains(AudioProfileDefaults.jpToken) ||
        tokens.contains(AudioProfileDefaults.japaneseToken);
    if (hasLfm && hasAudio && !isJapanese) {
      return CeraAudioProfile(
        ttsSystemPrompt: AudioProfileDefaults.usFemaleTtsSystemPrompt,
        interleavedSystemPrompt:
            AudioProfileDefaults.usFemaleInterleavedSystemPrompt,
        voices: _kDefaultLfmAudioVoices,
        sampleTexts: profile.sampleTexts,
      );
    }
    return profile;
  }

  @override
  void dispose() {
    _disposed = true;
    _audioPlayer.dispose();
    _generationSub?.cancel();
    final completer = _generationCompleter;
    if (completer != null && !completer.isCompleted) {
      completer.complete();
    }
    _ceraEngine?.terminate();
    super.dispose();
  }
}

/// Normalizes and cleans raw text for speech synthesis by removing timestamps,
/// stage directions in parentheses, corrupted Unicode characters, and formatting markers.
String normalizeSpeechText(String input) {
  var text = input;
  // Strip Unicode replacement characters and corrupted bytes
  text = text.replaceAll('\uFFFD', ' ');
  // Strip bracketed and parenthesized timestamps (e.g., [0:00 - 0:04] or (0:00 - 0:38))
  text = text.replaceAll(
    RegExp(r'\[\s*\d+:\d+\s*(?:[-~]\s*\d+:\d+)?\s*\]'),
    ' ',
  );
  text = text.replaceAll(
    RegExp(r'\(\s*\d+:\d+\s*(?:[-~]\s*\d+:\d+)?\s*\)'),
    ' ',
  );
  // Strip script stage directions in parentheses (e.g., (App loading...), (Card: ...))
  text = text.replaceAll(
    RegExp(
      r'\([^)]*(?:Card|App|Screen|Demo|Intro|Summary|Deep dive|Full draft|Confirmation|Gmail|LinkedIn|transition|scene|visual|photo)[^)]*\)',
      caseSensitive: false,
    ),
    ' ',
  );
  // Strip voiceover script headers (e.g., Voiceover Script "...")
  text = text.replaceAll(
    RegExp(r'Voiceover Script[^\n]*\n?', caseSensitive: false),
    ' ',
  );
  // Normalize quotes and dashes into natural speech pauses
  text = text.replaceAll(RegExp(r'["“”]'), '');
  text = text.replaceAll(RegExp(r'[\u2014\u2013]'), ', ');
  text = text.replaceAll(RegExp(r'\s+'), ' ').trim();
  return text;
}

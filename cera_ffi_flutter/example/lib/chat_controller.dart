import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

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
    _loadDownloadedRecords();
  }

  final Future<String?> Function() _defaultStoreDir;
  final AudioPlayerService _audioPlayer = AudioPlayerService();
  Cera? _ceraEngine;
  StreamSubscription<String>? _generationSub;
  Completer<void>? _generationCompleter;
  bool _disposed = false;

  /// Audio player service instance.
  AudioPlayerService get audioPlayer => _audioPlayer;

  /// Loads locally tracked downloaded model records from persistent storage.
  Future<void> _loadDownloadedRecords() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList('cera_downloaded_models') ?? [];
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
        value = value.copyWith(uiMode: intent.mode);
    }
  }

  /// Explicitly and cleanly unloads the currently active Cera model and frees
  /// all underlying memory and native resources.
  Future<void> _unloadCurrentModel() async {
    if (_generationSub != null) {
      await _generationSub?.cancel();
      _generationSub = null;
    }
    if (_generationCompleter != null && !_generationCompleter!.isCompleted) {
      _generationCompleter!.complete();
      _generationCompleter = null;
    }
    if (_ceraEngine != null) {
      final old = _ceraEngine;
      _ceraEngine = null;
      try {
        await old?.close();
      } catch (err) {
        debugPrint('cera: error closing previous engine: $err');
      }
    }
  }

  Future<void> _onUnloadModel() async {
    await _unloadCurrentModel();
    if (_disposed) return;
    value = value.copyWith(
      loadedModel: () => null,
      capabilities: () => null,
      backend: () => null,
      status: 'Model unloaded.',
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

      final backendStr = prefs.getString('cera_backend');
      final backend = switch (backendStr) {
        'cpu' => CeraBackend.cpu,
        'gpu' => CeraBackend.gpu,
        _ => CeraBackend.auto,
      };
      final turboQuant = prefs.getBool('cera_turboquant') ?? false;
      final maxImageDim = prefs.getInt('cera_max_image_dim');
      final audioModeStr = prefs.getString('cera_audio_chat_mode');
      final audioMode = switch (audioModeStr) {
        'asr' || 'speechToText' => AudioChatMode.speechToText,
        'tts' || 'textToSpeech' => AudioChatMode.textToSpeech,
        'textOnly' => AudioChatMode.textOnly,
        _ => AudioChatMode.interleaved,
      };
      final chatVoice =
          prefs.getString('cera_chat_voice') ??
          prefs.getString('cera_tts_voice') ??
          'Use the US female voice.';
      final ttsStudioVoice =
          prefs.getString('cera_tts_studio_voice') ??
          'Use the US female voice.';
      final restoredSettings = ChatSettings(
        backend: backend,
        turboQuant: turboQuant,
        maxImageLongSize: maxImageDim == 0 ? null : (maxImageDim ?? 256),
        audioChatMode: audioMode,
        chatVoice: chatVoice,
        ttsStudioVoice: ttsStudioVoice,
      );
      value = value.copyWith(settings: restoredSettings);

      final bundleName = prefs.getString('cera_last_bundle_name');
      final quant = prefs.getString('cera_last_bundle_quant');
      if (bundleName != null && quant != null) {
        final displayName = bundleName.endsWith('-GGUF')
            ? bundleName.substring(0, bundleName.length - '-GGUF'.length)
            : bundleName;

        await _load(
          modelSource: BundleModelSource(
            name: '$displayName · $quant',
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
          label: '$displayName · $quant',
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
        await prefs.remove('cera_last_bundle_name');
        await prefs.remove('cera_last_bundle_quant');
      } catch (_) {}
    }
  }

  Future<void> _onUpdateSettings(UpdateSettingsIntent intent) async {
    final current = value.settings;
    final newBackend = intent.backend ?? current.backend;
    final newTurbo = intent.turboQuant ?? current.turboQuant;
    final newMaxImage = intent.clearMaxImageLongSize
        ? null
        : (intent.maxImageLongSize ?? current.maxImageLongSize);
    final newAudioMode = intent.audioChatMode ?? current.audioChatMode;
    final newChatVoice =
        intent.chatVoice ?? intent.ttsVoice ?? current.chatVoice;
    final newTtsStudioVoice = intent.ttsStudioVoice ?? current.ttsStudioVoice;

    final updated = current.copyWith(
      backend: newBackend,
      turboQuant: newTurbo,
      maxImageLongSize: () => newMaxImage,
      audioChatMode: newAudioMode,
      chatVoice: newChatVoice,
      ttsStudioVoice: newTtsStudioVoice,
    );

    value = value.copyWith(settings: updated);

    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('cera_backend', newBackend.name);
      await prefs.setBool('cera_turboquant', newTurbo);
      if (newMaxImage != null) {
        await prefs.setInt('cera_max_image_dim', newMaxImage);
      } else {
        await prefs.setInt('cera_max_image_dim', 0);
      }
      await prefs.setString('cera_audio_chat_mode', switch (newAudioMode) {
        AudioChatMode.speechToText => 'asr',
        AudioChatMode.textToSpeech => 'tts',
        AudioChatMode.textOnly => 'textOnly',
        AudioChatMode.interleaved => 'interleaved',
      });
      await prefs.setString('cera_chat_voice', newChatVoice);
      await prefs.setString('cera_tts_studio_voice', newTtsStudioVoice);
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
        dispatch(
          LoadBundleIntent(
            bundleName: model.bundleName,
            quant: model.quant,
            displayName: model.displayName ?? model.bundleName,
            storeDir: _defaultStoreDir,
          ),
        );
      } else if (model is ModelSource) {
        dispatch(LoadLocalModelIntent(model));
      }
    }
  }

  Future<void> _onLoadBundle(LoadBundleIntent intent) async {
    final label = '${intent.displayName} · ${intent.quant}';
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
        knownFileSizes = await probeBundleFileSizes(bundleName, quant);
      } catch (_) {}
    }

    final uniqueSizes = <String, int>{};
    for (final entry in knownFileSizes.entries) {
      final key = entry.key.contains('/')
          ? entry.key.split('/').last
          : entry.key;
      if (!key.endsWith('.json')) {
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
        if (_disposed) return;
        final fileName = progress.url.contains('/')
            ? progress.url.split('/').last
            : progress.url;

        // Skip tiny manifest JSON from triggering 100% false spikes
        if (fileName.endsWith('.json') && expectedTotalBytes > 0) {
          return;
        }

        fileDownloaded[fileName] = progress.bytesDownloaded;
        if (progress.totalBytes != null && progress.totalBytes! > 0) {
          fileTotals[fileName] = progress.totalBytes!;
        } else if (knownFileSizes.containsKey(fileName)) {
          fileTotals[fileName] = knownFileSizes[fileName]!;
        } else if (knownFileSizes.containsKey(progress.url)) {
          fileTotals[fileName] = knownFileSizes[progress.url]!;
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

        final totalStr = totalBytes > 0
            ? '${(totalBytes / 1024 / 1024).toStringAsFixed(1)} MB'
            : null;

        final bytesInfo = totalStr != null
            ? '$downloadedMb / $totalStr'
            : '$downloadedMb MB';

        final fileSuffix = fileName.isNotEmpty ? ' · $fileName' : '';

        value = value.copyWith(
          downloadFraction: () => fraction,
          status: 'Downloading $label · $pctStr ($bytesInfo)$fileSuffix',
        );
      });

      if (_disposed) {
        await cera.close();
        return;
      }

      _ceraEngine = cera;
      final visionTag = cera.capabilities.imageIn ? ' · Vision' : '';

      value = value.copyWith(
        loadedModel: () => modelSource,
        capabilities: () => cera.capabilities,
        backend: () => cera.backend,
        status: '${modelSource.name} · ${cera.backend}$visionTag',
        isLoading: false,
        downloadFraction: () => null,
      );

      // Persist preferences
      try {
        final prefs = await SharedPreferences.getInstance();
        if (isBundle && bundleName != null && quant != null) {
          await prefs.setString('cera_last_bundle_name', bundleName);
          await prefs.setString('cera_last_bundle_quant', quant);
          await _saveDownloadedRecord(
            bundleName,
            quant,
            displayName ?? bundleName,
          );
        } else {
          await prefs.remove('cera_last_bundle_name');
          await prefs.remove('cera_last_bundle_quant');
        }
      } catch (_) {}
    } catch (err, stack) {
      debugPrint('cera: model failed to load: $err\n$stack');
      if (!_disposed) {
        value = value.copyWith(
          isLoading: false,
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
      final list = prefs.getStringList('cera_downloaded_models') ?? [];
      final exists = list.any((item) {
        try {
          final map = jsonDecode(item) as Map<String, dynamic>;
          return map['bundleName'] == bundleName && map['quant'] == quant;
        } catch (_) {
          return false;
        }
      });
      if (!exists) {
        final record = jsonEncode({
          'bundleName': bundleName,
          'quant': quant,
          'displayName': displayName,
        });
        list.add(record);
        await prefs.setStringList('cera_downloaded_models', list);
        await _loadDownloadedRecords();
      }
    } catch (_) {}
  }

  Future<void> _onSendMessage(String prompt) async {
    final cera = _ceraEngine;
    final imageBytes = value.pendingImageBytes;
    final imageName = value.pendingImageName;

    if ((prompt.trim().isEmpty && imageBytes == null) ||
        cera == null ||
        value.isBusy ||
        _disposed) {
      return;
    }

    final userTurn = Turn(
      role: 'user',
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
      initialStatus = 'Analyzing image...';
    } else if (isTts) {
      initialStatus = 'Synthesizing neural speech...';
    } else if (isInterleaved) {
      initialStatus = 'Thinking & speaking...';
    } else {
      initialStatus = 'Thinking...';
    }

    final assistantTurn = Turn(
      role: 'assistant',
      text: '',
      modelName: value.loadedModel?.name,
      isGenerating: true,
      statusText: initialStatus,
    );

    final newTurns = List<Turn>.from(value.turns)
      ..addAll([userTurn, assistantTurn]);

    value = value.copyWith(
      turns: newTurns,
      isGenerating: true,
      pendingImageBytes: () => null,
      pendingImageName: () => null,
    );

    final promptText = prompt.trim();
    final framedPrompt = promptText.isNotEmpty
        ? promptText
        : (imageBytes != null ? 'Describe this image.' : '');

    debugPrint(
      '[cera:chat] Submitting user message: "$framedPrompt" '
      '(image: ${imageBytes != null ? "${imageBytes.length} bytes" : "none"}, audioMode: ${audioMode.name})',
    );

    final voicePersona = value.uiMode == AppUIMode.ttsStudio
        ? value.settings.ttsStudioVoice
        : value.settings.chatVoice;

    final String? systemPrompt = isTts
        ? 'Perform TTS. $voicePersona'.trim()
        : (isInterleaved
              ? 'Respond with interleaved text and audio. $voicePersona'.trim()
              : null);

    final messages = <CeraMessage>[
      if (systemPrompt != null) CeraMessage.system(systemPrompt),
      CeraMessage.user(framedPrompt),
    ];

    String formattedPrompt;
    try {
      formattedPrompt = await cera.applyChatTemplate(messages);
    } catch (_) {
      formattedPrompt = framedPrompt;
    }

    if (imageBytes != null) {
      try {
        _updateLastTurn(
          (t) => t.copyWith(statusText: () => 'Encoding image patches...'),
        );
        final maxLong = value.settings.maxImageLongSize;
        debugPrint(
          '[cera:chat] Encoding image patches with maxLongSize ${maxLong ?? "native"}...',
        );
        await cera.appendImage(imageBytes, maxLongSize: maxLong);
        debugPrint(
          '[cera:chat] Image successfully encoded and seeded into KV cache',
        );
        _updateLastTurn(
          (t) => t.copyWith(statusText: () => 'Generating response...'),
        );
      } catch (err) {
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

    if (_disposed || !value.isGenerating) return;
    await _runGeneration(formattedPrompt);
  }

  Future<void> _onSendAudioPrompt(SendAudioPromptIntent intent) async {
    final cera = _ceraEngine;
    if (intent.pcmSamples.isEmpty ||
        cera == null ||
        value.isBusy ||
        _disposed) {
      return;
    }

    final durationSec = intent.pcmSamples.length / intent.sampleRate;
    final promptText = intent.prompt.trim();
    final isAsr = value.settings.audioChatMode == AudioChatMode.speechToText;

    final userTurn = Turn(
      role: 'user',
      text: promptText,
      audioDurationSeconds: durationSec,
      audioSamples: intent.pcmSamples,
    );

    final assistantTurn = Turn(
      role: 'assistant',
      text: '',
      modelName: value.loadedModel?.name,
      isGenerating: true,
      statusText: isAsr ? 'Transcribing speech...' : 'Processing audio...',
    );

    final newTurns = List<Turn>.from(value.turns)
      ..addAll([userTurn, assistantTurn]);

    value = value.copyWith(turns: newTurns, isGenerating: true);

    if (isAsr) {
      try {
        debugPrint(
          '[cera:chat] Running Speech to Text (ASR) on ${intent.pcmSamples.length} samples at ${intent.sampleRate} Hz...',
        );
        final transcribed = await cera.transcribe(
          intent.pcmSamples,
          sampleRate: intent.sampleRate,
        );
        debugPrint('[cera:chat] ASR result: "$transcribed"');
        final recognizedText = transcribed.trim().isNotEmpty
            ? transcribed.trim()
            : '(No speech detected)';
        _updateLastTurn(
          (t) => t.copyWith(
            text: recognizedText,
            isGenerating: false,
            statusText: () => null,
          ),
        );
        value = value.copyWith(isGenerating: false);
        return;
      } catch (err) {
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

    try {
      debugPrint(
        '[cera:chat] Encoding audio prompt (${intent.pcmSamples.length} samples at ${intent.sampleRate} Hz, text: "$promptText")...',
      );
      await cera.appendAudio(
        intent.pcmSamples,
        sampleRate: intent.sampleRate,
        prompt: promptText,
      );
      debugPrint(
        '[cera:chat] Audio successfully encoded and seeded into KV cache',
      );
      _updateLastTurn(
        (t) => t.copyWith(statusText: () => 'Generating response...'),
      );
    } catch (err) {
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

    if (_disposed || !value.isGenerating) return;
    await _runGeneration('');
  }

  void _updateLastTurn(Turn Function(Turn current) updater) {
    if (value.turns.isEmpty) return;
    final turns = List<Turn>.from(value.turns);
    final last = turns.removeLast();
    turns.add(updater(last));
    value = value.copyWith(turns: turns);
  }

  Future<void> _runGeneration(String formattedPrompt) async {
    final cera = _ceraEngine;
    if (cera == null || _disposed) return;

    final stopwatch = Stopwatch()..start();
    int? firstTokenMs;
    int tokenCount = 0;
    final done = Completer<void>();
    _generationCompleter = done;

    debugPrint(
      '[cera:chat] Prefilling prompt (${formattedPrompt.length} chars) and starting generation...',
    );

    final generatedAudioSamples = <double>[];
    final isTts = value.settings.audioChatMode == AudioChatMode.textToSpeech;
    final shouldStreamAudio = (value.capabilities?.audioOut ?? false) &&
        (value.settings.audioChatMode == AudioChatMode.interleaved ||
            value.settings.audioChatMode == AudioChatMode.textToSpeech);

    if (shouldStreamAudio) {
      _audioPlayer.startStream(sampleRate: 24000);
    }

    try {
      final stream = cera.generate(
        formattedPrompt,
        maxTokens: 512,
        temperature: isTts ? 0.0 : null,
        onAudio: shouldStreamAudio
            ? (pcm, rate) {
                debugPrint(
                  '[cera:chat] Received ${pcm.length} audio samples at $rate Hz',
                );
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
          tokenCount++;
          if (firstTokenMs == null) {
            firstTokenMs = stopwatch.elapsedMilliseconds;
            debugPrint('[cera:chat] First token received in ${firstTokenMs}ms');
          }
          _updateLastTurn(
            (t) => t.copyWith(
              isGenerating: true,
              statusText: () => null,
              text: t.text + piece,
            ),
          );
        },
        onError: (Object err) {
          _updateLastTurn(
            (t) => t.copyWith(
              isGenerating: false,
              statusText: () => null,
              text: t.text.isEmpty
                  ? 'Error: $err'
                  : '${t.text}\n\n[Error: $err]',
            ),
          );
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
      _updateLastTurn(
        (t) => t.copyWith(
          isGenerating: false,
          statusText: () => null,
          text: t.text.isEmpty ? 'Error: $err' : '${t.text}\n\n[Error: $err]',
        ),
      );
    } finally {
      stopwatch.stop();
      _audioPlayer.finishStream();
      _generationSub = null;
      _generationCompleter = null;
    }

    final lastText = value.turns.isNotEmpty ? value.turns.last.text : '';
    int totalTokens = 0;
    if (lastText.isNotEmpty) {
      try {
        final encoded = await cera.encode(lastText, addSpecial: false);
        totalTokens = encoded.length;
      } catch (_) {
        totalTokens = tokenCount;
      }
    } else {
      totalTokens = tokenCount;
    }

    final totalMs = stopwatch.elapsedMilliseconds;
    final ttft = firstTokenMs;
    final decodeMs = ttft != null ? (totalMs - ttft) : totalMs;
    final tps = totalTokens > 1 && decodeMs > 0
        ? ((totalTokens - 1) / (decodeMs / 1000.0))
        : (totalTokens == 1 && totalMs > 0
              ? (totalTokens / (totalMs / 1000.0))
              : 0.0);

    final audioDurationSec = generatedAudioSamples.isNotEmpty
        ? (generatedAudioSamples.length / 24000.0)
        : null;
    final audioRtf = (audioDurationSec != null && totalMs > 0)
        ? (audioDurationSec / (totalMs / 1000.0))
        : null;

    final stats = (totalTokens > 0 || audioDurationSec != null)
        ? TurnStats(
            tokens: totalTokens,
            totalMs: totalMs,
            ttftMs: ttft,
            tps: tps,
            audioDurationSeconds: audioDurationSec,
            audioRtf: audioRtf,
          )
        : null;

    _updateLastTurn(
      (t) =>
          t.copyWith(isGenerating: false, statusText: () => null, stats: stats),
    );

    debugPrint(
      '[cera:chat] Generation completed: $totalTokens tokens in ${totalMs}ms '
      '(${tps.toStringAsFixed(1)} tok/s, TTFT: ${ttft ?? totalMs}ms'
      '${audioDurationSec != null ? ", ${audioDurationSec.toStringAsFixed(1)}s audio, RTF: ${audioRtf?.toStringAsFixed(2)}x" : ""})',
    );

    if (!_disposed) {
      value = value.copyWith(isGenerating: false);
    }
  }

  Future<void> _onStopGeneration() async {
    _audioPlayer.stop();
    if (_ceraEngine != null) {
      try {
        await _ceraEngine?.cancel();
      } catch (_) {}
    }
    if (_generationSub != null) {
      await _generationSub?.cancel();
      _generationSub = null;
    }
    if (_generationCompleter != null && !_generationCompleter!.isCompleted) {
      _generationCompleter!.complete();
      _generationCompleter = null;
    }
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
    } on UnsupportedError {
      // The WebGPU backend owns its KV cache on the GPU with no in-place reset;
      // reopen the engine with the current model to clear it cleanly.
      final current = value.loadedModel;
      if (current != null) {
        await _ceraEngine?.close();
        _ceraEngine = null;
        try {
          final reloaded = await current.open(
            options: value.settings.ceraOptions,
          );
          if (_disposed) {
            await reloaded.close();
            return;
          }
          _ceraEngine = reloaded;
        } catch (err) {
          if (_disposed) return;
          value = value.copyWith(
            loadedModel: () => null,
            backend: () => null,
            capabilities: () => null,
            status: 'Failed to reload model: $err',
          );
        }
      }
    } catch (err) {
      debugPrint('[cera:chat] Engine reset failed: $err');
    }
    if (!_disposed) {
      value = value.copyWith(turns: []);
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _audioPlayer.dispose();
    _generationSub?.cancel();
    if (_generationCompleter != null && !_generationCompleter!.isCompleted) {
      _generationCompleter!.complete();
    }
    _ceraEngine?.close();
    super.dispose();
  }
}

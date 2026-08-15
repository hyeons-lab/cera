import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'chat_intent.dart';
import 'chat_state.dart';
import 'model_source.dart';

/// MVI Controller / Store managing state transitions, Cera engine lifecycle,
/// download progress, and token generation.
class ChatController extends ValueNotifier<ChatState> {
  ChatController({Future<String?> Function()? defaultStoreDir})
    : _defaultStoreDir = defaultStoreDir ?? (() async => null),
      super(const ChatState()) {
    _loadDownloadedRecords();
  }

  final Future<String?> Function() _defaultStoreDir;
  Cera? _ceraEngine;
  StreamSubscription<String>? _generationSub;
  Completer<void>? _generationCompleter;
  bool _disposed = false;

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
      case RemoveDownloadedModelIntent():
        await _onRemoveDownloadedModel(intent.record);
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

      final bundleName = prefs.getString('cera_last_bundle_name');
      final quant = prefs.getString('cera_last_bundle_quant');
      if (bundleName != null && quant != null) {
        final displayName = bundleName.endsWith('-GGUF')
            ? bundleName.substring(0, bundleName.length - '-GGUF'.length)
            : bundleName;

        // Ensure active model is recorded in downloaded models
        await _saveDownloadedRecord(bundleName, quant, displayName);

        await _load(
          bundleSource: BundleModelSource(
            name: '$displayName · $quant',
            bundleName: bundleName,
            quant: quant,
            getStoreDir: _defaultStoreDir,
          ),
          openFn: (onProgress) async => Cera.openBundle(
            bundleName,
            quant,
            storeDir: await _defaultStoreDir(),
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

  Future<void> _onLoadBundle(LoadBundleIntent intent) async {
    final label = '${intent.displayName} · ${intent.quant}';
    final bundleSource = BundleModelSource(
      name: label,
      bundleName: intent.bundleName,
      quant: intent.quant,
      getStoreDir: intent.storeDir,
    );

    await _load(
      bundleSource: bundleSource,
      openFn: (onProgress) async => Cera.openBundle(
        intent.bundleName,
        intent.quant,
        storeDir: await intent.storeDir(),
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
      bundleSource: intent.source,
      openFn: (_) => intent.source.open(),
      label: intent.source.name,
      isBundle: false,
    );
  }

  Future<void> _load({
    required LoadedModel bundleSource,
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

    try {
      final cera = await openFn((progress) {
        if (_disposed) return;
        final pct = progress.fraction == null
            ? '${(progress.bytesDownloaded / 1024 / 1024).toStringAsFixed(0)} MB'
            : '${(progress.fraction! * 100).toStringAsFixed(0)}%';
        value = value.copyWith(
          downloadFraction: () => progress.fraction,
          status: 'Downloading $label · $pct',
        );
      });

      if (_disposed) {
        await cera.close();
        return;
      }

      _ceraEngine = cera;
      final visionTag = cera.capabilities.imageIn ? ' · Vision' : '';

      value = value.copyWith(
        loadedModel: () => bundleSource,
        capabilities: () => cera.capabilities,
        backend: () => cera.backend,
        status: '${bundleSource.name} · ${cera.backend}$visionTag',
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

  Future<void> _onRemoveDownloadedModel(DownloadedModelRecord record) async {
    final updated = value.downloadedModels
        .where((r) => r.id != record.id)
        .toList();
    value = value.copyWith(downloadedModels: updated);
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = updated.map((r) => jsonEncode(r.toJson())).toList();
      await prefs.setStringList('cera_downloaded_models', list);
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

    final assistantTurn = Turn(
      role: 'assistant',
      text: '',
      modelName: value.loadedModel?.name,
      isGenerating: true,
      statusText: imageBytes != null ? 'Analyzing image...' : 'Thinking...',
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

    String formattedPrompt;
    try {
      formattedPrompt = await cera.applyChatTemplate([
        CeraMessage.user(framedPrompt),
      ]);
    } catch (_) {
      formattedPrompt = framedPrompt;
    }

    if (imageBytes != null) {
      try {
        await cera.appendImage(imageBytes);
      } catch (err) {
        assistantTurn.isGenerating = false;
        assistantTurn.statusText = null;
        assistantTurn.text = 'Failed to process image: $err';
        notifyListeners();
        value = value.copyWith(isGenerating: false);
        return;
      }
    }

    await _runGeneration(assistantTurn, formattedPrompt);
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
    final userTurn = Turn(
      role: 'user',
      text: promptText.isNotEmpty
          ? promptText
          : '🎙️ Audio (${durationSec.toStringAsFixed(1)}s)',
      audioDurationSeconds: durationSec,
    );

    final assistantTurn = Turn(
      role: 'assistant',
      text: '',
      modelName: value.loadedModel?.name,
      isGenerating: true,
      statusText: 'Listening to audio...',
    );

    final newTurns = List<Turn>.from(value.turns)
      ..addAll([userTurn, assistantTurn]);

    value = value.copyWith(
      turns: newTurns,
      isGenerating: true,
      pendingImageBytes: () => null,
      pendingImageName: () => null,
    );

    String formattedPrompt;
    if (promptText.isNotEmpty) {
      try {
        formattedPrompt = await cera.applyChatTemplate([
          CeraMessage.user(promptText),
        ]);
      } catch (_) {
        formattedPrompt = promptText;
      }
    } else {
      formattedPrompt = '';
    }

    try {
      await cera.appendAudio(intent.pcmSamples, sampleRate: intent.sampleRate);
    } catch (err) {
      assistantTurn.isGenerating = false;
      assistantTurn.statusText = null;
      assistantTurn.text = 'Failed to process audio: $err';
      notifyListeners();
      value = value.copyWith(isGenerating: false);
      return;
    }

    await _runGeneration(assistantTurn, formattedPrompt);
  }

  Future<void> _runGeneration(
    Turn assistantTurn,
    String formattedPrompt,
  ) async {
    final cera = _ceraEngine;
    if (cera == null || _disposed) return;

    final stopwatch = Stopwatch()..start();
    int? firstTokenMs;
    int tokenCount = 0;
    final done = Completer<void>();
    _generationCompleter = done;

    try {
      final stream = cera.generate(formattedPrompt);

      final sub = stream.listen(
        (piece) {
          tokenCount++;
          firstTokenMs ??= stopwatch.elapsedMilliseconds;
          assistantTurn.isGenerating = true;
          assistantTurn.statusText = null;
          assistantTurn.text += piece;
          notifyListeners();
        },
        onError: (Object err) {
          assistantTurn.isGenerating = false;
          assistantTurn.statusText = null;
          assistantTurn.text = 'Error: $err';
          notifyListeners();
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
      assistantTurn.isGenerating = false;
      assistantTurn.statusText = null;
      assistantTurn.text = 'Error: $err';
      notifyListeners();
    } finally {
      stopwatch.stop();
      _generationSub = null;
      _generationCompleter = null;
    }

    int totalTokens = 0;
    if (assistantTurn.text.isNotEmpty) {
      try {
        final encoded = await cera.encode(
          assistantTurn.text,
          addSpecial: false,
        );
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

    assistantTurn.isGenerating = false;
    assistantTurn.statusText = null;
    if (totalTokens > 0) {
      assistantTurn.stats = TurnStats(
        tokens: totalTokens,
        totalMs: totalMs,
        ttftMs: ttft,
        tps: tps,
      );
    }

    if (!_disposed) {
      value = value.copyWith(
        isGenerating: false,
        turns: List<Turn>.from(value.turns),
      );
    }
  }

  Future<void> _onStopGeneration() async {
    if (_generationSub != null) {
      await _generationSub?.cancel();
      _generationSub = null;
    }
    if (_generationCompleter != null && !_generationCompleter!.isCompleted) {
      _generationCompleter!.complete();
      _generationCompleter = null;
    }
    for (final turn in value.turns) {
      if (turn.isGenerating) {
        turn.isGenerating = false;
        turn.statusText = null;
      }
    }
    value = value.copyWith(isGenerating: false);
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
          final reloaded = await current.open();
          _ceraEngine = reloaded;
        } catch (_) {}
      }
    } catch (_) {}
    value = value.copyWith(turns: []);
  }

  @override
  void dispose() {
    _disposed = true;
    _generationSub?.cancel();
    _ceraEngine?.close();
    super.dispose();
  }
}

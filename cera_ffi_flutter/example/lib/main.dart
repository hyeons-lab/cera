// Example Flutter app for `cera_ffi_flutter`: pick a GGUF, chat with it, watch
// tokens stream in.
//
// One code path serves every platform, web included, because it is written
// against `Cera`, the portable async API, rather than against the generated
// bindings. That is the whole point of the type: the bindings are synchronous
// and `dart:ffi`-based, and neither of those can exist in a browser, so an app
// that wants to run everywhere cannot be written against them.
//
// Note what is *absent* compared to a direct-bindings version: no
// `dart:isolate`, no `dart:io`, no per-turn model reload. The blocking work
// already happens off the Dart thread on both transports, so there is nothing
// left for an isolate to fix.
//
// Running this on the web needs the wasm runtime installed once:
//
//   just wasm-web-wgpu
//   cd cera_ffi_flutter/example
//   dart run cera_ffi_flutter:install_web --from ../../cera-wasm/examples/webgpu/pkg
//   flutter run -d chrome

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'benchmark.dart';
import 'model_source.dart';

void main() => runApp(const CeraExampleApp());

class CeraExampleApp extends StatelessWidget {
  const CeraExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    const bgDark = Color(0xFF0B0C0E);
    const surfaceDark = Color(0xFF14161B);
    const borderDark = Color(0xFF232732);
    const textPrimary = Color(0xFFF1F5F9);
    const textMuted = Color(0xFF94A3B8);
    const primaryBlue = Color(0xFF3B82F6);

    final darkScheme = ColorScheme.dark(
      surface: surfaceDark,
      primary: primaryBlue,
      onPrimary: Colors.white,
      secondary: const Color(0xFF64748B),
      onSurface: textPrimary,
      onSurfaceVariant: textMuted,
      outline: borderDark,
      outlineVariant: const Color(0xFF1C1E26),
      surfaceContainerLowest: const Color(0xFF07080A),
      surfaceContainerLow: const Color(0xFF0F1115),
      surfaceContainer: surfaceDark,
      surfaceContainerHigh: const Color(0xFF1B1E26),
      surfaceContainerHighest: const Color(0xFF232732),
      error: const Color(0xFFEF4444),
      onError: Colors.white,
    );

    return MaterialApp(
      title: 'Cera Example',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: darkScheme,
        scaffoldBackgroundColor: bgDark,
        canvasColor: bgDark,
        appBarTheme: const AppBarTheme(
          backgroundColor: bgDark,
          surfaceTintColor: Colors.transparent,
          elevation: 0,
          scrolledUnderElevation: 0,
          titleTextStyle: TextStyle(
            color: textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w600,
            letterSpacing: -0.2,
          ),
          iconTheme: IconThemeData(color: textPrimary),
        ),
        cardTheme: CardThemeData(
          color: surfaceDark,
          surfaceTintColor: Colors.transparent,
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
            side: const BorderSide(color: borderDark, width: 1),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: primaryBlue,
            foregroundColor: Colors.white,
            disabledBackgroundColor: const Color(0xFF1C2230),
            disabledForegroundColor: const Color(0xFF475569),
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(10),
            ),
            textStyle: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 14,
            ),
          ),
        ),
        iconButtonTheme: IconButtonThemeData(
          style: IconButton.styleFrom(
            foregroundColor: textPrimary,
            disabledForegroundColor: const Color(0xFF475569),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: surfaceDark,
          hintStyle: const TextStyle(color: Color(0xFF64748B), fontSize: 14),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 14,
          ),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: borderDark, width: 1),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: borderDark, width: 1),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: primaryBlue, width: 1.5),
          ),
          disabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFF1A1C24), width: 1),
          ),
        ),
        dividerTheme: const DividerThemeData(
          color: borderDark,
          thickness: 1,
          space: 1,
        ),
      ),
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: darkScheme,
        scaffoldBackgroundColor: bgDark,
      ),
      home: const ChatPage(),
    );
  }
}

/// Generation statistics for an assistant turn.
class TurnStats {
  const TurnStats({
    required this.tokens,
    required this.totalMs,
    this.ttftMs,
    required this.tokensPerSecond,
  });

  final int tokens;
  final int totalMs;
  final int? ttftMs;
  final double tokensPerSecond;
}

/// One message in the transcript.
class Turn {
  Turn({
    required this.role,
    required this.text,
    this.imageBytes,
    this.imageName,
    this.isGenerating = false,
    this.statusText,
    this.stats,
  });

  final String role;
  String text;
  final Uint8List? imageBytes;
  final String? imageName;
  bool isGenerating;
  String? statusText;
  TurnStats? stats;

  bool get isUser => role == 'user';
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  final _turns = <Turn>[];

  /// The loaded model, kept alive across turns so the conversation shares one
  /// KV cache and the weights are paid for once.
  Cera? _cera;
  StreamSubscription<String>? _generation;

  /// The loaded model descriptor (file or bundle), retained so the benchmark
  /// page can run side-by-side CPU vs GPU measurements on the same model.
  LoadedModel? _loadedModel;

  /// Releases the `_send` that is waiting on the current turn. See its use.
  void Function()? _finishTurn;

  /// An attached image waiting to be sent with the next prompt.
  Uint8List? _pendingImageBytes;
  String? _pendingImageName;

  String _status = 'No model loaded';
  bool _loading = false;

  /// Download completion in `0.0..1.0` while a bundle is being fetched, or null
  /// for "not downloading, or downloading something of unknown size". Both
  /// nulls render the same way (no determinate bar), so they need not be
  /// distinguished here.
  double? _downloadFraction;

  bool get _busy => _loading || _generation != null;

  @override
  void initState() {
    super.initState();
    _restoreLastModel();
  }

  /// Restores the previously loaded bundle model when the page reloads.
  Future<void> _restoreLastModel() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final bundleName = prefs.getString('cera_last_bundle_name');
      final quant = prefs.getString('cera_last_bundle_quant');
      if (bundleName != null && quant != null && mounted) {
        final displayName = bundleName.endsWith('-GGUF')
            ? bundleName.substring(0, bundleName.length - '-GGUF'.length)
            : bundleName;
        final label = '$displayName · $quant';
        final bundleSource = BundleModelSource(
          name: label,
          bundleName: bundleName,
          quant: quant,
          getStoreDir: _storeDir,
        );
        await _load(
          bundleSource,
          () async => Cera.openBundle(
            bundleName,
            quant,
            storeDir: await _storeDir(),
            onProgress: (progress) {
              if (!mounted) return;
              setState(() {
                _downloadFraction = progress.fraction;
                final pct = progress.fraction == null
                    ? '${(progress.bytesDownloaded / 1024 / 1024).toStringAsFixed(0)} MB'
                    : '${(progress.fraction! * 100).toStringAsFixed(0)}%';
                _status = 'Downloading $label · $pct';
              });
            },
          ),
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

  @override
  void dispose() {
    unawaited(_generation?.cancel());
    // Cancelling a subscription suppresses `onDone`, so without this the
    // pending `_send` frame would hold this State, its transcript and the
    // engine closure for the life of the app. Same reason `_stop` calls it.
    _finishTurn?.call();
    unawaited(_cera?.close());
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  /// Runs `open` as the app's one model-loading path, with the bookkeeping that
  /// every loader needs: closing the previous model, the busy flag, the
  /// dispose-during-load guard, and error reporting.
  ///
  /// Shared by the file picker and the bundle menu. They differ only in how
  /// they produce a [Cera], and having each carry its own copy of this is how
  /// the two quietly drift apart on which of these steps they remember.
  Future<void> _load(LoadedModel model, Future<Cera> Function() open) async {
    // Guarded here, not only on the buttons that call it. Both entry points
    // await a dialog first, and the file picker's is a browser-native dialog
    // that does not disable the Flutter buttons behind it, so a second load can
    // legitimately arrive while the first is still running. Two in flight would
    // each open an engine and the last to finish would overwrite the other's
    // without closing it, leaking a full set of model weights. That is the
    // hazard `_pickBundle` reasons about below; keeping the invariant in the
    // function that depends on it means a future caller cannot forget it.
    //
    // Says so rather than dropping the request silently: the user picked a file
    // or a bundle and would otherwise see the status line still naming the
    // first load, with no sign the second went nowhere.
    if (_loading) {
      setState(() => _status = 'Still loading; ignored ${model.name}');
      return;
    }
    setState(() {
      _loading = true;
      _downloadFraction = null;
      _status = 'Loading ${model.name}…';
      _turns.clear();
      _loadedModel = null;
      _pendingImageBytes = null;
      _pendingImageName = null;
    });

    try {
      // Close any previous model first: two sets of weights will not fit
      // alongside each other on a phone, and on the web they compete for one
      // wasm heap. Inside the try, so a failure here cannot leave `_loading`
      // stuck true and the whole UI disabled with nothing shown.
      await _cera?.close();
      _cera = null;
      final cera = await open();
      // Every setState here follows an await, so it needs the guard: loading a
      // multi-hundred-megabyte model takes long enough for the page to be
      // disposed underneath it.
      if (!mounted) {
        await cera.close();
        return;
      }
      final visionTag = cera.capabilities.imageIn ? ' · Vision' : '';
      setState(() {
        _cera = cera;
        _loadedModel = model;
        _status = '${model.name} · ${cera.backend}$visionTag';
      });

      try {
        final prefs = await SharedPreferences.getInstance();
        if (model is BundleModelSource) {
          await prefs.setString('cera_last_bundle_name', model.bundleName);
          await prefs.setString('cera_last_bundle_quant', model.quant);
        } else {
          await prefs.remove('cera_last_bundle_name');
          await prefs.remove('cera_last_bundle_quant');
        }
      } catch (_) {}
    } catch (err, stack) {
      // Log as well as display: the status line truncates, and the full message
      // is the only thing that says which step failed.
      debugPrint('cera: model failed to load: $err\n$stack');
      if (mounted) setState(() => _status = 'Failed to load: $err');
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
          _downloadFraction = null;
        });
      }
    }
  }

  Future<void> _pickModel() async {
    final ModelSource? source;
    try {
      source = await pickModelSource(dialogTitle: 'Choose a .gguf model');
    } catch (err) {
      // The pick itself failing, whether the picker threw or the file it
      // returned cannot be read. Nothing awaits this method, so without a catch
      // the failure would leave the zone as an unhandled error and the user
      // with no sign that anything happened.
      if (mounted) setState(() => _status = 'Could not open a model: $err');
      return;
    }
    if (source == null || !mounted) return;
    // Rebound so the closures below see a non-nullable value: a local declared
    // without an initializer does not promote inside one.
    final model = source;

    await _load(model, () => model.open());
  }

  /// Where bundle downloads are cached.
  ///
  /// Only Android and iOS need an answer: an app there may write solely inside
  /// its own container, which no environment variable names, so `openBundle`
  /// refuses to guess rather than failing partway into a download. Desktop
  /// falls through to the default, `$HOME/.cache/cera`, which is the CLI's own
  /// cache, so a model pulled by `cera chat` is already here, and vice versa.
  /// The web takes a single directory NAME inside the origin's private
  /// filesystem rather than a path, so a native path would be rejected there;
  /// null takes its default, which is what you want.
  Future<String?> _storeDir() async {
    if (kIsWeb) return null;
    final mobile =
        defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;
    if (!mobile) return null;
    return (await getApplicationSupportDirectory()).path;
  }

  /// Offers the published catalog, then downloads and opens what was chosen.
  Future<void> _pickBundle() async {
    String? currentBundleName;
    String? currentQuant;
    final loaded = _loadedModel;
    if (loaded is BundleModelSource) {
      currentBundleName = loaded.bundleName;
      currentQuant = loaded.quant;
    }

    final choice = await showDialog<_BundleChoice>(
      context: context,
      builder: (_) => _BundlePickerDialog(
        currentBundleName: currentBundleName,
        currentQuant: currentQuant,
      ),
    );
    if (choice == null || !mounted) return;

    final label = '${choice.bundle.displayName} · ${choice.quant}';
    final bundleSource = BundleModelSource(
      name: label,
      bundleName: choice.bundle.name,
      quant: choice.quant,
      getStoreDir: _storeDir,
    );

    await _load(
      bundleSource,
      // `_storeDir()` is awaited INSIDE the callback, not before `_load`. On
      // mobile it is a real platform-channel round trip, and awaiting it out
      // here would leave the buttons enabled (nothing sets `_loading` until
      // `_load` runs) long enough to start a second load: both would see
      // `_cera` still null, and whichever finished last would overwrite the
      // other's engine without closing it, leaking the model weights.
      () async => Cera.openBundle(
        choice.bundle.name,
        choice.quant,
        storeDir: await _storeDir(),
        onProgress: (progress) {
          // Guarded before setState: progress keeps arriving for a moment after
          // the page is disposed, since disposal does not cancel the download.
          if (!mounted) return;
          setState(() {
            _downloadFraction = progress.fraction;
            final pct = progress.fraction == null
                ? '${(progress.bytesDownloaded / 1024 / 1024).toStringAsFixed(0)} MB'
                : '${(progress.fraction! * 100).toStringAsFixed(0)}%';
            _status = 'Downloading $label · $pct';
          });
        },
      ),
    );
  }

  Future<void> _pickImage() async {
    final cera = _cera;
    if (cera == null || !cera.capabilities.imageIn || _busy) return;

    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
        dialogTitle: 'Select an image to attach',
      );
      final file = result?.files.single;
      if (file == null || !mounted) return;

      final bytes = file.bytes;
      if (bytes == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not read image file bytes')),
          );
        }
        return;
      }
      setState(() {
        _pendingImageBytes = bytes;
        _pendingImageName = file.name;
      });
    } catch (err) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not attach image: $err')));
      }
    }
  }

  Future<void> _send() async {
    final prompt = _input.text.trim();
    final cera = _cera;
    final imageBytes = _pendingImageBytes;
    final imageName = _pendingImageName;
    if ((prompt.isEmpty && imageBytes == null) || cera == null || _busy) return;

    _input.clear();
    final assistantTurn = Turn(
      role: 'assistant',
      text: '',
      isGenerating: true,
      statusText: imageBytes != null ? 'Analyzing image...' : 'Thinking...',
    );
    setState(() {
      _pendingImageBytes = null;
      _pendingImageName = null;
      _turns.add(
        Turn(
          role: 'user',
          text: prompt,
          imageBytes: imageBytes,
          imageName: imageName,
        ),
      );
      _turns.add(assistantTurn);
    });
    _scrollToBottom();

    // If an image is attached, append it into the session before prompt generation.
    if (imageBytes != null) {
      try {
        await cera.appendImage(imageBytes);
      } catch (err) {
        if (mounted) {
          setState(() {
            assistantTurn.isGenerating = false;
            assistantTurn.text = 'Error appending image: $err';
          });
        }
        return;
      }
    }

    if (mounted) {
      setState(() => assistantTurn.statusText = 'Generating...');
    }

    final messagePrompt = prompt.isEmpty ? 'Describe this image.' : prompt;
    String framed;
    try {
      framed = await cera.applyChatTemplate([CeraMessage.user(messagePrompt)]);
    } on StateError catch (err) {
      if (mounted) {
        setState(() {
          assistantTurn.isGenerating = false;
          assistantTurn.text = 'Error: $err';
        });
      }
      return;
    } catch (_) {
      framed = messagePrompt;
    }
    if (!mounted) return;

    // Completed from the stream's terminal callbacks AND from `_stop`.
    // Cancelling a subscription suppresses `onDone`, so a Stop press would
    // otherwise leave the `await` below pending for the life of the app.
    final done = Completer<void>();
    _finishTurn = () {
      if (!done.isCompleted) done.complete();
    };

    final stopwatch = Stopwatch()..start();
    int tokenCount = 0;
    int? firstTokenMs;

    final sub = cera
        .generate(framed, maxTokens: 256)
        .listen(
          (piece) {
            firstTokenMs ??= stopwatch.elapsedMilliseconds;
            tokenCount++;
            setState(() => assistantTurn.text += piece);
            _scrollToBottom();
          },
          onError: (Object err) {
            setState(() {
              assistantTurn.isGenerating = false;
              assistantTurn.text = 'Error: $err';
            });
            if (!done.isCompleted) done.complete();
          },
          onDone: () {
            if (!done.isCompleted) done.complete();
          },
          cancelOnError: true,
        );
    if (!mounted) {
      await sub.cancel();
      return;
    }
    setState(() => _generation = sub);

    await done.future;
    stopwatch.stop();
    final totalMs = stopwatch.elapsedMilliseconds;
    final ttft = firstTokenMs;
    final decodeMs = ttft != null ? (totalMs - ttft) : totalMs;
    final tps = tokenCount > 1 && decodeMs > 0
        ? ((tokenCount - 1) / (decodeMs / 1000.0))
        : (tokenCount == 1 && totalMs > 0
              ? (tokenCount / (totalMs / 1000.0))
              : 0.0);

    if (mounted) {
      setState(() {
        assistantTurn.isGenerating = false;
        if (tokenCount > 0) {
          assistantTurn.stats = TurnStats(
            tokens: tokenCount,
            totalMs: totalMs,
            ttftMs: ttft,
            tokensPerSecond: tps,
          );
        }
      });
    }

    _finishTurn = null;
    await sub.cancel();
    if (mounted) setState(() => _generation = null);
  }

  void _stop() {
    // Cancel the SUBSCRIPTION, not the engine. `Cera.cancel` is best-effort and
    // reaches neither web backend's running decode, whereas dropping the
    // subscription stops delivery immediately on every platform, which is what
    // a Stop button owes the user. The stream's own onCancel still asks the
    // engine to stop where it can.
    //
    // Deliberately not awaited: on the web's CPU backend that future waits on a
    // worker reply the worker cannot dequeue until its synchronous decode
    // finishes, so awaiting it would leave the Stop button and the disabled
    // input up for the rest of a decode the user just stopped.
    unawaited(_generation?.cancel());
    _finishTurn?.call();
    setState(() => _generation = null);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cera'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(28),
          child: Container(
            decoration: const BoxDecoration(
              border: Border(
                bottom: BorderSide(color: Color(0xFF1E222D), width: 1),
              ),
            ),
            padding: const EdgeInsets.only(left: 16, right: 16, bottom: 8),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Row(
                children: [
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _cera != null
                          ? const Color(0xFF10B981)
                          : (_loading
                                ? const Color(0xFFF59E0B)
                                : const Color(0xFF64748B)),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _status,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: const Color(0xFF94A3B8),
                        fontFamily: 'monospace',
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        actions: [
          IconButton(
            // Disabled until a model is loaded, or while this page is busy
            // (loading or generating).
            onPressed: (_busy || _loadedModel == null)
                ? null
                : () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => BenchmarkPage(model: _loadedModel!),
                    ),
                  ),
            icon: const Icon(Icons.speed),
            tooltip: 'Benchmark CPU vs GPU',
          ),
          IconButton(
            onPressed: _busy ? null : _pickBundle,
            icon: const Icon(Icons.cloud_download_outlined),
            tooltip: 'Download a published model',
          ),
          IconButton(
            onPressed: _busy ? null : _pickModel,
            icon: const Icon(Icons.folder_open),
            tooltip: 'Open a .gguf model',
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: Column(
        children: [
          // Up for any load, not only a download: `_downloadFraction` is null
          // for a local file open and for a server that sent no length, and
          // `value: null` is an indeterminate bar, which is the honest
          // rendering of both.
          if (_loading)
            LinearProgressIndicator(
              value: _downloadFraction,
              backgroundColor: const Color(0xFF14161B),
              valueColor: const AlwaysStoppedAnimation<Color>(
                Color(0xFF3B82F6),
              ),
              minHeight: 2,
            ),
          Expanded(
            child: _turns.isEmpty
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: const Color(0xFF14161B),
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: const Color(0xFF232732),
                              ),
                            ),
                            child: const Icon(
                              Icons.chat_bubble_outline_rounded,
                              size: 32,
                              color: Color(0xFF64748B),
                            ),
                          ),
                          const SizedBox(height: 16),
                          const Text(
                            'Download a published model, or open a .gguf, to start.',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Color(0xFF94A3B8),
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  )
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 16,
                    ),
                    itemCount: _turns.length,
                    itemBuilder: (context, i) => _Bubble(turn: _turns[i]),
                  ),
          ),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: const BoxDecoration(
              color: Color(0xFF0B0C0E),
              border: Border(
                top: BorderSide(color: Color(0xFF1E222D), width: 1),
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_pendingImageBytes != null) ...[
                  Container(
                    margin: const EdgeInsets.only(bottom: 10),
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: const Color(0xFF14161B),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: const Color(0xFF232732)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(6),
                          child: Image.memory(
                            _pendingImageBytes!,
                            width: 42,
                            height: 42,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Flexible(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                _pendingImageName ?? 'Attached image',
                                style: const TextStyle(
                                  color: Color(0xFFF1F5F9),
                                  fontSize: 13,
                                  fontWeight: FontWeight.w500,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                              Text(
                                '${(_pendingImageBytes!.lengthInBytes / 1024).toStringAsFixed(0)} KB',
                                style: const TextStyle(
                                  color: Color(0xFF94A3B8),
                                  fontSize: 11,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
                        IconButton(
                          icon: const Icon(Icons.close_rounded, size: 18),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(
                            minWidth: 28,
                            minHeight: 28,
                          ),
                          color: const Color(0xFF94A3B8),
                          onPressed: () {
                            setState(() {
                              _pendingImageBytes = null;
                              _pendingImageName = null;
                            });
                          },
                          tooltip: 'Remove image',
                        ),
                      ],
                    ),
                  ),
                ],
                Row(
                  children: [
                    if (_cera?.capabilities.imageIn == true) ...[
                      IconButton(
                        onPressed: _busy ? null : _pickImage,
                        icon: Icon(
                          _pendingImageBytes != null
                              ? Icons.image_rounded
                              : Icons.add_photo_alternate_outlined,
                          color: _pendingImageBytes != null
                              ? const Color(0xFF60A5FA)
                              : null,
                        ),
                        tooltip: 'Attach image for vision model',
                      ),
                      const SizedBox(width: 4),
                    ],
                    Expanded(
                      child: TextField(
                        controller: _input,
                        enabled: _cera != null && !_busy,
                        onSubmitted: (_) => _send(),
                        style: const TextStyle(
                          color: Color(0xFFF1F5F9),
                          fontSize: 14,
                        ),
                        decoration: InputDecoration(
                          hintText: _cera?.capabilities.imageIn == true
                              ? (_pendingImageBytes != null
                                    ? 'Ask about this image…'
                                    : 'Ask something, or attach an image…')
                              : 'Ask something…',
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    if (_generation != null)
                      IconButton.filled(
                        onPressed: _stop,
                        style: IconButton.styleFrom(
                          backgroundColor: const Color(0xFFDC2626),
                          foregroundColor: Colors.white,
                        ),
                        icon: const Icon(Icons.stop_rounded),
                        tooltip: 'Stop generating',
                      )
                    else
                      IconButton.filled(
                        onPressed: _cera != null && !_busy ? _send : null,
                        style: IconButton.styleFrom(
                          backgroundColor: const Color(0xFF2563EB),
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: const Color(0xFF161820),
                          disabledForegroundColor: const Color(0xFF475569),
                        ),
                        icon: _loading
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    Color(0xFF94A3B8),
                                  ),
                                ),
                              )
                            : const Icon(Icons.arrow_upward_rounded),
                      ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// A bundle and one of its quantizations, chosen from the catalog.
class _BundleChoice {
  const _BundleChoice(this.bundle, this.quant);

  /// The catalog entry. Carried whole rather than as a bare id so the display
  /// name comes from `CeraBundle.displayName` instead of being trimmed again
  /// here.
  final CeraBundle bundle;

  final String quant;
}

/// Lists the published bundles and pops with the chosen `<name>, <quant>`.
class _BundlePickerDialog extends StatefulWidget {
  const _BundlePickerDialog({this.currentBundleName, this.currentQuant});

  final String? currentBundleName;
  final String? currentQuant;

  @override
  State<_BundlePickerDialog> createState() => _BundlePickerDialogState();
}

class _BundlePickerDialogState extends State<_BundlePickerDialog> {
  // Held as a Future and given to a FutureBuilder rather than resolved into
  // state, so the request is issued exactly once: initState runs once, whereas
  // build runs on every expansion tile toggle.
  late final Future<List<CeraBundle>> _bundles = Cera.listBundles();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      backgroundColor: const Color(0xFF14161B),
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Color(0xFF232732)),
      ),
      title: const Text('Published models'),
      content: SizedBox(
        width: 440,
        height: 480,
        child: FutureBuilder<List<CeraBundle>>(
          future: _bundles,
          builder: (context, snapshot) {
            if (snapshot.hasError) {
              // The catalog is one request with a 30 second deadline and no
              // retry, so the useful thing to show is the reason plus a way to
              // give up; a spinner that never ends would be worse.
              return Center(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    'Could not reach the catalog:\n${snapshot.error}',
                    style: TextStyle(color: theme.colorScheme.error),
                  ),
                ),
              );
            }
            final rawBundles = snapshot.data;
            if (rawBundles == null) {
              return const Center(
                child: CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF3B82F6)),
                ),
              );
            }
            final bundles = rawBundles.toList()
              ..sort(
                (a, b) => a.displayName.toLowerCase().compareTo(
                  b.displayName.toLowerCase(),
                ),
              );
            return ListView.separated(
              itemCount: bundles.length,
              separatorBuilder: (_, _) =>
                  const Divider(color: Color(0xFF1E222D), height: 1),
              itemBuilder: (context, i) {
                final bundle = bundles[i];
                final isCurrentBundle = bundle.name == widget.currentBundleName;
                final quants = bundle.quants.toList()
                  ..sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase()));
                final n = quants.length;
                return ExpansionTile(
                  // Without a key the expanded state is not preserved across
                  // ListView recycling, so expanding a tile and scrolling away
                  // collapses it: only about six of the ~29 entries fit.
                  key: PageStorageKey(bundle.name),
                  initiallyExpanded: isCurrentBundle,
                  title: Row(
                    children: [
                      Expanded(
                        child: Text(
                          bundle.displayName,
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ),
                      if (isCurrentBundle)
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 7,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: const Color(0xFF0E3E2F),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: const Color(0x6610B981)),
                          ),
                          child: const Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.check_circle_rounded,
                                size: 12,
                                color: Color(0xFF34D399),
                              ),
                              SizedBox(width: 4),
                              Text(
                                'Active',
                                style: TextStyle(
                                  color: Color(0xFF34D399),
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                  subtitle: Text(
                    '$n quantization${n == 1 ? "" : "s"}',
                    style: const TextStyle(
                      color: Color(0xFF8E95A5),
                      fontSize: 12,
                    ),
                  ),
                  children: [
                    for (final quant in quants)
                      Builder(
                        builder: (context) {
                          final isLoadedQuant =
                              isCurrentBundle && quant == widget.currentQuant;
                          return Container(
                            color: isLoadedQuant
                                ? const Color(0xFF162338)
                                : null,
                            child: ListTile(
                              dense: true,
                              title: Row(
                                children: [
                                  Text(
                                    quant,
                                    style: TextStyle(
                                      fontFamily: 'monospace',
                                      fontWeight: isLoadedQuant
                                          ? FontWeight.w700
                                          : FontWeight.normal,
                                      color: isLoadedQuant
                                          ? const Color(0xFF93C5FD)
                                          : null,
                                    ),
                                  ),
                                  if (isLoadedQuant) ...[
                                    const SizedBox(width: 8),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 6,
                                        vertical: 1,
                                      ),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFF1E3A5F),
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: const Text(
                                        'Loaded',
                                        style: TextStyle(
                                          color: Color(0xFF60A5FA),
                                          fontSize: 10,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                    ),
                                  ],
                                ],
                              ),
                              trailing: Icon(
                                isLoadedQuant
                                    ? Icons.check_rounded
                                    : Icons.download_rounded,
                                size: 18,
                                color: isLoadedQuant
                                    ? const Color(0xFF60A5FA)
                                    : const Color(0xFF94A3B8),
                              ),
                              onTap: () => Navigator.of(
                                context,
                              ).pop(_BundleChoice(bundle, quant)),
                            ),
                          );
                        },
                      ),
                  ],
                );
              },
            );
          },
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator({this.label});

  final String? label;

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1200),
  )..repeat();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          if (widget.label != null && widget.label!.isNotEmpty) ...[
            Text(
              widget.label!,
              style: const TextStyle(
                color: Color(0xFF94A3B8),
                fontSize: 13,
                fontStyle: FontStyle.italic,
              ),
            ),
            const SizedBox(width: 8),
          ],
          AnimatedBuilder(
            animation: _controller,
            builder: (context, _) {
              return Row(
                mainAxisSize: MainAxisSize.min,
                children: List.generate(3, (index) {
                  final offset = index * 0.2;
                  final raw = (_controller.value - offset) % 1.0;
                  final wave = math.sin(raw * math.pi);
                  final scale = 0.6 + 0.4 * (wave < 0 ? 0 : wave);
                  final opacity = 0.25 + 0.75 * (wave < 0 ? 0 : wave);
                  return Container(
                    margin: const EdgeInsets.symmetric(horizontal: 2.5),
                    width: 6.5,
                    height: 6.5,
                    decoration: BoxDecoration(
                      color: Color.fromRGBO(
                        56,
                        189,
                        248,
                        opacity.clamp(0.2, 1.0),
                      ),
                      shape: BoxShape.circle,
                    ),
                    transform: Matrix4.diagonal3Values(scale, scale, 1.0),
                  );
                }),
              );
            },
          ),
        ],
      ),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.turn});

  final Turn turn;

  @override
  Widget build(BuildContext context) {
    final isUser = turn.isUser;
    final showTyping = !isUser && turn.text.isEmpty;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        constraints: const BoxConstraints(maxWidth: 580),
        decoration: BoxDecoration(
          color: isUser ? const Color(0xFF2563EB) : const Color(0xFF14161B),
          border: isUser
              ? null
              : Border.all(color: const Color(0xFF232732), width: 1),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
        ),
        child: Column(
          crossAxisAlignment: isUser
              ? CrossAxisAlignment.end
              : CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (turn.imageBytes != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Container(
                  constraints: const BoxConstraints(maxHeight: 240),
                  child: Image.memory(turn.imageBytes!, fit: BoxFit.contain),
                ),
              ),
              if (turn.text.isNotEmpty || showTyping) const SizedBox(height: 8),
            ],
            if (showTyping)
              _TypingIndicator(label: turn.statusText ?? 'Generating...')
            else if (turn.text.isNotEmpty || turn.imageBytes == null)
              Text(
                turn.text,
                style: TextStyle(
                  color: isUser ? Colors.white : const Color(0xFFF1F5F9),
                  fontSize: 14,
                  height: 1.45,
                ),
              ),
            if (turn.stats != null) ...[
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF0B0C0E),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: const Color(0xFF1E222D)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.bolt_rounded,
                      size: 13,
                      color: Color(0xFF38BDF8),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      '${turn.stats!.tokens} tokens · ${turn.stats!.tokensPerSecond.toStringAsFixed(1)} tok/s · ${(turn.stats!.totalMs / 1000.0).toStringAsFixed(2)}s${turn.stats!.ttftMs != null ? " (TTFT ${turn.stats!.ttftMs}ms)" : ""}',
                      style: const TextStyle(
                        fontSize: 11,
                        color: Color(0xFF8E95A5),
                        fontFamily: 'monospace',
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

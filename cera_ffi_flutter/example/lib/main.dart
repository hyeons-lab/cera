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

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'benchmark.dart';
import 'model_source.dart';

void main() => runApp(const CeraExampleApp());

class CeraExampleApp extends StatelessWidget {
  const CeraExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cera Example',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const ChatPage(),
    );
  }
}

/// One message in the transcript.
class Turn {
  Turn({required this.role, required this.text});

  final String role;
  String text;

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

  String _status = 'No model loaded';
  bool _loading = false;

  /// Download completion in `0.0..1.0` while a bundle is being fetched, or null
  /// for "not downloading, or downloading something of unknown size". Both
  /// nulls render the same way (no determinate bar), so they need not be
  /// distinguished here.
  double? _downloadFraction;

  bool get _busy => _loading || _generation != null;

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
      setState(() {
        _cera = cera;
        _loadedModel = model;
        _status = '${model.name} · ${cera.backend}';
      });
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
    final choice = await showDialog<_BundleChoice>(
      context: context,
      builder: (_) => const _BundlePickerDialog(),
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

  Future<void> _send() async {
    final prompt = _input.text.trim();
    final cera = _cera;
    if (prompt.isEmpty || cera == null || _busy) return;

    _input.clear();
    setState(() {
      _turns.add(Turn(role: 'user', text: prompt));
      _turns.add(Turn(role: 'assistant', text: ''));
    });
    _scrollToBottom();

    // Render the turn through the model's chat template so it answers rather
    // than continuing the transcript. A GGUF without one throws, and a raw
    // prompt is the honest fallback there. A closed engine throws here too, so
    // report that rather than pressing on into a generate that cannot work.
    String framed;
    try {
      framed = await cera.applyChatTemplate([CeraMessage.user(prompt)]);
    } on StateError catch (err) {
      if (mounted) setState(() => _turns.last.text = 'Error: $err');
      return;
    } catch (_) {
      framed = prompt;
    }
    if (!mounted) return;

    // Completed from the stream's terminal callbacks AND from `_stop`.
    // Cancelling a subscription suppresses `onDone`, so a Stop press would
    // otherwise leave the `await` below pending for the life of the app.
    final done = Completer<void>();
    _finishTurn = () {
      if (!done.isCompleted) done.complete();
    };
    final sub = cera
        .generate(framed, maxTokens: 256)
        .listen(
          (piece) {
            setState(() => _turns.last.text += piece);
            _scrollToBottom();
          },
          onError: (Object err) {
            setState(() => _turns.last.text = 'Error: $err');
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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cera'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(24),
          child: Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, bottom: 8),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                _status,
                style: Theme.of(context).textTheme.bodySmall,
                overflow: TextOverflow.ellipsis,
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
        ],
      ),
      body: Column(
        children: [
          // Up for any load, not only a download: `_downloadFraction` is null
          // for a local file open and for a server that sent no length, and
          // `value: null` is an indeterminate bar, which is the honest
          // rendering of both.
          if (_loading) LinearProgressIndicator(value: _downloadFraction),
          Expanded(
            child: _turns.isEmpty
                ? const Center(
                    child: Text(
                      'Download a published model, or open a .gguf, to start.',
                    ),
                  )
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(12),
                    itemCount: _turns.length,
                    itemBuilder: (context, i) => _Bubble(turn: _turns[i]),
                  ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _input,
                    enabled: _cera != null && !_busy,
                    onSubmitted: (_) => _send(),
                    decoration: const InputDecoration(
                      hintText: 'Ask something…',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                if (_generation != null)
                  IconButton.filled(
                    onPressed: _stop,
                    icon: const Icon(Icons.stop),
                    tooltip: 'Stop generating',
                  )
                else
                  IconButton.filled(
                    onPressed: _cera != null && !_busy ? _send : null,
                    icon: _loading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.send),
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
  const _BundlePickerDialog();

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
    return AlertDialog(
      title: const Text('Published models'),
      content: SizedBox(
        width: 420,
        height: 460,
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
                  ),
                ),
              );
            }
            final bundles = snapshot.data;
            if (bundles == null) {
              return const Center(child: CircularProgressIndicator());
            }
            return ListView.builder(
              itemCount: bundles.length,
              itemBuilder: (context, i) {
                final bundle = bundles[i];
                final n = bundle.quants.length;
                return ExpansionTile(
                  // Without a key the expanded state is not preserved across
                  // ListView recycling, so expanding a tile and scrolling away
                  // collapses it: only about six of the ~29 entries fit.
                  key: PageStorageKey(bundle.name),
                  title: Text(bundle.displayName),
                  subtitle: Text('$n quantization${n == 1 ? "" : "s"}'),
                  children: [
                    for (final quant in bundle.quants)
                      ListTile(
                        dense: true,
                        title: Text(quant),
                        onTap: () => Navigator.of(
                          context,
                        ).pop(_BundleChoice(bundle, quant)),
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

class _Bubble extends StatelessWidget {
  const _Bubble({required this.turn});

  final Turn turn;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: turn.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: const BoxConstraints(maxWidth: 520),
        decoration: BoxDecoration(
          color: turn.isUser
              ? scheme.primaryContainer
              : scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(turn.text.isEmpty ? '…' : turn.text),
      ),
    );
  }
}

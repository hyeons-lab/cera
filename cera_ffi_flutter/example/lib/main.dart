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
import 'package:flutter/material.dart';

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

  /// Releases the `_send` that is waiting on the current turn. See its use.
  void Function()? _finishTurn;

  String _status = 'No model loaded';
  bool _loading = false;

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

    setState(() {
      _loading = true;
      _status = 'Loading ${model.name}…';
      _turns.clear();
    });

    try {
      // Close any previous model first: two sets of weights will not fit
      // alongside each other on a phone, and on the web they compete for one
      // wasm heap. Inside the try, so a failure here cannot leave `_loading`
      // stuck true and the whole UI disabled with nothing shown.
      await _cera?.close();
      _cera = null;
      // Not `reusable`: this page opens the model once and keeps it, so paying
      // for a second copy of the weights on the web would buy nothing.
      final cera = await model.open();
      // Every setState here follows an await, so it needs the guard: loading a
      // multi-hundred-megabyte model takes long enough for the page to be
      // disposed underneath it.
      if (!mounted) {
        await cera.close();
        return;
      }
      setState(() {
        _cera = cera;
        _status = '${model.name} · ${cera.backend}';
      });
    } catch (err, stack) {
      // Log as well as display: the status line truncates, and the full message
      // is the only thing that says which step failed.
      debugPrint('cera: model failed to load: $err\n$stack');
      if (mounted) setState(() => _status = 'Failed to load: $err');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
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
            // Disabled while this page is busy. Not because the benchmark
            // shares state with it (it opens its own engines from its own
            // pick), but because a second set of weights loading alongside a
            // generation in flight is how a browser tab runs out of memory.
            // An idle chat model stays resident either way; this rules out the
            // concurrent case, not coexistence.
            onPressed: _busy
                ? null
                : () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const BenchmarkPage(),
                    ),
                  ),
            icon: const Icon(Icons.speed),
            tooltip: 'Benchmark CPU vs GPU',
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
          Expanded(
            child: _turns.isEmpty
                ? const Center(child: Text('Open a .gguf model to start.'))
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

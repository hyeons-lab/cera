// A side-by-side CPU vs GPU generation benchmark.
//
// The point is the browser. `Cera` picks WebGPU when it can and falls back to
// the wasm CPU build when it cannot, and that fallback is silent by design, so
// nothing in the chat UI tells you which one answered or what it cost. On the
// numbers this project measures, WebGPU runs ~58 tok/s against ~1.4 for wasm
// CPU on the same machine and model, which is a 40x difference a user can feel
// but not name. This page names it.
//
// It runs the same prompt twice, once on each backend, from the same weights,
// and reports what each cost.
//
// ## Why this page opens its own model
//
// On the web `openBytes` **transfers** the model's `ArrayBuffer` into the
// worker, which neuters the caller's view of it (see `_detach` in
// `cera_ffi/lib/src/async/cera_web.dart`). So the chat page's bytes are gone
// the moment its engine loads, and two engines need two buffers regardless.
// Picking here keeps a pristine master and hands each run its own copy, rather
// than reaching into the chat page's state for bytes that no longer exist.
//
// ## Native
//
// `CeraBackend.gpu` is documented as behaving like `auto` natively, because
// "the GPU" is Metal on Apple and wgpu elsewhere and only `auto` probes for
// whichever exists. So off the web the second row is "whatever auto picked",
// not necessarily a GPU, and the UI says so rather than implying a comparison
// it did not run.

import 'dart:async';
import 'dart:typed_data';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

/// The prompt every run shares.
///
/// Short on purpose: a long one would spend the measurement on prefill, and
/// prefill and decode are separately reported below precisely because they
/// scale differently.
const _prompt = 'In one sentence, why is the sky blue?';

/// Tokens to decode per run.
///
/// Small because the CPU arm is the slow one and sets the wait: at the ~1.4
/// tok/s the wasm CPU build measures in a browser, 32 tokens is already ~20
/// seconds of watching a spinner. Long enough to average out a cold first
/// token, short enough that nobody gives up.
const _maxTokens = 32;

/// What one arm of the benchmark measured.
class _BenchResult {
  _BenchResult.ok({
    required this.label,
    required this.backend,
    required this.promptTokens,
    required this.ttft,
    required this.decodeTokens,
    required this.decodeSpan,
    required this.output,
  }) : error = null;

  _BenchResult.failed({required this.label, required this.error})
    : backend = null,
      promptTokens = 0,
      ttft = Duration.zero,
      decodeTokens = 0,
      decodeSpan = Duration.zero,
      output = '';

  /// What was asked for ("CPU", "GPU"), which is not always what ran.
  final String label;

  /// What the engine reports it actually used, adapter included on the web.
  final String? backend;

  final int promptTokens;

  /// Time from the `generate` call to the first fragment.
  ///
  /// Prefill plus one decode step, not prefill alone: the first fragment cannot
  /// exist until a token has been sampled. Reported as time-to-first-token
  /// rather than as a prefill rate for that reason.
  final Duration ttft;

  /// Tokens produced, counted by re-encoding the output rather than by counting
  /// stream events. The stream emits fragments, and a fragment is not a token:
  /// a token can be a partial word and a multi-byte character can span several
  /// of them, so counting events would report a number that merely looks right.
  final int decodeTokens;

  /// First fragment to last, i.e. the decode loop with prefill excluded.
  final Duration decodeSpan;

  final String output;
  final String? error;

  bool get ok => error == null;

  /// Tokens per second over the decode loop.
  ///
  /// The first token is excluded from both halves: it arrived at [ttft], so
  /// counting it here would credit the decode rate with prefill's work. Null
  /// when a run was too short to have a rate at all.
  double? get decodeTokensPerSecond {
    final tokens = decodeTokens - 1;
    final seconds = decodeSpan.inMicroseconds / 1e6;
    if (tokens < 1 || seconds <= 0) return null;
    return tokens / seconds;
  }
}

/// Runs one arm: open on `backend`, generate, measure, close.
///
/// Takes an opener rather than a `Cera` because each arm needs its own engine,
/// and rather than bytes because how the model is opened is platform-specific.
/// On the web there is no path, so the caller hands over a private copy of the
/// bytes (the worker transfers what it is given, leaving the original
/// unreadable). On native the caller opens the path instead and the engine maps
/// the file: passing bytes there would pull a multi-gigabyte GGUF through the
/// Dart heap once per arm, for nothing.
Future<_BenchResult> _runBenchmark({
  required Future<Cera> Function(CeraOptions) open,
  required CeraBackend backend,
  required String label,
}) async {
  Cera? cera;
  try {
    cera = await open(CeraOptions(backend: backend));

    // Frame the prompt the way the chat page does, so the measurement covers
    // the path an app actually takes. A GGUF without a template is not an
    // error worth failing the run over; the raw prompt still generates.
    //
    // Keep the reason, though. An instruct model handed a bare prompt predicts
    // EOS as its first token and decodes nothing, so this fallback is the most
    // likely cause of the "produced no output" failure below. Swallowing the
    // error reports a dead benchmark with no hint of why it died, and the
    // causes are worth telling apart: a base model with no template at all is
    // expected, whereas a template that exists and fails to render is a bug in
    // the engine (one such: a template calling Python methods minijinja lacks).
    String framed;
    String? templateError;
    try {
      framed = await cera.applyChatTemplate([CeraMessage.user(_prompt)]);
    } catch (e) {
      framed = _prompt;
      templateError = '$e';
    }
    // `addSpecial: false`: `framed` came out of the chat template, which has
    // already emitted the model's BOS. Encoding with the default would prepend
    // a second one (and append EOS when the GGUF asks for it), so the reported
    // prompt size would not be the prompt that was actually fed.
    final promptTokens = (await cera.encode(framed, addSpecial: false)).length;

    final started = Stopwatch()..start();
    Duration? ttft;
    final buffer = StringBuffer();

    // `temperature: 0` for greedy decoding, so both arms do the same work.
    // The web's GPU backend decodes greedily whatever it is passed, so without
    // this the CPU arm would be sampling while the GPU arm was not, and the
    // comparison would carry a difference that is not the backend.
    await for (final piece in cera.generate(
      framed,
      maxTokens: _maxTokens,
      temperature: 0,
    )) {
      ttft ??= started.elapsed;
      buffer.write(piece);
    }
    final total = started.elapsed;
    final output = buffer.toString();

    // A run that produced nothing has no rate to report, and dividing by its
    // zero-length decode span would invent one.
    if (ttft == null || output.isEmpty) {
      return _BenchResult.failed(
        label: label,
        error: templateError == null
            ? 'the model produced no output'
            : 'the model produced no output; its chat template failed to '
                  'render, so the prompt went in unframed: $templateError',
      );
    }

    // `addSpecial: false`: this is a fragment of a generation, not a fresh
    // prompt, so prepending BOS would inflate the count by one.
    final decodeTokens = (await cera.encode(output, addSpecial: false)).length;

    return _BenchResult.ok(
      label: label,
      backend: cera.backend,
      promptTokens: promptTokens,
      ttft: ttft,
      decodeTokens: decodeTokens,
      decodeSpan: total - ttft,
      output: output,
    );
  } catch (err) {
    // Expected, not exceptional: `CeraBackend.gpu` is documented to fail rather
    // than fall back on the web, so a machine without WebGPU lands here. That is
    // a result to display, not a crash.
    return _BenchResult.failed(label: label, error: '$err');
  } finally {
    await cera?.close();
  }
}

class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key});

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  /// Web only. Kept pristine and never handed to an engine directly: every run
  /// gets a copy, because the web transfers (and so neuters) whatever it is
  /// given. Null on native, where [_path] is used instead.
  Uint8List? _master;

  /// Native only. Both arms open this path independently and the engine maps
  /// the file, so no copy of the weights passes through the Dart heap.
  String? _path;
  String _modelName = '';

  final _results = <_BenchResult>[];
  String? _running;
  bool _picking = false;

  bool get _busy => _running != null || _picking;

  Future<void> _pickModel() async {
    setState(() => _picking = true);
    try {
      final result = await FilePicker.platform.pickFiles(
        dialogTitle: 'Choose a .gguf model to benchmark',
        type: FileType.any,
        // Same rule as the chat page: ask for bytes only where there is no
        // path to open. Requesting them on native reads the whole GGUF into
        // the Dart heap, which for a multi-gigabyte model is slow at best and
        // an OOM at worst, and it throws away the memory mapping the engine
        // would otherwise use. Two arms do not change that: each opens the
        // path itself.
        withData: !Cera.supportsPaths,
      );
      final file = result?.files.single;
      if (file == null) return;
      final path = file.path;
      if (path == null && file.bytes == null) return;
      if (!mounted) return;
      setState(() {
        _path = path;
        _master = path == null ? file.bytes : null;
        _modelName = file.name;
        _results.clear();
      });
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  Future<void> _run() async {
    final master = _master;
    final path = _path;
    if ((master == null && path == null) || _busy) return;

    setState(() => _results.clear());

    // CPU first. It is the slow arm, and running it while the page is fresh
    // means the GPU arm is not the one waiting on a garbage collection the CPU
    // arm's allocations caused.
    for (final (label, backend) in <(String, CeraBackend)>[
      ('CPU', CeraBackend.cpu),
      ('GPU', CeraBackend.gpu),
    ]) {
      setState(() => _running = label);
      final result = await _runBenchmark(
        // Native opens the path twice and lets the engine map it. The web has
        // no path, so each arm gets a fresh copy: the previous one was
        // transferred into the worker and is no longer readable. `sublist(0)`
        // rather than `Uint8List.fromList`, which copies element by element.
        open: path != null
            ? (options) => Cera.openPath(path, options: options)
            : (options) => Cera.openBytes(master!.sublist(0), options: options),
        backend: backend,
        label: label,
      );
      if (!mounted) return;
      setState(() => _results.add(result));
    }
    if (mounted) setState(() => _running = null);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('CPU vs GPU')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Runs the same prompt on both backends and reports what each cost.',
            style: theme.textTheme.bodyMedium,
          ),
          const SizedBox(height: 4),
          Text(
            kIsWeb
                ? 'In the browser this is WebGPU against the wasm CPU build. '
                      'The CPU run is the slow one; give it a moment.'
                : 'Natively, asking for the GPU behaves as "auto", so the '
                      'second run is whichever backend auto picked. The browser '
                      'is where this comparison is a real one.',
            style: theme.textTheme.bodySmall,
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : _pickModel,
                icon: const Icon(Icons.folder_open),
                label: const Text('Choose model'),
              ),
              const SizedBox(width: 12),
              FilledButton.icon(
                // Either source arms the button: native holds a path and no
                // bytes, the web the reverse.
                onPressed: (_master == null && _path == null) || _busy
                    ? null
                    : _run,
                icon: const Icon(Icons.speed),
                label: const Text('Run'),
              ),
            ],
          ),
          if (_modelName.isNotEmpty) ...[
            const SizedBox(height: 12),
            Text('Model: $_modelName', style: theme.textTheme.bodySmall),
          ],
          if (_running != null) ...[
            const SizedBox(height: 20),
            Row(
              children: [
                const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                const SizedBox(width: 12),
                Text('Running $_running… ($_maxTokens tokens)'),
              ],
            ),
          ],
          const SizedBox(height: 20),
          for (final r in _results) ...[
            _ResultCard(result: r),
            const SizedBox(height: 12),
          ],
          if (_results.length == 2) _Speedup(results: _results),
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});

  final _BenchResult result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final rate = result.decodeTokensPerSecond;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(result.label, style: theme.textTheme.titleMedium),
                const Spacer(),
                if (rate != null)
                  Text(
                    '${rate.toStringAsFixed(1)} tok/s',
                    style: theme.textTheme.titleMedium,
                  ),
              ],
            ),
            const SizedBox(height: 6),
            if (!result.ok)
              Text(
                result.error!,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              )
            else ...[
              // The resolved backend, not the requested one: on the web `auto`
              // can silently land on the CPU, and this is the line that would
              // show it.
              Text(
                'Backend: ${result.backend}',
                style: theme.textTheme.bodySmall,
              ),
              Text(
                'Time to first token: ${result.ttft.inMilliseconds} ms '
                '(prefill of ${result.promptTokens} tokens, plus one decode step)',
                style: theme.textTheme.bodySmall,
              ),
              Text(
                'Decoded ${result.decodeTokens} tokens in '
                '${result.decodeSpan.inMilliseconds} ms',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 8),
              Text(
                result.output.trim(),
                style: theme.textTheme.bodySmall?.copyWith(
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// The headline number, shown only when both arms produced one.
class _Speedup extends StatelessWidget {
  const _Speedup({required this.results});

  final List<_BenchResult> results;

  @override
  Widget build(BuildContext context) {
    final cpu = results[0].decodeTokensPerSecond;
    final gpu = results[1].decodeTokensPerSecond;
    if (cpu == null || gpu == null || cpu <= 0 || gpu <= 0) {
      return const SizedBox.shrink();
    }
    // Do not assume the GPU won. Natively the second row is whatever `auto`
    // picked, which can be the CPU again, and even a real GPU can lose to
    // per-token overhead on a small model. Reporting "0.8x faster" would be a
    // plainly false sentence on a page whose whole purpose is an honest
    // number, so state the slower case as a slowdown and the tie as a tie.
    final ratio = gpu / cpu;
    final headline = switch (ratio) {
      _ when ratio >= 1.05 =>
        'GPU decoded ${ratio.toStringAsFixed(1)}x faster than CPU.',
      _ when ratio <= 0.95 =>
        'GPU decoded ${(1 / ratio).toStringAsFixed(1)}x slower than CPU.',
      _ => 'GPU and CPU decoded at about the same rate.',
    };
    return Card(
      color: Theme.of(context).colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Text(headline, style: Theme.of(context).textTheme.titleMedium),
      ),
    );
  }
}

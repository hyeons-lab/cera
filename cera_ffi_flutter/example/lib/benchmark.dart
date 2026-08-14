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
// ## Model source
//
// Two engines need two opens, and on the web a load may consume the buffer it
// is handed: see `ModelSource.open`, which owns that rule for both pages.
// This page can reuse a `ModelSource` chosen by the chat page or pick its own,
// and opens it once per arm with `reusable: true`.
//
// ## Native
//
// `CeraBackend.gpu` is documented as behaving like `auto` natively, because
// "the GPU" is Metal on Apple and wgpu elsewhere and only `auto` probes for
// whichever exists. So off the web the second row is "whatever auto picked",
// not necessarily a GPU, and the UI says so rather than implying a comparison
// it did not run.

import 'dart:async' show unawaited;

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import 'model_source.dart';

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
/// seconds of watching a spinner. Long enough that a per-token rate means
/// something, short enough that nobody gives up.
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
    this.note,
  }) : error = null;

  _BenchResult.failed({required this.label, required this.error})
    : backend = null,
      promptTokens = 0,
      ttft = Duration.zero,
      decodeTokens = 0,
      decodeSpan = Duration.zero,
      output = '',
      note = null;

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
  ///
  /// Re-encoding is an estimate too, just a closer one. A BPE round trip is not
  /// an identity: text the model emitted as several tokens can re-encode as
  /// one, a generation cut off at [_maxTokens] can end mid-word, and any EOS
  /// the model produced is not in the text to be counted at all. The engine
  /// does not report its own token count through this API, so the honest
  /// summary is that this is within a token or two, not exact.
  final int decodeTokens;

  /// First fragment to last, i.e. the decode loop with prefill excluded.
  final Duration decodeSpan;

  final String output;
  final String? error;

  /// A caveat about a run that otherwise succeeded, or null when there is none.
  ///
  /// A number produced under a caveat is worse than no number if the caveat is
  /// invisible: a template that failed to render still generates, just from an
  /// unframed prompt, which is not the path the page claims to be measuring.
  final String? note;

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
/// Takes an opener rather than a `Cera`, because each arm needs its own engine
/// and this function is the thing that owns an engine's lifetime. What opening
/// means per platform is `ModelSource.open`'s business, not this one's.
///
/// `onEngine` receives the engine once it is open and null once it is closed,
/// so the caller can reach it while it is live. Teardown is the only reason
/// that hook exists: closing a run the page walked away from, whether it was
/// mid-generation or still loading.
Future<_BenchResult> _runBenchmark({
  required Future<Cera> Function(CeraOptions) open,
  required CeraBackend backend,
  required String label,
  required void Function(Cera?) onEngine,
}) async {
  Cera? cera;
  try {
    cera = await open(CeraOptions(backend: backend));
    onEngine(cera);

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
    // `addSpecial: false`, because the default appends EOS when the GGUF asks
    // for it, and an EOS is not part of a prompt. The engine handles BOS on its
    // own: it prepends one when the GGUF asks and the framed text does not
    // already start with it, which is the same rule on both transports.
    //
    // So this is a close count, not an exact one. Against a template that emits
    // BOS itself it is exact; against a model whose GGUF wants BOS and whose
    // template does not emit one, the engine prefills one token more than the
    // card reports. Off by one on a figure quoted beside a millisecond timing
    // is worth stating rather than hiding.
    final promptTokens = (await cera.encode(framed, addSpecial: false)).length;

    final started = Stopwatch()..start();
    Duration? ttft;
    final buffer = StringBuffer();

    // Stamped as each fragment arrives, so the decode span ends at the last
    // token rather than at the stream's close. On the web those are not the
    // same instant: the worker posts its final token event and then the reply
    // that ends the stream, so the close trails the last token by a message
    // delivery. Small, but it is postMessage latency rather than decoding, and
    // it lands on the arm with the smaller numbers.
    Duration? lastPiece;

    // `temperature: 0` for greedy decoding on both arms. Sampling is honored on
    // every backend, the web's GPU one included, so leaving it at the default
    // would not skew the comparison by itself. Pinning it still buys two
    // things: identical work on both arms rather than two different token
    // streams, and the GPU arm's fast path, since sampling there reads the
    // whole logits row back per token where greedy reads back one id.
    await for (final piece in cera.generate(
      framed,
      maxTokens: _maxTokens,
      temperature: 0,
    )) {
      // One read, not two. Reading the stopwatch separately for each would put
      // a microsecond between them on the first fragment, and a run that
      // emitted only that fragment would then have a positive span to divide
      // by: one token over one microsecond, reported as a million tok/s,
      // instead of the page declining to state a rate it does not have.
      final at = started.elapsed;
      ttft ??= at;
      lastPiece = at;
      buffer.write(piece);
    }
    final total = lastPiece ?? started.elapsed;
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
      note: templateError == null
          ? null
          : 'this model\'s chat template failed to render, so the prompt went '
                'in unframed: $templateError',
    );
  } catch (err) {
    // Expected, not exceptional: `CeraBackend.gpu` is documented to fail rather
    // than fall back on the web, so a machine without WebGPU lands here. That is
    // a result to display, not a crash.
    return _BenchResult.failed(label: label, error: '$err');
  } finally {
    // Teardown failures are swallowed on purpose. An exception thrown from a
    // `finally` replaces whatever the block was about to return, so a close
    // that failed after a clean run would discard the measurement and surface
    // as the arm's result instead. The engine is unreachable after this either
    // way; what it was worth measuring has already been measured.
    try {
      await cera?.close();
    } catch (_) {
      // Nothing to do: the result is already decided.
    }
    onEngine(null);
  }
}

class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key, this.initialSource});

  /// A model chosen elsewhere (e.g. the chat page) to benchmark without
  /// re-picking, or null to start without one.
  final ModelSource? initialSource;

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  /// The picked model, opened once per arm.
  ///
  /// Kept whole across both runs rather than consumed by the first, which is
  /// what `reusable` in [ModelSource.open] is for.
  ModelSource? _source;

  final _results = <_BenchResult>[];
  String? _running;
  bool _picking = false;

  /// The engine of the arm in flight, or null between arms.
  ///
  /// Held for one reason: so [dispose] can close it. Nothing in [build] reads
  /// it, and it deliberately does not drive any of the UI.
  Cera? _live;

  bool get _busy => _running != null || _picking;

  @override
  void initState() {
    super.initState();
    _source = widget.initialSource;
  }

  @override
  void dispose() {
    // `terminate`, not `close`. The page is gone, so the run has nowhere to
    // report to, and the orderly shutdown is the one that cannot be relied on
    // here: on the web `close` waits for a worker that is busy decoding, which
    // on the CPU backend means waiting out the whole run (~20 seconds at the
    // rate this page exists to show) with the model still resident.
    // `terminate` kills the worker outright and takes its heap with it.
    //
    // The in-flight `await for` then ends with an error, which `_runBenchmark`
    // already turns into a failed result, and the `mounted` check in [_run]
    // discards it and stops the loop before a second arm opens anything.
    //
    // Unawaited because dispose cannot be async, and safe against the
    // `finally` that will also close this engine: both calls are idempotent.
    unawaited(_live?.terminate());
    super.dispose();
  }

  Future<void> _pickModel() async {
    setState(() => _picking = true);
    try {
      final source = await pickModelSource(
        dialogTitle: 'Choose a .gguf model to benchmark',
      );
      // A cancelled dialog is not a failure and says nothing.
      if (source == null || !mounted) return;
      setState(() {
        _source = source;
        _results.clear();
      });
    } catch (err) {
      // Both the picker failing and a file that cannot be read land here, and
      // neither can be left silent: nothing awaits this method, so the error
      // would otherwise escape into the zone and the user would see only a
      // spinner that stopped.
      //
      // Drop the previous pick as well as reporting. A snackbar fades, and
      // leaving the old model named on screen with Run still armed would let
      // the next run measure the old file and look like it had honored the new
      // one.
      if (mounted) {
        setState(() {
          _source = null;
          _results.clear();
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not open a model: $err')));
      }
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  Future<void> _run() async {
    final source = _source;
    if (source == null || _busy) return;

    setState(() => _results.clear());

    // `finally`, because `_running` is what disables both buttons and spins the
    // indicator. `_runBenchmark` reports its own failures as results rather
    // than throwing, but "rather than" is not "never": anything escaping it
    // would otherwise leave the page permanently busy, with a spinner that
    // never stops and no way back short of leaving the page.
    try {
      // CPU first, and the order is load-bearing twice over: [build] hands the
      // speedup card these results by position, and running the slow arm first
      // puts its card on screen while the fast one is still being measured.
      for (final (label, backend) in <(String, CeraBackend)>[
        ('CPU', CeraBackend.cpu),
        ('GPU', CeraBackend.gpu),
      ]) {
        setState(() => _running = label);
        final result = await _runBenchmark(
          // `reusable`, because the second arm opens this same source: natively
          // that is the same path mapped twice, and on the web it is a fresh
          // copy of the bytes per arm rather than one buffer the first load may
          // have taken.
          open: (options) => source.open(options: options, reusable: true),
          backend: backend,
          label: label,
          onEngine: (engine) {
            _live = engine;
            // The window dispose cannot see on its own. Opening a model is the
            // longest part of an arm, and until it returns there is nothing to
            // close, so a page disposed during the load would otherwise leave
            // this arm to load and generate in full against a page that is
            // gone. Close it the moment it exists instead.
            //
            // `mounted` is the disposed test: this only ever runs from an async
            // continuation, and by then the framework has unmounted the element
            // that `dispose` belonged to.
            //
            // `close` rather than the `terminate` in [dispose], because nothing
            // is generating yet: this is the orderly case, where the engine can
            // release the model on its own terms.
            if (!mounted && engine != null) unawaited(engine.close());
          },
        );
        // Unmounted means dispose closed the engine out from under this arm, so
        // the result is a teardown artifact rather than a measurement. Dropping
        // it also ends the loop, which is what stops the second arm from
        // opening a model for a page that is gone.
        if (!mounted) return;
        setState(() => _results.add(result));
      }
    } finally {
      if (mounted) setState(() => _running = null);
    }
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
                onPressed: _source == null || _busy ? null : _run,
                icon: const Icon(Icons.speed),
                label: const Text('Run'),
              ),
            ],
          ),
          if (_source case final source?) ...[
            const SizedBox(height: 12),
            Text('Model: ${source.name}', style: theme.textTheme.bodySmall),
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
          if (_results.length == 2)
            _Speedup(cpu: _results[0], gpu: _results[1]),
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
              // What the engine says it opened. On the web that is the
              // resolved backend, adapter included. Natively it echoes the
              // preference instead ("native (auto)" for a GPU request), which
              // is the honest rendering of a platform where asking for the GPU
              // means asking auto to find one.
              Text(
                'Backend: ${result.backend}',
                style: theme.textTheme.bodySmall,
              ),
              Text(
                'Time to first token: ${result.ttft.inMilliseconds} ms '
                '(prefill of ${result.promptTokens} tokens, plus one decode step)',
                style: theme.textTheme.bodySmall,
              ),
              // Says which tokens the span covers, because the rate above is
              // over one fewer of them: the first arrived at TTFT and is
              // counted there. Without that the two lines look like they
              // should divide into each other, and they do not.
              //
              // Keyed off the rate rather than off the token count, so the two
              // lines cannot disagree. A run can decode several tokens and
              // still have no rate, since a fragment is only emitted once it
              // completes a character: hold one back and the last piece lands
              // at the same instant as the first, leaving a zero-length span
              // that "the last N of them in 0 ms" would report as real.
              Text(
                rate == null
                    ? 'Decoded ${result.decodeTokens} '
                          '${result.decodeTokens == 1 ? 'token' : 'tokens'}, '
                          'all at once, so there is no decode rate to report'
                    : 'Decoded ${result.decodeTokens} tokens, the last '
                          '${result.decodeTokens - 1} of them in '
                          '${result.decodeSpan.inMilliseconds} ms',
                style: theme.textTheme.bodySmall,
              ),
              if (result.note != null)
                Text(
                  result.note!,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
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
///
/// Named arms rather than a list, so the mapping from result to role is stated
/// where it is made. The call site still supplies it by position, because the
/// order is fixed by the loop that produced the results.
class _Speedup extends StatelessWidget {
  const _Speedup({required this.cpu, required this.gpu});

  final _BenchResult cpu;
  final _BenchResult gpu;

  @override
  Widget build(BuildContext context) {
    final cpuRate = cpu.decodeTokensPerSecond;
    final gpuRate = gpu.decodeTokensPerSecond;
    if (cpuRate == null || gpuRate == null || cpuRate <= 0 || gpuRate <= 0) {
      return const SizedBox.shrink();
    }
    // Do not assume the GPU won. Natively the second row is whatever `auto`
    // picked, which can be the CPU again, and even a real GPU can lose to
    // per-token overhead on a small model. Reporting "0.8x faster" would be a
    // plainly false sentence on a page whose whole purpose is an honest
    // number, so state the slower case as a slowdown and the tie as a tie.
    final ratio = gpuRate / cpuRate;
    final headline = switch (ratio) {
      >= 1.05 => 'GPU decoded ${ratio.toStringAsFixed(1)}x faster than CPU.',
      <= 0.95 =>
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

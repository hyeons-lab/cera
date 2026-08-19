/// Native implementation of [Cera], over the generated `dart:ffi` bindings.
///
/// No isolate anywhere in here, which is worth stating because it looks like
/// the obvious design. Loading and decoding, the two long operations, already
/// happen off the Dart thread: `from_path_async`, `from_bytes_async` and
/// `generate_streaming_async` are `#[uniffi::export(async_runtime = "tokio")]`
/// on the Rust side and run their bodies on a tokio blocking worker, completing
/// the Dart future from there. An isolate would add a hop and a hazard, since
/// the engine handle carries a [Finalizer] and shipping it between isolates
/// puts ownership in two places.
///
/// One step is **not** off-thread: `appendTokens`, and therefore prefill. It is
/// a plain synchronous export with no async twin, so a long prompt blocks the
/// calling isolate for the length of its prefill. Decode, the part that
/// dominates a short prompt, is unaffected. Fixing it properly means an
/// `append_tokens_async` on the Rust side rather than an isolate here.
library;

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import '../generated/cera_ffi.dart';
import 'cera.dart';

/// Maps the portable backend choice onto the native one.
///
/// [CeraBackend.gpu] becomes [BackendPreference.auto] rather than
/// [BackendPreference.gpu]: natively "the GPU" is two different backends, Metal
/// on Apple platforms and `wgpu` elsewhere, and `auto` is the only value that
/// probes for whichever exists. Requesting `gpu` on macOS would skip the faster
/// Metal path to insist on `wgpu`, which is not what the caller meant.
BackendPreference _backendOf(CeraBackend backend) => switch (backend) {
  CeraBackend.cpu => BackendPreference.cpu,
  CeraBackend.auto || CeraBackend.gpu => BackendPreference.auto,
};

EngineConfig _configOf(CeraOptions options, [BundleRepo? repo]) => EngineConfig(
  contextSize: options.contextSize,
  backend: _backendOf(options.backend),
  bundleRepo: repo,
);

/// Maps the generated capability record onto the portable one.
///
/// Restated rather than re-exported because the generated type is part of the
/// binding surface, and this API has to carry the same meaning on the web,
/// where those bindings are stubs.
CeraCapabilities _capabilitiesOf(ModalityCapabilities caps) => CeraCapabilities(
  textIn: caps.textIn,
  textOut: caps.textOut,
  imageIn: caps.imageIn,
  audioIn: caps.audioIn,
  audioOut: caps.audioOut,
);

/// This platform has a filesystem, so [Cera.openPath] works. See
/// [Cera.supportsPaths].
const bool supportsPaths = true;

/// Opens a model from the filesystem. See [Cera.openPath].
Future<Cera> openPath(String path, CeraOptions options) async => _NativeCera(
  await CeraEngine.fromPathAsync(path, _configOf(options)),
  options,
);

/// Opens a model from memory. See [Cera.openBytes].
///
/// Always `fromPartsAsync`, even with no mmproj: it is a superset of
/// `fromBytesAsync` (a null projector is exactly the text-only load), and
/// routing both through one constructor means the text and multimodal paths
/// cannot drift in how they resolve the inference type.
Future<Cera> openBytes(
  Uint8List bytes,
  CeraOptions options,
  Uint8List? mmproj,
  String? inferenceType,
) async => _NativeCera(
  await CeraEngine.fromPartsAsync(
    bytes,
    mmproj,
    inferenceType,
    _configOf(options),
  ),
  options,
);

/// Lists the published bundles. See [Cera.listBundles].
///
/// `options` is unused here and present only so the two implementations keep
/// identical signatures; it carries the web asset locations, which native has
/// no equivalent of.
Future<List<CeraBundle>> listBundles(CeraOptions options) async {
  // The async twin, so the HTTP round-trip lands on a tokio blocking worker
  // rather than stalling the isolate. A picker is usually opened from a button
  // press, which is exactly the thread not to block.
  final entries = await listLeapBundlesAsync();
  final list = [
    for (final entry in entries)
      CeraBundle(
        name: entry.name,
        quants: entry.quants.toList()
          ..sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase())),
      ),
  ];
  list.sort(
    (a, b) =>
        a.displayName.toLowerCase().compareTo(b.displayName.toLowerCase()),
  );
  return list;
}

/// Downloads and opens a published bundle. See [Cera.openBundle].
Future<Cera> openBundle(
  String name,
  String quant,
  CeraOptions options,
  void Function(CeraDownload progress)? onProgress,
  String? storeDir,
) async {
  final dir = storeDir ?? _defaultStoreDir();
  // Two constructors rather than always attaching a sink: `withProgress`
  // installs a callback vtable that Rust invokes from its download thread, and
  // a caller who did not ask for progress should not pay for that plumbing.
  final repo = onProgress == null
      ? BundleRepo.create(dir)
      : BundleRepo.withProgress(dir, _ProgressSink(onProgress));
  try {
    final engine = await CeraEngine.fromBundleIdAsync(
      name,
      quant,
      _configOf(options, repo),
    );
    return _NativeCera(engine, options);
  } finally {
    // The engine holds its own reference (the config lowers an `Arc` clone), so
    // releasing this handle does not disturb a loaded model. Without it the
    // repo would sit until its finalizer ran, holding the progress callback
    // registration alive with it.
    repo.close();
  }
}

/// Where downloads are cached when the caller names no directory.
///
/// `$HOME/.cache/cera`, which is exactly what the `cera` CLI uses, so a desktop
/// app and the CLI share one set of downloads rather than each pulling its own
/// copy of a multi-gigabyte model.
///
/// Deliberately unavailable on mobile. An app there writes inside its own
/// container and `HOME` either is unset or points somewhere unwritable, so a
/// default would surface as a permission failure partway into a download. The
/// throw names the parameter and the package that answers the question instead.
String _defaultStoreDir() {
  if (Platform.isAndroid || Platform.isIOS) {
    throw ArgumentError(
      'Cera.openBundle needs an explicit storeDir on Android and iOS: an app '
      'may only write inside its own container, which the environment does not '
      'name. Pass a directory from package:path_provider, e.g. '
      '(await getApplicationSupportDirectory()).path.',
    );
  }
  // `USERPROFILE` because Windows sets no `HOME`.
  final home =
      Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'];
  if (home == null || home.isEmpty) {
    throw ArgumentError(
      'Cera.openBundle could not choose a storeDir: neither HOME nor '
      'USERPROFILE is set. Pass storeDir explicitly.',
    );
  }
  return [home, '.cache', 'cera'].join(Platform.pathSeparator);
}

/// Forwards Rust's download callbacks to the caller's closure.
///
/// A class rather than a closure because the binding takes the trait. Its
/// vtable is installed as a cross-thread listener, so calls arrive on the
/// isolate's event loop rather than inline on the download thread; that makes
/// it safe to touch UI state from `onProgress` directly.
class _ProgressSink implements DownloadProgressSink {
  _ProgressSink(this.emit);

  final void Function(CeraDownload progress) emit;

  @override
  void onProgress(String url, int bytesDownloaded, int? totalBytes) {
    emit(
      CeraDownload(
        url: url,
        bytesDownloaded: bytesDownloaded,
        totalBytes: totalBytes,
      ),
    );
  }
}

class _NativeCera implements Cera {
  _NativeCera(this._engine, this._options)
    : _session = _engine.newSession(
        SessionConfig(
          kvCompression: _options.turboQuant
              ? const KvCompressionTurboQuant(seed: 0, keys: true, values: true)
              : null,
        ),
      ),
      // Read once. Both are fixed by the GGUF, and `metadata()` builds a whole
      // record across the FFI boundary, which is not something to do on every
      // prompt for two fields.
      _bosToken = _engine.metadata().addBosToken ? _engine.bosToken() : null,
      // Fixed by the bundle at load time, so reading it per query would cross
      // the FFI boundary for a record that cannot change.
      _capabilities = _capabilitiesOf(_engine.capabilities());

  final CeraEngine _engine;
  final CeraOptions _options;

  /// The BOS id to prepend, or null when this model does not want one.
  final int? _bosToken;

  final CeraCapabilities _capabilities;

  @override
  CeraCapabilities get capabilities => _capabilities;

  Session _session;
  bool _closed = false;
  List<int>? _pendingAudioSuffixTokens;

  /// Completes when the generation currently queued or running has finished.
  ///
  /// Generations are serialized rather than run concurrently or refused. They
  /// have to be serialized: they share one KV cache, and a second concurrent
  /// call would block the calling thread on the FFI `Session`'s mutex for the
  /// remainder of the first decode.
  ///
  /// Refusing was the first attempt and is worse in the case that matters. A
  /// Stop button cancels the subscription, but the decode behind it keeps
  /// running (it stops at its next between-token check, and on the web's GPU
  /// backend not at all), so the very next prompt would be rejected for a
  /// generation the user believes they already stopped. Waiting is invisible
  /// where cancel works and merely slow where it does not.
  Future<void> _queue = Future<void>.value();

  @override
  String get backend => 'native (${_backendOf(_options.backend).name})';

  void _ensureOpen() {
    if (_closed) {
      throw StateError('this Cera engine is closed');
    }
  }

  /// Frames a prompt into the ids to feed the session.
  ///
  /// Plain `encodeText` plus an explicit BOS rather than
  /// `encodeTextSpecial(prompt, true)`, because that convenience does two
  /// things and only one of them is wanted. It appends EOS as well whenever
  /// the GGUF sets `add_eos_token`, which ends the turn before the model has
  /// seen the generation header, and it prepends BOS without checking whether
  /// the text already produced one, which a chat template emitting its own BOS
  /// does.
  ///
  /// The two halves have different precedents. The encoding matches
  /// `Session::append_text`, which is also a plain `encode` and also adds no
  /// EOS. The BOS rule matches `WebGpuSession::generate`, which is where it is
  /// spelled out; `append_text` has no BOS handling at all, because the CLI
  /// frames its own prompts before calling it.
  ///
  /// BOS belongs at position 0 only. The KV cache persists across calls, so on
  /// a later turn there is nothing to prepend.
  List<int> _frame(String prompt) {
    final ids = _engine.encodeText(prompt);
    final bos = _bosToken;
    // `position()`, not a flag this class maintains. Whether BOS belongs here
    // is exactly "is the KV cache empty", and only the session knows: a prompt
    // can be rejected before any of it lands (cache untouched) or fail partway
    // through prefill (cache advanced), and the two need opposite answers.
    if (_session.position() == 0 &&
        bos != null &&
        (ids.isEmpty || ids.first != bos)) {
      return [bos, ...ids];
    }
    return ids;
  }

  @override
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double? temperature,
    double? topP,
    int? topK,
    int? seed,
    void Function(List<double> pcm, int sampleRate)? onAudio,
  }) {
    _ensureOpen();
    // A controller rather than an `async*` body: the tokens arrive on a
    // callback driven by Rust, not from anything this function can await.
    final controller = StreamController<String>();
    // Built from the session's default options (populated from the bundle
    // manifest when loaded from a bundle) rather than hardcoded constants,
    // allowing explicit caller overrides to take precedence.
    var opts = _session.defaultGenerateOpts().copyWith(
      maxTokens: maxTokens,
      // Emit per token. The default buffers 16, which turns a stream into four
      // lumps for a short reply.
      flushEveryTokens: 1,
    );
    if (temperature != null) opts = opts.copyWith(temperature: temperature);
    if (topP != null) opts = opts.copyWith(topP: topP);
    if (topK != null) opts = opts.copyWith(topK: topK);

    // Whether this generation has already produced its terminal event.
    //
    // `onCancel` fires on NORMAL completion too, not only on an explicit
    // `subscription.cancel()`: closing a controller cancels its subscription as
    // part of delivering done. Cancelling the session there would be actively
    // harmful rather than merely redundant, because the engine's cancel flag is
    // sticky (only `reset` and `clear_cancel` lower it). The next turn's
    // `appendTokens` runs before `generate` would have cleared it, and chunked
    // prefill checks the flag between chunks, so any later prompt longer than
    // one ubatch would fail with `Cancelled`.
    var finished = false;

    /// Whether this generation has reached the front of the queue and begun.
    ///
    /// Without it, cancelling a stream that is still WAITING its turn calls
    /// `_session.cancel()`, and the session is shared, so it truncates whoever
    /// is decoding right now. Serialization is what made queued-but-not-started
    /// a reachable state, and this is the flag that distinguishes it.
    var started = false;

    controller.onCancel = () async {
      if (started && !finished && !_closed) _session.cancel();
    };

    // Kick off after the caller has had a chance to subscribe; `onListen` is
    // the hook that guarantees it, and a stream nobody listens to should not
    // burn a decode.
    // Joined at listen, not at call, so a stream that is returned and never
    // listened to neither waits for its turn nor makes anything wait for it.
    controller.onListen = () async {
      final ahead = _queue;
      final mine = Completer<void>();
      _queue = mine.future;
      try {
        await ahead;
      } catch (_) {}
      // Release the queue BEFORE closing the controller on every early exit.
      // `close()`'s future does not complete while a subscription is paused, so
      // awaiting it first would leave `_queue` uncompleted and block every
      // later generation, plus `reset` and `cancel`, which both await it.
      if (_closed || !controller.hasListener) {
        mine.complete();
        // A subscription cancelled while queued wants no error and no decode;
        // a closed engine is a real failure worth reporting.
        if (_closed && controller.hasListener) {
          controller.addError(StateError('this Cera engine is closed'));
        }
        if (!controller.isClosed) await controller.close();
        return;
      }
      started = true;
      final sink = _StreamingSink(
        engine: _engine,
        emit: (piece) {
          if (!controller.isClosed) controller.add(piece);
        },
        onAudio: onAudio,
        isOpen: () => !_closed,
        done: (error) {
          finished = true;
          _clearCancel();
          if (!mine.isCompleted) mine.complete();
          if (controller.isClosed) return;
          if (error != null) controller.addError(error);
          controller.close();
        },
      );
      try {
        // `seed` is per session, not per call, so honoring it means a new
        // session. That also clears the conversation, so it is only honored on
        // a fresh engine. Inside the try because it closes the old session
        // before opening the new one, and a throw in between would otherwise
        // leave the engine holding a closed handle.
        if (seed != null && _session.position() == 0) {
          // Build the replacement BEFORE closing the old one. The other order
          // leaves `_session` a closed handle if `newSession` throws, and
          // `close()` calls `cancel()` on it first, which throws in turn and
          // skips `_engine.close()`: the model weights, the largest allocation
          // in the app, would leak until the finalizer ran.
          final reseeded = _engine.newSession(
            SessionConfig(
              seed: seed,
              kvCompression: _options.turboQuant
                  ? const KvCompressionTurboQuant(
                      seed: 0,
                      keys: true,
                      values: true,
                    )
                  : null,
            ),
          );
          _session.close();
          _session = reseeded;
        }
        List<int> tokens;
        final pendingSuffix = _pendingAudioSuffixTokens;
        _pendingAudioSuffixTokens = null;
        if (pendingSuffix != null && prompt.trim().isEmpty) {
          tokens = pendingSuffix;
        } else if (pendingSuffix != null) {
          tokens = [...pendingSuffix, ..._frame(prompt)];
        } else {
          tokens = _frame(prompt);
        }
        if (tokens.isNotEmpty) {
          _session.appendTokens(tokens);
        }
        unawaited(
          _session
              .generateStreamingAsync(opts, sink)
              .then((_) => sink.finish(), onError: sink.fail),
        );
      } on Object catch (e, st) {
        finished = true;
        _clearCancel();
        if (!mine.isCompleted) mine.complete();
        controller.addError(e, st);
        controller.close();
      }
    };
    return controller.stream;
  }

  @override
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  }) async {
    _ensureOpen();
    return _engine.applyChatTemplate(
      messages
          .map((m) => ChatMessage(role: m.role, content: m.content))
          .toList(),
      addGenerationPrompt,
    );
  }

  @override
  Future<List<int>> encode(String text, {bool addSpecial = true}) async {
    _ensureOpen();
    return _engine.encodeTextSpecial(text, addSpecial);
  }

  @override
  Future<String> decode(List<int> tokens) async {
    _ensureOpen();
    return _engine.decodeTokens(tokens);
  }

  @override
  Future<void> appendImage(Uint8List bytes, {int? maxLongSize}) async {
    // Queued behind any running generation for the same reason generations are
    // queued behind each other: this appends patch embeddings to the very KV
    // cache a decode is writing to, and interleaving the two would splice the
    // image into the middle of the answer.
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      _ensureOpen();
      _session.appendImage(bytes, maxLongSize);
    } finally {
      mine.complete();
    }
  }

  @override
  Future<void> appendAudio(
    List<double> pcm, {
    int sampleRate = 16000,
    String? prompt,
  }) async {
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      _ensureOpen();

      final markerCandidates = [
        '<|reserved_4|>',
        '<|reserved_5|>',
        '<|reserved_6|>',
        '<|reserved_7|>',
        '<|audio_start|>',
      ];
      String? markerName;
      int? markerId;
      for (final candidate in markerCandidates) {
        final id = _engine.specialTokenId(candidate);
        if (id != null) {
          markerName = candidate;
          markerId = id;
          break;
        }
      }
      markerName ??= '<|reserved_4|>';

      final userContent = (prompt != null && prompt.trim().isNotEmpty)
          ? '${prompt.trim()}\n$markerName'
          : markerName;

      String formatted;
      try {
        formatted = _engine.applyChatTemplate([
          ChatMessage(role: 'user', content: userContent),
        ], true);
      } catch (_) {
        formatted = userContent;
      }

      final allTokens = _engine.encodeText(formatted);
      final splitIdx = markerId != null ? allTokens.indexOf(markerId) : -1;

      if (splitIdx > 0) {
        _session.appendTokens(allTokens.sublist(0, splitIdx));
      } else if (splitIdx == -1 && prompt != null && prompt.trim().isNotEmpty) {
        final promptTokens = _frame(prompt);
        if (promptTokens.isNotEmpty) _session.appendTokens(promptTokens);
      }
      final floatList = pcm is Float32List ? pcm : Float32List.fromList(pcm);
      _session.appendAudio(floatList, sampleRate);
      if (splitIdx >= 0 && splitIdx + 1 < allTokens.length) {
        _pendingAudioSuffixTokens = allTokens.sublist(splitIdx + 1);
      } else {
        _pendingAudioSuffixTokens = null;
      }
    } finally {
      mine.complete();
    }
  }

  @override
  Future<String> transcribe(List<double> pcm, {required int sampleRate}) async {
    // Queued despite not touching the session: it is a full prefill plus decode
    // on the shared engine, so running it under a generation would contend on
    // the same model for the length of the clip.
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      _ensureOpen();
      return _engine.transcribe(pcm, sampleRate);
    } finally {
      mine.complete();
    }
  }

  @override
  Future<void> reset() async {
    _ensureOpen();
    // Cancels the running generation as well as clearing the conversation, as
    // `Cera.reset` documents.
    // Wait for any running decode first: resetting under one would leave it
    // emitting into a session state it no longer matches, and a GPU-backed
    // model shares KV state across sessions besides.
    _session.cancel();
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      _pendingAudioSuffixTokens = null;
      // Re-checked: `close()` can land during the await above, and reset would
      // then run on a disposed handle and surface the binding's raw error
      // instead of this one.
      _ensureOpen();
      // `Session.reset`, not a fresh session. It clears KV, position and token
      // history and lowers the cancel flag, all of which rebuilding also did,
      // but it keeps the session's own config: rebuilding with a default
      // `SessionConfig` silently discarded a seed installed by `generate(seed:)`.
      _session.reset();
    } finally {
      mine.complete();
    }
  }

  @override
  Future<void> cancel() async {
    if (_closed) return;
    _session.cancel();
    // A running decode lowers the flag from its terminal callback. With
    // nothing running there is no such callback, and leaving it raised would
    // fail the NEXT prompt's prefill rather than anything the caller did. The
    // worker's `cancel` op pairs the two calls for the same reason.
    await _queue;
    _clearCancel();
  }

  /// Lowers the cancel flag once a generation has ended.
  ///
  /// The flag is sticky: `Session::generate` clears it only on entry, which is
  /// after `appendTokens` has already run. Left raised, the next turn's
  /// chunked prefill sees it between chunks and fails with `Cancelled` for any
  /// prompt longer than one ubatch. Cancelling is therefore a two-step
  /// operation, and this is the second step; the worker's `cancel` op pairs
  /// them the same way.
  void _clearCancel() {
    if (!_closed) _session.clearCancel();
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    // Set first, so an in-flight sink stops touching the handles it is about
    // to lose. The decode itself is not interruptible from here; `cancel`
    // asks it to stop at its next between-token check, and the sink's
    // `isOpen` guard covers the window until it does.
    _closed = true;
    _session.cancel();
    _session.close();
    _engine.close();
  }

  @override
  // There is no faster stop here. `close` already cancels the session and drops
  // every handle, and the decode belongs to a background thread that stops at
  // its next between-token check either way. The distinction this method draws
  // exists for the web, where `close` waits on a worker that may be busy; see
  // `Cera.terminate`.
  Future<void> terminate() => close();
}

/// Turns the binding's token-id callbacks into a stream of text.
///
/// The decode is incremental and has to stay UTF-8-safe. A multi-byte character
/// can span several byte-fallback tokens, so detokenizing each chunk on its own
/// splits it and yields U+FFFD in place of a perfectly good character. Instead
/// every chunk re-decodes the whole prefix and emits the delta, holding back a
/// trailing U+FFFD until the token that completes it arrives.
///
/// Re-decoding the prefix is quadratic in tokens generated. It is also one FFI
/// call per emitted chunk over a buffer of a few hundred ids, against a decode
/// step that just ran a whole transformer, so the cost does not register.
class _StreamingSink implements ModalitySink {
  _StreamingSink({
    required this.engine,
    required this.emit,
    this.onAudio,
    required this.isOpen,
    required this.done,
  });

  /// Detokenizer. The session's engine, so the vocabulary matches.
  final CeraEngine engine;

  /// Receives each new fragment of text.
  final void Function(String) emit;

  /// Receives raw Float32 PCM audio chunks when model is in audio mode.
  final void Function(List<double> pcm, int sampleRate)? onAudio;

  /// Whether the engine is still open.
  ///
  /// Checked before every `decodeTokens`. A `close()` during an in-flight
  /// generate frees the engine handle while Rust is still delivering tokens,
  /// and the generated bindings answer a call on a freed handle by throwing
  /// from `_ensureOpen`. Thrown from here that error would escape into the
  /// `unawaited` future, leaving the terminal callback unfired and the caller
  /// awaiting a stream that never closes.
  final bool Function() isOpen;

  /// Called exactly once, with the error if the generation failed.
  final void Function(Object? error) done;

  final List<int> _ids = [];
  int _emitted = 0;
  bool _finished = false;

  /// The replacement character, the signal that a decode ended mid-sequence.
  static const _replacement = '\uFFFD';

  @override
  void onTextTokens(List<int> tokens) {
    if (tokens.isEmpty || _finished || !isOpen()) return;
    _ids.addAll(tokens);
    final full = engine.decodeTokens(_ids);
    // Hold back a trailing replacement char: it means the last token is half a
    // character, and the next one completes it. A genuine U+FFFD in the model's
    // output is merely delayed by one token.
    final stable = full.endsWith(_replacement)
        ? full.substring(0, full.length - _replacement.length)
        : full;
    if (stable.length > _emitted) {
      emit(stable.substring(_emitted));
      _emitted = stable.length;
    }
  }

  @override
  void onAudioFrames(List<double> pcm, int sampleRate) {
    if (_finished || !isOpen()) return;
    onAudio?.call(pcm, sampleRate);
  }

  @override
  void onDone(FinishReason reason) {
    // Deliberately not the completion signal. `onDone` fires on the Rust
    // blocking worker before the future resolves, and closing the controller
    // here would race the last `onTextTokens` delivery. `finish` runs when the
    // future completes, which orders after every callback.
  }

  /// Flushes any held-back partial character and reports success. Idempotent.
  void finish() {
    if (_finished) return;
    _finished = true;
    try {
      if (_ids.isNotEmpty && isOpen()) {
        final full = engine.decodeTokens(_ids);
        if (full.length > _emitted) emit(full.substring(_emitted));
      }
      done(null);
    } catch (e) {
      done(e);
    }
  }

  /// Reports failure. Idempotent, and mutually exclusive with [finish].
  void fail(Object error) {
    if (_finished) return;
    _finished = true;
    done(error);
  }
}

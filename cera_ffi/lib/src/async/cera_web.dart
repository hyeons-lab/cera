/// Web implementation of [Cera], over a Web Worker running `cera-wasm`.
///
/// The worker is not an optimization. The CPU backend's decode loop is one
/// synchronous wasm call that returns only when generation is finished, so on
/// the main thread it freezes the tab outright. Off it, the tab stays
/// responsive and per-token messages still arrive as they are produced, because
/// the thread receiving them is not the blocked one.
///
/// Everything crossing `postMessage` is plain JSON-ish data, so the worker has
/// no Dart dependency and can be driven from a hand-written page. The interop
/// surface below is hand-declared rather than taken from `package:web` to keep
/// this package's dependency list at what it actually needs; it is four APIs.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'cera.dart';

@JS('Worker')
extension type _Worker._(JSObject _) implements JSObject {
  external factory _Worker(String url, _WorkerOptions options);
  external void postMessage(JSAny? message, JSArray<JSAny> transfer);
  external set onmessage(JSFunction handler);
  external set onerror(JSFunction handler);
  external void terminate();
}

extension type _WorkerOptions._(JSObject _) implements JSObject {
  external factory _WorkerOptions({String type});
}

extension type _ImagePayload._(JSObject _) implements JSObject {
  external factory _ImagePayload({
    JSArrayBuffer bytes,
  });
}

extension type _AudioPayload._(JSObject _) implements JSObject {
  external factory _AudioPayload({
    JSAny pcm,
    int sampleRate,
  });
}

/// A request to the worker. One literal type covers every op; the fields an op
/// does not use are simply absent, which reaches JS as `undefined`.
extension type _Request._(JSObject _) implements JSObject {
  external factory _Request({
    int id,
    String op,
    String? moduleUrl,
    JSArrayBuffer? bytes,
    JSArrayBuffer? mmproj,
    int? contextSize,
    String? backend,
    String? inferenceType,
    int? maxLongSize,
    JSAny? pcm,
    int? sampleRate,
    String? prompt,
    int? maxTokens,
    double? temperature,
    double? topP,
    int? topK,
    int? seed,
    String? messagesJson,
    bool? addGenerationPrompt,
    String? text,
    bool? addSpecial,
    JSArray<JSNumber>? tokens,
    String? bundleId,
    String? quant,
    String? storeDir,
    bool? turboQuant,
    bool? wantsAudio,
    bool? wantsThought,
    JSArray<_ImagePayload>? images,
    _AudioPayload? audio,
  });
}

extension type _MessageEvent._(JSObject _) implements JSObject {
  external JSObject? get data;
}

extension type _ErrorEvent._(JSObject _) implements JSObject {
  external String? get message;
}

extension type _Reply._(JSObject _) implements JSObject {
  external int get id;
  external bool? get ok;
  external String? get event;
  external String? get text;
  external String? get error;
  external String? get kind;
  external JSAny? get result;
  // Download-progress fields, present only on `event: 'progress'`. The two
  // counts are `JSAny?` rather than `int?`: they cross as JS numbers and
  // `total` is genuinely null when the server sent no length, which a nullable
  // Dart primitive cannot represent across interop.
  external String? get url;
  external JSAny? get done;
  external JSAny? get total;
  // Audio streaming fields, present on `event: 'audio'`.
  external JSAny? get pcm;
  external JSAny? get sampleRate;
}

extension type _OpenResult._(JSObject _) implements JSObject {
  external String get backend;
  external _Capabilities get capabilities;
  external JSAny? get cancelBuffer;
}

@JS('Atomics')
external _Atomics get _atomics;

extension type _Atomics._(JSObject _) implements JSObject {
  external int store(JSInt32Array typedArray, int index, int value);
  external int load(JSInt32Array typedArray, int index);
}

/// One entry of the `listBundles` reply.
extension type _BundleEntry._(JSObject _) implements JSObject {
  external String get name;
  external JSArray<JSString> get quants;
}

extension type _Capabilities._(JSObject _) implements JSObject {
  external bool get textIn;
  external bool get textOut;
  external bool get imageIn;
  external bool get audioIn;
  external bool get audioOut;
}

/// The web has no filesystem, so [Cera.openPath] always throws. See
/// [Cera.supportsPaths].
const bool supportsPaths = false;

/// Opening from a path cannot work here: there is no filesystem to read.
///
/// This throws rather than being absent so that the two implementations keep
/// identical surfaces. A missing name would be a compile error on whichever
/// platform is built second, in a package the app never imported directly.
// `async`, so the failure arrives as a rejected future rather than a
// synchronous throw. The native twin returns a future, and a portable API whose
// error surfaces differently per platform is the thing this file exists to
// avoid: `Cera.openPath(...).catchError(...)` would work on one and blow up on
// the other.
Future<Cera> openPath(String path, CeraOptions options) async =>
    throw UnsupportedError(
      'Cera.openPath is not available on the web: a browser has no filesystem '
      'to read a GGUF from. Fetch or pick the file and use Cera.openBytes '
      'instead.',
    );

/// Opens a model from memory. See [Cera.openBytes].
Future<Cera> openBytes(
  Uint8List bytes,
  CeraOptions options,
  Uint8List? mmproj,
  String? inferenceType,
) async {
  final worker = _WorkerCera(options);
  try {
    await worker._start(bytes, mmproj, inferenceType);
  } on Object {
    // The worker is already constructed by the time `open` can fail, and a
    // failed open hands the caller nothing to `close()`. Without this, every
    // rejected file leaks a worker and whatever wasm heap it allocated, which
    // is the common case while a user is picking files.
    worker._shutDown(StateError('this Cera engine failed to open'));
    rethrow;
  }
  return worker;
}

/// Lists the published bundles. See [Cera.listBundles].
///
/// Spins up a worker, asks, and terminates it. That is heavier than a plain
/// HTTP request, and deliberate: the catalog is fetched and parsed by the same
/// Rust code the CLI uses, so a browser picker and `cera list-bundles` cannot
/// disagree about what exists. Doing it in Dart would mean a second parser and
/// a second set of rules about which entries are offerable.
Future<List<CeraBundle>> listBundles(CeraOptions options) async {
  final worker = _WorkerCera(options);
  try {
    worker._spawn();
    final result = await worker._send(
      worker._newId(),
      (id) => _Request(id: id, op: 'listBundles', moduleUrl: worker._moduleUrl),
    );
    final list = [
      for (final entry in (result as JSArray<_BundleEntry>).toDart)
        CeraBundle(
          name: entry.name,
          quants: [for (final q in entry.quants.toDart) q.toDart]
            ..sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase())),
        ),
    ];
    list.sort(
      (a, b) =>
          a.displayName.toLowerCase().compareTo(b.displayName.toLowerCase()),
    );
    return list;
  } finally {
    // Always, including on failure: a worker left running holds the wasm module
    // it imported, and a picker that is opened and closed repeatedly would
    // otherwise accumulate one per attempt.
    worker._shutDown(StateError('this Cera bundle listing is finished'));
  }
}

/// Downloads and opens a published bundle. See [Cera.openBundle].
Future<Cera> openBundle(
  String name,
  String quant,
  CeraOptions options,
  void Function(CeraDownload progress)? onProgress,
  String? storeDir,
) async {
  final worker = _WorkerCera(options);
  try {
    await worker._startBundle(name, quant, storeDir, onProgress);
  } on Object {
    // As in `openBytes`: the worker exists before the open can fail, and a
    // failed open hands the caller nothing to `close()`. A mistyped bundle name
    // is the common case here, so leaking one worker per attempt is easy to hit.
    worker._shutDown(StateError('this Cera engine failed to open'));
    rethrow;
  }
  return worker;
}

CeraCapabilities _capabilitiesOf(_Capabilities caps) => CeraCapabilities(
  textIn: caps.textIn,
  textOut: caps.textOut,
  imageIn: caps.imageIn,
  audioIn: caps.audioIn,
  audioOut: caps.audioOut,
);

class _WorkerCera implements Cera {
  _WorkerCera(this._options);

  final CeraOptions _options;

  /// Null until the worker is constructed.
  ///
  /// Not `late final`: `new Worker(url)` itself throws for an invalid or
  /// cross-origin `workerUrl`, and the failure path then runs [_shutDown],
  /// which would hit a LateInitializationError and bury the SecurityError that
  /// actually explains the problem.
  _Worker? _worker;

  /// In-flight requests, by id. Each resolves on its `{ok}` reply.
  final _pending = <int, Completer<JSAny?>>{};

  /// Sinks for streaming ops, by id. Present only while a generate is running.
  final _streams = <int, StreamController<String>>{};

  /// Progress callbacks for in-flight bundle downloads, by id.
  final _progress = <int, void Function(CeraDownload progress)>{};

  /// Callbacks for streaming audio frames, by id.
  final _audioCallbacks =
      <int, void Function(List<double> pcm, int sampleRate)>{};

  /// Callbacks for streaming thought chunks, by id.
  final _thoughtCallbacks = <int, void Function(String thought)>{};

  JSInt32Array? _cancelArray;

  int _nextId = 0;
  String _backend = 'unknown';
  bool _closed = false;

  /// Completes when the generation currently queued or running has finished.
  ///
  /// Serialized like the native path, and for a sharper reason. The worker's
  /// `onmessage` is `async`, so a second request is dequeued while a GPU
  /// `generateTokens` is still awaiting, and the two calls re-enter `&mut self`
  /// on the same `WebGpuSession`. wasm-bindgen answers that with "recursive use
  /// of an object detected", which tells the caller nothing.
  Future<void> _queue = Future<void>.value();

  @override
  String get backend => _backend;

  /// Reported by the worker as part of the open reply rather than fetched
  /// afterwards, so it costs no extra round trip and cannot be observed before
  /// it is known.
  late final CeraCapabilities _capabilities;

  @override
  CeraCapabilities get capabilities => _capabilities;

  /// The resolved wasm module URL, set by [_spawn]. Every op that may run
  /// before a model is open has to pass it, since the worker imports the module
  /// lazily and `listBundles` is typically the first call an app makes.
  String _moduleUrl = '';

  /// Starts the worker and installs its handlers, without opening anything.
  ///
  /// Split from [_start] so [listBundles] can reach the worker without a model:
  /// a picker runs before there is anything to load, and constructing a worker
  /// is the only way to reach the wasm catalog parser.
  void _spawn() {
    // Both URLs are resolved against the page before they leave Dart.
    //
    // `new Worker(url)` would resolve a relative URL against the document
    // anyway, but the module URL is used by a dynamic `import()` *inside* the
    // worker, where two things differ: the base is the worker script rather
    // than the page, and a specifier with no leading `./` is a bare specifier,
    // which an ES module rejects outright ("Failed to resolve module specifier").
    // Resolving here makes both absolute and removes the difference.
    final workerUrl = Uri.base.resolve(_options.web.workerUrl).toString();
    _moduleUrl = Uri.base.resolve(_options.web.moduleUrl).toString();
    final worker = _Worker(workerUrl, _WorkerOptions(type: 'module'));
    _worker = worker;
    worker.onmessage = ((_MessageEvent event) => _receive(event.data)).toJS;
    // Without this, a worker that fails to load never replies and every request
    // hangs forever with nothing logged. The usual cause is a bad `workerUrl`.
    worker.onerror =
        ((_ErrorEvent error) {
          // Shut down rather than only failing the in-flight batch. A worker that
          // errored answers nothing further, so leaving it "open" means every
          // later request parks a completer that cannot resolve, which is the hang
          // this handler exists to prevent.
          //
          // The event's own message leads, because `onerror` fires for two
          // different things: a worker that never loaded, and an uncaught error
          // inside one that did (a wasm out-of-memory mid-generate being the
          // likely one). Only the first is a path problem, so the path advice is
          // an afterthought rather than the headline.
          final detail = error.message ?? 'no detail from the worker';
          _shutDown(
            StateError(
              'the Cera web worker at "$workerUrl" failed: $detail. '
              'If it never loaded, check that cera_worker.js, cera_wasm.js and '
              'cera_wasm_bg.wasm are served from the configured paths (see '
              'CeraWebAssets), which `dart run cera_ffi:install_web` sets up '
              '(`dart run cera_ffi_flutter:install_web` from a Flutter app).',
            ),
          );
        }).toJS;
  }

  Future<void> _start(
    Uint8List bytes,
    Uint8List? mmproj,
    String? inferenceType,
  ) async {
    _spawn();
    final buffer = _detach(bytes);
    final projBuffer = mmproj == null ? null : _detach(mmproj);
    final result = await _send(
      _newId(),
      (id) => _Request(
        id: id,
        op: 'open',
        moduleUrl: _moduleUrl,
        bytes: buffer,
        mmproj: projBuffer,
        contextSize: _options.contextSize,
        backend: _options.backend.name,
        inferenceType: inferenceType,
        turboQuant: _options.turboQuant,
      ),
      // Hand the model's memory to the worker rather than copying it. A
      // multi-hundred-megabyte structured clone is both a pause and a moment
      // where the tab holds two copies. The projector rides along: it is the
      // smaller of the two but the same argument applies.
      transfer: <JSAny>[buffer, if (projBuffer != null) projBuffer],
    );
    _adoptOpenResult(result);
  }

  /// Downloads and opens a published bundle in this worker.
  ///
  /// Unlike [_start] nothing is transferred: the weights are fetched inside the
  /// worker and go straight into wasm memory, which is the whole reason this is
  /// a separate op rather than a download in Dart followed by `openBytes`.
  Future<void> _startBundle(
    String name,
    String quant,
    String? storeDir,
    void Function(CeraDownload progress)? onProgress,
  ) async {
    _spawn();
    final id = _newId();
    // Registered before the request goes out: the first progress event can
    // arrive before `_send` returns, exactly as a token event can.
    if (onProgress != null) _progress[id] = onProgress;
    try {
      final result = await _send(
        id,
        (id) => _Request(
          id: id,
          op: 'openBundle',
          moduleUrl: _moduleUrl,
          bundleId: name,
          quant: quant,
          storeDir: storeDir,
          contextSize: _options.contextSize,
          backend: _options.backend.name,
          turboQuant: _options.turboQuant,
        ),
      );
      _adoptOpenResult(result);
    } finally {
      _progress.remove(id);
    }
  }

  void _adoptOpenResult(JSAny? result) {
    final opened = result as _OpenResult;
    _backend = opened.backend;
    _capabilities = _capabilitiesOf(opened.capabilities);
    final buf = opened.cancelBuffer;
    if (buf != null) {
      try {
        _cancelArray = JSInt32Array(buf as JSArrayBuffer);
      } catch (_) {}
    }
  }

  /// Produces an `ArrayBuffer` that is safe to transfer.
  ///
  /// A `Uint8List` can be a window onto a larger buffer, and transferring that
  /// buffer would hand over the whole thing plus neuter the caller's other
  /// views onto it. Copy in that case; the common case is a whole-buffer list
  /// and costs nothing.
  JSArrayBuffer _detach(Uint8List bytes) {
    return Uint8List.fromList(bytes).buffer.toJS;
  }

  JSArrayBuffer _detachF32(Float32List floats) {
    return Float32List.fromList(floats).buffer.toJS;
  }

  void _receive(JSObject? data) {
    if (data == null) return;
    final reply = data as _Reply;
    if (reply.event == 'token') {
      _streams[reply.id]?.add(reply.text ?? '');
      return;
    }
    if (reply.event == 'thought') {
      _thoughtCallbacks[reply.id]?.call(reply.text ?? '');
      return;
    }
    if (reply.event == 'audio') {
      final callback = _audioCallbacks[reply.id];
      final pcmJs = reply.pcm;
      final sampleRate =
          ((reply.sampleRate as JSNumber?)?.toDartDouble ?? 24000).toInt();
      if (callback != null && pcmJs != null) {
        final List<double> pcm;
        if (pcmJs.isA<JSFloat32Array>()) {
          pcm = (pcmJs as JSFloat32Array).toDart;
        } else {
          pcm =
              (pcmJs as JSArray<JSNumber>).toDart
                  .map((n) => n.toDartDouble)
                  .toList();
        }
        callback(pcm, sampleRate);
      }
      return;
    }
    if (reply.event == 'progress') {
      final sink = _progress[reply.id];
      if (sink != null) {
        final total = reply.total;
        sink(
          CeraDownload(
            url: reply.url ?? '',
            bytesDownloaded:
                ((reply.done as JSNumber?)?.toDartDouble ?? 0).toInt(),
            // Distinguishes "the server sent no length" from "zero so far";
            // the worker forwards null rather than omitting the field.
            totalBytes:
                total == null ? null : (total as JSNumber).toDartDouble.toInt(),
          ),
        );
      }
      return;
    }
    final completer = _pending.remove(reply.id);
    if (completer == null) return;
    if (reply.ok ?? false) {
      completer.complete(reply.result);
      return;
    }
    final message = reply.error ?? 'unknown worker error';
    // A worker error that names itself unsupported becomes the type the API
    // documents. Everything else stays a [CeraWebException], because
    // reconstructing a typed error from a string would be guessing.
    completer.completeError(
      reply.kind == 'unsupported'
          ? UnsupportedError(message)
          : CeraWebException(message),
    );
  }

  /// Fails every outstanding request. Used when the worker itself dies, where
  /// no per-request reply is ever coming.
  void _failAll(Object error) {
    // Streams first. A pending completer's `onError` handler closes its own
    // controller, so failing the completers first left this loop adding an
    // error to an already-closed controller, which throws "Cannot add event
    // after closing" into the zone on every close that races a generate.
    final streams = _streams.values.toList();
    _streams.clear();
    for (final controller in streams) {
      if (controller.isClosed) continue;
      controller.addError(error);
      unawaited(controller.close());
    }
    final pending = _pending.values.toList();
    _pending.clear();
    _audioCallbacks.clear();
    _thoughtCallbacks.clear();
    for (final completer in pending) {
      if (!completer.isCompleted) completer.completeError(error);
    }
  }

  /// Reserves a request id.
  ///
  /// Separate from [_send] because a streaming op has to register its sink
  /// under the id *before* the request goes out: the first token event can
  /// arrive before the send returns.
  int _newId() => _nextId++;

  Future<JSAny?> _send(
    int id,
    _Request Function(int id) build, {
    List<JSAny> transfer = const [],
  }) {
    if (_closed) {
      throw StateError('this Cera engine is closed');
    }
    final worker = _worker;
    if (worker == null) {
      throw StateError('this Cera engine is not open');
    }
    final completer = Completer<JSAny?>();
    _pending[id] = completer;
    worker.postMessage(build(id), transfer.toJS);
    return completer.future;
  }

  @override
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double? temperature,
    double? topP,
    int? topK,
    int? seed,
    void Function(String thought)? onThought,
    void Function(List<double> pcm, int sampleRate)? onAudio,
  }) {
    if (_closed) {
      throw StateError('this Cera engine is closed');
    }
    final controller = StreamController<String>();
    // See the native implementation: `onCancel` fires on normal completion as
    // well as on an explicit cancel, and cancelling a finished generation is
    // not free. It costs a round trip, and on the CPU backend it leaves the
    // worker's session holding a sticky cancel flag that nothing clears, which
    // fails the next turn's prefill.
    var finished = false;

    /// Whether this generation has reached the front of the queue and begun.
    ///
    /// Without it, cancelling a stream that is still WAITING its turn sends the
    /// `cancel` op, and the worker's session is shared, so it would truncate
    /// whoever is decoding right now.
    var started = false;

    // Joined at listen, not at call, so a stream that is returned and never
    // listened to neither waits for its turn nor makes anything wait for it.
    controller.onListen = () async {
      final ahead = _queue;
      final mine = Completer<void>();
      _queue = mine.future;
      try {
        await ahead;
      } catch (_) {}
      // Release the queue BEFORE closing the controller on every early exit;
      // `close()`'s future does not complete while a subscription is paused, so
      // the other order can leave `_queue` uncompleted forever.
      if (_closed || !controller.hasListener) {
        mine.complete();
        if (_closed && controller.hasListener) {
          controller.addError(StateError('this Cera engine is closed'));
        }
        if (!controller.isClosed) await controller.close();
        return;
      }
      started = true;
      final id = _newId();
      _streams[id] = controller;
      if (onThought != null) {
        _thoughtCallbacks[id] = onThought;
      }
      if (onAudio != null) {
        _audioCallbacks[id] = onAudio;
      }
      void terminate(Object? error, StackTrace? stack) {
        finished = true;
        _streams.remove(id);
        _audioCallbacks.remove(id);
        _thoughtCallbacks.remove(id);
        if (!mine.isCompleted) mine.complete();
        if (controller.isClosed) return;
        if (error != null) controller.addError(error, stack);
        controller.close();
      }

      try {
        final request = _send(
          id,
          (id) => _Request(
            id: id,
            op: 'generate',
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            seed: seed,
            wantsAudio: onAudio != null,
            wantsThought: onThought != null,
          ),
        );
        unawaited(
          request.then((_) => terminate(null, null), onError: terminate),
        );
      } on Object catch (e, st) {
        // `_send` throws synchronously if the engine closed between the
        // `generate` call and the first listen. Reported to the stream, not
        // rethrown: `onListen` runs inside the controller's zone, so a throw
        // here would surface as an unhandled error while the caller waited on
        // a stream that never completes.
        terminate(e, st);
      }
    };
    controller.onCancel = () async {
      if (started && !finished && !_closed) await cancel();
    };
    return controller.stream;
  }

  @override
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  }) async {
    // Rendered on the worker side, so the messages cross as JSON rather than as
    // a structure both sides would have to agree on field by field.
    final messagesJson = jsonEncode([
      for (final m in messages) {'role': m.role, 'content': m.content},
    ]);
    final result = await _send(
      _newId(),
      (id) => _Request(
        id: id,
        op: 'applyChatTemplate',
        messagesJson: messagesJson,
        addGenerationPrompt: addGenerationPrompt,
      ),
    );
    return (result as JSString).toDart;
  }

  @override
  Future<List<int>> encode(String text, {bool addSpecial = true}) async {
    final result = await _send(
      _newId(),
      (id) =>
          _Request(id: id, op: 'encode', text: text, addSpecial: addSpecial),
    );
    return (result as JSArray<JSNumber>).toDart
        .map((n) => n.toDartInt)
        .toList();
  }

  @override
  Future<String> decode(List<int> tokens) async {
    final result = await _send(
      _newId(),
      (id) => _Request(
        id: id,
        op: 'decode',
        tokens: tokens.map((t) => t.toJS).toList().toJS,
      ),
    );
    return (result as JSString).toDart;
  }

  @override
  Future<void> appendImage(Uint8List bytes, {int? maxLongSize}) async {
    // Queued behind any running generation, as on native: this appends patch
    // embeddings to the very KV cache a decode is writing to.
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      final buffer = _detach(bytes);
      await _send(
        _newId(),
        (id) => _Request(
          id: id,
          op: 'appendImage',
          bytes: buffer,
          maxLongSize: maxLongSize,
        ),
        transfer: <JSAny>[buffer],
      );
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
      final floatList = pcm is Float32List ? pcm : Float32List.fromList(pcm);
      final buffer = _detachF32(floatList);
      await _send(
        _newId(),
        (id) => _Request(
          id: id,
          op: 'appendAudio',
          pcm: Float32List.view(buffer.toDart).toJS,
          sampleRate: sampleRate,
          prompt: prompt,
        ),
        transfer: <JSAny>[buffer],
      );
    } finally {
      mine.complete();
    }
  }

  @override
  Future<String> transcribe(List<double> pcm, {required int sampleRate}) async {
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      final floatList = pcm is Float32List ? pcm : Float32List.fromList(pcm);
      final buffer = _detachF32(floatList);
      final result = await _send(
        _newId(),
        (id) => _Request(
          id: id,
          op: 'transcribe',
          pcm: Float32List.view(buffer.toDart).toJS,
          sampleRate: sampleRate,
        ),
        transfer: <JSAny>[buffer],
      );
      return (result as JSString).toDart;
    } finally {
      mine.complete();
    }
  }

  @override
  Future<void> appendUserMessage(CeraUserMessage message) async {
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      final transferred = <JSAny>[];
      final jsImages = <_ImagePayload>[];
      for (final img in message.images) {
        final buf = _detach(img);
        transferred.add(buf);
        jsImages.add(_ImagePayload(bytes: buf));
      }
      _AudioPayload? jsAudio;
      if (message.audio != null) {
        final floatList = message.audio!.pcm is Float32List
            ? message.audio!.pcm as Float32List
            : Float32List.fromList(message.audio!.pcm);
        final buf = _detachF32(floatList);
        transferred.add(buf);
        jsAudio = _AudioPayload(
          pcm: Float32List.view(buf.toDart).toJS,
          sampleRate: message.audio!.sampleRate,
        );
      }
      await _send(
        _newId(),
        (id) => _Request(
          id: id,
          op: 'appendUserMessage',
          text: message.text,
          images: jsImages.isEmpty ? null : jsImages.toJS,
          audio: jsAudio,
        ),
        transfer: transferred,
      );
    } finally {
      mine.complete();
    }
  }

  @override
  Stream<String> sendMessage(
    CeraUserMessage message, {
    int maxTokens = 256,
    double? temperature,
    double? topP,
    int? topK,
    int? seed,
    void Function(String thought)? onThought,
    void Function(List<double> pcm, int sampleRate)? onAudio,
  }) {
    final controller = StreamController<String>();
    var finished = false;
    var started = false;

    controller.onListen = () async {
      final ahead = _queue;
      final mine = Completer<void>();
      _queue = mine.future;
      try {
        await ahead;
      } catch (_) {}
      if (_closed || !controller.hasListener) {
        mine.complete();
        if (_closed && controller.hasListener) {
          controller.addError(StateError('this Cera engine is closed'));
        }
        if (!controller.isClosed) await controller.close();
        return;
      }
      started = true;
      final id = _newId();
      _streams[id] = controller;
      if (onThought != null) {
        _thoughtCallbacks[id] = onThought;
      }
      if (onAudio != null) {
        _audioCallbacks[id] = onAudio;
      }
      void terminate(Object? error, StackTrace? stack) {
        finished = true;
        _streams.remove(id);
        _audioCallbacks.remove(id);
        _thoughtCallbacks.remove(id);
        if (!mine.isCompleted) mine.complete();
        if (controller.isClosed) return;
        if (error != null) controller.addError(error, stack);
        controller.close();
      }

      try {
        final transferred = <JSAny>[];
        final jsImages = <_ImagePayload>[];
        for (final img in message.images) {
          final buf = _detach(img);
          transferred.add(buf);
          jsImages.add(_ImagePayload(bytes: buf));
        }
        _AudioPayload? jsAudio;
        if (message.audio != null) {
          final floatList = message.audio!.pcm is Float32List
              ? message.audio!.pcm as Float32List
              : Float32List.fromList(message.audio!.pcm);
          final buf = _detachF32(floatList);
          transferred.add(buf);
          jsAudio = _AudioPayload(
            pcm: Float32List.view(buf.toDart).toJS,
            sampleRate: message.audio!.sampleRate,
          );
        }

        final request = _send(
          id,
          (id) => _Request(
            id: id,
            op: 'sendMessage',
            text: message.text,
            images: jsImages.isEmpty ? null : jsImages.toJS,
            audio: jsAudio,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            seed: seed,
            wantsAudio: onAudio != null,
            wantsThought: onThought != null,
          ),
          transfer: transferred,
        );
        unawaited(
          request.then((_) => terminate(null, null), onError: terminate),
        );
      } on Object catch (e, st) {
        terminate(e, st);
      }
    };
    controller.onCancel = () async {
      if (started && !finished && !_closed) await cancel();
    };
    return controller.stream;
  }

  @override
  Future<void> reset() async {
    // Wait for the queue first, exactly as the native twin does. A queued
    // generation has posted nothing yet, so resetting immediately would order
    // `generate(A); generate(B); reset()` as A, reset, B here and A, B, reset
    // natively, and B would then run against a conversation the caller cleared
    // before starting it.
    await cancel();
    final ahead = _queue;
    final mine = Completer<void>();
    _queue = mine.future;
    try {
      try {
        await ahead;
      } catch (_) {}
      // `_send` throws synchronously on a closed engine; `async` here turns that
      // into a failed future, which is what the native twin returns. The two
      // transports would otherwise differ on the one call an app is most likely
      // to leave unawaited.
      await _send(_newId(), (id) => _Request(id: id, op: 'reset'));
    } finally {
      mine.complete();
    }
  }

  @override
  Future<void> cancel() async {
    // Also a no-op on a closed engine rather than an error, matching native:
    // there is nothing left to stop, and cancel is the call most often fired
    // from a dispose path that has already closed.
    if (_closed) return;
    if (_cancelArray != null) {
      try {
        _atomics.store(_cancelArray!, 0, 1);
      } catch (_) {}
    }
    await _send(_newId(), (id) => _Request(id: id, op: 'cancel'));
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    // Send BEFORE flipping the flag. `_send` refuses to post once `_closed` is
    // set, so setting it first made this request throw into the catch below,
    // and the worker's own `close` (which frees the model) never ran. The
    // `terminate()` on the next line hid it: the worker died either way, just
    // without releasing its wasm heap first.
    try {
      await _send(_newId(), (id) => _Request(id: id, op: 'close'));
    } on Object {
      // A worker that already died cannot acknowledge, and terminating is the
      // point of this call, so a failure here is not worth propagating.
    }
    _shutDown(StateError('this Cera engine is closed'));
  }

  @override
  Future<void> terminate() async {
    // Straight to the terminate, with no `close` op posted first. That request
    // is what frees the model politely, and it is also what cannot be served
    // while a decode is running: the CPU backend will not dequeue it until the
    // run is over, and the GPU backend would dequeue it mid-call and free a
    // session the running `generateTokens` still holds. Terminating needs
    // nothing from the worker, and takes its heap with it.
    //
    // Async only to match the interface: there is nothing here to await.
    _shutDown(StateError('this Cera engine was terminated'));
  }

  /// Terminates the worker and fails everything still waiting on it.
  ///
  /// Idempotent, and the single path to a dead worker: both an explicit
  /// [close] and a worker that failed to load end here, so neither can leave a
  /// terminated worker that still looks open and accepts requests nothing will
  /// ever answer.
  void _shutDown(Object reason) {
    if (_closed) return;
    _closed = true;
    _worker?.terminate();
    _failAll(reason);
  }
}

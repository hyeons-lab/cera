/// A portable, asynchronous inference API that works on every target the
/// package supports, including the web.
///
/// The generated bindings in `src/generated/cera_ffi.dart` are the full
/// surface, and on native targets they are the right thing to use directly.
/// They cannot be the web story though, and not for want of a port: a browser
/// runs the engine in a Web Worker, `postMessage` is asynchronous, and a worker
/// offers no synchronous escape hatch (`Atomics.wait` is forbidden on the main
/// thread). A synchronous `engine.generate(...)` is therefore not
/// implementable there at any cost.
///
/// So the split is not "native API plus a web shim". It is one asynchronous API
/// with two transports: a Rust async runtime on native, a Web Worker on web.
/// Code written against [Cera] compiles and runs unchanged on both.
library;

import 'dart:typed_data';

import 'cera_web.dart' if (dart.library.ffi) 'cera_io.dart' as impl;

/// Which compute backend to run on.
///
/// A preference, not a guarantee, and what each value can promise differs by
/// platform because the backends do: natively the choice is between the CPU
/// and whichever GPU backend exists (Metal on Apple, `wgpu` elsewhere), and on
/// the web it is between WebGPU and the wasm CPU build. See each value.
enum CeraBackend {
  /// Pick the fastest backend that can serve the model, falling back rather
  /// than failing.
  ///
  /// On the web that ordering is not a close call: WebGPU measured ~58 tok/s
  /// against ~1.4 tok/s for the wasm CPU build on the same machine and model.
  /// Prefer leaving this alone.
  auto,

  /// Force the CPU backend.
  cpu,

  /// Prefer the GPU, and on the web fail the open rather than fall back.
  ///
  /// The two platforms differ, because only one of them can honor the strict
  /// reading. On the **web** this does what it says: no WebGPU, no adapter, or
  /// a non-LFM2 model (the GPU path is LFM2-only) is an error instead of the
  /// silent CPU fallback [auto] would give. Choose it when a 40x quiet
  /// slowdown would be worse than a failure.
  ///
  /// **Natively this behaves as [auto].** "The GPU" is two backends there,
  /// Metal on Apple platforms and wgpu elsewhere, and the engine's own `auto`
  /// is the only setting that probes for whichever exists; pinning wgpu would
  /// skip the faster Metal path on macOS, which is not what asking for the GPU
  /// means. A native GPU request therefore still falls back to the CPU rather
  /// than failing.
  gpu,
}

/// Where the web implementation loads its JavaScript and wasm from.
///
/// Ignored on native targets. The defaults match what
/// `dart run cera_ffi:install_web` writes into an app's `web/` directory, so
/// an app that ran it needs none of this.
class CeraWebAssets {
  /// Creates an asset location. Both URLs may be relative to the page.
  const CeraWebAssets({
    this.workerUrl = 'cera/cera_worker.js',
    this.moduleUrl = 'cera/cera_wasm.js',
  });

  /// URL of `cera_worker.js`, loaded as an ES module worker.
  final String workerUrl;

  /// URL of wasm-bindgen's `cera_wasm.js` loader.
  ///
  /// Its `cera_wasm_bg.wasm` sibling is fetched by that loader relative to this
  /// URL, so the two files have to stay in the same directory.
  final String moduleUrl;
}

/// Knobs fixed for the lifetime of an engine.
class CeraOptions {
  /// Creates an option set. Every field has a usable default.
  const CeraOptions({
    this.contextSize = 4096,
    this.backend = CeraBackend.auto,
    this.turboQuant = false,
    this.web = const CeraWebAssets(),
  });

  /// KV-cache window, in tokens. Costs memory proportional to its size.
  final int contextSize;

  /// Which compute backend to run on.
  final CeraBackend backend;

  /// Whether to enable TurboQuant KV-cache compression. Defaults to false (cera default).
  final bool turboQuant;

  /// Web asset locations. Ignored on native targets.
  final CeraWebAssets web;
}

/// What a loaded model accepts as input and emits as output.
///
/// Derived at load time from the bundle's declared inference type, not from
/// what a caller asked for: it describes what the engine *can* do. A VL bundle
/// whose mmproj failed to parse reports `imageIn: false` and the engine still
/// serves text, so this is the honest thing to gate a UI on.
class CeraCapabilities {
  /// Creates a capability set.
  const CeraCapabilities({
    required this.textIn,
    required this.textOut,
    required this.imageIn,
    required this.audioIn,
    required this.audioOut,
  });

  /// Whether the model accepts text prompts. True for every model that
  /// currently loads.
  final bool textIn;

  /// Whether the model emits text.
  final bool textOut;

  /// Whether [Cera.appendImage] will work.
  final bool imageIn;

  /// Whether [Cera.transcribe] will work.
  final bool audioIn;

  /// Whether the model emits audio. No path here surfaces audio output yet;
  /// reported so a caller can tell a speech model from a transcription one.
  final bool audioOut;

  @override
  String toString() =>
      'CeraCapabilities(textIn: $textIn, textOut: $textOut, '
      'imageIn: $imageIn, audioIn: $audioIn, audioOut: $audioOut)';
}

/// One model published on `LiquidAI/LeapBundles`, with the quantizations it
/// offers.
///
/// The pair a picker needs: [name] and one of [quants] are exactly the two
/// arguments [Cera.openBundle] takes.
class CeraBundle {
  /// Creates a catalog entry.
  const CeraBundle({required this.name, required this.quants});

  /// The bundle id, e.g. `LFM2-1.2B-GGUF`.
  final String name;

  /// Quantization labels this bundle publishes, e.g. `Q4_0`, `Q8_0`.
  ///
  /// Sorted ascending, as is [name] across the returned list, so a menu built
  /// from this is stable across runs even when the upstream API reorders its
  /// response.
  final List<String> quants;

  /// [name] without its `-GGUF` suffix, for showing in a menu.
  ///
  /// Every entry in the live catalog carries that suffix and it is noise in a
  /// list. `cera list-bundles` and the CLI picker trim it the same way, so a
  /// menu built on this names a model exactly as the CLI does. Pass [name],
  /// not this, to [Cera.openBundle].
  String get displayName =>
      name.endsWith('-GGUF')
          ? name.substring(0, name.length - '-GGUF'.length)
          : name;

  @override
  String toString() => 'CeraBundle($name, quants: ${quants.join(", ")})';
}

/// Progress of one file within a bundle download.
///
/// A bundle is several files (the manifest, the model, and for a multimodal
/// bundle its projector), so this arrives for each in turn and [url] is what
/// distinguishes them. [bytesDownloaded] is monotonic within one file and
/// starts over at the next.
///
/// Reported roughly every 256 KB plus once at end of stream, on both
/// transports, so the last event for a file always carries its final count.
/// A fully cached bundle opens without emitting any of these.
class CeraDownload {
  /// Creates a progress event.
  const CeraDownload({
    required this.url,
    required this.bytesDownloaded,
    required this.totalBytes,
  });

  /// The file being fetched.
  final String url;

  /// Bytes of this file received so far.
  final int bytesDownloaded;

  /// This file's total size, or null when the server sent no length.
  ///
  /// Null is not rare enough to ignore: a chunked-transfer response has no
  /// `Content-Length`, so a progress bar needs an indeterminate state rather
  /// than a division.
  final int? totalBytes;

  /// Completion of this file in `0.0..1.0`, or null when [totalBytes] is
  /// unknown or zero.
  double? get fraction {
    final total = totalBytes;
    if (total == null || total <= 0) return null;
    return (bytesDownloaded / total).clamp(0.0, 1.0);
  }

  @override
  String toString() =>
      'CeraDownload($url, $bytesDownloaded/${totalBytes ?? "?"})';
}

/// A loaded model, ready to generate.
///
/// Obtain one from [openPath], [openBytes] or [openBundle], and [close] it when
/// done: it owns the model weights, which are the largest allocation in most
/// apps, and nothing collects them for you on the web.
abstract interface class Cera {
  /// Whether [openPath] works on this platform. False only on the web.
  ///
  /// Worth branching on rather than catching the [UnsupportedError], because
  /// the interesting decision happens earlier: a file picker has to be asked
  /// for the file's *bytes* up front on the web, and asking for them on native
  /// reads a multi-gigabyte model into the heap that the engine would otherwise
  /// have memory-mapped.
  static bool get supportsPaths => impl.supportsPaths;

  /// Loads a GGUF from the filesystem.
  ///
  /// Not available on the web, which has no filesystem to point at; use
  /// [openBytes] there, or unconditionally if the same code has to serve both.
  /// The load runs off the calling thread on every platform.
  static Future<Cera> openPath(
    String path, {
    CeraOptions options = const CeraOptions(),
  }) => impl.openPath(path, options);

  /// Loads a GGUF already in memory.
  ///
  /// This is the constructor that works everywhere, and on the web the only
  /// one. Expect peak memory of roughly twice `bytes` during the call: the
  /// engine builds its own representation before the caller's copy goes away.
  ///
  /// **On the web, treat `bytes` as consumed.** Where it can, the
  /// implementation transfers the underlying buffer to the worker instead of
  /// copying it, which is what keeps a multi-hundred-megabyte model from being
  /// cloned, and a transferred buffer is detached: the caller's list reads as
  /// empty afterwards. Whether that happens depends on the list's shape and the
  /// web compiler, so do not rely on either outcome. Opening a second engine,
  /// or retrying after a failure, needs a freshly fetched or copied list.
  ///
  /// **Multimodal models need `mmproj`.** A VL or audio model is two GGUFs:
  /// the vision tower and the audio encoder live in a separate "mmproj" file,
  /// and passing only the model loads it as text-only. [openPath] reads the
  /// second file from the bundle's manifest; from memory there is no manifest,
  /// so it has to be handed over.
  ///
  /// Modality is inferred from the arguments, because it cannot be read from
  /// the header: every published LFM2-VL model reports the same architecture
  /// string a text model does, since the vision half is entirely in the mmproj.
  /// So supplying one is taken as the statement of intent it is. Pass
  /// `inferenceType` to override (`llama.cpp/text-to-text`,
  /// `llama.cpp/image-to-text`, `llama.cpp/lfm2-audio-v1`).
  ///
  /// A malformed mmproj is not fatal: the model still serves text, and
  /// [capabilities] reports `imageIn: false` rather than promising something
  /// [appendImage] would then refuse.
  static Future<Cera> openBytes(
    Uint8List bytes, {
    Uint8List? mmproj,
    String? inferenceType,
    CeraOptions options = const CeraOptions(),
  }) => impl.openBytes(bytes, options, mmproj, inferenceType);

  /// The bundles published on `LiquidAI/LeapBundles`, for a picker to offer.
  ///
  /// Each entry pairs a name with its quantizations, which is exactly what
  /// [openBundle] takes. One small HTTP request, deliberately uncached, so a
  /// picker opened twice in a session reflects newly published bundles.
  ///
  /// Every platform reads the same catalog through the same parser, so a menu
  /// built on this shows the same list as `cera list-bundles`.
  ///
  /// Throws if the catalog cannot be reached. There is a 30 second deadline
  /// and no retry: a picker that reports it could not reach the network beats
  /// one that never opens.
  static Future<List<CeraBundle>> listBundles({
    CeraOptions options = const CeraOptions(),
  }) => impl.listBundles(options);

  /// Downloads a published bundle by `name` and `quant` and loads it, reusing
  /// anything already cached.
  ///
  /// The pair comes from [listBundles]. Prefer this over fetching a GGUF
  /// yourself and calling [openBytes]: a bundle's manifest names every file it
  /// needs and states its modality outright, so a vision or audio bundle
  /// arrives complete, whereas [openBytes] has only its arguments to go on.
  ///
  /// It is also the cheaper path in memory on every platform, and dramatically
  /// so on the web: the weights go from the cache into the engine without
  /// passing through the caller, which costs one copy of the model rather than
  /// two and, in a browser, sidesteps the roughly 2 GiB limit on a single
  /// contiguous JavaScript allocation that [openBytes] runs into.
  ///
  /// `onProgress` fires per file while downloading and not at all on a fully
  /// cached bundle, so do not wait for a first event before showing the UI.
  ///
  /// **The download cannot be cancelled.** There is no way to abandon it once
  /// started, and dropping the returned future does not stop it, so `onProgress`
  /// keeps firing until the bundle is fetched. A caller that can be disposed
  /// mid-download (a page or a route) has to guard its own callback rather than
  /// expect the download to stop with it. Worth knowing before offering this on
  /// a multi-gigabyte model, where the wait is minutes.
  ///
  /// `storeDir` is where downloads are cached, and it means different things
  /// per platform. Natively it is a filesystem path, defaulting on desktop to
  /// `$HOME/.cache/cera`, which is also where the `cera` CLI caches, so the two
  /// share downloads rather than each pulling its own copy. **On the web it is
  /// a single directory NAME inside the origin's private filesystem, not a
  /// path**: one containing `/` or `..` is rejected. So the same value cannot
  /// simply be passed everywhere; leave it null on the web, where the default
  /// is right anyway.
  ///
  /// **Required on Android and iOS**, where it throws [ArgumentError] if
  /// omitted: an app there may write only inside its own container, which no
  /// environment variable names, so there is nothing sound to default to. Pass
  /// a path from `path_provider`, e.g.
  /// `(await getApplicationSupportDirectory()).path`. Failing at the call is
  /// deliberate, rather than defaulting to something that would surface as a
  /// permission error partway into a multi-gigabyte download.
  ///
  /// The backend follows [CeraOptions.backend] as it does for [openBytes],
  /// including the web's GPU-then-CPU fallback. Note that the browser GPU path
  /// serves LFM2 bundles only, so a non-LFM2 choice there means the wasm CPU
  /// backend and a large slowdown; [backend] reports which one took effect.
  static Future<Cera> openBundle(
    String name,
    String quant, {
    void Function(CeraDownload progress)? onProgress,
    String? storeDir,
    CeraOptions options = const CeraOptions(),
  }) => impl.openBundle(name, quant, options, onProgress, storeDir);

  /// Describes the backend in use, and on the web the GPU adapter with it.
  ///
  /// Worth surfacing in a debug view, because [CeraBackend.auto] falls back
  /// silently and a 40x throughput difference on the web is not something to
  /// discover from a user report.
  ///
  /// Precise only on the web, where the fallback decision happens in this
  /// package and is therefore observable. Natively it reports the preference
  /// the engine was *given* (and since [CeraBackend.gpu] is passed through as
  /// `auto`, both report `native (auto)`): the engine resolves Metal / wgpu /
  /// CPU internally at load time and the FFI surface exposes no accessor for
  /// what it picked.
  String get backend;

  /// What this model accepts and emits. Fixed for the engine's lifetime.
  ///
  /// Gate on this rather than catching from [appendImage] / [transcribe]: those
  /// throw, and by then a user has already picked a file.
  CeraCapabilities get capabilities;

  /// Generates a continuation of `prompt`, streaming decoded text as it is
  /// produced.
  ///
  /// The stream emits fragments, not tokens or whole words: a token can be a
  /// partial word, and a multi-byte character can span several tokens, so the
  /// only safe thing to do with a fragment is append it. Concatenating the
  /// whole stream gives the complete output.
  ///
  /// The conversation continues across calls, sharing one KV cache. Call
  /// [reset] to start over.
  ///
  /// Cancelling the subscription requests an early stop, subject to the
  /// limitations on [cancel].
  ///
  /// **Generations are serialized.** Calling this while a previous one is
  /// still running returns a stream that waits its turn rather than running
  /// concurrently. They share one KV cache, so sequential is the only
  /// meaningful order; the wait is invisible after a completed generation and
  /// noticeable only after a cancelled one, whose decode may still be
  /// finishing (see [cancel]).
  ///
  /// The sampling parameters are honored on every backend, including the
  /// web's GPU one. Greedy decoding is `temperature: 0` or `topK: 1`, the
  /// same rule everywhere.
  ///
  /// Sampling does cost more on the web's GPU backend than greedy decoding
  /// does: greedy takes the argmax on the GPU and reads back a token id, while
  /// sampling has to read the whole logits row back for the sampler to see it,
  /// once per token. Worth knowing if a run is slower than the greedy numbers
  /// suggested; it is not a reason to avoid it.
  ///
  /// `seed` behaves differently per backend, because the sampler's lifetime
  /// does. Natively and on the web's CPU backend it is a session-level knob
  /// applied when the session is created, so it takes effect only on the first
  /// generation of a session. The web's GPU backend builds its sampler per
  /// call, so a seed applies to whichever call passes it.
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double? temperature,
    double? topP,
    int? topK,
    int? seed,
  });

  /// Renders `messages` through the model's chat template.
  ///
  /// Feed the result straight to [generate]: the rendered text contains the
  /// template's special markers, and every path here lowers those to their
  /// token ids rather than to the characters spelling them out.
  ///
  /// `addGenerationPrompt` appends the assistant-turn header, which is what
  /// makes the model reply rather than continue the transcript.
  ///
  /// Throws if the GGUF declares no chat template.
  Future<String> applyChatTemplate(
    List<CeraMessage> messages, {
    bool addGenerationPrompt = true,
  });

  /// Tokenizes `text`.
  ///
  /// `addSpecial` prepends BOS (and appends EOS) when the GGUF asks for them,
  /// matching `llama_tokenize(..., add_special)`. It is the right default for
  /// a whole prompt and the wrong one for a fragment.
  Future<List<int>> encode(String text, {bool addSpecial = true});

  /// Detokenizes `tokens` back to text.
  Future<String> decode(List<int> tokens);

  /// Feeds an image into the conversation, to be described or asked about by
  /// the next [generate].
  ///
  /// `bytes` is an encoded PNG or JPEG, not raw pixels; it is decoded, resized
  /// and run through the vision tower, and the resulting patch embeddings are
  /// appended to the same KV cache [generate] writes to. So the order matters:
  /// append the image, then generate with the question.
  ///
  /// `maxLongSize` caps the longer edge in pixels before encoding, trading
  /// detail for tokens and time. Omit it for the model's own default; pass 0 to
  /// disable the cap entirely.
  ///
  /// Throws if this model has no vision encoder, i.e. whenever
  /// [capabilities] reports `imageIn: false`. Serialized against [generate] the
  /// same way generations are serialized against each other, since both append
  /// to one cache.
  Future<void> appendImage(Uint8List bytes, {int? maxLongSize});

  /// Feeds mono PCM audio into the conversation, to be processed by the next
  /// [generate].
  ///
  /// Feed mono PCM audio into the live conversation.
  ///
  /// Feeds raw audio frames directly through the model's audio encoder into the
  /// LLM's KV cache. An optional `prompt` text can accompany the audio.
  ///
  /// `pcm` is normalized to roughly [-1.0, 1.0]. Non-16 kHz inputs are
  /// automatically resampled by the engine to the model's required rate.
  ///
  /// Throws if this model has no audio encoder, i.e. whenever [capabilities]
  /// reports `audioIn: false`. Serialized against [generate] the same way
  /// generations are serialized against each other.
  Future<void> appendAudio(
    List<double> pcm, {
    int sampleRate = 16000,
    String? prompt,
  });

  /// `pcm` is normalized to roughly [-1.0, 1.0]. Non-16kHz inputs are
  /// automatically resampled to 16 kHz by the engine.
  ///
  /// Independent of the conversation: this runs the model's own "Perform ASR."
  /// mode start to finish and returns the whole transcript, rather than
  /// streaming or appending to the session. Throws when [capabilities] reports
  /// `audioIn: false`.
  Future<String> transcribe(List<double> pcm, {required int sampleRate});

  /// Clears the conversation, keeping the model loaded.
  ///
  /// Throws [UnsupportedError] on the web's GPU backend, whose KV cache lives
  /// on the GPU with no way to clear it; close and reopen the engine there.
  ///
  /// Cancels any generation still running, waits for it and for anything
  /// queued behind it, and only then clears: dropping the conversation out from
  /// under a running decode would leave it appending to a cache it no longer
  /// matches. The next [generate] is a start-of-sequence again, BOS included.
  Future<void> reset();

  /// Requests that an in-flight [generate] stop early.
  ///
  /// **Reliable only on native.** On the web it is best-effort and today
  /// reaches neither backend's running decode, for two unrelated reasons:
  ///
  /// - CPU: the decode loop is one synchronous wasm call occupying the worker,
  ///   so the message is not delivered until it has already finished.
  /// - GPU: `WebGpuSession` exposes no cancel entry point to call.
  ///
  /// Cancelling the [generate] stream's subscription is the more idiomatic
  /// form and the better one for a Stop button: delivery to your code stops at
  /// once, whether or not the decode behind it does. Note that *awaiting* that
  /// cancellation is a different matter on the web's CPU backend, where the
  /// returned future waits on a worker reply that cannot be dequeued until the
  /// synchronous decode finishes. Drop the subscription rather than awaiting it
  /// if a Stop button must return promptly.
  ///
  /// [terminate] does stop the work itself on every platform, but by destroying
  /// the engine, so it answers "I am done with this" rather than "stop this
  /// one generation".
  Future<void> cancel();

  /// Releases the model and everything derived from it. Safe to call twice.
  ///
  /// On native targets this does not make a plain Dart script exit. [generate]
  /// registers a callback interface, whose vtable holds static
  /// `NativeCallable`s for the process's lifetime, and a live `NativeCallable`
  /// keeps its isolate alive. A CLI has to `exit()`; a Flutter app is running
  /// anyway and never notices.
  Future<void> close();

  /// Tears the engine down now, abandoning whatever it was doing.
  ///
  /// [close] is the orderly form and the one to reach for: it releases the
  /// model and, natively, stops the decode at its next between-token check. Use
  /// this instead when the result no longer has anywhere to go, which in a UI
  /// means a page that has been disposed.
  ///
  /// The difference is the web, where [close] first asks the worker to free the
  /// model and waits for it to answer. A running decode makes that request
  /// either slow or unsafe, depending on the backend: the CPU decode is one
  /// synchronous wasm call occupying the worker, so the message is not
  /// delivered until the run has finished anyway, while a GPU decode yields
  /// between tokens and so *can* dequeue it, whereupon freeing the session
  /// re-enters an object the running call still holds. This skips that request
  /// and terminates the worker, which the browser does from the outside: it
  /// needs no cooperation from the code running inside, so it stops a decode of
  /// either kind, and the worker's whole heap goes with it rather than being
  /// released call by call.
  ///
  /// Natively there is nothing to terminate that [close] does not already do,
  /// so this is [close].
  ///
  /// In-flight work does not simply vanish: a [generate] stream still running
  /// ends with an error rather than silently, because a caller awaiting one
  /// deserves to be told the engine went away. Safe to call twice, and safe to
  /// call after [close].
  Future<void> terminate();
}

/// An error raised inside the web worker.
///
/// Declared here rather than in the web implementation so that callers can name
/// it: the implementations are selected by conditional import and are not
/// exported, so a type declared in one of them is unreachable from an app even
/// though the docs promise it.
///
/// Its own type rather than the generated `FfiException` hierarchy, because the
/// failure crossed a `postMessage` boundary, where a wasm-bindgen error is not
/// structured-cloneable and arrives as a string. Reconstructing a typed error
/// from that string would be guessing. Never thrown on native targets.
class CeraWebException implements Exception {
  /// Creates an exception carrying the worker's message.
  const CeraWebException(this.message);

  /// The error text as the worker reported it.
  final String message;

  @override
  String toString() => 'CeraWebException: $message';
}

/// One turn in a chat transcript, for [Cera.applyChatTemplate].
///
/// Deliberately not the generated `ChatMessage`: that type is part of the
/// binding surface, and this API has to carry the same meaning on a target
/// where those bindings are stubs.
class CeraMessage {
  /// Creates a message.
  const CeraMessage(this.role, this.content);

  /// A system-prompt turn.
  const CeraMessage.system(this.content) : role = 'system';

  /// A user turn.
  const CeraMessage.user(this.content) : role = 'user';

  /// An assistant turn.
  const CeraMessage.assistant(this.content) : role = 'assistant';

  /// Conventionally `system`, `user`, or `assistant`.
  ///
  /// Not validated here: the string flows into the model's own Jinja template,
  /// and what an unrecognized role does is that template's business.
  final String role;

  /// The turn's text.
  final String content;
}

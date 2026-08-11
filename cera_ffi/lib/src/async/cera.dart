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
    this.web = const CeraWebAssets(),
  });

  /// KV-cache window, in tokens. Costs memory proportional to its size.
  final int contextSize;

  /// Which compute backend to run on.
  final CeraBackend backend;

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

/// A loaded model, ready to generate.
///
/// Obtain one from [openPath] or [openBytes], and [close] it when done: it owns
/// the model weights, which are the largest allocation in most apps, and
/// nothing collects them for you on the web.
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

  /// Transcribes mono PCM audio to text.
  ///
  /// `pcm` is normalized to roughly [-1.0, 1.0] and `sampleRate` must already
  /// match what the model's audio encoder expects; nothing here resamples.
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
  Future<void> cancel();

  /// Releases the model and everything derived from it. Safe to call twice.
  ///
  /// On native targets this does not make a plain Dart script exit. [generate]
  /// registers a callback interface, whose vtable holds static
  /// `NativeCallable`s for the process's lifetime, and a live `NativeCallable`
  /// keeps its isolate alive. A CLI has to `exit()`; a Flutter app is running
  /// anyway and never notices.
  Future<void> close();
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

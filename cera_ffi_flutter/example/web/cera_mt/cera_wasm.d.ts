/* tslint:disable */
/* eslint-disable */

/**
 * Modality capability flags for a loaded model. Returned by
 * `CeraEngine.capabilities` and `Session.capabilities`. Mirrors
 * the `ModalityCapabilities` shape exposed by the JVM/Apple
 * bindings (cera-ffi) so cross-platform consumers can probe the
 * same fields regardless of which binding they're driving.
 *
 * These describe the bundle that was actually loaded. A model
 * opened with `fromGgufBytes` is text-only by construction, since
 * one GGUF cannot carry a vision tower or an audio encoder: load
 * the multimodal projector alongside it via `fromGgufParts` for
 * `imageIn` / `audioIn` to be true.
 */
export interface Capabilities {
    readonly textIn: boolean;
    readonly textOut: boolean;
    readonly imageIn: boolean;
    readonly audioIn: boolean;
    readonly audioOut: boolean;
}



/**
 * One entry in the chat-message array passed to
 * `Tokenizer.applyChatTemplate` / `applyChatTemplateWithTools`.
 * Mirrors the OpenAI / Anthropic SDK shape. For tool results, use
 * `role: "tool"` with the JSON result string as `content`.
 */
export interface ChatMessage {
    role: string;
    content: string;
}

/**
 * A tool the model may call, in the OpenAI "function" shape. Pass an
 * array of these (JSON-stringified) to `applyChatTemplateWithTools`
 * and `toolGrammar`. `parameters` is a JSON Schema object for the
 * arguments.
 */
export interface ToolDef {
    name: string;
    description?: string;
    parameters?: object;
}

/**
 * A tool call parsed from model output by `parseToolCalls`. `arguments` is
 * normally an object, but a malformed Hermes/Qwen reply may pass through a
 * non-object JSON value — narrow before assuming an object map.
 */
export interface ToolCall {
    name: string;
    // Usually an object map (see the doc comment above); typed `unknown`
    // because a malformed reply can pass through a non-object JSON value.
    arguments: unknown;
}



/**
 * Summary of a loaded model, returned by `CeraEngine.metadata`.
 * Mirrors the `ModelMetadata` record the JVM/Apple bindings
 * (cera-ffi) expose, so cross-platform consumers read the same
 * field names from either binding.
 *
 * Every field is also available as an individual getter on
 * `CeraEngine`. This is the one-call form: each getter is a
 * separate wasm boundary crossing, so prefer this when you want
 * several of them at once (rendering a model-info panel, say).
 */
export interface ModelMetadata {
    readonly architecture: string;
    readonly maxSeqLen: number;
    readonly vocabSize: number;
    readonly hasChatTemplate: boolean;
    readonly quantization: string;
    readonly addBosToken: boolean;
    readonly addEosToken: boolean;
}



/**
 * Remote bundle store over the Origin Private File System.
 *
 * Construct once and reuse: it holds only the store directory name, so
 * copies are cheap and concurrent downloads through the same instance
 * are fine (every method takes `&self`).
 *
 * ```js
 * const repo = new BundleRepo();                 // "cera-models"
 * const engine = await CeraEngine.fromBundleId(
 *     repo, "LFM2-1.2B-GGUF", "Q4_0", 4096,
 *     (url, done, total) => console.log(url, done / total),
 * );
 * ```
 */
export class BundleRepo {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Cached bytes for `url`, downloading first if needed.
     *
     * **Copies the whole file into a JS `Uint8Array`**, so peak memory
     * is roughly twice the file. That's fine for a manifest and wrong
     * for a model: to load a model, use `CeraEngine.fromBundleId` or
     * `fromManifestUrl`, which keep the bytes inside wasm and hand
     * them straight to the engine.
     */
    bytes(url: string, expected_sha256?: string | null, on_progress?: Function | null): Promise<Uint8Array>;
    /**
     * Total bytes currently cached, summed by walking the tree.
     *
     * Returns 0 when nothing has been downloaded yet (the directory
     * doesn't exist). Like the native `cache_size`, this is a real
     * O(n) walk rather than a constant-time query, and it counts the
     * `.sha256` sidecars along with the payloads.
     *
     * This is deliberately not `navigator.storage.estimate()`: that
     * reports the whole origin's usage, so a page that also stores
     * user data would see it attributed to the model cache.
     */
    cacheSize(): Promise<number>;
    /**
     * Delete everything this repo has cached. Idempotent: clearing an
     * empty or never-created store succeeds.
     *
     * Unlike the native store this doesn't recreate the directory
     * afterwards, because there's nothing to preserve: OPFS
     * directories are created on demand by the next download.
     *
     * An in-flight download through the same repo will fail when its
     * file vanishes. As on native, serializing a user-driven "clear
     * downloads" action against active loads is the caller's job.
     */
    clearCache(): Promise<void>;
    /**
     * Download `url` into the cache if it isn't already there.
     *
     * `expectedSha256` pins the content hash; when omitted, integrity
     * falls back to `x-linked-etag` (if CORS exposes it) and then to a
     * `Content-Length` size check. See the module docs for why a
     * browser cannot do better.
     *
     * `onProgress(url, bytesDownloaded, totalBytes)` fires at most
     * once per 256 KB and once at end of stream; `totalBytes` is
     * `null` when the server sends no length. It is not called at all
     * on a cache hit, since there is no streaming work to report.
     */
    download(url: string, expected_sha256?: string | null, on_progress?: Function | null): Promise<void>;
    /**
     * Whether `url` is present in the cache. Existence only: this does
     * not verify the hash or size, so a truthy answer means "a
     * download landed here", not "the bytes are known good".
     */
    isCached(url: string): Promise<boolean>;
    /**
     * Create a repo rooted at `storeDir` inside the origin's private
     * filesystem, defaulting to `"cera-models"`. Nothing is created
     * until the first download, so constructing this is free and
     * cannot fail on a storage error.
     *
     * `storeDir` is a single directory name, not a path: it goes
     * through the same allowlist as URL-derived cache segments, so a
     * name containing `/` or `..` is rejected here rather than
     * silently addressing something else.
     */
    constructor(store_dir?: string | null);
    /**
     * Drop the cache entry for one URL, leaving the rest intact.
     * Returns whether anything was removed. Removes the `.sha256`
     * sidecar alongside the payload, so a later re-download can't
     * match a stale hash.
     */
    remove(url: string): Promise<boolean>;
    /**
     * Cached text for `url` (UTF-8), downloading first if needed.
     * Intended for manifests; throws if the bytes aren't valid UTF-8.
     */
    text(url: string): Promise<string>;
    /**
     * The OPFS directory this repo caches under. Matches what was
     * passed to the constructor.
     */
    readonly storeDir: string;
}

/**
 * Loaded inference engine — wraps `cera::CeraEngine` with sync access
 * to model metadata and the tokenizer.
 *
 * JS callers fetch the GGUF (e.g. via `fetch().arrayBuffer()`), pass
 * the bytes to `CeraEngine.fromGgufBytes`, and use the returned
 * handle to read model info or pull a `Tokenizer`. Session-based
 * inference (`generate`, streaming) is intentionally not exposed yet
 * — that shape needs an async/streaming design that lives in a
 * follow-up PR.
 *
 * **Memory:** the loaded GGUF stays resident in wasm linear memory
 * for the lifetime of this object. Call `.free()` (auto-emitted by
 * wasm-bindgen) when done to release it; without that, the entire
 * model lives until the page unloads.
 */
export class CeraEngine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Constructs a `GenerateOpts` seeded with the advisory defaults from the
     * model manifest (if any), falling back to standard defaults for unmentioned fields.
     */
    defaultGenerateOpts(): GenerateOpts;
    /**
     * Load a published LeapBundle by id and quantization, downloading
     * through `repo` and reusing whatever it already cached.
     *
     * This is the browser equivalent of the native
     * `CeraEngine::from_bundle_id`. The manifest picks up every file
     * the bundle names, so a VL or audio bundle arrives complete: no
     * separate mmproj argument and no guessing at the modality, unlike
     * `fromGgufParts` which has only its arguments to go on.
     *
     * `onProgress(url, bytesDownloaded, totalBytes)` fires during
     * downloads only; a fully cached bundle loads without calling it.
     * `totalBytes` is `null` when the server doesn't say.
     *
     * **Memory:** every file lands in wasm linear memory and stays for
     * the engine's lifetime. The bytes are never handed to JS on the
     * way, so this costs one copy of the model rather than two.
     */
    static fromBundleId(repo: BundleRepo, bundle_id: string, quant: string, context_size?: number | null, on_progress?: Function | null): Promise<CeraEngine>;
    /**
     * Load a model from in-memory GGUF bytes. `contextSize` defaults
     * to 4096 if omitted; the actual KV-cache cap is the smaller of
     * the requested size and the model's own `max_seq_len`.
     *
     * The backend is forced to CPU — wasm has no native GPU/Metal
     * backend. Throws on parse failure, unsupported quantization,
     * or unrecognized architecture.
     */
    static fromGgufBytes(bytes: Uint8Array, context_size?: number | null): CeraEngine;
    /**
     * Load a multi-file bundle: the model GGUF plus its multimodal
     * projector ("mmproj"). This is the constructor a VL or audio model
     * needs, and `fromGgufBytes` structurally cannot be: the vision tower
     * and the audio encoder live in a *second* GGUF, and that one takes a
     * single buffer.
     *
     * `mmproj` may be `null`, in which case this is exactly
     * `fromGgufBytes` with an explicit context size.
     *
     * **Modality is inferred from the arguments, not just the header.**
     * Every published LFM2-VL model reports `architecture = "lfm2"`, the
     * same string a text model reports, because the vision half is entirely
     * in the mmproj. So passing an `mmproj` alongside a text-arch model is
     * taken as the statement of intent it is and loads as image-to-text;
     * audio models already identify themselves and are unaffected. Pass
     * `inferenceType` explicitly to override (`"llama.cpp/text-to-text"`,
     * `"llama.cpp/image-to-text"`, `"llama.cpp/lfm2-audio-v1"`).
     *
     * A malformed or mismatched mmproj is **not** fatal: it warns and the
     * bundle still serves text, with `capabilities.imageIn` staying false
     * and `appendImage` throwing "no vision encoder attached". That mirrors
     * the native loaders rather than failing a whole page load over a
     * sidecar.
     *
     * **Memory:** both buffers stay resident in wasm linear memory for the
     * engine's lifetime. A VL bundle is the model *plus* the tower.
     */
    static fromGgufParts(bytes: Uint8Array, mmproj?: Uint8Array | null, context_size?: number | null, inference_type?: string | null): CeraEngine;
    /**
     * Load a bundle from the URL of its manifest JSON, for bundles
     * hosted somewhere other than `LiquidAI/LeapBundles`.
     *
     * Files the manifest names are fetched relative to it. Entries
     * with a nested path are refused rather than guessed at: see
     * `bundle::join_url`.
     */
    static fromManifestUrl(repo: BundleRepo, manifest_url: string, context_size?: number | null, on_progress?: Function | null): Promise<CeraEngine>;
    /**
     * Construct a new `Session` for this engine. The `config`
     * freezes per-session knobs — sampler `seed`, `nKeep`
     * pinned-prefix size, `ubatchSize` chunked-prefill batch,
     * `maxSeqLen` KV cap. For the cera defaults
     * (`maxSeqLen = null` → engine's effective cap, i.e.
     * `min(engine.contextSize, model.maxSeqLen)`; `nKeep = 0`,
     * `seed = null`, `ubatchSize = 512`), pass a freshly-
     * constructed `new SessionConfig()`.
     *
     * `config` is **borrowed**, not consumed — JS callers can
     * reuse the same `SessionConfig` across multiple `newSession`
     * calls. Inner state is cloned per-session at the boundary.
     * This mirrors how `Session.generate` borrows `GenerateOpts`.
     * (wasm-bindgen doesn't support `Option<&T>` for wrapper
     * types, so a default-config caller passes
     * `new SessionConfig()` rather than omitting the arg.)
     *
     * The returned `Session` keeps its own `Arc` clones of the
     * engine's model and tokenizer, so freeing the engine doesn't
     * invalidate any in-flight sessions.
     */
    newSession(config: SessionConfig): Session;
    /**
     * The token id of `format`'s tool-call start marker (e.g.
     * `<|tool_call_start|>`) in this model's vocab, for use as a lazy
     * grammar trigger in `GenerateOpts.grammarTriggerTokens`.
     * `undefined` when this tokenizer lacks that special token.
     */
    toolCallStartToken(format: ToolFormat): number | undefined;
    /**
     * The tool-call format auto-detected from this model's architecture, or
     * `undefined` when the architecture has no known tool convention.
     *
     * Engine-level counterpart to the free `detectToolFormat(architecture)`
     * function: this one already knows the loaded model's architecture, so
     * it cannot disagree with it.
     */
    toolFormat(): ToolFormat | undefined;
    /**
     * Transcribe mono `f32` PCM audio (roughly normalized to `[-1.0, 1.0]`)
     * to text, using the model's own audio encoder and chat template.
     *
     * `sampleRate` is the rate of the samples you pass; cera resamples to
     * whatever the encoder wants. A typical browser source is
     * `AudioBuffer.getChannelData(0)` after decoding through
     * `AudioContext.decodeAudioData`, whose `sampleRate` you read off the
     * same `AudioBuffer`.
     *
     * Requires an audio bundle loaded through `fromGgufParts` with its
     * mmproj; otherwise this throws `"modality not supported by this
     * model"`. This runs a full prefill + decode, so it is *slow* on the
     * wasm CPU backend for anything but short clips.
     */
    transcribe(pcm: Float32Array, sample_rate: number): string;
    /**
     * `true` when the GGUF declares `tokenizer.ggml.add_bos_token`.
     * Callers that hand-build a token sequence from `Tokenizer.encode`
     * should prepend `Tokenizer.bosToken` when this is `true` (and
     * the model has a BOS) — cera's encoder returns the raw tokens
     * without that prefix.
     */
    readonly addBosToken: boolean;
    /**
     * `true` when the GGUF declares `tokenizer.ggml.add_eos_token`. Prefer
     * `Tokenizer.encodeSpecial`, which applies this (and BOS) automatically.
     */
    readonly addEosToken: boolean;
    /**
     * Model architecture string from the GGUF metadata
     * (e.g. `"lfm2"`, `"llama"`).
     */
    readonly architecture: string;
    /**
     * Modality capability flags reported by the loaded model.
     * See the `Capabilities` interface in the generated `.d.ts`
     * for the field shape.
     *
     * These reflect the bundle you actually loaded. A model opened with
     * `fromGgufBytes` is text-only by construction and reports
     * `{ textIn: true, textOut: true }` with everything else false, because
     * a single GGUF cannot carry a vision tower or an audio encoder. To get
     * `imageIn` or `audioIn`, load the mmproj too via `fromGgufParts`.
     *
     * A bundle whose mmproj failed to parse reports the flag as false and
     * logs a warning, so this stays an accurate answer about what the
     * engine can do rather than what the caller intended.
     */
    readonly capabilities: Capabilities;
    /**
     * Requested context-window size (KV cache cap) the engine was
     * configured with. Mirrors what `fromGgufBytes(bytes,
     * contextSize)` resolved to — i.e. the value of `contextSize`
     * you passed in, or `4096` if you omitted it. Unlike
     * `cera-ffi`'s `EngineConfig::try_from`, the wasm load path
     * has no `0` → `maxSeqLen` translation: a `contextSize` of `0`
     * trips cera core's `context_size > 0` load assertion and
     * `fromGgufBytes` throws.
     *
     * Note this is the **engine-level requested** cap, not a
     * per-session ceiling. cera core clamps the model's
     * `maxSeqLen` at load time to `min(contextSize,
     * gguf_max_seq_len)`, so `engine.maxSeqLen` is already the
     * effective ceiling — `contextSize` is informational ("what
     * cap did I load with?") rather than a value to `Math.min`
     * against `maxSeqLen` at call sites.
     */
    readonly contextSize: number;
    /**
     * `true` when the loaded GGUF carries an embedded Jinja chat
     * template. JS callers can use this to decide whether to render
     * `Tokenizer.chatTemplate` themselves vs falling back to a
     * hard-coded prompt format.
     */
    readonly hasChatTemplate: boolean;
    /**
     * Maximum sequence length the model was trained for. Independent
     * of the engine's `contextSize` config — that one is the KV
     * cache cap, this is the model's positional encoding ceiling.
     */
    readonly maxSeqLen: number;
    /**
     * Everything `CeraEngine`'s individual metadata getters report, in one
     * object. See the `ModelMetadata` interface in the generated `.d.ts`.
     */
    readonly metadata: ModelMetadata;
    /**
     * Quantization label from the GGUF (e.g. `"Q4_0"`, `"Q4_K_M"`).
     * Useful for telling users what they actually loaded when the
     * download URL doesn't make it obvious.
     */
    readonly quantization: string;
    /**
     * Returns a `Tokenizer` handle bound to this engine's vocab.
     * Each call allocates a fresh JS object but the underlying
     * tokenizer state is shared via `Arc` — cheap to call, JS
     * callers can cache the result if they prefer one handle.
     */
    readonly tokenizer: Tokenizer;
    readonly vocabSize: number;
}

/**
 * Per-call generation options. Constructed via `new GenerateOpts()`
 * in JS (returns the cera defaults: `maxTokens=256`,
 * `temperature=0.7`, `topP=0.9`, `topK=40`, no stop tokens, flush
 * every 16 tokens or 50 ms).
 *
 * `minP` and `repetitionPenalty` are honored in the stochastic path
 * (`temperature > 0` and `topK != 1`); greedy/argmax decoding ignores them.
 */
export class GenerateOpts {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Remove any grammar constraint, returning to unconstrained decoding.
     */
    clearGrammar(): void;
    constructor();
    /**
     * Constrain decoding to a GBNF grammar (source text, e.g. a JSON grammar).
     * Each step masks the logits so only tokens the grammar accepts are
     * sampled. Throws a `JsError` if the grammar fails to compile; replaces any
     * grammar set by a prior call. A setter can't surface the parse error, so
     * this is a method rather than a `grammar` property.
     */
    setGrammar(gbnf: string): void;
    flushEveryMs: number;
    flushEveryTokens: number;
    /**
     * Lazy-grammar trigger token IDs (tool calling). When non-empty and a
     * grammar is set (`GenerateOpts.setGrammar`), the grammar stays inactive until
     * the model emits one of these tokens (e.g. the tool-call start marker
     * from `Tokenizer.specialTokenId`), then constrains the call and
     * deactivates on completion. Empty (default) → the grammar is active from
     * the first token.
     */
    grammarTriggerTokens: Uint32Array;
    /**
     * Whether a grammar constraint is currently set.
     */
    readonly hasGrammar: boolean;
    /**
     * Ignore end-of-generation: EOS and `stopTokens` are not honored, so
     * decode always runs to `maxTokens`. For benchmark loops that must
     * cover an exact token count. `false` by default.
     */
    ignoreEos: boolean;
    maxTokens: number;
    /**
     * Min-p (relative) nucleus cutoff: drop tokens below `minP * pMax`.
     * `0.0` (default) disables it. Honored in the stochastic path.
     */
    minP: number;
    /**
     * Repetition penalty over tokens generated this call. `1.0` (default)
     * disables it. Honored in the stochastic path.
     */
    repetitionPenalty: number;
    /**
     * Token IDs that, if produced, end decoding with
     * `finishReason = "Stop"`. Empty by default.
     */
    stopTokens: Uint32Array;
    temperature: number;
    topK: number;
    topP: number;
}

/**
 * Summary returned from a completed `Session.generate` call.
 */
export class GenerateSummary {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly decodeMs: number;
    /**
     * Why decode ended. One of `"MaxTokens"`, `"Stop"`,
     * `"Cancelled"`, `"ContextFull"`, or `"Error(<message>)"` —
     * the `Error(...)` form preserves the inner string verbatim
     * (no surrounding quotes), so JS callers can log it directly.
     */
    readonly finishReason: string;
    readonly promptEvalMs: number;
    readonly promptEvalTokens: number;
    readonly tokensGenerated: number;
}

/**
 * A loaded LoRA adapter, ready to attach to a [`Session`] via `attachLora`.
 * Load it once (from bytes — the browser has no filesystem) and reuse the
 * handle across sessions; the factors are reference-counted internally.
 */
export class LoraAdapters {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
     * bytes. `alpha` is read from the adapter's `adapter.lora.alpha` metadata.
     */
    static fromGgufBytes(bytes: Uint8Array): LoraAdapters;
    /**
     * Load a PEFT `.safetensors` adapter from bytes. PEFT keeps `alpha` in a
     * sibling `adapter_config.json`, so pass it explicitly (`undefined` ⇒
     * scale = 1, i.e. `alpha == rank`).
     */
    static fromSafetensorsBytes(bytes: Uint8Array, alpha?: number | null): LoraAdapters;
    /**
     * Number of `(layer, target)` low-rank deltas the adapter carries.
     */
    targetCount(): number;
}

/**
 * Parsed view of a LeapBundles `*.json` manifest.
 *
 * JS callers fetch the manifest bytes (e.g. via `fetch().arrayBuffer()`)
 * and pass them to `Manifest.parse`. The wrapper exposes the typed
 * fields cera already understands; the raw `serde_json::Value`
 * retained on the inner `cera::manifest::Manifest` is intentionally
 * **not** exposed here — JS callers can re-parse the JSON themselves
 * for forward-compat fields, and we don't want to commit to a
 * `serde-wasm-bindgen` round-trip on every getter.
 */
export class Manifest {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Parse a JSON manifest from raw bytes. Throws a `JsError` on
     * malformed JSON or when required fields are missing or wrongly
     * typed (e.g. no `load_time_parameters.model`). Unknown
     * `inference_type` values are **not** an error — they round-trip
     * through `cera::manifest::InferenceType::Unknown(String)` and
     * surface verbatim via the `inferenceType` getter, so JS callers
     * can decide how to react instead of catching here.
     */
    static parse(json_bytes: Uint8Array): Manifest;
    /**
     * URL of the audio-decoder GGUF for audio-out models.
     */
    readonly audioDecoderUrl: string | undefined;
    /**
     * URL of the audio-tokenizer checkpoint (typically `.safetensors`).
     */
    readonly audioTokenizerUrl: string | undefined;
    /**
     * Jinja chat template override from the manifest, if present.
     * `undefined` means "use the template embedded in the GGUF
     * metadata" (cera's standard fallback).
     */
    readonly chatTemplate: string | undefined;
    /**
     * Raw `inference_type` string (e.g. `llama.cpp/text-to-text`).
     * Round-trips through cera's enum, so unknown variants come back
     * as their original string — no information loss.
     */
    readonly inferenceType: string;
    /**
     * URL (or local path string) for the primary model GGUF.
     */
    readonly modelUrl: string;
    /**
     * URL of the multimodal projector GGUF if the manifest declares
     * one (VL / audio models). `undefined` for plain text models.
     */
    readonly multimodalProjectorUrl: string | undefined;
    readonly schemaVersion: string;
}

/**
 * Stateful generation handle. Built via `CeraEngine.newSession(config)`.
 *
 * JS callers seed the conversation by calling `appendText` /
 * `appendTokens` and then drive decode with `generate(opts, cb)`.
 * The callback fires once per flush boundary (every
 * `flushEveryTokens` decoded tokens, or `flushEveryMs` ms,
 * whichever comes first) with the new tokens.
 *
 * **Worker note:** `generate` is synchronous and will block the
 * thread it runs on for the duration of decode (potentially
 * seconds). On the browser main thread that freezes the page —
 * always call from a Web Worker. On Node it also blocks the JS
 * event loop (libuv's background I/O thread pool keeps running,
 * but JS callbacks queue): use `worker_threads` for server
 * processes that need to handle other requests during inference;
 * one-off scripts are fine to run sync.
 *
 * **Cancellation:** since the worker thread is blocked inside
 * `generate`, the worker's own `onmessage` handler can't run —
 * incoming `postMessage({kind:'cancel'})` queues but doesn't
 * dispatch until `generate` returns, so a flag set by that
 * handler can't be updated mid-decode. To cancel during a
 * running `generate` call, either call `session.cancel()` from inside
 * the token callback based on state it can observe directly
 * (elapsed time, token budget, accumulated content), or use
 * cross-thread shared memory signalling (`SharedArrayBuffer` +
 * `Atomics`) — see `cera-wasm/README.md` for the full
 * `SharedArrayBuffer` pattern, which requires cross-origin
 * isolation in browsers.
 */
export class Session {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Append PCM audio samples (mono `f32`, normalized to roughly
     * `[-1.0, 1.0]`) at `sample_rate` Hz.
     *
     * Non-16kHz inputs are automatically linearly resampled to 16 kHz.
     * `samples` arrives as `Float32Array` on the JS side. The
     * wasm-bindgen boundary copies the typed-array contents into
     * wasm linear memory once — there's no per-element boxing
     * (contrast with Kotlin's `List<Float>` 4× memory overhead
     * flagged in PR #78). The `&[f32]` Rust signature matches
     * `appendTokens(&[u32])` and avoids the per-call `Vec`
     * allocation that an owned parameter would require.
     *
     * Errors today are thrown as JS `Error`s; the message string
     * is the underlying `cera::CeraError::Display` text (same as
     * `appendText` / `appendTokens` produce):
     * - `"empty input"` if `samples.length === 0` — fast-fail at
     *   the wasm boundary, parity with `appendText` /
     *   `appendTokens` empty-input rejection.
     * - `"modality not supported by this model"` when
     *   `session.capabilities.audioIn === false`. Load the bundle's
     *   mmproj through `CeraEngine.fromGgufParts` to get an
     *   audio-capable session; `fromGgufBytes` cannot produce one.
     * - `"backend: Session::append_audio: no audio encoder attached..."`
     *   when the bundle claimed audio but its mmproj failed to parse.
     */
    appendAudio(samples: Float32Array, sample_rate: number): void;
    /**
     * Encode an image and append its embeddings to the KV cache.
     *
     * `bytes` is an encoded image file (PNG or JPEG), not raw pixels: pass
     * a `Uint8Array` over a `fetch` response, a `File`/`Blob`
     * `arrayBuffer()`, or a canvas `toBlob` result.
     *
     * `maxLongSize` caps the longest side of the **encoded** image in
     * pixels, trading detail for speed and token count:
     *
     * - `null`/omitted: use the session default
     *   (`setImageMaxLongSize`, itself unset by default).
     * - `0`: force *no* cap for this call, overriding a session default.
     * - `n`: cap at `n` pixels.
     *
     * Requires a VL bundle (`capabilities.imageIn === true`), which means
     * loading via `CeraEngine.fromGgufParts` with the vision mmproj.
     * Otherwise this throws `"modality not supported by this model"`.
     * Building `cera-wasm` with `--no-default-features` (dropping the `vl`
     * feature) produces the same error, since the image decoders are gone.
     */
    appendImage(bytes: Uint8Array, max_long_size?: number | null): void;
    /**
     * Tokenize `text` using the session's tokenizer and append the
     * result to the KV cache. Equivalent to
     * `appendTokens(tokenizer.encode(text))` but avoids the round
     * trip through JS for the encoded buffer.
     */
    appendText(text: string): void;
    /**
     * Append already-tokenized IDs to the KV cache. Use when you
     * need control over BOS/EOS framing or you've cached tokens
     * from a previous encode.
     */
    appendTokens(tokens: Uint32Array): void;
    /**
     * Attach a [`LoraAdapters`] to this session. Applied to every subsequent
     * forward pass — generation **and** hidden-states extraction — until
     * removed or replaced (hot-swap), and preserved across `reset()`. Throws if
     * the adapter's dimensions don't match the loaded model. Only affects tokens
     * processed after the call (doesn't retroactively re-adapt cached KV).
     */
    attachLora(adapters: LoraAdapters): void;
    /**
     * Flip the cancel atomic, requesting that any in-flight
     * `generate` call exit at its next checkpoint with
     * `finishReason = "Cancelled"`. Safe to call from any thread
     * (including a Worker that owns this session — though wasm
     * without SharedArrayBuffer makes cross-thread sharing
     * unusual).
     */
    cancel(): void;
    /**
     * Clear the cancel flag without dropping any session state.
     * Use this after observing a cancellation signal — either a
     * thrown cancellation error from `appendText` / `appendTokens`
     * (mid-prefill cancellation surfaces as a thrown error) or
     * `summary.finishReason === "Cancelled"` on the value
     * returned from `generate` (cancellation during decode is
     * reported via the finish reason, not a thrown error) — when
     * you want to resume work on the same session without losing
     * the accumulated KV cache.
     *
     * Compared to `reset()`:
     * - `clearCancel`: keeps KV state, `position`, and the
     *   sampler intact; only flips the cancel atomic back to
     *   `false`. Use for "interrupted but continuing" flows.
     * - `reset()`: drops KV cache, `position`, last logits, and
     *   re-seeds the sampler. Use for "clear conversation"
     *   flows.
     *
     * **Call sequencing:** invoke this *after* `generate` /
     * `appendText` / `appendTokens` has returned. Even though
     * the underlying cera method takes `&self`, wasm-bindgen's
     * JS-side borrow check on the `Session` wrapper rejects any
     * method call (including this `&self` one) while another
     * method is still borrowing the same handle — calling
     * `session.clearCancel()` from inside a `generate` token
     * callback would throw "recursive use of an object". The
     * `&self` Rust shape matters in the native binding
     * (`cera-ffi`) where there's no JS-side borrow check; in
     * wasm it just means there's no `&mut self` cost on the cera
     * core side.
     */
    clearCancel(): void;
    /**
     * Decode tokens until `opts.maxTokens`, a stop token, EOS, or
     * `cancel()` fires. The `onTextTokens` callback is invoked once
     * per flush boundary with a `Uint32Array` of the latest tokens
     * (*not* the cumulative buffer — concatenate yourself if you
     * want the full sequence).
     *
     * Returns the `GenerateSummary` once decode finishes. Throws
     * `JsError` on backend failure (the summary's `finishReason`
     * already covers logical end conditions like `"Stop"` or
     * `"ContextFull"`).
     */
    generate(opts: GenerateOpts, on_text_tokens: Function, on_audio_frames?: Function | null): GenerateSummary;
    /**
     * Whether a LoRA adapter is currently attached to this session.
     */
    hasLora(): boolean;
    /**
     * Model hidden dimension `D` — reshape a `[T*D]` hidden-states buffer into
     * `[T][D]` with this. Reads a cached field (set at construction), so — unlike
     * the `&mut self` compute methods — it's safe to call from inside a `generate`
     * callback without a wasm-bindgen borrow panic.
     */
    hiddenSize(): number;
    /**
     * Tokenize `text` and return its per-token hidden states as a `Float32Array`.
     */
    hiddenStatesForText(text: string): Float32Array;
    /**
     * Per-token last-layer hidden states (post-final-RMSNorm — the llama.cpp
     * `--pooling none` vector) for `tokens`, as a `Float32Array` of length
     * `tokens.length * hiddenSize` (row-major; token `t` channel `c` at
     * `t*hiddenSize + c`). The wasm boundary copies the buffer into the JS heap
     * once. Side-effect-free — does not disturb the generation KV.
     */
    hiddenStatesForTokens(tokens: Uint32Array): Float32Array;
    /**
     * Mean-pooled hidden state — a single `Float32Array` of length `hiddenSize`
     * (the common classifier path: pool in Rust, ship `D` floats not `T*D`).
     */
    hiddenStatesMeanPooled(tokens: Uint32Array): Float32Array;
    /**
     * Remove any attached LoRA adapter, returning to base-model inference.
     */
    removeLora(): void;
    /**
     * Drop accumulated state and return the session to a freshly-
     * opened shape. Clears the KV cache, `position`, the last
     * logits, and the cancel flag, then re-seeds the sampler from
     * the `SessionConfig.seed` originally passed to `newSession`.
     *
     * Use this for "clear conversation" UI actions — it skips the
     * per-session setup cost that `engine.newSession(config)`
     * would pay (model + tokenizer Arc clones, sampler ctor),
     * while still leaving the session indistinguishable from a
     * fresh one.
     *
     * Sampler re-seed semantics:
     * - `SessionConfig.seed = some bigint` — deterministic
     *   sessions stay deterministic across `reset()`; the next
     *   `generate` produces the same first token sequence as the
     *   original.
     * - `SessionConfig.seed = null` — the sampler picks a new
     *   random seed on each `reset()`, so successive
     *   conversations decorrelate.
     *
     * Engine-level disk prefix cache (when configured on
     * `CeraEngine`) is not touched — those entries are
     * engine-scoped, not session-scoped.
     *
     * **Threading:** unlike `cancel()` (which only flips an
     * atomic and is safe to call concurrently with anything),
     * `reset()` takes `&mut self` and rebuilds non-atomic
     * internal state (KV cache, sampler). Must be called on
     * the owning thread, with no in-flight `generate` /
     * `appendText` / `appendTokens` running. The wasm-bindgen
     * borrow check enforces this within a single Worker; if
     * you share a `Session` across Workers via
     * `SharedArrayBuffer`-style schemes, it's on you to
     * serialize calls.
     */
    reset(): void;
    /**
     * Set the session-default cap on the longest side of an appended
     * image, in pixels. `null` clears it (no cap).
     *
     * Applies to later `appendImage` calls that pass no explicit
     * `maxLongSize`. A per-call value always wins.
     */
    setImageMaxLongSize(max_long_size?: number | null): void;
    /**
     * Modality capability flags reported by the model backing
     * this session. Same shape as `CeraEngine.capabilities` —
     * see that getter for the `Capabilities` field documentation
     * and the synthetic-text caveat that applies to all
     * `fromGgufBytes`-loaded models today.
     */
    readonly capabilities: Capabilities;
    /**
     * Current KV cache position (number of tokens currently held).
     */
    readonly position: number;
}

/**
 * Per-session knobs frozen at `CeraEngine.newSession(config)` time.
 * Constructed via `new SessionConfig()` in JS (returns the cera
 * defaults: `maxSeqLen=null` → engine's effective max, `nKeep=0`,
 * `seed=null`, `ubatchSize=512`, `kvCompression=null`).
 *
 * Set `kvCompression` to a `TurboQuantConfig` to compress the
 * KV cache (~3 bits/elem for keys, ~2 bits/elem for values).
 * See the per-property doc for trade-offs.
 */
export class SessionConfig {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    /**
     * KV cache compression configuration. `null` (default) stores
     * keys and values as f32 — best fidelity, biggest memory
     * footprint. Set to a `TurboQuantConfig` to **request**
     * TurboQuant compression — keys to ~3 bits/elem, values to
     * ~2 bits/elem (plus a norm word per vector); the same `seed`
     * reproduces the same per-layer Hadamard rotations
     * deterministically.
     *
     * **Silent fallbacks to be aware of:**
     * - TurboQuant only kicks in when the loaded model's
     *   attention `head_dim` is a power of two (a constraint of
     *   the Hadamard rotation). If it isn't, cera logs a warning
     *   and falls back to the uncompressed f32 path even with
     *   this set — there's no JS-visible error, just no
     *   compression.
     * - `nKeep` (context-shift) is incompatible with TurboQuant.
     *   Setting both gets a warning at session creation and the
     *   `nKeep` value is ignored on KV overflow (the cache
     *   overflows hard instead of shifting). Pick one.
     * - This config drives the CPU session. `WebGpuSession` takes
     *   no `SessionConfig` — it accepts its own `kvCompression`
     *   argument on `create` instead, and its `kvCompression`
     *   getter reports the mode that actually took effect. Its
     *   `head_dim` constraint is stricter than the CPU's: a power
     *   of two that is also `<= 128` and a multiple of 32, and
     *   keys *and* values must both be compressed (a single-sided
     *   debug config falls back to f32 there).
     *
     * Setting this consumes the JS-side `TurboQuantConfig`
     * handle (wasm-bindgen's `Option<T>` parameter shape). Read
     * back via the getter — which returns a fresh handle that's
     * a snapshot, not a live link — if you need to inspect the
     * current config without affecting it.
     *
     * Assign a fresh config per session. Reusing an already-
     * consumed handle does **not** throw in a release build:
     * wasm-bindgen lowers it to pointer 0, which arrives as
     * `None`, so the second session silently gets uncompressed
     * KV. (A `--dev` build does throw "Attempt to use a moved
     * value" — so this is a bug that only appears in release.)
     */
    get kvCompression(): TurboQuantConfig | undefined;
    set kvCompression(value: TurboQuantConfig | null | undefined);
    /**
     * Cap on total tokens held in KV. `null` (the common case)
     * defers to the engine's effective max — i.e.
     * `min(engine.contextSize, model.maxSeqLen)`. Set to a
     * smaller value here to further lower the cap; values larger
     * than the engine's effective max are still capped at it.
     */
    get maxSeqLen(): number | undefined;
    set maxSeqLen(value: number | null | undefined);
    /**
     * Number of leading tokens pinned in KV across context shifts —
     * a system prompt or persistent prefix that should survive
     * when the cache fills. `0` (default) disables the pin.
     */
    nKeep: number;
    /**
     * Deterministic sampler seed. `null` (default) uses a fresh
     * random seed per session — set this to make a session's
     * outputs reproducible across runs (useful for testing /
     * demos / regression checks).
     */
    get seed(): bigint | undefined;
    set seed(value: bigint | null | undefined);
    /**
     * Chunked-prefill batch size (tokens per micro-batch during
     * the prefill pass). Smaller values give finer-grained
     * `Session.cancel()` checkpoints during long prompt eval at
     * some perf cost. cera's default is `512`.
     */
    ubatchSize: number;
}

/**
 * BPE tokenizer wrapper. Constructed via `CeraEngine.tokenizer`;
 * no standalone `from*` constructor (the GGUF metadata required to
 * build one is reachable only through the engine).
 *
 * Round-trip note: `decode(encode(text))` is **not** guaranteed to
 * be byte-identical to `text` for inputs containing tokens that
 * don't survive BPE merge replay (rare in practice — BOS/EOS,
 * some byte-level edge cases). When you need exact reproduction,
 * keep the original string around.
 */
export class Tokenizer {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Render the model's embedded Jinja chat template against a
     * `[{ role, content }, ...]` array, returning the prompt
     * string ready for `Tokenizer.encode` + `Session.appendTokens`.
     *
     * `addGenerationPrompt` defaults to `true` (the common case
     * when sending to the model expecting a response). Set to
     * `false` when you only want the conversation rendered without
     * the trailing assistant-prompt suffix.
     *
     * Throws `JsError` on:
     * - the model not carrying a chat template
     *   (`engine.hasChatTemplate === false`),
     * - malformed `messages` (not an array, or entries missing
     *   `role`/`content` strings),
     * - a Jinja render failure (template references an undefined
     *   variable, etc.).
     */
    applyChatTemplate(messages: ChatMessage[], add_generation_prompt?: boolean | null): string;
    /**
     * Like `applyChatTemplate`, but also injects a `tools` array so a
     * tool-trained model renders its tool-definition block. `toolsJson` is a
     * JSON string encoding an array of `ToolDef` (`[{name, description?,
     * parameters?}]`). Throws on invalid `toolsJson` or a render failure.
     */
    applyChatTemplateWithTools(messages: ChatMessage[], tools_json: string, add_generation_prompt?: boolean | null): string;
    /**
     * Detokenize back to a UTF-8 string. Lossy for tokens whose
     * byte sequences don't decode to valid UTF-8 — those are
     * replaced with U+FFFD per `String::from_utf8_lossy`.
     */
    decode(tokens: Uint32Array): string;
    /**
     * Tokenize a UTF-8 string. Returns the token IDs as a
     * `Uint32Array`. No BOS/EOS prefix — callers that want them
     * should prepend `bosToken` / append `eosToken` manually, or use
     * `encodeSpecial`.
     */
    encode(text: string): Uint32Array;
    /**
     * Encode with optional special markers — the analog of llama.cpp's
     * `llama_tokenize(..., add_special)`. When `addSpecial` is true, BOS is
     * prepended iff the GGUF declares `tokenizer.ggml.add_bos_token` and EOS
     * appended iff it declares `tokenizer.ggml.add_eos_token`, so token counts
     * match llama.cpp. With `addSpecial = false` this is exactly `encode`.
     */
    encodeSpecial(text: string, add_special: boolean): Uint32Array;
    /**
     * `true` when `id` is registered as a control or user-defined
     * special token in the model's GGUF metadata
     * (`tokenizer.ggml.token_type` types `3` / `4`). Useful for
     * output filtering — e.g. dropping `<|im_end|>` from a
     * `Session.generate` token-callback batch before joining the
     * IDs into UI-rendered text — and for token-class
     * classification in analysis tools.
     *
     * Out-of-range IDs (>= vocab size) and regular vocab tokens
     * both return `false`. Companion to `specialTokenId` which
     * goes the other direction (name → ID).
     */
    isSpecialToken(id: number): boolean;
    /**
     * Look up a special-token ID by its literal name (e.g.
     * `"<|im_start|>"`, `"<|tool_calls_section_begin|>"`).
     * Returns `undefined` when no entry exists for that name in
     * the model's special-token registry.
     *
     * Lookup scope: only tokens flagged as control or
     * user-defined in the GGUF metadata are registered for this
     * lookup. cera reads `tokenizer.ggml.token_type` and admits
     * tokens with type `3` (control) or type `4` (user-defined);
     * regular vocab entries are not reachable via this method
     * even though their names exist in `tokenizer.ggml.tokens`.
     * Names accepted here are the literal vocab strings indexed
     * by the special token's ID.
     *
     * Useful for constructing prompts with specific control
     * tokens directly (chat-template-like flows) without
     * round-tripping through `applyChatTemplate`. For BOS / EOS
     * prefer `bosToken` / `eosToken` (named getters that don't
     * risk a typo in the lookup string).
     *
     * Mirrors `CeraEngine.specialTokenId` from cera-ffi (where
     * it lives engine-side); cera-wasm hangs it off `Tokenizer`
     * to match the established `engine.tokenizer.<method>`
     * access pattern.
     */
    specialTokenId(name: string): number | undefined;
    /**
     * Whether the GGUF asks for a BOS token to be prepended
     * (`tokenizer.ggml.add_bos_token`).
     *
     * Needed to frame a prompt correctly without `encodeSpecial`: that
     * helper prepends BOS *and* appends EOS together, and a chat-template
     * prompt wants the first and not the second. Callers doing their own
     * framing read this and prepend `bosToken` themselves.
     */
    readonly addBosToken: boolean;
    /**
     * BOS token ID, if the GGUF metadata declares one.
     */
    readonly bosToken: number | undefined;
    /**
     * Raw embedded Jinja chat template from the GGUF metadata, if
     * any. Most callers should use [`Self::apply_chat_template`]
     * (`applyChatTemplate` in JS) instead — this getter is for
     * inspection or for callers who want to render with a
     * different Jinja runtime.
     */
    readonly chatTemplate: string | undefined;
    /**
     * EOS token ID, if the GGUF metadata declares one.
     */
    readonly eosToken: number | undefined;
    readonly vocabSize: number;
}

/**
 * The tool-call wire format a model family uses. Get one from
 * `detectToolFormat(architecture)` or choose explicitly.
 */
export enum ToolFormat {
    /**
     * LFM2 / LFM2.5: Pythonic `[get_weather(city="Paris")]`.
     */
    Lfm2Pythonic = 0,
    /**
     * Hermes / Qwen: JSON `{"name":…,"arguments":{…}}`.
     */
    Hermes = 1,
}

/**
 * TurboQuant KV-cache compression configuration. Construct via
 * `new TurboQuantConfig(seed)` for the common production setup
 * (both `keys` and `values` compressed); flip the per-side
 * toggles for debugging (e.g. to isolate how much drift each
 * side contributes).
 *
 * - **Keys**: 2-bit PolarQuant + 1-bit QJL residual
 *   (3 bits/elem + a packed norm word per vector).
 * - **Values**: 2-bit PolarQuant only (2 bits/elem + a packed
 *   norm word per vector).
 *
 * `seed` drives the per-layer randomized Hadamard rotations —
 * the same seed produces the same rotations deterministically,
 * so a seeded session with TurboQuant on stays bitwise-
 * reproducible across runs.
 */
export class TurboQuantConfig {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Construct with the common production setup: both keys and
     * values compressed. Pass an explicit `seed` so the per-layer
     * rotations are reproducible.
     */
    constructor(seed: bigint);
    /**
     * Compress the K side of the KV cache. Default `true`.
     * Useful to flip off when debugging quality regressions to
     * isolate K-side vs V-side contribution.
     */
    keys: boolean;
    /**
     * Hadamard-rotation seed. Same seed → same rotations →
     * reproducible KV cache contents (necessary for bitwise-
     * identical replay across sessions).
     */
    seed: bigint;
    /**
     * Compress the V side of the KV cache. Default `true`.
     */
    values: boolean;
}

/**
 * Returns the version of the `cera` core library this binding wraps.
 *
 * Note this is **`cera`'s** version, not `cera-wasm`'s — JS callers
 * usually want to know what core lib is driving the engine, since
 * the wrapper crate version may evolve independently.
 */
export function ceraVersion(): string;

/**
 * Describe the CPU backend tier this build resolved at runtime, e.g.
 * `"tier=wasm_simd128 [simd128]"`.
 *
 * Diagnostic: it tells you whether the SIMD kernels are actually live,
 * which is the difference between roughly 1.4 and 0.64 tokens/second on the
 * wasm CPU path. A browser without the simd128 proposal, or a `.wasm` built
 * without `+simd128`, reports the scalar tier.
 */
export function cpuBackendReport(): string;

/**
 * Detect the tool-call format for a model architecture string (`"lfm2"`,
 * `"qwen3"`, …). Returns `undefined` for architectures with no known
 * convention.
 *
 * Prefer `CeraEngine.toolFormat` when you have an engine: it reads the
 * loaded model's own architecture, so it cannot be given the wrong string.
 */
export function detectToolFormat(architecture: string): ToolFormat | undefined;

export function initThreadPool(num_threads: number): Promise<any>;

/**
 * Bundles published on `LiquidAI/LeapBundles`, as
 * `[{ name, quants: [...] }]`.
 *
 * One GET to the HuggingFace model-info endpoint, grouped by the same
 * parser the native CLI uses, so the browser and the CLI list the same
 * catalog. Entries whose names wouldn't survive
 * `CeraEngine.fromBundleId` are filtered out rather than offered.
 */
export function listLeapBundles(): Promise<any>;

/**
 * Parse tool calls out of generated model text. Returns a JSON string
 * encoding an array of `ToolCall` (`[{name, arguments}]`) — `JSON.parse` it.
 * An empty array means the reply had no tool call.
 */
export function parseToolCalls(text: string, format: ToolFormat): string;

/**
 * Ask the browser to exempt this origin's storage from eviction under
 * disk pressure, resolving to whether persistence is now in effect.
 *
 * Worth calling before a multi-GB download. Without it, a browser is
 * free to evict the cache: Chrome does so only under real pressure,
 * but Safari discards non-persisted storage after roughly a week of
 * the site going unused, which shows up as a surprise re-download.
 * Some browsers grant it silently based on engagement, others prompt,
 * and a `false` result is normal rather than an error.
 *
 * **Requesting persistence is a Window-only capability.** `persist()`
 * is not exposed on a worker's `StorageManager`, and a worker is
 * exactly where an engine embedder tends to run, so calling it there
 * would throw for a completely ordinary caller. From a worker this
 * falls back to `persisted()`, which *is* exposed and reports whether
 * the page already obtained persistence. So the return value always
 * answers "is this origin's storage protected", and a worker that
 * wants to *change* the answer has to ask its page to call this.
 */
export function persistStorage(): Promise<boolean>;

/**
 * Build a GBNF grammar constraining output to a valid call for one of the
 * tools in `toolsJson` (a JSON array of `ToolDef`). Feed the result to
 * `GenerateOpts.setGrammar` and set `GenerateOpts.grammarTriggerTokens` for a lazy
 * tool-call trigger.
 */
export function toolGrammar(tools_json: string, format: ToolFormat): string;

export class wbg_rayon_PoolBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    build(): void;
    numThreads(): number;
    receiver(): number;
}

export function wbg_rayon_start_worker(receiver: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly __wbg_bundlerepo_free: (a: number, b: number) => void;
    readonly __wbg_ceraengine_free: (a: number, b: number) => void;
    readonly __wbg_generateopts_free: (a: number, b: number) => void;
    readonly __wbg_generatesummary_free: (a: number, b: number) => void;
    readonly __wbg_loraadapters_free: (a: number, b: number) => void;
    readonly __wbg_manifest_free: (a: number, b: number) => void;
    readonly __wbg_session_free: (a: number, b: number) => void;
    readonly __wbg_sessionconfig_free: (a: number, b: number) => void;
    readonly __wbg_tokenizer_free: (a: number, b: number) => void;
    readonly __wbg_turboquantconfig_free: (a: number, b: number) => void;
    readonly bundlerepo_bytes: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly bundlerepo_cacheSize: (a: number) => number;
    readonly bundlerepo_clearCache: (a: number) => number;
    readonly bundlerepo_download: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly bundlerepo_isCached: (a: number, b: number, c: number) => number;
    readonly bundlerepo_new: (a: number, b: number, c: number) => void;
    readonly bundlerepo_remove: (a: number, b: number, c: number) => number;
    readonly bundlerepo_storeDir: (a: number, b: number) => void;
    readonly bundlerepo_text: (a: number, b: number, c: number) => number;
    readonly ceraVersion: (a: number) => void;
    readonly ceraengine_addBosToken: (a: number) => number;
    readonly ceraengine_addEosToken: (a: number) => number;
    readonly ceraengine_architecture: (a: number, b: number) => void;
    readonly ceraengine_capabilities: (a: number) => number;
    readonly ceraengine_contextSize: (a: number) => number;
    readonly ceraengine_defaultGenerateOpts: (a: number) => number;
    readonly ceraengine_fromBundleId: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly ceraengine_fromGgufBytes: (a: number, b: number, c: number, d: number) => void;
    readonly ceraengine_fromGgufParts: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly ceraengine_fromManifestUrl: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly ceraengine_hasChatTemplate: (a: number) => number;
    readonly ceraengine_maxSeqLen: (a: number) => number;
    readonly ceraengine_metadata: (a: number) => number;
    readonly ceraengine_newSession: (a: number, b: number, c: number) => void;
    readonly ceraengine_quantization: (a: number, b: number) => void;
    readonly ceraengine_tokenizer: (a: number) => number;
    readonly ceraengine_toolCallStartToken: (a: number, b: number) => number;
    readonly ceraengine_toolFormat: (a: number) => number;
    readonly ceraengine_transcribe: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly ceraengine_vocabSize: (a: number) => number;
    readonly cpuBackendReport: (a: number) => void;
    readonly detectToolFormat: (a: number, b: number) => number;
    readonly generateopts_clearGrammar: (a: number) => void;
    readonly generateopts_flushEveryMs: (a: number) => number;
    readonly generateopts_flushEveryTokens: (a: number) => number;
    readonly generateopts_grammarTriggerTokens: (a: number, b: number) => void;
    readonly generateopts_hasGrammar: (a: number) => number;
    readonly generateopts_ignoreEos: (a: number) => number;
    readonly generateopts_maxTokens: (a: number) => number;
    readonly generateopts_minP: (a: number) => number;
    readonly generateopts_new: () => number;
    readonly generateopts_repetitionPenalty: (a: number) => number;
    readonly generateopts_setGrammar: (a: number, b: number, c: number, d: number) => void;
    readonly generateopts_set_flushEveryMs: (a: number, b: number) => void;
    readonly generateopts_set_flushEveryTokens: (a: number, b: number) => void;
    readonly generateopts_set_grammarTriggerTokens: (a: number, b: number, c: number) => void;
    readonly generateopts_set_ignoreEos: (a: number, b: number) => void;
    readonly generateopts_set_maxTokens: (a: number, b: number) => void;
    readonly generateopts_set_minP: (a: number, b: number) => void;
    readonly generateopts_set_repetitionPenalty: (a: number, b: number) => void;
    readonly generateopts_set_stopTokens: (a: number, b: number, c: number) => void;
    readonly generateopts_set_temperature: (a: number, b: number) => void;
    readonly generateopts_set_topK: (a: number, b: number) => void;
    readonly generateopts_set_topP: (a: number, b: number) => void;
    readonly generateopts_stopTokens: (a: number, b: number) => void;
    readonly generateopts_temperature: (a: number) => number;
    readonly generateopts_topK: (a: number) => number;
    readonly generateopts_topP: (a: number) => number;
    readonly generatesummary_decodeMs: (a: number) => number;
    readonly generatesummary_finishReason: (a: number, b: number) => void;
    readonly generatesummary_promptEvalMs: (a: number) => number;
    readonly generatesummary_promptEvalTokens: (a: number) => number;
    readonly generatesummary_tokensGenerated: (a: number) => number;
    readonly listLeapBundles: () => number;
    readonly loraadapters_fromGgufBytes: (a: number, b: number, c: number) => void;
    readonly loraadapters_fromSafetensorsBytes: (a: number, b: number, c: number, d: number) => void;
    readonly loraadapters_targetCount: (a: number) => number;
    readonly manifest_audioDecoderUrl: (a: number, b: number) => void;
    readonly manifest_audioTokenizerUrl: (a: number, b: number) => void;
    readonly manifest_chatTemplate: (a: number, b: number) => void;
    readonly manifest_inferenceType: (a: number, b: number) => void;
    readonly manifest_modelUrl: (a: number, b: number) => void;
    readonly manifest_multimodalProjectorUrl: (a: number, b: number) => void;
    readonly manifest_parse: (a: number, b: number, c: number) => void;
    readonly manifest_schemaVersion: (a: number, b: number) => void;
    readonly parseToolCalls: (a: number, b: number, c: number, d: number) => void;
    readonly persistStorage: () => number;
    readonly session_appendAudio: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly session_appendImage: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly session_appendText: (a: number, b: number, c: number, d: number) => void;
    readonly session_appendTokens: (a: number, b: number, c: number, d: number) => void;
    readonly session_attachLora: (a: number, b: number, c: number) => void;
    readonly session_cancel: (a: number) => void;
    readonly session_capabilities: (a: number) => number;
    readonly session_clearCancel: (a: number) => void;
    readonly session_generate: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly session_hasLora: (a: number) => number;
    readonly session_hiddenSize: (a: number) => number;
    readonly session_hiddenStatesForText: (a: number, b: number, c: number, d: number) => void;
    readonly session_hiddenStatesForTokens: (a: number, b: number, c: number, d: number) => void;
    readonly session_hiddenStatesMeanPooled: (a: number, b: number, c: number, d: number) => void;
    readonly session_position: (a: number) => number;
    readonly session_removeLora: (a: number) => void;
    readonly session_reset: (a: number, b: number) => void;
    readonly session_setImageMaxLongSize: (a: number, b: number) => void;
    readonly sessionconfig_kvCompression: (a: number) => number;
    readonly sessionconfig_maxSeqLen: (a: number) => number;
    readonly sessionconfig_nKeep: (a: number) => number;
    readonly sessionconfig_new: () => number;
    readonly sessionconfig_seed: (a: number, b: number) => void;
    readonly sessionconfig_set_kvCompression: (a: number, b: number) => void;
    readonly sessionconfig_set_maxSeqLen: (a: number, b: number) => void;
    readonly sessionconfig_set_nKeep: (a: number, b: number) => void;
    readonly sessionconfig_set_seed: (a: number, b: number, c: bigint) => void;
    readonly tokenizer_addBosToken: (a: number) => number;
    readonly tokenizer_applyChatTemplate: (a: number, b: number, c: number, d: number) => void;
    readonly tokenizer_applyChatTemplateWithTools: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly tokenizer_bosToken: (a: number) => number;
    readonly tokenizer_chatTemplate: (a: number, b: number) => void;
    readonly tokenizer_decode: (a: number, b: number, c: number, d: number) => void;
    readonly tokenizer_encode: (a: number, b: number, c: number, d: number) => void;
    readonly tokenizer_encodeSpecial: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly tokenizer_eosToken: (a: number) => number;
    readonly tokenizer_isSpecialToken: (a: number, b: number) => number;
    readonly tokenizer_specialTokenId: (a: number, b: number, c: number) => number;
    readonly tokenizer_vocabSize: (a: number) => number;
    readonly toolGrammar: (a: number, b: number, c: number, d: number) => void;
    readonly turboquantconfig_keys: (a: number) => number;
    readonly turboquantconfig_new: (a: bigint) => number;
    readonly turboquantconfig_seed: (a: number) => bigint;
    readonly turboquantconfig_set_keys: (a: number, b: number) => void;
    readonly turboquantconfig_set_seed: (a: number, b: bigint) => void;
    readonly turboquantconfig_set_values: (a: number, b: number) => void;
    readonly turboquantconfig_values: (a: number) => number;
    readonly sessionconfig_set_ubatchSize: (a: number, b: number) => void;
    readonly sessionconfig_ubatchSize: (a: number) => number;
    readonly __wbg_wbg_rayon_poolbuilder_free: (a: number, b: number) => void;
    readonly initThreadPool: (a: number) => number;
    readonly wbg_rayon_poolbuilder_build: (a: number) => void;
    readonly wbg_rayon_poolbuilder_numThreads: (a: number) => number;
    readonly wbg_rayon_poolbuilder_receiver: (a: number) => number;
    readonly wbg_rayon_start_worker: (a: number) => void;
    readonly __wasm_bindgen_func_elem_5375: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_5392: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_230: (a: number, b: number, c: number) => void;
    readonly __wasm_bindgen_func_elem_5377: (a: number, b: number, c: number) => void;
    readonly memory: WebAssembly.Memory;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export5: (a: number, b: number, c: number) => void;
    readonly __wbindgen_thread_destroy: (a?: number, b?: number, c?: number) => void;
    readonly __wbindgen_start: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number }} module - Passing `SyncInitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number } | SyncInitInput, memory?: WebAssembly.Memory): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number }} module_or_path - Passing `InitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number } | InitInput | Promise<InitInput>, memory?: WebAssembly.Memory): Promise<InitOutput>;

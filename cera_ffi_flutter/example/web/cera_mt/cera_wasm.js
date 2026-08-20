/* @ts-self-types="./cera_wasm.d.ts" */
import { startWorkers } from './snippets/wasm-bindgen-rayon-38edf6e439f6d70d/src/workerHelpers.js';

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BundleRepoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bundlerepo_free(ptr, 0);
    }
    /**
     * Cached bytes for `url`, downloading first if needed.
     *
     * **Copies the whole file into a JS `Uint8Array`**, so peak memory
     * is roughly twice the file. That's fine for a manifest and wrong
     * for a model: to load a model, use `CeraEngine.fromBundleId` or
     * `fromManifestUrl`, which keep the bytes inside wasm and hand
     * them straight to the engine.
     * @param {string} url
     * @param {string | null} [expected_sha256]
     * @param {Function | null} [on_progress]
     * @returns {Promise<Uint8Array>}
     */
    bytes(url, expected_sha256, on_progress) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(expected_sha256) ? 0 : passStringToWasm0(expected_sha256, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.bundlerepo_bytes(this.__wbg_ptr, ptr0, len0, ptr1, len1, isLikeNone(on_progress) ? 0 : addHeapObject(on_progress));
        return takeObject(ret);
    }
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
     * @returns {Promise<number>}
     */
    cacheSize() {
        const ret = wasm.bundlerepo_cacheSize(this.__wbg_ptr);
        return takeObject(ret);
    }
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
     * @returns {Promise<void>}
     */
    clearCache() {
        const ret = wasm.bundlerepo_clearCache(this.__wbg_ptr);
        return takeObject(ret);
    }
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
     * @param {string} url
     * @param {string | null} [expected_sha256]
     * @param {Function | null} [on_progress]
     * @returns {Promise<void>}
     */
    download(url, expected_sha256, on_progress) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(expected_sha256) ? 0 : passStringToWasm0(expected_sha256, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.bundlerepo_download(this.__wbg_ptr, ptr0, len0, ptr1, len1, isLikeNone(on_progress) ? 0 : addHeapObject(on_progress));
        return takeObject(ret);
    }
    /**
     * Whether `url` is present in the cache. Existence only: this does
     * not verify the hash or size, so a truthy answer means "a
     * download landed here", not "the bytes are known good".
     * @param {string} url
     * @returns {Promise<boolean>}
     */
    isCached(url) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bundlerepo_isCached(this.__wbg_ptr, ptr0, len0);
        return takeObject(ret);
    }
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
     * @param {string | null} [store_dir]
     */
    constructor(store_dir) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            var ptr0 = isLikeNone(store_dir) ? 0 : passStringToWasm0(store_dir, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            var len0 = WASM_VECTOR_LEN;
            wasm.bundlerepo_new(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            BundleRepoFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Drop the cache entry for one URL, leaving the rest intact.
     * Returns whether anything was removed. Removes the `.sha256`
     * sidecar alongside the payload, so a later re-download can't
     * match a stale hash.
     * @param {string} url
     * @returns {Promise<boolean>}
     */
    remove(url) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bundlerepo_remove(this.__wbg_ptr, ptr0, len0);
        return takeObject(ret);
    }
    /**
     * The OPFS directory this repo caches under. Matches what was
     * passed to the constructor.
     * @returns {string}
     */
    get storeDir() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.bundlerepo_storeDir(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Cached text for `url` (UTF-8), downloading first if needed.
     * Intended for manifests; throws if the bytes aren't valid UTF-8.
     * @param {string} url
     * @returns {Promise<string>}
     */
    text(url) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bundlerepo_text(this.__wbg_ptr, ptr0, len0);
        return takeObject(ret);
    }
}
if (Symbol.dispose) BundleRepo.prototype[Symbol.dispose] = BundleRepo.prototype.free;

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(CeraEngine.prototype);
        obj.__wbg_ptr = ptr;
        CeraEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CeraEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ceraengine_free(ptr, 0);
    }
    /**
     * `true` when the GGUF declares `tokenizer.ggml.add_bos_token`.
     * Callers that hand-build a token sequence from `Tokenizer.encode`
     * should prepend `Tokenizer.bosToken` when this is `true` (and
     * the model has a BOS) — cera's encoder returns the raw tokens
     * without that prefix.
     * @returns {boolean}
     */
    get addBosToken() {
        const ret = wasm.ceraengine_addBosToken(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * `true` when the GGUF declares `tokenizer.ggml.add_eos_token`. Prefer
     * `Tokenizer.encodeSpecial`, which applies this (and BOS) automatically.
     * @returns {boolean}
     */
    get addEosToken() {
        const ret = wasm.ceraengine_addEosToken(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Model architecture string from the GGUF metadata
     * (e.g. `"lfm2"`, `"llama"`).
     * @returns {string}
     */
    get architecture() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.ceraengine_architecture(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
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
     * @returns {Capabilities}
     */
    get capabilities() {
        const ret = wasm.ceraengine_capabilities(this.__wbg_ptr);
        return takeObject(ret);
    }
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
     * @returns {number}
     */
    get contextSize() {
        const ret = wasm.ceraengine_contextSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Constructs a `GenerateOpts` seeded with the advisory defaults from the
     * model manifest (if any), falling back to standard defaults for unmentioned fields.
     * @returns {GenerateOpts}
     */
    defaultGenerateOpts() {
        const ret = wasm.ceraengine_defaultGenerateOpts(this.__wbg_ptr);
        return GenerateOpts.__wrap(ret);
    }
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
     * @param {BundleRepo} repo
     * @param {string} bundle_id
     * @param {string} quant
     * @param {number | null} [context_size]
     * @param {Function | null} [on_progress]
     * @returns {Promise<CeraEngine>}
     */
    static fromBundleId(repo, bundle_id, quant, context_size, on_progress) {
        _assertClass(repo, BundleRepo);
        const ptr0 = passStringToWasm0(bundle_id, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(quant, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.ceraengine_fromBundleId(repo.__wbg_ptr, ptr0, len0, ptr1, len1, isLikeNone(context_size) ? 0x100000001 : (context_size) >>> 0, isLikeNone(on_progress) ? 0 : addHeapObject(on_progress));
        return takeObject(ret);
    }
    /**
     * Load a model from in-memory GGUF bytes. `contextSize` defaults
     * to 4096 if omitted; the actual KV-cache cap is the smaller of
     * the requested size and the model's own `max_seq_len`.
     *
     * The backend is forced to CPU — wasm has no native GPU/Metal
     * backend. Throws on parse failure, unsupported quantization,
     * or unrecognized architecture.
     * @param {Uint8Array} bytes
     * @param {number | null} [context_size]
     * @returns {CeraEngine}
     */
    static fromGgufBytes(bytes, context_size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.ceraengine_fromGgufBytes(retptr, ptr0, len0, isLikeNone(context_size) ? 0x100000001 : (context_size) >>> 0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return CeraEngine.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {Uint8Array} bytes
     * @param {Uint8Array | null} [mmproj]
     * @param {number | null} [context_size]
     * @param {string | null} [inference_type]
     * @returns {CeraEngine}
     */
    static fromGgufParts(bytes, mmproj, context_size, inference_type) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            var ptr1 = isLikeNone(mmproj) ? 0 : passArray8ToWasm0(mmproj, wasm.__wbindgen_export);
            var len1 = WASM_VECTOR_LEN;
            var ptr2 = isLikeNone(inference_type) ? 0 : passStringToWasm0(inference_type, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            var len2 = WASM_VECTOR_LEN;
            wasm.ceraengine_fromGgufParts(retptr, ptr0, len0, ptr1, len1, isLikeNone(context_size) ? 0x100000001 : (context_size) >>> 0, ptr2, len2);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return CeraEngine.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Load a bundle from the URL of its manifest JSON, for bundles
     * hosted somewhere other than `LiquidAI/LeapBundles`.
     *
     * Files the manifest names are fetched relative to it. Entries
     * with a nested path are refused rather than guessed at: see
     * `bundle::join_url`.
     * @param {BundleRepo} repo
     * @param {string} manifest_url
     * @param {number | null} [context_size]
     * @param {Function | null} [on_progress]
     * @returns {Promise<CeraEngine>}
     */
    static fromManifestUrl(repo, manifest_url, context_size, on_progress) {
        _assertClass(repo, BundleRepo);
        const ptr0 = passStringToWasm0(manifest_url, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ceraengine_fromManifestUrl(repo.__wbg_ptr, ptr0, len0, isLikeNone(context_size) ? 0x100000001 : (context_size) >>> 0, isLikeNone(on_progress) ? 0 : addHeapObject(on_progress));
        return takeObject(ret);
    }
    /**
     * `true` when the loaded GGUF carries an embedded Jinja chat
     * template. JS callers can use this to decide whether to render
     * `Tokenizer.chatTemplate` themselves vs falling back to a
     * hard-coded prompt format.
     * @returns {boolean}
     */
    get hasChatTemplate() {
        const ret = wasm.ceraengine_hasChatTemplate(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Maximum sequence length the model was trained for. Independent
     * of the engine's `contextSize` config — that one is the KV
     * cache cap, this is the model's positional encoding ceiling.
     * @returns {number}
     */
    get maxSeqLen() {
        const ret = wasm.ceraengine_maxSeqLen(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Everything `CeraEngine`'s individual metadata getters report, in one
     * object. See the `ModelMetadata` interface in the generated `.d.ts`.
     * @returns {ModelMetadata}
     */
    get metadata() {
        const ret = wasm.ceraengine_metadata(this.__wbg_ptr);
        return takeObject(ret);
    }
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
     * @param {SessionConfig} config
     * @returns {Session}
     */
    newSession(config) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(config, SessionConfig);
            wasm.ceraengine_newSession(retptr, this.__wbg_ptr, config.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return Session.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Quantization label from the GGUF (e.g. `"Q4_0"`, `"Q4_K_M"`).
     * Useful for telling users what they actually loaded when the
     * download URL doesn't make it obvious.
     * @returns {string}
     */
    get quantization() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.ceraengine_quantization(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Returns a `Tokenizer` handle bound to this engine's vocab.
     * Each call allocates a fresh JS object but the underlying
     * tokenizer state is shared via `Arc` — cheap to call, JS
     * callers can cache the result if they prefer one handle.
     * @returns {Tokenizer}
     */
    get tokenizer() {
        const ret = wasm.ceraengine_tokenizer(this.__wbg_ptr);
        return Tokenizer.__wrap(ret);
    }
    /**
     * The token id of `format`'s tool-call start marker (e.g.
     * `<|tool_call_start|>`) in this model's vocab, for use as a lazy
     * grammar trigger in `GenerateOpts.grammarTriggerTokens`.
     * `undefined` when this tokenizer lacks that special token.
     * @param {ToolFormat} format
     * @returns {number | undefined}
     */
    toolCallStartToken(format) {
        const ret = wasm.ceraengine_toolCallStartToken(this.__wbg_ptr, format);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * The tool-call format auto-detected from this model's architecture, or
     * `undefined` when the architecture has no known tool convention.
     *
     * Engine-level counterpart to the free `detectToolFormat(architecture)`
     * function: this one already knows the loaded model's architecture, so
     * it cannot disagree with it.
     * @returns {ToolFormat | undefined}
     */
    toolFormat() {
        const ret = wasm.ceraengine_toolFormat(this.__wbg_ptr);
        return ret === 2 ? undefined : ret;
    }
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
     * @param {Float32Array} pcm
     * @param {number} sample_rate
     * @returns {string}
     */
    transcribe(pcm, sample_rate) {
        let deferred3_0;
        let deferred3_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(pcm, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.ceraengine_transcribe(retptr, this.__wbg_ptr, ptr0, len0, sample_rate);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr2 = r0;
            var len2 = r1;
            if (r3) {
                ptr2 = 0; len2 = 0;
                throw takeObject(r2);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get vocabSize() {
        const ret = wasm.ceraengine_vocabSize(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) CeraEngine.prototype[Symbol.dispose] = CeraEngine.prototype.free;

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(GenerateOpts.prototype);
        obj.__wbg_ptr = ptr;
        GenerateOptsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GenerateOptsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_generateopts_free(ptr, 0);
    }
    /**
     * Remove any grammar constraint, returning to unconstrained decoding.
     */
    clearGrammar() {
        wasm.generateopts_clearGrammar(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    get flushEveryMs() {
        const ret = wasm.generateopts_flushEveryMs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get flushEveryTokens() {
        const ret = wasm.generateopts_flushEveryTokens(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Lazy-grammar trigger token IDs (tool calling). When non-empty and a
     * grammar is set (`GenerateOpts.setGrammar`), the grammar stays inactive until
     * the model emits one of these tokens (e.g. the tool-call start marker
     * from `Tokenizer.specialTokenId`), then constrains the call and
     * deactivates on completion. Empty (default) → the grammar is active from
     * the first token.
     * @returns {Uint32Array}
     */
    get grammarTriggerTokens() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.generateopts_grammarTriggerTokens(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export5(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Whether a grammar constraint is currently set.
     * @returns {boolean}
     */
    get hasGrammar() {
        const ret = wasm.generateopts_hasGrammar(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Ignore end-of-generation: EOS and `stopTokens` are not honored, so
     * decode always runs to `maxTokens`. For benchmark loops that must
     * cover an exact token count. `false` by default.
     * @returns {boolean}
     */
    get ignoreEos() {
        const ret = wasm.generateopts_ignoreEos(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    get maxTokens() {
        const ret = wasm.generateopts_maxTokens(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Min-p (relative) nucleus cutoff: drop tokens below `minP * pMax`.
     * `0.0` (default) disables it. Honored in the stochastic path.
     * @returns {number}
     */
    get minP() {
        const ret = wasm.generateopts_minP(this.__wbg_ptr);
        return ret;
    }
    constructor() {
        const ret = wasm.generateopts_new();
        this.__wbg_ptr = ret >>> 0;
        GenerateOptsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Repetition penalty over tokens generated this call. `1.0` (default)
     * disables it. Honored in the stochastic path.
     * @returns {number}
     */
    get repetitionPenalty() {
        const ret = wasm.generateopts_repetitionPenalty(this.__wbg_ptr);
        return ret;
    }
    /**
     * Constrain decoding to a GBNF grammar (source text, e.g. a JSON grammar).
     * Each step masks the logits so only tokens the grammar accepts are
     * sampled. Throws a `JsError` if the grammar fails to compile; replaces any
     * grammar set by a prior call. A setter can't surface the parse error, so
     * this is a method rather than a `grammar` property.
     * @param {string} gbnf
     */
    setGrammar(gbnf) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(gbnf, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.generateopts_setGrammar(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} v
     */
    set flushEveryMs(v) {
        wasm.generateopts_set_flushEveryMs(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set flushEveryTokens(v) {
        wasm.generateopts_set_flushEveryTokens(this.__wbg_ptr, v);
    }
    /**
     * @param {Uint32Array} v
     */
    set grammarTriggerTokens(v) {
        const ptr0 = passArray32ToWasm0(v, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        wasm.generateopts_set_grammarTriggerTokens(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {boolean} v
     */
    set ignoreEos(v) {
        wasm.generateopts_set_ignoreEos(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set maxTokens(v) {
        wasm.generateopts_set_maxTokens(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set minP(v) {
        wasm.generateopts_set_minP(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set repetitionPenalty(v) {
        wasm.generateopts_set_repetitionPenalty(this.__wbg_ptr, v);
    }
    /**
     * @param {Uint32Array} v
     */
    set stopTokens(v) {
        const ptr0 = passArray32ToWasm0(v, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        wasm.generateopts_set_stopTokens(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} v
     */
    set temperature(v) {
        wasm.generateopts_set_temperature(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set topK(v) {
        wasm.generateopts_set_topK(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set topP(v) {
        wasm.generateopts_set_topP(this.__wbg_ptr, v);
    }
    /**
     * Token IDs that, if produced, end decoding with
     * `finishReason = "Stop"`. Empty by default.
     * @returns {Uint32Array}
     */
    get stopTokens() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.generateopts_stopTokens(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export5(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number}
     */
    get temperature() {
        const ret = wasm.generateopts_temperature(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get topK() {
        const ret = wasm.generateopts_topK(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get topP() {
        const ret = wasm.generateopts_topP(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) GenerateOpts.prototype[Symbol.dispose] = GenerateOpts.prototype.free;

/**
 * Summary returned from a completed `Session.generate` call.
 */
export class GenerateSummary {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(GenerateSummary.prototype);
        obj.__wbg_ptr = ptr;
        GenerateSummaryFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GenerateSummaryFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_generatesummary_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get decodeMs() {
        const ret = wasm.generatesummary_decodeMs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Why decode ended. One of `"MaxTokens"`, `"Stop"`,
     * `"Cancelled"`, `"ContextFull"`, or `"Error(<message>)"` —
     * the `Error(...)` form preserves the inner string verbatim
     * (no surrounding quotes), so JS callers can log it directly.
     * @returns {string}
     */
    get finishReason() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.generatesummary_finishReason(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {number}
     */
    get promptEvalMs() {
        const ret = wasm.generatesummary_promptEvalMs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get promptEvalTokens() {
        const ret = wasm.generatesummary_promptEvalTokens(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get tokensGenerated() {
        const ret = wasm.generatesummary_tokensGenerated(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) GenerateSummary.prototype[Symbol.dispose] = GenerateSummary.prototype.free;

/**
 * A loaded LoRA adapter, ready to attach to a [`Session`] via `attachLora`.
 * Load it once (from bytes — the browser has no filesystem) and reuse the
 * handle across sessions; the factors are reference-counted internally.
 */
export class LoraAdapters {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(LoraAdapters.prototype);
        obj.__wbg_ptr = ptr;
        LoraAdaptersFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LoraAdaptersFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_loraadapters_free(ptr, 0);
    }
    /**
     * Load a llama.cpp-format GGUF adapter (`convert_lora_to_gguf` output) from
     * bytes. `alpha` is read from the adapter's `adapter.lora.alpha` metadata.
     * @param {Uint8Array} bytes
     * @returns {LoraAdapters}
     */
    static fromGgufBytes(bytes) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.loraadapters_fromGgufBytes(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return LoraAdapters.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Load a PEFT `.safetensors` adapter from bytes. PEFT keeps `alpha` in a
     * sibling `adapter_config.json`, so pass it explicitly (`undefined` ⇒
     * scale = 1, i.e. `alpha == rank`).
     * @param {Uint8Array} bytes
     * @param {number | null} [alpha]
     * @returns {LoraAdapters}
     */
    static fromSafetensorsBytes(bytes, alpha) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.loraadapters_fromSafetensorsBytes(retptr, ptr0, len0, isLikeNone(alpha) ? 0x100000001 : Math.fround(alpha));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return LoraAdapters.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Number of `(layer, target)` low-rank deltas the adapter carries.
     * @returns {number}
     */
    targetCount() {
        const ret = wasm.loraadapters_targetCount(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) LoraAdapters.prototype[Symbol.dispose] = LoraAdapters.prototype.free;

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Manifest.prototype);
        obj.__wbg_ptr = ptr;
        ManifestFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ManifestFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_manifest_free(ptr, 0);
    }
    /**
     * URL of the audio-decoder GGUF for audio-out models.
     * @returns {string | undefined}
     */
    get audioDecoderUrl() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_audioDecoderUrl(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export5(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * URL of the audio-tokenizer checkpoint (typically `.safetensors`).
     * @returns {string | undefined}
     */
    get audioTokenizerUrl() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_audioTokenizerUrl(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export5(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Jinja chat template override from the manifest, if present.
     * `undefined` means "use the template embedded in the GGUF
     * metadata" (cera's standard fallback).
     * @returns {string | undefined}
     */
    get chatTemplate() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_chatTemplate(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export5(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Raw `inference_type` string (e.g. `llama.cpp/text-to-text`).
     * Round-trips through cera's enum, so unknown variants come back
     * as their original string — no information loss.
     * @returns {string}
     */
    get inferenceType() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_inferenceType(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * URL (or local path string) for the primary model GGUF.
     * @returns {string}
     */
    get modelUrl() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_modelUrl(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * URL of the multimodal projector GGUF if the manifest declares
     * one (VL / audio models). `undefined` for plain text models.
     * @returns {string | undefined}
     */
    get multimodalProjectorUrl() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_multimodalProjectorUrl(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export5(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Parse a JSON manifest from raw bytes. Throws a `JsError` on
     * malformed JSON or when required fields are missing or wrongly
     * typed (e.g. no `load_time_parameters.model`). Unknown
     * `inference_type` values are **not** an error — they round-trip
     * through `cera::manifest::InferenceType::Unknown(String)` and
     * surface verbatim via the `inferenceType` getter, so JS callers
     * can decide how to react instead of catching here.
     * @param {Uint8Array} json_bytes
     * @returns {Manifest}
     */
    static parse(json_bytes) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(json_bytes, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.manifest_parse(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return Manifest.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {string}
     */
    get schemaVersion() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.manifest_schemaVersion(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) Manifest.prototype[Symbol.dispose] = Manifest.prototype.free;

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Session.prototype);
        obj.__wbg_ptr = ptr;
        SessionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SessionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_session_free(ptr, 0);
    }
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
     * @param {Float32Array} samples
     * @param {number} sample_rate
     */
    appendAudio(samples, sample_rate) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(samples, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_appendAudio(retptr, this.__wbg_ptr, ptr0, len0, sample_rate);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {Uint8Array} bytes
     * @param {number | null} [max_long_size]
     */
    appendImage(bytes, max_long_size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_appendImage(retptr, this.__wbg_ptr, ptr0, len0, isLikeNone(max_long_size) ? 0x100000001 : (max_long_size) >>> 0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Tokenize `text` using the session's tokenizer and append the
     * result to the KV cache. Equivalent to
     * `appendTokens(tokenizer.encode(text))` but avoids the round
     * trip through JS for the encoded buffer.
     * @param {string} text
     */
    appendText(text) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(text, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_appendText(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Append already-tokenized IDs to the KV cache. Use when you
     * need control over BOS/EOS framing or you've cached tokens
     * from a previous encode.
     * @param {Uint32Array} tokens
     */
    appendTokens(tokens) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(tokens, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_appendTokens(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Attach a [`LoraAdapters`] to this session. Applied to every subsequent
     * forward pass — generation **and** hidden-states extraction — until
     * removed or replaced (hot-swap), and preserved across `reset()`. Throws if
     * the adapter's dimensions don't match the loaded model. Only affects tokens
     * processed after the call (doesn't retroactively re-adapt cached KV).
     * @param {LoraAdapters} adapters
     */
    attachLora(adapters) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(adapters, LoraAdapters);
            wasm.session_attachLora(retptr, this.__wbg_ptr, adapters.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Flip the cancel atomic, requesting that any in-flight
     * `generate` call exit at its next checkpoint with
     * `finishReason = "Cancelled"`. Safe to call from any thread
     * (including a Worker that owns this session — though wasm
     * without SharedArrayBuffer makes cross-thread sharing
     * unusual).
     */
    cancel() {
        wasm.session_cancel(this.__wbg_ptr);
    }
    /**
     * Modality capability flags reported by the model backing
     * this session. Same shape as `CeraEngine.capabilities` —
     * see that getter for the `Capabilities` field documentation
     * and the synthetic-text caveat that applies to all
     * `fromGgufBytes`-loaded models today.
     * @returns {Capabilities}
     */
    get capabilities() {
        const ret = wasm.session_capabilities(this.__wbg_ptr);
        return takeObject(ret);
    }
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
    clearCancel() {
        wasm.session_clearCancel(this.__wbg_ptr);
    }
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
     * @param {GenerateOpts} opts
     * @param {Function} on_text_tokens
     * @param {Function | null} [on_audio_frames]
     * @returns {GenerateSummary}
     */
    generate(opts, on_text_tokens, on_audio_frames) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(opts, GenerateOpts);
            wasm.session_generate(retptr, this.__wbg_ptr, opts.__wbg_ptr, addBorrowedObject(on_text_tokens), isLikeNone(on_audio_frames) ? 0 : addHeapObject(on_audio_frames));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return GenerateSummary.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            heap[stack_pointer++] = undefined;
        }
    }
    /**
     * Whether a LoRA adapter is currently attached to this session.
     * @returns {boolean}
     */
    hasLora() {
        const ret = wasm.session_hasLora(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Model hidden dimension `D` — reshape a `[T*D]` hidden-states buffer into
     * `[T][D]` with this. Reads a cached field (set at construction), so — unlike
     * the `&mut self` compute methods — it's safe to call from inside a `generate`
     * callback without a wasm-bindgen borrow panic.
     * @returns {number}
     */
    hiddenSize() {
        const ret = wasm.session_hiddenSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Tokenize `text` and return its per-token hidden states as a `Float32Array`.
     * @param {string} text
     * @returns {Float32Array}
     */
    hiddenStatesForText(text) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(text, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_hiddenStatesForText(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Per-token last-layer hidden states (post-final-RMSNorm — the llama.cpp
     * `--pooling none` vector) for `tokens`, as a `Float32Array` of length
     * `tokens.length * hiddenSize` (row-major; token `t` channel `c` at
     * `t*hiddenSize + c`). The wasm boundary copies the buffer into the JS heap
     * once. Side-effect-free — does not disturb the generation KV.
     * @param {Uint32Array} tokens
     * @returns {Float32Array}
     */
    hiddenStatesForTokens(tokens) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(tokens, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_hiddenStatesForTokens(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Mean-pooled hidden state — a single `Float32Array` of length `hiddenSize`
     * (the common classifier path: pool in Rust, ship `D` floats not `T*D`).
     * @param {Uint32Array} tokens
     * @returns {Float32Array}
     */
    hiddenStatesMeanPooled(tokens) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(tokens, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.session_hiddenStatesMeanPooled(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Current KV cache position (number of tokens currently held).
     * @returns {number}
     */
    get position() {
        const ret = wasm.session_position(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Remove any attached LoRA adapter, returning to base-model inference.
     */
    removeLora() {
        wasm.session_removeLora(this.__wbg_ptr);
    }
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
    reset() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.session_reset(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Set the session-default cap on the longest side of an appended
     * image, in pixels. `null` clears it (no cap).
     *
     * Applies to later `appendImage` calls that pass no explicit
     * `maxLongSize`. A per-call value always wins.
     * @param {number | null} [max_long_size]
     */
    setImageMaxLongSize(max_long_size) {
        wasm.session_setImageMaxLongSize(this.__wbg_ptr, isLikeNone(max_long_size) ? 0x100000001 : (max_long_size) >>> 0);
    }
}
if (Symbol.dispose) Session.prototype[Symbol.dispose] = Session.prototype.free;

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SessionConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sessionconfig_free(ptr, 0);
    }
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
     * @returns {TurboQuantConfig | undefined}
     */
    get kvCompression() {
        const ret = wasm.sessionconfig_kvCompression(this.__wbg_ptr);
        return ret === 0 ? undefined : TurboQuantConfig.__wrap(ret);
    }
    /**
     * Cap on total tokens held in KV. `null` (the common case)
     * defers to the engine's effective max — i.e.
     * `min(engine.contextSize, model.maxSeqLen)`. Set to a
     * smaller value here to further lower the cap; values larger
     * than the engine's effective max are still capped at it.
     * @returns {number | undefined}
     */
    get maxSeqLen() {
        const ret = wasm.sessionconfig_maxSeqLen(this.__wbg_ptr);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * Number of leading tokens pinned in KV across context shifts —
     * a system prompt or persistent prefix that should survive
     * when the cache fills. `0` (default) disables the pin.
     * @returns {number}
     */
    get nKeep() {
        const ret = wasm.sessionconfig_nKeep(this.__wbg_ptr);
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.sessionconfig_new();
        this.__wbg_ptr = ret >>> 0;
        SessionConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Deterministic sampler seed. `null` (default) uses a fresh
     * random seed per session — set this to make a session's
     * outputs reproducible across runs (useful for testing /
     * demos / regression checks).
     * @returns {bigint | undefined}
     */
    get seed() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.sessionconfig_seed(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getBigInt64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : BigInt.asUintN(64, r2);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {TurboQuantConfig | null} [v]
     */
    set kvCompression(v) {
        let ptr0 = 0;
        if (!isLikeNone(v)) {
            _assertClass(v, TurboQuantConfig);
            ptr0 = v.__destroy_into_raw();
        }
        wasm.sessionconfig_set_kvCompression(this.__wbg_ptr, ptr0);
    }
    /**
     * @param {number | null} [v]
     */
    set maxSeqLen(v) {
        wasm.sessionconfig_set_maxSeqLen(this.__wbg_ptr, isLikeNone(v) ? 0x100000001 : (v) >>> 0);
    }
    /**
     * @param {number} v
     */
    set nKeep(v) {
        wasm.sessionconfig_set_nKeep(this.__wbg_ptr, v);
    }
    /**
     * @param {bigint | null} [v]
     */
    set seed(v) {
        wasm.sessionconfig_set_seed(this.__wbg_ptr, !isLikeNone(v), isLikeNone(v) ? BigInt(0) : v);
    }
    /**
     * @param {number} v
     */
    set ubatchSize(v) {
        wasm.sessionconfig_set_ubatchSize(this.__wbg_ptr, v);
    }
    /**
     * Chunked-prefill batch size (tokens per micro-batch during
     * the prefill pass). Smaller values give finer-grained
     * `Session.cancel()` checkpoints during long prompt eval at
     * some perf cost. cera's default is `512`.
     * @returns {number}
     */
    get ubatchSize() {
        const ret = wasm.sessionconfig_ubatchSize(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) SessionConfig.prototype[Symbol.dispose] = SessionConfig.prototype.free;

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Tokenizer.prototype);
        obj.__wbg_ptr = ptr;
        TokenizerFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TokenizerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tokenizer_free(ptr, 0);
    }
    /**
     * Whether the GGUF asks for a BOS token to be prepended
     * (`tokenizer.ggml.add_bos_token`).
     *
     * Needed to frame a prompt correctly without `encodeSpecial`: that
     * helper prepends BOS *and* appends EOS together, and a chat-template
     * prompt wants the first and not the second. Callers doing their own
     * framing read this and prepend `bosToken` themselves.
     * @returns {boolean}
     */
    get addBosToken() {
        const ret = wasm.tokenizer_addBosToken(this.__wbg_ptr);
        return ret !== 0;
    }
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
     * @param {ChatMessage[]} messages
     * @param {boolean | null} [add_generation_prompt]
     * @returns {string}
     */
    applyChatTemplate(messages, add_generation_prompt) {
        let deferred2_0;
        let deferred2_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.tokenizer_applyChatTemplate(retptr, this.__wbg_ptr, addHeapObject(messages), isLikeNone(add_generation_prompt) ? 0xFFFFFF : add_generation_prompt ? 1 : 0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr1 = r0;
            var len1 = r1;
            if (r3) {
                ptr1 = 0; len1 = 0;
                throw takeObject(r2);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Like `applyChatTemplate`, but also injects a `tools` array so a
     * tool-trained model renders its tool-definition block. `toolsJson` is a
     * JSON string encoding an array of `ToolDef` (`[{name, description?,
     * parameters?}]`). Throws on invalid `toolsJson` or a render failure.
     * @param {ChatMessage[]} messages
     * @param {string} tools_json
     * @param {boolean | null} [add_generation_prompt]
     * @returns {string}
     */
    applyChatTemplateWithTools(messages, tools_json, add_generation_prompt) {
        let deferred3_0;
        let deferred3_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(tools_json, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.tokenizer_applyChatTemplateWithTools(retptr, this.__wbg_ptr, addHeapObject(messages), ptr0, len0, isLikeNone(add_generation_prompt) ? 0xFFFFFF : add_generation_prompt ? 1 : 0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            var ptr2 = r0;
            var len2 = r1;
            if (r3) {
                ptr2 = 0; len2 = 0;
                throw takeObject(r2);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * BOS token ID, if the GGUF metadata declares one.
     * @returns {number | undefined}
     */
    get bosToken() {
        const ret = wasm.tokenizer_bosToken(this.__wbg_ptr);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * Raw embedded Jinja chat template from the GGUF metadata, if
     * any. Most callers should use [`Self::apply_chat_template`]
     * (`applyChatTemplate` in JS) instead — this getter is for
     * inspection or for callers who want to render with a
     * different Jinja runtime.
     * @returns {string | undefined}
     */
    get chatTemplate() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.tokenizer_chatTemplate(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export5(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Detokenize back to a UTF-8 string. Lossy for tokens whose
     * byte sequences don't decode to valid UTF-8 — those are
     * replaced with U+FFFD per `String::from_utf8_lossy`.
     * @param {Uint32Array} tokens
     * @returns {string}
     */
    decode(tokens) {
        let deferred2_0;
        let deferred2_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(tokens, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.tokenizer_decode(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred2_0 = r0;
            deferred2_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export5(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Tokenize a UTF-8 string. Returns the token IDs as a
     * `Uint32Array`. No BOS/EOS prefix — callers that want them
     * should prepend `bosToken` / append `eosToken` manually, or use
     * `encodeSpecial`.
     * @param {string} text
     * @returns {Uint32Array}
     */
    encode(text) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(text, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.tokenizer_encode(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export5(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Encode with optional special markers — the analog of llama.cpp's
     * `llama_tokenize(..., add_special)`. When `addSpecial` is true, BOS is
     * prepended iff the GGUF declares `tokenizer.ggml.add_bos_token` and EOS
     * appended iff it declares `tokenizer.ggml.add_eos_token`, so token counts
     * match llama.cpp. With `addSpecial = false` this is exactly `encode`.
     * @param {string} text
     * @param {boolean} add_special
     * @returns {Uint32Array}
     */
    encodeSpecial(text, add_special) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(text, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.tokenizer_encodeSpecial(retptr, this.__wbg_ptr, ptr0, len0, add_special);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export5(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * EOS token ID, if the GGUF metadata declares one.
     * @returns {number | undefined}
     */
    get eosToken() {
        const ret = wasm.tokenizer_eosToken(this.__wbg_ptr);
        return ret === 0x100000001 ? undefined : ret;
    }
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
     * @param {number} id
     * @returns {boolean}
     */
    isSpecialToken(id) {
        const ret = wasm.tokenizer_isSpecialToken(this.__wbg_ptr, id);
        return ret !== 0;
    }
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
     * @param {string} name
     * @returns {number | undefined}
     */
    specialTokenId(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.tokenizer_specialTokenId(this.__wbg_ptr, ptr0, len0);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * @returns {number}
     */
    get vocabSize() {
        const ret = wasm.tokenizer_vocabSize(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) Tokenizer.prototype[Symbol.dispose] = Tokenizer.prototype.free;

/**
 * The tool-call wire format a model family uses. Get one from
 * `detectToolFormat(architecture)` or choose explicitly.
 * @enum {0 | 1}
 */
export const ToolFormat = Object.freeze({
    /**
     * LFM2 / LFM2.5: Pythonic `[get_weather(city="Paris")]`.
     */
    Lfm2Pythonic: 0, "0": "Lfm2Pythonic",
    /**
     * Hermes / Qwen: JSON `{"name":…,"arguments":{…}}`.
     */
    Hermes: 1, "1": "Hermes",
});

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(TurboQuantConfig.prototype);
        obj.__wbg_ptr = ptr;
        TurboQuantConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TurboQuantConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_turboquantconfig_free(ptr, 0);
    }
    /**
     * Compress the K side of the KV cache. Default `true`.
     * Useful to flip off when debugging quality regressions to
     * isolate K-side vs V-side contribution.
     * @returns {boolean}
     */
    get keys() {
        const ret = wasm.turboquantconfig_keys(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Construct with the common production setup: both keys and
     * values compressed. Pass an explicit `seed` so the per-layer
     * rotations are reproducible.
     * @param {bigint} seed
     */
    constructor(seed) {
        const ret = wasm.turboquantconfig_new(seed);
        this.__wbg_ptr = ret >>> 0;
        TurboQuantConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Hadamard-rotation seed. Same seed → same rotations →
     * reproducible KV cache contents (necessary for bitwise-
     * identical replay across sessions).
     * @returns {bigint}
     */
    get seed() {
        const ret = wasm.turboquantconfig_seed(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {boolean} v
     */
    set keys(v) {
        wasm.turboquantconfig_set_keys(this.__wbg_ptr, v);
    }
    /**
     * @param {bigint} v
     */
    set seed(v) {
        wasm.turboquantconfig_set_seed(this.__wbg_ptr, v);
    }
    /**
     * @param {boolean} v
     */
    set values(v) {
        wasm.turboquantconfig_set_values(this.__wbg_ptr, v);
    }
    /**
     * Compress the V side of the KV cache. Default `true`.
     * @returns {boolean}
     */
    get values() {
        const ret = wasm.turboquantconfig_values(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) TurboQuantConfig.prototype[Symbol.dispose] = TurboQuantConfig.prototype.free;

/**
 * Returns the version of the `cera` core library this binding wraps.
 *
 * Note this is **`cera`'s** version, not `cera-wasm`'s — JS callers
 * usually want to know what core lib is driving the engine, since
 * the wrapper crate version may evolve independently.
 * @returns {string}
 */
export function ceraVersion() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.ceraVersion(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Describe the CPU backend tier this build resolved at runtime, e.g.
 * `"tier=wasm_simd128 [simd128]"`.
 *
 * Diagnostic: it tells you whether the SIMD kernels are actually live,
 * which is the difference between roughly 1.4 and 0.64 tokens/second on the
 * wasm CPU path. A browser without the simd128 proposal, or a `.wasm` built
 * without `+simd128`, reports the scalar tier.
 * @returns {string}
 */
export function cpuBackendReport() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.cpuBackendReport(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export5(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Detect the tool-call format for a model architecture string (`"lfm2"`,
 * `"qwen3"`, …). Returns `undefined` for architectures with no known
 * convention.
 *
 * Prefer `CeraEngine.toolFormat` when you have an engine: it reads the
 * loaded model's own architecture, so it cannot be given the wrong string.
 * @param {string} architecture
 * @returns {ToolFormat | undefined}
 */
export function detectToolFormat(architecture) {
    const ptr0 = passStringToWasm0(architecture, wasm.__wbindgen_export, wasm.__wbindgen_export2);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detectToolFormat(ptr0, len0);
    return ret === 2 ? undefined : ret;
}

/**
 * @param {number} num_threads
 * @returns {Promise<any>}
 */
export function initThreadPool(num_threads) {
    const ret = wasm.initThreadPool(num_threads);
    return takeObject(ret);
}

/**
 * Bundles published on `LiquidAI/LeapBundles`, as
 * `[{ name, quants: [...] }]`.
 *
 * One GET to the HuggingFace model-info endpoint, grouped by the same
 * parser the native CLI uses, so the browser and the CLI list the same
 * catalog. Entries whose names wouldn't survive
 * `CeraEngine.fromBundleId` are filtered out rather than offered.
 * @returns {Promise<any>}
 */
export function listLeapBundles() {
    const ret = wasm.listLeapBundles();
    return takeObject(ret);
}

/**
 * Parse tool calls out of generated model text. Returns a JSON string
 * encoding an array of `ToolCall` (`[{name, arguments}]`) — `JSON.parse` it.
 * An empty array means the reply had no tool call.
 * @param {string} text
 * @param {ToolFormat} format
 * @returns {string}
 */
export function parseToolCalls(text, format) {
    let deferred3_0;
    let deferred3_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.parseToolCalls(retptr, ptr0, len0, format);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
        var ptr2 = r0;
        var len2 = r1;
        if (r3) {
            ptr2 = 0; len2 = 0;
            throw takeObject(r2);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export5(deferred3_0, deferred3_1, 1);
    }
}

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
 * @returns {Promise<boolean>}
 */
export function persistStorage() {
    const ret = wasm.persistStorage();
    return takeObject(ret);
}

/**
 * Build a GBNF grammar constraining output to a valid call for one of the
 * tools in `toolsJson` (a JSON array of `ToolDef`). Feed the result to
 * `GenerateOpts.setGrammar` and set `GenerateOpts.grammarTriggerTokens` for a lazy
 * tool-call trigger.
 * @param {string} tools_json
 * @param {ToolFormat} format
 * @returns {string}
 */
export function toolGrammar(tools_json, format) {
    let deferred3_0;
    let deferred3_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(tools_json, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.toolGrammar(retptr, ptr0, len0, format);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
        var ptr2 = r0;
        var len2 = r1;
        if (r3) {
            ptr2 = 0; len2 = 0;
            throw takeObject(r2);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export5(deferred3_0, deferred3_1, 1);
    }
}

export class wbg_rayon_PoolBuilder {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(wbg_rayon_PoolBuilder.prototype);
        obj.__wbg_ptr = ptr;
        wbg_rayon_PoolBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        wbg_rayon_PoolBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 0);
    }
    build() {
        wasm.wbg_rayon_poolbuilder_build(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    numThreads() {
        const ret = wasm.wbg_rayon_poolbuilder_numThreads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    receiver() {
        const ret = wasm.wbg_rayon_poolbuilder_receiver(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) wbg_rayon_PoolBuilder.prototype[Symbol.dispose] = wbg_rayon_PoolBuilder.prototype.free;

/**
 * @param {number} receiver
 */
export function wbg_rayon_start_worker(receiver) {
    wasm.wbg_rayon_start_worker(receiver);
}

function __wbg_get_imports(memory) {
    const import0 = {
        __proto__: null,
        __wbg_Error_2e59b1b37a9a34c3: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg___wbindgen_debug_string_dd5d2d07ce9e6c57: function(arg0, arg1) {
            const ret = debugString(getObject(arg1));
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_falsy_c6ddfae1bb56d5ef: function(arg0) {
            const ret = !getObject(arg0);
            return ret;
        },
        __wbg___wbindgen_is_function_49868bde5eb1e745: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_344c8750a8525473: function(arg0) {
            const ret = getObject(arg0) === null;
            return ret;
        },
        __wbg___wbindgen_is_object_40c5a80572e8f9d3: function(arg0) {
            const val = getObject(arg0);
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_b29b5c5a8065ba1a: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_c0cca72b82b86f4d: function(arg0) {
            const ret = getObject(arg0) === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_73fdd881ebd2e7a3: function() {
            const ret = wasm.memory;
            return addHeapObject(ret);
        },
        __wbg___wbindgen_module_7d79cdce5fe2ca41: function() {
            const ret = wasmModule;
            return addHeapObject(ret);
        },
        __wbg___wbindgen_rethrow_828b2014a519945b: function(arg0) {
            throw takeObject(arg0);
        },
        __wbg___wbindgen_string_get_914df97fcfa788f2: function(arg0, arg1) {
            const obj = getObject(arg1);
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_81fc77679af83bc6: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_3c3b4f651835fbcb: function(arg0) {
            getObject(arg0)._wbg_cb_unref();
        },
        __wbg_arrayBuffer_7bba74066875530e: function(arg0) {
            const ret = getObject(arg0).arrayBuffer();
            return addHeapObject(ret);
        },
        __wbg_async_5727feb662848999: function(arg0) {
            const ret = getObject(arg0).async;
            return ret;
        },
        __wbg_body_9a25d64338506fbe: function(arg0) {
            const ret = getObject(arg0).body;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_buffer_445cfbc3a2377e52: function(arg0) {
            const ret = getObject(arg0).buffer;
            return addHeapObject(ret);
        },
        __wbg_call_368fa9c372d473ba: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2), getObject(arg3));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_call_d578befcc3145dee: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_call_f2ac1622600b957f: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
            const ret = getObject(arg0).call(getObject(arg1), getObject(arg2), getObject(arg3), getObject(arg4));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_ceraengine_new: function(arg0) {
            const ret = CeraEngine.__wrap(arg0);
            return addHeapObject(ret);
        },
        __wbg_close_37e34297940956fd: function(arg0) {
            const ret = getObject(arg0).close();
            return addHeapObject(ret);
        },
        __wbg_close_e526ab9e090e8cc1: function(arg0) {
            getObject(arg0).close();
        },
        __wbg_createSyncAccessHandle_3be98daf699667a7: function(arg0) {
            const ret = getObject(arg0).createSyncAccessHandle();
            return addHeapObject(ret);
        },
        __wbg_createWritable_d5314165379c13be: function(arg0) {
            const ret = getObject(arg0).createWritable();
            return addHeapObject(ret);
        },
        __wbg_crypto_38df2bab126b63dc: function(arg0) {
            const ret = getObject(arg0).crypto;
            return addHeapObject(ret);
        },
        __wbg_data_fb9bcfd0c825e8e0: function(arg0) {
            const ret = getObject(arg0).data;
            return addHeapObject(ret);
        },
        __wbg_fetch_3679c69372c815bb: function(arg0, arg1, arg2) {
            const ret = fetch(getStringFromWasm0(arg0, arg1), getObject(arg2));
            return addHeapObject(ret);
        },
        __wbg_fetch_f020711f1a8019c7: function(arg0, arg1) {
            const ret = fetch(getStringFromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_flush_63f2ba6bf37bcfd5: function() { return handleError(function (arg0) {
            getObject(arg0).flush();
        }, arguments); },
        __wbg_getDirectoryHandle_a38f7b2c1aa52af4: function(arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).getDirectoryHandle(getStringFromWasm0(arg1, arg2), getObject(arg3));
            return addHeapObject(ret);
        },
        __wbg_getDirectory_3af764c18446017f: function(arg0) {
            const ret = getObject(arg0).getDirectory();
            return addHeapObject(ret);
        },
        __wbg_getFileHandle_326ca47811ae37a1: function(arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).getFileHandle(getStringFromWasm0(arg1, arg2), getObject(arg3));
            return addHeapObject(ret);
        },
        __wbg_getFile_0e25dfe508c6bd0a: function(arg0) {
            const ret = getObject(arg0).getFile();
            return addHeapObject(ret);
        },
        __wbg_getRandomValues_c44a50d8cfdaebeb: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).getRandomValues(getObject(arg1));
        }, arguments); },
        __wbg_getReader_3bcb712b2f3b80aa: function(arg0) {
            const ret = getObject(arg0).getReader();
            return addHeapObject(ret);
        },
        __wbg_getSize_6037025a1b5d08db: function() { return handleError(function (arg0) {
            const ret = getObject(arg0).getSize();
            return ret;
        }, arguments); },
        __wbg_get_4848e350b40afc16: function(arg0, arg1) {
            const ret = getObject(arg0)[arg1 >>> 0];
            return addHeapObject(ret);
        },
        __wbg_get_5caaa5a9aae7e0b1: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = getObject(arg1).get(getStringFromWasm0(arg2, arg3));
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        }, arguments); },
        __wbg_get_f96702c6245e4ef9: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(getObject(arg0), getObject(arg1));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_headers_e08dcb5aa09b9a63: function(arg0) {
            const ret = getObject(arg0).headers;
            return addHeapObject(ret);
        },
        __wbg_instanceof_Window_c0fee4c064502536: function(arg0) {
            let result;
            try {
                result = getObject(arg0) instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_isArray_db61795ad004c139: function(arg0) {
            const ret = Array.isArray(getObject(arg0));
            return ret;
        },
        __wbg_kind_eed6e8caeeb164cb: function(arg0) {
            const ret = getObject(arg0).kind;
            return (__wbindgen_enum_FileSystemHandleKind.indexOf(ret) + 1 || 3) - 1;
        },
        __wbg_length_0c32cb8543c8e4c8: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_length_6e821edde497a532: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_msCrypto_bd5a034af96bcba6: function(arg0) {
            const ret = getObject(arg0).msCrypto;
            return addHeapObject(ret);
        },
        __wbg_new_40792555590ec35c: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return __wasm_bindgen_func_elem_5392(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return addHeapObject(ret);
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_4f9fafbb3909af72: function() {
            const ret = new Object();
            return addHeapObject(ret);
        },
        __wbg_new_753190ec436990fe: function(arg0) {
            const ret = new Int32Array(getObject(arg0));
            return addHeapObject(ret);
        },
        __wbg_new_a560378ea1240b14: function(arg0) {
            const ret = new Uint8Array(getObject(arg0));
            return addHeapObject(ret);
        },
        __wbg_new_f3c9df4f38f3f798: function() {
            const ret = new Array();
            return addHeapObject(ret);
        },
        __wbg_new_from_slice_2580ff33d0d10520: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_new_from_slice_798885084b9cc1d2: function(arg0, arg1) {
            const ret = new Uint32Array(getArrayU32FromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_new_from_slice_d85ad974cf8f6f35: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_new_typed_14d7cc391ce53d2c: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return __wasm_bindgen_func_elem_5392(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return addHeapObject(ret);
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_length_9cedd08484b73942: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_new_worker_8edf8ebda6a768a7: function(arg0, arg1) {
            const ret = new Worker(getStringFromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_next_072e78dcf497124d: function() { return handleError(function (arg0) {
            const ret = getObject(arg0).next();
            return addHeapObject(ret);
        }, arguments); },
        __wbg_node_84ea875411254db1: function(arg0) {
            const ret = getObject(arg0).node;
            return addHeapObject(ret);
        },
        __wbg_now_e7c6795a7f81e10f: function(arg0) {
            const ret = getObject(arg0).now();
            return ret;
        },
        __wbg_of_f30df5d78b1d5cf3: function(arg0, arg1, arg2) {
            const ret = Array.of(getObject(arg0), getObject(arg1), getObject(arg2));
            return addHeapObject(ret);
        },
        __wbg_ok_36f7b13b74596c24: function(arg0) {
            const ret = getObject(arg0).ok;
            return ret;
        },
        __wbg_performance_3fcf6e32a7e1ed0a: function(arg0) {
            const ret = getObject(arg0).performance;
            return addHeapObject(ret);
        },
        __wbg_persist_f1b861b64c5ab232: function() { return handleError(function (arg0) {
            const ret = getObject(arg0).persist();
            return addHeapObject(ret);
        }, arguments); },
        __wbg_persisted_0a2eeeb65f567dac: function() { return handleError(function (arg0) {
            const ret = getObject(arg0).persisted();
            return addHeapObject(ret);
        }, arguments); },
        __wbg_postMessage_6010c627e5408e23: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).postMessage(getObject(arg1));
        }, arguments); },
        __wbg_process_44c7a14e11e9f69e: function(arg0) {
            const ret = getObject(arg0).process;
            return addHeapObject(ret);
        },
        __wbg_prototypesetcall_3e05eb9545565046: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
        },
        __wbg_push_6bdbc990be5ac37b: function(arg0, arg1) {
            const ret = getObject(arg0).push(getObject(arg1));
            return ret;
        },
        __wbg_queueMicrotask_abaf92f0bd4e80a4: function(arg0) {
            const ret = getObject(arg0).queueMicrotask;
            return addHeapObject(ret);
        },
        __wbg_queueMicrotask_df5a6dac26d818f3: function(arg0) {
            queueMicrotask(getObject(arg0));
        },
        __wbg_randomFillSync_6c25eac9869eb53c: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).randomFillSync(takeObject(arg1));
        }, arguments); },
        __wbg_read_308df9569547d888: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = getObject(arg0).read(getArrayU8FromWasm0(arg1, arg2));
            return ret;
        }, arguments); },
        __wbg_read_316bf844c93a6ccc: function(arg0) {
            const ret = getObject(arg0).read();
            return addHeapObject(ret);
        },
        __wbg_read_8569bf7e69cc3089: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).read(getArrayU8FromWasm0(arg1, arg2), getObject(arg3));
            return ret;
        }, arguments); },
        __wbg_removeEntry_27b46d2f2b6fb820: function(arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).removeEntry(getStringFromWasm0(arg1, arg2), getObject(arg3));
            return addHeapObject(ret);
        },
        __wbg_removeEntry_f038ab74448d1824: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).removeEntry(getStringFromWasm0(arg1, arg2));
            return addHeapObject(ret);
        },
        __wbg_require_b4edbdcf3e2a1ef0: function() { return handleError(function () {
            const ret = module.require;
            return addHeapObject(ret);
        }, arguments); },
        __wbg_resolve_0a79de24e9d2267b: function(arg0) {
            const ret = Promise.resolve(getObject(arg0));
            return addHeapObject(ret);
        },
        __wbg_set_8ee2d34facb8466e: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(getObject(arg0), getObject(arg1), getObject(arg2));
            return ret;
        }, arguments); },
        __wbg_set_at_da2d1d4dc8ed37da: function(arg0, arg1) {
            getObject(arg0).at = arg1;
        },
        __wbg_set_create_0654e513e8ccb2be: function(arg0, arg1) {
            getObject(arg0).create = arg1 !== 0;
        },
        __wbg_set_create_4b5cddb7e7c14744: function(arg0, arg1) {
            getObject(arg0).create = arg1 !== 0;
        },
        __wbg_set_method_1971272fe557e972: function(arg0, arg1, arg2) {
            getObject(arg0).method = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_onmessage_733b5167b7dd9b01: function(arg0, arg1) {
            getObject(arg0).onmessage = getObject(arg1);
        },
        __wbg_set_recursive_98efa95de44b1bbf: function(arg0, arg1) {
            getObject(arg0).recursive = arg1 !== 0;
        },
        __wbg_set_signal_8564a226c5c6853c: function(arg0, arg1) {
            getObject(arg0).signal = getObject(arg1);
        },
        __wbg_size_7306c9406e13bf29: function(arg0) {
            const ret = getObject(arg0).size;
            return ret;
        },
        __wbg_startWorkers_8b582d57e92bd2d4: function(arg0, arg1, arg2) {
            const ret = startWorkers(takeObject(arg0), takeObject(arg1), wbg_rayon_PoolBuilder.__wrap(arg2));
            return addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_a1248013d790bf5f: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_f2e0f995a21329ff: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_SELF_24f78b6d23f286ea: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_WINDOW_59fd959c540fe405: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_statusText_128c4dad452b4075: function(arg0, arg1) {
            const ret = getObject(arg1).statusText;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_status_44ecb0ac1da253f4: function(arg0) {
            const ret = getObject(arg0).status;
            return ret;
        },
        __wbg_subarray_0f98d3fb634508ad: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_text_43bdfba45e602cf9: function() { return handleError(function (arg0) {
            const ret = getObject(arg0).text();
            return addHeapObject(ret);
        }, arguments); },
        __wbg_then_00eed3ac0b8e82cb: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).then(getObject(arg1), getObject(arg2));
            return addHeapObject(ret);
        },
        __wbg_then_50c1ba21bde9ae37: function(arg0, arg1) {
            const ret = getObject(arg0).then(getObject(arg1));
            return addHeapObject(ret);
        },
        __wbg_then_a0c8db0381c8994c: function(arg0, arg1) {
            const ret = getObject(arg0).then(getObject(arg1));
            return addHeapObject(ret);
        },
        __wbg_timeOrigin_f3d5cb4f4a06c2b7: function(arg0) {
            const ret = getObject(arg0).timeOrigin;
            return ret;
        },
        __wbg_timeout_12a0abd46970a7bb: function(arg0) {
            const ret = AbortSignal.timeout(arg0 >>> 0);
            return addHeapObject(ret);
        },
        __wbg_truncate_f9ca0aa6efd94ce6: function() { return handleError(function (arg0, arg1) {
            getObject(arg0).truncate(arg1 >>> 0);
        }, arguments); },
        __wbg_value_b39d2197b4e92689: function(arg0) {
            const ret = getObject(arg0).value;
            return addHeapObject(ret);
        },
        __wbg_values_2ecd25f48dfd2b37: function(arg0) {
            const ret = getObject(arg0).values();
            return addHeapObject(ret);
        },
        __wbg_versions_276b2795b1c6a219: function(arg0) {
            const ret = getObject(arg0).versions;
            return addHeapObject(ret);
        },
        __wbg_waitAsync_85b896c39ac58fbb: function(arg0, arg1, arg2) {
            const ret = Atomics.waitAsync(getObject(arg0), arg1 >>> 0, arg2);
            return addHeapObject(ret);
        },
        __wbg_waitAsync_f6bff47f206d803d: function() {
            const ret = Atomics.waitAsync;
            return addHeapObject(ret);
        },
        __wbg_write_726121caffd5fc3e: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = getObject(arg0).write(getArrayU8FromWasm0(arg1, arg2), getObject(arg3));
            return ret;
        }, arguments); },
        __wbg_write_fc53b37dcc29642e: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = getObject(arg0).write(getArrayU8FromWasm0(arg1, arg2));
            return addHeapObject(ret);
        }, arguments); },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 1, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, __wasm_bindgen_func_elem_230);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 2486, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, __wasm_bindgen_func_elem_5375);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 2488, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, __wasm_bindgen_func_elem_5377);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000004: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_link_30693bc4809a2ef3: function(arg0) {
            const val = `onmessage = function (ev) {
                let [ia, index, value] = ev.data;
                ia = new Int32Array(ia.buffer);
                let result = Atomics.wait(ia, index, value);
                postMessage(result);
            };
            `;
            const ret = typeof URL.createObjectURL === 'undefined' ? "data:application/javascript," + encodeURIComponent(val) : URL.createObjectURL(new Blob([val], { type: "text/javascript" }));
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_object_clone_ref: function(arg0) {
            const ret = getObject(arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
        memory: memory || new WebAssembly.Memory({initial:26,maximum:65536,shared:true}),
    };
    return {
        __proto__: null,
        "./cera_wasm_bg.js": import0,
    };
}

function __wasm_bindgen_func_elem_230(arg0, arg1, arg2) {
    wasm.__wasm_bindgen_func_elem_230(arg0, arg1, addHeapObject(arg2));
}

function __wasm_bindgen_func_elem_5377(arg0, arg1, arg2) {
    wasm.__wasm_bindgen_func_elem_5377(arg0, arg1, addHeapObject(arg2));
}

function __wasm_bindgen_func_elem_5375(arg0, arg1, arg2) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.__wasm_bindgen_func_elem_5375(retptr, arg0, arg1, addHeapObject(arg2));
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        if (r1) {
            throw takeObject(r0);
        }
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

function __wasm_bindgen_func_elem_5392(arg0, arg1, arg2, arg3) {
    wasm.__wasm_bindgen_func_elem_5392(arg0, arg1, addHeapObject(arg2), addHeapObject(arg3));
}


const __wbindgen_enum_FileSystemHandleKind = ["file", "directory"];
const BundleRepoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bundlerepo_free(ptr >>> 0, 1));
const CeraEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ceraengine_free(ptr >>> 0, 1));
const GenerateOptsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_generateopts_free(ptr >>> 0, 1));
const GenerateSummaryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_generatesummary_free(ptr >>> 0, 1));
const LoraAdaptersFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_loraadapters_free(ptr >>> 0, 1));
const ManifestFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_manifest_free(ptr >>> 0, 1));
const SessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_session_free(ptr >>> 0, 1));
const SessionConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sessionconfig_free(ptr >>> 0, 1));
const TokenizerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tokenizer_free(ptr >>> 0, 1));
const TurboQuantConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_turboquantconfig_free(ptr >>> 0, 1));
const wbg_rayon_PoolBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wbg_rayon_poolbuilder_free(ptr >>> 0, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function addBorrowedObject(obj) {
    if (stack_pointer == 1) throw new Error('out of js stack');
    heap[--stack_pointer] = obj;
    return stack_pointer;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_export4(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export3(addHeapObject(e));
    }
}

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_export4(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let stack_pointer = 1024;

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : undefined);
if (cachedTextDecoder) cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len));
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder() : undefined);

if (cachedTextEncoder) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module, thread_stack_size) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof thread_stack_size !== 'undefined' && (typeof thread_stack_size !== 'number' || thread_stack_size === 0 || thread_stack_size % 65536 !== 0)) {
        throw new Error('invalid stack size');
    }

    wasm.__wbindgen_start(thread_stack_size);
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module, memory, thread_stack_size} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports(memory);
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module, thread_stack_size);
}

async function __wbg_init(module_or_path, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path, memory, thread_stack_size} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('cera_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports(memory);

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module, thread_stack_size);
}

export { initSync, __wbg_init as default };

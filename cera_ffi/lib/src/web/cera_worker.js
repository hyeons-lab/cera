// Web Worker host for the Cera inference engine.
//
// `cera_ffi_flutter`'s web implementation drives this over `postMessage`. It is
// plain JS with no Dart or Flutter dependency, so a hand-written page can drive
// it too; the protocol below is the whole contract.
//
// It exists because the CPU path's `Session.generate` is a SYNCHRONOUS wasm
// call that runs the whole decode loop before returning. On the main thread
// that freezes the tab for the duration. Off it, the tab stays live and the
// per-token `postMessage`es still arrive as they are produced, because the
// receiving thread is not the blocked one.
//
// ## Why every import here is dynamic
//
// `import('./cera_wasm.js')` rather than a static `import ... from`. A static
// import of a name the loaded artifact does not export is a hard module error:
// the worker never evaluates, never registers `onmessage`, and the page hangs
// with nothing in the console. Since the same script has to serve builds with
// and without optional exports, and has to resolve its module from a URL the
// host supplies, the import cannot be static.
//
// ## Protocol
//
// Request:  {id, op, ...args}
// Reply:    {id, ok: true, result} | {id, ok: false, error}
// Stream:   {id, event: 'token', text}                      (generate)
//           {id, event: 'progress', url, done, total}       (openBundle)
//
// Every request gets exactly one reply, and `id` correlates them. Streaming
// events carry the same `id` and always precede that request's reply.

'use strict';

/**
 * An error the host should surface as its platform's "not supported here"
 * type rather than as a generic failure.
 *
 * `postMessage` cannot carry an Error subclass (structured clone drops the
 * prototype), so the kind travels as a field on the reply and the host maps it
 * back. Without this the documented `UnsupportedError` for `reset` on the GPU
 * backend arrived as an ordinary worker exception, and a caller catching what
 * the docs promised caught nothing.
 */
function unsupported(message) {
  const err = new Error(message);
  err.ceraKind = 'unsupported';
  return err;
}

/** The wasm module namespace, once `open` has imported it. */
let wasm = null;

/** Active engine/session state. Exactly one of the two paths is populated. */
let cpu = null; // {engine, session, tokenizer}
let gpu = null; // {session, tokenizer}

/** Human-readable description of the backend that actually loaded. */
let backendLabel = 'none';

/**
 * Tokens currently in the live session's KV cache. Zero means BOS.
 *
 * Read from the session rather than counted here. A counter cannot be right:
 * a prompt can be rejected before any forward runs (cache untouched) or fail
 * partway through decode (cache advanced), and the two need opposite answers.
 */
function position() {
  if (gpu) return gpu.session.position;
  if (cpu) return cpu.session.position;
  return 0;
}

/**
 * The tokenizer of whichever path is live.
 *
 * Both paths expose one: the CPU engine has always had `engine.tokenizer`, and
 * `WebGpuSession` gained the same getter so the GPU path is not restricted to
 * raw prompt completion. Without it a chat model cannot be driven on the GPU at
 * all: rendering a chat template needs the tokenizer, and feeding the rendered
 * text back needs an encoder that lowers `<|im_start|>` to its id rather than
 * to the characters spelling it.
 */
function tokenizer() {
  if (gpu) return gpu.tokenizer;
  if (cpu) return cpu.tokenizer;
  throw new Error('no model is open');
}

/**
 * Encode `text` into the ids to feed the model.
 *
 * Plain `encode` plus an explicit BOS rather than `encodeSpecial(text, true)`,
 * which does two things where only one is wanted: it also appends EOS when the
 * GGUF sets `add_eos_token`, ending the turn before the model has seen the
 * generation header. `encode` still lowers the template's `<|im_start|>`-style
 * markers to their own ids, which is the part that matters.
 *
 * `first` gates the BOS, which belongs at position 0 only. The KV cache
 * persists across calls, so a later BOS would land mid-sequence at a nonzero
 * position and desync RoPE from every other Cera path. The already-present
 * check matters too: a chat template that emits its own BOS would otherwise
 * get a second one, which is not a crash, just a quietly worse first token.
 *
 * Matches the native Dart implementation exactly, and matches
 * `WebGpuSession::generate` on the BOS rule and `Session::append_text` on the
 * encoding. (`append_text` itself has no BOS handling; the CLI frames its own
 * prompts before calling it.)
 */
function encodePrompt(text, first) {
  const tk = tokenizer();
  const ids = Array.from(tk.encode(text));
  const bos = tk.bosToken;
  if (first && tk.addBosToken && bos != null && ids[0] !== bos) {
    ids.unshift(bos);
  }
  return Uint32Array.from(ids);
}

/**
 * Load the wasm module once, from a URL the host resolves.
 *
 * The URL is not derived from `import.meta.url` because the worker script and
 * the wasm artifacts need not be co-located: the script ships inside the pub
 * package and the artifacts are installed into the app's `web/` directory.
 */
async function ensureModule(moduleUrl) {
  if (wasm) return;
  // Absolutize first. A dynamic `import()` treats a specifier with no leading
  // `./`, `../` or scheme as a BARE specifier (a package name), and rejects it
  // outright with "Failed to resolve module specifier", even though the very
  // same string works in `new Worker(...)`. Resolving against the worker's own
  // location turns any of the three forms into something importable.
  const module = await import(new URL(moduleUrl, self.location.href).href);
  // wasm-bindgen's `--target web` default export fetches and instantiates the
  // `.wasm` sibling. It resolves that path against the JS module's own URL, so
  // the two files must stay next to each other.
  //
  // Assigned only after instantiation succeeds. Caching the namespace first
  // makes a failed load stick: every later `open` short-circuits on the cached
  // value and then fails inside wasm-bindgen with "wasm not initialized"
  // instead of retrying the load that actually broke.
  await module.default();
  if (typeof module.initThreadPool === 'function' && self.crossOriginIsolated) {
    try {
      const concurrency = self.navigator?.hardwareConcurrency || 4;
      await module.initThreadPool(concurrency);
      console.info(`[cera:worker] multi-threaded wasm initialized with ${concurrency} threads`);
    } catch (e) {
      console.warn(`[cera:worker] initThreadPool failed: ${e}`);
    }
  }
  wasm = module;
}

/**
 * Try the GPU path. Returns false (rather than throwing) for every reason the
 * GPU cannot serve this model, so `auto` can fall through to the CPU.
 *
 * The failure modes are not all detectable up front: `navigator.gpu` can exist
 * while `requestAdapter` yields nothing, and `WebGpuSession` is LFM2-only, so a
 * dense-transformer GGUF throws from `create` after WebGPU itself came up fine.
 * Both must degrade rather than fail the open.
 */
async function tryGpu(bytes, contextSize, mmproj) {
  if (!self.navigator || !self.navigator.gpu) return false;
  if (typeof wasm.WebGpuSession !== 'function') return false;
  // `createWithParts` is the newer of the two and is absent from a wasm build
  // predating it, so fall back rather than throwing a TypeError that `auto`
  // would then read as "no GPU" for entirely the wrong reason.
  if (mmproj && typeof wasm.WebGpuSession.createWithParts !== 'function') return false;
  let session;
  try {
    session = mmproj
      ? await wasm.WebGpuSession.createWithParts(bytes, mmproj, contextSize)
      : await wasm.WebGpuSession.create(bytes, contextSize);
  } catch (_) {
    return false;
  }
  gpu = { session, tokenizer: session.tokenizer };
  backendLabel = `webgpu: ${session.adapter}`;
  return true;
}

/**
 * Try the GPU path for a bundle. Returns `null` on success, or a string saying
 * why the GPU could not serve it, so `auto` can fall through to the CPU and
 * still report the reason if that fails too.
 *
 * A reason rather than `tryGpu`'s bare boolean because this path can fail for
 * something the caller must hear about. `tryGpu` is handed bytes and can only
 * fail on the model itself; this one downloads, so a network failure arrives
 * here as well, and a silent fall-through would then report whatever the CPU
 * retry says about a cache that was never populated.
 *
 * Separate from `tryGpu` for the same reason the two constructors are separate:
 * this one keeps the weights inside wasm. Handing them to JS to reuse `tryGpu`
 * would cost a second full copy of the model and reintroduce the ~2 GiB ceiling
 * on a single `ArrayBuffer`, which is what the bundle constructors exist to
 * avoid.
 *
 * A failure *after* the download (a non-LFM2 architecture, the common one) has
 * already populated the cache, so the CPU retry behind it loads from the store
 * rather than downloading again.
 */
async function tryGpuBundle(repo, bundleId, quant, contextSize, onProgress) {
  if (!self.navigator || !self.navigator.gpu) return 'this browser exposes no navigator.gpu';
  // Absent from a wasm build without the `wgpu` feature, and from one predating
  // this constructor. Both fall back rather than throwing a TypeError that
  // `auto` would read as "no GPU" for the wrong reason.
  if (typeof wasm.WebGpuSession !== 'function') {
    return 'this wasm build has no WebGpuSession (built without the `wgpu` feature)';
  }
  if (typeof wasm.WebGpuSession.fromBundleId !== 'function') {
    return 'this wasm build predates WebGpuSession.fromBundleId';
  }
  let session;
  try {
    session = await wasm.WebGpuSession.fromBundleId(
      repo,
      bundleId,
      quant,
      contextSize,
      // No KV compression: `SessionConfig.kvCompression` is not exposed through
      // this protocol either, so requesting it on one path only would make the
      // two backends disagree about memory for reasons a caller cannot see.
      undefined,
      onProgress,
    );
  } catch (err) {
    return String((err && err.message) || err);
  }
  gpu = { session, tokenizer: session.tokenizer };
  backendLabel = `webgpu: ${session.adapter}`;
  return null;
}

function openCpu(bytes, contextSize, mmproj, inferenceType) {
  // `fromGgufParts` covers the text-only case too (a null projector is exactly
  // `fromGgufBytes`), so both paths resolve the inference type through one
  // constructor instead of two that have to agree.
  const engine = wasm.CeraEngine.fromGgufParts(bytes, mmproj, contextSize, inferenceType);
  initCpuSession(engine);
}

function initCpuSession(engine) {
  const config = new wasm.SessionConfig();
  try {
    const session = engine.newSession(config);
    cpu = { engine, session, tokenizer: engine.tokenizer };
  } finally {
    config.free();
  }
  backendLabel = 'wasm cpu';
}

/**
 * Open a bundle on the CPU engine.
 *
 * `fromBundleId` rather than pulling the bytes out of the repo and calling
 * `fromGgufParts`: the manifest states the modality outright and names every
 * file the bundle needs, so a VL or audio bundle arrives complete with no
 * guessing, and the weights never cross into JS.
 */
async function openCpuBundle(repo, bundleId, quant, contextSize, onProgress) {
  const engine = await wasm.CeraEngine.fromBundleId(
    repo,
    bundleId,
    quant,
    contextSize,
    onProgress,
  );
  initCpuSession(engine);
}

/**
 * What the live model accepts and emits.
 *
 * Both the GPU and CPU sessions report their own loaded capabilities (text,
 * vision, audio) based on the attached multimodal projectors.
 */
function capabilitiesOf() {
  const caps = gpu ? gpu.session.capabilities : cpu.engine.capabilities;
  return {
    textIn: caps.textIn,
    textOut: caps.textOut,
    imageIn: caps.imageIn,
    audioIn: caps.audioIn,
    audioOut: caps.audioOut,
  };
}

const OPS = {
  /**
   * Import the module, then open a model. `backend` is 'auto' | 'gpu' | 'cpu'.
   *
   * `bytes` arrives as a transferred ArrayBuffer, so the host's copy is gone by
   * the time this runs. It is wrapped, not copied, but wasm-bindgen does copy
   * it into linear memory (or into GPU buffers) during construction, so peak
   * usage is briefly twice the model size on both paths.
   */
  async open({ moduleUrl, bytes, mmproj, contextSize, backend, inferenceType }) {
    await ensureModule(moduleUrl);
    const view = new Uint8Array(bytes);
    // Transferred separately from `bytes`, and absent for a text-only model.
    const proj = mmproj ? new Uint8Array(mmproj) : undefined;
    const ctx = contextSize ?? undefined;
    const type = inferenceType ?? undefined;
    if (backend === 'gpu') {
      if (!(await tryGpu(view, ctx, proj))) {
        throw new Error(
          'the WebGPU backend is unavailable: either this browser exposes no ' +
            'navigator.gpu, no adapter could be acquired, or the model is not ' +
            'an LFM2 GGUF (the only architecture with a browser GPU path). ' +
            'Use backend: auto to fall back to the CPU instead.',
        );
      }
    } else if (backend === 'cpu') {
      openCpu(view, ctx, proj, type);
    } else if (!(await tryGpu(view, ctx, proj))) {
      openCpu(view, ctx, proj, type);
    }
    return { backend: backendLabel, capabilities: capabilitiesOf() };
  },

  /**
   * The bundles published on `LiquidAI/LeapBundles`, as
   * `[{ name, quants: [...] }]`.
   *
   * Takes `moduleUrl` like `open` does, and for the same reason: a picker is
   * the first thing an app shows, so this usually runs before any model has
   * been opened and cannot assume the module is already imported.
   */
  async listBundles({ moduleUrl }) {
    await ensureModule(moduleUrl);
    return wasm.listLeapBundles();
  },

  /**
   * Download a published bundle by `<bundleId, quant>` and open it, streaming
   * `progress` events while files are fetched.
   *
   * The counterpart to `open` for callers who have a bundle id rather than
   * bytes, and the better path in a browser wherever both would work: the
   * weights go from the store straight into wasm linear memory, never through
   * a JS `ArrayBuffer`, so this costs one copy of the model instead of two and
   * is not bounded by the ~2 GiB limit on a single contiguous JS allocation.
   *
   * `storeDir` names the OPFS directory to cache under; omitting it takes
   * `BundleRepo`'s own default. A cached bundle re-opens without any network
   * access and fires no progress events.
   */
  async openBundle(req, post) {
    const { moduleUrl, bundleId, quant, contextSize, backend, storeDir } = req;
    await ensureModule(moduleUrl);
    const repo = new wasm.BundleRepo(storeDir ?? undefined);
    const ctx = contextSize ?? undefined;
    // `total` is null when the server sends no Content-Length, and crosses as
    // null rather than being dropped so the host can tell "unknown size" from
    // "no progress yet".
    const onProgress = (url, done, total) => {
      post({ event: 'progress', url, done, total: total ?? null });
    };
    try {
      if (backend === 'cpu') {
        await openCpuBundle(repo, bundleId, quant, ctx, onProgress);
        return { backend: backendLabel, capabilities: capabilitiesOf() };
      }
      const why = await tryGpuBundle(repo, bundleId, quant, ctx, onProgress);
      if (why === null) {
        return { backend: backendLabel, capabilities: capabilitiesOf() };
      }
      if (backend === 'gpu') {
        throw new Error(
          `the WebGPU backend could not load bundle ${bundleId}/${quant}: ${why}. ` +
            'Only LFM2 bundles have a browser GPU path. Use backend: auto to fall ' +
            'back to the CPU instead.',
        );
      }
      // `auto`. The CPU load is the fallback, but the GPU reason is the more
      // useful half of a double failure: it is the one that says whether the
      // download itself failed, so it rides along rather than being discarded.
      try {
        await openCpuBundle(repo, bundleId, quant, ctx, onProgress);
      } catch (err) {
        throw new Error(
          `${String((err && err.message) || err)} (the WebGPU path was tried first ` +
            `and reported: ${why})`,
        );
      }
      return { backend: backendLabel, capabilities: capabilitiesOf() };
    } finally {
      // The repo is a wasm-bindgen handle and nothing else holds it once the
      // load has read what it needs, so it has to be freed explicitly like the
      // session/engine/tokenizer handles in `close`. In `finally` because the
      // `auto` path can leave by three different routes.
      repo.free();
    }
  },

  /**
   * Feed an image into the live conversation.
   *
   * Both paths append patch embeddings to the same KV cache `generate` writes
   * to, so ordering is the caller's: image first, then the question.
   */
  appendImage({ bytes, maxLongSize }) {
    const t0 = performance.now();
    const view = new Uint8Array(bytes);
    const cap = maxLongSize ?? undefined;
    console.info(
      `[cera:worker] appendImage: received ${view.byteLength} bytes of image data (maxLongSize: ${cap ?? 'model-default'})`,
    );
    if (gpu) {
      console.info(
        '[cera:worker] appendImage: preprocessing and encoding image patches for WebGPU KV cache...',
      );
      gpu.session.appendImage(view, cap);
    } else {
      console.info(
        '[cera:worker] appendImage: preprocessing and encoding image patches on CPU session (this may take several seconds)...',
      );
      cpu.session.appendImage(view, cap);
    }
    const elapsed = (performance.now() - t0).toFixed(1);
    console.info(`[cera:worker] appendImage: image embeddings seeded into KV cache in ${elapsed}ms`);
    return null;
  },

  /**
   * Feed mono PCM audio into the live conversation.
   */
  appendAudio({ pcm, sampleRate }) {
    const t0 = performance.now();
    const samples = Float32Array.from(pcm);
    const sr = sampleRate ?? 16000;
    console.info(
      `[cera:worker] appendAudio: processing ${samples.length} audio samples at ${sr}Hz (${(samples.length / sr).toFixed(1)}s)...`,
    );
    if (gpu) {
      console.info('[cera:worker] appendAudio: encoding audio frames for WebGPU KV cache...');
      gpu.session.appendAudio(samples, sr);
    } else {
      console.info('[cera:worker] appendAudio: encoding audio frames on CPU session...');
      cpu.session.appendAudio(samples, sr);
    }
    const elapsed = (performance.now() - t0).toFixed(1);
    console.info(`[cera:worker] appendAudio: audio embeddings seeded into KV cache in ${elapsed}ms`);
    return null;
  },

  /**
   * Transcribe mono PCM.
   *
   * Engine-level and CPU-only: `transcribe` runs its own prefill and decode on
   * the engine, and the GPU path has no engine behind it (`WebGpuSession` owns
   * a model directly and exposes no audio entry point).
   */
  transcribe({ pcm, sampleRate }) {
    if (!cpu) {
      throw unsupported(
        'transcribe is not supported on the WebGPU backend: WebGpuSession has ' +
          'no audio path. Open the model with backend: cpu to transcribe.',
      );
    }
    console.info(`[cera:worker] transcribe: running ASR on ${pcm.length} samples at ${sampleRate}Hz...`);
    const t0 = performance.now();
    const result = cpu.engine.transcribe(Float32Array.from(pcm), sampleRate);
    console.info(`[cera:worker] transcribe: completed in ${(performance.now() - t0).toFixed(1)}ms`);
    return result;
  },

  capabilities() {
    return capabilitiesOf();
  },

  /**
   * Prefill `prompt` and decode up to `maxTokens`, streaming each decoded piece
   * back as a `token` event before the reply.
   *
   * Both paths append to a live KV cache rather than resetting, so consecutive
   * calls continue one conversation; `reset` starts over.
   */
  async generate(req, post) {
    const { prompt, maxTokens } = req;
    const currentPos = position();
    console.info(
      `[cera:worker] generate: starting generation on "${backendLabel}" (context position: ${currentPos}, maxTokens: ${maxTokens})`,
    );
    // Seeding is per SESSION in the CPU wasm API, not per generate:
    // `GenerateOpts` has no seed field and assigning one just creates a dead JS
    // property. Honoring it therefore means rebuilding the session, which is
    // only meaningful before anything has been fed. The GPU path takes its seed
    // as a `generateTokens` argument instead, so it needs none of this.
    if (req.seed != null && currentPos === 0 && cpu) {
      const config = new wasm.SessionConfig();
      config.seed = BigInt(req.seed);
      try {
        cpu.session.free();
        cpu.session = cpu.engine.newSession(config);
      } finally {
        config.free();
      }
    }
    const ids = encodePrompt(prompt, currentPos === 0);
    console.info(`[cera:worker] generate: prompt encoded into ${ids.length} tokens`);
    const started = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;
    let text = '';
    const onToken = (piece) => {
      tokenCount++;
      if (!firstTokenTime) {
        firstTokenTime = performance.now();
        const ttft = (firstTokenTime - started).toFixed(1);
        console.info(`[cera:worker] generate: first token emitted in ${ttft}ms (TTFT)`);
      }
      text += piece;
      post({ event: 'token', text: piece });
    };
    if (gpu) {
      // Caller-framed: `generateTokens` prepends nothing, which is what makes
      // the BOS rule in `encodePrompt` the single place BOS is decided.
      //
      // Sampling knobs pass straight through. `null` for any of them means the
      // wasm side falls back to `SamplerConfig`'s default, which is the same
      // thing omitting them from `GenerateOpts` does on the CPU path, so the
      // two backends answer a bare `generate()` the same way.
      await gpu.session.generateTokens(
        ids,
        maxTokens,
        req.temperature ?? null,
        req.topP ?? null,
        req.topK ?? null,
        req.seed != null ? BigInt(req.seed) : null,
        onToken,
      );
    } else {
      const tk = cpu.tokenizer;
      if (ids.length > 0) {
        cpu.session.appendTokens(ids);
      }
      const opts = new wasm.GenerateOpts();
      opts.maxTokens = maxTokens;
      if (req.temperature != null) opts.temperature = req.temperature;
      if (req.topP != null) opts.topP = req.topP;
      if (req.topK != null) opts.topK = req.topK;
      // Emit per token rather than per buffer-full; the point of a worker is
      // that the host sees output as it is produced.
      opts.flushEveryTokens = 1;
      let uncommittedTokens = [];
      try {
        cpu.session.generate(opts, (toks) => {
          for (let i = 0; i < toks.length; i++) {
            uncommittedTokens.push(toks[i]);
          }
          const decoded = tk.decode(Uint32Array.from(uncommittedTokens));
          if (decoded.endsWith('\uFFFD')) {
            // Incomplete multibyte UTF-8 character spanning across token chunks;
            // hold back until the completing token arrives.
            return;
          }
          if (decoded.length > 0) {
            onToken(decoded);
            uncommittedTokens = [];
          }
        });
        if (uncommittedTokens.length > 0) {
          const remaining = tk.decode(Uint32Array.from(uncommittedTokens));
          if (remaining.length > 0) {
            onToken(remaining);
          }
          uncommittedTokens = [];
        }
      } finally {
        opts.free();
      }
    }
    const ms = performance.now() - started;
    const decodeMs = firstTokenTime ? (performance.now() - firstTokenTime) : ms;
    const tps =
      tokenCount > 1 && decodeMs > 0
        ? ((tokenCount - 1) / (decodeMs / 1000)).toFixed(1)
        : (tokenCount / (ms / 1000)).toFixed(1);
    console.info(
      `[cera:worker] generate: completed ${tokenCount} tokens in ${ms.toFixed(1)}ms (${tps} tok/s)`,
    );
    return { text, elapsedMs: ms };
  },

  applyChatTemplate({ messagesJson, addGenerationPrompt }) {
    // The messages cross as JSON so the host does not have to build a JS array
    // of objects through interop, but `applyChatTemplate` wants the real array:
    // it type-checks its argument and rejects a string with "messages must be
    // an array".
    return tokenizer().applyChatTemplate(JSON.parse(messagesJson), addGenerationPrompt);
  },

  encode({ text, addSpecial }) {
    return Array.from(tokenizer().encodeSpecial(text, addSpecial));
  },

  decode({ tokens }) {
    return tokenizer().decode(Uint32Array.from(tokens));
  },

  /**
   * Drop the conversation, keeping the loaded weights.
   *
   * `WebGpuSession` has no reset, so the GPU path reports the limitation rather
   * than silently continuing a conversation the caller believes it cleared.
   */
  reset() {
    if (cpu) {
      // `Session.reset`, not a fresh session. It clears KV, position and token
      // history and lowers the cancel flag, which is all rebuilding did, while
      // keeping the session's own config: a rebuild with a default
      // `SessionConfig` silently discarded a seed the `generate` op installed.
      cpu.session.reset();
      return null;
    }
    throw unsupported(
      'reset is not supported on the WebGPU backend: WebGpuSession owns its ' +
        'KV cache on the GPU and exposes no way to clear it. Close the engine ' +
        'and open it again to start a new conversation.',
    );
  },

  /**
   * Request an early stop. Reaches neither backend's in-flight decode, for two
   * unrelated reasons, so treat it as best-effort.
   *
   * On the CPU path the decode loop is one synchronous wasm call. This message
   * is not dequeued until that call returns, so the flag is only ever set
   * between generations. Reaching an in-flight decode would need a
   * SharedArrayBuffer flag, hence cross-origin isolation, which is the
   * requirement this design otherwise avoids entirely.
   *
   * On the GPU path there is nothing to call: `WebGpuSession` exposes no
   * cancel. Its decode does yield to the event loop between tokens, so a stop
   * is implementable there, but it needs a Rust-side entry point that does not
   * exist yet.
   *
   * Clearing the CPU flag right after setting it is deliberate. The engine's
   * cancel flag is sticky, and `Session::generate` clears it only at entry,
   * after `appendTokens` has already run; leaving it set poisons the next
   * turn's chunked prefill with `Cancelled`.
   */
  cancel() {
    if (cpu) {
      cpu.session.cancel();
      cpu.session.clearCancel();
    }
    return null;
  },

  close() {
    // The tokenizer handles are separate wasm-bindgen objects holding their own
    // `Arc<BpeTokenizer>` clone, so freeing the engine does not reclaim them.
    // Terminating the worker would, but this protocol is documented as usable
    // standalone, where open/close cycles would otherwise accumulate them.
    if (cpu) {
      cpu.tokenizer.free();
      cpu.session.free();
      cpu.engine.free();
      cpu = null;
    }
    if (gpu) {
      gpu.tokenizer.free();
      gpu.session.free();
      gpu = null;
    }
    backendLabel = 'none';
    return null;
  },
};

self.onmessage = async (e) => {
  const req = e.data;
  const { id, op } = req;
  const post = (msg) => self.postMessage({ id, ...msg });
  try {
    const handler = OPS[op];
    if (!handler) throw new Error(`unknown op ${op}`);
    const result = await handler(req, post);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    // Errors cross `postMessage` as strings: a wasm-bindgen `JsError` is not
    // structured-cloneable, so posting it raw would fail the send and leave the
    // host awaiting a reply that never comes.
    self.postMessage({
      id,
      ok: false,
      error: String((err && err.message) || err),
      kind: (err && err.ceraKind) || 'error',
    });
  }
};

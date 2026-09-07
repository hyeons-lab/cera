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

console.info('[cera:worker:version] v0.5.4 (build: 2026-09-04-rev24-local-models-webgpu)');

const LOCAL_MODELS_DIR = '/models-local';

function toLocalModelUrl(url) {
  if (typeof url !== 'string' || !url.startsWith('https://huggingface.co/')) return url;
  const base = url.split('?')[0].split('/').pop();
  return `${LOCAL_MODELS_DIR}/${base}`;
}

if (typeof self !== 'undefined' && self.location && (self.location.hostname === 'localhost' || self.location.hostname === '127.0.0.1')) {
  const origFetch = self.fetch.bind(self);
  self.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : (input instanceof Request ? input.url : String(input));
    const mapped = toLocalModelUrl(url);
    if (mapped !== url) {
      try {
        const localReq = input instanceof Request ? new Request(mapped, input) : mapped;
        const res = await origFetch(localReq, init);
        if (res.ok) return res;
      } catch (_) {}
    }
    return origFetch(input, init);
  };
}

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
let currentModelLabel = 'unknown';
let pendingAudioSuffixTokens = null;
let pendingImage = null;
let isCancelled = false;
let warnedSpecWebGpu = false;
let cancelSharedBuffer = null;
let cancelArray = null;
try {
  if (typeof SharedArrayBuffer !== 'undefined') {
    cancelSharedBuffer = new SharedArrayBuffer(4);
    cancelArray = new Int32Array(cancelSharedBuffer);
  }
} catch (_) {}

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
  const ids =
    typeof tk.encodeSpecial === 'function'
      ? Array.from(tk.encodeSpecial(text, false))
      : Array.from(tk.encode(text));
  const bos = tk.bosToken ?? tk.bosTokenId;
  if (first && tk.addBosToken && bos != null && ids[0] !== bos) {
    ids.unshift(bos);
  }
  return Uint32Array.from(ids);
}

let currentModuleUrl = null;

/**
 * Load the wasm module once, from a URL the host resolves.
 *
 * The URL is not derived from `import.meta.url` because the worker script and
 * the wasm artifacts need not be co-located: the script ships inside the pub
 * package and the artifacts are installed into the app's `web/` directory.
 */
async function ensureModule(moduleUrl) {
  if (wasm && currentModuleUrl === moduleUrl) return;
  if (wasm && currentModuleUrl !== moduleUrl) {
    OPS.close();
    wasm = null;
  }
  // Absolutize first. A dynamic `import()` treats a specifier with no leading
  // `./`, `../` or scheme as a BARE specifier (a package name), and rejects it
  // outright with "Failed to resolve module specifier", even though the very
  // same string works in `new Worker(...)`. Resolving against the worker's own
  // location turns any of the three forms into something importable.
  let module;
  try {
    module = await import(new URL(moduleUrl, self.location.href).href);
    await module.default();
    currentModuleUrl = moduleUrl;
  } catch (err) {
    const defaultUrl = 'cera/cera_wasm.js';
    if (moduleUrl !== defaultUrl) {
      console.warn(`[cera:worker] failed to load ${moduleUrl} (${err}), falling back to ${defaultUrl}`);
      module = await import(new URL(defaultUrl, self.location.href).href);
      await module.default();
      currentModuleUrl = defaultUrl;
    } else {
      throw err;
    }
  }
  if (typeof module.initThreadPool === 'function') {
    if (self.crossOriginIsolated) {
      try {
        const concurrency = self.navigator?.hardwareConcurrency || 4;
        await module.initThreadPool(concurrency);
        console.info(`[cera:worker] multi-threaded wasm initialized with ${concurrency} threads`);
      } catch (e) {
        console.warn(`[cera:worker] initThreadPool failed: ${e}`);
      }
    } else {
      console.warn(
        '[cera:worker] multi-threaded wasm build detected, but crossOriginIsolated is FALSE. ' +
          'COOP (Cross-Origin-Opener-Policy: same-origin) and COEP (Cross-Origin-Embedder-Policy: require-corp) ' +
          'headers are required for SharedArrayBuffer to enable multi-core execution. Running single-threaded.',
      );
    }
  } else {
    console.info('[cera:worker] wasm loaded (single-threaded WebGPU/CPU build)');
  }
  wasm = module;
}

/**
 * Try the GPU path. Returns false (rather than throwing) for every reason the
 * GPU cannot serve this model, so `auto` can fall through to the CPU.
 *
 * The failure modes are not all detectable up front: `navigator.gpu` can exist
 * while `requestAdapter` yields nothing, or an unsupported custom architecture
 * throws from `create` after WebGPU itself came up fine.
 * Both must degrade rather than fail the open.
 */
async function tryGpu(bytes, contextSize, mmproj, turboQuant) {
  if (!self.navigator || !self.navigator.gpu) return false;
  if (typeof wasm.WebGpuSession !== 'function') return false;
  // `createWithParts` is the newer of the two and is absent from a wasm build
  // predating it, so fall back rather than throwing a TypeError that `auto`
  // would then read as "no GPU" for entirely the wrong reason.
  if (mmproj && typeof wasm.WebGpuSession.createWithParts !== 'function') return false;
  const kvCompression =
    turboQuant && typeof wasm.TurboQuantConfig === 'function'
      ? new wasm.TurboQuantConfig(BigInt(0))
      : undefined;
  let session;
  try {
    session = mmproj
      ? await wasm.WebGpuSession.createWithParts(bytes, mmproj, contextSize, kvCompression)
      : await wasm.WebGpuSession.create(bytes, contextSize, kvCompression);
  } catch (_) {
    try {
      kvCompression?.free();
    } catch (_) {}
    return false;
  }
  gpu = {
    session,
    tokenizer: session.tokenizer,
    cancelHandle: typeof session.cancelHandle === 'function' ? session.cancelHandle() : null,
  };
  backendLabel = `webgpu: ${session.adapter}`;
  currentModelLabel = 'custom GGUF';
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
 * A failure *after* the download (e.g. an unsupported custom architecture) has
 * already populated the cache, so the CPU retry behind it loads from the store
 * rather than downloading again.
 */
async function tryGpuBundle(repo, bundleId, quant, contextSize, onProgress, turboQuant) {
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
  const kvCompression =
    turboQuant && typeof wasm.TurboQuantConfig === 'function'
      ? new wasm.TurboQuantConfig(BigInt(0))
      : undefined;
  let session;
  try {
    session = await wasm.WebGpuSession.fromBundleId(
      repo,
      bundleId,
      quant,
      contextSize,
      kvCompression,
      onProgress,
    );
  } catch (err) {
    try {
      kvCompression?.free();
    } catch (_) {}
    return String((err && err.message) || err);
  }
  gpu = {
    session,
    tokenizer: session.tokenizer,
    cancelHandle: typeof session.cancelHandle === 'function' ? session.cancelHandle() : null,
  };
  backendLabel = `webgpu: ${session.adapter}`;
  currentModelLabel = `${bundleId} (${quant})`;
  return null;
}

function openCpu(bytes, contextSize, mmproj, inferenceType, turboQuant, ubatchSize) {
  // `fromGgufParts` covers the text-only case too (a null projector is exactly
  // `fromGgufBytes`), so both paths resolve the inference type through one
  // constructor instead of two that have to agree.
  const engine = wasm.CeraEngine.fromGgufParts(bytes, mmproj, contextSize, inferenceType);
  initCpuSession(engine, turboQuant, ubatchSize);
  currentModelLabel = 'custom GGUF';
}

function initCpuSession(engine, turboQuant, ubatchSize) {
  const config = new wasm.SessionConfig();
  if (ubatchSize != null && ubatchSize >= 0) {
    config.ubatchSize = ubatchSize;
  }
  let tq = null;
  if (turboQuant && typeof wasm.TurboQuantConfig === 'function') {
    tq = new wasm.TurboQuantConfig(BigInt(0));
    config.kvCompression = tq;
  }
  try {
    const session = engine.newSession(config);
    cpu = { engine, session, tokenizer: engine.tokenizer, turboQuant: Boolean(turboQuant), ubatchSize };
  } finally {
    config.free();
    tq?.free();
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
async function openCpuBundle(repo, bundleId, quant, contextSize, onProgress, turboQuant, ubatchSize) {
  const engine = await wasm.CeraEngine.fromBundleId(
    repo,
    bundleId,
    quant,
    contextSize,
    onProgress,
  );
  initCpuSession(engine, turboQuant, ubatchSize);
  currentModelLabel = `${bundleId} (${quant})`;
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
  async open({ moduleUrl, bytes, mmproj, contextSize, backend, inferenceType, turboQuant, ubatchSize }) {
    OPS.close();
    await ensureModule(moduleUrl);
    const view = new Uint8Array(bytes);
    // Transferred separately from `bytes`, and absent for a text-only model.
    const proj = mmproj ? new Uint8Array(mmproj) : undefined;
    const ctx = contextSize ?? undefined;
    const type = inferenceType ?? undefined;
    if (backend === 'gpu') {
      if (!(await tryGpu(view, ctx, proj, turboQuant))) {
        throw new Error(
          'the WebGPU backend is unavailable: either this browser exposes no ' +
            'navigator.gpu, no adapter could be acquired, or the model architecture ' +
            'is unsupported on the browser GPU path. ' +
            'Use backend: auto to fall back to the CPU instead.',
        );
      }
    } else if (backend === 'cpu') {
      openCpu(view, ctx, proj, type, turboQuant, ubatchSize);
    } else if (!(await tryGpu(view, ctx, proj, turboQuant))) {
      openCpu(view, ctx, proj, type, turboQuant, ubatchSize);
    }
    return {
      backend: backendLabel,
      capabilities: capabilitiesOf(),
      cancelBuffer: cancelSharedBuffer,
    };
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
   * `BundleRepo's own default. A cached bundle re-opens without any network
   * access and fires no progress events.
   */
  async openBundle(req, post) {
    OPS.close();
    const { moduleUrl, bundleId, quant, contextSize, backend, storeDir, turboQuant, ubatchSize } = req;
    await ensureModule(moduleUrl);
    const repo = new wasm.BundleRepo(storeDir ?? undefined);
    const ctx = contextSize ?? undefined;
    // `total` is null when the server sends no Content-Length, and crosses as
    // null on the post rather than being omitted.
    const onProgress = (url, done, total) => post({ event: 'progress', url, done, total: total ?? null });
    try {
      if (backend === 'gpu') {
        const why = await tryGpuBundle(repo, bundleId, quant, ctx, onProgress, turboQuant);
        if (why != null) {
          throw new Error(
            `the WebGPU backend could not open bundle "${bundleId}": ${why}. ` +
              'Use backend: auto to fall back to the CPU instead.',
          );
        }
        return {
          backend: backendLabel,
          capabilities: capabilitiesOf(),
          cancelBuffer: cancelSharedBuffer,
        };
      }
      if (backend === 'cpu') {
        await openCpuBundle(repo, bundleId, quant, ctx, onProgress, turboQuant, ubatchSize);
        return {
          backend: backendLabel,
          capabilities: capabilitiesOf(),
          cancelBuffer: cancelSharedBuffer,
        };
      }
      // `auto`. The GPU path is tried first.
      const why = await tryGpuBundle(repo, bundleId, quant, ctx, onProgress, turboQuant);
      if (why == null) {
        return {
          backend: backendLabel,
          capabilities: capabilitiesOf(),
          cancelBuffer: cancelSharedBuffer,
        };
      }
      // `auto`. The CPU load is the fallback, but the GPU reason is the more
      // useful half of a double failure: it is the one that says whether the
      // download itself failed, so it rides along rather than being discarded.
      console.warn(`[cera:worker] WebGPU backend failed to load bundle "${bundleId}", falling back to CPU:`, why);
      try {
        await openCpuBundle(repo, bundleId, quant, ctx, onProgress, turboQuant, ubatchSize);
      } catch (err) {
        throw new Error(
          `${String((err && err.message) || err)} (the WebGPU path was tried first ` +
            `and reported: ${why})`,
        );
      }
      return {
        backend: backendLabel,
        capabilities: capabilitiesOf(),
        cancelBuffer: cancelSharedBuffer,
      };
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
   * Queued to be spliced into the proper user-turn `<|image_start|>` envelope
   * at the start of the next `generate` call.
   */
  async appendImage({ bytes, maxLongSize }) {
    const t0 = performance.now();
    const view = new Uint8Array(bytes);
    const cap = maxLongSize ?? undefined;
    console.info(
      `[cera:worker] appendImage: received ${view.byteLength} bytes of raw image data (maxLongSize cap: ${cap ?? 'model-default'})`,
    );
    pendingImage = { view, cap };
    const elapsed = (performance.now() - t0).toFixed(1);
    console.info(`[cera:worker] appendImage: queued image for envelope splicing in ${elapsed}ms`);
    return null;
  },

  /**
   * Feed mono PCM audio into the live conversation.
   */
  async appendAudio({ pcm, sampleRate, prompt }) {
    const t0 = performance.now();
    const samples = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm);
    const sr = sampleRate ?? 16000;
    console.info(
      `[cera:worker] appendAudio: processing ${samples.length} audio samples at ${sr}Hz (${(samples.length / sr).toFixed(1)}s)...`,
    );

    const tk = tokenizer();
    let markerName = '<|reserved_4|>';
    let markerId = tk.specialTokenId(markerName);
    if (markerId == null) {
      for (const candidate of ['<|reserved_5|>', '<|reserved_6|>', '<|reserved_7|>', '<|audio_start|>']) {
        const id = tk.specialTokenId(candidate);
        if (id != null) {
          markerName = candidate;
          markerId = id;
          break;
        }
      }
    }

    const currentPos = position();
    const userContent =
      prompt && prompt.trim().length > 0
        ? `${prompt.trim()} ${markerName}`
        : markerName;
    const messages = [];
    if (currentPos === 0) {
      const systemPrompt = capabilitiesOf().audioOut
        ? 'Respond with interleaved text and audio.'
        : 'Respond to the user.';
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userContent });
    let formatted;
    try {
      formatted = tk.applyChatTemplate(messages, true);
    } catch (_) {
      formatted = userContent;
    }

    const allTokens = Array.from(tk.encodeSpecial(formatted, false));
    const splitIdx = markerId != null ? allTokens.indexOf(markerId) : -1;
    let prefix = splitIdx > 0 ? allTokens.slice(0, splitIdx) : [];
    const bosId = tk.bosToken ?? tk.bosTokenId;
    if (splitIdx === -1 && prompt && prompt.trim() !== '') {
      const currentPos = position();
      prefix = Array.from(encodePrompt(prompt, currentPos === 0));
    } else if (position() === 0 && tk.addBosToken && bosId != null) {
      if (prefix.length === 0 || prefix[0] !== bosId) {
        prefix.unshift(bosId);
      }
    }
    const suffix = splitIdx >= 0 ? allTokens.slice(splitIdx + 1) : [];

    if (gpu) {
      console.info('[cera:worker] appendAudio: encoding audio frames for WebGPU KV cache...');
      if (prefix.length > 0) {
        gpu.session.appendTokens(new Uint32Array(prefix));
      }
      gpu.session.appendAudio(samples, sr);
    } else {
      console.info('[cera:worker] appendAudio: encoding audio frames on CPU session...');
      if (prefix.length > 0) {
        cpu.session.appendTokens(Uint32Array.from(prefix));
      }
      cpu.session.appendAudio(samples, sr);
    }
    pendingAudioSuffixTokens = suffix.length > 0 ? Uint32Array.from(suffix) : null;

    const elapsed = (performance.now() - t0).toFixed(1);
    console.info(`[cera:worker] appendAudio: audio framed with template prefix & embeddings seeded into KV cache in ${elapsed}ms`);
    return null;
  },

  /**
   * Transcribe mono PCM.
   */
  async transcribe({ pcm, sampleRate }) {
    const samples = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm);
    const sr = sampleRate ?? 16000;
    if (gpu) {
      console.info(
        `[cera:worker] transcribe: running ASR on WebGPU (${samples.length} samples at ${sr}Hz)...`,
      );
      const t0 = performance.now();
      const tk = gpu.tokenizer ?? gpu.session.tokenizer;
      let markerName = '<|reserved_4|>';
      let markerId = tk.specialTokenId(markerName);
      if (markerId == null) {
        for (const candidate of ['<|reserved_5|>', '<|reserved_6|>', '<|reserved_7|>', '<|audio_start|>']) {
          const id = tk.specialTokenId(candidate);
          if (id != null) {
            markerName = candidate;
            markerId = id;
            break;
          }
        }
      }
      const messages = [
        { role: 'system', content: 'Perform ASR.' },
        { role: 'user', content: markerName },
      ];
      let formatted;
      try {
        formatted = tk.applyChatTemplate(messages, true);
      } catch (err) {
        throw new Error(`failed to render ASR chat template: ${err}`);
      }
      const allTokens = Array.from(tk.encodeSpecial(formatted, false));
      const bosId = tk.bosToken ?? tk.bosTokenId;
      if (position() === 0 && tk.addBosToken && bosId != null) {
        if (allTokens.length === 0 || allTokens[0] !== bosId) {
          allTokens.unshift(bosId);
        }
      }
      const splitIdx = markerId != null ? allTokens.indexOf(markerId) : -1;

      if (splitIdx >= 0) {
        const prefix = allTokens.slice(0, splitIdx);
        if (prefix.length > 0) {
          await gpu.session.generateTokens(
            new Uint32Array(prefix),
            0,
            null,
            null,
            null,
            null,
            () => {},
          );
        }
        gpu.session.appendAudio(samples, sr);
        const suffix = allTokens.slice(splitIdx + 1);
        let text = '';
        await gpu.session.generateTokens(
          new Uint32Array(suffix),
          256,
          0.0,
          1.0,
          1,
          null,
          (piece) => {
            text += piece;
          },
        );
        const elapsed = (performance.now() - t0).toFixed(1);
        console.info(`[cera:worker] transcribe: WebGPU ASR completed in ${elapsed}ms -> "${text.trim()}"`);
        return text.trim();
      } else {
        gpu.session.appendAudio(samples, sr);
        let text = '';
        await gpu.session.generateTokens(
          new Uint32Array(allTokens),
          256,
          0.0,
          1.0,
          1,
          null,
          (piece) => {
            text += piece;
          },
        );
        const elapsed = (performance.now() - t0).toFixed(1);
        console.info(`[cera:worker] transcribe: WebGPU ASR completed in ${elapsed}ms -> "${text.trim()}"`);
        return text.trim();
      }
    }
    if (cpu) {
      const t0 = performance.now();
      const result = cpu.engine.transcribe(samples, sr);
      console.info(`[cera:worker] transcribe: CPU ASR completed in ${(performance.now() - t0).toFixed(1)}ms`);
      return result;
    }
    throw unsupported('no model loaded to transcribe');
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
    isCancelled = false;
    if (cancelArray) {
      Atomics.store(cancelArray, 0, 0);
    }
    if (cpu && cpu.session && typeof cpu.session.clearCancel === 'function') {
      try {
        cpu.session.clearCancel();
      } catch (_) {}
    }
    if (gpu) {
      if (gpu.cancelHandle && typeof gpu.cancelHandle.clearCancel === 'function') {
        try {
          gpu.cancelHandle.clearCancel();
        } catch (_) {}
      } else if (gpu.session && typeof gpu.session.clearCancel === 'function') {
        try {
          gpu.session.clearCancel();
        } catch (_) {}
      }
    }
    const { prompt, maxTokens } = req;
    const currentPos = position();
    console.info(
      `[cera:worker] generate: starting generation for "${currentModelLabel}" on "${backendLabel}" (context position: ${currentPos}, maxTokens: ${maxTokens})`,
    );
    // Seeding is per SESSION in the CPU wasm API, not per generate:
    // `GenerateOpts` has no seed field and assigning one just creates a dead JS
    // property. Honoring it therefore means rebuilding the session, which is
    // only meaningful before anything has been fed. The GPU path takes its seed
    // as a `generateTokens` argument instead, so it needs none of this.
    if (req.seed != null && currentPos === 0 && cpu) {
      const config = new wasm.SessionConfig();
      config.seed = BigInt(req.seed);
      if (cpu.ubatchSize != null && cpu.ubatchSize >= 0) {
        config.ubatchSize = cpu.ubatchSize;
      }
      let tq = null;
      if (cpu.turboQuant && typeof wasm.TurboQuantConfig === 'function') {
        tq = new wasm.TurboQuantConfig(BigInt(0));
        config.kvCompression = tq;
      }
      try {
        cpu.session.free();
        cpu.session = cpu.engine.newSession(config);
      } finally {
        config.free();
        tq?.free();
      }
    }
    let ids;
    const pendingImg = pendingImage;
    pendingImage = null;

    if (pendingImg != null) {
      const tk = tokenizer();
      const imgStartId = tk.specialTokenId('<|image_start|>') ?? tk.specialTokenId('<vision_start>');
      const imgEndId = tk.specialTokenId('<|image_end|>') ?? tk.specialTokenId('<vision_end>');
      const imgMarkerId = tk.specialTokenId('<image>') ?? tk.specialTokenId('<|image_pad|>');

      let prefixTokens = [];
      let suffixTokens = [];

      const userHeader = '<|im_start|>user\n';
      const userHeaderIdx = prompt ? prompt.lastIndexOf(userHeader) : -1;

      if (prompt && prompt.includes('<image>') && imgMarkerId != null) {
        const allTokens = Array.from(tk.encodeSpecial(prompt, false));
        const splitIdx = allTokens.indexOf(imgMarkerId);
        if (splitIdx >= 0) {
          prefixTokens = allTokens.slice(0, splitIdx);
          suffixTokens = allTokens.slice(splitIdx + 1);
        } else {
          suffixTokens = allTokens;
        }
      } else if (userHeaderIdx !== -1) {
        const prefixText = prompt.slice(0, userHeaderIdx + userHeader.length);
        const suffixText = prompt.slice(userHeaderIdx + userHeader.length);
        prefixTokens = Array.from(encodePrompt(prefixText, currentPos === 0));
        suffixTokens = Array.from(tk.encodeSpecial(suffixText, false));
      } else if (prompt && prompt.trim() !== '') {
        prefixTokens = [];
        suffixTokens = Array.from(encodePrompt(prompt, currentPos === 0));
      }

      const bosId = tk.bosToken ?? tk.bosTokenId;
      if (currentPos === 0 && tk.addBosToken && bosId != null) {
        if (prefixTokens.length === 0) {
          prefixTokens.push(bosId);
        } else if (prefixTokens[0] !== bosId) {
          prefixTokens.unshift(bosId);
        }
      }

      const t0 = performance.now();
      if (gpu) {
        if (prefixTokens.length > 0) {
          console.info(`[cera:worker] generate (VL): appending ${prefixTokens.length} prefix tokens before image...`);
          gpu.session.appendTokens(new Uint32Array(prefixTokens));
        }
        if (imgStartId != null) {
          gpu.session.appendTokens(new Uint32Array([imgStartId]));
        }
        console.info(`[cera:worker] generate (VL): encoding and seeding image embeddings into WebGPU KV cache...`);
        await gpu.session.appendImage(pendingImg.view, pendingImg.cap);
        if (imgEndId != null) {
          gpu.session.appendTokens(new Uint32Array([imgEndId]));
        }
        console.info(`[cera:worker] generate (VL): image envelope seeded in ${(performance.now() - t0).toFixed(1)}ms; generating with ${suffixTokens.length} prompt suffix tokens`);
        ids = new Uint32Array(suffixTokens);
      } else {
        if (prefixTokens.length > 0) {
          console.info(`[cera:worker] generate (VL): appending ${prefixTokens.length} prefix tokens before image...`);
          cpu.session.appendTokens(Uint32Array.from(prefixTokens));
        }
        if (imgStartId != null) {
          cpu.session.appendTokens(Uint32Array.from([imgStartId]));
        }
        console.info(`[cera:worker] generate (VL): encoding and seeding image embeddings on CPU session...`);
        cpu.session.appendImage(pendingImg.view, pendingImg.cap);
        if (imgEndId != null) {
          cpu.session.appendTokens(Uint32Array.from([imgEndId]));
        }
        console.info(`[cera:worker] generate (VL): image envelope seeded in ${(performance.now() - t0).toFixed(1)}ms; generating with ${suffixTokens.length} prompt suffix tokens`);
        ids = Uint32Array.from(suffixTokens);
      }
    } else {
      const pendingSuffix = pendingAudioSuffixTokens;
      pendingAudioSuffixTokens = null;
      if (pendingSuffix != null && (!prompt || prompt.trim() === '')) {
        ids = pendingSuffix;
        console.info(`[cera:worker] generate: using ${ids.length} pending audio suffix tokens for generation`);
      } else if (pendingSuffix != null) {
        const promptIds = encodePrompt(prompt, currentPos === 0);
        ids = new Uint32Array(pendingSuffix.length + promptIds.length);
        ids.set(pendingSuffix, 0);
        ids.set(promptIds, pendingSuffix.length);
        console.info(`[cera:worker] generate: combined ${pendingSuffix.length} audio suffix tokens and ${promptIds.length} prompt tokens`);
      } else {
        ids = encodePrompt(prompt, currentPos === 0);
        console.info(`[cera:worker] generate: prompt encoded into ${ids.length} tokens`);
      }
    }
    const started = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;
    let text = '';
    const onToken = (piece) => {
      if (isCancelled || (cancelArray && Atomics.load(cancelArray, 0) === 1)) {
        isCancelled = true;
        if (cpu?.session?.cancel) {
          try { cpu.session.cancel(); } catch (_) {}
        }
        if (gpu?.cancelHandle?.cancel) {
          try { gpu.cancelHandle.cancel(); } catch (_) {}
        } else if (gpu?.session?.cancel) {
          try { gpu.session.cancel(); } catch (_) {}
        }
        return;
      }
      tokenCount++;
      if (!firstTokenTime) {
        firstTokenTime = performance.now();
        const ttft = (firstTokenTime - started).toFixed(1);
        console.info(`[cera:worker] generate: first token emitted in ${ttft}ms (TTFT)`);
      }
      if (
        piece.includes('<|text_end|>') ||
        piece.includes('<|im_end|>') ||
        piece.includes('<|endoftext|>')
      ) {
        const clean = piece.replace(/<\|(text_end|im_end|endoftext)\|>/g, '');
        if (clean.length > 0) {
          text += clean;
          post({ event: 'token', text: clean });
        }
        return;
      }
      text += piece;
      post({ event: 'token', text: piece });
    };
    console.info(
      `[cera:worker] generate op started: model="${currentModelLabel}", backend=${gpu ? 'gpu' : 'cpu'}, maxTokens=${maxTokens}, ids=${ids.length}`,
    );
    const onAudio = req.wantsAudio
      ? (pcm, sampleRate) => {
          if (isCancelled || (cancelArray && Atomics.load(cancelArray, 0) === 1)) {
            isCancelled = true;
            if (cpu?.session?.cancel) {
              try { cpu.session.cancel(); } catch (_) {}
            }
            if (gpu?.cancelHandle?.cancel) {
              try { gpu.cancelHandle.cancel(); } catch (_) {}
            } else if (gpu?.session?.cancel) {
              try { gpu.session.cancel(); } catch (_) {}
            }
            return;
          }
          const pcmArray = new Float32Array(pcm);
          post({ event: 'audio', pcm: pcmArray, sampleRate }, [pcmArray.buffer]);
        }
      : null;
    if (gpu) {
      if (req.spec != null && !warnedSpecWebGpu) {
        console.warn(
          '[cera:worker] speculative decoding is currently only supported on the CPU WASM backend; ignoring spec options on WebGPU',
        );
        warnedSpecWebGpu = true;
      }
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
        onAudio,
      );
    } else {
      const tk = cpu.tokenizer;
      if (ids.length > 0) {
        cpu.session.appendTokens(ids);
      }
      const opts =
        typeof cpu.engine.defaultGenerateOpts === 'function'
          ? cpu.engine.defaultGenerateOpts()
          : new wasm.GenerateOpts();
      opts.maxTokens = maxTokens;
      if (req.temperature != null) opts.temperature = req.temperature;
      if (req.topP != null) opts.topP = req.topP;
      if (req.topK != null) opts.topK = req.topK;
      if (req.spec != null && typeof opts.setSpecDecode === 'function') {
        opts.setSpecDecode(req.spec.ngram, req.spec.k);
      }
      // Emit per token rather than per buffer-full; the point of a worker is
      // that the host sees output as it is produced.
      let uncommittedTokens = [];
      try {
        console.info('[cera:worker] calling cpu.session.generate...');
        cpu.session.generate(
          opts,
          (toks) => {
            if (isCancelled || (cancelArray && Atomics.load(cancelArray, 0) === 1)) {
              isCancelled = true;
              if (cpu) cpu.session.cancel();
              throw new Error('cancelled');
            }
            console.info(`[cera:worker] cpu emitted ${toks.length} tokens:`, Array.from(toks));
            for (let i = 0; i < toks.length; i++) {
              uncommittedTokens.push(toks[i]);
            }
            const decoded = tk.decode(Uint32Array.from(uncommittedTokens));
            if (decoded.endsWith('\uFFFD') && uncommittedTokens.length < 4) {
              // Incomplete multibyte UTF-8 character spanning across token chunks;
              // hold back until the completing token arrives (max 4 bytes).
              return;
            }
            if (decoded.length > 0) {
              onToken(decoded);
              uncommittedTokens = [];
            }
          },
          onAudio,
        );
        if (uncommittedTokens.length > 0) {
          const remaining = tk.decode(Uint32Array.from(uncommittedTokens));
          if (remaining.length > 0) {
            onToken(remaining);
          }
          uncommittedTokens = [];
        }
      } catch (err) {
        if (
          isCancelled ||
          (cancelArray && Atomics.load(cancelArray, 0) === 1) ||
          String(err && err.message).includes('cancelled')
        ) {
          console.info('[cera:worker] generate cancelled cleanly');
        } else {
          throw err;
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
      `[cera:worker] generate: completed ${tokenCount} tokens for "${currentModelLabel}" in ${ms.toFixed(1)}ms (${tps} tok/s)`,
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
    pendingAudioSuffixTokens = null;
    pendingImage = null;
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
   * Request an early stop for in-flight generation.
   *
   * On the CPU path, multi-threaded wasm signals cancellation via SharedArrayBuffer
   * or during token emission; on the WebGPU path, decode yields between tokens and
   * cancels cleanly via `WebGpuCancelHandle` or `gpu.session.cancel()`.
   */
  cancel() {
    isCancelled = true;
    if (cancelArray) {
      try {
        Atomics.store(cancelArray, 0, 1);
      } catch (_) {}
    }
    if (cpu) {
      try {
        cpu.session.cancel();
      } catch (_) {}
    }
    if (gpu) {
      if (gpu.cancelHandle) {
        try {
          gpu.cancelHandle.cancel();
        } catch (_) {}
      } else {
        try {
          gpu.session.cancel();
        } catch (_) {}
      }
    }
    return null;
  },

  close() {
    this.cancel();
    warnedSpecWebGpu = false;
    pendingAudioSuffixTokens = null;
    pendingImage = null;
    // The tokenizer handles are separate wasm-bindgen objects holding their own
    // `Arc<BpeTokenizer>` clone, so freeing the engine does not reclaim them.
    // Terminating the worker would, but this protocol is documented as usable
    // standalone, where open/close cycles would otherwise accumulate them.
    if (cpu) {
      cpu.tokenizer?.free();
      cpu.session?.free();
      cpu.engine?.free();
      cpu = null;
    }
    if (gpu) {
      gpu.cancelHandle?.free();
      gpu.tokenizer?.free();
      gpu.session?.free();
      gpu = null;
    }
    backendLabel = 'none';
    return null;
  },
};

self.onmessage = async (e) => {
  const req = e.data;
  const { id, op } = req;
  const post = (msg, transfer) =>
    transfer
      ? self.postMessage({ id, ...msg }, transfer)
      : self.postMessage({ id, ...msg });
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

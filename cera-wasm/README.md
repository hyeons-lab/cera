# cera-wasm

`wasm-bindgen` browser / Node bindings for the
[cera](https://github.com/hyeons-lab/cera) inference engine.

> The 0.6.2 API includes chat callback recovery, checkpoint validation and schema corrections. See the [0.6 API guide](../docs/API_0_6.md) for contracts and compatibility limits, and [Releases](https://github.com/hyeons-lab/cera/releases) for published builds.

The CPU API covers explicit model loading, metadata/capability probes,
tokenization, raw Session generation and the Chat coordinator. `Session.appendAudio`
resamples/encodes/prefills audio when an audio-capable model and its encoder
companion have been loaded, for example through `CeraEngine.fromGgufParts`.
The async browser `WebGpuSession` surface is separate; see
[GPU acceleration](#gpu-acceleration-experimental-webgpu).

## Explicit CPU model loading

This checkout exports `ModelSource`, `ModelLoader`, `ModelHandle` and
`GenerativeModel` alongside the existing engine, browser factories and WebGPU API.
The loader accepts owned
GGUF bytes or `ModelParts` containing companion bytes, inference type, chat
template and complete Text/Audio/Other generation defaults.

For a module generated with `wasm-bindgen --target nodejs`:

```javascript
const loader = new api.ModelLoader(
    api.ModelSource.bytes(bytes), new api.LoadConfig(4096, 'cpu'));
const model = loader.buildGenerative();
const engine = model.engine();
const config = new api.SessionConfig();
config.seed = 42n;
const session = model.createSession(config);
config.free();
// Session resources survive the release of the loading and engine handles.
loader.free();
model.free();
const tokenizer = engine.tokenizer;
engine.free();
try {
    session.appendTokens(tokenizer.encode('The capital of France is'));
    const options = new api.GenerateOpts();
    Object.assign(options, {maxTokens: 32, temperature: 0.7});
    try {
        const tokens = [];
        const summary = session.generate(options, batch => tokens.push(...batch));
        summary.free();
        console.log(tokenizer.decode(new Uint32Array(tokens)));
    } finally { options.free(); }
} finally {
    tokenizer.free();
    session.free();
}
```

The [complete Node example](examples/explicit_loading.cjs) includes module/file
loading and cleanup for partial construction failures. Run it from the repo root:

```sh
cargo build -p cera-wasm --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir /tmp/cera-wasm-node \
  target/wasm32-unknown-unknown/debug/cera_wasm.wasm
node cera-wasm/examples/explicit_loading.cjs \
  /tmp/cera-wasm-node/cera_wasm.js model.gguf "The capital of France is"
```

Loading is synchronous; use a worker in a browser UI. Source/config handles move
into `ModelLoader` and must not be reused or freed after that transfer. Both build
methods consume the loader's source even on failure; subsequent attempts throw
an Error with `code === 'Consumed'`. Other loading errors carry structured `code`
and variant-specific properties. Session methods retain their existing Error
messages. This is raw prompt completion without a chat template. Reuse a Session
for live KV continuation; the example does not establish performance budgets.

`LoadConfig` and multipart/default fields retain their existing probe spelling;
`buildGenerative`, `asGenerative`, `createSession` and `toJson` follow the production
JavaScript method style. `ModelSource.bytes` and `.parts` are the new CPU sources.
Existing async browser resolution and WebGPU loading use their existing APIs.

## CPU chat lifecycle

`session.intoChat()` transfers the CPU Session into a `ChatSession`.
`chat.phase` and `chat.position` are properties, not methods. Ingest messages as
`{role: 'user', content: '...'}` objects, then call `complete(opts)` or
`generateStreaming(opts, onText)`. Returned ingestion summaries and turn results
are WASM handles; free them when no longer needed.

The phases are `Idle`, `PromptReady`, `TurnComplete`, `Interrupted`, `RawContext`
and `Unusable`. Only `TurnComplete` permits the next ordinary user turn. A token
limit or cancelled call can return successfully with phase `Interrupted`;
reset or replace messages before adding another turn. `clearCancel()` only clears
the flag. Zero-token/no-progress calls can preserve `PromptReady`, allowing a
generation retry without replay. `intoSession()` reclaims execution and leaves
the old Chat handle moved.

`completeJson(opts, schema)` and `generateStreamingJson(opts, schema, onText)`
compile the [JSON Schema subset](../docs/API_0_6.md#json-schema-constraints).
Unimplemented validation keywords are not enforced, and a token limit can leave
incomplete JSON. Do not re-enter the same Chat handle from a streaming callback,
including `cancel()`, property reads, reset or disposal: recursive WASM borrowing
can leave the handle unusable. Ordinary JS values thrown by a Chat callback
return an error after Rust releases its borrow; inspection/reset remains possible
after the outer call returns. This guarantee excludes recursive handle access
and does not apply to every raw Session callback API.

CPU Session and Chat expose `checkpoint()` and `restore(bytes)`. These differ
from the separate browser `WebGpuSession` API below; native Metal/wgpu Session
checkpointing is unsupported. See the [checkpoint matrix](../docs/API_0_6.md#checkpoints-and-compatibility).

## Install

```sh
npm install @hyeons-lab/cera-wasm
```

For now, download the artifact matching your consumer shape from
the latest [CI run](https://github.com/hyeons-lab/cera/actions/workflows/ci.yml)
on `main`; three are produced per build:

| Artifact | wasm-pack target | When to use |
|---|---|---|
| `cera-wasm-pkg-bundler` | `bundler` | webpack 5+, Vite, Rollup (with wasm plugin), Parcel: the typical app build |
| `cera-wasm-pkg-web` | `web` | `<script type="module">` direct in the browser, or any bundler-less ESM workflow |
| `cera-wasm-pkg-nodejs` | `nodejs` | `require('@hyeons-lab/cera-wasm')` from CommonJS Node, or older Node without ESM `import` |

```sh
npm install /path/to/downloaded/pkg-bundler  # or pkg-web / pkg-nodejs
```

> **One npm package, three target shapes:** all three artifacts use
> `package.json.name` = `@hyeons-lab/cera-wasm`. When its npm job is selected,
> the [publish workflow](../.github/workflows/publish.yml) publishes only the
> **bundler** target under that name. Install the `web` and `nodejs` shapes
> from their CI artifacts; separate npm package names are not configured.

## Usage

The examples below assume the **`bundler`** target. The `web`
target also needs a one-time `await init()` call before the first
export; see the
[wasm-pack docs](https://rustwasm.github.io/docs/wasm-pack/tutorials/npm-browser-packages/getting-started.html)
for the init pattern. The `nodejs` target does **not** need an
explicit init: `require('@hyeons-lab/cera-wasm')` returns a ready
module (the entry self-loads the wasm via `fs.readFileSync`).

### Manifest parsing

```js
import { ceraVersion, Manifest } from '@hyeons-lab/cera-wasm';

console.log(ceraVersion());  // e.g. "0.4.0"

const res = await fetch('/path/to/manifest.json');
const bytes = new Uint8Array(await res.arrayBuffer());
const manifest = Manifest.parse(bytes);

console.log(manifest.inferenceType);   // "llama.cpp/text-to-text"
console.log(manifest.modelUrl);        // "https://.../model.gguf"
console.log(manifest.schemaVersion);   // "1.0.0"
```

### Loading a model + tokenizing

```js
import { CeraEngine } from '@hyeons-lab/cera-wasm';

// Fetch the GGUF (use the `modelUrl` from a parsed manifest, or a
// direct URL).
const res = await fetch('/path/to/model.gguf');
const bytes = new Uint8Array(await res.arrayBuffer());

// Construct the engine. Optional `contextSize` defaults to 4096.
// Backend is forced to CPU on wasm.
const engine = CeraEngine.fromGgufBytes(bytes, 2048);

console.log(engine.architecture);     // "lfm2"
console.log(engine.maxSeqLen);        // 2048 (clamped to min(contextSize, gguf max))
console.log(engine.contextSize);      // 2048: what you passed (or 4096 if omitted)
console.log(engine.vocabSize);
console.log(engine.quantization);     // "Q4_0", "Q8_0", "BF16", etc.
console.log(engine.hasChatTemplate);  // true / false
console.log(engine.addBosToken);      // honor when hand-building token sequences

// Modality capability probe. Plain JS object; no .free() needed,
// destructurable. Today every model loaded via fromGgufBytes
// reports text-only because wasm uses cera's synthetic-text
// manifest path; a model-aware loader (planned) will surface real
// audioIn / imageIn flags.
const { textIn, audioIn, audioOut, imageIn } = engine.capabilities;

// Tokenize a string.
const tok = engine.tokenizer;
const ids = tok.encode('hello world');  // Uint32Array
console.log(ids);
console.log(tok.decode(ids));           // round-trips to "hello world"

// Look up a control token by literal vocab name (only tokens
// flagged with token_type 3 / 4 in GGUF metadata are reachable).
const imStart = tok.specialTokenId('<|im_start|>');  // number | undefined

// Filter control tokens from a streamed batch before rendering
// to UI; keeps `<|im_end|>` etc. out of the displayed text.
const visible = ids.filter(id => !tok.isSpecialToken(id));

// Render the GGUF-embedded Jinja chat template against a message
// list. `addGenerationPrompt` defaults to `true` for the
// "send-to-model-and-await-response" case.
if (engine.hasChatTemplate) {
    const prompt = tok.applyChatTemplate([
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user',   content: 'Hello!' },
    ]);
    console.log(prompt);
}

// Release the model bytes from wasm memory when done.
engine.free();
```

> **Memory note:** `CeraEngine` keeps the entire GGUF resident in
> wasm linear memory. Always `engine.free()` (or use the
> `[Symbol.dispose]()` pattern with `using` in TC39 explicit
> resource management) when you're done; otherwise the model
> stays alive until the page unloads.

### Inference (text)

```js
import { CeraEngine, SessionConfig, GenerateOpts } from '@hyeons-lab/cera-wasm';

const engine = CeraEngine.fromGgufBytes(gguf, 2048);
const tok = engine.tokenizer;
// `newSession` requires a SessionConfig. Pass `new SessionConfig()`
// for the cera defaults (random sampler seed, no n_keep pin, etc).
// See "Reproducibility (seeded sampler)" below for the knob list.
const session = engine.newSession(new SessionConfig());

// Seed the conversation. Use `session.appendText(prompt)` for the
// common case (tokenizer is invoked internally), or
// `session.appendTokens(ids)` when you need control over BOS/EOS.
session.appendText('Hello, what is the capital of France?');

// Configure decoding. All fields default to cera's native defaults
// (max 256 tokens, temperature 0.7, top-p 0.9, top-k 40, no stops,
// flush every 16 tokens or 50 ms).
const opts = new GenerateOpts();
opts.maxTokens = 64;
opts.temperature = 0.0;
// `tok.eosToken` is `number | undefined`. `new Uint32Array([undefined])`
// silently coerces to `0`, which would stop decoding the moment
// token 0 is produced. Always guard the lookup.
if (tok.eosToken != null) {
    opts.stopTokens = new Uint32Array([tok.eosToken]);
}

// Stream tokens as they decode. The callback fires per flush
// boundary (every `flushEveryTokens` decoded tokens, OR every
// `flushEveryMs` ms, whichever hits first) with just the *new*
// tokens, not the cumulative buffer.
let acc = [];
const summary = session.generate(opts, (newTokens) => {
    acc.push(...newTokens);
});
// Decode once so a UTF-8 sequence split across batches stays intact.
console.log(tok.decode(acc));

console.log('\n---');
console.log('finish:', summary.finishReason);          // "Stop" | "MaxTokens" | ...
console.log('tokens:', summary.tokensGenerated);
console.log('decode ms:', summary.decodeMs);

session.free();
engine.free();
```

### Sampling knobs & constrained decoding

Beyond the fields shown above, `GenerateOpts` also exposes `minP` and
`repetitionPenalty`. Both apply on the **stochastic path only**; greedy/argmax
decoding (selected by a temperature of `0` **or** a top-k of `1`) ignores them:

```js
const opts = new GenerateOpts();
opts.temperature = 0.8;
opts.minP = 0.05;               // drop tokens below 5% of the top token's prob
opts.repetitionPenalty = 1.1;   // penalize already-emitted tokens
```

Constrain output to a GBNF grammar (e.g. force valid JSON). Unlike the
other knobs this is a **method**, not a settable property; `setGrammar`
can throw if the grammar fails to compile, which a plain setter can't
surface:

```js
const opts = new GenerateOpts();
opts.setGrammar('root ::= "{" [a-z]+ "}"');  // throws on a malformed grammar
opts.hasGrammar;       // → true (getter: property access, no parens)
// opts.clearGrammar(); // back to unconstrained decoding
```

Each decode step then masks the logits so only grammar-accepted tokens
are sampled; a later `setGrammar` replaces any grammar set by a prior call.

### Tool calling

Render a set of tools into the prompt, generate, then parse the calls back out.
The wire format is per-model-family; detect it from the GGUF architecture
(`detectToolFormat`) or pick one explicitly (`ToolFormat.Lfm2Pythonic` for
LFM2/LFM2.5, `ToolFormat.Hermes` for Qwen2.5/Qwen3). Tools cross the JS boundary
as a JSON string; `parseToolCalls` returns a JSON string you `JSON.parse`.

```js
import {
  SessionConfig, GenerateOpts,
  detectToolFormat, ToolFormat, toolGrammar, parseToolCalls,
} from '@hyeons-lab/cera-wasm';

// `engine` is the CeraEngine from the setup above.
const tok = engine.tokenizer;
const tools = JSON.stringify([{
  name: 'get_weather',
  description: 'Get the current weather for a city',
  parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
}]);

const format = detectToolFormat(engine.architecture) ?? ToolFormat.Lfm2Pythonic;
const prompt = tok.applyChatTemplateWithTools(
  [{ role: 'user', content: "What's the weather in Paris?" }], tools, true);

const session = engine.newSession(new SessionConfig());
session.appendText(prompt);

const opts = new GenerateOpts();
opts.maxTokens = 128;
// Optional: constrain tool-call syntax (grammar + lazy start-marker trigger).
// The start marker differs by format; it must be a special token in the model's
// vocab for the trigger to fire (LFM2's is; Hermes markers usually aren't, so
// this stays unconstrained there; validate the generated output).
const startMarker = format === ToolFormat.Hermes ? '<tool_call>' : '<|tool_call_start|>';
const trigger = tok.specialTokenId(startMarker);
if (trigger != null) {
  opts.setGrammar(toolGrammar(tools, format));   // separate tool-grammar subset
  opts.grammarTriggerTokens = new Uint32Array([trigger]);
}

// generate streams the new tokens per flush; accumulate them.
const acc = [];
session.generate(opts, (newTokens) => acc.push(...newTokens));
const reply = tok.decode(acc);
const calls = JSON.parse(parseToolCalls(reply, format));  // [{ name, arguments }]
for (const c of calls) console.log(c.name, c.arguments);
```

`parseToolCalls` returns `[]` (as `"[]"`) when the model answered in prose. With
the lazy trigger, generation stays unconstrained until the model emits the
start marker, then the grammar constrains function/argument names and outer value
syntax. Required arguments, duplicates and nested item/property schemas are not
validated. A call can still be truncated; parse and validate arguments before
execution. See the [tool grammar limits](../docs/API_0_6.md#tool-call-constraints).

### LoRA adapters & hidden states

Load a LoRA adapter from bytes (GGUF or PEFT `.safetensors`) and attach it to a
session; applied at inference time, never merged, so it hot-swaps freely (and
detaches on demand with `session.removeLora()`). Then pull per-token hidden
states out of the engine (reflecting the active adapter) for classifier /
embedding heads.

```js
import { LoraAdapters } from '@hyeons-lab/cera-wasm';

// Pass your PEFT adapter's `lora_alpha` (from its adapter_config.json) as the
// 2nd arg; `undefined` ⇒ alpha defaults to the rank (scale = 1). The loader
// does not read adapter_config.json for you.
const adapters = LoraAdapters.fromSafetensorsBytes(safetensorsBytes, undefined);
// ...or LoraAdapters.fromGgufBytes(ggufBytes)
session.attachLora(adapters);          // hot-swap-able; session.removeLora() to detach
session.hasLora();                     // → true

const tokens = engine.tokenizer.encode('a transcript chunk'); // Uint32Array
const pooled = session.hiddenStatesMeanPooled(tokens);  // Float32Array, length hiddenSize
const perToken = session.hiddenStatesForTokens(tokens); // Float32Array, tokens.length * hiddenSize

adapters.free();  // frees the JS handle; the attached copy stays live in the session
```

To stack several adapters with per-adapter runtime scales, build a
`LoraStack` and install it with `setLoraAdapters` (contributions stack per
target; the swap is atomic, so a bad stack leaves the previous set
untouched). An empty stack detaches, the same as `removeLora()`:

```js
import { LoraStack } from '@hyeons-lab/cera-wasm';

const stack = new LoraStack();
stack.push(styleAdapters, 0.8);
stack.push(taskAdapters, 1.0);
session.setLoraAdapters(stack);
// ...when done with the stack itself (the session holds its own copy):
stack.free();
```

For a one-shot extraction through a different stack without touching the
session set, pass the stack to a per-call override instead (an empty stack
extracts from the base model even when the session has adapters attached):

```js
const pooled = session.hiddenStatesMeanPooledWithAdapters(tokens, stack);
// ...or hiddenStatesForTokensWithAdapters / hiddenStatesForTextWithAdapters
```

### Reproducibility (seeded sampler)

Pass a `SessionConfig` with a fixed `seed` to `newSession` so the
sampler RNG is deterministic across runs. Same seed + same prompt
+ same `GenerateOpts` → identical token sequence:

```js
import {
    CeraEngine,
    SessionConfig,
    GenerateOpts,
    TurboQuantConfig,
} from '@hyeons-lab/cera-wasm';

const cfg = new SessionConfig();
cfg.seed = 42n;        // BigInt: wasm-bindgen maps Rust u64 to JS BigInt

// You can also tune:
//   cfg.maxSeqLen = 1024;   // further lower the KV cap below the
//                           // engine's effective max
//                           // (= min(engine.contextSize, model.maxSeqLen)).
//                           // Setting a value above that effective max
//                           // does NOT raise it; re-construct the engine
//                           // with a larger contextSize for that.
//   cfg.nKeep = 16;         // pin the first 16 tokens across context shifts
//                           // (useful for keeping a system prompt resident)
//   cfg.ubatchSize = 256;   // smaller chunked-prefill batches give finer
//                           // session.cancel() checkpoints during long prompts

// Optional: turn on TurboQuant KV compression. Compresses keys to
// ~3 bits/elem and values to ~2 bits/elem (plus a norm word per
// vector). Pass an explicit seed so the per-layer Hadamard
// rotations are reproducible; paired with `cfg.seed` above this
// keeps the whole session bitwise-identical across runs.
//
// Caveats:
// - Only kicks in when the model's `head_dim` is a power of two.
//   cera logs a warning and falls back to f32 if not; no JS
//   error.
// - Applies to `engine.newSession(cfg)` only. `WebGpuSession`
//   takes no `SessionConfig`; pass a `TurboQuantConfig` as the
//   third argument of `WebGpuSession.create` instead, and read
//   `session.kvCompression` for the mode that took effect.
// - Don't combine with `cfg.nKeep > 0` (context-shift); cera
//   warns at session creation and ignores nKeep on overflow.
const tq = new TurboQuantConfig(1234n);  // ctor sets keys + values = true
// tq.keys = false;  // flip per-side toggles for debugging
cfg.kvCompression = tq;  // setter consumes `tq`; read back via getter to inspect

const session = engine.newSession(cfg);
session.appendText('once upon a time');
const opts = new GenerateOpts();
opts.maxTokens = 16;
opts.temperature = 1.0;  // non-greedy so the seed actually matters
const out = [];
session.generate(opts, (toks) => out.push(...toks));
// `out` is identical for any session built with the same seed + prompt + opts.
```

Two finer-grained knobs sit on top of the session seed. `opts.seed` is a
per-request override: it restarts the sampler RNG when that one call starts
(KV and position are untouched, so it is safe mid-conversation) without
changing the session default. `session.setSeed(seed)` replaces the
persistent default instead (and restarts the RNG immediately), surviving
`reset()`; pass `undefined` to re-seed from entropy:

```js
opts.seed = 7n;          // this call only; default `undefined` continues the stream
session.setSeed(1234n);  // new default from here on, reset()-proof
```

> **Worker note:** `Session.generate` is **synchronous** and blocks
> the thread it runs on for the full decode duration (potentially
> seconds). On the browser main thread that freezes the page;
> always call from a Web Worker:
>
> ```js
> // worker.js
> import { CeraEngine, SessionConfig, GenerateOpts } from '@hyeons-lab/cera-wasm';
> self.onmessage = async (ev) => {
>     const engine = CeraEngine.fromGgufBytes(ev.data.gguf);
>     const session = engine.newSession(new SessionConfig());
>     session.appendText(ev.data.prompt);
>     const opts = new GenerateOpts();
>     opts.maxTokens = 128;
>     session.generate(opts, (toks) => self.postMessage({ kind: 'tokens', toks }));
>     self.postMessage({ kind: 'done' });
> };
> ```
>
> On Node the sync call also blocks the JS event loop; libuv's
> background I/O thread pool keeps running, but JS callbacks (HTTP
> handlers, timers, etc.) queue up and don't fire until generate
> returns. For server processes that need to handle other requests
> during inference, run generate inside a `worker_threads` Worker.
> For one-off scripts the block is fine.

### Cancellation

CPU `Session.generate()` and `ChatSession.generateStreaming()` are synchronous
and hold a mutable WASM borrow until they return. Do not call methods or read
properties on the same handle from their callbacks, including `cancel()`.
Recursive access raises a borrow error and can leave the handle unusable. Raw
Session callbacks must also avoid throwing; they do not have Chat's ordinary
callback-error recovery.

Call CPU cancellation controls between operations. A flag set before generation
is observed by the next call; `clearCancel()` clears it before a retry. During
generation, the worker's own message handlers cannot run. Polling a
`SharedArrayBuffer` in a callback does not make a recursive `session.cancel()`
safe, and the direct CPU bindings expose no independent cancellation handle.

Set `opts.maxTokens` before generation to bound decode work. If an application
must stop a blocked CPU worker immediately, terminate the worker and recreate
its engine/session; termination discards its in-memory execution state. A token
limit can leave Chat `Interrupted`, so apply the lifecycle rules before a new
user turn.

The separate async browser `WebGpuSession` exposes `cancelHandle()`. Obtain that
handle before starting generation and call its `cancel()` method to signal the
operation without borrowing the active session. Clear it before reuse and free
the cancellation handle when it is no longer needed.

### Resuming after cancel vs starting over

After a cancellation lands (`finishReason === "Cancelled"` from
`generate`, or a thrown error from `appendText` / `appendTokens`),
two primitives let JS callers continue with the same `Session`
without paying `engine.newSession(config)` setup cost again:

| API | KV cache | `position` | Sampler | When to use |
|---|---|---|---|---|
| `session.clearCancel()` | preserved | preserved | preserved | "interrupted but continuing": keep the conversation context, append more tokens, generate again |
| `session.reset()` | dropped | reset to 0 | re-seeded from the session default (the `setSeed` value when one was set, else `cfg.seed`) | "clear conversation" UI button: start fresh |

`clearCancel()` takes `&self`, `reset()` takes `&mut self`;
remember `reset()` must be invoked outside any in-flight `generate`
(wasm-bindgen's borrow check rejects re-entry).

```js
try {
    session.appendTokens(longPrompt);  // may throw "cancelled" mid-prefill
} catch (e) {
    if (String(e).includes('cancelled')) {
        session.clearCancel();           // resume without losing KV
        session.appendTokens(remainingTokens);
    } else {
        throw e;
    }
}
```

Bundlers without native wasm support need a loader plugin; see the
[`wasm-pack` bundler guide](https://rustwasm.github.io/docs/wasm-pack/tutorials/npm-browser-packages/getting-started.html)
for webpack / Rollup / Parcel specifics.

For no-bundler workflows (`<script type="module">` directly in the
browser, or `require('@hyeons-lab/cera-wasm')` from CommonJS Node),
download the `cera-wasm-pkg-web` or `cera-wasm-pkg-nodejs` artifact
instead; see the `Install` section above for the per-target
table.

## GPU acceleration (experimental WebGPU)

An **experimental** WebGPU-backed session lights up when the crate is
built with the `wgpu` cargo feature (which pulls in `cera/gpu`). Its API is
separate from the CPU `Session`: WebGPU can't do blocking GPU readback on
the JS event loop, so the whole prefill + decode loop is `async`. The
default builds above are CPU-only; the standard `Session` remains the
supported path.

The WebGPU loader admits `lfm2`, `lfm2moe`, `llama`, `qwen2`, `qwen3`, `granite`,
`minicpm`, `minicpm5`, `nanbeige`, `phi3` and `phi`, subject to supported tensor
layouts and device limits. Generation supports greedy decoding and stochastic
sampling. Pass temperature, top-p, top-k and seed before the token callback;
use `undefined` for a sampling default. A seed is a JavaScript `bigint`.

```js
import { WebGpuSession, TurboQuantConfig } from '@hyeons-lab/cera-wasm';

// Async constructor: initializes WebGPU (requestAdapter / requestDevice
// resolve on the event loop), parses the GGUF, and uploads the model to
// the GPU. `contextSize` defaults to 4096. Throws if WebGPU is
// unavailable, the GGUF layout is unsupported, or device initialization fails.
//
// The optional third argument requests TurboQuant on the GPU-resident KV
// cache (~3-bit keys / ~2-bit values). Omit it, or pass null, for
// uncompressed f32 KV; existing two-argument calls are unaffected. Like
// the `SessionConfig` setter, `create` CONSUMES the config handle. Build a
// fresh one per session: a reused handle does NOT throw in a release build,
// it lowers to a null pointer that Rust reads as "no compression", so the
// second session silently runs uncompressed.
const session = await WebGpuSession.create(ggufBytes, 2048, new TurboQuantConfig(1234n));

// To request TurboQuant but keep the default contextSize, pass an explicit
// placeholder for argument 2: `undefined` and `null` both work (the generated
// signature is `context_size?: number | null`):
//
//   await WebGpuSession.create(ggufBytes, undefined, new TurboQuantConfig(1234n));
//
// Do NOT collapse it to `create(ggufBytes, tq)`. TypeScript rejects that, but
// plain JS does not: in a release build the config object is coerced by
// `>>> 0` to a contextSize of 0 and the compression argument goes missing, so
// you get an unusable session with no compression and no error. (A `--dev`
// build throws instead, so this only bites in release.)

// Adapter + backend description: confirms the GPU path is live.
console.log(session.adapter);  // e.g. "<adapter> (BrowserWebGpu)"

// The mode that ACTUALLY took effect: "turboquant(seed=N)" or
// "uncompressed". Three things silently downgrade to f32: a head_dim that
// isn't a power of two <= 128 and a multiple of 32; a single-sided config
// (the WebGPU kernels only compress keys AND values together, so the
// `tq.keys = false` debug toggle above falls back here); and reusing an
// already-consumed config handle, as noted above. None of them reach the
// browser console, so this getter is the only way to tell.
console.log(session.kvCompression);

// Greedy generate. `onToken(text)` fires per decoded piece as it's
// produced; the full string is also returned. Stateful: the on-GPU KV
// cache persists across calls, so a second `generate()` continues the
// same sequence rather than restarting.
const text = await session.generate('The capital of France is', 32, 0, undefined, undefined, undefined, (piece) => {
    outputEl.textContent += piece;  // stream into the DOM (browser-safe)
});

// Checkpoint VRAM state: asynchronously snapshot KV and conv buffers to binary bytes.
const checkpointBytes = await session.checkpoint();

// Restore synchronously: validate structural fingerprint, row geometry, sequence
// length, KV precision and compression identity, including the TurboQuant seed.
// CPU f16 checkpoints are rejected. The fingerprint is not a hash of model weights.
session.importCheckpoint(checkpointBytes);
```

Checkpoint compatibility uses the effective `session.kvCompression` mode after
fallback. Recreate older TurboQuant snapshots whose fingerprints omitted
compression identity. CPU f16 snapshots are not importable into WebGPU; plain
f32 fingerprints are unchanged. Full model/checkpoint parity requires testing
with the intended model/backend; a WebGPU device/readback smoke test alone does
not establish it.

## Building from source

```sh
just wasm        # bundler target → cera-wasm/pkg-bundler/
just wasm-web    # browser ESM    → cera-wasm/pkg-web/
just wasm-node   # CommonJS Node  → cera-wasm/pkg-nodejs/
```

Requires `wasm-pack` (`cargo install wasm-pack`) and `wasm-opt`
(macOS: `brew install binaryen`; linux: `apt-get install binaryen`).

## Multi-threaded build

Threaded variants light up `cera`'s rayon paths (batched prefill
GEMM, parallel GEMV row sweeps, dequant_rows_to_f32) on wasm via
`wasm-bindgen-rayon`. The generated package surfaces an
`initThreadPool(numThreads)` JS export that callers `await` once
before driving inference.

```sh
just wasm-web-mt    # browser ESM + threads → cera-wasm/pkg-web-mt/
just wasm-node-mt   # CommonJS Node + threads → cera-wasm/pkg-nodejs-mt/
```

Both recipes set the atomics target-feature, the
shared-memory/import-memory link args, and `-Z build-std=panic_abort,std`
together. Single-threaded `wasm`/`wasm-web`/`wasm-node` recipes are
unchanged. `--target bundler` is intentionally not provided;
`wasm-bindgen-rayon` doesn't have canonical bundler-side worker glue.

Extra prerequisite (single-threaded builds don't need this): the
`rust-src` rustup component for `-Z build-std`:

```sh
rustup component add rust-src
```

### Browser usage

```js
import init, { initThreadPool, CeraEngine } from './cera_wasm.js';

await init();
await initThreadPool(navigator.hardwareConcurrency);
// ...drive inference normally; rayon paths now run on the worker pool
```

The host page **must** be served with cross-origin isolation headers
so the browser hands out a real `SharedArrayBuffer`:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without these `initThreadPool` rejects with a `SharedArrayBuffer is
not defined` error. Most static-file dev servers don't set them by
default; see the `wasm-bindgen-rayon` README for snippets covering
Vite, webpack-dev-server, `http-server`, and similar tools.

### Node usage

`pkg-nodejs-mt/` ships as a CommonJS module, so top-level `await`
isn't available; wrap the init in an async IIFE (or run the
snippet from a `.mjs` / `"type": "module"` ESM file):

```js
const { initThreadPool, CeraEngine } = require('@hyeons-lab/cera-wasm');
const os = require('os');

(async () => {
    await initThreadPool(os.cpus().length);
    // ...drive inference normally; rayon paths now run on worker_threads
})();
```

Node has no equivalent of the browser's COOP/COEP gate; the
threaded build runs out-of-the-box on `worker_threads`.

## License

Apache-2.0 OR MIT, matching the rest of the cera workspace.

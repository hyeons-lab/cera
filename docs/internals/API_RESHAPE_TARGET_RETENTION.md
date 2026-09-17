# API retention by binding target

Plan30 records the loading promotion contract at `2477a68` plus the completed
increments through Plan29. This document selects the next implementation's
signatures; it does not publish them or close P0-L. The
[loading exit audit](API_RESHAPE_LOADING_EXIT.md) and
[current handoff](API_RESHAPE_HANDOFF.md) distinguish design evidence from shipping.

## Supported homes and target boundaries

All existing entry points in the inventory remain supported during additive
loading. An entry point absent from a target is not silently added by this map.
No deprecation follows from a generative-only `ModelHandle`.

| Target | Existing API home | Loading candidate evidence | Promotion requirement |
|---|---|---|---|
| Rust core | `cera::CeraEngine`, `Session`, raw `Model`/tokenizer, repository and standalone loaders | Private core loader; memory/files/remote/companion/cache and ownership fixtures | Add public loader/model types alongside existing types, with the same feature gates and retained engine ownership |
| Swift and Kotlin native | `cera-ffi` UniFFI objects; shipped Swift/SPM and Kotlin generated bindings | Isolated generated candidate on macOS arm64, 39 cases per language | Reuse production `EngineConfig`, `BundleRepo`, `CeraEngine`, `Session` and their existing value types; prove shared access through the new model |
| Python native | Generated `cera-ffi` Python module | Same UniFFI source declarations; no Plan29 Python loader consumer | Generate and compile/run the selected additive Python surface in P1 before claiming it |
| Dart native/Flutter | Generated UniFFI Dart layer plus the handwritten async `Cera` wrapper | Existing native GPU/Whisper/async work has separate evidence; no Plan29 Dart loader consumer | Preserve generated and handwritten surfaces; regenerate/check Dart and test the additive caller before migration |
| JavaScript/TypeScript CPU WASM | `cera-wasm` `CeraEngine`, `Tokenizer`, `Session`, records and browser repository | Node CPU byte/part candidate, 21 cases | Expose shared existing CPU engine/session wrappers; preserve JS names, typed arrays, config accessors and exception shapes |
| Browser WebGPU | `WebGpuSession`, `WebGpuCancelHandle`, OPFS repository/weight source | Existing native wgpu tests do not execute this browser API | Retain the existing async factories and cancel handle; require browser execution before offering a typed WebGPU replacement |
| Dart web | Worker-backed `Cera` plus generated web bridge | Node is not a Dart worker/browser test | Preserve worker initialization, transfer/queue/cancellation behavior; migrate with real worker/browser checks |

Swift/Kotlin host execution does not establish iOS, Android arm64/armv7/x86_64
packaging or JVM/native threading on a device. Node does not establish browser
OPFS, WebGPU or worker behavior. Those are P1/P2 checks for the affected shipment,
not results inferred from source declarations.

## Constructor and source retention

The complete callable signatures for reviewed existing homes are in the
[declaration baseline](../../tests/api_contracts/retained_api.json).
The [checker](../../tests/api_contracts/README.md) reports changes by owner and
method, including changed argument/return types and async removal.

| Input or factory | Rust core | Existing native binding | Existing CPU WASM / WebGPU | Additive decision |
|---|---|---|---|---|
| GGUF file, JSON manifest, directory | `from_path`; `mmap` | `from_path` and `from_path_async` | No native filesystem path factory | Native `ModelSource::Path`; preserve exactly-one-manifest and existing path resolution |
| Primary bytes | `from_bytes` | `from_bytes` and `from_bytes_async` | `CeraEngine.fromGgufBytes`; WebGPU `create` | Native/CPU WASM bytes source; keep old factories and source ownership |
| Reader | `from_reader<R: Read>` | Not exported | Not exported | Rust-only synchronous reader; no Send/Seek/Clone requirement |
| Multipart files | `from_files(ModelFiles)`; `mmap` | No current files constructor | No current files constructor | Native files source adds the full eight-field record; retain path-specific extras |
| Multipart bytes | `from_parts(ModelBytes)` | `from_parts` and `from_parts_async` | `CeraEngine.fromGgufParts`; WebGPU `createWithParts` | Full parts source for native/CPU WASM; do not claim WebGPU's narrower projector factory carries every multipart field |
| Bundle ID | `from_bundle_id`; `remote+mmap` | `from_bundle_id` and `from_bundle_id_async` | CPU `fromBundleId` and WebGPU `fromBundleId`, both async | Native bundle source shares the configured repository; browser factories retain their distinct repository/progress contracts |
| HF spec/URL + quant/strategy | `from_hf`, `from_hf_with_strategy`, `from_hf_url`; `remote+mmap` | No current direct HF factory | No corresponding CPU/WebGPU HF factory | Native HF source delegates to existing resolver; browser remote work remains on its supported factories |
| Explicit remote manifest | Rust `from_path` loads a local manifest that can reference remote assets | Same path/async-path behavior with repository | CPU `fromManifestUrl` is async | Preserve browser manifest URL loading; a native Path is not a replacement for it |

The native async constructors retain their exact existing payloads, `FfiError`
behavior and execution on blocking workers. Dropping their future may abort queued
work; already running blocking model loading is not generally cancellable.
The new synchronous builder does not replace any of these entry points. Designing
an additive async builder is separate from keeping the old async APIs callable.
Swift's existing generated task-cancellation limitation remains open; preserving
an async signature does not prove cancellation propagation.

## Selected additive signatures

These are the signatures to implement and prove next, not executable public
imports today. Existing candidate names/types that differ are listed explicitly.

For Rust, retain the prototype's exact source methods and gates:

```rust
pub type LoadConfig = EngineConfig;
// No string inference between paths, URLs and repository specifications.
ModelSource::bytes(bytes: impl Into<Arc<[u8]>>)
ModelSource::parts(parts: ModelBytes)
ModelSource::reader(reader: impl Read + 'a)
ModelSource::path(path: impl Into<PathBuf>)                 // mmap
ModelSource::files(files: ModelFiles)                       // mmap
ModelSource::bundle_id(id: impl Into<String>, quant: impl Into<String>) // remote+mmap
ModelSource::hugging_face(spec: impl Into<String>, quant: Option<&str>, strategy: Option<&str>)
ModelLoader::new(source: ModelSource<'a>) -> ModelLoader<'a>
ModelLoader::config(self, config: LoadConfig) -> Self
ModelLoader::build(self) -> Result<ModelHandle, LoadError>
ModelLoader::build_generative(self) -> Result<GenerativeModel, LoadError>
GenerativeModel::create_session(&self, config: SessionConfig) -> Result<Session, CeraError>
```

The HF method also requires `remote+mmap`. Keep `ModelHandle`/`ModelKind` and
`LoadError` non-exhaustive in Rust. The generative accessor shares ownership and
returns `None` for other kinds; it never consumes the dynamic handle.

For native UniFFI, select these Rust declaration shapes, with normal generated
Swift/Kotlin spelling:

```rust
ModelLoader::new(source: ModelSource, config: EngineConfig) -> Self
ModelLoader::build(&self) -> Result<Arc<ModelHandle>, LoadError>
ModelLoader::build_generative(&self) -> Result<Arc<GenerativeModel>, LoadError>
ModelHandle::kind(&self) -> String
ModelHandle::as_generative(&self) -> Option<Arc<GenerativeModel>>
GenerativeModel::engine(&self) -> Arc<CeraEngine>
GenerativeModel::create_session(&self, config: SessionConfig) -> Result<Arc<Session>, FfiError>
```

`ModelSource` uses the candidate's six associated payload variants: Bytes, Parts,
Path, Files, BundleId and HuggingFace. The candidate calls this enum `Source`;
production promotion selects the unambiguous name `ModelSource`. Files/parts
must retain the [full payload inventory](API_RESHAPE_P0_LOADING.md#source-inventory).
The Plan30 foreign `ModelParts.generation_defaults: Option<SamplingDefaults>`
represented only five text sampling fields and always mapped to `GenerationDefaults::Text`.
Plan31 adds native `GenerationDefaults` variants and CPU WASM factories; their
runtime and review status is tracked in the handoff.
Native and CPU WASM parts now carry all three core variants through generated
consumers in Plan31: Text; Audio with its additional decoding-thread count,
audio temperature and audio top-k; and Other with its raw JSON, including null.
Preserve absent defaults separately from a present variant with absent fields.
The native representation is `GenerationDefaults::Text { sampling }`,
`Audio { sampling, number_of_decoding_threads, audio_temperature, audio_top_k }`,
and `Other { raw_json }`. CPU WASM uses the owned `GenerationDefaults.text`,
`.audio` and `.other` factories. `ModelParts.generation_defaults` is optional;
`SamplingDefaults` keeps its five optional fields. Other JSON text is parsed into
the core Value; invalid JSON returns structured InvalidConfig with field
`generation_defaults.raw_json`, reason `invalid_json` and the rejected value.
JSON whitespace and object-key ordering are not preserved by the core Value.
The loaded-manifest observation remains probe-only. See the
[runnable defaults examples](../../tests/api_loading/README.md#multipart-generation-defaults).
Legacy native/WASM multipart constructors accept no defaults payload, so retaining
those constructors does not close this gap. Manifest-backed loading and Rust
`ModelBytes` already carry richer defaults but do not replace in-memory foreign
multipart input.

Use the existing native **EngineConfig record**, including its BackendPreference
enum, u64 context, optional existing BundleRepo object, draft path and depthformer
boolean. Do not export the prototype's string-backend LoadConfig as the replacement
for that record. The probe intentionally accepts invalid backend strings to test
error transport; this is not a reason to weaken production's typed backend field.
The new loader should map failed checked context conversion to structured
InvalidConfig while old constructors continue to return their existing FfiError.
Native zero, default, wide-context and sentinel semantics remain as proved in
Plan27; the existing EngineConfig conversion is the single production conversion
path. A refactor of that conversion must preserve old constructor errors.

`GenerativeModel::engine` must wrap/share the same loaded core engine. It must
not resolve the source again, duplicate model weights, reconstruct a Session or
copy live KV. The returned engine may outlive the typed/dynamic parents. Its
existing methods provide the retained native operation home below. `create_session`
accepts the full existing SessionConfig and returns the existing production Session
object, not the probe's fixed-configuration Session wrapper. Session-operation
errors keep their existing type; the new LoadError is confined to loading.

For CPU WASM, keep the tested `ModelSource.bytes/parts`, `LoadConfig(u32, String)`,
owned loader, kind string and nullable shared accessor representation. The selected
model additions are `engine() -> CeraEngine` sharing the existing core Arc and
`create_session(config: &SessionConfig) -> Result<Session, JsError>`, with JS names
frozen in Plan35 as `buildGenerative`, `asGenerative`, `createSession` and
`GenerationDefaults.toJson`. Keep existing async browser
repository/factory APIs and WebGpuSession on their current homes. There is no
selected WASM native-path or Rust-reader source.

The [eight-class generated signature baseline](../../tests/api_contracts/wasm_loading.json)
records constructors, nullable accessors and all payload fields. Check an actual
`cera_wasm.d.ts` or probe `loading_web.d.ts` with
`python3 tests/api_contracts/check_wasm_loading.py <declarations.d.ts>`.
LoadConfig/multipart/default payload fields retain snake_case; model methods use
camelCase. Source/config/parts/default objects move at their value-taking calls;
`createSession` borrows its config. Existing browser and WebGPU classes remain
covered by the separate legacy declaration baseline.

Probe-only `info`, `files`, `repositoryForProbe`, `resolveForProbe`, future-handle
and native-context-width functions are test observations; none is promoted just
because it was useful to an assertion. Existing metadata, config, defaults,
repository and tokenizer operations provide supported observations where exposed.

## Operation homes after loading

| Operation | Rust | Native Swift/Kotlin/Python/Dart | CPU WASM / WebGPU |
|---|---|---|---|
| Metadata, manifest, config, capabilities, generation defaults | Existing GenerativeModel forwards plus CeraEngine | Existing CeraEngine via selected shared `model.engine()`; keep metadata/context/default/capability methods | CPU CeraEngine metadata/context/default/capability getters; Manifest retains URL/template metadata; WebGPU keeps its own capability/adapter getters |
| Tokenize, detokenize, vocab and special-token lookup | Borrowed tokenizer or tokenizer Arc | CeraEngine `encode_text`, `encode_text_special`, `decode_tokens`, vocab/BOS/EOS/special methods | CPU/WebGPU `.tokenizer` property returns the existing Tokenizer wrapper and its methods |
| Chat templates, tool format and parsing helpers | Existing tokenizer/tools APIs | Existing CeraEngine template/tool methods and module `detect_tool_format`, `parse_tool_calls`, `tool_grammar` | Existing Tokenizer template methods and module tools; no new template override behavior |
| Engine prefix-cache controls | GenerativeModel `configure_cache`, `clear_warm_cache`, `clear_cache` forwards | CeraEngine `clear_prefix_cache`, `wipe_all_prefix_caches`; native does not currently export all raw cache configuration | No equivalent CPU/WebGPU engine cache management export in this inventory; do not invent one or map it to Session reset |
| Session creation | `create_session(SessionConfig)` retains encoders, drafter, defaults and existing GPU lease rules | Existing CeraEngine `new_session` remains; selected typed `create_session` returns the same Session type | CPU `newSession`; WebGPU factories create their separate session directly |
| Raw ingestion, generation, streaming, cancellation, reset | Existing Session methods | All 25 inventoried Session methods remain, including async generation, multimodal sends, LoRA and hidden states | CPU Session's 18 methods; WebGpuSession's 20 methods and separate cancel handle remain; their generation policies are not made equivalent by naming |
| LoRA and embeddings | Existing Session/adapter/classifier paths | Existing LoraAdapters, attach/remove/has and hidden-state methods; PiiClassifier adapter constructor retained | Existing CPU LoraAdapters/Session methods; no invented WebGPU exported adapter/embedding facade |
| LFM2-Audio transcription | GenerativeModel/CeraEngine `transcribe` | CeraEngine `transcribe`, via shared engine; keep standalone Whisper separate | CPU CeraEngine transcription; WebGPU generation retains its audio callbacks |
| Encoder/classifier operations | Existing CeraEngine `detect_pii`, `detect_pii_with_lora` and raw model | CeraEngine `detect_pii` and PiiClassifier constructors/detect | No classifier facade created by a Generative-only handle |
| Raw encoder/GPU attachments, GGUF escape hatch and audio markers | Existing CeraEngine auxiliary getters, GPU attachment flags, doc-hidden `vision_encoder_gguf`, `AUDIO_MARKER_CANDIDATES`, `split_tokens_at_marker`, module `init_dspark_drafter` | Not newly exported; keep currently exposed capabilities/session attachments | Existing supported browser attachment/factory behavior remains; no raw Rust pointer export |
| VAD, hotword and Whisper | Existing standalone loaders | FfiSileroVad, FfiVadIterator, FfiHotwordDetector/Iterator, FfiWhisperModel and every current config/event/default helper remain | No new standalone browser export is claimed; unified loading remains F5 |
| Remote catalog and cache | Existing BundleRepo resolve/download/cache/progress plus catalog helper | Existing BundleRepo's six methods, DownloadProgressSink and sync/async catalog helpers | Existing browser repository's nine methods, catalog/persistent-storage helpers and OPFS implementation remain separate |

The Rust [ownership inventory](API_RESHAPE_P0_OWNERSHIP.md) retains the raw model
and borrowed auxiliary homes in full. This map does not promise that a raw trait
method works on every backend. CPU/model state ownership, native GPU lifetime
leases, adapter isolation and prefix-cache policies keep their prior contracts.
Old generated classes, record fields/defaults and enum variants remain available;
source-name retention alone does not establish binary or future-version ABI
compatibility.

## Loading gate checklist after this audit

- [x] Record source/constructor/config/feature, kind and core ownership inventories.
- [x] Execute candidate typed/dynamic loading and structured errors in Rust,
  native Swift/Kotlin and CPU Node, including remote source/progress retention.
- [x] Freeze the per-target retained homes and selected production signatures above.
- [x] Add a checked declaration baseline and removal/rename/add/type/async controls.
- [x] Prove the selected native EngineConfig/BackendPreference/BundleRepo profile
  through the candidate using production records, defaults and conversion.
- [x] Prove shared engine access and full SessionConfig/production Session return
  in native and CPU WASM consumers, with tokenizer/cache/metadata/session use
  after parent release and without source reload. Keep Rust retained forwards.
- [x] Represent and execute full Text/Audio/Other multipart generation defaults
  in native and CPU WASM, preserving audio-only fields, raw JSON and absence.
- [x] Reconcile Plan32 results and close P0-L's loading prototype gate.
- [x] Publish the selected Rust loading surface (Plan33).
- [x] Implement native production loading and execute generated Swift/Kotlin consumers (Plan34).
- [x] Finish native publication audit/review (Plan34).
- [ ] Publish the CPU WASM loading surface.

Plan32 proves production config reuse and shared engine/session access in native
and CPU WASM consumers. The final 47 Swift/47 Kotlin/26 Node cases include
tokenizer use after parent release, native warm-cache control calls preserving
live positions, config transport, streaming and cancellation. Cache performance
and populated prefix reuse are not measured by this tiny fixture. The
[handoff](API_RESHAPE_HANDOFF.md#plan32-complete-2026-09-12t1716-0700) records
full-build and final consumer-only evidence, ignored-config controls and clean
max reviews. Plan31 executes full multipart defaults. Existing source/error
fixtures must remain intact. No chat/R0/R1/KV budget, Leap runtime or device-release
gate is closed by this declaration audit. The current guard covers 69 reviewed
rows (300 method/function/constant declarations and 34 records/enums and three callback traits); it is not
a compiler, feature/export-attribute check or generated ABI comparison.

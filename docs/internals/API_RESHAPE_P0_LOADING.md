# P0-L loading contracts

Updated: 2026-09-08T05:39-0700. Memory, local filesystem, remote and retained
operation/ownership prototypes pass their scoped checks and reviews. Plan 08
is complete as a bounded CPU and retained-contract increment.
[Plan 09 generated Swift/Kotlin and Node loading consumers](API_RESHAPE_P0_BINDINGS.md)
pass bounded CPU runtime, harness, Clippy and Rustdoc checks. Five max-effort
review rounds ended with all three final reviewers reporting NO FINDINGS;
documentation is audited and plan 09 is complete within that scope. Plan 10's
primary metadata reuse, scoped validation and two max-effort review rounds pass.
That bounded increment is complete; all three final reviewers returned NO FINDINGS.
Plan 11's anonymous-model warm-only policy, scoped Rust/generated validation and
one clean max-effort review round are complete. Evidence and limits are in the [ownership audit](API_RESHAPE_P0_OWNERSHIP.md#plan-11-anonymous-model-persistent-cache-policy).
Plan 12 is complete with executable CPU vision/DSpark companions, [runnable examples](API_RESHAPE_EXAMPLES.md), scoped validation and three max-effort review rounds; all final reviewers are clean.
Plan 13 audio execution, examples and three max-effort review rounds are complete;
all final reviewers are clean. P0-L is not complete. Baseline: `60fc11c25a51`.
See the [main plan](API_RESHAPE_PLAN.md#7-sequencing) and
[current handoff](API_RESHAPE_HANDOFF.md).

For current promotion prerequisites, see the [Plan26 loading exit audit](API_RESHAPE_LOADING_EXIT.md). It separates concrete candidate gaps from later chat and release gates; the dated evidence below retains its original scope.

## Source inventory

The inventory below describes Cera's existing behavior. A source adapter must
preserve it unless the main plan explicitly authorizes a change.

| Source | Existing constructor | Gate | Ownership and resolution |
|---|---|---|---|
| Local GGUF, JSON manifest, directory | `CeraEngine::from_path` | `mmap` (implies `std-fs`) | Mmap primary; directories require exactly one JSON manifest; preserve manifest-relative paths and remote URL resolution |
| Shared bytes | `from_bytes` | None | `Arc<[u8]>` backs GGUF tensor views; synthetic text manifest with `<bytes>` identity |
| Reader | `from_reader` | None | Consumes any `Read`, buffers the entire stream once; no Seek/Send/Clone requirement; synthetic text manifest with `<reader>` identity |
| Multipart paths | `from_files` | `mmap` | Preserve primary path for GPU backend dispatch and auxiliary resolution relative to the primary directory |
| Multipart bytes | `from_parts` | None | Shared primary/auxiliary byte buffers; metadata/default overrides remain attached to their source |
| Bundle ID + quantization | `from_bundle_id` | `remote` + `mmap` | Requires configured `bundle_repo`; preserves known-bundle lookup, manifest fallback, DSpark quant suffix handling and download progress |
| Explicit HF spec/URL | `from_hf`, `from_hf_with_strategy`, `from_hf_url` | `remote` + `mmap` | Requires configured repository; preserves URL/subpath, quantization, strategy, cache root and progress via the existing resolver |

Evidence: [engine constructors](../../cera/src/engine.rs),
[GGUF backing and reader](../../cera/src/gguf.rs),
[feature definitions](../../cera/Cargo.toml),
[HF resolver](../../cera/src/bundle/hf.rs),
[repository and progress](../../cera/src/bundle/repo.rs).

`ModelFiles` has eight fields: `model`, `multimodal_projector`, `audio_decoder`,
`audio_tokenizer`, `draft_model`, `extras`, `inference_type`, `chat_template`.
`ModelBytes` also has eight; it replaces `extras` with `generation_defaults`.
Do not invent byte-form extras or discard path-form extras when wrapping these
records. Keep explicit sources; no string-to-HF/path inference.

Multipart inference selection differs by constructor. Files honor an explicit
type, otherwise inspect the primary architecture. Parts honor an explicit type,
otherwise inspect architecture and upgrade inferred text to vision only after
parsing a supplied projector GGUF. That parse does not establish usable typed
vision weights: a header-only GGUF still upgrades the declaration. File loads
can attach supplied vision weights while the declared type stays text; the
image-ingestion capability gate still rejects images in that mode. Invalid optional auxiliaries can
warn/fall back; explicit text ignores a projector. Do not strengthen this into
blanket rejection as part of the wrapper. Manifest chat-template overrides
are retained separately; the current core renderer uses the embedded GGUF
template. Activating an override would change prompts and requires an explicit
rendering correction rather than a loader-wrapper change. Successfully loaded
draft bytes take precedence over the
config/manifest draft path; path fallback chooses config before manifest.

HF filename selection remains in the existing resolver. Explicit subpaths win;
otherwise requested quantization or the existing preference order selects the
primary. Vision companion selection prefers matching quantization, then Q8_0,
then F16/BF16, then the first candidate. Audio and draft companions have their
own existing policies. The facade delegates this logic. Plan 07 freezes resolver fixtures and
streaming-quantization option selection; actual conversion and broader remote
coverage remain open.

## Load configuration

| Field | Baseline default | Rule to preserve |
|---|---|---|
| `context_size: usize` | 4096 | Load allocation capacity, capped by the model; distinct from session cap |
| `backend: BackendPreference` | Auto | Retain target/feature-dependent backend dispatch and explicit unavailable-backend errors |
| `draft_model: Option<PathBuf>` | None | Config path precedes manifest draft path; absent mmap disables that path loader |
| `gpu_depthformer: bool` | false | Engine true enables a false session setting; the session can also inherit `CERA_GPU_DF=1`; false is not a universal override |
| `bundle_repo: Option<BundleRepo>` | None | Field exists only with `remote`; preserve repository/cache/progress ownership |

Evidence: [engine config and session creation](../../cera/src/engine.rs),
[session initialization](../../cera/src/session.rs).
The prototype aliases `LoadConfig` to `EngineConfig`; it does not freeze a new
public struct or claim the other 31 configuration mappings are audited.

## Model kinds and retained capabilities

The generic CPU loader accepts `lfm2`, `lfm2moe`, `llama`, `qwen2`, `qwen3`,
`granite`, `bert`, and `modernbert`. The last two are encoder-only. Inference-type
detection is a separate policy and must not be mistaken for proof of a supported
generative architecture. Whisper has its own detection helper, including metadata
and tensor-name fallback; VAD and hotword (`kws`) have separate loaders. The prototype rejects known
non-generative kinds before tokenizer/weight/backend construction, and reports
unknown architecture separately. It does not implement a Whisper/VAD/hotword facade.
Evidence: [model dispatch](../../cera/src/model/mod.rs),
[Whisper detection](../../cera/src/model/whisper.rs), [VAD](../../cera/src/vad.rs),
[hotword](../../cera/src/hotword.rs).

Plan24 incorporates upstream 0.5.6 without routing these models through the
generative engine. Memory, filesystem and loopback HF fixtures require `kws` to
produce `KindMismatch { expected: Generative, actual: Hotword, .. }` before backend
construction. Keep standalone `HotwordDetector` file/bytes/GGUF loading and
`HotwordIterator` ownership supported throughout the additive migration. The
foreign detector exposes file/bytes constructors; the foreign iterator loads
model and optional VAD paths with `fromFiles` and preserves mutable stream state.

The upstream fourth async engine constructor, `from_parts_async` (`fromPartsAsync`
in Swift/Kotlin/Dart), belongs in P0-L's multipart migration and consumer matrix.
It uses the same assembly/configuration as `from_parts`; dropping its future
aborts queued work but cannot interrupt a running engine build. Preserve native
Dart text and multimodal loading through this constructor, including caller byte
ownership, optional projector, inference type and error behavior.

The [retained-operation and ownership audit](API_RESHAPE_P0_OWNERSHIP.md)
now assigns every public engine constructor/operation and associated helper a
private typed-prototype or supported legacy home. It records backend state,
auxiliary/draft ownership and the difference between manifest capability flags
and loaded components. LoRA, raw embeddings and per-token hidden states remain
on Session/raw-model paths. No encoder, Whisper, VAD or hotword facade is invented.

Plan 08's CPU tests prove bounded interleaving, parallel execution, cancellation,
reset, extraction/adapter isolation, backend KV-format restrictions and cold
cache control effects. GPU text KV remains model-owned: per-call locks and
separate Session objects do not establish independent conversations. Real
backend, auxiliary/draft and foreign evidence still gates public promotion.

## Executable evidence and limits

[loading_prototype.rs](../../cera/src/engine/loading_prototype.rs) is included
only by the library test harness. It provides bytes/readers/parts, mmap-gated
Path/Files and remote+mmap-gated BundleId/HuggingFace sources, a consuming loader, typed/dynamic generative results and
shared ownership. These types add no public exports. Actual foreign loader
representations, non-generative handles and unknown future variants remain open.

The [filesystem fixtures](../../cera/src/engine/loading_prototype/tests/filesystem.rs)
use the same deterministic one-block F32 CPU model as the memory tests. They
cover GGUF files, uppercase extensions, manifest files, single-manifest
directories, inferred/explicit multipart files, generation parity and sessions
after model handles are dropped. All eight ModelFiles payload fields and the
manifest's separate raw/normalized views are exercised. Relative auxiliary paths
use the manifest or primary directory as appropriate; absolute paths remain
absolute. Missing optional text-side auxiliaries stay nonfatal. Raw future JSON,
chat-template override metadata and five sampling defaults remain available.
An explicit public-renderer assertion preserves the embedded-template output
when the manifest supplies a different template. Earlier field documentation
promised override precedence that the existing renderer does not implement;
plan 06 corrects the documentation without changing prompts.

Private resolver helpers in `engine.rs` are shared with the legacy constructors.
Their extension dispatch, directory policy, normalization and error text are
preserved. Primary paths remain separate from normalized manifest fields so
backend dispatch still receives the caller's original primary. One existing
quirk is deliberately retained: for a relative ModelFiles primary containing a
parent directory, the normalized manifest model field includes that parent
again. Actual loading still opens the original primary. Fixing that metadata
quirk requires a separately declared compatibility correction; the new wrapper
must not accidentally reopen the duplicated path.

Known non-generative kinds fail before tokenizer, weight and backend assembly;
unknown architectures get their own typed error. Legacy automatic file detection
classifies encoder architectures as unsupported inference types. The typed
prototype inspects automatic GGUF/file sources first to report KindMismatch,
then checks the final mapped primary again. Explicit unsupported manifest types
keep the legacy error ordering before primary-file opening. Missing files,
invalid JSON, unsupported extensions, empty/ambiguous directories, file URIs
and remote URLs without a configured repository retain legacy source errors.

Load configuration is retained, including backend, requested context size,
optional draft path and depthformer flag. Model capacity is capped separately
from the requested load size; session limits can reduce but not enlarge it.
The remote-enabled local fixture retains the configured repository root and
same progress callback without creating a cache or downloading anything. This
proves ownership/configuration forwarding, not remote resolution or progress
behavior during downloads. Successful draft attachment/precedence, depthformer
execution and device/backend dispatch remain separate gates.

Bytes/readers reuse the existing private assembly function. Plan 10 also retains
the parsed primary through multipart assembly and filesystem auto-detection,
removing the temporary duplicate parses from plans 02 and 06. These
CPU fixtures do not establish loading performance, real-model quality or
warm-chat support.

## Plan 06 validation

Run from the implementation worktree with `/opt/homebrew/bin` on PATH and
`CARGO_TARGET_DIR=/Users/dberrios/development/cera/target`.

| Command | Result |
|---|---|
| `cargo test -p cera --lib --locked --offline --quiet` | 622 passed, five existing ignored tests |
| `cargo test -p cera --lib engine::loading_prototype --locked --offline` | 13 passed |
| Same focused command with `--no-default-features` | Six memory tests passed; filesystem source absent |
| Same focused command with `--no-default-features --features mmap` | 13 passed |
| `cargo test -p cera --features remote --lib engine:: --locked --offline --quiet` | 45 passed; includes repository ownership fixture |
| `cargo clippy -p cera --features remote --all-targets --locked --offline -- -D warnings` | Passed |
| `cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote --no-deps --lib --locked --offline` | Passed |
| `cargo fmt --all --check`; `git diff --check` | Passed |

`cargo test -p cera --features remote --test engine_load --test manifest_parse
--test bundle_from_id --locked --offline --quiet` also passes: 16 passed,
four existing ignored tests requiring real models/downloads. The first
max-effort round corrected an effective-template contract error and a stale
handoff sequence. The new renderer assertion, remote all-target Clippy and
remote Rustdoc pass after those fixes. Two max-effort review rounds completed,
each with three fresh reviewers; all three final reviewers returned NO FINDINGS.
No actionable findings remain, and none were skipped.
Full workspace CI, live remote downloads, real multimodal/draft fixtures,
foreign loader generation and device/performance checks were not run here.

## Plan 07 remote evidence

Private `resolve_bundle_source` and `resolve_hf_source` helpers now serve both
the legacy public constructors and the test-only source adapters. They preserve
repository requirements, bundle suffix handling, metadata/defaults fetch order,
file normalization and assembly. The mapped primary still receives a kind check
before tokenizer/weight/backend assembly. Remote resolution and auxiliary
downloads can occur before that check; it is not a pre-download kind guarantee.

The [remote fixtures](../../cera/src/engine/loading_prototype/tests/remote.rs)
use actual HTTP, repository caching and CPU assembly. Their
[loopback harness](../../cera/src/engine/loading_prototype/tests/remote/http.rs)
runs each case in a separate process with its own endpoint/proxy settings and a
synthetic token. It does not modify the parallel parent process's environment,
forward proxy traffic or contact public servers. A deadline and an explicit
one-test-passed assertion prevent hangs or false success from empty child runs.

| Fixture | Evidence |
|---|---|
| HF specs and URLs | Default/inline/explicit quantization, explicit subpath precedence, revisions, URL alias, sampling defaults, actual two-token generation parity and sessions after releasing model handles |
| Cache and progress | One GET per selected model across repeated loads, HEAD validation, persisted checksums, no cache-hit progress, monotonic byte counts and retained callback identity/store root |
| Remote failures | Missing repository before input validation, metadata 404/401/invalid JSON, unavailable quantization, invalid conversion quantization, structured final-kind errors before an unavailable Metal backend |
| Manifest downloads | Primary, projector, decoder, tokenizer, draft and extra file URLs resolve to the configured store; raw manifest metadata stays intact; malformed GGUF/HTTP/checksum errors match legacy constructors |
| Cached bundles | Cached catalog manifest with relative primary, known-bundle bypass, both DSpark suffix forms, explicit draft precedence, preserved known VL defaults |
| Cache integrity | Equal-length corrupt bytes trigger a replacement GET, missing checksum sidecar is repaired by rehash, caller-pinned hash skips HEAD, failed HEAD reuses cached bytes, checksum failure leaves no final/partial/sidecar file |

The fixed public bundle URLs use preseeded cache files and locally rejected
HTTPS tunnels. This proves the existing failed-HEAD fallback and constructor
routing, not a live catalog download. The HF metadata path still fetches
generation defaults on every invocation, including before an unavailable-quant
error; model cache reuse does not imply network-free source discovery.

The fixtures use the downloader's currently accepted
`X-Linked-Etag: "sha256:<hex>"` form. They do not establish how every live origin
formats its headers or strengthen the existing sidecar/HEAD-failure trust policy.
All primary fixtures contain the same tiny F32 model: quantization names test
file selection, not conversion output or quality. HF download fixtures pad that
valid GGUF to 600 KiB with unreferenced trailing bytes to cross the progress
callback threshold. They require multiple callbacks, an intermediate count
below the final total and nondecreasing counts. Auxiliary/draft fixtures prove
resolution and existing nonfatal fallback, not successful multimodal or draft
execution. Companion file-selection policies are covered independently by the
[HF policy fixtures](../../cera/src/bundle/hf/loading_contracts.rs), including
vision fallback order, distinct audio/draft fallbacks, retained extras and
explicit nested-file boundaries.

Streaming conversion still uses its existing implementation. Extracted private
option construction freezes target/default strategy selection, invalid-strategy
fallback to Auto, cache root, callback ownership and auth lookup ordering.
Invalid quantization fails before auth lookup. Actual SafeTensors conversion,
its download/cancellation behavior and converted model parity remain untested
by this increment.

Run from the implementation worktree with `/opt/homebrew/bin` on PATH and
`CARGO_TARGET_DIR=/Users/dberrios/development/cera/target`. Loopback fixtures need
permission to bind localhost; they use no external service.

| Command | Result |
|---|---|
| `cargo test -p cera --features remote --lib --locked --offline --quiet` | 659 passed, five existing ignored |
| `cargo test -p cera --lib --locked --offline --quiet` | 625 passed, five existing ignored |
| `cargo test -p cera --no-default-features --lib loading_ --locked --offline --quiet` | Nine passed; remote/filesystem adapters absent |
| Same focused command with `--features remote` | Ten passed; remote source adapters absent without mmap |
| Same focused command with `--features remote,mmap` | 23 passed, including all five loopback scenarios |
| `cargo test -p cera --features remote --test engine_load --test manifest_parse --test bundle_from_id --locked --offline --quiet` | 16 passed, four existing ignored model/download cases |
| `cargo clippy -p cera --features remote --all-targets --locked --offline -- -D warnings` | Passed |
| `cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote --no-deps --lib --locked --offline` | Passed |
| `cargo fmt --all --check`; `git diff --check`; local Markdown checks | Passed; 62 local links |

The first max-effort review found that the initial tiny download produced only
a final progress callback, making its monotonicity assertion vacuous. The larger
fixture and explicit intermediate-count requirement fix that gap. The transfer
also exposed inherited nonblocking accepted sockets on macOS in the test server;
those sockets now use bounded blocking I/O, and cleanup preserves the original
failure while unwinding. All 23 minimal remote+mmap loading tests, the full remote
library suite (659 passed/five existing ignored) and remote all-target Clippy
pass after the fixes. Two max-effort review rounds completed with three fresh
reviewers each; all three final reviewers returned NO FINDINGS. No actionable
findings remain, and none were skipped. Full workspace CI,
live remote/CDN, actual conversion, foreign loader generation, device and
performance checks remain outside this increment. No public API, generation or
warm-chat changes are claimed.

Remaining P0-L: live catalog/CDN, conversion and broader remote failure probes;
real auxiliary/drafter and exhaustive precedence fixtures; full foreign loading
matrices beyond plan 09's generated consumers and same-build kind controls; device
execution ownership proof or enforced sharing restrictions; stable persistent
identity beyond the anonymous-model warm-only restriction in plan 11. Plan 10 removes prototype primary metadata reparsing
within the scope below. The complete retained-home map and bounded CPU evidence are in the
[plan 08 audit](API_RESHAPE_P0_OWNERSHIP.md); they do not complete P0-L.

## Plan 10 single-pass primary metadata

Updated: 2026-09-07T21:39-0700. Implementation, scoped validation and
two max-effort review rounds are complete. Root plan:
`devlog/plans/000341-10-single-pass-loading.md`. Increment baseline snapshot:
`/private/tmp/cera-api-plan10-91f9lv82`.

`CeraEngine::from_parts_with_primary` shares the existing auxiliary selection,
manifest construction and assembly with the already classified multipart
primary. `ResolvedPathSource` can retain auto-detection's `GgufFile`; its private
`open_primary` keeps the unsupported-inference gate before using that object or
opening a manifest primary. Legacy and typed loaders share these paths. Private
checked resolvers run the typed kind callback at the first parse, preserving its
position before inference rejection and auxiliary resolution. Explicit inference
still resolves files before opening the primary. All public signatures remain.

Autodetected bare paths retain their mapping only when the resolved manifest
primary equals the original path. This preserves the legacy lossy-string path
failure for non-UTF-8 filenames. Explicit ModelFiles retains its original byte
path, including when manifest normalization duplicates a relative parent. The
Linux-only invalid-UTF-8 regression was not executed locally: APFS rejected the
temporary filename outside the sandbox as well as inside it. Linux execution
remains required to establish that platform-specific assertion.

A test-only per-thread counter in the actual GGUF parser observes constructor
deltas. New regressions failed before the fix: typed multipart primary twice,
legacy bare-file primary twice, typed bare-file primary three times. The same
fixtures now require exactly one primary parse for bytes, readers, multipart
bytes, bare files, explicit/inferred ModelFiles, manifests and directories.
Typed and dynamic loads also generate after parent release. Ordering controls
observe legacy sidecar parsing before inference rejection, typed rejection
before sidecar parsing/resolution, and explicit unsupported inference rejecting
before a missing primary is opened. Auxiliary parse counts are explicit in the
ordering control; the one-primary-parse claim excludes auxiliary parsing,
backend-owned reopens and conversion.

Run from the implementation worktree with Homebrew on PATH and
`CARGO_TARGET_DIR=/Users/dberrios/development/cera/target`:

| Command | Result |
|---|---|
| `cargo test -p cera --lib --locked --offline` | 636 passed, five existing ignored |
| Same command with `--features remote` | 670 passed, five existing ignored |
| `cargo test -p cera --no-default-features --lib engine::loading_prototype --locked --offline` | 13 passed |
| Same focused command with `--features mmap` | 23 passed |
| Same focused command with `--features remote,mmap` | 29 passed |
| `cargo test -p cera --features remote --test engine_load --test manifest_parse --test bundle_from_id --locked --offline --quiet` | 16 passed, four existing ignored model/download cases |
| `cargo clippy -p cera --features remote --all-targets --locked --offline -- -D warnings` | Passed |
| `cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote --no-deps --lib --locked --offline` | Passed |
| `cargo fmt --all --check`; `git diff --check` | Passed |

Remote fixtures use local loopback servers and needed sandbox escalation for
listeners. `python3 tests/api_loading/test_harness.py` passes ten tests. A fresh
`python3 tests/api_loading/run.py --target /Users/dberrios/development/cera/target/api-loading`
passes 21 command expectations and all 35 Swift/Kotlin/Node consumer cases;
current report is `tests/api_loading/build/run-56svuhvn/results.json`. Native
and WASM scoped Clippy/Rustdoc pass on that exact mirror; see
[binding evidence](API_RESHAPE_P0_BINDINGS.md#plan-10-core-refresh).

No public prototype promotion, generation kernel, session or KV representation
change is included. These parse counts do not measure load latency, real-model
quality, device ownership or warm-chat performance. Broader P0-L and all chat,
recovery, performance and Leap facade/package gates remain open.

Round one used three max-effort reviewers and found one distinct issue: the
Linux-only invalid-path assertion compared a payload prefix with an error display
that adds `backend:`. It now matches the Backend payload. Five local parse/order
tests, remote all-target Clippy and format/diff checks pass after that fix; Linux
execution is still unverified. Round two used three fresh max-effort reviewers
and returned all NO FINDINGS. One distinct finding was fixed; none were skipped
and none remain open. Fresh source-hash consumer run-56svuhvn passes after the
fix and all 372 source hashes match the current tree. Native/WASM Clippy and
Rustdoc pass on its exact mirror with warnings denied; all 11 recorded artifact
hashes still match afterward. The final documentation audit passes for 13
Markdown files, 103 local targets and four anchors. Full workspace CI/device
checks were not run. Plan 10 is complete within this bounded scope.

## Plan 12 executable vision and DSpark companions

Completed 2026-09-07T22:54-0700; scoped validation and final review pass. The
[example guide](API_RESHAPE_EXAMPLES.md) provides Rust, Swift and Kotlin loading
examples, a complete runnable Rust continuation program and companion test commands.

The new [auxiliary tests](../../cera/src/engine/loading_prototype/tests/auxiliary.rs)
build complete synthetic GGUFs with one nonzero attention/FFN block in each
primary/companion. Vision uses an LFM2 target supporting embedding input;
drafting uses a dense Llama target supporting speculative verification. A
one-block ViT/projector encodes actual pixels. Two DSpark sidecars share base
embeddings/output weights and have distinct rank-one Markov heads and depths,
so selected weights produce different draft tokens. No downloaded model is used.

| Contract | Executable evidence |
|---|---|
| Multipart vision | Legacy, typed and dynamic loaders distinguish absent/corrupt/raw-header/valid companions under inferred, explicit text and explicit image modes |
| File/manifest vision | Same optional failures and valid weights through files, manifest and directory; filesystem text mode can attach weights without image capability |
| Vision execution | Encoder output matches independently loaded weights and differs with other weights; PNG ingestion plus token continuation matches direct embedding input; sessions survive parent release and Unix source deletion |
| Draft selection | Valid byte companions win; invalid/absent bytes use a configured path with mmap; config wins over manifest/files, and a failed config path does not fall through to the manifest |
| Draft execution | Actual DSpark token proposals, observed calls from Session generation, greedy target output parity, interleaving/reset and retained sessions after parent release and Unix source deletion |
| Draft state | Shared-weight concrete DSpark instances compare hidden states with independent controls; equal-length/same-final-token prompts prove dependence on earlier context |
| Feature boundaries | Without mmap, a valid configured draft file stays inert while byte companions execute; PNG ingestion requires vl-preprocess |

Three temporary negative controls reject production mutations: removing vision
attachment produces the specific missing-encoder error; reversing config/manifest
draft precedence selects depth 2 instead of 3; ignoring earlier draft tokens
fails the numerical history assertion. All mutations were restored byte-for-byte. Logs are under `/private/tmp/cera-api-plan12-csjc23vh/controls`.
No production loader or inference behavior changes in this increment.

Scoped checks:

```bash
cargo test -p cera --lib --locked --offline --quiet
cargo test -p cera --no-default-features --lib --locked --offline --quiet
cargo test -p cera --no-default-features --features mmap --lib engine::loading_prototype::tests::auxiliary --locked --offline --quiet
cargo test -p cera --features remote --lib --locked --offline --quiet
cargo clippy -p cera --features remote,gpu,metal --all-targets --locked --offline -- -D warnings
cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote,gpu,metal --no-deps --lib --locked --offline
```

Default library: 647 passed/five existing ignored. No-default: 568 passed.
Minimal mmap companion suite: six passed. Remote: 681 passed/five ignored after
allowing loopback listeners; the sandboxed attempt had nine PermissionDenied
failures and 672 passes. Both Clippy configurations and Rustdoc pass with warnings
denied; Cargo still reports the existing block 0.1.6 future-incompatibility notice.

Audio encoder/vocoder/detokenizer loading and precedence, actual remote companion
execution, conversion, device ownership/execution, performance budgets, stable
named cache identity and the full P0-L promotion matrix remain open. These
fixtures establish loading/runtime contracts, not pretrained model quality,
GPU correctness, custom-Drafter isolation or incremental-chat support.

The runtime source-directory guard deletes mapped inputs before execution only
on Unix. Other platforms retain the directory until all sessions/engines have
been dropped, since [Windows rejects deletion of mapped files](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-deletefile).
Windows execution was not run locally. This correction preserves all constructor
and execution cases; only the early-unlink proof is Unix-specific.

Final code validation at 2026-09-07T22:51-0700: default 647/five ignored and no-default 568
pass after all test corrections; remote companion cases pass seven. Full remote
681/five ignored was verified earlier in this increment; later changes touch
only test assertions, fixture cleanup and documentation. Final full-feature and
minimal Clippy pass. Current generated run-s3z9znv1 passes 35 consumers and ten
harness tests; its native/WASM Clippy/Rustdoc and separate Rust walkthrough pass.
All 377 source, 11 generated/native/WASM and four consumer hashes match after
checks. The walkthrough records matching three-token outputs at positions 5/9.
[Binding evidence](API_RESHAPE_P0_BINDINGS.md#plan-12-executable-examples-and-companion-test-refresh)
records the exact target and commands. Documentation audit: 16 Markdown files,
159 local targets, 13 anchors, zero errors before completion-record edits.

Plan 12 completed 2026-09-07T22:54-0700. Three max-effort rounds used three fresh reviewers
each. All three final reviewers returned NO FINDINGS; three distinct findings
were fixed (history sensitivity, cache documentation scope and portable source
cleanup), none skipped, none open. No production API is promoted. The next
bounded increment is audio companion loading/precedence, with examples kept in
sync as the implementation advances.


## Plan 13 executable audio companions

Started 2026-09-08T04:45-0700; completed 2026-09-08T05:39-0700. Root plan: `devlog/plans/000341-13-audio-loading-contracts.md`.
Baseline and logs: `/private/tmp/cera-api-plan13-sg4a3peg`.

The [audio walkthrough](API_RESHAPE_AUDIO_EXAMPLE.md) and
[executable tests](../../cera/src/engine/loading_prototype/tests/auxiliary/audio.rs)
use full CPU tensor payloads. Legacy, typed and dynamic loaders execute multipart,
files, JSON and directory sources. Path cases unlink sources before execution
on Unix; other platforms retain a directory guard while mappings remain alive.
No production loading or inference behavior changes in this increment.

- [x] Optional missing/corrupt/header-only encoder and output weights.
- [x] Explicit audio/text routing and plain-LFM2 multipart inference behavior.
- [x] Encoder dimension error before position changes; text remains usable.
- [x] Decoder hidden-size filter discards decoder and detokenizer together.
- [x] Byte decoder fallback and first-success detokenizer precedence.
- [x] PCM input/resampling versus independently loaded embedding controls and same-length silence.
- [x] Depthformer, code embedding, spectrum and PCM source selection controls.
- [x] Retained sessions generate audio after parent release and independent resets.
- [x] Same-current-code detokenizer history and reset controls.
- [x] Runnable audio example, fixture export and three README links.
- [x] Final source/artifact verification, review and documentation audit.

The source-selection table and caveats live in the example guide. Files do not
fall back to the projector for output. A parsed invalid vocoder blocks the byte
decoder fallback; detokenizer parsing tries later sources. Capabilities remain
modality declarations. The one-code decoder fixture makes session audio output
deterministic while executing its real attention/FFN path. It does not establish
sampling quality or a trained checkpoint's audio behavior.

Validation: default library 653 passed/six ignored (five pre-existing and
the manual fixture exporter); no-default 574/one ignored. Minimal mmap and remote
audio suites each pass six/one ignored. Remote+gpu+metal all-target Clippy,
no-default lib/tests Clippy and remote+gpu+metal Rustdoc pass with warnings denied.
The full remote library was not rerun for this test-only change; Plan 12 retains
its separately scoped result. Commands:

```bash
cargo test -p cera --lib --locked --offline --quiet
cargo test -p cera --no-default-features --lib --locked --offline --quiet
cargo test -p cera --no-default-features --features mmap --lib engine::loading_prototype::tests::auxiliary::audio --locked --offline --quiet
cargo test -p cera --features remote --lib engine::loading_prototype::tests::auxiliary::audio --locked --offline --quiet
cargo clippy -p cera --features remote,gpu,metal --all-targets --locked --offline -- -D warnings
cargo clippy -p cera --no-default-features --lib --tests --locked --offline -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p cera --features remote,gpu,metal --no-deps --lib --locked --offline
```

Four temporary production regressions fail the intended assertions: removing
encoder attachment, removing vocoder attachment, putting the tokenizer before
the vocoder, and accepting a hidden-size mismatch. `controls.json` records each
log. `engine.rs` was restored byte-for-byte after every control.

Remaining gates include actual remote audio execution, real-device sharing and
fallback, foreign audio consumers, conversion execution, named cache identities,
chat/recovery and numeric performance budgets. CPU audio decoder state is scoped
to a generation call; this does not promise continuous audio synthesis across
separate generate calls. The Leap facade remains unimplemented.


The corrected external audio walkthrough executes successfully: two input audio
positions, six text tokens, 22,560 PCM samples at 24 kHz and final position 22.
Fresh `run-b1_7do81` passes the 21 standard expectations and all 35 foreign text
cases, ten harness tests, both Rust walkthrough execution/Clippy checks and four
exact mirror native/WASM Clippy/Rustdoc checks. The [binding audit](API_RESHAPE_P0_BINDINGS.md#plan-13-audio-example-and-generated-consumer-refresh)
records source/artifact hashes and separates Rust audio from foreign text proof.
Final default/minimal library runs, focused mmap/remote audio tests and both
Clippy configurations pass again after the same-length PCM/silence assertion. Production source files match the baseline.


Plan 13 is complete after three max-effort rounds with three fresh reviewers per
round; all final reviewers are clean. Two findings were fixed: the external
example's ambiguous generic conversion and the guide's missing explicit-text
projector qualifier. No findings were skipped or remain open. The additional
same-length silence assertion pins PCM sensitivity. Thread capacity staggered
reviewer launches; all nine completed. No production API is promoted.

## Plan 14 remote companion execution

Implementation, runnable examples, scoped validation and one max-effort review
round with three clean reviewers are complete. The [remote guide](API_RESHAPE_REMOTE_EXAMPLES.md) gives exact
commands and links the complete test programs. This increment combines the
existing discovery/download/cache path with Plans 12/13's executable CPU weights.
Shared fixtures compile once; their bytes and tensor builders are unchanged.

- HF vision: same-quant selection, Q8 fallback, missing-sidecar integrity repair,
  selected numerical outputs and retained session execution.
- HF audio: encoder input, dedicated decoder versus extra vocoder, detokenizer
  from a full decoder or separate tokenizer, finite PCM and text continuation.
- Draft: actual observed proposals from HF/local JSON/preseeded bundle manifests,
  configuration precedence and primary-only generation/continuation parity.
- Failures: malformed downloaded optional GGUFs remain nonfatal for text; SHA
  mismatch/404 downloads fail before assembly and leave no cache residue.

The fixture server rejects external tunnels and observes actual GET counts.
The public bundle catalog is preseeded; live catalog/CDN and conversion remain
open. Matching cache SHA sidecars are trusted without rehashing; the repair case
explicitly starts without one. These tests do not measure device or KV performance.

No production loader/inference or foreign consumer changes are included. Plan
13 generated artifacts remain scoped to their original source snapshot.

Final validation: remote library 691 passed/six ignored; default 653/six;
no-default 574/one; minimal remote+mmap nine/zero; remote without mmap loading
24/one. Full-feature and minimal all-target Clippy, warnings-denied Rustdoc,
Cargo/build-support formatting and document/diff checks pass. Exact commands
and logs: `/private/tmp/cera-api-plan14-tmuigupg/validation.json`.
One max-effort round completed with three fresh reviewers, all NO FINDINGS;
none were skipped. Thread limits required staggered launches, and three
tool-orchestration errors required retries before the final reviews completed.
Production and existing consumer source scope matches Plan 13 except documented
README/test-module changes. Plan 14 is complete within this CPU fixture scope.

## Plan 15 executable SafeTensors conversion

The [runnable conversion guide](API_RESHAPE_CONVERSION_EXAMPLES.md) links the full
three-test fixture and exact commands. Single and sharded inputs execute real
streaming conversion through shared HF resolution, followed by typed/dynamic and
legacy loading. Lossless tensors and logits match an independent original GGUF;
F16/Q8_0/Q4_0 tensor types and per-weight error are checked. Revision/strategy cache
separation, exact HTTP ranges, cached reloads and retained continuation are covered.

The low-level cancellation option primes a checkpoint after five of eleven tensors.
Loader retry resumes six remaining tensors or restarts all eleven for an invalid
checkpoint. An appended uncommitted tail is removed. Malformed headers and decoded
element-count errors retain legacy errors and never publish a final artifact;
tensor failures can leave temporary files. Progress represents output bytes.

No production policy changes. Converted-cache reuse trusts local manifest/file
existence; the sidecar is written on conversion but is not verified on that hit
path. Checkpoints do not identify all immutable source weights. Live CDN, changed
upstream data, concurrency, other architectures/dtypes, default Q4_K_M execution,
device sharing and numeric performance remain separate gates. Plan13 binding
evidence is historical.

Completed 2026-09-08T17:55-0700. Full remote library: 694 passed/six ignored. Default: 653/six;
no-default: 574/one. Minimal remote+mmap: 12 passed; remote without mmap loading: 24/one.
Full-feature and minimal all-target Clippy, warnings-denied Rustdoc,
Cargo/build-support formatting and diff checks pass. Exact commands and logs:
`/private/tmp/cera-api-plan15-f066p24d/validation.json`. The document audit passes
19 Markdown files, 214 local targets and 23 anchors. No full workspace, live CDN,
device or performance gate is claimed.

Two max-effort rounds completed with three fresh reviewers each; the final round
is clean and no findings were skipped. First-round findings strengthened the
quantized-weight oracle with relative L2 error below 15%, and checkpoint recovery
with a 128 KiB tail beyond the output and an exact aligned file-length assertion.
A temporary double-count of the header was corrected before validation. Both
negative controls now fail specifically: zeroed Q4_0 FFN at relative error 1,
missing truncation at 145120 bytes versus 30688 expected. Production restored to its
original hash. Response writes tolerate only BrokenPipe/ConnectionReset to handle
client disconnects while retaining other fixture failures. See the
[completed handoff](API_RESHAPE_HANDOFF.md#completed-plan-15-safetensors-conversion).

## Plan 16 named persistent cache identities

Named built-in LFM2, wgpu and Metal prefix caches now combine their caller
namespace with a versioned digest of all loaded GGUF backing bytes. CPU resolves
it once on first cold configuration outside cache locks; GPU resolves before
releasing source weights. DSpark includes draft and base through an additive
provided `GpuWeightSource` hook. Custom sources default to caller-managed IDs.

Same-path atomic replacement cannot restore the original weights' cold prefix;
unchanged reload restores two tokens and computes only the suffix. Old path-only
files are ignored and preserved. Original mapped sessions remain usable. Metal
uses the parsed mapping itself, with direct execution checked after replacement.

See the [complete cache examples](API_RESHAPE_CACHE_EXAMPLES.md) and
[current validation record](API_RESHAPE_HANDOFF.md#completed-plan-16-named-cache-identities).
ModelLoader remains private; numeric performance budgets, converted checkpoint
identity, live CDN and wider device/architecture coverage remain open.

## Plan 17 converted cache integrity

The [conversion guide](API_RESHAPE_CONVERSION_EXAMPLES.md) demonstrates same-size
weight corruption, truncation, changed manifest defaults, missing/malformed
completion records, sidecar repair and cancellation during repair. Versioned
receipts match requested conversion options and verify exact manifest/GGUF bytes.
Tensor override changes force conversion; legacy or option-mismatched checkpoints
restart. Unchanged requests still resume six tensors from the five-tensor fixture
checkpoint. Local artifact checks do not pin upstream refs or verify partial
checkpoint contents. Full converted-load hashing cost remains a performance gate.
All11 scoped checks pass, with remote701/seven ignored, default657/seven,
no-default574/one, remote without mmap602/one and minimal remote+mmap15.
Two max-effort rounds with three fresh reviewers each complete; final round all
NO FINDINGS after correcting the manifest fixture to mutate/assert the effective
nested temperature. The handoff records the extra minimal all-target lint failure,
scoped library replacement and unchanged source evidence. No whole P0-L phase,
full workspace/CI gate or fresh generated consumer run is claimed.

## Plan18 partial-checkpoint integrity — completed 2026-09-09T04:22-0700

The [conversion guide](API_RESHAPE_CONVERSION_EXAMPLES.md#interrupted-and-failed-conversions)
now demonstrates verified prefix resume, corrupt checkpoint restart and repeated
interruption. All seven focused CPU conversion tests pass, including fourteen
recovery cases and independent prefix hashes at five/ten tensors. The converter
uses private checkpoint helpers to hash writes incrementally, match generated header/layout and exact
tensor boundary, read the saved prefix once and retain that verified file handle
for truncation/writing. Completed-cache receipts and append/generate code remain
unchanged. All eleven scoped checks pass: remote703/seven ignored, default657/seven,
no-default574/one, remote without mmap603/one and minimal remote+mmap16; full/minimal
Clippy, Rustdoc and format/diff checks pass. One max-effort round with three fresh
concurrent reviewers completed; only two documentation visibility wording
nitpicks were reported and corrected. Four Rust source hashes match the reviewed
snapshot. The handoff records the existing minimal all-target feature gap, exact
validation evidence and historical generated-consumer scope.
Upstream pinning, identical-layout source changes, concurrent conversions,
platform sharing, large-model budgets and full P0-L/chat/Leap gates remain open.

## Plan19 upstream conversion revisions — completed 2026-09-09T12:16-0700

An internal snapshot parser preserves HfModelInfo's public fields while validating
a resolved full commit. Conversion binds that identity into version-two requests
and pins all input URLs before cache/checkpoint reuse. The non-main metadata URL
now uses the revision path. Four new upstream fixtures cover mutable inputs,
changed weights with identical layout, cache refresh, rejected resolution and
explicit revisions/authentication. Existing recovery gains legacy-source migration.
All twenty remote-loading fixtures pass in the final remote/minimal suites.
Eleven scoped gates pass: remote707/seven ignored, default657/seven, minimal574/one,
remote without mmap603/one and minimal remote+mmap20; Clippy/Rustdoc/format/diff
checks pass. A fixture-only Clippy issue was corrected and affected gates reran.
Two max-effort rounds with three fresh reviewers each are complete; final round
all NO FINDINGS after correcting three README retention claims. Ten Rust source
hashes are unchanged between rounds. The handoff records exact evidence and the
historical minimal all-target vision feature gap. [Runnable examples](API_RESHAPE_CONVERSION_EXAMPLES.md#pin-inputs-while-a-repository-changes)
document mandatory metadata refresh, unchanged-commit reuse and remaining costs.
This scope does not pin non-conversion GGUF loads or establish concurrent-cache,
live-CDN/device, source-attestation, public-API or full-phase completion.

## Plan20 direct HF snapshots — validation 2026-09-09T20:24-0700

[Five executable examples](API_RESHAPE_HF_EXAMPLES.md) cover direct HF GGUF
revision pinning and existing cache behavior. Discovery now uses Plan19's private
snapshot parser; generation defaults and co-located file URLs use that commit.
A known external DSpark draft resolves its own repository commit before the
discovered manifest is returned.
Public structs/signatures, pure metadata/manifest helpers, explicit manifest and
bundle catalog semantics remain unchanged. Existing vision/audio/draft HF fixtures
publish SHA and serve only pinned paths while retaining CPU execution assertions.

A/B weights have the same layout/length but different values. Mutable main URLs
and an old branch cache entry serve B while metadata resolves A; loading must
return A. A,B,B,A resolution gets each GGUF once, reuses each commit without new
progress and fetches each commit's defaults twice. An A session ingests two tokens
before B loads, then three more after parent release; position/logits match an
independent A model. Invalid/missing SHA or404 rejects cached reuse without changing
bytes/progress; uppercase explicit commit/subpath and mismatch checks pass. Metadata
and files require fixture authentication. Standalone metadata accepts absent SHA.

Original production failed all three initial tests: wrong model bytes, accepting
invalid metadata and uppercase mutable URL404. Initial sandbox loopback denial was
rerun with permission; `baseline-behavior.log` holds actual failures. No additional
production mutation controls are claimed. `focused.log` passes all23 remote tests.

The first eleven scoped gates passed: remote710/seven ignored, minimalremote+mmap23,
default657/seven, no-default574/one and remote-without-mmap603/one. Full
remote,gpu,metal all-target Clippy, no-default remote library Clippy, warnings-denied
Rustdoc, Cargo/build-support format and diff checks pass. Evidence:
`/private/tmp/cera-api-plan20-dyeb42gn/validation.json`. After the external draft fix, all eleven scoped gates pass again: remote711/seven
ignored and minimalremote+mmap24; the other counts are unchanged. A fresh review
round is pending.
These are scoped core gates, not the full workspace/device/CI matrix. The historical
minimal-remote all-target vision-test feature gap remains separate.

Direct GGUF loads require one primary metadata query plus one if an external
draft is selected, even on cache hits/full commits; conversion still resolves twice. Old branch entries remain, while different commits
have separate paths. Defaults remain optional/best effort. Missing SHA fails
end-to-end discovery before format selection. Remote metadata failures cannot use
offline cache fallback; local paths remain available. Endpoint commit semantics are
trusted. Catalog/explicit-manifest/browser pinning, concurrency, cache GC, network
and device performance remain open. Session/KV code and generated consumers are
unchanged; Plan16's generated report remains historical.

Round1's three reviewers found that the pure resolver's known external DSpark
fallback still used main. End-to-end discovery now separately resolves that draft
repository, while pure resolver/catalog behavior stays unchanged. The fourth
fixture holds primary weights fixed while draft commits change A,B,B,missing,A.
It verifies exact draft bytes, separate cache paths, observed proposals, retained
A-drafter execution, cache reuse and failed-resolution behavior. The round1 code
fails this new fixture on wrong draft bytes; an earlier fixture type mismatch was
corrected before that failure was measured. Both logs are retained. Contracts
review had two orchestration interruptions and resumed to a completed finding.

Round2 correctness/contracts found the new internal draft-URL parsing lost a
configured HF_ENDPOINT path prefix. The fifth example covers repo-ID loads through
/hub/mirror with co-located and external drafts, exact bytes/cache paths, request
counts and observed proposals. Round2 code fails with HF draft URL lacks a file
path. Internal generated URLs are now normalized against the configured base;
public URL parsing is unchanged. One contracts-review orchestration interruption
was resumed; reuse reported no findings. All eleven scoped checks pass again,
remote712/seven ignored and minimalremote+mmap25, with other counts unchanged.
Round3 review is pending.

### Plan20 completion — 2026-09-10T07:28-0700

The final external-draft and prefixed-mirror implementation passes all eleven scoped gates and the
third max-effort round's three fresh reviews are clean. No open/skipped findings
remain. Eight Rust source hashes match round3 and final snapshots; logs/five test
binaries and document targets/anchors are audited. Completion records supersede
pending review entries above. Eighteen bounded increments now complete, sixteen
core plus two Leap export experiments; no major phase complete. Next21; no active
jobs, commits or pushes. See the current handoff for exact remaining gates.

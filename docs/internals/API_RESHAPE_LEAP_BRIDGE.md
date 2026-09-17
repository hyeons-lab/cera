# Leap protocol and native bridge evidence

Updated: 2026-09-07T07:40-0700. Plan 05 expands the isolated C0 experiment. The full
offline matrix passes. Two max-effort review rounds completed with three fresh
reviewers each; the final round returned all NO FINDINGS.
The [harness](../../tests/leap_compat/README.md) reproduces the checks using
public artifact pins. The [previous export record](API_RESHAPE_LEAP_EXPORTS.md)
describes plan 04's smaller candidate. C0, C1 and C2 remain open.

## Recorded validation

`tests/leap_compat/build/run-qemg14y5/results.json` records 34 successful
expected checks, including 11 deliberate compiler rejections and seven native
build/compile/run/linkage checks. Swift and Kotlin both encoded and generated
`[0, 1]`, retained position 2 through extraction and failure, and returned
identical bit patterns for all 96 floats from the distinct query `[1, 0, 1]`. The report records 37 probe input
hashes, both KMP profile outputs, pinned public artifacts and native input/output
hashes. Evidence files are ignored; rerun the harness to regenerate them.

The run used Swift 6.3.3 in Swift 5 mode targeting macOS 15 arm64, Kotlin/JVM
2.4.0 with JDK 21, and KMP Kotlin 2.3.20/SKIE 0.10.11/coroutines 1.10.2.
Python integrity tests (3), lint/format and Swift/Kotlin formatting pass.
The native Kotlin compilation reports two existing unused-expression warnings
in the checked-in generated binding; it does not deny binding warnings. The
unchanged consumer compilations do deny warnings. No full workspace CI, device
build, real-model benchmark or release validation was performed. No production Rust code changed in plan 05.

The first review identified a probe that could mask destructive KV reset/replay
because extraction reused the live prompt. The final fixtures use different
content and length and retain the control comparison. Review also corrected the
profile boundary wording below. Both fixes passed the full matrix and fresh
review; no findings were skipped and no actionable findings remain for plan 05.

## Protocol exports

The candidate uses KMP/SKIE with no Leap dependency. Both profiles compile
larger unchanged Swift/Kotlin consumers, including complete member lists for
`Conversation` and the stable `ModelRunner`, callback/handler implementations,
parser subclass overrides, text messages, response construction, options and
explicit Flow/SKIE/enum types. These fixtures also compile against their pinned
public reference artifacts. They cover selected expressions, not every export.

Both profiles export the experimental LoRA lists/scales and hidden-state
dimensions, flat arrays, indexed rows and matrix copies. The extended profile
adds the corresponding ModelRunner methods and newer consumer checks. Swift consumers compile against the
pinned v0.10.13-SNAPSHOT framework and extended candidate. Kotlin's newer
fixture compiles against the candidate, but a newer public Kotlin/Android
positive baseline is still required. Maven JVM 0.10.9 rejects the newer APIs.

Stable Swift custom runners use either imported `__` async methods or `__`
completion-handler methods. Both forms pass with the stable profile. Both
require additional witnesses with the extended profile, just as with the
snapshot. A negative control verifies that Swift protocol-extension defaults
cannot fulfill the Objective-C requirements. Kotlin can use the extended
interface's default newer methods, which explicitly throw unsupported errors.

**Packaging remains undecided.** These profiles are alternative build inputs
for the experiment; they do not make one module preserve both Swift custom
conformance contracts. C0 must resolve version-specific products, a more capable
overlay or another explicit migration policy before promising drop-in support.
Do not silently select the extended profile for stable custom conformers.

## Native CPU boundary

Separate Swift and JVM executables use the existing Cera bindings with the
extended candidate's `HiddenStates` container. Neither implements `ModelRunner`.
The harness builds `cera-ffi` from this worktree and generates a 30,400-byte,
one-block, two-token-vocabulary F32 GGUF; no external model is required.

Each executable checks:

- Invalid model bytes throw a typed Cera FFI error.
- Encoding `ab` produces tokens `[0, 1]`.
- Two sessions remain usable after releasing the model handle; Swift also
  verifies that the model wrapper was deallocated.
- Hidden-state extraction for the distinct query `[1, 0, 1]` returns a finite
  3 × 32 matrix while preserving the live prompt `[0, 1]` and its position; invalid token 9999 throws the typed invalid-token
  error and also preserves position.
- An indexed row is an independent copy of the returned flat data.
- Two generated tokens after successful and failed extraction match an
  untouched control session with the same seed and prompt.

The harness compares every float bit, token and recorded position across the
two processes. The native executable links the candidate's static framework,
the existing Cera wrapper and the freshly built Cera library. A dynamic linkage
inspection rejects a deprecated inference-engine or dynamic Leap dependency.
Original Leap binaries are used only for compile-time baseline checks.

This is evidence for the raw CPU binding boundary, including limited extraction
isolation and ownership. It does not establish model quality, LoRA loading,
chat KV reuse, cancellation, concurrency, GPU behavior or performance.
The Swift conversion makes a primitive Kotlin array call per float; Kotlin uses
a bulk byte-buffer conversion. Neither path has a transfer budget or benchmark.
Large embeddings need a measured bulk transfer design before production use.

## Explicitly incomplete surface

- Loading options expose only the fields exercised by the fixtures, with no
  Cera option translation. Only three Swift generation `with` helpers and one
  loading helper are present; full defaults/builders remain unverified.
- `GenerationOptions.functionCallParser` defaults to nil/null in this sketch;
  the reference's parser default is not implemented. The parser base throws
  for parse/dump, and `LeapFunction` is only a nominal signature placeholder.
- Text messages and response construction are covered; media input types,
  serialization, tool schemas and payload value semantics are incomplete.
  Audio array equality currently follows the generated data-class behavior;
  reference value semantics are still a separate gate.
- Exporting the parser's protected Kotlin StringBuilder produces Kotlin/Native
  append/insert name-collision warnings. The tested subclass overrides compile;
  the complete protected-member contract needs further probes.
- Hidden-state shape/index checks exist in the candidate, but invalid Kotlin
  constructor/index exceptions across Objective-C are not a tested Swift API.
  The native error checks concern Cera's typed FFI exceptions.
- Conversation/history/streams remain abstract. There is no model runner,
  downloader, unload/cancellation policy or working chat facade here.

## Next gates

1. Resolve the stable/newer Swift package contract and pin a newer public Kotlin
   baseline. Complete options/defaults/builders, parsers/tools/media and remaining
   subclasses against unchanged public-artifact consumers.
2. Integrate an actual Cera-backed runner after mapping session/history ownership,
   in-band stream errors, callback order, backpressure, cancellation and recovery.
3. Prove LoRA composition and per-request adapter isolation, seed semantics and
   embeddings without changing live chat state. Measure bulk transfer and warm
   KV behavior under the P0.2/R1 budgets.
4. Validate iOS device/simulator, Android, Swift concurrency modes, SPM/Maven
   substitution, downloader products, package collisions and actual migrations.

The full phase checklist remains in the [handoff](API_RESHAPE_HANDOFF.md) and
[compatibility workstream](API_RESHAPE_LEAP_COMPAT.md). No production dependency
or API is changed by this increment.

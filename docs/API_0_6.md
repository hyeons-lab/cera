# Cera 0.6 API guide

This guide describes the 0.6.2 source API, including the corrections to chat,
checkpoint validation, structured output and audio processing. Build generated
bindings and the native library from the same revision. Package installation
commands select published artifacts; a local version bump does not publish them.

## Choose an API

| Use case | API |
| --- | --- |
| Explicit Rust model loading | `ModelLoader` → `GenerativeModel` → `Session` |
| Stateful Rust chat | `Session::into_chat()` → `SessionChat` |
| Native Swift, Kotlin, Python or Dart chat | `Session.intoChat()` → `ChatSession` (Python uses snake_case) |
| Portable Flutter/Dart generation | Async `Cera` facade; its prompt strings are managed by the caller |
| Browser/Node CPU chat | `cera-wasm` `Session.intoChat()` → `ChatSession` |
| Browser GPU generation | Separate async `WebGpuSession` API |

A typed loader is single-use, including after a failed build. A successful
Session-to-Chat conversion transfers execution state; use the new Chat handle
until `into_session()` / `intoSession()` returns it. Native conversion failure
preserves the original Session. Rust returns it alongside the validation error.
Old native handles cannot cancel the transferred execution state.

Chat validates the model/template profile and session configuration. It rejects
unsupported profiles, sliding context (`n_keep != 0`) and audio-output sessions.
ChatML, Llama and Gemma framing are supported through validated profiles; an
arbitrary Jinja template is not automatically safe for incremental continuation.

## Chat lifecycle and recovery

| Phase | Meaning and next action |
| --- | --- |
| `Idle` | Ingest an initial user message, or a system/user message batch. |
| `PromptReady` | Input is prefilled; generate the assistant response. |
| `TurnComplete` | The profile's terminal marker ended the turn; ingest the next user turn. |
| `Interrupted` | Token budget, cancellation or a nonterminal stop ended generation; reset or replace messages before a new user turn. |
| `RawContext` | Raw execution changed chat framing; replace messages before continuing chat. |
| `Unusable` | A checked reset must succeed, or the session must be recreated. |

A successful generation call or a normally closed text stream does **not** imply
`TurnComplete`. A zero-token call or cancellation before any decode progress
can preserve `PromptReady`; clear cancellation if needed and retry generation
without replaying that prompt. Inspect the actual phase before adding the next
turn. A no-progress `ContextFull` result requires resetting or shortening the
replacement history before generation can advance. `clear_cancel()` /
`clearCancel()` clears a flag; it does not repair an interrupted conversation.
`replace_messages()` / `replaceMessages()` replays a complete replacement history
and reports input tokens from that replacement, rather than from the old position.

Rust ingestion errors carry a recovery outcome. Native bindings expose the
retained outcome through `recoveryStatus()`: unchanged/restored context can be
retained, reset context must be replayed, and unusable state requires reset or
recreation. Read diagnostics after the operation returns. Native `phase()` and
`recoveryStatus()` can return `Busy`; `position()` reads the position atomically
but fails on a moved Chat handle. See the [recovery details](internals/API_RESHAPE_RECOVERY.md).

Executable examples handle the token-limit path and reclaim Session:
[Rust](../cera/examples/chat.rs), [Swift](../cera-ffi/examples/Chat.swift),
[Kotlin](../cera-ffi/examples/Chat.kt), [Python](../cera-ffi/examples/chat.py),
and [Dart](../cera_ffi/example/chat.dart).

## Streaming and cancellation

Swift `AsyncThrowingStream`, Kotlin `Flow`, Python iterators and Dart `Stream`
helpers emit text fragments, not one string per token. Rust `stream_text` buffers
split UTF-8 sequences. Invalid bytes or a trailing incomplete sequence use a
replacement character. Concatenate fragments as received.

In native bindings, use Chat's cancellation method or cancel stream consumption
to request a stop.
In Python, explicitly close the iterator when leaving early; breaking out of a
loop does not close an iterator that is still retained. Let the in-flight
operation finish before resetting, transferring or reusing the session. Normal
stream completion must not be treated as a request to cancel the next call.
Errors still require the lifecycle/recovery checks above.

In native sink callbacks, collect output or call Chat's cancellation method.
Native Chat may hold its operation lock while calling a sink; synchronous
re-entry into another mutating Chat method can deadlock.
Native sink methods must not throw: catch application exceptions inside the
callback and request cancellation. An escaping exception can poison locked
state; a poisoned session requires recreation, not `clearCancel()` or reset.
Raw Swift async Session calls also need explicit `session.cancel()`; the
generated wrapper does not propagate Swift Task cancellation. The Chat stream
helper has its own termination handler.

In browser/Node Chat callbacks, do not re-enter the same Chat handle, including
`cancel()`, property reads, reset or disposal. WASM generation holds a mutable
borrow, and recursive access can leave the handle unusable. An ordinary thrown
JS value propagates as an error after Rust releases its borrow, so inspection or
reset is possible after the outer call returns. This recovery guarantee excludes
recursive handle access and does not promise rollback of generated tokens.
The same re-entry restriction applies to raw CPU Session callbacks, which also
must not throw. The direct CPU bindings cannot signal cancellation through the
active handle; see the [WASM cancellation limits](../cera-wasm/README.md#cancellation).

The generated Python module needs its adjacent `cera_ffi_streaming.py` helper for
`chat.stream`, `chat.stream_json` and `opts.with_json_schema`.
`just bindings` regenerates the module and installs the helper import. Copy the
helper along with the generated module when packaging bindings manually.

## JSON Schema constraints

The compiler implements a subset of JSON Schema. A grammar constrains emitted
prefixes; a token limit or cancellation can still leave incomplete JSON. Parse
and validate the completed result before using it.

| Construct | Behavior |
| --- | --- |
| Primitive types | String, integer, number, boolean and null. |
| `enum`, `const` | Scalar JSON values. |
| Object `properties`, `required` | Declared keys appear at most once in the compiler's fixed order; optional keys may be omitted. Required keys need a corresponding property schema. |
| Object extra keys | With declared properties, output is limited to those properties. `additionalProperties: false` also constrains an empty object schema. |
| Arrays | One `items` schema; nonnegative `minItems`/`maxItems` up to 1024. Inverted bounds are rejected; an omitted maximum permits unbounded length. |
| `$defs`, `definitions`, `$ref` | Local definitions and chained references. Unresolved references are rejected. Direct references reject semantic siblings, except for supported object-property/required merges inside `allOf`. |
| `anyOf`, `oneOf` | Grammar alternatives; `oneOf` does not enforce exclusive matching between overlapping branches. |
| `allOf` | A single scalar wrapper or compatible object-property/required merges. Conflicting property schemas, non-object intersections and unsupported sibling constraints are rejected. |

Supported keywords cannot be combined arbitrarily. `enum`, `const`, `anyOf` and
`oneOf` take precedence over sibling type, object and array constraints; those
sibling constraints are not combined with the selected construct. For example,
`minItems` alongside `anyOf` does not constrain arrays described inside its
alternatives. Put applicable constraints inside each alternative and validate
the result separately when the full schema matters.

This is not a general schema validator: numeric ranges, string length/pattern/
format checks, uniqueness and other unlisted validation keywords are not
enforced. Some unsupported keywords are ignored rather than rejected. Use a
separate validator when those constraints matter. See the
[compiler](../cera/src/grammar/json_schema.rs) for the accepted subset.

Rust uses `GenerateOpts::with_json_schema`; native bindings provide
`jsonSchemaToGrammar` and language helpers; browser/Node options expose
`setJsonSchema`. Swift/Kotlin/Python option helpers return a constrained **copy**,
so use the returned value. `complete_json` / `completeJson` and stream helpers
still follow the same Chat phase rules.

## Tool-call constraints

Tool-call grammars use a separate compiler and do not inherit the JSON Schema
table above. They constrain declared function/argument names and outer value
syntax, but allow omitted required arguments and duplicates; nested array-item
and object-property schemas are not enforced. Lazy triggering permits prose
without a call, and token limits or cancellation can truncate a started call.
Parse the result and validate its arguments against the complete tool schema
before executing it.

## Checkpoints and compatibility

| Surface | Supported checkpoint behavior |
| --- | --- |
| Rust/native CPU Session and Chat | Snapshot host KV/recurrent state, token history, logits and execution counters; Chat also stores phase, tool configuration and terminal bookkeeping. |
| Native Metal/wgpu Session and Chat | Checkpoint and restore are rejected because the model owns device state that a host snapshot cannot represent. |
| Browser/Node CPU Session and Chat | Binary `checkpoint()` / `restore(bytes)` APIs. |
| Browser `WebGpuSession` | Async device-buffer checkpoint/export; synchronous restore/import into a compatible GPU session. |

Rust checkpoints support `to_bytes` / `from_bytes` and file save/load with atomic
rename. Native bindings expose `exportCheckpoint` / `importCheckpoint` and
`saveCheckpoint` / `loadCheckpoint`. Python uses snake_case. A snapshot is not a
model file or a standalone transcript: load the matching model/configuration first.

Restore checks structural fingerprint, sequence position, layer geometry, KV
precision and compression identity, including the TurboQuant seed. WebGPU also
rejects CPU f16 snapshots and uses the compression mode that actually took effect
after fallback. The fingerprint is not a cryptographic hash of the model weights;
keep the model identity alongside saved state. Do not assume arbitrary
cross-model or cross-backend portability.

**0.6.2 compatibility:** recreate f16 and TurboQuant checkpoints made before
compression identity was included in the fingerprint. Plain f32 fingerprints
are unchanged. Native GPU checkpoint attempts that previously produced incomplete
host snapshots now fail explicitly.

## Audio pipeline

`AudioPipeline` / `FfiAudioPipeline` coordinates wake detection, VAD and optional
Whisper transcription over 16 kHz mono float PCM. Process chunks in stream order
on one pipeline. Event sample offsets and timestamps use the stream's global
clock across utterances; explicit reset starts a new stream clock.

The pipeline consumes the remainder of the chunk after a wake event, splits at
detector boundaries, and caps buffered utterances and pre-roll. Duration-driven
splits preserve ongoing VAD state. Resuming wake listening clears discontinuous
old audio while preserving unexpired cooldown.

`AudioPipelineConfig.hotword_config` supplies detector configuration unless an
explicit builder/attached-iterator configuration takes precedence. The Flutter
example's voice controller and portable `Cera` facade are separate from this
native pipeline API; native generated bindings are unavailable in a browser.

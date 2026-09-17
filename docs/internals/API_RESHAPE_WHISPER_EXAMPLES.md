# Whisper transcription in Swift and Kotlin

Cera exposes the standalone `FfiWhisperModel` API through the generated Swift,
SwiftPM and Kotlin bindings. Build the native library and wrappers from the
same revision. This is separate from `CeraEngine.transcribe`, which
uses LFM2-Audio, and from the private `ModelLoader` prototype. F5's unified
Whisper/VAD/hotword loader is still planned.

## Transcribe a recording

Supply a local Whisper GGUF supported by Cera and decoded **16 kHz mono float32
PCM**, normally normalized to `[-1, 1]`. These methods do not open WAV/MP3 files,
capture a microphone, downmix or resample. One call handles the first 30 seconds
(480,000 samples), padding shorter input and trimming longer input. Segment
long recordings before calling; this API does not implement a long-form decoder.

Swift (`import Cera` with the generated package):

```swift
import Cera
import Foundation

func transcribeRecording(modelPath: String, pcm16kMono: [Float]) async throws -> String {
    let model = try await Task.detached {
        try FfiWhisperModel.fromFile(path: modelPath)
    }.value
    var opts = whisperDefaultTranscribeOpts()
    opts.language = "en"
    return try await model.transcribeAsync(pcm: pcm16kMono, opts: opts)
}
```

Kotlin (the same generated `uniffi.cera_ffi` package used on JVM/Android):

```kotlin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import uniffi.cera_ffi.FfiWhisperModel
import uniffi.cera_ffi.whisperDefaultTranscribeOpts

suspend fun transcribeRecording(
    modelPath: String,
    pcm16kMono: List<Float>,
): String {
    return withContext(Dispatchers.IO) {
        FfiWhisperModel.fromFile(modelPath).use { model ->
            val opts = whisperDefaultTranscribeOpts().copy(language = "en")
            model.transcribeAsync(pcm16kMono, opts)
        }
    }
}
```

Both helpers are compiled and executed by the
[Swift](../../tests/whisper_ffi/WhisperProbe.swift) and
[Kotlin](../../tests/whisper_ffi/WhisperProbe.kt) probes. Loading is synchronous,
so these examples move it off the UI thread. Reuse a loaded model for repeated
recordings instead of parsing it each time. Swift releases the handle through
ARC; Kotlin's `use` closes it after the operation.

The synchronous alternative is `model.transcribe(pcm: pcm, opts: opts)` in Swift
and `model.transcribe(pcm, opts)` in Kotlin. It runs the decoder on the caller's
thread. The async method runs it on a Rust blocking worker and returns one
complete string. Dropping the Rust future aborts queued work and sets a per-call
cancellation flag for a running decoder. The decoder stops at its next cooperative
check; cancellation does not synchronously interrupt a running kernel. Kotlin's
generated cancellable continuation frees that future on coroutine cancellation.
The pinned Swift generator uses an unsafe continuation without a task cancellation
handler: **`Task.cancel()` does not cancel this Swift transcription call**. It
continues until the decoder completes. Closing that Swift cancellation gap
remains a foreign migration gate. The FFI option record does not expose a separate
cancellation handle, and synchronous calls have no foreign cancellation method.
There is no token stream or progress callback here. This is the upstream 0.5.5
behavior retained by Plan24 and supersedes Plan21's queued-work-only Rust future
cancellation limitation.

## Options and ownership

| Field or method | Behavior |
|---|---|
| `language` | `nil`/`null` or `"auto"` requests detection on multilingual models; an explicit code such as `"en"` selects it. |
| `translate` | Defaults to `false`; `true` requests translation into English on a multilingual model. |
| `timestamps` | Defaults to `false`; `true` enables the core's timestamp-token behavior, with the same string return type. |
| `maxTokens` | `UInt32?` / `UInt?`; default or absent value is 448, further limited by the model's text context. |
| `temperature` | `Float?`; default or absent value is `0.0` (greedy). |
| `opts: nil` / `opts = null` | Uses all default options. |
| `isMultilingual()` | Checks for the model's transcription task token. |
| `languages()` | Returns the standard 100-code Whisper table, even for an English-only model; it is not a per-model capability list. |

`fromBytes(bytes:)` accepts Swift `Data`; `fromBytes(bytes)` accepts Kotlin
`ByteArray`. The FFI copies the supplied bytes into owned Rust storage; the
caller can release or mutate its original buffer. File loading retains its
weight backing. Leave a mapped file's contents unchanged while the model is
alive. Calls share immutable weights and allocate their own Whisper decode
state; this does not change generative `Session` KV ownership.

Malformed GGUF and missing-file errors cross as Swift `FfiError.Backend` or
Kotlin `FfiException.Backend`. Empty PCM returns an empty string.

## Run the checks

From the worktree:

```bash
cargo test -p cera-ffi --test whisper_ffi --locked --offline

# Optional: inspect the small generated models and raw PCM input.
cargo run -p cera-ffi --example whisper_fixture -- /tmp/whisper-inputs

# macOS arm64, Xcode tools, Kotlin compiler, and JDK 21 on PATH.
# JAVA_HOME must select JDK 21. The artifact directory holds the two pinned
# JNA/coroutines jars listed in tests/leap_compat/artifacts.json.
python3 tests/whisper_ffi/run.py --artifacts /path/to/cached-jars
```

The probe builds the current library, compiles the checked-in generated wrappers,
and verifies 11 cases in each language: owned bytes, retained file weights,
language metadata, defaults, sync transcription, shared-model async calls,
distinct-model async calls, empty input, malformed bytes, a missing file, and
the recording helper above. It requires exact `aaa`/`aa`/`bb` outputs and writes
command logs plus a JSON report with source, fixture, dependency and binary
hashes into a fresh temporary directory. No downloaded model test is skipped.

The [fixture](../../cera-ffi/tests/common/whisper_fixture.rs) contains complete,
tiny synthetic weights. Real preprocessing, encoder and decoder paths execute,
but its projection deliberately picks a fixed character regardless of the
audio. This proves loading, marshaling and lifetime behavior, **not speech
recognition accuracy**. Recorded-audio quality, long recordings, device
performance, Android/iOS packaging and release publication remain separate gates.

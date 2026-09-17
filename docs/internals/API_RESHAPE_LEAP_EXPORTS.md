# Leap C0 export evidence

Historical plan 04 record. The [protocol/native increment](API_RESHAPE_LEAP_BRIDGE.md)
extends this candidate; the old evidence hashes do not describe current inputs.
Use that document and the handoff for current results and next steps.

Updated: 2026-09-07T00:56-0700. The first export increment is implemented and
its full compile matrix passes. C0 remains open: these are isolated compiler
probes, not an implemented runtime or a complete replacement package.

Two max-effort review rounds completed, each with three fresh reviewers. The
first found a Swift custom-conformance documentation/coverage gap; the async
fixtures and corrected contract below resolve it. The final round was clean.

## Reproduction and artifact identity

Run the [probe harness](../../tests/leap_compat/README.md). The machine-readable
[artifact manifest](../../tests/leap_compat/artifacts.json) pins the exact
Maven JVM 0.10.9 binary, Swift v0.10.9/v0.10.13-SNAPSHOT core frameworks, and
coroutines JAR by SHA256. Every run verifies original binaries, records fixture
hashes and tool versions, and creates separate compiler logs. No source archive
or development checkout is needed to reproduce the probes.

Public references: [Swift stable release](https://github.com/Liquid4All/leap-sdk/releases/tag/v0.10.9),
[Swift snapshot release](https://github.com/Liquid4All/leap-sdk/releases/tag/v0.10.13-SNAPSHOT),
[Maven JVM metadata](https://repo.maven.apache.org/maven2/ai/liquid/leap/leap-sdk-jvm/maven-metadata.xml),
[conversation documentation](https://docs.liquid.ai/deployment/on-device/sdk/conversation-generation).
The snapshot release notes mention embeddings; the exact availability below is
established by the pinned artifact checks rather than the archived docs alone.

## Observed matrix

| Consumer | Swift stable 0.10.9 | Swift snapshot 0.10.13 | Kotlin/JVM stable 0.10.9 |
|---|---|---|---|
| Core generation/options/history and callback interfaces | Compiles | Same source compiles | Compiles |
| Explicit Flow/SKIE type and exhaustive response switching | Compiles | Same source compiles | Compiles |
| Custom stable `ModelRunner` implementation (both Swift conformance forms) | Compiles | Rejected: new required methods | Compiles |
| LoRA list loading/swap and hidden-state results | Rejected: missing exports | Compiles | Rejected: missing exports |
| Custom runner plus newer method implementations | Not applicable | Compiles | Newer baseline still needed |

The newer Kotlin fixture is a required target, not a verified positive baseline:
its rejection establishes that Maven 0.10.9 is insufficient. Pin a public newer
distribution and get that same fixture compiling before claiming compatibility.
No claim is made about an unpublished Android artifact or an Android device build.

Swift custom runners can implement the imported `__` async methods or the
corresponding `__` completion-handler methods. The double-underscore spelling
is required for these protocol witnesses; the friendlier public async names
alone do not satisfy conformance. Both forms have compiled forwarding fixtures,
including the boxed `KotlinInt` result and Sendable completion callbacks.
Adding the snapshot's `__hiddenStates` and `__setLoraAdapters` implementations
restores conformance without editing either stable fixture. A future
replacement must deliberately support custom conformers from both baselines;
merely adding new abstract protocol requirements would break stable conformers.

The snapshot also exports load-time `loraAdapters`, `LoraAdapterConfig(path, scale)`,
and `HiddenStates` with flat data, dimensions, indexed rows and matrix copies.
Compilation proves their names/types, not validation, numerical results, copying,
adapter composition, state isolation, or failure behavior. Those remain C1 gates.

## Architecture experiment

The same typed Swift consumers are compiled against each independent candidate:

| Candidate | Nonthrowing typed stream | Objective-C stream bridge | Response enum switching |
|---|---|---|---|
| Alias to `AsyncThrowingStream<T, Error>` | Rejected: `Failure` differs from `Never` | Rejected: bridge missing | Not tested |
| Alias to `AsyncStream<T>` | Compiles | Rejected: bridge missing | Not tested |
| KMP/SKIE export | Compiles | Compiles | Compiles |

The KMP candidate also compiles the unchanged Kotlin typed Flow/enum consumer.
It builds without a Leap SDK dependency and contains no model implementation.
Kotlin 2.3.20 plus SKIE 0.10.11 is a tested combination for this limited export;
see the public [SKIE release note](https://skie.touchlab.co/changelog/0.10.11)
and [installation guide](https://skie.touchlab.co/Installation).

**Decision:** carry KMP/SKIE forward as the next candidate for the shared
compatibility surface. Plain standard-stream aliases are insufficient for the
tested contract. This does not rule out a more complete Swift overlay, freeze
the full architecture, or prove a native Cera bridge. The next spike must extend
the candidate rather than mistake this narrow positive result for SDK parity.

## Remaining C0 work

1. Extend the KMP candidate to the complete runner/conversation protocols,
   supported subclasses, options/defaults/builders, message payloads, tools and
   LoRA/hidden-state exports. Compile the already validated larger consumers
   against it unchanged, including stable Swift custom conformers.
2. Pin the newer Kotlin/Android LoRA/embedding distribution and validate the
   newer Kotlin fixture positively. Snapshot Swift is not evidence of a matching
   Maven release.
3. Add a compiled Cera bridge prototype on Apple and JVM/Android and confirm
   ownership, exceptions, cancellation, callback order, and float-buffer transfer.
   The current KMP project has neither a bridge nor inference implementations.
4. Expand beyond macOS arm64/JVM and Swift 5 mode: iOS device/simulator, Android,
   the supported Swift concurrency modes, SPM/Maven dependency substitution,
   downloader products, and package/link collision checks. Decide optional UI,
   macros, cloud and background-service coverage explicitly.
5. Preserve all C1/C2 runtime and warm-KV gates in the
   [compatibility workstream](API_RESHAPE_LEAP_COMPAT.md). No performance or
   runtime conclusions follow from this compile matrix.

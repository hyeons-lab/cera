# Leap SDK compatibility layer

Status: required migration workstream; no compatibility package is implemented
yet. Swift and Kotlin have equal priority. Cera replaces the deprecated SDK's
runtime; the compatibility distribution must not depend on that runtime.

C0 progress: the [first export probes](API_RESHAPE_LEAP_EXPORTS.md) and
[protocol/native increment](API_RESHAPE_LEAP_BRIDGE.md) validate selected
unchanged public-artifact consumers, two KMP/SKIE export profiles and a raw Cera
CPU boundary in Swift/JVM. Stable/newer Swift packaging and the newer Kotlin
positive baseline remain unresolved. C0 and full runtime compatibility remain
open; no warm-chat performance claim follows from these probes.

## Baselines and references

Verified on 2026-09-06:

- Maven Central reports 0.10.9 for both latest and release in the
  [core metadata](https://repo.maven.apache.org/maven2/ai/liquid/leap/leap-sdk/maven-metadata.xml)
  and [JVM metadata](https://repo.maven.apache.org/maven2/ai/liquid/leap/leap-sdk-jvm/maven-metadata.xml).
- Swift's latest stable release is [v0.10.9](https://github.com/Liquid4All/leap-sdk/releases/tag/v0.10.9).
  The newest published prerelease is
  [v0.10.13-SNAPSHOT](https://github.com/Liquid4All/leap-sdk/releases/tag/v0.10.13-SNAPSHOT).
  Keep stable and snapshot fixtures separate; a snapshot is not evidence of a
  matching Maven Central release.
- The public [overview](https://docs.liquid.ai/deployment/on-device/sdk/overview)
  and [changelog](https://docs.liquid.ai/deployment/on-device/leap-sdk-changelog)
  are archived, and the latter still identifies 0.10.7. They explain concepts;
  their version label does not override published release metadata.

Stable 0.10.9 is the regression baseline, not a ceiling on compatibility. LoRA
and per-token embeddings are explicitly required target capabilities. C0 must
pin the exact distribution that exports each newer method on each platform and
compile consumers against it. Do not claim those methods shipped in 0.10.9 from
the documentation alone. Cite only public documentation/release pages in this
plan and migration material; no development-repository references.

## Meaning of drop-in

The acceptance target is **source compatibility after changing the dependency
declaration and rebuilding**: existing supported imports, constructor labels,
interfaces/protocols, option builders, and generation call sites compile
unchanged. Binary compatibility with an already linked app is not promised.
Keep a precise symbol/behavior/platform matrix; a text-only milestone is not a
complete SDK replacement. No stubs that silently ignore a requested feature.

Provide replacement Swift products/modules for supported `LeapSDK` and
`LeapModelDownloader` imports and Kotlin package names used by supported apps.
Use explicit replacement coordinates/dependency substitution under the Cera
distribution; do not accidentally resolve the original engine transitively.
Test dependency collisions, linking, native packaging, and installed app startup.
Public packaging reference:
[Quick Start](https://docs.liquid.ai/deployment/on-device/sdk/quick-start).

Do not assume handwritten Swift classes plus `AsyncThrowingStream` reproduce
Kotlin/Native protocol identity, concrete SKIE flow types, or `onEnum(of:)`.
C0 must compile type-annotated streams, protocol implementations, subclasses,
builders, and enum switching, not just one inferred `for await` loop. Compare a
KMP export facade and platform overlays using these fixtures before choosing the
bridge architecture. Reuse Cera's binding implementations where the contract
fits; any extra native bridge requires its own compiled prototype and review.

## Contract matrix

| Surface | Adapter responsibility | Required evidence |
|---|---|---|
| Loading and downloaders | Preserve local/manifest/bundle source semantics, progress/cancel and supported cache/download controls; translate explicit companions | Offline load and cache hits, failures, cancellation; platform background/service behavior audited separately |
| `ModelRunner` / `Conversation` | Runner owns lifecycle; conversation owns transcript and a Cera execution session | History creation/edit/export, concurrent calls, unload with live conversations, read-only history after unload |
| `ChatMessage` and content | Lossless role/text/reasoning/tool/media representation | Round trips, optional/null fields, JSON export/import, image/audio layouts |
| `GenerationOptions` | Resolve nullable overrides against model defaults into a complete Cera config | Every field, default/precedence, explicit seed, constraints, thinking and parser controls |
| Streaming | Preserve response variants/order, completion/error semantics, Flow/SKIE behavior and cancellation | Split UTF-8, slow consumers, early termination, callback reentrancy, exactly-once terminal delivery |
| LoRA at load and runtime | Preserve adapter lists, independent scales, clearing, ownership and serialized replacement | Multiple overlapping targets, failures without partial change, generation/unload races, repeat activation and memory lifetime |
| Hidden states / embeddings | Per-token raw final-layer output with independent per-call adapters | Tokenization/BOS, shape/byte order, no pooling/L2 normalization, capacity errors and generation-state isolation |
| Tools, constrained generation, audio and other products | Preserve each promised capability or record a release-blocking gap | Feature-specific contract tests; no successful empty output for unsupported operations |

Public concepts: [loading](https://docs.liquid.ai/deployment/on-device/sdk/model-loading),
[conversation and generation](https://docs.liquid.ai/deployment/on-device/sdk/conversation-generation),
[messages](https://docs.liquid.ai/deployment/on-device/sdk/messages-content),
[generation options](https://docs.liquid.ai/deployment/on-device/sdk/advanced-features).
These links do not establish the exact signatures of newer LoRA/embedding
extensions; C0 must verify their exported surface before freezing it.

## Execution rules

The adapter's history is outside the core Session, preserving D7. First use and
history edits call explicit replacement. On supported, completed warm turns,
append only new input and required boundary tokens to the resident session.
Track whether rendering inputs changed: history edits, tools, system/schema
injection, template settings and interrupted generations can invalidate that
append path. Never infer completion from zero emitted tokens alone. R0/R1 state
and error contracts remain required.

Leap options are nullable overrides; Cera's new generation config is complete.
Perform that translation deliberately. A supplied per-request seed cannot be
ignored because Cera currently seeds at session creation/reset. Solve seed
application without a normal-turn KV reset, or keep that mapping blocked.
An altered RNG policy needs explicit fixtures/release notes; numerical output
identity across different inference engines is not automatically promised.

Serialize generation, LoRA swaps, hidden-state extraction, and unload at the
runner boundary where required. A mutex alone cannot isolate model-owned GPU
state across conversations. Multiple live conversations need a proved backend
isolation strategy; silently replaying full history on every alternating turn
does not satisfy the warm profile. Publish lifecycle/phase changes before
delivering terminal callbacks. Cancellation must reach Cera's existing atomic
handle without waiting for a generation lock. Bounded queues must neither drop
events nor grow with unlimited buffered model output; preserve the original
platform event/collector threading semantics instead of applying a blanket
main-thread rule to every low-level callback.

Prefix/disk cache options are distinct from the live Session KV requirement.
Preserve opt-in disk behavior and storage ownership. A replacement may gain
live KV reuse while retaining the documented default of no disk cache. Compare
equivalent rendering/sampling policies and report changed defaults explicitly.

### LoRA coverage

Target load-time `loraAdapters` and runtime `setLoraAdapters`, including a list
of path/finite-scale pairs whose contributions stack; an empty set clears the
active adapters. Preserve replacement completion ordering and keep validation
failures from partially installing a new set. Adapter caching and unload must
be explicit and measured; deactivation must not be documented as memory release
unless the chosen compatibility contract actually provides it.

Cera currently exposes a single attached `LoraAdapterWeights` object per
Session. That is not proof of arbitrarily many independently scaled adapters
on overlapping targets. Implement and test any needed composition/bridge as
separate runtime work before advertising list compatibility. Keep core LoRA's
future-forward-only behavior and reset retention unchanged. Verify how runner
adapter changes affect existing conversations and their cached rows; any
required adapter-layer replay is an explicit LoRA operation with separate
performance evidence, not a hidden cost on every ordinary warm turn.

### Embedding coverage

Target `hiddenStates(text, adapters)` and the `HiddenStates` result with
`data`, `tokenCount`, `embeddingDim`, indexed vector copies and matrix copies.
Require raw text tokenization with model-default BOS, no chat template, no
implicit pooling or L2 normalization, and checked row-major shape arithmetic.
Empty/oversized input and unsupported backends must have verified error behavior.

An empty per-call adapter list means the base model, independently of adapters
active for generation. Extraction must leave conversation history, KV, pending
boundaries, logits, RNG and generation adapters intact, including failure paths.
Cera's existing hidden-state methods inherit the session adapter, so direct
delegation on a chat session is insufficient. Use proved isolated execution and
serialize backend-owned resources as necessary; do not temporarily swap a live
chat adapter and assume that restoring its pointer restores all state. Preserve
float layout without per-element boxing of a large token-by-dimension buffer.

## Delivery gates

**C0 — Compatibility contract and export prototype (alongside P0).** Inventory
all target symbols, stable/snapshot differences, errors/defaults and retained
capabilities. Pin package metadata and artifact hashes. Compile unchanged Swift
and Kotlin fixture apps against originals, then the proposed replacement shapes.
Include LoRA/embeddings and explicitly resolve any absent artifact or undocumented
export. Decide KMP/overlay architecture from those results. UI, cloud client,
macros, background downloading, and Android model service each need an explicit
support decision; none becomes supported through the core facade by implication.

**C1 — Runtime adapters and missing prerequisites.** Implement loading and the
documented conversation/streaming contract over Cera. Chat shipping requires
P0.1/P0.2, R0/R1 and the relevant P1 surface. LoRA composition, per-request RNG,
embedding isolation, tool lifecycle and platform streaming must each pass their
own tests where needed. Platform Flow/SKIE adapters may wrap callbacks without
forcing a Rust pull-stream redesign; they still need cancellation/backpressure
proof. Do not describe these semantic additions as simple API aliases.

**C2 — Migration release (alongside P2).** Rebuild actual Swift/Kotlin consumers
with only dependency changes for the declared supported matrix. Run lifecycle,
media/tool, LoRA and embedding contracts and the warm-session performance suite
through these public entry points. Confirm no old engine runtime is required.
Publish migration instructions and compatibility limits. Keep the bridge usable
while apps incrementally adopt Cera's native API; its support period is distinct
from P3's removal of superseded Cera methods and requires an explicit release
policy before deprecating the bridge itself.

# Checked rewind and ingestion recovery

Plan37 replaces the unconditional `append_user_message` failure rollback with
bounded Session recovery. CPU Llama/LFM2 support checked rewind and full reset.
Plan38 implements checked Metal/native wgpu reset; unknown implementations and
browser synchronous GPU reset remain unsupported. R0 remains open for new chat/browser recovery and the full raw/chat state inventory.

## User-message recovery

`append_user_message` retains `Result<(), CeraError>` and returns the original
failure. `Session::last_ingest_recovery()` separately reports `Unchanged`,
`Restored`, `Reset` or `Unusable`, plus any rejected rewind and secondary reset
error. `Session::is_usable()` reports whether inference is permitted. These are
additive Rust diagnostics. Native bindings expose an owned `recoveryStatus()`
snapshot; existing foreign operation errors keep their shapes.

The checkpoint saves position, history length, last logits and prefill metrics.
It copies no KV or accumulated history. A mutation marker is set **before** each
forward/shift attempt, so zero reported progress cannot masquerade as an
unchanged append. A shift prevents restoration even if the final position equals
the checkpoint. Sampler and drafter are untouched by ordinary ingestion; a
shift resets the drafter and forces reset instead of restoration. A backend
rewind is checked before ingestion and revalidated when recovery runs.

A complete reset invalidates cache positions and clears rolling convolution,
logits, draft history/drafter and prefill metrics, and rebuilds the seeded sampler. CPU reset recreates the
compressed state too; the uncompressed scratch helper is insufficient. Attached
adapters/encoders, configuration and the existing position/cancellation handles
survive. Automatic recovery **never writes the cancellation flag**, including
when another thread requests cancellation during reset. Caller-requested
`reset()` retains its flag-clearing behavior.

The [runnable recovery example](../../cera/examples/ingestion_recovery.rs)
interrupts a message, reads the outcome, supplies context again after reset (or
recreation), explicitly clears cancellation, and retries:

```sh
cargo run -p cera --example ingestion_recovery -- model.gguf "The answer is" " probably yes"
# CPU LFM2 with TurboQuant: a mutated failed append requires a full reset.
cargo run -p cera --example ingestion_recovery -- lfm2.gguf "The answer is" " probably yes" --compressed
```

Metal fences the command queue before clearing shared convolution memory. Native
wgpu submits convolution clears, then uses a strict four-byte readback to wait
for completion and propagate mapping/device errors. Device attention tails,
including compressed tails, are invalidated by resetting the backend length.
Recovery adds no full-cache readback or copy to successful ingestion; existing
fresh-prefill prefix-cache insertion can still snapshot device state. These
operations keep the model's prefix cache and attached weights available. Metal f16 and wgpu
f32 remain the respective standard device formats; requesting F16 also follows
those existing backend policies.

The same runnable example selects a device with an explicit flag:

```sh
cargo run -p cera --features metal --example ingestion_recovery -- model.gguf "The answer is" " probably yes" --metal
cargo run -p cera --features gpu --example ingestion_recovery -- model.gguf "The answer is" " probably yes" --wgpu --compressed
```

Native device tests inspect snapshots immediately after reset, then compare raw
embedding/token continuation and Session replay with independently loaded models.
They cover hybrid and dense architectures, standard/F16/TurboQuant requests,
actual compression, adapters in the compressed cases, cancellation and ownership.
A destroyed wgpu device must return a reset error and cannot re-enable Session
inference. Browser GPU completion needs an async recovery API; the synchronous
Model reset remains unsupported there. Run device checks explicitly on a host
with both backends; unavailable hardware is a failure, never a successful skip:

```sh
cargo test -p cera --lib --features gpu,metal gpu_recovery -- --ignored --nocapture --test-threads=1
```

On `Unchanged`/`Restored`, retry subject to the previous state and cancellation.
On `Reset`, the old context is gone. On `Unusable`, all Session ingestion,
generation and hidden-state extraction reject before executing or clearing
cancellation. A successful checked explicit reset or recreation is required;
a legacy backend reset's zero counter alone cannot re-enable an unusable session.
The ingestion guard also marks unwound forward/recovery attempts unusable.

Raw `append_tokens`, `append_embeddings` and composed media helpers retain their
existing partial-prefill behavior outside `append_user_message`; they do not set
the user-message diagnostic. Raw callers must reset/replay after a partial or
composed-input failure unless they independently establish a valid continuation.
Foreign legacy callers should reset/replay on an ingestion failure and recreate
if reset is rejected. New chat/browser recovery and generation/speculative
failure semantics remain follow-up work; this does not close the full R0 gate.

## Available raw operations

`Model::check_kv_rewind(state, position)` checks current capability without
mutation. `Model::try_truncate_kv(state, position)` checks again and either
rewinds all supported KV layers or returns `KvRewindError` without mutation.
State must belong to the model and its prefix must have been produced in the same
causal execution mode. The caller must know that no adapter change, context shift or
replacement has invalidated that prefix. An earlier successful check cannot
reserve a convolution checkpoint against subsequent prefill.

The [runnable Rust example](../../cera/examples/checked_rewind.rs) loads an actual
CPU model, prefills a prefix and a discarded suffix, rewinds, and forwards a
replacement suffix. It uses the loaded model's raw CPU state and does not reuse
logits from discarded tokens:

```sh
cargo run -p cera --example checked_rewind -- model.gguf "The answer is" " uncertain" " clear"
```

`InferenceState::check_truncate_to` and `try_truncate_to` expose the CPU-state
primitive for advanced callers. The model methods are the correct boundary when
a backend may own execution state. No full KV snapshot, accumulated token copy,
or device readback is added. Validation scans layer metadata and at most the
existing 64-entry history ring per convolution layer.

## Backend matrix

| Backend/cache | Checked KV rewind | Reset and remaining proof for R0 |
|---|---|---|
| CPU Llama/dense, f32/f16 | Supported when complete row layouts and target bounds validate | Session restoration and continuation tested; CPU reset supported |
| CPU LFM2 hybrid, f32/f16 | Supported when every convolution target is still in its ring; zero clears each ring | Revalidated after ingestion; expired checkpoints use a verified CPU reset |
| CPU LFM2 with bidirectional attention or classifier LoRA | Rejected as `NonCausal` before mutation; a retained prefix can depend on the discarded suffix | Checked CPU reset supported; replay context for a fresh computation |
| CPU TurboQuant, including one-sided compression | Rejected without mutation, including target zero and no-op | Checked CPU reset rebuilds both compressed and uncompressed payloads; three compression combinations tested |
| Native wgpu and Metal, hybrid convolution | Checked tail rewind returns `BackendUnsupported`; legacy counter-only rewind remains | Locked checked reset clears device convolution and both positions; completion is verified before reporting Reset |
| Native wgpu/Metal attention-only | Checked tail rewind not implemented | Checked reset supported; implementing rewind remains a separate optimization |
| Browser WebGPU | Synchronous checked rewind/reset remain unsupported | Requires asynchronous completion and error semantics; do not block the JS event loop |
| BERT or external Model implementations | Defaults return `BackendUnsupported` | Checked reset defaults to unsupported; explicit opt-in requires execution-state proof |
| CPU WASM | Same CPU core methods compile; no new JS rewind export | Same core recovery/guards; richer foreign diagnostic exports and runtime proof remain |

Bounds, compression, late-layer layout faults and missing convolution snapshots
are rejected before any layer changes. Missing convolution snapshots no longer
cause a silent reset on the **new checked path**. The legacy raw truncation methods keep their
signatures and behavior; user-message recovery now uses the checked path. A no-op on uncompressed state
does not change or validate cache contents; it only validates the requested
position. None of these primitives reconstructs a prefix destroyed by shifting.

## Native recovery status

Swift/Kotlin `session.recoveryStatus()` returns `SessionRecoveryStatus` with
`usable`, `position` and optional `lastIngestRecovery`. The latter contains a
`RecoveryOutcome`, optional typed `KvRewindFailure` and optional existing
`FfiError` (`FfiException` in Kotlin) for a secondary reset failure. The original
message operation still throws its original error. Rewind causes preserve bounds,
layer/position and layout details without requiring message parsing.

The snapshot uses a nonblocking session lock. It throws `Busy` while an
operation holds that lock, including a token callback invoked under the lock,
and `Backend` for a poisoned lock. Terminal callbacks and final buffered text
flushes can run after unlocking; status may already be available there.
Recreate a poisoned session. Read status after the failed operation returns;
it is a snapshot, not a reservation against later calls by other threads. An
unusable session's position is diagnostic and must not be treated as valid KV.

- `Unchanged`/`Restored`: retain the prefix if `usable` is true.
- `Reset`: replay context before retrying.
- `Unusable`/`Unknown`: successfully reset or recreate; conservatively recreate
  for an unknown future outcome.
- No retained report: no whole-message failure is recorded. This gives no
  transaction guarantee for raw append calls.

Observation never changes cancellation or consumes the report. Successful
whole-message ingestion and explicit reset clear it; raw append calls and cancel
controls leave it unchanged. A combined send/generate failure during generation
has no ingestion report if its ingestion succeeded.

The complete [Swift example](../../cera-ffi/examples/IngestionRecovery.swift) and
[Kotlin example](../../cera-ffi/examples/IngestionRecovery.kt) demonstrate a
cancelled multi-token message, outcome inspection, conditional prefix replay and
an explicit `clearCancel()` before retry. Both use public imports. The
[native consumer checks](../../tests/api_recovery/README.md) compare resumed
sampling with independently loaded models and exercise streaming reentrancy.
Generated Python/Dart expose the same native snapshot. CPU WASM currently has
only raw append methods, so an equivalent diagnostic belongs with its future
whole-message/chat facade. Browser WebGPU recovery still requires an async API.

## Existing rewind callers

| Caller | Current purpose | Required follow-up |
|---|---|---|
| [Session::append_user_message](../../cera/src/session.rs) | Bounded complete recovery with checked rewind/reset and unusable enforcement | Native status is exposed; carry recovery into new chat/browser methods |
| `Session::generate_greedy_spec` in the same file | Drop verified tokens after budget/stop boundary | Guard backend capability and retain correct generated/pending-token and phase bookkeeping |
| [spec::verify_draft](../../cera/src/spec.rs) | Drop rejected draft suffix | Account for fallible rewind without weakening logits/token oracle |
| `spec::greedy_generate_spec` in the same file | Drop accepted tokens beyond a stop/budget | Preserve output/position accounting on refusal or recovery |
| [DSparkSessionDrafter::prepare_draft_step](../../cera/src/model/dspark.rs) | Reset a divergent draft prefix or rewind to synchronized context | Include drafter state and synced token history in session recovery inventory |
| `DSparkSessionDrafter::draft` in the same file | Restore its local draft prefix | Validate target/context continuity without claiming target-model restoration |
| [GpuLfm2Model::truncate_kv_direct](../../cera/src/model/gpu_lfm2.rs) | Public direct counter rewind; no in-tree call sites | Retain signature; classify external callers before changing its behavior |
| `Model::truncate_kv` defaults and CPU state `truncate_to` | Existing raw/speculative extension points | Keep available; use checked methods for new recovery guarantees |

Plan37 migrates only user-message rollback. Speculative and drafter failure
semantics remain separate work; their old signatures do not imply checked recovery.

## Session recovery still required

- [x] Establish a bounded existing-Session checkpoint covering position/mirror,
      logits, draft history/drafter state, prefill metrics and sampler behavior.
- [x] Detect existing destructive shifts and expired convolution checkpoints.
- [ ] Include future replacement/chat phase/profile/pending-token bookkeeping.
- [x] Add verified CPU execution reset preserving concurrent external cancellation.
- [x] Preserve primary errors, record secondary reset errors and report
      `Unchanged`, `Restored`, `Reset` or `Unusable` honestly through the additive Rust diagnostic.
- [x] Enforce unusable state through all current Session execution entry points.
- [x] Expose an owned native recovery snapshot with typed secondary errors.
- [ ] Carry enforcement and richer outcomes into new chat/browser methods.
- [x] Prove native device reset before new prefill and CPU metadata restoration with fault injection.
- [ ] Complete browser async device recovery and future WASM chat diagnostics.
- [ ] Complete P0.1's chat/raw caller and named profile inventory, then P0.2/R1's
      bindings, phase/boundary behavior and numerical KV performance gates.

The [main contract](API_RESHAPE_PLAN.md#43-ingestion-failure-and-recovery-prerequisite-r0)
defines the recovery outcomes. This inventory fixes the primitive's scope; it does
not close R0 or the broader chat gates.

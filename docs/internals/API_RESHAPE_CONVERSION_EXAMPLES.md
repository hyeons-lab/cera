# Run SafeTensors conversion through the loading API

Status: Plans15 and17–19 are complete within this CPU fixture scope. Conversion
uses a resolved upstream commit and verifies local completed/checkpoint bytes.
Eleven scoped gates and the final max-effort review are complete. See the
[handoff](API_RESHAPE_HANDOFF.md#completed-plan-19-upstream-conversion-revisions).

These eleven executable tests serve a complete one-block Llama fixture over
loopback HTTP, convert its SafeTensors weights to GGUF, and load the result through
the private `ModelLoader` prototype and existing `CeraEngine` constructors.
They need cached Cargo dependencies and permission to listen on `127.0.0.1`.
No account, external model download or generated binding workspace is required.

Run from the implementation worktree root:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::conversion --locked --offline \
  -- --test-threads=1
```

For the same contracts without default features, replace `--features remote`
with `--no-default-features --features remote,mmap`. Cargo's `--offline` controls
dependency resolution. The fixture separately isolates HTTP with a subprocess,
a local `HF_ENDPOINT` and a proxy that rejects external tunnels.

## Load, convert, retain and continue

The [complete conversion example](../../cera/src/engine/loading_prototype/tests/remote/conversion.rs)
uses this private API sequence inside its isolated fixture:

```rust
let source = ModelSource::hugging_face(
    "fixture/single:invalid@release",
    Some("Q8_0"),       // Explicit quantization overrides the spec suffix.
    Some("fast-mse"),
);
let model = ModelLoader::new(source)
    .config(load_config) // CPU, BundleRepo with a temporary store and progress callback
    .build_generative()?;
let mut session = model.create_session(SessionConfig::default())?;
drop(model);
session.append_tokens(&[0, 1])?;
session.generate(&options, &mut sink)?;
session.append_tokens(&[1])?;
```

The fixture repositories exist only in the test server. The linked test supplies
all configuration, weights and callbacks; this excerpt is not a standalone
public API program. For standalone Rust and generated Swift/Kotlin examples,
see the [API walkthroughs](API_RESHAPE_EXAMPLES.md).

Seven conversion configurations run through typed, dynamic and legacy loaders:

| Repository and revision | Output | Strategy | Cache directory under the configured store |
| --- | --- | --- | --- |
| `fixture/single`, `main` | F32 | auto | `huggingface.co/fixture/single/quantized/F32/` |
| `fixture/single`, `release` | F32 | auto | `huggingface.co/fixture/single/quantized/F32@release/` |
| `fixture/sharded`, `main` | F32 | auto | `huggingface.co/fixture/sharded/quantized/F32/` |
| `fixture/single`, `release` | Q8_0 | fast-mse | `huggingface.co/fixture/single/quantized/Q8_0-fast-mse@release/` |
| `fixture/single`, `main` | Q4_0 | hqq | `huggingface.co/fixture/single/quantized/Q4_0-hqq/` |
| `fixture/single`, `main` | Q4_0 | auto | `huggingface.co/fixture/single/quantized/Q4_0/` |
| `fixture/single`, `main` | F16 | unknown name falls back to auto | `huggingface.co/fixture/single/quantized/F16/` |

The [input builder](../../cera/src/engine/loading_prototype/tests/remote/conversion/fixture.rs)
maps eleven known tensors explicitly from the independent original GGUF fixture.
The F32 outputs must reproduce every tensor's shape and bytes and match original
inference logits. Quantized outputs must have the expected GGML types and bounded
per-weight error against those original tensors, plus relative L2 error below
15% for every quantized tensor. This relative bound rejects zeroed matrices.
HQQ and auto produce different
Q4_0 files. These tiny weights leave norms and the small embedding in F32 for
Q4_0/Q8_0; the target label does not mean every tensor uses that storage type.

Each retained session distinguishes two same-length prompts, generates two tokens
and appends another token, ending at position seven in the complete test. F32
execution is compared with the original fixture; quantized loader execution is
compared with loading the emitted artifact directly. This proves loader parity,
not quantization quality on real tasks or a performance budget.

Single-file conversion uses HTTP 206 byte ranges. Sharded conversion also covers
a server returning a bounded full HTTP 200 body despite the Range header. Exact
header/tensor ranges and request counts prove that an unchanged-commit cached
reload fetches only repository metadata, without reconverting or downloading
tensors again. File and progress URLs use the resolved 40-character commit.
Cache directories retain the requested revision shown in the table above.

Conversion writes `model.gguf`, a SHA sidecar, a quantization manifest and a
versioned `model.gguf.receipt.json` completion record to disk.
It streams source tensors without storing a complete source shard. Progress
reports **GGUF output bytes**, including header/alignment, rather than downloaded
source bytes. Eleven tensor notifications plus final output completion are
checked. Reloads emit no new conversion progress. Chat template and generation
defaults survive both conversion and cached loading.

## Interrupted and failed conversions

The [recovery examples](../../cera/src/engine/loading_prototype/tests/remote/conversion/recovery.rs)
interrupt the existing low-level converter after five of eleven tensors using
its cancellation option. No completed model, manifest, SHA sidecar or completion
receipt exists yet. The test appends a 128 KiB uncheckpointed tail, larger than the
complete fixture output, then retries through `ModelLoader`.

Run the recovery examples alone:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::conversion::recovery \
  --locked --offline -- --test-threads=1
```

| Saved checkpoint state | Retry behavior |
| --- | --- |
| Matching options, layout, boundary and verified prefix; extra tail | Truncate the tail and convert only the six remaining tensors |
| Changed payload bytes without a size change, damaged header, or short prefix | Restart all eleven tensors and restore original weights |
| Missing, malformed or wrong prefix digest | Restart all eleven tensors |
| Missing or mismatched header/layout digest | Restart all eleven tensors |
| Wrong total/completed count or byte boundary | Restart all eleven tensors |
| Legacy checkpoint without the request descriptor, integrity fields or resolved source commit | Restart all eleven tensors |
| Changed tensor overrides | Restart with the newly requested output types |

The fifteen-case matrix checks exact restored weights, output length and HTTP
ranges, plus execution after parent release. The valid retry emits seven progress
events (six tensors and completion); a restart emits twelve. The wrong-boundary
case sets the saved byte count to zero and supplies a matching empty-prefix digest,
so hashing alone cannot pass that case: the tensor boundary must also match.

A separate example, `repeated_interruption_preserves_the_entire_verified_prefix`,
performs two interrupted conversions followed by a load:

1. Stop after five tensors and independently hash the temporary file's actual bytes.
2. Append trailing junk, resume five more tensors, then stop at ten. Verify the
   first prefix remains intact and the saved digest covers all ten tensors and the header.
3. Append trailing junk again, then load through `ModelLoader`. Only one tensor
   remains: two progress events and final output matching the independent F32 fixture.

Exact HTTP ranges prove each tensor is fetched once across those three attempts;
only the two shard-header ranges repeat. The final session continues through
position seven after its parent model is released. The loading prototype itself
does not yet expose cancellation; setup uses the existing public lower-level API.

Checkpoint creation hashes accepted output bytes incrementally, including header
and alignment padding. After flushing, saving a checkpoint clones the running
SHA256 state; it does not rescan the growing file every five tensors. A retry
regenerates the GGUF header/layout into a hashing sink and checks the exact byte
boundary implied by the saved tensor count. It reads the saved prefix once with
a 64 KiB buffer, then truncates and writes through that same open file handle.
The resumed digest includes previously verified bytes. Legacy integrity fields
are optional when parsing but required for reuse, so old checkpoints restart.

A zero-length SafeTensors header and a decoded element-count mismatch preserve
the legacy error and publish no final artifact. A tensor failure can leave a
temporary file for recovery; the example documents that existing behavior.

## Detect and repair a damaged converted cache

Run just the new integrity examples:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::conversion::integrity \
  --locked --offline -- --test-threads=1
```

The [cache repair fixture](../../cera/src/engine/loading_prototype/tests/remote/conversion/integrity.rs)
converts F32 once and keeps the original loaded model alive. It then replaces one
late FFN weight with a different value without changing the GGUF's size. The next
`ModelLoader` load verifies the completion receipt, detects the hash mismatch,
converts all eleven tensors again and restores the original bytes. Repaired and
retained original models execute against an independent fixture through position
seven after parent release. File replacements use rename so the retained model's
mapping is preserved.

| Change between loads | Next load |
| --- | --- |
| Same-size weight corruption or truncated GGUF | Reconvert and verify exact restored weights |
| Parseable manifest with changed temperature | Reconvert and restore original defaults |
| Missing or malformed completion receipt | Reconvert once; legacy caches follow this rule |
| Missing, malformed or wrong SHA sidecar; valid receipt and bytes | Repair sidecar without tensor downloads |
| Same ordered tensor overrides | Reuse verified output without conversion progress |
| Changed tensor override from F32 to F16 or back | Reconvert; verify actual tensor type and numerical values |
| Cancelled repair | Keep the completion receipt absent; next load repairs again |

The same file also demonstrates the existing public low-level override option:

```rust
let opts = QuantizeOptions {
    target_quant: TargetQuant::F32,
    tensor_overrides: vec![("blk.0.ffn_down.weight".into(), TargetQuant::F16)],
    cache_dir, // isolated writable cache directory
    auth_token: None,
    ..Default::default()
};
let manifest = stream_quantize_hf_repo(&HfSpec::parse("fixture/overrides:F32")?, opts)?;
```

Imports are `cera::bundle::HfSpec` and `cera::convert::{QuantizeOptions,
TargetQuant, stream_quantize_hf_repo}`. The fixture repository is supplied by the
linked test, so this excerpt needs that server or a real SafeTensors repository.
`ModelLoader` remains a private prototype and does not expose tensor overrides.

Each completed-cache hit reads and hashes the entire GGUF with bounded streaming
memory, and hashes the exact manifest bytes. The completion record also matches
repository, requested and resolved revisions, quantization, strategy and ordered
overrides. It
is published last by renaming a unique temporary record. A failed repair removes
the prior record before fetching inputs, so it cannot bless old output. Missing
sidecar repair is best-effort after the actual model hash verifies.

This adds a model-size read on each converted load, including cached loads; no
session append/generate operation gains hashing. Large-model load-time and device
budgets remain unmeasured. HTTP range counts prove verified reloads fetch no
source tensors and unchanged-request recovery still downloads only six remaining
tensors from a five-of-eleven checkpoint.

## Pin inputs while a repository changes

Hugging Face documents [full commit revisions for downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download#from-specific-version)
and [resolving a revision once for multiple files](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#pin-a-revision-advanced).
The converter now resolves the requested branch/tag/commit through model metadata
before checking the completed cache or saved checkpoint. Its private snapshot
parser requires a full 40-character hexadecimal commit SHA. Every config,
tokenizer, template, generation-default, shard-header and tensor request then uses
that same commit in `/resolve/<commit>/...`. A changed commit restarts a partial
conversion or refreshes completed output; an unchanged commit retains verified reuse.

Run the four [upstream revision examples](../../cera/src/engine/loading_prototype/tests/remote/conversion/upstream.rs):

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::conversion::upstream \
  --locked --offline -- --test-threads=1
```

The fixture has commit A (`1111111111111111111111111111111111111111`) and commit B
(`2222222222222222222222222222222222222222`). B halves the F32 weights while
preserving the exact SafeTensors header, shapes and byte lengths. These IDs refer
only to the test server; they are not published model revisions.

| Example | Checked result |
| --- | --- |
| Metadata resolves A, mutable `main` file URLs serve B | All inputs use A; original weights and retained execution match the independent fixture |
| Cancel after five A tensors, retry resolves B | Restart all eleven B tensors despite the identical header/layout |
| Completed B, reload B, then reload A | Reuse B without progress, then reconvert A; the retained B model still executes correctly |
| Missing, short or non-hex SHA; HTTP404 resolution failure | Return an error, leave old bytes intact and emit no new conversion progress; no stale remote fallback |
| Named release and explicit full commit, with explicit authentication | Use the revision endpoint and pinned file URLs; both metadata and files accept the supplied token |
| Explicit full commit disagrees with metadata | Reject before creating a conversion cache directory |
| Legacy receipt/checkpoint lacks resolved commit | Rebuild once; subsequent same-commit completed loads reuse |

For example, the complete moving-branch test uses the existing public converter
and its cancellation option, then retries with that same requested spec:

```rust
let spec = HfSpec::parse("fixture/move:F32")?;
// The linked test supplies the temporary cache, callback and cancellation flag.
let interrupted = stream_quantize_hf_repo(&spec, first_options);
assert!(matches!(interrupted, Err(CeraError::Cancelled)));
let manifest = stream_quantize_hf_repo(&spec, retry_options)?;
let engine = CeraEngine::from_path(&manifest.files.model, cpu_config())?;
let mut session = engine.new_session(SessionConfig::default())?;
drop(engine);
session.append_tokens(&[0, 1])?;
```

This is a fixture excerpt, not a standalone public server. Imports are
`cera::bundle::HfSpec`, `cera::convert::stream_quantize_hf_repo`,
`cera::{CeraEngine, CeraError, SessionConfig}`. The linked executable test defines
all remaining values and also checks independent logits through position seven.
`ModelLoader` remains private; the lower-level converter's public signature and
`HfModelInfo` fields have not changed.

The non-main metadata URL uses `/api/models/<owner>/<repo>/revision/<revision>`,
matching the [official Hub client](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py).
The internal metadata fetch honors the converter's explicit authentication option.
Credentials are not stored in conversion records. The version-two request stores
both requested revision and resolved commit; version-one records rebuild once.

**Remote conversion now requires a successful metadata lookup on every call**,
including cached loads and explicit commit requests. The existing loader does
one discovery request and the converter does another; an unchanged cached load
therefore makes two metadata requests and no tensor requests. Direct low-level
conversion makes one metadata request. There is no offline or stale-metadata
fallback in this path; an already converted local GGUF can be loaded by local
path without resolving a remote revision. Consolidating duplicate discovery and
measuring network/load latency remain performance work. Append/generate and live
session KV do not gain network requests or hashing.

## Remaining limits

Commit pinning relies on the configured endpoint honoring immutable URLs. It is
not cryptographic source attestation. Local receipts and checkpoints detect
accidental corruption, not coordinated modification of records and artifacts.
Concurrent conversion into the same cache directory, power-loss durability,
other input dtypes/architectures, live CDN behavior, non-conversion GGUF source
pinning and device execution remain separate gates. Replacing live mapped files
is tested on macOS; broader platform sharing semantics remain open. The default
Q4_K_M option has prior policy tests but is not an executed output format in this
fixture. Large-model hashing and network costs remain unmeasured.

Plans17–19 change internal remote conversion/cache behavior and tests; public
converter signatures remain unchanged. Generated native and WASM loading probes
exclude `remote`, and their consumers are unchanged; Plan16's generated report
is historical evidence for its recorded snapshot. Public API promotion, live-KV
performance budgets and the actual Leap Swift/Kotlin runtime and packages remain open.

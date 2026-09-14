# Run revision-aware HF GGUF loading

Plan20 extends native HF discovery to pin direct GGUF files and generation
defaults to one resolved commit per repository. The loader prototype remains private; the same
behavior applies to existing `CeraEngine::from_hf` and `from_hf_url` constructors
in this worktree. All eleven scoped checks and three max-effort review rounds are complete after
fixing external-draft discovery and prefixed mirrors. See the
[handoff](API_RESHAPE_HANDOFF.md).

Run the five [complete executable examples](../../cera/src/engine/loading_prototype/tests/remote/snapshots.rs)
from the implementation worktree:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::snapshots \
  --locked --offline -- --test-threads=1
```

The tests supply valid one-block Llama GGUF weights, a temporary BundleRepo and
an isolated HTTP server. They need cached dependencies and permission to listen
on `127.0.0.1`; no external model or account is used. Cargo's `--offline` controls
dependency fetching. The fixture separately prevents external HTTP traffic.
To exercise the minimal feature combination, use
`--no-default-features --features remote,mmap` instead of `--features remote`.

## Load another revision while a session stays live

The first example runs this sequence through typed, dynamic and legacy loaders.
The excerpt uses configuration and server state supplied by the linked test;
`fixture/snapshot` exists only inside that server.

```rust
let first = ModelLoader::new(ModelSource::hugging_face(
    "fixture/snapshot:F32", None, None,
))
.config(cfg.clone())
.build_generative()?;
let mut retained = first.create_session(SessionConfig::default())?;
retained.append_tokens(&[0, 1])?;
drop(first);

// The fixture moves main from commit A to B before this load.
let second = CeraEngine::from_hf("fixture/snapshot:F32", None, cfg.clone())?;
retained.append_tokens(&[1, 0, 1])?;
```

A and B have identical GGUF layouts and byte lengths, but B halves all F32
weights and uses temperature 0.75 instead of A's 0.25. Mutable `main` file URLs
serve B even when metadata resolves A. A preseeded `main` cache entry also holds
B. The loader must fetch A's commit URLs and produce exact A file bytes and
inference logits; defaults must come from A too. After loading B, the live A
session continues to position five and matches an independently loaded A control.
This is CPU raw-token continuation evidence, not a warm-chat performance budget.

The metadata sequence is A, B, B, A. Each commit gets its own URL-derived cache
path under the configured store:

```text
<host>/fixture/snapshot/resolve/<40-character-commit>/nested/model-F32.gguf
```

The third load uses the dynamic handle, and the fourth uses the legacy URL
constructor. Each validates the expected bytes/defaults and CPU inference.
There are four metadata requests, one GGUF GET per commit and two defaults GETs
per commit. Unchanged-commit reloads emit no download progress; existing cache
integrity checks can still issue HEAD requests. A and B remain on disk together,
and the old branch entry remains unchanged. This change adds no cache cleanup.

## Reject unresolved and mismatched revisions

The second example loads A, then returns four invalid metadata responses: absent
SHA, short SHA, forty non-hex characters and HTTP404. Each load fails before
file/default downloads or cache reuse, preserving the previous model bytes and
progress count. Restoring valid A metadata reuses the downloaded file. HF loading
still needs metadata when the caller supplies a full commit, so use a local
GGUF path for offline loading.

The third example loads an explicit nested file URL using an uppercase full
commit. Matching metadata produces lowercase commit URLs; the explicit subpath
wins over a conflicting quantization preference. A different requested full
commit whose metadata reports A fails before downloading its files. Metadata
and file routes require the fixture's synthetic authentication token. Standalone
`fetch_model_info` still accepts a metadata payload without SHA; commit validation
belongs to the end-to-end loading/conversion path.

## Resolve external drafts independently

The fourth example exercises the known DSpark fallback for a repository named
`fixture/LFM2.5-2.6B`. Its primary stays at one commit while the separate draft
repository resolves A, B, B, missing SHA, A. Mutable draft URLs and an old branch
cache entry serve B. Each successful load must use the resolved draft bytes and
path; observed generation proposals distinguish A's two-token draft from B's
three-token draft. An A session retained after parent release executes its A
drafter after all later loads. It is created before those loads and first runs
generation afterward; the first example above separately proves live primary
continuation across loading B.

The primary GGUF is downloaded once, each draft commit once, with no new progress
on unchanged-commit reuse. Five primary metadata calls and five draft metadata
calls include the failed attempt. Missing draft SHA fails loading and preserves
the previous cache/progress; there is no mutable-URL fallback. This tests separate
revision consistency, not compatibility between arbitrary primary/draft versions.

## Keep a mirror's endpoint prefix

The fifth example loads repo-ID sources through an `HF_ENDPOINT` ending in
`/hub/mirror`. Both co-located and external drafts retain that prefix in metadata,
file and cache paths. It verifies exact draft bytes and observed proposals after
parent release. Three metadata requests and four GGUF GETs stay under the prefix;
no mutable draft URL is requested. The fix normalizes internally generated draft
URLs against the configured base before recovering their repository information.
Public full-URL parsing is unchanged; this example covers repo-ID loading.

## Companion files and scope

Run the existing [remote companion examples](API_RESHAPE_REMOTE_EXAMPLES.md):

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::companions \
  --locked --offline -- --test-threads=1
```

Their HF fixtures now publish full commit metadata and serve weights only at
commit URLs. Existing selection, cache repair and CPU execution assertions cover
vision projectors, audio encoders/decoders/tokenizers and draft models. Defaults
remain optional and best effort; a defaults fetch failure retains the existing
fallback behavior. Pure `resolve_hf_manifest`, explicit manifest URLs, fixed bundle
catalog URLs and browser loading remain caller-managed and are not pinned by
this change.

[SafeTensors conversion](API_RESHAPE_CONVERSION_EXAMPLES.md) keeps its own second
metadata resolution and requested-revision output directory. A direct GGUF load
makes one metadata request for the primary repository and one more if it selects
an external draft, including on cache hits. Loading via conversion makes two
queries to its source repository. Missing or
invalid SHA now fails discovery before format/quant selection. Combining the two
conversion queries and measuring real network/large-model costs remain open.
No session/KV execution code, generated bindings or Leap runtime changed.

The endpoint must honor commit URLs. This follows the public HF
[revision download contract](https://huggingface.co/docs/huggingface_hub/en/guides/download#from-specific-version)
and [revision pinning guidance](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#pin-a-revision-advanced).
Local hashing and immutable addressing do not add cryptographic source attestation
or concurrent-writer coordination. Live CDN/device tests and the remaining
[API phases](API_RESHAPE_HANDOFF.md) stay open.

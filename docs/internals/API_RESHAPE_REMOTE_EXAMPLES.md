# Run remote companion loading examples

These executable examples exercise the private loading API against a loopback
HTTP server. They generate complete synthetic CPU GGUFs, discover and download
companions, verify cached bytes, release the model and execute the retained
session. They need no Hugging Face account, model download or external network.
The test process needs permission to bind `127.0.0.1`.

Run from the implementation worktree root:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --features remote --lib \
  engine::loading_prototype::tests::remote::companions --locked --offline \
  -- --test-threads=1
```

The four tests are complete runnable examples. Each starts a server in the
parent process, then executes only its own test in a child configured with
`HF_ENDPOINT`, an isolated cache and a proxy that rejects external tunnels.
`--offline` applies to Cargo dependency resolution; subprocess isolation is what
keeps the HTTP fixture traffic local. Dependencies must already be cached.

| Example | Concrete inputs and checks |
| --- | --- |
| [Vision](../../cera/src/engine/loading_prototype/tests/remote/companions/vision.rs) | `fixture/vision:Q4_K_M`, `Q8_0` and `F16`; same-quant projector selection and Q8 fallback. Wrong equal-length cached weights without a SHA sidecar are replaced. Real projection outputs distinguish selected weights. Default features also execute PNG ingestion after model release. |
| [Audio](../../cera/src/engine/loading_prototype/tests/remote/companions/audio.rs) | `fixture/audio-full:Q4_K_M` and `fixture/audio-split:Q4_K_M`; 1,600 PCM samples at 16 kHz consume two positions. Six text tokens produce 22,560 PCM samples at 24 kHz, ending at position 22. Independent encoder/vocoder controls verify the selected decoder and detokenizer. |
| [Draft](../../cera/src/engine/loading_prototype/tests/remote/companions/draft.rs) | HF, local JSON with remote assets and a preseeded bundle catalog. Downloaded draft proposes `[0, 0]`; a local configuration override proposes `[1, 1, 1]`. Observed generation calls prove that the chosen drafter runs after model release; output and continuation match a primary-only control. |
| [Failures](../../cera/src/engine/loading_prototype/tests/remote/companions/vision.rs) | Downloaded invalid/header-only projectors leave text inference usable. HTTP 404 and SHA mismatch stop loading and leave no completed, partial or SHA cache file. |

The [shared loader helper](../../cera/src/engine/loading_prototype/tests/remote/companions.rs)
runs typed, dynamic-handle and legacy HF constructors against the same fixture.
The private API sequence is:

```rust
let model = ModelLoader::new(ModelSource::hugging_face(
    "fixture/audio-full:Q4_K_M", None, None,
))
.config(load_config) // CPU, context 256, BundleRepo with an isolated store/progress
.build_generative()?;
let mut session = model.create_session(SessionConfig::default())?;
drop(model);
session.append_audio(&pcm, 16_000)?;
session.append_tokens(&[0, 1])?;
session.generate(&options, &mut sink)?;
```

This excerpt runs inside the linked fixture setup; the `fixture/*` repositories
exist only in its server. These types remain private to the loading experiment.
For a complete standalone Rust program and Swift/Kotlin text examples, see the
[loading walkthroughs](API_RESHAPE_EXAMPLES.md) and
[standalone audio example](API_RESHAPE_AUDIO_EXAMPLE.md).

To exercise remote loading without the default preprocessing features:

```bash
cargo test -p cera --no-default-features --features remote,mmap --lib \
  engine::loading_prototype::tests::remote --locked --offline \
  -- --test-threads=1
```

This runs the five original remote contracts, four companion examples and three
[conversion contracts](API_RESHAPE_CONVERSION_EXAMPLES.md).
The minimal vision case executes projection and embedding input; PNG ingestion
requires `vl-preprocess` and runs in the default-feature command above.

## What these examples establish

Selected remote files are downloaded once per isolated cache, including audio
manifest extras. An unselected decoder/draft is never downloaded. Actual weights,
SHA sidecars and progress completion are checked for selected companions. A
full dedicated audio decoder supplies its own detokenizer; when it has no
detokenizer, the selected tokenizer supplies one. A separately listed vocoder
extra is downloaded but does not replace the dedicated decoder.

Remote asset download failures are fatal even for optional companions. The
nonfatal policy for corrupt or incomplete optional GGUFs applies after a
successful download, or when loading local files. Capability flags alone do not
prove that usable weights are attached.

Cache repair here begins without a SHA sidecar. The current cache trusts a
matching sidecar without rehashing the file; these tests do not establish
detection of tampering accompanied by a matching sidecar. Bundle catalog JSON
is preseeded because its public URL is fixed; only its asset downloads execute
over loopback. Live public catalog/CDN behavior remains unverified. Actual
conversion has separate [executable examples](API_RESHAPE_CONVERSION_EXAMPLES.md).

Quant suffixes are selection labels on F32 fixtures, not quantization-quality
coverage. Synthetic PCM is finite but not trained speech. These are CPU loading
and execution checks; device sharing, performance budgets, public API promotion
and the Leap Swift/Kotlin runtime remain separate gates. The Plan 13 generated
foreign-consumer report is historical evidence for that source snapshot; this
test/documentation increment does not rerun generated bindings.

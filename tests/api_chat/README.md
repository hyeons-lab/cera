# Chat contract prototype

This executable P0.1 prototype lives under `cera/tests/api_chat`, outside the
published API, so its Rust modules also travel with packaged crate tests. It defines owned
messages with ordered content, batch/replacement operations, six phases, typed
validation/recovery outcomes and a text collector over one decode call. The
application retains its transcript. `Chat` retains an execution object, shared
tokenizer/profile, phase and one optional terminal-residency bit.

Run the offline fixtures from the worktree:

```sh
cargo test -p cera --test chat_contract --locked --offline
python3 tests/api_chat/test_runner.py
```

Run **all** fixtures, including the complete public tokenizer, using the exact
artifact pinned in [profile.json](profile.json):

```sh
curl --fail --location \
  'https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/8fdc9d526b7ed346b19257551b05816c7912ecc2/LFM2-350M-Q4_0.gguf' \
  --output /tmp/LFM2-350M-Q4_0.gguf
python3 tests/api_chat/run.py \
  --model /tmp/LFM2-350M-Q4_0.gguf \
  --output /tmp/cera-chat-contract \
  --target-dir /tmp/cera-chat-target \
  --target aarch64-apple-darwin
```

Choose the Rust host triple for `--target`; add `--no-default-features` for the
portable core configuration. The runner checks SHA-256, runs the normally ignored
public-artifact fixture explicitly, and invokes Cargo's exact emitted test binary
directly. It requires positive harness output, records core/build/harness source
and executable hashes, and rejects input/artifact drift during validation. An
inherited Cargo runner cannot substitute another program. The harness downloads
nothing itself. A missing or incorrect model fails rather than becoming a
successful skip. Consult the
[public model license](https://huggingface.co/LiquidAI/LFM2-350M-GGUF/blob/8fdc9d526b7ed346b19257551b05816c7912ecc2/LICENSE)
before redistributing its artifacts; the model is not vendored here.

The [tests](../../cera/tests/api_chat/tests.rs) execute this flow using the prototype:

```rust,ignore
let messages = vec![
    Message::text(Role::System, "Be concise."),
    Message::text(Role::User, "One?"),
    Message::text(Role::Assistant, "One."),
    Message::text(Role::User, "Two?"),
];
chat.ingest_messages(&messages)?; // One append and one final assistant prefix.
assert_eq!(chat.phase(), SessionPhase::PromptReady);
let answer = chat.complete(&GenerateOpts::default())?;
// The caller records answer.text in its own transcript.
// After a proven terminal boundary, the next user batch can be ingested.
```

This snippet describes the compiled test contract; no production `Chat` constructor
is published. The only backend bridge is the private unit-test adapter described
under [Actual Session transactions](#actual-session-transactions) below.
Streaming callers use `generate_into` with the existing `ModalitySink`.
`complete` collects token IDs and decodes the whole sequence with the existing
tokenizer, returning its original `GenerateSummary`. Use streaming to retain
partial output after a decode error. This profile rejects audio-output execution
at construction and before preparation/decode and rejects input
images/audio/tools before append or reset.

The structural ten-turn fixture uses nonempty scripted answers and the real public
BPE vocabulary/merges, including Unicode and whitespace in user messages. It
compares every prepared token prefix with full-history rendering, with no reset
in the trace executor. The legacy regression runs **production Session decode**
through a scripted `Model` in greedy and stochastic modes: EOS is sampled but not
resident, and the next isolated user append repeats BOS. Its corrected suffix
matches the full-render token fixture. These are tokenizer/control-flow proofs;
they do not execute the model's numerical weights or establish R1 KV performance.

See the [chat inventory and remaining gates](../../docs/internals/API_RESHAPE_CHAT.md)
for native/browser adapters, real decode observations and performance work.

## Actual Session transactions

The core unit-test adapter runs the same contract against actual Session append,
checked recovery/reset and observed decode. It remains private until the numerical,
backend and binding gates pass. Without the pinned model, the offline cases run with:

```sh
cargo test -p cera --lib --locked --offline -- session::chat::
```

Run the entire suite, including both required public-tokenizer cases, with:

```sh
python3 tests/api_chat/run.py \
  --core-transactions \
  --model /path/to/LFM2-350M-Q4_0.gguf \
  --output /path/to/persistent-evidence/core-chat \
  --target-dir /path/to/persistent-build-cache \
  --target aarch64-apple-darwin
```

Add `--no-default-features` for the minimal native build. Either runner invocation
accepts `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0` to keep build caches
small without changing optimization or assertions. The runner hashes the pinned
model and complete source inputs, obtains the exact Cargo-emitted unit-test binary,
executes its `session::chat::` suite directly (logged as `core-transactions`), and
requires both public fixtures, no ignored tests and the exact pinned case count
(`CORE_CASES`; the isolated suite pins `ISOLATED_CASES`). Adding or removing a
fixture must update the pin. Four Python controls exercise false-success and
source-provenance rejection: `python3 tests/api_chat/test_runner.py`.

The [actual-state tests](../../cera/src/session/chat/tests.rs) inspect physical
prefill inputs and resident attention rows through scripted logits; what they
cover, and what they deliberately do not prove, is listed in the
[Plan42 section of the chat inventory](../../docs/internals/API_RESHAPE_CHAT.md#actual-session-transaction-bridge-plan42).

Keep reports and pinned models in a persistent ignored project directory when
continuing across sessions. Temporary-directory cleanup can remove otherwise valid
historical evidence. Rebuild and rehash missing artifacts before making fresh
runtime claims; source and handoff records do not recreate a lost binary report.

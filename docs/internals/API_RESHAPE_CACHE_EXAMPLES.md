# Run persistent cache identity examples

Status: Plan16 is complete within its bounded cache-identity scope. Four CPU
contracts, two GPU source/ownership checks, direct Metal execution and all four
regression controls pass. Feature/generated/example checks and two max-effort
review rounds are complete; the final round left only corrected comment wording. See the
[handoff](API_RESHAPE_HANDOFF.md#completed-plan-16-named-cache-identities).

The [complete executable examples](../../cera/src/engine/loading_prototype/tests/ownership/named.rs)
create two-block LFM2 weights locally. They need no account, model download or HTTP
server. Run from the implementation worktree:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --lib \
  engine::loading_prototype::tests::ownership::named --locked --offline
```

For the same tests with fewer features, add
`--no-default-features --features disk-cache,mmap`. The loader in these examples
is still the private prototype. The tests provide complete weights and configuration
for this sequence:

```rust
let model = ModelLoader::new(ModelSource::path(&model_path))
    .config(cpu_config)
    .build_generative()?;
model.configure_cache(KvCacheConfig {
    cache_dir: Some(cache_dir),
    max_warm_entries: 0, // This example proves disk reuse without warm entries.
    ..KvCacheConfig::default()
});
let mut session = model.create_session(session_config)?;
session.append_tokens(&[0, 1])?;
drop(model);
session.append_tokens(&[0])?;
```

`CeraEngine::configure_cache` exposes the same existing production behavior; the
new loader name is not yet publicly exported. CPU LFM2 implements this two-tier
prefix cache. Classic CPU Llama retains live session KV but its prefix-cache methods
are currently no-ops; this example does not claim otherwise.

## Replace weights and retain the original session

The first test changes only a late feed-forward tensor, preserving the header,
embeddings, file length and every other tensor. It atomically renames a new file
over `model.gguf` while the first model and session remain alive. This preserves
the original mapping; it does not truncate or overwrite a live mapped file.

A newly loaded model at the same path must compute all three prompt tokens,
matching an independent model with prefix caching disabled. It must not restore
the two-token disk prefix written by the original weights. Both old and new
sessions continue with their own numerical results. Clearing either model's cache
leaves the other identity's files alone, and clearing does not reset live KV.
The test runs with both uncompressed and F16 KV.

The second test reloads unchanged weights and computes only the final token of a
three-token prompt, restoring the first two from disk. Test-only counters observe
actual CPU prefill work rather than inferring reuse from identical output. A
separate counter checks that default warm use hashes no bytes, first cold
configuration hashes the model once, and reconfiguration, append and generation
do not scan the model again.

The third test writes a real snapshot under the previous path-only namespace.
New loads ignore that file, compute the full prompt correctly and preserve the
old file when their own cache is cleared. Existing old files are not migrated.

The fourth test pauses the first content hash while another thread creates an
F16 session. Session creation must finish before hashing resumes. The cold cache
then uses that session's current compression tag, and a reload restores the
two-token prefix. This checks that scanning the weights holds neither the cache
lock nor its compression-tag lock.

## GPU source ownership

Run the source and mapping checks alongside the CPU contracts:

```bash
cargo test -p cera --features gpu,metal --lib \
  engine::loading_prototype::tests::ownership::named --locked --offline
```

On macOS this runs six tests without needing a GPU device. DSpark's identity must
change when either its draft or its base GGUF changes. The Metal mapping check
keeps the exact parsed file mapping alive across path replacement and parent
release. Metal now uses that mapping for its no-copy buffer instead of reopening
the path after parsing.

On a Mac with an available Metal device, also run the ignored execution test:

```bash
cargo test -p cera --features metal --lib \
  engine::loading_prototype::tests::ownership::named::metal_execution_uses_parsed_weights_after_path_replacement \
  --locked --offline -- --exact --ignored
```

It parses weights A, replaces the path with weights B, then constructs Metal from
the retained parsed source. Its output must match an independent Metal A control
and differ from B. The local run passes; a sandbox that hides Metal devices cannot
run it. This proves the tested backing ownership, while GPU conversation isolation
and a wider device matrix remain open.

## Identity and cost

Named built-in prefix caches retain the caller's namespace and append a versioned
SHA256 digest of every loaded GGUF backing buffer. Source count and byte lengths
frame the ordered inputs; DSpark includes both draft and base. Backend and KV-format tags
remain separate. The existing disk format reduces the namespace/layout to its
64-bit fingerprint; this is a correctness boundary, not a new adversarial integrity
or authentication guarantee. Anonymous models remain warm-only even if a disk
directory is configured. Low-level `KvPrefixCache::new` remains caller-managed.

The existing public `GpuWeightSource` trait has an additive provided
`cache_identity_sources()` method. Built-in LFM2, Llama and DSpark opt in. A custom
source defaults to its existing caller-managed namespace; it can opt in only when
the returned GGUF files fully determine its weights, in stable order. The caller
must still distinguish configuration overrides that change computation.

CPU LFM2 already owns its GGUF and hashes it lazily on first disk configuration,
before acquiring cache/tag locks. It reads the current compression tag after hashing.
wgpu/Metal upload from temporary source models: they hash named sources once
before releasing those sources, avoiding an additional retained CPU weight buffer.
This adds a load-time scan when `disk-cache` is compiled in, even if the GPU caller
never enables disk caching. Builds without that feature do no identity hashing.
There is no hashing in the append/generate path.

Run the isolated hashing-cost measurement in release mode:

```bash
cargo test -p cera --release --no-default-features --features disk-cache --lib \
  model::cache_identity::tests::measure_loaded_byte_identity_cost --locked --offline \
  -- --exact --ignored --nocapture
```

It reports five scans of a 64 MiB resident synthetic tensor plus its GGUF header.
This measures the identity helper, not cold filesystem I/O, GPU upload, model
startup, inference throughput or a device performance budget. Supported macOS,
iOS, Linux and Android aarch64 builds enable SHA256's ARM acceleration; other
targets retain the crate's default backend. Builds without `disk-cache` or `remote`
do not enable SHA256. The local release helper scanned 67,108,992 bytes five
times: median 37,285 microseconds (37.3 ms), minimum 36,501, maximum 44,967.
This was measured on an aarch64 macOS host with Rust 1.99 nightly while a separate
generated-consumer build was active. It is a measured one-time resident-byte scan,
not an isolated load benchmark or an inference latency guarantee.

Caller-mutated parsed metadata/configuration and in-place mutation of live mapped
files require separate contracts. Converted-file/checkpoint identity, all-device
ownership, full performance budgets and public API promotion remain open. The
Leap Swift/Kotlin runtime is a separate unfinished workstream.

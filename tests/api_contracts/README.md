# Retained API declaration checks

Run from the worktree, without compiling or downloading a model:

```bash
python3 tests/api_contracts/check.py
python3 tests/api_contracts/test_check.py
uvx --offline ruff check tests/api_contracts
uvx --offline ruff format --check tests/api_contracts
```

The [reviewed baseline](retained_api.json) lists 69 declaration groups in five
Rust source files: core engine/repository, native UniFFI and CPU/WebGPU/browser
WASM. It records 300 methods/functions/constants and 34 records/enums and three callback traits. The
[target retention map](../../docs/internals/API_RESHAPE_TARGET_RETENTION.md)
explains their supported homes and the selected additive loading signatures.
This inventory supplements the [generated loading runtime](../api_loading/README.md).

For example, removing `CeraEngine::from_parts_async` reports a missing native
method; dropping its `async` or changing its payload reports a changed signature.
An extra public method on an inventoried owner is reported for classification.
The mutation controls exercise these cases on temporary source copies, plus the
nested WebGPU multipart factory and native u64 context record. They never edit
the actual source tree.

Wrapping the native `clear_prefix_cache` method in a block comment also reports
it as missing. Nested comments and string/character literals are masked during
declaration scanning; literal payloads remain in compared record defaults.
Controls cover fake declarations inside comments/raw strings and unclosed input.
Whitespace inside literals is compared exactly; a changed string default or error
message remains visible even when it differs only in spaces or newlines.

The checker reads formatted inherent `impl Type { ... }` blocks, module-level
public functions, public constants and selected braced records/enums/traits. It compares
normalized declarations, not bodies. It handles the current nested WebGPU module
and multiple inherent impls; a missing known declaration or duplicate row fails.
It intentionally does not parse arbitrary Rust, trait implementations, newly
introduced owners, inline modules, macro-generated methods, generated language
exports, or cfg/export attributes outside the captured declarations. Method JS
names, feature availability, callable foreign payloads and ABI compatibility need
surface diffs, compilers and runtime consumers in their affected target gates.

There is no automatic baseline-update mode. When an intentional API change is
authorized, review the old/new declarations and retention map together, then
edit the expected entries. Keep legacy declarations during additive promotion;
do not erase them from the baseline merely to make a missing-method report pass.
No production signatures change in this increment. The inventory does not claim
that every backend supports every retained raw operation.

## Additive CPU WASM loading declarations

[wasm_loading.json](wasm_loading.json) freezes the eight new loading classes from
wasm-bindgen 0.2.117. Generate the actual module, then compare its declarations:

```sh
python3 tests/api_contracts/check_wasm_loading.py /tmp/cera-wasm-node/cera_wasm.d.ts
python3 tests/api_contracts/test_wasm_loading.py /tmp/cera-wasm-node/cera_wasm.d.ts
```

This checks method/constructor names, return types and complete optional payload
fields. Ownership, errors and inference still require the executable Node tests.
The existing 69-surface Rust baseline remains unchanged.

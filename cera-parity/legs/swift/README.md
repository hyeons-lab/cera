# cera-parity Swift-via-UniFFI leg

SPM package that builds an executable
(`.build/release/CeraParitySwift`) loading the generated UniFFI Swift
bindings + the cera-ffi cdylib. Read by the Rust harness
(`cera_parity::run_swift_uniffi`) over stdin/stdout JSON; sister to
`legs/kotlin/`.

## Layout

```
legs/swift/
├── Package.swift                 — two-target SPM manifest
└── Sources/
    ├── cera_ffiFFI/              — .systemLibrary target
    │   ├── module.modulemap      — exposes cera_ffiFFI.h as `cera_ffiFFI`
    │   └── cera_ffiFFI.h         — symlink → ../../../../../cera-ffi/bindings/swift/cera_ffiFFI.h
    └── CeraParitySwift/          — .executableTarget
        ├── main.swift            — runner (stdin JSON → bindings → stdout JSON)
        └── cera_ffi.swift        — symlink → ../../../../../cera-ffi/bindings/swift/cera_ffi.swift
```

The two binding files are git symlinks (mode 120000), so
`just bindings` regenerations propagate without a manual copy step.
The system-library target name is `cera_ffiFFI` exactly because the
generated `cera_ffi.swift` does `#if canImport(cera_ffiFFI) ; import
cera_ffiFFI` — any other name and the C FFI declarations fall out of
scope at compile time.

## Build

From the workspace root:

```bash
WS=$(pwd)
cargo build -p cera-ffi    # produces $WS/target/debug/libcera_ffi.dylib
swift build -c release \
  --package-path "$WS/cera-parity/legs/swift" \
  -Xlinker -L"$WS/target/debug"
```

`-Xlinker -L<dir>` is the SPM-portable way to add a library search
path. `LDFLAGS` is not honored consistently by `swift build`.
`--package-path` keeps the build out-of-process from the workspace
shell so no `cd` / subshell is required.

## Run (manual smoke test)

```bash
WS=$(pwd)
echo '{
  "bundle": "LFM2-350M-Extract-GGUF",
  "quant": "Q4_0",
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "seed": 0,
  "cache_dir": "'"$WS"'/target/tmp/cera-parity-cache"
}' | DYLD_LIBRARY_PATH="$WS/target/debug" \
  "$WS/cera-parity/legs/swift/.build/release/CeraParitySwift"
```

Emits a `RunOutput` JSON document on stdout. First run downloads the
~210 MB fixture into `cache_dir`; subsequent runs are cache hits.
Cache root mirrors the harness default so the manual smoke shares
state with `cargo test`.

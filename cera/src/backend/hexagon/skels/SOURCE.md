# Bundled Hexagon DSP skels

`libggml-htp-v{73,75,79,81}.so` are the DSP-side (Hexagon ELF) worker
libraries the NPU backend loads into the CDSP unsigned PD at runtime.
They are built from unmodified llama.cpp sources; the host side
(`sys.rs`, `queue.rs`, `params.rs`) speaks their `htp_iface` IDL, so the
skel and host versions must move together — that is why they are
vendored here instead of fetched at install time.

## License (read before redistributing)

These binaries are compiled from MIT-licensed sources:

- Copyright (c) 2023-2026 The ggml authors (llama.cpp)
- Full text: `LICENSE` in this directory (upstream `LICENSE`, verbatim)

The MIT license requires the copyright + permission notice to accompany
all copies, **including binaries**. Any artifact embedding these skels
(the `cera` crate, `libcera_ffi.so`, the Android AAR) must carry this
attribution: this directory travels with the crate, and the AAR ships
`assets/NOTICE` with the same text (see
`cera-ffi-kotlin/cera-ffi-android/src/main/assets/NOTICE`).

Toolchain note: the skels were built with the proprietary Qualcomm
Hexagon SDK (compiler + headers/inlines only; they link no proprietary
shared library — `NEEDED` is DSP-side `libc.so` + `libgcc.so`,
resolved on-device). Shipping toolchain *output* is the SDK's intended
use, but the team should confirm once against the installed SDK's EULA
text; if it ever restricts binary redistribution, fall back to building
skels at release time from the pinned source below rather than
vendoring them.

## Provenance

- Source: llama.cpp `upstream/master` at `cf302539c` (clean tree, no
  local patches), `ggml/src/ggml-hexagon/htp/` + `htp_iface.idl`.
- Build: llama.cpp `build_htp_skel(v73|v75|v79|v81)` CMake targets
  (`ggml/src/ggml-hexagon/CMakeLists.txt`), one `-DDSP_VERSION` per
  file. Requires `HEXAGON_SDK_ROOT` (proprietary Qualcomm toolchain;
  headers/inlines only — the skels link no proprietary shared
  library; `NEEDED` is DSP-side `libc.so` + `libgcc.so`, resolved
  on-device).
- SDK version used for these binaries: unknown (built before this
  vendoring; behavior validated on Snapdragon 8 Elite / v79).
- License of the compiled sources: MIT (llama.cpp). Toolchain output
  is not SDK redistribution.

## Integrity (md5)

- v73: 66c9716e5dc1e2ebd80a239c93d747a5
- v75: 893f2be5814f7e2e85742668cd482117
- v79: 8ad4796cced0a610f6c6a5a13db8d8c5
- v81: c49cc4767b4fc198f1d0b76297d2183b

## Rebuilding

```bash
# in a llama.cpp checkout with HEXAGON_SDK_ROOT set (Linux):
cmake -S . -B build-snapdragon -DGGML_HEXAGON=ON <android preset>
cmake --build build-snapdragon --target ggml-htp-v73 ggml-htp-v75 ggml-htp-v79 ggml-htp-v81
# outputs: build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-vXX.so
```

After replacing any skel: update the md5s above and re-run the full
on-device determinism matrix (logits m=1..8 x5, greedy md5 2x2x6,
256-token pair) — skel/host skew fails silently as wrong numerics.

## Coverage

v73 (8+ Gen 1 / 8 Gen 2 / 7+ Gen 2 / X Elite), v75 (8 Gen 3 / 8s Gen 3),
v79 (8 Elite), v81 (next-gen). No v68/v69 (888 / 8 Gen 1): outside
llama.cpp upstream scope; see the packaging doc for the support matrix.

# Bundled Hexagon DSP skels

`libggml-htp-v{73,75,79,81,85}.so` are the DSP-side (Hexagon ELF) worker
libraries the NPU backend loads into the CDSP unsigned PD at runtime.
They are built from llama.cpp sources plus one patch (`patches/0001-*.patch`, described
under Provenance); the host side
(`sys.rs`, `queue.rs`, `params.rs`) speaks their `htp_iface` IDL, so the
skel and host versions must move together: that is why they are
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
shared library; `NEEDED` is DSP-side `libc.so` + `libgcc.so`,
resolved on-device). Shipping toolchain *output* is the SDK's intended
use, but the team should confirm once against the installed SDK's EULA
text; if it ever restricts binary redistribution, fall back to building
skels at release time from the pinned source below rather than
vendoring them.

## Provenance

- Source: llama.cpp branch `feat/hexagon-v85` at commit `00ccd6970`
  (`ggml/src/ggml-hexagon/htp/` + `htp_iface.idl`), including extended
  Conv1D, ConvTranspose1D, Snake, and unary operations.
- Patch: `patches/0001-hvx-q8-activation-quantizer-f32-reciprocal.patch`
  (`hvx-mm-kernels-tiled.h`, the `q8_0` and `q8_1` tiled activation
  quantizers). The upstream quantizer computed the Q8 scale and its
  reciprocal in f16: for a block whose absolute maximum is below about
  2e-3 the reciprocal overflowed f16 and the int8 values were wrongly
  scaled (50% or more relative error on that block). Every HVX matmul
  of 1 to 7 rows (all decode steps, short prompts, a prompt's tail
  chunk) was affected. The patch stores the smallest f16 at least
  `amax / 127` (rounded up, like the CPU quantizers, so no element quantizes
  past 127 even for a subnormal scale) and multiplies by the f32 reciprocal of
  that stored scale. Measured on a
  Galaxy S25 Ultra (v79) against the CPU: full-logit cosine on 2 to 7 row
  chunks went from 0.95..0.99 to 0.9995 or better (LFM2.5-2.6B Q4_0 from
  -0.003..0.99 to 0.991..0.9999), single tokens from as low as 0.377 to
  0.9995, with unchanged decode throughput. v73, v75 and v81/v85 are the
  same C code rebuilt; only v79 has been run on hardware.
- Build: Hexagon SDK 6.6.0.0 with Hexagon Tools 19.0.07 (hexagon-clang with
  whole-program LTO), compiling `libggml-htp-v{73,75,79,81}.so` targets with
  `-DDSP_VERSION`, and copying `v81` to `v85`.
- License of the compiled sources: MIT (llama.cpp). Toolchain output
  is not SDK redistribution.

## Integrity (md5)

- v73: 57a35a38a7ba96cbb1532307260b0dcc
- v75: b2df0421b98d5fe3638535a3437bfc7a
- v79: fd6c087439f663b440c4729d68059cee
- v81: 54a097f1e742a9944a5cbabab220d340
- v85: 54a097f1e742a9944a5cbabab220d340

(Before the patch: v73 2dd73769..., v75 890dcd26..., v79 7da1d562...,
v81/v85 43349eab....)

## Rebuilding

```bash
# in a llama.cpp checkout with HEXAGON_SDK_ROOT set (Linux):
cmake -S . -B build-snapdragon -DGGML_HEXAGON=ON <android preset>
cmake --build build-snapdragon --target ggml-htp-v73 ggml-htp-v75 ggml-htp-v79 ggml-htp-v81 ggml-htp-v85
# outputs: build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-vXX.so
```

After replacing any skel: update the md5s above and re-run the full
on-device determinism matrix (logits m=1..8 x5, greedy md5 2x2x6,
256-token pair): skel/host skew fails silently as wrong numerics.

## Coverage

v73 (8+ Gen 1 / 8 Gen 2 / 7+ Gen 2 / X Elite), v75 (8 Gen 3 / 8s Gen 3),
v79 (8 Elite), v81 (next-gen). No v68/v69 (888 / 8 Gen 1): outside
llama.cpp upstream scope; see the packaging doc for the support matrix.

# Bundled Hexagon DSP skels

`libcera-htp-v{73,75,79,81}.so` are the DSP-side (Hexagon ELF) worker
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

- Source: llama.cpp at upstream commit `de7fa0a3c` (2026-10-08) plus three local Hexagon commits on
  top (`1824e49e4`, `db89d9e20`, `957e73059`: the extended Conv1D, ConvTranspose1D, Snake and unary
  operations, and the HVX float-clamp fix), then the quantizer patch below as `51c61fce4`.
  These commits live in a local, unpublished llama.cpp branch
  (`ggml/src/ggml-hexagon/htp/` + `htp_iface.idl`); the previous skels were built from
  `hyeons-lab/llama.cpp` commit `00ccd6970`.
- Host contract: the DSP now answers `NO_SUPPORT` to a `Cpy` or `Concat` whose
  kernel params the host did not precompute, and the matmul, flash-attention, `get_rows` and
  `ssm_conv` params changed layout. `params.rs` mirrors them. The Cpy, Concat, GetRows and SsmConv builders and the HMX chunk and
  2-D solvers are checked word for word against llama.cpp's own code
  (`scripts/hexagon-golden/gen.py` with `LLAMA_CPP` set to a checkout, `testdata/*_golden.txt`);
  the matmul and flash-attention words are assembled from those and pinned by the op-sequence
  hashes only.
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
  that stored scale. A block holding a NaN or infinity gets a NaN or
  infinite scale and all-zero values (the CPU contract), so the poison reaches
  the dot product. Measured on a
  Galaxy S25 Ultra (v79) against the CPU: full-logit cosine on 2 to 7 row
  chunks went from 0.95..0.99 to 0.9995 or better (LFM2.5-2.6B Q4_0 from
  -0.003..0.99 to 0.991..0.9999), single tokens from as low as 0.377 to
  0.9995, with unchanged decode throughput (measured on the previous build; the patch is ported
  onto the reworked quantizers). v73, v75 and v81 are the same C code rebuilt; only v79 has been
  run on hardware.
- Build: Hexagon SDK 6.6.0.0 with Hexagon Tools 19.0.07 (hexagon-clang with
  whole-program LTO), compiling `libggml-htp-v{73,75,79,81}.so` targets with
  `-DDSP_VERSION`.
- License of the compiled sources: MIT (llama.cpp). Toolchain output
  is not SDK redistribution.

## Integrity (md5)

- v73: 8e4244443225ff0bba5479a4d650c6e0
- v75: d0334de0b516b13d5a3fe205fb0e1ac0
- v79: fe6dbaddf669fda4e118e6000482fd90
- v81: 531d875fe26736ad0c6aa33b23ef6b09

These are the md5s of the shipped, renamed files. The unrenamed build outputs were
v73 bb29e473361edaaec976677dbf375813, v75 26c87c573b64443ab74e35b9770c497d,
v79 d41218ff47b146e6c7afd0870e7975fc, v81 04b17a165e8a01c146e6eceb193bbb2f.

## Rebuilding

```bash
# in the llama.cpp checkout described under Source (commit 51c61fce4), Docker toolchain image
# ghcr.io/snapdragon-toolchain/arm64-android:v0.7 (Hexagon SDK 6.6.0.0, Tools 19.0.07):
cp docs/backend/snapdragon/CMakeUserPresets.json .
docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace <image> bash -lc \
  'cmake --preset arm64-android-snapdragon-release -B build-snapdragon &&
   cmake --build build-snapdragon --target htp-v73 htp-v75 htp-v79 htp-v81'
# outputs: build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-vXX.so
```

## Renaming (llama.cpp coexistence)

The build outputs are named `libggml-htp-vXX.so`, the same names llama.cpp's own Hexagon backend
ships, so an app embedding both would have one overwrite the other in `ADSP_LIBRARY_PATH`, and
the DSP loader could treat the equal SONAMEs as one library. The shipped files are therefore
renamed `libcera-htp-vXX.so` (the host opens them by that name, `HexagonArch::skel_filename`)
and the SONAME in each is patched to match. `ggml` and `cera` are both four characters, so the
patch is a same-length byte substitution that moves no offsets:

```bash
for v in 73 75 79 81; do
  cp libggml-htp-v$v.so libcera-htp-v$v.so
  perl -0pi -e "s/libggml-htp-v$v\.so/libcera-htp-v$v.so/g" libcera-htp-v$v.so
done
# check: llvm-readelf -d libcera-htp-vXX.so | grep SONAME
```

Without the quantizer patch (the commit before `51c61fce4`) the skels carry the f16-reciprocal
quantizer bug, and their md5s differ from the ones above.

After replacing any skel: update the md5s above and re-run the full
on-device determinism matrix (logits m=1..8 x5, greedy md5 2x2x6,
256-token pair): skel/host skew fails silently as wrong numerics.

## Coverage

v73 (8+ Gen 1 / 8 Gen 2 / 7+ Gen 2 / X Elite), v75 (8 Gen 3 / 8s Gen 3),
v79 (8 Elite), v81 (next-gen). No v68/v69 (888 / 8 Gen 1): outside
llama.cpp upstream scope; see the packaging doc for the support matrix.

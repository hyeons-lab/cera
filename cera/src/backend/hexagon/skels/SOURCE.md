# Bundled Hexagon DSP skels

`libcera-htp-v{73,75,79,81,85}.so` are the DSP-side (Hexagon ELF) worker
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

- Source: llama.cpp at upstream commit `e95426f16` (2026-11-07) plus three local Hexagon commits on
  top (`ead90bfd3`, `cf01d2b60`, `c2d4a889a`: the extended Conv1D, ConvTranspose1D, Snake and unary
  operations, and the HVX float-clamp fix), then the quantizer patch below as `93b1163c9` and
  `b5d25b652`.
  These commits live in a local, unpublished llama.cpp branch (`feat/hexagon-nov-rebase`;
  `ggml/src/ggml-hexagon/htp/` + `htp_iface.idl`, plus the op-remap entries for our extensions in
  `htp-kparams.cpp`); the previous skels were built from upstream `de7fa0a3c` with the same
  extensions.
- Host contract: every op's `kernel_params` come from the host, and every layout changed with the
  refresh (the FastRPC `start` call also gained `vtcm_size` and `power_mode`; `HtpOpCode` is
  regenerated from `htp-ops.h`). `params.rs`, `matmul_params.rs` and `eltwise.rs` port llama.cpp's
  `htp-kparams.cpp` solvers and the layout helpers in `htp/*-ops.h` word for word, and each is
  pinned by golden files (`testdata/*_golden.txt`) that `scripts/hexagon-golden/gen.py` produces by
  compiling llama.cpp's own, unmodified solver sources against ggml-base and running them on a case
  table (`LLAMA_CPP` set to a checkout). A golden proves the host sends what llama.cpp's host
  would send; it cannot prove the DSP accepts it for a shape upstream never produces, so the
  on-device runs below matter. Deliberate deviations from the C++ (each pinned by a test):
  - a copy whose source view overlaps itself (cera builds a conv's K-window that way) skips the
    tiled-transpose kernel, whose input DMA faulted the DSP, and takes the generic reshape kernel;
  - HMX matmul needs 8 rows (upstream: 5), because the HMX path is racy below 8 on a v79;
  - single-row flash attention stays on HVX and HMX attention needs 5 query rows, as before the
    refresh: the solver's long-KV crossover to HMX for one row measured slower on a v79 (LFM2
    decode at 2048 tokens about 10% down, a Whisper decode step 9.2 ms against 6.4 ms);
  - a softmax mask given as a strided view (the audio encoder's relative-position shift) is
    accepted, since `softmax-ops.c` DMAs mask rows by stride;
  - session thread counts of 0 are treated as 1 where the C divides by zero.
- Patch: `patches/0001-hvx-q8-activation-quantizer-f32-reciprocal.patch`
  (`hvx-mm-kernels-tiled.h`, the `q8_0` and `q8_1` tiled activation
  quantizers, including the Q2_K `sums16` path that shares one scale per 256-element
  super-block, and `hvx-quant.h`, the Q8_0 row quantizer; the shared scale rule lives in the new
  `hvx-q8-scale.h`). The upstream quantizer computed the Q8 scale and its
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
  0.9995, with unchanged decode throughput (measured on an earlier build; the patch is ported
  onto the reworked quantizers). The row quantizer `hvx_quantize_row_q8_0_f32` had the same f16
  reciprocal in its four-block path, and its scalar tail stored the f16-rounded scale but
  multiplied by an unrounded reciprocal. It quantizes cache and gather rows (the Q8_0 KV cache
  through `set_rows`, `get_rows`, flash attention; `CERA_HEXAGON_KV_Q8`, off by default), not
  matmul activations. It now uses the same rule as the tiled quantizers; on a v79 the device test
  `device_set_rows_q8_0_keeps_small_blocks` (blocks with maxima from 1 down to 1e-5, vector path and
  scalar tail) came back 100% off on every block below about 2e-3 before the fix and within one
  quantization step (0.58% of the block maximum) after it. The tiled quantizers were only
  refactored onto the shared helper: full-logit cosines against the CPU are unchanged to six digits
  on four models (including a 1,300-token context), and decode replays stay bit-identical.
  v73, v75, v81 and v85 are the same C code rebuilt; only v79 has been run on hardware.
  The v85 skel is built with the newer toolchain below, so the same sources were also built for
  v79 with it and run on the same device: Q8_0 SetRows test, full-logit cosines against the CPU
  (four models, including a 1,300-token context), decode replays and throughput all match the v0.7
  build (cosines identical to six digits, decode replays bit-identical, throughput within noise).
- Build: Hexagon SDK 6.6.0.0 with Hexagon Tools 19.0.07 (hexagon-clang with
  whole-program LTO), compiling `libggml-htp-v{73,75,79,81}.so` targets with
  `-DDSP_VERSION`. Upstream's CMake then runs `htp-swiv.py`, which appends a SWIV (software image
  validation) section: a CRC32 over the contents of every loadable segment. Upstream builds with
  toolchain image v0.8 (Hexagon SDK 6.7.0.0, Hexagon Tools 21.0.04), which the v85 target needs;
  the v0.7 image has no v85 mapping, so v85 alone is built with v0.8 (ELF `e_flags` 0x85, as the
  v81 file's is 0x81) while v73 to v81 stay on v0.7. The data segments of
  these skels are 4 KB aligned (`p_align = 0x1000`), the DSP loader's own requirement; Android's
  16 KB page rule applies to the app's `.so` files, not to these embedded blobs.
- License of the compiled sources: MIT (llama.cpp). Toolchain output
  is not SDK redistribution.

## Integrity (md5)

- v73: 8089a9c1c6de3e34d4ad989b04a3a8d3
- v75: c33d7288d58e0e23d5573fc0b3f9e775
- v79: d0f7324da78d2fffbf3cf25639fa6890
- v81: a1c31cd344e31f0a96b04d61adc2dc3c
- v85: 756d4cdcf6573351b454c76de3139b57

These are the md5s of the shipped, renamed files. The unrenamed build outputs were
v73 7961e516e3744fa9d4f5c6d2ad3bae04, v75 041365ae69f9fcc0a4567f7118ba2377,
v79 36a6fa4c73efbb6cec7d29ab8545145e, v81 0a8519321dc95621ff3b2398a04b9a12 and
v85 b9b676dcd658872d2d9610d1bbe980b4. `scripts/hexagon-skels/finalize.py --check <skel>` verifies a
file's SWIV CRC and lists the SONAME inside.

## Rebuilding

```bash
# in the llama.cpp checkout described under Source (commit b5d25b652), Docker toolchain image
# ghcr.io/snapdragon-toolchain/arm64-android:v0.7 (Hexagon SDK 6.6.0.0, Tools 19.0.07):
cp docs/backend/snapdragon/CMakeUserPresets.json .
docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace <image> bash -lc \
  'cmake --preset arm64-android-snapdragon-release -B build-snapdragon &&
   cmake --build build-snapdragon --target htp-v73 htp-v75 htp-v79 htp-v81'
# outputs: build-snapdragon/ggml/src/ggml-hexagon/htp-vXX-prefix/src/htp-vXX-build/libggml-htp-vXX.so
# (htp-v79 is also installed one level up; the other targets stop before their install step when
# one target in the same invocation fails)

# v85 alone, with toolchain image ghcr.io/snapdragon-toolchain/arm64-android:v0.8 (Hexagon SDK
# 6.7.0.0, Tools 21.0.04), in its own build directory so the v0.7 outputs stay untouched:
docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace <image-v0.8> bash -lc \
  'cmake --preset arm64-android-snapdragon-release -B build-snapdragon-v08 &&
   cmake --build build-snapdragon-v08 --target htp-v85'
```

## Renaming (llama.cpp coexistence)

The build outputs are named `libggml-htp-vXX.so`, the same names llama.cpp's own Hexagon backend
ships, so an app embedding both would have one overwrite the other in `ADSP_LIBRARY_PATH`, and
the DSP loader could treat the equal SONAMEs as one library. The shipped files are therefore
renamed `libcera-htp-vXX.so` (the host opens them by that name, `HexagonArch::skel_filename`)
and the SONAME in each is patched to match. `ggml` and `cera` are both four characters, so the
patch is a same-length byte substitution that moves no offsets. It does change loadable bytes,
which the SWIV CRC covers, so the CRC is recomputed in the same step:

```bash
for v in 73 75 79 81 85; do
  python3 scripts/hexagon-skels/finalize.py libggml-htp-v$v.so cera/src/backend/hexagon/skels/libcera-htp-v$v.so
done
```

The tool first recomputes the input's CRC and refuses to continue unless it equals the one
upstream's script stored, so its CRC computation is checked against the real one on every run.

Without the quantizer patch (the commit before `93b1163c9`) the skels carry the f16-reciprocal
quantizer bug, and their md5s differ from the ones above.

After replacing any skel: update the md5s above and re-run the on-device checks. Skel/host skew
fails silently as wrong numerics, and a golden file cannot see a shape the DSP rejects:
`hexagon_logit_parity` (NPU against CPU logits, short and long context), `hexagon_decode_probe`
(logits and greedy reproducibility), and the family probes (`hexagon_vit_probe`,
`hexagon_whisper_probe`, `hexagon_vad_probe`, `hexagon_conformer_probe`,
`hexagon_nemotron3_probe`), each against the previous skels. `CERA_HEXAGON_STEP=1` with
`CERA_HEXAGON_DEBUG=1` names the op group that stops a DSP batch from being answered.

## Coverage

v73 (8+ Gen 1 / 8 Gen 2 / 7+ Gen 2 / X Elite), v75 (8 Gen 3 / 8s Gen 3),
v79 (8 Elite), v81 (next-gen), v85 (Snapdragon 8 Elite Gen 6). No v68/v69 (888 / 8 Gen 1)
or v83: outside llama.cpp upstream scope; see the packaging doc for the support matrix.

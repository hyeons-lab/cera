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
  operations, and the HVX float-clamp fix), then the quantizer patch below as `93b1163c9`.
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
  super-block). The upstream quantizer computed the Q8 scale and its
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
  onto the reworked quantizers). Not covered: `hvx_quantize_row_q8_0_f32` (`hvx-quant.h`) still has
  the f16 reciprocal. It quantizes cache and gather rows (the Q8_0 KV cache through `set_rows`,
  `get_rows`, flash attention), not matmul activations, and `CERA_HEXAGON_KV_Q8` is off by default.
  v73, v75, v81 and v85 are the same C code rebuilt; only v79 has been run on hardware.
- Build: Hexagon SDK 6.6.0.0 with Hexagon Tools 19.0.07 (hexagon-clang with
  whole-program LTO), compiling `libggml-htp-v{73,75,79,81}.so` targets with
  `-DDSP_VERSION`. Upstream's CMake then runs `htp-swiv.py`, which appends a SWIV (software image
  validation) section: a CRC32 over the contents of every loadable segment. Upstream moved its own
  builds to toolchain image v0.8 (SDK 6.7), which the native v85 target needs; the v0.7 image used
  here has no v85 mapping, so the shipped v85 skel is the v81 build renamed. The data segments of
  these skels are 4 KB aligned (`p_align = 0x1000`), the DSP loader's own requirement; Android's
  16 KB page rule applies to the app's `.so` files, not to these embedded blobs.
- License of the compiled sources: MIT (llama.cpp). Toolchain output
  is not SDK redistribution.

## Integrity (md5)

- v73: 0b461e10fa560638cbdea5491cfb59dd
- v75: 8b97f3f546f57d47ed561daf97152a16
- v79: 9aae26d385a63b438e267498ac33450d
- v81: 879206b0859edc3f4ce164317eb6af59
- v85: db965f4730864b9d7c7ce932cb7ef46f

These are the md5s of the shipped, renamed files. The unrenamed build outputs were
v73 4bb07b3937f64bd7c0462a0807918116, v75 20a6fc16c93f628c481aff79363f93f5,
v79 707ce806982a9c703ee85f0795c16d00, v81 95611a06ab5b84d3002bf9424bc64d02
(and v85 is that same v81 file). `scripts/hexagon-skels/finalize.py --check <skel>` verifies a
file's SWIV CRC and lists the SONAME inside.

## Rebuilding

```bash
# in the llama.cpp checkout described under Source (commit 93b1163c9), Docker toolchain image
# ghcr.io/snapdragon-toolchain/arm64-android:v0.7 (Hexagon SDK 6.6.0.0, Tools 19.0.07):
cp docs/backend/snapdragon/CMakeUserPresets.json .
docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace <image> bash -lc \
  'cmake --preset arm64-android-snapdragon-release -B build-snapdragon &&
   cmake --build build-snapdragon --target htp-v73 htp-v75 htp-v79 htp-v81'
# outputs: build-snapdragon/ggml/src/ggml-hexagon/htp-vXX-prefix/src/htp-vXX-build/libggml-htp-vXX.so
# (htp-v79 is also installed one level up; the other targets stop before their install step when
# one target in the same invocation fails)
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
for v in 73 75 79 81; do
  python3 scripts/hexagon-skels/finalize.py libggml-htp-v$v.so cera/src/backend/hexagon/skels/libcera-htp-v$v.so
done
# the v85 file is the v81 build under its own name:
python3 scripts/hexagon-skels/finalize.py --as v85 libggml-htp-v81.so cera/src/backend/hexagon/skels/libcera-htp-v85.so
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

# Bundled Hexagon DSP skels

`libcera-htp-v{73,75,79,81,85}.so` are the DSP-side (Hexagon ELF) worker
libraries the NPU backend loads into the CDSP unsigned PD at runtime.
They are built from llama.cpp sources plus a patch series (`patches/`, described
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

- Source: llama.cpp at upstream commit `e95426f16` (2026-11-07) plus the seven patches in `patches/`,
  applied in filename order with `git apply`. They are exported from eight commits of a local,
  unpublished llama.cpp branch (`feat/hexagon-nov-rebase`), and the series applied to a bare
  `e95426f16` gives exactly that branch's tree (compared by tree hash):
  - `0001` is the quantizer patch below (`93b1163c9` and `b5d25b652`, as one combined diff);
  - `0002` adds the Conv1D op and the Snake activation (`ead90bfd3`);
  - `0003` adds ConvTranspose1D, Sin, Cos, HardSigmoid, HardSwish and ELU, with the op-remap
    entries for these extensions in `htp-kparams.cpp` (`cf01d2b60`);
  - `0004` corrects the HVX float clamping intrinsics and pointer arithmetic in `unary-ops.c`
    (`c2d4a889a`);
  - `0005` pins the ConvTranspose1D operand order, makes the depthwise Conv1D vector path
    alignment-safe and makes both conv ops refuse inconsistent shapes (`d8b012aee`);
  - `0006` makes Snake and ELU read their alpha as given (`3cf6cd686`);
  - `0007` makes both conv ops refuse results the worker would leave partly unwritten
    (`95e217f83`).

  `0001` shares no file with the others, so its place in the order does not matter; `0002` to
  `0007` all touch `conv1d-ops.c` or `unary-ops.c` and must stay in order. The previous skels were
  built from upstream `de7fa0a3c` with the same extensions.
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
  - HMX matmul needs 8 rows (upstream: 5), because the HMX path is racy below 8 on a v79; a
    session from `MmSession::new` carries that floor (`hmx_min_rows`), so every solver applies it,
    and the golden tests build their sessions with upstream's floor to pin the unmodified solver;
  - short chunks of small heads stay on HVX: `flash_attn_hmx_rows_ok` allows HMX attention from 5
    query rows for head dims up to 128 (larger heads keep the solver's choice), and decode asks for
    HVX only, as before the refresh: the solver's long-KV crossover to HMX for one row measured
    slower on a v79 (LFM2 decode at 2048 tokens about 10% down, a Whisper decode step 9.2 ms
    against 6.4 ms);
  - a softmax mask given as a strided view (the audio encoder's relative-position shift) is
    accepted, since `softmax-ops.c` DMAs mask rows by stride;
  - a session with no threads counts as one in the matmul and elementwise solvers and is refused
    by the row-sum and row-gather layouts, where the C divides by the count;
  - an elementwise (binary, unary, GLU) operand whose extent from first to last byte exceeds the
    `u32` size of a registered descriptor is refused; the C holds sizes in 32 bits and would plan
    on wrapped, smaller ones;
  - a unary or GLU operand whose planes or batches sit a gap apart (a slice of a larger buffer) is
    not collapsed: upstream's merge and split steps rewrite the plane strides from the row stride
    and the new extents, so such a view came out addressing other bytes than it went in with. The
    DSP reads the uncollapsed descriptor correctly (`device_unary_ops_read_views_with_gaps_between_planes`).
  Kept as upstream, not deviations: the HMX flash-attention layout's VTCM size is recomputed with
  the final pipeline flag and is not re-checked against the budget. The fusion pass's kernel checks
  (`is_supported_mul_mat_nx_kernel` and its routed twin in `htp-fusion.cpp`) are copied by hand
  into the `mmnx` and `mmidnx` oracles; `gen.py` fails when their upstream text changes.
- Quantizer patch: `patches/0001-hvx-q8-activation-quantizer-f32-reciprocal.patch`
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
  v73, v75 and v81 are the same C code rebuilt and have not been run on hardware; v79 (Galaxy S25
  Ultra) and v85 (an SM8975, below) have.
- Extension patches `0005` to `0007` (the ops `0002` and `0003` added): ConvTranspose1D took its
  kernel from `src[0]` or `src[1]` by comparing widths, so an input no wider than the kernel swapped
  the roles and returned garbage; it is now always `src[0]` `[K, C_out, C_in]` and the input
  `src[1]`, the ggml order. The depthwise stride-1 Conv1D vector path loaded and stored through
  plain `HVX_Vector` pointers at arbitrary element offsets, which hit the 128-byte block below the
  address; it goes through `hvx_vmemu` now and only runs for contiguous rows. Both conv ops return
  `NO_SUPPORT` when the groups do not divide the channels or the weight, input, bias and result
  extents disagree, when a fourth extent is not 1, and (ConvTranspose1D) when the result has other
  batches than the input, whose slabs would come back unwritten (`0007`). The host refuses the
  same shapes first (`conv1d_shapes_ok`, `conv_transpose1d_shapes_ok`). Snake and ELU
  used to read an all-zero alpha as "not supplied" and take 1.0; they read it as given (Snake with
  alpha 0 is the identity, ELU with alpha 0 a ReLU), so callers pass 1.0 for the default.
  Evidence on a v79 (`dispatch.rs` device tests): against the skel before the patches the
  depthwise Conv1D cases were off by 0.7 to 37 (relative to the largest output), ConvTranspose1D
  with an input no wider than the kernel by about 1, Snake and ELU with alpha 0 by 17%, and shapes
  with more groups than output channels were accepted; with the patched skel every conv case is
  within 1.2e-7, alpha 0 is exact, and Sin, Cos, Snake,
  ELU, HardSigmoid and HardSwish stay within 4e-7 on rows ending in a partial vector. Full-logit
  cosines against the CPU (LFM2.5-350M at 11 and 1,300 prompt tokens, LFM2.5-2.6B, LFM3-1.7B),
  argmax agreement and decode replays are the same as before the patches. The DSP's own refusals
  are tested with ops sent past the host's validation (`enqueue_op_unvalidated`, eight cases, each
  breaking one rule, then a valid conv): all refused with `NoSupport`, and the skel before `0007`
  accepts the fourth-extent case. cera has no host path
  that sends these ops yet, so those device tests are their only numeric coverage; the same tests
  pin the Q8_0 row quantizer's NaN and infinity contract and a Q4_1 matmul over activations with
  block maxima down to 3e-4 (cosine 0.99999), which the quantizer patch fixed earlier.
  The v85 skel has run on an SM8975 (Android 17, `DSP architecture V85`): the same 14 device tests
  pass with the same numbers as on the v79 (conv within 1.2e-7, alpha 0 exact, Q8_0 SetRows within
  0.58% of a block's maximum, Q4_1 and partial-tile matmuls at cosine 0.99999), and against the
  CPU the full-logit cosine is 0.9997 or better over 12 decode steps of LFM2-700M, 0.9985 at a
  1,300-token context and 0.9985 for LFM3, with the argmax agreeing on every step and decode
  replays bit-identical.
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

- v73: 76c72db29fae2ecc6b47a32a955abbb4
- v75: ab2eb03f4ae3ae5329911456ce853a97
- v79: 06b17ec6e60dabdbb2ec2135d16c2c65
- v81: 1f0d967ff00122fa977317aa30e048c5
- v85: 6b159fca504a9b984dc588108d88274e

These are the md5s of the shipped, renamed files. The unrenamed build outputs were
v73 377925c5e8768578e8ae3a945805d8c7, v75 4482706490c14c933b0bcbe72e3e2045,
v79 d6ff8c37af1094ffeb18476596d894c7, v81 9e17813855cc6e0a3ffc0956abd35708 and
v85 d188dbd182bc6978e0d7ee046b9d100a. `scripts/hexagon-skels/finalize.py --check <skel>` verifies a
file's SWIV CRC and lists the SONAME inside.

## Rebuilding

```bash
# in a llama.cpp checkout of upstream e95426f16 with the series from patches/ applied
# (CERA is this repository's root):
git checkout e95426f16
for p in "$CERA"/cera/src/backend/hexagon/skels/patches/*.patch; do git apply "$p"; done
# Docker toolchain image
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

Without the quantizer patch (`0001`) the skels carry the f16-reciprocal quantizer bug, and their
md5s differ from the ones above.

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

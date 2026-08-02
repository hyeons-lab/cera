# Slang multi-target: Mac test instructions

> TEMPORARY file on the `feat/slang-multitarget` branch. Delete before any merge.

## What this branch is

cera maintains every GPU kernel twice: 31 WGSL files and 44 MSL files, about 11k
lines kept in agreement by hand. Slang can emit both from one source, and slangc
is already in the build from the SPIR-V passthrough work (#333/#334). This wires
up multi-target emission (one `.slang` to WGSL *and* MSL) and pilots it on
`softmax`.

`softmax` was chosen deliberately. The two handwritten kernels already agree on
their binding contract but **disagree on their reduction**: `softmax.metal` uses
a two-stage `simd_max`/`simd_sum`, `softmax.wgsl` walks a shared-memory tree
because cera never requests `wgpu::Features::SUBGROUP`. So this is not a test of
whether Slang can emit two languages; it is a test of whether a generated kernel
can keep a per-target fast path instead of flattening both to the portable one.

`__target_switch` does that, and the untaken branch is eliminated rather than
compiled and skipped: the generated MSL has no trace of the tree, the generated
WGSL has no subgroup op. Only the two reduction helpers branch. Layout, the
grid-stride loops, the max shift, and the normalize pass are written once.

**Nothing is on the production path.** `SOFTMAX_SLANG` sits beside the
handwritten `SOFTMAX` on both backends, and only the parity suite uses it. A
wrong generated kernel breaks nothing.

## Why your Mac is the point

The MSL half has **never been compiled or executed anywhere**. It cannot be built
on a Linux or Windows dev box at all, and CI's runner is a paravirtualized Apple
device, which settles correctness but says nothing about speed.

| | status |
|---|--------|
| WGSL emitted, validated by naga, run on lavapipe vs CPU reference | verified, 5e-7 |
| no-slangc fallback build from committed files | verified |
| MSL compiles | **unknown** |
| MSL numerically correct on Apple silicon | **unknown** |
| MSL as fast as the handwritten kernel | **unknown, and not measurable by CI** |

## Run it

slangc is NOT required: build.rs falls back to the committed `.wgsl` / `.metal`
next to the `.slang`, which is exactly the artifact this test should exercise.

```sh
git fetch origin
git checkout feat/slang-multitarget
cargo test -p cera --features metal --test slang_multitarget_parity -- --nocapture
```

Expect four tests, all passing:

```
test msl::softmax_slang_shorter_than_workgroup ... ok
test msl::softmax_slang_ragged ... ok
test msl::softmax_slang_exact_multiple ... ok
test msl::generated_msl_keeps_simd_reduction ... ok
```

Each numeric test prints its own error, e.g.
`msl softmax n=1000: max_abs_err=... sum=1.000000 OK`.

To make a missing Metal device a hard failure rather than a skip:

```sh
CERA_REQUIRE_METAL=1 cargo test -p cera --features metal --test slang_multitarget_parity
```

## Reading a failure

The four tests fail for different reasons and the distinction matters.

- **A numeric test fails** (`max_abs_err > tol`, or `sum != 1`): the generated
  MSL is wrong. Send the printed `max_abs_err` and which lengths failed. n=100 is
  the interesting one, since most of the 256 threads contribute only the identity
  to both reductions, so a wrong `-inf`/`0` seed in the metal branch shows up
  there and nowhere else.
- **`generated_msl_keeps_simd_reduction` fails**: the kernel is *correct* but
  `__target_switch` selected the portable tree, so Metal silently lost its simd
  path. This is asserted on the shader text because there is no runtime way to
  observe which branch survived, and every numeric test would still pass.
- **Compilation fails** (`compile generated MSL` panics): slangc's MSL is not
  accepted by Metal's runtime compiler. Send the full error; this is the single
  most likely failure and the main thing worth knowing.

## What to send back

1. Pass/fail per test, plus the `max_abs_err` lines.
2. `system_profiler SPDisplaysDataType | head -5` so the GPU is on record.
3. If anything failed, the full output.

## The performance question is not answered here

Correctness is all these tests check. The question that decides whether Slang
adoption goes any further is whether the **generated** MSL is as fast as the
**handwritten** MSL, and there is no harness for that yet: comparing
`shaders::SOFTMAX_SLANG` against `shaders::SOFTMAX` on identical input needs a
small bench that does not exist on this branch. Worth building only if the
correctness run above passes.

Note also that `softmax` is not where the money is. The eight shaders using
`simdgroup_matrix` (the prefill GEMMs) are the hot path, they are hand-tuned
around simdgroup matrix ops, and nothing here says whether an emitter can match
them. A reduction helper is a much easier case than a tiled matmul, so a clean
pass here should not be read as a green light for the GEMMs.

## Regenerating the committed outputs

Only needed if you edit the `.slang`. Requires slangc 2026.13.1 (the CI drift
check byte-compares against that version):

```sh
just slang
```

That now regenerates both the SPIR-V passthrough kernels and the multi-target
WGSL/MSL pair. CI fails if a committed output differs from what the pinned
slangc produces; for the `.metal` file that guard is the only thing standing
between a stale artifact and a failure that surfaces on a user's device.

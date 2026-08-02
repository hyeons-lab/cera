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

## Then the performance question

Correctness is all the tests above check. What decides whether Slang adoption
goes further is whether the **generated** MSL is as fast as the **handwritten**
one. Run this second, once the parity suite passes:

```sh
cargo run -p cera --features metal --release --example slang_softmax_bench
```

Release matters: a debug build measures the harness.

It prints two sections. **Agreement** first, dispatching both kernels on the same
input at three sizes, because a faster arm that disagrees is not a faster arm.
Then **timing**, at n = 1024 / 4096 / 16384 / 65536:

```
       n   handwritten     generated     ratio
    1024          8.31          8.44     0.985x
```

`ratio > 1.00` means the generated kernel is faster. Treat anything within a few
percent of 1.00 as no difference; this is wall-clock timing of a kernel that runs
in microseconds.

The harness alternates arms every round, discards a warm-up round per size,
amortizes submit cost over 200 dispatches per timed encoder, and reports the
median of 7 rounds rather than the mean. That discipline is not ceremony: on the
AMD host the same style of measurement drifted 4x mid-session, and only the
interleaved ratio survived it.

If the generated MSL lost its simd path, the bench says so before the table
rather than leaving you to infer it from a suspiciously round regression.

**What to send back:** the agreement lines and the whole timing table.

Note also that `softmax` is not where the money is. The eight shaders using
`simdgroup_matrix` (the prefill GEMMs) are the hot path, they are hand-tuned
around simdgroup matrix ops, and nothing here says whether an emitter can match
them. A reduction helper is a much easier case than a tiled matmul, so a clean
pass here should not be read as a green light for the GEMMs.

## Results (Apple M1 Max, 2026-08-02)

Ran on an Apple M1 Max (32-core GPU, Metal 4), macOS, from the worktree at branch
tip. slangc 2026.13.1 on PATH.

**Correctness.** All four parity tests pass, including
`generated_msl_keeps_simd_reduction` (the generated MSL kept its two-stage
`simd_max`/`simd_sum`; it did not fall back to the tree). Numeric agreement:

```
test msl::generated_msl_keeps_simd_reduction ... ok
msl softmax n=100:  max_abs_err=8.941e-8 sum=1.000000 OK
msl softmax n=1000: max_abs_err=9.313e-9 sum=1.000000 OK
msl softmax n=2048: max_abs_err=9.313e-10 sum=0.999999 OK
test msl::softmax_slang_shorter_than_workgroup ... ok
test msl::softmax_slang_ragged ... ok
test msl::softmax_slang_exact_multiple ... ok
test result: ok. 4 passed
```

**Speed: a real small-n regression, found and fixed.** The first bench run showed
the generated kernel losing at the small sizes and only reaching parity once the
work grew:

```
       n   handwritten     generated     ratio
    1024         10.33         13.61    0.759x   <- ~24% slower
    4096         14.39         14.96    0.962x
   16384         46.54         46.99    0.990x
   65536        171.04        170.86    1.001x
```

Cause: the generated MSL emitted **five** `threadgroup_barrier`s where the
handwritten `softmax.metal` has **four**. The extra one was the inter-phase
(max -> exp/sum) barrier, which `softmax.slang` issued unconditionally. On Metal
it is redundant: `block_max` already ends with a trailing barrier, and the
handwritten kernel does that exact phase transition with none added. At small n
the kernel is barrier-latency-bound (~4 elements per thread at n=1024), so 5/4
barriers tracks the ~24% gap almost exactly.

Fix: made that barrier target-conditional in `softmax.slang` (`__target_switch`,
dropped on `metal`, kept on the portable `default`/tree path), then `just slang`.
The generated MSL is now 4 barriers, structurally identical to the reference;
`softmax.wgsl` is byte-unchanged. After:

```
       n   handwritten     generated     ratio
    1024         13.12         13.42    0.978x   <- was 0.759x
    4096         13.58         13.88    0.979x
   16384         43.96         43.75    1.005x
   65536        170.26        171.51    0.993x
```

Every size is now within ~2% of parity (the bench's own "no difference" band),
agreement still bit-identical (`0.000e0`) at all sizes. This is exactly the class
of silent regression the perf bench exists to catch: correctness never saw it,
only the ratio did. (Absolute µs are not comparable across the two runs; only the
within-run interleaved ratio is, which is why the handwritten n=1024 column moved
too.)

## Round 2: the simdgroup_matrix probe (new, needs a Mac run)

Since the softmax result landed, the branch gained `coopmat_probe.slang`, which
answers the question that decides whether the eight hand-tuned
`simdgroup_matrix` GEMMs are portable at all.

**What it establishes.** `linalg::CoopMat` lowers to genuine Metal MMA
(`simdgroup_load`, `simdgroup_multiply_accumulate`, `simdgroup_store`,
`make_filled_simdgroup_matrix`), and the *same source* still emits valid WGSL.
Unguarded it does not: WGSL has no cooperative-matrix type and the entry point
fails with `E36107 unavailable features in entry point`. It compiles only because
`__target_switch` eliminates the metal branch **before** entry-point capability
validation runs. That ordering is undocumented, so it is pinned by a test rather
than left as a comment.

**Nothing dispatches the probe.** It is a fixed 8x8 with no tiling, no
threadgroup staging, no ragged edges and no dequant, and the assertions are on
the emitted shader text, because there is no runtime way to observe which
instructions the compiler selected.

Two of the three new tests already ran here (they are `gpu`-gated). The third is
Metal-gated and has **never executed on Apple silicon**:

```sh
cargo test -p cera --features metal --test slang_multitarget_parity -- --nocapture
```

Expect **seven** tests now, up from four. The new one to watch:

```
test coopmat_probe_reaches_metal_mma ... ok
```

If it fails, the assertion message names the missing instruction. That means
Slang stopped lowering `linalg::CoopMat` to Metal MMA, which would retire the
GEMM-porting idea entirely, so it is worth reporting verbatim rather than
working around.

The other two (`coopmat_probe_wgsl_falls_back_cleanly`,
`coopmat_probe_documents_the_f16_requirement`) are `gpu`-gated and will not run
in a `--features metal` build; they pass on Linux already.

**The f16 constraint, since it will bite any real port.** Typing the operands
`half` makes the WGSL emission open with `enable f16`, and cera requests
`SHADER_F16` only when the adapter reports it (`GpuContext::new`). This is the
first shader in the tree to emit `enable f16` at all, and it is why nothing
builds a pipeline from the probe. A real GEMM port has to keep f32 operands or
gate on the feature.

**No bench for this one.** `slang_softmax_bench` is unchanged and still only
compares the two softmax kernels. Timing the probe would measure a single 8x8
tile, which says nothing about a tiled GEMM, so it would be a number that invites
over-reading rather than one worth having.

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

# Slang multi-target shaders

This directory holds GPU kernels written once in [Slang](https://shader-slang.org/)
and emitted to **both** WGSL (for the wgpu backend) and MSL (for the Metal
backend) from a single source. It exists to test whether cera can stop
maintaining every GPU kernel twice (WGSL and MSL kept in agreement by hand) and
generate both from one file instead.

Each `<name>.slang` is compiled by `just slang` (slangc 2026.13.1, pinned) into
the committed `<name>.wgsl` and `<name>.metal` next to it. Those committed
outputs are the build's fallback when slangc is absent (CI runners without a
Slang toolchain); `build.rs` regenerates from the `.slang` into `OUT_DIR` when
slangc is present, so the `.slang` is the source of truth. CI byte-compares the
committed outputs against what the pinned slangc produces, so a stale `.metal`
or `.wgsl` fails the build rather than silently shipping to a device.

**These are the production kernels.** The evaluation is over: with the four
exceptions below, every kernel here is what the backends dispatch, and the
handwritten WGSL/MSL twins they replaced have been deleted. A wrong generated
kernel is now a wrong kernel, which is why `tests/slang_multitarget_parity.rs`
pins each one against the CPU reference.

The exceptions, and why:

- **The LFM2A audio-encoder tier is Metal-live and WGSL-inert.**
  `conv2d_direct`, `transpose_blocked`, `glu_split`, `chan_affine_silu`,
  `activations` and `audio_xl_attention` are dispatched by the Metal audio
  encoder (`model/audio_encoder_gpu.rs`) and pinned numerically against the CPU
  encoder by `tests/audio_encoder_metal_parity.rs`. Their WGSL halves are
  generated, committed and drift-checked like everything else here, but nothing
  dispatches them yet: the wgpu audio encoder is a later change. So for these
  six, `slang_multitarget_parity.rs` carries only generation checks (entry points
  present, no subgroup ops, no `enable f16`), and wiring wgpu means adding
  numeric cases, not just an ops impl. They are written as one source rather than
  handwritten pairs precisely so that is the only work left.

- `elementwise.slang` covers all four entry points `elementwise.wgsl` has, but
  only four of the eight in `elementwise.metal`. The other four (`memcpy_f32`,
  `scale_f32`, `mul_out`, `cast_f32_to_f16`) are Metal-only, with no WGSL
  counterpart to share a body with, so the Metal side cannot switch. Flipping
  only wgpu would give up the single shared source that is the point of this
  directory, so both backends keep the handwritten kernel and the port stays
  reachable as `ELEMENTWISE_SLANG`.
- `gemm_q8_0.slang` is a cooperative-matrix experiment, not a replacement for
  the hand-tuned `simdgroup_matrix` GEMM. See `GEMM_Q8_0_SLANG`.
- `coopmat_probe.slang` is a capability probe rather than a kernel. Nothing
  dispatches it; `tests/slang_multitarget_parity.rs` asserts it compiles.

## The one primitive that makes it work: `__target_switch`

The two backends do not always want the same code. `__target_switch` lets a
single source carry a per-target fast path, and the untaken branch is
*eliminated* before entry-point capability validation, not compiled and skipped.
So the `case metal:` body may use a capability WGSL cannot express at all (a
subgroup op, a cooperative matrix, a raw pointer cast), and the emitted WGSL
contains no trace of it. This is what makes the single source honest rather than
a `#ifdef` lie.

Where Slang's portable abstractions are too slow on Metal, the `case metal:`
branch drops to Metal directly via `__target_intrinsic`, which emits a verbatim
MSL string. Because those helpers are only *called* inside `case metal:`, the
WGSL branch never references them and stays a clean portable shader. See
`gemm_q8_0.slang` for the load/store intrinsics this required.

## Pilot kernels and what each settled

### `softmax.slang` (reduction with a per-target fast path)

`softmax` was chosen because the two handwritten kernels agree on their binding
contract but **disagree on their reduction**: Metal uses a two-stage
`simd_max`/`simd_sum`, WGSL walks a shared-memory tree because cera never
requests `wgpu::Features::SUBGROUP`. `__target_switch` keeps both, writing the
layout, grid-stride loops, max-shift, and normalize pass once and branching only
the reduction. `generated_msl_keeps_simd_reduction` asserts on the shader text
that the MSL kept its simd path (there is no runtime way to observe which branch
survived).

Finding: structural equivalence is not performance equivalence. The first
generated MSL emitted **five** `threadgroup_barrier`s where the handwritten
kernel has four; the parity suite passed, but the extra inter-phase barrier cost
~24% at n=1024, where the kernel is barrier-latency-bound. Making that barrier
target-conditional (dropped on Metal, kept on the portable tree path) closed it
to within ~2% at every size.

### `coopmat_probe.slang` (does Slang reach Metal MMA at all?)

A minimal probe that answers the question deciding the whole GEMM effort: does
`linalg::CoopMat` lower to genuine Metal `simdgroup_matrix` hardware? It does.
The generated MSL carries `simdgroup_load` / `simdgroup_multiply_accumulate` /
`simdgroup_store` with the fp16-operand, fp32-accumulator type shape the
hand-tuned GEMMs use, and the *same source* still emits valid WGSL (the CoopMat
path is eliminated before WGSL capability validation runs). The f16 operand
typing forces `enable f16` in WGSL, so a real GEMM port must keep f32 operands or
gate on `SHADER_F16`.

### `gemm_q8_0.slang` (the real hot-path kernel)

A port of the hand-tuned Q8_0 simdgroup GEMM (`../gemm_q8_0.metal`): same 64x32
tile, same 8 KB threadgroup budget, same mixed-precision MMA. This is the kernel
that decides whether a migration is worth doing, and it is where the interesting
performance work happened.

A naive port that expressed the hot path in portable Slang ran at **0.64x** the
handwritten kernel. Profiling (three plausible causes measured and *rejected*
first, not guessed) traced almost all of it to the Q8_0 dequant: `src0` is bound
`StructuredBuffer<uint>` because WGSL has no byte-addressable storage, so the
portable path rebuilds every quant with a word load, a shift, a mask and a manual
sign-extend. That per-byte arithmetic, not the MMAs, was the gap, and it inflated
register pressure enough to cut occupancy too.

The fix is metal-only direct memory access via `__target_intrinsic`, leaving the
portable branch untouched:

| step | large-shape ratio | occupancy (maxThreadsPerTG) |
|------|-------------------|-----------------------------|
| naive portable port | 0.64 | 768 |
| direct `char` int8 read | 0.79 | 896 |
| vectorized `float2x4` input staging | 0.83 | 896 |
| roll `ik`, unroll inner loops | 0.90 | 896 |
| `packed_char4` vectorized dequant | **0.93** | 896 |

The generated kernel ends up bit-identical to the handwritten one, at *higher*
occupancy (896 vs 832). It does not issue fewer dequant loads, though an earlier
revision of this file claimed 4 `packed_char4` against 16 scalar `int8`: AIR
shows 16 scalar `load i8` per k-tile in both. What `packed_char4` removes is the
shift/mask/sign-extend arithmetic around each load, not the loads.

The residual ~7% turned out **not** to be the `CoopMat` abstraction, which an
earlier revision of this file claimed. Native AGX disassembly shows the kernel
body makes no `linalg_*` calls; the cost is address arithmetic. The native
compiler folds load displacements into the load immediate for the handwritten
kernel and not for the generated one:

| kernel | `threadgroup_load`s | base regs | distinct immediates | MMAs |
|--------|---------------------|-----------|---------------------|------|
| handwritten | 26 | 4 | 21 | 32 |
| generated | 13 | 11 | 2 | 16 |
| generated, rewritten | 25 | 4 | 21 | 32 |

The MMA column is there because the first two rows are not like for like: the
generated loop is unrolled half as far, so it has half the loads to place, and a
base-register count only means anything at a matched unroll factor. The third
row is the comparable one.

Three properties of the emitted MSL are each necessary and only jointly
sufficient for that folding: scratch as a `[[threadgroup(0)]]` parameter,
pointer-typed k-loop induction, and `#pragma unroll` on the k-loop. Slang can
express none of them, so `cera/build_support/msl_postpass.rs` rewrites the
generated MSL in `build.rs` (OUT_DIR only, so the committed artifact stays
byte-identical for the drift check below). That takes the large shapes to
**~0.98x**. See that file for the ablations and the upstream issues.

Two levers were measured and **rejected** (recorded here and in the `.slang`
header so they are not retried). The first is the
`simdgroup_barrier(mem_flags::mem_none)` operand-load hint, expressible via
`__target_intrinsic` but neutral on Apple silicon in both the dequant-bound and
scheduling-bound regimes.

The second is unrolling `ik` **in Slang**: it gives every iteration's
operands distinct registers, dropping occupancy 896 to 704, and measures 0.89x.
That is a different lever from the `#pragma unroll` the post-pass adds, which
attaches AIR loop metadata rather than unrolling the source.

## The migration verdict

A generated simdgroup GEMM reaches ~0.93x bit-identical on its own, and ~0.98x
with the MSL post-pass, but only because the hot-path *memory access* drops to
`__target_intrinsic` raw-MSL strings (`load_i8x4_direct`, `load_f16_direct`,
`stage_input_f2x4`). The compute (tiling, MMA) ports cleanly through `CoopMat`;
the byte-unpacking and vectorized staging do not, because WGSL's buffer and
groupshared abstractions cost too much. So "one portable source" holds for the
math and leaks raw Metal at the memory layer. Any port of the remaining
hand-tuned `simdgroup_matrix` GEMMs inherits that shape.

## Working with these shaders

Regenerate the committed outputs after editing any `.slang` (requires slangc
2026.13.1, the version CI byte-compares against):

```sh
just slang
```

Note for `gemm_q8_0.metal`: the committed file is raw slangc output, and so is
what slangc hands `build.rs` on your machine, but neither is what ends up in the
binary. `build_support/msl_postpass.rs` rewrites whichever one it gets into
OUT_DIR. Editing the `.slang` can move the anchors that the pass keys on. It
declines with a `cargo:warning` rather than breaking the build, so the shader
stays correct and quietly loses ~5%; the parity test
`msl::generated_gemm_is_post_processed` fails in that case, which is the signal
to watch rather than the build log.

Correctness (runs on any Metal device; the two `gpu`-gated probe tests need the
`gpu`/wgpu feature instead):

```sh
cargo test -p cera --features metal --test slang_multitarget_parity -- --nocapture
```

`CERA_REQUIRE_METAL=1` turns a missing Metal device into a hard failure rather
than a skip.

Performance (release matters; a debug build measures the harness):

```sh
cargo run -p cera --features metal --release --example slang_gemm_bench
cargo run -p cera --features metal --release --example slang_elementwise_bench
```

Only two benches remain, and that is a consequence of the flip rather than an
omission: a generated-vs-handwritten bench needs a handwritten twin, and outside
`elementwise` and `gemm_q8_0` there is no longer one to compare against. The
softmax, norm, norm2 and conv benches were deleted with the twins they timed.

Every bench prints an agreement check first (a faster arm that disagrees is not a
faster arm), then interleaves the two kernels round by round and reports the
median ratio, because uninterleaved wall-clock timing of microsecond kernels
drifts. `slang_gemm_bench` also prints `maxTotalThreadsPerThreadgroup` for both
kernels: it is the register-pressure proxy (the driver sets it from registers per
thread, so lower means a heavier kernel and lower occupancy) that located the
dequant as the bottleneck.

Treat the bench as a required step per tier, not an optional extra. It is what
caught both of these in the first place: a `pow`/`powr` divergence in `rope` that
was numerically invisible at small positions, and a 0.72x regression in
`conv1d_fused_batch` whose output was bit-identical to the handwritten kernel.
Neither is something a correctness test can see: the `rope` divergence because
both kernels are compared to the CPU rather than to each other, and the 0.72x
because the kernel stayed bit-identical and only got slower. Once a bench has
located a
codegen regression, pin it with a cheap structural assertion so the next one
fails a test rather than waiting for someone to re-run the bench:
`generated_conv_batch_unrolls_its_register_loops` does that for the 0.72x case.

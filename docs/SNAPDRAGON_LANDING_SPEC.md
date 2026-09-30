# Snapdragon Landing Spec: NPU + CPU + GPU

The NPU work lands with fixed CPU and GPU. This spec defines the three
workstreams, their diagnoses, and the measured gates that release the
`feat/hexagon-backend` branch.

## Baseline

Galaxy S25 Ultra, `LFM2-VL-450M-Q4_0.gguf`, `--prompt 512 --decode 128`,
5 passes x 5 runs, interleaved. Full matrix:
`benchmarks/BASELINE.md`, "Galaxy S25 Ultra" section.

| Test | cera | llama | Gap |
|---|---|---:|---|
| NPU pp | 8685 | 8103 | cera 1.07x |
| NPU tg | 124.6 | 158.2 | llama 1.27x |
| GPU pp | 2117 | 3307 | llama 1.56x |
| GPU tg | 78.1 | 134.4 | llama 1.72x |
| CPU pp, matched 6c | 918 | 860 | tied |
| CPU tg, matched 6c | 92.4 | 182.5 | llama 1.98x |
| CPU pp, best | 918 | 1642 | llama 1.79x |
| CPU tg, best | 92.4 | 257.1 | llama 2.78x |

## WS1: CPU decode — fused-kernel `target_feature` (P0, DONE)

**Diagnosis (verified, corrected twice).** The first theory (llama
`smmla` vs cera `vdot`) died to a kernel probe: cera's plain Q4_0
GEMV does 25.5 GB/s isolated — faster than llama end-to-end — while
end-to-end decode implied 5 GB/s. The second theory (register spills
in the 2-row fused loop) died the same way: a 1-row rewrite measured
identically slow. The true cause came from `simpleperf`: 40% of
decode cycles sat in a 6-instruction `ldr,ldr,ldr,sdot,str,ret`
helper — the `vdotq_s32` intrinsic itself, emitted as a real outlined
function. Five fused kernels call `vdotq_s32` without carrying
`#[target_feature(enable = "neon,dotprod")]`, so LLVM cannot inline
the intrinsic and emits a call per dot product (plus spill/reload of
every live vector around each call). Affected: `gemv_q4_0_fused2`,
`gemv_q4_0_concat3`, `gemv_q4_0_gate_up_swiglu`, `gemv_q4k_swiglu`,
`gemv_q5k_swiglu`. Audit method, reusable: cross-build the lib asm
for `aarch64-linux-android` and grep for `bl ...vdotq_s32` /
`bl ...vmmlaq` — must be zero (i8mm side was already clean).

**Fix (landed in branch).**

1. Added the missing `#[target_feature(enable = "neon,dotprod")]`
   to all five fused kernels (all five already gate on
   `cpu_features().tier >= NeonDotprod` at runtime, so this is
   sound on non-dotprod CPUs). Codegen audit now shows zero
   outlined dotprod/i8mm calls.
2. Restructured the Q4_0 `gate_up_swiglu` and `fused2` main loops
   from 2-row to 1-row interleave (the 2-row shape holds 8
   accumulators + 16 nibble vectors and would spill once inlined;
   the 1-row shape keeps ~23 live).
3. Added the two missing parity tests
   (`q4_0_gate_up_swiglu_matches_separate_gemv`,
   `q4_0_fused2_matches_separate_gemv`; Q4K/Q5K already had theirs).
   All run natively on M1 (dotprod), not skipped.
4. Kept `cera/examples/simd_gemv_bench.rs`: the kernel-vs-framework
   probe that split this diagnosis (plain vs fused vs unfused arms
   at LM-head and FFN shapes). Future CPU kernel work starts here.

**Result (S25U, same quick-check conditions both sides).**

| Test | before | after | llama |
|---|---:|---:|---|
| decode 1t (prime) | 24.0 | 86.0 | 83.3 |
| decode 6t (perf x6) | 92.4 | 208.5 | 182.5 |
| fused kernel | 2.4 GB/s | 21.4 GB/s | — |

Gate was >= 165 tok/s matched-6c; measured 208.5 — cera now leads
llama by 14% there and ties single-thread. Q4_K E2E (350M,
exercises the q4k fix): 161 tok/s, sane, no crash.

**Still open (not landing blockers).** An i8mm `smmla` GEMV would
still add headroom on top (llama's nrc=2 path exists; ours doesn't)
— file as follow-up now that the 11x bug is out of the way. The Mac
CPU-decode gap is NOT explained by this fix (M1 builds with dotprod
global, so nothing outlined there) and keeps its own profile-first
follow-up. The `silu_mul_inplace` rayon path anti-scales badly at 6
threads in the probe (2.3ms for 4608 elems); no H2H decode path uses
it (all fused), but non-SwiGLU models would — follow-up.

## WS2: CPU prefill scaling (P1, PARKED — see WS2_REPACKED_I8MM_SPEC.md)

**What shipped.** Q4_0 smmla repack + batched prefill GEMM (t1 376→523,
t8 1205→1724 on LFM2.5-350M pp512), then GEMM-form flash attention
(4-query lane-QK + grouped AV + bit-exact vector exp, attn −34%:
t1→533, t8→~1722). Full story in `WS2_REPACKED_I8MM_SPEC.md`.

**What was tried and reverted (all with on-device A/B + counters).**
B-interleave by 4 (no GEMM gain, +8–20ms pack); 2D GEMM partitioning
(slower everywhere — panels don't stick in L1, W re-streams; t8 IPC
4.77→2.94 is L1D-stall-bound, not B-bandwidth); fused-SiLU vectorization
(t1 exact tie — ALU hides under stalls). The original "pool loses it"
theory died: the pool was fine; per-thread kernel throughput (smmla)
was the gap, then memory stalls.

**Target (mixed, parked).** The original ≥1315 tok/s (llama-450M-t8 minus
20%) is MET on 350M (~1722; 450M matrix re-run pending in gate 4) — but
the WS2b stretch goals (t1≥550, t8≥2000, llama parity 611/2279 on 350M)
are open research (LDNP-W + panels, cross-pass fusion). Parked to land
NPU/CPU/GPU; revisit with per-symbol profiles. No matched-width
regressions (decode lead held).

## WS3: GPU Adreno quantized GEMV + host overhead (P0)

**Diagnosis (verified kernel-bound, host side open).**
`CERA_GPU_PROFILE=1` on Adreno 830: stable decode ~7.6 ms GPU time,
`ffn` 53.6% + `lm_head` 19.3% + `conv` 14% — the quantized GEMVs are
~73% of GPU time. Effective 16 GB/s vs llama OpenCL's 28 GB/s
end-to-end. The Slang kernels (`gemv_q4_0_fast.slang`) use scalar
`u32` word loads with shift/branch nibble extraction and no packed
integer-dot anywhere in the shader tree; llama's hand-tuned OpenCL
does not pay that tax. Separately, GPU time accounts for only ~60%
of decode wall (7.6 vs 12.8 ms) — the rest is host-side (encode,
submit, poll) and/or KV-depth attention the short profile didn't
cover. Dispatch count is already minimal (1.0 submits/token), so this
is not a batching problem.

**Work.**

1. Re-profile at matrix conditions (128-token prompt, 128 decode) to
   split the wall gap into kernel time vs host time with authority.
   If host time dominates, audit encode/submit/poll before touching
   kernels.
2. Kernel rework in Slang (single source -> WGSL/Metal): vectorized
   weight loads, packed int8 dot products where the target exposes
   them (`dot4I8Packed` on WGSL; verify Adreno codegen, don't assume),
   occupancy check at FFN vs LM-head shapes (cf. the occupancy lesson
   in BASELINE's GPU decode profile — what wins at one shape can
   regress another; A/B per shape).
3. Prefill rides along: `gemm_stream_q4_0` was 92.6% of prefill GPU
   time, so the same dequant throughput work applies. One kernel
   family, both phases.

**Target.** GPU decode within 25% of llama OpenCL (>= 108 tok/s) and
GPU prefill within 25% (>= 2650 tok/s). Wider tolerance than CPU:
Adreno-vs-slang codegen is less predictable than NEON intrinsics.

## Validation and landing gates

1. **Parity.** New kernels exact-match existing references
   (dotprod kernel + scalar oracle); full `cargo test` + the
   hexagon 21 green.
2. **Determinism.** The on-device NPU determinism matrix
   (see `ANDROID_NPU_PACKAGING.md` gate 4) re-run green — CPU/GPU
   kernel work must not re-phase shared paths.
3. **No regressions.** Re-run the Tensor G4 CPU section and the Mac
   CPU/GPU rows that this work could move; any regression vs BASELINE
   blocks landing. (The Mac CPU-decode gap is a separate, open item —
   M1 has no i8mm, so WS1 does not address it; do not conflate.)
4. **H2H re-run.** Same command, same device, same model:
   `scripts/bench_android.sh --model LFM2-VL-450M-Q4_0.gguf
   --llama-bench /data/local/tmp/llama-bench --prompt 512
   --decode 128`. Gates: CPU decode <= 1.1x gap, GPU decode <= 1.25x,
   CPU prefill <= 1.2x, GPU prefill <= 1.25x, NPU cells not regressed
   (prefill lead held, decode gap not widened).
5. **Hygiene.** `cargo fmt --check`, zero clippy lints workspace-wide
   (+ `--features hexagon`), BASELINE + this spec updated with the
   re-run numbers.

## Out of scope

Mac CPU-decode gap (separate cause, needs its own profile); MoE and
Mamba2-family NPU coverage; VL vision encoder on NPU; stock-APK story
(LiteRT + delegate, separate project); v68/v69 skels + EULA + v75/v73
smoke (tracked NPU open items, not landing blockers).

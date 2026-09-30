# WS2: Repacked-i8mm Q4_0 GEMM — Diagnosis and Fix Spec

## Implementation status (2026-09-23)

**Landed in worktree (tested, profiled):**

- Smmla repack + 3 kernels (colmajor / rowmajor / fused gate_up) behind the
  `NeonI8mm` tier, tier-gated single repack at load (no double memory).
  Parity vs vdot pinned by 4 on-device tests (`smmla_*`, i8mm-enforced).
- `CERA_PROFILE_PREFILL` extended: GEMM phases now exclude B-quant; new
  `quant` (all in-layer B-quants: **9.7ms**, never the issue) and
  `attn_scores` phases.
- Attention Q/output transposes parallelized (+5.5% t8).
- Flash query-blocking (QB=32): neutral at 512 tokens (-1.5%, L2-resident
  either way), +52% on 2048-ctx late chunks / +14% overall at n=2048
  (measured QB1 vs QB32 A/B). Scaffolding the GEMM-form flash below.

**Measured (same file, pp512, CPU):** cera t1 376→**523** (llama 611),
t8 1205→**1724** (llama 2279). Spec acceptance (t1≥550, t8≥2000) NOT met —
the remaining gap is two more kernels + t8 structure (see follow-ups).

**Follow-ups (spec'd, not implemented):**

1. **B-interleave by 4 tokens** — IMPLEMENTED, MEASURED, REVERTED. The
   chunk-major packer + single-`vld1q_s8` kernels were built (per-block
   primitives, mixed-variant guard, all 6 prefill sites) and showed **no GEMM
   gain** (same bytes, memory-bound) while the pack cost **+20ms scattered
   (9.7→30ms)** or **+8ms as quantize+transpose (→18ms)** on the pp512 t8
   quant phase. Reverted to columnar B + `vld1_s8`+combine RHS (quant back to
   ~8.5ms); the "only structural extra" is instruction-count in a
   memory-bound kernel, i.e. not worth any pack cost. Do not retry without a
   pack that costs <1ms.
4. **Fused-SiLU exp vectorization** — IMPLEMENTED, MEASURED ZERO, REVERTED.
   Bit-exact 4-wide NEON SiLU (shared `silu_mul_lane4`, fused loop +
   `silu_inplace`/`silu_mul_inplace`) showed t1 TIE (443.6→443.6ms gate)
   and t8 ≤0 in interleaved A/B. Lesson, same as flash-exp (+1ms only):
   this workload is L1D-stall-bound (t8 IPC 4.77→2.94), so ALU work hides
   under stalls — only moved bytes (smmla repack, flash V-loads) win.
   Reverted; kept absolute-literal pins in `test_silu_mul_inplace`.

**WS2b CPU close (this round):** t1 523→**533**, t8 1724→**~1720–1770**
(±3% thermal band; flash is the only keeper: attn −34%). Spec acceptance
(t1≥550, t8≥2000) NOT met. Remaining t8 gap is NOT B-bandwidth (2D
negative with counters); suspects are L1-stall-bound kernels needing
LDNP-W + panel-sticking (combo, unproven, asm risk) or fusion across
passes. Parked: land NPU/CPU/GPU first, revisit CPU prefill with
per-symbol profiles.

2. **GEMM-form flash** — IMPLEMENTED, MEASURED (+2.5% t8, +1.5% t1):
   4-query groups sharing one K/V stream; Q transposed once per block, K
   streamed through on-the-fly 4-wide transposes into `vfmaq_laneq` outer
   products (zero `vaddvq`); V loads once per 4 queries; plus bit-exact
   4-wide NEON `ggml_expf` (scalar exp turned out to be ~4% of attention,
   not ~33% — the AV accumulator read-modify-write traffic dominates, so
   the vector exp bought only ~1ms; a host probe confirmed the compiler
   does NOT auto-vectorize the scalar loop, 4.5× in isolation). Interleaved
   old/new A/B: attn_scores −32–35% (41→27ms pp512 t8). Production pinned:
   t1 523→531, t8 1724→1767. Follow-up: register-tile AV over head_dim
   (acc traffic 32→1 accesses/score, est. −12% attention).
3. **2D GEMM partitioning** — IMPLEMENTED, MEASURED, REVERTED. A (row-group
   × token-panel) grid over all three smmla wrappers (shared tiler +
   `par_range_prefill` dispatch, NP≈n/4, `CERA_GEMM_2D=0` kill switch for
   A/B) was SLOWER than 1D everywhere: gate −6–9%, down −9–21%. NP sweep
   (32/64/128) favored smaller panels but never beat 1D; SR_G=1 (maximal
   L1-residence) tied 1D. Counters explain it: t8 IPC collapses 4.77→2.94
   with +40% L1D refills at flat L2 refills, and 2D *raised* L1D refills
   146M→159M (panels don't stick — the W stream evicts them — while W
   re-streams ×panels). B was already L2-cheap; 1D (minimal W) is optimal
   on the traffic axes. The t8 scaling gap is NOT B-bandwidth — still open
   (per-symbol profiles + LDNP-W next if revisited). Do not retry panels
   without LDNP (non-temporal) W loads, which need inline asm (no
   `std::arch` LDNP intrinsic).

## TL;DR (original diagnosis)

The CPU prefill gap is **not thread scaling**. Both engines scale ~3.3–3.7x
t1→t8 to the same device wall. The gap is **per-thread GEMM throughput**:
llama.cpp runs repacked-Q4_0 + `smmla` (i8mm); cera runs repacked-Q4_0 +
`vdot`. Fix: port ggml's `ggml_gemm_q4_0_8x8_q8_0` dataflow (folded-nibble
W repack + row-major Q8 B + `smmla` 4-token × 8-channel tiles) behind the
`NeonI8mm` tier.

## Evidence (same session, same file, same device)

File: `/data/local/tmp/LFM2-350M-Q4_K_M.gguf` — **mislabeled, contains Q4_0**
tensors (verified via `cera inspect`). All cells below: pp512, CPU (`-ngl 0`).

| engine | t1 (tok/s) | t8 (tok/s) | t1→t8 |
|---|---|---|---|
| llama.cpp (09-22 H2H libs) | 597 | 2196 | 3.68x |
| cera (this worktree) | 376 | 1205 | 3.21x |
| **gap** | **1.59x** | **1.82x** | scaling ~equal |

- llama on the user's named file (`LFM2.5-VL-450M-Q4_0.gguf`): 2176 tok/s @t8 —
  same backbone, same story.
- The earlier "tie at t1 (385=385)" was across mismatched files; it does not
  reproduce same-file. Do not cite it.
- Phase profile (`CERA_PROFILE_PREFILL`): every GEMM phase scales uniformly
  ~3.3x t1→t8 (ffn_gate_up 571→170ms, ffn_down 306→90ms, attn_qkv 56→17ms,
  conv_in 128→40ms). Uniformity ⇒ shared substrate (the kernel), not one op.
- n-sweep (64→512): scaling flat at 3.3–3.5x ⇒ not B-matrix cache capacity.
- simpleperf @t8: 7 pool workers equally hot (~11.5% each), no barrier
  blocking, ~13% spin-wait (serial gaps). Threads are fed; cycles are stalls.
- Tier force (`CERA_CPU_TIER`) on the **naive** (non-repacked) Q4_0 GEMM:
  base 119.7 = dotprod 103.3 = i8mm 115.8 GFLOPS ⇒ the naive kernel is
  nibble-unpack-bound; the dot instruction only matters once weights are
  repacked (repacked-vdot does 271 GFLOPS @t1). The remaining 271→~410 gap is
  the dot + micropanel.

## llama.cpp truth (host checkout `~/development/llama.cpp`)

- `ggml/src/ggml-cpu/arch/arm/repack.cpp`: `ggml_gemm_q4_0_8x8_q8_0` —
  hand-written asm, `smmla` (`__ARM_FEATURE_MATMUL_INT8`), 8-channel tiles.
- `ggml/src/ggml-cpu/repack.h`: `block_q4_0x8` = 8×fp16 scales + 128B nibbles
  (still packed; kernel unpacks on the fly).
- Reference `ggml_gemm_q4_0_8x8_q8_0_generic` (repack.cpp:1768) semantics:
  - W panel per 32-k: `[k-sub 0..1][row 0..7][elem 0..7]`, each byte folding
    k-elems `(k*8+i)` (low nibble) and `(16+k*8+i)` (high nibble).
  - B: plain row-major Q8_0 blocks, 4 consecutive token-rows, **no interleave**.
  - Tile: 4 tokens × 8 channels over full k; output row-major.

## cera deltas to close

1. **W repack**: cera `Repacked::Q40` = 8-row groups, 256B *unpacked* i8 +
   f32 scales. ggml = 128B *folded nibbles* + fp16 scales (half the traffic,
   smmla-shaped). Add a smmla repack variant; keep `Q40` for the vdot path
   (non-i8mm fallback).
2. **B layout**: cera `quantize_rows` emits *column*-packed B (token j at
   `j*k`), which the vdot kernel consumes. The smmla kernel needs *row*-major
   Q8_0 (4-token groups). B is per-dispatch scratch — add a row-major packing
   (or transpose) used only by the smmla path.
3. **Kernel**: new `#[target_feature(enable = "neon,i8mm")]` kernel over the
   smmla repack, 4-token × 8-channel tiles, `core::arch::aarch64::vmmlaq_s32`
   (already used by the naive i8mm kernels — intrinsics, no asm needed).
   Serves all repacked-Q40 prefill GEMMs (gate_up fused, down, attn, conv —
   all funnel through `gemm_preq_repacked_q4_0*_dispatch`).
4. **Dispatch**: use smmla repack+kernel when `cpu_features().tier ==
   NeonI8mm`; else existing repacked-vdot. Repack both layouts at load (or
   lazily per tier) — memory cost ~1.3MB extra per 2.6MB Q4_0 weight.

## Acceptance

- Parity: smmla-GEMM vs repacked-vdot and vs naive-i8mm on randomized +
   adversarial (full-range quants/scales) inputs; tolerances per the existing
   four i8mm GEMM tests. Gate under `require_i8mm_kernel_or_skip`.
- Perf (SD 8 Elite, pp512, Q4_0 file above): prefill t1 376→≥550,
   t8 1205→≥2000 (llama parity band ±10%: 597/2196). No decode regression
   (decode does not use this kernel, but the suite must stay green).
- `just ci` green (fmt, clippy, host tests); new tests run in the `simd-i8mm`
   CI leg.

## Non-goals

- Q4_K / Q8_0 / Q6_K smmla repacks (same pattern, separate work).
- GEMV i8mm (covered by the landing spec's GEMV stream; decode already leads).
- Changing the vdot repack layout or the naive kernels.

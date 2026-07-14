# GPU root-cause: two rounds of being wrong

This document has been wrong twice. Both times the error had the same shape —
trusting a number without checking what it actually counted — so the history is
kept rather than quietly overwritten.

## Round 1 — the code-reading inferences (wrong)

From `prefill (12) ≈ decode (11.2)` on Adreno I inferred the GPU was
**latency-bound on per-token round-trips**, and named three root causes:

1. A **logits readback every token** forces a pipeline flush → T5.
2. **Many small kernel dispatches** per layer × 16 layers → T6.
3. **Prefill isn't batched** — it loops the per-token GEMV → T8.

All three were inferences from reading code. None were measured.

## Round 2 — the counters (also wrong, in the opposite direction)

I added GPU I/O counters and read **1.0 submits/token, 1.0 readbacks/token,
0.045 prefill submits/prompt-token**, identical on Mac and Adreno. I concluded all
three inferences were false, closed T6 as "already done", and declared the
kernels — not the plumbing — to be the problem.

**The counters were lying.** They incremented only inside
`GpuContext::submit_encoder`, but `GpuLfm2Model` submitted its work through direct
`self.ctx.queue.submit(...)` calls that bypassed the choke point entirely. The
counter saw only the final 4-byte logits download. The tell was sitting in the
data and I walked past it: prefill's "23 submits" was *exactly* the readback count.

Caught in review by the `github-actions` bot on #255. It was right.

## What the counters say now that every submit is counted

`GpuLfm2Model` and `WgpuVitOps` now route every submit through `submit_encoder`,
so the count is real. Prompt 512, greedy decode:

| | LFM2-VL-450M **Q4_0** (Mac) | LFM2.5-350M **Q4_K_M** (Mac) | LFM2.5-350M **Q4_K_M** (Adreno) |
|---|---:|---:|---:|
| decode submits / token | **19.0** | **19.0** | **19.0** |
| decode readbacks / token | 1.0 | 1.0 | 1.0 |
| decode readback **bytes** / token | 4 | 4 | 4 |
| **prefill submits** (512-tok prompt) | **25** | **8728** | **8728** |
| decode tok/s | 22.6 | 29.9 | 12.8 |
| prefill tok/s | 128 | 43 | 13 |

## Each claim, re-judged

1. **"Per-token logits readback" — still FALSE.** The readback really is
   **4 bytes/token**: greedy decode runs argmax in a WGSL shader and reads back
   only the token id. This is the one round-2 conclusion that survives.

2. **"Many dispatches per token" — TRUE after all.** Decode is **19 submits per
   token** — one per layer (16) plus argmax and friends — not 1.0. The forward pass
   is *not* recorded into a single command buffer, despite the `wgpu.rs` header
   comment that claimed it was. **T6 reopens**; closing it was an artifact of the
   broken counter.

   > **Postscript (2026-07-14): the count is real, but it is not a defect, and T6 is
   > closed WONTFIX.** Merging the forward pass into one command buffer was built and
   > measured: submits fall 19 → 3 and decode gets **~30% slower on both platforms**
   > (Mac 62.0 → 45.3, Adreno 12.4 → 8.6 tok/s). Decode is GPU-execution-bound —
   > ~15–18 ms of GPU work per token vs ~1.6–2.4 ms of CPU encode — so submitting each
   > layer as it is encoded lets the GPU start layer *i* while the CPU builds layer
   > *i+1*. Batching them idles the GPU through the whole encode phase. **A submit
   > count is a cost proxy, not a cost.** See PR #259.

3. **"Prefill isn't batched" — TRUE, but for a reason nobody guessed.** A 512-token
   prefill issues **8728 submits** (~17 per prompt-token): it is running the
   per-token path 512 times.

   But this is **not an Adreno property** — the *same model on Mac* issues the same
   8728. The batched path is gated on `all_matmul_weights_batched_supported()`,
   which admits only `Q4_0 | Q8_0 | Q4KM`. A `Q4_K_M` file is **not** uniformly
   Q4_K: this one carries **11 Q6_K tensors** (llama.cpp promotes certain tensors to
   Q6_K in the `_M` mix). One unsupported dtype makes the predicate return `false`,
   and prefill **silently falls back** to the sequential loop.

## The confound that produced round 2's "real story"

Round 2 concluded: "identical round-trip structure, yet Adreno is 13x slower at
prefill → the Adreno *kernels* are slow." That comparison was
**Q4_0-on-Mac vs Q4_K_M-on-Adreno** — different quants, therefore different code
paths: Mac took the batched GEMM, Adreno took the per-token fallback. It was never
a platform comparison.

Held to the **same model**, the honest gaps are:

- prefill **43 (Mac) vs 13 (Adreno) — 3.3x**, not 13x
- decode **29.9 (Mac) vs 12.8 (Adreno) — 2.3x**

There *is* a real Adreno kernel gap, but it is ~3x, not ~13x, and the headline
prefill disaster was mostly a **quantization gate**, not silicon.

## What changed in the plan

- **T6 (one submit per token) — REOPENED, now a top GPU lever.** 19 submits per
  token, one per layer, on every platform. Nothing was ever batched.
  **→ Superseded 2026-07-14: T6 is CLOSED WONTFIX.** Built it; it is a ~30%
  regression on Mac *and* Adreno (see the postscript above). The submits are cheap
  and they buy GPU/CPU overlap. The decode lever is GPU work per token — and **T5b
  has since measured which work**: the quantized GEMV loads, not the submit count and
  not the weight format. See the T5b entry below; the ~15–18 ms/token that was
  unattributed when this bullet was written is now broken down in `BASELINE.md`.
- **T8 — REFRAMED** from "make the batched prefill GEMM fast on Adreno" to **"let
  the batched prefill GEMM actually run"**: add a batched **Q6_K** kernel (or
  dequantize the 11 Q6_K tensors at load) so `Q4_K_M` models stop falling off the
  fast path. Platform-independent; it should lift Mac prefill too.
- **The silent fallback is itself a bug.** `all_matmul_weights_batched_supported()`
  returning `false` costs ~340x the submits and says nothing at all. It should at
  minimum log which tensor and dtype knocked it off the batched path.
- **T5 (GPU sampling)** — unchanged. Greedy is already on-GPU; the non-greedy path
  still downloads full vocab logits per token. `bench` runs temp=0, so it never
  hits this.
- **T5b (per-kernel GPU timestamps) — DONE, and it answers the decode question.**
  It needed **no new code**: `CERA_GPU_PROFILE=1` and the whole `GpuProfiler` already
  existed and had never been run. The 15–18 ms/token is now attributed (table in
  `BASELINE.md`). Headline: **the quantized decode GEMVs sustain ~25 GB/s while the
  f16 GEMV sustains 106 GB/s on the same GPU** — a ~4x gap in achieved bandwidth,
  with FFN alone at 51% of GPU time.
- **The decode lever is the quantized GEMV load pattern.** Decode is memory-bound
  (proved by A/B: the same model at Q8_0 moves 1.89x the FFN bytes and takes 2.10x
  the time), but the quantized kernels only reach 6.7% of the M1 Max's 400 GB/s. They
  read weights as scalar `u32` loads with shift/branch byte extraction; `gemv_f16`
  reads aligned vectors and is 4x more efficient. Fix the reads, not the math.
- **T7 (f16 weights) — DEAD AS SCOPED.** At the f16 kernel's own 106 GB/s, f16 FFN
  weights (453 MB) would take ~4.3 ms against Q4_0's 4.49 ms: a wash. Converting
  weight formats cannot help while the quantized kernels are 4x off their achievable
  bandwidth. (f16 *KV* is untouched by this and still open.)

## Lesson

Round 1: I inferred root causes from reading code, and was wrong. Round 2: I
measured — but never validated the instrument — and was wrong again, and *more*
confidently, because now I had numbers.

A counter wired into one code path does not measure the system; it measures that
path. The check that would have caught this takes a minute: confirm the count
scales with something you can predict in advance. Submits should scale with layer
count. They didn't, and I never asked.

And when comparing two platforms, hold the model fixed. Half of "Adreno is
terrible" was Adreno running a different code path than the machine it was being
compared against.

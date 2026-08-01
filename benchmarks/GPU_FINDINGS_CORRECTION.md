# GPU root-cause: five rounds of being wrong

This document records five rounds of getting the GPU story wrong, kept rather
than quietly overwritten. They do not all share one shape:

- **Rounds 1-2 and 5**: trusting a number without checking what it actually
  counted (code-reading inferences, a counter wired into one code path, a
  benchmark that scored kernels by their position in its own table).
- **Round 3**: a *correct* measurement generalised into a cost model for the
  whole system.
- **Round 4**: what finally worked: decompose the budget before optimising a
  part of it.

## Round 1: the code-reading inferences (wrong)

From `prefill (12) ≈ decode (11.2)` on Adreno I inferred the GPU was
**latency-bound on per-token round-trips**, and named three root causes:

1. A **logits readback every token** forces a pipeline flush → T5.
2. **Many small kernel dispatches** per layer × 16 layers → T6.
3. **Prefill isn't batched**: it loops the per-token GEMV → T8.

All three were inferences from reading code. None were measured.

## Round 2: the counters (also wrong, in the opposite direction)

I added GPU I/O counters and read **1.0 submits/token, 1.0 readbacks/token,
0.045 prefill submits/prompt-token**, identical on Mac and Adreno. I concluded all
three inferences were false, closed T6 as "already done", and declared the
kernels, not the plumbing, to be the problem.

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

1. **"Per-token logits readback": still FALSE.** The readback really is
   **4 bytes/token**: greedy decode runs argmax in a WGSL shader and reads back
   only the token id. This is the one round-2 conclusion that survives.

2. **"Many dispatches per token": TRUE after all.** Decode is **19 submits per
   token**, one per layer (16) plus argmax and friends, not 1.0. The forward pass
   is *not* recorded into a single command buffer, despite the `wgpu.rs` header
   comment that claimed it was. **T6 reopens**; closing it was an artifact of the
   broken counter.

   > **Postscript (2026-07-14): the count is real, but it is not a defect, and T6 is
   > closed WONTFIX.** Merging the forward pass into one command buffer was built and
   > measured: submits fall 19 → 3 and decode gets **~30% slower on both platforms**
   > (Mac 62.0 → 45.3, Adreno 12.4 → 8.6 tok/s). Decode is GPU-execution-bound
   > (~15–18 ms of GPU work per token vs ~1.6–2.4 ms of CPU encode), so submitting each
   > layer as it is encoded lets the GPU start layer *i* while the CPU builds layer
   > *i+1*. Batching them idles the GPU through the whole encode phase. **A submit
   > count is a cost proxy, not a cost.** See PR #259.

3. **"Prefill isn't batched": TRUE, but for a reason nobody guessed.** A 512-token
   prefill issues **8728 submits** (~17 per prompt-token): it is running the
   per-token path 512 times.

   But this is **not an Adreno property**: the *same model on Mac* issues the same
   8728. The batched path is gated on `all_matmul_weights_batched_supported()`,
   which admits only `Q4_0 | Q8_0 | Q4KM`. A `Q4_K_M` file is **not** uniformly
   Q4_K: this one carries **11 Q6_K tensors** (llama.cpp promotes certain tensors to
   Q6_K in the `_M` mix). One unsupported dtype makes the predicate return `false`,
   and prefill **silently falls back** to the sequential loop.

## The confound that produced round 2's "real story"

Round 2 concluded: "identical round-trip structure, yet Adreno is 13x slower at
prefill → the Adreno *kernels* are slow." That comparison was
**Q4_0-on-Mac vs Q4_K_M-on-Adreno**, different quants, therefore different code
paths: Mac took the batched GEMM, Adreno took the per-token fallback. It was never
a platform comparison.

Held to the **same model**, the honest gaps are:

- prefill **43 (Mac) vs 13 (Adreno): 3.3x**, not 13x
- decode **29.9 (Mac) vs 12.8 (Adreno): 2.3x**

There *is* a real Adreno kernel gap, but it is ~3x, not ~13x, and the headline
prefill disaster was mostly a **quantization gate**, not silicon.

## What changed in the plan

- **T6 (one submit per token): REOPENED, now a top GPU lever.** 19 submits per
  token, one per layer, on every platform. Nothing was ever batched.
  **→ Superseded 2026-07-14: T6 is CLOSED WONTFIX.** Built it; it is a ~30%
  regression on Mac *and* Adreno (see the postscript above). The submits are cheap
  and they buy GPU/CPU overlap. The decode lever is GPU work per token, and **T5b
  has since measured which work**: the quantized GEMV loads, not the submit count and
  not the weight format. See the T5b entry below; the ~15–18 ms/token that was
  unattributed when this bullet was written is now broken down in `BASELINE.md`.
- **T8: REFRAMED** from "make the batched prefill GEMM fast on Adreno" to **"let
  the batched prefill GEMM actually run"**: add a batched **Q6_K** kernel (or
  dequantize the 11 Q6_K tensors at load) so `Q4_K_M` models stop falling off the
  fast path. Platform-independent; it should lift Mac prefill too.
- **The silent fallback is itself a bug.** `all_matmul_weights_batched_supported()`
  returning `false` costs ~340x the submits and says nothing at all. It should at
  minimum log which tensor and dtype knocked it off the batched path.
- **T5 (GPU sampling)**: unchanged. Greedy is already on-GPU; the non-greedy path
  still downloads full vocab logits per token. `bench` runs temp=0, so it never
  hits this.
- **T5b (per-kernel GPU timestamps): DONE, and it answers the decode question.**
  It needed **no new code**: `CERA_GPU_PROFILE=1` and the whole `GpuProfiler` already
  existed and had never been run. The 15–18 ms/token is now attributed (table in
  `BASELINE.md`). Headline: **the quantized decode GEMVs sustain ~25 GB/s while the
  f16 GEMV sustains 106 GB/s on the same GPU**, a ~4x gap in achieved bandwidth,
  with FFN alone at 51% of GPU time.
- **The decode lever is the quantized GEMV load pattern.** Decode is memory-bound
  (proved by A/B: the same model at Q8_0 moves 1.89x the FFN bytes and takes 2.10x
  the time), but the quantized kernels only reach 6.7% of the M1 Max's 400 GB/s. They
  read weights as scalar `u32` loads with shift/branch byte extraction; `gemv_f16`
  reads aligned vectors and is 4x more efficient. Fix the reads, not the math.

  > **→ Partly superseded 2026-07-28 (see Round 5).** "Fix the reads" held for two
  > kernels, #316's register spills and #321's byte-at-a-time q6_k loads, but it
  > is not a general law. Three q4_k rewrites aimed squarely at its load pattern
  > produced no net win, and the variant that most reduced loads per element was
  > the one that regressed (occupancy, not bandwidth). Treat "read the bytes
  > better" as a per-kernel hypothesis to A/B, not a diagnosis that transfers.
  > The "6.7% of peak" figure itself is **unaffected**: it comes from
  > `CERA_GPU_PROFILE` in-model, not from the microbenchmark, and still stands.
- **T7 (f16 weights): DEAD AS SCOPED.** At the f16 kernel's own 106 GB/s, f16 FFN
  weights (453 MB) would take ~4.3 ms against Q4_0's 4.49 ms: a wash. Converting
  weight formats cannot help while the quantized kernels are 4x off their achievable
  bandwidth. (f16 *KV* is untouched by this and still open.)

## Round 3: extrapolating one successful ablation (wrong)

#318 merged the LFM2 conv block's three compute passes into one and measured
**+17% decode**, five reps out of five. From that I derived a per-pass cost of
~38–50 µs, wrote it into `BASELINE.md` and the devlog, and planned the obvious
follow-up: merge the attention and FFN passes too, for an estimated further ~18%.

The first step landed exactly the predicted pass reduction (42 → 36) and bought
**nothing**: 59.2 → 58.9 tok/s wall-clock, 5626 → 5632 µs of GPU pass time. A
boundary in that path is worth **~6 µs**, not ~38. Worse, when A/B'd separately
the merges were *costing* ~15%: the same branch measured +18% with them and +39%
without. They were implemented, measured, and reverted rather than shipped.

**The error:** a cost-per-unit derived from one successful change is a
description of that change, not a model of the system. This was the second time:
#316 reported "~0.45 ms fixed cost per dispatch" from a straight-line fit through
two points of a curve, and that was wrong too.

## Round 4: what the time was actually going to (right, eventually)

Having stopped extrapolating, the next move was to decompose a token instead of
theorising about it. On an 11 ms decode step:

| segment | time | |
|---|---:|---|
| layer-loop CPU encode | 1.30 ms | not the bottleneck |
| tail, ending in `submit_and_wait` | 6.44 ms | real GPU execution |
| **argmax's own blocking submit** | **1.30 ms** | for a 0.13 ms kernel |
| **4-byte result readback** | **1.34 ms** | its own second submit |

Two of the three GPU round trips per token were carrying essentially nothing.
Folding both into the output projection's submission took decode **60.2 → 111.5
tok/s (~1.85×)**, more than every kernel and pass optimisation before it
combined, from a change that deletes code (#319).

One near-miss worth recording: probing the readback showed *submit 8 µs, map+poll
1.52 ms*, which reads like "the cost is the map, so folding the copy cannot
help." That is wrong: the poll waits for **that submission** to execute and
signal. Acting on the first reading would have skipped a +75% fix.

## Round 5: the microbenchmark scored kernels by table position (wrong)

`cera/examples/wgpu_gemv_bench` reported a kernel's bandwidth as a function of
**where it sat in the kernel x shape table**, not only of the kernel. `q6_k` at
the ffn gate/up shape reads **0.0305 ms running last and 0.0952 ms running
first**; reversing the kernel order moves the penalty to whatever is now first.

`q4_k` was listed first, so it wore that penalty in every table this bench ever
printed. Two published conclusions came out of that and both were false:

| published claim | what it actually is |
|---|---|
| an "FFN per-dispatch floor" no kernel amortises | no such floor; it came from a q4_k m-sweep taken under the bug |
| q4_k is ~3x q4_0's time at identical bytes/element | they are close at every FFN shape under every measurement regime tried; the 3x is not there |

Fixed by running the whole table twice and reporting only the second round. A
750 ms global clock-ramp warmup was tried first and measurably does **not** fix
it, so the mechanism is not clock ramp and is still unidentified; a residual of
up to ~19% remains on one kernel per run, so differences under ~20% from this
harness are not meaningful.

This is the **second of three** instrument defects found in `wgpu_gemv_bench`.
#321 fixed the first (one compute pass per iteration, worth up to 2.1x at the
LM-head shape: 53 -> 113 GB/s). The third surfaced while writing this entry:
every measurement also carried one blocking submit + `poll(Wait)`, a ~1.0-1.5 ms
round trip, amortised over `ITERS` and never subtracted, so a cell read
`T + C/ITERS`. `q6_k` at ffn gate/up read 46.0 / 63.6 / 79.3 GB/s at
ITERS = 50 / 200 / 1000; the fit gives C ~ 1.16 ms, one round trip. Now removed by
two-point timing, at the cost of higher variance, which is why no replacement
GEMV table is published.

None of the three was caught by reading the harness. The first two surfaced when a
result refused to behave: here, a q4_k rewrite that failed to move the number it
targeted; the third when the `ITERS` sweep was finally run while writing this
entry. Three q4_k kernel variants were built and A/B'd off the bad premise before
the premise itself was checked; none shipped.

The checks that would have caught them are two lines of process: **permute the order
and require the numbers to agree, and sweep `ITERS` and require the same.**

## Lesson

Round 1: I inferred root causes from reading code, and was wrong. Round 2: I
measured, but never validated the instrument, and was wrong again, and *more*
confidently, because now I had numbers. Round 3: I measured correctly, then
generalised a single result into a cost model and spent a branch on it. Round 4
worked because it decomposed the thing being optimised before optimising it.

The recurring failure is not bad measurement; it is **reasoning past the
measurement**: one number, extended into a story about the system. The habit that
catches it is cheap: before optimising a component, account for the whole budget
and check the component is actually in it.

A counter wired into one code path does not measure the system; it measures that
path. The check that would have caught this takes a minute: confirm the count
scales with something you can predict in advance. Submits should scale with layer
count. They didn't, and I never asked.

And when comparing two platforms, hold the model fixed. Half of "Adreno is
terrible" was Adreno running a different code path than the machine it was being
compared against.

Round 5 is round 2 again in a different costume: an unvalidated instrument, and
numbers confident enough to plan a branch around. The instrument-level check is
the cheap one and it keeps being the one skipped: for a counter, confirm the
count scales with something predictable; for a benchmark, confirm a measurement
does not depend on the order things were measured in.

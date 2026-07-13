# GPU root-cause: my earlier analysis was wrong

T0 (the profiling harness) did its job immediately: it **invalidated two of the
GPU tasks I had planned**, before we spent any effort on them.

## What I claimed (in ANDROID_BENCHMARK_RESULTS.md)

I inferred from `prefill (12) ≈ decode (11.2)` that the GPU was **latency-bound
on per-token round-trips**, and named as root causes:

1. "A **logits readback to the CPU every token** forces a pipeline flush" → task T5.
2. "**Many small kernel dispatches** per layer x 16 layers swamp the compute" → task T6.
3. "Prefill **isn't batched** — it loops the per-token GEMV 512 times" → task T8.

All three were inferences from reading the code. None were measured.

## What the counters actually measure

I added GPU I/O counters (`cera bench --gpu-io`) that count every `queue.submit`
and every GPU→CPU readback. Results, LFM2.5-350M, prompt 512:

| | Mac (wgpu) | Android (Adreno) |
|---|---:|---:|
| decode submits / token | 1.0 | **1.0** |
| decode readbacks / token | 1.0 | **1.0** |
| decode readback **bytes** / token | 4 | **4** |
| prefill submits / prompt-token | 0.045 | **0.045** |
| **decode tok/s** | 54 | **13.2** |
| **prefill tok/s** | 189 (3.5x decode) | **14 (≈ decode)** |

## What that means — each claim, judged

1. **"Per-token logits readback" — FALSE.** The readback is **4 bytes/token**, not
   `vocab_size * 4`. Greedy decode already runs **argmax in a WGSL shader**
   (`forward_greedy_inner` + the `argmax_f32` kernel) and reads back only the
   token id. GPU sampling already exists. → **T6 closed; T5 re-scoped.**

2. **"Many dispatches per token" — FALSE.** Measured **1.0 submits per token**. The
   whole forward pass is already recorded into a single command encoder per token,
   exactly as the `wgpu.rs` header comment says. → **T6 closed (nothing to batch).**

3. **"Prefill isn't batched" — FALSE.** Prefill issues **0.045 submits/prompt-token**
   (~23 submits for a 512-token prefill) — and *identically on both platforms*. It
   IS batched. → **T8 re-scoped.**

## The real story

**The round-trip structure is identical on Mac and Adreno, and already minimal.
Yet Adreno is 4x slower at decode and 13x slower at prefill.** Round-trips are
therefore *not* the bottleneck — **the GPU kernels themselves are slow on Adreno.**

The sharpest form of this: on Adreno, prefill is batched (proven by the submit
count) but **batching buys it nothing** — prefill (14) barely exceeds decode
(13.2), whereas on Mac the same batched code gets 3.5x (189 vs 54). The batched
GEMM *runs*; it just gets no parallel throughput out of Adreno.

So the residual suspects are the ones about **the kernels**, not the plumbing:
- f32 weights + f32 KV → bandwidth (T7)
- the prefill GEMM's occupancy / tile shape on Adreno (T8)
- Mac-tuned workgroup sizes vs Adreno's 64/128-lane wavefronts (T9)

## What changed in the plan

- **T6 (one submit per token) — CLOSED.** Already true (1.0 submits/token).
- **T5 (GPU sampling) — RE-SCOPED + deprioritized.** Greedy is already on-GPU. The
  real remaining gap: the **non-greedy** path (`forward_inner`) still downloads
  the full vocab logits per token for temperature/top-k/top-p. `bench` uses
  temp=0 so it never hits this — which is exactly why the benchmark didn't show
  it. Worth fixing for real sampled generation, but it does **not** explain 13 tok/s.
- **T8 — RE-SCOPED** from "add batching" to "make the already-batched prefill GEMM
  actually fast on Adreno."
- **T5b — NEW, and now the #1 GPU task:** per-kernel GPU timestamp profiling on
  Adreno, to find *which* kernel eats the time. T7/T8/T9 now depend on it —
  without it, all three are guesses.

## Lesson

Every one of my three GPU root causes was a plausible, code-reading-based
inference — and all three were wrong. The counters took ~1 hour to add and would
have saved days of optimizing dispatch batching and GPU sampling that were
**already done**. Measure before optimizing; "prefill ≈ decode" had a completely
different cause than the obvious one.

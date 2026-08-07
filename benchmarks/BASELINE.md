# Perf baseline

The reference every perf task (T1-T10) diffs against. Re-measure with the same
commands before claiming a delta: a throughput number without a matching
profile/counter movement is not evidence.

**Primary model:** LFM2.5-350M-Q4_K_M (the most common real-world quant); the Mac
section adds Llama-3.2-1B Q4_0 to cover the dense-transformer path.
**Settings:** prefill 512, decode 128, `--no-cache`, medians (p50). Run counts and
the decode-run prompt depth differ per section and are stated there; prefill and
decode are always measured as separate runs (see the traps).

**Sections carry their own provenance; they were not all measured at once:**

| section | measured at | device |
|---|---|---|
| Android decode at equilibrium | `b0ffd7e` (incl. #342) | Pixel 10 Pro Fold |
| Mac cera vs llama.cpp | `b0ffd7e` (incl. #342) | M1 Max |
| GPU I/O counters | mixed, see the † note in that section | Mac / Adreno |
| GPU decode profile | `0f00dec` (incl. #316, #318, #319, #320) | M1 Max |
| CPU thread scaling | `433c3eb` (incl. #344-#348) | Pixel 10 Pro Fold |
| Q4_K i8mm kernel counters | #351 (see the PR for the exact tree) | Pixel 10 Pro Fold |

**The first four sections predate #344-#348 and #351 and have not been
re-measured against them.** They are still the reference for what they cover;
they are just not a "current tip" reading. The two sections added below carry
the newer work and state their own protocol.

**Pin the llama.cpp build, and keep the binary.** The previous revision did name
its build (`b9980`, vendored via pipette-llamacpp), which was right, but that
binary is no longer on the device: what is there now is `f12cc6d0f` (9371). The
two disagree by up to 2.45x on the same model, device and config (`-t 1` prefill:
170.6 then, 69.7 now). Some of that spread is the cooling regime and the harness
change rather than the build, exactly as for cera, so read it as "these numbers
are not comparable" rather than as a measured build regression. Naming the build
is necessary but not sufficient. A
llama row is only reproducible if the binary it was measured with still exists.
Note 9371 is *older* than b9980; it is simply what is staged on the device, and
no attempt was made to pick a favourable one.

## Android: Pixel 10 Pro Fold (Tensor G5)

8 cores: 2 efficiency (cpu0-1) + 5 perf (cpu2-6) + 1 prime (cpu7).
CPU has `asimddp` + `i8mm`.

**Measure at thermal equilibrium.** A short decode measurement on this device
runs inside a violent thermal transient: the BIG cluster goes 26 C to 74 C in
about twelve seconds of load and falls back to 30 C within twenty seconds of
stopping. Where a 2-second measurement lands in that ramp decides its result,
which is why repeating one identical pinned config gave 7.1 / 63.3 / 64.9 /
40.0 / 22.7 / 22.1 tok/s, a 9.1x spread. Two fixes were measured: gating each
invocation on a cold SoC still left 1.48x, because the transient is the problem
rather than the start point; driving the device to equilibrium and measuring
there gave 1.12x and ~4.3% CoV on both engines.

`scripts/bench_android.sh` implements this: warm-up passes to reach steady
state, then measured passes back to back with no idle between them (idling drops
the device back into the transient), engines interleaved, reporting median and
CoV per cell plus the BIG-cluster temperature range as evidence equilibrium held.
This measures **sustained** throughput, which is the reproducible quantity and
the one that matters for comparing engines or commits. It is deliberately not
the peak a cold phone reaches for two seconds, which is roughly 1.5x higher and
not repeatable.

### Decode, frequency-matched, LFM2.5-350M Q4_K_M

cera `b0ffd7e`, llama.cpp `f12cc6d0f` (9371), both pinned to the same five perf
cores, 16 interleaved invocations each with the mid-run CPU frequency recorded,
512-token decode.

| Engine | All samples | Filtered to 3052 MHz |
|---|---|---|
| cera | n=16, median 51.8, CoV 12.4% | n=8, median **58.3**, CoV 7.1% |
| llama.cpp | n=16, median 50.8, CoV 14.1% | n=3, median **56.9**, CoV 4.5% |

**No significant difference: 1.02x at 0.7 sigma.** At matched CPU frequency the
two engines decode at the same speed on this device. Unfiltered, the same 32
invocations would support almost any ratio you wanted to quote, which is what
several earlier revisions of this file did.

Two caveats. llama reached the maximum clock in only 3 of 16 invocations against
cera's 8, so its filtered median rests on a small sample; the asymmetry is
probably cera's heavier in-process warmup driving the governor up before its
timed run, and it is worth investigating separately. And cera decodes from a
128-token prompt while `llama-bench`'s `tg` starts from an empty context, which
disfavours cera slightly.

The full matrix (all pinning configs, prefill, GPU) has not been re-measured
under this protocol yet; a run needs roughly half an hour of device time because
equilibrium has to be reached and held. The GPU row from the previous protocol
is unaffected by any of this and reproduced exactly (21.0 / 20.9 / 20.9, 1.00x)
because the GPU has its own clock domain.

llama.cpp's Vulkan build does run on this device's PowerVR GPU, and its prefill
(638) is the highest prefill measured here, but it is excluded from comparison:
it logs `Compute pipeline creation failed for mul_mat_vec_q4_k_f32_f32`, decodes
at 3.06 tok/s, and a sustained decode run wedged the phone into a `watchdog,apc`
reset.

## Android: Pixel 9 Pro XL (Tensor G4)

8 cores: 4 efficiency A520 (cpu0-3, 1.95 GHz) + 3 mid A720 (cpu4-6, 2.6 GHz) +
1 prime X4 (cpu7, 3.105 GHz). Mali Immortalis-G715, reached through wgpu/Vulkan.

**The cluster layout differs from the Fold's, which is why the harness derives
taskset masks instead of hardcoding them.** The literal `7c` is "the perf
cluster" on a Tensor G5 and "two efficiency cores plus the mid cluster" on a G4.
`bench_android.sh` groups cores by `cpuinfo_max_freq` (highest tier prime, next
mid) and labels each row with the mask it actually used. On a G5 layout the
derivation reproduces the old literals exactly, so the Fold numbers above stand.

cera `b0ffd7e`, llama.cpp `992871d3c` (8764), LFM2.5-350M Q4_K_M, median of 4
measured passes after 2 discarded warm-up passes, background apps stopped first.
Charging over USB throughout (80%), because adb here is USB and unplugging would
drop the connection.

**Two independent matrices were run at this commit**, the second adding the
`t8-all` cell. The table is the second run; the first is kept below purely as a
reproducibility control, and it earns its place by exposing a cell that does not
reproduce.

| Engine | Config | Prefill | CoV | Decode | CoV |
|---|---|---|---|---|---|
| cera | default RowPool (8c, unpinned) | 65.5 | 10.8% | 43.3 | 2.1% |
| cera | pin prime (cpu7) | 39.0 | 3.6% | 39.6 | 11.3% |
| cera | pin mid (cpu4-6) | 50.5 | 1.7% | **43.0** | 9.2% |
| cera | wgpu/Vulkan | 99.5 | 0.5% | 20.8 | 1.0% |
| llama.cpp | t1 prime | 36.8 | 2.7% | 30.0 | 1.7% |
| llama.cpp | t3 mid | 54.5 | 2.1% | 37.6 | 2.0% |
| llama.cpp | t4 big (cpu4-7) | 73.1 | 0.1% | **40.9** | 4.4% |
| llama.cpp | t8 all (unpinned) | **143.6** | 4.3% | 25.2 | 7.8% |

At matched core counts, which is the only engine comparison worth quoting. Both
runs are shown because agreement across two matrices is the actual evidence:

| | cera | llama.cpp | run 2 | run 1 |
|---|---|---|---|---|
| prefill, 1 prime core | 39.0 | 36.8 | cera 1.06x (2.6 sigma) | cera 1.08x |
| decode, 1 prime core | 39.6 | 30.0 | cera 1.32x (4.3 sigma) | cera 1.32x |
| prefill, 3 mid cores | 50.5 | 54.5 | llama 1.08x (5.6 sigma) | llama 1.05x |
| decode, 3 mid cores | 43.0 | 37.6 | cera 1.14x (2.7 sigma) | cera 1.18x |

**The `default RowPool` row is a 4-core config, not an 8-core one, so it must not
be read against `llama t8`.** `cpu_features.rs` classifies cores by kernel EAS
capacity (`CAP_MID = 400`), runs workers only on performance cores, self-pinned
fastest-first, capped at `MAX_AUTO_THREADS = 6`. This SoC's four A520s fall below
that threshold, so cera's default is 4 threads on cpu4-7. Its core-matched
counterpart is `llama t4-big` (73.1 vs 65.5, llama 1.12x), *not* `llama t8`.

**cera's prefill matches llama's at equal core counts; it simply declines the
efficiency cores.** With `CERA_THREADS=8`, cera prefill measures 138-143 across
three alternating rounds against llama's 143.6. There is no per-core kernel
deficit here, only a pool-width policy.

**But the pool cannot simply be widened, because decode falls off a cliff the
moment a worker lands on an efficiency core:**

| `CERA_THREADS` | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| decode tok/s | 61.0 | 4.5 | 3.0 | 2.3 | 1.8 |

The device has exactly four performance cores, and the fifth thread costs 13.5x.
A straggler on a core only 1.6x slower cannot produce that, so the mechanism is
the pinning/spin interaction rather than simple load imbalance: `pin_cores` leaves
surplus workers unpinned once it runs out of performance cores, and spinning
waiters then contend with the very thread the per-token barrier is waiting on.
Prefill does not care, because its barrier is per-batch rather than per-token.

Two consequences. First, **`CAP_MID` is load-bearing safety, not tuning**: it is
the only thing keeping a worker off an efficiency core on this class of SoC, and
a part whose efficiency cores report capacity >= 400 would decode ~13x slower.
Second, the fix for prefill is a **phase-specific width**, which the existing
`CERA_DECODE_THREADS` knob already supports:

| config | prefill | decode |
|---|---|---|
| default | 107 | 62.5 |
| `CERA_THREADS=8` | 117 | 1.5 |
| `CERA_THREADS=8 CERA_DECODE_THREADS=4` | 129 | 54.1 |

Widening prefill alone is worth roughly 1.2x here. The residual decode gap in the
last row (54.1 vs 62.5) is from single-run samples and is not yet resolved
against noise; confirm it before acting on the split as a default.

llama shows the same asymmetry from the other side: its `t8` decodes at 25.2
against its own 4-core 40.9. Neither engine wants every core for both phases.

**Retracted: "cera's default RowPool costs 1.34x on decode."** The first matrix
measured default decode at 33.2 against pin-mid's 44.5 and the gap looked
overwhelming at 15 sigma. The second matrix puts default at 43.3 against
pin-mid's 43.0, a 1.01x difference at 0.1 sigma. The claim was one run fitted to
a mechanism.

What the two runs together do establish is worse than a fixed penalty and more
useful to know:

| Cell | prefill r1 -> r2 | decode r1 -> r2 |
|---|---|---|
| cera default RowPool (unpinned) | 94.0 -> 65.5 (**-30%**) | 33.2 -> 43.3 (**+30%**) |
| cera pin-prime | 38.5 -> 39.0 (+1%) | 40.4 -> 39.6 (-2%) |
| cera pin-mid | 50.0 -> 50.5 (+1%) | 44.5 -> 43.0 (-3%) |
| cera wgpu | 99.5 -> 99.5 (0%) | 20.1 -> 20.8 (+4%) |
| llama t1 / t3 / t4 | +3% / +4% / 0% | -2% / 0% / +1% |

**Six of seven cells reproduce within 4%; the unpinned RowPool cell moves 30% in
both directions.** Within either run it looks solid (CoV 0.7% and 2.1%), which is
what makes it dangerous: it is stable across passes and bimodal across runs, so a
single matrix reports it with false confidence. The wgpu row is the control that
rules out "the device changed", since it is also unpinned but does its work on
the GPU and reproduced to 0.0%. Quote pinned cera configs, or quote the default
with both numbers.

One number still **not** a head-to-head result: cera's GPU prefill (99.5) against
any llama CPU row is cross-backend, and llama's Vulkan backend was not run here.
Best-vs-best decode is also not established: cera pin-mid 43.0 against llama t4
40.9 is 1.05x at 1.0 sigma, inside the noise.

Nothing here is comparable to the Fold section: different SoC and a different
llama.cpp build (`992871d3c`/8764 here, `f12cc6d0f`/9371 there).

## Mac: cera vs llama.cpp on M1 Max

cera `b0ffd7e`, llama.cpp `75ad0b23e` (9770, Homebrew, BLAS + Metal), 15 runs
each, p50 +/- stddev. Prefill and decode are separate runs on both sides, which
is what `llama-bench` does internally and what the traps section explains.

**LFM2.5-350M Q4_K_M**

| Engine | Backend | Prefill | Decode |
|---|---|---:|---:|
| cera | Metal | 9274 +/- 181 | 281.9 +/- 49.4 |
| llama.cpp | Metal | 9933 +/- 821 | 235.0 +/- 37.6 |
| cera | CPU (`blas`) | 1403 +/- 27 | 156.9 +/- 11.5 |
| llama.cpp | CPU | 1381 +/- 44 | **308.5 +/- 17.4** |

**Llama-3.2-1B Q4_0** (dense transformer, so a different model path)

| Engine | Backend | Prefill | Decode |
|---|---|---:|---:|
| cera | Metal | 3439 +/- 46 | 182.1 +/- 21.5 |
| llama.cpp | Metal | 3813 +/- 47 | 170.4 +/- 31.6 |
| cera | CPU (`blas`) | 537 +/- 7 | 68.8 +/- 11.9 |
| llama.cpp | CPU | 652 +/- 26 | **120.7 +/- 10.2** |

**Three gaps here are larger than their own noise; the rest are not.** Taking
each in turn rather than as one headline:

- **llama.cpp CPU decode, 1.75-1.97x.** Unambiguous on both models, distributions
  nowhere near overlapping (308.5 +/- 17.4 vs 156.9 +/- 11.5; 120.7 +/- 10.2 vs
  68.8 +/- 11.9). This is the one clear engine-level gap in the whole file.
- **llama.cpp Metal prefill on Llama-1B, 1.11x** (3813 +/- 47 vs 3439 +/- 46).
  A 374 gap against ~46 stddev on both sides, so real despite being small.
- **llama.cpp CPU prefill on Llama-1B, 1.21x** (652 +/- 26 vs 537 +/- 7).

Not resolvable at this sample size:

- **Metal decode.** Medians favour cera (1.20x on 350M, 1.07x on 1B) but the
  coefficients of variation are 17.5% and 11.8% for cera against 16.0% and 18.5%
  for llama, so the distributions overlap. Across four sessions cera's 350M Metal
  decode p50 came back 345.0 / 324.9 / 291.6 / 281.9, spanning the entire claimed
  effect. Do not quote a Metal decode ratio from a single matrix.
- **Metal prefill on 350M** (9933 +/- 821 vs 9274 +/- 181): llama's own stddev is
  wider than the 659 gap.
- **CPU prefill on 350M** (1403 +/- 27 vs 1381 +/- 44): inside both stddevs.

Note the decode comparisons carry the same KV-depth asymmetry as the Android
table: cera decodes from a 128-token prompt while `llama-bench`'s `tg` starts
from an empty context, which disfavours cera. A cera decode win is therefore a
lower bound, and a llama decode win is not inflated by it.

cera's CPU rows use the opt-in `blas` feature, because llama.cpp always links
BLAS and a no-BLAS comparison is not like-for-like. Measured both ways in one session on
LFM2.5-350M Q4_K_M, `blas` is worth 2.62x on prefill (1403 vs 535) and is within
noise on decode (156.9 vs 149.7), as expected for a GEMM-vs-GEMV split. Both
halves come from the same session on purpose: the backend-comparison table in
`README.md` reports 434 for that same no-BLAS cell at an older commit, and a 23%
unexplained gap on one cell is exactly why a ratio should not be assembled from
numbers taken at different times.

## GPU I/O counters (`cera bench --gpu-io`)

> **These numbers were wrong in the first cut of this doc** (1.0 submits/token,
> 0.045 prefill submits/prompt-token). The counter only saw submits routed through
> `submit_encoder`, and the model bypassed it with direct `queue.submit` calls, so
> it was counting the logits readback and nothing else. Every submit is now routed
> through the choke point. The full post-mortem is in `GPU_FINDINGS_CORRECTION.md`.

Prompt 512, greedy decode:

| | LFM2-VL-450M **Q4_0** (Mac) | LFM2.5-350M **Q4_K_M** (Mac) | LFM2.5-350M **Q4_K_M** (Adreno) |
|---|---:|---:|---:|
| decode submits / token | **17.0** *(was 19)* | 19.0 † | 19.0 † |
| decode compute passes / token | **47.0** *(was 67)* | n/m | n/m |
| decode readbacks / token | 1.0 | 1.0 | 1.0 |
| decode readback **bytes** / token | 4 | 4 | 4 |

† Not re-measured since #319; the mechanism that removed two submits is
model-independent, so expect 17 there too (one submit per layer, plus one tail).
The pass counter postdates #318.
| **prefill submits** (512-tok prompt) | **25** | **8728** | **8728** |
| prefill readbacks (512-tok prompt) | 23 | 23 | n/m |
| prefill readback **bytes** (512-tok prompt) | 12,926,976 | 12,926,976 | n/m |

`n/m` = not measured; the prefill readback counters postdate the Adreno run and
re-measuring needs the device.

Read:

- **Decode issues one submit per layer plus one tail submit** (17 for this
  16-layer model). Merging the *per-layer* submits into a single command buffer
  remains WONTFIX: it measured ~30% slower on both Mac and Adreno, because decode
  is GPU-bound and the per-layer submits overlap GPU execution with CPU encode
  (T6, see `GPU_FINDINGS_CORRECTION.md`).
- **The tail used to be three submits, and two of them were pure waste** (#319).
  The greedy argmax had its own encoder and its own blocking submit, and the
  4-byte result readback submitted *again* to stage the copy. A submit costs a GPU
  round trip regardless of payload: ~1.3 ms for a kernel doing 0.13 ms of work,
  and ~1.5 ms to move four bytes. Both now ride along in the output projection's
  submission. This is the opposite of T6 and does not contradict it: what is
  expensive is a *blocking* round trip that carries nothing, not per-layer
  pipelining.
- Greedy decode **does** sample on the GPU: the readback is a 4-byte token id, not
  vocab logits. That part was always true.
- **Prefill batching is gated on quantization, not platform.** The batched path
  requires every matmul weight to be `Q4_0`/`Q8_0`/`Q4_K`. A `Q4_K_M` file carries
  **11 Q6_K tensors**, which fails the check, so prefill **silently** falls back to
  the per-token loop: 8728 submits instead of 25. The same model does this on Mac
  too; it is not an Adreno effect (T8).
- **Prefill reads back ~12.9 MB per 512-token prompt** (23 readbacks), against
  decode's 4 bytes/token. Not negligible, and its own optimization target.
- **That readback volume is identical on both paths**: 23 readbacks and
  12,926,976 bytes whether prefill runs batched (Q4_0, 25 submits) or falls back
  (Q4_K_M, 8728 submits). So the readbacks are *not* a symptom of the fallback
  above; fixing the dtype gate will not touch them. Two independent problems.

Because of that gate, the Mac-vs-Adreno rows in earlier revisions of this doc
compared a **Q4_0** model on Mac against a **Q4_K_M** model on Adreno, i.e. the
batched path against the fallback path. Same-model gaps are **3.3x prefill** and
**2.3x decode**, not the 13x/4x reported before.

Caveat: `bench` decodes greedily (temp=0), which takes the on-GPU argmax path. The
**non-greedy** path still downloads full vocab logits per token, invisible to
these numbers by construction.

## GPU decode profile (`CERA_GPU_PROFILE=1`)

Per-kernel GPU timestamps. LFM2-VL-450M **Q4_0**, wgpu/Metal, M1 Max (400 GB/s),
greedy decode, `--no-cache`.

Both columns are the same model on the same machine with the same command, so
they are directly comparable. "before" is this doc's original measurement (commit
`c6f845d`); "now" is after #316 (GEMV register spill), #318 (conv block in one
pass), #319 (decode round trips) and #320 (LM head on the stored weight).

| span | before | now | |
|---|---:|---:|---|
| `ffn` (16×) | 4492 µs | **3234 µs** | −28% (#316) |
| conv block (10×) | 1539 µs *(pre+mid+post)* | **1062 µs** *(one `conv` span)* | −31% (#316, #318) |
| LM head (1×) | 1265 µs *(`gemv_f16`)* | **856 µs** *(`lm_head`)* | −32% (#320) |
| `attn_pre` (6×) | 732 µs | 647 µs | |
| `flash_attention` (6×) | 476 µs | 524 µs | |
| `attn_post` (6×) | 258 µs | 221 µs | |
| `rmsnorm` / `argmax` | 24 / 139 µs | 27 / 130 µs | |
| **sum of GPU passes** | **~8.9 ms** | **~6.7 ms** | |
| **decode, unprofiled** | **63.4 tok/s** (15.8 ms/token) | **112.8 tok/s** (8.9 ms/token) | **1.78×** |

Profiling overhead is ~18% here (93.0 tok/s profiled vs 112.8 unprofiled), so
treat the span times as inflated by roughly that much.

Read:

- **The quantized GEMVs are still the bottleneck.** In production they sustain
  ~26 GB/s where the f16 kernel does ~93 GB/s on the same GPU, and `ffn` alone is
  48% of GPU pass time.

  Earlier revisions of this doc had the isolated-vs-in-model gap the other way
  round, the microbenchmark reading ~13 GB/s where production sustained ~26 GB/s,
  and treated that inversion as an open puzzle. It was the benchmark, and that
  puzzle is gone. Whether a gap remains in the other direction cannot be said
  until the harness is trustworthy again; see below.

  The standing explanation has been *how the bytes are read*: quantized kernels
  fetch via scalar `u32` loads with shift/branch byte extraction where `gemv_f16`
  reads aligned vectors. That held for two kernels (#316 register spills, #321
  q6_k byte-at-a-time loads) but **it is not a general law, and it failed the last
  time it was applied**: three q4_k rewrites aimed at its load pattern produced no
  net win, and the one that most reduced loads per element was the one that
  regressed. Treat "read the bytes better" as a hypothesis to A/B per kernel, not
  as a diagnosis that transfers.

  **RETRACTED (2026-07-27): there is no "FFN per-dispatch floor".** This section
  used to claim that at FFN shapes a fixed per-dispatch cost dominates and no
  kernel amortizes it, citing a q4_k m-sweep of 9.4 / 13.2 / 22.1 / 25.9 / 31.2 /
  48.0 GB/s that climbed monotonically and never plateaued. That sweep, and the
  whole GEMV table that used to sit here, were measured with a benchmark bug: a
  cell's number depended on **where it sat in the table**, and whichever kernel
  ran first absorbed a large penalty. `q6_k` at ffn gate/up reports **0.0305 ms
  running last and 0.0952 ms running first**; reversing the kernel order moves the
  penalty to whatever is now first. q4_k was listed first, so its whole row,
  including the sweep that produced the "floor", was the artifact. Fixed by
  running the table twice and reporting the second round; a 750 ms global clock
  ramp was tried first and does **not** fix it, so the mechanism is not clock ramp
  and is still unidentified.

  **A second harness defect was then found (the third overall in this benchmark,
  counting the one #321 fixed), and no replacement table is published here.**
  Every measurement also carried one blocking submit + `poll(Wait)`, a
  GPU round trip costing ~1.0-1.5 ms whatever it carries, amortised over `ITERS`
  and never subtracted. So a cell read `T + C/ITERS`, not `T`. It is worst exactly
  where the kernel is cheapest: `q6_k` at ffn gate/up measured **46.0 / 63.6 /
  79.3 GB/s at ITERS = 50 / 200 / 1000**, and fitting that gives C ~ 1.16 ms, one
  round trip. The bench now uses two-point timing (`run(n)` and `run(2n)`,
  differenced) which cancels C exactly, but differencing two noisy measurements
  raises variance, and the resulting per-cell numbers do not yet replicate tightly
  enough to publish. A 7-run table measured after fixing the first defect but
  before finding this one was drafted here and then withdrawn; an independent
  replication had already failed to reproduce its `ffn down` row.

  **So: the retractions below stand, and no per-kernel bandwidth ranking is
  currently authoritative.** Retracting a wrong number does not require a right
  one, and publishing a third table that also fails to replicate would repeat the
  error this section exists to record. What survives is what held under *every*
  measurement regime tried (the buggy table, the 7-run table, the ITERS sweeps,
  and an independent replication):

  - **There is no "FFN per-dispatch floor."** q6_k and the f32 control both run
    far above the claimed 13-26 GB/s ceiling at FFN shapes under every regime. A
    real size-dependent cost exists (the f32 control rises monotonically with `m`
    at fixed k=1024), but not the hard floor that was published.
  - **q4_k is not ~3x slower than q4_0.** They are close at every FFN shape under
    every regime, at identical bytes/element (144/256 == 18/32 == 0.5625). Which
    of the two leads at a given shape moves with the measurement regime, so no
    ordering is claimed; the ~3x gap is simply not there.
  - **q4_k is materially behind at the LM head**, by roughly a quarter to a third,
    reproducibly and in the most stable shape measured.
  - **q8_0 is the worst kernel by a wide margin at the FFN and LM-head shapes**,
    far outside any noise seen here, and is the best remaining target. (At
    `attn qkv/out`, the noisiest shape measured, it is not separable from q4_0,
    so the claim is not made there.) Size the work from a fresh
    in-model measurement, not from this harness: in isolation it looks ~40% less
    efficient per byte than q4_0, while the in-model A/B below implies ~10%
    (1.89x bytes for 2.10x time). That disagreement is unresolved.

  **Before publishing a GEMV table again**, characterise the variance of
  `cera/examples/wgpu_gemv_bench`:
  fix the two defects found here (done), then establish how many runs a cell needs
  to replicate, and verify by permuting kernel order *and* sweeping `ITERS`:
  both must leave the conclusion unchanged. Until then prefer an **ABBA A/B within
  one process** on the specific pair in question, which is what the kernel-variant
  results below rest on and why they are still quoted.

  ```bash
  # medians of >=3; cells printing `noise` need a larger ITERS
  cargo run --release -p cera --features gpu --example wgpu_gemv_bench
  ITERS=1000 cargo run --release -p cera --features gpu --example wgpu_gemv_bench
  ```

  Three q4_k kernel variants were written and A/B'd against the original, ABBA
  ordered, with q4_k held in a fixed non-first slot. **None was a net win and all
  were discarded.** Recorded here so they are not re-attempted blind:

  **Change in wall time vs the original kernel: negative is faster.**

  | variant | ffn gate/up | lm head |
  |---|---|---|
  | 16 elems/thread + block interleave, hoisted scales | −11% | **+18%** |
  | hoisted scale-byte indexing alone | +17% | +15% |
  | 16 elems/thread + block interleave alone | −8% (noisy) | **+4.9%** (tight) |

  Hoisting the scale-byte indexing out of the block loop was actively worse at
  every shape: it keeps six extra registers live for the whole kernel to save
  `select`s the backend was already folding. The thread remap won at ffn gate/up
  but cost a tight, reproducible regression at the LM head: more live registers
  per thread means fewer resident threadgroups, and at m=65536 occupancy is what
  buys the latency hiding. Trading a certain LM-head regression for a noisy FFN
  gain is a bad deal on this GPU and a worse one on mobile.

- **Decode is memory-bound, confirmed by A/B, not by inspection.** The same model
  at Q8_0 moves 1.89× the FFN bytes and took **2.10×** the FFN time. Time tracks
  bytes. An ALU/dequant bound was the obvious story from reading
  `gemv_q4_0_fast.wgsl` and it is **wrong**: Q8_0 is *cheaper* to unpack and got
  proportionally slower anyway.

- **~24% of decode wall time is now outside every GPU pass** (6.7 ms of passes vs
  8.9 ms wall), down from ~44%. #319 is what closed it, by removing two blocking
  GPU round trips from the tail. The earlier claim that this gap was "not
  recoverable by merging submits" was right about *submits* and wrong as a general
  conclusion; see `GPU_FINDINGS_CORRECTION.md`, round 3.

- **Compute-pass count is NOT a general lever: do not re-derive one from #318.**
  Merging the conv block's three passes into one was worth +17%, but merging
  passes in the attention/FFN path measured **neutral-to-negative**: a boundary
  there is worth ~6 µs, not the ~20–38 µs the conv result implied, and carrying
  those merges cost ~15%. Implemented, measured, reverted (#319). A cost-per-unit
  derived from one successful change describes that change, not the system.

- Not yet measured on Adreno. Every number in this section is M1 Max, and the
  register/occupancy trades that decided the q4_k variants above will differ on a
  mobile tiler with a tighter register budget, so which kernel wins may differ
  too, not just by how much.

## CPU thread scaling on big.LITTLE (Pixel 10 Pro Fold)

cera `433c3eb`, prefill 512 / decode 128, `--runs 3 --warmup 1`, arms
interleaved with a rotating order, each invocation gated on BIG-cluster
temperature <= 40 C **and** 1-minute loadavg <= 5. **The two subsections below
use different models**; each states its own, because they were measured
separately.

**These are width-dependent results, and they exist only on tiered silicon.**
Both mechanisms key off cores differing in speed: capacity-aware chunk sizing
scales a worker's steal by its core's weight, and the prefill pool drops pinning
once it is wider than `fast_cores`. Neither is reachable unless
`detect_topology_sysfs` returned a topology, and it returns `None` for
homogeneous parts by design, leaving `pin_cores` empty and `fast_cores` at 0. So
on any uniform host both are inert. That is why the CI benchmark workflow cannot
track them and this file has to. `cera cpu` prints which case the host is in:
`placement=tiered (4.92x fastest:slowest)` here, `placement=flat` on a hosted
runner.

### Prefill pinning, `CERA_THREADS=8` (#348)

LFM2.5-1.2B-Q4_K_M.

Three arms, 5 rounds. `main-nopin` is `CERA_PIN=0` on the old binary, included
because it is *not* the same configuration as the fix: it also unpins the decode
pool, the caller, and rayon's mask, so it says whether the narrow change
captures the win rather than merely resembling it.

| arm | prefill p50 | range | decode p50 |
|---|---:|---|---:|
| main (pin at every width) | 128.0 | 76-132 | 88.0 |
| **pinfix** (drop pinning past `fast_cores`) | **197.0** | 129-205 | 88.8 |
| main + `CERA_PIN=0` | 201.0 | 191-204 | 71.4 |

**+54% prefill, and decode is unharmed** (88.8 against 88.0). The blunt
`CERA_PIN=0` reaches the same prefill (201) but costs decode 19% (71.4 against
88.0), which is the whole reason the fix is scoped to the prefill pool instead
of being a global switch.

At the default width this changes nothing, by construction: the default pool is
6 wide against `fast_cores=6`, so `prefill_should_pin` is still true and the
pinned path is unchanged. The default width is also what CI measures.

### Steal-chunk sizing by core capacity (#347)

LFM2.5-**350M**-Q4_K_M, not the 1.2B used above. These are the figures in
`worker_chunk_rows`' doc table, reproduced here so the two live in one place;
that table is the primary record.

+22% prefill and +19.5% decode at `CERA_THREADS=8`; **+2.9% prefill at the
default width**, where the pool holds only performance cores and there is little
spread left to correct for. The small default-width figure is the honest headline
for shipping configurations; the 8-thread figure is what the mechanism does when
a widened pool actually straddles the A520s.

Two scheduling alternatives were measured against this and **both lost**:
ggml-style fixed-size chunks (16 rows, 64 for GEMV) and guided self-scheduling
(`chunk = ceil(remaining/P)`). At `CERA_THREADS=8`, medians over 4 rounds:
count-based sizing 146.0, `size-16` 142.5, gss 137.0, `size-8` 136.5, `size-32`
135.0. The shipped fixed-count sizing wins; the tail-exposure argument for
fixed-size chunks dies on `MIN_CHUNK_ROWS`.

## Q4_K i8mm kernel counters (#351)

**This is a counter movement without a matching throughput number, and it is
recorded as such.** The file's own rule cuts the other way too: a throughput
number without a counter movement is not evidence, and a counter movement alone
is not a speedup claim. The device was at loadavg 10-27 for the whole
measurement window against a gate of 5, so no trustworthy wall-clock delta was
obtained.

Profiling (simpleperf `-e cpu-clock:u`, 350M-Q4_K_M, both engines at their best
thread count) localized the entire remaining prefill gap to one kernel:

| region | cera | llama.cpp | ratio |
|---|---:|---:|---|
| Q4_K kernel | 3.558 | 2.550 | 1.40x slower |
| Q6_K kernel | 0.573 | 0.579 | parity |
| everything else | 0.608 | 0.870 | 0.70x (cera faster) |

cera's pool overhead is 5.1% of prefill against ggml's 7.7%, so the threading
side is not the constraint. Note llama.cpp is **not** using a repacked GEMM
here: its `q4_K_8x8` path is gated on 256-bit SVE and this SoC has 128-bit, so a
plain per-row `vec_dot` was winning. The deficit was implementation, not
algorithm.

Instructions per `smmla` (both bodies contain 8 `smmla`; both kernels cover 8
elements per `smmla`, so the ratio is comparable per unit of arithmetic):

| kernel | before | after | stores | loads |
|---|---:|---:|---|---|
| Q4_K | 99.9 | **44.9** | 63 → 26 | 148 → 73 |
| Q6_K | 68.1 | **54.2** | 39 → 36 | 108 → 103 |

The cause was `half::f16::to_f32` running `is_aarch64_feature_detected!("fp16")`
per call and lowering to two non-inlined `bl`s, spilling the kernel's live
accumulators. Q4_K called it 4x per superblock and Q6_K 2x, which is exactly the
2:1 ratio in their spill counts.

**A refuted hypothesis worth not re-testing:** widening the Q4_K activation loads
from `vld1_s8` to `vld1q_s8` to match Q6_K. Both binaries disassemble identically
in the hot kernel (800 instructions, 10 `ldr q`, 8 `smmla`, 73 loads); LLVM
already merges the adjacent 8-byte loads.

## Reproduce

```bash
# Android (build first: cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu)
scripts/bench_android.sh --model LFM2.5-350M-Q4_K_M.gguf --serial <adb-serial> \
  --llama-bench /data/local/tmp/.../llama-bench --decode-prompt 128 \
  --passes 5 --equil-warm 2

# Android CPU profile (hotspot must move when a kernel task lands)
scripts/profile_android_cpu.sh --model LFM2.5-350M-Q4_K_M.gguf --mask 80

# Mac / desktop matrix
scripts/bench_matrix.sh

# Mac cera vs llama.cpp, matched to llama-bench's separate pp/tg runs
cera bench -m <model.gguf> --device metal --no-cache --context-size 8192 \
  --prompt-tokens 512 --max-tokens 0   --runs 15 --warmup 3   # prefill
cera bench -m <model.gguf> --device metal --no-cache --context-size 8192 \
  --prompt-tokens 128 --max-tokens 128 --runs 15 --warmup 3   # decode
# CPU rows: same two commands with --device cpu
llama-bench -m <model.gguf> -p 512 -n 128 -ngl 99 -r 15   # Metal rows
llama-bench -m <model.gguf> -p 512 -n 128 -ngl 0  -r 15   # CPU rows
# cera CPU rows need the opt-in BLAS feature to match llama's always-on BLAS:
#   cargo build --release -p cera-cli --features gpu,metal,blas

# Per-kernel GPU timestamps (prints a span table per forward pass, to stderr)
CERA_GPU_PROFILE=1 cera bench -m <model.gguf> --device gpu \
  --runs 1 --warmup 1 --max-tokens 4 --no-cache
```

## Known measurement traps

- **Don't trust an unpinned multithreaded run.** Free-scheduled runs on
  big.LITTLE were bimodal (llama `-t 4` came back `100 ± 99` pp512), and pinning
  is what made them repeatable. Not every row above is pinned: cera's headline
  `default RowPool` row and the wgpu row are deliberately unpinned, because
  RowPool sizes itself and that is the shipping configuration. Compare pinned to
  pinned when attributing a difference to a kernel.
- **Warm the run, cool the device.** These are different things and both matter.
  cera's first Android numbers were ~15% low because runs 1-2 were cold, which
  `--warmup 2` fixed; that is warming the *clocks and caches*. Letting the SoC
  heat across a matrix is the opposite problem, and costs far more (see the
  cooling note below). `bench` prints thermal headroom per run; if it climbs
  toward 1.0 the number is thermally limited, not a ceiling.
- **`--no-cache` matters.** Without it the KV prefix cache makes prefill look
  arbitrarily fast on repeat runs.
- **Measure prefill with `--max-tokens 0`.** Timing prefill in a run that also
  decodes reads ~10% low and inflates variance ~8x (LFM2.5-350M Q4_K_M on Metal,
  one session: 9228 p50 / 128 stddev with `--max-tokens 0`, versus 8259 p50 /
  1028 stddev with `--max-tokens 128`, same binary and model. The table above
  reports 9274 for that cell from a later session; session-to-session drift of
  that size is normal here and is why the tables carry stddev). `llama-bench` measures `pp` and `tg`
  as separate runs, so comparing its prefill against a combined cera run
  understates cera by that margin. Decode is the mirror image: it wants a
  realistic prompt, so measure it separately with `--prompt-tokens 128
  --max-tokens 128`.
- **Throughput tracks CPU frequency, and the governor's ramp is the dominant
  noise source.** Measured against llama-bench decode over 16 instrumented
  invocations: `corr(tok/s, CPU frequency) = +0.83`, and filtering to the samples
  taken at the maximum 3052 MHz cut the coefficient of variation from **17.1% to
  2.6%** (cera: 13.2% to 6.0%). `sched_pixel` has 24 operating points from 177
  MHz to 3052 MHz and ramps on recent load, so a short measurement can finish
  before the cores reach the top, and whether they do depends on load history.
  That is why the failure is episodic rather than random.
  **Temperature correlates with frequency at +0.95 in the same direction**, so a
  hot device here is a fast one and thermal throttling is not the mechanism; an
  earlier revision of this file had that causality backwards.
  It cannot be fixed from outside the device, and all of these were measured and
  rejected: pinning the governor (needs root), pre-warming the cores (decays
  across the adb round trip), longer measurements (trade ramp noise for thermal
  decline, and the median falls), more warmup iterations (16.3% vs 17.3% CoV),
  mmap vs no-mmap, threadpool `--poll`/`--cpu-strict`, and memory pressure
  (`SwapFree` was flat across every invocation). So `bench_android.sh` records
  `cpu_mhz_min`/`cpu_mhz_max` per cell; a cell whose range does not sit at the
  top of the ladder was not measured at speed.
- **Gate on background load, not only on temperature.** The rejected-causes list
  above dismisses "memory pressure" on the evidence that `SwapFree` was flat, and
  that remains true, but it is a narrower claim than it reads as: it says swap was
  not the mechanism, not that a busy device measures the same as an idle one. It
  does not. On a session where the phone sat at 1-minute loadavg 10-27, the same
  binary and model read prefill anywhere from 25 to 208 tok/s, and both arms of an
  A/B degraded together, so the ratio survived while every absolute number was
  worthless. Read `/proc/loadavg` alongside the thermal gate and discard samples
  taken above about 5; a cold SoC is a necessary condition for a valid
  measurement, not a sufficient one. Gating on temperature alone will happily
  admit a run competing with whatever else the phone decided to do.
- **A/B direction survives a loaded device; magnitude does not.** Corollary of the
  above, and it decides what you may write down. Non-overlapping paired ranges
  across rotated rounds are still evidence of a sign. The medians are not evidence
  of a size, because the compression is not uniform across the range. Report the
  direction and say the magnitude is unmeasured, rather than quoting a ratio of
  two numbers that were both wrong.
- **Sample a sensor while the load is running, not around it.** This has now gone
  wrong twice in this harness, and both times the broken column was added
  specifically to explain variance. First, battery temperature was used as the
  thermal gate: it moves ~0.5 C while the silicon swings 48 C, so it could not see
  the effect at all. Then the replacement read the BIG cluster and the CPU
  frequency *between* invocations, which is after an adb round trip of idle: those
  columns logged 45 C and 700 MHz for workloads that ran at 85 C and 3105 MHz.
  Frequency is now sampled on-device at 1 Hz out of sysfs and temperature from the
  host at 0.2 Hz, both concurrent with the invocation. `thermal_zone*` is
  root-only, which is why temperature still costs a `dumpsys`.
  Two corollaries worth stealing: a sampler you spawn on the device keeps running
  after you kill the host-side `adb` client (adbd holds the pty open, so the loop
  never takes SIGPIPE), so it must be killed explicitly or it perturbs the very
  measurement it serves; and for a backgrounded pipeline `$!` is the *last*
  command, so killing it leaves the rest of the pipeline alive.
- **This affects both engines equally.** Under identical interleaved conditions
  cera measured 16.3% CoV against llama.cpp's 17.1%. An earlier revision claimed
  llama was ~2x noisier; that was a sampling window, not a property of either
  engine, and it is why a cross-engine decode ratio needs frequency-matched
  samples rather than more repetitions.
- **Record battery level and power state, and gate on level.** Android reduces
  peak clocks at low battery, so a session that drains while it measures compares
  its early cells against its late ones at different power budgets. The runs
  behind an earlier revision of this file drained 93% to 14% with the level
  recorded nowhere, which makes every cross-matrix comparison in that revision
  suspect independently of the thermal story. `bench_android.sh` now refuses to
  start below `--min-battery` (default 30) and writes `batt_start`, `batt_end`
  and a three-way `power_state` into every row. The state is three-way on
  purpose: "plugged in but not charging" is its own power envelope, because the
  charger is current-limiting, and folding it into a boolean hides a real
  difference in what the SoC may draw.
- **Battery temperature is not SoC temperature, and gating on it is useless.**
  Under load the BIG cluster reaches 74 C while the battery reads 23 C: a 0.5 C
  move on the sensor that is easy to read against a 48 C swing on the one that
  matters. Every thermal claim in an earlier revision of this file used battery
  temperature and was wrong for that reason. Read the live cluster temperatures
  from `dumpsys thermalservice`, and specifically from the "Current temperatures
  from HAL" section: the "Cached temperatures" section printed above it is stale
  and keeps reading hot long after the device has cooled.
- **Do not idle between measurements on Android.** It is the opposite of the
  right move. The device cools in about twenty seconds, so idling drops it back
  into the thermal transient and makes the next measurement unrepeatable. This
  is why the `--settle` option was removed: warm-up passes reach equilibrium
  without leaving it. Keep
  the load continuous and measure at equilibrium.
- **A ratio needs both engines measured in the same thermal regime.** At
  equilibrium both cera and llama.cpp sit at ~4.3% CoV and a 1.10x gap is 3.9
  sigma; measured cold, the same pair swings 9x and supports nothing.
- **The variance is between invocations, not within them.** On the same pinned
  config, `taskset 7c`, cera decode returned 60.5 / 94.0 / 75.7 across three
  matrices while its stddev *inside* each invocation stayed at 1.0-5.3. Two
  consequences: raising `--runs` cannot fix it (it samples the tight
  within-invocation distribution harder), and CPU affinity cannot either (the
  unstable configs are the pinned ones). The sampling unit has to be the whole
  invocation: repeat the matrix and report the spread across matrices. To make
  that diagnosable, `bench_android.sh` records the SoC big-cluster temperature
  range (`soc_big_min`/`soc_big_max`, read from the live HAL sensor) and the
  big-core clock range (`cpu_mhz_min`/`cpu_mhz_max`) for every measurement on
  both engines, plus `batt_start`/`batt_end`. If those ranges differ between two
  invocations that disagree, the drift is thermal; if they do not, it is not.
  The clock range is the sharper signal of the two, since a cell measured at a
  different frequency is not comparable however cool the SoC reads.
- **Cool the phone, and prove it stayed cool.** Thermal state dominates
  everything else on Android. The same matrix on the same commit gave best decode
  95.3 (battery 27.8 -> 28.8 C, TEC cooler) and 61.3 (27.6 -> 32.3 C, cooler
  ineffective). Record battery temperature before and after and discard runs where
  it climbs more than ~1-2 C.
- **Engine ordering is a thermal bias.** `scripts/bench_android.sh` used to run
  every cera config before llama, so without active cooling llama was measured on
  a hotter device, biasing every ratio in cera's favour. It now interleaves the
  two engines and reaches equilibrium with warm-up passes (`--equil-warm`,
  discarded) before the `--passes` it measures, rather than idling between
  cells. The Android table above was
  re-measured under the fixed script; any older Android number quoted elsewhere
  in the repo predates it and carries the bias.
- **Use the other engine as a control.** When a re-run moved cera decode 95.3 ->
  61.3 it also moved llama 91.2 -> 50.9. Since no cera commit can affect
  llama.cpp, that identified the run as thermally compromised rather than a
  regression, which a cera-only measurement could not have done.

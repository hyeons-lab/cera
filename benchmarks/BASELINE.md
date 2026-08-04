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

### Decode at equilibrium, LFM2.5-350M Q4_K_M

cera `b0ffd7e`, llama.cpp `f12cc6d0f` (9371), both pinned to the same five perf
cores, 6 invocations each at equilibrium (BIG 64-70 C), 512-token decode.

**Provisional: battery level was not recorded for this measurement** (it was
somewhere around 55-60%, reconstructed from ad-hoc probes, and falling). Battery
gating landed after it was taken. Treat the ratio as indicative and re-measure
with `batt_start`/`batt_end` populated before quoting it.

| Engine | Config | Decode median | CoV | spread |
|---|---|---:|---:|---:|
| **cera** | `taskset 7c` | **65.5** | 4.2% | 1.12x |
| llama.cpp | `-t 5`, `taskset 7c` | 59.3 | 4.5% | 1.13x |

**cera 1.10x**, a 6.2 tok/s difference against a 1.6 combined standard error,
so 3.9 sigma. It is a lower bound: cera decodes from a 128-token prompt while
`llama-bench`'s `tg` starts from an empty context, which disfavours cera.

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

## Reproduce

```bash
# Android (build first: cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu)
scripts/bench_android.sh --model LFM2.5-350M-Q4_K_M.gguf --serial <adb-serial> \
  --llama-bench /data/local/tmp/.../llama-bench --settle 30 --decode-prompt 128

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
  right move. The device cools in about twenty seconds, so a settle drops it back
  into the thermal transient and makes the next measurement unrepeatable. Keep
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
  that diagnosable, `bench_android.sh` records `batt_c_pre`/`batt_c_post` for
  every measurement on both engines, and cera's own `hr0`/`hrmax` thermal
  headroom (0=cool, 1.0=throttling) for its own. If `hr0` differs between two
  invocations that disagree, the drift is thermal; if it does not, it is not.
- **Cool the phone, and prove it stayed cool.** Thermal state dominates
  everything else on Android. The same matrix on the same commit gave best decode
  95.3 (battery 27.8 -> 28.8 C, TEC cooler) and 61.3 (27.6 -> 32.3 C, cooler
  ineffective). Record battery temperature before and after and discard runs where
  it climbs more than ~1-2 C.
- **Engine ordering is a thermal bias.** `scripts/bench_android.sh` used to run
  every cera config before llama, so without active cooling llama was measured on
  a hotter device, biasing every ratio in cera's favour. It now interleaves the
  two engines and defaults to `--settle 30`. The Android table above was
  re-measured under the fixed script; any older Android number quoted elsewhere
  in the repo predates it and carries the bias.
- **Use the other engine as a control.** When a re-run moved cera decode 95.3 ->
  61.3 it also moved llama 91.2 -> 50.9. Since no cera commit can affect
  llama.cpp, that identified the run as thermally compromised rather than a
  regression, which a cera-only measurement could not have done.

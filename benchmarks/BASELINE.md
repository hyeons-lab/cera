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
| Android CPU/GPU table below | `b0ffd7e` (incl. #342) | Pixel 10 Pro Fold, TEC-cooled |
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

## Android: Pixel 10 Pro Fold (Tensor G5), TEC-cooled

8 cores: 2 efficiency (cpu0-1) + 5 perf (cpu2-6) + 1 prime (cpu7).
CPU has `asimddp` + `i8mm`.

cera `b0ffd7e`, llama.cpp `f12cc6d0f` (9371), `scripts/bench_android.sh` with
separate prefill/decode runs and interleaved engines, 5 runs, `--settle 30`.
Battery 27.6 -> 30.2 C across the matrix. The llama rows carry no stddev because
this matrix predates the harness change that keeps `llama-bench`'s `±` value; the
next run will have it, which is what a decode claim here would need.

| Engine | Config | Prefill | Decode ‡ |
|---|---|---:|---:|
| cera CPU | default RowPool | **208 +/- 6** | 63.0 +/- 10.6 |
| cera CPU | pinned perf cluster (`taskset 7c`) | 93 +/- 2 | 75.7 +/- 2.9 |
| cera CPU | pinned prime (`taskset 80`) | 75 +/- <1 | 60.6 +/- 3.1 |
| cera GPU | wgpu / Vulkan | 92 +/- 1 | 20.7 +/- 0.3 |
| llama.cpp | `-t 6` pinned perf+prime | **209.2** | 53.6 |
| llama.cpp | `-t 1` pinned prime | 77.7 | 54.5 |
| llama.cpp | `-t 5` pinned perf | 161.3 | 48.6 |

**Prefill is a tie: 208 +/- 6 against 209.2.** This is best-config against
best-config, and cera's best is its unpinned `default RowPool` (which sizes
itself and is the shipping configuration) against llama pinned across perf+prime;
mask-matched, llama leads on every pinned pair. That is the headline correction of
this revision. The previous entry reported llama 1.50x ahead on prefill, which
was entirely an artifact: the harness measured cera with one combined
`--prompt-tokens 512 --max-tokens 128` invocation while `llama-bench` times its
`pp512` separately, and simply splitting the cera run moved it 149 -> 208.

**‡ Decode is not measurable on this device at n=5, for either engine.** Do not
quote a decode ratio from this table. Across five matrices llama's own `-t 5`
config, an unchanged binary at fixed settings, returned 62.7 / 60.8 / 91.2 / 43.4
/ 48.6 tok/s, a 2.1x spread; cera's best config moved 66.7 / 83.7 / 95.3 / 61.3 /
75.7 over the same matrices. Cooling flattens the trend but does not close this,
so the residual is scheduler placement on big.LITTLE rather than temperature.
Resolving it needs many more runs per config, not another matrix.

For the same reason no code delta is claimed against the previous `c6f845d`
entry: the decode column cannot support one, and the cooling regime changed too
(that entry says "on a fan").

llama.cpp's Vulkan build does run on this device's PowerVR GPU, and its prefill
(638) is the highest prefill measured on the device, but it is excluded above: it
logs `Compute pipeline creation failed for mul_mat_vec_q4_k_f32_f32`, decodes at
3.06 tok/s, and a sustained decode run wedged the phone into a `watchdog,apc`
reset. cera's wgpu row is the same GPU with neither failure, though at 20.7
decode it is far below its own CPU path, so GPU is not the right choice on this
device for either engine.

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

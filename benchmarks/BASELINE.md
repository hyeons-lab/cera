# Perf baseline

The reference every perf task (T1-T10) diffs against. Re-measure with the same
commands before claiming a delta — a throughput number without a matching
profile/counter movement is not evidence.

**Model:** LFM2.5-350M-Q4_K_M (the most common real-world quant).
**Settings:** prefill 512 / decode 128, `--no-cache`, medians (p50) over >=5 runs.

**Sections carry their own provenance — they were not all measured at once:**

| section | measured at | device |
|---|---|---|
| Android CPU/GPU table below | `c6f845d` (incl. #254) | Pixel 10 Pro Fold |
| GPU I/O counters | mixed — see the † note | Mac / Adreno |
| GPU decode profile | `0f00dec` (incl. #316, #318, #319, #320) | M1 Max |

**The Android GPU row has not been re-measured since `c6f845d`** and predates
every wgpu decode change listed above. Do not read it as current: the same work
is worth 1.78× on an M1 Max, and the Android number will have moved by some
unknown amount. Re-running it needs the device.

## Android — Pixel 10 Pro Fold (Tensor G5), on a fan

8 cores: 2 efficiency (cpu0-1) + 5 perf (cpu2-6) + 1 prime (cpu7).
CPU has `asimddp` + `i8mm`.

| Engine | Config | Prefill | Decode |
|---|---|---:|---:|
| **cera** CPU | default RowPool | **102** | **70.3** |
| cera CPU | pinned prime (`taskset 80`) | 113 | 66.1 |
| cera CPU | pinned perf cluster (`taskset 7c`) | 49 | 46.5 |
| **cera** GPU | wgpu / Vulkan (stale — see above) | 12 | 11.2 |
| llama.cpp | `-t 1` pinned prime | 170.6 | 73.3 |
| llama.cpp | `-t 5` pinned perf | 261.1 | **85.7** |
| llama.cpp | `-t 6` pinned perf+prime | **393.5** | 70.8 |

Best-vs-best: **decode** cera 70.3 vs llama 85.7 (llama 1.22x) ·
**prefill** cera 102 vs llama 393.5 (llama 3.9x).

llama.cpp b9980 (vendored via pipette-llamacpp). It has no Android GPU backend
in this runtime, so the GPU row is cera-CPU-vs-cera-GPU only.

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
  round trip regardless of payload — ~1.3 ms for a kernel doing 0.13 ms of work,
  and ~1.5 ms to move four bytes. Both now ride along in the output projection's
  submission. This is the opposite of T6 and does not contradict it: what is
  expensive is a *blocking* round trip that carries nothing, not per-layer
  pipelining.
- Greedy decode **does** sample on the GPU: the readback is a 4-byte token id, not
  vocab logits. That part was always true.
- **Prefill batching is gated on quantization, not platform.** The batched path
  requires every matmul weight to be `Q4_0`/`Q8_0`/`Q4_K`. A `Q4_K_M` file carries
  **11 Q6_K tensors**, which fails the check, so prefill **silently** falls back to
  the per-token loop — 8728 submits instead of 25. The same model does this on Mac
  too; it is not an Adreno effect (T8).
- **Prefill reads back ~12.9 MB per 512-token prompt** (23 readbacks), against
  decode's 4 bytes/token. Not negligible, and its own optimization target.
- **That readback volume is identical on both paths** — 23 readbacks and
  12,926,976 bytes whether prefill runs batched (Q4_0, 25 submits) or falls back
  (Q4_K_M, 8728 submits). So the readbacks are *not* a symptom of the fallback
  above; fixing the dtype gate will not touch them. Two independent problems.

Because of that gate, the Mac-vs-Adreno rows in earlier revisions of this doc
compared a **Q4_0** model on Mac against a **Q4_K_M** model on Adreno — i.e. the
batched path against the fallback path. Same-model gaps are **3.3x prefill** and
**2.3x decode**, not the 13x/4x reported before.

Caveat: `bench` decodes greedily (temp=0), which takes the on-GPU argmax path. The
**non-greedy** path still downloads full vocab logits per token — invisible to
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

- **The quantized GEMVs are still the bottleneck, and still the biggest remaining
  lever.** In production they sustain ~26 GB/s where the f16 kernel does ~93 GB/s
  on the same GPU. `ffn` alone is 48% of GPU pass time. The gap is in *how the
  bytes are read* — the quantized kernels fetch via scalar `u32` loads with
  shift/branch byte extraction, `gemv_f16` reads aligned vectors. #316 closed part
  of it by removing register spills, and the q6_k load pattern is now fixed too.

  **But the FFN's ~26 GB/s is not a load-pattern limit, and treating it as one
  overestimates the prize.** Ablation: deleting q4_k's redundant per-block header
  loads (all 32 threads re-read the same 4 words) is worth **+24% at m=65536 and
  exactly nothing at FFN shapes** — 13.3 vs 13.2 GB/s. Sweeping m for q4_k gives
  9.4 / 13.2 / 22.1 / 25.9 / 31.2 / 48.0 GB/s at m = 1024 … 65536, climbing
  monotonically and never plateauing: at FFN shapes there is a fixed per-dispatch
  cost the kernel never amortizes, so it is latency/occupancy-bound, not
  bandwidth-bound. Load work pays on the LM head and on nothing else in decode.

  Measure the kernels in isolation with `cargo run --release -p cera --features
  gpu --example wgpu_gemv_bench`. At `m=65536`, the bandwidth-bound row:

  | kernel | GB/s | vs f32 |
  |---|---:|---:|
  | f32 | **202** | — |
  | q4_0 | 113 | 56% |
  | q6_k | 113 | 56% |
  | q4_k | 48 | 24% |
  | q8_0 | 34 | 17% |

  **Still only trust the `m=65536` row.** The bench used to open a compute pass
  per iteration, charging ~38 µs of pass overhead to every measurement; that is
  fixed, and it moved q4_0 from 53 to 113 GB/s and f32 from 145 to 202 — the
  earlier figures in this file understated the kernels by up to 2.1x. Even
  corrected, the small-m rows report ~13 GB/s where a real FFN pass sustains ~26,
  for reasons not yet explained. Rotating output buffers to break write-after-write
  chains between iterations was tried and changed nothing, so that is not it.

  q6_k reaching parity with q4_0 is recent — it read weights a byte at a time,
  reloading the same word up to four times, until that was fixed. **q4_k and q8_0
  have not had the same treatment and are the obvious next targets**: q4_k is what
  every FFN matrix in a Q4_K_M model uses.

- **Decode is memory-bound, confirmed by A/B, not by inspection.** The same model
  at Q8_0 moves 1.89× the FFN bytes and took **2.10×** the FFN time. Time tracks
  bytes. An ALU/dequant bound was the obvious story from reading
  `gemv_q4_0_fast.wgsl` and it is **wrong** — Q8_0 is *cheaper* to unpack and got
  proportionally slower anyway.

- **~24% of decode wall time is now outside every GPU pass** (6.7 ms of passes vs
  8.9 ms wall), down from ~44%. #319 is what closed it, by removing two blocking
  GPU round trips from the tail. The earlier claim that this gap was "not
  recoverable by merging submits" was right about *submits* and wrong as a general
  conclusion — see `GPU_FINDINGS_CORRECTION.md`, round 3.

- **Compute-pass count is NOT a general lever — do not re-derive one from #318.**
  Merging the conv block's three passes into one was worth +17%, but merging
  passes in the attention/FFN path measured **neutral-to-negative**: a boundary
  there is worth ~6 µs, not the ~20–38 µs the conv result implied, and carrying
  those merges cost ~15%. Implemented, measured, reverted (#319). A cost-per-unit
  derived from one successful change describes that change, not the system.

- Not yet measured on Adreno. The access-pattern penalty is likely worse on a
  mobile tiler.

## Reproduce

```bash
# Android (build first: cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu)
scripts/bench_android.sh --model LFM2.5-350M-Q4_K_M.gguf --serial <adb-serial> \
  --llama-bench /data/local/tmp/.../llama-bench

# Android CPU profile (hotspot must move when a kernel task lands)
scripts/profile_android_cpu.sh --model LFM2.5-350M-Q4_K_M.gguf --mask 80

# Mac / desktop matrix
scripts/bench_matrix.sh

# Per-kernel GPU timestamps (prints a span table per forward pass, to stderr)
CERA_GPU_PROFILE=1 cera bench -m <model.gguf> --device gpu \
  --runs 1 --warmup 1 --max-tokens 4 --no-cache
```

## Known measurement traps

- **Don't trust an unpinned multithreaded run.** Free-scheduled runs on
  big.LITTLE were bimodal (llama `-t 4` came back `100 ± 99` pp512). Every
  number above is from a pinned config; the pinned re-runs are stable
  (llama `-t 1`: `73.3 ± 0.26`).
- **Warm the device.** cera's first Android numbers were ~15% low because runs
  1-2 were cold; `--warmup 2` fixed it. `bench` prints thermal headroom per run —
  if it climbs toward 1.0, the number is thermally limited, not a ceiling.
- **`--no-cache` matters.** Without it the KV prefix cache makes prefill look
  arbitrarily fast on repeat runs.

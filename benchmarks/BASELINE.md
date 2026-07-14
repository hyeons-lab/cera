# Perf baseline

The reference every perf task (T1-T10) diffs against. Re-measure with the same
commands before claiming a delta — a throughput number without a matching
profile/counter movement is not evidence.

**Model:** LFM2.5-350M-Q4_K_M (the most common real-world quant).
**Settings:** prefill 512 / decode 128, `--no-cache`, medians (p50) over >=5 runs.
**Commit:** `c6f845d` (main, incl. #254 Q4_K NEON int8-dotprod GEMV).

## Android — Pixel 10 Pro Fold (Tensor G5), on a fan

8 cores: 2 efficiency (cpu0-1) + 5 perf (cpu2-6) + 1 prime (cpu7).
CPU has `asimddp` + `i8mm`.

| Engine | Config | Prefill | Decode |
|---|---|---:|---:|
| **cera** CPU | default RowPool | **102** | **70.3** |
| cera CPU | pinned prime (`taskset 80`) | 113 | 66.1 |
| cera CPU | pinned perf cluster (`taskset 7c`) | 49 | 46.5 |
| **cera** GPU | wgpu / Vulkan | **12** | **11.2** |
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
| decode submits / token | **19.0** | **19.0** | **19.0** |
| decode readbacks / token | 1.0 | 1.0 | 1.0 |
| decode readback **bytes** / token | 4 | 4 | 4 |
| **prefill submits** (512-tok prompt) | **25** | **8728** | **8728** |
| prefill readbacks (512-tok prompt) | 23 | 23 | n/m |
| prefill readback **bytes** (512-tok prompt) | 12,926,976 | 12,926,976 | n/m |

`n/m` = not measured; the prefill readback counters postdate the Adreno run and
re-measuring needs the device.

Read:

- **Decode issues 19 submits per token** — one per layer, plus argmax. The forward
  pass is *not* batched into a single command buffer (T6).
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

## Reproduce

```bash
# Android (build first: cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu)
scripts/bench_android.sh --model LFM2.5-350M-Q4_K_M.gguf --serial <adb-serial> \
  --llama-bench /data/local/tmp/.../llama-bench

# Android CPU profile (hotspot must move when a kernel task lands)
scripts/profile_android_cpu.sh --model LFM2.5-350M-Q4_K_M.gguf --mask 80

# Mac / desktop matrix
scripts/bench_matrix.sh
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

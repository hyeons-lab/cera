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

These are the numbers that killed the "GPU is latency-bound on round-trips"
theory — see `GPU_FINDINGS_CORRECTION.md`. They are **identical** on Mac and
Adreno, while throughput differs 4-13x:

| | Mac (wgpu) | Android (Adreno) |
|---|---:|---:|
| decode submits / token | 1.0 | 1.0 |
| decode readbacks / token | 1.0 | 1.0 |
| decode readback **bytes** / token | 4 | 4 |
| prefill submits / prompt-token | 0.045 | 0.045 |

Read: the forward pass is **already one submit per token**, and greedy decode
**already samples on the GPU** (4-byte token-id readback, not vocab logits).
Prefill is **already batched**. So round-trip count is not the GPU bottleneck —
the Adreno *kernels* are. Any GPU change must therefore be justified by a
per-kernel timing (T5b), not by a round-trip count.

Caveat: `bench` decodes greedily (temp=0), which takes the on-GPU argmax path.
The **non-greedy** path still downloads full vocab logits per token — invisible
to these numbers by construction.

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

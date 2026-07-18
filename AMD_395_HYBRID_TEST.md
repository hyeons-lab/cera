# AMD 395 (Strix Halo) — hybrid prefill/decode + coopmat test brief

**For the AI agent running on the Windows AMD Ryzen AI Max+ 395 laptop.** This branch
(`perf/hybrid-prefill-decode`) adds a hybrid inference path (GPU prefill → CPU decode)
and a `bench --device hybrid` mode. It was validated on a Pixel (Tensor G5 PowerVR),
where the result was decisive: **the mobile GPU is slower than the CPU at everything, so
the hybrid has no benefit there.** The 395 is the opposite regime — a real RDNA 3.5 iGPU
+ UMA — so this is where we find out whether the GPU-prefill/CPU-decode crossover (and
GPU inference generally) actually pays off. Your job is to measure it cleanly.

## Background: what we already know (Pixel, LFM2-VL-450M-Q4_0)

| Mode | Prefill tok/s | Decode tok/s | Verdict |
|---|---|---|---|
| all-CPU | 131 | 90.7 | fastest end-to-end |
| all-GPU (wgpu) | 115 | 19.9 | GPU decode 4.5× slower than CPU |
| hybrid | 121 (GPU) | 71.1 (CPU) | no win — GPU prefill has no edge |

The hybrid only wins if **GPU prefill > CPU prefill AND CPU decode > GPU decode**. On the
Pixel the first condition failed (the PowerVR is ~30 GFLOP/s effective, weaker than 8 NEON
cores). **On the 395's RDNA 3.5 the GPU should be far stronger — so measure whether the
crossover finally exists.** Note: on a UMA APU, CPU and GPU share the ~256 GB/s LPDDR5X
bus, and the KV handoff is nearly free (no PCIe copy).

## The hypothesis to test on the 395

1. Does **GPU prefill beat CPU prefill** on the 395? (On the Pixel it didn't.)
2. Does **CPU decode still beat GPU decode**, or does the strong RDNA 3.5 iGPU win decode
   too (making all-GPU the right answer)?
3. Where does **hybrid** land vs all-GPU and all-CPU, end-to-end?

Likely outcomes: either (a) all-GPU wins outright (strong iGPU → keep everything on GPU),
or (b) hybrid wins (GPU prefill fast, CPU decode still faster for batch-1). Both are
useful findings. Report the numbers, don't assume.

## Build

Prereqs: Rust toolchain, a Vulkan SDK / recent AMD driver (RADV on Linux or AMD's Windows
Vulkan driver). The GPU backend is the `gpu` cargo feature (wgpu). Release build for real
numbers:

```powershell
# Windows (from the repo root, on branch perf/hybrid-prefill-decode)
cargo build --release -p cera-cli --features gpu
# binary: target/release/cera.exe
```
```bash
# Linux
cargo build --release -p cera-cli --features gpu   # target/release/cera
```

Force the Vulkan backend (so results are comparable to the Pixel's Vulkan path and to the
coopmat work below), otherwise wgpu may pick DX12 on Windows:

```
set WGPU_BACKEND=vulkan      # Windows cmd
$env:WGPU_BACKEND="vulkan"   # PowerShell
export WGPU_BACKEND=vulkan   # Linux
```

## Run the three-mode bench

Use the same small text LFM2 model the Pixel used, or any LFM2/dense GGUF you have. If you
need one, `LFM2.5-350M-Q4_0.gguf` or a 1.6B LFM2 is ideal. Run all three modes with
identical params:

```bash
MODEL=path/to/LFM2.5-350M-Q4_0.gguf
for DEV in cpu gpu hybrid; do
  echo "=== $DEV ==="
  ./cera bench -m "$MODEL" --device $DEV \
    --prompt-tokens 512 --max-tokens 64 \
    --runs 5 --warmup 1 --no-cache --context-size 4096
done
```

Then sweep prompt length to find the crossover — GPU prefill's advantage grows with prompt
size: repeat with `--prompt-tokens 128`, `512`, `2048`. And test a bigger model (1.6B) —
the GPU's edge grows with model size.

Add `--gpu-io` to the `gpu`/`hybrid` runs to see submit/readback counts (useful for
diagnosing whether prefill is batched — should be ~tens of submits, not thousands).

### What the hybrid mode reports

`--device hybrid` builds two engines (wgpu + CPU) on the same GGUF, prefills on wgpu, hands
off the KV+conv snapshot to the CPU session, and decodes there. It prints:
- `prefill tok/s (wgpu)` and `decode tok/s (CPU)`,
- `handoff` ms + snapshot MiB (the one-time KV readback cost),
- a **coherence** line: `hybrid vs all-CPU greedy match = N/M leading tokens`. On a *real*
  text prompt (use `--prompt "..."` instead of `--prompt-tokens`) this should be M/M — if
  it's low on a real prompt, the transplant is broken; investigate. (On synthetic
  `--prompt-tokens` a low match is expected — degenerate distribution, not a bug.)

## Data to collect + how to interpret

For each (model, prompt-length) report the table:

| Mode | prefill tok/s | decode tok/s | end-to-end for N prompt + M decode |
|---|---|---|---|

Compute end-to-end = prompt/prefill_tps + handoff + decode_count/decode_tps. The winner is
the mode with the lowest end-to-end (or split it: TTFT = prefill time; generation = decode).
Key questions to answer explicitly:
- Is GPU prefill faster than CPU prefill on the 395? By how much, and from what prompt
  length does it pull ahead?
- Does the 395 iGPU win decode, or does CPU still win batch-1 decode?
- Does hybrid beat both, and is the handoff cost (which shrinks on UMA) negligible?

Watch for: thermal (the bench prints headroom on some platforms; on the 395 watch clocks),
and whether the wgpu adapter picked is the iGPU (not a llvmpipe software fallback — check
the "Using wgpu backend" line and adapter logs with `RUST_LOG=cera=info,wgpu=warn`).

## Secondary test: coopmat / WMMA on RDNA 3.5

The other branch `experiment/vulkan-coopmat-gemm` has a standalone Slang→SPIR-V
cooperative-matrix GEMM microbench (`experiments/vulkan-coopmat/`, an `ash` crate). On the
Pixel's PowerVR it topped out at ~1.6× a tiled baseline and the driver corrupted anything
past 1 live MatA. **The 395's RDNA 3.5 has real WMMA matrix hardware and a mature driver, so
coopmat should shine here — this is the test that was impossible on mobile.** If you have
time after the hybrid bench:

```bash
git checkout experiment/vulkan-coopmat-gemm
cd experiments/vulkan-coopmat
cargo run --release --bin coopmat-probe    # dumps VkCooperativeMatrixPropertiesKHR — expect f16/bf16/int8 WMMA configs
cargo run --release --bin coopmat-gemm     # int8 coopmat GEMM vs tiled baseline, GFLOP/s + speedup
```
Report the probe's supported configs (RDNA 3.5 should expose f16→f32 and likely bf16/int8)
and the microbench speedup. If coopmat is a big multiple here (not 1.6×), that revives the
"Slang authors all kernels + coopmat GEMM" direction for desktop.

## Report back

Write your findings into a devlog entry and/or reply with: the three-mode tables per
model/prompt-length, the crossover answer (does GPU prefill beat CPU prefill and where),
the coopmat probe+speedup if run, and a recommendation (all-GPU vs hybrid vs all-CPU for
the 395). Be honest and numeric — a null result ("all-GPU just wins, hybrid pointless
here") is a valid and valuable outcome.

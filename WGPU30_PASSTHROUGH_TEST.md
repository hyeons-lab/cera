# wgpu-30 Slang passthrough: AMD 395 result + test instructions

> TEMPORARY file on the `experiment/wgpu-30-recheck` branch. Delete before any merge.

## What this branch is

Migrates the wgpu backend 24 -> 30. naga-30's WGSL->SPIR-V codegen regresses the
prefill reg-tile GEMM ~28% on the Pixel's PowerVR GPU. The fix: author the hot kernel
in Slang, compile to SPIR-V with slangc, and feed it straight to the Vulkan driver via
`create_shader_module_passthrough` (bypassing naga entirely). It is **default ON**
wherever the device supports `PASSTHROUGH_SHADERS` (Vulkan); Metal/WebGPU fall back to
naga automatically.

Ported loaders so far: **Q4_0** and **Q8_0** (use one of those quant types; a K-quant
model just runs naga and shows no difference).

## RESULT: AMD RDNA 3.5 (Radeon 8060S), 2026-08-01

**The PowerVR regression does not generalize. Keep passthrough default-on; do not
gate it to PowerVR.**

Measured on an AMD Ryzen AI MAX+ 395 (Radeon 8060S iGPU) over Vulkan, native
Windows, AMD proprietary driver 25.30.33.05 (LLPC), model LFM2.5-350M. Arms
interleaved with alternating order; one run discarded per case.

### Kernel level (`CERA_GPU_PROFILE=1`, ratio = naga / passthrough)

| span | ratio | independent sets |
|-------|-------|------------------|
| `mul_mat_tile` (Q4_0) | 1.038 - 1.052 | 3 |
| `mul_mat_q8_0` (Q8_0) | 1.079 - 1.088 | 2 |
| `attention_prefill` (control) | 1.001 - 1.003 | every set |

Passthrough is genuinely faster, roughly twice as much on Q8_0 as on Q4_0. In every
set the worst passthrough run still beat the best naga run. `attention_prefill` is
untouched by this change and stays within 0.3%, which is what makes the GEMM deltas
credible rather than a whole-GPU drift.

### End to end (prefill tok/s, mean of per-round p50, 4 rounds)

| model | passthrough | naga-30 | ratio |
|-------|------------:|--------:|------:|
| LFM2.5-350M-Q4_0 | 2,850 | 2,736 | 1.042x |
| LFM2.5-350M-Q8_0 | 2,764 | 2,646 | 1.044x |
| SmolLM-135M-Q4_0 | 2,878 | 2,780 | 1.035x |
| LFM2.5-350M-Q4_0 @ 2048 prompt | 1,155 | 1,140 | 1.013x |

Top-6 logits are bit-identical between arms, so the Slang port is equivalent.

Compare against the Pixel reference below: 1.53-1.61x there, 1.04x here.

### Why the end-to-end win shrinks at longer prompts

Prefill is chunked per ubatch (512) and the profiler resets per chunk, so one run
prints several `GPU Profile` blocks. GEMM cost per chunk is flat while attention
grows superlinearly with start position:

| chunk | GEMM share | attention share |
|-------|-----------:|----------------:|
| 1 | 58.2% | 39.1% |
| 2 | 28.4% | 70.0% |
| 3 | 18.3% | 80.8% |
| 4 | 13.0% | 86.3% |
| whole 2048-token prefill | 22.1% | 77.9% |

So a GEMM win dilutes as prompts lengthen. At long context **attention is the lever,
not the GEMM**, which is the more actionable finding here.

### What is NOT established

- At 512 tokens a 4% kernel gain over a 57% share predicts ~2.3% off total GPU time,
  but end-to-end measured 4.2%. The 2048 case reconciles (~1.6% predicted vs 1.3%
  measured); the 512 case does not. Roughly half the 512-token end-to-end gap is
  unattributed, so do not read the 1.04x as purely the GEMM.
- Absolute throughput on this host is not stable across a session. See the clock-state
  warning under "Gotchas".
- Only Q4_0 and Q8_0 are ported, and only LFM2 + SmolLM were exercised. No dense
  transformer at scale, no vision path.

## Device matrix (fill in as machines get tested)

One row per device. Two devices are done; the rest is the open work. Vulkan only:
on Metal and WebGPU the feature is unavailable and both arms run naga, so those
rows would read 1.00x for a reason that has nothing to do with codegen.

| device | GPU / driver | backend | Q4_0 kernel | Q8_0 kernel | end to end | verdict |
|--------|--------------|---------|-------------|-------------|-----------|---------|
| Pixel 10 Pro Fold | Tensor G5 / PowerVR | Vulkan | not measured | not measured | 1.61x / 1.53x | big win, motivated this branch |
| Ryzen AI MAX+ 395 | Radeon 8060S / AMD 25.30.33.05 | Vulkan | 1.038-1.052x | 1.079-1.088x | 1.035-1.044x | small real win |
| _(your machine)_ | | | | | | |

To add a row, run the A/B and the kernel timing below, then paste:

- **Q4_0 / Q8_0 kernel**: ratio of the `mul_mat_tile` / `mul_mat_q8_0` span,
  naga divided by passthrough. Include the `attention_prefill` control ratio in
  the verdict cell if it moved more than ~1%, since that invalidates the row.
- **end to end**: the `prefill tok/s` p50 ratio at 512 prompt tokens.
- **backend**: whatever the `GPU initialized` log line reports. If it is not
  Vulkan, the row is not measuring this change.

Discrete RDNA, Intel Arc, and Mesa/RADV on native Linux are the interesting gaps.
A second Vulkan device that shows ~1.0x or worse would reopen the gating question
that the AMD result currently closes.

## Reference numbers (Pixel 10 Pro Fold, Tensor G5 / PowerVR, cool)

| model | naga-30 prefill | passthrough prefill | ratio |
|-------|----------------:|--------------------:|------:|
| LFM2-VL-450M-Q4_0 | 75 tok/s | 121 tok/s | 1.61x |
| LFM2-VL-450M-Q8_0 | 58.5 tok/s | 89.5 tok/s | 1.53x |

## Reproducing, or testing another device

### Prereqs

- Rust toolchain (this branch pins `nightly-2026-07-10` via `rust-toolchain.toml`).
- A **Q4_0 or Q8_0** GGUF model.
- A Vulkan driver for the GPU (Mesa/RADV on Linux; AMD's driver on Windows).
- slangc is NOT required: the committed `.spv` fallbacks are used when slangc is absent.

### Build

```bash
git fetch origin
git checkout experiment/wgpu-30-recheck
cargo build --release -p cera-cli --features gpu
```

### Run the A/B (default = passthrough, `=0` = naga)

```bash
BIN=target/release/cera
M=/path/to/LFM2.5-350M-Q4_0.gguf

# passthrough (default):
$BIN bench -m "$M" --device gpu --prompt-tokens 512 --max-tokens 16 --runs 5 --warmup 2 --no-cache

# naga baseline:
CERA_WGPU_SPIRV_PASSTHROUGH=0 $BIN bench -m "$M" --device gpu --prompt-tokens 512 --max-tokens 16 --runs 5 --warmup 2 --no-cache
```

Compare the `prefill tok/s: ... p50=...` line between the two runs.

### Kernel-level timing

```bash
CERA_GPU_PROFILE=1 $BIN bench -m "$M" --device gpu --prompt-tokens 512 --max-tokens 2 \
  --runs 2 --warmup 1 --no-cache 2>&1 | grep -E 'mul_mat_tile|mul_mat_q8_0|attention_prefill'
```

Use `attention_prefill` as a control: it must stay flat between arms, since this change
does not touch it. If it moves, something else on the machine did.

### Confirm passthrough actually engaged

- **Perf differs** between the default and `=0` runs -> passthrough is active. If they are
  **identical**, it fell back to naga (non-Vulkan backend, or the driver does not advertise
  `PASSTHROUGH_SHADERS`).
- Or `RUST_LOG=cera=debug` and look for `mul_mat_reg_tile_q4_0: SPIR-V passthrough (slang)`
  at model load. That same log also prints the adapter name + backend at GPU init.

### Correctness check (should be bit-identical)

```bash
IDS=$(seq 100 611 | paste -sd, -)
$BIN logits -m "$M" --device gpu --token-ids "$IDS" --top-k 6
CERA_WGPU_SPIRV_PASSTHROUGH=0 $BIN logits -m "$M" --device gpu --token-ids "$IDS" --top-k 6
```

The two top-6 lists (token id + logit) should match exactly.

## Gotchas that cost real time

These all produced wrong or empty answers during the AMD run.

- **WSL2 cannot do this test.** No `/dev/dri`, so RADV enumerates nothing, and no dzn ICD
  to use `/dev/dxg`. Only lavapipe (software) comes up, which measures LLVM, not the GPU.
  Run from native Windows or native Linux.
- **On Windows, force the Vulkan backend**, otherwise wgpu may pick DX12, which has no
  `PASSTHROUGH_SHADERS`. Both arms then silently run naga and print two matching numbers
  that read as a genuine null result:
  ```powershell
  $env:WGPU_BACKEND="vulkan"
  ```
- **An absent `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` key does not mean Vulkan is missing.**
  Current AMD drivers register the ICD through PnP software-component properties. Use
  `vulkaninfo --summary` to check, not the registry.
- **`bench` reports on stderr**, so any capture needs `2>&1`.
- **The GEMM span label is per dtype**: `mul_mat_tile` for Q4_0 but `mul_mat_q8_0` for
  Q8_0. A grep for the former silently returns nothing on a Q8_0 model.
- **Sum the profile blocks, do not average them.** Prefill is chunked and the profiler
  resets per chunk, so averaging per-chunk share lines yields a meaningless number.
- **Absolute throughput drifted ~4x mid-session on the AMD host** and stayed there across
  12 consecutive in-process runs before recovering on its own. It was not the
  `WorkloadsSessionHost` GPU service (that sits at ~93% on the compute engine during both
  the fast and slow windows). Cause unknown. Interleaved A/B ratios survived it intact
  (1.038 slow vs 1.040 fast), so **interleave the arms and discard a warm-up run**; do not
  compare absolute numbers across sessions.

## Still open

- Whether the ~4% holds on discrete RDNA or on Intel Arc / Mesa RADV, all untested.
- Porting more loaders (K-quants) to the Slang path.
- Given attention is 78% of a 2048-token prefill here, whether `attention_prefill` is
  worth the same treatment.

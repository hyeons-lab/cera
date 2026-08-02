# wgpu-30 Slang passthrough: AMD 395 test instructions

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

The goal of this test: a first non-PowerVR Vulkan data point (AMD RDNA 3.5 / Radeon
8060S), does passthrough help, hurt, or is it neutral there?

## Prereqs

- Rust toolchain.
- A **Q4_0 or Q8_0** GGUF model (e.g. `LFM2-VL-450M-Q4_0`).
- A Vulkan driver for the iGPU (Mesa/RADV on Linux; AMD's driver on Windows).
- slangc is NOT required: the committed `.spv` fallbacks are used when slangc is absent.

## Build

```bash
git fetch origin
git checkout experiment/wgpu-30-recheck
cargo build --release -p cera-cli --features gpu
```

## Run the A/B (default = passthrough, `=0` = naga)

```bash
BIN=target/release/cera
M=/path/to/LFM2-VL-450M-Q4_0.gguf

# passthrough (default):
$BIN bench -m "$M" --device gpu --prompt-tokens 512 --max-tokens 16 --runs 5 --warmup 2 --no-cache

# naga baseline:
CERA_WGPU_SPIRV_PASSTHROUGH=0 $BIN bench -m "$M" --device gpu --prompt-tokens 512 --max-tokens 16 --runs 5 --warmup 2 --no-cache
```

Compare the `prefill tok/s: ... p50=...` line between the two runs.

### Windows

Force the Vulkan backend, otherwise wgpu may pick DX12 (no `PASSTHROUGH_SHADERS`, silent
naga fallback -> no difference):

```powershell
$env:WGPU_BACKEND="vulkan"
```

## Confirm passthrough actually engaged

- **Perf differs** between the default and `=0` runs -> passthrough is active. If they are
  **identical**, it fell back to naga (non-Vulkan backend, or the driver does not advertise
  `PASSTHROUGH_SHADERS`).
- Or `RUST_LOG=cera=debug` and look for `mul_mat_reg_tile_q4_0: SPIR-V passthrough (slang)`
  at model load. That same debug log also prints the adapter name + backend at GPU init.

## Optional: correctness check (should be bit-identical)

```bash
IDS=$(seq 100 611 | paste -sd, -)
$BIN logits -m "$M" --device gpu --token-ids "$IDS" --top-k 6
CERA_WGPU_SPIRV_PASSTHROUGH=0 $BIN logits -m "$M" --device gpu --token-ids "$IDS" --top-k 6
```

The two top-6 lists (token id + logit) should match exactly.

## What to send back

1. The two `prefill tok/s` p50 numbers (default vs `=0`).
2. The adapter line from `RUST_LOG=cera=debug` (GPU name + backend), to confirm it ran on
   the AMD iGPU over Vulkan.
3. Whether the optional logits check matched.

### How to read it

- **default ~= `=0`**: naga-30 is fine on RDNA (regression is PowerVR-specific); default-on
  is harmless.
- **default > `=0`**: naga regresses on RDNA too; broader win.
- **default < `=0`**: Slang codegen is worse for AMD's compiler; gate passthrough to PowerVR
  instead of all-Vulkan.

## Reference numbers (Pixel 10 Pro Fold, Tensor G5 / PowerVR, cool)

| model | naga-30 prefill | passthrough prefill | ratio |
|-------|----------------:|--------------------:|------:|
| LFM2-VL-450M-Q4_0 | 75 tok/s | 121 tok/s | 1.61x |
| LFM2-VL-450M-Q8_0 | 58.5 tok/s | 89.5 tok/s | 1.53x |

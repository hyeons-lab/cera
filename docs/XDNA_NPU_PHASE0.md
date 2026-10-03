# AMD XDNA NPU: Phase 0 and 0b checklist (native Windows)

Run this on the Ryzen AI Max+ 395 (128 GB) under native Windows. WSL2 cannot reach the NPU and distorts GPU numbers, so it is not used for any measurement here. Write every result into a results file (a table per section below, with the environment from section 0 attached); a number without its environment is not a result.

This is the first stage of the AMD XDNA backend effort. Phase 0 measures the CPU and iGPU baselines and sets the go/no-go bar for an NPU backend. Phase 0b decides whether such a backend is feasible at all: whether usable kernels exist and can be redistributed, and whether Windows lets us load them. No backend engine code is written before 0b passes. If it fails, nothing here ships beyond the vendor-neutral `--device npu` selection.

## 0. Record the environment (once, and again after any driver change)

- Windows build number, BIOS version.
- **Firmware UMA carve-out** for GPU/NPU-visible memory. It changes what fits and what bandwidth is reachable.
- NPU driver version (Device Manager, `amdnpu`), Ryzen AI Software version, XRT version if installed, GPU driver version.
- Power plan, and whether the machine is on AC. Use the same for every run.
- Power measurement tool, chosen before measuring (vendor tool or HWiNFO-class). Note which sensor it reads (package, NPU, whole system). Runs are only comparable with the same sensor.
- Idle package power for 60 s before each series, as the floor to subtract or report.

## 1. Confirm cera builds and runs on Windows

1. Install the MSVC toolchain and `rustup` (`rust-toolchain.toml` pins the version).
2. `cargo build --release -p cera-cli`
3. `cargo test -p cera --lib`
4. `cargo run --release -p cera-cli -- inspect --model <path>.gguf`

If anything fails here, fix or report it before attributing it to the NPU work. Known Windows risks: `gguf.rs` mmap and path handling, and the repo's `.cargo/config.toml` has no `[target.x86_64-pc-windows-msvc]` section, so Windows builds get no `target-cpu` flags. That is fine for the AVX2 and AVX-512 kernels, which are `#[target_feature]`-gated with runtime dispatch, but record whether the CPU tier selected is AVX-512 (see `cpu_features.rs`) and whether `RUSTFLAGS="-C target-cpu=native"` changes the result.

## 2. Phase 0 baselines

Models (copy into `~/.leap/models` equivalents): `Llama-3.2-1B-Q4_0.gguf` and `Llama-3.2-1B-Instruct-Q8_0.gguf`. A tiny `.gguf` is a failed download, not a model.

```
cargo run --release -p cera-cli -- bench --model <model.gguf> --device cpu --prompt-tokens 512 --max-tokens 128 --runs 20
cargo run --release -p cera-cli -- bench --model <model.gguf> --device gpu --prompt-tokens 512 --max-tokens 128 --runs 20
```

Run each combination of {model} x {`cpu`, `gpu`}, plus a long-prompt run (for example `--prompt-tokens 2048`) to see the prefill side. Record prefill tok/s, decode tok/s (p50, plus p10 and p90), and package power over the run.

Rules for trustworthy numbers:

- **Load a fresh model per measurement.** GPU models keep KV, conv state and prefix cache on the model, and reuse has produced false results before. `bench` loads once per invocation, so use one invocation per measurement.
- Check host load before and after; background load swings decode by tens of percent. Interleave A/B runs and report ratios, not single absolutes.
- Watch thermals on the Flow Z13 form factor; discard runs after throttling.
- Set `CERA_GPU_HOST_PROFILE=1` once to print the adapter name and **which wgpu backend was chosen** (Vulkan or DX12).

### Vulkan versus DX12 (needs a small code change first)

`GpuContext::new_async` in `cera/src/backend/wgpu.rs` builds the instance with `Backends::all()` and `BackendOptions::default()` and does not read `WGPU_BACKEND`, so the backend cannot be forced today. This matters: the Slang SPIR-V passthrough is Vulkan-gated, so DX12 runs the naga WGSL path, and the two can differ a lot. Before comparing them, land a minimal change that honors the standard wgpu backend environment override (for example via `wgpu::Backends::from_env()`), then run the `gpu` rows once per backend. Until then, report the one backend that `CERA_GPU_HOST_PROFILE=1` shows.

### Go/no-go bar

From the table above, set the NPU bar and record it in the results file. Starting proposal: continue only if the NPU reaches at least 50% of the iGPU's decode tok/s at 2x its tokens per joule or better. Adjust with the real numbers; get the bar agreed before Phase 1.

## 3. Phase 0b: feasibility gate

### 3a. License audit (no hardware needed)

For each candidate kernel source, answer in the results file: may it be redistributed in an Apache-2.0 OR MIT project, in binary `.xclbin` form, with what attribution?

- IRON and MLIR-AIE (stated Apache license; confirm the exact license of the operator library and of any bundled third-party kernels or headers).
- FastFlowLM / Lemonade kernels (closed-source binaries; confirm terms in writing, otherwise treat as reference only).
- Ryzen AI Software redistributables.

### 3b. Inventory what Windows exposes

- Is XRT for Windows installed with Ryzen AI Software, and which DLLs and headers exist (note the exact file names; the effort so far assumes an `xrt_core` library but that is unverified)?
- Can an arbitrary `.xclbin` be loaded through that XRT, or only vendor-packaged graphs (ONNX Runtime Vitis AI EP, FastFlowLM)? This is the most important single answer in the whole effort.
- Which weight dtypes do the available GEMM kernels accept (INT8, bfp16, bf16, native 4-bit or `mx4`)? This decides whether Q4_0 weights stay at 4 bits in memory or widen to 8.

### 3c. Spike

1. Build one IRON GEMV `.xclbin` on Linux (WSL2 on the same machine is fine for building, not for running). Pin the toolchain version.
2. On Windows, load it through raw XRT with weights repacked into page-aligned buffers and run it against a known input.
3. Measure **per-run dispatch latency** and **runlist / hardware-context chaining** for a sequence of dependent runs. This replaces the calculated, unmeasured "10 to 30 microseconds per dispatch" figure and decides how many runs a token needs.
4. Compare the effective weight-streaming bandwidth to the iGPU's from step 2. The working hypothesis is that the NPU's DRAM bandwidth is well below the iGPU's (unverified); this settles it.

### 3d. Decision

Write one of these into the results file, with the evidence:

- **(a)** raw XRT works on Windows: continue Windows-only.
- **(b)** only a vendor runtime is available: evaluate binding to it versus stopping.
- **(c)** raw access needs the open `amdxdna` stack: provision native Linux (spare NVMe, kernel 6.14 or newer, `/dev/accel/accel0` present, UMA set for 128 GB) and re-run the Phase 0 baselines there. Do not compare Windows and Linux numbers directly.
- **Stop:** the license, kernel availability or the go/no-go bar fails. Delete the `xdna` work; the unified `Npu` preference (Phase A) stands on its own.

The 0b exit criteria also include the written weight-format, activation-quantization and parity-tolerance decisions (the weight target format, how activations are quantized, and a tie-tolerant top-k parity policy, never bit-exact against the CPU; one repack must serve both decode and prefill).

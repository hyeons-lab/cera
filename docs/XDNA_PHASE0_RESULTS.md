# AMD XDNA NPU: Phase 0 and 0b results

Results for the checklist in [XDNA_NPU_PHASE0.md](XDNA_NPU_PHASE0.md). Status: **Phase 0b decided.** The deciding question is answered (yes, 3c step 2), 3d evaluates the go/no-go bar against the section 2 baselines, and 3e records the follow-up Q4 kernel spike and the decision taken on it. Phase 0 items that are still open (the CPU tier, `target-cpu=native`, DX12) are marked open, not skipped.

Sizes of buffers and transfers are binary (KiB, MiB); bandwidth is decimal GB/s (10^9 bytes per second); model file sizes (773 MB) are decimal. All measurements ran natively on Windows (the IRON kernels were built in WSL2; see 3c step 2). They were launched from a WSL2 shell through Windows interop, which starts the Windows executable as a normal Windows process. No measurement here ran inside the Linux VM.

## 0. Environment

Recorded 2026-10-03.

| Item | Value |
|---|---|
| Machine | ASUS ROG Flow Z13 (GZ302EA), AMD Ryzen AI Max+ 395 w/ Radeon 8060S |
| Windows | 11 Pro, build 26200 |
| BIOS | GZ302EA.311 |
| Memory visible to the OS | 95.6 GiB (`Win32_ComputerSystem`); xrt-smi reports 97943 MiB (its "MB") |
| Firmware UMA carve-out | Not read from firmware. 128 GiB installed minus about 96 GiB visible suggests about 32 GiB reserved; confirm in the BIOS before relying on it |
| NPU device | "NPU Strix Halo", BDF `00c5:00:01.1` |
| NPU driver | 32.0.203.311 (`kipudrv.inf`, published as `oem205.inf`) |
| NPU firmware | 1.0.21.44 |
| XRT | 2.19.0, hash `77c7088d804602a53c3eb489b9cb37b709bcd751` (2025-10-02), shipped inside the NPU driver package |
| Ryzen AI Software | Not installed |
| GPU driver | 32.0.31032.1003 (Radeon 8060S) |
| Power plan | The ASUS operating modes appear as Windows power plans. Everything measured on 2026-10-03 (sections 1, 3b and 3c, and the first baseline matrix) ran in **"performance"** (`27fa6203-3987-4dcc-918d-748559d549ec`), except the last run of that day's decode-dominant set, which hit the "Silent" cap (section 2). From 2026-10-12 everything ran in **"Turbo"** (`6fecc5ae-f350-48a5-b669-b472cb895ccf`), chosen for the remaining measurements. xrt-smi reports NPU power mode "Performance" in both |
| Power limits | Not readable directly; inferred from HWiNFO's percent-of-limit readings (power divided by percent used). Turbo: PPT FAST about 85 W, PPT SLOW about 67 W, STAPM about 86 W. "Silent" capped the package at about 15 W (PPT and STAPM at about 97% used, thermal limit 56%, on AC) |
| Sleep | The Turbo plan turns the display off after 300 s without input and the machine then enters Modern Standby; a running benchmark does not prevent it. On 2026-10-12 the event log shows three Modern Standby periods: 08:37:43 to 09:17:05, which cost one baseline measurement (redone), and 10:26 to 10:46 and 10:55 to 11:03, which interrupted early 3e runs of `spike.exe` launched directly rather than under the sampler (those runs are not reported). After the first one, `hwinfo_sm.py` holds `SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED \| ES_DISPLAY_REQUIRED)` while it measures, and sleep on AC was set to never at about 11:05. Every reported measurement ran outside those periods |
| Power source | AC (battery status 2, 80% charge at the time) |
| Power measurement tool | HWiNFO 8.54-6090 (free edition), read through its shared-memory interface by `tools/xdna-spike/hwinfo_sm.py` about 9 times a second (it waits 0.1 s after each full read). HWiNFO itself updated the values about every 1.97 s (its default polling period; a 250 ms setting did not survive an HWiNFO restart), so every measurement runs about 45 s or longer (at least about 22 updates; the shortest was 44.6 s) |
| Sensor used | **CPU Package Power** (`CPU [#0]: AMD RYZEN AI MAX+ 395: Enhanced`). See the inventory below for why |
| Idle package power | Depends heavily on what else the machine is doing, so every energy comparison below uses a floor measured next to it. Performance mode, browser closed, machine in use: **18.06 W** (16.81 to 24.67; Total System Power 16.57, SoC 5.45, Core+SoC 6.94). Same with a browser open: 22.82 W. Turbo, machine in use: 21.70 W (19.55 to 26.32). Turbo, machine left alone: **12.65 and 13.47 W** around the read-only kernel and int16 GEMV runs, **14.85 and 15.47 W** around the Q4 GEMV run (3e), **16.52 and 16.20 W** around the GPU rerun |
| Host load during baselines | Brave closed; about 19 `msedgewebview2.exe` processes (embedded web views in other apps) stayed running |

### Power sensors HWiNFO exposes on this machine

Read on 2026-10-03 through shared memory (15 sensors, 331 readings):

- **Power readings:** CPU Package Power; per-core power (Core 0 to 15); CPU Core Power (SVI3 TFN); CPU SoC Power (SVI3 TFN); Core+SoC Power (SVI3 TFN); APU STAPM; Total System Power.
- **NPU** (`NPU [#1]: AMD NPU 8x8`): NPU Clock (1810 MHz at the time), NPU Core Voltage, NPU D3D Usage, NPU Computing 0 Usage, NPU D3D Memory Dynamic. **No NPU power reading.**
- **iGPU** (`iGPU [#0]: AMD Radeon 8060S`): clocks (core, memory, SoC, VCN), core voltage, utilization, temperature. **No GPU power reading.**
- **Battery:** charge rate reads 0 W on AC, so it cannot stand in for wall power.

There is no per-device rail for the NPU or the iGPU, so neither can be measured in isolation. CPU Package Power is used because it covers the whole APU. That it includes the iGPU is inferred, not documented: GPU decode runs, whose CPU threads mostly wait on the GPU, raise it from the 18 W idle floor to 44 to 50 W, about as much as CPU runs do. "Total System Power" reads **below** package power (16.6 against 18.1 W at idle), so it is a firmware estimate, not wall power, and is recorded only as a cross-check. APU STAPM is a smoothed limit-tracking value and is not used for per-run energy.

## 1. cera on Windows

| Step | Result |
|---|---|
| Toolchain | `nightly-2026-07-10-x86_64-pc-windows-msvc`, picked up from `rust-toolchain.toml` |
| `cargo build --release -p cera-cli` | Pass. The binary is `cera.exe` |
| `cargo test --release -p cera --lib` | 1162 passed, **2 failed**, 16 ignored (see below) |
| `cera inspect --model Llama-3.2-1B-Instruct-Q4_0.gguf` | Pass: 147 tensors, 35 metadata keys, `llama`, 16 blocks. GGUF mmap and Windows path handling work |
| GPU build | `cargo build --release -p cera-cli --features gpu` passes. The CLI's default feature set is empty, so a plain build has no GPU backend; all `--device gpu` numbers here use the `gpu` build |
| wgpu backend chosen | **Vulkan**: `CERA_GPU_HOST_PROFILE=1` prints `adapter=AMD Radeon(TM) 8060S Graphics backend=Vulkan SUBGROUP=true SUBGROUP_BARRIER=true TIMESTAMP=true SHADER_F16=true` |
| CPU tier selected | Open |
| `RUSTFLAGS="-C target-cpu=native"` effect | Open |

The two test failures are Windows portability problems in the tests, not in the engine:

- `model::lfm2::no_repack_tests::no_repack_skips_cpu_repacks` panics at its `std::env::var("HOME").expect("HOME unset")` (in `lfm2.rs`). Windows sets `USERPROFILE`, not `HOME`. On a machine without the model this test is meant to skip, but it panics before reaching the skip check.
- `engine::tests::resolution::resolve_all_manifest_files_walks_every_field` fails `assert_eq!(manifest.files.model, "/models/bundles/model.gguf")` (in `engine.rs`). The cause is inferred, not confirmed from the panic output: the expected value is a string with `/` separators, while the resolved path is built with `Path::join`, which uses `\` on Windows.

### Models

`Llama-3.2-1B-Q4_0.gguf` (the base model named in the checklist) has no recorded download source in the repo, and the repo copy is unusual: its blocks 0 and 1 are Q4_1 (see `cpu.rs`). The Instruct Q4_0 from the same publisher as the Q8_0 stands in for it. Q4_0 numbers here are therefore **not** directly comparable to Mac numbers taken on the repo's base Q4_0 file.

| File | Source | Size (bytes) | SHA-256 |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-Q8_0.gguf` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 1321083008 | `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3` (matches `scripts/fetch_test_models.sh`) |
| `Llama-3.2-1B-Instruct-Q4_0.gguf` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 773025920 | `fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8` |

## 2. Phase 0 baselines

Recorded 2026-10-03 (Performance) and 2026-10-12 (Turbo). `cera bench`, one fresh process per measurement, package power sampled over that process by `tools/xdna-spike/hwinfo_sm.py` (driver: `tools/xdna-spike/baselines.sh`). GPU rows use the Vulkan backend; the `WGPU_BACKEND` override for a DX12 comparison has not landed. Models are the Instruct Q4_0 and Q8_0 files from section 1.

### Matrix: 128 decode tokens after a 512- or 2048-token prompt, 20 runs per process, two interleaved rounds

Decode tok/s p50 (round 1 / round 2):

| | Performance, GPU | Performance, CPU | Turbo, GPU | Turbo, CPU |
|---|---|---|---|---|
| Q4_0, 512-token prompt | 68.3 / 64.2 | 57.6 / 54.7 | 78.3 / 78.5 | 57.0 / 56.2 |
| Q8_0, 512-token prompt | 27.3 / 27.2 | 41.8 / 42.8 | 47.6 / 47.4 | 43.0 / 43.3 |
| Q4_0, 2048-token prompt | 40.3 / 39.1 | 46.4 / 47.8 | 46.9 / 47.2 | 48.0 / 49.1 |
| Q8_0, 2048-token prompt | 21.7 / 21.1 | 36.9 / 37.0 | 34.1 / 34.2 | 38.8 / 37.7 |

Prefill tok/s p50 (round 1 / round 2):

| | Performance, GPU | Performance, CPU | Turbo, GPU | Turbo, CPU |
|---|---|---|---|---|
| Q4_0, 512-token prompt | 1615 / 1414 | 256 / 255 | 2446 / 2415 | 257 / 258 |
| Q8_0, 512-token prompt | 1083 / 991 | 217 / 218 | 1971 / 1895 | 223 / 222 |
| Q4_0, 2048-token prompt | 1002 / 1015 | 235 / 234 | 1806 / 1793 | 240 / 239 |
| Q8_0, 2048-token prompt | 794 / 796 | 201 / 200 | 1498 / 1495 | 206 / 205 |

Mean package power over each process: 41 to 50 W in Performance, 52 to 70 W in Turbo. The full p10, p90 and power per row are in the logs `baselines.sh` writes. Round 1 and round 2 agree within about 6.5% on every decode row. Prefill agrees within 4% except on two Performance GPU rows, which differ by up to 14% (Q4_0: 1615 / 1414; Q8_0: 1083 / 991).

What this says:

- **The GPU depends on the power envelope; the CPU barely does.** Turbo raises GPU decode by 15 to 75% and GPU prefill by 50 to 91%, while CPU numbers move -1 to +5%. GPU Q8_0 decode in particular goes from 27 to 47 tok/s, so its weakness in Performance mode was mostly the envelope, not only the kernel.
- **GPU decode is also sensitive to what else the machine is doing.** The same Turbo GPU Q4_0 decode-dominant run measured 73.7 tok/s with the machine in use and **93.4 tok/s** with the machine left alone (below). Rows are comparable with each other within one session, not across sessions.
- **cera's GPU path is far from the iGPU's memory limit.** 93.4 tok/s at about 773 MB per token is about 72 GB/s of weight traffic, against a theoretical 256 GB/s for the board's memory (a specification, not a measurement).

### Decode-dominant runs, for energy per token

16-token prompt, 512 decode tokens, 10 runs per process; decode is 73 to 83% of each process's wall time.

| Turbo, machine in use (floor 21.70 W) | Decode tok/s p50 | Package W (mean over the process) | tok/J gross | tok/J net of floor | Net J per GB of weights |
|---|---|---|---|---|---|
| GPU Q4_0 | 73.7 | 66.5 | 1.11 | 1.65 | 0.79 |
| GPU Q8_0 | 51.2 | 68.0 | 0.75 | 1.11 | 0.68 |
| CPU Q4_0 | 60.3 | 65.6 | 0.92 | 1.37 | 0.94 |
| CPU Q8_0 | 44.5 | 57.2 | 0.78 | 1.25 | 0.60 |

The Performance-mode decode-dominant set from 2026-10-03 is not reported: its last run (CPU Q8_0) hit the Silent mode's 15 W cap (25.1 tok/s at 15.0 W), so the mode changed partway through that set. The matrix earlier the same day drew 41 to 50 W on every row, which the 15 W cap rules out, so its Performance columns stand.

### NPU power, and the like-for-like comparison

Turbo, machine left alone, each load next to 60 s idle floors. HWiNFO has no NPU or GPU power rail, so each device's cost is read as package power, gross and above the adjacent floor. Power is the **steady-state** mean: only the samples where the device under test reports at least 90% usage (HWiNFO's "NPU Computing 0 Usage" or "GPU Utilization"), which excludes host-side setup and, for the GPU, model loading. NPU throughput is bytes per run over the mean run time (the mean and p50 run times agree within 0.1% for these runs); the GPU's is decode tok/s p50 times 773 MB per token. All floors show the NPU at 0% usage.

The published rows come from three sessions on 2026-10-12: the read-only kernel and the int16 GEMV ran back to back between one pair of floors (12.65 W before, 13.47 W after; both rows use their mean), the Q4 GEMV of 3e between its own pair (14.85 and 15.47 W), and the GPU between a third pair (16.52 and 16.20 W). `tools/xdna-spike/npu_power.sh npu` reruns the three NPU loads with a floor between each pair of loads, and `npu_power.sh gpu` reruns the GPU reference (`cera bench --device gpu --prompt-tokens 16 --max-tokens 512 --runs 10` on the Q4_0 model). Both use `hwinfo_sm.py --steady`.

| Load (about 60 s, back to back) | Throughput | Steady package W | Adjacent floor W | Net W | J per GB gross | J per GB net |
|---|---|---|---|---|---|---|
| GPU, cera Q4_0 decode, 93.4 tok/s (580 samples) | 72.2 GB/s of weights | 67.25 | 16.36 | 50.89 | 0.931 | 0.705 |
| NPU, read-only kernel (`readbw`, 256 MiB per run, n = 12000, 494 samples) | 55.4 GB/s | 23.02 | 13.06 | 9.96 | 0.416 | 0.180 |
| NPU, int16 GEMV (8192 x 2048, 8 cores, n = 30000, 514 samples) | 16.95 GB/s | 20.46 | 13.06 | 7.40 | 1.207 | 0.437 |

The read-only kernel reached 55.4 GB/s in Turbo, above its 45 to 49 GB/s in Performance, so the NPU's data path also depends on the envelope.

An earlier version of this table divided whole-process mean power by p50 throughput (GPU 64.18 W, read-only kernel 22.64 W, GEMV 20.22 W), while 3e used steady-state power. Using one method throughout moves the GPU to 1.39 tok/J gross (from 1.46) and the read-only kernel to 0.416 J/GB (from 0.409); no conclusion changes.

Projected onto Llama-3.2-1B Q4_0 (773 MB per token), adding the measured floor of about 112 dependent dispatches x 46 us = 5.2 ms per token. That floor comes from the 288 x 288 chain in 3c, measured in the **Performance** plan, while the bandwidth and power here are Turbo; the dispatch floor in Turbo was not measured. Weights and dispatch only, with no other compute, so these are **ceilings, not demonstrated**:

| NPU rate used | Decode | Share of GPU (93.4) | tok/J gross (GPU 1.39) | tok/J net (GPU 1.84) |
|---|---|---|---|---|
| Read-only kernel, 55.4 GB/s | about 52 tok/s | 56% | 2.27 (1.63x the GPU) | 5.24 (2.86x) |
| int16 GEMV, 16.95 GB/s | about 20 tok/s | 21% | 0.96 (0.69x) | 2.66 (1.45x) |

Gross is the honest number for battery life: the floor is paid for every second the work takes, and a slower device pays it for longer.

## 3a. License audit

Recorded 2026-10-03, from upstream repositories and official pages. [P] means read in a primary source (a LICENSE file, file headers, an official EULA or terms page); [I] means inferred.

| Candidate | License (pinned) | Ship as `.xclbin` plus `insts.bin` in an Apache-2.0 OR MIT project? | Obligations | Confidence |
|---|---|---|---|---|
| Xilinx/mlir-aie, including IRON and `aie_kernels/` | Apache-2.0 WITH LLVM-exception, [LICENSE at `20c0463`](https://github.com/Xilinx/mlir-aie/blob/20c0463afeaaa52cb94e5ebaa88c3197cd135cd7/LICENSE). `aie_kernels/` has no separate license; 126 files carry the Apache SPDX header and 3 carry MIT [P] | Yes, with conditions | The Apache text and AMD/Xilinx copyright lines, plus the MIT text for the 3 MIT headers. No NOTICE file exists [P] | High |
| `aie_api` headers (submodule `third_party/aie_api`) | MIT, [LICENSE at `b042222`](https://github.com/Xilinx/aie_api/blob/b0422227afd25e9b47930624c857f384b500b524/LICENSE); all 254 headers carry the MIT tag [P] | Yes, with conditions | The MIT notice (header templates are compiled into the kernel) | High |
| Peano (Xilinx/llvm-aie) | Apache-2.0 WITH LLVM-exception, [LICENSE.TXT at `94b320c`](https://github.com/Xilinx/llvm-aie/blob/94b320c083c7f403992161d8b92ba7bedd4f8f24/LICENSE.TXT) [P] | Yes | None for compiler output. The linked `crt0`, `libc`, `libm` and compiler-rt builtins are all under the same license, whose exception waives 4(a), 4(b) and 4(d) for compiled-in portions [P] | High |
| amd/IRON | Apache-2.0, [LICENSE at `b73e3f8`](https://github.com/amd/IRON/blob/b73e3f8bb6cdcd6e15cc8ad70e6845dc4807ceec/LICENSE) [P] | Yes, with conditions | The Apache text; mark modified files (4(b)) | High |
| amd/RyzenAI-SW | MIT, [LICENSE.txt at `43b2dab`](https://github.com/amd/RyzenAI-SW/blob/43b2dabe4d1bf084d0421953b134707b8cb7275a/LICENSE.txt). Example apps only; no xclbins, kernel sources or DLLs in any of its 669 paths [P] | Not applicable | MIT notice if code is copied | High |
| FastFlowLM (ROCm/FastFlowLM) | Runtime code MIT; NPU kernels proprietary binaries, "NOT open source", per [TERMS.md](https://github.com/ROCm/FastFlowLM/blob/6c9ed874908e4b793014a87bce4b43af644ef0db/TERMS.md) [P]. The README at the same commit says the kernels are free for any use, contradicting TERMS.md [P] | **No** | No redistribution grant exists in either document | High that the kernels are closed; Medium on current terms |
| Lemonade | Apache-2.0, [LICENSE at `7dc09e2`](https://github.com/lemonade-sdk/lemonade/blob/7dc09e21a2655f0644963dd6cf857bac2d8ba0af/LICENSE). Ships no NPU kernels of its own [P] | Not applicable | Apache text if code is copied | High |
| Ryzen AI Software redistributables, and the driver's own XRT DLLs and AMD-built xclbins | AMD proprietary EULA ([licensing page](https://ryzenai.docs.amd.com/en/latest/licenses.md)). The EULA PDF itself could not be downloaded; the verdict rests on a copy of the generic AMD Software EULA [P, secondary] | **No**; users install the driver themselves | Not applicable if not bundled | Medium |
| Xilinx/XRT headers | User space Apache-2.0; only the Linux kernel drivers are GPL-2.0, [LICENSE at `2bfd112`](https://github.com/Xilinx/XRT/blob/2bfd112944d30647ae66f350466791ad9e224786/LICENSE). Seven public headers, including `xrt/detail/xclbin.h`, carry both Apache-2.0 and GPL-2.0 SPDX lines; user space takes the Apache option [I] | Yes, to compile a shim against | Apache text only if XRT code is vendored | High |

Notes:

- No proprietary AIE API header is involved when building with Peano [P]. Kernels include `<aie_api/aie.hpp>` (the MIT submodule) and Peano's libc headers. The only proprietary-looking include, `<adf.h>`, appears only in `aie_api/adf/*.hpp`, which mlir-aie kernels do not use. Building with Vitis `xchesscc` instead would pull in Vitis-licensed headers [I], so the backend should use Peano.
- `aie_kernels/flm_gemma4/` holds AMD's Apache-licensed reimplementations of FastFlowLM's Gemma 4 operators [P], a clean-room-licensed alternative to FLM's closed kernels.
- At mlir-aie `20c0463` (the default branch on 2026-10-03) kernels are organized by family (`linalg/`, `norm/`, ...) with no `aie2p/` directory. The `v1.4.3` tag used for the spike still has `aie_kernels/aie2p/`.
- **Verdict:** kernels built from mlir-aie and amd/IRON with Peano may ship in cera with a third-party notice listing Apache-2.0 WITH LLVM-exception (mlir-aie), Apache-2.0 (IRON) and MIT (`aie_api`, plus the 3 MIT headers), with the AMD/Xilinx copyright lines. FastFlowLM kernels, the Ryzen AI Software binaries and the driver's XRT DLLs are not bundled; the driver is a user-installed dependency.
- **Open:** read the actual Ryzen AI EULA to confirm the "No" row (`account.amd.com` timed out twice). It does not block the plan, since nothing from that row would be bundled.

## 3b. What Windows exposes

### Libraries

The NPU driver package (`C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*`) contains a full XRT user-mode runtime. No separate XRT install is needed for the libraries:

- `xrt_coreutil.dll` (also copied to `C:\Windows\System32`) and `xrt_core.dll`. The checklist's assumption of an `xrt_core` library is **confirmed**.
- `xrt-smi.exe` (also in `C:\Windows\System32\AMD`), `xdp_core.dll` and the XDP profiling plugins.
- Vendor runtimes on top: `vitis-ai-runtime.dll`, `vitis-ai-runtime2.dll`, `aks.dll`, `akskernels.dll`, and `RadeonML*.dll` (including `RadeonML_ipu.dll`).

**No XRT headers or import libraries** are on the machine. A build would take the headers from the open-source XRT repository at the matching commit (`77c7088`) and produce an import library from the DLL exports.

### API surface

`dumpbin /exports xrt_coreutil.dll`:

- The **C API** is the legacy one: `xrtDeviceOpen`, `xrtDeviceLoadXclbin*`, `xrtPLKernelOpen`, `xrtRun*`, `xrtBO*`, `xrtXclbin*`. It has no hardware-context functions.
- `hw_context`, `kernel`, `run`, `runlist`, `elf`, `module`, `bo`, `xclbin`, `fence` and `queue` are exported **only as MSVC-mangled C++ symbols** (for example `??0hw_context@xrt@@...`). The NPU load path goes through `xrt::hw_context`, so a Rust backend needs a small C++ shim compiled with MSVC against the 2.19 headers. Calling the C API alone is not enough.
- `xrt::runlist` is exported, so hardware-side run chaining is present in this runtime. 3c step 1 and the dependent-chain section measure it.

`xrt_core.dll` exports only 14 lines' worth of symbols. It is the device shim loaded by `xrt_coreutil`, not an API to call directly.

### Vendor validation

`xrt-smi validate` passes all three of its tests:

| Test | Result |
|---|---|
| gemm | 51.3 TOPS |
| latency | 129.4 us average |
| throughput | 37292.0 ops/s |

These are AMD's own validation kernels (the `validate_*` and `gemm_*` `.xclbin`s in the driver package), not our kernels. They prove that the driver loads `.xclbin` files through this XRT, but only files AMD signed and packaged. Read the latency number as the vendor tool's round trip for a minimal job. Its exact definition is not documented in the output, and it is **not** a substitute for the 3c dispatch measurement. Even so, it is 4 to 13 times the "10 to 30 microseconds per dispatch" figure in the plan. If it holds for our kernels, roughly 112 dependent GEMV dispatches per Llama-3.2-1B token would cost about 14.5 ms serialized, and chaining (runlist or batching several matmuls per run) becomes a requirement rather than an optimization. The throughput figure works out to about 27 us per operation when jobs overlap. Later runs (3c step 1) put this same job at about 70 us when run interleaved with our own client, so the 129.4 us came from a slow system state, not from what the job normally costs.

### Can an arbitrary `.xclbin` be loaded?

**Yes.** An xclbin we compiled ourselves with the open-source toolchain, unsigned and not packaged by AMD, loads and runs with correct results through the driver's XRT. See 3c step 2.

### Weight dtypes of the available kernels

The driver package ships GEMM overlays named `aie2p_gemm_strix_4x2_bf16`, `aie2p_gemm_vm_strix_4x4` and `aieml_gemm_*` (`aieml` and `phx` are Phoenix-generation names), each with a JSON metadata file. The metadata describes the tile graph (ports, rows, columns) but names no data types. A string search of every `.xclbin` and `.json` in the package for `bfp16`, `bfp8`, `bfp4`, `mx4`, `mx6`, `mx9`, `mxfp4`, `int4`, `int8`, `bf16` and similar found **no matches** inside the files; "bf16" appears only in filenames. So nothing on the machine supports a "native BFP4" weight path.

In IRON (mlir-aie `v1.4.3`), the AIE2P matmul kernels (`aie_kernels/aie2p/mm.cc`, `mm_bfp.cc`, `mm_bfp_mixed.cc`) use `bfloat16` inputs with `accfloat` accumulation, and `bfp16ebs8`: block floating point with one shared exponent per 8 elements and 8-bit mantissas, so roughly 9 bits per weight. No 4-bit type appears in these kernels. With those kernel types, the weights widen to about 8 or 9 bits, roughly doubling the bytes streamed per token.

**4-bit support in the AIE2P ISA and `aie_api`**, read from the headers that ship with the toolchain (`aie_api` in mlir-aie `v1.4.3`, Peano `22.0.0.2026090701`):

- `aie_api` supports `int4` and `uint4` vectors (32 to 256 lanes) on XDNA2 (`aie_api/aie_doc.hpp` type table) and defines `aie::mmul<M, K, N, int8, int4, 32>`, plus the unsigned mixes, for AIE2P (`aie_api/detail/aie2p/mmul_8_4.hpp`, under `__AIE_ARCH__ == 21`; arch 21 is AIE2P per `detail/config.hpp`, and our kernels compile for `aie2p`). The AIE2P path calls `::mac_4x16_16x16_conf` with the 4-bit operand passed directly.
- In Peano's `aie2p/aie2p_vmult.h`, that intrinsic (`mac_4x16_16x16(v64int8 a, v256int4 b, v64acc32 acc)`) is **not a native 4-bit multiply**. It unpacks the int4 vector to int8 (`unpack(extract_v128int4(b, ...))`) and issues two `mac_4x8_8x16` int8 multiplies. So int4 is a native storage and load type on AIE2P, with a hardware unpack, but the arithmetic runs at the int8 rate.
- What that means: Q4 weights **can stay at 4 bits in DDR** and feed the stock `aie::mmul<int8, int4>` without a hand-written unpack. The cost is int8-rate compute, which a bandwidth-bound GEMV can afford. Two constraints follow: activations must be int8 (there is no `bf16 x int4` mmul), and Q4_0's per-32-element f16 block scales must be applied per block after the int8 x int4 dot products, which a stock mmul does not do. This was read from the headers; 3e then ran `aie::mmul<4, 16, 16, int8, int4>` on the NPU and verified it against a CPU reference.
- The weight-format decision for the 0b exit criteria is still open, but the 4-bit option no longer depends on a custom unpack kernel.

## 3c. Spike

Split into two steps. Step 1 checks that our own code can drive the NPU through XRT and measures dispatch cost with AMD's shipped kernels. Step 2 builds an IRON `.xclbin` and loads it, which answers the arbitrary-xclbin question.

### Step 1: host-side XRT access and dispatch timing

Recorded 2026-10-03, on AC power with the "performance" plan; no power numbers yet.

A standalone C++ program (`tools/xdna-spike/spike.cpp`: a measurement tool, not backend code, and not part of the cargo workspace) built with MSVC `cl.exe` 19.44.35227.0 (`/std:c++17 /Zc:__cplusplus /EHsc /O2 /MD`):

- **Headers:** XRT at commit `77c7088d804602a53c3eb489b9cb37b709bcd751` (`src/runtime_src/core/include`). `xrt/detail/version.h` is generated by CMake in a normal XRT build; here it was filled in by hand from `src/CMake/config/version.h.in` for 2.19.0. `/Zc:__cplusplus` is required: without it MSVC reports C++98, and `xrt/detail/any.h` falls back to `boost::any`.
- **Import library:** generated from the driver's own DLL (`dumpbin /exports xrt_coreutil.dll`, 507 exports, then `lib /def`). The program links against the MSVC-mangled C++ symbols with no unresolved externals and runs against `C:\Windows\System32\xrt_coreutil.dll`. **So host-side API access from our own binary works** with only the driver installed: no Ryzen AI Software, no XRT install.
- **Job:** the same nop job as `xrt-smi validate`'s latency test: `validate_17f0_10.xclbin` (the `_11` and `_20` files are byte-for-byte the same xclbin, same UUID), kernel `DPU_PDI_0`, DPU-sequence opcode 1, an all-zero instruction buffer. The flow matches `TestNPULatency.cpp` at the same commit. That test reports `elapsed / iterations` with no warmup, which is why it gives a mean and not a distribution.
- **Timing:** `std::chrono::steady_clock` around `start()` plus `wait2()` (single runs), around `runlist.execute()` plus `wait()` (chains), and around L `start()` calls followed by L `wait2()` calls (streams). 50 warmup runs for single runs and 10 for each chain length. One fresh process per series.

Single runs, one at a time (start, wait, repeat), three separate processes:

| Process | n | min | p50 | p90 | p99 | max | mean (us) |
|---|---|---|---|---|---|---|---|
| 1 | 5000 | 59.0 | 79.1 | 115.7 | 161.0 | 322.5 | 86.7 |
| 2 | 5000 | 60.6 | 81.4 | 103.8 | 153.6 | 487.4 | 84.9 |
| 3 | 5000 | 60.6 | 73.9 | 91.5 | 140.9 | 331.6 | 77.3 |

Our mean is 77 to 87 us, against the 129.4 us xrt-smi reported earlier in the day for the same job. Follow-up, explaining the gap:

- **Warmup does not explain it.** The same series with no warmup (`latency-cold`), three processes of n = 5000: means 73.6, 69.1 and 78.6 us (p50 69.4, 64.9, 76.8; p99 129.3, 102.3, 132.4). Warm series run beside them: means 68.5, 69.8 and 73.7 us. Short cold series (n = 50): means 86.4, 91.8 and 85.6 us.
- **xrt-smi agrees with us when run beside us.** `xrt-smi validate -r latency --verbose` (10000 iterations, "Using DPU Sequence", 20-byte instruction buffer), in a separate set of three processes interleaved with three more `latency-cold` processes: xrt-smi 72.4, 69.8 and 72.0 us (its mean); ours 69.4, 70.2 and **119.0** us (means).
- So the 129.4 us was a system-state excursion, not a difference in method. Some processes land around 70 us and some around 120 us, whichever tool is running. The cause (power or clock state, or host scheduling) is not identified. Any single-process latency number should be read with that spread in mind.

`xrt::runlist` with L runs per list, n = 500 lists per length, one process:

| L | per list p50 | per list p99 | per run p50 | per run p90 | per run p99 (us) |
|---|---|---|---|---|---|
| 1 | 88.5 | 143.4 | 88.5 | 103.0 | 143.4 |
| 2 | 107.3 | 148.5 | 53.6 | 59.9 | 74.2 |
| 4 | 154.2 | 185.9 | 38.5 | 42.2 | 46.5 |
| 8 | 235.4 | 304.9 | 29.4 | 32.0 | 38.1 |
| 16 | 458.4 | 567.9 | 28.6 | 30.6 | 35.5 |
| 32 | 814.2 | 991.6 | 25.4 | 26.7 | 31.0 |
| 64 | 1483.6 | 1790.3 | 23.2 | 25.8 | 28.0 |
| 128 | 2854.1 | 3374.7 | 22.3 | 24.5 | 26.4 |

The same L runs started back to back without a runlist ("stream"), then waited on, n = 500 per length, one process. Per-run p50 (us): L=1 70.7, L=2 46.7, L=4 32.8, L=8 27.0, L=16 24.4, L=32 23.2, L=64 22.0, L=128 27.5.

The stream slowdown at L=128 is reproducible, not noise. Three more processes, per-run p50 at L = 32 / 64 / 128 / 256 (us): 22.6 / 21.8 / 26.8 / 23.2, then 23.1 / 21.9 / 27.1 / 23.3, then 22.7 / 22.0 / 27.1 / 23.0. L=128 is consistently about 5 us per run slower than 64 and about 4 us slower than 256, while the runlist at L=128 does not show it (22.3). A queue-depth effect specific to about 128 outstanding commands is the likely reading; the mechanism is not identified.

What this says:

- A list of L nop runs costs about **60 us fixed plus about 22 us per run** (L=128: 2854 us p50). The per-run cost approaches a floor of about 20 to 23 us from L of about 32 up.
- **The runlist is no faster than plain back-to-back submission** at any length measured, except L = 128, where plain submission has its own slowdown (above). The roughly 22 us per run therefore looks like a per-command cost on the device or firmware side, not host overhead that chaining removes. That is an inference from these two curves only; it was not checked with device-side timestamps.
- **These are nop runs.** They carry no data dependencies and do no compute. A real GEMV adds its own compute and DMA time on top, and dependent runs may not overlap the way independent ones can.
- For planning: if a Llama-3.2-1B token takes about 112 separate matmul dispatches, the dispatch floor alone is about 112 x 22 us = 2.5 ms when they are queued together (a ceiling of about 400 tok/s from dispatch cost alone, not demonstrated), and about 112 x 80 us = 9 ms if each waits for the previous one to finish on the host. "One dispatch per token" remains unmeasured; fusing several matmuls into one run is what would lower the floor.

### Step 1b: do other shipped xclbins load?

The same program created a hardware context and opened the DPU kernel for each of these, with no error:

| xclbin | Kernels | Parse, register and create the context (us) |
|---|---|---|
| `validate_17f0_10` | `vadd`, `DPU_PDI_0` | 8686 |
| `aie2p_gemm_strix_4x2_bf16` | `vadd`, `DPU` | 9684 |
| `aie2p_gemm_vm_strix_4x4` | `vadd`, `DPU` | 12833 |
| `gemm_17f0_10` | `vadd`, `DPU_1x4` | 16176 |
| `AMD_AIE2P_4x4_Overlay_3.5.0.0-2354_ipu_2` | `vadd`, `DPU_PDI_0` to `DPU_PDI_14`, `XDP_KERNEL` | 96150 |
| `aieml_gemm_vm_phx_4x4` | `vadd`, `DPU` | 13216 |

The last row is a Phoenix-generation (`aieml`, `phx`) xclbin, and it also got a context on this Strix Halo NPU. So context creation checks little about the target. Getting a context **does not show that an xclbin will run**; the configuration may only be validated or loaded at the first run. None of these GEMM xclbins was run, because each needs an instruction stream for a particular GEMM shape that is not shipped next to it.

All of these are AMD-built files. Whether the driver accepts an xclbin we built (signing or another check) is still unknown; step 2 answers it.

### Step 2: IRON xclbin

Recorded 2026-10-03.

**Toolchain:** mlir-aie `v1.4.3` (`95b3d1ccc0bfe5183bae1fa9014cfdc1fb4d96c8`) from its prebuilt wheel, Peano (llvm-aie) `22.0.0.2026090701+3e93bf7b`, Python 3.12, all inside WSL2 (Ubuntu 26.04) on the same machine. `xclbinutil` came from Ubuntu's `libxrt-utils` 2.21.75 package, unpacked without root. Builds take about 4 s per design. Exact steps and every design file are in `tools/xdna-spike/`.

**Artifacts and submission:** each build emits an `.xclbin` (for example 10,209 bytes for the 288 x 288 GEMV) and a separate instruction binary (`insts.bin`, 420 bytes, 105 32-bit words). The xclbin contains one kernel named `MLIR_AIE`. The host loads the instruction words into a `XCL_BO_FLAGS_CACHEABLE` buffer and passes them as kernel arguments: `(opcode 3, instruction bo, instruction word count, data buffers...)`. Data buffers are `XRT_BO_FLAGS_HOST_ONLY`. No signing step was run, and none was needed.

**Result: the xclbin runs, unsigned, from WSL to Windows.** mlir-aie's own `matrix_vector` example (int16 inputs, int32 output, one core, 32 x 32 tiles), built for `npu2`, loaded and ran through the driver's `xrt_coreutil.dll` on native Windows. The run completed (`ERT_CMD_STATE_COMPLETED`), and **all 288 outputs match a CPU reference exactly**. This is checklist outcome **(a)**. mlir-aie's makefiles even carry a WSL path that builds host code on Windows through `powershell.exe`, so building in WSL and running on Windows is a flow upstream itself expects.

Every IRON measurement below was verified against a CPU reference once per process, on an untimed run before timing (the timed runs are not checked): exactly for the integer designs; the float Q4 GEMV of 3e within 1e-4 of each row's magnitude; for dependent chains, the first link.

**GEMV timing** (one process per row, n = 2000 for the small case and n = 500 otherwise, int16 weights):

| Design | Weights | p50 (us) | p99 (us) | Effective weight bandwidth at p50 |
|---|---|---|---|---|
| upstream, M = K = 288, 1 core | 162 KiB | 109.0 | 208.2 | 1.5 GB/s (dispatch-bound) |
| M = K = 2048, 1 core | 8 MiB | 1528.9 | 2106.8 | 5.5 GB/s |
| M = 8192, K = 2048, 4 cores (placed in column 0), A on 4 shim channels | 32 MiB | 2739.1 | 3085.2 | 12.3 GB/s |
| M = 8192, K = 2048, 8 cores (placed in columns 0 to 1), A on 8 shim channels | 32 MiB | 1996.1 | 2252.7 | 16.8 GB/s |

The multi-core rows use a copy of the example with `n_cores` as a parameter, and with every core's input stream issued before waiting on any output (`tools/xdna-spike/iron/gemv_multi.py`; built with `n_cols` 4 and 8, which sets the device partition width, while IRON's placer chooses the core columns). In the compiled designs each core's A stream enters on its own shim MM2S channel (shim columns 0 to 1 for 4 cores, 0 to 3 for 8), is staged in a MemTile, and reaches the core through a transposing access pattern on the MemTile-to-core hop, in 2 KiB (32 x 32 int16) tiles. Going from 4 to 8 cores doubles the A channels and gains only 37%, and even the 8-core build uses half of the 16 shim channels the read-only kernel below uses. These numbers bound what this unoptimized design reaches; they are not the NPU's limit.

### 3c.4: how fast can the NPU stream from DDR?

Measured 2026-10-03 in the **Performance** plan. The ceiling comes from two copy designs that use the full array differently (64 MiB in, 64 MiB out, n = 50 per process, two processes each):

| Design | p50 (us) | DDR read + write at p50 | Per direction |
|---|---|---|---|
| `transform_parallel` passthrough (`getting_started/00_memcpy`): 16 workers fed through the shim DMA of all 8 columns x 2 channels, with the placer putting their cores in columns 0 to 3 | 2158.7, 2159.6 | 62.2 GB/s (best single run 64.7) | 31.1 GB/s |
| DMA-only bypass (`basic/memcpy`), 8 columns x 2 channels | 2152.3, 2128.9 | 62.4 to 63.1 GB/s (best 64.5) | 31.2 to 31.5 GB/s |

Both designs stream through all 8 columns' shim DMA, 2 channels each; one passes every tile through a core and the other moves data by DMA alone. They land within 1.5% of each other. So **about 62 to 64 GB/s of combined DDR traffic is the NPU's ceiling** on this machine in the Performance plan (inferred from the agreement, not from a datasheet). A spot check of the `transform_parallel` design in Turbo (one process, 10 runs, 2026-10-12) gave 73.7 GB/s combined at p50, so the copy ceiling also depends on the envelope.

**Read-only ceiling.** GEMV traffic is almost all reads, so a read-dominant design settles the number that matters (`tools/xdna-spike/iron/readbw.py`). One worker per (column, shim channel) pair, 16 in all, each consumes its slice of the input. For every tile it adds the tile's first word into a 16-word output, so the host can check every tile arrived and output traffic is 1 KiB in total. Each process verified one untimed run before timing.

| Read size | Tile (int32 words) | p50 (us) | p90 (us) | GB/s at p50 | GB/s at p90 | Best GB/s |
|---|---|---|---|---|---|---|
| 64 MiB | 1024 | 1781.2, 1600.3, 1615.9, 1760.0, 1815.8 | 1800.6, 1782.7, 1808.0, 1792.6, 1842.9 | 37.0 to 41.9 | 36.4 to 37.6 | 42.4 |
| 64 MiB | 2048 | 1615.2, 1738.8 | 1801.4, 1799.9 | 38.6 to 41.6 | 37.3 | 42.3 |
| 64 MiB | 4096 | 1604.6, 1790.8 | 1791.0, 1826.7 | 37.5 to 41.8 | 36.7 to 37.5 | 42.4 |
| 256 MiB | 4096 | 5743.8, 5745.0, 6059.5, 5916.1 | 6127.6, 6151.8, 6584.3, 6261.1 | 44.3 to 46.7 | 40.8 to 43.8 | 49.3 |

n = 50 per row at 64 MiB, and 50 or 30 at 256 MiB; one process per entry. Tile size makes no difference. At 64 MiB the timings fall into two clusters (about 1600 and about 1780 us), like the latency runs. A two-point fit through the faster cluster (64 MiB at about 1600 us, 256 MiB at about 5745 us) gives about 220 us of fixed cost per run and a streaming rate of **about 48.6 GB/s**.

The 256 MiB copy, run interleaved with the 256 MiB read-only design, stays at 62.2 and 64.2 GB/s combined (best 66.9 and 68.1), 31.1 and 32.1 GB/s per direction. So **reads alone peak at about 45 to 49 GB/s**, and the copy's higher combined figure comes from writes using separate capacity. This is the best a deliberately simple design reached, so it is a lower bound on the hardware's read ceiling, but it is the rate a GEMV-shaped kernel can plan around.

For scale, using the per-token weight traffic of Llama-3.2-1B Q4_0 (about 773 MB, which includes the block scales). These are **ceilings, not demonstrated rates**: the read bandwidth comes from a trivial reduction kernel, and when this section was written the only real GEMV reached 16.8 to 18.3 GB/s (16.8 isolated, about 18.3 per link in a chain). 3e's Q4 kernel later reached 54.8 GB/s in Turbo.

| Read bandwidth | Weight streaming per token | Decode ceiling from weights alone (not demonstrated) |
|---|---|---|
| 45 GB/s (read-only, typical p50) | 17.2 ms | about 58 tok/s |
| 49 GB/s (read-only, best) | 15.8 ms | about 63 tok/s |

That keeps the weights at 4 bits, which the `int8 x int4` path in 3b allows (3e ran it). With weights widened to `bfp16ebs8` (about 9 bits per weight), bytes and time double, and the ceilings fall to about 29 to 32 tok/s (not demonstrated). There is no pure bandwidth test for the iGPU; the closest number is cera's GPU decode, which moves about 72 GB/s of weights in Turbo (section 2). The NPU's read ceiling (55.4 GB/s in Turbo) is therefore about 77% of what cera's GPU path achieves today, not a fraction of the iGPU's own limit. The board's theoretical peak (256-bit LPDDR5X-8000, about 256 GB/s) is a specification, not a measurement.

### Dependent GEMV chains

Run i+1 takes run i's output buffer as its input vector B (`spike gemvchain`). Ordering comes only from in-order execution on one hardware context, which is what a backend would rely on. The values after the first link are meaningless; only link 0 is checked against the CPU, and it verified. Each length is timed as one `xrt::runlist` and as L back-to-back `start()` calls followed by L waits. Per-run p50 (p90) in us, one process per design:

| Design | L = 1 | L = 8 | L = 32 | L = 112 |
|---|---|---|---|---|
| 288 x 288, 1 core, runlist | 131.3 (159.5) | 59.5 (65.5) | 49.7 (60.5) | 45.9 (48.9) |
| 288 x 288, 1 core, stream | 110.9 (120.7) | 69.0 (70.9) | 59.7 (60.8) | 53.4 (58.7) |
| 8192 x 2048, 8 cores, runlist | 2086.7 (2129.7) | 1901.7 (1932.8) | 1861.3 (1925.0) | 1843.4 (1864.3) |
| 8192 x 2048, 8 cores, stream | 1928.8 (1974.8) | 1839.9 (1861.1) | 1834.0 (1848.5) | 1834.7 (1885.4) |

n = 200 chains per length for the small design and 20 for the large one.

What this says:

- For a **small** dependent GEMV, each link costs about 46 us (runlist, L = 112), about twice the 22 us nop floor. The kernel's own device time was not measured separately, so this does not say how much of the 46 us is overhead. Unlike the nops, the runlist beats plain submission here, by about 15%.
- For a **large** dependent GEMV, chaining saves about 90 to 250 us per link against an isolated run, and the per-link cost settles at about 1834 to 1843 us. The 22 us per-command cost is about 1% of a link this size. Whether it overlaps the link's own DMA cannot be separated without the kernel's pure device time.
- For decode: a token is about 112 dependent matmuls, most of them small for a 1B model. A floor of about 46 us per small dependent link is about 5 ms per token, comparable to the 15.8 to 17.2 ms of weight streaming. Fusing several matmuls into one run therefore matters as much as raw bandwidth does.

### Still open in 3c

- How much per-command cost a real fused kernel hides behind its DMA (needs device-side timestamps or the kernel's isolated device time).

## 3d. Decision

**Provisionally (a): raw XRT works on Windows.** Our own unsigned IRON xclbins load and run correctly through the XRT that ships with the NPU driver, driven from our own MSVC binary, with no Ryzen AI Software. The license audit (3a) allows shipping IRON and Peano-built kernels.

Written before the 3e spike; numbers use the steady-state method of section 2.

**Against the proposed bar** ("at least 50% of iGPU decode at 2x its tokens per joule or better"), measured in Turbo with the machine left alone (section 2):

- **Real kernel at the time: fails both.** The int16 GEMV projects to about 20 tok/s (21% of the GPU's 93.4) at 0.69x the GPU's gross tokens per joule.
- **Best case, not demonstrated: passes speed, fails the 2x efficiency bar on gross energy.** If a real 4-bit kernel ran at the read-only kernel's 55.4 GB/s, decode would reach about 52 tok/s (56% of the GPU) at about 1.63x the GPU's gross tokens per joule (2.86x net of the idle floor). That assumes weight streaming and dispatch are the only costs, the `int8 x int4` path works (3e later ran it), and the 5.2 ms of per-token dispatch is not reduced by fusion.
- The NPU's real advantage is low absolute power: about 7 to 10 W above idle against about 51 W for the GPU. That frees the GPU and keeps the machine cool, but on energy per token the GPU's much higher speed wins most of it back.

**Recommendation: stop before backend work.** 3e tested it with a real kernel. Even the undemonstrated best case falls short of the 2x efficiency bar on whole-package energy, a real kernel is far from that best case, and a backend is weeks to months of Windows-only work (C++ shim, hand-written kernels for every op, fusion to beat the dispatch floor, a Linux kernel toolchain, AMD driver churn). The vendor-neutral `--device npu` selection (Phase A) stands on its own.

**What would change it:** a time-boxed optimized Q4 GEMV spike (all 16 shim channels, long linear transfers, `int8 x int4` mmul, per-block scale epilogue) showing that a real kernel reaches **at least about 40 GB/s at no more than about 0.44 J/GB gross** (about 17.7 W package), together with evidence that fusion can bring per-token dispatch down to about 1 ms. Worked back from the bar: 0.773 GB / 40 GB/s = 19.3 ms plus 1 ms of dispatch is about 49 tok/s (53% of the GPU), and at about 17.7 W that is about 2.78 tok/J, 2.0x the GPU's gross 1.39. For scale, the trivial read-only kernel is at 0.416 J/GB and 55.4 GB/s. The criterion was first stated as 0.42 J/GB against the GPU's whole-process power (1.46 tok/J); on the steady-state method used throughout it is 0.44 J/GB. Worth running only if NPU efficiency matters enough to the product to justify the backend afterwards, which is a product decision.

The other 0b exit criteria remain open: the weight format (`int8 x int4` with a per-block scale epilogue, versus `bfp16ebs8`), activation quantization (int8 if 4-bit weights are kept), and the parity-tolerance policy.

## 3e. Q4 GEMV kernel spike

Run 2026-10-12 as a follow-up to the recommendation in 3d. Purpose: find out whether a **real** 4-bit GEMV kernel can approach the trivial read-only kernel's bandwidth and energy. Turbo mode, machine otherwise idle, everything verified against a CPU reference. Code: `tools/xdna-spike/iron/q4gemv/` (design and kernel) and `spike q4gemv` (host packing, reference, timing); `npu_power.sh npu` reruns the energy measurement (the published one was its own session; see section 2).

**Format:** Q4_0-equivalent at 4.5 bits per weight. Weights are signed int4 (Q4_0's `q - 8`) with one bf16 scale per 32-weight block per row. Q4_0 stores f16 scales, and XDNA2 has no float16 vector type, so the host converts them to bf16. Activations are int8 with one f32 scale per 32-element block (Q8_0-style). Quantization follows llama.cpp's Q4_0 and Q8_0 rounding.

**Kernel:** `aie::mmul<4, 16, 16, int8, int4>` per 16-column half-block, with only row 0 of A populated (the bandwidth margin makes the wasted MACs free). Each block's int32 result is scaled by the bf16 row scales times the activation scale and accumulated in `accfloat`.

**Design:** 16 workers, one per (column, shim channel) pair. Each worker gets one linear stream on its own channel: an 18,432-byte slot carrying x and its scales, then its weight tiles (16 rows x 2,048 columns = 18,432 bytes each), pre-packed by the host in consumption order. x rides the weight channel because the weights need all 16 shim channels. Within a tile, all int4 blocks come first and all scales last, which keeps every 128-byte int4 load 64-byte aligned. The first version interleaved them at a 288-byte stride, and AIE vector loads silently read from the aligned-down address, so results were wrong until this was fixed.

**Correctness:** with unit scales the NPU matches the integer reference exactly; with real scales the worst error is 1.66e-7 of the row's magnitude (f32 rounding), on 8,192 x 2,048 and 65,536 x 2,048.

**Bandwidth** (65,536 x 2,048, 72 MiB of weights per run, including scales):

| Submission | n | p50 (us) | p99 (us) | GB/s at p50 |
|---|---|---|---|---|
| one run at a time (start, wait) | 100 runs, two processes | 2637.6, 2700.4 | 5859.2, 5833.2 | 28.6, 28.0 (best 53.0) |
| 4 runs in flight | 60 batches | 1713.8 | 2491.2 | 44.1 |
| 16 runs in flight | 60 batches | 1456.5 | 1653.1 | 51.8 |
| 16 runs in flight, clean idle floors (below) | 2,600 batches | **1378.8** | 1382.5 | **54.76** |

With runs in flight, each sample is one batch's wall time divided by the batch size, so it is a per-run mean within the batch and n counts batches. Those p99 values describe batches, not single runs, and are not comparable with the one-at-a-time p99.

One run at a time alternates between fast phases (about 1,450 us, 52 GB/s, 15 to 20 runs long) and slow phases (about 2,600 to 3,000 us with a repeating 5.8, 4.1 and 3.6 ms spike pattern). Keeping runs in flight removes the slow phases, so a backend must keep the NPU fed rather than wait on each matrix. The cause of the slow phases (power-management state changes while the NPU idles between runs, or periodic memory contention) was not isolated. On 8,192 x 2,048 (9 MiB), one run at a time gives 34.1 GB/s at p50.

Some early runs of this spike were interrupted by Modern Standby (10:26 to 10:46 and 10:55 to 11:03): they launched `spike.exe` directly, without the keep-awake request `hwinfo_sm.py` holds. Every number in this table comes from runs after 11:03; the energy run ran under `hwinfo_sm.py`. Sleep on AC was set to never at about 11:05.

**Energy:** the 16-in-flight run for 58 s, bracketed by 60 s idle floors with HWiNFO's NPU usage at 0% (an earlier attempt was discarded because its floors showed the NPU 33 to 50% busy from another process; M365Copilot was then closed). Steady state is the samples with NPU usage at 90% or more (492 samples, 4.5 to 58.0 s; the same samples as at 99%), which excludes host-side setup, as in section 2:

| | Value |
|---|---|
| Steady package power | 27.55 W |
| Idle floor (before, after) | 14.85 W, 15.47 W |
| Net of floor | 12.39 W |
| **J per GB of weights, gross** | **0.503** |
| J per GB, net | 0.226 |

On the same method, the trivial read-only kernel is 0.416 J/GB gross at 23.0 W, and cera's GPU Q4_0 decode 0.931 J/GB gross at 67.3 W. Net of their floors, the real kernel draws 12.39 W against the read-only kernel's 9.96 W, about 2.4 W more at the same rate.

**Projection for Llama-3.2-1B Q4_0** (773 MB per token, weights and dispatch only, so a ceiling, not demonstrated), against the GPU's 93.4 tok/s at 1.39 tok/J gross (1.84 net). The 5.2 ms dispatch floor comes from the dependent GEMV chains in 3c, measured in the Performance plan:

| Per-token dispatch | Decode | Share of GPU | tok/J gross | tok/J net |
|---|---|---|---|---|
| 5.2 ms (measured floor for 112 dependent links) | 51.8 tok/s | 55% | 1.88 (1.35x) | 4.18 (2.28x) |
| 1 ms (fusion, not demonstrated) | 66.1 tok/s | 71% | 2.40 (1.73x) | 5.34 (2.91x) |
| none (perfect fusion) | 70.8 tok/s | 76% | 2.57 (1.85x) | 5.72 (3.12x) |

**Against the spike's stop criterion** (at least about 40 GB/s at no more than about 0.44 J/GB gross; 0.42 as first stated): **bandwidth passes easily (54.8 GB/s); gross energy fails (0.503 J/GB).** Even with perfect fusion, whole-package tokens per joule reaches about 1.85x the GPU, short of the proposed 2x. Net of the idle floor, it clears 2x at every dispatch level.

**Outcome.** A real Q4 kernel reaches the read ceiling, so speed is no longer the obstacle, but on the bar as proposed the recommendation in 3d stands. Two grounds could still justify continuing, and both are product decisions, not measurements: (1) counting energy net of the idle floor (2.3 to 3.1x the GPU), which is right when the machine would be awake anyway, for example while other work runs; (2) valuing the NPU's low absolute power (about 12 W above idle against about 51 W for the GPU) for running a model alongside GPU or CPU work, rather than as a faster or more efficient replacement for them. Not yet accounted for in any projection: attention, normalization, RoPE, activation quantization on the NPU, and demonstrated dispatch fusion, all of which would lower the projected rates.

**Decision (2026-10-12):** proceed with an optimized XDNA backend, on ground (2). The bar's gross-energy criterion is not met and is not claimed to be.

## 4. Backend decision record (0b close-out, 2026-10-14)

This section closes the 0b exit criteria left open at the end of 3d, before any backend engine code. The backend proceeds on ground (2) of 3e (low absolute power alongside CPU/GPU work); the failed gross-energy bar of 3d is replaced by the explicit numeric bar below, not carried forward silently.

**Environment re-verification (2026-10-14, `feat/xdna-backend`).** NPU driver still 32.0.203.311 (`oem205.inf`), power plan still Turbo (`6fecc5ae-f350-48a5-b669-b472cb895ccf`), sleep on AC still never. `spike q4gemv` rebuilt from `tools/xdna-spike/` and re-run at 16 runs in flight (65,536 x 2,048): **54.79 GB/s at p50** (recorded 54.76), correctness PASS, worst error 1.66e-7 of row magnitude (recorded 1.66e-7). The Phase 0 numbers still hold on this machine.

**Weight format.** `int8 x int4` as validated by the 3e spike: signed int4 weights (Q4_0's `q - 8`), one scale per 32-weight block per row, blocks-then-scales layout (128-byte loads stay 64-byte aligned), host converts Q4_0's f16 scales to bf16 (XDNA2 has no float16 vector type). `bfp16ebs8` is not used. This format must not foreclose a future prefill GEMM, but no GEMM design is owed in this pass.

**Activation quantization.** Q8_0-style: int8 activations, one f32 scale per 32-element block, llama.cpp Q8_0 rounding, matching the spike.

**Tensor-type policy** (target file is mixed-type: 110 Q4_0 tensors, Q6_K `token_embd.weight` 128256x2048 tied to the LM head, Q4_1 `ffn_down` in blocks 0-1). Embedding lookup stays a host gather from the original Q6_K data; only the LM-head matmul needs an NPU form, and section 5 shows it streaming at 53.6 GB/s in the padded (131072-row) Q4 form, so the head is converted from Q6_K into the Q4 repack once at load and the parity cost of that conversion is measured at the full-logits oracle. Streaming the head from the host CPU per token (~200 MB/token) is not an accepted fallback. For the two Q4_1 tensors, both options are priced before the backend kernel work: the NPU min-term epilogue (needs a per-block activation sum) against host fallback for 2 of 112 GEMVs per token; the choice and its measured cost are recorded in an addendum when section 5's numbers are in.

**Prefill placement.** Prefill runs on the CPU from the original GGUF weights; attention and the KV cache are host-side in both phases, so the cache never changes owner and the NPU path is entered only at decode. The final bench reports CPU prefill and NPU decode separately. (Amended 2026-10-14; the first backend plan left this unstated.)

**Projection formula.** Token time = sum over shape classes of (bytes / measured dependent rate) + dependent-link count x measured per-link cost, with hardware-context switches counted as links at their measured cost. Energy is gross package power over the whole run (idle floor reported alongside, never subtracted for the bar). Section 5 evaluates the per-shape architecture under this formula (9.1 tok/s; fails), and section 6 the fused one (~47-51 tok/s projected; passes the speed leg).

**Parity policy.** Never bit-exact (0b exit criteria). Teacher-forced: a fixed prompt is run on the CPU reference, its greedy token sequence is forced as input to both implementations, and logits are compared per forced position: top-1 agreement except where the CPU top-2 logit margin falls inside the tie window, and top-10 overlap of at least 9/10, over at least 512 forced positions. (Amended 2026-10-14 from per-position comparison over generated tokens, which is not well-defined once greedy trajectories diverge.)

**Numeric bar for the backend.** End-to-end Llama-3.2-1B-Instruct Q4_0 decode of at least 35 tok/s at at least 1.5 gross tok/J on the decode-dominant run (Turbo, Phase 0 method, fresh process per measurement, gross package power with the adjacent idle floor reported). Below CPU speed is accepted; low absolute draw alongside other work is the point. The thresholds are ratified here and are not renegotiated downward after measurement. `--device npu` is never selected silently by `Auto`, reports unavailable-with-a-reason on driverless or unsupported machines, and support is gated by attempting to load the npu2 xclbin, never by BDF or device-name strings.

**Test baseline.** `cargo test -p cera --lib` on Windows carries the two pre-existing failures recorded in section 1 (`model::lfm2::no_repack_tests::no_repack_skips_cpu_repacks`, `engine::tests::resolution::resolve_all_manifest_files_walks_every_field`). Backend validation claims mean: no failures beyond that recorded set, unless those two are fixed first.

**Non-goals this pass.** Prefill on the NPU, LFM2/hybrid models, Linux, non-Strix Halo XDNA silicon.

## 5. Dependent-chain spike (backend plan step 2, 2026-10-14)

The kill gate the backend plan required before any backend engine code: Q4 GEMV chains with data-dependent submissions at true Llama-3.2-1B shapes, submitted the way decode submits. The Q4 design was generalized first (same kernel and host code for every shape): the output-element grouping now shrinks for small row counts (16/8/4/2/1 tiles), and K wider than 2048 streams in 2048-column chunks that the kernel accumulates (split-K), so one design serves K = 2048 and K = 8192. The head runs at its padded row count 131072. All shapes verified against the CPU reference (worst error at most 1.76e-7 of row magnitude), `spike q4gemv`, one run at a time (start+wait):

| Shape (rows x K) | Used for | p50 per run | Streamed | GB/s at p50 |
|---|---|---|---|---|
| 512 x 2048 | K/V projections | 90.8 us | 0.56 MiB | 6.5 |
| 2048 x 2048 | Q/O projections | 135.9 us | 2.25 MiB | 17.4 |
| 8192 x 2048 | gate/up | 272.4 us | 9.00 MiB | 34.6 |
| 2048 x 8192 | down | 263.6 us | 9.00 MiB | 35.8 |
| 131072 x 2048 | LM head (padded from 128256) | 2816.3 us | 144.00 MiB | 53.6 |

Small shapes are dispatch-bound (a ~90-140 us floor per run); only the head streams near the 3e ceiling. `spike q4chain` then runs a full token of dependent GEMVs (16 layers x q/k/v/o/gate/up/down, then the head; 113 links, 666 MiB streamed per token), each link's f32 output host-quantized into the next link's input, one start+wait per link:

| Submission pattern | Per token (p50) | Decode rate | Per link (mean) | Effective streaming |
|---|---|---|---|---|
| Mixed shapes (5 kernels, as a per-shape backend submits) | 109.9 ms | **9.1 tok/s** | 971.9 us | 6.35 GB/s |
| Same kernel repeated 113x (`SPIKE_Q4CHAIN_SAME=1`, diagnostic) | 16.5 ms | 60.5 tok/s (this shape only) | 146.4 us | 16.13 GB/s |

**Finding: switching the xclbin (hardware context) between GEMVs costs roughly 800 us per switch, and it dominates everything else.** Dependence itself is cheap: with one kernel, a dependent link costs the same ~146 us as the standalone start+wait measurement, host quantization and buffer syncs included. With five kernels alternating the way a per-shape backend would submit, each link pays a context switch (an AIE array reconfiguration, on this evidence) and decode collapses to 9.1 tok/s. No amount of intra-layer pipelining repairs that, because the layer's stages still alternate shapes.

**Gate verdict: the plan's stop criterion is met for the architecture the plan assumed** (per-shape kernels, host-dependent submission): 9.1 tok/s projected against the section 4 bar of 35 tok/s, and 6.35 GB/s effective against the 40 GB/s guideline. Gross J/GB was not measured for the chain; at 9.1 tok/s the speed leg already fails. Per the plan, backend engine work stops here unless the architecture changes. The constraint the spike actually establishes is narrower and more useful than "the NPU is slow": **all GEMVs in a token must run under one xclbin**, either as one fused design (dependent GEMVs inside a single run, the direction of the WSL `q4chain` prototype, which chains square 2048x2048 GEMVs on-device with on-core quantization) or as a single design whose shapes are runtime parameters. The Q6_K head question is moot until that exists; the head at padded Q4 shapes already streams at 53.6 GB/s.

Two build lessons worth keeping: the IRON JIT cache (the `.prj` directory next to a built xclbin) does not notice kernel-source changes, so a rebuilt shape can silently reuse a stale kernel (delete the `.prj` when `q4gemv.cc` changes; this cost a long detour here, found by checking the NPU output against a reference that repeats x chunk 0, `SPIKE_Q4_REF_MODE=1`), and `tools/xdna-spike/fix_and_build.sh` builds the designs from this Windows (CRLF) checkout inside WSL.

## 6. Fused single-xclbin layer (backend plan step 2b, 2026-10-14)

The section 5 constraint (all GEMVs in a token under one xclbin) taken as a design: `iron/q4layer/` runs all seven GEMV stages of a Llama-3.2-1B layer in **one run**, with activations in f32 slots in DDR, each stage's input broadcast to the 16 workers and quantized on the core between stages. Stage chaining follows the real data flow with attention proxied by its input (this measures GEMV streaming, submission, and on-device chaining, not attention): q/k/v read the hidden slot, o reads q's slot, gate/up read o's slot, down (split-K, K=8192) reads gate's slot. `spike q4layer` checks every stage's output slot against a host simulation of the same chain, then times the run:

| Measurement | Value |
|---|---|
| Stage verification (7 stages, worst error over all rows) | PASS, at most 1.66e-7 of row magnitude |
| Per layer run (start+wait, p50, n=50) | **1.064 ms** |
| Weights streamed per run | 32.62 MiB at 32.16 GB/s |
| Projection: 16 layers + head (2.82 ms) + 2 context switches (~1.6 ms) | ~21.4 ms/token, **~47 tok/s**, before host RMSNorm/RoPE/attention/KV work |

Against the section 4 bar (35 tok/s), the fused architecture clears the speed leg with headroom for the host work not yet measured; per-shape submission (section 5) failed it at 9.1 tok/s. Gross energy was not measured for the fused run (the section 4 power leg stays a manual gate for the backend proper). The head remains a separate xclbin in this accounting (one run per token at 53.6 GB/s, section 5); folding it into the layer design or accepting two switches per token is a backend-step decision.

Two kernel lessons from getting here, both measured: (1) A drain smaller than the out fifo's element leaves the fifo inconsistent and the *next* stage silently produces zeros; the design therefore uses a fixed two-tile output grouping so every drain consumes whole elements (the q4gemv pattern). (2) On-core activation quantization must be vectorized: the scalar loop cost ~150 ns per element (~294 us per stage, more than the GEMV itself), while `aie::to_fixed<int8>` brings a 2048-wide quantization to ~18 us. (An `accum::to_vector<int32>` conversion produced wrong values and was abandoned; `to_fixed` is the working float-to-int8 path.)

## 7. Resumption verification (2026-10-16)

Sections 4 to 6 were written in a prior session and left uncommitted on this branch. On resumption, the three decisive measurements were re-run from the committed `spike.exe` and artifacts, with the power plan confirmed as Turbo (`6fecc5ae-f350-48a5-b669-b472cb895ccf`): Q4 GEMV at 16 runs in flight (65,536 x 2,048) reproduced at **54.82 GB/s** at p50 with the reference check passing (recorded: 54.76 to 54.79); `spike q4chain out` reproduced at **110.15 ms/token, 9.1 tok/s** (recorded: 109.9 ms, 9.1); and `spike q4layer` reproduced all seven stage checks passing at **1.047 ms/layer** at p50 (recorded: 1.064 ms), projecting 51.1 tok/s by the spike's own accounting. The prior session's numbers and the section 5 verdict stand: per-shape submission fails the bar through xclbin switching cost, and the fused single-xclbin layer is the architecture the backend is built on. Section 4 was also amended on resumption (prefill placement, teacher-forced parity, gross-energy bar, projection formula); sections 5 and 6 are unchanged.

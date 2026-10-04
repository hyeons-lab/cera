# AMD XDNA NPU: Phase 0 and 0b results

Results for the checklist in [XDNA_NPU_PHASE0.md](XDNA_NPU_PHASE0.md). Status: **in progress**. Phase 0b's deciding question is answered (yes, see 3c step 2); the go/no-go bar still waits on the section 2 baselines. Sections without numbers are listed as open, not skipped.

All commands ran natively on Windows. They were launched from a WSL2 shell through Windows interop, which starts the Windows executable as a normal Windows process. No measurement here ran inside the Linux VM.

## 0. Environment

Recorded 2026-10-03.

| Item | Value |
|---|---|
| Machine | ASUS ROG Flow Z13 (GZ302EA), AMD Ryzen AI Max+ 395 w/ Radeon 8060S |
| Windows | 11 Pro, build 26200 |
| BIOS | GZ302EA.311 |
| Memory visible to the OS | 95.6 GB (`Win32_ComputerSystem`); xrt-smi reports 97943 MB |
| Firmware UMA carve-out | Not read from firmware. 128 GB installed minus about 96 GB visible suggests about 32 GB reserved; confirm in the BIOS before relying on it |
| NPU device | "NPU Strix Halo", BDF `00c5:00:01.1` |
| NPU driver | 32.0.203.311 (`kipudrv.inf`, published as `oem205.inf`) |
| NPU firmware | 1.0.21.44 |
| XRT | 2.19.0, hash `77c7088d804602a53c3eb489b9cb37b709bcd751` (2025-10-02), shipped inside the NPU driver package |
| Ryzen AI Software | Not installed |
| GPU driver | 32.0.31032.1003 (Radeon 8060S) |
| Power plan | High performance (`27fa6203-...`, "performance"); xrt-smi reports NPU power mode "Performance" |
| Power source | AC (battery status 2, 80% charge at the time) |
| Power measurement tool | HWiNFO (chosen). **Not yet installed**, so no power numbers exist yet. The sensor used (package, NPU or system) will be recorded with the first series |
| Idle package power | Open (needs the power tool) |

## 1. cera on Windows

| Step | Result |
|---|---|
| Toolchain | `nightly-2026-07-10-x86_64-pc-windows-msvc`, picked up from `rust-toolchain.toml` |
| `cargo build --release -p cera-cli` | Pass. The binary is `cera.exe` |
| `cargo test --release -p cera --lib` | 1162 passed, **2 failed**, 16 ignored (see below) |
| `cera inspect --model Llama-3.2-1B-Instruct-Q4_0.gguf` | Pass: 147 tensors, 35 metadata keys, `llama`, 16 blocks. GGUF mmap and Windows path handling work |
| CPU tier selected | Open |
| `RUSTFLAGS="-C target-cpu=native"` effect | Open |

The two test failures are Windows portability problems in the tests, not in the engine:

- `model::lfm2::no_repack_tests::no_repack_skips_cpu_repacks` panics at `std::env::var("HOME").expect("HOME unset")` (`lfm2.rs:5852`). Windows sets `USERPROFILE`, not `HOME`. On a machine without the model this test is meant to skip, but it panics before reaching the skip check.
- `engine::tests::resolution::resolve_all_manifest_files_walks_every_field` fails `assert_eq!(manifest.files.model, "/models/bundles/model.gguf")` (`engine.rs:2777`). The cause is inferred, not confirmed from the panic output: the expected value is a string with `/` separators, while the resolved path is built with `Path::join`, which uses `\` on Windows.

### Models

`Llama-3.2-1B-Q4_0.gguf` (the base model named in the checklist) has no recorded download source in the repo, and the repo copy is unusual: its blocks 0 and 1 are Q4_1 (see `cpu.rs`). The Instruct Q4_0 from the same publisher as the Q8_0 stands in for it. Q4_0 numbers here are therefore **not** directly comparable to Mac numbers taken on the repo's base Q4_0 file.

| File | Source | Size (bytes) | SHA-256 |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-Q8_0.gguf` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 1321083008 | `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3` (matches `scripts/fetch_test_models.sh`) |
| `Llama-3.2-1B-Instruct-Q4_0.gguf` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | 773025920 | `fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8` |

## 2. Phase 0 baselines

Open. These runs wait for HWiNFO logging, so that every row has its power number. The `WGPU_BACKEND` override for the Vulkan-versus-DX12 comparison has not landed yet.

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
- `xrt::runlist` is exported, so hardware-side run chaining is at least present in this runtime. It is not yet exercised.

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

In IRON (mlir-aie `v1.4.3`), the AIE2P matmul kernels (`aie_kernels/aie2p/mm.cc`, `mm_bfp.cc`, `mm_bfp_mixed.cc`) use `bfloat16` inputs with `accfloat` accumulation, and `bfp16ebs8`: block floating point with one shared exponent per 8 elements and 8-bit mantissas, so roughly 9 bits per weight. No 4-bit type appears in these kernels. Q4_0 weights at 4 bits in memory would therefore need a custom kernel that unpacks 4-bit blocks on the core; with the shipped kernel types, the weights widen to about 8 or 9 bits, roughly doubling the bytes streamed per token. The weight-format decision for the 0b exit criteria is still open.

## 3c. Spike

Split into two steps. Step 1 checks that our own code can drive the NPU through XRT and measures dispatch cost with AMD's shipped kernels. Step 2 builds an IRON `.xclbin` and loads it, which answers the arbitrary-xclbin question.

### Step 1: host-side XRT access and dispatch timing

Recorded 2026-10-03, on AC power with the "performance" plan; no power numbers yet.

A standalone C++ program (`tools/xdna-spike/spike.cpp`: a measurement tool, not backend code, and not part of the cargo workspace) built with MSVC `cl.exe` 19.44.35227.0 (`/std:c++17 /Zc:__cplusplus /O2 /MD`):

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
- **xrt-smi agrees with us when run beside us.** `xrt-smi validate -r latency --verbose` (10000 iterations, "Using DPU Sequence", 20-byte instruction buffer), interleaved with our cold series: xrt-smi 72.4, 69.8 and 72.0 us; ours 69.4, 70.2 and **119.0** us.
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

The stream slowdown at L=128 is reproducible, not noise. Three more processes, per-run p50 at L = 32 / 64 / 128 / 256 (us): 22.6 / 21.8 / 26.8 / 23.2, then 23.1 / 21.9 / 27.1 / 23.3, then 22.7 / 22.0 / 27.1 / 23.0. L=128 is consistently about 5 us per run slower than both 64 and 256, while the runlist at L=128 does not show it (22.3). A queue-depth effect specific to about 128 outstanding commands is the likely reading; the mechanism is not identified.

What this says:

- A list of L nop runs costs about **60 us fixed plus about 22 us per run** (L=128: 2854 us p50). The per-run cost approaches a floor of about 20 to 23 us from L of about 32 up.
- **The runlist is no faster than plain back-to-back submission** at any length measured. The roughly 22 us per run therefore looks like a per-command cost on the device or firmware side, not host overhead that chaining removes. That is an inference from these two curves only; it was not checked with device-side timestamps.
- **These are nop runs.** They carry no data dependencies and do no compute. A real GEMV adds its own compute and DMA time on top, and dependent runs may not overlap the way independent ones can.
- For planning: if a Llama-3.2-1B token takes about 112 separate matmul dispatches, the dispatch floor alone is about 112 x 22 us = 2.5 ms when they are queued together (at most about 400 tok/s from dispatch cost), and about 112 x 80 us = 9 ms if each waits for the previous one to finish on the host. "One dispatch per token" remains unmeasured; fusing several matmuls into one run is what would lower the floor.

### Step 1b: do other shipped xclbins load?

The same program created a hardware context and opened the DPU kernel for each of these, with no error:

| xclbin | Kernels | Context creation (us) |
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

Every IRON run below also verified exactly against a CPU reference.

**GEMV timing** (one process per row, n = 2000 for the small case and n = 500 otherwise, int16 weights):

| Design | Weights | p50 (us) | p99 (us) | Effective weight bandwidth at p50 |
|---|---|---|---|---|
| upstream, M = K = 288, 1 core | 162 KB | 109.0 | 208.2 | 1.5 GB/s (dispatch-bound) |
| M = K = 2048, 1 core | 8 MB | 1528.9 | 2106.8 | 5.5 GB/s |
| M = 8192, K = 2048, 4 cores, 4 columns | 32 MB | 2739.1 | 3085.2 | 12.3 GB/s |
| M = 8192, K = 2048, 8 cores, 8 columns | 32 MB | 1996.1 | 2252.7 | 16.8 GB/s |

The multi-core rows use a copy of the example with `n_cores` and `n_cols` as parameters, and with every core's input stream issued before waiting on any output (`tools/xdna-spike/iron/gemv_multi.py`). Going from 4 to 8 cores gains only 37%, and the 8-core run does less work per core than the 1-core run, so these rows are **not** compute-bound either. The example design moves A in 2 KB tiles through a transposing DMA pattern with no MemTile staging. These numbers bound what this unoptimized design reaches; they are not the NPU's limit.

### 3c.4: how fast can the NPU stream from DDR?

The ceiling comes from two copy designs that use the full array differently (64 MB in, 64 MB out, n = 50 per process, two processes each):

| Design | p50 (us) | DDR read + write at p50 | Per direction |
|---|---|---|---|
| `transform_parallel` passthrough (`getting_started/00_memcpy`), cores in columns 0 to 3 | 2158.7, 2159.6 | 62.2 GB/s (best single run 64.7) | 31.1 GB/s |
| DMA-only bypass (`basic/memcpy`), 8 columns x 2 channels | 2152.3, 2128.9 | 62.4 to 63.1 GB/s (best 64.5) | 31.2 to 31.5 GB/s |

Two designs with different column counts, channel usage and core involvement land within 1.5% of each other. So **about 62 to 64 GB/s of combined DDR traffic is the NPU's ceiling** on this machine (inferred from the agreement, not from a datasheet).

What this does **not** settle: whether reads alone, with no write stream, can exceed 31 GB/s. GEMV traffic is almost all reads, so the read-only ceiling is the number that matters. It lies between 31 and about 64 GB/s and needs a read-dominant test.

For scale, using the per-token weight traffic of Llama-3.2-1B Q4_0 (about 773 MB):

| Read bandwidth | Weight streaming per token | Decode ceiling from weights alone |
|---|---|---|
| 31 GB/s | 24.9 ms | about 40 tok/s |
| 62 GB/s | 12.5 ms | about 80 tok/s |

That is at 4 bits per weight. With weights widened to the 8 to 9 bit `bfp16ebs8` (see 3b), bytes and time double, so the ceilings halve to about 20 and 40 tok/s. The iGPU's bandwidth is not measured yet (section 2), so the comparison checklist 3c.4 asks for is still open. The board's theoretical peak (256-bit LPDDR5X-8000, about 256 GB/s) is a specification, not a measurement.

### Still open in 3c

- A chain of **dependent** GEMVs (each one's output feeds the next) at L = 1, 8, 32 and 112, compared with the nop floor.
- A read-dominant bandwidth test.
- Whether dispatch cost overlaps with compute and DMA in a real kernel.

## 3d. Decision

**Provisionally (a): raw XRT works on Windows.** Our own unsigned IRON xclbins load and run correctly through the XRT that ships with the NPU driver, driven from our own MSVC binary, with no Ryzen AI Software. The license audit (3a) allows shipping IRON and Peano-built kernels.

The go/no-go bar is not decided. It needs the section 2 baselines (iGPU decode tok/s and power), and the evidence so far points the hard way on raw speed. The NPU's measured DDR ceiling (about 62 GB/s combined, 31 GB/s per direction) caps 1B-model decode at roughly 40 to 80 tok/s at 4-bit weights, before any compute or dispatch cost. Clearing "50% of iGPU decode" therefore depends on how fast the iGPU actually is, and the case for the NPU will likely rest on tokens per joule. Both need HWiNFO.

The other 0b exit criteria remain open: the weight format (4-bit custom kernel versus `bfp16ebs8`), activation quantization, and the parity-tolerance policy.

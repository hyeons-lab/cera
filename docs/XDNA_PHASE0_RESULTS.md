# AMD XDNA NPU: Phase 0 and 0b results

Results for the checklist in [XDNA_NPU_PHASE0.md](XDNA_NPU_PHASE0.md). Status: **in progress**. Sections without numbers are listed as open, not skipped.

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

Open.

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

These are AMD's own validation kernels (the `validate_*` and `gemm_*` `.xclbin`s in the driver package), not our kernels. They prove that the driver loads `.xclbin` files through this XRT, but only files AMD signed and packaged. Read the latency number as the vendor tool's round trip for a minimal job. Its exact definition is not documented in the output, and it is **not** a substitute for the 3c dispatch measurement. Even so, it is 4 to 13 times the "10 to 30 microseconds per dispatch" figure in the plan. If it holds for our kernels, roughly 112 dependent GEMV dispatches per Llama-3.2-1B token would cost about 14.5 ms serialized, and chaining (runlist or batching several matmuls per run) becomes a requirement rather than an optimization. The throughput figure works out to about 27 us per operation when jobs overlap.

### Can an arbitrary `.xclbin` be loaded?

**Open: the deciding question.** Nothing so far rules it in or out. The 3c spike answers it.

### Weight dtypes of the available kernels

The driver package ships GEMM overlays named `aie2p_gemm_strix_4x2_bf16`, `aie2p_gemm_vm_strix_4x4` and `aieml_gemm_*` (`aieml` and `phx` are Phoenix-generation names), each with a JSON metadata file. The metadata describes the tile graph (ports, rows, columns) but names no data types. A string search of every `.xclbin` and `.json` in the package for `bfp16`, `bfp8`, `bfp4`, `mx4`, `mx6`, `mx9`, `mxfp4`, `int4`, `int8`, `bf16` and similar found **no matches** inside the files; "bf16" appears only in filenames. So nothing on the machine supports a "native BFP4" weight path. The weight format stays open until the IRON operator library's supported types are checked (3a and 3c).

## 3c. Spike

Open.

## 3d. Decision

Open.

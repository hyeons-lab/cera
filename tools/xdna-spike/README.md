# xdna-spike

A measurement tool for the AMD XDNA NPU feasibility study (`docs/XDNA_NPU_PHASE0.md`, results in `docs/XDNA_PHASE0_RESULTS.md`). It is **not backend code** and is not part of the cargo workspace. It drives the NPU on native Windows through the XRT runtime that ships inside the NPU driver, and times jobs.

- `spike.cpp`: the Windows host program (MSVC, C++17).
- `build.ps1`: builds `spike.exe`, including an import library generated from the driver's own `xrt_coreutil.dll`.
- `hwinfo_sm.py`: reads HWiNFO sensors through its shared-memory interface (Windows Python); `list`, `idle <s>` and `run -- <command>` with per-run energy, and `--steady "<label>>=<value>"` to average only the samples where a device is busy. It keeps Windows awake while it measures.
- `npu_power.sh`: `npu` measures the read-only kernel, the int16 GEMV and the Q4 GEMV back to back; `gpu` measures the GPU reference (cera Q4_0 decode). Each load is bracketed by idle floors.
- `baselines.sh`: the section 2 CPU and GPU baseline runs (`matrix`, `decode`), each wrapped in `hwinfo_sm.py`.
- `common.sh`: paths shared by the two scripts (see *Running the measurement scripts*).
- `iron/`: IRON designs and a build script, compiled on Linux (WSL2 works) into `.xclbin` plus instruction binaries that `spike.exe` loads.

## Commands

```
spike latency      <validate.xclbin> <iters>               nop job, start+wait per run, 50 warmup runs
spike latency-cold <validate.xclbin> <iters>               same, no warmup (as xrt-smi does it)
spike chain        <validate.xclbin> <iters> <L> [<L>...]  xrt::runlist of L nop runs
spike stream       <validate.xclbin> <iters> <L> [<L>...]  L nop runs started back to back, then waited on
spike load         <any.xclbin>                            register, create a hw_context, open the DPU kernel
spike gemv         <xclbin> <insts.bin> <M> <K> [iters]    IRON int16 GEMV, checked against a CPU reference
spike memcpy       <xclbin> <insts.bin> <n_int32> [iters]  IRON memcpy, checked, reports DDR GB/s
spike readbw       <xclbin> <insts.bin> <n_int32> <workers> [tile] [iters]
                                                           read-dominant design (iron/readbw.py), checked, reports read GB/s
spike q4gemv       <xclbin> <insts.bin> <M> <K> [iters]    Q4_0-style GEMV (iron/q4gemv), checked against a CPU reference
                                                           env: SPIKE_Q4_STREAM=L keeps L runs in flight (each sample is then
                                                           a batch's wall time / L, and n counts batches); SPIKE_Q4_TRACE=1
                                                           prints samples in order; SPIKE_Q4_UNIT, SPIKE_Q4_HI, SPIKE_Q4_BT
                                                           are layout diagnostics
spike gemvchain    <xclbin> <insts.bin> <M> <K> <iters> <L> [<L>...]
                                                           L dependent GEMVs (run i+1 reads run i's output), runlist and stream
```

Each command prints min, p50, p90, p99, max and mean in microseconds, and exits 1 if its check fails. Sizes are printed in binary units (KiB, MiB) and bandwidth in decimal GB/s (10^9 bytes per second). Run one process per series. Shapes passed to the IRON commands must match the ones the xclbin was built for; the host cannot read them back from the xclbin.

The nop job is the one `xrt-smi validate`'s latency test uses (`TestNPULatency.cpp` at XRT `77c7088`): `validate_17f0_*.xclbin` from the driver package, kernel `DPU_PDI_0`, DPU-sequence opcode 1, an all-zero instruction buffer. The validate xclbin is at `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*\validate_17f0_10.xclbin`.

IRON kernels are submitted the way the mlir-aie host tests do it: kernel `MLIR_AIE`, arguments `(opcode 3, instruction bo, instruction word count, buffers...)`, the instruction bo allocated `XCL_BO_FLAGS_CACHEABLE` and data bos `XRT_BO_FLAGS_HOST_ONLY`.

## Building `spike.exe` (Windows)

Needs Visual Studio with the C++ workload, and the AMD NPU driver. Ryzen AI Software and a separate XRT install are not needed.

1. XRT headers at the commit the driver's XRT reports (`xrt-smi examine`; 2.19.0, `77c7088d804602a53c3eb489b9cb37b709bcd751` for driver 32.0.203.311), cloned to `xrt/` next to `build.ps1`:

   ```
   git init xrt && cd xrt
   git remote add origin https://github.com/Xilinx/XRT.git
   git fetch --depth 1 origin 77c7088d804602a53c3eb489b9cb37b709bcd751 && git checkout FETCH_HEAD
   cd ..
   ```

   Run these from `tools/xdna-spike` in bash (WSL or Git Bash) or PowerShell 7; Windows PowerShell 5.1 does not accept `&&`.

2. Generate `src/runtime_src/core/include/xrt/detail/version.h`, which a normal XRT build makes with CMake, from `src/CMake/config/version.h.in`. Replace `@XRT_VERSION_STRING@` with `2.19.0`, `@XRT_VERSION_MAJOR@` with `2`, `@XRT_VERSION_MINOR@` with `19`, `@XRT_VERSION_PATCH@` with `0`, `@XRT_HASH@` with the commit, the commit-count fields with `0`, and the remaining fields with empty strings. Only `XRT_VERSION_CODE` matters to the headers.

3. From `tools/xdna-spike`: `powershell -ExecutionPolicy Bypass -File build.ps1`. The script:
   - runs `dumpbin /exports` on `C:\Windows\System32\xrt_coreutil.dll` (the copy `spike.exe` loads) and writes `xrt_coreutil.def`;
   - runs `lib /def` to make `xrt_coreutil.lib`;
   - compiles with `cl /std:c++17 /Zc:__cplusplus /EHsc /O2 /MD`. `/Zc:__cplusplus` is required: without it MSVC reports C++98, and `xrt/detail/any.h` then wants `boost::any`.

At run time `spike.exe` picks up `C:\Windows\System32\xrt_coreutil.dll`.

## Building the IRON designs (Linux or WSL2)

Tested with mlir-aie `v1.4.3` (`95b3d1ccc0bfe5183bae1fa9014cfdc1fb4d96c8`) and llvm-aie (Peano) `22.0.0.2026090701+3e93bf7b`, with Python 3.12.

0. `export IRON_HOME=~/xdna-iron` (the default `iron/build.sh` uses), and keep it set for the steps below.
1. `git clone https://github.com/Xilinx/mlir-aie.git $IRON_HOME/mlir-aie`, then `git -C $IRON_HOME/mlir-aie checkout v1.4.3`. The install script refuses commits with no matching wheel, so use a release tag.
2. `PYTHON=/path/to/python3.12 source utils/env_install.sh $IRON_HOME/ironenv` from that checkout. The script requires exactly Python 3.12 (`uv python install 3.12` provides one).
3. `xclbinutil` packages the xclbin. Without root, unpack it from Ubuntu's `libxrt-utils` package plus its two Boost libraries into `$IRON_HOME/xrt-local` (package names as of Ubuntu 26.04; on another release, use the Boost version `apt-cache depends libxrt-utils` lists):

   ```
   mkdir -p $IRON_HOME/debs && cd $IRON_HOME/debs
   apt-get download libxrt-utils libboost-filesystem1.90.0 libboost-program-options1.90.0
   for d in *.deb; do dpkg-deb -x "$d" $IRON_HOME/xrt-local; done
   cd -
   ```

4. `iron/build.sh gemv 8192 2048 8 8`, `iron/build.sh memcpy 16777216`, `iron/build.sh readbw 67108864 4096` or `iron/build.sh q4gemv 65536`. For `readbw` on npu2 pass `16` as the worker count to `spike` (8 columns x 2 channels). `IRON_HOME` defaults to `~/xdna-iron`, and `OUT_DIR` defaults to `tools/xdna-spike/out`, where the measurement scripts look. For `q4gemv`, M must be a multiple of 4096 and at least 8192, and K a multiple of 64 (2,048 is the tested value). For the GEMV, `M / (32 x n_cores)` must be at most 64 (a DMA buffer-descriptor repeat limit), and `n_cols` must be at least `n_cores`.

The memcpy bypass variant in the results is mlir-aie's own `programming_examples/basic/memcpy/memcpy.py --dev npu2 -l 16777216 -co 8 -ch 2 -b True` with `--xclbin-path` and `--insts-path`.

## Running the measurement scripts

Run `baselines.sh` and `npu_power.sh` from WSL on the machine under test; every program they start runs natively on Windows. Prerequisites:

- HWiNFO running elevated, with "Shared Memory Support" enabled and the Sensors window open. The free edition turns shared memory off after 12 hours; untick and re-tick the option to restore it.
- `spike.exe` built by `build.ps1` (for `npu_power.sh npu`), and the IRON artifacts in `out/` from `iron/build.sh readbw 67108864 4096`, `iron/build.sh gemv 8192 2048 8 8` and `iron/build.sh q4gemv 65536`. The script checks all of them before it starts measuring.
- `cera.exe` built on Windows with the MSVC toolchain, from a Windows shell in the checkout: `cargo build --release -p cera-cli --features gpu` (for `baselines.sh` and `npu_power.sh gpu`). Run from WSL, the same command builds a Linux binary, which the scripts cannot use.
- The Llama-3.2-1B Instruct Q4_0 and Q8_0 GGUFs under `%USERPROFILE%\.leap\models\<name>\<name>.gguf`.

`common.sh` derives every path from the checkout; override any of them through the environment:

| Variable | Default |
|---|---|
| `PY` | `py.exe` (the Windows Python launcher) |
| `SM` | `hwinfo_sm.py` next to the scripts |
| `SPIKE` | `spike.exe` next to the scripts |
| `ARTIFACTS` | `out/` next to the scripts |
| `CERA` | `target\release\cera.exe` in the checkout |
| `MD` | `%USERPROFILE%\.leap\models` |
| `OUT` | `results/` next to the scripts (logs and per-sample CSVs) |

Check that every idle floor in the logs shows the NPU at 0% usage; another process using the NPU invalidates the floor.

## Licenses

Files written for cera (`spike.cpp`, `build.ps1`, `hwinfo_sm.py`, `common.sh`, `baselines.sh`, `npu_power.sh`, `iron/build.sh`, `iron/readbw.py` and `iron/q4gemv/`) are under cera's license, Apache-2.0 OR MIT. `iron/gemv_multi.py` and `iron/memcpy_bw.py` are modified copies of mlir-aie v1.4.3 examples and keep their `Apache-2.0 WITH LLVM-exception` headers, with the source file and the modifications noted at the top of each. The XRT headers are not vendored here.

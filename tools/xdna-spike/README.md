# xdna-spike

A measurement tool for the AMD XDNA NPU feasibility study (`docs/XDNA_NPU_PHASE0.md`, results in `docs/XDNA_PHASE0_RESULTS.md`). It is **not backend code** and is not part of the cargo workspace. It drives the NPU on native Windows through the XRT runtime that ships inside the NPU driver, and times jobs.

- `spike.cpp`: the Windows host program (MSVC, C++17).
- `build.ps1`: builds `spike.exe`, including an import library generated from the driver's own `xrt_coreutil.dll`.
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
```

Each command prints min, p50, p90, p99, max and mean in microseconds. Run one process per series.

The nop job is the one `xrt-smi validate`'s latency test uses (`TestNPULatency.cpp` at XRT `77c7088`): `validate_17f0_*.xclbin` from the driver package, kernel `DPU_PDI_0`, DPU-sequence opcode 1, an all-zero instruction buffer. The validate xclbin is at `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_*\validate_17f0_10.xclbin`.

IRON kernels are submitted the way the mlir-aie host tests do it: kernel `MLIR_AIE`, arguments `(opcode 3, instruction bo, instruction word count, buffers...)`, the instruction bo allocated `XCL_BO_FLAGS_CACHEABLE` and data bos `XRT_BO_FLAGS_HOST_ONLY`.

## Building `spike.exe` (Windows)

Needs Visual Studio with the C++ workload, and the AMD NPU driver. Ryzen AI Software and a separate XRT install are not needed.

1. XRT headers at the commit the driver's XRT reports (`xrt-smi examine`; 2.19.0, `77c7088d804602a53c3eb489b9cb37b709bcd751` for driver 32.0.203.311), cloned to `xrt/` next to `build.ps1`:

   ```
   git init xrt && cd xrt
   git remote add origin https://github.com/Xilinx/XRT.git
   git fetch --depth 1 origin 77c7088d804602a53c3eb489b9cb37b709bcd751 && git checkout FETCH_HEAD
   ```

2. Generate `src/runtime_src/core/include/xrt/detail/version.h`, which a normal XRT build makes with CMake, from `src/CMake/config/version.h.in`. Replace `@XRT_VERSION_STRING@` with `2.19.0`, `@XRT_VERSION_MAJOR@` with `2`, `@XRT_VERSION_MINOR@` with `19`, `@XRT_VERSION_PATCH@` with `0`, `@XRT_HASH@` with the commit, the commit-count fields with `0`, and the remaining fields with empty strings. Only `XRT_VERSION_CODE` matters to the headers.

3. `powershell -ExecutionPolicy Bypass -File build.ps1`. The script:
   - runs `dumpbin /exports` on `xrt_coreutil.dll` in the newest `kipudrv.inf_amd64_*` driver package and writes `xrt_coreutil.def`;
   - runs `lib /def` to make `xrt_coreutil.lib`;
   - compiles with `cl /std:c++17 /Zc:__cplusplus /EHsc /O2 /MD`. `/Zc:__cplusplus` is required: without it MSVC reports C++98, and `xrt/detail/any.h` then wants `boost::any`.

At run time `spike.exe` picks up `C:\Windows\System32\xrt_coreutil.dll`.

## Building the IRON designs (Linux or WSL2)

Tested with mlir-aie `v1.4.3` (`95b3d1ccc0bfe5183bae1fa9014cfdc1fb4d96c8`) and llvm-aie (Peano) `22.0.0.2026090701+3e93bf7b`, with Python 3.12.

1. `git clone https://github.com/Xilinx/mlir-aie.git $IRON_HOME/mlir-aie`, then `git checkout v1.4.3`. The install script refuses commits with no matching wheel, so use a release tag.
2. `PYTHON=/path/to/python3.12 source utils/env_install.sh $IRON_HOME/ironenv` from that checkout. The script requires exactly Python 3.12 (`uv python install 3.12` provides one).
3. `xclbinutil` packages the xclbin. Without root, unpack it from Ubuntu's `libxrt-utils` package plus its two Boost libraries into `$IRON_HOME/xrt-local`:

   ```
   apt-get download libxrt-utils libboost-filesystem1.90.0 libboost-program-options1.90.0
   for d in *.deb; do dpkg-deb -x "$d" $IRON_HOME/xrt-local; done
   ```

4. `iron/build.sh gemv 8192 2048 8 8` or `iron/build.sh memcpy 16777216`. `IRON_HOME` defaults to `~/xdna-iron`, and `OUT_DIR` defaults to `./out`. For the GEMV, `M / (32 x n_cores)` must be at most 64 (a DMA buffer-descriptor repeat limit), and `n_cols` must be at least `n_cores`.

The memcpy bypass variant in the results is mlir-aie's own `programming_examples/basic/memcpy/memcpy.py --dev npu2 -l 16777216 -co 8 -ch 2 -b True` with `--xclbin-path` and `--insts-path`.

## Licenses

`spike.cpp`, `build.ps1` and `iron/build.sh` are cera's (Apache-2.0 OR MIT). `iron/gemv_multi.py` and `iron/memcpy_bw.py` are modified copies of mlir-aie examples and keep their `Apache-2.0 WITH LLVM-exception` headers, with the modifications noted at the top of each file. The XRT headers are not vendored here.

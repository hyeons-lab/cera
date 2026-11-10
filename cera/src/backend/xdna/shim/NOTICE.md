# XDNA shim third-party material

- `include/` — XRT user-space headers from Xilinx/XRT at commit
  `77c7088d804602a53c3eb489b9cb37b709bcd751` (release 2.19.0, the version the
  AMD NPU driver 32.0.203.311 ships), Apache-2.0. `xrt/detail/version.h` was
  generated from `src/CMake/config/version.h.in` for 2.19.0 as described in
  `tools/xdna-spike/README.md`. Only headers are vendored; the runtime itself
  (`xrt_coreutil.dll`) is part of the user-installed AMD driver and is never
  bundled.
- `xrt_coreutil.def` — the export name list of that driver DLL, used at build
  time to generate an import library so the shim compiles on machines without
  the driver installed.
- `cera_xdna_shim.cpp` — cera's own code, Apache-2.0 OR MIT.

Kernel `.xclbin` artifacts and their IRON/mlir-aie license notices live with
the kernels, not here.

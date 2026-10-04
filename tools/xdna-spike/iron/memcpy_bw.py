# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Adapted for cera tools/xdna-spike from mlir-aie
# programming_examples/getting_started/00_memcpy/memcpy.py.
# Compile-only wrapper around getting_started/00_memcpy: transform_parallel
# over every shim DMA in/out pair (num_channels=2), library passthrough kernel.
import argparse

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli


@iron.jit
def my_memcpy(input0: In, output: Out, *, size: CompileTime[int]):
    return transform_parallel(
        kernels.passthrough(tile_size=1024, dtype=np.int32),
        np.ndarray[(size,), np.dtype[np.int32]],
        tile_size=1024,
        num_channels=2,
        pass_size_to_kernel=True,
    )


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p)
    p.add_argument("-l", "--length", type=int, default=16777216)
    opts = p.parse_args()
    run_design_cli(
        my_memcpy,
        opts,
        compile_kwargs=lambda o: dict(size=o.length),
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()

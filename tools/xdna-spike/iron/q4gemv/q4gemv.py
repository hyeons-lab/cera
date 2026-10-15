# Q4_0-style GEMV design for tools/xdna-spike (Phase 0b kernel spike).
#
# One worker per (column, shim channel) pair, 16 on npu2. Each worker owns
# M/16 output rows and gets one linear stream from DDR on its own shim channel:
# first an 18432-byte slot carrying x (int8) and its block scales (f32), then
# its weight tiles (16 rows x K, 18432 bytes each at K = 2048), pre-packed by
# the host in consumption order so every transfer is a plain linear read.
# x rides the weight channel because the weights need all 16 shim channels.
import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import get_current_device
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

ROWS = 16  # output rows per tile
QBLK = 32
OUT_TILES = 16  # tiles per output element (256 floats)
NUM_CHANNELS = 2


def tile_bytes(K):
    return (K // QBLK) * (2 * 128 + ROWS * 2)


@iron.jit
def q4gemv(w: In, x: In, out: Out, *, M: CompileTime[int], K: CompileTime[int]):
    workers_n = get_current_device().cols * NUM_CHANNELS
    tb = tile_bytes(K)
    if M % (workers_n * ROWS * OUT_TILES):
        raise ValueError(f"M must be a multiple of {workers_n * ROWS * OUT_TILES}")
    rows_per_worker = M // workers_n
    tiles_per_worker = rows_per_worker // ROWS
    outs_per_worker = tiles_per_worker // OUT_TILES
    if K % 64:
        raise ValueError("K must be a multiple of 64 (q4gemv.cc copies x 64 bytes at a time)")
    if outs_per_worker < 2:
        raise ValueError("need at least 2 output elements per worker (M >= 8192 on npu2)")

    w_ty = np.ndarray[(workers_n * tiles_per_worker * tb,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(tb,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(M,), np.dtype[np.float32]]
    slot_ty = np.ndarray[(tb,), np.dtype[np.uint8]]
    outel_ty = np.ndarray[(ROWS * OUT_TILES,), np.dtype[np.float32]]

    step = ExternalFunction(
        "q4_step",
        source_file=str(Path(__file__).parent / "q4gemv.cc"),
        arg_types=[slot_ty, outel_ty, np.int32, np.int32],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DKMAX={K}"],
    )

    of_ins = [ObjectFifo(slot_ty, name=f"w_{i}") for i in range(workers_n)]
    of_outs = [ObjectFifo(outel_ty, name=f"y_{i}") for i in range(workers_n)]

    def core_body(of_in, of_out, step):
        # First output element: acquire it, load x from the stream's first
        # slot, then its tiles. step(t = -1) loads x and ignores y.
        y = of_out.acquire(1)
        slot = of_in.acquire(1)
        step(slot, y, K, -1)
        of_in.release(1)
        for t in range_(OUT_TILES):
            tile = of_in.acquire(1)
            step(tile, y, K, t)
            of_in.release(1)
        of_out.release(1)
        for _ in range_(outs_per_worker - 1):
            y = of_out.acquire(1)
            for t in range_(OUT_TILES):
                tile = of_in.acquire(1)
                step(tile, y, K, t)
                of_in.release(1)
            of_out.release(1)

    workers = [Worker(core_body, [of_ins[i].cons(), of_outs[i].prod(), step]) for i in range(workers_n)]

    per_worker_w = tiles_per_worker * tb
    x_tap = TensorAccessPattern((1, tb), 0, [1, 1, 1, tb], [0, 0, 0, 1])
    w_taps = [
        TensorAccessPattern((1, workers_n * per_worker_w), per_worker_w * i, [1, 1, 1, per_worker_w], [0, 0, 0, 1])
        for i in range(workers_n)
    ]
    y_taps = [
        TensorAccessPattern((1, M), rows_per_worker * i, [1, 1, 1, rows_per_worker], [0, 0, 0, 1])
        for i in range(workers_n)
    ]

    def sequence(w_in, x_in, y_out, in_prods, out_conses):
        tg_in = TaskGroup()
        for i in range(workers_n):
            in_prods[i].fill(x_in, x_tap, group=tg_in)
            in_prods[i].fill(w_in, w_taps[i], group=tg_in)
        tg_out = TaskGroup()
        for i in range(workers_n):
            out_conses[i].drain(y_out, y_taps[i], wait=True, group=tg_out)
        tg_out.finish()
        tg_in.finish()

    rt = Runtime(sequence, [w_ty, x_ty, out_ty, [of.prod() for of in of_ins], [of.cons() for of in of_outs]])
    return Program(get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p)
    p.add_argument("-M", type=int, default=8192)
    p.add_argument("-K", type=int, default=2048)
    opts = p.parse_args()
    run_design_cli(
        q4gemv,
        opts,
        compile_kwargs=lambda o: dict(M=o.M, K=o.K),
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()

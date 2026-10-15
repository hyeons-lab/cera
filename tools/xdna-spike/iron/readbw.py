# Read-dominant DDR bandwidth design for tools/xdna-spike.
#
# One worker per (column, shim channel) pair, as in iron.algorithms
# transform_parallel(num_channels=2), but each worker only consumes its input
# stream: for every tile it adds the tile's first word into word 0 of a small
# output tile, so the host can check every tile arrived. Output traffic is 16
# int32 words per worker, so DDR traffic is almost all reads.
import argparse

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import CompileTime, In, Out
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import get_current_device
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

OUT_WORDS = 16
NUM_CHANNELS = 2


@iron.jit
def readbw(input0: In, output: Out, *, size: CompileTime[int], tile_size: CompileTime[int] = 1024):
    num_columns = get_current_device().cols
    workers_n = num_columns * NUM_CHANNELS
    if size % (tile_size * workers_n) != 0:
        raise ValueError(f"size must be a multiple of tile_size x {workers_n}")
    per_worker = size // workers_n
    tiles_per_worker = per_worker // tile_size

    in_ty = np.ndarray[(size,), np.dtype[np.int32]]
    out_ty = np.ndarray[(workers_n * OUT_WORDS,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    acc_ty = np.ndarray[(OUT_WORDS,), np.dtype[np.int32]]

    of_ins = [ObjectFifo(tile_ty, name=f"in_{w}") for w in range(workers_n)]
    of_outs = [ObjectFifo(acc_ty, name=f"out_{w}") for w in range(workers_n)]

    def core_body(of_in, of_out):
        acc = of_out.acquire(1)
        acc[0] = 0
        for _ in range_(tiles_per_worker):
            elem = of_in.acquire(1)
            acc[0] = acc[0] + elem[0]
            of_in.release(1)
        of_out.release(1)

    workers = [Worker(core_body, [of_ins[w].cons(), of_outs[w].prod()]) for w in range(workers_n)]
    in_taps = [
        TensorAccessPattern((1, size), per_worker * w, [1, 1, 1, per_worker], [0, 0, 0, 1])
        for w in range(workers_n)
    ]
    out_taps = [
        TensorAccessPattern((1, workers_n * OUT_WORDS), OUT_WORDS * w, [1, 1, 1, OUT_WORDS], [0, 0, 0, 1])
        for w in range(workers_n)
    ]

    def sequence(a, c, in_prods, out_conses):
        tg_in = TaskGroup()
        for w in range(workers_n):
            in_prods[w].fill(a, in_taps[w], group=tg_in)
        tg_out = TaskGroup()
        for w in range(workers_n):
            out_conses[w].drain(c, out_taps[w], wait=True, group=tg_out)
        tg_out.finish()
        tg_in.finish()

    rt = Runtime(
        sequence,
        [in_ty, out_ty, [of.prod() for of in of_ins], [of.cons() for of in of_outs]],
    )
    return Program(get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p)
    p.add_argument("-l", "--length", type=int, default=16777216)
    p.add_argument("-t", "--tile", type=int, default=1024)
    opts = p.parse_args()
    run_design_cli(
        readbw,
        opts,
        compile_kwargs=lambda o: dict(size=o.length, tile_size=o.tile),
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()

# MemTile-tiled Q4_0 prefill GEMM design for tools/xdna-spike (Gate 1).
#
# Maps the 8x4 = 32 AIE2P compute cores on npu2 as:
#   - 4 rows: split the token batch B = 4 * B_CORE (32 tokens at B_CORE=8)
#   - 8 cols: split N_out into 8 * N_col (N_col multiple of 16)
#
# MemTiles stage and broadcast both operands:
#   - X_L3L2[r] -> X_L2L1[r] (4 rows): each row's X chunk (B_CORE x 2048)
#     streams once from DDR and broadcasts horizontally across all 8 cols.
#   - W_L3L2[c] -> W_L2L1[c] (8 cols): each col's Q4_0 tiles (16 x 2048,
#     exact repack.rs blocks-then-scales layout) stream once from DDR and
#     broadcast vertically across all 4 rows in that col.
#   - Y_L1L2[c][r] -> Y_L2L3[c] (8 cols): 4 rows join in MemTile[c] and drain
#     directly to DDR Y[B, N_out] row-major.
#
# All Llama-3.2-1B shapes (512x2048, 2048x2048, 8192x2048, 2048x8192) share
# the exact same compiled xclbin (core loops read n_passes/pass_tiles/k_chunks
# from X's header) with per-shape instruction binaries.
import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import arith
from aie.extras import types as T
from aie.helpers.dialects.scf import else_, if_
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

N_ROWS = 4
N_COLS = 8
B_CORE = 8
B = N_ROWS * B_CORE  # 32 tokens per prefill batch
ROWS = 16
QBLK = 32
CHUNK_K = 2048
NB_CHUNK = CHUNK_K // QBLK
TILE_F32 = 4608  # 18,432 bytes = one 16x2048 blocks-then-scales Q4 tile
X_DATA_BYTES = B_CORE * CHUNK_K + B_CORE * NB_CHUNK * 2  # 16,384 + 1,024 = 17,408
X_HDR_BYTES = 64
X_SLOT_F32 = (X_DATA_BYTES + X_HDR_BYTES) // 4  # 4,368 floats = 17,472 bytes
HDR_F32_OFF = X_DATA_BYTES // 4  # 4,352
Y_CORE_F32 = B_CORE * ROWS  # 128 floats = 512 bytes
Y_COL_F32 = N_ROWS * Y_CORE_F32  # 512 floats = 2,048 bytes
MAX_ACC_TILES = 8


@iron.jit
def q4prefill(w_in: In, x_in: In, y_out: Out, *, N: CompileTime[int], K: CompileTime[int]):
    dev = get_current_device()
    if dev.cols != N_COLS:
        raise ValueError(f"q4prefill requires {N_COLS} columns, got {dev.cols}")
    if N <= 0 or N % (N_COLS * ROWS) != 0:
        raise ValueError(f"N must be a positive multiple of {N_COLS * ROWS}, got {N}")
    if K <= 0 or K % CHUNK_K != 0:
        raise ValueError(f"K must be a positive multiple of {CHUNK_K}, got {K}")

    n_col = N // N_COLS
    tiles_per_col = n_col // ROWS
    k_chunks = K // CHUNK_K
    pass_tiles = MAX_ACC_TILES if (k_chunks > 1 and tiles_per_col > MAX_ACC_TILES) else tiles_per_col
    if tiles_per_col % pass_tiles != 0:
        raise ValueError(f"tiles_per_col={tiles_per_col} not divisible by pass_tiles={pass_tiles}")
    n_passes = tiles_per_col // pass_tiles

    w_total_f32 = N_COLS * tiles_per_col * k_chunks * TILE_F32
    x_total_f32 = k_chunks * N_ROWS * X_SLOT_F32
    y_total_f32 = B * N

    w_ty = np.ndarray[(w_total_f32,), np.dtype[np.float32]]
    x_ty = np.ndarray[(x_total_f32,), np.dtype[np.float32]]
    y_ty = np.ndarray[(y_total_f32,), np.dtype[np.float32]]

    x_slot_ty = np.ndarray[(X_SLOT_F32,), np.dtype[np.float32]]
    w_slot_ty = np.ndarray[(TILE_F32,), np.dtype[np.float32]]
    y_core_ty = np.ndarray[(Y_CORE_F32,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_F32,), np.dtype[np.float32]]

    tile_fn = ExternalFunction(
        "q4_prefill_tile",
        source_file=str(Path(__file__).parent / "q4prefill.cc"),
        arg_types=[x_slot_ty, w_slot_ty, y_core_ty, np.int32],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DB_CORE={B_CORE}"],
    )
    acc_fn = ExternalFunction(
        "q4_prefill_acc",
        source_file=str(Path(__file__).parent / "q4prefill_acc.cc"),
        arg_types=[x_slot_ty, w_slot_ty, np.int32, np.int32],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DB_CORE={B_CORE}"],
    )

    x_l3l2 = [ObjectFifo(x_slot_ty, name=f"X_L3L2_{r}", depth=1) for r in range(N_ROWS)]
    x_l2l1 = [x_l3l2[r].cons().forward(obj_type=x_slot_ty, name=f"X_L2L1_{r}") for r in range(N_ROWS)]

    w_l3l2 = [ObjectFifo(w_slot_ty, name=f"W_L3L2_{c}", depth=2) for c in range(N_COLS)]
    w_l2l1 = [w_l3l2[c].cons().forward(obj_type=w_slot_ty, name=f"W_L2L1_{c}") for c in range(N_COLS)]

    y_l2l3 = [ObjectFifo(y_col_ty, name=f"Y_L2L3_{c}", depth=2) for c in range(N_COLS)]
    y_l1l2_cols = [
        y_l2l3[c].prod().join(
            [r * Y_CORE_F32 for r in range(N_ROWS)],
            obj_types=[y_core_ty] * N_ROWS,
            names=[f"Y_L1L2_{c}_{r}" for r in range(N_ROWS)],
            depths=[2] * N_ROWS,
        )
        for c in range(N_COLS)
    ]

    def run_multi_k_pass(of_x, of_w, of_y, tile_fn, acc_fn, x_first, pass_tiles, k_chunks, use_first_x):
        c0 = arith.constant(T.i32(), 0)
        c1 = arith.constant(T.i32(), 1)
        c2 = arith.constant(T.i32(), 2)
        # ch = 0: store into y_acc[t]
        x = x_first if use_first_x else of_x.acquire(1)
        for t in range_(pass_tiles):
            w = of_w.acquire(1)
            acc_fn(x, w, t, c0)
            of_w.release(1)
        of_x.release(1)
        # ch = 1 .. k_chunks - 2: accumulate into y_acc[t]
        mid_chunks = arith.subi(k_chunks, c2)
        for _ in range_(mid_chunks):
            x = of_x.acquire(1)
            for t in range_(pass_tiles):
                w = of_w.acquire(1)
                acc_fn(x, w, t, c1)
                of_w.release(1)
            of_x.release(1)
        # ch = k_chunks - 1: add y_acc[t] and drain to out
        x = of_x.acquire(1)
        for t in range_(pass_tiles):
            w = of_w.acquire(1)
            y = of_y.acquire(1)
            tile_fn(x, w, y, t)
            of_w.release(1)
            of_y.release(1)
        of_x.release(1)

    def core_body(of_x, of_w, of_y, tile_fn, acc_fn):
        x0 = of_x.acquire(1)
        np_cnt = arith.bitcast(T.i32(), x0[HDR_F32_OFF + 0])
        pt_cnt = arith.bitcast(T.i32(), x0[HDR_F32_OFF + 1])
        kc_cnt = arith.bitcast(T.i32(), x0[HDR_F32_OFF + 2])

        c1 = arith.constant(T.i32(), 1)
        cm1 = arith.constant(T.i32(), -1)
        is_multi_k = arith.cmpi(arith.CmpIPredicate.sgt, kc_cnt, c1)

        with if_(is_multi_k) as if_op:
            run_multi_k_pass(of_x, of_w, of_y, tile_fn, acc_fn, x0, pt_cnt, kc_cnt, True)
            rem_passes = arith.subi(np_cnt, c1)
            for _ in range_(rem_passes):
                run_multi_k_pass(of_x, of_w, of_y, tile_fn, acc_fn, x0, pt_cnt, kc_cnt, False)
        with else_(if_op):
            for _ in range_(pt_cnt):
                w = of_w.acquire(1)
                y = of_y.acquire(1)
                tile_fn(x0, w, y, cm1)
                of_w.release(1)
                of_y.release(1)
            of_x.release(1)

    workers_grid = Worker.grid(
        N_ROWS,
        N_COLS,
        lambda r, c: Worker(
            core_body,
            [x_l2l1[r].cons(), w_l2l1[c].cons(), y_l1l2_cols[c][r].prod(), tile_fn, acc_fn],
            stack_size=2048,
        ),
    )
    workers = [w for row in workers_grid for w in row]

    def tap(total, offset, size):
        return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])

    def sequence(w_in, x_in, y_out, x_prods, w_prods, y_conses):
        for p in range(n_passes):
            t0 = p * pass_tiles
            tg = TaskGroup()
            for ch in range(k_chunks):
                for r in range(N_ROWS):
                    x_prods[r].fill(
                        x_in,
                        tap(x_total_f32, (ch * N_ROWS + r) * X_SLOT_F32, X_SLOT_F32),
                        group=tg,
                    )
                for c in range(N_COLS):
                    if k_chunks == 1:
                        w_prods[c].fill(
                            w_in,
                            tap(w_total_f32, c * tiles_per_col * TILE_F32, pass_tiles * TILE_F32),
                            group=tg,
                        )
                    else:
                        w_prods[c].fill(
                            w_in,
                            TensorAccessPattern(
                                (1, w_total_f32),
                                (c * tiles_per_col * k_chunks + t0 * k_chunks + ch) * TILE_F32,
                                [1, pass_tiles, 9, 512],
                                [0, k_chunks * TILE_F32, 512, 1],
                            ),
                            group=tg,
                        )
            for c in range(N_COLS):
                y_conses[c].drain(
                    y_out,
                    TensorAccessPattern(
                        (B, N),
                        c * n_col + t0 * ROWS,
                        [1, pass_tiles, B, ROWS],
                        [0, ROWS, N, 1],
                    ),
                    wait=True,
                    group=tg,
                )
            tg.finish()

    rt = Runtime(
        sequence,
        [
            w_ty,
            x_ty,
            y_ty,
            [f.prod() for f in x_l3l2],
            [f.prod() for f in w_l3l2],
            [f.cons() for f in y_l2l3],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-K", type=int, default=2048)
    opts = p.parse_args()
    run_design_cli(
        q4prefill,
        opts,
        compile_kwargs=lambda o: dict(N=o.N, K=o.K),
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()

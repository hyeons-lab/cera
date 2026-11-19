# Dense bfloat16 x bfloat16 -> float32 GEMM across all 32 NPU2 cores (4 rows x 8 cols).
# Computes C[M_TILE=128, N_TILE=256] = A[128, K] * B[256, K]^T in f32 accumulation.
#
# Layout:
#   A (arg 3): row-major [128, K] bfloat16 (passed as uint32[128 * (K / 2)])
#   B (arg 4): per-col contiguous packed tiles [8 cols][K_CHUNKS][k8=8][n8=4][8_k][8_n] bfloat16
#              (passed as uint32[8 * K_CHUNKS * 1024])
#   C (arg 5): row-major [128, 256] float32

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import arith
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, Out
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.ir import IndexType, IntegerType
from aie.utils import get_current_device
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

N_ROWS = 4
N_COLS = 8
M_CORE = 32
N_CORE = 32
K_CHUNK = 64
M_TILE = N_ROWS * M_CORE  # 128
N_TILE = N_COLS * N_CORE  # 256

X_U32 = (M_CORE * K_CHUNK) // 2  # 1024 uint32 = 4096 B
W_U32 = (N_CORE * K_CHUNK) // 2  # 1024 uint32 = 4096 B
Y_F32 = M_CORE * N_CORE          # 1024 float32 = 4096 B
Y_COL_F32 = N_ROWS * Y_F32       # 4096 float32 = 16384 B

MAX_K_CHUNKS = 128  # K up to 8192


@iron.jit
def gemm_bf16(a: In, b: In, c: Out, p: In, k_chunks: CompileTime[int] = 16):
    dev = get_current_device()
    if dev.cols < N_COLS:
        raise ValueError(f"need {N_COLS} columns, got {dev.cols}")
    if k_chunks <= 0 or k_chunks > MAX_K_CHUNKS:
        raise ValueError(f"k_chunks must be in 1..{MAX_K_CHUNKS}, got {k_chunks}")

    k_total = k_chunks * K_CHUNK
    a_u32_total = M_TILE * (k_total // 2)
    b_u32_total = N_COLS * k_chunks * W_U32
    c_f32_total = M_TILE * N_TILE

    x_slot_ty = np.ndarray[(X_U32,), np.dtype[np.uint32]]
    w_slot_ty = np.ndarray[(W_U32,), np.dtype[np.int32]]
    y_core_ty = np.ndarray[(Y_F32,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_F32,), np.dtype[np.float32]]

    a_bo_ty = np.ndarray[(a_u32_total,), np.dtype[np.uint32]]
    b_bo_ty = np.ndarray[(b_u32_total,), np.dtype[np.int32]]
    c_bo_ty = np.ndarray[(c_f32_total,), np.dtype[np.float32]]
    p_bo_ty = np.ndarray[(W_U32,), np.dtype[np.int32]]

    here = Path(__file__).parent
    gemm_op = ExternalFunction(
        "gemm_bf16_tile",
        source_file=str(here / "gemm_bf16_core.cc"),
        arg_types=[x_slot_ty, w_slot_ty, y_core_ty, np.int32],
        include_dirs=[cxx_header_path()],
        compile_flags=["-O2"],
    )

    x_l3l2 = [ObjectFifo(x_slot_ty, name=f"X_L3L2_{r}", depth=2) for r in range(N_ROWS)]
    x_l2l1 = [
        x_l3l2[r].cons().forward(obj_type=x_slot_ty, name=f"X_L2L1_{r}", depth=2)
        for r in range(N_ROWS)
    ]

    w_l3l2 = [ObjectFifo(w_slot_ty, name=f"W_L3L2_{c}", depth=2) for c in range(N_COLS)]
    w_l2l1 = [
        w_l3l2[c].cons().forward(obj_type=w_slot_ty, name=f"W_L2L1_{c}", depth=2)
        for c in range(N_COLS)
    ]

    y_l2l3 = [ObjectFifo(y_col_ty, name=f"Y_L2L3_{c}", depth=1) for c in range(N_COLS)]
    y_l1l2_cols = [
        y_l2l3[c].prod().join(
            [r * Y_F32 for r in range(N_ROWS)],
            obj_types=[y_core_ty] * N_ROWS,
            names=[f"Y_L1L2_{c}_{r}" for r in range(N_ROWS)],
            depths=[1] * N_ROWS,
        )
        for c in range(N_COLS)
    ]

    def core_body(x_in, w_in, y_out, fn_gemm):
        idx_ty = IndexType.get()
        i32_ty = IntegerType.get_signless(32)
        idx0 = arith.constant(idx_ty, 0)
        i32_0 = arith.constant(i32_ty, 0)
        i32_1 = arith.constant(i32_ty, 1)

        cp = w_in.acquire(1)
        k_iters = arith.index_cast(idx_ty, cp[idx0])
        w_in.release(1)

        yo = y_out.acquire(1)
        for kc in range_(k_iters):
            is_acc = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
            flag = arith.select(is_acc, i32_1, i32_0)
            xp = x_in.acquire(1)
            wp = w_in.acquire(1)
            fn_gemm(xp, wp, yo, flag)
            x_in.release(1)
            w_in.release(1)
        y_out.release(1)

    workers = [
        Worker(
            core_body,
            [
                x_l2l1[r].cons(),
                w_l2l1[c].cons(),
                y_l1l2_cols[c][r].prod(),
                gemm_op,
            ],
            stack_size=3072,
        )
        for c in range(N_COLS)
        for r in range(N_ROWS)
    ]

    def tap(total, off, sizes, strides):
        return TensorAccessPattern((1, total), off, sizes, strides)

    row_u32 = k_total // 2
    col_w_u32 = k_chunks * W_U32

    def sequence(a_buf, b_buf, c_buf, p_buf, x_prods, w_prods, y_conses):
        tg = TaskGroup()
        for r in range(N_ROWS):
            x_prods[r].fill(
                a_buf,
                tap(a_u32_total, r * M_CORE * row_u32, [1, k_chunks, M_CORE, 32], [0, 32, row_u32, 1]),
                group=tg,
            )
        for c in range(N_COLS):
            w_prods[c].fill(
                p_buf,
                tap(W_U32, 0, [1, 1, 1, W_U32], [0, 0, 0, 1]),
                group=tg,
            )
            w_prods[c].fill(
                b_buf,
                tap(b_u32_total, c * col_w_u32, [1, 1, k_chunks, W_U32], [0, 0, W_U32, 1]),
                group=tg,
            )
        for c in range(N_COLS):
            y_conses[c].drain(
                c_buf,
                tap(c_f32_total, c * N_CORE, [1, 1, M_TILE, N_CORE], [0, 0, N_TILE, 1]),
                wait=True,
                group=tg,
            )
        tg.finish()

    rt = Runtime(
        sequence,
        [
            a_bo_ty,
            b_bo_ty,
            c_bo_ty,
            p_bo_ty,
            [of.prod() for of in x_l3l2],
            [of.prod() for of in w_l3l2],
            [of.cons() for of in y_l2l3],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(description="Dense bf16 GEMM kernel for AMD XDNA2")
    p.add_argument("--k-chunks", type=int, default=16)
    add_compile_args(p)
    opts = p.parse_args()
    run_design_cli(
        gemm_bf16,
        opts,
        compile_kwargs=lambda o: dict(k_chunks=o.k_chunks),
        device=lambda o: device_from_args(o, n_cols=N_COLS),
    )


if __name__ == "__main__":
    main()

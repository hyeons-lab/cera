# Fused B=32 Q4_0 LFM3 ShortConv prefill layer (32 cores: 4 rows x 8 cols, npu2).
# Dedicated Mode 0 kernel using compact schedule-table interpreter so PM <= 16 KiB & DM <= 64 KiB.

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import arith
from aie.helpers.dialects.scf import else_, if_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, ExternalFunction, In, Out
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.device import NPU2, Tile
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.ir import IndexType, IntegerType
from aie.utils import get_current_device
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

N_ROWS = 4
N_COLS = 8
B_CORE = 8
B = N_ROWS * B_CORE  # 32
ROWS = 16
CHUNK_K = 2048

W_ELEMS = 4608
X_ELEMS = 5120
Y_CORE_ELEMS = 128
Y_COL_ELEMS = N_ROWS * Y_CORE_ELEMS  # 512

H = 2048
MAX_KV_CHUNKS = 256
N_KV_HEADS = 4

# Same unified weight / activation DDR offsets as q4prefill_lfm3_attn so BOs share directly.
W_CONV_IN_B = 0
W_CONV_IN_C = 128 * W_ELEMS
W_CONV_IN_X = 256 * W_ELEMS
W_CONV_OUT = 384 * W_ELEMS
W_CONV_WT = 512 * W_ELEMS
W_CONV_FFN_GATE = 520 * W_ELEMS
W_CONV_FFN_UP = 1000 * W_ELEMS
W_CONV_FFN_DOWN = 1480 * W_ELEMS
W_TOTAL = 1992 * W_ELEMS

SLOT_HIDDEN_IN = 0
SLOT_ATTN_OUT = 163840
SLOT_RES_TMP = 229376
SLOT_NORMED_B = 294912
SLOT_GATE = 360448
SLOT_CONV_B = SLOT_GATE
SLOT_CONV_C = SLOT_GATE + 65536
SLOT_CONV_X = SLOT_GATE + 131072
SLOT_UP = 607232
SLOT_XQ = 854016
SLOT_NORM_0 = 977408
SLOT_NORM_1 = 982016
ACT_TOTAL = 986624 + 4608 + 1024

KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * W_ELEMS
KV_OUT_TOTAL = 8 * W_ELEMS


@iron.jit(use_cache=False)
def q4prefill_lfm3_conv(
    w: In,
    y: Out,
    y_next: Out,
    kv: In,
    kv_out: Out,
):
    here = Path(__file__).resolve().parent
    cxx_inc = cxx_header_path()

    w_ty = np.ndarray[(W_TOTAL,), np.dtype[np.float32]]
    y_ty = np.ndarray[(ACT_TOTAL,), np.dtype[np.float32]]
    yn_ty = np.ndarray[(65536,), np.dtype[np.float32]]
    kv_ty = np.ndarray[(KV_TOTAL,), np.dtype[np.float32]]
    kvo_ty = np.ndarray[(KV_OUT_TOTAL,), np.dtype[np.float32]]

    w_pkt_ty = np.ndarray[(W_ELEMS,), np.dtype[np.float32]]
    x_pkt_ty = np.ndarray[(X_ELEMS,), np.dtype[np.float32]]
    x_l3l2_ty = np.ndarray[(N_ROWS * X_ELEMS,), np.dtype[np.float32]]
    y_core_ty = np.ndarray[(Y_CORE_ELEMS,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_ELEMS,), np.dtype[np.float32]]

    common_flags = ["-I", str(here), "-I", str(cxx_inc)]
    core_flags = common_flags + ["-Os"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm3_conv_gemm_op",
        source_file=str(here / "q4_lfm3_gemm_op.cc"),
        arg_types=[x_pkt_ty, w_pkt_ty, np.int32, np.int32],
        compile_flags=common_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm3_conv_w_op",
        source_file=str(here / "q4_lfm3_conv_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=core_flags,
    )
    q4_x_fn = ExternalFunction(
        "q4_lfm3_conv_x_op",
        source_file=str(here / "q4_lfm3_conv_x_op.cc"),
        arg_types=[x_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=core_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm3_conv_y_op",
        source_file=str(here / "q4prefill_lfm3_conv_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=core_flags,
    )

    W_L3L2 = [ObjectFifo(w_pkt_ty, depth=2, name=f"W_L3L2_{c}") for c in range(N_COLS)]
    W_L2L1 = [W_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=1, name=f"W_L2L1_{c}") for c in range(N_COLS)]

    X_L3L2 = [ObjectFifo(x_l3l2_ty, depth=1, name=f"X_L3L2_{c}") for c in range(N_COLS)]
    X_L2L1 = [
        X_L3L2[c].cons().split(
            offsets=[r * X_ELEMS for r in range(N_ROWS)],
            obj_types=[x_pkt_ty] * N_ROWS,
            names=[f"X_L2L1_{c}_{r}" for r in range(N_ROWS)],
            depths=[1] * N_ROWS,
        )
        for c in range(N_COLS)
    ]

    Y_L2L3 = [ObjectFifo(y_col_ty, depth=2, name=f"Y_L2L3_{c}") for c in range(N_COLS)]
    Y_L1L2 = [
        Y_L2L3[c].prod().join(
            offsets=[r * Y_CORE_ELEMS for r in range(N_ROWS)],
            obj_types=[y_core_ty] * N_ROWS,
            names=[f"Y_L1L2_{c}_{r}" for r in range(N_ROWS)],
            depths=[1] * N_ROWS,
        )
        for c in range(N_COLS)
    ]

    KIND_W = 0
    KIND_X = 1
    KIND_Y = 2
    KIND_GEMM = 3

    def build_worker_sched(c, r):
        rows = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0):
            rows.append((kind, cnt, op, a0, a1, a2))

        def emit_rms(norm_op_kind, norm_slot, skip_norm, f32_out):
            if norm_op_kind >= 0:
                emit(KIND_W, 1, norm_op_kind, c, norm_slot)
            emit(KIND_X, 4, 0, c, 0, skip_norm | (f32_out << 1))
            if f32_out == 0:
                emit(KIND_Y, 5, 0, 0)
            else:
                emit(KIND_Y, 16, 6, 0)

        def emit_gemm(n_passes, pass_tiles, k_chunks, xd_mode):
            emit(KIND_GEMM, n_passes, xd_mode, k_chunks, pass_tiles)

        def emit_res():
            for p in range(4):
                emit(KIND_X, 1, 2, c, 0, p)
                emit(KIND_X, 1, 2, c, 1, p)
            emit(KIND_Y, 16, 4, 0)

        # 1. Attn RMSNorm + Q8 (metadata loaded via first op=0 w_pkt)
        emit_rms(0, 0, 0, 0)
        # 2. B, C, X IN_PROJ (each 2 passes x 8 tiles, k_chunks=1, xd_mode=0)
        for _ in range(3):
            emit_gemm(2, 8, 1, 0)
        # 3. 7-tap causal ShortConv + conv_state update
        emit(KIND_W, 1, 3, r, 0)   # conv_wt
        emit(KIND_W, 5, 5, r, 0)   # stages 0..4: conv_state, B0, B1, X0, X1
        emit(KIND_W, 1, 5, r, 5)   # C0 -> o_chunk[0..15]
        emit(KIND_Y, 8, 8, 0)
        emit(KIND_W, 1, 5, r, 6)   # C1 -> o_chunk[16..31]
        emit(KIND_Y, 8, 8, 0)
        emit(KIND_Y, 3, 8, 8)      # arg = 8, 9, 10 -> new_conv_state -> kv_out
        # 4. Q8(SLOT_ATTN_OUT) + OUT_PROJ + post_attn_norm + Res1
        emit_rms(-1, 0, 1, 0)
        emit_gemm(2, 8, 1, 0)
        emit_rms(4, 1, 0, 1)
        emit_res()
        # 5. FFN norm + Gate/Up + SiLU*Up(7680 padded->8192) + Down + post_ffn_norm + Res2
        emit_rms(4, 0, 0, 0)
        emit_gemm(15, 4, 1, 0)
        emit_gemm(15, 4, 1, 0)
        for kc in range(4):
            for j in range(4):
                if kc == 3 and j == 3:
                    emit(KIND_X, 1, 3, c, 3, 2)
                else:
                    emit(KIND_X, 1, 3, c, j, 0)
                    emit(KIND_X, 1, 3, c, j, 1)
                emit(KIND_Y, 1, 5, 0)
            emit(KIND_Y, 1, 5, 1)
        emit_gemm(2, 8, 4, 1)
        emit_rms(4, 1, 0, 1)
        emit_res()
        return np.array(rows, dtype=np.int32).reshape(-1)

    sample_sched = build_worker_sched(0, 0)
    n_steps = len(sample_sched) // 6
    sched_ty = np.ndarray[(len(sample_sched),), np.dtype[np.int32]]
    sched_bufs = [
        [Buffer(sched_ty, initial_value=build_worker_sched(c, r), name=f"sc_{c}_{r}") for r in range(N_ROWS)]
        for c in range(N_COLS)
    ]

    def core_body(w_fifo, x_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_w_fn, q4_x_fn, q4_y_fn):
        idx_ty = IndexType.get()
        i32_ty = IntegerType.get_signless(32)
        idx0 = arith.constant(idx_ty, 0)
        idx1 = arith.constant(idx_ty, 1)
        idx2 = arith.constant(idx_ty, 2)
        idx3 = arith.constant(idx_ty, 3)
        idx4 = arith.constant(idx_ty, 4)
        idx5 = arith.constant(idx_ty, 5)
        idx6 = arith.constant(idx_ty, 6)
        i32_0 = arith.constant(i32_ty, 0)
        i32_1 = arith.constant(i32_ty, 1)
        i32_2 = arith.constant(i32_ty, 2)

        def i32(v):
            return arith.index_cast(i32_ty, v)

        for s in range_(n_steps):
            base = arith.muli(s, idx6)
            kind = sched_buf[base]
            cnt = sched_buf[arith.addi(base, idx1)]
            op = sched_buf[arith.addi(base, idx2)]
            a0 = sched_buf[arith.addi(base, idx3)]
            a1 = sched_buf[arith.addi(base, idx4)]
            a2 = sched_buf[arith.addi(base, idx5)]

            is_w = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_0)
            with if_(is_w, hasElse=True) as if_w:
                for i in range_(cnt):
                    wp = w_fifo.acquire(1)
                    q4_w_fn(wp, op, a0, arith.addi(a1, i32(i)))
                    w_fifo.release(1)
            with else_(if_w):
                is_x = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_1)
                with if_(is_x, hasElse=True) as if_x:
                    for i in range_(cnt):
                        xp = x_fifo.acquire(1)
                        q4_x_fn(xp, op, a0, arith.addi(a1, i32(i)), a2)
                        x_fifo.release(1)
                with else_(if_x):
                    is_y = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_2)
                    with if_(is_y, hasElse=True) as if_y:
                        for i in range_(cnt):
                            yp = y_fifo.acquire(1)
                            q4_y_fn(yp, op, arith.addi(a0, i32(i)))
                            y_fifo.release(1)
                    with else_(if_y):
                        last_k = arith.subi(a0, i32_1)
                        for _p in range_(cnt):
                            for kc in range_(a0):
                                is_sub = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
                                add_bit = arith.select(is_sub, i32_1, i32_0)
                                flag = arith.ori(add_bit, arith.shli(op, i32_1))
                                is_last = arith.cmpi(arith.CmpIPredicate.eq, i32(kc), last_k)
                                xp = x_fifo.acquire(1)
                                for pt in range_(a1):
                                    wp = w_fifo.acquire(1)
                                    q4_gemm_fn(xp, wp, i32(pt), flag)
                                    w_fifo.release(1)
                                x_fifo.release(1)
                                with if_(is_last, hasElse=False):
                                    for pt in range_(a1):
                                        yp = y_fifo.acquire(1)
                                        q4_y_fn(yp, i32_1, i32(pt))
                                        y_fifo.release(1)

    workers = [
        Worker(
            core_body,
            fn_args=[
                W_L2L1[c].cons(),
                X_L2L1[c][r].cons(),
                Y_L1L2[c][r].prod(),
                sched_bufs[c][r],
                q4_gemm_fn,
                q4_w_fn,
                q4_x_fn,
                q4_y_fn,
            ],
            tile=Tile(c, 2 + r),
            stack_size=2048,
            while_true=True,
        )
        for c in range(N_COLS)
        for r in range(N_ROWS)
    ]

    w_prods = [W_L3L2[c].prod() for c in range(N_COLS)]
    x_prods = [X_L3L2[c].prod() for c in range(N_COLS)]
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, *fifos):
        wp_h = fifos[0:N_COLS]
        xp_h = fifos[N_COLS : 2 * N_COLS]
        yc_h = fifos[2 * N_COLS : 3 * N_COLS]

        def fill_w_1d(src_buf, tdims, off_per_col, tg, wait=False):
            for c in range(N_COLS):
                wp_h[c].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off_per_col[c], sizes=[1, W_ELEMS], strides=[0, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_w_broadcast(src_buf, tdims, off, tg, wait=False):
            fill_w_1d(src_buf, tdims, [off] * N_COLS, tg, wait=wait)

        def fill_x_2d(src_buf, tdims, off, row_stride, tg, wait=False):
            for c in range(N_COLS):
                xp_h[c].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off, sizes=[N_ROWS, 8, 640], strides=[row_stride, 640, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_x_xq(tg, kc=0, wait=False):
            xq_off = SLOT_XQ + kc * (N_ROWS * X_ELEMS)
            for c in range(N_COLS):
                xp_h[c].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL], offset=xq_off, sizes=[1, N_ROWS * X_ELEMS], strides=[0, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def drain_y_contig_rows(dst_buf, dst_tdims, dst_off, tg):
            for tl in range(4):
                for c in range(N_COLS):
                    yc_h[c].drain(
                        dst_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=dst_tdims,
                            offset=dst_off + tl * H + c * 256,
                            sizes=[2, N_ROWS, 128],
                            strides=[128, H * B_CORE, 1],
                        ),
                        wait=(tl == 3),
                        group=tg,
                    )
            tg.finish()
            tg_hi = TaskGroup()
            for tl in range(4, 8):
                for c in range(N_COLS):
                    yc_h[c].drain(
                        dst_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=dst_tdims,
                            offset=dst_off + tl * H + c * 256,
                            sizes=[2, N_ROWS, 128],
                            strides=[128, H * B_CORE, 1],
                        ),
                        wait=(tl == 7),
                        group=tg_hi,
                    )
            tg_hi.finish()

        def run_rms_to_xq(norm_off, x_slot, x_stride):
            tg = TaskGroup()
            if norm_off is not None:
                fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg, wait=True)
            for p in range(3):
                fill_x_2d(y_buf, [ACT_TOTAL], x_slot + p * (2 * H), x_stride, tg, wait=(p == 2))
            tg.finish()
            tg2 = TaskGroup()
            fill_x_2d(y_buf, [ACT_TOTAL], x_slot + 3 * (2 * H), x_stride, tg2)
            for part in range(4):
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_XQ + c * 512 + part * 128,
                            sizes=[N_ROWS, Y_CORE_ELEMS],
                            strides=[X_ELEMS, 1],
                        ),
                        wait=(part == 3),
                        group=tg2,
                    )
            tg2.finish()
            tg3 = TaskGroup()
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_XQ + 4096 + c * 128,
                        sizes=[N_ROWS, Y_CORE_ELEMS],
                        strides=[X_ELEMS, 1],
                    ),
                    wait=True,
                    group=tg3,
                )
            tg3.finish()

        def run_post_norm_and_res(norm_off, res_slot, dest_buf, dest_tdims, dest_slot):
            tg1 = TaskGroup()
            fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg1, wait=True)
            for p in range(3):
                fill_x_2d(y_buf, [ACT_TOTAL], res_slot + p * (2 * H), B_CORE * H, tg1, wait=(p == 2))
            tg1.finish()
            tg2 = TaskGroup()
            fill_x_2d(y_buf, [ACT_TOTAL], res_slot + 3 * (2 * H), B_CORE * H, tg2)
            drain_y_contig_rows(y_buf, [ACT_TOTAL], SLOT_NORMED_B, tg2)

            for p in range(3):
                tgp = TaskGroup()
                fill_x_2d(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + p * (2 * H), B_CORE * H, tgp)
                fill_x_2d(y_buf, [ACT_TOTAL], SLOT_NORMED_B + p * (2 * H), B_CORE * H, tgp, wait=True)
                tgp.finish()
            tga = TaskGroup()
            fill_x_2d(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + 3 * (2 * H), B_CORE * H, tga, wait=True)
            tga.finish()
            tgb = TaskGroup()
            fill_x_2d(y_buf, [ACT_TOTAL], SLOT_NORMED_B + 3 * (2 * H), B_CORE * H, tgb)
            drain_y_contig_rows(dest_buf, dest_tdims, dest_slot, tgb)

        def run_gemm_2d(w_off, n_passes, pass_tiles, k_chunks, out_slot, N_total):
            tpc = n_passes * pass_tiles
            col_ch = N_total // N_COLS
            for p in range(n_passes):
                if k_chunks == 1:
                    tg = TaskGroup()
                    fill_x_xq(tg, 0, wait=False)
                    for c in range(N_COLS):
                        tile0 = c * tpc + p * pass_tiles
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=w_off + tile0 * W_ELEMS,
                                sizes=[1, pass_tiles * W_ELEMS],
                                strides=[0, 1],
                            ),
                            wait=False,
                            group=tg,
                        )
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=out_slot + c * col_ch + p * pass_tiles * ROWS,
                                sizes=[pass_tiles, B, ROWS],
                                strides=[ROWS, N_total, 1],
                            ),
                            wait=True,
                            group=tg,
                        )
                    tg.finish()
                else:
                    for kc0 in range(0, k_chunks, 3):
                        kc1 = min(kc0 + 3, k_chunks)
                        is_last_blk = kc1 == k_chunks
                        tg = TaskGroup()
                        for kc in range(kc0, kc1):
                            is_last_kc = kc == (kc1 - 1)
                            fill_x_xq(tg, kc, wait=is_last_kc)
                            for c in range(N_COLS):
                                tile0 = c * tpc + p * pass_tiles
                                wp_h[c].fill(
                                    w_buf,
                                    tap=TensorAccessPattern(
                                        tensor_dims=[W_TOTAL],
                                        offset=w_off + (tile0 * k_chunks + kc) * W_ELEMS,
                                        sizes=[pass_tiles, 18, 256],
                                        strides=[k_chunks * W_ELEMS, 256, 1],
                                    ),
                                    wait=is_last_kc,
                                    group=tg,
                                )
                        if is_last_blk:
                            for c in range(N_COLS):
                                yc_h[c].drain(
                                    y_buf,
                                    tap=TensorAccessPattern(
                                        tensor_dims=[ACT_TOTAL],
                                        offset=out_slot + c * col_ch + p * pass_tiles * ROWS,
                                        sizes=[pass_tiles, B, ROWS],
                                        strides=[ROWS, N_total, 1],
                                    ),
                                    wait=True,
                                    group=tg,
                                )
                        tg.finish()

        # 1. Attn RMSNorm
        run_rms_to_xq(SLOT_NORM_0, SLOT_HIDDEN_IN, B_CORE * H)
        # 2. B, C, X IN_PROJ
        for mat_off, out_slot in [
            (W_CONV_IN_B, SLOT_CONV_B),
            (W_CONV_IN_C, SLOT_CONV_C),
            (W_CONV_IN_X, SLOT_CONV_X),
        ]:
            tg = TaskGroup()
            for p in range(2):
                fill_x_xq(tg, 0)
                for c in range(N_COLS):
                    wp_h[c].fill(
                        w_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[W_TOTAL],
                            offset=mat_off + (c * 16 + p * 8) * W_ELEMS,
                            sizes=[1, 8 * W_ELEMS],
                            strides=[0, 1],
                        ),
                        group=tg,
                    )
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=out_slot + c * 8192 + p * 128,
                            sizes=[8, B, ROWS],
                            strides=[ROWS, 256, 1],
                        ),
                        wait=(p == 1),
                        group=tg,
                    )
            tg.finish()

        # 3. 7-tap ShortConv (split >3 W fills per channel across TaskGroups)
        tg_sc0 = TaskGroup()
        fill_w_1d(w_buf, [W_TOTAL], [W_CONV_WT + c * W_ELEMS for c in range(N_COLS)], tg_sc0)
        fill_w_1d(kv_buf, [KV_TOTAL], [c * W_ELEMS for c in range(N_COLS)], tg_sc0)
        fill_w_1d(y_buf, [ACT_TOTAL], [SLOT_CONV_B + c * 8192 for c in range(N_COLS)], tg_sc0, wait=True)
        tg_sc0.finish()

        tg_sc1 = TaskGroup()
        fill_w_1d(y_buf, [ACT_TOTAL], [SLOT_CONV_B + c * 8192 + 4096 for c in range(N_COLS)], tg_sc1)
        for half in range(2):
            fill_w_1d(y_buf, [ACT_TOTAL], [SLOT_CONV_X + c * 8192 + half * 4096 for c in range(N_COLS)], tg_sc1, wait=(half == 1))
        tg_sc1.finish()

        tg = TaskGroup()
        for half in range(2):
            fill_w_1d(y_buf, [ACT_TOTAL], [SLOT_CONV_C + c * 8192 + half * 4096 for c in range(N_COLS)], tg)
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_ATTN_OUT + half * 16 * H + c * 256,
                        sizes=[8, N_ROWS, 2, 64],
                        strides=[2 * H, 64, H, 1],
                    ),
                    group=tg,
                )
        for c in range(N_COLS):
            yc_h[c].drain(
                kvo_buf,
                tap=TensorAccessPattern(
                    tensor_dims=[KV_OUT_TOTAL],
                    offset=c * W_ELEMS,
                    sizes=[3, N_ROWS, 2, 64],
                    strides=[512, 64, 256, 1],
                ),
                wait=True,
                group=tg,
            )
        tg.finish()

        # 4. Q8(SLOT_ATTN_OUT) + OUT_PROJ + post_attn_norm + Res1
        run_rms_to_xq(None, SLOT_ATTN_OUT, B_CORE * H)
        run_gemm_2d(W_CONV_OUT, 2, 8, 1, SLOT_RES_TMP, H)
        run_post_norm_and_res(SLOT_NORM_0, SLOT_RES_TMP, y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN)
        # 5. FFN
        run_rms_to_xq(SLOT_NORM_1, SLOT_HIDDEN_IN, B_CORE * H)
        for base_off, out_slot in [(W_CONV_FFN_GATE, SLOT_GATE), (W_CONV_FFN_UP, SLOT_UP)]:
            for p0 in range(0, 15, 3):
                tg = TaskGroup()
                for dp in range(3):
                    p = p0 + dp
                    fill_x_xq(tg, 0)
                    for c in range(N_COLS):
                        tile0 = p * 32 + c * 4
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=base_off + tile0 * W_ELEMS,
                                sizes=[1, 4 * W_ELEMS],
                                strides=[0, 1],
                            ),
                            group=tg,
                        )
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=out_slot + p * 16384 + c * 64,
                                sizes=[4, B, ROWS],
                                strides=[ROWS, 512, 1],
                            ),
                            wait=(dp == 2),
                            group=tg,
                        )
                tg.finish()

        for kc in range(4):
            xq_base = SLOT_XQ + kc * (N_ROWS * X_ELEMS)
            for j in range(4):
                tg = TaskGroup()
                p = kc * 4 + j
                if p < 15:
                    fill_x_2d(y_buf, [ACT_TOTAL], SLOT_GATE + p * 16384, 4096, tg)
                    fill_x_2d(y_buf, [ACT_TOTAL], SLOT_UP + p * 16384, 4096, tg)
                else:
                    fill_x_2d(y_buf, [ACT_TOTAL], SLOT_UP, 4096, tg)
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=xq_base + j * 1024 + c * 128,
                            sizes=[N_ROWS, Y_CORE_ELEMS],
                            strides=[X_ELEMS, 1],
                        ),
                        wait=(j < 3),
                        group=tg,
                    )
                if j == 3:
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=xq_base + 4096 + c * 128,
                                sizes=[N_ROWS, Y_CORE_ELEMS],
                                strides=[X_ELEMS, 1],
                            ),
                            wait=True,
                            group=tg,
                        )
                tg.finish()

        run_gemm_2d(W_CONV_FFN_DOWN, 2, 8, 4, SLOT_RES_TMP, H)
        run_post_norm_and_res(SLOT_NORM_1, SLOT_RES_TMP, yn_buf, [65536], 0)

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *x_prods, *y_cons])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    opts = parser.parse_args()
    run_design_cli(q4prefill_lfm3_conv, opts, device=NPU2(), compile_kwargs=lambda _o: {})


if __name__ == "__main__":
    main()

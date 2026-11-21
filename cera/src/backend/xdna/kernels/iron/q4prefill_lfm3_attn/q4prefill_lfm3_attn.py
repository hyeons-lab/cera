# Fused B=32 Q4_0 LFM3 Attention prefill layer (32 cores: 4 rows x 8 cols, npu2).
# Modes 1, 2, 3 via g_params[6]:
#   1 = Causal / SWA Attention
#   2 = Bidir Global Phase A (QKV + Gate + RoPE -> SLOT_Q_BF16 + kv_out + SLOT_ATTN_GATE)
#   3 = Bidir Global Phase B (Reload SLOT_Q_BF16 -> FA -> Load Gate -> O -> FFN tail)

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import arith
from aie.helpers.dialects.scf import else_, if_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, Out
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
Q_DIM = 2048
KV_DIM = 512
N_KV_HEADS = 4
MAX_KV_CHUNKS = 256

W_ATTN_Q = 0
W_ATTN_K = 128 * W_ELEMS
W_ATTN_V = 160 * W_ELEMS
W_ATTN_GATE = 192 * W_ELEMS
W_ATTN_O = 193 * W_ELEMS
W_ATTN_FFN_GATE = 321 * W_ELEMS
W_ATTN_FFN_UP = 801 * W_ELEMS
W_ATTN_FFN_DOWN = 1281 * W_ELEMS
W_TOTAL = 1992 * W_ELEMS

SLOT_HIDDEN_IN = 0
SLOT_Q = 65536
SLOT_K = 131072
SLOT_V = 147456
SLOT_ATTN_OUT = 163840
SLOT_RES_TMP = 229376
SLOT_NORMED_B = 294912
SLOT_GATE = 360448
SLOT_FA_STATE = SLOT_GATE
SLOT_UP = 607232
SLOT_XQ = 854016
SLOT_Q_BF16 = 935936
SLOT_ATTN_GATE = 972800
SLOT_NORM_0 = 977408
SLOT_NORM_1 = 982016
SLOT_ROPE_PKT = 986624
ACT_TOTAL = 986624 + 4608 + 1024

KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * W_ELEMS
KV_OUT_TOTAL = 8 * W_ELEMS


@iron.jit(use_cache=False)
def q4prefill_lfm3_attn(
    w: In,
    y: Out,
    y_next: Out,
    kv: In,
    kv_out: Out,
    layer_mode: CompileTime[int] = 1,
    kv_chunks: CompileTime[int] = 0,
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
    oz_flags = common_flags + ["-Oz", "-fno-unroll-loops"]
    fa_flags = oz_flags + ["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]
    os_flags = common_flags + ["-Os"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm3_attn_gemm_op",
        source_file=str(here / "q4_lfm3_gemm_op.cc"),
        arg_types=[x_pkt_ty, w_pkt_ty, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm3_attn_w_op",
        source_file=str(here / "q4_lfm3_attn_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=fa_flags,
    )
    q4_x_fn = ExternalFunction(
        "q4_lfm3_attn_x_op",
        source_file=str(here / "q4_lfm3_attn_x_op.cc"),
        arg_types=[x_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm3_attn_y_op",
        source_file=str(here / "q4prefill_lfm3_attn_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=oz_flags,
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
    KIND_FA = 4

    M1 = 2   # 1 << 1
    M2 = 4   # 1 << 2
    M3 = 8   # 1 << 3
    M4 = 16  # 1 << 4
    M5 = 32  # 1 << 5
    M12 = M1 | M2
    M13 = M1 | M3
    M345 = M3 | M4 | M5
    M45 = M4 | M5
    M135 = M1 | M3 | M5

    def build_worker_sched(c, r):
        words = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0, mask=14):
            w0 = (kind & 0xFF) | ((cnt & 0xFF) << 8) | ((op & 0xFF) << 16) | ((a0 & 0xFF) << 24)
            w1 = (a1 & 0xFF) | ((a2 & 0xFF) << 8) | ((mask & 0xFFFF) << 16)
            words.extend((w0, w1))

        def emit_rms(norm_op_kind, norm_slot, skip_norm, f32_out, mask):
            if norm_op_kind >= 0:
                emit(KIND_W, 1, norm_op_kind, c, norm_slot, 0, mask)
            emit(KIND_X, 4, 0, c, 0, skip_norm | (f32_out << 1), mask)
            if f32_out == 0:
                emit(KIND_Y, 5, 0, 0, 0, 0, mask)
            else:
                emit(KIND_Y, 16, 6, 0, 0, 0, mask)

        def emit_res(mask):
            for p in range(4):
                emit(KIND_X, 1, 2, c, 0, p, mask)
                emit(KIND_X, 1, 2, c, 1, p, mask)
            emit(KIND_Y, 16, 4, 0, 0, 0, mask)

        # Stage A (Modes 1 & 2): Attn RMSNorm + QKV + Gate + Q/K Norm + NeoX RoPE + KV drain
        emit_rms(-1, 0, 0, 0, M12)
        emit(KIND_GEMM, 2, 0, 1, 8, 1, M12)               # Q
        emit(KIND_GEMM, 1, 0, 1, 4, 1, M12)               # K
        emit(KIND_GEMM, 1, 0, 1, 4, 1, M12)               # V
        emit(KIND_GEMM, 1, 0, 1, 1, 1 if c == 0 else 0, M12)  # ATTN_GATE
        emit(KIND_W, 1, 1, r, 0, 0, M12)                  # ROPE_PKT
        emit(KIND_X, 4, 1, c * 4 + r, 0, 0, M12)          # Q norm + RoPE
        emit(KIND_Y, 8, 7, 0, 0, 0, M2)                   # Mode 2 only: drain SLOT_Q_BF16
        emit(KIND_X, 4, 1, c * 4 + r, 0, 1, M12)          # K norm + RoPE
        emit(KIND_X, 4, 1, c * 4 + r, 0, 2, M12)          # V convert
        emit(KIND_Y, 8, 2, 0, 0, 0, M12 if (c % 2) == 0 else 0)  # Drain KV -> kv_out

        # Stage B (Modes 3, 4, 5): Reload SLOT_Q_BF16 (+ SLOT_FA_STATE for Modes 4 & 5)
        emit(KIND_X, 1, 1, c * 4 + r, 0, 3, M345)
        emit(KIND_X, 1, 1, c * 4 + r, 0, 4, M45)

        # Stage FA (Modes 1, 3, 4, 5)
        emit(KIND_FA, 1, 16 + r, 0, 0, 0, M135 | M4)

        # Mode 4 only: Drain updated o_acc + row_m/row_l -> SLOT_FA_STATE
        emit(KIND_Y, 17, 8, 0, 0, 0, M4)

        # Stage C (Modes 1, 3, 5): Gate + O_PROJ + FFN + Sandwich tail
        emit(KIND_W, 1, 2, c, r, 0, M135)                 # SLOT_ATTN_GATE
        emit(KIND_Y, 16, 3, 0, 0, 0, M135)                # Drain SLOT_ATTN_OUT
        emit_rms(-1, 0, 1, 0, M135)                       # Q8(SLOT_ATTN_OUT)
        emit(KIND_GEMM, 2, 0, 1, 8, 1, M135)              # O_PROJ
        emit_rms(4, 1, 0, 1, M135)                        # post_attn_norm
        emit_res(M135)
        emit_rms(4, 0, 0, 0, M135)                        # ffn_norm
        emit(KIND_GEMM, 15, 0, 1, 4, 1, M135)             # FFN_GATE
        emit(KIND_GEMM, 15, 0, 1, 4, 1, M135)             # FFN_UP
        for kc in range(4):
            for j in range(4):
                if kc == 3 and j == 3:
                    emit(KIND_X, 1, 3, c, 3, 2, M135)
                else:
                    emit(KIND_X, 1, 3, c, j, 0, M135)
                    emit(KIND_X, 1, 3, c, j, 1, M135)
                emit(KIND_Y, 1, 5, 0, 0, 0, M135)
            emit(KIND_Y, 1, 5, 1, 0, 0, M135)
        emit(KIND_GEMM, 2, 1, 4, 8, 1, M135)              # FFN_DOWN
        emit_rms(4, 1, 0, 1, M135)                        # post_ffn_norm
        emit_res(M135)
        return np.array(words, dtype=np.int32)

    sample_sched = build_worker_sched(0, 0)
    n_steps = len(sample_sched) // 2
    sched_ty = np.ndarray[(len(sample_sched),), np.dtype[np.int32]]
    sched_bufs = [
        [Buffer(sched_ty, initial_value=build_worker_sched(c, r), name=f"sc_{c}_{r}") for r in range(N_ROWS)]
        for c in range(N_COLS)
    ]

    def make_worker_body(c):
        def core_body(w_fifo, x_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_w_fn, q4_x_fn, q4_y_fn):
            idx_ty = IndexType.get()
            i32_ty = IntegerType.get_signless(32)
            idx0 = arith.constant(idx_ty, 0)
            idx1 = arith.constant(idx_ty, 1)
            idx_kvc = arith.constant(idx_ty, 4098)
            idx_lmode = arith.constant(idx_ty, 4102)
            i32_0 = arith.constant(i32_ty, 0)
            i32_1 = arith.constant(i32_ty, 1)
            i32_2 = arith.constant(i32_ty, 2)
            i32_3 = arith.constant(i32_ty, 3)
            i32_8 = arith.constant(i32_ty, 8)
            i32_16 = arith.constant(i32_ty, 16)
            i32_24 = arith.constant(i32_ty, 24)
            i32_34 = arith.constant(i32_ty, 34)
            i32_255 = arith.constant(i32_ty, 255)

            def i32(v):
                return arith.index_cast(i32_ty, v)

            # Peeled step 0: consume SLOT_NORM_0 + metadata header
            wp0 = w_fifo.acquire(1)
            kv_ch = arith.bitcast(i32_ty, wp0[idx_kvc])
            lmode = arith.bitcast(i32_ty, wp0[idx_lmode])
            q4_w_fn(wp0, i32_0, arith.constant(i32_ty, c), i32_0)
            w_fifo.release(1)

            lmode_bit = arith.shli(i32_1, lmode)
            has_kv = arith.cmpi(arith.CmpIPredicate.sgt, kv_ch, i32_0)
            use_cur = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(lmode_bit, i32_34), i32_0)

            for s in range_(n_steps):
                base = arith.shli(s, idx1)
                w1 = sched_buf[arith.addi(base, idx1)]
                mask = arith.shrui(w1, i32_16)
                active = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(mask, lmode_bit), i32_0)
                with if_(active, hasElse=False):
                    w0 = sched_buf[base]
                    kind = arith.andi(w0, i32_255)
                    cnt = arith.andi(arith.shrui(w0, i32_8), i32_255)
                    op = arith.andi(arith.shrui(w0, i32_16), i32_255)
                    a0 = arith.shrui(w0, i32_24)
                    a1 = arith.andi(w1, i32_255)
                    a2 = arith.andi(arith.shrui(w1, i32_8), i32_255)

                    is_w = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_0)
                    with if_(is_w, hasElse=True) as if_w:
                        for i in range_(cnt):
                            wp = w_fifo.acquire(1)
                            q4_w_fn(wp, op, arith.addi(a0, i32(i)), a1)
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
                                is_g = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_3)
                                with if_(is_g, hasElse=True) as if_g:
                                    last_k = arith.subi(a0, i32_1)
                                    do_drain = arith.cmpi(arith.CmpIPredicate.ne, a2, i32_0)
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
                                            with if_(arith.andi(is_last, do_drain), hasElse=False):
                                                for pt in range_(a1):
                                                    yp = y_fifo.acquire(1)
                                                    q4_y_fn(yp, i32_1, i32(pt))
                                                    y_fifo.release(1)
                                with else_(if_g):
                                    # KIND_FA: op = 16 + row
                                    for kc in range_(kv_ch):
                                        is_f = arith.cmpi(arith.CmpIPredicate.eq, kc, idx0)
                                        wp = w_fifo.acquire(1)
                                        q4_w_fn(wp, op, i32(kc), arith.select(is_f, i32_2, i32_0))
                                        w_fifo.release(1)
                                    with if_(use_cur, hasElse=False):
                                        wp_c = w_fifo.acquire(1)
                                        q4_w_fn(wp_c, op, i32_0, arith.select(has_kv, i32_1, i32_3))
                                        w_fifo.release(1)

        return core_body

    workers = [
        Worker(
            make_worker_body(c),
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
            stack_size=3072,
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

        def run_qkv_gate_and_rope(drain_q_bf16=False):
            for mat_off, npa, ptl, out_sl, n_tot in [
                (W_ATTN_Q, 2, 8, SLOT_Q, Q_DIM),
                (W_ATTN_K, 1, 4, SLOT_K, KV_DIM),
                (W_ATTN_V, 1, 4, SLOT_V, KV_DIM),
            ]:
                tg = TaskGroup()
                tpc = npa * ptl
                col_ch = n_tot // N_COLS
                for p in range(npa):
                    fill_x_xq(tg, 0)
                    for c in range(N_COLS):
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=mat_off + (c * tpc + p * ptl) * W_ELEMS,
                                sizes=[1, ptl * W_ELEMS],
                                strides=[0, 1],
                            ),
                            group=tg,
                        )
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=out_sl + c * col_ch + p * ptl * ROWS,
                                sizes=[ptl, B, ROWS],
                                strides=[ROWS, n_tot, 1],
                            ),
                            wait=(p == npa - 1),
                            group=tg,
                        )
                tg.finish()

            tg_g = TaskGroup()
            fill_x_xq(tg_g, 0, wait=True)
            fill_w_broadcast(w_buf, [W_TOTAL], W_ATTN_GATE, tg_g)
            yc_h[0].drain(
                y_buf,
                tap=TensorAccessPattern(
                    tensor_dims=[ACT_TOTAL], offset=SLOT_ATTN_GATE, sizes=[B, 16], strides=[16, 1]
                ),
                wait=True,
                group=tg_g,
            )
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT, tg_g, wait=True)
            tg_g.finish()

            tg_q0 = TaskGroup()
            for p in range(3):
                fill_x_2d(y_buf, [ACT_TOTAL], SLOT_Q + p * (2 * Q_DIM), B_CORE * Q_DIM, tg_q0, wait=(p == 2))
            tg_q0.finish()

            tg_q1 = TaskGroup()
            fill_x_2d(
                y_buf,
                [ACT_TOTAL],
                SLOT_Q + 3 * (2 * Q_DIM),
                B_CORE * Q_DIM,
                tg_q1,
                wait=(not drain_q_bf16),
            )
            if drain_q_bf16:
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_Q_BF16 + c * 4096,
                            sizes=[8, N_ROWS, 128],
                            strides=[128, 1024, 1],
                        ),
                        wait=True,
                        group=tg_q1,
                    )
            tg_q1.finish()

            tg_k0 = TaskGroup()
            for p in range(3):
                fill_x_2d(y_buf, [ACT_TOTAL], SLOT_K + p * (2 * KV_DIM), B_CORE * KV_DIM, tg_k0, wait=(p == 2))
            tg_k0.finish()
            tg_k1 = TaskGroup()
            fill_x_2d(y_buf, [ACT_TOTAL], SLOT_K + 3 * (2 * KV_DIM), B_CORE * KV_DIM, tg_k1, wait=True)
            tg_k1.finish()

            tg_v0 = TaskGroup()
            for p in range(3):
                fill_x_2d(y_buf, [ACT_TOTAL], SLOT_V + p * (2 * KV_DIM), B_CORE * KV_DIM, tg_v0, wait=(p == 2))
            tg_v0.finish()

            tg2 = TaskGroup()
            fill_x_2d(y_buf, [ACT_TOTAL], SLOT_V + 3 * (2 * KV_DIM), B_CORE * KV_DIM, tg2)
            for kv_h in range(N_KV_HEADS):
                c = kv_h * 2
                yc_h[c].drain(
                    kvo_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[KV_OUT_TOTAL],
                        offset=kv_h * W_ELEMS,
                        sizes=[4, N_ROWS, 128],
                        strides=[128, 512, 1],
                    ),
                    group=tg2,
                )
                yc_h[c].drain(
                    kvo_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[KV_OUT_TOTAL],
                        offset=kv_h * W_ELEMS + 2304,
                        sizes=[4, N_ROWS, 128],
                        strides=[128, 512, 1],
                    ),
                    wait=True,
                    group=tg2,
                )
            tg2.finish()

        def run_fa_o_and_tail(include_cur_kv, tg=None):
            if tg is None:
                tg = TaskGroup()
            w_tasks = []
            for chk0 in range(0, kv_chunks, 64):
                cnt = min(64, kv_chunks - chk0)
                w_tasks.append((kv_buf, [KV_TOTAL], lambda c, s=chk0: ((c // 2) * MAX_KV_CHUNKS + s) * W_ELEMS, cnt * W_ELEMS))
            if include_cur_kv:
                w_tasks.append((kvo_buf, [KV_OUT_TOTAL], lambda c: (c // 2) * W_ELEMS, W_ELEMS))
            w_tasks.append((y_buf, [ACT_TOTAL], lambda _c: SLOT_ATTN_GATE, W_ELEMS))

            # Emit <= 3 fills per channel per TaskGroup before draining.
            for b0 in range(0, len(w_tasks), 3):
                b1 = min(b0 + 3, len(w_tasks))
                is_last_b = b1 == len(w_tasks)
                cur_tg = tg if b0 == 0 else TaskGroup()
                for wi in range(b0, b1):
                    buf, tdims, off_fn, slen = w_tasks[wi]
                    do_wait = (not is_last_b) and (wi == b1 - 1)
                    for c in range(N_COLS):
                        wp_h[c].fill(
                            buf,
                            tap=TensorAccessPattern(
                                tensor_dims=tdims,
                                offset=off_fn(c),
                                sizes=[1, slen],
                                strides=[0, 1],
                            ),
                            wait=do_wait,
                            group=cur_tg,
                        )
                if is_last_b:
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_ATTN_OUT + c * 256,
                                sizes=[16, B, ROWS],
                                strides=[ROWS, H, 1],
                            ),
                            wait=True,
                            group=cur_tg,
                        )
                cur_tg.finish()

            run_rms_to_xq(None, SLOT_ATTN_OUT, B_CORE * H)
            run_gemm_2d(W_ATTN_O, 2, 8, 1, SLOT_RES_TMP, H)
            run_post_norm_and_res(SLOT_NORM_0, SLOT_RES_TMP, y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN)
            run_rms_to_xq(SLOT_NORM_1, SLOT_HIDDEN_IN, B_CORE * H)

            for base_off, out_slot in [(W_ATTN_FFN_GATE, SLOT_GATE), (W_ATTN_FFN_UP, SLOT_UP)]:
                for p0 in range(0, 15, 3):
                    tgp = TaskGroup()
                    for dp in range(3):
                        p = p0 + dp
                        fill_x_xq(tgp, 0)
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
                                group=tgp,
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
                                group=tgp,
                            )
                    tgp.finish()

            for kc in range(4):
                xq_base = SLOT_XQ + kc * (N_ROWS * X_ELEMS)
                for j in range(4):
                    tgq = TaskGroup()
                    p = kc * 4 + j
                    if p < 15:
                        fill_x_2d(y_buf, [ACT_TOTAL], SLOT_GATE + p * 16384, 4096, tgq)
                        fill_x_2d(y_buf, [ACT_TOTAL], SLOT_UP + p * 16384, 4096, tgq)
                    else:
                        fill_x_2d(y_buf, [ACT_TOTAL], SLOT_UP, 4096, tgq)
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
                            group=tgq,
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
                                group=tgq,
                            )
                    tgq.finish()

            run_gemm_2d(W_ATTN_FFN_DOWN, 2, 8, 4, SLOT_RES_TMP, H)
            run_post_norm_and_res(SLOT_NORM_1, SLOT_RES_TMP, yn_buf, [65536], 0)

        def load_q_bf16_and_state(load_state):
            tg = TaskGroup()
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_NORM_0, tg, wait=True)
            for r in range(N_ROWS):
                for c in range(N_COLS):
                    xp_h[c].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_Q_BF16 + c * 4096 + r * 1024,
                            sizes=[1, X_ELEMS],
                            strides=[0, 1],
                        ),
                        wait=(r == 1 or r == 3),
                        group=tg,
                    )
            tg.finish()
            if load_state:
                tg_fs = TaskGroup()
                for c in range(N_COLS):
                    xp_h[c].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_FA_STATE + c * 20480,
                            sizes=[1, N_ROWS * X_ELEMS],
                            strides=[0, 1],
                        ),
                        wait=True,
                        group=tg_fs,
                    )
                tg_fs.finish()

        if layer_mode == 1:
            run_rms_to_xq(SLOT_NORM_0, SLOT_HIDDEN_IN, B_CORE * H)
            run_qkv_gate_and_rope(drain_q_bf16=False)
            run_fa_o_and_tail(include_cur_kv=True)
        elif layer_mode == 2:
            run_rms_to_xq(SLOT_NORM_0, SLOT_HIDDEN_IN, B_CORE * H)
            run_qkv_gate_and_rope(drain_q_bf16=True)
        elif layer_mode == 3:
            load_q_bf16_and_state(load_state=False)
            run_fa_o_and_tail(include_cur_kv=False)
        elif layer_mode == 4:
            load_q_bf16_and_state(load_state=True)
            w_tasks = []
            for chk0 in range(0, kv_chunks, 64):
                cnt = min(64, kv_chunks - chk0)
                w_tasks.append((kv_buf, [KV_TOTAL], lambda c, s=chk0: ((c // 2) * MAX_KV_CHUNKS + s) * W_ELEMS, cnt * W_ELEMS))
            for b0 in range(0, len(w_tasks), 3):
                b1 = min(b0 + 3, len(w_tasks))
                is_last_b = b1 == len(w_tasks)
                cur_tg = TaskGroup()
                for wi in range(b0, b1):
                    buf, tdims, off_fn, slen = w_tasks[wi]
                    do_wait = (not is_last_b) and (wi == b1 - 1)
                    for c in range(N_COLS):
                        wp_h[c].fill(
                            buf,
                            tap=TensorAccessPattern(
                                tensor_dims=tdims,
                                offset=off_fn(c),
                                sizes=[1, slen],
                                strides=[0, 1],
                            ),
                            wait=do_wait,
                            group=cur_tg,
                        )
                if is_last_b:
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_FA_STATE + c * 20480,
                                sizes=[17, N_ROWS, 128],
                                strides=[128, 5120, 1],
                            ),
                            wait=True,
                            group=cur_tg,
                        )
                cur_tg.finish()
        else:
            load_q_bf16_and_state(load_state=True)
            run_fa_o_and_tail(include_cur_kv=True)

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *x_prods, *y_cons])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=1)
    parser.add_argument("--kv-chunks", type=int, default=0)
    opts = parser.parse_args()
    run_design_cli(
        q4prefill_lfm3_attn,
        opts,
        device=NPU2(),
        compile_kwargs=lambda o: {
            "layer_mode": o.layer_mode,
            "kv_chunks": o.kv_chunks,
        },
    )


if __name__ == "__main__":
    main()

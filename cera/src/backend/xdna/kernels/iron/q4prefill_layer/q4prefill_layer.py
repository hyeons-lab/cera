# Fused B=32 Q4 prefill transformer layer (32 cores: 4 rows x 8 cols, npu2).
# Executes an entire transformer layer in one NPU run:
#   RMSNorm+Q8 -> Q/K/V GEMMs -> RoPE+bf16 KV -> Tiled FlashAttention (bo_kv + cur_kv)
#   -> Quantize+O GEMM+ResAdd -> RMSNorm+Q8 -> Gate/Up GEMMs -> SiLU*Up+Q8
#   -> Down GEMM+ResAdd -> bo_y_next.

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
B_CORE = 8
B = N_ROWS * B_CORE  # 32
ROWS = 16
CHUNK_K = 2048

W_ELEMS = 4608  # 18,432 B
X_ELEMS = 5120  # 20,480 B (16,384 xq + 4,096 xd)
Y_CORE_ELEMS = 128  # 512 B (8 tokens x 16 dims)
Y_COL_ELEMS = N_ROWS * Y_CORE_ELEMS  # 512

# Llama-3.2-1B layer dimensions
H = 2048
Q_DIM = 2048
KV_DIM = 512
FFN = 8192
MAX_KV_CHUNKS = 64  # 64 * 64 = 4,096 context tokens per layer in bo_kv

# bo_w offsets (f32 words)
def _w_elems(N, K):
    return (N // ROWS) * (K // CHUNK_K) * W_ELEMS

W_Q_OFF = 0
W_Q_LEN = _w_elems(Q_DIM, H)
W_K_OFF = W_Q_OFF + W_Q_LEN
W_K_LEN = _w_elems(KV_DIM, H)
W_V_OFF = W_K_OFF + W_K_LEN
W_V_LEN = _w_elems(KV_DIM, H)
W_O_OFF = W_V_OFF + W_V_LEN
W_O_LEN = _w_elems(H, Q_DIM)
W_GATE_OFF = W_O_OFF + W_O_LEN
W_GATE_LEN = _w_elems(FFN, H)
W_UP_OFF = W_GATE_OFF + W_GATE_LEN
W_UP_LEN = _w_elems(FFN, H)
W_DOWN_OFF = W_UP_OFF + W_UP_LEN
W_DOWN_LEN = _w_elems(H, FFN)
W_TOTAL = W_DOWN_OFF + W_DOWN_LEN

# bo_y / bo_y_next layout (f32 words)
SLOT_HIDDEN_IN = 0                        # 65,536 (32 x 2048)
SLOT_XQ = SLOT_HIDDEN_IN + B * H          # 81,920 (max 4 K-chunks x 4 rows x 5120)
SLOT_Q = SLOT_XQ + 4 * N_ROWS * X_ELEMS   # 73,728 (8 cols x 2 x 4608)
SLOT_KV_F32 = SLOT_Q + N_COLS * 2 * W_ELEMS    # 36,864 (8 cols x 4608)
SLOT_KV_CUR = SLOT_KV_F32 + N_COLS * W_ELEMS   # 36,864 (8 cols x 4608 bf16)
SLOT_ATTN_OUT = SLOT_KV_CUR + N_COLS * W_ELEMS # 65,536
SLOT_O_OUT = SLOT_ATTN_OUT + B * H             # 65,536
SLOT_HIDDEN_MID = SLOT_O_OUT + B * H           # 65,536
SLOT_GATE = SLOT_HIDDEN_MID + B * H            # 262,144
SLOT_UP = SLOT_GATE + B * FFN                  # 262,144
SLOT_DOWN_OUT = SLOT_UP + B * FFN              # 65,536
SLOT_META = SLOT_DOWN_OUT + B * H              # 12,288 (norm_w[4096], params[16], rope_cs[4608..])
ACT_TOTAL = SLOT_META + 12288

KV_TOTAL = N_COLS * MAX_KV_CHUNKS * W_ELEMS


@iron.jit
def q4prefill_layer(w: In, y: Out, y_next: Out, kv: In, kv_chunks: CompileTime[int] = 0):
    dev = get_current_device()
    if dev.cols < N_COLS:
        raise ValueError(f"need {N_COLS} columns, got {dev.cols}")

    w_slot_ty = np.ndarray[(W_ELEMS,), np.dtype[np.float32]]
    x_slot_ty = np.ndarray[(X_ELEMS,), np.dtype[np.float32]]
    y_core_ty = np.ndarray[(Y_CORE_ELEMS,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_ELEMS,), np.dtype[np.float32]]

    w_bo_ty = np.ndarray[(W_TOTAL,), np.dtype[np.float32]]
    act_bo_ty = np.ndarray[(ACT_TOTAL,), np.dtype[np.float32]]
    kv_bo_ty = np.ndarray[(KV_TOTAL,), np.dtype[np.float32]]

    here = Path(__file__).parent
    inc = [cxx_header_path(), str(here)]
    flags = ["-Oz"]

    gemm_op = ExternalFunction(
        "q4_layer_gemm_op",
        source_file=str(here / "q4_gemm_op.cc"),
        arg_types=[x_slot_ty, w_slot_ty, np.int32, np.int32],
        include_dirs=inc,
        compile_flags=["-O2"],
    )
    fa_op = ExternalFunction(
        "q4_layer_fa_op",
        source_file=str(here / "q4_fa_op.cc"),
        arg_types=[w_slot_ty, np.int32, np.int32, np.int32],
        include_dirs=inc,
        compile_flags=["-Os"],
    )
    w_op = ExternalFunction(
        "q4_layer_w_op",
        source_file=str(here / "q4prefill_layer_core.cc"),
        arg_types=[w_slot_ty, np.int32, np.int32, np.int32],
        include_dirs=inc,
        compile_flags=flags,
    )
    x_op = ExternalFunction(
        "q4_layer_x_op",
        source_file=str(here / "q4_x_op.cc"),
        arg_types=[x_slot_ty, np.int32, np.int32, np.int32, np.int32],
        include_dirs=inc,
        compile_flags=flags,
    )
    y_op = ExternalFunction(
        "q4_layer_y_op",
        source_file=str(here / "q4_y_op.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        include_dirs=inc,
        compile_flags=flags,
    )

    x_l3l2 = [ObjectFifo(x_slot_ty, name=f"X_L3L2_{r}", depth=1) for r in range(N_ROWS)]
    x_l2l1 = [
        x_l3l2[r].cons().forward(obj_type=x_slot_ty, name=f"X_L2L1_{r}", depth=1)
        for r in range(N_ROWS)
    ]

    w_l3l2 = [ObjectFifo(w_slot_ty, name=f"W_L3L2_{c}", depth=2) for c in range(N_COLS)]
    w_l2l1 = [
        w_l3l2[c].cons().forward(obj_type=w_slot_ty, name=f"W_L2L1_{c}", depth=1)
        for c in range(N_COLS)
    ]

    y_l2l3 = [ObjectFifo(y_col_ty, name=f"Y_L2L3_{c}", depth=2) for c in range(N_COLS)]
    y_l1l2_cols = [
        y_l2l3[c].prod().join(
            [r * Y_CORE_ELEMS for r in range(N_ROWS)],
            obj_types=[y_core_ty] * N_ROWS,
            names=[f"Y_L1L2_{c}_{r}" for r in range(N_ROWS)],
            depths=[2] * N_ROWS,
        )
        for c in range(N_COLS)
    ]

    def i32(v):
        return arith.index_cast(IntegerType.get_signless(32), v)

    KIND_W = 0
    KIND_X = 1
    KIND_Y = 2
    KIND_GEMM = 3

    def build_worker_sched(c, r):
        rows = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0):
            rows.append((kind, cnt, op, a0, a1, a2))

        def emit_rms(load_norm, skip_norm):
            if load_norm > 0:
                emit(KIND_W, 1, 0, c, load_norm)
            emit(KIND_X, 4, 0, c, 0, skip_norm)
            emit(KIND_Y, 5, 0)

        def emit_gemm(N, K):
            tiles_per_col = (N // N_COLS) // ROWS
            k_chunks = K // CHUNK_K
            pass_tiles = 8 if (k_chunks > 1 and tiles_per_col > 8) else tiles_per_col
            n_passes = tiles_per_col // pass_tiles
            xd_mode = 1 if k_chunks > 1 else 0
            emit(KIND_GEMM, n_passes, xd_mode, k_chunks, pass_tiles)

        def emit_res():
            for _ in range(4):
                emit(KIND_X, 1, 2, c, 0, 0)
                emit(KIND_X, 1, 2, c, 1, 0)
                emit(KIND_Y, 4, 4)

        # 1. Attn RMSNorm + Q8 (metadata + attn norm weights loaded by peeled step 0)
        emit_rms(0, 0)
        # 2. Q, K, V GEMMs
        emit_gemm(Q_DIM, H)
        emit_gemm(KV_DIM, H)
        emit_gemm(KV_DIM, H)
        # 3. RoPE + bf16 KV drain + L1 q_bf16
        emit(KIND_X, 1, 1, r, 0, 0)
        emit(KIND_W, 1, 1, r, 0)
        emit(KIND_W, 1, 1, r, 1)
        emit(KIND_W, 1, 2, r, 0)
        emit(KIND_Y, 4, 2)
        # 4. FlashAttention (runtime kv_ch past chunks + cur_kv)
        emit(KIND_W, 1, 10 + r, 0, 0)
        emit(KIND_Y, 16, 3)
        # 5. Quantize AttnOut + O GEMM + ResAdd1
        emit_rms(-1, 1)
        emit_gemm(H, Q_DIM)
        emit_res()
        # 6. FFN RMSNorm + Gate + Up + SiLU*Up+Q8 + Down + ResAdd2
        emit_rms(1, 0)
        emit_gemm(FFN, H)
        emit_gemm(FFN, H)
        for _ in range(4):
            for j in range(4):
                emit(KIND_X, 1, 3, c, j, 0)
                emit(KIND_X, 1, 3, c, j, 1)
                emit(KIND_Y, 1, 5, j)
            emit(KIND_Y, 1, 5, 4)
        emit_gemm(H, FFN)
        emit_res()
        return np.array(rows, dtype=np.int32).reshape(-1)

    def core_body(x_in, w_in, y_out, sched_buf, f_gemm, f_fa, f_w, f_x, f_y, col_c, n_steps):
        idx_ty = IndexType.get()
        i32_ty = IntegerType.get_signless(32)
        idx0 = arith.constant(idx_ty, 0)
        idx1 = arith.constant(idx_ty, 1)
        idx2 = arith.constant(idx_ty, 2)
        idx3 = arith.constant(idx_ty, 3)
        idx4 = arith.constant(idx_ty, 4)
        idx5 = arith.constant(idx_ty, 5)
        idx6 = arith.constant(idx_ty, 6)
        idx_kvch = arith.constant(idx_ty, 4098)
        i32_0 = arith.constant(i32_ty, 0)
        i32_1 = arith.constant(i32_ty, 1)
        i32_2 = arith.constant(i32_ty, 2)
        i32_3 = arith.constant(i32_ty, 3)
        i32_10 = arith.constant(i32_ty, 10)

        # Peel step 0 (SLOT_META broadcast) so runtime kv_chunks (g_params[2])
        # drives the FlashAttention loop while sched_buf stays identical across buckets.
        wp0 = w_in.acquire(1)
        kv_ch = arith.bitcast(i32_ty, wp0[idx_kvch])
        has_kv = arith.cmpi(arith.CmpIPredicate.sgt, kv_ch, i32_0)
        f_w(wp0, i32_0, arith.constant(i32_ty, col_c), i32_0)
        w_in.release(1)

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
                is_fa = arith.cmpi(arith.CmpIPredicate.sge, op, i32_10)
                with if_(is_fa, hasElse=True) as if_fa:
                    row_arg = arith.subi(op, i32_10)
                    with if_(has_kv):
                        wp = w_in.acquire(1)
                        f_fa(wp, row_arg, i32_0, i32_2)
                        w_in.release(1)
                        extra_kv = arith.subi(kv_ch, i32_1)
                        for kc in range_(extra_kv):
                            wp = w_in.acquire(1)
                            f_fa(wp, row_arg, arith.addi(i32_1, i32(kc)), i32_0)
                            w_in.release(1)
                    cur_mode = arith.select(has_kv, i32_1, i32_3)
                    wp = w_in.acquire(1)
                    f_fa(wp, row_arg, i32_0, cur_mode)
                    w_in.release(1)
                with else_(if_fa):
                    for i in range_(cnt):
                        wp = w_in.acquire(1)
                        f_w(wp, op, arith.addi(a0, i32(i)), a1)
                        w_in.release(1)
            with else_(if_w):
                is_x = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_1)
                with if_(is_x, hasElse=True) as if_x:
                    for i in range_(cnt):
                        xp = x_in.acquire(1)
                        f_x(xp, op, a0, arith.addi(a1, i32(i)), a2)
                        x_in.release(1)
                with else_(if_x):
                    is_y = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_2)
                    with if_(is_y, hasElse=True) as if_y:
                        for i in range_(cnt):
                            yp = y_out.acquire(1)
                            f_y(yp, op, arith.addi(a0, i32(i)))
                            y_out.release(1)
                    with else_(if_y):
                        is_single_k = arith.cmpi(arith.CmpIPredicate.eq, a0, i32_1)
                        last_k = arith.subi(a0, i32_1)
                        for _p in range_(cnt):
                            for kc in range_(a0):
                                is_sub = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
                                first_bit = arith.select(is_sub, i32_1, i32_0)
                                flag = arith.ori(first_bit, arith.shli(op, i32_1))
                                is_last = arith.cmpi(arith.CmpIPredicate.eq, i32(kc), last_k)
                                xp = x_in.acquire(1)
                                for t in range_(a1):
                                    t_slot = arith.select(is_single_k, i32_0, i32(t))
                                    wp = w_in.acquire(1)
                                    f_gemm(xp, wp, t_slot, flag)
                                    w_in.release(1)
                                    with if_(is_last):
                                        yp = y_out.acquire(1)
                                        f_y(yp, 1, t_slot)
                                        y_out.release(1)
                                x_in.release(1)

    workers = []
    for c in range(N_COLS):
        for r in range(N_ROWS):
            sched = build_worker_sched(c, r)
            n_steps = len(sched) // 6
            sched_buf = Buffer(initial_value=sched, name=f"sched_{c}_{r}")
            workers.append(
                Worker(
                    lambda xi, wi, yo, sb, fg, ff, fw, fx, fy, cc=c, ns=n_steps: core_body(
                        xi, wi, yo, sb, fg, ff, fw, fx, fy, cc, ns
                    ),
                    [
                        x_l2l1[r].cons(),
                        w_l2l1[c].cons(),
                        y_l1l2_cols[c][r].prod(),
                        sched_buf,
                        gemm_op,
                        fa_op,
                        w_op,
                        x_op,
                        y_op,
                    ],
                    stack_size=3072,
                )
            )

    def tap_1d(total, off, length):
        return TensorAccessPattern((1, total), off, [1, 1, 1, length], [0, 0, 0, 1])

    def tap(total, off, sizes, strides):
        return TensorAccessPattern((1, total), off, sizes, strides)

    def seq_rmsnorm(y_buf, xp, wp, yp, in_slot, out_xq_slot, load_norm):
        tg = TaskGroup()
        if load_norm >= 0:
            for c in range(N_COLS):
                wp[c].fill(y_buf, tap_1d(ACT_TOTAL, SLOT_META, W_ELEMS), group=tg)
        for r in range(N_ROWS):
            for p in range(4):
                xp[r].fill(
                    y_buf,
                    tap_1d(ACT_TOTAL, in_slot + r * 8 * H + p * 2 * H, X_ELEMS),
                    group=tg,
                )
        for c in range(N_COLS):
            yp[c].drain(
                y_buf,
                tap(ACT_TOTAL, out_xq_slot + c * 512, [1, 4, 4, 128], [0, 128, X_ELEMS, 1]),
                wait=True,
                group=tg,
            )
            yp[c].drain(
                y_buf,
                tap(ACT_TOTAL, out_xq_slot + 4096 + c * 128, [1, 1, 4, 128], [0, 0, X_ELEMS, 1]),
                wait=True,
                group=tg,
            )
        tg.finish()

    def seq_gemm(w_buf, y_buf, xp, wp, yp, w_off, xq_off, out_off, out_tap_fn, N, K):
        tiles_per_col = (N // N_COLS) // ROWS
        k_chunks = K // CHUNK_K
        pass_tiles = 8 if (k_chunks > 1 and tiles_per_col > 8) else tiles_per_col
        n_passes = tiles_per_col // pass_tiles
        col_w_elems = tiles_per_col * k_chunks * W_ELEMS
        for p in range(n_passes):
            tg = TaskGroup()
            for kc in range(k_chunks):
                for r in range(N_ROWS):
                    xp[r].fill(
                        y_buf,
                        tap_1d(ACT_TOTAL, xq_off + (kc * N_ROWS + r) * X_ELEMS, X_ELEMS),
                        group=tg,
                    )
                for c in range(N_COLS):
                    wp[c].fill(
                        w_buf,
                        tap(
                            W_TOTAL,
                            w_off + c * col_w_elems + p * pass_tiles * k_chunks * W_ELEMS + kc * W_ELEMS,
                            [1, pass_tiles, W_ELEMS // 512, 512],
                            [0, k_chunks * W_ELEMS, 512, 1],
                        ),
                        group=tg,
                    )
            for c in range(N_COLS):
                yp[c].drain(y_buf, out_tap_fn(out_off, c, p, pass_tiles), wait=True, group=tg)
            tg.finish()

    def seq_res(y_buf, xp, yp, a_slot, b_slot, dst_buf, dst_slot):
        for j in range(4):
            tg = TaskGroup()
            for r in range(N_ROWS):
                xp[r].fill(
                    y_buf,
                    tap(ACT_TOTAL, a_slot + r * 8 * H + j * 512, [1, 1, 10, 512], [0, 0, H, 1]),
                    group=tg,
                )
            for r in range(N_ROWS):
                xp[r].fill(
                    y_buf,
                    tap(ACT_TOTAL, b_slot + r * 8 * H + j * 512, [1, 1, 10, 512], [0, 0, H, 1]),
                    group=tg,
                )
            for c in range(N_COLS):
                yp[c].drain(
                    dst_buf,
                    tap(ACT_TOTAL, dst_slot + j * 512 + c * 64, [1, 4, 32, 16], [0, 16, H, 1]),
                    wait=True,
                    group=tg,
                )
            tg.finish()

    def row_major_tap(N):
        def fn(out_off, c, p, pass_tiles):
            n_col = N // N_COLS
            return tap(
                ACT_TOTAL,
                out_off + c * n_col + p * pass_tiles * ROWS,
                [1, pass_tiles, B, ROWS],
                [0, ROWS, N, 1],
            )
        return fn

    def q_col_tap(out_off, c, _p, _pt):
        return tap(
            ACT_TOTAL,
            out_off + c * 2 * W_ELEMS,
            [2, 8, B, ROWS],
            [W_ELEMS, ROWS, 128, 1],
        )

    def kv_col_tap(sub_off):
        def fn(out_off, c, _p, _pt):
            return tap(
                ACT_TOTAL,
                out_off + c * W_ELEMS + sub_off,
                [1, 4, B, ROWS],
                [0, ROWS, 64, 1],
            )
        return fn

    def sequence(w_buf, y_buf, y_next_buf, kv_buf, x_prods, w_prods, y_conses):
        # 1. Attn RMSNorm + Q8
        seq_rmsnorm(y_buf, x_prods, w_prods, y_conses, SLOT_HIDDEN_IN, SLOT_XQ, 0)
        # 2. Q, K, V
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_Q_OFF, SLOT_XQ, SLOT_Q, q_col_tap, Q_DIM, H)
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_K_OFF, SLOT_XQ, SLOT_KV_F32, kv_col_tap(0), KV_DIM, H)
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_V_OFF, SLOT_XQ, SLOT_KV_F32, kv_col_tap(2048), KV_DIM, H)
        # 3. RoPE + KV drain
        tg_rope = TaskGroup()
        for r in range(N_ROWS):
            x_prods[r].fill(y_buf, tap_1d(ACT_TOTAL, SLOT_META + 4608, X_ELEMS), group=tg_rope)
        for c in range(N_COLS):
            w_prods[c].fill(y_buf, tap_1d(ACT_TOTAL, SLOT_Q + c * 2 * W_ELEMS, W_ELEMS), group=tg_rope)
            w_prods[c].fill(y_buf, tap_1d(ACT_TOTAL, SLOT_Q + c * 2 * W_ELEMS + W_ELEMS, W_ELEMS), group=tg_rope)
            w_prods[c].fill(y_buf, tap_1d(ACT_TOTAL, SLOT_KV_F32 + c * W_ELEMS, W_ELEMS), group=tg_rope)
            y_conses[c].drain(
                y_buf,
                tap(ACT_TOTAL, SLOT_KV_CUR + c * W_ELEMS, [1, 2, 4, 128], [0, 128, 256, 1]),
                wait=True,
                group=tg_rope,
            )
            y_conses[c].drain(
                y_buf,
                tap(ACT_TOTAL, SLOT_KV_CUR + c * W_ELEMS + 2304, [1, 2, 4, 128], [0, 128, 256, 1]),
                wait=True,
                group=tg_rope,
            )
        tg_rope.finish()
        # 4. FlashAttention
        tg_fa = TaskGroup()
        for c in range(N_COLS):
            if kv_chunks > 0:
                w_prods[c].fill(
                    kv_buf,
                    tap_1d(KV_TOTAL, c * MAX_KV_CHUNKS * W_ELEMS, kv_chunks * W_ELEMS),
                    group=tg_fa,
                )
            w_prods[c].fill(y_buf, tap_1d(ACT_TOTAL, SLOT_KV_CUR + c * W_ELEMS, W_ELEMS), group=tg_fa)
            y_conses[c].drain(
                y_buf,
                tap(ACT_TOTAL, SLOT_ATTN_OUT + c * 256, [1, 16, B, ROWS], [0, ROWS, H, 1]),
                wait=True,
                group=tg_fa,
            )
        tg_fa.finish()
        # 5. Quantize AttnOut + O GEMM + ResAdd1
        seq_rmsnorm(y_buf, x_prods, w_prods, y_conses, SLOT_ATTN_OUT, SLOT_XQ, -1)
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_O_OFF, SLOT_XQ, SLOT_O_OUT, row_major_tap(H), H, Q_DIM)
        seq_res(y_buf, x_prods, y_conses, SLOT_HIDDEN_IN, SLOT_O_OUT, y_buf, SLOT_HIDDEN_MID)
        # 6. FFN RMSNorm + Gate + Up + SiLU*Up + Down + ResAdd2
        seq_rmsnorm(y_buf, x_prods, w_prods, y_conses, SLOT_HIDDEN_MID, SLOT_XQ, 1)
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_GATE_OFF, SLOT_XQ, SLOT_GATE, row_major_tap(FFN), FFN, H)
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_UP_OFF, SLOT_XQ, SLOT_UP, row_major_tap(FFN), FFN, H)
        for kc in range(4):
            xq_chunk_off = SLOT_XQ + kc * N_ROWS * X_ELEMS
            for j in range(4):
                tg_silu = TaskGroup()
                for r in range(N_ROWS):
                    x_prods[r].fill(
                        y_buf,
                        tap(ACT_TOTAL, SLOT_GATE + r * 8 * FFN + kc * 2048 + j * 512, [1, 1, 10, 512], [0, 0, FFN, 1]),
                        group=tg_silu,
                    )
                for r in range(N_ROWS):
                    x_prods[r].fill(
                        y_buf,
                        tap(ACT_TOTAL, SLOT_UP + r * 8 * FFN + kc * 2048 + j * 512, [1, 1, 10, 512], [0, 0, FFN, 1]),
                        group=tg_silu,
                    )
                for c in range(N_COLS):
                    y_conses[c].drain(
                        y_buf,
                        tap(ACT_TOTAL, xq_chunk_off + j * 1024 + c * 128, [1, 1, 4, 128], [0, 0, X_ELEMS, 1]),
                        wait=True,
                        group=tg_silu,
                    )
                if j == 3:
                    for c in range(N_COLS):
                        y_conses[c].drain(
                            y_buf,
                            tap(ACT_TOTAL, xq_chunk_off + 4096 + c * 128, [1, 1, 4, 128], [0, 0, X_ELEMS, 1]),
                            wait=True,
                            group=tg_silu,
                        )
                tg_silu.finish()
        seq_gemm(w_buf, y_buf, x_prods, w_prods, y_conses, W_DOWN_OFF, SLOT_XQ, SLOT_DOWN_OUT, row_major_tap(H), H, FFN)
        seq_res(y_buf, x_prods, y_conses, SLOT_HIDDEN_MID, SLOT_DOWN_OUT, y_next_buf, SLOT_HIDDEN_IN)

    rt = Runtime(
        sequence,
        [
            w_bo_ty,
            act_bo_ty,
            act_bo_ty,
            kv_bo_ty,
            [of.prod() for of in x_l3l2],
            [of.prod() for of in w_l3l2],
            [of.cons() for of in y_l2l3],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kv-chunks", type=int, default=0)
    add_compile_args(p)
    opts = p.parse_args()
    run_design_cli(
        q4prefill_layer,
        opts,
        compile_kwargs=lambda o: dict(kv_chunks=o.kv_chunks),
        device=lambda o: device_from_args(o, n_cols=N_COLS),
    )


if __name__ == "__main__":
    main()

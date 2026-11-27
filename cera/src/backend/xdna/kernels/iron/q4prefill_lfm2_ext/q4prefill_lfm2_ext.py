# Unified Fused B=64 Q4_0 LFM2 / LFM2.5 >32K Prefill Layer (32 cores: 4 rows x 8 cols, npu2).
# K_HALF=1024, W_L2L1 depth=2, X_ELEMS=4608, Y_CORE_ELEMS=64.
# Modes via g_params[6]:
#   0 = ShortConv (d_conv=2, param-slot) + SwiGLU FFN (8192)
#   2 = Phase A (RMSNorm + QKV + QK-norm + NeoX RoPE -> SLOT_FA_Q + kvo_buf)
#   3 = FA Partial blk==0 (Reload q0/q1 only -> FA -> Drain SLOT_FA_STATE + SLOT_FA_LM)
#   4 = FA Partial blk>=1 (Reload q0/q1 + SLOT_FA_STATE + SLOT_FA_LM -> FA -> Drain)
#   5 = FA Final (Reload q0/q1 + SLOT_FA_STATE + SLOT_FA_LM -> FA + cur_kv -> O + SwiGLU FFN)
#   6 = FA Final without FFN (ends at O + residual -> yn_buf)
#   7 = ShortConv (d_conv=2, param-slot) without FFN (ends at CONV_OUT + residual -> yn_buf)

import argparse
import sys
from pathlib import Path

_IRON_ROOT = str(Path(__file__).resolve().parent.parent)
if _IRON_ROOT not in sys.path:
    sys.path.insert(0, _IRON_ROOT)

from compile_params import validate_compile_params

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
B_SUB = 8
B_CORE = 16
B = N_ROWS * B_CORE  # 64
ROWS = 16
CHUNK_K = 1024

W_ELEMS = 2304
X_ELEMS = 4608
Y_CORE_ELEMS = 64
Y_COL_ELEMS = N_ROWS * Y_CORE_ELEMS  # 256

H = 2048
Q_DIM = 2048
KV_DIM = 512
N_KV_HEADS = 8
MAX_KV_CHUNKS = 1024

# Weight offsets in 2304-float (9216 B) K=1024 half-tile packets
W_CONV_IN_B = 0
W_CONV_IN_C = 256 * W_ELEMS
W_CONV_IN_X = 512 * W_ELEMS
W_CONV_OUT = 768 * W_ELEMS
W_CONV_WT = 1024 * W_ELEMS
W_CONV_FFN_GATE = 1040 * W_ELEMS
W_CONV_FFN_UP = 2064 * W_ELEMS
W_CONV_FFN_DOWN = 3088 * W_ELEMS

W_ATTN_Q = 0
W_ATTN_K = 256 * W_ELEMS
W_ATTN_V = 320 * W_ELEMS
W_ATTN_O = 384 * W_ELEMS
W_ATTN_FFN_GATE = 640 * W_ELEMS
W_ATTN_FFN_UP = 1664 * W_ELEMS
W_ATTN_FFN_DOWN = 2688 * W_ELEMS
W_TOTAL = 4112 * W_ELEMS

SLOT_HIDDEN_IN = 0
SLOT_Q = 131072
SLOT_K = 262144
SLOT_V = 294912
SLOT_ATTN_OUT = 327680
SLOT_RES_TMP = 458752
SLOT_CONV_B = 589824
SLOT_CONV_C = 720896
SLOT_CONV_X = 851968
SLOT_FFN_XQ = SLOT_CONV_B  # [589824..737280): 8*4*4608 f32 for FFN SiLU*Up Q8, disjoint from SLOT_XQ (FFN_NORM)
SLOT_FA_Q = 737280         # [737280..811008): 2 * 2 * 18432 f32 saved q_bf16 for Mode 2->4/5
SLOT_FA_STATE = 811008     # [811008..958464): 2 * 4 * 18432 f32 saved o_acc (sub=0, sub=1) for Mode 4<->5
SLOT_EXT_PARAMS_0 = 958464 # [958464..960768): Mode 4 fa_part0 (blk 0) params at +2048
SLOT_EXT_PARAMS_1 = 960768 # [960768..963072): Mode 4 fa_part (blk >= 1) params at +2048
SLOT_EXT_PARAMS_2 = 963072 # [963072..965376): Mode 5/6 fa_fin params at +2048
SLOT_FA_LM = 965376        # [965376..971264): 8 * 512..2304 f32 saved row_l + row_m for Mode 4<->5
SLOT_XQ = 983040
SLOT_XQ_K0 = SLOT_XQ
SLOT_XQ_K1 = SLOT_XQ + N_ROWS * X_ELEMS
SLOT_NORM_0 = 1146880
SLOT_NORM_1 = 1149184
SLOT_ROPE_PKT_0 = 1151488
SLOT_ROPE_PKT_1 = 1153792
ACT_TOTAL = 1153792 + W_ELEMS

YN_TOTAL = B * H  # 131072
KV_CHUNK_F32 = W_ELEMS  # 2304 (9216 B contiguous per 32-token chunk, 2 chunks = 4608 in decode)
CONV_TILE_F32 = 2 * W_ELEMS  # 4608 (18432 B per col for conv_wt & conv state, shared with decode)
KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * KV_CHUNK_F32
KV_OUT_TOTAL = 2 * N_KV_HEADS * W_ELEMS

VALID_MODES = (0, 2, 3, 4, 5, 6, 7)
VALID_PARAM_SLOTS = (0, 1, 2, 3)


def _validate_compile_params(layer_mode: int, kv_chunks: int, param_slot: int = 0) -> None:
    validate_compile_params(
        layer_mode,
        kv_chunks,
        valid_modes=VALID_MODES,
        nonzero_kv_modes=(3, 4),
        max_kv_chunks=MAX_KV_CHUNKS,
        param_slot=param_slot,
        valid_param_slots=VALID_PARAM_SLOTS,
    )


@iron.jit(use_cache=False)
def q4prefill_lfm2(
    w: In,
    y: Out,
    y_next: Out,
    kv: In,
    kv_out: Out,
    layer_mode: CompileTime[int] = 0,
    kv_chunks: CompileTime[int] = 0,
    param_slot: CompileTime[int] = 0,
):
    _validate_compile_params(layer_mode, kv_chunks, param_slot)
    here = Path(__file__).resolve().parent
    cxx_inc = cxx_header_path()

    w_ty = np.ndarray[(W_TOTAL,), np.dtype[np.float32]]
    y_ty = np.ndarray[(ACT_TOTAL,), np.dtype[np.float32]]
    yn_ty = np.ndarray[(YN_TOTAL,), np.dtype[np.float32]]
    kv_ty = np.ndarray[(KV_TOTAL,), np.dtype[np.float32]]
    kvo_ty = np.ndarray[(KV_OUT_TOTAL,), np.dtype[np.float32]]

    w_pkt_ty = np.ndarray[(W_ELEMS,), np.dtype[np.float32]]
    x_pkt_ty = np.ndarray[(X_ELEMS,), np.dtype[np.float32]]
    x_l3l2_ty = np.ndarray[(N_ROWS * X_ELEMS,), np.dtype[np.float32]]
    y_core_ty = np.ndarray[(Y_CORE_ELEMS,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_ELEMS,), np.dtype[np.float32]]

    base_flags = ["-I", str(here), "-I", str(cxx_inc), "-fno-unroll-loops", "-fno-jump-tables", "-mllvm", "-aie-enable-tail-merge"]
    oz_flags = base_flags + ["-Oz"]
    fa_flags = base_flags + ["-O2", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]
    gemm_flags = ["-I", str(here), "-I", str(cxx_inc), "-O2", "-fno-unroll-loops", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]
    core_flags = base_flags + ["-Oz"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm2_pre_gemm_op",
        source_file=str(here / "q4_lfm2_gemm_op.cc"),
        arg_types=[x_pkt_ty, w_pkt_ty, np.int32, np.int32],
        compile_flags=gemm_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm2_pre_w_op",
        source_file=str(here / "q4_lfm2_pre_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=fa_flags,
    )
    q4_x_fn = ExternalFunction(
        "q4_lfm2_pre_x_op",
        source_file=str(here / "q4_lfm2_pre_x_op.cc"),
        arg_types=[x_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm2_pre_y_op",
        source_file=str(here / "q4prefill_lfm2_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=core_flags,
    )

    W_L3L2 = [ObjectFifo(w_pkt_ty, depth=4, name=f"W_L3L2_{c}") for c in range(N_COLS)]
    W_L2L1 = [W_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=2, name=f"W_L2L1_{c}") for c in range(N_COLS)]

    N_X_SHIMS = 2
    X_L3L2 = [ObjectFifo(x_l3l2_ty, depth=2, name=f"X_L3L2_{x}") for x in range(N_X_SHIMS)]
    X_L2L1 = [
        X_L3L2[x].cons().split(
            offsets=[r * X_ELEMS for r in range(N_ROWS)],
            obj_types=[x_pkt_ty] * N_ROWS,
            names=[f"X_L2L1_{x}_{r}" for r in range(N_ROWS)],
            depths=[1] * N_ROWS,
        )
        for x in range(N_X_SHIMS)
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

    M0 = 1
    M2 = 4
    M3 = 8
    M4 = 16
    M5 = 32
    M6 = 64
    M7 = 128
    M07 = M0 | M7
    M027 = M0 | M2 | M7
    M34 = M3 | M4
    M3456 = M3 | M4 | M5 | M6
    M456 = M4 | M5 | M6
    M56 = M5 | M6
    M05 = M0 | M5

    def build_worker_sched(c, r):
        rows = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0, mask=M0):
            rows.append((kind, cnt, op, a0, a1, a2, mask))

        def emit_rms_2sub(norm_op_kind, skip_norm, mask):
            if norm_op_kind >= 0:
                emit(KIND_W, 1, norm_op_kind, c, 0, 0, mask)
            for _ in range(2):
                emit(KIND_X, 4, 0, c, 0, skip_norm, mask)
                emit(KIND_Y, 9, 0, 0, 0, 0, mask)

        def emit_gemm(n_passes, pass_tiles, k_chunks, xd_mode, mask):
            for _ in range(n_passes):
                emit(KIND_GEMM, k_chunks, xd_mode, pass_tiles, 0, 0, mask)
                emit(KIND_Y, pass_tiles * 4, 1, 0, 0, 0, mask)

        def emit_gemm_res(k_chunks, xd_mode, use_cxd, mask):
            emit(KIND_GEMM, k_chunks, xd_mode, 16, 0, use_cxd, mask)
            emit(KIND_W, 8, 6, r, 0, 0, mask)
            emit(KIND_Y, 64, 1, 0, 0, 0, mask)

        # Stage 1 (shared across Modes 0, 2, 7): Attn RMSNorm + Q8
        emit_rms_2sub(-1, 0, M027)

        # Mode 0 / 7 (ShortConv): IN_B/C/X -> conv(direct XQ + conv_xd) -> CONV_OUT + Res1 fused!
        for _ in range(3):
            emit_gemm(1, 16, 2, 0, M07)
        emit(KIND_W, 1, 5, r, 15, 0, M07)
        emit(KIND_W, 1, 5, r, 0, 0, M07)
        emit(KIND_W, 8, 5, r, 1, 0, M07)
        emit(KIND_W, 2, 5, r, 9, 0, M07)
        emit(KIND_Y, 4, 0, 0, 0, 0, M07)
        emit(KIND_W, 2, 5, r, 11, 0, M07)
        emit(KIND_Y, 5, 0, 4, 0, 0, M07)
        emit(KIND_W, 1, 5, r, 16, 0, M07)
        emit(KIND_W, 7, 5, r, 2, 0, M07)
        emit(KIND_W, 2, 5, r, 9, 0, M07)
        emit(KIND_Y, 4, 0, 0, 0, 0, M07)
        emit(KIND_W, 2, 5, r, 11, 0, M07)
        emit(KIND_Y, 5, 0, 4, 0, 0, M07)
        emit(KIND_Y, 2, 8, 8, 0, 0, M07)
        emit(KIND_W, 2, 3, r, 0, 0, M07)
        # CONV_OUT + fused Res1
        emit_gemm_res(2, 0, 1, M07)

        # Mode 2 (Phase A): KV -> kvo_buf directly (V first, then K sub=0, then K sub=1)
        emit(KIND_GEMM, 2, 0, 8, 0, 0, M2)
        emit(KIND_Y, 8, 2, 8, 0, 0, M2)
        emit(KIND_W, 1, 1, r, 0, 0, M2)
        emit(KIND_Y, 4, 7, 0, 0, 0, M2)
        emit(KIND_W, 1, 1, r, 1, 0, M2)
        emit(KIND_Y, 4, 7, 0, 0, 0, M2)

        # ATTN_Q + in-L1 Q_norm/RoPE (sub=1 first -> q1_bf16, then sub=0 -> q_bf16)
        emit(KIND_GEMM, 2, 0, 16, 0, 0, M2)
        emit(KIND_W, 1, 2, r, 1, 0, M2)
        emit(KIND_W, 1, 2, r, 0, 0, M2)
        emit(KIND_Y, 32, 7, 0, 0, 0, M2)

        # Modes 3, 4, 5, 6 (>32K paged single-pass quad FA)
        emit(KIND_X, 2, 1, c, 0, 2, M3456)
        emit(KIND_X, 2, 1, c, 0, 3, M3456)
        emit(KIND_X, 4, 1, c, 0, 0, M456)
        emit(KIND_X, 4, 1, c, 0, 1, M456)
        emit(KIND_W, 1, 7, r, 0, 0, M456)
        emit(KIND_W, 1, 16 + r, 0, 0, 0, M3456)
        emit(KIND_Y, 34, 10, 0, 0, 0, M34)
        emit(KIND_Y, 32, 6, 0, 0, 0, M34)
        emit(KIND_Y, 9, 3, 0, 0, 0, M56)
        emit(KIND_Y, 9, 4, 0, 0, 0, M56)

        # ATTN_O + fused Res1 (M56)
        emit_gemm_res(2, 0, 0, M56)

        # Shared tail: FFN_NORM -> 8x(GATE/UP + SiLU*Up) -> DOWN + fused Res2
        emit_rms_2sub(4, 0, M05)
        for _ in range(8):
            emit(KIND_GEMM, 2, 0, 16, 0, 2, M05)
            emit(KIND_Y, 9, 9, 0, 0, 0, M05)
        emit_gemm_res(8, 1, 0, M05)
        packed = [
            (k | (cn << 8) | (op << 16) | (a0 << 24), a1 | (a2 << 8) | (mk << 16))
            for (k, cn, op, a0, a1, a2, mk) in rows
        ]
        return np.array(packed, dtype=np.int32).reshape(-1)

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
            idx9 = arith.constant(idx_ty, 9)
            idx_kvc = arith.constant(idx_ty, 2050)
            idx_lmode = arith.constant(idx_ty, 2054)
            i32_0 = arith.constant(i32_ty, 0)
            i32_1 = arith.constant(i32_ty, 1)
            i32_2 = arith.constant(i32_ty, 2)
            i32_3 = arith.constant(i32_ty, 3)
            i32_5 = arith.constant(i32_ty, 5)
            i32_8 = arith.constant(i32_ty, 8)
            i32_9 = arith.constant(i32_ty, 9)
            i32_16 = arith.constant(i32_ty, 16)
            i32_20 = arith.constant(i32_ty, 20)
            i32_24 = arith.constant(i32_ty, 24)
            i32_255 = arith.constant(i32_ty, 255)

            def i32(v):
                return arith.index_cast(i32_ty, v)

            wp0 = w_fifo.acquire(1)
            kv_ch = arith.bitcast(i32_ty, wp0[idx_kvc])
            lmode = arith.bitcast(i32_ty, wp0[idx_lmode])
            q4_w_fn(wp0, i32_0, arith.constant(i32_ty, c), i32_0)
            w_fifo.release(1)

            lmode_bit = arith.shli(i32_1, lmode)
            is_m4 = arith.cmpi(arith.CmpIPredicate.slt, lmode, i32_5)

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
                        is_fa0 = arith.cmpi(arith.CmpIPredicate.sge, op, i32_16)
                        extra_kv = arith.select(is_m4, i32_0, i32_2)
                        fa_cnt = arith.addi(kv_ch, extra_kv)
                        n_w = arith.select(is_fa0, fa_cnt, cnt)
                        for i in range_(n_w):
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
                                for p in range_(cnt):
                                    is_sub = arith.cmpi(arith.CmpIPredicate.ne, p, idx0)
                                    flag = arith.ori(arith.select(is_sub, i32_1, i32_0), arith.ori(arith.shli(op, i32_1), arith.shli(a2, i32_2)))
                                    xp = x_fifo.acquire(1)
                                    for pt in range_(a0):
                                        wp = w_fifo.acquire(1)
                                        q4_gemm_fn(xp, wp, i32(pt), flag)
                                        w_fifo.release(1)
                                    x_fifo.release(1)

        return core_body

    workers = [
        Worker(
            make_worker_body(c),
            fn_args=[
                W_L2L1[c].cons(),
                X_L2L1[c // (N_COLS // N_X_SHIMS)][r].cons(),
                Y_L1L2[c][r].prod(),
                sched_bufs[c][r],
                q4_gemm_fn,
                q4_w_fn,
                q4_x_fn,
                q4_y_fn,
            ],
            tile=Tile(c, 2 + r),
            stack_size=2880,
            data_size=24704,
            while_true=True,
        )
        for c in range(N_COLS)
        for r in range(N_ROWS)
    ]

    w_prods = [W_L3L2[c].prod() for c in range(N_COLS)]
    x_prods = [X_L3L2[x].prod() for x in range(N_X_SHIMS)]
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, *fifos):
        wp_h = fifos[0:N_COLS]
        xp_h = fifos[N_COLS : N_COLS + N_X_SHIMS]
        yc_h = fifos[N_COLS + N_X_SHIMS : N_COLS + N_X_SHIMS + N_COLS]

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
            for x in range(N_X_SHIMS):
                xp_h[x].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off, sizes=[N_ROWS, 8, 576], strides=[row_stride, 576, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_x_xq(tg, kc=0, wait=False, x_slot=SLOT_XQ):
            xq_off = x_slot + kc * (N_ROWS * X_ELEMS)
            for x in range(N_X_SHIMS):
                xp_h[x].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL], offset=xq_off, sizes=[1, N_ROWS * X_ELEMS], strides=[0, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def drain_xq_sub(tg, sub, wait_last=True, do_finish=True):
            xq_base = sub * 2048
            xd_base = 4096 + sub * 256
            for c in range(N_COLS):
                kh_slot = SLOT_XQ_K0 if c < 4 else SLOT_XQ_K1
                c_kh = c & 3
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=kh_slot + xq_base + c_kh * 512,
                        sizes=[8, N_ROWS, Y_CORE_ELEMS],
                        strides=[Y_CORE_ELEMS, X_ELEMS, 1],
                    ),
                    wait=False,
                    group=tg,
                )
            for c in range(N_COLS):
                kh_slot = SLOT_XQ_K0 if c < 4 else SLOT_XQ_K1
                c_kh = c & 3
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=kh_slot + xd_base + c_kh * Y_CORE_ELEMS,
                        sizes=[N_ROWS, Y_CORE_ELEMS],
                        strides=[X_ELEMS, 1],
                    ),
                    wait=wait_last,
                    group=tg,
                )
            if do_finish:
                tg.finish()

        def drain_y_contig_rows(dst_buf, dst_tdims, dst_off, sub, tg, fa_perm=False, do_finish=True):
            sub_off = sub * (32 * H if fa_perm else 8 * H)
            row_st = B_SUB * H if fa_perm else B_CORE * H
            for c in range(N_COLS):
                yc_h[c].drain(
                    dst_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=dst_tdims,
                        offset=dst_off + sub_off + c * 256,
                        sizes=[8, 4, N_ROWS, Y_CORE_ELEMS],
                        strides=[H, Y_CORE_ELEMS, row_st, 1],
                    ),
                    wait=True,
                    group=tg,
                )
            if do_finish:
                tg.finish()

        def drain_gemm_64(tg, out_slot, pass_off, pass_tiles, N_total, wait=True, fa_perm=False):
            col_ch = N_total // N_COLS
            n_p4 = pass_tiles // 4
            if fa_perm or n_p4 == 1:
                sizes = [2, B_SUB, N_ROWS, Y_CORE_ELEMS] if fa_perm else [B_CORE, N_ROWS, Y_CORE_ELEMS]
                strides = [32 * N_total, N_total, B_SUB * N_total, 1] if fa_perm else [N_total, B_CORE * N_total, 1]
                for p4 in range(n_p4):
                    is_last = wait and (p4 == n_p4 - 1)
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=out_slot + c * col_ch + pass_off + p4 * 64,
                                sizes=sizes,
                                strides=strides,
                            ),
                            wait=is_last,
                            group=tg,
                        )
            else:
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=out_slot + c * col_ch + pass_off,
                            sizes=[n_p4, B_CORE, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, N_total, B_CORE * N_total, 1],
                        ),
                        wait=wait,
                        group=tg,
                    )

        def run_rms_to_xq(norm_off, x_slot, x_stride, fa_perm=False):
            sub1_off = x_slot + (32 * H if fa_perm else B_SUB * H)
            tg0 = TaskGroup()
            if norm_off is not None:
                fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg0, wait=True)
            for p in range(4):
                fill_x_2d(y_buf, [ACT_TOTAL], x_slot + p * (2 * H), x_stride, tg0, wait=(p == 3))
            tg0.finish()
            tg1 = TaskGroup()
            drain_xq_sub(tg1, 0, wait_last=False, do_finish=False)
            for p in range(4):
                fill_x_2d(y_buf, [ACT_TOTAL], sub1_off + p * (2 * H), x_stride, tg1, wait=(p == 3))
            drain_xq_sub(tg1, 1, wait_last=True, do_finish=False)
            tg1.finish()

        def run_gemm_2d(w_off, n_passes, pass_tiles, k_chunks, out_slot, N_total, x_slot=SLOT_XQ, fa_perm=False, prev_drain=None, defer_drain=False):
            tpc = n_passes * pass_tiles
            cur_drain = prev_drain
            for p in range(n_passes):
                for kc0 in range(0, k_chunks, 4):
                    kc1 = min(kc0 + 4, k_chunks)
                    is_last_blk = kc1 == k_chunks
                    tg = TaskGroup()
                    for kc in range(kc0, kc1):
                        is_last_kc = kc == (kc1 - 1)
                        fill_x_xq(tg, kc, wait=is_last_kc, x_slot=x_slot)
                        for c in range(N_COLS):
                            tile0 = c * tpc + p * pass_tiles
                            wp_h[c].fill(
                                w_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[W_TOTAL],
                                    offset=w_off + (tile0 * k_chunks + kc) * W_ELEMS,
                                    sizes=[pass_tiles, 9, 256],
                                    strides=[k_chunks * W_ELEMS, 256, 1],
                                ),
                                wait=is_last_kc,
                                group=tg,
                            )
                    if cur_drain is not None:
                        cur_drain.finish()
                        cur_drain = None
                    if is_last_blk:
                        if defer_drain:
                            tg.finish()
                            cur_drain = TaskGroup()
                            drain_gemm_64(cur_drain, out_slot, p * pass_tiles * ROWS, pass_tiles, N_total, wait=True, fa_perm=fa_perm)
                            return cur_drain
                        else:
                            drain_gemm_64(tg, out_slot, p * pass_tiles * ROWS, pass_tiles, N_total, wait=True, fa_perm=fa_perm)
                    tg.finish()
            return cur_drain

        def drain_gemm_to(tg, dst_buf, dst_tdims, out_slot, pass_off, pass_tiles, N_total, wait=True, fa_perm=False):
            col_ch = N_total // N_COLS
            n_p4 = pass_tiles // 4
            if fa_perm or n_p4 == 1:
                sizes = [2, B_SUB, N_ROWS, Y_CORE_ELEMS] if fa_perm else [B_CORE, N_ROWS, Y_CORE_ELEMS]
                strides = [32 * N_total, N_total, B_SUB * N_total, 1] if fa_perm else [N_total, B_CORE * N_total, 1]
                for p4 in range(n_p4):
                    is_last = wait and (p4 == n_p4 - 1)
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            dst_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=dst_tdims,
                                offset=out_slot + c * col_ch + pass_off + p4 * 64,
                                sizes=sizes,
                                strides=strides,
                            ),
                            wait=is_last,
                            group=tg,
                        )
            else:
                for c in range(N_COLS):
                    yc_h[c].drain(
                        dst_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=dst_tdims,
                            offset=out_slot + c * col_ch + pass_off,
                            sizes=[n_p4, B_CORE, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, N_total, B_CORE * N_total, 1],
                        ),
                        wait=wait,
                        group=tg,
                    )

        def run_gemm_and_res(w_off, k_chunks, dest_buf, dest_tdims, dest_slot, x_slot=SLOT_XQ, fa_perm=False, prev_drain=None, pre_cxd=False):
            row_st = B_SUB * H if fa_perm else B_CORE * H
            cur_drain = prev_drain
            merge_res = (k_chunks == 2 and not pre_cxd)
            for kc0 in range(0, k_chunks, 4):
                kc1 = min(kc0 + 4, k_chunks)
                tg = TaskGroup()
                if pre_cxd and kc0 == 0:
                    fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ATTN_OUT + 2304, tg, wait=False)
                for kc in range(kc0, kc1):
                    is_last_kc = kc == (kc1 - 1)
                    fill_x_xq(tg, kc, wait=is_last_kc, x_slot=x_slot)
                    for c in range(N_COLS):
                        tile0 = c * 16
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=w_off + (tile0 * k_chunks + kc) * W_ELEMS,
                                sizes=[16, 9, 256],
                                strides=[k_chunks * W_ELEMS, 256, 1],
                            ),
                            wait=(is_last_kc and not merge_res),
                            group=tg,
                        )
                if cur_drain is not None:
                    cur_drain.finish()
                    cur_drain = None
                if merge_res:
                    for sub in range(2):
                        sub_tok_off = sub * (32 * H if fa_perm else B_SUB * H)
                        for c in range(N_COLS):
                            wp_h[c].fill(
                                y_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[ACT_TOTAL],
                                    offset=SLOT_HIDDEN_IN + sub_tok_off + c * 256,
                                    sizes=[N_ROWS, B_SUB, 288],
                                    strides=[row_st, H, 1],
                                ),
                                wait=(sub == 1),
                                group=tg,
                            )
                    drain_gemm_to(tg, dest_buf, dest_tdims, dest_slot, 0, 16, H, wait=True, fa_perm=fa_perm)
                tg.finish()
            if not merge_res:
                tg_r = TaskGroup()
                for sub in range(2):
                    sub_tok_off = sub * (32 * H if fa_perm else B_SUB * H)
                    for c in range(N_COLS):
                        wp_h[c].fill(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_HIDDEN_IN + sub_tok_off + c * 256,
                                sizes=[N_ROWS, B_SUB, 288],
                                strides=[row_st, H, 1],
                            ),
                            wait=(sub == 1),
                            group=tg_r,
                        )
                drain_gemm_to(tg_r, dest_buf, dest_tdims, dest_slot, 0, 16, H, wait=True, fa_perm=fa_perm)
                tg_r.finish()

        def drain_sc_q_pair(tg, sub, q_pair, wait=False):
            q_hi = q_pair >> 1
            row_base = (sub * 2 + q_hi) * X_ELEMS
            for c in range(N_COLS):
                kh_slot = SLOT_XQ_K0 if c < 4 else SLOT_XQ_K1
                c_kh = c & 3
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=kh_slot + row_base + c_kh * 512,
                        sizes=[2, 2, N_ROWS, Y_CORE_ELEMS],
                        strides=[2048, Y_CORE_ELEMS, 128, 1],
                    ),
                    wait=wait,
                    group=tg,
                )

        def drain_sc_xd(tg, sub, wait=False):
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_ATTN_OUT + sub * 2304 + c * 256,
                        sizes=[1, N_ROWS * Y_CORE_ELEMS],
                        strides=[0, 1],
                    ),
                    wait=wait,
                    group=tg,
                )

        # Peeled metadata + param_slot broadcast
        param_off = (SLOT_NORM_0, SLOT_EXT_PARAMS_0, SLOT_EXT_PARAMS_1, SLOT_EXT_PARAMS_2)[param_slot]
        tg0 = TaskGroup()
        fill_w_broadcast(y_buf, [ACT_TOTAL], param_off, tg0, wait=True)
        tg0.finish()

        if layer_mode in (0, 2, 7):
            is_attn_m = layer_mode == 2
            run_rms_to_xq(None, SLOT_HIDDEN_IN, (B_SUB * H if is_attn_m else B_CORE * H), fa_perm=is_attn_m)

        if layer_mode in (0, 7):
            dg = run_gemm_2d(W_CONV_IN_B, 1, 16, 2, SLOT_CONV_B, H, defer_drain=True)
            dg = run_gemm_2d(W_CONV_IN_C, 1, 16, 2, SLOT_CONV_C, H, prev_drain=dg, defer_drain=True)
            dg = run_gemm_2d(W_CONV_IN_X, 1, 16, 2, SLOT_CONV_X, H, prev_drain=dg, defer_drain=True)

            def fill_conv_4(slot_base, q0, tg, wait=False):
                off = slot_base + q0 * (8 * H)
                for c in range(N_COLS):
                    wp_h[c].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=off + c * 256,
                            sizes=[32, 288],
                            strides=[H, 1],
                        ),
                        wait=wait,
                        group=tg,
                    )

            tg_sc_a = TaskGroup()
            fill_w_1d(w_buf, [W_TOTAL], [W_CONV_WT + c * CONV_TILE_F32 for c in range(N_COLS)], tg_sc_a)
            fill_w_1d(kv_buf, [KV_TOTAL], [c * CONV_TILE_F32 for c in range(N_COLS)], tg_sc_a)
            fill_conv_4(SLOT_CONV_B, 0, tg_sc_a)
            if dg is not None:
                dg.finish()
            fill_conv_4(SLOT_CONV_X, 0, tg_sc_a, wait=True)
            tg_sc_a.finish()

            # Stage 5..8 (C0) + stage 9..10 (C0[0..1]) -> emit(Y, 4) -> drain_sc_q_pair(0, 0)
            tg_sc_b0 = TaskGroup()
            fill_conv_4(SLOT_CONV_C, 0, tg_sc_b0, wait=True)
            drain_sc_q_pair(tg_sc_b0, 0, 0, wait=True)
            tg_sc_b0.finish()

            # Stage 11..12 (C0[2..3]) -> emit(Y, 5) -> drain_sc_q_pair(0, 2) + drain_sc_xd(0)
            # Overlap B4 + X4 + C4 for sub=1 (total 3 W? No: B4, X4, C4 is 3 W_L3L2 fills? wait fill_conv_4 is 1 W BD!)
            tg_sc_b1 = TaskGroup()
            fill_conv_4(SLOT_CONV_B, 4, tg_sc_b1)
            fill_conv_4(SLOT_CONV_X, 4, tg_sc_b1)
            fill_conv_4(SLOT_CONV_C, 4, tg_sc_b1, wait=True)
            drain_sc_q_pair(tg_sc_b1, 0, 2, wait=False)
            drain_sc_xd(tg_sc_b1, 0, wait=True)
            tg_sc_b1.finish()

            tg_sc_d = TaskGroup()
            drain_sc_q_pair(tg_sc_d, 1, 0, wait=False)
            drain_sc_q_pair(tg_sc_d, 1, 2, wait=False)
            drain_sc_xd(tg_sc_d, 1, wait=False)
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ATTN_OUT, tg_sc_d, wait=True)
            for c in range(N_COLS):
                yc_h[c].drain(
                    kvo_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[KV_OUT_TOTAL],
                        offset=c * CONV_TILE_F32,
                        sizes=[2, N_ROWS, Y_CORE_ELEMS],
                        strides=[256, Y_CORE_ELEMS, 1],
                    ),
                    wait=True,
                    group=tg_sc_d,
                )
            tg_sc_d.finish()

            dst_buf = yn_buf if layer_mode == 7 else y_buf
            dst_tdims = [YN_TOTAL] if layer_mode == 7 else [ACT_TOTAL]
            dst_slot = 0 if layer_mode == 7 else SLOT_HIDDEN_IN
            run_gemm_and_res(W_CONV_OUT, 2, dst_buf, dst_tdims, dst_slot, pre_cxd=True)
        elif layer_mode == 2:
            tg_kv = TaskGroup()
            for kh in range(2):
                fill_x_xq(tg_kv, kh, wait=(kh == 1))
                for mat_off in (W_ATTN_K, W_ATTN_V):
                    is_v = mat_off == W_ATTN_V
                    for c in range(N_COLS):
                        tile0 = c * 4
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=mat_off + (tile0 * 2 + kh) * W_ELEMS,
                                sizes=[4, 9, 256],
                                strides=[2 * W_ELEMS, 256, 1],
                            ),
                            wait=(is_v and kh == 1),
                            group=tg_kv,
                        )
            tg_kv.finish()

            tg_k_rope = TaskGroup()
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT_0, tg_k_rope, wait=False)
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT_1, tg_k_rope, wait=True)
            for c in range(N_COLS):
                for sub_kvo, kvo_base in ((0, c * W_ELEMS), (1, (N_COLS + c) * W_ELEMS)):
                    yc_h[c].drain(
                        kvo_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[KV_OUT_TOTAL],
                            offset=kvo_base + 1024,
                            sizes=[4, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, 256, 1],
                        ),
                        wait=False,
                        group=tg_k_rope,
                    )
                for sub_kvo, kvo_base in ((0, c * W_ELEMS), (1, (N_COLS + c) * W_ELEMS)):
                    yc_h[c].drain(
                        kvo_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[KV_OUT_TOTAL],
                            offset=kvo_base,
                            sizes=[4, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, 256, 1],
                        ),
                        wait=(sub_kvo == 1),
                        group=tg_k_rope,
                    )
            tg_k_rope.finish()

            tg_q = TaskGroup()
            for kc in range(2):
                fill_x_xq(tg_q, kc, wait=(kc == 1))
                for c in range(N_COLS):
                    tile0 = c * 16
                    wp_h[c].fill(
                        w_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[W_TOTAL],
                            offset=W_ATTN_Q + (tile0 * 2 + kc) * W_ELEMS,
                            sizes=[16, 9, 256],
                            strides=[2 * W_ELEMS, 256, 1],
                        ),
                        wait=False,
                        group=tg_q,
                    )
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT_1, tg_q, wait=False)
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT_0, tg_q, wait=True)
            tg_q.finish()

            for sub in range(2):
                tg_qd = TaskGroup()
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_FA_Q + sub * (2 * N_ROWS * X_ELEMS) + (c // 4) * (N_ROWS * X_ELEMS) + (c & 3) * 1024,
                            sizes=[16, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, X_ELEMS, 1],
                        ),
                        wait=True,
                        group=tg_qd,
                    )
                tg_qd.finish()
        else:
            # Modes 3, 4, 5, 6 (>32K paged single-pass quad FA)
            n_x_pkts = 4 if layer_mode == 3 else 12
            w_tasks = []
            if layer_mode != 3:
                w_tasks.append((y_buf, [ACT_TOTAL], lambda c: SLOT_FA_LM + c * 512, [1, W_ELEMS], [0, 1]))
            n128 = kv_chunks // 128
            rem = kv_chunks % 128
            if n128 > 0:
                w_tasks.append((kv_buf, [KV_TOTAL], lambda c: (c * MAX_KV_CHUNKS) * KV_CHUNK_F32, [n128, 128, 9, 256], [128 * KV_CHUNK_F32, KV_CHUNK_F32, 256, 1]))
            if rem > 0:
                w_tasks.append((kv_buf, [KV_TOTAL], lambda c, s=n128 * 128: (c * MAX_KV_CHUNKS + s) * KV_CHUNK_F32, [rem, 9, 256], [KV_CHUNK_F32, 256, 1]))
            if layer_mode not in (3, 4):
                w_tasks.append((kvo_buf, [KV_OUT_TOTAL], lambda c: c * W_ELEMS, [2, 9, 256], [N_COLS * W_ELEMS, 256, 1]))
            cur_tg = TaskGroup()
            for x in range(N_X_SHIMS):
                xp_h[x].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_FA_Q,
                        sizes=[n_x_pkts, N_ROWS * X_ELEMS],
                        strides=[N_ROWS * X_ELEMS, 1],
                    ),
                    wait=True,
                    group=cur_tg,
                )
            for wi, (buf, tdims, off_fn, sz, st) in enumerate(w_tasks):
                do_wait = wi == len(w_tasks) - 1
                for c in range(N_COLS):
                    wp_h[c].fill(buf, tap=TensorAccessPattern(tensor_dims=tdims, offset=off_fn(c), sizes=sz, strides=st), wait=do_wait, group=cur_tg)
            if layer_mode in (3, 4):
                for c in range(N_COLS):
                    yc_h[c].drain(y_buf, tap=TensorAccessPattern(tensor_dims=[ACT_TOTAL], offset=SLOT_FA_STATE + (c // 2) * (N_ROWS * X_ELEMS) + (c & 1) * 2048, sizes=[32, N_ROWS, Y_CORE_ELEMS], strides=[Y_CORE_ELEMS, X_ELEMS, 1]), wait=False, group=cur_tg)
                for c in range(N_COLS):
                    yc_h[c].drain(y_buf, tap=TensorAccessPattern(tensor_dims=[ACT_TOTAL], offset=SLOT_FA_LM + c * 512, sizes=[2, N_ROWS, Y_CORE_ELEMS], strides=[Y_CORE_ELEMS, 128, 1]), wait=False, group=cur_tg)
                for c in range(N_COLS):
                    yc_h[c].drain(y_buf, tap=TensorAccessPattern(tensor_dims=[ACT_TOTAL], offset=SLOT_FA_STATE + 4 * N_ROWS * X_ELEMS + (c // 2) * (N_ROWS * X_ELEMS) + (c & 1) * 2048, sizes=[32, N_ROWS, Y_CORE_ELEMS], strides=[Y_CORE_ELEMS, X_ELEMS, 1]), wait=True, group=cur_tg)
            else:
                drain_xq_sub(cur_tg, 0, wait_last=False, do_finish=False)
                drain_xq_sub(cur_tg, 1, wait_last=True, do_finish=False)
            cur_tg.finish()
            if layer_mode in (5, 6):
                dst_buf = yn_buf if layer_mode == 6 else y_buf
                dst_tdims = [YN_TOTAL] if layer_mode == 6 else [ACT_TOTAL]
                dst_slot = 0 if layer_mode == 6 else SLOT_HIDDEN_IN
                run_gemm_and_res(W_ATTN_O, 2, dst_buf, dst_tdims, dst_slot, fa_perm=True)

        if layer_mode not in (0, 5):
            return

        run_rms_to_xq(SLOT_NORM_1, SLOT_HIDDEN_IN, B_CORE * H)

        ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
        ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
        ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

        tg_ffn_drain = None
        for kc in range(8):
            xq_base = SLOT_FFN_XQ + kc * (N_ROWS * X_ELEMS)
            tgp = TaskGroup()
            for kh in range(2):
                fill_x_xq(tgp, kh, wait=(kh == 1))
                for base_off in (ffn_g_base, ffn_u_base):
                    do_wait = (kh == 1 and base_off == ffn_u_base)
                    for c in range(N_COLS):
                        tile0_lo = kc * 64 + c * 4
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=base_off + (tile0_lo * 2 + kh) * W_ELEMS,
                                sizes=[2, 4, 9, 256],
                                strides=[64 * W_ELEMS, 2 * W_ELEMS, 256, 1],
                            ),
                            wait=do_wait,
                            group=tgp,
                        )
            if tg_ffn_drain is not None:
                tg_ffn_drain.finish()
                tg_ffn_drain = None
            tgp.finish()
            tg_ffn_drain = TaskGroup()
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=xq_base + c * (2 * Y_CORE_ELEMS),
                        sizes=[4, 2, N_ROWS, Y_CORE_ELEMS],
                        strides=[1024, Y_CORE_ELEMS, X_ELEMS, 1],
                    ),
                    wait=False,
                    group=tg_ffn_drain,
                )
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=xq_base + 4096 + c * Y_CORE_ELEMS,
                        sizes=[N_ROWS, Y_CORE_ELEMS],
                        strides=[X_ELEMS, 1],
                    ),
                    wait=True,
                    group=tg_ffn_drain,
                )

        run_gemm_and_res(ffn_d_base, 8, yn_buf, [YN_TOTAL], 0, x_slot=SLOT_FFN_XQ, prev_drain=tg_ffn_drain)

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *x_prods, *y_cons])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=0, choices=VALID_MODES)
    parser.add_argument("--kv-chunks", type=int, default=0)
    parser.add_argument("--param-slot", type=int, default=0, choices=VALID_PARAM_SLOTS)
    opts = parser.parse_args()
    _validate_compile_params(opts.layer_mode, opts.kv_chunks, opts.param_slot)
    run_design_cli(
        q4prefill_lfm2,
        opts,
        device=NPU2(),
        compile_kwargs=lambda o: {
            "layer_mode": o.layer_mode,
            "kv_chunks": o.kv_chunks,
            "param_slot": o.param_slot,
        },
    )


if __name__ == "__main__":
    main()

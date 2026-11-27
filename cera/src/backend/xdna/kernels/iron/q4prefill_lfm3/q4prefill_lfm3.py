# Unified Fused B=64 Q4_0 LFM3 Prefill Layer (32 cores: 4 rows x 8 cols, npu2).
# K_HALF=1024, W_L2L1 depth=2, X_ELEMS=4608, Y_CORE_ELEMS=64.
# Modes via g_params[6]:
#   0 = ShortConv (d_conv=6, l_cache=7) + SwiGLU FFN (8192)
#   1 = Causal / SWA GQA (16 Q heads, 4 KV heads, head_dim=128) + QK-norm + NeoX RoPE + Gate + SwiGLU FFN
#   2 = Phase A (QKV + Gate + Q/K Norm + RoPE -> SLOT_FA_Q + kvo_buf + SLOT_ATTN_GATE)
#   3 = Bidir Global Phase B (Reload SLOT_FA_Q + Gate -> FA -> O -> FFN tail)
#   4 = FA Partial (Reload SLOT_FA_Q + SLOT_FA_STATE -> FA -> Drain SLOT_FA_STATE)
#   5 = FA Final (Reload SLOT_FA_Q + SLOT_FA_STATE + Gate -> FA + cur_kv -> O -> FFN tail)

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
N_KV_HEADS = 4
MAX_KV_CHUNKS = 256

# Weight offsets in 2304-float (9216 B) K=1024 half-tile packets
W_CONV_IN_B = 0
W_CONV_IN_C = 256 * W_ELEMS
W_CONV_IN_X = 512 * W_ELEMS
W_CONV_OUT = 768 * W_ELEMS
W_CONV_WT = 1024 * W_ELEMS
W_CONV_FFN_GATE = 1040 * W_ELEMS
W_CONV_FFN_UP = 2000 * W_ELEMS
W_CONV_FFN_DOWN = 2960 * W_ELEMS

W_ATTN_Q = 0
W_ATTN_K = 256 * W_ELEMS
W_ATTN_V = 320 * W_ELEMS
W_ATTN_GATE = 384 * W_ELEMS
W_ATTN_O = 386 * W_ELEMS
W_ATTN_FFN_GATE = 642 * W_ELEMS
W_ATTN_FFN_UP = 1602 * W_ELEMS
W_ATTN_FFN_DOWN = 2562 * W_ELEMS
W_TOTAL = 3988 * W_ELEMS

SLOT_HIDDEN_IN = 0
SLOT_Q = 131072
SLOT_K = 262144
SLOT_V = 294912
SLOT_ATTN_OUT = 327680
SLOT_RES_TMP = 458752
SLOT_CONV_B = 589824
SLOT_CONV_C = 720896
SLOT_CONV_X = 851968
SLOT_FFN_XQ = SLOT_CONV_B  # [589824..737280)
SLOT_FA_Q = 737280         # [737280..811008)
SLOT_FA_STATE = 811008     # [811008..958464)
SLOT_XQ = 983040
SLOT_XQ_K0 = SLOT_XQ
SLOT_XQ_K1 = SLOT_XQ + N_ROWS * X_ELEMS
SLOT_ATTN_GATE = 1019904
SLOT_NORM_0 = 1022208
SLOT_NORM_1 = 1024512
SLOT_NORM_2 = 1026816
SLOT_NORM_3 = 1029120
SLOT_ROPE_PKT_0A = 1031424
SLOT_ROPE_PKT_0B = 1033728
SLOT_ROPE_PKT_1A = 1036032
SLOT_ROPE_PKT_1B = 1038336
ACT_TOTAL = SLOT_ROPE_PKT_1B + W_ELEMS  # 1040640

YN_TOTAL = B * H  # 131072
KV_CHUNK_F32 = 2 * W_ELEMS  # 4608
KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * KV_CHUNK_F32
KV_OUT_TOTAL = 2 * N_KV_HEADS * KV_CHUNK_F32  # 8 * 4608

VALID_MODES = (0, 1, 2, 3, 4, 5)
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
def q4prefill_lfm3(
    w: In,
    y: Out,
    y_next: Out,
    kv: In,
    kv_out: Out,
    layer_mode: CompileTime[int] = 1,
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

    fa_flags = ["-I", str(here), "-I", str(cxx_inc), "-Os", "-fno-unroll-loops", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]
    oz_flags = ["-I", str(here), "-I", str(cxx_inc), "-Oz", "-fno-unroll-loops"]
    gemm_flags = oz_flags

    q4_gemm_fn = ExternalFunction(
        "q4_lfm3_pre_gemm_op",
        source_file=str(here / "q4_lfm3_pre_gemm_op.cc"),
        arg_types=[x_pkt_ty, w_pkt_ty, np.int32, np.int32],
        compile_flags=gemm_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm3_pre_w_op",
        source_file=str(here / "q4_lfm3_pre_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=fa_flags,
    )
    q4_x_fn = ExternalFunction(
        "q4_lfm3_pre_x_op",
        source_file=str(here / "q4_lfm3_pre_x_op.cc"),
        arg_types=[x_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm3_pre_y_op",
        source_file=str(here / "q4prefill_lfm3_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=oz_flags,
    )

    W_L3L2 = [ObjectFifo(w_pkt_ty, depth=2, name=f"W_L3L2_{c}") for c in range(N_COLS)]
    W_L2L1 = [W_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=2, name=f"W_L2L1_{c}") for c in range(N_COLS)]

    X_L3L2 = ObjectFifo(x_l3l2_ty, depth=2, name="X_L3L2_0")
    X_L2L1 = X_L3L2.cons().split(
        offsets=[r * X_ELEMS for r in range(N_ROWS)],
        obj_types=[x_pkt_ty] * N_ROWS,
        names=[f"X_L2L1_0_{r}" for r in range(N_ROWS)],
        depths=[1] * N_ROWS,
    )

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
    M1 = 2
    M2 = 4
    M3 = 8
    M4 = 16
    M5 = 32
    M012 = M0 | M1 | M2
    M12 = M1 | M2
    M135 = M1 | M3 | M5
    M345 = M3 | M4 | M5
    M45 = M4 | M5
    M35 = M3 | M5
    M0135 = M0 | M1 | M3 | M5

    def build_worker_sched(c, r):
        rows = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0, mask=63):
            rows.append((kind, cnt, op, a0, a1, a2, mask))

        def emit_rms_sub(norm_op_kind, norm_slot, skip_norm, f32_out, mask):
            if norm_op_kind >= 0:
                emit(KIND_W, 1, norm_op_kind, c, norm_slot, 0, mask)
            emit(KIND_X, 4, 0, c, 0, skip_norm | (f32_out << 1), mask)
            if f32_out == 0:
                emit(KIND_Y, 8, 0, 0, 0, 0, mask)
                emit(KIND_Y, 1, 0, 8, 0, 0, mask)

        def emit_rms_2sub(norm_op_kind, norm_slot, mask, skip_norm=0):
            emit_rms_sub(norm_op_kind, norm_slot, skip_norm, 0, mask)
            emit_rms_sub(-1, 0, skip_norm, 0, mask)

        def emit_gemm(n_passes, pass_tiles, k_chunks, fa_perm, mask, xd_mode=0):
            for _p in range(n_passes):
                emit(KIND_GEMM, k_chunks, xd_mode, pass_tiles, 0, 0, mask)
                emit(KIND_Y, (pass_tiles // 4) * B_CORE, 1, 0, 0, 0, mask)

        def emit_post_norm_res_2sub(norm_slot, mask):
            for s in range(2):
                emit_rms_sub(4 if s == 0 else -1, norm_slot, 0, 1, mask)
                emit(KIND_W, 4, 6, r, 0, 0, mask)
                emit(KIND_Y, 32, 4, 0, 0, 0, mask)

        # 1. Attn RMSNorm -> SLOT_XQ_K0/K1 (Modes 0, 1, 2)
        emit_rms_2sub(-1, 0, M012)

        # Mode 0: ShortConv (d_conv=6, l_cache=7)
        emit_gemm(2, 8, 2, 0, M0)  # B
        emit_gemm(2, 8, 2, 0, M0)  # C
        emit_gemm(2, 8, 2, 0, M0)  # X
        emit(KIND_W, 1, 3, r, 15, 0, M0)  # WT
        emit(KIND_W, 1, 3, r, 0, 0, M0)   # STATE
        for st_b, st_x, st_c in [(1, 5, 9), (16, 5, 9)]:
            emit(KIND_W, 4, 3, r, st_b, 0, M0)
            emit(KIND_W, 4, 3, r, st_x, 0, M0)
            for s_c in range(st_c, st_c + 4):
                emit(KIND_W, 1, 3, r, s_c, 0, M0)
                emit(KIND_Y, 8, 8, 0, 0, 0, M0)
        emit(KIND_Y, 6, 8, 8, 0, 0, M0)  # drain 6-tap state -> kvo_buf
        emit_rms_2sub(-1, 0, M0, skip_norm=1)
        emit_gemm(2, 8, 2, 0, M0)        # OUT_PROJ

        # Modes 1, 2: Q, K, V, ATTN_GATE + KV RoPE & drain
        emit_gemm(2, 8, 2, 0, M12)       # Q
        emit_gemm(1, 4, 2, 0, M12)       # K
        emit_gemm(1, 4, 2, 0, M12)       # V
        emit(KIND_GEMM, 2, 0, 1, 0, 0, M12)  # ATTN_GATE (1 tile, 2 K-halves)
        emit(KIND_Y, 4, 11, 0, 0, 0, M12 if c == 0 else 0)  # drain gate -> SLOT_ATTN_GATE

        kv_drain_mask = M12 if (c % 2) == 0 else 0
        for sub in range(2):
            emit(KIND_W, 2, 1, r, 0, 0, M12)               # ROPE_PKT_{sub}A/B
            emit(KIND_X, 4, 1, c, 0, 1, M12)               # K norm + RoPE
            emit(KIND_X, 4, 1, c, 0, 2, M12)               # V convert
            emit(KIND_Y, 8, 2, 0, 0, 0, kv_drain_mask)
            emit(KIND_Y, 8, 2, 8, 0, 0, kv_drain_mask)
            emit(KIND_X, 4, 1, c, 0, 0, M12)               # Q{sub} norm + RoPE
            emit(KIND_Y, 16, 7, 0, 0, 0, M2)               # M2: drain Q{sub} -> SLOT_FA_Q
            emit(KIND_W, 1, 2, r * 64 + c, sub, 0, M1)     # M1: ATTN_GATE sub
            emit(KIND_W, 1, 16 + sub * 4 + r, 0, 0, 0, M1) # M1: FA{sub}
            emit(KIND_Y, 9, 3, 0, 0, 0, M1)                # M1: drain FA{sub} Q8 -> SLOT_XQ

        # Modes 3, 4, 5: Reload Q0/Q1 + FA_STATE (M4/M5) + GATE (M3/M5) -> FA -> drain FA_STATE (M4) / Q8 (M3/M5)
        for sub in range(2):
            emit(KIND_X, 2, 1, c, 0, 3, M345)                   # Reload q_bf16
            emit(KIND_X, 4, 1, c, 0, 4, M45)                    # Reload o_acc+row_l+row_m
            emit(KIND_W, 1, 2, r * 64 + c, sub, 0, M35)         # ATTN_GATE sub
            emit(KIND_W, 1, 16 + sub * 4 + r, 0, 0, 0, M345)    # FA
            emit(KIND_Y, 33, 10, 0, 0, 0, M4)                   # M4: drain state
            emit(KIND_Y, 9, 3, 0, 0, 0, M35)                    # M3/M5: drain FA Q8

        emit_gemm(2, 8, 2, 1, M135, xd_mode=0)  # O_PROJ

        # Shared sandwich tail (Modes 0, 1, 3, 5): Post-Attn Norm + Res1 -> FFN Norm -> SwiGLU -> Post-FFN Norm + Res2
        emit_post_norm_res_2sub(1, M0135)  # SLOT_NORM_1 = post_attn_norm
        emit_rms_2sub(4, 2, M0135)         # SLOT_NORM_2 = ffn_norm
        emit(KIND_GEMM, 32, 0, 8, 1, 0, M0135)
        emit_gemm(2, 8, 8, 0, M0135, xd_mode=1)  # FFN_DOWN (8 K_HALF chunks)
        emit_post_norm_res_2sub(3, M0135)  # SLOT_NORM_3 = post_ffn_norm

        packed = [
            (k | (cn << 8) | (op << 16) | (a0 << 24), a1 | (a2 << 8) | (mk << 16))
            for (k, cn, op, a0, a1, a2, mk) in rows
        ]
        return np.array(packed, dtype=np.uint32).view(np.int32).reshape(-1)

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
            idx_kvc = arith.constant(idx_ty, 2050)
            idx_lmode = arith.constant(idx_ty, 2054)
            i32_0 = arith.constant(i32_ty, 0)
            i32_1 = arith.constant(i32_ty, 1)
            i32_2 = arith.constant(i32_ty, 2)
            i32_3 = arith.constant(i32_ty, 3)
            i32_4 = arith.constant(i32_ty, 4)
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
            is_m34 = arith.ori(
                arith.cmpi(arith.CmpIPredicate.eq, lmode, i32_3),
                arith.cmpi(arith.CmpIPredicate.eq, lmode, i32_4),
            )

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
                        is_fa1 = arith.cmpi(arith.CmpIPredicate.sge, op, i32_20)
                        extra_kv = arith.select(is_m34, i32_0, arith.select(is_fa1, i32_2, i32_1))
                        fa_cnt = arith.shli(arith.addi(kv_ch, extra_kv), i32_1)
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
                                is_fused_ffn = arith.cmpi(arith.CmpIPredicate.eq, a1, i32_1)
                                for p in range_(cnt):
                                    kh = arith.andi(i32(p), i32_1)
                                    is_sub = arith.cmpi(arith.CmpIPredicate.ne, p, idx0)
                                    norm_flag = arith.ori(arith.select(is_sub, i32_1, i32_0), arith.shli(op, i32_1))
                                    flag = arith.select(is_fused_ffn, kh, norm_flag)
                                    xp = x_fifo.acquire(1)
                                    for pt in range_(a0):
                                        wp = w_fifo.acquire(1)
                                        q4_gemm_fn(xp, wp, i32(pt), flag)
                                        w_fifo.release(1)
                                    x_fifo.release(1)
                                    do_drain = arith.andi(is_fused_ffn, arith.cmpi(arith.CmpIPredicate.eq, kh, i32_1))
                                    with if_(do_drain, hasElse=False):
                                        j = arith.andi(arith.shrui(i32(p), i32_1), i32_1)
                                        arg0 = arith.shli(j, i32_2)
                                        sb_cnt = arith.index_cast(idx_ty, arith.addi(arith.addi(kh, i32_3), j))
                                        for sb in range_(sb_cnt):
                                            is_xd = arith.cmpi(arith.CmpIPredicate.eq, i32(sb), i32_4)
                                            y_arg = arith.select(is_xd, i32_8, arith.addi(arg0, i32(sb)))
                                            yp = y_fifo.acquire(1)
                                            q4_y_fn(yp, i32_9, y_arg)
                                            y_fifo.release(1)

        return core_body

    workers = [
        Worker(
            make_worker_body(c),
            fn_args=[
                W_L2L1[c].cons(),
                X_L2L1[r].cons(),
                Y_L1L2[c][r].prod(),
                sched_bufs[c][r],
                q4_gemm_fn,
                q4_w_fn,
                q4_x_fn,
                q4_y_fn,
            ],
            tile=Tile(c, 2 + r),
            stack_size=2880,
            while_true=True,
        )
        for c in range(N_COLS)
        for r in range(N_ROWS)
    ]

    w_prods = [W_L3L2[c].prod() for c in range(N_COLS)]
    x_prods = [X_L3L2.prod()]
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, *fifos):
        wp_h = fifos[0:N_COLS]
        xp_h = fifos[N_COLS]
        yc_h = fifos[N_COLS + 1 : N_COLS + 1 + N_COLS]

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
            xp_h.fill(
                src_buf,
                tap=TensorAccessPattern(
                    tensor_dims=tdims, offset=off, sizes=[N_ROWS, 8, 576], strides=[row_stride, 576, 1]
                ),
                wait=wait,
                group=tg,
            )

        def fill_x_xq(tg, kc=0, wait=False, x_slot=SLOT_XQ):
            xq_off = x_slot + kc * (N_ROWS * X_ELEMS)
            xp_h.fill(
                y_buf,
                tap=TensorAccessPattern(
                    tensor_dims=[ACT_TOTAL], offset=xq_off, sizes=[1, N_ROWS * X_ELEMS], strides=[0, 1]
                ),
                wait=wait,
                group=tg,
            )

        def drain_xq_sub(tg, sub, wait_last=True):
            xq_base = sub * 2048
            xd_base = 4096 + sub * 256
            for part0 in (0, 4):
                for c in range(N_COLS):
                    kh_slot = SLOT_XQ_K0 if c < 4 else SLOT_XQ_K1
                    c_kh = c & 3
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=kh_slot + xq_base + c_kh * 512 + part0 * Y_CORE_ELEMS,
                            sizes=[4, N_ROWS, Y_CORE_ELEMS],
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
            tg.finish()

        def drain_y_contig_rows(dst_buf, dst_tdims, dst_off, sub, tg):
            sub_off = sub * (8 * H)
            for tl0 in (0, 4):
                for c in range(N_COLS):
                    yc_h[c].drain(
                        dst_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=dst_tdims,
                            offset=dst_off + sub_off + tl0 * H + c * 256,
                            sizes=[4, 4, N_ROWS, Y_CORE_ELEMS],
                            strides=[H, Y_CORE_ELEMS, B_CORE * H, 1],
                        ),
                        wait=(tl0 == 4),
                        group=tg,
                    )
            tg.finish()

        def drain_gemm_64(tg, out_slot, pass_off, pass_tiles, N_total, wait=True, fa_perm=False):
            col_ch = N_total // N_COLS
            n_p4 = pass_tiles // 4
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

        def run_rms_to_xq(norm_off, x_slot, x_stride):
            for sub in range(2):
                sub_off = x_slot + sub * (B_SUB * H)
                tg = TaskGroup()
                if sub == 0 and norm_off is not None:
                    fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg, wait=True)
                for p in range(4):
                    fill_x_2d(y_buf, [ACT_TOTAL], sub_off + p * (2 * H), x_stride, tg, wait=(p == 3))
                drain_xq_sub(tg, sub)

        def run_post_norm_res(norm_off, res_slot, dest_buf, dest_tdims, dest_slot):
            for sub in range(2):
                tg = TaskGroup()
                if sub == 0:
                    fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg, wait=True)
                sub_x = res_slot + sub * (B_SUB * H)
                for p in range(4):
                    fill_x_2d(y_buf, [ACT_TOTAL], sub_x + p * (2 * H), B_CORE * H, tg, wait=(p == 3))
                sub_h = SLOT_HIDDEN_IN + sub * (B_SUB * H)
                for c in range(N_COLS):
                    wp_h[c].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=sub_h + c * 256,
                            sizes=[N_ROWS, B_SUB, 288],
                            strides=[B_CORE * H, H, 1],
                        ),
                        wait=True,
                        group=tg,
                    )
                drain_y_contig_rows(dest_buf, dest_tdims, dest_slot, sub, tg)

        def run_gemm_2d(w_off, n_passes, pass_tiles, k_chunks, out_slot, N_total, x_slot=SLOT_XQ, fa_perm=False):
            tpc = n_passes * pass_tiles
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
                    if is_last_blk:
                        drain_gemm_64(tg, out_slot, p * pass_tiles * ROWS, pass_tiles, N_total, wait=True, fa_perm=fa_perm)
                    tg.finish()

        # Peeled metadata + param slot broadcast for all modes
        param_off = (SLOT_NORM_0, SLOT_NORM_1, SLOT_NORM_2, SLOT_NORM_3)[param_slot]
        tg0 = TaskGroup()
        fill_w_broadcast(y_buf, [ACT_TOTAL], param_off, tg0, wait=True)
        tg0.finish()

        if layer_mode in (0, 1, 2):
            run_rms_to_xq(None, SLOT_HIDDEN_IN, B_CORE * H)

        if layer_mode == 0:
            for mat_off, out_slot in [(W_CONV_IN_B, SLOT_CONV_B), (W_CONV_IN_C, SLOT_CONV_C), (W_CONV_IN_X, SLOT_CONV_X)]:
                for p in range(2):
                    tg = TaskGroup()
                    for kh in range(2):
                        fill_x_xq(tg, kh, wait=(kh == 1))
                        for c in range(N_COLS):
                            tile0 = c * 16 + p * 8
                            wp_h[c].fill(
                                w_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[W_TOTAL],
                                    offset=mat_off + (tile0 * 2 + kh) * W_ELEMS,
                                    sizes=[8, 9, 256],
                                    strides=[2 * W_ELEMS, 256, 1],
                                ),
                                wait=(kh == 1),
                                group=tg,
                            )
                    drain_gemm_64(tg, out_slot, p * 128, 8, H, wait=True)
                    tg.finish()

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
            fill_w_1d(w_buf, [W_TOTAL], [W_CONV_WT + c * KV_CHUNK_F32 for c in range(N_COLS)], tg_sc_a)
            fill_w_1d(kv_buf, [KV_TOTAL], [c * KV_CHUNK_F32 for c in range(N_COLS)], tg_sc_a)
            fill_conv_4(SLOT_CONV_B, 0, tg_sc_a)
            fill_conv_4(SLOT_CONV_X, 0, tg_sc_a, wait=True)
            tg_sc_a.finish()

            tg_sc_b = TaskGroup()
            fill_conv_4(SLOT_CONV_C, 0, tg_sc_b)
            fill_conv_4(SLOT_CONV_B, 4, tg_sc_b)
            fill_conv_4(SLOT_CONV_X, 4, tg_sc_b)
            fill_conv_4(SLOT_CONV_C, 4, tg_sc_b, wait=True)
            for q0 in (0, 4):
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_ATTN_OUT + q0 * (8 * H) + c * 256,
                            sizes=[32, N_ROWS, Y_CORE_ELEMS],
                            strides=[H, Y_CORE_ELEMS, 1],
                        ),
                        wait=False,
                        group=tg_sc_b,
                    )
            for c in range(N_COLS):
                yc_h[c].drain(
                    kvo_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[KV_OUT_TOTAL],
                        offset=c * KV_CHUNK_F32,
                        sizes=[6, N_ROWS, Y_CORE_ELEMS],
                        strides=[256, Y_CORE_ELEMS, 1],
                    ),
                    wait=True,
                    group=tg_sc_b,
                )
            tg_sc_b.finish()

            run_rms_to_xq(None, SLOT_ATTN_OUT, B_CORE * H)
            run_gemm_2d(W_CONV_OUT, 2, 8, 2, SLOT_RES_TMP, H)
        elif layer_mode in (1, 2):
            for p in range(2):
                tg_q = TaskGroup()
                for kh in range(2):
                    fill_x_xq(tg_q, kh, wait=(kh == 1))
                    for c in range(N_COLS):
                        tile0 = c * 16 + p * 8
                        wp_h[c].fill(
                            w_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[W_TOTAL],
                                offset=W_ATTN_Q + (tile0 * 2 + kh) * W_ELEMS,
                                sizes=[8, 9, 256],
                                strides=[2 * W_ELEMS, 256, 1],
                            ),
                            wait=(kh == 1),
                            group=tg_q,
                        )
                drain_gemm_64(tg_q, SLOT_Q, p * 8 * ROWS, 8, Q_DIM, wait=True)
                tg_q.finish()

            tg_kv = TaskGroup()
            for mat_off, out_sl in [(W_ATTN_K, SLOT_K), (W_ATTN_V, SLOT_V)]:
                is_v = mat_off == W_ATTN_V
                for kh in range(2):
                    fill_x_xq(tg_kv, kh, wait=(is_v and kh == 1))
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
                drain_gemm_64(tg_kv, out_sl, 0, 4, KV_DIM, wait=is_v)
            tg_kv.finish()

            tg_ag = TaskGroup()
            for kh in range(2):
                fill_x_xq(tg_ag, kh, wait=(kh == 1))
                fill_w_broadcast(w_buf, [W_TOTAL], W_ATTN_GATE + kh * W_ELEMS, tg_ag, wait=(kh == 1))
            yc_h[0].drain(
                y_buf,
                tap=TensorAccessPattern(
                    tensor_dims=[ACT_TOTAL],
                    offset=SLOT_ATTN_GATE,
                    sizes=[2, 2, N_ROWS, Y_CORE_ELEMS],
                    strides=[128, Y_CORE_ELEMS, 256, 1],
                ),
                wait=True,
                group=tg_ag,
            )
            tg_ag.finish()

            def run_q_and_fa(sub):
                q_off = SLOT_Q + sub * (32 * Q_DIM)
                if layer_mode == 2:
                    tg_rq = TaskGroup()
                    for p in range(4):
                        fill_x_2d(y_buf, [ACT_TOTAL], q_off + p * (2 * Q_DIM), B_SUB * Q_DIM, tg_rq, wait=(p == 3))
                    tg_rq.finish()
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
                    return

                w_tasks = []
                w_tasks.append((y_buf, [ACT_TOTAL], lambda _c: SLOT_ATTN_GATE, [1, W_ELEMS], [0, 1]))
                for chk0 in range(0, kv_chunks, 128):
                    cnt = min(128, kv_chunks - chk0)
                    w_tasks.append((
                        kv_buf,
                        [KV_TOTAL],
                        lambda c, s=chk0: ((c // 2) * MAX_KV_CHUNKS + s) * KV_CHUNK_F32,
                        [cnt, 18, 256],
                        [KV_CHUNK_F32, 256, 1],
                    ))
                w_tasks.append((
                    kvo_buf,
                    [KV_OUT_TOTAL],
                    lambda c: (c // 2) * KV_CHUNK_F32,
                    [1, 18, 256],
                    [KV_CHUNK_F32, 256, 1],
                ))
                if sub == 1:
                    w_tasks.append((
                        kvo_buf,
                        [KV_OUT_TOTAL],
                        lambda c: (N_KV_HEADS + (c // 2)) * KV_CHUNK_F32,
                        [1, 18, 256],
                        [KV_CHUNK_F32, 256, 1],
                    ))

                tg_rq = TaskGroup()
                for p in range(4):
                    fill_x_2d(y_buf, [ACT_TOTAL], q_off + p * (2 * Q_DIM), B_SUB * Q_DIM, tg_rq, wait=(p == 3))
                tg_rq.finish()

                for b0 in range(0, len(w_tasks), 3):
                    b1 = min(b0 + 3, len(w_tasks))
                    is_last_b = b1 == len(w_tasks)
                    cur_tg = TaskGroup()
                    for wi in range(b0, b1):
                        buf, tdims, off_fn, sz, st = w_tasks[wi]
                        do_wait = wi == b1 - 1
                        for c in range(N_COLS):
                            wp_h[c].fill(
                                buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=tdims, offset=off_fn(c), sizes=sz, strides=st
                                ),
                                wait=do_wait,
                                group=cur_tg,
                            )
                    if is_last_b:
                        drain_xq_sub(cur_tg, sub)
                    else:
                        cur_tg.finish()

            for sub in range(2):
                rope_a = SLOT_ROPE_PKT_0A if sub == 0 else SLOT_ROPE_PKT_1A
                rope_b = SLOT_ROPE_PKT_0B if sub == 0 else SLOT_ROPE_PKT_1B
                tg_kv_k = TaskGroup()
                fill_w_broadcast(y_buf, [ACT_TOTAL], rope_a, tg_kv_k)
                fill_w_broadcast(y_buf, [ACT_TOTAL], rope_b, tg_kv_k, wait=True)
                for p in range(4):
                    fill_x_2d(y_buf, [ACT_TOTAL], SLOT_K + sub * 32 * KV_DIM + p * (2 * KV_DIM), B_SUB * KV_DIM, tg_kv_k, wait=(p == 3))
                tg_kv_k.finish()
                tg_kv_v = TaskGroup()
                for p in range(4):
                    fill_x_2d(y_buf, [ACT_TOTAL], SLOT_V + sub * 32 * KV_DIM + p * (2 * KV_DIM), B_SUB * KV_DIM, tg_kv_v, wait=(p == 3))
                for c in (0, 2, 4, 6):
                    kv_h = c >> 1
                    off_kv = (sub * N_KV_HEADS + kv_h) * KV_CHUNK_F32
                    yc_h[c].drain(
                        kvo_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[KV_OUT_TOTAL],
                            offset=off_kv,
                            sizes=[8, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, 512, 1],
                        ),
                        group=tg_kv_v,
                    )
                    yc_h[c].drain(
                        kvo_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[KV_OUT_TOTAL],
                            offset=off_kv + W_ELEMS,
                            sizes=[8, N_ROWS, Y_CORE_ELEMS],
                            strides=[Y_CORE_ELEMS, 512, 1],
                        ),
                        wait=True,
                        group=tg_kv_v,
                    )
                tg_kv_v.finish()
                run_q_and_fa(sub)

            if layer_mode == 1:
                run_gemm_2d(W_ATTN_O, 2, 8, 2, SLOT_RES_TMP, H, fa_perm=True)
        else:
            # Modes 3, 4, 5
            for sub in range(2):
                tg_rq = TaskGroup()
                for half in range(2):
                    xp_h.fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_FA_Q + sub * (2 * N_ROWS * X_ELEMS) + half * (N_ROWS * X_ELEMS),
                            sizes=[1, N_ROWS * X_ELEMS],
                            strides=[0, 1],
                        ),
                        wait=(half == 1),
                        group=tg_rq,
                    )
                tg_rq.finish()

                if layer_mode in (4, 5):
                    tg_fs = TaskGroup()
                    for cp in range(4):
                        xp_h.fill(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_FA_STATE + sub * (4 * N_ROWS * X_ELEMS) + cp * (N_ROWS * X_ELEMS),
                                sizes=[1, N_ROWS * X_ELEMS],
                                strides=[0, 1],
                            ),
                            wait=(cp == 3),
                            group=tg_fs,
                        )
                    tg_fs.finish()

                w_tasks = []
                if layer_mode in (3, 5):
                    w_tasks.append((y_buf, [ACT_TOTAL], lambda _c: SLOT_ATTN_GATE, [1, W_ELEMS], [0, 1]))
                for chk0 in range(0, kv_chunks, 128):
                    cnt = min(128, kv_chunks - chk0)
                    w_tasks.append((
                        kv_buf,
                        [KV_TOTAL],
                        lambda c, s=chk0: ((c // 2) * MAX_KV_CHUNKS + s) * KV_CHUNK_F32,
                        [cnt, 18, 256],
                        [KV_CHUNK_F32, 256, 1],
                    ))
                if layer_mode == 5:
                    w_tasks.append((
                        kvo_buf,
                        [KV_OUT_TOTAL],
                        lambda c: (c // 2) * KV_CHUNK_F32,
                        [1, 18, 256],
                        [KV_CHUNK_F32, 256, 1],
                    ))
                    if sub == 1:
                        w_tasks.append((
                            kvo_buf,
                            [KV_OUT_TOTAL],
                            lambda c: (N_KV_HEADS + (c // 2)) * KV_CHUNK_F32,
                            [1, 18, 256],
                            [KV_CHUNK_F32, 256, 1],
                        ))

                for b0 in range(0, len(w_tasks), 3):
                    b1 = min(b0 + 3, len(w_tasks))
                    is_last_b = b1 == len(w_tasks)
                    cur_tg = TaskGroup()
                    for wi in range(b0, b1):
                        buf, tdims, off_fn, sz, st = w_tasks[wi]
                        do_wait = wi == b1 - 1
                        for c in range(N_COLS):
                            wp_h[c].fill(
                                buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=tdims, offset=off_fn(c), sizes=sz, strides=st
                                ),
                                wait=do_wait,
                                group=cur_tg,
                            )
                    if is_last_b:
                        if layer_mode == 4:
                            for c in range(N_COLS):
                                yc_h[c].drain(
                                    y_buf,
                                    tap=TensorAccessPattern(
                                        tensor_dims=[ACT_TOTAL],
                                        offset=SLOT_FA_STATE + sub * (4 * N_ROWS * X_ELEMS) + (c // 2) * (N_ROWS * X_ELEMS) + (c & 1) * 2112,
                                        sizes=[33, N_ROWS, Y_CORE_ELEMS],
                                        strides=[Y_CORE_ELEMS, X_ELEMS, 1],
                                    ),
                                    wait=True,
                                    group=cur_tg,
                                )
                            cur_tg.finish()
                        else:
                            drain_xq_sub(cur_tg, sub)
                    else:
                        cur_tg.finish()
            if layer_mode in (3, 5):
                run_gemm_2d(W_ATTN_O, 2, 8, 2, SLOT_RES_TMP, H, fa_perm=True)

        if layer_mode not in (0, 1, 3, 5):
            return

        # Shared tail: Post-Attn Norm + Res1 -> FFN_NORM -> SwiGLU FFN(8192) -> DOWN -> Post-FFN Norm + Res2
        run_post_norm_res(SLOT_NORM_1, SLOT_RES_TMP, y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN)
        run_rms_to_xq(SLOT_NORM_2, SLOT_HIDDEN_IN, B_CORE * H)

        ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
        ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
        ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

        for kc in range(8):
            xq_base = SLOT_FFN_XQ + kc * (N_ROWS * X_ELEMS)
            for j in range(2):
                p = kc * 2 + j
                tgp = TaskGroup()
                for kh in range(2):
                    fill_x_xq(tgp, kh, wait=(kh == 1))
                    for base_off in (ffn_g_base, ffn_u_base):
                        for c in range(N_COLS):
                            tile0 = p * 32 + c * 4
                            is_pad_p = p == 15
                            wp_h[c].fill(
                                w_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[W_TOTAL],
                                    offset=(3984 * W_ELEMS) if is_pad_p else (base_off + (tile0 * 2 + kh) * W_ELEMS),
                                    sizes=[4, 9, 256],
                                    strides=[(W_ELEMS if is_pad_p else (2 * W_ELEMS)), 256, 1],
                                ),
                                wait=(kh == 1) and (base_off == ffn_u_base),
                                group=tgp,
                            )
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=xq_base + j * 1024 + c * (2 * Y_CORE_ELEMS),
                            sizes=[2, 2, N_ROWS, Y_CORE_ELEMS],
                            strides=[2048, Y_CORE_ELEMS, X_ELEMS, 1],
                        ),
                        wait=(j == 0),
                        group=tgp,
                    )
                if j == 1:
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=xq_base + 4096 + c * Y_CORE_ELEMS,
                                sizes=[N_ROWS, Y_CORE_ELEMS],
                                strides=[X_ELEMS, 1],
                            ),
                            wait=True,
                            group=tgp,
                        )
                tgp.finish()

        run_gemm_2d(ffn_d_base, 2, 8, 8, SLOT_RES_TMP, H, x_slot=SLOT_FFN_XQ)
        run_post_norm_res(SLOT_NORM_3, SLOT_RES_TMP, yn_buf, [YN_TOTAL], 0)

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *x_prods, *y_cons])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=1, choices=VALID_MODES)
    parser.add_argument("--kv-chunks", type=int, default=0)
    parser.add_argument("--param-slot", type=int, default=0, choices=VALID_PARAM_SLOTS)
    opts = parser.parse_args()
    _validate_compile_params(opts.layer_mode, opts.kv_chunks, opts.param_slot)
    run_design_cli(
        q4prefill_lfm3,
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

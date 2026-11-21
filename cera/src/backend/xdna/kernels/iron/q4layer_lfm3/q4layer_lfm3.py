# Fused B=1 Q4_0 LFM3 decode layer (32 cores: 4 rows x 8 cols, npu2).
# Modes via g_params[6]:
#   0 = ShortConv + FFN decode (using prefill Conv weights + conv_state)
#   1 = GQA + SWA/Causal Attention + FFN decode (using prefill Attn weights + kv_buf)
#   2 = Attn Phase A (RMS + QKV + Gate + RoPE -> kvo_buf + SLOT_DEC_Q_BF16 + SLOT_ATTN_GATE)
#   3 = Attn FA Partial (Reload SLOT_DEC_Q_BF16 + SLOT_DEC_FA_STATE -> FA -> Drain SLOT_DEC_FA_STATE)
#   4 = Attn FA Final (Reload SLOT_HIDDEN_IN + Gate + Q_BF16 + FA_STATE -> FA + cur_kv -> O + FFN)

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
ROWS = 16
CHUNK_K = 2048

W_ELEMS = 4608
X_ELEMS = 640
Y_CORE_ELEMS = 64
Y_COL_ELEMS = N_ROWS * Y_CORE_ELEMS  # 256

H = 2048
Q_DIM = 2048
KV_DIM = 512
N_KV_HEADS = 4
MAX_KV_CHUNKS = 256

# Weight offsets matching q4prefill_lfm3_conv and q4prefill_lfm3_attn identically.
W_CONV_IN_B = 0
W_CONV_IN_C = 128 * W_ELEMS
W_CONV_IN_X = 256 * W_ELEMS
W_CONV_OUT = 384 * W_ELEMS
W_CONV_WT = 512 * W_ELEMS
W_CONV_FFN_GATE = 520 * W_ELEMS
W_CONV_FFN_UP = 1000 * W_ELEMS
W_CONV_FFN_DOWN = 1480 * W_ELEMS

W_ATTN_Q = 0
W_ATTN_K = 128 * W_ELEMS
W_ATTN_V = 160 * W_ELEMS
W_ATTN_GATE = 192 * W_ELEMS
W_ATTN_O = 193 * W_ELEMS
W_ATTN_FFN_GATE = 321 * W_ELEMS
W_ATTN_FFN_UP = 801 * W_ELEMS
W_ATTN_FFN_DOWN = 1281 * W_ELEMS

W_TOTAL = 1992 * W_ELEMS

# Activation buffer slots (128+ float safety padding after each 2048-float source
# so X_ELEMS=640 sequential chunk reads remain in-bounds).
SLOT_HIDDEN_IN = 0
SLOT_H1 = 2560
SLOT_RES_TMP = 5120
SLOT_XQ = 7680
SLOT_FFN_XQ = 10240          # 4 * 2560 = 10240 floats (4 K-chunks of 2048 dims)
SLOT_ATTN_OUT = 20480        # 4096 floats ([8 cols][2 parts][4 rows][64])
SLOT_Q = 24576
SLOT_K = 27136
SLOT_V = SLOT_K + 512
SLOT_ATTN_GATE = SLOT_K + 1024  # 28160; 28160 + 4608 = 32768 == SLOT_SW
SLOT_SW = 32768              # 8192 + 512 pad
SLOT_NORM_0 = 41472          # 4608 floats (norm_w[2048], post_norm_w[2048], params[16])
SLOT_NORM_1 = 46080          # 4608 floats (ffn_norm_w[2048], post_ffn_norm_w[2048])
SLOT_ROPE_PKT = 50688        # 4608 floats (rope_cs[128], ..., q_norm[4096], k_norm[4224])
SLOT_DEC_FIN_NORM = 55296    # 4608 floats (post_norm_w[2048..4095], params_fin[4096..4111])
ACT_TOTAL = SLOT_DEC_FIN_NORM + 4608

# Aliases inside bo_y for Conv B/C/X (2560 stride each inside SLOT_SW space)
SLOT_CONV_B = SLOT_SW
SLOT_CONV_C = SLOT_SW + 2560
SLOT_CONV_X = SLOT_SW + 5120
SLOT_CONV_OUT = SLOT_Q       # reuse 2560-float scratch cleanly
SLOT_DEC_Q_BF16 = SLOT_SW             # 2048 floats (8 cols * 256)
SLOT_DEC_FA_STATE = SLOT_SW + 2048    # 6144 floats (8 cols * 3 pkts * 256) -> ends at 40960 < 41472

KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * W_ELEMS
KV_OUT_TOTAL = 8 * W_ELEMS


@iron.jit(use_cache=False)
def q4layer_lfm3(
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
    yn_ty = np.ndarray[(2048,), np.dtype[np.float32]]
    kv_ty = np.ndarray[(KV_TOTAL,), np.dtype[np.float32]]
    kvo_ty = np.ndarray[(KV_OUT_TOTAL,), np.dtype[np.float32]]

    w_pkt_ty = np.ndarray[(W_ELEMS,), np.dtype[np.float32]]
    x_pkt_ty = np.ndarray[(X_ELEMS,), np.dtype[np.float32]]
    y_core_ty = np.ndarray[(Y_CORE_ELEMS,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_ELEMS,), np.dtype[np.float32]]

    common_flags = ["-I", str(here), "-I", str(cxx_inc)]
    oz_flags = common_flags + ["-Oz"]
    os_flags = common_flags + ["-Os", "-fno-unroll-loops"]
    o3_flags = common_flags + ["-O3"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm3_dec_gemm_op",
        source_file=str(here / "q4_lfm3_dec_gemm_op.cc"),
        arg_types=[x_pkt_ty, w_pkt_ty, np.int32, np.int32],
        compile_flags=o3_flags,
    )
    q4_fa_fn = ExternalFunction(
        "q4_lfm3_dec_fa_op",
        source_file=str(here / "q4_lfm3_dec_fa_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=os_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm3_dec_w_op",
        source_file=str(here / "q4_lfm3_dec_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_x_fn = ExternalFunction(
        "q4_lfm3_dec_x_op",
        source_file=str(here / "q4_lfm3_dec_x_op.cc"),
        arg_types=[x_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm3_dec_y_op",
        source_file=str(here / "q4layer_lfm3_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=oz_flags,
    )

    W_L3L2 = [ObjectFifo(w_pkt_ty, depth=4, name=f"W_L3L2_{c}") for c in range(N_COLS)]
    W_L2L1 = [W_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=2, name=f"W_L2L1_{c}") for c in range(N_COLS)]

    X_L3L2 = [ObjectFifo(x_pkt_ty, depth=2, name=f"X_L3L2_{c}") for c in range(N_COLS)]
    X_L2L1 = [X_L3L2[c].cons().forward(obj_type=x_pkt_ty, depth=2, name=f"X_L2L1_{c}") for c in range(N_COLS)]

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

    M0 = 1   # layer_mode == 0 (Conv)
    M1 = 2   # layer_mode == 1 (Attn)
    M2 = 4   # layer_mode == 2 (Attn Phase A)
    M3 = 8   # layer_mode == 3 (Attn FA Partial)
    M4 = 16  # layer_mode == 4 (Attn FA Final)
    M01 = M0 | M1
    M012 = M0 | M1 | M2
    M014 = M0 | M1 | M4
    M12 = M1 | M2
    M14 = M1 | M4
    M34 = M3 | M4
    M134 = M1 | M3 | M4

    def build_worker_sched(c, r):
        core_id = c * 4 + r
        rows = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0, mask=M01):
            rows.append((kind, cnt, op, a0, a1, a2, mask))

        def emit_rms(norm_slot, load_params, flags_x, mask):
            if norm_slot >= 0:
                emit(KIND_W, 1, 0, core_id, load_params, 0, mask)
            emit(KIND_X, 4, 0, core_id, 0, flags_x, mask)
            emit(KIND_Y, 1, 0, 0, 0, 0, mask)

        def emit_post_res(update_res, mask):
            emit(KIND_X, 4, 1, core_id, 0, update_res, mask)
            emit(KIND_Y, 1, 2, 0, 0, 0, mask)

        def emit_gemm(n_passes, k_chunks, dst_gate, do_drain, mask):
            # a0 = n_passes, a1 = k_chunks, a2 = dst_gate | (do_drain << 1) | (r << 2)
            emit(KIND_GEMM, n_passes, 0, n_passes, k_chunks, dst_gate | (do_drain << 1) | (r << 2), mask)

        # Peeled step 0 already loaded SLOT_NORM_0 + g_params into every worker.
        # Stage 1: Initial RMSNorm + Q8 (save_res=1)
        emit_rms(-1, 0, 1, M012)

        # Mode 0 (Conv): B, C, X share SLOT_XQ -> in-core Conv -> Q8(conv_y) -> SLOT_XQ -> out_proj
        emit(KIND_GEMM, 4, 3, 4, 1, 0, M0)   # B (->ffn_gate), C (->conv_c), X (->y_gemm)
        emit(KIND_W, 1, 3, r, 0, 0, M0)       # W_CONV_WT
        emit(KIND_W, 1, 4, r, 0, 0, M0)       # conv_state_in
        emit(KIND_Y, 1, 9, 0, 0, 0, M0)       # In-core Conv + Q8(conv_y) & drain -> SLOT_XQ
        emit(KIND_Y, 6, 3, 0, 0, 0, M0)       # Drain conv_state[0..5] -> kvo_buf
        emit_gemm(4, 1, 0, 1, M0)             # OUT_PROJ

        # Mode 1 & 2 Stage A (Attn): Q, stacked (K, V, ATTN_GATE) -> Q/K Norm + RoPE
        emit_gemm(4, 1, 0, 1, M12)       # Q (2048)
        emit_gemm(3, 1, 0, 1, M12)       # p0=K, p1=V, p2=ATTN_GATE -> SLOT_K[0, +512, +1024]
        emit(KIND_W, 1, 1, 0, 0, 0, M12)             # SLOT_ROPE_PKT
        emit(KIND_W, 1, 2, c * 2 + r, 0, 0, M12)     # SLOT_ATTN_GATE
        emit(KIND_X, 4, 2, core_id, 0, 0, M12)       # Q norm + RoPE (r < 2)
        emit(KIND_X, 2, 2, core_id, 0, 1, M12)       # K norm + RoPE (r == 2) & V bf16 (r == 3)
        emit(KIND_Y, 1, 5, 0, 0, 0, M12)             # Drain K/V cur -> kvo_buf
        emit(KIND_Y, 1, 5, 0, 0, 0, M2)              # Mode 2: Drain q_bf16 -> SLOT_DEC_Q_BF16
        emit(KIND_Y, 3, 8, 0, 0, 0, M2)              # Mode 2: Reset+drain fa_state -> SLOT_DEC_FA_STATE

        # Modes 3 & 4 Stage B: Reload res_acc (M4), gate (M4), q_bf16 & fa_state (M34)
        emit(KIND_X, 4, 4, core_id, 0, 0, M4)        # Load res_acc from SLOT_HIDDEN_IN
        emit(KIND_W, 1, 2, c * 2 + r, 0, 0, M4)      # Load gate_sig from SLOT_ATTN_GATE
        emit(KIND_X, 1, 5, core_id, 0, 0, M34)       # Load q_bf16
        emit(KIND_X, 2, 6, core_id, 0, 0, M34)       # Load fa_state (o_acc, row_m, row_l)

        # Modes 1, 3, 4: FA over kv_chunks (+ cur_kv for M14)
        emit(KIND_FA, 1, r, 0, 0, 0, M134)
        emit(KIND_Y, 3, 7, 0, 0, 0, M3)              # Mode 3: Drain fa_state -> SLOT_DEC_FA_STATE
        emit(KIND_Y, 2, 4, 0, 0, 0, M14)             # Gate + Q8(o_acc) -> SLOT_ATTN_OUT
        emit_gemm(4, 1, 0, 1, M14)                   # O_PROJ

        # Shared tail (Modes 0, 1, 4): post_attn_norm + res -> ffn_norm -> FFN -> post_ffn_norm + res
        emit_post_res(1, M014)                      # post_attn_norm + res -> SLOT_H1 + update res_acc
        emit_rms(1, 0, 0, M014)                      # SLOT_NORM_1 -> ffn_norm + Q8(SLOT_H1)

        # FFN: GATE & UP share SLOT_XQ in-core; npass=4 chunks reorder per-core contiguous 64-ch tiles for in-core Q8,
        # while the trailing npass=3 chunk (7680 pad boundary) drains SiLU*Up to SLOT_SW + run_rms_to_xq.
        for npass in [4, 4, 4]:
            emit(KIND_GEMM, npass, 1, npass, 1, 0, M014)
        emit(KIND_GEMM, 3, 2, 3, 1, 0, M014)
        emit_rms(-1, 0, 2, M014)

        emit_gemm(4, 4, 0, 1, M014)                 # FFN_DOWN (4 passes x 4 K-chunks) -> SLOT_RES_TMP
        emit_post_res(0, M014)                      # post_ffn_norm + res -> yn_buf
        return np.array(rows, dtype=np.int32).reshape(-1)

    sample_sched = build_worker_sched(0, 0)
    n_steps = len(sample_sched) // 7
    sched_ty = np.ndarray[(len(sample_sched),), np.dtype[np.int32]]
    sched_bufs = [
        [Buffer(sched_ty, initial_value=build_worker_sched(c, r), name=f"sc_{c}_{r}") for r in range(N_ROWS)]
        for c in range(N_COLS)
    ]

    def make_worker_body(c, r):
        core_id = c * 4 + r

        def core_body(w_fifo, x_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_fa_fn, q4_w_fn, q4_x_fn, q4_y_fn):
            idx_ty = IndexType.get()
            i32_ty = IntegerType.get_signless(32)
            idx0 = arith.constant(idx_ty, 0)
            idx1 = arith.constant(idx_ty, 1)
            idx2 = arith.constant(idx_ty, 2)
            idx3 = arith.constant(idx_ty, 3)
            idx4 = arith.constant(idx_ty, 4)
            idx5 = arith.constant(idx_ty, 5)
            idx6 = arith.constant(idx_ty, 6)
            idx7 = arith.constant(idx_ty, 7)
            idx_kvc = arith.constant(idx_ty, 4098)
            idx_lmode = arith.constant(idx_ty, 4102)
            i32_0 = arith.constant(i32_ty, 0)
            i32_1 = arith.constant(i32_ty, 1)
            i32_2 = arith.constant(i32_ty, 2)
            i32_3 = arith.constant(i32_ty, 3)
            i32_4 = arith.constant(i32_ty, 4)
            i32_6 = arith.constant(i32_ty, 6)
            i32_10 = arith.constant(i32_ty, 10)
            my_c = arith.constant(i32_ty, c)
            my_r = arith.constant(i32_ty, r)
            my_cid = arith.constant(i32_ty, core_id)

            def i32(v):
                return arith.index_cast(i32_ty, v)

            wp0 = w_fifo.acquire(1)
            kv_ch = arith.bitcast(i32_ty, wp0[idx_kvc])
            lmode = arith.bitcast(i32_ty, wp0[idx_lmode])
            q4_w_fn(wp0, i32_0, my_cid, i32_1)
            w_fifo.release(1)

            lmode_bit = arith.shli(i32_1, lmode)
            is_m1 = arith.cmpi(arith.CmpIPredicate.eq, lmode, i32_1)
            is_m4 = arith.cmpi(arith.CmpIPredicate.eq, lmode, i32_4)
            use_cur = arith.ori(is_m1, is_m4)
            has_kv = arith.cmpi(arith.CmpIPredicate.sgt, kv_ch, i32_0)

            for s in range_(n_steps):
                base = arith.muli(s, idx7)
                mask = sched_buf[arith.addi(base, idx6)]
                active = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(mask, lmode_bit), i32_0)
                with if_(active, hasElse=False):
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
                                    # GEMM: op=0 normal, op=1 fused Gate+Up+Q8 (contiguous core channels),
                                    # op=2 fused Gate+Up+SiLU*Up float drain (trailing npass=3 chunk),
                                    # op=3 fused Conv B/C/X (contiguous core channels, no drain).
                                    is_bcx = arith.cmpi(arith.CmpIPredicate.eq, op, i32_3)
                                    is_fg_q8 = arith.cmpi(arith.CmpIPredicate.eq, op, i32_1)
                                    is_fg = arith.ori(
                                        is_fg_q8,
                                        arith.cmpi(arith.CmpIPredicate.eq, op, i32_2),
                                    )
                                    n_ghalf = arith.select(is_bcx, idx3, arith.select(is_fg, idx2, idx1))
                                    dst_gate_normal = arith.andi(a2, i32_1)
                                    do_drain_normal = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_2), i32_0)
                                    do_drain = arith.ori(is_fg, do_drain_normal)
                                    y_op_drain = arith.select(is_fg, arith.select(is_fg_q8, i32_6, i32_10), i32_1)
                                    y_arg_drain = arith.select(is_fg, a0, i32_0)
                                    last_k = arith.subi(a1, i32_1)
                                    for kc in range_(a1):
                                        is_sub = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
                                        add_bit = arith.select(is_sub, i32_1, i32_0)
                                        is_last = arith.cmpi(arith.CmpIPredicate.eq, i32(kc), last_k)
                                        xp = x_fifo.acquire(1)
                                        for gh in range_(n_ghalf):
                                            is_gh0 = arith.cmpi(arith.CmpIPredicate.eq, gh, idx0)
                                            is_gh1 = arith.cmpi(arith.CmpIPredicate.eq, gh, idx1)
                                            gh_fg = arith.select(is_gh0, i32_1, i32_0)
                                            gh_bcx = arith.select(is_gh0, i32_1, arith.select(is_gh1, i32_2, i32_0))
                                            gh_dst = arith.select(is_bcx, gh_bcx, arith.select(is_fg, gh_fg, dst_gate_normal))
                                            gflags = arith.ori(add_bit, arith.shli(gh_dst, i32_1))
                                            for p in range_(a0):
                                                for pt in range_(4):
                                                    wp = w_fifo.acquire(1)
                                                    is_mine = arith.cmpi(arith.CmpIPredicate.eq, i32(pt), my_r)
                                                    with if_(is_mine, hasElse=False):
                                                        q4_gemm_fn(xp, wp, i32(p), gflags)
                                                    w_fifo.release(1)
                                        x_fifo.release(1)
                                        with if_(arith.andi(is_last, do_drain), hasElse=False):
                                            yp = y_fifo.acquire(1)
                                            q4_y_fn(yp, y_op_drain, y_arg_drain)
                                            y_fifo.release(1)
                                with else_(if_g):
                                    # KIND_FA
                                    for kc in range_(kv_ch):
                                        is_f = arith.andi(
                                            is_m1,
                                            arith.cmpi(arith.CmpIPredicate.eq, kc, idx0),
                                        )
                                        wp = w_fifo.acquire(1)
                                        q4_fa_fn(wp, my_r, i32(kc), arith.select(is_f, i32_2, i32_0))
                                        w_fifo.release(1)
                                    with if_(use_cur, hasElse=False):
                                        wp_c = w_fifo.acquire(1)
                                        cur_no_kv = arith.select(is_m1, i32_3, i32_1)
                                        q4_fa_fn(wp_c, my_r, i32_0, arith.select(has_kv, i32_1, cur_no_kv))
                                        w_fifo.release(1)

        return core_body

    workers = [
        Worker(
            make_worker_body(c, r),
            fn_args=[
                W_L2L1[c].cons(),
                X_L2L1[c].cons(),
                Y_L1L2[c][r].prod(),
                sched_bufs[c][r],
                q4_gemm_fn,
                q4_fa_fn,
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
    x_prods = [X_L3L2[c].prod() for c in range(N_COLS)]
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, *fifos):
        wp_h = fifos[0:N_COLS]
        xp_h = fifos[N_COLS : 2 * N_COLS]
        yc_h = fifos[2 * N_COLS : 3 * N_COLS]

        def fill_w_tap(src_buf, tdims, off_per_col, sizes, strides, tg, wait=False):
            for c in range(N_COLS):
                wp_h[c].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off_per_col[c], sizes=sizes, strides=strides
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_w_1d(src_buf, tdims, off_per_col, slen, tg, wait=False):
            fill_w_tap(src_buf, tdims, off_per_col, [1, slen], [0, 1], tg, wait=wait)

        def fill_w_broadcast(src_buf, tdims, off, tg, wait=False):
            fill_w_1d(src_buf, tdims, [off] * N_COLS, W_ELEMS, tg, wait=wait)

        def fill_x_broadcast(src_buf, tdims, off, tg, wait=False):
            for c in range(N_COLS):
                xp_h[c].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off, sizes=[1, X_ELEMS], strides=[0, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_x_xq(slot_off, tg, wait=False):
            for c in range(N_COLS):
                xp_h[c].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL], offset=slot_off, sizes=[32, 20], strides=[64, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def drain_linear(dst_buf, tdims, dst_off, tg):
            for c in range(N_COLS):
                yc_h[c].drain(
                    dst_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=dst_off + c * 256, sizes=[1, 256], strides=[0, 1]
                    ),
                    wait=True,
                    group=tg,
                )

        def run_rms_to_xq(norm_off, x_slot, xq_slot, has_attn_tap=False):
            tg = TaskGroup()
            if norm_off is not None:
                fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg, wait=True)
            if has_attn_tap:
                # Never used now that attn_out quantizes directly in L1.
                pass
            for k in range(3):
                fill_x_broadcast(y_buf, [ACT_TOTAL], x_slot + k * 512, tg, wait=(k == 2))
            tg.finish()
            tg2 = TaskGroup()
            fill_x_broadcast(y_buf, [ACT_TOTAL], x_slot + 3 * 512, tg2)
            drain_linear(y_buf, [ACT_TOTAL], xq_slot, tg2)
            tg2.finish()

        def run_post_norm_and_res(x_slot, dst_buf, dst_tdims, dst_off):
            tg = TaskGroup()
            for k in range(3):
                fill_x_broadcast(y_buf, [ACT_TOTAL], x_slot + k * 512, tg, wait=(k == 2))
            tg.finish()
            tg2 = TaskGroup()
            fill_x_broadcast(y_buf, [ACT_TOTAL], x_slot + 3 * 512, tg2)
            drain_linear(dst_buf, dst_tdims, dst_off, tg2)
            tg2.finish()

        def run_gemm_passes(w_base, n_passes, k_chunks, xq_base, out_slot, do_drain=True, out_col_only0=False):
            for kc in range(k_chunks):
                is_last_k = kc == k_chunks - 1
                is_last_step = is_last_k and do_drain
                tg = TaskGroup()
                fill_x_xq(xq_base + kc * 2560, tg)
                for p in range(n_passes):
                    is_last_p = p == n_passes - 1
                    w_wait = is_last_p and ((not is_last_step) or out_col_only0)
                    if k_chunks == 1:
                        fill_w_1d(
                            w_buf,
                            [W_TOTAL],
                            [w_base + (p * 32 + c * 4) * W_ELEMS for c in range(N_COLS)],
                            4 * W_ELEMS,
                            tg,
                            wait=w_wait,
                        )
                    else:
                        fill_w_tap(
                            w_buf,
                            [W_TOTAL],
                            [w_base + ((p * 32 + c * 4) * k_chunks + kc) * W_ELEMS for c in range(N_COLS)],
                            [4, 18, 256],
                            [k_chunks * W_ELEMS, 256, 1],
                            tg,
                            wait=w_wait,
                        )
                if is_last_step:
                    if out_col_only0:
                        yc_h[0].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL], offset=out_slot, sizes=[1, 256], strides=[0, 1]
                            ),
                            wait=True,
                            group=tg,
                        )
                    else:
                        for c in range(N_COLS):
                            yc_h[c].drain(
                                y_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[ACT_TOTAL],
                                    offset=out_slot + c * 64,
                                    sizes=[4, 4, 16],
                                    strides=[16, 512, 1],
                                ),
                                wait=True,
                                group=tg,
                            )
                tg.finish()

        # Peeled step 0 + initial RMSNorm+Q8 (Modes 0, 1, 2 only; Modes 3 & 4 only need SLOT_NORM_0 + g_dec.params).
        if layer_mode in (0, 1, 2):
            tg0 = TaskGroup()
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_NORM_0, tg0, wait=True)
            for k in range(3):
                fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + k * 512, tg0, wait=(k == 2))
            tg0.finish()
            tg0b = TaskGroup()
            fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + 3 * 512, tg0b)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg0b)
            tg0b.finish()
        else:
            norm_slot = SLOT_NORM_1 if layer_mode == 3 else SLOT_DEC_FIN_NORM
            tg0 = TaskGroup()
            fill_w_broadcast(y_buf, [ACT_TOTAL], norm_slot, tg0, wait=True)
            tg0.finish()

        if layer_mode == 0:
            tgb = TaskGroup()
            fill_x_xq(SLOT_XQ, tgb)
            for p in range(4):
                fill_w_tap(
                    w_buf,
                    [W_TOTAL],
                    [W_CONV_IN_B + (c * 16 + p) * W_ELEMS for c in range(N_COLS)],
                    [4, 18, 256],
                    [4 * W_ELEMS, 256, 1],
                    tgb,
                    wait=(p == 3),
                )
            tgb.finish()

            tgc_m = TaskGroup()
            for p in range(4):
                fill_w_tap(
                    w_buf,
                    [W_TOTAL],
                    [W_CONV_IN_C + (c * 16 + p) * W_ELEMS for c in range(N_COLS)],
                    [4, 18, 256],
                    [4 * W_ELEMS, 256, 1],
                    tgc_m,
                    wait=(p == 3),
                )
            tgc_m.finish()

            tgx_m = TaskGroup()
            for p in range(4):
                fill_w_tap(
                    w_buf,
                    [W_TOTAL],
                    [W_CONV_IN_X + (c * 16 + p) * W_ELEMS for c in range(N_COLS)],
                    [4, 18, 256],
                    [4 * W_ELEMS, 256, 1],
                    tgx_m,
                    wait=(p == 3),
                )
            tgx_m.finish()

            tgc = TaskGroup()
            fill_w_1d(w_buf, [W_TOTAL], [W_CONV_WT + c * W_ELEMS for c in range(N_COLS)], W_ELEMS, tgc)
            fill_w_1d(kv_buf, [KV_TOTAL], [c * W_ELEMS for c in range(N_COLS)], W_ELEMS, tgc)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tgc)
            tgc.finish()

            # Drain rolling state (6 taps x 256 floats/col -> kvo_buf)
            for k in range(0, 6, 3):
                tgs = TaskGroup()
                for dk in range(3):
                    kk = k + dk
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            kvo_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[KV_OUT_TOTAL],
                                offset=c * W_ELEMS + kk * 256,
                                sizes=[1, 256],
                                strides=[0, 1],
                            ),
                            wait=(dk == 2),
                            group=tgs,
                        )
                tgs.finish()

            run_gemm_passes(W_CONV_OUT, 4, 1, SLOT_XQ, SLOT_RES_TMP)
        elif layer_mode in (1, 2):
            run_gemm_passes(W_ATTN_Q, 4, 1, SLOT_XQ, SLOT_Q)

            tg_kvg = TaskGroup()
            fill_x_xq(SLOT_XQ, tg_kvg)
            for w_base in [W_ATTN_K, W_ATTN_V, W_ATTN_GATE]:
                fill_w_1d(
                    w_buf,
                    [W_TOTAL],
                    [w_base + (c * 4) * W_ELEMS for c in range(N_COLS)],
                    4 * W_ELEMS,
                    tg_kvg,
                )
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_K + c * 64,
                        sizes=[4, 4, 16],
                        strides=[16, 512, 1],
                    ),
                    wait=True,
                    group=tg_kvg,
                )
            tg_kvg.finish()

            # RoPE + Gate weights + Q norm/RoPE
            tgr = TaskGroup()
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT, tgr)
            fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ATTN_GATE, tgr, wait=True)
            tgr.finish()

            tgr_q = TaskGroup()
            for k in range(3):
                fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_Q + k * 512, tgr_q, wait=(k == 2))
            tgr_q.finish()

            tgr_kv = TaskGroup()
            fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_Q + 3 * 512, tgr_kv)
            fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_K, tgr_kv)
            fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_V, tgr_kv)
            drain_linear(kvo_buf, [KV_OUT_TOTAL], 0, tgr_kv)
            tgr_kv.finish()

            if layer_mode == 2:
                tgd = TaskGroup()
                drain_linear(y_buf, [ACT_TOTAL], SLOT_DEC_Q_BF16, tgd)
                tgd.finish()
                tgd_fs = TaskGroup()
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_DEC_FA_STATE + c * 768,
                            sizes=[3, 256],
                            strides=[256, 1],
                        ),
                        wait=True,
                        group=tgd_fs,
                    )
                tgd_fs.finish()
            else:
                # FA over past kv_chunks + cur_kv -> quantized attn_out drained to SLOT_ATTN_OUT
                w_tasks = []
                for chk0 in range(0, kv_chunks, 128):
                    cnt = min(128, kv_chunks - chk0)
                    w_tasks.append((kv_buf, [KV_TOTAL], lambda c, s=chk0: ((c // 2) * MAX_KV_CHUNKS + s) * W_ELEMS, cnt * W_ELEMS))
                w_tasks.append((kvo_buf, [KV_OUT_TOTAL], lambda c: (c // 2) * 512, W_ELEMS))

                for b0 in range(0, len(w_tasks), 3):
                    b1 = min(b0 + 3, len(w_tasks))
                    is_last_b = b1 == len(w_tasks)
                    tgf = TaskGroup()
                    for wi in range(b0, b1):
                        buf, tdims, off_fn, slen = w_tasks[wi]
                        do_wait = (not is_last_b) and (wi == b1 - 1)
                        for c in range(N_COLS):
                            wp_h[c].fill(
                                buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=tdims, offset=off_fn(c), sizes=[1, slen], strides=[0, 1]
                                ),
                                wait=do_wait,
                                group=tgf,
                            )
                    if is_last_b:
                        for part in range(2):
                            for c in range(N_COLS):
                                yc_h[c].drain(
                                    y_buf,
                                    tap=TensorAccessPattern(
                                        tensor_dims=[ACT_TOTAL],
                                        offset=SLOT_ATTN_OUT + c * 256 + part * 64,
                                        sizes=[2, 2, 64],
                                        strides=[2048, 128, 1],
                                    ),
                                    wait=(part == 1),
                                    group=tgf,
                                )
                    tgf.finish()

                # O_PROJ reading from [32 x 64]-strided SLOT_ATTN_OUT via fill_x_xq
                run_gemm_passes(W_ATTN_O, 4, 1, SLOT_ATTN_OUT, SLOT_RES_TMP)
        else:
            if layer_mode == 4:
                tgr_in = TaskGroup()
                for k in range(3):
                    fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + k * 512, tgr_in, wait=(k == 2))
                tgr_in.finish()
                tgr_in2 = TaskGroup()
                fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + 3 * 512, tgr_in2)
                fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ATTN_GATE, tgr_in2, wait=True)
                tgr_in2.finish()

            tg_fs = TaskGroup()
            for c in range(N_COLS):
                xp_h[c].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL], offset=SLOT_DEC_Q_BF16 + c * 256, sizes=[1, X_ELEMS], strides=[0, 1]
                    ),
                    wait=True,
                    group=tg_fs,
                )
            for c in range(N_COLS):
                xp_h[c].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL], offset=SLOT_DEC_FA_STATE + c * 768, sizes=[1, X_ELEMS], strides=[0, 1]
                    ),
                    group=tg_fs,
                )
            for c in range(N_COLS):
                xp_h[c].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_DEC_FA_STATE + c * 768 + 512,
                        sizes=[1, X_ELEMS],
                        strides=[0, 1],
                    ),
                    wait=True,
                    group=tg_fs,
                )
            tg_fs.finish()

            w_tasks = []
            for chk0 in range(0, kv_chunks, 128):
                cnt = min(128, kv_chunks - chk0)
                w_tasks.append((kv_buf, [KV_TOTAL], lambda c, s=chk0: ((c // 2) * MAX_KV_CHUNKS + s) * W_ELEMS, cnt * W_ELEMS))
            if layer_mode == 4:
                w_tasks.append((kvo_buf, [KV_OUT_TOTAL], lambda c: (c // 2) * 512, W_ELEMS))

            for b0 in range(0, len(w_tasks), 3):
                b1 = min(b0 + 3, len(w_tasks))
                is_last_b = b1 == len(w_tasks)
                tgf = TaskGroup()
                for wi in range(b0, b1):
                    buf, tdims, off_fn, slen = w_tasks[wi]
                    do_wait = (not is_last_b) and (wi == b1 - 1)
                    for c in range(N_COLS):
                        wp_h[c].fill(
                            buf,
                            tap=TensorAccessPattern(
                                tensor_dims=tdims, offset=off_fn(c), sizes=[1, slen], strides=[0, 1]
                            ),
                            wait=do_wait,
                            group=tgf,
                        )
                if is_last_b:
                    if layer_mode == 3:
                        for c in range(N_COLS):
                            yc_h[c].drain(
                                y_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[ACT_TOTAL],
                                    offset=SLOT_DEC_FA_STATE + c * 768,
                                    sizes=[3, 256],
                                    strides=[256, 1],
                                ),
                                wait=True,
                                group=tgf,
                            )
                    else:
                        for part in range(2):
                            for c in range(N_COLS):
                                yc_h[c].drain(
                                    y_buf,
                                    tap=TensorAccessPattern(
                                        tensor_dims=[ACT_TOTAL],
                                        offset=SLOT_ATTN_OUT + c * 256 + part * 64,
                                        sizes=[2, 2, 64],
                                        strides=[2048, 128, 1],
                                    ),
                                    wait=(part == 1),
                                    group=tgf,
                                )
                tgf.finish()

            if layer_mode == 4:
                run_gemm_passes(W_ATTN_O, 4, 1, SLOT_ATTN_OUT, SLOT_RES_TMP)

        # Shared tail (Modes 0, 1, 4)
        if layer_mode in (0, 1, 4):
            run_post_norm_and_res(SLOT_RES_TMP, y_buf, [ACT_TOTAL], SLOT_H1)
            run_rms_to_xq(SLOT_NORM_1, SLOT_H1, SLOT_XQ)

            ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
            ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
            ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

            tile_off = 0
            for kc, npass in enumerate([4, 4, 4, 3]):
                tg1 = TaskGroup()
                fill_x_xq(SLOT_XQ, tg1)
                for p in range(npass):
                    if npass == 4:
                        fill_w_tap(
                            w_buf,
                            [W_TOTAL],
                            [ffn_g_base + (tile_off + c * 16 + p) * W_ELEMS for c in range(N_COLS)],
                            [4, 18, 256],
                            [4 * W_ELEMS, 256, 1],
                            tg1,
                            wait=(p == npass - 1),
                        )
                    else:
                        fill_w_1d(
                            w_buf,
                            [W_TOTAL],
                            [ffn_g_base + tile_off * W_ELEMS + (p * 32 + c * 4) * W_ELEMS for c in range(N_COLS)],
                            4 * W_ELEMS,
                            tg1,
                            wait=(p == npass - 1),
                        )
                tg1.finish()
                tg2 = TaskGroup()
                for p in range(npass):
                    if npass == 4:
                        fill_w_tap(
                            w_buf,
                            [W_TOTAL],
                            [ffn_u_base + (tile_off + c * 16 + p) * W_ELEMS for c in range(N_COLS)],
                            [4, 18, 256],
                            [4 * W_ELEMS, 256, 1],
                            tg2,
                        )
                    else:
                        fill_w_1d(
                            w_buf,
                            [W_TOTAL],
                            [ffn_u_base + tile_off * W_ELEMS + (p * 32 + c * 4) * W_ELEMS for c in range(N_COLS)],
                            4 * W_ELEMS,
                            tg2,
                        )
                if npass == 4:
                    drain_linear(y_buf, [ACT_TOTAL], SLOT_FFN_XQ + kc * 2560, tg2)
                else:
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_SW + kc * 2048 + c * 64,
                                sizes=[4, 4, 16],
                                strides=[16, 512, 1],
                            ),
                            wait=True,
                            group=tg2,
                        )
                tg2.finish()
                if npass != 4:
                    run_rms_to_xq(None, SLOT_SW + kc * 2048, SLOT_FFN_XQ + kc * 2560)
                tile_off += npass * 32

            run_gemm_passes(ffn_d_base, 4, 4, SLOT_FFN_XQ, SLOT_RES_TMP)
            run_post_norm_and_res(SLOT_RES_TMP, yn_buf, [2048], 0)

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *x_prods, *y_cons])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=1)
    parser.add_argument("--kv-chunks", type=int, default=0)
    opts = parser.parse_args()
    run_design_cli(
        q4layer_lfm3,
        opts,
        device=NPU2(),
        compile_kwargs=lambda o: {
            "layer_mode": o.layer_mode,
            "kv_chunks": o.kv_chunks,
        },
    )


if __name__ == "__main__":
    main()

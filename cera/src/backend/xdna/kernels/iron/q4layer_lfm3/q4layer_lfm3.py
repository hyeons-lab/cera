# Fused B=1 Q4_0 LFM3 decode layer (32 cores: 4 rows x 8 cols, npu2).
# Modes via g_params[6]:
#   0 = ShortConv + FFN decode (using prefill Conv weights + conv_state)
#   1 = GQA + SWA/Causal Attention + FFN decode (using prefill Attn weights + kv_buf)
#   2 = Attn Phase A (RMS + QKV + Gate + RoPE -> kvo_buf + SLOT_DEC_Q_BF16 + SLOT_ATTN_GATE)
#   3 = Attn FA Partial (Reload SLOT_DEC_Q_BF16 + SLOT_DEC_FA_STATE -> FA -> Drain SLOT_DEC_FA_STATE)
#   4 = Attn FA Final (Reload SLOT_HIDDEN_IN + Gate + Q_BF16 + FA_STATE -> FA + cur_kv -> O + FFN)

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
SLOT_FFN_XQ = 10240          # 4 * 2560 = 10240 floats (4 K-chunks of 2048 dims, stride 2560)
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


VALID_MODES = (0, 1, 2, 3, 4)


def _validate_compile_params(layer_mode: int, kv_chunks: int) -> None:
    validate_compile_params(
        layer_mode,
        kv_chunks,
        valid_modes=VALID_MODES,
        nonzero_kv_modes=(3,),
        max_kv_chunks=(2 * MAX_KV_CHUNKS if layer_mode == 1 else MAX_KV_CHUNKS),
    )


@iron.jit(use_cache=False)
def q4layer_lfm3(
    w: In,
    y: Out,
    y_next: Out,
    kv: In,
    kv_out: Out,
    kv_hi: In,
    layer_mode: CompileTime[int] = 1,
    kv_chunks: CompileTime[int] = 0,
):
    _validate_compile_params(layer_mode, kv_chunks)
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
    core_flags = oz_flags
    fa_flags = common_flags + ["-O2", "-fno-unroll-loops", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]
    o3_flags = common_flags + ["-O3"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm3_dec_gemm_op",
        source_file=str(here / "q4_lfm3_dec_gemm_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32],
        compile_flags=o3_flags,
    )
    q4_fa_fn = ExternalFunction(
        "q4_lfm3_dec_fa_op",
        source_file=str(here / "q4_lfm3_dec_fa_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=fa_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm3_dec_w_op",
        source_file=str(here / "q4_lfm3_dec_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm3_dec_y_op",
        source_file=str(here / "q4layer_lfm3_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=core_flags,
    )

    W0_L3L2 = [ObjectFifo(w_pkt_ty, depth=4, name=f"W0_L3L2_{c}") for c in range(N_COLS)]
    W0_L2L1 = [W0_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=2, name=f"W0_L2L1_{c}") for c in range(N_COLS)]
    W1_L3L2 = [ObjectFifo(w_pkt_ty, depth=4, name=f"W1_L3L2_{c}") for c in range(N_COLS)]
    W1_L2L1 = [W1_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=2, name=f"W1_L2L1_{c}") for c in range(N_COLS)]

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
    KIND_Y = 1
    KIND_GEMM = 2
    KIND_FA = 3

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
            emit(KIND_W, 1, 6, core_id, 0, flags_x, mask)
            emit(KIND_Y, 1, 0, 0, 0, 0, mask)

        def emit_post_res(update_res, mask):
            emit(KIND_W, 1, 7, core_id, 0, update_res, mask)
            emit(KIND_Y, 1, 2, 0, 0, 0, mask)

        def emit_gemm(n_passes, k_chunks, op_flags, a2_flags, mask):
            emit(KIND_GEMM, n_passes, op_flags, n_passes, k_chunks, a2_flags, mask)

        # Peeled step 0 already loaded SLOT_NORM_0 + g_params into every worker.
        # Stage 1: Initial RMSNorm + Q8 (save_res=1)
        emit_rms(-1, 0, 1, M012)

        # Mode 0 (Conv): tg_conv (XQ + B/C/X + WT + kv_in -> Q8 drain XQ + state[0..2]) + tgs_out (state[3..5] + OUT_PROJ)
        emit_gemm(4, 1, 0b1011, 0, M0)        # B/C/X (pt_outer=1, load_x=1, mode=2)
        emit(KIND_W, 1, 3, r, 0, 0, M0)       # W_CONV_WT
        emit(KIND_W, 1, 4, r, 0, 0, M0)       # conv_state_in
        emit(KIND_Y, 1, 9, 0, 0, 0, M0)       # In-core Conv + Q8(conv_y) -> SLOT_XQ
        emit(KIND_Y, 6, 3, 0, 0, 0, M0)       # Drain conv_state[0..5] -> kvo_buf
        emit_gemm(4, 1, 0b0010, 2, M0)        # OUT_PROJ (load_x=1, do_drain=1)

        # Mode 1 & 2 Stage A (Attn): Q, stacked (K, V, ATTN_GATE) -> Q/K Norm + RoPE + gate_sig (from SLOT_Q..ATTN_GATE)
        emit_gemm(4, 1, 0b0010, 2, M12)       # Q (2048, load_x=1, do_drain=1)
        emit_gemm(3, 1, 0b0000, 2, M12)       # p0=K, p1=V, p2=ATTN_GATE (reuse XQ)
        emit(KIND_W, 1, 8, core_id, 0, 0, M12)# Q norm+RoPE + gate_sig + K norm/RoPE + V bf16 from packed SLOT_Q..ATTN_GATE
        emit(KIND_Y, 1, 5, 0, 0, 0, M12)      # Drain K/V cur -> kvo_buf
        emit(KIND_Y, 1, 5, 0, 0, 0, M2)       # Mode 2: Drain q_bf16 -> SLOT_DEC_Q_BF16
        emit(KIND_Y, 3, 8, 0, 0, 0, M2)       # Mode 2: Reset+drain fa_state -> SLOT_DEC_FA_STATE

        # Modes 3 & 4 Stage B: Reload res_acc (M4), gate (M4), q_bf16 & fa_state (M34)
        emit(KIND_W, 1, 7, core_id, 0, 2, M4) # Load res_acc from SLOT_HIDDEN_IN
        emit(KIND_W, 1, 2, c * 2 + r, 0, 0, M4)# Load gate_sig from SLOT_ATTN_GATE
        emit(KIND_W, 1, 10, core_id, 0, 0, M34)# Load q_bf16
        emit(KIND_W, 1, 11, core_id, 0, 0, M34)# Load fa_state (o_acc + row_m/row_l)

        # Modes 1, 3, 4: FA over kv_chunks (+ cur_kv for M14) [active on W0 (r < 2) only]
        emit(KIND_FA, 1, r, 0, 0, 0, M134)
        emit(KIND_Y, 3, 7, 0, 0, 0, M3)       # Mode 3: Drain fa_state -> SLOT_DEC_FA_STATE
        emit(KIND_Y, 2, 4, 0, 0, 0, M14)       # Gate + Q8(o_acc) -> SLOT_ATTN_OUT
        emit_gemm(4, 1, 0b0010, 2, M14)       # O_PROJ (load_x=1, do_drain=1)

        # Shared tail (Modes 0, 1, 4): post_attn_norm + res -> load SLOT_NORM_1 -> ffn_norm -> FFN -> post_ffn_norm + res
        emit_post_res(1, M014)
        emit_rms(1, 0, 0, M014)

        for g_slot in range(4):
            load_x = 1 if g_slot == 0 else 0
            # Gate chunks 0..3 cached across ffn_gate (slot 0) + ffn_gates_hi[0..2] (slots 1..3)
            if c < 6 or g_slot < 3:
                emit_gemm(4, 1, (load_x << 1) | 1, 1 | (g_slot << 3), M014)
            else:
                emit(0, 0, 0, 0, 0, 0, 0)
        for g_slot in range(4):
            # Up chunks 0..3 + SiLU*Gate + Q8 drain directly -> SLOT_FFN_XQ (cols 6..7 emit zero for chunk 3)
            if c < 6 or g_slot < 3:
                emit_gemm(4, 1, 1, 2 | 4 | (g_slot << 3) | 32, M014)
            else:
                emit(KIND_Y, 1, 10, 0, 0, 0, M014)

        emit_gemm(4, 4, 0b0010, 2, M014)      # FFN_DOWN (4 passes x 4 K-chunks, load_x per kc)
        emit_post_res(0, M014)
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
        half_r_val = r & 1
        is_w0_val = 1 if r < 2 else 0

        def core_body(w_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_fa_fn, q4_w_fn, q4_y_fn):
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
            i32_5 = arith.constant(i32_ty, 5)
            i32_6 = arith.constant(i32_ty, 6)
            i32_10 = arith.constant(i32_ty, 10)
            i32_32 = arith.constant(i32_ty, 32)
            my_r = arith.constant(i32_ty, r)
            half_r = arith.constant(i32_ty, half_r_val)
            is_w0 = arith.constant(i32_ty, is_w0_val)
            my_cid = arith.constant(i32_ty, core_id)

            def i32(v):
                return arith.index_cast(i32_ty, v)

            wp0 = w_fifo.acquire(1)
            kv_ch = arith.bitcast(i32_ty, wp0[idx_kvc])
            lmode = arith.bitcast(i32_ty, wp0[idx_lmode])
            q4_w_fn(wp0, i32_0, my_cid, i32_1, i32_0)
            w_fifo.release(1)

            lmode_bit = arith.shli(i32_1, lmode)
            is_m1 = arith.cmpi(arith.CmpIPredicate.eq, lmode, i32_1)
            is_m4 = arith.cmpi(arith.CmpIPredicate.eq, lmode, i32_4)
            use_cur = arith.ori(is_m1, is_m4)
            has_kv = arith.cmpi(arith.CmpIPredicate.sgt, kv_ch, i32_0)
            kv_ch_idx = arith.index_cast(idx_ty, kv_ch)

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
                            q4_w_fn(wp, op, arith.addi(a0, i32(i)), a1, a2)
                            w_fifo.release(1)
                    with else_(if_w):
                        is_y = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_1)
                        with if_(is_y, hasElse=True) as if_y:
                            for i in range_(cnt):
                                yp = y_fifo.acquire(1)
                                q4_y_fn(yp, op, arith.addi(a0, i32(i)))
                                y_fifo.release(1)
                        with else_(if_y):
                            is_g = arith.cmpi(arith.CmpIPredicate.eq, kind, i32_2)
                            with if_(is_g, hasElse=True) as if_g:
                                pt_outer = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(op, i32_1), i32_0)
                                load_x = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(op, i32_2), i32_0)
                                mode_sel = arith.shrui(op, i32_2)
                                is_bcx = arith.cmpi(arith.CmpIPredicate.eq, mode_sel, i32_2)
                                n_ghalf = arith.select(is_bcx, idx3, idx1)
                                dst_gate = arith.andi(a2, i32_1)
                                do_drain = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_2), i32_0)
                                drain_q8 = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_4), i32_0)
                                g_slot = arith.andi(arith.shrui(a2, i32_3), i32_3)
                                do_silu = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_32), i32_0)
                                y_drain_op = arith.select(do_silu, arith.select(drain_q8, i32_6, i32_10), i32_1)
                                y_drain_arg = arith.select(do_silu, g_slot, i32_0)
                                n_outer = arith.select(pt_outer, idx2, arith.index_cast(idx_ty, a0))
                                n_inner = arith.select(pt_outer, arith.index_cast(idx_ty, a0), idx2)
                                packed_xq = arith.cmpi(arith.CmpIPredicate.sgt, a1, i32_1)
                                xq_a1 = arith.select(packed_xq, i32_1, i32_0)
                                with if_(load_x, hasElse=False):
                                    xp = w_fifo.acquire(1)
                                    q4_w_fn(xp, i32_5, i32_0, xq_a1, i32_0)
                                    w_fifo.release(1)
                                g_slot_shift = arith.shli(g_slot, i32_5)
                                for gh in range_(n_ghalf):
                                    is_gh0 = arith.cmpi(arith.CmpIPredicate.eq, gh, idx0)
                                    is_gh1 = arith.cmpi(arith.CmpIPredicate.eq, gh, idx1)
                                    gh_bcx = arith.select(is_gh0, i32_1, arith.select(is_gh1, i32_2, i32_0))
                                    gh_dst = arith.select(is_bcx, gh_bcx, dst_gate)
                                    dst_shift = arith.shli(gh_dst, i32_1)
                                    for oi in range_(n_outer):
                                        for ii in range_(n_inner):
                                            cur_pt = arith.select(pt_outer, i32(oi), i32(ii))
                                            cur_p = arith.select(pt_outer, i32(ii), i32(oi))
                                            for kc in range_(a1):
                                                is_sub = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
                                                add_bit = arith.select(is_sub, i32_1, i32_0)
                                                x_slot = arith.shli(i32(kc), i32_3)
                                                gflags = arith.ori(
                                                    add_bit, arith.ori(dst_shift, arith.ori(x_slot, g_slot_shift))
                                                )
                                                wp = w_fifo.acquire(1)
                                                is_mine = arith.cmpi(arith.CmpIPredicate.eq, cur_pt, half_r)
                                                with if_(is_mine, hasElse=False):
                                                    q4_gemm_fn(wp, cur_p, gflags)
                                                w_fifo.release(1)
                                with if_(do_drain, hasElse=False):
                                    yp = y_fifo.acquire(1)
                                    q4_y_fn(yp, y_drain_op, y_drain_arg)
                                    y_fifo.release(1)
                            with else_(if_g):
                                if is_w0_val:
                                    for kc in range_(kv_ch_idx):
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
                W0_L2L1[c].cons() if r < 2 else W1_L2L1[c].cons(),
                Y_L1L2[c][r].prod(),
                sched_bufs[c][r],
                q4_gemm_fn,
                q4_fa_fn,
                q4_w_fn,
                q4_y_fn,
            ],
            tile=Tile(c, 2 + r),
            stack_size=2880,
            while_true=True,
        )
        for c in range(N_COLS)
        for r in range(N_ROWS)
    ]

    w0_prods = [W0_L3L2[c].prod() for c in range(N_COLS)]
    w1_prods = [W1_L3L2[c].prod() for c in range(N_COLS)]
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, kv_hi_buf, *fifos):
        w0p_h = fifos[0:N_COLS]
        w1p_h = fifos[N_COLS : 2 * N_COLS]
        yc_h = fifos[2 * N_COLS : 3 * N_COLS]

        def fill_w_pair_tap(src_buf, tdims, off0_per_col, off1_per_col, sizes, strides, tg, wait=False):
            for c in range(N_COLS):
                w0p_h[c].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off0_per_col[c], sizes=sizes, strides=strides
                    ),
                    wait=wait,
                    group=tg,
                )
                w1p_h[c].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off1_per_col[c], sizes=sizes, strides=strides
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_w_pair_1d(src_buf, tdims, off0_per_col, off1_per_col, slen, tg, wait=False):
            fill_w_pair_tap(src_buf, tdims, off0_per_col, off1_per_col, [1, slen], [0, 1], tg, wait=wait)

        def fill_bcast(src_buf, tdims, off, tg, wait=False):
            fill_w_pair_1d(src_buf, tdims, [off] * N_COLS, [off] * N_COLS, W_ELEMS, tg, wait=wait)

        def fill_per_col_bcast(src_buf, tdims, off_per_col, tg, wait=False):
            fill_w_pair_1d(src_buf, tdims, off_per_col, off_per_col, W_ELEMS, tg, wait=wait)

        def drain_linear(dst_buf, tdims, dst_off, tg, wait=True):
            for c in range(N_COLS):
                yc_h[c].drain(
                    dst_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=dst_off + c * 256, sizes=[1, 256], strides=[0, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def drain_strided(out_slot, tg, wait=True):
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=out_slot + c * 64,
                        sizes=[4, 4, 16],
                        strides=[16, 512, 1],
                    ),
                    wait=wait,
                    group=tg,
                )

        def emit_std_gemm(tg, w_base, n_passes, xq_slot, out_slot, load_x=True, do_drain=True, wait=True):
            if load_x:
                fill_bcast(y_buf, [ACT_TOTAL], xq_slot, tg)
            fill_w_pair_tap(
                w_buf,
                [W_TOTAL],
                [w_base + (c * 4) * W_ELEMS for c in range(N_COLS)],
                [w_base + (c * 4 + 2) * W_ELEMS for c in range(N_COLS)],
                [n_passes, 18, 512],
                [32 * W_ELEMS, 512, 1],
                tg,
                wait=(wait and not do_drain),
            )
            if do_drain:
                drain_strided(out_slot, tg, wait=wait)

        # Peeled step 0 + initial RMSNorm+Q8 (Modes 0, 1, 2) or params+norm only (Modes 3, 4)
        if layer_mode in (0, 1, 2):
            tg0 = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_NORM_0, tg0)
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN, tg0, wait=True)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg0)
            tg0.finish()
        else:
            norm_slot = SLOT_NORM_1 if layer_mode == 3 else SLOT_DEC_FIN_NORM
            tg0 = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], norm_slot, tg0, wait=True)
            tg0.finish()

        if layer_mode == 0:
            # TG 1: XQ + fused B/C/X + WT + kv_in -> Q8 drain SLOT_XQ + conv_state[0..2] (4 MM2S, 4 S2MM)
            tgc = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_XQ, tgc)
            fill_w_pair_tap(
                w_buf,
                [W_TOTAL],
                [W_CONV_IN_B + c * 16 * W_ELEMS for c in range(N_COLS)],
                [W_CONV_IN_B + (c * 16 + 8) * W_ELEMS for c in range(N_COLS)],
                [3, 144, 256],
                [128 * W_ELEMS, 256, 1],
                tgc,
            )
            fill_w_pair_1d(
                w_buf,
                [W_TOTAL],
                [W_CONV_WT + c * W_ELEMS for c in range(N_COLS)],
                [W_CONV_WT + c * W_ELEMS for c in range(N_COLS)],
                W_ELEMS,
                tgc,
            )
            fill_per_col_bcast(kv_buf, [KV_TOTAL], [c * W_ELEMS for c in range(N_COLS)], tgc, wait=True)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tgc, wait=False)
            for c in range(N_COLS):
                yc_h[c].drain(
                    kvo_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[KV_OUT_TOTAL],
                        offset=c * W_ELEMS,
                        sizes=[1, 768],
                        strides=[0, 1],
                    ),
                    wait=True,
                    group=tgc,
                )
            tgc.finish()

            # TG 2: conv_state[3..5] drains (1 S2MM BD) + OUT_PROJ (2 MM2S + 1 S2MM)
            tgs_out = TaskGroup()
            for c in range(N_COLS):
                yc_h[c].drain(
                    kvo_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[KV_OUT_TOTAL],
                        offset=c * W_ELEMS + 768,
                        sizes=[1, 768],
                        strides=[0, 1],
                    ),
                    wait=False,
                    group=tgs_out,
                )
            emit_std_gemm(tgs_out, W_CONV_OUT, 4, SLOT_XQ, SLOT_RES_TMP, wait=True)
            tgs_out.finish()
        elif layer_mode in (1, 2):
            # Merge XQ + Q + (K, V) + ATTN_GATE (without spilling into W_ATTN_O) in 1 TG
            tg_qkv = TaskGroup()
            emit_std_gemm(tg_qkv, W_ATTN_Q, 4, SLOT_XQ, SLOT_Q, load_x=True, do_drain=True, wait=False)
            emit_std_gemm(tg_qkv, W_ATTN_K, 2, SLOT_XQ, SLOT_K, load_x=False, do_drain=False, wait=False)
            fill_w_pair_1d(
                w_buf,
                [W_TOTAL],
                [W_ATTN_GATE] * N_COLS,
                [W_ATTN_GATE] * N_COLS,
                2 * W_ELEMS,
                tg_qkv,
                wait=False,
            )
            drain_strided(SLOT_K, tg_qkv, wait=True)
            tg_qkv.finish()

            # Q[2048] + (K+V+GATE)[1056] in 1 bcast (rope+q/k norms already loaded in tg0!) -> kvo_buf (+ Mode 2)
            tgr = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_Q, tgr, wait=True)
            drain_linear(kvo_buf, [KV_OUT_TOTAL], 0, tgr, wait=(layer_mode != 2))
            if layer_mode == 2:
                drain_linear(y_buf, [ACT_TOTAL], SLOT_DEC_Q_BF16, tgr, wait=False)
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
                        group=tgr,
                    )
                tgr.finish()
                return
            tgr.finish()
        if layer_mode in (3, 4):
            tg_b = TaskGroup()
            if layer_mode == 4:
                fill_bcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN, tg_b)
                fill_bcast(y_buf, [ACT_TOTAL], SLOT_ATTN_GATE, tg_b)
            fill_per_col_bcast(y_buf, [ACT_TOTAL], [SLOT_DEC_Q_BF16 + c * 256 for c in range(N_COLS)], tg_b)
            fill_per_col_bcast(y_buf, [ACT_TOTAL], [SLOT_DEC_FA_STATE + c * 768 for c in range(N_COLS)], tg_b, wait=True)
            tg_b.finish()

        if layer_mode in (1, 3, 4):
            w_tasks = []
            for chk0 in range(0, kv_chunks, 128):
                cnt = min(128, kv_chunks - chk0)
                if chk0 < MAX_KV_CHUNKS:
                    blk_bo = kv_buf
                    blk_chk0 = chk0
                else:
                    blk_bo = kv_hi_buf
                    blk_chk0 = chk0 - MAX_KV_CHUNKS
                w_tasks.append((blk_bo, [KV_TOTAL], lambda c, s=blk_chk0: ((c // 2) * MAX_KV_CHUNKS + s) * W_ELEMS, cnt * W_ELEMS))
            if layer_mode != 3:
                w_tasks.append((kvo_buf, [KV_OUT_TOTAL], lambda c: (c // 2) * 512, W_ELEMS))

            for b0 in range(0, len(w_tasks), 3):
                b1 = min(b0 + 3, len(w_tasks))
                is_last_b = b1 == len(w_tasks)
                tgf = TaskGroup()
                for wi in range(b0, b1):
                    buf, tdims, off_fn, slen = w_tasks[wi]
                    do_wait = (not is_last_b) and (wi == b1 - 1)
                    for c in range(N_COLS):
                        w0p_h[c].fill(
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

            if layer_mode == 3:
                return

            tg_o = TaskGroup()
            emit_std_gemm(tg_o, W_ATTN_O, 4, SLOT_ATTN_OUT, SLOT_RES_TMP)
            tg_o.finish()

        # Shared tail (Modes 0, 1, 4): res1 (using SLOT_NORM_0.post_norm_w) -> SLOT_NORM_1 + rms1 -> FFN -> down -> res2
        if layer_mode in (0, 1, 4):
            tg_res1 = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_RES_TMP, tg_res1, wait=False)
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_NORM_1, tg_res1, wait=False)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_H1, tg_res1, wait=True)
            tg_res1.finish()

            tg_rms1 = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_H1, tg_rms1, wait=True)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg_rms1)
            tg_rms1.finish()

            ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
            ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
            ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

            # Unified FFN Gate+Up across kc=0..3 directly into SLOT_FFN_XQ (no intermediate SW drain or tg_q3)
            tg_ffn = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_XQ, tg_ffn)
            for ffn_base in (ffn_g_base, ffn_u_base):
                do_wait = ffn_base == ffn_u_base
                for c in range(N_COLS):
                    n_k = 4 if c < 6 else 3
                    off_c = ffn_base + c * 16 * W_ELEMS
                    w0p_h[c].fill(
                        w_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[W_TOTAL],
                            offset=off_c,
                            sizes=[n_k, 144, 256],
                            strides=[128 * W_ELEMS, 256, 1],
                        ),
                        wait=do_wait,
                        group=tg_ffn,
                    )
                    w1p_h[c].fill(
                        w_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[W_TOTAL],
                            offset=off_c + 8 * W_ELEMS,
                            sizes=[n_k, 144, 256],
                            strides=[128 * W_ELEMS, 256, 1],
                        ),
                        wait=do_wait,
                        group=tg_ffn,
                    )
            for c in range(N_COLS):
                yc_h[c].drain(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL],
                        offset=SLOT_FFN_XQ + c * 256,
                        sizes=[4, 256],
                        strides=[2560, 1],
                    ),
                    wait=True,
                    group=tg_ffn,
                )
            tg_ffn.finish()

            # FFN_DOWN in 1 TG (packed 4XQ in 1 bcast + all 4 W chunks in 1 BD) -> SLOT_RES_TMP
            tg_down = TaskGroup()
            fill_w_pair_tap(
                y_buf,
                [ACT_TOTAL],
                [SLOT_FFN_XQ] * N_COLS,
                [SLOT_FFN_XQ] * N_COLS,
                [4, 32, 36],
                [2560, 64, 1],
                tg_down,
            )
            fill_w_pair_tap(
                w_buf,
                [W_TOTAL],
                [ffn_d_base + c * 16 * W_ELEMS for c in range(N_COLS)],
                [ffn_d_base + (c * 16 + 8) * W_ELEMS for c in range(N_COLS)],
                [4, 2, 72, 256],
                [128 * W_ELEMS, 4 * W_ELEMS, 256, 1],
                tg_down,
                wait=False,
            )
            drain_strided(SLOT_RES_TMP, tg_down)
            tg_down.finish()

            tg_res2 = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_RES_TMP, tg_res2, wait=True)
            drain_linear(yn_buf, [2048], 0, tg_res2)
            tg_res2.finish()

    rt = Runtime(
        seq_fn,
        fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, kv_ty, *w0_prods, *w1_prods, *y_cons],
    )
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=1, choices=VALID_MODES)
    parser.add_argument("--kv-chunks", type=int, default=0)
    opts = parser.parse_args()
    _validate_compile_params(opts.layer_mode, opts.kv_chunks)
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

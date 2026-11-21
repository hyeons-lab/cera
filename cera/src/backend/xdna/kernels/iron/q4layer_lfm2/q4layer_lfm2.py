# Fused B=1 Q4_0 LFM2 / LFM2.5 decode layer (32 cores: 4 rows x 8 cols, npu2).
# Modes via g_params[6]:
#   0 = ShortConv (d_conv=2) + FFN (8192) decode
#   1 = GQA (32 Q heads, 8 KV heads, head_dim=64) + QK-norm + NeoX RoPE + FFN (8192) decode

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
N_KV_HEADS = 8
MAX_KV_CHUNKS = 256

# Weight offsets matching q4prefill_lfm2_conv and q4prefill_lfm2_attn identically.
W_CONV_IN_B = 0
W_CONV_IN_C = 128 * W_ELEMS
W_CONV_IN_X = 256 * W_ELEMS
W_CONV_OUT = 384 * W_ELEMS
W_CONV_WT = 512 * W_ELEMS
W_CONV_FFN_GATE = 520 * W_ELEMS
W_CONV_FFN_UP = 1032 * W_ELEMS
W_CONV_FFN_DOWN = 1544 * W_ELEMS

W_ATTN_Q = 0
W_ATTN_K = 128 * W_ELEMS
W_ATTN_V = 160 * W_ELEMS
W_ATTN_O = 192 * W_ELEMS
W_ATTN_FFN_GATE = 320 * W_ELEMS
W_ATTN_FFN_UP = 832 * W_ELEMS
W_ATTN_FFN_DOWN = 1344 * W_ELEMS

W_TOTAL = 2056 * W_ELEMS

SLOT_HIDDEN_IN = 0
SLOT_H1 = 2560
SLOT_RES_TMP = 5120
SLOT_XQ = 7680
SLOT_FFN_XQ = 10240          # 4 * 2560 = 10240 floats (4 K-chunks of 2048 dims, stride 2560)
SLOT_ATTN_OUT = 20480        # 2048 floats (32 * 64 strided Q8 blocks)
SLOT_Q = 24576
SLOT_K = 27136
SLOT_V = SLOT_K + 512        # 27648: contiguous with K so 1 w_pkt carries K+V!
SLOT_SW = 32768              # 8192 + 512 pad (holds Stage A/B FA partials)
SLOT_NORM_0 = 41472          # 4608 floats (norm_w[2048], unused[2048], params[16])
SLOT_NORM_1 = 46080          # 4608 floats (ffn_norm_w[2048])
SLOT_ROPE_PKT = 50688        # 4608 floats (rope_cs[64], ..., q_norm[4096], k_norm[4224])
ACT_TOTAL = 50688 + 4608

SLOT_CONV_B = SLOT_SW
SLOT_CONV_C = SLOT_SW + 2560
SLOT_CONV_X = SLOT_SW + 5120
SLOT_CONV_OUT = SLOT_Q

KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * W_ELEMS
KV_OUT_TOTAL = 8 * W_ELEMS


@iron.jit(use_cache=False)
def q4layer_lfm2(
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
    gemm_flags = common_flags + ["-O3"]
    core_flags = common_flags + ["-O2", "-fno-unroll-loops"]
    oz_flags = common_flags + ["-Oz", "-fno-unroll-loops"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm2_dec_gemm_op",
        source_file=str(here / "q4_lfm2_dec_gemm_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32],
        compile_flags=gemm_flags,
    )
    q4_fa_fn = ExternalFunction(
        "q4_lfm2_dec_fa_op",
        source_file=str(here / "q4_lfm2_dec_fa_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=core_flags,
    )
    q4_w_fn = ExternalFunction(
        "q4_lfm2_dec_w_op",
        source_file=str(here / "q4_lfm2_dec_w_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm2_dec_y_op",
        source_file=str(here / "q4layer_lfm2_core.cc"),
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

    M0 = 1
    M1 = 2
    M2 = 4
    M3 = 8
    M4 = 16
    M01 = M0 | M1
    M012 = M0 | M1 | M2
    M12 = M1 | M2
    M34 = M3 | M4
    M14 = M1 | M4
    M014 = M0 | M1 | M4
    M_FA = M1 | M3 | M4

    def build_worker_sched(c, r):
        core_id = c * 4 + r
        rows = []

        def emit(kind, cnt, op, a0=0, a1=0, a2=0, mask=M01):
            w0 = (kind & 0xFF) | ((cnt & 0xFF) << 8) | ((op & 0xFF) << 16) | ((a0 & 0xFF) << 24)
            w1 = (a1 & 0xFF) | ((a2 & 0xFF) << 8) | ((mask & 0xFFFF) << 16)
            rows.extend((w0, w1))

        def emit_rms(norm_slot, load_params, flags_x, mask):
            if norm_slot >= 0:
                emit(KIND_W, 1, 0, core_id, load_params, 0, mask)
            emit(KIND_W, 1, 6, core_id, 0, flags_x, mask)
            emit(KIND_Y, 1, 0, 0, 0, 0, mask)

        def emit_res(update_res, mask):
            emit(KIND_W, 1, 7, core_id, 0, update_res, mask)
            emit(KIND_Y, 1, 2, 0, 0, 0, mask)

        def emit_gemm(n_passes, k_chunks, op_flags, a2_flags, mask):
            # op_flags: bit 0=pt_outer, bit 1=load_x, bits[3:2]=mode(0=std,1=fg_up,2=bcx)
            # a2_flags: bit 0=dst_gate, bit 1=do_drain, bits[3:2]=y_op_drain_sel(0->op1,1->op6)
            emit(KIND_GEMM, n_passes, op_flags, n_passes, k_chunks, a2_flags, mask)

        # Stage 1 (Modes 0, 1, 2): Initial RMSNorm + Q8 (save_res=1); step 0 already loaded norm0+params
        emit_rms(-1, 0, 1, M012)

        # Mode 0 (Conv, d_conv=2): cached XQ -> fused B/C/X -> conv_wt + conv_state -> ShortConv -> out_proj -> H1
        emit_gemm(4, 1, 0b1011, 0, M0)        # pt_outer=1, load_x=1, mode=2 (B->gate, C->conv_c, X->y_gemm)
        emit(KIND_W, 1, 3, r, 0, 0, M0)       # W_CONV_WT (3 taps)
        emit(KIND_W, 1, 4, r, 0, 0, M0)       # conv_state_in (2 frames)
        emit(KIND_Y, 1, 7, 0, 0, 0, M0)       # In-core Conv + Q8(conv_y) & drain -> SLOT_XQ
        emit(KIND_Y, 2, 3, 0, 0, 0, M0)       # Drain conv_state[0..1] -> kvo_buf
        emit_gemm(4, 1, 0b0010, 2 | (2 << 2), M0)  # OUT_PROJ (load_x=1, drain op=11 res+y -> SLOT_H1)

        # Modes 1 & 2 (Phase A): Q, K, V -> Q/K Norm + NeoX RoPE (from SLOT_Q+K+V) -> K/V drain
        emit_gemm(4, 1, 0b0010, 2, M12)       # Q (2048, load_x=1, do_drain=1)
        emit_gemm(1, 1, 0b0000, 2, M12)       # K (512, reuse XQ)
        emit_gemm(1, 1, 0b0000, 2, M12)       # V (512, reuse XQ)
        emit(KIND_W, 1, 8, core_id, 0, 0, M12)# Q + K norm/RoPE + V bf16 from packed SLOT_Q..V
        emit(KIND_Y, 1, 5, 0, 0, 0, M12)      # Drain K/V cur -> kvo_buf
        emit(KIND_Y, 1, 8, 0, 0, 0, M2)       # Mode 2: Drain q_bf16 -> SLOT_SW + c*1024

        # Modes 3 & 4: Stage B setup (save res_acc in M4 + reload q_bf16 & fa_state)
        emit(KIND_W, 1, 7, core_id, 0, 2, M4) # res_acc <- SLOT_HIDDEN_IN
        emit(KIND_W, 1, 10, core_id, 0, 0, M34) # q_bf16 <- SLOT_SW + c*1024
        emit(KIND_W, 1, 11, core_id, 0, 0, M34) # o_acc + row_m/l <- SLOT_SW + c*1024 + 256

        emit(KIND_FA, 1, r, 0, 0, 0, M_FA)    # FA over kv_chunks (+ cur_kv if not M3)
        emit(KIND_Y, 2, 9, 0, 0, 0, M3)       # Mode 3: Drain o_acc + {row_m, row_l} -> SLOT_SW + c*1024 + 256
        emit(KIND_Y, 1, 4, 0, 0, 0, M14)      # Normalize & Q8(o_acc) -> SLOT_ATTN_OUT
        emit_gemm(4, 1, 0b0010, 2 | (2 << 2), M14) # O_PROJ (load_x=1, drain op=11 res+y -> SLOT_H1)

        # Shared tail (Modes 0, 1, 4): load norm_1 + rms1(save_res=1) -> fused FFN -> down(res+y -> yn_buf)
        emit(KIND_W, 1, 0, core_id, 0, 0, M014) # SLOT_NORM_1
        emit_rms(-1, 0, 1, M014)

        for kc in range(4):
            load_x = 1 if kc == 0 else 0
            # mode=1 (Gate+Up in-core contiguous), pt_outer=1, do_drain=1, y_op=6 (SiLU*Up -> Q8)
            emit_gemm(4, 1, (1 << 2) | (load_x << 1) | 1, 2 | (1 << 2), M014)

        emit_gemm(4, 4, 0b0010, 2 | (2 << 2), M014) # FFN_DOWN (packed 4XQ + 4 k-chunks, drain op=11 res+y -> yn)
        return np.array(rows, dtype=np.int32).reshape(-1)

    sample_sched = build_worker_sched(0, 0)
    n_steps = len(sample_sched) // 2
    sched_ty = np.ndarray[(len(sample_sched),), np.dtype[np.int32]]
    sched_bufs = [
        [Buffer(sched_ty, initial_value=build_worker_sched(c, r), name=f"sc_{c}_{r}") for r in range(N_ROWS)]
        for c in range(N_COLS)
    ]

    def make_worker_body(c, r):
        core_id = c * 4 + r
        half_r_val = r & 1

        def core_body(w_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_fa_fn, q4_w_fn, q4_y_fn):
            idx_ty = IndexType.get()
            i32_ty = IntegerType.get_signless(32)
            idx0 = arith.constant(idx_ty, 0)
            idx1 = arith.constant(idx_ty, 1)
            idx2 = arith.constant(idx_ty, 2)
            idx3 = arith.constant(idx_ty, 3)
            idx_kvc = arith.constant(idx_ty, 4098)
            idx_lmode = arith.constant(idx_ty, 4102)
            i32_0 = arith.constant(i32_ty, 0)
            i32_1 = arith.constant(i32_ty, 1)
            i32_2 = arith.constant(i32_ty, 2)
            i32_3 = arith.constant(i32_ty, 3)
            i32_4 = arith.constant(i32_ty, 4)
            i32_5 = arith.constant(i32_ty, 5)
            i32_6 = arith.constant(i32_ty, 6)
            i32_8 = arith.constant(i32_ty, 8)
            i32_11 = arith.constant(i32_ty, 11)
            i32_16 = arith.constant(i32_ty, 16)
            i32_24 = arith.constant(i32_ty, 24)
            i32_255 = arith.constant(i32_ty, 255)
            my_r = arith.constant(i32_ty, r)
            half_r = arith.constant(i32_ty, half_r_val)
            my_cid = arith.constant(i32_ty, core_id)

            def i32(v):
                return arith.index_cast(i32_ty, v)

            wp0 = w_fifo.acquire(1)
            kv_ch = arith.bitcast(i32_ty, wp0[idx_kvc])
            lmode = arith.bitcast(i32_ty, wp0[idx_lmode])
            q4_w_fn(wp0, i32_0, my_cid, i32_1, i32_0)
            w_fifo.release(1)

            lmode_bit = arith.shli(i32_1, lmode)

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
                                is_fg = arith.cmpi(arith.CmpIPredicate.eq, mode_sel, i32_1)
                                is_bcx = arith.cmpi(arith.CmpIPredicate.eq, mode_sel, i32_2)
                                n_ghalf = arith.select(is_bcx, idx3, arith.select(is_fg, idx2, idx1))
                                dst_gate = arith.andi(a2, i32_1)
                                do_drain = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_2), i32_0)
                                drain_sel = arith.shrui(a2, i32_2)
                                drain_q8 = arith.cmpi(arith.CmpIPredicate.eq, drain_sel, i32_1)
                                drain_res = arith.cmpi(arith.CmpIPredicate.eq, drain_sel, i32_2)
                                y_drain_op = arith.select(drain_q8, i32_6, arith.select(drain_res, i32_11, i32_1))
                                n_outer = arith.select(pt_outer, idx2, arith.index_cast(idx_ty, a0))
                                n_inner = arith.select(pt_outer, arith.index_cast(idx_ty, a0), idx2)
                                packed_xq = arith.cmpi(arith.CmpIPredicate.sgt, a1, i32_1)
                                xq_a1 = arith.select(packed_xq, i32_1, i32_0)
                                with if_(load_x, hasElse=False):
                                    xp = w_fifo.acquire(1)
                                    q4_w_fn(xp, i32_5, i32_0, xq_a1, i32_0)
                                    w_fifo.release(1)
                                for gh in range_(n_ghalf):
                                    is_gh0 = arith.cmpi(arith.CmpIPredicate.eq, gh, idx0)
                                    is_gh1 = arith.cmpi(arith.CmpIPredicate.eq, gh, idx1)
                                    gh_fg = arith.select(is_gh0, i32_1, i32_0)
                                    gh_bcx = arith.select(is_gh0, i32_1, arith.select(is_gh1, i32_2, i32_0))
                                    gh_dst = arith.select(is_bcx, gh_bcx, arith.select(is_fg, gh_fg, dst_gate))
                                    dst_shift = arith.shli(gh_dst, i32_1)
                                    for oi in range_(n_outer):
                                        for ii in range_(n_inner):
                                            cur_pt = arith.select(pt_outer, i32(oi), i32(ii))
                                            cur_p = arith.select(pt_outer, i32(ii), i32(oi))
                                            for kc in range_(a1):
                                                is_sub = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
                                                add_bit = arith.select(is_sub, i32_1, i32_0)
                                                x_slot = arith.shli(i32(kc), i32_3)
                                                gflags = arith.ori(add_bit, arith.ori(dst_shift, x_slot))
                                                wp = w_fifo.acquire(1)
                                                is_mine = arith.cmpi(arith.CmpIPredicate.eq, cur_pt, half_r)
                                                with if_(is_mine, hasElse=False):
                                                    q4_gemm_fn(wp, cur_p, gflags)
                                                w_fifo.release(1)
                                with if_(do_drain, hasElse=False):
                                    yp = y_fifo.acquire(1)
                                    q4_y_fn(yp, y_drain_op, i32_0)
                                    y_fifo.release(1)
                            with else_(if_g):
                                for kc in range_(kv_ch):
                                    is_f = arith.cmpi(arith.CmpIPredicate.eq, kc, idx0)
                                    wp = w_fifo.acquire(1)
                                    q4_fa_fn(wp, my_r, i32(kc), arith.select(is_f, i32_2, i32_0))
                                    w_fifo.release(1)
                                has_kv = arith.cmpi(arith.CmpIPredicate.sgt, kv_ch, i32_0)
                                use_cur = arith.cmpi(arith.CmpIPredicate.ne, lmode, i32_3)
                                with if_(use_cur, hasElse=False):
                                    wp_c = w_fifo.acquire(1)
                                    q4_fa_fn(wp_c, my_r, i32_0, arith.select(has_kv, i32_1, i32_3))
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
            stack_size=2560,
            while_true=True,
        )
        for c in range(N_COLS)
        for r in range(N_ROWS)
    ]

    w0_prods = [W0_L3L2[c].prod() for c in range(N_COLS)]
    w1_prods = [W1_L3L2[c].prod() for c in range(N_COLS)]
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, *fifos):
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

        # Peeled step 0 (+ initial RMSNorm+Q8 in Modes 0, 1, 2)
        tg0 = TaskGroup()
        fill_bcast(y_buf, [ACT_TOTAL], SLOT_NORM_0, tg0, wait=(layer_mode not in (0, 1, 2)))
        if layer_mode in (0, 1, 2):
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN, tg0, wait=True)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg0)
        tg0.finish()

        if layer_mode == 0:
            # Merged Conv: XQ + fused B/C/X + W_CONV_WT + kv_buf -> Conv + Q8 -> SLOT_XQ + conv_state[0..1]
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
            for kk in range(2):
                for c in range(N_COLS):
                    yc_h[c].drain(
                        kvo_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[KV_OUT_TOTAL],
                            offset=c * W_ELEMS + kk * 256,
                            sizes=[1, 256],
                            strides=[0, 1],
                        ),
                        wait=(kk == 1),
                        group=tgc,
                    )
            tgc.finish()

            tg_cout = TaskGroup()
            emit_std_gemm(tg_cout, W_CONV_OUT, 4, SLOT_XQ, SLOT_H1)
            tg_cout.finish()
        else:
            if layer_mode in (1, 2):
                # Merge XQ + Q + (K+V) in 1 TaskGroup (3 MM2S BDs <= 4, 3 S2MM BDs <= 4)
                tg_qkv = TaskGroup()
                emit_std_gemm(tg_qkv, W_ATTN_Q, 4, SLOT_XQ, SLOT_Q, load_x=True, do_drain=True, wait=False)
                fill_w_pair_tap(
                    w_buf,
                    [W_TOTAL],
                    [W_ATTN_K + (c * 4) * W_ELEMS for c in range(N_COLS)],
                    [W_ATTN_K + (c * 4 + 2) * W_ELEMS for c in range(N_COLS)],
                    [2, 18, 512],
                    [32 * W_ELEMS, 512, 1],
                    tg_qkv,
                    wait=False,
                )
                drain_strided(SLOT_K, tg_qkv, wait=False)
                drain_strided(SLOT_V, tg_qkv, wait=True)
                tg_qkv.finish()

                # Q[2048] + (K+V)[1024] in 1 bcast (rope+q/k norms already loaded in tg0!) -> kvo_buf (+ Mode 2 q_bf16)
                tgr = TaskGroup()
                fill_bcast(y_buf, [ACT_TOTAL], SLOT_Q, tgr, wait=True)
                drain_linear(kvo_buf, [KV_OUT_TOTAL], 0, tgr, wait=(layer_mode != 2))
                if layer_mode == 2:
                    for c in range(N_COLS):
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_SW + c * 1024,
                                sizes=[1, 256],
                                strides=[0, 1],
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
                fill_per_col_bcast(y_buf, [ACT_TOTAL], [SLOT_SW + c * 1024 for c in range(N_COLS)], tg_b)
                fill_per_col_bcast(y_buf, [ACT_TOTAL], [SLOT_SW + c * 1024 + 256 for c in range(N_COLS)], tg_b, wait=True)
                tg_b.finish()

            w_tasks = []
            for chk0 in range(0, kv_chunks, 64):
                cnt = min(64, kv_chunks - chk0)
                w_tasks.append((kv_buf, [KV_TOTAL], lambda c, s=chk0: (c * MAX_KV_CHUNKS + s) * W_ELEMS, cnt * W_ELEMS))
            if layer_mode != 3:
                w_tasks.append((kvo_buf, [KV_OUT_TOTAL], lambda c: c * 256, W_ELEMS))

            for b0 in range(0, len(w_tasks), 3):
                b1 = min(b0 + 3, len(w_tasks))
                is_last_b = b1 == len(w_tasks)
                tgf = TaskGroup()
                for wi in range(b0, b1):
                    buf, tdims, off_fn, slen = w_tasks[wi]
                    do_wait = (not is_last_b) and (wi == b1 - 1)
                    offs = [off_fn(c) for c in range(N_COLS)]
                    fill_w_pair_1d(buf, tdims, offs, offs, slen, tgf, wait=do_wait)
                if is_last_b:
                    if layer_mode == 3:
                        for c in range(N_COLS):
                            yc_h[c].drain(
                                y_buf,
                                tap=TensorAccessPattern(
                                    tensor_dims=[ACT_TOTAL],
                                    offset=SLOT_SW + c * 1024 + 256,
                                    sizes=[2, N_ROWS, 64],
                                    strides=[64, 128, 1],
                                ),
                                wait=True,
                                group=tgf,
                            )
                    else:
                        drain_linear(y_buf, [ACT_TOTAL], SLOT_ATTN_OUT, tgf)
                tgf.finish()

            if layer_mode == 3:
                return

            tg_o = TaskGroup()
            emit_std_gemm(tg_o, W_ATTN_O, 4, SLOT_ATTN_OUT, SLOT_H1)
            tg_o.finish()

        # Shared tail (Modes 0, 1, 4): rms1 (NORM_1 + H1 -> XQ) -> FFN Gate+Up -> fused FFN_DOWN+res2 (-> yn_buf)
        tg_rms1 = TaskGroup()
        fill_bcast(y_buf, [ACT_TOTAL], SLOT_NORM_1, tg_rms1)
        fill_bcast(y_buf, [ACT_TOTAL], SLOT_H1, tg_rms1, wait=True)
        drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg_rms1)
        tg_rms1.finish()

        ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
        ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
        ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

        # kc=0 (XQ + Gate0 + Up0 -> FFN_XQ+0)
        tg_f0 = TaskGroup()
        fill_bcast(y_buf, [ACT_TOTAL], SLOT_XQ, tg_f0)
        fill_w_pair_1d(w_buf, [W_TOTAL], [ffn_g_base + c * 16 * W_ELEMS for c in range(N_COLS)], [ffn_g_base + (c * 16 + 8) * W_ELEMS for c in range(N_COLS)], 8 * W_ELEMS, tg_f0)
        fill_w_pair_1d(w_buf, [W_TOTAL], [ffn_u_base + c * 16 * W_ELEMS for c in range(N_COLS)], [ffn_u_base + (c * 16 + 8) * W_ELEMS for c in range(N_COLS)], 8 * W_ELEMS, tg_f0, wait=True)
        drain_linear(y_buf, [ACT_TOTAL], SLOT_FFN_XQ, tg_f0)
        tg_f0.finish()

        # kc=1,2 fused (4 MM2S BDs, 2 S2MM BDs)
        tg_f12 = TaskGroup()
        for kc in (1, 2):
            off = kc * 128
            fill_w_pair_1d(w_buf, [W_TOTAL], [ffn_g_base + (off + c * 16) * W_ELEMS for c in range(N_COLS)], [ffn_g_base + (off + c * 16 + 8) * W_ELEMS for c in range(N_COLS)], 8 * W_ELEMS, tg_f12)
            fill_w_pair_1d(w_buf, [W_TOTAL], [ffn_u_base + (off + c * 16) * W_ELEMS for c in range(N_COLS)], [ffn_u_base + (off + c * 16 + 8) * W_ELEMS for c in range(N_COLS)], 8 * W_ELEMS, tg_f12, wait=(kc == 2))
            drain_linear(y_buf, [ACT_TOTAL], SLOT_FFN_XQ + kc * 2560, tg_f12, wait=(kc == 2))
        tg_f12.finish()

        # kc=3
        off3 = 3 * 128
        tg_f3 = TaskGroup()
        fill_w_pair_1d(w_buf, [W_TOTAL], [ffn_g_base + (off3 + c * 16) * W_ELEMS for c in range(N_COLS)], [ffn_g_base + (off3 + c * 16 + 8) * W_ELEMS for c in range(N_COLS)], 8 * W_ELEMS, tg_f3)
        fill_w_pair_1d(w_buf, [W_TOTAL], [ffn_u_base + (off3 + c * 16) * W_ELEMS for c in range(N_COLS)], [ffn_u_base + (off3 + c * 16 + 8) * W_ELEMS for c in range(N_COLS)], 8 * W_ELEMS, tg_f3, wait=True)
        drain_linear(y_buf, [ACT_TOTAL], SLOT_FFN_XQ + 3 * 2560, tg_f3)
        tg_f3.finish()

        # Fused FFN_DOWN (packed 4XQ in 1 bcast + all 4 W chunks in 1 BD) + inline residual -> yn_buf
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
        for c in range(N_COLS):
            yc_h[c].drain(
                yn_buf,
                tap=TensorAccessPattern(
                    tensor_dims=[2048],
                    offset=c * 64,
                    sizes=[4, 4, 16],
                    strides=[16, 512, 1],
                ),
                wait=True,
                group=tg_down,
            )
        tg_down.finish()

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w0_prods, *w1_prods, *y_cons])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=1)
    parser.add_argument("--kv-chunks", type=int, default=0)
    opts = parser.parse_args()
    run_design_cli(
        q4layer_lfm2,
        opts,
        device=NPU2(),
        compile_kwargs=lambda o: {
            "layer_mode": o.layer_mode,
            "kv_chunks": o.kv_chunks,
        },
    )


if __name__ == "__main__":
    main()

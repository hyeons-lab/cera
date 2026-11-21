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
SLOT_FFN_XQ = 10240          # 4 * 2560 = 10240 floats (4 K-chunks of 2048 dims)
SLOT_ATTN_OUT = 20480        # 2048 floats (32 * 64 strided Q8 blocks)
SLOT_Q = 24576
SLOT_K = 27136
SLOT_V = 29696
SLOT_SW = 32768              # 8192 + 512 pad
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
        arg_types=[x_pkt_ty, w_pkt_ty, np.int32, np.int32],
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
        arg_types=[w_pkt_ty, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_x_fn = ExternalFunction(
        "q4_lfm2_dec_x_op",
        source_file=str(here / "q4_lfm2_dec_x_op.cc"),
        arg_types=[x_pkt_ty, np.int32, np.int32, np.int32, np.int32],
        compile_flags=oz_flags,
    )
    q4_y_fn = ExternalFunction(
        "q4_lfm2_dec_y_op",
        source_file=str(here / "q4layer_lfm2_core.cc"),
        arg_types=[y_core_ty, np.int32, np.int32],
        compile_flags=core_flags,
    )

    W_L3L2 = [ObjectFifo(w_pkt_ty, depth=2, name=f"W_L3L2_{c}") for c in range(N_COLS)]
    W_L2L1 = [W_L3L2[c].cons().forward(obj_type=w_pkt_ty, depth=2, name=f"W_L2L1_{c}") for c in range(N_COLS)]

    N_X_SHIMS = 8
    X_L3L2 = [ObjectFifo(x_pkt_ty, depth=2, name=f"X_L3L2_{x}") for x in range(N_X_SHIMS)]
    X_L2L1 = [X_L3L2[x].cons().forward(obj_type=x_pkt_ty, depth=2, name=f"X_L2L1_{x}") for x in range(N_X_SHIMS)]

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
            emit(KIND_X, 4, 0, core_id, 0, flags_x, mask)
            emit(KIND_Y, 1, 0, 0, 0, 0, mask)

        def emit_res(update_res, mask):
            emit(KIND_X, 4, 1, core_id, 0, update_res, mask)
            emit(KIND_Y, 1, 2, 0, 0, 0, mask)

        def emit_gemm(n_passes, k_chunks, dst_gate, do_drain, mask):
            emit(KIND_GEMM, n_passes, 0, n_passes, k_chunks, dst_gate | (do_drain << 1) | (r << 2), mask)

        # Stage 1 (Modes 0, 1, 2): Initial RMSNorm + Q8 (save_res=1)
        emit_rms(-1, 0, 1, M012)

        # Mode 0 (Conv, d_conv=2): B, C, X -> conv_wt + conv_state -> ShortConv -> out_proj
        emit_gemm(4, 1, 0, 1, M0)
        emit_gemm(4, 1, 0, 1, M0)
        emit_gemm(4, 1, 0, 1, M0)
        emit(KIND_W, 1, 3, r, 0, 0, M0)   # W_CONV_WT (3 taps)
        emit(KIND_W, 1, 4, r, 0, 0, M0)   # conv_state_in (2 frames)
        emit(KIND_X, 3, 3, r, 0, 0, M0)  # B, C, X -> conv_y + new state
        emit(KIND_Y, 1, 7, 0, 0, 0, M0)   # Quantize Q8(conv_y) in-core & drain -> SLOT_XQ
        emit(KIND_Y, 2, 3, 0, 0, 0, M0)   # Drain conv_state[0..1] -> kvo_buf
        emit_gemm(4, 1, 0, 1, M0)         # OUT_PROJ

        # Modes 1 & 2 (Phase A): Q, K, V -> Q/K Norm + NeoX RoPE -> K/V drain
        emit_gemm(4, 1, 0, 1, M12)        # Q (2048)
        emit_gemm(1, 1, 0, 1, M12)        # K (512)
        emit_gemm(1, 1, 0, 1, M12)        # V (512)
        emit(KIND_W, 1, 1, 0, 0, 0, M12)  # SLOT_ROPE_PKT
        emit(KIND_X, 4, 2, core_id, 0, 0, M12)  # Q norm + RoPE (all 32 cores)
        emit(KIND_X, 2, 2, core_id, 0, 1, M12)  # K norm + RoPE (r==2) & V bf16 (r==3)
        emit(KIND_Y, 1, 5, 0, 0, 0, M12)  # Drain K/V cur -> kvo_buf
        emit(KIND_Y, 1, 8, 0, 0, 0, M2)   # Mode 2: Drain q_bf16 -> SLOT_SW + c*1024

        # Modes 3 & 4: Reload q_bf16 & conditional FA state (+ save res_acc in M4)
        emit(KIND_X, 4, 1, core_id, 0, 2, M4)
        emit(KIND_X, 1, 2, core_id, 0, 2, M34)
        emit(KIND_X, 1, 2, core_id, 0, 3, M34)

        emit(KIND_FA, 1, r, 0, 0, 0, M_FA)  # FA over kv_chunks (+ cur_kv if not M3)
        emit(KIND_Y, 2, 9, 0, 0, 0, M3)     # Mode 3: Drain o_acc + {row_m, row_l} -> SLOT_SW + c*1024 + 256
        emit(KIND_Y, 1, 4, 0, 0, 0, M14)    # Normalize & Q8(o_acc) -> SLOT_ATTN_OUT
        emit_gemm(4, 1, 0, 1, M14)          # O_PROJ

        # Shared tail (Modes 0, 1, 4): Res1 -> FFN_NORM -> FFN(8192) -> Res2
        emit_res(1, M014)
        emit_rms(1, 0, 0, M014)

        for _ in range(4):
            emit_gemm(4, 1, 1, 0, M014)   # FFN_GATE chunk
            emit_gemm(4, 1, 0, 0, M014)   # FFN_UP chunk
            emit(KIND_Y, 1, 6, 4, 0, 0, M014)
            emit_rms(-1, 0, 2, M014)

        emit_gemm(4, 4, 0, 1, M014)       # FFN_DOWN
        emit_res(0, M014)
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

        def core_body(w_fifo, x_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_fa_fn, q4_w_fn, q4_x_fn, q4_y_fn):
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
            i32_255 = arith.constant(i32_ty, 255)
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
                                    dst_gate = arith.andi(a2, i32_1)
                                    do_drain = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_2), i32_0)
                                    last_k = arith.subi(a1, i32_1)
                                    for kc in range_(a1):
                                        is_sub = arith.cmpi(arith.CmpIPredicate.ne, kc, idx0)
                                        add_bit = arith.select(is_sub, i32_1, i32_0)
                                        gflags = arith.ori(add_bit, arith.shli(dst_gate, i32_1))
                                        is_last = arith.cmpi(arith.CmpIPredicate.eq, i32(kc), last_k)
                                        xp = x_fifo.acquire(1)
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
                                            q4_y_fn(yp, i32_1, i32_0)
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
            stack_size=2560,
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
            for x in range(N_X_SHIMS):
                xp_h[x].fill(
                    src_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=tdims, offset=off, sizes=[1, X_ELEMS], strides=[0, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

        def fill_x_xq(slot_off, tg, wait=False):
            for x in range(N_X_SHIMS):
                xp_h[x].fill(
                    y_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[ACT_TOTAL], offset=slot_off, sizes=[32, 20], strides=[64, 1]
                    ),
                    wait=wait,
                    group=tg,
                )

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

        def run_rms_to_xq(norm_off, x_slot, xq_slot):
            tg = TaskGroup()
            if norm_off is not None:
                fill_w_broadcast(y_buf, [ACT_TOTAL], norm_off, tg, wait=True)
            for k in range(4):
                fill_x_broadcast(y_buf, [ACT_TOTAL], x_slot + k * 512, tg, wait=(k == 3))
            drain_linear(y_buf, [ACT_TOTAL], xq_slot, tg)
            tg.finish()

        def run_res_add(x_slot, dst_buf, dst_tdims, dst_off):
            tg = TaskGroup()
            for k in range(4):
                fill_x_broadcast(y_buf, [ACT_TOTAL], x_slot + k * 512, tg, wait=(k == 3))
            drain_linear(dst_buf, dst_tdims, dst_off, tg)
            tg.finish()

        def emit_gemm_passes_into(
            tg,
            w_base,
            n_passes,
            xq_base,
            out_slot,
            do_drain=True,
            wait_wx=True,
        ):
            fill_x_xq(xq_base, tg, wait=wait_wx)
            if n_passes == 4:
                fill_w_tap(
                    w_buf,
                    [W_TOTAL],
                    [w_base + c * 4 * W_ELEMS for c in range(N_COLS)],
                    [4, 36, 512],
                    [32 * W_ELEMS, 512, 1],
                    tg,
                    wait=wait_wx,
                )
            else:
                for p in range(n_passes):
                    is_last_p = p == n_passes - 1
                    fill_w_1d(
                        w_buf,
                        [W_TOTAL],
                        [w_base + (p * 32 + c * 4) * W_ELEMS for c in range(N_COLS)],
                        4 * W_ELEMS,
                        tg,
                        wait=(wait_wx and is_last_p),
                    )
            if do_drain:
                for c in range(N_COLS):
                    yc_h[c].drain(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=out_slot + c * 64,
                            sizes=[4, 4, 16],
                            strides=[16, 512, 1],
                        ),
                        wait=wait_wx,
                        group=tg,
                    )

        def run_gemm_passes(w_base, n_passes, k_chunks, xq_base, out_slot, do_drain=True):
            if k_chunks == 1:
                tg = TaskGroup()
                emit_gemm_passes_into(
                    tg,
                    w_base,
                    n_passes,
                    xq_base,
                    out_slot,
                    do_drain=do_drain,
                    wait_wx=True,
                )
                tg.finish()
            else:
                tg = TaskGroup()
                for kc in range(k_chunks):
                    is_last_kc = kc == k_chunks - 1
                    is_last_step = is_last_kc and do_drain
                    fill_x_xq(xq_base + kc * 2560, tg, wait=(is_last_kc and (not is_last_step)))
                    fill_w_tap(
                        w_buf,
                        [W_TOTAL],
                        [w_base + (c * 4 * k_chunks + kc) * W_ELEMS for c in range(N_COLS)],
                        [n_passes, 4, 9, 512],
                        [32 * k_chunks * W_ELEMS, k_chunks * W_ELEMS, 512, 1],
                        tg,
                        wait=(is_last_kc and (not is_last_step)),
                    )
                if do_drain:
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

        # Peeled step 0 (+ initial RMSNorm+Q8 in Modes 0, 1, 2)
        tg0 = TaskGroup()
        fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_NORM_0, tg0, wait=True)
        if layer_mode in (0, 1, 2):
            for k in range(4):
                fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + k * 512, tg0, wait=(k == 3))
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg0)
        tg0.finish()

        if layer_mode == 0:
            # Merge Conv B, C, X IN_PROJ (each 1 xp + 1 wp + 1 yc -> 3 xp + 3 wp + 3 yc <= 4/ch)
            tg_bcx = TaskGroup()
            emit_gemm_passes_into(tg_bcx, W_CONV_IN_B, 4, SLOT_XQ, SLOT_CONV_B, wait_wx=False)
            emit_gemm_passes_into(tg_bcx, W_CONV_IN_C, 4, SLOT_XQ, SLOT_CONV_C, wait_wx=False)
            emit_gemm_passes_into(tg_bcx, W_CONV_IN_X, 4, SLOT_XQ, SLOT_CONV_X, wait_wx=True)
            tg_bcx.finish()

            tgc = TaskGroup()
            fill_w_1d(w_buf, [W_TOTAL], [W_CONV_WT + c * W_ELEMS for c in range(N_COLS)], W_ELEMS, tgc)
            fill_w_1d(kv_buf, [KV_TOTAL], [c * W_ELEMS for c in range(N_COLS)], W_ELEMS, tgc, wait=True)
            for slot_bcx in [SLOT_CONV_B, SLOT_CONV_C, SLOT_CONV_X]:
                for x in range(N_X_SHIMS):
                    xp_h[x].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL], offset=slot_bcx + x * 256, sizes=[1, X_ELEMS], strides=[0, 1]
                        ),
                        wait=(slot_bcx == SLOT_CONV_X),
                        group=tgc,
                    )
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tgc, wait=False)
            # Drain rolling state (2 frames x 256 floats/col -> kvo_buf) in same TG (3 yc <= 4/ch)
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

            run_gemm_passes(W_CONV_OUT, 4, 1, SLOT_XQ, SLOT_RES_TMP)
        else:
            if layer_mode in (1, 2):
                # Merge Q, K, V projections (3 xp + 3 wp + 3 yc <= 4/ch)
                tg_qkv = TaskGroup()
                emit_gemm_passes_into(tg_qkv, W_ATTN_Q, 4, SLOT_XQ, SLOT_Q, wait_wx=False)
                emit_gemm_passes_into(tg_qkv, W_ATTN_K, 1, SLOT_XQ, SLOT_K, wait_wx=False)
                emit_gemm_passes_into(tg_qkv, W_ATTN_V, 1, SLOT_XQ, SLOT_V, wait_wx=True)
                tg_qkv.finish()

                # Merge RoPE packet fill (1 wp) + Q norm/RoPE 3/4 (3 xp)
                tgr_q = TaskGroup()
                fill_w_broadcast(y_buf, [ACT_TOTAL], SLOT_ROPE_PKT, tgr_q, wait=True)
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
                    tg_qd = TaskGroup()
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
                            group=tg_qd,
                        )
                    tg_qd.finish()
                    return

            if layer_mode in (3, 4):
                if layer_mode == 4:
                    tg_res = TaskGroup()
                    for k in range(4):
                        fill_x_broadcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN + k * 512, tg_res, wait=(k == 3))
                    tg_res.finish()
                tg_lq = TaskGroup()
                for c in range(N_COLS):
                    xp_h[c].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_SW + c * 1024,
                            sizes=[1, X_ELEMS],
                            strides=[0, 1],
                        ),
                        wait=True,
                        group=tg_lq,
                    )
                tg_lq.finish()
                tg_lfs = TaskGroup()
                for c in range(N_COLS):
                    xp_h[c].fill(
                        y_buf,
                        tap=TensorAccessPattern(
                            tensor_dims=[ACT_TOTAL],
                            offset=SLOT_SW + c * 1024 + 256,
                            sizes=[1, X_ELEMS],
                            strides=[0, 1],
                        ),
                        wait=True,
                        group=tg_lfs,
                    )
                tg_lfs.finish()

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

            run_gemm_passes(W_ATTN_O, 4, 1, SLOT_ATTN_OUT, SLOT_RES_TMP)

        # Shared tail (no post_attn_norm / post_ffn_norm)
        run_res_add(SLOT_RES_TMP, y_buf, [ACT_TOTAL], SLOT_H1)
        run_rms_to_xq(SLOT_NORM_1, SLOT_H1, SLOT_XQ)

        ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
        ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
        ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

        tile_off = 0
        for kc in range(4):
            # Merge FFN GATE (1 xp + 1 wp) + FFN UP (1 xp + 1 wp) + drain SLOT_SW (1 yc)
            tg_ffn = TaskGroup()
            emit_gemm_passes_into(
                tg_ffn,
                ffn_g_base + tile_off * W_ELEMS,
                4,
                SLOT_XQ,
                0,
                do_drain=False,
                wait_wx=False,
            )
            emit_gemm_passes_into(
                tg_ffn,
                ffn_u_base + tile_off * W_ELEMS,
                4,
                SLOT_XQ,
                0,
                do_drain=False,
                wait_wx=True,
            )
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
                    group=tg_ffn,
                )
            tg_ffn.finish()
            run_rms_to_xq(None, SLOT_SW + kc * 2048, SLOT_FFN_XQ + kc * 2560)
            tile_off += 4 * 32

        run_gemm_passes(ffn_d_base, 4, 4, SLOT_FFN_XQ, SLOT_RES_TMP)
        run_res_add(SLOT_RES_TMP, yn_buf, [2048], 0)

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *x_prods, *y_cons])
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

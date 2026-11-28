# Fused B=1 Q4_0 LFM2 / LFM2.5 decode layer (32 cores: 4 rows x 8 cols, npu2).
# Modes via g_params[6]:
#   0 = ShortConv (d_conv=2) + FFN (8192) decode
#   1 = GQA (32 Q heads, 8 KV heads, head_dim=64) + QK-norm + NeoX RoPE + FFN (8192) decode
#   2 = Attn Phase A (>32K: RMS + QKV + RoPE -> kvo_buf + q_bf16 + zero fa_state in SLOT_SW)
#   3 = Attn FA Partial (>32K: Reload q_bf16 + fa_state -> FA -> Drain fa_state)
#   4 = Attn FA Final (>32K: Reload hidden + q_bf16 + fa_state -> FA + cur_kv -> O + FFN)
#   5 = ShortConv without FFN (ends at CONV_OUT + residual -> yn_buf)
#   6 = GQA without FFN (ends at O + residual -> yn_buf)
#   7 = Attn FA Final without FFN (ends at O + residual -> yn_buf)

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
N_KV_HEADS = 8
MAX_KV_CHUNKS = 1024
KV_CHUNK_F32 = 2304
KV_PKT_CHUNKS = 2
MAX_KV_PKTS = MAX_KV_CHUNKS // KV_PKT_CHUNKS  # 512 packets of 4608 f32 (64 tokens each)

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

KV_TOTAL = MAX_KV_CHUNKS * N_KV_HEADS * KV_CHUNK_F32
KV_OUT_TOTAL = 8 * W_ELEMS


VALID_MODES = (0, 1, 2, 3, 4, 5, 6, 7)


def _validate_compile_params(layer_mode: int, kv_chunks: int) -> None:
    validate_compile_params(
        layer_mode,
        kv_chunks,
        valid_modes=VALID_MODES,
        nonzero_kv_modes=(3,),
        max_kv_chunks=MAX_KV_PKTS,
    )


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
    _validate_compile_params(layer_mode, kv_chunks)
    here = Path(__file__).resolve().parent
    cxx_inc = cxx_header_path()

    w_ty = np.ndarray[(W_TOTAL,), np.dtype[np.float32]]
    y_ty = np.ndarray[(ACT_TOTAL,), np.dtype[np.float32]]
    yn_ty = np.ndarray[(2048,), np.dtype[np.float32]]
    kv_ty = np.ndarray[(KV_TOTAL,), np.dtype[np.float32]]
    kvo_ty = np.ndarray[(KV_OUT_TOTAL,), np.dtype[np.float32]]

    w_pkt_ty = np.ndarray[(W_ELEMS,), np.dtype[np.float32]]
    kv_blk_ty = np.ndarray[(1024,), np.dtype[np.float32]]
    x_pkt_ty = np.ndarray[(X_ELEMS,), np.dtype[np.float32]]
    y_core_ty = np.ndarray[(Y_CORE_ELEMS,), np.dtype[np.float32]]
    y_col_ty = np.ndarray[(Y_COL_ELEMS,), np.dtype[np.float32]]

    common_flags = ["-I", str(here), "-I", str(cxx_inc)]
    gemm_flags = common_flags + ["-O3"]
    core_flags = common_flags + ["-O2", "-fno-unroll-loops"]
    oz_flags = common_flags + ["-Oz", "-fno-unroll-loops"]
    fa_flags = common_flags + ["-O2", "-fno-unroll-loops", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1"]
    q4_gemm_fn = ExternalFunction(
        "q4_lfm2_dec_gemm_op",
        source_file=str(here / "q4_lfm2_dec_gemm_op.cc"),
        arg_types=[w_pkt_ty, np.int32, np.int32],
        compile_flags=gemm_flags,
    )
    q4_fa_fn = ExternalFunction(
        "q4_lfm2_dec_fa_op",
        source_file=str(here / "q4_lfm2_dec_fa_op.cc"),
        arg_types=[kv_blk_ty, np.int32, np.int32, np.int32],
        compile_flags=fa_flags,
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

    W_L3L2 = [ObjectFifo(w_pkt_ty, depth=4, name=f"W_L3L2_{c}") for c in range(N_COLS)]
    W_L2L1 = [
        W_L3L2[c].cons().forward(
            obj_type=w_pkt_ty,
            depth=2,
            name=f"W_L2L1_{c}",
        )
        for c in range(N_COLS)
    ]
    KV_L3L2 = [ObjectFifo(kv_blk_ty, depth=2, name=f"KV_L3L2_{c}") for c in range(N_COLS)]
    KV_L2L1 = [KV_L3L2[c].cons().forward(obj_type=kv_blk_ty, depth=1, name=f"KV_L2L1_{c}") for c in range(N_COLS)]

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
    M5 = 32
    M6 = 64
    M7 = 128
    M05 = M0 | M5
    M01 = M0 | M1
    M012 = M0 | M1 | M2 | M5 | M6
    M12 = M1 | M2 | M6
    M34 = M3 | M4 | M7
    M47 = M4 | M7
    M14 = M1 | M4 | M6 | M7
    M014 = M0 | M1 | M4
    M_FA = M1 | M3 | M4 | M6 | M7

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

        # Stage 1 (Modes 0, 1, 2, 5, 6): Initial RMSNorm + Q8 (save_res=1); step 0 already loaded norm0+params
        emit_rms(-1, 0, 1, M012)

        # Mode 0 & 5 (Conv, d_conv=2): cached XQ -> fused B/C/X -> conv_wt + conv_state -> ShortConv -> out_proj -> H1/yn
        emit_gemm(4, 1, 0b1011, 0, M05)       # pt_outer=1, load_x=1, mode=2 (B->gate, C->conv_c, X->y_gemm)
        emit(KIND_W, 1, 3, r, 0, 0, M05)      # W_CONV_WT (3 taps)
        emit(KIND_W, 1, 4, r, 0, 0, M05)      # conv_state_in (2 frames)
        emit(KIND_Y, 1, 7, 0, 0, 0, M05)      # In-core Conv + Q8(conv_y) & drain -> SLOT_XQ
        emit(KIND_Y, 2, 3, 0, 0, 0, M05)      # Drain conv_state[0..1] -> kvo_buf
        emit_gemm(4, 1, 0b0010, 2 | (2 << 2), M05) # OUT_PROJ (load_x=1, drain op=11 res+y -> SLOT_H1 / yn_buf)

        # Modes 1, 2, 6 (Phase A): Q, K, V -> Q/K Norm + NeoX RoPE (from SLOT_Q+K+V) -> K/V drain
        emit_gemm(4, 1, 0b0010, 2, M12)       # Q (2048, load_x=1, do_drain=1)
        emit_gemm(1, 1, 0b0000, 2, M12)       # K (512, reuse XQ)
        emit_gemm(1, 1, 0b0000, 2, M12)       # V (512, reuse XQ)
        emit(KIND_W, 1, 8, core_id, 0, 0, M12)# Q + K norm/RoPE + V bf16 from packed SLOT_Q..V
        emit(KIND_Y, 1, 5, 0, 0, 0, M12)      # Drain K/V cur -> kvo_buf
        emit(KIND_Y, 1, 8, 0, 0, 0, M2)       # Mode 2: Drain q_bf16 -> SLOT_SW + c*1024
        emit(KIND_Y, 2, 9, 0, 0, 0, M2)       # Mode 2: Reset+drain fa_state -> SLOT_SW + c*1024 + 256

        # Modes 3, 4, 7: Stage B setup (save res_acc in M4/M7 + reload q_bf16 & fa_state)
        emit(KIND_W, 1, 7, core_id, 0, 2, M47) # res_acc <- SLOT_HIDDEN_IN
        emit(KIND_W, 1, 10, core_id, 0, 0, M34) # q_bf16 <- SLOT_SW + c*1024
        emit(KIND_W, 1, 11, core_id, 0, 0, M34) # o_acc + row_m/l <- SLOT_SW + c*1024 + 256

        emit(KIND_FA, 1, r, 0, 0, 0, M_FA)    # FA over kv_chunks (+ cur_kv if not M3)
        emit(KIND_Y, 2, 9, 0, 0, 0, M3)       # Mode 3: Drain o_acc + {row_m, row_l} -> SLOT_SW + c*1024 + 256
        emit(KIND_Y, 1, 4, 0, 0, 0, M14)      # Normalize & Q8(o_acc) -> SLOT_ATTN_OUT
        emit_gemm(4, 1, 0b0010, 2 | (2 << 2), M14) # O_PROJ (load_x=1, drain op=11 res+y -> SLOT_H1)

        # Shared tail (Modes 0, 1, 4): load norm_1 + rms1(save_res=1) -> fused FFN -> down(res+y -> yn_buf)
        emit(KIND_W, 1, 0, core_id, 0, 0, M014) # SLOT_NORM_1
        emit_rms(-1, 0, 1, M014)

        for g_slot in range(4):
            load_x = 1 if g_slot == 0 else 0
            # Gate chunks 0..3 cached in ffn_gate + dead xq_buf[1..4]: pt_outer=1, dst_gate=1, do_drain=0
            emit_gemm(4, 1, (load_x << 1) | 1, 1 | (g_slot << 4), M014)
        for g_slot in range(4):
            # Up chunks 0..3 + SiLU*Gate + Q8 drain: pt_outer=1, do_drain=1, drain_q8=1 (op=6)
            emit_gemm(4, 1, 1, 2 | (1 << 2) | (g_slot << 4), M014)

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
        half2_val = r >> 1

        def core_body(w_fifo, kv_fifo, y_fifo, sched_buf, q4_gemm_fn, q4_fa_fn, q4_w_fn, q4_y_fn):
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
            my_h2 = arith.constant(i32_ty, half2_val)
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
                                is_bcx = arith.cmpi(arith.CmpIPredicate.eq, mode_sel, i32_2)
                                n_ghalf = arith.select(is_bcx, idx3, idx1)
                                dst_gate = arith.andi(a2, i32_1)
                                do_drain = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(a2, i32_2), i32_0)
                                drain_sel = arith.andi(arith.shrui(a2, i32_2), i32_3)
                                g_slot = arith.andi(arith.shrui(a2, i32_4), i32_3)
                                drain_q8 = arith.cmpi(arith.CmpIPredicate.eq, drain_sel, i32_1)
                                drain_res = arith.cmpi(arith.CmpIPredicate.eq, drain_sel, i32_2)
                                y_drain_op = arith.select(drain_q8, i32_6, arith.select(drain_res, i32_11, i32_1))
                                y_drain_arg = arith.select(drain_q8, g_slot, i32_0)
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
                                                pt_ok = arith.cmpi(arith.CmpIPredicate.eq, cur_pt, half_r)
                                                for h2 in range_(idx2):
                                                    wp = w_fifo.acquire(1)
                                                    is_mine = arith.andi(
                                                        pt_ok,
                                                        arith.cmpi(arith.CmpIPredicate.eq, i32(h2), my_h2),
                                                    )
                                                    with if_(is_mine, hasElse=False):
                                                        q4_gemm_fn(wp, cur_p, gflags)
                                                    w_fifo.release(1)
                                with if_(do_drain, hasElse=False):
                                    yp = y_fifo.acquire(1)
                                    q4_y_fn(yp, y_drain_op, y_drain_arg)
                                    y_fifo.release(1)
                            with else_(if_g):
                                for kc in range_(kv_ch):
                                    is_f = arith.cmpi(arith.CmpIPredicate.eq, kc, idx0)
                                    for sub in range_(idx2):
                                        is_f_sub = arith.andi(is_f, arith.cmpi(arith.CmpIPredicate.eq, sub, idx0))
                                        m_f = arith.select(is_f_sub, i32_2, i32_0)
                                        cidx = arith.addi(arith.shli(i32(kc), i32_1), i32(sub))
                                        for sv in range_(idx2):
                                            kvb = kv_fifo.acquire(1)
                                            q4_fa_fn(
                                                kvb,
                                                my_r,
                                                cidx,
                                                arith.ori(m_f, arith.shli(i32(sv), i32_2)),
                                            )
                                            kv_fifo.release(1)
                                has_kv = arith.cmpi(arith.CmpIPredicate.sgt, kv_ch, i32_0)
                                use_cur = arith.cmpi(arith.CmpIPredicate.ne, lmode, i32_3)
                                with if_(use_cur, hasElse=False):
                                    cb = kv_fifo.acquire(1)
                                    q4_fa_fn(cb, my_r, i32_0, arith.select(has_kv, i32_1, i32_3))
                                    kv_fifo.release(1)

        return core_body

    workers = [
        Worker(
            make_worker_body(c, r),
            fn_args=[
                W_L2L1[c].cons(),
                KV_L2L1[c].cons(),
                Y_L1L2[c][r].prod(),
                sched_bufs[c][r],
                q4_gemm_fn,
                q4_fa_fn,
                q4_w_fn,
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
    y_cons = [Y_L2L3[c].cons() for c in range(N_COLS)]
    kv_prods = [KV_L3L2[c].prod() for c in range(N_COLS)]

    def seq_fn(w_buf, y_buf, yn_buf, kv_buf, kvo_buf, *fifos):
        wp_h = fifos[0:N_COLS]
        yc_h = fifos[N_COLS : 2 * N_COLS]
        kvp_h = fifos[2 * N_COLS : 3 * N_COLS]

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

        def fill_bcast(src_buf, tdims, off, tg, wait=False):
            fill_w_tap(src_buf, tdims, [off] * N_COLS, [1, W_ELEMS], [0, 1], tg, wait=wait)

        def fill_per_col_bcast(src_buf, tdims, off_per_col, tg, wait=False):
            fill_w_tap(src_buf, tdims, off_per_col, [1, W_ELEMS], [0, 1], tg, wait=wait)

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

        def drain_strided(out_slot, tg, wait=True, dst_buf=None, dst_tdims=None):
            db = y_buf if dst_buf is None else dst_buf
            td = [ACT_TOTAL] if dst_tdims is None else dst_tdims
            for c in range(N_COLS):
                yc_h[c].drain(
                    db,
                    tap=TensorAccessPattern(
                        tensor_dims=td,
                        offset=out_slot + c * 64,
                        sizes=[4, 4, 16],
                        strides=[16, 512, 1],
                    ),
                    wait=wait,
                    group=tg,
                )

        def emit_std_gemm(tg, w_base, n_passes, xq_slot, out_slot, load_x=True, do_drain=True, wait=True, dst_buf=None, dst_tdims=None):
            if load_x:
                fill_bcast(y_buf, [ACT_TOTAL], xq_slot, tg)
            for p in range(n_passes):
                fill_w_tap(
                    w_buf,
                    [W_TOTAL],
                    [w_base + (c * 4 + p * 32) * W_ELEMS for c in range(N_COLS)],
                    [2, 2, 18, 256],
                    [W_ELEMS, 2 * W_ELEMS, 256, 1],
                    tg,
                    wait=(wait and not do_drain and p == n_passes - 1),
                )
            if do_drain:
                drain_strided(out_slot, tg, wait=wait, dst_buf=dst_buf, dst_tdims=dst_tdims)

        # Peeled step 0 (+ initial RMSNorm+Q8 in Modes 0, 1, 2, 5, 6)
        tg0_slot = (
            SLOT_NORM_1 if layer_mode == 3
            else (SLOT_ROPE_PKT if layer_mode in (4, 7) else SLOT_NORM_0)
        )
        tg0 = TaskGroup()
        fill_bcast(y_buf, [ACT_TOTAL], tg0_slot, tg0, wait=(layer_mode not in (0, 1, 2, 5, 6)))
        if layer_mode in (0, 1, 2, 5, 6):
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN, tg0, wait=True)
            drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg0)
        tg0.finish()

        if layer_mode in (0, 5):
            # Merged Conv: XQ + fused B/C/X + W_CONV_WT + kv_buf -> Conv + Q8 -> SLOT_XQ + conv_state[0..1]
            tgc = TaskGroup()
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_XQ, tgc)
            for p in range(3):
                fill_w_tap(
                    w_buf,
                    [W_TOTAL],
                    [W_CONV_IN_B + (c * 16 + p * 128) * W_ELEMS for c in range(N_COLS)],
                    [8, 2, 18, 256],
                    [W_ELEMS, 8 * W_ELEMS, 256, 1],
                    tgc,
                )
            fill_per_col_bcast(
                w_buf,
                [W_TOTAL],
                [W_CONV_WT + c * W_ELEMS for c in range(N_COLS)],
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
            if layer_mode == 5:
                emit_std_gemm(tg_cout, W_CONV_OUT, 4, SLOT_XQ, 0, wait=True, dst_buf=yn_buf, dst_tdims=[2048])
                tg_cout.finish()
                return
            emit_std_gemm(tg_cout, W_CONV_OUT, 4, SLOT_XQ, SLOT_H1, wait=True)
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_NORM_1, tg_cout, wait=True)
            tg_cout.finish()
        else:
            if layer_mode in (1, 2, 6):
                # Merge XQ + Q + (K+V) in 1 TaskGroup (3 MM2S BDs <= 4, 3 S2MM BDs <= 4)
                tg_qkv = TaskGroup()
                emit_std_gemm(tg_qkv, W_ATTN_Q, 4, SLOT_XQ, SLOT_Q, load_x=True, do_drain=True, wait=False)
                for p in range(2):
                    fill_w_tap(
                        w_buf,
                        [W_TOTAL],
                        [W_ATTN_K + (c * 4 + p * 32) * W_ELEMS for c in range(N_COLS)],
                        [2, 2, 18, 256],
                        [W_ELEMS, 2 * W_ELEMS, 256, 1],
                        tg_qkv,
                        wait=False,
                    )
                drain_strided(SLOT_K, tg_qkv, wait=False)
                drain_strided(SLOT_V, tg_qkv, wait=True)
                tg_qkv.finish()

                # Q[2048] + (K+V)[1024] in 1 bcast (rope+q/k norms already loaded in tg0!) -> kvo_buf (+ Mode 2 q_bf16 + fa_state)
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
                            wait=False,
                            group=tgr,
                        )
                        yc_h[c].drain(
                            y_buf,
                            tap=TensorAccessPattern(
                                tensor_dims=[ACT_TOTAL],
                                offset=SLOT_SW + c * 1024 + 256,
                                sizes=[2, N_ROWS, 64],
                                strides=[64, 128, 1],
                            ),
                            wait=True,
                            group=tgr,
                        )
                    tgr.finish()
                    return
                tgr.finish()

            if layer_mode in (3, 4, 7):
                tg_b = TaskGroup()
                if layer_mode in (4, 7):
                    fill_bcast(y_buf, [ACT_TOTAL], SLOT_HIDDEN_IN, tg_b)
                fill_per_col_bcast(y_buf, [ACT_TOTAL], [SLOT_SW + c * 1024 for c in range(N_COLS)], tg_b)
                fill_per_col_bcast(y_buf, [ACT_TOTAL], [SLOT_SW + c * 1024 + 256 for c in range(N_COLS)], tg_b, wait=True)
                tg_b.finish()

            kv_tasks = []
            for chk0 in range(0, kv_chunks, 64):
                cnt = min(64, kv_chunks - chk0)
                kv_tasks.append((
                    kv_buf,
                    [KV_TOTAL],
                    lambda c, s=chk0: (c * MAX_KV_PKTS + s) * W_ELEMS,
                    [cnt, 2, 4, 512],
                    [W_ELEMS, 2304, 512, 1],
                ))
            if layer_mode != 3:
                kv_tasks.append((
                    kvo_buf,
                    [KV_OUT_TOTAL],
                    lambda c: c * 256,
                    [1, 1, 1, 1024],
                    [0, 0, 0, 1],
                ))

            for b0 in range(0, len(kv_tasks), 4):
                b1 = min(b0 + 4, len(kv_tasks))
                is_last_b = b1 == len(kv_tasks)
                tgf = TaskGroup()
                for wi in range(b0, b1):
                    buf, tdims, off_fn, sz, st = kv_tasks[wi]
                    do_wait = (not is_last_b) and (wi == b1 - 1)
                    if wi == len(kv_tasks) - 1 and layer_mode == 3:
                        do_wait = True
                    for c in range(N_COLS):
                        kvp_h[c].fill(
                            buf,
                            tap=TensorAccessPattern(tensor_dims=tdims, offset=off_fn(c), sizes=sz, strides=st),
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

            tg_o = TaskGroup()
            if layer_mode in (6, 7):
                emit_std_gemm(tg_o, W_ATTN_O, 4, SLOT_ATTN_OUT, 0, wait=True, dst_buf=yn_buf, dst_tdims=[2048])
                tg_o.finish()
                return
            emit_std_gemm(tg_o, W_ATTN_O, 4, SLOT_ATTN_OUT, SLOT_H1, wait=True)
            fill_bcast(y_buf, [ACT_TOTAL], SLOT_NORM_1, tg_o, wait=True)
            tg_o.finish()

        # Shared tail (Modes 0, 1, 4): rms1 (preloaded NORM_1 + H1 -> XQ) -> unified FFN Gate+Up -> fused FFN_DOWN+res2 (-> yn_buf)
        tg_rms1 = TaskGroup()
        fill_bcast(y_buf, [ACT_TOTAL], SLOT_H1, tg_rms1, wait=True)
        drain_linear(y_buf, [ACT_TOTAL], SLOT_XQ, tg_rms1)
        tg_rms1.finish()

        ffn_g_base = W_CONV_FFN_GATE if layer_mode == 0 else W_ATTN_FFN_GATE
        ffn_u_base = W_CONV_FFN_UP if layer_mode == 0 else W_ATTN_FFN_UP
        ffn_d_base = W_CONV_FFN_DOWN if layer_mode == 0 else W_ATTN_FFN_DOWN

        # Unified FFN Gate+Up across kc=0..3 split into Gate + Up TaskGroups (<=5 MM2S tasks/TG) -> SLOT_FFN_XQ
        tg_ffn_g = TaskGroup()
        fill_bcast(y_buf, [ACT_TOTAL], SLOT_XQ, tg_ffn_g)
        for p in range(4):
            fill_w_tap(
                w_buf,
                [W_TOTAL],
                [ffn_g_base + (c * 16 + p * 128) * W_ELEMS for c in range(N_COLS)],
                [8, 2, 18, 256],
                [W_ELEMS, 8 * W_ELEMS, 256, 1],
                tg_ffn_g,
                wait=(p == 3),
            )
        tg_ffn_g.finish()

        tg_ffn_u = TaskGroup()
        for p in range(4):
            fill_w_tap(
                w_buf,
                [W_TOTAL],
                [ffn_u_base + (c * 16 + p * 128) * W_ELEMS for c in range(N_COLS)],
                [8, 2, 18, 256],
                [W_ELEMS, 8 * W_ELEMS, 256, 1],
                tg_ffn_u,
                wait=(p == 3),
            )
        for c in range(N_COLS):
            yc_h[c].drain(
                y_buf,
                tap=TensorAccessPattern(
                    tensor_dims=[ACT_TOTAL],
                    offset=SLOT_FFN_XQ + c * 256,
                    sizes=[4, 256],
                    strides=[2048, 1],
                ),
                wait=True,
                group=tg_ffn_u,
            )
        tg_ffn_u.finish()

        # Fused FFN_DOWN (packed 4XQ in 1 bcast + all 4 W chunks in 1 BD) + inline residual -> yn_buf
        tg_down = TaskGroup()
        fill_w_tap(
            y_buf,
            [ACT_TOTAL],
            [SLOT_FFN_XQ] * N_COLS,
            [4, 32, 36],
            [2048, 64, 1],
            tg_down,
        )
        for p in range(4):
            fill_w_tap(
                w_buf,
                [W_TOTAL],
                [ffn_d_base + (c * 16 + p * 128) * W_ELEMS for c in range(N_COLS)],
                [8, 2, 18, 256],
                [W_ELEMS, 8 * W_ELEMS, 256, 1],
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

    rt = Runtime(seq_fn, fn_args=[w_ty, y_ty, yn_ty, kv_ty, kvo_ty, *w_prods, *y_cons, *kv_prods])
    return Program(get_current_device(), rt, workers).resolve_program()


def main():
    parser = argparse.ArgumentParser()
    add_compile_args(parser)
    parser.add_argument("--layer-mode", type=int, default=1, choices=VALID_MODES)
    parser.add_argument("--kv-chunks", type=int, default=0)
    opts = parser.parse_args()
    _validate_compile_params(opts.layer_mode, opts.kv_chunks)
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

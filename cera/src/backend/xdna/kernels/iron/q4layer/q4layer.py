# Fused Q4 transformer-layer design for cera/src/backend/xdna/kernels (cera XDNA backend).
#
# One NPU run executes a full Llama-3.2-1B transformer layer under a single
# xclbin, because switching xclbins between GEMVs costs ~800 us (see
# docs/XDNA_PHASE0_RESULTS.md section 5). 16 workers, one per (column, shim
# channel) pair. Activations live in f32 slots in DDR; each stage's input
# slot is broadcast to every worker, quantized on the core (q4layer.cc
# q4_step), and each worker drains its output rows into the next slot, with
# the task group finished before the next stage starts.
#
# Stage chaining executes the full transformer layer on-NPU in 5 TaskGroups
# with at most 2 MM2S BDs per channel per group:
# 1. q+k+v read slot 0 (RMSNorm + quantize, saving worker's 128 residual floats
#    and 64 RoPE cos/sin floats in L1) and stream per-worker contiguous (q, k, v)
#    weights, applying interleaved RoPE to q (-> slot 1) and k (-> slot 2) and
#    draining v (-> slot 3).
# 2. 16-token block online-softmax GQA attention streams Q4LAYER_KV_CHUNKS
#    (1..9) cached KV chunks plus the current (k, v) token and writes its
#    2048-float output to slot 1 + ATTN_OUT.
# 3. o reads slot 1 + ATTN_OUT, adds the L1-saved slot 0 residual, and drains
#    post-attn hidden h1 into slot 4.
# 4. gate+up read slot 4 (FFN RMSNorm + quantize, saving worker's 128 h1
#    residual floats in L1), stream per-worker contiguous (gate, up) weights,
#    keep worker's 512 gate rows in L1, fuse SiLU(gate) * up on-core during up,
#    and drain 8192 floats of SiLU(gate) * up into slot 5.
# 5. down reads slot 5 in one contiguous 2*TILE_F32 DMA, runs split-K 8192->2048
#    GEMV adding the L1-saved h1 residual, and drains directly into y_next[0..2048].
import argparse
import os as _os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects import arith
from aie.extras import types as T
from aie.helpers.dialects.scf import else_, if_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import ExternalFunction, In, Out
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import get_current_device
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

ROWS = 16
QBLK = 32
NUM_CHANNELS = 2
CHUNK_K = 2048  # columns per streamed weight-tile chunk (split-K)
TILE_F32 = 4608  # one stream element: 18432 bytes viewed as f32
ACT_CHUNK = TILE_F32  # f32 activations quantized per stream element
SILU_HALF = TILE_F32 // 2  # f32 words per half-element (K/V)
ACT_SLOT = 8192  # f32 stride of one activation slot in DDR (max K)
TSTRIDE = 8  # tile/chunk encoding stride in q4_step's t (see q4layer.cc)

NORM_W = 2048
ROPE_CS = 4096
ATTN_OUT = 2048
ATTN_CHUNK_T = (SILU_HALF * 2) // 64
ATTN_CHUNKS = 9
N_SLOTS = 8
KV_CACHE_BASE = N_SLOTS * ACT_SLOT
KV_HEAD_STRIDE = ATTN_CHUNKS * TILE_F32
KV_SLOT = 8 * KV_HEAD_STRIDE


def stage_params(M, K, workers_n):
    rows_pw = M // workers_n
    tiles_pw = rows_pw // ROWS
    return {
        "rows_pw": rows_pw,
        "tiles_pw": tiles_pw,
        "ot": 2,
        "outs_pw": tiles_pw // 2,
        "kchunks": K // CHUNK_K,
    }


@iron.jit
def q4layer(w: In, y: Out, y_next: Out):
    workers_n = get_current_device().cols * NUM_CHANNELS
    if workers_n != 16:
        raise ValueError(
            f"q4layer requires 16 workers (8 AIE2P columns x 2 shim channels), got {workers_n}"
        )

    q_s = stage_params(2048, 2048, workers_n)
    k_s = stage_params(512, 2048, workers_n)
    v_s = stage_params(512, 2048, workers_n)
    o_s = stage_params(2048, 2048, workers_n)
    gate_s = stage_params(8192, 2048, workers_n)
    up_s = stage_params(8192, 2048, workers_n)
    down_s = stage_params(2048, 8192, workers_n)

    # Weight stream layout (f32 words):
    # - Group 0 (q+k+v): per worker [q (8 tiles) | k (2 tiles) | v (2 tiles)] = 12 * TILE_F32
    # - Group 1 (o):     per worker [o (8 tiles)]                             =  8 * TILE_F32
    # - Group 2 (g+u):   per worker [gate (32 tiles) | up (32 tiles)]         = 64 * TILE_F32
    # - Group 3 (down):  per worker [down (8 tiles * 4 kchunks)]              = 32 * TILE_F32
    qkv_pw_w = (q_s["tiles_pw"] + k_s["tiles_pw"] + v_s["tiles_pw"]) * TILE_F32
    o_pw_w = o_s["tiles_pw"] * TILE_F32
    gu_pw_w = (gate_s["tiles_pw"] + up_s["tiles_pw"]) * TILE_F32
    down_pw_w = down_s["tiles_pw"] * down_s["kchunks"] * TILE_F32

    w_qkv_base = 0
    w_o_base = w_qkv_base + workers_n * qkv_pw_w
    w_gu_base = w_o_base + workers_n * o_pw_w
    w_down_base = w_gu_base + workers_n * gu_pw_w
    w_total = w_down_base + workers_n * down_pw_w

    act_total = KV_CACHE_BASE + KV_SLOT
    head_mode = _os.environ.get("Q4LAYER_HEAD") == "1"
    kv_chunks = int(_os.environ.get("Q4LAYER_KV_CHUNKS", str(ATTN_CHUNKS)))
    if not (1 <= kv_chunks <= ATTN_CHUNKS):
        raise ValueError(f"Q4LAYER_KV_CHUNKS must be in 1..={ATTN_CHUNKS}, got {kv_chunks}")

    head_s = stage_params(131072, 2048, workers_n)
    head_per_worker_w = head_s["tiles_pw"] * TILE_F32
    head_w_total = workers_n * head_per_worker_w
    head_y_total = TILE_F32 + 131072

    slot_ty = np.ndarray[(TILE_F32,), np.dtype[np.float32]]
    yel_ty = np.ndarray[(ROWS * 2,), np.dtype[np.float32]]
    w_ty = np.ndarray[(head_w_total if head_mode else w_total,), np.dtype[np.float32]]
    y_ty = np.ndarray[(head_y_total if head_mode else act_total,), np.dtype[np.float32]]

    step = ExternalFunction(
        "q4_step",
        source_file=str(Path(__file__).parent / "q4layer.cc"),
        arg_types=[slot_ty, yel_ty, np.int32, np.int32],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DKMAX={ACT_SLOT}", f"-DACT_CHUNK={ACT_CHUNK}"],
    )

    of_ins = [ObjectFifo(slot_ty, name=f"w_{i}") for i in range(workers_n)]
    of_outs = [ObjectFifo(yel_ty, name=f"y_{i}") for i in range(workers_n)]

    def make_core_body(wid):
        def core_body(of_in, of_out, step):
            # Stage 0 prefix (2 output pairs = 4 tiles), shared by q4layer and q4head.
            for o in range(2):
                y = of_out.acquire(1)
                if o == 0:
                    y0 = y
                    x = of_in.acquire(1)
                    head_flag = arith.bitcast(T.i32(), x[ROPE_CS + 129])
                    is_head = arith.cmpi(
                        arith.CmpIPredicate.ne, head_flag, arith.constant(T.i32(), 0)
                    )
                    skip_kv = arith.bitcast(T.i32(), x[ROPE_CS + 131])
                    extra_kv = arith.subi(
                        arith.constant(T.i32(), ATTN_CHUNKS - 1), skip_kv
                    )
                    step(x, y, ROPE_CS + wid, -100)
                    of_in.release(1)
                for tt in range(2):
                    tile = of_in.acquire(1)
                    step(tile, y, CHUNK_K, tt * TSTRIDE)
                    of_in.release(1)
                of_out.release(1)

            with if_(is_head) as if_op:
                for _ in range_(head_s["outs_pw"] - 2):
                    y = of_out.acquire(1)
                    for tt in range(2):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE)
                        of_in.release(1)
                    of_out.release(1)
            with else_(if_op):
                # Remaining 2 output pairs of q + 1 pair of k + 1 pair of v = 4 pairs.
                for _ in range_(4):
                    y = of_out.acquire(1)
                    for tt in range(2):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE)
                        of_in.release(1)
                    of_out.release(1)

                # GQA attention over (1 + extra_kv) cached KV chunks + current token.
                for o in range(4):
                    y = of_out.acquire(1)
                    if o == 0:
                        kv0 = of_in.acquire(1)
                        step(kv0, y, ATTN_CHUNK_T, -201)
                        of_in.release(1)
                        for _ in range_(extra_kv):
                            kv0 = of_in.acquire(1)
                            step(kv0, y, ATTN_CHUNK_T, -200)
                            of_in.release(1)
                        kv1 = of_in.acquire(1)
                        step(kv1, y, 1, -202)
                        of_in.release(1)
                    step(kv1, y, 0, -110 - o)
                    of_out.release(1)

                # Stage 3 (o): quantize attn_out + add saved slot 0 residual -> slot 4.
                x = of_in.acquire(1)
                step(x, y0, NORM_W, -3)
                of_in.release(1)
                for _ in range_(o_s["outs_pw"]):
                    y = of_out.acquire(1)
                    for tt in range(2):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE)
                        of_in.release(1)
                    of_out.release(1)

                # Stage 4+5 (gate + up -> on-core SiLU(gate)*up -> slot 5).
                x = of_in.acquire(1)
                step(x, y0, wid, -100)
                of_in.release(1)
                for _ in range_(gate_s["tiles_pw"]):
                    tile = of_in.acquire(1)
                    step(tile, y0, CHUNK_K, 0)
                    of_in.release(1)
                for _ in range_(up_s["outs_pw"]):
                    y = of_out.acquire(1)
                    for tt in range(2):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE)
                        of_in.release(1)
                    of_out.release(1)

                # Stage 6 (down): 2 contiguous activation elements + split-K + saved h1 residual -> y_next.
                x0 = of_in.acquire(1)
                step(x0, y0, ACT_CHUNK, -2)
                of_in.release(1)
                x1 = of_in.acquire(1)
                step(x1, y0, 8192 - ACT_CHUNK, -3)
                of_in.release(1)
                for _ in range_(down_s["outs_pw"]):
                    y = of_out.acquire(1)
                    for tt in range(2):
                        for ch in range(down_s["kchunks"]):
                            tile = of_in.acquire(1)
                            step(tile, y, CHUNK_K, tt * TSTRIDE + ch)
                            of_in.release(1)
                    of_out.release(1)

        return core_body

    workers = [
        Worker(make_core_body(i), [of_ins[i].cons(), of_outs[i].prod(), step], stack_size=2560)
        for i in range(workers_n)
    ]

    def tap(total, offset, size):
        return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])

    def tap_silu(total, offset):
        return TensorAccessPattern((1, total), offset, [1, 2, 3, SILU_HALF // 3], [0, ACT_SLOT, SILU_HALF // 3, 1])

    def sequence(w_in, y_buf, y_next, in_prods, out_conses):
        if head_mode:
            tg = TaskGroup()
            for i in range(workers_n):
                in_prods[i].fill(y_buf, tap(head_y_total, 0, TILE_F32), group=tg)
                in_prods[i].fill(w_in, tap(head_w_total, i * head_per_worker_w, head_per_worker_w), group=tg)
            for i in range(workers_n):
                out_conses[i].drain(
                    y_next,
                    tap(head_y_total, TILE_F32 + i * head_s["rows_pw"], head_s["rows_pw"]),
                    wait=True,
                    group=tg,
                )
            tg.finish()
            return

        # Group 1: q + k + v
        tg = TaskGroup()
        for i in range(workers_n):
            in_prods[i].fill(y_buf, tap(act_total, 0, TILE_F32), group=tg)
            in_prods[i].fill(w_in, tap(w_total, w_qkv_base + i * qkv_pw_w, qkv_pw_w), group=tg)
        for i in range(workers_n):
            out_conses[i].drain(y_buf, tap(act_total, 1 * ACT_SLOT + i * q_s["rows_pw"], q_s["rows_pw"]), wait=False, group=tg)
            out_conses[i].drain(y_buf, tap(act_total, 2 * ACT_SLOT + i * k_s["rows_pw"], k_s["rows_pw"]), wait=False, group=tg)
            out_conses[i].drain(y_buf, tap(act_total, 3 * ACT_SLOT + i * v_s["rows_pw"], v_s["rows_pw"]), wait=True, group=tg)
        tg.finish()

        # Group 2: GQA attention (kv_chunks cached chunks + current token)
        tg = TaskGroup()
        for i in range(workers_n):
            kv_h = i // 2
            in_prods[i].fill(
                y_buf,
                tap(act_total, KV_CACHE_BASE + kv_h * KV_HEAD_STRIDE, kv_chunks * TILE_F32),
                group=tg,
            )
            in_prods[i].fill(y_buf, tap_silu(act_total, 2 * ACT_SLOT + kv_h * 64), group=tg)
        for i in range(workers_n):
            out_conses[i].drain(y_buf, tap(act_total, 1 * ACT_SLOT + ATTN_OUT + i * 128, 128), wait=True, group=tg)
        tg.finish()

        # Group 3: o (+ saved slot 0 residual -> slot 4)
        tg = TaskGroup()
        for i in range(workers_n):
            in_prods[i].fill(y_buf, tap(act_total, 1 * ACT_SLOT + ATTN_OUT, TILE_F32), group=tg)
            in_prods[i].fill(w_in, tap(w_total, w_o_base + i * o_pw_w, o_pw_w), group=tg)
        for i in range(workers_n):
            out_conses[i].drain(y_buf, tap(act_total, 4 * ACT_SLOT + i * o_s["rows_pw"], o_s["rows_pw"]), wait=True, group=tg)
        tg.finish()

        # Group 4: gate + up (-> on-core SiLU(gate)*up -> slot 5)
        tg = TaskGroup()
        for i in range(workers_n):
            in_prods[i].fill(y_buf, tap(act_total, 4 * ACT_SLOT, TILE_F32), group=tg)
            in_prods[i].fill(w_in, tap(w_total, w_gu_base + i * gu_pw_w, gu_pw_w), group=tg)
        for i in range(workers_n):
            out_conses[i].drain(y_buf, tap(act_total, 5 * ACT_SLOT + i * up_s["rows_pw"], up_s["rows_pw"]), wait=True, group=tg)
        tg.finish()

        # Group 5: down (+ saved slot 4 residual -> y_next[0..2048])
        tg = TaskGroup()
        for i in range(workers_n):
            in_prods[i].fill(y_buf, tap(act_total, 5 * ACT_SLOT, 2 * TILE_F32), group=tg)
            in_prods[i].fill(w_in, tap(w_total, w_down_base + i * down_pw_w, down_pw_w), group=tg)
        for i in range(workers_n):
            out_conses[i].drain(y_next, tap(act_total, i * down_s["rows_pw"], down_s["rows_pw"]), wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [w_ty, y_ty, y_ty, [of.prod() for of in of_ins], [of.cons() for of in of_outs]])
    return Program(get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p)
    opts = p.parse_args()
    run_design_cli(
        q4layer,
        opts,
        compile_kwargs=lambda o: dict(),
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()

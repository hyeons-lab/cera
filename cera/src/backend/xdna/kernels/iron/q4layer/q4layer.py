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
# Stage chaining executes the full transformer layer on-NPU:
# q/k/v read the hidden slot (RMSNorm + quantize), q and k apply interleaved-pair
# RoPE (RopeType::Norm) on their output tiles before draining, GQA online-softmax
# attention runs across cached KV tokens + the current (k, v) token and writes
# its 2048-float output to slot 1 offset ATTN_OUT, o reads slot 1 offset ATTN_OUT
# and adds the hidden residual into slot 4, gate/up read slot 4 (FFN RMSNorm +
# quantize), and down reads gate and up's slots (SiLU(gate) * up + quantize,
# 8192 wide, split-K) and adds slot 4's residual into slot 7 (layer out).
import argparse
import os as _os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.dialects.scf import else_, if_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, Out
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
SILU_HALF = TILE_F32 // 2  # f32 words per half-element (gate/up or K/V)
ACT_SLOT = 8192  # f32 stride of one activation slot in DDR (max K)
TSTRIDE = 8  # tile/chunk encoding stride in q4_step's t (see q4layer.cc)
MAX_OT = 16

# (input slot, rows M, cols K, output slot); slots: 0 hidden, 1 q (+ attn_out at ATTN_OUT),
# 2 k (+ k_cache at KV_CACHE_OFF), 3 v (+ v_cache at KV_CACHE_OFF),
# 4 post-attn residual (h1), 5 gate, 6 up, 7 layer out (h2).
STAGES = [
    (0, 2048, 2048, 1),  # q (RMSNorm + GEMV + RoPE -> slot 1 [0..2048))
    (0, 512, 2048, 2),  # k (RMSNorm + GEMV + RoPE -> slot 2 [0..512))
    (0, 512, 2048, 3),  # v (RMSNorm + GEMV -> slot 3 [0..512), followed by GQA attention -> slot 1 [2048..4096))
    (1, 2048, 2048, 4),  # o (reads attn_out at slot 1 + ATTN_OUT; + slot 0 residual -> slot 4)
    (4, 8192, 2048, 5),  # gate (FFN RMSNorm on slot 4)
    (4, 8192, 2048, 6),  # up (FFN RMSNorm on slot 4)
    (5, 2048, 8192, 7),  # down (SiLU(slot 5) * slot 6; + slot 4 residual -> slot 7)
]
N_SLOTS = 8
# Slots 0 and 4 are [activation f32 x 2048 | RMSNorm weights f32 x 2048]:
# stages reading slot 0 (q, k, v) or slot 4 (gate, up) stream both in their
# one activation element and the core norms + quantizes in one pass
# (q4layer.cc, t == -100). Slot 0 also carries the 128-float RoPE cos/sin table
# at offset ROPE_CS (4096), which fits inside the same 4608-float element.
NORM_W = 2048
ROPE_CS = 4096
ATTN_OUT = 2048
ATTN_CHUNK_T = (SILU_HALF * 2) // 64
ATTN_CHUNKS = 9
KV_CACHE_BASE = N_SLOTS * ACT_SLOT
KV_HEAD_STRIDE = ATTN_CHUNKS * TILE_F32
KV_SLOT = 8 * KV_HEAD_STRIDE

# Diagnostic: Q4LAYER_STAGES=N builds only the first N stages (the host still
# packs and checks all seven; only the built stages can pass).
STAGES = STAGES[: int(_os.environ.get("Q4LAYER_STAGES", "7"))]
if _os.environ.get("Q4LAYER_SPEC"):
    STAGES = [tuple(int(x) for x in s.split(":")) for s in _os.environ["Q4LAYER_SPEC"].split(";")]
elif _os.environ.get("Q4LAYER_PICK"):
    STAGES = [STAGES[int(i)] for i in _os.environ["Q4LAYER_PICK"].split(",")]


def stage_params(M, K, workers_n):
    # Fixed output grouping OT=2 for every stage (as in q4gemv): each out
    # fifo element holds exactly ROWS*OT outputs and every drain consumes
    # whole elements. A drain smaller than the element leaves the fifo
    # inconsistent and the next stage reads zeros (measured: a stage after
    # a partially drained stage produces nothing).
    rows_pw = M // workers_n
    tiles_pw = rows_pw // ROWS
    return {
        "rows_pw": rows_pw,
        "tiles_pw": tiles_pw,
        "ot": 2,
        "outs_pw": tiles_pw // 2,
        "kchunks": K // CHUNK_K,
        "achunks": (K + ACT_CHUNK - 1) // ACT_CHUNK,
    }


@iron.jit
def q4layer(w: In, y: Out, y_next: Out):
    workers_n = get_current_device().cols * NUM_CHANNELS
    stages = [stage_params(M, K, workers_n) | {"in_slot": i, "M": M, "K": K, "out_slot": o} for (i, M, K, o) in STAGES]

    # Weight stream layout (f32 words): stages in order, per stage each
    # worker's tiles contiguous, tile = TILE_F32 words.
    w_total = 0
    for s in stages:
        s["w_base"] = w_total
        s["per_worker_w"] = s["tiles_pw"] * s["kchunks"] * TILE_F32
        w_total += workers_n * s["per_worker_w"]

    act_total = KV_CACHE_BASE + KV_SLOT
    head_mode = _os.environ.get("Q4LAYER_HEAD") == "1"
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
        compile_flags=[f"-DKMAX={max(s['K'] for s in stages)}", f"-DACT_CHUNK={ACT_CHUNK}"],
    )
    for s in stages:
        s["norm"] = s["out_slot"] in (1, 5)
        s["silu"] = s["in_slot"] == 5
        s["rope"] = s["out_slot"] == 2
        s["res_slot"] = 0 if s["out_slot"] == 4 else 4 if s["out_slot"] == 7 else None
        if s["out_slot"] in (2, 3, 6):
            s["achunks"] = 0
        elif s["silu"]:
            assert s["K"] == 8192, "silu*up stage is the 8192-wide FFN down"
            s["achunks"] = s["K"] // CHUNK_K
        assert not s["norm"] or (s["K"] == 2048 and s["achunks"] == 1), "norm stages are 2048-wide"
        assert s["res_slot"] is None or s["rows_pw"] == 128, "residual stages have 128 rows per worker"
    of_ins = [ObjectFifo(slot_ty, name=f"w_{i}") for i in range(workers_n)]
    of_outs = [ObjectFifo(yel_ty, name=f"y_{i}") for i in range(workers_n)]

    def core_body(of_in, of_out, step):
        s0 = stages[0]
        for o in range(2):
            y = of_out.acquire(1)
            if o == 0:
                x = of_in.acquire(1)
                is_head = x[ROPE_CS + 129] > x[ROPE_CS + 130]
                step(x, y, ROPE_CS, -100)
                of_in.release(1)
            for tt in range(s0["ot"]):
                for ch in range(s0["kchunks"]):
                    tile = of_in.acquire(1)
                    step(tile, y, CHUNK_K, tt * TSTRIDE + ch)
                    of_in.release(1)
            of_out.release(1)
        with if_(is_head) as if_op:
            for _ in range_(head_s["outs_pw"] - 2):
                y = of_out.acquire(1)
                for tt in range(head_s["ot"]):
                    tile = of_in.acquire(1)
                    step(tile, y, CHUNK_K, tt * TSTRIDE)
                    of_in.release(1)
                of_out.release(1)
        with else_(if_op):
            for _ in range_(s0["outs_pw"] - 2):
                y = of_out.acquire(1)
                for tt in range(s0["ot"]):
                    for ch in range(s0["kchunks"]):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE + ch)
                        of_in.release(1)
                of_out.release(1)
            for s in stages[1:]:
                y = of_out.acquire(1)
                for c in range(s["achunks"]):
                    x = of_in.acquire(1)
                    if s["silu"]:
                        step(x, y, CHUNK_K, -102 - c)
                    elif s["norm"]:
                        step(x, y, ROPE_CS if s["out_slot"] == 1 else 0, -100)
                    else:
                        n = min(ACT_CHUNK, s["K"] - c * ACT_CHUNK)
                        step(x, y, n, -1 if _os.environ.get("Q4LAYER_XMODE") == "-1" else -3 - c)
                    of_in.release(1)
                if s["res_slot"] is not None:
                    r = of_in.acquire(1)
                    step(r, y, 0, -101)
                    of_in.release(1)
                if s["rope"]:
                    rp = of_in.acquire(1)
                    step(rp, y, 0, -106)
                    of_in.release(1)
                for tt in range(s["ot"]):
                    for ch in range(s["kchunks"]):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE + ch)
                        of_in.release(1)
                of_out.release(1)
                if s["outs_pw"] > 1:
                    for _ in range_(s["outs_pw"] - 1):
                        y = of_out.acquire(1)
                        for tt in range(s["ot"]):
                            for ch in range(s["kchunks"]):
                                tile = of_in.acquire(1)
                                step(tile, y, CHUNK_K, tt * TSTRIDE + ch)
                                of_in.release(1)
                        of_out.release(1)
                if s["out_slot"] == 3:
                    for o in range(4):
                        y = of_out.acquire(1)
                        if o == 0:
                            kv0 = of_in.acquire(1)
                            step(kv0, y, ATTN_CHUNK_T, -201)
                            of_in.release(1)
                            for _ in range_(ATTN_CHUNKS - 1):
                                kv0 = of_in.acquire(1)
                                step(kv0, y, ATTN_CHUNK_T, -200)
                                of_in.release(1)
                            kv1 = of_in.acquire(1)
                            step(kv1, y, 1, -202)
                            of_in.release(1)
                        step(kv1, y, 0, -110 - o)
                        of_out.release(1)

    workers = [Worker(core_body, [of_ins[i].cons(), of_outs[i].prod(), step], stack_size=2560) for i in range(workers_n)]

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
        one_group = _os.environ.get("Q4LAYER_ONEGROUP") == "1"
        tg = TaskGroup() if one_group else None
        for s in stages:
            if not one_group and s["out_slot"] in (1, 4, 5, 7):
                tg = TaskGroup()
            in_base = s["in_slot"] * ACT_SLOT + (ATTN_OUT if s["in_slot"] == 1 else 0)
            for i in range(workers_n):
                if s["silu"]:
                    for c in range(s["achunks"]):
                        in_prods[i].fill(y_buf, tap_silu(act_total, in_base + c * CHUNK_K), group=tg)
                else:
                    for c in range(s["achunks"]):
                        in_prods[i].fill(y_buf, tap(act_total, in_base + c * ACT_CHUNK, TILE_F32), group=tg)
                if s["res_slot"] is not None:
                    in_prods[i].fill(y_buf, tap(act_total, s["res_slot"] * ACT_SLOT + i * s["rows_pw"], TILE_F32), group=tg)
                if s["rope"]:
                    in_prods[i].fill(y_buf, tap(act_total, ROPE_CS + (i % 2) * 32, TILE_F32), group=tg)
                in_prods[i].fill(w_in, tap(w_total, s["w_base"] + i * s["per_worker_w"], s["per_worker_w"]), group=tg)
            for i in range(workers_n):
                out_conses[i].drain(
                    y_next if s["out_slot"] == 7 else y_buf,
                    tap(act_total, (0 if s["out_slot"] == 7 else s["out_slot"] * ACT_SLOT) + i * s["rows_pw"], s["rows_pw"]),
                    wait=s["out_slot"] in (3, 4, 6, 7),
                    group=tg,
                )
            if not one_group and s["out_slot"] in (3, 4, 6, 7):
                tg.finish()
            if s["out_slot"] == 3:
                if not one_group:
                    tg = TaskGroup()
                for i in range(workers_n):
                    kv_h = i // 2
                    in_prods[i].fill(y_buf, tap(act_total, KV_CACHE_BASE + kv_h * KV_HEAD_STRIDE, KV_HEAD_STRIDE), group=tg)
                    in_prods[i].fill(y_buf, tap_silu(act_total, 2 * ACT_SLOT + kv_h * 64), group=tg)
                for i in range(workers_n):
                    out_conses[i].drain(y_buf, tap(act_total, 1 * ACT_SLOT + ATTN_OUT + i * 128, 128), wait=True, group=tg)
                if not one_group:
                    tg.finish()
        if one_group:
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

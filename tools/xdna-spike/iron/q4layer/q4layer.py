# Fused Q4 transformer-layer design for tools/xdna-spike (backend plan step 2b).
#
# One NPU run executes all seven GEMV stages of a Llama-3.2-1B layer under a
# single xclbin, because switching xclbins between GEMVs costs ~800 us (see
# docs/XDNA_PHASE0_RESULTS.md section 5). 16 workers, one per (column, shim
# channel) pair. Activations live in f32 slots in DDR; each stage's input
# slot is broadcast to every worker, quantized on the core (q4layer.cc
# q4_step t <= -2), and each worker drains its output rows into the next
# slot, with the task group finished before the next stage starts.
#
# Stage chaining follows the real data flow with attention replaced by its
# input (the spike measures GEMV streaming and submission, not attention):
# q/k/v read the hidden slot, o reads q's slot, gate/up read o's slot, down
# reads gate's slot (8192 wide, split-K), and down's output is the layer out.
import argparse
import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import ExternalFunction, In, Out
from aie.iron.dataflow import ObjectFifo
from aie.iron.program import Program
from aie.iron.runtime import Runtime, TaskGroup
from aie.iron.worker import Worker
from aie.utils import get_current_device
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

ROWS = 16
NUM_CHANNELS = 2
CHUNK_K = 2048  # columns per streamed weight-tile chunk (split-K)
TILE_F32 = 4608  # one stream element: 18432 bytes viewed as f32
ACT_CHUNK = TILE_F32  # f32 activations quantized per stream element
ACT_SLOT = 8192  # f32 stride of one activation slot in DDR (max K)
# Extra f32 padding past the last activation slot so a 2-chunk TILE_F32 DMA
# read at in_slot == N_SLOTS - 1 (4608 + 4608 = 9216 words) stays in bounds.
ACT_TAIL = max(0, 2 * TILE_F32 - ACT_SLOT)
TSTRIDE = 8  # tile/chunk encoding stride in q4_step's t (see q4layer.cc)

# (input slot, rows M, cols K, output slot); slots: 0 hidden, 1 q, 2 k, 3 v,
# 4 o, 5 gate, 6 up, 7 layer out.
STAGES = [
    (0, 2048, 2048, 1),  # q
    (0, 512, 2048, 2),  # k
    (0, 512, 2048, 3),  # v
    (1, 2048, 2048, 4),  # o (input: q output as attention proxy)
    (4, 8192, 2048, 5),  # gate
    (4, 8192, 2048, 6),  # up
    (5, 2048, 8192, 7),  # down (input: gate output as gated-product proxy)
]
N_SLOTS = 8


def _parse_spec(raw: str):
    out = []
    for item in raw.strip().rstrip(";").split(";"):
        part = item.strip()
        if not part:
            continue
        fields = [x.strip() for x in part.split(":")]
        if len(fields) != 4:
            raise ValueError(f"invalid Q4LAYER_SPEC entry '{part}': expected in_slot:M:K:out_slot")
        out.append(tuple(int(x) for x in fields))
    return out


# Diagnostic: Q4LAYER_STAGES=N builds only the first N stages (the host still
# packs and checks all seven; only the built stages can pass).
STAGES = STAGES[: int(os.environ.get("Q4LAYER_STAGES", "7"))]
if os.environ.get("Q4LAYER_SPEC"):
    STAGES = _parse_spec(os.environ["Q4LAYER_SPEC"])
elif os.environ.get("Q4LAYER_PICK"):
    STAGES = [STAGES[int(i.strip())] for i in os.environ["Q4LAYER_PICK"].split(",") if i.strip()]
if not STAGES:
    raise ValueError("Q4LAYER stage list must be non-empty")


def stage_params(in_slot, M, K, out_slot, workers_n):
    # Fixed output grouping OT=2 for every stage (as in q4gemv): each out
    # fifo element holds exactly ROWS*OT outputs and every drain consumes
    # whole elements. A drain smaller than the element leaves the fifo
    # inconsistent and the next stage reads zeros (measured: a stage after
    # a partially drained stage produces nothing).
    if not (0 <= in_slot < N_SLOTS and 0 <= out_slot < N_SLOTS):
        raise ValueError(f"slot index out of range 0..{N_SLOTS - 1}: in={in_slot}, out={out_slot}")
    if in_slot == out_slot:
        raise ValueError(f"in_slot and out_slot must differ, got slot {in_slot}")
    min_m = workers_n * ROWS * 2
    if M < min_m or M > ACT_SLOT or M % min_m != 0:
        raise ValueError(
            f"M must be a multiple of {min_m} in {min_m}..={ACT_SLOT} for OT=2, got M={M}"
        )
    if K < CHUNK_K or K > ACT_SLOT or K % CHUNK_K != 0:
        raise ValueError(f"K must be a multiple of {CHUNK_K} in {CHUNK_K}..={ACT_SLOT}, got K={K}")
    rows_pw = M // workers_n
    tiles_pw = rows_pw // ROWS
    kchunks = K // CHUNK_K
    if kchunks > TSTRIDE:
        raise ValueError(f"kchunks={kchunks} exceeds TSTRIDE={TSTRIDE}")
    return {
        "in_slot": in_slot,
        "M": M,
        "K": K,
        "out_slot": out_slot,
        "rows_pw": rows_pw,
        "tiles_pw": tiles_pw,
        "ot": 2,
        "outs_pw": tiles_pw // 2,
        "kchunks": kchunks,
        "achunks": (K + ACT_CHUNK - 1) // ACT_CHUNK,
    }


@iron.jit
def q4layer(w: In, y: Out):
    workers_n = get_current_device().cols * NUM_CHANNELS
    if workers_n != 16:
        raise ValueError(f"q4layer requires 16 workers (8 cols x 2 channels), got {workers_n}")
    stages = [stage_params(i, M, K, o, workers_n) for (i, M, K, o) in STAGES]

    # Weight stream layout (f32 words): stages in order, per stage each
    # worker's tiles contiguous, tile = TILE_F32 words.
    w_total = 0
    for s in stages:
        s["w_base"] = w_total
        s["per_worker_w"] = s["tiles_pw"] * s["kchunks"] * TILE_F32
        w_total += workers_n * s["per_worker_w"]

    act_total = N_SLOTS * ACT_SLOT + ACT_TAIL
    slot_ty = np.ndarray[(TILE_F32,), np.dtype[np.float32]]
    yel_ty = np.ndarray[(ROWS * 2,), np.dtype[np.float32]]
    w_ty = np.ndarray[(w_total,), np.dtype[np.float32]]
    y_ty = np.ndarray[(act_total,), np.dtype[np.float32]]

    step = ExternalFunction(
        "q4_step",
        source_file=str(Path(__file__).parent / "q4layer.cc"),
        arg_types=[slot_ty, yel_ty, np.int32, np.int32],
        include_dirs=[cxx_header_path()],
        compile_flags=[f"-DKMAX={ACT_SLOT}", f"-DACT_CHUNK={ACT_CHUNK}"],
    )
    of_ins = [ObjectFifo(slot_ty, name=f"w_{i}") for i in range(workers_n)]
    of_outs = [ObjectFifo(yel_ty, name=f"y_{i}") for i in range(workers_n)]

    def core_body(of_in, of_out, step):
        for s in stages:
            for o in range(s["outs_pw"]):
                y = of_out.acquire(1)
                if o == 0:
                    for c in range(s["achunks"]):
                        x = of_in.acquire(1)
                        n = min(ACT_CHUNK, s["K"] - c * ACT_CHUNK)
                        step(x, y, n, -1 if os.environ.get("Q4LAYER_XMODE") == "-1" else -3 - c)
                        of_in.release(1)
                for tt in range(s["ot"]):
                    for ch in range(s["kchunks"]):
                        tile = of_in.acquire(1)
                        step(tile, y, CHUNK_K, tt * TSTRIDE + ch)
                        of_in.release(1)
                of_out.release(1)

    workers = [Worker(core_body, [of_ins[i].cons(), of_outs[i].prod(), step], stack_size=2048) for i in range(workers_n)]

    def tap(total, offset, size):
        return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])

    def sequence(w_in, y_buf, in_prods, out_conses):
        one_group = os.environ.get("Q4LAYER_ONEGROUP") == "1"
        tg = TaskGroup() if one_group else None
        for s in stages:
            if not one_group:
                tg = TaskGroup()
            for i in range(workers_n):
                for c in range(s["achunks"]):
                    in_prods[i].fill(y_buf, tap(act_total, s["in_slot"] * ACT_SLOT + c * ACT_CHUNK, TILE_F32), group=tg)
                in_prods[i].fill(w_in, tap(w_total, s["w_base"] + i * s["per_worker_w"], s["per_worker_w"]), group=tg)
            for i in range(workers_n):
                out_conses[i].drain(y_buf, tap(act_total, s["out_slot"] * ACT_SLOT + i * s["rows_pw"], s["rows_pw"]), wait=True, group=tg)
            if not one_group:
                tg.finish()
        if one_group:
            tg.finish()

    rt = Runtime(sequence, [w_ty, y_ty, [of.prod() for of in of_ins], [of.cons() for of in of_outs]])
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

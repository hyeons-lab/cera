#!/usr/bin/env python3
"""Unit tests for XDNA IRON compile-time parameter validation, ABI layout, and schedules."""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib.util
import io
import math
import os
import re
import sys
import types
import unittest
from unittest import mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNELS_DIR = os.path.join(REPO_ROOT, "cera", "src", "backend", "xdna", "kernels")
IRON_ROOT = os.path.join(KERNELS_DIR, "iron")
KERNELS_RS = os.path.join(REPO_ROOT, "cera", "src", "backend", "xdna", "kernels.rs")
BUILD_KERNELS_SH = os.path.join(KERNELS_DIR, "build_kernels.sh")
BUILD_LFM2_SH = os.path.join(KERNELS_DIR, "build_lfm2_kernels.sh")

sys.path.insert(0, IRON_ROOT)
from compile_params import validate_compile_params


def _eval_int_ast(node: ast.AST, env: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_int_ast(node.left, env)
        right = _eval_int_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = _eval_int_ast(node.left, env)
        right = _eval_int_ast(node.comparators[0], env)
        if isinstance(node.ops[0], ast.Eq):
            return int(left == right)
        if isinstance(node.ops[0], ast.NotEq):
            return int(left != right)
    if isinstance(node, ast.IfExp):
        return _eval_int_ast(node.body if _eval_int_ast(node.test, env) else node.orelse, env)
    raise ValueError(f"unsupported expression {ast.dump(node)}")


def _eval_top_level_ints(py_path: str) -> dict[str, int]:
    """Evaluate simple integer constant assignments at module scope via AST."""
    with open(py_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=py_path)
    env: dict[str, int] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                env[stmt.targets[0].id] = _eval_int_ast(stmt.value, env)
            except ValueError:
                pass
    return env


def _parse_rust_usize_consts(rs_path: str) -> dict[str, int]:
    """Parse `pub const NAME: usize = <expr>;` entries with simple arithmetic via AST."""
    with open(rs_path, encoding="utf-8") as fh:
        src = fh.read()
    consts: dict[str, int] = {}
    for match in re.finditer(r"pub const ([A-Z0-9_]+): usize = ([^;]+);", src):
        name, raw_expr = match.group(1), match.group(2).strip()
        try:
            tree = ast.parse(raw_expr.replace("/", "//"), mode="eval")
            consts[name] = _eval_int_ast(tree.body, consts)
        except (SyntaxError, ValueError):
            pass
    return consts


def _parse_rust_bucket_keys(rs_path: str) -> dict[str, list[int]]:
    """Parse `(chunk, ...)` bucket keys from arrays and `lfm2_insts_buckets!` in `kernels.rs`."""
    with open(rs_path, encoding="utf-8") as fh:
        src = fh.read()
    buckets: dict[str, list[int]] = {}
    for m in re.finditer(
        r"pub const ([A-Z0-9_]+): \[\(usize, &\[u8\]\); \d+\] = \[(.*?)\];",
        src,
        re.DOTALL,
    ):
        buckets[m.group(1)] = [int(k) for k in re.findall(r"\((\d+),", m.group(2))]
    for m in re.finditer(
        r'lfm2_insts_buckets!\(\s*(?:///[^\n]*\n\s*)*([A-Z0-9_]+),\s*\d+,\s*"[^"]+",\s*\[(.*?)\]\s*\);',
        src,
        re.DOTALL,
    ):
        buckets[m.group(1)] = [int(k) for k in re.findall(r"\b(\d+)\b", m.group(2))]
    return buckets


def _parse_shell_bucket_vars(sh_path: str) -> dict[str, list[int]]:
    """Parse space-separated integer bucket lists assigned in a shell build script."""
    with open(sh_path, encoding="utf-8") as fh:
        src = fh.read()
    out: dict[str, list[int]] = {}
    for m in re.finditer(r'^([A-Z0-9_]+)="([0-9 ]+)"', src, re.MULTILINE):
        out[m.group(1)] = [int(x) for x in m.group(2).split()]
    return out


def _inspect_generator_wiring(
    py_path: str,
    jit_fn_name: str,
    expected_modes: tuple[int, ...],
    expected_nonzero: tuple[int, ...],
    expected_param_slots: tuple[int, ...] | None = None,
    expected_max_kv_by_mode: dict[int, int] | None = None,
) -> tuple[bool, bool, bool]:
    """Check `_validate_compile_params` delegation and call sites in `@iron.jit` and `main`."""
    with open(py_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=py_path)
    env: dict[str, object] = {}
    int_env: dict[str, int] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            try:
                int_env[name] = _eval_int_ast(stmt.value, int_env)
            except ValueError:
                pass
            if name in {"VALID_MODES", "VALID_PARAM_SLOTS", "MAX_KV_CHUNKS", "MAX_KV_PKTS"}:
                try:
                    env[name] = int_env[name] if name in int_env else ast.literal_eval(stmt.value)
                except ValueError:
                    pass

    def eval_kw(node: ast.AST) -> object:
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        return ast.literal_eval(node)

    delegated = False
    jit_calls_validate = False
    main_calls_validate = False

    def calls_fn(fn_node: ast.FunctionDef, target: str) -> bool:
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == target:
                return True
        return False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_compile_params":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "validate_compile_params":
                    kw_nodes = {k.arg: k.value for k in sub.keywords}
                    kw = {
                        name: eval_kw(val)
                        for name, val in kw_nodes.items()
                        if name in {"valid_modes", "nonzero_kv_modes", "valid_param_slots"}
                    }
                    param_slot_fwd = "param_slot" in kw_nodes
                    slots_ok = expected_param_slots is None or (
                        tuple(kw.get("valid_param_slots", ())) == expected_param_slots  # type: ignore[arg-type]
                        and param_slot_fwd
                    )
                    max_node = kw_nodes.get("max_kv_chunks")
                    if max_node is None:
                        max_ok = False
                    elif expected_max_kv_by_mode is not None:
                        max_ok = all(
                            _eval_int_ast(max_node, {**int_env, "layer_mode": m}) == expected_max_kv_by_mode[m]
                            for m in expected_modes
                        )
                    else:
                        max_ok = _eval_int_ast(max_node, int_env) == env.get("MAX_KV_CHUNKS")
                    delegated = (
                        tuple(kw.get("valid_modes", ())) == expected_modes  # type: ignore[arg-type]
                        and tuple(kw.get("nonzero_kv_modes", ())) == expected_nonzero  # type: ignore[arg-type]
                        and slots_ok
                        and max_ok
                    )
        elif isinstance(node, ast.FunctionDef) and node.name == jit_fn_name:
            jit_calls_validate = calls_fn(node, "_validate_compile_params")
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            main_calls_validate = calls_fn(node, "_validate_compile_params")
    return delegated, jit_calls_validate, main_calls_validate


def _make_mock_modules() -> tuple[dict[str, types.ModuleType], list[dict[str, object]]]:
    """Build lightweight `aie.*` (and `numpy` fallback if missing) stubs for offline tests."""
    dispatched: list[dict[str, object]] = []

    class _FakeSubscriptable:
        def __getitem__(self, _item: object) -> "_FakeSubscriptable":
            return self

    class _FakeTap:
        def __init__(self, *, tensor_dims, offset, sizes, strides) -> None:
            self.tensor_dims = tensor_dims
            self.offset = offset
            self.sizes = sizes
            self.strides = strides

    class _FakeTaskGroup:
        def finish(self) -> None:
            pass

    class _FakeFifoHandle:
        def __init__(self, name: str, kind: str) -> None:
            self.name = name
            self.kind = kind
            self.events: list[tuple[str, int]] = []

        def fill(self, _buf, *, tap: _FakeTap, wait: bool = False, group=None) -> None:
            del wait, group
            self.events.append(("fill", math.prod(tap.sizes)))

        def drain(self, _buf, *, tap: _FakeTap, wait: bool = False, group=None) -> None:
            del wait, group
            self.events.append(("drain", math.prod(tap.sizes)))

    class _FakeFifoView:
        def __init__(self, name: str, role: str) -> None:
            self.name = name
            self.role = role

        def prod(self) -> "_FakeFifoView":
            return self

        def cons(self) -> "_FakeFifoView":
            return self

        def forward(self, **_kw: object) -> "_FakeFifoView":
            return self

        def split(self, *, offsets, obj_types, names, depths) -> list["_FakeFifoView"]:
            del offsets, obj_types, depths
            return [_FakeFifoView(n, "cons") for n in names]

        def join(self, *, offsets, obj_types, names, depths) -> list["_FakeFifoView"]:
            del offsets, obj_types, depths
            return [_FakeFifoView(n, "prod") for n in names]

    class _FakeObjectFifo:
        def __init__(self, _ty: object, *, depth: int = 1, name: str = "") -> None:
            del depth
            self.name = name

        def prod(self) -> _FakeFifoView:
            return _FakeFifoView(self.name, "prod")

        def cons(self) -> _FakeFifoView:
            return _FakeFifoView(self.name, "cons")

    class _FakeBuffer:
        def __init__(self, _ty: object, *, initial_value=None, name: str = "") -> None:
            self.initial_value = initial_value
            self.name = name

    class _FakeWorker:
        def __init__(self, core_body, *, fn_args, tile, **kw: object) -> None:
            del kw
            self.core_body = core_body
            self.fn_args = fn_args
            self.tile = tile

    class _FakeTile:
        def __init__(self, col: int, row: int) -> None:
            self.col = col
            self.row = row

    class _FakeRuntime:
        def __init__(self, seq_fn, *, fn_args) -> None:
            self.seq_fn = seq_fn
            self.fn_args = fn_args

    class _FakeProgram:
        def __init__(self, device, rt: _FakeRuntime, workers: list[_FakeWorker]) -> None:
            self.device = device
            self.rt = rt
            self.workers = workers

        def resolve_program(self) -> dict[str, object]:
            return {"runtime": self.rt, "workers": self.workers}

    class _FakeAieIron(types.ModuleType):
        @staticmethod
        def jit(**_kw: object):
            def deco(fn):
                return fn
            return deco

    mods: dict[str, types.ModuleType] = {}
    mods["aie"] = types.ModuleType("aie")
    mods["aie.iron"] = _FakeAieIron("aie.iron")

    for subname in (
        "aie.dialects",
        "aie.helpers",
        "aie.helpers.dialects",
        "aie.helpers.dialects.scf",
        "aie.helpers.taplib",
        "aie.helpers.taplib.tap",
        "aie.iron.controlflow",
        "aie.iron.dataflow",
        "aie.iron.device",
        "aie.iron.program",
        "aie.iron.runtime",
        "aie.iron.worker",
        "aie.ir",
        "aie.utils",
        "aie.utils.config",
        "aie.utils.hostruntime",
        "aie.utils.hostruntime.argparse",
        "aie.utils.hostruntime.cli",
    ):
        mods[subname] = types.ModuleType(subname)

    for attr in ("CompileTime", "In", "Out"):
        setattr(mods["aie.iron"], attr, _FakeSubscriptable())
    setattr(mods["aie.iron"], "Buffer", _FakeBuffer)
    setattr(mods["aie.iron"], "ExternalFunction", lambda *a, **kw: ("ext_fn", a, kw))
    setattr(mods["aie.dialects"], "arith", object())
    setattr(mods["aie.helpers.dialects.scf"], "else_", object())
    setattr(mods["aie.helpers.dialects.scf"], "if_", object())
    setattr(mods["aie.helpers.taplib.tap"], "TensorAccessPattern", _FakeTap)
    setattr(mods["aie.iron.controlflow"], "range_", range)
    setattr(mods["aie.iron.dataflow"], "ObjectFifo", _FakeObjectFifo)
    setattr(mods["aie.iron.device"], "NPU2", lambda: "NPU2")
    setattr(mods["aie.iron.device"], "Tile", _FakeTile)
    setattr(mods["aie.iron.program"], "Program", _FakeProgram)
    setattr(mods["aie.iron.runtime"], "Runtime", _FakeRuntime)
    setattr(mods["aie.iron.runtime"], "TaskGroup", _FakeTaskGroup)
    setattr(mods["aie.iron.worker"], "Worker", _FakeWorker)
    setattr(mods["aie.ir"], "IndexType", object())
    setattr(mods["aie.ir"], "IntegerType", object())
    setattr(mods["aie.utils"], "get_current_device", lambda: "NPU2")
    setattr(mods["aie.utils.config"], "cxx_header_path", lambda: "/tmp/cxx")
    setattr(mods["aie.utils.hostruntime.argparse"], "add_compile_args", lambda _p: None)

    def fake_run_design_cli(fn, opts, *, device, compile_kwargs):
        dispatched.append({"fn": fn.__name__, "device": device, "kwargs": compile_kwargs(opts)})

    setattr(mods["aie.utils.hostruntime.cli"], "run_design_cli", fake_run_design_cli)
    mods["_fake_fifo_handle"] = _FakeFifoHandle  # type: ignore[assignment]

    if importlib.util.find_spec("numpy") is not None:
        import numpy  # noqa: F401 - pre-register before mock.patch.dict snapshots sys.modules
    else:
        class _FakeArray(list):
            def reshape(self, _shape: int) -> list[int]:
                flat: list[int] = []
                for item in self:
                    if isinstance(item, (list, tuple)):
                        flat.extend(int(x) for x in item)
                    else:
                        flat.append(int(item))
                return flat

            def view(self, _dtype: object) -> "_FakeArray":
                return self

        np_mod = types.ModuleType("numpy")
        setattr(np_mod, "ndarray", _FakeSubscriptable())
        setattr(np_mod, "dtype", _FakeSubscriptable())
        setattr(np_mod, "float32", object())
        setattr(np_mod, "int32", object())
        setattr(np_mod, "uint32", object())
        setattr(np_mod, "array", lambda vals, dtype=None: _FakeArray(vals))
        mods["numpy"] = np_mod

    return mods, dispatched


def _load_iron_module(subdir: str, tag: str) -> object:
    path = os.path.join(IRON_ROOT, subdir, f"{subdir}.py")
    spec = importlib.util.spec_from_file_location(f"_mocked_{subdir}_{tag}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decode_sched(subdir: str, raw: list[int] | object) -> list[tuple[int, int, int, int, int, int, int]]:
    vals = [int(x) for x in raw]  # type: ignore[attr-defined]
    steps: list[tuple[int, int, int, int, int, int, int]] = []
    if subdir == "q4layer_lfm3":
        assert len(vals) % 7 == 0
        for i in range(0, len(vals), 7):
            steps.append(tuple(vals[i : i + 7]))  # type: ignore[arg-type]
    else:
        assert len(vals) % 2 == 0
        for i in range(0, len(vals), 2):
            w0 = vals[i] & 0xFFFFFFFF
            w1 = vals[i + 1] & 0xFFFFFFFF
            kind = w0 & 0xFF
            cnt = (w0 >> 8) & 0xFF
            op = (w0 >> 16) & 0xFF
            a0 = (w0 >> 24) & 0xFF
            a1 = w1 & 0xFF
            a2 = (w1 >> 8) & 0xFF
            mask = (w1 >> 16) & 0xFFFF
            steps.append((kind, cnt, op, a0, a1, a2, mask))
    return steps


def _check_seq_vs_sched_balance(
    tc: unittest.TestCase,
    subdir: str,
    fn_name: str,
    mod: object,
    mode: int,
    kv_chunks: int,
    param_slot: int | None,
    fifo_cls: type,
) -> None:
    """Verify worker schedule masks and exact DMA vs worker FIFO element counts for a compiled variant."""
    n_pos = 6 if subdir == "q4layer_lfm3" else 5
    kw: dict[str, object] = {"layer_mode": mode, "kv_chunks": kv_chunks}
    if param_slot is not None:
        kw["param_slot"] = param_slot
    prog = getattr(mod, fn_name)(*([None] * n_pos), **kw)
    rt = prog["runtime"]
    workers = prog["workers"]
    assert len(workers) == 32

    if subdir == "q4layer_lfm3":
        # w0p_h[8], w1p_h[8], yc_h[8]
        fifos = (
            [fifo_cls(f"w0_{c}", "w0") for c in range(8)]
            + [fifo_cls(f"w1_{c}", "w1") for c in range(8)]
            + [fifo_cls(f"y_{c}", "y") for c in range(8)]
        )
        rt.seq_fn(*([object()] * 6), *fifos)
        w0_fills = [sum(n for ev, n in fifos[c].events if ev == "fill") for c in range(8)]
        w1_fills = [sum(n for ev, n in fifos[8 + c].events if ev == "fill") for c in range(8)]
        y_drains = [sum(n for ev, n in fifos[16 + c].events if ev == "drain") for c in range(8)]
    elif subdir == "q4layer_lfm2":
        fifos = (
            [fifo_cls(f"w_{c}", "w") for c in range(8)]
            + [fifo_cls(f"y_{c}", "y") for c in range(8)]
            + [fifo_cls(f"kv_{c}", "kv") for c in range(8)]
        )
        rt.seq_fn(*([object()] * 5), *fifos)
        w_fills = [sum(n for ev, n in fifos[c].events if ev == "fill") for c in range(8)]
        y_drains = [sum(n for ev, n in fifos[8 + c].events if ev == "drain") for c in range(8)]
        kv_fills = [sum(n for ev, n in fifos[16 + c].events if ev == "fill") for c in range(8)]
    else:
        # Prefill LFM2 / LFM2_EXT / LFM3: wp_h[8], xp_h[2], yc_h[8]
        fifos = (
            [fifo_cls(f"w_{c}", "w") for c in range(8)]
            + [fifo_cls(f"x_{x}", "x") for x in range(2)]
            + [fifo_cls(f"y_{c}", "y") for c in range(8)]
        )
        rt.seq_fn(*([object()] * 5), *fifos)
        w_fills = [sum(n for ev, n in fifos[c].events if ev == "fill") for c in range(8)]
        x_fills = [sum(n for ev, n in fifos[8 + x].events if ev == "fill") for x in range(2)]
        y_drains = [sum(n for ev, n in fifos[10 + c].events if ev == "drain") for c in range(8)]

    w_elems = mod.W_ELEMS  # type: ignore[attr-defined]
    y_core_elems = mod.Y_CORE_ELEMS  # type: ignore[attr-defined]
    mode_bit = 1 << mode
    seen_bits: set[int] = set()

    for w in workers:
        c, r = w.tile.col, w.tile.row - 2
        sched_buf = w.fn_args[2 if subdir in {"q4layer_lfm3"} else 3]
        steps = _decode_sched(subdir, sched_buf.initial_value)
        for step in steps:
            mk = step[6]
            for bit in range(16):
                if mk & (1 << bit):
                    seen_bits.add(bit)

        active_steps = [st for st in steps if st[6] & mode_bit]
        tc.assertGreater(len(active_steps), 0, f"{subdir} mode={mode} c={c} r={r} has no active schedule steps")

        if subdir == "q4layer_lfm3":
            w_pkts = 1  # peeled wp0
            y_pkts = 0
            for kind, cnt, _op, a0, a1, _a2, _mk in active_steps:
                if kind == 0:
                    w_pkts += cnt
                elif kind == 1:
                    y_pkts += cnt
                elif kind == 2:
                    load_x = (_op & 2) != 0
                    mode_sel = _op >> 2
                    n_ghalf = 3 if mode_sel == 2 else 1
                    do_drain = (_a2 & 2) != 0
                    if load_x:
                        w_pkts += 1
                    w_pkts += n_ghalf * a0 * 2 * a1
                    if do_drain:
                        y_pkts += 1
                elif kind == 3 and r < 2:
                    w_pkts += kv_chunks + (1 if mode in (1, 4) else 0)
            if r == 0:
                tc.assertEqual(w0_fills[c], w_pkts * w_elems, f"{subdir} m={mode} k={kv_chunks} c={c} r={r} w0")
            elif r == 2:
                tc.assertEqual(w1_fills[c], w_pkts * w_elems, f"{subdir} m={mode} k={kv_chunks} c={c} r={r} w1")
            if r == 0:
                tc.assertEqual(y_drains[c], y_pkts * 4 * y_core_elems, f"{subdir} m={mode} k={kv_chunks} c={c} y")
        elif subdir == "q4layer_lfm2":
            w_pkts = 1  # peeled wp0
            y_pkts = 0
            kv_blks = 0
            for kind, cnt, op, a0, a1, a2, _mk in active_steps:
                if kind == 0:
                    w_pkts += cnt
                elif kind == 1:
                    y_pkts += cnt
                elif kind == 2:
                    load_x = (op & 2) != 0
                    mode_sel = op >> 2
                    n_ghalf = 3 if mode_sel == 2 else 1
                    do_drain = (a2 & 2) != 0
                    if load_x:
                        w_pkts += 1
                    w_pkts += n_ghalf * a0 * 4 * a1
                    if do_drain:
                        y_pkts += 1
                elif kind == 3:
                    kv_blks += kv_chunks * 4 + (1 if mode != 3 else 0)
            tc.assertEqual(w_fills[c], w_pkts * w_elems, f"{subdir} m={mode} k={kv_chunks} c={c} r={r} w")
            tc.assertEqual(kv_fills[c], kv_blks * 1024, f"{subdir} m={mode} k={kv_chunks} c={c} r={r} kv")
            if r == 0:
                tc.assertEqual(y_drains[c], y_pkts * 4 * y_core_elems, f"{subdir} m={mode} k={kv_chunks} c={c} y")
        else:
            # Prefill LFM2 / LFM2_EXT / LFM3
            x_elems = mod.X_ELEMS  # type: ignore[attr-defined]
            w_pkts = 1  # peeled wp0
            x_pkts = 0
            y_pkts = 0
            for kind, cnt, op, a0, _a1, a2, _mk in active_steps:
                if kind == 0:
                    if op >= 16:
                        if subdir == "q4prefill_lfm3":
                            w_pkts += 2 * (kv_chunks + a2)
                        elif subdir == "q4prefill_lfm2_ext":
                            extra_kv = 0 if mode < 5 else 2
                            w_pkts += kv_chunks + extra_kv
                        else:
                            w_pkts += kv_chunks + 2
                    else:
                        w_pkts += cnt
                elif kind == 1:
                    x_pkts += cnt
                elif kind == 2:
                    y_pkts += cnt
                elif kind == 3:
                    x_pkts += cnt
                    w_pkts += cnt * a0
            tc.assertEqual(w_fills[c], w_pkts * w_elems, f"{subdir} m={mode} k={kv_chunks} c={c} r={r} w")
            tc.assertEqual(x_fills[c // 4], x_pkts * 4 * x_elems, f"{subdir} m={mode} k={kv_chunks} c={c} r={r} x")
            if r == 0:
                tc.assertEqual(y_drains[c], y_pkts * 4 * y_core_elems, f"{subdir} m={mode} k={kv_chunks} c={c} y")

    tc.assertEqual(seen_bits, set(mod.VALID_MODES), f"{subdir} schedule mask bits {seen_bits} != VALID_MODES {mod.VALID_MODES}")  # type: ignore[attr-defined]


class TestXdnaCompileParams(unittest.TestCase):
    def test_decode_scripts_valid_and_invalid_compile_params(self) -> None:
        valid_lfm2 = (0, 1, 2, 3, 4, 5, 6, 7)
        for mode in valid_lfm2:
            for chunks in (1, 256, 512):
                validate_compile_params(mode, chunks, valid_modes=valid_lfm2, nonzero_kv_modes=(3,), max_kv_chunks=512)
            if mode != 3:
                validate_compile_params(mode, 0, valid_modes=valid_lfm2, nonzero_kv_modes=(3,), max_kv_chunks=512)
        with self.assertRaisesRegex(ValueError, "kv_chunks"):
            validate_compile_params(1, 513, valid_modes=valid_lfm2, nonzero_kv_modes=(3,), max_kv_chunks=512)

        valid_lfm3 = (0, 1, 2, 3, 4)
        for mode in valid_lfm3:
            mode_max = 512 if mode == 1 else 256
            for chunks in (1, 128, mode_max):
                validate_compile_params(mode, chunks, valid_modes=valid_lfm3, nonzero_kv_modes=(3,), max_kv_chunks=mode_max)
            if mode != 3:
                validate_compile_params(mode, 0, valid_modes=valid_lfm3, nonzero_kv_modes=(3,), max_kv_chunks=mode_max)

        with self.assertRaisesRegex(ValueError, "layer_mode=3 requires kv_chunks >= 1"):
            validate_compile_params(3, 0, valid_modes=valid_lfm3, nonzero_kv_modes=(3,))

        with self.assertRaisesRegex(ValueError, r"out of range 0\.\.=4"):
            validate_compile_params(1, 5, valid_modes=valid_lfm3, nonzero_kv_modes=(3,), max_kv_chunks=4)

        for bad_mode in (-1, 5, 99, True, False, "1"):
            with self.assertRaisesRegex(ValueError, "unsupported layer_mode"):
                validate_compile_params(bad_mode, 1, valid_modes=valid_lfm3, nonzero_kv_modes=(3,))  # type: ignore[arg-type]

        for bad_chunks in (-1, 257, True, False, "4"):
            with self.assertRaisesRegex(ValueError, "kv_chunks"):
                validate_compile_params(4, bad_chunks, valid_modes=valid_lfm3, nonzero_kv_modes=(3,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "kv_chunks"):
            validate_compile_params(1, 513, valid_modes=valid_lfm3, nonzero_kv_modes=(3,), max_kv_chunks=512)

    def test_prefill_scripts_valid_and_invalid_compile_params(self) -> None:
        valid_lfm2 = (0, 1, 5, 6)
        for mode in valid_lfm2:
            for chunks in (0, 1, 1024):
                validate_compile_params(mode, chunks, valid_modes=valid_lfm2, nonzero_kv_modes=(), max_kv_chunks=1024)

        for bad_mode in (2, 3, 4):
            with self.assertRaisesRegex(ValueError, "unsupported layer_mode"):
                validate_compile_params(bad_mode, 1, valid_modes=valid_lfm2, nonzero_kv_modes=(), max_kv_chunks=1024)
        with self.assertRaisesRegex(ValueError, "kv_chunks"):
            validate_compile_params(5, 1025, valid_modes=valid_lfm2, nonzero_kv_modes=(), max_kv_chunks=1024)

        valid_lfm2_ext = (0, 2, 3, 4, 5, 6, 7)
        for slot in (0, 1, 2, 3):
            validate_compile_params(
                3,
                1,
                valid_modes=valid_lfm2_ext,
                nonzero_kv_modes=(3, 4),
                max_kv_chunks=1024,
                param_slot=slot,
                valid_param_slots=(0, 1, 2, 3),
            )
        with self.assertRaisesRegex(ValueError, "unsupported layer_mode"):
            validate_compile_params(
                1,
                0,
                valid_modes=valid_lfm2_ext,
                nonzero_kv_modes=(3, 4),
                max_kv_chunks=1024,
                param_slot=0,
                valid_param_slots=(0, 1, 2, 3),
            )
        with self.assertRaisesRegex(ValueError, "kv_chunks"):
            validate_compile_params(
                5,
                1025,
                valid_modes=valid_lfm2_ext,
                nonzero_kv_modes=(3, 4),
                max_kv_chunks=1024,
                param_slot=0,
                valid_param_slots=(0, 1, 2, 3),
            )

        valid_lfm3 = (0, 1, 2, 3, 4, 5, 6)
        with self.assertRaisesRegex(ValueError, "valid_param_slots is required"):
            validate_compile_params(1, 0, valid_modes=valid_lfm3, param_slot=0)

        for slot in (0, 1, 2, 3):
            validate_compile_params(
                6,
                1,
                valid_modes=valid_lfm3,
                nonzero_kv_modes=(3, 4, 6),
                param_slot=slot,
                valid_param_slots=(0, 1, 2, 3),
            )
        with self.assertRaisesRegex(ValueError, "layer_mode=6 requires kv_chunks >= 1"):
            validate_compile_params(
                6,
                0,
                valid_modes=valid_lfm3,
                nonzero_kv_modes=(3, 4, 6),
                param_slot=0,
                valid_param_slots=(0, 1, 2, 3),
            )
        for bad_slot in (-1, 4, True, "0"):
            with self.assertRaisesRegex(ValueError, "unsupported param_slot"):
                validate_compile_params(
                    1,
                    1,
                    valid_modes=valid_lfm3,
                    nonzero_kv_modes=(3, 4, 6),
                    param_slot=bad_slot,  # type: ignore[arg-type]
                    valid_param_slots=(0, 1, 2, 3),
                )

    def test_iron_generators_delegate_and_match_kernels_rs_abi(self) -> None:
        rs = _parse_rust_usize_consts(KERNELS_RS)

        lfm2_dec = os.path.join(IRON_ROOT, "q4layer_lfm2", "q4layer_lfm2.py")
        lfm3_dec = os.path.join(IRON_ROOT, "q4layer_lfm3", "q4layer_lfm3.py")
        lfm2_pre = os.path.join(IRON_ROOT, "q4prefill_lfm2", "q4prefill_lfm2.py")
        lfm2_pre_ext = os.path.join(IRON_ROOT, "q4prefill_lfm2_ext", "q4prefill_lfm2_ext.py")
        lfm3_pre = os.path.join(IRON_ROOT, "q4prefill_lfm3", "q4prefill_lfm3.py")

        self.assertEqual(
            _inspect_generator_wiring(
                lfm2_dec,
                "q4layer_lfm2",
                (0, 1, 2, 3, 4, 5, 6, 7),
                (3,),
                expected_max_kv_by_mode={m: 512 for m in range(8)},
            ),
            (True, True, True),
        )
        self.assertEqual(
            _inspect_generator_wiring(
                lfm3_dec,
                "q4layer_lfm3",
                (0, 1, 2, 3, 4),
                (3,),
                expected_max_kv_by_mode={0: 256, 1: 512, 2: 256, 3: 256, 4: 256},
            ),
            (True, True, True),
        )
        self.assertEqual(
            _inspect_generator_wiring(lfm2_pre, "q4prefill_lfm2", (0, 1, 5, 6), ()),
            (True, True, True),
        )
        self.assertEqual(
            _inspect_generator_wiring(
                lfm2_pre_ext, "q4prefill_lfm2", (0, 2, 3, 4, 5, 6, 7), (3, 4), (0, 1, 2, 3)
            ),
            (True, True, True),
        )
        self.assertEqual(
            _inspect_generator_wiring(
                lfm3_pre, "q4prefill_lfm3", (0, 1, 2, 3, 4, 5, 6), (3, 4, 6), (0, 1, 2, 3)
            ),
            (True, True, True),
        )

        # Ensure skip_cur_kv (`g_params[10]`) check stays in unified `q4prefill_lfm3_core.cc`.
        lfm3_core_cc = os.path.join(IRON_ROOT, "q4prefill_lfm3", "q4prefill_lfm3_core.cc")
        with open(lfm3_core_cc, encoding="utf-8") as fh:
            self.assertIn("chunk_iter >= kv_ch && g_params[10]", fh.read())

        c_lfm2_dec = _eval_top_level_ints(lfm2_dec)
        self.assertEqual(c_lfm2_dec["SLOT_NORM_0"], rs["LFM2_DEC_SLOT_NORM_0"])
        self.assertEqual(c_lfm2_dec["SLOT_NORM_1"], rs["LFM2_DEC_SLOT_NORM_1"])
        self.assertEqual(c_lfm2_dec["SLOT_ROPE_PKT"], rs["LFM2_DEC_SLOT_ROPE_PKT"])
        self.assertEqual(c_lfm2_dec["MAX_KV_CHUNKS"], rs["LFM2_MAX_KV_CHUNKS"])
        self.assertEqual(c_lfm2_dec["MAX_KV_PKTS"], rs["LFM2_DEC_MAX_KV_PKTS"])
        self.assertEqual(c_lfm2_dec["W_TOTAL"] * 4, rs["LFM2_W_BYTES"])
        self.assertEqual(c_lfm2_dec["ACT_TOTAL"] * 4, rs["LFM2_DEC_ACT_BYTES"])
        self.assertEqual(c_lfm2_dec["KV_TOTAL"] * 4, rs["LFM2_KV_BYTES"])
        self.assertEqual(c_lfm2_dec["KV_OUT_TOTAL"] * 4, rs["LFM2_KVO_BYTES"])
        self.assertLessEqual(c_lfm2_dec["SLOT_V"] + 512, c_lfm2_dec["SLOT_SW"])

        c_lfm3_dec = _eval_top_level_ints(lfm3_dec)
        self.assertEqual(c_lfm3_dec["SLOT_NORM_0"], rs["LFM3_DEC_SLOT_NORM_0"])
        self.assertEqual(c_lfm3_dec["SLOT_NORM_1"], rs["LFM3_DEC_SLOT_NORM_1"])
        self.assertEqual(c_lfm3_dec["SLOT_ROPE_PKT"], rs["LFM3_DEC_SLOT_ROPE_PKT"])
        self.assertEqual(c_lfm3_dec["SLOT_DEC_FIN_NORM"], rs["LFM3_DEC_SLOT_FIN_NORM"])
        self.assertEqual(c_lfm3_dec["MAX_KV_CHUNKS"], rs["LFM3_MAX_KV_CHUNKS"])
        self.assertLessEqual(c_lfm3_dec["W_TOTAL"] * 4, rs["LFM3_W_BYTES"])
        self.assertEqual(c_lfm3_dec["ACT_TOTAL"] * 4, rs["LFM3_DEC_ACT_BYTES"])
        self.assertEqual(c_lfm3_dec["KV_TOTAL"] * 4, rs["LFM3_KV_BYTES"])
        self.assertEqual(c_lfm3_dec["KV_OUT_TOTAL"] * 4, rs["LFM3_KVO_BYTES"])
        self.assertLessEqual(c_lfm3_dec["SLOT_DEC_FA_STATE"] + 6144, c_lfm3_dec["SLOT_NORM_0"])

        c_lfm2_pre = _eval_top_level_ints(lfm2_pre)
        self.assertEqual(c_lfm2_pre["SLOT_NORM_0"], rs["LFM2_PREFILL_SLOT_NORM_0"])
        self.assertEqual(c_lfm2_pre["SLOT_NORM_1"], rs["LFM2_PREFILL_SLOT_NORM_1"])
        self.assertEqual(c_lfm2_pre["SLOT_ROPE_PKT_0"], rs["LFM2_PREFILL_SLOT_ROPE_PKT_0"])
        self.assertEqual(c_lfm2_pre["SLOT_ROPE_PKT_1"], rs["LFM2_PREFILL_SLOT_ROPE_PKT_1"])
        self.assertEqual(c_lfm2_pre["MAX_KV_CHUNKS"], rs["LFM2_MAX_KV_CHUNKS"])
        self.assertEqual(c_lfm2_pre["W_TOTAL"] * 4, rs["LFM2_W_BYTES"])
        self.assertEqual(c_lfm2_pre["ACT_TOTAL"] * 4, rs["LFM2_PREFILL_ACT_BYTES"])
        self.assertEqual(c_lfm2_pre["YN_TOTAL"] * 4, rs["LFM2_PREFILL_YN_BYTES"])
        self.assertEqual(c_lfm2_pre["KV_TOTAL"] * 4, rs["LFM2_KV_BYTES"])
        self.assertLessEqual(c_lfm2_pre["SLOT_FA_STATE"] + 147456, c_lfm2_pre["SLOT_XQ"])

        c_lfm2_ext = _eval_top_level_ints(lfm2_pre_ext)
        self.assertEqual(c_lfm2_ext["SLOT_EXT_PARAMS_0"], rs["LFM2_PREFILL_SLOT_EXT_PARAMS_0"])
        self.assertEqual(c_lfm2_ext["SLOT_EXT_PARAMS_1"], rs["LFM2_PREFILL_SLOT_EXT_PARAMS_1"])
        self.assertEqual(c_lfm2_ext["SLOT_EXT_PARAMS_2"], rs["LFM2_PREFILL_SLOT_EXT_PARAMS_2"])
        self.assertEqual(c_lfm2_ext["SLOT_NORM_0"], rs["LFM2_PREFILL_SLOT_NORM_0"])
        self.assertEqual(c_lfm2_ext["SLOT_NORM_1"], rs["LFM2_PREFILL_SLOT_NORM_1"])
        self.assertEqual(c_lfm2_ext["SLOT_ROPE_PKT_0"], rs["LFM2_PREFILL_SLOT_ROPE_PKT_0"])
        self.assertEqual(c_lfm2_ext["SLOT_ROPE_PKT_1"], rs["LFM2_PREFILL_SLOT_ROPE_PKT_1"])
        self.assertEqual(c_lfm2_ext["MAX_KV_CHUNKS"], rs["LFM2_MAX_KV_CHUNKS"])
        self.assertEqual(c_lfm2_ext["W_TOTAL"] * 4, rs["LFM2_W_BYTES"])
        self.assertEqual(c_lfm2_ext["ACT_TOTAL"] * 4, rs["LFM2_PREFILL_ACT_BYTES"])
        self.assertEqual(c_lfm2_ext["YN_TOTAL"] * 4, rs["LFM2_PREFILL_YN_BYTES"])
        self.assertEqual(c_lfm2_ext["KV_TOTAL"] * 4, rs["LFM2_KV_BYTES"])

        c_lfm3_pre = _eval_top_level_ints(lfm3_pre)
        self.assertEqual(c_lfm3_pre["SLOT_FA_Q"], rs["LFM3_PREFILL_SLOT_Q_BF16"])
        self.assertEqual(c_lfm3_pre["SLOT_ATTN_GATE"], rs["LFM3_PREFILL_SLOT_ATTN_GATE"])
        self.assertEqual(c_lfm3_pre["SLOT_NORM_0"], rs["LFM3_PREFILL_SLOT_NORM_0"])
        self.assertEqual(c_lfm3_pre["SLOT_NORM_1"], rs["LFM3_PREFILL_SLOT_NORM_1"])
        self.assertEqual(c_lfm3_pre["SLOT_NORM_2"], rs["LFM3_PREFILL_SLOT_NORM_2"])
        self.assertEqual(c_lfm3_pre["SLOT_NORM_3"], rs["LFM3_PREFILL_SLOT_NORM_3"])
        self.assertEqual(c_lfm3_pre["SLOT_ROPE_PKT_0A"], rs["LFM3_PREFILL_SLOT_ROPE_PKT_0A"])
        self.assertEqual(c_lfm3_pre["SLOT_ROPE_PKT_0B"], rs["LFM3_PREFILL_SLOT_ROPE_PKT_0B"])
        self.assertEqual(c_lfm3_pre["SLOT_ROPE_PKT_1A"], rs["LFM3_PREFILL_SLOT_ROPE_PKT_1A"])
        self.assertEqual(c_lfm3_pre["SLOT_ROPE_PKT_1B"], rs["LFM3_PREFILL_SLOT_ROPE_PKT_1B"])
        self.assertEqual(c_lfm3_pre["MAX_KV_CHUNKS"], rs["LFM3_MAX_KV_CHUNKS"])
        self.assertLessEqual(c_lfm3_pre["W_TOTAL"] * 4, rs["LFM3_W_BYTES"])
        self.assertEqual(c_lfm3_pre["ACT_TOTAL"] * 4, rs["LFM3_PREFILL_ACT_BYTES"])
        self.assertEqual(c_lfm3_pre["KV_TOTAL"] * 4, rs["LFM3_KV_BYTES"])
        self.assertEqual(c_lfm3_pre["KV_OUT_TOTAL"] * 4, rs["LFM3_KVO_BYTES"])
        self.assertEqual(c_lfm3_pre["YN_TOTAL"] * 4, rs["LFM3_PREFILL_YN_BYTES"])

    def test_mocked_aie_cli_dispatch_and_validation(self) -> None:
        scripts = [
            ("q4layer_lfm2", "q4layer_lfm2", "3", "4"),
            ("q4layer_lfm3", "q4layer_lfm3", "3", "4"),
            ("q4prefill_lfm2", "q4prefill_lfm2", None, "5"),
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", "3", "5"),
            ("q4prefill_lfm3", "q4prefill_lfm3", "4", "5"),
        ]
        mods, dispatched = _make_mock_modules()
        with mock.patch.dict(sys.modules, mods):
            for idx, (subdir, fn_name, nz_mode, ok_mode) in enumerate(scripts):
                path = os.path.join(IRON_ROOT, subdir, f"{subdir}.py")
                mod = _load_iron_module(subdir, f"cli_{idx}")

                has_param_slot = subdir in {"q4prefill_lfm2_ext", "q4prefill_lfm3"}
                extra_args = ["--param-slot", "2"] if has_param_slot else []
                with mock.patch.object(sys, "argv", [path, "--layer-mode", ok_mode, "--kv-chunks", "0", *extra_args]):
                    mod.main()
                self.assertEqual(dispatched[-1]["fn"], fn_name)
                expected_kw = {"layer_mode": int(ok_mode), "kv_chunks": 0}
                if has_param_slot:
                    expected_kw["param_slot"] = 2
                self.assertEqual(dispatched[-1]["kwargs"], expected_kw)

                n_pos = 6 if subdir == "q4layer_lfm3" else 5
                if nz_mode is not None:
                    with self.assertRaises(ValueError):
                        getattr(mod, fn_name)(*([None] * n_pos), layer_mode=int(nz_mode), kv_chunks=0)
                    with self.assertRaises(ValueError):
                        with mock.patch.object(sys, "argv", [path, "--layer-mode", nz_mode, "--kv-chunks", "0"]):
                            mod.main()
                if has_param_slot:
                    with self.assertRaises(ValueError):
                        getattr(mod, fn_name)(*([None] * n_pos), layer_mode=int(ok_mode), kv_chunks=0, param_slot=4)

                upper_limit = mod.MAX_KV_PKTS if subdir == "q4layer_lfm2" else mod.MAX_KV_CHUNKS
                for bad_chunks in ("-1", str(upper_limit + 1)):
                    with self.assertRaises(ValueError):
                        with mock.patch.object(sys, "argv", [path, "--layer-mode", ok_mode, "--kv-chunks", bad_chunks]):
                            mod.main()

                if has_param_slot:
                    with self.assertRaises(ValueError):
                        with mock.patch.object(sys, "argv", [path, "--layer-mode", "3", "--kv-chunks", "0"]):
                            mod.main()
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            with mock.patch.object(sys, "argv", [path, "--param-slot", "4"]):
                                mod.main()

                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        with mock.patch.object(sys, "argv", [path, "--layer-mode", "9"]):
                            mod.main()

            # Script-specific boundary assertions.
            lfm2_dec = _load_iron_module("q4layer_lfm2", "bounds")
            with mock.patch.object(sys, "argv", ["q4layer_lfm2.py", "--layer-mode", "1", "--kv-chunks", "512"]):
                lfm2_dec.main()
            with self.assertRaisesRegex(ValueError, "out of range 0..=512"):
                with mock.patch.object(sys, "argv", ["q4layer_lfm2.py", "--layer-mode", "1", "--kv-chunks", "513"]):
                    lfm2_dec.main()

            lfm3_dec = _load_iron_module("q4layer_lfm3", "bounds")
            for c in ("511", "512"):
                with mock.patch.object(sys, "argv", ["q4layer_lfm3.py", "--layer-mode", "1", "--kv-chunks", c]):
                    lfm3_dec.main()
            with self.assertRaisesRegex(ValueError, "out of range 0..=512"):
                with mock.patch.object(sys, "argv", ["q4layer_lfm3.py", "--layer-mode", "1", "--kv-chunks", "513"]):
                    lfm3_dec.main()

            lfm2_pre = _load_iron_module("q4prefill_lfm2", "bounds")
            for bad_mode in ("2", "4"):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        with mock.patch.object(sys, "argv", ["q4prefill_lfm2.py", "--layer-mode", bad_mode]):
                            lfm2_pre.main()

            lfm2_ext = _load_iron_module("q4prefill_lfm2_ext", "bounds")
            with mock.patch.object(sys, "argv", ["q4prefill_lfm2_ext.py"]):
                lfm2_ext.main()
            self.assertEqual(dispatched[-1]["kwargs"], {"layer_mode": 0, "kv_chunks": 0, "param_slot": 0})
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    with mock.patch.object(sys, "argv", ["q4prefill_lfm2_ext.py", "--layer-mode", "1"]):
                        lfm2_ext.main()

    def test_shell_build_buckets_and_worker_schedule_mode_parity(self) -> None:
        sh_lfm3 = _parse_shell_bucket_vars(BUILD_KERNELS_SH)
        sh_lfm2 = _parse_shell_bucket_vars(BUILD_LFM2_SH)
        rs_buckets = _parse_rust_bucket_keys(KERNELS_RS)

        self.assertEqual(sh_lfm3["LFM3_ATTN_BUCKETS"], rs_buckets["Q4PREFILL_LFM3_ATTN_BUCKETS"])
        self.assertEqual(sh_lfm3["LFM3_ATTN_BUCKETS"], rs_buckets["Q4PREFILL_LFM3_FA_FIN_BUCKETS"])
        self.assertEqual(sh_lfm3["LFM3_ATTN_BUCKETS"], rs_buckets["Q4LAYER_LFM3_FA_FIN_BUCKETS"])
        self.assertEqual(sh_lfm3["LFM3_DEC_ATTN_BUCKETS"], rs_buckets["Q4LAYER_LFM3_KV_BUCKETS"])
        self.assertEqual(sh_lfm3["LFM3_BIDIR_B_BUCKETS"], rs_buckets["Q4PREFILL_LFM3_BIDIR_B_BUCKETS"])

        self.assertEqual(sh_lfm2["LFM2_PRE_BUCKETS"], rs_buckets["Q4PREFILL_LFM2_ATTN_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_PRE_BUCKETS"], rs_buckets["Q4PREFILL_LFM2_ATTN_NO_FFN_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_EXT_BUCKETS"], rs_buckets["Q4PREFILL_LFM2_FA_FIN_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_EXT_BUCKETS"], rs_buckets["Q4PREFILL_LFM2_FA_FIN_NO_FFN_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_DEC_BUCKETS"], rs_buckets["Q4LAYER_LFM2_KV_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_DEC_BUCKETS"], rs_buckets["Q4LAYER_LFM2_ATTN_NO_FFN_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_DEC_BUCKETS"], rs_buckets["Q4LAYER_LFM2_FA_FIN_BUCKETS"])
        self.assertEqual(sh_lfm2["LFM2_DEC_BUCKETS"], rs_buckets["Q4LAYER_LFM2_FA_FIN_NO_FFN_BUCKETS"])

        invocations: list[tuple[str, str, int, int, int | None]] = [
            # build_kernels.sh: q4prefill_lfm3
            ("q4prefill_lfm3", "q4prefill_lfm3", 0, 0, None),
            ("q4prefill_lfm3", "q4prefill_lfm3", 2, 0, None),
            ("q4prefill_lfm3", "q4prefill_lfm3", 6, 256, 1),
            ("q4prefill_lfm3", "q4prefill_lfm3", 4, 256, 2),
            *[("q4prefill_lfm3", "q4prefill_lfm3", 1, c, None) for c in sh_lfm3["LFM3_ATTN_BUCKETS"]],
            *[("q4prefill_lfm3", "q4prefill_lfm3", 5, c, 3) for c in sh_lfm3["LFM3_ATTN_BUCKETS"]],
            *[("q4prefill_lfm3", "q4prefill_lfm3", 3, c, None) for c in sh_lfm3["LFM3_BIDIR_B_BUCKETS"]],
            # build_kernels.sh: q4layer_lfm3
            ("q4layer_lfm3", "q4layer_lfm3", 0, 0, None),
            ("q4layer_lfm3", "q4layer_lfm3", 2, 0, None),
            ("q4layer_lfm3", "q4layer_lfm3", 3, 256, None),
            *[("q4layer_lfm3", "q4layer_lfm3", 1, c, None) for c in sh_lfm3["LFM3_DEC_ATTN_BUCKETS"]],
            *[("q4layer_lfm3", "q4layer_lfm3", 4, c, None) for c in sh_lfm3["LFM3_ATTN_BUCKETS"]],
            # build_lfm2_kernels.sh: q4prefill_lfm2
            ("q4prefill_lfm2", "q4prefill_lfm2", 0, 0, None),
            ("q4prefill_lfm2", "q4prefill_lfm2", 5, 0, None),
            *[("q4prefill_lfm2", "q4prefill_lfm2", 1, c, None) for c in sh_lfm2["LFM2_PRE_BUCKETS"]],
            *[("q4prefill_lfm2", "q4prefill_lfm2", 6, c, None) for c in sh_lfm2["LFM2_PRE_BUCKETS"]],
            # build_lfm2_kernels.sh: q4prefill_lfm2_ext
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", 2, 0, 0),
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", 0, 0, 0),
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", 7, 0, 0),
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", 3, 1024, 1),
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", 4, 1024, 2),
            *[("q4prefill_lfm2_ext", "q4prefill_lfm2", 5, c, 3) for c in sh_lfm2["LFM2_EXT_BUCKETS"]],
            *[("q4prefill_lfm2_ext", "q4prefill_lfm2", 6, c, 3) for c in sh_lfm2["LFM2_EXT_BUCKETS"]],
            # build_lfm2_kernels.sh: q4layer_lfm2
            ("q4layer_lfm2", "q4layer_lfm2", 0, 0, None),
            ("q4layer_lfm2", "q4layer_lfm2", 5, 0, None),
            ("q4layer_lfm2", "q4layer_lfm2", 2, 0, None),
            ("q4layer_lfm2", "q4layer_lfm2", 3, 512, None),
            *[("q4layer_lfm2", "q4layer_lfm2", 1, c, None) for c in sh_lfm2["LFM2_DEC_BUCKETS"]],
            *[("q4layer_lfm2", "q4layer_lfm2", 6, c, None) for c in sh_lfm2["LFM2_DEC_BUCKETS"]],
            *[("q4layer_lfm2", "q4layer_lfm2", 4, c, None) for c in sh_lfm2["LFM2_DEC_BUCKETS"]],
            *[("q4layer_lfm2", "q4layer_lfm2", 7, c, None) for c in sh_lfm2["LFM2_DEC_BUCKETS"]],
        ]

        mods, dispatched = _make_mock_modules()
        fifo_cls = mods["_fake_fifo_handle"]
        with mock.patch.dict(sys.modules, mods):
            loaded = {
                subdir: _load_iron_module(subdir, "sched")
                for subdir in ("q4layer_lfm2", "q4layer_lfm3", "q4prefill_lfm2", "q4prefill_lfm2_ext", "q4prefill_lfm3")
            }
            covered_modes: dict[str, set[int]] = {s: set() for s in loaded}
            checked_repr_variants: set[tuple[str, int, int, int | None]] = set()
            for subdir, fn_name, mode, kv_chunks, param_slot in invocations:
                mod = loaded[subdir]
                argv = [f"{subdir}.py", f"--layer-mode={mode}", f"--kv-chunks={kv_chunks}"]
                if param_slot is not None:
                    argv.append(f"--param-slot={param_slot}")
                with mock.patch.object(sys, "argv", argv):
                    mod.main()  # type: ignore[attr-defined]
                self.assertEqual(dispatched[-1]["fn"], fn_name)
                covered_modes[subdir].add(mode)

                # Check representative schedule + DMA FIFO balance per (mode, kv_chunks=0/min/max)
                rep_key = (subdir, mode, 1 if kv_chunks > 0 else 0, param_slot)
                if rep_key not in checked_repr_variants or kv_chunks in (256, 511, 512, 1024):
                    checked_repr_variants.add(rep_key)
                    _check_seq_vs_sched_balance(self, subdir, fn_name, mod, mode, kv_chunks, param_slot, fifo_cls)  # type: ignore[arg-type]

            for subdir, mod in loaded.items():
                self.assertEqual(covered_modes[subdir], set(mod.VALID_MODES))  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for XDNA IRON compile-time parameter validation and ABI layout."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
import re
import sys
import types
import unittest
from unittest import mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IRON_ROOT = os.path.join(REPO_ROOT, "cera", "src", "backend", "xdna", "kernels", "iron")
KERNELS_RS = os.path.join(REPO_ROOT, "cera", "src", "backend", "xdna", "kernels.rs")

sys.path.insert(0, IRON_ROOT)
from compile_params import validate_compile_params


def _eval_top_level_ints(py_path: str) -> dict[str, int]:
    """Evaluate simple integer constant assignments at module scope via AST."""
    with open(py_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=py_path)
    env: dict[str, int] = {}

    def eval_node(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise ValueError(f"unsupported expression {ast.dump(node)}")

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                env[stmt.targets[0].id] = eval_node(stmt.value)
            except ValueError:
                pass
    return env


def _parse_rust_usize_consts(rs_path: str) -> dict[str, int]:
    """Parse `pub const NAME: usize = <expr>;` entries with simple arithmetic via AST."""
    with open(rs_path, encoding="utf-8") as fh:
        src = fh.read()
    consts: dict[str, int] = {}

    def eval_expr(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in consts:
            return consts[node.id]
        if isinstance(node, ast.BinOp):
            left = eval_expr(node.left)
            right = eval_expr(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        raise ValueError("unresolvable rust const")

    for match in re.finditer(r"pub const ([A-Z0-9_]+): usize = ([^;]+);", src):
        name, raw_expr = match.group(1), match.group(2).strip()
        try:
            tree = ast.parse(raw_expr.replace("/", "//"), mode="eval")
            consts[name] = eval_expr(tree.body)
        except (SyntaxError, ValueError):
            pass
    return consts


def _inspect_generator_wiring(
    py_path: str,
    jit_fn_name: str,
    expected_modes: tuple[int, ...],
    expected_nonzero: tuple[int, ...],
    expected_param_slots: tuple[int, ...] | None = None,
) -> tuple[bool, bool, bool]:
    """Check `_validate_compile_params` delegation and call sites in `@iron.jit` and `main`."""
    with open(py_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=py_path)
    env: dict[str, object] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if stmt.targets[0].id in {"VALID_MODES", "VALID_PARAM_SLOTS", "MAX_KV_CHUNKS"}:
                try:
                    env[stmt.targets[0].id] = ast.literal_eval(stmt.value)
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
                    kw = {k.arg: eval_kw(k.value) for k in sub.keywords if k.arg in {"valid_modes", "nonzero_kv_modes", "valid_param_slots", "max_kv_chunks"}}
                    param_slot_fwd = any(k.arg == "param_slot" for k in sub.keywords)
                    slots_ok = expected_param_slots is None or (
                        tuple(kw.get("valid_param_slots", ())) == expected_param_slots  # type: ignore[arg-type]
                        and param_slot_fwd
                    )
                    max_ok = kw.get("max_kv_chunks") == env.get("MAX_KV_CHUNKS")
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
    """Build lightweight `aie.*` (and `numpy` fallback if missing) stubs for offline script dispatch tests."""
    dispatched: list[dict[str, object]] = []

    class _FakeSubscriptable:
        def __getitem__(self, _item: object) -> "_FakeSubscriptable":
            return self

    class _FakeAieIron(types.ModuleType):
        @staticmethod
        def jit(**_kw: object):
            def deco(fn):
                return fn
            return deco

    mods: dict[str, types.ModuleType] = {}
    aie_pkg = types.ModuleType("aie")
    mods["aie"] = aie_pkg

    iron_mod = _FakeAieIron("aie.iron")
    mods["aie.iron"] = iron_mod

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

    for attr in ("Buffer", "CompileTime", "ExternalFunction", "In", "Out"):
        setattr(mods["aie.iron"], attr, _FakeSubscriptable())
    setattr(mods["aie.dialects"], "arith", object())
    setattr(mods["aie.helpers.dialects.scf"], "else_", object())
    setattr(mods["aie.helpers.dialects.scf"], "if_", object())
    setattr(mods["aie.helpers.taplib.tap"], "TensorAccessPattern", object())
    setattr(mods["aie.iron.controlflow"], "range_", range)
    setattr(mods["aie.iron.dataflow"], "ObjectFifo", object())
    setattr(mods["aie.iron.device"], "NPU2", lambda: "NPU2")
    setattr(mods["aie.iron.device"], "Tile", object())
    setattr(mods["aie.iron.program"], "Program", object())
    setattr(mods["aie.iron.runtime"], "Runtime", object())
    setattr(mods["aie.iron.runtime"], "TaskGroup", object())
    setattr(mods["aie.iron.worker"], "Worker", object())
    setattr(mods["aie.ir"], "IndexType", object())
    setattr(mods["aie.ir"], "IntegerType", object())
    setattr(mods["aie.utils"], "get_current_device", lambda: "NPU2")
    setattr(mods["aie.utils.config"], "cxx_header_path", lambda: "/tmp/cxx")
    setattr(mods["aie.utils.hostruntime.argparse"], "add_compile_args", lambda _p: None)

    def fake_run_design_cli(fn, opts, *, device, compile_kwargs):
        dispatched.append({"fn": fn.__name__, "device": device, "kwargs": compile_kwargs(opts)})

    setattr(mods["aie.utils.hostruntime.cli"], "run_design_cli", fake_run_design_cli)

    if importlib.util.find_spec("numpy") is None:
        np_mod = types.ModuleType("numpy")
        setattr(np_mod, "ndarray", _FakeSubscriptable())
        setattr(np_mod, "dtype", _FakeSubscriptable())
        setattr(np_mod, "float32", object())
        setattr(np_mod, "int32", object())
        mods["numpy"] = np_mod

    return mods, dispatched


class TestXdnaCompileParams(unittest.TestCase):
    def test_decode_scripts_valid_and_invalid_compile_params(self) -> None:
        valid_modes = (0, 1, 2, 3, 4)
        for mode in valid_modes:
            for chunks in (1, 128, 256):
                validate_compile_params(mode, chunks, valid_modes=valid_modes, nonzero_kv_modes=(3,))
            if mode != 3:
                validate_compile_params(mode, 0, valid_modes=valid_modes, nonzero_kv_modes=(3,))

        with self.assertRaisesRegex(ValueError, "layer_mode=3 requires kv_chunks >= 1"):
            validate_compile_params(3, 0, valid_modes=valid_modes, nonzero_kv_modes=(3,))

        with self.assertRaisesRegex(ValueError, r"out of range 0\.\.=4"):
            validate_compile_params(1, 5, valid_modes=valid_modes, nonzero_kv_modes=(3,), max_kv_chunks=4)

        for bad_mode in (-1, 5, 99, True, False, "1"):
            with self.assertRaisesRegex(ValueError, "unsupported layer_mode"):
                validate_compile_params(bad_mode, 1, valid_modes=valid_modes, nonzero_kv_modes=(3,))  # type: ignore[arg-type]

        for bad_chunks in (-1, 257, True, False, "4"):
            with self.assertRaisesRegex(ValueError, "kv_chunks"):
                validate_compile_params(1, bad_chunks, valid_modes=valid_modes, nonzero_kv_modes=(3,))  # type: ignore[arg-type]

    def test_prefill_scripts_valid_and_invalid_compile_params(self) -> None:
        valid_lfm2 = (0, 1, 2, 4, 5, 6)
        for mode in valid_lfm2:
            for chunks in (1, 1024):
                validate_compile_params(mode, chunks, valid_modes=valid_lfm2, nonzero_kv_modes=(4,), max_kv_chunks=1024)
            if mode != 4:
                validate_compile_params(mode, 0, valid_modes=valid_lfm2, nonzero_kv_modes=(4,), max_kv_chunks=1024)

        with self.assertRaisesRegex(ValueError, "unsupported layer_mode"):
            validate_compile_params(3, 1, valid_modes=valid_lfm2, nonzero_kv_modes=(4,), max_kv_chunks=1024)
        with self.assertRaisesRegex(ValueError, "layer_mode=4 requires kv_chunks >= 1"):
            validate_compile_params(4, 0, valid_modes=valid_lfm2, nonzero_kv_modes=(4,), max_kv_chunks=1024)
        with self.assertRaisesRegex(ValueError, "kv_chunks"):
            validate_compile_params(5, 1025, valid_modes=valid_lfm2, nonzero_kv_modes=(4,), max_kv_chunks=1024)

        valid_lfm2_ext = (0, 1, 2, 3, 4, 5, 6, 7)
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

        self.assertEqual(_inspect_generator_wiring(lfm2_dec, "q4layer_lfm2", (0, 1, 2, 3, 4, 5, 6, 7), (3,)), (True, True, True))
        self.assertEqual(_inspect_generator_wiring(lfm3_dec, "q4layer_lfm3", (0, 1, 2, 3, 4), (3,)), (True, True, True))
        self.assertEqual(_inspect_generator_wiring(lfm2_pre, "q4prefill_lfm2", (0, 1, 2, 4, 5, 6), (4,)), (True, True, True))
        self.assertEqual(_inspect_generator_wiring(lfm2_pre_ext, "q4prefill_lfm2", (0, 1, 2, 3, 4, 5, 6, 7), (3, 4), (0, 1, 2, 3)), (True, True, True))
        self.assertEqual(_inspect_generator_wiring(lfm3_pre, "q4prefill_lfm3", (0, 1, 2, 3, 4, 5, 6), (3, 4, 6), (0, 1, 2, 3)), (True, True, True))

        c_lfm2_dec = _eval_top_level_ints(lfm2_dec)
        self.assertEqual(c_lfm2_dec["SLOT_NORM_0"], rs["LFM2_DEC_SLOT_NORM_0"])
        self.assertEqual(c_lfm2_dec["SLOT_NORM_1"], rs["LFM2_DEC_SLOT_NORM_1"])
        self.assertEqual(c_lfm2_dec["SLOT_ROPE_PKT"], rs["LFM2_DEC_SLOT_ROPE_PKT"])
        self.assertEqual(c_lfm2_dec["MAX_KV_CHUNKS"], rs["LFM2_MAX_KV_CHUNKS"])
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
            ("q4prefill_lfm2", "q4prefill_lfm2", "4", "5"),
            ("q4prefill_lfm2_ext", "q4prefill_lfm2", "3", "5"),
            ("q4prefill_lfm3", "q4prefill_lfm3", "4", "5"),
        ]
        mods, dispatched = _make_mock_modules()
        with mock.patch.dict(sys.modules, mods):
            for idx, (subdir, fn_name, nz_mode, ok_mode) in enumerate(scripts):
                path = os.path.join(IRON_ROOT, subdir, f"{subdir}.py")
                mod_name = f"_mocked_{subdir}_{idx}"
                spec = importlib.util.spec_from_file_location(mod_name, path)
                assert spec is not None and spec.loader is not None
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                has_param_slot = subdir in {"q4prefill_lfm2_ext", "q4prefill_lfm3"}
                extra_args = ["--param-slot", "2"] if has_param_slot else []
                with mock.patch.object(sys, "argv", [path, "--layer-mode", ok_mode, "--kv-chunks", "0", *extra_args]):
                    mod.main()
                self.assertEqual(dispatched[-1]["fn"], fn_name)
                expected_kw = {"layer_mode": int(ok_mode), "kv_chunks": 0}
                if has_param_slot:
                    expected_kw["param_slot"] = 2
                self.assertEqual(dispatched[-1]["kwargs"], expected_kw)

                # Inner `@iron.jit` guard also fires when the lowered function is called directly.
                n_pos = 6 if subdir == "q4layer_lfm3" else 5
                with self.assertRaises(ValueError):
                    getattr(mod, fn_name)(*([None] * n_pos), layer_mode=int(nz_mode), kv_chunks=0)
                if has_param_slot:
                    with self.assertRaises(ValueError):
                        getattr(mod, fn_name)(*([None] * n_pos), layer_mode=int(ok_mode), kv_chunks=0, param_slot=4)

                for bad_chunks in ("-1", str(mod.MAX_KV_CHUNKS + 1)):
                    with self.assertRaises(ValueError):
                        with mock.patch.object(sys, "argv", [path, "--layer-mode", ok_mode, "--kv-chunks", bad_chunks]):
                            mod.main()

                with self.assertRaises(ValueError):
                    with mock.patch.object(sys, "argv", [path, "--layer-mode", nz_mode, "--kv-chunks", "0"]):
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


if __name__ == "__main__":
    unittest.main()

"""Guard the evidence harness against false positives and source drift."""

import fcntl
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run as runner
from commands import Commands, cargo_artifact, exhaustive_rejection
from consumers import CASES, WEB_CASES, result
from prepare import (
    REPO,
    ROOT,
    cargo_configs,
    inventory,
    once,
    prepare,
    verify_inventory,
)
from run import probe_environment


class HarnessTests(unittest.TestCase):
    def test_runs_with_the_same_target_parent_get_distinct_build_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            probe = root / "probe"
            probe.mkdir()
            targets = []
            with patch.object(runner, "ROOT", probe):
                for _ in range(2):
                    with (
                        patch.object(
                            sys,
                            "argv",
                            [
                                "run.py",
                                "--prepare-only",
                                "--target",
                                str(root / "target"),
                            ],
                        ),
                        redirect_stdout(io.StringIO()) as stdout,
                    ):
                        runner.main()
                    output = Path(stdout.getvalue().strip())
                    report = json.loads((output / "results.json").read_text())
                    self.assertEqual(report["status"], "prepared-only")
                    target = Path(report["target"])
                    self.assertEqual(target.parent, root / "target")
                    self.assertTrue(target.is_dir())
                    targets.append(target)
            self.assertNotEqual(*targets)

    def test_probe_environment_removes_runtime_library_overrides(self):
        inherited = {
            "PATH": "/probe/bin",
            "CARGO_BUILD_TARGET": "wasm32-unknown-unknown",
            "RUSTFLAGS": "-C target-feature=+simd128",
            "DYLD_LIBRARY_PATH": "/stale/target/debug",
            "DYLD_FALLBACK_LIBRARY_PATH": "/stale/fallback",
            "DYLD_INSERT_LIBRARIES": "/stale/lib.dylib",
            "LD_LIBRARY_PATH": "/stale/lib",
            "LD_PRELOAD": "/stale/lib.so",
            "CERA_GPU_DF": "1",
            "JAVA_TOOL_OPTIONS": "-Dexample=override",
            "_JAVA_OPTIONS": "-Duniffi.component.loading_native.libraryOverride=stale",
            "JDK_JAVA_OPTIONS": "-Dexample=override",
            "JAVA_OPTS": "-Dexample=override",
            "KOTLIN_OPTS": "-Dexample=override",
            "NODE_OPTIONS": "--require /stale/preload.cjs",
            "NODE_PATH": "/stale/modules",
        }
        self.assertEqual(
            probe_environment(inherited),
            {
                key: inherited[key]
                for key in ("PATH", "CARGO_BUILD_TARGET", "RUSTFLAGS")
            },
        )
        self.assertIn("DYLD_LIBRARY_PATH", inherited)

    def test_artifacts_require_the_built_source_and_unambiguous_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src/lib.rs"
            artifact = root / "target/aarch64-apple-darwin/debug/probe.dylib"
            artifact.parent.mkdir(parents=True)
            artifact.touch()
            record = {
                "reason": "compiler-artifact",
                "target": {"src_path": str(source)},
                "filenames": [str(artifact), str(artifact.with_suffix(".rlib"))],
                "executable": str(artifact),
            }
            output = json.dumps(record)
            self.assertEqual(cargo_artifact(output, source, suffix=".dylib"), artifact)
            self.assertEqual(cargo_artifact(output, source, executable=True), artifact)
            for invalid in (
                output + "\n" + output,
                output.replace(str(source), str(root / "other.rs")),
                output.replace("compiler-artifact", "build-finished"),
                output.replace(str(artifact), "target/probe.dylib"),
            ):
                with self.assertRaises(RuntimeError):
                    cargo_artifact(invalid, source, suffix=".dylib")
            artifact.unlink()
            with self.assertRaises(RuntimeError):
                cargo_artifact(output, source, suffix=".dylib")

    def test_cargo_config_snapshot_detects_new_modified_and_removed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "mirror/workspace"
            home = root / "cargo-home"
            expected = cargo_configs(workspace, home)
            config = root / ".cargo/config.toml"
            config.parent.mkdir()
            config.write_text('[build]\ntarget = "aarch64-apple-darwin"\n')
            added = cargo_configs(workspace, home)
            self.assertIsNone(expected[str(config)])
            self.assertNotEqual(expected, added)
            config.write_text('[build]\ntarget = "wasm32-unknown-unknown"\n')
            self.assertNotEqual(added, cargo_configs(workspace, home))
            config.unlink()
            self.assertEqual(expected, cargo_configs(workspace, home))
            self.assertIn(str(home / "config"), expected)
            self.assertIn(str(workspace / ".cargo/config.toml"), expected)

    def test_negative_control_requires_specific_diagnostic(self):
        message = {
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "code": {"code": "E0004"},
                "message": "non-exhaustive patterns",
                "spans": [{"is_primary": True, "file_name": "src/bin/exhaustive.rs"}],
            },
        }
        valid = json.dumps(message)
        self.assertTrue(exhaustive_rejection(101, valid))
        for code, text in [
            (1, valid),
            (0, valid),
            (101, "linker failed"),
            (101, valid + "\n" + valid),
            (101, valid.replace("E0004", "E0308")),
            (101, valid.replace("exhaustive.rs", "other.rs")),
            (101, valid.replace("true", "false")),
        ]:
            self.assertFalse(exhaustive_rejection(code, text))

    def test_result_rejects_missing_duplicate_and_invalid_evidence(self):
        record = {
            "cases": sorted(CASES | WEB_CASES),
            "generation": {"tokens": [0, 1, 0], "position": 5},
        }
        self.assertEqual(result(json.dumps(record), False), record)
        for mutate in (
            lambda r: r["cases"].pop(),
            lambda r: r["cases"].remove("kind-kws"),
            lambda r: r["cases"].append(r["cases"][0]),
            lambda r: r.update(cases=dict.fromkeys(r["cases"], False)),
            lambda r: r.update(cases=None),
            lambda r: r["cases"].append(1),
            lambda r: r["generation"].update(tokens=[]),
            lambda r: r["generation"].update(tokens=[True, 1, 0]),
            lambda r: r["generation"].update(position=4),
        ):
            bad = json.loads(json.dumps(record))
            mutate(bad)
            with self.assertRaises(RuntimeError):
                result(json.dumps(bad), False)
        with self.assertRaises(RuntimeError):
            result(json.dumps(record), True)
        with self.assertRaises(RuntimeError):
            result(json.dumps(record) + "\n" + json.dumps(record), False)

    def test_source_mirror_preserves_algorithms_and_production_binding_types(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            workspace, report = prepare(Path(temporary))
            changed = []
            for name in {
                **report["source_sha256"],
                **report["ffi_source_sha256"],
                **report["wasm_source_sha256"],
            }:
                before, after = (
                    (REPO / name).read_bytes(),
                    (workspace / name).read_bytes(),
                )
                if before != after:
                    changed.append(name)
                    # Reverse only the allowed test-accessor
                    # adaptations, then require byte-identical production code.
                    text = after.decode()
                    if name == "cera/src/session.rs":
                        text = once(
                            text,
                            "\nimpl Session {\n"
                            "    pub fn config_for_loading_probe(&self) -> SessionConfig { self.config.clone() }\n"
                            "}\n",
                            "",
                        )
                    elif name == "cera-ffi/src/lib.rs":
                        text = once(
                            text,
                            "\n" + (ROOT / "native/session_observation.rs").read_text(),
                            "",
                        )
                    elif name == "cera-wasm/src/lib.rs":
                        text = once(
                            text,
                            "\n" + (ROOT / "wasm/session_observation.rs").read_text(),
                            "",
                        )
                    elif name == "cera-wasm/src/loading.rs":
                        text = once(
                            text,
                            "\n" + (ROOT / "wasm/loading_observation.rs").read_text(),
                            "",
                        )
                    else:
                        self.fail(f"Unexpected production source adaptation: {name}")
                    self.assertEqual(text.encode(), before, name)
            self.assertEqual(set(changed), set(report["exposed_sha256"]))
            self.assertEqual(len(changed), 4)
            with self.assertRaises(ValueError):
                once("anchor anchor", "anchor", "new")

    def test_inventory_rejects_changed_added_and_removed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_text("original")
            expected = inventory(root)
            verify_inventory(root, expected)
            for action in (
                lambda: source.write_text("changed"),
                lambda: source.unlink(),
                lambda: (root / "added").touch(),
            ):
                source.write_text("original")
                action()
                with self.assertRaises(RuntimeError):
                    verify_inventory(root, expected)

    def test_commands_record_failure_launch_error_and_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = Commands(root, dict(os.environ), root)
            self.assertEqual(
                commands.run("ok", [sys.executable, "-c", "print('okay')"]), "okay"
            )
            with self.assertRaises(RuntimeError):
                commands.run("failure", [sys.executable, "-c", "raise SystemExit(9)"])
            self.assertEqual(commands.results[-1]["returncode"], 9)
            with self.assertRaises(FileNotFoundError):
                commands.run("missing", [root / "missing-command"])
            self.assertEqual(commands.results[-1]["status"], "launch-failed")
            with self.assertRaises(subprocess.TimeoutExpired):
                commands.run(
                    "timeout",
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    timeout=0.1,
                )
            self.assertEqual(commands.results[-1]["status"], "interrupted-or-timeout")

    def test_timeout_kills_descendant_when_leader_exits_on_sigterm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready"
            child = (
                "import fcntl, os, signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "lock = open(sys.argv[1], 'w')\n"
                "fcntl.flock(lock, fcntl.LOCK_EX)\n"
                "lock.write(str(os.getpgrp())); lock.flush()\n"
                "time.sleep(30)\n"
            )
            leader = (
                "import subprocess, sys, time\n"
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
                "time.sleep(30)\n"
            )
            commands = Commands(root, dict(os.environ), root)
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    commands.run(
                        "descendant-timeout",
                        [sys.executable, "-c", leader, child, ready],
                        timeout=2,
                    )
                self.assertTrue(ready.read_text(), "Descendant must acquire the lock")
                # A live descendant retains this lock even after its parent exits.
                with ready.open() as lock:
                    deadline = time.monotonic() + 1
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.01)
            finally:
                if ready.exists() and ready.read_text():
                    try:
                        os.killpg(int(ready.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    (ROOT / "build").mkdir(exist_ok=True)
    unittest.main()
